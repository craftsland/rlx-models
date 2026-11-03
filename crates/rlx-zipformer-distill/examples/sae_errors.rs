// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Where and why the student's tokens still differ from the teacher's.
//!
//! 1. Teacher and student stream whole utterances through `rlx-kroko-core`
//!    (int8, as on the MCU); every output frame keeps both encoder outputs.
//! 2. Each frame is labelled through the teacher's predictor and joiner, under
//!    the teacher's greedy context: agree, deletion (teacher emits, student
//!    blank), insertion, or substitution.
//! 3. A sparse autoencoder (ReLU, L1 weighted by decoder norms) is trained
//!    with RLX on the teacher's encoder outputs from the distillation records,
//!    so its features are directions of the space the student must reproduce.
//! 4. Both models' frames go through the SAE's encoder. For every feature: how
//!    often the student errs where the teacher uses it, and how much of the
//!    teacher's activation the student reproduces (recall); features with the
//!    most errors beyond the base rate are listed with the word pieces they
//!    fire on. Errors are also broken down by position in the utterance (the
//!    student trains on 8-chunk windows from an empty state) and by length.
//!
//! ```text
//! sae_errors --student DIR --teacher .cache/zipformer-m/native \
//!     --data DISTILL_DIR --list LIST.tsv [--limit 100] [--width 4096] \
//!     [--l1 0.004] [--steps 2000] [--device metal]
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rlx_kroko::mcu::{KrokoExportOptions, export_rkr};
use rlx_kroko_core as core;
use rlx_tensor::parallel::Feeds;
use rlx_tensor::{DataParallel, Func, LrSchedule, Replica, apply_gradients, parse_device, shape};
use rlx_zipformer_distill::data::Records;
use rlx_zipformer_distill::init::Teacher;
use rlx_zipformer_distill::joiner::{DIM, Joiner};

/// Frames per SAE batch.
const BATCH: usize = 4096;

/// A frame's outcome under the teacher's context.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Agree,
    Deletion,
    Insertion,
    Substitution,
}

struct Utt {
    seconds: f64,
    words: usize,
    student_errs: usize,
    teacher_errs: usize,
}

/// One output frame of the dev set.
struct Frame {
    utt: usize,
    pos: usize,
    kind: Kind,
    /// The piece the teacher emits here, or the one it is inside of.
    piece: usize,
}

fn norm(s: &str) -> Vec<String> {
    s.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '\'' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn edits(r: &[String], h: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, x) in r.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, y) in h.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(x != y))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[h.len()]
}

/// An int8 `.rkr` of a converted directory, as `u32` words (alignment).
fn rkr(dir: &Path) -> Result<(Vec<u32>, usize)> {
    let blob = export_rkr(dir, KrokoExportOptions::default())?;
    let words = blob
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect();
    Ok((words, blob.len()))
}

/// Stream `pcm` through `model`: the valid encoder frames `[T, 512]` and the text.
fn stream(model: &core::Model<'_>, bytes: &[u8], pcm: &[f32]) -> Result<(Vec<f32>, String)> {
    let cfg = *model.config();
    let need = core::scratch_len(model, pcm.len());
    let (mut fs, mut is, mut page) = (
        vec![0f32; need.f32s],
        vec![0i8; need.i8s],
        vec![0u32; need.page_words],
    );
    let mut ids = vec![0u16; 4096];
    let frames = core::fbank_frames(pcm.len());
    let mut enc = Vec::new();
    let mut chunk = 0usize;
    let n = core::transcribe_with(
        model,
        &mut core::SlicePager(bytes),
        pcm,
        &mut fs,
        &mut is,
        &mut page,
        &mut ids,
        &mut |e, to| {
            let consumed = cfg
                .chunk_shift
                .min(frames.saturating_sub(chunk * cfg.chunk_shift));
            let valid = to.min(consumed.div_ceil(4));
            enc.extend_from_slice(&e[..valid * cfg.enc_out]);
            chunk += 1;
        },
    )
    .map_err(|e| anyhow!("{e}"))?;
    let mut buf = vec![0u8; n * 32 + 1];
    let len = model.decode(&ids[..n], &mut buf);
    Ok((enc, String::from_utf8_lossy(&buf[..len]).into_owned()))
}

fn sae_graph(width: usize, l1: f32, with_loss: bool) -> Func {
    Func::new("sae", move |s| {
        let h = s.input("h", shape![BATCH, DIM]);
        let we = s.param("w_enc", shape![DIM, width]);
        let be = s.param("b_enc", shape![width]);
        let bd = s.param("b_dec", shape![DIM]);
        let z = (&(&h - &bd).matmul(&we) + &be).relu();
        if !with_loss {
            // Features only: the decoder is not part of this graph.
            return z;
        }
        let wd = s.param("w_dec", shape![width, DIM]);
        let r = &z.matmul(&wd) + &bd;
        let d = &r - &h;
        let mse = &(&d * &d).mean_all() / &(&h * &h).mean_all();
        let norms = (&wd * &wd).sum(vec![1], false).sqrt();
        let sparsity = (&z * &norms).sum(vec![1], false).mean_all();
        &mse + &(&sparsity * l1)
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let num = |k: &str, d: f64| -> Result<f64> {
        Ok(get(k).map(|v| v.parse()).transpose()?.unwrap_or(d))
    };
    let student_dir = Path::new(get("--student").context("--student")?);
    let teacher_dir = Path::new(get("--teacher").context("--teacher")?);
    let list = Path::new(get("--list").context("--list")?);
    let data = Records::open(Path::new(get("--data").context("--data")?))?;
    let limit = num("--limit", 100.0)? as usize;
    let width = num("--width", 4096.0)? as usize;
    let l1 = num("--l1", 0.004)? as f32;
    let steps = num("--steps", 2000.0)? as usize;
    let dev =
        parse_device(get("--device").map_or("cpu", String::as_str)).map_err(|e| anyhow!("{e}"))?;

    let teacher = Teacher::open(teacher_dir)?;
    let joiner = Joiner::open(&teacher, teacher_dir)?;
    let pieces: Vec<String> = std::fs::read_to_string(teacher_dir.join("tokens.txt"))?
        .lines()
        .map(|l| {
            l.rsplit_once(' ')
                .map_or(l, |p| p.0)
                .replace('\u{2581}', "_")
        })
        .collect();

    // 1–2. Stream the dev clips through both models; label every frame.
    let (tw, tlen) = rkr(teacher_dir)?;
    let (sw, slen) = rkr(student_dir)?;
    // SAFETY: u32 storage viewed as bytes.
    let tb = unsafe { std::slice::from_raw_parts(tw.as_ptr().cast::<u8>(), tlen) };
    let sb = unsafe { std::slice::from_raw_parts(sw.as_ptr().cast::<u8>(), slen) };
    let tm = core::Model::parse(tb).map_err(|e| anyhow!("{e}"))?;
    let sm = core::Model::parse(sb).map_err(|e| anyhow!("{e}"))?;
    let base = list.parent().unwrap_or(Path::new("."));
    let (mut utts, mut frames) = (Vec::new(), Vec::new());
    let (mut t_enc, mut s_enc) = (Vec::new(), Vec::new());
    for line in std::fs::read_to_string(list)?.lines().take(limit) {
        let Some((file, text)) = line.split_once('\t') else {
            continue;
        };
        let (pcm, rate) = rlx_core::asr_bench::read_wav_mono(&base.join(file))?;
        let pcm = if rate == 16_000 {
            pcm
        } else {
            rlx_core::audio::resample_linear(&pcm, rate, 16_000)
        };
        let (te, tt) = stream(&tm, tb, &pcm)?;
        let (se, st) = stream(&sm, sb, &pcm)?;
        let n = (te.len() / DIM).min(se.len() / DIM);
        let ctx = joiner.contexts(&te[..n * DIM]);
        let u = utts.len();
        for (t, c) in ctx.iter().enumerate() {
            let row = t * DIM..(t + 1) * DIM;
            let (a, b) = (joiner.best(&te[row.clone()], *c), joiner.best(&se[row], *c));
            let kind = match (joiner.is_silent(a), joiner.is_silent(b)) {
                _ if a == b => Kind::Agree,
                (false, true) => Kind::Deletion,
                (true, false) => Kind::Insertion,
                _ => Kind::Substitution,
            };
            let piece = if joiner.is_silent(a) {
                c[1] as usize
            } else {
                a
            };
            frames.push(Frame {
                utt: u,
                pos: t,
                kind,
                piece,
            });
        }
        t_enc.extend_from_slice(&te[..n * DIM]);
        s_enc.extend_from_slice(&se[..n * DIM]);
        let r = norm(text);
        utts.push(Utt {
            seconds: pcm.len() as f64 / 16_000.0,
            words: r.len(),
            student_errs: edits(&r, &norm(&st)),
            teacher_errs: edits(&r, &norm(&tt)),
        });
    }
    let words: usize = utts.iter().map(|u| u.words).sum();
    let wer = |f: fn(&Utt) -> usize, us: &[&Utt]| {
        let w: usize = us.iter().map(|u| u.words).sum();
        100.0 * us.iter().map(|u| f(u)).sum::<usize>() as f64 / w.max(1) as f64
    };
    let all: Vec<&Utt> = utts.iter().collect();
    println!(
        "{} utterances, {words} words: student WER {:.2}%, teacher {:.2}%; {} frames",
        utts.len(),
        wer(|u| u.student_errs, &all),
        wer(|u| u.teacher_errs, &all),
        frames.len()
    );
    let n_err = frames.iter().filter(|f| f.kind != Kind::Agree).count();
    let count = |k: Kind| frames.iter().filter(|f| f.kind == k).count();
    println!(
        "frames off: {:.2}% — deletions {}, insertions {}, substitutions {}",
        100.0 * n_err as f64 / frames.len() as f64,
        count(Kind::Deletion),
        count(Kind::Insertion),
        count(Kind::Substitution)
    );

    // Position in the utterance (training windows are 8 chunks = 64 frames
    // from an empty state) and utterance length.
    println!("\nby position (output frames, 25 per second):");
    for (lo, hi) in [(0, 64), (64, 128), (128, 256), (256, usize::MAX)] {
        let sel: Vec<usize> = (0..frames.len())
            .filter(|&i| frames[i].pos >= lo && frames[i].pos < hi)
            .collect();
        if sel.is_empty() {
            continue;
        }
        let off = sel
            .iter()
            .filter(|&&i| frames[i].kind != Kind::Agree)
            .count();
        let (mut num, mut den) = (0f64, 0f64);
        for &i in &sel {
            let r = i * DIM..(i + 1) * DIM;
            for (a, b) in t_enc[r.clone()].iter().zip(&s_enc[r]) {
                num += f64::from((a - b) * (a - b));
                den += f64::from(a * a);
            }
        }
        let hi_s = if hi == usize::MAX {
            "end".into()
        } else {
            format!("{:.1} s", hi as f64 / 25.0)
        };
        println!(
            "  {:>5.1} s – {hi_s:>6}: {:>6} frames, {:5.2}% off, encoder error {:.3} (of the teacher's mean square)",
            lo as f64 / 25.0,
            sel.len(),
            100.0 * off as f64 / sel.len() as f64,
            num / den
        );
    }
    println!("by utterance length:");
    for (lo, hi) in [(0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, f64::MAX)] {
        let us: Vec<&Utt> = utts
            .iter()
            .filter(|u| u.seconds >= lo && u.seconds < hi)
            .collect();
        if !us.is_empty() {
            println!(
                "  {lo:>4.0}–{:>4} s: {:>3} utterances, student WER {:5.2}%, teacher {:5.2}%",
                if hi == f64::MAX {
                    "".into()
                } else {
                    format!("{hi:.0}")
                },
                us.len(),
                wer(|u| u.student_errs, &us),
                wer(|u| u.teacher_errs, &us)
            );
        }
    }

    // 3. The SAE, on the teacher's encoder outputs from the records.
    let per = data.k * 8;
    let n_train = (steps * BATCH).min(data.n * per).min(2_000_000);
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut train = Vec::with_capacity(n_train * DIM);
    let (mut f, mut t) = (Vec::new(), Vec::new());
    while train.len() < n_train * DIM {
        f.clear();
        t.clear();
        data.read_into((next() % data.n as u64) as usize, &mut f, &mut t);
        let j = (next() % per as u64) as usize;
        train.extend_from_slice(&t[j * DIM..(j + 1) * DIM]);
    }
    // One scale for everything: unit mean square per element.
    let rms =
        (train.iter().map(|v| f64::from(v * v)).sum::<f64>() / train.len() as f64).sqrt() as f32;
    for v in train.iter_mut().chain(&mut t_enc).chain(&mut s_enc) {
        *v /= rms;
    }
    let mean: Vec<f32> = (0..DIM)
        .map(|d| train.iter().skip(d).step_by(DIM).sum::<f32>() / n_train as f32)
        .collect();
    let mut init = |n: usize, sd: f32| -> Vec<f32> {
        (0..n)
            .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * sd * 3.46)
            .collect()
    };
    let wd0 = init(width * DIM, 1.0);
    // Unit-norm decoder rows; the encoder starts as their transpose.
    let wd0: Vec<f32> = wd0
        .chunks(DIM)
        .flat_map(|r| {
            let n = r.iter().map(|v| v * v).sum::<f32>().sqrt();
            r.iter().map(move |v| v / n).collect::<Vec<_>>()
        })
        .collect();
    let we0: Vec<f32> = (0..DIM * width)
        .map(|i| wd0[(i % width) * DIM + i / width])
        .collect();
    let mut m = sae_graph(width, l1, true).init_params(|name, dims| match name {
        "w_enc" => we0.clone(),
        "w_dec" => wd0.clone(),
        "b_dec" => mean.clone(),
        _ => vec![0.0; dims.iter().product()],
    });
    let dp = DataParallel::new(vec![Replica::new(dev, m.clone(), BATCH)]);
    let mut opt = rlx_optim::AdamW::new(1e-3).with_weight_decay(0.0);
    let sched = LrSchedule::WarmupCosine {
        base: 1e-3,
        min: 1e-4,
        warmup: steps / 20,
        total: steps,
    };
    let batches = n_train / BATCH;
    let t0 = std::time::Instant::now();
    for step in 0..steps {
        let b = step % batches;
        let feeds: Vec<Feeds> = vec![vec![(
            "h".to_string(),
            train[b * BATCH * DIM..(b + 1) * BATCH * DIM].to_vec(),
        )]];
        let (outs, _) = dp.value_and_grad(&m, feeds);
        let (next_m, loss, _) = apply_gradients(&m, &mut opt, &sched, step, 1.0, outs);
        m = next_m;
        if step % 500 == 0 || step + 1 == steps {
            eprintln!(
                "sae step {step}: loss {loss:.4}, {:.1} s",
                t0.elapsed().as_secs_f64()
            );
        }
    }

    // 4. Features of both models' frames.
    let enc_f = sae_graph(width, l1, false)
        .init_params(|name, _| m.param_binding(name).expect("param").to_vec());
    let enc_dp = DataParallel::new(vec![Replica::new(dev, enc_f.clone(), BATCH)]);
    let feats = |x: &[f32]| -> Vec<f32> {
        let n = x.len() / DIM;
        let mut out = Vec::with_capacity(n * width);
        for c in x.chunks(BATCH * DIM) {
            let mut h = c.to_vec();
            h.resize(BATCH * DIM, 0.0);
            let z = enc_dp.run(&enc_f, vec![vec![("h".to_string(), h)]]);
            out.extend_from_slice(&z[0][0][..c.len() / DIM * width]);
        }
        out
    };
    let (zt, zs) = (feats(&t_enc), feats(&s_enc));
    // Reconstruction quality and sparsity on the teacher's dev frames.
    let wdec = m.param_binding("w_dec").expect("w_dec");
    let bdec = m.param_binding("b_dec").expect("b_dec");
    let (mut num, mut den, mut l0) = (0f64, 0f64, 0usize);
    for (i, z) in zt.chunks(width).enumerate().step_by(7) {
        let mut r = bdec.to_vec();
        for (j, &a) in z.iter().enumerate() {
            if a > 0.0 {
                l0 += 1;
                for (rv, w) in r.iter_mut().zip(&wdec[j * DIM..(j + 1) * DIM]) {
                    *rv += a * w;
                }
            }
        }
        for (a, b) in r.iter().zip(&t_enc[i * DIM..(i + 1) * DIM]) {
            num += f64::from((a - b) * (a - b));
            den += f64::from(b * b);
        }
    }
    let sampled = zt.len() / width / 7 + 1;
    let alive = (0..width)
        .filter(|&j| zt.iter().skip(j).step_by(width).any(|&a| a > 0.0))
        .count();
    println!(
        "\nSAE: {width} features, {:.1} active per frame, {alive} used on the dev frames, explains {:.1}% of the teacher's encoder output",
        l0 as f64 / sampled as f64,
        100.0 * (1.0 - num / den)
    );

    let rate = n_err as f64 / frames.len() as f64;
    struct Feat {
        j: usize,
        n: usize,
        errs: usize,
        excess: f64,
        recall: f64,
    }
    let mut stats: Vec<Feat> = (0..width)
        .filter_map(|j| {
            let (mut n, mut errs, mut kept, mut total) = (0, 0, 0f64, 0f64);
            for (i, fr) in frames.iter().enumerate() {
                let a = zt[i * width + j];
                if a > 0.0 {
                    n += 1;
                    errs += usize::from(fr.kind != Kind::Agree);
                    kept += f64::from(a.min(zs[i * width + j].max(0.0)));
                    total += f64::from(a);
                }
            }
            (n >= 20).then(|| Feat {
                j,
                n,
                errs,
                excess: errs as f64 - rate * n as f64,
                recall: kept / total,
            })
        })
        .collect();
    // Does losing a feature go with errors? Error rate by recall quartile.
    let mut by_recall: Vec<&Feat> = stats.iter().collect();
    by_recall.sort_by(|a, b| a.recall.total_cmp(&b.recall));
    println!("features by how much of them the student reproduces (recall):");
    for q in 0..4 {
        let part = &by_recall[q * by_recall.len() / 4..(q + 1) * by_recall.len() / 4];
        let (n, e): (usize, usize) = part.iter().fold((0, 0), |(n, e), f| (n + f.n, e + f.errs));
        println!(
            "  recall {:.2}–{:.2}: {} features, frames off where active {:.2}%",
            part.first().map_or(0.0, |f| f.recall),
            part.last().map_or(0.0, |f| f.recall),
            part.len(),
            100.0 * e as f64 / n.max(1) as f64
        );
    }
    stats.sort_by(|a, b| b.excess.total_cmp(&a.excess));
    println!(
        "\nfeatures with the most errors beyond the {:.2}% base rate:",
        100.0 * rate
    );
    let mut covered = vec![false; frames.len()];
    for f in stats.iter().take(15) {
        let mut tally: HashMap<usize, usize> = HashMap::new();
        let mut kinds = [0usize; 4];
        let mut first = 0usize;
        let mut utts_hit = std::collections::HashSet::new();
        for (i, fr) in frames.iter().enumerate() {
            if zt[i * width + f.j] > 0.0 {
                *tally.entry(fr.piece).or_default() += 1;
                if fr.kind != Kind::Agree {
                    kinds[fr.kind as usize] += 1;
                    covered[i] = true;
                    first += usize::from(fr.pos < 64);
                }
                utts_hit.insert(fr.utt);
            }
        }
        let mut top: Vec<(usize, usize)> = tally.into_iter().collect();
        top.sort_by_key(|t| std::cmp::Reverse(t.1));
        let names: Vec<String> = top
            .iter()
            .take(6)
            .map(|(p, c)| format!("{}×{c}", pieces.get(*p).map_or("?", String::as_str)))
            .collect();
        println!(
            "  #{:<4} active {:>5} frames in {:>3} utts, {:5.1}% off ({:.1}× base), recall {:.2}; del/ins/sub {}/{}/{}, {} in the first 2.56 s; pieces {}",
            f.j,
            f.n,
            utts_hit.len(),
            100.0 * f.errs as f64 / f.n as f64,
            f.errs as f64 / f.n as f64 / rate,
            f.recall,
            kinds[1],
            kinds[2],
            kinds[3],
            first,
            names.join(" ")
        );
    }
    println!(
        "these 15 features are active on {:.1}% of the frames that are off",
        100.0 * covered.iter().filter(|&&c| c).count() as f64 / n_err.max(1) as f64
    );
    std::process::exit(0)
}
