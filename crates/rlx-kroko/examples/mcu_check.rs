// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Export Kroko to the MCU `.rkr` file and run the no_std pipeline on a
//! clip, paging layers through a `Pager` exactly as from NAND.
//!
//! ```text
//! cargo run -p rlx-kroko --release --example mcu_check -- \
//!     --dir .cache/kroko/en64-native --wav clip16k.wav [--out m.rkr] [--bits 8|4|9] [--dump DIR]
//! ```
//! `--list DIR/list.tsv [--limit N]` scores WER instead (lines
//! `<file.wav>\t<transcript>`; lowercased, punctuation stripped).
//! `--dump DIR` writes the fbank features and every chunk's encoder output
//! (raw f32) for `scripts/kroko_onnx_reference.py` to compare against.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rlx_kroko::mcu::{KrokoExportOptions, export_rkr};
use rlx_kroko_core as core;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let mut opts = KrokoExportOptions::default();
    if let Some(b) = get("--bits") {
        opts.layer_bits = b.parse()?;
    }
    let t0 = std::time::Instant::now();
    let blob = export_rkr(Path::new(get("--dir").context("--dir")?), opts)?;
    eprintln!(
        "exported {:.2} MB in {:.1} s",
        blob.len() as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    if let Some(o) = get("--out") {
        std::fs::write(o, &blob)?;
    }
    let words: Vec<u32> = blob
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect();
    // SAFETY: u32 storage viewed as bytes.
    let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), blob.len()) };
    let model = core::Model::parse(bytes).map_err(|e| anyhow!("{e}"))?;
    eprintln!(
        "resident {:.2} MB, paged {:.2} MB per chunk, page buffer {:.2} MB",
        model.resident_len() as f64 / 1e6,
        model.paged_bytes() as f64 / 1e6,
        model.page_len() as f64 / 1e6
    );

    if let Some(list) = get("--list") {
        let limit = get("--limit")
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(usize::MAX);
        return wer(&model, bytes, list, limit);
    }
    let (pcm, rate) =
        rlx_core::asr_bench::read_wav_mono(Path::new(get("--wav").context("--wav")?))?;
    let pcm = if rate == 16_000 {
        pcm
    } else {
        rlx_core::audio::resample_linear(&pcm, rate, 16_000)
    };
    let need = core::scratch_len(&model, pcm.len());
    let mut fs = vec![0f32; need.f32s];
    let mut is = vec![0i8; need.i8s];
    let mut page = vec![0u32; need.page_words];
    let mut ids = vec![0u16; 4096];
    let mut pager = core::SlicePager(bytes);
    let mut chunks: Vec<Vec<f32>> = Vec::new();
    let t1 = std::time::Instant::now();
    let n = core::transcribe_with(
        &model,
        &mut pager,
        pcm.as_slice(),
        &mut fs,
        &mut is,
        &mut page,
        &mut ids,
        &mut |enc, _| chunks.push(enc.to_vec()),
    )
    .map_err(|e| anyhow!("{e}"))?;
    let secs = t1.elapsed().as_secs_f64();
    let mut text = vec![0u8; n * 32 + 1];
    let len = model.decode(&ids[..n], &mut text);
    println!("mcu: {}", String::from_utf8_lossy(&text[..len]));
    eprintln!(
        "{} chunks, {:.0} ms ({:.2}× real time on this host), scratch {:.1} MiB + page {:.1} MiB",
        chunks.len(),
        secs * 1e3,
        pcm.len() as f64 / 16_000.0 / secs,
        (need.f32s * 4 + need.i8s) as f64 / 1048576.0,
        need.page_words as f64 * 4.0 / 1048576.0
    );
    if let Some(dir) = get("--dump") {
        std::fs::create_dir_all(dir)?;
        let mut fb = vec![0f32; core::fbank_frames(pcm.len()) * model.config().n_mels];
        let frames = core::fbank_features(&model, pcm.as_slice(), &mut fb);
        let raw = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        std::fs::write(
            format!("{dir}/fbank.f32"),
            raw(&fb[..frames * model.config().n_mels]),
        )?;
        for (i, c) in chunks.iter().enumerate() {
            std::fs::write(format!("{dir}/enc_{i}.f32"), raw(c))?;
        }
        std::fs::write(
            format!("{dir}/ids.txt"),
            ids[..n]
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        )?;
        eprintln!(
            "dumped fbank ({frames} frames) and {} chunks to {dir}",
            chunks.len()
        );
    }
    Ok(())
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

fn wer(model: &core::Model<'_>, bytes: &[u8], list: &str, limit: usize) -> Result<()> {
    let base = Path::new(list).parent().unwrap_or(Path::new("."));
    let (mut words, mut errs, mut n) = (0usize, 0usize, 0usize);
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
        let need = core::scratch_len(model, pcm.len());
        let mut fs = vec![0f32; need.f32s];
        let mut is = vec![0i8; need.i8s];
        let mut page = vec![0u32; need.page_words];
        let mut ids = vec![0u16; 4096];
        let k = core::transcribe(
            model,
            &mut core::SlicePager(bytes),
            pcm.as_slice(),
            &mut fs,
            &mut is,
            &mut page,
            &mut ids,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let mut buf = vec![0u8; k * 32 + 1];
        let len = model.decode(&ids[..k], &mut buf);
        let (r, h) = (norm(text), norm(&String::from_utf8_lossy(&buf[..len])));
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
        words += r.len();
        errs += prev[h.len()];
        n += 1;
    }
    println!(
        "{n} utterances, {words} words: WER {:.2}%",
        100.0 * errs as f64 / words as f64
    );
    Ok(())
}
