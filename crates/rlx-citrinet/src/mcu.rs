// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Citrinet from a `.nemo`: the `.rct` exporter for [`rlx_citrinet_core`],
//! and an independent f32 reference forward to check it against.
//!
//! The checkpoint is read with `rlx-nemo`, which parses the archive without
//! executing pickle code. BatchNorm is folded into the preceding pointwise
//! (and residual) conv. This checkpoint stores no filterbank or window, so
//! the exporter builds NeMo's: a Slaney mel filterbank and a symmetric
//! (`periodic=False`) Hann window of `win_length`, centered in `n_fft`.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_mcu::blob::Writer;
use rlx_nemo::NemoModel;

const BN_EPS: f32 = 1e-3;

/// One Jasper block from `encoder.jasper`.
#[derive(Debug, Clone)]
pub struct BlockCfg {
    pub filters: usize,
    pub repeat: usize,
    pub kernel: usize,
    pub stride: usize,
    pub stride_last: bool,
    pub residual: bool,
    pub se: bool,
    pub se_reduction: usize,
}

/// What the exporter and reference need from the checkpoint config.
#[derive(Debug, Clone)]
pub struct CitrinetSpec {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop: usize,
    pub blocks: Vec<BlockCfg>,
    pub vocab: Vec<String>,
}

impl CitrinetSpec {
    /// Read `preprocessor`, `encoder.jasper` and `decoder.vocabulary`.
    pub fn from_nemo(model: &NemoModel) -> Result<Self> {
        let cfg = model.config();
        let sr = cfg.get_usize("preprocessor.sample_rate").unwrap_or(16_000);
        let n_fft = cfg.get_usize("preprocessor.n_fft").unwrap_or(512);
        let win =
            (cfg.get_f64("preprocessor.window_size").unwrap_or(0.025) * sr as f64).round() as usize;
        let hop = (cfg.get_f64("preprocessor.window_stride").unwrap_or(0.01) * sr as f64).round()
            as usize;
        let n_mels = cfg
            .get_usize("preprocessor.features")
            .or_else(|| cfg.get_usize("encoder.feat_in"))
            .unwrap_or(80);
        if cfg
            .get_str("preprocessor.normalize")
            .unwrap_or("per_feature")
            != "per_feature"
        {
            bail!("citrinet mcu export: only per_feature normalization is supported");
        }
        let seq = cfg
            .get("encoder.jasper")
            .and_then(|v| v.as_sequence())
            .ok_or_else(|| anyhow!("no encoder.jasper list"))?;
        let first = |v: &serde_yaml::Value, k: &str| -> Option<usize> {
            let x = v.get(k)?;
            x.as_u64()
                .or_else(|| x.as_sequence()?.first()?.as_u64())
                .map(|u| u as usize)
        };
        let mut blocks = Vec::new();
        for b in seq {
            if b.get("separable").and_then(|v| v.as_bool()) != Some(true) {
                bail!("citrinet mcu export: only separable blocks are supported");
            }
            if first(b, "dilation").unwrap_or(1) != 1 {
                bail!("citrinet mcu export: dilation is not supported");
            }
            let mode = b
                .get("residual_mode")
                .and_then(|v| v.as_str())
                .unwrap_or("add");
            let stride = first(b, "stride").unwrap_or(1);
            if stride > 1
                && mode != "stride_add"
                && b.get("residual").and_then(|v| v.as_bool()) == Some(true)
            {
                bail!("citrinet mcu export: strided residual needs residual_mode stride_add");
            }
            if b.get("se_context_size")
                .and_then(|v| v.as_i64())
                .unwrap_or(-1)
                != -1
            {
                bail!("citrinet mcu export: only global SE (se_context_size = -1)");
            }
            blocks.push(BlockCfg {
                filters: first(b, "filters").ok_or_else(|| anyhow!("block filters"))?,
                repeat: first(b, "repeat").unwrap_or(1),
                kernel: first(b, "kernel").ok_or_else(|| anyhow!("block kernel"))?,
                stride,
                stride_last: b
                    .get("stride_last")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                residual: b.get("residual").and_then(|v| v.as_bool()).unwrap_or(false),
                se: b.get("se").and_then(|v| v.as_bool()).unwrap_or(false),
                se_reduction: first(b, "se_reduction_ratio").unwrap_or(8),
            });
        }
        let vocab: Vec<String> = cfg
            .get("decoder.vocabulary")
            .and_then(|v| v.as_sequence())
            .ok_or_else(|| anyhow!("no decoder.vocabulary"))?
            .iter()
            .map(|v| v.as_str().map(String::from).unwrap_or_default())
            .collect();
        Ok(Self {
            sample_rate: sr,
            n_mels,
            n_fft,
            win_length: win,
            hop,
            blocks,
            vocab,
        })
    }
}

/// Folded block weights, shared by the exporter and the reference.
pub struct Weights {
    /// Per block, per repeat: `(dw [cin·k], pw [out, cin], pw bias [out])`.
    pub repeats: Vec<Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>>,
    /// Per block: SE `(fc0 [h, c], fc2 [c, h])`.
    pub se: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    /// Per block: residual `(w [out, in], bias [out])`.
    pub res: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    /// CTC head `([classes, c], [classes])`.
    pub head: (Vec<f32>, Vec<f32>),
    /// Window centered in `n_fft`.
    pub window: Vec<f32>,
    /// Dense filterbank `[n_mels, n_fft/2+1]`.
    pub fb: Vec<f32>,
}

fn t(model: &NemoModel, k: &str, n: usize) -> Result<Vec<f32>> {
    let v = model
        .tensor(k)
        .with_context(|| format!("nemo tensor {k}"))?
        .data;
    ensure!(v.len() == n, "{k}: {} values, expected {n}", v.len());
    Ok(v)
}

/// `(scale, shift)` of an eval-mode BatchNorm.
fn bn(model: &NemoModel, p: &str, c: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let g = t(model, &format!("{p}.weight"), c)?;
    let b = t(model, &format!("{p}.bias"), c)?;
    let m = t(model, &format!("{p}.running_mean"), c)?;
    let v = t(model, &format!("{p}.running_var"), c)?;
    let s: Vec<f32> = (0..c).map(|i| g[i] / (v[i] + BN_EPS).sqrt()).collect();
    let sh: Vec<f32> = (0..c).map(|i| b[i] - m[i] * s[i]).collect();
    Ok((s, sh))
}

fn fold(w: &mut [f32], cin: usize, s: &[f32]) {
    for (o, &so) in s.iter().enumerate() {
        for v in &mut w[o * cin..(o + 1) * cin] {
            *v *= so;
        }
    }
}

/// librosa-style Slaney mel filterbank, `[n_mels, n_fft/2+1]`.
fn slaney_fb(sr: f64, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let hz_to_mel = |hz: f64| {
        let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
        let logstep = 6.4f64.ln() / 27.0;
        if hz >= min_log_hz {
            min_log_hz / f_sp + (hz / min_log_hz).ln() / logstep
        } else {
            hz / f_sp
        }
    };
    let mel_to_hz = |m: f64| {
        let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
        let min_log_mel = min_log_hz / f_sp;
        let logstep = 6.4f64.ln() / 27.0;
        if m >= min_log_mel {
            min_log_hz * ((m - min_log_mel) * logstep).exp()
        } else {
            f_sp * m
        }
    };
    let n_freq = n_fft / 2 + 1;
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut fb = vec![0f32; n_mels * n_freq];
    for m in 0..n_mels {
        let (l, c, u) = (pts[m], pts[m + 1], pts[m + 2]);
        let enorm = 2.0 / (u - l);
        for k in 0..n_freq {
            let f = k as f64 * sr / n_fft as f64;
            let tri = ((f - l) / (c - l)).min((u - f) / (u - c)).max(0.0);
            fb[m * n_freq + k] = (tri * enorm) as f32;
        }
    }
    fb
}

/// Load and fold every weight.
pub fn load_weights(model: &NemoModel, spec: &CitrinetSpec) -> Result<Weights> {
    let mut repeats = Vec::new();
    let mut se = Vec::new();
    let mut res = Vec::new();
    let mut c_in = spec.n_mels;
    for (bi, b) in spec.blocks.iter().enumerate() {
        let p = |s: String| format!("encoder.encoder.{bi}.{s}");
        let mut reps = Vec::new();
        for r in 0..b.repeat {
            let cin = if r == 0 { c_in } else { b.filters };
            // mconv: [dw, pw, bn, act, drop] per repeat; the last repeat has no act/drop.
            let base = r * 5;
            let dw = t(
                model,
                &p(format!("mconv.{base}.conv.weight")),
                cin * b.kernel,
            )?;
            let mut pw = t(
                model,
                &p(format!("mconv.{}.conv.weight", base + 1)),
                b.filters * cin,
            )?;
            let (s, sh) = bn(model, &p(format!("mconv.{}", base + 2)), b.filters)?;
            fold(&mut pw, cin, &s);
            reps.push((dw, pw, sh));
        }
        repeats.push(reps);
        let se_idx = (b.repeat - 1) * 5 + 3;
        se.push(if b.se {
            let h = b.filters / b.se_reduction;
            Some((
                t(
                    model,
                    &p(format!("mconv.{se_idx}.fc.0.weight")),
                    h * b.filters,
                )?,
                t(
                    model,
                    &p(format!("mconv.{se_idx}.fc.2.weight")),
                    b.filters * h,
                )?,
            ))
        } else {
            None
        });
        res.push(if b.residual {
            let mut w = t(model, &p("res.0.0.conv.weight".into()), b.filters * c_in)?;
            let (s, sh) = bn(model, &p("res.0.1".into()), b.filters)?;
            fold(&mut w, c_in, &s);
            Some((w, sh))
        } else {
            None
        });
        c_in = b.filters;
    }
    let nc = spec.vocab.len() + 1;
    let head = (
        t(model, "decoder.decoder_layers.0.weight", nc * c_in)?,
        t(model, "decoder.decoder_layers.0.bias", nc)?,
    );
    let mut window = vec![0f32; spec.n_fft];
    let left = (spec.n_fft - spec.win_length) / 2;
    for n in 0..spec.win_length {
        let v = 0.5
            - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (spec.win_length - 1) as f64).cos();
        window[left + n] = v as f32;
    }
    let fb = slaney_fb(spec.sample_rate as f64, spec.n_fft, spec.n_mels);
    Ok(Weights {
        repeats,
        se,
        res,
        head,
        window,
        fb,
    })
}

/// Build the `.rct` blob. `bits` 8 (int8 per row) or 9 (int8 per group of 16).
pub fn export_rct(model: &NemoModel, bits: u32) -> Result<Vec<u8>> {
    let spec = CitrinetSpec::from_nemo(model)?;
    let w = load_weights(model, &spec)?;
    Ok(stage_blob(&spec, &w, bits, 0..spec.blocks.len()))
}

/// How to cut the model into pipeline stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stages {
    /// This many stages of about equal compute (`rlx_citrinet_core::block_costs`
    /// over a 5 s utterance, balanced with `rlx_mcu::split::balance`).
    Balanced(usize),
    /// Explicit block boundaries: `[0, c1, …, n_blocks]`.
    Cuts(Vec<usize>),
}

/// One `.rct` per stage, each holding only its blocks (the first also does
/// the log-mel frontend, the last the CTC head). Returns the block
/// boundaries used and the blobs. Chained, the stages give the same bits as
/// [`export_rct`]'s single model.
pub fn export_rct_stages(
    model: &NemoModel,
    bits: u32,
    stages: &Stages,
) -> Result<(Vec<usize>, Vec<Vec<u8>>)> {
    let spec = CitrinetSpec::from_nemo(model)?;
    let w = load_weights(model, &spec)?;
    let n = spec.blocks.len();
    let cuts = match stages {
        Stages::Cuts(c) => c.clone(),
        Stages::Balanced(k) => {
            let full = Blob::new(stage_blob(&spec, &w, bits, 0..n));
            let m =
                rlx_citrinet_core::Model::parse(full.bytes()).map_err(|e| anyhow!("parse: {e}"))?;
            let mut costs = vec![0u64; n];
            rlx_citrinet_core::block_costs(&m, 5 * spec.sample_rate, &mut costs);
            let mut cuts = vec![0usize; k + 1];
            rlx_mcu::split::balance(&costs, &mut cuts).map_err(|e| anyhow!("balance: {e}"))?;
            cuts
        }
    };
    anyhow::ensure!(
        cuts.len() >= 2
            && cuts[0] == 0
            && cuts[cuts.len() - 1] == n
            && cuts.windows(2).all(|p| p[0] < p[1]),
        "stage cuts must rise strictly from 0 to {n}, got {cuts:?}"
    );
    let blobs = cuts
        .windows(2)
        .map(|p| stage_blob(&spec, &w, bits, p[0]..p[1]))
        .collect();
    Ok((cuts, blobs))
}

/// A blob copied into word-aligned storage, as the core requires.
pub struct Blob(Vec<u32>, usize);

impl Blob {
    /// Copy `bytes` into aligned storage.
    pub fn new(bytes: Vec<u8>) -> Self {
        let words = bytes
            .chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                u32::from_le_bytes(w)
            })
            .collect();
        Self(words, bytes.len())
    }
    /// The bytes, 4-byte aligned.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: u32 storage viewed as bytes, within its length.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), self.1) }
    }
}

fn stage_blob(
    spec: &CitrinetSpec,
    w: &Weights,
    bits: u32,
    range: std::ops::Range<usize>,
) -> Vec<u8> {
    let nc = spec.vocab.len() + 1;
    let head = range.end == spec.blocks.len();
    let mut out = Writer::new(rlx_citrinet_core::MAGIC, rlx_citrinet_core::VERSION);
    for v in [
        spec.sample_rate,
        spec.n_mels,
        spec.n_fft,
        spec.hop,
        range.len(),
        nc,
        nc - 1,
        range.start,
        spec.blocks.len(),
        usize::from(head),
    ] {
        out.u32(v as u32);
    }
    out.f32s(&w.window);
    let n_freq = spec.n_fft / 2 + 1;
    let (mut starts, mut offs, mut bands) = (Vec::new(), vec![0u32], Vec::new());
    for m in 0..spec.n_mels {
        let row = &w.fb[m * n_freq..(m + 1) * n_freq];
        let first = row.iter().position(|&v| v != 0.0).unwrap_or(0);
        let last = row.iter().rposition(|&v| v != 0.0).unwrap_or(0);
        if row[first] != 0.0 {
            bands.extend_from_slice(&row[first..=last]);
        }
        starts.push(first as u32);
        offs.push(bands.len() as u32);
    }
    out.u32s(&starts);
    out.u32s(&offs);
    out.f32s(&bands);
    let mut meta = Vec::new();
    let mut c_in = spec.n_mels;
    for (bi, b) in spec.blocks.iter().enumerate() {
        if !range.contains(&bi) {
            c_in = b.filters;
            continue;
        }
        meta.extend([
            c_in as u32,
            b.filters as u32,
            b.repeat as u32,
            b.kernel as u32,
            b.stride as u32,
            u32::from(b.stride_last),
            u32::from(b.residual),
            u32::from(b.se),
            (b.filters / b.se_reduction) as u32,
        ]);
        c_in = b.filters;
    }
    out.u32s(&meta);
    let mut c_in = spec.n_mels;
    for (bi, b) in spec.blocks.iter().enumerate() {
        if !range.contains(&bi) {
            c_in = b.filters;
            continue;
        }
        for (r, (dw, pw, pb)) in w.repeats[bi].iter().enumerate() {
            let cin = if r == 0 { c_in } else { b.filters };
            // Tap-major `[k, cin]`: the conv's inner loop runs over channels.
            let mut tap_major = vec![0f32; dw.len()];
            for ch in 0..cin {
                for j in 0..b.kernel {
                    tap_major[j * cin + ch] = dw[ch * b.kernel + j];
                }
            }
            out.f32s(&tap_major);
            out.mat(pw, b.filters, cin, bits, 16);
            out.f32s(pb);
        }
        if let Some((f0, f2)) = &w.se[bi] {
            out.f32s(f0);
            out.f32s(f2);
        }
        if let Some((rw, rb)) = &w.res[bi] {
            out.mat(rw, b.filters, c_in, bits, 16);
            out.f32s(rb);
        }
        c_in = b.filters;
    }
    if head {
        out.mat(&w.head.0, nc, c_in, bits, 16);
        out.f32s(&w.head.1);
    }
    let (mut po, mut pb) = (vec![0u32], Vec::new());
    for p in spec.vocab.iter().map(String::as_str).chain([""]) {
        pb.extend_from_slice(p.as_bytes());
        po.push(pb.len() as u32);
    }
    out.u32s(&po);
    out.bytes(&pb);
    out.finish()
}

/// f32 reference forward over a time-major `[t, n_mels]` mel: returns the
/// encoder output `[t', c]` and greedy CTC ids. Independent of the core's
/// code, for checking it.
pub fn reference(
    spec: &CitrinetSpec,
    w: &Weights,
    mel: &[f32],
    frames: usize,
) -> (Vec<f32>, Vec<u32>) {
    let mut x = mel.to_vec();
    let (mut t, mut c) = (frames, spec.n_mels);
    for (bi, b) in spec.blocks.iter().enumerate() {
        let xin = x.clone();
        let (t_in, c_in) = (t, c);
        let mut cur = x;
        let (mut tl, mut cl) = (t, c);
        for (r, (dw, pw, pb)) in w.repeats[bi].iter().enumerate() {
            let s = if b.stride > 1 && (!b.stride_last || r + 1 == b.repeat) {
                b.stride
            } else {
                1
            };
            let k = b.kernel;
            let to = (tl - 1) / s + 1;
            let mut y = vec![0f32; to * cl];
            for o in 0..to {
                for j in 0..k {
                    let si = (o * s + j) as isize - (k / 2) as isize;
                    if si < 0 || si as usize >= tl {
                        continue;
                    }
                    for ch in 0..cl {
                        y[o * cl + ch] += dw[ch * k + j] * cur[si as usize * cl + ch];
                    }
                }
            }
            let mut z = vec![0f32; to * b.filters];
            for o in 0..to {
                for f in 0..b.filters {
                    let mut acc = pb[f];
                    for ch in 0..cl {
                        acc += pw[f * cl + ch] * y[o * cl + ch];
                    }
                    z[o * b.filters + f] = if r + 1 < b.repeat { acc.max(0.0) } else { acc };
                }
            }
            cur = z;
            tl = to;
            cl = b.filters;
        }
        if let Some((f0, f2)) = &w.se[bi] {
            let h = b.filters / b.se_reduction;
            let mut mean = vec![0f32; cl];
            for o in 0..tl {
                for ch in 0..cl {
                    mean[ch] += cur[o * cl + ch] / tl as f32;
                }
            }
            let hid: Vec<f32> = (0..h)
                .map(|j| {
                    (0..cl)
                        .map(|ch| f0[j * cl + ch] * mean[ch])
                        .sum::<f32>()
                        .max(0.0)
                })
                .collect();
            let gate: Vec<f32> = (0..cl)
                .map(|ch| {
                    1.0 / (1.0 + (-(0..h).map(|j| f2[ch * h + j] * hid[j]).sum::<f32>()).exp())
                })
                .collect();
            for o in 0..tl {
                for ch in 0..cl {
                    cur[o * cl + ch] *= gate[ch];
                }
            }
        }
        if let Some((rw, rb)) = &w.res[bi] {
            for o in 0..tl {
                let si = (o * b.stride).min(t_in - 1);
                for f in 0..cl {
                    let mut acc = rb[f];
                    for ch in 0..c_in {
                        acc += rw[f * c_in + ch] * xin[si * c_in + ch];
                    }
                    cur[o * cl + f] += acc;
                }
            }
        }
        for v in &mut cur {
            *v = v.max(0.0);
        }
        x = cur;
        t = tl;
        c = cl;
    }
    let nc = spec.vocab.len() + 1;
    let mut ids = Vec::new();
    let mut prev = usize::MAX;
    for o in 0..t {
        let mut best = (0usize, f32::NEG_INFINITY);
        for k in 0..nc {
            let mut acc = w.head.1[k];
            for ch in 0..c {
                acc += w.head.0[k * c + ch] * x[o * c + ch];
            }
            if acc > best.1 {
                best = (k, acc);
            }
        }
        if best.0 != prev && best.0 != nc - 1 {
            ids.push(best.0 as u32);
        }
        prev = best.0;
    }
    (x, ids)
}

/// Write the `.rct` for `nemo` to `out`; returns the byte count.
pub fn export_rct_file(nemo: &Path, out: &Path, bits: u32) -> Result<usize> {
    let model = NemoModel::open(nemo).with_context(|| format!("open {}", nemo.display()))?;
    let blob = export_rct(&model, bits)?;
    std::fs::write(out, &blob)?;
    Ok(blob.len())
}
