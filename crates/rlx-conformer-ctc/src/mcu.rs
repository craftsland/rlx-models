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

//! Export a `.nemo` to the `.rcc` int8 blob that
//! [`rlx_conformer_ctc_core`] runs on MCUs.
//!
//! Weights of every matrix product are quantized symmetric int8 with one
//! scale per output channel. BatchNorm is folded into the depthwise conv,
//! the second subsampling conv is reordered to `[out][kh][kw][in]`, and the
//! mel filterbank is stored banded (its nonzero run per filter).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use rlx_conformer_ctc_core::blob::Writer;
use rlx_nemo::NemoModel;

use crate::config::AsrConfig;
use crate::mel::Frontend;
use crate::tokenizer::SpmTokenizer;
use crate::weights::keys;

const BN_EPS: f32 = 1e-5;

struct Src<'a> {
    model: &'a NemoModel,
}

impl Src<'_> {
    fn t(&self, key: &str, len: usize) -> Result<Vec<f32>> {
        let t = self
            .model
            .tensor(key)
            .with_context(|| format!("nemo tensor {key:?}"))?;
        ensure!(
            t.data.len() == len,
            "{key}: {} values, expected {len}",
            t.data.len()
        );
        Ok(t.data)
    }
}

/// Build the `.rcc` blob for a classic `striding` Conformer-CTC checkpoint.
pub fn export_rcc(model: &NemoModel) -> Result<Vec<u8>> {
    let cfg = AsrConfig::from_nemo(model.config())?;
    ensure!(
        cfg.subsampling == "striding" && cfg.subsampling_factor == 4,
        "export-mcu supports striding x4 subsampling, got {} x{}",
        cfg.subsampling,
        cfg.subsampling_factor
    );
    ensure!(
        cfg.normalize == "per_feature",
        "normalize {:?}",
        cfg.normalize
    );
    let s = Src { model };
    let (d, ff, ch, k) = (
        cfg.d_model,
        cfg.ff_dim(),
        cfg.subsampling_conv_channels,
        cfg.conv_kernel,
    );
    let n_freq = cfg.n_fft / 2 + 1;
    let f2 = cfg.freq_after_subsample();

    let mut w = Writer::new(
        rlx_conformer_ctc_core::blob::MAGIC,
        rlx_conformer_ctc_core::blob::VERSION,
    );
    for v in [
        cfg.sample_rate,
        cfg.n_mels,
        cfg.n_fft,
        cfg.hop_length,
        d,
        cfg.n_layers,
        cfg.n_heads,
        ff,
        k,
        ch,
        cfg.num_classes,
        cfg.blank_id,
    ] {
        w.u32(v as u32);
    }

    // Frontend: the model's own window (centered in n_fft) and filterbank.
    let fe = Frontend::from_model(model, &cfg)?;
    let mut window = vec![0.0f32; cfg.n_fft];
    let left = cfg.n_fft.saturating_sub(fe.window.len()) / 2;
    for (i, &v) in fe.window.iter().enumerate() {
        window[left + i] = v;
    }
    w.f32s(&window);
    let (mut starts, mut offs, mut weights) = (Vec::new(), vec![0u32], Vec::new());
    for m in 0..cfg.n_mels {
        let row = &fe.fb[m * n_freq..(m + 1) * n_freq];
        let first = row.iter().position(|&v| v != 0.0).unwrap_or(0);
        let last = row.iter().rposition(|&v| v != 0.0).unwrap_or(0);
        let run = if row[first] == 0.0 {
            &row[0..0]
        } else {
            &row[first..=last]
        };
        starts.push(first as u32);
        weights.extend_from_slice(run);
        offs.push(weights.len() as u32);
    }
    w.u32s(&starts);
    w.u32s(&offs);
    w.f32s(&weights);

    // Subsampling: conv0 [ch,1,3,3] f32; conv2 [ch,ch,3,3] → [out][kh][kw][in] int8.
    w.f32s(&s.t(&keys::pre_encode_conv(0, "weight"), ch * 9)?);
    w.f32s(&s.t(&keys::pre_encode_conv(0, "bias"), ch)?);
    let c2 = s.t(&keys::pre_encode_conv(2, "weight"), ch * ch * 9)?;
    let mut c2r = vec![0.0f32; ch * ch * 9];
    for o in 0..ch {
        for i in 0..ch {
            for kh in 0..3 {
                for kw in 0..3 {
                    c2r[o * 9 * ch + (kh * 3 + kw) * ch + i] = c2[((o * ch + i) * 3 + kh) * 3 + kw];
                }
            }
        }
    }
    w.q8(&c2r, ch, 9 * ch);
    w.f32s(&s.t(&keys::pre_encode_conv(2, "bias"), ch)?);
    w.q8(&s.t(keys::PRE_ENCODE_OUT_W, d * ch * f2)?, d, ch * f2);
    w.f32s(&s.t(keys::PRE_ENCODE_OUT_B, d)?);

    for l in 0..cfg.n_layers {
        let p = |suffix: &str| keys::enc_layer(l, suffix);
        let vec = |w: &mut Writer, key: &str, n: usize| -> Result<()> {
            w.f32s(&s.t(&p(key), n)?);
            Ok(())
        };
        let mat = |w: &mut Writer, key: &str, rows: usize, cols: usize| -> Result<()> {
            w.q8(&s.t(&p(key), rows * cols)?, rows, cols);
            Ok(())
        };
        let ffn = |w: &mut Writer, n: [&str; 6]| -> Result<()> {
            vec(w, n[0], d)?;
            vec(w, n[1], d)?;
            mat(w, n[2], ff, d)?;
            vec(w, n[3], ff)?;
            mat(w, n[4], d, ff)?;
            vec(w, n[5], d)
        };
        ffn(
            &mut w,
            [
                keys::NORM_FF1_W,
                keys::NORM_FF1_B,
                keys::FF1_L1_W,
                keys::FF1_L1_B,
                keys::FF1_L2_W,
                keys::FF1_L2_B,
            ],
        )?;
        vec(&mut w, keys::NORM_ATT_W, d)?;
        vec(&mut w, keys::NORM_ATT_B, d)?;
        for (wk, bk) in [
            (keys::ATT_Q_W, keys::ATT_Q_B),
            (keys::ATT_K_W, keys::ATT_K_B),
            (keys::ATT_V_W, keys::ATT_V_B),
            (keys::ATT_O_W, keys::ATT_O_B),
        ] {
            mat(&mut w, wk, d, d)?;
            vec(&mut w, bk, d)?;
        }
        mat(&mut w, keys::ATT_POS_W, d, d)?;
        vec(&mut w, keys::ATT_POS_U, d)?;
        vec(&mut w, keys::ATT_POS_V, d)?;

        vec(&mut w, keys::NORM_CONV_W, d)?;
        vec(&mut w, keys::NORM_CONV_B, d)?;
        mat(&mut w, keys::CONV_PW1_W, 2 * d, d)?;
        vec(&mut w, keys::CONV_PW1_B, 2 * d)?;
        // Depthwise with BatchNorm folded: (conv + b) * s + (beta - mean * s).
        let dw = s.t(&p(keys::CONV_DW_W), d * k)?;
        let dwb = s.t(&p(keys::CONV_DW_B), d)?;
        let gamma = s.t(&p(keys::CONV_BN_W), d)?;
        let beta = s.t(&p(keys::CONV_BN_B), d)?;
        let mean = s.t(&p(keys::CONV_BN_MEAN), d)?;
        let var = s.t(&p(keys::CONV_BN_VAR), d)?;
        let mut fw = vec![0.0f32; d * k];
        let mut fb = vec![0.0f32; d];
        for c in 0..d {
            let sc = gamma[c] / (var[c] + BN_EPS).sqrt();
            for j in 0..k {
                fw[c * k + j] = dw[c * k + j] * sc;
            }
            fb[c] = dwb[c] * sc + beta[c] - mean[c] * sc;
        }
        w.f32s(&fw);
        w.f32s(&fb);
        mat(&mut w, keys::CONV_PW2_W, d, d)?;
        vec(&mut w, keys::CONV_PW2_B, d)?;

        ffn(
            &mut w,
            [
                keys::NORM_FF2_W,
                keys::NORM_FF2_B,
                keys::FF2_L1_W,
                keys::FF2_L1_B,
                keys::FF2_L2_W,
                keys::FF2_L2_B,
            ],
        )?;
        vec(&mut w, keys::NORM_OUT_W, d)?;
        vec(&mut w, keys::NORM_OUT_B, d)?;
    }

    w.q8(&s.t(keys::CTC_W, cfg.num_classes * d)?, cfg.num_classes, d);
    w.f32s(&s.t(keys::CTC_B, cfg.num_classes)?);

    // SentencePiece pieces for on-device detokenization (blank has none).
    let tok = model
        .tokenizers()
        .iter()
        .find(|t| t.name.ends_with(".model"))
        .and_then(|t| SpmTokenizer::from_model_bytes(&t.bytes).ok());
    let (mut offs, mut bytes) = (vec![0u32], Vec::new());
    for id in 0..cfg.num_classes {
        let piece = tok.as_ref().and_then(|t| t.piece(id)).unwrap_or("");
        bytes.extend_from_slice(piece.as_bytes());
        offs.push(bytes.len() as u32);
    }
    w.u32s(&offs);
    w.bytes(&bytes);
    Ok(w.finish())
}

/// Read `nemo`, write the `.rcc` to `out`; returns the byte count.
pub fn export_rcc_file(nemo: &Path, out: &Path) -> Result<usize> {
    let model = NemoModel::open(nemo).with_context(|| format!("open {}", nemo.display()))?;
    let blob = export_rcc(&model)?;
    std::fs::write(out, &blob).with_context(|| format!("write {}", out.display()))?;
    Ok(blob.len())
}

/// A blob copied into 4-byte-aligned storage, as [`rlx_conformer_ctc_core`]
/// requires.
pub struct AlignedBlob {
    words: Vec<u32>,
    len: usize,
}

impl AlignedBlob {
    /// Copy `bytes` into aligned storage.
    pub fn new(bytes: &[u8]) -> Self {
        let mut words = vec![0u32; bytes.len().div_ceil(4)];
        for (i, c) in bytes.chunks(4).enumerate() {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            words[i] = u32::from_le_bytes(b);
        }
        Self {
            words,
            len: bytes.len(),
        }
    }

    /// The aligned bytes.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: u32 storage reinterpreted as bytes; `len <= 4 * words.len()`.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.len) }
    }
}

/// Run the int8 MCU pipeline on the host: PCM → token ids (+ encoder output
/// `[t, d_model]` when `features` is set).
pub fn transcribe_rcc(
    blob: &[u8],
    pcm: &[f32],
    features: Option<&mut Vec<f32>>,
) -> Result<Vec<u32>> {
    use rlx_conformer_ctc_core as core;
    let model = core::Model::parse(blob).map_err(|e| anyhow::anyhow!("{e}"))?;
    let fe = core::Frontend::new(model.config().n_fft);
    let need = core::scratch_len(&model, pcm.len());
    let mut fs = vec![0f32; need.f32s];
    let mut is = vec![0i8; need.i8s];
    let mut ids = vec![0u16; pcm.len() / model.config().hop + 1];
    let n = match features {
        Some(out) => {
            let cfg = model.config();
            let t = cfg.enc_frames(core::Frontend::frames(&model, pcm.len()));
            out.resize(t * cfg.d_model, 0.0);
            core::transcribe_with_features(&model, &fe, pcm, &mut fs, &mut is, &mut ids, out)
        }
        None => core::transcribe(&model, &fe, pcm, &mut fs, &mut is, &mut ids),
    }
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(ids[..n].iter().map(|&i| i as u32).collect())
}
