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

//! Moonshine v2 streaming ASR ([arXiv:2602.12241](https://arxiv.org/abs/2602.12241),
//! `moonshine-ai/moonshine-streaming-tiny`) without an operating system.
//!
//! ```text
//! PCM ─► 80-sample frames ─► CMVN ─► asinh ─► linear ─► SiLU          (streamed)
//!     ─► causal conv k5/s2 ─► SiLU ─► causal conv k5/s2                (streamed)
//!     ─► N × [LN ─► windowed self-attn ─► LN ─► GELU MLP] ─► LN
//!     ─► + learned positions ─► decoder: RoPE self-attn (KV cache)
//!        + cross-attn (int8 K/V computed once) + SwiGLU MLP ─► greedy
//! ```
//!
//! Weights are int8 or int4 per matrix (the exporter decides; the 32k-row
//! embedding and LM head are int4 by default). Nothing allocates: size the
//! two scratch arenas with [`scratch_len`]. The model is read in place —
//! memory-mapped flash, or PSRAM it was copied into from NAND or SD.

#![cfg_attr(not(feature = "std"), no_std)]

mod decoder;
mod encoder;
pub mod model;

pub use model::{Config, Model};
pub use rlx_mcu::{Error, Pcm};

/// Element counts of the two scratch arenas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchLen {
    /// `f32` elements.
    pub f32s: usize,
    /// `i8` elements.
    pub i8s: usize,
}

impl ScratchLen {
    /// Total bytes.
    pub fn bytes(&self) -> usize {
        self.f32s * 4 + self.i8s
    }
}

/// Scratch needed to transcribe `n` samples.
pub fn scratch_len(model: &Model<'_>, n: usize) -> ScratchLen {
    let cfg = model.config();
    let (_, _, t) = cfg.lengths(n);
    let m = cfg.max_tokens(n);
    let x = t * cfg.d_enc.max(cfg.d_dec);
    let enc = encoder::f32_len(cfg, t);
    let cross = decoder::cross_f32_len(cfg, t);
    let dec = decoder::f32_len(cfg, t, m);
    ScratchLen {
        f32s: x + enc
            .max(cross + decoder::cross_build_len(cfg, t))
            .max(cross + dec),
        i8s: xq_len(cfg) + encoder::i8_len(cfg, t).max(decoder::cross_i8_len(cfg, t) + t),
    }
}

/// Two quantized activation rows of the widest matrix input.
pub(crate) fn xq_len(cfg: &Config) -> usize {
    let w = [
        cfg.frame_len,
        model::CONV_K * 2 * cfg.d_enc,
        cfg.enc_ff,
        cfg.enc_attn(),
        cfg.d_enc,
        cfg.d_dec,
        cfg.dec_ff,
        cfg.dec_attn(),
    ];
    2 * w.iter().copied().max().unwrap_or(0)
}

/// PCM → token ids (greedy), start token excluded, end token excluded.
/// `fs` / `is` must be at least [`scratch_len`] for `pcm.len()`.
pub fn transcribe<P: Pcm + ?Sized>(
    model: &Model<'_>,
    pcm: &P,
    fs: &mut [f32],
    is: &mut [i8],
    ids: &mut [u16],
) -> Result<usize, Error> {
    run(model, pcm, fs, is, ids, None)
}

/// [`transcribe`], also copying the encoder output after `final_norm`
/// (`[frames, d_enc]`) into `hidden`. For parity checks.
#[doc(hidden)]
pub fn transcribe_with_hidden<P: Pcm + ?Sized>(
    model: &Model<'_>,
    pcm: &P,
    fs: &mut [f32],
    is: &mut [i8],
    ids: &mut [u16],
    hidden: &mut [f32],
) -> Result<usize, Error> {
    run(model, pcm, fs, is, ids, Some(hidden))
}

/// The encoder alone, stopped after `layers` layers (`usize::MAX` = all,
/// with `final_norm`); writes `[frames, d_enc]` into `out`. For locating a
/// mismatch layer by layer.
#[doc(hidden)]
pub fn encode_partial<P: Pcm + ?Sized>(
    model: &Model<'_>,
    pcm: &P,
    layers: usize,
    fs: &mut [f32],
    is: &mut [i8],
    out: &mut [f32],
) -> Result<usize, Error> {
    let cfg = *model.config();
    let need = scratch_len(model, pcm.len());
    if fs.len() < need.f32s || is.len() < need.i8s {
        return Err(Error::ScratchTooSmall);
    }
    let (_, _, t) = cfg.lengths(pcm.len());
    let (x, rest) = fs.split_at_mut(t * cfg.d_enc.max(cfg.d_dec));
    let (xq, irest) = is.split_at_mut(xq_len(&cfg));
    encoder::run(model, pcm, layers, x, rest, xq, irest);
    let k = out.len().min(t * cfg.d_enc);
    out[..k].copy_from_slice(&x[..k]);
    Ok(t)
}

fn run<P: Pcm + ?Sized>(
    model: &Model<'_>,
    pcm: &P,
    fs: &mut [f32],
    is: &mut [i8],
    ids: &mut [u16],
    hidden: Option<&mut [f32]>,
) -> Result<usize, Error> {
    let cfg = *model.config();
    let n = pcm.len();
    if n < cfg.frame_len {
        return Ok(0);
    }
    let need = scratch_len(model, n);
    if fs.len() < need.f32s || is.len() < need.i8s {
        return Err(Error::ScratchTooSmall);
    }
    let (_, _, t) = cfg.lengths(n);
    let (x, rest) = fs.split_at_mut(t * cfg.d_enc.max(cfg.d_dec));
    let (xq, irest) = is.split_at_mut(xq_len(&cfg));
    let live = encoder::run(model, pcm, usize::MAX, x, rest, xq, irest);
    if let Some(h) = hidden {
        let k = h.len().min(t * cfg.d_enc);
        h[..k].copy_from_slice(&x[..k]);
    }
    let _ = live;
    decoder::run(model, n, t, x, rest, xq, irest, ids)
}
