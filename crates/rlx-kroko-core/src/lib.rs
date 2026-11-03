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

//! Kroko streaming ASR — a k2 Zipformer2 transducer with a stateless
//! predictor — without an operating system, for a chip with external
//! storage.
//!
//! ```text
//! PCM ─► Kaldi fbank (80) ─► chunks of 141 frames, step 128
//!     ─► conv embed (3 convs + ConvNeXt, cached tail) ─► 64 frames × 192
//!     ─► 6 Zipformer2 stacks at ×1 ×2 ×4 ×8 ×4 ×2 (per-layer caches)
//!     ─► concat, ×2 downsample, project ─► 32 frames × 512
//!     ─► joiner(encoder, predictor(last 2 tokens)) ─► greedy, 1 token/frame
//! ```
//!
//! Kroko-L has 149 M parameters — far more than fits in an MCU's RAM, so
//! the model file is split ([`model`]): a small resident part read in place,
//! and one block per Zipformer layer streamed through a [`Pager`] into a
//! page buffer, every chunk. A real-time run reads the whole paged part
//! once per 1.28 s chunk.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod model;
mod zipformer;

pub use model::{Config, Model, Pager, SlicePager, Stack};
pub use rlx_mcu::{Error, Pcm};

use rlx_mcu::kernels::{carve, dequant_row, linear};
use rlx_mcu::mel::Frontend;

use model::{EMBED_C0, EMBED_C4, EMBED_C7, EMBED_HID};

const PREEMPH: f32 = 0.97;
const HOP: usize = 160;
const WIN: usize = 400;
const NFFT: usize = 512;
/// Seconds of silence appended so the last words leave the lookahead.
const TAIL_PAD_S: f64 = 0.66;

#[inline]
fn softplus(z: f32) -> f32 {
    if z > 20.0 {
        z
    } else {
        libm::log1pf(libm::expf(z))
    }
}

/// Zipformer SwooshL: `softplus(x − 4) − 0.08x − 0.035`.
#[inline]
pub(crate) fn swoosh_l(x: f32) -> f32 {
    softplus(x - 4.0) - 0.08 * x - 0.035
}

/// Zipformer SwooshR: `softplus(x − 1) − 0.08x − 0.313261687`.
#[inline]
pub(crate) fn swoosh_r(x: f32) -> f32 {
    softplus(x - 1.0) - 0.08 * x - 0.313_261_7
}

/// Element counts of the arenas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchLen {
    /// `f32` elements (fbank, states, scratch).
    pub f32s: usize,
    /// `i8` elements.
    pub i8s: usize,
    /// `u32` words of the layer page buffer.
    pub page_words: usize,
}

impl ScratchLen {
    /// Total bytes.
    pub fn bytes(&self) -> usize {
        self.f32s * 4 + self.i8s + self.page_words * 4
    }
}

/// Fbank frames for `n` samples (tail padding included, snip_edges=false).
pub fn fbank_frames(n: usize) -> usize {
    let padded = n + libm::round(TAIL_PAD_S * 16_000.0) as usize;
    (padded + HOP / 2) / HOP
}

fn embed_len(cfg: &Config) -> usize {
    let t0 = cfg.chunk_size - 2;
    let t4 = (t0 - 3) / 2 + 1;
    let t7 = cfg.conv7_frames();
    let (f0, f4, f7) = (cfg.n_mels, (cfg.n_mels - 3) / 2 + 1, cfg.conv7_freq());
    let frames = cfg.frames();
    t0 * f0 * EMBED_C0
        + t4 * f4 * EMBED_C4
        + (t7 + 3) * f7 * EMBED_C7
        + frames * f7 * EMBED_C7
        + 2 * f7 * EMBED_HID
        + frames * f7 * EMBED_C7
}

fn embed_state_len(cfg: &Config) -> usize {
    3 * cfg.conv7_freq() * EMBED_C7
}

/// Scratch to transcribe `n` samples.
pub fn scratch_len(model: &Model<'_>, n: usize) -> ScratchLen {
    let cfg = model.config();
    let fb = fbank_frames(n) * cfg.n_mels;
    let frames = cfg.frames();
    let work = embed_len(cfg).max(zipformer::scratch_len(cfg));
    let max_in = [
        EMBED_C7 * cfg.conv7_freq(),
        EMBED_HID,
        cfg.max_dim() * 4,
        cfg.enc_out,
        cfg.dec_hidden,
    ]
    .into_iter()
    .chain(cfg.stacks[..cfg.n_stacks].iter().flat_map(|s| s.ff))
    .max()
    .unwrap_or(0);
    ScratchLen {
        f32s: fb
            + 2 * NFFT
            + cfg.chunk_size * cfg.n_mels
            + embed_state_len(cfg)
            + zipformer::state_len(cfg)
            + frames * cfg.embed_dim
            + (frames / 2 + 1) * cfg.enc_out
            + 3 * cfg.dec_hidden
            + cfg.vocab
            + work,
        i8s: 2 * max_in,
        page_words: model.page_len().div_ceil(4),
    }
}

/// Kaldi fbank (snip_edges=false, povey window, DC removal, pre-emphasis,
/// power spectrum, HTK mel 20–7600 Hz, natural log) of `pcm` plus the tail
/// padding, time-major `[frames, n_mels]`.
fn fbank<P: Pcm + ?Sized>(
    model: &Model<'_>,
    fe: &Frontend,
    pcm: &P,
    out: &mut [f32],
    work: &mut [f32],
) -> usize {
    let cfg = model.config();
    let n = pcm.len();
    let padded = n + libm::round(TAIL_PAD_S * 16_000.0) as usize;
    let frames = (padded + HOP / 2) / HOP;
    let sample = |i: usize| if i < n { pcm.at(i) } else { 0.0 };
    let reflect = |mut i: isize| {
        let size = padded as isize;
        while i < 0 || i >= size {
            i = if i < 0 { -i - 1 } else { 2 * size - 1 - i };
        }
        i as usize
    };
    let (re, im) = work[..2 * NFFT].split_at_mut(NFFT);
    for f in 0..frames {
        let start = (f * HOP + HOP / 2) as isize - (WIN / 2) as isize;
        let mut mean = 0f32;
        for j in 0..WIN {
            mean += sample(reflect(start + j as isize));
        }
        mean /= WIN as f32;
        let mut prev = 0f32;
        for j in 0..WIN {
            let cur = sample(reflect(start + j as isize)) - mean;
            let e = if j == 0 {
                cur - PREEMPH * cur
            } else {
                cur - PREEMPH * prev
            };
            prev = cur;
            re[j] = e * model.window[j];
            im[j] = 0.0;
        }
        re[WIN..].fill(0.0);
        im[WIN..].fill(0.0);
        fe.fft(re, im);
        for b in 0..NFFT / 2 {
            re[b] = re[b] * re[b] + im[b] * im[b];
        }
        for m in 0..cfg.n_mels {
            let (s0, a, b) = (
                model.band_start[m] as usize,
                model.band_off[m] as usize,
                model.band_off[m + 1] as usize,
            );
            let mut e = 0f32;
            for (k, &wk) in model.band_w[a..b].iter().enumerate() {
                e += wk * re[s0 + k];
            }
            out[f * cfg.n_mels + m] = libm::logf(e.max(f32::EPSILON));
        }
    }
    frames
}

/// The fbank features `transcribe` computes (tail padding included),
/// time-major `[frames, n_mels]`; returns the frame count. For checking the
/// encoder against a reference fed the same features.
#[doc(hidden)]
pub fn fbank_features<P: Pcm + ?Sized>(model: &Model<'_>, pcm: &P, out: &mut [f32]) -> usize {
    let fe = Frontend::new(NFFT);
    let mut work = [0f32; 2 * NFFT];
    fbank(model, &fe, pcm, out, &mut work)
}

/// Conv embed for one chunk of `chunk_size` fbank frames → `x[frames, embed_dim]`.
fn embed(
    model: &Model<'_>,
    feats: &[f32],
    cache: &mut [f32],
    scratch: &mut [f32],
    xq: &mut [i8],
    x: &mut [f32],
) {
    let cfg = model.config();
    let e = &model.embed;
    let nm = cfg.n_mels;
    let t0 = cfg.chunk_size - 2;
    let f4 = (nm - 3) / 2 + 1;
    let t4 = (t0 - 3) / 2 + 1;
    let (t7, f7) = (cfg.conv7_frames(), cfg.conv7_freq());
    let frames = cfg.frames();
    let mut r = &mut scratch[..];
    let a0 = carve(&mut r, t0 * nm * EMBED_C0);
    let a4 = carve(&mut r, t4 * f4 * EMBED_C4);
    let a7 = carve(&mut r, (t7 + 3) * f7 * EMBED_C7); // cache + conv7
    let dw = carve(&mut r, frames * f7 * EMBED_C7);
    let hid = carve(&mut r, 2 * f7 * EMBED_HID);
    let flat = carve(&mut r, frames * f7 * EMBED_C7);

    // conv0: 1→8, 3×3, freq padding 1; layout [t][f][c].
    for t in 0..t0 {
        for f in 0..nm {
            for c in 0..EMBED_C0 {
                let mut acc = e.conv0.1[c];
                for kh in 0..3 {
                    for kw in 0..3 {
                        let fi = f as isize + kw as isize - 1;
                        if fi < 0 || fi as usize >= nm {
                            continue;
                        }
                        acc += e.conv0.0[c * 9 + kh * 3 + kw] * feats[(t + kh) * nm + fi as usize];
                    }
                }
                a0[(t * nm + f) * EMBED_C0 + c] = swoosh_r(acc);
            }
        }
    }
    // conv4: 8→32, 3×3, stride 2×2.
    for t in 0..t4 {
        for f in 0..f4 {
            for c in 0..EMBED_C4 {
                let mut acc = e.conv4.1[c];
                for kh in 0..3 {
                    for kw in 0..3 {
                        let src = &a0[((2 * t + kh) * nm + 2 * f + kw) * EMBED_C0..][..EMBED_C0];
                        for (ci, &v) in src.iter().enumerate() {
                            acc += e.conv4.0[((c * EMBED_C0 + ci) * 3 + kh) * 3 + kw] * v;
                        }
                    }
                }
                a4[(t * f4 + f) * EMBED_C4 + c] = swoosh_r(acc);
            }
        }
    }
    // conv7: 32→128, 3×3, stride 1×2, after the 3 cached frames.
    a7[..3 * f7 * EMBED_C7].copy_from_slice(cache);
    for t in 0..t7 {
        for f in 0..f7 {
            for c in 0..EMBED_C7 {
                let mut acc = e.conv7.1[c];
                for kh in 0..3 {
                    for kw in 0..3 {
                        let src = &a4[((t + kh) * f4 + 2 * f + kw) * EMBED_C4..][..EMBED_C4];
                        for (ci, &v) in src.iter().enumerate() {
                            acc += e.conv7.0[((c * EMBED_C4 + ci) * 3 + kh) * 3 + kw] * v;
                        }
                    }
                }
                a7[((t + 3) * f7 + f) * EMBED_C7 + c] = swoosh_r(acc);
            }
        }
    }
    // The next chunk's cache: conv7 frames `frames-3 .. frames`.
    cache.copy_from_slice(&a7[frames * f7 * EMBED_C7..(frames + 3) * f7 * EMBED_C7]);
    // ConvNeXt depthwise 7×7 over (cache + conv7), time valid, freq pad 3.
    for t in 0..frames {
        for f in 0..f7 {
            for c in 0..EMBED_C7 {
                let mut acc = e.dw.1[c];
                for kh in 0..7 {
                    for kw in 0..7 {
                        let fi = f as isize + kw as isize - 3;
                        if fi < 0 || fi as usize >= f7 {
                            continue;
                        }
                        acc += e.dw.0[c * 49 + kh * 7 + kw]
                            * a7[((t + kh) * f7 + fi as usize) * EMBED_C7 + c];
                    }
                }
                dw[(t * f7 + f) * EMBED_C7 + c] = acc;
            }
        }
    }
    // Pointwise 128→384 SwooshL →128, plus the conv7 bypass, one frame at a
    // time; written channel-major (`[t][c·f7 + f]`) for the output linear.
    let out2 = &mut x[..f7 * EMBED_C7];
    for t in 0..frames {
        let rows = &dw[t * f7 * EMBED_C7..(t + 1) * f7 * EMBED_C7];
        linear(rows, f7, &e.pw1.0, Some(e.pw1.1), hid, xq);
        for v in &mut hid[..f7 * EMBED_HID] {
            *v = swoosh_l(*v);
        }
        linear(hid, f7, &e.pw2.0, Some(e.pw2.1), out2, xq);
        for f in 0..f7 {
            for c in 0..EMBED_C7 {
                let byp = a7[((t + 3) * f7 + f) * EMBED_C7 + c];
                flat[t * f7 * EMBED_C7 + c * f7 + f] = out2[f * EMBED_C7 + c] + byp;
            }
        }
    }
    linear(flat, frames, &e.out.0, Some(e.out.1), x, xq);
    // BiasNorm.
    let d = cfg.embed_dim;
    let g = libm::expf(e.norm_log_scale);
    for t in 0..frames {
        let row = &mut x[t * d..(t + 1) * d];
        let mut m = 0f32;
        for (c, &v) in row.iter().enumerate() {
            let z = v - e.norm_bias[c];
            m += z * z;
        }
        let inv = g / libm::sqrtf(m / d as f32);
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// The encoder alone over precomputed fbank `feats` (`[frames, n_mels]`, as
/// [`fbank_features`] makes them), from a zero state: `on_chunk` gets each
/// chunk's encoder output `[frames, enc_out]`. For distillation targets: a
/// window of an utterance's features runs exactly as a fresh utterance would.
/// `fs` needs [`scratch_len`] for `frames * 160` samples.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn encode_features<G: Pager + ?Sized>(
    model: &Model<'_>,
    pager: &mut G,
    feats: &[f32],
    frames: usize,
    fs: &mut [f32],
    is: &mut [i8],
    page: &mut [u32],
    on_chunk: &mut dyn FnMut(&[f32], usize),
) -> Result<(), Error> {
    let cfg = *model.config();
    let need = scratch_len(model, frames * HOP);
    if fs.len() < need.f32s || is.len() < need.i8s || page.len() < need.page_words {
        return Err(Error::ScratchTooSmall);
    }
    if feats.len() < frames * cfg.n_mels {
        return Err(Error::Shape);
    }
    let mut r = &mut fs[..];
    let chunk_buf = carve(&mut r, cfg.chunk_size * cfg.n_mels);
    let ecache = carve(&mut r, embed_state_len(&cfg));
    let state = carve(&mut r, zipformer::state_len(&cfg));
    let x = carve(&mut r, cfg.frames() * cfg.embed_dim);
    let enc = carve(&mut r, (cfg.frames() / 2 + 1) * cfg.enc_out);
    let work = r;
    let xq = &mut is[..need.i8s];
    ecache.fill(0.0);
    state.fill(0.0);
    let (mut processed, mut offset) = (0usize, 0usize);
    while offset < frames {
        chunk_buf.fill(0.0);
        let avail = cfg.chunk_size.min(frames - offset);
        chunk_buf[..avail * cfg.n_mels]
            .copy_from_slice(&feats[offset * cfg.n_mels..(offset + avail) * cfg.n_mels]);
        embed(model, chunk_buf, ecache, work, xq, x);
        let to = zipformer::chunk(model, pager, page, state, processed, x, work, xq, enc)?;
        on_chunk(&enc[..to * cfg.enc_out], to);
        processed += cfg.frames();
        offset += cfg.chunk_shift;
    }
    Ok(())
}

/// Stateless predictor over `context` (−1 = none) → `out[dec_hidden]`.
fn predictor(
    model: &Model<'_>,
    context: [i32; 2],
    emb: &mut [f32],
    conv: &mut [f32],
    out: &mut [f32],
    xq: &mut [i8],
) {
    let dh = model.config().dec_hidden;
    for (p, &tok) in context.iter().enumerate() {
        let row = &mut emb[p * dh..(p + 1) * dh];
        if tok < 0 {
            row.fill(0.0);
        } else {
            dequant_row(&model.dec_embed, tok as usize, row);
        }
    }
    let per = model.dec_conv.len() / (dh * 2);
    for (o, cv) in conv[..dh].iter_mut().enumerate() {
        let start = (o / per) * per;
        let mut acc = 0f32;
        for i in 0..per {
            for p in 0..2 {
                acc += emb[p * dh + start + i] * model.dec_conv[(o * per + i) * 2 + p];
            }
        }
        *cv = acc.max(0.0);
    }
    linear(
        &conv[..dh],
        1,
        &model.dec_proj.0,
        Some(model.dec_proj.1),
        out,
        xq,
    );
}

/// PCM → token ids (greedy transducer, one token per frame at most).
#[allow(clippy::too_many_arguments)]
pub fn transcribe<P: Pcm + ?Sized, G: Pager + ?Sized>(
    model: &Model<'_>,
    pager: &mut G,
    pcm: &P,
    fs: &mut [f32],
    is: &mut [i8],
    page: &mut [u32],
    ids: &mut [u16],
) -> Result<usize, Error> {
    run(model, pager, pcm, fs, is, page, ids, &mut |_, _| {})
}

/// [`transcribe`], calling `on_chunk(encoder_out, frames)` after every
/// chunk. For parity checks against the reference graph.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn transcribe_with<P: Pcm + ?Sized, G: Pager + ?Sized>(
    model: &Model<'_>,
    pager: &mut G,
    pcm: &P,
    fs: &mut [f32],
    is: &mut [i8],
    page: &mut [u32],
    ids: &mut [u16],
    on_chunk: &mut dyn FnMut(&[f32], usize),
) -> Result<usize, Error> {
    run(model, pager, pcm, fs, is, page, ids, on_chunk)
}

#[allow(clippy::too_many_arguments)]
fn run<P: Pcm + ?Sized, G: Pager + ?Sized>(
    model: &Model<'_>,
    pager: &mut G,
    pcm: &P,
    fs: &mut [f32],
    is: &mut [i8],
    page: &mut [u32],
    ids: &mut [u16],
    on_chunk: &mut dyn FnMut(&[f32], usize),
) -> Result<usize, Error> {
    let cfg = *model.config();
    let need = scratch_len(model, pcm.len());
    if fs.len() < need.f32s || is.len() < need.i8s || page.len() < need.page_words {
        return Err(Error::ScratchTooSmall);
    }
    let fe = Frontend::new(NFFT);
    let mut r = &mut fs[..];
    let fb = carve(&mut r, fbank_frames(pcm.len()) * cfg.n_mels);
    let fft = carve(&mut r, 2 * NFFT);
    let chunk_buf = carve(&mut r, cfg.chunk_size * cfg.n_mels);
    let ecache = carve(&mut r, embed_state_len(&cfg));
    let state = carve(&mut r, zipformer::state_len(&cfg));
    let x = carve(&mut r, cfg.frames() * cfg.embed_dim);
    let enc = carve(&mut r, (cfg.frames() / 2 + 1) * cfg.enc_out);
    let emb = carve(&mut r, 2 * cfg.dec_hidden);
    let dconv = carve(&mut r, cfg.dec_hidden);
    let logits = carve(&mut r, cfg.vocab);
    let work = r;
    let dec = &mut [0f32; 1024][..cfg.dec_hidden.min(1024)];
    if cfg.dec_hidden > 1024 || cfg.enc_out != cfg.dec_hidden {
        return Err(Error::Unsupported);
    }
    let xq = &mut is[..need.i8s];
    ecache.fill(0.0);
    state.fill(0.0);

    let frames = fbank(model, &fe, pcm, fb, fft);
    let mut context = [-1i32, cfg.blank as i32];
    predictor(model, context, emb, dconv, dec, xq);
    let (mut n, mut processed, mut offset) = (0usize, 0usize, 0usize);
    while offset < frames {
        chunk_buf.fill(0.0);
        let avail = cfg.chunk_size.min(frames - offset);
        chunk_buf[..avail * cfg.n_mels]
            .copy_from_slice(&fb[offset * cfg.n_mels..(offset + avail) * cfg.n_mels]);
        embed(model, chunk_buf, ecache, work, xq, x);
        let to = zipformer::chunk(model, pager, page, state, processed, x, work, xq, enc)?;
        on_chunk(&enc[..to * cfg.enc_out], to);
        let consumed = cfg.chunk_shift.min(frames - offset);
        let valid = to.min(consumed.div_ceil(4));
        for t in 0..valid {
            let er = &enc[t * cfg.enc_out..(t + 1) * cfg.enc_out];
            for (h, (&ev, &dv)) in er.iter().zip(dec.iter()).enumerate() {
                emb[h] = libm::tanhf(ev + dv);
            }
            linear(
                &emb[..cfg.enc_out],
                1,
                &model.joiner.0,
                Some(model.joiner.1),
                logits,
                xq,
            );
            let mut best = 0;
            for k in 1..cfg.vocab {
                if logits[k] > logits[best] {
                    best = k;
                }
            }
            if best == cfg.blank || best == cfg.unk {
                continue;
            }
            if n == ids.len() {
                return Err(Error::OutputTooSmall);
            }
            ids[n] = best as u16;
            n += 1;
            context = [context[1], best as i32];
            predictor(model, context, emb, dconv, dec, xq);
        }
        processed += cfg.frames();
        offset += cfg.chunk_shift;
    }
    Ok(n)
}
