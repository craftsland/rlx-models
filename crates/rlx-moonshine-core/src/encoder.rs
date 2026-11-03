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

//! Frontend, the two causal convs, and the windowed encoder.
//!
//! The frontend and convs run row by row through small rings, so the 200 Hz
//! intermediate (`frames × d`, about 2.5 MB per 10 s in f32) never exists;
//! only the 50 Hz encoder states do. Attention is int8 and only visits the
//! `(left, right)` window, never a `T × T` matrix.

use rlx_mcu::Pcm;
use rlx_mcu::blob::{quant, quantize_row};
use rlx_mcu::kernels::{carve, dot_i8, gelu, layer_norm, linear, silu, softmax};

use crate::model::{CONV_K, CONV_STRIDE, Config, Model};

const LN_EPS: f32 = 1e-5;
const CMVN_EPS: f32 = 1e-6;
/// Additive bias of a masked key, as the reference graph uses (finite, so a
/// window with no live key averages instead of producing NaN).
const MASKED: f32 = -1e9;

fn width(cfg: &Config) -> usize {
    cfg.d_enc.max(cfg.enc_attn())
}

/// f32 scratch the encoder needs besides the `[T, d]` states.
pub(crate) fn f32_len(cfg: &Config, t: usize) -> usize {
    let d = cfg.d_enc;
    let stage = cfg.frame_len + CONV_K * d + CONV_K * 2 * d + CONV_K * 2 * d + 2 * d;
    let w = width(cfg);
    let layers =
        t * w * 2 + 2 * t * cfg.enc_attn() + 2 * cfg.enc_ff + 2 * d + t + t + cfg.enc_head_dim;
    stage.max(layers)
}

/// i8 scratch for encoder attention.
pub(crate) fn i8_len(cfg: &Config, t: usize) -> usize {
    let hd = cfg.enc_head_dim;
    t * hd + hd * t + t + hd
}

/// Rows of a prefix mask after one causal conv: output `t` is valid when any
/// input under its kernel was.
fn conv_valid(valid: usize, out_len: usize) -> usize {
    (0..out_len)
        .filter(|&t| (CONV_STRIDE * t).saturating_sub(CONV_K - 1) < valid)
        .count()
}

/// Run the encoder; leaves `final_norm(hidden)` in `x[..T·d]`. Returns how
/// many encoder frames hold real audio.
pub(crate) fn run<P: Pcm + ?Sized>(
    model: &Model<'_>,
    pcm: &P,
    stop_after: usize,
    x: &mut [f32],
    rest: &mut [f32],
    xq: &mut [i8],
    irest: &mut [i8],
) -> usize {
    let cfg = *model.config();
    let n = pcm.len();
    let (f, t1, t) = cfg.lengths(n);
    let v0 = n / cfg.frame_len;
    let v1 = conv_valid(v0, t1);
    let v2 = conv_valid(v1, t);
    stage(model, pcm, f, t1, t, (v0, v1, v2), x, rest, xq);
    for l in 0..cfg.enc_layers.min(stop_after) {
        layer(model, l, t, v2, x, rest, xq, irest);
    }
    if stop_after < cfg.enc_layers {
        return v2;
    }
    let d = cfg.d_enc;
    let h = &mut rest[..t * d];
    layer_norm(x, t, d, model.final_norm, None, LN_EPS, h);
    x[..t * d].copy_from_slice(h);
    v2
}

/// One frontend frame: per-frame CMVN then `asinh(k · x / rms)`.
fn frame<P: Pcm + ?Sized>(pcm: &P, idx: usize, k: f32, out: &mut [f32]) {
    let fl = out.len();
    let start = idx * fl;
    let mut mean = 0f32;
    for (j, o) in out.iter_mut().enumerate() {
        let s = start + j;
        *o = if s < pcm.len() { pcm.at(s) } else { 0.0 };
        mean += *o;
    }
    mean /= fl as f32;
    let mut var = 0f32;
    for o in out.iter_mut() {
        *o -= mean;
        var += *o * *o;
    }
    var /= fl as f32;
    let rms = libm::sqrtf(var + CMVN_EPS);
    for o in out.iter_mut() {
        *o = libm::asinhf(k * *o / rms);
    }
}

/// Frames → linear → SiLU → conv1 → SiLU → conv2, streamed: conv2 row `t`
/// needs conv1 rows `2t-4 ..= 2t`, each of which needs frontend rows
/// `2r-4 ..= 2r`. Both levels keep the last five rows in a ring.
#[allow(clippy::too_many_arguments)]
fn stage<P: Pcm + ?Sized>(
    model: &Model<'_>,
    pcm: &P,
    f: usize,
    t1: usize,
    t: usize,
    (v0, v1, v2): (usize, usize, usize),
    x: &mut [f32],
    rest: &mut [f32],
    xq: &mut [i8],
) {
    let cfg = model.config();
    let d = cfg.d_enc;
    let k = libm::expf(model.log_k);
    let mut r = &mut rest[..];
    let fbuf = carve(&mut r, cfg.frame_len);
    let ring0 = carve(&mut r, CONV_K * d);
    let ring1 = carve(&mut r, CONV_K * 2 * d);
    let patch = carve(&mut r, CONV_K * 2 * d);
    let row = carve(&mut r, 2 * d);
    let (mut have0, mut have1) = (0usize, 0usize); // rows computed so far

    for out in 0..t {
        // conv1 rows needed by conv2 row `out`.
        let need1 = (CONV_STRIDE * out + 1).min(t1);
        while have1 < need1 {
            let r1 = have1;
            let need0 = (CONV_STRIDE * r1 + 1).min(f);
            while have0 < need0 {
                let r0 = have0;
                frame(pcm, r0, k, fbuf);
                let dst = &mut ring0[(r0 % CONV_K) * d..(r0 % CONV_K + 1) * d];
                linear(fbuf, 1, &model.embed_lin, None, dst, xq);
                for v in dst.iter_mut() {
                    *v = if r0 < v0 { silu(*v) } else { 0.0 };
                }
                have0 += 1;
            }
            conv_row(ring0, d, r1, &model.conv1, patch, row, xq);
            let dst = &mut ring1[(r1 % CONV_K) * 2 * d..(r1 % CONV_K + 1) * 2 * d];
            for (o, &v) in dst.iter_mut().zip(row.iter()) {
                *o = if r1 < v1 { silu(v) } else { 0.0 };
            }
            have1 += 1;
        }
        conv_row(ring1, 2 * d, out, &model.conv2, patch, row, xq);
        let dst = &mut x[out * d..(out + 1) * d];
        for (o, &v) in dst.iter_mut().zip(row.iter()) {
            *o = if out < v2 { v } else { 0.0 };
        }
    }
}

/// Causal conv output `t`: inputs `2t-4 ..= 2t` (zero before 0) from a
/// 5-row ring of width `c_in`, weights `[out][tap][in]`.
fn conv_row(
    ring: &[f32],
    c_in: usize,
    t: usize,
    w: &(rlx_mcu::blob::Mat<'_>, &[f32]),
    patch: &mut [f32],
    row: &mut [f32],
    xq: &mut [i8],
) {
    for j in 0..CONV_K {
        let dst = &mut patch[j * c_in..(j + 1) * c_in];
        match (CONV_STRIDE * t + j).checked_sub(CONV_K - 1) {
            Some(src) => {
                dst.copy_from_slice(&ring[(src % CONV_K) * c_in..(src % CONV_K + 1) * c_in])
            }
            None => dst.fill(0.0),
        }
    }
    let out = w.0.rows();
    linear(
        &patch[..CONV_K * c_in],
        1,
        &w.0,
        Some(w.1),
        &mut row[..out],
        xq,
    );
}

/// `x += attn(LN(x))`, `x += MLP(LN(x))` for encoder layer `l`.
#[allow(clippy::too_many_arguments)]
fn layer(
    model: &Model<'_>,
    l: usize,
    t: usize,
    live: usize,
    x: &mut [f32],
    rest: &mut [f32],
    xq: &mut [i8],
    irest: &mut [i8],
) {
    let cfg = model.config();
    let (d, a, nh, hd, ff) = (
        cfg.d_enc,
        cfg.enc_attn(),
        cfg.enc_heads,
        cfg.enc_head_dim,
        cfg.enc_ff,
    );
    let w = width(cfg);
    let ly = model.enc_layer(l);
    let (left, right) = model.window(l);
    let mut r = &mut rest[..];
    let h = carve(&mut r, t * w);
    let q = carve(&mut r, t * w);
    let k = carve(&mut r, t * a);
    let v = carve(&mut r, t * a);
    let mid = carve(&mut r, 2 * ff);
    let out2 = carve(&mut r, 2 * d);
    let scores = carve(&mut r, t);
    let ks = carve(&mut r, t);
    let vs = carve(&mut r, hd);
    let mut ir = &mut irest[..];
    let kq = carve(&mut ir, t * hd);
    let vq = carve(&mut ir, hd * t);
    let probq = carve(&mut ir, t);
    let qq = carve(&mut ir, hd);

    layer_norm(x, t, d, ly.ln1, None, LN_EPS, h);
    linear(h, t, &ly.q, None, q, xq);
    linear(h, t, &ly.k, None, k, xq);
    linear(h, t, &ly.v, None, v, xq);
    let scale = 1.0 / libm::sqrtf(hd as f32);
    for head in 0..nh {
        let off = head * hd;
        for j in 0..t {
            ks[j] = quantize_row(&k[j * a + off..j * a + off + hd], &mut kq[j * hd..]);
        }
        for c in 0..hd {
            let mut m = 0f32;
            for j in 0..t {
                m = m.max(libm::fabsf(v[j * a + off + c]));
            }
            vs[c] = m / 127.0;
            let inv = if m == 0.0 { 0.0 } else { 127.0 / m };
            for j in 0..t {
                vq[c * t + j] = quant(v[j * a + off + c] * inv);
            }
        }
        for i in 0..t {
            // Keys `i-left+1 ..= i` and `i+1 ..= i+right-1`: the query's own
            // frame is always seen, even in a causal (`right = 0`) layer.
            let lo = (i + 1).saturating_sub(left);
            let hi = (i + right.max(1)).min(t); // exclusive
            let sq = quantize_row(&q[i * a + off..i * a + off + hd], qq);
            let row = &mut scores[..hi - lo];
            for (s, j) in row.iter_mut().zip(lo..hi) {
                let dot = dot_i8(qq, &kq[j * hd..(j + 1) * hd]) as f32 * (sq * ks[j]);
                *s = dot * scale + if j < live { 0.0 } else { MASKED };
            }
            softmax(row);
            let sp = quantize_row(row, probq);
            for c in 0..hd {
                let acc = dot_i8(&probq[..hi - lo], &vq[c * t + lo..c * t + hi]);
                h[i * a + off + c] = acc as f32 * (sp * vs[c]);
            }
        }
    }
    linear(h, t, &ly.o, None, q, xq);
    for i in 0..t * d {
        x[i] += q[i];
    }

    layer_norm(x, t, d, ly.ln2, None, LN_EPS, h);
    let mut i = 0;
    while i < t {
        let rows = (t - i).min(2);
        linear(
            &h[i * d..(i + rows) * d],
            rows,
            &ly.fc1.0,
            Some(ly.fc1.1),
            mid,
            xq,
        );
        for m in &mut mid[..rows * ff] {
            *m = gelu(*m);
        }
        linear(&mid[..rows * ff], rows, &ly.fc2.0, Some(ly.fc2.1), out2, xq);
        for (xv, &o) in x[i * d..(i + rows) * d].iter_mut().zip(out2.iter()) {
            *xv += o;
        }
        i += rows;
    }
}
