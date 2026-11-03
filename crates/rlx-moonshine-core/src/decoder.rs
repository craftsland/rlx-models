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

//! The Moonshine decoder, one token at a time.
//!
//! Cross-attention K/V are computed once per utterance and kept int8 (one
//! scale per frame and head for K, per channel for V). Self-attention keeps
//! an f32 KV cache, which is small: a few dozen tokens per utterance. The LM
//! head is an argmax over its rows, so the 32k-wide logits row never exists.

use rlx_mcu::Error;
use rlx_mcu::blob::{quant, quantize_row};
use rlx_mcu::kernels::{
    argmax_linear, carve, dequant_row, dot_i8, layer_norm, linear, silu, softmax,
};

use crate::model::{Config, Model};

const LN_EPS: f32 = 1e-5;

/// Persistent f32 scratch: cross-attention scales.
pub(crate) fn cross_f32_len(cfg: &Config, t: usize) -> usize {
    cfg.dec_layers * (t * cfg.dec_heads + cfg.dec_attn())
}

/// Persistent i8 scratch: cross-attention K and Vᵀ.
pub(crate) fn cross_i8_len(cfg: &Config, t: usize) -> usize {
    cfg.dec_layers * 2 * t * cfg.dec_attn()
}

/// Temporary f32 scratch while building the cross caches.
pub(crate) fn cross_build_len(cfg: &Config, t: usize) -> usize {
    2 * t * cfg.dec_attn() + if cfg.has_proj { t * cfg.d_dec } else { 0 }
}

/// f32 scratch of the token loop.
pub(crate) fn f32_len(cfg: &Config, t: usize, m: usize) -> usize {
    let (d, a, ff) = (cfg.d_dec, cfg.dec_attn(), cfg.dec_ff);
    2 * cfg.dec_layers * m * a + 2 * d + 4 * a + 2 * ff + ff + t.max(m)
}

/// GPT-J rotary on one head (adjacent pairs, first `n_rot` dims).
fn rope(x: &mut [f32], pos: usize, n_rot: usize, theta: f64) {
    for i in 0..n_rot / 2 {
        let freq = 1.0 / libm::pow(theta, (2 * i) as f64 / n_rot as f64);
        let ang = pos as f64 * freq;
        let (s, c) = (libm::sin(ang) as f32, libm::cos(ang) as f32);
        let (a, b) = (x[2 * i], x[2 * i + 1]);
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

/// Decode greedily. `x[..T·d]` holds the encoder output on entry.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    model: &Model<'_>,
    n: usize,
    t: usize,
    x: &mut [f32],
    rest: &mut [f32],
    xq: &mut [i8],
    irest: &mut [i8],
    ids: &mut [u16],
) -> Result<usize, Error> {
    let cfg = *model.config();
    let (de, d, a, nh, hd) = (
        cfg.d_enc,
        cfg.d_dec,
        cfg.dec_attn(),
        cfg.dec_heads,
        cfg.dec_head_dim,
    );
    if t > cfg.max_pos {
        return Err(Error::Unsupported);
    }

    // Decoder-side states: hidden + learned positions, then the projection.
    let mut pos_row = [0f32; 1024];
    if de > pos_row.len() {
        return Err(Error::Unsupported);
    }
    for r in 0..t {
        dequant_row(&model.pos_emb, r, &mut pos_row[..de]);
        for (v, &p) in x[r * de..(r + 1) * de].iter_mut().zip(&pos_row[..de]) {
            *v += p;
        }
    }

    let mut rr = &mut rest[..];
    let cross_scales = carve(&mut rr, cross_f32_len(&cfg, t));
    let work = rr;
    if let Some(p) = &model.proj {
        let tmp = &mut work[..t * d];
        linear(&x[..t * de], t, p, None, tmp, xq);
        x[..t * d].copy_from_slice(tmp);
    }

    // Cross-attention caches.
    let mut ir = &mut irest[..];
    let cross_q8 = carve(&mut ir, cross_i8_len(&cfg, t));
    {
        let (kb, vb) = work[..2 * t * a].split_at_mut(t * a);
        let mut cs = &mut cross_scales[..];
        let mut cq = &mut cross_q8[..];
        for l in 0..cfg.dec_layers {
            let ly = model.dec_layer(l);
            let ks = carve(&mut cs, t * nh);
            let vs = carve(&mut cs, a);
            let kq = carve(&mut cq, t * a);
            let vt = carve(&mut cq, a * t);
            linear(&x[..t * d], t, &ly.ca.k, None, kb, xq);
            linear(&x[..t * d], t, &ly.ca.v, None, vb, xq);
            for j in 0..t {
                for h in 0..nh {
                    let off = j * a + h * hd;
                    ks[j * nh + h] = quantize_row(&kb[off..off + hd], &mut kq[off..off + hd]);
                }
            }
            for c in 0..a {
                let mut m = 0f32;
                for j in 0..t {
                    m = m.max(libm::fabsf(vb[j * a + c]));
                }
                vs[c] = m / 127.0;
                let inv = if m == 0.0 { 0.0 } else { 127.0 / m };
                for j in 0..t {
                    vt[c * t + j] = quant(vb[j * a + c] * inv);
                }
            }
        }
    }

    // Token loop.
    let limit = cfg.max_tokens(n);
    let mut wr = &mut work[..];
    let kc = carve(&mut wr, cfg.dec_layers * limit * a);
    let vc = carve(&mut wr, cfg.dec_layers * limit * a);
    let xs = carve(&mut wr, d);
    let hs = carve(&mut wr, d);
    let q = carve(&mut wr, a);
    let k = carve(&mut wr, a);
    let v = carve(&mut wr, a);
    let ctx = carve(&mut wr, a);
    let mid = carve(&mut wr, 2 * cfg.dec_ff);
    let act = carve(&mut wr, cfg.dec_ff);
    let scores = carve(&mut wr, t.max(limit));
    let mut qq = [0i8; 128];
    let mut probq_store = &mut ir[..];
    let probq = carve(&mut probq_store, t);

    let scale = 1.0 / libm::sqrtf(hd as f32);
    let theta = cfg.rope_theta as f64;
    let ff = cfg.dec_ff;
    let mut tok = cfg.start;
    let mut out = 0usize;
    let mut pos = 0usize;
    while pos + 1 < limit {
        dequant_row(&model.embed_tokens, tok, xs);
        let mut cs = &cross_scales[..];
        let mut cq = &cross_q8[..];
        for l in 0..cfg.dec_layers {
            let ly = model.dec_layer(l);
            // Causal self-attention with RoPE.
            layer_norm(xs, 1, d, ly.ln_in, None, LN_EPS, hs);
            linear(hs, 1, &ly.sa.q, None, q, xq);
            linear(hs, 1, &ly.sa.k, None, k, xq);
            linear(hs, 1, &ly.sa.v, None, v, xq);
            for h in 0..nh {
                rope(&mut q[h * hd..(h + 1) * hd], pos, cfg.n_rot, theta);
                rope(&mut k[h * hd..(h + 1) * hd], pos, cfg.n_rot, theta);
            }
            let base = l * limit * a;
            kc[base + pos * a..base + (pos + 1) * a].copy_from_slice(k);
            vc[base + pos * a..base + (pos + 1) * a].copy_from_slice(v);
            for h in 0..nh {
                let off = h * hd;
                let row = &mut scores[..=pos];
                for (j, s) in row.iter_mut().enumerate() {
                    let kr = &kc[base + j * a + off..base + j * a + off + hd];
                    let mut dot = 0f32;
                    for c in 0..hd {
                        dot += q[off + c] * kr[c];
                    }
                    *s = dot * scale;
                }
                softmax(row);
                for c in 0..hd {
                    let mut acc = 0f32;
                    for (j, &p) in row.iter().enumerate() {
                        acc += p * vc[base + j * a + off + c];
                    }
                    ctx[off + c] = acc;
                }
            }
            linear(ctx, 1, &ly.sa.o, None, hs, xq);
            for (x, &o) in xs.iter_mut().zip(hs.iter()) {
                *x += o;
            }

            // Cross-attention over the encoder states, int8.
            let ks = &cs[..t * nh];
            let vs = &cs[t * nh..t * nh + a];
            cs = &cs[t * nh + a..];
            let kq = &cq[..t * a];
            let vt = &cq[t * a..2 * t * a];
            cq = &cq[2 * t * a..];
            layer_norm(xs, 1, d, ly.ln_post, None, LN_EPS, hs);
            linear(hs, 1, &ly.ca.q, None, q, xq);
            for h in 0..nh {
                let off = h * hd;
                let sq = quantize_row(&q[off..off + hd], &mut qq[..hd]);
                let row = &mut scores[..t];
                for (j, s) in row.iter_mut().enumerate() {
                    let o = j * a + off;
                    *s = dot_i8(&qq[..hd], &kq[o..o + hd]) as f32 * (sq * ks[j * nh + h]) * scale;
                }
                softmax(row);
                let sp = quantize_row(row, probq);
                for c in 0..hd {
                    let ch = off + c;
                    let acc = dot_i8(&probq[..t], &vt[ch * t..(ch + 1) * t]);
                    ctx[ch] = acc as f32 * (sp * vs[ch]);
                }
            }
            linear(ctx, 1, &ly.ca.o, None, hs, xq);
            for (x, &o) in xs.iter_mut().zip(hs.iter()) {
                *x += o;
            }

            // SwiGLU MLP: fc1 → (up, gate) → silu(gate) · up → fc2.
            layer_norm(xs, 1, d, ly.ln_final, None, LN_EPS, hs);
            linear(hs, 1, &ly.fc1.0, Some(ly.fc1.1), mid, xq);
            for i in 0..ff {
                act[i] = mid[i] * silu(mid[ff + i]);
            }
            linear(act, 1, &ly.fc2.0, Some(ly.fc2.1), hs, xq);
            for (x, &o) in xs.iter_mut().zip(hs.iter()) {
                *x += o;
            }
        }
        layer_norm(xs, 1, d, model.dec_norm, None, LN_EPS, hs);
        let next = argmax_linear(hs, &model.lm_head, xq);
        pos += 1;
        if next == cfg.eos {
            break;
        }
        if out == ids.len() {
            return Err(Error::OutputTooSmall);
        }
        ids[out] = next as u16;
        out += 1;
        tok = next;
    }
    Ok(out)
}
