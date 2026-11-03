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

//! The streaming Zipformer2 encoder, one chunk at a time.
//!
//! Six stacks at different frame rates (downsample, layers, upsample, blend
//! with the stack input); every layer keeps six caches — attention keys,
//! non-linear-attention values, two value caches and two conv tails — that
//! carry the left context from chunk to chunk. Layer weights are paged in
//! one layer at a time.

use rlx_mcu::Error;
use rlx_mcu::kernels::{carve, linear, sigmoid, softmax};

use crate::model::{Config, Layer, Model, Pager, Stack};
use crate::{swoosh_l, swoosh_r};

/// Additive mask of a not-yet-filled left-context frame.
const MASKED: f32 = -1000.0;

/// f32 state of one layer: `(key, nonlin, val1, val2, conv1, conv2)` lengths.
pub(crate) fn layer_state_lens(s: &Stack) -> [usize; 6] {
    let l = s.left;
    let hv = s.heads * s.vdim;
    [
        l * s.heads * s.qdim,
        l * s.nonlin(),
        l * hv,
        l * hv,
        (s.kernel / 2) * s.dim,
        (s.kernel / 2) * s.dim,
    ]
}

/// f32 state of the whole encoder.
pub(crate) fn state_len(cfg: &Config) -> usize {
    cfg.stacks[..cfg.n_stacks]
        .iter()
        .map(|s| s.layers * layer_state_lens(s).iter().sum::<usize>())
        .sum()
}

fn stack_frames(cfg: &Config, s: &Stack) -> usize {
    cfg.frames().div_ceil(s.down)
}

/// f32 scratch of one chunk.
pub(crate) fn scratch_len(cfg: &Config) -> usize {
    let t0 = cfg.frames();
    let md = cfg.max_dim();
    let mut layer = 0;
    for s in &cfg.stacks[..cfg.n_stacks] {
        let t = stack_frames(cfg, s);
        let (l, h, d) = (s.left, s.heads, s.dim);
        let w_attn = 2 * h * s.qdim + h * cfg.pos_head_dim;
        let ffm =
            s.ff.iter()
                .copied()
                .max()
                .unwrap_or(0)
                .max(3 * s.nonlin())
                .max(2 * d)
                .max(w_attn);
        let need = t * d // orig
            + t * ffm // proj / ff hidden
            + t * d.max(h * s.vdim).max(s.nonlin()) // mixed
            + t * d // out
            + h * t * (l + t) // attention weights
            + (l + 2 * t - 1) * (cfg.pos_dim + h * cfg.pos_head_dim) // positions
            + (l + t) * (h * s.qdim).max(s.nonlin()).max(h * s.vdim).max(d) // concat
            + (s.kernel / 2 + t) * d // conv concat
            + t * d; // conv chunk branch
        layer = layer.max(need);
    }
    3 * t0 * md + layer
}

/// Compact relative positions (k2 `CompactRelPositionalEncoding`): rows for
/// offsets `-(left+t-1) ..= t-1`.
fn rel_positions(t: usize, left: usize, pos_dim: usize, out: &mut [f32]) {
    let len = left + 2 * t - 1;
    let compression = libm::sqrtf(pos_dim as f32);
    let length_scale = pos_dim as f32 / (2.0 * core::f32::consts::PI);
    for row in 0..len {
        let x = row as f32 - (left + t - 1) as f32;
        let sign = if x < 0.0 {
            -1.0
        } else if x > 0.0 {
            1.0
        } else {
            0.0
        };
        let compressed = compression
            * sign
            * (libm::logf(libm::fabsf(x) + compression) - libm::logf(compression));
        let angle = libm::atanf(compressed / length_scale);
        let r = &mut out[row * pos_dim..(row + 1) * pos_dim];
        for d in 0..pos_dim / 2 {
            let f = (d + 1) as f32;
            r[2 * d] = libm::cosf(angle * f);
            r[2 * d + 1] = libm::sinf(angle * f);
        }
        r[pos_dim - 1] = 1.0;
    }
}

/// `concat(state, new)` into `cat`, and its last `state.len()` values back
/// into `state` (the new left context).
fn roll(state: &mut [f32], new: &[f32], cat: &mut [f32]) {
    let (sl, nl) = (state.len(), new.len());
    cat[..sl].copy_from_slice(state);
    cat[sl..sl + nl].copy_from_slice(new);
    state.copy_from_slice(&cat[nl..nl + sl]);
}

/// `x += ff(x)`: in_proj → SwooshL → out_proj.
fn ff(
    x: &mut [f32],
    t: usize,
    d: usize,
    w: &(
        (rlx_mcu::blob::Mat<'_>, &[f32]),
        (rlx_mcu::blob::Mat<'_>, &[f32]),
    ),
    hid: &mut [f32],
    out: &mut [f32],
    xq: &mut [i8],
) {
    let wd = w.0.0.rows();
    linear(x, t, &w.0.0, Some(w.0.1), hid, xq);
    for v in &mut hid[..t * wd] {
        *v = swoosh_l(*v);
    }
    linear(hid, t, &w.1.0, Some(w.1.1), out, xq);
    for i in 0..t * d {
        x[i] += out[i];
    }
}

/// `orig + (x - orig) · scale`, in place in `x`.
fn bypass(x: &mut [f32], orig: &[f32], t: usize, d: usize, scale: &[f32]) {
    for r in 0..t {
        for c in 0..d {
            let i = r * d + c;
            x[i] = orig[i] + (x[i] - orig[i]) * scale[c];
        }
    }
}

struct Bufs<'s> {
    orig: &'s mut [f32],
    proj: &'s mut [f32],
    mixed: &'s mut [f32],
    out: &'s mut [f32],
    attn: &'s mut [f32],
    pos: &'s mut [f32],
    cat: &'s mut [f32],
    ccat: &'s mut [f32],
    chunk: &'s mut [f32],
}

/// One Zipformer layer over `x[t, d]`, in place.
#[allow(clippy::too_many_arguments)]
fn layer(
    cfg: &Config,
    s: &Stack,
    w: &Layer<'_>,
    x: &mut [f32],
    t: usize,
    state: &mut [f32],
    valid_left: usize,
    b: &mut Bufs<'_>,
    xq: &mut [i8],
) {
    let (d, l, h, q, pd) = (s.dim, s.left, s.heads, s.qdim, cfg.pos_head_dim);
    let [kl, nl, vl, _, cl, _] = layer_state_lens(s);
    let (key_st, rest) = state.split_at_mut(kl);
    let (nonlin_st, rest) = rest.split_at_mut(nl);
    let (val1_st, rest) = rest.split_at_mut(vl);
    let (val2_st, rest) = rest.split_at_mut(vl);
    let (conv1_st, conv2_st) = rest.split_at_mut(cl);
    b.orig[..t * d].copy_from_slice(&x[..t * d]);
    let src = l + t;

    // Attention weights, shared by the non-linear attention (head 0) and
    // both self-attentions.
    let wa = 2 * h * q + h * pd;
    linear(x, t, &w.attn_in.0, Some(w.attn_in.1), b.proj, xq);
    for r in 0..t {
        b.mixed[r * h * q..(r + 1) * h * q]
            .copy_from_slice(&b.proj[r * wa + h * q..r * wa + 2 * h * q]);
    }
    roll(key_st, &b.mixed[..t * h * q], b.cat);
    let np = l + 2 * t - 1;
    let (ptab, pproj) = b.pos.split_at_mut(np * cfg.pos_dim);
    rel_positions(t, l, cfg.pos_dim, ptab);
    for r in 0..np {
        for o in 0..h * pd {
            let mut acc = 0f32;
            for p in 0..cfg.pos_dim {
                acc += ptab[r * cfg.pos_dim + p] * w.linear_pos[o * cfg.pos_dim + p];
            }
            pproj[r * h * pd + o] = acc;
        }
    }
    let masked = l - valid_left.min(l);
    for hh in 0..h {
        for tt in 0..t {
            let qrow = &b.proj[tt * wa + hh * q..tt * wa + hh * q + q];
            let prow = &b.proj[tt * wa + 2 * h * q + hh * pd..tt * wa + 2 * h * q + hh * pd + pd];
            let row = &mut b.attn[(hh * t + tt) * src..(hh * t + tt + 1) * src];
            for (j, sc) in row.iter_mut().enumerate() {
                let krow = &b.cat[j * h * q + hh * q..j * h * q + hh * q + q];
                let mut acc = 0f32;
                for i in 0..q {
                    acc += qrow[i] * krow[i];
                }
                let rel = (t - 1 - tt) + j;
                let pr = &pproj[rel * h * pd + hh * pd..rel * h * pd + hh * pd + pd];
                for i in 0..pd {
                    acc += prow[i] * pr[i];
                }
                *sc = acc + if j < masked { MASKED } else { 0.0 };
            }
            softmax(row);
        }
    }

    ff(x, t, d, &w.ff[0], b.proj, b.out, xq);

    // Non-linear attention.
    let nh = s.nonlin();
    linear(x, t, &w.nonlin_in.0, Some(w.nonlin_in.1), b.proj, xq);
    for r in 0..t {
        for c in 0..nh {
            let p = &b.proj[r * 3 * nh..(r + 1) * 3 * nh];
            b.mixed[r * nh + c] = p[nh + c] * libm::tanhf(p[c]);
        }
    }
    roll(nonlin_st, &b.mixed[..t * nh], b.cat);
    for tt in 0..t {
        let a = &b.attn[tt * src..(tt + 1) * src];
        for c in 0..nh {
            let mut acc = 0f32;
            for (j, &aw) in a.iter().enumerate() {
                acc += aw * b.cat[j * nh + c];
            }
            b.mixed[tt * nh + c] = acc * b.proj[tt * 3 * nh + 2 * nh + c];
        }
    }
    linear(b.mixed, t, &w.nonlin_out.0, Some(w.nonlin_out.1), b.out, xq);
    for i in 0..t * d {
        x[i] += b.out[i];
    }

    self_attn(x, t, d, s, &w.sa[0], val1_st, b, xq);
    conv(x, t, d, s.kernel, &w.conv[0], conv1_st, b, xq);
    ff(x, t, d, &w.ff[1], b.proj, b.out, xq);
    bypass(x, b.orig, t, d, w.bypass_mid);
    self_attn(x, t, d, s, &w.sa[1], val2_st, b, xq);
    conv(x, t, d, s.kernel, &w.conv[1], conv2_st, b, xq);
    ff(x, t, d, &w.ff[2], b.proj, b.out, xq);

    // BiasNorm: x / rms(x - bias) · e^log_scale.
    let g = libm::expf(w.norm_log_scale);
    for r in 0..t {
        let xr = &mut x[r * d..(r + 1) * d];
        let mut m = 0f32;
        for (c, &v) in xr.iter().enumerate() {
            let z = v - w.norm_bias[c];
            m += z * z;
        }
        let inv = g / libm::sqrtf(m / d as f32);
        for v in xr.iter_mut() {
            *v *= inv;
        }
    }
    bypass(x, b.orig, t, d, w.bypass);
}

/// `x += out_proj(Σ_j attn[h] · cat(values)[j])` per head.
#[allow(clippy::too_many_arguments)]
fn self_attn(
    x: &mut [f32],
    t: usize,
    d: usize,
    s: &Stack,
    w: &(
        (rlx_mcu::blob::Mat<'_>, &[f32]),
        (rlx_mcu::blob::Mat<'_>, &[f32]),
    ),
    st: &mut [f32],
    b: &mut Bufs<'_>,
    xq: &mut [i8],
) {
    let (h, v) = (s.heads, s.vdim);
    let hv = h * v;
    let src = s.left + t;
    linear(x, t, &w.0.0, Some(w.0.1), b.proj, xq);
    roll(st, &b.proj[..t * hv], b.cat);
    for hh in 0..h {
        for tt in 0..t {
            let a = &b.attn[(hh * t + tt) * src..(hh * t + tt + 1) * src];
            for i in 0..v {
                let mut acc = 0f32;
                for (j, &aw) in a.iter().enumerate() {
                    acc += aw * b.cat[j * hv + hh * v + i];
                }
                b.mixed[tt * hv + hh * v + i] = acc;
            }
        }
    }
    linear(b.mixed, t, &w.1.0, Some(w.1.1), b.out, xq);
    for i in 0..t * d {
        x[i] += b.out[i];
    }
}

/// Zipformer conv module: GLU, then a causal depthwise conv over the cached
/// tail plus a chunk-local one with learned edge scales, SwooshR, out_proj.
#[allow(clippy::too_many_arguments)]
fn conv(
    x: &mut [f32],
    t: usize,
    d: usize,
    k: usize,
    w: &crate::model::Conv<'_>,
    st: &mut [f32],
    b: &mut Bufs<'_>,
    xq: &mut [i8],
) {
    let half = k / 2;
    linear(x, t, &w.in_proj.0, Some(w.in_proj.1), b.proj, xq);
    for r in 0..t {
        for c in 0..d {
            b.mixed[r * d + c] = b.proj[r * 2 * d + c] * sigmoid(b.proj[r * 2 * d + d + c]);
        }
    }
    roll(st, &b.mixed[..t * d], b.ccat);
    let (cw, cb) = w.causal;
    let (kw, kb) = w.chunk;
    for tt in 0..t {
        for c in 0..d {
            let mut y = cb[c];
            for j in 0..=half {
                y += cw[c * (half + 1) + j] * b.ccat[(tt + j) * d + c];
            }
            let mut z = kb[c];
            for j in 0..k {
                let si = tt + j;
                if si < half || si - half >= t {
                    continue;
                }
                z += kw[c * k + j] * b.mixed[(si - half) * d + c];
            }
            let mut edge = 0f32;
            if t < k {
                edge += w.scale_left[c * k + tt] + w.scale_right[c * k + k - t + tt];
            } else {
                if tt < k {
                    edge += w.scale_left[c * k + tt];
                }
                if tt >= t - k {
                    edge += w.scale_right[c * k + tt - (t - k)];
                }
            }
            b.chunk[tt * d + c] = swoosh_r(y + z * (1.0 + edge));
        }
    }
    linear(b.chunk, t, &w.out_proj.0, Some(w.out_proj.1), b.out, xq);
    for i in 0..t * d {
        x[i] += b.out[i];
    }
}

/// One chunk: `x[frames, embed_dim]` → `out[frames/2, enc_out]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chunk<G: Pager + ?Sized>(
    model: &Model<'_>,
    pager: &mut G,
    page: &mut [u32],
    state: &mut [f32],
    processed: usize,
    x: &[f32],
    scratch: &mut [f32],
    xq: &mut [i8],
    out: &mut [f32],
) -> Result<usize, Error> {
    let cfg = model.config();
    let t0 = cfg.frames();
    let md = cfg.max_dim();
    let mut sr = &mut scratch[..];
    let cur = carve(&mut sr, t0 * md);
    let sin = carve(&mut sr, t0 * md);
    let full = carve(&mut sr, t0 * md);
    let mut lr = &mut sr[..];
    let mut bufs = Bufs {
        orig: &mut [],
        proj: &mut [],
        mixed: &mut [],
        out: &mut [],
        attn: &mut [],
        pos: &mut [],
        cat: &mut [],
        ccat: &mut [],
        chunk: &mut [],
    };
    {
        // Size every buffer for the widest stack once.
        let (mut o, mut p, mut m, mut ou, mut a, mut po, mut c, mut cc, mut ch) =
            (0, 0, 0, 0, 0, 0, 0, 0, 0);
        for s in &cfg.stacks[..cfg.n_stacks] {
            let t = stack_frames(cfg, s);
            let (l, h, d) = (s.left, s.heads, s.dim);
            let w_attn = 2 * h * s.qdim + h * cfg.pos_head_dim;
            let ffm =
                s.ff.iter()
                    .copied()
                    .max()
                    .unwrap_or(0)
                    .max(3 * s.nonlin())
                    .max(2 * d)
                    .max(w_attn);
            o = o.max(t * d);
            p = p.max(t * ffm);
            m = m.max(t * d.max(h * s.vdim).max(s.nonlin()).max(h * s.qdim));
            ou = ou.max(t * d);
            a = a.max(h * t * (l + t));
            po = po.max((l + 2 * t - 1) * (cfg.pos_dim + h * cfg.pos_head_dim));
            c = c.max((l + t) * (h * s.qdim).max(s.nonlin()).max(h * s.vdim).max(d));
            cc = cc.max((s.kernel / 2 + t) * d);
            ch = ch.max(t * d);
        }
        bufs.orig = carve(&mut lr, o);
        bufs.proj = carve(&mut lr, p);
        bufs.mixed = carve(&mut lr, m);
        bufs.out = carve(&mut lr, ou);
        bufs.attn = carve(&mut lr, a);
        bufs.pos = carve(&mut lr, po);
        bufs.cat = carve(&mut lr, c);
        bufs.ccat = carve(&mut lr, cc);
        bufs.chunk = carve(&mut lr, ch);
    }

    // Columns of the final concat each stack contributes.
    let mut cols = [(0usize, 0usize); crate::model::MAX_STACKS];
    let n = cfg.n_stacks;
    let mut cur_dim = cfg.stacks[n - 1].dim;
    cols[n - 1] = (0, cur_dim);
    for s in (0..n - 1).rev() {
        let dim = cfg.stacks[s].dim;
        if dim > cur_dim {
            cols[s] = (cur_dim, dim);
            cur_dim = dim;
        }
    }

    let mut prev_dim = cfg.embed_dim;
    cur[..t0 * prev_dim].copy_from_slice(&x[..t0 * prev_dim]);
    let mut state_off = 0;
    let mut layer_idx = 0;
    for (si, s) in cfg.stacks[..n].iter().enumerate() {
        let d = s.dim;
        // Convert channels (truncate or zero-pad) into the stack input.
        for r in 0..t0 {
            for c in 0..d {
                sin[r * d + c] = if c < prev_dim {
                    cur[r * prev_dim + c]
                } else {
                    0.0
                };
            }
        }
        let t = stack_frames(cfg, s);
        if s.down > 1 {
            let wd = model.down_w[si];
            for r in 0..t {
                for c in 0..d {
                    let mut acc = 0f32;
                    for (i, &wi) in wd.iter().enumerate() {
                        let src = (r * s.down + i).min(t0 - 1);
                        acc += wi * sin[src * d + c];
                    }
                    cur[r * d + c] = acc;
                }
            }
        } else {
            cur[..t0 * d].copy_from_slice(&sin[..t0 * d]);
        }
        let valid_left = processed / s.down;
        let lens = layer_state_lens(s);
        let per_layer: usize = lens.iter().sum();
        for _ in 0..s.layers {
            let (off, len) = model.layer_block(layer_idx);
            // SAFETY: u32 page viewed as bytes, within its length.
            let bytes = unsafe {
                core::slice::from_raw_parts_mut(page.as_mut_ptr().cast::<u8>(), page.len() * 4)
            };
            let bytes = bytes.get_mut(..len).ok_or(Error::ScratchTooSmall)?;
            pager.read(off, bytes)?;
            let w = Layer::parse(bytes, s, cfg.pos_dim, cfg.pos_head_dim)?;
            let st = &mut state[state_off..state_off + per_layer];
            layer(cfg, s, &w, cur, t, st, valid_left, &mut bufs, xq);
            state_off += per_layer;
            layer_idx += 1;
        }
        if s.down > 1 {
            // Upsample by repetition, then blend with the stack input.
            // Blend in place in `sin`: each element reads only itself and the
            // downsampled frame it repeats.
            let comb = model.combiner[si];
            for r in 0..t0 {
                for c in 0..d {
                    let up = cur[(r / s.down) * d + c];
                    let o = &mut sin[r * d + c];
                    *o += (up - *o) * comb[c];
                }
            }
            cur[..t0 * d].copy_from_slice(&sin[..t0 * d]);
        }
        let (lo, hi) = cols[si];
        if hi > lo {
            for r in 0..t0 {
                for c in lo..hi {
                    full[r * md + c] = cur[r * d + c];
                }
            }
        }
        prev_dim = d;
    }
    // `full` now holds the final concat; downsample it by 2 and project.
    let fd = cur_dim;
    let df = model.down_out.len().max(1);
    let to = t0.div_ceil(df);
    for r in 0..to {
        for c in 0..fd {
            let mut acc = 0f32;
            for (i, &wi) in model.down_out.iter().enumerate() {
                let src = (r * df + i).min(t0 - 1);
                acc += wi * full[src * md + c];
            }
            sin[r * fd + c] = acc;
        }
    }
    linear(sin, to, &model.enc_proj.0, Some(model.enc_proj.1), out, xq);
    Ok(to)
}
