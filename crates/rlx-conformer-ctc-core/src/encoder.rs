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

//! The int8 forward pass: striding subsampling, Conformer blocks, CTC head.
//!
//! Every matrix product is int8 × int8 → i32: weights carry one scale per
//! output channel (from the exporter), activations are quantized per row on
//! the fly. Attention is int8 too — `(q+u)·k`, `(q+v)·p` and `softmax·V`.
//! Normalization, activations, softmax, residual adds and the depthwise conv
//! stay f32; they are a small share of the work and the chips this targets
//! (ESP32-S3/P4) have single-precision FPUs.

use crate::Error;
use crate::blob::quantize_row;
use crate::model::{Config, Ffn, Layer, Model, half};
use rlx_mcu::kernels::{
    carve, dot_1x4, dot_2x4, dot_i8, layer_norm, linear_q8 as linear, sigmoid, silu,
};

const LN_EPS: f32 = 1e-5;

/// Element counts of the two scratch arenas for audio up to a length.
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

/// Split the f32 arena into named buffers (sizes depend only on config and
/// frame counts, so `scratch_len` and `transcribe` use the same layout).
struct F32Layout {
    mel: usize,
    fft: usize,
    s1row: usize,
    s2row: usize,
    x: usize,
    h: usize,
    big: usize,
    q: usize,
    k: usize,
    v: usize,
    pe: usize,
    p: usize,
    row: usize,
    scales: usize,
}

impl F32Layout {
    fn new(cfg: &Config, mel_frames: usize) -> Self {
        let t = cfg.enc_frames(mel_frames);
        let (d, ch) = (cfg.d_model, cfg.sub_channels);
        let np = 2 * t - 1;
        Self {
            mel: mel_frames * cfg.n_mels,
            fft: 2 * cfg.n_fft,
            s1row: cfg.f1() * ch,
            s2row: ch * cfg.f2(),
            x: t * d,
            h: t * d,
            big: t * cfg.ff_dim.max(2 * d),
            q: t * d,
            k: t * d,
            v: t * d,
            pe: np * d,
            p: np * d,
            // One score row, or one logits row.
            row: t.max(cfg.num_classes),
            // Key, position and value scales for one head.
            scales: t + np + cfg.head_dim(),
        }
    }

    fn total(&self) -> usize {
        self.mel
            + self.fft
            + self.s1row
            + self.s2row
            + self.x
            + self.h
            + self.big
            + self.q
            + self.k
            + self.v
            + self.pe
            + self.p
            + self.row
            + self.scales
    }
}

struct I8Layout {
    xq: usize,
    ring: usize,
    kq: usize,
    pq: usize,
    vq: usize,
    qq: usize,
    probq: usize,
}

impl I8Layout {
    fn new(cfg: &Config, mel_frames: usize) -> Self {
        let t = cfg.enc_frames(mel_frames);
        let (d, ch, hd) = (cfg.d_model, cfg.sub_channels, cfg.head_dim());
        Self {
            xq: 2 * (ch * cfg.f2()).max(cfg.ff_dim).max(9 * ch).max(d),
            ring: 3 * cfg.f1() * ch,
            kq: t * hd,
            pq: (2 * t - 1) * hd,
            vq: hd * t,
            qq: 2 * hd,
            probq: t,
        }
    }

    fn total(&self) -> usize {
        self.xq + self.ring + self.kq + self.pq + self.vq + self.qq + self.probq
    }
}

/// Scratch needed to transcribe up to `max_samples` of audio.
pub fn scratch_len(model: &Model<'_>, max_samples: usize) -> ScratchLen {
    let cfg = model.config();
    let frames = 1 + max_samples / cfg.hop;
    ScratchLen {
        f32s: F32Layout::new(cfg, frames).total(),
        i8s: I8Layout::new(cfg, frames).total(),
    }
}

/// Encoder + CTC over a time-major `[frames, n_mels]` mel. Writes collapsed,
/// blank-free token ids into `ids` and returns how many.
pub(crate) fn run(
    model: &Model<'_>,
    mel_frames: usize,
    fs: &mut [f32],
    is: &mut [i8],
    ids: &mut [u16],
    features: Option<&mut [f32]>,
) -> Result<usize, Error> {
    let cfg = *model.config();
    let fl = F32Layout::new(&cfg, mel_frames);
    let il = I8Layout::new(&cfg, mel_frames);
    if fs.len() < fl.total() || is.len() < il.total() {
        return Err(Error::ScratchTooSmall);
    }
    let mut f = fs;
    let mel = carve(&mut f, fl.mel);
    let _fft = carve(&mut f, fl.fft);
    let s1row = carve(&mut f, fl.s1row);
    let s2row = carve(&mut f, fl.s2row);
    let x = carve(&mut f, fl.x);
    let h = carve(&mut f, fl.h);
    let big = carve(&mut f, fl.big);
    let q = carve(&mut f, fl.q);
    let k = carve(&mut f, fl.k);
    let v = carve(&mut f, fl.v);
    let pe = carve(&mut f, fl.pe);
    let p = carve(&mut f, fl.p);
    let row = carve(&mut f, fl.row);
    let scales = carve(&mut f, fl.scales);
    let mut i = is;
    let xq = carve(&mut i, il.xq);
    let ring = carve(&mut i, il.ring);
    let kq = carve(&mut i, il.kq);
    let pq = carve(&mut i, il.pq);
    let vq = carve(&mut i, il.vq);
    let qq = carve(&mut i, il.qq);
    let probq = carve(&mut i, il.probq);

    let t = subsample(model, mel, mel_frames, s1row, s2row, ring, xq, x);
    let d = cfg.d_model;
    let xscale = libm::sqrtf(d as f32);
    for v in &mut x[..t * d] {
        *v *= xscale;
    }
    rel_pos_encoding(t, d, pe);

    for li in 0..cfg.n_layers {
        let l = model.layer(li);
        ffn(&l.ff1, &cfg, t, x, h, big, xq);
        attention(
            &l,
            &cfg,
            t,
            AttnBufs {
                x,
                h,
                q,
                k,
                v,
                pe,
                p,
                row,
                scales,
                xq,
                kq,
                pq,
                vq,
                qq,
                probq,
            },
        );
        conv_module(&l, &cfg, t, x, h, big, xq);
        ffn(&l.ff2, &cfg, t, x, h, big, xq);
        layer_norm(x, t, d, l.out_ln.0, Some(l.out_ln.1), LN_EPS, h);
        x[..t * d].copy_from_slice(&h[..t * d]);
    }

    if let Some(out) = features {
        let n = out.len().min(t * d);
        out[..n].copy_from_slice(&x[..n]);
    }

    // CTC head, greedy: argmax per frame, collapse repeats, drop blank.
    let (w, b) = model.ctc;
    let nc = cfg.num_classes;
    let mut n = 0;
    let mut prev = usize::MAX;
    for ti in 0..t {
        linear(&x[ti * d..(ti + 1) * d], 1, &w, Some(b), &mut row[..nc], xq);
        let mut best = 0;
        for c in 1..nc {
            if row[c] > row[best] {
                best = c;
            }
        }
        if best != prev && best != cfg.blank_id {
            if n == ids.len() {
                return Err(Error::OutputTooSmall);
            }
            ids[n] = best as u16;
            n += 1;
        }
        prev = best;
    }
    Ok(n)
}

/// Two k=3, s=2, pad=1 Conv2d + ReLU over `[time, freq]`, then the linear to
/// `d_model`. Streams over time: each second-stage row needs three
/// first-stage rows, kept int8-quantized (one scale per row) in a 3-row ring
/// laid out `[freq][channel]`, so a 3×3 patch is three contiguous runs.
#[allow(clippy::too_many_arguments)]
fn subsample(
    model: &Model<'_>,
    mel: &[f32],
    mel_frames: usize,
    s1row: &mut [f32],
    s2row: &mut [f32],
    ring: &mut [i8],
    xq: &mut [i8],
    x: &mut [f32],
) -> usize {
    let cfg = model.config();
    let (nm, ch) = (cfg.n_mels, cfg.sub_channels);
    let (f1, f2) = (cfg.f1(), cfg.f2());
    let t1n = half(mel_frames);
    let t2n = half(t1n);
    let rowlen = f1 * ch;
    let mut ring_scale = [0.0f32; 3];
    let mut ring_row = [usize::MAX; 3];
    let (w1, b1) = (&model.sub1.0, model.sub1.1);

    for t2 in 0..t2n {
        // First-stage rows 2*t2-1 ..= 2*t2+1 into the ring.
        for r in (2 * t2).saturating_sub(1)..=(2 * t2 + 1) {
            if r >= t1n || ring_row[r % 3] == r {
                continue;
            }
            for fo in 0..f1 {
                for c in 0..ch {
                    let wc = &model.sub0_w[c * 9..c * 9 + 9];
                    let mut acc = model.sub0_b[c];
                    for kh in 0..3 {
                        let ti = (2 * r + kh) as isize - 1;
                        if ti < 0 || ti as usize >= mel_frames {
                            continue;
                        }
                        for kw in 0..3 {
                            let fi = (2 * fo + kw) as isize - 1;
                            if fi < 0 || fi as usize >= nm {
                                continue;
                            }
                            acc += wc[kh * 3 + kw] * mel[ti as usize * nm + fi as usize];
                        }
                    }
                    s1row[fo * ch + c] = acc.max(0.0);
                }
            }
            let slot = r % 3;
            ring_scale[slot] = quantize_row(
                &s1row[..rowlen],
                &mut ring[slot * rowlen..(slot + 1) * rowlen],
            );
            ring_row[slot] = r;
        }

        // Second stage: weights `[out][kh][kw][in]`, so for one kh the
        // patch over (kw, in) is one contiguous run of the ring row. Interior
        // frequency positions share the full kw range, so they go in pairs
        // through the 2×4 tile; the edges, where padding trims kw, go alone.
        let span = |fo: usize| {
            let lo = (2 * fo) as isize - 1;
            let kw0 = usize::from(lo < 0);
            let kw1 = (f1 as isize - lo).min(3) as usize;
            ((lo + kw0 as isize) as usize, kw0, kw1)
        };
        let mut fo = 0;
        while fo < f2 {
            let (fs0, kw0, kw1) = span(fo);
            let pair = fo + 1 < f2 && span(fo + 1).1 == kw0 && span(fo + 1).2 == kw1;
            let fs1 = fs0 + 2;
            let mut o = 0;
            while o < ch {
                let m = (ch - o).min(4);
                let mut acc = [[0.0f32; 4]; 2];
                for kh in 0..3 {
                    let r = (2 * t2 + kh) as isize - 1;
                    if r < 0 || r as usize >= t1n {
                        continue;
                    }
                    let slot = r as usize % 3;
                    let len = (kw1 - kw0) * ch;
                    let base = slot * rowlen;
                    let src0 = &ring[base + fs0 * ch..base + fs0 * ch + len];
                    let seg = |j: usize| &w1.row(o + j)[(kh * 3 + kw0) * ch..(kh * 3 + kw1) * ch];
                    let sc = ring_scale[slot];
                    if m == 4 && pair {
                        let src1 = &ring[base + fs1 * ch..base + fs1 * ch + len];
                        let a = dot_2x4(src0, src1, [seg(0), seg(1), seg(2), seg(3)]);
                        for j in 0..4 {
                            acc[0][j] += a[0][j] as f32 * sc;
                            acc[1][j] += a[1][j] as f32 * sc;
                        }
                    } else if m == 4 {
                        let a = dot_1x4(src0, [seg(0), seg(1), seg(2), seg(3)]);
                        for j in 0..4 {
                            acc[0][j] += a[j] as f32 * sc;
                        }
                    } else {
                        for j in 0..m {
                            acc[0][j] += dot_i8(src0, seg(j)) as f32 * sc;
                            if pair {
                                let src1 = &ring[base + fs1 * ch..base + fs1 * ch + len];
                                acc[1][j] += dot_i8(src1, seg(j)) as f32 * sc;
                            }
                        }
                    }
                }
                for (p, accp) in acc.iter().enumerate().take(1 + usize::from(pair)) {
                    for j in 0..m {
                        let y = accp[j] * w1.scales[o + j] + b1[o + j];
                        // Flattened as `[channel][freq]` (NeMo's transpose + reshape).
                        s2row[(o + j) * f2 + fo + p] = y.max(0.0);
                    }
                }
                o += m;
            }
            fo += 1 + usize::from(pair);
        }
        let (w, b) = &model.pre_out;
        linear(
            &s2row[..ch * f2],
            1,
            w,
            Some(b),
            &mut x[t2 * cfg.d_model..(t2 + 1) * cfg.d_model],
            xq,
        );
    }
    t2n
}

/// NeMo `RelPositionalEncoding`: rows for relative positions `t-1 … -(t-1)`.
fn rel_pos_encoding(t: usize, d: usize, pe: &mut [f32]) {
    let ln10k = libm::log(10000.0);
    for row in 0..2 * t - 1 {
        let pos = (t as f64 - 1.0) - row as f64;
        for i in (0..d).step_by(2) {
            let div = libm::exp(-ln10k * i as f64 / d as f64);
            let a = pos * div;
            pe[row * d + i] = libm::sin(a) as f32;
            if i + 1 < d {
                pe[row * d + i + 1] = libm::cos(a) as f32;
            }
        }
    }
}

/// `x += ½ · W2 silu(W1 LN(x))`.
fn ffn(
    w: &Ffn<'_>,
    cfg: &Config,
    t: usize,
    x: &mut [f32],
    h: &mut [f32],
    big: &mut [f32],
    xq: &mut [i8],
) {
    let (d, ff) = (cfg.d_model, cfg.ff_dim);
    layer_norm(x, t, d, w.ln.0, Some(w.ln.1), LN_EPS, h);
    linear(h, t, &w.l1.0, Some(w.l1.1), big, xq);
    for v in &mut big[..t * ff] {
        *v = silu(*v);
    }
    linear(big, t, &w.l2.0, Some(w.l2.1), h, xq);
    for i in 0..t * d {
        x[i] += 0.5 * h[i];
    }
}

struct AttnBufs<'s> {
    x: &'s mut [f32],
    h: &'s mut [f32],
    q: &'s mut [f32],
    k: &'s mut [f32],
    v: &'s mut [f32],
    pe: &'s [f32],
    p: &'s mut [f32],
    row: &'s mut [f32],
    scales: &'s mut [f32],
    xq: &'s mut [i8],
    kq: &'s mut [i8],
    pq: &'s mut [i8],
    vq: &'s mut [i8],
    qq: &'s mut [i8],
    probq: &'s mut [i8],
}

/// Relative-position multi-head self-attention (Transformer-XL style), int8.
///
/// The `rel_shift` of the reference graph is folded into indexing: score
/// `(i, j)` reads position row `t - 1 - i + j` (relative position `i - j`).
fn attention(l: &Layer<'_>, cfg: &Config, t: usize, b: AttnBufs<'_>) {
    let AttnBufs {
        x,
        h,
        q,
        k,
        v,
        pe,
        p,
        row,
        scales,
        xq,
        kq,
        pq,
        vq,
        qq,
        probq,
    } = b;
    let (d, nh, hd) = (cfg.d_model, cfg.n_heads, cfg.head_dim());
    let np = 2 * t - 1;
    let scale = 1.0 / libm::sqrtf(hd as f32);

    layer_norm(x, t, d, l.att_ln.0, Some(l.att_ln.1), LN_EPS, h);
    linear(h, t, &l.q.0, Some(l.q.1), q, xq);
    linear(h, t, &l.k.0, Some(l.k.1), k, xq);
    linear(h, t, &l.v.0, Some(l.v.1), v, xq);
    linear(pe, np, &l.pos, None, p, xq);

    let (ks, rest) = scales.split_at_mut(t);
    let (ps, vs) = rest.split_at_mut(np);
    let mut tmp = [0.0f32; 128];
    assert!(hd <= tmp.len());
    let (quq, qvq) = qq.split_at_mut(hd);

    for head in 0..nh {
        let off = head * hd;
        for j in 0..t {
            ks[j] = quantize_row(&k[j * d + off..j * d + off + hd], &mut kq[j * hd..]);
        }
        for r in 0..np {
            ps[r] = quantize_row(&p[r * d + off..r * d + off + hd], &mut pq[r * hd..]);
        }
        // V transposed to `[hd, t]`, one scale per channel.
        for c in 0..hd {
            let mut m = 0.0f32;
            for j in 0..t {
                m = m.max(libm::fabsf(v[j * d + off + c]));
            }
            vs[c] = m / 127.0;
            let inv = if m == 0.0 { 0.0 } else { 127.0 / m };
            for j in 0..t {
                vq[c * t + j] = crate::blob::quant(v[j * d + off + c] * inv);
            }
        }
        let u = &l.pos_u[off..off + hd];
        let w = &l.pos_v[off..off + hd];
        for i in 0..t {
            let qi = &q[i * d + off..i * d + off + hd];
            for c in 0..hd {
                tmp[c] = qi[c] + u[c];
            }
            let su = quantize_row(&tmp[..hd], quq);
            for c in 0..hd {
                tmp[c] = qi[c] + w[c];
            }
            let sv = quantize_row(&tmp[..hd], qvq);
            let mut mx = f32::NEG_INFINITY;
            for j in 0..t {
                let ac = dot_i8(quq, &kq[j * hd..(j + 1) * hd]) as f32 * (su * ks[j]);
                let r = t - 1 - i + j;
                let bd = dot_i8(qvq, &pq[r * hd..(r + 1) * hd]) as f32 * (sv * ps[r]);
                let s = (ac + bd) * scale;
                row[j] = s;
                mx = mx.max(s);
            }
            let mut sum = 0.0f32;
            for s in &mut row[..t] {
                *s = libm::expf(*s - mx);
                sum += *s;
            }
            let inv = 1.0 / sum;
            for s in &mut row[..t] {
                *s *= inv;
            }
            let sp = quantize_row(&row[..t], probq);
            for c in 0..hd {
                let acc = dot_i8(&probq[..t], &vq[c * t..(c + 1) * t]);
                h[i * d + off + c] = acc as f32 * (sp * vs[c]);
            }
        }
    }
    // `h` now holds the attention context; project and add.
    linear(h, t, &l.o.0, Some(l.o.1), q, xq);
    for i in 0..t * d {
        x[i] += q[i];
    }
}

/// `x += PW2 silu(DW(GLU(PW1 LN(x))))`, BatchNorm folded into DW.
fn conv_module(
    l: &Layer<'_>,
    cfg: &Config,
    t: usize,
    x: &mut [f32],
    h: &mut [f32],
    big: &mut [f32],
    xq: &mut [i8],
) {
    let (d, kz) = (cfg.d_model, cfg.conv_kernel);
    let pad = (kz - 1) / 2;
    layer_norm(x, t, d, l.conv_ln.0, Some(l.conv_ln.1), LN_EPS, h);
    linear(h, t, &l.pw1.0, Some(l.pw1.1), big, xq);
    // GLU over channels: a * sigmoid(b), into `h`.
    for r in 0..t {
        for c in 0..d {
            h[r * d + c] = big[r * 2 * d + c] * sigmoid(big[r * 2 * d + d + c]);
        }
    }
    // Depthwise, same padding, into `big` (as `[t, d]`), then SiLU.
    for r in 0..t {
        for c in 0..d {
            let wc = &l.dw[c * kz..(c + 1) * kz];
            let mut acc = l.dw_b[c];
            for (j, &wj) in wc.iter().enumerate() {
                let s = r as isize + j as isize - pad as isize;
                if s >= 0 && (s as usize) < t {
                    acc += wj * h[s as usize * d + c];
                }
            }
            big[r * d + c] = silu(acc);
        }
    }
    linear(big, t, &l.pw2.0, Some(l.pw2.1), h, xq);
    for i in 0..t * d {
        x[i] += h[i];
    }
}

/// The mel buffer at the head of the f32 arena, for the frontend to fill.
pub(crate) fn mel_and_fft<'s>(
    cfg: &Config,
    mel_frames: usize,
    fs: &'s mut [f32],
) -> Result<(&'s mut [f32], &'s mut [f32]), Error> {
    let fl = F32Layout::new(cfg, mel_frames);
    if fs.len() < fl.total() {
        return Err(Error::ScratchTooSmall);
    }
    let (mel, rest) = fs.split_at_mut(fl.mel);
    Ok((mel, &mut rest[..fl.fft]))
}
