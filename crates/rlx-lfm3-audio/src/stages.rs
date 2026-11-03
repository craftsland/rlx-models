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

//! The four graphs: speech encoder, LM step, depth decoder, codec decoder.

use anyhow::{Result, bail};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, NodeId, Op, Shape};

use crate::config::{CODEC_FRAME_STACK, Lfm3AudioConfig};
use crate::graph::Gb;
use crate::trunk::{self, StepState};

/// `Op::Conv` over NCHW `[1, ·, hw.0, hw.1]` with a `k×k` kernel, stride
/// `s`, padding `p`. Returns the output and its spatial size.
#[allow(clippy::too_many_arguments)]
fn conv2d(
    gb: &mut Gb,
    x: NodeId,
    hw: (usize, usize),
    key: &str,
    out_ch: usize,
    in_per_group: usize,
    k: usize,
    s: usize,
    p: usize,
    groups: usize,
) -> Result<(NodeId, (usize, usize))> {
    let (h, w) = hw;
    let (ho, wo) = ((h + 2 * p - k) / s + 1, (w + 2 * p - k) / s + 1);
    let weight = gb.tensor(&format!("{key}.weight"), &[out_ch, in_per_group, k, k])?;
    let y = gb.g.add_node(
        Op::Conv {
            kernel_size: vec![k, k],
            stride: vec![s, s],
            padding: vec![p, p],
            dilation: vec![1, 1],
            groups,
        },
        vec![x, weight],
        Shape::new(&[1, out_ch, ho, wo], DType::F32),
    );
    let bias = gb.tensor(&format!("{key}.bias"), &[1, out_ch, 1, 1])?;
    Ok((gb.g.add(y, bias), (ho, wo)))
}

/// Speech encoder + projector over `frames` log-mel frames (a padded bucket).
///
/// Inputs: `mel` `[1, 1, n_mels, frames]`; `mask1` / `mask2` `[1, 1, 1, ·]`
/// zero the downsampler columns past the clip after its first two stages
/// (`Lfm3AudioDownsampler.mask_padding`). Output: `[ceil(frames/8), lm_hidden]`
/// audio embeddings; rows past the clip's own `ceil(len/8)` are padding.
pub(crate) fn encoder(gb: &mut Gb, cfg: &Lfm3AudioConfig, frames: usize) -> Result<()> {
    let (mels, ch) = (cfg.num_mel_bins, cfg.downsample_channels);
    let t1 = frames.div_ceil(2);
    let t2 = t1.div_ceil(2);
    let t3 = t2.div_ceil(2);
    let mel = gb.input("mel", &[1, 1, mels, frames]);
    let mask1 = gb.input("mask1", &[1, 1, 1, t1]);
    let mask2 = gb.input("mask2", &[1, 1, 1, t2]);

    let (x, hw) = conv2d(gb, mel, (mels, frames), "a.conv1d.0", ch, 1, 3, 2, 1, 1)?;
    let x = gb.g.silu(x);
    let x = gb.g.mul(x, mask1);
    let (x, hw) = conv2d(gb, x, hw, "a.conv1d.2", ch, 1, 3, 2, 1, ch)?;
    let (x, hw) = conv2d(gb, x, hw, "a.conv1d.3", ch, ch, 1, 1, 0, 1)?;
    let x = gb.g.silu(x);
    let x = gb.g.mul(x, mask2);
    let (x, hw) = conv2d(gb, x, hw, "a.conv1d.5", ch, 1, 3, 2, 1, ch)?;
    let (x, hw) = conv2d(gb, x, hw, "a.conv1d.6", ch, ch, 1, 1, 0, 1)?;
    let x = gb.g.silu(x);
    debug_assert_eq!(hw.1, t3);
    // [1, ch, mels/8, t3] -> [t3, ch * mels/8] (channel-major, as `flatten(1, 2)`).
    let feat = ch * hw.0;
    let x = gb.g.reshape_(x, vec![feat as i64, t3 as i64]);
    let x = gb.g.transpose_(x, vec![1, 0]);
    let x = gb.linear_bias(x, "a.pre_encode.out.weight", "a.pre_encode.out.bias", t3)?;

    let h = trunk::sequence(gb, &cfg.encoder, x, t3)?;

    // Projector: LayerNorm -> Linear -> GELU (erf) -> Linear.
    let (g_ln, _) = gb.take("mm.a.mlp.0.weight")?;
    let (b_ln, _) = gb.take("mm.a.mlp.0.bias")?;
    let n = g_ln.len();
    let g_ln = gb.constant("mm.a.mlp.0.weight", g_ln, &[n]);
    let b_ln = gb.constant("mm.a.mlp.0.bias", b_ln, &[n]);
    let h = gb.g.ln(h, g_ln, b_ln, 1e-5);
    let h = gb.linear_bias(h, "mm.a.mlp.1.weight", "mm.a.mlp.1.bias", t3)?;
    let h = gb.g.gelu(h);
    let out = gb.linear_bias(h, "mm.a.mlp.3.weight", "mm.a.mlp.3.bias", t3)?;
    gb.g.set_outputs(vec![out]);
    Ok(())
}

/// Vocabulary rows per slice of the tied LM head (see [`Gb::linear_sliced`]).
const HEAD_SLICE_ROWS: usize = 8192;

/// One LM step graph over `rows` new positions with `max_kv` cache slots.
///
/// Inputs: `x` `[rows, hidden]` input embeddings plus the [`trunk::step`]
/// inputs. Outputs: the last row's final-normed hidden state `[1, hidden]`,
/// its logits `[1, vocab]` (tied head on the packed `token_embd`) unless
/// `host_head` leaves them to the caller, then the caches in [`StepState`]
/// order.
pub(crate) fn lm_step(
    gb: &mut Gb,
    cfg: &Lfm3AudioConfig,
    rows: usize,
    max_kv: usize,
    host_head: bool,
) -> Result<StepState> {
    let x = gb.input("x", &[rows, cfg.lm.hidden]);
    let (h, caches, state) = trunk::step(gb, &cfg.lm, x, rows, max_kv)?;
    let last = if rows == 1 {
        h
    } else {
        gb.g.narrow_(h, 0, rows - 1, 1)
    };
    let mut outs = vec![last];
    if !host_head {
        outs.push(gb.linear_sliced(last, "token_embd.weight", 1, HEAD_SLICE_ROWS)?);
    }
    outs.extend(caches);
    gb.g.set_outputs(outs);
    Ok(state)
}

/// The depth decoder: one LM hidden state + 12 slot embeddings (each the
/// codebook's known code embedding or the mask embedding) → logits
/// `[codebooks, codebook_size]`.
///
/// Inputs: `hidden` `[1, lm_hidden]`, `slots` `[codebooks, depth_hidden]`.
/// The per-codebook embedding `a.gen.code.embd` `[codebooks, codebook_size,
/// dim]` is also the tied output head.
pub(crate) fn depth(gb: &mut Gb, cfg: &Lfm3AudioConfig) -> Result<()> {
    const TABLE: &str = "a.gen.code.embd.weight";
    let (nq, cs, d) = (cfg.num_codebooks, cfg.codebook_size, cfg.depth.hidden);
    let hidden = gb.input("hidden", &[1, cfg.lm.hidden]);
    let slots = gb.input("slots", &[nq, d]);
    // 12288 output rows: sliced so the dequant scratch stays one slice.
    let din = gb.linear_sliced(hidden, "a.gen.code.proj_in.weight", 1, 2048)?;
    let (b, _) = gb.take("a.gen.code.proj_in.bias")?;
    let n = b.len();
    let b = gb.constant("a.gen.code.proj_in.bias", b, &[1, n]);
    let din = gb.g.add(din, b);
    let din = gb.g.reshape_(din, vec![nq as i64, d as i64]);
    let din = gb.g.add(din, slots);
    let out = trunk::sequence(gb, &cfg.depth, din, nq)?;

    // Per-codebook RMSNorm (unit norm, then that codebook's gain row).
    let y = gb.rms_unit(out, d, cfg.depth.eps);
    let gains = gb.tensor("a.gen.code.head_norm.weight", &[nq, d])?;
    let y = gb.g.mul(y, gains);
    // Tied heads. A packed table stays packed: codebook q is rows
    // `q*cs..(q+1)*cs`, a contiguous byte range, scored as its own matmul.
    let logits = if gb.is_packed(TABLE) {
        let mut rows = Vec::with_capacity(nq);
        for q in 0..nq {
            let yq = gb.g.narrow_(y, 0, q, 1);
            rows.push(gb.linear_rows(yq, TABLE, 1, q * cs, cs)?);
        }
        gb.g.concat_(rows, 0)
    } else {
        let (table, _) = gb.take(TABLE)?;
        if table.len() != nq * cs * d {
            bail!(
                "depth table holds {} values, expected {nq}x{cs}x{d}",
                table.len()
            );
        }
        // [nq, 1, d] x [nq, d, cs].
        let mut head = vec![0f32; nq * d * cs];
        for q in 0..nq {
            for c in 0..cs {
                for j in 0..d {
                    head[q * d * cs + j * cs + c] = table[(q * cs + c) * d + j];
                }
            }
        }
        let head = gb.constant(TABLE, head, &[nq, d, cs]);
        let y = gb.g.reshape_(y, vec![nq as i64, 1, d as i64]);
        let l = gb.g.mm(y, head);
        gb.g.reshape_(l, vec![nq as i64, cs as i64])
    };
    gb.g.set_outputs(vec![logits]);
    Ok(())
}

/// The codec decoder up to the iSTFT input: FSQ latents of `frames` code
/// frames → `[frames * 6, n_fft + 2]` (log-magnitudes, then phases).
///
/// Input: `latents` `[frames, codebooks * fsq_dim]`.
pub(crate) fn codec(gb: &mut Gb, cfg: &Lfm3AudioConfig, frames: usize) -> Result<()> {
    let lat_dim = cfg.num_codebooks * cfg.fsq_dim;
    let h = cfg.codec.hidden;
    let lat = gb.input("latents", &[frames, lat_dim]);
    let x = gb.linear_bias(
        lat,
        "a.gen.wav.quant.first.out_proj.weight",
        "a.gen.wav.quant.first.out_proj.bias",
        frames,
    )?;
    let x = gb.linear_bias(
        x,
        "a.gen.wav.tfm.in_proj.weight",
        "a.gen.wav.tfm.in_proj.bias",
        frames,
    )?;
    // Nearest upsampling by the frame stack: every frame repeated 6 times.
    let x = gb.g.reshape_(x, vec![frames as i64, 1, h as i64]);
    let x =
        gb.g.concat_(std::iter::repeat_n(x, CODEC_FRAME_STACK).collect(), 1);
    let steps = frames * CODEC_FRAME_STACK;
    let x = gb.g.reshape_(x, vec![steps as i64, h as i64]);
    let y = trunk::sequence(gb, &cfg.codec, x, steps)?;
    let spec = gb.linear_bias(
        y,
        "a.gen.wav.tfm.out_proj.weight",
        "a.gen.wav.tfm.out_proj.bias",
        steps,
    )?;
    gb.g.set_outputs(vec![spec]);
    Ok(())
}
