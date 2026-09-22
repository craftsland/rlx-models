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

//! **MambaIRv2** — Attentive State Space restoration.
//!
//! Each `AttentiveLayer` is two halves over the same tokens:
//!
//! 1. **Shifted-window attention**, the same machinery as [`super::swin`], but
//!    with the QKV projection hoisted out of the attention module and a
//!    `ConvFFN` (a feed-forward whose hidden activations get a depthwise 5×5
//!    added back) instead of an MLP.
//! 2. **Attentive State Space (ASSM)**, a Mamba-1 selective scan run *not* in
//!    raster order but in **semantic** order: every token is routed to one of
//!    `num_tokens` learned prototypes, the sequence is sorted by prototype, the
//!    scan runs over that permutation, and the result is scattered back. Tokens
//!    that look alike become neighbours, so a 1-D scan sees them together.
//!
//! Both halves are wrapped in a LayerScale residual (`scale1` / `scale2`).
//!
//! # This port is deterministic; the reference is not
//!
//! Upstream routes with `F.gumbel_softmax(pred_route, hard=True)` and there is
//! **no `self.training` guard anywhere in the file** — so the released model
//! draws fresh Gumbel noise on every forward pass, and the token ordering (and
//! therefore the scan, and therefore the image) differs run to run.
//! `torch.sort(..., stable=False)` over only `num_tokens` distinct values adds a
//! second source of nondeterminism through tie-breaking.
//!
//! This port takes the `argmax` of the routing logits with a **stable** sort.
//! That is a deliberate deviation, chosen because a nondeterministic upscaler is
//! not useful and because matching PyTorch's RNG stream *and* its unstable
//! sort's tie-breaking is not achievable in any case. Expect a faithful
//! rendering of what the model means, not a bit-match of any single upstream
//! run — which is not reproducible even upstream.
//!
//! Two simplifications fall out of dropping the noise, both exact:
//!
//! * `route` ends in `LogSoftmax`, which is monotonic, so `argmax` is unchanged
//!   by it and the op is skipped entirely.
//! * `full_embedding = embeddingB.weight @ embeddingA.weight` is a product of
//!   two *parameters*, so it is a compile-time constant folded on the host; the
//!   prompt becomes one `Gather`.

use anyhow::{Context, Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::HirNodeId;
use rlx_ir::op::Op;
use rlx_ir::{DType, HirGraphExt, Shape};

use super::swin::{self, Win};
use crate::config::{MambaIrParams, ModelConfig, ResiConnection, Upsampler};
use crate::nn::Builder;

/// `ConvFFN`: a feed-forward whose hidden activations get a depthwise
/// convolution added back before the down-projection.
fn conv_ffn(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    g: Win,
    c: usize,
    hidden: usize,
    kernel: usize,
) -> Result<HirNodeId> {
    let h = b.linear(wm, &format!("{prefix}.fc1"), x, hidden, true)?;
    let h = b.gelu(h);
    // The depthwise branch is *added* to the activations, not substituted.
    let nchw = swin::to_nchw(b, h, g, hidden);
    let dw = b.conv(
        wm,
        &format!("{prefix}.dwconv.depthwise_conv.0"),
        nchw,
        hidden,
        [kernel, kernel],
        [1, 1],
        [(kernel - 1) / 2, (kernel - 1) / 2],
        hidden,
        true,
    )?;
    let dw = b.gelu(dw);
    let dw = swin::to_tokens(b, dw, g, hidden);
    let h = b.g().add(h, dw);
    b.linear(wm, &format!("{prefix}.fc2"), h, c, true)
}

/// The window-attention half of an `AttentiveLayer`.
///
/// Differs from [`super::swin`]'s block in that QKV is projected *before* the
/// shift and window partition — the reference shifts the already-projected
/// `[B, H, W, 3C]` tensor — so the projection happens once on the whole map.
#[allow(clippy::too_many_arguments)]
fn window_half(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    g: Win,
    c: usize,
    heads: usize,
    shift: usize,
    qkv_bias: bool,
) -> Result<HirNodeId> {
    ensure!(
        c.is_multiple_of(heads),
        "{prefix}: {c} channels do not split into {heads} heads"
    );
    let hd = c / heads;
    let qkv = b.linear(wm, &format!("{prefix}.wqkv"), x, 3 * c, qkv_bias)?;

    let shifted = swin::roll(b, qkv, g, 3 * c, -(shift as isize));
    let windows = swin::window_partition(b, shifted, g, 3 * c);

    let head_split = |b: &mut Builder, t: HirNodeId| {
        let r = b
            .g()
            .reshape_(t, vec![g.nw as i64, g.n as i64, heads as i64, hd as i64]);
        b.g().transpose_(r, vec![0, 2, 1, 3])
    };
    let q = b.g().narrow_(windows, 2, 0, c);
    let k = b.g().narrow_(windows, 2, c, c);
    let v = b.g().narrow_(windows, 2, 2 * c, c);
    let q = head_split(b, q);
    let k = head_split(b, k);
    let v = head_split(b, v);

    let scale = b.scalar_like("mamba_attn_scale", 1.0 / (hd as f32).sqrt(), q);
    let q = b.g().mul(q, scale);
    let kt = b.g().transpose_(k, vec![0, 1, 3, 2]);
    let mut attn = b.g().mm(q, kt);

    let index = swin::rpe_index(g.ws);
    let bias = swin::rpe_bias(
        b,
        wm,
        &format!("{prefix}.win_mhsa.relative_position_bias_table"),
        &index,
        g.n,
        g.n,
        (2 * g.ws - 1) * (2 * g.ws - 1),
        heads,
        "mamba_rpe",
    )?;
    attn = b.g().add(attn, bias);

    if shift > 0 {
        let m = swin::shift_mask(g, shift);
        let mask = b.constant("mamba_shift_mask", m, &[1, g.nw, 1, g.n, g.n]);
        let grouped = b.g().reshape_(
            attn,
            vec![1, g.nw as i64, heads as i64, g.n as i64, g.n as i64],
        );
        let masked = b.g().add(grouped, mask);
        attn = b.g().reshape_(
            masked,
            vec![g.nw as i64, heads as i64, g.n as i64, g.n as i64],
        );
    }

    let attn = b.g().sm(attn, -1);
    let out = b.g().mm(attn, v);
    let out = b.g().transpose_(out, vec![0, 2, 1, 3]);
    let out = b.g().reshape_(out, vec![g.nw as i64, g.n as i64, c as i64]);
    let projected = b.linear(wm, &format!("{prefix}.win_mhsa.proj"), out, c, true)?;
    let merged = swin::window_reverse(b, projected, g, c);
    Ok(swin::roll(b, merged, g, c, shift as isize))
}

/// Gather `[1, L, width]` along the token axis by a `[1, L]` index.
fn permute_tokens(
    b: &mut Builder,
    x: HirNodeId,
    index: HirNodeId,
    l: usize,
    width: usize,
) -> HirNodeId {
    let flat = b.g().reshape_(index, vec![l as i64]);
    let table = b.g().reshape_(x, vec![l as i64, width as i64]);
    let picked = b.g().gather_(table, flat, 0);
    b.g().reshape_(picked, vec![1, l as i64, width as i64])
}

/// Argsort along the token axis. For a permutation this is its inverse, which
/// is how the scan's output is scattered back to raster order.
fn argsort_tokens(b: &mut Builder, x: HirNodeId, l: usize) -> HirNodeId {
    let shape = Shape::new(&[1, l], DType::F32);
    b.g().add_node(
        Op::ArgSort {
            axis: 1,
            descending: false,
        },
        vec![x],
        shape,
    )
}

/// The Attentive State Space half.
#[allow(clippy::too_many_arguments)]
fn assm(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    // The enclosing `AttentiveLayer`, which owns `embeddingA`.
    layer: &str,
    x: HirNodeId,
    g: Win,
    c: usize,
    p: &MambaIrParams,
) -> Result<HirNodeId> {
    let l = g.h * g.w;
    let hidden = (c as f32 * p.mlp_ratio) as usize;
    let d_state = p.d_state;

    // ── routing ──────────────────────────────────────────────────────
    // `route` is Linear → GELU → Linear → LogSoftmax. LogSoftmax is monotonic,
    // so it cannot change the argmax and is skipped.
    let r = b.linear(wm, &format!("{prefix}.route.0"), x, c / 3, true)?;
    let r = b.gelu(r);
    let logits = b.linear(wm, &format!("{prefix}.route.2"), r, p.num_tokens, true)?;
    let idx = b.g().add_node(
        Op::ArgMax {
            axis: 2,
            keep_dim: false,
        },
        vec![logits],
        Shape::new(&[1, l], DType::F32),
    );

    // `embeddingB.weight @ embeddingA.weight` is a product of two parameters,
    // so the prototype table is a compile-time constant.
    let eb = b.take_host(
        wm,
        &format!("{prefix}.embeddingB.weight"),
        &[p.num_tokens, p.inner_rank],
    )?;
    let ea = b.take_host(
        wm,
        &format!("{layer}.embeddingA.weight"),
        &[p.inner_rank, d_state],
    )?;
    let mut full = vec![0.0f32; p.num_tokens * d_state];
    for t in 0..p.num_tokens {
        for s in 0..d_state {
            let mut acc = 0.0f32;
            for k in 0..p.inner_rank {
                acc += eb[t * p.inner_rank + k] * ea[k * d_state + s];
            }
            full[t * d_state + s] = acc;
        }
    }
    let full = b.constant("mamba_prototypes", full, &[p.num_tokens, d_state]);
    let flat_idx = b.g().reshape_(idx, vec![l as i64]);
    let prompt = b.g().gather_(full, flat_idx, 0);
    let prompt = b.g().reshape_(prompt, vec![1, l as i64, d_state as i64]);

    // Sort tokens by prototype: `perm` is the semantic order, and argsort of a
    // permutation is its inverse, which folds the scan's output back.
    let perm = argsort_tokens(b, idx, l);
    let inverse = argsort_tokens(b, perm, l);

    // ── input projection ─────────────────────────────────────────────
    let nchw = swin::to_nchw(b, x, g, c);
    let proj = b.conv(
        wm,
        &format!("{prefix}.in_proj.0"),
        nchw,
        hidden,
        [1, 1],
        [1, 1],
        [0, 0],
        1,
        true,
    )?;
    // Conditional positional encoding, applied as a gate.
    let cpe = b.conv(
        wm,
        &format!("{prefix}.CPE.0"),
        proj,
        hidden,
        [3, 3],
        [1, 1],
        [1, 1],
        hidden,
        true,
    )?;
    let gate = b.sigmoid(cpe);
    let gated = b.g().mul(proj, gate);
    let tokens = swin::to_tokens(b, gated, g, hidden);

    // ── selective scan, in semantic order ────────────────────────────
    let ordered = permute_tokens(b, tokens, perm, l, hidden);
    let ordered_prompt = permute_tokens(b, prompt, perm, l, d_state);
    let scanned = selective_scan(
        b,
        wm,
        &format!("{prefix}.selectiveScan"),
        ordered,
        ordered_prompt,
        l,
        hidden,
        d_state,
    )?;

    let normed = b.layer_norm(wm, &format!("{prefix}.out_norm"), scanned, 1e-5)?;
    let out = b.linear(wm, &format!("{prefix}.out_proj"), normed, c, true)?;
    Ok(permute_tokens(b, out, inverse, l, c))
}

/// Mamba-1 selective scan with MambaIRv2's prompt added to `C`.
///
/// `rlx`'s [`Op::SelectiveScan`] implements
/// `h[t] = exp(Δ[t]·A)·h[t−1] + Δ[t]·B[t]·x[t]`, `y[t] = C[t]·h[t]`, which is
/// exactly the recurrence here. The reference's `D·x` skip is not part of that
/// op and is added afterwards.
#[allow(clippy::too_many_arguments)]
fn selective_scan(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    prompt: HirNodeId,
    l: usize,
    hidden: usize,
    d_state: usize,
) -> Result<HirNodeId> {
    // `dt_rank = ceil(d_model / 16)`, and `d_model` here is the scan's own
    // width because MambaIRv2 builds `Selective_Scan` with `expand=1`.
    let dt_rank = hidden.div_ceil(16);

    // `x_proj_weight` is stored as `[K=1, dt_rank + 2·d_state, hidden]`.
    let rows = dt_rank + 2 * d_state;
    let xw = b.take_host(wm, &format!("{prefix}.x_proj_weight"), &[1, rows, hidden])?;
    // Transposed once here so the graph can use a plain matmul.
    let mut xw_t = vec![0.0f32; hidden * rows];
    for r in 0..rows {
        for h in 0..hidden {
            xw_t[h * rows + r] = xw[r * hidden + h];
        }
    }
    let xw_t = b.constant("mamba_x_proj", xw_t, &[hidden, rows]);
    let dbl = b.g().mm(x, xw_t);

    let dts = b.g().narrow_(dbl, 2, 0, dt_rank);
    let bs = b.g().narrow_(dbl, 2, dt_rank, d_state);
    let cs = b.g().narrow_(dbl, 2, dt_rank + d_state, d_state);

    let dw = b.take_host(
        wm,
        &format!("{prefix}.dt_projs_weight"),
        &[1, hidden, dt_rank],
    )?;
    let mut dw_t = vec![0.0f32; dt_rank * hidden];
    for h in 0..hidden {
        for r in 0..dt_rank {
            dw_t[r * hidden + h] = dw[h * dt_rank + r];
        }
    }
    let dw_t = b.constant("mamba_dt_proj", dw_t, &[dt_rank, hidden]);
    let delta = b.g().mm(dts, dw_t);

    // `delta_softplus=True` with the projection bias folded in.
    let dbias = b.take(wm, &format!("{prefix}.dt_projs_bias"), &[1, hidden])?;
    let dbias = b.g().reshape_(dbias, vec![1, 1, hidden as i64]);
    let delta = b.g().add(delta, dbias);
    let delta = b.act(rlx_ir::op::Activation::Softplus, delta);

    // `A = −exp(A_logs)`, folded on the host — `A_logs` is frozen.
    let a_logs = b.take_host(wm, &format!("{prefix}.A_logs"), &[hidden, d_state])?;
    let a: Vec<f32> = a_logs.iter().map(|v| -v.exp()).collect();
    let a = b.constant("mamba_a", a, &[hidden, d_state]);

    // The Attentive State Space contribution: the routed prompt biases C.
    let cs = b.g().add(cs, prompt);

    let y = b.g().add_node(
        Op::SelectiveScan {
            state_size: d_state,
        },
        vec![x, delta, a, bs, cs],
        Shape::new(&[1, l, hidden], DType::F32),
    );

    // The `D · x` skip connection, which the op does not cover.
    let d = b.take(wm, &format!("{prefix}.Ds"), &[hidden])?;
    let d = b.g().reshape_(d, vec![1, 1, hidden as i64]);
    let skip = b.g().mul(x, d);
    Ok(b.g().add(y, skip))
}

/// One `AttentiveLayer`: window attention then state space, each with its own
/// `ConvFFN` and LayerScale residual.
#[allow(clippy::too_many_arguments)]
fn attentive_layer(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    g: Win,
    c: usize,
    heads: usize,
    shift: usize,
    p: &MambaIrParams,
) -> Result<HirNodeId> {
    let hidden = (c as f32 * p.mlp_ratio) as usize;

    // ── part 1: shifted-window attention ─────────────────────────────
    let shortcut = x;
    let n1 = b.layer_norm(wm, &format!("{prefix}.norm1"), x, 1e-5)?;
    let attn = window_half(b, wm, prefix, n1, g, c, heads, shift, p.qkv_bias)?;
    let x_win = b.g().add(attn, shortcut);
    let n2 = b.layer_norm(wm, &format!("{prefix}.norm2"), x_win, 1e-5)?;
    let ff = conv_ffn(
        b,
        wm,
        &format!("{prefix}.convffn1"),
        n2,
        g,
        c,
        hidden,
        p.convffn_kernel_size,
    )?;
    let x_win = b.g().add(ff, x_win);
    // LayerScale on the *shortcut*, not on the branch — the reference weights
    // the residual down, which is why `scale1` initializes at 1e-4.
    let s1 = b.take(wm, &format!("{prefix}.scale1"), &[c])?;
    let scaled = b.g().mul(shortcut, s1);
    let x = b.g().add(scaled, x_win);

    // ── part 2: attentive state space ────────────────────────────────
    let shortcut = x;
    let n3 = b.layer_norm(wm, &format!("{prefix}.norm3"), x, 1e-5)?;
    let state = assm(b, wm, &format!("{prefix}.assm"), prefix, n3, g, c, p)?;
    let x_aca = b.g().add(state, x);
    let n4 = b.layer_norm(wm, &format!("{prefix}.norm4"), x_aca, 1e-5)?;
    let ff = conv_ffn(
        b,
        wm,
        &format!("{prefix}.convffn2"),
        n4,
        g,
        c,
        hidden,
        p.convffn_kernel_size,
    )?;
    let x = b.g().add(x_aca, ff);
    let s2 = b.take(wm, &format!("{prefix}.scale2"), &[c])?;
    let scaled = b.g().mul(shortcut, s2);
    Ok(b.g().add(scaled, x))
}

/// Build MambaIRv2.
pub fn build(
    b: &mut Builder,
    wm: &mut WeightMap,
    cfg: &ModelConfig,
    p: &MambaIrParams,
    x_in: HirNodeId,
) -> Result<HirNodeId> {
    let [_, _, h, w] = b.dims4(x_in);
    let g = Win::new(h, w, p.window_size)?;
    let c = p.embed_dim;

    let mean = b.constant("mamba_mean", vec![0.4488, 0.4371, 0.4040], &[1, 3, 1, 1]);
    let centered = b.g().sub(x_in, mean);
    let range = b.scalar_like("mamba_range", p.img_range, centered);
    let x = b.g().mul(centered, range);

    let shallow = b.conv3x3(wm, "conv_first", x, c)?;
    let mut tokens = swin::to_tokens(b, shallow, g, c);
    if p.patch_norm {
        tokens = b.layer_norm(wm, "patch_embed.norm", tokens, 1e-5)?;
    }

    for (gi, &depth) in p.depths.iter().enumerate() {
        let group_in = tokens;
        for d in 0..depth {
            tokens = attentive_layer(
                b,
                wm,
                &format!("layers.{gi}.residual_group.layers.{d}"),
                tokens,
                g,
                c,
                p.num_heads[gi],
                if d % 2 == 0 { 0 } else { p.window_size / 2 },
                p,
            )
            .with_context(|| format!("MambaIRv2 group {gi} layer {d}"))?;
        }
        if p.resi_connection != ResiConnection::Identity {
            let nchw = swin::to_nchw(b, tokens, g, c);
            let conved = match p.resi_connection {
                ResiConnection::Conv1 => b.conv3x3(wm, &format!("layers.{gi}.conv"), nchw, c)?,
                _ => {
                    let t = b.conv3x3(wm, &format!("layers.{gi}.conv.0"), nchw, c / 4)?;
                    let t = b.leaky_relu(t, 0.2);
                    let t = b.conv1x1(wm, &format!("layers.{gi}.conv.2"), t, c / 4)?;
                    let t = b.leaky_relu(t, 0.2);
                    b.conv3x3(wm, &format!("layers.{gi}.conv.4"), t, c)?
                }
            };
            tokens = swin::to_tokens(b, conved, g, c);
        }
        tokens = b.g().add(tokens, group_in);
    }

    let tokens = b.layer_norm(wm, "norm", tokens, 1e-5)?;
    let deep = swin::to_nchw(b, tokens, g, c);
    let body = b.conv3x3(wm, "conv_after_body", deep, c)?;
    let feat = b.g().add(body, shallow);

    let r = cfg.scale;
    let out = match p.upsampler {
        Upsampler::PixelShuffleDirect => {
            let up = b.conv3x3(wm, "upsample.0", feat, cfg.out_ch * r * r)?;
            b.pixel_shuffle(up, r)?
        }
        Upsampler::PixelShuffle => {
            let nf = p.num_feat;
            let h0 = b.conv3x3(wm, "conv_before_upsample.0", feat, nf)?;
            let mut h0 = b.leaky_relu(h0, 0.01);
            let stages: Vec<usize> = if r == 3 {
                vec![3]
            } else {
                ensure!(
                    r.is_power_of_two(),
                    "MambaIRv2's upsampler supports 2^n and 3, not ×{r}"
                );
                vec![2; r.trailing_zeros() as usize]
            };
            for (i, s) in stages.into_iter().enumerate() {
                h0 = b.conv3x3(wm, &format!("upsample.{}", i * 2), h0, nf * s * s)?;
                h0 = b.pixel_shuffle(h0, s)?;
            }
            b.conv3x3(wm, "conv_last", h0, cfg.out_ch)?
        }
        other => anyhow::bail!("MambaIRv2 does not use the {other:?} upsampler"),
    };

    let inv = b.scalar_like("mamba_inv_range", 1.0 / p.img_range, out);
    let scaled = b.g().mul(out, inv);
    let mean = b.constant(
        "mamba_mean_out",
        vec![0.4488, 0.4371, 0.4040],
        &[1, 3, 1, 1],
    );
    Ok(b.g().add(scaled, mean))
}
