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

//! **SPAN** — Swift Parameter-free Attention Network (NTIRE 2024), and
//! **SPANV2** — team XiaomiMM's NTIRE 2026 Efficient SR challenge winner.
//!
//! # SPAN
//!
//! Six `SPAB` blocks over a `Conv3XC` stem, with a 1×1 fusion of four taps and
//! a pixel-shuffle head. The "parameter-free attention" is literally that: no
//! learned attention weights, just
//!
//! ```text
//!   sim_att = sigmoid(out3) − 0.5
//!   out     = (out3 + x) · sim_att
//! ```
//!
//! ## `Conv3XC` is reparameterized, not run as written
//!
//! `Conv3XC` trains as a 1×1 → 3×3 → 1×1 expansion plus a 1×1 skip and
//! *collapses to a single 3×3 at inference*. Checkpoints ship the training-time
//! branches, so [`crate::weights::fuse_conv3xc`] folds them exactly as the
//! reference `update_params()` does, and the graph below only ever sees one
//! convolution. Running the unfused form would be arithmetically identical but
//! four times the work.
//!
//! # SPANV2
//!
//! Five `SPABV2` blocks, ReLU instead of SiLU, and a *learned* attention map
//! (`guidance_map_conv`, a 1×1) replacing SPAN's parameter-free sigmoid. The
//! structural novelty is the near-pixel branch: a depthwise 3×3 initialized to
//! exact nearest-neighbour upsampling, concatenated with the deep features and
//! fused by a depthwise-separable pair before one pixel shuffle. Because
//! low-frequency content dominates natural images, that prior is most of the
//! answer and the blocks only learn the residual.
//!
//! The reference calls a fused CUDA kernel (`span_attention`) for
//! `(x + f3) · conv(f3)`; it exists to collapse three DRAM round-trips, not to
//! change the math, so this port emits the elementwise form the reference's own
//! `use_span_attn=False` path uses.

use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirNodeId;

use crate::config::ModelConfig;
use crate::nn::Builder;

/// One `SPAB`: three fused 3×3s, SiLU between, parameter-free attention out.
///
/// Returns `(out, out1)` — SPAN taps `out1` of the *last* block for its
/// concatenation, so the intermediate cannot be discarded.
fn spab(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    channels: usize,
) -> Result<(HirNodeId, HirNodeId)> {
    let out1 = b.conv3x3(wm, &format!("{prefix}.c1_r.eval_conv"), x, channels)?;
    let a1 = b.silu(out1);
    let out2 = b.conv3x3(wm, &format!("{prefix}.c2_r.eval_conv"), a1, channels)?;
    let a2 = b.silu(out2);
    let out3 = b.conv3x3(wm, &format!("{prefix}.c3_r.eval_conv"), a2, channels)?;

    let sig = b.sigmoid(out3);
    let half = b.scalar_like("spab_half", 0.5, out3);
    let sim_att = b.g().sub(sig, half);
    let sum = b.g().add(out3, x);
    let out = b.g().mul(sum, sim_att);
    Ok((out, out1))
}

/// Build SPAN.
pub fn build(
    b: &mut Builder,
    wm: &mut WeightMap,
    cfg: &ModelConfig,
    feature_channels: usize,
    norm: bool,
    img_range: f32,
    rgb_mean: [f32; 3],
    x_in: HirNodeId,
) -> Result<HirNodeId> {
    // The reference normalizes the *input* only; there is no matching
    // denormalization of the output, which is why the mean is not added back.
    let x = if norm {
        let mean = b.constant("span_mean", rgb_mean.to_vec(), &[1, 3, 1, 1]);
        let centered = b.g().sub(x_in, mean);
        let range = b.scalar_like("span_range", img_range, centered);
        b.g().mul(centered, range)
    } else {
        x_in
    };

    let out_feature = b.conv3x3(wm, "conv_1.eval_conv", x, feature_channels)?;

    let (b1, _) = spab(b, wm, "block_1", out_feature, feature_channels)?;
    let (b2, _) = spab(b, wm, "block_2", b1, feature_channels)?;
    let (b3, _) = spab(b, wm, "block_3", b2, feature_channels)?;
    let (b4, _) = spab(b, wm, "block_4", b3, feature_channels)?;
    let (b5, _) = spab(b, wm, "block_5", b4, feature_channels)?;
    let (b6, b5_2) = spab(b, wm, "block_6", b5, feature_channels)?;

    let b6 = b.conv3x3(wm, "conv_2.eval_conv", b6, feature_channels)?;
    let cat = b.g().concat_(vec![out_feature, b6, b1, b5_2], 1);
    let fused = b.conv1x1(wm, "conv_cat", cat, feature_channels)?;

    let r = cfg.scale;
    let up = b.conv3x3(wm, "upsampler.0", fused, cfg.out_ch * r * r)?;
    b.pixel_shuffle(up, r)
}

/// One `SPABV2`.
///
/// When the block's input and output widths agree it applies the learned
/// guidance map; the widening first block cannot (there is nothing to add `x`
/// to) and falls back to a plain ReLU, exactly as the reference does.
fn spabv2(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    in_c: usize,
    channels: usize,
    bias: bool,
) -> Result<HirNodeId> {
    let f1 = b.conv(
        wm,
        &format!("{prefix}.c1.conv"),
        x,
        channels,
        [3, 3],
        [1, 1],
        [1, 1],
        1,
        bias,
    )?;
    let f1 = b.relu(f1);
    let f2 = b.conv(
        wm,
        &format!("{prefix}.c2.conv"),
        f1,
        channels,
        [3, 3],
        [1, 1],
        [1, 1],
        1,
        bias,
    )?;
    let f2 = b.relu(f2);
    let f3 = b.conv(
        wm,
        &format!("{prefix}.c3.conv"),
        f2,
        channels,
        [3, 3],
        [1, 1],
        [1, 1],
        1,
        bias,
    )?;

    if in_c == channels {
        let guidance = b.conv1x1(wm, &format!("{prefix}.guidance_map_conv"), f3, channels)?;
        let sum = b.g().add(x, f3);
        Ok(b.g().mul(sum, guidance))
    } else {
        Ok(b.relu(f3))
    }
}

/// Build SPANV2.
pub fn build_v2(
    b: &mut Builder,
    wm: &mut WeightMap,
    cfg: &ModelConfig,
    feature_channels: usize,
    conv_bias: bool,
    x: HirNodeId,
) -> Result<HirNodeId> {
    let r = cfg.scale;
    let in_ch = cfg.in_ch;

    // Near-pixel branch: depthwise 3×3, one group per input channel, widening
    // to `in_ch · r²` so the shared pixel shuffle turns it into pixel repeat.
    let near = b.conv(
        wm,
        "conv_near",
        x,
        in_ch * r * r,
        [3, 3],
        [1, 1],
        [1, 1],
        in_ch,
        false,
    )?;

    let mut h = spabv2(b, wm, "block_1", x, in_ch, feature_channels, conv_bias)?;
    for i in 2..=5 {
        h = spabv2(
            b,
            wm,
            &format!("block_{i}"),
            h,
            feature_channels,
            feature_channels,
            conv_bias,
        )?;
    }

    let cat = b.g().concat_(vec![near, h], 1);
    let cat_channels = in_ch * r * r + feature_channels;
    let dw = b.conv(
        wm,
        "depthwise_conv",
        cat,
        cat_channels,
        [3, 3],
        [1, 1],
        [1, 1],
        cat_channels,
        conv_bias,
    )?;
    let pw = b.conv(
        wm,
        "pointwise_conv",
        dw,
        in_ch * r * r,
        [1, 1],
        [1, 1],
        [0, 0],
        1,
        conv_bias,
    )?;
    b.pixel_shuffle(pw, r)
}
