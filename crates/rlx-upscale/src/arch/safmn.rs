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

//! **SAFMN** — Spatially-Adaptive Feature Modulation (ICCV 2023).
//!
//! An efficient network that gets its receptive field from *pooling* rather
//! than from large kernels or attention. Each block splits the channels into
//! four groups, pools group `i` down by `2ⁱ`, runs one depthwise 3×3 there, and
//! resamples back — so the fourth group's 3×3 sees an effective 24×24 window at
//! a sixteenth of the cost. The concatenated result is a multiplicative gate on
//! the block input:
//!
//! ```text
//!   SAFM(x) = gelu(aggr(concat_i up_i(dw_i(pool_i(x_i))))) * x
//!   block   = x + SAFM(norm1(x));  x = x + CCM(norm2(x))
//! ```
//!
//! It is the cheapest thing in this crate that still competes on quality, and
//! it is what most of the recent lightweight community models train.
//!
//! # Why the tile size is constrained
//!
//! The pooling is `adaptive_max_pool2d` to `(h / 2ⁱ, w / 2ⁱ)`. Adaptive pooling
//! with a non-dividing size uses *ragged* windows — some 2 wide, some 3 — which
//! no fixed kernel reproduces. With four levels the deepest ratio is 8, so the
//! tile must be a multiple of 8; [`ModelConfig::size_multiple`] declares that
//! and the runner pads to it. Getting this wrong would not fail, it would
//! quietly pool a slightly different region than the reference — so
//! [`Builder::max_pool`] refuses the ragged case outright.

use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirNodeId;

use crate::config::ModelConfig;
use crate::nn::Builder;

/// The reference splits every block's channels into this many scales.
pub const N_LEVELS: usize = 4;

/// The deepest pooling ratio, and therefore the required tile multiple.
pub const SIZE_MULTIPLE: usize = 1 << (N_LEVELS - 1);

/// `LayerNorm` over channels, in the reference's `channels_first` form.
fn norm(b: &mut Builder, wm: &mut WeightMap, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    // eps is 1e-6 here, not torch's 1e-5 default.
    b.layer_norm_nchw(wm, prefix, x, 1e-6)
}

/// Spatially-Adaptive Feature Modulation.
fn safm(b: &mut Builder, wm: &mut WeightMap, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let [_, c, _, _] = b.dims4(x);
    ensure!(
        c % N_LEVELS == 0,
        "SAFM: {c} channels do not split into {N_LEVELS} levels"
    );
    let chunk = c / N_LEVELS;

    let mut parts = Vec::with_capacity(N_LEVELS);
    for i in 0..N_LEVELS {
        // `torch.chunk` along the channel axis.
        let xi = b.g().narrow_(x, 1, i * chunk, chunk);
        let r = 1usize << i;
        // Level 0 is the identity path: the reference skips pooling entirely
        // rather than pooling by 1, and interpolating back would resample.
        let pooled = b.max_pool(xi, r)?;
        let dw = b.conv(
            wm,
            &format!("{prefix}.mfr.{i}"),
            pooled,
            chunk,
            [3, 3],
            [1, 1],
            [1, 1],
            chunk,
            true,
        )?;
        parts.push(b.nearest_upsample(dw, r));
    }

    let cat = b.g().concat_(parts, 1);
    let aggr = b.conv1x1(wm, &format!("{prefix}.aggr"), cat, c)?;
    let gate = b.gelu(aggr);
    Ok(b.g().mul(gate, x))
}

/// Convolutional Channel Mixer — a 3×3 → GELU → 1×1 feed-forward.
///
/// `prefix` is the CCM module itself, which holds a `Sequential` also called
/// `ccm` — so the keys really do read `feats.0.ccm.ccm.0`.
fn ccm(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    hidden: usize,
) -> Result<HirNodeId> {
    let [_, c, _, _] = b.dims4(x);
    let h = b.conv3x3(wm, &format!("{prefix}.ccm.0"), x, hidden)?;
    let h = b.gelu(h);
    b.conv1x1(wm, &format!("{prefix}.ccm.2"), h, c)
}

/// Build SAFMN.
pub fn build(
    b: &mut Builder,
    wm: &mut WeightMap,
    cfg: &ModelConfig,
    dim: usize,
    n_blocks: usize,
    hidden: usize,
    x_in: HirNodeId,
) -> Result<HirNodeId> {
    let feat = b.conv3x3(wm, "to_feat", x_in, dim)?;

    let mut h = feat;
    for i in 0..n_blocks {
        let p = format!("feats.{i}");
        let n1 = norm(b, wm, &format!("{p}.norm1"), h)?;
        let a = safm(b, wm, &format!("{p}.safm"), n1)?;
        h = b.g().add(h, a);

        let n2 = norm(b, wm, &format!("{p}.norm2"), h)?;
        let f = ccm(b, wm, &format!("{p}.ccm"), n2, hidden)?;
        h = b.g().add(h, f);
    }
    // The trunk's own long skip, *outside* the block loop.
    let h = b.g().add(h, feat);

    let out = b.conv3x3(wm, "to_img.0", h, cfg.out_ch * cfg.scale * cfg.scale)?;
    b.pixel_shuffle(out, cfg.scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every level's pooled size must be exact, or the reference's ragged
    /// adaptive windows would be silently replaced by a uniform kernel.
    #[test]
    fn size_multiple_covers_every_level() {
        for i in 0..N_LEVELS {
            assert_eq!(SIZE_MULTIPLE % (1 << i), 0, "level {i} does not divide");
        }
        assert_eq!(SIZE_MULTIPLE, 8);
    }

    /// The effective receptive field of the deepest level's 3×3, in input
    /// pixels — the whole reason the architecture pools at all.
    #[test]
    fn deepest_level_sees_a_wide_window() {
        assert_eq!(3 * (1 << (N_LEVELS - 1)), 24);
    }
}
