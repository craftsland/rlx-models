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

//! **DySample** — "Learning to Upsample by Learning to Sample"
//! ([arXiv:2308.15085](https://arxiv.org/abs/2308.15085)).
//!
//! A learned resampler that replaces pixel shuffle in the newer RealPLKSR
//! models (`*_dysample`). Instead of assigning each sub-pixel a fixed source,
//! it predicts a *sampling offset* per sub-pixel and reads the low-resolution
//! feature map with bilinear `grid_sample`. That removes the blockiness pixel
//! shuffle can leave on diagonal edges.
//!
//! ```text
//!   offset = offset_conv(x) · sigmoid(scope_conv(x)) · 0.5 + init_pos
//!   coords = 2·(pixel_centre + offset) / [W, H] − 1        (normalized to [−1, 1])
//!   out    = grid_sample(x, pixel_shuffle(coords), bilinear, border)
//! ```
//!
//! # Channel layout
//!
//! The offset convolution emits `2·groups·scale²` channels laid out as
//! `(axis, group, sub-pixel)` — axis-major, then group, then the `scale²`
//! sub-pixel slot. That ordering is what makes the subsequent pixel shuffle
//! land each group's `scale²` offsets on the right sub-pixels, and it is the
//! only reason the reference's `view(B, 2, -1, H, W)` is correct. Getting it
//! backwards yields a plausible image with the offsets transposed.
//!
//! `init_pos` is a registered buffer, so it is read from the checkpoint when
//! present; the fallback recomputes it from `(scale, groups)`.

use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirNodeId;
use rlx_ir::hir::{GridMode, GridPad};

use crate::nn::Builder;

/// `init_pos[a, t, i, q] = h[q]` for the x axis and `h[i]` for the y axis,
/// where `h[k] = ((1 − scale)/2 + k) / scale` — the sub-pixel centres of one
/// upsampled cell, expressed as an offset from the source pixel centre.
pub fn init_pos(scale: usize, groups: usize) -> Vec<f32> {
    let s = scale as f32;
    let h: Vec<f32> = (0..scale)
        .map(|k| ((1.0 - s) / 2.0 + k as f32) / s)
        .collect();
    let mut out = Vec::with_capacity(2 * groups * scale * scale);
    for axis in 0..2 {
        for _t in 0..groups {
            for i in 0..scale {
                for q in 0..scale {
                    out.push(if axis == 0 { h[q] } else { h[i] });
                }
            }
        }
    }
    out
}

/// Build a `DySample` head. `x` is `[1, in_channels, H, W]`; the result is
/// `[1, out_ch, H·scale, W·scale]` (or `in_channels` wide if `end_conv` is off).
#[allow(clippy::too_many_arguments)]
pub fn build(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    out_ch: usize,
    scale: usize,
    groups: usize,
    end_conv: bool,
) -> Result<HirNodeId> {
    let [n, in_c, h, w] = b.dims4(x);
    ensure!(
        in_c >= groups && in_c % groups == 0,
        "dysample: {in_c} channels do not split into {groups} groups"
    );
    let s2 = scale * scale;
    let off_c = 2 * groups * s2;

    let offset = b.conv(
        wm,
        &format!("{prefix}.offset"),
        x,
        off_c,
        [1, 1],
        [1, 1],
        [0, 0],
        1,
        true,
    )?;
    let scope = b.conv(
        wm,
        &format!("{prefix}.scope"),
        x,
        off_c,
        [1, 1],
        [1, 1],
        [0, 0],
        1,
        false,
    )?;
    let gate = b.sigmoid(scope);
    let scaled = b.g().mul(offset, gate);
    let half = b.scalar_like("dysample_half", 0.5, scaled);
    let scaled = b.g().mul(scaled, half);

    let init_key = format!("{prefix}.init_pos");
    let pos = if b.has(wm, &init_key) {
        b.take_as(wm, &init_key, &[1, off_c, 1, 1])?
    } else {
        b.constant(
            "dysample_init_pos",
            init_pos(scale, groups),
            &[1, off_c, 1, 1],
        )
    };
    let offset = b.g().add(scaled, pos);

    // `coords = 2·(centre + offset)/norm − 1` is refactored into one multiply
    // and one add so the pixel centres and the normalizer collapse into two
    // host-side constants instead of four graph ops.
    //
    // Axis 0 is the *width* coordinate, matching `grid_sample`'s (x, y) order.
    let mut pre = vec![0.0f32; 2 * h * w];
    for yy in 0..h {
        for xx in 0..w {
            pre[yy * w + xx] = 2.0 * (xx as f32 + 0.5) / w as f32 - 1.0;
            pre[h * w + yy * w + xx] = 2.0 * (yy as f32 + 0.5) / h as f32 - 1.0;
        }
    }
    let pre = b.constant("dysample_pre", pre, &[1, 2, 1, h, w]);
    let inv = b.constant(
        "dysample_inv",
        vec![2.0 / w as f32, 2.0 / h as f32],
        &[1, 2, 1, 1, 1],
    );

    let off5 = b.g().reshape_(
        offset,
        vec![n as i64, 2, (groups * s2) as i64, h as i64, w as i64],
    );
    let norm = b.g().mul(off5, inv);
    let coords = b.g().add(norm, pre);

    // [B, 2, G·s², H, W] → [B, 2G, sH, sW] → [B·G, sH, sW, 2]
    let flat = b
        .g()
        .reshape_(coords, vec![n as i64, off_c as i64, h as i64, w as i64]);
    let shuffled = b.pixel_shuffle(flat, scale)?;
    let (oh, ow) = (h * scale, w * scale);
    let split = b.g().reshape_(
        shuffled,
        vec![n as i64, 2, groups as i64, oh as i64, ow as i64],
    );
    let perm = b.g().transpose_(split, vec![0, 2, 3, 4, 1]);
    let grid = b
        .g()
        .reshape_(perm, vec![(n * groups) as i64, oh as i64, ow as i64, 2]);

    let xg = b.g().reshape_(
        x,
        vec![
            (n * groups) as i64,
            (in_c / groups) as i64,
            h as i64,
            w as i64,
        ],
    );
    let sampled = b
        .g()
        .grid_sample2d(xg, grid, GridMode::Bilinear, GridPad::Border, false);
    let out = b
        .g()
        .reshape_(sampled, vec![n as i64, in_c as i64, oh as i64, ow as i64]);

    if end_conv {
        b.conv1x1(wm, &format!("{prefix}.end_conv"), out, out_ch)
    } else {
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scale 2, 1 group: sub-pixel centres are ±¼ of a source pixel, with the
    /// x offset varying along the column index and y along the row index. A
    /// transposed layout would give the same *set* of numbers in the wrong
    /// slots, so this pins the order.
    #[test]
    fn init_pos_places_subpixel_centres() {
        let p = init_pos(2, 1);
        assert_eq!(p.len(), 2 * 4);
        // x axis: h[q] for (i, q) in row-major order → [-0.25, 0.25, -0.25, 0.25]
        assert_eq!(&p[0..4], &[-0.25, 0.25, -0.25, 0.25]);
        // y axis: h[i] → [-0.25, -0.25, 0.25, 0.25]
        assert_eq!(&p[4..8], &[-0.25, -0.25, 0.25, 0.25]);
    }

    #[test]
    fn init_pos_repeats_per_group() {
        let p = init_pos(2, 3);
        assert_eq!(p.len(), 2 * 3 * 4);
        // Each group is an identical copy within its axis block.
        assert_eq!(&p[0..4], &p[4..8]);
        assert_eq!(&p[0..4], &p[8..12]);
        // The y block starts after all three x groups.
        assert_eq!(&p[12..16], &[-0.25, -0.25, 0.25, 0.25]);
    }
}
