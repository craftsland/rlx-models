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

//! **ESRGAN** / **RRDBNet** — the most widely deployed super-resolution
//! architecture there is.
//!
//! Nothing in the community zoos comes close in volume: `4x-UltraSharp`,
//! `4x_foolhardy_Remacri`, `4x_NMKD-Siax`, BSRGAN, RealSR and most of
//! OpenModelDB are all this one network with different weights. Real-ESRGAN's
//! repository is the most-starred in the field.
//!
//! The network is residual-in-residual and nothing else — no attention, no
//! normalization:
//!
//! ```text
//!   RDB   = 5 convs, each fed the concatenation of every earlier output
//!           (growth `gc`), scaled by 0.2 and added to its input
//!   RRDB  = 3 RDBs in series, scaled by 0.2 and added to its input
//!   trunk = nb RRDBs, one conv, added to the stem   (the "shortcut block")
//!   tail  = [nearest ×2 → conv → lrelu] per octave, then two more convs
//! ```
//!
//! # Key layouts
//!
//! Checkpoints come in three shapes and the reader normalizes all of them to
//! the **old** one before this module sees them — see
//! [`crate::weights::esrgan_to_old_arch`]. The old layout is a flattened
//! `nn.Sequential`, so layer *indices* move with the scale factor:
//!
//! | | ×1 | ×2 | ×4 | ×8 |
//! |---|---|---|---|---|
//! | upconv convs | — | 3 | 3, 6 | 3, 6, 9 |
//! | `conv_hr` | 2 | 5 | 8 | 11 |
//! | `conv_last` | 4 | 7 | 10 | 13 |
//!
//! which is why the scale is recovered from the sequence length rather than
//! from any tensor's shape.
//!
//! # Two variants
//!
//! * **ESRGAN+** adds a 1×1 path inside each RDB (`conv1x1`), detected by that
//!   key's presence.
//! * **Unshuffled** Real-ESRGAN ×1/×2 models feed the trunk a
//!   `pixel_unshuffle`d input so it always runs at ×4, then divide the factor
//!   back out. Detected by the stem taking 4× or 16× the output channels.

use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirNodeId;

use crate::config::ModelConfig;
use crate::nn::Builder;

/// The slope every activation in this network uses.
const SLOPE: f32 = 0.2;

/// The residual scaling applied at both block levels.
const RESIDUAL_SCALE: f32 = 0.2;

/// One `ResidualDenseBlock_5C`: five convolutions, each seeing every earlier
/// output concatenated onto the block input.
fn residual_dense_block(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    nf: usize,
    gc: usize,
    plus: bool,
) -> Result<HirNodeId> {
    let mut feats = vec![x];
    let mut x2 = None;

    for i in 1..=5usize {
        let cat = if feats.len() == 1 {
            feats[0]
        } else {
            b.g().concat_(feats.clone(), 1)
        };
        // The fifth convolution collapses back to `nf` and, in `CNA` mode, is
        // the one without an activation.
        let out_c = if i == 5 { nf } else { gc };
        let mut y = b.conv3x3(wm, &format!("{prefix}.conv{i}.0"), cat, out_c)?;
        if i < 5 {
            y = b.leaky_relu(y, SLOPE);
        }

        if plus {
            // ESRGAN+ adds a 1×1 of the block input to the second output, and
            // that second output again to the fourth.
            if i == 2 {
                let side = b.conv1x1(wm, &format!("{prefix}.conv1x1"), x, gc)?;
                y = b.g().add(y, side);
                x2 = Some(y);
            } else if i == 4 {
                let prev = x2.expect("conv2 runs before conv4");
                y = b.g().add(y, prev);
            }
        }
        feats.push(y);
    }

    let last = *feats.last().expect("five convolutions always push");
    let s = b.scalar_like("rdb_scale", RESIDUAL_SCALE, last);
    let scaled = b.g().mul(last, s);
    Ok(b.g().add(scaled, x))
}

/// One `RRDB`: three dense blocks in series, residual-scaled.
fn rrdb(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    nf: usize,
    gc: usize,
    plus: bool,
) -> Result<HirNodeId> {
    let mut h = x;
    for j in 1..=3usize {
        h = residual_dense_block(b, wm, &format!("{prefix}.RDB{j}"), h, nf, gc, plus)?;
    }
    let s = b.scalar_like("rrdb_scale", RESIDUAL_SCALE, h);
    let scaled = b.g().mul(h, s);
    Ok(b.g().add(scaled, x))
}

/// Build ESRGAN / RRDBNet.
#[allow(clippy::too_many_arguments)]
pub fn build(
    b: &mut Builder,
    wm: &mut WeightMap,
    cfg: &ModelConfig,
    num_filters: usize,
    num_blocks: usize,
    growth: usize,
    plus: bool,
    shuffle_factor: usize,
    x_in: HirNodeId,
) -> Result<HirNodeId> {
    // Unshuffled variants run the trunk at a fixed ×4 and hand back the
    // factor afterwards, so the trunk's scale is the *model* scale times it.
    let trunk_scale = cfg.scale * shuffle_factor;
    ensure!(
        trunk_scale.is_power_of_two() || trunk_scale == 3,
        "ESRGAN supports 2^n and 3, not a trunk scale of {trunk_scale}"
    );

    let x = if shuffle_factor > 1 {
        b.pixel_unshuffle(x_in, shuffle_factor)?
    } else {
        x_in
    };

    // ── stem ─────────────────────────────────────────────────────────
    let stem = b.conv3x3(wm, "model.0", x, num_filters)?;

    // ── trunk: nb RRDBs, one convolution, added back to the stem ─────
    let mut h = stem;
    for i in 0..num_blocks {
        h = rrdb(
            b,
            wm,
            &format!("model.1.sub.{i}"),
            h,
            num_filters,
            growth,
            plus,
        )?;
    }
    let h = b.conv3x3(wm, &format!("model.1.sub.{num_blocks}"), h, num_filters)?;
    let mut h = b.g().add(stem, h);

    // ── tail ─────────────────────────────────────────────────────────
    // The flattened `Sequential` puts each upconv's convolution three indices
    // apart, because `[Upsample, Conv2d, LeakyReLU]` is three modules.
    let octaves = if trunk_scale == 3 {
        1
    } else {
        trunk_scale.trailing_zeros() as usize
    };
    for stage in 1..=octaves {
        let factor = if trunk_scale == 3 { 3 } else { 2 };
        let up = b.nearest_upsample(h, factor);
        let conv = b.conv3x3(wm, &format!("model.{}", stage * 3), up, num_filters)?;
        h = b.leaky_relu(conv, SLOPE);
    }

    let last_up = octaves * 3;
    let hr = b.conv3x3(wm, &format!("model.{}", last_up + 2), h, num_filters)?;
    let hr = b.leaky_relu(hr, SLOPE);
    b.conv3x3(wm, &format!("model.{}", last_up + 4), hr, cfg.out_ch)
}

#[cfg(test)]
mod tests {
    /// The index of every tail convolution, for a given number of ×2 octaves.
    /// Pinned because these move with the scale and a silent off-by-three
    /// would read a real checkpoint's `conv_hr` as its `conv_last`.
    fn tail_indices(octaves: usize) -> (Vec<usize>, usize, usize) {
        let ups: Vec<usize> = (1..=octaves).map(|s| s * 3).collect();
        (ups, octaves * 3 + 2, octaves * 3 + 4)
    }

    #[test]
    fn tail_layout_matches_the_flattened_sequential() {
        // ×1: no upconv, conv_hr at 2, conv_last at 4 (seq len 5).
        assert_eq!(tail_indices(0), (vec![], 2, 4));
        // ×2: one upconv at 3 (seq len 8).
        assert_eq!(tail_indices(1), (vec![3], 5, 7));
        // ×4: the common case (seq len 11).
        assert_eq!(tail_indices(2), (vec![3, 6], 8, 10));
        // ×8 (seq len 14).
        assert_eq!(tail_indices(3), (vec![3, 6, 9], 11, 13));
    }

    /// `scale = 2^((seq_len − 5) / 3)` is how the scale is recovered, so the
    /// two must agree for every case above.
    #[test]
    fn sequence_length_recovers_the_scale() {
        for octaves in 0..4usize {
            let (_, _, last) = tail_indices(octaves);
            let seq_len = last + 1;
            assert_eq!((seq_len - 5) / 3, octaves);
        }
    }
}
