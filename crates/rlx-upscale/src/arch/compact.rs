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

//! **Compact** (`SRVGGNetCompact`) — the Real-ESRGAN compact net.
//!
//! A plain VGG-style stack: `conv → act` repeated, one widening conv, pixel
//! shuffle, plus a nearest-neighbour residual base. All convolution happens at
//! low resolution — nothing is ever convolved in HR space, which is the whole
//! reason it is fast enough for video.
//!
//! The `body` is an `nn.ModuleList` of alternating convs and activations, so
//! checkpoint indices interleave: conv at even indices, activation at odd. Only
//! PReLU contributes parameters, so a ReLU or LeakyReLU checkpoint simply has
//! gaps in the numbering — the indices still advance.

use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirNodeId;

use crate::config::{CompactAct, ModelConfig};
use crate::nn::Builder;

pub fn build(
    b: &mut Builder,
    wm: &mut WeightMap,
    cfg: &ModelConfig,
    num_feat: usize,
    num_conv: usize,
    act: CompactAct,
    x: HirNodeId,
) -> Result<HirNodeId> {
    let mut h = b.conv3x3(wm, "body.0", x, num_feat)?;
    h = activate(b, wm, act, "body.1", h)?;

    for i in 0..num_conv {
        let conv_idx = 2 + 2 * i;
        h = b.conv3x3(wm, &format!("body.{conv_idx}"), h, num_feat)?;
        h = activate(b, wm, act, &format!("body.{}", conv_idx + 1), h)?;
    }

    let r = cfg.scale;
    let last = 2 + 2 * num_conv;
    let h = b.conv3x3(wm, &format!("body.{last}"), h, cfg.out_ch * r * r)?;
    let shuffled = b.pixel_shuffle(h, r)?;

    // The network predicts a residual over the nearest-upsampled input, so a
    // randomly-initialized model is already an identity resampler.
    let base = b.nearest_upsample(x, r);
    Ok(b.g().add(shuffled, base))
}

fn activate(
    b: &mut Builder,
    wm: &mut WeightMap,
    act: CompactAct,
    prefix: &str,
    x: HirNodeId,
) -> Result<HirNodeId> {
    Ok(match act {
        CompactAct::Relu => b.relu(x),
        CompactAct::LeakyRelu => b.leaky_relu(x, 0.1),
        CompactAct::PRelu => b.prelu(wm, prefix, x)?,
    })
}
