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

//! **PLKSR** and **RealPLKSR** — Partial Large Kernel super-resolution.
//!
//! The idea is that a very large kernel (17×17 by default) buys long-range
//! context, but applying it to every channel is wasteful — so it is applied to
//! only the first `pdim = int(dim · split_ratio)` channels and the rest are
//! passed through untouched. Everything else in the block is cheap 3×3/1×1 work.
//!
//! Both variants share the skeleton
//!
//! ```text
//!   feats = conv3×3(3 → dim) → [PLKBlock] × n → conv3×3(dim → 3·r²)
//!   out   = pixel_shuffle(feats(x) + repeat_interleave(x, r²))
//! ```
//!
//! where the `repeat_interleave` + `pixel_shuffle` pair is exactly nearest
//! upsampling — the same residual-base trick Compact uses, spelled differently.
//!
//! # Where the two differ
//!
//! | | PLKSR | RealPLKSR |
//! |---|---|---|
//! | channel mixer | `CCM` / `ICCM` / `DCCM`, GELU | `DCCM` only, **Mish** |
//! | mixer key | `channe_mixer` *(sic)* | `channel_mixer` |
//! | block tail | — | `GroupNorm` or channel-first `LayerNorm` |
//! | `with_idt` | optional identity on the LK branch | — |
//! | upsampler | `PixelShuffle` | `PixelShuffle` or `DySample` |
//!
//! The misspelled `channe_mixer` is load-bearing: it is the key in every
//! released PLKSR checkpoint, and correcting it would simply fail to load.

use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirNodeId;

use crate::config::{ModelConfig, PlksrCcm, Upsampler};
use crate::nn::Builder;

/// The large-kernel branch: convolve the leading `pdim` channels, concatenate
/// the untouched remainder back on.
fn partial_large_kernel(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    dim: usize,
    pdim: usize,
    kernel_size: usize,
    with_idt: bool,
) -> Result<HirNodeId> {
    ensure!(
        pdim <= dim,
        "{prefix}: partial kernel width {pdim} exceeds block width {dim}"
    );
    if pdim == dim {
        let y = b.conv(
            wm,
            &format!("{prefix}.conv"),
            x,
            pdim,
            [kernel_size, kernel_size],
            [1, 1],
            [kernel_size / 2, kernel_size / 2],
            1,
            true,
        )?;
        return Ok(if with_idt { b.g().add(y, x) } else { y });
    }

    let head = b.g().narrow_(x, 1, 0, pdim);
    let tail = b.g().narrow_(x, 1, pdim, dim - pdim);
    let mut y = b.conv(
        wm,
        &format!("{prefix}.conv"),
        head,
        pdim,
        [kernel_size, kernel_size],
        [1, 1],
        [kernel_size / 2, kernel_size / 2],
        1,
        true,
    )?;
    if with_idt {
        y = b.g().add(y, head);
    }
    Ok(b.g().concat_(vec![y, tail], 1))
}

/// Element-wise attention: `x · sigmoid(conv3×3(x))`.
fn element_attention(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    dim: usize,
) -> Result<HirNodeId> {
    let f = b.conv3x3(wm, &format!("{prefix}.f.0"), x, dim)?;
    let gate = b.sigmoid(f);
    Ok(b.g().mul(x, gate))
}

/// Build PLKSR.
#[allow(clippy::too_many_arguments)]
pub fn build(
    b: &mut Builder,
    wm: &mut WeightMap,
    cfg: &ModelConfig,
    dim: usize,
    n_blocks: usize,
    kernel_size: usize,
    pdim: usize,
    ccm: PlksrCcm,
    use_ea: bool,
    with_idt: bool,
    x: HirNodeId,
) -> Result<HirNodeId> {
    let mut h = b.conv3x3(wm, "feats.0", x, dim)?;

    for i in 1..=n_blocks {
        let p = format!("feats.{i}");
        let skip = h;
        // Channel mixer. The three variants differ only in which of the two
        // convolutions is 3×3 and which is 1×1.
        let mixer = format!("{p}.channe_mixer");
        h = match ccm {
            PlksrCcm::Ccm => {
                let t = b.conv3x3(wm, &format!("{mixer}.0"), h, dim * 2)?;
                let t = b.gelu(t);
                b.conv1x1(wm, &format!("{mixer}.2"), t, dim)?
            }
            PlksrCcm::Iccm => {
                let t = b.conv1x1(wm, &format!("{mixer}.0"), h, dim * 2)?;
                let t = b.gelu(t);
                b.conv3x3(wm, &format!("{mixer}.2"), t, dim)?
            }
            PlksrCcm::Dccm => {
                let t = b.conv3x3(wm, &format!("{mixer}.0"), h, dim * 2)?;
                let t = b.gelu(t);
                b.conv3x3(wm, &format!("{mixer}.2"), t, dim)?
            }
        };
        h = partial_large_kernel(
            b,
            wm,
            &format!("{p}.lk"),
            h,
            dim,
            pdim,
            kernel_size,
            with_idt,
        )?;
        if use_ea {
            h = element_attention(b, wm, &format!("{p}.attn"), h, dim)?;
        }
        h = b.conv1x1(wm, &format!("{p}.refine"), h, dim)?;
        h = b.g().add(h, skip);
    }

    let r = cfg.scale;
    let last = n_blocks + 1;
    let h = b.conv3x3(wm, &format!("feats.{last}"), h, cfg.out_ch * r * r)?;
    let base = b.repeat_interleave_channels(x, r * r);
    let sum = b.g().add(h, base);
    b.pixel_shuffle(sum, r)
}

/// Build RealPLKSR.
#[allow(clippy::too_many_arguments)]
pub fn build_real(
    b: &mut Builder,
    wm: &mut WeightMap,
    cfg: &ModelConfig,
    dim: usize,
    n_blocks: usize,
    kernel_size: usize,
    pdim: usize,
    use_ea: bool,
    norm_groups: Option<usize>,
    upsampler: Upsampler,
    dysample_groups: usize,
    dysample_end_conv: bool,
    x: HirNodeId,
) -> Result<HirNodeId> {
    let mut h = b.conv3x3(wm, "feats.0", x, dim)?;

    for i in 1..=n_blocks {
        let p = format!("feats.{i}");
        let skip = h;
        // `layer_norm` (channel-first, pre-block) and `norm` (GroupNorm,
        // post-block) are mutually exclusive in the reference.
        if norm_groups.is_none() {
            h = b.layer_norm_nchw(wm, &format!("{p}.layer_norm"), h, 1e-6)?;
        }
        let mixer = format!("{p}.channel_mixer");
        let t = b.conv3x3(wm, &format!("{mixer}.0"), h, dim * 2)?;
        let t = b.mish(t);
        h = b.conv3x3(wm, &format!("{mixer}.2"), t, dim)?;

        h = partial_large_kernel(b, wm, &format!("{p}.lk"), h, dim, pdim, kernel_size, false)?;
        if use_ea {
            h = element_attention(b, wm, &format!("{p}.attn"), h, dim)?;
        }
        h = b.conv1x1(wm, &format!("{p}.refine"), h, dim)?;
        if let Some(groups) = norm_groups {
            h = b.group_norm(wm, &format!("{p}.norm"), h, groups, 1e-5)?;
        }
        h = b.g().add(h, skip);
    }

    let r = cfg.scale;
    // `nn.Dropout2d` occupies an index in `feats` but carries no parameters,
    // so the final convolution sits two past the last block, not one.
    let last = n_blocks + 2;
    let h = b.conv3x3(wm, &format!("feats.{last}"), h, cfg.out_ch * r * r)?;
    let base = b.repeat_interleave_channels(x, r * r);
    let sum = b.g().add(h, base);

    match upsampler {
        Upsampler::DySample => crate::arch::dysample::build(
            b,
            wm,
            "to_img",
            sum,
            cfg.out_ch,
            r,
            dysample_groups,
            dysample_end_conv,
        ),
        _ => b.pixel_shuffle(sum, r),
    }
}
