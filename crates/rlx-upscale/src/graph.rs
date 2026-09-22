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

//! `ModelConfig` + weights + tile shape → an rlx graph.

use anyhow::{Context, Result, ensure};
use rlx_core::weight_map::WeightMap;
use std::collections::HashMap;

use crate::arch;
use crate::config::{ArchParams, ModelConfig};
use crate::nn::Builder;

/// A built graph and the host blobs its parameters refer to.
pub struct BuiltGraph {
    pub graph: rlx_ir::Graph,
    pub params: HashMap<String, Vec<f32>>,
    pub nodes: usize,
    /// Checkpoint tensors the graph never asked for. Non-empty is a strong
    /// signal that detection mis-read the architecture — the graph would still
    /// run and still produce an image, just not the right one.
    pub unused: Vec<String>,
}

/// Build `cfg`'s network for a `h × w` input tile.
pub fn build(cfg: &ModelConfig, wm: WeightMap, h: usize, w: usize) -> Result<BuiltGraph> {
    build_with(cfg, wm, h, w, Builder::new(cfg.arch.name()))
}

/// [`build`], but any weight the checkpoint lacks is synthesized from `seed`.
///
/// For exercising an architecture's shapes end to end without its checkpoint —
/// see [`Builder::with_autofill`]. Not for inference.
pub fn build_autofill(
    cfg: &ModelConfig,
    wm: WeightMap,
    h: usize,
    w: usize,
    seed: u64,
) -> Result<BuiltGraph> {
    build_with(cfg, wm, h, w, Builder::with_autofill(cfg.arch.name(), seed))
}

fn build_with(
    cfg: &ModelConfig,
    mut wm: WeightMap,
    h: usize,
    w: usize,
    mut b: Builder,
) -> Result<BuiltGraph> {
    ensure!(h > 0 && w > 0, "tile must be non-empty, got {h}×{w}");
    let multiple = cfg.size_multiple();
    ensure!(
        h.is_multiple_of(multiple) && w.is_multiple_of(multiple),
        "{} needs input dimensions that are multiples of {multiple}, got {h}×{w}",
        cfg.arch.name()
    );

    let x = b.input(cfg.in_ch, h, w);

    let out = match &cfg.params {
        ArchParams::Esrgan {
            num_filters,
            num_blocks,
            growth,
            plus,
            shuffle_factor,
        } => arch::esrgan::build(
            &mut b,
            &mut wm,
            cfg,
            *num_filters,
            *num_blocks,
            *growth,
            *plus,
            *shuffle_factor,
            x,
        ),
        ArchParams::Compact {
            num_feat,
            num_conv,
            act,
        } => arch::compact::build(&mut b, &mut wm, cfg, *num_feat, *num_conv, *act, x),
        ArchParams::Span {
            feature_channels,
            norm,
            img_range,
            rgb_mean,
        } => arch::span::build(
            &mut b,
            &mut wm,
            cfg,
            *feature_channels,
            *norm,
            *img_range,
            *rgb_mean,
            x,
        ),
        ArchParams::SpanV2 {
            feature_channels,
            conv_bias,
        } => arch::span::build_v2(&mut b, &mut wm, cfg, *feature_channels, *conv_bias, x),
        ArchParams::Plksr {
            dim,
            n_blocks,
            kernel_size,
            pdim,
            ccm,
            use_ea,
            with_idt,
        } => arch::plksr::build(
            &mut b,
            &mut wm,
            cfg,
            *dim,
            *n_blocks,
            *kernel_size,
            *pdim,
            *ccm,
            *use_ea,
            *with_idt,
            x,
        ),
        ArchParams::RealPlksr {
            dim,
            n_blocks,
            kernel_size,
            pdim,
            use_ea,
            norm_groups,
            upsampler,
            dysample_groups,
            dysample_end_conv,
        } => arch::plksr::build_real(
            &mut b,
            &mut wm,
            cfg,
            *dim,
            *n_blocks,
            *kernel_size,
            *pdim,
            *use_ea,
            *norm_groups,
            *upsampler,
            *dysample_groups,
            *dysample_end_conv,
            x,
        ),
        ArchParams::Swin(p) => arch::swin::build(&mut b, &mut wm, cfg, p, x),
        ArchParams::Dat(p) => arch::dat::build(&mut b, &mut wm, cfg, p, x),
        ArchParams::MambaIr(p) => arch::mambair::build(&mut b, &mut wm, cfg, p, x),
        ArchParams::Safmn {
            dim,
            n_blocks,
            hidden,
        } => arch::safmn::build(&mut b, &mut wm, cfg, *dim, *n_blocks, *hidden, x),
        ArchParams::OmniSr {
            num_feat,
            res_num,
            block_num,
            window_size,
        } => arch::omnisr::build(
            &mut b,
            &mut wm,
            cfg,
            *num_feat,
            *res_num,
            *block_num,
            *window_size,
            x,
        ),
        ArchParams::RealCugan { variant, pro } => {
            arch::realcugan::build(&mut b, &mut wm, cfg, *variant, *pro, x)
        }
    }
    .with_context(|| format!("building {} for a {h}×{w} tile", cfg.summary()))?;

    let unused = leftover(&wm);
    let (graph, params, nodes) = b.finish(out)?;
    Ok(BuiltGraph {
        graph,
        params,
        nodes,
        unused,
    })
}

/// Tensors the build never consumed, excluding the bookkeeping buffers a
/// state dict carries that are not weights.
fn leftover(wm: &WeightMap) -> Vec<String> {
    let mut v: Vec<String> = wm
        .keys()
        .filter(|k| {
            // `relative_position_index` and friends are index tables recomputed
            // from the window size; `no_norm` is a presence flag. None are
            // parameters, and all legitimately go unread.
            !k.ends_with("relative_position_index")
                && !k.ends_with("relative_position_index_SA")
                && !k.ends_with("relative_position_index_OCA")
                // Shift masks (`attn_mask`, DAT's `attn_mask_0` / `_1`) are a
                // function of the tile geometry, recomputed per tile size.
                && !k.contains("attn_mask")
                && !k.contains("rpi_sa")
                && !k.contains("rpi_oca")
                && !k.ends_with("no_norm")
                // DAT's displacement grid, recomputed by `rpe_biases`.
                && !k.ends_with("rpe_biases")
                // BatchNorm's update counter: bookkeeping, not a parameter.
                && !k.ends_with("num_batches_tracked")
                && !k.ends_with("total_ops")
                && !k.ends_with("total_params")
        })
        .map(|k| k.to_string())
        .collect();
    v.sort();
    v
}
