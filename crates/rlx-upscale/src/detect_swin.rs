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

//! Detection for the tier-2 transformers.
//!
//! The Swin lineage shares most of its key layout, so the three families are
//! separated by one distinguishing key each:
//!
//! | family | tell |
//! |---|---|
//! | HAT | `layers.0.residual_group.overlap_attn.*` |
//! | DRCT | `layers.0.swin1.*` and `layers.0.adjust1.*` |
//! | SwinIR | `layers.0.residual_group.blocks.0.*` and neither of the above |
//!
//! Window size falls out of the relative-position bias table, which has exactly
//! `(2W−1)²` rows; HAT's overlap ratio falls out of its second table, which has
//! `(W + W_ext − 1)²`. Both are solved rather than guessed.

use anyhow::{Result, ensure};

use crate::config::{
    Arch, ArchParams, DatParams, HatParams, MambaIrParams, ModelConfig, ResiConnection, SwinParams,
    Upsampler,
};
use crate::detect::{max_index, resi_connection, scale_from_pixelshuffle, shape, swin_upsampler};
use crate::weights::Checkpoint;

/// Solve `(2W−1)² = rows` for the window size.
fn window_from_table(rows: usize) -> Result<usize> {
    let side = (rows as f64).sqrt().round() as usize;
    ensure!(
        side * side == rows && side % 2 == 1,
        "a relative position bias table with {rows} rows is not (2W−1)² for any integer W"
    );
    Ok(side.div_ceil(2))
}

/// Try the Swin lineage. `Ok(None)` means "not one of these", which lets the
/// caller fall through to the next family rather than fail.
pub fn detect(ck: &Checkpoint) -> Result<Option<ModelConfig>> {
    // MambaIRv2 nests its blocks under `residual_group.layers`, not
    // `residual_group.blocks`, so it cannot collide with SwinIR or HAT — but
    // check it first anyway, since only it has the routing dictionary.
    if ck.contains("layers.0.residual_group.layers.0.assm.embeddingB.weight") {
        return detect_mambair(ck).map(Some);
    }
    if ck.contains("layers.0.swin1.attn.relative_position_bias_table") {
        return detect_drct(ck).map(Some);
    }
    if ck.contains("layers.0.residual_group.blocks.0.attn.relative_position_bias_table") {
        let hat = ck.contains("layers.0.residual_group.overlap_attn.relative_position_bias_table");
        return detect_swinir_or_hat(ck, hat, false).map(Some);
    }
    // Swin2SR has *no* bias table at all: V2 generates the bias from an MLP, so
    // the tell is that MLP and the coordinate table feeding it.
    if ck.contains("layers.0.residual_group.blocks.0.attn.cpb_mlp.2.weight") {
        ensure!(
            !ck.contains("conv_aux.weight"),
            "this is a Swin2SR `pixelshuffle_aux` model (it has conv_aux / \
             conv_bicubic). That tail takes a second, bicubically upscaled input \
             alongside the image and emits an auxiliary output, so it does not fit \
             a single-image-in, single-image-out runner and is not supported."
        );
        return detect_swinir_or_hat(ck, false, true).map(Some);
    }
    Ok(None)
}

/// SwinIR, HAT and Swin2SR share an outer structure and differ in what states
/// the per-group head count and window size.
///
/// V1 reads both off the `relative_position_bias_table` (`[(2ws−1)², nH]`); V2
/// has no such table and reads the heads from `cpb_mlp.2` (`[nH, 512]`) and the
/// window from `relative_coords_table` (`[1, 2ws−1, 2ws−1, 2]`).
fn detect_swinir_or_hat(ck: &Checkpoint, is_hat: bool, v2: bool) -> Result<ModelConfig> {
    let first = shape(ck, "conv_first.weight")?;
    let embed_dim = first[0];
    let in_ch = first[1];

    // Walk groups until one is missing; within a group, blocks until one is.
    let mut depths = Vec::new();
    let mut num_heads = Vec::new();
    let mut gi = 0usize;
    loop {
        let table = if v2 {
            format!("layers.{gi}.residual_group.blocks.0.attn.cpb_mlp.2.weight")
        } else {
            format!("layers.{gi}.residual_group.blocks.0.attn.relative_position_bias_table")
        };
        let Some(t) = ck.shape_of(&table) else { break };
        // `[nH, 512]` for V2's MLP, `[(2ws−1)², nH]` for V1's table.
        num_heads.push(if v2 { t[0] } else { t[1] });
        let mut depth = 0usize;
        while ck.contains(&format!(
            "layers.{gi}.residual_group.blocks.{depth}.attn.qkv.weight"
        )) {
            depth += 1;
        }
        depths.push(depth);
        gi += 1;
    }
    ensure!(!depths.is_empty(), "no residual groups found");

    let window_size = if v2 {
        let t = shape(
            ck,
            "layers.0.residual_group.blocks.0.attn.relative_coords_table",
        )?;
        ensure!(
            t.len() == 4 && t[1] == t[2] && t[3] == 2,
            "Swin2SR coordinate table should be [1, 2ws−1, 2ws−1, 2], got {t:?}"
        );
        ensure!(t[1] % 2 == 1, "coordinate table span {} is not 2ws−1", t[1]);
        t[1].div_ceil(2)
    } else {
        let table = shape(
            ck,
            "layers.0.residual_group.blocks.0.attn.relative_position_bias_table",
        )?;
        window_from_table(table[0])?
    };

    let fc1 = shape(ck, "layers.0.residual_group.blocks.0.mlp.fc1.weight")?;
    let mlp_ratio = fc1[0] as f32 / embed_dim as f32;

    let (upsampler, scale, out_ch) = swin_upsampler(ck)?;
    let num_feat = ck
        .shape_of("conv_before_upsample.0.weight")
        .map_or(64, |s| s[0]);

    let hat = if is_hat {
        let cab0 = shape(
            ck,
            "layers.0.residual_group.blocks.0.conv_block.cab.0.weight",
        )?;
        let squeeze = shape(
            ck,
            "layers.0.residual_group.blocks.0.conv_block.cab.3.attention.1.weight",
        )?;
        let oca = shape(
            ck,
            "layers.0.residual_group.overlap_attn.relative_position_bias_table",
        )?;
        // (W + W_ext − 1)² = rows
        let span = (oca[0] as f64).sqrt().round() as usize;
        ensure!(
            span * span == oca[0],
            "HAT overlap bias table has {} rows, not a perfect square",
            oca[0]
        );
        let ext = span + 1 - window_size;
        ensure!(
            ext >= window_size,
            "HAT overlap window ({ext}) is smaller than the base window ({window_size})"
        );
        Some(HatParams {
            compress_ratio: embed_dim / cab0[0].max(1),
            squeeze_factor: embed_dim / squeeze[0].max(1),
            // Not represented in the weights; the reference default.
            conv_scale: 0.01,
            overlap_ratio: (ext - window_size) as f32 / window_size as f32,
        })
    } else {
        None
    };

    Ok(ModelConfig {
        arch: if is_hat {
            Arch::Hat
        } else if v2 {
            Arch::Swin2Sr
        } else {
            Arch::SwinIr
        },
        scale,
        in_ch,
        out_ch,
        params: ArchParams::Swin(SwinParams {
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            // V2 splits the bias into separate `q_bias` / `v_bias` parameters
            // and leaves `k` unbiased, so the V1 key is absent even when there
            // is a bias.
            qkv_bias: ck.contains("layers.0.residual_group.blocks.0.attn.qkv.bias"),
            v2,
            patch_norm: ck.contains("patch_embed.norm.weight"),
            resi_connection: resi_connection(ck),
            upsampler,
            num_feat,
            img_range: 1.0,
            hat,
            drct_gc: None,
        }),
    })
}

fn detect_drct(ck: &Checkpoint) -> Result<ModelConfig> {
    let first = shape(ck, "conv_first.weight")?;
    let embed_dim = first[0];
    let in_ch = first[1];

    let mut depths = Vec::new();
    let mut num_heads = Vec::new();
    let mut gi = 0usize;
    while let Some(t) = ck.shape_of(&format!(
        "layers.{gi}.swin1.attn.relative_position_bias_table"
    )) {
        num_heads.push(t[1]);
        // An RDG is always five blocks; record it explicitly rather than
        // hard-coding 5 at the build site.
        let mut depth = 0usize;
        while ck.contains(&format!("layers.{gi}.swin{}.attn.qkv.weight", depth + 1)) {
            depth += 1;
        }
        ensure!(
            depth == 5,
            "DRCT group {gi} has {depth} blocks; an RDG has exactly 5"
        );
        depths.push(depth);
        gi += 1;
    }
    ensure!(!depths.is_empty(), "no RDG groups found");

    let table = shape(ck, "layers.0.swin1.attn.relative_position_bias_table")?;
    let window_size = window_from_table(table[0])?;

    let fc1 = shape(ck, "layers.0.swin1.mlp.fc1.weight")?;
    let mlp_ratio = fc1[0] as f32 / embed_dim as f32;

    let gc = shape(ck, "layers.0.adjust1.weight")?[0];
    let (upsampler, scale, out_ch) = swin_upsampler(ck)?;
    let num_feat = ck
        .shape_of("conv_before_upsample.0.weight")
        .map_or(64, |s| s[0]);

    Ok(ModelConfig {
        arch: Arch::Drct,
        scale,
        in_ch,
        out_ch,
        params: ArchParams::Swin(SwinParams {
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias: ck.contains("layers.0.swin1.attn.qkv.bias"),
            patch_norm: ck.contains("patch_embed.norm.weight"),
            // An RDG closes with `x5 · 0.2 + x`, no convolution.
            resi_connection: ResiConnection::Identity,
            upsampler,
            num_feat,
            img_range: 1.0,
            hat: None,
            drct_gc: Some(gc),
            // DRCT has no Swin V2 variant.
            v2: false,
        }),
    })
}

fn detect_mambair(ck: &Checkpoint) -> Result<ModelConfig> {
    let first = shape(ck, "conv_first.weight")?;
    let embed_dim = first[0];
    let in_ch = first[1];

    let mut depths = Vec::new();
    let mut num_heads = Vec::new();
    let mut gi = 0usize;
    loop {
        let table =
            format!("layers.{gi}.residual_group.layers.0.win_mhsa.relative_position_bias_table");
        let Some(t) = ck.shape_of(&table) else { break };
        num_heads.push(t[1]);
        let depth = (0..)
            .take_while(|d| {
                ck.contains(&format!(
                    "layers.{gi}.residual_group.layers.{d}.norm1.weight"
                ))
            })
            .count();
        ensure!(depth > 0, "MambaIRv2 group {gi} has no layers");
        depths.push(depth);
        gi += 1;
    }
    ensure!(!depths.is_empty(), "no ASSB groups found");

    let table = shape(
        ck,
        "layers.0.residual_group.layers.0.win_mhsa.relative_position_bias_table",
    )?;
    let window_size = window_from_table(table[0])?;

    // `embeddingA` is `[inner_rank, d_state]` and `embeddingB` is
    // `[num_tokens, inner_rank]`, so the routing dictionary is fully described
    // by the two together.
    let ea = shape(ck, "layers.0.residual_group.layers.0.embeddingA.weight")?;
    let eb = shape(
        ck,
        "layers.0.residual_group.layers.0.assm.embeddingB.weight",
    )?;
    ensure!(
        ea[0] == eb[1],
        "routing dictionary is inconsistent: embeddingA is {ea:?}, embeddingB is {eb:?}"
    );

    let fc1 = shape(ck, "layers.0.residual_group.layers.0.convffn1.fc1.weight")?;
    let dw = shape(
        ck,
        "layers.0.residual_group.layers.0.convffn1.dwconv.depthwise_conv.0.weight",
    )?;

    let (upsampler, scale, out_ch) = swin_upsampler(ck)?;
    let num_feat = ck
        .shape_of("conv_before_upsample.0.weight")
        .map_or(64, |s| s[0]);

    Ok(ModelConfig {
        arch: Arch::MambaIrV2,
        scale,
        in_ch,
        out_ch,
        params: ArchParams::MambaIr(MambaIrParams {
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio: fc1[0] as f32 / embed_dim as f32,
            d_state: ea[1],
            num_tokens: eb[0],
            inner_rank: ea[0],
            convffn_kernel_size: dw[2],
            qkv_bias: ck.contains("layers.0.residual_group.layers.0.wqkv.bias"),
            patch_norm: ck.contains("patch_embed.norm.weight"),
            resi_connection: resi_connection(ck),
            upsampler,
            num_feat,
            img_range: 1.0,
        }),
    })
}

/// DAT. Separate from the Swin lineage: different block layout, rectangular
/// windows, and a dynamic position bias MLP instead of a lookup table.
pub fn detect_dat(ck: &Checkpoint) -> Result<Option<ModelConfig>> {
    if !ck.contains("layers.0.blocks.0.attn.attns.0.pos.pos3.2.weight") {
        return Ok(None);
    }
    let first = shape(ck, "conv_first.weight")?;
    let embed_dim = first[0];
    let in_ch = first[1];

    let mut depths = Vec::new();
    let mut num_heads = Vec::new();
    let mut gi = 0usize;
    while ck.contains(&format!("layers.{gi}.blocks.0.norm1.weight")) {
        let depth = (0..)
            .take_while(|d| ck.contains(&format!("layers.{gi}.blocks.{d}.norm1.weight")))
            .count();
        depths.push(depth);
        // `pos3` emits one bias per head in the *spatial* half of the block.
        let pos = shape(
            ck,
            &format!("layers.{gi}.blocks.0.attn.attns.0.pos.pos3.2.weight"),
        )?;
        num_heads.push(pos[0] * 2);
        gi += 1;
    }
    ensure!(!depths.is_empty(), "no DAT groups found");

    let sgfn = shape(ck, "layers.0.blocks.0.ffn.fc1.weight")?;
    let expansion_factor = sgfn[0] as f32 / embed_dim as f32;

    let (upsampler, scale, out_ch) = swin_upsampler(ck)?;
    let _ = scale_from_pixelshuffle(1, 1);
    let _ = max_index(ck, "layers.", ".conv.weight");

    Ok(Some(ModelConfig {
        arch: Arch::Dat,
        scale,
        in_ch,
        out_ch,
        params: ArchParams::Dat(DatParams {
            embed_dim,
            depths,
            num_heads,
            // Not recoverable from the weights alone; DAT's released configs
            // all use 8×32, swapped on odd blocks.
            split_size: [8, 32],
            expansion_factor,
            qkv_bias: ck.contains("layers.0.blocks.0.attn.qkv.bias"),
            resi_connection: resi_connection(ck),
            upsampler: match upsampler {
                Upsampler::DySample => Upsampler::PixelShuffle,
                other => other,
            },
            img_range: 1.0,
        }),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn ck(entries: &[(&str, Vec<usize>)]) -> Checkpoint {
        let mut t = HashMap::new();
        for (k, s) in entries {
            let n: usize = s.iter().product();
            t.insert(k.to_string(), (vec![0.0; n], s.clone()));
        }
        Checkpoint::from_tensors(t)
    }

    /// A ×2 SwinIR "classical SR": one upsample stage, 1conv residual.
    fn swinir_keys(window: usize, heads: usize, dim: usize) -> Vec<(&'static str, Vec<usize>)> {
        let rows = (2 * window - 1) * (2 * window - 1);
        vec![
            ("conv_first.weight", vec![dim, 3, 3, 3]),
            (
                "layers.0.residual_group.blocks.0.attn.relative_position_bias_table",
                vec![rows, heads],
            ),
            (
                "layers.0.residual_group.blocks.0.attn.qkv.weight",
                vec![3 * dim, dim],
            ),
            (
                "layers.0.residual_group.blocks.0.attn.qkv.bias",
                vec![3 * dim],
            ),
            (
                "layers.0.residual_group.blocks.0.mlp.fc1.weight",
                vec![2 * dim, dim],
            ),
            (
                "layers.0.residual_group.blocks.1.attn.qkv.weight",
                vec![3 * dim, dim],
            ),
            ("layers.0.conv.weight", vec![dim, dim, 3, 3]),
            ("conv_after_body.weight", vec![dim, dim, 3, 3]),
            ("conv_before_upsample.0.weight", vec![64, dim, 3, 3]),
            ("upsample.0.weight", vec![256, 64, 3, 3]),
            ("conv_last.weight", vec![3, 64, 3, 3]),
        ]
    }

    #[test]
    fn detects_swinir_with_window_and_depth_from_shapes() {
        let cfg = detect(&ck(&swinir_keys(8, 6, 180))).unwrap().unwrap();
        assert_eq!(cfg.arch, Arch::SwinIr);
        assert_eq!(cfg.scale, 2);
        match cfg.params {
            ArchParams::Swin(p) => {
                assert_eq!(p.embed_dim, 180);
                assert_eq!(p.window_size, 8);
                assert_eq!(p.depths, vec![2]);
                assert_eq!(p.num_heads, vec![6]);
                assert_eq!(p.mlp_ratio, 2.0);
                assert_eq!(p.resi_connection, ResiConnection::Conv1);
                assert_eq!(p.upsampler, Upsampler::PixelShuffle);
                assert!(p.hat.is_none());
            }
            other => panic!("wrong params: {other:?}"),
        }
    }

    /// HAT is SwinIR plus an overlap block; the overlap ratio is solved from
    /// the second bias table rather than assumed.
    #[test]
    fn detects_hat_and_solves_the_overlap_ratio() {
        let (window, dim, heads) = (16usize, 180usize, 6usize);
        let mut e = swinir_keys(window, heads, dim);
        // overlap_ratio 0.5 ⇒ ext = 24 ⇒ (16 + 24 − 1)² = 1521 rows
        e.push((
            "layers.0.residual_group.overlap_attn.relative_position_bias_table",
            vec![1521, heads],
        ));
        e.push((
            "layers.0.residual_group.blocks.0.conv_block.cab.0.weight",
            vec![dim / 3, dim, 3, 3],
        ));
        e.push((
            "layers.0.residual_group.blocks.0.conv_block.cab.3.attention.1.weight",
            vec![dim / 30, dim, 1, 1],
        ));
        let cfg = detect(&ck(&e)).unwrap().unwrap();
        assert_eq!(cfg.arch, Arch::Hat);
        match cfg.params {
            ArchParams::Swin(p) => {
                let h = p.hat.expect("HAT params");
                assert_eq!(h.compress_ratio, 3);
                assert_eq!(h.squeeze_factor, 30);
                assert!((h.overlap_ratio - 0.5).abs() < 1e-6, "{}", h.overlap_ratio);
            }
            other => panic!("wrong params: {other:?}"),
        }
    }

    #[test]
    fn detects_drct_with_growth_channels() {
        let (window, dim, gc, heads) = (16usize, 180usize, 32usize, 6usize);
        let rows = (2 * window - 1) * (2 * window - 1);
        let mut e = vec![
            ("conv_first.weight".to_string(), vec![dim, 3, 3, 3]),
            (
                "layers.0.swin1.attn.relative_position_bias_table".to_string(),
                vec![rows, heads],
            ),
            (
                "layers.0.swin1.mlp.fc1.weight".to_string(),
                vec![2 * dim, dim],
            ),
            ("layers.0.adjust1.weight".to_string(), vec![gc, dim, 1, 1]),
            ("conv_after_body.weight".to_string(), vec![dim, dim, 3, 3]),
            (
                "conv_before_upsample.0.weight".to_string(),
                vec![64, dim, 3, 3],
            ),
            ("upsample.0.weight".to_string(), vec![256, 64, 3, 3]),
            ("conv_last.weight".to_string(), vec![3, 64, 3, 3]),
        ];
        for i in 1..=5 {
            e.push((
                format!("layers.0.swin{i}.attn.qkv.weight"),
                vec![3 * dim, dim],
            ));
        }
        let entries: Vec<(&str, Vec<usize>)> = e
            .iter()
            .map(|(k, v)| (Box::leak(k.clone().into_boxed_str()) as &str, v.clone()))
            .collect();
        let cfg = detect(&ck(&entries)).unwrap().unwrap();
        assert_eq!(cfg.arch, Arch::Drct);
        match cfg.params {
            ArchParams::Swin(p) => {
                assert_eq!(p.drct_gc, Some(gc));
                assert_eq!(p.depths, vec![5]);
                // An RDG has no closing convolution.
                assert_eq!(p.resi_connection, ResiConnection::Identity);
            }
            other => panic!("wrong params: {other:?}"),
        }
    }

    /// MambaIRv2's routing dictionary is a factorization: `embeddingA` is
    /// `[inner_rank, d_state]` and `embeddingB` is `[num_tokens, inner_rank]`,
    /// so all three numbers come from the pair and the shared rank must agree.
    #[test]
    fn detects_mambairv2_from_its_routing_dictionary() {
        let (window, dim, heads) = (16usize, 174usize, 6usize);
        let rows = (2 * window - 1) * (2 * window - 1);
        let l = "layers.0.residual_group.layers.0";
        let mut e: Vec<(String, Vec<usize>)> = vec![
            ("conv_first.weight".into(), vec![dim, 3, 3, 3]),
            (
                format!("{l}.win_mhsa.relative_position_bias_table"),
                vec![rows, heads],
            ),
            (format!("{l}.norm1.weight"), vec![dim]),
            (format!("{l}.wqkv.weight"), vec![3 * dim, dim]),
            (format!("{l}.wqkv.bias"), vec![3 * dim]),
            (format!("{l}.embeddingA.weight"), vec![64, 16]),
            (format!("{l}.assm.embeddingB.weight"), vec![128, 64]),
            (format!("{l}.convffn1.fc1.weight"), vec![2 * dim, dim]),
            (
                format!("{l}.convffn1.dwconv.depthwise_conv.0.weight"),
                vec![2 * dim, 1, 5, 5],
            ),
            ("layers.0.conv.weight".into(), vec![dim, dim, 3, 3]),
            ("conv_after_body.weight".into(), vec![dim, dim, 3, 3]),
            ("conv_before_upsample.0.weight".into(), vec![64, dim, 3, 3]),
            ("upsample.0.weight".into(), vec![256, 64, 3, 3]),
            ("upsample.2.weight".into(), vec![256, 64, 3, 3]),
            ("conv_last.weight".into(), vec![3, 64, 3, 3]),
        ];
        for d in 1..6 {
            e.push((
                format!("layers.0.residual_group.layers.{d}.norm1.weight"),
                vec![dim],
            ));
        }
        let entries: Vec<(&str, Vec<usize>)> = e
            .iter()
            .map(|(k, v)| (Box::leak(k.clone().into_boxed_str()) as &str, v.clone()))
            .collect();
        let cfg = detect(&ck(&entries)).unwrap().unwrap();
        assert_eq!(cfg.arch, Arch::MambaIrV2);
        assert_eq!(cfg.scale, 4);
        match cfg.params {
            ArchParams::MambaIr(p) => {
                assert_eq!(p.embed_dim, 174);
                assert_eq!(p.depths, vec![6]);
                assert_eq!(p.window_size, 16);
                assert_eq!((p.num_tokens, p.inner_rank, p.d_state), (128, 64, 16));
                assert_eq!(p.convffn_kernel_size, 5);
                assert_eq!(p.mlp_ratio, 2.0);
            }
            other => panic!("wrong params: {other:?}"),
        }
    }

    /// An `inner_rank` that differs between the two factors means the dictionary
    /// was misread; better to say so than to build a graph around it.
    #[test]
    fn an_inconsistent_routing_dictionary_is_rejected() {
        let l = "layers.0.residual_group.layers.0";
        let rows = (2 * 16 - 1) * (2 * 16 - 1);
        let entries: Vec<(&str, Vec<usize>)> = vec![
            ("conv_first.weight", vec![174, 3, 3, 3]),
            (
                Box::leak(format!("{l}.win_mhsa.relative_position_bias_table").into_boxed_str()),
                vec![rows, 6],
            ),
            (
                Box::leak(format!("{l}.norm1.weight").into_boxed_str()),
                vec![174],
            ),
            (
                Box::leak(format!("{l}.embeddingA.weight").into_boxed_str()),
                vec![64, 16],
            ),
            (
                Box::leak(format!("{l}.assm.embeddingB.weight").into_boxed_str()),
                vec![128, 32],
            ),
        ];
        let e = detect(&ck(&entries)).unwrap_err();
        assert!(e.to_string().contains("inconsistent"), "{e}");
    }

    /// Every tail carries `conv_last`, so the bare-`conv_last` denoising
    /// variant has to be checked *after* the others or it never matches — and
    /// before this was handled, a released `005_colorDN_…` checkpoint simply
    /// failed to load.
    #[test]
    fn detects_the_x1_denoising_tail() {
        let mut e = swinir_keys(8, 6, 180);
        e.retain(|(k, _)| !k.starts_with("upsample.") && !k.starts_with("conv_before_upsample"));
        let cfg = detect(&ck(&e)).unwrap().unwrap();
        assert_eq!(cfg.arch, Arch::SwinIr);
        assert_eq!(cfg.scale, 1);
        match cfg.params {
            ArchParams::Swin(p) => assert_eq!(p.upsampler, Upsampler::Residual),
            other => panic!("wrong params: {other:?}"),
        }
    }

    #[test]
    fn a_bias_table_with_an_impossible_row_count_is_rejected() {
        // 100 rows would need (2W−1) = 10, which is even.
        assert!(window_from_table(100).is_err());
        assert_eq!(window_from_table(225).unwrap(), 8);
    }

    #[test]
    fn non_swin_checkpoints_fall_through() {
        assert!(
            detect(&ck(&[("body.0.weight", vec![64, 3, 3, 3])]))
                .unwrap()
                .is_none()
        );
    }
}
