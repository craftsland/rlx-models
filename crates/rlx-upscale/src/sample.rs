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

//! One small, runnable configuration per architecture.
//!
//! Two things need exactly the same list and must not drift apart: the op
//! coverage matrix (`examples/op_coverage.rs`, which decides *statically*
//! which backends can run what) and the cross-backend numerical parity test
//! (`tests/backends.rs`, which checks that they actually do). If the matrix
//! covered an architecture the parity test skipped, a backend could be
//! declared fine and be wrong; if the parity test covered one the matrix
//! skipped, a real op gap would go unreported.
//!
//! These are deliberately *small* — an architecture's op set and its backend
//! behaviour are properties of its structure, not its width — but every
//! structural branch that changes which ops are emitted gets its own entry.

use crate::config::*;

/// One representative configuration per architecture.
///
/// These are deliberately *small* — the op set is a property of the
/// architecture, not of its width — but every structural branch that changes
/// which ops are emitted gets its own entry.
pub fn representative_configs() -> Vec<(&'static str, ModelConfig, usize)> {
    let conv = |arch, scale, params| {
        (
            ModelConfig {
                arch,
                scale,
                in_ch: 3,
                out_ch: 3,
                params,
            },
            16usize,
        )
    };
    let swin = |arch, scale, p: SwinParams| {
        (
            ModelConfig {
                arch,
                scale,
                in_ch: 3,
                out_ch: 3,
                params: ArchParams::Swin(p),
            },
            16usize,
        )
    };
    let base_swin = |v2: bool, hat: Option<HatParams>| SwinParams {
        embed_dim: 24,
        depths: vec![2, 2],
        num_heads: vec![4, 4],
        window_size: 4,
        mlp_ratio: 2.0,
        qkv_bias: true,
        patch_norm: true,
        resi_connection: ResiConnection::Conv1,
        upsampler: Upsampler::PixelShuffle,
        num_feat: 16,
        img_range: 1.0,
        hat,
        drct_gc: None,
        v2,
    };

    let mut v: Vec<(&'static str, ModelConfig, usize)> = Vec::new();
    let mut push =
        |name: &'static str, (cfg, tile): (ModelConfig, usize)| v.push((name, cfg, tile));

    push(
        "ESRGAN",
        conv(
            Arch::Esrgan,
            4,
            ArchParams::Esrgan {
                num_filters: 8,
                num_blocks: 2,
                growth: 4,
                plus: true,
                shuffle_factor: 1,
            },
        ),
    );
    push(
        "Compact",
        conv(
            Arch::Compact,
            4,
            ArchParams::Compact {
                num_feat: 8,
                num_conv: 2,
                act: CompactAct::PRelu,
            },
        ),
    );
    push(
        "SPAN",
        conv(
            Arch::Span,
            4,
            ArchParams::Span {
                feature_channels: 12,
                norm: true,
                img_range: 255.0,
                rgb_mean: [0.4488, 0.4371, 0.4040],
            },
        ),
    );
    push(
        "SPANV2",
        conv(
            Arch::SpanV2,
            4,
            ArchParams::SpanV2 {
                feature_channels: 16,
                conv_bias: false,
            },
        ),
    );
    push(
        "PLKSR",
        conv(
            Arch::Plksr,
            4,
            ArchParams::Plksr {
                dim: 16,
                n_blocks: 2,
                kernel_size: 9,
                pdim: 4,
                ccm: PlksrCcm::Ccm,
                use_ea: true,
                with_idt: false,
            },
        ),
    );
    push(
        "RealPLKSR",
        conv(
            Arch::RealPlksr,
            2,
            ArchParams::RealPlksr {
                dim: 16,
                n_blocks: 2,
                kernel_size: 9,
                pdim: 4,
                use_ea: true,
                norm_groups: Some(4),
                upsampler: Upsampler::DySample,
                dysample_groups: 4,
                dysample_end_conv: true,
            },
        ),
    );
    push(
        "SAFMN",
        conv(
            Arch::Safmn,
            4,
            ArchParams::Safmn {
                dim: 8,
                n_blocks: 2,
                hidden: 16,
            },
        ),
    );
    push(
        "RealCUGAN",
        (
            ModelConfig {
                arch: Arch::RealCugan,
                scale: 2,
                in_ch: 3,
                out_ch: 3,
                params: ArchParams::RealCugan {
                    variant: CuganVariant::X2,
                    pro: true,
                },
            },
            32,
        ),
    );
    push(
        "OmniSR",
        conv(
            Arch::OmniSr,
            4,
            ArchParams::OmniSr {
                num_feat: 16,
                res_num: 1,
                block_num: 1,
                window_size: 8,
            },
        ),
    );
    push("SwinIR", swin(Arch::SwinIr, 2, base_swin(false, None)));
    push("Swin2SR", swin(Arch::Swin2Sr, 2, base_swin(true, None)));
    push(
        "HAT",
        swin(
            Arch::Hat,
            2,
            base_swin(
                false,
                Some(HatParams {
                    compress_ratio: 3,
                    squeeze_factor: 6,
                    conv_scale: 0.01,
                    overlap_ratio: 0.5,
                }),
            ),
        ),
    );
    push("DRCT", {
        let mut p = base_swin(false, None);
        p.resi_connection = ResiConnection::Identity;
        p.drct_gc = Some(8);
        p.depths = vec![6];
        p.num_heads = vec![4];
        swin(Arch::Drct, 2, p)
    });
    push(
        "DAT",
        (
            ModelConfig {
                arch: Arch::Dat,
                scale: 2,
                in_ch: 3,
                out_ch: 3,
                params: ArchParams::Dat(DatParams {
                    embed_dim: 24,
                    depths: vec![2],
                    num_heads: vec![4],
                    split_size: [2, 4],
                    expansion_factor: 2.0,
                    qkv_bias: true,
                    resi_connection: ResiConnection::Conv1,
                    upsampler: Upsampler::PixelShuffle,
                    img_range: 1.0,
                }),
            },
            16,
        ),
    );
    push(
        "MambaIRv2",
        (
            ModelConfig {
                arch: Arch::MambaIrV2,
                scale: 2,
                in_ch: 3,
                out_ch: 3,
                params: ArchParams::MambaIr(MambaIrParams {
                    embed_dim: 24,
                    depths: vec![2],
                    num_heads: vec![4],
                    window_size: 4,
                    d_state: 8,
                    mlp_ratio: 2.0,
                    convffn_kernel_size: 5,
                    inner_rank: 8,
                    num_tokens: 8,
                    qkv_bias: true,
                    patch_norm: true,
                    resi_connection: ResiConnection::Conv1,
                    upsampler: Upsampler::PixelShuffle,
                    num_feat: 16,
                    img_range: 1.0,
                }),
            },
            16,
        ),
    );
    v
}
