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

//! Every architecture, built and executed.
//!
//! These use [`rlx_upscale::graph::build_autofill`], which synthesizes any
//! weight the (empty) checkpoint lacks. That deliberately does **not** check
//! numerics against PyTorch — that needs the real checkpoints, which are not
//! redistributable and do not belong in a test suite. What it does check is
//! every shape, reshape, permutation, window partition, concatenation and
//! index table in the graph, on a real compile-and-run. Those are where porting
//! bugs live: a transposed permutation or an off-by-one window is a shape error
//! or a non-finite output, not a subtle numeric drift.
//!
//! Configurations are scaled down (small dims, shallow depth) so the suite runs
//! in seconds, but every *structural* feature of each family is exercised.

use rlx_core::weight_map::WeightMap;
use rlx_runtime::{Device, Session};
use rlx_upscale::config::*;
use rlx_upscale::graph;

/// Build with synthesized weights, compile, run one tile, and return the output.
fn run(cfg: &ModelConfig, tile: usize) -> Vec<f32> {
    let built = graph::build_autofill(
        cfg,
        WeightMap::from_tensors(Default::default()),
        tile,
        tile,
        7,
    )
    .unwrap_or_else(|e| panic!("building {}: {e:#}", cfg.summary()));
    assert!(
        built.nodes > 5,
        "{} produced a suspiciously small graph ({} nodes)",
        cfg.summary(),
        built.nodes
    );

    let opts = rlx_core::flow_bridge::compile_options_for_profile(
        &rlx_flow::CompileProfile::encoder(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(built.graph, &opts);
    rlx_core::flow_util::attach_built_params(&mut compiled, built.params, &[]);

    let n = cfg.in_ch * tile * tile;
    let input: Vec<f32> = (0..n).map(|i| ((i * 31) % 97) as f32 / 97.0).collect();
    let out = compiled.run(&[("image", &input)]).remove(0);

    let want = cfg.out_ch * (tile * cfg.scale) * (tile * cfg.scale);
    assert_eq!(
        out.len(),
        want,
        "{}: output is {} samples, expected {want}",
        cfg.summary(),
        out.len()
    );
    assert!(
        out.iter().all(|v| v.is_finite()),
        "{}: output contains non-finite values",
        cfg.summary()
    );
    // An all-constant output would pass every shape assertion while meaning the
    // network collapsed — check the image actually varies.
    let lo = out.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        hi - lo > 1e-6,
        "{}: output is constant ({lo} … {hi})",
        cfg.summary()
    );
    out
}

fn conv_cfg(arch: Arch, scale: usize, params: ArchParams) -> ModelConfig {
    ModelConfig {
        arch,
        scale,
        in_ch: 3,
        out_ch: 3,
        params,
    }
}

// ── tier 1 ──────────────────────────────────────────────────────────────

/// OmniSR. The tile is two windows across, which is the smallest size at which
/// the grid partition is actually strided — at one window the grid and block
/// partitions coincide and a swap between them would go unnoticed.
#[test]
fn omnisr_builds_and_runs() {
    for scale in [1usize, 2, 3, 4] {
        run(
            &conv_cfg(
                Arch::OmniSr,
                scale,
                ArchParams::OmniSr {
                    num_feat: 16,
                    res_num: 2,
                    block_num: 1,
                    window_size: 8,
                },
            ),
            16,
        );
    }
}

/// ESA downsamples hard — a valid stride-2 3×3 then a 7×7 stride-3 max pool —
/// so a tile that clears the window constraint can still leave it nothing to
/// pool. That must be an error rather than a degenerate tensor.
#[test]
fn omnisr_rejects_a_tile_too_small_for_esa() {
    let cfg = conv_cfg(
        Arch::OmniSr,
        2,
        ArchParams::OmniSr {
            num_feat: 16,
            res_num: 1,
            block_num: 1,
            window_size: 8,
        },
    );
    // 8×8 → conv2 gives 3×3, which a 7×7 pooling kernel cannot cover.
    let e = graph::build_autofill(&cfg, WeightMap::from_tensors(Default::default()), 8, 8, 7)
        .err()
        .expect("an 8px tile leaves ESA nothing to pool");
    assert!(
        format!("{e:#}").contains("does not fit"),
        "unexpected error: {e:#}"
    );
}

/// Real-CUGAN, every wrapper. The tile is the smallest each variant allows,
/// which is where the valid-padding crops are tightest.
#[test]
fn realcugan_builds_and_runs() {
    for variant in [
        CuganVariant::X2,
        CuganVariant::X3,
        CuganVariant::X4,
        CuganVariant::X2Fast,
    ] {
        for pro in [false, true] {
            let cfg = ModelConfig {
                arch: Arch::RealCugan,
                scale: variant.scale(),
                in_ch: 3,
                out_ch: 3,
                params: ArchParams::RealCugan { variant, pro },
            };
            let tile = cfg.min_tile();
            assert!(
                tile >= 32,
                "{} min tile {tile} is too small",
                variant.name()
            );
            run(&cfg, tile);
        }
    }
}

/// A tile narrower than the reflect pad has nothing to reflect, which is the
/// one hard floor Real-CUGAN has. The valid-convolution shrinkage looks like it
/// should bind first and never does — every even tile down to 2 survives the
/// crops — so this is the constraint worth pinning.
#[test]
fn realcugan_rejects_a_tile_narrower_than_its_reflect_pad() {
    for variant in [CuganVariant::X2, CuganVariant::X3, CuganVariant::X2Fast] {
        let cfg = ModelConfig {
            arch: Arch::RealCugan,
            scale: variant.scale(),
            in_ch: 3,
            out_ch: 3,
            params: ArchParams::RealCugan {
                variant,
                pro: false,
            },
        };
        assert!(
            cfg.min_tile() > variant.input_pad(),
            "{} min tile {} does not clear its pad of {}",
            variant.name(),
            cfg.min_tile(),
            variant.input_pad()
        );

        let too_small = variant.input_pad() / variant.size_multiple() * variant.size_multiple();
        let e = graph::build_autofill(
            &cfg,
            WeightMap::from_tensors(Default::default()),
            too_small,
            too_small,
            7,
        )
        .err()
        .unwrap_or_else(|| {
            panic!(
                "{}: a {too_small}px tile has nothing to reflect",
                variant.name()
            )
        });
        let msg = format!("{e:#}");
        assert!(msg.contains("reflect-pads by"), "unexpected error: {msg}");
    }
}

/// The fast variant's reference minimum of 40 is not a round number — it is its
/// 38-pixel pad plus two. Pinned so the derivation stays visible.
#[test]
fn realcugan_fast_minimum_matches_the_reference() {
    let cfg = ModelConfig {
        arch: Arch::RealCugan,
        scale: 2,
        in_ch: 3,
        out_ch: 3,
        params: ArchParams::RealCugan {
            variant: CuganVariant::X2Fast,
            pro: false,
        },
    };
    assert_eq!(cfg.min_tile(), 40);
}

/// SAFMN at every scale it ships at, and at both the base and -L widths.
///
/// The tile is 16 — two multiples of 8 — so the deepest level really pools to
/// 2x2 and the nearest-upsample back is exercised at every ratio.
#[test]
fn safmn_builds_and_runs() {
    for scale in [1usize, 2, 4] {
        for (dim, n_blocks) in [(8usize, 2usize), (36, 3)] {
            run(
                &conv_cfg(
                    Arch::Safmn,
                    scale,
                    ArchParams::Safmn {
                        dim,
                        n_blocks,
                        hidden: dim * 2,
                    },
                ),
                16,
            );
        }
    }
}

/// The pooling ratios only collapse to a fixed kernel on an exact multiple, so
/// a ragged tile has to fail loudly rather than pool a different region.
///
/// Two guards cover this and the outer one should win: the graph builder checks
/// `size_multiple` up front and can name the model, while `Builder::max_pool`
/// only sees a ratio. The inner guard stays as the backstop for a future
/// architecture that pools without declaring a multiple.
#[test]
fn safmn_rejects_a_tile_that_does_not_divide() {
    let cfg = conv_cfg(
        Arch::Safmn,
        2,
        ArchParams::Safmn {
            dim: 8,
            n_blocks: 1,
            hidden: 16,
        },
    );
    assert_eq!(cfg.size_multiple(), 8);
    let e = graph::build_autofill(&cfg, WeightMap::from_tensors(Default::default()), 20, 20, 7)
        .err()
        .expect("a tile of 20 does not divide by 8");
    let msg = format!("{e:#}");
    assert!(
        msg.contains("multiples of 8") && msg.contains("20×20"),
        "unexpected error: {msg}"
    );
}

/// ESRGAN at every scale it ships at, plus the ESRGAN+ side paths.
#[test]
fn esrgan_builds_and_runs() {
    for scale in [1usize, 2, 4] {
        for plus in [false, true] {
            run(
                &conv_cfg(
                    Arch::Esrgan,
                    scale,
                    ArchParams::Esrgan {
                        num_filters: 16,
                        num_blocks: 2,
                        growth: 8,
                        plus,
                        shuffle_factor: 1,
                    },
                ),
                16,
            );
        }
    }
}

/// Real-ESRGAN's ×1 and ×2 variants unshuffle the input so the trunk always
/// runs at ×4. The trunk scale is therefore `scale × shuffle_factor`, and
/// getting that backwards yields an output of the wrong size.
#[test]
fn esrgan_unshuffled_variants_build() {
    for (scale, shuffle) in [(2usize, 2usize), (1, 4)] {
        let cfg = conv_cfg(
            Arch::Esrgan,
            scale,
            ArchParams::Esrgan {
                num_filters: 16,
                num_blocks: 2,
                growth: 8,
                plus: false,
                shuffle_factor: shuffle,
            },
        );
        run(&cfg, 16);
    }
}

#[test]
fn compact_builds_and_runs() {
    for act in [CompactAct::PRelu, CompactAct::Relu, CompactAct::LeakyRelu] {
        for scale in [1, 2, 4] {
            run(
                &conv_cfg(
                    Arch::Compact,
                    scale,
                    ArchParams::Compact {
                        num_feat: 8,
                        num_conv: 3,
                        act,
                    },
                ),
                16,
            );
        }
    }
}

#[test]
fn span_builds_and_runs() {
    for norm in [true, false] {
        run(
            &conv_cfg(
                Arch::Span,
                4,
                ArchParams::Span {
                    feature_channels: 12,
                    norm,
                    img_range: 255.0,
                    rgb_mean: [0.4488, 0.4371, 0.4040],
                },
            ),
            16,
        );
    }
}

#[test]
fn spanv2_builds_and_runs() {
    for conv_bias in [true, false] {
        run(
            &conv_cfg(
                Arch::SpanV2,
                4,
                ArchParams::SpanV2 {
                    feature_channels: 16,
                    conv_bias,
                },
            ),
            16,
        );
    }
}

#[test]
fn plksr_builds_and_runs_every_channel_mixer() {
    for ccm in [PlksrCcm::Ccm, PlksrCcm::Iccm, PlksrCcm::Dccm] {
        for with_idt in [false, true] {
            run(
                &conv_cfg(
                    Arch::Plksr,
                    2,
                    ArchParams::Plksr {
                        dim: 16,
                        n_blocks: 3,
                        kernel_size: 9,
                        pdim: 4,
                        ccm,
                        use_ea: true,
                        with_idt,
                    },
                ),
                16,
            );
        }
    }
}

#[test]
fn realplksr_builds_and_runs() {
    // GroupNorm tail and the channel-first LayerNorm variant.
    for norm_groups in [Some(4), None] {
        run(
            &conv_cfg(
                Arch::RealPlksr,
                4,
                ArchParams::RealPlksr {
                    dim: 16,
                    n_blocks: 3,
                    kernel_size: 9,
                    pdim: 4,
                    use_ea: true,
                    norm_groups,
                    upsampler: Upsampler::PixelShuffle,
                    dysample_groups: 0,
                    dysample_end_conv: false,
                },
            ),
            16,
        );
    }
}

/// The `*_dysample` models are the current community recommendation, and their
/// head is the only part of tier 1 that uses `grid_sample`.
#[test]
fn realplksr_with_dysample_builds_and_runs() {
    for (scale, groups) in [(4usize, 4usize), (2, 4), (3, 3)] {
        run(
            &conv_cfg(
                Arch::RealPlksr,
                scale,
                ArchParams::RealPlksr {
                    dim: 16,
                    n_blocks: 2,
                    kernel_size: 9,
                    pdim: 4,
                    use_ea: true,
                    norm_groups: Some(4),
                    upsampler: Upsampler::DySample,
                    dysample_groups: groups,
                    dysample_end_conv: true,
                },
            ),
            12,
        );
    }
}

/// A ×1 restoration model must be an exact pass-through in the resamplers
/// rather than a degenerate reshape.
#[test]
fn scale_one_models_build() {
    run(
        &conv_cfg(
            Arch::RealPlksr,
            1,
            ArchParams::RealPlksr {
                dim: 16,
                n_blocks: 2,
                kernel_size: 5,
                pdim: 4,
                use_ea: false,
                norm_groups: Some(4),
                upsampler: Upsampler::PixelShuffle,
                dysample_groups: 0,
                dysample_end_conv: false,
            },
        ),
        16,
    );
}

// ── tier 2 ──────────────────────────────────────────────────────────────

fn swin_cfg(arch: Arch, scale: usize, p: SwinParams) -> ModelConfig {
    ModelConfig {
        arch,
        scale,
        in_ch: 3,
        out_ch: 3,
        params: ArchParams::Swin(p),
    }
}

fn base_swin(window: usize) -> SwinParams {
    SwinParams {
        embed_dim: 24,
        depths: vec![2, 2],
        num_heads: vec![4, 4],
        window_size: window,
        mlp_ratio: 2.0,
        qkv_bias: true,
        patch_norm: true,
        resi_connection: ResiConnection::Conv1,
        upsampler: Upsampler::PixelShuffle,
        num_feat: 16,
        img_range: 1.0,
        hat: None,
        drct_gc: None,
        v2: false,
    }
}

#[test]
fn swinir_builds_and_runs_every_upsampler() {
    for up in [
        Upsampler::PixelShuffle,
        Upsampler::PixelShuffleDirect,
        Upsampler::NearestConv,
    ] {
        let mut p = base_swin(4);
        p.upsampler = up;
        // `nearest+conv` is a ×4-only tail in the reference.
        let scale = if up == Upsampler::NearestConv { 4 } else { 2 };
        run(&swin_cfg(Arch::SwinIr, scale, p), 16);
    }
}

/// Swin2SR: the same outer network on Swin **V2** blocks. Every upsampler the
/// released checkpoints use, at the scales they use it.
#[test]
fn swin2sr_builds_and_runs_every_upsampler() {
    for up in [
        Upsampler::PixelShuffle,
        Upsampler::PixelShuffleDirect,
        Upsampler::NearestConv,
    ] {
        let mut p = base_swin(4);
        p.upsampler = up;
        p.v2 = true;
        let scale = if up == Upsampler::NearestConv { 4 } else { 2 };
        run(&swin_cfg(Arch::Swin2Sr, scale, p), 16);
    }
}

/// V1 and V2 must not produce the same graph. They share every shape — the
/// difference is cosine attention, a folded MLP bias, res-post-norm and the
/// patch projection — so a `v2` flag that failed to reach `swin_block` would
/// leave every test passing and every Swin2SR checkpoint silently wrong.
#[test]
fn v2_blocks_differ_from_v1() {
    let build = |v2: bool| {
        let mut p = base_swin(4);
        p.v2 = v2;
        let cfg = swin_cfg(if v2 { Arch::Swin2Sr } else { Arch::SwinIr }, 2, p);
        graph::build_autofill(&cfg, WeightMap::from_tensors(Default::default()), 16, 16, 5)
            .unwrap_or_else(|e| panic!("building v2={v2}: {e:#}"))
    };
    let (v1, v2) = (build(false), build(true));
    assert_ne!(
        v1.nodes, v2.nodes,
        "V2 produced an identically sized graph, so the flag did not reach the blocks"
    );
}

/// SwinIR's denoising / JPEG-artifact branch: no upsampler at all, ×1, and the
/// network's output is a *correction* added to the input. A released
/// checkpoint of this shape (`005_colorDN_…`) failed to load before it was
/// handled, because every other tail carries `conv_last` too.
#[test]
fn swinir_residual_denoising_tail_builds() {
    let mut p = base_swin(4);
    p.upsampler = Upsampler::Residual;
    run(&swin_cfg(Arch::SwinIr, 1, p), 16);
}

/// The residual tail only makes sense at ×1 — it adds its output to the input,
/// which have to be the same size.
#[test]
fn the_residual_tail_rejects_a_scale_above_one() {
    let mut p = base_swin(4);
    p.upsampler = Upsampler::Residual;
    let cfg = swin_cfg(Arch::SwinIr, 2, p);
    match graph::build_autofill(&cfg, WeightMap::from_tensors(Default::default()), 16, 16, 1) {
        Ok(_) => panic!("a ×2 residual tail was accepted"),
        // `to_string()` is only the outermost context; the cause is in the chain.
        Err(e) => assert!(
            format!("{e:#}").contains("×1 restoration"),
            "unexpected: {e:#}"
        ),
    }
}

/// `3conv` bottlenecks the residual connection to a quarter of the width; the
/// reference only ever pairs it with the large classical models.
#[test]
fn swinir_builds_with_a_3conv_residual() {
    let mut p = base_swin(4);
    p.resi_connection = ResiConnection::Conv3;
    run(&swin_cfg(Arch::SwinIr, 2, p), 16);
}

/// An odd depth means the last block is shifted, which is the case that
/// exercises the cyclic roll *and* the wrap-around mask together.
#[test]
fn swinir_shifted_blocks_build() {
    let mut p = base_swin(4);
    p.depths = vec![3];
    p.num_heads = vec![4];
    run(&swin_cfg(Arch::SwinIr, 2, p), 16);
}

#[test]
fn hat_builds_and_runs() {
    let mut p = base_swin(4);
    p.hat = Some(HatParams {
        compress_ratio: 3,
        squeeze_factor: 6,
        conv_scale: 0.01,
        overlap_ratio: 0.5,
    });
    run(&swin_cfg(Arch::Hat, 2, p), 16);
}

/// HAT's overlap window grows with `overlap_ratio`; the unfold index and the
/// bias table both have to follow it.
#[test]
fn hat_builds_across_overlap_ratios() {
    for ratio in [0.25f32, 0.5, 0.75] {
        let mut p = base_swin(4);
        p.hat = Some(HatParams {
            compress_ratio: 3,
            squeeze_factor: 6,
            conv_scale: 0.01,
            overlap_ratio: ratio,
        });
        run(&swin_cfg(Arch::Hat, 2, p), 16);
    }
}

#[test]
fn drct_builds_and_runs() {
    let mut p = base_swin(4);
    p.depths = vec![5, 5];
    p.num_heads = vec![4, 4];
    p.resi_connection = ResiConnection::Identity;
    p.drct_gc = Some(8);
    run(&swin_cfg(Arch::Drct, 2, p), 16);
}

/// Window 8 with a 16px tile is two windows across — the smallest geometry
/// where the shift actually wraps rather than degenerating.
#[test]
fn swin_handles_a_two_window_tile() {
    let mut p = base_swin(8);
    p.depths = vec![2];
    p.num_heads = vec![4];
    run(&swin_cfg(Arch::SwinIr, 2, p), 16);
}

/// A tile that is not a whole number of windows must be rejected at build time
/// rather than silently mis-partitioned.
#[test]
fn a_tile_that_does_not_tile_the_window_is_rejected() {
    let p = base_swin(8);
    let cfg = swin_cfg(Arch::SwinIr, 2, p);
    let e =
        match graph::build_autofill(&cfg, WeightMap::from_tensors(Default::default()), 20, 20, 1) {
            Ok(_) => panic!("a 20px tile silently partitioned into 8px windows"),
            Err(e) => e,
        };
    assert!(
        e.to_string().contains("multiples of 8"),
        "unexpected error: {e:#}"
    );
}

/// DAT alternates spatial (rectangular, two orientations) and channel
/// (transposed) attention blocks, so a depth of 4 is the smallest that
/// exercises both kinds *and* the shift schedule's period-4 alternation.
#[test]
fn dat_builds_and_runs() {
    let cfg = ModelConfig {
        arch: Arch::Dat,
        scale: 2,
        in_ch: 3,
        out_ch: 3,
        params: ArchParams::Dat(DatParams {
            embed_dim: 32,
            depths: vec![4],
            num_heads: vec![4],
            split_size: [4, 8],
            expansion_factor: 2.0,
            qkv_bias: true,
            resi_connection: ResiConnection::Conv1,
            upsampler: Upsampler::PixelShuffle,
            img_range: 1.0,
        }),
    };
    run(&cfg, 16);
}

/// Two groups means the shift schedule flips parity between them — the branch
/// that reads `gi % 2` in the reference.
#[test]
fn dat_shift_schedule_alternates_between_groups() {
    let cfg = ModelConfig {
        arch: Arch::Dat,
        scale: 2,
        in_ch: 3,
        out_ch: 3,
        params: ArchParams::Dat(DatParams {
            embed_dim: 32,
            depths: vec![2, 2],
            num_heads: vec![4, 4],
            split_size: [4, 8],
            expansion_factor: 2.0,
            qkv_bias: true,
            resi_connection: ResiConnection::Conv1,
            upsampler: Upsampler::PixelShuffleDirect,
            img_range: 1.0,
        }),
    };
    run(&cfg, 16);
}

/// A tile that divides one window extent but not the other must be rejected:
/// the two branches transpose the window, so both extents have to tile.
#[test]
fn dat_rejects_a_tile_that_only_one_orientation_tiles() {
    let cfg = ModelConfig {
        arch: Arch::Dat,
        scale: 2,
        in_ch: 3,
        out_ch: 3,
        params: ArchParams::Dat(DatParams {
            embed_dim: 32,
            depths: vec![1],
            num_heads: vec![4],
            split_size: [4, 16],
            expansion_factor: 2.0,
            qkv_bias: true,
            resi_connection: ResiConnection::Conv1,
            upsampler: Upsampler::PixelShuffleDirect,
            img_range: 1.0,
        }),
    };
    // 8 divides 4 but not 16.
    match graph::build_autofill(&cfg, WeightMap::from_tensors(Default::default()), 8, 8, 1) {
        Ok(_) => panic!("an 8px tile silently partitioned into 16px windows"),
        Err(e) => assert!(e.to_string().contains("multiples of"), "unexpected: {e:#}"),
    }
}

/// MambaIRv2: shifted-window attention plus a selective scan over a
/// *semantically sorted* token order. Depth 2 covers both the unshifted and
/// shifted layer, and the sort/gather/inverse-gather round trip is the part
/// most likely to be wrong.
#[test]
fn mambairv2_builds_and_runs() {
    let cfg = ModelConfig {
        arch: Arch::MambaIrV2,
        scale: 2,
        in_ch: 3,
        out_ch: 3,
        params: ArchParams::MambaIr(MambaIrParams {
            embed_dim: 24,
            depths: vec![2],
            num_heads: vec![4],
            window_size: 8,
            mlp_ratio: 2.0,
            d_state: 8,
            num_tokens: 16,
            inner_rank: 12,
            convffn_kernel_size: 5,
            qkv_bias: true,
            patch_norm: true,
            resi_connection: ResiConnection::Conv1,
            upsampler: Upsampler::PixelShuffleDirect,
            num_feat: 16,
            img_range: 1.0,
        }),
    };
    run(&cfg, 16);
}

/// Two groups and the classical tail, so the group residual and the
/// multi-stage pixel-shuffle head are both exercised.
#[test]
fn mambairv2_builds_with_two_groups_and_a_classical_tail() {
    let cfg = ModelConfig {
        arch: Arch::MambaIrV2,
        scale: 4,
        in_ch: 3,
        out_ch: 3,
        params: ArchParams::MambaIr(MambaIrParams {
            embed_dim: 24,
            depths: vec![2, 2],
            num_heads: vec![4, 4],
            window_size: 8,
            mlp_ratio: 2.0,
            d_state: 8,
            num_tokens: 16,
            inner_rank: 12,
            convffn_kernel_size: 5,
            qkv_bias: true,
            patch_norm: true,
            resi_connection: ResiConnection::Conv1,
            upsampler: Upsampler::PixelShuffle,
            num_feat: 16,
            img_range: 1.0,
        }),
    };
    run(&cfg, 16);
}
