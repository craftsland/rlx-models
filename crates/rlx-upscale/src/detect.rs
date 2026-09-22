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

//! Recovering a [`ModelConfig`] from a bare state dict.
//!
//! There is no metadata to read: a community upscaler is a `.pth` of tensors
//! and nothing else. Every hyperparameter therefore has to be inferred from key
//! names and tensor shapes, the way `spandrel` does it — the architecture is
//! identified by a key only it has, then its dimensions are read off the
//! tensors that carry them.
//!
//! # What cannot be inferred
//!
//! A few reference hyperparameters leave no trace in the weights, and guessing
//! silently would produce a wrong image rather than an error:
//!
//! * **Compact's ReLU vs LeakyReLU** — neither has parameters. PReLU is
//!   detectable (it has a slope tensor); the other two are not distinguishable,
//!   so `relu` is assumed. Every released Compact model is PReLU.
//! * **PLKSR's `with_idt`** — pure forward-pass logic. The reference default
//!   (`false`) is assumed.
//! * **RealPLKSR's `norm_groups`** — a `GroupNorm` weight is `[dim]` whatever
//!   the group count. The reference default (4) is assumed.
//!
//! These are recorded here rather than buried, because each is a place a
//! future divergence will come from.

use anyhow::{Result, bail, ensure};

use crate::config::{
    Arch, ArchParams, CompactAct, CuganVariant, ModelConfig, PlksrCcm, ResiConnection, Upsampler,
};
use crate::weights::Checkpoint;

/// Identify a checkpoint and recover its hyperparameters.
pub fn detect(ck: &Checkpoint) -> Result<ModelConfig> {
    // Ordered most-specific first: SPANV2 and SPAN share no keys, but PLKSR and
    // RealPLKSR are told apart only by a one-letter difference in the mixer name.
    if ck.contains("conv_near.weight") && ck.contains("depthwise_conv.weight") {
        return detect_spanv2(ck);
    }
    if ck.contains("conv_1.sk.weight") || ck.contains("conv_1.eval_conv.weight") {
        return detect_span(ck);
    }
    if ck.contains("feats.1.channel_mixer.0.weight") {
        return detect_realplksr(ck);
    }
    if ck.contains("feats.1.channe_mixer.0.weight") {
        return detect_plksr(ck);
    }
    if ck.contains("body.0.weight") && ck.contains("body.2.weight") {
        return detect_compact(ck);
    }
    // `feats.N` is shared with PLKSR, which is why this sits below both of
    // those: SAFMN's own tell is the multi-scale filter bank inside a block.
    if ck.contains("to_feat.weight") && ck.contains("feats.0.safm.aggr.weight") {
        return detect_safmn(ck);
    }
    if ck.contains("unet1.conv1.conv.0.weight") && ck.contains("unet2.conv5.weight") {
        return detect_realcugan(ck);
    }
    if ck.contains("residual_layer.0.residual_layer.0.layer.0.fn.0.weight") {
        return detect_omnisr(ck);
    }
    // ESRGAN has to be probed *after* Compact: a Real-ESRGAN compact model's
    // `body.N.weight` and an RRDBNet's `body.{i}.rdb…` both start `body.`, but
    // only the dense blocks go deeper.
    if ck.contains("model.1.sub.0.RDB1.conv1.0.weight")
        || ck.contains("model.1.sub.0.RDB1.conv1x1.weight")
        || ck.contains("body.0.rdb1.conv1.weight")
        || ck.contains("RRDB_trunk.0.RDB1.conv1.weight")
    {
        return detect_esrgan(ck);
    }
    if let Some(cfg) = crate::detect_swin::detect(ck)? {
        return Ok(cfg);
    }
    if let Some(cfg) = crate::detect_swin::detect_dat(ck)? {
        return Ok(cfg);
    }

    let mut sample: Vec<&str> = ck.keys();
    sample.truncate(8);
    bail!(
        "unrecognized upscaler architecture ({} tensors; first keys: {sample:?}).\n\
         Supported: ESRGAN, Compact, SPAN, SPANV2, PLKSR, RealPLKSR, SAFMN, \
         RealCUGAN, OmniSR, SwinIR, HAT, DRCT, DAT, MambaIRv2.",
        ck.len()
    )
}

/// Shape of a required tensor.
pub(crate) fn shape<'a>(ck: &'a Checkpoint, key: &str) -> Result<&'a [usize]> {
    ck.shape_of(key)
        .ok_or_else(|| anyhow::anyhow!("checkpoint is missing {key}"))
}

/// Recover an integer upscale factor from a `out_ch · r²`-wide convolution.
pub(crate) fn scale_from_pixelshuffle(out_channels: usize, out_ch: usize) -> Result<usize> {
    ensure!(
        out_ch > 0 && out_channels.is_multiple_of(out_ch),
        "pixel-shuffle head is {out_channels} wide, not a multiple of {out_ch} output channels"
    );
    let sq = out_channels / out_ch;
    let r = (sq as f64).sqrt().round() as usize;
    ensure!(
        r * r == sq,
        "pixel-shuffle head implies a scale factor of √{sq}, which is not an integer"
    );
    Ok(r)
}

/// Highest `N` for which `{prefix}{N}{suffix}` exists.
pub(crate) fn max_index(ck: &Checkpoint, prefix: &str, suffix: &str) -> Option<usize> {
    let mut best = None;
    for k in ck.keys() {
        let Some(rest) = k.strip_prefix(prefix) else {
            continue;
        };
        let Some(idx) = rest.strip_suffix(suffix) else {
            continue;
        };
        if let Ok(n) = idx.parse::<usize>() {
            best = Some(best.map_or(n, |b: usize| b.max(n)));
        }
    }
    best
}

/// ESRGAN / RRDBNet.
///
/// Assumes the checkpoint has already been normalized to the old-arch key
/// layout by [`crate::weights::esrgan_to_old_arch`], because the scale is
/// recovered from that layout's flattened sequence length and the other two
/// naming schemes do not have one.
fn detect_esrgan(ck: &Checkpoint) -> Result<ModelConfig> {
    let stem = shape(ck, "model.0.weight")?;
    ensure!(stem.len() == 4, "model.0.weight should be rank 4");
    let num_filters = stem[0];
    let stem_ch = stem[1];

    // `model.{n}` for the largest n present — the flattened `Sequential`'s
    // length, which is what encodes the scale.
    let seq_len = max_index(ck, "model.", ".weight")
        .ok_or_else(|| anyhow::anyhow!("no model.N.weight in checkpoint"))?
        + 1;
    ensure!(
        seq_len >= 5 && (seq_len - 5) % 3 == 0,
        "ESRGAN sequence length {seq_len} does not fit [conv, trunk, 3·octaves, conv, act, conv]"
    );
    let octaves = (seq_len - 5) / 3;
    let trunk_scale = 1usize << octaves;

    let num_blocks = (0..)
        .take_while(|i| ck.contains(&format!("model.1.sub.{i}.RDB1.conv1.0.weight")))
        .count();
    ensure!(num_blocks > 0, "no RRDB blocks found");

    // The first dense convolution maps `nf` → `gc`.
    let growth = shape(ck, "model.1.sub.0.RDB1.conv1.0.weight")?[0];
    let plus = ck.contains("model.1.sub.0.RDB1.conv1x1.weight");

    let out_ch = shape(ck, &format!("model.{}.weight", seq_len - 1))?[0];

    // Real-ESRGAN's ×1 and ×2 variants unshuffle the input by 2 or 4 so the
    // trunk always runs at ×4; the stem then takes 4× or 16× the channels.
    let shuffle_factor = if stem_ch == out_ch * 4 {
        2
    } else if stem_ch == out_ch * 16 {
        4
    } else {
        1
    };
    ensure!(
        trunk_scale.is_multiple_of(shuffle_factor),
        "an unshuffle of {shuffle_factor} does not divide a trunk scale of {trunk_scale}"
    );

    Ok(ModelConfig {
        arch: Arch::Esrgan,
        scale: trunk_scale / shuffle_factor,
        in_ch: stem_ch / (shuffle_factor * shuffle_factor),
        out_ch,
        params: ArchParams::Esrgan {
            num_filters,
            num_blocks,
            growth,
            plus,
            shuffle_factor,
        },
    })
}

fn detect_compact(ck: &Checkpoint) -> Result<ModelConfig> {
    let first = shape(ck, "body.0.weight")?;
    ensure!(first.len() == 4, "body.0.weight should be rank 4");
    let num_feat = first[0];
    let in_ch = first[1];

    let last = max_index(ck, "body.", ".weight")
        .ok_or_else(|| anyhow::anyhow!("no body.N.weight in checkpoint"))?;
    // body = [conv, act] + num_conv × [conv, act] + [conv]
    ensure!(
        last >= 2 && last % 2 == 0,
        "Compact body ends at index {last}; the final conv must sit at an even index"
    );
    let num_conv = (last - 2) / 2;

    let tail = shape(ck, &format!("body.{last}.weight"))?;
    let out_ch = in_ch;
    let scale = scale_from_pixelshuffle(tail[0], out_ch)?;

    let act = if ck.contains("body.1.weight") {
        CompactAct::PRelu
    } else {
        CompactAct::Relu
    };

    Ok(ModelConfig {
        arch: Arch::Compact,
        scale,
        in_ch,
        out_ch,
        params: ArchParams::Compact {
            num_feat,
            num_conv,
            act,
        },
    })
}

/// OmniSR.
///
/// The window size is the *only* hyperparameter that has to be solved for,
/// and the relative-position table states it: it has `(2·ws − 1)²` rows. The
/// rest are counted from the key namespace, and the head counts and hidden
/// widths are structural — see [`crate::arch::omnisr`].
///
/// # `pe=False` is not supported, and is detectable
///
/// Without the position embedding there is no bias table and therefore nothing
/// left in the weights that states the window size. The reference's default is
/// `pe=True` and every released checkpoint has it; a checkpoint without one is
/// reported rather than run at a guessed window, because the wrong window
/// partitions the image differently and still produces an image.
fn detect_omnisr(ck: &Checkpoint) -> Result<ModelConfig> {
    let stem = shape(ck, "input.weight")?;
    ensure!(stem.len() == 4, "input.weight should be rank 4");
    let num_feat = stem[0];
    let in_ch = stem[1];

    let res_num = max_index(
        ck,
        "residual_layer.",
        ".residual_layer.0.layer.0.fn.0.weight",
    )
    .ok_or_else(|| anyhow::anyhow!("no OSAG groups in checkpoint"))?
        + 1;
    let block_num = max_index(
        ck,
        "residual_layer.0.residual_layer.",
        ".layer.0.fn.0.weight",
    )
    .ok_or_else(|| anyhow::anyhow!("no OSA blocks in checkpoint"))?
        + 1;

    let pe_key = "residual_layer.0.residual_layer.0.layer.2.fn.rel_pos_bias.weight";
    let table = ck.shape_of(pe_key).ok_or_else(|| {
        anyhow::anyhow!(
            "OmniSR checkpoint has no relative position bias ({pe_key}), so it was \
             trained with pe=False. The window size is then stated nowhere in the \
             weights, and guessing it would partition the image differently while \
             still producing an image, so this is not supported."
        )
    })?;
    ensure!(table.len() == 2, "{pe_key} should be rank 2");
    // `(2·ws − 1)²` rows.
    let span = (table[0] as f64).sqrt().round() as usize;
    ensure!(
        span * span == table[0] && span % 2 == 1,
        "OmniSR position table has {} rows, which is not (2·ws − 1)² for any ws",
        table[0]
    );
    let window_size = span.div_ceil(2);

    let out_ch = in_ch;
    let scale = scale_from_pixelshuffle(shape(ck, "up.0.weight")?[0], out_ch)?;

    Ok(ModelConfig {
        arch: Arch::OmniSr,
        scale,
        in_ch,
        out_ch,
        params: ArchParams::OmniSr {
            num_feat,
            res_num,
            block_num,
            window_size,
        },
    })
}

/// Real-CUGAN. The four wrappers share almost every key, so the variant is read
/// from the two places they differ.
///
/// # The ×4 / ×2-fast ambiguity is real
///
/// `UpCunet4x(in=4)` and `UpCunet2x_fast(in=1)` produce *identical* state-dict
/// shapes — the fast variant pixel-unshuffles a 1-channel input into 4 channels,
/// which is indistinguishable from a 4-channel input. Both here and in spandrel
/// the tie is broken toward the fast variant, because a 4-channel-input ×4
/// Real-CUGAN is not a thing anyone ships and a 1-channel fast model is. For
/// the ordinary 3-channel case there is no ambiguity at all.
fn detect_realcugan(ck: &Checkpoint) -> Result<ModelConfig> {
    let stem = shape(ck, "unet1.conv1.conv.0.weight")?;
    ensure!(
        stem.len() == 4,
        "unet1.conv1.conv.0.weight should be rank 4"
    );
    let mut in_ch = stem[1];
    let pro = ck.contains("pro");

    let (variant, out_ch) = if let Some(f) = ck.shape_of("conv_final.weight") {
        ensure!(
            f[0] % 4 == 0,
            "Real-CUGAN conv_final is {} wide, not a multiple of 4",
            f[0]
        );
        let out_ch = f[0] / 4;
        if out_ch * 4 == in_ch {
            in_ch /= 4;
            (CuganVariant::X2Fast, out_ch)
        } else {
            (CuganVariant::X4, out_ch)
        }
    } else {
        let bottom = shape(ck, "unet1.conv_bottom.weight")?;
        ensure!(
            bottom.len() == 4,
            "unet1.conv_bottom.weight should be rank 4"
        );
        // The ×3 tail is a 5×5 stride-3 transposed convolution; ×2's is 4×4.
        let variant = if bottom[2] == 5 {
            CuganVariant::X3
        } else {
            CuganVariant::X2
        };
        (variant, shape(ck, "unet2.conv_bottom.weight")?[0])
    };

    Ok(ModelConfig {
        arch: Arch::RealCugan,
        scale: variant.scale(),
        in_ch,
        out_ch,
        params: ArchParams::RealCugan { variant, pro },
    })
}

/// SAFMN. Everything is readable: `dim` from the stem, the block count from
/// the `feats` sequence, the expanded width from the CCM's first convolution
/// and the scale from the pixel-shuffle head.
///
/// `ffn_scale` is deliberately *not* recovered as a float and re-multiplied —
/// the reference truncates `int(dim * ffn_scale)`, so a 36-dim model at 2.5
/// gives 90 and dividing back gives 2.5 exactly, but a ratio that lands on a
/// non-representable product would rebuild a differently-sized layer. The
/// hidden width itself is what the graph needs, so that is what is stored.
fn detect_safmn(ck: &Checkpoint) -> Result<ModelConfig> {
    let stem = shape(ck, "to_feat.weight")?;
    ensure!(stem.len() == 4, "to_feat.weight should be rank 4");
    let dim = stem[0];
    let in_ch = stem[1];

    let n_blocks = max_index(ck, "feats.", ".safm.aggr.weight")
        .ok_or_else(|| anyhow::anyhow!("no feats.N.safm.aggr.weight in checkpoint"))?
        + 1;

    // The NTIRE-2023 challenge entry reuses every one of these key names while
    // being a different network: no per-block norms but a single top-level GRN,
    // `bias=False` throughout the trunk, no feature-space long skip but a
    // bilinear image-space one, and — least visibly — pooling by `2^(i+1)`
    // rather than `2^i`, which changes the required tile multiple from 8 to 16.
    // Building it as a base SAFMN would load real weights into the wrong graph
    // and return a plausible, wrong image, so it is named and refused.
    ensure!(
        ck.contains("feats.0.norm1.weight"),
        "this looks like the NTIRE-2023 SAFMN variant ({}), which shares SAFMN's \
         key names but is a different network (global response normalization, \
         no trunk skip, biasless convolutions, and pooling by 2^(i+1)). It is \
         not supported — neither is it by spandrel or chaiNNer.",
        if ck.contains("norm.gamma") {
            "it carries norm.gamma/norm.beta instead of per-block norms"
        } else {
            "no per-block norm1 weight"
        }
    );

    let hidden = shape(ck, "feats.0.ccm.ccm.0.weight")?[0];

    let out_ch = in_ch;
    let head = shape(ck, "to_img.0.weight")?;
    let scale = scale_from_pixelshuffle(head[0], out_ch)?;

    // The channel split is structural, not a hyperparameter: `dim // n_levels`
    // sizes every depthwise filter, so an indivisible dim would build a graph
    // whose weights cannot be loaded.
    ensure!(
        dim % crate::arch::safmn::N_LEVELS == 0,
        "SAFMN: {dim} channels do not split into {} levels",
        crate::arch::safmn::N_LEVELS
    );

    Ok(ModelConfig {
        arch: Arch::Safmn,
        scale,
        in_ch,
        out_ch,
        params: ArchParams::Safmn {
            dim,
            n_blocks,
            hidden,
        },
    })
}

fn detect_span(ck: &Checkpoint) -> Result<ModelConfig> {
    // Prefer the training branch: it is what actually determines the fused
    // kernel, and `eval_conv` may be a stale cache.
    let stem = if ck.contains("conv_1.sk.weight") {
        shape(ck, "conv_1.sk.weight")?
    } else {
        shape(ck, "conv_1.eval_conv.weight")?
    };
    let feature_channels = stem[0];
    let in_ch = stem[1];

    let head = shape(ck, "upsampler.0.weight")?;
    let out_ch = in_ch;
    let scale = scale_from_pixelshuffle(head[0], out_ch)?;

    Ok(ModelConfig {
        arch: Arch::Span,
        scale,
        in_ch,
        out_ch,
        params: ArchParams::Span {
            feature_channels,
            // The reference registers a `no_norm` buffer precisely when
            // normalization is *off*, so its absence means it is on.
            norm: !ck.contains("no_norm"),
            img_range: 255.0,
            rgb_mean: [0.4488, 0.4371, 0.4040],
        },
    })
}

fn detect_spanv2(ck: &Checkpoint) -> Result<ModelConfig> {
    let c1 = shape(ck, "block_1.c1.conv.weight")?;
    ensure!(c1.len() == 4, "block_1.c1.conv.weight should be rank 4");
    let feature_channels = c1[0];
    let in_ch = c1[1];

    // Depthwise near-pixel branch: [in_ch · r², 1, 3, 3].
    let near = shape(ck, "conv_near.weight")?;
    let scale = scale_from_pixelshuffle(near[0], in_ch)?;

    Ok(ModelConfig {
        arch: Arch::SpanV2,
        scale,
        in_ch,
        out_ch: in_ch,
        params: ArchParams::SpanV2 {
            feature_channels,
            conv_bias: ck.contains("block_1.c1.conv.bias"),
        },
    })
}

/// Shared shape reads for the two PLKSR variants.
struct PlksrCommon {
    dim: usize,
    in_ch: usize,
    kernel_size: usize,
    pdim: usize,
    use_ea: bool,
    last: usize,
}

fn plksr_common(ck: &Checkpoint) -> Result<PlksrCommon> {
    let first = shape(ck, "feats.0.weight")?;
    let lk = shape(ck, "feats.1.lk.conv.weight")?;
    ensure!(lk.len() == 4, "feats.1.lk.conv.weight should be rank 4");
    let last = max_index(ck, "feats.", ".weight")
        .ok_or_else(|| anyhow::anyhow!("no feats.N.weight in checkpoint"))?;
    Ok(PlksrCommon {
        dim: first[0],
        in_ch: first[1],
        kernel_size: lk[2],
        pdim: lk[0],
        use_ea: ck.contains("feats.1.attn.f.0.weight"),
        last,
    })
}

fn detect_plksr(ck: &Checkpoint) -> Result<ModelConfig> {
    let c = plksr_common(ck)?;
    // `feats` = [conv] + n_blocks × [PLKBlock] + [conv], so the last index is
    // n_blocks + 1.
    ensure!(c.last >= 2, "PLKSR feats ends at index {}", c.last);
    let n_blocks = c.last - 1;

    // The mixer's two convolutions distinguish the three CCM variants by
    // kernel size alone.
    let k0 = shape(ck, "feats.1.channe_mixer.0.weight")?[2];
    let k2 = shape(ck, "feats.1.channe_mixer.2.weight")?[2];
    let ccm = match (k0, k2) {
        (3, 1) => PlksrCcm::Ccm,
        (1, 3) => PlksrCcm::Iccm,
        (3, 3) => PlksrCcm::Dccm,
        _ => bail!("unrecognized PLKSR channel mixer with {k0}×{k0} then {k2}×{k2} convolutions"),
    };

    let tail = shape(ck, &format!("feats.{}.weight", c.last))?;
    let out_ch = c.in_ch;
    let scale = scale_from_pixelshuffle(tail[0], out_ch)?;

    Ok(ModelConfig {
        arch: Arch::Plksr,
        scale,
        in_ch: c.in_ch,
        out_ch,
        params: ArchParams::Plksr {
            dim: c.dim,
            n_blocks,
            kernel_size: c.kernel_size,
            pdim: c.pdim,
            ccm,
            use_ea: c.use_ea,
            with_idt: false,
        },
    })
}

fn detect_realplksr(ck: &Checkpoint) -> Result<ModelConfig> {
    let c = plksr_common(ck)?;
    // RealPLKSR inserts a parameterless `Dropout2d` before the final conv, so
    // the last index is n_blocks + 2 — one further out than PLKSR.
    ensure!(c.last >= 3, "RealPLKSR feats ends at index {}", c.last);
    let n_blocks = c.last - 2;

    let norm_groups = if ck.contains("feats.1.layer_norm.weight") {
        None
    } else {
        Some(4)
    };

    let tail = shape(ck, &format!("feats.{}.weight", c.last))?;
    let out_ch = c.in_ch;
    let scale = scale_from_pixelshuffle(tail[0], out_ch)?;

    let (upsampler, dysample_groups, dysample_end_conv) = if ck.contains("to_img.offset.weight") {
        let off = shape(ck, "to_img.offset.weight")?;
        // out_channels = 2 · groups · scale²
        let denom = 2 * scale * scale;
        ensure!(
            off[0] % denom == 0,
            "DySample offset conv is {} wide, not a multiple of 2·{scale}²",
            off[0]
        );
        (
            Upsampler::DySample,
            off[0] / denom,
            ck.contains("to_img.end_conv.weight"),
        )
    } else {
        (Upsampler::PixelShuffle, 0, false)
    };

    Ok(ModelConfig {
        arch: Arch::RealPlksr,
        scale,
        in_ch: c.in_ch,
        out_ch,
        params: ArchParams::RealPlksr {
            dim: c.dim,
            n_blocks,
            kernel_size: c.kernel_size,
            pdim: c.pdim,
            use_ea: c.use_ea,
            norm_groups,
            upsampler,
            dysample_groups,
            dysample_end_conv,
        },
    })
}

/// `resi_connection` from which convolution closes a residual group.
pub(crate) fn resi_connection(ck: &Checkpoint) -> ResiConnection {
    if ck.contains("layers.0.conv.0.weight") {
        ResiConnection::Conv3
    } else if ck.contains("layers.0.conv.weight") {
        ResiConnection::Conv1
    } else {
        ResiConnection::Identity
    }
}

/// The upsampler tail shared by SwinIR / HAT / DRCT / DAT.
///
/// Returns `(kind, scale, out_channels)`. The three tails are told apart by
/// which convolutions surround the pixel shuffles, and the order below matters:
/// **all three** carry `upsample.0.weight`, so the presence of `conv_last` /
/// `conv_before_upsample` has to be checked first. Matching on `upsample.0`
/// alone reads a lightweight model as a classical one and then fails looking
/// for a `conv_last` it never had.
pub(crate) fn swin_upsampler(ck: &Checkpoint) -> Result<(Upsampler, usize, usize)> {
    // `nearest+conv` (real-world SR): one ×2 refinement stage per `conv_up`.
    if ck.contains("conv_up1.weight") {
        let out_ch = shape(ck, "conv_last.weight")?[0];
        let scale = if ck.contains("conv_up3.weight") {
            8
        } else if ck.contains("conv_up2.weight") {
            4
        } else {
            2
        };
        return Ok((Upsampler::NearestConv, scale, out_ch));
    }

    // Classical SR: conv_before_upsample → [conv, shuffle]* → conv_last. Each
    // stage widens `num_feat` by 4 (×2) or 9 (×3); the convolutions sit at even
    // indices with the shuffles between them.
    if let Some(before) = ck.shape_of("conv_before_upsample.0.weight") {
        let num_feat = before[0];
        let out_ch = shape(ck, "conv_last.weight")?[0];
        let mut scale = 1usize;
        let mut i = 0usize;
        while let Some(stage) = ck.shape_of(&format!("upsample.{i}.weight")) {
            ensure!(
                num_feat > 0 && stage[0] % num_feat == 0,
                "upsample stage {i} is {} wide against {num_feat} features",
                stage[0]
            );
            match stage[0] / num_feat {
                4 => scale *= 2,
                9 => scale *= 3,
                other => bail!("upsample stage {i} widens by {other}×, expected 4 or 9"),
            }
            i += 2;
        }
        ensure!(
            i > 0,
            "conv_before_upsample with no upsample stages after it"
        );
        return Ok((Upsampler::PixelShuffle, scale, out_ch));
    }

    // Lightweight SR (`UpsampleOneStep`): a single conv straight to
    // `out_ch · scale²`, no surrounding convolutions at all.
    if let Some(one) = ck.shape_of("upsample.0.weight") {
        let out_ch = 3;
        let scale = scale_from_pixelshuffle(one[0], out_ch)?;
        return Ok((Upsampler::PixelShuffleDirect, scale, out_ch));
    }

    // Denoising / JPEG-artifact reduction: no upsampler at all, just a
    // convolution back to image width whose output is added to the input. This
    // has to come last, because `conv_last` is present in the other tails too.
    if let Some(last) = ck.shape_of("conv_last.weight") {
        return Ok((Upsampler::Residual, 1, last[0]));
    }

    bail!(
        "no recognizable upsampler tail (expected conv_up1, conv_before_upsample.0, \
         upsample.0, or a bare conv_last for the ×1 denoising variant)"
    )
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

    #[test]
    fn detects_compact_x4_prelu() {
        let mut e = vec![
            ("body.0.weight", vec![64, 3, 3, 3]),
            ("body.1.weight", vec![64]),
        ];
        for i in 0..16 {
            e.push((
                Box::leak(format!("body.{}.weight", 2 + 2 * i).into_boxed_str()),
                vec![64, 64, 3, 3],
            ));
            e.push((
                Box::leak(format!("body.{}.weight", 3 + 2 * i).into_boxed_str()),
                vec![64],
            ));
        }
        e.push(("body.34.weight", vec![48, 64, 3, 3]));
        let cfg = detect(&ck(&e)).unwrap();
        assert_eq!(cfg.arch, Arch::Compact);
        assert_eq!(cfg.scale, 4);
        assert_eq!(
            cfg.params,
            ArchParams::Compact {
                num_feat: 64,
                num_conv: 16,
                act: CompactAct::PRelu
            }
        );
    }

    /// The three ESRGAN naming schemes must all land on the same config —
    /// that is the whole point of normalizing on load.
    #[test]
    fn all_three_esrgan_key_layouts_agree() {
        let nf = 64usize;
        let gc = 32usize;
        let nb = 3usize;

        // Old arch: already canonical.
        let mut old: Vec<(String, Vec<usize>)> = vec![
            ("model.0.weight".into(), vec![nf, 3, 3, 3]),
            (format!("model.1.sub.{nb}.weight"), vec![nf, nf, 3, 3]),
            ("model.3.weight".into(), vec![nf, nf, 3, 3]),
            ("model.6.weight".into(), vec![nf, nf, 3, 3]),
            ("model.8.weight".into(), vec![nf, nf, 3, 3]),
            ("model.10.weight".into(), vec![3, nf, 3, 3]),
        ];
        // New arch (Real-ESRGAN) and BSRGAN, same network.
        let mut new: Vec<(String, Vec<usize>)> = vec![
            ("conv_first.weight".into(), vec![nf, 3, 3, 3]),
            ("conv_body.weight".into(), vec![nf, nf, 3, 3]),
            ("conv_up1.weight".into(), vec![nf, nf, 3, 3]),
            ("conv_up2.weight".into(), vec![nf, nf, 3, 3]),
            ("conv_hr.weight".into(), vec![nf, nf, 3, 3]),
            ("conv_last.weight".into(), vec![3, nf, 3, 3]),
        ];
        let mut bsr: Vec<(String, Vec<usize>)> = vec![
            ("conv_first.weight".into(), vec![nf, 3, 3, 3]),
            ("trunk_conv.weight".into(), vec![nf, nf, 3, 3]),
            ("upconv1.weight".into(), vec![nf, nf, 3, 3]),
            ("upconv2.weight".into(), vec![nf, nf, 3, 3]),
            ("HRconv.weight".into(), vec![nf, nf, 3, 3]),
            ("conv_last.weight".into(), vec![3, nf, 3, 3]),
        ];
        for i in 0..nb {
            for j in 1..=3 {
                for k in 1..=5 {
                    let in_c = nf + (k - 1) * gc;
                    let out_c = if k == 5 { nf } else { gc };
                    old.push((
                        format!("model.1.sub.{i}.RDB{j}.conv{k}.0.weight"),
                        vec![out_c, in_c, 3, 3],
                    ));
                    new.push((
                        format!("body.{i}.rdb{j}.conv{k}.weight"),
                        vec![out_c, in_c, 3, 3],
                    ));
                    bsr.push((
                        format!("RRDB_trunk.{i}.RDB{j}.conv{k}.weight"),
                        vec![out_c, in_c, 3, 3],
                    ));
                }
            }
        }

        let mut configs = Vec::new();
        for layout in [old, new, bsr] {
            let entries: Vec<(&str, Vec<usize>)> = layout
                .iter()
                .map(|(k, v)| (Box::leak(k.clone().into_boxed_str()) as &str, v.clone()))
                .collect();
            configs.push(detect(&ck(&entries)).unwrap());
        }
        assert_eq!(configs[0], configs[1], "new arch disagrees with old");
        assert_eq!(configs[0], configs[2], "BSRGAN arch disagrees with old");
        assert_eq!(configs[0].arch, Arch::Esrgan);
        assert_eq!(configs[0].scale, 4);
        assert_eq!(
            configs[0].params,
            ArchParams::Esrgan {
                num_filters: nf,
                num_blocks: nb,
                growth: gc,
                plus: false,
                shuffle_factor: 1,
            }
        );
    }

    #[test]
    fn detects_span_x2() {
        let cfg = detect(&ck(&[
            ("conv_1.sk.weight", vec![48, 3, 1, 1]),
            ("upsampler.0.weight", vec![12, 48, 3, 3]),
        ]))
        .unwrap();
        assert_eq!(cfg.arch, Arch::Span);
        assert_eq!(cfg.scale, 2);
        match cfg.params {
            ArchParams::Span {
                feature_channels,
                norm,
                ..
            } => {
                assert_eq!(feature_channels, 48);
                // No `no_norm` buffer ⇒ normalization is on.
                assert!(norm);
            }
            other => panic!("wrong params: {other:?}"),
        }
    }

    /// SPANV2 must win over SPAN: both have `block_*`, but only SPANV2 has the
    /// near-pixel branch, and misreading one as the other silently changes the
    /// activation and drops the guidance map.
    #[test]
    fn detects_spanv2_and_prefers_it_over_span() {
        let cfg = detect(&ck(&[
            ("block_1.c1.conv.weight", vec![32, 3, 3, 3]),
            ("conv_near.weight", vec![48, 1, 3, 3]),
            ("depthwise_conv.weight", vec![80, 1, 3, 3]),
            ("pointwise_conv.weight", vec![48, 80, 1, 1]),
        ]))
        .unwrap();
        assert_eq!(cfg.arch, Arch::SpanV2);
        assert_eq!(cfg.scale, 4);
        assert_eq!(
            cfg.params,
            ArchParams::SpanV2 {
                feature_channels: 32,
                conv_bias: false
            }
        );
    }

    /// The one-letter difference between `channe_mixer` and `channel_mixer` is
    /// the *only* thing separating the two PLKSR families, and they differ in
    /// activation, block tail and final-conv index.
    #[test]
    fn plksr_and_realplksr_are_told_apart_by_the_mixer_key() {
        let plksr = detect(&ck(&[
            ("feats.0.weight", vec![64, 3, 3, 3]),
            ("feats.1.channe_mixer.0.weight", vec![128, 64, 3, 3]),
            ("feats.1.channe_mixer.2.weight", vec![64, 128, 1, 1]),
            ("feats.1.lk.conv.weight", vec![16, 16, 17, 17]),
            ("feats.1.attn.f.0.weight", vec![64, 64, 3, 3]),
            ("feats.29.weight", vec![48, 64, 3, 3]),
        ]))
        .unwrap();
        assert_eq!(plksr.arch, Arch::Plksr);
        match plksr.params {
            ArchParams::Plksr {
                n_blocks,
                ccm,
                pdim,
                kernel_size,
                ..
            } => {
                assert_eq!(n_blocks, 28);
                assert_eq!(ccm, PlksrCcm::Ccm);
                assert_eq!(pdim, 16);
                assert_eq!(kernel_size, 17);
            }
            other => panic!("wrong params: {other:?}"),
        }

        let real = detect(&ck(&[
            ("feats.0.weight", vec![64, 3, 3, 3]),
            ("feats.1.channel_mixer.0.weight", vec![128, 64, 3, 3]),
            ("feats.1.channel_mixer.2.weight", vec![64, 128, 3, 3]),
            ("feats.1.lk.conv.weight", vec![16, 16, 17, 17]),
            ("feats.1.attn.f.0.weight", vec![64, 64, 3, 3]),
            ("feats.1.norm.weight", vec![64]),
            ("feats.30.weight", vec![48, 64, 3, 3]),
        ]))
        .unwrap();
        assert_eq!(real.arch, Arch::RealPlksr);
        match real.params {
            // Same 30-index checkpoint would be read as 29 blocks by PLKSR's
            // rule; the dropout slot is why it is 28.
            ArchParams::RealPlksr {
                n_blocks,
                norm_groups,
                upsampler,
                ..
            } => {
                assert_eq!(n_blocks, 28);
                assert_eq!(norm_groups, Some(4));
                assert_eq!(upsampler, Upsampler::PixelShuffle);
            }
            other => panic!("wrong params: {other:?}"),
        }
    }

    #[test]
    fn detects_dysample_groups_from_the_offset_width() {
        let cfg = detect(&ck(&[
            ("feats.0.weight", vec![64, 3, 3, 3]),
            ("feats.1.channel_mixer.0.weight", vec![128, 64, 3, 3]),
            ("feats.1.channel_mixer.2.weight", vec![64, 128, 3, 3]),
            ("feats.1.lk.conv.weight", vec![16, 16, 17, 17]),
            ("feats.30.weight", vec![48, 64, 3, 3]),
            // 2 · groups · 4² = 128 ⇒ groups = 4
            ("to_img.offset.weight", vec![128, 48, 1, 1]),
            ("to_img.end_conv.weight", vec![3, 48, 1, 1]),
        ]))
        .unwrap();
        match cfg.params {
            ArchParams::RealPlksr {
                upsampler,
                dysample_groups,
                dysample_end_conv,
                use_ea,
                ..
            } => {
                assert_eq!(upsampler, Upsampler::DySample);
                assert_eq!(dysample_groups, 4);
                assert!(dysample_end_conv);
                // No `attn.f.0.weight` in this fixture.
                assert!(!use_ea);
            }
            other => panic!("wrong params: {other:?}"),
        }
    }

    #[test]
    fn a_non_square_pixelshuffle_head_is_rejected_not_rounded() {
        let e = detect(&ck(&[
            ("conv_1.sk.weight", vec![48, 3, 1, 1]),
            // 15 channels ⇒ scale² = 5, not a square.
            ("upsampler.0.weight", vec![15, 48, 3, 3]),
        ]))
        .unwrap_err();
        assert!(
            e.to_string().contains("not an integer"),
            "unexpected error: {e}"
        );
    }

    #[test]
    fn an_unknown_checkpoint_names_the_supported_families() {
        let e = detect(&ck(&[("some.random.weight", vec![1])])).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("unrecognized"), "{msg}");
        assert!(msg.contains("RealPLKSR"), "{msg}");
    }
}
