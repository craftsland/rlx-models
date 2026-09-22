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

//! Architecture identity and hyperparameters.
//!
//! Community upscaler checkpoints are **bare state dicts**: no config JSON, no
//! architecture tag, nothing but tensor names and shapes. Every field here is
//! therefore recovered by [`crate::detect`] from the checkpoint itself, and
//! this module's job is to describe — exactly and serializably — what was
//! recovered. A `ModelConfig` round-trips through JSON so a detection can be
//! pinned in a test fixture without the weights.

use serde::{Deserialize, Serialize};

/// The supported architecture families.
///
/// Tier 1 (`Compact` … `RealPlksr`) are pure convolutional nets that run in
/// well under a gigabyte and are fast enough for video. Tier 2 (`SwinIr` …
/// `Dat`) are window-attention transformers: higher fidelity, 20–35× slower,
/// and only bounded in memory because [`crate::tile`] cuts the image up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Arch {
    /// ESRGAN / RRDBNet — residual-in-residual dense blocks. By volume the
    /// most widely deployed super-resolution architecture there is.
    Esrgan,
    /// `SRVGGNetCompact` — the Real-ESRGAN "compact" net. Plain conv stack.
    Compact,
    /// SPAN — Swift Parameter-free Attention Network (NTIRE 2024).
    Span,
    /// SPANV2 — the NTIRE 2026 Efficient SR challenge winner (team XiaomiMM).
    SpanV2,
    /// PLKSR — Partial Large Kernel SR.
    Plksr,
    /// RealPLKSR — PLKSR retuned for real-world degradation.
    RealPlksr,
    /// SwinIR — the window-attention SR baseline.
    SwinIr,
    /// HAT — Hybrid Attention Transformer (window + channel + overlapping cross).
    Hat,
    /// DRCT — Dense-Residual-Connected Transformer.
    Drct,
    /// DAT — Dual Aggregation Transformer.
    Dat,
    /// MambaIRv2 — Attentive State Space restoration.
    MambaIrV2,
    /// SAFMN — Spatially-Adaptive Feature Modulation. Receptive field from
    /// pooling rather than from attention or large kernels.
    Safmn,
    /// Real-CUGAN — two valid-padded U-Nets in series. The anime upscaler.
    RealCugan,
    /// OmniSR — Omni Self-Attention: MaxViT-style block + grid attention.
    OmniSr,
    /// Swin2SR — SwinIR rebuilt on Swin **V2** blocks.
    Swin2Sr,
}

/// Which Real-CUGAN wrapper a checkpoint is.
///
/// The four differ in more than their scale: the input pad, the tail, and at
/// ×3 even the transposed-convolution kernel all change, so this is the
/// variant and the scale is derived from it rather than the other way round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CuganVariant {
    X2,
    X3,
    X4,
    /// `UpCunet2x_fast`: pixel-unshuffles the input so the U-Nets run at half
    /// resolution, and pads by 38 instead of 18 to compensate.
    X2Fast,
}

impl CuganVariant {
    pub fn name(self) -> &'static str {
        match self {
            CuganVariant::X2 => "2x",
            CuganVariant::X3 => "3x",
            CuganVariant::X4 => "4x",
            CuganVariant::X2Fast => "2x-fast",
        }
    }

    pub fn scale(self) -> usize {
        match self {
            CuganVariant::X2 | CuganVariant::X2Fast => 2,
            CuganVariant::X3 => 3,
            CuganVariant::X4 => 4,
        }
    }

    /// Reflect padding applied to the input, per side.
    pub fn input_pad(self) -> usize {
        match self {
            CuganVariant::X2 => 18,
            CuganVariant::X3 => 14,
            CuganVariant::X4 => 19,
            CuganVariant::X2Fast => 38,
        }
    }

    /// The multiple the input must be padded to.
    ///
    /// The reference rounds the input up to this before padding; the ×3
    /// variant needs 4 because its stride-3 tail and two stride-2 downsamples
    /// do not otherwise divide, and the fast variant needs 4 because its
    /// `PixelUnshuffle(2)` halves the map before the U-Nets see it.
    pub fn size_multiple(self) -> usize {
        match self {
            CuganVariant::X2 | CuganVariant::X4 => 2,
            CuganVariant::X3 | CuganVariant::X2Fast => 4,
        }
    }
}

impl Arch {
    /// Family name as it appears in model zoos.
    pub fn name(self) -> &'static str {
        match self {
            Arch::Esrgan => "ESRGAN",
            Arch::Compact => "Compact",
            Arch::Span => "SPAN",
            Arch::SpanV2 => "SPANV2",
            Arch::Plksr => "PLKSR",
            Arch::RealPlksr => "RealPLKSR",
            Arch::SwinIr => "SwinIR",
            Arch::Hat => "HAT",
            Arch::Drct => "DRCT",
            Arch::Dat => "DAT",
            Arch::MambaIrV2 => "MambaIRv2",
            Arch::Safmn => "SAFMN",
            Arch::RealCugan => "RealCUGAN",
            Arch::OmniSr => "OmniSR",
            Arch::Swin2Sr => "Swin2SR",
        }
    }

    /// Tier 1 architectures are convolutional; tier 2 are transformers.
    pub fn is_transformer(self) -> bool {
        matches!(
            self,
            Arch::SwinIr | Arch::Swin2Sr | Arch::Hat | Arch::Drct | Arch::Dat | Arch::MambaIrV2
        )
    }

    /// A starting tile size for this family, before [`ModelConfig::default_tile`]
    /// shrinks it to fit the model's actual attention cost.
    ///
    /// A conv net's cost is linear in pixels, so its tile only needs to be big
    /// enough to amortize the halo. A transformer's is not.
    fn nominal_tile(self) -> usize {
        if self.is_transformer() { 256 } else { 512 }
    }
}

/// The activation used by [`Arch::Compact`]'s body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompactAct {
    Relu,
    /// Per-channel learned slope — the Real-ESRGAN default.
    PRelu,
    /// Fixed slope 0.1.
    LeakyRelu,
}

/// How a network turns its feature map into the output image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Upsampler {
    /// `Conv2d(feat, out*r²) → PixelShuffle(r)`, repeated per ×2 stage for
    /// scales 4 and 8. SwinIR/HAT/DRCT/DAT "classical SR".
    PixelShuffle,
    /// One `Conv2d(feat, out*r²) → PixelShuffle(r)`. SwinIR/DAT "lightweight".
    PixelShuffleDirect,
    /// Nearest-neighbour + conv refinement. SwinIR "real-world SR" (`nearest+conv`).
    NearestConv,
    /// Learned-offset resampling (`DySample`), used by newer RealPLKSR models.
    DySample,
    /// No upsampler at all: one convolution back to image width, added to the
    /// input. SwinIR's denoising / JPEG-artifact branch, which is always ×1 —
    /// the network predicts a correction rather than a magnification.
    Residual,
}

/// Recovered hyperparameters for one checkpoint.
///
/// Fields are shared across families where they mean the same thing (`scale`,
/// `in_ch`, `out_ch`) and grouped into an [`ArchParams`] otherwise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub arch: Arch,
    /// Upscaling factor. 1 is legal — several restoration models are ×1.
    pub scale: usize,
    pub in_ch: usize,
    pub out_ch: usize,
    pub params: ArchParams,
}

/// Per-family hyperparameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case")]
pub enum ArchParams {
    Esrgan {
        num_filters: usize,
        num_blocks: usize,
        /// Growth channels inside each dense block. Always 32 in practice, but
        /// it is readable from the weights so it is read.
        growth: usize,
        /// ESRGAN+ adds a 1×1 path inside every dense block.
        plus: bool,
        /// Real-ESRGAN's ×1/×2 variants `pixel_unshuffle` the input so the
        /// trunk always runs at ×4. 1 means no unshuffle.
        shuffle_factor: usize,
    },
    Compact {
        num_feat: usize,
        num_conv: usize,
        act: CompactAct,
    },
    Span {
        feature_channels: usize,
        /// Whether the checkpoint carries the `no_norm` buffer. When absent the
        /// model subtracts the DIV2K RGB mean and scales by `img_range`.
        norm: bool,
        img_range: f32,
        rgb_mean: [f32; 3],
    },
    SpanV2 {
        feature_channels: usize,
        /// `bias=False` in the reference; a checkpoint may still carry biases.
        conv_bias: bool,
    },
    Plksr {
        dim: usize,
        n_blocks: usize,
        kernel_size: usize,
        /// Channels the large kernel actually touches (`int(dim * split_ratio)`).
        pdim: usize,
        ccm: PlksrCcm,
        use_ea: bool,
        /// `with_idt`: the large-kernel branch adds its own input back.
        with_idt: bool,
    },
    RealPlksr {
        dim: usize,
        n_blocks: usize,
        kernel_size: usize,
        pdim: usize,
        use_ea: bool,
        /// Per-block `GroupNorm` group count. `None` ⇒ the block uses the
        /// channel-first `LayerNorm` variant instead.
        norm_groups: Option<usize>,
        upsampler: Upsampler,
        /// `DySample` group count (only meaningful for [`Upsampler::DySample`]).
        dysample_groups: usize,
        /// `DySample` ends with a 1×1 conv unless the scale is 1.
        dysample_end_conv: bool,
    },
    /// SwinIR, HAT and DRCT share one parameter set — they are the same
    /// backbone with different block bodies.
    Swin(SwinParams),
    Dat(DatParams),
    MambaIr(MambaIrParams),
    RealCugan {
        variant: CuganVariant,
        /// "pro" checkpoints expect a compressed input range. The `pro` buffer's
        /// presence is the flag; its value is unused.
        pro: bool,
    },
    OmniSr {
        num_feat: usize,
        /// `res_num` — how many `OSAG` groups.
        res_num: usize,
        /// `block_num` — `OSA_Block`s inside each group.
        block_num: usize,
        window_size: usize,
    },
    Safmn {
        dim: usize,
        n_blocks: usize,
        /// `int(dim * ffn_scale)` — read from the CCM's first convolution
        /// rather than recomputed, because the reference truncates the product
        /// and a checkpoint trained at 2.5x would not round-trip.
        hidden: usize,
    },
}

/// The channel mixer inside a PLKSR block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlksrCcm {
    /// 3×3 → GELU → 1×1.
    Ccm,
    /// 1×1 → GELU → 3×3.
    Iccm,
    /// 3×3 → GELU → 3×3.
    Dccm,
}

/// Shared parameters of the SwinIR / HAT / DRCT backbone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwinParams {
    pub embed_dim: usize,
    /// Blocks per residual group.
    pub depths: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub window_size: usize,
    pub mlp_ratio: f32,
    pub qkv_bias: bool,
    /// Whether the patch embedding is followed by a `LayerNorm`.
    pub patch_norm: bool,
    /// `1conv` (a single 3×3) or `3conv` (the 3-layer bottleneck) after each group.
    pub resi_connection: ResiConnection,
    pub upsampler: Upsampler,
    /// Working width of the upsampling tail. The reference hard-codes 64.
    pub num_feat: usize,
    pub img_range: f32,
    /// HAT only — the channel-attention branch inside each block.
    pub hat: Option<HatParams>,
    /// DRCT only — growth channels of the dense connections.
    pub drct_gc: Option<usize>,
    /// Swin **V2** blocks: cosine attention, a learned per-head temperature,
    /// an MLP-generated position bias, and res-post-norm. This is what makes a
    /// SwinIR checkpoint a Swin2SR one — the outer structure is identical.
    pub v2: bool,
}

/// The convolution that closes each residual group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResiConnection {
    /// A single 3×3.
    Conv1,
    /// 3×3 → lrelu → 1×1 → lrelu → 3×3, with a ¼ channel bottleneck.
    Conv3,
    /// No convolution at all (DRCT's `identity`).
    Identity,
}

/// HAT's extra attention branches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HatParams {
    /// Channel-attention block bottleneck ratio.
    pub compress_ratio: usize,
    /// Channel-attention squeeze factor.
    pub squeeze_factor: usize,
    /// Weight on the CAB branch added to each window-attention block.
    pub conv_scale: f32,
    /// Overlapping-cross-attention window growth.
    pub overlap_ratio: f32,
}

/// MambaIRv2's parameters.
///
/// Shares the window geometry of the Swin lineage, and adds the state-space
/// half: `d_state` is the SSM state width, and the routing dictionary is
/// `num_tokens` prototypes each built from an `inner_rank`-dimensional factor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MambaIrParams {
    pub embed_dim: usize,
    pub depths: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub window_size: usize,
    pub mlp_ratio: f32,
    /// Selective-scan state width.
    pub d_state: usize,
    /// Size of the routing dictionary.
    pub num_tokens: usize,
    /// Rank of the factorization behind each prototype.
    pub inner_rank: usize,
    /// Kernel of the depthwise branch inside each `ConvFFN`.
    pub convffn_kernel_size: usize,
    pub qkv_bias: bool,
    pub patch_norm: bool,
    pub resi_connection: ResiConnection,
    pub upsampler: Upsampler,
    pub num_feat: usize,
    pub img_range: f32,
}

/// DAT's parameters. DAT alternates spatial and channel aggregation blocks and
/// uses rectangular windows that swap orientation on odd blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatParams {
    pub embed_dim: usize,
    pub depths: Vec<usize>,
    pub num_heads: Vec<usize>,
    /// `[height, width]` of the rectangular attention window.
    pub split_size: [usize; 2],
    pub expansion_factor: f32,
    pub qkv_bias: bool,
    pub resi_connection: ResiConnection,
    pub upsampler: Upsampler,
    pub img_range: f32,
}

impl ModelConfig {
    /// Total residual-group count, for reporting.
    pub fn num_groups(&self) -> usize {
        match &self.params {
            ArchParams::Swin(p) => p.depths.len(),
            ArchParams::Dat(p) => p.depths.len(),
            ArchParams::MambaIr(p) => p.depths.len(),
            ArchParams::OmniSr { res_num, .. } => *res_num,
            _ => 0,
        }
    }

    /// The multiple every input dimension must be padded to.
    ///
    /// Window attention partitions the feature map exactly, so a transformer
    /// tile has to be a whole number of windows; the conv nets accept any size.
    pub fn size_multiple(&self) -> usize {
        match &self.params {
            ArchParams::Swin(p) => p.window_size,
            ArchParams::MambaIr(p) => p.window_size,
            // DAT's rectangular window swaps orientation between blocks, so
            // both extents must divide.
            ArchParams::Dat(p) => lcm(p.split_size[0], p.split_size[1]),
            // Not a window constraint: SAFMN pools by 1, 2, 4 and 8, and
            // `adaptive_max_pool2d` only reduces to a fixed kernel when every
            // ratio divides exactly.
            ArchParams::Safmn { .. } => crate::arch::safmn::SIZE_MULTIPLE,
            ArchParams::RealCugan { variant, .. } => variant.size_multiple(),
            ArchParams::OmniSr { window_size, .. } => *window_size,
            _ => 1,
        }
    }

    /// Widest `C_in · k²` among this model's convolutions — the row width of
    /// the im2col workspace, which is what makes a convolution expensive in
    /// *memory* rather than in FLOPs.
    ///
    /// This is where the families diverge most sharply. RealPLKSR's whole idea
    /// is a 17×17 kernel, and `16 · 17² = 4624` is eight times Compact's
    /// `64 · 3² = 576` — so at the same tile it needs an order of magnitude
    /// more scratch, which is exactly what measurement shows (0.25 MB per
    /// input pixel against 0.02).
    fn widest_im2col_row(&self) -> u64 {
        match &self.params {
            ArchParams::Compact { num_feat, .. } => (num_feat * 9) as u64,
            // The widest im2col row is the last dense convolution, which sees
            // the block input plus four growth outputs.
            ArchParams::Esrgan {
                num_filters,
                growth,
                ..
            } => ((num_filters + 4 * growth) * 9) as u64,
            ArchParams::Span {
                feature_channels, ..
            } => (feature_channels * 9) as u64,
            ArchParams::SpanV2 {
                feature_channels, ..
            } => {
                let cat = self.in_ch * self.scale * self.scale + feature_channels;
                ((feature_channels * 9).max(cat)) as u64
            }
            ArchParams::Plksr {
                dim,
                kernel_size,
                pdim,
                ..
            }
            | ArchParams::RealPlksr {
                dim,
                kernel_size,
                pdim,
                ..
            } => ((pdim * kernel_size * kernel_size).max(dim * 2 * 9)) as u64,
            // The transformers convolve only at the stem, the group tails and
            // the upsampler; their cost is attention, not im2col.
            ArchParams::Swin(p) => (p.embed_dim * 9) as u64,
            ArchParams::Dat(p) => (p.embed_dim * 9) as u64,
            // MambaIRv2's ConvFFN depthwise runs at the mlp-expanded width.
            ArchParams::MambaIr(p) => {
                let hidden = (p.embed_dim as f32 * p.mlp_ratio) as usize;
                (hidden * p.convffn_kernel_size * p.convffn_kernel_size) as u64
            }
            // The CCM's 3x3 at the expanded width; the SAFM depthwises are a
            // quarter of the channels each and run on pooled maps.
            ArchParams::Safmn { dim, hidden, .. } => ((*dim).max(*hidden) * 9) as u64,
            // UNet2's deepest stage: a 3x3 over 256 channels.
            ArchParams::RealCugan { .. } => 256 * 9,
            // The gated feed-forward's depthwise runs at twice the width.
            ArchParams::OmniSr { num_feat, .. } => (num_feat * 2 * 9) as u64,
        }
    }

    /// Estimated peak working set at a tile size, in bytes.
    ///
    /// The larger of the convolution and attention terms, each multiplied by
    /// how many such buffers are live at peak. Both multipliers are **fitted to
    /// measured peak RSS**, not reasoned from the graph: how well the arena
    /// reuses a transient is a property of the backend's memory planner, not of
    /// this crate.
    ///
    /// Treat this as a planning aid with a factor-of-two error bar, not an
    /// allocator limit. It exists so that a model whose scratch is an order of
    /// magnitude larger than its neighbour's gets a proportionally smaller
    /// tile; `--tile` is the real control.
    pub fn peak_working_set_bytes(&self, tile: usize) -> u64 {
        let px = (tile * tile) as u64;
        let conv = self.conv_live_buffers() * self.widest_im2col_row() * px * 4;
        let attn = self.attention_live_buffers() * self.attention_bytes(tile);
        conv.max(attn)
    }

    /// How many im2col workspaces are live at peak.
    ///
    /// A conv family is a *stack* of its widest convolution, so a dozen are in
    /// flight; a transformer convolves only at the stem, each group's tail and
    /// the upsampler, so its scratch is dwarfed by attention and a nominal
    /// couple is enough. Using one number for both makes a 180-dim SwinIR look
    /// as scratch-hungry as a 28-block RealPLKSR, which it is not.
    fn conv_live_buffers(&self) -> u64 {
        match &self.params {
            // MambaIRv2 is a transformer that convolves like a conv net: each
            // of its layers runs *two* `ConvFFN`s, and each of those is a
            // depthwise 5x5 at twice the embedding width. Over nine groups of
            // six that is 108 such convolutions, not the handful at group
            // boundaries the rest of the lineage has — measured, it wants the
            // conv-net count, and treating it as a transformer under-estimated
            // a real MambaIRv2-L by 5x (1.8 GB against 10.6 GB).
            ArchParams::MambaIr(_) => CONV_LIVE_BUFFERS,
            _ if self.arch.is_transformer() => 2,
            _ => CONV_LIVE_BUFFERS,
        }
    }

    /// How many attention-sized buffers are live at peak.
    ///
    /// Fitted to measured peak RSS, and the two families differ by 3×:
    ///
    /// | model | tile | largest matrix | measured RSS | ratio |
    /// |---|---|---|---|---|
    /// | SwinIR-S ×2 | 256 | 100 MB | 851 MB | 8.5 |
    /// | HAT-L ×4 | 96 | 127 MB | 2.96 GB | 23 |
    /// | HAT-L ×4 | 128 | 226 MB | 4.36 GB | 19 |
    /// | HAT-L ×4 | 176 | 428 MB | 7.32 GB | 17 |
    ///
    /// HAT-L's three points fit `0.200 MB/px² · tile² + 1.12 GB` to within
    /// 1%, so `24` tracks it to 0.8–1.3× across the range — conservative at the
    /// large end, where the fixed term matters least.
    ///
    /// HAT is dearer than the window-attention matrix alone suggests because
    /// every block also carries a channel-attention branch, and each group ends
    /// with an overlapping-cross-attention block holding an unfolded `ows²`
    /// key/value pair on top of its own larger attention matrix.
    fn attention_live_buffers(&self) -> u64 {
        match &self.params {
            ArchParams::Swin(p) if p.hat.is_some() => 24,
            _ => 8,
        }
    }

    /// Bytes in the largest single attention matrix at a given tile size.
    ///
    /// This is what actually decides whether a model fits. It is **not**
    /// governed by parameter count: HAT-L and SwinIR-M are both 180-dim, but
    /// HAT's overlapping cross-attention reads a `ows × ows` neighbourhood per
    /// `ws × ws` query window, so at `ws = 16`, `overlap_ratio = 0.5` its
    /// attention matrix is `(24/16)² = 2.25×` the plain one — and HAT's window
    /// is twice SwinIR's to begin with, which is another 4×.
    ///
    /// Zero for the convolutional families, whose cost is linear in pixels.
    pub fn attention_bytes(&self, tile: usize) -> u64 {
        let b = |n: u64| n * 4;
        match &self.params {
            ArchParams::Swin(p) => {
                let ws = p.window_size.max(1);
                if tile < ws {
                    return 0;
                }
                let nw = (tile / ws).pow(2) as u64;
                let n = (ws * ws) as u64;
                let heads = p.num_heads.iter().copied().max().unwrap_or(1) as u64;
                let plain = b(nw * heads * n * n);
                match &p.hat {
                    Some(h) => {
                        let ows = ws + (ws as f32 * h.overlap_ratio) as usize;
                        plain.max(b(nw * heads * n * (ows * ows) as u64))
                    }
                    None => plain,
                }
            }
            ArchParams::MambaIr(p) => {
                let ws = p.window_size.max(1);
                if tile < ws {
                    return 0;
                }
                let nw = (tile / ws).pow(2) as u64;
                let n = (ws * ws) as u64;
                let heads = p.num_heads.iter().copied().max().unwrap_or(1) as u64;
                b(nw * heads * n * n)
            }
            // Both attentions are per-window and four-headed; the channel
            // flavour's matrix is `(dim/heads)²`, smaller than the spatial
            // one's `n²` at every released configuration.
            ArchParams::OmniSr {
                num_feat,
                window_size,
                ..
            } => {
                let ws = (*window_size).max(1);
                if tile < ws {
                    return 0;
                }
                let nw = (tile / ws).pow(2) as u64;
                let n = (ws * ws) as u64;
                let hd = (num_feat / 4) as u64;
                b(nw * 4 * n * n).max(b(nw * 4 * hd * hd))
            }
            ArchParams::Dat(p) => {
                let n = (p.split_size[0] * p.split_size[1]).max(1) as u64;
                if !((tile * tile) as u64).is_multiple_of(n) {
                    return 0;
                }
                let nw = (tile * tile) as u64 / n;
                // The channel halves are attended separately, each with half
                // the heads.
                let heads = (p.num_heads.iter().copied().max().unwrap_or(2) / 2).max(1) as u64;
                b(nw * heads * n * n)
            }
            _ => 0,
        }
    }

    /// The tile this model should use when the caller does not pick one.
    ///
    /// Shrinks the family's nominal tile — along the window grid — until
    /// [`Self::peak_working_set_bytes`] fits [`MEMORY_BUDGET`].
    pub fn default_tile(&self) -> usize {
        self.tile_for_budget(MEMORY_BUDGET)
    }

    /// The largest window-aligned tile whose estimated peak fits `budget`.
    ///
    /// **A smaller tile is not free.** Only the `(tile − 2·overlap)²` interior
    /// of each tile is kept, so shrinking the tile multiplies redundant work:
    /// HAT-L on a 512×384 input took 539 s at tile 176 but 1028 s at tile 128,
    /// nearly twice as long for 40% less memory. On a machine with headroom,
    /// raise the budget (`RLX_MAX_RAM_BYTES`) rather than accept the default.
    pub fn tile_for_budget(&self, budget: u64) -> usize {
        let m = self.size_multiple().max(1);
        let round = |v: usize| (v / m).max(1) * m;
        let floor = self.min_tile();
        let mut tile = round(self.arch.nominal_tile()).max(floor);
        while tile > floor && self.peak_working_set_bytes(tile) > budget {
            tile -= m;
        }
        tile
    }

    /// The smallest tile this model can be run at.
    ///
    /// For the window-attention and conv families this is "two windows": below
    /// that a tile has no interior left once the halo comes off, and quality
    /// degrades for want of context.
    ///
    /// Real-CUGAN has a *hard* floor on top of that one. It reflect-pads its
    /// input by 14–38 pixels, and a reflection wider than the thing being
    /// reflected is undefined — PyTorch rejects it outright. So the tile must
    /// exceed the pad, which is where the reference's own minimum of 40 for the
    /// fast variant comes from: its pad is 38. The valid-padding shrinkage,
    /// which looks like it should be the binding constraint, never is — the
    /// halo always buys back more than the convolutions consume.
    pub fn min_tile(&self) -> usize {
        let m = self.size_multiple().max(1);
        let round_up = |v: usize| v.div_ceil(m) * m;
        match &self.params {
            // 32 is the reference's floor for the non-fast wrappers, above
            // their pads; the fast one's 40 is `pad + 2` exactly.
            ArchParams::RealCugan { variant, .. } => round_up((variant.input_pad() + 2).max(32)),
            _ => round_up(2 * m),
        }
    }

    /// A one-line human summary, e.g. `SPAN ×4 (48 feat)`.
    pub fn summary(&self) -> String {
        let detail = match &self.params {
            ArchParams::Esrgan {
                num_filters,
                num_blocks,
                plus,
                shuffle_factor,
                ..
            } => {
                let mut d = format!("{num_filters} nf, {num_blocks} nb");
                if *plus {
                    d.push_str(", plus");
                }
                if *shuffle_factor > 1 {
                    d.push_str(&format!(", unshuffle {shuffle_factor}"));
                }
                d
            }
            ArchParams::Compact {
                num_feat, num_conv, ..
            } => format!("{num_feat} feat, {num_conv} conv"),
            ArchParams::Span {
                feature_channels, ..
            } => format!("{feature_channels} feat"),
            ArchParams::SpanV2 {
                feature_channels, ..
            } => format!("{feature_channels} feat"),
            ArchParams::Plksr {
                dim,
                n_blocks,
                kernel_size,
                ..
            } => format!("{dim} dim, {n_blocks} blocks, k{kernel_size}"),
            ArchParams::RealPlksr {
                dim,
                n_blocks,
                kernel_size,
                upsampler,
                ..
            } => {
                let up = if *upsampler == Upsampler::DySample {
                    ", dysample"
                } else {
                    ""
                };
                format!("{dim} dim, {n_blocks} blocks, k{kernel_size}{up}")
            }
            ArchParams::Swin(p) => format!(
                "{} dim, depths {:?}, win {}",
                p.embed_dim, p.depths, p.window_size
            ),
            ArchParams::Dat(p) => format!(
                "{} dim, depths {:?}, split {:?}",
                p.embed_dim, p.depths, p.split_size
            ),
            ArchParams::MambaIr(p) => format!(
                "{} dim, depths {:?}, win {}, d_state {}",
                p.embed_dim, p.depths, p.window_size, p.d_state
            ),
            ArchParams::Safmn {
                dim,
                n_blocks,
                hidden,
            } => format!(
                "{dim} dim, {n_blocks} blocks, ffn x{:.3}",
                *hidden as f32 / *dim as f32
            ),
            ArchParams::OmniSr {
                num_feat,
                res_num,
                block_num,
                window_size,
            } => {
                format!("{num_feat} feat, {res_num} groups × {block_num} blocks, win {window_size}")
            }
            ArchParams::RealCugan { variant, pro } => {
                format!("{}{}", variant.name(), if *pro { ", pro" } else { "" })
            }
        };
        format!("{} ×{} ({detail})", self.arch.name(), self.scale)
    }
}

/// Default ceiling on estimated peak working set, in bytes.
///
/// 2 GiB — aimed at a consumer GPU rather than at the machine this was
/// developed on. Override per run with `RLX_MAX_RAM_BYTES` (the workspace-wide
/// convention) or pin the tile outright with `--tile`; the estimate is a
/// planning aid, not an allocator limit.
pub const MEMORY_BUDGET: u64 = 2 << 30;

/// How many im2col workspaces of the widest convolution are live at peak.
///
/// Fitted, not derived. Measured peak RSS against tile area on a 512×384 input:
/// RealPLKSR ×2 (`4624`-wide rows) came to 0.251 MB per input pixel where one
/// live workspace would be 0.018, and Compact ×4 (`576`-wide) to 0.020 against
/// 0.002 — so roughly a dozen either way. How many a graph actually holds is a
/// property of the backend's memory planner, which is why this is measured.
const CONV_LIVE_BUFFERS: u64 = 12;

fn lcm(a: usize, b: usize) -> usize {
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    if a == 0 || b == 0 {
        a.max(b).max(1)
    } else {
        a / gcd(a, b) * b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trips_through_json() {
        let cfg = ModelConfig {
            arch: Arch::RealPlksr,
            scale: 4,
            in_ch: 3,
            out_ch: 3,
            params: ArchParams::RealPlksr {
                dim: 64,
                n_blocks: 28,
                kernel_size: 17,
                pdim: 16,
                use_ea: true,
                norm_groups: Some(4),
                upsampler: Upsampler::DySample,
                dysample_groups: 4,
                dysample_end_conv: true,
            },
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert_eq!(cfg, serde_json::from_str::<ModelConfig>(&json).unwrap());
    }

    fn swin(arch: Arch, window: usize, heads: usize, hat: Option<HatParams>) -> ModelConfig {
        ModelConfig {
            arch,
            scale: 4,
            in_ch: 3,
            out_ch: 3,
            params: ArchParams::Swin(SwinParams {
                embed_dim: 180,
                depths: vec![6; 6],
                num_heads: vec![heads; 6],
                window_size: window,
                mlp_ratio: 2.0,
                qkv_bias: true,
                patch_norm: true,
                resi_connection: ResiConnection::Conv1,
                upsampler: Upsampler::PixelShuffle,
                num_feat: 64,
                img_range: 1.0,
                hat,
                drct_gc: None,
                v2: false,
            }),
        }
    }

    fn conv(arch: Arch, params: ArchParams) -> ModelConfig {
        ModelConfig {
            arch,
            scale: 4,
            in_ch: 3,
            out_ch: 3,
            params,
        }
    }

    /// A convolutional model has no attention term at all, so its tile is set
    /// purely by im2col scratch.
    #[test]
    fn a_conv_net_has_no_attention_term() {
        let cfg = conv(
            Arch::Span,
            ArchParams::Span {
                feature_channels: 48,
                norm: true,
                img_range: 255.0,
                rgb_mean: [0.4488, 0.4371, 0.4040],
            },
        );
        assert_eq!(cfg.attention_bytes(512), 0);
        assert!(cfg.peak_working_set_bytes(cfg.default_tile()) <= MEMORY_BUDGET);
    }

    /// Kernel *area* drives scratch, not channel width. RealPLKSR's 17×17 over
    /// 16 channels is a 4624-wide im2col row against Compact's 576 at the same
    /// 64-channel body — eight times the scratch, so a much smaller tile. This
    /// is the case a per-family constant gets wrong, and it is why RealPLKSR
    /// measured 0.25 MB per input pixel where Compact measured 0.02.
    #[test]
    fn a_large_kernel_shrinks_the_tile_more_than_a_wide_body() {
        let plksr = conv(
            Arch::RealPlksr,
            ArchParams::RealPlksr {
                dim: 64,
                n_blocks: 28,
                kernel_size: 17,
                pdim: 16,
                use_ea: true,
                norm_groups: None,
                upsampler: Upsampler::DySample,
                dysample_groups: 4,
                dysample_end_conv: true,
            },
        );
        let compact = conv(
            Arch::Compact,
            ArchParams::Compact {
                num_feat: 64,
                num_conv: 32,
                act: CompactAct::PRelu,
            },
        );
        assert!(
            plksr.default_tile() < compact.default_tile(),
            "RealPLKSR {} vs Compact {}",
            plksr.default_tile(),
            compact.default_tile()
        );
        for cfg in [&plksr, &compact] {
            assert!(cfg.peak_working_set_bytes(cfg.default_tile()) <= MEMORY_BUDGET);
        }
    }

    /// HAT-L and SwinIR-M are both 180-dim with six groups of six, yet HAT's
    /// overlapping cross-attention needs a far smaller tile — twice the window
    /// and a 2.25x wider key neighbourhood. A per-family constant cannot express
    /// that, which is why the tile is derived from the memory cost.
    #[test]
    fn the_tile_follows_memory_cost_not_the_family() {
        let swinir = swin(Arch::SwinIr, 8, 6, None);
        let hat = swin(
            Arch::Hat,
            16,
            6,
            Some(HatParams {
                compress_ratio: 3,
                squeeze_factor: 30,
                conv_scale: 0.01,
                overlap_ratio: 0.5,
            }),
        );
        assert!(
            hat.default_tile() < swinir.default_tile(),
            "HAT-L {} vs SwinIR-M {}",
            hat.default_tile(),
            swinir.default_tile()
        );
        for cfg in [&swinir, &hat] {
            let t = cfg.default_tile();
            assert!(t % cfg.size_multiple() == 0, "{t} is off the window grid");
            assert!(
                cfg.peak_working_set_bytes(t) <= MEMORY_BUDGET,
                "{} at tile {t} wants {} MB",
                cfg.summary(),
                cfg.peak_working_set_bytes(t) >> 20
            );
        }
    }

    /// Attention is quadratic in the tile area, so doubling the tile is 4x.
    #[test]
    fn attention_cost_is_quadratic_in_tile_area() {
        let cfg = swin(Arch::SwinIr, 8, 6, None);
        let (a, b) = (cfg.attention_bytes(64), cfg.attention_bytes(128));
        assert_eq!(b, a * 4);
    }

    #[test]
    fn transformers_and_conv_nets_are_classified() {
        assert!(Arch::Hat.is_transformer());
        assert!(!Arch::Span.is_transformer());
    }

    /// DAT's window swaps orientation between blocks, so a tile must be a
    /// multiple of *both* extents — not just the one that happens to be first.
    #[test]
    fn dat_size_multiple_is_the_lcm_of_both_window_extents() {
        let cfg = ModelConfig {
            arch: Arch::Dat,
            scale: 4,
            in_ch: 3,
            out_ch: 3,
            params: ArchParams::Dat(DatParams {
                embed_dim: 180,
                depths: vec![6; 6],
                num_heads: vec![6; 6],
                split_size: [8, 32],
                expansion_factor: 4.0,
                qkv_bias: true,
                resi_connection: ResiConnection::Conv1,
                upsampler: Upsampler::PixelShuffle,
                img_range: 1.0,
            }),
        };
        assert_eq!(cfg.size_multiple(), 32);
    }
}
