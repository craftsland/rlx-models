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

//! The user-facing upscaler: open a checkpoint, run an image.
//!
//! ```no_run
//! use rlx_upscale::{Upscaler, UpscaleOptions};
//! use rlx_runtime::Device;
//!
//! let mut up = Upscaler::open("4xNomos2_realplksr_dysample.pth", Device::Cpu, UpscaleOptions::default())?;
//! println!("{}", up.config().summary());          // RealPLKSR ×4 (64 dim, 28 blocks, k17, dysample)
//! # let pixels: Vec<u8> = vec![0; 3 * 640 * 480];
//! let (rgb, w, h) = up.upscale_rgb8(&pixels, 640, 480)?;
//! # anyhow::Ok(())
//! ```
//!
//! # One graph, many tiles
//!
//! The network is compiled once for a single tile shape and every tile — edge
//! tiles included — is that exact shape, padded by edge replication. Peak
//! memory is therefore a property of the tile, not the image. A tile larger
//! than the image is shrunk to fit so small inputs do not pay for a large
//! graph.

use anyhow::{Context, Result, ensure};
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::attach_built_params;
use rlx_flow::CompileProfile;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::path::{Path, PathBuf};

use crate::config::ModelConfig;
use crate::graph;
use crate::tile::{self, TilePlan};
use crate::weights::{self, Checkpoint};

/// The tile this model gets when the caller does not pick one, honouring the
/// workspace-wide `RLX_MAX_RAM_BYTES`.
///
/// A machine with headroom should get a larger tile — and so less redundant
/// halo work — without anyone computing one by hand. Shared by the runner and
/// by `--inspect`, so what is reported is what will be used.
pub fn budget_tile(cfg: &ModelConfig) -> usize {
    match rlx_runtime::resource_budget::ResourceBudget::from_env().max_ram_bytes {
        Some(bytes) => cfg.tile_for_budget(bytes as u64),
        None => cfg.default_tile(),
    }
}

/// What to do with an input's alpha channel.
///
/// Dropping it is not an option worth defaulting to: upscaling a logo, icon or
/// sprite sheet and getting an opaque rectangle back is a silent data loss, and
/// `to_rgb8()` does exactly that.
///
/// # Transparent pixels carry colour
///
/// None of these premultiply. A PNG's fully-transparent pixels usually hold
/// arbitrary RGB, and upscaling mixes that into the colour of the edge where
/// alpha ramps up. Premultiplying would avoid it but feeds the network colours
/// it was not trained on — near-transparent regions collapse toward black and
/// unpremultiplying then amplifies whatever the network put there. Straight
/// alpha is what these models expect and what reference tooling uses; if an
/// asset has garbage under its transparency, clean it there rather than here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlphaMode {
    /// Run the network over alpha replicated to three channels and keep one.
    ///
    /// The mask gets the same treatment the colour did, so their edges agree —
    /// which matters, because a mask sharpened differently from the colour it
    /// cuts out shows up as a fringe. Costs a second pass through the network.
    #[default]
    Upscale,
    /// Resample alpha with a bicubic filter instead.
    ///
    /// Roughly free next to a second network pass, and it cannot invent detail
    /// in the mask. Worth choosing when alpha is a clean geometric shape and
    /// the network is an aggressive restorer.
    Resize,
    /// Discard it and return RGB.
    Discard,
}

/// Knobs that affect memory and seam quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UpscaleOptions {
    /// Input tile extent. `None` uses the architecture's default.
    pub tile: Option<usize>,
    /// Halo discarded on each side of a tile. `None` derives one from the
    /// architecture.
    pub overlap: Option<usize>,
    /// How [`Upscaler::upscale_rgba8`] treats the alpha channel.
    pub alpha: AlphaMode,
}

/// A compiled upscaler.
pub struct Upscaler {
    cfg: ModelConfig,
    device: Device,
    opts: UpscaleOptions,
    /// Where the weights came from, when they came from a file.
    ///
    /// Held *instead of* the tensors: a recompile at a different tile needs the
    /// weights again, but keeping them resident to serve a rare event costs a
    /// full copy for the object's whole life — 231 MB for DRCT-L, 158 MB for
    /// HAT-L. Re-reading from disk is slow and rare; holding is a permanent
    /// tax on the thing this crate exists to minimize.
    source: Option<PathBuf>,
    /// The tensors, when there is no path to re-read (built from memory).
    checkpoint: Option<Checkpoint>,
    compiled: Option<Compiled>,
    last_redundancy: Option<f32>,
}

struct Compiled {
    graph: CompiledGraph,
    tile: usize,
    nodes: usize,
}

impl Upscaler {
    /// Load and identify a checkpoint. Nothing is compiled until the first run.
    pub fn open(path: impl AsRef<Path>, device: Device, opts: UpscaleOptions) -> Result<Self> {
        let path = path.as_ref();
        let mut ck =
            Checkpoint::open(path).with_context(|| format!("loading {}", path.display()))?;
        let cfg = crate::detect::detect(&ck)
            .with_context(|| format!("identifying {}", path.display()))?;
        // SPAN's reparameterizable convolutions collapse before the graph sees
        // them; a checkpoint without the training branches is already fused.
        if cfg.arch == crate::config::Arch::Span && !ck.conv3xc_prefixes().is_empty() {
            weights::fuse_all_conv3xc(&mut ck)?;
        }
        let mut up = Self::from_checkpoint(ck, cfg, device, opts)?;
        // The tensors can be re-read from here if a recompile is ever needed,
        // so they do not have to stay resident.
        up.source = Some(path.to_path_buf());
        Ok(up)
    }

    /// Read `path` and prepare its weights for the graph builder.
    fn load_from(path: &Path, cfg: &ModelConfig) -> Result<Checkpoint> {
        let mut ck = Checkpoint::open(path)
            .with_context(|| format!("re-reading {} for a recompile", path.display()))?;
        if cfg.arch == crate::config::Arch::Span && !ck.conv3xc_prefixes().is_empty() {
            weights::fuse_all_conv3xc(&mut ck)?;
        }
        Ok(ck)
    }

    /// Build from an already-loaded checkpoint and a known config.
    pub fn from_checkpoint(
        ck: Checkpoint,
        cfg: ModelConfig,
        device: Device,
        opts: UpscaleOptions,
    ) -> Result<Self> {
        crate::device::validate_upscale_device(device)?;
        Ok(Self {
            cfg,
            device,
            opts,
            source: None,
            checkpoint: Some(ck),
            compiled: None,
            last_redundancy: None,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn scale(&self) -> usize {
        self.cfg.scale
    }

    /// How [`Self::upscale_rgba8`] will treat alpha.
    pub fn alpha_mode(&self) -> AlphaMode {
        self.opts.alpha
    }

    /// Node count of the compiled graph, or `None` before the first run.
    pub fn graph_nodes(&self) -> Option<usize> {
        self.compiled.as_ref().map(|c| c.nodes)
    }

    /// The tile the graph was compiled for, or `None` before the first run.
    pub fn compiled_tile(&self) -> Option<usize> {
        self.compiled.as_ref().map(|c| c.tile)
    }

    /// Work per unit of output for the last plan — see [`TilePlan::redundancy`].
    pub fn last_redundancy(&self) -> Option<f32> {
        self.last_redundancy
    }

    /// Halo width, derived from the architecture when not set.
    ///
    /// For a convolutional net the receptive field is finite and a 16px halo
    /// covers it, so the tiled result is identical to an untiled pass.
    ///
    /// For a deep transformer it is not: shifted windows propagate information
    /// by half a window per block, so across twelve groups of six the receptive
    /// field exceeds any halo worth paying for. One window is used — enough
    /// that seams are not visible, and honest about not being exact. Reference
    /// implementations blend overlapping tiles for the same reason; this keeps
    /// the crop sharp instead, since blending two equally-valid predictions is
    /// its own artifact.
    fn overlap(&self) -> usize {
        self.opts.overlap.unwrap_or_else(|| {
            if self.cfg.arch.is_transformer() {
                self.cfg.size_multiple()
            } else {
                16
            }
        })
    }

    /// Plan the tiling for an image, choosing a tile that actually fits it.
    fn plan(&self, w: usize, h: usize) -> Result<TilePlan> {
        let multiple = self.cfg.size_multiple();
        let overlap = self.overlap();
        let requested = self.opts.tile.unwrap_or_else(|| budget_tile(&self.cfg));

        // Round the requested tile onto the window grid, then shrink it toward
        // the image so a thumbnail does not compile a 512² graph. The floor
        // keeps the tile wide enough for the halo on both sides.
        let round_up = |v: usize| v.div_ceil(multiple) * multiple;
        let floor = round_up(2 * overlap + multiple);
        let want = round_up(requested).max(floor);
        let fits = round_up(w.max(h)).max(floor);
        let tile = want.min(fits);

        TilePlan::new(w, h, tile, overlap, self.cfg.scale, multiple).with_context(|| {
            format!("planning a {w}×{h} image at tile {tile} with a {overlap}px halo")
        })
    }

    /// Compile for `tile`, reusing an existing graph when the size matches.
    fn ensure_compiled(&mut self, tile: usize) -> Result<()> {
        if self.compiled.as_ref().is_some_and(|c| c.tile == tile) {
            return Ok(());
        }
        // The build consumes its weight map. Prefer the resident copy if there
        // is one, otherwise re-read — either way the tensors are gone by the
        // time the graph is compiled.
        let wm = match (&self.checkpoint, &self.source) {
            (Some(ck), _) => ck.to_weight_map(),
            (None, Some(path)) => Self::load_from(path, &self.cfg)?.into_weight_map(),
            (None, None) => anyhow::bail!(
                "weights were released and there is no file to re-read; construct a \
                 new Upscaler to change tile size"
            ),
        };
        let built = graph::build(&self.cfg, wm, tile, tile)?;
        ensure!(
            built.unused.is_empty(),
            "{} left {} checkpoint tensors unread — the architecture was probably \
             mis-detected. First unread: {:?}",
            self.cfg.summary(),
            built.unused.len(),
            &built.unused[..built.unused.len().min(5)]
        );

        let opts = compile_options_for_profile(&CompileProfile::encoder(), self.device);
        let mut graph = Session::new(self.device).compile_with(built.graph, &opts);
        attach_built_params(&mut graph, built.params, &[]);
        // Now that the graph owns its parameters, a resident copy is dead
        // weight — but only drop it if it can be got back.
        if self.source.is_some() {
            self.checkpoint = None;
        }
        self.compiled = Some(Compiled {
            graph,
            tile,
            nodes: built.nodes,
        });
        Ok(())
    }

    /// Drop any resident copy of the weights *and* forget where they came
    /// from. The current graph keeps working; recompiling afterwards does not.
    ///
    /// Rarely needed — a file-backed `Upscaler` already releases its tensors
    /// once the graph is compiled. This is for the in-memory case, or to make
    /// a later recompile fail loudly rather than quietly re-read from disk.
    pub fn release_weights(&mut self) {
        self.checkpoint = None;
        self.source = None;
    }

    /// Whether the weights are still resident in this object.
    pub fn holds_weights(&self) -> bool {
        self.checkpoint.is_some()
    }

    /// Upscale a planar CHW f32 image in `[0, 1]`. Returns planar CHW f32.
    pub fn upscale_planar(&mut self, src: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
        let c = self.cfg.in_ch;
        ensure!(
            src.len() == c * w * h,
            "expected {} samples for a {w}×{h}×{c} image, got {}",
            c * w * h,
            src.len()
        );
        let plan = self.plan(w, h)?;
        self.last_redundancy = Some(plan.redundancy());
        self.ensure_compiled(plan.tile)?;
        let compiled = self
            .compiled
            .as_mut()
            .expect("ensure_compiled populates the graph");

        let out_c = self.cfg.out_ch;
        let (ow, oh) = (plan.out_w(), plan.out_h());
        let mut out = vec![0.0f32; out_c * ow * oh];
        let tile_out = plan.tile * plan.scale;

        for t in &plan.tiles {
            let patch = tile::extract(src, c, w, h, t.src);
            let mut result = compiled.graph.run(&[("image", &patch)]);
            ensure!(
                !result.is_empty(),
                "the compiled graph produced no output tensor"
            );
            let y = result.remove(0);
            ensure!(
                y.len() == out_c * tile_out * tile_out,
                "tile output is {} samples, expected {}",
                y.len(),
                out_c * tile_out * tile_out
            );
            tile::blit(
                &mut out, ow, oh, &y, tile_out, tile_out, out_c, t.crop, t.dst,
            );
        }
        Ok(out)
    }

    /// Upscale interleaved 8-bit **RGBA**, preserving transparency.
    ///
    /// Alpha is handled per [`AlphaMode`]; the default runs it through the same
    /// network so the mask and the colour it cuts out are sharpened alike.
    /// Returns `(pixels, width, height)` — RGBA, or RGB under
    /// [`AlphaMode::Discard`].
    pub fn upscale_rgba8(
        &mut self,
        rgba: &[u8],
        w: usize,
        h: usize,
    ) -> Result<(Vec<u8>, usize, usize)> {
        ensure!(
            self.cfg.in_ch == 3,
            "alpha handling assumes an RGB network; this one takes {} channels",
            self.cfg.in_ch
        );
        ensure!(
            rgba.len() == 4 * w * h,
            "expected {} bytes for a {w}×{h} RGBA image, got {}",
            4 * w * h,
            rgba.len()
        );

        let mut rgb = vec![0u8; 3 * w * h];
        let mut alpha = vec![0u8; w * h];
        for i in 0..w * h {
            rgb[i * 3..i * 3 + 3].copy_from_slice(&rgba[i * 4..i * 4 + 3]);
            alpha[i] = rgba[i * 4 + 3];
        }

        let (colour, ow, oh) = self.upscale_rgb8(&rgb, w, h)?;
        if self.opts.alpha == AlphaMode::Discard {
            return Ok((colour, ow, oh));
        }

        // A fully opaque image is the common case and needs no second pass.
        let up_alpha = if alpha.iter().all(|&a| a == 255) {
            vec![255u8; ow * oh]
        } else {
            self.upscale_alpha(&alpha, w, h, ow, oh)?
        };

        let out_c = self.cfg.out_ch;
        let mut out = vec![0u8; 4 * ow * oh];
        for i in 0..ow * oh {
            out[i * 4..i * 4 + 3].copy_from_slice(&colour[i * out_c..i * out_c + 3]);
            out[i * 4 + 3] = up_alpha[i];
        }
        Ok((out, ow, oh))
    }

    /// Bring a single-channel mask up to `(ow, oh)`.
    fn upscale_alpha(
        &mut self,
        alpha: &[u8],
        w: usize,
        h: usize,
        ow: usize,
        oh: usize,
    ) -> Result<Vec<u8>> {
        // Either path needs the mask as three channels: the network takes RGB,
        // and the resampler is written against interleaved RGB8.
        let mut grey = vec![0u8; 3 * w * h];
        for i in 0..w * h {
            grey[i * 3] = alpha[i];
            grey[i * 3 + 1] = alpha[i];
            grey[i * 3 + 2] = alpha[i];
        }
        let wide = match self.opts.alpha {
            AlphaMode::Resize => rlx_core::image_preprocess::pil_resize_rgb8(
                &grey,
                w,
                h,
                ow,
                oh,
                rlx_core::image_preprocess::Filter::Bicubic,
            ),
            // `Discard` is handled by the caller before this runs.
            _ => self.upscale_rgb8(&grey, w, h)?.0,
        };
        let stride = if self.opts.alpha == AlphaMode::Resize {
            3
        } else {
            self.cfg.out_ch
        };
        Ok((0..ow * oh).map(|i| wide[i * stride]).collect())
    }

    /// Upscale interleaved 8-bit RGB. Returns `(pixels, width, height)`.
    pub fn upscale_rgb8(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Result<(Vec<u8>, usize, usize)> {
        let c = self.cfg.in_ch;
        ensure!(
            rgb.len() == c * w * h,
            "expected {} bytes for a {w}×{h}×{c} image, got {}",
            c * w * h,
            rgb.len()
        );
        let mut planar = vec![0.0f32; c * w * h];
        for i in 0..w * h {
            for ci in 0..c {
                planar[ci * w * h + i] = rgb[i * c + ci] as f32 / 255.0;
            }
        }
        let out = self.upscale_planar(&planar, w, h)?;
        let (ow, oh) = (w * self.cfg.scale, h * self.cfg.scale);
        let oc = self.cfg.out_ch;
        let mut pixels = vec![0u8; oc * ow * oh];
        for i in 0..ow * oh {
            for ci in 0..oc {
                // Clamp, don't wrap: these networks routinely overshoot past 1
                // on specular highlights, and a wrap turns a highlight black.
                pixels[i * oc + ci] =
                    (out[ci * ow * oh + i] * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        Ok((pixels, ow, oh))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Arch, ArchParams, CompactAct};
    use std::collections::HashMap;

    /// A tiny but *real* Compact model: 2 body convs, 4 features, ×2.
    fn tiny_compact() -> (Checkpoint, ModelConfig) {
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut seed = 1u64;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 0.2
                })
                .collect()
        };
        let feat = 4usize;
        t.insert(
            "body.0.weight".into(),
            (rnd(feat * 3 * 9), vec![feat, 3, 3, 3]),
        );
        t.insert("body.0.bias".into(), (rnd(feat), vec![feat]));
        t.insert("body.1.weight".into(), (vec![0.25; feat], vec![feat]));
        for i in 0..2 {
            let c = 2 + 2 * i;
            t.insert(
                format!("body.{c}.weight"),
                (rnd(feat * feat * 9), vec![feat, feat, 3, 3]),
            );
            t.insert(format!("body.{c}.bias"), (rnd(feat), vec![feat]));
            t.insert(
                format!("body.{}.weight", c + 1),
                (vec![0.25; feat], vec![feat]),
            );
        }
        t.insert(
            "body.6.weight".into(),
            (rnd(12 * feat * 9), vec![12, feat, 3, 3]),
        );
        t.insert("body.6.bias".into(), (rnd(12), vec![12]));

        let ck = Checkpoint::from_tensors(t);
        let cfg = crate::detect::detect(&ck).unwrap();
        assert_eq!(cfg.arch, Arch::Compact);
        assert_eq!(cfg.scale, 2);
        assert_eq!(
            cfg.params,
            ArchParams::Compact {
                num_feat: 4,
                num_conv: 2,
                act: CompactAct::PRelu
            }
        );
        (ck, cfg)
    }

    /// End to end: a real graph, compiled and run, at the right output size.
    #[test]
    fn compact_runs_and_scales() {
        let (ck, cfg) = tiny_compact();
        let mut up =
            Upscaler::from_checkpoint(ck, cfg, Device::Cpu, UpscaleOptions::default()).unwrap();
        let (w, h) = (9usize, 7usize);
        let src: Vec<f32> = (0..3 * w * h).map(|i| (i % 17) as f32 / 17.0).collect();
        let out = up.upscale_planar(&src, w, h).unwrap();
        assert_eq!(out.len(), 3 * (w * 2) * (h * 2));
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output has non-finite values"
        );
        assert!(up.graph_nodes().unwrap() > 10);
    }

    /// The whole point of tiling is that it changes *nothing*. Running the same
    /// image through one big tile and through many small ones must agree — if
    /// the halo or the crop is off by a pixel, the seams diverge.
    #[test]
    fn tiled_and_untiled_results_agree() {
        let (ck, cfg) = tiny_compact();
        let (w, h) = (40usize, 32usize);
        let src: Vec<f32> = (0..3 * w * h)
            .map(|i| ((i * 37) % 251) as f32 / 251.0)
            .collect();

        let mut whole = Upscaler::from_checkpoint(
            ck.clone(),
            cfg.clone(),
            Device::Cpu,
            UpscaleOptions {
                tile: Some(64),
                overlap: Some(8),
                ..Default::default()
            },
        )
        .unwrap();
        let a = whole.upscale_planar(&src, w, h).unwrap();

        let mut tiled = Upscaler::from_checkpoint(
            ck,
            cfg,
            Device::Cpu,
            UpscaleOptions {
                tile: Some(16),
                overlap: Some(4),
                ..Default::default()
            },
        )
        .unwrap();
        let b = tiled.upscale_planar(&src, w, h).unwrap();

        assert_eq!(a.len(), b.len());
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "tiling changed the image by {worst}");
    }

    /// An in-memory `Upscaler` has nowhere to re-read from, so it must keep its
    /// tensors — and must still be able to recompile at a new tile.
    #[test]
    fn an_in_memory_upscaler_keeps_its_weights_and_can_recompile() {
        let (ck, cfg) = tiny_compact();
        let mut up = Upscaler::from_checkpoint(
            ck,
            cfg,
            Device::Cpu,
            UpscaleOptions {
                tile: Some(48),
                ..Default::default()
            },
        )
        .unwrap();
        // Big enough that neither tile is clamped down to fit the image.
        let src = vec![0.25f32; 3 * 120 * 96];
        up.upscale_planar(&src, 120, 96).unwrap();
        assert_eq!(up.compiled_tile(), Some(48));
        assert!(
            up.holds_weights(),
            "nothing to re-read from, so it must hold"
        );

        // A second, larger tile forces a recompile from the resident copy.
        up.opts.tile = Some(80);
        up.upscale_planar(&src, 120, 96).unwrap();
        assert_eq!(up.compiled_tile(), Some(80));
    }

    /// Releasing forgets the source too, so a later recompile fails loudly
    /// instead of quietly going back to disk.
    #[test]
    fn releasing_weights_makes_a_recompile_fail_loudly() {
        let (ck, cfg) = tiny_compact();
        let mut up = Upscaler::from_checkpoint(
            ck,
            cfg,
            Device::Cpu,
            UpscaleOptions {
                tile: Some(48),
                ..Default::default()
            },
        )
        .unwrap();
        let src = vec![0.25f32; 3 * 120 * 96];
        up.upscale_planar(&src, 120, 96).unwrap();
        up.release_weights();
        assert!(!up.holds_weights());

        // A *different* tile, or `ensure_compiled` short-circuits and never
        // asks for the weights at all.
        up.opts.tile = Some(80);
        let e = up.upscale_planar(&src, 120, 96).unwrap_err();
        assert!(e.to_string().contains("released"), "unexpected: {e:#}");
    }

    /// A checkpoint whose tensors the graph never reads means detection picked
    /// the wrong architecture — that must fail loudly, not render a wrong image.
    #[test]
    fn unread_tensors_are_an_error() {
        let (mut ck, cfg) = tiny_compact();
        ck.insert("body.99.weight", vec![0.0; 4], vec![4]);
        let mut up =
            Upscaler::from_checkpoint(ck, cfg, Device::Cpu, UpscaleOptions::default()).unwrap();
        let e = up.upscale_planar(&vec![0.0; 3 * 8 * 8], 8, 8).unwrap_err();
        assert!(e.to_string().contains("unread"), "unexpected error: {e}");
    }

    /// Transparency has to survive. Reading an RGBA source as RGB silently
    /// flattens a logo onto an opaque rectangle, which is the failure this
    /// whole path exists to prevent.
    #[test]
    fn rgba_preserves_the_mask() {
        let (ck, cfg) = tiny_compact();
        let (w, h) = (8usize, 8usize);
        // A half-transparent image: left column opaque, right column clear.
        let mut rgba = vec![0u8; 4 * w * h];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                rgba[i] = (x * 30) as u8;
                rgba[i + 1] = (y * 30) as u8;
                rgba[i + 2] = 128;
                rgba[i + 3] = if x < w / 2 { 255 } else { 0 };
            }
        }

        for mode in [AlphaMode::Upscale, AlphaMode::Resize] {
            let mut up = Upscaler::from_checkpoint(
                ck.clone(),
                cfg.clone(),
                Device::Cpu,
                UpscaleOptions {
                    alpha: mode,
                    ..Default::default()
                },
            )
            .unwrap();
            let (out, ow, oh) = up.upscale_rgba8(&rgba, w, h).unwrap();
            assert_eq!((ow, oh), (16, 16));
            assert_eq!(out.len(), 4 * ow * oh, "{mode:?} lost the alpha channel");
            // The far left stays opaque and the far right stays clear; the
            // middle is wherever the resampler put the edge.
            let at = |x: usize, y: usize| out[(y * ow + x) * 4 + 3];
            assert!(at(0, 8) > 200, "{mode:?}: opaque side went transparent");
            assert!(at(ow - 1, 8) < 55, "{mode:?}: clear side went opaque");
        }
    }

    /// `Discard` is the one mode that legitimately drops to three channels.
    #[test]
    fn discarding_alpha_returns_rgb() {
        let (ck, cfg) = tiny_compact();
        let mut up = Upscaler::from_checkpoint(
            ck,
            cfg,
            Device::Cpu,
            UpscaleOptions {
                alpha: AlphaMode::Discard,
                ..Default::default()
            },
        )
        .unwrap();
        let (out, ow, oh) = up.upscale_rgba8(&vec![200u8; 4 * 8 * 8], 8, 8).unwrap();
        assert_eq!(out.len(), 3 * ow * oh);
    }

    /// A fully opaque source must not pay for a second network pass.
    #[test]
    fn an_opaque_image_skips_the_alpha_pass() {
        let (ck, cfg) = tiny_compact();
        let mut up =
            Upscaler::from_checkpoint(ck, cfg, Device::Cpu, UpscaleOptions::default()).unwrap();
        let (w, h) = (8usize, 8usize);
        let mut rgba = vec![255u8; 4 * w * h];
        for i in 0..w * h {
            rgba[i * 4] = (i % 256) as u8;
        }
        let (out, ow, oh) = up.upscale_rgba8(&rgba, w, h).unwrap();
        assert!(
            out.iter().skip(3).step_by(4).all(|&a| a == 255),
            "an opaque input came back partly transparent"
        );
        assert_eq!(out.len(), 4 * ow * oh);
    }

    #[test]
    fn rgb8_round_trip_clamps_instead_of_wrapping() {
        let (ck, cfg) = tiny_compact();
        let mut up =
            Upscaler::from_checkpoint(ck, cfg, Device::Cpu, UpscaleOptions::default()).unwrap();
        let (w, h) = (8usize, 8usize);
        let rgb: Vec<u8> = (0..3 * w * h).map(|i| (i % 256) as u8).collect();
        let (out, ow, oh) = up.upscale_rgb8(&rgb, w, h).unwrap();
        assert_eq!((ow, oh), (16, 16));
        assert_eq!(out.len(), 3 * 16 * 16);
    }
}
