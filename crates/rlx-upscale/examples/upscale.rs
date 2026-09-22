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

//! The whole API, end to end, in about thirty lines.
//!
//! Point it at any community upscaler checkpoint — the file carries no config,
//! so the architecture and every hyperparameter are recovered from the tensor
//! names and shapes alone — and at an image.
//!
//! Two things worth knowing beyond `open` / `upscale_rgb8`:
//!
//! * The tile is chosen from the model's memory cost, and peak memory follows
//!   the *tile*, not the image. A 6000×4000 photo and a thumbnail cost the same
//!   per step.
//! * A source with transparency goes through `upscale_rgba8`. Reading
//!   everything as RGB is the easy mistake — `to_rgb8()` will happily flatten a
//!   logo onto an opaque rectangle and say nothing.
//!
//! Run:
//!   cargo run --release -p rlx-upscale --example upscale -- \
//!     --model 4xNomos2_realplksr_dysample.pth --image photo.jpg
//!
//! With a GPU backend:
//!   cargo run --release -p rlx-upscale --features metal --example upscale -- \
//!     --model model.pth --image photo.jpg --device metal

use anyhow::{Context, Result, bail};
use rlx_cli::req;
use rlx_runtime::Device;
use rlx_upscale::{AlphaMode, UpscaleOptions, Upscaler, parse_upscale_device};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut device = Device::Cpu;
    let mut opts = UpscaleOptions::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => model = Some(PathBuf::from(req(&args, &mut i)?)),
            "--image" => image = Some(PathBuf::from(req(&args, &mut i)?)),
            "--out" => out = Some(PathBuf::from(req(&args, &mut i)?)),
            "--device" => device = parse_upscale_device(&req(&args, &mut i)?)?,
            "--tile" => opts.tile = Some(req(&args, &mut i)?.parse().context("--tile")?),
            "--alpha" => {
                opts.alpha = match req(&args, &mut i)?.as_str() {
                    "upscale" => AlphaMode::Upscale,
                    "resize" => AlphaMode::Resize,
                    "discard" => AlphaMode::Discard,
                    other => bail!("--alpha must be upscale | resize | discard, got {other:?}"),
                }
            }
            other => bail!("unknown argument {other:?}"),
        }
    }
    let model = model.context("--model is required")?;
    let image = image.context("--image is required")?;

    // ── 1. open ───────────────────────────────────────────────────────
    // Identifies the architecture, recovers its hyperparameters, and (for SPAN)
    // folds the reparameterizable convolutions. Nothing is compiled yet.
    let mut up = Upscaler::open(&model, device, opts)?;
    let cfg = up.config().clone();
    println!("{}", cfg.summary());
    println!(
        "  scale ×{}, input must tile by {}",
        cfg.scale,
        cfg.size_multiple()
    );

    // ── 2. read ───────────────────────────────────────────────────────
    let img = image::open(&image).with_context(|| format!("opening {}", image.display()))?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let has_alpha = img.color().has_alpha();

    // ── 3. run ────────────────────────────────────────────────────────
    // The graph is compiled on this first call, for one tile shape, and reused
    // for every tile including the edges (which are edge-replicated to fit).
    let t0 = Instant::now();
    let (pixels, ow, oh) = if has_alpha {
        up.upscale_rgba8(img.to_rgba8().as_raw(), w, h)?
    } else {
        up.upscale_rgb8(img.to_rgb8().as_raw(), w, h)?
    };
    let elapsed = t0.elapsed();
    // `Discard` drops back to three channels even from an RGBA source.
    let rgba_out = has_alpha && up.alpha_mode() != AlphaMode::Discard;

    println!(
        "  {w}×{h} → {ow}×{oh} in {:.2}s on {device:?}",
        elapsed.as_secs_f64()
    );
    println!(
        "  tile {}, {:.1}× halo work, {} graph nodes, {} MB peak RSS",
        up.compiled_tile().unwrap_or(0),
        up.last_redundancy().unwrap_or(1.0),
        up.graph_nodes().unwrap_or(0),
        rlx_core::asr_bench::peak_rss_mb(),
    );

    // ── 4. write ──────────────────────────────────────────────────────
    // PNG, always: these models add detail that JPEG would immediately discard.
    let dest = out.unwrap_or_else(|| {
        let stem = image.file_stem().unwrap_or_default().to_string_lossy();
        image.with_file_name(format!("{stem}_upscaled.png"))
    });
    if let Some(parent) = dest.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    if rgba_out {
        image::RgbaImage::from_raw(ow as u32, oh as u32, pixels)
            .context("output buffer does not match its declared size")?
            .save(&dest)
    } else {
        image::RgbImage::from_raw(ow as u32, oh as u32, pixels)
            .context("output buffer does not match its declared size")?
            .save(&dest)
    }
    .with_context(|| format!("writing {}", dest.display()))?;
    println!(
        "  wrote {} ({})",
        dest.display(),
        if rgba_out { "RGBA" } else { "RGB" }
    );
    Ok(())
}
