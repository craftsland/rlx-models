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

//! `rlx-upscale` command line: identify a checkpoint, or run it over images.

use anyhow::{Context, Result, bail, ensure};
use rlx_cli::req;
use rlx_runtime::Device;
use std::path::PathBuf;

use crate::device::parse_upscale_device;
use crate::runner::{UpscaleOptions, Upscaler};
use crate::weights::Checkpoint;

const HELP: &str = "\
rlx-upscale — single-image super-resolution on RLX

Usage:
  rlx-upscale --model <model.pth> --image <in> [--image <in> ...] [--out <dir>]
  rlx-upscale --model <model.pth> --inspect

Model:
  --model <path>     A community upscaler checkpoint (.pth / .pt / .safetensors).
                     The architecture and every hyperparameter are recovered
                     from the state dict — these files carry no config.
                     Supported: Compact, SPAN, SPANV2, PLKSR, RealPLKSR,
                     SwinIR, HAT, DRCT, DAT.

Running:
  --image <path>     Image to upscale. Repeat for a batch.
  --out <dir>        Where to write results [default: alongside each input].
  --suffix <s>       Appended to the output stem [default: _upscaled].
  --device <dev>     cpu | metal | mlx | cuda | rocm | gpu (wgpu) | vulkan
                     [default: cpu]

Memory:
  --tile <n>         Input tile extent. Peak memory is set by this, not by the
                     image: every tile is the same shape and the network is
                     compiled once. Smaller is leaner and slower.
                     [default: the largest window-aligned tile whose estimated
                     peak fits 2 GiB, or RLX_MAX_RAM_BYTES when set. A smaller
                     tile is not free: only the interior survives the crop, so
                     halving it multiplies redundant work. `--inspect` prints
                     what a checkpoint works out to]
  --overlap <n>      Halo discarded on each side of a tile. Large enough to
                     cover the receptive field means the result is identical to
                     an untiled pass. [default: 16, or one window]

Alpha:
  --alpha <mode>     upscale | resize | discard [default: upscale]
                     Transparency is preserved when the input has it.
                     `upscale` runs the mask through the same network so its
                     edges match the colour's; `resize` uses a bicubic filter
                     instead, which is far cheaper and cannot invent detail in
                     the mask.

Other:
  --inspect          Print the detected architecture as JSON and exit.
  -h, --help         This message.
";

/// Entry point for the `rlx-upscale` binary.
pub fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{HELP}");
        return Ok(());
    }

    let mut model: Option<PathBuf> = None;
    let mut images: Vec<PathBuf> = Vec::new();
    let mut out_dir: Option<PathBuf> = None;
    let mut suffix = String::from("_upscaled");
    let mut device = Device::Cpu;
    let mut opts = UpscaleOptions::default();
    let mut inspect = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => model = Some(PathBuf::from(req(&args, &mut i)?)),
            "--image" => images.push(PathBuf::from(req(&args, &mut i)?)),
            "--out" => out_dir = Some(PathBuf::from(req(&args, &mut i)?)),
            "--suffix" => suffix = req(&args, &mut i)?,
            "--device" => device = parse_upscale_device(&req(&args, &mut i)?)?,
            "--tile" => opts.tile = Some(req(&args, &mut i)?.parse().context("--tile")?),
            "--overlap" => opts.overlap = Some(req(&args, &mut i)?.parse().context("--overlap")?),
            "--alpha" => {
                let m = req(&args, &mut i)?;
                opts.alpha = match m.as_str() {
                    "upscale" => crate::runner::AlphaMode::Upscale,
                    "resize" => crate::runner::AlphaMode::Resize,
                    "discard" => crate::runner::AlphaMode::Discard,
                    other => bail!("unknown --alpha mode {other:?} (upscale | resize | discard)"),
                };
            }
            "--inspect" => {
                inspect = true;
                i += 1;
            }
            other => bail!("unknown argument {other:?} (try --help)"),
        }
    }

    let model = model.context("--model is required")?;

    if inspect {
        let ck = Checkpoint::open(&model)?;
        let cfg = crate::detect::detect(&ck)?;
        println!("{}", serde_json::to_string_pretty(&cfg)?);
        let tile = crate::runner::budget_tile(&cfg);
        let attn = cfg.attention_bytes(tile);
        eprintln!(
            "{} — {} tensors, input must be a multiple of {}, default tile {tile}{}",
            cfg.summary(),
            ck.len(),
            cfg.size_multiple(),
            if attn > 0 {
                format!(" ({} MB peak attention)", attn >> 20)
            } else {
                String::new()
            }
        );
        return Ok(());
    }

    ensure!(!images.is_empty(), "no --image given (try --help)");

    let mut up = Upscaler::open(&model, device, opts)?;
    eprintln!("{} on {device:?}", up.config().summary());

    for path in &images {
        let out = run_one(&mut up, path, out_dir.as_deref(), &suffix)?;
        println!("{}", out.display());
    }
    Ok(())
}

#[cfg(feature = "image-io")]
fn run_one(
    up: &mut Upscaler,
    path: &std::path::Path,
    out_dir: Option<&std::path::Path>,
    suffix: &str,
) -> Result<PathBuf> {
    let img = image::open(path).with_context(|| format!("opening {}", path.display()))?;
    ensure!(
        up.config().in_ch == 3,
        "this model takes {} input channels; only RGB is wired up",
        up.config().in_ch
    );

    // Transparency is preserved when the source has it. Reading everything as
    // RGB would silently flatten a logo or sprite onto an opaque rectangle.
    let alpha = img.color().has_alpha();
    let (pixels, ow, oh, channels) = if alpha {
        let rgba = img.to_rgba8();
        let (w, h) = (rgba.width() as usize, rgba.height() as usize);
        let (px, ow, oh) = up.upscale_rgba8(rgba.as_raw(), w, h)?;
        (
            px,
            ow,
            oh,
            if up.alpha_mode() == crate::runner::AlphaMode::Discard {
                3
            } else {
                4
            },
        )
    } else {
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let (px, ow, oh) = up.upscale_rgb8(rgb.as_raw(), w, h)?;
        (px, ow, oh, 3)
    };
    let (w, h) = (img.width() as usize, img.height() as usize);
    eprintln!(
        "  {}: {w}×{h} → {ow}×{oh} (tile {}, {:.1}× halo work, {} graph nodes)",
        path.file_name().unwrap_or_default().to_string_lossy(),
        up.compiled_tile().unwrap_or(0),
        up.last_redundancy().unwrap_or(1.0),
        up.graph_nodes().unwrap_or(0),
    );

    let stem = path
        .file_stem()
        .context("input has no file name")?
        .to_string_lossy()
        .into_owned();
    // Always PNG: these models produce detail that JPEG then throws away, and
    // re-compressing an upscale is the one thing users notice.
    let name = format!("{stem}{suffix}.png");
    let dest = match out_dir {
        Some(d) => {
            std::fs::create_dir_all(d)?;
            d.join(name)
        }
        None => path.with_file_name(name),
    };
    if channels == 4 {
        image::RgbaImage::from_raw(ow as u32, oh as u32, pixels)
            .context("output buffer does not match its declared size")?
            .save(&dest)
    } else {
        image::RgbImage::from_raw(ow as u32, oh as u32, pixels)
            .context("output buffer does not match its declared size")?
            .save(&dest)
    }
    .with_context(|| format!("writing {}", dest.display()))?;
    Ok(dest)
}

#[cfg(not(feature = "image-io"))]
fn run_one(
    _up: &mut Upscaler,
    _path: &std::path::Path,
    _out_dir: Option<&std::path::Path>,
    _suffix: &str,
) -> Result<PathBuf> {
    bail!("rebuild with the `image-io` feature to read and write images")
}
