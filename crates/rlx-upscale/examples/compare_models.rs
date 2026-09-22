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

//! Run every checkpoint in a folder over one image and put the results side by
//! side, so a model can be chosen on evidence rather than on filename.
//!
//! Each model gets three numbers and a PNG:
//!
//! * **time** and an **estimated peak** — what it costs. The estimate is
//!   per-model; measured RSS is a process-wide high-water mark and would show
//!   every later model inheriting the largest earlier one, so `tile_sweep`
//!   measures that properly in a fresh process per configuration.
//! * **fidelity `r`** — correlation between the result downscaled back by the
//!   scale factor and the input. This measures *structure*, not taste.
//!
//! # Why `r` and not PSNR
//!
//! Read the fidelity column as "how much does this model change the picture",
//! **not** as "how good is it". The two kinds of model here differ on purpose:
//!
//! * Models trained on clean bicubic degradation reproduce the input almost
//!   exactly — SPANV2 round-trips at `r = 0.9998`, 50 dB.
//! * Restoration models are trained to *change* it: remove compression
//!   artifacts, denoise, re-saturate. A 2× anime SPAN round-trips at
//!   `r = 0.95`, 13.8 dB, while producing a visibly better picture than the one
//!   scoring 50.
//!
//! PSNR cannot tell the second kind from a broken port, which is why the test
//! suite uses `r`: it is invariant to the per-channel gain and offset a
//! restoration model applies, and is destroyed by any spatial scrambling. A
//! value near 1 means the geometry survived; a low one means either heavy
//! restyling or a bug, and the PNG next to it tells you which.
//!
//! Look at the images. This table narrows the field; it does not rank it.
//!
//! Run:
//!   cargo run --release -p rlx-upscale --example compare_models -- \
//!     --models ~/models --image photo.png --out ./compare
//!
//!   cargo run --release -p rlx-upscale --features metal --example compare_models -- \
//!     --models ~/models --image photo.png --out ./compare --device metal

use anyhow::{Context, Result, bail};
use rlx_cli::req;
use rlx_runtime::Device;
use rlx_upscale::{UpscaleOptions, Upscaler, parse_upscale_device};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut models: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut out = PathBuf::from("./compare");
    let mut device = Device::Cpu;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--models" => models = Some(PathBuf::from(req(&args, &mut i)?)),
            "--image" => image = Some(PathBuf::from(req(&args, &mut i)?)),
            "--out" => out = PathBuf::from(req(&args, &mut i)?),
            "--device" => device = parse_upscale_device(&req(&args, &mut i)?)?,
            other => bail!("unknown argument {other:?}"),
        }
    }
    let models = models.context("--models is required (a directory of checkpoints)")?;
    let image = image.context("--image is required")?;
    std::fs::create_dir_all(&out)?;

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&models)
        .with_context(|| format!("reading {}", models.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("pth" | "pt" | "bin" | "ckpt" | "safetensors")
            )
        })
        .collect();
    paths.sort();
    anyhow::ensure!(!paths.is_empty(), "no checkpoints in {}", models.display());

    let img = image::open(&image)
        .with_context(|| format!("opening {}", image.display()))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let src = img.as_raw().clone();
    println!("input {w}×{h} on {device:?}\n");
    println!(
        "{:<34} {:>5} {:>6} {:>9} {:>9} {:>8}",
        "model", "scale", "tile", "time", "est peak", "fidelity"
    );

    for path in &paths {
        let label = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        // A folder of downloads will contain things this crate does not read;
        // name them and carry on rather than abandoning the comparison.
        let mut up = match Upscaler::open(path, device, UpscaleOptions::default()) {
            Ok(u) => u,
            Err(e) => {
                println!("{:<34} {}", truncate(&label, 34), first_line(&e));
                continue;
            }
        };
        let cfg = up.config().clone();
        let scale = cfg.scale;
        let t0 = Instant::now();
        let (pixels, ow, oh) = match up.upscale_rgb8(&src, w, h) {
            Ok(v) => v,
            Err(e) => {
                println!("{:<34} {}", truncate(&label, 34), first_line(&e));
                continue;
            }
        };
        let secs = t0.elapsed().as_secs_f64();
        let tile = up.compiled_tile().unwrap_or(0);
        let est = cfg.peak_working_set_bytes(tile) >> 20;

        let fidelity = if scale > 1 {
            format!(
                "{:.4}",
                correlation(&downscale(&pixels, ow, oh, scale), &src)
            )
        } else {
            "—".into()
        };
        println!(
            "{:<34} {:>5} {:>6} {:>8.2}s {:>6} MB {:>8}",
            truncate(&label, 34),
            format!("×{scale}"),
            tile,
            secs,
            est,
            fidelity,
        );

        let dest = out.join(format!("{label}.png"));
        image::RgbImage::from_raw(ow as u32, oh as u32, pixels)
            .context("output buffer does not match its declared size")?
            .save(&dest)?;
    }
    println!(
        "\nimages in {} — compare those, not just the table.\n\
         for measured memory run `--example tile_sweep`, which isolates each \
         configuration in its own process.",
        out.display()
    );
    Ok(())
}

fn first_line(e: &anyhow::Error) -> String {
    format!("{e:#}")
        .lines()
        .next()
        .unwrap_or("failed")
        .to_string()
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    let keep: String = s.chars().skip(s.chars().count() - (n - 1)).collect();
    format!("…{keep}")
}

/// Box-downscale interleaved RGB by an integer factor.
fn downscale(px: &[u8], w: usize, h: usize, f: usize) -> Vec<u8> {
    let (ow, oh) = (w / f, h / f);
    let mut out = vec![0u8; 3 * ow * oh];
    for y in 0..oh {
        for x in 0..ow {
            for c in 0..3 {
                let mut acc = 0u32;
                for dy in 0..f {
                    for dx in 0..f {
                        acc += px[(((y * f + dy) * w) + x * f + dx) * 3 + c] as u32;
                    }
                }
                out[(y * ow + x) * 3 + c] = (acc / (f * f) as u32) as u8;
            }
        }
    }
    out
}

/// Mean per-channel Pearson correlation — invariant to the gain and offset a
/// restoration model applies, destroyed by any spatial rearrangement.
fn correlation(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len()) / 3;
    let mut total = 0.0;
    for c in 0..3 {
        let (mut mx, mut my) = (0.0, 0.0);
        for i in 0..n {
            mx += a[i * 3 + c] as f64;
            my += b[i * 3 + c] as f64;
        }
        mx /= n as f64;
        my /= n as f64;
        let (mut num, mut dx, mut dy) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let (u, v) = (a[i * 3 + c] as f64 - mx, b[i * 3 + c] as f64 - my);
            num += u * v;
            dx += u * u;
            dy += v * v;
        }
        total += if dx > f64::EPSILON && dy > f64::EPSILON {
            num / (dx * dy).sqrt()
        } else {
            f64::from(dx <= f64::EPSILON && dy <= f64::EPSILON)
        };
    }
    total / 3.0
}
