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

//! Build a labelled comparison sheet: every model against every image, cropped
//! to the same region, at the scale each model actually produces.
//!
//! `compare_models` gives you numbers and a folder of PNGs. This gives you the
//! thing you actually look at — one sheet per input, panels side by side with a
//! nearest-neighbour baseline first, so the question "what did the network add"
//! has a visible answer.
//!
//! # Comparing fairly
//!
//! Models of different scale cannot share a panel size without someone lying.
//! Each sheet therefore fixes a **source** region and renders it at each
//! model's native output size, so a ×2 panel really is half the pixels of a ×4
//! one. Nothing is resampled for display except the baseline, which is nearest
//! by definition. The label carries the scale so a smaller panel reads as
//! "fewer pixels", not "worse".
//!
//! Inputs are read as RGB: a sheet is composited on an opaque background, so
//! flattening transparency here is deliberate rather than the silent
//! `to_rgb8()` mistake the library itself avoids. Use `rlx-upscale --alpha` or
//! the `upscale` example when the mask matters.
//!
//! `--crop WxH` takes a centre crop of the source *before* running anything, so
//! the models only process what the sheet will show. On a 448×640 photo with a
//! dozen models that is the difference between a coffee and a glance, and a
//! 160×120 region is where texture differences are visible anyway.
//!
//! Run:
//!   cargo run --release -p rlx-upscale --features metal --example gallery -- \
//!     --models ~/models --images ~/pics --out ./sheets --device metal
//!
//!   # one tight region, to inspect texture
//!   cargo run --release -p rlx-upscale --features metal --example gallery -- \
//!     --models ~/models --images photo.png --out ./sheets --crop 160x120

use anyhow::{Context, Result, bail};
use rlx_cli::req;
use rlx_runtime::Device;
use rlx_upscale::{UpscaleOptions, Upscaler, parse_upscale_device};
use std::path::{Path, PathBuf};
use std::time::Instant;

mod font;

/// An RGB8 image with its dimensions.
struct Img {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

impl Img {
    fn crop(&self, x0: usize, y0: usize, w: usize, h: usize) -> Img {
        let mut px = vec![0u8; w * h * 3];
        for y in 0..h {
            let s = ((y0 + y.min(self.h - 1 - y0)) * self.w + x0) * 3;
            let n = w.min(self.w - x0) * 3;
            px[y * w * 3..y * w * 3 + n].copy_from_slice(&self.px[s..s + n]);
        }
        Img { w, h, px }
    }

    fn nearest(&self, f: usize) -> Img {
        let (w, h) = (self.w * f, self.h * f);
        let mut px = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let s = ((y / f) * self.w + x / f) * 3;
                px[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&self.px[s..s + 3]);
            }
        }
        Img { w, h, px }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut models: Option<PathBuf> = None;
    let mut images: Vec<PathBuf> = Vec::new();
    let mut out = PathBuf::from("./sheets");
    let mut device = Device::Cpu;
    let mut crop: Option<(usize, usize)> = None;
    let mut cols = 4usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--models" => models = Some(PathBuf::from(req(&args, &mut i)?)),
            "--images" => images.push(PathBuf::from(req(&args, &mut i)?)),
            "--out" => out = PathBuf::from(req(&args, &mut i)?),
            "--device" => device = parse_upscale_device(&req(&args, &mut i)?)?,
            "--crop" => {
                let v = req(&args, &mut i)?;
                let (w, h) = v
                    .split_once(['x', 'X'])
                    .context("--crop wants WxH, e.g. 160x120")?;
                crop = Some((
                    w.trim().parse().context("--crop width")?,
                    h.trim().parse().context("--crop height")?,
                ));
            }
            "--cols" => {
                cols = req(&args, &mut i)?
                    .parse::<usize>()
                    .context("--cols")?
                    .max(1)
            }
            other => bail!("unknown argument {other:?}"),
        }
    }
    let models = models.context("--models is required")?;
    anyhow::ensure!(!images.is_empty(), "--images is required");
    std::fs::create_dir_all(&out)?;

    let checkpoints = list(&models, &["pth", "pt", "bin", "ckpt", "safetensors"])?;
    anyhow::ensure!(
        !checkpoints.is_empty(),
        "no checkpoints in {}",
        models.display()
    );
    let inputs = expand(&images)?;

    for image in &inputs {
        let image_name = stem(image);
        let full = load(image)?;
        // Crop *before* running: every model then processes only what the sheet
        // shows, instead of a whole photo of which 95% is discarded.
        let src = match crop {
            Some((cw, ch)) => {
                let (cw, ch) = (cw.min(full.w), ch.min(full.h));
                full.crop((full.w - cw) / 2, (full.h - ch) / 2, cw, ch)
            }
            None => full,
        };
        println!("\n{image_name}  running models on {}×{}", src.w, src.h);

        let mut panels: Vec<(String, Img)> =
            vec![("nearest x4 baseline".to_string(), src.nearest(4))];

        for ck in &checkpoints {
            let label = stem(ck);
            let mut up = match Upscaler::open(ck, device, UpscaleOptions::default()) {
                Ok(u) => u,
                Err(e) => {
                    println!("  {label:<40} skipped: {}", first_line(&e));
                    continue;
                }
            };
            let scale = up.config().scale;
            let arch = up.config().arch.name().to_string();
            let t0 = Instant::now();
            let (px, ow, oh) = match up.upscale_rgb8(&src.px, src.w, src.h) {
                Ok(v) => v,
                Err(e) => {
                    println!("  {label:<40} failed: {}", first_line(&e));
                    continue;
                }
            };
            println!(
                "  {label:<40} {arch:<10} x{scale}  {:.2}s",
                t0.elapsed().as_secs_f64()
            );
            panels.push((
                format!("{arch} x{scale} {}", short(&label, 22)),
                Img { w: ow, h: oh, px },
            ));
        }

        // Group by scale so like sizes sit together. Cells are sized to the
        // largest panel, so interleaving a ×1 among ×4s leaves ragged holes;
        // sorted, the ×1 and ×2 rows read as "fewer pixels", which is the
        // honest thing a smaller panel should say.
        panels[1..].sort_by_key(|(_, p)| std::cmp::Reverse(p.w));

        // A ×1 restoration model produces the smallest panels and a ×4 the
        // largest; the sheet is laid out on the largest so nothing is scaled.
        let pw = panels.iter().map(|(_, p)| p.w).max().unwrap_or(1);
        let ph = panels.iter().map(|(_, p)| p.h).max().unwrap_or(1);
        let dest = out.join(format!("{image_name}_sheet.png"));
        let (sw, sh) = sheet(&panels, pw, ph, cols, &dest)?;
        println!(
            "  → {} ({sw}×{sh}, {} panels)",
            dest.display(),
            panels.len()
        );
    }
    Ok(())
}

/// Lay panels out on a grid, each in a cell of `pw × ph` with a caption strip.
fn sheet(
    panels: &[(String, Img)],
    pw: usize,
    ph: usize,
    cols: usize,
    dest: &Path,
) -> Result<(usize, usize)> {
    const GAP: usize = 6;
    const BAR: usize = 14;
    let rows = panels.len().div_ceil(cols);
    let (cw, ch) = (pw, ph + BAR);
    let w = cols * cw + (cols + 1) * GAP;
    let h = rows * ch + (rows + 1) * GAP;
    // Mid grey: lighter than any caption text, darker than a blown highlight,
    // so neither the labels nor a white sky bleeds into the background.
    let mut canvas = vec![90u8; w * h * 3];

    for (i, (label, img)) in panels.iter().enumerate() {
        let cx = GAP + (i % cols) * (cw + GAP);
        let cy = GAP + (i / cols) * (ch + GAP);
        font::draw(&mut canvas, w, cx + 2, cy + 3, label, [235, 235, 235]);
        // Panels are their natural size; a smaller one sits top-left in its
        // cell rather than being stretched to fill it.
        for y in 0..img.h.min(ph) {
            let d = ((cy + BAR + y) * w + cx) * 3;
            let n = img.w.min(pw) * 3;
            canvas[d..d + n].copy_from_slice(&img.px[y * img.w * 3..y * img.w * 3 + n]);
        }
    }
    image::RgbImage::from_raw(w as u32, h as u32, canvas)
        .context("sheet buffer does not match its declared size")?
        .save(dest)
        .with_context(|| format!("writing {}", dest.display()))?;
    Ok((w, h))
}

fn load(p: &Path) -> Result<Img> {
    let img = image::open(p)
        .with_context(|| format!("opening {}", p.display()))?
        .to_rgb8();
    Ok(Img {
        w: img.width() as usize,
        h: img.height() as usize,
        px: img.into_raw(),
    })
}

fn list(dir: &Path, exts: &[&str]) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| exts.contains(&e))
        })
        .collect();
    v.sort();
    Ok(v)
}

fn expand(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut v = Vec::new();
    for p in paths {
        if p.is_dir() {
            v.extend(list(p, &["png", "jpg", "jpeg", "webp", "bmp"])?);
        } else {
            v.push(p.clone());
        }
    }
    Ok(v)
}

fn stem(p: &Path) -> String {
    p.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

fn short(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    s.chars().skip(s.chars().count() - n).collect()
}

fn first_line(e: &anyhow::Error) -> String {
    format!("{e:#}")
        .lines()
        .next()
        .unwrap_or("failed")
        .to_string()
}
