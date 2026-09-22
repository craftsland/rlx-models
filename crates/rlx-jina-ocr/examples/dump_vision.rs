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

//! Dump the projected vision pack for one image, for comparison against the
//! reference implementation's `compute_inputs_embeds` rows.
//!
//! ```bash
//! cargo run -p rlx-jina-ocr --release --example dump_vision -- \
//!     --image crates/rlx-jina-ocr/fixtures/sample.png --out /tmp/rlx_vision.f32
//! ```
//!
//! Writes raw little-endian f32, `[n_tokens, hidden]` row-major — load with
//! `np.fromfile(path, dtype="<f4").reshape(-1, 1280)`.

use anyhow::{Context, Result};
use rlx_jina_ocr::config::JinaOcrConfig;
use rlx_jina_ocr::hub::default_model_dir;
use rlx_jina_ocr::preprocess::{ImageMode, image_token_count_for, preprocess_path};
use rlx_unlimited_ocr::deep_encoder::DeepEncoder;
use rlx_unlimited_ocr::projector::Projector;
use rlx_unlimited_ocr::weights::UnlimitedOcrWeightStore;
use std::io::Write;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut image = rlx_jina_ocr::fixtures::sample_image_path();
    let mut out = PathBuf::from("/tmp/rlx_vision.f32");
    let mut mode: Option<ImageMode> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--image" => {
                image = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--out" => {
                out = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--mode" => {
                mode = ImageMode::parse(&args[i + 1]);
                i += 2;
            }
            other => anyhow::bail!("unknown flag {other}"),
        }
    }

    let dir = default_model_dir()?;
    let cfg = JinaOcrConfig::from_model_dir(&dir)?;
    cfg.validate()?;
    let mode = mode.unwrap_or_else(|| ImageMode::from_processor(&cfg.processor));

    let pre = preprocess_path(&image, mode)?;
    let n_expected = image_token_count_for(&cfg, &pre);
    println!(
        "image {:?} {}x{} mode {mode:?} grid {:?} tiles {} -> {n_expected} placeholder tokens",
        image,
        pre.orig_w,
        pre.orig_h,
        pre.spatial_crop,
        pre.tiles.len()
    );

    let store = UnlimitedOcrWeightStore::open(&dir)?;
    let mut encoder = DeepEncoder::from_config(&cfg.lm);
    let mut projector = Projector::from_config(&cfg.lm.projector);
    encoder.load(&store).context("load deep encoder")?;
    projector.load(&store).context("load projector")?;

    let pack = encoder.encode_and_project(std::slice::from_ref(&pre), &projector)?;
    let hidden = cfg.lm.hidden_size;
    let rows = pack.len() / hidden;
    println!("vision pack: {rows} x {hidden} (expected {n_expected})");

    let mean = pack.iter().sum::<f32>() / pack.len() as f32;
    let var = pack.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / pack.len() as f32;
    let (lo, hi) = pack
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
    println!(
        "mean {mean:.5} std {:.5} min {lo:.4} max {hi:.4}",
        var.sqrt()
    );
    let norm = |r: usize| -> f32 {
        pack[r * hidden..(r + 1) * hidden]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
    };
    println!(
        "row-norms: first {:.4} row630 {:.4} last {:.4}",
        norm(0),
        norm(630.min(rows - 1)),
        norm(rows - 1)
    );
    println!(
        "first row[:6] {:?}",
        pack[..6]
            .iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );

    let mut f = std::fs::File::create(&out)?;
    for v in &pack {
        f.write_all(&v.to_le_bytes())?;
    }
    println!("wrote {:?} ({} bytes)", out, pack.len() * 4);
    Ok(())
}
