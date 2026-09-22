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

//! Image preprocessing — `DeepseekOCRProcessor.preprocess_image`.
//!
//! Same two-branch pipeline as Unlimited-OCR (that crate's primitives do the
//! pixel work), with jina's own parameters:
//!
//! | | Unlimited-OCR | jina-ocr-v1 |
//! |---|---|---|
//! | `dynamic_preprocess(max_num=…)` | 32 | **9** |
//! | tiling threshold | const 640 | `image_size` |
//!
//! Everything else matches: `ImageOps.pad` letterbox onto a 127-grey canvas,
//! `mean = std = 0.5` normalization to `[-1, 1]`, PIL-bicubic (Catmull-Rom)
//! resampling, and the `q x (q+1) + 1` / `(q·H) x (q·W + 1)` placeholder runs.

use crate::config::{
    BASE_SIZE, DYNAMIC_MAX_NUM, DYNAMIC_MIN_NUM, JinaOcrConfig, ProcessorConfig, TILE_SIZE,
};
use anyhow::{Context, Result};
use image::DynamicImage;
use image::imageops::FilterType;
use rlx_unlimited_ocr::config::{DOWNSAMPLE_RATIO, PATCH_SIZE};
use rlx_unlimited_ocr::preprocess::{
    PAD_COLOR, PreprocessedImage, dynamic_preprocess, load_image_exif_corrected, pad_to_square,
    rgb_to_chw_normalized,
};
use std::path::Path;

/// PIL's default `Image.BICUBIC`, which is Catmull-Rom.
const RESAMPLE: FilterType = FilterType::CatmullRom;

/// Which resolution strategy `preprocess_image` takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageMode {
    /// `crop_mode = True` — a `base`-pixel global view plus up to
    /// [`DYNAMIC_MAX_NUM`] `tile`-pixel local tiles.
    Gundam { base: u32, tile: u32 },
    /// `crop_mode = False` — one `size`-pixel view, no tiling.
    Native { size: u32 },
}

impl Default for ImageMode {
    fn default() -> Self {
        ImageMode::Gundam {
            base: BASE_SIZE,
            tile: TILE_SIZE,
        }
    }
}

impl ImageMode {
    /// The mode `processor_config.json` selects.
    pub fn from_processor(p: &ProcessorConfig) -> Self {
        if p.crop_mode {
            ImageMode::Gundam {
                base: p.base_size,
                tile: p.image_size,
            }
        } else {
            ImageMode::Native { size: p.image_size }
        }
    }

    /// `gundam` / `native` (aliases: `crop`, `base`).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "gundam" | "crop" => Some(Self::default()),
            "native" | "base" => Some(ImageMode::Native { size: BASE_SIZE }),
            _ => None,
        }
    }
}

/// Load an image for OCR.
///
/// Deliberate divergence from `example.py`'s `Image.open(...).convert("RGB")`:
/// EXIF orientation is applied. PIL leaves the tag unapplied, which transcribes
/// a rotated phone photo sideways; the sibling Unlimited-OCR processor calls
/// `ImageOps.exif_transpose` for the same reason. Images without an orientation
/// tag — every document render, every fixture — are bit-identical either way.
pub fn load_image(path: &Path) -> Result<DynamicImage> {
    load_image_exif_corrected(path).with_context(|| format!("load image {path:?}"))
}

/// `preprocess_image(image, base_size, image_size, crop_mode)`.
pub fn preprocess_one(image: &DynamicImage, mode: ImageMode) -> PreprocessedImage {
    let (orig_w, orig_h) = (image.width(), image.height());
    match mode {
        ImageMode::Native { size } => {
            // `if image_size <= 640: image = image.resize((size, size))` — an
            // aspect-ignoring stretch — then `pad_img`, a no-op at that point.
            let view = if size <= TILE_SIZE {
                image.resize_exact(size, size, RESAMPLE).to_rgb8()
            } else {
                pad_to_square(image, size, PAD_COLOR)
            };
            PreprocessedImage {
                global: rgb_to_chw_normalized(&view),
                global_size: size,
                tiles: Vec::new(),
                tile_size: 0,
                spatial_crop: [1, 1],
                orig_w,
                orig_h,
            }
        }
        ImageMode::Gundam { base, tile } => {
            // `if not (w <= image_size and h <= image_size): dynamic_preprocess(...)`.
            let (tiles_rgb, width_crop_num, height_crop_num) = if orig_w <= tile && orig_h <= tile {
                (Vec::new(), 1, 1)
            } else {
                let dt = dynamic_preprocess(image, DYNAMIC_MIN_NUM, DYNAMIC_MAX_NUM, tile);
                (dt.tiles, dt.width_crop_num, dt.height_crop_num)
            };
            let global_view = pad_to_square(image, base, PAD_COLOR);
            PreprocessedImage {
                global: rgb_to_chw_normalized(&global_view),
                global_size: base,
                tiles: tiles_rgb.iter().map(rgb_to_chw_normalized).collect(),
                tile_size: tile,
                spatial_crop: [width_crop_num, height_crop_num],
                orig_w,
                orig_h,
            }
        }
    }
}

/// Load + preprocess one page from disk.
pub fn preprocess_path(path: &Path, mode: ImageMode) -> Result<PreprocessedImage> {
    Ok(preprocess_one(&load_image(path)?, mode))
}

/// Placeholder tokens this image expands to, in prompt order (global view
/// first, then the tile grid — matching `tokenized_image`).
pub fn image_token_count(image: &PreprocessedImage) -> usize {
    image.token_count(PATCH_SIZE, DOWNSAMPLE_RATIO)
}

/// [`image_token_count`] resolved against a config's patch/downsample settings.
pub fn image_token_count_for(cfg: &JinaOcrConfig, image: &PreprocessedImage) -> usize {
    image.token_count(cfg.lm.patch_size, cfg.lm.downsample_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn img(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        }))
    }

    #[test]
    fn processor_config_selects_gundam() {
        assert_eq!(
            ImageMode::from_processor(&ProcessorConfig::default()),
            ImageMode::Gundam {
                base: BASE_SIZE,
                tile: TILE_SIZE
            }
        );
        assert_eq!(
            ImageMode::from_processor(&ProcessorConfig {
                crop_mode: false,
                image_size: 640,
                base_size: 1024,
            }),
            ImageMode::Native { size: 640 }
        );
    }

    #[test]
    fn small_image_skips_tiling_in_gundam_mode() {
        let pre = preprocess_one(&img(320, 200), ImageMode::default());
        assert!(pre.tiles.is_empty());
        assert_eq!(pre.spatial_crop, [1, 1]);
        assert_eq!(pre.global_size, BASE_SIZE);
        assert_eq!(pre.global.len(), 3 * 1024 * 1024);
        // Global view only: q = 16 -> 16*17 + 1.
        assert_eq!(image_token_count(&pre), 16 * 17 + 1);
    }

    /// The threshold is the *tile* size, not a constant — jina compares the
    /// source against `image_size`.
    #[test]
    fn tiling_threshold_follows_tile_size() {
        let mode = ImageMode::Gundam {
            base: 1024,
            tile: 256,
        };
        // 320x200 is under the 640 default but over a 256 tile.
        let pre = preprocess_one(&img(320, 200), mode);
        assert!(!pre.tiles.is_empty());
        assert_eq!(pre.tile_size, 256);
    }

    #[test]
    fn large_image_tiles_and_respects_max_num_9() {
        let pre = preprocess_one(&img(4000, 800), ImageMode::default());
        let [w, h] = pre.spatial_crop;
        assert!(w * h >= DYNAMIC_MIN_NUM, "at least min_num tiles");
        assert!(
            w * h <= DYNAMIC_MAX_NUM,
            "jina caps the grid at 9 tiles, got {w}x{h}"
        );
        assert_eq!(pre.tiles.len(), (w * h) as usize);
        for t in &pre.tiles {
            assert_eq!(t.len(), 3 * 640 * 640);
        }
        // Wide page -> wide grid.
        assert!(w > h, "expected a wide grid for a 5:1 page, got {w}x{h}");
    }

    #[test]
    fn token_count_is_global_plus_tile_grid() {
        let pre = preprocess_one(&img(2000, 1400), ImageMode::default());
        let [w, h] = pre.spatial_crop;
        let expected = (16 * 17 + 1) + (10 * h as usize) * (10 * w as usize + 1);
        assert_eq!(image_token_count(&pre), expected);
    }

    #[test]
    fn native_mode_small_size_stretches_without_padding() {
        let pre = preprocess_one(&img(1000, 200), ImageMode::Native { size: 640 });
        assert!(pre.tiles.is_empty());
        assert_eq!(pre.global_size, 640);
        assert_eq!(pre.global.len(), 3 * 640 * 640);
        assert_eq!(image_token_count(&pre), 10 * 11 + 1);
    }

    #[test]
    fn native_mode_large_size_letterboxes() {
        let pre = preprocess_one(&img(1000, 200), ImageMode::Native { size: 1024 });
        assert_eq!(pre.global.len(), 3 * 1024 * 1024);
        // Letterbox border is the 127-grey pad colour -> ~0 after normalization.
        let top_left = pre.global[0];
        assert!(top_left.abs() < 0.01, "expected grey pad, got {top_left}");
    }

    #[test]
    fn normalization_maps_to_unit_range() {
        let pre = preprocess_one(&img(64, 64), ImageMode::Native { size: 64 });
        assert!(pre.global.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn original_dimensions_are_retained_for_box_scaling() {
        let pre = preprocess_one(&img(1234, 567), ImageMode::default());
        assert_eq!((pre.orig_w, pre.orig_h), (1234, 567));
    }

    #[test]
    fn mode_parse_accepts_documented_names() {
        assert_eq!(ImageMode::parse("gundam"), Some(ImageMode::default()));
        assert_eq!(
            ImageMode::parse("native"),
            Some(ImageMode::Native { size: BASE_SIZE })
        );
        assert_eq!(ImageMode::parse("nope"), None);
    }
}
