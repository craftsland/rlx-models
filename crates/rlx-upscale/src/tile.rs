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

//! Tiled inference: how an image of any size runs through one fixed-shape graph.
//!
//! rlx graphs are shape-static, so a tile size is a compile-time constant. That
//! is a feature here rather than a constraint — it is exactly what bounds peak
//! memory. Every tile is the **same** `tile × tile` shape, so the network is
//! compiled once and peak working set is set by the tile, not the image. A 6000
//! × 4000 photo and a 256 × 256 thumbnail cost the same per step.
//!
//! # Seams
//!
//! Each output block is produced from a source region grown by `overlap` on
//! every side, and only the block's own pixels are kept. As long as `overlap`
//! covers the network's receptive field, the kept pixels are bit-identical to
//! what a whole-image pass would have produced, so no blending is needed and
//! there is no seam to hide. Blending would in fact be *worse*: it would smear
//! two equally-correct predictions together.
//!
//! Regions that fall outside the image are filled by edge replication, matching
//! how these networks are trained and evaluated on padded inputs.

use anyhow::{Result, ensure};

/// An axis-aligned region in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// One unit of work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    /// Where to read from in the source image. Always `tile × tile`; parts may
    /// lie outside the image and are edge-replicated.
    pub src: Rect,
    /// The sub-rect of the tile's *output* that is kept, in output pixels,
    /// relative to the tile's own top-left.
    pub crop: Rect,
    /// Where `crop` lands in the destination image, in output pixels.
    pub dst: Rect,
}

/// A full tiling of one image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePlan {
    pub tile: usize,
    pub overlap: usize,
    pub scale: usize,
    pub src_w: usize,
    pub src_h: usize,
    pub tiles: Vec<Tile>,
}

impl TilePlan {
    /// Plan a tiling.
    ///
    /// `tile` is the compiled input extent; `overlap` is the halo discarded on
    /// each side. `multiple` forces both the tile and the step onto a grid —
    /// window attention needs the tile to be a whole number of windows.
    pub fn new(
        src_w: usize,
        src_h: usize,
        tile: usize,
        overlap: usize,
        scale: usize,
        multiple: usize,
    ) -> Result<Self> {
        ensure!(src_w > 0 && src_h > 0, "image is empty");
        ensure!(scale >= 1, "scale must be at least 1");
        let multiple = multiple.max(1);
        ensure!(
            tile.is_multiple_of(multiple),
            "tile {tile} is not a multiple of {multiple}"
        );
        ensure!(
            tile > 2 * overlap,
            "tile {tile} leaves no interior after a {overlap}px halo on each side"
        );

        // The step must also land on the grid, or successive tiles would
        // partition their windows differently and the halo would stop matching.
        let raw_step = tile - 2 * overlap;
        let step = (raw_step / multiple) * multiple;
        ensure!(
            step > 0,
            "tile {tile} with a {overlap}px halo leaves a step smaller than {multiple}"
        );

        let mut tiles = Vec::new();
        let mut y = 0usize;
        while y < src_h {
            let bh = step.min(src_h - y);
            let mut x = 0usize;
            while x < src_w {
                let bw = step.min(src_w - x);
                // The source window is the block grown by the halo; it may run
                // off any edge, which the extractor replicates.
                let sx = x as isize - overlap as isize;
                let sy = y as isize - overlap as isize;
                tiles.push(Tile {
                    src: Rect {
                        // Stored unsigned after clamping is *not* possible here:
                        // the offset must stay negative so the crop below lines
                        // up. Encode it as the clamped origin plus the shift.
                        x: sx.max(0) as usize,
                        y: sy.max(0) as usize,
                        w: tile,
                        h: tile,
                    },
                    crop: Rect {
                        // Distance from the tile's top-left to the block, in
                        // output pixels. At the image edge the source window was
                        // clamped, so the block starts at 0 instead of `overlap`.
                        x: (x - sx.max(0) as usize) * scale,
                        y: (y - sy.max(0) as usize) * scale,
                        w: bw * scale,
                        h: bh * scale,
                    },
                    dst: Rect {
                        x: x * scale,
                        y: y * scale,
                        w: bw * scale,
                        h: bh * scale,
                    },
                });
                x += step;
            }
            y += step;
        }

        Ok(Self {
            tile,
            overlap,
            scale,
            src_w,
            src_h,
            tiles,
        })
    }

    pub fn out_w(&self) -> usize {
        self.src_w * self.scale
    }

    pub fn out_h(&self) -> usize {
        self.src_h * self.scale
    }

    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Work done per unit of output, as a multiple of 1.
    ///
    /// Only the `(tile − 2·overlap)²` interior of each tile survives the crop,
    /// so the halo is paid for and thrown away. This is the hidden cost of a
    /// small tile: at `tile = 80, overlap = 16` it is 2.8×, at
    /// `tile = 176, overlap = 16` only 1.5×.
    pub fn redundancy(&self) -> f32 {
        let step = self.tile.saturating_sub(2 * self.overlap).max(1);
        (self.tile as f32 / step as f32).powi(2)
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }
}

/// Copy a `tile × tile` window out of a planar CHW image, replicating the edge
/// wherever the window falls outside.
pub fn extract(src: &[f32], channels: usize, src_w: usize, src_h: usize, region: Rect) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * region.h * region.w];
    for c in 0..channels {
        let plane = &src[c * src_h * src_w..(c + 1) * src_h * src_w];
        let dst = &mut out[c * region.h * region.w..(c + 1) * region.h * region.w];
        for y in 0..region.h {
            let sy = (region.y + y).min(src_h - 1);
            for x in 0..region.w {
                let sx = (region.x + x).min(src_w - 1);
                dst[y * region.w + x] = plane[sy * src_w + sx];
            }
        }
    }
    out
}

/// Blit `crop` out of a tile's planar CHW output into `dst` at `at`.
#[allow(clippy::too_many_arguments)]
pub fn blit(
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
    tile_out: &[f32],
    tile_w: usize,
    tile_h: usize,
    channels: usize,
    crop: Rect,
    at: Rect,
) {
    for c in 0..channels {
        let src_plane = &tile_out[c * tile_h * tile_w..(c + 1) * tile_h * tile_w];
        let dst_plane = &mut dst[c * dst_h * dst_w..(c + 1) * dst_h * dst_w];
        for y in 0..crop.h {
            let dy = at.y + y;
            if dy >= dst_h {
                break;
            }
            let sy = crop.y + y;
            if sy >= tile_h {
                break;
            }
            let n = crop.w.min(dst_w.saturating_sub(at.x)).min(tile_w - crop.x);
            let s = sy * tile_w + crop.x;
            let d = dy * dst_w + at.x;
            dst_plane[d..d + n].copy_from_slice(&src_plane[s..s + n]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every output pixel must be written exactly once: a gap is a black band,
    /// a double-write means two tiles disagreed about who owns a pixel.
    fn coverage(plan: &TilePlan) -> Vec<u32> {
        let mut hits = vec![0u32; plan.out_w() * plan.out_h()];
        for t in &plan.tiles {
            for y in 0..t.dst.h {
                for x in 0..t.dst.w {
                    hits[(t.dst.y + y) * plan.out_w() + t.dst.x + x] += 1;
                }
            }
        }
        hits
    }

    #[test]
    fn tiles_cover_every_output_pixel_exactly_once() {
        for (w, h, tile, overlap, scale) in [
            (100, 80, 64, 8, 4),
            (256, 256, 128, 16, 2),
            (37, 91, 32, 4, 3),
            (1, 1, 16, 2, 4),
            (513, 7, 64, 8, 1),
        ] {
            let plan = TilePlan::new(w, h, tile, overlap, scale, 1).unwrap();
            let hits = coverage(&plan);
            assert!(
                hits.iter().all(|&n| n == 1),
                "{w}×{h} tile {tile}/{overlap} ×{scale}: {} gaps, {} overlaps",
                hits.iter().filter(|&&n| n == 0).count(),
                hits.iter().filter(|&&n| n > 1).count(),
            );
        }
    }

    /// Whatever the plan says to crop must actually exist inside the tile's
    /// output — an off-by-one at the image edge reads past the tile.
    #[test]
    fn crops_stay_inside_the_tile_output() {
        let plan = TilePlan::new(100, 80, 64, 8, 4, 1).unwrap();
        let out = plan.tile * plan.scale;
        for t in &plan.tiles {
            assert!(
                t.crop.x + t.crop.w <= out && t.crop.y + t.crop.h <= out,
                "crop {:?} escapes a {out}×{out} tile output",
                t.crop
            );
        }
    }

    /// Window attention partitions by absolute position, so both the tile and
    /// the step have to land on the window grid.
    #[test]
    fn step_is_forced_onto_the_window_grid() {
        let plan = TilePlan::new(200, 200, 192, 16, 1, 16).unwrap();
        // 192 − 32 = 160, already a multiple of 16.
        assert_eq!(plan.tiles[1].dst.x, 160);
        let hits = coverage(&plan);
        assert!(hits.iter().all(|&n| n == 1));

        // A halo that is not itself a multiple still yields a gridded step.
        let plan = TilePlan::new(200, 200, 192, 20, 1, 16).unwrap();
        assert_eq!(plan.tiles[1].dst.x % 16, 0);
        let hits = coverage(&plan);
        assert!(hits.iter().all(|&n| n == 1));
    }

    /// The halo is paid for and cropped away, so a small tile multiplies work.
    /// This is the cost that trades against the memory the small tile saves.
    #[test]
    fn redundancy_grows_as_the_tile_shrinks() {
        let big = TilePlan::new(512, 384, 176, 16, 4, 16).unwrap();
        let small = TilePlan::new(512, 384, 80, 16, 4, 16).unwrap();
        assert!(
            (big.redundancy() - 1.49).abs() < 0.02,
            "{}",
            big.redundancy()
        );
        assert!(
            (small.redundancy() - 2.78).abs() < 0.02,
            "{}",
            small.redundancy()
        );
        assert!(small.len() > big.len());
        // No halo at all is exactly one unit of work per unit of output.
        assert_eq!(
            TilePlan::new(64, 64, 32, 0, 1, 1).unwrap().redundancy(),
            1.0
        );
    }

    #[test]
    fn a_tile_smaller_than_its_halo_is_rejected() {
        assert!(TilePlan::new(64, 64, 16, 8, 4, 1).is_err());
        assert!(TilePlan::new(64, 64, 20, 4, 4, 16).is_err());
    }

    /// The halo is replicated, not zero-filled: a zero border would make the
    /// network hallucinate a dark edge that then gets cropped into the seam.
    #[test]
    fn extract_replicates_the_edge() {
        let src: Vec<f32> = (0..(3 * 3)).map(|i| i as f32).collect();
        let got = extract(
            &src,
            1,
            3,
            3,
            Rect {
                x: 1,
                y: 1,
                w: 3,
                h: 3,
            },
        );
        // Rows/cols past the edge repeat the last real row/col.
        assert_eq!(got, vec![4.0, 5.0, 5.0, 7.0, 8.0, 8.0, 7.0, 8.0, 8.0]);
    }

    #[test]
    fn blit_writes_the_cropped_window_into_place() {
        let tile_out: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 16];
        blit(
            &mut dst,
            4,
            4,
            &tile_out,
            4,
            4,
            1,
            Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            },
            Rect {
                x: 2,
                y: 2,
                w: 2,
                h: 2,
            },
        );
        assert_eq!(dst[2 * 4 + 2], 5.0);
        assert_eq!(dst[2 * 4 + 3], 6.0);
        assert_eq!(dst[3 * 4 + 2], 9.0);
        assert_eq!(dst[3 * 4 + 3], 10.0);
        assert_eq!(dst[0], 0.0);
    }
}
