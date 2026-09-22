// RLX — versatile ML compiler + runtime. GPLv3.
//! **DeepSeek-V4.1 vision tower** — the DeepSeek-ViT encoder plus the aligner
//! that maps its patch grid into the language model's embedding space.
//!
//! The ViT is a plain pre-norm transformer over one image's patches with **full
//! bidirectional attention** and **2-D RoPE**: the first half of each head's
//! rotary pairs is driven by the patch's row, the second half by its column. The
//! aligner then folds a `downsample_ratio × downsample_ratio` neighbourhood of
//! patches into one language token (a strided unfold, zero-padded to a whole
//! number of blocks) and runs it through a two-layer GELU MLP.
//!
//! The result replaces the `IMAGE` slots of the prompt in row-major order; the
//! three span delimiters (`image_start` / `image_end` / `image_newline`) are
//! learned embeddings the caller splices in.
//!
//! Reference: `deepseek-ai/DeepSeek-V4.1-Flash` `inference/vision.py`.

use crate::dsv41::{DeepseekV41Spec, VisionSpec};
use crate::dsv41_block::Ctx;
use crate::standard_decoder::synth_const;
use crate::weight_loader::WeightLoader;
use anyhow::{Result, anyhow};
use rlx_ir::GraphExt;
use rlx_ir::graph::{Graph, NodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// The ViT's RMSNorm epsilon. It is the `vision.py` default and is **not** the
/// language model's `norm_eps` (1e-20 in the GA checkpoint) — using that one
/// here would divide by an unregularized norm.
pub const VISION_NORM_EPS: f32 = 1e-6;

/// 2-D rotary tables for an `n_h × n_w` patch grid.
///
/// `rope_dim = vision_dim / n_heads / 2`, and each position's frequency vector is
/// `[row · inv_freq, col · inv_freq]` — so the concatenated table is `head_dim/2`
/// wide, exactly the half-split NeoX rotation needs.
fn vision_rope_tables(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    n_h: usize,
    n_w: usize,
    rope_dim: usize,
    theta: f64,
) -> (NodeId, NodeId) {
    let half = rope_dim / 2;
    let n = n_h * n_w;
    let width = 2 * half;
    let mut cos = vec![0f32; n * width];
    let mut sin = vec![0f32; n * width];
    for r in 0..n_h {
        for c in 0..n_w {
            let p = r * n_w + c;
            for i in 0..half {
                let inv = 1.0 / theta.powf(2.0 * i as f64 / rope_dim as f64);
                let (sh, ch) = (r as f64 * inv).sin_cos();
                let (sw, cw) = (c as f64 * inv).sin_cos();
                cos[p * width + i] = ch as f32;
                sin[p * width + i] = sh as f32;
                cos[p * width + half + i] = cw as f32;
                sin[p * width + half + i] = sw as f32;
            }
        }
    }
    (
        synth_const(g, params, "v41.vit.rope.cos", cos, &[n, width]),
        synth_const(g, params, "v41.vit.rope.sin", sin, &[n, width]),
    )
}

/// `cat([x1·cos - x2·sin, x2·cos + x1·sin])` over the two halves of each head —
/// the reference `apply_rotary`. `x` is `[n, heads·head_dim]`.
fn vision_rope(
    g: &mut Graph,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    n: usize,
    heads: usize,
    head_dim: usize,
) -> NodeId {
    let half = head_dim / 2;
    let x3 = g.reshape_(x, vec![n as i64, heads as i64, head_dim as i64]);
    let x1 = g.narrow_(x3, 2, 0, half);
    let x2 = g.narrow_(x3, 2, half, half);
    // the table is per-position, shared across heads
    let cos3 = g.reshape_(cos, vec![n as i64, 1, half as i64]);
    let sin3 = g.reshape_(sin, vec![n as i64, 1, half as i64]);
    let a = g.mul(x1, cos3);
    let b = g.mul(x2, sin3);
    let lo = g.sub(a, b);
    let c = g.mul(x2, cos3);
    let d = g.mul(x1, sin3);
    let hi = g.add(c, d);
    let cat = g.concat_(vec![lo, hi], 2);
    g.reshape_(cat, vec![n as i64, (heads * head_dim) as i64])
}

/// A `[out, in]` weight + optional `[out]` bias applied to `[n, in]`.
fn linear(
    ctx: &mut Ctx<'_>,
    key: &str,
    x: NodeId,
    n: usize,
    out: usize,
    bias: bool,
) -> Result<NodeId> {
    let w = ctx.param(&format!("{key}.weight"), true)?;
    let mut y = ctx.g.mm(x, w);
    if bias {
        let b = ctx.param(&format!("{key}.bias"), false)?;
        let b2 = ctx.g.reshape_(b, vec![1, out as i64]);
        y = ctx.g.add(y, b2);
    }
    Ok(ctx.g.reshape_(y, vec![n as i64, out as i64]))
}

/// Build the vision tower as its own graph: one image's patches in, language-model
/// embeddings out.
///
/// The single input is `patches [n_h·n_w, 3·patch·patch]`, flattened per patch in
/// row-major grid order — the reference's `PatchEmbed` does `proj(x.flatten(1))`
/// on an `[N, 3, p, p]` stack, which is the same memory. The output is
/// `[n_tokens, dim]` with `n_tokens = ceil(n_h/r) · ceil(n_w/r)`, which the
/// caller splices into the `IMAGE` slots of the prompt.
///
/// It is a separate graph from the text stack because it runs separately: one
/// image at a time, whatever the prompt length.
pub fn build_v41_vision(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    n_h: usize,
    n_w: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let vs = spec
        .vision
        .clone()
        .ok_or_else(|| anyhow!("deepseek_v41: this checkpoint has no vision tower"))?;
    let lm_dim = spec.dim;
    let mut ctx = Ctx::new("deepseek_v41_vision", spec, weights, packed, n_h * n_w);
    let patches = ctx.g.input(
        "patches",
        Shape::new(&[n_h * n_w, 3 * vs.patch_size * vs.patch_size], DType::F32),
    );
    let out = build_tower(&mut ctx, &vs, lm_dim, patches, n_h, n_w)?;
    Ok(ctx.finish(vec![out]))
}

/// The tower proper, on a caller-supplied graph: `patches` in, aligned language
/// embeddings out.
fn build_tower(
    ctx: &mut Ctx<'_>,
    spec: &VisionSpec,
    lm_dim: usize,
    patches: NodeId,
    n_h: usize,
    n_w: usize,
) -> Result<NodeId> {
    if spec.n_heads == 0 || !spec.dim.is_multiple_of(spec.n_heads) {
        return Err(anyhow!(
            "deepseek_v41 vision: dim {} is not divisible by n_heads {}",
            spec.dim,
            spec.n_heads
        ));
    }
    let n = n_h * n_w;
    let vd = spec.dim;
    let head_dim = vd / spec.n_heads;
    let rope_dim = head_dim / 2;
    let eps = VISION_NORM_EPS;
    let zb = ctx.zero_bias("v41.vit.zb", vd);

    let mut x = linear(ctx, "vision.patch_embed.proj", patches, n, vd, true)?;
    let (cos, sin) = vision_rope_tables(
        &mut ctx.g,
        &mut ctx.params,
        n_h,
        n_w,
        rope_dim,
        spec.rope_theta,
    );

    for il in 0..spec.n_layers {
        let bp = format!("vision.blocks.{il}");
        let n1 = ctx.norm(&format!("{bp}.norm1.weight"))?;
        let h = ctx.g.rms_norm(x, n1, zb, eps);
        let qkv = linear(ctx, &format!("{bp}.attn.wqkv"), h, n, 3 * vd, true)?;
        let q = ctx.g.narrow_(qkv, 1, 0, vd);
        let k = ctx.g.narrow_(qkv, 1, vd, vd);
        let v = ctx.g.narrow_(qkv, 1, 2 * vd, vd);
        let q = vision_rope(&mut ctx.g, q, cos, sin, n, spec.n_heads, head_dim);
        let k = vision_rope(&mut ctx.g, k, cos, sin, n, spec.n_heads, head_dim);
        // full bidirectional attention over the single image's patches
        let attn = ctx.g.attention_kind(
            q,
            k,
            v,
            spec.n_heads,
            head_dim,
            MaskKind::None,
            Shape::new(&[n, vd], DType::F32),
        );
        let o = linear(ctx, &format!("{bp}.attn.wo"), attn, n, vd, true)?;
        x = ctx.g.add(x, o);

        let n2 = ctx.norm(&format!("{bp}.norm2.weight"))?;
        let h = ctx.g.rms_norm(x, n2, zb, eps);
        // w1 emits gate and up fused: [2·inter, dim]
        let gu = linear(
            ctx,
            &format!("{bp}.mlp.w1"),
            h,
            n,
            2 * spec.inter_dim,
            false,
        )?;
        let gate = ctx.g.narrow_(gu, 1, 0, spec.inter_dim);
        let up = ctx.g.narrow_(gu, 1, spec.inter_dim, spec.inter_dim);
        let act = ctx.g.silu(gate);
        let glu = ctx.g.mul(act, up);
        let down = linear(ctx, &format!("{bp}.mlp.w2"), glu, n, vd, false)?;
        x = ctx.g.add(x, down);
    }
    let nf = ctx.norm("vision.norm.weight")?;
    let x = ctx.g.rms_norm(x, nf, zb, eps);

    build_v41_aligner(ctx, spec, lm_dim, x, n_h, n_w)
}

/// The aligner: fold each `r × r` patch neighbourhood into one token, then a
/// two-layer GELU MLP into the language model's width.
///
/// The grid is zero-padded on the right and bottom to a whole number of blocks
/// (`F.pad(x, (0, -n_w % r, 0, -n_h % r))`), and the unfold lays each block out
/// **channel-major then row-major within the block**, which is what
/// `F.unfold`'s `[C·r·r, L]` ordering means.
fn build_v41_aligner(
    ctx: &mut Ctx<'_>,
    spec: &VisionSpec,
    lm_dim: usize,
    x: NodeId, // [n_h·n_w, vision_dim]
    n_h: usize,
    n_w: usize,
) -> Result<NodeId> {
    let r = spec.downsample_ratio.max(1);
    let vd = spec.dim;
    let (ph, pw) = (n_h.div_ceil(r) * r, n_w.div_ceil(r) * r);
    let (bh, bw) = (ph / r, pw / r);
    let tokens = bh * bw;
    let in_dim = vd * r * r;

    // Zero-pad the grid, then gather the unfold pattern in one shot: entry
    // (block, c·r·r + dy·r + dx) reads patch (by·r+dy, bx·r+dx), channel c.
    let padded = if ph != n_h || pw != n_w {
        // build a [ph·pw, vd] grid: real rows where inside, zero row otherwise
        let zero = ctx.konst("v41.aligner.zero", vec![0f32; vd], &[1, vd]);
        let mut rows: Vec<NodeId> = Vec::with_capacity(ph * pw);
        for y in 0..ph {
            for xx in 0..pw {
                rows.push(if y < n_h && xx < n_w {
                    ctx.g.narrow_(x, 0, y * n_w + xx, 1)
                } else {
                    zero
                });
            }
        }
        ctx.g.concat_(rows, 0)
    } else {
        x
    };

    // [tokens, r·r, vd] → transpose to channel-major → [tokens, vd·r·r]
    let mut picks: Vec<NodeId> = Vec::with_capacity(tokens * r * r);
    for by in 0..bh {
        for bx in 0..bw {
            for dy in 0..r {
                for dx in 0..r {
                    let src = (by * r + dy) * pw + (bx * r + dx);
                    picks.push(ctx.g.narrow_(padded, 0, src, 1));
                }
            }
        }
    }
    let gathered = ctx.g.concat_(picks, 0); // [tokens·r·r, vd]
    let g3 = ctx
        .g
        .reshape_(gathered, vec![tokens as i64, (r * r) as i64, vd as i64]);
    let g3 = ctx.g.transpose_(g3, vec![0, 2, 1]); // [tokens, vd, r·r] — channel-major
    let flat = ctx.g.reshape_(g3, vec![tokens as i64, in_dim as i64]);

    let h = linear(ctx, "aligner.w1", flat, tokens, lm_dim, true)?;
    let h = ctx.g.gelu(h);
    linear(ctx, "aligner.w2", h, tokens, lm_dim, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_are_row_then_column() {
        let mut g = Graph::new("t");
        let mut p = HashMap::new();
        let (cos, _sin) = vision_rope_tables(&mut g, &mut p, 2, 3, 4, 10000.0);
        let _ = cos;
        let c = &p["v41.vit.rope.cos"];
        // rope_dim 4 → half 2 → width 4; 6 positions
        assert_eq!(c.len(), 6 * 4);
        // position (0,0) is the identity rotation
        assert!(c[..4].iter().all(|v| (*v - 1.0).abs() < 1e-6));
        // position (0,2): row 0 leaves the first half at cos 0 = 1, column 2
        // drives the second half
        let p02 = &c[2 * 4..3 * 4];
        assert!((p02[0] - 1.0).abs() < 1e-6 && (p02[1] - 1.0).abs() < 1e-6);
        assert!((p02[2] - 2f32.cos()).abs() < 1e-5);
        // position (1,0): row 1 drives the first half, column 0 the second
        let p10 = &c[3 * 4..4 * 4];
        assert!((p10[0] - 1f32.cos()).abs() < 1e-5);
        assert!((p10[2] - 1.0).abs() < 1e-6 && (p10[3] - 1.0).abs() < 1e-6);
    }
}

/// What an image position is, within an image's span in the prompt.
///
/// Every one of these carries `image_token_id` in `input_ids` — only the type
/// tells them apart, and only [`ImageTokenType::Image`] slots receive aligner
/// rows. The other three take learned embeddings (`image_start`, `image_end`,
/// `image_newline`), which is why splicing a flat run of aligner output across
/// the span is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageTokenType {
    Start,
    Image,
    NewLine,
    End,
}

/// The span layout for an `n_llm_h × n_llm_w` token grid:
/// `START + (IMAGE × n_llm_w + NEWLINE) × n_llm_h + END`.
pub fn image_token_types(n_llm_h: usize, n_llm_w: usize) -> Vec<ImageTokenType> {
    let mut t = Vec::with_capacity(num_image_tokens(n_llm_h, n_llm_w));
    t.push(ImageTokenType::Start);
    for _ in 0..n_llm_h {
        t.extend(std::iter::repeat_n(ImageTokenType::Image, n_llm_w));
        t.push(ImageTokenType::NewLine);
    }
    t.push(ImageTokenType::End);
    t
}

/// Prompt positions one image occupies — one per aligner token, plus a newline
/// per row, plus the two delimiters.
pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

/// The token grid the aligner produces from a pixel size.
fn llm_grid(best_h: usize, best_w: usize, patch: usize, ratio: usize) -> (usize, usize) {
    (
        (best_h / patch).div_ceil(ratio),
        (best_w / patch).div_ceil(ratio),
    )
}

/// The largest aspect-preserving pixel size whose token grid still fits in
/// `max_n_token`.
///
/// The two degenerate branches matter: an image tall or wide enough that one
/// side would round below a single cell collapses to one column or one row,
/// rather than producing an empty grid.
fn solve_resize_ratio(
    height: f64,
    width: f64,
    patch: usize,
    ratio: usize,
    max_n_token: usize,
) -> (usize, usize) {
    let cell = patch * ratio;
    let r = height / width;
    let max_w = ((max_n_token as f64 - 2.0) / r + 0.25).sqrt() - 0.5;
    let max_h = max_w * r;
    if max_w < 1.0 {
        return ((max_n_token - 2) / 2 * cell, cell);
    }
    if max_h < 1.0 {
        return (cell, (max_n_token - 3) * cell);
    }
    let beta = (max_w.floor() * cell as f64 / width).min(max_h.floor() * cell as f64 / height);
    (
        (height * beta / patch as f64).floor() as usize * patch,
        (width * beta / patch as f64).floor() as usize * patch,
    )
}

/// The resize plan for an image of a given original size: `(n_llm_h, n_llm_w,
/// best_height, best_width)`.
///
/// A pure function of the size and five config values, mirroring
/// `image_processor.plan_image_grid`. The order of the adjustments is
/// load-bearing: the aspect cap narrows the image *before* the `min_pixels`
/// floor scales it back up, so swapping them changes the grid for any wide image
/// below the floor.
pub fn plan_image_grid(
    spec: &VisionSpec,
    width: usize,
    height: usize,
) -> (usize, usize, usize, usize) {
    let p = spec.patch_size.max(1);
    let ratio = spec.downsample_ratio.max(1);
    let mut w = width as f64;
    let mut h = height as f64;
    if let Some(cap) = spec.max_wh_ratio
        && w > h * cap
    {
        w = h * cap;
    }
    let area = w * h;
    if area > 0.0 && area < spec.min_pixels as f64 {
        let r = (spec.min_pixels as f64 / area).sqrt();
        // the reference truncates here; rounding instead shifts the grid
        w = (w * r).trunc();
        h = (h * r).trunc();
    }
    let mut best_w = (w / p as f64).ceil() as usize * p;
    let mut best_h = (h / p as f64).ceil() as usize * p;
    let (mut n_h, mut n_w) = llm_grid(best_h, best_w, p, ratio);
    if num_image_tokens(n_h, n_w) > spec.max_n_token {
        let (bh, bw) = solve_resize_ratio(h, w, p, ratio, spec.max_n_token);
        (best_h, best_w) = (bh, bw);
        (n_h, n_w) = llm_grid(best_h, best_w, p, ratio);
    }
    (n_h, n_w, best_h, best_w)
}
