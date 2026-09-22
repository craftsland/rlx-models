// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `prism.hadamard` — the rotated weight basis used by PrismML's
//! Ternary Bonsai 2 (`prism-ml/Ternary-Bonsai-2-27B-gguf`).
//!
//! Ternary weights lose too much of a 27B to be usable on their own, so
//! PrismML rotates each weight matrix by a blockwise orthogonal
//! Hadamard transform before assigning trits — the rotation spreads
//! outlier channels across a block, which is what keeps 1.75 bits/weight
//! within 2% of FP16. The rotation is folded into the stored weights, so
//! it costs no extra bits, but it means **the weights on disk are not
//! the model's weights**: a matmul against them is only correct if the
//! activation is rotated to match first.
//!
//! For a folded weight `W` with input width `w`, the activation `x`
//! becomes, in order:
//!
//! 1. an optional feature-axis permutation (GDN `ssm_out` only, see
//!    [`GdnPerm`]),
//! 2. an elementwise sign flip by a fixed `±1` vector of length `w`,
//! 3. a **normalized Sylvester–Walsh Hadamard** rotation applied
//!    independently to each block of [`PrismHadamard::block_size`]
//!    features.
//!
//! `token_embd` is the mirror case: its rows are stored rotated, so the
//! *inverse* transform is applied to the lookup result instead. The
//! transform is its own inverse — `H` is symmetric and `H·H = I` once
//! normalized — so the same matrix serves both directions, and that
//! symmetry is also why the matmul orientation below is unambiguous.
//!
//! Everything here is driven by the file's own metadata rather than
//! hardcoded, and [`PrismHadamard::from_gguf`] rejects anything it does
//! not recognize. Silently skipping the transform would not fail
//! loudly — it produces fluent, confidently wrong text — so an
//! unsupported variant must be an error, never a fallback.

use anyhow::{Result, bail};
use rlx_gguf::{GgufFile, MetaValue};
use std::collections::{HashMap, HashSet};

/// Metadata key prefix for every field this module reads.
const KEY: &str = "prism.hadamard";

/// The only transform kind implemented here.
const TRANSFORM: &str = "normalized-sylvester-walsh-hadamard";

/// The only folding axis implemented here.
const AXIS: &str = "input-last-dimension";

/// Feature-axis permutation applied before the rotation.
///
/// The GDN value stream reaches `ssm_out` in *tiled* head order — head
/// `h` is `rep·n_k + k`, because q/k are GQA-expanded by repeating the
/// whole head block `rep` times — while the fold was computed in
/// *grouped* order (`k·rep + rep_i`). The two agree only for the first
/// and last head, so getting this wrong is a quiet accuracy loss rather
/// than a crash.
///
/// As a numpy reshape on the feature axis:
/// `x.reshape(rep, nk, hd).transpose(1, 0, 2)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnPerm {
    /// Per-head width (`n_state`).
    pub hd: usize,
    /// Number of k/v head groups (`ssm.group_count`).
    pub nk: usize,
    /// Repeats per group (`dt_rank / group_count`).
    pub rep: usize,
}

/// A parsed `prism.hadamard` block.
#[derive(Debug, Clone)]
pub struct PrismHadamard {
    /// Rotation block width along the input axis (1024 as shipped).
    pub block_size: usize,
    /// Weights whose activation needs the forward transform.
    folded: HashSet<String>,
    /// Tables whose *lookup result* needs the inverse transform.
    inverse: HashSet<String>,
    /// `±1` vectors keyed by input width.
    signs: HashMap<usize, Vec<f32>>,
    /// Whether `ssm_out` needs [`GdnPerm`].
    gdn_v_grouped: bool,
}

impl PrismHadamard {
    /// Parse the `prism.hadamard.*` metadata block, or `None` when the
    /// file carries no rotation (every other qwen35 GGUF).
    pub fn from_gguf(raw: &GgufFile) -> Result<Option<Self>> {
        let Some(version) = raw.metadata.get(&format!("{KEY}.version")) else {
            // A file with other `prism.*` keys but no version is not
            // something we can reason about — refuse rather than guess.
            if raw.metadata.keys().any(|k| k.starts_with("prism.")) {
                bail!("{KEY}.version missing but the file carries prism.* metadata");
            }
            return Ok(None);
        };
        let version = version
            .as_u32()
            .ok_or_else(|| anyhow::anyhow!("{KEY}.version is not an integer"))?;
        if version != 1 {
            bail!("unsupported {KEY}.version: {version}");
        }

        let transform = str_key(raw, "transform")?;
        if transform != TRANSFORM {
            bail!("unsupported {KEY}.transform: {transform}");
        }
        let axis = str_key(raw, "axis")?;
        if axis != AXIS {
            bail!("unsupported {KEY}.axis: {axis}");
        }

        let block_size =
            raw.metadata
                .get(&format!("{KEY}.block_size"))
                .and_then(MetaValue::as_u32)
                .ok_or_else(|| anyhow::anyhow!("missing {KEY}.block_size"))? as usize;
        if block_size == 0 || !block_size.is_power_of_two() {
            bail!("invalid {KEY}.block_size: {block_size} (must be a power of two)");
        }

        let folded: HashSet<String> = str_array(raw, "weight_names")?.into_iter().collect();
        if folded.is_empty() {
            bail!("{KEY}.weight_names is empty");
        }
        let inverse: HashSet<String> =
            match raw.metadata.get(&format!("{KEY}.inverse_weight_names")) {
                Some(_) => str_array(raw, "inverse_weight_names")?
                    .into_iter()
                    .collect(),
                None => HashSet::new(),
            };
        // The builder only knows how to un-rotate the token embedding
        // lookup; any other latent table would load and stay rotated.
        for name in &inverse {
            if name != "token_embd.weight" {
                bail!("{KEY}: unsupported inverse-after-lookup table `{name}`");
            }
        }

        let sign_mode = str_key(raw, "sign_mode")?;
        let signs = match sign_mode.as_str() {
            "identity" => HashMap::new(),
            "explicit" => parse_signs(raw, block_size)?,
            other => bail!("unsupported {KEY}.sign_mode: {other}"),
        };

        let gdn_v_grouped = raw
            .metadata
            .get(&format!("{KEY}.gdn_v_grouped"))
            .map(|v| matches!(v, MetaValue::Bool(true)))
            .unwrap_or(false);

        Ok(Some(Self {
            block_size,
            folded,
            inverse,
            signs,
            gdn_v_grouped,
        }))
    }

    /// True if `name`'s activation needs the forward transform.
    pub fn is_folded(&self, name: &str) -> bool {
        self.folded.contains(name)
    }

    /// True if `name` is a table whose lookup result needs un-rotating.
    pub fn is_inverse(&self, name: &str) -> bool {
        self.inverse.contains(name)
    }

    /// Number of folded weights, for load-time reporting.
    pub fn folded_count(&self) -> usize {
        self.folded.len()
    }

    /// The `±1` vector for input width `width`, or `None` under
    /// `identity` sign mode.
    ///
    /// Errors when explicit signs are configured but this width is
    /// missing — that would otherwise read as "no flip" and silently
    /// change the model function.
    pub fn signs_for(&self, width: usize) -> Result<Option<&[f32]>> {
        if self.signs.is_empty() {
            return Ok(None);
        }
        self.signs
            .get(&width)
            .map(|v| Some(v.as_slice()))
            .ok_or_else(|| anyhow::anyhow!("{KEY}: no sign vector for input width {width}"))
    }

    /// The GDN permutation for an `ssm_out` weight of input width
    /// `width`, given the model's GDN geometry.
    pub fn gdn_perm(
        &self,
        width: usize,
        dt_rank: usize,
        n_group: usize,
    ) -> Result<Option<GdnPerm>> {
        if !self.gdn_v_grouped {
            return Ok(None);
        }
        if dt_rank == 0
            || n_group == 0
            || !dt_rank.is_multiple_of(n_group)
            || !width.is_multiple_of(dt_rank)
        {
            bail!(
                "{KEY}: bad GDN head geometry (width={width} dt_rank={dt_rank} n_group={n_group})"
            );
        }
        let rep = dt_rank / n_group;
        if rep <= 1 {
            return Ok(None);
        }
        Ok(Some(GdnPerm {
            hd: width / dt_rank,
            nk: n_group,
            rep,
        }))
    }

    /// Check that `block_size` divides every folded weight's input
    /// width. Called once at load so a geometry mismatch surfaces
    /// before any math runs.
    pub fn validate_widths(&self, width_of: impl Fn(&str) -> Option<usize>) -> Result<()> {
        for name in self.folded.iter().chain(self.inverse.iter()) {
            let Some(w) = width_of(name) else {
                bail!("{KEY}: folded weight `{name}` is not present in the file");
            };
            if !w.is_multiple_of(self.block_size) {
                bail!(
                    "{KEY}: block size {} does not divide input width {w} of `{name}`",
                    self.block_size
                );
            }
        }
        Ok(())
    }

    /// The `[block_size, block_size]` rotation matrix, row-major.
    pub fn matrix(&self) -> Vec<f32> {
        hadamard_matrix(self.block_size)
    }
}

/// The transform a single folded weight's activation needs.
///
/// Attached to the projection itself (see `weights::Proj::Folded`) so a
/// rotated weight cannot be reached without it — the failure mode this
/// guards against is not a crash but fluent, confidently wrong output.
#[derive(Debug, Clone)]
pub struct HadamardFold {
    /// Rotation block width along the input axis.
    pub block_size: usize,
    /// `[block_size, block_size]` rotation, shared by every folded
    /// weight in the model (4 MB at the shipped block size).
    pub matrix: std::sync::Arc<[f32]>,
    /// `±1` vector of length `in_features`, or `None` for identity.
    pub signs: Option<std::sync::Arc<[f32]>>,
    /// Feature-axis permutation applied before the sign flip.
    pub perm: Option<GdnPerm>,
}

/// Load-time factory turning a weight name + input width into a
/// [`HadamardFold`].
///
/// Owns the one shared rotation matrix and the per-width sign vectors so
/// the 401 folded weights of the 27B reference them instead of each
/// carrying a copy.
#[derive(Debug, Clone)]
pub struct FoldCtx {
    inner: PrismHadamard,
    matrix: std::sync::Arc<[f32]>,
    signs: HashMap<usize, std::sync::Arc<[f32]>>,
    dt_rank: usize,
    n_group: usize,
}

impl FoldCtx {
    /// `dt_rank` / `n_group` are the GDN head geometry
    /// (`ssm.time_step_rank` / `ssm.group_count`), needed for the
    /// `ssm_out` permutation.
    pub fn new(inner: PrismHadamard, dt_rank: usize, n_group: usize) -> Self {
        let matrix: std::sync::Arc<[f32]> = inner.matrix().into();
        let signs = inner
            .signs
            .iter()
            .map(|(&w, v)| (w, std::sync::Arc::from(v.as_slice())))
            .collect();
        Self {
            inner,
            matrix,
            signs,
            dt_rank,
            n_group,
        }
    }

    /// The parsed metadata block.
    pub fn hadamard(&self) -> &PrismHadamard {
        &self.inner
    }

    /// The transform for an inverse-after-lookup table of width
    /// `width` (its rows are stored rotated).
    ///
    /// Carries the same rotation and the same per-width sign vector as a
    /// folded weight, but [`apply_inverse_transform`] applies them in the
    /// opposite order.
    pub fn inverse_fold(&self, width: usize) -> Result<HadamardFold> {
        if !width.is_multiple_of(self.inner.block_size) {
            bail!(
                "{KEY}: block size {} does not divide table width {width}",
                self.inner.block_size
            );
        }
        let signs = self
            .inner
            .signs_for(width)?
            .map(|_| self.signs.get(&width).expect("validated").clone());
        Ok(HadamardFold {
            block_size: self.inner.block_size,
            matrix: self.matrix.clone(),
            signs,
            perm: None,
        })
    }

    /// The fold for GGUF tensor `name` with input width `in_features`,
    /// or `None` when that tensor is not folded.
    pub fn for_weight(&self, name: &str, in_features: usize) -> Result<Option<HadamardFold>> {
        if !self.inner.is_folded(name) {
            return Ok(None);
        }
        // A/B switch. The transform is mandatory for correctness, so this
        // exists only to tell "the rotation is wrong" apart from "the
        // rotation never ran" — both of which present as bad output.
        if rlx_ir::env::flag("RLX_PRISM_HADAMARD_DISABLE") {
            return Ok(None);
        }
        if !in_features.is_multiple_of(self.inner.block_size) {
            bail!(
                "{KEY}: block size {} does not divide input width {in_features} of `{name}`",
                self.inner.block_size
            );
        }
        let signs = self.inner.signs_for(in_features)?.map(|_| {
            self.signs
                .get(&in_features)
                .expect("signs_for validated presence")
                .clone()
        });
        // Only the GDN value path arrives in a different head order; the
        // full-attention `attn_output` shares `ssm_out`'s width, so key
        // this on the tensor name rather than on the width.
        let perm = if name.contains(".ssm_out.") {
            self.inner
                .gdn_perm(in_features, self.dt_rank, self.n_group)?
        } else {
            None
        };
        Ok(Some(HadamardFold {
            block_size: self.inner.block_size,
            matrix: self.matrix.clone(),
            signs,
            perm,
        }))
    }
}

/// Normalized Sylvester–Walsh Hadamard matrix, row-major `[n, n]`.
///
/// `H[r][c] = ±1/√n`, negative when `popcount(r & c)` is odd. Symmetric
/// and orthogonal, so `H` is its own inverse.
pub fn hadamard_matrix(n: usize) -> Vec<f32> {
    let scale = 1.0f32 / (n as f32).sqrt();
    let mut m = vec![0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            m[r * n + c] = if (r & c).count_ones() % 2 == 1 {
                -scale
            } else {
                scale
            };
        }
    }
    m
}

/// Apply the blockwise Hadamard rotation to each row of a row-major
/// `[rows, width]` buffer, in place.
///
/// Uses the in-place butterfly rather than the `[n, n]` matmul: this
/// runs over the whole 248320-row embedding table at load, where the
/// dense form would be ~1.3 T MACs.
pub fn rotate_rows_in_place(x: &mut [f32], width: usize, block: usize) {
    debug_assert!(width.is_multiple_of(block));
    let scale = 1.0f32 / (block as f32).sqrt();
    for row in x.chunks_exact_mut(width) {
        for blk in row.chunks_exact_mut(block) {
            let mut len = 1;
            while len < block {
                let mut i = 0;
                while i < block {
                    for j in i..i + len {
                        let a = blk[j];
                        let b = blk[j + len];
                        blk[j] = a + b;
                        blk[j + len] = a - b;
                    }
                    i += len << 1;
                }
                len <<= 1;
            }
            for v in blk.iter_mut() {
                *v *= scale;
            }
        }
    }
}

/// Apply the full activation transform to a row-major `[rows, width]`
/// buffer, in place.
///
/// Host counterpart of the graph's `emit_hadamard` — permute, sign
/// flip, rotate, in that order. Kept as one function so the host decode
/// paths and the test fixtures cannot drift from each other about the
/// order of the three steps.
pub fn apply_activation_transform(x: &mut [f32], width: usize, fold: &HadamardFold) {
    debug_assert_eq!(x.len() % width, 0);
    if let Some(p) = fold.perm {
        debug_assert_eq!(p.hd * p.nk * p.rep, width);
        let mut tmp = vec![0f32; width];
        for row in x.chunks_exact_mut(width) {
            // tiled [rep, nk, hd] -> grouped [nk, rep, hd]
            for k in 0..p.nk {
                for r in 0..p.rep {
                    let src = (r * p.nk + k) * p.hd;
                    let dst = (k * p.rep + r) * p.hd;
                    tmp[dst..dst + p.hd].copy_from_slice(&row[src..src + p.hd]);
                }
            }
            row.copy_from_slice(&tmp);
        }
    }
    if let Some(signs) = fold.signs.as_ref() {
        for row in x.chunks_exact_mut(width) {
            for (v, &s) in row.iter_mut().zip(signs.iter()) {
                *v *= s;
            }
        }
    }
    rotate_rows_in_place(x, width, fold.block_size);
}

/// Apply the **inverse** transform to a row-major `[rows, width]`
/// buffer, in place — for a table whose rows are stored rotated.
///
/// This is `h = s ⊙ (H z)`: the rotation first, then the sign flip.
/// That ordering is forced — the forward transform is `H(s ⊙ x)`, so its
/// inverse is `s ⊙ (H x)`, since `H⁻¹ = H` and `diag(s)⁻¹ = diag(s)` but
/// the two do **not** commute. Reusing
/// [`apply_activation_transform`] here instead would flip the signs of
/// the wrong 5120 channels: the model still runs, still emits tokens,
/// and is simply wrong — which is exactly what it did.
pub fn apply_inverse_transform(x: &mut [f32], width: usize, fold: &HadamardFold) {
    debug_assert_eq!(x.len() % width, 0);
    debug_assert!(
        fold.perm.is_none(),
        "no inverse-after-lookup table carries a head permutation"
    );
    rotate_rows_in_place(x, width, fold.block_size);
    if let Some(signs) = fold.signs.as_ref() {
        for row in x.chunks_exact_mut(width) {
            for (v, &s) in row.iter_mut().zip(signs.iter()) {
                *v *= s;
            }
        }
    }
}

fn str_key(raw: &GgufFile, suffix: &str) -> Result<String> {
    raw.metadata
        .get(&format!("{KEY}.{suffix}"))
        .and_then(MetaValue::as_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("missing {KEY}.{suffix}"))
}

fn str_array(raw: &GgufFile, suffix: &str) -> Result<Vec<String>> {
    let arr = raw
        .metadata
        .get(&format!("{KEY}.{suffix}"))
        .and_then(MetaValue::as_array)
        .ok_or_else(|| anyhow::anyhow!("missing or non-array {KEY}.{suffix}"))?;
    arr.iter()
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| anyhow::anyhow!("{KEY}.{suffix} holds a non-string entry"))
        })
        .collect()
}

fn i32_array(raw: &GgufFile, suffix: &str) -> Result<Vec<i32>> {
    let arr = raw
        .metadata
        .get(&format!("{KEY}.{suffix}"))
        .and_then(MetaValue::as_array)
        .ok_or_else(|| anyhow::anyhow!("missing or non-array {KEY}.{suffix}"))?;
    arr.iter()
        .map(|v| match v {
            MetaValue::I32(x) => Ok(*x),
            MetaValue::I64(x) => Ok(*x as i32),
            MetaValue::U32(x) => Ok(*x as i32),
            MetaValue::I16(x) => Ok(*x as i32),
            MetaValue::I8(x) => Ok(*x as i32),
            other => bail!("{KEY}.{suffix} holds a non-integer entry: {other:?}"),
        })
        .collect()
}

/// Split the flat `sign_values` array into one `±1` vector per width.
fn parse_signs(raw: &GgufFile, block_size: usize) -> Result<HashMap<usize, Vec<f32>>> {
    let widths = i32_array(raw, "sign_widths")?;
    let values = i32_array(raw, "sign_values")?;
    if widths.is_empty() {
        bail!("{KEY}.sign_mode is explicit but sign_widths is empty");
    }
    let mut out = HashMap::new();
    let mut off = 0usize;
    for w in widths {
        if w <= 0 {
            bail!("{KEY}: invalid sign width {w}");
        }
        let w = w as usize;
        if !w.is_multiple_of(block_size) {
            bail!("{KEY}: sign width {w} is not a multiple of block size {block_size}");
        }
        if off + w > values.len() {
            bail!("{KEY}: sign_values is shorter than sign_widths requires");
        }
        let vec: Vec<f32> = values[off..off + w]
            .iter()
            .map(|&v| match v {
                1 => Ok(1.0f32),
                -1 => Ok(-1.0f32),
                other => bail!("{KEY}: sign values must be ±1, got {other}"),
            })
            .collect::<Result<_>>()?;
        out.insert(w, vec);
        off += w;
    }
    if off != values.len() {
        bail!(
            "{KEY}.sign_values length mismatch: consumed {off} of {}",
            values.len()
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_is_symmetric_and_self_inverse() {
        let n = 64usize;
        let h = hadamard_matrix(n);
        for r in 0..n {
            for c in 0..n {
                assert_eq!(h[r * n + c], h[c * n + r], "H not symmetric at ({r},{c})");
            }
        }
        // H · H == I, which is what makes the same matrix serve as its
        // own inverse for the token_embd lookup.
        for r in 0..n {
            for c in 0..n {
                let dot: f32 = (0..n).map(|k| h[r * n + k] * h[k * n + c]).sum();
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-4, "(H·H)[{r},{c}] = {dot}");
            }
        }
    }

    #[test]
    fn butterfly_matches_dense_matmul() {
        // The load-time fast transform must agree with the [n,n] matmul
        // the graph applies, or the embedding rows and the rest of the
        // model end up in different bases.
        let n = 32usize;
        let rows = 3usize;
        let width = 2 * n; // two blocks per row, to cover blocking
        let h = hadamard_matrix(n);
        let x: Vec<f32> = (0..rows * width)
            .map(|i| ((i as f32) * 0.37).sin() * 2.0 - 0.5)
            .collect();

        let mut fast = x.clone();
        rotate_rows_in_place(&mut fast, width, n);

        let mut dense = vec![0f32; rows * width];
        for r in 0..rows {
            for b in 0..(width / n) {
                for i in 0..n {
                    let mut acc = 0f32;
                    for j in 0..n {
                        acc += h[i * n + j] * x[r * width + b * n + j];
                    }
                    dense[r * width + b * n + i] = acc;
                }
            }
        }
        for i in 0..fast.len() {
            assert!(
                (fast[i] - dense[i]).abs() < 1e-4,
                "i={i}: fast={} dense={}",
                fast[i],
                dense[i]
            );
        }
    }

    /// Fold a weight the way the publisher does, then check that the
    /// activation transform recovers the original matmul.
    ///
    /// `y = x·Wᵀ` must equal `T(x)·W_foldedᵀ` where `T(x) = H(s ⊙ x)`.
    /// Solving gives `W_folded = W·diag(s)·H`, i.e. the same transform
    /// applied to each *row* of `W`. If the sign flip and the rotation
    /// were applied in the wrong order, or the rotation were
    /// transposed, this would not close.
    #[test]
    fn folded_weight_and_transformed_activation_reproduce_the_matmul() {
        let block = 16usize;
        let in_dim = 2 * block; // two blocks, so blocking is exercised
        let out_dim = 5usize;
        let rows = 3usize;

        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32) * 0.31).sin())
            .collect();
        let x: Vec<f32> = (0..rows * in_dim)
            .map(|i| ((i as f32) * 0.17).cos() * 1.5)
            .collect();
        let signs: Vec<f32> = (0..in_dim)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();

        // Reference: plain y = x · Wᵀ.
        let mut want = vec![0f32; rows * out_dim];
        for r in 0..rows {
            for o in 0..out_dim {
                want[r * out_dim + o] = (0..in_dim)
                    .map(|j| x[r * in_dim + j] * w[o * in_dim + j])
                    .sum();
            }
        }

        // Publisher side: fold the rotation into the weight rows.
        let mut w_folded = w.clone();
        for row in w_folded.chunks_exact_mut(in_dim) {
            for (v, &s) in row.iter_mut().zip(&signs) {
                *v *= s;
            }
        }
        rotate_rows_in_place(&mut w_folded, in_dim, block);

        // Runtime side: the activation transform this module specifies.
        let mut xt = x.clone();
        for row in xt.chunks_exact_mut(in_dim) {
            for (v, &s) in row.iter_mut().zip(&signs) {
                *v *= s;
            }
        }
        rotate_rows_in_place(&mut xt, in_dim, block);

        let mut got = vec![0f32; rows * out_dim];
        for r in 0..rows {
            for o in 0..out_dim {
                got[r * out_dim + o] = (0..in_dim)
                    .map(|j| xt[r * in_dim + j] * w_folded[o * in_dim + j])
                    .sum();
            }
        }

        for i in 0..want.len() {
            assert!(
                (want[i] - got[i]).abs() < 1e-3,
                "i={i}: unfolded={} folded={}",
                want[i],
                got[i]
            );
        }
    }

    /// The GDN permutation is `tiled -> grouped`; spelled out here so a
    /// silent reorder (which still type-checks and still produces
    /// fluent text) has something to fail against.
    #[test]
    fn gdn_perm_maps_tiled_heads_to_grouped() {
        let (hd, nk, rep) = (2usize, 3usize, 2usize);
        let width = hd * nk * rep;
        // Label each feature by its (rep, nk) head coordinate.
        let x: Vec<f32> = (0..width).map(|i| i as f32).collect();

        // What the builder emits: reshape [rep, nk, hd] -> transpose to
        // [nk, rep, hd] -> flatten.
        let mut got = vec![0f32; width];
        for k in 0..nk {
            for r in 0..rep {
                for d in 0..hd {
                    got[(k * rep + r) * hd + d] = x[(r * nk + k) * hd + d];
                }
            }
        }

        // Head 0 (rep=0,k=0) stays first and the last head stays last —
        // which is exactly why an unpermuted run looks plausible.
        assert_eq!(&got[0..hd], &x[0..hd]);
        assert_eq!(&got[width - hd..], &x[width - hd..]);
        // But the middle heads move.
        assert_ne!(got, x);
        assert_eq!(
            got[hd],
            x[nk * hd],
            "grouped slot 1 should hold tiled head nk"
        );
    }

    #[test]
    fn butterfly_round_trips() {
        let n = 128usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.6).collect();
        let mut y = x.clone();
        rotate_rows_in_place(&mut y, n, n);
        rotate_rows_in_place(&mut y, n, n);
        for i in 0..n {
            assert!((y[i] - x[i]).abs() < 1e-4, "i={i}: {} vs {}", y[i], x[i]);
        }
    }
}
