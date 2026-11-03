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

//! Multi-head latent attention without Q-LoRA, with SSMax queries, NoPE
//! layers and cross-layer KV sharing.
//!
//! ```text
//!   q        = x @ W_q                     [s, h, 128]  = [nope_l | rope_l]
//!   latent   = x @ W_kv_a                  [s, 512 + rope_l]   (owning layers only)
//!   k_pass   = rms_norm(latent[:512])      k_rot = rope(latent[512:])
//!   k_h      = [k_pass @ k_b_h | k_rot]    v_h = k_pass @ v_b_hᵀ
//!   q        = ssmax(cat(q_nope, rope(q_rot)))
//!   out      = softmax(q·kᵀ / √128 + mask) · v  @ W_o
//! ```
//!
//! On a NoPE layer `rope_l = 0` and the whole 128-wide head is non-rotary, so
//! `attn_k_b` is `[h, 512, 128]` there and `[h, 512, 64]` on a RoPE layer.
//!
//! A layer whose `kv_source` is an earlier layer has no `attn_kv_a_mqa`: it
//! reuses that layer's `(k_pass, k_rot)` — already normalized and rotated — and
//! expands it through its **own** `attn_k_b` / `attn_v_b`.
//!
//! GGUF orientation (ne reversed): `attn_k_b` `[h, kv_lora, nope_l]`,
//! `attn_v_b` `[h, v_head, kv_lora]`. Prefill expands the latent and runs the
//! fused attention; decode absorbs the query into latent space so the cache is
//! the latent only. The two are algebraically identical, which is what
//! `tests/decode_equivalence.rs` relies on.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_glm5next::common::{linear, rms_norm, scalar};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::rope::{PosTables, apply_ssmax, rotate_interleaved};

/// Additive penalty standing in for `-inf` (finite, so no all-masked NaN).
pub const MASK_PENALTY: f32 = -1.0e30;

#[derive(Debug, Clone, Copy)]
pub struct AttnDims {
    pub hidden: usize,
    pub heads: usize,
    pub kv_lora: usize,
    /// Full query/key head width (128 on every layer).
    pub qk_head: usize,
    pub v_head: usize,
    pub eps: f32,
    pub seq: usize,
}

/// A layer's normalized latent and rotated shared key, as later layers reuse it.
#[derive(Debug, Clone, Copy)]
pub struct Latent {
    /// `[seq, kv_lora]`.
    pub k_pass: HirNodeId,
    /// `[seq, rope_l]`, `None` on a NoPE layer.
    pub k_rot: Option<HirNodeId>,
}

/// Rotate and SSMax-scale a `[rows, h, qk]` query.
fn finish_query(
    emit: &mut Emit<'_>,
    prefix: &str,
    q3: HirNodeId,
    heads: usize,
    nope: usize,
    rope_dim: usize,
    t: &PosTables,
) -> Result<HirNodeId> {
    let ss_w = emit.load_param(&format!("{prefix}.attn_ssmax.weight"), false)?;
    let ss_b = emit.load_param(&format!("{prefix}.attn_ssmax.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    let q3 = if rope_dim > 0 {
        let q_nope = gb.narrow_(q3, 2, 0, nope);
        let q_rot = gb.narrow_(q3, 2, nope, rope_dim);
        let q_rot = rotate_interleaved(&mut gb, q_rot, t, heads);
        gb.concat_(vec![q_nope, q_rot], 2)
    } else {
        q3
    };
    Ok(apply_ssmax(&mut gb, q3, ss_w, ss_b, t, heads))
}

/// Compute an owning layer's latent row(s) from `x2d` (`[rows, hidden]`).
fn own_latent(
    emit: &mut Emit<'_>,
    prefix: &str,
    x2d: HirNodeId,
    d: AttnDims,
    rope_dim: usize,
    t: &PosTables,
) -> Result<Latent> {
    let ckv = linear(emit, &format!("{prefix}.attn_kv_a_mqa.weight"), x2d)?; // [rows, kv + rope]
    let raw = {
        let mut gb = HirMut::new(emit.hir());
        gb.narrow_(ckv, 1, 0, d.kv_lora)
    };
    let k_pass = rms_norm(
        emit,
        &format!("{prefix}.attn_kv_a_norm"),
        raw,
        d.kv_lora,
        d.eps,
    )?;
    let k_rot = (rope_dim > 0).then(|| {
        let mut gb = HirMut::new(emit.hir());
        let r = gb.narrow_(ckv, 1, d.kv_lora, rope_dim);
        let r = gb.reshape_(r, vec![t.rows as i64, 1, rope_dim as i64]);
        let r = rotate_interleaved(&mut gb, r, t, 1);
        gb.reshape_(r, vec![t.rows as i64, rope_dim as i64])
    });
    Ok(Latent { k_pass, k_rot })
}

/// Emit one prefill attention layer for block `prefix` over `x2d`
/// (`[seq, hidden]`, already `attn_norm`ed). Returns `([seq, hidden], latent)`.
///
/// `shared` is the KV source's latent on a KV-sharing layer, `None` on an
/// owning one. `dsa_bias` is the `[1, 1, s, s]` additive DSA mask (it already
/// includes causality); `None` means DSA is the identity and the mask is
/// plain causal.
#[allow(clippy::too_many_arguments)]
pub fn emit_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    x2d: HirNodeId,
    d: AttnDims,
    rope_dim: usize,
    shared: Option<Latent>,
    t: &PosTables,
    dsa_bias: Option<HirNodeId>,
) -> Result<(HirNodeId, Latent)> {
    let f = DType::F32;
    let (s, h, qk, vd) = (d.seq, d.heads, d.qk_head, d.v_head);
    let nope = qk - rope_dim;
    let (si, hi, qki, vdi) = (s as i64, h as i64, qk as i64, vd as i64);

    let q = linear(emit, &format!("{prefix}.attn_q.weight"), x2d)?; // [s, h*qk]
    let q3 = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(q, vec![si, hi, qki])
    };
    let q3 = finish_query(emit, prefix, q3, h, nope, rope_dim, t)?;

    let latent = match shared {
        Some(l) => l,
        None => own_latent(emit, prefix, x2d, d, rope_dim, t)?,
    };

    let k_b = emit.load_param(&format!("{prefix}.attn_k_b.weight"), false)?; // [h, kv, nope]
    let v_b = emit.load_param(&format!("{prefix}.attn_v_b.weight"), false)?; // [h, v, kv]

    let attn = {
        let mut gb = HirMut::new(emit.hir());
        // Batched operand on the left: a rank-2 lhs against a batched rhs is
        // the direction rlx-cpu does not take its batched-GEMM path for (see
        // rlx-glm5next `tests/mm_broadcast.rs`).
        let k_pass_t = gb.transpose_(latent.k_pass, vec![1, 0]); // [kv, s]
        let k_b_t = gb.transpose_(k_b, vec![0, 2, 1]); // [h, nope, kv]
        let k_nope = {
            let t = gb.mm(k_b_t, k_pass_t); // [h, nope, s]
            gb.transpose_(t, vec![0, 2, 1]) // [h, s, nope]
        };
        let keys = match latent.k_rot {
            Some(kr) => {
                let kr = gb.reshape_(kr, vec![1, si, rope_dim as i64]);
                let kr = gb.expand_(kr, vec![hi, si, rope_dim as i64]);
                gb.concat_(vec![k_nope, kr], 2) // [h, s, qk]
            }
            None => k_nope,
        };
        let values = {
            let t = gb.mm(v_b, k_pass_t); // [h, v, s]
            gb.transpose_(t, vec![0, 2, 1]) // [h, s, v]
        };
        let kf = {
            let t = gb.transpose_(keys, vec![1, 0, 2]);
            gb.reshape_(t, vec![1, si, hi * qki])
        };
        let vf = {
            let t = gb.transpose_(values, vec![1, 0, 2]);
            gb.reshape_(t, vec![1, si, hi * vdi])
        };
        let qf = gb.reshape_(q3, vec![1, si, hi * qki]);
        // The op's default score scale is head_dim^-0.5 = qk_head^-0.5, which
        // is the reference's `scaling` (no YaRN mscale_all_dim).
        let out_shape = Shape::new(&[1, s, h * vd], f);
        let out = match dsa_bias {
            None => gb.attention_kind(qf, kf, vf, h, qk, MaskKind::Causal, out_shape),
            Some(bias) => {
                let bias = gb.expand_(bias, vec![1, hi, si, si]);
                gb.attention_bias(qf, kf, vf, bias, h, qk, out_shape)
            }
        };
        gb.reshape_(out, vec![si, hi * vdi])
    };
    let out = linear(emit, &format!("{prefix}.attn_output.weight"), attn)?;
    Ok((out, latent))
}

/// One step's view of a latent cache: every cached row plus this token's.
#[derive(Debug, Clone, Copy)]
pub struct DecodeKv {
    /// `[cap + 1, kv_lora + rope_l]` — `[k_pass | rotated k_rot]` per row; the
    /// last row is the current token.
    pub all: HirNodeId,
    /// `[1, cap + 1]` — `1.0` valid, `0.0` empty slot.
    pub mask: HirNodeId,
    pub len: usize,
}

/// Emit one **decode step** (a single token) of block `prefix`.
///
/// `cache` is `Ok((latent_in, mask))` on an owning layer — the cached rows
/// `[1, cap, W]` — or `Err(kv)` on a KV-sharing layer, the source layer's
/// [`DecodeKv`] from this same step. Returns `(out [1, hidden], kv, new_row)`;
/// `new_row` (`[1, W]`, owning layers only) is what the caller stores.
pub fn emit_attention_decode(
    emit: &mut Emit<'_>,
    prefix: &str,
    x2d: HirNodeId,
    d: AttnDims,
    rope_dim: usize,
    cache: std::result::Result<(HirNodeId, HirNodeId, usize), DecodeKv>,
    t: &PosTables,
) -> Result<(HirNodeId, DecodeKv, Option<HirNodeId>)> {
    let (h, qk, vd, kv) = (d.heads, d.qk_head, d.v_head, d.kv_lora);
    let nope = qk - rope_dim;
    let width = kv + rope_dim;
    let (hi, qki, vdi) = (h as i64, qk as i64, vd as i64);

    let q = linear(emit, &format!("{prefix}.attn_q.weight"), x2d)?; // [1, h*qk]
    let q3 = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(q, vec![1, hi, qki])
    };
    let q3 = finish_query(emit, prefix, q3, h, nope, rope_dim, t)?;

    let (kvs, new_row) = match cache {
        Err(shared) => (shared, None),
        Ok((latent_in, mask, cap)) => {
            let l = own_latent(emit, prefix, x2d, d, rope_dim, t)?;
            let mut gb = HirMut::new(emit.hir());
            let row = match l.k_rot {
                Some(kr) => gb.concat_(vec![l.k_pass, kr], 1),
                None => l.k_pass,
            }; // [1, W]
            let cached = gb.reshape_(latent_in, vec![cap as i64, width as i64]);
            let all = gb.concat_(vec![cached, row], 0); // [cap + 1, W]
            (
                DecodeKv {
                    all,
                    mask,
                    len: cap + 1,
                },
                Some(row),
            )
        }
    };

    let k_b = emit.load_param(&format!("{prefix}.attn_k_b.weight"), false)?; // [h, kv, nope]
    let v_b = emit.load_param(&format!("{prefix}.attn_v_b.weight"), false)?; // [h, v, kv]
    let scale = scalar(
        emit,
        &format!("{prefix}.attn.dec.scale"),
        (qk as f32).powf(-0.5),
    );
    let one = scalar(emit, &format!("{prefix}.attn.dec.one"), 1.0);
    let penalty = scalar(emit, &format!("{prefix}.attn.dec.neg"), MASK_PENALTY);

    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let len = kvs.len as i64;
        let qh = gb.transpose_(q3, vec![1, 0, 2]); // [h, 1, qk]
        let q_nope = gb.narrow_(qh, 2, 0, nope);
        // Absorb: q_nope · (k_pass @ k_b) = (q_nope @ k_bᵀ) · k_pass.
        let k_b_t = gb.transpose_(k_b, vec![0, 2, 1]); // [h, nope, kv]
        let q_lat = gb.mm(q_nope, k_b_t); // [h, 1, kv]
        let k_pass = gb.narrow_(kvs.all, 1, 0, kv); // [len, kv]
        let k_pass_t = gb.transpose_(k_pass, vec![1, 0]);
        let mut scores = gb.mm(q_lat, k_pass_t); // [h, 1, len]
        if rope_dim > 0 {
            let q_rot = gb.narrow_(qh, 2, nope, rope_dim); // [h, 1, rope]
            let k_rot = gb.narrow_(kvs.all, 1, kv, rope_dim); // [len, rope]
            let k_rot_t = gb.transpose_(k_rot, vec![1, 0]);
            let rs = gb.mm(q_rot, k_rot_t);
            scores = gb.add(scores, rs);
        }
        let scores = gb.mul(scores, scale);
        let inv = gb.sub(one, kvs.mask);
        let bias = gb.mul(inv, penalty);
        let bias = gb.reshape_(bias, vec![1, 1, len]);
        let scores = gb.add(scores, bias);
        let probs = gb.sm(scores, -1);
        let out_lat = gb.mm(probs, k_pass); // [h, 1, kv]
        let v_b_t = gb.transpose_(v_b, vec![0, 2, 1]); // [h, kv, v]
        let out = gb.mm(out_lat, v_b_t); // [h, 1, v]
        gb.reshape_(out, vec![1, hi * vdi])
    };
    let out = linear(emit, &format!("{prefix}.attn_output.weight"), attn)?;
    Ok((out, kvs, new_row))
}
