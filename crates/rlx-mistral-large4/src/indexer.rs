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

//! The DSA lightning indexer — per-token top-k key selection (DeepSeek-V3.2
//! style, no pooling), with Mistral Large 4's two changes: the indexer query
//! comes straight from the hidden state (`LayerNorm → W_q_b → SSMax`, there is
//! no Q-LoRA to share), and NoPE layers skip RoPE in the indexer too.
//!
//! ```text
//!   q      = ssmax(W_q_b · LN_q(x))            [s, n_h, d]   (rope on [:rope_l])
//!   k      = LN_k(W_k · x)                     [s, d]
//!   score  = Σ_h  (W_w · x)_h / √n_h · relu(q_h · k / √d)     [s, s]
//!   keep   = top-k(score + causal, index_topk)
//! ```
//!
//! llama.cpp additionally Hadamard-rotates `q` and `k` before scoring. That is
//! an orthogonal transform applied to both, so it cannot change `q · k`; it is
//! not emitted here.
//!
//! **Shared** layers (`indexer_types = "shared"`) have no indexer weights and
//! reuse the previous full layer's selection — the caller threads the bias.
//!
//! ## When this is a no-op
//!
//! With `seq <= index_topk` every query's causal keys fit the budget, so the
//! selection keeps all of them and the mask is exactly causal.
//! [`IndexerDims::is_dense`] tests that, and the caller then uses the fused
//! causal attention instead — exact, not an approximation. Set
//! [`IndexerDims::force_emit`] to run the selection anyway (tests).

use anyhow::Result;
use rlx_flow::Emit;
use rlx_glm5next::common::{linear, scalar};
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, BinaryOp, Op, ScatterNdReduction};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::attention::MASK_PENALTY;
use crate::rope::{PosTables, apply_ssmax, rotate_interleaved};

#[derive(Debug, Clone, Copy)]
pub struct IndexerDims {
    pub hidden: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// Rotary width on this layer (0 on NoPE layers).
    pub rope_dim: usize,
    pub topk: usize,
    pub norm_eps: f32,
    pub seq: usize,
    /// Emit the selection even where [`Self::is_dense`] proves it is the
    /// identity, so the two paths can be compared.
    pub force_emit: bool,
}

impl IndexerDims {
    pub fn is_dense(&self) -> bool {
        self.seq <= self.topk
    }
}

/// `[s, s]` lower-triangular visibility.
fn causal(seq: usize) -> Vec<f32> {
    let mut m = vec![0.0; seq * seq];
    for q in 0..seq {
        for k in 0..=q {
            m[q * seq + k] = 1.0;
        }
    }
    m
}

/// Emit the `[1, 1, s, s]` additive mask implied by this layer's selection
/// (causality included), or `None` when the selection is the identity.
pub fn emit_dsa_bias(
    emit: &mut Emit<'_>,
    prefix: &str,
    x2d: HirNodeId, // [s, hidden], attn_norm'ed
    d: IndexerDims,
    t: &PosTables,
) -> Result<Option<HirNodeId>> {
    if d.is_dense() && !d.force_emit {
        return Ok(None);
    }
    let f = DType::F32;
    let (s, nh, hd) = (d.seq, d.n_heads, d.head_dim);
    let (si, nhi, hdi) = (s as i64, nh as i64, hd as i64);
    let k_sel = d.topk.min(s);
    let p = format!("{prefix}.indexer");

    let qn_w = emit.load_param(&format!("{p}.q_norm.weight"), false)?;
    let qn_b = emit.load_param(&format!("{p}.q_norm.bias"), false)?;
    let q_in = {
        let mut gb = HirMut::new(emit.hir());
        gb.ln(x2d, qn_w, qn_b, d.norm_eps)
    };
    let q = linear(emit, &format!("{p}.attn_q_b.weight"), q_in)?; // [s, nh*hd]
    let k_raw = linear(emit, &format!("{p}.attn_k.weight"), x2d)?; // [s, hd]
    let kn_w = emit.load_param(&format!("{p}.k_norm.weight"), false)?;
    let kn_b = emit.load_param(&format!("{p}.k_norm.bias"), false)?;
    let head_w = linear(emit, &format!("{p}.proj.weight"), x2d)?; // [s, nh]
    let ss_w = emit.load_param(&format!("{p}.ssmax.weight"), false)?;
    let ss_b = emit.load_param(&format!("{p}.ssmax.bias"), false)?;

    let visible_causal = emit.synth_param(
        &format!("{p}.dsa.causal"),
        causal(s),
        Shape::new(&[s, s], f),
    );
    let ones = emit.synth_param(
        &format!("{p}.dsa.ones"),
        vec![1.0; s * k_sel],
        Shape::new(&[s, k_sel], f),
    );
    let zeros = {
        let z = emit.synth_zeros(&format!("{p}.dsa.zeros"), s * s);
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(z, vec![si, si])
    };
    let one = scalar(emit, &format!("{p}.dsa.one"), 1.0);
    let penalty = scalar(emit, &format!("{p}.dsa.neg"), MASK_PENALTY);
    let qk_scale = scalar(emit, &format!("{p}.dsa.scale"), (hd as f32).powf(-0.5));
    let head_scale = scalar(emit, &format!("{p}.dsa.hscale"), (nh as f32).powf(-0.5));

    let mut gb = HirMut::new(emit.hir());
    let q3 = gb.reshape_(q, vec![si, nhi, hdi]);
    let q3 = apply_ssmax(&mut gb, q3, ss_w, ss_b, t, nh);
    let keys = gb.ln(k_raw, kn_w, kn_b, d.norm_eps); // [s, hd]

    // RoPE occupies the head's *leading* `rope_dim` dims in the indexer
    // (`[rope | pass]`), unlike the attention head (`[nope | rope]`).
    let (q3, keys) = if d.rope_dim > 0 {
        let r = d.rope_dim;
        let q_rot = gb.narrow_(q3, 2, 0, r);
        let q_pass = gb.narrow_(q3, 2, r, hd - r);
        let q_rot = rotate_interleaved(&mut gb, q_rot, t, nh);
        let q3 = gb.concat_(vec![q_rot, q_pass], 2);
        let k3 = gb.reshape_(keys, vec![si, 1, hdi]);
        let k_rot = gb.narrow_(k3, 2, 0, r);
        let k_pass = gb.narrow_(k3, 2, r, hd - r);
        let k_rot = rotate_interleaved(&mut gb, k_rot, t, 1);
        let k3 = gb.concat_(vec![k_rot, k_pass], 2);
        (q3, gb.reshape_(k3, vec![si, hdi]))
    } else {
        (q3, keys)
    };

    let qh = gb.transpose_(q3, vec![1, 0, 2]); // [nh, s, hd]
    let k_t = gb.transpose_(keys, vec![1, 0]); // [hd, s]
    let scores = gb.mm(qh, k_t); // [nh, s, s]
    let scores = gb.mul(scores, qk_scale);
    let scores = gb.add_node(
        Op::Activation(Activation::Relu),
        vec![scores],
        Shape::new(&[nh, s, s], f),
    );
    let w = gb.mul(head_w, head_scale); // [s, nh]
    let w = gb.transpose_(w, vec![1, 0]);
    let w = gb.reshape_(w, vec![nhi, si, 1]);
    let weighted = gb.mul(scores, w);
    let index_scores = gb.sum(weighted, vec![0], false); // [s, s]

    // Future keys get the penalty so they lose to every visible key.
    let invisible = gb.sub(one, visible_causal);
    let pen = gb.mul(invisible, penalty);
    let masked = gb.add(index_scores, pen);
    let chosen = gb.add_node(
        Op::TopK { k: k_sel },
        vec![masked],
        Shape::new(&[s, k_sel], f),
    );

    // Top-k does not filter: early queries with fewer than `k_sel` visible keys
    // still get `k_sel` picks, some in the future. Scatter, then intersect with
    // causality so those picks stay masked.
    let picked = gb.add_node(
        Op::ScatterElements {
            axis: 1,
            reduction: ScatterNdReduction::Max,
        },
        vec![zeros, chosen, ones],
        Shape::new(&[s, s], f),
    );
    let visible = gb.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![picked, visible_causal],
        Shape::new(&[s, s], f),
    );
    let hidden_m = gb.sub(one, visible);
    let bias = gb.mul(hidden_m, penalty);
    Ok(Some(gb.reshape_(bias, vec![1, 1, si, si])))
}
