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

//! The routed MoE: softmax router, top-k renormalized, plain SwiGLU experts,
//! plus one always-on shared expert.
//!
//! ```text
//!   p      = softmax(x @ W_gᵀ)                   [s, E]     (f32, no bias)
//!   idx    = top-k(p)                            w = p[idx] / Σ p[idx] · routed_scale
//!   y      = Σ_k w_k · down_e(silu(gate_e x) · up_e x)  +  shared(x)
//! ```
//!
//! The router is spelled with generic ops (`Softmax`, `TopK`,
//! `GatherElements`) rather than a custom gate kernel, so it lowers on every
//! backend. Grouped routing is rejected by [`MistralLarge4Config::validate`]
//! (`n_group = 1` in every published checkpoint).
//!
//! Expert banks arrive in GGUF's `[E, out, in]` slab order. Packed banks feed
//! `DequantGroupedMatMul` directly; dense ones are transposed for
//! `GroupedMatMul`.
//!
//! [`MistralLarge4Config::validate`]: crate::config::MistralLarge4Config::validate

use anyhow::Result;
use rlx_flow::Emit;
use rlx_glm5next::common::{linear, packed_bank, scalar};
use rlx_ir::hir::HirMut;
use rlx_ir::op::Op;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

#[derive(Debug, Clone, Copy)]
pub struct MoeDims {
    pub hidden: usize,
    pub moe_inter: usize,
    pub n_routed: usize,
    pub top_k: usize,
    pub routed_scaling: f32,
    pub seq: usize,
}

/// One routed bank, ready to issue per-expert GEMMs.
#[derive(Clone, Copy)]
enum Bank {
    /// `[E, in, out]` f32 (already transposed from GGUF's `[E, out, in]`).
    Dense(HirNodeId),
    Packed(HirNodeId, QuantScheme),
}

impl Bank {
    fn issue(
        self,
        gb: &mut HirMut<'_>,
        x: HirNodeId,
        eidx: HirNodeId,
        rows: usize,
        out: usize,
    ) -> HirNodeId {
        match self {
            Bank::Dense(w) => gb.grouped_matmul(x, w, eidx),
            Bank::Packed(wq, scheme) => gb.0.dequant_grouped_matmul_packed(
                x,
                wq,
                eidx,
                scheme,
                Shape::new(&[rows, out], DType::F32),
            ),
        }
    }
}

/// Load the three routed banks, all packed or all dense — a half-packed layer
/// would mix layouts.
fn load_banks(emit: &mut Emit<'_>, prefix: &str) -> Result<[Bank; 3]> {
    let keys = [
        format!("{prefix}.ffn_gate_exps.weight"),
        format!("{prefix}.ffn_up_exps.weight"),
        format!("{prefix}.ffn_down_exps.weight"),
    ];
    let mut packed = Vec::with_capacity(3);
    for k in &keys {
        match packed_bank(emit, k)? {
            Some((node, scheme, _)) => packed.push(Bank::Packed(node, scheme)),
            None => break,
        }
    }
    if packed.len() == 3 {
        return Ok([packed[0], packed[1], packed[2]]);
    }
    if !packed.is_empty() {
        anyhow::bail!("mistral-large4: {prefix} has some routed banks packed and some not");
    }
    let mut dense = Vec::with_capacity(3);
    for k in &keys {
        let w = emit.load_param(k, false)?;
        let mut gb = HirMut::new(emit.hir());
        dense.push(Bank::Dense(gb.transpose_(w, vec![0, 2, 1])));
    }
    Ok([dense[0], dense[1], dense[2]])
}

/// Emit the routing: `(top_idx [s, k], weights [s, k])`.
pub fn emit_router(
    emit: &mut Emit<'_>,
    prefix: &str,
    x2d: HirNodeId,
    d: MoeDims,
) -> Result<(HirNodeId, HirNodeId)> {
    let f = DType::F32;
    let (s, k) = (d.seq, d.top_k);
    let logits = linear(emit, &format!("{prefix}.ffn_gate_inp.weight"), x2d)?; // [s, E]
    let tiny = scalar(emit, &format!("{prefix}.moe.tiny"), 1e-20);
    let scale = scalar(emit, &format!("{prefix}.moe.scale"), d.routed_scaling);
    let mut gb = HirMut::new(emit.hir());
    let probs = gb.sm(logits, -1);
    let idx = gb.add_node(Op::TopK { k }, vec![probs], Shape::new(&[s, k], f));
    let vals = gb.add_node(
        Op::GatherElements { axis: 1 },
        vec![probs, idx],
        Shape::new(&[s, k], f),
    );
    let denom = gb.sum(vals, vec![1], true);
    let denom = gb.add(denom, tiny);
    let w = gb.div(vals, denom);
    let w = gb.mul(w, scale);
    Ok((idx, w))
}

/// Emit the MoE FFN of block `prefix` over `x2d` (`[seq, hidden]`,
/// `ffn_norm`ed). Returns `[seq, hidden]`.
pub fn emit_moe(
    emit: &mut Emit<'_>,
    prefix: &str,
    x2d: HirNodeId,
    d: MoeDims,
) -> Result<HirNodeId> {
    let rows = d.seq;
    let banks = load_banks(emit, prefix)?;
    let (idx, w) = emit_router(emit, prefix, x2d, d)?;

    let sg = linear(emit, &format!("{prefix}.ffn_gate_shexp.weight"), x2d)?;
    let su = linear(emit, &format!("{prefix}.ffn_up_shexp.weight"), x2d)?;
    let sh = {
        let mut gb = HirMut::new(emit.hir());
        let a = gb.silu(sg);
        gb.mul(a, su)
    };
    let shared = linear(emit, &format!("{prefix}.ffn_down_shexp.weight"), sh)?;

    let mut gb = HirMut::new(emit.hir());
    let mut acc = shared;
    for ki in 0..d.top_k {
        let col = gb.narrow_(idx, 1, ki, 1);
        let eidx = gb.reshape_(col, vec![rows as i64]);
        let wcol = gb.narrow_(w, 1, ki, 1); // [rows, 1]
        let g = banks[0].issue(&mut gb, x2d, eidx, rows, d.moe_inter);
        let u = banks[1].issue(&mut gb, x2d, eidx, rows, d.moe_inter);
        let a = gb.silu(g);
        let hx = gb.mul(a, u);
        let down = banks[2].issue(&mut gb, hx, eidx, rows, d.hidden);
        let weighted = gb.mul(down, wcol);
        acc = gb.add(acc, weighted);
    }
    Ok(acc)
}

/// The dense SwiGLU MLP of the leading layers. Returns `[seq, hidden]`.
pub fn emit_dense_mlp(emit: &mut Emit<'_>, prefix: &str, x2d: HirNodeId) -> Result<HirNodeId> {
    let g = linear(emit, &format!("{prefix}.ffn_gate.weight"), x2d)?;
    let u = linear(emit, &format!("{prefix}.ffn_up.weight"), x2d)?;
    let h = {
        let mut gb = HirMut::new(emit.hir());
        let a = gb.silu(g);
        gb.mul(a, u)
    };
    linear(emit, &format!("{prefix}.ffn_down.weight"), h)
}

/// The routed half of the MoE over **slot-indexed** banks, for experts paged
/// from disk.
///
/// The banks hold exactly `rows * top_k` experts, slot `r * top_k + k` being
/// the expert row `r` fired at position `k` — gathered by the host
/// ([`crate::paged`]) before the graph runs. That makes the expert index a
/// constant instead of router output, so the graph never needs the 128-expert
/// banks. The routing weights `w` (`[rows, top_k]`) come in as an input,
/// computed by the same [`emit_router`] that chose the slots.
///
/// Returns `shared(h) + Σ_k w_k · expert_slot(h)` (`[rows, hidden]`). The banks
/// must be packed: overwriting a dense bank would race the in-graph transpose
/// that `GroupedMatMul` needs, which constant folding can bake in.
pub fn emit_moe_slots(
    emit: &mut Emit<'_>,
    prefix: &str,
    h2d: HirNodeId,
    w: HirNodeId,
    d: MoeDims,
) -> Result<HirNodeId> {
    let rows = d.seq;
    let banks = load_banks(emit, prefix)?;
    if banks.iter().any(|b| matches!(b, Bank::Dense(_))) {
        anyhow::bail!("mistral-large4: {prefix}: paged experts need quantized (packed) banks");
    }
    let slots: Vec<HirNodeId> = (0..d.top_k)
        .map(|ki| {
            let v: Vec<f32> = (0..rows).map(|r| (r * d.top_k + ki) as f32).collect();
            emit.synth_param(
                &format!("{prefix}.moe.slot.{ki}"),
                v,
                Shape::new(&[rows], DType::F32),
            )
        })
        .collect();
    let sg = linear(emit, &format!("{prefix}.ffn_gate_shexp.weight"), h2d)?;
    let su = linear(emit, &format!("{prefix}.ffn_up_shexp.weight"), h2d)?;
    let sh = {
        let mut gb = HirMut::new(emit.hir());
        let a = gb.silu(sg);
        gb.mul(a, su)
    };
    let shared = linear(emit, &format!("{prefix}.ffn_down_shexp.weight"), sh)?;
    let mut gb = HirMut::new(emit.hir());
    let mut acc = shared;
    for (ki, &eidx) in slots.iter().enumerate() {
        let wcol = gb.narrow_(w, 1, ki, 1);
        let g = banks[0].issue(&mut gb, h2d, eidx, rows, d.moe_inter);
        let u = banks[1].issue(&mut gb, h2d, eidx, rows, d.moe_inter);
        let a = gb.silu(g);
        let hx = gb.mul(a, u);
        let down = banks[2].issue(&mut gb, hx, eidx, rows, d.hidden);
        let weighted = gb.mul(down, wcol);
        acc = gb.add(acc, weighted);
    }
    Ok(acc)
}
