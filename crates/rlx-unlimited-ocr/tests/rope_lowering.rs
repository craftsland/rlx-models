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

//! Minimal reproduction for the compiled prefill RoPE path.
//!
//! `lm_graph::apply_rope_bhsd` is `reshape -> transpose(BSHD->BHSD) -> Op::Rope
//! -> transpose back -> reshape`, fed cos/sin tables declared at
//! `[max_positions, half_dim]` while the tensor being rotated only has `seq`
//! positions.
//!
//! The host reference is `nn::apply_rope`, which the eager [`lm_flow`] path
//! uses and which reproduces the published checkpoint's logits exactly. RoPE is
//! a rotation, so whatever the layout, it must preserve every token's per-head
//! norm — that is the invariant these tests pin.
//!
//! Run on every backend compiled in and available (`--features all-backends`),
//! across the RoPE flavours the shared builders actually emit: both pairing
//! conventions, partial rotation, GQA head counts, decode-width sequences and
//! multi-batch. A single BHSD/NeoX/full-rotation shape proves very little —
//! the rank-4 bug below was invisible at `heads == 1`, and the CPU thunk's
//! rank-2 sibling bug had been invisible for the same reason.

use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_flow::ModelFlow;
use rlx_flow::blocks::RopeTablesStage;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::op::RopeStyle;
use rlx_ir::{DType, Op, Shape};
use rlx_runtime::Device;
use rlx_unlimited_ocr::nn;

use rlx_core::backend_matrix::{Failures, available_devices, max_abs_diff};
use std::collections::HashMap;

const MAX_POS: usize = 32_768;
const THETA: f64 = 1_000_000.0;

/// One RoPE configuration to check.
#[derive(Clone, Copy, Debug)]
struct Case {
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    /// `n_rot < head_dim` leaves the tail of each head unrotated.
    n_rot: usize,
    style: RopeStyle,
}

impl Case {
    const fn new(batch: usize, seq: usize, heads: usize, head_dim: usize) -> Self {
        Self {
            batch,
            seq,
            heads,
            head_dim,
            n_rot: head_dim,
            style: RopeStyle::NeoX,
        }
    }
    const fn n_rot(mut self, n: usize) -> Self {
        self.n_rot = n;
        self
    }
    const fn gptj(mut self) -> Self {
        self.style = RopeStyle::GptJ;
        self
    }
    fn elems(&self) -> usize {
        self.batch * self.seq * self.heads * self.head_dim
    }
}

/// The decoder's own geometry plus the neighbours that would have caught the
/// rank-4 bug earlier: `heads == 1` (where it is a no-op), decode-width
/// `seq == 1`, batch > 1, GQA-shaped head counts, partial rotation and the
/// GPT-J/interleaved pairing other checkpoints use.
const CASES: &[Case] = &[
    Case::new(1, 4, 10, 128),          // smoke, model head geometry
    Case::new(1, 927, 10, 128),        // the real prefill length
    Case::new(1, 1, 10, 128),          // decode width
    Case::new(2, 16, 10, 128),         // batch > 1
    Case::new(1, 16, 1, 128),          // heads == 1: where the bug hid
    Case::new(1, 16, 2, 64),           // GQA-shaped kv heads
    Case::new(1, 16, 8, 64).n_rot(32), // partial rotation
    Case::new(1, 16, 4, 64).gptj(),    // interleaved pairing
];

fn input_for(c: &Case) -> Vec<f32> {
    (0..c.elems())
        .map(|i| ((i as f32 * 0.013).sin()) * 0.5 + 0.01)
        .collect()
}

/// `apply_rope_bhsd` as the prefill graph builds it, for one case.
fn compiled_rope(x: &[f32], c: Case, device: Device) -> Vec<f32> {
    let (cos_data, sin_data) = nn::rope_tables(MAX_POS, c.head_dim, THETA);
    let half = c.head_dim / 2;
    let hidden = c.heads * c.head_dim;
    let shape = Shape::new(&[c.batch, c.seq, hidden], DType::F32);

    let built = ModelFlow::new("rope_repro")
        .input("x", shape.clone())
        .rope_tables(RopeTablesStage::param(MAX_POS, half, cos_data, sin_data))
        .plugin_named("rope", move |emit, hidden_val| {
            let x_id = hidden_val.expect("hidden").hir_id();
            let cos = emit.state.rope_cos.expect("cos");
            let sin = emit.state.rope_sin.expect("sin");
            let mut g = HirMut::new(emit.hir());
            let g = &mut g;
            let x4 = g.reshape_(
                x_id,
                vec![
                    c.batch as i64,
                    c.seq as i64,
                    c.heads as i64,
                    c.head_dim as i64,
                ],
            );
            let bhsd = g.transpose_(x4, vec![0, 2, 1, 3]);
            let roped = g.rope_n_styled(bhsd, cos, sin, c.head_dim, c.n_rot, c.style);
            let bshd = g.transpose_(roped, vec![0, 2, 1, 3]);
            let out = g.reshape_(
                bshd,
                vec![c.batch as i64, c.seq as i64, (c.heads * c.head_dim) as i64],
            );
            Ok(Some(emit.wrap(
                out,
                Shape::new(&[c.batch, c.seq, c.heads * c.head_dim], DType::F32),
            )))
        })
        .output("y")
        .build(&mut WeightLoaderSource(&mut WeightMap::from_tensors(
            HashMap::new(),
        )))
        .expect("build rope repro");

    let mut compiled = compile_built(built, device).expect("compile");
    compiled.run(&[("x", x)]).swap_remove(0)
}

/// The same chain with the `Op::Rope` node removed — `reshape -> transpose ->
/// transpose -> reshape` is the identity. If this fails the harness is wrong
/// and the RoPE results below would be artefacts rather than findings.
fn compiled_identity(x: &[f32], c: Case, device: Device) -> Vec<f32> {
    let (cos_data, sin_data) = nn::rope_tables(MAX_POS, c.head_dim, THETA);
    let half = c.head_dim / 2;
    let hidden = c.heads * c.head_dim;
    let shape = Shape::new(&[c.batch, c.seq, hidden], DType::F32);

    let built = ModelFlow::new("identity_repro")
        .input("x", shape.clone())
        .rope_tables(RopeTablesStage::param(MAX_POS, half, cos_data, sin_data))
        .plugin_named("identity", move |emit, hidden_val| {
            let x_id = hidden_val.expect("hidden").hir_id();
            let mut g = HirMut::new(emit.hir());
            let g = &mut g;
            let x4 = g.reshape_(
                x_id,
                vec![
                    c.batch as i64,
                    c.seq as i64,
                    c.heads as i64,
                    c.head_dim as i64,
                ],
            );
            let bhsd = g.transpose_(x4, vec![0, 2, 1, 3]);
            let bshd = g.transpose_(bhsd, vec![0, 2, 1, 3]);
            let out = g.reshape_(
                bshd,
                vec![c.batch as i64, c.seq as i64, (c.heads * c.head_dim) as i64],
            );
            Ok(Some(emit.wrap(
                out,
                Shape::new(&[c.batch, c.seq, c.heads * c.head_dim], DType::F32),
            )))
        })
        .output("y")
        .build(&mut WeightLoaderSource(&mut WeightMap::from_tensors(
            HashMap::new(),
        )))
        .expect("build identity repro");

    let mut compiled = compile_built(built, device).expect("compile");
    compiled.run(&[("x", x)]).swap_remove(0)
}

/// Host reference: `nn::apply_rope` for the NeoX/full-rotation case, extended
/// here to the interleaved pairing and partial rotation so every case has an
/// oracle that does not come from the thing under test.
fn host_rope(x: &[f32], c: Case) -> Vec<f32> {
    let (cos, sin) = nn::rope_tables(MAX_POS, c.head_dim, THETA);
    let half = c.head_dim / 2;
    let rot_half = c.n_rot / 2;
    let mut out = x.to_vec();
    for b in 0..c.batch {
        for t in 0..c.seq {
            let pos = t; // positions start at 0 in prefill
            for h in 0..c.heads {
                let base = ((b * c.seq + t) * c.heads + h) * c.head_dim;
                let head = &mut out[base..base + c.head_dim];
                let src: Vec<f32> = head.to_vec();
                for i in 0..rot_half {
                    let (cv, sv) = (cos[pos * half + i], sin[pos * half + i]);
                    match c.style {
                        RopeStyle::NeoX => {
                            let (x1, x2) = (src[i], src[rot_half + i]);
                            head[i] = x1 * cv - x2 * sv;
                            head[rot_half + i] = x2 * cv + x1 * sv;
                        }
                        RopeStyle::GptJ => {
                            let (x1, x2) = (src[2 * i], src[2 * i + 1]);
                            head[2 * i] = x1 * cv - x2 * sv;
                            head[2 * i + 1] = x2 * cv + x1 * sv;
                        }
                    }
                }
            }
        }
    }
    out
}

fn head_norm(x: &[f32], c: &Case, b: usize, t: usize, h: usize) -> f32 {
    let base = ((b * c.seq + t) * c.heads + h) * c.head_dim;
    x[base..base + c.head_dim]
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
}

/// Harness control — must pass for the RoPE tests to mean anything.
#[test]
fn control_reshape_transpose_roundtrip_is_identity() {
    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        for &c in CASES {
            let x = input_for(&c);
            let y = compiled_identity(&x, c, device);
            if y.len() != x.len() {
                fails.push(name, format!("{c:?}: length {} != {}", y.len(), x.len()));
                continue;
            }
            let worst = max_abs_diff(&x, &y);
            if worst >= 1e-4 {
                let zeros = y.iter().filter(|v| **v == 0.0).count();
                fails.push(
                    name,
                    format!(
                        "{c:?}: transpose round-trip is not identity (max diff {worst}); \
                         {zeros}/{} outputs zero — the harness itself is broken here",
                        y.len()
                    ),
                );
            }
        }
    }
    fails.assert_empty("reshape/transpose round-trip");
}

/// RoPE rotates each pair, so every head's norm is invariant — whatever the
/// layout, pairing convention or rotation width. Any lowering that drops or
/// zeroes elements breaks this.
#[test]
fn compiled_rope_preserves_every_head_norm() {
    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        for &c in CASES {
            let x = input_for(&c);
            let y = compiled_rope(&x, c, device);
            if y.len() != x.len() {
                fails.push(name, format!("{c:?}: length {} != {}", y.len(), x.len()));
                continue;
            }
            let zeros = y.iter().filter(|v| **v == 0.0).count();
            let mut worst = (0usize, 0usize, 0usize, 0f32);
            for b in 0..c.batch {
                for t in 0..c.seq {
                    for h in 0..c.heads {
                        let (a, bn) = (head_norm(&x, &c, b, t, h), head_norm(&y, &c, b, t, h));
                        let rel = (a - bn).abs() / a.max(1e-6);
                        if rel > worst.3 {
                            worst = (b, t, h, rel);
                        }
                    }
                }
            }
            if worst.3 >= 1e-3 {
                fails.push(
                    name,
                    format!(
                        "{c:?}: head norms changed (batch {} token {} head {} rel err {:.4}); \
                         {zeros}/{} outputs exactly zero",
                        worst.0,
                        worst.1,
                        worst.2,
                        worst.3,
                        y.len()
                    ),
                );
            }
        }
    }
    fails.assert_empty("RoPE head-norm invariance");
}

/// The compiled lowering must agree with the host reference elementwise.
#[test]
fn compiled_rope_matches_host_reference() {
    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        for &c in CASES {
            let x = input_for(&c);
            let want = host_rope(&x, c);
            let got = compiled_rope(&x, c, device);
            let worst = max_abs_diff(&want, &got);
            if worst >= 1e-3 {
                fails.push(name, format!("{c:?}: differs from the host by {worst:.5}"));
            }
        }
    }
    fails.assert_empty("compiled RoPE");
}

/// Partial rotation must leave the tail of each head untouched — the classic
/// way a `n_rot` bug hides, since the rotated half still looks plausible.
#[test]
fn partial_rotation_leaves_the_tail_untouched() {
    let c = Case::new(1, 16, 8, 64).n_rot(32);
    let x = input_for(&c);
    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        let y = compiled_rope(&x, c, device);
        for t in 0..c.seq {
            for h in 0..c.heads {
                let base = (t * c.heads + h) * c.head_dim;
                for j in c.n_rot..c.head_dim {
                    if (x[base + j] - y[base + j]).abs() >= 1e-6 {
                        fails.push(
                            name,
                            format!(
                                "token {t} head {h} dim {j}: tail changed {} -> {}",
                                x[base + j],
                                y[base + j]
                            ),
                        );
                    }
                }
            }
        }
    }
    fails.assert_empty("partial-rotation tail");
}

/// `Op::RopeBackward` must be the forward rotation with the sin table negated,
/// and must write the *whole* gradient.
///
/// It carried the same rank-4 shape bug as the forward path in every backend —
/// unnoticed because nothing here trains a BHSD-RoPE model, and a mostly-zero
/// gradient degrades training silently rather than failing.
#[test]
fn rope_backward_matches_the_negated_forward_rotation() {
    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        for &c in CASES {
            let dy = input_for(&c);
            // d/dx of a rotation by theta is a rotation by -theta: same kernel,
            // negated sin. Build the oracle from the forward host reference so
            // it does not come from the thing under test.
            let want = host_rope_signed(&dy, c, -1.0);
            let got = compiled_rope_backward(&dy, c, device);
            if got.len() != dy.len() {
                fails.push(name, format!("{c:?}: length {} != {}", got.len(), dy.len()));
                continue;
            }
            let zeros = got.iter().filter(|v| **v == 0.0).count();
            let worst = max_abs_diff(&want, &got);
            if worst >= 1e-3 {
                fails.push(
                    name,
                    format!(
                        "{c:?}: differs from the negated forward rotation by {worst:.5}; \
                         {zeros}/{} gradient elements are exactly zero",
                        got.len()
                    ),
                );
            }
        }
    }
    fails.assert_empty("RopeBackward");
}

/// Forward rotation with `sin` scaled by `sign` (`-1` gives the backward pass).
fn host_rope_signed(x: &[f32], c: Case, sign: f32) -> Vec<f32> {
    let (cos, sin) = nn::rope_tables(MAX_POS, c.head_dim, THETA);
    let half = c.head_dim / 2;
    let rot_half = c.n_rot / 2;
    let mut out = x.to_vec();
    for b in 0..c.batch {
        for t in 0..c.seq {
            for h in 0..c.heads {
                let base = ((b * c.seq + t) * c.heads + h) * c.head_dim;
                let head = &mut out[base..base + c.head_dim];
                let src: Vec<f32> = head.to_vec();
                for i in 0..rot_half {
                    let cv = cos[t * half + i];
                    let sv = sin[t * half + i] * sign;
                    match c.style {
                        RopeStyle::NeoX => {
                            let (x1, x2) = (src[i], src[rot_half + i]);
                            head[i] = x1 * cv - x2 * sv;
                            head[rot_half + i] = x2 * cv + x1 * sv;
                        }
                        RopeStyle::GptJ => {
                            let (x1, x2) = (src[2 * i], src[2 * i + 1]);
                            head[2 * i] = x1 * cv - x2 * sv;
                            head[2 * i + 1] = x2 * cv + x1 * sv;
                        }
                    }
                }
            }
        }
    }
    out
}

/// `Op::RopeBackward` over the same BHSD chain the forward test uses.
fn compiled_rope_backward(dy: &[f32], c: Case, device: Device) -> Vec<f32> {
    let (cos_data, sin_data) = nn::rope_tables(MAX_POS, c.head_dim, THETA);
    let half = c.head_dim / 2;
    let hidden = c.heads * c.head_dim;

    let built = ModelFlow::new("rope_bwd")
        .input("dy", Shape::new(&[c.batch, c.seq, hidden], DType::F32))
        .rope_tables(RopeTablesStage::param(MAX_POS, half, cos_data, sin_data))
        .plugin_named("rope_bwd", move |emit, hidden_val| {
            let dy_id = hidden_val.expect("hidden").hir_id();
            let cos = emit.state.rope_cos.expect("cos");
            let sin = emit.state.rope_sin.expect("sin");
            let mut g = HirMut::new(emit.hir());
            let g = &mut g;
            let x4 = g.reshape_(
                dy_id,
                vec![
                    c.batch as i64,
                    c.seq as i64,
                    c.heads as i64,
                    c.head_dim as i64,
                ],
            );
            let bhsd = g.transpose_(x4, vec![0, 2, 1, 3]);
            let shape4 = g.shape(bhsd).clone();
            let grad = g.add_node(
                Op::RopeBackward {
                    head_dim: c.head_dim,
                    n_rot: c.n_rot,
                    style: c.style,
                },
                vec![bhsd, cos, sin],
                shape4,
            );
            let bshd = g.transpose_(grad, vec![0, 2, 1, 3]);
            let out = g.reshape_(
                bshd,
                vec![c.batch as i64, c.seq as i64, (c.heads * c.head_dim) as i64],
            );
            Ok(Some(emit.wrap(
                out,
                Shape::new(&[c.batch, c.seq, c.heads * c.head_dim], DType::F32),
            )))
        })
        .output("y")
        .build(&mut WeightLoaderSource(&mut WeightMap::from_tensors(
            HashMap::new(),
        )))
        .expect("build rope_bwd");

    let mut compiled = compile_built(built, device).expect("compile");
    compiled.run(&[("dy", dy)]).swap_remove(0)
}
