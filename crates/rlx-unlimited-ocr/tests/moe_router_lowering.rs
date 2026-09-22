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

//! Numeric check for the compiled MoE router chain: `softmax -> TopK ->
//! gather`, exactly as `lm_graph::build_moe_ffn` emits it.
//!
//! The reference gate (`MoEGate.forward`, `scoring_func="softmax"`,
//! `topk_method="greedy"`, `norm_topk_prob=false`) takes the top-k of the
//! softmax over *all* experts and weights each selected expert by its raw
//! probability. `nn::topk_softmax` implements that on the host and the eager
//! decoder reproduces the published logits with it, so it is the oracle here.

use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_flow::ModelFlow;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Op, Shape};
use rlx_runtime::Device;
use rlx_unlimited_ocr::nn;
use std::collections::HashMap;

use rlx_core::backend_matrix::{Failures, available_devices};

const ROWS: usize = 7;
const EXPERTS: usize = 64;
const TOP_K: usize = 6;

fn logits() -> Vec<f32> {
    (0..ROWS * EXPERTS)
        .map(|i| ((i as f32 * 0.37).sin() * 2.0) + ((i / EXPERTS) as f32 * 0.1))
        .collect()
}

/// Compiled `softmax -> TopK -> gather`; returns `(indices, weights)`.
fn compiled_router(logits: &[f32], device: Device) -> (Vec<f32>, Vec<f32>) {
    let built = ModelFlow::new("router_repro")
        .input("logits", Shape::new(&[ROWS, EXPERTS], DType::F32))
        .plugin_named("router", move |emit, hidden| {
            let l = hidden.expect("hidden").hir_id();
            let mut g = HirMut::new(emit.hir());
            let probs = g.sm(l, -1);
            let idx = g.add_node(
                Op::TopK { k: TOP_K },
                vec![probs],
                Shape::new(&[ROWS, TOP_K], DType::F32),
            );
            // `Op::GatherElements` (take_along_axis), NOT `gather_` — see the
            // comment in `build_moe_ffn`. ONNX Gather would return
            // `[rows, rows, k]` here.
            let w = g.add_node(
                Op::GatherElements { axis: 1 },
                vec![probs, idx],
                Shape::new(&[ROWS, TOP_K], DType::F32),
            );
            let flat = w;
            let cat = g.concat_(vec![idx, flat], 1);
            Ok(Some(
                emit.wrap(cat, Shape::new(&[ROWS, 2 * TOP_K], DType::F32)),
            ))
        })
        .output("y")
        .build(&mut WeightLoaderSource(&mut WeightMap::from_tensors(
            HashMap::new(),
        )))
        .expect("build router repro");

    let mut compiled = compile_built(built, device).expect("compile");
    let out = compiled.run(&[("logits", logits)]).swap_remove(0);
    let mut idx = Vec::with_capacity(ROWS * TOP_K);
    let mut wts = Vec::with_capacity(ROWS * TOP_K);
    for r in 0..ROWS {
        idx.extend_from_slice(&out[r * 2 * TOP_K..r * 2 * TOP_K + TOP_K]);
        wts.extend_from_slice(&out[r * 2 * TOP_K + TOP_K..(r + 1) * 2 * TOP_K]);
    }
    (idx, wts)
}

/// `nn::topk_softmax` — the host oracle the eager decoder uses.
fn host_router(logits: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut idx = Vec::new();
    let mut wts = Vec::new();
    for r in 0..ROWS {
        let top = nn::topk_softmax(&logits[r * EXPERTS..(r + 1) * EXPERTS], TOP_K);
        for (i, w) in top {
            idx.push(i as f32);
            wts.push(w);
        }
    }
    (idx, wts)
}

/// The op must return expert *indices*, not the probabilities themselves — a
/// tensor of probabilities is all < 1 and would route every token to expert 0.
#[test]
fn topk_returns_expert_indices() {
    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        let (idx, _) = compiled_router(&logits(), device);
        if idx.len() != ROWS * TOP_K {
            fails.push(name, format!("index count {}", idx.len()));
            continue;
        }
        let max = idx.iter().cloned().fold(f32::MIN, f32::max);
        if max < 1.0 {
            fails.push(
                name,
                format!("looks like probabilities, not indices (max {max})"),
            );
            continue;
        }
        if let Some(bad) = idx
            .iter()
            .find(|v| v.fract() != 0.0 || **v < 0.0 || (**v as usize) >= EXPERTS)
        {
            fails.push(name, format!("index {bad} out of range"));
        }
    }
    fails.assert_empty("TopK expert indices");
}

/// Selected expert sets must match the host oracle (order-insensitive: the
/// reference uses `torch.topk(..., sorted=False)` and sums over k).
#[test]
fn compiled_router_selects_the_same_experts() {
    let logits = logits();
    let (hidx, _) = host_router(&logits);
    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        let (cidx, _) = compiled_router(&logits, device);
        for r in 0..ROWS {
            let mut c: Vec<u32> = cidx[r * TOP_K..(r + 1) * TOP_K]
                .iter()
                .map(|v| *v as u32)
                .collect();
            let mut h: Vec<u32> = hidx[r * TOP_K..(r + 1) * TOP_K]
                .iter()
                .map(|v| *v as u32)
                .collect();
            c.sort_unstable();
            h.sort_unstable();
            if c != h {
                fails.push(name, format!("row {r}: picked {c:?}, host picked {h:?}"));
            }
        }
    }
    fails.assert_empty("MoE expert selection");
}

/// Each selected expert's weight must be its raw softmax probability, matched
/// to the expert it was gathered for, and the k weights must NOT sum to 1
/// (`norm_topk_prob=false`).
#[test]
fn compiled_router_weights_are_raw_softmax_probs() {
    let logits = logits();
    let (hidx, hw) = host_router(&logits);
    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        let (cidx, cw) = compiled_router(&logits, device);
        for r in 0..ROWS {
            for k in 0..TOP_K {
                let e = cidx[r * TOP_K + k] as u32;
                let Some(pos) = (0..TOP_K).find(|&j| hidx[r * TOP_K + j] as u32 == e) else {
                    fails.push(name, format!("row {r}: picked expert {e}, host did not"));
                    continue;
                };
                let (got, want) = (cw[r * TOP_K + k], hw[r * TOP_K + pos]);
                if (got - want).abs() >= 1e-4 {
                    fails.push(
                        name,
                        format!("row {r} expert {e}: weight {got} != softmax prob {want}"),
                    );
                }
            }
            let sum: f32 = cw[r * TOP_K..(r + 1) * TOP_K].iter().sum();
            if sum >= 0.999 {
                fails.push(
                    name,
                    format!("row {r}: weights sum to {sum}; must NOT be normalized"),
                );
            }
        }
    }
    fails.assert_empty("MoE router weights");
}

/// `build_moe_ffn` slices one expert column per k with
/// `narrow(idx, 1, ki, 1)` then `reshape([rows])`. Taken together, the k columns
/// must cover each row's selected experts exactly once.
///
/// Deliberately order-agnostic: `Op::TopK`'s ordering is not part of its
/// contract here (CPU/Metal return descending-by-probability, MLX ascending by
/// index) and `build_moe_ffn` sums over k, so only the *set* matters — as long
/// as the weight for column k belongs to the expert in column k, which
/// [`compiled_router_weights_are_raw_softmax_probs`] covers.
#[test]
fn narrow_columns_cover_each_rows_experts_exactly_once() {
    let logits = logits();
    let (hidx, _) = host_router(&logits);

    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        let mut cols: Vec<Vec<f32>> = Vec::with_capacity(TOP_K);
        for ki in 0..TOP_K {
            let built = ModelFlow::new("narrow_col")
                .input("logits", Shape::new(&[ROWS, EXPERTS], DType::F32))
                .plugin_named("col", move |emit, hidden| {
                    let l = hidden.expect("hidden").hir_id();
                    let mut g = HirMut::new(emit.hir());
                    let probs = g.sm(l, -1);
                    let idx = g.add_node(
                        Op::TopK { k: TOP_K },
                        vec![probs],
                        Shape::new(&[ROWS, TOP_K], DType::F32),
                    );
                    let col = g.narrow_(idx, 1, ki, 1);
                    let out = g.reshape_(col, vec![ROWS as i64]);
                    Ok(Some(emit.wrap(out, Shape::new(&[ROWS], DType::F32))))
                })
                .output("y")
                .build(&mut WeightLoaderSource(&mut WeightMap::from_tensors(
                    HashMap::new(),
                )))
                .expect("build narrow_col");
            let mut compiled = compile_built(built, device).expect("compile");
            cols.push(compiled.run(&[("logits", &logits)]).swap_remove(0));
        }

        for r in 0..ROWS {
            let mut got: Vec<u32> = (0..TOP_K).map(|k| cols[k][r] as u32).collect();
            let mut want: Vec<u32> = hidx[r * TOP_K..(r + 1) * TOP_K]
                .iter()
                .map(|v| *v as u32)
                .collect();
            got.sort_unstable();
            want.sort_unstable();
            if got != want {
                fails.push(
                    name,
                    format!("row {r}: columns hold {got:?}, expected the set {want:?}"),
                );
            }
        }
    }
    fails.assert_empty("narrow over the TopK index");
}
