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

//! Numeric check for `Op::GroupedMatMul`, the routed-expert projection.
//!
//! [`expert_pack::pack_experts_in_map`] stacks each expert already transposed,
//! so the bank is `[E, K, N]` and row `r` of the output is
//! `input[r] @ bank[expert_idx[r]]`.
//!
//! `backend_quick_check::tiny_moe_cpu` runs this op but only asserts the
//! results are finite, so a wrong expert selection or a transposed bank passes
//! it while quietly degrading every MoE layer.

use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_flow::ModelFlow;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Op, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

use rlx_core::backend_matrix::{Failures, available_devices, max_abs_diff};

const ROWS: usize = 5;
const E: usize = 4;
const K: usize = 8;
const N: usize = 6;

/// `[E, K, N]` bank, distinct per expert so a wrong pick is visible.
fn bank() -> Vec<f32> {
    (0..E * K * N)
        .map(|i| {
            let e = i / (K * N);
            ((i % (K * N)) as f32 * 0.01) + e as f32
        })
        .collect()
}

fn input() -> Vec<f32> {
    (0..ROWS * K).map(|i| (i as f32 * 0.07).sin()).collect()
}

/// `out[r] = input[r] @ bank[expert[r]]`, computed on the host.
fn host_grouped(input: &[f32], bank: &[f32], expert: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; ROWS * N];
    for r in 0..ROWS {
        let e = expert[r];
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..K {
                acc += input[r * K + k] * bank[e * K * N + k * N + n];
            }
            out[r * N + n] = acc;
        }
    }
    out
}

fn compiled_grouped(
    input: &[f32],
    bank_data: &[f32],
    expert: &[usize],
    device: Device,
) -> Vec<f32> {
    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    tensors.insert("bank".into(), (bank_data.to_vec(), vec![E, K, N]));
    let mut wm = WeightMap::from_tensors(tensors);
    let idx: Vec<f32> = expert.iter().map(|&e| e as f32).collect();

    let built = ModelFlow::new("grouped_repro")
        .input("x", Shape::new(&[ROWS, K], DType::F32))
        .input("expert", Shape::new(&[ROWS], DType::F32))
        .plugin_named("grouped", move |emit, hidden| {
            let x = hidden.expect("hidden").hir_id();
            let expert_id = emit.state.inputs["expert"].0;
            let w = emit.load_param("bank", false).expect("bank param");
            let mut g = HirMut::new(emit.hir());
            let out = g.add_node(
                Op::GroupedMatMul,
                vec![x, w, expert_id],
                Shape::new(&[ROWS, N], DType::F32),
            );
            Ok(Some(emit.wrap(out, Shape::new(&[ROWS, N], DType::F32))))
        })
        .output("y")
        .build(&mut WeightLoaderSource(&mut wm))
        .expect("build grouped repro");

    let mut compiled = compile_built(built, device).expect("compile");
    compiled
        .run(&[("x", input), ("expert", &idx)])
        .swap_remove(0)
}

#[test]
fn grouped_matmul_matches_host_for_k_major_bank() {
    let bank = bank();
    let input = input();
    let expert = [0usize, 3, 1, 2, 0];
    let want = host_grouped(&input, &bank, &expert);

    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        let got = compiled_grouped(&input, &bank, &expert, device);
        if got.len() != want.len() {
            fails.push(name, format!("length {} != {}", got.len(), want.len()));
            continue;
        }
        let worst = max_abs_diff(&want, &got);
        if worst >= 1e-3 {
            fails.push(name, format!("differs by {worst:.5}; got {got:?}"));
        }
    }
    fails.assert_empty("GroupedMatMul");
}

/// Every row routed to the same expert must equal a plain matmul against that
/// expert's slice — the simplest possible statement of the op's contract.
#[test]
fn grouped_matmul_with_one_expert_is_a_plain_matmul() {
    let bank = bank();
    let input = input();
    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        for e in 0..E {
            let expert = vec![e; ROWS];
            let want = host_grouped(&input, &bank, &expert);
            let got = compiled_grouped(&input, &bank, &expert, device);
            let worst = max_abs_diff(&want, &got);
            if worst >= 1e-3 {
                fails.push(name, format!("expert {e}: differs by {worst:.5}"));
            }
        }
    }
    fails.assert_empty("GroupedMatMul (single expert)");
}
