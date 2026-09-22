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

//! A tensor's value must not depend on who consumes it.
//!
//! Slices and transposes leave a *view*: same buffer, different strides. A
//! backend that reads such a view back to the host has to materialize it first.
//! rlx-mlx did that, but tested `flags().row_contiguous` **before** evaluating
//! the array — and on a lazily built graph those flags describe nothing yet. So
//! the check passed, the `memcpy` walked the base buffer linearly, and a sliced
//! column came back as the first N elements of the parent.
//!
//! Nothing downstream could notice: the MoE router's expert ids stayed in
//! range, so every token was simply routed to the wrong expert and the model
//! produced fluent nonsense. See `materialize_row_contiguous` in
//! `rlx-mlx-sys/cpp/rlx_mlx_shim.cpp`.
//!
//! These tests pin the invariant on every backend, for each way of producing a
//! view, by reading the same view through two different consumers.

use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Op, QuantScheme, Shape};
use rlx_unlimited_ocr::lm_precision::f32_to_q8_0_bytes;
use std::collections::HashMap;

use rlx_core::backend_matrix::{Failures, available_devices};

const ROWS: usize = 5;
const COLS: usize = 7;
/// Expert-slab geometry for the host-lowered consumer (multiple of 32 for GGUF).
const KD: usize = 32;
const NC: usize = 32;

/// How the view under test is produced.
#[derive(Clone, Copy, Debug)]
enum View {
    /// Column `c` of `[ROWS, COLS]` — stride `COLS`, offset `c`. The MoE
    /// router's `narrow(topk, 1, k, 1)`.
    Column(usize),
    /// Row `r` — already contiguous, so this is the control.
    Row(usize),
    /// Column `c` of the transpose, i.e. row `c` read with a stride.
    TransposedRow(usize),
    /// Every other column, via two narrows.
    DoubleNarrow,
}

impl View {
    /// Emit the view and reshape it to `[ROWS]` (or `[COLS]` for a row).
    fn emit(self, g: &mut HirMut, src: HirNodeId) -> (HirNodeId, usize) {
        match self {
            View::Column(c) => {
                let col = g.narrow_(src, 1, c, 1);
                (g.reshape_(col, vec![ROWS as i64]), ROWS)
            }
            View::Row(r) => {
                let row = g.narrow_(src, 0, r, 1);
                (g.reshape_(row, vec![COLS as i64]), COLS)
            }
            View::TransposedRow(c) => {
                let t = g.transpose_(src, vec![1, 0]);
                let row = g.narrow_(t, 0, c, 1);
                (g.reshape_(row, vec![ROWS as i64]), ROWS)
            }
            View::DoubleNarrow => {
                let a = g.narrow_(src, 1, 1, 4);
                let b = g.narrow_(a, 1, 2, 1);
                (g.reshape_(b, vec![ROWS as i64]), ROWS)
            }
        }
    }

    /// Element count the view yields.
    fn len(self) -> usize {
        match self {
            View::Row(_) => COLS,
            _ => ROWS,
        }
    }

    /// Host oracle over a row-major `[ROWS, COLS]` buffer.
    fn host(self, src: &[f32]) -> Vec<f32> {
        match self {
            View::Column(c) => (0..ROWS).map(|r| src[r * COLS + c]).collect(),
            View::Row(r) => (0..COLS).map(|c| src[r * COLS + c]).collect(),
            View::TransposedRow(c) => (0..ROWS).map(|r| src[r * COLS + c]).collect(),
            // narrow(1,1,4) keeps cols 1..5; narrow(1,2,1) of that is col 3.
            View::DoubleNarrow => (0..ROWS).map(|r| src[r * COLS + 3]).collect(),
        }
    }
}

const VIEWS: &[View] = &[
    View::Column(0),
    View::Column(3),
    View::Column(COLS - 1),
    View::Row(2),
    View::TransposedRow(4),
    View::DoubleNarrow,
];

fn source() -> Vec<f32> {
    // Distinct values so a wrong stride is unambiguous.
    (0..ROWS * COLS).map(|i| i as f32).collect()
}

/// Graph returning the view directly as the output.
fn build_direct(view: View) -> BuiltModel {
    ModelFlow::new("view_direct")
        .input("src", Shape::new(&[ROWS, COLS], DType::F32))
        .plugin_named("v", move |emit, hidden| {
            let src = hidden.expect("hidden").hir_id();
            let mut g = HirMut::new(emit.hir());
            let (out, len) = view.emit(&mut g, src);
            Ok(Some(emit.wrap(out, Shape::new(&[len], DType::F32))))
        })
        .output("y")
        .build(&mut WeightLoaderSource(&mut WeightMap::from_tensors(
            HashMap::new(),
        )))
        .expect("build view_direct")
}

/// Graph routing the view through a host-lowered consumer.
///
/// `DequantGroupedMatMul` reads its index operand on the host, which is exactly
/// the path that mis-read a strided slice. Expert `e` is a constant slab of
/// value `e + 1` and every `x` row sums to 1, so the output *is* the index the
/// consumer saw — the view's value as observed by a second, independent reader.
fn build_via_consumer(view: View, bytes: Vec<u8>) -> BuiltModel {
    let len = view.len();
    ModelFlow::new("view_consumer")
        .input("src", Shape::new(&[ROWS, COLS], DType::F32))
        .input("x", Shape::new(&[len, KD], DType::F32))
        .plugin_named("v", move |emit, hidden| {
            let src = hidden.expect("hidden").hir_id();
            let x = emit.state.inputs["x"].0;
            let w = emit.hir().param("w", Shape::new(&[bytes.len()], DType::U8));
            emit.state
                .typed_params
                .push(("w".to_string(), bytes.clone(), DType::U8));
            let mut g = HirMut::new(emit.hir());
            let (idx, len) = view.emit(&mut g, src);
            let y = g.add_node(
                Op::DequantGroupedMatMul {
                    scheme: QuantScheme::GgufQ8_0,
                },
                vec![x, w, idx],
                Shape::new(&[len, NC], DType::F32),
            );
            let col = g.narrow_(y, 1, 0, 1);
            let out = g.reshape_(col, vec![len as i64]);
            Ok(Some(emit.wrap(out, Shape::new(&[len], DType::F32))))
        })
        .output("y")
        .build(&mut WeightLoaderSource(&mut WeightMap::from_tensors(
            HashMap::new(),
        )))
        .expect("build view_consumer")
}

/// The same view, read by a host-lowered consumer, must hold the same values.
/// This is the pin for the rlx-mlx unevaluated-flags bug.
#[test]
fn views_read_back_correctly_inside_a_host_lowered_consumer() {
    let src = source();
    // Source values are 0..ROWS*COLS, so they double as expert ids.
    let n_experts = ROWS * COLS;
    let mut bank = Vec::with_capacity(n_experts * NC * KD);
    for e in 0..n_experts {
        bank.extend(std::iter::repeat_n((e + 1) as f32, NC * KD));
    }
    let bytes = f32_to_q8_0_bytes(&bank).expect("q8_0 encode");

    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        for &view in VIEWS {
            let len = view.len();
            let x: Vec<f32> = std::iter::repeat_n(1.0 / KD as f32, len * KD).collect();
            let got: Vec<f32> = compile_built(build_via_consumer(view, bytes.clone()), device)
                .expect("compile")
                .run(&[("src", &src), ("x", &x)])
                .swap_remove(0)
                .iter()
                .map(|v| v.round() - 1.0)
                .collect();
            let want = view.host(&src);
            if got != want {
                fails.push(
                    name,
                    format!("{view:?}: consumer saw {got:?}, view holds {want:?}"),
                );
            }
        }
    }
    fails.assert_empty("view read back inside a host-lowered consumer");
}

/// The view read straight out of the graph must match the host.
#[test]
fn views_read_back_correctly_as_graph_outputs() {
    let src = source();
    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        for &view in VIEWS {
            let want = view.host(&src);
            let got = compile_built(build_direct(view), device)
                .expect("compile")
                .run(&[("src", &src)])
                .swap_remove(0);
            if got != want {
                fails.push(name, format!("{view:?}: got {got:?}, want {want:?}"));
            }
        }
    }
    fails.assert_empty("view read back as a graph output");
}
