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

//! Numeric parity for the packed-quant projections, on every backend.
//!
//! `--lm-precision q8_0|q4_0` keeps GGUF blocks in the IR and multiplies via
//! `Op::DequantMatMul` / `Op::DequantGroupedMatMul` instead of dequantizing at
//! load time. Every backend is handed the *same bytes*, so the comparison here
//! is not about quantization error: the oracle dequantizes those bytes on the
//! host and does a plain matmul, and each backend must land on the same answer.
//!
//! This is the axis `backend_quick_check`'s Q8/Q4 tests leave open — they
//! assert the packed graph compiles and returns finite numbers, not that it
//! returns the right ones.

use rlx_core::flow_bridge::WeightLoaderSource;
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_flow::ModelFlow;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Op, QuantScheme, Shape};
use rlx_runtime::Device;
use rlx_unlimited_ocr::lm_precision::{
    f32_to_q4_0_bytes, f32_to_q8_0_bytes, q4_0_bytes_to_f32, q8_0_bytes_to_f32,
};
use std::collections::HashMap;

use rlx_core::backend_matrix::{Failures, available_devices, max_abs_diff};
use rlx_unlimited_ocr::lm_precision::device_supports_packed_quant;

/// Backends that are *allowed* to run a packed-GGUF graph.
///
/// Gated on [`device_supports_packed_quant`] rather than a literal list, so a
/// backend clamped off for correctness is skipped here automatically — and
/// picked back up the moment the clamp is lifted.
fn packed_quant_devices() -> Vec<(&'static str, rlx_runtime::Device)> {
    available_devices()
        .into_iter()
        .filter(|(name, d)| {
            let ok = device_supports_packed_quant(*d);
            if !ok {
                eprintln!("[parity] skip {name}: packed quant is clamped off for this backend");
            }
            ok
        })
        .collect()
}

fn input(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|i| (i as f32 * 0.031).sin() * 0.8)
        .collect()
}

fn weight(n: usize, k: usize, seed: f32) -> Vec<f32> {
    (0..n * k)
        .map(|i| ((i as f32 * 0.017 + seed).cos()) * 0.35)
        .collect()
}

/// `y = x @ w^T` with `w` row-major `[n, k]`.
fn host_matmul_t(x: &[f32], w: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * n];
    for r in 0..rows {
        for j in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[r * k + i] * w[j * k + i];
            }
            out[r * n + j] = acc;
        }
    }
    out
}

struct Packed {
    bytes: Vec<u8>,
    scheme: QuantScheme,
    /// What those bytes dequantize to — the oracle's weight.
    dequant: Vec<f32>,
}

fn pack(w: &[f32], scheme: QuantScheme) -> Packed {
    let (bytes, dequant) = match scheme {
        QuantScheme::GgufQ8_0 => {
            let b = f32_to_q8_0_bytes(w).expect("q8_0 encode");
            let d = q8_0_bytes_to_f32(&b, w.len()).expect("q8_0 decode");
            (b, d)
        }
        QuantScheme::GgufQ4_0 => {
            let b = f32_to_q4_0_bytes(w).expect("q4_0 encode");
            let d = q4_0_bytes_to_f32(&b, w.len()).expect("q4_0 decode");
            (b, d)
        }
        other => panic!("unsupported scheme {other:?}"),
    };
    Packed {
        bytes,
        scheme,
        dequant,
    }
}

/// One `Op::DequantMatMul` against a packed `[N, K]` weight.
fn compiled_dequant_matmul(
    x: &[f32],
    p: &Packed,
    rows: usize,
    k: usize,
    n: usize,
    device: Device,
) -> Vec<f32> {
    let scheme = p.scheme;
    let bytes = p.bytes.clone();
    let built = ModelFlow::new("dqmm")
        .input("x", Shape::new(&[rows, k], DType::F32))
        .plugin_named("dqmm", move |emit, hidden| {
            let x = hidden.expect("hidden").hir_id();
            let w = emit.hir().param("w", Shape::new(&[bytes.len()], DType::U8));
            emit.state
                .typed_params
                .push(("w".to_string(), bytes.clone(), DType::U8));
            let mut g = HirMut::new(emit.hir());
            let out = g.add_node(
                Op::DequantMatMul { scheme },
                vec![x, w],
                Shape::new(&[rows, n], DType::F32),
            );
            Ok(Some(emit.wrap(out, Shape::new(&[rows, n], DType::F32))))
        })
        .output("y")
        .build(&mut WeightLoaderSource(&mut WeightMap::from_tensors(
            HashMap::new(),
        )))
        .expect("build dqmm");

    let mut compiled = compile_built(built, device).expect("compile");
    compiled.run(&[("x", x)]).swap_remove(0)
}

/// One `Op::DequantGroupedMatMul` against a packed `[E, N, K]` expert bank.
#[allow(clippy::too_many_arguments)]
fn compiled_dequant_grouped(
    x: &[f32],
    p: &Packed,
    expert: &[usize],
    rows: usize,
    k: usize,
    n: usize,
    device: Device,
) -> Vec<f32> {
    let scheme = p.scheme;
    let bytes = p.bytes.clone();
    // One-hot-ish scores whose argmax is the wanted expert.
    let e_count = p.dequant.len() / (n * k);
    let mut sel = vec![0f32; rows * e_count];
    for (r, &e) in expert.iter().enumerate() {
        sel[r * e_count + e] = 1.0;
    }
    let built = ModelFlow::new("dqgmm")
        .input("x", Shape::new(&[rows, k], DType::F32))
        .input("expert_sel", Shape::new(&[rows, e_count], DType::F32))
        .plugin_named("dqgmm", move |emit, hidden| {
            let x = hidden.expect("hidden").hir_id();
            // Derive the expert index IN-GRAPH from a TopK, exactly as
            // `build_moe_ffn` does. A host-supplied index tensor can take a
            // different backend path (no device-side dependency, no sync), so
            // feeding one would not exercise what the model actually runs.
            let expert_id = {
                let sel = emit.state.inputs["expert_sel"].0;
                let mut g = HirMut::new(emit.hir());
                let idx = g.add_node(
                    Op::TopK { k: 1 },
                    vec![sel],
                    Shape::new(&[rows, 1], DType::F32),
                );
                g.reshape_(idx, vec![rows as i64])
            };
            let w = emit.hir().param("w", Shape::new(&[bytes.len()], DType::U8));
            emit.state
                .typed_params
                .push(("w".to_string(), bytes.clone(), DType::U8));
            let mut g = HirMut::new(emit.hir());
            let out = g.add_node(
                Op::DequantGroupedMatMul { scheme },
                vec![x, w, expert_id],
                Shape::new(&[rows, n], DType::F32),
            );
            Ok(Some(emit.wrap(out, Shape::new(&[rows, n], DType::F32))))
        })
        .output("y")
        .build(&mut WeightLoaderSource(&mut WeightMap::from_tensors(
            HashMap::new(),
        )))
        .expect("build dqgmm");

    let mut compiled = compile_built(built, device).expect("compile");
    compiled
        .run(&[("x", x), ("expert_sel", &sel)])
        .swap_remove(0)
}

/// `(rows, k, n)` — the decoder's real projection geometry: attention
/// `1280x1280`, shared-expert `1280->1792` and its `1792->1280` return, and the
/// `129280`-wide LM head, at both decode (1 row) and prefill row counts.
const MATMUL_SHAPES: &[(usize, usize, usize)] = &[
    (4, 256, 128),
    (1, 1280, 1280),
    (128, 1280, 1280),
    (128, 1280, 1792),
    (128, 1792, 1280),
    (1, 1280, 129280),
];

fn check_matmul(scheme: QuantScheme, label: &str) {
    let mut fails = Failures::default();
    for &(rows, k, n) in MATMUL_SHAPES {
        let x = input(rows, k);
        let w = weight(n, k, 0.3);
        let p = pack(&w, scheme);
        // Oracle uses the *dequantized* bytes, so quantization error is excluded.
        let want = host_matmul_t(&x, &p.dequant, rows, k, n);
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);

        for (name, device) in packed_quant_devices() {
            let got = compiled_dequant_matmul(&x, &p, rows, k, n, device);
            if got.len() != want.len() {
                fails.push(
                    name,
                    format!("{rows}x{k}x{n}: length {} != {}", got.len(), want.len()),
                );
                continue;
            }
            let rel = max_abs_diff(&want, &got) / scale;
            if rel >= 1e-2 {
                fails.push(name, format!("{rows}x{k}x{n}: relative error {rel:.4}"));
            }
        }
    }
    fails.assert_empty(&format!("DequantMatMul {label}"));
}

/// Shapes worth covering: a small smoke shape, and the decoder's real MoE
/// expert geometry (`hidden = 1280`, `moe_intermediate = 896`, 64 experts) at
/// both a decode-like and a prefill-like row count. Backend fast paths switch
/// on `k_dim`, `n`, the expert count and whether every expert is a singleton,
/// so a single small shape proves very little.
const GROUPED_SHAPES: &[(usize, usize, usize, usize)] = &[
    // (rows, k, n, experts)
    (4, 256, 128, 3),
    (1, 1280, 896, 8),    // decode: one token, every expert a singleton
    (128, 1280, 896, 64), // prefill: many tokens over the full expert bank
    (128, 896, 1280, 64), // the down_proj direction
];

fn check_grouped(scheme: QuantScheme, label: &str) {
    let mut fails = Failures::default();
    for &(rows, k, n, e_count) in GROUPED_SHAPES {
        let x = input(rows, k);
        // `[E, N, K]` bank: expert e is rows `e*N..(e+1)*N`.
        let mut bank = Vec::with_capacity(e_count * n * k);
        for e in 0..e_count {
            bank.extend(weight(n, k, 0.3 + e as f32));
        }
        let p = pack(&bank, scheme);
        // Spread rows across experts deterministically.
        let expert: Vec<usize> = (0..rows).map(|r| (r * 7 + 1) % e_count).collect();

        let mut want = vec![0f32; rows * n];
        for (r, &e) in expert.iter().enumerate() {
            let slice = &p.dequant[e * n * k..(e + 1) * n * k];
            let row = host_matmul_t(&x[r * k..(r + 1) * k], slice, 1, k, n);
            want[r * n..(r + 1) * n].copy_from_slice(&row);
        }
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);

        for (name, device) in packed_quant_devices() {
            let got = compiled_dequant_grouped(&x, &p, &expert, rows, k, n, device);
            if got.len() != want.len() {
                fails.push(
                    name,
                    format!(
                        "{rows}x{k}x{n} E{e_count}: length {} != {}",
                        got.len(),
                        want.len()
                    ),
                );
                continue;
            }
            let rel = max_abs_diff(&want, &got) / scale;
            if rel >= 1e-2 {
                fails.push(
                    name,
                    format!("{rows}x{k}x{n} E{e_count}: relative error {rel:.4}"),
                );
            }
        }
    }
    fails.assert_empty(&format!("DequantGroupedMatMul {label}"));
}

#[test]
fn dequant_matmul_q8_0_matches_host() {
    check_matmul(QuantScheme::GgufQ8_0, "Q8_0");
}

#[test]
fn dequant_matmul_q4_0_matches_host() {
    check_matmul(QuantScheme::GgufQ4_0, "Q4_0");
}

#[test]
fn dequant_grouped_matmul_q8_0_matches_host() {
    check_grouped(QuantScheme::GgufQ8_0, "Q8_0");
}

#[test]
fn dequant_grouped_matmul_q4_0_matches_host() {
    check_grouped(QuantScheme::GgufQ4_0, "Q4_0");
}

/// The MoE layer as `build_moe_ffn` actually emits it: softmax -> TopK(k) ->
/// per-k `narrow` of the index column -> `DequantGroupedMatMul` -> weight by the
/// gathered probability -> accumulate. A single grouped matmul with a
/// host-supplied index does not exercise this.
fn check_moe_accumulate(scheme: QuantScheme, top_k: usize, label: &str) {
    const ROWS: usize = 64;
    const KDIM: usize = 1280;
    const NDIM: usize = 896;
    const NE: usize = 64;

    let x = input(ROWS, KDIM);
    let mut bank = Vec::with_capacity(NE * NDIM * KDIM);
    for e in 0..NE {
        bank.extend(weight(NDIM, KDIM, 0.3 + e as f32 * 0.11));
    }
    let p = pack(&bank, scheme);
    let logits: Vec<f32> = (0..ROWS * NE)
        .map(|i| ((i as f32 * 0.29).sin() * 2.0) + ((i / NE) as f32 * 0.05))
        .collect();

    // Host oracle: same routing rule, dequantized bytes.
    let mut want = vec![0f32; ROWS * NDIM];
    for r in 0..ROWS {
        let top = nn_topk(&logits[r * NE..(r + 1) * NE], top_k);
        for (e, prob) in top {
            let slab = &p.dequant[e * NDIM * KDIM..(e + 1) * NDIM * KDIM];
            let row = host_matmul_t(&x[r * KDIM..(r + 1) * KDIM], slab, 1, KDIM, NDIM);
            for (acc, v) in want[r * NDIM..(r + 1) * NDIM].iter_mut().zip(&row) {
                *acc += v * prob;
            }
        }
    }
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);

    let scheme_c = scheme;
    let mut fails = Failures::default();
    for (name, device) in packed_quant_devices() {
        let bytes = p.bytes.clone();
        let built = ModelFlow::new("moe_acc")
            .input("x", Shape::new(&[ROWS, KDIM], DType::F32))
            .input("logits", Shape::new(&[ROWS, NE], DType::F32))
            .plugin_named("moe", move |emit, hidden| {
                let x = hidden.expect("hidden").hir_id();
                let l = emit.state.inputs["logits"].0;
                let w = emit.hir().param("w", Shape::new(&[bytes.len()], DType::U8));
                emit.state
                    .typed_params
                    .push(("w".to_string(), bytes.clone(), DType::U8));
                let mut g = HirMut::new(emit.hir());
                let probs = g.sm(l, -1);
                let idx = g.add_node(
                    Op::TopK { k: top_k },
                    vec![probs],
                    Shape::new(&[ROWS, top_k], DType::F32),
                );
                let wts = g.add_node(
                    Op::GatherElements { axis: 1 },
                    vec![probs, idx],
                    Shape::new(&[ROWS, top_k], DType::F32),
                );
                let mut acc: Option<rlx_ir::HirNodeId> = None;
                for ki in 0..top_k {
                    let ecol = g.narrow_(idx, 1, ki, 1);
                    let eidx = g.reshape_(ecol, vec![ROWS as i64]);
                    let y = g.add_node(
                        Op::DequantGroupedMatMul { scheme: scheme_c },
                        vec![x, w, eidx],
                        Shape::new(&[ROWS, NDIM], DType::F32),
                    );
                    let pcol = g.narrow_(wts, 1, ki, 1);
                    let p2 = g.reshape_(pcol, vec![ROWS as i64, 1]);
                    let wy = g.mul(y, p2);
                    acc = Some(match acc {
                        None => wy,
                        Some(a) => g.add(a, wy),
                    });
                }
                let out = acc.expect("topk >= 1");
                Ok(Some(emit.wrap(out, Shape::new(&[ROWS, NDIM], DType::F32))))
            })
            .output("y")
            .build(&mut WeightLoaderSource(&mut WeightMap::from_tensors(
                HashMap::new(),
            )))
            .expect("build moe_acc");
        let mut compiled = compile_built(built, device).expect("compile");
        let got = compiled
            .run(&[("x", &x), ("logits", &logits)])
            .swap_remove(0);
        let rel = max_abs_diff(&want, &got) / scale;
        if rel >= 2e-2 {
            fails.push(name, format!("relative error {rel:.4}"));
        }
    }
    fails.assert_empty(&format!("MoE accumulate {label}"));
}

/// Top-k of softmax over all experts, raw probabilities (`norm_topk_prob=false`).
fn nn_topk(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    rlx_unlimited_ocr::nn::topk_softmax(logits, k)
}

#[test]
fn moe_accumulate_q8_0_matches_host() {
    // k=1 isolates "one grouped matmul, device index"; k=6 adds the
    // accumulate over several nodes sharing one weight param.
    check_moe_accumulate(QuantScheme::GgufQ8_0, 1, "Q8_0 k=1");
    check_moe_accumulate(QuantScheme::GgufQ8_0, 6, "Q8_0 k=6");
}

#[test]
fn moe_accumulate_q4_0_matches_host() {
    check_moe_accumulate(QuantScheme::GgufQ4_0, 1, "Q4_0 k=1");
    check_moe_accumulate(QuantScheme::GgufQ4_0, 6, "Q4_0 k=6");
}
