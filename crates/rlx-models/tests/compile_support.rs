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

//! Shared tier-1 compile helpers for integration tests.
//!
//! Prefer [`compile_qwen35_prefill`] / [`compile_qwen35_decode`] (and the
//! SAM/Qwen3/encoder variants) over [`compile_legacy`], which uses default
//! [`CompileOptions`] only. Production runners build options via
//! [`rlx_models::flow_bridge::compile_options_from_profile`].

#![allow(dead_code)]

use rlx_flow::CompileProfile;
use rlx_ir::Graph;
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};
use std::collections::HashMap;

pub fn attach_params(compiled: &mut CompiledGraph, params: &HashMap<String, Vec<f32>>) {
    for (name, data) in params {
        compiled.set_param(name, data.as_slice());
    }
}

pub fn compile_legacy(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    let mut compiled = Session::new(device).compile_with(graph, &CompileOptions::new());
    attach_params(&mut compiled, &params);
    compiled
}

pub fn compile_with_options(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
    opts: &CompileOptions,
) -> CompiledGraph {
    let mut compiled = Session::new(device).compile_with(graph, opts);
    attach_params(&mut compiled, &params);
    compiled
}

pub fn compile_with_profile(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
    profile: &CompileProfile,
) -> CompiledGraph {
    let mut compiled = rlx_models::flow_bridge::compile_graph_with_profile(device, graph, profile)
        .expect("compile_graph_with_profile");
    attach_params(&mut compiled, &params);
    compiled
}

pub fn compile_sam(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::sam_encoder())
}

pub fn compile_encoder(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::encoder())
}

pub fn compile_qwen35_prefill(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::qwen35_prefill())
}

pub fn compile_qwen35_decode(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::qwen35_decode())
}

pub fn compile_qwen3_prefill(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::qwen3_prefill())
}

pub fn compile_llama32_prefill(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::llama32_prefill())
}

pub fn compile_llama32_decode(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::llama32_decode())
}

pub fn compile_llada2(
    device: Device,
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
) -> CompiledGraph {
    compile_with_profile(device, graph, params, &CompileProfile::llada2_diffusion())
}

/// Largest elementwise difference, relative to the reference's magnitude.
pub fn max_rel_diff(want: &[f32], got: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    want.iter()
        .zip(got)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
        / scale
}

/// A backend's output must match the CPU reference, not merely be finite.
///
/// "All values are finite" is the weakest useful property, and it is what let a
/// rank-4 RoPE lowering that zeroed almost every element, a MoE router that read
/// another token's probability, and a host readback that walked a strided buffer
/// linearly all sit in a green suite — every one of them produced finite,
/// in-range, plausible numbers. Comparing against CPU costs one extra run of a
/// tiny graph and catches all three.
pub fn assert_matches_cpu(backend: &str, cpu: &[f32], dev: &[f32], tol: f32) {
    assert!(
        cpu.iter().all(|v| v.is_finite()),
        "CPU reference itself is not finite"
    );
    assert!(
        dev.iter().all(|v| v.is_finite()),
        "{backend} produced non-finite output"
    );
    assert_eq!(
        cpu.len(),
        dev.len(),
        "{backend} returned {} values, CPU returned {}",
        dev.len(),
        cpu.len()
    );
    let rel = max_rel_diff(cpu, dev);
    // Show a few values on failure: "diverges by 1.000000" is the signature of
    // an all-zero result, which reads very differently from a real numeric
    // drift and points at a different class of bug.
    assert!(
        rel.is_finite() && rel < tol,
        "{backend} diverges from CPU by {rel:.6} (tolerance {tol})\n  \
         cpu[..8] = {:?}\n  {backend}[..8] = {:?}\n  \
         {backend} all-zero: {}",
        &cpu[..cpu.len().min(8)],
        &dev[..dev.len().min(8)],
        dev.iter().all(|v| *v == 0.0),
    );
}
