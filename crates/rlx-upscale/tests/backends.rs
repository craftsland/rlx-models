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

//! Cross-backend agreement.
//!
//! Every backend compiled into this build runs the same graph on the same input
//! and must agree with CPU. Only backends whose feature is enabled *and* whose
//! hardware is present are exercised; the rest are reported as skipped rather
//! than silently passing.

use rlx_core::weight_map::WeightMap;
use rlx_runtime::{Device, Session};
use rlx_upscale::config::*;
use rlx_upscale::graph;

// The pushes are `cfg`-gated, so neither the `mut` nor the empty-then-push
// shape is avoidable: which arms exist depends on the enabled features.
#[allow(unused_mut, clippy::vec_init_then_push)]
fn enabled_devices() -> Vec<(&'static str, Device)> {
    let mut v = vec![];
    #[cfg(feature = "metal")]
    v.push(("metal", Device::Metal));
    #[cfg(feature = "mlx")]
    v.push(("mlx", Device::Mlx));
    #[cfg(feature = "cuda")]
    v.push(("cuda", Device::Cuda));
    #[cfg(feature = "rocm")]
    v.push(("rocm", Device::Rocm));
    #[cfg(feature = "gpu")]
    v.push(("wgpu", Device::Gpu));
    #[cfg(feature = "vulkan")]
    v.push(("vulkan", Device::Vulkan));
    #[cfg(feature = "coreml")]
    v.push(("ane", Device::Ane));
    v
}

fn run(cfg: &ModelConfig, tile: usize, device: Device) -> Vec<f32> {
    let built = graph::build_autofill(
        cfg,
        WeightMap::from_tensors(Default::default()),
        tile,
        tile,
        7,
    )
    .unwrap();
    let opts = rlx_core::flow_bridge::compile_options_for_profile(
        &rlx_flow::CompileProfile::encoder(),
        device,
    );
    let mut compiled = Session::new(device).compile_with(built.graph, &opts);
    rlx_core::flow_util::attach_built_params(&mut compiled, built.params, &[]);
    let n = cfg.in_ch * tile * tile;
    let input: Vec<f32> = (0..n).map(|i| ((i * 31) % 97) as f32 / 97.0).collect();
    compiled.run(&[("image", &input)]).remove(0)
}

/// The canonical per-architecture set, shared with the op-coverage matrix so
/// the two cannot drift — see [`rlx_upscale::sample`].
fn cases() -> Vec<(&'static str, ModelConfig, usize)> {
    rlx_upscale::sample::representative_configs()
}

#[test]
fn every_enabled_backend_agrees_with_cpu() {
    let devices = enabled_devices();
    if devices.is_empty() {
        eprintln!("no GPU backend features enabled; CPU-only build");
        return;
    }
    let mut checked = 0usize;
    for (name, cfg, tile) in cases() {
        let reference = run(&cfg, tile, Device::Cpu);
        for (label, device) in &devices {
            if !rlx_runtime::device_ext::is_available(*device) {
                eprintln!("{label}: not available on this machine, skipping");
                continue;
            }
            let got = run(&cfg, tile, *device);
            assert_eq!(
                got.len(),
                reference.len(),
                "{name} on {label}: wrong length"
            );
            let worst = got
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let scale = reference.iter().map(|v| v.abs()).fold(1e-3f32, f32::max);
            assert!(
                worst / scale < 2e-3,
                "{name} on {label}: worst absolute difference {worst:.3e} \
                 against a peak of {scale:.3e}"
            );
            eprintln!("{name:<20} {label:<8} max|Δ| {worst:.2e}");
            checked += 1;
        }
    }
    if checked == 0 {
        eprintln!("no enabled backend was actually available");
    }
}
