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

// basic test: tiny synthetic qwen35 graph on Metal (macOS only).

mod compile_support;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod parity {
    use rlx_models::qwen35::synth;
    use rlx_models::{Qwen35RunnerBuilder, build_qwen35_graph_sized};
    use rlx_runtime::Device;

    /// Build and run the tiny qwen35 graph on `device`, returning last-token logits.
    fn run(device: Device, enable_mtp_head: bool) -> Vec<f32> {
        let cfg = synth::tiny_cfg();
        let weights = synth::synth_weights(&cfg);
        let (graph, params, _packed) =
            build_qwen35_graph_sized(&cfg, weights, 1, 4, true, true, enable_mtp_head)
                .expect("build");
        let mut compiled = super::compile_support::compile_qwen35_prefill(device, graph, params);
        let ids = vec![1.0f32, 2.0, 3.0, 4.0];
        let outs = compiled.run(&[("input_ids", &ids), ("last_token_idx", &[3.0f32])]);
        outs[0].clone()
    }

    /// The metal result must equal CPU's, not merely be finite.
    ///
    /// This test used to assert only that the logits were finite — a bar that a
    /// lowering emitting almost all zeros still clears.
    #[test]
    fn qwen35_tiny_graph_matches_cpu_on_metal() {
        let _ = Qwen35RunnerBuilder::default();
        // Sweep the MTP draft head: it adds a whole second head to the graph,
        // and building it was broken outright until recently, so "the main
        // logits still match" is a different claim with it on than with it off.
        for mtp in [false, true] {
            let cpu = run(Device::Cpu, mtp);
            let dev = run(Device::Metal, mtp);
            super::compile_support::assert_matches_cpu(
                &format!("metal (mtp_head={mtp})"),
                &cpu,
                &dev,
                2e-3,
            );
        }
    }
}
