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

//! Metal quick check: dynamic prefill + decode must match CPU.

mod compile_support;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod parity {
    use rlx_models::Qwen35RunnerBuilder;
    use rlx_models::qwen35::synth;
    use rlx_runtime::Device;

    /// Dynamic prefill + decode on `device`; returns both steps' logits so a
    /// divergence in either one is caught.
    fn run(device: Device) -> Vec<f32> {
        let cfg = synth::tiny_cfg();
        let weights = synth::synth_weights(&cfg);
        let mut runner = Qwen35RunnerBuilder::default()
            .inline_weights(cfg.clone(), weights)
            .device(device)
            .max_seq(8)
            .dynamic_prefill(true)
            .dynamic_decode(true)
            .bucketed_decode(false)
            .last_logits_only(true)
            .build()
            .expect("dynamic runner");

        let mut out = runner
            .prefill_get_last_logits(&[1, 2, 3])
            .expect("dynamic prefill");
        assert_eq!(out.len(), cfg.vocab_size, "prefill logits width");
        let step = runner.decode_get_logits(4).expect("dynamic decode");
        assert_eq!(step.len(), cfg.vocab_size, "decode logits width");
        out.extend_from_slice(&step);
        out
    }

    /// The metal result must equal CPU's, not merely be finite.
    ///
    /// This previously ran on metal alone and asserted finiteness — and did
    /// not check the decode logits at all beyond their length.
    #[test]
    fn qwen35_dynamic_matches_cpu_on_metal() {
        if !rlx_runtime::is_available(Device::Metal) {
            eprintln!("skip: metal not available");
            return;
        }
        let cpu = run(Device::Cpu);
        let dev = run(Device::Metal);
        super::compile_support::assert_matches_cpu("metal", &cpu, &dev, 2e-3);
    }
}
