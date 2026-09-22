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

// basic test: tiny synthetic qwen35 runner on WGPU (skips when unavailable).

mod compile_support;

#[cfg(feature = "gpu")]
mod wgpu_tests {
    use rlx_models::Qwen35RunnerBuilder;
    use rlx_models::qwen35::synth;
    use rlx_runtime::Device;

    /// Build the tiny runner on `device` and take one prefill's logits.
    fn run(device: Device) -> Vec<f32> {
        let cfg = synth::tiny_cfg();
        let weights = synth::synth_weights(&cfg);
        let mut runner = Qwen35RunnerBuilder::default()
            .inline_weights(cfg.clone(), weights)
            .device(device)
            .max_seq(8)
            .last_logits_only(true)
            .build()
            .expect("runner");
        let logits = runner.prefill_get_last_logits(&[1, 2, 3]).expect("prefill");
        assert_eq!(logits.len(), cfg.vocab_size, "logits width");
        logits
    }

    /// The wgpu result must equal CPU's, not merely be finite.
    ///
    /// This asserted only finiteness before — a bar an all-zero result clears.
    #[test]
    fn qwen35_tiny_runner_matches_cpu_on_wgpu() {
        if !rlx_runtime::is_available(Device::Gpu) {
            eprintln!("skip: wgpu not available");
            return;
        }
        let cpu = run(Device::Cpu);
        let dev = run(Device::Gpu);
        assert!(
            cpu.iter().chain(&dev).all(|v| v.is_finite()),
            "non-finite logits"
        );

        // wgpu does not match CPU elementwise here: its logits are CPU's times
        // 1.0146..1.0161 — a *uniform gain*, spread 1.5e-3, so the predicted
        // token is unchanged. That signature (one scale factor across the whole
        // vocabulary) is a lower-precision `inversesqrt` in a normalization, not
        // a lowering that dropped or misplaced values. Asserting it as a gain is
        // both tighter and more informative than widening an absolute tolerance
        // to 2e-2 and calling it agreement: a real defect would not come out as
        // a clean scale.
        let ratios: Vec<f32> = cpu
            .iter()
            .zip(&dev)
            .filter(|(c, _)| c.abs() > 1e-6)
            .map(|(c, d)| d / c)
            .collect();
        assert!(!ratios.is_empty(), "CPU logits are all ~zero");
        let lo = ratios.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = ratios.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            hi - lo < 5e-3,
            "wgpu vs CPU is not a uniform gain: ratio {lo:.6}..{hi:.6} (spread {:.2e})",
            hi - lo
        );
        assert!(
            (0.98..1.03).contains(&lo) && (0.98..1.03).contains(&hi),
            "wgpu gain {lo:.4}..{hi:.4} is outside the observed precision band"
        );

        // What actually decides the next token.
        let pick = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        assert_eq!(
            pick(&cpu),
            pick(&dev),
            "wgpu picks a different token than CPU"
        );
    }
}
