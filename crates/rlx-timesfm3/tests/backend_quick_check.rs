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

//! Per-backend quick check on the synthetic tiny model.

use ndarray::Array3;
use rlx_runtime::Device;
use rlx_timesfm3::{
    TimesFM3Config, TimesFM3Model, TimesFM3Session, available_devices, device_label,
};

/// Every backend must forecast the same numbers as CPU, not merely finite ones.
///
/// This already swept the devices, but asked each only for the right shape and
/// finite values — which a backend returning zeros, or one silently falling
/// back to a different path, passes just as easily. The synthetic model is
/// seeded, so the forecasts are deterministic and directly comparable.
#[test]
fn all_available_backends_decode_like_cpu() {
    let cfg = TimesFM3Config::synth_tiny();
    let ctx: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    let arr = Array3::from_shape_vec((1, 1, ctx.len()), ctx).unwrap();

    let forecast = |dev: Device| -> Vec<f32> {
        let model = TimesFM3Model::synth(cfg.clone(), 11);
        let mut session = TimesFM3Session::from_model(model, dev);
        let out = session
            .decode(arr.view(), 4, None, None, None)
            .unwrap_or_else(|e| panic!("{} decode failed: {e}", device_label(dev)));
        assert_eq!(out.shape(), &[1, 1, 4, 9], "{}", device_label(dev));
        out.iter().copied().collect()
    };

    let cpu = forecast(Device::Cpu);
    assert!(
        cpu.iter().all(|v| v.is_finite()),
        "CPU reference is not finite"
    );
    assert!(
        cpu.iter().any(|v| v.abs() > 1e-9),
        "CPU forecast is all zero, so nothing below would mean anything"
    );
    let scale = cpu.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);

    let mut bad = Vec::new();
    for dev in available_devices() {
        if dev == Device::Cpu {
            continue;
        }
        let got = forecast(dev);
        let rel = got
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / scale;
        if !rel.is_finite() || rel >= 2e-3 {
            let zero = got.iter().all(|v| *v == 0.0);
            bad.push(format!(
                "  {}: differs from CPU by {rel:.6}{}",
                device_label(dev),
                if zero { " — output is ALL ZERO" } else { "" }
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "forecast differs from CPU on {} backend(s):\n{}",
        bad.len(),
        bad.join("\n")
    );
}
