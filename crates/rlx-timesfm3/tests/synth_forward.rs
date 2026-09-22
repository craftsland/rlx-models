// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use ndarray::Array3;
use rlx_timesfm3::{TimesFM3Config, TimesFM3Model};

#[test]
fn synth_forward_finite() {
    let cfg = TimesFM3Config::synth_tiny();
    let model = TimesFM3Model::synth(cfg.clone(), 99);
    let ctx: Vec<f32> = (0..cfg.input_patch_len * 4)
        .map(|i| (i as f32 * 0.07).sin())
        .collect();
    let arr = Array3::from_shape_vec((1, 1, ctx.len()), ctx).unwrap();
    let out = model.decode(arr.view(), 32, None, None, None);
    assert_eq!(out.shape()[2], 32);
    assert!(out.iter().all(|v| v.is_finite()));

    // `is_finite` plus a shape is passed by a decoder returning all zeros, or a
    // constant, or one that ignores its seed. These are the properties that say
    // the model actually ran.
    assert!(
        out.iter().any(|v| v.abs() > 1e-9),
        "forecast is identically zero"
    );
    let first = out.iter().next().copied().unwrap();
    assert!(
        out.iter().any(|v| (v - first).abs() > 1e-9),
        "forecast is a constant {first}"
    );

    // Same seed, same numbers.
    let again = TimesFM3Model::synth(cfg.clone(), 99).decode(arr.view(), 32, None, None, None);
    assert_eq!(out, again, "decoding is not deterministic for a fixed seed");

    // A different seed is a different model, so a different forecast.
    let other = TimesFM3Model::synth(cfg.clone(), 100).decode(arr.view(), 32, None, None, None);
    assert!(
        out.iter()
            .zip(other.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9),
        "seed 99 and seed 100 forecast identically — the seed is ignored"
    );
}
