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

//! `sliding_window == 0` (plain causal) decode, as DeepSeek-OCR derivatives
//! such as `jinaai/jina-ocr-v1` use it.
//!
//! With a window, the KV cache reaches a fixed length and one compiled decode
//! graph serves the whole run. Without one the history grows every token, so
//! the exact-shape `MaskKind::Causal` graph would be rebuilt for each token.
//! The bucketed path pads the past up to [`KV_BUCKET`] and masks the padding
//! with `MaskKind::Custom`, compiling once per bucket instead.
//!
//! The load-bearing claim is that masked padding changes nothing: bucketed
//! decode must produce the same logits as the exact-shape graph.

use rlx_core::flow_util::compile_built;
use rlx_runtime::Device;
use rlx_unlimited_ocr::config::UnlimitedOcrConfig;
use rlx_unlimited_ocr::lm_device::{CompiledLm, KV_BUCKET};
use rlx_unlimited_ocr::lm_graph::{
    build_unlimited_ocr_decode_built_ext, build_unlimited_ocr_prefill_built, compute_rope_slice,
};
use rlx_unlimited_ocr::lm_precision::{LmWeightPrecision, ResolvedLmPrecision};
use std::sync::Arc;

use rlx_core::backend_matrix::{Failures, available_devices};

mod tiny;
use tiny::{cfg, fill, synthetic_weights};

const SEQ: usize = 5;

/// Prefill once; returns `(logits, per-layer KV side outputs)`.
fn prefill(cfg: &UnlimitedOcrConfig, device: Device) -> Vec<Vec<f32>> {
    let embeds = fill(SEQ * cfg.hidden_size, 0.5);
    let mut wm = synthetic_weights(cfg);
    let built = build_unlimited_ocr_prefill_built(cfg, &mut wm, 1, SEQ).expect("prefill build");
    let mut compiled = compile_built(built, device).expect("prefill compile");
    let outs = compiled.run(&[("inputs_embeds", embeds.as_slice())]);
    assert_eq!(outs.len(), 1 + 2 * cfg.num_hidden_layers);
    outs
}

/// One decode step from `past`, either exact-shape causal or padded to `bucket`
/// with an explicit keep-mask.
#[allow(clippy::too_many_arguments)]
fn decode_once(
    cfg: &UnlimitedOcrConfig,
    past: &[Vec<f32>],
    step: &[f32],
    pos: usize,
    bucket: Option<usize>,
    device: Device,
) -> Vec<f32> {
    let h = cfg.hidden_size;
    let n_layers = cfg.num_hidden_layers;
    let valid = past[1].len() / h;
    let past_seq = bucket.unwrap_or(valid);
    assert!(past_seq >= valid);

    let mut wm = synthetic_weights(cfg);
    let built = build_unlimited_ocr_decode_built_ext(cfg, &mut wm, 1, past_seq, bucket.is_some())
        .expect("decode build");
    let mut compiled = compile_built(built, device).expect("decode compile");

    let (cos, sin) = compute_rope_slice(cfg, pos);
    let mask: Vec<f32> = {
        let mut m = vec![0f32; past_seq + 1];
        m[..valid].fill(1.0);
        m[past_seq] = 1.0;
        m
    };
    let padded: Vec<(String, Vec<f32>)> = (0..n_layers)
        .flat_map(|i| {
            let mut k = past[1 + 2 * i].clone();
            let mut v = past[1 + 2 * i + 1].clone();
            // Pad with a loud sentinel: if the mask leaks, the logits blow up.
            k.resize(past_seq * h, 7.0);
            v.resize(past_seq * h, 7.0);
            [(format!("past_k_{i}"), k), (format!("past_v_{i}"), v)]
        })
        .collect();

    let mut pairs: Vec<(&str, &[f32])> = vec![
        ("inputs_embeds", step),
        ("rope_cos", cos.as_slice()),
        ("rope_sin", sin.as_slice()),
    ];
    if bucket.is_some() {
        pairs.push(("mask", mask.as_slice()));
    }
    for (n, d) in &padded {
        pairs.push((n.as_str(), d.as_slice()));
    }
    compiled.run(&pairs).swap_remove(0)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The core claim: padding the KV cache and masking the padding is a no-op —
/// on every backend, since `MaskKind::Custom` is a different kernel path from
/// `MaskKind::Causal` in each of them.
#[test]
fn bucketed_decode_matches_exact_shape_decode() {
    let cfg = cfg();
    cfg.validate().expect("config");
    let step = fill(cfg.hidden_size, 0.9);

    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        let past = prefill(&cfg, device);
        let exact = decode_once(&cfg, &past, &step, SEQ, None, device);
        if exact.len() != cfg.vocab_size || !exact.iter().all(|v| v.is_finite()) {
            fails.push(name, "exact-shape decode produced bad logits");
            continue;
        }
        // Several bucket widths: how much padding there is must not matter.
        for bucket in [8usize, 32, 64] {
            let bucketed = decode_once(&cfg, &past, &step, SEQ, Some(bucket), device);
            let diff = max_abs_diff(&exact, &bucketed);
            if diff >= 1e-3 {
                fails.push(
                    name,
                    format!("bucket {bucket}: diverged from exact-shape decode by {diff}"),
                );
            }
        }
    }
    fails.assert_empty("bucketed vs exact-shape decode");
}

/// A sanity check on the check: with the mask removed, the sentinel padding
/// *does* change the answer — so the test above is not vacuously comparing two
/// identical code paths.
#[test]
fn unmasked_padding_would_corrupt_the_result() {
    let cfg = cfg();
    let step = fill(cfg.hidden_size, 0.9);
    let h = cfg.hidden_size;

    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        let past = prefill(&cfg, device);
        let exact = decode_once(&cfg, &past, &step, SEQ, None, device);
        // One extra KV row of the same sentinel the bucketed path pads with,
        // fed to a causal graph that has no way to know it is padding.
        let mut poisoned = past.clone();
        for i in 0..cfg.num_hidden_layers {
            poisoned[1 + 2 * i].extend(std::iter::repeat_n(7.0f32, h));
            poisoned[1 + 2 * i + 1].extend(std::iter::repeat_n(7.0f32, h));
        }
        let with_extra_row = decode_once(&cfg, &poisoned, &step, SEQ, None, device);
        if max_abs_diff(&exact, &with_extra_row) <= 1e-4 {
            fails.push(name, "an extra unmasked KV row did not change the logits");
        }
    }
    fails.assert_empty("negative control");
}

/// `CompiledLm` must reuse one compiled graph across a bucket's worth of steps
/// — the reason bucketing exists. Per-token compilation would make a 4k-token
/// transcript unusable.
#[test]
fn compiled_lm_reuses_one_graph_per_bucket() {
    let cfg = cfg();
    let mut wm = synthetic_weights(&cfg);
    let pack = Arc::new(
        rlx_unlimited_ocr::expert_pack::PackedLmWeights::from_weight_map(
            &mut wm,
            cfg.clone(),
            ResolvedLmPrecision::F32,
        )
        .expect("pack"),
    );
    let mut lm = CompiledLm::new(Device::Cpu, Arc::clone(&pack));

    let embeds = fill(SEQ * cfg.hidden_size, 0.5);
    let (logits, mut kv) = lm.prefill(&embeds, SEQ).expect("prefill");
    assert_eq!(logits.len(), cfg.vocab_size);

    let steps = 6usize;
    for s in 0..steps {
        let step = fill(cfg.hidden_size, 0.9 + s as f32 * 0.01);
        let out = lm.decode_step(&step, SEQ + s, &mut kv).expect("decode");
        assert_eq!(out.len(), cfg.vocab_size);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "step {s} produced non-finite logits"
        );
    }

    // All `steps` decodes stayed inside the first bucket, so exactly one decode
    // graph should have been compiled.
    assert!(
        SEQ + steps <= KV_BUCKET,
        "test must stay within one bucket to make the claim"
    );
    assert_eq!(
        lm.compiled_decode_graphs(),
        1,
        "full-causal decode compiled more than one graph inside a single bucket"
    );
}

/// The windowed path (Unlimited-OCR's own 128) must be untouched by all this.
#[test]
fn windowed_config_still_uses_the_exact_shape_path() {
    let mut cfg = cfg();
    cfg.model_type = "unlimited-ocr".into();
    cfg.sliding_window = 4;
    let mut wm = synthetic_weights(&cfg);
    let pack = Arc::new(
        rlx_unlimited_ocr::expert_pack::PackedLmWeights::from_weight_map(
            &mut wm,
            cfg.clone(),
            ResolvedLmPrecision::F32,
        )
        .expect("pack"),
    );
    let mut lm = CompiledLm::new(Device::Cpu, Arc::clone(&pack));
    let embeds = fill(SEQ * cfg.hidden_size, 0.5);
    let (_, mut kv) = lm.prefill(&embeds, SEQ).expect("prefill");
    for s in 0..6 {
        let step = fill(cfg.hidden_size, 0.9 + s as f32 * 0.01);
        let out = lm.decode_step(&step, SEQ + s, &mut kv).expect("decode");
        assert!(out.iter().all(|v| v.is_finite()));
    }
    // Ring steady state: past length stops growing, so the graph count stays small.
    assert!(
        lm.compiled_decode_graphs() <= cfg.sliding_window + 1,
        "windowed decode compiled {} graphs",
        lm.compiled_decode_graphs()
    );
}

/// `LmWeightPrecision` plumbing is orthogonal to the window; keep the enum
/// import meaningful and assert the default is unchanged.
#[test]
fn default_precision_is_auto() {
    assert_eq!(LmWeightPrecision::default(), LmWeightPrecision::Auto);
}
