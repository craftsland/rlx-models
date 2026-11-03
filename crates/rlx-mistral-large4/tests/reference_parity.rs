// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Logits parity against the independent torch port of
//! `modular_mistral_large4.py`, on a tiny random checkpoint.
//!
//! ```text
//! python3 scripts/mistral_large4_reference.py /tmp/ml4-ref
//! RLX_ML4_REF_DIR=/tmp/ml4-ref cargo test -p rlx-mistral-large4 --test reference_parity
//! ```
//!
//! Skipped when `RLX_ML4_REF_DIR` is unset. The fixture covers RoPE and NoPE
//! layers, a layer sharing an earlier NoPE layer's KV, `"shared"` indexers
//! after both kinds, a dense first layer, MoE with `routed_scaling != 1`,
//! softcapping, and — in the `sparse` config — a DSA selection that keeps
//! fewer keys than the causal set (`index_topk = 5 < seq = 9`).

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_mistral_large4::{
    DecodeSession, MistralLarge4Config, PrefillOptions, build_mistral_large4_decode_flow,
    build_mistral_large4_text_flow,
};
use rlx_runtime::Device;
use std::path::PathBuf;

/// f32 graph vs f64 reference over 8 layers. The legitimate gap is ~1e-5;
/// one wrong DSA pick or RoPE pair moves logits by O(0.1) (sparse vs dense
/// alone differs by ~3).
const TOL: f32 = 2e-3;

fn dev() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn ref_dir() -> Option<PathBuf> {
    let d = rlx_ir::env::read("RLX_ML4_REF_DIR")
        .ok()
        .filter(|s| !s.is_empty())?;
    let p = PathBuf::from(d);
    if p.join("weights.safetensors").exists() {
        Some(p)
    } else {
        eprintln!("RLX_ML4_REF_DIR={p:?} has no weights.safetensors; skipping");
        None
    }
}

struct Ref {
    ids: Vec<f32>,
    seq: usize,
    vocab: usize,
    logits: Vec<f32>,
}

fn load(dir: &std::path::Path, which: &str) -> (MistralLarge4Config, WeightMap, Ref) {
    let cfg = MistralLarge4Config::from_hf_file(dir.join(format!("config_{which}.json"))).unwrap();
    let w = WeightMap::from_file(dir.join("weights.safetensors").to_str().unwrap()).unwrap();
    let j: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join(format!("ref_{which}.json"))).unwrap(),
    )
    .unwrap();
    let f = |k: &str| -> Vec<f32> {
        j[k].as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };
    let r = Ref {
        ids: f("input_ids"),
        seq: j["seq"].as_u64().unwrap() as usize,
        vocab: j["vocab"].as_u64().unwrap() as usize,
        logits: f("logits"),
    };
    (cfg, w, r)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn prefill(
    cfg: &MistralLarge4Config,
    w: &mut WeightMap,
    r: &Ref,
    opts: PrefillOptions,
) -> Vec<f32> {
    let built = build_mistral_large4_text_flow(cfg, w, r.seq, opts).unwrap();
    let mut c = compile_built(built, dev()).unwrap();
    c.run(&[("input_ids", &r.ids[..])]).remove(0)
}

#[test]
fn sparse_dsa_matches_reference() {
    let Some(dir) = ref_dir() else { return };
    let (cfg, mut w, r) = load(&dir, "sparse");
    assert!(
        !cfg.dsa_is_dense(r.seq),
        "the sparse fixture must exercise selection"
    );
    let got = prefill(&cfg, &mut w, &r, PrefillOptions::default());
    let d = max_abs(&got, &r.logits);
    eprintln!("sparse: max |Δ| = {d:.3e}");
    assert!(d < TOL, "sparse logits differ from the reference by {d}");
}

#[test]
fn dense_matches_reference_both_paths() {
    let Some(dir) = ref_dir() else { return };
    let (cfg, mut w, r) = load(&dir, "dense");
    assert!(cfg.dsa_is_dense(r.seq));
    // The causal short-circuit…
    let fast = prefill(&cfg, &mut w, &r, PrefillOptions::default());
    let d = max_abs(&fast, &r.logits);
    eprintln!("dense (causal): max |Δ| = {d:.3e}");
    assert!(d < TOL, "dense logits differ from the reference by {d}");
    // …and the selection machinery forced on, which must agree with it:
    // below the budget top-k keeps every causal key.
    let (_, mut w, _) = load(&dir, "dense");
    let forced = prefill(
        &cfg,
        &mut w,
        &r,
        PrefillOptions {
            with_lm_head: true,
            force_dsa: true,
        },
    );
    let d2 = max_abs(&forced, &fast);
    eprintln!("dense forced vs causal: max |Δ| = {d2:.3e}");
    assert!(
        d2 < 1e-4,
        "forced DSA selection is not the identity below the budget ({d2})"
    );
}

#[test]
fn decode_reproduces_prefill() {
    let Some(dir) = ref_dir() else { return };
    let (cfg, mut w, r) = load(&dir, "dense");
    let cap = r.seq;
    let (built, layout) = build_mistral_large4_decode_flow(&cfg, &mut w, cap, true).unwrap();
    let mut c = compile_built(built, dev()).unwrap();
    let mut sess = DecodeSession::new(layout, cap);
    let mut worst = 0f32;
    for (t, &id) in r.ids.iter().enumerate() {
        let tok = [id];
        let out = {
            let mut inputs = sess.inputs();
            inputs.insert(0, ("input_ids", &tok[..]));
            c.run(&inputs).remove(0)
        };
        let logits = sess.commit(&out).unwrap().to_vec();
        let want = &r.logits[t * r.vocab..(t + 1) * r.vocab];
        worst = worst.max(max_abs(&logits, want));
    }
    eprintln!("decode vs reference: max |Δ| = {worst:.3e}");
    assert!(worst < TOL, "decode drifts from the reference by {worst}");
}
