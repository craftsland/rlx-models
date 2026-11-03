// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Real published weights, three blocks at a time.
//!
//! ```text
//! python3 scripts/mistral_large4_subset.py SUBSET.gguf            # ~1.1 GB of range reads
//! python3 scripts/mistral_large4_reference.py --gguf SUBSET.gguf /tmp/ml4-real   # gguf-py + torch
//! RLX_ML4_REAL_DIR=/tmp/ml4-real cargo test -p rlx-mistral-large4 --release --test real_weights
//! ```
//!
//! Blocks 35, 38 and 39 of `UD-IQ1_S`: a NoPE layer owning its KV, a RoPE
//! layer with a shared indexer, and a layer attending to block 35's latent.
//! The reference dequantizes with gguf-py (llama.cpp's own reader) and runs the
//! torch port, so the K-quant/IQ dequant, every tensor orientation and the
//! block math are each checked against an independent implementation, on
//! trained values. `index_topk` is lowered (to 6 at `seq = 16`) on both sides
//! so the DSA selection is real.

use rlx_core::flow_bridge::PackedWeightLoaderSource;
use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_mistral_large4::{
    HIDDEN_INPUT, MistralLarge4Config, build_mistral_large4_blocks_with_source,
};
use rlx_runtime::Device;

fn dev() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

struct Fixture {
    gguf: String,
    cfg: MistralLarge4Config,
    blocks: Vec<usize>,
    seq: usize,
    input: Vec<f32>,
    want: Vec<f32>,
}

fn fixture() -> Option<Fixture> {
    let dir = rlx_ir::env::read("RLX_ML4_REAL_DIR")
        .ok()
        .filter(|s| !s.is_empty())?;
    let p = std::path::Path::new(&dir).join("ref_real.json");
    if !p.exists() {
        eprintln!("skip: {p:?} missing");
        return None;
    }
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let gguf = j["gguf"].as_str().unwrap().to_string();
    if !std::path::Path::new(&gguf).exists() {
        eprintln!("skip: {gguf} missing");
        return None;
    }
    let mut cfg = MistralLarge4Config::from_gguf_path(&gguf).unwrap();
    cfg.index_topk = j["index_topk"].as_u64().unwrap() as usize;
    let floats = |k: &str| -> Vec<f32> {
        j[k].as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };
    Some(Fixture {
        blocks: j["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect(),
        seq: j["seq"].as_u64().unwrap() as usize,
        input: floats("hidden_in"),
        want: floats("hidden_out"),
        gguf,
        cfg,
    })
}

/// Relative max error: the residual stream after three blocks is O(10..100),
/// and f32 vs f64 over 6144-wide reductions lands ~1e-5 relative. A wrong
/// DSA pick, RoPE pair or orientation is O(1).
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale
}

const TOL: f32 = 1e-3;

#[test]
fn blocks_match_reference_dequantized() {
    let Some(f) = fixture() else { return };
    assert!(
        !f.cfg.dsa_is_dense(f.seq),
        "the fixture must exercise DSA selection"
    );
    let mut loader = rlx_core::weight_loader::load_from_path(&f.gguf).unwrap();
    let mut w = WeightMap::from_weight_loader_dequant_all(loader.as_mut()).unwrap();
    let built = build_mistral_large4_blocks_with_source(
        &f.cfg,
        &mut WeightMapSource(&mut w),
        f.seq,
        &f.blocks,
        false,
    )
    .unwrap();
    let mut c = compile_built(built, dev()).unwrap();
    let got = c.run(&[(HIDDEN_INPUT, &f.input[..])]).remove(0);
    let e = rel_err(&got, &f.want);
    eprintln!("dequantized: rel max |Δ| = {e:.3e}");
    assert!(
        e < TOL,
        "blocks {:?} differ from the reference by {e} (relative)",
        f.blocks
    );
}

#[test]
fn blocks_match_reference_packed() {
    let Some(f) = fixture() else { return };
    let mut loader = rlx_core::weight_loader::load_from_path(&f.gguf).unwrap();
    let built = build_mistral_large4_blocks_with_source(
        &f.cfg,
        &mut PackedWeightLoaderSource(loader.as_mut()),
        f.seq,
        &f.blocks,
        false,
    )
    .unwrap();
    let mut c = compile_built(built, dev()).unwrap();
    let got = c.run(&[(HIDDEN_INPUT, &f.input[..])]).remove(0);
    let e = rel_err(&got, &f.want);
    eprintln!("packed: rel max |Δ| = {e:.3e}");
    assert!(
        e < TOL,
        "packed blocks {:?} differ from the reference by {e} (relative)",
        f.blocks
    );
}
