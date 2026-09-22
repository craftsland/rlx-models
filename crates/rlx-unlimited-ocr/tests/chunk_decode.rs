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

//! Multi-token decode (`seq > 1`) — the speculative *verify* pass.
//!
//! Scoring a whole draft in one forward is only sound if it gives exactly what
//! running those tokens one at a time gives. That is the invariant here, and it
//! is what makes greedy speculative decoding lossless: whatever the draft head
//! proposes, the accepted tokens are the ones the target would have produced
//! anyway.
//!
//! The risk the test targets is the mask. A chunk needs causality *within* it —
//! query `i` must not see draft token `j > i` — on top of the padding mask over
//! the bucketed cache. Getting that wrong leaks future tokens into earlier
//! positions, which inflates acceptance and silently changes the output.

use rlx_core::backend_matrix::{Failures, available_devices, max_rel_diff};
use rlx_core::flow_util::compile_built;
use rlx_runtime::Device;
use rlx_unlimited_ocr::config::UnlimitedOcrConfig;
use rlx_unlimited_ocr::expert_pack::PackedLmWeights;
use rlx_unlimited_ocr::lm_graph::{
    build_unlimited_ocr_decode_chunk_built, build_unlimited_ocr_prefill_built_from_pack,
};
use rlx_unlimited_ocr::lm_precision::ResolvedLmPrecision;
use std::sync::Arc;

mod tiny;
use tiny::{cfg, fill, synthetic_weights};

const PROMPT: usize = 6;
const BUCKET: usize = 32;

fn pack_for(cfg: &UnlimitedOcrConfig) -> Arc<PackedLmWeights> {
    let mut wm = synthetic_weights(cfg);
    Arc::new(
        PackedLmWeights::from_weight_map(&mut wm, cfg.clone(), ResolvedLmPrecision::F32)
            .expect("pack"),
    )
}

/// Prefill, returning `[logits, k0, v0, k1, v1, ...]`.
fn prefill(cfg: &UnlimitedOcrConfig, pack: &Arc<PackedLmWeights>, device: Device) -> Vec<Vec<f32>> {
    let embeds = fill(PROMPT * cfg.hidden_size, 0.5);
    let built = build_unlimited_ocr_prefill_built_from_pack(cfg, pack, 1, PROMPT).expect("build");
    let mut compiled = compile_built(built, device).expect("compile");
    compiled.run(&[("inputs_embeds", embeds.as_slice())])
}

/// Pad a layer's KV to `BUCKET` rows.
fn padded(rows: &[f32], hidden: usize) -> Vec<f32> {
    let mut out = rows.to_vec();
    out.resize(BUCKET * hidden, 7.0); // loud sentinel: masked padding must not leak
    out
}

/// Additive bias `[1, heads, seq, BUCKET + seq]`: a query at chunk position `i`
/// (absolute `valid + i`) sees real cache rows `< valid` and chunk positions
/// `<= i`; everything else is `-inf`.
fn bias_mask(heads: usize, seq: usize, valid: usize) -> Vec<f32> {
    let keys = BUCKET + seq;
    let mut m = vec![f32::NEG_INFINITY; heads * seq * keys];
    for h in 0..heads {
        for i in 0..seq {
            let row = (h * seq + i) * keys;
            for k in 0..valid {
                m[row + k] = 0.0;
            }
            for j in 0..=i {
                m[row + BUCKET + j] = 0.0;
            }
        }
    }
    m
}

/// `[1, BUCKET + 1]` binary keep-mask for a single-token step.
fn keep_mask(valid: usize) -> Vec<f32> {
    let mut m = vec![0f32; BUCKET + 1];
    m[..valid].fill(1.0);
    m[BUCKET] = 1.0;
    m
}

/// Run `seq` tokens as one chunk; returns per-position logits.
fn decode_chunk(
    cfg: &UnlimitedOcrConfig,
    pack: &Arc<PackedLmWeights>,
    past: &[Vec<f32>],
    valid: usize,
    embeds: &[f32],
    seq: usize,
    device: Device,
) -> Vec<Vec<f32>> {
    let h = cfg.hidden_size;
    let n_layers = cfg.num_hidden_layers;
    let built = build_unlimited_ocr_decode_chunk_built(cfg, pack, 1, BUCKET, seq, true, false)
        .expect("build");
    let mut compiled = compile_built(built, device).expect("compile");

    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for i in 0..seq {
        let (c, s) = rlx_unlimited_ocr::lm_graph::compute_rope_slice(cfg, valid + i);
        cos.extend(c);
        sin.extend(s);
    }
    let mask = bias_mask(cfg.num_attention_heads, seq, valid);
    let feeds: Vec<(String, Vec<f32>)> = (0..n_layers)
        .flat_map(|i| {
            [
                (format!("past_k_{i}"), padded(&past[1 + 2 * i], h)),
                (format!("past_v_{i}"), padded(&past[1 + 2 * i + 1], h)),
            ]
        })
        .collect();

    let mut pairs: Vec<(&str, &[f32])> = vec![
        ("inputs_embeds", embeds),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
        ("mask", &mask),
    ];
    for (n, d) in &feeds {
        pairs.push((n.as_str(), d.as_slice()));
    }
    let outs = compiled.run(&pairs);
    let logits = &outs[0];
    let vocab = cfg.vocab_size;
    (0..seq)
        .map(|i| logits[i * vocab..(i + 1) * vocab].to_vec())
        .collect()
}

/// Run the same tokens one at a time, threading KV forward.
fn decode_stepwise(
    cfg: &UnlimitedOcrConfig,
    pack: &Arc<PackedLmWeights>,
    past: &[Vec<f32>],
    valid0: usize,
    embeds: &[f32],
    seq: usize,
    device: Device,
) -> Vec<Vec<f32>> {
    let h = cfg.hidden_size;
    let n_layers = cfg.num_hidden_layers;
    let built = build_unlimited_ocr_decode_chunk_built(cfg, pack, 1, BUCKET, 1, true, false)
        .expect("build");
    let mut compiled = compile_built(built, device).expect("compile");

    let mut cache: Vec<Vec<f32>> = (0..2 * n_layers).map(|i| padded(&past[1 + i], h)).collect();
    let mut out = Vec::new();
    for i in 0..seq {
        // History grows by one per step, so the absolute position is the
        // starting length plus how many steps have been taken.
        let valid = valid0 + i;
        let (cos, sin) = rlx_unlimited_ocr::lm_graph::compute_rope_slice(cfg, valid);
        let mask = keep_mask(valid);
        let names: Vec<(String, String)> = (0..n_layers)
            .map(|l| (format!("past_k_{l}"), format!("past_v_{l}")))
            .collect();
        let step = &embeds[i * h..(i + 1) * h];
        let mut pairs: Vec<(&str, &[f32])> = vec![
            ("inputs_embeds", step),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
            ("mask", &mask),
        ];
        for (l, (kn, vn)) in names.iter().enumerate() {
            pairs.push((kn.as_str(), cache[2 * l].as_slice()));
            pairs.push((vn.as_str(), cache[2 * l + 1].as_slice()));
        }
        let outs = compiled.run(&pairs);
        out.push(outs[0].clone());
        // Append this token's KV at `valid` for the next step.
        for l in 0..n_layers {
            for (slot, full) in [(2 * l, 1 + 2 * l), (2 * l + 1, 1 + 2 * l + 1)] {
                let t = &outs[full];
                let n_full = t.len() / h;
                let row = &t[(n_full - 1) * h..n_full * h];
                cache[slot][valid * h..(valid + 1) * h].copy_from_slice(row);
            }
        }
    }
    out
}

/// The load-bearing invariant for speculative decoding.
#[test]
fn chunk_decode_matches_stepwise_decode() {
    let cfg: UnlimitedOcrConfig = cfg();
    let h = cfg.hidden_size;

    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        let pack = pack_for(&cfg);
        let past = prefill(&cfg, &pack, device);
        for seq in [2usize, 4] {
            let embeds = fill(seq * h, 0.9);
            let chunk = decode_chunk(&cfg, &pack, &past, PROMPT, &embeds, seq, device);
            let step = decode_stepwise(&cfg, &pack, &past, PROMPT, &embeds, seq, device);
            for i in 0..seq {
                let rel = max_rel_diff(&step[i], &chunk[i]);
                if rel >= 2e-3 {
                    fails.push(
                        name,
                        format!("seq {seq} position {i}: chunk vs stepwise differ by {rel:.5}"),
                    );
                }
            }
        }
    }
    fails.assert_empty("chunk decode");
}

/// Negative control: if the chunk mask were not causal *within* the chunk,
/// position 0 would see later draft tokens. Feeding a different tail must
/// therefore leave position 0's logits untouched.
#[test]
fn earlier_chunk_positions_do_not_see_later_ones() {
    let cfg: UnlimitedOcrConfig = cfg();
    let h = cfg.hidden_size;
    let seq = 4usize;

    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        let pack = pack_for(&cfg);
        let past = prefill(&cfg, &pack, device);
        let a = fill(seq * h, 0.9);
        let mut b = a.clone();
        // Perturb every position except the first.
        for v in b[h..].iter_mut() {
            *v += 0.5;
        }
        let ca = decode_chunk(&cfg, &pack, &past, PROMPT, &a, seq, device);
        let cb = decode_chunk(&cfg, &pack, &past, PROMPT, &b, seq, device);

        let rel0 = max_rel_diff(&ca[0], &cb[0]);
        if rel0 >= 1e-4 {
            fails.push(
                name,
                format!("position 0 changed by {rel0:.5} when only later positions moved"),
            );
        }
        // And the perturbation must actually reach the later positions, or the
        // check above is vacuous.
        let rel_last = max_rel_diff(&ca[seq - 1], &cb[seq - 1]);
        if rel_last < 1e-4 {
            fails.push(name, "perturbation did not affect the last position");
        }
    }
    fails.assert_empty("chunk causality");
}

/// The same invariant through `CompiledLm`, which is what the runner uses:
/// `decode_chunk` must equal repeated `decode_step`, and a partial accept must
/// leave the cache exactly where the accepted tokens put it.
#[test]
fn compiled_lm_chunk_matches_steps_and_rolls_back() {
    use rlx_unlimited_ocr::lm_device::CompiledLm;

    let cfg: UnlimitedOcrConfig = cfg();
    let h = cfg.hidden_size;
    let seq = 4usize;

    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        let pack = pack_for(&cfg);
        let embeds = fill(PROMPT * h, 0.5);
        let chunk_in = fill(seq * h, 0.9);

        // Reference: one token at a time.
        let mut lm = CompiledLm::new(device, Arc::clone(&pack));
        let (_, mut kv) = lm.prefill(&embeds, PROMPT).expect("prefill");
        let mut stepwise = Vec::new();
        for i in 0..seq {
            let step = &chunk_in[i * h..(i + 1) * h];
            stepwise.push(lm.decode_step(step, PROMPT + i, &mut kv).expect("step"));
        }
        let steps_valid = kv.valid();

        // Same tokens as one chunk.
        let mut lm2 = CompiledLm::new(device, Arc::clone(&pack));
        let (_, mut kv2) = lm2.prefill(&embeds, PROMPT).expect("prefill");
        let chunked = lm2
            .decode_chunk(&chunk_in, PROMPT, seq, &mut kv2)
            .expect("chunk");

        if kv2.valid() != steps_valid {
            fails.push(
                name,
                format!(
                    "cache holds {} rows after a chunk, {steps_valid} after steps",
                    kv2.valid()
                ),
            );
        }
        for i in 0..seq {
            let rel = max_rel_diff(&stepwise[i], &chunked[i]);
            if rel >= 2e-3 {
                fails.push(
                    name,
                    format!("position {i}: chunk vs steps differ by {rel:.5}"),
                );
            }
        }

        // Accept only the first two, then continue one at a time. The result
        // must match the stepwise run that never speculated.
        let mut lm3 = CompiledLm::new(device, Arc::clone(&pack));
        let (_, mut kv3) = lm3.prefill(&embeds, PROMPT).expect("prefill");
        lm3.decode_chunk(&chunk_in, PROMPT, seq, &mut kv3)
            .expect("chunk");
        lm3.rollback(&mut kv3, PROMPT + 2);
        if kv3.valid() != PROMPT + 2 {
            fails.push(name, format!("rollback left {} rows", kv3.valid()));
            continue;
        }
        let next = &chunk_in[2 * h..3 * h];
        let after = lm3
            .decode_step(next, PROMPT + 2, &mut kv3)
            .expect("post-rollback step");
        let rel = max_rel_diff(&stepwise[2], &after);
        if rel >= 2e-3 {
            fails.push(
                name,
                format!("after rollback, position 2 differs by {rel:.5}"),
            );
        }
    }
    fails.assert_empty("CompiledLm chunk decode");
}
