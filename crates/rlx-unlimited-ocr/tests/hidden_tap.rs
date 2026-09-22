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

//! The **pre**-final-norm hidden state published by the prefill/decode graphs.
//!
//! That is the tensor `DeepseekV2Model` hands to FastMTP as
//! `pre_norm_hidden_states`, not the one `lm_head` consumes — the draft block
//! applies its own `hnorm` to it. The check is self-proving and needs no
//! reference: normalizing the tapped hidden with `model.norm` and pushing it
//! through the LM head on the host must reproduce the graph's own logits. That
//! pins both halves — tapping post-norm instead double-norms here and misses by
//! ~0.19, and so would tapping the right tensor under the wrong norm weight.
//!
//! Swept over both LM-head lowerings and both graph kinds, because the tap is a
//! separate stage in each of four places. An earlier version of this test
//! covered only the raw-weight prefill branch, and reverting the packed branch
//! to post-norm left it green — the real checkpoint runs the packed one.

use rlx_core::backend_matrix::{Failures, available_devices, max_rel_diff};
use rlx_core::flow_util::compile_built;
use rlx_unlimited_ocr::config::UnlimitedOcrConfig;
use rlx_unlimited_ocr::expert_pack::PackedLmWeights;
use rlx_unlimited_ocr::lm_device::CompiledLm;
use rlx_unlimited_ocr::lm_graph::build_unlimited_ocr_prefill_built_from_pack_ext;
use rlx_unlimited_ocr::lm_precision::ResolvedLmPrecision;
use rlx_unlimited_ocr::weights::UnlimitedOcrWeightPrefix;
use std::sync::Arc;

mod tiny;
use tiny::{cfg, fill, synthetic_weights};

const SEQ: usize = 5;

/// `y = x @ w^T` for a `[vocab, hidden]` LM head.
fn host_lm_head(hidden: &[f32], w: &[f32], vocab: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; vocab];
    for v in 0..vocab {
        out[v] = (0..h).map(|i| hidden[i] * w[v * h + i]).sum();
    }
    out
}

/// `model.norm` — RMSNorm, no bias.
fn host_rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

/// The LM-head and final-norm weights, for the host oracle.
fn oracle_weights(cfg: &UnlimitedOcrConfig) -> (Vec<f32>, Vec<f32>) {
    let mut wm = synthetic_weights(cfg);
    let (w, _) = wm
        .take(UnlimitedOcrWeightPrefix::lm_head())
        .expect("lm_head weight");
    let (n, _) = wm
        .take(UnlimitedOcrWeightPrefix::lm_norm())
        .expect("final norm weight");
    (w, n)
}

fn pack_at(cfg: &UnlimitedOcrConfig, prec: ResolvedLmPrecision) -> Arc<PackedLmWeights> {
    let mut wm = synthetic_weights(cfg);
    Arc::new(PackedLmWeights::from_weight_map(&mut wm, cfg.clone(), prec).expect("pack"))
}

/// `F32` takes the raw `FlowStage::LmHead` lowering; `Q8_0` keeps the quantized
/// weight in the IR and takes the plugin. The tap is wired separately in each.
///
/// The oracle multiplies by the *unquantized* head, so the quantized arm needs
/// room for that one difference — it is the only thing that differs between the
/// two sides, since hidden and logits both come from the same run.
const PRECISIONS: [(ResolvedLmPrecision, f32); 2] = [
    (ResolvedLmPrecision::F32, 1e-3),
    (ResolvedLmPrecision::Q8_0, 3e-2),
];

#[test]
fn tapped_hidden_reproduces_the_graphs_own_logits() {
    let cfg: UnlimitedOcrConfig = cfg();
    let h = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let n_layers = cfg.num_hidden_layers;
    let eps = cfg.rms_norm_eps as f32;
    let embeds = fill(SEQ * h, 0.5);
    let (lm_w, norm_w) = oracle_weights(&cfg);

    let mut fails = Failures::default();
    for (prec, tol) in PRECISIONS {
        for (name, device) in available_devices() {
            let pack = pack_at(&cfg, prec);
            let built = build_unlimited_ocr_prefill_built_from_pack_ext(&cfg, &pack, 1, SEQ, true)
                .expect("prefill build");
            let mut compiled = compile_built(built, device).expect("compile");
            let outs = compiled.run(&[("inputs_embeds", embeds.as_slice())]);

            // logits, then n_layers KV pairs, then the hidden — in that order.
            if outs.len() != 2 + 2 * n_layers {
                fails.push(
                    name,
                    format!(
                        "{prec:?}: expected {} outputs, got {}",
                        2 + 2 * n_layers,
                        outs.len()
                    ),
                );
                continue;
            }
            let logits = &outs[0];
            let all_hidden = outs.last().expect("hidden tap");
            // Prefill taps every position so a draft head can be primed over
            // the prompt; the logits are the last position's, so compare that
            // row.
            if all_hidden.len() != SEQ * h {
                fails.push(
                    name,
                    format!(
                        "{prec:?}: hidden is {} elements, want {SEQ}*{h}",
                        all_hidden.len()
                    ),
                );
                continue;
            }
            let hidden = &all_hidden[(SEQ - 1) * h..];
            let want = host_lm_head(&host_rms_norm(hidden, &norm_w, eps), &lm_w, vocab, h);
            let rel = max_rel_diff(logits, &want);
            if !rel.is_finite() || rel >= tol {
                fails.push(
                    name,
                    format!("{prec:?}/prefill: norm(hidden) @ lm_head^T is off by {rel:.5}"),
                );
            }
        }
    }
    fails.assert_empty("hidden-state tap (prefill)");
}

/// The chunked verify pass taps a hidden per position; every one of them has to
/// satisfy the same identity, or a speculative round reseeds its draft head
/// from the wrong row.
#[test]
fn chunked_decode_taps_a_correct_hidden_per_position() {
    let cfg: UnlimitedOcrConfig = cfg();
    let h = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let eps = cfg.rms_norm_eps as f32;
    let (lm_w, norm_w) = oracle_weights(&cfg);
    let embeds = fill(SEQ * h, 0.5);
    let n = SEQ - 1;

    let mut fails = Failures::default();
    for (prec, tol) in PRECISIONS {
        for (name, device) in available_devices() {
            let pack = pack_at(&cfg, prec);
            let mut lm = CompiledLm::new(device, pack);
            let (_, mut kv) = lm.prefill(&embeds[..h], 1).expect("prefill");
            let (per_pos, per_hidden) = lm
                .decode_chunk_with_hidden(&embeds[h..], 1, n, &mut kv)
                .expect("decode_chunk_with_hidden");
            if per_hidden.len() != n {
                fails.push(
                    name,
                    format!(
                        "{prec:?}: {} hidden rows for a {n}-token chunk",
                        per_hidden.len()
                    ),
                );
                continue;
            }
            for i in 0..n {
                let want = host_lm_head(
                    &host_rms_norm(&per_hidden[i], &norm_w, eps),
                    &lm_w,
                    vocab,
                    h,
                );
                let rel = max_rel_diff(&per_pos[i], &want);
                if !rel.is_finite() || rel >= tol {
                    fails.push(
                        name,
                        format!("{prec:?}/chunk position {i}: off by {rel:.5}"),
                    );
                }
            }
        }
    }
    fails.assert_empty("hidden-state tap (chunked decode)");
}

/// Without the flag the output list is unchanged, so existing `1 + 2*i` KV
/// indexing keeps working.
#[test]
fn tap_is_opt_in_and_does_not_shift_existing_outputs() {
    let cfg: UnlimitedOcrConfig = cfg();
    let n_layers = cfg.num_hidden_layers;
    let embeds = fill(SEQ * cfg.hidden_size, 0.5);

    let mut lens = Vec::new();
    for with_hidden in [false, true] {
        let pack = pack_at(&cfg, ResolvedLmPrecision::F32);
        let built =
            build_unlimited_ocr_prefill_built_from_pack_ext(&cfg, &pack, 1, SEQ, with_hidden)
                .expect("build");
        let mut compiled = compile_built(built, rlx_runtime::Device::Cpu).expect("compile");
        let outs = compiled.run(&[("inputs_embeds", embeds.as_slice())]);
        lens.push(outs.len());
    }
    assert_eq!(lens[0], 1 + 2 * n_layers, "default output count changed");
    assert_eq!(
        lens[1],
        2 + 2 * n_layers,
        "tap should add exactly one output"
    );
}

/// The non-pack entry point never requests the tap, so it must still build.
#[test]
fn plain_prefill_build_is_unaffected() {
    let cfg: UnlimitedOcrConfig = cfg();
    let mut wm = synthetic_weights(&cfg);
    assert!(
        rlx_unlimited_ocr::lm_graph::build_unlimited_ocr_prefill_built(&cfg, &mut wm, 1, SEQ)
            .is_ok()
    );
}
