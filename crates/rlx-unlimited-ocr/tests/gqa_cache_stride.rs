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

//! The KV cache row width is `num_kv_heads * head_dim`, **not** `hidden_size`.
//!
//! Those two are equal whenever `num_kv_heads == num_attention_heads`, which is
//! true of jina-ocr-v1 and of every config the rest of this suite uses — so
//! code that conflated them ran green. Under GQA the cached row is narrower,
//! and using `hidden_size` as the stride mis-sizes every cache buffer. In
//! practice prefill's own length check catches it, so a grouped-query
//! checkpoint errored out rather than answering wrongly — but nothing in the
//! suite covered GQA at all, so which of the two it would be was untested.
//!
//! The oracle needs no reference implementation. Prefilling `n` tokens and
//! prefilling `n-1` then decoding the last one are two routes to the same
//! logits, and only the second one touches the cache. Any stride error shows up
//! as a divergence between them.

use rlx_core::backend_matrix::{Failures, available_devices, max_rel_diff};
use rlx_runtime::Device;
use rlx_unlimited_ocr::config::UnlimitedOcrConfig;
use rlx_unlimited_ocr::expert_pack::PackedLmWeights;
use rlx_unlimited_ocr::lm_device::CompiledLm;
use rlx_unlimited_ocr::lm_precision::ResolvedLmPrecision;
use std::sync::Arc;

mod tiny;
use tiny::{cfg, cfg_gqa, fill, synthetic_weights};

const SEQ: usize = 6;
const TOL: f32 = 2e-3;

fn pack(cfg: &UnlimitedOcrConfig) -> Arc<PackedLmWeights> {
    let mut wm = synthetic_weights(cfg);
    Arc::new(
        PackedLmWeights::from_weight_map(&mut wm, cfg.clone(), ResolvedLmPrecision::F32)
            .expect("pack"),
    )
}

/// Prefill all `SEQ` tokens; last-position logits.
fn all_at_once(cfg: &UnlimitedOcrConfig, p: &Arc<PackedLmWeights>, d: Device) -> Vec<f32> {
    let embeds = fill(SEQ * cfg.hidden_size, 0.5);
    let mut lm = CompiledLm::new(d, Arc::clone(p));
    lm.prefill(&embeds, SEQ).expect("prefill").0
}

/// Prefill `SEQ - 1`, then decode the last token through the cache.
fn prefill_then_step(cfg: &UnlimitedOcrConfig, p: &Arc<PackedLmWeights>, d: Device) -> Vec<f32> {
    let h = cfg.hidden_size;
    let embeds = fill(SEQ * h, 0.5);
    let mut lm = CompiledLm::new(d, Arc::clone(p));
    let (_, mut kv) = lm
        .prefill(&embeds[..(SEQ - 1) * h], SEQ - 1)
        .expect("prefill");
    lm.decode_step(&embeds[(SEQ - 1) * h..], SEQ - 1, &mut kv)
        .expect("decode")
}

/// Prefill one token, then feed the rest as a single verify-style chunk.
fn prefill_then_chunk(cfg: &UnlimitedOcrConfig, p: &Arc<PackedLmWeights>, d: Device) -> Vec<f32> {
    let h = cfg.hidden_size;
    let embeds = fill(SEQ * h, 0.5);
    let mut lm = CompiledLm::new(d, Arc::clone(p));
    let (_, mut kv) = lm.prefill(&embeds[..h], 1).expect("prefill");
    let n = SEQ - 1;
    let per_pos = lm
        .decode_chunk(&embeds[h..], 1, n, &mut kv)
        .expect("decode_chunk");
    per_pos[n - 1].clone()
}

#[test]
fn cached_decode_matches_full_prefill_under_gqa() {
    let mut fails = Failures::default();
    // Both configs, so the test also proves it is not GQA-specific breakage.
    for (label, cfg) in [("mha-4:4", cfg()), ("gqa-4:2", cfg_gqa())] {
        assert_eq!(
            cfg.kv_hidden(),
            cfg.num_key_value_heads * cfg.head_dim(),
            "kv_hidden definition"
        );
        for (name, device) in available_devices() {
            let p = pack(&cfg);
            let want = all_at_once(&cfg, &p, device);
            for (route, got) in [
                ("decode_step", prefill_then_step(&cfg, &p, device)),
                ("decode_chunk", prefill_then_chunk(&cfg, &p, device)),
            ] {
                let rel = max_rel_diff(&want, &got);
                if !rel.is_finite() || rel >= TOL {
                    fails.push(name, format!("{label}/{route}: rel diff {rel:.5}"));
                }
            }
        }
    }
    fails.assert_empty("GQA KV cache stride");
}

/// The two configs must not be the same model, or the GQA arm proves nothing.
#[test]
fn the_gqa_config_really_is_narrower() {
    let mha = cfg();
    let gqa = cfg_gqa();
    assert_eq!(mha.kv_hidden(), mha.hidden_size, "MHA: widths coincide");
    assert!(
        gqa.kv_hidden() < gqa.hidden_size,
        "GQA cache row ({}) should be narrower than hidden ({})",
        gqa.kv_hidden(),
        gqa.hidden_size
    );
    assert_eq!(gqa.kv_group_size(), 2);
}
