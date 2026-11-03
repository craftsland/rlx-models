// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The checkpoint contract, from the real checkpoint and nothing hand-written:
//!
//! * `mistral_large4_config.json` — `mistralai/Mistral-Large-4-1T-A52B`'s
//!   `config.json`, verbatim;
//! * `mistral_large4_meta.gguf` — the published GGUF's metadata, header only;
//! * `mistral_large4_tensors.tsv` — all 1602 tensors of `UD-IQ1_S`.
//!
//! The last two come from `scripts/mistral_large4_manifest.py` (range reads).

use rlx_mistral_large4::MistralLarge4Config;
use std::collections::{BTreeMap, BTreeSet};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn hf() -> MistralLarge4Config {
    MistralLarge4Config::from_hf_file(format!("{FIX}/mistral_large4_config.json")).unwrap()
}

fn gguf() -> MistralLarge4Config {
    MistralLarge4Config::from_gguf_path(format!("{FIX}/mistral_large4_meta.gguf")).unwrap()
}

/// name → GGML dims (innermost first).
fn manifest() -> BTreeMap<String, Vec<usize>> {
    std::fs::read_to_string(format!("{FIX}/mistral_large4_tensors.tsv"))
        .unwrap()
        .lines()
        .map(|l| {
            let mut it = l.split('\t');
            let name = it.next().unwrap().to_string();
            let dims = it
                .next()
                .unwrap()
                .split(',')
                .map(|d| d.parse().unwrap())
                .collect();
            (name, dims)
        })
        .collect()
}

#[test]
fn hf_config_reads_the_shipped_schedules() {
    let c = hf();
    assert_eq!(c.num_hidden_layers, 72);
    assert_eq!(
        (c.hidden_size, c.vocab_size, c.num_attention_heads),
        (6144, 204800, 64)
    );
    assert_eq!(
        (
            c.qk_head_dim(),
            c.qk_rope_head_dim,
            c.v_head_dim,
            c.kv_lora_rank
        ),
        (128, 64, 128, 512)
    );
    // NoPE every 4th layer.
    for i in 0..72 {
        assert_eq!(c.use_rope[i], (i + 1) % 4 != 0, "layer {i}");
    }
    // Layers 39, 43, … 71 share layer 35's KV; 63 layers own a cache.
    let sharers: Vec<usize> = (0..72).filter(|&i| !c.owns_kv(i)).collect();
    assert_eq!(sharers, (39..72).step_by(4).collect::<Vec<_>>());
    assert!(sharers.iter().all(|&i| c.kv_source[i] == 35));
    assert_eq!(c.indexer_full.iter().filter(|&&f| f).count(), 44);
    assert!(!c.moe_layers[0] && c.moe_layers[1..].iter().all(|&m| m));
    assert_eq!(c.final_logit_softcapping, Some(30.0));
    assert_eq!(
        (c.n_routed_experts, c.num_experts_per_tok, c.index_topk),
        (128, 4, 2048)
    );
    // Widths that differ by layer kind.
    assert_eq!((c.nope_dim(0), c.nope_dim(3)), (64, 128));
    assert_eq!((c.latent_width(0), c.latent_width(35)), (576, 512));
}

#[test]
fn gguf_metadata_agrees_with_config_json() {
    let (h, g) = (hf(), gguf());
    assert_eq!(h.use_rope, g.use_rope);
    assert_eq!(h.indexer_full, g.indexer_full);
    assert_eq!(h.kv_source, g.kv_source);
    assert_eq!(h.moe_layers, g.moe_layers);
    assert_eq!(
        (
            h.qk_nope_head_dim,
            h.qk_rope_head_dim,
            h.v_head_dim,
            h.kv_lora_rank
        ),
        (
            g.qk_nope_head_dim,
            g.qk_rope_head_dim,
            g.v_head_dim,
            g.kv_lora_rank
        )
    );
    assert_eq!(
        (h.index_n_heads, h.index_head_dim, h.index_topk),
        (g.index_n_heads, g.index_head_dim, g.index_topk)
    );
    assert_eq!(
        (
            h.n_routed_experts,
            h.num_experts_per_tok,
            h.moe_intermediate_size,
            h.intermediate_size
        ),
        (
            g.n_routed_experts,
            g.num_experts_per_tok,
            g.moe_intermediate_size,
            g.intermediate_size
        )
    );
    assert_eq!(h.final_logit_softcapping, g.final_logit_softcapping);
    assert!((h.rms_norm_eps - g.rms_norm_eps).abs() < 1e-9);
    assert!((h.indexer_norm_eps - g.indexer_norm_eps).abs() < 1e-9);
    // The YaRN tables both readers produce are the same table.
    let (a, b) = (
        rlx_mistral_large4::rope::base_inv_freq(&h),
        rlx_mistral_large4::rope::base_inv_freq(&g),
    );
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() <= 1e-12 * x.abs(), "{x} vs {y}");
    }
    // Interpolated at the low end (factor 2048), extrapolated at the high end.
    let plain0 = 1.0f64;
    assert!((a[0] - plain0).abs() < 1e-12, "highest frequency is kept");
    let plain_last = 10000f64.powf(-62.0 / 64.0);
    assert!(
        (plain_last / a[31] - 2048.0).abs() < 1e-6,
        "lowest is divided by the factor"
    );
}

#[test]
fn tensor_manifest_matches_the_published_gguf_both_ways() {
    let c = gguf();
    let want: BTreeSet<String> = c.tensor_manifest().into_iter().collect();
    let have: BTreeSet<String> = manifest().into_keys().collect();
    let missing: Vec<_> = want.difference(&have).collect();
    let unread: Vec<_> = have.difference(&want).collect();
    assert!(missing.is_empty(), "read but not in the GGUF: {missing:?}");
    assert!(unread.is_empty(), "in the GGUF but never read: {unread:?}");
    assert_eq!(have.len(), 1602);
}

#[test]
fn tensor_shapes_match_the_emitters() {
    let c = gguf();
    let m = manifest();
    let (d, h, kv, vd) = (
        c.hidden_size,
        c.num_attention_heads,
        c.kv_lora_rank,
        c.v_head_dim,
    );
    let (nh, ihd) = (c.index_n_heads, c.index_head_dim);
    let (e, im) = (c.n_routed_experts, c.moe_intermediate_size);
    let at = |n: String| m.get(&n).unwrap_or_else(|| panic!("{n} missing")).clone();
    assert_eq!(at("rope_freqs.weight".into()), vec![c.qk_rope_head_dim / 2]);
    assert_eq!(at("token_embd.weight".into()), vec![d, c.vocab_size]);
    for i in 0..c.num_hidden_layers {
        let b = format!("blk.{i}");
        // GGML dims are innermost first: [in, out] for a projection.
        assert_eq!(
            at(format!("{b}.attn_q.weight")),
            vec![d, h * c.qk_head_dim()]
        );
        assert_eq!(
            at(format!("{b}.attn_k_b.weight")),
            vec![c.nope_dim(i), kv, h],
            "{b}"
        );
        assert_eq!(at(format!("{b}.attn_v_b.weight")), vec![kv, vd, h]);
        assert_eq!(at(format!("{b}.attn_output.weight")), vec![h * vd, d]);
        assert_eq!(at(format!("{b}.attn_ssmax.weight")), vec![h]);
        if c.owns_kv(i) {
            assert_eq!(
                at(format!("{b}.attn_kv_a_mqa.weight")),
                vec![d, c.latent_width(i)],
                "{b}"
            );
        }
        if c.indexer_full[i] {
            assert_eq!(
                at(format!("{b}.indexer.attn_q_b.weight")),
                vec![d, nh * ihd]
            );
            assert_eq!(at(format!("{b}.indexer.attn_k.weight")), vec![d, ihd]);
            assert_eq!(at(format!("{b}.indexer.proj.weight")), vec![d, nh]);
            assert_eq!(at(format!("{b}.indexer.ssmax.weight")), vec![nh]);
        }
        if c.moe_layers[i] {
            assert_eq!(at(format!("{b}.ffn_gate_inp.weight")), vec![d, e]);
            // An expert bank's expert count is the LAST GGML dim.
            assert_eq!(at(format!("{b}.ffn_gate_exps.weight")), vec![d, im, e]);
            assert_eq!(at(format!("{b}.ffn_up_exps.weight")), vec![d, im, e]);
            assert_eq!(at(format!("{b}.ffn_down_exps.weight")), vec![im, d, e]);
            assert_eq!(at(format!("{b}.ffn_down_shexp.weight")), vec![im, d]);
        } else {
            assert_eq!(
                at(format!("{b}.ffn_gate.weight")),
                vec![d, c.intermediate_size]
            );
        }
    }
}
