// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Disk-paged experts == resident experts.
//!
//! A tiny `mistral-large4` GGUF is written to a temp dir — real metadata keys,
//! the shipped layer kinds (NoPE, KV sharing, shared indexers, dense layer 0)
//! and **Q8_0** expert banks, so paging moves packed bytes as it does on the
//! real checkpoint. [`PagedMistralLarge4`] then decodes it one token at a time,
//! paging each fired expert from the file, and every step's logits must match
//! the resident prefill graph over the same file.
//!
//! `paged_blocks_match_reference` runs the same machinery on the real
//! range-fetched subset (env-gated, see `tests/real_weights.rs`).

use rlx_core::flow_bridge::PackedWeightLoaderSource;
use rlx_core::flow_util::compile_built;
use rlx_gguf::{GgmlType, GgufWriter, MetaValue};
use rlx_mistral_large4::{
    MistralLarge4Config, PagedMistralLarge4, PagedOptions, PagedStack, PrefillOptions, ShardedGguf,
    build_mistral_large4_text_flow_with_source,
};
use rlx_runtime::Device;
use std::path::PathBuf;

const D: usize = 32;
const HEADS: usize = 2;
const KV: usize = 16;
const NOPE: usize = 8;
const ROPE: usize = 8;
const V: usize = 16; // = NOPE + ROPE
const IDX_H: usize = 2;
const IDX_D: usize = 16;
const E: usize = 6;
const TOPK: usize = 2;
const IM: usize = 32; // Q8_0 needs every bank's inner dim to be a multiple of 32
const INTER: usize = 64;
const VOCAB: usize = 48;
const LAYERS: usize = 8;
const SEQ: usize = 7;

fn dev() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(17);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((((s >> 33) as f32) / (u32::MAX as f32 / 2.0)) - 0.5) * 2.0 * scale
        })
        .collect()
}

fn use_rope(i: usize) -> bool {
    !(i + 1).is_multiple_of(4)
}
const KV_SRC: [usize; LAYERS] = [0, 1, 2, 3, 4, 5, 6, 3];
const IDX_FULL: [bool; LAYERS] = [true, true, false, true, false, true, true, false];

/// Write the checkpoint; returns its path.
fn write_tiny_gguf() -> PathBuf {
    let mut w = GgufWriter::new();
    w.set_arch("mistral-large4");
    let u = |v: usize| MetaValue::U32(v as u32);
    let k = |s: &str| format!("mistral-large4.{s}");
    for (key, v) in [
        ("block_count", LAYERS),
        ("embedding_length", D),
        ("feed_forward_length", INTER),
        ("attention.head_count", HEADS),
        ("attention.head_count_kv", 1),
        ("attention.key_length_mla", NOPE + ROPE),
        ("attention.value_length_mla", V),
        ("attention.kv_lora_rank", KV),
        ("rope.dimension_count", ROPE),
        ("vocab_size", VOCAB),
        ("expert_count", E),
        ("expert_used_count", TOPK),
        ("expert_shared_count", 1),
        ("expert_feed_forward_length", IM),
        ("expert_gating_func", 1),
        ("leading_dense_block_count", 1),
        ("attention.indexer.head_count", IDX_H),
        ("attention.indexer.key_length", IDX_D),
        ("attention.indexer.top_k", 2048),
        ("rope.scaling.original_context_length", 8),
        ("context_length", 4096),
    ] {
        w.set_meta(k(key), u(v));
    }
    for (key, v) in [
        ("rope.freq_base", 10000.0f32),
        ("attention.layer_norm_rms_epsilon", 1e-5),
        ("attention.layer_norm_epsilon", 1e-6),
        ("expert_weights_scale", 1.5),
        ("rope.scaling.factor", 4.0),
        ("rope.scaling.yarn_beta_fast", 32.0),
        ("rope.scaling.yarn_beta_slow", 1.0),
        ("final_logit_softcapping", 5.0),
    ] {
        w.set_meta(k(key), MetaValue::F32(v));
    }
    w.set_meta(k("rope.scaling.type"), MetaValue::String("yarn".into()));
    w.set_meta(k("expert_weights_norm"), MetaValue::Bool(true));
    w.set_meta(
        k("attention.rope_pattern"),
        MetaValue::Array((0..LAYERS).map(|i| MetaValue::Bool(use_rope(i))).collect()),
    );
    w.set_meta(
        k("attention.indexer.types"),
        MetaValue::Array(IDX_FULL.iter().map(|&b| MetaValue::Bool(b)).collect()),
    );
    w.set_meta(
        k("attention.kv_source_map"),
        MetaValue::Array(KV_SRC.iter().map(|&s| MetaValue::U32(s as u32)).collect()),
    );

    let mut seed = 0u64;
    // `shape` in rlx order ([out, in]); GGUF wants it innermost-first.
    let mut put = |w: &mut GgufWriter, name: String, shape: Vec<usize>, scale: f32, q8: bool| {
        seed += 1;
        let n: usize = shape.iter().product();
        let data = fill(n, seed, scale);
        let ggml: Vec<usize> = shape.iter().rev().copied().collect();
        let (dtype, bytes) = if q8 {
            (
                GgmlType::Q8_0,
                rlx_gguf::quantize(&data, GgmlType::Q8_0).unwrap(),
            )
        } else {
            (
                GgmlType::F32,
                data.iter().flat_map(|x| x.to_le_bytes()).collect(),
            )
        };
        w.add_tensor_bytes(name, ggml, dtype, bytes).unwrap();
    };
    let ones = |w: &mut GgufWriter, name: String, n: usize| {
        let b = vec![1.0f32; n]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        w.add_tensor_bytes(name, vec![n], GgmlType::F32, b).unwrap();
    };
    let lin = |n: usize| (n as f32).powf(-0.5);
    put(
        &mut w,
        "token_embd.weight".into(),
        vec![VOCAB, D],
        1.0,
        false,
    );
    put(
        &mut w,
        "output.weight".into(),
        vec![VOCAB, D],
        lin(D),
        false,
    );
    ones(&mut w, "output_norm.weight".into(), D);
    // `exp(-alpha)` with alpha small: around 1, never 0 (it divides).
    let rf: Vec<u8> = fill(ROPE / 2, 99, 0.2)
        .iter()
        .flat_map(|a| (-a).exp().to_le_bytes())
        .collect();
    w.add_tensor_bytes("rope_freqs.weight", vec![ROPE / 2], GgmlType::F32, rf)
        .unwrap();
    let qk = NOPE + ROPE;
    for i in 0..LAYERS {
        let b = format!("blk.{i}");
        let rope_l = if use_rope(i) { ROPE } else { 0 };
        ones(&mut w, format!("{b}.attn_norm.weight"), D);
        ones(&mut w, format!("{b}.ffn_norm.weight"), D);
        put(
            &mut w,
            format!("{b}.attn_q.weight"),
            vec![HEADS * qk, D],
            lin(D),
            false,
        );
        if KV_SRC[i] == i {
            put(
                &mut w,
                format!("{b}.attn_kv_a_mqa.weight"),
                vec![KV + rope_l, D],
                lin(D),
                false,
            );
            ones(&mut w, format!("{b}.attn_kv_a_norm.weight"), KV);
        }
        put(
            &mut w,
            format!("{b}.attn_k_b.weight"),
            vec![HEADS, KV, qk - rope_l],
            lin(KV),
            false,
        );
        put(
            &mut w,
            format!("{b}.attn_v_b.weight"),
            vec![HEADS, V, KV],
            lin(KV),
            false,
        );
        put(
            &mut w,
            format!("{b}.attn_output.weight"),
            vec![D, HEADS * V],
            lin(HEADS * V),
            false,
        );
        put(
            &mut w,
            format!("{b}.attn_ssmax.weight"),
            vec![HEADS],
            0.3,
            false,
        );
        ones(&mut w, format!("{b}.attn_ssmax.bias"), HEADS);
        if IDX_FULL[i] {
            let p = format!("{b}.indexer");
            ones(&mut w, format!("{p}.q_norm.weight"), D);
            put(&mut w, format!("{p}.q_norm.bias"), vec![D], 0.1, false);
            put(
                &mut w,
                format!("{p}.attn_q_b.weight"),
                vec![IDX_H * IDX_D, D],
                lin(D),
                false,
            );
            put(
                &mut w,
                format!("{p}.attn_k.weight"),
                vec![IDX_D, D],
                lin(D),
                false,
            );
            ones(&mut w, format!("{p}.k_norm.weight"), IDX_D);
            put(&mut w, format!("{p}.k_norm.bias"), vec![IDX_D], 0.1, false);
            put(
                &mut w,
                format!("{p}.proj.weight"),
                vec![IDX_H, D],
                lin(D),
                false,
            );
            put(&mut w, format!("{p}.ssmax.weight"), vec![IDX_H], 0.3, false);
            ones(&mut w, format!("{p}.ssmax.bias"), IDX_H);
        }
        if i == 0 {
            put(
                &mut w,
                format!("{b}.ffn_gate.weight"),
                vec![INTER, D],
                lin(D),
                false,
            );
            put(
                &mut w,
                format!("{b}.ffn_up.weight"),
                vec![INTER, D],
                lin(D),
                false,
            );
            put(
                &mut w,
                format!("{b}.ffn_down.weight"),
                vec![D, INTER],
                lin(INTER),
                false,
            );
        } else {
            put(
                &mut w,
                format!("{b}.ffn_gate_inp.weight"),
                vec![E, D],
                1.0,
                false,
            );
            put(
                &mut w,
                format!("{b}.ffn_gate_exps.weight"),
                vec![E, IM, D],
                lin(D),
                true,
            );
            put(
                &mut w,
                format!("{b}.ffn_up_exps.weight"),
                vec![E, IM, D],
                lin(D),
                true,
            );
            put(
                &mut w,
                format!("{b}.ffn_down_exps.weight"),
                vec![E, D, IM],
                lin(IM),
                true,
            );
            put(
                &mut w,
                format!("{b}.ffn_gate_shexp.weight"),
                vec![IM, D],
                lin(D),
                false,
            );
            put(
                &mut w,
                format!("{b}.ffn_up_shexp.weight"),
                vec![IM, D],
                lin(D),
                false,
            );
            put(
                &mut w,
                format!("{b}.ffn_down_shexp.weight"),
                vec![D, IM],
                lin(IM),
                false,
            );
        }
    }
    // One directory per call: tests run in parallel and each removes its own.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("rlx-ml4-paged-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.gguf");
    w.write_to_path(&path).unwrap();
    path
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn paged_decode_matches_resident_prefill() {
    let path = write_tiny_gguf();
    let cfg = MistralLarge4Config::from_gguf_path(&path).unwrap();
    assert_eq!(cfg.n_routed_experts, E);
    let ids: Vec<u32> = [3u32, 17, 5, 40, 22, 9, 31][..SEQ].to_vec();

    // Resident: the whole prefill graph, banks taken packed from the same file.
    let resident = {
        let mut loader = rlx_core::weight_loader::load_from_path(path.to_str().unwrap()).unwrap();
        let built = build_mistral_large4_text_flow_with_source(
            &cfg,
            &mut PackedWeightLoaderSource(loader.as_mut()),
            SEQ,
            PrefillOptions::default(),
        )
        .unwrap();
        let mut c = compile_built(built, dev()).unwrap();
        let f: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
        c.run(&[("input_ids", &f[..])]).remove(0)
    };

    // Paged: one token per step, experts read from disk with no cache.
    let opts = PagedOptions {
        device: dev(),
        cap: 16,
        cache_bytes: 0,
    };
    let mut m = PagedMistralLarge4::open(&path, &opts).unwrap();
    let mut worst = 0f32;
    for (t, &id) in ids.iter().enumerate() {
        let logits = m.step(id).unwrap();
        let want = &resident[t * VOCAB..(t + 1) * VOCAB];
        worst = worst.max(max_abs(&logits, want));
    }
    let st = m.stack().pager_stats();
    eprintln!(
        "paged vs resident: max |Δ| = {worst:.3e}; pager {st:?}; whole-bank uploads {}",
        m.stack().whole_bank_uploads()
    );
    // Same weights and kernels, prefill vs one-token reduction order: ~1e-6.
    // A wrong slot, bank or routing weight is O(0.1).
    assert!(
        worst < 1e-4,
        "paged decode differs from resident prefill by {worst}"
    );
    // 7 MoE layers × 2 fired experts × 3 banks per token, all off disk.
    assert_eq!(st.hits, 0, "a zero budget must not cache");
    assert!(st.misses as usize >= SEQ * 7 * TOPK * 3);

    // Same again from a reset context, with a cache big enough to hold every
    // expert: identical logits, and now repeats are hits.
    let opts = PagedOptions {
        cache_bytes: 1 << 30,
        ..opts
    };
    let mut m = PagedMistralLarge4::open(&path, &opts).unwrap();
    for _ in 0..2 {
        m.reset();
        let mut worst = 0f32;
        for (t, &id) in ids.iter().enumerate() {
            let logits = m.step(id).unwrap();
            worst = worst.max(max_abs(&logits, &resident[t * VOCAB..(t + 1) * VOCAB]));
        }
        assert!(worst < 1e-4, "cached paging differs by {worst}");
    }
    assert!(m.stack().pager_stats().hits > 0);
    // `RLX_ML4_TINY_OUT=FILE` keeps the checkpoint, e.g. to drive the CLI.
    if let Some(out) = rlx_ir::env::read("RLX_ML4_TINY_OUT")
        .ok()
        .filter(|s| !s.is_empty())
    {
        std::fs::copy(&path, out).unwrap();
    }
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn paged_blocks_match_reference() {
    // `python3 scripts/mistral_large4_reference.py --gguf SUBSET --topk 2048 DIR`
    let Some(dir) = rlx_ir::env::read("RLX_ML4_REAL_DENSE_DIR")
        .ok()
        .filter(|s| !s.is_empty())
    else {
        return;
    };
    let j: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(PathBuf::from(&dir).join("ref_real.json")).unwrap(),
    )
    .unwrap();
    let gguf = j["gguf"].as_str().unwrap();
    let floats = |k: &str| -> Vec<f32> {
        j[k].as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };
    let (input, want) = (floats("hidden_in"), floats("hidden_out"));
    let blocks: Vec<usize> = j["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let seq = j["seq"].as_u64().unwrap() as usize;
    let cfg = MistralLarge4Config::from_gguf_path(gguf).unwrap();
    assert!(
        cfg.dsa_is_dense(seq),
        "generate this reference with --topk 2048"
    );
    let mut shards = ShardedGguf::open(gguf).unwrap();
    let opts = PagedOptions {
        device: dev(),
        cap: seq,
        cache_bytes: 0,
    };
    let mut stack = PagedStack::new(&cfg, &mut shards, &blocks, &opts).unwrap();
    let d = cfg.hidden_size;
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut worst = 0f32;
    for t in 0..seq {
        let y = stack.step(input[t * d..(t + 1) * d].to_vec()).unwrap();
        worst = worst.max(max_abs(&y, &want[t * d..(t + 1) * d]) / scale);
        if t == 0 {
            // The first step pays one-off compilation; time the rest.
            stack.reset_times();
        }
    }
    eprintln!(
        "paged real blocks {blocks:?}: rel max |Δ| = {worst:.3e}; {:?}; {:?}",
        stack.pager_stats(),
        stack.times()
    );
    assert!(
        worst < 1e-3,
        "paged real blocks differ from the reference by {worst}"
    );
}

/// The generic LM path reaches the paged model: the header sniff resolves the
/// architecture, the registry builds the runner, and every `LmRunner` entry
/// point agrees with stepping the model directly.
#[test]
fn generic_runner_matches_paged_model() {
    use rlx_core::lm_registry::{LmBuildOptions, build_lm_runner};
    let path = write_tiny_gguf();
    let sniff = rlx_cli::auto_dispatch::auto_sniff(&path).unwrap();
    assert_eq!(sniff.runner_name, rlx_mistral_large4::RUNNER);
    rlx_mistral_large4::register();
    let opts = LmBuildOptions {
        device: Some(dev()),
        max_seq: Some(32),
        ..Default::default()
    };
    // A directory resolves to the GGUF inside it.
    let mut r = build_lm_runner(sniff.runner_name, path.parent().unwrap(), &opts).unwrap();
    assert_eq!(r.family(), "mistral-large4");
    assert_eq!(r.vocab_size(), VOCAB);

    // Reference: the model stepped by hand, greedy.
    let prompt = [3u32, 17, 5];
    let mut m = PagedMistralLarge4::open(
        &path,
        &PagedOptions {
            device: dev(),
            cap: 32,
            cache_bytes: 0,
        },
    )
    .unwrap();
    let mut want = Vec::new();
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = m.step(t).unwrap();
    }
    let prompt_logits = logits.clone();
    for _ in 0..5 {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        want.push(next);
        logits = m.step(next).unwrap();
    }

    let got = r.generate(&prompt, 5, &mut |_| true).unwrap();
    assert_eq!(got, want, "generate");
    // predict_logits on the prompt resets; on prompt + more, it only feeds the suffix.
    let p = r.predict_logits(&prompt).unwrap();
    assert!(max_abs(&p, &prompt_logits) < 1e-5);
    let mut longer = prompt.to_vec();
    longer.extend(&want[..2]);
    let a = r.predict_logits(&longer).unwrap();
    // Host-driven pair: same logits as feeding the tokens one by one.
    let b0 = r.prefill_logits(&prompt).unwrap();
    assert!(max_abs(&b0, &prompt_logits) < 1e-5);
    r.decode_logits(want[0]).unwrap();
    let b = r.decode_logits(want[1]).unwrap();
    assert!(max_abs(&a, &b) < 1e-5, "prefix reuse vs host-driven decode");
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}
