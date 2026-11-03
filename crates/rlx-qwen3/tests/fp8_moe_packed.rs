// RLX models — Qwen3.
// Copyright (C) 2026 Eugene Hauptmann.
// SPDX-License-Identifier: GPL-3.0-only
//! A Qwen3-MoE checkpoint in the Qwen3-30B-A3B-FP8 layout — attention and
//! every expert linear `F8_E4M3` with a 128×128 bf16 `weight_scale_inv` grid,
//! one tensor per expert, router and norms unquantized — generates the same
//! tokens through the native packed generator (experts as block-FP8
//! `DequantGroupedMatMul` banks) as through the f32 generator that dequantizes
//! everything at load.

use rlx_qwen3::{Qwen3Config, Qwen3Generator};
use rlx_runtime::Device;
use safetensors::tensor::{Dtype, TensorView};

const VOCAB: usize = 512;
const HIDDEN: usize = 256;
const EXPERT: usize = 256;
const EXPERTS: usize = 4;
const LAYERS: usize = 2;

fn unit(i: usize, salt: usize) -> f32 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (salt as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x % 20_001) as f32 / 10_000.0 - 1.0
}

fn bf16(v: f32) -> [u8; 2] {
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

/// `(name, dtype, shape, bytes)` for the whole checkpoint.
fn tensors() -> Vec<(String, Dtype, Vec<usize>, Vec<u8>)> {
    let mut out = Vec::new();
    let salt = std::cell::Cell::new(0usize);
    let dense = |out: &mut Vec<_>, name: String, shape: Vec<usize>, base: f32, spread: f32| {
        salt.set(salt.get() + 1);
        let salt = salt.get();
        let n: usize = shape.iter().product();
        let bytes = (0..n)
            .flat_map(|i| bf16(base + spread * unit(i, salt)))
            .collect();
        out.push((name, Dtype::BF16, shape, bytes));
    };
    let fp8 = |out: &mut Vec<_>, name: String, o: usize, i: usize| {
        salt.set(salt.get() + 1);
        let salt = salt.get();
        let codes = (0..o * i)
            .map(|t| rlx_ir::quant::ScaledFormat::F8E4M3.encode(400.0 * unit(t, salt)))
            .collect();
        let grid = (0..(o / 128) * (i / 128))
            .flat_map(|t| bf16(0.0002 * (1.0 + (t % 5) as f32)))
            .collect();
        out.push((name.clone(), Dtype::F8_E4M3, vec![o, i], codes));
        out.push((
            format!("{name}_scale_inv"),
            Dtype::BF16,
            vec![o / 128, i / 128],
            grid,
        ));
    };
    dense(
        &mut out,
        "model.embed_tokens.weight".into(),
        vec![VOCAB, HIDDEN],
        0.0,
        0.5,
    );
    dense(&mut out, "model.norm.weight".into(), vec![HIDDEN], 1.0, 0.1);
    dense(
        &mut out,
        "lm_head.weight".into(),
        vec![VOCAB, HIDDEN],
        0.0,
        0.1,
    );
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        dense(
            &mut out,
            format!("{p}.input_layernorm.weight"),
            vec![HIDDEN],
            1.0,
            0.1,
        );
        dense(
            &mut out,
            format!("{p}.post_attention_layernorm.weight"),
            vec![HIDDEN],
            1.0,
            0.1,
        );
        dense(
            &mut out,
            format!("{p}.self_attn.q_norm.weight"),
            vec![64],
            1.0,
            0.1,
        );
        dense(
            &mut out,
            format!("{p}.self_attn.k_norm.weight"),
            vec![64],
            1.0,
            0.1,
        );
        for (proj, o, i) in [
            ("q", 256, HIDDEN),
            ("k", 128, HIDDEN),
            ("v", 128, HIDDEN),
            ("o", HIDDEN, 256),
        ] {
            fp8(&mut out, format!("{p}.self_attn.{proj}_proj.weight"), o, i);
        }
        dense(
            &mut out,
            format!("{p}.mlp.gate.weight"),
            vec![EXPERTS, HIDDEN],
            0.0,
            0.5,
        );
        for e in 0..EXPERTS {
            let q = format!("{p}.mlp.experts.{e}");
            fp8(&mut out, format!("{q}.gate_proj.weight"), EXPERT, HIDDEN);
            fp8(&mut out, format!("{q}.up_proj.weight"), EXPERT, HIDDEN);
            fp8(&mut out, format!("{q}.down_proj.weight"), HIDDEN, EXPERT);
        }
    }
    out
}

const CONFIG: &str = r#"{
  "model_type": "qwen3_moe", "vocab_size": 512, "hidden_size": 256,
  "intermediate_size": 512, "moe_intermediate_size": 256,
  "num_experts": 4, "num_experts_per_tok": 2, "norm_topk_prob": true,
  "decoder_sparse_step": 1, "mlp_only_layers": [],
  "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
  "head_dim": 64, "max_position_embeddings": 256, "rms_norm_eps": 1e-6,
  "rope_theta": 10000.0, "tie_word_embeddings": false, "hidden_act": "silu",
  "quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
    "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
}"#;

fn greedy(g: &mut Qwen3Generator, prompt: &[u32], n: usize) -> (Vec<f32>, Vec<u32>) {
    let first = g.prefill_get_last_logits(prompt).unwrap();
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };
    let mut toks = vec![argmax(&first)];
    while toks.len() < n {
        let logits = g.decode_get_logits(*toks.last().unwrap()).unwrap();
        toks.push(argmax(&logits));
    }
    (first, toks)
}

#[test]
fn fp8_moe_packed_matches_the_dequantized_model() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), CONFIG).unwrap();
    let data = tensors();
    let views: Vec<(String, TensorView<'_>)> = data
        .iter()
        .map(|(n, dt, s, b)| (n.clone(), TensorView::new(*dt, s.clone(), b).unwrap()))
        .collect();
    let file = dir.path().join("model.safetensors");
    safetensors::serialize_to_file(views, None, &file).unwrap();
    let path = file.to_str().unwrap();
    let cfg = Qwen3Config::from_json_str(CONFIG).unwrap();
    assert!(cfg.is_moe());

    let prompt = [5u32, 77, 300, 12, 90, 41];
    let mut f32_gen = Qwen3Generator::from_path(cfg.clone(), path, Device::Cpu).unwrap();
    let (want_logits, want) = greedy(&mut f32_gen, &prompt, 8);
    let mut packed = Qwen3Generator::new_native_packed_decode(cfg, path, Device::Cpu).unwrap();
    let (got_logits, got) = greedy(&mut packed, &prompt, 8);

    let peak = want_logits.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = want_logits
        .iter()
        .zip(&got_logits)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        err <= 1e-4 * peak.max(1.0),
        "prompt logits differ by {err} (peak {peak})"
    );
    assert_eq!(
        got, want,
        "packed MoE decode diverged from the dequantized model"
    );
}
