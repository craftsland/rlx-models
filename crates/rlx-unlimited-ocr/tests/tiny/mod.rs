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

//! A tiny DeepSeek-OCR-shaped decoder with synthetic weights.
//!
//! Small enough to compile and run per test, but keeps every structural
//! feature the real checkpoint has: a dense layer 0 then MoE layers, routed +
//! shared experts, GQA-capable head counts, and jina-ocr-v1's full-causal
//! `sliding_window = 0` with `rope_theta = 1e6`.

#![allow(dead_code)]

use rlx_core::weight_map::WeightMap;
use rlx_unlimited_ocr::config::{
    ClipTowerConfig, ProjectorConfig, SamTowerConfig, UnlimitedOcrConfig, UnlimitedOcrVisionConfig,
};
use rlx_unlimited_ocr::expert_pack::pack_experts_in_map;
use rlx_unlimited_ocr::weights::UnlimitedOcrWeightPrefix;
use std::collections::HashMap;

pub fn cfg() -> UnlimitedOcrConfig {
    UnlimitedOcrConfig {
        model_type: "deepseek_vl_v2".into(),
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        n_routed_experts: 4,
        n_shared_experts: 2,
        num_experts_per_tok: 2,
        moe_intermediate_size: 32,
        intermediate_size: 64,
        first_k_dense_replace: 1,
        vocab_size: 128,
        max_position_embeddings: 1024,
        // The whole point of this file.
        sliding_window: 0,
        use_mla: false,
        rms_norm_eps: 1e-6,
        // jina-ocr-v1's base, not Unlimited-OCR's 10_000.
        rope_theta: 1_000_000.0,
        hidden_act: "silu".into(),
        bos_token_id: 0,
        eos_token_id: 1,
        pad_token_id: 2,
        image_token_id: 128_815,
        v_head_dim: Some(16),
        vision_config: UnlimitedOcrVisionConfig {
            sam: SamTowerConfig::default(),
            clip: ClipTowerConfig::default(),
            image_size: 1024,
        },
        projector: ProjectorConfig {
            input_dim: 2048,
            n_embed: 64,
            projector_type: "linear".into(),
        },
        patch_size: 16,
        downsample_ratio: 4,
    }
}

/// Same model with grouped-query attention: 4 query heads sharing 2 K/V heads.
///
/// Exists because `num_kv_heads == num_heads` makes the cache row width equal
/// `hidden_size` by coincidence, which hides any code that conflates the two.
pub fn cfg_gqa() -> UnlimitedOcrConfig {
    UnlimitedOcrConfig {
        num_key_value_heads: 2,
        ..cfg()
    }
}

pub fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.017 + seed).sin()) * 0.02)
        .collect()
}

pub fn synthetic_weights(cfg: &UnlimitedOcrConfig) -> WeightMap {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let ff = cfg.intermediate_size;
    let moe_ff = cfg.moe_intermediate_size;
    let n_e = cfg.n_routed_experts;
    let shared_ff = moe_ff * cfg.n_shared_experts;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    t.insert(
        UnlimitedOcrWeightPrefix::embed_tokens().into(),
        (fill(v * h, 0.1), vec![v, h]),
    );
    t.insert(
        UnlimitedOcrWeightPrefix::lm_norm().into(),
        (fill(h, 0.2), vec![h]),
    );
    t.insert(
        UnlimitedOcrWeightPrefix::lm_head().into(),
        (fill(v * h, 0.3), vec![v, h]),
    );
    for layer in 0..cfg.num_hidden_layers {
        t.insert(
            UnlimitedOcrWeightPrefix::lm_input_layernorm(layer),
            (fill(h, 1.0 + layer as f32), vec![h]),
        );
        t.insert(
            UnlimitedOcrWeightPrefix::lm_post_attention_layernorm(layer),
            (fill(h, 2.0 + layer as f32), vec![h]),
        );
        for (pi, proj) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
            // K and V project to the *pre-repeat* width, which is narrower than
            // `hidden_size` under GQA. `o_proj` consumes the post-repeat width,
            // so it stays square.
            let rows = match *proj {
                "k_proj" | "v_proj" => cfg.kv_hidden(),
                _ => h,
            };
            t.insert(
                UnlimitedOcrWeightPrefix::lm_attn(layer, proj),
                (fill(rows * h, 3.0 + pi as f32), vec![rows, h]),
            );
        }
        if cfg.is_dense_layer(layer) {
            for (pi, proj) in ["gate_proj", "up_proj"].iter().enumerate() {
                t.insert(
                    UnlimitedOcrWeightPrefix::lm_dense_mlp(layer, proj),
                    (fill(ff * h, 4.0 + pi as f32), vec![ff, h]),
                );
            }
            t.insert(
                UnlimitedOcrWeightPrefix::lm_dense_mlp(layer, "down_proj"),
                (fill(h * ff, 4.5), vec![h, ff]),
            );
        } else {
            t.insert(
                UnlimitedOcrWeightPrefix::lm_moe_gate(layer),
                (fill(n_e * h, 5.0), vec![n_e, h]),
            );
            for (pi, proj) in ["gate_proj", "up_proj"].iter().enumerate() {
                t.insert(
                    UnlimitedOcrWeightPrefix::lm_moe_shared_expert(layer, proj),
                    (fill(shared_ff * h, 6.0 + pi as f32), vec![shared_ff, h]),
                );
            }
            t.insert(
                UnlimitedOcrWeightPrefix::lm_moe_shared_expert(layer, "down_proj"),
                (fill(h * shared_ff, 6.5), vec![h, shared_ff]),
            );
            for e in 0..n_e {
                for (pi, proj) in ["gate_proj", "up_proj"].iter().enumerate() {
                    t.insert(
                        UnlimitedOcrWeightPrefix::lm_moe_expert(layer, e, proj),
                        (
                            fill(moe_ff * h, 7.0 + e as f32 + pi as f32 * 0.1),
                            vec![moe_ff, h],
                        ),
                    );
                }
                t.insert(
                    UnlimitedOcrWeightPrefix::lm_moe_expert(layer, e, "down_proj"),
                    (fill(h * moe_ff, 8.0 + e as f32), vec![h, moe_ff]),
                );
            }
        }
    }

    let mut map = WeightMap::from_tensors(t);
    for layer in 0..cfg.num_hidden_layers {
        if !cfg.is_dense_layer(layer) {
            pack_experts_in_map(&mut map, layer, n_e, h, moe_ff).expect("pack experts");
        }
    }
    map
}
