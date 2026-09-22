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

//! Checks this crate's view of `jinaai/jina-ocr-v1` against the **published
//! checkpoint's own safetensors headers**, without downloading 6.7 GB.
//!
//! `tests/fixtures/jina_ocr_checkpoint_inventory.json` was read from the two
//! shards over HTTP range requests: 8-byte header length, then the header JSON,
//! which carries every tensor's dtype and shape. Numeric path segments are
//! collapsed to `{}` with their index sets recorded, so the full 2 722-key set
//! is reconstructed here.
//!
//! This is the test that would have caught a mis-derived topology — a wrong
//! expert count, a transposed projector, an MoE draft block — at the point
//! where it is a one-line fix rather than a garbage transcript.

use rlx_jina_ocr::config::JinaOcrConfig;
use std::collections::BTreeMap;

const INVENTORY: &str = include_str!("fixtures/jina_ocr_checkpoint_inventory.json");

/// `key -> shape`, reconstructed from the collapsed fixture.
fn checkpoint_tensors() -> BTreeMap<String, Vec<usize>> {
    let doc: serde_json::Value = serde_json::from_str(INVENTORY).expect("parse inventory fixture");
    let mut out = BTreeMap::new();
    for entry in doc["tensors"].as_array().expect("tensors array") {
        let pattern = entry["pattern"].as_str().expect("pattern");
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|v| v.as_u64().expect("dim") as usize)
            .collect();
        assert_eq!(entry["dtype"], "BF16", "{pattern} is not bf16");

        let axes: Vec<Vec<usize>> = entry
            .get("axes")
            .and_then(|a| a.as_array())
            .map(|axes| {
                axes.iter()
                    .map(|axis| {
                        if let Some(r) = axis.get("range") {
                            let lo = r[0].as_u64().unwrap() as usize;
                            let hi = r[1].as_u64().unwrap() as usize;
                            (lo..=hi).collect()
                        } else {
                            axis["values"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_u64().unwrap() as usize)
                                .collect()
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut expanded = 0usize;
        for combo in cartesian(&axes) {
            let mut key = String::with_capacity(pattern.len() + 8);
            let mut idx = combo.iter();
            for (i, seg) in pattern.split('.').enumerate() {
                if i > 0 {
                    key.push('.');
                }
                if seg == "{}" {
                    key.push_str(&idx.next().expect("index for placeholder").to_string());
                } else {
                    key.push_str(seg);
                }
            }
            assert!(out.insert(key, shape.clone()).is_none(), "duplicate key");
            expanded += 1;
        }
        assert_eq!(
            expanded,
            entry["count"].as_u64().expect("count") as usize,
            "{pattern}: expanded key count disagrees with the recorded count"
        );
    }
    out
}

fn cartesian(axes: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut acc = vec![Vec::new()];
    for axis in axes {
        let mut next = Vec::with_capacity(acc.len() * axis.len());
        for prefix in &acc {
            for &v in axis {
                let mut row = prefix.clone();
                row.push(v);
                next.push(row);
            }
        }
        acc = next;
    }
    acc
}

fn config() -> JinaOcrConfig {
    // The crate's own parse of the published `config.json`.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/jina_ocr_config.json");
    JinaOcrConfig::from_file(&path).expect("parse bundled config fixture")
}

fn assert_shape(t: &BTreeMap<String, Vec<usize>>, key: &str, want: &[usize]) {
    let got = t
        .get(key)
        .unwrap_or_else(|| panic!("checkpoint has no tensor {key}"));
    assert_eq!(got.as_slice(), want, "{key} shape");
}

#[test]
fn inventory_reconstructs_the_published_tensor_set() {
    let doc: serde_json::Value = serde_json::from_str(INVENTORY).unwrap();
    let tensors = checkpoint_tensors();
    assert_eq!(doc["model_id"], JinaOcrConfig::HF_MODEL_ID);
    assert_eq!(
        tensors.len(),
        doc["total_tensors"].as_u64().unwrap() as usize
    );
    assert_eq!(tensors.len(), 2722);
    assert_eq!(doc["total_size_bytes"].as_u64().unwrap(), 6_744_476_160);
}

#[test]
fn decoder_tensor_shapes_match_the_config() {
    let cfg = config();
    let t = checkpoint_tensors();
    let h = cfg.lm.hidden_size;
    let v = cfg.lm.vocab_size;

    assert_shape(&t, "model.embed_tokens.weight", &[v, h]);
    assert_shape(&t, "lm_head.weight", &[v, h]);
    assert_shape(&t, "model.norm.weight", &[h]);
    assert_shape(&t, "model.image_newline", &[h]);
    assert_shape(&t, "model.view_seperator", &[h]);

    for i in 0..cfg.lm.num_hidden_layers {
        assert_shape(
            &t,
            &format!("model.layers.{i}.input_layernorm.weight"),
            &[h],
        );
        assert_shape(
            &t,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
            &[h],
        );
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            assert_shape(
                &t,
                &format!("model.layers.{i}.self_attn.{proj}.weight"),
                &[h, h],
            );
        }
    }
}

/// `first_k_dense_replace = 1`: layer 0 is a plain SwiGLU MLP with no router,
/// every later layer is MoE. Getting this boundary wrong loads a router that
/// does not exist (or skips one that does).
#[test]
fn dense_then_moe_layer_split_matches_first_k_dense_replace() {
    let cfg = config();
    let t = checkpoint_tensors();
    let h = cfg.lm.hidden_size;
    let inter = cfg.lm.intermediate_size;
    let moe_inter = cfg.lm.moe_intermediate_size;

    for i in 0..cfg.lm.num_hidden_layers {
        let router = format!("model.layers.{i}.mlp.gate.weight");
        if cfg.lm.is_dense_layer(i) {
            assert!(!t.contains_key(&router), "dense layer {i} has a router");
            assert_shape(
                &t,
                &format!("model.layers.{i}.mlp.gate_proj.weight"),
                &[inter, h],
            );
            assert_shape(
                &t,
                &format!("model.layers.{i}.mlp.up_proj.weight"),
                &[inter, h],
            );
            assert_shape(
                &t,
                &format!("model.layers.{i}.mlp.down_proj.weight"),
                &[h, inter],
            );
        } else {
            assert_shape(&t, &router, &[cfg.lm.n_routed_experts, h]);
            // Shared experts are fused into one MLP of `n_shared * moe_inter`.
            let shared = cfg.lm.n_shared_experts * moe_inter;
            assert_shape(
                &t,
                &format!("model.layers.{i}.mlp.shared_experts.gate_proj.weight"),
                &[shared, h],
            );
            assert_shape(
                &t,
                &format!("model.layers.{i}.mlp.shared_experts.down_proj.weight"),
                &[h, shared],
            );
            for e in 0..cfg.lm.n_routed_experts {
                let p = format!("model.layers.{i}.mlp.experts.{e}");
                assert_shape(&t, &format!("{p}.gate_proj.weight"), &[moe_inter, h]);
                assert_shape(&t, &format!("{p}.up_proj.weight"), &[moe_inter, h]);
                assert_shape(&t, &format!("{p}.down_proj.weight"), &[h, moe_inter]);
            }
        }
    }
}

#[test]
fn deep_encoder_and_projector_shapes_match_the_config() {
    let cfg = config();
    let t = checkpoint_tensors();
    let sam = &cfg.lm.vision_config.sam;
    let clip = &cfg.lm.vision_config.clip;

    // SAM-ViT-B: 768-wide, 12 blocks, 16 px patches over a 1024 px view.
    assert_shape(
        &t,
        "model.sam_model.patch_embed.proj.weight",
        &[sam.hidden_size, 3, sam.patch_size, sam.patch_size],
    );
    let grid = sam.image_size / sam.patch_size;
    assert_shape(
        &t,
        "model.sam_model.pos_embed",
        &[1, grid, grid, sam.hidden_size],
    );
    for i in 0..sam.num_hidden_layers {
        assert_shape(
            &t,
            &format!("model.sam_model.blocks.{i}.attn.qkv.weight"),
            &[3 * sam.hidden_size, sam.hidden_size],
        );
    }
    // Neck 768 -> 256, then net_2/net_3 halve the grid twice into 1024 ch.
    assert_shape(
        &t,
        "model.sam_model.neck.0.weight",
        &[sam.out_chans, sam.hidden_size, 1, 1],
    );
    assert_shape(
        &t,
        "model.sam_model.net_2.weight",
        &[sam.downsample_channels[0], sam.out_chans, 3, 3],
    );
    assert_shape(
        &t,
        "model.sam_model.net_3.weight",
        &[sam.downsample_channels[1], sam.downsample_channels[0], 3, 3],
    );

    // CLIP-L/14-224: 1024-wide, 24 layers, 256 patches + CLS.
    assert_shape(
        &t,
        "model.vision_model.embeddings.patch_embedding.weight",
        &[clip.hidden_size, 3, clip.patch_size, clip.patch_size],
    );
    let patches = (clip.image_size / clip.patch_size).pow(2);
    assert_shape(
        &t,
        "model.vision_model.embeddings.position_embedding.weight",
        &[patches + 1, clip.hidden_size],
    );
    for i in 0..clip.num_hidden_layers {
        let p = format!("model.vision_model.transformer.layers.{i}");
        assert_shape(
            &t,
            &format!("{p}.self_attn.qkv_proj.weight"),
            &[3 * clip.hidden_size, clip.hidden_size],
        );
        assert_shape(
            &t,
            &format!("{p}.mlp.fc1.weight"),
            &[clip.intermediate_size, clip.hidden_size],
        );
    }

    // The projector consumes concat(CLIP, SAM) = 1024 + 1024.
    assert_eq!(
        cfg.lm.projector.input_dim,
        clip.hidden_size + sam.downsample_channels[1]
    );
    assert_shape(
        &t,
        "model.projector.layers.weight",
        &[cfg.lm.projector.n_embed, cfg.lm.projector.input_dim],
    );
    assert_shape(
        &t,
        "model.projector.layers.bias",
        &[cfg.lm.projector.n_embed],
    );
}

/// SAM's decomposed relative-position tables are sized `2 * span - 1`: the four
/// `global_attn_indexes` blocks span the full 64-wide grid, the rest a 14-wide
/// window. This is the only place in the checkpoint where the window size and
/// the global-block indices are observable.
#[test]
fn sam_relative_position_tables_reveal_window_and_global_blocks() {
    let cfg = config();
    let t = checkpoint_tensors();
    let sam = &cfg.lm.vision_config.sam;
    let head_dim = sam.hidden_size / sam.num_attention_heads;
    let grid = sam.image_size / sam.patch_size;

    for i in 0..sam.num_hidden_layers {
        let is_global = sam.global_attn_indexes.contains(&i);
        let span = if is_global { grid } else { sam.window_size };
        for axis in ["rel_pos_h", "rel_pos_w"] {
            assert_shape(
                &t,
                &format!("model.sam_model.blocks.{i}.attn.{axis}"),
                &[2 * span - 1, head_dim],
            );
        }
    }
}

/// Every tensor name [`rlx_jina_ocr::mtp`] reads, with the shapes that pin the
/// draft block as **dense** (`6848`, not `896`) and its input as the
/// **concatenated pair** (`2 * 1280`).
#[test]
fn fastmtp_head_shapes_match_the_config() {
    let cfg = config();
    let t = checkpoint_tensors();
    let h = cfg.lm.hidden_size;
    let p = rlx_jina_ocr::mtp::head_prefix(0);

    assert!(cfg.mtp.is_enabled());
    assert_shape(&t, &format!("{p}enorm.weight"), &[h]);
    assert_shape(&t, &format!("{p}hnorm.weight"), &[h]);
    assert_shape(&t, &format!("{p}eh_proj.weight"), &[h, 2 * h]);
    assert_shape(&t, &format!("{p}mtp_block.input_layernorm.weight"), &[h]);
    assert_shape(
        &t,
        &format!("{p}mtp_block.post_attention_layernorm.weight"),
        &[h],
    );
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        assert_shape(
            &t,
            &format!("{p}mtp_block.self_attn.{proj}.weight"),
            &[h, h],
        );
    }
    assert!(!cfg.mtp.moe, "config claims an MoE draft block");
    assert_shape(
        &t,
        &format!("{p}mtp_block.mlp.gate_proj.weight"),
        &[cfg.lm.intermediate_size, h],
    );
    assert_shape(
        &t,
        &format!("{p}mtp_block.mlp.down_proj.weight"),
        &[h, cfg.lm.intermediate_size],
    );

    // Only one head ships, and `mtp_num_heads` must not promise more.
    assert!(!t.contains_key(&format!(
        "{}enorm.weight",
        rlx_jina_ocr::mtp::head_prefix(1)
    )));
    assert_eq!(cfg.mtp.num_heads, 1);
}

/// `mtp_share_*` are not optional: the checkpoint simply has no `shared_head`
/// tensors, so a head built without sharing would read zeros.
#[test]
fn fastmtp_shares_norm_and_lm_head_because_none_are_stored() {
    let cfg = config();
    let t = checkpoint_tensors();
    assert!(
        !t.keys().any(|k| k.contains("shared_head")),
        "checkpoint unexpectedly ships shared_head tensors"
    );
    assert!(!t.keys().any(|k| k.starts_with("mtp_embed_tokens")));
    assert!(cfg.mtp.share_norm && cfg.mtp.share_lm_head && cfg.mtp.share_embedding_weights);
}

/// Nothing outside the four known families, so no component is silently unread.
#[test]
fn every_checkpoint_tensor_belongs_to_a_known_component() {
    let t = checkpoint_tensors();
    let known = [
        "model.sam_model.",
        "model.vision_model.",
        "model.projector.",
        "model.layers.",
        "mtp_module.",
    ];
    let singletons = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.image_newline",
        "model.view_seperator",
        "lm_head.weight",
    ];
    for key in t.keys() {
        let ok = known.iter().any(|p| key.starts_with(p)) || singletons.contains(&key.as_str());
        assert!(ok, "unaccounted tensor {key}");
    }
}
