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

//! Checks that need the real 6.7 GB checkpoint.
//!
//! ```bash
//! rlx-jina-ocr --download            # or: huggingface-cli download jinaai/jina-ocr-v1
//! cargo test -p rlx-jina-ocr --test hf_parity -- --nocapture
//! ```
//!
//! Every test here skips (not fails) when the checkpoint is absent, so a clone
//! without 6.7 GB of weights still runs a green suite —
//! `tests/checkpoint_inventory.rs` covers the architecture offline.

use anyhow::Result;
use rlx_jina_ocr::config::{JinaOcrConfig, NGRAM_SIZE, NGRAM_WHITELIST, NGRAM_WINDOW};
use rlx_jina_ocr::fixtures::{require_model_dir, require_probe_image};
use rlx_jina_ocr::preprocess::{ImageMode, image_token_count_for, preprocess_path};
use rlx_jina_ocr::runner::{JinaOcrRunner, default_sample_opts};
use rlx_runtime::Device;
use std::path::PathBuf;

/// `Some(dir)` when the checkpoint is available, else print + skip.
fn model_dir() -> Option<PathBuf> {
    match require_model_dir() {
        Some(dir) => Some(dir),
        None => {
            eprintln!(
                "[hf_parity] skipped — set RLX_JINA_OCR_DIR or run `rlx-jina-ocr --download` \
                 (need config.json + tokenizer.json + safetensors shards)"
            );
            None
        }
    }
}

#[test]
fn published_config_resolves_to_full_causal_attention() -> Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let cfg = JinaOcrConfig::from_model_dir(&dir)?;
    cfg.validate()?;

    assert_eq!(cfg.lm.model_type, JinaOcrConfig::MODEL_TYPE);
    assert_eq!(
        cfg.lm.sliding_window, 0,
        "jina-ocr-v1 has no sliding window"
    );
    assert_eq!(cfg.lm.rope_theta, 1_000_000.0);
    assert_eq!(cfg.lm.hidden_size, 1280);
    assert_eq!(cfg.lm.num_hidden_layers, 12);
    assert_eq!(cfg.lm.n_routed_experts, 64);
    assert_eq!(cfg.lm.vocab_size, 129_280);
    assert!(cfg.processor.crop_mode);
    assert_eq!(cfg.processor.base_size, 1024);
    assert_eq!(cfg.processor.image_size, 640);
    Ok(())
}

#[test]
fn checkpoint_carries_every_tensor_the_runner_reads() -> Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let runner = JinaOcrRunner::open(&dir, Device::Cpu)?;
    let store = runner.store();
    let cfg = runner.config();

    for key in [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.image_newline",
        "model.view_seperator",
        "model.projector.layers.weight",
        "model.projector.layers.bias",
    ] {
        assert!(store.contains(key), "missing {key}");
    }
    for layer in 0..cfg.lm.num_hidden_layers {
        let expected = if cfg.lm.is_dense_layer(layer) {
            0
        } else {
            cfg.lm.n_routed_experts
        };
        assert_eq!(
            store.count_experts(layer),
            expected,
            "layer {layer} experts"
        );
        assert_eq!(
            store.is_moe_layer(layer),
            !cfg.lm.is_dense_layer(layer),
            "layer {layer} router"
        );
    }
    assert!(rlx_jina_ocr::mtp::checkpoint_has_mtp(store));
    eprintln!("[hf_parity] {} tensors", store.keys().len());
    Ok(())
}

/// The prompt must carry exactly the placeholder run the vision pack fills, and
/// must not start with BOS.
#[test]
fn prompt_ids_match_the_processor_layout() -> Result<()> {
    let (Some(dir), Some(img)) = (model_dir(), require_probe_image()) else {
        return Ok(());
    };
    let mut runner = JinaOcrRunner::open(&dir, Device::Cpu)?;
    let mode = runner.default_image_mode();
    assert!(matches!(mode, ImageMode::Gundam { .. }));

    let pre = preprocess_path(&img, mode)?;
    let n_image = image_token_count_for(runner.config(), &pre);
    let ids = runner.build_prompt_ids(rlx_jina_ocr::DEFAULT_OCR_PROMPT, &pre)?;

    let image_token_id = runner.config().image_token_id();
    let placed = ids.iter().filter(|&&t| t == image_token_id).count();
    assert_eq!(placed, n_image, "placeholder run length");
    assert_ne!(
        ids[0],
        rlx_jina_ocr::config::BOS_TOKEN_ID,
        "jina's processor calls text_encode(bos=False)"
    );

    // Global view (q=16 -> 273) plus the tile grid, in that order.
    let [w, h] = pre.spatial_crop;
    let expected = 16 * 17
        + 1
        + if pre.has_tiles() {
            (10 * h as usize) * (10 * w as usize + 1)
        } else {
            0
        };
    assert_eq!(n_image, expected, "grid {w}x{h}");
    eprintln!(
        "[hf_parity] prompt={} ids, {n_image} image tokens, grid {w}x{h}",
        ids.len()
    );
    Ok(())
}

#[test]
fn fastmtp_head_loads_from_the_checkpoint() -> Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let runner = JinaOcrRunner::open(&dir, Device::Cpu)?;
    let head = rlx_jina_ocr::MtpHead::load(runner.config(), runner.store())?
        .expect("checkpoint ships a FastMTP head");
    assert_eq!(head.hidden_size(), runner.config().lm.hidden_size);
    assert_eq!(head.steps(), runner.config().mtp.num_speculative_steps);
    Ok(())
}

/// One greedy token end to end: vision tower, projector, prefill, LM head.
/// Slow (loads ~6.7 GB); `--ignored` keeps it out of the default run.
#[test]
#[ignore = "loads the full 6.7 GB checkpoint"]
fn first_token_greedy_smoke() -> Result<()> {
    let (Some(dir), Some(img)) = (model_dir(), require_probe_image()) else {
        return Ok(());
    };
    let mut runner = JinaOcrRunner::open(&dir, Device::Cpu)?;
    let pre = preprocess_path(&img, runner.default_image_mode())?;
    let mut opts = default_sample_opts();
    opts.max_new_tokens = 1;

    let out = runner.generate(&pre, rlx_jina_ocr::DEFAULT_OCR_PROMPT, &opts)?;
    assert_eq!(out.new_tokens, 1);
    let first = out.token_ids[out.prompt_len];
    assert!((first as usize) < runner.config().lm.vocab_size);
    eprintln!("[hf_parity] first token={first} text={:?}", out.raw_text);
    Ok(())
}

/// Full transcription. Very slow on CPU; run with `--ignored` and a GPU device.
#[test]
#[ignore = "full 4k-token transcription"]
fn transcribe_sample_page() -> Result<()> {
    let (Some(dir), Some(img)) = (model_dir(), require_probe_image()) else {
        return Ok(());
    };
    let device = rlx_jina_ocr::resolve_device(None).unwrap_or(Device::Cpu);
    let mut session = rlx_jina_ocr::JinaOcrSession::open(
        &dir,
        rlx_jina_ocr::InferenceOptions::for_ocr()
            .device(device)
            .max_new_tokens(2048),
    )?;
    let result = session.run_single(&img)?;
    eprintln!(
        "[hf_parity] {} new tokens\n{}",
        result.new_tokens, result.markdown
    );
    assert!(!result.markdown.trim().is_empty(), "empty transcript");
    Ok(())
}

/// The decode recipe the model card documents.
#[test]
fn sample_defaults_match_the_reference_processor() {
    let opts = default_sample_opts();
    assert_eq!(opts.no_repeat_ngram_size, NGRAM_SIZE);
    assert_eq!(opts.ngram_window, NGRAM_WINDOW);
    assert_eq!(opts.ngram_whitelist, NGRAM_WHITELIST.to_vec());
    assert_eq!(opts.temperature, 0.0, "greedy");
}
