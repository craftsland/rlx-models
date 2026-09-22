// RLX — versatile ML compiler + runtime. GPLv3.
//! The CLI against a real checkpoint directory.
//!
//! `cli::run` is the only part of the stack a user actually types, and it is the
//! part unit tests never reach: argument parsing, checkpoint opening, the runner,
//! the tokenizer and the output path all have to line up at once.

use rlx_core::dsv41_weights::write_synthetic_checkpoint;
use serde_json::{Value, json};
use std::path::PathBuf;

fn toy_config(vocab: usize) -> Value {
    json!({
        "model_type": "deepseek_v41",
        "vocab_size": vocab,
        "hidden_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2,
        "head_dim": 16, "qk_rope_head_dim": 8, "q_lora_rank": 8,
        "o_groups": 1, "o_lora_rank": 4, "hc_mult": 2,
        "compress_ratios": [0, 0, 2, 2],
        "kv_source_layer_ids": [2], "index_source_layer_ids": [2],
        "index_head_dim": 8, "index_n_heads": 1, "index_topk": 2,
        "sliding_window": 8,
        "moe_intermediate_size": 8, "n_routed_experts": 4,
        "num_experts_per_tok": 2, "n_shared_experts": 1,
        "scoring_func": "sigmoid", "norm_topk_prob": true,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "compress_rope_theta": 10000.0,
    })
}

/// A word-level tokenizer whose ids are exactly `w1`..`w{vocab-1}`.
fn tokenizer_json(vocab: usize) -> String {
    let mut map = serde_json::Map::new();
    map.insert("<unk>".into(), json!(0));
    for i in 1..vocab {
        map.insert(format!("w{i}"), json!(i));
    }
    serde_json::to_string(&json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false},
        "model": {"type": "WordLevel", "vocab": Value::Object(map), "unk_token": "<unk>"}
    }))
    .unwrap()
}

fn checkpoint(tag: &str) -> PathBuf {
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
    let dir = PathBuf::from(base).join(format!("rlx_dsv41_cli_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = toy_config(32);
    write_synthetic_checkpoint(&dir, &cfg).expect("checkpoint written");
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json(32)).unwrap();
    dir
}

fn argv(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| s.to_string()).collect()
}

/// The command a user actually types, end to end.
#[test]
fn generates_from_a_text_prompt() {
    let dir = checkpoint("text");
    let args = argv(&[
        "--model",
        dir.to_str().unwrap(),
        "--prompt",
        "w3 w7 w11",
        "--max-tokens",
        "3",
    ]);
    rlx_dsv41::cli::run(&args).expect("generation succeeds");
}

/// Ids in, ids out — the path that needs no tokenizer.
#[test]
fn generates_from_prompt_ids() {
    let dir = checkpoint("ids");
    let args = argv(&[
        "--model",
        dir.to_str().unwrap(),
        "--prompt-ids",
        "5, 9, 2",
        "--max-tokens",
        "3",
        "--ids",
    ]);
    rlx_dsv41::cli::run(&args).expect("generation succeeds");
}

/// `--paged` must be a memory strategy, not a different model. A budget of 64 KiB
/// is far below one layer's bank, so this only passes if paging works.
#[test]
fn paged_runs_under_a_tight_budget() {
    let dir = checkpoint("paged");
    let args = argv(&[
        "--model",
        dir.to_str().unwrap(),
        "--prompt-ids",
        "5,9,2,14",
        "--max-tokens",
        "3",
        "--paged",
        "--expert-budget",
        "64K",
        "--stats",
    ]);
    rlx_dsv41::cli::run(&args).expect("paged generation succeeds");
}

/// A directory that is not a checkpoint fails with the path in the message,
/// rather than a panic from somewhere inside the loader.
#[test]
fn a_bad_model_path_names_itself() {
    let args = argv(&["--model", "/definitely/not/here", "--prompt-ids", "1"]);
    let e = rlx_dsv41::cli::run(&args).unwrap_err().to_string();
    assert!(e.contains("/definitely/not/here"), "unhelpful error: {e}");
}

/// Sampling flags are wired through, and a seeded run stays on rails.
#[test]
fn sampling_flags_are_accepted() {
    let dir = checkpoint("sample");
    let args = argv(&[
        "--model",
        dir.to_str().unwrap(),
        "--prompt-ids",
        "5,9",
        "--max-tokens",
        "4",
        "--temperature",
        "0.8",
        "--top-k",
        "8",
        "--top-p",
        "0.95",
        "--seed",
        "7",
    ]);
    rlx_dsv41::cli::run(&args).expect("sampled generation succeeds");
}
