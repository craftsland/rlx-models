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

//! [`jinaai/jina-ocr-v1`](https://huggingface.co/jinaai/jina-ocr-v1) — a
//! DeepSeek-OCR derivative: SAM-ViT-B + CLIP-L/14-224 "DeepEncoder", a linear
//! `2048 -> 1280` projector, and a 3B / 570M-active DeepSeek-V2 MoE decoder
//! (12 layers, 64 routed + 2 shared experts, top-6), plus a FastMTP draft head.
//!
//! ## Relationship to [`rlx_unlimited_ocr`]
//!
//! `baidu/Unlimited-OCR` is the same architecture under the same tensor names,
//! so the vision towers, projector, expert packing and compiled MoE decoder are
//! reused wholesale rather than duplicated. This crate contributes what is
//! actually different about jina-ocr-v1:
//!
//! | | Unlimited-OCR | jina-ocr-v1 |
//! |---|---|---|
//! | attention | rolling window, 128 | **full causal** (no `sliding_window` key) |
//! | `rope_theta` | 10 000 (default) | **1 000 000** |
//! | prompt | `<image>document parsing.` + BOS | `<\|User\|>:` template, **no BOS** |
//! | tiling `max_num` | 32 | **9** |
//! | n-gram guard | 35 / 128 | 35 / **1024** + `<td>` whitelist |
//! | draft head | — | **FastMTP**, K=3 recursive |
//!
//! Losing any one of those is silent — wrong attention span, wrong RoPE base or
//! a stray BOS all still produce fluent-looking text — so each is pinned by a
//! test in its own module.
//!
//! ## Components
//!
//! 1. **Config** — [`config`] (`config.json` + `processor_config.json`)
//! 2. **Preprocessing** — [`preprocess`] (Gundam global view + up to 9 tiles)
//! 3. **Prompt** — [`prompt`] (chat template, BOS-less id assembly)
//! 4. **Decode** — [`runner`] (shared DeepEncoder + MoE LM)
//! 5. **Post-processing** — [`postprocess`] (`decode_ocr`, ref/det → markdown)
//! 6. **Speculation** — [`mtp`] (FastMTP draft head; see its docs for status)
//! 7. **Session / CLI** — [`infer`], [`cli`]

pub mod cli;
pub mod config;
pub mod fixtures;
pub mod hub;
pub mod infer;
pub mod mtp;
pub mod postprocess;
pub mod preprocess;
pub mod prompt;
pub mod runner;

#[cfg(feature = "hf-download")]
pub mod download;

#[cfg(feature = "tokenizer")]
pub mod tokenizer;

pub use config::{
    BASE_SIZE, BOS_TOKEN_ID, EOS_TOKEN_ID, IMAGE_TOKEN_ID, JinaOcrConfig, MtpConfig, PAD_TOKEN_ID,
    ProcessorConfig, TILE_SIZE,
};
pub use infer::{InferenceOptions, JinaOcrSession, OcrResult};
pub use mtp::{MtpHead, MtpKvCache, MtpWeights};
pub use postprocess::{Crop, Ref, clean_decoded_text, extract_markdown_and_crops, parse_refs};
pub use preprocess::{ImageMode, load_image, preprocess_one, preprocess_path};
pub use prompt::{Content, DEFAULT_OCR_PROMPT, Message, Role, apply_chat_template};
pub use runner::{JinaOcrRunner, OcrOutput, default_sample_opts};

// Re-exported so callers do not need a direct dependency on the shared crate.
pub use rlx_unlimited_ocr::device::{pick_auto_device, resolve_device};
pub use rlx_unlimited_ocr::generation::SampleOpts;
pub use rlx_unlimited_ocr::lm_precision::LmWeightPrecision;
pub use rlx_unlimited_ocr::preprocess::PreprocessedImage;
pub use rlx_unlimited_ocr::runner::RunnerOptions;
pub use rlx_unlimited_ocr::weights::UnlimitedOcrWeightStore as JinaOcrWeightStore;

#[cfg(feature = "tokenizer")]
pub use tokenizer::JinaTokenizer;
