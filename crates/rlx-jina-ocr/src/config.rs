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

//! `jinaai/jina-ocr-v1` configuration — HuggingFace `config.json` +
//! `processor_config.json`.
//!
//! The checkpoint is a DeepSeek-OCR derivative: `model_type =
//! "deepseek_vl_v2"`, a SAM-ViT-B + CLIP-L/14-224 DeepEncoder, a linear
//! `2048 -> 1280` projector and a 12-layer DeepSeek-V2 MoE decoder, under the
//! exact tensor names [`rlx_unlimited_ocr`] already reads. So the decoder
//! config is expressed as an [`UnlimitedOcrConfig`], with two differences that
//! this module pins explicitly rather than inheriting:
//!
//! * **No sliding window.** `jina-ocr-v1`'s `config.json` has no
//!   `sliding_window` key, and `modeling_deepseekv2.py` reads it as
//!   `getattr(self, "sliding_window", None)` — i.e. plain causal attention over
//!   the whole history. Unlimited-OCR's own card *does* set `128`, so
//!   [`UnlimitedOcrConfig`]'s fallback for a missing key is `128`; leaving that
//!   in place would silently truncate attention to the last 128 tokens.
//! * **`rope_theta = 1e6`,** where Unlimited-OCR omits the key and falls back
//!   to `10_000`. This one *is* present in jina's JSON and parses correctly;
//!   it is asserted in tests because getting it wrong is silent.
//!
//! On top of the decoder, the card carries a FastMTP speculative head
//! ([`MtpConfig`]) whose weights ship in the same safetensors shards.

use anyhow::{Context, Result, ensure};
use rlx_unlimited_ocr::config::UnlimitedOcrConfig;
use serde::Deserialize;
use std::path::Path;

/// Vision placeholder token id (`config.json: image_token_index`).
pub const IMAGE_TOKEN_ID: u32 = 128_815;
/// `<｜begin▁of▁sentence｜>`.
pub const BOS_TOKEN_ID: u32 = 0;
/// `<｜end▁of▁sentence｜>`.
pub const EOS_TOKEN_ID: u32 = 1;
/// `<｜▁pad▁｜>`.
pub const PAD_TOKEN_ID: u32 = 2;

/// `<td>` / `</td>` — whitelisted out of the no-repeat-n-gram guard so long
/// tables are not truncated (`_DEFAULT_NGRAM_WHITELIST` in
/// `modeling_deepseekocr.py`).
pub const NGRAM_WHITELIST: [u32; 2] = [128_821, 128_822];
/// `_DEFAULT_NGRAM_SIZE` in `modeling_deepseekocr.py`.
pub const NGRAM_SIZE: usize = 35;
/// `_DEFAULT_NGRAM_WINDOW` in `modeling_deepseekocr.py`.
pub const NGRAM_WINDOW: usize = 1024;

/// `processor_config.json: base_size` — global-view side length.
pub const BASE_SIZE: u32 = 1024;
/// `processor_config.json: image_size` — local-tile side length.
pub const TILE_SIZE: u32 = 640;
/// `dynamic_preprocess(min_num=2, ...)`.
pub const DYNAMIC_MIN_NUM: u32 = 2;
/// `dynamic_preprocess(max_num=9, ...)` — jina's default, where Unlimited-OCR
/// uses 32. Caps the Gundam tile grid at 9 tiles.
pub const DYNAMIC_MAX_NUM: u32 = 9;

/// FastMTP speculative-decoding head (`mtp_*` / `num_nextn_predict_layers`).
///
/// One dense draft block is applied recursively for `num_speculative_steps`
/// draft tokens, re-feeding its own output hidden state each step. Embedding,
/// final norm and LM head are shared with the target model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpConfig {
    /// `mtp_num_heads` — distinct draft blocks in the checkpoint.
    pub num_heads: usize,
    /// `num_nextn_predict_layers` — draft blocks the predictor instantiates.
    pub num_nextn_predict_layers: usize,
    /// `mtp_num_speculative_steps` — draft depth K per verification round.
    pub num_speculative_steps: usize,
    /// `mtp_recursive` — reuse one head for all K steps, feeding its own
    /// hidden state forward (as opposed to re-grounding on the target's).
    pub recursive: bool,
    /// `mtp_share_embedding_weights`.
    pub share_embedding_weights: bool,
    /// `mtp_share_lm_head`.
    pub share_lm_head: bool,
    /// `mtp_share_norm`.
    pub share_norm: bool,
    /// `mtp_moe` — draft block uses MoE (false: dense SwiGLU MLP).
    pub moe: bool,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            num_heads: 1,
            num_nextn_predict_layers: 1,
            num_speculative_steps: 3,
            recursive: true,
            share_embedding_weights: true,
            share_lm_head: true,
            share_norm: true,
            moe: false,
        }
    }
}

impl MtpConfig {
    /// Whether the checkpoint carries a usable FastMTP draft head.
    pub fn is_enabled(&self) -> bool {
        self.num_heads > 0 && self.num_speculative_steps > 0
    }

    fn from_raw(raw: &RawJinaOcrConfig) -> Self {
        let d = Self::default();
        Self {
            num_heads: raw.mtp_num_heads.unwrap_or(d.num_heads),
            num_nextn_predict_layers: raw
                .num_nextn_predict_layers
                .unwrap_or(d.num_nextn_predict_layers),
            num_speculative_steps: raw
                .mtp_num_speculative_steps
                .unwrap_or(d.num_speculative_steps),
            recursive: raw.mtp_recursive.unwrap_or(d.recursive),
            share_embedding_weights: raw
                .mtp_share_embedding_weights
                .unwrap_or(d.share_embedding_weights),
            share_lm_head: raw.mtp_share_lm_head.unwrap_or(d.share_lm_head),
            share_norm: raw.mtp_share_norm.unwrap_or(d.share_norm),
            moe: raw.mtp_moe.unwrap_or(d.moe),
        }
    }
}

/// Image-processor settings (`processor_config.json`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessorConfig {
    /// Global-view side length in Gundam mode (`base_size`).
    pub base_size: u32,
    /// Local-tile side length, or the single view size when `crop_mode` is off.
    pub image_size: u32,
    /// Gundam (global view + dynamic tiles) vs native single-resolution.
    pub crop_mode: bool,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            base_size: BASE_SIZE,
            image_size: TILE_SIZE,
            crop_mode: true,
        }
    }
}

impl ProcessorConfig {
    pub fn from_json_str(data: &str) -> Result<Self> {
        let raw: RawProcessorConfig =
            serde_json::from_str(data).context("parse jina-ocr-v1 processor_config.json")?;
        let d = Self::default();
        Ok(Self {
            base_size: raw.base_size.unwrap_or(d.base_size),
            image_size: raw.image_size.unwrap_or(d.image_size),
            crop_mode: raw.crop_mode.unwrap_or(d.crop_mode),
        })
    }

    /// Parse `processor_config.json` from a checkpoint dir, falling back to the
    /// published defaults when the file is absent.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("processor_config.json");
        if !path.is_file() {
            return Ok(Self::default());
        }
        let data = std::fs::read_to_string(&path)
            .with_context(|| format!("read jina-ocr-v1 processor config {path:?}"))?;
        Self::from_json_str(&data)
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawProcessorConfig {
    base_size: Option<u32>,
    image_size: Option<u32>,
    crop_mode: Option<bool>,
}

/// Only the jina-specific fields; the decoder/vision/projector body is parsed
/// by [`UnlimitedOcrConfig`] from the same JSON text.
#[derive(Debug, Clone, Default, Deserialize)]
struct RawJinaOcrConfig {
    model_type: Option<String>,
    /// jina spells the placeholder id `image_token_index`; Unlimited-OCR uses
    /// `image_token_id`. Both default to 128815.
    image_token_index: Option<u32>,
    image_token_id: Option<u32>,
    /// Present only if a future revision adds a window; absence means *none*.
    sliding_window: Option<usize>,
    sliding_window_size: Option<usize>,
    rope_theta: Option<f64>,
    mtp_num_heads: Option<usize>,
    num_nextn_predict_layers: Option<usize>,
    mtp_num_speculative_steps: Option<usize>,
    mtp_recursive: Option<bool>,
    mtp_share_embedding_weights: Option<bool>,
    mtp_share_lm_head: Option<bool>,
    mtp_share_norm: Option<bool>,
    mtp_moe: Option<bool>,
}

/// Resolved `jinaai/jina-ocr-v1` checkpoint config.
#[derive(Debug, Clone)]
pub struct JinaOcrConfig {
    /// Decoder + vision + projector, in the shape [`rlx_unlimited_ocr`] builds.
    pub lm: UnlimitedOcrConfig,
    pub mtp: MtpConfig,
    pub processor: ProcessorConfig,
}

impl JinaOcrConfig {
    pub const HF_MODEL_ID: &'static str = "jinaai/jina-ocr-v1";
    pub const MODEL_TYPE: &'static str = "deepseek_vl_v2";

    /// Parse `config.json`. `processor` is left at the published defaults —
    /// use [`Self::from_model_dir`] to pick up `processor_config.json` too.
    pub fn from_json_str(data: &str) -> Result<Self> {
        let raw: RawJinaOcrConfig =
            serde_json::from_str(data).context("parse jina-ocr-v1 config.json")?;
        let mut lm = UnlimitedOcrConfig::from_json_str(data)?;

        // `UnlimitedOcrConfig` defaults `rope_theta` to 10_000 when the key is
        // absent; jina states 1e6 and a wrong base is silent, so re-apply the
        // raw value when the card carries one.
        if let Some(theta) = raw.rope_theta {
            lm.rope_theta = theta;
        }
        if let Some(model_type) = raw.model_type.clone() {
            lm.model_type = model_type;
        }

        // `sliding_window` absent => plain causal attention. Pin it rather than
        // inherit Unlimited-OCR's 128 fallback (see module docs).
        lm.sliding_window = raw.sliding_window.or(raw.sliding_window_size).unwrap_or(0);
        lm.image_token_id = raw
            .image_token_index
            .or(raw.image_token_id)
            .unwrap_or(IMAGE_TOKEN_ID);

        Ok(Self {
            mtp: MtpConfig::from_raw(&raw),
            lm,
            processor: ProcessorConfig::default(),
        })
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("read jina-ocr-v1 config {path:?}"))?;
        Self::from_json_str(&data).with_context(|| format!("parse jina-ocr-v1 config {path:?}"))
    }

    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let mut cfg = Self::from_file(&dir.join("config.json"))?;
        cfg.processor = ProcessorConfig::from_model_dir(dir)?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        self.lm.validate()?;
        ensure!(
            self.lm.sliding_window == 0,
            "jina-ocr-v1 uses plain causal attention; got sliding_window={}",
            self.lm.sliding_window
        );
        ensure!(
            !self.mtp.is_enabled() || self.mtp.num_nextn_predict_layers == 1,
            "FastMTP recursive mode requires num_nextn_predict_layers=1, got {}",
            self.mtp.num_nextn_predict_layers
        );
        Ok(())
    }

    pub fn image_token_id(&self) -> u32 {
        self.lm.image_token_id
    }

    pub fn eos_token_id(&self) -> u32 {
        self.lm.eos_token_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim `jinaai/jina-ocr-v1` `config.json` (vision/projector blocks
    /// trimmed to what the parser reads).
    const REAL_CONFIG: &str = r#"{
        "architectures": ["DeepseekOCRForCausalLM"],
        "attention_bias": false,
        "bos_token_id": 0,
        "candidate_resolutions": [[1024, 1024]],
        "eos_token_id": 1,
        "first_k_dense_replace": 1,
        "global_view_pos": "head",
        "hidden_act": "silu",
        "hidden_size": 1280,
        "image_token_index": 128815,
        "intermediate_size": 6848,
        "language_config": {
            "bos_token_id": 0,
            "eos_token_id": 1,
            "first_k_dense_replace": 1,
            "hidden_size": 1280,
            "intermediate_size": 6848,
            "max_position_embeddings": 32768,
            "moe_intermediate_size": 896,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_attention_heads": 10,
            "num_experts_per_tok": 6,
            "num_hidden_layers": 12,
            "num_key_value_heads": 10,
            "rope_theta": 1000000,
            "use_mla": false,
            "vocab_size": 129280
        },
        "max_position_embeddings": 32768,
        "model_type": "deepseek_vl_v2",
        "moe_intermediate_size": 896,
        "moe_layer_freq": 1,
        "mtp_moe": false,
        "mtp_num_heads": 1,
        "mtp_num_speculative_steps": 3,
        "mtp_recursive": true,
        "mtp_share_embedding_weights": true,
        "mtp_share_lm_head": true,
        "mtp_share_norm": true,
        "num_nextn_predict_layers": 1,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "norm_topk_prob": false,
        "num_attention_heads": 10,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 12,
        "num_key_value_heads": 10,
        "pad_token_id": 2,
        "projector_config": {
            "input_dim": 2048,
            "model_type": "mlp_projector",
            "n_embed": 1280,
            "projector_type": "linear"
        },
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000,
        "routed_scaling_factor": 1.0,
        "scoring_func": "softmax",
        "tie_word_embeddings": false,
        "tile_tag": "2D",
        "topk_method": "greedy",
        "use_mla": false,
        "v_head_dim": 0,
        "vision_config": {
            "image_size": 1024,
            "mlp_ratio": 3.7362,
            "model_name": "deeplip_b_l",
            "model_type": "vision",
            "width": {
                "clip-l-14-224": {"heads": 16, "image_size": 224, "layers": 24, "patch_size": 14, "width": 1024},
                "sam_vit_b": {
                    "downsample_channels": [512, 1024],
                    "global_attn_indexes": [2, 5, 8, 11],
                    "heads": 12,
                    "layers": 12,
                    "width": 768
                }
            }
        },
        "vocab_size": 129280
    }"#;

    fn real() -> JinaOcrConfig {
        JinaOcrConfig::from_json_str(REAL_CONFIG).expect("parse")
    }

    #[test]
    fn real_config_parses_and_validates() {
        let cfg = real();
        cfg.validate().expect("validate");
        assert_eq!(cfg.lm.model_type, JinaOcrConfig::MODEL_TYPE);
        assert_eq!(cfg.lm.hidden_size, 1280);
        assert_eq!(cfg.lm.num_hidden_layers, 12);
        assert_eq!(cfg.lm.num_attention_heads, 10);
        assert_eq!(cfg.lm.num_key_value_heads, 10);
        assert_eq!(cfg.lm.head_dim(), 128);
        assert_eq!(cfg.lm.n_routed_experts, 64);
        assert_eq!(cfg.lm.n_shared_experts, 2);
        assert_eq!(cfg.lm.num_experts_per_tok, 6);
        assert_eq!(cfg.lm.moe_intermediate_size, 896);
        assert_eq!(cfg.lm.intermediate_size, 6848);
        assert_eq!(cfg.lm.first_k_dense_replace, 1);
        assert_eq!(cfg.lm.vocab_size, 129_280);
        assert!(!cfg.lm.use_mla);
    }

    /// The whole point of a separate config: jina has no `sliding_window` key,
    /// and inheriting Unlimited-OCR's 128 fallback would clamp attention to the
    /// last 128 tokens with nothing to catch it.
    #[test]
    fn no_sliding_window_key_means_full_causal() {
        assert!(!REAL_CONFIG.contains("sliding_window"));
        assert_eq!(real().lm.sliding_window, 0);
    }

    /// `rope_theta` is 1e6 here vs Unlimited-OCR's implicit 10_000.
    #[test]
    fn rope_theta_is_one_million() {
        assert_eq!(real().lm.rope_theta, 1_000_000.0);
    }

    #[test]
    fn image_token_index_alias_is_read() {
        assert_eq!(real().image_token_id(), IMAGE_TOKEN_ID);
        assert_eq!(real().eos_token_id(), EOS_TOKEN_ID);
    }

    #[test]
    fn mtp_head_matches_card() {
        let mtp = real().mtp;
        assert!(mtp.is_enabled());
        assert!(mtp.recursive);
        assert!(!mtp.moe);
        assert_eq!(mtp.num_heads, 1);
        assert_eq!(mtp.num_nextn_predict_layers, 1);
        assert_eq!(mtp.num_speculative_steps, 3);
        assert!(mtp.share_embedding_weights && mtp.share_lm_head && mtp.share_norm);
    }

    #[test]
    fn dense_layer_zero_then_moe() {
        let cfg = real();
        assert!(cfg.lm.is_dense_layer(0));
        for i in 1..cfg.lm.num_hidden_layers {
            assert!(!cfg.lm.is_dense_layer(i), "layer {i} should be MoE");
        }
    }

    #[test]
    fn processor_config_parses_real_card() {
        let p = ProcessorConfig::from_json_str(
            r#"{"base_size": 1024, "crop_mode": true, "image_size": 640,
                "processor_class": "DeepseekOCRProcessor", "use_fast": null}"#,
        )
        .expect("parse");
        assert_eq!(p, ProcessorConfig::default());
        assert_eq!(p.base_size, BASE_SIZE);
        assert_eq!(p.image_size, TILE_SIZE);
        assert!(p.crop_mode);
    }

    /// `compute_n_queries(size) = ceil((size // 16) / 4)`.
    #[test]
    fn query_grid_sides_match_hf() {
        let cfg = real();
        assert_eq!(cfg.lm.num_queries(BASE_SIZE as usize), 16);
        assert_eq!(cfg.lm.num_queries(TILE_SIZE as usize), 10);
    }
}
