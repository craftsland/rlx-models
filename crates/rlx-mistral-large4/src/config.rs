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

//! `mistral-large4` configuration — read from GGUF metadata or a HF
//! `config.json` (`model_type = mistral_large4`, its `text_config`).
//!
//! ```text
//!   HF text_config                     GGUF metadata (mistral-large4.*)
//!   ─────────────────────────────────  ─────────────────────────────────────────
//!   qk_nope + qk_rope_head_dim (128)   attention.key_length_mla
//!   qk_rope_head_dim (64)              rope.dimension_count
//!   v_head_dim                         attention.value_length_mla
//!   no_rope_layers[i] (1 = RoPE)       attention.rope_pattern[i] (true = RoPE)
//!   indexer_types[i] ("full")          attention.indexer.types[i] (true = full)
//!   kv_source_layers[i]                attention.kv_source_map[i]
//!   first_k_dense_replace              leading_dense_block_count
//!   final_logit_softcapping            final_logit_softcapping
//!   rope_parameters (yarn)             rope.scaling.{factor, original_context_length, yarn_beta_*}
//!   (indexer LayerNorm eps, 1e-6)      attention.layer_norm_epsilon
//! ```
//!
//! The learned RoPE scale (LeRoPE `alpha`) is a **tensor**, not a key:
//! `rope_freqs.weight = exp(-alpha)` in a GGUF — see [`crate::rope`].

use anyhow::{Context, Result, bail};
use rlx_core::standard_decoder::RopeScaling;
use rlx_gguf::{GgufFile, MetaValue};
use serde::Deserialize;
use std::path::Path;

/// `general.architecture` values this crate accepts.
pub const ACCEPTED_ARCHES: &[&str] = &["mistral-large4"];

/// HF `model_type` values (outer and text config).
pub const HF_MODEL_TYPES: &[&str] = &["mistral_large4", "mistral_large4_text"];

/// GGUF `expert_gating_func` value for softmax routing.
const GATING_SOFTMAX: u32 = 1;

/// Indexer LayerNorm epsilon when the checkpoint does not say (HF hardcodes it).
const DEFAULT_INDEXER_NORM_EPS: f32 = 1e-6;

/// The Mistral Large 4 text decoder.
#[derive(Debug, Clone)]
pub struct MistralLarge4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Dense MLP width (the leading dense layers).
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    /// Epsilon of the indexer's two LayerNorms (`q_input_norm`, `k_norm`).
    pub indexer_norm_eps: f32,
    pub max_position_embeddings: usize,

    // ── MLA (no Q-LoRA) ──
    pub kv_lora_rank: usize,
    /// Non-rotary query/key width on a RoPE layer. A NoPE layer uses
    /// `qk_nope_head_dim + qk_rope_head_dim` for it instead.
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,

    // ── per-layer schedules ──
    /// `true` where the layer applies RoPE (attention and indexer alike).
    pub use_rope: Vec<bool>,
    /// `true` where the layer runs its own DSA indexer; `false` reuses the
    /// previous full layer's top-k.
    pub indexer_full: Vec<bool>,
    /// The layer whose latent KV each layer attends to (itself = owns it).
    pub kv_source: Vec<usize>,
    /// `true` where the FFN is the routed MoE, `false` the dense MLP.
    pub moe_layers: Vec<bool>,

    // ── DSA indexer ──
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,

    // ── MoE ──
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub moe_intermediate_size: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f32,

    // ── RoPE ──
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,

    /// `logits = c · tanh(logits / c)`; `None` disables it.
    pub final_logit_softcapping: Option<f32>,
    pub tie_word_embeddings: bool,
}

impl MistralLarge4Config {
    /// Full query/key head width — the same on every layer (128).
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Rotary width of layer `i` (0 on a NoPE layer).
    pub fn rope_dim(&self, i: usize) -> usize {
        if self.use_rope[i] {
            self.qk_rope_head_dim
        } else {
            0
        }
    }

    /// Non-rotary width of layer `i`.
    pub fn nope_dim(&self, i: usize) -> usize {
        self.qk_head_dim() - self.rope_dim(i)
    }

    /// Whether layer `i` computes and caches its own latent KV.
    pub fn owns_kv(&self, i: usize) -> bool {
        self.kv_source[i] == i
    }

    /// Width of one cached latent row of owning layer `i`:
    /// `[k_pass (kv_lora) | rotated k_rot (rope_dim)]`.
    pub fn latent_width(&self, i: usize) -> usize {
        self.kv_lora_rank + self.rope_dim(i)
    }

    /// Whether DSA selection is the identity at `seq` keys (it then keeps
    /// every causal key, so attention is plain causal).
    pub fn dsa_is_dense(&self, seq: usize) -> bool {
        seq <= self.index_topk
    }

    /// Every tensor block `i` holds, in GGUF spelling.
    pub fn block_tensor_names(&self, i: usize) -> Vec<String> {
        let b = format!("blk.{i}");
        let mut v: Vec<String> = [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k_b.weight",
            "attn_v_b.weight",
            "attn_output.weight",
            "attn_ssmax.weight",
            "attn_ssmax.bias",
            "ffn_norm.weight",
        ]
        .iter()
        .map(|n| format!("{b}.{n}"))
        .collect();
        if self.owns_kv(i) {
            v.push(format!("{b}.attn_kv_a_mqa.weight"));
            v.push(format!("{b}.attn_kv_a_norm.weight"));
        }
        if self.indexer_full[i] {
            for n in [
                "q_norm.weight",
                "q_norm.bias",
                "attn_q_b.weight",
                "attn_k.weight",
                "k_norm.weight",
                "k_norm.bias",
                "proj.weight",
                "ssmax.weight",
                "ssmax.bias",
            ] {
                v.push(format!("{b}.indexer.{n}"));
            }
        }
        if self.moe_layers[i] {
            for n in [
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
                "ffn_down_shexp.weight",
            ] {
                v.push(format!("{b}.{n}"));
            }
        } else {
            for n in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                v.push(format!("{b}.{n}"));
            }
        }
        v
    }

    /// Every tensor the text model reads, in GGUF spelling.
    pub fn tensor_manifest(&self) -> Vec<String> {
        let mut v = vec![
            "token_embd.weight".to_string(),
            "output_norm.weight".to_string(),
            "rope_freqs.weight".to_string(),
        ];
        if !self.tie_word_embeddings {
            v.push("output.weight".to_string());
        }
        for i in 0..self.num_hidden_layers {
            v.extend(self.block_tensor_names(i));
        }
        v
    }

    pub fn validate(&self) -> Result<()> {
        let n = self.num_hidden_layers;
        if n == 0 || self.hidden_size == 0 || self.num_attention_heads == 0 {
            bail!("mistral-large4: empty model dimensions");
        }
        for (name, len) in [
            ("use_rope", self.use_rope.len()),
            ("indexer_full", self.indexer_full.len()),
            ("kv_source", self.kv_source.len()),
            ("moe_layers", self.moe_layers.len()),
        ] {
            if len != n {
                bail!("mistral-large4: {name} has {len} entries, expected one per layer ({n})");
            }
        }
        if !self.indexer_full[0] {
            bail!(
                "mistral-large4: layer 0 must run a full indexer (shared layers reuse an earlier one)"
            );
        }
        for (i, &src) in self.kv_source.iter().enumerate() {
            if src > i || self.kv_source[src] != src {
                bail!(
                    "mistral-large4: layer {i} shares KV with layer {src}, which is not an \
                     earlier layer that owns its cache"
                );
            }
            if self.use_rope[src] != self.use_rope[i] {
                bail!(
                    "mistral-large4: layer {i} and its KV source {src} disagree on RoPE — the \
                     shared key would carry the wrong positional part"
                );
            }
        }
        if !self.qk_rope_head_dim.is_multiple_of(2) || self.qk_rope_head_dim > self.index_head_dim {
            bail!(
                "mistral-large4: qk_rope_head_dim = {} must be even and fit the indexer head ({})",
                self.qk_rope_head_dim,
                self.index_head_dim
            );
        }
        if self.qk_head_dim() != self.v_head_dim {
            // `Op::Attention` carries one head width for Q, K and V.
            bail!(
                "mistral-large4: qk head {} != v head {} is not supported by the fused prefill \
                 attention",
                self.qk_head_dim(),
                self.v_head_dim
            );
        }
        if self.n_group != 1 || self.topk_group != 1 {
            bail!(
                "mistral-large4: group-limited routing (n_group = {}, topk_group = {}) is not \
                 implemented; every published checkpoint uses 1/1",
                self.n_group,
                self.topk_group
            );
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.n_routed_experts {
            bail!(
                "mistral-large4: num_experts_per_tok = {} with {} experts",
                self.num_experts_per_tok,
                self.n_routed_experts
            );
        }
        if self.n_shared_experts != 1 {
            bail!(
                "mistral-large4: n_shared_experts = {} (only one shared expert is emitted)",
                self.n_shared_experts
            );
        }
        if self.index_topk == 0 {
            bail!("mistral-large4: index_topk must be >= 1");
        }
        Ok(())
    }
}

// ─────────────────────────── GGUF ───────────────────────────

fn meta<'a>(raw: &'a GgufFile, key: &str) -> Option<&'a MetaValue> {
    raw.metadata.get(key)
}

fn key(k: &str) -> String {
    format!("mistral-large4.{k}")
}

fn meta_u32(raw: &GgufFile, k: &str) -> Result<u32> {
    meta(raw, &key(k))
        .and_then(MetaValue::as_u32)
        .with_context(|| {
            format!(
                "mistral-large4: missing or non-integer GGUF key `{}`",
                key(k)
            )
        })
}

fn meta_u32_or(raw: &GgufFile, k: &str, default: u32) -> u32 {
    meta(raw, &key(k))
        .and_then(MetaValue::as_u32)
        .unwrap_or(default)
}

fn as_f32(v: &MetaValue) -> Option<f32> {
    match v {
        MetaValue::F32(x) => Some(*x),
        MetaValue::F64(x) => Some(*x as f32),
        _ => None,
    }
}

fn as_bool(v: &MetaValue) -> Option<bool> {
    match v {
        MetaValue::Bool(b) => Some(*b),
        other => other.as_u32().map(|x| x != 0),
    }
}

fn meta_f32(raw: &GgufFile, k: &str) -> Option<f32> {
    meta(raw, &key(k)).and_then(as_f32)
}

fn meta_array<'a>(raw: &'a GgufFile, k: &str) -> Option<&'a [MetaValue]> {
    match meta(raw, &key(k)) {
        Some(MetaValue::Array(v)) => Some(v.as_slice()),
        _ => None,
    }
}

/// A per-layer array, or `default` for every layer when the key is absent.
fn per_layer<T: Clone>(
    raw: &GgufFile,
    k: &str,
    n: usize,
    default: T,
    conv: impl Fn(&MetaValue) -> Option<T>,
) -> Result<Vec<T>> {
    let Some(arr) = meta_array(raw, k) else {
        return Ok(vec![default; n]);
    };
    if arr.len() != n {
        bail!(
            "mistral-large4: `{}` has {} entries, expected {n}",
            key(k),
            arr.len()
        );
    }
    arr.iter()
        .enumerate()
        .map(|(i, v)| {
            conv(v).with_context(|| format!("mistral-large4: `{}`[{i}] has the wrong type", key(k)))
        })
        .collect()
}

impl MistralLarge4Config {
    /// Parse a `mistral-large4` GGUF's metadata (metadata only — works on a
    /// split set's first shard).
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = match meta(raw, "general.architecture") {
            Some(MetaValue::String(s)) => s.as_str(),
            _ => "",
        };
        if !ACCEPTED_ARCHES.contains(&arch) {
            bail!(
                "MistralLarge4Config::from_gguf: expected general.architecture ∈ \
                 {ACCEPTED_ARCHES:?}, got {arch:?}"
            );
        }
        let n = meta_u32(raw, "block_count")? as usize;
        let qk_head = meta_u32(raw, "attention.key_length_mla")? as usize;
        let rope_dim = meta_u32(raw, "rope.dimension_count")? as usize;
        if rope_dim > qk_head {
            bail!("mistral-large4: rope.dimension_count {rope_dim} > key_length_mla {qk_head}");
        }
        let gating = meta_u32_or(raw, "expert_gating_func", GATING_SOFTMAX);
        if gating != GATING_SOFTMAX {
            bail!(
                "mistral-large4: expert_gating_func = {gating}, only softmax ({GATING_SOFTMAX}) is implemented"
            );
        }
        if let Some(MetaValue::Bool(false)) = meta(raw, &key("expert_weights_norm")) {
            bail!("mistral-large4: expert_weights_norm = false is not implemented");
        }
        let first_dense = meta_u32_or(raw, "leading_dense_block_count", 1) as usize;
        let scaling = match meta(raw, &key("rope.scaling.type")) {
            Some(MetaValue::String(s)) if s == "yarn" => RopeScaling::Yarn {
                factor: meta_f32(raw, "rope.scaling.factor").unwrap_or(1.0) as f64,
                original_max_position_embeddings: meta_u32(
                    raw,
                    "rope.scaling.original_context_length",
                )? as f64,
                beta_fast: meta_f32(raw, "rope.scaling.yarn_beta_fast").unwrap_or(32.0) as f64,
                beta_slow: meta_f32(raw, "rope.scaling.yarn_beta_slow").unwrap_or(1.0) as f64,
                // The converter asserts YaRN *without* magnitude scaling
                // (`apply_scale = false`), i.e. HF `attention_factor = 1.0`.
                attention_factor: Some(1.0),
            },
            None => RopeScaling::None,
            Some(other) => bail!("mistral-large4: unsupported rope.scaling.type {other:?}"),
        };
        let softcap = meta_f32(raw, "final_logit_softcapping").filter(|c| *c > 0.0);
        let cfg = Self {
            vocab_size: meta_u32(raw, "vocab_size")? as usize,
            hidden_size: meta_u32(raw, "embedding_length")? as usize,
            intermediate_size: meta_u32(raw, "feed_forward_length")? as usize,
            num_hidden_layers: n,
            num_attention_heads: meta_u32(raw, "attention.head_count")? as usize,
            rms_norm_eps: meta_f32(raw, "attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
            indexer_norm_eps: meta_f32(raw, "attention.layer_norm_epsilon")
                .unwrap_or(DEFAULT_INDEXER_NORM_EPS),
            max_position_embeddings: meta_u32_or(raw, "context_length", 1 << 20) as usize,
            kv_lora_rank: meta_u32(raw, "attention.kv_lora_rank")? as usize,
            qk_nope_head_dim: qk_head - rope_dim,
            qk_rope_head_dim: rope_dim,
            v_head_dim: meta_u32(raw, "attention.value_length_mla")? as usize,
            use_rope: per_layer(raw, "attention.rope_pattern", n, true, as_bool)?,
            indexer_full: per_layer(raw, "attention.indexer.types", n, true, as_bool)?,
            kv_source: per_layer(raw, "attention.kv_source_map", n, usize::MAX, |v| {
                v.as_u32().map(|x| x as usize)
            })?
            .into_iter()
            .enumerate()
            .map(|(i, s)| if s == usize::MAX { i } else { s })
            .collect(),
            moe_layers: (0..n).map(|i| i >= first_dense).collect(),
            index_n_heads: meta_u32(raw, "attention.indexer.head_count")? as usize,
            index_head_dim: meta_u32(raw, "attention.indexer.key_length")? as usize,
            index_topk: meta_u32(raw, "attention.indexer.top_k")? as usize,
            n_routed_experts: meta_u32(raw, "expert_count")? as usize,
            num_experts_per_tok: meta_u32(raw, "expert_used_count")? as usize,
            n_shared_experts: meta_u32_or(raw, "expert_shared_count", 1) as usize,
            moe_intermediate_size: meta_u32(raw, "expert_feed_forward_length")? as usize,
            n_group: meta_u32_or(raw, "expert_group_count", 1).max(1) as usize,
            topk_group: meta_u32_or(raw, "expert_group_used_count", 1).max(1) as usize,
            routed_scaling_factor: meta_f32(raw, "expert_weights_scale").unwrap_or(1.0),
            rope_theta: meta_f32(raw, "rope.freq_base").unwrap_or(10000.0) as f64,
            rope_scaling: scaling,
            final_logit_softcapping: softcap,
            tie_word_embeddings: false,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse the GGUF at `path` (metadata only).
    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        // Header only: `from_path` would read the tensor data too, which for a
        // single-file checkpoint is the whole model.
        let raw = GgufFile::header_from_path(path)
            .with_context(|| format!("mistral-large4: reading GGUF metadata from {path:?}"))?;
        Self::from_gguf(&raw)
    }
}

// ─────────────────────────── HF config.json ───────────────────────────

#[derive(Debug, Deserialize)]
struct HfText {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    #[serde(default = "one")]
    n_shared_experts: usize,
    #[serde(default = "one")]
    n_group: usize,
    #[serde(default = "one")]
    topk_group: usize,
    #[serde(default = "onef")]
    routed_scaling_factor: f32,
    #[serde(default = "one")]
    first_k_dense_replace: usize,
    #[serde(default)]
    mlp_layer_types: Option<Vec<String>>,
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    #[serde(default)]
    indexer_types: Option<Vec<String>>,
    #[serde(default)]
    no_rope_layers: Option<Vec<u8>>,
    #[serde(default = "four")]
    no_rope_layer_interval: usize,
    #[serde(default)]
    kv_source_layers: Option<Vec<usize>>,
    #[serde(default)]
    final_logit_softcapping: Option<f32>,
    #[serde(default)]
    tie_word_embeddings: bool,
}

fn default_eps() -> f32 {
    1e-5
}
fn one() -> usize {
    1
}
fn onef() -> f32 {
    1.0
}
fn four() -> usize {
    4
}

impl MistralLarge4Config {
    /// Parse a HF `config.json` (the multimodal outer config or the bare
    /// `text_config`).
    pub fn from_hf_json(json: &str) -> Result<Self> {
        let root: serde_json::Value =
            serde_json::from_str(json).context("mistral-large4: parsing config.json")?;
        let text = root.get("text_config").unwrap_or(&root).clone();
        if let Some(mt) = text.get("model_type").and_then(|v| v.as_str())
            && !HF_MODEL_TYPES.contains(&mt)
        {
            bail!("mistral-large4: model_type {mt:?} is not one of {HF_MODEL_TYPES:?}");
        }
        let t: HfText =
            serde_json::from_value(text.clone()).context("mistral-large4: reading text_config")?;
        let n = t.num_hidden_layers;
        let use_rope = match &t.no_rope_layers {
            Some(v) => v.iter().map(|&x| x != 0).collect(),
            None => (0..n)
                .map(|i| (i + 1) % t.no_rope_layer_interval != 0)
                .collect(),
        };
        let indexer_full = match &t.indexer_types {
            Some(v) => v
                .iter()
                .map(|s| match s.as_str() {
                    "full" => Ok(true),
                    "shared" => Ok(false),
                    other => bail!("mistral-large4: unknown indexer type {other:?}"),
                })
                .collect::<Result<Vec<_>>>()?,
            None => vec![true; n],
        };
        let moe_layers = match &t.mlp_layer_types {
            Some(v) => v
                .iter()
                .map(|s| match s.as_str() {
                    "sparse" => Ok(true),
                    "dense" => Ok(false),
                    other => bail!("mistral-large4: unknown mlp layer type {other:?}"),
                })
                .collect::<Result<Vec<_>>>()?,
            None => (0..n).map(|i| i >= t.first_k_dense_replace).collect(),
        };
        // transformers 5 spells it `rope_parameters`; older configs `rope_scaling`.
        let rope = text
            .get("rope_parameters")
            .or_else(|| text.get("rope_scaling"))
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let theta = rope
            .get("rope_theta")
            .or_else(|| text.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);
        let getf = |k: &str, d: f64| rope.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let scaling = match rope.get("rope_type").and_then(|v| v.as_str()) {
            Some("yarn") => RopeScaling::Yarn {
                factor: getf("factor", 1.0),
                original_max_position_embeddings: getf("original_max_position_embeddings", 512.0),
                beta_fast: getf("beta_fast", 32.0),
                beta_slow: getf("beta_slow", 1.0),
                attention_factor: Some(getf("attention_factor", 1.0)),
            },
            None | Some("default") => RopeScaling::None,
            Some(other) => bail!("mistral-large4: unsupported rope_type {other:?}"),
        };
        let cfg = Self {
            vocab_size: t.vocab_size,
            hidden_size: t.hidden_size,
            intermediate_size: t.intermediate_size,
            num_hidden_layers: n,
            num_attention_heads: t.num_attention_heads,
            rms_norm_eps: t.rms_norm_eps,
            indexer_norm_eps: DEFAULT_INDEXER_NORM_EPS,
            max_position_embeddings: t.max_position_embeddings.unwrap_or(1 << 20),
            kv_lora_rank: t.kv_lora_rank,
            qk_nope_head_dim: t.qk_nope_head_dim,
            qk_rope_head_dim: t.qk_rope_head_dim,
            v_head_dim: t.v_head_dim,
            use_rope,
            indexer_full,
            kv_source: t.kv_source_layers.unwrap_or_else(|| (0..n).collect()),
            moe_layers,
            index_n_heads: t.index_n_heads,
            index_head_dim: t.index_head_dim,
            index_topk: t.index_topk,
            n_routed_experts: t.n_routed_experts,
            num_experts_per_tok: t.num_experts_per_tok,
            n_shared_experts: t.n_shared_experts,
            moe_intermediate_size: t.moe_intermediate_size,
            n_group: t.n_group,
            topk_group: t.topk_group,
            routed_scaling_factor: t.routed_scaling_factor,
            rope_theta: theta,
            rope_scaling: scaling,
            final_logit_softcapping: t.final_logit_softcapping.filter(|c| *c > 0.0),
            tie_word_embeddings: t.tie_word_embeddings,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_hf_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let s = std::fs::read_to_string(path)
            .with_context(|| format!("mistral-large4: reading {path:?}"))?;
        Self::from_hf_json(&s)
    }
}
