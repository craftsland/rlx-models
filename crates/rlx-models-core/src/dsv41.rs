// RLX — versatile ML compiler + runtime. GPLv3.
//! **DeepSeek-V4.1-Flash** (`model_type: deepseek_v41`) — config, shapes and the
//! host-side pieces the graph builders need.
//!
//! V4.1 keeps V4's five subsystems (Hyper-Connections, grouped o-LoRA MLA,
//! sink-attention over a sliding window, a learned KV compressor and a learned
//! sparse Indexer) and adds four things the V4 port has no notion of:
//!
//! 1. **CSA2 KV sharing.** `compress_ratio > 0` no longer means the layer
//!    compresses its own KV. Only [`kv_source_layers`](DeepseekV41Spec::kv_source_layers)
//!    run a `Compressor`; every layer after one reads that layer's cache until the
//!    next source. Same for the Indexer and [`index_source_layers`](DeepseekV41Spec::index_source_layers).
//! 2. **A hierarchical (two-level) Indexer.** One designated layer
//!    ([`candidate_source_layer`](DeepseekV41Spec::candidate_source_layer)) picks
//!    `candidate_topk_blocks` blocks of `candidate_block_size` compressed positions;
//!    later index sources score only inside those blocks.
//! 3. **Engram** — n-gram hash lookups mixed into the residual stream at a few
//!    layers ([`crate::dsv41_engram`]).
//! 4. **Vision** — a DeepSeek-ViT with 2-D RoPE plus an unfold/MLP aligner
//!    ([`crate::dsv41_vision`]).
//!
//! It also drops V4's `hc_head`: the final `hc_mult → 1` reduce reuses the *last
//! block's* FFN pre-mix rather than a dedicated gate, because V4.1 threads the
//! Hyper-Connection coefficients forward — each sublayer computes the mix the
//! *next* one consumes.
//!
//! Reference: `deepseek-ai/DeepSeek-V4.1-Flash` `inference/{model.py,engram.py,
//! vision.py,kernel.py}` and the released `config.json`.

use anyhow::{Result, anyhow};
use serde_json::Value;

/// Vision tower shape (`vision_config` / the flat `vision_*` keys). Absent — or
/// `vision_n_layers == 0` — means the checkpoint is text-only.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionSpec {
    pub n_layers: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub inter_dim: usize,
    pub patch_size: usize,
    pub rope_theta: f64,
    /// Aligner square-unfold factor; `r² · vision_dim` features feed `aligner.w1`.
    pub downsample_ratio: usize,
    pub max_n_token: usize,
    pub min_pixels: usize,
    pub max_wh_ratio: Option<f64>,
}

/// Engram (`engram_*`) — the n-gram conditional-memory tables. Every field is
/// load-bearing for the hash: the multipliers are derived from
/// [`compressed_vocab_size`](EngramSpec::compressed_vocab_size) and the bucket
/// moduli are primes drawn in order from `vocab_size - 1`, so a mismatch in
/// either silently rehashes the whole table.
#[derive(Debug, Clone, PartialEq)]
pub struct EngramSpec {
    /// Backbone layers carrying an Engram (GA 4.1: `[1, 14]`).
    pub layer_ids: Vec<usize>,
    /// Rows in each layer's table (GA 4.1: `[384006168, 384016682]`).
    pub num_embeddings: Vec<usize>,
    /// Largest n-gram; a position contributes `max_ngram_size - 1` hashes.
    pub max_ngram_size: usize,
    /// Prime-search start for every `(n-gram size, head)` bucket range.
    pub vocab_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// Token id that fills look-back slots with no history (matches training).
    pub pad_token_id: usize,
    /// Size of the *normalized* token space n-grams are hashed over.
    pub compressed_vocab_size: usize,
}

impl EngramSpec {
    /// Columns of hash ids per position: `(max_ngram_size - 1) · n_heads`.
    pub fn n_hash_cols(&self) -> usize {
        self.max_ngram_size.saturating_sub(1) * self.n_heads
    }
    /// Index of `layer` within [`layer_ids`](Self::layer_ids), or `None`.
    pub fn layer_hash_index(&self, layer: usize) -> Option<usize> {
        self.layer_ids.iter().position(|&l| l == layer)
    }
}

/// Router scoring function. V4.1 ships `sqrtsoftplus`; the other two exist so a
/// derivative checkpoint that flips `scoring_func` still loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreFunc {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}

/// DeepSeek-V4.1-Flash (`deepseek_v41`) spec.
///
/// `compress_ratios` carries one entry per layer **including** the DSpark stages,
/// which occupy layer ids `n_layers .. n_layers + n_mtp_layers` — the same
/// indexing the reference `ModelArgs` uses, so `layer_id` is a single namespace
/// across the backbone and the draft head.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekV41Spec {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub hc_mult: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    /// `o_groups` — the block-diagonal group count of the o-LoRA down-projection.
    pub n_groups: usize,
    pub o_lora_rank: usize,
    /// Per-layer KV-compression ratio; `0` = sliding window only. One entry per
    /// layer, DSpark stages included.
    pub compress_ratios: Vec<usize>,
    /// Layers that own a `Compressor` (and, when also an index source, the index
    /// keys). Everyone else reads the nearest preceding source's cache.
    pub kv_source_layers: Vec<usize>,
    /// Layers that run an `Indexer`; the rest reuse the published top-k.
    pub index_source_layers: Vec<usize>,
    pub index_head_dim: usize,
    pub index_n_heads: usize,
    pub index_topk: usize,
    /// Level-one of the hierarchical indexer. `None` disables candidate
    /// pre-filtering (the reference encodes that as `candidate_source_layer < 0`).
    pub candidate_source_layer: Option<usize>,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub window_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_activated_experts: usize,
    pub n_shared_experts: usize,
    pub score_func: ScoreFunc,
    pub gate_temp: f32,
    pub norm_topk_prob: bool,
    pub route_scale: f32,
    /// Clamped-SwiGLU bound (`up ∈ [-L, L]`, `gate ≤ L`); `0` disables. V4.1 = 10.
    pub swiglu_limit: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// RoPE base for compressed KV — one latent stands for `compress_ratio`
    /// tokens, so its positions are further apart (V4.1 = 160000).
    pub compress_rope_theta: f64,
    /// YaRN `original_max_position_embeddings`; `0` disables YaRN. Applied to the
    /// compressed table only — pure sliding-window layers use raw `rope_theta`.
    pub original_seq_len: usize,
    pub rope_factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub hc_mult_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub engram: Option<EngramSpec>,
    pub vision: Option<VisionSpec>,
    /// Raw id of `<|deepseek_image|>`; every position of an image span carries it.
    pub image_token_id: usize,
    pub n_mtp_layers: usize,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: usize,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: usize,
    /// DSpark stages route over their own, smaller expert bank (GA 4.1: 128/top-3).
    /// `0` falls back to the backbone counts.
    pub dspark_n_routed_experts: usize,
    pub dspark_n_activated_experts: usize,
    /// `quantization_config.weight_block_size[0]`; `0` = the checkpoint is dense.
    pub weight_block_size: usize,
    /// `quantization_config.expert_dtype == "fp4"` — routed experts are stored as
    /// nibble pairs, so their column count on disk is halved.
    pub expert_fp4: bool,
}

/// Reader over a config that may be in either of two spellings.
///
/// The HF `config.json` nests the language model under `text_config` and uses
/// `transformers` key names; the flat `inference/config.json` the reference repo
/// ships uses the modelling code's own shorter names at the top level. Every
/// accessor here looks in `text_config` first and falls back to the root, and the
/// `*a` variants take both spellings — so one call site covers both files.
struct Cfg<'a> {
    root: &'a Value,
    text: &'a Value,
}

impl<'a> Cfg<'a> {
    fn new(root: &'a Value) -> Self {
        Cfg {
            root,
            text: root.get("text_config").unwrap_or(root),
        }
    }

    fn get(&self, k: &str) -> Option<&'a Value> {
        self.text.get(k).or_else(|| self.root.get(k))
    }

    fn u(&self, k: &str) -> Option<usize> {
        self.get(k).and_then(Value::as_u64).map(|x| x as usize)
    }

    fn f(&self, k: &str) -> Option<f64> {
        self.get(k).and_then(Value::as_f64)
    }

    fn b(&self, k: &str) -> Option<bool> {
        self.get(k).and_then(Value::as_bool)
    }

    /// Non-numeric entries are dropped rather than failing the parse: a missing
    /// or malformed list means "empty", which the callers already handle.
    fn vec_u(&self, k: &str) -> Vec<usize> {
        self.get(k)
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|e| e.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// HF name `k`, then the flat config's name `alt`.
    fn ua(&self, k: &str, alt: &str) -> Option<usize> {
        self.u(k).or_else(|| self.u(alt))
    }

    fn fa(&self, k: &str, alt: &str) -> Option<f64> {
        self.f(k).or_else(|| self.f(alt))
    }

    fn va(&self, k: &str, alt: &str) -> Vec<usize> {
        let r = self.vec_u(k);
        if r.is_empty() { self.vec_u(alt) } else { r }
    }

    /// A required key. The error names the HF spelling, which is the one a user
    /// editing a `config.json` will be looking at.
    fn need(&self, k: &str, alt: &str) -> Result<usize> {
        self.ua(k, alt)
            .ok_or_else(|| anyhow!("deepseek_v41 config missing `{k}`"))
    }

    /// A field of a nested object, e.g. `rope_scaling.factor`.
    fn obj_u(&self, obj: &str, k: &str) -> Option<usize> {
        self.get(obj)
            .and_then(|o| o.get(k))
            .and_then(Value::as_u64)
            .map(|x| x as usize)
    }

    fn obj_f(&self, obj: &str, k: &str) -> Option<f64> {
        self.get(obj).and_then(|o| o.get(k)).and_then(Value::as_f64)
    }
}

/// The n-gram hash memory, present only when the config names the layers it
/// sits on.
fn parse_engram(c: &Cfg) -> Option<EngramSpec> {
    let layer_ids = c.vec_u("engram_layer_ids");
    if layer_ids.is_empty() {
        return None;
    }
    Some(EngramSpec {
        num_embeddings: c.vec_u("engram_num_embeddings"),
        layer_ids,
        max_ngram_size: c.u("engram_max_ngram_size").unwrap_or(1),
        vocab_size: c.u("engram_vocab_size").unwrap_or(0),
        n_heads: c.u("engram_n_heads").unwrap_or(0),
        head_dim: c.u("engram_head_dim").unwrap_or(0),
        pad_token_id: c.ua("engram_pad_token_id", "engram_pad_id").unwrap_or(2),
        compressed_vocab_size: c.u("engram_compressed_vocab_size").unwrap_or(0),
    })
}

/// The ViT, present only on a vision-enabled checkpoint. Its keys live under a
/// nested `vision_config` (HF) or as flat `vision_*` keys.
///
/// The defaults are the released checkpoint's values, so a config that names
/// only `vision_n_layers` still describes the shipped tower.
fn parse_vision(c: &Cfg) -> Option<VisionSpec> {
    let vc = c.root.get("vision_config");
    let vu = |k: &str, alt: &str| {
        vc.and_then(|o| o.get(k))
            .and_then(Value::as_u64)
            .map(|x| x as usize)
            .or_else(|| c.u(alt))
    };
    let vf = |k: &str, alt: &str| {
        vc.and_then(|o| o.get(k))
            .and_then(Value::as_f64)
            .or_else(|| c.f(alt))
    };
    let n_layers = vu("num_hidden_layers", "vision_n_layers").unwrap_or(0);
    if n_layers == 0 {
        return None;
    }
    Some(VisionSpec {
        n_layers,
        dim: vu("hidden_size", "vision_dim").unwrap_or(1024),
        n_heads: vu("num_attention_heads", "vision_n_heads").unwrap_or(16),
        inter_dim: vu("intermediate_size", "vision_inter_dim").unwrap_or(2816),
        patch_size: vu("patch_size", "vision_patch_size").unwrap_or(14),
        rope_theta: vf("rope_theta", "vision_rope_theta").unwrap_or(10000.0),
        downsample_ratio: vu("downsample_ratio", "vision_downsample_ratio").unwrap_or(3),
        max_n_token: vu("max_image_tokens", "vision_max_n_token").unwrap_or(1024),
        min_pixels: vu("min_pixels", "vision_min_pixels").unwrap_or(295936),
        max_wh_ratio: vf("max_wh_ratio", "vision_max_wh_ratio"),
    })
}

/// How the routed-expert gate turns logits into scores.
fn parse_score_func(c: &Cfg) -> Result<ScoreFunc> {
    match c
        .get("scoring_func")
        .or_else(|| c.get("score_func"))
        .and_then(Value::as_str)
        .unwrap_or("sqrtsoftplus")
    {
        "softmax" => Ok(ScoreFunc::Softmax),
        "sigmoid" => Ok(ScoreFunc::Sigmoid),
        "sqrtsoftplus" => Ok(ScoreFunc::SqrtSoftplus),
        other => Err(anyhow!("deepseek_v41: unknown scoring_func `{other}`")),
    }
}

/// `(expert_fp4, weight_block_size)`.
///
/// The flat `inference/config.json` names no block size — the reference
/// hard-codes 32 in `kernel.py` — so any quantized config that omits it means
/// 32, and only a dense one means "no blocks".
fn parse_quant(c: &Cfg) -> (bool, usize) {
    let qc = c.root.get("quantization_config");
    let expert_fp4 = qc
        .and_then(|q| q.get("expert_dtype"))
        .or_else(|| c.root.get("expert_dtype"))
        .and_then(Value::as_str)
        == Some("fp4");
    let quantized =
        expert_fp4 || qc.is_some() || c.root.get("dtype").and_then(Value::as_str) == Some("fp8");
    let block = qc
        .and_then(|q| q.get("weight_block_size"))
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .unwrap_or(if quantized { 32 } else { 0 });
    (expert_fp4, block)
}

impl DeepseekV41Spec {
    /// Routed/activated expert counts for `layer_id` — backbone layers use the
    /// main bank, DSpark stages (`layer_id >= n_layers`) their own. Mirrors
    /// `ModelArgs.get_moe_config`.
    pub fn moe_dims(&self, layer_id: usize) -> (usize, usize) {
        if layer_id < self.n_layers {
            (self.n_routed_experts, self.n_activated_experts)
        } else {
            (
                if self.dspark_n_routed_experts > 0 {
                    self.dspark_n_routed_experts
                } else {
                    self.n_routed_experts
                },
                if self.dspark_n_activated_experts > 0 {
                    self.dspark_n_activated_experts
                } else {
                    self.n_activated_experts
                },
            )
        }
    }

    /// Compression ratio of `layer_id` (`0` = sliding-window only).
    pub fn ratio(&self, layer_id: usize) -> usize {
        self.compress_ratios.get(layer_id).copied().unwrap_or(0)
    }

    pub fn is_kv_source(&self, layer_id: usize) -> bool {
        layer_id < self.n_layers && self.kv_source_layers.contains(&layer_id)
    }

    pub fn is_index_source(&self, layer_id: usize) -> bool {
        layer_id < self.n_layers && self.index_source_layers.contains(&layer_id)
    }

    /// The layer whose compressed-KV cache `layer_id` reads — the nearest
    /// `kv_source_layers` entry at or before it. `None` when no source precedes
    /// it, which for a `ratio > 0` layer is a malformed config.
    pub fn kv_source_for(&self, layer_id: usize) -> Option<usize> {
        self.kv_source_layers
            .iter()
            .copied()
            .filter(|&s| s <= layer_id)
            .max()
    }

    /// The layer whose Indexer top-k `layer_id` reuses.
    pub fn index_source_for(&self, layer_id: usize) -> Option<usize> {
        self.index_source_layers
            .iter()
            .copied()
            .filter(|&s| s <= layer_id)
            .max()
    }

    /// True when `layer_id`'s Indexer must mask its scores by the candidate
    /// blocks published upstream (`0 <= candidate_source_layer < layer_id`).
    pub fn uses_candidates(&self, layer_id: usize) -> bool {
        matches!(self.candidate_source_layer, Some(c) if c < layer_id)
    }

    /// True when `layer_id` publishes the candidate blocks.
    pub fn is_candidate_source(&self, layer_id: usize) -> bool {
        self.candidate_source_layer == Some(layer_id)
    }

    /// `head_dim - rope_head_dim` — the un-rotated ("nope") part of each head.
    pub fn nope_head_dim(&self) -> usize {
        self.head_dim.saturating_sub(self.rope_head_dim)
    }

    /// Attention output width per o-LoRA group: `n_heads · head_dim / o_groups`.
    pub fn dim_per_group(&self) -> usize {
        self.n_heads * self.head_dim / self.n_groups.max(1)
    }

    /// Checkpoint prefix for `layer_id` — backbone layers live under `layers.N`,
    /// DSpark stages under `mtp.N` (stage-relative index).
    pub fn layer_prefix(&self, layer_id: usize) -> String {
        if layer_id < self.n_layers {
            format!("layers.{layer_id}")
        } else {
            format!("mtp.{}", layer_id - self.n_layers)
        }
    }

    /// Parse the released **HuggingFace `config.json`** (nested `text_config` /
    /// `vision_config`) *or* the flat reference `inference/config.json`. Both
    /// name the same quantities differently, so every field is looked up under
    /// both spellings and the first hit wins.
    pub fn from_config(v: &Value) -> Result<Self> {
        let c = Cfg::new(v);
        let (expert_fp4, weight_block_size) = parse_quant(&c);
        Ok(DeepseekV41Spec {
            vocab_size: c.need("vocab_size", "vocab_size")?,
            dim: c.need("hidden_size", "dim")?,
            n_layers: c.need("num_hidden_layers", "n_layers")?,
            hc_mult: c.u("hc_mult").unwrap_or(4),
            n_heads: c.need("num_attention_heads", "n_heads")?,
            head_dim: c.need("head_dim", "head_dim")?,
            rope_head_dim: c.ua("qk_rope_head_dim", "rope_head_dim").unwrap_or(64),
            q_lora_rank: c.u("q_lora_rank").unwrap_or(0),
            n_groups: c.u("o_groups").unwrap_or(1),
            o_lora_rank: c.need("o_lora_rank", "o_lora_rank")?,
            compress_ratios: c.vec_u("compress_ratios"),
            kv_source_layers: c.va("kv_source_layer_ids", "kv_source_layers"),
            index_source_layers: c.va("index_source_layer_ids", "index_source_layers"),
            index_head_dim: c.u("index_head_dim").unwrap_or(0),
            index_n_heads: c.u("index_n_heads").unwrap_or(0),
            index_topk: c.u("index_topk").unwrap_or(0),
            // `candidate_source_layer < 0` (flat config) disables candidate
            // blocks; HF omits the key entirely when off.
            candidate_source_layer: c
                .get("candidate_source_layer_id")
                .or_else(|| c.get("candidate_source_layer"))
                .and_then(Value::as_i64)
                .and_then(|x| (x >= 0).then_some(x as usize)),
            candidate_topk_blocks: c.u("candidate_topk_blocks").unwrap_or(0),
            candidate_block_size: c.u("candidate_block_size").unwrap_or(0),
            window_size: c
                .ua("sliding_window", "window_size")
                .unwrap_or(usize::MAX / 4),
            moe_intermediate_size: c.need("moe_intermediate_size", "moe_inter_dim")?,
            n_routed_experts: c.need("n_routed_experts", "n_routed_experts")?,
            n_activated_experts: c
                .ua("num_experts_per_tok", "n_activated_experts")
                .unwrap_or(8),
            n_shared_experts: c.u("n_shared_experts").unwrap_or(0),
            score_func: parse_score_func(&c)?,
            gate_temp: c.f("gate_temp").unwrap_or(1.0) as f32,
            norm_topk_prob: c.b("norm_topk_prob").unwrap_or(true),
            route_scale: c.fa("routed_scaling_factor", "route_scale").unwrap_or(1.0) as f32,
            swiglu_limit: c.f("swiglu_limit").unwrap_or(0.0) as f32,
            rms_norm_eps: c.fa("rms_norm_eps", "norm_eps").unwrap_or(1e-6) as f32,
            rope_theta: c.f("rope_theta").unwrap_or(10000.0),
            compress_rope_theta: c.f("compress_rope_theta").unwrap_or(10000.0),
            // YaRN lives under `rope_scaling` (HF) or as flat `rope_*` keys.
            original_seq_len: c
                .obj_u("rope_scaling", "original_max_position_embeddings")
                .or_else(|| c.u("original_seq_len"))
                .unwrap_or(0),
            rope_factor: c
                .obj_f("rope_scaling", "factor")
                .or_else(|| c.f("rope_factor"))
                .unwrap_or(1.0),
            beta_fast: c
                .obj_f("rope_scaling", "beta_fast")
                .or_else(|| c.f("beta_fast"))
                .unwrap_or(32.0),
            beta_slow: c
                .obj_f("rope_scaling", "beta_slow")
                .or_else(|| c.f("beta_slow"))
                .unwrap_or(1.0),
            hc_mult_sinkhorn_iters: c.u("hc_sinkhorn_iters").unwrap_or(20),
            hc_eps: c.f("hc_eps").unwrap_or(1e-6) as f32,
            engram: parse_engram(&c),
            vision: parse_vision(&c),
            image_token_id: c.u("image_token_id").unwrap_or(0),
            n_mtp_layers: c
                .ua("num_nextn_predict_layers", "n_mtp_layers")
                .unwrap_or(0),
            dspark_block_size: c.u("dspark_block_size").unwrap_or(0),
            dspark_noise_token_id: c.u("dspark_noise_token_id").unwrap_or(0),
            dspark_target_layer_ids: c.vec_u("dspark_target_layer_ids"),
            dspark_markov_rank: c.u("dspark_markov_rank").unwrap_or(256),
            dspark_n_routed_experts: c.u("dspark_n_routed_experts").unwrap_or(0),
            dspark_n_activated_experts: c
                .ua("dspark_num_experts_per_tok", "dspark_n_activated_experts")
                .unwrap_or(0),
            weight_block_size,
            expert_fp4,
        })
    }

    /// Structural checks that would otherwise surface as a silently wrong model.
    pub fn validate(&self) -> Result<()> {
        let want = self.n_layers + self.n_mtp_layers;
        if self.compress_ratios.len() < want {
            return Err(anyhow!(
                "deepseek_v41: compress_ratios has {} entries, need {} (n_layers {} + n_mtp_layers {})",
                self.compress_ratios.len(),
                want,
                self.n_layers,
                self.n_mtp_layers
            ));
        }
        if self.rope_head_dim > self.head_dim {
            return Err(anyhow!(
                "deepseek_v41: rope_head_dim {} exceeds head_dim {}",
                self.rope_head_dim,
                self.head_dim
            ));
        }
        if self.n_groups == 0 || !(self.n_heads * self.head_dim).is_multiple_of(self.n_groups) {
            return Err(anyhow!(
                "deepseek_v41: o_groups {} must divide n_heads·head_dim {}",
                self.n_groups,
                self.n_heads * self.head_dim
            ));
        }
        // A layer that attends over compressed KV must have a source at or before
        // it, otherwise it would read an empty cache and silently drop half its
        // context. Same for the Indexer that selects those positions.
        for il in 0..self.n_layers {
            if self.ratio(il) == 0 {
                continue;
            }
            if self.kv_source_for(il).is_none() {
                return Err(anyhow!(
                    "deepseek_v41: layer {il} has compress_ratio {} but no kv_source_layer at or before it",
                    self.ratio(il)
                ));
            }
            if self.index_head_dim > 0 && self.index_source_for(il).is_none() {
                return Err(anyhow!(
                    "deepseek_v41: layer {il} compresses KV but no index_source_layer precedes it"
                ));
            }
        }
        // Sources must agree with their consumers on the ratio: a source
        // compresses at its own ratio, so a consumer reading it at a different
        // one would mis-map every compressed position to a wrong token span.
        for il in 0..self.n_layers {
            if self.ratio(il) == 0 {
                continue;
            }
            let src = self.kv_source_for(il).expect("checked above");
            if self.ratio(src) != self.ratio(il) {
                return Err(anyhow!(
                    "deepseek_v41: layer {il} (ratio {}) reads kv source layer {src} (ratio {})",
                    self.ratio(il),
                    self.ratio(src)
                ));
            }
        }
        if let Some(e) = &self.engram {
            if e.num_embeddings.len() != e.layer_ids.len() {
                return Err(anyhow!(
                    "deepseek_v41: engram_num_embeddings has {} entries for {} engram layers",
                    e.num_embeddings.len(),
                    e.layer_ids.len()
                ));
            }
            if e.max_ngram_size < 2 || e.n_heads == 0 || e.head_dim == 0 {
                return Err(anyhow!(
                    "deepseek_v41: engram needs max_ngram_size>=2, n_heads>0, head_dim>0"
                ));
            }
        }
        Ok(())
    }
}

/// One tensor the port reads, named and shaped as the checkpoint stores it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorNeed {
    pub name: String,
    /// Logical `[out, in]` (or `[n]`). FP4-packed tensors are stored with half
    /// as many columns — see [`packed_fp4`](Self::packed_fp4).
    pub shape: Vec<usize>,
    /// The checkpoint stores this quantized, with a companion `<stem>.scale`.
    pub quantized: bool,
    /// Stored as `[out, in / 2]` nibble pairs rather than one byte per element.
    pub packed_fp4: bool,
}

impl TensorNeed {
    fn plain(name: String, shape: Vec<usize>) -> Self {
        TensorNeed {
            name,
            shape,
            quantized: false,
            packed_fp4: false,
        }
    }
    fn quant(name: String, shape: Vec<usize>) -> Self {
        TensorNeed {
            name,
            shape,
            quantized: true,
            packed_fp4: false,
        }
    }
    fn fp4(name: String, shape: Vec<usize>, packed: bool) -> Self {
        TensorNeed {
            name,
            shape,
            quantized: true,
            packed_fp4: packed,
        }
    }
    /// Shape as it sits on disk.
    pub fn stored_shape(&self) -> Vec<usize> {
        if self.packed_fp4 && self.shape.len() == 2 {
            vec![self.shape[0], self.shape[1] / 2]
        } else {
            self.shape.clone()
        }
    }
    /// Companion scale tensor's name, when there is one.
    pub fn scale_name(&self) -> Option<String> {
        self.quantized
            .then(|| {
                self.name
                    .strip_suffix(".weight")
                    .map(|s| format!("{s}.scale"))
            })
            .flatten()
    }
}

/// Accumulator for [`DeepseekV41Spec::expected_tensors`], so each tensor is one
/// line rather than a four-line `push`.
struct Needs(Vec<TensorNeed>);

impl Needs {
    fn plain(&mut self, name: String, shape: &[usize]) {
        self.0.push(TensorNeed::plain(name, shape.to_vec()));
    }
    fn quant(&mut self, name: String, shape: &[usize]) {
        self.0.push(TensorNeed::quant(name, shape.to_vec()));
    }
    fn fp4(&mut self, name: String, shape: &[usize], packed: bool) {
        self.0.push(TensorNeed::fp4(name, shape.to_vec(), packed));
    }
}

impl DeepseekV41Spec {
    /// Every tensor the port can read, across all of its paths, with the shape it
    /// must have.
    ///
    /// This is the union over prefill, decode, vision and DSpark — a host
    /// planning which shards or byte ranges to fetch wants the union, not one
    /// path's subset. Tensors the checkpoint carries but the port never reads
    /// (the DSpark stages' `gate.bias_vl`, since drafts are text-only) are
    /// deliberately absent.
    ///
    /// It is derived from the same predicates the builders use — `is_kv_source`,
    /// `is_index_source`, the Engram placement — so it stays in step with them;
    /// `builder_requests_match_the_manifest` pins that at toy scale.
    pub fn expected_tensors(&self) -> Vec<TensorNeed> {
        let mut n = Needs(Vec::new());
        let d = self.dim;
        n.plain("embed.weight".into(), &[self.vocab_size, d]);
        n.plain("head.weight".into(), &[self.vocab_size, d]);
        n.plain("norm.weight".into(), &[d]);
        for il in 0..self.n_layers + self.n_mtp_layers {
            self.push_layer(&mut n, il);
        }
        self.push_dspark(&mut n);
        self.push_vision(&mut n);
        n.0
    }

    /// One decoder layer: the hyper-connection mixes, attention, whatever CSA2
    /// and Engram machinery this layer carries, and the MoE FFN.
    fn push_layer(&self, n: &mut Needs, il: usize) {
        let lp = self.layer_prefix(il);
        // The DSpark draft stages reuse the layer layout but skip everything
        // that only makes sense in the backbone: no compressor, no Indexer, no
        // Engram, no VL routing bias.
        let backbone = il < self.n_layers;
        let (d, hc) = (self.dim, self.hc_mult);
        let mix_hc = (2 + hc) * hc;
        for side in ["attn", "ffn"] {
            n.plain(format!("{lp}.hc_{side}_fn"), &[mix_hc, hc * d]);
            n.plain(format!("{lp}.hc_{side}_base"), &[mix_hc]);
            n.plain(format!("{lp}.hc_{side}_scale"), &[3]);
        }
        n.plain(format!("{lp}.attn_norm.weight"), &[d]);
        n.plain(format!("{lp}.ffn_norm.weight"), &[d]);
        self.push_attn(n, &lp);
        if backbone {
            self.push_csa2(n, &lp, il);
            self.push_engram(n, &lp, il);
        }
        self.push_moe(n, &lp, il, backbone);
    }

    /// The attention weights every layer has: the Q LoRA pair, the single MQA
    /// `kv` latent that serves as both key and value, the per-head sink, and the
    /// grouped output LoRA.
    fn push_attn(&self, n: &mut Needs, lp: &str) {
        let (d, nh, hd, ql) = (self.dim, self.n_heads, self.head_dim, self.q_lora_rank);
        n.quant(format!("{lp}.attn.wq_a.weight"), &[ql, d]);
        n.plain(format!("{lp}.attn.q_norm.weight"), &[ql]);
        n.quant(format!("{lp}.attn.wq_b.weight"), &[nh * hd, ql]);
        n.quant(format!("{lp}.attn.wkv.weight"), &[hd, d]);
        n.plain(format!("{lp}.attn.kv_norm.weight"), &[hd]);
        n.plain(format!("{lp}.attn.attn_sink"), &[nh]);
        let og = self.n_groups * self.o_lora_rank;
        n.quant(
            format!("{lp}.attn.wo_a.weight"),
            &[og, self.dim_per_group()],
        );
        n.quant(format!("{lp}.attn.wo_b.weight"), &[d, og]);
    }

    /// CSA2, which splits across two *different* layer sets: a kv source carries
    /// the compressor and the Indexer's key half, an index source carries the
    /// Indexer's query half.
    fn push_csa2(&self, n: &mut Needs, lp: &str, il: usize) {
        let (d, hd, ihd) = (self.dim, self.head_dim, self.index_head_dim);
        if self.is_kv_source(il) {
            n.plain(format!("{lp}.attn.compressor.wkv.weight"), &[hd, d]);
            n.plain(format!("{lp}.attn.compressor.norm.weight"), &[hd]);
            // ratio 1 compresses nothing, so there is no group to weight
            if self.ratio(il) > 1 {
                n.plain(format!("{lp}.attn.compressor.wgate.weight"), &[hd, d]);
            }
            if ihd > 0 {
                n.plain(format!("{lp}.attn.indexer.wk.weight"), &[ihd, hd]);
                n.plain(format!("{lp}.attn.indexer.k_norm.weight"), &[ihd]);
            }
        }
        if self.is_index_source(il) && ihd > 0 {
            let inh = self.index_n_heads;
            n.quant(
                format!("{lp}.attn.indexer.wq_b.weight"),
                &[inh * ihd, self.q_lora_rank],
            );
            n.plain(format!("{lp}.attn.indexer.weights_proj.weight"), &[inh, d]);
        }
    }

    /// The n-gram hash memory, on the two layers that carry one.
    fn push_engram(&self, n: &mut Needs, lp: &str, il: usize) {
        let Some(e) = &self.engram else { return };
        let Some(k) = e.layer_hash_index(il) else {
            return;
        };
        let (d, hc) = (self.dim, self.hc_mult);
        n.quant(
            format!("{lp}.engram.embed.weight"),
            &[e.num_embeddings[k], e.head_dim],
        );
        n.quant(
            format!("{lp}.engram.wkv.weight"),
            &[d * (hc + 1), e.n_hash_cols() * e.head_dim],
        );
        n.plain(format!("{lp}.engram.q_weight"), &[hc, d]);
        n.plain(format!("{lp}.engram.k_weight"), &[hc, d]);
    }

    /// The MoE FFN: the router, the shared expert, and the routed expert bank.
    /// The bank size comes from [`Self::moe_dims`], since a DSpark stage routes
    /// over its own smaller set.
    fn push_moe(&self, n: &mut Needs, lp: &str, il: usize, backbone: bool) {
        let (d, inter) = (self.dim, self.moe_intermediate_size);
        let (n_e, _) = self.moe_dims(il);
        n.plain(format!("{lp}.ffn.gate.weight"), &[n_e, d]);
        n.plain(format!("{lp}.ffn.gate.bias"), &[n_e]);
        // the VL routing bias only exists — and is only read — on a
        // vision-enabled checkpoint, and DSpark drafts are text
        if self.vision.is_some() && backbone {
            n.plain(format!("{lp}.ffn.gate.bias_vl"), &[n_e]);
        }
        let se = self.n_shared_experts.max(1) * inter;
        n.quant(format!("{lp}.ffn.shared_experts.w1.weight"), &[se, d]);
        n.quant(format!("{lp}.ffn.shared_experts.w3.weight"), &[se, d]);
        n.quant(format!("{lp}.ffn.shared_experts.w2.weight"), &[d, se]);
        for e in 0..n_e {
            let p4 = self.expert_fp4;
            n.fp4(format!("{lp}.ffn.experts.{e}.w1.weight"), &[inter, d], p4);
            n.fp4(format!("{lp}.ffn.experts.{e}.w3.weight"), &[inter, d], p4);
            n.fp4(format!("{lp}.ffn.experts.{e}.w2.weight"), &[d, inter], p4);
        }
    }

    /// The DSpark pieces that sit outside the draft stages' layer loop: the
    /// projection folding the main model's target hidden states into the first
    /// stage, and the last stage's Markov and confidence heads.
    fn push_dspark(&self, n: &mut Needs) {
        if self.n_mtp_layers == 0 {
            return;
        }
        let d = self.dim;
        let first = self.layer_prefix(self.n_layers);
        n.quant(
            format!("{first}.main_proj.weight"),
            &[d, d * self.dspark_target_layer_ids.len()],
        );
        n.plain(format!("{first}.main_norm.weight"), &[d]);
        let last = self.layer_prefix(self.n_layers + self.n_mtp_layers - 1);
        let r = self.dspark_markov_rank;
        n.plain(format!("{last}.norm.weight"), &[d]);
        n.plain(
            format!("{last}.markov_head.embed.weight"),
            &[self.vocab_size, r],
        );
        n.plain(
            format!("{last}.markov_head.head.weight"),
            &[self.vocab_size, r],
        );
        n.plain(format!("{last}.confidence_head.proj.weight"), &[1, d + r]);
    }

    /// The ViT, the aligner, and the three learned span delimiters.
    fn push_vision(&self, n: &mut Needs) {
        let Some(vs) = &self.vision else { return };
        let (d, vd, vi) = (self.dim, vs.dim, vs.inter_dim);
        n.plain(
            "vision.patch_embed.proj.weight".into(),
            &[vd, 3 * vs.patch_size * vs.patch_size],
        );
        n.plain("vision.patch_embed.proj.bias".into(), &[vd]);
        for b in 0..vs.n_layers {
            let bp = format!("vision.blocks.{b}");
            n.plain(format!("{bp}.norm1.weight"), &[vd]);
            n.plain(format!("{bp}.attn.wqkv.weight"), &[3 * vd, vd]);
            n.plain(format!("{bp}.attn.wqkv.bias"), &[3 * vd]);
            n.plain(format!("{bp}.attn.wo.weight"), &[vd, vd]);
            n.plain(format!("{bp}.attn.wo.bias"), &[vd]);
            n.plain(format!("{bp}.norm2.weight"), &[vd]);
            // w1 is the fused SwiGLU gate+up pair, hence 2·inter
            n.plain(format!("{bp}.mlp.w1.weight"), &[2 * vi, vd]);
            n.plain(format!("{bp}.mlp.w2.weight"), &[vd, vi]);
        }
        n.plain("vision.norm.weight".into(), &[vd]);
        let r = vs.downsample_ratio.max(1);
        n.plain("aligner.w1.weight".into(), &[d, vd * r * r]);
        n.plain("aligner.w1.bias".into(), &[d]);
        n.plain("aligner.w2.weight".into(), &[d, d]);
        n.plain("aligner.w2.bias".into(), &[d]);
        for t in ["image_start", "image_end", "image_newline"] {
            n.plain(t.into(), &[d]);
        }
    }
}

/// YaRN "NTK-by-parts" inverse frequency for rope dimension pair `i`.
///
/// Dimensions whose wavelength already fits inside the training context keep
/// their frequency, those far beyond it are divided by `factor`, and the
/// `beta_fast..beta_slow` band in between is faded across with a linear ramp.
/// `original_seq_len == 0` disables the correction. Mirrors
/// `precompute_freqs_cis` — note V4.1 applies **no** `mscale`, so the attention
/// softmax scale stays `head_dim^-0.5`.
pub fn yarn_inv_freq(
    i: usize,
    rope_dim: usize,
    base: f64,
    original_seq_len: usize,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> f64 {
    let d = rope_dim as f64;
    let freq = 1.0 / base.powf(2.0 * i as f64 / d);
    if original_seq_len == 0 || factor <= 1.0 {
        return freq;
    }
    let corrected = |rot: f64| {
        d * (original_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
    };
    let low = corrected(beta_fast).floor().max(0.0);
    let high = corrected(beta_slow).ceil().min(d - 1.0);
    let ramp = ((i as f64 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
    let smooth = 1.0 - ramp;
    freq / factor * (1.0 - smooth) + freq * smooth
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ga_config() -> Value {
        serde_json::from_str(GA_CONFIG_JSON).expect("released config.json parses")
    }

    /// Verbatim from `deepseek-ai/DeepSeek-V4.1-Flash/config.json`, trimmed to
    /// the keys the spec reads.
    const GA_CONFIG_JSON: &str = r#"{
  "architectures": ["DeepseekV41ForCausalLM"],
  "model_type": "deepseek_v41",
  "image_token_id": 129264,
  "quantization_config": {
    "quant_method": "fp8", "activation_scheme": "dynamic",
    "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"
  },
  "text_config": {
    "model_type": "deepseek_v41_text",
    "vocab_size": 129280, "hidden_size": 5120, "moe_intermediate_size": 2304,
    "num_hidden_layers": 40, "num_attention_heads": 64, "num_key_value_heads": 1,
    "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 1280,
    "o_lora_rank": 1024, "o_groups": 8, "hidden_act": "silu",
    "swiglu_limit": 10.0, "rms_norm_eps": 1e-20, "tie_word_embeddings": false,
    "max_position_embeddings": 1048576, "rope_theta": 10000,
    "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                     "beta_slow": 1, "original_max_position_embeddings": 65536},
    "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
    "norm_topk_prob": true, "routed_scaling_factor": 1.5, "sliding_window": 128,
    "compress_ratios": [0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
    "compress_rope_theta": 160000,
    "kv_source_layer_ids": [2,8,14,20],
    "index_source_layer_ids": [2,8,14,20,24,28,32,36],
    "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512,
    "candidate_source_layer_id": 20, "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4, "engram_vocab_size": 16000000,
    "engram_n_heads": 8, "engram_head_dim": 256, "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092,
    "num_nextn_predict_layers": 3, "dspark_block_size": 5,
    "dspark_noise_token_id": 128799, "dspark_target_layer_ids": [37,38,39],
    "dspark_markov_rank": 256, "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3
  },
  "vision_config": {
    "model_type": "deepseek_v41_vision", "num_hidden_layers": 32,
    "hidden_size": 1024, "num_attention_heads": 16, "intermediate_size": 2816,
    "patch_size": 14, "rope_theta": 10000, "downsample_ratio": 3,
    "max_image_tokens": 1024, "min_pixels": 295936, "max_wh_ratio": null
  }
}"#;

    #[test]
    fn from_config_matches_real_checkpoint() {
        let s = DeepseekV41Spec::from_config(&ga_config()).unwrap();
        s.validate().unwrap();
        assert_eq!((s.vocab_size, s.dim, s.n_layers), (129280, 5120, 40));
        assert_eq!((s.n_heads, s.head_dim, s.rope_head_dim), (64, 512, 64));
        assert_eq!((s.q_lora_rank, s.o_lora_rank, s.n_groups), (1280, 1024, 8));
        assert_eq!(s.dim_per_group(), 4096); // matches wo_a [8·1024, 4096]
        assert_eq!((s.n_routed_experts, s.n_activated_experts), (384, 6));
        assert_eq!(s.moe_intermediate_size, 2304);
        assert_eq!(s.score_func, ScoreFunc::SqrtSoftplus);
        assert_eq!(s.route_scale, 1.5);
        assert_eq!(s.swiglu_limit, 10.0);
        assert_eq!(s.rms_norm_eps, 1e-20);
        assert_eq!(s.window_size, 128);
        assert_eq!(s.compress_ratios.len(), 43);
        assert_eq!(s.kv_source_layers, vec![2, 8, 14, 20]);
        assert_eq!(s.index_source_layers, vec![2, 8, 14, 20, 24, 28, 32, 36]);
        assert_eq!(
            (s.index_n_heads, s.index_head_dim, s.index_topk),
            (32, 128, 512)
        );
        assert_eq!(s.candidate_source_layer, Some(20));
        assert_eq!((s.candidate_topk_blocks, s.candidate_block_size), (2048, 8));
        assert_eq!((s.hc_mult, s.hc_mult_sinkhorn_iters), (4, 20));
        assert_eq!((s.original_seq_len, s.rope_factor), (65536, 16.0));
        assert_eq!(s.compress_rope_theta, 160000.0);
        assert_eq!(s.image_token_id, 129264);
        assert_eq!(s.n_mtp_layers, 3);
        assert_eq!(s.dspark_target_layer_ids, vec![37, 38, 39]);
        assert_eq!(s.moe_dims(39), (384, 6));
        assert_eq!(s.moe_dims(40), (128, 3)); // DSpark stage 0 has its own bank
        let e = s.engram.as_ref().unwrap();
        assert_eq!(e.layer_ids, vec![1, 14]);
        assert_eq!(e.num_embeddings, vec![384006168, 384016682]);
        assert_eq!((e.max_ngram_size, e.n_heads, e.head_dim), (4, 8, 256));
        assert_eq!(e.compressed_vocab_size, 99092);
        assert_eq!(e.n_hash_cols(), 24); // wkv in = 24·256 = 6144 ✓
        let v = s.vision.as_ref().unwrap();
        assert_eq!((v.n_layers, v.dim, v.n_heads), (32, 1024, 16));
        assert_eq!(
            (v.inter_dim, v.patch_size, v.downsample_ratio),
            (2816, 14, 3)
        );
        assert_eq!(v.max_wh_ratio, None);
    }

    fn ga_inference_config() -> Value {
        serde_json::from_str(GA_INFERENCE_JSON).expect("released inference/config.json parses")
    }

    /// Verbatim from `deepseek-ai/DeepSeek-V4.1-Flash/inference/config.json`.
    const GA_INFERENCE_JSON: &str = r#"{
  "vocab_size": 129280, "dim": 5120, "moe_inter_dim": 2304, "n_layers": 40,
  "n_mtp_layers": 3, "dspark_block_size": 5, "dspark_noise_token_id": 128799,
  "dspark_target_layer_ids": [37,38,39], "dspark_markov_rank": 256,
  "dspark_n_routed_experts": 128, "dspark_n_activated_experts": 3,
  "n_heads": 64, "n_routed_experts": 384, "n_shared_experts": 1,
  "n_activated_experts": 6, "score_func": "sqrtsoftplus", "route_scale": 1.5,
  "swiglu_limit": 10.0, "q_lora_rank": 1280, "head_dim": 512,
  "rope_head_dim": 64, "norm_eps": 1e-20, "o_groups": 8, "o_lora_rank": 1024,
  "window_size": 128, "kv_source_layers": [2,8,14,20],
  "index_source_layers": [2,8,14,20,24,28,32,36], "original_seq_len": 65536,
  "rope_theta": 10000, "rope_factor": 16, "beta_fast": 32, "beta_slow": 1,
  "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512,
  "candidate_source_layer": 20, "candidate_topk_blocks": 2048,
  "candidate_block_size": 8, "hc_mult": 4, "hc_sinkhorn_iters": 20,
  "hc_eps": 1e-06, "engram_layer_ids": [1,14], "engram_vocab_size": 16000000,
  "engram_num_embeddings": [384006168, 384016682], "engram_max_ngram_size": 4,
  "engram_pad_id": 2, "engram_compressed_vocab_size": 99092,
  "dtype": "fp8", "expert_dtype": "fp4", "compress_rope_theta": 160000,
  "compress_ratios": [0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
  "vision_n_layers": 32, "vision_dim": 1024, "vision_n_heads": 16,
  "vision_inter_dim": 2816, "vision_patch_size": 14,
  "vision_downsample_ratio": 3, "vision_max_n_token": 1024,
  "vision_min_pixels": 295936, "vision_max_wh_ratio": null,
  "image_token_id": 129264, "engram_n_heads": 8, "engram_head_dim": 256,
  "vision_rope_theta": 10000
}"#;

    /// The flat `inference/config.json` must produce the same spec as the HF one
    /// for every field both files carry.
    #[test]
    fn flat_inference_config_agrees_with_hf_config() {
        let hf = DeepseekV41Spec::from_config(&ga_config()).unwrap();
        let flat = DeepseekV41Spec::from_config(&ga_inference_config()).unwrap();
        flat.validate().unwrap();
        assert_eq!(hf, flat);
    }

    #[test]
    fn kv_and_index_sourcing_follows_csa2_layout() {
        let s = DeepseekV41Spec::from_config(&ga_config()).unwrap();
        // ratio-2 half: layers 2..19 read the nearest of {2,8,14}
        assert_eq!(s.kv_source_for(2), Some(2));
        assert_eq!(s.kv_source_for(7), Some(2));
        assert_eq!(s.kv_source_for(8), Some(8));
        assert_eq!(s.kv_source_for(13), Some(8));
        assert_eq!(s.kv_source_for(19), Some(14));
        // ratio-1 half: everything from 20 on reads layer 20
        assert_eq!(s.kv_source_for(20), Some(20));
        assert_eq!(s.kv_source_for(39), Some(20));
        // index sources are denser in the second half
        assert_eq!(s.index_source_for(23), Some(20));
        assert_eq!(s.index_source_for(24), Some(24));
        assert_eq!(s.index_source_for(27), Some(24));
        assert_eq!(s.index_source_for(39), Some(36));
        // only layer 20 publishes candidates; only later index sources consume them
        assert!(s.is_candidate_source(20));
        assert!(!s.uses_candidates(20));
        assert!(!s.uses_candidates(14));
        assert!(s.uses_candidates(24));
        // layers 0/1 are pure sliding-window and own nothing
        assert_eq!(s.ratio(0), 0);
        assert!(!s.is_kv_source(0));
        assert!(s.is_kv_source(2));
        assert!(s.is_index_source(24));
        assert!(!s.is_kv_source(24)); // index source WITHOUT its own compressor
    }

    #[test]
    fn validate_rejects_orphaned_compressed_layer() {
        let mut s = DeepseekV41Spec::from_config(&ga_config()).unwrap();
        s.kv_source_layers = vec![8, 14, 20]; // drop the layer-2 source
        let err = s.validate().unwrap_err().to_string();
        assert!(err.contains("layer 2"), "{err}");
    }

    #[test]
    fn validate_rejects_ratio_mismatch_against_source() {
        let mut s = DeepseekV41Spec::from_config(&ga_config()).unwrap();
        s.compress_ratios[5] = 4; // layer 5 reads layer 2's ratio-2 cache
        let err = s.validate().unwrap_err().to_string();
        assert!(err.contains("reads kv source layer 2"), "{err}");
    }

    #[test]
    fn yarn_matches_reference_precompute_freqs_cis() {
        // Reference `precompute_freqs_cis(dim=64, base=160000, original_seq_len=65536,
        // factor=16, beta_fast=32, beta_slow=1)` — the released compressed-KV table.
        // low/high bracket the ramp; outside it the two limbs must be exact.
        let f = |i| yarn_inv_freq(i, 64, 160000.0, 65536, 16.0, 32.0, 1.0);
        let plain = |i: usize| 1.0f64 / 160000f64.powf(2.0 * i as f64 / 64.0);
        // i = 0 is deep inside the "already fits" region → unscaled
        assert!((f(0) - plain(0)).abs() < 1e-18);
        // the last pair is far beyond the training context → divided by factor
        assert!((f(31) - plain(31) / 16.0).abs() < 1e-18 * plain(31));
        // monotone decreasing, and every value between the two limbs
        for i in 0..32 {
            let v = f(i);
            assert!(
                v <= plain(i) + 1e-18 && v >= plain(i) / 16.0 - 1e-18,
                "i={i}"
            );
        }
        // YaRN off reproduces the plain table exactly
        for i in 0..32 {
            assert_eq!(
                yarn_inv_freq(i, 64, 10000.0, 0, 16.0, 32.0, 1.0),
                plain_base(i, 10000.0)
            );
        }
        fn plain_base(i: usize, base: f64) -> f64 {
            1.0 / base.powf(2.0 * i as f64 / 64.0)
        }
    }
}
