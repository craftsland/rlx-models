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

//! The Mistral Large 4 text prefill graph.
//!
//! ```text
//!   token_embd
//!     → N × ( attn_norm → [DSA indexer | reuse] → MLA (+ KV share) → +residual
//!             ffn_norm  → dense MLP | MoE                          → +residual )
//!     → output_norm → output → c·tanh(·/c)
//! ```
//!
//! Two things cross layers, which is why the stack is one plugin rather than
//! one per layer: the latent KV a later layer shares (`kv_source`), and the DSA
//! selection a `"shared"` indexer layer reuses.

use anyhow::{Result, anyhow, bail};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow, WeightSource};
use rlx_glm5next::common::{linear, rms_norm, scalar};
use rlx_ir::hir::HirMut;
use rlx_ir::op::Activation;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use std::collections::HashMap;

use crate::attention::{AttnDims, Latent, emit_attention};
use crate::config::MistralLarge4Config;
use crate::indexer::{IndexerDims, emit_dsa_bias};
use crate::moe::{MoeDims, emit_dense_mlp, emit_moe};
use crate::rope::emit_pos_tables;

/// GGUF's token-embedding tensor name.
pub const EMBED_KEY: &str = "token_embd.weight";

/// What a prefill graph emits.
#[derive(Debug, Clone, Copy)]
pub struct PrefillOptions {
    /// Emit `logits [1, seq, vocab]`; otherwise the normed final hidden state.
    pub with_lm_head: bool,
    /// Run the DSA selection even where it is provably the identity
    /// (`seq <= index_topk`). Tests only.
    pub force_dsa: bool,
}

impl Default for PrefillOptions {
    fn default() -> Self {
        Self {
            with_lm_head: true,
            force_dsa: false,
        }
    }
}

pub(crate) fn attn_dims(cfg: &MistralLarge4Config, seq: usize) -> AttnDims {
    AttnDims {
        hidden: cfg.hidden_size,
        heads: cfg.num_attention_heads,
        kv_lora: cfg.kv_lora_rank,
        qk_head: cfg.qk_head_dim(),
        v_head: cfg.v_head_dim,
        eps: cfg.rms_norm_eps,
        seq,
    }
}

pub(crate) fn moe_dims(cfg: &MistralLarge4Config, seq: usize) -> MoeDims {
    MoeDims {
        hidden: cfg.hidden_size,
        moe_inter: cfg.moe_intermediate_size,
        n_routed: cfg.n_routed_experts,
        top_k: cfg.num_experts_per_tok,
        routed_scaling: cfg.routed_scaling_factor,
        seq,
    }
}

/// The FFN half of layer `i`, residual included.
pub(crate) fn emit_ffn_site(
    emit: &mut Emit<'_>,
    cfg: &MistralLarge4Config,
    i: usize,
    x2d: HirNodeId,
    seq: usize,
) -> Result<HirNodeId> {
    let prefix = format!("blk.{i}");
    let normed = rms_norm(
        emit,
        &format!("{prefix}.ffn_norm"),
        x2d,
        cfg.hidden_size,
        cfg.rms_norm_eps,
    )?;
    let y = if cfg.moe_layers[i] {
        emit_moe(emit, &prefix, normed, moe_dims(cfg, seq))?
    } else {
        emit_dense_mlp(emit, &prefix, normed)?
    };
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.add(x2d, y))
}

/// `output_norm → output → softcap` over `x2d` (`[rows, hidden]`).
pub(crate) fn emit_head(
    emit: &mut Emit<'_>,
    cfg: &MistralLarge4Config,
    x2d: HirNodeId,
    rows: usize,
    with_lm_head: bool,
) -> Result<HirNodeId> {
    let x = rms_norm(emit, "output_norm", x2d, cfg.hidden_size, cfg.rms_norm_eps)?;
    if !with_lm_head {
        return Ok(x);
    }
    let key = if cfg.tie_word_embeddings {
        EMBED_KEY
    } else {
        "output.weight"
    };
    let y = linear(emit, key, x)?;
    let Some(cap) = cfg.final_logit_softcapping else {
        return Ok(y);
    };
    let inv = scalar(emit, "ml4.softcap.inv", 1.0 / cap);
    let c = scalar(emit, "ml4.softcap.c", cap);
    let mut gb = HirMut::new(emit.hir());
    let y = gb.mul(y, inv);
    let y = gb.activation(
        Activation::Tanh,
        y,
        Shape::new(&[rows, cfg.vocab_size], DType::F32),
    );
    Ok(gb.mul(y, c))
}

/// Emit layers `layers` (in order) over `x2d` (`[seq, hidden]`).
///
/// `layers` is normally `0..n`; a test may pass any increasing list (e.g. the
/// blocks of a range-fetched subset), as long as every KV-sharing layer's
/// source and every `"shared"` indexer's provider come earlier in it.
pub(crate) fn emit_stack(
    emit: &mut Emit<'_>,
    cfg: &MistralLarge4Config,
    mut x2d: HirNodeId,
    seq: usize,
    layers: &[usize],
    t: &crate::rope::PosTables,
    force_dsa: bool,
) -> Result<HirNodeId> {
    let hidden = cfg.hidden_size;
    let ad = attn_dims(cfg, seq);
    let mut latents: HashMap<usize, Latent> = HashMap::new();
    // `None` inside means "selection is the identity" — distinct from no
    // full indexer having run yet, which is an error for a shared layer.
    let mut dsa: Option<Option<HirNodeId>> = None;
    for &i in layers {
        let prefix = format!("blk.{i}");
        let normed = rms_norm(
            emit,
            &format!("{prefix}.attn_norm"),
            x2d,
            hidden,
            cfg.rms_norm_eps,
        )?;
        if cfg.indexer_full[i] {
            let id = IndexerDims {
                hidden,
                n_heads: cfg.index_n_heads,
                head_dim: cfg.index_head_dim,
                rope_dim: cfg.rope_dim(i),
                topk: cfg.index_topk,
                norm_eps: cfg.indexer_norm_eps,
                seq,
                force_emit: force_dsa,
            };
            dsa = Some(emit_dsa_bias(emit, &prefix, normed, id, t)?);
        }
        let Some(bias) = dsa else {
            bail!(
                "mistral-large4: layer {i} reuses a DSA selection but no full indexer layer precedes it"
            );
        };
        let shared = if cfg.owns_kv(i) {
            None
        } else {
            let src = cfg.kv_source[i];
            Some(*latents.get(&src).ok_or_else(|| {
                anyhow!("mistral-large4: layer {i}'s KV source {src} was not emitted before it")
            })?)
        };
        let (attn, latent) =
            emit_attention(emit, &prefix, normed, ad, cfg.rope_dim(i), shared, t, bias)?;
        if cfg.owns_kv(i) {
            latents.insert(i, latent);
        }
        x2d = {
            let mut gb = HirMut::new(emit.hir());
            gb.add(x2d, attn)
        };
        x2d = emit_ffn_site(emit, cfg, i, x2d, seq)?;
    }
    Ok(x2d)
}

/// Prefill positions `0..seq` and their tables.
fn prefill_tables(
    emit: &mut Emit<'_>,
    cfg: &MistralLarge4Config,
    seq: usize,
) -> Result<crate::rope::PosTables> {
    let pos = emit.synth_param(
        "ml4.pos",
        (0..seq).map(|p| p as f32).collect(),
        Shape::new(&[seq, 1], DType::F32),
    );
    emit_pos_tables(emit, cfg, pos, seq)
}

/// Build the prefill graph for a fixed `seq`. Input `input_ids [1, seq]`.
pub fn build_mistral_large4_text_flow(
    cfg: &MistralLarge4Config,
    weights: &mut WeightMap,
    seq: usize,
    opts: PrefillOptions,
) -> Result<BuiltModel> {
    build_mistral_large4_text_flow_with_source(cfg, &mut WeightMapSource(weights), seq, opts)
}

/// [`build_mistral_large4_text_flow`] over any [`WeightSource`] — pass
/// `rlx_core::flow_bridge::PackedWeightLoaderSource` over a GGUF loader to run
/// the projections and expert banks packed.
pub fn build_mistral_large4_text_flow_with_source(
    cfg: &MistralLarge4Config,
    weights: &mut dyn WeightSource,
    seq: usize,
    opts: PrefillOptions,
) -> Result<BuiltModel> {
    cfg.validate()?;
    if seq == 0 {
        bail!("mistral-large4: seq must be >= 1");
    }
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let out_width = if opts.with_lm_head {
        cfg.vocab_size
    } else {
        hidden
    };
    let cfg = cfg.clone();
    let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();

    let flow = ModelFlow::new("mistral_large4")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, seq], f))
        .embed(EMBED_KEY)
        .plugin_named("decoder", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("decoder needs the embedding"))?
                .hir_id();
            let t = prefill_tables(emit, &cfg, seq)?;
            let x2d = {
                let mut gb = HirMut::new(emit.hir());
                gb.reshape_(x, vec![seq as i64, hidden as i64])
            };
            let x2d = emit_stack(emit, &cfg, x2d, seq, &layers, &t, opts.force_dsa)?;
            let y = emit_head(emit, &cfg, x2d, seq, opts.with_lm_head)?;
            let mut gb = HirMut::new(emit.hir());
            let y = gb.reshape_(y, vec![1, seq as i64, out_width as i64]);
            Ok(Some(emit.wrap(y, Shape::new(&[1, seq, out_width], f))))
        });
    flow.output("logits").build_with(weights, None)
}

/// Input name of [`build_mistral_large4_blocks_with_source`].
pub const HIDDEN_INPUT: &str = "hidden";

/// Build a graph running only `layers` over a given hidden state:
/// `hidden [1, seq, hidden_size]` in, the residual stream after the last
/// layer out (no final norm). For validating blocks of a checkpoint too large
/// to load whole — see `scripts/mistral_large4_subset.py`.
pub fn build_mistral_large4_blocks_with_source(
    cfg: &MistralLarge4Config,
    weights: &mut dyn WeightSource,
    seq: usize,
    layers: &[usize],
    force_dsa: bool,
) -> Result<BuiltModel> {
    cfg.validate()?;
    if seq == 0 || layers.is_empty() {
        bail!("mistral-large4: need seq >= 1 and at least one layer");
    }
    if layers.windows(2).any(|w| w[0] >= w[1]) || layers.iter().any(|&i| i >= cfg.num_hidden_layers)
    {
        bail!(
            "mistral-large4: layers {layers:?} must be increasing and < {}",
            cfg.num_hidden_layers
        );
    }
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let cfg = cfg.clone();
    let layers = layers.to_vec();
    let shape = Shape::new(&[1, seq, hidden], f);
    let s2 = shape.clone();
    let flow = ModelFlow::new("mistral_large4_blocks")
        .with_profile(CompileProfile::llama32_prefill())
        .input(HIDDEN_INPUT, shape)
        .plugin_named("blocks", move |emit, _prev| {
            let x = emit.flow_input(HIDDEN_INPUT)?.hir_id();
            let t = prefill_tables(emit, &cfg, seq)?;
            let x2d = {
                let mut gb = HirMut::new(emit.hir());
                gb.reshape_(x, vec![seq as i64, hidden as i64])
            };
            let y = emit_stack(emit, &cfg, x2d, seq, &layers, &t, force_dsa)?;
            let mut gb = HirMut::new(emit.hir());
            let y = gb.reshape_(y, vec![1, seq as i64, hidden as i64]);
            Ok(Some(emit.wrap(y, s2.clone())))
        });
    flow.output("hidden_out").build_with(weights, None)
}
