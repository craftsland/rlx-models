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

//! Incremental decode: one token per `run()`, over a **latent-only** cache.
//!
//! Only layers that own their KV keep a cache — a KV-sharing layer attends to
//! its source's rows (cache + this step's new row), so the nine layers that
//! share layer 35's KV add nothing. One cached row is
//! `kv_lora + rope_l` floats: 576 on a RoPE layer, 512 on a NoPE one.
//!
//! The cache and position are graph **inputs**; the step returns one packed
//! output, `[ logits | new latent row per owning layer ]`, which
//! [`DecodeSession`] writes back.
//!
//! ## Why decode stops at `index_topk`
//!
//! Below the budget DSA keeps every key (see [`crate::indexer`]), so decode is
//! exact without an indexer-key cache. Past it the selection is real, and
//! [`build_mistral_large4_decode_flow`] rejects such a capacity rather than
//! running dense attention, which would be a different model.

use anyhow::{Result, anyhow, bail};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow, WeightSource};
use rlx_glm5next::common::rms_norm;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use std::collections::HashMap;

use crate::attention::{DecodeKv, emit_attention_decode};
use crate::config::MistralLarge4Config;
use crate::flow::{EMBED_KEY, attn_dims, emit_ffn_site, emit_head};
use crate::rope::emit_pos_tables;

/// Input name of owning layer `i`'s latent cache, `[1, cap, latent_width(i)]`.
pub fn kv_cache_input(layer: usize) -> String {
    format!("state.kv{layer}")
}
/// Key-validity mask shared by every layer, `[1, cap + 1]`.
pub const MASK_INPUT: &str = "state.mask";
/// This token's absolute position, `[1, 1]`.
pub const POS_INPUT: &str = "pos";

/// Offsets of everything packed into the decode output.
#[derive(Debug, Clone)]
pub struct DecodeLayout {
    pub head_width: usize,
    /// Per owning layer, in layer order: `(layer, offset, width)`.
    pub kv: Vec<(usize, usize, usize)>,
    pub total: usize,
}

impl DecodeLayout {
    pub fn new(cfg: &MistralLarge4Config, head_width: usize) -> Self {
        let mut off = head_width;
        let mut kv = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            if cfg.owns_kv(i) {
                let w = cfg.latent_width(i);
                kv.push((i, off, w));
                off += w;
            }
        }
        Self {
            head_width,
            kv,
            total: off,
        }
    }
}

/// Build the single-token decode graph for a cache of `cap` positions.
pub fn build_mistral_large4_decode_flow(
    cfg: &MistralLarge4Config,
    weights: &mut WeightMap,
    cap: usize,
    with_lm_head: bool,
) -> Result<(BuiltModel, DecodeLayout)> {
    build_mistral_large4_decode_flow_with_source(
        cfg,
        &mut WeightMapSource(weights),
        cap,
        with_lm_head,
    )
}

/// [`build_mistral_large4_decode_flow`] over any [`WeightSource`].
pub fn build_mistral_large4_decode_flow_with_source(
    cfg: &MistralLarge4Config,
    weights: &mut dyn WeightSource,
    cap: usize,
    with_lm_head: bool,
) -> Result<(BuiltModel, DecodeLayout)> {
    cfg.validate()?;
    if !cfg.dsa_is_dense(cap + 1) {
        bail!(
            "mistral-large4: decode capacity {cap} (+1 for the current token) exceeds \
             index_topk = {}, where DSA stops keeping every key; decoding past it needs \
             a cached-indexer path that is not emitted yet",
            cfg.index_topk
        );
    }
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let head_width = if with_lm_head { cfg.vocab_size } else { hidden };
    let layout = DecodeLayout::new(cfg, head_width);

    let mut flow = ModelFlow::new("mistral_large4_decode")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, 1], f))
        .input(POS_INPUT, Shape::new(&[1, 1], f));
    for &(i, _, w) in &layout.kv {
        flow = flow.input(kv_cache_input(i), Shape::new(&[1, cap, w], f));
    }
    flow = flow
        .input(MASK_INPUT, Shape::new(&[1, cap + 1], f))
        .embed(EMBED_KEY);

    let cfg = cfg.clone();
    let total = layout.total;
    flow = flow.plugin_named("decode", move |emit, prev| {
        let x = prev
            .ok_or_else(|| anyhow!("decode needs the embedding"))?
            .hir_id();
        let pos = emit.flow_input(POS_INPUT)?.hir_id();
        let mask = emit.flow_input(MASK_INPUT)?.hir_id();
        let t = emit_pos_tables(emit, &cfg, pos, 1)?;
        let mut x2d = {
            let mut gb = HirMut::new(emit.hir());
            gb.reshape_(x, vec![1, hidden as i64])
        };
        let ad = attn_dims(&cfg, 1);
        let mut kvs: HashMap<usize, DecodeKv> = HashMap::new();
        let mut rows: Vec<HirNodeId> = Vec::new();

        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("blk.{i}");
            let normed = rms_norm(
                emit,
                &format!("{prefix}.attn_norm"),
                x2d,
                hidden,
                cfg.rms_norm_eps,
            )?;
            let cache = if cfg.owns_kv(i) {
                Ok((emit.flow_input(&kv_cache_input(i))?.hir_id(), mask, cap))
            } else {
                let src = cfg.kv_source[i];
                Err(*kvs
                    .get(&src)
                    .ok_or_else(|| anyhow!("mistral-large4: KV source {src} missing"))?)
            };
            let (attn, kv, row) =
                emit_attention_decode(emit, &prefix, normed, ad, cfg.rope_dim(i), cache, &t)?;
            if let Some(row) = row {
                kvs.insert(i, kv);
                rows.push(row);
            }
            x2d = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(x2d, attn)
            };
            x2d = emit_ffn_site(emit, &cfg, i, x2d, 1)?;
        }

        let head = emit_head(emit, &cfg, x2d, 1, with_lm_head)?;
        let mut gb = HirMut::new(emit.hir());
        let mut all = vec![head];
        all.extend(rows);
        let out = gb.concat_(all, 1);
        Ok(Some(emit.wrap(out, Shape::new(&[1, total], f))))
    });

    let built = flow.output("packed").build_with(weights, None)?;
    Ok((built, layout))
}

/// Host-side decode state.
pub struct DecodeSession {
    pub layout: DecodeLayout,
    cap: usize,
    pos: usize,
    pos_buf: [f32; 1],
    kv: Vec<Vec<f32>>,
    names: Vec<String>,
    mask: Vec<f32>,
}

impl DecodeSession {
    pub fn new(layout: DecodeLayout, cap: usize) -> Self {
        let mut mask = vec![0f32; cap + 1];
        mask[cap] = 1.0; // the current token always sees itself
        Self {
            kv: layout
                .kv
                .iter()
                .map(|&(_, _, w)| vec![0f32; cap * w])
                .collect(),
            names: layout
                .kv
                .iter()
                .map(|&(i, _, _)| kv_cache_input(i))
                .collect(),
            layout,
            cap,
            pos: 0,
            pos_buf: [0.0],
            mask,
        }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    /// Bytes of cache held on the host.
    pub fn state_bytes(&self) -> usize {
        (self.kv.iter().map(Vec::len).sum::<usize>() + self.mask.len()) * 4
    }

    /// Input bindings for the next step (everything but `input_ids`).
    pub fn inputs(&mut self) -> Vec<(&str, &[f32])> {
        self.pos_buf[0] = self.pos as f32;
        let mut v: Vec<(&str, &[f32])> = vec![(POS_INPUT, &self.pos_buf[..])];
        for (n, c) in self.names.iter().zip(&self.kv) {
            v.push((n.as_str(), c.as_slice()));
        }
        v.push((MASK_INPUT, self.mask.as_slice()));
        v
    }

    /// Store a step's new latent rows; returns the logits (or hidden) slice.
    pub fn commit<'a>(&mut self, packed: &'a [f32]) -> Result<&'a [f32]> {
        if packed.len() < self.layout.total {
            bail!(
                "decode output is {} wide, expected {}",
                packed.len(),
                self.layout.total
            );
        }
        if self.pos >= self.cap {
            bail!(
                "decode ran past the compiled cache capacity of {}",
                self.cap
            );
        }
        for (li, &(_, off, w)) in self.layout.kv.iter().enumerate() {
            self.kv[li][self.pos * w..(self.pos + 1) * w].copy_from_slice(&packed[off..off + w]);
        }
        self.mask[self.pos] = 1.0;
        self.pos += 1;
        Ok(&packed[..self.layout.head_width])
    }
}
