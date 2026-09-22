// RLX — versatile ML compiler + runtime. GPLv3.
//! **DeepSeek-V4.1** single-token decode step with a KV cache.
//!
//! Re-running the prefill for every token is O(n²); this is the O(1)-per-token
//! path. The graph takes the new token id plus the cached state and returns
//! `logits [1, vocab]` together with the pieces the host appends to the cache.
//!
//! Three kinds of state cross a step, and they are not symmetric:
//!
//! * **Sliding-window KV** — one roped latent per position, per layer. Every
//!   layer owns its own.
//! * **Compressed KV and index keys** — owned only by `kv_source_layers` and read
//!   by everything up to the next source, so the cache is keyed by *source*
//!   layer, not by consumer.
//! * **The compressor's partial group** — a `ratio > 1` layer emits one latent
//!   per `ratio` tokens, so on the other `ratio - 1` steps it has nothing to
//!   append and instead hands back the running `wkv`/`wgate` pair to accumulate.
//!
//! The Hyper-Connection pre-mix is *not* cross-step state: it is threaded within
//! one forward and restarts from the one-hot mix for each token, exactly as in
//! prefill. Engram look-back *is* cross-step, and the host supplies it by passing
//! the earlier tokens as `history` to
//! [`EngramHashPlan::hash_ids`](crate::dsv41_engram::EngramHashPlan::hash_ids).
//!
//! Correctness is the standard KV-cache induction: fed the same tokens, the
//! accumulated cache equals what prefill computes internally, so decode logits
//! equal prefill logits — which is what `decode_matches_prefill` asserts.

use crate::dsv41::DeepseekV41Spec;
use crate::dsv41_block::{
    AttnOut, Ctx, Qkv, RopeTables, build_attn_out, build_qkv, open_mask, rope_table,
};
use crate::dsv41_csa::{Compressed, CompressorW, CsaShared, pool_groups, publish_topk_mask};
use crate::dsv41_engram::build_v41_engram;
use crate::dsv41_graph::V41Inputs;
use crate::dsv41_hc::{
    HcSide, hc_mean, hc_reduce, hc_sublayer_plain, identity_pre_mix, main_hidden_node,
};
use crate::dsv41_moe::build_v41_moe;
use crate::standard_decoder::{load_dense_dequant, rope_tail};
use crate::weight_loader::WeightLoader;
use anyhow::{Result, anyhow};
use rlx_ir::GraphExt;
use rlx_ir::graph::{Graph, NodeId};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// Names of the per-step cache inputs and outputs, so a host can wire buffers
/// without re-deriving the layout.
pub mod names {
    /// Sliding-window KV cache for layer `il`: `[cache_len, head_dim]`.
    pub fn window_kv(il: usize) -> String {
        format!("winkv.{il}")
    }
    /// The new roped window latent to append: `[1, head_dim]`.
    pub fn window_kv_new(il: usize) -> String {
        format!("kvnew.{il}")
    }
    /// Compressed KV owned by source layer `src`: `[compress_len, head_dim]`.
    pub fn compress_kv(src: usize) -> String {
        format!("compkv.{src}")
    }
    /// A newly completed compressed latent to append: `[1, head_dim]`.
    pub fn compress_kv_new(src: usize) -> String {
        format!("compnew.{src}")
    }
    /// Index keys owned by source layer `src`: `[compress_len, index_head_dim]`.
    pub fn index_k(src: usize) -> String {
        format!("indexk.{src}")
    }
    /// A newly completed index key to append: `[1, index_head_dim]`.
    pub fn index_k_new(src: usize) -> String {
        format!("indexknew.{src}")
    }
    /// The compressor's accumulated group so far: `[filled, head_dim]` each.
    pub fn group_kv(src: usize) -> String {
        format!("groupkv.{src}")
    }
    pub fn group_score(src: usize) -> String {
        format!("groupscore.{src}")
    }
    /// This step's contribution to a still-incomplete group: `[1, head_dim]`.
    pub fn group_kv_new(src: usize) -> String {
        format!("groupkvnew.{src}")
    }
    pub fn group_score_new(src: usize) -> String {
        format!("groupscorenew.{src}")
    }
    /// The compressor's carried group, **replacing** whatever was there.
    ///
    /// A single decode step contributes one row to a group and the cache appends
    /// it; a multi-token chunk can complete several groups and be left holding a
    /// different partial one entirely, so it hands back the whole group rather
    /// than a delta. Append semantics would concatenate the old leftover onto
    /// the new one and silently widen the group.
    pub fn group_kv_set(src: usize) -> String {
        format!("groupkvset.{src}")
    }
    pub fn group_score_set(src: usize) -> String {
        format!("groupscoreset.{src}")
    }
    /// Engram n-gram row ids for layer `il`: `[1, n_hash_cols]`.
    ///
    /// An **input**, not a baked constant. The rows change every position while
    /// the graph's shape does not, so baking them would make a compiled step
    /// reusable in shape but wrong in content — a decode loop that caches
    /// sessions would silently reuse another position's n-grams.
    pub fn engram_rows(il: usize) -> String {
        format!("engramrows.{il}")
    }
}

/// Per-source compressed-cache geometry for one decode step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressStep {
    pub source_layer: usize,
    pub ratio: usize,
    /// Latents already in the cache when the step begins (`pos / ratio`).
    pub len_before: usize,
    /// Latents visible to this step's query (`(pos + 1) / ratio`).
    pub len_after: usize,
    /// Whether this step completes a group and emits a new latent.
    pub fires: bool,
    /// Entries of the partial group the host carries in (`pos % ratio`).
    pub group_filled: usize,
}

impl CompressStep {
    fn new(source_layer: usize, ratio: usize, pos: usize) -> Self {
        CompressStep {
            source_layer,
            ratio,
            len_before: pos / ratio,
            len_after: (pos + 1) / ratio,
            fires: (pos + 1).is_multiple_of(ratio),
            group_filled: pos % ratio,
        }
    }
}

/// The cache geometry a step needs, derived from `pos` alone — the host uses it
/// to size buffers before building the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V41DecodePlan {
    pub pos: usize,
    /// Window entries the host must supply. One fewer than the window, because
    /// this step's own token takes the last slot: the ring holds `window_size`
    /// positions *including* the new one, so feeding `min(pos, window_size)`
    /// would let the query see one position further back than prefill does.
    pub cache_len: usize,
    /// One entry per distinct `kv_source_layer` reachable at this position.
    pub sources: Vec<CompressStep>,
}

impl V41DecodePlan {
    pub fn new(spec: &DeepseekV41Spec, pos: usize) -> Self {
        let mut sources: Vec<CompressStep> = Vec::new();
        for il in 0..spec.n_layers {
            if !spec.is_kv_source(il) {
                continue;
            }
            let ratio = spec.ratio(il);
            if ratio == 0 {
                continue;
            }
            sources.push(CompressStep::new(il, ratio, pos));
        }
        V41DecodePlan {
            pos,
            cache_len: pos.min(spec.window_size.saturating_sub(1)),
            sources,
        }
    }

    pub fn for_source(&self, src: usize) -> Option<&CompressStep> {
        self.sources.iter().find(|s| s.source_layer == src)
    }
}

/// Per-source compressed geometry for a **chunk** of `k` tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkCompress {
    pub source_layer: usize,
    pub ratio: usize,
    /// Latents in the cache when the chunk begins (`start / ratio`).
    pub len_before: usize,
    /// Latents visible to the chunk's *last* query (`(start + k) / ratio`).
    pub len_after: usize,
    /// Rows of a partial group the host carries in (`start % ratio`).
    pub group_filled: usize,
    /// Rows of a partial group the chunk leaves behind.
    pub group_left: usize,
}

impl ChunkCompress {
    fn new(source_layer: usize, ratio: usize, start: usize, k: usize) -> Self {
        ChunkCompress {
            source_layer,
            ratio,
            len_before: start / ratio,
            len_after: (start + k) / ratio,
            group_filled: start % ratio,
            group_left: (start + k) % ratio,
        }
    }

    /// New latents this chunk completes.
    pub fn produced(&self) -> usize {
        self.len_after - self.len_before
    }
}

/// The geometry of a `k`-token step starting at absolute position `start`.
///
/// Generalizes both existing paths: `start = 0` with an empty cache is a
/// prefill, and `k = 1` is a decode step. Having one plan for all three is what
/// lets a prompt be processed in chunks and a speculative draft block be
/// verified in a single pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V41ChunkPlan {
    pub start: usize,
    pub k: usize,
    /// Window rows the host supplies, covering positions
    /// `start - cache_len .. start`.
    pub cache_len: usize,
    pub sources: Vec<ChunkCompress>,
}

impl V41ChunkPlan {
    pub fn new(spec: &DeepseekV41Spec, start: usize, k: usize) -> Self {
        let sources = (0..spec.n_layers)
            .filter(|&il| spec.is_kv_source(il) && spec.ratio(il) > 0)
            .map(|il| ChunkCompress::new(il, spec.ratio(il), start, k))
            .collect();
        V41ChunkPlan {
            start,
            k,
            // the window holds at most `window_size - 1` positions from before
            // the chunk; the chunk's own tokens take the remaining slots
            cache_len: start.min(spec.window_size.saturating_sub(1)),
            sources,
        }
    }

    pub fn for_source(&self, src: usize) -> Option<&ChunkCompress> {
        self.sources.iter().find(|s| s.source_layer == src)
    }

    /// Absolute position of query row `i`.
    pub fn pos(&self, i: usize) -> usize {
        self.start + i
    }
}

/// Host-side KV cache for a decode loop.
///
/// Holds exactly the three kinds of state [`build_deepseek_v41_decode`] expects,
/// and knows how to feed a step and fold its outputs back in. The window is a
/// bounded rolling buffer; the compressed caches grow.
pub struct V41DecodeCache {
    window_size: usize,
    head_dim: usize,
    index_head_dim: usize,
    /// `[il] -> rows of head_dim`, most recent last, at most `window_size - 1`.
    window: Vec<Vec<f32>>,
    compress: HashMap<usize, Vec<f32>>,
    index_k: HashMap<usize, Vec<f32>>,
    group_kv: HashMap<usize, Vec<f32>>,
    group_score: HashMap<usize, Vec<f32>>,
}

impl V41DecodeCache {
    pub fn new(spec: &DeepseekV41Spec) -> Self {
        V41DecodeCache {
            window_size: spec.window_size,
            head_dim: spec.head_dim,
            index_head_dim: spec.index_head_dim,
            window: vec![Vec::new(); spec.n_layers],
            compress: HashMap::new(),
            index_k: HashMap::new(),
            group_kv: HashMap::new(),
            group_score: HashMap::new(),
        }
    }

    /// Named input buffers for the step described by `plan`, ready to hand to
    /// `Session::run` alongside `input_ids`.
    pub fn step_inputs(&self, plan: &V41DecodePlan) -> Vec<(String, &[f32])> {
        let mut out: Vec<(String, &[f32])> = Vec::new();
        if plan.cache_len > 0 {
            for (il, rows) in self.window.iter().enumerate() {
                debug_assert_eq!(rows.len(), plan.cache_len * self.head_dim);
                out.push((names::window_kv(il), rows.as_slice()));
            }
        }
        for s in &plan.sources {
            let src = s.source_layer;
            if s.len_before > 0 {
                if let Some(c) = self.compress.get(&src) {
                    out.push((names::compress_kv(src), c.as_slice()));
                }
                if self.index_head_dim > 0
                    && let Some(k) = self.index_k.get(&src)
                {
                    out.push((names::index_k(src), k.as_slice()));
                }
            }
            if s.ratio > 1 && s.fires && s.group_filled > 0 {
                if let Some(v) = self.group_kv.get(&src) {
                    out.push((names::group_kv(src), v.as_slice()));
                }
                if let Some(v) = self.group_score.get(&src) {
                    out.push((names::group_score(src), v.as_slice()));
                }
            }
        }
        out
    }

    /// Seed the cache from a batched prefill.
    ///
    /// `names`/`values` are what
    /// [`crate::dsv41_graph::build_deepseek_v41_prefill`] returned with
    /// `emit_cache`. Priming leaves the cache exactly where stepping the same
    /// prompt one token at a time would have left it, so decoding continues from
    /// position `seq` — which is what makes prompt processing one graph run
    /// instead of `seq` of them.
    ///
    /// Unlike [`Self::apply`] there is no plan: a prefill emits whole tensors
    /// rather than one row, and it has already dropped the groups it consumed.
    pub fn prime(&mut self, names: &[String], values: &[Vec<f32>]) -> Result<()> {
        if names.len() != values.len() {
            return Err(anyhow!(
                "deepseek_v41 prefill: {} cache names for {} tensors",
                names.len(),
                values.len()
            ));
        }
        self.window.iter_mut().for_each(Vec::clear);
        self.compress.clear();
        self.index_k.clear();
        self.group_kv.clear();
        self.group_score.clear();
        for (name, v) in names.iter().zip(values) {
            self.fold(name, v)?;
        }
        Ok(())
    }

    /// Fold one named cache tensor in, whether it came from a prefill or a step.
    fn fold(&mut self, name: &str, v: &[f32]) -> Result<()> {
        let Some((tag, idx)) = name.rsplit_once('.') else {
            return Ok(()); // `logits`
        };
        let Ok(i) = idx.parse::<usize>() else {
            return Ok(());
        };
        match tag {
            "kvnew" => {
                let rows = self.window.get_mut(i).ok_or_else(|| {
                    anyhow!("deepseek_v41: window KV for layer {i}, which does not exist")
                })?;
                rows.extend_from_slice(v);
                // the new token occupies the last ring slot, so the buffer
                // handed to the NEXT step holds at most window_size - 1
                let cap = self.window_size.saturating_sub(1) * self.head_dim;
                if rows.len() > cap {
                    rows.drain(..rows.len() - cap);
                }
            }
            "compnew" => self.compress.entry(i).or_default().extend_from_slice(v),
            "indexknew" => self.index_k.entry(i).or_default().extend_from_slice(v),
            "groupkvnew" => self.group_kv.entry(i).or_default().extend_from_slice(v),
            "groupscorenew" => self.group_score.entry(i).or_default().extend_from_slice(v),
            "groupkvset" => {
                self.group_kv.insert(i, v.to_vec());
            }
            "groupscoreset" => {
                self.group_score.insert(i, v.to_vec());
            }
            _ => {}
        }
        Ok(())
    }

    /// Window rows cached for layer `il`.
    ///
    /// Exposed because "the tokens came out right" is a weak check on a cache: an
    /// over-long compressed cache is silently truncated to the declared input
    /// width, so it reads correctly for a while and only misaligns once the next
    /// latent is appended past the stale one. Asserting the lengths catches that
    /// at the round it happens.
    pub fn window_len(&self, il: usize) -> usize {
        self.window
            .get(il)
            .map(|r| r.len() / self.head_dim.max(1))
            .unwrap_or(0)
    }

    /// Compressed latents cached for source layer `src`.
    pub fn compressed_len(&self, src: usize) -> usize {
        self.compress
            .get(&src)
            .map(|c| c.len() / self.head_dim.max(1))
            .unwrap_or(0)
    }

    /// Rows of a partial compressor group carried for `src`.
    pub fn group_len(&self, src: usize) -> usize {
        self.group_kv
            .get(&src)
            .map(|c| c.len() / self.head_dim.max(1))
            .unwrap_or(0)
    }

    /// Named input buffers for a `k`-token chunk.
    pub fn chunk_inputs(&self, plan: &V41ChunkPlan) -> Vec<(String, &[f32])> {
        let mut out: Vec<(String, &[f32])> = Vec::new();
        if plan.cache_len > 0 {
            for (il, rows) in self.window.iter().enumerate() {
                out.push((names::window_kv(il), rows.as_slice()));
            }
        }
        for s in &plan.sources {
            let src = s.source_layer;
            if s.len_before > 0 {
                if let Some(c) = self.compress.get(&src) {
                    out.push((names::compress_kv(src), c.as_slice()));
                }
                if self.index_head_dim > 0
                    && let Some(k) = self.index_k.get(&src)
                {
                    out.push((names::index_k(src), k.as_slice()));
                }
            }
            if s.group_filled > 0 {
                if let Some(v) = self.group_kv.get(&src) {
                    out.push((names::group_kv(src), v.as_slice()));
                }
                if let Some(v) = self.group_score.get(&src) {
                    out.push((names::group_score(src), v.as_slice()));
                }
            }
        }
        out
    }

    /// Fold a chunk's outputs back in.
    ///
    /// The plan is needed to *clear* a partial group the chunk consumed exactly.
    /// A chunk ending on a group boundary has no leftover rows to hand back, so
    /// it emits nothing for that source — and "emits nothing" has to mean "the
    /// group is now empty", not "leave the old one".
    ///
    /// In practice a stale group left there heals itself: the next chunk starts
    /// on a multiple of `ratio`, so it declares no group input at all and then
    /// overwrites the entry. This is therefore hygiene rather than a fix for
    /// wrong output — but a cache that misdescribes its own contents is not
    /// something to leave standing, and the length accessors below are only
    /// meaningful if it does not.
    pub fn apply_chunk(
        &mut self,
        plan: &V41ChunkPlan,
        names: &[String],
        values: &[Vec<f32>],
    ) -> Result<()> {
        if names.len() != values.len() {
            return Err(anyhow!(
                "deepseek_v41 chunk: {} names for {} outputs",
                names.len(),
                values.len()
            ));
        }
        for (name, v) in names.iter().zip(values) {
            self.fold(name, v)?;
        }
        for s in &plan.sources {
            if s.group_left == 0 {
                self.group_kv.remove(&s.source_layer);
                self.group_score.remove(&s.source_layer);
            }
        }
        Ok(())
    }

    /// Fold one step's outputs back in. `names`/`values` are the name list
    /// [`build_deepseek_v41_decode`] returned and the session's outputs, in the
    /// same order.
    pub fn apply(
        &mut self,
        plan: &V41DecodePlan,
        names: &[String],
        values: &[Vec<f32>],
    ) -> Result<()> {
        if names.len() != values.len() {
            return Err(anyhow!(
                "deepseek_v41 decode: {} output names for {} outputs",
                names.len(),
                values.len()
            ));
        }
        for (name, v) in names.iter().zip(values) {
            self.fold(name, v)?;
        }
        // a completed group clears what it consumed
        for s in &plan.sources {
            if s.ratio > 1 && s.fires {
                self.group_kv.remove(&s.source_layer);
                self.group_score.remove(&s.source_layer);
            }
        }
        Ok(())
    }
}

/// State threaded through one decode step's layer loop — the decode counterpart
/// of prefill's `SharedAttn`.
#[derive(Default)]
pub(crate) struct StepShared {
    pub csa: CsaShared,
    pub len_after: usize,
}

/// Build one decode step at absolute position `pos`.
///
/// Returns the graph, its parameters, and the output-name list in graph-output
/// order (`logits` first). Inputs are named by [`names`]; the host sizes them
/// from [`V41DecodePlan`].
pub fn build_deepseek_v41_decode(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    pos: usize,
    inputs: &V41Inputs,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    spec.validate()?;
    let plan = V41DecodePlan::new(spec, pos);
    let mut ctx = Ctx::new("deepseek_v41_decode", spec, weights, packed, 1);
    let (d, hc) = (spec.dim, spec.hc_mult);
    let rope = StepRope::build(&mut ctx, &plan);
    let mut step = StepOutputs::default();

    let input_ids = ctx.g.input("input_ids", Shape::new(&[1, 1], DType::I32));
    let (embed_w, _, _) =
        load_dense_dequant(&mut ctx.g, &mut ctx.params, ctx.weights, "embed.weight")?;
    let h0 = ctx.g.gather_(embed_w, input_ids, 0);
    let h0 = ctx.g.reshape_(h0, vec![1, 1, d as i64]);
    let ones = ctx.konst("v41d.hc.ones", vec![1f32; hc], &[1, hc, 1]);
    let mut h = ctx.g.mul(h0, ones);
    let mut pre_mix = identity_pre_mix(&mut ctx.g, &mut ctx.params, 1, hc);

    let mut shared = StepShared::default();
    let mut main_hiddens: Vec<NodeId> = Vec::new();
    for il in 0..spec.n_layers {
        let lp = spec.layer_prefix(il);

        if let Some(e) = &spec.engram
            && e.layer_hash_index(il).is_some()
        {
            let cols = e.n_hash_cols();
            let n_eng = e.layer_ids.len();
            if inputs.engram_rows.len() != n_eng * cols {
                return Err(anyhow!(
                    "deepseek_v41 decode: engram_rows has {} entries, expected {} \
                     (1 token × {n_eng} layers × {cols} cols)",
                    inputs.engram_rows.len(),
                    n_eng * cols
                ));
            }
            let row_ids = ctx
                .g
                .input(names::engram_rows(il), Shape::new(&[1, cols], DType::F32));
            h = build_v41_engram(&mut ctx, &lp, h, row_ids, None, e)?;
        }

        if inputs.emit_main_hidden && spec.dspark_target_layer_ids.contains(&il) {
            main_hiddens.push(hc_mean(&mut ctx.g, h, 1, d));
        }

        let rope_l = rope.for_layer(spec.ratio(il));
        let (nh, pre_attn) =
            hc_sublayer_plain(&mut ctx, HcSide::Attn, &lp, h, pre_mix, |ctx, xa| {
                build_decode_attention(ctx, il, xa, &plan, &rope_l, &mut shared, &mut step)
            })?;
        h = nh;
        let (nh, pre_ffn) =
            hc_sublayer_plain(&mut ctx, HcSide::Ffn, &lp, h, pre_attn, |ctx, xf| {
                build_v41_moe(ctx, il, xf, 1, None)
            })?;
        h = nh;
        pre_mix = pre_ffn;
    }

    let x = hc_reduce(&mut ctx.g, h, pre_mix, 1, hc);
    let gain = ctx.norm("norm.weight")?;
    let zb = ctx.zero_bias("v41d.zb.d", d);
    let eps = ctx.eps();
    let x = ctx.g.rms_norm(x, gain, zb, eps);
    let logits = ctx.project("head.weight", x, 1, spec.vocab_size)?;
    let logits = ctx.g.reshape_(logits, vec![1, spec.vocab_size as i64]);

    let mut all = vec![logits];
    let mut names_out = vec!["logits".to_string()];
    if inputs.emit_main_hidden {
        all.push(main_hidden_node(&mut ctx.g, &main_hiddens, spec, 1)?);
        names_out.push("main_hidden".to_string());
    }
    all.extend(step.nodes);
    names_out.extend(step.names);
    let (g, params) = ctx.finish(all);
    Ok((g, params, names_out))
}

/// The cache pieces one step hands back, in graph-output order.
#[derive(Default)]
pub(crate) struct StepOutputs {
    pub nodes: Vec<NodeId>,
    pub names: Vec<String>,
}

impl StepOutputs {
    fn push(&mut self, node: NodeId, name: String) {
        self.nodes.push(node);
        self.names.push(name);
    }
}

/// The RoPE tables one decode step needs: the current position for `q` and the
/// window KV, plus — for each ratio that *completes* a group this step — the
/// position its new latent stands for.
pub(crate) struct StepRope {
    win: (NodeId, NodeId, NodeId),
    comp: (NodeId, NodeId, NodeId),
    latent: HashMap<usize, (NodeId, NodeId)>,
}

impl StepRope {
    pub(crate) fn build(ctx: &mut Ctx<'_>, plan: &V41DecodePlan) -> Self {
        let spec = ctx.spec;
        let (rd, pos) = (ctx.rd(), plan.pos);
        let (theta, ctheta) = (spec.rope_theta, spec.compress_rope_theta);
        let yarn = (spec.original_seq_len > 0 && spec.rope_factor > 1.0).then_some((
            spec.original_seq_len,
            spec.rope_factor,
            spec.beta_fast,
            spec.beta_slow,
        ));
        let win = rope_table(
            &mut ctx.g,
            &mut ctx.params,
            &[pos],
            rd,
            theta,
            None,
            "d.win",
        );
        let comp = rope_table(
            &mut ctx.g,
            &mut ctx.params,
            &[pos],
            rd,
            ctheta,
            yarn,
            "d.comp",
        );

        let firing: Vec<usize> = plan
            .sources
            .iter()
            .filter(|s| s.fires)
            .map(|s| s.ratio)
            .collect();
        let mut latent = HashMap::new();
        for ratio in firing {
            if latent.contains_key(&ratio) {
                continue;
            }
            // a latent completed now stands for the first token of its group
            let p = pos + 1 - ratio;
            let (c, s, _) = rope_table(
                &mut ctx.g,
                &mut ctx.params,
                &[p],
                rd,
                ctheta,
                yarn,
                &format!("d.lat{ratio}"),
            );
            latent.insert(ratio, (c, s));
        }
        StepRope { win, comp, latent }
    }

    pub(crate) fn for_layer(&self, ratio: usize) -> RopeTables {
        let (cos, sin, sin_inv) = if ratio > 0 { self.comp } else { self.win };
        let (cos_c, sin_c) = self
            .latent
            .get(&ratio)
            .copied()
            .unwrap_or((self.comp.0, self.comp.1));
        RopeTables {
            cos,
            sin,
            sin_inv,
            cos_c,
            sin_c,
        }
    }
}

/// Extend this source layer's compressed caches by (at most) one latent, and
/// publish them for the layers that read them.
fn extend_compressed(
    ctx: &mut Ctx<'_>,
    lp: &str,
    il: usize,
    x: NodeId,
    step: &CompressStep,
    rope: &RopeTables,
    shared: &mut StepShared,
    out: &mut StepOutputs,
) -> Result<()> {
    let f = DType::F32;
    let (hd, ihd, rd, eps) = (
        ctx.spec.head_dim,
        ctx.spec.index_head_dim,
        ctx.rd(),
        ctx.eps(),
    );
    let (latent, carried) = build_step_compressor(ctx, lp, x, step)?;
    for (node, name) in carried {
        out.push(node, name);
    }

    // the caches as they stood before this step
    let cached_comp = (step.len_before > 0).then(|| {
        ctx.g.input(
            names::compress_kv(il),
            Shape::new(&[step.len_before, hd], f),
        )
    });
    let cached_ik = (step.len_before > 0 && ihd > 0).then(|| {
        ctx.g
            .input(names::index_k(il), Shape::new(&[step.len_before, ihd], f))
    });

    match latent {
        Some(lat) => {
            let (cos_l, sin_l) = (rope.cos_c, rope.sin_c);
            if ihd > 0 {
                let wk = ctx.param(&format!("{lp}.attn.indexer.wk.weight"), true)?;
                let gain = ctx.norm(&format!("{lp}.attn.indexer.k_norm.weight"))?;
                let zb = ctx.zero_bias(&format!("{lp}.dzb.ihd"), ihd);
                let k = ctx.g.mm(lat, wk);
                let k = ctx.g.rms_norm(k, gain, zb, eps);
                let k = rope_tail(&mut ctx.g, k, cos_l, sin_l, 1, 1, ihd, rd);
                out.push(k, names::index_k_new(il));
                shared.csa.index_k = Some(match cached_ik {
                    Some(c) => ctx.g.concat_(vec![c, k], 0),
                    None => k,
                });
            }
            let comp = rope_tail(&mut ctx.g, lat, cos_l, sin_l, 1, 1, hd, rd);
            out.push(comp, names::compress_kv_new(il));
            shared.csa.compress_kv = Some(match cached_comp {
                Some(c) => ctx.g.concat_(vec![c, comp], 0),
                None => comp,
            });
        }
        None => {
            shared.csa.compress_kv = cached_comp;
            shared.csa.index_k = cached_ik;
        }
    }
    shared.len_after = step.len_after;
    Ok(())
}

/// One decode step's attention. Unlike prefill there is no causal masking to do:
/// every window slot the host supplied and every cached latent is visible to
/// this query by construction, so only the Indexer's budget can drop anything.
pub(crate) fn build_decode_attention(
    ctx: &mut Ctx<'_>,
    il: usize,
    x: NodeId,
    plan: &V41DecodePlan,
    rope: &RopeTables,
    shared: &mut StepShared,
    out: &mut StepOutputs,
) -> Result<NodeId> {
    let lp = ctx.spec.layer_prefix(il);
    let hd = ctx.spec.head_dim;
    let cache_len = plan.cache_len;
    let f = DType::F32;

    let Qkv { qr, q, kv } = build_qkv(ctx, &lp, x, rope.cos, rope.sin, 1)?;
    out.push(kv, names::window_kv_new(il));

    // the window this query attends over: everything cached, plus itself
    let window = if cache_len > 0 {
        let cached = ctx
            .g
            .input(names::window_kv(il), Shape::new(&[cache_len, hd], f));
        ctx.g.concat_(vec![cached, kv], 0)
    } else {
        kv
    };
    let n_window = cache_len + 1;

    if let Some(step) = plan.for_source(il) {
        extend_compressed(ctx, &lp, il, x, step, rope, shared, out)?;
    }

    let ncomp = if ctx.spec.ratio(il) > 0 {
        plan.for_source(ctx.spec.kv_source_for(il).unwrap_or(il))
            .map(|s| s.len_after)
            .unwrap_or(0)
    } else {
        0
    };

    let (kv_all, mask, n_keys) = if ncomp == 0 {
        let m = open_mask(ctx, &format!("{lp}.d.maskw"), 1, n_window);
        (window, m, n_window)
    } else {
        if shared.len_after != ncomp {
            return Err(anyhow!(
                "deepseek_v41 decode: layer {il} expects {ncomp} compressed positions, \
                 source has {}",
                shared.len_after
            ));
        }
        let comp = shared.csa.compress_kv.ok_or_else(|| {
            anyhow!("deepseek_v41 decode: layer {il} reads compressed KV with no source")
        })?;
        // one query, and it has passed every latent in the cache
        let comp_geom = Compressed::new(ctx, vec![ncomp], ncomp, &format!("{lp}.d.maskc"));
        let causal_c = comp_geom.causal;
        if ctx.spec.is_index_source(il) && ctx.spec.index_head_dim > 0 {
            publish_topk_mask(ctx, il, x, qr, &comp_geom, rope, &mut shared.csa, ".d")?;
        }
        let comp_mask = shared.csa.topk_mask.unwrap_or(causal_c);
        let win_mask = open_mask(ctx, &format!("{lp}.d.maskw"), 1, n_window);
        let kv_all = ctx.g.concat_(vec![window, comp], 0);
        let full = ctx.g.concat_(vec![win_mask, comp_mask], 1);
        (kv_all, full, n_window + ncomp)
    };

    build_attn_out(
        ctx,
        &lp,
        &AttnOut {
            q,
            kv_all,
            mask,
            n_keys,
            rows: 1,
            cos: rope.cos,
            sin_inv: rope.sin_inv,
        },
    )
}

/// One decode step of the KV Compressor.
///
/// `ratio == 1` completes a group every step, so it is a plain projection with no
/// carried state. Above that, the step either completes the group — pooling the
/// carried `ratio - 1` entries together with this token's — or contributes to it,
/// in which case there is no latent and the raw `wkv`/`wgate` pair is handed back.
fn build_step_compressor(
    ctx: &mut Ctx<'_>,
    lp: &str,
    x: NodeId,
    step: &CompressStep,
) -> Result<(Option<NodeId>, Vec<(NodeId, String)>)> {
    let f = DType::F32;
    let hd = ctx.spec.head_dim;
    let w = CompressorW::load(ctx, lp, ".d")?;
    let kv = w.latent(ctx, x); // [1, hd]

    if step.ratio == 1 {
        return Ok((Some(w.norm(ctx, kv)), Vec::new()));
    }

    let wgate = ctx.param(&format!("{lp}.attn.compressor.wgate.weight"), true)?;
    let score = ctx.g.mm(x, wgate); // [1, hd]
    if !step.fires {
        // still filling: hand both halves back for the host to accumulate
        return Ok((
            None,
            vec![
                (kv, names::group_kv_new(step.source_layer)),
                (score, names::group_score_new(step.source_layer)),
            ],
        ));
    }

    let filled = step.group_filled;
    debug_assert_eq!(filled, step.ratio - 1, "a firing step completes the group");
    let (kv_group, sc_group) = if filled > 0 {
        let gk = ctx.g.input(
            names::group_kv(step.source_layer),
            Shape::new(&[filled, hd], f),
        );
        let gs = ctx.g.input(
            names::group_score(step.source_layer),
            Shape::new(&[filled, hd], f),
        );
        (
            ctx.g.concat_(vec![gk, kv], 0),
            ctx.g.concat_(vec![gs, score], 0),
        )
    } else {
        (kv, score)
    };
    let pooled = pool_groups(ctx, kv_group, sc_group, 1, step.ratio, hd);
    Ok((Some(w.norm(ctx, pooled)), Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> DeepseekV41Spec {
        DeepseekV41Spec::from_config(&serde_json::json!({
            "vocab_size": 64, "dim": 32, "num_hidden_layers": 6, "head_dim": 32,
            "num_attention_heads": 2, "o_lora_rank": 8, "n_routed_experts": 4,
            "moe_intermediate_size": 16, "o_groups": 2, "q_lora_rank": 16,
            "rope_head_dim": 16, "sliding_window": 4,
            "compress_ratios": [0, 0, 2, 2, 1, 1],
            "kv_source_layers": [2, 4], "index_source_layers": [2, 4, 5],
            "index_n_heads": 2, "index_head_dim": 32, "index_topk": 3,
        }))
        .unwrap()
    }

    #[test]
    fn plan_tracks_group_filling_and_firing() {
        let s = spec();
        // ratio-2 source (layer 2) fires on odd positions; ratio-1 (layer 4) always
        for pos in 0..8 {
            let p = V41DecodePlan::new(&s, pos);
            let r2 = p.for_source(2).unwrap();
            let r1 = p.for_source(4).unwrap();
            assert_eq!(r2.ratio, 2);
            assert_eq!(r2.fires, pos % 2 == 1, "pos {pos}");
            assert_eq!(r2.group_filled, pos % 2, "pos {pos}");
            assert_eq!(r2.len_before, pos / 2, "pos {pos}");
            assert_eq!(r2.len_after, pos.div_ceil(2), "pos {pos}");
            assert!(r1.fires, "pos {pos}: a ratio-1 source fires every step");
            assert_eq!(r1.len_after, pos + 1, "pos {pos}");
            assert_eq!(r1.group_filled, 0);
        }
    }

    #[test]
    fn plan_leaves_the_last_window_slot_for_this_token() {
        let s = spec(); // sliding_window 4
        // key count is cache_len + 1 and must equal prefill's min(pos+1, window)
        for pos in 0..10 {
            let got = V41DecodePlan::new(&s, pos).cache_len + 1;
            assert_eq!(got, (pos + 1).min(4), "pos {pos}");
        }
    }

    #[test]
    fn plan_lists_only_kv_sources() {
        let s = spec();
        let p = V41DecodePlan::new(&s, 5);
        let srcs: Vec<usize> = p.sources.iter().map(|x| x.source_layer).collect();
        assert_eq!(srcs, vec![2, 4], "index-only sources own no cache");
    }
}
