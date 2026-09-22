// RLX — versatile ML compiler + runtime. GPLv3.
//! **DeepSeek-V4.1-Flash** prefill / pipeline-stage graph builder.
//!
//! One block is
//!
//! ```text
//! [engram?] → hc_sublayer(Attn, pre_mix_in) → hc_sublayer(Ffn, attn_pre)
//! ```
//!
//! with the Hyper-Connection wrapper, the q/kv prologue and the o-LoRA epilogue
//! all living in the crate-internal `dsv41_block`, shared with decode and the
//! DSpark draft head. What is *here* is the part only a prefill does: assembling a
//! whole sequence's keys.
//!
//! Attention attends over two key sets concatenated into one softmax: a sliding
//! window of raw KV, plus (on `compress_ratio > 0` layers) the `index_topk` best
//! compressed positions. Which layer *produces* those compressed positions and
//! which merely reads them is the CSA2 split — see [`crate::dsv41`].
//!
//! Precision-simulation is deliberately omitted: the reference round-trips
//! activations through FP8/FP4 in place (`act_quant(..., inplace=True)`). Those
//! calls change no semantics, only precision, so this builder computes the
//! F32-exact value. Weight-side quant is handled at load time by
//! [`crate::dsv41_quant`].
//!
//! Reference: `deepseek-ai/DeepSeek-V4.1-Flash` `inference/model.py`.

use crate::dsv41::DeepseekV41Spec;
use crate::dsv41_block::{
    Ctx, Qkv, RopeTables, Tap, build_o_lora, build_qkv, build_sink_attn, inverse_rope, rope_table,
    sliding_window_mask,
};
use crate::dsv41_csa::{Compressed, CompressorW, CsaShared, pool_groups, publish_topk_mask};
use crate::dsv41_decode::names;
use crate::dsv41_engram::build_v41_engram;
use crate::dsv41_hc::{
    BodyOut, HcSide, SublayerOut, hc_mean, hc_reduce, hc_sublayer, identity_pre_mix,
    main_hidden_node,
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

/// What attention layers hand down the stack instead of recomputing, mirroring
/// the reference `SharedAttentionRuntime`. Every field is written by its source
/// layer before any consumer reads it, so one slot each is enough.
#[derive(Default, Clone, Copy)]
pub(crate) struct SharedAttn {
    pub csa: CsaShared,
    /// Compressed-position count the above were built for.
    pub ncomp: usize,
    /// Pre-RoPE compressor latent, kept only for the `RLX_DSV41_DBG=comp` tap.
    pub dbg_latent: Option<NodeId>,
}

/// **KV Compressor** — pool `ratio` consecutive tokens into one latent with a
/// learned softmax gate, then RMSNorm. `ratio == 1` is a plain projection with no
/// gate at all (and the checkpoint carries no `wgate` for such a layer).
///
/// `x` is the post-`attn_norm` hidden `[seq, dim]`; only the first
/// `seq - seq % ratio` tokens participate (the trailing partial group is what the
/// reference holds back in `kv_state`). Returns the **pre-RoPE** latent
/// `[seq/ratio, head_dim]` — the Indexer needs it unrotated.
fn build_v41_compressor(
    ctx: &mut Ctx<'_>,
    lp: &str,
    x: NodeId,
    seq: usize,
    ratio: usize,
) -> Result<(NodeId, Option<(NodeId, NodeId)>)> {
    let hd = ctx.spec.head_dim;
    let ncomp = seq / ratio;
    let w = CompressorW::load(ctx, lp, "")?;
    let kv = w.latent(ctx, x); // [seq, hd]

    let (pooled, tail) = if ratio == 1 {
        (ctx.g.reshape_(kv, vec![ncomp as i64, hd as i64]), None)
    } else {
        let wgate = ctx.param(&format!("{lp}.attn.compressor.wgate.weight"), true)?;
        let score = ctx.g.mm(x, wgate);
        // A trailing partial group produces no latent, so it is dropped here —
        // but a decode step continuing from this prefill has to keep filling it,
        // so it is handed back rather than discarded.
        let used = ncomp * ratio;
        let rest = seq - used;
        let tail = (rest > 0).then(|| {
            (
                ctx.g.narrow_(kv, 0, used, rest),
                ctx.g.narrow_(score, 0, used, rest),
            )
        });
        let kv = ctx.g.narrow_(kv, 0, 0, used);
        let score = ctx.g.narrow_(score, 0, 0, used);
        (pool_groups(ctx, kv, score, ncomp, ratio, hd), tail)
    };
    Ok((w.norm(ctx, pooled), tail))
}

/// Derive this layer's compressed KV and index keys, and publish them for the
/// layers that read them — the producing half of CSA2.
fn publish_compressed(
    ctx: &mut Ctx<'_>,
    lp: &str,
    il: usize,
    x: NodeId,
    seq: usize,
    rope: &RopeTables,
    shared: &mut SharedAttn,
    cache: &mut PrefillCache,
) -> Result<()> {
    let ratio = ctx.spec.ratio(il);
    let ncomp = seq / ratio;
    let (hd, ihd, rd, eps) = (
        ctx.spec.head_dim,
        ctx.spec.index_head_dim,
        ctx.rd(),
        ctx.eps(),
    );

    let (latent, group) = build_v41_compressor(ctx, lp, x, seq, ratio)?;
    if let Some((kv_tail, score_tail)) = group {
        cache.push(kv_tail, names::group_kv_new(il));
        cache.push(score_tail, names::group_score_new(il));
    }
    // The Indexer needs the latent before RoPE, so derive the index keys first.
    if ihd > 0 {
        let wk = ctx.param(&format!("{lp}.attn.indexer.wk.weight"), true)?;
        let gain = ctx.norm(&format!("{lp}.attn.indexer.k_norm.weight"))?;
        let zb = ctx.zero_bias(&format!("{lp}.zb.ihd"), ihd);
        let k = ctx.g.mm(latent, wk);
        let k = ctx.g.rms_norm(k, gain, zb, eps);
        shared.csa.index_k = Some(rope_tail(
            &mut ctx.g, k, rope.cos_c, rope.sin_c, ncomp, 1, ihd, rd,
        ));
    }
    // a latent stands for the first token of its group → position j·ratio
    shared.csa.compress_kv = Some(rope_tail(
        &mut ctx.g, latent, rope.cos_c, rope.sin_c, ncomp, 1, hd, rd,
    ));
    shared.ncomp = ncomp;
    shared.dbg_latent = Some(latent);
    if let Some(c) = shared.csa.compress_kv {
        cache.push(c, names::compress_kv_new(il));
    }
    if let Some(k) = shared.csa.index_k {
        cache.push(k, names::index_k_new(il));
    }
    Ok(())
}

/// A debug tap fired: the graph ends at `node` and emits no cache.
fn finish_tapped(ctx: Ctx<'_>, node: NodeId) -> (Graph, HashMap<String, Vec<f32>>, Vec<String>) {
    let (g, p) = ctx.finish(vec![node]);
    (g, p, Vec::new())
}

/// The decode-cache tensors a prefill emits, so a decode loop can continue from
/// where the prompt left off.
///
/// Without these the prompt has to be replayed one position at a time, since the
/// prefill graph keeps its KV internal. The names are decode's own append names,
/// so [`crate::dsv41_decode::V41DecodeCache::prime`] folds them in with the same
/// code that folds a step.
#[derive(Default)]
pub(crate) struct PrefillCache {
    pub nodes: Vec<NodeId>,
    pub names: Vec<String>,
}

impl PrefillCache {
    pub(crate) fn push(&mut self, node: NodeId, name: String) {
        self.nodes.push(node);
        self.names.push(name);
    }
}

/// One prefill attention block. Anything this layer *produces* for the layers
/// downstream of it is written back into `shared`.
pub(crate) fn build_v41_attention(
    ctx: &mut Ctx<'_>,
    tap: &mut Tap,
    il: usize,
    x: NodeId,
    seq: usize,
    rope: &RopeTables,
    win_mask: NodeId,
    shared: &mut SharedAttn,
    cache: &mut PrefillCache,
) -> Result<BodyOut> {
    let lp = ctx.spec.layer_prefix(il);
    let Qkv { qr, q, kv } = build_qkv(ctx, &lp, x, rope.cos, rope.sin, seq)?;
    // the sliding window a decode step continues from; it trims to its own cap
    cache.push(kv, names::window_kv_new(il));

    let ratio = ctx.spec.ratio(il);
    let ncomp = seq.checked_div(ratio).unwrap_or(0);

    // ── CSA2: produce or reuse the compressed KV and its index keys ──
    if ctx.spec.is_kv_source(il) && ncomp > 0 {
        publish_compressed(ctx, &lp, il, x, seq, rope, shared, cache)?;
    }

    let (kv_all, mask, n_keys) = if ncomp == 0 {
        (kv, win_mask, seq)
    } else {
        let comp = shared.csa.compress_kv.ok_or_else(|| {
            anyhow!("deepseek_v41: layer {il} reads compressed KV but no source produced it")
        })?;
        if shared.ncomp != ncomp {
            return Err(anyhow!(
                "deepseek_v41: layer {il} expects {ncomp} compressed positions, source produced {}",
                shared.ncomp
            ));
        }
        // causal visibility: latent c is visible to query qi once qi has passed
        // its last token, i.e. c < (qi+1)/ratio
        let lens: Vec<usize> = (0..seq).map(|qi| (qi + 1) / ratio).collect();
        let comp_geom = Compressed::new(ctx, lens, ncomp, &format!("{lp}.v41.maskc"));
        let causal_c = comp_geom.causal;
        if ctx.spec.is_index_source(il) && ctx.spec.index_head_dim > 0 {
            publish_topk_mask(ctx, il, x, qr, &comp_geom, rope, &mut shared.csa, "")?;
        }
        if tap.hit("compkv", il) {
            return Ok(BodyOut::Tapped(comp));
        }
        if tap.hit("topk", il) {
            let m = shared
                .csa
                .topk_mask
                .ok_or_else(|| anyhow!("layer {il} has no compressed-position mask"))?;
            return Ok(BodyOut::Tapped(m));
        }
        if tap.hit("comp", il) {
            let l = shared
                .dbg_latent
                .ok_or_else(|| anyhow!("layer {il} produced no compressor latent to tap"))?;
            return Ok(BodyOut::Tapped(l));
        }
        let comp_mask = shared.csa.topk_mask.unwrap_or(causal_c);
        let kv_all = ctx.g.concat_(vec![kv, comp], 0);
        let full_mask = ctx.g.concat_(vec![win_mask, comp_mask], 1);
        (kv_all, full_mask, seq + ncomp)
    };
    // a `comp`/`compkv`/`topk` tap on a layer that compresses nothing
    if ncomp == 0 && (tap.hit("compkv", il) || tap.hit("topk", il) || tap.hit("comp", il)) {
        return Err(anyhow!(
            "deepseek_v41: layer {il} has compress_ratio 0, so there is nothing to tap"
        ));
    }

    let o = build_sink_attn(ctx, &lp, q, kv_all, mask, n_keys, seq)?;
    if tap.hit("sa", il) {
        return Ok(BodyOut::Tapped(o));
    }
    let o = inverse_rope(ctx, o, rope.cos, rope.sin_inv, seq);
    if tap.hit("oinv", il) {
        return Ok(BodyOut::Tapped(o));
    }
    let out = build_o_lora(ctx, &lp, o, seq)?;
    if tap.hit("attn", il) {
        return Ok(BodyOut::Tapped(out));
    }
    Ok(BodyOut::Out(out))
}

/// Everything the host must supply alongside `input_ids` for one prefill.
#[derive(Default, Clone)]
pub struct V41Inputs {
    /// Engram row indices, `[seq · n_engram_layers · n_hash_cols]` in that order
    /// — the output of [`crate::dsv41_engram::EngramHashPlan::hash_ids`]. Empty
    /// when the checkpoint has no Engram.
    pub engram_rows: Vec<i64>,
    /// `true` at positions inside an image span. Those positions take no part in
    /// an n-gram and route through the VL gate bias. Empty means text-only.
    pub image_positions: Vec<bool>,
    /// Also emit `main_hidden [rows, dim · n_targets]` — the mean over the
    /// Hyper-Connection copies of the stream entering each
    /// `dspark_target_layer_ids` layer, which is what the DSpark draft head
    /// conditions on ([`crate::dsv41_dspark`]). Off by default so the graph has a
    /// single output.
    pub emit_main_hidden: bool,
    /// Also emit the decode cache this prompt leaves behind, so a decode loop
    /// can continue from it instead of replaying the prompt one token at a time.
    /// The names come back alongside the graph; feed them to
    /// [`crate::dsv41_decode::V41DecodeCache::prime`].
    pub emit_cache: bool,
    /// Take the stream from an `inputs_embeds [seq, dim]` input instead of
    /// looking `input_ids` up in the embedding table.
    ///
    /// This is how a multimodal prompt is fed: the caller runs the vision tower
    /// ([`crate::dsv41_vision::build_v41_vision`]), writes its rows into the
    /// prompt's image positions, and hands the whole thing in. Patching after a
    /// lookup would not work — the lookup has no row for an image.
    pub embeds: bool,
}

impl V41Inputs {
    /// This layer's slice of the Engram row ids, as a `[seq, cols]` constant.
    fn engram_rows_for(
        &self,
        ctx: &mut Ctx<'_>,
        lp: &str,
        seq: usize,
        n_layers: usize,
        hash_idx: usize,
        cols: usize,
    ) -> Result<NodeId> {
        let want = seq * n_layers * cols;
        if self.engram_rows.len() != want {
            return Err(anyhow!(
                "deepseek_v41: engram_rows has {} entries, expected {want} \
                 (seq {seq} × {n_layers} layers × {cols} cols)",
                self.engram_rows.len()
            ));
        }
        let slice: Vec<f32> = (0..seq)
            .flat_map(|i| {
                let base = (i * n_layers + hash_idx) * cols;
                self.engram_rows[base..base + cols]
                    .iter()
                    .map(|&v| v as f32)
            })
            .collect();
        Ok(ctx.konst(&format!("{lp}.engram.rows"), slice, &[seq, cols]))
    }

    /// `[seq, 1]` selectors for image spans: `1.0` inside one for the VL routing
    /// bias, and its complement for the Engram (image tokens take no part in an
    /// n-gram). `None` when the prompt has no image.
    fn image_masks(&self, ctx: &mut Ctx<'_>, seq: usize) -> Option<(NodeId, NodeId)> {
        if !self.image_positions.iter().any(|&b| b) {
            return None;
        }
        let inside: Vec<f32> = (0..seq)
            .map(|i| f32::from(self.image_positions.get(i).copied().unwrap_or(false)))
            .collect();
        let outside: Vec<f32> = inside.iter().map(|v| 1.0 - v).collect();
        let a = ctx.konst("v41.mask.image", inside, &[seq, 1]);
        let b = ctx.konst("v41.mask.engram_alive", outside, &[seq, 1]);
        Some((a, b))
    }
}

/// Build a **DeepSeek-V4.1** prefill graph for `seq` tokens. Returns the graph
/// and its parameter map; the single graph input is `input_ids [1, seq]`, the
/// output `logits [seq, vocab_size]`.
pub fn build_deepseek_v41_prefill(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    inputs: &V41Inputs,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    build_deepseek_v41_stage(spec, weights, seq, &StageSpan::whole(spec), inputs, packed)
}

/// The per-position RoPE tables a stage needs: one per base, plus one per
/// distinct compression ratio sampled at `j · ratio`.
pub(crate) struct StageRope {
    win: (NodeId, NodeId, NodeId),
    comp: (NodeId, NodeId, NodeId),
    per_ratio: HashMap<usize, (NodeId, NodeId)>,
}

impl StageRope {
    pub(crate) fn build(ctx: &mut Ctx<'_>, seq: usize, layers: &std::ops::Range<usize>) -> Self {
        let spec = ctx.spec;
        let rd = ctx.rd();
        let (theta, ctheta) = (spec.rope_theta, spec.compress_rope_theta);
        // YaRN scales the compressed table only; pure sliding-window layers use
        // the raw base.
        let yarn = (spec.original_seq_len > 0 && spec.rope_factor > 1.0).then_some((
            spec.original_seq_len,
            spec.rope_factor,
            spec.beta_fast,
            spec.beta_slow,
        ));
        let positions: Vec<usize> = (0..seq).collect();
        let win = rope_table(
            &mut ctx.g,
            &mut ctx.params,
            &positions,
            rd,
            theta,
            None,
            "win",
        );
        let comp = rope_table(
            &mut ctx.g,
            &mut ctx.params,
            &positions,
            rd,
            ctheta,
            yarn,
            "comp",
        );

        let ratios: Vec<usize> = layers.clone().map(|il| spec.ratio(il)).collect();
        let mut per_ratio = HashMap::new();
        for ratio in ratios {
            if ratio == 0 || per_ratio.contains_key(&ratio) || seq / ratio == 0 {
                continue;
            }
            let pos: Vec<usize> = (0..seq / ratio).map(|j| j * ratio).collect();
            let (c, s, _) = rope_table(
                &mut ctx.g,
                &mut ctx.params,
                &pos,
                rd,
                ctheta,
                yarn,
                &format!("c{ratio}"),
            );
            per_ratio.insert(ratio, (c, s));
        }
        StageRope {
            win,
            comp,
            per_ratio,
        }
    }

    pub(crate) fn for_layer(&self, ratio: usize) -> RopeTables {
        let (cos, sin, sin_inv) = if ratio > 0 { self.comp } else { self.win };
        let (cos_c, sin_c) = self
            .per_ratio
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

/// Which slice of the model a graph covers.
///
/// `first`/`last` decide whether the stage owns the embedding and the LM head;
/// as bare booleans at a call site they read as `..., true, false, ...`, which
/// is why they are named here.
#[derive(Clone, Debug)]
pub struct StageSpan {
    pub layers: std::ops::Range<usize>,
    /// Owns the embedding, so the graph takes `input_ids`.
    pub first: bool,
    /// Owns the final norm and LM head, so the graph emits `logits`.
    pub last: bool,
}

impl StageSpan {
    /// The whole model in one graph.
    pub fn whole(spec: &DeepseekV41Spec) -> Self {
        StageSpan {
            layers: 0..spec.n_layers,
            first: true,
            last: true,
        }
    }

    /// An interior slice, taking a hidden state in and handing one on.
    pub fn middle(layers: std::ops::Range<usize>) -> Self {
        StageSpan {
            layers,
            first: false,
            last: false,
        }
    }

    /// Every `kv_source_layer` has to sit in the same stage as the layers that
    /// read it, or the consumers find an empty compressed cache. This reports
    /// the first layer that would be orphaned by the split.
    pub fn check_csa2(&self, spec: &DeepseekV41Spec) -> Result<()> {
        for il in self.layers.clone() {
            if spec.ratio(il) == 0 {
                continue;
            }
            match spec.kv_source_for(il) {
                Some(src) if self.layers.contains(&src) => {}
                Some(src) => {
                    return Err(anyhow!(
                        "deepseek_v41: stage {:?} holds layer {il} but its KV source is layer \
                         {src}; a stage must start at or before each source it reads",
                        self.layers
                    ));
                }
                None => {
                    return Err(anyhow!(
                        "deepseek_v41: layer {il} compresses KV but no source precedes it"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// One **pipeline stage**: transformer layers `layers` only, loading only those
/// layers' weights (plus embeddings when `first` and the norm/head when `last`).
///
/// The graph input is `input_ids [1, seq]` when `first`, else the boundary pair
/// `hidden_in [seq, hc_mult, dim]` and `pre_mix_in [seq, hc_mult]` — V4.1 threads
/// the Hyper-Connection pre-mix across blocks, so a stage boundary has to carry
/// it too or the next stage silently restarts from the one-hot mix.
///
/// A split also has to keep every `kv_source_layer` in the same stage as the
/// layers that read it; [`DeepseekV41Spec::kv_source_for`] says where those runs
/// begin.
///
/// The output is `logits [seq, vocab]` when `last`, else `hidden_out` plus
/// `pre_mix_out`.
pub fn build_deepseek_v41_stage(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    span: &StageSpan,
    inputs: &V41Inputs,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    spec.validate()?;
    span.check_csa2(spec)?;
    let StageSpan {
        layers,
        first,
        last,
    } = span.clone();
    let mut ctx = Ctx::new("deepseek_v41_stage", spec, weights, packed, seq);
    let mut tap = Tap::from_env();
    let (d, hc) = (spec.dim, spec.hc_mult);
    let f = DType::F32;

    let rope = StageRope::build(&mut ctx, seq, &layers);

    let win_mask = sliding_window_mask(&mut ctx, seq, spec.window_size, "v41.mask.win");
    let (image_mask, engram_alive) = match inputs.image_masks(&mut ctx, seq) {
        Some((a, b)) => (Some(a), Some(b)),
        None => (None, None),
    };

    let (mut h, mut pre_mix) = if first {
        // `inputs_embeds` replaces the lookup entirely rather than patching it
        // afterwards: the caller has already spliced the vision tower's output
        // into the image positions, and re-gathering would overwrite them.
        let h0 = if inputs.embeds {
            ctx.g.input("inputs_embeds", Shape::new(&[seq, d], f))
        } else {
            let input_ids = ctx.g.input("input_ids", Shape::new(&[1, seq], DType::I32));
            let (embed_w, _, _) =
                load_dense_dequant(&mut ctx.g, &mut ctx.params, ctx.weights, "embed.weight")?;
            ctx.g.gather_(embed_w, input_ids, 0) // [1, seq, d]
        };
        let h0 = ctx.g.reshape_(h0, vec![seq as i64, 1, d as i64]);
        // expand to hc_mult copies for Hyper-Connections
        let ones = ctx.konst("v41.hc.ones", vec![1f32; hc], &[1, hc, 1]);
        let h = ctx.g.mul(h0, ones);
        let pre = identity_pre_mix(&mut ctx.g, &mut ctx.params, seq, hc);
        (h, pre)
    } else {
        let h = ctx.g.input("hidden_in", Shape::new(&[seq, hc, d], f));
        let p = ctx.g.input("pre_mix_in", Shape::new(&[seq, hc], f));
        (h, p)
    };

    let mut shared = SharedAttn::default();
    let mut cache = PrefillCache::default();
    let mut main_hiddens: Vec<NodeId> = Vec::new();
    for il in layers.clone() {
        let lp = spec.layer_prefix(il);

        // ── Engram, before the block reads the stream ──
        if let Some(e) = &spec.engram
            && let Some(hash_idx) = e.layer_hash_index(il)
        {
            let cols = e.n_hash_cols();
            let rows_node =
                inputs.engram_rows_for(&mut ctx, &lp, seq, e.layer_ids.len(), hash_idx, cols)?;
            h = build_v41_engram(&mut ctx, &lp, h, rows_node, engram_alive, e)?;
            if tap.hit("engram", il) {
                return Ok(finish_tapped(ctx, h));
            }
        }

        if inputs.emit_main_hidden && spec.dspark_target_layer_ids.contains(&il) {
            main_hiddens.push(hc_mean(&mut ctx.g, h, seq, d));
        }

        let rope_l = rope.for_layer(spec.ratio(il));

        // ── attention sub-block ──
        let attn = hc_sublayer(&mut ctx, HcSide::Attn, &lp, h, pre_mix, |ctx, xa| {
            if tap.hit("xa", il) {
                return Ok(BodyOut::Tapped(xa));
            }
            build_v41_attention(
                ctx,
                &mut tap,
                il,
                xa,
                seq,
                &rope_l,
                win_mask,
                &mut shared,
                &mut cache,
            )
        })?;
        let attn_pre = match attn {
            SublayerOut::Tapped(n) => return Ok(finish_tapped(ctx, n)),
            SublayerOut::Done { h: nh, pre_out } => {
                h = nh;
                pre_out
            }
        };

        // ── FFN sub-block ──
        let ffn = hc_sublayer(&mut ctx, HcSide::Ffn, &lp, h, attn_pre, |ctx, xf| {
            let out = build_v41_moe(ctx, il, xf, seq, image_mask)?;
            Ok(if tap.hit("ffn", il) {
                BodyOut::Tapped(out)
            } else {
                BodyOut::Out(out)
            })
        })?;
        match ffn {
            SublayerOut::Tapped(n) => return Ok(finish_tapped(ctx, n)),
            SublayerOut::Done { h: nh, pre_out } => {
                h = nh;
                pre_mix = pre_out;
            }
        }
        if tap.hit("block", il) {
            return Ok(finish_tapped(ctx, h));
        }
    }
    tap.check_fired(&format!("layers {layers:?}"))?;

    let mut outs = if last {
        let x = hc_reduce(&mut ctx.g, h, pre_mix, seq, hc);
        let gain = ctx.norm("norm.weight")?;
        let zb = ctx.zero_bias("v41.zb.d", d);
        let x = ctx.g.rms_norm(x, gain, zb, ctx.spec.rms_norm_eps);
        let logits = ctx.project("head.weight", x, seq, spec.vocab_size)?;
        vec![
            ctx.g
                .reshape_(logits, vec![seq as i64, spec.vocab_size as i64]),
        ]
    } else {
        vec![h, pre_mix]
    };
    if inputs.emit_main_hidden {
        let mh = main_hidden_node(&mut ctx.g, &main_hiddens, spec, seq)?;
        outs.push(mh);
    }
    let mut cache_names = Vec::new();
    if inputs.emit_cache {
        outs.extend(cache.nodes.iter().copied());
        cache_names = cache.names.clone();
    }
    let (g, params) = ctx.finish(outs);
    Ok((g, params, cache_names))
}
