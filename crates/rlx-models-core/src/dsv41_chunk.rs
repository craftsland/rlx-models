// RLX — versatile ML compiler + runtime. GPLv3.
//! A **`k`-token step against an existing cache** — the shape prefill and decode
//! are both special cases of.
//!
//! [`crate::dsv41_graph`] builds a whole sequence from nothing;
//! [`crate::dsv41_decode`] extends a cache by exactly one token. Neither covers
//! the middle, and two things need it:
//!
//! * **chunked prefill** — processing a prompt in pieces, so a paged run's
//!   expert working set is bounded by the chunk rather than by the prompt;
//! * **speculative decoding** — verifying a DSpark draft block by running the
//!   backbone over all of its tokens at once instead of one at a time, which is
//!   the entire point of drafting.
//!
//! The awkward part is the compressor. A group of `ratio` positions can start
//! before the chunk and finish inside it, so the host carries the partial group
//! in, the chunk may complete several, and it hands a *different* partial group
//! back. That is why the cache gained replace-semantics names
//! ([`names::group_kv_set`]) alongside the append ones a single step uses.
//!
//! Correctness here rests on three exact identities, which
//! `dsv41_chunk_parity` checks: a chunk at `start = 0` over an empty cache is a
//! prefill, a chunk of `k = 1` is a decode step, and a prompt split into chunks
//! is the same as the whole prompt at once.

use crate::dsv41::DeepseekV41Spec;
use crate::dsv41_block::{AttnOut, Ctx, Qkv, build_attn_out, build_qkv, rope_table};
use crate::dsv41_csa::{Compressed, CompressorW, CsaShared, pool_groups, publish_topk_mask};
use crate::dsv41_decode::{V41ChunkPlan, names};
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

/// The cache tensors a chunk produces, in graph-output order.
#[derive(Default)]
struct ChunkOutputs {
    nodes: Vec<NodeId>,
    names: Vec<String>,
}

impl ChunkOutputs {
    fn push(&mut self, node: NodeId, name: String) {
        self.nodes.push(node);
        self.names.push(name);
    }
}

/// Rotary tables for a chunk: the query positions themselves, plus — per ratio —
/// the positions the latents this chunk produces stand for.
struct ChunkRope {
    win: (NodeId, NodeId, NodeId),
    comp: (NodeId, NodeId, NodeId),
    /// `ratio -> (cos, sin)` over the *visible* latents, `len_after` of them.
    latent: HashMap<usize, (NodeId, NodeId)>,
}

impl ChunkRope {
    fn build(ctx: &mut Ctx<'_>, plan: &V41ChunkPlan) -> Self {
        let spec = ctx.spec;
        let rd = ctx.rd();
        let (theta, ctheta) = (spec.rope_theta, spec.compress_rope_theta);
        let yarn = (spec.original_seq_len > 0 && spec.rope_factor > 1.0).then_some((
            spec.original_seq_len,
            spec.rope_factor,
            spec.beta_fast,
            spec.beta_slow,
        ));
        let q_pos: Vec<usize> = (0..plan.k).map(|i| plan.pos(i)).collect();
        let win = rope_table(
            &mut ctx.g,
            &mut ctx.params,
            &q_pos,
            rd,
            theta,
            None,
            "v41c.rope.win",
        );
        let comp = rope_table(
            &mut ctx.g,
            &mut ctx.params,
            &q_pos,
            rd,
            ctheta,
            yarn,
            "v41c.rope.comp",
        );
        let mut latent = HashMap::new();
        for s in &plan.sources {
            if latent.contains_key(&s.ratio) {
                continue;
            }
            // a latent stands for the first token of its group
            let pos: Vec<usize> = (0..s.len_after).map(|j| j * s.ratio).collect();
            let (c, si, _) = rope_table(
                &mut ctx.g,
                &mut ctx.params,
                &pos,
                rd,
                ctheta,
                yarn,
                &format!("v41c.rope.lat{}", s.ratio),
            );
            latent.insert(s.ratio, (c, si));
        }
        ChunkRope { win, comp, latent }
    }

    fn for_layer(&self, spec: &DeepseekV41Spec, il: usize) -> crate::dsv41_block::RopeTables {
        let ratio = spec.ratio(il);
        let (cos, sin, sin_inv) = if ratio > 0 { self.comp } else { self.win };
        let (cos_c, sin_c) = self.latent.get(&ratio).copied().unwrap_or((cos, sin));
        crate::dsv41_block::RopeTables {
            cos,
            sin,
            sin_inv,
            cos_c,
            sin_c,
        }
    }
}

/// Additive `[k, cache_len + k]` mask for the sliding window.
///
/// Query row `i` is at absolute position `start + i`; key column `j` covers
/// absolute `start - cache_len + j`. A key is visible when it is not in the
/// future and is within `window_size` of the query — the same rule prefill uses,
/// written against absolute positions so a chunk in the middle of a sequence
/// sees exactly what a full prefill would have let it see.
fn chunk_window_mask(ctx: &mut Ctx<'_>, plan: &V41ChunkPlan, tag: &str) -> NodeId {
    let window = ctx.spec.window_size.max(1);
    let (k, cl) = (plan.k, plan.cache_len);
    let n_keys = cl + k;
    let mut m = vec![0f32; k * n_keys];
    for i in 0..k {
        let p = plan.pos(i);
        for j in 0..n_keys {
            let q = plan.start + j - cl;
            if q > p || p - q >= window {
                m[i * n_keys + j] = crate::dsv41_block::NEG;
            }
        }
    }
    ctx.konst(tag, m, &[k, n_keys])
}

/// The compressor over a chunk: carry a partial group in, emit whatever latents
/// complete, carry the new partial group out.
fn chunk_compressor(
    ctx: &mut Ctx<'_>,
    lp: &str,
    il: usize,
    x: NodeId,
    s: &crate::dsv41_decode::ChunkCompress,
    out: &mut ChunkOutputs,
) -> Result<Option<NodeId>> {
    let hd = ctx.spec.head_dim;
    let f = DType::F32;
    let w = CompressorW::load(ctx, lp, ".c")?;
    let kv = w.latent(ctx, x); // [k, hd]

    if s.ratio == 1 {
        // every position is its own group, so nothing is ever carried
        return Ok(Some(w.norm(ctx, kv)));
    }
    let wgate = ctx.param(&format!("{lp}.attn.compressor.wgate.weight"), true)?;
    let score = ctx.g.mm(x, wgate);

    // prepend the carried-in rows so groups that span the boundary are whole
    let (kv_all, score_all) = if s.group_filled > 0 {
        let gk = ctx
            .g
            .input(names::group_kv(il), Shape::new(&[s.group_filled, hd], f));
        let gs = ctx
            .g
            .input(names::group_score(il), Shape::new(&[s.group_filled, hd], f));
        (
            ctx.g.concat_(vec![gk, kv], 0),
            ctx.g.concat_(vec![gs, score], 0),
        )
    } else {
        (kv, score)
    };

    let avail = s.group_filled + ctx.rows;
    let produced = s.produced();
    let used = produced * s.ratio;
    debug_assert_eq!(avail - used, s.group_left, "group accounting");

    if s.group_left > 0 {
        out.push(
            ctx.g.narrow_(kv_all, 0, used, s.group_left),
            names::group_kv_set(il),
        );
        out.push(
            ctx.g.narrow_(score_all, 0, used, s.group_left),
            names::group_score_set(il),
        );
    }
    if produced == 0 {
        return Ok(None);
    }
    let kv_used = ctx.g.narrow_(kv_all, 0, 0, used);
    let sc_used = ctx.g.narrow_(score_all, 0, 0, used);
    let pooled = pool_groups(ctx, kv_used, sc_used, produced, s.ratio, hd);
    Ok(Some(w.norm(ctx, pooled)))
}

/// Publish this chunk's compressed KV and index keys, cached prefix included.
fn chunk_publish(
    ctx: &mut Ctx<'_>,
    lp: &str,
    il: usize,
    x: NodeId,
    s: &crate::dsv41_decode::ChunkCompress,
    rope: &crate::dsv41_block::RopeTables,
    shared: &mut CsaShared,
    out: &mut ChunkOutputs,
) -> Result<()> {
    let (hd, ihd, rd, eps) = (
        ctx.spec.head_dim,
        ctx.spec.index_head_dim,
        ctx.rd(),
        ctx.eps(),
    );
    let f = DType::F32;
    let produced = s.produced();
    let latent = chunk_compressor(ctx, lp, il, x, s, out)?;

    let cached_comp = (s.len_before > 0).then(|| {
        ctx.g
            .input(names::compress_kv(il), Shape::new(&[s.len_before, hd], f))
    });
    let cached_ik = (s.len_before > 0 && ihd > 0).then(|| {
        ctx.g
            .input(names::index_k(il), Shape::new(&[s.len_before, ihd], f))
    });

    let Some(lat) = latent else {
        shared.compress_kv = cached_comp;
        shared.index_k = cached_ik;
        return Ok(());
    };
    // the new latents' rotary positions are the tail of the visible table
    let (cos_l, sin_l) = (rope.cos_c, rope.sin_c);
    let take_tail = |ctx: &mut Ctx<'_>, t: NodeId| {
        if produced == s.len_after {
            t
        } else {
            ctx.g.narrow_(t, 0, s.len_before, produced)
        }
    };
    let cos_new = take_tail(ctx, cos_l);
    let sin_new = take_tail(ctx, sin_l);

    if ihd > 0 {
        let wk = ctx.param(&format!("{lp}.attn.indexer.wk.weight"), true)?;
        let gain = ctx.norm(&format!("{lp}.attn.indexer.k_norm.weight"))?;
        let zb = ctx.zero_bias(&format!("{lp}.czb.ihd"), ihd);
        let kk = ctx.g.mm(lat, wk);
        let kk = ctx.g.rms_norm(kk, gain, zb, eps);
        let kk = rope_tail(&mut ctx.g, kk, cos_new, sin_new, produced, 1, ihd, rd);
        out.push(kk, names::index_k_new(il));
        shared.index_k = Some(match cached_ik {
            Some(c) => ctx.g.concat_(vec![c, kk], 0),
            None => kk,
        });
    }
    let comp = rope_tail(&mut ctx.g, lat, cos_new, sin_new, produced, 1, hd, rd);
    out.push(comp, names::compress_kv_new(il));
    shared.compress_kv = Some(match cached_comp {
        Some(c) => ctx.g.concat_(vec![c, comp], 0),
        None => comp,
    });
    Ok(())
}

/// One chunk's attention for layer `il`.
fn chunk_attention(
    ctx: &mut Ctx<'_>,
    il: usize,
    x: NodeId,
    plan: &V41ChunkPlan,
    rope: &crate::dsv41_block::RopeTables,
    win_mask: NodeId,
    shared: &mut CsaShared,
    out: &mut ChunkOutputs,
) -> Result<NodeId> {
    let lp = ctx.spec.layer_prefix(il);
    let hd = ctx.spec.head_dim;
    let f = DType::F32;
    let (k, cl) = (plan.k, plan.cache_len);

    let Qkv { qr, q, kv } = build_qkv(ctx, &lp, x, rope.cos, rope.sin, k)?;
    out.push(kv, names::window_kv_new(il));
    let window = if cl > 0 {
        let cached = ctx.g.input(names::window_kv(il), Shape::new(&[cl, hd], f));
        ctx.g.concat_(vec![cached, kv], 0)
    } else {
        kv
    };
    let n_window = cl + k;

    if let Some(s) = plan.for_source(il) {
        chunk_publish(ctx, &lp, il, x, s, rope, shared, out)?;
    }

    let ncomp = ctx
        .spec
        .kv_source_for(il)
        .filter(|_| ctx.spec.ratio(il) > 0)
        .and_then(|src| plan.for_source(src))
        .map(|s| s.len_after)
        .unwrap_or(0);

    let (kv_all, mask, n_keys) = if ncomp == 0 {
        (window, win_mask, n_window)
    } else {
        let comp = shared.compress_kv.ok_or_else(|| {
            anyhow!("deepseek_v41 chunk: layer {il} reads compressed KV with no source")
        })?;
        let ratio = ctx.spec.ratio(il);
        // latent c is visible to query i once it has passed the group's last
        // token — the same rule prefill uses, against absolute positions
        let lens: Vec<usize> = (0..k).map(|i| (plan.pos(i) + 1) / ratio).collect();
        let comp_geom = Compressed::new(ctx, lens, ncomp, &format!("{lp}.c.maskc"));
        let causal_c = comp_geom.causal;
        if ctx.spec.is_index_source(il) && ctx.spec.index_head_dim > 0 {
            publish_topk_mask(ctx, il, x, qr, &comp_geom, rope, shared, ".c")?;
        }
        let comp_mask = shared.topk_mask.unwrap_or(causal_c);
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
            rows: k,
            cos: rope.cos,
            sin_inv: rope.sin_inv,
        },
    )
}

/// Build a `k`-token step at absolute position `plan.start`.
///
/// Returns the graph, its parameters, and the output-name list in graph order
/// (`logits` first). Feed it `input_ids [1, k]` plus
/// [`crate::dsv41_decode::V41DecodeCache::chunk_inputs`], and fold the result
/// back with `apply_chunk`.
pub fn build_deepseek_v41_chunk(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    plan: &V41ChunkPlan,
    inputs: &V41Inputs,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    spec.validate()?;
    if plan.k == 0 {
        return Err(anyhow!("deepseek_v41 chunk: k must be at least 1"));
    }
    let (d, hc, k) = (spec.dim, spec.hc_mult, plan.k);
    let mut ctx = Ctx::new("deepseek_v41_chunk", spec, weights, packed, k);
    let rope = ChunkRope::build(&mut ctx, plan);
    let win_mask = chunk_window_mask(&mut ctx, plan, "v41c.mask.win");
    let mut out = ChunkOutputs::default();

    let input_ids = ctx.g.input("input_ids", Shape::new(&[1, k], DType::I32));
    let (embed_w, _, _) =
        load_dense_dequant(&mut ctx.g, &mut ctx.params, ctx.weights, "embed.weight")?;
    let h0 = ctx.g.gather_(embed_w, input_ids, 0);
    let h0 = ctx.g.reshape_(h0, vec![k as i64, 1, d as i64]);
    let ones = ctx.konst("v41c.hc.ones", vec![1f32; hc], &[1, hc, 1]);
    let mut h = ctx.g.mul(h0, ones);
    let mut pre_mix = identity_pre_mix(&mut ctx.g, &mut ctx.params, k, hc);

    let mut shared = CsaShared::default();
    let mut main_hiddens: Vec<NodeId> = Vec::new();
    for il in 0..spec.n_layers {
        let lp = spec.layer_prefix(il);
        if let Some(e) = &spec.engram
            && let Some(hash_idx) = e.layer_hash_index(il)
        {
            let cols = e.n_hash_cols();
            let n_eng = e.layer_ids.len();
            let want = k * n_eng * cols;
            if inputs.engram_rows.len() != want {
                return Err(anyhow!(
                    "deepseek_v41 chunk: engram_rows has {} entries, expected {want} \
                     ({k} tokens × {n_eng} layers × {cols} cols)",
                    inputs.engram_rows.len()
                ));
            }
            let slice: Vec<f32> = (0..k)
                .flat_map(|i| {
                    let base = (i * n_eng + hash_idx) * cols;
                    inputs.engram_rows[base..base + cols]
                        .iter()
                        .map(|&v| v as f32)
                })
                .collect();
            let rows = ctx.konst(&format!("{lp}.engram.rows"), slice, &[k, cols]);
            h = crate::dsv41_engram::build_v41_engram(&mut ctx, &lp, h, rows, None, e)?;
        }
        if inputs.emit_main_hidden && spec.dspark_target_layer_ids.contains(&il) {
            main_hiddens.push(hc_mean(&mut ctx.g, h, k, d));
        }

        let rope_l = rope.for_layer(spec, il);
        let (nh, pre_attn) =
            hc_sublayer_plain(&mut ctx, HcSide::Attn, &lp, h, pre_mix, |ctx, xa| {
                chunk_attention(ctx, il, xa, plan, &rope_l, win_mask, &mut shared, &mut out)
            })?;
        h = nh;
        let (nh, pre_ffn) =
            hc_sublayer_plain(&mut ctx, HcSide::Ffn, &lp, h, pre_attn, |ctx, xf| {
                build_v41_moe(ctx, il, xf, k, None)
            })?;
        h = nh;
        pre_mix = pre_ffn;
    }

    let x = hc_reduce(&mut ctx.g, h, pre_mix, k, hc);
    let gain = ctx.norm("norm.weight")?;
    let zb = ctx.zero_bias("v41c.zb.d", d);
    let eps = ctx.eps();
    let x = ctx.g.rms_norm(x, gain, zb, eps);
    let logits = ctx.project("head.weight", x, k, spec.vocab_size)?;
    let logits = ctx
        .g
        .reshape_(logits, vec![k as i64, spec.vocab_size as i64]);

    let mut all = vec![logits];
    let mut names_out = vec!["logits".to_string()];
    if inputs.emit_main_hidden {
        all.push(main_hidden_node(&mut ctx.g, &main_hiddens, spec, k)?);
        names_out.push("main_hidden".to_string());
    }
    all.extend(out.nodes);
    names_out.extend(out.names);
    let (g, params) = ctx.finish(all);
    Ok((g, params, names_out))
}
