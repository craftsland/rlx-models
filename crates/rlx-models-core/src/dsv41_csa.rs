// RLX — versatile ML compiler + runtime. GPLv3.
//! **CSA2** (Compressed Sparse Attention 2): the compressor, the hierarchical
//! Indexer, and the masks they publish.
//!
//! Only the layers in `kv_source_layer_ids` run a compressor, pooling every
//! `ratio` positions into one latent; every later layer attends over that shared
//! cache instead of compressing again. The Indexer then scores those compressed
//! positions and keeps the top `index_topk` per query, and — above
//! `candidate_source_layer_id` — a coarser block-level filter narrows the field
//! first.
//!
//! Prefill and decode use the same code here. They differ only in how many query
//! rows there are, which [`Compressed`] carries, and in a name suffix that keeps
//! the two graphs' generated parameters apart.
//!
//! Reference: `deepseek-ai/DeepSeek-V4.1-Flash` `inference/model.py`.

use crate::dsv41_block::{Ctx, NEG, RopeTables};
use crate::standard_decoder::{rope_tail, synth_const};
use anyhow::{Result, anyhow};
use rlx_ir::GraphExt;
use rlx_ir::graph::{Graph, NodeId};
use rlx_ir::op::Op;
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// What the Indexer scores from: the query in rank space, the stream it weights
/// the heads by, the cached index keys, and the two projections.
struct IndexerIn {
    qr: NodeId,
    x: NodeId,
    index_k: NodeId,
    wq_b: NodeId,
    weights_proj: NodeId,
}

/// **Indexer scoring** — `Σ_h relu(⟨q[s,h], k[t]⟩) · weights[s,h]`, the learned
/// relevance of compressed position `t` to query `s`.
///
/// `weights = weights_proj(x) · (index_head_dim^-0.5 · index_n_heads^-0.5)`.
/// The reference's `fp4_act_quant` on `q`/`k` is precision-only and omitted.
fn build_v41_index_score(
    ctx: &mut Ctx<'_>,
    inp: &IndexerIn,
    cos: NodeId,
    sin: NodeId,
    rows: usize,
    ncomp: usize,
    tag: &str,
) -> NodeId {
    let (nh, ihd, rd) = (ctx.spec.index_n_heads, ctx.spec.index_head_dim, ctx.rd());
    let (sq, n, d, nc) = (rows as i64, nh as i64, ihd as i64, ncomp as i64);
    let q = ctx.g.mm(inp.qr, inp.wq_b); // [rows, nh·ihd]
    let q = rope_tail(&mut ctx.g, q, cos, sin, rows, nh, ihd, rd);
    let q2 = ctx.g.reshape_(q, vec![(rows * nh) as i64, d]);
    let kt = ctx.g.transpose_(inp.index_k, vec![1, 0]); // [ihd, ncomp]
    let sc = ctx.g.mm(q2, kt); // [rows·nh, ncomp]
    let sc = ctx.g.relu(sc);
    let sc = ctx.g.reshape_(sc, vec![sq, n, nc]);
    let w = ctx.g.mm(inp.x, inp.weights_proj); // [rows, nh]
    let scale = (ihd as f32).powf(-0.5) * (nh as f32).powf(-0.5);
    let sc_c = ctx.konst(&format!("{tag}.idx.scale"), vec![scale], &[1, 1]);
    let w = ctx.g.mul(w, sc_c);
    let w = ctx.g.reshape_(w, vec![sq, n, 1]);
    let prod = ctx.g.mul(sc, w);
    ctx.g.sum(prod, vec![1], false) // [rows, ncomp]
}

/// Level one of the hierarchical Indexer: keep the `topk_blocks` highest-scoring
/// blocks of `block_size` compressed positions per query, as an additive
/// `[seq, ncomp]` mask.
///
/// A block scores by its best position. Two constants do the bookkeeping the
/// reference does with `±inf`: `reach` marks blocks no position of which the
/// query can see yet (they must never be kept, even when fewer than `topk_blocks`
/// blocks are reachable), and `pin` force-selects the block holding the query's
/// newest position — it is only partly filled and would otherwise lose to an
/// older, full block.
/// Which blocks each query can reach, and which one holds its newest latent.
///
/// Both are `[rows, nblocks]` additive constants. `reach` is `NEG` on a block the
/// query cannot see at all; `pin` is `-NEG` on the single block holding the
/// query's most recent latent, which has to survive the block top-k whatever it
/// scores — otherwise a query can lose the position it just wrote.
fn candidate_block_bounds(lens: &[usize], nblocks: usize, block: usize) -> (Vec<f32>, Vec<f32>) {
    let rows = lens.len();
    let mut reach = vec![0f32; rows * nblocks];
    let mut pin = vec![0f32; rows * nblocks];
    for (qi, &len) in lens.iter().enumerate() {
        for b in 0..nblocks {
            if b * block >= len {
                reach[qi * nblocks + b] = NEG;
            }
        }
        // a query that has passed no latent at all pins nothing
        if len > 0 {
            pin[qi * nblocks + (len - 1) / block] = -NEG;
        }
    }
    (reach, pin)
}

/// Max-pool `score [rows, ncomp]` down to one value per block.
///
/// The last block is padded out with `NEG` rather than zero, so a partial block
/// cannot win the top-k on filler positions that do not exist.
fn block_max(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    score: NodeId,
    rows: usize,
    ncomp: usize,
    block: usize,
    tag: &str,
) -> NodeId {
    let nblocks = ncomp.div_ceil(block);
    let padded = nblocks * block;
    let score_p = if padded > ncomp {
        let pad = synth_const(
            g,
            params,
            &format!("{tag}.cand.pad"),
            vec![NEG; rows * (padded - ncomp)],
            &[rows, padded - ncomp],
        );
        g.concat_(vec![score, pad], 1)
    } else {
        score
    };
    let blk = g.reshape_(score_p, vec![rows as i64, nblocks as i64, block as i64]);
    g.add_node(
        Op::Reduce {
            op: rlx_ir::op::ReduceOp::Max,
            axes: vec![2],
            keep_dim: false,
        },
        vec![blk],
        Shape::new(&[rows, nblocks], DType::F32),
    )
}

/// Broadcast each block's verdict back over the `block` positions it covers, and
/// trim the padding `block_max` added.
fn expand_blocks(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    keep: NodeId,
    rows: usize,
    ncomp: usize,
    block: usize,
    tag: &str,
) -> NodeId {
    let nblocks = ncomp.div_ceil(block);
    let keep3 = g.reshape_(keep, vec![rows as i64, nblocks as i64, 1]);
    let ones = synth_const(
        g,
        params,
        &format!("{tag}.cand.ones"),
        vec![1f32; block],
        &[1, 1, block],
    );
    let wide = g.mul(keep3, ones);
    let wide = g.reshape_(wide, vec![rows as i64, (nblocks * block) as i64]);
    g.narrow_(wide, 1, 0, ncomp)
}

/// The hierarchical Indexer's coarse pass: keep the `topk_blocks` best *blocks*
/// of compressed positions and mask the rest, so the fine top-k that follows
/// chooses from a narrowed field.
fn build_v41_candidate_mask(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    score: NodeId, // [rows, ncomp], already causally masked
    comp: &Compressed,
    block: usize,
    topk_blocks: usize,
    tag: &str,
) -> NodeId {
    let (rows, ncomp) = (comp.rows(), comp.ncomp);
    let nblocks = ncomp.div_ceil(block);
    let blk_score = block_max(g, params, score, rows, ncomp, block, tag);

    let (reach, pin) = candidate_block_bounds(&comp.lens, nblocks, block);
    let reach_c = synth_const(
        g,
        params,
        &format!("{tag}.cand.reach"),
        reach,
        &[rows, nblocks],
    );
    let pin_c = synth_const(g, params, &format!("{tag}.cand.pin"), pin, &[rows, nblocks]);
    let blk_score = g.add(blk_score, reach_c);
    let blk_score = g.add(blk_score, pin_c);

    // Keep exactly `topk_blocks` blocks (the pinned one always among them), then
    // drop any that were only picked because fewer blocks were reachable.
    let keep = exact_topk_mask(
        g,
        params,
        blk_score,
        reach_c,
        rows,
        nblocks,
        topk_blocks,
        &format!("{tag}.cand"),
    );

    let wide = expand_blocks(g, params, keep, rows, ncomp, block, tag);
    g.add(wide, comp.causal)
}

/// Build this layer's Indexer and publish the compressed-position mask the
/// later layers attend through.
///
/// Prefill and decode differ only in how many query rows there are (which
/// `comp` already carries) and in `suffix`, which keeps the decode graph's
/// parameter names from colliding with prefill's. Everything else — the score,
/// the candidate filter, the top-k re-mask — is identical, so they share this.
pub(crate) fn publish_topk_mask(
    ctx: &mut Ctx<'_>,
    il: usize,
    x: NodeId,
    qr: NodeId,
    comp: &Compressed,
    rope: &RopeTables,
    shared: &mut CsaShared,
    suffix: &str,
) -> Result<()> {
    let spec = ctx.spec;
    let topk = spec.index_topk;
    let (is_cand_src, uses_cand) = (spec.is_candidate_source(il), spec.uses_candidates(il));
    let (cand_block, cand_topk) = (spec.candidate_block_size, spec.candidate_topk_blocks);
    let lp = spec.layer_prefix(il);
    let tag = format!("{lp}{suffix}");
    let (rows, ncomp, causal_c) = (comp.rows(), comp.ncomp, comp.causal);

    let index_k = shared.index_k.ok_or_else(|| {
        anyhow!("deepseek_v41: layer {il} indexes but no kv source produced index keys")
    })?;
    let wq_b = ctx.param(&format!("{lp}.attn.indexer.wq_b.weight"), true)?;
    let weights_proj = ctx.param(&format!("{lp}.attn.indexer.weights_proj.weight"), true)?;
    let inp = IndexerIn {
        qr,
        x,
        index_k,
        wq_b,
        weights_proj,
    };
    let mut score = build_v41_index_score(ctx, &inp, rope.cos, rope.sin, rows, ncomp, &tag);
    score = ctx.g.add(score, causal_c);

    if is_cand_src && cand_block > 0 {
        shared.candidates = Some(build_v41_candidate_mask(
            &mut ctx.g,
            &mut ctx.params,
            score,
            comp,
            cand_block,
            cand_topk,
            &tag,
        ));
    } else if uses_cand && let Some(cand) = shared.candidates {
        score = ctx.g.add(score, cand);
    }

    // The re-mask has to carry the candidate filter too, or a position outside
    // every candidate block could come back through a short row's filler picks.
    let base = match shared.candidates {
        Some(c) if uses_cand => c,
        _ => causal_c,
    };
    // Selecting `index_topk` of `ncomp` cannot change anything when every
    // reachable position already fits — which is the whole short-context regime,
    // so skip the TopK there rather than pay for a no-op.
    shared.topk_mask = Some(if ncomp > topk && topk > 0 {
        exact_topk_mask(
            &mut ctx.g,
            &mut ctx.params,
            score,
            base,
            rows,
            ncomp,
            topk,
            &tag,
        )
    } else {
        base
    });
    Ok(())
}

/// Gated mean-pool of the compressor's groups: `[nwin · ratio, hd]` in,
/// `[nwin, hd]` out.
///
/// The gate softmax is over the `ratio` axis **per feature**, which is why that
/// axis has to be moved last and back rather than reduced where it sits. Prefill
/// pools the whole sequence at once and decode pools one completed group, but
/// the arithmetic is the same — so it lives here, once.
pub(crate) fn pool_groups(
    ctx: &mut Ctx<'_>,
    kv: NodeId,
    score: NodeId,
    nwin: usize,
    ratio: usize,
    hd: usize,
) -> NodeId {
    let (nw, r, d) = (nwin as i64, ratio as i64, hd as i64);
    let kv3 = ctx.g.reshape_(kv, vec![nw, r, d]);
    let sc3 = ctx.g.reshape_(score, vec![nw, r, d]);
    let sct = ctx.g.transpose_(sc3, vec![0, 2, 1]); // [nwin, hd, ratio]
    let w = ctx.g.sm(sct, -1);
    let w = ctx.g.transpose_(w, vec![0, 2, 1]); // [nwin, ratio, hd]
    let prod = ctx.g.mul(kv3, w);
    let pooled = ctx.g.sum(prod, vec![1], false); // [nwin, hd]
    ctx.g.reshape_(pooled, vec![nw, d])
}

/// The compressor's weights and its output norm — loaded identically by prefill
/// and decode, with `suffix` keeping the two graphs' generated names apart.
pub(crate) struct CompressorW {
    wkv: NodeId,
    gain: NodeId,
    zb: NodeId,
}

impl CompressorW {
    pub fn load(ctx: &mut Ctx<'_>, lp: &str, suffix: &str) -> Result<Self> {
        let hd = ctx.spec.head_dim;
        Ok(CompressorW {
            wkv: ctx.param(&format!("{lp}.attn.compressor.wkv.weight"), true)?,
            gain: ctx.norm(&format!("{lp}.attn.compressor.norm.weight"))?,
            zb: ctx.zero_bias(&format!("{lp}{suffix}.comp.zb"), hd),
        })
    }

    /// The latent each position contributes to its group, before pooling.
    pub fn latent(&self, ctx: &mut Ctx<'_>, x: NodeId) -> NodeId {
        ctx.g.mm(x, self.wkv)
    }

    pub fn norm(&self, ctx: &mut Ctx<'_>, pooled: NodeId) -> NodeId {
        let eps = ctx.eps();
        ctx.g.rms_norm(pooled, self.gain, self.zb, eps)
    }
}

/// CSA2's layer-to-layer channel.
///
/// A kv-source layer publishes the compressed cache and the Indexer's key half
/// here; the layers after it read them instead of compressing again. The two
/// masks are published separately — `topk_mask` by an index source, `candidates`
/// by the single candidate source — because those are different layer sets.
///
/// Prefill and decode both carry one, which is what lets the Indexer be built
/// once for both.
#[derive(Default, Clone, Copy)]
pub(crate) struct CsaShared {
    /// RoPE'd compressed KV `[ncomp, head_dim]` from the last `kv_source_layer`.
    pub compress_kv: Option<NodeId>,
    /// RoPE'd index keys `[ncomp, index_head_dim]` from the same layer.
    pub index_k: Option<NodeId>,
    /// Additive `[rows, ncomp]` mask from the last `index_source_layer`.
    pub topk_mask: Option<NodeId>,
    /// Additive `[rows, ncomp]` candidate-block mask from `candidate_source_layer`.
    pub candidates: Option<NodeId>,
}

/// The compressed-cache geometry one attention layer works against.
///
/// `lens[i]` is how many latents query `i` has passed — which is what makes the
/// mask causal over a cache that grows once every `ratio` tokens rather than
/// once per token. At decode there is a single query and it has passed all of
/// them, so `lens` is `[ncomp]`.
pub(crate) struct Compressed {
    /// How many compressed positions exist.
    pub ncomp: usize,
    pub lens: Vec<usize>,
    /// `[rows(), ncomp]` additive mask hiding what each query cannot see.
    pub causal: NodeId,
}

impl Compressed {
    pub fn new(ctx: &mut Ctx<'_>, lens: Vec<usize>, ncomp: usize, tag: &str) -> Self {
        let causal = compressed_causal_mask(&mut ctx.g, &mut ctx.params, &lens, ncomp, tag);
        Compressed {
            ncomp,
            lens,
            causal,
        }
    }

    /// Queries — `seq` in prefill, 1 in decode.
    pub fn rows(&self) -> usize {
        self.lens.len()
    }
}

/// Additive `[rows, ncomp]` mask that hides every compressed position a query
/// cannot see yet. `compress_lens[i]` is how many latents query `i` has passed.
fn compressed_causal_mask(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    compress_lens: &[usize],
    ncomp: usize,
    tag: &str,
) -> NodeId {
    let rows = compress_lens.len();
    let mut m = vec![0f32; rows * ncomp];
    for (qi, &len) in compress_lens.iter().enumerate() {
        for c in 0..ncomp {
            if c >= len {
                m[qi * ncomp + c] = NEG;
            }
        }
    }
    synth_const(g, params, tag, m, &[rows, ncomp])
}

fn exact_topk_mask(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    score: NodeId,      // [rows, width], already causally masked
    causal_add: NodeId, // [rows, width]
    rows: usize,
    width: usize,
    k: usize,
    tag: &str,
) -> NodeId {
    let f = DType::F32;
    let k = k.min(width);
    let idx = g.add_node(Op::TopK { k }, vec![score], Shape::new(&[rows, k], f));
    let base = synth_const(
        g,
        params,
        &format!("{tag}.tk.base"),
        vec![NEG; rows * width],
        &[rows, width],
    );
    let updates = synth_const(
        g,
        params,
        &format!("{tag}.tk.keep"),
        vec![0f32; rows * k],
        &[rows, k],
    );
    let kept = g.add_node(
        Op::ScatterElements {
            axis: 1,
            reduction: rlx_ir::op::ScatterNdReduction::None,
        },
        vec![base, idx, updates],
        Shape::new(&[rows, width], f),
    );
    g.add(kept, causal_add)
}

#[cfg(test)]
mod tests {
    use super::{NEG, candidate_block_bounds};

    /// A query reaches block `b` iff it has passed at least one of that block's
    /// positions — `b · block < len`, not `<=`. The off-by-one here would let a
    /// query consider a block whose first latent does not exist yet.
    #[test]
    fn a_block_is_reachable_once_its_first_position_exists() {
        let (nblocks, block) = (4usize, 2usize);
        // query 0 has passed 0 latents, query 1 has 1, query 2 has 2, query 3 has 5
        let (reach, _) = candidate_block_bounds(&[0, 1, 2, 5], nblocks, block);
        let reachable = |qi: usize| {
            (0..nblocks)
                .filter(|&b| reach[qi * nblocks + b] == 0.0)
                .collect::<Vec<_>>()
        };
        assert_eq!(reachable(0), Vec::<usize>::new(), "no latents, no blocks");
        assert_eq!(reachable(1), vec![0], "one latent reaches only block 0");
        assert_eq!(reachable(2), vec![0], "block 1 starts at position 2");
        assert_eq!(reachable(3), vec![0, 1, 2], "5 latents reach blocks 0..=2");
    }

    /// The pin marks the block holding the query's *newest* latent — index
    /// `(len - 1) / block`, not `len / block`, which would point one block past
    /// the end whenever `len` lands on a boundary.
    #[test]
    fn the_pin_marks_the_block_holding_the_newest_latent() {
        let (nblocks, block) = (4usize, 2usize);
        let (_, pin) = candidate_block_bounds(&[1, 2, 3, 8], nblocks, block);
        let pinned = |qi: usize| {
            (0..nblocks)
                .filter(|&b| pin[qi * nblocks + b] != 0.0)
                .collect::<Vec<_>>()
        };
        assert_eq!(pinned(0), vec![0], "latent 0 is in block 0");
        assert_eq!(pinned(1), vec![0], "latent 1 is still block 0, not block 1");
        assert_eq!(pinned(2), vec![1], "latent 2 opens block 1");
        assert_eq!(pinned(3), vec![3], "latent 7 is in block 3");
    }

    /// The pinned block is always a reachable one — it holds a latent the query
    /// has passed — and the pin must dominate any real score, so the newest
    /// block wins its top-k slot outright.
    #[test]
    fn the_pin_always_lands_on_a_reachable_block_and_dominates() {
        for len in 1..=8usize {
            let (reach, pin) = candidate_block_bounds(&[len], 4, 2);
            let at = (0..4)
                .find(|&b| pin[b] != 0.0)
                .expect("something is pinned");
            assert_eq!(
                reach[at], 0.0,
                "len {len}: the pinned block must be reachable"
            );
            assert!(
                reach[at] + pin[at] > -NEG / 2.0,
                "len {len}: the pin must outrank any real score"
            );
        }
        // and an unreachable, unpinned block stays masked
        let (reach, pin) = candidate_block_bounds(&[3], 4, 2);
        assert_eq!(reach[3] + pin[3], NEG);
    }

    /// A query with no latents must pin nothing at all — indexing `len - 1`
    /// would underflow.
    #[test]
    fn a_query_with_no_latents_pins_nothing() {
        let (_, pin) = candidate_block_bounds(&[0], 3, 4);
        assert!(pin.iter().all(|&v| v == 0.0));
    }
}
