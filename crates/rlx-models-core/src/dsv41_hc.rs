// RLX — versatile ML compiler + runtime. GPLv3.
//! **Hyper-Connections** with a lagged pre-mix, as DeepSeek-V4.1 uses them.
//!
//! The residual stream is `hc_mult` parallel copies rather than one vector. Each
//! sublayer reads a mix of those copies, runs its body on the mixed vector, and
//! writes the result back across the copies — with the mixing weights coming
//! from a Sinkhorn-normalized matrix the *previous* sublayer computed.
//!
//! That lag is the part worth remembering: `hc_sublayer` returns the mix the
//! **next** sublayer will consume, so the pre-mix threaded through the layer loop
//! is always one step ahead of the stream. V4.1 also has no `hc_head`, which is
//! where it parts company with V4.
//!
//! Reference: `deepseek-ai/DeepSeek-V4.1-Flash` `inference/model.py`.

use crate::dsv41::DeepseekV41Spec;
use crate::dsv41_block::Ctx;
use crate::standard_decoder::{build_hc_sinkhorn, const1, synth_const};
use anyhow::{Result, anyhow};
use rlx_ir::GraphExt;
use rlx_ir::graph::{Graph, NodeId};
use std::collections::HashMap;

/// Hyper-Connection mixes: RMS-normalize the flattened `hc·dim` stream, project
/// to the `(2+hc)·hc` mixing vector, then Sinkhorn-split it into
/// `(pre, post, comb)`. `hc_fn_t` is the transposed `[hc·dim, (2+hc)·hc]` weight.
///
/// The RMS here uses `norm_eps` while the Sinkhorn uses `hc_eps`; the reference
/// keeps them separate and V4.1 sets them 14 orders of magnitude apart.
fn hc_mixes(
    ctx: &mut Ctx<'_>,
    x: NodeId,
    hc_fn_t: NodeId,
    scale: NodeId,
    base: NodeId,
    tag: &str,
) -> (NodeId, NodeId, NodeId) {
    let (rows, hc, d) = (ctx.rows, ctx.spec.hc_mult, ctx.spec.dim);
    let (norm_eps, hc_eps) = (ctx.eps(), ctx.spec.hc_eps);
    let iters = ctx.spec.hc_mult_sinkhorn_iters;
    let x_flat = ctx.g.reshape_(x, vec![rows as i64, (hc * d) as i64]);
    let sq = ctx.g.mul(x_flat, x_flat);
    let ms = ctx.g.mean(sq, vec![1], true);
    let eps_c = const1(
        &mut ctx.g,
        &mut ctx.params,
        &format!("{tag}.hcm.eps"),
        norm_eps,
    );
    let ms = ctx.g.add(ms, eps_c);
    let rsq = ctx.g.rsqrt(ms);
    let mixes = ctx.g.mm(x_flat, hc_fn_t);
    let mixes = ctx.g.mul(mixes, rsq);
    build_hc_sinkhorn(
        &mut ctx.g,
        &mut ctx.params,
        mixes,
        scale,
        base,
        rows,
        hc,
        hc_eps,
        iters,
        tag,
    )
}

/// HC post-expand, `1 → hc` streams:
/// `y[j] = post[j]·x_out + Σ_i comb[i, j]·residual[i]`.
///
/// Note the contraction: the reference sums over the **first** axis of `comb`
/// (`(comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)` aligns `comb`'s
/// leading `hc` with `residual`'s, then reduces it), i.e. `combᵀ · residual`.
/// Contracting the other way is a plausible-looking transpose that survives every
/// shape check and quietly permutes the stream mixing.
pub(crate) fn hc_post(
    g: &mut Graph,
    x_out: NodeId,
    residual: NodeId,
    post: NodeId,
    comb: NodeId,
    rows: usize,
    hc: usize,
    d: usize,
) -> NodeId {
    let (r, h, dd) = (rows as i64, hc as i64, d as i64);
    let post3 = g.reshape_(post, vec![r, h, 1]);
    let xo3 = g.reshape_(x_out, vec![r, 1, dd]);
    let term1 = g.mul(post3, xo3); // [rows, hc, d]
    let comb4 = g.reshape_(comb, vec![r, h, h, 1]); // [rows, i, j, 1]
    let res4 = g.reshape_(residual, vec![r, h, 1, dd]); // [rows, i, 1, d]
    let prod = g.mul(comb4, res4); // [rows, i, j, d]
    let term2 = g.sum(prod, vec![1], false); // Σ_i → [rows, j, d]
    g.add(term1, term2)
}

/// Collapse the `hc` copies into one sublayer input: `Σ_hc pre·x`.
/// `x` is `[rows, hc, d]`, `pre` is `[rows, hc]`.
pub(crate) fn hc_reduce(g: &mut Graph, x: NodeId, pre: NodeId, rows: usize, hc: usize) -> NodeId {
    let pre3 = g.reshape_(pre, vec![rows as i64, hc as i64, 1]);
    let yh = g.mul(pre3, x);
    g.sum(yh, vec![1], false)
}

/// The one-hot initial pre-mix (`make_identity_pre_mix`): stream 0 only.
pub(crate) fn identity_pre_mix(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    rows: usize,
    hc: usize,
) -> NodeId {
    let mut d = vec![0f32; rows * hc];
    for r in 0..rows {
        d[r * hc] = 1.0;
    }
    synth_const(g, params, "v41.hc.identity_pre", d, &[rows, hc])
}

/// Mean over the `hc` copies: what a DSpark target layer contributes to
/// `main_hidden`. Taken from the stream *entering* the layer (after any Engram
/// write), not from its output.
pub(crate) fn hc_mean(g: &mut Graph, x: NodeId, rows: usize, d: usize) -> NodeId {
    let m = g.mean(x, vec![1], false); // [rows, d]
    g.reshape_(m, vec![rows as i64, d as i64])
}

/// Concatenate the collected target-layer means into `main_hidden`.
///
/// A stage that holds none of the target layers is an error rather than an empty
/// tensor: DSpark would otherwise be conditioned on nothing at all.
pub(crate) fn main_hidden_node(
    g: &mut Graph,
    parts: &[NodeId],
    spec: &DeepseekV41Spec,
    rows: usize,
) -> Result<NodeId> {
    let want = spec.dspark_target_layer_ids.len();
    if parts.len() != want {
        return Err(anyhow!(
            "deepseek_v41: main_hidden needs all {want} target layers {:?}, this stage held {}",
            spec.dspark_target_layer_ids,
            parts.len()
        ));
    }
    let cat = g.concat_(parts.to_vec(), 1);
    Ok(g.reshape_(cat, vec![rows as i64, (spec.dim * want) as i64]))
}

/// Which Hyper-Connection parameter set a sublayer uses.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum HcSide {
    Attn,
    Ffn,
}

impl HcSide {
    fn tag(self) -> &'static str {
        match self {
            HcSide::Attn => "attn",
            HcSide::Ffn => "ffn",
        }
    }
}

/// One Hyper-Connection-wrapped sublayer: mix, collapse, norm, `body`, expand.
///
/// ```text
/// (pre, post, comb) = sinkhorn(mixes(h))      // this sublayer's coefficients
/// y                 = body(norm(Σ_hc pre_in · h))
/// h'                = post · y + combᵀ · h
/// ```
///
/// The catch is `pre_in`. V4.1 threads the pre-mix **forward**: a sublayer
/// collapses the stream with the mix the *previous* one computed, not its own,
/// and hands its own `pre` to the next. So attention uses what the previous
/// layer's FFN produced, the FFN uses what this attention produced, and the
/// stack ends by collapsing with the last FFN's — there is no `hc_head`. Getting
/// that wrong is invisible to any shape check, which is why all three builders
/// call this rather than each rolling it.
///
/// Returns the new stream and the `pre` this sublayer produced — or, if a debug
/// tap fired inside the body, that tap's tensor and nothing more.
pub(crate) fn hc_sublayer<F>(
    ctx: &mut Ctx<'_>,
    side: HcSide,
    lp: &str,
    h: NodeId,
    pre_in: NodeId,
    body: F,
) -> Result<SublayerOut>
where
    F: FnOnce(&mut Ctx<'_>, NodeId) -> Result<BodyOut>,
{
    let split = hc_split(ctx, side, lp, h, pre_in)?;
    let y = match body(ctx, split.x)? {
        BodyOut::Out(y) => y,
        // a tap fired: stop here rather than build `hc_post` on a tensor that is
        // not the sublayer's output at all
        BodyOut::Tapped(n) => return Ok(SublayerOut::Tapped(n)),
    };
    Ok(SublayerOut::Done {
        h: hc_join(ctx, y, h, &split),
        pre_out: split.pre_out,
    })
}

/// A sublayer's Hyper-Connection state, held open across the body.
///
/// Splitting it out lets a builder **cut the graph inside the sublayer** — which
/// is what paged execution needs, since the routed experts cannot be chosen until
/// the router has run on the host. Everything here is cheap to carry across that
/// boundary: `post` and `comb` are `[rows, hc]` and `[rows, hc, hc]`.
pub(crate) struct HcSplit {
    /// The mixed and normed sublayer input — what the body consumes.
    pub x: NodeId,
    /// The mix the *next* sublayer consumes. V4.1's pre-mix is lagged, so this
    /// is produced here and used one sublayer later.
    pub pre_out: NodeId,
    /// Per-copy output weights for writing the body's result back.
    pub post: NodeId,
    /// Residual recombination matrix.
    pub comb: NodeId,
}

/// Everything a sublayer does *before* its body.
pub(crate) fn hc_split(
    ctx: &mut Ctx<'_>,
    side: HcSide,
    lp: &str,
    h: NodeId,
    pre_in: NodeId,
) -> Result<HcSplit> {
    let (rows, hc, d) = (ctx.rows, ctx.spec.hc_mult, ctx.spec.dim);
    let eps = ctx.eps();
    let t = side.tag();

    let fn_w = ctx.transposed(&format!("{lp}.hc_{t}_fn"))?;
    let scale = ctx.param(&format!("{lp}.hc_{t}_scale"), false)?;
    let base = ctx.param(&format!("{lp}.hc_{t}_base"), false)?;
    let (pre_out, post, comb) = hc_mixes(
        ctx,
        h,
        fn_w,
        scale,
        base,
        &format!("{lp}.{}", if side == HcSide::Attn { "a" } else { "f" }),
    );

    let x = hc_reduce(&mut ctx.g, h, pre_in, rows, hc);
    let gain = ctx.norm(&format!("{lp}.{t}_norm.weight"))?;
    let zb = ctx.zero_bias(&format!("{lp}.{t}.zb"), d);
    let x = ctx.g.rms_norm(x, gain, zb, eps);
    Ok(HcSplit {
        x,
        pre_out,
        post,
        comb,
    })
}

/// Everything a sublayer does *after* its body: write `y` back across the
/// Hyper-Connection copies and recombine the residual.
pub(crate) fn hc_join(ctx: &mut Ctx<'_>, y: NodeId, h: NodeId, split: &HcSplit) -> NodeId {
    let (rows, hc, d) = (ctx.rows, ctx.spec.hc_mult, ctx.spec.dim);
    hc_post(&mut ctx.g, y, h, split.post, split.comb, rows, hc, d)
}

/// What a sublayer body produced.
pub(crate) enum BodyOut {
    Out(NodeId),
    /// A debug tap fired; this tensor is the graph's output and nothing after it
    /// should be built.
    Tapped(NodeId),
}

/// What [`hc_sublayer`] produced.
pub(crate) enum SublayerOut {
    Done { h: NodeId, pre_out: NodeId },
    Tapped(NodeId),
}

/// [`hc_sublayer`] for the paths that have no taps (decode, the DSpark draft).
pub(crate) fn hc_sublayer_plain<F>(
    ctx: &mut Ctx<'_>,
    side: HcSide,
    lp: &str,
    h: NodeId,
    pre_in: NodeId,
    body: F,
) -> Result<(NodeId, NodeId)>
where
    F: FnOnce(&mut Ctx<'_>, NodeId) -> Result<NodeId>,
{
    match hc_sublayer(ctx, side, lp, h, pre_in, |c, x| {
        body(c, x).map(BodyOut::Out)
    })? {
        SublayerOut::Done { h, pre_out } => Ok((h, pre_out)),
        SublayerOut::Tapped(_) => unreachable!("hc_sublayer_plain body never taps"),
    }
}
