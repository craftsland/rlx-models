// RLX — versatile ML compiler + runtime. GPLv3.
//! The **DeepSeek-V4.1** MoE FFN: top-k routed experts over a per-stage bank,
//! plus one shared expert every token goes through.
//!
//! The routing detail worth knowing is that the correction bias steers
//! *selection* only — the weights come from the unbiased scores — and that a
//! vision-enabled checkpoint carries a second bias for tokens inside an image
//! span. The DSpark stages route over their own, smaller bank
//! (`dspark_n_routed_experts`), which is why the expert counts come from
//! [`DeepseekV41Spec::moe_dims`] rather than from the top-level fields.

use crate::dsv41::{DeepseekV41Spec, ScoreFunc};
use crate::dsv41_block::Ctx;
use crate::standard_decoder::{const1, softplus_stable};
use anyhow::{Result, anyhow};
use rlx_ir::GraphExt;
use rlx_ir::graph::{Graph, NodeId};
use rlx_ir::op::Op;
use rlx_ir::{DType, Shape};

/// Load `n_experts` per-expert `[out, in]` weights and stack them into the
/// `[E, in, out]` bank [`Op::GroupedMatMul`] expects.
///
/// The bank axis order matters: a `[E, out, in]` bank is silently mis-read, so
/// each expert is transposed on the way in.
fn load_expert_bank(
    ctx: &mut Ctx<'_>,
    lp: &str,
    proj: &str,
    n_experts: usize,
    k: usize,
    n: usize,
) -> Result<NodeId> {
    let mut data = Vec::with_capacity(n_experts * k * n);
    for e in 0..n_experts {
        let key = format!("{lp}.ffn.experts.{e}.{proj}.weight");
        let (w, shape) = ctx.weights.take_transposed(&key)?;
        if shape != vec![k, n] {
            return Err(anyhow!(
                "deepseek_v41: {key} is {shape:?} transposed, expected [{k}, {n}]"
            ));
        }
        data.extend_from_slice(&w);
    }
    let key = format!("{lp}.ffn.experts.{proj}.bank");
    let node = ctx
        .g
        .param(&key, Shape::new(&[n_experts, k, n], DType::F32));
    ctx.params.insert(key, data);
    Ok(node)
}

/// Clamped SwiGLU: `up ∈ [-L, L]`, `gate ≤ L`, then `silu(gate)·up`. The clamps
/// come from training, where they keep FP8/FP4 activations in range.
fn clamped_swiglu(g: &mut Graph, gate: NodeId, up: NodeId, limit: f32) -> NodeId {
    let (gate, up) = if limit > 0.0 {
        (g.clamp_(gate, f32::MIN, limit), g.clamp_(up, -limit, limit))
    } else {
        (gate, up)
    };
    let a = g.silu(gate);
    g.mul(a, up)
}

/// Router: scores, the selection bias, top-k, and the weights that scale the
/// chosen experts.
///
/// Returns `(top_idx, top_w)`, both `[rows, top_k]`. The bias steers *selection*
/// only — `top_w` is gathered from the **unbiased** scores.
fn build_router(
    ctx: &mut Ctx<'_>,
    lp: &str,
    x: NodeId,
    rows: usize,
    top_k: usize,
    image_mask: Option<NodeId>,
) -> Result<(NodeId, NodeId)> {
    let f = DType::F32;
    let spec = ctx.spec;
    let (score_func, gate_temp) = (spec.score_func, spec.gate_temp);
    let (norm_topk, route_scale) = (spec.norm_topk_prob, spec.route_scale);
    let has_vision = spec.vision.is_some();

    let gate_w = ctx.transposed(&format!("{lp}.ffn.gate.weight"))?;
    let mut logits = ctx.g.mm(x, gate_w); // [rows, n_experts]
    if (gate_temp - 1.0).abs() > f32::EPSILON {
        let t = const1(
            &mut ctx.g,
            &mut ctx.params,
            &format!("{lp}.moe.temp"),
            1.0 / gate_temp,
        );
        logits = ctx.g.mul(logits, t);
    }
    let scores = match score_func {
        ScoreFunc::Softmax => ctx.g.sm(logits, -1),
        ScoreFunc::Sigmoid => ctx.g.sigmoid(logits),
        ScoreFunc::SqrtSoftplus => {
            let sp = softplus_stable(&mut ctx.g, &mut ctx.params, logits, &format!("{lp}.moe.sp"));
            ctx.g.sqrt(sp)
        }
    };

    let bias = ctx.param(&format!("{lp}.ffn.gate.bias"), false)?;
    // `bias_vl` only exists on vision-enabled checkpoints, and only matters when
    // this prefill actually contains an image span.
    let bias = match (image_mask, has_vision) {
        (Some(m), true) => {
            let bias_vl = ctx.param(&format!("{lp}.ffn.gate.bias_vl"), false)?;
            let delta = ctx.g.sub(bias_vl, bias);
            let m2 = ctx.g.reshape_(m, vec![rows as i64, 1]);
            let scaled = ctx.g.mul(m2, delta);
            ctx.g.add(scaled, bias)
        }
        _ => bias,
    };
    let route = ctx.g.add(scores, bias);

    let top_idx = ctx.g.add_node(
        Op::TopK { k: top_k },
        vec![route],
        Shape::new(&[rows, top_k], f),
    );
    let mut top_w = ctx.g.add_node(
        Op::GatherElements { axis: 1 },
        vec![scores, top_idx],
        Shape::new(&[rows, top_k], f),
    );
    if norm_topk && top_k > 1 {
        // `+ 1e-20`, not `rms_norm_eps` — the reference pins this constant to
        // what training used regardless of `norm_eps`.
        let denom = ctx.g.sum(top_w, vec![1], true);
        let tiny = const1(
            &mut ctx.g,
            &mut ctx.params,
            &format!("{lp}.moe.tiny"),
            1e-20,
        );
        let denom = ctx.g.add(denom, tiny);
        top_w = ctx.g.div(top_w, denom);
    }
    if (route_scale - 1.0).abs() > f32::EPSILON {
        let sc = const1(
            &mut ctx.g,
            &mut ctx.params,
            &format!("{lp}.moe.rscale"),
            route_scale,
        );
        top_w = ctx.g.mul(top_w, sc);
    }
    Ok((top_idx, top_w))
}

/// One MoE FFN: top-k routed experts over this layer's bank, plus the shared
/// expert every token goes through.
pub(crate) fn build_v41_moe(
    ctx: &mut Ctx<'_>,
    il: usize,
    x: NodeId,
    rows: usize,
    image_mask: Option<NodeId>,
) -> Result<NodeId> {
    let lp = ctx.spec.layer_prefix(il);
    let d = ctx.spec.dim;
    let inter = ctx.spec.moe_intermediate_size;
    let swiglu_limit = ctx.spec.swiglu_limit;
    let n_shared = ctx.spec.n_shared_experts.max(1);
    let (n_experts, top_k) = ctx.spec.moe_dims(il);
    if top_k == 0 {
        return Err(anyhow!("deepseek_v41: layer {il} has top_k = 0"));
    }

    let (top_idx, top_w) = build_router(ctx, &lp, x, rows, top_k, image_mask)?;

    let w1 = load_expert_bank(ctx, &lp, "w1", n_experts, d, inter)?;
    let w3 = load_expert_bank(ctx, &lp, "w3", n_experts, d, inter)?;
    let w2 = load_expert_bank(ctx, &lp, "w2", n_experts, inter, d)?;
    let routed = accumulate_experts(
        ctx,
        x,
        top_idx,
        top_w,
        w1,
        w3,
        w2,
        rows,
        top_k,
        inter,
        d,
        swiglu_limit,
    );

    let se = n_shared * inter;
    let sg = ctx.project(&format!("{lp}.ffn.shared_experts.w1.weight"), x, rows, se)?;
    let su = ctx.project(&format!("{lp}.ffn.shared_experts.w3.weight"), x, rows, se)?;
    let sglu = clamped_swiglu(&mut ctx.g, sg, su, swiglu_limit);
    let sdown = ctx.project(&format!("{lp}.ffn.shared_experts.w2.weight"), sglu, rows, d)?;
    Ok(ctx.g.add(routed, sdown))
}

/// Graph-input names for one layer's paged MoE.
pub mod paged_names {
    /// `[n_sel, in, out]` gathered bank for one projection.
    pub fn bank(il: usize, proj: &str) -> String {
        format!("moe.{il}.bank.{proj}")
    }
    /// `[rows, top_k]` slot indices into the gathered bank.
    pub fn slots(il: usize) -> String {
        format!("moe.{il}.slots")
    }
    /// `[rows, top_k]` routing weights.
    pub fn weights(il: usize) -> String {
        format!("moe.{il}.w")
    }
}

/// The MoE FFN with the routed experts supplied from outside the graph.
///
/// [`build_v41_moe`] puts all `n_routed_experts` in the graph as a parameter.
/// That is ~34 GB per layer at the released size, so a runner that cannot hold
/// the bank routes on the host ([`route_on_host`]), pages in only the experts a
/// token actually needs, and hands them here as a `[n_sel, in, out]` **input**
/// with the indices rewritten to slots ([`Routing::to_slots`]).
///
/// The shared expert stays a parameter: every token goes through it, so there is
/// nothing to page.
pub(crate) fn build_v41_moe_paged(
    ctx: &mut Ctx<'_>,
    il: usize,
    x: NodeId,
    rows: usize,
    n_sel: usize,
    top_k: usize,
) -> Result<NodeId> {
    let lp = ctx.spec.layer_prefix(il);
    let f = DType::F32;
    let d = ctx.spec.dim;
    let inter = ctx.spec.moe_intermediate_size;
    let swiglu_limit = ctx.spec.swiglu_limit;
    let n_shared = ctx.spec.n_shared_experts.max(1);
    if top_k == 0 {
        return Err(anyhow!("deepseek_v41: layer {il} has top_k = 0"));
    }
    #[allow(clippy::nonminimal_bool)]
    if n_sel == 0 {
        return Err(anyhow!(
            "deepseek_v41: layer {il} has an empty gathered bank"
        ));
    }

    let w1 = ctx.g.input(
        paged_names::bank(il, "w1"),
        Shape::new(&[n_sel, d, inter], f),
    );
    let w3 = ctx.g.input(
        paged_names::bank(il, "w3"),
        Shape::new(&[n_sel, d, inter], f),
    );
    let w2 = ctx.g.input(
        paged_names::bank(il, "w2"),
        Shape::new(&[n_sel, inter, d], f),
    );
    let slots = ctx
        .g
        .input(paged_names::slots(il), Shape::new(&[rows, top_k], f));
    let top_w = ctx
        .g
        .input(paged_names::weights(il), Shape::new(&[rows, top_k], f));

    let routed = accumulate_experts(
        ctx,
        x,
        slots,
        top_w,
        w1,
        w3,
        w2,
        rows,
        top_k,
        inter,
        d,
        swiglu_limit,
    );
    let se = n_shared * inter;
    let sg = ctx.project(&format!("{lp}.ffn.shared_experts.w1.weight"), x, rows, se)?;
    let su = ctx.project(&format!("{lp}.ffn.shared_experts.w3.weight"), x, rows, se)?;
    let sglu = clamped_swiglu(&mut ctx.g, sg, su, swiglu_limit);
    let sdown = ctx.project(&format!("{lp}.ffn.shared_experts.w2.weight"), sglu, rows, d)?;
    Ok(ctx.g.add(routed, sdown))
}

/// The paged MoE as a standalone graph: `x [rows, dim]` in, the FFN's output
/// out, with the gathered bank and the resolved routing as inputs.
///
/// The crate-internal form builds into a caller's graph, which is what the layer
/// drivers need. This wraps it for a caller that wants the MoE on its own —
/// checking it against a reference, or serving it as one stage of a pipeline.
#[allow(clippy::too_many_arguments)]
pub fn build_paged_moe_graph(
    spec: &DeepseekV41Spec,
    weights: &mut dyn crate::weight_loader::WeightLoader,
    il: usize,
    rows: usize,
    n_sel: usize,
    top_k: usize,
    packed: &mut std::collections::HashMap<
        String,
        (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>),
    >,
) -> Result<(Graph, std::collections::HashMap<String, Vec<f32>>)> {
    let mut ctx = Ctx::new("deepseek_v41_paged_moe", spec, weights, packed, rows);
    let x = ctx.g.input("x", Shape::new(&[rows, spec.dim], DType::F32));
    let out = build_v41_moe_paged(&mut ctx, il, x, rows, n_sel, top_k)?;
    Ok(ctx.finish(vec![out]))
}

/// Sum `top_k` expert contributions, each a clamped SwiGLU through the bank at
/// that slot's index. Shared by the parameter and paged forms so the two cannot
/// drift.
#[allow(clippy::too_many_arguments)]
fn accumulate_experts(
    ctx: &mut Ctx<'_>,
    x: NodeId,
    idx: NodeId,
    weights: NodeId,
    w1: NodeId,
    w3: NodeId,
    w2: NodeId,
    rows: usize,
    top_k: usize,
    inter: usize,
    d: usize,
    swiglu_limit: f32,
) -> NodeId {
    let f = DType::F32;
    let mut acc: Option<NodeId> = None;
    for ki in 0..top_k {
        let e_col = ctx.g.narrow_(idx, 1, ki, 1);
        let e_idx = ctx.g.reshape_(e_col, vec![rows as i64]);
        let w_col = ctx.g.narrow_(weights, 1, ki, 1);
        let gate = ctx.g.add_node(
            Op::GroupedMatMul,
            vec![x, w1, e_idx],
            Shape::new(&[rows, inter], f),
        );
        let up = ctx.g.add_node(
            Op::GroupedMatMul,
            vec![x, w3, e_idx],
            Shape::new(&[rows, inter], f),
        );
        let glu = clamped_swiglu(&mut ctx.g, gate, up, swiglu_limit);
        let down = ctx.g.add_node(
            Op::GroupedMatMul,
            vec![glu, w2, e_idx],
            Shape::new(&[rows, d], f),
        );
        let weighted = ctx.g.mul(down, w_col);
        acc = Some(match acc {
            None => weighted,
            Some(a) => ctx.g.add(a, weighted),
        });
    }
    acc.expect("top_k > 0 checked by the callers")
}

/// What the router chose: `[rows · top_k]` expert ids and their weights.
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    pub top_k: usize,
    /// Expert id per (row, slot).
    pub idx: Vec<usize>,
    /// Weight per (row, slot), already normalized and scaled.
    pub w: Vec<f32>,
}

impl Routing {
    /// Every expert this routing touches, ascending — the set a pager has to
    /// make resident, and the order that defines the gathered bank's slots.
    pub fn distinct(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.idx.clone();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Rewrite `idx` from expert ids into slot indices within `bank`, which must
    /// be the output of [`Self::distinct`].
    ///
    /// This is what lets the graph carry only the routed experts: the bank has
    /// `bank.len()` entries instead of `n_routed_experts`, and the indices point
    /// into it.
    pub fn to_slots(&self, bank: &[usize]) -> Vec<usize> {
        let mut pos = std::collections::HashMap::with_capacity(bank.len());
        for (s, &e) in bank.iter().enumerate() {
            pos.insert(e, s);
        }
        self.idx.iter().map(|e| pos[e]).collect()
    }
}

/// `softplus(x)` in the same stable form the graph uses — `relu(x) +
/// log1p(exp(-|x|))`. Writing it any other way diverges from the graph on large
/// magnitudes, which is exactly where routing decisions are made.
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// The routed-expert gate, evaluated on the host.
///
/// This exists so a paged runner can learn which experts a token needs *before*
/// building the graph that uses them. It has to agree with the in-graph router node
/// for node — including the two details that are easy to get wrong:
///
/// * the correction bias steers **selection only**; `w` comes from the unbiased
///   scores;
/// * `Op::TopK` breaks ties toward the **smaller index**, and the gate really
///   does produce ties (a fresh checkpoint's bias is uniform), so a host router
///   that used a different tie rule would silently pick different experts than
///   the graph does.
///
/// `x` is `[rows, dim]`, `gate_w` is `[n_experts, dim]` as stored.
#[allow(clippy::too_many_arguments)]
pub fn route_on_host(
    spec: &DeepseekV41Spec,
    x: &[f32],
    rows: usize,
    gate_w: &[f32],
    bias: &[f32],
    bias_vl: Option<&[f32]>,
    image_positions: &[bool],
    top_k: usize,
) -> Result<Routing> {
    let d = spec.dim;
    let n_e = gate_w.len() / d.max(1);
    if n_e == 0 || gate_w.len() != n_e * d {
        return Err(anyhow!(
            "deepseek_v41 router: gate weight is {} elements, not a multiple of dim {d}",
            gate_w.len()
        ));
    }
    if bias.len() != n_e {
        return Err(anyhow!(
            "deepseek_v41 router: bias is {} long, expected {n_e}",
            bias.len()
        ));
    }
    if top_k == 0 || top_k > n_e {
        return Err(anyhow!(
            "deepseek_v41 router: top_k {top_k} out of range for {n_e} experts"
        ));
    }

    let mut idx = vec![0usize; rows * top_k];
    let mut wout = vec![0f32; rows * top_k];
    let mut scores = vec![0f32; n_e];
    let mut route = vec![0f32; n_e];
    for r in 0..rows {
        for e in 0..n_e {
            let mut acc = 0f64;
            for c in 0..d {
                acc += (x[r * d + c] * gate_w[e * d + c]) as f64;
            }
            scores[e] = acc as f32;
        }
        if (spec.gate_temp - 1.0).abs() > f32::EPSILON {
            for s in scores.iter_mut() {
                *s /= spec.gate_temp;
            }
        }
        match spec.score_func {
            ScoreFunc::Softmax => {
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }
            }
            ScoreFunc::Sigmoid => {
                for s in scores.iter_mut() {
                    *s = 1.0 / (1.0 + (-*s).exp());
                }
            }
            ScoreFunc::SqrtSoftplus => {
                for s in scores.iter_mut() {
                    *s = softplus(*s).sqrt();
                }
            }
        }
        // a position inside an image span routes through the VL bias instead
        let in_image = image_positions.get(r).copied().unwrap_or(false);
        let b = match (in_image, bias_vl) {
            (true, Some(v)) => v,
            _ => bias,
        };
        for e in 0..n_e {
            route[e] = scores[e] + b[e];
        }

        // top_k by score, ties to the smaller index — `Op::TopK`'s contract
        let mut order: Vec<usize> = (0..n_e).collect();
        order.sort_by(|&a, &c| {
            route[c]
                .partial_cmp(&route[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&c))
        });
        let mut sum = 0f32;
        for s in 0..top_k {
            let e = order[s];
            idx[r * top_k + s] = e;
            // weights come from the UNBIASED scores
            wout[r * top_k + s] = scores[e];
            sum += scores[e];
        }
        if spec.norm_topk_prob && top_k > 1 {
            // `+ 1e-20`, matching the graph — not `rms_norm_eps`
            let denom = sum + 1e-20;
            for s in 0..top_k {
                wout[r * top_k + s] /= denom;
            }
        }
        if (spec.route_scale - 1.0).abs() > f32::EPSILON {
            for s in 0..top_k {
                wout[r * top_k + s] *= spec.route_scale;
            }
        }
    }
    Ok(Routing {
        top_k,
        idx,
        w: wout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41_block::Ctx;
    use crate::weight_loader::SyntheticLoader;
    use rlx_runtime::{Device, Session};
    use std::collections::BTreeMap;

    fn spec(n_e: usize, top_k: usize, extra: serde_json::Value) -> DeepseekV41Spec {
        let mut cfg = serde_json::json!({
            "vocab_size": 32, "hidden_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 1, "head_dim": 8, "o_lora_rank": 2,
            "n_routed_experts": n_e, "num_experts_per_tok": top_k,
            "moe_intermediate_size": 4,
        });
        for (k, v) in extra.as_object().unwrap() {
            cfg[k] = v.clone();
        }
        DeepseekV41Spec::from_config(&cfg).expect("toy config parses")
    }

    /// Run `build_router` on CPU and `route_on_host` on the same weights, and
    /// require the same experts and weights out of both.
    ///
    /// This is the contract the paged runner rests on: it has to know which
    /// experts a token needs *before* it builds the graph, so a host router that
    /// disagrees with the graph loads the wrong weights and produces a plausible
    /// wrong answer rather than an error.
    fn agree(sp: &DeepseekV41Spec, rows: usize, n_e: usize, top_k: usize, gate: Vec<f32>) {
        agree_with_bias(
            sp,
            rows,
            n_e,
            top_k,
            gate,
            SyntheticLoader::values("router.bias", &[n_e]),
        )
    }

    fn agree_with_bias(
        sp: &DeepseekV41Spec,
        rows: usize,
        n_e: usize,
        top_k: usize,
        gate: Vec<f32>,
        bias: Vec<f32>,
    ) {
        let d = sp.dim;
        let x = SyntheticLoader::values("router.x", &[rows * d]);

        let mut shapes = BTreeMap::new();
        shapes.insert("layers.0.ffn.gate.weight".to_string(), vec![n_e, d]);
        shapes.insert("layers.0.ffn.gate.bias".to_string(), vec![n_e]);
        let mut loader = SyntheticLoader::new(shapes);
        loader.preset("layers.0.ffn.gate.weight", gate.clone());
        loader.preset("layers.0.ffn.gate.bias", bias.clone());

        let mut packed = std::collections::HashMap::new();
        let mut ctx = Ctx::new("router", sp, &mut loader, &mut packed, rows);
        let xn = ctx
            .g
            .input("x", rlx_ir::Shape::new(&[rows, d], rlx_ir::DType::F32));
        let (ti, tw) = build_router(&mut ctx, "layers.0", xn, rows, top_k, None).expect("router");
        let (g, params) = ctx.finish(vec![ti, tw]);

        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            Device::Cpu,
        );
        let mut s = Session::new(Device::Cpu).compile_with(g, &opts);
        for (k, v) in &params {
            s.set_param(k, v);
        }
        let out = s.run(&[("x", x.as_slice())]);
        let g_idx: Vec<usize> = out[0].iter().map(|f| *f as usize).collect();
        let g_w = &out[1];

        let host =
            route_on_host(sp, &x, rows, &gate, &bias, None, &[], top_k).expect("host router");
        assert_eq!(host.idx, g_idx, "expert selection differs from the graph");
        for (i, (a, b)) in host.w.iter().zip(g_w).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                "weight {i}: host {a} vs graph {b}"
            );
        }
    }

    #[test]
    fn host_router_matches_the_graph_router() {
        let sp = spec(8, 3, serde_json::json!({}));
        let gate = SyntheticLoader::values("router.gate", &[8 * sp.dim]);
        agree(&sp, 5, 8, 3, gate);
    }

    /// Every scoring function, since each is a different node chain.
    #[test]
    fn host_router_matches_for_each_score_func() {
        for f in ["softmax", "sigmoid", "sqrtsoftplus"] {
            let sp = spec(
                8,
                3,
                serde_json::json!({ "scoring_func": f, "routed_scaling_factor": 2.5 }),
            );
            let gate = SyntheticLoader::values("router.gate", &[8 * sp.dim]);
            agree(&sp, 4, 8, 3, gate);
        }
    }

    /// A router whose scores are *exactly* equal, so the whole selection is
    /// decided by the tie-break and nothing else.
    ///
    /// Both the gate and the bias have to be zeroed: zeroing only the gate makes
    /// the logits equal, but then the distinct bias orders them again and the
    /// test stops testing ties at all.
    #[test]
    fn host_router_matches_the_graph_tie_rule() {
        let sp = spec(8, 3, serde_json::json!({}));
        agree_with_bias(&sp, 3, 8, 3, vec![0f32; 8 * sp.dim], vec![0f32; 8]);
    }

    /// Ties among only *some* experts — the realistic case, where a few
    /// candidates are separated and the last top-k slot is contested.
    #[test]
    fn host_router_matches_when_only_the_last_slot_is_contested() {
        let sp = spec(8, 3, serde_json::json!({}));
        // two clear winners, then five experts tied for the third slot
        let mut bias = vec![0f32; 8];
        bias[5] = 1.0;
        bias[1] = 0.5;
        agree_with_bias(&sp, 3, 8, 3, vec![0f32; 8 * sp.dim], bias);
    }

    /// `norm_topk_prob = false` leaves the weights unnormalized, and `top_k = 1`
    /// skips normalization even when it is on.
    #[test]
    fn host_router_matches_without_weight_normalization() {
        let sp = spec(6, 2, serde_json::json!({ "norm_topk_prob": false }));
        let gate = SyntheticLoader::values("router.g2", &[6 * sp.dim]);
        agree(&sp, 4, 6, 2, gate);
        let sp1 = spec(6, 1, serde_json::json!({}));
        let gate1 = SyntheticLoader::values("router.g3", &[6 * sp1.dim]);
        agree(&sp1, 4, 6, 1, gate1);
    }

    /// `distinct` + `to_slots` must round-trip: reading the gathered bank at the
    /// slot index must land on the same expert the router named.
    #[test]
    fn slot_indices_point_back_at_the_chosen_experts() {
        let r = Routing {
            top_k: 3,
            idx: vec![7, 2, 7, 0, 2, 5],
            w: vec![0.0; 6],
        };
        let bank = r.distinct();
        assert_eq!(bank, vec![0, 2, 5, 7], "deduped and ascending");
        let slots = r.to_slots(&bank);
        for (i, &s) in slots.iter().enumerate() {
            assert_eq!(bank[s], r.idx[i], "slot {s} must hold expert {}", r.idx[i]);
        }
    }
}

#[cfg(test)]
mod paged_tests {
    use super::*;
    use crate::dsv41_block::Ctx;
    use crate::weight_loader::SyntheticLoader;
    use rlx_runtime::{Device, Session};
    use std::collections::BTreeMap;
    use std::collections::HashMap;

    fn moe_spec(n_e: usize, top_k: usize) -> DeepseekV41Spec {
        DeepseekV41Spec::from_config(&serde_json::json!({
            "vocab_size": 32, "hidden_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 1, "head_dim": 8, "o_lora_rank": 2,
            "n_routed_experts": n_e, "num_experts_per_tok": top_k,
            "moe_intermediate_size": 6, "n_shared_experts": 1,
            "routed_scaling_factor": 1.7, "swiglu_limit": 7.0,
        }))
        .unwrap()
    }

    fn manifest(sp: &DeepseekV41Spec, n_e: usize) -> BTreeMap<String, Vec<usize>> {
        let (d, inter) = (sp.dim, sp.moe_intermediate_size);
        let mut m = BTreeMap::new();
        m.insert("layers.0.ffn.gate.weight".into(), vec![n_e, d]);
        m.insert("layers.0.ffn.gate.bias".into(), vec![n_e]);
        for e in 0..n_e {
            m.insert(
                format!("layers.0.ffn.experts.{e}.w1.weight"),
                vec![inter, d],
            );
            m.insert(
                format!("layers.0.ffn.experts.{e}.w3.weight"),
                vec![inter, d],
            );
            m.insert(
                format!("layers.0.ffn.experts.{e}.w2.weight"),
                vec![d, inter],
            );
        }
        m.insert(
            "layers.0.ffn.shared_experts.w1.weight".into(),
            vec![inter, d],
        );
        m.insert(
            "layers.0.ffn.shared_experts.w3.weight".into(),
            vec![inter, d],
        );
        m.insert(
            "layers.0.ffn.shared_experts.w2.weight".into(),
            vec![d, inter],
        );
        m
    }

    fn run(
        g: rlx_ir::graph::Graph,
        params: &HashMap<String, Vec<f32>>,
        ins: &[(&str, &[f32])],
    ) -> Vec<f32> {
        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            Device::Cpu,
        );
        let mut s = Session::new(Device::Cpu).compile_with(g, &opts);
        for (k, v) in params {
            s.set_param(k, v);
        }
        s.run(ins)[0].clone()
    }

    /// `[out, in]` as stored, transposed to the `[in, out]` a bank slot holds.
    fn transposed(name: &str, out: usize, inn: usize) -> Vec<f32> {
        let v = SyntheticLoader::values(name, &[out, inn]);
        let mut t = vec![0f32; v.len()];
        for i in 0..out {
            for j in 0..inn {
                t[j * out + i] = v[i * inn + j];
            }
        }
        t
    }

    /// Paging must not change the answer.
    ///
    /// The parameter form puts all `n_routed_experts` in the graph and routes
    /// in-graph; the paged form routes on the host, gathers only the experts the
    /// routing named, and indexes them by slot. Those are two very different
    /// graphs, and the whole point of the second is that it computes the same
    /// thing — so this compares them element for element.
    #[test]
    fn paged_moe_matches_the_parameter_moe() {
        let (n_e, top_k, rows) = (8usize, 3usize, 5usize);
        let sp = moe_spec(n_e, top_k);
        let (d, inter) = (sp.dim, sp.moe_intermediate_size);
        let x = SyntheticLoader::values("moe.x", &[rows * d]);

        // ── parameter form ──
        let mut loader = SyntheticLoader::new(manifest(&sp, n_e));
        let mut packed = HashMap::new();
        let mut ctx = Ctx::new("moe_all", &sp, &mut loader, &mut packed, rows);
        let xn = ctx.g.input("x", Shape::new(&[rows, d], DType::F32));
        let out = build_v41_moe(&mut ctx, 0, xn, rows, None).expect("bank moe");
        let (g, params) = ctx.finish(vec![out]);
        let want = run(g, &params, &[("x", x.as_slice())]);

        // ── paged form ──
        let gate = SyntheticLoader::values("layers.0.ffn.gate.weight", &[n_e, d]);
        let bias = SyntheticLoader::values("layers.0.ffn.gate.bias", &[n_e]);
        let routing =
            route_on_host(&sp, &x, rows, &gate, &bias, None, &[], top_k).expect("host routing");
        let bank = routing.distinct();
        let slots: Vec<f32> = routing.to_slots(&bank).iter().map(|&s| s as f32).collect();
        assert!(
            bank.len() < n_e,
            "this fixture should not route to every expert, or paging proves nothing"
        );

        let gather = |proj: &str, out: usize, inn: usize| -> Vec<f32> {
            let mut v = Vec::new();
            for &e in &bank {
                v.extend(transposed(
                    &format!("layers.0.ffn.experts.{e}.{proj}.weight"),
                    out,
                    inn,
                ));
            }
            v
        };
        let b1 = gather("w1", inter, d);
        let b3 = gather("w3", inter, d);
        let b2 = gather("w2", d, inter);

        let mut loader = SyntheticLoader::new(manifest(&sp, n_e));
        let mut packed = HashMap::new();
        let mut ctx = Ctx::new("moe_paged", &sp, &mut loader, &mut packed, rows);
        let xn = ctx.g.input("x", Shape::new(&[rows, d], DType::F32));
        let out = build_v41_moe_paged(&mut ctx, 0, xn, rows, bank.len(), top_k).expect("paged moe");
        let (g, params) = ctx.finish(vec![out]);
        let got = run(
            g,
            &params,
            &[
                ("x", x.as_slice()),
                (&paged_names::bank(0, "w1"), b1.as_slice()),
                (&paged_names::bank(0, "w3"), b3.as_slice()),
                (&paged_names::bank(0, "w2"), b2.as_slice()),
                (&paged_names::slots(0), slots.as_slice()),
                (&paged_names::weights(0), routing.w.as_slice()),
            ],
        );

        assert_eq!(got.len(), want.len());
        let tol = 2e-5;
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() <= tol * b.abs().max(1.0),
                "element {i}: paged {a} vs bank {b}"
            );
        }
    }
}
