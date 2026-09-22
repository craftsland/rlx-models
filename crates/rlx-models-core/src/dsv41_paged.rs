// RLX — versatile ML compiler + runtime. GPLv3.
//! **Paged decode**: one token at a time, one layer at a time, with only the
//! routed experts in memory.
//!
//! [`crate::dsv41_decode::build_deepseek_v41_decode`] builds a step as a single
//! graph, which means every layer's whole expert bank is a graph parameter. At
//! the released size that is ~34 GB per layer as f32 and the model cannot be run
//! at all. But a token only *uses* `num_experts_per_tok` experts per layer, so
//! the working set is four orders of magnitude smaller than the bank — if the
//! runner can find out which experts before it builds the graph.
//!
//! It can, by cutting each layer in two at the router:
//!
//! ```text
//!   head(il):  h, pre_mix, carry  ─►  h_attn, post, comb, pre_out, x_ffn, carry'
//!   host:      route(x_ffn) ─► expert ids ─► pager ─► gathered bank
//!   tail(il):  h_attn, post, comb, x_ffn, bank, slots, w  ─►  h_out
//! ```
//!
//! The cut is cheap to cross. `post` is `[1, hc]`, `comb` is `[1, hc, hc]`, and
//! the CSA2 carry is a handful of `[ncomp, head_dim]` rows — all of it host-side
//! state the runner already keeps for the KV cache.
//!
//! Two properties make the decode case much easier than prefill:
//!
//! * there is **one query row**, so the gathered bank is always exactly `top_k`
//!   wide and the tail graph has a fixed shape — compiled once per layer and
//!   reused for the whole run;
//! * the compressed KV already crosses graph boundaries through the cache, so
//!   feeding it to a consumer layer as an input is what the monolithic graph was
//!   doing internally anyway.
//!
//! The host router ([`crate::dsv41_moe::route_on_host`]) is checked against the
//! in-graph one, because choosing different experts than the graph would is a
//! silently wrong answer rather than an error.

use crate::dsv41::DeepseekV41Spec;
use crate::dsv41_block::{Ctx, Tap, sliding_window_mask};
use crate::dsv41_csa::CsaShared;
use crate::dsv41_decode::V41DecodePlan;

use crate::dsv41_hc::{HcSide, hc_join, hc_split};
use crate::dsv41_moe::{build_v41_moe_paged, paged_names};
use crate::weight_loader::WeightLoader;
use anyhow::{Result, anyhow};
use rlx_ir::GraphExt;
use rlx_ir::graph::Graph;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// Graph-input names for the state a layer boundary carries.
pub mod carry {
    /// Hidden stream entering the layer, `[1, hc_mult, dim]`.
    pub const H_IN: &str = "carry.h";
    /// Lagged Hyper-Connection pre-mix, `[1, hc_mult]`.
    pub const PRE_IN: &str = "carry.pre";
    /// The full compressed cache for a source, *including* this step's latent.
    pub fn compressed(src: usize) -> String {
        format!("carry.comp.{src}")
    }
    /// The matching Indexer keys.
    pub fn index_k(src: usize) -> String {
        format!("carry.ik.{src}")
    }
    /// Additive `[1, ncomp]` mask published by the last index source.
    pub const TOPK: &str = "carry.topk";
    /// Additive `[1, ncomp]` candidate-block mask.
    pub const CANDIDATES: &str = "carry.cand";

    /// Tail-graph inputs.
    pub const H_ATTN: &str = "carry.h_attn";
    pub const POST: &str = "carry.post";
    pub const COMB: &str = "carry.comb";
    pub const X_FFN: &str = "carry.x_ffn";
}

/// Output names a head graph emits, in graph-output order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadOutputs {
    /// `h_attn`, `post`, `comb`, `pre_out`, `x_ffn`, then the step/carry names.
    pub names: Vec<String>,
}

/// What a layer needs supplied at its boundary, decided from the spec and the
/// step plan so the driver and the builder cannot disagree about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerCarry {
    /// Compressed positions this layer attends over; `0` when it has no
    /// compressed KV at all.
    pub ncomp: usize,
    /// The kv source this layer reads, when it is not that source itself.
    pub comp_from: Option<usize>,
    /// This layer runs a compressor, so it derives its own compressed KV.
    pub is_source: bool,
    /// This layer publishes the top-k mask rather than consuming one.
    pub publishes_topk: bool,
    /// This layer consumes a candidate-block mask published earlier.
    pub consumes_candidates: bool,
}

impl LayerCarry {
    pub fn of(spec: &DeepseekV41Spec, plan: &V41DecodePlan, il: usize) -> Self {
        let is_source = plan.for_source(il).is_some();
        let ncomp = if spec.ratio(il) > 0 {
            plan.for_source(spec.kv_source_for(il).unwrap_or(il))
                .map(|s| s.len_after)
                .unwrap_or(0)
        } else {
            0
        };
        let publishes_topk = spec.is_index_source(il) && spec.index_head_dim > 0 && ncomp > 0;
        LayerCarry {
            ncomp,
            comp_from: (!is_source && ncomp > 0)
                .then(|| spec.kv_source_for(il))
                .flatten(),
            is_source,
            publishes_topk,
            consumes_candidates: ncomp > 0 && spec.uses_candidates(il) && !publishes_topk,
        }
    }
}

/// Build one layer's **head**: everything up to the router's input.
///
/// The returned names line up with the graph's outputs. The first five are
/// always `h_attn`, `post`, `comb`, `pre_out`, `x_ffn`; after them come whatever
/// this layer publishes — the new window KV, any new compressed latent, and the
/// masks a later layer will need.
pub fn build_paged_head(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    il: usize,
    plan: &V41DecodePlan,
) -> Result<(Graph, HashMap<String, Vec<f32>>, HeadOutputs)> {
    let (d, hc) = (spec.dim, spec.hc_mult);
    let f = DType::F32;
    let carry_spec = LayerCarry::of(spec, plan, il);
    let mut ctx = Ctx::new(&format!("v41_paged_head_{il}"), spec, weights, packed, 1);

    let mut h = ctx.g.input(carry::H_IN, Shape::new(&[1, hc, d], f));
    let pre_in = ctx.g.input(carry::PRE_IN, Shape::new(&[1, hc], f));

    let lp = spec.layer_prefix(il);
    if let Some(e) = &spec.engram
        && e.layer_hash_index(il).is_some()
    {
        let cols = e.n_hash_cols();
        let row_ids = ctx.g.input(
            crate::dsv41_decode::names::engram_rows(il),
            Shape::new(&[1, cols], DType::F32),
        );
        h = crate::dsv41_engram::build_v41_engram(&mut ctx, &lp, h, row_ids, None, e)?;
    }

    // ── the CSA2 state this layer does not derive for itself ──
    let mut shared = crate::dsv41_decode::StepShared {
        csa: CsaShared::default(),
        len_after: carry_spec.ncomp,
    };
    if let Some(src) = carry_spec.comp_from {
        let n = carry_spec.ncomp;
        shared.csa.compress_kv = Some(
            ctx.g
                .input(carry::compressed(src), Shape::new(&[n, spec.head_dim], f)),
        );
        if spec.index_head_dim > 0 {
            shared.csa.index_k = Some(ctx.g.input(
                carry::index_k(src),
                Shape::new(&[n, spec.index_head_dim], f),
            ));
        }
    }
    if carry_spec.ncomp > 0 && !carry_spec.publishes_topk && spec.index_head_dim > 0 {
        shared.csa.topk_mask = Some(
            ctx.g
                .input(carry::TOPK, Shape::new(&[1, carry_spec.ncomp], f)),
        );
    }
    if carry_spec.consumes_candidates {
        shared.csa.candidates = Some(
            ctx.g
                .input(carry::CANDIDATES, Shape::new(&[1, carry_spec.ncomp], f)),
        );
    }

    let rope = crate::dsv41_decode::StepRope::build(&mut ctx, plan);
    let rope_l = rope.for_layer(spec.ratio(il));
    let mut step = crate::dsv41_decode::StepOutputs::default();
    let (h_attn, pre_attn) =
        crate::dsv41_hc::hc_sublayer_plain(&mut ctx, HcSide::Attn, &lp, h, pre_in, |ctx, xa| {
            crate::dsv41_decode::build_decode_attention(
                ctx,
                il,
                xa,
                plan,
                &rope_l,
                &mut shared,
                &mut step,
            )
        })?;

    // ── the FFN's first half; the body is the tail graph's job ──
    let split = hc_split(&mut ctx, HcSide::Ffn, &lp, h_attn, pre_attn)?;

    let mut outs = vec![h_attn, split.post, split.comb, split.pre_out, split.x];
    let mut names_out: Vec<String> = ["h_attn", "post", "comb", "pre_out", "x_ffn"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for (n, nm) in step.nodes.iter().zip(&step.names) {
        outs.push(*n);
        names_out.push(nm.clone());
    }
    if carry_spec.publishes_topk {
        if let Some(m) = shared.csa.topk_mask {
            outs.push(m);
            names_out.push(carry::TOPK.to_string());
        }
        if let Some(c) = shared.csa.candidates {
            outs.push(c);
            names_out.push(carry::CANDIDATES.to_string());
        }
    }
    if carry_spec.is_source {
        // the driver needs the full post-update cache to hand to later layers
        if let Some(c) = shared.csa.compress_kv {
            outs.push(c);
            names_out.push(carry::compressed(il));
        }
        if let Some(k) = shared.csa.index_k {
            outs.push(k);
            names_out.push(carry::index_k(il));
        }
    }
    let (g, params) = ctx.finish(outs);
    Ok((g, params, HeadOutputs { names: names_out }))
}

/// Build one layer's **tail**: the MoE over a gathered bank, then the
/// Hyper-Connection write-back.
///
/// The shape is fixed for decode — one query row selects exactly `top_k`
/// distinct experts — so this is compiled once per layer and reused for every
/// token of the run.
pub fn build_paged_tail(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    il: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (d, hc) = (spec.dim, spec.hc_mult);
    let f = DType::F32;
    let (_, top_k) = spec.moe_dims(il);
    let mut ctx = Ctx::new(&format!("v41_paged_tail_{il}"), spec, weights, packed, 1);

    let h_attn = ctx.g.input(carry::H_ATTN, Shape::new(&[1, hc, d], f));
    let post = ctx.g.input(carry::POST, Shape::new(&[1, hc], f));
    let comb = ctx.g.input(carry::COMB, Shape::new(&[1, hc, hc], f));
    let x_ffn = ctx.g.input(carry::X_FFN, Shape::new(&[1, d], f));

    let y = build_v41_moe_paged(&mut ctx, il, x_ffn, 1, top_k, top_k)?;
    let split = crate::dsv41_hc::HcSplit {
        x: x_ffn,
        pre_out: post,
        post,
        comb,
    };
    let h = hc_join(&mut ctx, y, h_attn, &split);
    let (g, params) = ctx.finish(vec![h]);
    Ok((g, params))
}

/// The final graph: collapse the Hyper-Connection copies, norm, and project to
/// the vocabulary.
pub fn build_paged_lm_head(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (d, hc) = (spec.dim, spec.hc_mult);
    let f = DType::F32;
    let mut ctx = Ctx::new("v41_paged_head_lm", spec, weights, packed, 1);
    let h = ctx.g.input(carry::H_IN, Shape::new(&[1, hc, d], f));
    let pre = ctx.g.input(carry::PRE_IN, Shape::new(&[1, hc], f));
    let x = crate::dsv41_hc::hc_reduce(&mut ctx.g, h, pre, 1, hc);
    let gain = ctx.norm("norm.weight")?;
    let zb = ctx.zero_bias("v41p.zb.d", d);
    let eps = ctx.eps();
    let x = ctx.g.rms_norm(x, gain, zb, eps);
    let logits = ctx.project("head.weight", x, 1, spec.vocab_size)?;
    let logits = ctx.g.reshape_(logits, vec![1, spec.vocab_size as i64]);
    let (g, params) = ctx.finish(vec![logits]);
    Ok((g, params))
}

/// The embedding, as its own graph so the driver can start a step.
pub fn build_paged_embed(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (d, hc) = (spec.dim, spec.hc_mult);
    let mut ctx = Ctx::new("v41_paged_embed", spec, weights, packed, 1);
    let input_ids = ctx.g.input("input_ids", Shape::new(&[1, 1], DType::I32));
    let (embed_w, _, _) = crate::standard_decoder::load_dense_dequant(
        &mut ctx.g,
        &mut ctx.params,
        ctx.weights,
        "embed.weight",
    )?;
    let h0 = ctx.g.gather_(embed_w, input_ids, 0);
    let h0 = ctx.g.reshape_(h0, vec![1, 1, d as i64]);
    let ones = ctx.konst("v41p.hc.ones", vec![1f32; hc], &[1, hc, 1]);
    let h = ctx.g.mul(h0, ones);
    let pre = crate::dsv41_hc::identity_pre_mix(&mut ctx.g, &mut ctx.params, 1, hc);
    let (g, params) = ctx.finish(vec![h, pre]);
    Ok((g, params))
}

/// Every `Op::Input` name a graph declares, so a driver can feed exactly what it
/// asks for.
///
/// The decode cache offers every layer's buffers at once; a per-layer graph
/// wants only its own, and handing a compiled graph an input it never declared
/// is at best ignored and at worst a shape error somewhere unrelated.
pub fn declared_inputs(g: &Graph) -> std::collections::HashSet<String> {
    g.nodes()
        .iter()
        .filter_map(|n| match &n.op {
            rlx_ir::op::Op::Input { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

/// One layer's compiled head, with the input names it declared.
struct Head {
    sess: rlx_runtime::CompiledGraph,
    outputs: HeadOutputs,
    /// Which of the decode cache's buffers this graph actually asked for.
    /// Handing a compiled graph an input it never declared is not an error the
    /// runtime reports, so the driver filters rather than trusting it.
    declared: std::collections::HashSet<String>,
}

/// Runs a decode step as embed → (head, route, page, tail) per layer → LM head.
///
/// Compiled graphs are kept across steps. The tails and the embedding and LM
/// head never change shape; the heads are keyed on the step shape, which after
/// the sliding window fills settles into a small repeating set.
pub struct PagedStepper {
    spec: DeepseekV41Spec,
    device: rlx_runtime::Device,
    opts: rlx_runtime::CompileOptions,
    embed: rlx_runtime::CompiledGraph,
    lm: rlx_runtime::CompiledGraph,
    tails: Vec<rlx_runtime::CompiledGraph>,
    heads: HashMap<(String, usize), Head>,
    /// `[layer] -> (gate weight [n_e, dim], bias [n_e], vl bias)`, for the host
    /// router. One gate is `n_routed_experts × dim` — megabytes, not gigabytes.
    gates: Vec<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)>,
    pager: std::sync::Arc<crate::dsv41_pager::ExpertPager>,
}

/// Host-side state that crosses a layer boundary within one step.
#[derive(Default)]
struct Carry {
    compressed: HashMap<usize, Vec<f32>>,
    index_k: HashMap<usize, Vec<f32>>,
    topk: Option<Vec<f32>>,
    candidates: Option<Vec<f32>>,
}

impl PagedStepper {
    /// Compile the shape-invariant graphs and load the routing gates.
    pub fn new(
        spec: &DeepseekV41Spec,
        weights: &mut dyn WeightLoader,
        pager: std::sync::Arc<crate::dsv41_pager::ExpertPager>,
        device: rlx_runtime::Device,
    ) -> Result<Self> {
        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            device,
        );
        let compile = |g: Graph, params: HashMap<String, Vec<f32>>| -> rlx_runtime::CompiledGraph {
            let mut s = rlx_runtime::Session::new(device).compile_with(g, &opts);
            for (n, d) in &params {
                s.set_param(n, d);
            }
            s
        };

        let mut packed = HashMap::new();
        let (g, p) = build_paged_embed(spec, weights, &mut packed)?;
        let embed = compile(g, p);
        let (g, p) = build_paged_lm_head(spec, weights, &mut packed)?;
        let lm = compile(g, p);

        let mut tails = Vec::with_capacity(spec.n_layers);
        let mut gates = Vec::with_capacity(spec.n_layers);
        for il in 0..spec.n_layers {
            let (g, p) = build_paged_tail(spec, weights, &mut packed, il)?;
            tails.push(compile(g, p));
            let lp = spec.layer_prefix(il);
            let (gw, _) = weights.take(&format!("{lp}.ffn.gate.weight"))?;
            let (b, _) = weights.take(&format!("{lp}.ffn.gate.bias"))?;
            let vl = spec
                .vision
                .is_some()
                .then(|| weights.take(&format!("{lp}.ffn.gate.bias_vl")).ok())
                .flatten()
                .map(|(v, _)| v);
            gates.push((gw, b, vl));
        }

        Ok(PagedStepper {
            spec: spec.clone(),
            device,
            opts,
            embed,
            lm,
            tails,
            heads: HashMap::new(),
            gates,
            pager,
        })
    }

    pub fn pager(&self) -> &crate::dsv41_pager::ExpertPager {
        &self.pager
    }

    /// One decode step. Returns `(logits, cache_output_names, cache_outputs)`,
    /// the latter two ready for [`crate::dsv41_decode::V41DecodeCache::apply`].
    pub fn step(
        &mut self,
        weights: &mut dyn WeightLoader,
        plan: &V41DecodePlan,
        token: u32,
        engram: &[(String, Vec<f32>)],
        cached: &[(String, &[f32])],
        shape_key: &str,
    ) -> Result<(Vec<f32>, Vec<String>, Vec<Vec<f32>>)> {
        let idf = [token as f32];
        let out = self.embed.run(&[("input_ids", idf.as_slice())]);
        let (mut h, mut pre) = (out[0].clone(), out[1].clone());

        let mut carry = Carry::default();
        let mut cache_names: Vec<String> = Vec::new();
        let mut cache_vals: Vec<Vec<f32>> = Vec::new();

        for il in 0..self.spec.n_layers {
            let spec_carry = LayerCarry::of(&self.spec, plan, il);
            let key = (shape_key.to_string(), il);
            if !self.heads.contains_key(&key) {
                let mut packed = HashMap::new();
                let (g, params, names) =
                    build_paged_head(&self.spec, weights, &mut packed, il, plan)?;
                let declared = declared_inputs(&g);
                let mut sess = rlx_runtime::Session::new(self.device).compile_with(g, &self.opts);
                for (n, d) in &params {
                    sess.set_param(n, d);
                }
                self.heads.insert(
                    key.clone(),
                    Head {
                        sess,
                        outputs: names,
                        declared,
                    },
                );
            }

            // feed: the carry, plus whatever of the decode cache this layer declared
            let (comp_name, ik_name);
            let mut feed: Vec<(&str, &[f32])> =
                vec![(carry::H_IN, h.as_slice()), (carry::PRE_IN, pre.as_slice())];
            if let Some(src) = spec_carry.comp_from {
                let c = carry.compressed.get(&src).ok_or_else(|| {
                    anyhow!("deepseek_v41 paged: layer {il} reads source {src} before it ran")
                })?;
                comp_name = carry::compressed(src);
                feed.push((comp_name.as_str(), c.as_slice()));
                if self.spec.index_head_dim > 0 {
                    let k = carry.index_k.get(&src).ok_or_else(|| {
                        anyhow!("deepseek_v41 paged: source {src} published no index keys")
                    })?;
                    ik_name = carry::index_k(src);
                    feed.push((ik_name.as_str(), k.as_slice()));
                }
            }
            if spec_carry.ncomp > 0
                && !spec_carry.publishes_topk
                && self.spec.index_head_dim > 0
                && let Some(t) = &carry.topk
            {
                feed.push((carry::TOPK, t.as_slice()));
            }
            if spec_carry.consumes_candidates
                && let Some(c) = &carry.candidates
            {
                feed.push((carry::CANDIDATES, c.as_slice()));
            }
            let head = self.heads.get_mut(&key).expect("inserted above");
            for (n, v) in engram {
                if head.declared.contains(n.as_str()) {
                    feed.push((n.as_str(), v.as_slice()));
                }
            }
            for (n, v) in cached {
                if head.declared.contains(n.as_str()) {
                    feed.push((n.as_str(), v));
                }
            }
            let outs = head.sess.run(&feed);
            let names = head.outputs.clone();

            let h_attn = outs[0].clone();
            let post = outs[1].clone();
            let comb = outs[2].clone();
            let pre_out = outs[3].clone();
            let x_ffn = outs[4].clone();
            for (i, nm) in names.names.iter().enumerate().skip(5) {
                if nm == carry::TOPK {
                    carry.topk = Some(outs[i].clone());
                } else if nm == carry::CANDIDATES {
                    carry.candidates = Some(outs[i].clone());
                } else if let Some(src) = nm.strip_prefix("carry.comp.") {
                    carry
                        .compressed
                        .insert(src.parse().unwrap_or(il), outs[i].clone());
                } else if let Some(src) = nm.strip_prefix("carry.ik.") {
                    carry
                        .index_k
                        .insert(src.parse().unwrap_or(il), outs[i].clone());
                } else {
                    cache_names.push(nm.clone());
                    cache_vals.push(outs[i].clone());
                }
            }

            // ── route on the host, page in only what it named ──
            let (n_e, top_k) = self.spec.moe_dims(il);
            let (gw, bias, vl) = &self.gates[il];
            let routing = crate::dsv41_moe::route_on_host(
                &self.spec,
                &x_ffn,
                1,
                gw,
                bias,
                vl.as_deref(),
                &[],
                top_k,
            )?;
            debug_assert!(top_k <= n_e);
            let bank = routing.distinct();
            let slots: Vec<f32> = routing.to_slots(&bank).iter().map(|&s| s as f32).collect();
            let w1 = self
                .pager
                .gather_bank(&self.spec, il, crate::dsv41_pager::Proj::W1, &bank)?;
            let w3 = self
                .pager
                .gather_bank(&self.spec, il, crate::dsv41_pager::Proj::W3, &bank)?;
            let w2 = self
                .pager
                .gather_bank(&self.spec, il, crate::dsv41_pager::Proj::W2, &bank)?;

            let b1 = paged_names::bank(il, "w1");
            let b3 = paged_names::bank(il, "w3");
            let b2 = paged_names::bank(il, "w2");
            let sl = paged_names::slots(il);
            let wn = paged_names::weights(il);
            let tail_out = self.tails[il].run(&[
                (carry::H_ATTN, h_attn.as_slice()),
                (carry::POST, post.as_slice()),
                (carry::COMB, comb.as_slice()),
                (carry::X_FFN, x_ffn.as_slice()),
                (b1.as_str(), w1.as_slice()),
                (b3.as_str(), w3.as_slice()),
                (b2.as_str(), w2.as_slice()),
                (sl.as_str(), slots.as_slice()),
                (wn.as_str(), routing.w.as_slice()),
            ]);
            h = tail_out[0].clone();
            pre = pre_out;
        }

        let logits = self
            .lm
            .run(&[(carry::H_IN, h.as_slice()), (carry::PRE_IN, pre.as_slice())])[0]
            .clone();
        Ok((logits, cache_names, cache_vals))
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Paged prefill
// ──────────────────────────────────────────────────────────────────────────

/// Prefill-side names for the state a layer boundary carries.
///
/// Separate from [`carry`] because the shapes differ: prefill carries `seq` query
/// rows, so its masks are `[seq, ncomp]` rather than `[1, ncomp]`.
pub mod pcarry {
    pub const H_IN: &str = "pcarry.h";
    pub const PRE_IN: &str = "pcarry.pre";
    pub fn compressed(src: usize) -> String {
        format!("pcarry.comp.{src}")
    }
    pub fn index_k(src: usize) -> String {
        format!("pcarry.ik.{src}")
    }
    pub const TOPK: &str = "pcarry.topk";
    pub const CANDIDATES: &str = "pcarry.cand";
    pub const H_ATTN: &str = "pcarry.h_attn";
    pub const POST: &str = "pcarry.post";
    pub const COMB: &str = "pcarry.comb";
    pub const X_FFN: &str = "pcarry.x_ffn";
}

/// What a prefill layer needs supplied at its boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillCarry {
    pub ncomp: usize,
    pub comp_from: Option<usize>,
    pub is_source: bool,
    pub publishes_topk: bool,
    pub consumes_candidates: bool,
}

impl PrefillCarry {
    pub fn of(spec: &DeepseekV41Spec, seq: usize, il: usize) -> Self {
        let ratio = spec.ratio(il);
        let is_source = spec.is_kv_source(il) && ratio > 0;
        let ncomp = seq.checked_div(ratio).unwrap_or(0);
        let publishes_topk = spec.is_index_source(il) && spec.index_head_dim > 0 && ncomp > 0;
        PrefillCarry {
            ncomp,
            comp_from: (!is_source && ncomp > 0)
                .then(|| spec.kv_source_for(il))
                .flatten(),
            is_source,
            publishes_topk,
            consumes_candidates: ncomp > 0 && spec.uses_candidates(il) && !publishes_topk,
        }
    }
}

/// One prefill layer's **head**: everything up to the router's input, for all
/// `seq` rows at once.
#[allow(clippy::too_many_arguments)]
pub fn build_paged_prefill_head(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    il: usize,
    seq: usize,
    engram_rows: &[i64],
    emit_cache: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, HeadOutputs)> {
    use crate::dsv41_graph::{PrefillCache, SharedAttn, StageRope, build_v41_attention};

    let (d, hc) = (spec.dim, spec.hc_mult);
    let f = DType::F32;
    let cs = PrefillCarry::of(spec, seq, il);
    let mut ctx = Ctx::new(&format!("v41_paged_pre_{il}"), spec, weights, packed, seq);

    let mut h = ctx.g.input(pcarry::H_IN, Shape::new(&[seq, hc, d], f));
    let pre_in = ctx.g.input(pcarry::PRE_IN, Shape::new(&[seq, hc], f));

    let lp = spec.layer_prefix(il);
    if let Some(e) = &spec.engram
        && let Some(hash_idx) = e.layer_hash_index(il)
    {
        let cols = e.n_hash_cols();
        let n_eng = e.layer_ids.len();
        let want = seq * n_eng * cols;
        if engram_rows.len() != want {
            return Err(anyhow!(
                "deepseek_v41 paged prefill: engram_rows has {} entries, expected {want}",
                engram_rows.len()
            ));
        }
        let slice: Vec<f32> = (0..seq)
            .flat_map(|i| {
                let base = (i * n_eng + hash_idx) * cols;
                engram_rows[base..base + cols].iter().map(|&v| v as f32)
            })
            .collect();
        let row_ids = ctx.konst(&format!("{lp}.engram.rows"), slice, &[seq, cols]);
        h = crate::dsv41_engram::build_v41_engram(&mut ctx, &lp, h, row_ids, None, e)?;
    }

    let mut shared = SharedAttn {
        ncomp: cs.ncomp,
        ..Default::default()
    };
    if let Some(src) = cs.comp_from {
        let n = cs.ncomp;
        shared.csa.compress_kv = Some(
            ctx.g
                .input(pcarry::compressed(src), Shape::new(&[n, spec.head_dim], f)),
        );
        if spec.index_head_dim > 0 {
            shared.csa.index_k = Some(ctx.g.input(
                pcarry::index_k(src),
                Shape::new(&[n, spec.index_head_dim], f),
            ));
        }
    }
    if cs.ncomp > 0 && !cs.publishes_topk && spec.index_head_dim > 0 {
        shared.csa.topk_mask = Some(ctx.g.input(pcarry::TOPK, Shape::new(&[seq, cs.ncomp], f)));
    }
    if cs.consumes_candidates {
        shared.csa.candidates = Some(
            ctx.g
                .input(pcarry::CANDIDATES, Shape::new(&[seq, cs.ncomp], f)),
        );
    }

    let rope = StageRope::build(&mut ctx, seq, &(il..il + 1));
    let rope_l = rope.for_layer(spec.ratio(il));
    let win_mask = sliding_window_mask(&mut ctx, seq, spec.window_size, "v41p.mask.win");
    let mut tap = Tap::from_env();
    let mut cache = PrefillCache::default();

    let (h_attn, pre_attn) =
        crate::dsv41_hc::hc_sublayer_plain(&mut ctx, HcSide::Attn, &lp, h, pre_in, |ctx, xa| {
            match build_v41_attention(
                ctx,
                &mut tap,
                il,
                xa,
                seq,
                &rope_l,
                win_mask,
                &mut shared,
                &mut cache,
            )? {
                crate::dsv41_hc::BodyOut::Out(y) => Ok(y),
                crate::dsv41_hc::BodyOut::Tapped(y) => Ok(y),
            }
        })?;

    let split = hc_split(&mut ctx, HcSide::Ffn, &lp, h_attn, pre_attn)?;
    let mut outs = vec![h_attn, split.post, split.comb, split.pre_out, split.x];
    let mut names_out: Vec<String> = ["h_attn", "post", "comb", "pre_out", "x_ffn"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    if cs.publishes_topk {
        if let Some(m) = shared.csa.topk_mask {
            outs.push(m);
            names_out.push(pcarry::TOPK.to_string());
        }
        if let Some(c) = shared.csa.candidates {
            outs.push(c);
            names_out.push(pcarry::CANDIDATES.to_string());
        }
    }
    if cs.is_source {
        if let Some(c) = shared.csa.compress_kv {
            outs.push(c);
            names_out.push(pcarry::compressed(il));
        }
        if let Some(k) = shared.csa.index_k {
            outs.push(k);
            names_out.push(pcarry::index_k(il));
        }
    }
    if emit_cache {
        for (n, nm) in cache.nodes.iter().zip(&cache.names) {
            outs.push(*n);
            names_out.push(nm.clone());
        }
    }
    let (g, params) = ctx.finish(outs);
    Ok((g, params, HeadOutputs { names: names_out }))
}

/// One prefill layer's **tail**: the MoE over a gathered bank of `n_sel`
/// experts, then the Hyper-Connection write-back.
///
/// Unlike decode, `n_sel` is not fixed: `seq` rows each pick `top_k` experts, so
/// the union is anywhere from `top_k` to `n_routed_experts`. The graph is
/// therefore compiled per `(layer, seq, n_sel)` and cached, which in practice is
/// a handful of shapes.
pub fn build_paged_prefill_tail(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    il: usize,
    seq: usize,
    n_sel: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (d, hc) = (spec.dim, spec.hc_mult);
    let f = DType::F32;
    let (_, top_k) = spec.moe_dims(il);
    let mut ctx = Ctx::new(
        &format!("v41_paged_pretail_{il}"),
        spec,
        weights,
        packed,
        seq,
    );

    let h_attn = ctx.g.input(pcarry::H_ATTN, Shape::new(&[seq, hc, d], f));
    let post = ctx.g.input(pcarry::POST, Shape::new(&[seq, hc], f));
    let comb = ctx.g.input(pcarry::COMB, Shape::new(&[seq, hc, hc], f));
    let x_ffn = ctx.g.input(pcarry::X_FFN, Shape::new(&[seq, d], f));

    let y = build_v41_moe_paged(&mut ctx, il, x_ffn, seq, n_sel, top_k)?;
    let split = crate::dsv41_hc::HcSplit {
        x: x_ffn,
        pre_out: post,
        post,
        comb,
    };
    let h = hc_join(&mut ctx, y, h_attn, &split);
    let (g, params) = ctx.finish(vec![h]);
    Ok((g, params))
}

/// The prefill LM head, over all `seq` rows.
pub fn build_paged_prefill_lm(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (d, hc) = (spec.dim, spec.hc_mult);
    let f = DType::F32;
    let mut ctx = Ctx::new("v41_paged_pre_lm", spec, weights, packed, seq);
    let h = ctx.g.input(pcarry::H_IN, Shape::new(&[seq, hc, d], f));
    let pre = ctx.g.input(pcarry::PRE_IN, Shape::new(&[seq, hc], f));
    let x = crate::dsv41_hc::hc_reduce(&mut ctx.g, h, pre, seq, hc);
    let gain = ctx.norm("norm.weight")?;
    let zb = ctx.zero_bias("v41pp.zb.d", d);
    let eps = ctx.eps();
    let x = ctx.g.rms_norm(x, gain, zb, eps);
    let logits = ctx.project("head.weight", x, seq, spec.vocab_size)?;
    let logits = ctx
        .g
        .reshape_(logits, vec![seq as i64, spec.vocab_size as i64]);
    let (g, params) = ctx.finish(vec![logits]);
    Ok((g, params))
}

/// The prefill embedding, over all `seq` rows.
pub fn build_paged_prefill_embed(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (d, hc) = (spec.dim, spec.hc_mult);
    let mut ctx = Ctx::new("v41_paged_pre_embed", spec, weights, packed, seq);
    let input_ids = ctx.g.input("input_ids", Shape::new(&[1, seq], DType::I32));
    let (embed_w, _, _) = crate::standard_decoder::load_dense_dequant(
        &mut ctx.g,
        &mut ctx.params,
        ctx.weights,
        "embed.weight",
    )?;
    let h0 = ctx.g.gather_(embed_w, input_ids, 0);
    let h0 = ctx.g.reshape_(h0, vec![seq as i64, 1, d as i64]);
    let ones = ctx.konst("v41pp.hc.ones", vec![1f32; hc], &[1, hc, 1]);
    let h = ctx.g.mul(h0, ones);
    let pre = crate::dsv41_hc::identity_pre_mix(&mut ctx.g, &mut ctx.params, seq, hc);
    let (g, params) = ctx.finish(vec![h, pre]);
    Ok((g, params))
}

/// Runs a whole prompt through the paged path: embed → (head, route, page,
/// tail) per layer → LM head, with all `seq` rows at once.
///
/// The difference from [`PagedStepper`] is what "the experts a token needs"
/// means. A decode step has one row and picks exactly `top_k`; a prompt of `seq`
/// rows picks the *union* over its rows, which is between `top_k` and
/// `n_routed_experts`. The tail graph is therefore keyed on that union's size
/// rather than fixed.
///
/// That also bounds the benefit: the union grows with the prompt, so a long
/// enough prompt gathers most of the bank and paging saves little on the *prompt*
/// pass. Decode is unaffected — its union stays at `top_k` however long the
/// context gets. Bounding the prompt case would mean processing it in chunks,
/// which needs the prefill graph to let a chunk's queries attend to keys from
/// before it; it currently assumes queries `0..seq` against keys `0..seq`.
pub struct PagedPrefill {
    spec: DeepseekV41Spec,
    device: rlx_runtime::Device,
    opts: rlx_runtime::CompileOptions,
    /// `(layer, seq, n_sel) -> tail`. Heads are not cached: they carry the
    /// prompt's Engram rows as constants, so a cached one would answer with
    /// another prompt's n-grams.
    tails: HashMap<(usize, usize, usize), rlx_runtime::CompiledGraph>,
    gates: Vec<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)>,
    pager: std::sync::Arc<crate::dsv41_pager::ExpertPager>,
}

/// One prompt's result: last-row logits, plus the decode cache it leaves behind.
pub struct PrefillOut {
    pub logits: Vec<f32>,
    pub cache_names: Vec<String>,
    pub cache_values: Vec<Vec<f32>>,
}

impl PagedPrefill {
    pub fn new(
        spec: &DeepseekV41Spec,
        weights: &mut dyn WeightLoader,
        pager: std::sync::Arc<crate::dsv41_pager::ExpertPager>,
        device: rlx_runtime::Device,
    ) -> Result<Self> {
        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            device,
        );
        let mut gates = Vec::with_capacity(spec.n_layers);
        for il in 0..spec.n_layers {
            let lp = spec.layer_prefix(il);
            let (gw, _) = weights.take(&format!("{lp}.ffn.gate.weight"))?;
            let (b, _) = weights.take(&format!("{lp}.ffn.gate.bias"))?;
            let vl = spec
                .vision
                .is_some()
                .then(|| weights.take(&format!("{lp}.ffn.gate.bias_vl")).ok())
                .flatten()
                .map(|(v, _)| v);
            gates.push((gw, b, vl));
        }
        Ok(PagedPrefill {
            spec: spec.clone(),
            device,
            opts,
            tails: HashMap::new(),
            gates,
            pager,
        })
    }

    fn compile(&self, g: Graph, params: HashMap<String, Vec<f32>>) -> rlx_runtime::CompiledGraph {
        let mut s = rlx_runtime::Session::new(self.device).compile_with(g, &self.opts);
        for (n, d) in &params {
            s.set_param(n, d);
        }
        s
    }

    /// Run `ids` in one pass.
    pub fn run(
        &mut self,
        weights: &mut dyn WeightLoader,
        ids: &[u32],
        engram_rows: &[i64],
        emit_cache: bool,
    ) -> Result<PrefillOut> {
        let seq = ids.len();
        if seq == 0 {
            return Err(anyhow!("deepseek_v41: cannot prefill an empty prompt"));
        }
        let mut packed = HashMap::new();
        let (g, p) = build_paged_prefill_embed(&self.spec, weights, &mut packed, seq)?;
        let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
        let out = self.compile(g, p).run(&[("input_ids", idf.as_slice())]);
        let (mut h, mut pre) = (out[0].clone(), out[1].clone());

        let mut carry = Carry::default();
        let mut cache_names: Vec<String> = Vec::new();
        let mut cache_vals: Vec<Vec<f32>> = Vec::new();

        for il in 0..self.spec.n_layers {
            let cs = PrefillCarry::of(&self.spec, seq, il);
            let mut packed = HashMap::new();
            let (g, params, names) = build_paged_prefill_head(
                &self.spec,
                weights,
                &mut packed,
                il,
                seq,
                engram_rows,
                emit_cache,
            )?;
            let declared = declared_inputs(&g);
            let mut sess = self.compile(g, params);

            let (comp_name, ik_name);
            let mut feed: Vec<(&str, &[f32])> = vec![
                (pcarry::H_IN, h.as_slice()),
                (pcarry::PRE_IN, pre.as_slice()),
            ];
            if let Some(src) = cs.comp_from {
                let c = carry.compressed.get(&src).ok_or_else(|| {
                    anyhow!("deepseek_v41 paged prefill: layer {il} reads source {src} first")
                })?;
                comp_name = pcarry::compressed(src);
                feed.push((comp_name.as_str(), c.as_slice()));
                if self.spec.index_head_dim > 0 {
                    let k = carry.index_k.get(&src).ok_or_else(|| {
                        anyhow!("deepseek_v41 paged prefill: source {src} has no index keys")
                    })?;
                    ik_name = pcarry::index_k(src);
                    feed.push((ik_name.as_str(), k.as_slice()));
                }
            }
            if declared.contains(pcarry::TOPK)
                && let Some(t) = &carry.topk
            {
                feed.push((pcarry::TOPK, t.as_slice()));
            }
            if declared.contains(pcarry::CANDIDATES)
                && let Some(c) = &carry.candidates
            {
                feed.push((pcarry::CANDIDATES, c.as_slice()));
            }
            let outs = sess.run(&feed);

            let h_attn = outs[0].clone();
            let post = outs[1].clone();
            let comb = outs[2].clone();
            let pre_out = outs[3].clone();
            let x_ffn = outs[4].clone();
            for (i, nm) in names.names.iter().enumerate().skip(5) {
                if nm == pcarry::TOPK {
                    carry.topk = Some(outs[i].clone());
                } else if nm == pcarry::CANDIDATES {
                    carry.candidates = Some(outs[i].clone());
                } else if let Some(src) = nm.strip_prefix("pcarry.comp.") {
                    carry
                        .compressed
                        .insert(src.parse().unwrap_or(il), outs[i].clone());
                } else if let Some(src) = nm.strip_prefix("pcarry.ik.") {
                    carry
                        .index_k
                        .insert(src.parse().unwrap_or(il), outs[i].clone());
                } else {
                    cache_names.push(nm.clone());
                    cache_vals.push(outs[i].clone());
                }
            }

            let (_, top_k) = self.spec.moe_dims(il);
            let (gw, bias, vl) = &self.gates[il];
            let routing = crate::dsv41_moe::route_on_host(
                &self.spec,
                &x_ffn,
                seq,
                gw,
                bias,
                vl.as_deref(),
                &[],
                top_k,
            )?;
            let bank = routing.distinct();
            let slots: Vec<f32> = routing.to_slots(&bank).iter().map(|&s| s as f32).collect();
            let w1 = self
                .pager
                .gather_bank(&self.spec, il, crate::dsv41_pager::Proj::W1, &bank)?;
            let w3 = self
                .pager
                .gather_bank(&self.spec, il, crate::dsv41_pager::Proj::W3, &bank)?;
            let w2 = self
                .pager
                .gather_bank(&self.spec, il, crate::dsv41_pager::Proj::W2, &bank)?;

            let key = (il, seq, bank.len());
            if !self.tails.contains_key(&key) {
                let mut packed = HashMap::new();
                let (g, p) = build_paged_prefill_tail(
                    &self.spec,
                    weights,
                    &mut packed,
                    il,
                    seq,
                    bank.len(),
                )?;
                let t = self.compile(g, p);
                self.tails.insert(key, t);
            }
            let (b1, b3, b2) = (
                paged_names::bank(il, "w1"),
                paged_names::bank(il, "w3"),
                paged_names::bank(il, "w2"),
            );
            let (sl, wn) = (paged_names::slots(il), paged_names::weights(il));
            let tail = self.tails.get_mut(&key).expect("inserted above");
            let tail_out = tail.run(&[
                (pcarry::H_ATTN, h_attn.as_slice()),
                (pcarry::POST, post.as_slice()),
                (pcarry::COMB, comb.as_slice()),
                (pcarry::X_FFN, x_ffn.as_slice()),
                (b1.as_str(), w1.as_slice()),
                (b3.as_str(), w3.as_slice()),
                (b2.as_str(), w2.as_slice()),
                (sl.as_str(), slots.as_slice()),
                (wn.as_str(), routing.w.as_slice()),
            ]);
            h = tail_out[0].clone();
            pre = pre_out;
        }

        let mut packed = HashMap::new();
        let (g, p) = build_paged_prefill_lm(&self.spec, weights, &mut packed, seq)?;
        let logits = self.compile(g, p).run(&[
            (pcarry::H_IN, h.as_slice()),
            (pcarry::PRE_IN, pre.as_slice()),
        ])[0]
            .clone();
        let v = self.spec.vocab_size;
        Ok(PrefillOut {
            logits: logits[(seq - 1) * v..].to_vec(),
            cache_names,
            cache_values: cache_vals,
        })
    }
}
