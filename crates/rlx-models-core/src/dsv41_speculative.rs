// RLX — versatile ML compiler + runtime. GPLv3.
//! **Speculative decoding** with DeepSeek-V4.1's own DSpark draft head.
//!
//! The backbone is expensive per token and nearly as cheap for `k` tokens as for
//! one — the cost is dominated by streaming the weights, not by the arithmetic.
//! DSpark exploits that: a small draft head proposes `dspark_block_size` tokens
//! from the backbone's own hidden states, and one backbone pass over the whole
//! block says how many of them the backbone would have produced anyway.
//!
//! Greedy speculation is **exact**, not approximate: every accepted token is one
//! the backbone itself would have chosen, and the first disagreement is resolved
//! in the backbone's favour. So the output is identical to plain greedy decoding
//! — which is the property [`SpeculativeDecoder`] is tested against, since a
//! speedup that changes the answer is not a speedup.
//!
//! Verification uses [`crate::dsv41_chunk`], which is the only way to run the
//! backbone over a block of positions against an existing cache.
//!
//! Rejection costs a second backbone pass over the accepted prefix, because the
//! cache updates from a rejected block cannot be kept. That is the right trade:
//! when drafting works the round costs one pass for `k` tokens, and when it
//! fails there was no saving to protect anyway.

use crate::dsv41::DeepseekV41Spec;
use crate::dsv41_chunk::build_deepseek_v41_chunk;
use crate::dsv41_decode::{V41ChunkPlan, V41DecodeCache};
use crate::dsv41_dspark::{
    build_v41_dspark_markov_step, build_v41_dspark_seed, build_v41_dspark_step, dspark_draft_ids,
    names as dnames,
};
use crate::dsv41_graph::V41Inputs;
use crate::weight_loader::WeightLoader;
use anyhow::{Context, Result, bail};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};
use std::collections::HashMap;

/// What one speculative round did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Round {
    /// Tokens the backbone committed to this round, in order. Always at least
    /// one — the backbone's own next token.
    pub tokens: Vec<u32>,
    /// How many of the draft head's proposals survived verification.
    pub accepted: usize,
    /// How many it proposed.
    pub proposed: usize,
}

/// Cumulative acceptance, which is what a speedup claim rests on.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeStats {
    pub rounds: usize,
    pub proposed: usize,
    pub accepted: usize,
    /// Backbone passes actually run, verification and re-commits included.
    pub backbone_passes: usize,
    pub tokens: usize,
}

impl SpeculativeStats {
    /// Fraction of drafted tokens the backbone agreed with.
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }

    /// Tokens produced per backbone pass. Above 1 is a win; at or below 1 the
    /// draft head is costing more than it saves.
    pub fn tokens_per_pass(&self) -> f64 {
        if self.backbone_passes == 0 {
            0.0
        } else {
            self.tokens as f64 / self.backbone_passes as f64
        }
    }
}

/// Drives DSpark drafting against the backbone.
pub struct SpeculativeDecoder {
    spec: DeepseekV41Spec,
    device: Device,
    opts: CompileOptions,
    /// Per stage, `window_size · head_dim` — the main model's stream, which is
    /// what the draft stages attend over.
    rings: Vec<Vec<f32>>,
    markov: Option<CompiledGraph>,
    stats: SpeculativeStats,
}

impl SpeculativeDecoder {
    pub fn new(spec: &DeepseekV41Spec, device: Device) -> Result<Self> {
        if spec.n_mtp_layers == 0 {
            bail!("deepseek_v41: this checkpoint has no DSpark draft stages");
        }
        if spec.dspark_block_size == 0 {
            bail!("deepseek_v41: dspark_block_size is 0, so there is nothing to draft");
        }
        Ok(SpeculativeDecoder {
            spec: spec.clone(),
            device,
            opts: crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
                &rlx_flow::CompileProfile::qwen3_prefill(),
                device,
            ),
            rings: vec![Vec::new(); spec.n_mtp_layers],
            markov: None,
            stats: SpeculativeStats::default(),
        })
    }

    pub fn stats(&self) -> SpeculativeStats {
        self.stats
    }

    fn compile(&self, g: rlx_ir::graph::Graph, p: &HashMap<String, Vec<f32>>) -> CompiledGraph {
        let mut s = Session::new(self.device).compile_with(g, &self.opts);
        for (n, d) in p {
            s.set_param(n, d);
        }
        s
    }

    /// Fill each stage's ring from the prompt's `main_hidden`.
    ///
    /// The draft stages attend over the *main* model's stream, so their caches
    /// are seeded from the backbone rather than from anything they produced.
    pub fn seed(
        &mut self,
        weights: &mut dyn WeightLoader,
        main_hidden: &[f32],
        seq: usize,
    ) -> Result<()> {
        let mut packed = HashMap::new();
        let (g, params, names) = build_v41_dspark_seed(&self.spec, weights, seq, &mut packed)?;
        let want = seq * self.spec.dim * self.spec.dspark_target_layer_ids.len();
        if main_hidden.len() != want {
            bail!(
                "deepseek_v41 dspark: main_hidden is {} long, expected {want}",
                main_hidden.len()
            );
        }
        let out = self
            .compile(g, &params)
            .run(&[("main_hidden", main_hidden)]);
        if names.len() != self.spec.n_mtp_layers {
            bail!(
                "deepseek_v41 dspark: seed produced {} rings for {} stages",
                names.len(),
                self.spec.n_mtp_layers
            );
        }
        self.rings = out;
        Ok(())
    }

    /// Propose `dspark_block_size` tokens after `accepted`, given the backbone's
    /// `main_hidden` at position `pos`.
    ///
    /// The Markov bias is sequential by construction — slot `i`'s bias depends on
    /// slot `i-1`'s sampled token — so the block is sampled one slot at a time
    /// even though the draft logits come out together.
    fn draft(
        &mut self,
        weights: &mut dyn WeightLoader,
        pos: usize,
        main_hidden: &[f32],
        accepted: u32,
    ) -> Result<Vec<u32>> {
        let hd = self.spec.head_dim;
        let block = self.spec.dspark_block_size;
        let vocab = self.spec.vocab_size;
        let cache_len = pos.min(self.spec.window_size.saturating_sub(1));

        let mut packed = HashMap::new();
        let (g, params, _) =
            build_v41_dspark_step(&self.spec, weights, pos, cache_len, &mut packed)?;
        let mut sess = self.compile(g, &params);

        let ids: Vec<f32> = dspark_draft_ids(&self.spec, accepted as i32)
            .iter()
            .map(|&i| i as f32)
            .collect();
        // this step overwrites the oldest ring slot, so it is fed the newest
        // `cache_len` entries and contributes the main token itself
        let tails: Vec<Vec<f32>> = self
            .rings
            .iter()
            .map(|r| r[r.len().saturating_sub(cache_len * hd)..].to_vec())
            .collect();
        let owned: Vec<String> = (0..self.spec.n_mtp_layers).map(dnames::window_kv).collect();
        let mut feed: Vec<(&str, &[f32])> =
            vec![("main_hidden", main_hidden), ("draft_ids", ids.as_slice())];
        for (s, t) in tails.iter().enumerate() {
            feed.push((owned[s].as_str(), t.as_slice()));
        }
        let out = sess.run(&feed);
        let (logits, hidden) = (out[0].clone(), out[1].clone());
        // each stage's new main-model latent joins its ring
        for (s, ring) in self.rings.iter_mut().enumerate() {
            if let Some(v) = out.get(2 + s) {
                ring.extend_from_slice(v);
                let cap = self.spec.window_size * hd;
                if ring.len() > cap {
                    let drop = ring.len() - cap;
                    ring.drain(..drop);
                }
            }
        }

        if self.markov.is_none() {
            let (g, p) = build_v41_dspark_markov_step(&self.spec, weights, 1)?;
            self.markov = Some(self.compile(g, &p));
        }
        let markov = self.markov.as_mut().expect("built above");
        let mut toks = Vec::with_capacity(block);
        let mut prev = accepted;
        for i in 0..block {
            let pt = [prev as f32];
            let row = &hidden[i * self.spec.dim..(i + 1) * self.spec.dim];
            let mk = markov.run(&[("token_ids", pt.as_slice()), ("hidden", row)]);
            let base = &logits[i * vocab..(i + 1) * vocab];
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (t, (a, b)) in base.iter().zip(&mk[0]).enumerate() {
                let v = a + b;
                if v > best_v {
                    best_v = v;
                    best = t;
                }
            }
            prev = best as u32;
            toks.push(prev);
        }
        Ok(toks)
    }

    /// Run the backbone over `tokens` at `start`.
    ///
    /// The cache updates come back rather than being folded in, because a
    /// verification pass does not know yet whether its tokens will be kept.
    fn backbone(
        &mut self,
        weights: &mut dyn WeightLoader,
        start: usize,
        tokens: &[u32],
        engram_rows: Vec<i64>,
        cache: &V41DecodeCache,
    ) -> Result<BackbonePass> {
        let plan = V41ChunkPlan::new(&self.spec, start, tokens.len());
        let inputs = V41Inputs {
            engram_rows,
            emit_main_hidden: true,
            ..Default::default()
        };
        let mut packed = HashMap::new();
        let (g, params, names) =
            build_deepseek_v41_chunk(&self.spec, weights, &plan, &inputs, &mut packed)
                .with_context(|| format!("backbone chunk at {start}+{}", tokens.len()))?;
        let mut sess = self.compile(g, &params);
        let idf: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        let cached = cache.chunk_inputs(&plan);
        let mut feed: Vec<(&str, &[f32])> = vec![("input_ids", idf.as_slice())];
        for (n, v) in &cached {
            feed.push((n.as_str(), *v));
        }
        let out = sess.run(&feed);
        drop(cached);
        self.stats.backbone_passes += 1;
        Ok(BackbonePass {
            plan,
            logits: out[0].clone(),
            main_hidden: out[1].clone(),
            names,
            outputs: out,
        })
    }

    /// One speculative round.
    ///
    /// `state` carries what the previous round left: the position of the last
    /// committed token, its id, the backbone's hidden state there, and the logits
    /// predicting the next position.
    ///
    /// `engram` computes the n-gram rows for a chunk given the history before it,
    /// which the caller owns because it depends on the tokenizer.
    pub fn round(
        &mut self,
        weights: &mut dyn WeightLoader,
        state: &mut SpeculativeState,
        history: &mut Vec<u32>,
        cache: &mut V41DecodeCache,
        engram: &dyn Fn(&[u32], &[u32]) -> Vec<i64>,
    ) -> Result<Round> {
        self.round_with(weights, state, history, cache, engram, None)
    }

    /// [`Self::round`] with the draft head's proposals supplied.
    ///
    /// On a synthetic checkpoint the draft head is random, so it agrees with the
    /// backbone essentially never — which leaves the multi-token commit path,
    /// the whole point of this module, never executed. Supplying the drafts is
    /// how that path gets tested: hand it the backbone's own continuation to
    /// exercise full acceptance, or a prefix of it to exercise partial.
    #[doc(hidden)]
    pub fn round_with(
        &mut self,
        weights: &mut dyn WeightLoader,
        state: &mut SpeculativeState,
        history: &mut Vec<u32>,
        cache: &mut V41DecodeCache,
        engram: &dyn Fn(&[u32], &[u32]) -> Vec<i64>,
        forced_drafts: Option<&[u32]>,
    ) -> Result<Round> {
        let vocab = self.spec.vocab_size;
        // the backbone's own next token — the ground truth this round is scored
        // against, and the fallback when the draft head is wrong
        let g0 = greedy(&state.next_logits);
        let drafts = match forced_drafts {
            Some(d) => {
                // still run the draft head, so its ring stays in step
                let _ = self.draft(weights, state.pos, &state.main_hidden, state.token)?;
                d.to_vec()
            }
            None => self.draft(weights, state.pos, &state.main_hidden, state.token)?,
        };
        self.stats.rounds += 1;
        self.stats.proposed += drafts.len();

        // A draft that disagrees on its very first token buys nothing, so the
        // block is abandoned and the backbone's token committed on its own.
        let candidates: Vec<u32> = if drafts.first() == Some(&g0) {
            drafts.clone()
        } else {
            vec![g0]
        };

        let start = state.pos + 1;
        let rows = engram(history, &candidates);
        let pass = self.backbone(weights, start, &candidates, rows, cache)?;

        // d1 is confirmed by `g0`; each later draft is confirmed by the row
        // before it, which predicts its position.
        let mut m = 1usize;
        if candidates.len() > 1 {
            while m < candidates.len() {
                let row = &pass.logits[(m - 1) * vocab..m * vocab];
                if greedy(row) != candidates[m] {
                    break;
                }
                m += 1;
            }
        }
        let accepted_drafts = if drafts.first() == Some(&g0) { m } else { 0 };
        self.stats.accepted += accepted_drafts;

        // Keep the verification pass when every candidate survived; otherwise its
        // cache updates cover tokens that are not being committed, so the prefix
        // is re-run.
        let committed = &candidates[..m];
        let final_pass = if m == candidates.len() {
            pass
        } else {
            let rows = engram(history, committed);

            self.backbone(weights, start, committed, rows, cache)?
        };
        cache.apply_chunk(&final_pass.plan, &final_pass.names, &final_pass.outputs)?;

        history.extend_from_slice(committed);
        let d = self.spec.dim * self.spec.dspark_target_layer_ids.len();
        state.pos = start + m - 1;
        state.token = committed[m - 1];
        state.next_logits = final_pass.logits[(m - 1) * vocab..m * vocab].to_vec();
        state.main_hidden = final_pass.main_hidden[(m - 1) * d..m * d].to_vec();
        self.stats.tokens += m;
        Ok(Round {
            tokens: committed.to_vec(),
            accepted: accepted_drafts,
            proposed: drafts.len(),
        })
    }
}

/// What a backbone pass produced, before anything is committed.
struct BackbonePass {
    plan: V41ChunkPlan,
    logits: Vec<f32>,
    main_hidden: Vec<f32>,
    names: Vec<String>,
    outputs: Vec<Vec<f32>>,
}

/// Where a speculative run has got to.
#[derive(Debug, Clone)]
pub struct SpeculativeState {
    /// Position of the last committed token.
    pub pos: usize,
    /// Its id.
    pub token: u32,
    /// The backbone's `main_hidden` at `pos`, which the draft head conditions on.
    pub main_hidden: Vec<f32>,
    /// Logits predicting `pos + 1`.
    pub next_logits: Vec<f32>,
}

/// Greedy argmax, ties to the lowest id — the same rule the runner's sampler
/// uses, so speculation and plain decoding cannot disagree on a tie.
pub fn greedy(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    let _ = best;
    row.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        })
        .0 as u32
}
