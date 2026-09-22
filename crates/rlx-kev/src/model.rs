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

//! The decision model: pack a request, run one prefill, read out
//! probabilities. No token is ever generated.
//!
//! # Why the row form
//!
//! Qwen3.5 interleaves attention with **Gated DeltaNet** layers, which are
//! recurrent and therefore ignore attention masks entirely. The packed
//! block-causal mask that isolates question branches on an attention-only
//! backbone is a no-op on those layers, so each question instead runs as its
//! own row: `state ++ branch_k`. Isolation is then exact by construction —
//! the rows are independent computations.
//!
//! That form has a property worth naming, because it is what lets this run on
//! a stock causal prefill graph with no custom mask at all: within a row the
//! state occupies positions `0..Ls` and the branch continues at `Ls..`, so
//! **position `i` is just `i`**. The branch-restart bookkeeping in
//! [`crate::encode`] only bites in the packed form.
//!
//! # Why right-padding is safe
//!
//! Rows are right-padded to a common length. Every layer here is causal —
//! attention masks keys above the query index, and the DeltaNet recurrence
//! only ever reads its own past — so a pad at position `p` cannot influence
//! any output at a position below `p`. All readouts are below every pad.
//! (This is the same reasoning kev states and measured as exact; it is *not*
//! true of a prefill that exports recurrent state, which is why we build the
//! plain prefill graph rather than the cache-exporting one.)

use anyhow::{Context, Result, bail};
use rlx_qwen35::{Qwen35Config, Qwen35Weights};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::api::{Answer, QuestionMeta, Record, SystemOneRequest, to_answers, to_record};
use crate::encode::{EncodeOpts, Encoder, TokenizerLike, rows_of};
use crate::pointer::PointerHead;

/// Serving limits. Training used 384 / 1024; the per-branch cap mirrors
/// Jev's ~32k, bounded in practice by the base model's window.
pub const INFER_MAX_STATE: usize = 8192;
pub const INFER_MAX_BRANCH: usize = 8192;

/// Sequence lengths are rounded up to a multiple of this so a steady
/// workload stops recompiling. Matches kev's `KEV_SHAPE_BUCKET`.
pub const SHAPE_BUCKET: usize = 64;

/// How the compiled shape is chosen.
///
/// RLX compiles per shape and each compiled graph owns its own parameter
/// arena, so a resident variant costs a full copy of the weights. That is why
/// there is a single compiled slot rather than a cache: two live variants of a
/// 4B f32 backbone is 32 GB of duplicated weights.
#[derive(Debug, Clone, Copy)]
pub enum ShapePolicy {
    /// Round up to the next power-of-two row count and `SHAPE_BUCKET`
    /// multiple, recompiling when the bucket changes. Keeps the base weights
    /// resident so a rebuild is possible.
    Bucketed { max_rows: usize, max_seq: usize },
    /// Always run at exactly this shape. Compiles once, then releases the
    /// base weights — the lowest-memory option, and the right one for a
    /// server with a predictable workload.
    Fixed { rows: usize, seq: usize },
}

impl Default for ShapePolicy {
    fn default() -> Self {
        Self::Bucketed {
            max_rows: 32,
            max_seq: 2048,
        }
    }
}

/// Readout offsets for one question inside its row.
struct Readout {
    row: usize,
    decide: usize,
    opts: Vec<usize>,
}

/// One loaded checkpoint, ready to answer requests.
pub struct DecisionModel<T: TokenizerLike> {
    encoder: Encoder<T>,
    head: PointerHead,
    cfg: Qwen35Config,
    /// Dropped once a [`ShapePolicy::Fixed`] graph has been built.
    weights: Option<Qwen35Weights>,
    device: Device,
    policy: ShapePolicy,
    /// `(rows, seq, graph)` — one slot, see [`ShapePolicy`].
    compiled: Option<(usize, usize, CompiledGraph)>,
    encode_opts: EncodeOpts,
}

impl<T: TokenizerLike> DecisionModel<T> {
    /// Assemble a model from its parts.
    ///
    /// Errors when the pointer head and the backbone disagree about the
    /// hidden size — a mismatch there reads as a shape error only if you are
    /// lucky, and as plausible garbage if the sizes happen to divide.
    pub fn new(
        encoder: Encoder<T>,
        head: PointerHead,
        cfg: Qwen35Config,
        weights: Qwen35Weights,
        device: Device,
        policy: ShapePolicy,
    ) -> Result<Self> {
        if head.hidden != cfg.hidden_size {
            bail!(
                "pointer head expects hidden={} but the backbone is {}",
                head.hidden,
                cfg.hidden_size
            );
        }
        if rlx_ir::env::parse_or::<usize>("RLX_QWEN35_SWA_WINDOW", 0) > 0 {
            bail!(
                "RLX_QWEN35_SWA_WINDOW is set; sliding-window prefill drops middle \
                 context and would change every probability this model returns"
            );
        }
        Ok(Self {
            encoder,
            head,
            cfg,
            weights: Some(weights),
            device,
            policy,
            compiled: None,
            encode_opts: EncodeOpts {
                max_state: INFER_MAX_STATE,
                max_branch: INFER_MAX_BRANCH,
                strict: false,
                option_isolation: false,
            },
        })
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    pub fn head(&self) -> &PointerHead {
        &self.head
    }

    pub fn encoder(&self) -> &Encoder<T> {
        &self.encoder
    }

    /// The compiled shape currently resident, if any.
    pub fn compiled_shape(&self) -> Option<(usize, usize)> {
        self.compiled.as_ref().map(|(r, s, _)| (*r, *s))
    }

    /// Per-question probability vectors for one record.
    pub fn probs(&mut self, rec: &Record) -> Result<Vec<Vec<f32>>> {
        let enc = self.encoder.encode(rec, self.encode_opts)?;
        let (state_ids, _state_pos, rows) = rows_of(&enc)?;
        if rows.is_empty() {
            bail!("record has no questions");
        }
        let ls = state_ids.len();

        // Each question becomes one causal row: the state, then its branch.
        let mut seqs: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
        let mut readouts: Vec<Readout> = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            let mut ids = Vec::with_capacity(ls + r.ids.len());
            ids.extend_from_slice(&state_ids);
            ids.extend_from_slice(&r.ids);
            seqs.push(ids);
            readouts.push(Readout {
                row: i,
                decide: ls + r.decide,
                opts: r.opts.iter().map(|o| ls + o).collect(),
            });
        }

        let longest = seqs.iter().map(Vec::len).max().unwrap_or(0);
        let (rows_n, seq_n) = self.resolve_shape(seqs.len(), longest)?;

        let hidden = self.forward(&seqs, rows_n, seq_n)?;
        let d = self.cfg.hidden_size;

        let mut out = Vec::with_capacity(readouts.len());
        for ro in &readouts {
            let base = ro.row * seq_n * d;
            let row_hidden = &hidden[base..base + seq_n * d];
            let h_decide = &row_hidden[ro.decide * d..(ro.decide + 1) * d];
            let mut h_opts = Vec::with_capacity(ro.opts.len() * d);
            for o in &ro.opts {
                h_opts.extend_from_slice(&row_hidden[o * d..(o + 1) * d]);
            }
            out.push(self.head.probs(h_decide, &h_opts)?);
        }
        Ok(out)
    }

    /// Answer a `POST /v1/systemone` request.
    pub fn answer(&mut self, req: &SystemOneRequest) -> Result<(Vec<(String, Answer)>, Usage)> {
        let (rec, meta) = to_record(req)?;
        let probs = self.probs(&rec)?;
        let answers = to_answers(&probs, &meta)?;
        let usage = self.usage(&rec, &meta)?;
        Ok((answers, usage))
    }

    fn usage(&self, rec: &Record, _meta: &[QuestionMeta]) -> Result<Usage> {
        let enc = self.encoder.encode(rec, self.encode_opts)?;
        Ok(Usage {
            input_tokens: enc.ids.len(),
            state_tokens: enc.state_len(),
        })
    }

    /// Pick (and validate) the compiled shape for a request.
    fn resolve_shape(&self, rows: usize, longest: usize) -> Result<(usize, usize)> {
        match self.policy {
            ShapePolicy::Fixed { rows: r, seq: s } => {
                if rows > r || longest > s {
                    bail!(
                        "request needs {rows}×{longest} but the model is pinned to \
                         {r}×{s}; raise the fixed shape or use a bucketed policy"
                    );
                }
                Ok((r, s))
            }
            ShapePolicy::Bucketed { max_rows, max_seq } => {
                let r = rows.next_power_of_two().min(max_rows).max(rows);
                let s = longest.div_ceil(SHAPE_BUCKET) * SHAPE_BUCKET;
                if r > max_rows || s > max_seq {
                    bail!(
                        "request needs {r}×{s} which exceeds the {max_rows}×{max_seq} \
                         limit ({rows} questions, {longest} tokens in the longest branch)"
                    );
                }
                Ok((r, s.max(SHAPE_BUCKET)))
            }
        }
    }

    /// Run the backbone, returning the final-norm hidden states
    /// `[rows, seq, hidden]`.
    fn forward(&mut self, seqs: &[Vec<u32>], rows: usize, seq: usize) -> Result<Vec<f32>> {
        self.ensure_compiled(rows, seq)?;

        // Pad id 0 rather than the tokenizer's: pads sit after every real
        // token and nothing causal can read forward, so the value is
        // unobservable — and 0 is in range for every vocabulary.
        let mut input_ids = vec![0f32; rows * seq];
        for (i, s) in seqs.iter().enumerate() {
            for (j, id) in s.iter().enumerate() {
                input_ids[i * seq + j] = *id as f32;
            }
        }

        let graph = self
            .compiled
            .as_mut()
            .map(|(_, _, g)| g)
            .expect("compiled above");
        let out = graph.run(&[("input_ids", &input_ids[..])]);
        let want = rows * seq * self.cfg.hidden_size;
        let hidden = out
            .first()
            .ok_or_else(|| anyhow::anyhow!("prefill graph produced no outputs"))?;
        if hidden.len() != want {
            bail!(
                "expected [{rows}, {seq}, {}] = {want} hidden values, got {}",
                self.cfg.hidden_size,
                hidden.len()
            );
        }
        Ok(hidden.clone())
    }

    fn ensure_compiled(&mut self, rows: usize, seq: usize) -> Result<()> {
        if matches!(self.compiled, Some((r, s, _)) if r == rows && s == seq) {
            return Ok(());
        }
        let weights = self.weights.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "base weights were released after the pinned {:?} build; a different \
                 shape cannot be compiled",
                self.compiled_shape()
            )
        })?;

        // Drop the previous variant first: its arena holds a full copy of the
        // weights, and holding two at once doubles resident memory.
        self.compiled = None;

        let (hir, params, packed) = rlx_qwen35::build_qwen35_prefill_flow_ext(
            &self.cfg, weights, rows, seq,
            /*with_lm_head*/ false,
            /*last_logits_only*/ false,
            /*enable_mtp_head*/ false,
            /*runtime_mrope*/ false,
            /*fast_mtp*/ false,
            /*export_normed_hidden*/ true,
        )
        .with_context(|| format!("building the qwen35 prefill graph at {rows}×{seq}"))?;
        if !packed.is_empty() {
            bail!(
                "this path expects dequantized f32 weights, but {} tensor(s) came \
                 back packed",
                packed.len()
            );
        }

        let mut compiled = Session::new(self.device)
            .compile_hir(hir)
            .with_context(|| format!("compiling the prefill graph at {rows}×{seq}"))?;
        // Drain rather than iterate: each uploaded tensor is freed as we go,
        // so the peak is weights + arena + one tensor, not weights + arena +
        // a second full copy.
        for (name, data) in params {
            compiled.set_param(&name, &data);
        }

        if let ShapePolicy::Fixed { .. } = self.policy {
            self.weights = None;
        }
        self.compiled = Some((rows, seq, compiled));
        Ok(())
    }
}

/// Token accounting for the response body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: usize,
    pub state_tokens: usize,
}
