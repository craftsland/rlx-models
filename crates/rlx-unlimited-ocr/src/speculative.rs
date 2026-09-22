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

//! Greedy speculative decoding: draft `k` tokens cheaply, then let the target
//! model check all of them in **one** forward.
//!
//! The whole point is that it is **lossless**. The target scores position `i`
//! against exactly the history it would have had anyway, so accepting a draft
//! token only when it equals the target's own pick means the emitted sequence
//! is bit-identical to plain greedy decode — a bad drafter costs speed, never
//! correctness. [`SpeculativeStats::accepted`] is therefore a performance
//! number, not a quality one.
//!
//! The drafter is a trait so that property is testable independently of any
//! particular draft head: a deliberately terrible drafter must still produce
//! the same tokens (see `tests/speculative_equivalence.rs`).

use crate::generation::{SampleOpts, sample_token};
use crate::lm_device::{CompiledLm, DeviceKvCache};
use anyhow::{Result, ensure};
use std::time::{Duration, Instant};

/// Proposes continuation tokens. Quality affects throughput only.
pub trait Drafter {
    /// How many tokens to propose per round.
    fn depth(&self) -> usize;

    /// Propose up to [`Self::depth`] tokens following `last_token`, which sits
    /// at `last_pos`. `hidden` is that position's post-final-norm state.
    ///
    /// Returning fewer than `depth` (or none) is allowed and simply shortens
    /// the round.
    fn draft(&mut self, last_token: u32, last_pos: usize, hidden: &[f32]) -> Result<Vec<u32>>;

    /// Drop any draft state past `pos` after a rejection.
    fn rollback(&mut self, pos: usize);

    /// Prime draft state from the prompt, before any proposing.
    ///
    /// `tokens` is the prompt and `hidden_all` its per-position pre-final-norm
    /// hidden states, `[tokens.len(), hidden]`. A draft head that is itself a
    /// transformer layer needs this: without it, it attends only to the rows it
    /// wrote while drafting and proposes with essentially no context, which
    /// costs acceptance rather than correctness. Defaults to a no-op for
    /// stateless drafters.
    fn prime(&mut self, tokens: &[u32], hidden_all: &[f32]) -> Result<()> {
        let _ = (tokens, hidden_all);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    /// Verify passes run.
    pub rounds: usize,
    /// Draft tokens proposed.
    pub proposed: usize,
    /// Draft tokens the target agreed with.
    pub accepted: usize,
    /// Tokens emitted in total (accepted + one target token per round).
    pub emitted: usize,
    /// One-off cost of priming the drafter over the prompt.
    pub prime_time: Duration,
    /// Time inside [`Drafter::draft`], summed.
    pub draft_time: Duration,
    /// Time inside the chunked target forward, summed.
    pub verify_time: Duration,
}

impl SpeculativeStats {
    /// Fraction of proposals accepted; `0.0` when nothing was proposed.
    pub fn acceptance_rate(&self) -> f32 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f32 / self.proposed as f32
        }
    }

    /// Tokens emitted per verify pass. At or below 1.0 speculation is not
    /// paying for itself and plain decode would be faster.
    pub fn tokens_per_round(&self) -> f32 {
        if self.rounds == 0 {
            0.0
        } else {
            self.emitted as f32 / self.rounds as f32
        }
    }
}

/// Run greedy speculative decoding until EOS or `max_new_tokens`.
///
/// `tokens` starts as the prompt and grows; `hidden_all` is the prompt's
/// per-position pre-final-norm states (from
/// [`CompiledLm::prefill_with_all_hidden`]) and `logits` its last-position
/// logits. `embed` maps a token id to its input embedding.
pub fn generate_speculative(
    lm: &mut CompiledLm,
    kv: &mut DeviceKvCache,
    drafter: &mut dyn Drafter,
    tokens: &mut Vec<u32>,
    logits: Vec<f32>,
    hidden_all: Vec<f32>,
    opts: &SampleOpts,
    eos: &[u32],
    embed: &mut dyn FnMut(&[u32]) -> Result<Vec<f32>>,
) -> Result<SpeculativeStats> {
    let hidden_size = lm.config().hidden_size;
    ensure!(!tokens.is_empty(), "speculative decode needs a prompt");
    ensure!(
        hidden_all.len() == tokens.len() * hidden_size,
        "prompt hidden is {} elements, want {}*{hidden_size}",
        hidden_all.len(),
        tokens.len()
    );

    let t_prime = Instant::now();
    drafter.prime(tokens, &hidden_all)?;
    let prime_time = t_prime.elapsed();

    let mut stats = SpeculativeStats {
        prime_time,
        ..SpeculativeStats::default()
    };
    let mut logits = logits;
    let mut seed_hidden = hidden_all[(tokens.len() - 1) * hidden_size..].to_vec();

    while stats.emitted < opts.max_new_tokens {
        // The token plain decode would emit now.
        let next = sample_token(&logits, opts, tokens);
        tokens.push(next);
        stats.emitted += 1;
        if eos.contains(&next) {
            break;
        }
        if stats.emitted >= opts.max_new_tokens {
            break;
        }

        // `next` sits at this index; the draft continues from it.
        let next_pos = tokens.len() - 1;
        let t_draft = Instant::now();
        let draft = drafter.draft(next, next_pos - 1, &seed_hidden)?;
        stats.draft_time += t_draft.elapsed();
        let k = draft.len().min(opts.max_new_tokens - stats.emitted);
        let draft = &draft[..k];
        stats.proposed += k;

        // Verify `next` plus the draft in one pass. Position 0 scores `next`,
        // so its logits predict the token after it — which is what we need even
        // when the draft is empty.
        let mut chunk: Vec<u32> = Vec::with_capacity(1 + k);
        chunk.push(next);
        chunk.extend_from_slice(draft);
        let embeds = embed(&chunk)?;
        let base_valid = kv.valid();
        let t_verify = Instant::now();
        let (per_pos, per_hidden) =
            lm.decode_chunk_with_hidden(&embeds, next_pos, chunk.len(), kv)?;
        stats.verify_time += t_verify.elapsed();
        stats.rounds += 1;

        // Walk the draft, comparing each proposal against what the target would
        // have produced with the same history.
        let mut accepted = 0usize;
        let mut cursor_logits = per_pos[0].clone();
        for (i, &proposed) in draft.iter().enumerate() {
            let target = sample_token(&cursor_logits, opts, tokens);
            if target != proposed {
                break;
            }
            tokens.push(target);
            accepted += 1;
            stats.accepted += 1;
            stats.emitted += 1;
            if eos.contains(&target) || stats.emitted >= opts.max_new_tokens {
                break;
            }
            cursor_logits = per_pos[i + 1].clone();
        }

        let done =
            tokens.last().is_some_and(|t| eos.contains(t)) || stats.emitted >= opts.max_new_tokens;

        // Keep only `next` + the accepted draft tokens in the cache.
        lm.rollback(kv, base_valid + 1 + accepted);
        drafter.rollback(next_pos + accepted);

        if done {
            break;
        }
        logits = cursor_logits;
        seed_hidden = per_hidden[accepted].clone();
    }
    Ok(stats)
}
