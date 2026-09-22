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

//! Greedy speculative decoding must be **lossless**.
//!
//! The emitted sequence has to be bit-identical to plain decode no matter what
//! the drafter proposes — that is the property that makes speculation a pure
//! speed trade rather than a quality one. So the drafters here are chosen to be
//! adversarial: one always proposes a constant, one proposes the *right* answer
//! shifted, one proposes nothing. Any of them silently changing the output
//! would mean the verify pass or the KV rollback is wrong.
//!
//! Run on every backend, because the verify pass uses the additive-bias
//! attention path that ordinary decode never touches.

use rlx_core::backend_matrix::{Failures, available_devices, max_rel_diff};
use rlx_runtime::Device;
use rlx_unlimited_ocr::config::UnlimitedOcrConfig;
use rlx_unlimited_ocr::expert_pack::PackedLmWeights;
use rlx_unlimited_ocr::generation::SampleOpts;
use rlx_unlimited_ocr::lm_device::{CompiledLm, DeviceKvCache};
use rlx_unlimited_ocr::lm_precision::ResolvedLmPrecision;
use rlx_unlimited_ocr::speculative::{Drafter, SpeculativeStats, generate_speculative};
use std::sync::Arc;

mod tiny;
use tiny::{cfg, fill, synthetic_weights};

const PROMPT: usize = 6;
const NEW: usize = 12;
/// In-vocabulary but very unlikely to be the greedy pick.
const BOGUS: u32 = 127;

/// Always proposes the same token — near-zero acceptance.
struct Constant(u32, usize);
impl Drafter for Constant {
    fn depth(&self) -> usize {
        self.1
    }
    fn draft(&mut self, _last: u32, _pos: usize, _h: &[f32]) -> anyhow::Result<Vec<u32>> {
        Ok(vec![self.0; self.1])
    }
    fn rollback(&mut self, _pos: usize) {}
}

/// Proposes nothing — speculation degenerates to plain decode.
struct Empty;
impl Drafter for Empty {
    fn depth(&self) -> usize {
        0
    }
    fn draft(&mut self, _last: u32, _pos: usize, _h: &[f32]) -> anyhow::Result<Vec<u32>> {
        Ok(Vec::new())
    }
    fn rollback(&mut self, _pos: usize) {}
}

/// Replays a known-good continuation — high acceptance, exercising the
/// multi-token accept path and the cache bookkeeping that follows it.
struct Oracle {
    truth: Vec<u32>,
    depth: usize,
}
impl Drafter for Oracle {
    fn depth(&self) -> usize {
        self.depth
    }
    fn draft(&mut self, last: u32, _pos: usize, _h: &[f32]) -> anyhow::Result<Vec<u32>> {
        // Find `last` in the reference run and propose what followed it.
        let Some(i) = self.truth.iter().position(|&t| t == last) else {
            return Ok(Vec::new());
        };
        Ok(self
            .truth
            .iter()
            .skip(i + 1)
            .take(self.depth)
            .copied()
            .collect())
    }
    fn rollback(&mut self, _pos: usize) {}
}

fn pack_for(cfg: &UnlimitedOcrConfig) -> Arc<PackedLmWeights> {
    let mut wm = synthetic_weights(cfg);
    Arc::new(
        PackedLmWeights::from_weight_map(&mut wm, cfg.clone(), ResolvedLmPrecision::F32)
            .expect("pack"),
    )
}

fn opts() -> SampleOpts {
    SampleOpts {
        temperature: 0.0,
        max_new_tokens: NEW,
        // Keep the n-gram guard on: it makes sampling history-dependent, which
        // is exactly the thing a verify pass can get subtly wrong.
        no_repeat_ngram_size: 4,
        ngram_window: 64,
        ..SampleOpts::default()
    }
}

/// Plain greedy decode — the reference sequence, plus the LM and cache it left
/// behind so a test can compare cache state and not just tokens.
fn plain_full(
    cfg: &UnlimitedOcrConfig,
    pack: &Arc<PackedLmWeights>,
    device: Device,
) -> (Vec<u32>, CompiledLm, DeviceKvCache) {
    let h = cfg.hidden_size;
    let mut lm = CompiledLm::new(device, Arc::clone(pack));
    let embeds = fill(PROMPT * h, 0.5);
    let (mut logits, mut kv) = lm.prefill(&embeds, PROMPT).expect("prefill");
    let mut tokens: Vec<u32> = (0..PROMPT as u32).collect();
    let o = opts();
    for i in 0..NEW {
        let next = rlx_unlimited_ocr::generation::sample_token(&logits, &o, &tokens);
        tokens.push(next);
        if next == cfg.eos_token_id {
            break;
        }
        if i + 1 == NEW {
            break;
        }
        let step = lm.embed_tokens(&[next]).expect("embed");
        logits = lm
            .decode_step(&step, tokens.len() - 1, &mut kv)
            .expect("decode");
    }
    (tokens[PROMPT..].to_vec(), lm, kv)
}

fn plain(cfg: &UnlimitedOcrConfig, pack: &Arc<PackedLmWeights>, device: Device) -> Vec<u32> {
    plain_full(cfg, pack, device).0
}

/// Speculative decode with `drafter`.
fn speculative(
    cfg: &UnlimitedOcrConfig,
    pack: &Arc<PackedLmWeights>,
    device: Device,
    drafter: &mut dyn Drafter,
) -> (Vec<u32>, SpeculativeStats, CompiledLm, DeviceKvCache) {
    let h = cfg.hidden_size;
    let mut lm = CompiledLm::new(device, Arc::clone(pack));
    let embeds = fill(PROMPT * h, 0.5);
    let (logits, mut kv, hidden) = lm
        .prefill_with_all_hidden(&embeds, PROMPT)
        .expect("prefill");
    let mut tokens: Vec<u32> = (0..PROMPT as u32).collect();
    let o = opts();
    let eos = [cfg.eos_token_id];

    // `embed` has to borrow the pack, not the LM, because the LM is borrowed
    // mutably by the loop.
    let pack2 = Arc::clone(pack);
    let mut embed = move |ids: &[u32]| pack2.embed_tokens_lookup(ids);

    let stats = generate_speculative(
        &mut lm,
        &mut kv,
        drafter,
        &mut tokens,
        logits,
        hidden,
        &o,
        &eos,
        &mut embed,
    )
    .expect("speculative decode");
    (tokens[PROMPT..].to_vec(), stats, lm, kv)
}

#[test]
fn speculative_output_is_identical_to_plain_decode() {
    let cfg: UnlimitedOcrConfig = cfg();
    let mut fails = Failures::default();

    for (name, device) in available_devices() {
        let pack = pack_for(&cfg);
        let want = plain(&cfg, &pack, device);

        let mut cases: Vec<(&str, Box<dyn Drafter>)> = vec![
            ("empty", Box::new(Empty)),
            ("constant-depth-1", Box::new(Constant(0, 1))),
            ("constant-depth-3", Box::new(Constant(0, 3))),
            (
                "oracle-depth-3",
                Box::new(Oracle {
                    truth: want.clone(),
                    depth: 3,
                }),
            ),
        ];

        for (label, drafter) in cases.iter_mut() {
            let (got, stats, _, _) = speculative(&cfg, &pack, device, drafter.as_mut());
            if got != want {
                fails.push(
                    name,
                    format!("{label}: got {got:?}, plain decode gives {want:?}"),
                );
            }
            if got.len() != want.len() {
                fails.push(
                    name,
                    format!("{label}: emitted {} vs {}", got.len(), want.len()),
                );
            }
            // The oracle drafter should actually be accepting, or the
            // multi-token accept path is never exercised.
            if *label == "oracle-depth-3" && stats.accepted == 0 && want.len() > 2 {
                fails.push(
                    name,
                    "oracle drafter accepted nothing — the accept path is untested here",
                );
            }
        }
    }
    fails.assert_empty("speculative decode equivalence");
}

/// A drafter that is always right should emit more than one token per verify
/// pass; one that is always wrong should emit exactly one.
#[test]
fn acceptance_statistics_track_draft_quality() {
    let cfg: UnlimitedOcrConfig = cfg();
    let mut fails = Failures::default();

    for (name, device) in available_devices() {
        let pack = pack_for(&cfg);
        let want = plain(&cfg, &pack, device);

        let (_, bad, _, _) = speculative(&cfg, &pack, device, &mut Constant(BOGUS, 3));
        if bad.accepted != 0 {
            fails.push(name, format!("bogus drafter accepted {}", bad.accepted));
        }

        let mut oracle = Oracle {
            truth: want.clone(),
            depth: 3,
        };
        let (_, good, _, _) = speculative(&cfg, &pack, device, &mut oracle);

        // Book-keeping identity. Every round emits the target's own token plus
        // whatever it accepted, and the run's final token is emitted without a
        // verify pass behind it — unless the round that produced it also hit the
        // limit. So the slack is exactly 0 or 1, never more; anything else means
        // a token was emitted that no forward pass scored, or a round ran whose
        // output was dropped. Stated this way it holds for *any* drafter, which
        // a tokens-per-round ratio does not: with zero acceptance the trailing
        // unverified token alone puts the ratio at 1.09 here, not 1.0.
        for (label, s) in [("bogus", bad), ("oracle", good)] {
            let slack = s.emitted as i64 - s.rounds as i64 - s.accepted as i64;
            if !(0..=1).contains(&slack) {
                fails.push(
                    name,
                    format!(
                        "{label}: emitted {} != rounds {} + accepted {} (+0 or 1)",
                        s.emitted, s.rounds, s.accepted
                    ),
                );
            }
        }

        // The speed claim: accepting drafts means fewer verify passes for the
        // same output.
        if good.rounds >= bad.rounds {
            fails.push(
                name,
                format!(
                    "oracle drafter needed {} rounds, bogus needed {} — speculation bought nothing",
                    good.rounds, bad.rounds
                ),
            );
        }
        if good.acceptance_rate() <= 0.0 {
            fails.push(name, "oracle acceptance rate is zero");
        }
    }
    fails.assert_empty("speculative statistics");
}

/// A rejected draft must leave the cache exactly as plain decode would.
///
/// This is the invariant the token-equality test cannot see. Speculation writes
/// the whole chunk into the cache and then rolls back to the accepted prefix;
/// skipping that rollback leaves the rejected tokens' keys in history, where
/// later queries attend to them. On a model this small that perturbation is not
/// enough to move an argmax, so the emitted tokens stay identical and the bug
/// reads as absent — deleting the rollback call passes every other test here.
#[test]
fn rejected_drafts_leave_no_trace_in_the_cache() {
    let cfg: UnlimitedOcrConfig = cfg();
    let mut fails = Failures::default();

    for (name, device) in available_devices() {
        let pack = pack_for(&cfg);
        let (want, plain_lm, plain_kv) = plain_full(&cfg, &pack, device);

        // Depth 3 and never right: every round writes 4 rows and must keep 1.
        let (got, stats, spec_lm, spec_kv) =
            speculative(&cfg, &pack, device, &mut Constant(BOGUS, 3));
        if got != want {
            fails.push(name, "tokens diverged");
            continue;
        }
        if stats.accepted != 0 {
            fails.push(
                name,
                format!("expected no acceptance, got {}", stats.accepted),
            );
        }
        if spec_kv.valid() != plain_kv.valid() {
            fails.push(
                name,
                format!(
                    "cache holds {} rows after speculation, plain decode leaves {} \
                     — rejected drafts were not rolled back",
                    spec_kv.valid(),
                    plain_kv.valid()
                ),
            );
            continue;
        }
        for l in 0..cfg.num_hidden_layers {
            let (pk, pv) = plain_lm.layer_kv(&plain_kv, l).expect("plain layer");
            let (sk, sv) = spec_lm.layer_kv(&spec_kv, l).expect("spec layer");
            // The chunk graph and the step graph are different lowerings of the
            // same projections, so compare numerically rather than bitwise.
            for (what, a, b) in [("k", pk, sk), ("v", pv, sv)] {
                let rel = max_rel_diff(a, b);
                if !rel.is_finite() || rel >= 1e-3 {
                    fails.push(name, format!("layer {l} {what}: cache differs by {rel:.5}"));
                }
            }
        }
    }
    fails.assert_empty("speculative cache rollback");
}
