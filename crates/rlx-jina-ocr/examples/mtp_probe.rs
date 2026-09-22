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

//! Recover FastMTP's wiring by scoring it, instead of guessing.
//!
//! The checkpoint ships draft weights but not the module that runs them — the
//! card points at a vLLM plugin — so the concatenation order into `eh_proj` is
//! not written down anywhere. It fails silently if wrong: the head still emits
//! in-vocabulary tokens, just uninformed ones, so the only symptom is a low
//! acceptance rate, which is slow and noisy to measure end to end.
//!
//! **The control comes first.** Before any draft number is printed, this checks
//! that the *target's* own tapped hidden state, run through the shared norm and
//! host LM head, reproduces the tokens the target actually generated. If that
//! is not ~100% then the tap, the norm or the host head is wrong and every
//! draft score below it is uninterpretable — which is exactly the trap that
//! made an earlier version of this probe waste a run.
//!
//! Scores on a greedy continuation rather than on the prompt. Every token of a
//! greedy continuation is by construction the target's own pick, so agreement
//! with it *is* the acceptance rate speculation would see. The image prompt is
//! 903 identical placeholders and ~23 template tokens, which measures nothing.
//!
//! `--text-only` skips the vision tower entirely and drives the decoder from
//! bare token ids. The continuation is then whatever the LM does with an
//! arbitrary prefix, which is fine here: the metric is self-agreement, and it
//! makes the run cheap enough to finish on a loaded machine.
//!
//! Run:
//!   cargo run --release -p rlx-jina-ocr --features tokenizer,metal \
//!     --example mtp_probe -- --device metal --text-only

use anyhow::{Context, Result, bail};
use rlx_jina_ocr::fixtures::{probe_image_path, resolve_model_dir_path};
use rlx_jina_ocr::mtp::{MtpKvCache, MtpWiring};
use rlx_jina_ocr::preprocess::preprocess_one;
use rlx_jina_ocr::runner::JinaOcrRunner;
use rlx_unlimited_ocr::preprocess::PreprocessedImage;
use std::path::PathBuf;

fn argmax_u32(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut device = "auto".to_string();
    let mut model_dir: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut gen_tokens = 24usize;
    let mut text_only = false;
    let mut kmax = 4usize;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        if flag == "--text-only" {
            text_only = true;
            i += 1;
            continue;
        }
        let val = args
            .get(i + 1)
            .cloned()
            .with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--device" => device = val?,
            "--model-dir" => model_dir = Some(val?.into()),
            "--image" => image = Some(val?.into()),
            "--gen" => gen_tokens = val?.parse()?,
            "--kmax" => kmax = val?.parse()?,
            other => bail!("unknown flag {other}"),
        }
        i += 2;
    }

    let dir = model_dir
        .or_else(|| resolve_model_dir_path("hf"))
        .context("no checkpoint; pass --model-dir")?;
    let dev = rlx_jina_ocr::resolve_device(Some(&device))?;
    let mut runner = JinaOcrRunner::open(&dir, dev)?;
    runner.load_weights()?;
    let h = runner.config().lm.hidden_size;

    // Prompt ids and the images they refer to.
    let (prompt_ids, images): (Vec<u32>, Vec<PreprocessedImage>) = if text_only {
        // Arbitrary but valid, and deliberately clear of the image placeholder
        // so no vision embedding is expected.
        ((1000u32..1032).collect(), Vec::new())
    } else {
        let img = rlx_jina_ocr::preprocess::load_image(&image.unwrap_or_else(probe_image_path))?;
        let mode = runner.default_image_mode();
        let pre = preprocess_one(&img, mode);
        let ids = runner.build_prompt_ids(rlx_jina_ocr::prompt::DEFAULT_OCR_PROMPT, &pre)?;
        (ids, vec![pre])
    };
    let n_prompt = prompt_ids.len();

    let mut opts = rlx_jina_ocr::runner::default_sample_opts();
    opts.max_new_tokens = gen_tokens;
    let all = runner
        .inner_mut()
        .generate_from_ids(prompt_ids.clone(), &images, &opts)?;
    let cont: Vec<u32> = all[n_prompt..].to_vec();
    if cont.len() < 4 {
        bail!("continuation too short to score ({} tokens)", cont.len());
    }
    let (prompt_hidden, _cont_logits, cont_hidden) =
        runner
            .inner_mut()
            .hidden_trace(&prompt_ids, &cont, &images)?;
    eprintln!(
        "[mtp-probe] {n_prompt} prompt tokens, {} generated, hidden {h}",
        cont.len()
    );

    let mut mtp = runner
        .load_mtp()?
        .context("checkpoint has no FastMTP head")?;

    // ---- control ----
    let mut tgt_hits = 0usize;
    mtp.with_head(|head, shared| -> Result<()> {
        for i in 0..cont.len() - 1 {
            let logits = (shared.lm_head)(&head.norm_for_lm_head(&cont_hidden[i], shared))?;
            tgt_hits += (argmax_u32(&logits) == cont[i + 1]) as usize;
        }
        Ok(())
    })?;
    let denom = cont.len() - 1;
    let tgt_pct = 100.0 * tgt_hits as f64 / denom as f64;
    println!("control: target hidden -> its own next token: {tgt_hits}/{denom} ({tgt_pct:.1}%)");
    if tgt_pct < 95.0 {
        println!(
            "  ^ expected ~100%. The tap, the shared norm or the host LM head is wrong,\n    \
             so the draft scores below cannot be interpreted."
        );
    }
    println!();

    let mut tokens = prompt_ids.clone();
    tokens.extend_from_slice(&cont);
    let hidden_at = |p: usize| -> &[f32] {
        if p < n_prompt {
            &prompt_hidden[p * h..(p + 1) * h]
        } else {
            &cont_hidden[p - n_prompt]
        }
    };

    println!(
        "{:<14} {:<8} {:<8} {:>16}",
        "concat", "primed", "scored", "top-1 vs target"
    );
    let mut best: Option<(f64, bool, bool)> = None;
    for embed_first in [true, false] {
        for primed in [true, false] {
            mtp.set_wiring(MtpWiring { embed_first });
            let (mut hits, mut total) = (0usize, 0usize);
            mtp.with_head(|head, shared| -> Result<()> {
                let mut kv = MtpKvCache::default();
                if primed {
                    head.prime(&prompt_ids, &prompt_hidden, &mut kv, shared)?;
                }
                for p in n_prompt..tokens.len() - 1 {
                    let out = head.step(tokens[p], p, hidden_at(p - 1), &mut kv, shared)?;
                    let logits = (shared.lm_head)(&head.norm_for_lm_head(&out, shared))?;
                    hits += (argmax_u32(&logits) == tokens[p + 1]) as usize;
                    total += 1;
                }
                Ok(())
            })?;
            let pct = 100.0 * hits as f64 / total.max(1) as f64;
            println!(
                "{:<14} {:<8} {:<8} {:>15.1}%",
                if embed_first {
                    "embed;hidden"
                } else {
                    "hidden;embed"
                },
                primed,
                total,
                pct
            );
            if best.is_none_or(|(b, _, _)| pct > b) {
                best = Some((pct, embed_first, primed));
            }
        }
    }
    if let Some((score, ef, pr)) = best {
        println!(
            "\nbest: concat={} primed={pr} at {score:.1}% top-1 agreement",
            if ef { "embed;hidden" } else { "hidden;embed" }
        );
    }

    // ---- draft depth curve ----
    //
    // `accepted / proposed` is a misleading way to judge a draft head, because
    // it averages a good first step with the decayed ones behind it: steps past
    // the first chain the draft's *own* hidden state instead of the target's.
    // What decides whether speculation pays is tokens-per-round at each depth,
    // and every depth can be read off one pass — run the full recursion at
    // `kmax`, record how long the accepted prefix was, and the survival curve
    // gives every shallower K for free.
    mtp.set_wiring(MtpWiring { embed_first: true });
    mtp.set_steps(kmax)?;
    let mut survive = vec![0usize; kmax]; // survive[i] = accepted prefix > i
    let mut rounds = 0usize;
    mtp.with_head(|head, shared| -> Result<()> {
        let mut kv = MtpKvCache::default();
        head.prime(&prompt_ids, &prompt_hidden, &mut kv, shared)?;
        for p in n_prompt..tokens.len() - 1 {
            // Propose from the target's state at p-1, exactly as the real loop
            // does, on a scratch copy so the cache advances only by the one
            // row the target actually commits.
            let mut scratch = kv.clone();
            let drafts = head.propose(tokens[p], p - 1, hidden_at(p - 1), &mut scratch, shared)?;
            rounds += 1;
            for (i, &d) in drafts.iter().enumerate() {
                match tokens.get(p + 1 + i) {
                    Some(&want) if want == d => survive[i] += 1,
                    // Past the end of what we generated: unknown, not a miss.
                    None => {}
                    _ => break,
                }
            }
            head.step(tokens[p], p, hidden_at(p - 1), &mut kv, shared)?;
        }
        Ok(())
    })?;

    println!(
        "\n{:<6} {:>12} {:>16} {:>14}",
        "K", "step accept", "E[accepted]", "tokens/round"
    );
    let mut cum = 0.0f64;
    for k in 0..kmax {
        let p_k = survive[k] as f64 / rounds.max(1) as f64;
        cum += p_k;
        println!(
            "{:<6} {:>11.1}% {:>16.2} {:>14.2}",
            k + 1,
            100.0 * p_k,
            cum,
            1.0 + cum
        );
    }
    println!(
        "\nEach extra draft step costs one host forward through the block plus a\n\
         full {}x{} lm_head projection, so depth only pays while the marginal\n\
         step-accept rate above is still worth that.",
        runner.config().lm.vocab_size,
        h
    );
    Ok(())
}
