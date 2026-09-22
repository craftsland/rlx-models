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

//! Can speculation pay for itself on this model at all?
//!
//! Speculation replaces one plain decode step per token with one *chunked*
//! forward per round, and wins only if
//!
//! ```text
//!   chunk(K+1) + K * draft   <   tokens_per_round * step
//! ```
//!
//! End-to-end timings cannot answer that here. On this page the vision encode
//! plus a 927-token prefill is ~100 s of a ~130 s run and is paid identically
//! either way, so it swamps the difference — and inferring the per-token decode
//! cost by subtracting an estimate of it produced two runs that disagreed about
//! the sign of the result.
//!
//! So measure the three terms directly, interleaved in one process on one
//! cache, where a change in machine load hits every term alike. Interleaving
//! matters: this box is shared, and the same benchmark returned 88 s, 129 s and
//! 255 s for identical work depending on what else was running.
//!
//! Run:
//!   cargo run --release -p rlx-jina-ocr --features tokenizer,metal \
//!     --example mtp_cost -- --device metal

use anyhow::{Context, Result, bail};
use rlx_jina_ocr::fixtures::{probe_image_path, resolve_model_dir_path};
use rlx_jina_ocr::mtp::MtpKvCache;
use rlx_jina_ocr::preprocess::preprocess_one;
use rlx_jina_ocr::runner::JinaOcrRunner;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn med(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut device = "auto".to_string();
    let mut model_dir: Option<PathBuf> = None;
    let mut reps = 12usize;
    let mut kmax = 3usize;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        let val = args
            .get(i + 1)
            .cloned()
            .with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--device" => device = val?,
            "--model-dir" => model_dir = Some(val?.into()),
            "--reps" => reps = val?.parse()?,
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

    let img = rlx_jina_ocr::preprocess::load_image(&probe_image_path())?;
    let mode = runner.default_image_mode();
    let pre = preprocess_one(&img, mode);
    let (prompt_ids, prompt_hidden) =
        runner.prompt_hidden_states(&pre, rlx_jina_ocr::prompt::DEFAULT_OCR_PROMPT)?;
    let n_prompt = prompt_ids.len();

    let mtp = runner
        .load_mtp()?
        .context("checkpoint has no FastMTP head")?;

    // One draft step, measured on the host. Priming is excluded: it is a
    // one-off, not a per-round cost.
    let seed = prompt_hidden[(n_prompt - 1) * h..].to_vec();
    let mut draft_times = Vec::new();
    mtp.with_head(|head, shared| -> Result<()> {
        let mut kv = MtpKvCache::default();
        head.prime(&prompt_ids, &prompt_hidden, &mut kv, shared)?;
        for r in 0..reps {
            let mut scratch = kv.clone();
            let t = Instant::now();
            head.step(
                prompt_ids[n_prompt - 1],
                n_prompt + r,
                &seed,
                &mut scratch,
                shared,
            )?;
            let out = head.norm_for_lm_head(&seed, shared);
            let _ = (shared.lm_head)(&out)?;
            draft_times.push(t.elapsed());
        }
        Ok(())
    })?;

    // Target-side costs, on a real cache at a realistic length. Interleaved so
    // a load spike cannot land on only one of them.
    let lm = runner.inner_mut().lm_mut().context("LM not loaded")?;
    let embeds = vec![0.01f32; (kmax + 1) * h];

    // A realistic amount of history: attention cost scales with it, and timing
    // against an 8-token cache would flatter the chunked path for free.
    let ctx = n_prompt;
    let (_, mut kv) = lm.prefill(&vec![0.01f32; ctx * h], ctx)?;
    let base = kv.valid();

    let mut step_times = Vec::new();
    let mut chunk_times: Vec<Vec<Duration>> = vec![Vec::new(); kmax + 1];
    // One warm-up of each shape first, so no timing pays for a graph compile.
    lm.decode_step(&embeds[..h], base, &mut kv)?;
    lm.rollback(&mut kv, base);
    for n in 1..=kmax + 1 {
        lm.decode_chunk(&embeds[..n * h], base, n, &mut kv)?;
        lm.rollback(&mut kv, base);
    }
    for _ in 0..reps {
        let t = Instant::now();
        lm.decode_step(&embeds[..h], base, &mut kv)?;
        step_times.push(t.elapsed());
        lm.rollback(&mut kv, base);

        for n in 1..=kmax + 1 {
            let t = Instant::now();
            lm.decode_chunk(&embeds[..n * h], base, n, &mut kv)?;
            chunk_times[n - 1].push(t.elapsed());
            lm.rollback(&mut kv, base);
        }
    }

    let step = med(step_times);
    let draft = med(draft_times);
    println!("median over {reps} reps, interleaved, at {ctx} tokens of context\n");
    println!(
        "plain decode step          {:>8.1} ms",
        step.as_secs_f64() * 1e3
    );
    println!(
        "one host draft step        {:>8.1} ms",
        draft.as_secs_f64() * 1e3
    );
    println!();
    println!(
        "{:<6} {:>12} {:>14} {:>16} {:>12}",
        "K", "chunk(K+1)", "round cost", "tokens/round*", "break-even"
    );
    // Measured on the bundled page; see examples/mtp_probe.rs.
    const TOKENS_PER_ROUND: [f64; 4] = [1.00, 1.26, 1.32, 1.32];
    for k in 1..=kmax {
        let chunk = med(chunk_times[k].clone()); // n = k + 1
        let round = chunk.as_secs_f64() + k as f64 * draft.as_secs_f64();
        let tpr = TOKENS_PER_ROUND[k.min(3)];
        let per_token = round / tpr;
        println!(
            "{:<6} {:>11.1}ms {:>13.1}ms {:>16.2} {:>11.2}x",
            k,
            chunk.as_secs_f64() * 1e3,
            round * 1e3,
            tpr,
            step.as_secs_f64() / per_token
        );
    }
    println!(
        "\n* tokens/round measured separately by examples/mtp_probe.rs.\n\
         break-even > 1.00x means speculation is faster than plain decode."
    );
    Ok(())
}
