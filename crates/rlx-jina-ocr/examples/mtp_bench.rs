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

//! End-to-end FastMTP: acceptance statistics, a time breakdown, and the check
//! that the output did not change.
//!
//! **Do not read the `speedup` line as a speed result.** It times the two arms
//! *sequentially*, so on a shared machine it compares load as much as code —
//! identical work here returned 88 s, 129 s and 255 s depending on what else
//! was running, and a sequential comparison once reported 1.05x for something a
//! later run put at 0.62x. Nor can the plain per-token cost be recovered by
//! subtracting an estimated prefill: on a 48-token transcription the vision
//! encode plus the 927-token prefill is ~101 s of a ~131 s run.
//!
//! For the speed question use `examples/mtp_cost.rs`, which times a decode
//! step, a chunked verify forward and a draft step interleaved in one process.
//! What this example is good for is the parts that do not depend on timing:
//! acceptance, tokens per round, where the time goes, and — the load-bearing
//! one — that speculative output stayed byte-identical to plain decode.
//!
//! Run:
//!   cargo run --release -p rlx-jina-ocr --features tokenizer,metal \
//!     --example mtp_bench -- --device metal --max-tokens 48

use anyhow::{Context, Result, bail};
use rlx_jina_ocr::fixtures::{probe_image_path, resolve_model_dir_path};
use rlx_jina_ocr::preprocess::preprocess_one;
use rlx_jina_ocr::runner::{JinaOcrRunner, default_sample_opts};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut device = "auto".to_string();
    let mut max_tokens = 48usize;
    let mut model_dir: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut k: Option<usize> = None;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        let val = args
            .get(i + 1)
            .cloned()
            .with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--device" => device = val?,
            "--max-tokens" => max_tokens = val?.parse()?,
            "--model-dir" => model_dir = Some(val?.into()),
            "--image" => image = Some(val?.into()),
            "--k" => k = Some(val?.parse()?),
            other => bail!("unknown flag {other}"),
        }
        i += 2;
    }

    let dir = model_dir
        .or_else(|| resolve_model_dir_path("hf"))
        .context("no checkpoint; pass --model-dir")?;
    let dev = rlx_jina_ocr::resolve_device(Some(&device))?;
    let image_path = image.unwrap_or_else(probe_image_path);
    let img = rlx_jina_ocr::preprocess::load_image(&image_path)?;

    let mut runner = JinaOcrRunner::open(&dir, dev)?;
    let mode = runner.default_image_mode();
    let pre = preprocess_one(&img, mode);
    let mut opts = default_sample_opts();
    opts.max_new_tokens = max_tokens;

    runner.load_weights()?;
    let mtp = runner.load_mtp()?;
    let Some(mut mtp) = mtp else {
        bail!("checkpoint has no FastMTP head");
    };
    // The checkpoint asks for K=3, but `examples/mtp_probe.rs` measures the
    // third draft step accepting 0% of the time on a real page — a full host
    // forward plus a 129280x1280 projection for nothing.
    if let Some(k) = k {
        mtp.set_steps(k)?;
    }
    eprintln!(
        "[mtp-bench] draft head loaded: K={} extra host RAM {:.0} MB",
        mtp.steps(),
        mtp.host_bytes() as f64 / 1e6
    );

    // Plain decode first, so the speculative run cannot benefit from a cold
    // graph cache the baseline paid for.
    let t0 = Instant::now();
    let plain = runner.generate(&pre, rlx_jina_ocr::prompt::DEFAULT_OCR_PROMPT, &opts)?;
    let plain_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let (spec, stats) =
        runner.generate_with_mtp(&pre, rlx_jina_ocr::prompt::DEFAULT_OCR_PROMPT, &opts, &mtp)?;
    let spec_s = t1.elapsed().as_secs_f64();

    let identical = plain.token_ids == spec.token_ids;
    println!("device            {dev:?}");
    println!("prompt tokens     {}", plain.prompt_len);
    println!(
        "plain             {plain_s:.2}s for {} tokens ({:.3}s/tok)",
        plain.new_tokens,
        plain_s / plain.new_tokens.max(1) as f64
    );
    println!(
        "speculative       {spec_s:.2}s for {} tokens ({:.3}s/tok)",
        spec.new_tokens,
        spec_s / spec.new_tokens.max(1) as f64
    );
    println!(
        "rounds            {} ({:.2} tokens/round)",
        stats.rounds,
        stats.tokens_per_round()
    );
    println!(
        "draft acceptance  {}/{} ({:.1}%)",
        stats.accepted,
        stats.proposed,
        stats.acceptance_rate() * 100.0
    );
    println!("  prime (one-off)  {:.2}s", stats.prime_time.as_secs_f64());
    println!(
        "  draft steps      {:.2}s total ({:.0} ms/round)",
        stats.draft_time.as_secs_f64(),
        1e3 * stats.draft_time.as_secs_f64() / stats.rounds.max(1) as f64
    );
    println!(
        "  verify forwards  {:.2}s total ({:.0} ms/round)",
        stats.verify_time.as_secs_f64(),
        1e3 * stats.verify_time.as_secs_f64() / stats.rounds.max(1) as f64
    );
    let accounted = stats.prime_time + stats.draft_time + stats.verify_time;
    println!(
        "  unaccounted      {:.2}s (vision encode + prefill + sampling)",
        spec_s - accounted.as_secs_f64()
    );
    println!(
        "speedup           {:.2}x  <- sequential arms; see examples/mtp_cost.rs",
        plain_s / spec_s.max(1e-9)
    );
    println!("identical output  {identical}");
    if !identical {
        // Losslessness is the whole premise; a mismatch is a bug, not a
        // tuning result.
        let n = plain
            .token_ids
            .iter()
            .zip(&spec.token_ids)
            .take_while(|(a, b)| a == b)
            .count();
        bail!(
            "speculative output diverged at token {n} — plain {:?} vs spec {:?}",
            plain.token_ids.get(n),
            spec.token_ids.get(n)
        );
    }
    Ok(())
}
