// RLX — versatile ML compiler + runtime. GPLv3.
//! `rlx-dsv41` CLI — text generation with DeepSeek-V4.1-Flash.
//!
//! Points `V41Runner` at a checkpoint directory (`config.json`,
//! `*.safetensors`, `tokenizer.json`) and prints what it generates.
//!
//! The flag worth knowing is `--paged`, which routes on the host and keeps only
//! the experts a token actually uses in memory. The released checkpoint's expert
//! banks are ~34 GB per layer as f32, so `--paged` is not an optimization there,
//! it is the difference between running and not.

use anyhow::{Context, Result, bail};
use rlx_cli::{parse_llama32_device, req};
use rlx_core::dsv41_pager::PagerStats;
use rlx_core::dsv41_quant::DEFAULT_BLOCK;
use rlx_core::dsv41_runner::{Generation, MoeExecution, RunnerOptions, SampleOpts, V41Runner};
use std::path::PathBuf;

const HELP: &str = "\
rlx-dsv41 — DeepSeek-V4.1-Flash (CSA2 + Engram + hyper-connections + MoE)

USAGE:
  rlx-dsv41 --model <DIR> --prompt <TEXT> [options]
  rlx-dsv41 --model <DIR> --prompt-ids 5,9,2,... [options]

OPTIONS:
  --model, --weights <PATH>  Checkpoint dir: config.json + *.safetensors
                             (+ tokenizer.json for text prompts).
  --prompt <TEXT>            Prompt text.
  --prompt-ids <a,b,...>     Raw token ids instead of --prompt.
  --device <cpu|metal|mlx|cuda|...>   Execution device (default cpu).
  --max-tokens <N>           New tokens to generate (default 64).
  --temperature <F>          Sampling temperature (default 0 = greedy).
  --top-k <N>                Keep only the N most likely tokens (default off).
  --top-p <F>                Nucleus sampling top-p (default 1 = off).
  --seed <N>                 Sampling seed (default 0).
  --stop <a,b,...>           Token ids that end generation.
  --paged                    Route on the host and page experts in on demand.
                             Required for any checkpoint whose expert banks do
                             not fit in memory.
  --expert-budget <BYTES>    Resident expert cache cap with --paged
                             (default 8G). Accepts K / M / G suffixes.
  --block <N>                quantization_config.weight_block_size[0] (default 32).
  --ids                      Print generated token ids as well as text.
  --stats                    Print timing and, with --paged, pager counters.
  -h, --help                 Show this help.
";

/// Parse a byte count, accepting `K`/`M`/`G` suffixes.
fn bytes(s: &str) -> Result<u64> {
    let t = s.trim();
    let (num, mul) = match t.chars().last() {
        Some('K' | 'k') => (&t[..t.len() - 1], 1u64 << 10),
        Some('M' | 'm') => (&t[..t.len() - 1], 1u64 << 20),
        Some('G' | 'g') => (&t[..t.len() - 1], 1u64 << 30),
        _ => (t, 1),
    };
    let n: u64 = num
        .trim()
        .parse()
        .with_context(|| format!("--expert-budget: expected a byte count, got `{s}`"))?;
    Ok(n * mul)
}

fn parse_ids(s: &str) -> Result<Vec<u32>> {
    s.split(',')
        .map(|t| t.trim().parse::<u32>().context("expected a u32"))
        .collect()
}

pub fn run(args: &[String]) -> Result<()> {
    let mut model: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut prompt: Option<String> = None;
    let mut prompt_ids: Option<Vec<u32>> = None;
    let mut sample = SampleOpts::greedy(64);
    let mut moe = MoeExecution::Resident;
    let mut expert_budget = 8u64 << 30;
    let mut block = DEFAULT_BLOCK;
    let mut show_ids = false;
    let mut stats = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "--weights" => model = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--prompt" => prompt = Some(req(args, &mut i)?),
            "--prompt-ids" => {
                prompt_ids = Some(parse_ids(&req(args, &mut i)?).context("--prompt-ids")?);
            }
            "--max-tokens" => {
                sample.max_new_tokens =
                    req(args, &mut i)?.parse().context("--max-tokens: usize")?;
            }
            "--temperature" => {
                sample.temperature = req(args, &mut i)?.parse().context("--temperature: f32")?;
            }
            "--top-k" => sample.top_k = req(args, &mut i)?.parse().context("--top-k: usize")?,
            "--top-p" => sample.top_p = req(args, &mut i)?.parse().context("--top-p: f32")?,
            "--seed" => sample.seed = req(args, &mut i)?.parse().context("--seed: u64")?,
            "--stop" => sample.stop = parse_ids(&req(args, &mut i)?).context("--stop")?,
            "--paged" => {
                moe = MoeExecution::Paged;
                i += 1;
            }
            "--expert-budget" => expert_budget = bytes(&req(args, &mut i)?)?,
            "--block" => block = req(args, &mut i)?.parse().context("--block: usize")?,
            "--ids" => {
                show_ids = true;
                i += 1;
            }
            "--stats" => {
                stats = true;
                i += 1;
            }
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            other => bail!("rlx-dsv41: unknown argument `{other}`\n\n{HELP}"),
        }
    }

    let Some(model) = model else {
        bail!("rlx-dsv41: --model is required\n\n{HELP}");
    };
    if prompt.is_none() && prompt_ids.is_none() {
        bail!("rlx-dsv41: one of --prompt or --prompt-ids is required\n\n{HELP}");
    }

    let opts = RunnerOptions {
        device: parse_llama32_device(&device)?,
        moe,
        expert_budget_bytes: expert_budget,
        block,
        ..Default::default()
    };
    let mut runner = V41Runner::open(&model, opts)
        .with_context(|| format!("opening the checkpoint at {model:?}"))?;

    let started = std::time::Instant::now();
    let out = match (&prompt, &prompt_ids) {
        (Some(p), _) => runner.generate(p, &sample)?,
        (None, Some(v)) => runner.generate_ids(v, &sample)?,
        (None, None) => unreachable!("checked above"),
    };
    let elapsed = started.elapsed();

    if show_ids || out.text.is_empty() {
        println!("{:?}", out.tokens);
    }
    if !out.text.is_empty() {
        println!("{}", out.text);
    }
    if let Some(t) = out.stopped_on {
        eprintln!("(stopped on token {t})");
    }
    if stats {
        report(&out, elapsed, runner.pager_stats());
    }
    Ok(())
}

fn report(out: &Generation, elapsed: std::time::Duration, pager: Option<PagerStats>) {
    let secs = elapsed.as_secs_f64().max(f64::EPSILON);
    eprintln!(
        "\n{} prompt + {} new tokens in {:.2}s ({:.2} tok/s)",
        out.prompt_tokens,
        out.tokens.len(),
        secs,
        out.tokens.len() as f64 / secs
    );
    if let Some(p) = pager {
        eprintln!(
            "pager: {} hits / {} misses ({:.0}% hit rate), {} evictions, \
             {:.1} MiB read, {:.1} MiB resident",
            p.hits,
            p.misses,
            p.hit_rate() * 100.0,
            p.evictions,
            p.bytes_read as f64 / (1 << 20) as f64,
            p.resident_bytes as f64 / (1 << 20) as f64,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{bytes, parse_ids, run};

    /// Byte suffixes, because an expert budget is naturally written `16G` and
    /// silently reading that as 16 bytes would evict on every single access.
    #[test]
    fn byte_suffixes_scale() {
        assert_eq!(bytes("1024").unwrap(), 1024);
        assert_eq!(bytes("4K").unwrap(), 4 << 10);
        assert_eq!(bytes("512M").unwrap(), 512 << 20);
        assert_eq!(bytes("16G").unwrap(), 16 << 30);
        assert_eq!(bytes("16g").unwrap(), 16 << 30);
        assert_eq!(bytes(" 2G ").unwrap(), 2 << 30);
    }

    #[test]
    fn a_bad_byte_count_names_the_flag() {
        let e = bytes("lots").unwrap_err().to_string();
        assert!(e.contains("--expert-budget"), "unhelpful: {e}");
    }

    #[test]
    fn ids_parse_with_spaces() {
        assert_eq!(parse_ids("1, 2,3 ").unwrap(), vec![1, 2, 3]);
        assert!(parse_ids("1,x").is_err());
    }

    /// Missing required arguments fail with the help text rather than a panic
    /// or a confusing downstream error.
    #[test]
    fn missing_arguments_are_reported() {
        let e = run(&["--prompt".into(), "hi".into()])
            .unwrap_err()
            .to_string();
        assert!(e.contains("--model is required"), "{e}");
        let e = run(&["--model".into(), "/nope".into()])
            .unwrap_err()
            .to_string();
        assert!(e.contains("--prompt"), "{e}");
        let e = run(&["--bogus".into()]).unwrap_err().to_string();
        assert!(e.contains("unknown argument `--bogus`"), "{e}");
    }

    /// `--help` is not an error.
    #[test]
    fn help_succeeds() {
        assert!(run(&["--help".into()]).is_ok());
    }
}
