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

//! Greedy generation with Mistral Large 4, routed experts paged from disk.
//!
//! ```text
//! rlx-mistral-large4 --weights …/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf \
//!     (--prompt-ids 1,1784,… | --prompt "…" --tokenizer tokenizer.json) \
//!     [--max-tokens 32] [--ctx 512] [--cache-gb 8] [--device cpu|metal] [--timing]
//! ```

use anyhow::{Context, Result, anyhow, bail};
use rlx_mistral_large4::{PagedMistralLarge4, PagedOptions};
use rlx_runtime::Device;
use std::io::Write;
use std::time::Instant;

struct Args {
    weights: String,
    prompt_ids: Option<Vec<u32>>,
    prompt: Option<String>,
    tokenizer: Option<String>,
    max_tokens: usize,
    ctx: usize,
    cache_gb: f64,
    device: Device,
    timing: bool,
}

const USAGE: &str = "usage: rlx-mistral-large4 --weights GGUF (--prompt-ids IDS | --prompt TEXT --tokenizer tokenizer.json) [--max-tokens N] [--ctx N] [--cache-gb G] [--device D] [--timing]";

fn parse() -> Result<Args> {
    let mut a = Args {
        weights: String::new(),
        prompt_ids: None,
        prompt: None,
        tokenizer: None,
        max_tokens: 32,
        ctx: 512,
        cache_gb: 8.0,
        device: Device::Cpu,
        timing: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| anyhow!("{flag} needs a value\n{USAGE}"))
        };
        match flag.as_str() {
            "--weights" => a.weights = val()?,
            "--prompt-ids" => {
                a.prompt_ids = Some(
                    val()?
                        .split(',')
                        .map(|s| s.trim().parse::<u32>())
                        .collect::<Result<_, _>>()
                        .context("--prompt-ids: comma-separated token ids")?,
                )
            }
            "--prompt" => a.prompt = Some(val()?),
            "--tokenizer" => a.tokenizer = Some(val()?),
            "--max-tokens" => a.max_tokens = val()?.parse()?,
            "--ctx" => a.ctx = val()?.parse()?,
            "--cache-gb" => a.cache_gb = val()?.parse()?,
            "--device" => a.device = rlx_cli::parse_device(&val()?)?,
            "--timing" => a.timing = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown argument {other}\n{USAGE}"),
        }
    }
    if a.weights.is_empty() {
        bail!("--weights is required\n{USAGE}");
    }
    Ok(a)
}

fn main() -> Result<()> {
    let a = parse()?;
    let tok = match &a.tokenizer {
        Some(p) => Some(tokenizers::Tokenizer::from_file(p).map_err(|e| anyhow!("{p}: {e}"))?),
        None => None,
    };
    let prompt: Vec<u32> = match (&a.prompt_ids, &a.prompt, &tok) {
        (Some(ids), _, _) => ids.clone(),
        (None, Some(text), Some(t)) => t
            .encode(text.as_str(), true)
            .map_err(|e| anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec(),
        (None, Some(_), None) => bail!("--prompt needs --tokenizer (or pass --prompt-ids)"),
        (None, None, _) => bail!("pass --prompt-ids or --prompt\n{USAGE}"),
    };

    let t = Instant::now();
    let opts = PagedOptions {
        device: a.device,
        cap: a.ctx,
        cache_bytes: (a.cache_gb * (1u64 << 30) as f64) as u64,
    };
    let mut m = PagedMistralLarge4::open(&a.weights, &opts)?;
    eprintln!(
        "[ml4] {} layers built in {:.1}s; prompt {} tokens",
        m.cfg.num_hidden_layers,
        t.elapsed().as_secs_f64(),
        prompt.len()
    );

    let t = Instant::now();
    let eos = Some(2);
    let mut out = std::io::stdout();
    let out_ids = m.generate_greedy(&prompt, a.max_tokens, eos, |id| {
        match &tok {
            Some(t) => {
                let s = t.decode(&[id], false).unwrap_or_default();
                let _ = write!(out, "{s}");
            }
            None => {
                let _ = write!(out, "{id} ");
            }
        }
        let _ = out.flush();
    })?;
    println!();
    let secs = t.elapsed().as_secs_f64();
    let steps = prompt.len() + out_ids.len();
    eprintln!(
        "[ml4] {} steps in {secs:.1}s ({:.2} s/step)",
        steps,
        secs / steps.max(1) as f64
    );
    if a.timing {
        let s = m.stack().pager_stats();
        eprintln!("[ml4] {:?}", m.stack().times());
        eprintln!(
            "[ml4] pager: {:.1} GB off disk, hit rate {:.1}%",
            s.bytes_read as f64 / 1e9,
            100.0 * s.hit_rate()
        );
    }
    Ok(())
}
