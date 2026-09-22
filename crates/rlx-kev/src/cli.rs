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

//! `rlx-kev` — answer a System One request from the command line.
//!
//! ```text
//! rlx-kev --run runs/kev-4b --base models/Qwen3.5-4B-Base --request ticket.json
//! rlx-kev --run runs/kev-4b --encode-only --request ticket.json
//! rlx-kev --run runs/kev-4b --info
//! ```
//!
//! `--encode-only` needs no base model and no weights: it dumps the packed
//! encoding as JSON, which is the artifact to diff against `kev.model.encode`
//! when checking the token protocol.

use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::api::{SystemOneRequest, answers_json, to_record};

/// Parsed command line.
#[derive(Debug, Clone)]
pub struct Args {
    pub run: PathBuf,
    pub base: Option<PathBuf>,
    pub request: Option<String>,
    pub device: String,
    pub temperature: Option<f32>,
    pub fixed: Option<(usize, usize)>,
    pub max_rows: usize,
    pub max_seq: usize,
    pub encode_only: bool,
    pub info: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            run: PathBuf::new(),
            base: None,
            request: None,
            device: "cpu".into(),
            temperature: None,
            fixed: None,
            max_rows: 32,
            max_seq: 2048,
            encode_only: false,
            info: false,
        }
    }
}

const USAGE: &str = "\
rlx-kev — Jev-style decision models (Qwen3.5 + LoRA + pointer head)

USAGE:
  rlx-kev --run <DIR> [--base <DIR>] [--request <FILE|->] [OPTIONS]

REQUIRED:
  --run <DIR>          kev checkpoint: head.pt, adapter, tokenizer.json

OPTIONS:
  --base <DIR>         Qwen3.5-*-Base snapshot (config.json + safetensors).
                       Required unless --encode-only / --info.
  --request <FILE|->   POST /v1/systemone body; `-` reads stdin.
  --device <NAME>      cpu (default), metal, mlx, cuda, rocm, gpu, vulkan
  --temperature <T>    override the checkpoint calibration; 1.0 = raw logits
  --fixed <R,S>        pin the compiled shape to R rows x S tokens and
                       release the base weights after building
  --max-rows <N>       bucketed-policy ceiling on questions (default 32)
  --max-seq <N>        bucketed-policy ceiling on tokens (default 2048)
  --encode-only        print the packed encoding as JSON and exit
  --info               print checkpoint metadata and exit
  -h, --help           this text
";

/// Parse `argv` (without the program name).
pub fn parse(argv: &[String]) -> Result<Option<Args>> {
    let mut a = Args::default();
    let mut it = argv.iter().peekable();
    let mut saw_run = false;

    let need = |it: &mut std::iter::Peekable<std::slice::Iter<'_, String>>,
                flag: &str|
     -> Result<String> {
        it.next()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
    };

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "--run" => {
                a.run = PathBuf::from(need(&mut it, "--run")?);
                saw_run = true;
            }
            "--base" => a.base = Some(PathBuf::from(need(&mut it, "--base")?)),
            "--request" => a.request = Some(need(&mut it, "--request")?),
            "--device" => a.device = need(&mut it, "--device")?,
            "--temperature" => {
                a.temperature = Some(need(&mut it, "--temperature")?.parse().context("--temperature")?)
            }
            "--fixed" => {
                let v = need(&mut it, "--fixed")?;
                let (r, s) = v
                    .split_once(',')
                    .ok_or_else(|| anyhow::anyhow!("--fixed wants R,S (got {v:?})"))?;
                a.fixed = Some((r.trim().parse().context("--fixed rows")?, s.trim().parse().context("--fixed seq")?));
            }
            "--max-rows" => a.max_rows = need(&mut it, "--max-rows")?.parse().context("--max-rows")?,
            "--max-seq" => a.max_seq = need(&mut it, "--max-seq")?.parse().context("--max-seq")?,
            "--encode-only" => a.encode_only = true,
            "--info" => a.info = true,
            other => bail!("unknown argument {other:?}\n\n{USAGE}"),
        }
    }
    if !saw_run {
        bail!("--run is required\n\n{USAGE}");
    }
    Ok(Some(a))
}

fn read_request(spec: &str) -> Result<SystemOneRequest> {
    let text = if spec == "-" {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .context("reading the request from stdin")?;
        s
    } else {
        std::fs::read_to_string(spec).with_context(|| format!("reading {spec}"))?
    };
    serde_json::from_str(&text).context("parsing the System One request")
}

/// Entry point for the `rlx-kev` binary.
pub fn main(argv: &[String]) -> Result<()> {
    let Some(args) = parse(argv)? else {
        return Ok(());
    };
    run(&args)
}

#[cfg(not(feature = "tokenizer"))]
fn run(_args: &Args) -> Result<()> {
    bail!("rlx-kev was built without the `tokenizer` feature; the CLI needs it")
}

#[cfg(feature = "tokenizer")]
fn run(args: &Args) -> Result<()> {
    use crate::checkpoint::Checkpoint;
    use crate::encode::{EncodeOpts, Encoder, HfTokenizer};
    use crate::model::{INFER_MAX_BRANCH, INFER_MAX_STATE, ShapePolicy};

    let mut ckpt = Checkpoint::open(&args.run)?;
    if let Some(t) = args.temperature {
        ckpt.set_temperature(t)?;
    }

    if args.info {
        let m = ckpt.meta();
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "run": args.run.display().to_string(),
            "base": m.base,
            "base_revision": m.base_revision,
            "lora": m.lora,
            "head_dim": m.head_dim,
            "option_isolation": m.option_isolation,
            "special_embeddings": m.special_embeddings,
            "weights_dtype": m.weights_dtype,
            "temperature": m.temperature,
        }))?);
        return Ok(());
    }

    let spec = args
        .request
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--request is required (or use --info)"))?;
    let req = read_request(spec)?;

    if args.encode_only {
        let tok = HfTokenizer::from_file(&ckpt.tokenizer_path())?;
        let encoder = Encoder::new(tok)?;
        let (rec, _meta) = to_record(&req)?;
        let enc = encoder.encode(
            &rec,
            EncodeOpts {
                max_state: INFER_MAX_STATE,
                max_branch: INFER_MAX_BRANCH,
                strict: false,
                option_isolation: ckpt.meta().option_isolation,
            },
        )?;
        println!("{}", serde_json::to_string(&serde_json::json!({
            "ids": enc.ids,
            "seg": enc.seg,
            "pos": enc.pos,
            "opt": enc.opt,
            "decide_idx": enc.decide_idx,
            "opt_idx": enc.opt_idx,
            "state_truncated": enc.state_truncated,
        }))?);
        return Ok(());
    }

    let base = args
        .base
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--base is required to run the model"))?;
    let device = rlx_runtime::parse_device(&args.device)
        .map_err(|e| anyhow::anyhow!("--device {}: {e}", args.device))?;
    let policy = match args.fixed {
        Some((rows, seq)) => ShapePolicy::Fixed { rows, seq },
        None => ShapePolicy::Bucketed {
            max_rows: args.max_rows,
            max_seq: args.max_seq,
        },
    };

    let started = std::time::Instant::now();
    let mut model = crate::load(&args.run, base, device, policy)?;
    let load_ms = started.elapsed().as_secs_f64() * 1e3;

    let t0 = std::time::Instant::now();
    let (answers, usage) = model.answer(&req)?;
    let latency_ms = t0.elapsed().as_secs_f64() * 1e3;

    let body = serde_json::json!({
        "model": req.model,
        "answers": answers_json(&answers),
        "usage": {
            "input_tokens": usage.input_tokens,
            "state_tokens": usage.state_tokens,
        },
        "latency_ms": (latency_ms * 10.0).round() / 10.0,
        "load_ms": (load_ms * 10.0).round() / 10.0,
        "shape": model.compiled_shape().map(|(r, s)| vec![r, s]),
    });
    println!("{}", serde_json::to_string_pretty(&body)?);
    Ok(())
}
