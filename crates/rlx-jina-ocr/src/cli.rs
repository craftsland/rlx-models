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

//! `rlx-jina-ocr` CLI — single-image document transcription.

use crate::fixtures::probe_image_path;
use crate::hub::{default_model_dir, resolve_weights_path};
use crate::infer::{InferenceOptions, JinaOcrSession};
use crate::preprocess::ImageMode;
use crate::runner::JinaOcrRunner;
use anyhow::{Context, Result, bail};
use rlx_cli::req;
use rlx_unlimited_ocr::device::resolve_device;
use rlx_unlimited_ocr::lm_precision::LmWeightPrecision;
use std::path::PathBuf;

struct CliArgs {
    weights: Option<PathBuf>,
    image: Option<PathBuf>,
    device: Option<String>,
    max_tokens: usize,
    mode: Option<ImageMode>,
    prompt: Option<String>,
    lm_precision: LmWeightPrecision,
    raw: bool,
    dry: bool,
    list_keys: bool,
    download_only: bool,
}

pub fn run(args: &[String]) -> Result<()> {
    let Some(cli) = parse_args(args)? else {
        return Ok(());
    };
    run_parsed(cli)
}

/// `Ok(None)` after `--help`.
fn parse_args(args: &[String]) -> Result<Option<CliArgs>> {
    let mut weights = None;
    let mut image = None;
    let mut device = None;
    let mut max_tokens = 4096;
    let mut mode = None;
    let mut prompt = None;
    let mut lm_precision = LmWeightPrecision::Auto;
    let mut raw = false;
    let mut dry = false;
    let mut list_keys = false;
    let mut download_only = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "--model-dir" => weights = Some(req(args, &mut i)?.into()),
            "--image" => image = Some(req(args, &mut i)?.into()),
            "--device" => device = Some(req(args, &mut i)?),
            "--max-tokens" => {
                max_tokens = req(args, &mut i)?.parse().context("--max-tokens")?;
            }
            "--mode" => {
                let s = req(args, &mut i)?;
                mode = Some(ImageMode::parse(&s).with_context(|| {
                    format!("--mode: unknown mode {s:?} (expected gundam|native)")
                })?);
            }
            "--prompt" => prompt = Some(req(args, &mut i)?),
            "--lm-precision" => {
                let s = req(args, &mut i)?;
                lm_precision = LmWeightPrecision::parse(&s).with_context(|| {
                    format!("--lm-precision: unknown {s:?} (expected f32|f16|bf16|q8_0|q4_0|auto)")
                })?;
            }
            "--raw" => {
                raw = true;
                i += 1;
            }
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--list-keys" => {
                list_keys = true;
                i += 1;
            }
            "--download" => {
                download_only = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(None);
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    Ok(Some(CliArgs {
        weights,
        image,
        device,
        max_tokens,
        mode,
        prompt,
        lm_precision,
        raw,
        dry,
        list_keys,
        download_only,
    }))
}

fn run_parsed(cli: CliArgs) -> Result<()> {
    if cli.download_only {
        #[cfg(feature = "hf-download")]
        {
            let dir = crate::download::fetch_default()?;
            eprintln!("\nDone. Snapshot:\n  {}", dir.display());
            return Ok(());
        }
        #[cfg(not(feature = "hf-download"))]
        {
            anyhow::bail!(
                "rebuild with --features hf-download, or run:\n  \
                 huggingface-cli download jinaai/jina-ocr-v1"
            );
        }
    }

    let model_dir = match &cli.weights {
        Some(w) => resolve_weights_path(w)?,
        None => default_model_dir()?,
    };
    if cli.weights.is_none() {
        eprintln!("[rlx-jina-ocr] weights {}", model_dir.display());
    }

    let device = resolve_device(cli.device.as_deref())?;

    if cli.dry || cli.list_keys {
        let runner = JinaOcrRunner::open(&model_dir, device)?;
        if cli.list_keys {
            let mut keys: Vec<_> = runner.store().keys().iter().cloned().collect();
            keys.sort();
            for k in keys {
                println!("{k}");
            }
        }
        if cli.dry {
            let cfg = runner.config();
            eprintln!(
                "[rlx-jina-ocr] dry ok — tensors={} hidden={} vocab={} experts={} \
                 sliding_window={} rope_theta={} mtp={} device={device:?}",
                runner.store().keys().len(),
                cfg.lm.hidden_size,
                cfg.lm.vocab_size,
                cfg.lm.n_routed_experts,
                cfg.lm.sliding_window,
                cfg.lm.rope_theta,
                crate::mtp::checkpoint_has_mtp(runner.store()),
            );
        }
        return Ok(());
    }

    #[cfg(not(feature = "tokenizer"))]
    {
        let _ = (cli.mode, cli.prompt, cli.max_tokens, cli.raw, cli.image);
        anyhow::bail!("rebuild with --features tokenizer to run transcription");
    }

    #[cfg(feature = "tokenizer")]
    {
        let mut options = InferenceOptions::for_ocr()
            .device(device)
            .weight_precision(cli.lm_precision)
            .max_new_tokens(cli.max_tokens)
            .render_markdown(!cli.raw);
        if let Some(mode) = cli.mode {
            options = options.mode(mode);
        }
        if let Some(prompt) = cli.prompt {
            options = options.prompt(prompt);
        }
        let mut session = JinaOcrSession::open(&model_dir, options)?;

        let image_path = cli.image.unwrap_or_else(probe_image_path);
        if !image_path.is_file() {
            bail!("image not found: {}", image_path.display());
        }
        let result = session.run_single(&image_path)?;

        println!("{}", result.markdown);
        for crop in &result.crops {
            eprintln!(
                "[rlx-jina-ocr] image {:?} -> {} at {:?}",
                crop.alt_text, crop.filename, crop.pixel_box
            );
        }
        eprintln!(
            "[rlx-jina-ocr] done — {} prompt + {} new tokens",
            result.prompt_len, result.new_tokens
        );
        Ok(())
    }
}

fn print_help() {
    eprintln!(
        "rlx-jina-ocr — jinaai/jina-ocr-v1 (DeepSeek-OCR DeepEncoder + MoE LM)\n\
         \n\
         Weights (optional — HF Hub cache by default):\n\
           [--model-dir PATH]        Dir, `hf`, or Hub id `jinaai/jina-ocr-v1`\n\
         \n\
         Input (default: bundled fixtures/sample.png):\n\
           [--image PATH]            Page/document image\n\
         \n\
         Device & decode:\n\
           [--device auto|cpu|metal|cuda|…]  default: auto (RLX_DEVICE)\n\
           [--lm-precision f32|f16|bf16|q8_0|q4_0|auto]  default: auto\n\
           [--max-tokens N]          default: 4096\n\
           [--mode gundam|native]    default: from processor_config.json (gundam)\n\
           [--prompt TEXT]           default: the card's OCR instruction\n\
           [--raw]                   Skip ref/det -> markdown rendering\n\
         \n\
         Other:\n\
           [--download]              Fetch weights (~6.7 GB) into the HF cache\n\
           [--dry] [--list-keys]\n\
         \n\
         Env: RLX_JINA_OCR_DIR, RLX_JINA_OCR_IMAGE, RLX_DEVICE"
    );
}
