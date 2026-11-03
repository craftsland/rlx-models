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

//! `rlx-conformer-ctc` CLI entry points.
//!
//! Commands:
//! - `transcribe --nemo <file> --wav <file> [--device …] [--warm]`
//! - `dump-keys  --nemo <file>`
//! - `export-mcu --nemo <file> --out <model.rcc>` — int8 blob for MCUs
//! - `transcribe-mcu --rcc <model.rcc> --wav <file> [--nemo <file>]` — run
//!   the MCU pipeline on the host; with `--nemo`, compare to the f32 model

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use rlx_cli::{parse_standard_device, req};
use rlx_nemo::NemoModel;
use rlx_runtime::Device;

use crate::config::AsrConfig;
use crate::{ConformerCtc, wav};

/// Dispatch a CLI invocation (`args[0]` is the subcommand).
pub fn run(args: &[String]) -> Result<()> {
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    match cmd {
        "transcribe" => transcribe(&args[1..]),
        "dump-keys" => dump_keys(&args[1..]),
        "export-mcu" => export_mcu(&args[1..]),
        "transcribe-mcu" => transcribe_mcu(&args[1..]),
        "help" | "-h" | "--help" => {
            print_usage();
            Ok(())
        }
        other => {
            eprintln!("unknown command {other:?}\n");
            print_usage();
            bail!("unknown command {other:?}");
        }
    }
}

fn print_usage() {
    eprintln!(
        "rlx-conformer-ctc — NVIDIA NeMo Conformer-CTC ASR on RLX\n\n\
         USAGE:\n  \
         rlx-conformer-ctc transcribe --nemo <file.nemo> --wav <audio.wav> \\\n    \
           [--device cpu|metal|mlx|cuda|gpu|vulkan|rocm] [--warm]\n  \
         rlx-conformer-ctc dump-keys  --nemo <file.nemo>\n  \
         rlx-conformer-ctc export-mcu --nemo <file.nemo> --out <model.rcc>\n  \
         rlx-conformer-ctc transcribe-mcu --rcc <model.rcc> --wav <audio.wav> [--nemo <file.nemo>]\n\n\
         --warm  run twice; print cold (compile+run) vs warm (cached) ms\n\n\
         Model: https://huggingface.co/nvidia/stt_en_conformer_ctc_small\n"
    );
}

struct Args {
    nemo: PathBuf,
    wav: Option<PathBuf>,
    device: String,
    warm: bool,
}

fn parse_args(args: &[String]) -> Result<Args> {
    let mut nemo: Option<PathBuf> = None;
    let mut wav: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut warm = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--nemo" => nemo = Some(PathBuf::from(req(args, &mut i)?)),
            "--wav" => wav = Some(PathBuf::from(req(args, &mut i)?)),
            "--device" => device = req(args, &mut i)?,
            "--warm" => {
                warm = true;
                i += 1;
            }
            other => bail!("unexpected argument {other:?}"),
        }
    }
    let nemo = nemo.ok_or_else(|| anyhow::anyhow!("--nemo <file.nemo> is required"))?;
    Ok(Args {
        nemo,
        wav,
        device,
        warm,
    })
}

fn transcribe(args: &[String]) -> Result<()> {
    let a = parse_args(args)?;
    let wav_path = a
        .wav
        .ok_or_else(|| anyhow::anyhow!("--wav <audio.wav> is required"))?;
    let device: Device = parse_standard_device("conformer-ctc", &a.device)?;

    let mut asr = ConformerCtc::open(&a.nemo, device)?;
    let cfg = asr.config().clone();

    let bytes = std::fs::read(&wav_path)?;
    let w = wav::parse(&bytes)?;
    let pcm = wav::resample(&w.samples, w.sample_rate, cfg.sample_rate as u32);
    eprintln!(
        "[rlx-conformer-ctc] device={device:?} samples={} ({} Hz -> {} Hz) d_model={} layers={}",
        pcm.len(),
        w.sample_rate,
        cfg.sample_rate,
        cfg.d_model,
        cfg.n_layers
    );

    let t0 = Instant::now();
    let text = asr.transcribe(&pcm)?;
    let cold_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if a.warm {
        let t1 = Instant::now();
        let text2 = asr.transcribe(&pcm)?;
        let warm_ms = t1.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[rlx-conformer-ctc] cold={cold_ms:.1} ms  warm={warm_ms:.1} ms  cached={}",
            asr.cached_encoder_count()
        );
        if text2 != text {
            bail!("warm transcript diverged from cold:\n  cold={text}\n  warm={text2}");
        }
    } else {
        eprintln!(
            "[rlx-conformer-ctc] {cold_ms:.1} ms (cached={})",
            asr.cached_encoder_count()
        );
    }

    println!("{text}");
    Ok(())
}

fn dump_keys(args: &[String]) -> Result<()> {
    let a = parse_args(args)?;
    let model = NemoModel::open(&a.nemo)?;
    let cfg = AsrConfig::from_nemo(model.config())?;
    eprintln!(
        "config: d_model={} layers={} heads={} ff={} subsample={} ({}) vocab={} blank={} classes={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.ff_dim(),
        cfg.subsampling_factor,
        cfg.subsampling,
        cfg.vocab_size,
        cfg.blank_id,
        cfg.num_classes,
    );
    eprintln!("tokenizer artifacts:");
    for t in model.tokenizers() {
        eprintln!("  {} ({} bytes)", t.name, t.bytes.len());
    }
    println!("# {} tensors", model.len());
    for name in model.names() {
        let shape = model.shape_of(&name).unwrap_or(&[]);
        println!("{name}\t{shape:?}");
    }
    Ok(())
}

/// `--key value` pairs, every flag required to take a value.
fn kv(args: &[String], known: &[&str]) -> Result<std::collections::HashMap<String, String>> {
    let mut m = std::collections::HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let k = args[i].as_str();
        if !known.contains(&k) {
            bail!("unexpected argument {k:?}");
        }
        m.insert(k.to_string(), req(args, &mut i)?);
    }
    Ok(m)
}

fn export_mcu(args: &[String]) -> Result<()> {
    let m = kv(args, &["--nemo", "--out"])?;
    let nemo = m
        .get("--nemo")
        .ok_or_else(|| anyhow::anyhow!("--nemo is required"))?;
    let out = m
        .get("--out")
        .ok_or_else(|| anyhow::anyhow!("--out is required"))?;
    let n = crate::mcu::export_rcc_file(nemo.as_ref(), out.as_ref())?;
    eprintln!("[rlx-conformer-ctc] wrote {out} ({:.2} MB)", n as f64 / 1e6);
    Ok(())
}

fn transcribe_mcu(args: &[String]) -> Result<()> {
    let m = kv(args, &["--rcc", "--wav", "--nemo"])?;
    let rcc = m
        .get("--rcc")
        .ok_or_else(|| anyhow::anyhow!("--rcc is required"))?;
    let wav_path = m
        .get("--wav")
        .ok_or_else(|| anyhow::anyhow!("--wav is required"))?;
    let blob = crate::mcu::AlignedBlob::new(&std::fs::read(rcc)?);
    let model = rlx_conformer_ctc_core::Model::parse(blob.bytes())
        .map_err(|e| anyhow::anyhow!("{rcc}: {e}"))?;
    let sr = model.config().sample_rate as u32;
    let w = wav::parse(&std::fs::read(wav_path)?)?;
    let pcm = wav::resample(&w.samples, w.sample_rate, sr);
    let need = rlx_conformer_ctc_core::scratch_len(&model, pcm.len());
    eprintln!(
        "[rlx-conformer-ctc] mcu int8: {} samples, scratch {:.1} KiB",
        pcm.len(),
        need.bytes() as f64 / 1024.0
    );

    let mut feats = Vec::new();
    let t0 = Instant::now();
    let ids = crate::mcu::transcribe_rcc(blob.bytes(), &pcm, Some(&mut feats))?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ids16: Vec<u16> = ids.iter().map(|&i| i as u16).collect();
    let mut text = vec![0u8; ids.len() * 32 + 1];
    let n = model.decode(&ids16, &mut text);
    eprintln!("[rlx-conformer-ctc] mcu int8 {ms:.1} ms");
    println!("{}", String::from_utf8_lossy(&text[..n]));

    if let Some(nemo) = m.get("--nemo") {
        let mut asr = ConformerCtc::open(nemo.as_ref(), Device::Cpu)?;
        let (enc, t) = asr.encode_unpadded(&pcm)?;
        let ref_ids = asr.ctc_ids(&enc, t)?;
        let (mut dot, mut na, mut nb, mut maxd) = (0f64, 0f64, 0f64, 0f32);
        for (&a, &b) in enc.iter().zip(&feats) {
            dot += a as f64 * b as f64;
            na += a as f64 * a as f64;
            nb += b as f64 * b as f64;
            maxd = maxd.max((a - b).abs());
        }
        eprintln!(
            "[rlx-conformer-ctc] vs f32: encoder cosine {:.5}, max|Δ| {maxd:.4}, ids {}",
            dot / (na.sqrt() * nb.sqrt()),
            if ref_ids == ids {
                "identical"
            } else {
                "differ"
            }
        );
        println!("f32: {}", asr.decode_text(&ref_ids));
    }
    Ok(())
}
