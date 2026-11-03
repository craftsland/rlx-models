// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a checkpoint to the MCU `.rms` blob and compare the int8/int4
//! no_std pipeline with the f32 rlx model on one clip.
//!
//! ```text
//! cargo run -p rlx-moonshine-streaming --release --example mcu_check -- \
//!     --weights weights/asr/moonshine-streaming-tiny --wav assets/harness/jfk_16k.wav \
//!     [--out tiny.rms] [--bits 8] [--vocab-bits 4] [--pos-bits 8] [--group 32]
//! ```

use anyhow::{Context, Result};
use rlx_moonshine_streaming::MoonshineStreaming;
use rlx_moonshine_streaming::mcu::{AlignedBlob, McuExportOptions, export_rms, transcribe_rms};
use rlx_runtime::Device;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let dir = std::path::PathBuf::from(get("--weights").context("--weights DIR")?);
    let wav = get("--wav").context("--wav FILE")?;
    let mut opts = McuExportOptions::default();
    if let Some(b) = get("--bits") {
        opts.bits = b.parse()?;
    }
    if let Some(b) = get("--vocab-bits") {
        opts.vocab_bits = b.parse()?;
    }
    if let Some(b) = get("--pos-bits") {
        opts.pos_bits = b.parse()?;
    }
    if let Some(g) = get("--group") {
        opts.group = g.parse()?;
    }

    let t0 = std::time::Instant::now();
    let blob = export_rms(&dir, opts)?;
    eprintln!(
        "exported {:.2} MB in {:.1} s ({opts:?})",
        blob.len() as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    if let Some(out) = get("--out") {
        std::fs::write(out, &blob)?;
        eprintln!("wrote {out}");
    }
    let blob = AlignedBlob::new(&blob);

    let (samples, rate) = rlx_core::asr_bench::read_wav_mono(std::path::Path::new(wav))?;
    let pcm = if rate == 16_000 {
        samples
    } else {
        rlx_core::audio::resample_linear(&samples, rate, 16_000)
    };

    let model =
        rlx_moonshine_core::Model::parse(blob.bytes()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let need = rlx_moonshine_core::scratch_len(&model, pcm.len());
    if let Some(dir) = get("--dump") {
        // Encoder output after 0..=layers layers, raw little-endian f32.
        let (_, _, t) = model.config().lengths(pcm.len());
        let mut out = vec![0f32; t * model.config().d_enc];
        for l in 0..=model.config().enc_layers {
            let mut fs = vec![0f32; need.f32s];
            let mut is = vec![0i8; need.i8s];
            rlx_moonshine_core::encode_partial(
                &model,
                pcm.as_slice(),
                l,
                &mut fs,
                &mut is,
                &mut out,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/mcu_{l}.f32"), bytes)?;
        }
        eprintln!("dumped {} layers to {dir}", model.config().enc_layers + 1);
    }
    let t1 = std::time::Instant::now();
    let (ids, text, hidden) = transcribe_rms(blob.bytes(), &pcm)?;
    eprintln!(
        "mcu: {:.0} ms, scratch {:.1} KiB, {} tokens",
        t1.elapsed().as_secs_f64() * 1e3,
        need.bytes() as f64 / 1024.0,
        ids.len()
    );
    println!("mcu: {text}");

    let mut asr = MoonshineStreaming::open(&dir, Device::Cpu)?;
    let enc = asr.encode(&pcm)?;
    if let Some(dir) = get("--dump") {
        let bytes: Vec<u8> = enc.hidden.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/rlx_hidden.f32"), bytes)?;
    }
    let seq = asr.generate(
        &enc,
        ((pcm.len() as f64 / 16_000.0) * 6.5).ceil() as usize + 8,
    )?;
    let ref_ids: Vec<u16> = seq[1..]
        .iter()
        .copied()
        .filter(|&t| t != 2)
        .map(|t| t as u16)
        .collect();
    println!("f32: {}", asr.transcribe(&pcm, None)?);
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in enc.hidden.iter().zip(&hidden) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    println!(
        "encoder cosine {:.5}; token ids {}",
        dot / (na.sqrt() * nb.sqrt()),
        if ref_ids == ids {
            "identical".to_string()
        } else {
            format!("differ ({} vs {})", ids.len(), ref_ids.len())
        }
    );
    Ok(())
}
