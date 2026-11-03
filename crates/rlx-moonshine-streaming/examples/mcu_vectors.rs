// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Reference vectors for the `rlx-moonshine-mcu` firmware: the clip as raw
//! i16, the token ids the host computes from exactly that input, and an
//! FNV-1a hash of the encoder output's bits.
//!
//! ```text
//! cargo run -p rlx-moonshine-streaming --release --example mcu_vectors -- \
//!     --rms tiny.rms --wav assets/harness/jfk_16k.wav --out crates/rlx-moonshine-mcu/data
//! ```

use anyhow::{Context, Result, anyhow};
use rlx_moonshine_core as core;
use rlx_moonshine_streaming::mcu::AlignedBlob;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let blob = AlignedBlob::new(&std::fs::read(get("--rms").context("--rms")?)?);
    let (pcm, rate) =
        rlx_core::asr_bench::read_wav_mono(std::path::Path::new(get("--wav").context("--wav")?))?;
    anyhow::ensure!(rate == 16_000, "need a 16 kHz clip");
    let pcm: Vec<i16> = pcm
        .iter()
        .map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect();
    let model = core::Model::parse(blob.bytes()).map_err(|e| anyhow!("{e}"))?;
    let need = core::scratch_len(&model, pcm.len());
    let (_, _, t) = model.config().lengths(pcm.len());
    let mut fs = vec![0f32; need.f32s];
    let mut is = vec![0i8; need.i8s];
    let mut ids = vec![0u16; model.config().max_tokens(pcm.len())];
    let mut hidden = vec![0f32; t * model.config().d_enc];
    let n = core::transcribe_with_hidden(
        &model,
        pcm.as_slice(),
        &mut fs,
        &mut is,
        &mut ids,
        &mut hidden,
    )
    .map_err(|e| anyhow!("{e}"))?;
    let mut h: u32 = 0x811c_9dc5;
    for v in &hidden {
        for b in v.to_bits().to_le_bytes() {
            h = (h ^ b as u32).wrapping_mul(0x0100_0193);
        }
    }
    let mut text = vec![0u8; n * 32 + 1];
    let len = model.decode(&ids[..n], &mut text);
    let out = std::path::Path::new(get("--out").context("--out")?);
    std::fs::create_dir_all(out)?;
    std::fs::write(
        out.join("pcm_i16.bin"),
        pcm.iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    std::fs::write(
        out.join("ids_u16.bin"),
        ids[..n]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    std::fs::write(out.join("hidden_fnv.txt"), format!("{h:#010x}\n"))?;
    println!(
        "{} samples, {n} ids, hidden fnv {h:#010x}, scratch {} B: {}",
        pcm.len(),
        need.bytes(),
        String::from_utf8_lossy(&text[..len])
    );
    Ok(())
}
