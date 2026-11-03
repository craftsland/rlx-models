// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Reference vectors for the `rlx-conformer-ctc-mcu` firmware: the clip as
//! raw i16, the token ids the host computes from exactly that i16 input, and
//! an FNV-1a hash of the encoder output's bits. The firmware recomputes all
//! three on the target; equal hashes mean the MCU build is bit-identical.
//!
//! ```text
//! cargo run -p rlx-conformer-ctc --release --example mcu_vectors -- \
//!     --rcc small-int8.rcc --wav clip16k.wav --out crates/rlx-conformer-ctc-mcu/data
//! ```

use anyhow::{Context, Result};
use rlx_conformer_ctc::mcu::AlignedBlob;
use rlx_conformer_ctc::wav;
use rlx_conformer_ctc_core as core;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .with_context(|| format!("{k} is required"))
    };
    let (rcc, wav_path, out) = (get("--rcc")?, get("--wav")?, get("--out")?);
    let blob = AlignedBlob::new(&std::fs::read(rcc)?);
    let model = core::Model::parse(blob.bytes()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let w = wav::parse(&std::fs::read(wav_path)?)?;
    anyhow::ensure!(w.sample_rate == 16_000, "need a 16 kHz clip");
    let pcm: Vec<i16> = w
        .samples
        .iter()
        .map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect();

    let fe = core::Frontend::new(model.config().n_fft);
    let need = core::scratch_len(&model, pcm.len());
    let mut fs = vec![0f32; need.f32s];
    let mut is = vec![0i8; need.i8s];
    let mut ids = vec![0u16; 1024];
    let cfg = model.config();
    let t = cfg.enc_frames(core::Frontend::frames(&model, pcm.len()));
    let mut feats = vec![0f32; t * cfg.d_model];
    let n = core::transcribe_with_features(
        &model,
        &fe,
        pcm.as_slice(),
        &mut fs,
        &mut is,
        &mut ids,
        &mut feats,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut h: u32 = 0x811c_9dc5;
    for v in &feats {
        for b in v.to_bits().to_le_bytes() {
            h = (h ^ b as u32).wrapping_mul(0x0100_0193);
        }
    }
    let mut text = vec![0u8; n * 32 + 1];
    let len = model.decode(&ids[..n], &mut text);

    let out = std::path::Path::new(out);
    std::fs::create_dir_all(out)?;
    let pcm_bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(out.join("pcm_i16.bin"), pcm_bytes)?;
    let id_bytes: Vec<u8> = ids[..n].iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(out.join("ids_u16.bin"), id_bytes)?;
    std::fs::write(out.join("features_fnv.txt"), format!("{h:#010x}\n"))?;
    println!(
        "{} samples, {n} ids, features fnv {h:#010x}, scratch {} B: {}",
        pcm.len(),
        need.bytes(),
        String::from_utf8_lossy(&text[..len])
    );
    Ok(())
}
