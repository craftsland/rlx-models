// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Reference vectors for the `rlx-citrinet-mcu` firmware: the clip as raw
//! i16, the token ids the host computes from exactly that input, and an
//! FNV-1a hash of the encoder output's bits.
//!
//! ```text
//! cargo run -p rlx-citrinet --release --example mcu_vectors -- \
//!     --rct c256-int8.rct --wav assets/harness/jfk_16k.wav --out crates/rlx-citrinet-mcu/data
//! ```
//! `--f32s N --i8s M`: the target's scratch. A clip whose scratch does not fit
//! is cut at pauses (`next_cut`, 25 ms windows) into pieces of at most
//! `max_samples` samples, written to `max_samples.txt`; ids are concatenated
//! and the hash runs over every piece's encoder output in order.

use anyhow::{Context, Result, anyhow};
use rlx_citrinet_core as core;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let raw = std::fs::read(get("--rct").context("--rct")?)?;
    let words: Vec<u32> = raw
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect();
    // SAFETY: u32 storage viewed as bytes.
    let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), raw.len()) };
    let (pcm, rate) =
        rlx_core::asr_bench::read_wav_mono(std::path::Path::new(get("--wav").context("--wav")?))?;
    anyhow::ensure!(rate == 16_000, "need a 16 kHz clip");
    let pcm: Vec<i16> = pcm
        .iter()
        .map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect();
    let model = core::Model::parse(bytes).map_err(|e| anyhow!("{e}"))?;
    let fe = core::Frontend::new(model.config().n_fft);
    let num = |k: &str| get(k).map(|v| v.parse::<usize>()).transpose();
    let max_len = match (num("--f32s")?, num("--i8s")?) {
        (Some(f), Some(i)) => core::max_samples(&model, f, i),
        _ => pcm.len(),
    };
    let mut ids = vec![0u16; pcm.len() / 160 + 1];
    let (mut n, mut start, mut pieces, mut peak) = (0, 0, 0, 0);
    let mut h: u32 = 0x811c_9dc5;
    while start < pcm.len() {
        let end = core::next_cut(pcm.as_slice(), start, max_len, 400);
        let piece = &pcm[start..end];
        let need = core::scratch_len(&model, piece.len());
        peak = peak.max(need.bytes());
        let mut fs = vec![0f32; need.f32s];
        let mut is = vec![0i8; need.i8s];
        n += core::transcribe(&model, &fe, piece, &mut fs, &mut is, &mut ids[n..])
            .map_err(|e| anyhow!("{e}"))?;
        for v in core::encoder_output(&model, piece.len(), &fs) {
            for b in v.to_bits().to_le_bytes() {
                h = (h ^ b as u32).wrapping_mul(0x0100_0193);
            }
        }
        println!("piece {pieces}: samples {start}..{end}");
        pieces += 1;
        start = end;
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
    std::fs::write(out.join("features_fnv.txt"), format!("{h:#010x}\n"))?;
    if max_len < pcm.len() {
        std::fs::write(out.join("max_samples.txt"), format!("{max_len}\n"))?;
    }
    println!(
        "{} samples in {pieces} piece(s), {n} ids, features fnv {h:#010x}, peak scratch {peak} B: {}",
        pcm.len(),
        String::from_utf8_lossy(&text[..len])
    );
    Ok(())
}
