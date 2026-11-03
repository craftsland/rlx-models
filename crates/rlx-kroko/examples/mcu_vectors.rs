// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Reference vectors for the `rlx-kroko-mcu` firmware: the clip as raw i16,
//! the token ids the host computes from exactly that input, and an FNV-1a
//! hash of every chunk's encoder output bits.
//!
//! ```text
//! cargo run -p rlx-kroko --release --example mcu_vectors -- \
//!     --rkr en64-int4.rkr --wav assets/harness/jfk_16k.wav --out crates/rlx-kroko-mcu/data
//! ```

use anyhow::{Context, Result, anyhow};
use rlx_kroko_core as core;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let raw = std::fs::read(get("--rkr").context("--rkr")?)?;
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
    let need = core::scratch_len(&model, pcm.len());
    let mut fs = vec![0f32; need.f32s];
    let mut is = vec![0i8; need.i8s];
    let mut page = vec![0u32; need.page_words];
    let mut ids = vec![0u16; 1024];
    let mut h: u32 = 0x811c_9dc5;
    let n = core::transcribe_with(
        &model,
        &mut core::SlicePager(bytes),
        pcm.as_slice(),
        &mut fs,
        &mut is,
        &mut page,
        &mut ids,
        &mut |enc, _| {
            for v in enc {
                for b in v.to_bits().to_le_bytes() {
                    h = (h ^ b as u32).wrapping_mul(0x0100_0193);
                }
            }
        },
    )
    .map_err(|e| anyhow!("{e}"))?;
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
    std::fs::write(out.join("encoder_fnv.txt"), format!("{h:#010x}\n"))?;
    println!(
        "{} samples, {n} ids, encoder fnv {h:#010x}, scratch {} B: {}",
        pcm.len(),
        need.bytes(),
        String::from_utf8_lossy(&text[..len])
    );
    Ok(())
}
