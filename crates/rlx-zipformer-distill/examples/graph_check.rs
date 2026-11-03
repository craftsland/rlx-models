// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Check the RLX student graph against `rlx-kroko-core`: build it with the
//! teacher's own config and weights, run `K` chunks of a clip's fbank, and
//! compare with the core's encoder output on the same features (the core
//! runs the int8 `.rkr`, so expect cosine ≈ 0.998, not bit equality).
//!
//! ```text
//! cargo run -p rlx-zipformer-distill --release --example graph_check -- \
//!     --teacher .cache/zipformer-m/native --rkr .cache/zipformer-m/m-int8.rkr \
//!     [--wav assets/harness/jfk_16k.wav] [--chunks 6] [--device cpu|metal]
//! ```

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rlx_kroko_core as core;
use rlx_zipformer_distill::config::{N_MELS, OUT_FRAMES, StudentConfig};
use rlx_zipformer_distill::{init::Teacher, model};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let teacher_dir = Path::new(get("--teacher").context("--teacher")?);
    let teacher = Teacher::open(teacher_dir)?;
    let k: usize = get("--chunks").map(|v| v.parse()).transpose()?.unwrap_or(6);
    let device = match get("--device").map(String::as_str) {
        Some("metal") => rlx_tensor::Device::Metal,
        _ => rlx_tensor::Device::Cpu,
    };

    // Features as the core computes them.
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
    let rkr = core::Model::parse(bytes).map_err(|e| anyhow!("{e}"))?;
    let wav = get("--wav").map_or("assets/harness/jfk_16k.wav", String::as_str);
    let (pcm, _) = rlx_core::asr_bench::read_wav_mono(Path::new(wav))?;
    let mut fb = vec![0f32; core::fbank_frames(pcm.len()) * N_MELS];
    let frames = core::fbank_features(&rkr, pcm.as_slice(), &mut fb);
    let seg = StudentConfig::segment_frames(k);
    anyhow::ensure!(frames >= seg, "clip too short for {k} chunks");
    let feats = &fb[..seg * N_MELS];

    // Core.
    let need = core::scratch_len(&rkr, seg * 160);
    let (mut fs, mut is, mut page) = (
        vec![0f32; need.f32s],
        vec![0i8; need.i8s],
        vec![0u32; need.page_words],
    );
    let mut want: Vec<f32> = Vec::new();
    core::encode_features(
        &rkr,
        &mut core::SlicePager(bytes),
        feats,
        seg,
        &mut fs,
        &mut is,
        &mut page,
        &mut |enc, _| want.extend_from_slice(enc),
    )
    .map_err(|e| anyhow!("{e}"))?;
    want.truncate(k * OUT_FRAMES * 512);

    // Graph with the teacher's config and weights.
    let t0 = std::time::Instant::now();
    let f = model::build_encoder(&teacher.cfg, 1, k);
    let f = f.init_params(|name, dims| {
        teacher
            .init(&teacher.cfg, name, dims)
            .unwrap_or_else(|e| panic!("{e}"))
    });
    let got = f.run_on(device, &[("feats", feats)]).remove(0);
    eprintln!(
        "graph built + ran in {:.1} s on {device:?}",
        t0.elapsed().as_secs_f64()
    );
    for c in 0..k {
        let r = c * OUT_FRAMES * 512..(c + 1) * OUT_FRAMES * 512;
        let (a, b) = (&got[r.clone()], &want[r]);
        let (mut d, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f32);
        for (&x, &y) in a.iter().zip(b) {
            d += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
            mx = mx.max((x - y).abs());
        }
        println!(
            "chunk {c}: cosine {:.5}  max|d| {mx:.3}",
            d / (na.sqrt() * nb.sqrt())
        );
    }
    Ok(())
}
