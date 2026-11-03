// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Export Citrinet to the MCU `.rct` blob and compare the int8 no_std
//! pipeline with the f32 reference, on one clip (`--wav`) or a list
//! (`--list`, lines `<file.wav>\t<transcript>`, scored as WER).
//!
//! ```text
//! cargo run -p rlx-citrinet --release --example mcu_check -- \
//!     --nemo stt_en_citrinet_256_ls.nemo (--wav clip.wav | --list DIR/list.tsv) [--out m.rct] [--bits 8|9] [--no-ref]
//! ```

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rlx_citrinet::mcu::{CitrinetSpec, export_rct, load_weights, reference};
use rlx_citrinet_core as core;
use rlx_nemo::NemoModel;

fn aligned(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split_whitespace()
        .map(String::from)
        .collect()
}

fn edits(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(x != y))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let nemo = NemoModel::open(Path::new(get("--nemo").context("--nemo")?))?;
    let bits: u32 = get("--bits").map(|b| b.parse()).transpose()?.unwrap_or(8);
    let blob = export_rct(&nemo, bits)?;
    eprintln!("exported {:.2} MB (bits {bits})", blob.len() as f64 / 1e6);
    if let Some(o) = get("--out") {
        std::fs::write(o, &blob)?;
    }
    let words_buf = aligned(&blob);
    // SAFETY: u32 storage viewed as bytes.
    let bytes = unsafe { std::slice::from_raw_parts(words_buf.as_ptr().cast::<u8>(), blob.len()) };
    let model = core::Model::parse(bytes).map_err(|e| anyhow!("{e}"))?;
    let spec = CitrinetSpec::from_nemo(&nemo)?;
    let w = load_weights(&nemo, &spec)?;
    let fe = core::Frontend::new(model.config().n_fft);
    let no_ref = args.iter().any(|a| a == "--no-ref");

    let run = |pcm: &[f32]| -> Result<(String, String, f64)> {
        let need = core::scratch_len(&model, pcm.len());
        let mut fs = vec![0f32; need.f32s];
        let mut is = vec![0i8; need.i8s];
        let frames = core::Frontend::frames(&model, pcm.len());
        let t = model.enc_frames(frames);
        let mut feats = vec![0f32; t * model.decoder_width()];
        let mut ids = vec![0u16; frames];
        let n = core::transcribe_with_features(
            &model, &fe, pcm, &mut fs, &mut is, &mut ids, &mut feats,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let mut txt = vec![0u8; n * 32 + 1];
        let len = model.decode(&ids[..n], &mut txt);
        let mcu = String::from_utf8_lossy(&txt[..len]).into_owned();
        if no_ref {
            return Ok((mcu, String::new(), 0.0));
        }
        // f32 reference over the same mel.
        let mut mel = vec![0f32; frames * model.config().n_mels];
        let mut work = vec![0f32; 2 * model.config().n_fft];
        fe.log_mel(&model, pcm, &mut mel, &mut work);
        let (rf, rids) = reference(&spec, &w, &mel, frames);
        let r16: Vec<u16> = rids.iter().map(|&i| i as u16).collect();
        let len = model.decode(&r16, &mut txt);
        let f32t = String::from_utf8_lossy(&txt[..len]).into_owned();
        let (mut d, mut a, mut b) = (0f64, 0f64, 0f64);
        for (&x, &y) in rf.iter().zip(&feats) {
            d += x as f64 * y as f64;
            a += x as f64 * x as f64;
            b += y as f64 * y as f64;
        }
        Ok((mcu, f32t, d / (a.sqrt() * b.sqrt())))
    };
    let read = |p: &Path| -> Result<Vec<f32>> {
        let (pcm, rate) = rlx_core::asr_bench::read_wav_mono(p)?;
        Ok(if rate == 16_000 {
            pcm
        } else {
            rlx_core::audio::resample_linear(&pcm, rate, 16_000)
        })
    };

    if let Some(wav) = get("--wav") {
        let pcm = read(Path::new(wav))?;
        let need = core::scratch_len(&model, pcm.len());
        let (mcu, f32t, cos) = run(&pcm)?;
        println!(
            "mcu: {mcu}\nf32: {f32t}\nencoder cosine {cos:.5}, scratch {:.1} KiB",
            need.bytes() as f64 / 1024.0
        );
    }
    if let Some(list) = get("--list") {
        let base = Path::new(list).parent().unwrap_or(Path::new("."));
        let (mut n_w, mut e32, mut e8, mut same, mut n) = (0, 0, 0, 0, 0);
        for line in std::fs::read_to_string(list)?.lines() {
            let Some((file, text)) = line.split_once('\t') else {
                continue;
            };
            let (mcu, f32t, _) = run(&read(&base.join(file))?)?;
            let r = words(text);
            n_w += r.len();
            e32 += edits(&r, &words(&f32t));
            e8 += edits(&r, &words(&mcu));
            same += usize::from(mcu == f32t);
            n += 1;
        }
        println!(
            "{n} utterances, {n_w} words: WER f32 {:.2}%, mcu {:.2}%; identical {same}/{n}",
            100.0 * e32 as f64 / n_w as f64,
            100.0 * e8 as f64 / n_w as f64
        );
    }
    Ok(())
}
