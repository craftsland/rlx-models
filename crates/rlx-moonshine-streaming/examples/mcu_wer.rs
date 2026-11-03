// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! WER of the MCU `.rms` pipeline against the f32 rlx model over a list.
//!
//! ```text
//! cargo run -p rlx-moonshine-streaming --release --example mcu_wer -- \
//!     --weights weights/asr/moonshine-streaming-tiny --rms tiny.rms --list DIR/list.tsv
//! ```
//! List lines: `<file.wav>\t<lowercase transcript>`. Both outputs are
//! lowercased and stripped of punctuation (apostrophes kept) before scoring.

use std::path::Path;

use anyhow::{Context, Result};
use rlx_moonshine_streaming::MoonshineStreaming;
use rlx_moonshine_streaming::mcu::{AlignedBlob, transcribe_rms};
use rlx_runtime::Device;

fn norm(s: &str) -> Vec<String> {
    s.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '\'' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_string)
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
    let dir = get("--weights").context("--weights")?;
    let blob = AlignedBlob::new(&std::fs::read(get("--rms").context("--rms")?)?);
    let list = get("--list").context("--list")?;
    let base = Path::new(list).parent().unwrap_or(Path::new("."));
    let mut asr = MoonshineStreaming::open(dir, Device::Cpu)?;
    let (mut words, mut e32, mut e8, mut same, mut n) = (0, 0, 0, 0, 0);
    for line in std::fs::read_to_string(list)?.lines() {
        let Some((file, text)) = line.split_once('\t') else {
            continue;
        };
        let (pcm, rate) = rlx_core::asr_bench::read_wav_mono(&base.join(file))?;
        let pcm = if rate == 16_000 {
            pcm
        } else {
            rlx_core::audio::resample_linear(&pcm, rate, 16_000)
        };
        let r = norm(text);
        let a = norm(&asr.transcribe(&pcm, None)?);
        let (_, b, _) = transcribe_rms(blob.bytes(), &pcm)?;
        let b = norm(&b);
        words += r.len();
        e32 += edits(&r, &a);
        e8 += edits(&r, &b);
        same += usize::from(a == b);
        n += 1;
        if n % 50 == 0 {
            eprintln!(
                "{n}: f32 {:.2}%  mcu {:.2}%",
                100.0 * e32 as f64 / words as f64,
                100.0 * e8 as f64 / words as f64
            );
        }
    }
    println!(
        "{n} utterances, {words} words: WER f32 {:.2}%, mcu {:.2}%; identical transcripts {same}/{n}",
        100.0 * e32 as f64 / words as f64,
        100.0 * e8 as f64 / words as f64
    );
    Ok(())
}
