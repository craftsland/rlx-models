// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! WER of int8 ASR on audio sent over the simulated link (IMA-ADPCM, optional
//! packet loss) against the same ASR on the original PCM.
//!
//! ```text
//! cargo run -p rlx-voice-link-host --release --example link_wer -- \
//!     --rcc small-int8.rcc --list DIR/list.tsv [--drop-every N]
//! ```

use std::path::Path;

use anyhow::{Context, Result};
use rlx_voice_link_host::{Asr, read_wav_i16, simulate};

fn edits(a: &[&str], b: &[&str]) -> usize {
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
    let asr = Asr::open(get("--rcc").context("--rcc")?.as_ref())?;
    let list = get("--list").context("--list")?;
    let drop_every = get("--drop-every").map(|s| s.parse()).transpose()?;
    let dir = Path::new(list).parent().unwrap_or(Path::new("."));
    let (mut words, mut e_pcm, mut e_link, mut n) = (0, 0, 0, 0);
    for line in std::fs::read_to_string(list)?.lines() {
        let Some((file, text)) = line.split_once('\t') else {
            continue;
        };
        let pcm = read_wav_i16(&dir.join(file))?;
        let r: Vec<&str> = text.split_whitespace().collect();
        let a = asr.transcribe(&pcm)?;
        let b = asr.transcribe(&simulate(&pcm, 244, drop_every).pcm)?;
        words += r.len();
        e_pcm += edits(&r, &a.split_whitespace().collect::<Vec<_>>());
        e_link += edits(&r, &b.split_whitespace().collect::<Vec<_>>());
        n += 1;
    }
    println!(
        "{n} utterances, {words} words: WER int8 on PCM {:.2}%, over the link (ADPCM{}) {:.2}%",
        100.0 * e_pcm as f64 / words as f64,
        drop_every.map_or(String::new(), |k| format!(", 1/{k} packets lost")),
        100.0 * e_link as f64 / words as f64
    );
    Ok(())
}
