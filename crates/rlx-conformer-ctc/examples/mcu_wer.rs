// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! WER of the int8 MCU pipeline (`.rcc`) against the f32 rlx model.
//!
//! ```text
//! cargo run -p rlx-conformer-ctc --release --example mcu_wer -- \
//!     --nemo stt_en_conformer_ctc_small.nemo --rcc small-int8.rcc \
//!     --list DIR/list.tsv            # lines: <file.wav>\t<lowercase transcript>
//! ```

use std::path::Path;

use anyhow::{Context, Result};
use rlx_conformer_ctc::mcu::{AlignedBlob, transcribe_rcc};
use rlx_conformer_ctc::{ConformerCtc, wav};
use rlx_runtime::Device;

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

fn words(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .with_context(|| format!("{k} is required"))
    };
    let nemo = get("--nemo")?;
    let rcc = get("--rcc")?;
    let list = get("--list")?;
    let dir = Path::new(list).parent().unwrap_or(Path::new("."));

    let mut asr = ConformerCtc::open(nemo.as_ref(), Device::Cpu)?;
    let blob = AlignedBlob::new(&std::fs::read(rcc)?);
    let model =
        rlx_conformer_ctc_core::Model::parse(blob.bytes()).map_err(|e| anyhow::anyhow!("{e}"))?;

    let (mut n_words, mut e_f32, mut e_q8, mut same, mut n) = (0, 0, 0, 0, 0);
    for line in std::fs::read_to_string(list)?.lines() {
        let Some((file, text)) = line.split_once('\t') else {
            continue;
        };
        let w = wav::parse(&std::fs::read(dir.join(file))?)?;
        let pcm = wav::resample(&w.samples, w.sample_rate, 16_000);
        let hyp_f32 = asr.transcribe(&pcm)?;
        let ids = transcribe_rcc(blob.bytes(), &pcm, None)?;
        let ids16: Vec<u16> = ids.iter().map(|&i| i as u16).collect();
        let mut buf = vec![0u8; ids.len() * 32 + 1];
        let len = model.decode(&ids16, &mut buf);
        let hyp_q8 = String::from_utf8_lossy(&buf[..len]).into_owned();

        let r = words(text);
        n_words += r.len();
        e_f32 += edits(&r, &words(&hyp_f32));
        e_q8 += edits(&r, &words(&hyp_q8));
        same += usize::from(hyp_f32 == hyp_q8);
        n += 1;
        if n % 50 == 0 {
            eprintln!(
                "{n}: WER f32 {:.2}%  int8 {:.2}%",
                100.0 * e_f32 as f64 / n_words as f64,
                100.0 * e_q8 as f64 / n_words as f64
            );
        }
    }
    println!(
        "{n} utterances, {n_words} words: WER f32 {:.2}%, int8 {:.2}%; identical transcripts {same}/{n}",
        100.0 * e_f32 as f64 / n_words as f64,
        100.0 * e_q8 as f64 / n_words as f64
    );
    Ok(())
}
