// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! A stretch of wearable audio through the low-power voice link, end to end:
//! the device's energy gate, the packetizer, the phone's reassembler, TEN-VAD
//! confirmation with `CMD_REJECT`, then ASR on what is kept.
//!
//! The audio: LibriSpeech utterances at random times over room noise, plus
//! things that are loud but not speech (door slams, typing, music, a fan).
//! It reports how long the radio carried audio, how much of that was not
//! speech, how much speech was missed, and WER of the gated transcripts
//! against ASR of each clean utterance.
//!
//! ```text
//! cargo run -p rlx-voice-link-host --release --example wearable_sim -- \
//!     --rcc .cache/conformer-ctc-mcu/small-int8.rcc --list DIR/list.tsv \
//!     [--utts 30] [--seed 1] [--json duty.json] [--verbose]
//! ```
//! Settings: `--open-frames N` (loud frames of 3 to open), `--hangover N`
//! (quiet frames to close), `--min-speech N` (phone TEN-VAD frames to keep).

use std::path::Path;

use anyhow::{Context, Result};
use rlx_voice_link::gate::{FRAME, Gate, GateConfig, Step};
use rlx_voice_link::{Event, HEADER, Packetizer, Reassembler, SAMPLE_RATE, samples_per_packet};
use rlx_voice_link_host::confirm::{ConfirmConfig, Confirmer, Verdict};
use rlx_voice_link_host::{Asr, read_wav_i16};

/// 300 ms of pre-roll.
const PRE: usize = 19;
const SR: usize = SAMPLE_RATE as usize;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }
    fn noise(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Speech(usize),
    Door,
    Typing,
    Music,
    Fan,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Speech(_) => "speech",
            Self::Door => "door",
            Self::Typing => "typing",
            Self::Music => "music",
            Self::Fan => "fan",
        }
    }
}

/// Synthesized distractors, i16-scaled f32.
fn distractor(kind: Kind, rng: &mut Rng) -> Vec<f32> {
    match kind {
        // A 0.4 s decaying broadband thump at about −12 dBFS.
        Kind::Door => (0..SR * 2 / 5)
            .map(|i| rng.noise() * 8000.0 * (-(i as f32) / (0.06 * SR as f32)).exp())
            .collect(),
        // 4 s of 5 ms clicks every 120–200 ms.
        Kind::Typing => {
            let mut v = vec![0f32; SR * 4];
            let mut t = 0usize;
            while t + SR / 200 < v.len() {
                for s in &mut v[t..t + SR / 200] {
                    *s = rng.noise() * 3000.0;
                }
                t += (rng.range(0.12, 0.2) * SR as f32) as usize;
            }
            v
        }
        // 8 s of three-harmonic notes changing every 0.4 s, about −24 dBFS.
        Kind::Music => {
            let mut v = vec![0f32; SR * 8];
            let mut f = 220.0;
            for (i, s) in v.iter_mut().enumerate() {
                if i % (SR * 2 / 5) == 0 {
                    f = [196.0, 220.0, 262.0, 294.0, 330.0, 392.0][(rng.next() % 6) as usize];
                }
                let t = i as f32 / SR as f32;
                let w = std::f32::consts::TAU * f * t;
                *s = 1200.0 * (w.sin() + 0.5 * (2.0 * w).sin() + 0.25 * (3.0 * w).sin());
            }
            v
        }
        // 30 s of steady low-passed noise, about −40 dBFS.
        Kind::Fan => {
            let mut y = 0f32;
            (0..SR * 30)
                .map(|_| {
                    y += 0.1 * (rng.noise() * 1600.0 - y);
                    y * 1.8
                })
                .collect()
        }
        Kind::Speech(_) => unreachable!(),
    }
}

fn words(s: &str) -> Vec<String> {
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

fn overlap(a: (usize, usize), b: (usize, usize)) -> usize {
    a.1.min(b.1).saturating_sub(a.0.max(b.0))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let asr = Asr::open(Path::new(get("--rcc").context("--rcc")?))?;
    let list = get("--list").context("--list")?;
    let n_utts: usize = get("--utts").map(|v| v.parse()).transpose()?.unwrap_or(30);
    let mut rng = Rng(get("--seed")
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(1u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        | 1);

    // Utterances, spread over the list.
    let base = Path::new(list).parent().unwrap_or(Path::new("."));
    let lines: Vec<String> = std::fs::read_to_string(list)?
        .lines()
        .map(String::from)
        .collect();
    let step = (lines.len() / n_utts.max(1)).max(1);
    let mut utts = Vec::new();
    for line in lines.iter().step_by(step).take(n_utts) {
        let (file, text) = line.split_once('\t').context("list line")?;
        utts.push((read_wav_i16(&base.join(file))?, text.to_string()));
    }

    // Events in random order with random gaps.
    let mut events: Vec<Kind> = (0..utts.len()).map(Kind::Speech).collect();
    events.extend([Kind::Door; 8]);
    events.extend([Kind::Typing; 6]);
    events.extend([Kind::Music; 4]);
    events.extend([Kind::Fan; 2]);
    for i in (1..events.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        events.swap(i, j);
    }
    let mut audio: Vec<f32> = Vec::new();
    let mut spans: Vec<(Kind, usize, usize)> = Vec::new();
    for &k in &events {
        let gap = (rng.range(4.0, 40.0) * SR as f32) as usize;
        audio.resize(audio.len() + gap, 0.0);
        let start = audio.len();
        match k {
            Kind::Speech(i) => audio.extend(utts[i].0.iter().map(|&s| s as f32)),
            _ => audio.extend(distractor(k, &mut rng)),
        }
        spans.push((k, start, audio.len()));
    }
    audio.resize(audio.len() + 10 * SR, 0.0);
    audio.resize(audio.len().div_ceil(FRAME) * FRAME, 0.0);
    // Room noise (about −62 dBFS, low-passed) under everything.
    let mut y = 0f32;
    let pcm: Vec<i16> = audio
        .iter()
        .map(|&s| {
            y += 0.3 * (rng.noise() * 80.0 - y);
            (s + y).round().clamp(-32768.0, 32767.0) as i16
        })
        .collect();

    // Device → radio → phone.
    let per = samples_per_packet(244);
    let mut gcfg = GateConfig::default();
    if let Some(v) = get("--open-frames") {
        gcfg.open_frames = v.parse()?;
    }
    if let Some(v) = get("--hangover") {
        gcfg.hangover_frames = v.parse()?;
    }
    let mut gate: Gate<PRE> = Gate::new(gcfg);
    let mut tx = Packetizer::new();
    let mut rxbuf = vec![0i16; 60 * SR];
    let mut rx = Reassembler::new(&mut rxbuf);
    let mut ccfg = ConfirmConfig::default();
    if let Some(v) = get("--min-speech") {
        ccfg.min_speech = v.parse()?;
    }
    let mut confirm = Confirmer::new(ccfg);
    let mut acc: Vec<i16> = Vec::with_capacity(per);
    let mut out = vec![0u8; HEADER + per / 2];
    let (mut packets, mut bytes, mut phone_vad_frames) = (0usize, 0usize, 0usize);
    let mut open_at = 0usize;
    let mut streamed: Vec<(usize, usize, bool)> = Vec::new(); // span, kept
    let mut kept: Vec<(usize, usize, Vec<i16>)> = Vec::new();
    let mut reject_now = false;

    // Send whole packets of `acc`; `end` flushes and closes.
    let mut send = |acc: &mut Vec<i16>,
                    end: bool,
                    tx: &mut Packetizer,
                    rx: &mut Reassembler,
                    confirm: &mut Confirmer,
                    packets: &mut usize,
                    bytes: &mut usize|
     -> Option<Vec<i16>> {
        let mut done = None;
        while acc.len() >= per || (end && !acc.is_empty()) || (end && done.is_none()) {
            let take = if acc.len() >= per {
                per
            } else {
                acc.len() & !1
            };
            let last = end && acc.len() <= per;
            let n = tx.packet(&acc[..take], last, &mut out);
            acc.drain(..acc.len().min(if last { acc.len() } else { take }));
            *packets += 1;
            *bytes += n;
            match rx.push(&out[..n]) {
                Event::Utterance(len) => {
                    done = Some(rx.utterance()[..len].to_vec());
                    break;
                }
                _ => {
                    confirm.update(rx.partial());
                }
            }
            if last {
                break;
            }
        }
        done
    };

    for (fi, frame) in pcm.as_chunks::<FRAME>().0.iter().enumerate() {
        let t = fi * FRAME;
        if reject_now {
            reject_now = false;
            phone_vad_frames += confirm.frames();
            if gate.reject() {
                send(
                    &mut acc,
                    true,
                    &mut tx,
                    &mut rx,
                    &mut confirm,
                    &mut packets,
                    &mut bytes,
                );
                streamed.push((open_at, t, false));
                acc.clear();
            }
        }
        match gate.push(frame) {
            Step::Idle => continue,
            Step::Open => {
                confirm.reset();
                let pre: Vec<i16> = gate.preroll().flatten().copied().collect();
                open_at = t - pre.len();
                acc.extend(pre);
                acc.extend_from_slice(frame);
            }
            Step::Send => acc.extend_from_slice(frame),
            Step::Close => {
                acc.extend_from_slice(frame);
                let utt = send(
                    &mut acc,
                    true,
                    &mut tx,
                    &mut rx,
                    &mut confirm,
                    &mut packets,
                    &mut bytes,
                );
                // Closed before the phone decided: decide on all of it.
                let verdict = match confirm.verdict() {
                    Some(v) => v,
                    None => {
                        let v = utt.as_deref().and_then(|u| confirm.update(u));
                        v.unwrap_or(Verdict::NotSpeech)
                    }
                };
                phone_vad_frames += confirm.frames();
                let keep = verdict == Verdict::Speech;
                streamed.push((open_at, t + FRAME, keep));
                if keep && let Some(u) = utt {
                    kept.push((open_at, t + FRAME, u));
                }
                continue;
            }
        }
        send(
            &mut acc,
            false,
            &mut tx,
            &mut rx,
            &mut confirm,
            &mut packets,
            &mut bytes,
        );
        if confirm.verdict() == Some(Verdict::NotSpeech) {
            reject_now = true;
        }
    }

    // Scoring.
    let total_s = pcm.len() as f64 / SR as f64;
    let secs = |n: usize| n as f64 / SR as f64;
    let speech: Vec<(usize, usize, usize)> = spans
        .iter()
        .filter_map(|&(k, a, b)| match k {
            Kind::Speech(i) => Some((i, a, b)),
            _ => None,
        })
        .collect();
    let speech_s: f64 = speech.iter().map(|&(_, a, b)| secs(b - a)).sum();
    let streamed_s: f64 = streamed.iter().map(|&(a, b, _)| secs(b - a)).sum();
    let kept_s: f64 = streamed
        .iter()
        .filter(|s| s.2)
        .map(|&(a, b, _)| secs(b - a))
        .sum();
    let in_speech = |span: (usize, usize)| -> usize {
        speech.iter().map(|&(_, a, b)| overlap(span, (a, b))).sum()
    };
    let streamed_speech_s: f64 = streamed
        .iter()
        .map(|&(a, b, _)| secs(in_speech((a, b))))
        .sum();
    let mut covered = 0usize;
    let mut missed = 0usize;
    for &(_, a, b) in &speech {
        let c: usize = kept.iter().map(|&(x, y, _)| overlap((a, b), (x, y))).sum();
        covered += c;
        if c * 2 < b - a {
            missed += 1;
        }
    }
    let false_kept = kept
        .iter()
        .filter(|&&(a, b, _)| in_speech((a, b)) * 10 < b - a)
        .count();
    let mut by_kind: std::collections::BTreeMap<&str, f64> = Default::default();
    for &(k, a, b) in &spans {
        if matches!(k, Kind::Speech(_)) {
            continue;
        }
        let s: usize = streamed
            .iter()
            .map(|&(x, y, _)| overlap((a, b), (x, y)))
            .sum();
        *by_kind.entry(k.name()).or_default() += secs(s);
    }

    // WER: gated transcripts in time order vs the references; and the same
    // ASR on each clean utterance, so gating loss is separated from ASR.
    let verbose = args.iter().any(|a| a == "--verbose");
    // Phone side, before ASR: pieces of one sentence that the gate split at a
    // pause (under 1 s apart) go to ASR together, and a transcript of only
    // filler words is dropped.
    let mut joined: Vec<(usize, usize, Vec<i16>)> = Vec::new();
    for (a, b, u) in kept.iter().cloned() {
        match joined.last_mut() {
            Some(last) if a < last.1 + SR => {
                let gap = a.saturating_sub(last.1);
                last.2.extend(std::iter::repeat_n(0i16, gap));
                last.2.extend(&u[last.1.saturating_sub(a).min(u.len())..]);
                last.1 = b;
            }
            _ => joined.push((a, b, u)),
        }
    }
    const FILLERS: [&str; 6] = ["uh", "um", "oh", "ah", "hmm", "mm"];
    let mut hyp = Vec::new();
    let mut dropped_filler = 0;
    for (a, b, u) in &joined {
        let text = asr.transcribe(u)?;
        let w = words(&text);
        let filler = w.iter().all(|x| FILLERS.contains(&x.as_str()));
        if verbose {
            let what: Vec<&str> = spans
                .iter()
                .filter(|&&(_, x, y)| overlap((*a, *b), (x, y)) > 0)
                .map(|&(k, _, _)| k.name())
                .collect();
            println!(
                "  asr {:7.2}..{:7.2} s [{}]{} {text}",
                secs(*a),
                secs(*b),
                what.join(","),
                if filler { " (dropped)" } else { "" }
            );
        }
        if filler {
            dropped_filler += 1;
            continue;
        }
        hyp.extend(w);
    }
    let mut refw = Vec::new();
    let mut clean_errs = 0;
    for &(i, _, _) in &speech {
        let r = words(&utts[i].1);
        clean_errs += edits(&r, &words(&asr.transcribe(&utts[i].0)?));
        refw.extend(r);
    }
    let wer_gated = 100.0 * edits(&refw, &hyp) as f64 / refw.len() as f64;
    let wer_clean = 100.0 * clean_errs as f64 / refw.len() as f64;

    let st = gate.stats();
    println!(
        "{:.1} min of audio, {:.1} s speech in {} utterances",
        total_s / 60.0,
        speech_s,
        speech.len()
    );
    println!(
        "radio carried audio {:.1} s ({:.1}% of the time): {:.1} s speech, {:.1} s not",
        streamed_s,
        100.0 * streamed_s / total_s,
        streamed_speech_s,
        streamed_s - streamed_speech_s
    );
    println!(
        "  not speech by source: {}",
        by_kind
            .iter()
            .map(|(k, v)| format!("{k} {v:.1} s"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "gate opened {} times; phone rejected {}, kept {} ({} with no speech); capped {}",
        st.opens,
        st.rejects,
        kept.len(),
        false_kept,
        st.capped
    );
    println!(
        "speech kept {:.1}% of {:.1} s; utterances missed (under half kept) {}/{}",
        100.0 * covered as f64 / (speech_s * SR as f64),
        speech_s,
        missed,
        speech.len()
    );
    println!(
        "packets {packets}, {:.1} KB; phone TEN-VAD frames {phone_vad_frames}",
        bytes as f64 / 1024.0
    );
    println!(
        "phone: {} pieces joined into {} ASR calls, {dropped_filler} filler-only dropped",
        kept.len(),
        joined.len()
    );
    println!(
        "WER gated {wer_gated:.2}% vs clean utterances {wer_clean:.2}% ({} words)",
        refw.len()
    );

    if let Some(path) = get("--json") {
        let j = serde_json::json!({
            "seconds": total_s,
            "speech_seconds": speech_s,
            "streamed_seconds": streamed_s,
            "kept_seconds": kept_s,
            "streamed_speech_seconds": streamed_speech_s,
            "device_frames": st.frames,
            "gate_opens": st.opens,
            "rejects": st.rejects,
            "packets": packets,
            "bytes": bytes,
            "phone_vad_frames": phone_vad_frames,
            "speech_kept_fraction": covered as f64 / (speech_s * SR as f64),
            "utterances_missed": missed,
            "utterances": speech.len(),
            "wer_gated": wer_gated,
            "wer_clean": wer_clean,
            "asr_calls": joined.len(),
            "filler_dropped": dropped_filler,
        });
        std::fs::write(path, serde_json::to_string_pretty(&j)?)?;
    }
    Ok(())
}
