// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! The whole voice link on one machine, no hardware: a simulated nRF52840
//! streams a clip packet by packet, a lossy radio drops some, the phone side
//! reassembles and transcribes, and a local LM (optional) answers.
//!
//! It drives the same `Packetizer` / `Reassembler` the firmware and the
//! phone use, so it doubles as the reference for the API:
//!
//! ```text
//! just export-conformer-ctc-mcu        # the int8 ASR model, once
//! just voice-link-demo                 # three JFK clips, 3% loss
//! just voice-link-demo --wav assets/harness/jfk_16k.wav --loss 15 --realtime
//! just voice-link-demo --lm-url http://127.0.0.1:8080/v1/chat/completions --lm-model qwen3
//! ```
//!
//! Flags: `--wav F` (repeatable, one utterance each; default the three JFK
//! clips in `assets/`: the 1961 inaugural line, a voice-cloned copy of it,
//! and a cloned "Rust" speech),
//! `--rcc F` (default `.cache/conformer-ctc-mcu/small-int8.rcc`), `--loss P`
//! (percent of packets dropped at random, default 3), `--seed N`,
//! `--att N` (ATT payload, default 244 = MTU 247), `--realtime` (pace the
//! device at the speed of speech), `--lm-url` / `--lm-model` / `--system`.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_voice_link::{HEADER, SAMPLE_RATE, samples_per_packet};
use rlx_voice_link_host::{Asr, Lm, Loss, Radio, read_wav_i16};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

struct Opts {
    wavs: Vec<PathBuf>,
    rcc: PathBuf,
    loss: f64,
    seed: u64,
    att: usize,
    realtime: bool,
    lm: Option<Lm>,
}

fn parse() -> Result<Opts> {
    let mut o = Opts {
        wavs: Vec::new(),
        rcc: PathBuf::from(format!("{ROOT}/.cache/conformer-ctc-mcu/small-int8.rcc")),
        loss: 3.0,
        seed: 1,
        att: 244,
        realtime: false,
        lm: None,
    };
    let (mut url, mut model, mut system) = (None, None, None);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--wav" => o.wavs.push(PathBuf::from(val()?)),
            "--rcc" => o.rcc = PathBuf::from(val()?),
            "--loss" => o.loss = val()?.parse()?,
            "--seed" => o.seed = val()?.parse()?,
            "--att" => o.att = val()?.parse()?,
            "--realtime" => o.realtime = true,
            "--lm-url" => url = Some(val()?),
            "--lm-model" => model = Some(val()?),
            "--system" => system = Some(val()?),
            other => bail!("unexpected argument {other:?} (see the header of voice_link_demo.rs)"),
        }
    }
    if o.wavs.is_empty() {
        for f in [
            "assets/harness/jfk_16k.wav",
            "assets/jfk/jfk_voice_clone.wav",
            "assets/jfk/jfk_rust_speech.wav",
        ] {
            o.wavs.push(PathBuf::from(format!("{ROOT}/{f}")));
        }
    }
    if !(0.0..100.0).contains(&o.loss) {
        bail!("--loss must be in [0, 100)");
    }
    if !(HEADER + 2..=512).contains(&o.att) {
        bail!("--att must be in {}..=512", HEADER + 2);
    }
    o.lm = url.map(|url| Lm {
        url,
        model: model.unwrap_or_else(|| "default".into()),
        system: Some(system.unwrap_or_else(|| {
            "You are a voice assistant. The user's words come from speech recognition; \
             answer in one or two short sentences."
                .into()
        })),
    });
    Ok(o)
}

fn main() -> Result<()> {
    let o = parse()?;
    let asr = Asr::open(&o.rcc).with_context(|| {
        format!(
            "{}: export it first with `just export-conformer-ctc-mcu`",
            o.rcc.display()
        )
    })?;
    let per = samples_per_packet(o.att);
    println!(
        "voice link demo — device: simulated nRF52840 · radio: {:.1}% loss, {} samples/packet ({:.1} ms) · \
         phone: int8 Conformer-CTC · LM: {}",
        o.loss,
        per,
        per as f64 * 1e3 / SAMPLE_RATE as f64,
        o.lm.as_ref()
            .map_or("none (pass --lm-url)", |l| l.url.as_str())
    );

    let mut radio = Radio::new(
        if o.loss > 0.0 {
            Loss::Random(o.loss)
        } else {
            Loss::None
        },
        o.seed,
    );
    radio.att_payload = o.att;
    radio.realtime = o.realtime;
    let (mut total_audio, mut total_asr, mut total_lost) = (0.0f64, 0.0f64, 0usize);

    for (turn, wav) in o.wavs.iter().enumerate() {
        let pcm = read_wav_i16(wav).with_context(|| format!("read {}", wav.display()))?;
        let secs = pcm.len() as f64 / SAMPLE_RATE as f64;
        let name = wav.strip_prefix(ROOT).unwrap_or(wav).display();
        println!("\n[{}] {name} ({secs:.2} s)", turn + 1);

        let t0 = Instant::now();
        let link = radio.send(&pcm);
        total_lost += link.dropped;
        let air = t0.elapsed();
        println!(
            "  device → phone   {} packets, {} dropped, {:.1} kB ({:.1} kbit/s){}",
            link.packets,
            link.dropped,
            link.bytes as f64 / 1e3,
            link.bytes as f64 * 8.0 / secs / 1e3,
            if o.realtime {
                format!(", streamed in {:.2} s", air.as_secs_f64())
            } else {
                String::new()
            }
        );

        let t1 = Instant::now();
        let heard = asr.transcribe(&link.pcm)?;
        let asr_s = t1.elapsed().as_secs_f64();
        let clean = asr.transcribe(&pcm)?;
        total_audio += secs;
        total_asr += asr_s;
        println!(
            "  phone ASR        {:.0} ms ({:.0}× faster than real time)",
            asr_s * 1e3,
            secs / asr_s
        );
        println!("  heard            \"{heard}\"");
        if heard != clean {
            println!("  without the link \"{clean}\"");
        } else {
            println!("  (same words as the original audio)");
        }

        if let Some(lm) = &o.lm
            && !heard.is_empty()
        {
            let t2 = Instant::now();
            match lm.reply(&heard) {
                Ok(reply) => println!(
                    "  LM ({:.1} s)       {}",
                    t2.elapsed().as_secs_f64(),
                    reply.trim()
                ),
                Err(e) => println!("  LM error         {e:#}"),
            }
        }
    }
    println!(
        "\n{:.1} s of speech, ASR {:.2} s total, {} packets lost on the link",
        total_audio, total_asr, total_lost
    );
    Ok(())
}
