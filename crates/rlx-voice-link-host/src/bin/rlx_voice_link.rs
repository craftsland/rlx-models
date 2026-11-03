// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! `rlx-voice-link` — receive speech from the nRF52840 voice-link firmware,
//! transcribe it, optionally ask a local LM.
//!
//! ```text
//! rlx-voice-link listen   --rcc small-int8.rcc [--lm-url URL --lm-model NAME [--system TEXT]]
//! rlx-voice-link simulate --rcc small-int8.rcc --wav clip.wav [--drop-every N] [--att 244] [LM flags]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_voice_link_host::{Asr, Lm, read_wav_i16, simulate};

struct Args {
    cmd: String,
    rcc: PathBuf,
    wav: Option<PathBuf>,
    drop_every: Option<usize>,
    att: usize,
    lm: Option<Lm>,
}

fn parse() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().unwrap_or_else(|| "help".into());
    let mut rcc = None;
    let (mut wav, mut drop_every, mut att) = (None, None, 244);
    let (mut lm_url, mut lm_model, mut system) = (None, None, None);
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--rcc" => rcc = Some(PathBuf::from(val()?)),
            "--wav" => wav = Some(PathBuf::from(val()?)),
            "--drop-every" => drop_every = Some(val()?.parse()?),
            "--att" => att = val()?.parse()?,
            "--lm-url" => lm_url = Some(val()?),
            "--lm-model" => lm_model = Some(val()?),
            "--system" => system = Some(val()?),
            other => bail!("unexpected argument {other:?}"),
        }
    }
    if cmd == "help" || cmd == "-h" || cmd == "--help" {
        eprintln!(
            "rlx-voice-link listen   --rcc MODEL.rcc [--lm-url URL --lm-model NAME [--system TEXT]]\n\
             rlx-voice-link simulate --rcc MODEL.rcc --wav CLIP.wav [--drop-every N] [--att 244] [LM flags]\n\n\
             MODEL.rcc: `just export-conformer-ctc-mcu`. LM: any OpenAI-compatible endpoint,\n\
             e.g. `just openai-serve -- --engine qwen3 --weights …` → --lm-url http://127.0.0.1:8080/v1/chat/completions"
        );
        std::process::exit(0);
    }
    let lm = lm_url.map(|url| Lm {
        url,
        model: lm_model.unwrap_or_else(|| "default".into()),
        system,
    });
    Ok(Args {
        cmd,
        rcc: rcc.context("--rcc MODEL.rcc is required")?,
        wav,
        drop_every,
        att,
        lm,
    })
}

fn handle(asr: &Asr, lm: Option<&Lm>, pcm: &[i16]) -> Result<()> {
    let t0 = Instant::now();
    let text = asr.transcribe(pcm)?;
    eprintln!(
        "[voice-link] {:.2} s audio → ASR {:.0} ms",
        pcm.len() as f64 / 16_000.0,
        t0.elapsed().as_secs_f64() * 1e3
    );
    println!("you: {text}");
    if let Some(lm) = lm
        && !text.is_empty()
    {
        let t1 = Instant::now();
        let reply = lm.reply(&text)?;
        eprintln!("[voice-link] LM {:.0} ms", t1.elapsed().as_secs_f64() * 1e3);
        println!("lm:  {reply}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let a = parse()?;
    let asr = Asr::open(&a.rcc)?;
    match a.cmd.as_str() {
        "simulate" => {
            let wav = a.wav.context("--wav is required for simulate")?;
            let pcm = read_wav_i16(&wav)?;
            let r = simulate(&pcm, a.att, a.drop_every);
            let secs = pcm.len() as f64 / 16_000.0;
            eprintln!(
                "[voice-link] {} packets, {} B ({:.1} kbit/s), {} lost",
                r.packets,
                r.bytes,
                r.bytes as f64 * 8.0 / secs / 1e3,
                r.lost
            );
            handle(&asr, a.lm.as_ref(), &r.pcm)
        }
        #[cfg(feature = "ble")]
        "listen" => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(rlx_voice_link_host::ble::listen(|pcm, lost| {
                if lost > 0 {
                    eprintln!("[voice-link] {lost} packets lost so far");
                }
                if let Err(e) = handle(&asr, a.lm.as_ref(), pcm) {
                    eprintln!("[voice-link] {e:#}");
                }
            }))
        }
        other => bail!("unknown command {other:?} (listen | simulate | help)"),
    }
}
