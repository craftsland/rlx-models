// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! `rlx-lfm3-audio asr|tts|chat …`.

#[cfg(feature = "tokenizer")]
use std::path::PathBuf;
#[cfg(feature = "tokenizer")]
use std::time::Instant;

#[cfg(feature = "tokenizer")]
use anyhow::anyhow;
use anyhow::{Result, bail};

pub const USAGE: &str = "\
rlx-lfm3-audio — LiquidAI LFM3-Audio-2B (GGUF)

USAGE:
  rlx-lfm3-audio asr  --weights DIR --wav CLIP.wav [--language English]
  rlx-lfm3-audio tts  --weights DIR --text \"…\" [--language en] [--voice female|male] [-o out.wav]
  rlx-lfm3-audio chat --weights DIR (--text \"…\" | --wav Q.wav) [--speak] [--system \"…\"] [-o reply.wav]

OPTIONS:
  --weights DIR|FILE   the GGUF directory (LM + mmproj), or the LM .gguf with the mmproj beside it
  --prefer Q4_K_M      LM quant to pick from a directory
  --mmproj-prefer Q8_0 mmproj quant (default F16; Q8_0 / Q4_K_M use less memory)
  --device cpu|metal|mlx|cuda|wgpu|vulkan   (default cpu)
  --max-tokens N       generation budget in slots (text tokens + speech frames; default 512)
  --temperature T      text sampling temperature (default 0 = greedy)
  --audio-temperature T, --audio-top-k K, --audio-penalty P   speech sampling (card defaults 1.0 / 128 / 1.1)
  --seed N             sampling seed (default 0)
  --speak              chat: answer with interleaved text and speech (<|mixed_start|>)
  --low-memory         one LM graph, stage graphs dropped after use (default on iOS)
  --timing             print stage timings and the process footprint
";

#[cfg(feature = "tokenizer")]
struct Args {
    cmd: String,
    weights: PathBuf,
    prefer: Option<String>,
    mmproj: Option<String>,
    device: String,
    wav: Option<PathBuf>,
    text: Option<String>,
    language: Option<String>,
    voice: String,
    system: Option<String>,
    out: Option<PathBuf>,
    max_tokens: usize,
    temperature: f32,
    audio_temperature: Option<f32>,
    audio_top_k: Option<usize>,
    audio_penalty: Option<f32>,
    seed: u64,
    speak: bool,
    low_memory: bool,
    timing: bool,
}

#[cfg(feature = "tokenizer")]
fn parse(args: &[String]) -> Result<Args> {
    let mut it = args.iter();
    let cmd = it
        .next()
        .ok_or_else(|| anyhow!("missing command\n\n{USAGE}"))?
        .clone();
    if cmd == "-h" || cmd == "--help" {
        print!("{USAGE}");
        std::process::exit(0);
    }
    let mut a = Args {
        cmd,
        weights: PathBuf::from("weights/lm/lfm3-audio-2b-gguf"),
        prefer: None,
        mmproj: None,
        device: "cpu".into(),
        wav: None,
        text: None,
        language: None,
        voice: "female".into(),
        system: None,
        out: None,
        max_tokens: 512,
        temperature: 0.0,
        audio_temperature: None,
        audio_top_k: None,
        audio_penalty: None,
        seed: 0,
        speak: false,
        low_memory: false,
        timing: false,
    };
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .cloned()
                .ok_or_else(|| anyhow!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--weights" => a.weights = val()?.into(),
            "--prefer" => a.prefer = Some(val()?),
            "--mmproj-prefer" => a.mmproj = Some(val()?),
            "--device" => a.device = val()?,
            "--wav" => a.wav = Some(val()?.into()),
            "--text" | "--prompt" => a.text = Some(val()?),
            "--language" => a.language = Some(val()?),
            "--voice" => a.voice = val()?,
            "--system" => a.system = Some(val()?),
            "-o" | "--out" => a.out = Some(val()?.into()),
            "--max-tokens" => a.max_tokens = val()?.parse()?,
            "--temperature" => a.temperature = val()?.parse()?,
            "--audio-temperature" => a.audio_temperature = Some(val()?.parse()?),
            "--audio-top-k" => a.audio_top_k = Some(val()?.parse()?),
            "--audio-penalty" => a.audio_penalty = Some(val()?.parse()?),
            "--seed" => a.seed = val()?.parse()?,
            "--speak" => a.speak = true,
            "--low-memory" => a.low_memory = true,
            "--timing" => a.timing = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown option {other}\n\n{USAGE}"),
        }
    }
    Ok(a)
}

#[cfg(not(feature = "tokenizer"))]
pub fn run(_args: &[String]) -> Result<()> {
    bail!("rlx-lfm3-audio needs the `tokenizer` feature (on by default)")
}

#[cfg(feature = "tokenizer")]
pub fn run(args: &[String]) -> Result<()> {
    use std::io::Write;

    use crate::config::CODEC_SAMPLE_RATE;
    use crate::sampling::Sampling;
    use crate::text::Tokenizer;
    use crate::{GenerateOptions, Lfm3Audio, prompt, wav};

    let a = parse(args)?;
    crate::diag::set_timing(a.timing);
    let device = rlx_cli::parse_device(&a.device)?;
    let t0 = Instant::now();
    let mut m = if a.weights.is_dir() {
        Lfm3Audio::open_dir(&a.weights, a.prefer.as_deref(), a.mmproj.as_deref(), device)?
    } else {
        let (lm, mm) = crate::resolve(&a.weights, None, a.mmproj.as_deref())?;
        Lfm3Audio::open(&lm, &mm, device)?
    };
    if a.low_memory {
        m.set_low_memory(true);
    }
    crate::diag::trace("model open");
    let tok = Tokenizer::new(m.lm_path());
    let encode = |s: &str| tok.encode(s);
    if a.timing {
        eprintln!("[lfm3-audio] load {:.2?}", t0.elapsed());
    }

    let mut opts = GenerateOptions::new(&m.cfg);
    opts.max_new_tokens = a.max_tokens;
    opts.seed = a.seed;
    opts.text = Sampling {
        temperature: a.temperature,
        top_k: 0,
        repetition_penalty: 1.0,
    };
    if let Some(t) = a.audio_temperature {
        opts.speech.temperature = t;
    }
    if let Some(k) = a.audio_top_k {
        opts.speech.top_k = k;
    }
    if let Some(p) = a.audio_penalty {
        opts.speech.repetition_penalty = p;
    }

    let audio_in = |m: &mut Lfm3Audio| -> Result<Option<Vec<f32>>> {
        let Some(path) = &a.wav else { return Ok(None) };
        let t = Instant::now();
        let pcm = wav::read_16k(path)?;
        let emb = m.encode_audio(&pcm)?;
        if a.timing {
            eprintln!(
                "[lfm3-audio] encoded {:.2} s of audio in {:.2?}",
                pcm.len() as f64 / 16000.0,
                t.elapsed()
            );
        }
        Ok(Some(emb))
    };

    let segments = match a.cmd.as_str() {
        "asr" => {
            let audio = audio_in(&mut m)?.ok_or_else(|| anyhow!("asr needs --wav"))?;
            opts.text_only = true;
            prompt::asr(
                &m.cfg.tokens,
                &encode,
                Some(a.language.as_deref().unwrap_or("English")),
                audio,
            )?
        }
        "tts" => {
            let text = a
                .text
                .as_deref()
                .ok_or_else(|| anyhow!("tts needs --text"))?;
            prompt::tts(
                &m.cfg.tokens,
                &encode,
                text,
                a.language.as_deref().unwrap_or("en"),
                &a.voice,
            )?
        }
        "chat" => {
            let audio = audio_in(&mut m)?;
            if audio.is_none() && a.text.is_none() {
                bail!("chat needs --text or --wav");
            }
            let system = a.system.clone().or_else(|| {
                a.speak
                    .then(|| "Respond with interleaved text and audio.".to_string())
            });
            opts.text_only = !a.speak;
            prompt::chat(
                &m.cfg.tokens,
                &encode,
                system.as_deref(),
                a.text.as_deref(),
                audio,
                a.speak.then_some("<|mixed_start|>"),
            )?
        }
        other => bail!("unknown command {other}\n\n{USAGE}"),
    };

    crate::diag::trace("prompt built");
    let t = Instant::now();
    let audio_out = m.cfg.tokens.audio_out;
    let mut shown = String::new();
    let mut text_ids = Vec::new();
    let out = m.generate(
        &segments,
        &opts,
        |id| {
            if id == audio_out {
                return;
            }
            text_ids.push(id);
            if let Ok(s) = tok.decode(&text_ids, true)
                && s.len() >= shown.len()
                && s.starts_with(shown.as_str())
            {
                print!("{}", &s[shown.len()..]);
                std::io::stdout().flush().ok();
                shown = s;
            }
        },
        |_| {},
    )?;
    println!();
    let gen_time = t.elapsed();
    if a.timing {
        eprintln!(
            "[lfm3-audio] generated {} text tokens + {} speech frames in {:.2?}",
            out.tokens.len() - out.frames.len(),
            out.frames.len(),
            gen_time
        );
    }
    if !out.frames.is_empty() {
        let t = Instant::now();
        let pcm = m.decode_frames(&out.frames)?;
        let path = a
            .out
            .clone()
            .unwrap_or_else(|| PathBuf::from("lfm3_audio_out.wav"));
        wav::write_mono_16(&path, &pcm, CODEC_SAMPLE_RATE)?;
        let secs = pcm.len() as f64 / CODEC_SAMPLE_RATE as f64;
        eprintln!(
            "[lfm3-audio] wrote {} ({secs:.2} s, {} frames; codec {:.2?}, RTF {:.2})",
            path.display(),
            out.frames.len(),
            t.elapsed(),
            (gen_time + t.elapsed()).as_secs_f64() / secs.max(1e-9),
        );
    }
    Ok(())
}
