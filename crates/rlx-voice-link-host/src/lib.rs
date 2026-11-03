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

//! The receiving end of [`rlx_voice_link`] on a Mac (or Linux): BLE
//! notifications → utterance PCM → speech recognition → a local language
//! model.
//!
//! * [`Asr`] — the int8 Conformer-CTC from [`rlx_conformer_ctc_core`], the
//!   same `.rcc` model and code an MCU runs;
//! * [`Lm`] — any OpenAI-compatible chat endpoint, e.g. `rlx-openai` serving
//!   a Qwen3 or Gemma checkpoint on the same machine;
//! * [`Radio`] / [`simulate`] — the whole chain without the radio: packetize, optionally
//!   lose packets, reassemble, transcribe;
//! * [`ble`] — the radio (feature `ble`, btleplug / CoreBluetooth).

use anyhow::{Context, Result, anyhow};
use rlx_conformer_ctc::mcu::AlignedBlob;
use rlx_conformer_ctc_core::{Frontend, Model, scratch_len, transcribe};
use rlx_voice_link::{Event, HEADER, Packetizer, Reassembler};

#[cfg(feature = "ble")]
pub mod ble;
pub mod confirm;

/// int8 Conformer-CTC over an `.rcc` file (`rlx-conformer-ctc export-mcu`).
pub struct Asr {
    blob: AlignedBlob,
}

impl Asr {
    /// Load and validate an `.rcc`.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let blob = AlignedBlob::new(
            &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
        );
        Model::parse(blob.bytes()).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        Ok(Self { blob })
    }

    /// 16 kHz PCM → text.
    pub fn transcribe(&self, pcm: &[i16]) -> Result<String> {
        if pcm.is_empty() {
            return Ok(String::new());
        }
        let model = Model::parse(self.blob.bytes()).map_err(|e| anyhow!("{e}"))?;
        let fe = Frontend::new(model.config().n_fft);
        let need = scratch_len(&model, pcm.len());
        let mut fs = vec![0f32; need.f32s];
        let mut is = vec![0i8; need.i8s];
        let mut ids = vec![0u16; pcm.len() / model.config().hop + 1];
        let n =
            transcribe(&model, &fe, pcm, &mut fs, &mut is, &mut ids).map_err(|e| anyhow!("{e}"))?;
        let mut text = vec![0u8; n * 32 + 1];
        let len = model.decode(&ids[..n], &mut text);
        Ok(String::from_utf8_lossy(&text[..len]).into_owned())
    }
}

/// A chat model behind an OpenAI-compatible `/v1/chat/completions`.
pub struct Lm {
    /// Full endpoint URL.
    pub url: String,
    /// `model` field of the request.
    pub model: String,
    /// Optional system prompt.
    pub system: Option<String>,
}

impl Lm {
    /// One user turn → the assistant's reply.
    pub fn reply(&self, user: &str) -> Result<String> {
        let mut messages = Vec::new();
        if let Some(s) = &self.system {
            messages.push(serde_json::json!({"role": "system", "content": s}));
        }
        messages.push(serde_json::json!({"role": "user", "content": user}));
        let body = serde_json::json!({"model": self.model, "messages": messages});
        let resp: serde_json::Value = ureq::post(&self.url)
            .send_json(body)
            .with_context(|| format!("POST {}", self.url))?
            .into_json()?;
        resp["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| anyhow!("no choices[0].message.content in {resp}"))
    }
}

/// How the simulated radio loses packets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Loss {
    /// Every packet arrives.
    None,
    /// Every `n`-th packet is lost.
    Every(usize),
    /// Each packet is lost with this probability, in percent.
    Random(f64),
}

/// What one utterance did on the simulated link.
#[derive(Debug)]
pub struct SimReport {
    /// Packets sent.
    pub packets: usize,
    /// Packets the radio lost.
    pub dropped: usize,
    /// Packets the receiver counted as missing (filled with silence).
    pub lost: u32,
    /// Bytes over the air, headers included.
    pub bytes: usize,
    /// The reassembled utterance; empty if it never completed.
    pub pcm: Vec<i16>,
}

/// A simulated device → radio → phone link: the device's [`Packetizer`], a
/// lossy channel and the phone's [`Reassembler`]. The loss pattern comes from
/// a seeded generator, so a run is reproducible, and it carries on across
/// utterances like a real radio would.
pub struct Radio {
    /// Packet loss model.
    pub loss: Loss,
    /// ATT payload bytes per packet (MTU − 3).
    pub att_payload: usize,
    /// Send packets at the pace a microphone produces them.
    pub realtime: bool,
    rng: u64,
    sent: usize,
    tx: Packetizer,
}

impl Radio {
    /// A link with ATT payload 244 (MTU 247), not paced.
    pub fn new(loss: Loss, seed: u64) -> Self {
        Self {
            loss,
            att_payload: 244,
            realtime: false,
            rng: seed.max(1),
            sent: 0,
            tx: Packetizer::new(),
        }
    }

    /// xorshift64*, uniform in [0, 100).
    fn percent(&mut self) -> f64 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let x = self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
        x as f64 / (1u64 << 53) as f64 * 100.0
    }

    fn lose(&mut self) -> bool {
        match self.loss {
            Loss::None => false,
            Loss::Every(n) => n > 0 && self.sent % n == n - 1,
            Loss::Random(p) => self.percent() < p,
        }
    }

    /// Send one utterance. Every packet, START included, can be lost except
    /// the END: on a real link the BLE layer acknowledges packets, and what
    /// goes missing is audio the firmware could not queue — the utterance
    /// still closes. An odd trailing sample is dropped, as the firmware does.
    pub fn send(&mut self, pcm: &[i16]) -> SimReport {
        let per = rlx_voice_link::samples_per_packet(self.att_payload);
        let pcm = &pcm[..pcm.len() & !1];
        let mut buf = vec![0i16; pcm.len() + per];
        let mut rx = Reassembler::new(&mut buf);
        let mut out = vec![0u8; HEADER + per / 2];
        let chunks: Vec<&[i16]> = if pcm.is_empty() {
            vec![&[]]
        } else {
            pcm.chunks(per).collect()
        };
        let (mut bytes, mut dropped, mut got) = (0, 0, 0);
        let start = std::time::Instant::now();
        for (i, c) in chunks.iter().enumerate() {
            let last = i + 1 == chunks.len();
            if self.realtime {
                let due = std::time::Duration::from_secs_f64(
                    ((i + 1) * per) as f64 / rlx_voice_link::SAMPLE_RATE as f64,
                );
                if let Some(wait) = due.checked_sub(start.elapsed()) {
                    std::thread::sleep(wait);
                }
            }
            let n = self.tx.packet(c, last, &mut out);
            bytes += n;
            let lost = !last && self.lose();
            self.sent += 1;
            if lost {
                dropped += 1;
                continue;
            }
            if let Event::Utterance(len) = rx.push(&out[..n]) {
                got = len;
            }
        }
        SimReport {
            packets: chunks.len(),
            dropped,
            lost: rx.lost(),
            bytes,
            pcm: rx.utterance()[..got].to_vec(),
        }
    }
}

/// One utterance through a fresh [`Radio`] (the `simulate` command):
/// `att_payload`-sized packets, every `drop_every`-th one lost.
pub fn simulate(pcm: &[i16], att_payload: usize, drop_every: Option<usize>) -> SimReport {
    let mut radio = Radio::new(drop_every.map_or(Loss::None, Loss::Every), 1);
    radio.att_payload = att_payload;
    radio.send(pcm)
}

/// Read a 16-bit mono 16 kHz WAV as i16 (resampled if it is not 16 kHz).
pub fn read_wav_i16(path: &std::path::Path) -> Result<Vec<i16>> {
    let w = rlx_conformer_ctc::wav::parse(&std::fs::read(path)?)?;
    let pcm = rlx_conformer_ctc::wav::resample(&w.samples, w.sample_rate, 16_000);
    Ok(pcm
        .iter()
        .map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect())
}
