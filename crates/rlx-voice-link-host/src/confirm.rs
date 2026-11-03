// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Phone side of the voice gate: TEN-VAD over the start of each utterance
//! the device opens, deciding whether it is speech.
//!
//! The device's [`rlx_voice_link::gate::Gate`] opens on anything louder than
//! the room. This runs the real detector the device cannot afford (about 1 M
//! instructions per 16 ms frame on an FPU core) and answers within
//! `window_frames`: [`Verdict::Speech`] as soon as `min_speech` frames score
//! above `threshold`, [`Verdict::NotSpeech`] if the window passes without
//! that. On `NotSpeech` the phone writes [`rlx_voice_link::CMD_REJECT`] and
//! drops the utterance.

use rlx_ten_vad_core::{HOP_SIZE, Vad};

/// Confirmation settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfirmConfig {
    /// Frames (16 ms each) of audio the decision may take.
    pub window_frames: usize,
    /// Speech frames that accept.
    pub min_speech: usize,
    /// TEN-VAD probability counted as speech.
    pub threshold: f32,
}

impl Default for ConfirmConfig {
    fn default() -> Self {
        Self {
            // The device sends ~300 ms of pre-roll first, so 50 frames is
            // 500 ms past the trigger.
            window_frames: 50,
            min_speech: 5,
            threshold: 0.5,
        }
    }
}

/// The decision about one utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Keep it: send it to ASR when it closes.
    Speech,
    /// Reject it: write `CMD_REJECT` and drop it.
    NotSpeech,
}

/// TEN-VAD over the head of one utterance at a time.
pub struct Confirmer {
    cfg: ConfirmConfig,
    vad: Vad,
    fed: usize,
    frames: usize,
    speech: usize,
    verdict: Option<Verdict>,
}

impl Confirmer {
    /// A confirmer waiting for its first utterance.
    pub fn new(cfg: ConfirmConfig) -> Self {
        Self {
            cfg,
            vad: Vad::new(),
            fed: 0,
            frames: 0,
            speech: 0,
            verdict: None,
        }
    }

    /// A new utterance opened.
    pub fn reset(&mut self) {
        self.vad.reset();
        self.fed = 0;
        self.frames = 0;
        self.speech = 0;
        self.verdict = None;
    }

    /// The decision so far.
    pub fn verdict(&self) -> Option<Verdict> {
        self.verdict
    }

    /// Frames scored for the current utterance (the phone's TEN-VAD work).
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Feed the utterance received so far (`Reassembler::partial`); scores
    /// whole frames not yet seen. Returns the verdict once there is one.
    pub fn update(&mut self, received: &[i16]) -> Option<Verdict> {
        while self.verdict.is_none() && received.len() >= self.fed + HOP_SIZE {
            let p = self
                .vad
                .process_i16(&received[self.fed..self.fed + HOP_SIZE]);
            self.fed += HOP_SIZE;
            self.frames += 1;
            if p > self.cfg.threshold {
                self.speech += 1;
            }
            if self.speech >= self.cfg.min_speech {
                self.verdict = Some(Verdict::Speech);
            } else if self.frames >= self.cfg.window_frames {
                self.verdict = Some(Verdict::NotSpeech);
            }
        }
        self.verdict
    }
}
