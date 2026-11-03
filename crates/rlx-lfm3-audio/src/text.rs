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

//! Text in and out through the GGUF-embedded byte-level BPE
//! (`tokenizer.ggml.*`, pre-tokenizer `lfm2`).

use std::path::{Path, PathBuf};

use anyhow::Result;

/// The LM GGUF's tokenizer (parsed once per path and cached).
#[derive(Debug, Clone)]
pub struct Tokenizer {
    gguf: PathBuf,
}

impl Tokenizer {
    pub fn new(lm_gguf: &Path) -> Self {
        Self {
            gguf: lm_gguf.to_path_buf(),
        }
    }

    /// Ids of `text`; special tokens written out (`<|im_start|>`) map to
    /// their ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        rlx_core::tokenizer::encode_prompt_from_gguf(&self.gguf, text)
    }

    /// Text of `ids`, special tokens dropped when `skip_special`.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        rlx_core::tokenizer::decode_ids_from_gguf(&self.gguf, ids, skip_special)
    }
}

impl crate::Lfm3Audio {
    /// Transcribe 16 kHz mono `pcm`. `language` is the English name of the
    /// spoken language (`"English"`; `None` lets the model pick); greedy, at
    /// most `max_tokens` tokens.
    pub fn transcribe(
        &mut self,
        tok: &Tokenizer,
        pcm: &[f32],
        language: Option<&str>,
        max_tokens: usize,
    ) -> Result<String> {
        let audio = self.encode_audio(pcm)?;
        let segs = crate::prompt::asr(&self.cfg.tokens, &|s| tok.encode(s), language, audio)?;
        let mut opts = crate::GenerateOptions::new(&self.cfg);
        opts.text_only = true;
        opts.max_new_tokens = max_tokens;
        let out = self.generate(&segs, &opts, |_| {}, |_| {})?;
        let text = tok.decode(&out.text_tokens(self.cfg.tokens.audio_out), true)?;
        Ok(text.trim().to_string())
    }

    /// Speak `text` as 24 kHz mono audio. `language` is a TTS code (`"en"`,
    /// [`crate::prompt::TTS_LANGUAGES`]), `voice` `"female"` or `"male"`;
    /// `seed` drives the card's speech sampling. At most `max_steps`
    /// generation steps, one per speech frame (12.5 frames per second).
    pub fn speak(
        &mut self,
        tok: &Tokenizer,
        text: &str,
        language: &str,
        voice: &str,
        seed: u64,
        max_steps: usize,
    ) -> Result<Vec<f32>> {
        let segs = crate::prompt::tts(&self.cfg.tokens, &|s| tok.encode(s), text, language, voice)?;
        let mut opts = crate::GenerateOptions::new(&self.cfg);
        opts.max_new_tokens = max_steps;
        opts.seed = seed;
        let out = self.generate(&segs, &opts, |_| {}, |_| {})?;
        self.decode_frames(&out.frames)
    }
}
