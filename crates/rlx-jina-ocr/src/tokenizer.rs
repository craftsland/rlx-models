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

//! `tokenizer.json` (LlamaTokenizerFast, 129 280 ids), loaded once.
//!
//! Held by value rather than re-read per call: prompt assembly tokenizes both
//! sides of the `<image>` marker, and decode runs once per generation, so a
//! per-call `Tokenizer::from_file` would parse a multi-megabyte JSON several
//! times per page.

use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Loaded tokenizer for a checkpoint directory.
pub struct JinaTokenizer {
    inner: Tokenizer,
}

impl JinaTokenizer {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let inner =
            Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("load {path:?}: {e}"))?;
        Ok(Self { inner })
    }

    /// `tokenizer.encode(text, add_special_tokens=False)`.
    ///
    /// No BOS: `tokenizer_config.json` sets `add_bos_token: false` and the
    /// processor calls `text_encode(..., bos=False)`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// `tokenizer.decode(ids, skip_special_tokens=False)`.
    ///
    /// Specials are kept here and stripped textually by
    /// [`crate::postprocess::clean_decoded_text`] — skipping them at decode
    /// time would turn an EOS-only continuation into an empty string with no
    /// way to tell it apart from a genuinely empty transcript.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
            .with_context(|| format!("decode {} ids", ids.len()))
    }

    /// Id of a token's exact surface form, when present in the vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
