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

//! [`LmRunner`] over [`PagedMistralLarge4`], and its registration with the
//! generic LM path under the runner name [`RUNNER`].
//!
//! The model advances one token per step, so the runner keeps the token
//! history its caches hold: a call whose prompt **extends** that history only
//! feeds the new suffix (which makes the trait's re-prefilling default
//! `generate` linear rather than quadratic), anything else resets and feeds
//! the whole prompt.

use anyhow::{Result, bail};
use rlx_runtime::LmRunner;
use std::path::Path;

use crate::paged::{PagedMistralLarge4, PagedOptions};

/// Runner name in `rlx_core::lm_registry` / `rlx_core::model_registry`.
pub const RUNNER: &str = "mistral-large4";

/// Default context when the caller gives no `max_seq`.
pub const DEFAULT_CONTEXT: usize = 2047;

/// Pager LRU budget when the caller gives no memory cap.
pub const DEFAULT_CACHE_BYTES: u64 = 8 << 30;

/// Mistral Large 4 as an [`LmRunner`].
pub struct MistralLarge4Runner {
    model: PagedMistralLarge4,
    /// Tokens the caches currently hold, in order.
    history: Vec<u32>,
    /// Logits after the last token of `history`.
    last: Vec<f32>,
}

impl MistralLarge4Runner {
    /// Open a checkpoint (any shard of a split GGUF, or a single file).
    pub fn open(path: impl AsRef<Path>, opts: &PagedOptions) -> Result<Self> {
        Ok(Self {
            model: PagedMistralLarge4::open(path, opts)?,
            history: Vec::new(),
            last: Vec::new(),
        })
    }

    pub fn model(&self) -> &PagedMistralLarge4 {
        &self.model
    }

    /// Make the caches hold exactly `prompt`; returns its last logits.
    fn sync(&mut self, prompt: &[u32]) -> Result<Vec<f32>> {
        if prompt.is_empty() {
            bail!("mistral-large4: empty prompt");
        }
        let reuse = prompt.len() > self.history.len() && prompt.starts_with(&self.history);
        if !reuse && prompt != self.history.as_slice() {
            self.model.reset();
            self.history.clear();
        }
        for &t in &prompt[self.history.len()..] {
            self.last = self.model.step(t)?;
            self.history.push(t);
        }
        Ok(self.last.clone())
    }

    fn feed(&mut self, token: u32) -> Result<Vec<f32>> {
        self.last = self.model.step(token)?;
        self.history.push(token);
        Ok(self.last.clone())
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0 as u32
}

impl LmRunner for MistralLarge4Runner {
    fn family(&self) -> &'static str {
        RUNNER
    }

    fn vocab_size(&self) -> usize {
        self.model.cfg.vocab_size
    }

    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.sync(prompt_ids)
    }

    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let mut logits = self.sync(prompt_ids)?;
        let mut out = Vec::with_capacity(n_new);
        for i in 0..n_new {
            let next = argmax(&logits);
            out.push(next);
            if !on_token(next) || i + 1 == n_new {
                break;
            }
            logits = self.feed(next)?;
        }
        Ok(out)
    }

    fn prefill_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.sync(prompt_ids)
    }

    fn decode_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.feed(token)
    }
}

/// Register [`MistralLarge4Runner`] with `rlx_core::lm_registry` under
/// [`RUNNER`]. `max_seq` sets the context (≤ 2047, the default); the expert
/// cache gets [`DEFAULT_CACHE_BYTES`] on top of the ~26 GB resident — open
/// [`MistralLarge4Runner`] directly to choose another budget.
pub fn register() {
    use rlx_core::lm_registry::{LmRunnerRegistration, register_lm_runner};
    register_lm_runner(
        LmRunnerRegistration::new(RUNNER, |path, opts| {
            Ok(Box::new(MistralLarge4Runner::open(path, &options(opts))?))
        })
        .mmproj_note("the Pixtral vision tower of Mistral Large 4 is not ported; text only"),
    );
}

/// [`PagedOptions`] from the generic build options.
pub fn options(opts: &rlx_core::lm_registry::LmBuildOptions) -> PagedOptions {
    PagedOptions {
        device: opts.device_or_auto(),
        cap: opts.max_seq.unwrap_or(DEFAULT_CONTEXT),
        cache_bytes: DEFAULT_CACHE_BYTES,
    }
}
