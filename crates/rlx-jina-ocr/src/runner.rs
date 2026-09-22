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

//! Wires the jina prompt/processor onto the shared DeepSeek-OCR stack.
//!
//! The DeepEncoder, projector and MoE decoder all come from
//! [`rlx_unlimited_ocr`] — identical architecture, identical tensor names. What
//! this crate supplies is the resolved [`JinaOcrConfig`] (no sliding window,
//! `rope_theta = 1e6`), the jina chat template, the BOS-less prompt-id layout
//! and the n-gram guard defaults from `modeling_deepseekocr.py`.

use crate::config::{JinaOcrConfig, NGRAM_SIZE, NGRAM_WHITELIST, NGRAM_WINDOW};
use crate::mtp::JinaMtp;
use crate::preprocess::{ImageMode, image_token_count_for, preprocess_one};
use crate::prompt::{self, DEFAULT_OCR_PROMPT};
use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use rlx_runtime::Device;
use rlx_unlimited_ocr::generation::SampleOpts;
use rlx_unlimited_ocr::lm_precision::{LmWeightPrecision, ResolvedLmPrecision};
use rlx_unlimited_ocr::preprocess::PreprocessedImage;
use rlx_unlimited_ocr::runner::{RunnerOptions, UnlimitedOcrRunner};
use rlx_unlimited_ocr::speculative::SpeculativeStats;
use rlx_unlimited_ocr::weights::UnlimitedOcrWeightStore;
use std::path::{Path, PathBuf};

#[cfg(feature = "tokenizer")]
use crate::tokenizer::JinaTokenizer;

/// The decode recipe from `modeling_deepseekocr.py`'s
/// `SlidingWindowNoRepeatNgramProcessor` defaults.
pub fn default_sample_opts() -> SampleOpts {
    SampleOpts {
        temperature: 0.0,
        repetition_penalty: 1.0,
        no_repeat_ngram_size: NGRAM_SIZE,
        ngram_window: NGRAM_WINDOW,
        ngram_whitelist: NGRAM_WHITELIST.to_vec(),
        max_new_tokens: 4096,
    }
}

/// A completed transcription.
#[derive(Debug, Clone, Default)]
pub struct OcrOutput {
    /// Generated text with special tokens stripped (`decode_ocr`).
    pub text: String,
    /// Raw decoded continuation, specials intact.
    pub raw_text: String,
    pub prompt_len: usize,
    pub new_tokens: usize,
    /// Prompt + generated ids.
    pub token_ids: Vec<u32>,
    /// Source pixel dimensions, for scaling grounded boxes.
    pub image_size: (u32, u32),
}

pub struct JinaOcrRunner {
    inner: UnlimitedOcrRunner,
    config: JinaOcrConfig,
    model_dir: PathBuf,
    #[cfg(feature = "tokenizer")]
    tokenizer: Option<JinaTokenizer>,
}

impl JinaOcrRunner {
    pub fn open(model_dir: &Path, device: Device) -> Result<Self> {
        Self::open_with(model_dir, RunnerOptions::new(device))
    }

    pub fn open_with(model_dir: &Path, opts: RunnerOptions) -> Result<Self> {
        let config = JinaOcrConfig::from_model_dir(model_dir)?;
        config.validate()?;
        let inner = UnlimitedOcrRunner::open_with_config(model_dir, config.lm.clone(), opts)?;
        Ok(Self {
            inner,
            config,
            model_dir: model_dir.to_path_buf(),
            #[cfg(feature = "tokenizer")]
            tokenizer: None,
        })
    }

    /// The shared Unlimited-OCR runner underneath.
    ///
    /// An escape hatch for harnesses that need the text-only paths — feeding
    /// token ids with no image at all, which the jina prompt builder cannot
    /// express because it always splices an image placeholder run.
    pub fn inner_mut(&mut self) -> &mut UnlimitedOcrRunner {
        &mut self.inner
    }

    pub fn config(&self) -> &JinaOcrConfig {
        &self.config
    }

    pub fn device(&self) -> Device {
        self.inner.device()
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn store(&self) -> &UnlimitedOcrWeightStore {
        self.inner.store()
    }

    pub fn resolved_weight_precision(&self) -> Option<ResolvedLmPrecision> {
        self.inner.resolved_weight_precision()
    }

    pub fn set_weight_precision(&mut self, p: LmWeightPrecision) {
        self.inner.set_weight_precision(p);
    }

    pub fn load_weights(&mut self) -> Result<()> {
        self.inner.load_weights()
    }

    /// The image mode `processor_config.json` selects.
    pub fn default_image_mode(&self) -> ImageMode {
        ImageMode::from_processor(&self.config.processor)
    }

    #[cfg(feature = "tokenizer")]
    fn tokenizer(&mut self) -> Result<&JinaTokenizer> {
        if self.tokenizer.is_none() {
            self.tokenizer = Some(JinaTokenizer::open(&self.model_dir)?);
        }
        Ok(self.tokenizer.as_ref().expect("tokenizer just loaded"))
    }

    /// Render the chat template and splice in the image placeholder run.
    ///
    /// `prompt` is the user instruction; the `<image>` marker and role prefixes
    /// come from the template, so it should not contain either.
    #[cfg(feature = "tokenizer")]
    pub fn build_prompt_ids(
        &mut self,
        prompt_text: &str,
        image: &PreprocessedImage,
    ) -> Result<Vec<u32>> {
        let n_image = image_token_count_for(&self.config, image);
        let image_token_id = self.config.image_token_id();
        let rendered = prompt::ocr_prompt_text(prompt_text);
        let tok = self.tokenizer()?;
        let ids = prompt::build_prompt_ids(&rendered, n_image, image_token_id, |chunk| {
            tok.encode(chunk)
        })?;
        ensure!(
            ids.iter().filter(|&&t| t == image_token_id).count() == n_image,
            "prompt-id assembly lost image placeholders"
        );
        Ok(ids)
    }

    /// Transcribe a preprocessed page.
    #[cfg(feature = "tokenizer")]
    pub fn generate(
        &mut self,
        image: &PreprocessedImage,
        prompt_text: &str,
        opts: &SampleOpts,
    ) -> Result<OcrOutput> {
        let prompt_ids = self.build_prompt_ids(prompt_text, image)?;
        let prompt_len = prompt_ids.len();
        let token_ids =
            self.inner
                .generate_from_ids(prompt_ids, std::slice::from_ref(image), opts)?;
        self.finish(prompt_len, token_ids, image)
    }

    /// Load the FastMTP draft head, when the checkpoint ships one.
    ///
    /// Loads the model weights if they are not already resident, because the
    /// draft head reuses the packed embedding table rather than duplicating it.
    /// It does still need its own host copy of the shared `lm_head` — see
    /// [`JinaMtp::host_bytes`] before enabling this on a memory budget.
    pub fn load_mtp(&mut self) -> Result<Option<JinaMtp>> {
        self.inner.load_weights()?;
        let pack = std::sync::Arc::clone(
            self.inner
                .packed()
                .context("packed LM weights missing after load")?,
        );
        JinaMtp::load(&self.config, self.inner.store(), pack)
    }

    /// Prompt ids and every position's pre-final-norm hidden state.
    ///
    /// For probing or priming a draft head without running generation.
    #[cfg(feature = "tokenizer")]
    pub fn prompt_hidden_states(
        &mut self,
        image: &PreprocessedImage,
        prompt_text: &str,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let prompt_ids = self.build_prompt_ids(prompt_text, image)?;
        let hidden = self
            .inner
            .prefill_hidden_states(&prompt_ids, std::slice::from_ref(image))?;
        Ok((prompt_ids, hidden))
    }

    /// [`UnlimitedOcrRunner::hidden_trace`] for a jina prompt.
    #[cfg(feature = "tokenizer")]
    pub fn hidden_trace(
        &mut self,
        image: &PreprocessedImage,
        prompt_text: &str,
        continuation: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let prompt_ids = self.build_prompt_ids(prompt_text, image)?;
        let (prompt_hidden, logits, hidden) =
            self.inner
                .hidden_trace(&prompt_ids, continuation, std::slice::from_ref(image))?;
        Ok((prompt_ids, prompt_hidden, logits, hidden))
    }

    /// [`Self::generate`] with FastMTP speculative decoding.
    ///
    /// The transcript is identical to [`Self::generate`]'s — draft tokens are
    /// kept only when they match the target's own greedy pick. The returned
    /// [`SpeculativeStats`] says whether it was worth it: each round costs one
    /// target forward either way, so an acceptance rate near zero makes this
    /// strictly slower than plain decode.
    #[cfg(feature = "tokenizer")]
    pub fn generate_with_mtp(
        &mut self,
        image: &PreprocessedImage,
        prompt_text: &str,
        opts: &SampleOpts,
        mtp: &JinaMtp,
    ) -> Result<(OcrOutput, SpeculativeStats)> {
        let prompt_ids = self.build_prompt_ids(prompt_text, image)?;
        let prompt_len = prompt_ids.len();
        let (token_ids, stats) = mtp.with_drafter(|drafter| {
            self.inner.generate_from_ids_speculative(
                prompt_ids,
                std::slice::from_ref(image),
                opts,
                drafter,
            )
        })?;
        Ok((self.finish(prompt_len, token_ids, image)?, stats))
    }

    #[cfg(feature = "tokenizer")]
    fn finish(
        &mut self,
        prompt_len: usize,
        token_ids: Vec<u32>,
        image: &PreprocessedImage,
    ) -> Result<OcrOutput> {
        let raw_text = self.tokenizer()?.decode(&token_ids[prompt_len..])?;
        Ok(OcrOutput {
            text: crate::postprocess::clean_decoded_text(&raw_text),
            raw_text,
            prompt_len,
            new_tokens: token_ids.len().saturating_sub(prompt_len),
            token_ids,
            image_size: (image.orig_w, image.orig_h),
        })
    }

    /// [`Self::generate`] straight from decoded pixels.
    #[cfg(feature = "tokenizer")]
    pub fn transcribe(
        &mut self,
        image: &DynamicImage,
        prompt_text: Option<&str>,
        opts: &SampleOpts,
    ) -> Result<OcrOutput> {
        let mode = self.default_image_mode();
        let pre = preprocess_one(image, mode);
        self.generate(&pre, prompt_text.unwrap_or(DEFAULT_OCR_PROMPT), opts)
    }

    /// [`Self::transcribe`] from a file path.
    #[cfg(feature = "tokenizer")]
    pub fn transcribe_path(
        &mut self,
        path: &Path,
        prompt_text: Option<&str>,
        opts: &SampleOpts,
    ) -> Result<OcrOutput> {
        let image =
            crate::preprocess::load_image(path).with_context(|| format!("read page {path:?}"))?;
        self.transcribe(&image, prompt_text, opts)
    }
}
