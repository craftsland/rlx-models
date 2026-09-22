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

//! High-level single-page OCR session.

use crate::config::JinaOcrConfig;
use crate::hub::default_model_dir;
use crate::postprocess::{Crop, extract_markdown_and_crops};
use crate::preprocess::ImageMode;
use crate::prompt::DEFAULT_OCR_PROMPT;
use crate::runner::{JinaOcrRunner, default_sample_opts};
use anyhow::Result;
use rlx_runtime::Device;
use rlx_unlimited_ocr::device::resolve_device;
use rlx_unlimited_ocr::generation::SampleOpts;
use rlx_unlimited_ocr::lm_precision::LmWeightPrecision;
use rlx_unlimited_ocr::runner::RunnerOptions;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct InferenceOptions {
    pub device: Device,
    pub sample: SampleOpts,
    /// `None` uses whatever `processor_config.json` selects (Gundam by default).
    pub mode: Option<ImageMode>,
    pub prompt: String,
    pub preload: bool,
    pub weight_precision: LmWeightPrecision,
    /// Convert grounded `<|ref|>…<|det|>` spans into markdown image links.
    pub render_markdown: bool,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self::for_ocr()
    }
}

impl InferenceOptions {
    pub fn for_ocr() -> Self {
        Self {
            device: resolve_device(None).unwrap_or(Device::Cpu),
            sample: default_sample_opts(),
            mode: None,
            prompt: DEFAULT_OCR_PROMPT.to_string(),
            preload: false,
            weight_precision: LmWeightPrecision::Auto,
            render_markdown: true,
        }
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn device_name(mut self, name: &str) -> Result<Self> {
        self.device = resolve_device(Some(name))?;
        Ok(self)
    }

    pub fn mode(mut self, mode: ImageMode) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    pub fn max_new_tokens(mut self, n: usize) -> Self {
        self.sample.max_new_tokens = n;
        self
    }

    pub fn weight_precision(mut self, p: LmWeightPrecision) -> Self {
        self.weight_precision = p;
        self
    }

    pub fn preload(mut self, preload: bool) -> Self {
        self.preload = preload;
        self
    }

    pub fn render_markdown(mut self, render: bool) -> Self {
        self.render_markdown = render;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct OcrResult {
    /// Markdown, with element refs dropped and image refs linked (or the raw
    /// cleaned text when `render_markdown` is off).
    pub markdown: String,
    /// Cleaned model output before markdown rendering.
    pub text: String,
    /// Grounded image regions, in output order.
    pub crops: Vec<Crop>,
    pub prompt_len: usize,
    pub new_tokens: usize,
    pub token_ids: Vec<u32>,
}

pub struct JinaOcrSession {
    runner: JinaOcrRunner,
    options: InferenceOptions,
}

impl JinaOcrSession {
    pub fn open(model_dir: impl AsRef<Path>, options: InferenceOptions) -> Result<Self> {
        let mut runner = JinaOcrRunner::open_with(
            model_dir.as_ref(),
            RunnerOptions::new(options.device).weight_precision(options.weight_precision),
        )?;
        if options.preload {
            runner.load_weights()?;
        }
        Ok(Self { runner, options })
    }

    pub fn open_default() -> Result<Self> {
        Self::open(default_model_dir()?, InferenceOptions::for_ocr())
    }

    pub fn device(&self) -> Device {
        self.runner.device()
    }

    pub fn model_dir(&self) -> &Path {
        self.runner.model_dir()
    }

    pub fn config(&self) -> &JinaOcrConfig {
        self.runner.config()
    }

    pub fn runner(&self) -> &JinaOcrRunner {
        &self.runner
    }

    pub fn runner_mut(&mut self) -> &mut JinaOcrRunner {
        &mut self.runner
    }

    /// Transcribe one page.
    #[cfg(feature = "tokenizer")]
    pub fn run_single(&mut self, image_path: impl AsRef<Path>) -> Result<OcrResult> {
        let path: PathBuf = image_path.as_ref().to_path_buf();
        let image = crate::preprocess::load_image(&path)?;
        let mode = self
            .options
            .mode
            .unwrap_or_else(|| self.runner.default_image_mode());
        let pre = crate::preprocess::preprocess_one(&image, mode);
        let out = self
            .runner
            .generate(&pre, &self.options.prompt, &self.options.sample)?;

        let (markdown, crops) = if self.options.render_markdown {
            extract_markdown_and_crops(&out.text, pre.orig_w, pre.orig_h, true)
        } else {
            (out.text.clone(), Vec::new())
        };
        Ok(OcrResult {
            markdown,
            text: out.text,
            crops,
            prompt_len: out.prompt_len,
            new_tokens: out.new_tokens,
            token_ids: out.token_ids,
        })
    }
}
