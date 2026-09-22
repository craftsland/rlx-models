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

//! Wires deep encoder + projector + compiled MoE LM for generation.
//!
//! Vision (SAM/CLIP/projector pack) stays host-f32; the MoE LM runs as a
//! compiled RLX graph on `Self::device` (including CPU).

use crate::config::{EOS_TOKEN_ID, IMAGE_TOKEN_ID, UnlimitedOcrConfig};
use crate::deep_encoder::DeepEncoder;
use crate::expert_pack::PackedLmWeights;
use crate::generation::{SampleOpts, sample_token};
use crate::lm_device::CompiledLm;
use crate::lm_precision::{LmWeightPrecision, ResolvedLmPrecision};
use crate::preprocess::PreprocessedImage;
use crate::projector::Projector;
use crate::speculative::{Drafter, SpeculativeStats, generate_speculative};
use crate::weights::UnlimitedOcrWeightStore;
use anyhow::{Result, ensure};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Runner construction options (device + LM weight storage precision).
#[derive(Debug, Clone)]
pub struct RunnerOptions {
    pub device: Device,
    /// Host pack precision for MoE / large mats. Default [`LmWeightPrecision::Auto`].
    pub weight_precision: LmWeightPrecision,
}

impl RunnerOptions {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            weight_precision: LmWeightPrecision::Auto,
        }
    }

    pub fn weight_precision(mut self, p: LmWeightPrecision) -> Self {
        self.weight_precision = p;
        self
    }
}

pub struct UnlimitedOcrRunner {
    model_dir: PathBuf,
    device: Device,
    weight_precision: LmWeightPrecision,
    config: UnlimitedOcrConfig,
    store: UnlimitedOcrWeightStore,
    encoder: DeepEncoder,
    projector: Projector,
    lm: Option<CompiledLm>,
    packed: Option<Arc<PackedLmWeights>>,
    loaded: bool,
}

impl UnlimitedOcrRunner {
    pub fn open(model_dir: &Path, device: Device) -> Result<Self> {
        Self::open_with(model_dir, RunnerOptions::new(device))
    }

    pub fn open_with(model_dir: &Path, opts: RunnerOptions) -> Result<Self> {
        Self::open_with_config(
            model_dir,
            UnlimitedOcrConfig::from_model_dir(model_dir)?,
            opts,
        )
    }

    /// [`Self::open_with`] with a caller-supplied config.
    ///
    /// DeepSeek-OCR derivatives share this decoder stack but resolve their
    /// `config.json` differently (`jinaai/jina-ocr-v1` has no `sliding_window`
    /// key at all, meaning *no window*, where Unlimited-OCR's default is 128).
    /// Such a crate parses its own card and hands the resolved config here
    /// instead of letting this crate's defaults apply.
    pub fn open_with_config(
        model_dir: &Path,
        config: UnlimitedOcrConfig,
        opts: RunnerOptions,
    ) -> Result<Self> {
        config.validate()?;
        let store = UnlimitedOcrWeightStore::open(model_dir)?;
        let encoder = DeepEncoder::from_config(&config);
        let projector = Projector::from_config(&config.projector);
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device: opts.device,
            weight_precision: opts.weight_precision,
            config,
            store,
            encoder,
            projector,
            lm: None,
            packed: None,
            loaded: false,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn weight_precision(&self) -> LmWeightPrecision {
        self.weight_precision
    }

    pub fn resolved_weight_precision(&self) -> Option<ResolvedLmPrecision> {
        self.packed.as_ref().map(|p| p.resolved_precision)
    }

    pub fn config(&self) -> &UnlimitedOcrConfig {
        &self.config
    }

    /// The packed LM weights, once [`Self::load_weights`] has run.
    ///
    /// Exposed so a host-side draft head can reuse the already-resident
    /// embedding table instead of loading a second copy of it.
    pub fn packed(&self) -> Option<&Arc<PackedLmWeights>> {
        self.packed.as_ref()
    }

    /// The compiled decoder, once [`Self::load_weights`] has run.
    ///
    /// An escape hatch for cost measurement: comparing a single-token decode
    /// against a chunked verify forward is the only way to tell whether
    /// speculation can pay on a given model, and inferring either from an
    /// end-to-end total means subtracting an estimated prefill cost — which on
    /// a loaded machine swamps the thing being measured.
    pub fn lm_mut(&mut self) -> Option<&mut CompiledLm> {
        self.lm.as_mut()
    }

    pub fn store(&self) -> &UnlimitedOcrWeightStore {
        &self.store
    }

    pub fn set_weight_precision(&mut self, p: LmWeightPrecision) {
        if self.weight_precision != p {
            self.weight_precision = p;
            // Force reload so the next generate() re-packs.
            self.loaded = false;
            self.lm = None;
            self.packed = None;
        }
    }

    pub fn load_weights(&mut self) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        self.encoder.load(&self.store)?;
        self.projector.load(&self.store)?;
        let packed = Arc::new(PackedLmWeights::from_store_for_device(
            &self.store,
            &self.config,
            self.weight_precision,
            Some(self.device),
        )?);
        eprintln!(
            "[rlx-unlimited-ocr] packed LM host cache ≈ {:.1} GiB ({})",
            packed.host_nbytes() as f64 / (1024.0 * 1024.0 * 1024.0),
            packed.resolved_precision,
        );
        self.lm = Some(CompiledLm::new(self.device, Arc::clone(&packed)));
        self.packed = Some(packed);
        self.loaded = true;
        Ok(())
    }

    pub fn build_prompt_ids(&self, prompt: &str, images: &[PreprocessedImage]) -> Result<Vec<u32>> {
        #[cfg(feature = "tokenizer")]
        {
            crate::tokenizer::build_prompt_ids(
                &self.model_dir,
                prompt,
                images,
                self.config.bos_token_id,
                IMAGE_TOKEN_ID,
            )
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            let _ = (prompt, images);
            anyhow::bail!("enable feature `tokenizer` to build prompt ids")
        }
    }

    /// Returns `(decoded_text, full_token_ids, prompt_len)`.
    pub fn generate(
        &mut self,
        prompt: &str,
        images: &[PreprocessedImage],
        opts: &SampleOpts,
    ) -> Result<(String, Vec<u32>, usize)> {
        self.load_weights()?;
        let prompt_ids = self.build_prompt_ids(prompt, images)?;
        let prompt_len = prompt_ids.len();
        let token_ids = self.generate_from_ids(prompt_ids, images, opts)?;

        #[cfg(feature = "tokenizer")]
        let text = crate::tokenizer::decode(&self.model_dir, &token_ids[prompt_len..])?;
        #[cfg(not(feature = "tokenizer"))]
        let text = format!("<tokens:{}>", token_ids.len() - prompt_len);

        Ok((text, token_ids, prompt_len))
    }

    /// [`Self::generate`] from caller-supplied prompt ids, returning the full
    /// token sequence (prompt + generated) without decoding it.
    ///
    /// The prompt-id layout is model-specific — DeepSeek-OCR derivatives differ
    /// in chat template and in whether a BOS is prepended — so crates reusing
    /// this decoder build their own ids and skip [`Self::build_prompt_ids`].
    /// `prompt_ids` must contain exactly the `image_token_id` placeholder run
    /// that `images` expands to, in order.
    pub fn generate_from_ids(
        &mut self,
        prompt_ids: Vec<u32>,
        images: &[PreprocessedImage],
        opts: &SampleOpts,
    ) -> Result<Vec<u32>> {
        self.load_weights()?;

        let vision = self.encoder.encode_and_project(images, &self.projector)?;
        let lm = self.lm.as_mut().expect("lm loaded");
        let mut inputs_embeds = lm.embed_tokens(&prompt_ids)?;
        crate::embed::fuse_inputs_embeds(
            &prompt_ids,
            &mut inputs_embeds,
            self.config.hidden_size,
            self.config.image_token_id,
            &vision,
        )?;

        let mut token_ids = prompt_ids;
        let (mut logits, mut kv) = lm.prefill(&inputs_embeds, token_ids.len())?;

        for _ in 0..opts.max_new_tokens {
            let next = sample_token(&logits, opts, &token_ids);
            token_ids.push(next);
            if next == self.config.eos_token_id || next == EOS_TOKEN_ID {
                break;
            }
            let step_embed = lm.embed_tokens(&[next])?;
            let pos = token_ids.len() - 1;
            logits = lm.decode_step(&step_embed, pos, &mut kv)?;
        }
        Ok(token_ids)
    }

    /// Prefill only, returning every position's pre-final-norm hidden state
    /// (`[prompt_ids.len(), hidden]`).
    ///
    /// For priming or probing a draft head without paying for generation.
    pub fn prefill_hidden_states(
        &mut self,
        prompt_ids: &[u32],
        images: &[PreprocessedImage],
    ) -> Result<Vec<f32>> {
        self.load_weights()?;
        let vision = self.encoder.encode_and_project(images, &self.projector)?;
        let hidden_size = self.config.hidden_size;
        let image_token_id = self.config.image_token_id;
        let lm = self.lm.as_mut().expect("lm loaded");
        let mut inputs_embeds = lm.embed_tokens(prompt_ids)?;
        crate::embed::fuse_inputs_embeds(
            prompt_ids,
            &mut inputs_embeds,
            hidden_size,
            image_token_id,
            &vision,
        )?;
        let (_, _, hidden) = lm.prefill_with_all_hidden(&inputs_embeds, prompt_ids.len())?;
        Ok(hidden)
    }

    /// Teacher-forced trace: prefill `prompt_ids`, then score `continuation`
    /// in one chunk.
    ///
    /// Returns `(prompt_hidden, per_position_logits, per_position_hidden)`.
    /// Continuation position `i` is the model's state after consuming
    /// `continuation[i]`, so its logits predict `continuation[i+1]`. Hidden
    /// states are pre-final-norm, matching [`Self::prefill_hidden_states`].
    ///
    /// The prompt's hidden states come back too because the caller almost
    /// always needs both, and re-deriving them costs another vision encode
    /// plus a full prefill.
    ///
    /// For evaluating a draft head against what the target actually does, on
    /// real generated text rather than on a prompt that is mostly image
    /// placeholders.
    pub fn hidden_trace(
        &mut self,
        prompt_ids: &[u32],
        continuation: &[u32],
        images: &[PreprocessedImage],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        ensure!(
            !continuation.is_empty(),
            "hidden_trace needs a continuation"
        );
        self.load_weights()?;
        let vision = self.encoder.encode_and_project(images, &self.projector)?;
        let hidden_size = self.config.hidden_size;
        let image_token_id = self.config.image_token_id;
        let lm = self.lm.as_mut().expect("lm loaded");
        let mut inputs_embeds = lm.embed_tokens(prompt_ids)?;
        crate::embed::fuse_inputs_embeds(
            prompt_ids,
            &mut inputs_embeds,
            hidden_size,
            image_token_id,
            &vision,
        )?;
        let (_, mut kv, prompt_hidden) =
            lm.prefill_with_all_hidden(&inputs_embeds, prompt_ids.len())?;
        let cont_embeds = lm.embed_tokens(continuation)?;
        let (logits, hidden) = lm.decode_chunk_with_hidden(
            &cont_embeds,
            prompt_ids.len(),
            continuation.len(),
            &mut kv,
        )?;
        Ok((prompt_hidden, logits, hidden))
    }

    /// [`Self::generate_from_ids`] with a draft head proposing continuations.
    ///
    /// Emits exactly what [`Self::generate_from_ids`] would — the verify pass
    /// only keeps a draft token when it equals the target's own greedy pick, so
    /// this is a speed trade and never a quality one. The returned
    /// [`SpeculativeStats`] is what tells you whether it actually bought
    /// anything: a draft head with a poor acceptance rate makes generation
    /// *slower*, because every round still costs one target forward.
    ///
    /// Requires the full-causal path — the chunked verify pass has no windowed
    /// equivalent, since a sliding window would drop history mid-chunk.
    pub fn generate_from_ids_speculative(
        &mut self,
        prompt_ids: Vec<u32>,
        images: &[PreprocessedImage],
        opts: &SampleOpts,
        drafter: &mut dyn Drafter,
    ) -> Result<(Vec<u32>, SpeculativeStats)> {
        ensure!(
            self.config.sliding_window == 0,
            "speculative decode needs the full-causal path, but sliding_window = {}",
            self.config.sliding_window
        );
        self.load_weights()?;

        let vision = self.encoder.encode_and_project(images, &self.projector)?;
        let hidden_size = self.config.hidden_size;
        let image_token_id = self.config.image_token_id;
        let eos = [self.config.eos_token_id, EOS_TOKEN_ID];
        let pack = Arc::clone(self.packed.as_ref().expect("packed loaded"));

        let lm = self.lm.as_mut().expect("lm loaded");
        let mut inputs_embeds = lm.embed_tokens(&prompt_ids)?;
        crate::embed::fuse_inputs_embeds(
            &prompt_ids,
            &mut inputs_embeds,
            hidden_size,
            image_token_id,
            &vision,
        )?;

        let mut token_ids = prompt_ids;
        let (logits, mut kv, hidden) =
            lm.prefill_with_all_hidden(&inputs_embeds, token_ids.len())?;

        // The embedding lookup has to go through the pack rather than the LM,
        // which the loop holds mutably.
        let mut embed = move |ids: &[u32]| pack.embed_tokens_lookup(ids);
        let stats = generate_speculative(
            lm,
            &mut kv,
            drafter,
            &mut token_ids,
            logits,
            hidden,
            opts,
            &eos,
            &mut embed,
        )?;
        Ok((token_ids, stats))
    }
}
