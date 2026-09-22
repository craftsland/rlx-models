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

//! Base weights with the LoRA already folded in.
//!
//! The merge happens **inside a [`WeightLoader`]** rather than as a pass over
//! an assembled bundle. `Qwen35Weights::from_loader` drains its loader tensor
//! by tensor, so intercepting `take` folds each delta at the moment the base
//! weight is materialized: one f32 copy, no second pass, and no separate
//! "merged checkpoint" to keep in sync.
//!
//! It also puts the merge on the *canonical* orientation. `take_transposed`
//! here is deliberately implemented as "take, merge, then transpose" instead
//! of delegating, because a delta folded into an already-transposed weight is
//! silently wrong — the model still runs and still returns probabilities.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::safetensors_checkpoint::SafetensorsMmapLoader;
use rlx_core::weight_loader::{HfTranslatingLoader, WeightLoader};
use rlx_qwen35::{Qwen35Config, Qwen35Weights};

use crate::lora::{LoraAdapter, module_path_for_gguf};

/// Wraps a loader and folds a peft LoRA into every targeted tensor as it is
/// taken.
pub struct LoraMergingLoader<L: WeightLoader> {
    inner: L,
    adapter: LoraAdapter,
    merged: HashSet<String>,
}

impl<L: WeightLoader> LoraMergingLoader<L> {
    pub fn new(inner: L, adapter: LoraAdapter) -> Self {
        Self {
            inner,
            adapter,
            merged: HashSet::new(),
        }
    }

    /// Adapter modules that were never folded into anything.
    ///
    /// A non-empty result means the name map and the adapter disagree, which
    /// would leave those layers running as the bare base model — accuracy
    /// drops but nothing errors, so callers should treat it as fatal.
    pub fn unmerged(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .adapter
            .modules()
            .filter(|m| !self.merged.contains(*m))
            .map(str::to_string)
            .collect();
        v.sort();
        v
    }

    pub fn merged_count(&self) -> usize {
        self.merged.len()
    }

    fn apply(&mut self, key: &str, data: &mut [f32], shape: &[usize]) -> Result<()> {
        let Some(module) = module_path_for_gguf(key) else {
            return Ok(());
        };
        if self.adapter.get(&module).is_none() {
            return Ok(());
        }
        if shape.len() != 2 {
            bail!("{key}: adapter targets {module} but the base tensor is rank {shape:?}");
        }
        if self.adapter.merge_into(&module, data, shape[0], shape[1])? {
            self.merged.insert(module);
        }
        Ok(())
    }
}

impl<L: WeightLoader> WeightLoader for LoraMergingLoader<L> {
    fn format_id(&self) -> &'static str {
        self.inner.format_id()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (mut data, shape) = self.inner.take(key)?;
        self.apply(key, &mut data, &shape)?;
        Ok((data, shape))
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take(key)?;
        if shape.len() != 2 {
            bail!("transpose requires 2D, got {shape:?}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        let mut t = vec![0f32; data.len()];
        for i in 0..rows {
            for j in 0..cols {
                t[j * rows + i] = data[i * cols + j];
            }
        }
        Ok((t, vec![cols, rows]))
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }

    fn arch_hint(&self) -> Option<&str> {
        self.inner.arch_hint()
    }
}

/// A base checkpoint plus the adapter, resolved into a runnable bundle.
pub struct KevWeights {
    pub cfg: Qwen35Config,
    pub weights: Qwen35Weights,
    /// How many adapter modules were folded in.
    pub merged: usize,
}

/// Load an HF Qwen3.5 base directory and fold `adapter` into it.
///
/// `base_dir` is a plain `Qwen/Qwen3.5-*-Base` snapshot (`config.json` +
/// safetensors shards). Fails if any adapter module finds no home rather
/// than running a partially-adapted model.
pub fn load_merged(base_dir: &Path, adapter: LoraAdapter) -> Result<KevWeights> {
    let cfg_path = base_dir.join("config.json");
    let cfg = Qwen35Config::from_hf_config_json(&cfg_path)
        .with_context(|| format!("reading {}", cfg_path.display()))?;

    let base = SafetensorsMmapLoader::open(base_dir)
        .with_context(|| format!("opening safetensors in {}", base_dir.display()))?;
    let mut loader = LoraMergingLoader::new(HfTranslatingLoader::new(base), adapter);

    let weights = Qwen35Weights::from_loader(&mut loader, &cfg)
        .with_context(|| format!("assembling Qwen3.5 weights from {}", base_dir.display()))?;

    let unmerged = loader.unmerged();
    if !unmerged.is_empty() {
        bail!(
            "{} adapter module(s) were never merged, so those layers would run as \
             the bare base model: {}{}",
            unmerged.len(),
            unmerged
                .iter()
                .take(6)
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            if unmerged.len() > 6 { ", …" } else { "" }
        );
    }

    Ok(KevWeights {
        cfg,
        weights,
        merged: loader.merged_count(),
    })
}
