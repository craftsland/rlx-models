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

//! # rlx-mistral-large4 — Mistral Large 4 (`mistral-large4`) for RLX
//!
//! [Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)
//! — 1 T total / 52 B active. GGUF: `general.architecture = mistral-large4`
//! ([unsloth/Mistral-Large-4-1T-A52B-GGUF](https://huggingface.co/unsloth/Mistral-Large-4-1T-A52B-GGUF));
//! HF: `MistralLarge4ForConditionalGeneration`, `model_type = mistral_large4`.
//!
//! ## Architecture (text)
//!
//! 72 layers, `hidden = 6144`, `vocab = 204800`, 64 heads.
//!
//! | Piece | Module | Notes |
//! |---|---|---|
//! | MLA, **no Q-LoRA**, `kv_lora 512`, head 128 | [`attention`] | per-head SSMax query scale `s·log(pos+1)+b` |
//! | NoPE every 4th layer | [`attention`], [`indexer`] | the whole head is non-rotary there |
//! | KV sharing | [`attention`] | layers 39, 43, … 71 reuse layer 35's latent |
//! | DSA lightning indexer, top-2048 | [`indexer`] | 28 `"shared"` layers reuse the previous selection |
//! | LeRoPE: YaRN × learned `exp(alpha)` | [`rope`] | interleaved pairs; `rope_freqs.weight = exp(-alpha)` |
//! | 128-expert MoE, top-4 + 1 shared | [`moe`] | softmax router, renormalized; layer 0 dense |
//! | logit softcap 30 | [`flow`] | `c·tanh(logits/c)` |
//! | incremental decode | [`decode`] | latent-only cache, owning layers only |
//! | whole model, experts paged from disk | [`paged`] | [`PagedMistralLarge4`]; binary `rlx-mistral-large4` |
//! | generic LM path | [`runner`] | [`MistralLarge4Runner`] (`LmRunner`), runner name `mistral-large4` |
//!
//! ## Usage
//!
//! ```no_run
//! use rlx_core::flow_util::compile_built;
//! use rlx_core::weight_map::WeightMap;
//! use rlx_mistral_large4::{MistralLarge4Config, PrefillOptions, build_mistral_large4_text_flow};
//! use rlx_runtime::Device;
//! # fn main() -> anyhow::Result<()> {
//! let cfg = MistralLarge4Config::from_hf_file("config.json")?;
//! let mut weights = WeightMap::from_file("model.safetensors")?; // GGUF tensor names
//! let built = build_mistral_large4_text_flow(&cfg, &mut weights, 16, PrefillOptions::default())?;
//! let mut compiled = compile_built(built, Device::Cpu)?;
//! let logits = compiled.run(&[("input_ids", &[1.0f32; 16][..])]);
//! # let _ = logits;
//! # Ok(())
//! # }
//! ```
//!
//! ## Status
//!
//! The text decoder matches `modeling_mistral_large4.py` and every tensor
//! shape of the published GGUF. Tests: config from the shipped `config.json`
//! and GGUF metadata, the tensor manifest in both directions against the real
//! 1602-tensor index, logits parity against an independent torch port
//! (`scripts/mistral_large4_reference.py`) covering RoPE/NoPE layers, KV
//! sharing, shared indexers and a sparse (`seq > index_topk`) DSA selection,
//! and decode == prefill.
//!
//! The whole model runs through [`paged`]: routed experts (214 of the 233 GB
//! at `UD-IQ1_S`) stay on disk and are read per token; paged decode matches
//! the resident graph (`tests/paged.rs`) and the torch reference on real
//! blocks. Not done: an actual 233 GB run on this machine, decode past
//! `index_topk` (needs an indexer-key cache), the Pixtral vision tower, and
//! the DSpark MTP draft head.

pub mod attention;
pub mod config;
pub mod decode;
pub mod flow;
pub mod indexer;
pub mod moe;
pub mod paged;
pub mod rope;
pub mod runner;

pub use config::{ACCEPTED_ARCHES, HF_MODEL_TYPES, MistralLarge4Config};
pub use decode::{DecodeLayout, DecodeSession, build_mistral_large4_decode_flow};
pub use flow::{
    EMBED_KEY, HIDDEN_INPUT, PrefillOptions, build_mistral_large4_blocks_with_source,
    build_mistral_large4_text_flow, build_mistral_large4_text_flow_with_source,
};
pub use paged::{PagedMistralLarge4, PagedOptions, PagedStack, ShardedGguf, StepTimes};
pub use runner::{MistralLarge4Runner, RUNNER, register};
