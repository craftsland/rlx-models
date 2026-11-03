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

//! Tokenizer entry points, re-exported from [`rlx_core::tokenizer`] (shared by
//! every family whose GGUF carries a byte-level BPE; they lived here first).

pub use rlx_core::tokenizer::{
    QWEN2_SPLIT_REGEX, QWEN35_SPLIT_REGEX, decode_ids, decode_ids_auto, decode_ids_from_gguf,
    encode_prompt, encode_prompt_auto, encode_prompt_from_gguf, qwen_split_regex,
    resolve_tokenizer_path,
};
#[cfg(feature = "qwen35-tokenizer")]
pub use rlx_core::tokenizer::{gguf_tokenizer_json, qwen_pre_tokenizer};
