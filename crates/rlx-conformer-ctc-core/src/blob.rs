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

//! The `.rcc` model file: the shared [`rlx_mcu::blob`] format with this
//! model's header. The record order is defined by [`Model::parse`] and the
//! exporter in `rlx-conformer-ctc`.
//!
//! [`Model::parse`]: crate::Model::parse

pub use rlx_mcu::blob::*;

/// `"RCC1"` read as a little-endian word.
pub const MAGIC: u32 = u32::from_le_bytes(*b"RCC1");
/// Bumped whenever the record order or a record layout changes.
pub const VERSION: u32 = 1;
