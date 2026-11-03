// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Distill a small streaming Zipformer2 from a larger one, trained with RLX
//! (the `rlx!`/`Func` training stack, Metal autodiff), into the format
//! `rlx-kroko`'s exporter and `rlx-kroko-core` (the `no_std` MCU runtime)
//! read.
//!
//! * [`model`]: the student's forward pass as an RLX graph. It is the same
//!   per-chunk computation as `rlx-kroko-core` (conv embed with its 3-frame
//!   cache, six stacks, per-layer key/value/conv caches, compact relative
//!   positions, chunk-local convs with edge scales), unrolled over `K`
//!   chunks of a segment from a zero state. Training and the MCU therefore
//!   run the same function.
//! * The target is the teacher's encoder output (after its joiner-side
//!   projection), computed by `rlx-kroko-core` on the same segments
//!   (`zipformer-distill-prep`). The student keeps the teacher's decoder and
//!   joiner, so matching the encoder output is the whole task.
//! * [`init`]: student parameters start from the teacher's, cut to the
//!   student's widths (same names, leading rows/columns, leading heads).
//! * [`export`]: parameters are named as `rlx-kroko` expects, so export is a
//!   save plus the teacher's decoder/joiner and a `config.json`.

pub mod config;
pub mod data;
pub mod export;
pub mod init;
pub mod joiner;
pub mod model;

pub use config::StudentConfig;
