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

//! Open-vocabulary speech recognition without an operating system.
//!
//! NVIDIA's `stt_en_conformer_ctc_small` (13M parameters) with int8 weights,
//! per-row dynamic int8 activations and int8 attention — the recipe the Oído
//! paper ([arXiv:2610.13447](https://arxiv.org/abs/2610.13447)) measured on an
//! ESP32-S3 — written from scratch for `no_std`:
//!
//! * the model is one flat `.rcc` blob read in place, so it can live in
//!   memory-mapped flash (about 14 MB for the small model);
//! * nothing allocates: [`scratch_len`] says how much RAM a given audio
//!   length needs, the caller provides it (PSRAM on an ESP32);
//! * all transcendentals go through `libm`, so a host build and an MCU build
//!   produce the same bits.
//!
//! The `.rcc` file is produced on the host by `rlx-conformer-ctc export-mcu`
//! from the original `.nemo`.
//!
//! ```no_run
//! # use rlx_conformer_ctc_core::{Model, Frontend, transcribe, scratch_len};
//! # let blob: &[u8] = &[]; let pcm: &[i16] = &[];
//! let model = Model::parse(blob).unwrap();
//! let fe = Frontend::new(model.config().n_fft);
//! let need = scratch_len(&model, pcm.len());
//! # let mut fs = vec![0f32; need.f32s]; let mut is = vec![0i8; need.i8s];
//! let mut ids = [0u16; 256];
//! let n = transcribe(&model, &fe, pcm, &mut fs, &mut is, &mut ids).unwrap();
//! let mut text = [0u8; 512];
//! let len = model.decode(&ids[..n], &mut text);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

pub mod blob;
mod encoder;
pub mod frontend;
pub mod model;

pub use encoder::{ScratchLen, scratch_len};
pub use frontend::Frontend;
pub use model::{Config, Model};

pub use rlx_mcu::{Error, Pcm};

/// PCM → token ids (greedy CTC). `fs` / `is` must be at least
/// [`scratch_len`] for `pcm.len()`. Returns the number of ids written.
pub fn transcribe<P: Pcm + ?Sized>(
    model: &Model<'_>,
    frontend: &Frontend,
    pcm: &P,
    fs: &mut [f32],
    is: &mut [i8],
    ids: &mut [u16],
) -> Result<usize, Error> {
    let cfg = model.config();
    let frames = Frontend::frames(model, pcm.len());
    {
        let (mel, work) = encoder::mel_and_fft(cfg, frames, fs)?;
        frontend.log_mel(model, pcm, mel, work);
    }
    encoder::run(model, frames, fs, is, ids, None)
}

/// [`transcribe`], also copying the encoder output (`[frames, d_model]`,
/// row-major) into `features`. For parity checks against the f32 model.
#[doc(hidden)]
pub fn transcribe_with_features<P: Pcm + ?Sized>(
    model: &Model<'_>,
    frontend: &Frontend,
    pcm: &P,
    fs: &mut [f32],
    is: &mut [i8],
    ids: &mut [u16],
    features: &mut [f32],
) -> Result<usize, Error> {
    let cfg = model.config();
    let frames = Frontend::frames(model, pcm.len());
    {
        let (mel, work) = encoder::mel_and_fft(cfg, frames, fs)?;
        frontend.log_mel(model, pcm, mel, work);
    }
    encoder::run(model, frames, fs, is, ids, Some(features))
}
