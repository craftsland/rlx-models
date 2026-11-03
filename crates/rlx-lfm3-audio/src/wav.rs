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

//! WAV in (any rate, resampled to 16 kHz) and out (16-bit mono).

use std::path::Path;

use anyhow::{Context, Result};

use crate::config::INPUT_SAMPLE_RATE;

/// Mono samples of `path` at the speech encoder's 16 kHz.
pub fn read_16k(path: &Path) -> Result<Vec<f32>> {
    let (pcm, rate) = rlx_core::asr_bench::read_wav_mono(path)
        .with_context(|| format!("read {}", path.display()))?;
    Ok(if rate == INPUT_SAMPLE_RATE {
        pcm
    } else {
        rlx_core::audio::resample_sinc(&pcm, rate as usize, INPUT_SAMPLE_RATE as usize)
    })
}

/// Write `pcm` (clipped to ±1) as 16-bit mono PCM.
pub fn write_mono_16(path: &Path, pcm: &[f32], rate: u32) -> Result<()> {
    let data_len = (pcm.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + pcm.len() * 2);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b).with_context(|| format!("write {}", path.display()))
}
