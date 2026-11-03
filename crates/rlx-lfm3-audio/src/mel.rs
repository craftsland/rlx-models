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

//! The speech encoder's log-mel frontend (`Lfm3AudioFeatureExtractor`).
//!
//! 16 kHz mono, 25 ms periodic Hann window every 10 ms, no centering, power
//! spectrum, 128 Slaney-scale / Slaney-normalized mel filters over 0–8 kHz,
//! `log10` clamped at −12, then `(x + 4) / 4`. The clip is right-padded with
//! zeros so no trailing samples are dropped.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
const LOG_FLOOR: f64 = -12.0;
const NORM_OFFSET: f64 = 4.0;
const NORM_DIVISOR: f64 = 4.0;

/// Samples after the right padding: at least one window, then whole hops.
pub fn padded_len(n: usize) -> usize {
    if n < N_FFT {
        N_FFT
    } else {
        n + (HOP - (n - N_FFT) % HOP) % HOP
    }
}

/// Spectrogram frames of a clip of `n` samples.
pub fn num_frames(n: usize) -> usize {
    1 + (padded_len(n) - N_FFT) / HOP
}

/// Slaney mel filters `[n_freqs, n_mels]` (`transformers.audio_utils.mel_filter_bank`
/// with `norm="slaney"`, `mel_scale="slaney"`).
pub fn mel_filters(n_freqs: usize, n_mels: usize, f_min: f64, f_max: f64, sr: f64) -> Vec<f64> {
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = 27.0 / 6.4f64.ln();
    let hz_to_mel = |f: f64| {
        if f >= MIN_LOG_HZ {
            MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() * logstep
        } else {
            3.0 * f / 200.0
        }
    };
    let mel_to_hz = |m: f64| {
        if m >= MIN_LOG_MEL {
            MIN_LOG_HZ * ((m - MIN_LOG_MEL) / logstep).exp()
        } else {
            200.0 * m / 3.0
        }
    };
    let (m0, m1) = (hz_to_mel(f_min), hz_to_mel(f_max));
    let pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|i| (sr / 2.0).floor() * i as f64 / (n_freqs - 1) as f64)
        .collect();
    let mut fb = vec![0.0; n_freqs * n_mels];
    for (k, &f) in fft_freqs.iter().enumerate() {
        for m in 0..n_mels {
            let down = (f - pts[m]) / (pts[m + 1] - pts[m]);
            let up = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1]);
            let enorm = 2.0 / (pts[m + 2] - pts[m]);
            fb[k * n_mels + m] = down.min(up).max(0.0) * enorm;
        }
    }
    fb
}

/// Normalized log-mel features `[n_mels, frames]` (mel-major) of 16 kHz audio.
pub fn log_mel(samples: &[f32], n_mels: usize) -> Vec<f32> {
    let frames = num_frames(samples.len());
    let n_freqs = N_FFT / 2 + 1;
    let mut padded = samples.to_vec();
    padded.resize(padded_len(samples.len()), 0.0);
    let window: Vec<f64> = (0..N_FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())
        .collect();
    let fb = mel_filters(n_freqs, n_mels, 0.0, 8000.0, 16000.0);
    let fft = FftPlanner::<f64>::new().plan_fft_forward(N_FFT);
    let mut buf = vec![Complex::new(0.0, 0.0); N_FFT];
    let mut power = vec![0.0f64; n_freqs];
    let mut out = vec![0f32; n_mels * frames];
    for t in 0..frames {
        let start = t * HOP;
        for (i, b) in buf.iter_mut().enumerate() {
            *b = Complex::new(padded[start + i] as f64 * window[i], 0.0);
        }
        fft.process(&mut buf);
        for (p, b) in power.iter_mut().zip(&buf) {
            *p = b.norm_sqr();
        }
        for m in 0..n_mels {
            let e: f64 = (0..n_freqs).map(|k| fb[k * n_mels + m] * power[k]).sum();
            let l = e.log10().max(LOG_FLOOR);
            out[m * frames + t] = ((l + NORM_OFFSET) / NORM_DIVISOR) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_covers_every_sample() {
        assert_eq!(padded_len(10), N_FFT);
        assert_eq!(padded_len(400), 400);
        assert_eq!(padded_len(401), 560);
        assert_eq!(num_frames(560), 2);
        assert_eq!(num_frames(16000), 99);
    }

    #[test]
    fn filters_are_slaney_normalized() {
        // Each Slaney filter integrates to ~1 over Hz (checked on filters
        // spanning many 40 Hz bins; the low ones are narrower than a bin).
        let fb = mel_filters(201, 128, 0.0, 8000.0, 16000.0);
        let hz_per_bin = 8000.0 / 200.0;
        for m in [64, 100, 120] {
            let area: f64 = (0..201).map(|k| fb[k * 128 + m]).sum::<f64>() * hz_per_bin;
            assert!((area - 1.0).abs() < 0.1, "filter {m}: {area}");
        }
    }
}
