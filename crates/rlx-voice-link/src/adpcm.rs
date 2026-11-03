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

//! IMA ADPCM: 16-bit PCM to 4 bits per sample (4:1), a few integer
//! operations per sample, so a 64 MHz Cortex-M4 encodes 16 kHz audio in well
//! under 1% of its time.

const STEPS: [i16; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];
const INDEX_ADJ: [i8; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

/// Codec state: the running prediction and step index. Sent at the start of
/// every packet, so a lost packet never desynchronizes the decoder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct State {
    /// Predicted sample.
    pub predictor: i16,
    /// Index into the step table, `0..=88`.
    pub index: u8,
}

impl State {
    fn apply(&mut self, code: u8) -> i16 {
        let step = STEPS[self.index as usize] as i32;
        let mut delta = step >> 3;
        if code & 4 != 0 {
            delta += step;
        }
        if code & 2 != 0 {
            delta += step >> 1;
        }
        if code & 1 != 0 {
            delta += step >> 2;
        }
        let p = if code & 8 != 0 {
            self.predictor as i32 - delta
        } else {
            self.predictor as i32 + delta
        };
        self.predictor = p.clamp(-32768, 32767) as i16;
        self.index = (self.index as i32 + INDEX_ADJ[code as usize] as i32).clamp(0, 88) as u8;
        self.predictor
    }

    /// Encode one sample to a 4-bit code.
    pub fn encode(&mut self, sample: i16) -> u8 {
        let step = STEPS[self.index as usize] as i32;
        let mut diff = sample as i32 - self.predictor as i32;
        let mut code = 0u8;
        if diff < 0 {
            code = 8;
            diff = -diff;
        }
        if diff >= step {
            code |= 4;
            diff -= step;
        }
        if diff >= step >> 1 {
            code |= 2;
            diff -= step >> 1;
        }
        if diff >= step >> 2 {
            code |= 1;
        }
        // Track exactly what the decoder will reconstruct.
        self.apply(code);
        code
    }

    /// Decode one 4-bit code.
    pub fn decode(&mut self, code: u8) -> i16 {
        self.apply(code & 0xf)
    }
}

/// Encode `pcm` into `out` (two samples per byte, low nibble first).
/// `pcm.len()` must be even and `out.len() >= pcm.len() / 2`.
pub fn encode(state: &mut State, pcm: &[i16], out: &mut [u8]) {
    for (pair, o) in pcm.as_chunks::<2>().0.iter().zip(out.iter_mut()) {
        let lo = state.encode(pair[0]);
        let hi = state.encode(pair[1]);
        *o = lo | (hi << 4);
    }
}

/// Decode `data` into `out` (`2 * data.len()` samples).
pub fn decode(state: &mut State, data: &[u8], out: &mut [i16]) {
    for (&b, o) in data.iter().zip(out.as_chunks_mut::<2>().0.iter_mut()) {
        o[0] = state.decode(b & 0xf);
        o[1] = state.decode(b >> 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_snr() {
        let pcm: [i16; 1600] = core::array::from_fn(|i| {
            let t = i as f64 / 16000.0;
            ((2.0 * core::f64::consts::PI * 440.0 * t).sin() * 8000.0
                + (2.0 * core::f64::consts::PI * 1300.0 * t).sin() * 3000.0) as i16
        });
        let mut enc = State::default();
        let mut bytes = [0u8; 800];
        encode(&mut enc, &pcm, &mut bytes);
        let mut dec = State::default();
        let mut out = [0i16; 1600];
        decode(&mut dec, &bytes, &mut out);
        assert_eq!(enc, dec, "encoder must track the decoder exactly");
        let (mut sig, mut err) = (0f64, 0f64);
        for (&a, &b) in pcm.iter().zip(&out).skip(100) {
            sig += (a as f64).powi(2);
            err += (a as f64 - b as f64).powi(2);
        }
        let snr = 10.0 * (sig / err).log10();
        assert!(snr > 20.0, "SNR {snr:.1} dB");
    }
}
