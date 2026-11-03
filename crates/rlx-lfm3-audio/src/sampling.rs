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

//! Token sampling as `transformers` does it: repetition penalty, then
//! temperature, then top-k, then a draw from the softmax. Temperature 0 is
//! greedy.

/// SplitMix64: small, seedable, good enough for sampling.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A uniformly random permutation of `0..n` (Fisher–Yates).
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        p
    }
}

/// One sampling setting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// 0 = greedy.
    pub temperature: f32,
    /// 0 = no top-k cut.
    pub top_k: usize,
    /// 1 = off. Divides positive (multiplies negative) logits of tokens in
    /// the history.
    pub repetition_penalty: f32,
}

impl Sampling {
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_k: 0,
        repetition_penalty: 1.0,
    };
}

/// Pick a token from `logits` (modified in place).
pub fn sample(
    logits: &mut [f32],
    s: Sampling,
    history: impl IntoIterator<Item = u32>,
    suppress: &[u32],
    rng: &mut Rng,
) -> u32 {
    if s.repetition_penalty != 1.0 {
        let mut seen = vec![false; logits.len()];
        for t in history {
            let t = t as usize;
            if t < logits.len() && !seen[t] {
                seen[t] = true;
                let v = logits[t];
                logits[t] = if v < 0.0 {
                    v * s.repetition_penalty
                } else {
                    v / s.repetition_penalty
                };
            }
        }
    }
    for &t in suppress {
        if let Some(v) = logits.get_mut(t as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
    if s.temperature <= 0.0 {
        return argmax(logits);
    }
    for v in logits.iter_mut() {
        *v /= s.temperature;
    }
    if s.top_k > 0 && s.top_k < logits.len() {
        let mut sorted: Vec<f32> = logits.to_vec();
        sorted.select_nth_unstable_by(s.top_k - 1, |a, b| b.total_cmp(a));
        let kth = sorted[s.top_k - 1];
        for v in logits.iter_mut() {
            if *v < kth {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = logits.iter().map(|&v| ((v - max) as f64).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut r = rng.uniform() * total;
    for (i, w) in weights.iter().enumerate() {
        if r < *w {
            return i as u32;
        }
        r -= w;
    }
    argmax(logits)
}

pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_respects_suppression_and_penalty() {
        let mut rng = Rng::new(1);
        let mut l = vec![1.0, 3.0, 2.0];
        assert_eq!(sample(&mut l, Sampling::GREEDY, [], &[1], &mut rng), 2);
        let s = Sampling {
            repetition_penalty: 2.0,
            ..Sampling::GREEDY
        };
        let mut l = vec![1.0, 3.0, 2.0];
        assert_eq!(sample(&mut l, s, [1], &[], &mut rng), 2);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut rng = Rng::new(7);
        let s = Sampling {
            temperature: 1.0,
            top_k: 1,
            repetition_penalty: 1.0,
        };
        for _ in 0..20 {
            let mut l = vec![0.5, 0.1, 0.9, 0.2];
            assert_eq!(sample(&mut l, s, [], &[], &mut rng), 2);
        }
    }

    #[test]
    fn permutation_is_a_permutation() {
        let mut p = Rng::new(3).permutation(12);
        p.sort_unstable();
        assert_eq!(p, (0..12).collect::<Vec<_>>());
    }
}
