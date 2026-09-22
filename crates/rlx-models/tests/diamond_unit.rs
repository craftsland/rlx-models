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

//! Host-side Diamond Maps math (no model weights).

use rlx_diamond::{
    BluenessReward, DenoiserReference, LatentReward, glass, glass_integrate::sample_posterior,
    guidance_coefficient, log_mean_exp, softmax_grad_aggregate,
};

struct LinearDenoiser;

impl DenoiserReference for LinearDenoiser {
    fn denoise(&self, _t_star: f32, x_star: &[f32], out: &mut [f32]) {
        for (o, &x) in out.iter_mut().zip(x_star.iter()) {
            *o = x * 0.9;
        }
    }
}

#[test]
fn glass_posterior_writes_a_deterministic_sample() {
    let x_t = vec![0.5f32; 8];
    let noise = vec![0.1f32; 8];
    let mut z = vec![0.0f32; 8];
    sample_posterior(&LinearDenoiser, 0.3, 1.0, &x_t, 5, &noise, &mut z);

    // `out_z` is handed in pre-filled with zeros, so the old
    // `assert!(z.iter().all(is_finite))` passed whether or not
    // `sample_posterior` wrote anything at all.
    assert!(z.iter().all(|v| v.is_finite()), "non-finite sample");
    assert!(
        z.iter().any(|v| v.abs() > 1e-9),
        "sample_posterior left out_z untouched"
    );
    // Same inputs, same state: it is a deterministic function of its arguments,
    // the noise included.
    let mut again = vec![0.0f32; 8];
    sample_posterior(&LinearDenoiser, 0.3, 1.0, &x_t, 5, &noise, &mut again);
    assert_eq!(z, again, "sample_posterior is not deterministic");
    // Every input element is identical, so every output element must be too —
    // the integrator must not be mixing across positions.
    assert!(
        z.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6),
        "identical inputs produced position-dependent output: {z:?}"
    );
    // The noise is an input, not decoration.
    let mut other = vec![0.0f32; 8];
    let louder: Vec<f32> = noise.iter().map(|v| v * 4.0).collect();
    sample_posterior(&LinearDenoiser, 0.3, 1.0, &x_t, 5, &louder, &mut other);
    assert!(
        z.iter().zip(&other).any(|(a, b)| (a - b).abs() > 1e-9),
        "scaling the noise changed nothing — it is being ignored"
    );
}

#[test]
fn blueness_reward_increases_with_blue_channel() {
    let r = BluenessReward { scale: 1.0 };
    let low = vec![0.0f32, 0.0, 0.1, 0.0, 0.0, 0.1];
    let high = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
    assert!(r.reward(&high) > r.reward(&low));
}

#[test]
fn value_softmax_grad() {
    let rewards = [0.0f32, 1.0, 0.5];
    let grads = [vec![1.0f32, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
    let g = softmax_grad_aggregate(&rewards, &grads);
    assert_eq!(g.len(), 2);
    let v = log_mean_exp(&rewards);
    assert!(v > 0.5);
}

#[test]
fn guidance_coeff_positive_midtime() {
    let b = guidance_coefficient(0.4);
    assert!(b > 0.0);
}

#[test]
fn early_stop_ddpm_is_linear_in_its_observations() {
    let (t, t_prime) = (0.2f32, 0.3f32);
    let s = glass::calc_s(t, t_prime);
    let y = glass::early_stop_ddpm(t, t_prime, s, 0.1, 0.2);
    assert!(y.is_finite(), "non-finite estimate");

    // `early_stop_ddpm` is `alpha(t') * sufficient_stat(..)`, and the
    // sufficient statistic is a fixed linear combination of `x_t` and `x_s`.
    // So the map `(x_t, x_s) -> y` is linear with no constant term: it must be
    // homogeneous and additive. That pins the shape of the estimator, which
    // `is_finite` alone did not do at all.
    let f = |a: f32, b: f32| glass::early_stop_ddpm(t, t_prime, s, a, b);
    assert!(
        f(0.0, 0.0).abs() < 1e-6,
        "not homogeneous: f(0,0) = {}",
        f(0.0, 0.0)
    );
    assert!(
        (f(0.2, 0.4) - 2.0 * y).abs() < 1e-5,
        "not homogeneous: f(2x) = {} vs 2 f(x) = {}",
        f(0.2, 0.4),
        2.0 * y
    );
    assert!(
        (f(0.3, 0.5) - (f(0.1, 0.2) + f(0.2, 0.3))).abs() < 1e-5,
        "not additive"
    );
    // And it must actually depend on both observations.
    assert!((f(0.9, 0.2) - y).abs() > 1e-9, "ignores x_t");
    assert!((f(0.1, 0.9) - y).abs() > 1e-9, "ignores x_s");
}
