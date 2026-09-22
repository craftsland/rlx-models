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

//! Run a check across every backend, and report all the failures at once.
//!
//! Most per-crate backend tests assert only that the output is *finite*. That
//! is the weakest useful property, and it is what let a rank-4 RoPE lowering
//! that zeroed almost every element, a router that read another token's
//! probability, and a host readback that walked a strided buffer linearly all
//! sit in a green suite. Every one of them produced finite, in-range,
//! plausible-looking numbers.
//!
//! The shape that finds those bugs is: pick an invariant or a host oracle, run
//! it on **every backend compiled in and present**, over **several variants of
//! the op** (head counts, sequence widths, rotation widths, pairing
//! conventions), and report every failing (backend, variant) pair together so
//! one run tells the whole story.
//!
//! ```ignore
//! // Sketch: `MY_CASES`, `run_on` and `host_oracle` stand in for the caller's
//! // own op and oracle. `ignore`, not `no_run`, because `no_run` still compiles
//! // the example and these placeholders have no definitions.
//! use rlx_models_core::backend_matrix::{Failures, available_devices};
//!
//! let mut fails = Failures::default();
//! for (name, device) in available_devices() {
//!     for case in MY_CASES {
//!         let got = run_on(device, case);
//!         let want = host_oracle(case);
//!         if max_abs_diff(&want, &got) >= 1e-4 {
//!             fails.push(name, format!("{case:?}: diverged"));
//!         }
//!     }
//! }
//! fails.assert_empty("my op");
//! ```

use rlx_runtime::Device;

/// Every backend worth trying, named.
///
/// Deliberately *not* gated on this crate's own `feature = "metal"` and
/// friends. Most model crates forward their backend features to
/// `rlx-runtime/<backend>` and not to this crate, so cfg-gating here reported
/// "cpu only" in exactly the crates that most needed the sweep — and a
/// candidate that is never listed cannot even be announced as a skip.
/// [`available_devices`] filters this with [`rlx_runtime::is_available`],
/// which is itself gated inside `rlx-runtime` and so answers correctly for
/// whatever was actually compiled in.
pub fn candidate_devices() -> Vec<(&'static str, Device)> {
    let mut out = vec![("cpu", Device::Cpu)];
    if cfg!(target_os = "macos") {
        out.push(("metal", Device::Metal));
        out.push(("mlx", Device::Mlx));
    }
    out.push(("wgpu", Device::Gpu));
    out.push(("vulkan", Device::Vulkan));
    out.push(("cuda", Device::Cuda));
    out.push(("rocm", Device::Rocm));
    out
}

/// Backends present on this machine.
///
/// Skips are announced rather than silent: a parity suite that quietly tested
/// only CPU is how the bugs above survived, and a run that covers one backend
/// should not look the same as one that covers five.
pub fn available_devices() -> Vec<(&'static str, Device)> {
    candidate_devices()
        .into_iter()
        .filter(|(name, d)| {
            let ok = rlx_runtime::is_available(*d);
            if !ok {
                eprintln!("[backend-matrix] skip {name}: backend not available");
            }
            ok
        })
        .collect()
}

/// Collected per-backend failures.
///
/// Deliberately accumulating rather than asserting eagerly: stopping at the
/// first bad backend hides whether a defect is one backend's or shared by
/// several. Three backends turned out to share one RoPE defect, which only
/// looked like one defect because they failed together.
#[derive(Default)]
pub struct Failures(Vec<String>);

impl Failures {
    pub fn push(&mut self, device: &str, detail: impl std::fmt::Display) {
        self.0.push(format!("  {device}: {detail}"));
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Panic with every failure listed, or return quietly.
    pub fn assert_empty(self, what: &str) {
        assert!(
            self.0.is_empty(),
            "{what} differs from the host oracle on {} backend/variant pair(s):\n{}",
            self.0.len(),
            self.0.join("\n")
        );
    }
}

/// Run `f` on CPU, then on every other available backend, and report all the
/// backends whose result differs.
///
/// The common shape for a model smoke test. Such tests usually assert only that
/// the output is finite, which is a bar that an all-zero result, a tensor with
/// one head's worth of real values and the rest zero, or a router that picked
/// the wrong expert all clear. Comparing against CPU costs one extra run and
/// catches all three.
///
/// `f` must rebuild whatever it needs — graphs are consumed by compilation, so
/// it is called once per backend.
pub fn assert_matches_cpu_on_all(what: &str, tol: f32, f: impl FnMut(Device) -> Vec<f32>) {
    assert_matches_cpu_except(what, tol, &[], f)
}

/// [`assert_matches_cpu_on_all`] with named backends excused by a known bug.
///
/// `known_bad` is `(backend, reason)`. Each is announced loudly rather than
/// skipped quietly — and if an excused backend *passes*, this fails. An
/// exclusion that silently outlives its bug is how a suite stops testing the
/// thing it was written for, so the list has to be wrong in only one direction.
pub fn assert_matches_cpu_except(
    what: &str,
    tol: f32,
    known_bad: &[(&str, &str)],
    mut f: impl FnMut(Device) -> Vec<f32>,
) {
    let cpu = f(Device::Cpu);
    assert!(
        !cpu.is_empty() && cpu.iter().all(|v| v.is_finite()),
        "{what}: the CPU reference is empty or non-finite, so nothing below means anything"
    );
    let mut fails = Failures::default();
    for (name, device) in available_devices() {
        if device == Device::Cpu {
            continue;
        }
        if let Some((_, reason)) = known_bad.iter().find(|(n, _)| *n == name) {
            // Run it anyway, so an exclusion that has outlived its bug shows up
            // — but absorb a panic, because "unsupported on this backend" is
            // raised as one (rlx-vulkan does exactly that for MLA's asymmetric
            // `v_head_dim`) and that is still the failure being excused.
            let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(device)));
            let ok = match &attempt {
                Ok(got) => {
                    got.len() == cpu.len()
                        && got.iter().all(|v| v.is_finite())
                        && max_rel_diff(&cpu, got) < tol
                }
                Err(_) => false,
            };
            if ok {
                fails.push(
                    name,
                    format!("excused as a known failure ({reason}) but now MATCHES CPU — drop the exclusion"),
                );
            } else {
                eprintln!("[backend-matrix] KNOWN FAILURE, excused: {name} — {reason}");
            }
            continue;
        }
        let got = f(device);
        if got.len() != cpu.len() {
            fails.push(
                name,
                format!("{} values, CPU gave {}", got.len(), cpu.len()),
            );
            continue;
        }
        if !got.iter().all(|v| v.is_finite()) {
            fails.push(name, "non-finite output");
            continue;
        }
        let rel = max_rel_diff(&cpu, &got);
        if !rel.is_finite() || rel >= tol {
            // Call out all-zero explicitly: it is the signature of a lowering
            // that wrote nothing, and reads very differently from drift.
            let zero = got.iter().all(|v| *v == 0.0);
            fails.push(
                name,
                format!(
                    "differs from CPU by {rel:.6} (tol {tol}){}",
                    if zero { " — output is ALL ZERO" } else { "" }
                ),
            );
        }
    }
    fails.assert_empty(what);
}

/// Largest absolute elementwise difference.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// [`max_abs_diff`] scaled by `max |want|`, so a tolerance can be stated
/// independently of the tensor's magnitude.
pub fn max_rel_diff(want: &[f32], got: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    max_abs_diff(want, got) / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_a_candidate() {
        assert!(candidate_devices().iter().any(|(n, _)| *n == "cpu"));
        assert!(available_devices().iter().any(|(n, _)| *n == "cpu"));
    }

    #[test]
    fn failures_report_every_pair() {
        let mut f = Failures::default();
        f.push("metal", "case A");
        f.push("wgpu", "case B");
        assert_eq!(f.len(), 2);
        let msg = std::panic::catch_unwind(move || f.assert_empty("op"))
            .unwrap_err()
            .downcast::<String>()
            .expect("string payload");
        assert!(msg.contains("metal: case A"), "{msg}");
        assert!(msg.contains("wgpu: case B"), "{msg}");
        assert!(msg.contains("2 backend/variant pair(s)"), "{msg}");
    }

    #[test]
    fn empty_failures_do_not_panic() {
        Failures::default().assert_empty("op");
    }

    #[test]
    fn rel_diff_is_scale_independent() {
        let want = [1.0f32, 2.0, 3.0];
        let got = [1.03f32, 2.0, 3.0];
        let big_want: Vec<f32> = want.iter().map(|v| v * 1000.0).collect();
        let big_got: Vec<f32> = got.iter().map(|v| v * 1000.0).collect();
        let a = max_rel_diff(&want, &got);
        let b = max_rel_diff(&big_want, &big_got);
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}
