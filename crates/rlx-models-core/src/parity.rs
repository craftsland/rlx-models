// RLX — versatile ML compiler + runtime. GPLv3.
//! Comparing two tensors, and saying something useful when they differ.
//!
//! Parity work in this repo is mostly "does the port match the reference, and if
//! not, where?" — so a bare `assert!(rel < tol)` throws away exactly the
//! information needed next: which element, how far off, and whether the whole
//! tensor moved or one lane did.
//!
//! [`Deviation`] normalizes by the reference's own scale, so a near-zero entry
//! cannot dominate the ratio, and it reports the offending index and both values.

use std::fmt;

/// How far two tensors are apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deviation {
    /// Elements compared.
    pub len: usize,
    /// Largest absolute difference.
    pub max_abs: f32,
    /// Index of that difference.
    pub at: usize,
    /// `got[at]` and `want[at]`.
    pub got: f32,
    pub want: f32,
    /// `max |want|`, floored so an all-zero reference cannot divide by zero.
    pub scale: f32,
    /// `max_abs / scale`.
    pub rel: f32,
    /// First index holding a non-finite value in `got`, if any. Always a
    /// failure, however small the tolerance: a NaN that never gets compared is
    /// how a broken kernel passes a parity test.
    pub non_finite: Option<usize>,
    /// How many elements differ at all — one lane wrong and the whole tensor
    /// wrong are very different bugs.
    pub differing: usize,
}

impl Deviation {
    /// Compare two equal-length slices.
    ///
    /// # Panics
    /// If the lengths differ — that is a shape bug, not a numeric one, and
    /// silently comparing the overlap would hide it.
    pub fn between(got: &[f32], want: &[f32]) -> Self {
        assert_eq!(
            got.len(),
            want.len(),
            "Deviation::between: {} elements vs {}",
            got.len(),
            want.len()
        );
        let scale = want.iter().fold(0f32, |a, b| a.max(b.abs())).max(1e-9);
        let (mut max_abs, mut at, mut differing) = (0f32, 0usize, 0usize);
        let mut non_finite = None;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            if non_finite.is_none() && !g.is_finite() {
                non_finite = Some(i);
            }
            if g != w {
                differing += 1;
            }
            let d = (g - w).abs();
            if d > max_abs {
                max_abs = d;
                at = i;
            }
        }
        Deviation {
            len: got.len(),
            max_abs,
            at,
            got: got.get(at).copied().unwrap_or(f32::NAN),
            want: want.get(at).copied().unwrap_or(f32::NAN),
            scale,
            rel: max_abs / scale,
            non_finite,
            differing,
        }
    }

    /// Within `tol` *relative to the reference's scale*, and finite throughout.
    pub fn is_within(&self, tol: f32) -> bool {
        self.non_finite.is_none() && self.rel < tol
    }
}

impl fmt::Display for Deviation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(i) = self.non_finite {
            return write!(f, "non-finite at index {i}");
        }
        write!(
            f,
            "rel {:.3e} (max |Δ| {:.3e} at {} of {}: got {:+.6}, want {:+.6}; scale {:.3e}; \
             {} of {} elements differ)",
            self.rel,
            self.max_abs,
            self.at,
            self.len,
            self.got,
            self.want,
            self.scale,
            self.differing,
            self.len
        )
    }
}

/// Assert two tensors agree to `tol`, reporting where they do not.
///
/// # Panics
/// With the full [`Deviation`] when they do not — including a non-finite value,
/// whatever the tolerance.
#[track_caller]
pub fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    let d = Deviation::between(got, want);
    assert!(d.is_within(tol), "{label}: {d} (tolerance {tol:e})");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_is_the_references_own_magnitude() {
        // an absolute difference of 0.1 is nothing against a tensor reaching 100
        let d = Deviation::between(&[100.1, 0.0], &[100.0, 0.0]);
        assert!(d.rel < 2e-3, "{d}");
        assert!(d.is_within(2e-3));
        // ...and everything against one reaching 0.001
        let d = Deviation::between(&[0.101, 0.0], &[0.001, 0.0]);
        assert!(d.rel > 1.0, "{d}");
        assert!(!d.is_within(0.5));
    }

    #[test]
    fn a_near_zero_entry_does_not_dominate() {
        // 1e-9 vs 0 is a relative error of infinity if normalized per element,
        // but nothing at all against a tensor whose scale is 1
        let d = Deviation::between(&[1.0, 1e-9], &[1.0, 0.0]);
        assert!(d.is_within(1e-6), "{d}");
    }

    #[test]
    fn non_finite_fails_at_any_tolerance() {
        let d = Deviation::between(&[f32::NAN, 1.0], &[0.0, 1.0]);
        assert_eq!(d.non_finite, Some(0));
        assert!(!d.is_within(f32::MAX));
        assert!(format!("{d}").contains("non-finite at index 0"));
    }

    #[test]
    fn differing_count_separates_one_lane_from_the_whole_tensor() {
        let one = Deviation::between(&[1.0, 2.0, 3.0, 9.0], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(one.differing, 1);
        // every element off by the same 5, so the max is identical
        let all = Deviation::between(&[6.0, 7.0, 8.0, 9.0], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(all.differing, 4);
        // same max |Δ|, very different diagnosis
        assert_eq!(one.max_abs, all.max_abs);
    }

    #[test]
    fn identical_tensors_are_exactly_zero() {
        let d = Deviation::between(&[1.0, -2.0, 3.0], &[1.0, -2.0, 3.0]);
        assert_eq!((d.max_abs, d.rel, d.differing), (0.0, 0.0, 0));
        assert!(d.is_within(0.0) || d.rel == 0.0);
    }

    #[test]
    #[should_panic(expected = "3 elements vs 2")]
    fn length_mismatch_is_a_panic_not_a_partial_compare() {
        Deviation::between(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
    }
}
