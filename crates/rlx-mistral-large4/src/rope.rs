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

//! LeRoPE (YaRN with learned per-frequency scales), interleaved rotation, and
//! the SSMax position term — the three position-dependent pieces of the model.
//!
//! ## Frequencies
//!
//! HF: `inv_freq = yarn_inv_freq · exp(alpha)`, `alpha` a learned buffer.
//! GGUF stores `rope_freqs.weight = exp(-alpha)` because ggml *divides* theta by
//! its frequency factors, so here:
//!
//! ```text
//!   inv_freq[i] = yarn_inv_freq[i] / rope_freqs[i]
//! ```
//!
//! The YaRN part is config data; the learned part is a weight. The table is
//! therefore built **in the graph** from the position input, which lets
//! prefill (constant positions) and decode (a runtime position) share it.
//!
//! ## Interleaved pairs
//!
//! The rotary dims pair up as `(x0, x1), (x2, x3), …` (ggml `ROPE_TYPE_NORM`).
//! HF writes the result de-interleaved — `cat(evens', odds')` — which is a fixed
//! permutation of the rotated dims. Queries and keys get the same permutation,
//! so every dot product is unchanged; this module writes the HF layout.
//!
//! ## SSMax
//!
//! Every query (attention and indexer) is multiplied per head by
//! `s_h · log(pos + 1) + b_h` — a learned, length-dependent softmax temperature.

use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::Activation;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::config::MistralLarge4Config;

/// GGUF name of the learned frequency divisors (`exp(-alpha)`).
pub const ROPE_FREQS_KEY: &str = "rope_freqs.weight";

/// The un-learned half of the frequencies: YaRN (or plain) `inv_freq` for the
/// `qk_rope_head_dim / 2` rotary pairs.
pub fn base_inv_freq(cfg: &MistralLarge4Config) -> Vec<f64> {
    let d = cfg.qk_rope_head_dim;
    (0..d / 2)
        .map(|i| cfg.rope_scaling.inv_freq(i, d, cfg.rope_theta))
        .collect()
}

/// Position-dependent tensors shared by every layer of one graph.
#[derive(Debug, Clone, Copy)]
pub struct PosTables {
    /// `[rows, half]`.
    pub cos: HirNodeId,
    /// `[rows, half]`.
    pub sin: HirNodeId,
    /// `[rows, 1]` — `log(pos + 1)`, the SSMax basis.
    pub log_pos: HirNodeId,
    pub rows: usize,
    pub half: usize,
}

/// Build [`PosTables`] from `pos` (`[rows, 1]`, f32 positions).
///
/// Loads [`ROPE_FREQS_KEY`]; call once per graph.
pub fn emit_pos_tables(
    emit: &mut Emit<'_>,
    cfg: &MistralLarge4Config,
    pos: HirNodeId,
    rows: usize,
) -> anyhow::Result<PosTables> {
    let f = DType::F32;
    let half = cfg.qk_rope_head_dim / 2;
    let base: Vec<f32> = base_inv_freq(cfg).iter().map(|&x| x as f32).collect();
    let base = emit.synth_param("ml4.rope.base_inv_freq", base, Shape::new(&[1, half], f));
    let learned = emit.load_param(ROPE_FREQS_KEY, false)?; // [half]
    let mscale = cfg.rope_scaling.mscale() as f32;
    let mscale = (mscale != 1.0)
        .then(|| emit.synth_param("ml4.rope.mscale", vec![mscale], Shape::new(&[1], f)));
    let one = emit.synth_param("ml4.rope.one", vec![1.0], Shape::new(&[1], f));

    let mut gb = HirMut::new(emit.hir());
    let learned = gb.reshape_(learned, vec![1, half as i64]);
    let inv = gb.div(base, learned); // [1, half]
    let ang = gb.mul(pos, inv); // [rows, half]
    let tshape = Shape::new(&[rows, half], f);
    let mut cos = gb.activation(Activation::Cos, ang, tshape.clone());
    let mut sin = gb.activation(Activation::Sin, ang, tshape);
    if let Some(m) = mscale {
        cos = gb.mul(cos, m);
        sin = gb.mul(sin, m);
    }
    let p1 = gb.add(pos, one);
    let log_pos = gb.activation(Activation::Log, p1, Shape::new(&[rows, 1], f));
    Ok(PosTables {
        cos,
        sin,
        log_pos,
        rows,
        half,
    })
}

/// Rotate `x` (`[rows, heads, 2·half]`) by interleaved pairs, writing the HF
/// de-interleaved layout.
pub fn rotate_interleaved(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    t: &PosTables,
    heads: usize,
) -> HirNodeId {
    let (r, h, half) = (t.rows as i64, heads as i64, t.half as i64);
    let pairs = gb.reshape_(x, vec![r, h, half, 2]);
    let even = gb.narrow_(pairs, 3, 0, 1);
    let even = gb.reshape_(even, vec![r, h, half]);
    let odd = gb.narrow_(pairs, 3, 1, 1);
    let odd = gb.reshape_(odd, vec![r, h, half]);
    let c = gb.reshape_(t.cos, vec![r, 1, half]);
    let s = gb.reshape_(t.sin, vec![r, 1, half]);
    let ec = gb.mul(even, c);
    let os = gb.mul(odd, s);
    let first = gb.sub(ec, os);
    let oc = gb.mul(odd, c);
    let es = gb.mul(even, s);
    let second = gb.add(oc, es);
    gb.concat_(vec![first, second], 2)
}

/// Multiply `q` (`[rows, heads, d]`) per head by `s_h · log(pos+1) + b_h`.
pub fn apply_ssmax(
    gb: &mut HirMut<'_>,
    q: HirNodeId,
    scale: HirNodeId, // [heads]
    bias: HirNodeId,  // [heads]
    t: &PosTables,
    heads: usize,
) -> HirNodeId {
    let h = heads as i64;
    let s = gb.reshape_(scale, vec![1, h]);
    let b = gb.reshape_(bias, vec![1, h]);
    let m = gb.mul(t.log_pos, s); // [rows, heads]
    let m = gb.add(m, b);
    let m = gb.reshape_(m, vec![t.rows as i64, h, 1]);
    gb.mul(q, m)
}
