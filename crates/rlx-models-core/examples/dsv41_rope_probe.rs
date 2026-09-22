// RLX — versatile ML compiler + runtime. GPLv3.
//! Isolate the partial (tail) RoPE across backends.
//!
//! DeepSeek-V4/V4.1 rotate only the **last** `rope_head_dim` dims of each head,
//! which `rope_tail` builds as narrow → reshape → rope → concat. The reshape
//! sits on a narrowed, non-contiguous view, which is exactly the shape of thing
//! that behaves differently per backend.
//!
//! This compares three forms on every available device:
//!
//! 1. `Op::Rope` on its own, over a contiguous `[rows, nh·rd]` input,
//! 2. the full `rope_tail` plumbing,
//! 3. a host reference computed in plain Rust.
//!
//! `cargo run -p rlx-models-core --features metal --example dsv41_rope_probe`

use rlx_ir::GraphExt;
use rlx_ir::graph::Graph;
use rlx_ir::op::RopeStyle;
use rlx_ir::{DType, Shape};
use rlx_models_core::device_capabilities::available_devices;
use rlx_models_core::parity::Deviation;
use rlx_models_core::weight_loader::SyntheticLoader;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

/// NeoX half-split rotation — the other pairing, and the usual thing a backend
/// does when it ignores the requested style.
fn reference_neox(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    nh: usize,
    hd: usize,
    rd: usize,
) -> Vec<f32> {
    let half = rd / 2;
    let mut out = x.to_vec();
    for r in 0..rows {
        for h in 0..nh {
            let base = r * nh * hd + h * hd + (hd - rd);
            for i in 0..half {
                let (a, b) = (x[base + i], x[base + half + i]);
                let (c, s) = (cos[r * half + i], sin[r * half + i]);
                out[base + i] = a * c - b * s;
                out[base + half + i] = b * c + a * s;
            }
        }
    }
    out
}

/// GPT-J interleaved rotation of the last `rd` dims of each head, in plain Rust.
fn reference(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    nh: usize,
    hd: usize,
    rd: usize,
) -> Vec<f32> {
    let half = rd / 2;
    let mut out = x.to_vec();
    for r in 0..rows {
        for h in 0..nh {
            let base = r * nh * hd + h * hd + (hd - rd);
            for i in 0..half {
                let (a, b) = (x[base + 2 * i], x[base + 2 * i + 1]);
                let (c, s) = (cos[r * half + i], sin[r * half + i]);
                out[base + 2 * i] = a * c - b * s;
                out[base + 2 * i + 1] = b * c + a * s;
            }
        }
    }
    out
}

fn run(dev: Device, build: impl FnOnce(&mut Graph, &mut HashMap<String, Vec<f32>>)) -> Vec<f32> {
    let mut g = Graph::new("rope_probe");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    build(&mut g, &mut params);
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        dev,
    );
    let mut s = Session::new(dev).compile_with(g, &opts);
    for (n, d) in &params {
        s.set_param(n, d);
    }
    s.run(&[])[0].clone()
}

fn report(label: &str, got: &[f32], want: &[f32]) -> bool {
    let d = Deviation::between(got, want);
    let ok = d.is_within(1e-5);
    println!("{} {label:<34} {d}", if ok { "ok  " } else { "FAIL" });
    ok
}

fn main() {
    // the released V4.1 attention geometry, scaled down in `rows` only
    let (rows, nh, hd, rd) = (6usize, 4usize, 32usize, 16usize);
    let half = rd / 2;
    let x = SyntheticLoader::values("probe.51ed", &[rows * nh * hd]);
    let cos = SyntheticLoader::values("probe.c05", &[rows * half]);
    let sin = SyntheticLoader::values("probe.5111", &[rows * half]);
    let want_tail = reference(&x, &cos, &sin, rows, nh, hd, rd);

    let mut bad = 0;
    for dev in available_devices() {
        println!("── {dev:?} ──");

        // 1. the rope op alone, contiguous input
        let mut xt = vec![0f32; rows * nh * rd];
        for r in 0..rows {
            for h in 0..nh {
                for i in 0..rd {
                    xt[(r * nh + h) * rd + i] = x[r * nh * hd + h * hd + (hd - rd) + i];
                }
            }
        }
        let want_bare = reference(&xt, &cos, &sin, rows, nh, rd, rd);
        let got = run(dev, |g, p| {
            let xn = g.param("x", Shape::new(&[rows, nh * rd], DType::F32));
            p.insert("x".into(), xt.clone());
            let cn = g.param("cos", Shape::new(&[rows, half], DType::F32));
            p.insert("cos".into(), cos.clone());
            let sn = g.param("sin", Shape::new(&[rows, half], DType::F32));
            p.insert("sin".into(), sin.clone());
            let y = g.rope_n_styled(xn, cn, sn, rd, rd, RopeStyle::GptJ);
            g.set_outputs(vec![y]);
        });
        if !report("Op::Rope alone (contiguous)", &got, &want_bare) {
            bad += 1;
            // if the backend silently applied the OTHER pairing, say so — that is
            // a one-line upstream fix rather than a numerical mystery
            let neox = reference_neox(&xt, &cos, &sin, rows, nh, rd, rd);
            let d = got
                .iter()
                .zip(&neox)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            if d < 1e-5 {
                println!("     ^ this is exactly RopeStyle::NeoX — the style argument is ignored");
            }
        }

        // 1b. the same thing, but rank-3 [1, rows, nh*rd]
        let got = run(dev, |g, p| {
            let xn = g.param("x", Shape::new(&[1, rows, nh * rd], DType::F32));
            p.insert("x".into(), xt.clone());
            let cn = g.param("cos", Shape::new(&[rows, half], DType::F32));
            p.insert("cos".into(), cos.clone());
            let sn = g.param("sin", Shape::new(&[rows, half], DType::F32));
            p.insert("sin".into(), sin.clone());
            let y = g.rope_n_styled(xn, cn, sn, rd, rd, RopeStyle::GptJ);
            g.set_outputs(vec![y]);
        });
        if !report("Op::Rope, rank-3 [1, rows, nh*rd]", &got, &want_bare) {
            bad += 1;
        }

        // 2. the full tail plumbing: narrow -> reshape -> rope -> concat
        let got = run(dev, |g, p| {
            let xn = g.param("x", Shape::new(&[rows, nh * hd], DType::F32));
            p.insert("x".into(), x.clone());
            let cn = g.param("cos", Shape::new(&[rows, half], DType::F32));
            p.insert("cos".into(), cos.clone());
            let sn = g.param("sin", Shape::new(&[rows, half], DType::F32));
            p.insert("sin".into(), sin.clone());
            let x3 = g.reshape_(xn, vec![rows as i64, nh as i64, hd as i64]);
            let nope = g.narrow_(x3, 2, 0, hd - rd);
            let tail = g.narrow_(x3, 2, hd - rd, rd);
            let tail_flat = g.reshape_(tail, vec![rows as i64, (nh * rd) as i64]);
            let roped = g.rope_n_styled(tail_flat, cn, sn, rd, rd, RopeStyle::GptJ);
            let roped3 = g.reshape_(roped, vec![rows as i64, nh as i64, rd as i64]);
            let cat = g.concat_(vec![nope, roped3], 2);
            let y = g.reshape_(cat, vec![rows as i64, (nh * hd) as i64]);
            g.set_outputs(vec![y]);
        });
        if !report("rope_tail plumbing", &got, &want_tail) {
            bad += 1;
        }

        // 3. the narrow -> reshape view on its own, no rope at all
        let mut want_view = vec![0f32; rows * nh * rd];
        for r in 0..rows {
            for h in 0..nh {
                for i in 0..rd {
                    want_view[(r * nh + h) * rd + i] = x[r * nh * hd + h * hd + (hd - rd) + i];
                }
            }
        }
        let got = run(dev, |g, p| {
            let xn = g.param("x", Shape::new(&[rows, nh * hd], DType::F32));
            p.insert("x".into(), x.clone());
            let x3 = g.reshape_(xn, vec![rows as i64, nh as i64, hd as i64]);
            let tail = g.narrow_(x3, 2, hd - rd, rd);
            let y = g.reshape_(tail, vec![rows as i64, (nh * rd) as i64]);
            g.set_outputs(vec![y]);
        });
        if !report("narrow(axis 2) -> reshape", &got, &want_view) {
            bad += 1;
        }
    }
    if bad > 0 {
        eprintln!("\n{bad} form(s) disagree with the host reference");
        std::process::exit(1);
    }
}
