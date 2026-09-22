// RLX — versatile ML compiler + runtime. GPLv3.
//! Isolate `Op::TopK`'s tie-breaking across backends.
//!
//! The op's contract is "ties broken by smaller index". That is not a detail for
//! DeepSeek-V4.1: the Indexer rectifies its head scores, so any compressed
//! position every head dislikes scores *exactly* zero, and a selection of
//! `index_topk` out of a field full of zeros is decided entirely by the tie rule.
//! A backend that breaks ties differently picks a different set of positions to
//! attend to — a silent divergence, not a crash.
//!
//! `cargo run -p rlx-models-core --features metal,mlx --example dsv41_topk_probe`

use rlx_ir::graph::Graph;
use rlx_ir::op::Op;
use rlx_ir::{DType, Shape};
use rlx_models_core::device_capabilities::available_devices;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

/// The contract: the `k` largest, ties going to the smaller index. Returned as a
/// sorted set, because every consumer in this port uses the selection as a set.
fn reference(x: &[f32], rows: usize, n: usize, k: usize) -> Vec<Vec<usize>> {
    (0..rows)
        .map(|r| {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| {
                x[r * n + b]
                    .partial_cmp(&x[r * n + a])
                    .unwrap()
                    .then(a.cmp(&b))
            });
            let mut top: Vec<usize> = idx[..k].to_vec();
            top.sort_unstable();
            top
        })
        .collect()
}

fn run(dev: Device, x: &[f32], rows: usize, n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut g = Graph::new("topk_probe");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let xn = g.param("x", Shape::new(&[rows, n], DType::F32));
    params.insert("x".into(), x.to_vec());
    let idx = g.add_node(Op::TopK { k }, vec![xn], Shape::new(&[rows, k], DType::F32));
    g.set_outputs(vec![idx]);
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        dev,
    );
    let mut s = Session::new(dev).compile_with(g, &opts);
    for (nm, d) in &params {
        s.set_param(nm, d);
    }
    let out = s.run(&[]);
    (0..rows)
        .map(|r| {
            let mut v: Vec<usize> = out[0][r * k..(r + 1) * k]
                .iter()
                .map(|f| *f as usize)
                .collect();
            v.sort_unstable();
            v
        })
        .collect()
}

fn main() {
    // Rows shaped like real Indexer scores: a rectified field with many exact
    // zeros, a few positive values, and a masked-out tail.
    const NEG: f32 = -1e30;
    let (rows, n, k) = (6usize, 8usize, 3usize);
    #[rustfmt::skip]
    let x: Vec<f32> = vec![
        // all tied at zero — the set is decided purely by the tie rule
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        // all masked out: still has to return exactly k indices
        NEG, NEG, NEG, NEG, NEG, NEG, NEG, NEG,
        // two winners, the third slot contested by four zeros
        0.0, 3e-5, 0.0, 5e-5, 0.0, 0.0, NEG, NEG,
        // ties at a non-zero value
        1e-4, 2e-4, 2e-4, 2e-4, 1e-4, NEG, NEG, NEG,
        // strictly ordered — no ties, so every backend must agree anyway
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        // ties spanning the k boundary from below
        5.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    ];
    let want = reference(&x, rows, n, k);

    let mut bad = 0;
    for dev in available_devices() {
        let got = run(dev, &x, rows, n, k);
        for r in 0..rows {
            if got[r] != want[r] {
                println!("FAIL {dev:?} row {r}: got {:?} want {:?}", got[r], want[r]);
                bad += 1;
            }
        }
        if got == want {
            println!("ok   {dev:?}: all {rows} rows match the documented tie rule");
        }
    }
    if bad > 0 {
        eprintln!("\n{bad} row(s) violate `Op::TopK`'s \"ties broken by smaller index\" contract");
        std::process::exit(1);
    }
}
