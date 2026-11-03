// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Inference and training parity of a small GPT on a device against the CPU
//! backend: logits, the loss, every parameter gradient (cosine), then a few
//! AdamW steps on each device side by side. Covers causal attention,
//! LayerNorm, GELU, gather embedding and softmax cross-entropy.
//!
//! ```text
//! cargo run -p rlx-tinystories --release --no-default-features --features rocm \
//!     --example device_parity -- --device rocm
//! ```

use anyhow::{Result, bail};
use rlx_tensor::{AdamW, DType, Device, LrSchedule};
use rlx_tinystories::config::GptConfig;
use rlx_tinystories::model;

fn compare(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na == 0.0 && nb == 0.0 {
        1.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let device = match get("--device").map(String::as_str).unwrap_or("cpu") {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "rocm" => Device::Rocm,
        "vulkan" => Device::Vulkan,
        "xdna" => Device::Xdna,
        "gpu" => Device::Gpu,
        d => bail!("unknown device {d}"),
    };
    let cfg = GptConfig {
        vocab: 256,
        block_size: 32,
        n_layer: 2,
        n_head: 4,
        n_embd: 64,
        batch: 2,
        label_smoothing: 0.0,
    };
    let bt = cfg.batch * cfg.block_size;
    let tok: Vec<f32> = (0..bt).map(|i| ((i * 37 + 11) % 256) as f32).collect();
    let tgt: Vec<f32> = (0..bt).map(|i| ((i * 53 + 7) % 256) as f32).collect();

    let logits = model::init(model::build(&cfg, cfg.batch, false, DType::F32), &cfg, 1);
    let want = logits.run_on(Device::Cpu, &[("tok_ids", &tok)]).remove(0);
    let got = logits.run_on(device, &[("tok_ids", &tok)]).remove(0);
    println!(
        "inference  logits [{}]: cosine {:.6}",
        got.len(),
        compare(&got, &want)
    );

    let m = model::init(model::build(&cfg, cfg.batch, true, DType::F32), &cfg, 1);
    let feed: &[(&str, &[f32])] = &[("tok_ids", &tok), ("tgt_ids", &tgt)];
    let vg = m.value_and_grad_all();
    let names = m.param_names();
    let want = vg.run_on(Device::Cpu, feed);
    let got = vg.run_on(device, feed);
    println!("training   loss {:.6} vs cpu {:.6}", got[0][0], want[0][0]);
    let mut worst = (1.0f64, String::new());
    for (i, name) in names.iter().enumerate() {
        let c = compare(&got[i + 1], &want[i + 1]);
        if c < worst.0 {
            worst = (c, name.clone());
        }
        if c < 0.999 {
            println!("  grad {name}: cosine {c:.5}");
        }
    }
    println!(
        "training   {} gradients; worst cosine {:.6} ({})",
        names.len(),
        worst.0,
        worst.1
    );
    let sched = LrSchedule::Constant(3e-3);
    let (mut mc, mut md) = (m.clone(), m);
    let (mut oc, mut od) = (AdamW::new(3e-3), AdamW::new(3e-3));
    for step in 0..3 {
        let (n1, l1) = mc.train_step_all_at_on(Device::Cpu, &mut oc, &sched, step, feed);
        let (n2, l2) = md.train_step_all_at_on(device, &mut od, &sched, step, feed);
        mc = n1;
        md = n2;
        println!(
            "train step {step}: loss cpu {:.6}  {device:?} {:.6}",
            l1[0], l2[0]
        );
    }
    std::process::exit(0)
}
