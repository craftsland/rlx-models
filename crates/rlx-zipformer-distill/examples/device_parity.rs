// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Inference and training parity of the student graph on a device against
//! the CPU backend: the encoder output, the loss, and the gradient of every
//! parameter (cosine and relative error), then a few training steps on each
//! device side by side. Random weights and inputs, so no data is needed.
//!
//! ```text
//! cargo run -p rlx-zipformer-distill --release --no-default-features \
//!     --features rocm --example device_parity -- --device rocm [--chunks 2] [--batch 2]
//! ```
//! Devices: `cpu`, `metal`, `rocm`, `vulkan`, `xdna`, `gpu` (wgpu), each
//! needing its feature.

use anyhow::{Result, bail};
use rlx_tensor::{Device, Func, LrSchedule};
use rlx_zipformer_distill::config::{N_MELS, OUT_FRAMES, StudentConfig};
use rlx_zipformer_distill::model;

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut d, mut na, mut nb, mut e) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
        e += (x as f64 - y as f64).powi(2);
    }
    let cos = if na == 0.0 && nb == 0.0 {
        1.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    };
    (cos, (e / nb.max(1e-30)).sqrt())
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
    let k: usize = get("--chunks").map(|v| v.parse()).transpose()?.unwrap_or(2);
    let batch: usize = get("--batch").map(|v| v.parse()).transpose()?.unwrap_or(2);
    let steps: usize = get("--steps").map(|v| v.parse()).transpose()?.unwrap_or(3);
    let cfg = StudentConfig::small10m();

    let mut rng = Rng(0x1234_5678_9abc);
    let init = |name: &str, dims: &[usize], rng: &mut Rng| -> Vec<f32> {
        let n: usize = dims.iter().product();
        if name.ends_with("bypass_scale") {
            return vec![0.5; n];
        }
        if name.ends_with("log_scale") {
            return vec![0.0; n];
        }
        let fan_in = dims
            .get(1..)
            .map_or(1, |d| d.iter().product::<usize>())
            .max(1);
        let std = (1.0 / fan_in as f32).sqrt();
        (0..n).map(|_| rng.unit() * std).collect()
    };
    let mut m: Func = model::build(&cfg, batch, k);
    let mut r2 = Rng(7);
    m = m.init_params(|name, dims| init(name, dims, &mut r2));
    let feats: Vec<f32> = (0..batch * StudentConfig::segment_frames(k) * N_MELS)
        .map(|_| rng.unit() * 3.0)
        .collect();
    let target: Vec<f32> = (0..batch * k * OUT_FRAMES * 512)
        .map(|_| rng.unit())
        .collect();
    let feed: &[(&str, &[f32])] = &[("feats", &feats), ("target", &target)];

    // Inference: the encoder output.
    let mut enc = model::build_encoder(&cfg, batch, k);
    let mut r3 = Rng(7);
    enc = enc.init_params(|name, dims| init(name, dims, &mut r3));
    let t = std::time::Instant::now();
    let want = enc.run_on(Device::Cpu, &[("feats", &feats)]).remove(0);
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = std::time::Instant::now();
    let got = enc.run_on(device, &[("feats", &feats)]).remove(0);
    let dev_ms = t.elapsed().as_secs_f64() * 1e3;
    let (cos, rel) = compare(&got, &want);
    println!(
        "inference  encoder [{} values]: cosine {cos:.6}, rel err {rel:.2e} (cpu {cpu_ms:.0} ms, {device:?} {dev_ms:.0} ms incl. compile)",
        got.len()
    );

    // Training: loss + every gradient.
    let vg = m.value_and_grad_all();
    let names = m.param_names();
    let want = vg.run_on(Device::Cpu, feed);
    let got = vg.run_on(device, feed);
    let (lc, lr) = compare(&got[0], &want[0]);
    println!(
        "training   loss {:.6} vs cpu {:.6} (rel {lr:.2e}, cos {lc:.6})",
        got[0][0], want[0][0]
    );
    let mut worst = (1.0f64, String::new());
    let mut bad = 0;
    for (i, name) in names.iter().enumerate() {
        let (c, r) = compare(&got[i + 1], &want[i + 1]);
        if c < worst.0 {
            worst = (c, name.clone());
        }
        if c < 0.999 {
            bad += 1;
            println!("  grad {name}: cosine {c:.5}, rel err {r:.2e}");
        }
    }
    println!(
        "training   {} gradients: {} below cosine 0.999; worst {:.6} ({})",
        names.len(),
        bad,
        worst.0,
        worst.1
    );

    // A few optimizer steps on each device from the same start.
    let sched = LrSchedule::Constant(1e-3);
    let (mut mc, mut md) = (m.clone(), m);
    let (mut oc, mut od) = (rlx_optim::AdamW::new(1e-3), rlx_optim::AdamW::new(1e-3));
    for step in 0..steps {
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
