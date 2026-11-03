// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Inference throughput of the student encoder on one device or several at
//! once (data parallel: each device encodes its share of every batch). By
//! default every local backend that can run it is a candidate; each is timed
//! alone, then the speed split, and the fastest layout wins. `--devices`
//! restricts the candidates (`name`) or fixes the split (`name:share`). Random weights and
//! inputs, so no data is needed; an 8-chunk segment is 2.56 s of audio.
//!
//! ```text
//! cargo run -p rlx-zipformer-distill --release --no-default-features \
//!     --features rocm --example throughput -- --batch 32
//! ```

use anyhow::Result;
use rlx_tensor::parallel::Feeds;
use rlx_tensor::{
    DataParallel, Device, Func, Replica, available_devices, device_label, parse_device,
    training_devices, tune,
};
use rlx_zipformer_distill::config::{N_MELS, StudentConfig};
use rlx_zipformer_distill::model;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let num = |k: &str, d: usize| -> Result<usize> {
        Ok(get(k).map(|v| v.parse()).transpose()?.unwrap_or(d))
    };
    let batch = num("--batch", 16)?;
    let k = num("--chunks", 8)?;
    let iters = num("--iters", 5)?;
    let cfg = StudentConfig::small10m();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
    };
    let master = model::build_encoder(&cfg, 1, k).init_params(|_, dims| {
        (0..dims.iter().product::<usize>())
            .map(|_| rand())
            .collect()
    });
    let frames = StudentConfig::segment_frames(k) * N_MELS;
    // Candidates: `--devices` (names, or `name:share` to fix the split), else
    // every local backend that can run the graph, one per chip.
    let spec: Vec<(Device, Option<usize>)> = match get("--devices") {
        Some(list) if list != "auto" => list
            .split(',')
            .map(|item| {
                let (name, share) = item.split_once(':').unwrap_or((item, ""));
                let dev = parse_device(name).map_err(|e| anyhow::anyhow!("{e}"))?;
                let share = (!share.is_empty()).then(|| share.parse()).transpose()?;
                Ok((dev, share))
            })
            .collect::<Result<_>>()?,
        _ => training_devices(&available_devices(), master.graph())
            .into_iter()
            .map(|d| (d, None))
            .collect(),
    };
    let replica = |dev: Device, b: usize| {
        let f = model::build_encoder(&cfg, b, k)
            .init_params(|name, _| master.param_binding(name).expect("param").to_vec());
        Replica::new(dev, f, b)
    };
    let feeds = |shares: &[usize]| -> Vec<Feeds> {
        shares
            .iter()
            .map(|&b| vec![("feats".to_string(), vec![0.1f32; b * frames])])
            .collect()
    };
    // Mean seconds per batch over `iters` runs after one warm-up (compile).
    let time = |dp: &DataParallel, shares: &[usize], m: &Func| -> f64 {
        dp.run(m, feeds(shares));
        let t = std::time::Instant::now();
        for _ in 0..iters {
            dp.run(m, feeds(shares));
        }
        t.elapsed().as_secs_f64() / iters as f64
    };

    let (dp, devs, shares) = if spec.iter().all(|s| s.1.is_some()) {
        let (devs, shares): (Vec<Device>, Vec<usize>) =
            spec.iter().map(|s| (s.0, s.1.unwrap_or(0))).unzip();
        let dp = DataParallel::new(
            devs.iter()
                .zip(&shares)
                .map(|(&d, &b)| replica(d, b))
                .collect(),
        );
        (dp, devs, shares)
    } else {
        let devs: Vec<Device> = spec.iter().map(|s| s.0).collect();
        let tuned = tune(
            &devs,
            batch,
            |d, b| replica(d, b).func,
            |dp, shares| time(dp, shares, &master),
        );
        for (layout, t) in &tuned.tried {
            let l: Vec<String> = layout
                .iter()
                .map(|(d, b)| format!("{}:{b}", device_label(*d)))
                .collect();
            eprintln!("  tried {}: {:.1} ms", l.join(","), t * 1e3);
        }
        let (devs, shares) = tuned.layout.iter().copied().unzip();
        (tuned.dp, devs, shares)
    };
    let total: usize = shares.iter().sum();
    let t = time(&dp, &shares, &master);
    let layout: Vec<String> = devs
        .iter()
        .zip(&shares)
        .map(|(d, b)| format!("{}:{b}", device_label(*d)))
        .collect();
    let audio = (total * k) as f64 * 0.32;
    println!(
        "{}: {:.1} ms per batch of {total}, {:.1} segments/s, {:.0}x real time",
        layout.join(","),
        t * 1e3,
        total as f64 / t,
        audio / t
    );
    std::process::exit(0)
}
