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

//! Work out what a folder of unlabelled checkpoints actually contains.
//!
//! A downloaded upscaler is a bare state dict: no config, no architecture tag,
//! and a filename that may say `4xFoo_v3_final.pth` and nothing more. This
//! walks a directory and, for each file, reports what it is and what it will
//! cost — without loading a single weight into a graph.
//!
//! Everything printed is *solved*, not guessed. Window size comes from the
//! relative-position bias table having exactly `(2W−1)²` rows; HAT's overlap
//! ratio from its second table; DySample's group count from the width of its
//! offset convolution; MambaIRv2's routing dictionary from the shapes of the
//! two embedding factors. Where a reference hyperparameter genuinely leaves no
//! trace in the weights it is assumed, and `crate::detect`'s module docs say
//! which three those are.
//!
//! Run:
//!   cargo run --release -p rlx-upscale --example identify -- ~/models
//!   cargo run --release -p rlx-upscale --example identify -- model.pth --json

use anyhow::{Context, Result};
use rlx_upscale::Checkpoint;
use rlx_upscale::detect::detect;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().any(|a| a == "--json");
    let roots: Vec<PathBuf> = args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .collect();
    anyhow::ensure!(
        !roots.is_empty(),
        "usage: identify <dir-or-file> [more...] [--json]"
    );

    let mut files = Vec::new();
    for root in &roots {
        if root.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(root)
                .with_context(|| format!("reading {}", root.display()))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    matches!(
                        p.extension().and_then(|e| e.to_str()),
                        Some("pth" | "pt" | "bin" | "ckpt" | "safetensors")
                    )
                })
                .collect();
            found.sort();
            files.extend(found);
        } else {
            files.push(root.clone());
        }
    }

    if !json {
        println!(
            "{:<38} {:<12} {:>5}  {:>5} {:>8} {:>7}",
            "file", "architecture", "scale", "tile", "peak est", "tensors"
        );
    }
    let (mut ok, mut failed) = (0usize, 0usize);
    for path in &files {
        match describe(path, json) {
            Ok(()) => ok += 1,
            Err(e) => {
                failed += 1;
                // An unreadable or unrecognized file is worth naming, not worth
                // aborting the walk for — the point is to triage a folder.
                if json {
                    println!(
                        "{}",
                        serde_json::json!({ "file": name(path), "error": format!("{e:#}") })
                    );
                } else {
                    println!("{:<38} {}", name(path), format!("{e:#}").replace('\n', " "));
                }
            }
        }
    }
    if !json {
        println!("\n{ok} identified, {failed} unrecognized");
    }
    Ok(())
}

fn name(p: &Path) -> String {
    p.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

fn describe(path: &Path, json: bool) -> Result<()> {
    let ck = Checkpoint::open(path)?;
    let cfg = detect(&ck)?;
    let tile = rlx_upscale::runner::budget_tile(&cfg);
    let peak = cfg.peak_working_set_bytes(tile);

    if json {
        println!(
            "{}",
            serde_json::json!({
                "file": name(path),
                "tensors": ck.len(),
                "tile": tile,
                "peak_estimate_mb": peak >> 20,
                "size_multiple": cfg.size_multiple(),
                "config": cfg,
            })
        );
    } else {
        println!(
            "{:<38} {:<12} {:>5}  {:>5} {:>6} MB {:>7}",
            truncate(&name(path), 38),
            cfg.arch.name(),
            format!("×{}", cfg.scale),
            tile,
            peak >> 20,
            ck.len(),
        );
        println!("{:38} {}", "", cfg.summary());
    }
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    // Keep the tail: the informative part of these names (`_x4`, `_dysample`)
    // is at the end, while the prefix is usually a dataset tag.
    let keep: String = s.chars().skip(s.chars().count() - (n - 1)).collect();
    format!("…{keep}")
}
