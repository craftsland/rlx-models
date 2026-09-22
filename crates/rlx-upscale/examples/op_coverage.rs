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

//! Which RLX ops each architecture actually needs.
//!
//! "Does this run on CUDA?" is not a question you want to answer by finding a
//! CUDA box. Every backend declares a `SUPPORTED_OPS` set and the compiler's
//! `LegalizeForBackend` pass fails the compile when a graph needs something
//! outside it — so the answer is decidable from the *op set* alone, on any
//! machine.
//!
//! This builds every architecture at a representative configuration and prints
//! the distinct `OpKind`s it emits. Cross-check the result against
//! `docs/op-coverage.md` in the RLX tree (`scripts/upscale_backend_matrix.py`
//! does exactly that).
//!
//! Run:
//!   cargo run -p rlx-upscale --example op_coverage
//!   cargo run -p rlx-upscale --example op_coverage -- --json

use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use rlx_upscale::graph;
use std::collections::{BTreeMap, BTreeSet};

fn main() -> Result<()> {
    let json = std::env::args().any(|a| a == "--json");
    let mut per_arch: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut all: BTreeSet<String> = BTreeSet::new();

    for (name, cfg, tile) in rlx_upscale::sample::representative_configs() {
        let built = graph::build_autofill(
            &cfg,
            WeightMap::from_tensors(Default::default()),
            tile,
            tile,
            7,
        )
        .map_err(|e| anyhow::anyhow!("building {name}: {e:#}"))?;

        let ops: BTreeSet<String> = built
            .graph
            .nodes()
            .iter()
            .map(|n| format!("{:?}", n.op.kind()))
            .collect();
        all.extend(ops.iter().cloned());
        per_arch.insert(name.to_string(), ops);
    }

    if json {
        let obj: BTreeMap<&str, serde_json::Value> = [
            (
                "per_arch",
                serde_json::to_value(&per_arch).expect("op names serialize"),
            ),
            (
                "union",
                serde_json::to_value(&all).expect("op names serialize"),
            ),
        ]
        .into_iter()
        .collect();
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    for (name, ops) in &per_arch {
        println!("{name:<12} {:>2} ops  {}", ops.len(), join(ops));
    }
    println!("\nunion over all architectures: {} ops", all.len());
    println!("{}", join(&all));
    Ok(())
}

fn join(ops: &BTreeSet<String>) -> String {
    ops.iter().cloned().collect::<Vec<_>>().join(" ")
}
