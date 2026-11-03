// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Write a trained student as a converted Kroko directory
//! (`model.safetensors`, `config.json`, `tokens.txt`) that `rlx-kroko`'s
//! `export_rkr` turns into an `.rkr`. The student's parameters already carry
//! the exporter's names; the decoder and joiner are the teacher's.

use std::path::Path;

use anyhow::{Context, Result};
use rlx_tensor::Func;
use safetensors::tensor::TensorView;

use crate::config::StudentConfig;
use crate::init::Teacher;

/// Save `model`'s parameters plus the teacher's `decoder.*` / `joiner.*`
/// tensors into `dir`, with `config.json` and the teacher's `tokens.txt`.
pub fn save_dir(
    model: &Func,
    cfg: &StudentConfig,
    teacher: &Teacher,
    teacher_dir: &Path,
    dir: &Path,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for name in model.param_names() {
        let data = model
            .param_binding(&name)
            .with_context(|| format!("{name} has no value"))?;
        let shape = model
            .param_shape_of(&name)
            .unwrap_or_else(|| vec![data.len()]);
        owned.push((
            name,
            shape,
            data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
    }
    for name in teacher.weights.keys() {
        if name.starts_with("decoder.") || name.starts_with("joiner.") {
            let (d, s) = teacher.weights.get(name).expect("listed key");
            owned.push((
                name.to_string(),
                s.to_vec(),
                d.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ));
        }
    }
    let views: Vec<(String, TensorView<'_>)> = owned
        .iter()
        .map(|(n, s, b)| {
            Ok((
                n.clone(),
                TensorView::new(safetensors::Dtype::F32, s.clone(), b)?,
            ))
        })
        .collect::<Result<_>>()?;
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors"))?;
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&cfg.to_json(teacher.vocab, teacher.context))?,
    )?;
    std::fs::copy(teacher_dir.join("tokens.txt"), dir.join("tokens.txt"))?;
    Ok(())
}
