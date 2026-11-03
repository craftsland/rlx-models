// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Distillation records: one `K`-chunk segment each, `fbank
//! [segment_frames(K), 80]` then the teacher's encoder output `[K·8, 512]`,
//! f32 little-endian, back to back in `segments.f32`; `meta.json` holds `K`
//! and the count. Read through `mmap`, so a multi-GB set never sits in RAM.
//!
//! An optional `contexts.u16` beside it holds, per record and output frame,
//! the teacher's predictor context (two token ids, [`NO_TOKEN`] for none)
//! from greedy decoding of the target ([`crate::joiner::Joiner::contexts`]);
//! with it the student trains on the teacher's token distributions.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, ensure};

use crate::config::{N_MELS, OUT_FRAMES, StudentConfig};
use crate::joiner::NO_TOKEN;

/// f32 values of one record's features.
pub fn feat_len(k: usize) -> usize {
    StudentConfig::segment_frames(k) * N_MELS
}

/// f32 values of one record's target.
pub fn target_len(k: usize) -> usize {
    k * OUT_FRAMES * 512
}

/// Appends records.
pub struct Writer {
    file: std::io::BufWriter<std::fs::File>,
    k: usize,
    n: usize,
}

impl Writer {
    /// Start `dir/segments.f32`.
    pub fn create(dir: &Path, k: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let file = std::fs::File::create(dir.join("segments.f32"))?;
        Ok(Self {
            file: std::io::BufWriter::with_capacity(1 << 24, file),
            k,
            n: 0,
        })
    }

    /// One record.
    pub fn push(&mut self, feats: &[f32], target: &[f32]) -> Result<()> {
        ensure!(feats.len() == feat_len(self.k) && target.len() == target_len(self.k));
        for v in feats.iter().chain(target) {
            self.file.write_all(&v.to_le_bytes())?;
        }
        self.n += 1;
        Ok(())
    }

    /// Flush and write `meta.json`; returns the record count.
    pub fn finish(mut self, dir: &Path, extra: serde_json::Value) -> Result<usize> {
        self.file.flush()?;
        let mut meta = serde_json::json!({"chunks": self.k, "records": self.n});
        if let (Some(m), Some(e)) = (meta.as_object_mut(), extra.as_object()) {
            m.extend(e.clone());
        }
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
        Ok(self.n)
    }
}

/// Output frames of one record (= its predictor contexts).
pub fn frames_len(k: usize) -> usize {
    k * OUT_FRAMES
}

/// Write `dir/contexts.u16` (one entry per output frame of every record).
pub fn write_contexts(dir: &Path, contexts: &[[u16; 2]]) -> Result<()> {
    let bytes: Vec<u8> = contexts
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(dir.join("contexts.u16"), bytes)?;
    Ok(())
}

/// A record set, memory-mapped.
pub struct Records {
    map: memmap2::Mmap,
    contexts: Option<memmap2::Mmap>,
    /// Chunks per segment.
    pub k: usize,
    /// Records.
    pub n: usize,
}

impl Records {
    /// Open `dir` (`segments.f32` + `meta.json`).
    pub fn open(dir: &Path) -> Result<Self> {
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
        let k = meta["chunks"].as_u64().context("meta chunks")? as usize;
        let n = meta["records"].as_u64().context("meta records")? as usize;
        let file = std::fs::File::open(dir.join("segments.f32"))?;
        // SAFETY: read-only map of a file this process does not modify.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        ensure!(
            map.len() == n * (feat_len(k) + target_len(k)) * 4,
            "segments.f32 size does not match meta.json"
        );
        let path = dir.join("contexts.u16");
        let contexts = if path.exists() {
            let file = std::fs::File::open(&path)?;
            // SAFETY: as above.
            let map = unsafe { memmap2::Mmap::map(&file)? };
            ensure!(
                map.len() == n * frames_len(k) * 4,
                "contexts.u16 size does not match meta.json"
            );
            Some(map)
        } else {
            None
        };
        Ok(Self {
            map,
            contexts,
            k,
            n,
        })
    }

    /// Whether `contexts.u16` is present.
    pub fn has_contexts(&self) -> bool {
        self.contexts.is_some()
    }

    /// Record `i`'s predictor context per output frame
    /// (`[NO_TOKEN, NO_TOKEN]` throughout without `contexts.u16`).
    pub fn contexts(&self, i: usize) -> Vec<[u16; 2]> {
        let f = frames_len(self.k);
        match &self.contexts {
            Some(m) => m[i * f * 4..(i + 1) * f * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| {
                    [
                        u16::from_le_bytes([b[0], b[1]]),
                        u16::from_le_bytes([b[2], b[3]]),
                    ]
                })
                .collect(),
            None => vec![[NO_TOKEN; 2]; f],
        }
    }

    /// Append record `i`'s features and target to the batch buffers.
    pub fn read_into(&self, i: usize, feats: &mut Vec<f32>, target: &mut Vec<f32>) {
        let (fl, tl) = (feat_len(self.k), target_len(self.k));
        let base = i * (fl + tl) * 4;
        let rec = &self.map[base..base + (fl + tl) * 4];
        let mut it = rec
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b));
        feats.extend(it.by_ref().take(fl));
        target.extend(it.take(tl));
    }
}
