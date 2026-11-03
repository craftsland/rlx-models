// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! The teacher's predictor and joiner in f32, as `rlx-kroko-core` runs them:
//! `logits = W · tanh(enc + dec) + b`, with `dec` the predictor output for
//! the last two tokens. Used to turn the teacher's encoder output into the
//! frame-level token distributions the student is trained to match: greedy
//! decoding finds the context in effect at every frame
//! ([`Joiner::contexts`]), and that context's predictor output goes into the
//! training graph next to the teacher's encoder output.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};

use crate::init::Teacher;

/// Encoder output / predictor width.
pub const DIM: usize = 512;

/// A context slot with no token yet (the core's `-1`).
pub const NO_TOKEN: u16 = u16::MAX;

/// The teacher's predictor and joiner.
pub struct Joiner {
    emb: Vec<f32>,
    conv: Vec<f32>,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
    /// Joiner output weight `[vocab, 512]`.
    pub out_w: Vec<f32>,
    /// Joiner output bias `[vocab]`.
    pub out_b: Vec<f32>,
    /// Tokens.
    pub vocab: usize,
    blank: usize,
    unk: usize,
    cache: std::sync::Mutex<HashMap<[u16; 2], std::sync::Arc<Vec<f32>>>>,
}

impl Joiner {
    /// The `decoder.*` / `joiner.*` tensors of `teacher` (`dir` for its
    /// blank and unk ids).
    pub fn open(teacher: &Teacher, dir: &Path) -> Result<Self> {
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let id = |k: &str| -> Result<usize> {
            Ok(cfg[k]
                .as_u64()
                .with_context(|| format!("config.json: {k}"))? as usize)
        };
        let get = |name: &str, len: usize| -> Result<Vec<f32>> {
            let (d, _) = teacher
                .weights
                .get(name)
                .with_context(|| format!("teacher has no {name}"))?;
            ensure!(d.len() == len, "{name}: {} values, expected {len}", d.len());
            Ok(d.to_vec())
        };
        let vocab = teacher.vocab;
        ensure!(
            teacher.context == 2,
            "predictor context {}",
            teacher.context
        );
        Ok(Self {
            emb: get("decoder.decoder.embedding.weight", vocab * DIM)?,
            conv: get("decoder.decoder.conv.weight", DIM * 4 * 2)?,
            proj_w: get("decoder.decoder_proj.weight", DIM * DIM)?,
            proj_b: get("decoder.decoder_proj.bias", DIM)?,
            out_w: get("joiner.output_linear.weight", vocab * DIM)?,
            out_b: get("joiner.output_linear.bias", vocab)?,
            vocab,
            blank: id("blank_id")?,
            unk: id("unk_id")?,
            cache: Default::default(),
        })
    }

    /// The context before any token: nothing, then blank.
    pub fn start(&self) -> [u16; 2] {
        [NO_TOKEN, self.blank as u16]
    }

    /// Predictor output `[512]` for the last two tokens (memoized).
    pub fn predictor(&self, ctx: [u16; 2]) -> std::sync::Arc<Vec<f32>> {
        if let Some(v) = self.cache.lock().expect("cache").get(&ctx) {
            return v.clone();
        }
        let mut emb = [0f32; 2 * DIM];
        for (p, &tok) in ctx.iter().enumerate() {
            if tok != NO_TOKEN {
                let t = tok as usize;
                emb[p * DIM..(p + 1) * DIM].copy_from_slice(&self.emb[t * DIM..(t + 1) * DIM]);
            }
        }
        // Grouped conv over the two positions (4 channels per group), ReLU.
        let mut conv = [0f32; DIM];
        for (o, c) in conv.iter_mut().enumerate() {
            let start = o / 4 * 4;
            let mut acc = 0f32;
            for i in 0..4 {
                for p in 0..2 {
                    acc += emb[p * DIM + start + i] * self.conv[(o * 4 + i) * 2 + p];
                }
            }
            *c = acc.max(0.0);
        }
        let out: Vec<f32> = (0..DIM)
            .map(|o| {
                let w = &self.proj_w[o * DIM..(o + 1) * DIM];
                self.proj_b[o] + w.iter().zip(&conv).map(|(a, b)| a * b).sum::<f32>()
            })
            .collect();
        let out = std::sync::Arc::new(out);
        self.cache.lock().expect("cache").insert(ctx, out.clone());
        out
    }

    /// The joiner's top token for encoder frame `enc` (`[512]`) under `ctx`.
    pub fn best(&self, enc: &[f32], ctx: [u16; 2]) -> usize {
        let dec = self.predictor(ctx);
        let h: Vec<f32> = enc
            .iter()
            .zip(dec.iter())
            .map(|(a, b)| (a + b).tanh())
            .collect();
        let mut best = (0, f32::NEG_INFINITY);
        for v in 0..self.vocab {
            let w = &self.out_w[v * DIM..(v + 1) * DIM];
            let z = self.out_b[v] + w.iter().zip(&h).map(|(a, b)| a * b).sum::<f32>();
            if z > best.1 {
                best = (v, z);
            }
        }
        best.0
    }

    /// The blank token id.
    pub fn blank(&self) -> usize {
        self.blank
    }

    /// Whether `token` emits nothing (blank or unk).
    pub fn is_silent(&self, token: usize) -> bool {
        token == self.blank || token == self.unk
    }

    /// Greedy decoding of `enc` (`[frames, 512]`) as the core does it (at
    /// most one token per frame; blank and unk emit nothing). Returns the
    /// context in effect at each frame — the predictor input the joiner saw.
    pub fn contexts(&self, enc: &[f32]) -> Vec<[u16; 2]> {
        let mut ctx = self.start();
        let mut out = Vec::with_capacity(enc.len() / DIM);
        let mut h = [0f32; DIM];
        for e in enc.as_chunks::<DIM>().0 {
            out.push(ctx);
            let dec = self.predictor(ctx);
            for ((h, &a), &b) in h.iter_mut().zip(e).zip(dec.iter()) {
                *h = (a + b).tanh();
            }
            let mut best = (0, f32::NEG_INFINITY);
            for v in 0..self.vocab {
                let w = &self.out_w[v * DIM..(v + 1) * DIM];
                let z = self.out_b[v] + w.iter().zip(&h).map(|(a, b)| a * b).sum::<f32>();
                if z > best.1 {
                    best = (v, z);
                }
            }
            if best.0 != self.blank && best.0 != self.unk {
                ctx = [ctx[1], best.0 as u16];
            }
        }
        out
    }
}
