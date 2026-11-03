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

//! Export a checkpoint to the `.rms` blob [`rlx_moonshine_core`] runs on
//! MCUs, and run that blob on the host.
//!
//! Every matrix is quantized symmetric: int8 with one scale per output
//! channel, or int4 with one scale per `group` values. The 32k-row token
//! embedding and LM head — 21 of the tiny model's 34M parameters — default to
//! int4; everything else to int8. The frontend linear and causal convs get
//! one scale per 16 values: the encoder amplifies their error (a 1.3 % miss
//! after the convs became a 23 % miss after the first layer with one scale
//! per row). Encoder LayerNorm offsets (`γ + 1`) are folded, the convs are
//! reordered to `[out][tap][in]`.

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_mcu::blob::Writer;
use rlx_moonshine_core as core;

use crate::StreamingConfig;

/// Widths of the exported matrices.
#[derive(Debug, Clone, Copy)]
pub struct McuExportOptions {
    /// Bits of the ordinary matrices (8 or 4).
    pub bits: u32,
    /// Bits of the token embedding and LM head (8 or 4).
    pub vocab_bits: u32,
    /// Bits of the learned position table (8 or 4).
    pub pos_bits: u32,
    /// Values per int4 scale.
    pub group: usize,
    /// Encoding of the frontend linear and convs: 9 = int8 with one scale
    /// per `front_group` values (default), 8 = int8 per row.
    pub front_bits: u32,
    /// Values per scale in the frontend matrices.
    pub front_group: usize,
}

impl Default for McuExportOptions {
    fn default() -> Self {
        Self {
            bits: 8,
            vocab_bits: 4,
            pos_bits: 8,
            group: 32,
            front_bits: 9,
            front_group: 16,
        }
    }
}

struct Src<'a>(&'a WeightMap);

impl Src<'_> {
    fn get(&self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.0
            .get(k)
            .map(|(d, s)| (d.to_vec(), s.to_vec()))
            .ok_or_else(|| anyhow!("moonshine-streaming mcu export: missing {k}"))
    }
    fn vec(&self, k: &str, n: usize) -> Result<Vec<f32>> {
        let (d, _) = self.get(k)?;
        ensure!(d.len() == n, "{k}: {} values, expected {n}", d.len());
        Ok(d)
    }
}

/// `[out, in, k]` → `[out][k][in]`.
fn conv_reorder(w: &[f32], out: usize, inp: usize, k: usize) -> Vec<f32> {
    let mut r = vec![0f32; w.len()];
    for o in 0..out {
        for i in 0..inp {
            for j in 0..k {
                r[(o * k + j) * inp + i] = w[(o * inp + i) * k + j];
            }
        }
    }
    r
}

/// SentencePiece pieces by id from `tokenizer.json` (model vocab plus
/// added tokens); ids with no piece get an empty string.
fn pieces(tokenizer_json: &str, vocab: usize) -> Result<Vec<String>> {
    let v: serde_json::Value = serde_json::from_str(tokenizer_json)?;
    let mut out = vec![String::new(); vocab];
    if let Some(m) = v["model"]["vocab"].as_object() {
        for (piece, id) in m {
            if let Some(id) = id.as_u64().map(|i| i as usize)
                && id < vocab
            {
                out[id] = piece.clone();
            }
        }
    }
    for t in v["added_tokens"].as_array().into_iter().flatten() {
        if let (Some(id), Some(c)) = (t["id"].as_u64(), t["content"].as_str())
            && (id as usize) < vocab
        {
            out[id as usize] = c.to_string();
        }
    }
    Ok(out)
}

/// Build the `.rms` blob from a checkpoint directory.
pub fn export_rms(dir: &Path, opts: McuExportOptions) -> Result<Vec<u8>> {
    let cfg = StreamingConfig::from_file(&dir.join("config.json"))?;
    let (_, weights) = rlx_core::weights::open_map(dir.join("model.safetensors"))
        .with_context(|| format!("open {}", dir.display()))?;
    let s = Src(&weights);
    let e = &cfg.encoder;
    let dc = &cfg.decoder;
    let (d, a, ff) = (
        e.hidden_size,
        e.num_attention_heads * e.head_dim,
        e.intermediate_size,
    );
    let dd = dc.hidden_size;
    let dhd = dc.dec_head_dim();
    let dff = dc.decoder_intermediate_size;
    let has_proj = weights.get("model.decoder.proj.weight").is_some();
    let (b, g) = (opts.bits, opts.group);

    let mut w = Writer::new(core::model::MAGIC, core::model::VERSION);
    for v in [
        e.frame_len,
        d,
        e.num_hidden_layers,
        e.num_attention_heads,
        e.head_dim,
        ff,
        dd,
        dc.decoder_num_hidden_layers,
        dc.decoder_num_attention_heads,
        dhd,
        dff,
        dc.vocab_size,
        dc.n_rot(dhd),
        dc.decoder_start_token_id as usize,
        dc.eos_token_id as usize,
        dc.max_position_embeddings,
        usize::from(has_proj),
        dc.rope_theta.round() as usize,
    ] {
        w.u32(v as u32);
    }
    let mat = |w: &mut Writer, k: &str, rows: usize, cols: usize, bits: u32| -> Result<()> {
        w.mat(&s.vec(k, rows * cols)?, rows, cols, bits, g);
        Ok(())
    };
    let unit_ln = |w: &mut Writer, k: &str| -> Result<()> {
        let v: Vec<f32> = s.vec(k, d)?.iter().map(|x| x + 1.0).collect();
        w.f32s(&v);
        Ok(())
    };

    let pre = "model.encoder";
    w.f32s(&s.vec(&format!("{pre}.embedder.comp.log_k"), 1)?);
    let (fb, fg) = (opts.front_bits, opts.front_group);
    let lin = s.vec(&format!("{pre}.embedder.linear.weight"), d * e.frame_len)?;
    w.mat(&lin, d, e.frame_len, fb, fg);
    let c1 = s.vec(&format!("{pre}.embedder.conv1.weight"), 2 * d * d * 5)?;
    w.mat(&conv_reorder(&c1, 2 * d, d, 5), 2 * d, 5 * d, fb, fg);
    w.f32s(&s.vec(&format!("{pre}.embedder.conv1.bias"), 2 * d)?);
    let c2 = s.vec(&format!("{pre}.embedder.conv2.weight"), d * 2 * d * 5)?;
    w.mat(&conv_reorder(&c2, d, 2 * d, 5), d, 10 * d, fb, fg);
    w.f32s(&s.vec(&format!("{pre}.embedder.conv2.bias"), d)?);
    let win: Vec<u32> = e
        .sliding_windows
        .iter()
        .flat_map(|&(l, r)| [l as u32, r as u32])
        .collect();
    w.u32s(&win);
    for l in 0..e.num_hidden_layers {
        let p = |x: &str| format!("{pre}.layers.{l}.{x}");
        unit_ln(&mut w, &p("input_layernorm.gamma"))?;
        mat(&mut w, &p("self_attn.q_proj.weight"), a, d, b)?;
        mat(&mut w, &p("self_attn.k_proj.weight"), a, d, b)?;
        mat(&mut w, &p("self_attn.v_proj.weight"), a, d, b)?;
        mat(&mut w, &p("self_attn.o_proj.weight"), d, a, b)?;
        unit_ln(&mut w, &p("post_attention_layernorm.gamma"))?;
        mat(&mut w, &p("mlp.fc1.weight"), ff, d, b)?;
        w.f32s(&s.vec(&p("mlp.fc1.bias"), ff)?);
        mat(&mut w, &p("mlp.fc2.weight"), d, ff, b)?;
        w.f32s(&s.vec(&p("mlp.fc2.bias"), d)?);
    }
    unit_ln(&mut w, &format!("{pre}.final_norm.gamma"))?;
    mat(
        &mut w,
        "model.decoder.pos_emb.weight",
        dc.max_position_embeddings,
        d,
        opts.pos_bits,
    )?;
    if has_proj {
        mat(&mut w, "model.decoder.proj.weight", dd, d, b)?;
    }
    mat(
        &mut w,
        "model.decoder.embed_tokens.weight",
        dc.vocab_size,
        dd,
        opts.vocab_bits,
    )?;
    let da = dc.decoder_num_attention_heads * dhd;
    for l in 0..dc.decoder_num_hidden_layers {
        let p = |x: &str| format!("model.decoder.layers.{l}.{x}");
        w.f32s(&s.vec(&p("input_layernorm.weight"), dd)?);
        for m in ["q_proj", "k_proj", "v_proj"] {
            mat(&mut w, &p(&format!("self_attn.{m}.weight")), da, dd, b)?;
        }
        mat(&mut w, &p("self_attn.o_proj.weight"), dd, da, b)?;
        w.f32s(&s.vec(&p("post_attention_layernorm.weight"), dd)?);
        for m in ["q_proj", "k_proj", "v_proj"] {
            mat(&mut w, &p(&format!("encoder_attn.{m}.weight")), da, dd, b)?;
        }
        mat(&mut w, &p("encoder_attn.o_proj.weight"), dd, da, b)?;
        w.f32s(&s.vec(&p("final_layernorm.weight"), dd)?);
        mat(&mut w, &p("mlp.fc1.weight"), 2 * dff, dd, b)?;
        w.f32s(&s.vec(&p("mlp.fc1.bias"), 2 * dff)?);
        mat(&mut w, &p("mlp.fc2.weight"), dd, dff, b)?;
        w.f32s(&s.vec(&p("mlp.fc2.bias"), dd)?);
    }
    w.f32s(&s.vec("model.decoder.norm.weight", dd)?);
    let head = if weights.get("proj_out.weight").is_some() {
        "proj_out.weight"
    } else {
        "model.decoder.embed_tokens.weight"
    };
    mat(&mut w, head, dc.vocab_size, dd, opts.vocab_bits)?;

    let tok = std::fs::read_to_string(dir.join("tokenizer.json"))
        .with_context(|| format!("read {}/tokenizer.json", dir.display()))?;
    let (mut offs, mut bytes) = (vec![0u32], Vec::new());
    for p in pieces(&tok, dc.vocab_size)? {
        bytes.extend_from_slice(p.as_bytes());
        offs.push(bytes.len() as u32);
    }
    w.u32s(&offs);
    w.bytes(&bytes);
    Ok(w.finish())
}

/// A blob in 4-byte-aligned storage, as the core requires.
pub struct AlignedBlob {
    words: Vec<u32>,
    len: usize,
}

impl AlignedBlob {
    /// Copy `bytes` into aligned storage.
    pub fn new(bytes: &[u8]) -> Self {
        let mut words = vec![0u32; bytes.len().div_ceil(4)];
        for (i, c) in bytes.chunks(4).enumerate() {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            words[i] = u32::from_le_bytes(b);
        }
        Self {
            words,
            len: bytes.len(),
        }
    }

    /// The aligned bytes.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: u32 storage viewed as bytes; `len <= 4 * words.len()`.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.len) }
    }
}

/// Host run of the MCU pipeline: `(token ids, text, encoder hidden)`.
pub fn transcribe_rms(blob: &[u8], pcm: &[f32]) -> Result<(Vec<u16>, String, Vec<f32>)> {
    let model = core::Model::parse(blob).map_err(|e| anyhow!("{e}"))?;
    let need = core::scratch_len(&model, pcm.len());
    let mut fs = vec![0f32; need.f32s];
    let mut is = vec![0i8; need.i8s];
    let mut ids = vec![0u16; model.config().max_tokens(pcm.len())];
    let (_, _, t) = model.config().lengths(pcm.len());
    let mut hidden = vec![0f32; t * model.config().d_enc];
    let n = core::transcribe_with_hidden(&model, pcm, &mut fs, &mut is, &mut ids, &mut hidden)
        .map_err(|e| anyhow!("{e}"))?;
    ids.truncate(n);
    let mut text = vec![0u8; n * 32 + 1];
    let len = model.decode(&ids, &mut text);
    Ok((
        ids,
        String::from_utf8_lossy(&text[..len]).into_owned(),
        hidden,
    ))
}
