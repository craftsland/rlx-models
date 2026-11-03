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

//! Export a Kroko model to the `.rkr` file [`rlx_kroko_core`] runs, from
//! the named float weights audio.cpp's `convert_kroko_onnx.py` recovers from
//! a free Kroko `.data` package (`model.safetensors`, `config.json`,
//! `tokens.txt`).
//!
//! Layer matrices are int8 (one scale per row) or int4 (one scale per 32);
//! the conv-embed matrices int8 with one scale per 16 values; the predictor
//! and joiner int8. The resident part comes first, then one block per
//! Zipformer layer, so the resident part can be loaded into RAM and the
//! blocks paged from NAND.

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_kroko_core::model::{EMBED_C0, EMBED_C4, EMBED_C7, EMBED_HID, MAGIC, VERSION};
use rlx_mcu::blob::Writer;

/// Matrix encodings.
#[derive(Debug, Clone, Copy)]
pub struct KrokoExportOptions {
    /// Zipformer layer matrices: 8 (int8 per row), 4 (int4), 9 (int8 per 16).
    pub layer_bits: u32,
    /// Group of int4 / grouped int8 scales.
    pub group: usize,
}

impl Default for KrokoExportOptions {
    fn default() -> Self {
        Self {
            layer_bits: 8,
            group: 32,
        }
    }
}

struct Src<'a>(&'a WeightMap);

impl Src<'_> {
    fn v(&self, k: &str, n: usize) -> Result<Vec<f32>> {
        let (d, _) = self
            .0
            .get(k)
            .ok_or_else(|| anyhow!("kroko export: missing {k}"))?;
        ensure!(d.len() == n, "{k}: {} values, expected {n}", d.len());
        Ok(d.to_vec())
    }
}

/// Kaldi povey window (`hann^0.85`) and HTK mel bands (20–7600 Hz) over the
/// 256 bins below Nyquist, computed in f32 as the reference does.
fn fbank_tables() -> (Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>) {
    let win: Vec<f32> = (0..400)
        .map(|i| {
            let hann = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / 399.0).cos();
            hann.powf(0.85)
        })
        .collect();
    let mel = |hz: f32| 1127.0f32 * (1.0 + hz / 700.0).ln();
    let (low, high) = (mel(20.0), mel(7600.0));
    let delta = (high - low) / 81.0;
    let (mut starts, mut offs, mut w) = (Vec::new(), vec![0u32], Vec::new());
    for band in 0..80 {
        let left = low + band as f32 * delta;
        let center = low + (band + 1) as f32 * delta;
        let right = low + (band + 2) as f32 * delta;
        let vals: Vec<f32> = (0..256)
            .map(|bin| {
                let p = mel(bin as f32 * 16000.0 / 512.0);
                if p > left && p < right {
                    if p <= center {
                        (p - left) / (center - left)
                    } else {
                        (right - p) / (right - center)
                    }
                } else {
                    0.0
                }
            })
            .collect();
        let first = vals.iter().position(|&v| v != 0.0).unwrap_or(0);
        let last = vals.iter().rposition(|&v| v != 0.0).unwrap_or(0);
        if vals[first] != 0.0 {
            w.extend_from_slice(&vals[first..=last]);
        }
        starts.push(first as u32);
        offs.push(w.len() as u32);
    }
    (win, starts, offs, w)
}

/// Build the `.rkr` file from a converted Kroko directory.
pub fn export_rkr(dir: &Path, opts: KrokoExportOptions) -> Result<Vec<u8>> {
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let (_, weights) = rlx_core::weights::open_map(dir.join("model.safetensors"))
        .with_context(|| format!("open {}/model.safetensors", dir.display()))?;
    let s = Src(&weights);
    let ints = |k: &str| -> Result<Vec<usize>> {
        cfg[k]
            .as_array()
            .ok_or_else(|| anyhow!("config.json: {k}"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|x| x as usize)
                    .ok_or_else(|| anyhow!("config.json: {k}"))
            })
            .collect()
    };
    let one = |k: &str| {
        cfg[k]
            .as_u64()
            .map(|x| x as usize)
            .ok_or_else(|| anyhow!("config.json: {k}"))
    };
    let (dims, layers, downs, lefts, heads) = (
        ints("encoder_dims")?,
        ints("num_encoder_layers")?,
        ints("downsampling_factors")?,
        ints("left_context_len")?,
        ints("num_heads")?,
    );
    let (qd, vd, kern) = (
        ints("query_head_dims")?,
        ints("value_head_dims")?,
        ints("cnn_module_kernels")?,
    );
    let n_stacks = dims.len();
    let vocab = one("vocab_size")?;
    let (pos_dim, pos_head) = (48usize, 4usize);
    let embed_dim = 192usize;
    let enc_out =
        s.0.get("encoder.encoder_proj.bias")
            .map(|(d, _)| d.len())
            .ok_or_else(|| anyhow!("encoder_proj"))?;
    let dec_hidden =
        s.0.get("decoder.decoder_proj.bias")
            .map(|(d, _)| d.len())
            .ok_or_else(|| anyhow!("decoder_proj"))?;
    let max_dim = *dims.iter().max().unwrap_or(&0);
    let f7 = ((80 - 3) / 2 + 1 - 3) / 2 + 1;
    let layer_prefix = |st: usize, l: usize| {
        if st == 0 {
            format!("encoder.encoder.encoders.0.layers.{l}.")
        } else {
            format!("encoder.encoder.encoders.{st}.encoder.layers.{l}.")
        }
    };
    let ff_dims: Vec<[usize; 3]> = (0..n_stacks)
        .map(|st| {
            let p = layer_prefix(st, 0);
            let w = |i: usize| {
                s.0.get(&format!("{p}feed_forward{i}.in_proj.bias"))
                    .map_or(0, |(d, _)| d.len())
            };
            [w(1), w(2), w(3)]
        })
        .collect();

    // Layer blocks.
    let g = opts.group;
    let mut blocks = Vec::new();
    for st in 0..n_stacks {
        let (d, h, k) = (dims[st], heads[st], kern[st]);
        let hv = h * vd[st];
        let nl = 3 * d / 4;
        for l in 0..layers[st] {
            let p = layer_prefix(st, l);
            let mut w = Writer::default();
            let lin = |w: &mut Writer, name: &str, o: usize, i: usize| -> Result<()> {
                let (bits, grp) = match opts.layer_bits {
                    // Grouped encodings need the group to divide the row.
                    4 | 9 if i.is_multiple_of(g) => (opts.layer_bits, g),
                    4 | 9 if i.is_multiple_of(16) => (opts.layer_bits, 16),
                    4 | 9 => (8, g),
                    b => (b, g),
                };
                w.mat(&s.v(&format!("{p}{name}.weight"), o * i)?, o, i, bits, grp);
                w.f32s(&s.v(&format!("{p}{name}.bias"), o)?);
                Ok(())
            };
            w.f32s(&s.v(&format!("{p}bypass.bypass_scale"), d)?);
            w.f32s(&s.v(&format!("{p}bypass_mid.bypass_scale"), d)?);
            w.f32s(&s.v(&format!("{p}norm.bias"), d)?);
            w.f32s(&s.v(&format!("{p}norm.log_scale"), 1)?);
            lin(
                &mut w,
                "self_attn_weights.in_proj",
                2 * h * qd[st] + h * pos_head,
                d,
            )?;
            w.f32s(&s.v(
                &format!("{p}self_attn_weights.linear_pos.weight"),
                h * pos_head * pos_dim,
            )?);
            for (i, &f) in ff_dims[st].iter().enumerate() {
                lin(&mut w, &format!("feed_forward{}.in_proj", i + 1), f, d)?;
                lin(&mut w, &format!("feed_forward{}.out_proj", i + 1), d, f)?;
            }
            lin(&mut w, "nonlin_attention.in_proj", 3 * nl, d)?;
            lin(&mut w, "nonlin_attention.out_proj", d, nl)?;
            for a in ["self_attn1", "self_attn2"] {
                lin(&mut w, &format!("{a}.in_proj"), hv, d)?;
                lin(&mut w, &format!("{a}.out_proj"), d, hv)?;
            }
            for c in ["conv_module1", "conv_module2"] {
                lin(&mut w, &format!("{c}.in_proj"), 2 * d, d)?;
                let dc = format!("{p}{c}.depthwise_conv");
                w.f32s(&s.v(&format!("{dc}.causal_conv.weight"), d * (k / 2 + 1))?);
                w.f32s(&s.v(&format!("{dc}.causal_conv.bias"), d)?);
                w.f32s(&s.v(&format!("{dc}.chunkwise_conv.weight"), d * k)?);
                w.f32s(&s.v(&format!("{dc}.chunkwise_conv.bias"), d)?);
                w.f32s(&s.v(&format!("{dc}.chunk_scale_left"), d * k)?);
                w.f32s(&s.v(&format!("{dc}.chunk_scale_right"), d * k)?);
                lin(&mut w, &format!("{c}.out_proj"), d, d)?;
            }
            blocks.push(w.finish());
        }
    }

    // Resident part; written twice so the table can hold final offsets.
    let pieces = std::fs::read_to_string(dir.join("tokens.txt"))?;
    let mut piece_list = vec![String::new(); vocab];
    for line in pieces.lines() {
        if let Some((p, id)) = line.rsplit_once(' ')
            && let Ok(id) = id.parse::<usize>()
            && id < vocab
        {
            piece_list[id] = p.to_string();
        }
    }
    let resident = |resident_len: usize, table: &[u32]| -> Result<Vec<u8>> {
        let mut w = Writer::new(MAGIC, VERSION);
        w.u32(resident_len as u32);
        for v in [
            16_000,
            80,
            one("chunk_size")?,
            one("chunk_shift")?,
            embed_dim,
            enc_out,
            vocab,
            one("blank_id")?,
            one("unk_id")?,
            one("context_size")?,
            dec_hidden,
            pos_dim,
            pos_head,
            n_stacks,
        ] {
            w.u32(v as u32);
        }
        for st in 0..n_stacks {
            let ff = ff_dims[st];
            for v in [
                dims[st], layers[st], downs[st], lefts[st], heads[st], qd[st], vd[st], kern[st],
                ff[0], ff[1], ff[2],
            ] {
                w.u32(v as u32);
            }
        }
        let (win, starts, offs, bands) = fbank_tables();
        w.f32s(&win);
        w.u32s(&starts);
        w.u32s(&offs);
        w.f32s(&bands);
        let e = "encoder.encoder_embed";
        w.f32s(&s.v(&format!("{e}.conv.0.weight"), EMBED_C0 * 9)?);
        w.f32s(&s.v(&format!("{e}.conv.0.bias"), EMBED_C0)?);
        w.f32s(&s.v(&format!("{e}.conv.4.weight"), EMBED_C4 * EMBED_C0 * 9)?);
        w.f32s(&s.v(&format!("{e}.conv.4.bias"), EMBED_C4)?);
        w.f32s(&s.v(&format!("{e}.conv.7.weight"), EMBED_C7 * EMBED_C4 * 9)?);
        w.f32s(&s.v(&format!("{e}.conv.7.bias"), EMBED_C7)?);
        w.f32s(&s.v(
            &format!("{e}.convnext.depthwise_conv.weight"),
            EMBED_C7 * 49,
        )?);
        w.f32s(&s.v(&format!("{e}.convnext.depthwise_conv.bias"), EMBED_C7)?);
        w.mat(
            &s.v(
                &format!("{e}.convnext.pointwise_conv1.weight"),
                EMBED_HID * EMBED_C7,
            )?,
            EMBED_HID,
            EMBED_C7,
            9,
            16,
        );
        w.f32s(&s.v(&format!("{e}.convnext.pointwise_conv1.bias"), EMBED_HID)?);
        w.mat(
            &s.v(
                &format!("{e}.convnext.pointwise_conv2.weight"),
                EMBED_C7 * EMBED_HID,
            )?,
            EMBED_C7,
            EMBED_HID,
            9,
            16,
        );
        w.f32s(&s.v(&format!("{e}.convnext.pointwise_conv2.bias"), EMBED_C7)?);
        w.mat(
            &s.v(&format!("{e}.out.weight"), embed_dim * EMBED_C7 * f7)?,
            embed_dim,
            EMBED_C7 * f7,
            9,
            16,
        );
        w.f32s(&s.v(&format!("{e}.out.bias"), embed_dim)?);
        w.f32s(&s.v(&format!("{e}.out_norm.bias"), embed_dim)?);
        w.f32s(&s.v(&format!("{e}.out_norm.log_scale"), 1)?);
        for st in 0..n_stacks {
            if downs[st] > 1 {
                w.f32s(&s.v(
                    &format!("encoder.encoder.encoders.{st}.downsample.weights"),
                    downs[st],
                )?);
                w.f32s(&s.v(
                    &format!("encoder.encoder.encoders.{st}.out_combiner.bypass_scale"),
                    dims[st],
                )?);
            } else {
                w.f32s(&[]);
                w.f32s(&[]);
            }
        }
        w.f32s(&s.v("encoder.encoder.downsample_output.weights", 2)?);
        w.mat(
            &s.v("encoder.encoder_proj.weight", enc_out * max_dim)?,
            enc_out,
            max_dim,
            8,
            g,
        );
        w.f32s(&s.v("encoder.encoder_proj.bias", enc_out)?);
        w.mat(
            &s.v("decoder.decoder.embedding.weight", vocab * dec_hidden)?,
            vocab,
            dec_hidden,
            8,
            g,
        );
        let conv =
            s.0.get("decoder.decoder.conv.weight")
                .ok_or_else(|| anyhow!("decoder conv"))?
                .0
                .to_vec();
        w.f32s(&conv);
        w.mat(
            &s.v("decoder.decoder_proj.weight", dec_hidden * dec_hidden)?,
            dec_hidden,
            dec_hidden,
            8,
            g,
        );
        w.f32s(&s.v("decoder.decoder_proj.bias", dec_hidden)?);
        w.mat(
            &s.v("joiner.output_linear.weight", vocab * enc_out)?,
            vocab,
            enc_out,
            8,
            g,
        );
        w.f32s(&s.v("joiner.output_linear.bias", vocab)?);
        w.u32s(table);
        let (mut po, mut pb) = (vec![0u32], Vec::new());
        for p in &piece_list {
            pb.extend_from_slice(p.as_bytes());
            po.push(pb.len() as u32);
        }
        w.u32s(&po);
        w.bytes(&pb);
        Ok(w.finish())
    };
    let draft = resident(0, &vec![0u32; 2 * blocks.len()])?;
    let r_len = draft.len();
    let mut table = Vec::new();
    let mut off = r_len;
    for b in &blocks {
        table.push(off as u32);
        table.push(b.len() as u32);
        off += b.len();
    }
    let mut out = resident(r_len, &table)?;
    ensure!(out.len() == r_len, "resident part changed size");
    for b in blocks {
        out.extend_from_slice(&b);
    }
    Ok(out)
}
