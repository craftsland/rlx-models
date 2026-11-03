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

//! Model description read from the two GGUF files.
//!
//! The LM (`general.architecture = lfm3`) carries its own hyperparameters;
//! the mmproj (`clip`, projector types `lfm3a` / `lfm3a_gen`) describes the
//! speech encoder and depth decoder. The codec decoder has no metadata at
//! all: its widths come from the tensors and its attention window (30 codec
//! steps, `LiquidCodecConfig.decoder_window`) is fixed by the release.

use anyhow::{Result, anyhow, bail};
use rlx_gguf::{GgufFile, MetaValue};

use crate::trunk::{LM_NAMES, MMPROJ_NAMES, Mixer, Rope, TrunkSpec};

/// `LiquidCodecConfig.decoder_window` of the released codec.
pub const CODEC_WINDOW: usize = 30;
/// Codec STFT: `n_fft = win = 1280`, `hop = 320`, 6 STFT frames per code
/// frame → 1920 samples per frame at 24 kHz.
pub const CODEC_N_FFT: usize = 1280;
pub const CODEC_HOP: usize = 320;
pub const CODEC_FRAME_STACK: usize = 6;
pub const CODEC_SAMPLE_RATE: u32 = 24_000;
/// Speech encoder input: 16 kHz, 128 Slaney mels, 25 ms / 10 ms.
pub const INPUT_SAMPLE_RATE: u32 = 16_000;
/// Encoder RoPE base (`Lfm3AudioEncoderConfig` / LFM2 default).
const ENCODER_THETA: f64 = 1e6;
/// Depth decoder and codec RoPE base.
const GEN_THETA: f64 = 1e6;

fn meta<'a>(f: &'a GgufFile, key: &str) -> Result<&'a MetaValue> {
    f.metadata
        .get(key)
        .ok_or_else(|| anyhow!("GGUF metadata `{key}` missing"))
}

fn meta_u(f: &GgufFile, key: &str) -> Result<usize> {
    meta(f, key)?
        .as_u32()
        .map(|v| v as usize)
        .ok_or_else(|| anyhow!("`{key}` is not an integer"))
}

fn meta_f(f: &GgufFile, key: &str) -> Result<f64> {
    match meta(f, key)? {
        MetaValue::F32(v) => Ok(*v as f64),
        MetaValue::F64(v) => Ok(*v),
        other => other
            .as_u32()
            .map(f64::from)
            .ok_or_else(|| anyhow!("`{key}` is not a number")),
    }
}

fn meta_f_or(f: &GgufFile, key: &str, default: f64) -> f64 {
    meta_f(f, key).unwrap_or(default)
}

fn meta_array_u(f: &GgufFile, key: &str) -> Result<Vec<usize>> {
    match meta(f, key)? {
        MetaValue::Array(items) => items
            .iter()
            .map(|v| {
                v.as_u32()
                    .map(|x| x as usize)
                    .ok_or_else(|| anyhow!("`{key}`: non-integer entry"))
            })
            .collect(),
        _ => bail!("`{key}` is not an array"),
    }
}

fn meta_array_bool(f: &GgufFile, key: &str) -> Result<Vec<bool>> {
    match meta(f, key)? {
        MetaValue::Array(items) => items
            .iter()
            .map(|v| match v {
                MetaValue::Bool(b) => Ok(*b),
                other => other
                    .as_u32()
                    .map(|x| x != 0)
                    .ok_or_else(|| anyhow!("`{key}`: non-boolean entry")),
            })
            .collect(),
        _ => bail!("`{key}` is not an array"),
    }
}

fn meta_str<'a>(f: &'a GgufFile, key: &str) -> Option<&'a str> {
    match f.metadata.get(key)? {
        MetaValue::String(s) => Some(s.as_str()),
        _ => None,
    }
}

/// A tensor's shape in safetensors order (outermost first).
fn shape(f: &GgufFile, name: &str) -> Result<Vec<usize>> {
    let t = f
        .get(name)
        .ok_or_else(|| anyhow!("tensor `{name}` missing"))?;
    let mut s: Vec<usize> = t.shape.clone();
    s.reverse();
    Ok(s)
}

fn has(f: &GgufFile, name: &str) -> bool {
    f.get(name).is_some()
}

/// Count `{prefix}{i}.{probe}` tensors from `i = 0`.
fn count_layers(f: &GgufFile, prefix: &str, probe: &str) -> usize {
    (0..)
        .take_while(|i| has(f, &format!("{prefix}{i}.{probe}")))
        .count()
}

/// Special token ids of the chat protocol.
#[derive(Debug, Clone)]
pub struct Tokens {
    pub bos: u32,
    pub im_start: u32,
    pub im_end: u32,
    /// `<|audio|>`: one per speech-encoder output in the prompt.
    pub audio: u32,
    /// `<|audio_out|>`: one per speech frame the model wrote.
    pub audio_out: u32,
    /// `<|audio_start|>`: what follows is speech until an end-of-audio frame.
    pub audio_start: u32,
    /// `<|text_end|>`: ends the text of an interleaved reply (or a turn).
    pub text_end: u32,
    /// `<|mixed_start|>`: blocks of text and speech follow.
    pub mixed_start: u32,
}

impl Tokens {
    fn from_vocab(f: &GgufFile) -> Result<Self> {
        let MetaValue::Array(vocab) = meta(f, "tokenizer.ggml.tokens")? else {
            bail!("tokenizer.ggml.tokens is not an array");
        };
        let find = |s: &str| -> Result<u32> {
            vocab
                .iter()
                .position(|v| matches!(v, MetaValue::String(t) if t == s))
                .map(|i| i as u32)
                .ok_or_else(|| anyhow!("token `{s}` not in the vocabulary"))
        };
        Ok(Self {
            bos: meta_u(f, "tokenizer.ggml.bos_token_id")? as u32,
            im_start: find("<|im_start|>")?,
            im_end: find("<|im_end|>")?,
            audio: find("<|audio|>")?,
            audio_out: find("<|audio_out|>")?,
            audio_start: find("<|audio_start|>")?,
            text_end: find("<|text_end|>")?,
            mixed_start: find("<|mixed_start|>")?,
        })
    }
}

/// Speech-sampling defaults stored in the mmproj (`clip.gen.audio.sampling.*`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechSampling {
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    /// Frames of history the repetition penalty looks at.
    pub repetition_window: usize,
}

impl Default for SpeechSampling {
    /// The model card's recommendation.
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 128,
            repetition_penalty: 1.1,
            repetition_window: 4096,
        }
    }
}

/// Everything the runner needs to know about the checkpoint.
#[derive(Debug, Clone)]
pub struct Lfm3AudioConfig {
    pub lm: TrunkSpec,
    pub vocab_size: usize,
    pub tokens: Tokens,
    pub encoder: TrunkSpec,
    pub num_mel_bins: usize,
    /// Channels of the stride-2 downsampler convolutions.
    pub downsample_channels: usize,
    pub depth: TrunkSpec,
    pub num_codebooks: usize,
    /// Codes per codebook, the end-of-audio code (`codebook_size - 1`) included.
    pub codebook_size: usize,
    /// MaskGIT passes: codebooks revealed per pass.
    pub maskgit_schedule: Vec<usize>,
    pub codec: TrunkSpec,
    /// FSQ latent width per codebook.
    pub fsq_dim: usize,
    pub speech_sampling: SpeechSampling,
}

impl Lfm3AudioConfig {
    /// The code of an end-of-audio frame.
    pub fn audio_eos_code(&self) -> u32 {
        (self.codebook_size - 1) as u32
    }

    /// Read both files' headers.
    pub fn from_gguf(lm: &GgufFile, mmproj: &GgufFile) -> Result<Self> {
        let arch = meta_str(lm, "general.architecture").unwrap_or("?");
        if arch != "lfm3" {
            bail!("expected an `lfm3` GGUF, got `{arch}`");
        }
        if meta_str(mmproj, "clip.audio.projector_type") != Some("lfm3a")
            || meta_str(mmproj, "clip.gen.audio.projector_type") != Some("lfm3a_gen")
        {
            bail!("expected the LFM3-Audio mmproj (projector types `lfm3a` + `lfm3a_gen`)");
        }
        let lm_spec = lm_spec(lm)?;
        let vocab_size = shape(lm, "token_embd.weight")?[0];
        let tokens = Tokens::from_vocab(lm)?;

        let (encoder, downsample_channels) = encoder_spec(mmproj)?;
        let depth = depth_spec(mmproj)?;
        let codec = codec_spec(mmproj)?;

        // `a.gen.code.embd.weight` is `[codebooks, codebook_size, dim]`.
        let embd = shape(mmproj, "a.gen.code.embd.weight")?;
        let (num_codebooks, codebook_size) = (embd[0], embd[1]);
        // `[fsq levels product, fsq_dim]`, the end-of-audio code excluded.
        let fsq = shape(mmproj, "a.gen.wav.quant.first.codebook.weight")?;
        if fsq[0] + 1 != codebook_size {
            bail!(
                "codec codebook has {} entries, the depth decoder {codebook_size}",
                fsq[0]
            );
        }
        let steps = 3.min(num_codebooks);
        let mut maskgit_schedule = vec![num_codebooks / steps; steps];
        maskgit_schedule[steps - 1] += num_codebooks - (num_codebooks / steps) * steps;

        let speech_sampling = SpeechSampling {
            temperature: meta_f_or(mmproj, "clip.gen.audio.sampling.temp", 1.0) as f32,
            top_k: meta_f_or(mmproj, "clip.gen.audio.sampling.top_k", 128.0) as usize,
            repetition_penalty: meta_f_or(mmproj, "clip.gen.audio.sampling.penalty_repeat", 1.1)
                as f32,
            repetition_window: meta_f_or(mmproj, "clip.gen.audio.sampling.penalty_last_n", 4096.0)
                as usize,
        };

        Ok(Self {
            lm: lm_spec,
            vocab_size,
            tokens,
            encoder,
            num_mel_bins: meta_u(mmproj, "clip.audio.num_mel_bins")?,
            downsample_channels,
            depth,
            num_codebooks,
            codebook_size,
            maskgit_schedule,
            codec,
            fsq_dim: fsq[1],
            speech_sampling,
        })
    }
}

/// `rope_freqs.weight` holds llama.cpp frequency divisors: `1` for the
/// rotated prefix, `1e30` (frequency ≈ 0) for the rest. Returns the rotated
/// fraction.
fn partial_from_factors(factors: &[f32]) -> Result<f64> {
    let angles = factors.iter().take_while(|&&f| f == 1.0).count();
    if factors[angles..].iter().any(|&f| f < 1e20) {
        bail!("rope_freqs: only a rotated prefix + zeroed tail is supported, got {factors:?}");
    }
    Ok(angles as f64 / factors.len() as f64)
}

fn rope_factors(f: &GgufFile) -> Result<Option<Vec<f32>>> {
    if !has(f, "rope_freqs.weight") {
        return Ok(None);
    }
    let (data, _) = f.dequant_f32("rope_freqs.weight")?;
    Ok(Some(data))
}

fn lm_spec(f: &GgufFile) -> Result<TrunkSpec> {
    let n = meta_u(f, "lfm3.block_count")?;
    let hidden = meta_u(f, "lfm3.embedding_length")?;
    let heads = meta_u(f, "lfm3.attention.head_count")?;
    let kv = meta_array_u(f, "lfm3.attention.head_count_kv")?;
    let swa = meta_array_bool(f, "lfm3.attention.sliding_window_pattern")?;
    let window = meta_u(f, "lfm3.attention.sliding_window")?;
    let head_dim = meta_u(f, "lfm3.attention.key_length").unwrap_or(hidden / heads);
    if kv.len() != n || swa.len() != n {
        bail!("lfm3: per-layer arrays do not cover {n} layers");
    }
    let partial = match rope_factors(f)? {
        Some(factors) => partial_from_factors(&factors)?,
        None => 1.0,
    };
    let ropes = vec![
        Rope {
            theta: meta_f(f, "lfm3.rope.freq_base")?,
            partial,
        },
        Rope {
            theta: meta_f(f, "lfm3.rope.freq_base_swa")?,
            partial: 1.0,
        },
    ];
    let mut kv_heads = 0;
    let layers = (0..n)
        .map(|i| {
            if kv[i] == 0 {
                Mixer::Conv
            } else {
                kv_heads = kv[i];
                if swa[i] {
                    Mixer::Attn {
                        rope: 1,
                        window: Some(window),
                    }
                } else {
                    Mixer::Attn {
                        rope: 0,
                        window: None,
                    }
                }
            }
        })
        .collect();
    Ok(TrunkSpec {
        prefix: "blk.".into(),
        final_norm: "output_norm.weight".into(),
        names: LM_NAMES,
        hidden,
        heads,
        kv_heads,
        head_dim,
        conv_kernel: meta_u(f, "lfm3.shortconv.l_cache")?,
        eps: meta_f(f, "lfm3.attention.layer_norm_rms_epsilon")? as f32,
        layers,
        ropes,
        causal: true,
        gate: true,
        sandwich: has(f, "blk.0.post_attention_norm.weight"),
    })
}

/// Layers whose attention tensors exist are attention, the rest convolution.
fn mixers(f: &GgufFile, prefix: &str, n: usize, window: Option<usize>) -> Vec<Mixer> {
    (0..n)
        .map(|i| {
            if has(f, &format!("{prefix}{i}.attn_q.weight")) {
                Mixer::Attn { rope: 0, window }
            } else {
                Mixer::Conv
            }
        })
        .collect()
}

fn conv_kernel(f: &GgufFile, prefix: &str, n: usize) -> Result<usize> {
    for i in 0..n {
        let name = format!("{prefix}{i}.conv_dw.weight");
        if has(f, &name) {
            return Ok(shape(f, &name)?[1]);
        }
    }
    Ok(1)
}

fn encoder_spec(f: &GgufFile) -> Result<(TrunkSpec, usize)> {
    let prefix = "a.blk.";
    let n = meta_u(f, "clip.audio.block_count")?;
    let hidden = meta_u(f, "clip.audio.embedding_length")?;
    let heads = meta_u(f, "clip.audio.attention.head_count")?;
    let window = meta_u(f, "clip.audio.window_size")?;
    let spec = TrunkSpec {
        prefix: prefix.into(),
        final_norm: "a.post_ln.weight".into(),
        names: MMPROJ_NAMES,
        hidden,
        heads,
        kv_heads: meta_u(f, "clip.audio.attention.head_count_kv")?,
        head_dim: hidden / heads,
        conv_kernel: conv_kernel(f, prefix, n)?,
        eps: meta_f(f, "clip.audio.attention.layer_norm_epsilon")? as f32,
        layers: mixers(f, prefix, n, Some(window)),
        ropes: vec![Rope {
            theta: ENCODER_THETA,
            partial: 1.0,
        }],
        causal: true,
        gate: false,
        sandwich: false,
    };
    // `a.conv1d.0.weight` is `[channels, 1, 3, 3]`.
    let channels = shape(f, "a.conv1d.0.weight")?[0];
    Ok((spec, channels))
}

fn depth_spec(f: &GgufFile) -> Result<TrunkSpec> {
    let prefix = "a.gen.code.blk.";
    let n = meta_u(f, "clip.gen.audio.block_count")?;
    let hidden = meta_u(f, "clip.gen.audio.embedding_length")?;
    let heads = meta_u(f, "clip.gen.audio.attention.head_count")?;
    Ok(TrunkSpec {
        prefix: prefix.into(),
        final_norm: "a.gen.code.output_norm.weight".into(),
        names: MMPROJ_NAMES,
        hidden,
        heads,
        kv_heads: meta_u(f, "clip.gen.audio.attention.head_count_kv")?,
        head_dim: shape(f, &format!("{prefix}0.attn_q_norm.weight"))?[0],
        conv_kernel: conv_kernel(f, prefix, n)?,
        eps: meta_f(f, "clip.gen.audio.attention.layer_norm_epsilon")? as f32,
        layers: mixers(f, prefix, n, None),
        ropes: vec![Rope {
            theta: GEN_THETA,
            partial: 1.0,
        }],
        causal: false,
        gate: has(f, &format!("{prefix}0.attn_gate.weight")),
        sandwich: has(f, &format!("{prefix}0.attn_post_norm.weight")),
    })
}

fn codec_spec(f: &GgufFile) -> Result<TrunkSpec> {
    let prefix = "a.gen.wav.tfm.blk.";
    let n = count_layers(f, prefix, "ln1.weight");
    let hidden = shape(f, "a.gen.wav.tfm.in_proj.weight")?[0];
    let attn = (0..n)
        .find(|i| has(f, &format!("{prefix}{i}.attn_q.weight")))
        .ok_or_else(|| anyhow!("codec decoder has no attention layer"))?;
    let head_dim = shape(f, &format!("{prefix}{attn}.attn_q_norm.weight"))?[0];
    let heads = shape(f, &format!("{prefix}{attn}.attn_q.weight"))?[0] / head_dim;
    let kv_heads = shape(f, &format!("{prefix}{attn}.attn_k.weight"))?[0] / head_dim;
    Ok(TrunkSpec {
        prefix: prefix.into(),
        final_norm: "a.gen.wav.tfm.output_norm.weight".into(),
        names: MMPROJ_NAMES,
        hidden,
        heads,
        kv_heads,
        head_dim,
        conv_kernel: conv_kernel(f, prefix, n)?,
        eps: 1e-5,
        layers: mixers(f, prefix, n, Some(CODEC_WINDOW)),
        ropes: vec![Rope {
            theta: GEN_THETA,
            partial: 1.0,
        }],
        causal: true,
        gate: has(f, &format!("{prefix}{attn}.attn_gate.weight")),
        sandwich: has(f, &format!("{prefix}0.attn_post_norm.weight")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_factor_prefix() {
        let mut f = vec![1.0f32; 32];
        f.extend(std::iter::repeat_n(1e30, 32));
        assert_eq!(partial_from_factors(&f).unwrap(), 0.5);
        assert_eq!(partial_from_factors(&[1.0; 8]).unwrap(), 1.0);
        assert!(partial_from_factors(&[1.0, 2.0]).is_err());
    }
}
