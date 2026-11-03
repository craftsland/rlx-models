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

//! The `.rms` model file (Moonshine streaming): header and record order.
//!
//! ```text
//! header   MAGIC VERSION, then 18 words: frame_len d_enc enc_layers enc_heads
//!          enc_head_dim enc_ff d_dec dec_layers dec_heads dec_head_dim dec_ff
//!          vocab n_rot start eos max_pos has_proj rope_theta
//! f32s     log_k (1)
//! mat      embedder linear [d_enc, frame_len]
//! mat+f32s conv1 [2 d_enc, 5 d_enc]  (reordered [out][tap][in]) + bias
//! mat+f32s conv2 [d_enc, 10 d_enc] + bias
//! u32s     sliding windows (left, right) per encoder layer
//! per encoder layer: f32s ln1 (γ+1) · mat q k v [H·hd, d] · mat o [d, H·hd]
//!          · f32s ln2 (γ+1) · mat fc1 [ff, d] + f32s · mat fc2 [d, ff] + f32s
//! f32s     final_norm (γ+1)
//! mat      pos_emb [max_pos, d_enc]
//! mat      proj [d_dec, d_enc]         (only when has_proj)
//! mat      embed_tokens [vocab, d_dec]
//! per decoder layer: f32s ln_in · mat q k v o · f32s ln_post · mat cross q k v o
//!          · f32s ln_final · mat fc1 [2 ff, d] + f32s · mat fc2 [d, ff] + f32s
//! f32s     decoder norm
//! mat      lm head [vocab, d_dec]
//! u32s+bytes token pieces (SentencePiece, `▁` and `<0xNN>` byte fallback)
//! ```

use rlx_mcu::Error;
use rlx_mcu::blob::{Cursor, Mat};

/// `"RMS1"` as a little-endian word.
pub const MAGIC: u32 = u32::from_le_bytes(*b"RMS1");
/// Bumped whenever the record order or a record layout changes.
pub const VERSION: u32 = 1;
/// Causal conv kernel and stride in the frontend.
pub const CONV_K: usize = 5;
/// Stride of both frontend convs.
pub const CONV_STRIDE: usize = 2;

/// Hyperparameters from the `.rms` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Samples per frontend frame (80 = 5 ms).
    pub frame_len: usize,
    /// Encoder width.
    pub d_enc: usize,
    /// Encoder layers.
    pub enc_layers: usize,
    /// Encoder heads.
    pub enc_heads: usize,
    /// Encoder head width.
    pub enc_head_dim: usize,
    /// Encoder MLP width.
    pub enc_ff: usize,
    /// Decoder width.
    pub d_dec: usize,
    /// Decoder layers.
    pub dec_layers: usize,
    /// Decoder heads.
    pub dec_heads: usize,
    /// Decoder head width.
    pub dec_head_dim: usize,
    /// Decoder MLP width (fc1 is twice this: up and gate).
    pub dec_ff: usize,
    /// Vocabulary.
    pub vocab: usize,
    /// Rotary width per head (GPT-J adjacent pairs).
    pub n_rot: usize,
    /// Decoder start token.
    pub start: usize,
    /// End-of-sequence token.
    pub eos: usize,
    /// Learned positions available (max encoder frames, max tokens).
    pub max_pos: usize,
    /// An encoder → decoder projection exists.
    pub has_proj: bool,
    /// RoPE base.
    pub rope_theta: usize,
}

impl Config {
    /// Encoder attention width.
    pub fn enc_attn(&self) -> usize {
        self.enc_heads * self.enc_head_dim
    }
    /// Decoder attention width.
    pub fn dec_attn(&self) -> usize {
        self.dec_heads * self.dec_head_dim
    }
    /// `(frames, conv1 rows, encoder frames)` for `n` samples.
    pub fn lengths(&self, n: usize) -> (usize, usize, usize) {
        let f = n.div_ceil(self.frame_len).max(1);
        let t1 = (f - 1) / CONV_STRIDE + 1;
        let t = (t1 - 1) / CONV_STRIDE + 1;
        (f, t1, t)
    }
    /// Sequence length budget for `n` samples, start token included: 6.5 new
    /// tokens per second plus 8, the rate Moonshine's own runtime uses,
    /// capped by the learned positions.
    pub fn max_tokens(&self, n: usize) -> usize {
        let t = libm::ceil(n as f64 / 16_000.0 * 6.5) as usize + 8;
        (t + 1).min(self.max_pos)
    }
}

pub(crate) struct EncLayer<'a> {
    pub ln1: &'a [f32],
    pub q: Mat<'a>,
    pub k: Mat<'a>,
    pub v: Mat<'a>,
    pub o: Mat<'a>,
    pub ln2: &'a [f32],
    pub fc1: (Mat<'a>, &'a [f32]),
    pub fc2: (Mat<'a>, &'a [f32]),
}

pub(crate) struct Attn<'a> {
    pub q: Mat<'a>,
    pub k: Mat<'a>,
    pub v: Mat<'a>,
    pub o: Mat<'a>,
}

pub(crate) struct DecLayer<'a> {
    pub ln_in: &'a [f32],
    pub sa: Attn<'a>,
    pub ln_post: &'a [f32],
    pub ca: Attn<'a>,
    pub ln_final: &'a [f32],
    pub fc1: (Mat<'a>, &'a [f32]),
    pub fc2: (Mat<'a>, &'a [f32]),
}

/// A model borrowed from an `.rms` blob.
pub struct Model<'a> {
    bytes: &'a [u8],
    cfg: Config,
    pub(crate) log_k: f32,
    pub(crate) embed_lin: Mat<'a>,
    pub(crate) conv1: (Mat<'a>, &'a [f32]),
    pub(crate) conv2: (Mat<'a>, &'a [f32]),
    windows: &'a [u32],
    enc_at: usize,
    enc_stride: usize,
    pub(crate) final_norm: &'a [f32],
    pub(crate) pos_emb: Mat<'a>,
    pub(crate) proj: Option<Mat<'a>>,
    pub(crate) embed_tokens: Mat<'a>,
    dec_at: usize,
    dec_stride: usize,
    pub(crate) dec_norm: &'a [f32],
    pub(crate) lm_head: Mat<'a>,
    piece_off: &'a [u32],
    piece_bytes: &'a [u8],
}

impl<'a> Model<'a> {
    /// Parse and shape-check a blob. `bytes` must be 4-byte aligned.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let mut c = Cursor::with_header(bytes, MAGIC, VERSION)?;
        let mut h = [0usize; 18];
        for v in &mut h {
            *v = c.usize()?;
        }
        let cfg = Config {
            frame_len: h[0],
            d_enc: h[1],
            enc_layers: h[2],
            enc_heads: h[3],
            enc_head_dim: h[4],
            enc_ff: h[5],
            d_dec: h[6],
            dec_layers: h[7],
            dec_heads: h[8],
            dec_head_dim: h[9],
            dec_ff: h[10],
            vocab: h[11],
            n_rot: h[12],
            start: h[13],
            eos: h[14],
            max_pos: h[15],
            has_proj: h[16] != 0,
            rope_theta: h[17],
        };
        if cfg.frame_len == 0
            || cfg.enc_layers == 0
            || cfg.dec_layers == 0
            || cfg.n_rot > cfg.dec_head_dim
            || !cfg.n_rot.is_multiple_of(2)
            || cfg.start >= cfg.vocab
            || cfg.eos >= cfg.vocab
            || cfg.enc_head_dim > 128
            || cfg.dec_head_dim > 128
            || (!cfg.has_proj && cfg.d_enc != cfg.d_dec)
        {
            return Err(Error::Unsupported);
        }
        let (d, a) = (cfg.d_enc, cfg.enc_attn());
        let log_k = c.f32s_n(1)?[0];
        let embed_lin = c.mat(d, cfg.frame_len)?;
        let conv1 = (c.mat(2 * d, CONV_K * d)?, c.f32s_n(2 * d)?);
        let conv2 = (c.mat(d, CONV_K * 2 * d)?, c.f32s_n(d)?);
        let windows = c.u32s()?;
        if windows.len() != 2 * cfg.enc_layers {
            return Err(Error::Shape);
        }
        let enc_at = c.pos();
        Self::parse_enc(&mut c, &cfg)?;
        let enc_stride = c.pos() - enc_at;
        let mut c = Cursor::at(bytes, enc_at + enc_stride * cfg.enc_layers);
        let final_norm = c.f32s_n(d)?;
        let pos_emb = c.mat(cfg.max_pos, d)?;
        let proj = if cfg.has_proj {
            Some(c.mat(cfg.d_dec, d)?)
        } else {
            None
        };
        let embed_tokens = c.mat(cfg.vocab, cfg.d_dec)?;
        let dec_at = c.pos();
        Self::parse_dec(&mut c, &cfg)?;
        let dec_stride = c.pos() - dec_at;
        let mut c = Cursor::at(bytes, dec_at + dec_stride * cfg.dec_layers);
        let dec_norm = c.f32s_n(cfg.d_dec)?;
        let lm_head = c.mat(cfg.vocab, cfg.d_dec)?;
        let piece_off = c.u32s()?;
        let piece_bytes = c.bytes()?;
        if piece_off
            .last()
            .is_some_and(|&e| e as usize > piece_bytes.len())
        {
            return Err(Error::Shape);
        }
        let _ = a;
        let m = Self {
            bytes,
            cfg,
            log_k,
            embed_lin,
            conv1,
            conv2,
            windows,
            enc_at,
            enc_stride,
            final_norm,
            pos_emb,
            proj,
            embed_tokens,
            dec_at,
            dec_stride,
            dec_norm,
            lm_head,
            piece_off,
            piece_bytes,
        };
        for i in 0..cfg.enc_layers {
            Self::parse_enc(&mut Cursor::at(bytes, enc_at + i * enc_stride), &cfg)?;
        }
        for i in 0..cfg.dec_layers {
            Self::parse_dec(&mut Cursor::at(bytes, dec_at + i * dec_stride), &cfg)?;
        }
        Ok(m)
    }

    fn parse_enc(c: &mut Cursor<'a>, cfg: &Config) -> Result<EncLayer<'a>, Error> {
        let (d, a, ff) = (cfg.d_enc, cfg.enc_attn(), cfg.enc_ff);
        Ok(EncLayer {
            ln1: c.f32s_n(d)?,
            q: c.mat(a, d)?,
            k: c.mat(a, d)?,
            v: c.mat(a, d)?,
            o: c.mat(d, a)?,
            ln2: c.f32s_n(d)?,
            fc1: (c.mat(ff, d)?, c.f32s_n(ff)?),
            fc2: (c.mat(d, ff)?, c.f32s_n(d)?),
        })
    }

    fn parse_dec(c: &mut Cursor<'a>, cfg: &Config) -> Result<DecLayer<'a>, Error> {
        let (d, a, ff) = (cfg.d_dec, cfg.dec_attn(), cfg.dec_ff);
        let attn = |c: &mut Cursor<'a>| -> Result<Attn<'a>, Error> {
            Ok(Attn {
                q: c.mat(a, d)?,
                k: c.mat(a, d)?,
                v: c.mat(a, d)?,
                o: c.mat(d, a)?,
            })
        };
        Ok(DecLayer {
            ln_in: c.f32s_n(d)?,
            sa: attn(c)?,
            ln_post: c.f32s_n(d)?,
            ca: attn(c)?,
            ln_final: c.f32s_n(d)?,
            fc1: (c.mat(2 * ff, d)?, c.f32s_n(2 * ff)?),
            fc2: (c.mat(d, ff)?, c.f32s_n(d)?),
        })
    }

    pub(crate) fn enc_layer(&self, i: usize) -> EncLayer<'a> {
        Self::parse_enc(
            &mut Cursor::at(self.bytes, self.enc_at + i * self.enc_stride),
            &self.cfg,
        )
        .unwrap_or_else(|_| unreachable!("checked in parse"))
    }

    pub(crate) fn dec_layer(&self, i: usize) -> DecLayer<'a> {
        Self::parse_dec(
            &mut Cursor::at(self.bytes, self.dec_at + i * self.dec_stride),
            &self.cfg,
        )
        .unwrap_or_else(|_| unreachable!("checked in parse"))
    }

    /// `(left, right)` attention window of encoder layer `i`.
    pub fn window(&self, i: usize) -> (usize, usize) {
        (
            self.windows[2 * i] as usize,
            self.windows[2 * i + 1] as usize,
        )
    }

    /// Header hyperparameters.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Bytes of the whole model.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Never true for a parsed model.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// SentencePiece piece of token `id`.
    pub fn piece(&self, id: usize) -> Option<&'a str> {
        let a = *self.piece_off.get(id)? as usize;
        let b = *self.piece_off.get(id + 1)? as usize;
        core::str::from_utf8(self.piece_bytes.get(a..b)?).ok()
    }

    /// Detokenize into `out`: `▁` → space, `<0xNN>` → that byte, special
    /// tokens (`<…>`) dropped, one leading space stripped. Returns the bytes
    /// written (stops early if `out` is full).
    pub fn decode(&self, ids: &[u16], out: &mut [u8]) -> usize {
        let mut n = 0;
        let mut push = |b: u8, n: &mut usize| -> bool {
            if *n == out.len() {
                return false;
            }
            out[*n] = b;
            *n += 1;
            true
        };
        for &id in ids {
            let Some(p) = self.piece(id as usize) else {
                continue;
            };
            if let Some(hex) = p.strip_prefix("<0x").and_then(|r| r.strip_suffix('>'))
                && hex.len() == 2
                && let Ok(b) = u8::from_str_radix(hex, 16)
            {
                if !push(b, &mut n) {
                    return n;
                }
                continue;
            }
            if p.starts_with('<') && p.ends_with('>') {
                continue;
            }
            for ch in p.chars() {
                if ch == '\u{2581}' {
                    if n > 0 && !push(b' ', &mut n) {
                        return n;
                    }
                    continue;
                }
                let mut buf = [0u8; 4];
                for &b in ch.encode_utf8(&mut buf).as_bytes() {
                    if !push(b, &mut n) {
                        return n;
                    }
                }
            }
        }
        n
    }
}
