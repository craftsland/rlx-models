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

//! The `.rkr` model file and the pager that streams its layers.
//!
//! The file has two parts. The **resident** part (header, frontend, conv
//! embed, stack combiners, predictor, joiner, token pieces and a table of
//! layer blocks) is small and lives in RAM — [`Model::parse`] reads it in
//! place. The **layer blocks** (one per Zipformer layer, 1–16 M parameters
//! each for Kroko-L) stay in external storage and are read one at a time
//! into a page buffer through a [`Pager`], once per chunk.
//!
//! ```text
//! header   MAGIC VERSION resident_len, 14 words (sample_rate n_mels
//!          chunk_size chunk_shift embed_dim enc_out vocab blank unk context
//!          dec_hidden pos_dim pos_head_dim n_stacks), then 11 words per
//!          stack (dim layers down left heads qdim vdim kernel ff1 ff2 ff3)
//! fbank    f32s povey window · u32s band starts · u32s band offsets · f32s weights
//! embed    conv0 conv4 conv7 (f32s w, b) · dw 7×7 (f32s w, b) · mat pw1 + b
//!          · mat pw2 + b · mat out + b · f32s out_norm bias · f32s log_scale
//! stacks   per stack: f32s downsample weights · f32s out_combiner scale
//! output   f32s downsample_output · mat encoder_proj + b
//! decoder  mat embedding · f32s context conv · mat proj + b · mat joiner + b
//! table    u32s (offset, len) per layer · u32s+bytes token pieces
//! layers   one block per layer, see `Layer`
//! ```

use rlx_mcu::blob::{Cursor, Mat};
pub use rlx_mcu::{Error, Pager, SlicePager};

/// `"RKR1"` as a little-endian word.
pub const MAGIC: u32 = u32::from_le_bytes(*b"RKR1");
/// Bumped whenever the record order or a record layout changes.
pub const VERSION: u32 = 1;
/// Most Zipformer stacks.
pub const MAX_STACKS: usize = 8;
/// Fixed conv embed shapes (k2 `Conv2dSubsampling` with a ConvNeXt).
pub const EMBED_C0: usize = 8;
/// Channels after the second embed conv.
pub const EMBED_C4: usize = 32;
/// Channels after the third embed conv.
pub const EMBED_C7: usize = 128;
/// ConvNeXt hidden width.
pub const EMBED_HID: usize = 384;

/// One Zipformer stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stack {
    /// Channels.
    pub dim: usize,
    /// Layers.
    pub layers: usize,
    /// Time downsampling factor.
    pub down: usize,
    /// Left context frames kept in the caches.
    pub left: usize,
    /// Attention heads.
    pub heads: usize,
    /// Query/key width per head.
    pub qdim: usize,
    /// Value width per head.
    pub vdim: usize,
    /// Conv module kernel.
    pub kernel: usize,
    /// Feed-forward widths.
    pub ff: [usize; 3],
}

impl Stack {
    /// Non-linear attention hidden width (`3·dim/4`).
    pub fn nonlin(&self) -> usize {
        3 * self.dim / 4
    }
}

/// Header hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// PCM sample rate.
    pub sample_rate: usize,
    /// Fbank bins.
    pub n_mels: usize,
    /// Fbank frames per chunk (with the lookahead).
    pub chunk_size: usize,
    /// Fbank frames the chunk advances.
    pub chunk_shift: usize,
    /// Conv embed output width (= first stack's input).
    pub embed_dim: usize,
    /// Encoder output width (joiner input).
    pub enc_out: usize,
    /// Token vocabulary.
    pub vocab: usize,
    /// Blank token.
    pub blank: usize,
    /// Unknown token (never emitted).
    pub unk: usize,
    /// Predictor context.
    pub context: usize,
    /// Predictor width.
    pub dec_hidden: usize,
    /// Relative position encoding width.
    pub pos_dim: usize,
    /// Position query width per head.
    pub pos_head_dim: usize,
    /// Stacks.
    pub n_stacks: usize,
    /// Stack shapes.
    pub stacks: [Stack; MAX_STACKS],
}

impl Config {
    /// Zipformer input frames per chunk (after the 2× embed).
    pub fn frames(&self) -> usize {
        self.chunk_shift / 2
    }
    /// Layers over all stacks.
    pub fn layers(&self) -> usize {
        self.stacks[..self.n_stacks].iter().map(|s| s.layers).sum()
    }
    /// Widest stack.
    pub fn max_dim(&self) -> usize {
        self.stacks[..self.n_stacks]
            .iter()
            .map(|s| s.dim)
            .max()
            .unwrap_or(0)
    }
    /// Embed conv output frames per chunk (before the ConvNeXt).
    pub fn conv7_frames(&self) -> usize {
        ((self.chunk_size - 2) - 3) / 2 + 1 - 2
    }
    /// Frequency bins after the embed convs.
    pub fn conv7_freq(&self) -> usize {
        ((self.n_mels - 3) / 2 + 1 - 3) / 2 + 1
    }
}

/// Weights of one Zipformer layer, parsed from its page.
pub(crate) struct Layer<'a> {
    pub bypass: &'a [f32],
    pub bypass_mid: &'a [f32],
    pub norm_bias: &'a [f32],
    pub norm_log_scale: f32,
    pub attn_in: (Mat<'a>, &'a [f32]),
    pub linear_pos: &'a [f32],
    pub ff: [((Mat<'a>, &'a [f32]), (Mat<'a>, &'a [f32])); 3],
    pub nonlin_in: (Mat<'a>, &'a [f32]),
    pub nonlin_out: (Mat<'a>, &'a [f32]),
    pub sa: [((Mat<'a>, &'a [f32]), (Mat<'a>, &'a [f32])); 2],
    pub conv: [Conv<'a>; 2],
}

pub(crate) struct Conv<'a> {
    pub in_proj: (Mat<'a>, &'a [f32]),
    pub causal: (&'a [f32], &'a [f32]),
    pub chunk: (&'a [f32], &'a [f32]),
    pub scale_left: &'a [f32],
    pub scale_right: &'a [f32],
    pub out_proj: (Mat<'a>, &'a [f32]),
}

impl<'a> Layer<'a> {
    pub(crate) fn parse(
        page: &'a [u8],
        s: &Stack,
        pos_dim: usize,
        pos_head: usize,
    ) -> Result<Self, Error> {
        let mut c = Cursor::new(page)?;
        let (d, h, k) = (s.dim, s.heads, s.kernel);
        let lin = |c: &mut Cursor<'a>, o: usize, i: usize| -> Result<(Mat<'a>, &'a [f32]), Error> {
            Ok((c.mat(o, i)?, c.f32s_n(o)?))
        };
        let bypass = c.f32s_n(d)?;
        let bypass_mid = c.f32s_n(d)?;
        let norm_bias = c.f32s_n(d)?;
        let norm_log_scale = c.f32s_n(1)?[0];
        let attn_in = lin(&mut c, 2 * h * s.qdim + h * pos_head, d)?;
        let linear_pos = c.f32s_n(h * pos_head * pos_dim)?;
        let ff = |c: &mut Cursor<'a>, w: usize| -> Result<_, Error> {
            Ok((lin(c, w, d)?, lin(c, d, w)?))
        };
        let ff = [
            ff(&mut c, s.ff[0])?,
            ff(&mut c, s.ff[1])?,
            ff(&mut c, s.ff[2])?,
        ];
        let nonlin_in = lin(&mut c, 3 * s.nonlin(), d)?;
        let nonlin_out = lin(&mut c, d, s.nonlin())?;
        let hv = h * s.vdim;
        let sa = [
            (lin(&mut c, hv, d)?, lin(&mut c, d, hv)?),
            (lin(&mut c, hv, d)?, lin(&mut c, d, hv)?),
        ];
        let conv = |c: &mut Cursor<'a>| -> Result<Conv<'a>, Error> {
            Ok(Conv {
                in_proj: lin(c, 2 * d, d)?,
                causal: (c.f32s_n(d * (k / 2 + 1))?, c.f32s_n(d)?),
                chunk: (c.f32s_n(d * k)?, c.f32s_n(d)?),
                scale_left: c.f32s_n(d * k)?,
                scale_right: c.f32s_n(d * k)?,
                out_proj: lin(c, d, d)?,
            })
        };
        let conv = [conv(&mut c)?, conv(&mut c)?];
        Ok(Self {
            bypass,
            bypass_mid,
            norm_bias,
            norm_log_scale,
            attn_in,
            linear_pos,
            ff,
            nonlin_in,
            nonlin_out,
            sa,
            conv,
        })
    }
}

pub(crate) struct Embed<'a> {
    pub conv0: (&'a [f32], &'a [f32]),
    pub conv4: (&'a [f32], &'a [f32]),
    pub conv7: (&'a [f32], &'a [f32]),
    pub dw: (&'a [f32], &'a [f32]),
    pub pw1: (Mat<'a>, &'a [f32]),
    pub pw2: (Mat<'a>, &'a [f32]),
    pub out: (Mat<'a>, &'a [f32]),
    pub norm_bias: &'a [f32],
    pub norm_log_scale: f32,
}

/// The resident part of a model.
pub struct Model<'a> {
    cfg: Config,
    resident_len: usize,
    pub(crate) window: &'a [f32],
    pub(crate) band_start: &'a [u32],
    pub(crate) band_off: &'a [u32],
    pub(crate) band_w: &'a [f32],
    pub(crate) embed: Embed<'a>,
    pub(crate) down_w: [&'a [f32]; MAX_STACKS],
    pub(crate) combiner: [&'a [f32]; MAX_STACKS],
    pub(crate) down_out: &'a [f32],
    pub(crate) enc_proj: (Mat<'a>, &'a [f32]),
    pub(crate) dec_embed: Mat<'a>,
    pub(crate) dec_conv: &'a [f32],
    pub(crate) dec_proj: (Mat<'a>, &'a [f32]),
    pub(crate) joiner: (Mat<'a>, &'a [f32]),
    table: &'a [u32],
    piece_off: &'a [u32],
    piece_bytes: &'a [u8],
}

impl<'a> Model<'a> {
    /// Parse the resident part: the first [`Model::resident_len`] bytes of
    /// the file (more is fine). 4-byte aligned.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let mut c = Cursor::with_header(bytes, MAGIC, VERSION)?;
        let resident_len = c.usize()?;
        let mut h = [0usize; 14];
        for v in &mut h {
            *v = c.usize()?;
        }
        let n_stacks = h[13];
        if n_stacks == 0 || n_stacks > MAX_STACKS {
            return Err(Error::Unsupported);
        }
        let mut stacks = [Stack::default(); MAX_STACKS];
        for s in stacks.iter_mut().take(n_stacks) {
            let mut w = [0usize; 11];
            for v in &mut w {
                *v = c.usize()?;
            }
            *s = Stack {
                dim: w[0],
                layers: w[1],
                down: w[2].max(1),
                left: w[3],
                heads: w[4],
                qdim: w[5],
                vdim: w[6],
                kernel: w[7],
                ff: [w[8], w[9], w[10]],
            };
            if s.kernel % 2 == 0 || s.heads == 0 || s.qdim > 64 || s.vdim > 64 {
                return Err(Error::Unsupported);
            }
        }
        let cfg = Config {
            sample_rate: h[0],
            n_mels: h[1],
            chunk_size: h[2],
            chunk_shift: h[3],
            embed_dim: h[4],
            enc_out: h[5],
            vocab: h[6],
            blank: h[7],
            unk: h[8],
            context: h[9],
            dec_hidden: h[10],
            pos_dim: h[11],
            pos_head_dim: h[12],
            n_stacks,
            stacks,
        };
        if cfg.context != 2 || !cfg.chunk_shift.is_multiple_of(2) || !cfg.pos_dim.is_multiple_of(2)
        {
            return Err(Error::Unsupported);
        }
        let window = c.f32s_n(400)?;
        let band_start = c.u32s()?;
        let band_off = c.u32s()?;
        let band_w = c.f32s()?;
        if band_start.len() != cfg.n_mels || band_off.len() != cfg.n_mels + 1 {
            return Err(Error::Shape);
        }
        let f7 = cfg.conv7_freq();
        let embed = Embed {
            conv0: (c.f32s_n(EMBED_C0 * 9)?, c.f32s_n(EMBED_C0)?),
            conv4: (c.f32s_n(EMBED_C4 * EMBED_C0 * 9)?, c.f32s_n(EMBED_C4)?),
            conv7: (c.f32s_n(EMBED_C7 * EMBED_C4 * 9)?, c.f32s_n(EMBED_C7)?),
            dw: (c.f32s_n(EMBED_C7 * 49)?, c.f32s_n(EMBED_C7)?),
            pw1: (c.mat(EMBED_HID, EMBED_C7)?, c.f32s_n(EMBED_HID)?),
            pw2: (c.mat(EMBED_C7, EMBED_HID)?, c.f32s_n(EMBED_C7)?),
            out: (
                c.mat(cfg.embed_dim, EMBED_C7 * f7)?,
                c.f32s_n(cfg.embed_dim)?,
            ),
            norm_bias: c.f32s_n(cfg.embed_dim)?,
            norm_log_scale: c.f32s_n(1)?[0],
        };
        let mut down_w = [&[][..]; MAX_STACKS];
        let mut combiner = [&[][..]; MAX_STACKS];
        for i in 0..n_stacks {
            down_w[i] = c.f32s()?;
            combiner[i] = c.f32s()?;
            let s = &cfg.stacks[i];
            let want_down = if s.down > 1 { s.down } else { 0 };
            let want_comb = if s.down > 1 { s.dim } else { 0 };
            if down_w[i].len() != want_down || combiner[i].len() != want_comb {
                return Err(Error::Shape);
            }
        }
        let down_out = c.f32s()?;
        let enc_proj = (c.mat(cfg.enc_out, cfg.max_dim())?, c.f32s_n(cfg.enc_out)?);
        let dh = cfg.dec_hidden;
        let dec_embed = c.mat(cfg.vocab, dh)?;
        let dec_conv = c.f32s()?;
        let dec_proj = (c.mat(dh, dh)?, c.f32s_n(dh)?);
        let joiner = (c.mat(cfg.vocab, cfg.enc_out)?, c.f32s_n(cfg.vocab)?);
        let table = c.u32s()?;
        if table.len() != 2 * cfg.layers() || dec_conv.len() % (dh * cfg.context) != 0 {
            return Err(Error::Shape);
        }
        let piece_off = c.u32s()?;
        let piece_bytes = c.bytes()?;
        if c.pos() > resident_len || resident_len > bytes.len() {
            return Err(Error::Truncated);
        }
        Ok(Self {
            cfg,
            resident_len,
            window,
            band_start,
            band_off,
            band_w,
            embed,
            down_w,
            combiner,
            down_out,
            enc_proj,
            dec_embed,
            dec_conv,
            dec_proj,
            joiner,
            table,
            piece_off,
            piece_bytes,
        })
    }

    /// Header hyperparameters.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Bytes of the resident part (the prefix to keep in RAM).
    pub fn resident_len(&self) -> usize {
        self.resident_len
    }

    /// `(file offset, bytes)` of layer `i`'s block.
    pub fn layer_block(&self, i: usize) -> (usize, usize) {
        (self.table[2 * i] as usize, self.table[2 * i + 1] as usize)
    }

    /// Largest layer block: the page buffer size.
    pub fn page_len(&self) -> usize {
        (0..self.cfg.layers())
            .map(|i| self.layer_block(i).1)
            .max()
            .unwrap_or(0)
    }

    /// Bytes read from storage per chunk (every layer once).
    pub fn paged_bytes(&self) -> usize {
        (0..self.cfg.layers()).map(|i| self.layer_block(i).1).sum()
    }

    /// BPE piece of token `id` (`▁` marks a word start).
    pub fn piece(&self, id: usize) -> Option<&'a str> {
        let a = *self.piece_off.get(id)? as usize;
        let b = *self.piece_off.get(id + 1)? as usize;
        core::str::from_utf8(self.piece_bytes.get(a..b)?).ok()
    }

    /// Detokenize into `out`: `▁` → space (none at the start), `<…>`
    /// specials dropped. Returns the bytes written.
    pub fn decode(&self, ids: &[u16], out: &mut [u8]) -> usize {
        let mut n = 0;
        for &id in ids {
            let Some(p) = self.piece(id as usize) else {
                continue;
            };
            if p.starts_with('<') && p.ends_with('>') {
                continue;
            }
            for ch in p.chars() {
                let mut buf = [0u8; 4];
                let s: &[u8] = if ch == '\u{2581}' {
                    if n == 0 {
                        continue;
                    }
                    b" "
                } else {
                    ch.encode_utf8(&mut buf).as_bytes()
                };
                if n + s.len() > out.len() {
                    return n;
                }
                out[n..n + s.len()].copy_from_slice(s);
                n += s.len();
            }
        }
        n
    }
}
