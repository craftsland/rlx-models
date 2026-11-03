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

//! A parsed `.rcc` model: zero-copy views into the blob.

use crate::Error;
use crate::blob::{Cursor, MAGIC, Q8, VERSION};

/// Hyperparameters stored in the `.rcc` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// PCM sample rate (Hz).
    pub sample_rate: usize,
    /// Mel bins.
    pub n_mels: usize,
    /// STFT size.
    pub n_fft: usize,
    /// STFT hop (samples).
    pub hop: usize,
    /// Model width.
    pub d_model: usize,
    /// Conformer blocks.
    pub n_layers: usize,
    /// Attention heads.
    pub n_heads: usize,
    /// Feed-forward width.
    pub ff_dim: usize,
    /// Depthwise conv kernel.
    pub conv_kernel: usize,
    /// Channels of the striding Conv2d subsampling.
    pub sub_channels: usize,
    /// CTC classes including blank.
    pub num_classes: usize,
    /// CTC blank id.
    pub blank_id: usize,
}

impl Config {
    /// `d_model / n_heads`.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
    /// Frequency bins after the first stride-2 conv.
    pub fn f1(&self) -> usize {
        half(self.n_mels)
    }
    /// Frequency bins after the second stride-2 conv.
    pub fn f2(&self) -> usize {
        half(self.f1())
    }
    /// Encoder frames for `mel_frames` input frames (two stride-2 convs).
    pub fn enc_frames(&self, mel_frames: usize) -> usize {
        half(half(mel_frames))
    }
}

/// Output length of a k=3, s=2, pad=1 conv.
pub(crate) fn half(n: usize) -> usize {
    (n + 2 - 3) / 2 + 1
}

/// One Conformer block's weights.
pub(crate) struct Layer<'a> {
    pub ff1: Ffn<'a>,
    pub att_ln: (&'a [f32], &'a [f32]),
    pub q: (Q8<'a>, &'a [f32]),
    pub k: (Q8<'a>, &'a [f32]),
    pub v: (Q8<'a>, &'a [f32]),
    pub o: (Q8<'a>, &'a [f32]),
    pub pos: Q8<'a>,
    pub pos_u: &'a [f32],
    pub pos_v: &'a [f32],
    pub conv_ln: (&'a [f32], &'a [f32]),
    pub pw1: (Q8<'a>, &'a [f32]),
    /// Depthwise weights `[d, k]` with BatchNorm folded in.
    pub dw: &'a [f32],
    pub dw_b: &'a [f32],
    pub pw2: (Q8<'a>, &'a [f32]),
    pub ff2: Ffn<'a>,
    pub out_ln: (&'a [f32], &'a [f32]),
}

pub(crate) struct Ffn<'a> {
    pub ln: (&'a [f32], &'a [f32]),
    pub l1: (Q8<'a>, &'a [f32]),
    pub l2: (Q8<'a>, &'a [f32]),
}

/// A model borrowed from an `.rcc` blob (typically memory-mapped flash).
pub struct Model<'a> {
    bytes: &'a [u8],
    cfg: Config,
    window: &'a [f32],
    band_start: &'a [u32],
    band_off: &'a [u32],
    band_w: &'a [f32],
    pub(crate) sub0_w: &'a [f32],
    pub(crate) sub0_b: &'a [f32],
    pub(crate) sub1: (Q8<'a>, &'a [f32]),
    pub(crate) pre_out: (Q8<'a>, &'a [f32]),
    layers_at: usize,
    layer_stride: usize,
    pub(crate) ctc: (Q8<'a>, &'a [f32]),
    piece_off: &'a [u32],
    piece_bytes: &'a [u8],
}

impl<'a> Model<'a> {
    /// Parse and shape-check a blob. `bytes` must be 4-byte aligned.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let mut c = Cursor::new(bytes)?;
        if c.u32()? != MAGIC {
            return Err(Error::BadMagic);
        }
        if c.u32()? != VERSION {
            return Err(Error::Version);
        }
        let mut h = [0usize; 12];
        for v in &mut h {
            *v = c.usize()?;
        }
        let cfg = Config {
            sample_rate: h[0],
            n_mels: h[1],
            n_fft: h[2],
            hop: h[3],
            d_model: h[4],
            n_layers: h[5],
            n_heads: h[6],
            ff_dim: h[7],
            conv_kernel: h[8],
            sub_channels: h[9],
            num_classes: h[10],
            blank_id: h[11],
        };
        if cfg.n_heads == 0
            || !cfg.d_model.is_multiple_of(cfg.n_heads)
            || !cfg.n_fft.is_power_of_two()
            || cfg.n_fft > rlx_mcu::mel::MAX_FFT
            || cfg.conv_kernel.is_multiple_of(2)
            || cfg.blank_id >= cfg.num_classes
        {
            return Err(Error::Unsupported);
        }
        let (d, ch) = (cfg.d_model, cfg.sub_channels);

        let window = c.f32s_n(cfg.n_fft)?;
        let band_start = c.u32s()?;
        let band_off = c.u32s()?;
        let band_w = c.f32s()?;
        if band_start.len() != cfg.n_mels
            || band_off.len() != cfg.n_mels + 1
            || band_off[cfg.n_mels] as usize != band_w.len()
        {
            return Err(Error::Shape);
        }
        for m in 0..cfg.n_mels {
            let len = band_off[m + 1]
                .checked_sub(band_off[m])
                .ok_or(Error::Shape)?;
            if (band_start[m] + len) as usize > cfg.n_fft / 2 + 1 {
                return Err(Error::Shape);
            }
        }

        let sub0_w = c.f32s_n(ch * 9)?;
        let sub0_b = c.f32s_n(ch)?;
        let sub1 = (c.q8(ch, 9 * ch)?, c.f32s_n(ch)?);
        let pre_out = (c.q8(d, ch * cfg.f2())?, c.f32s_n(d)?);

        let layers_at = c.pos();
        Self::parse_layer(&mut c, &cfg)?;
        let layer_stride = c.pos() - layers_at;
        let mut c = Cursor::at(bytes, layers_at + layer_stride * cfg.n_layers);
        if cfg.n_layers == 0 || c.pos() > bytes.len() {
            return Err(Error::Truncated);
        }
        let ctc = (c.q8(cfg.num_classes, d)?, c.f32s_n(cfg.num_classes)?);
        let piece_off = c.u32s()?;
        let piece_bytes = c.bytes()?;
        if piece_off
            .last()
            .is_some_and(|&e| e as usize > piece_bytes.len())
        {
            return Err(Error::Shape);
        }

        let m = Self {
            bytes,
            cfg,
            window,
            band_start,
            band_off,
            band_w,
            sub0_w,
            sub0_b,
            sub1,
            pre_out,
            layers_at,
            layer_stride,
            ctc,
            piece_off,
            piece_bytes,
        };
        // Every block has the same layout, so checking each parses also
        // checks the stride assumption.
        for i in 0..cfg.n_layers {
            m.try_layer(i)?;
        }
        Ok(m)
    }

    fn parse_layer(c: &mut Cursor<'a>, cfg: &Config) -> Result<Layer<'a>, Error> {
        let (d, ff, k) = (cfg.d_model, cfg.ff_dim, cfg.conv_kernel);
        let hd = cfg.head_dim();
        let ln = |c: &mut Cursor<'a>| -> Result<(&'a [f32], &'a [f32]), Error> {
            Ok((c.f32s_n(d)?, c.f32s_n(d)?))
        };
        let ffn = |c: &mut Cursor<'a>, ln: (&'a [f32], &'a [f32])| -> Result<Ffn<'a>, Error> {
            Ok(Ffn {
                ln,
                l1: (c.q8(ff, d)?, c.f32s_n(ff)?),
                l2: (c.q8(d, ff)?, c.f32s_n(d)?),
            })
        };
        let l = ln(c)?;
        let ff1 = ffn(c, l)?;
        let att_ln = ln(c)?;
        let q = (c.q8(d, d)?, c.f32s_n(d)?);
        let kk = (c.q8(d, d)?, c.f32s_n(d)?);
        let v = (c.q8(d, d)?, c.f32s_n(d)?);
        let o = (c.q8(d, d)?, c.f32s_n(d)?);
        let pos = c.q8(d, d)?;
        let pos_u = c.f32s_n(cfg.n_heads * hd)?;
        let pos_v = c.f32s_n(cfg.n_heads * hd)?;
        let conv_ln = ln(c)?;
        let pw1 = (c.q8(2 * d, d)?, c.f32s_n(2 * d)?);
        let dw = c.f32s_n(d * k)?;
        let dw_b = c.f32s_n(d)?;
        let pw2 = (c.q8(d, d)?, c.f32s_n(d)?);
        let l = ln(c)?;
        let ff2 = ffn(c, l)?;
        let out_ln = ln(c)?;
        Ok(Layer {
            ff1,
            att_ln,
            q,
            k: kk,
            v,
            o,
            pos,
            pos_u,
            pos_v,
            conv_ln,
            pw1,
            dw,
            dw_b,
            pw2,
            ff2,
            out_ln,
        })
    }

    fn try_layer(&self, i: usize) -> Result<Layer<'a>, Error> {
        let mut c = Cursor::at(self.bytes, self.layers_at + i * self.layer_stride);
        Self::parse_layer(&mut c, &self.cfg)
    }

    pub(crate) fn layer(&self, i: usize) -> Layer<'a> {
        // Checked for every `i` in `parse`.
        self.try_layer(i).unwrap_or_else(|_| unreachable!())
    }

    /// Header hyperparameters.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// STFT window, centered in `n_fft`.
    pub fn window(&self) -> &'a [f32] {
        self.window
    }

    /// Mel filter `m` as `(first FFT bin, weights)`.
    #[inline]
    pub fn mel_band(&self, m: usize) -> (usize, &'a [f32]) {
        let (a, b) = (self.band_off[m] as usize, self.band_off[m + 1] as usize);
        (self.band_start[m] as usize, &self.band_w[a..b])
    }

    /// SentencePiece piece for token `id` (`▁` marks a word start).
    pub fn piece(&self, id: usize) -> Option<&'a str> {
        let a = *self.piece_off.get(id)? as usize;
        let b = *self.piece_off.get(id + 1)? as usize;
        core::str::from_utf8(self.piece_bytes.get(a..b)?).ok()
    }

    /// Detokenize `ids` into `out` (UTF-8). Returns the bytes written; stops
    /// early, on a character boundary, if `out` is full.
    pub fn decode(&self, ids: &[u16], out: &mut [u8]) -> usize {
        let mut n = 0;
        let mut push = |s: &[u8], n: &mut usize| -> bool {
            if *n + s.len() > out.len() {
                return false;
            }
            out[*n..*n + s.len()].copy_from_slice(s);
            *n += s.len();
            true
        };
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
                if !push(s, &mut n) {
                    return n;
                }
            }
        }
        n
    }
}
