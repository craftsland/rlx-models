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

//! NVIDIA NeMo **Citrinet** CTC ASR (`stt_en_citrinet_256_ls`) without an
//! operating system.
//!
//! A stack of Jasper blocks over log-mel features: each repeat is a
//! depthwise conv, a pointwise conv and BatchNorm (folded into the pointwise
//! conv), ReLU between repeats; each block ends in squeeze-and-excitation
//! (global average over time), adds a 1×1 residual and applies ReLU. Three
//! blocks stride by 2. No attention, so memory grows only linearly with the
//! utterance.
//!
//! Pointwise and residual convs (≈95 % of the MACs) are int8 matrices; the
//! depthwise convs and the SE MLP stay f32 — small, and the target chips
//! have FPUs. Built on [`rlx_mcu`]: the same in-place blob format,
//! kernels and NeMo log-mel frontend as `rlx-conformer-ctc-core`.

#![cfg_attr(not(feature = "std"), no_std)]

pub use rlx_mcu::mel::{Frontend, MelSource};
pub use rlx_mcu::{Error, Pcm, next_cut, split};

use rlx_mcu::blob::{Cursor, Mat};
use rlx_mcu::kernels::{carve, linear, sigmoid};

/// `"RCT1"` as a little-endian word.
pub const MAGIC: u32 = u32::from_le_bytes(*b"RCT1");
/// Bumped whenever the record order or a record layout changes.
/// 2: depthwise weights tap-major `[kernel, cin]`; a file may hold a range
/// of blocks (a pipeline stage, see [`run_stage`]).
pub const VERSION: u32 = 2;
/// Most blocks a model may have.
pub const MAX_BLOCKS: usize = 64;
/// Most depthwise + pointwise repeats per block.
pub const MAX_REPEAT: usize = 8;
/// Widest depthwise kernel.
pub const MAX_KERNEL: usize = 128;
/// Words of block metadata per block.
const META: usize = 9;

/// Header hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// PCM sample rate (Hz).
    pub sample_rate: usize,
    /// Mel bins.
    pub n_mels: usize,
    /// STFT size.
    pub n_fft: usize,
    /// STFT hop.
    pub hop: usize,
    /// Jasper blocks in this file.
    pub n_blocks: usize,
    /// CTC classes including blank.
    pub num_classes: usize,
    /// CTC blank id.
    pub blank_id: usize,
    /// Index of this file's first block in the whole model (0: the file
    /// starts from PCM).
    pub first_block: usize,
    /// Blocks in the whole model.
    pub total_blocks: usize,
    /// Holds the CTC head (the file ends in token ids).
    pub has_head: bool,
}

/// One Jasper block's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    /// Input channels.
    pub c_in: usize,
    /// Output channels.
    pub c_out: usize,
    /// Depthwise + pointwise repeats.
    pub repeat: usize,
    /// Depthwise kernel (odd).
    pub kernel: usize,
    /// Time stride.
    pub stride: usize,
    /// Only the last repeat strides (NeMo `stride_last`).
    pub stride_last: bool,
    /// Has a 1×1 residual path (strided with the block, `stride_add`).
    pub residual: bool,
    /// Has squeeze-and-excitation.
    pub se: bool,
    /// SE hidden width.
    pub se_hidden: usize,
}

impl Block {
    fn strides(&self, r: usize) -> bool {
        self.stride > 1 && (!self.stride_last || r + 1 == self.repeat)
    }
    /// Output frames for `t` input frames (same padding).
    pub fn out_len(&self, t: usize) -> usize {
        let mut t = t;
        for r in 0..self.repeat {
            if self.strides(r) {
                t = (t - 1) / self.stride + 1;
            }
        }
        t
    }
}

/// A model borrowed from an `.rct` blob.
pub struct Model<'a> {
    bytes: &'a [u8],
    cfg: Config,
    window: &'a [f32],
    band_start: &'a [u32],
    band_off: &'a [u32],
    band_w: &'a [f32],
    meta: &'a [u32],
    block_at: [usize; MAX_BLOCKS],
    decoder: Option<(Mat<'a>, &'a [f32])>,
    piece_off: &'a [u32],
    piece_bytes: &'a [u8],
}

struct Repeat<'a> {
    dw: &'a [f32],
    pw: Mat<'a>,
    pw_b: &'a [f32],
}

impl<'a> Model<'a> {
    /// Parse and shape-check a blob. `bytes` must be 4-byte aligned.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let mut c = Cursor::with_header(bytes, MAGIC, VERSION)?;
        let mut h = [0usize; 10];
        for v in &mut h {
            *v = c.usize()?;
        }
        let cfg = Config {
            sample_rate: h[0],
            n_mels: h[1],
            n_fft: h[2],
            hop: h[3],
            n_blocks: h[4],
            num_classes: h[5],
            blank_id: h[6],
            first_block: h[7],
            total_blocks: h[8],
            has_head: h[9] != 0,
        };
        if cfg.n_blocks == 0
            || cfg.n_blocks > MAX_BLOCKS
            || cfg.first_block + cfg.n_blocks > cfg.total_blocks
            || cfg.has_head != (cfg.first_block + cfg.n_blocks == cfg.total_blocks)
            || !cfg.n_fft.is_power_of_two()
            || cfg.n_fft > rlx_mcu::mel::MAX_FFT
            || cfg.blank_id >= cfg.num_classes
        {
            return Err(Error::Unsupported);
        }
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
        let meta = c.u32s()?;
        if meta.len() != META * cfg.n_blocks {
            return Err(Error::Shape);
        }
        let mut block_at = [0usize; MAX_BLOCKS];
        let mut prev_out = if cfg.first_block == 0 {
            cfg.n_mels
        } else {
            Self::block_meta(meta, 0).c_in
        };
        for (i, at) in block_at.iter_mut().enumerate().take(cfg.n_blocks) {
            let b = Self::block_meta(meta, i);
            if b.c_in != prev_out
                || b.kernel % 2 == 0
                || b.kernel > MAX_KERNEL
                || b.repeat == 0
                || b.repeat > MAX_REPEAT
                || b.stride == 0
            {
                return Err(Error::Shape);
            }
            *at = c.pos();
            Self::skip_block(&mut c, &b)?;
            prev_out = b.c_out;
        }
        let decoder = if cfg.has_head {
            Some((
                c.mat(cfg.num_classes, prev_out)?,
                c.f32s_n(cfg.num_classes)?,
            ))
        } else {
            None
        };
        let piece_off = c.u32s()?;
        let piece_bytes = c.bytes()?;
        if piece_off
            .last()
            .is_some_and(|&e| e as usize > piece_bytes.len())
        {
            return Err(Error::Shape);
        }
        Ok(Self {
            bytes,
            cfg,
            window,
            band_start,
            band_off,
            band_w,
            meta,
            block_at,
            decoder,
            piece_off,
            piece_bytes,
        })
    }

    fn block_meta(meta: &[u32], i: usize) -> Block {
        let m = &meta[i * META..(i + 1) * META];
        Block {
            c_in: m[0] as usize,
            c_out: m[1] as usize,
            repeat: m[2] as usize,
            kernel: m[3] as usize,
            stride: m[4] as usize,
            stride_last: m[5] != 0,
            residual: m[6] != 0,
            se: m[7] != 0,
            se_hidden: m[8] as usize,
        }
    }

    fn skip_block(c: &mut Cursor<'a>, b: &Block) -> Result<(), Error> {
        for r in 0..b.repeat {
            let cin = if r == 0 { b.c_in } else { b.c_out };
            c.f32s_n(cin * b.kernel)?;
            c.mat(b.c_out, cin)?;
            c.f32s_n(b.c_out)?;
        }
        if b.se {
            c.f32s_n(b.se_hidden * b.c_out)?;
            c.f32s_n(b.c_out * b.se_hidden)?;
        }
        if b.residual {
            c.mat(b.c_out, b.c_in)?;
            c.f32s_n(b.c_out)?;
        }
        Ok(())
    }

    /// Header hyperparameters.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Shape of block `i`.
    pub fn block(&self, i: usize) -> Block {
        Self::block_meta(self.meta, i)
    }

    /// Channels of the encoder output (the CTC head's input); for a stage
    /// without the head, of its output.
    pub fn decoder_width(&self) -> usize {
        self.block(self.cfg.n_blocks - 1).c_out
    }

    /// Channels a stage expects as input (`n_mels` for the first).
    pub fn input_width(&self) -> usize {
        self.block(0).c_in
    }

    /// Output frames of this file's blocks for `frames` input frames.
    pub fn enc_frames(&self, frames: usize) -> usize {
        (0..self.cfg.n_blocks).fold(frames, |t, i| self.block(i).out_len(t))
    }

    /// WordPiece piece of token `id`.
    pub fn piece(&self, id: usize) -> Option<&'a str> {
        let a = *self.piece_off.get(id)? as usize;
        let b = *self.piece_off.get(id + 1)? as usize;
        core::str::from_utf8(self.piece_bytes.get(a..b)?).ok()
    }

    /// Detokenize WordPiece ids into `out`: `##x` continues the word, an
    /// apostrophe token joins its neighbours (`wife ' s` → `wife's`),
    /// anything else starts a word; `[…]` specials are dropped. Returns the
    /// bytes written (stops early if `out` is full).
    pub fn decode(&self, ids: &[u16], out: &mut [u8]) -> usize {
        let mut n = 0;
        let mut glue = false;
        for &id in ids {
            let Some(p) = self.piece(id as usize) else {
                continue;
            };
            if p.is_empty() || (p.starts_with('[') && p.ends_with(']')) {
                continue;
            }
            let apostrophe = p == "'";
            let (text, space) = match p.strip_prefix("##") {
                Some(rest) => (rest, false),
                None => (p, n > 0 && !glue && !apostrophe),
            };
            glue = apostrophe;
            let need = text.len() + usize::from(space);
            if n + need > out.len() {
                return n;
            }
            if space {
                out[n] = b' ';
                n += 1;
            }
            out[n..n + text.len()].copy_from_slice(text.as_bytes());
            n += text.len();
        }
        n
    }
}

impl MelSource for Model<'_> {
    fn hop(&self) -> usize {
        self.cfg.hop
    }
    fn n_mels(&self) -> usize {
        self.cfg.n_mels
    }
    fn window(&self) -> &[f32] {
        self.window
    }
    fn mel_band(&self, m: usize) -> (usize, &[f32]) {
        let (a, b) = (self.band_off[m] as usize, self.band_off[m + 1] as usize);
        (self.band_start[m] as usize, &self.band_w[a..b])
    }
}

/// Element counts of the two scratch arenas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchLen {
    /// `f32` elements.
    pub f32s: usize,
    /// `i8` elements.
    pub i8s: usize,
}

impl ScratchLen {
    /// Total bytes.
    pub fn bytes(&self) -> usize {
        self.f32s * 4 + self.i8s
    }
}

/// Outputs computed per step: two rows share each weight load in
/// [`linear`]'s 2-row tile.
const STEP: usize = 2;

struct Layout {
    /// The work buffer: block activations, convolved in place.
    work: usize,
    /// Residual-path output of the current block.
    res: usize,
    /// Ring of the conv's input rows (`kernel + stride` rows).
    ring: usize,
    /// Depthwise outputs of one step (`STEP` rows).
    dw: usize,
    fft: usize,
    se: usize,
    row: usize,
    xq: usize,
}

impl Layout {
    fn f32s(&self) -> usize {
        self.work + self.res + self.ring + self.dw + self.fft + self.se + self.row
    }
}

/// Where a repeat's input must start in the work buffer so that writing its
/// outputs from offset 0 never overwrites an input row before that row has
/// been copied into the ring. 0 unless the repeat widens the rows.
fn input_shift(t_in: usize, cin: usize, cout: usize, stride: usize, kernel: usize) -> usize {
    // Rows that do not widen never overtake their input.
    if cout <= cin {
        return 0;
    }
    let t_out = (t_in - 1) / stride + 1;
    let p = kernel / 2;
    let mut shift = 0;
    let mut o = 0;
    while o < t_out {
        let m = STEP.min(t_out - o);
        // First input row not yet in the ring after this step.
        let next = (o + m - 1) * stride + p + 1;
        if next < t_in {
            shift = shift.max(((o + m) * cout).saturating_sub(next * cin));
        }
        o += m;
    }
    shift
}

/// Scratch for this file's blocks over `frames` input frames.
fn layout(model: &Model<'_>, frames: usize) -> Layout {
    let cfg = model.config();
    let c0 = model.input_width();
    let (mut t, mut work, mut res, mut ring, mut se, mut xq, mut wide) =
        (frames, frames * c0, 0, 0, 0, 0, c0);
    for i in 0..cfg.n_blocks {
        let b = model.block(i);
        for r in 0..b.repeat {
            let cin = if r == 0 { b.c_in } else { b.c_out };
            let s = if b.strides(r) { b.stride } else { 1 };
            let t_out = (t - 1) / s + 1;
            let shift = input_shift(t, cin, b.c_out, s, b.kernel);
            work = work.max(shift + t * cin).max(t_out * b.c_out);
            ring = ring.max((b.kernel + s) * cin);
            t = t_out;
        }
        if b.residual {
            res = res.max(t * b.c_out);
        }
        se = se.max(b.c_out + b.se_hidden);
        xq = xq.max(b.c_in.max(b.c_out));
        wide = wide.max(b.c_in.max(b.c_out));
    }
    let first = cfg.first_block == 0;
    Layout {
        work,
        res,
        ring,
        dw: STEP * wide,
        fft: if first { 2 * cfg.n_fft } else { 0 },
        se,
        row: if cfg.has_head { cfg.num_classes } else { 0 },
        xq: 2 * xq,
    }
}

/// Input frames of a stage: log-mel frames of `n` samples for the first,
/// else `n` itself (activation rows from the previous stage).
fn input_frames(model: &Model<'_>, n: usize) -> usize {
    if model.config().first_block == 0 {
        Frontend::frames(model, n)
    } else {
        n
    }
}

/// Scratch needed to transcribe `n` samples. Grows linearly with `n`: the
/// work buffer holds one block's activations, convolved in place, and the
/// residual buffer one block's output.
///
/// For a later pipeline stage `n` is the number of activation rows it
/// receives, not samples.
pub fn scratch_len(model: &Model<'_>, n: usize) -> ScratchLen {
    let l = layout(model, input_frames(model, n));
    ScratchLen {
        f32s: l.f32s(),
        i8s: l.xq,
    }
}

/// The longest utterance (samples) whose scratch fits `f32s` + `i8s`
/// elements. Scratch grows with the utterance; cut longer audio with
/// [`next_cut`].
pub fn max_samples(model: &Model<'_>, f32s: usize, i8s: usize) -> usize {
    let fits = |n: usize| {
        let s = scratch_len(model, n);
        s.f32s <= f32s && s.i8s <= i8s
    };
    let (mut lo, mut hi) = (0usize, 1usize << 26);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// After [`transcribe`] on `n` samples, the encoder output `[frames, c]` is
/// still at the start of the f32 scratch; this is that slice. Lets a target
/// without room for a copy check it.
#[doc(hidden)]
pub fn encoder_output<'s>(model: &Model<'_>, n: usize, fs: &'s [f32]) -> &'s [f32] {
    let t = model.enc_frames(input_frames(model, n));
    &fs[..t * model.decoder_width()]
}

/// Relative compute of each block of the whole model over an `n`-sample
/// utterance, into `costs[first_block..first_block + n_blocks]`: int8 MACs
/// plus depthwise f32 MACs weighted 4× (they cost about that much more per
/// MAC on a Cortex-M33). What [`rlx_mcu::split::balance`] needs to cut a
/// model into stages of equal time.
pub fn block_costs(model: &Model<'_>, n: usize, costs: &mut [u64]) {
    let cfg = model.config();
    let mut t = input_frames(model, n);
    for i in 0..cfg.n_blocks {
        let b = model.block(i);
        let mut cost = 0u64;
        for r in 0..b.repeat {
            let cin = if r == 0 { b.c_in } else { b.c_out };
            let t_out = if b.strides(r) {
                (t - 1) / b.stride + 1
            } else {
                t
            };
            cost += (t_out * cin * b.c_out) as u64 + 4 * (t_out * cin * b.kernel) as u64;
            t = t_out;
        }
        if b.residual {
            cost += (t * b.c_in * b.c_out) as u64;
        }
        if i + 1 == cfg.n_blocks && cfg.has_head {
            cost += (t * b.c_out * cfg.num_classes) as u64;
        }
        if let Some(c) = costs.get_mut(cfg.first_block + i) {
            *c = cost;
        }
    }
}

/// Where a pipeline stage's input goes before [`run_stage`]: the first
/// `rows * input_width` values of the f32 scratch.
pub fn stage_input<'s>(model: &Model<'_>, rows: usize, fs: &'s mut [f32]) -> &'s mut [f32] {
    &mut fs[..rows * model.input_width()]
}

/// A stage's output after [`run_stage`] returned [`StageOutput::Rows`]: the
/// first `rows * decoder_width` values of the f32 scratch.
pub fn stage_output<'s>(model: &Model<'_>, rows: usize, fs: &'s [f32]) -> &'s [f32] {
    &fs[..rows * model.decoder_width()]
}

/// One depthwise output row: `y[ch] = Σ_j w[j, ch] · x_j[ch]` over the taps
/// `j0..j0 + at.len()`, whose input rows start at `ring[at[..]]`; `w` is
/// tap-major `[kernel, cin]`. Eight channels at a time stay in registers
/// across all taps. Every output sums its taps in ascending order starting
/// from 0.0, exactly as a tap-by-tap accumulation into `y` would.
fn depthwise_row(w: &[f32], ring: &[f32], at: &[usize], j0: usize, cin: usize, y: &mut [f32]) {
    const B: usize = 8;
    let mut ch = 0;
    while ch + B <= cin {
        let mut acc = [0.0f32; B];
        for (j, &a) in at.iter().enumerate() {
            let wj: &[f32; B] = w[(j0 + j) * cin + ch..][..B]
                .try_into()
                .unwrap_or(&[0.0; B]);
            let xj: &[f32; B] = ring[a + ch..][..B].try_into().unwrap_or(&[0.0; B]);
            for b in 0..B {
                acc[b] += wj[b] * xj[b];
            }
        }
        y[ch..ch + B].copy_from_slice(&acc);
        ch += B;
    }
    for c in ch..cin {
        let mut acc = 0.0f32;
        for (j, &a) in at.iter().enumerate() {
            acc += w[(j0 + j) * cin + c] * ring[a + c];
        }
        y[c] = acc;
    }
}

/// One repeat in place: depthwise conv (same padding, `stride`), then the
/// pointwise matrix, optionally ReLU. The input `[t_in, cin]` starts at
/// `shift` in `work`; the output `[t_out, cout]` is written from offset 0.
/// Input rows go through `ring` as they enter the conv window, so outputs
/// may overwrite rows already consumed. Returns `t_out`.
#[allow(clippy::too_many_arguments)]
fn repeat_in_place(
    work: &mut [f32],
    t_in: usize,
    cin: usize,
    rp: &Repeat<'_>,
    cout: usize,
    kernel: usize,
    stride: usize,
    relu: bool,
    ring: &mut [f32],
    dw: &mut [f32],
    xq: &mut [i8],
) -> usize {
    let shift = input_shift(t_in, cin, cout, stride, kernel);
    if shift > 0 {
        work.copy_within(..t_in * cin, shift);
    }
    let p = kernel / 2;
    let slots = kernel + stride;
    let t_out = (t_in - 1) / stride + 1;
    let mut loaded = 0;
    let mut o = 0;
    while o < t_out {
        let m = STEP.min(t_out - o);
        let need = ((o + m - 1) * stride + p + 1).min(t_in);
        while loaded < need {
            let src = shift + loaded * cin;
            let slot = (loaded % slots) * cin;
            ring[slot..slot + cin].copy_from_slice(&work[src..src + cin]);
            loaded += 1;
        }
        for i in 0..m {
            let c = (o + i) * stride;
            // Taps inside the input, and where each one's row sits in the ring.
            let j0 = p.saturating_sub(c);
            let j1 = kernel.min(t_in + p - c);
            let mut at = [0usize; MAX_KERNEL];
            for (j, a) in at.iter_mut().enumerate().take(j1).skip(j0) {
                *a = ((c + j - p) % slots) * cin;
            }
            depthwise_row(
                rp.dw,
                ring,
                &at[j0..j1],
                j0,
                cin,
                &mut dw[i * cin..(i + 1) * cin],
            );
        }
        let y = &mut work[o * cout..(o + m) * cout];
        linear(&dw[..m * cin], m, &rp.pw, Some(rp.pw_b), y, xq);
        if relu {
            for v in y.iter_mut() {
                *v = v.max(0.0);
            }
        }
        o += m;
    }
    t_out
}

/// PCM → token ids (greedy CTC) with a whole model. Returns the number of
/// ids written. `Error::Unsupported` for a pipeline stage: use [`run_stage`].
pub fn transcribe<P: Pcm + ?Sized>(
    model: &Model<'_>,
    frontend: &Frontend,
    pcm: &P,
    fs: &mut [f32],
    is: &mut [i8],
    ids: &mut [u16],
) -> Result<usize, Error> {
    match run_stage(model, StageInput::Pcm(pcm, frontend), fs, is, ids)? {
        StageOutput::Ids(n) => Ok(n),
        StageOutput::Rows(_) => Err(Error::Unsupported),
    }
}

/// What a pipeline stage starts from.
pub enum StageInput<'a, P: ?Sized> {
    /// Audio, for the first stage (`first_block == 0`).
    Pcm(&'a P, &'a Frontend),
    /// This many activation rows, already in [`stage_input`], for a later
    /// stage.
    Rows(usize),
}

/// What a pipeline stage ends with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageOutput {
    /// Token ids written (the last stage).
    Ids(usize),
    /// Activation rows left in [`stage_output`] for the next stage.
    Rows(usize),
}

/// Run this file's blocks: from PCM (first stage) or received rows, to token
/// ids (last stage) or rows for the next stage. A whole model is a single
/// stage from PCM to ids. Scratch: [`scratch_len`] with the samples (first
/// stage) or the rows.
pub fn run_stage<P: Pcm + ?Sized>(
    model: &Model<'_>,
    input: StageInput<'_, P>,
    fs: &mut [f32],
    is: &mut [i8],
    ids: &mut [u16],
) -> Result<StageOutput, Error> {
    run(model, input, fs, is, ids, None)
}

/// [`transcribe`], also copying the encoder output (`[frames, c]`) into
/// `features`. For parity checks.
#[doc(hidden)]
pub fn transcribe_with_features<P: Pcm + ?Sized>(
    model: &Model<'_>,
    frontend: &Frontend,
    pcm: &P,
    fs: &mut [f32],
    is: &mut [i8],
    ids: &mut [u16],
    features: &mut [f32],
) -> Result<usize, Error> {
    match run(
        model,
        StageInput::Pcm(pcm, frontend),
        fs,
        is,
        ids,
        Some(features),
    )? {
        StageOutput::Ids(n) => Ok(n),
        StageOutput::Rows(_) => Err(Error::Unsupported),
    }
}

fn run<P: Pcm + ?Sized>(
    model: &Model<'_>,
    input: StageInput<'_, P>,
    fs: &mut [f32],
    is: &mut [i8],
    ids: &mut [u16],
    features: Option<&mut [f32]>,
) -> Result<StageOutput, Error> {
    let cfg = *model.config();
    let frames = match &input {
        StageInput::Pcm(pcm, _) if cfg.first_block == 0 => Frontend::frames(model, pcm.len()),
        StageInput::Rows(rows) if cfg.first_block > 0 => *rows,
        _ => return Err(Error::Unsupported),
    };
    if frames == 0 {
        return Err(Error::Shape);
    }
    let l = layout(model, frames);
    if fs.len() < l.f32s() || is.len() < l.xq {
        return Err(Error::ScratchTooSmall);
    }
    let mut f = &mut fs[..];
    let work = carve(&mut f, l.work);
    let res = carve(&mut f, l.res);
    let ring = carve(&mut f, l.ring);
    let dw = carve(&mut f, l.dw);
    let fft = carve(&mut f, l.fft);
    let se = carve(&mut f, l.se);
    let row = carve(&mut f, l.row);
    let xq = &mut is[..l.xq];

    let mut t = match input {
        StageInput::Pcm(pcm, frontend) => frontend.log_mel(model, pcm, work, fft),
        StageInput::Rows(rows) => rows,
    };

    for bi in 0..cfg.n_blocks {
        let blk = model.block(bi);
        let mut cur = Cursor::at(model.bytes, model.block_at[bi]);
        let mut reps: [Option<Repeat<'_>>; MAX_REPEAT] = Default::default();
        for (rep, slot) in reps.iter_mut().enumerate().take(blk.repeat) {
            let cin = if rep == 0 { blk.c_in } else { blk.c_out };
            *slot = Some(Repeat {
                dw: cur.f32s_n(cin * blk.kernel)?,
                pw: cur.mat(blk.c_out, cin)?,
                pw_b: cur.f32s_n(blk.c_out)?,
            });
        }
        let se_w = if blk.se {
            Some((
                cur.f32s_n(blk.se_hidden * blk.c_out)?,
                cur.f32s_n(blk.c_out * blk.se_hidden)?,
            ))
        } else {
            None
        };
        let co = blk.c_out;
        let t_out = blk.out_len(t);
        // The residual reads the block input, which the repeats overwrite:
        // compute it first. `stride_add` takes every `stride`-th input frame.
        if blk.residual {
            let rw = cur.mat(co, blk.c_in)?;
            let rb = cur.f32s_n(co)?;
            let mut o = 0;
            while o < t_out {
                let m = STEP.min(t_out - o);
                for i in 0..m {
                    let src = ((o + i) * blk.stride.max(1)).min(t - 1);
                    dw[i * blk.c_in..(i + 1) * blk.c_in]
                        .copy_from_slice(&work[src * blk.c_in..(src + 1) * blk.c_in]);
                }
                linear(
                    &dw[..m * blk.c_in],
                    m,
                    &rw,
                    Some(rb),
                    &mut res[o * co..(o + m) * co],
                    xq,
                );
                o += m;
            }
        }
        let mut tl = t;
        for (rep, rp) in reps.iter().take(blk.repeat).enumerate() {
            let rp = rp.as_ref().ok_or(Error::Shape)?;
            let cin = if rep == 0 { blk.c_in } else { blk.c_out };
            let s = if blk.strides(rep) { blk.stride } else { 1 };
            tl = repeat_in_place(
                work,
                tl,
                cin,
                rp,
                co,
                blk.kernel,
                s,
                rep + 1 < blk.repeat,
                ring,
                dw,
                xq,
            );
        }
        let b = &mut work[..tl * co];
        if let Some((fc0, fc2)) = se_w {
            let h = blk.se_hidden;
            let (mean, hid) = se.split_at_mut(co);
            mean.fill(0.0);
            for ti in 0..tl {
                for (m, &v) in mean.iter_mut().zip(&b[ti * co..(ti + 1) * co]) {
                    *m += v;
                }
            }
            let inv = 1.0 / tl as f32;
            for m in mean.iter_mut() {
                *m *= inv;
            }
            for (j, hv) in hid[..h].iter_mut().enumerate() {
                let mut acc = 0f32;
                for (ch, &m) in mean.iter().enumerate() {
                    acc += fc0[j * co + ch] * m;
                }
                *hv = acc.max(0.0);
            }
            for ch in 0..co {
                let mut acc = 0f32;
                for j in 0..h {
                    acc += fc2[ch * h + j] * hid[j];
                }
                mean[ch] = sigmoid(acc);
            }
            for ti in 0..tl {
                for (v, &g) in b[ti * co..(ti + 1) * co].iter_mut().zip(mean.iter()) {
                    *v *= g;
                }
            }
        }
        if blk.residual {
            for (v, &rv) in b.iter_mut().zip(res.iter()) {
                *v += rv;
            }
        }
        for v in b.iter_mut() {
            *v = v.max(0.0);
        }
        t = tl;
    }

    let a = &work[..];
    let co = model.decoder_width();
    if let Some(out) = features {
        let k = out.len().min(t * co);
        out[..k].copy_from_slice(&a[..k]);
    }
    let Some(decoder) = &model.decoder else {
        return Ok(StageOutput::Rows(t));
    };
    let nc = cfg.num_classes;
    let (mut n, mut prev) = (0, usize::MAX);
    for ti in 0..t {
        linear(
            &a[ti * co..(ti + 1) * co],
            1,
            &decoder.0,
            Some(decoder.1),
            &mut row[..nc],
            xq,
        );
        let mut best = 0;
        for k in 1..nc {
            if row[k] > row[best] {
                best = k;
            }
        }
        if best != prev && best != cfg.blank_id {
            if n == ids.len() {
                return Err(Error::OutputTooSmall);
            }
            ids[n] = best as u16;
            n += 1;
        }
        prev = best;
    }
    Ok(StageOutput::Ids(n))
}
