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

//! C API over [`rlx_voice_link::Reassembler`] and the int8 Conformer-CTC in
//! [`rlx_conformer_ctc_core`], so an iOS or macOS app runs the same receive
//! path and the same ASR as the Rust host and the MCU. The header is
//! `include/rlx_voice_link.h`; the Swift package in `swift/` wraps it.
//!
//! Every function tolerates a null handle (returns an error value or does
//! nothing). Handles are not thread-safe; use each from one thread.

use core::ffi::c_int;
use core::ptr;

use rlx_conformer_ctc_core::{Frontend, Model, scratch_len, transcribe};
use rlx_voice_link::{Event, Reassembler};

/// Utterance reassembler with its own buffer.
pub struct RvlReceiver {
    // Field order matters: `inner` borrows `buf` and is dropped first.
    inner: Reassembler<'static>,
    buf: *mut [i16],
}

impl Drop for RvlReceiver {
    fn drop(&mut self) {
        // SAFETY: `buf` came from `Box::into_raw` in `rvl_receiver_new` and
        // `inner`, its only borrower, is no longer used past this point.
        unsafe { drop(Box::from_raw(self.buf)) }
    }
}

/// A receiver keeping utterances up to `max_samples` (16 kHz). Free with
/// [`rvl_receiver_free`].
#[unsafe(no_mangle)]
pub extern "C" fn rvl_receiver_new(max_samples: usize) -> *mut RvlReceiver {
    let buf = Box::into_raw(vec![0i16; max_samples.max(2)].into_boxed_slice());
    // SAFETY: `buf` stays allocated until `RvlReceiver::drop`, which drops
    // the reassembler before freeing it.
    let inner = Reassembler::new(unsafe { &mut *buf });
    Box::into_raw(Box::new(RvlReceiver { inner, buf }))
}

/// Free a receiver (null is a no-op).
///
/// # Safety
/// `r` is null or came from [`rvl_receiver_new`] and is not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvl_receiver_free(r: *mut RvlReceiver) {
    if !r.is_null() {
        // SAFETY: per the contract above.
        drop(unsafe { Box::from_raw(r) });
    }
}

/// Feed one notification. Returns `1` when an utterance completed (read it
/// with [`rvl_receiver_utterance`]), `0` for audio appended, `-1` if the
/// packet was dropped or an argument is null.
///
/// # Safety
/// `r` is a live receiver; `data` points to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvl_receiver_push(
    r: *mut RvlReceiver,
    data: *const u8,
    len: usize,
) -> c_int {
    if r.is_null() || data.is_null() {
        return -1;
    }
    // SAFETY: per the contract above.
    let (r, packet) = unsafe { (&mut *r, core::slice::from_raw_parts(data, len)) };
    match r.inner.push(packet) {
        Event::Utterance(_) => 1,
        Event::Audio => 0,
        Event::Dropped => -1,
    }
}

/// The last completed utterance: writes its address to `*out` and returns
/// the sample count. Valid until the next push.
///
/// # Safety
/// `r` is a live receiver; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvl_receiver_utterance(
    r: *const RvlReceiver,
    out: *mut *const i16,
) -> usize {
    if r.is_null() || out.is_null() {
        return 0;
    }
    // SAFETY: per the contract above.
    let u = unsafe { (*r).inner.utterance() };
    unsafe { *out = u.as_ptr() };
    u.len()
}

/// Packets lost (filled with silence) so far.
///
/// # Safety
/// `r` is null or a live receiver.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvl_receiver_lost(r: *const RvlReceiver) -> u32 {
    if r.is_null() {
        return 0;
    }
    // SAFETY: per the contract above.
    unsafe { (*r).inner.lost() }
}

/// An int8 Conformer-CTC model (`.rcc`).
pub struct RvlAsr {
    words: Vec<u32>,
    len: usize,
    fs: Vec<f32>,
    is: Vec<i8>,
}

impl RvlAsr {
    fn bytes(&self) -> &[u8] {
        // SAFETY: u32 storage viewed as bytes, `len <= 4 * words.len()`.
        unsafe { core::slice::from_raw_parts(self.words.as_ptr().cast(), self.len) }
    }
}

/// Load an `.rcc` from memory (copied, so the caller may free `data`).
/// Returns null if it is not a valid model.
///
/// # Safety
/// `data` points to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvl_asr_open(data: *const u8, len: usize) -> *mut RvlAsr {
    if data.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: per the contract above.
    let src = unsafe { core::slice::from_raw_parts(data, len) };
    let mut words = vec![0u32; len.div_ceil(4)];
    for (i, c) in src.chunks(4).enumerate() {
        let mut b = [0u8; 4];
        b[..c.len()].copy_from_slice(c);
        words[i] = u32::from_le_bytes(b);
    }
    let asr = RvlAsr {
        words,
        len,
        fs: Vec::new(),
        is: Vec::new(),
    };
    if Model::parse(asr.bytes()).is_err() {
        return ptr::null_mut();
    }
    Box::into_raw(Box::new(asr))
}

/// Free a model (null is a no-op).
///
/// # Safety
/// `a` is null or came from [`rvl_asr_open`] and is not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvl_asr_free(a: *mut RvlAsr) {
    if !a.is_null() {
        // SAFETY: per the contract above.
        drop(unsafe { Box::from_raw(a) });
    }
}

/// Transcribe 16 kHz PCM into UTF-8 at `out` (not NUL-terminated; at most
/// `cap` bytes, cut on a character boundary). Returns the byte count, or a
/// negative value on error.
///
/// # Safety
/// `a` is a live model; `pcm` points to `n` samples; `out` to `cap`
/// writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvl_asr_transcribe(
    a: *mut RvlAsr,
    pcm: *const i16,
    n: usize,
    out: *mut u8,
    cap: usize,
) -> isize {
    if a.is_null() || out.is_null() || (pcm.is_null() && n > 0) {
        return -1;
    }
    if n == 0 {
        return 0;
    }
    // SAFETY: per the contract above.
    let (a, pcm, out) = unsafe {
        (
            &mut *a,
            core::slice::from_raw_parts(pcm, n),
            core::slice::from_raw_parts_mut(out, cap),
        )
    };
    let mut fs = core::mem::take(&mut a.fs);
    let mut is = core::mem::take(&mut a.is);
    let r = (|| {
        let model = Model::parse(a.bytes()).map_err(|_| -2isize)?;
        let need = scratch_len(&model, n);
        // Scratch is kept between calls and only grows.
        if fs.len() < need.f32s {
            fs.resize(need.f32s, 0.0);
        }
        if is.len() < need.i8s {
            is.resize(need.i8s, 0);
        }
        let fe = Frontend::new(model.config().n_fft);
        let mut ids = vec![0u16; n / model.config().hop + 1];
        let k = transcribe(&model, &fe, pcm, &mut fs, &mut is, &mut ids).map_err(|_| -3isize)?;
        Ok::<isize, isize>(model.decode(&ids[..k], out) as isize)
    })();
    a.fs = fs;
    a.is = is;
    r.unwrap_or_else(|e| e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_voice_link::{HEADER, Packetizer};

    #[test]
    fn receiver_round_trip_and_nulls() {
        let r = rvl_receiver_new(4096);
        let pcm: Vec<i16> = (0..952).map(|i| ((i * 13) % 2000) as i16 - 1000).collect();
        let mut tx = Packetizer::new();
        let mut buf = [0u8; HEADER + 238];
        let mut last = 0;
        for (i, c) in pcm.chunks(476).enumerate() {
            let n = tx.packet(c, i == 1, &mut buf);
            last = unsafe { rvl_receiver_push(r, buf.as_ptr(), n) };
        }
        assert_eq!(last, 1);
        let mut p = ptr::null();
        let n = unsafe { rvl_receiver_utterance(r, &mut p) };
        assert_eq!(n, 952);
        assert!(!p.is_null());
        assert_eq!(unsafe { rvl_receiver_lost(r) }, 0);
        unsafe { rvl_receiver_free(r) };

        unsafe {
            assert_eq!(rvl_receiver_push(ptr::null_mut(), buf.as_ptr(), 6), -1);
            rvl_receiver_free(ptr::null_mut());
            assert!(rvl_asr_open([0u8; 8].as_ptr(), 8).is_null());
            assert_eq!(
                rvl_asr_transcribe(ptr::null_mut(), ptr::null(), 0, ptr::null_mut(), 0),
                -1
            );
        }
    }
}
