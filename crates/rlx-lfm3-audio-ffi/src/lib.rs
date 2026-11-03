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

//! C ABI over [`rlx_lfm3_audio`]: open a model, transcribe 16 kHz audio,
//! synthesize 24 kHz speech. Declared in `include/rlx_lfm3_audio.h`.
//!
//! Every call that can fail returns NULL (or `false`) and leaves a message
//! for [`lfm3a_last_error`]. Strings and audio returned to the caller are
//! freed with [`lfm3a_free_string`] / [`lfm3a_free_audio`]. A handle is not
//! thread-safe; use it from one thread at a time.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::path::Path;

use rlx_lfm3_audio::Lfm3Audio;
use rlx_lfm3_audio::text::Tokenizer;

/// Generation steps per `lfm3a_speak` call (one per speech frame, 12.5 per
/// second: ~80 s).
const MAX_SPEECH_STEPS: usize = 1024;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(e: impl std::fmt::Display) {
    let msg = CString::new(format!("{e:#}").replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|c| *c.borrow_mut() = msg);
}

/// A loaded model.
pub struct Lfm3aModel {
    model: Lfm3Audio,
    tok: Tokenizer,
}

/// # Safety
/// `p` is NULL or a NUL-terminated string.
unsafe fn str_arg<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the caller passes a NUL-terminated string.
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

fn run<T>(f: impl FnOnce() -> anyhow::Result<T>) -> Option<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Some(v),
        Ok(Err(e)) => {
            set_error(e);
            None
        }
        Err(_) => {
            set_error("panic inside rlx-lfm3-audio");
            None
        }
    }
}

/// The message of the last failed call on this thread ("" if none). Valid
/// until the next failing call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn lfm3a_last_error() -> *const c_char {
    LAST_ERROR.with(|c| c.borrow().as_ptr())
}

/// Open the LM GGUF and its mmproj on `device` (`"cpu"`, `"metal"`,
/// `"coreml"`, …; NULL = `"metal"`). `low_memory` keeps one LM graph and drops
/// stage graphs after use (see `Lfm3Audio::set_low_memory`). NULL on failure.
///
/// # Safety
/// The strings are NULL or NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lfm3a_open(
    lm_path: *const c_char,
    mmproj_path: *const c_char,
    device: *const c_char,
    low_memory: bool,
) -> *mut Lfm3aModel {
    // SAFETY: forwarded caller contract.
    let (lm, mm, dev) = unsafe { (str_arg(lm_path), str_arg(mmproj_path), str_arg(device)) };
    run(|| {
        let lm = lm.ok_or_else(|| anyhow::anyhow!("lm_path is NULL or not UTF-8"))?;
        let mm = mm.ok_or_else(|| anyhow::anyhow!("mmproj_path is NULL or not UTF-8"))?;
        let device = rlx_cli::parse_device(dev.unwrap_or("metal"))?;
        let mut model = Lfm3Audio::open(Path::new(lm), Path::new(mm), device)?;
        model.set_low_memory(low_memory);
        let tok = Tokenizer::new(Path::new(lm));
        Ok(Box::into_raw(Box::new(Lfm3aModel { model, tok })))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Free a model from [`lfm3a_open`].
///
/// # Safety
/// `m` is NULL or a handle from `lfm3a_open`, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lfm3a_close(m: *mut Lfm3aModel) {
    if !m.is_null() {
        // SAFETY: the handle came from `Box::into_raw` in `lfm3a_open`.
        drop(unsafe { Box::from_raw(m) });
    }
}

/// Transcribe `n` mono samples at 16 kHz. `language` is the English name of
/// the spoken language (`"English"`, `"French"`, …; NULL lets the model pick).
/// Returns the text (free with [`lfm3a_free_string`]) or NULL.
///
/// # Safety
/// `m` is a live handle, `pcm` points at `n` floats, `language` is NULL or
/// NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lfm3a_transcribe(
    m: *mut Lfm3aModel,
    pcm: *const f32,
    n: usize,
    language: *const c_char,
    max_tokens: usize,
) -> *mut c_char {
    // SAFETY: forwarded caller contract.
    let (m, lang) = unsafe { (m.as_mut(), str_arg(language)) };
    run(|| {
        let m = m.ok_or_else(|| anyhow::anyhow!("model handle is NULL"))?;
        if pcm.is_null() {
            anyhow::bail!("pcm is NULL");
        }
        // SAFETY: `pcm` points at `n` floats (caller contract).
        let pcm = unsafe { std::slice::from_raw_parts(pcm, n) };
        let max_tokens = if max_tokens == 0 { 256 } else { max_tokens };
        let text = m.model.transcribe(&m.tok, pcm, lang, max_tokens)?;
        Ok(CString::new(text.replace('\0', " "))?.into_raw())
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Synthesize `text` as 24 kHz mono speech. `language` is a code (`"en"`,
/// `"fr"`, …; NULL = `"en"`), `voice` `"female"` or `"male"` (NULL =
/// female). `seed` drives the speech sampling. Writes the sample count to
/// `out_len` and returns the samples (free with [`lfm3a_free_audio`]), or NULL.
///
/// # Safety
/// `m` is a live handle, the strings are NULL or NUL-terminated, `out_len`
/// is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lfm3a_speak(
    m: *mut Lfm3aModel,
    text: *const c_char,
    language: *const c_char,
    voice: *const c_char,
    seed: u64,
    out_len: *mut usize,
) -> *mut f32 {
    // SAFETY: forwarded caller contract.
    let (m, text, lang, voice) =
        unsafe { (m.as_mut(), str_arg(text), str_arg(language), str_arg(voice)) };
    run(|| {
        let m = m.ok_or_else(|| anyhow::anyhow!("model handle is NULL"))?;
        let text = text.ok_or_else(|| anyhow::anyhow!("text is NULL or not UTF-8"))?;
        if out_len.is_null() {
            anyhow::bail!("out_len is NULL");
        }
        let pcm = m
            .model
            .speak(
                &m.tok,
                text,
                lang.unwrap_or("en"),
                voice.unwrap_or("female"),
                seed,
                MAX_SPEECH_STEPS,
            )?
            .into_boxed_slice();
        // SAFETY: `out_len` is writable (checked non-NULL above).
        unsafe { *out_len = pcm.len() };
        Ok(Box::into_raw(pcm) as *mut f32)
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Free a string from this library.
///
/// # Safety
/// `s` is NULL or a string returned by this library, not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lfm3a_free_string(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: the string came from `CString::into_raw`.
        drop(unsafe { CString::from_raw(s) });
    }
}

/// Free audio from [`lfm3a_speak`] (`len` = the length it reported).
///
/// # Safety
/// `pcm` is NULL or audio returned by `lfm3a_speak` with that `len`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lfm3a_free_audio(pcm: *mut f32, len: usize) {
    if !pcm.is_null() {
        // SAFETY: the buffer came from a `Box<[f32]>` of `len` samples.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(pcm, len)) });
    }
}

/// This process's physical footprint in bytes (what iOS's memory limit
/// counts); 0 off Apple platforms.
#[unsafe(no_mangle)]
pub extern "C" fn lfm3a_footprint() -> u64 {
    rlx_lfm3_audio::footprint()
}
