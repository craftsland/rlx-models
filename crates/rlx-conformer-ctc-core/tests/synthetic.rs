// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! A tiny random model written with `blob::Writer` and run end to end — no
//! checkpoint needed. Real-weight parity lives in `rlx-conformer-ctc`
//! (`transcribe-mcu --nemo`, `examples/mcu_wer.rs`).

use rlx_conformer_ctc_core::blob::Writer;
use rlx_conformer_ctc_core::{Error, Frontend, Model, scratch_len, transcribe};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
    }
    fn v(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f()).collect()
    }
}

const N_MELS: usize = 16;
const N_FFT: usize = 64;
const D: usize = 16;
const HEADS: usize = 2;
const FF: usize = 32;
const K: usize = 5;
const CH: usize = 8;
const CLASSES: usize = 6;
const LAYERS: usize = 2;

fn tiny_model() -> Vec<u8> {
    let mut r = Rng(7);
    let mut w = Writer::new(
        rlx_conformer_ctc_core::blob::MAGIC,
        rlx_conformer_ctc_core::blob::VERSION,
    );
    for v in [
        16_000,
        N_MELS,
        N_FFT,
        16,
        D,
        LAYERS,
        HEADS,
        FF,
        K,
        CH,
        CLASSES,
        CLASSES - 1,
    ] {
        w.u32(v as u32);
    }
    w.f32s(&[0.5; N_FFT]);
    // Two bins per filter.
    w.u32s(&(0..N_MELS as u32).map(|m| m * 2).collect::<Vec<_>>());
    w.u32s(&(0..=N_MELS as u32).map(|m| m * 2).collect::<Vec<_>>());
    w.f32s(&[1.0; 2 * N_MELS]);
    w.f32s(&r.v(CH * 9));
    w.f32s(&r.v(CH));
    w.q8(&r.v(CH * 9 * CH), CH, 9 * CH);
    w.f32s(&r.v(CH));
    let f2 = (N_MELS.div_ceil(2)).div_ceil(2);
    w.q8(&r.v(D * CH * f2), D, CH * f2);
    w.f32s(&r.v(D));
    let ln = |w: &mut Writer| {
        w.f32s(&[1.0; D]);
        w.f32s(&[0.0; D]);
    };
    for _ in 0..LAYERS {
        let ffn = |w: &mut Writer, r: &mut Rng| {
            ln(w);
            w.q8(&r.v(FF * D), FF, D);
            w.f32s(&r.v(FF));
            w.q8(&r.v(D * FF), D, FF);
            w.f32s(&r.v(D));
        };
        ffn(&mut w, &mut r);
        ln(&mut w);
        for _ in 0..4 {
            w.q8(&r.v(D * D), D, D);
            w.f32s(&r.v(D));
        }
        w.q8(&r.v(D * D), D, D);
        w.f32s(&r.v(D));
        w.f32s(&r.v(D));
        ln(&mut w);
        w.q8(&r.v(2 * D * D), 2 * D, D);
        w.f32s(&r.v(2 * D));
        w.f32s(&r.v(D * K));
        w.f32s(&r.v(D));
        w.q8(&r.v(D * D), D, D);
        w.f32s(&r.v(D));
        ffn(&mut w, &mut r);
        ln(&mut w);
    }
    w.q8(&r.v(CLASSES * D), CLASSES, D);
    w.f32s(&r.v(CLASSES));
    let pieces = ["<unk>", "\u{2581}a", "b", "\u{2581}c", "d"];
    let mut offs = vec![0u32];
    let mut bytes = Vec::new();
    for p in pieces.iter().chain([&""]) {
        bytes.extend_from_slice(p.as_bytes());
        offs.push(bytes.len() as u32);
    }
    w.u32s(&offs);
    w.bytes(&bytes);
    w.finish()
}

fn aligned(bytes: &[u8]) -> Vec<u32> {
    let mut v = vec![0u32; bytes.len().div_ceil(4)];
    for (i, c) in bytes.chunks(4).enumerate() {
        let mut b = [0u8; 4];
        b[..c.len()].copy_from_slice(c);
        v[i] = u32::from_le_bytes(b);
    }
    v
}

fn as_bytes(v: &[u32], len: usize) -> &[u8] {
    // SAFETY: u32 storage viewed as bytes, within bounds.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), len) }
}

#[test]
fn runs_end_to_end_and_is_deterministic() {
    let blob = tiny_model();
    let words = aligned(&blob);
    let model = Model::parse(as_bytes(&words, blob.len())).unwrap();
    assert_eq!(model.config().d_model, D);
    let fe = Frontend::new(N_FFT);
    let pcm: Vec<i16> = (0..4000)
        .map(|i| ((i * 37 % 200) as i16 - 100) * 50)
        .collect();
    let need = scratch_len(&model, pcm.len());
    let run = || {
        let mut fs = vec![f32::NAN; need.f32s];
        let mut is = vec![0i8; need.i8s];
        let mut ids = [0u16; 512];
        let n = transcribe(&model, &fe, pcm.as_slice(), &mut fs, &mut is, &mut ids).unwrap();
        ids[..n].to_vec()
    };
    let a = run();
    assert_eq!(a, run());
    assert!(
        a.iter().all(|&i| (i as usize) < CLASSES - 1),
        "blank leaked: {a:?}"
    );
    let mut text = [0u8; 256];
    let n = model.decode(&a, &mut text);
    assert!(std::str::from_utf8(&text[..n]).is_ok());

    // Too little scratch is an error, not a panic.
    let mut fs = vec![0f32; need.f32s - 1];
    let mut is = vec![0i8; need.i8s];
    let mut ids = [0u16; 512];
    assert_eq!(
        transcribe(&model, &fe, pcm.as_slice(), &mut fs, &mut is, &mut ids),
        Err(Error::ScratchTooSmall)
    );
}

#[test]
fn rejects_bad_files() {
    let blob = tiny_model();
    let words = aligned(&blob);
    let bytes = as_bytes(&words, blob.len());
    assert!(matches!(
        Model::parse(&bytes[..bytes.len() / 2]),
        Err(Error::Truncated | Error::Shape)
    ));
    let mut bad = words.clone();
    bad[0] ^= 1;
    assert_eq!(
        Model::parse(as_bytes(&bad, blob.len())).err(),
        Some(Error::BadMagic)
    );
    let unaligned = &[0u8; 64];
    let off = if (unaligned.as_ptr() as usize).is_multiple_of(4) {
        1
    } else {
        0
    };
    assert_eq!(
        Model::parse(&unaligned[off..]).err(),
        Some(Error::Unaligned)
    );
}
