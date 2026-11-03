// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! The simulated link (`Radio`) and the full JFK path: device → lossy radio
//! → reassembly → int8 ASR.
//!
//! The radio tests need nothing. The JFK tests need the `.rcc` model
//! (`just export-conformer-ctc-mcu`, or `RLX_RCC=/path`); without it they
//! print why and pass, so `cargo test` stays usable on a fresh clone.

use std::path::PathBuf;

use rlx_voice_link::adpcm;
use rlx_voice_link_host::{Asr, Loss, Radio, read_wav_i16, simulate};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

fn tone(n: usize) -> Vec<i16> {
    (0..n)
        .map(|i| ((i as f64 * 0.031).sin() * 9000.0 + (i as f64 * 0.17).sin() * 2000.0) as i16)
        .collect()
}

/// ADPCM round trip of the whole signal in one stream: what a lossless link
/// must deliver, sample for sample.
fn adpcm_reference(pcm: &[i16]) -> Vec<i16> {
    let pcm = &pcm[..pcm.len() & !1];
    let mut codes = vec![0u8; pcm.len() / 2];
    adpcm::encode(&mut adpcm::State::default(), pcm, &mut codes);
    let mut out = vec![0i16; pcm.len()];
    adpcm::decode(&mut adpcm::State::default(), &codes, &mut out);
    out
}

#[test]
fn lossless_link_delivers_the_adpcm_stream_exactly() {
    let pcm = tone(16_001); // odd: the last sample is dropped, as on the device
    let r = Radio::new(Loss::None, 1).send(&pcm);
    assert_eq!(r.dropped, 0);
    assert_eq!(r.lost, 0);
    assert_eq!(r.packets, 16_000usize.div_ceil(476));
    assert_eq!(r.pcm, adpcm_reference(&pcm));
}

#[test]
fn loss_keeps_length_and_is_reproducible() {
    let pcm = tone(48_000);
    for p in [1.0, 10.0, 30.0, 60.0] {
        let a = Radio::new(Loss::Random(p), 42).send(&pcm);
        let b = Radio::new(Loss::Random(p), 42).send(&pcm);
        assert_eq!(a.pcm, b.pcm, "same seed, same result at {p}%");
        assert_eq!(a.dropped, b.dropped);
        // Gaps are filled with silence of the right length, so timing holds.
        assert_eq!(a.pcm.len(), pcm.len(), "{p}% loss changed the length");
        assert_eq!(a.lost as usize, a.dropped, "receiver counted every gap");
    }
    let a = Radio::new(Loss::Random(10.0), 1).send(&pcm);
    let b = Radio::new(Loss::Random(10.0), 2).send(&pcm);
    assert_ne!(a.pcm, b.pcm, "different seeds, different losses");
}

#[test]
fn random_loss_rate_is_close_to_the_setting() {
    let mut radio = Radio::new(Loss::Random(10.0), 7);
    let (mut sent, mut dropped) = (0, 0);
    for _ in 0..20 {
        let r = radio.send(&tone(32_000));
        // The END packet is never dropped; leave it out of the rate.
        sent += r.packets - 1;
        dropped += r.dropped;
    }
    let rate = 100.0 * dropped as f64 / sent as f64;
    assert!((7.0..13.0).contains(&rate), "10% setting gave {rate:.1}%");
}

#[test]
fn every_nth_and_lost_start_packets() {
    // Every 2nd packet lost, the first utterance's START among them on the
    // second send: the utterance still arrives at full length.
    let pcm = tone(4760); // 10 packets
    let mut radio = Radio::new(Loss::Every(2), 1);
    for _ in 0..3 {
        let r = radio.send(&pcm);
        assert_eq!(r.pcm.len(), 4760);
        assert!(r.dropped >= 4, "{r:?}");
    }
    let r = simulate(&pcm, 244, Some(3));
    assert_eq!(r.pcm.len(), 4760);
    assert_eq!(r.dropped, 3);
}

#[test]
fn small_mtu_and_empty_utterance() {
    let pcm = tone(1000);
    let mut radio = Radio::new(Loss::None, 1);
    radio.att_payload = 20; // no MTU exchange: 28 samples a packet
    let r = radio.send(&pcm);
    assert_eq!(r.packets, 1000usize.div_ceil(28));
    assert_eq!(r.pcm, adpcm_reference(&pcm));
    let r = Radio::new(Loss::None, 1).send(&[]);
    assert_eq!(
        (r.packets, r.pcm.len()),
        (1, 0),
        "a lone END closes an empty utterance"
    );
}

// ---- JFK, end to end -------------------------------------------------------

const ASK_NOT: &str =
    "ask not what your country can do for you ask what you can do for your country";
const RUST: &str = "my fellow americans to day we choose to write our software in rust \
                    not because it is easy but because it is fast safe and free of garbage collection";

fn asr() -> Option<Asr> {
    let path = std::env::var_os("RLX_RCC")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(format!("{ROOT}/.cache/conformer-ctc-mcu/small-int8.rcc"))
        });
    if !path.exists() {
        eprintln!(
            "skipping JFK tests: {} missing (run `just export-conformer-ctc-mcu` or set RLX_RCC)",
            path.display()
        );
        return None;
    }
    Some(Asr::open(&path).expect("valid .rcc"))
}

fn jfk(name: &str) -> Vec<i16> {
    read_wav_i16(&PathBuf::from(format!("{ROOT}/{name}"))).expect("JFK clip in assets/")
}

/// Word error rate of `hyp` against `reference`.
fn wer(reference: &str, hyp: &str) -> f64 {
    let (r, h): (Vec<&str>, Vec<&str>) = (
        reference.split_whitespace().collect(),
        hyp.split_whitespace().collect(),
    );
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, x) in r.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, y) in h.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(x != y))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[h.len()] as f64 / r.len() as f64
}

const CLIPS: [(&str, &str); 3] = [
    ("assets/harness/jfk_16k.wav", ASK_NOT),
    ("assets/jfk/jfk_voice_clone.wav", ASK_NOT),
    ("assets/jfk/jfk_rust_speech.wav", RUST),
];

#[test]
fn jfk_exact_through_a_clean_or_lightly_lossy_link() {
    let Some(asr) = asr() else { return };
    for (clip, want) in CLIPS {
        let pcm = jfk(clip);
        assert_eq!(
            asr.transcribe(&pcm).unwrap(),
            want,
            "{clip}: original audio"
        );
        for (loss, seed) in [
            (Loss::None, 1),
            (Loss::Random(5.0), 1),
            (Loss::Random(5.0), 9),
        ] {
            let r = Radio::new(loss, seed).send(&pcm);
            assert_eq!(r.pcm.len(), pcm.len() & !1, "{clip} {loss:?}: length");
            assert_eq!(
                asr.transcribe(&r.pcm).unwrap(),
                want,
                "{clip} at {loss:?} seed {seed}: {} of {} packets lost",
                r.dropped,
                r.packets
            );
        }
    }
}

#[test]
fn jfk_word_error_grows_gracefully_with_loss() {
    let Some(asr) = asr() else { return };
    // Bounds with headroom over what seed 7 measured. The 5 s "ask not"
    // clips stayed word-exact even at 50% loss; the 12 s Rust clip measured
    // 13.8% WER at 15% loss ("rus", a dropped "but"), 6.9% at 30% and 34.5%
    // at 50% — worse, but never empty or a different sentence.
    for (loss, max_wer) in [(15.0, 0.2), (30.0, 0.4), (50.0, 0.7)] {
        for (clip, want) in CLIPS {
            let pcm = jfk(clip);
            let r = Radio::new(Loss::Random(loss), 7).send(&pcm);
            assert_eq!(r.pcm.len(), pcm.len() & !1, "{clip} {loss}%: length");
            let got = asr.transcribe(&r.pcm).unwrap();
            let w = wer(want, &got);
            eprintln!(
                "{loss:>4}% loss, {:>3}/{:<3} packets, WER {:>5.1}%  {clip}: {got}",
                r.dropped,
                r.packets,
                100.0 * w
            );
            assert!(
                w <= max_wer,
                "{clip} at {loss}%: WER {:.1}%: {got:?}",
                100.0 * w
            );
        }
    }
}
