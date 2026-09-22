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

//! NeuCodec decoder: RLX path must be byte-identical to eager ndarray.
//!
//! ```sh
//! NEUTTS_DECODER_PATH=/path/to/neucodec_decoder.safetensors \
//!   cargo test -p rlx-neutts --features codec,rlx decoder_rlx_matches_eager --release
//! ```

use rlx_neutts::NeuCodecDecoder;

#[test]
fn decoder_rlx_backend_decodes_usable_audio() {
    // Renamed from `decoder_rlx_matches_eager`, which is what it claimed to do
    // and never did: it compared no values against the eager path, only the
    // backend *name*. A real comparison is not reachable from an integration
    // test (`decode_forward` is `pub(crate)`) and would be tautological anyway
    // — `decoder::rlx::decode` currently delegates straight to it. When the rlx
    // path grows its own implementation, the parity test belongs beside it as a
    // lib test with access to both.
    if !cfg!(feature = "rlx") {
        return;
    }
    // Burn path is preferred when both `burn` and `rlx` are enabled; use
    // `decode_output_matches_eager_forward` (lib test) for burn/wgpu parity.
    if cfg!(feature = "burn") {
        eprintln!("skip decoder_rlx_matches_eager: burn takes precedence over rlx");
        return;
    }

    let Some(path) = rlx_neutts::decoder::decoder_weights_path_if_available() else {
        eprintln!("skip: set NEUTTS_DECODER_PATH");
        return;
    };

    let dec = NeuCodecDecoder::from_file(&path).expect("load decoder");
    let codes: Vec<i32> = vec![0, 42, 128, 512, 1023];

    let audio = dec.decode(&codes).expect("decode");
    assert!(
        dec.backend_name().contains("rlx") || dec.backend_name().contains("eager"),
        "unexpected backend: {}",
        dec.backend_name()
    );
    assert!(!audio.is_empty(), "no audio produced");
    assert!(audio.iter().all(|s| s.is_finite()), "non-finite sample");
    assert_eq!(
        audio.len(),
        codes.len() * dec.hop_length(),
        "one hop of samples per code"
    );
    // Silence is finite and the right length, so the checks above pass on a
    // decoder that emits nothing at all.
    assert!(
        audio.iter().any(|s| s.abs() > 1e-9),
        "decoded audio is identically zero"
    );
    let first = audio[0];
    assert!(
        audio.iter().any(|s| (s - first).abs() > 1e-9),
        "decoded audio is the constant {first}"
    );
    // Different codes must give different audio, or the codebook lookup is
    // being ignored.
    let other: Vec<i32> = codes.iter().map(|c| (c + 7) % 1024).collect();
    let audio2 = dec.decode(&other).expect("decode second");
    assert!(
        audio.iter().zip(&audio2).any(|(a, b)| (a - b).abs() > 1e-9),
        "two different code sequences decoded identically"
    );
}
