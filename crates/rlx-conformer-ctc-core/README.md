# rlx-conformer-ctc-core

Open-vocabulary English speech recognition with no operating system, no
allocator and no neural accelerator: NVIDIA's
[`stt_en_conformer_ctc_small`](https://huggingface.co/nvidia/stt_en_conformer_ctc_small)
(13M parameters) in int8, `no_std`.

The recipe is the one the Oído paper
([arXiv:2610.13447](https://arxiv.org/abs/2610.13447)) measured on an ESP32-S3:
int8 weights with one scale per output channel, int8 activations quantized per
row on the fly, and int8 attention. This crate is an independent
implementation; it reads its own `.rcc` format, exported here from the
original `.nemo`, not Oído's files.

## Results

| | f32 (`rlx-conformer-ctc`) | int8 (this crate) |
|---|---|---|
| LibriSpeech dev-clean, 500 utterances / 10,103 words, greedy | 3.70% WER | **3.60% WER** |
| identical transcripts | — | 402 / 500 |
| JFK clip (5.2 s): encoder cosine vs f32 | — | 0.9986, same token ids |
| model file | 49 MB `.nemo` | **13.9 MB `.rcc`** |
| RAM for 5.2 s of audio | — | 1.47 MB scratch |

The 500 utterances are every fifth row of the dev-clean split (37 speakers).
Oído reports 3.70% / 8.23% on test-clean / test-other for its int8 build.

On the MCU side, see [`rlx-conformer-ctc-mcu`](../rlx-conformer-ctc-mcu): the
same code on bare-metal rv32imafc (the ESP32-P4's ISA) produces bit-identical
encoder output to the host build.

## Use

```bash
just fetch-conformer-ctc
just export-conformer-ctc-mcu            # → .cache/conformer-ctc-mcu/small-int8.rcc
just conformer-ctc-mcu clip16k.wav       # int8 on the host, compared with f32
```

```rust
use rlx_conformer_ctc_core::{Frontend, Model, scratch_len, transcribe};

let model = Model::parse(blob)?;              // 4-byte aligned, e.g. mapped flash
let fe = Frontend::new(model.config().n_fft);
let need = scratch_len(&model, pcm.len());    // size your PSRAM buffers from this
let n = transcribe(&model, &fe, pcm, fs, is, &mut ids)?;   // pcm: &[i16] or &[f32]
let len = model.decode(&ids[..n], &mut text);
```

## Design

* **The model is read in place.** `.rcc` is a flat little-endian blob of
  word-aligned records; `Model::parse` checks shapes and keeps slices into it.
  On a chip that means memory-mapped flash, no copy into RAM.
* **No allocator.** `scratch_len` gives the size of one f32 and one i8 arena
  for a given audio length; `transcribe` carves them. Scratch grows with audio
  length (attention is over the whole utterance), about 0.28 MB per second.
* **Streaming subsampling.** The two stride-2 convs run row by row through a
  3-row int8 ring, so the 176×T/2×40 first-stage activation (5.6 MB at 4 s in
  f32) is never materialized.
* **Integer kernels.** int8 × int8 → i32 in 2-row × 4-output tiles; integer
  sums are exact in any order, so tiling changes no bits.
* **Same bits everywhere.** Transcendentals go through `libm` and nothing
  depends on fused multiply-add, so host and MCU builds agree exactly.
* LayerNorm, softmax, SiLU, GLU, the depthwise conv (BatchNorm folded in) and
  the log-mel frontend stay f32: a small share of the work, and the target
  chips have single-precision FPUs.

## Limits

* Offline only: `per_feature` mel normalization and full attention need the
  whole utterance (push-to-talk style). A streaming model is a different
  checkpoint (cache-aware FastConformer, see `rlx-nemotron-asr`).
* Classic `striding` ×4 subsampling (the `stt_en_conformer_ctc_*` family);
  FastConformer `dw_striding` is not handled.
* Speed on silicon: the portable kernels cost about 13.9 G instructions for
  5.2 s of audio on rv32imafc, roughly 6.7× real time on a 400 MHz ESP32-P4.
  Real time needs the chips' SIMD (ESP32-S3/P4 PIE int8 MAC instructions),
  which QEMU cannot run, so those kernels are not written yet.
