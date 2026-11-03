# rlx-kroko-core

**Kroko** streaming ASR ([Banafo/Kroko-ASR](https://huggingface.co/Banafo/Kroko-ASR)
community models, CC-BY-SA): a k2 Zipformer2 transducer with a stateless
predictor. No OS, no allocator, and weights paged from external storage.

```text
PCM ─► Kaldi fbank ─► chunks of 141 frames (step 128 = 1.28 s)
    ─► conv embed (3 convs + ConvNeXt, cached tail) ─► 64 × 192
    ─► 6 Zipformer2 stacks at ×1 ×2 ×4 ×8 ×4 ×2, 19 layers, per-layer caches
    ─► concat, ×2 downsample, project ─► 32 × 512 ─► joiner + predictor, greedy
```

## Paging

Kroko-L has **149 M parameters**, far more than an MCU's RAM. The `.rkr`
file is split in two:

* a **resident part** (2.3 MB: frontend, embed, predictor, joiner, tokens,
  layer table), kept in RAM and read in place;
* **one block per Zipformer layer**, read through a `Pager` into a page
  buffer each time the layer runs. On a board, `Pager` is your NAND, eMMC or
  SD driver; on the host and in QEMU it copies out of the image
  (`SlicePager`).

Every chunk reads every layer once, so real time needs the whole paged part
read per 1.28 s:

| layers | file | paged per chunk | page buffer | storage bandwidth for real time |
|---|---|---|---|---|
| int8 | 154 MB | 152 MB | 16 MB | ≈ 119 MB/s |
| int4 | 99 MB | 97 MB | 10 MB | ≈ 76 MB/s |

## Results (EN community 64-L, `Kroko-EN-Community-64-L-Streaming-001`)

| | int8 layers | int4 layers |
|---|---|---|
| JFK: token ids vs Kroko's own `encoder.onnx` (onnxruntime, same fbank) | identical (24 tokens) | identical |
| encoder output cosine vs ONNX, worst chunk | 0.9994 | 0.9937 |
| LibriSpeech dev-clean, first 100 of the 500-utterance list (1,992 words) | 2.96% WER | 3.01% WER |
| scratch (f32 + i8) | 6.9 MiB | 6.9 MiB |
| bare-metal rv32imafc (QEMU), 5.2 s clip | — | bit-identical to host; **124 G instructions** |

**Not real time on an MCU.** It needs about 1.5 GMAC per second of audio,
and 124 G instructions for 5.2 s is roughly 60× slower than real time on a
400 MHz ESP32-P4. Treat this as "the large model runs correctly from
NAND", not as a live recognizer. The smaller Kroko that esp32-pie-asr ran
live (65.7 M parameters) is not published; the same code would load it.

## A smaller streaming model: icefall Zipformer2 medium

The core is not tied to Kroko. Any Zipformer2 streaming transducer in the
sherpa-onnx int8 format converts with `scripts/sherpa_zipformer_to_kroko.py`.
That script wraps audio.cpp's converter: it lists the layers from the
encoder's metadata, finds the stack downsamplers by structure (older exports
name nodes by number), recovers the BiasNorm scales the exporter folded into
constants, and builds `tokens.txt` from `bpe.model`.
[sherpa-onnx-streaming-zipformer-en-2023-06-26](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26)
(icefall, LibriSpeech, 66M parameters, 320 ms chunks, left context 64)
runs unchanged:

| | Kroko-L (`64-L`) | Zipformer2 medium |
|---|---|---|
| layers per stack | 2,2,4,5,4,2 | 2,2,3,4,3,2 |
| chunk (latency) | 1.28 s | **320 ms** + 130 ms look-ahead |
| file int8 / int4 | 154 / 99 MB | 69 / 45 MB |
| JFK vs ONNX (onnxruntime) | identical tokens | identical tokens, worst chunk cosine 0.9985 |
| dev-clean 100 utts, int8 / int4 | 2.96% / 3.01% | **2.91% / 3.01%** |
| fixed memory (resident + scratch, int4) | 2.3 + 6.9 MiB | 2.0 + 6.9 MiB |
| rv32imafc, 5.2 s clip (int4, QEMU) | 124 G instr | **67.4 G instr**, bit-identical |
| weights read per second of audio (int4) | 76 MB/s | 135 MB/s |

Memory is fixed: it does not grow with the utterance, and a token comes out
every 320 ms. It is still far from a wearable budget:
- **Compute:** 13 G instructions per second of audio on portable rv32 code.
- **Weight bandwidth:** every chunk reads all the weights. A shorter chunk
  lowers latency but raises bandwidth, about 135 MB/s at int4 here.

A wearable needs weights that fit on-chip or in XIP flash, and an NPU. That
means about 10M parameters or fewer at int4/int8 (≤ 5–10 MB, ≤ 30 MB/s at
320 ms chunks). No such English streaming Zipformer2 is published; icefall's
recipes can train one, and this core and converter load it as is.

```bash
just fetch-zipformer-m                  # download + convert (onnx/numpy only)
just zipformer-m-check                  # export int4 + JFK
just zipformer-m-check assets/harness/jfk_16k.wav 8 --list /tmp/ls-dev-clean/list.tsv --limit 100
just zipformer-m-qemu                   # bare-metal rv32, bit-identity + instructions
```

`rlx-kroko-mcu` takes the model and its vectors from `RKR_MODEL` /
`RKR_DATA`. The defaults are Kroko-L int4 and `data/`.

## Run

```bash
hf download Banafo/Kroko-ASR Kroko-EN-Community-64-L-Streaming-001.data --local-dir .cache/kroko
# unpack + recover named float weights (audio.cpp's converter, onnx + numpy, no code execution)
python3 ../audio.cpp/tools/community_models/convert_kroko_onnx.py .cache/kroko/Kroko-EN-Community-64-L-Streaming-001.data .cache/kroko/en64-native
just kroko-mcu-check                    # export .rkr, run on JFK
just kroko-mcu-onnx-reference           # compare every chunk with Kroko's encoder.onnx
just kroko-mcu-qemu                     # firmware: bit-identity with paging + instruction count
```
