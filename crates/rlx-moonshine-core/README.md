# rlx-moonshine-core

Moonshine v2 streaming ASR ([arXiv:2602.12241](https://arxiv.org/abs/2602.12241),
[`moonshine-ai/moonshine-streaming-tiny`](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny), MIT)
with no OS and no allocator, on [`rlx-mcu`](https://github.com/MIT-RLX/rlx/tree/main/crates/backends/rlx-mcu).

| | f32 (`rlx-moonshine-streaming`) | MCU (this crate) |
|---|---|---|
| JFK clip | "Ask not what your country can do for you. Ask what you can do for your country." | same 19 token ids |
| encoder output cosine vs f32 | — | 0.985 |
| LibriSpeech dev-clean, first 200 of the 500-utterance list | 6.42% WER | 5.71% WER |
| model file | 176 MB safetensors | 37.4 MB `.rms` |
| scratch for 5.2 s | — | 2.7 MB |
| bare-metal rv32imafc (QEMU) | — | bit-identical to host; 20.7 G instructions for 5.2 s |

The WER run stopped at 200 utterances when the machine ran out of memory.
The f32 reference keeps one compiled graph per clip length, so it is the
side that grows. Read those two numbers as "int8 is not worse", not as a
final figure.

Weights: int8 per row by default; the 32k-row token embedding and LM head
are int4 (one scale per 32), the frontend linear and convs int8 with one scale
per 16. The model has about 44M parameters: the vocabulary pair is 21M, the
encoder 9M, the decoder 12M.

```bash
hf download moonshine-ai/moonshine-streaming-tiny --local-dir weights/asr/moonshine-streaming-tiny
just moonshine-mcu-check                 # export + compare with f32 on JFK
just moonshine-mcu-qemu                  # firmware, bit-identity + instruction count
```

## Design

* The frontend (80-sample frames, CMVN, asinh, linear) and both stride-2
  causal convs stream through 5-row rings, so the 200 Hz intermediate never
  exists.
* Encoder attention visits only each layer's `(left, right)` window; int8
  scores and values.
* The decoder keeps an f32 self-attention KV cache and int8 cross-attention
  K/V computed once per utterance. The LM head is an argmax over its rows,
  so the 32k logits never exist.
* Offline per utterance for now. The encoder is streaming-shaped (bounded
  windows), so chunked encoding is the natural next step.

Speed: about 10× slower than real time on a 400 MHz ESP32-P4 with the
portable kernels (20.7 G instructions for 5.2 s). Real time needs the
chip's SIMD; see `rlx-conformer-ctc-mcu`.
