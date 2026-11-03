# rlx-citrinet-core

NVIDIA NeMo **Citrinet** CTC ASR
([`stt_en_citrinet_256_ls`](https://huggingface.co/nvidia/stt_en_citrinet_256_ls),
CC-BY-4.0) with no OS and no allocator, on [`rlx-mcu`](https://github.com/MIT-RLX/rlx/tree/main/crates/backends/rlx-mcu).
It uses only convolutions (separable convs with squeeze-and-excitation), so
memory grows only linearly with the clip and it has the cheapest compute of
the MCU models here.

| | f32 reference | MCU (int8) |
|---|---|---|
| JFK clip | "ask not what your country can do for you ask what you can do for your country" | same |
| encoder output cosine | — | 0.999 |
| LibriSpeech dev-clean, 500 utterances / 10,103 words | 5.98% → see note | **3.58% WER** (461/500 transcripts identical to f32 before the detokenizer fix) |
| model file | 36.7 MB `.nemo` | **13.3 MB `.rct`** |
| scratch, 5.2 s | — | 0.86 MB (in place; was 2.3 MB) |
| bare-metal rv32imafc (QEMU, ESP32-P4 ISA) | — | bit-identical to host; **6.1 G instructions** for 5.2 s (portable code, no SIMD) |
| bare-metal Cortex-M33 + DSP (QEMU, nRF54H20 app core) | — | bit-identical to host; **2.59 G instructions** for 5.2 s in 768 KB of RAM ([`rlx-citrinet-nrf54h`](../rlx-citrinet-nrf54h)) |

The published LibriSpeech test-clean WER of the checkpoint is 3.78%. The
first run scored 5.98% (f32) and 5.95% (int8) because the WordPiece
detokenizer printed `wife ' s`. The model gets apostrophes right; the decoder
now joins an apostrophe to its neighbours. After that fix, int8 scores
3.58%.

* Pointwise and residual 1×1 convs (about 95% of the MACs) are int8
  matrices with BatchNorm folded in. Depthwise convs and the SE MLP stay f32.
* The checkpoint stores no filterbank, so the exporter builds NeMo's: a
  Slaney mel filterbank and a symmetric Hann window. The log-mel frontend
  is shared with `rlx-conformer-ctc-core` (`rlx_mcu::mel`).
* The checkpoint is read with `rlx-nemo`, which parses the `.nemo` without
  running pickle code.

## Memory

Each conv runs in place. Input rows go through a ring of `kernel + stride`
rows as they enter the conv window, so the outputs can overwrite rows the
window has passed. The residual path is computed at the start of a block,
before the repeats overwrite its input, and the log-mel lands straight in the
work buffer. Scratch is one block's activations plus one block's residual
output. For the 5.2 s clip that is 858 KB, against 2.3 MB before, with the
same bits (each value is computed by the same operations in the same order).

Scratch still grows with the utterance (squeeze-and-excitation averages
over the whole clip). `max_samples(model, f32s, i8s)` gives the longest
utterance a scratch fits. `next_cut` (from `rlx-mcu`) splits longer audio at
its quietest 25 ms in the allowed span, the way a VAD would, and gives the same
cuts on every target.

## Pipeline stages

`rlx_citrinet::mcu::export_rct_stages(model, bits, &Stages::Balanced(n))`
writes one `.rct` per stage. Each holds only its blocks: the first also
holds the log-mel frontend, the last the CTC head. `Stages::Cuts(vec![0, 7, 23])`
sets the boundaries by hand. `block_costs` weights the blocks for
`rlx_mcu::split::balance`. `run_stage` starts from PCM or from received
activation rows (`stage_input`) and ends in token ids or rows for the next
stage (`stage_output`). `rlx_mcu::split::{send_rows, recv_rows}` frame the rows
for SPI/UART with a CRC. Chained, the stages give the same bits as the whole
model:

| `--stages` | blocks per stage | compute | link, 5.2 s |
|---|---|---|---|
| 2 | 0..7, 7..23 | 50% / 50% | 261 KB |
| 3 | 0..5, 5..11, 11..23 | 35 / 33 / 32% | 261 KB, 131 KB |
| 4 | 0..4, 4..7, 7..13, 13..23 | 27 / 23 / 25 / 25% | 261, 261, 131 KB |

File format version 2 stores the depthwise weights tap-major `[kernel, cin]`
and a block range per file.

```bash
hf download nvidia/stt_en_citrinet_256_ls --local-dir .cache/citrinet
just citrinet-mcu-check                 # export, int8 vs f32 on JFK
just citrinet-mcu-qemu                  # rv32 firmware: bit-identity + instruction count
just citrinet-mcu-split 3               # 3 stage files, plan, pipeline == whole model
just citrinet-nrf54h-qemu               # Cortex-M33 + DSP firmware (nRF54H20)
just citrinet-nrf54h-pipeline 2         # 2 simulated nRF54H20s over UART
```

The f32 reference in `rlx-citrinet` (`mcu::reference`) is an independent,
plain-loop implementation. Its transcripts agree with the int8 core, and
the int8 core's encoder output matches it to cosine 0.999.
