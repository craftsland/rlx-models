# rlx-conformer-ctc-mcu

Bare-metal RISC-V firmware running
[`rlx-conformer-ctc-core`](../rlx-conformer-ctc-core): int8 Conformer-CTC
speech recognition with no OS and no allocator.

It targets QEMU's `virt` board with `rv32imafc`, the ESP32-P4's ISA, so the
compute path is what that chip executes; on silicon only the memory map
(model in mapped flash, scratch in PSRAM) and the UART address change. The
ESP32-S3 is Xtensa and needs the esp toolchain; the kernels are the same code.

## Run

```bash
just fetch-conformer-ctc && just export-conformer-ctc-mcu
just conformer-ctc-mcu-qemu
```

```text
conformer-ctc int8: 83200 samples, model 13899832 B, scratch 1467795 B
text: ask not what your country can do for you ask what you can do for your country
ids match, encoder fnv 0xca3b8a74 match
INSNS 13865826010
```

The firmware transcribes a 5.2 s clip linked into the image and checks it
against vectors the host computed from the same `.rcc`: equal token ids and an
equal FNV-1a hash of the encoder output's bits. A pass means the MCU build is
bit-identical to the host build. `mcycle` on the `virt` board is
wall-clock-derived and not a cycle count; `INSNS` (QEMU plugin from
`rlx-ten-vad-mcu`) is reproducible.

The model is not committed. `build.rs` links
`.cache/conformer-ctc-mcu/small-int8.rcc`, or `RCC_MODEL=/path`. After changing
kernels or the model, regenerate `data/`:

```bash
cargo run -p rlx-conformer-ctc --release --example mcu_vectors -- \
  --rcc .cache/conformer-ctc-mcu/small-int8.rcc --wav assets/harness/jfk_16k.wav \
  --out crates/rlx-conformer-ctc-mcu/data
```

## Cost

| kernel version | instructions, 5.2 s clip |
|---|---|
| plain int8 dot products | 20.97 G |
| 2×4 tiles in every linear | 14.47 G |
| + paired positions in the subsampling conv | **13.87 G** |

At one instruction per cycle that is about 35 s on a 400 MHz ESP32-P4, 6.7×
real time. For reference, Oído reports 1.97× real time on an ESP32-S3 and
esp32-pie-asr runs a Kroko transducer live on an ESP32-P4, both with
hand-written SIMD. That is the next step here: the PIE int8 MAC instructions
(16 lanes) would replace the inner loop of `dot_2x4`, which QEMU cannot run, so
it has to be measured on a board.
