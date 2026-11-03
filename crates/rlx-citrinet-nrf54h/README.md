# rlx-citrinet-nrf54h

Bare-metal firmware running Citrinet-256 CTC speech recognition
([`rlx-citrinet-core`](../rlx-citrinet-core), int8) on the **nRF54H20**
application core: a Cortex-M33 at 320 MHz with FPU and the DSP extension. No
OS, no allocator, no accelerator. It runs on QEMU's MPS2 AN505 (a Cortex-M33
board). `link.x` gives each region the size it has on the chip, so an image
that links here fits the chip:

| region | nRF54H20 | holds |
|---|---|---|
| external flash on EXMIF | (board) | the 13.3 MB model, read in place |
| MRAM | 2 MB | code, constants, the test clip (217 KB used) |
| general-purpose RAM (RAM0x) | 768 KB | scratch (708 KB), stack |

Scratch grows with the utterance (Citrinet's squeeze-and-excitation averages
over the whole clip). In 708 KB it holds up to 4.3 s. Longer audio is cut at
its quietest point (`next_cut`, as a VAD would) and transcribed piece by
piece. The JFK clip (5.2 s) splits at the pause between its two sentences.

## Results (QEMU, 5.2 s JFK clip)

| build | instructions | |
|---|---|---|
| portable kernels (`--no-default-features`) | 7.31 G | bit-identical to host |
| `SXTB16`/`SMLAD` kernels (default `dsp`) | **2.59 G** | bit-identical to host |

Real time at 320 MHz is 1.66 G cycles for 5.2 s, so one core needs about
1.6× more than it has, before counting stalls on external flash. Instruction
counts are not cycles. On this core most instructions take one cycle, loads
two, and EXMIF reads add wait states that only a board can measure.

Where the 2.59 G goes (`tools/pcprof.py`): int8 matmuls 1.89 G (1.59 G of it in
the assembly inner loop, 19 instructions per 16 MACs), depthwise convs 0.54 G,
log-mel 0.07 G.

What got it from 7.31 G to 2.59 G, all with the same bits:

* **2×2 int8 tile in inline assembly** (`rlx_mcu::kernels`, feature `dsp`).
  Activations are widened to i16 once per row pair, in the lane order
  `SXTB16` gives a weight word. Each `SMLAD` then adds two products, and the
  loop uses exactly the 12 registers Thumb inline assembly gets. The compiler's
  2×4 tile spilled to the stack (Cortex-M has too few registers).
* **Tap-major depthwise weights** (`.rct` version 2). Eight channels
  accumulate in registers across all taps, in the same order as before.

## Two or more chips

`just citrinet-nrf54h-pipeline N` cuts the model into N balanced stages
(`examples/mcu_split.rs`) and runs one QEMU per stage. The QEMUs are chained
UART to UART over local TCP. Stage 0 transcribes and sends activation frames
(`rlx_mcu::split`: magic, sequence number, shape, CRC-32). The last stage
checks the token ids and the encoder bits against the same vectors as the
single-chip run.

| chips | blocks | compute (5.2 s) | link |
|---|---|---|---|
| 2 | 0..7 / 7..23 | 1.36 G / 1.24 G | 140 + 121 rows × 256 f32 |
| 3 | 0..5 / 5..11 / 11..23 | 0.97 G / 0.83 G / 0.80 G | 261 rows, then 131 |

With two chips each core does about half the work: 1.3 G instructions for
5.2 s, under the 1.66 G cycles one 320 MHz core has. With three, under 1 G
each. Compute is a chip's instructions minus its UART wait loop: the raw
count of a later stage includes polling while it waits, so profile with
`PROFILE=DIR` and subtract `rust_main`, where the polling loop is inlined.

The nRF54H20's second Cortex-M33 (the radio core, 256 MHz) can take a stage
the same way over shared RAM. Pick the cut to match its clock
(`--cuts 0,8,23`), and keep in mind that it also runs the Bluetooth stack.

```bash
just citrinet-nrf54h-qemu                       # one chip, DSP kernels
just citrinet-nrf54h-qemu --no-default-features # portable kernels, same bits
just citrinet-nrf54h-pipeline 2                 # two simulated chips over UART
```

Regenerate `data/` after changing the model or kernels:

```bash
cargo run -p rlx-citrinet --release --example mcu_vectors -- \
    --rct .cache/citrinet-mcu/c256-int8.rct --wav assets/harness/jfk_16k.wav \
    --out crates/rlx-citrinet-nrf54h/data --f32s 180224 --i8s 4096
```

Not run on silicon. On a board, `link.x`'s `ORIGIN`s move to the nRF54H20's
addresses, the model is programmed into the EXMIF flash, and `Uart` becomes
the SPIM/UARTE driver between chips.
