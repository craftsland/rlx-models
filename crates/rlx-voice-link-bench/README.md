# rlx-voice-link-bench

Bare-metal Cortex-M4F firmware (the nRF52840's core) that measures what the
voice link's device side costs per 16 ms frame. It runs on QEMU's MPS2 AN505,
which executes ARMv7E-M code unchanged, and is profiled with
`rlx-citrinet-nrf54h/tools/libpcprof.dylib`.

| stage | instructions / frame | share of a 64 MHz nRF52840 | runs |
|---|---|---|---|
| `gate::Gate::push` | 1,242 | 0.12% | all day |
| `Packetizer::packet` (IMA-ADPCM) | 15,825 | 1.5% | while an utterance is open |
| TEN-VAD `Vad::process_i16` | 1,020,757 | ≈100% | on the phone instead |

```bash
just voice-link-bench
```

The numbers feed `rlx-voice-link-host`'s `examples/energy_model.rs`.
