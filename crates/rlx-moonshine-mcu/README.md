# rlx-moonshine-mcu

Bare-metal RISC-V firmware (QEMU `virt`, rv32imafc = ESP32-P4 ISA) running
[`rlx-moonshine-core`](../rlx-moonshine-core). It transcribes the JFK clip
linked into the image and checks the token ids and an FNV-1a hash of the
encoder output against host-computed vectors:

```text
moonshine-streaming int8/int4: 83200 samples, model 37390296 B, scratch 2684100 B
text: Ask not what your country can do for you. Ask what you can do for your country.
ids match, encoder hidden fnv 0xb0c8d7a4 match
INSNS 20714118086
```

The 37 MB model is linked in from `.cache/moonshine-mcu/tiny.rms`, or from
`RMS_MODEL=/path`. Regenerate `data/` with
`cargo run -p rlx-moonshine-streaming --release --example mcu_vectors -- --rms … --wav … --out crates/rlx-moonshine-mcu/data`.
