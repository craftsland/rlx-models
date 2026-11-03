# rlx-citrinet-mcu

Bare-metal RISC-V firmware (QEMU `virt`, rv32imafc = ESP32-P4 ISA) running
[`rlx-citrinet-core`](../rlx-citrinet-core) on the JFK clip, checked against
host vectors (token ids plus an FNV-1a hash of the encoder output):

```text
citrinet int8: 83200 samples, model 13261940 B, scratch 2310020 B
text: ask not what your country can do for you ask what you can do for your country
ids match, encoder fnv 0x42059eb3 match
INSNS 6631948070
```

The model is linked from `.cache/citrinet-mcu/c256-int8.rct` (or
`RCT_MODEL=/path`). Regenerate `data/` with
`cargo run -p rlx-citrinet --release --example mcu_vectors -- --rct … --wav … --out crates/rlx-citrinet-mcu/data`.
