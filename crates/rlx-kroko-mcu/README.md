# rlx-kroko-mcu

Bare-metal RISC-V firmware (QEMU `virt`, rv32imafc, 256 MiB) running
[`rlx-kroko-core`](../rlx-kroko-core). Only the resident 2.3 MB of the model
is parsed in place; every Zipformer layer is copied into a page buffer
through the `Pager` each chunk, as it would be from NAND.

```text
kroko streaming: 83200 samples, model 99031276 B (resident 2266584 B, page 10086428 B), scratch 17288324 B
text: Ask not what your country can do for you. Ask what you can do for your country
ids match, encoder fnv 0x3c63f54f match
INSNS 124038611200
```

The 99 MB int4 model is linked from `.cache/kroko-mcu/en64-int4.rkr` (or
`RKR_MODEL=/path`). Regenerate `data/` with
`cargo run -p rlx-kroko --release --example mcu_vectors -- --rkr … --wav … --out crates/rlx-kroko-mcu/data`.
