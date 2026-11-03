#!/usr/bin/env python3
"""Compare rlx-kroko-core with the original Kroko ONNX graphs.

Feeds the fbank features `mcu_check --dump DIR` wrote to Kroko's own
`encoder.onnx` chunk by chunk (states carried over, as sherpa-onnx does),
compares every chunk's `encoder_out` with the core's, and runs greedy search
with `decoder.onnx` / `joiner.onnx`.

    python3 scripts/kroko_onnx_reference.py .cache/kroko/en64 /tmp/kroko_dbg
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

model_dir, dump = Path(sys.argv[1]), Path(sys.argv[2])
enc = ort.InferenceSession(str(model_dir / "encoder.onnx"), providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(str(model_dir / "decoder.onnx"), providers=["CPUExecutionProvider"])
joi = ort.InferenceSession(str(model_dir / "joiner.onnx"), providers=["CPUExecutionProvider"])
meta = enc.get_modelmeta().custom_metadata_map
T, shift = int(meta["T"]), int(meta["decode_chunk_len"])
fb = np.fromfile(dump / "fbank.f32", np.float32).reshape(-1, 80)

def zeros(i):
    shape = [d if isinstance(d, int) else 1 for d in i.shape]
    dt = np.int64 if "int64" in i.type else np.float32
    return np.zeros(shape, dt)

state = {i.name: zeros(i) for i in enc.get_inputs() if i.name != "x"}
outs = [o.name for o in enc.get_outputs()]
blank, ctx = 0, [-1, 0]
def dec_out(c):
    y = np.array([c], np.int64)  # -1 = no token; the graph masks it
    return dec.run(None, {"y": y})[0]
d = dec_out(ctx)
ids, worst = [], 1.0
for k, off in enumerate(range(0, fb.shape[0], shift)):
    x = np.zeros((1, T, 80), np.float32)
    n = min(T, fb.shape[0] - off)
    x[0, :n] = fb[off:off + n]
    res = enc.run(outs, {"x": x, **state})
    e = res[0][0]
    for name, val in zip(outs[1:], res[1:]):
        state[name.replace("new_", "", 1)] = val
    p = dump / f"enc_{k}.f32"
    if p.exists():
        m = np.fromfile(p, np.float32).reshape(e.shape)
        cos = float((e * m).sum() / np.sqrt((e * e).sum() * (m * m).sum()))
        worst = min(worst, cos)
        print(f"chunk {k}: encoder cosine {cos:.5f}  max|d| {np.abs(e - m).max():.3f}")
    valid = min(e.shape[0], -(-min(shift, fb.shape[0] - off) // 4))
    for t in range(valid):
        logit = joi.run(None, {"encoder_out": e[t:t + 1], "decoder_out": d})[0][0]
        tok = int(logit.argmax())
        if tok in (blank, 2):
            continue
        ids.append(tok)
        ctx = [ctx[1], tok]
        d = dec_out(ctx)
core = (dump / "ids.txt").read_text().split()
print("onnx ids:", " ".join(map(str, ids)))
print("core ids:", " ".join(core))
print("identical" if list(map(int, core)) == ids else "DIFFER", f"(worst chunk cosine {worst:.5f})")
