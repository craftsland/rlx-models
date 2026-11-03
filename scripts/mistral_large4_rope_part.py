#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# SPDX-License-Identifier: GPL-3.0-only
"""Range-fetch `rope_freqs.weight` (128 bytes) from the `UD-IQ1_M` split and
write it as a one-tensor stand-in split part, so a directory holding only some
real parts (e.g. 00001 + 00007) opens as a split with RoPE tables.

    python3 scripts/mistral_large4_rope_part.py DIR/ml4-iq1m-rope-freqs-part.gguf
"""
import sys, struct, importlib.util, urllib.request
import os
spec = importlib.util.spec_from_file_location("sub", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mistral_large4_subset.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
out = sys.argv[1]
for no in range(2, 8):
    url = f"https://huggingface.co/{m.REPO}/resolve/main/UD-IQ1_M/Mistral-Large-4-1T-A52B-UD-IQ1_M-{no:05d}-of-00007.gguf"
    kvs, ts, ds, al = m.header(url)
    hit = [t for t in ts if t[0] == "rope_freqs.weight"]
    if hit: break
name, dims, ty, off = hit[0]
n = m.nbytes(dims, ty)
req = urllib.request.Request(url, headers={"Range": f"bytes={ds+off}-{ds+off+n-1}"})
data = urllib.request.urlopen(req).read()
assert len(data) == n and ty == 0
print(f"rope_freqs from shard {no}: dims {dims}, first {struct.unpack('<4f', data[:16])}")
def s(b): return struct.pack("<Q", len(b)) + b
kv = [(b"split.no", 2, struct.pack("<H", no - 1)), (b"split.count", 2, struct.pack("<H", 7))]
hdr = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv))
for k, t, raw in kv: hdr += s(k) + struct.pack("<I", t) + raw
hdr += s(name.encode()) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, 0)
hdr += b"\0" * ((-len(hdr)) % 32)
open(out, "wb").write(hdr + data)
