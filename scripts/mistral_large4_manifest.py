#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# SPDX-License-Identifier: GPL-3.0-only
"""Fixtures for rlx-mistral-large4 from the published GGUF, by HTTP range reads.

Mistral-Large-4-1T-A52B's smallest quant is 233 GB. Its contract — metadata
and every tensor's name, shape and type — is a few MB of headers across the
split's shards, so this reads only those and writes:

  * `mistral_large4_meta.gguf` — a header-only GGUF (zero tensors) carrying
    every `general.*` / `mistral-large4.*` key of shard 1, verbatim. The
    `tokenizer.*` keys are dropped: 200k-entry vocab/merge arrays are most of
    the header and the text model does not read them.
  * `mistral_large4_tensors.tsv` — `name<TAB>ggml dims (innermost first)<TAB>type`
    for all tensors of all shards.

    python3 scripts/mistral_large4_manifest.py crates/rlx-mistral-large4/tests/fixtures
"""
import argparse
import os
import struct
import urllib.request

REPO = "unsloth/Mistral-Large-4-1T-A52B-GGUF"
TYPES = {0: "F32", 1: "F16", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
         16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
         22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 30: "BF16"}
T_STR, T_ARR = 8, 9
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class RangeReader:
    def __init__(self, url, chunk=4 << 20):
        self.url, self.buf, self.pos, self.chunk = url, b"", 0, chunk

    def read(self, n):
        while len(self.buf) < self.pos + n:
            lo = len(self.buf)
            hi = lo + max(self.chunk, self.pos + n - lo) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req) as r:
                d = r.read()
            if not d:
                raise EOFError(self.url)
            self.buf += d
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b


def u32(r):
    return struct.unpack("<I", r.read(4))[0]


def u64(r):
    return struct.unpack("<Q", r.read(8))[0]


def rstr(r):
    return r.read(u64(r))


def rval_raw(r, t):
    """Read one value of type `t`, returning its exact on-disk bytes."""
    if t == T_STR:
        s = rstr(r)
        return struct.pack("<Q", len(s)) + s
    if t == T_ARR:
        et, n = u32(r), u64(r)
        out = struct.pack("<IQ", et, n)
        for _ in range(n):
            out += rval_raw(r, et)
        return out
    return r.read(struct.calcsize(FMT[t]))


def header(url):
    r = RangeReader(url)
    assert r.read(4) == b"GGUF"
    ver, nt, nkv = u32(r), u64(r), u64(r)
    kvs = []
    for _ in range(nkv):
        k = rstr(r)
        t = u32(r)
        kvs.append((k, t, rval_raw(r, t)))
    ts = []
    for _ in range(nt):
        name = rstr(r).decode()
        nd = u32(r)
        dims = [u64(r) for _ in range(nd)]
        ty = u32(r)
        u64(r)  # offset
        ts.append((name, dims, ty))
    return ver, kvs, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--quant", default="UD-IQ1_S")
    ap.add_argument("--shards", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = f"https://huggingface.co/{REPO}/resolve/main/{a.quant}"
    tensors, meta = [], None
    for i in range(1, a.shards + 1):
        url = f"{base}/Mistral-Large-4-1T-A52B-{a.quant}-{i:05d}-of-{a.shards:05d}.gguf"
        ver, kvs, ts = header(url)
        if meta is None:
            meta = (ver, kvs)
        tensors += ts
        print(f"shard {i}: {len(ts)} tensors")
    ver, kvs = meta
    keep = [kv for kv in kvs if not kv[0].startswith((b"tokenizer.", b"split.", b"quantize."))]
    blob = b"GGUF" + struct.pack("<IQQ", ver, 0, len(keep))
    for k, t, raw in keep:
        blob += struct.pack("<Q", len(k)) + k + struct.pack("<I", t) + raw
    with open(os.path.join(a.out, "mistral_large4_meta.gguf"), "wb") as f:
        f.write(blob)
    with open(os.path.join(a.out, "mistral_large4_tensors.tsv"), "w") as f:
        for name, dims, ty in tensors:
            f.write(f"{name}\t{','.join(map(str, dims))}\t{TYPES.get(ty, ty)}\n")
    print(f"{len(keep)} metadata keys, {len(tensors)} tensors -> {a.out}")


if __name__ == "__main__":
    main()
