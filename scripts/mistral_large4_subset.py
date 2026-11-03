#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# SPDX-License-Identifier: GPL-3.0-only
"""Build a small standalone `mistral-large4` GGUF from the published split, by
HTTP range requests.

Mistral-Large-4-1T-A52B is 233 GB at its smallest quant. The default here,
blocks 35, 38 and 39 with the first 6 of 128 experts, is about 1.1 GB, and
between them those blocks cover every attention variant the model has:

  * 35 — NoPE, owns its KV (and is the source the late layers share), full indexer;
  * 38 — RoPE, owns its KV, **shared** indexer (reuses the previous selection);
  * 39 — NoPE, **no** `attn_kv_a_mqa` (attends to layer 35's latent), full indexer.

A quantized expert bank is `[experts, out, in]` with the expert axis
outermost, so the first N experts are a byte prefix. The router
(`ffn_gate_inp`, F32 `[hidden, experts]` in GGML order) is narrowed to the
same N rows, and `expert_count` is rewritten to N, so the result routes
consistently — a valid N-expert model for block tests, not the real routing.

    python3 scripts/mistral_large4_subset.py weights/lm/mistral-large4-subset/ml4-35-38-39.gguf
    python3 scripts/mistral_large4_reference.py --gguf <that file> /tmp/ml4-real   # needs gguf-py
    RLX_ML4_REAL_DIR=/tmp/ml4-real cargo test -p rlx-mistral-large4 --release --test real_weights
"""
import argparse
import os
import struct
import urllib.request

REPO = "unsloth/Mistral-Large-4-1T-A52B-GGUF"
# ggml type -> (block elements, block bytes)
BLOCK = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144),
         13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98),
         19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
         29: (256, 56), 30: (1, 2)}
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


def rval(r, t):
    """(type, raw bytes, decoded scalar or None)."""
    if t == T_STR:
        s = rstr(r)
        return struct.pack("<Q", len(s)) + s, None
    if t == T_ARR:
        et, n = u32(r), u64(r)
        out = struct.pack("<IQ", et, n)
        for _ in range(n):
            out += rval(r, et)[0]
        return out, None
    raw = r.read(struct.calcsize(FMT[t]))
    return raw, struct.unpack(FMT[t], raw)[0]


def header(url):
    r = RangeReader(url)
    assert r.read(4) == b"GGUF"
    ver, nt, nkv = u32(r), u64(r), u64(r)
    kvs = []
    align = 32
    for _ in range(nkv):
        k = rstr(r)
        t = u32(r)
        raw, val = rval(r, t)
        if k == b"general.alignment":
            align = val
        kvs.append((k, t, raw))
    ts = []
    for _ in range(nt):
        name = rstr(r).decode()
        nd = u32(r)
        dims = [u64(r) for _ in range(nd)]
        ty, off = u32(r), u64(r)
        ts.append((name, dims, ty, off))
    data_start = (r.pos + align - 1) // align * align
    return kvs, ts, data_start, align


def nbytes(dims, ty):
    n = 1
    for d in dims:
        n *= d
    blk, sz = BLOCK[ty]
    assert n % blk == 0
    return n // blk * sz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--blocks", default="35,38,39")
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--quant", default="UD-IQ1_S")
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    blocks = {int(b) for b in a.blocks.split(",")}
    base = f"https://huggingface.co/{REPO}/resolve/main/{a.quant}"
    url = lambda i: f"{base}/Mistral-Large-4-1T-A52B-{a.quant}-{i:05d}-of-{a.shards:05d}.gguf"

    meta = None
    plan = []  # (name, dims, ty, shard_url, src_abs, nbytes)
    for i in range(1, a.shards + 1):
        kvs, ts, data_start, _ = header(url(i))
        if meta is None:
            meta = kvs
        for name, dims, ty, off in ts:
            keep = name == "rope_freqs.weight"
            if name.startswith("blk.") and int(name.split(".")[1]) in blocks:
                keep = True
            if not keep:
                continue
            dims = list(dims)
            if name.endswith("_exps.weight"):
                dims[2] = min(a.experts, dims[2])
            elif name.endswith("ffn_gate_inp.weight"):
                dims[1] = min(a.experts, dims[1])  # rows are contiguous per expert
            plan.append((name, dims, ty, url(i), data_start + off, nbytes(dims, ty)))
    total = sum(p[5] for p in plan)
    print(f"{len(plan)} tensors, {total / 1e6:.1f} MB")
    if a.dry_run:
        return

    drop = (b"tokenizer.", b"split.", b"quantize.")
    kv_out = []
    for k, t, raw in meta:
        if k.startswith(drop):
            continue
        if k == b"mistral-large4.expert_count":
            raw = struct.pack(FMT[t], a.experts)
        kv_out.append((k, t, raw))
    align = 32
    hdr = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(kv_out))
    for k, t, raw in kv_out:
        hdr += struct.pack("<Q", len(k)) + k + struct.pack("<I", t) + raw
    off = 0
    dst = []
    for name, dims, ty, *_ , sz in plan:
        nb = name.encode()
        hdr += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
        hdr += b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, off)
        dst.append(off)
        off = (off + sz + align - 1) // align * align
    hdr += b"\x00" * ((-len(hdr)) % align)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = 0
    with open(a.out, "wb") as f:
        f.write(hdr)
        f.truncate(len(hdr) + off)
        for (name, _, _, u, src, sz), d in zip(plan, dst):
            req = urllib.request.Request(u, headers={"Range": f"bytes={src}-{src + sz - 1}"})
            with urllib.request.urlopen(req) as r:
                buf = r.read()
            assert len(buf) == sz, f"{name}: short read {len(buf)} != {sz}"
            f.seek(len(hdr) + d)
            f.write(buf)
            done += sz
            print(f"  {done / 1e6:8.1f} / {total / 1e6:.1f} MB  {name}", flush=True)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
