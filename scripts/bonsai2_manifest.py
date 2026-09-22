#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# SPDX-License-Identifier: GPL-3.0-only
"""Build the `Ternary-Bonsai-2-27B` metadata fixture used by `rlx-qwen35`.

The published model is 5.95 GB (PTQ1_0) or 7.21 GB (PQ2_0), which is far
too much to pull just to check that `prism.hadamard.*` parses and that
the folded weight list lines up with the tensors it names. It does not
have to be pulled: a GGUF header carries every tensor's name, shape and
dtype ahead of the data, so an HTTP range request over the first few MB
has everything the loader reasons about.

This writes a **header-only** GGUF (no data section) carrying the real
`general.*` / `qwen35.*` / `prism.hadamard.*` metadata and all 851 real
tensor infos, with the two tokenizer arrays (248320 tokens + 247587
merges, ~11 MB) dropped. The result is ~250 KB and parses with
`GgufFile::header_from_path`.

    python3 scripts/bonsai2_manifest.py \
        crates/rlx-qwen35/tests/fixtures/bonsai2_manifest.gguf
"""
import argparse, struct, sys, urllib.request

REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
FILE = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
URL = f"https://huggingface.co/{REPO}/resolve/main/{FILE}"

# Dropped from the fixture: huge, and nothing here reads them.
SKIP_PREFIX = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges",
               "tokenizer.ggml.token_type", "tokenizer.chat_template")

T_STR, T_ARR = 8, 9
FMT = {0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2), 4: ('<I', 4),
       5: ('<i', 4), 6: ('<f', 4), 7: ('<?', 1), 10: ('<Q', 8), 11: ('<q', 8),
       12: ('<d', 8)}


class RangeReader:
    def __init__(self, url, chunk=4 << 20):
        self.url, self.buf, self.pos, self.chunk = url, b'', 0, chunk

    def _fetch(self, upto):
        while len(self.buf) < upto:
            lo = len(self.buf)
            hi = lo + max(self.chunk, upto - lo) - 1
            req = urllib.request.Request(self.url, headers={'Range': f'bytes={lo}-{hi}'})
            with urllib.request.urlopen(req) as r:
                d = r.read()
            if not d:
                raise EOFError("short read")
            self.buf += d

    def read(self, n):
        self._fetch(self.pos + n)
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b


def u32(r): return struct.unpack('<I', r.read(4))[0]
def u64(r): return struct.unpack('<Q', r.read(8))[0]
def rstr(r): return r.read(u64(r)).decode('utf-8', 'replace')


def rval(r, t):
    if t == T_STR:
        return (T_STR, rstr(r))
    if t == T_ARR:
        et = u32(r)
        n = u64(r)
        return (T_ARR, (et, [rval(r, et)[1] for _ in range(n)]))
    f, sz = FMT[t]
    return (t, struct.unpack(f, r.read(sz))[0])


def wstr(s):
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def wval(t, v):
    if t == T_STR:
        return wstr(v)
    if t == T_ARR:
        et, items = v
        out = struct.pack('<I', et) + struct.pack('<Q', len(items))
        for it in items:
            out += wval(et, it)
        return out
    f, _ = FMT[t]
    return struct.pack(f, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--url', default=URL)
    a = ap.parse_args()

    r = RangeReader(a.url)
    if r.read(4) != b'GGUF':
        sys.exit('not a GGUF')
    ver = u32(r)
    n_tensors = u64(r)
    n_kv = u64(r)

    kv = []
    for _ in range(n_kv):
        k = rstr(r)
        t = u32(r)
        v = rval(r, t)
        if not k.startswith(SKIP_PREFIX):
            kv.append((k, v[0], v[1]))

    tensors = []
    for _ in range(n_tensors):
        name = rstr(r)
        nd = u32(r)
        dims = [u64(r) for _ in range(nd)]
        typ = u32(r)
        off = u64(r)
        tensors.append((name, dims, typ, off))

    body = b'GGUF' + struct.pack('<I', ver)
    body += struct.pack('<Q', len(tensors)) + struct.pack('<Q', len(kv))
    for k, t, v in kv:
        body += wstr(k) + struct.pack('<I', t) + wval(t, v)
    for name, dims, typ, off in tensors:
        body += wstr(name) + struct.pack('<I', len(dims))
        for d in dims:
            body += struct.pack('<Q', d)
        body += struct.pack('<I', typ) + struct.pack('<Q', off)

    with open(a.out, 'wb') as f:
        f.write(body)
    print(f'wrote {a.out}: {len(body)} bytes, {len(tensors)} tensors, {len(kv)} kv '
          f'(read {r.pos} bytes of the {FILE} header)')


if __name__ == '__main__':
    main()
