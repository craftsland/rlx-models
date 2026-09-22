"""Range-fetch a handful of named tensors from the DeepSeek-V4.1 shards into one
small local safetensors file.

The shards are 10 GB each and there are 48 of them, but a safetensors header
gives every tensor's byte range, so a real-weight test only has to pay for the
tensors it actually touches. `--layer 0` is ~130 MB.

    python3 fetch_subset.py --out /tmp/dsv41_layer0.safetensors --layer 0
"""
import argparse, json, os, struct, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
INDEX = f"https://huggingface.co/{REPO}/raw/main/model.safetensors.index.json"


def curl(url, rng=None, timeout=600):
    cmd = ["curl", "-fsSL", "--max-time", str(timeout)]
    if rng:
        cmd += ["-r", rng]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl {url} {rng}: {r.stderr[:200]!r}")
    return r.stdout


def header(shard):
    raw = curl(BASE + shard, "0-7")
    n = struct.unpack("<Q", raw[:8])[0]
    return json.loads(curl(BASE + shard, f"8-{8 + n - 1}")[:n]), n + 8


def quant_sample_tensors():
    """One tensor per quantization layout, so a loader test can see real bytes of
    each. The Engram table is 98 GB, so only a slice of its rows is fetched (see
    `--engram-rows`)."""
    return [
        "layers.0.attn.wq_a.weight",          # FP8, one scale per 32x32 tile
        "layers.0.ffn.experts.0.w1.weight",   # FP4 nibble pairs, row-group scales
    ]


def layer_tensors(layer, index):
    """Everything layer `layer`'s attention block needs, plus its HC parameters.

    A KV-source layer also carries a compressor, and an index source an indexer;
    which tensors exist is exactly what `DeepseekV41Spec::is_kv_source` /
    `is_index_source` predict, so the set is derived from the checkpoint index
    rather than from a hard-coded layer list.
    """
    lp = f"layers.{layer}"
    names = [
        f"{lp}.attn_norm.weight",
        f"{lp}.hc_attn_fn", f"{lp}.hc_attn_base", f"{lp}.hc_attn_scale",
        f"{lp}.attn.wq_a.weight", f"{lp}.attn.q_norm.weight", f"{lp}.attn.wq_b.weight",
        f"{lp}.attn.wkv.weight", f"{lp}.attn.kv_norm.weight",
        f"{lp}.attn.attn_sink", f"{lp}.attn.wo_a.weight", f"{lp}.attn.wo_b.weight",
    ]
    for extra in [
        f"{lp}.attn.compressor.wkv.weight", f"{lp}.attn.compressor.wgate.weight",
        f"{lp}.attn.compressor.norm.weight",
        f"{lp}.attn.indexer.wk.weight", f"{lp}.attn.indexer.k_norm.weight",
        f"{lp}.attn.indexer.wq_b.weight", f"{lp}.attn.indexer.weights_proj.weight",
    ]:
        if extra in index:
            names.append(extra)
    scales = [n[: -len("weight")] + "scale" for n in names if n.endswith(".weight")]
    return names + [sk for sk in scales if sk in index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, action="append", default=None,
                    help="repeatable; layer 0 is sliding-window only, layer 2 is a "
                         "KV + index source with a compressor and an indexer")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--what", default="attn", choices=["attn", "quant", "all"],
                    help="attn = one layer's attention block; quant = one tensor "
                         "per quantization layout plus a slice of the Engram table")
    ap.add_argument("--tokenizer", action="store_true",
                    help="also fetch tokenizer.json (6.4 MB) for the Engram token-map test")
    ap.add_argument("--engram-rows", default="0:64",
                    help="slice of layers.1.engram.embed to include (row-wise FP8)")
    ap.add_argument("--embed-rows", default="1000:1160",
                    help="slice of embed.weight to include, so the layer can be "
                         "driven by real token embeddings instead of noise")
    args = ap.parse_args()

    index = json.loads(curl(INDEX))["weight_map"]
    want = []
    if args.what in ("attn", "all"):
        for l in (args.layer or [0]):
            want += layer_tensors(l, index)
    if args.what in ("quant", "all"):
        want += quant_sample_tensors()
        want += [n.replace(".weight", ".scale") for n in quant_sample_tensors()]
    print(f"resolving {len(want)} tensors", file=sys.stderr)
    shards = sorted({index[n] for n in want})
    heads = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for sh, (h, base) in zip(shards, ex.map(header, shards)):
            heads[sh] = (h, base)

    def fetch_rows(name, lo, hi, out_name, itemsize=2):
        """Row-major tensors have contiguous row ranges, so a slice is one range
        request even when the whole tensor is 98 GB."""
        sh = index[name]
        if sh not in heads:
            heads[sh] = header(sh)
        h, base = heads[sh]
        s0, _ = h[name]["data_offsets"]
        dim = h[name]["shape"][1]
        a = base + s0 + lo * dim * itemsize
        b = base + s0 + hi * dim * itemsize - 1
        return (out_name, h[name]["dtype"], [hi - lo, dim], curl(BASE + sh, f"{a}-{b}"))

    def fetch(name):
        sh = index[name]
        h, base = heads[sh]
        s, e = h[name]["data_offsets"]
        blob = curl(BASE + sh, f"{base + s}-{base + e - 1}")
        assert len(blob) == e - s, (name, len(blob), e - s)
        return name, h[name]["dtype"], h[name]["shape"], blob

    total = sum(h[n]["data_offsets"][1] - h[n]["data_offsets"][0]
                for n in want for h, _ in [heads[index[n]]])
    print(f"downloading {total / 1e6:.1f} MB", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        got = list(ex.map(fetch, want))
    if args.embed_rows and args.what in ("attn", "all"):
        lo, hi = (int(x) for x in args.embed_rows.split(":"))
        print(f"+ embed.weight rows {lo}..{hi}", file=sys.stderr)
        got.append(fetch_rows("embed.weight", lo, hi, "embed.rows", 2))
    if args.engram_rows and args.what in ("quant", "all"):
        lo, hi = (int(x) for x in args.engram_rows.split(":"))
        print(f"+ engram table rows {lo}..{hi}", file=sys.stderr)
        got.append(fetch_rows("layers.1.engram.embed.weight", lo, hi, "engram.rows.weight", 1))
        got.append(fetch_rows("layers.1.engram.embed.scale", lo, hi, "engram.rows.scale", 1))
    if args.tokenizer:
        print("+ tokenizer.json", file=sys.stderr)
        tk = os.path.join(os.path.dirname(os.path.abspath(args.out)), "tokenizer.json")
        with open(tk, "wb") as f:
            f.write(curl(f"https://huggingface.co/{REPO}/resolve/main/tokenizer.json"))

    hdr, off, blobs = {}, 0, []
    for name, dtype, shape, blob in got:
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(blob)]}
        off += len(blob)
        blobs.append(blob)
    raw = json.dumps(hdr, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, {len(hdr)} tensors)")


if __name__ == "__main__":
    main()
