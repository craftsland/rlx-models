"""Range-fetch one real MoE layer's *routed* experts and dump its reference output.

The full bank is 384 experts — 6.8 GB for one layer — but a token only uses
`num_experts_per_tok` of them. So this fetches the gate first, runs the real
routing to find out *which* experts a given hidden state needs, and fetches only
those. About 150 MB instead of 6.8 GB, and the weights are the real quantized
bytes with the real routing decisions rather than a synthetic stand-in.

    python3 dump_real_moe.py --layer 0 --out-weights /tmp/moe.safetensors \
        --out-ref ../../tests/fixtures/dsv41_real_moe_ref.json
"""

import argparse
import json
import struct

import numpy as np
import torch

from fetch_subset import BASE, INDEX, curl, header

# e2m1 nibbles are 4 bits *including* the sign, so the table is 16 wide —
# indexing an 8-entry magnitude table drops every negative weight.
E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def e8m0(b: np.ndarray) -> np.ndarray:
    return np.ldexp(1.0, b.astype(np.int32) - 127).astype(np.float32)


def read_tensors(names, index):
    """Fetch `names` and return `{name: (bytes, dtype, shape)}`, one range per tensor."""
    by_shard = {}
    for n in names:
        by_shard.setdefault(index["weight_map"][n], []).append(n)
    out = {}
    for shard, want in by_shard.items():
        hdr, base = header(shard)
        for n in want:
            s, e = hdr[n]["data_offsets"]
            out[n] = (curl(BASE + shard, f"{base + s}-{base + e - 1}"), hdr[n]["dtype"], hdr[n]["shape"])
            print(f"  {n}: {(e - s) / 1e6:.1f} MB")
    return out


def to_f32(raw, dtype, shape):
    if dtype == "BF16":
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape).float().numpy()
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    raise ValueError(dtype)


def dequant_fp4(codes, scales, rows, cols_packed, block):
    """FP4 nibble pairs with one E8M0 scale per row per `block` logical columns."""
    c = np.frombuffer(codes, dtype=np.uint8).reshape(rows, cols_packed)
    lo, hi = c & 0x0F, c >> 4
    v = np.empty((rows, cols_packed * 2), dtype=np.float32)
    v[:, 0::2], v[:, 1::2] = E2M1[lo], E2M1[hi]
    s = e8m0(np.frombuffer(scales, dtype=np.uint8).reshape(rows, -1))
    return v * np.repeat(s, block, axis=1)[:, : v.shape[1]]


def dequant_fp8_tile(codes, scales, rows, cols, block):
    w = (
        torch.frombuffer(bytearray(codes), dtype=torch.float8_e4m3fn)
        .reshape(rows, cols)
        .float()
        .numpy()
    )
    s = e8m0(np.frombuffer(scales, dtype=np.uint8).reshape(-1, (cols + block - 1) // block))
    s = np.repeat(np.repeat(s, block, axis=0), block, axis=1)[:rows, :cols]
    return (w * s).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", type=int, default=2, help="hidden rows to route")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--out-weights", required=True)
    ap.add_argument("--out-ref", required=True)
    a = ap.parse_args()

    cfg = json.load(open("config.json"))
    t = cfg.get("text_config", cfg)
    dim = t["hidden_size"]
    inter = t["moe_intermediate_size"]
    top_k = t.get("num_experts_per_tok", 8)
    n_shared = t.get("n_shared_experts", 1)
    swiglu_limit = t.get("swiglu_limit", 0.0)
    route_scale = t.get("routed_scaling_factor", 1.0)
    norm_topk = t.get("norm_topk_prob", True)
    L = a.layer
    index = json.loads(curl(INDEX))

    print("fetching the gate")
    gate = read_tensors([f"layers.{L}.ffn.gate.weight", f"layers.{L}.ffn.gate.bias"], index)
    gw = to_f32(*gate[f"layers.{L}.ffn.gate.weight"])
    gb = to_f32(*gate[f"layers.{L}.ffn.gate.bias"])

    # a deterministic hidden state, the same rule SyntheticLoader uses so the
    # Rust side can reproduce it without carrying the values
    rng = np.random.default_rng(0)
    x = (rng.random((a.rows, dim), dtype=np.float32) - 0.5) * 2.0 / np.sqrt(dim)

    scores = np.sqrt(np.log1p(np.exp(-np.abs(x @ gw.T))) + np.maximum(x @ gw.T, 0.0))
    route = scores + gb
    idx = np.argsort(-route, axis=1, kind="stable")[:, :top_k]
    w = np.take_along_axis(scores, idx, axis=1)
    if norm_topk and top_k > 1:
        w = w / (w.sum(axis=1, keepdims=True) + 1e-20)
    w = w * route_scale
    chosen = sorted(set(int(e) for e in idx.reshape(-1)))
    print(f"routing chose {len(chosen)} distinct experts: {chosen}")

    names = []
    for e in chosen:
        for p in ("w1", "w2", "w3"):
            names += [f"layers.{L}.ffn.experts.{e}.{p}.weight", f"layers.{L}.ffn.experts.{e}.{p}.scale"]
    for p in ("w1", "w2", "w3"):
        names += [f"layers.{L}.ffn.shared_experts.{p}.weight", f"layers.{L}.ffn.shared_experts.{p}.scale"]
    print(f"fetching {len(names)} tensors")
    raw = read_tensors(names, index)

    def expert(e, p):
        wr, dt, sh = raw[f"layers.{L}.ffn.experts.{e}.{p}.weight"]
        sr, _, _ = raw[f"layers.{L}.ffn.experts.{e}.{p}.scale"]
        assert dt == "I8", dt
        return dequant_fp4(wr, sr, sh[0], sh[1], a.block)

    def shared(p):
        wr, dt, sh = raw[f"layers.{L}.ffn.shared_experts.{p}.weight"]
        sr, _, _ = raw[f"layers.{L}.ffn.shared_experts.{p}.scale"]
        assert dt == "F8_E4M3", dt
        return dequant_fp8_tile(wr, sr, sh[0], sh[1], a.block)

    def clamped_swiglu(g, u):
        if swiglu_limit > 0:
            g = np.minimum(g, swiglu_limit)
            u = np.clip(u, -swiglu_limit, swiglu_limit)
        return (g / (1.0 + np.exp(-g))) * u

    out = np.zeros((a.rows, dim), dtype=np.float32)
    for r in range(a.rows):
        for k in range(top_k):
            e = int(idx[r, k])
            g = x[r] @ expert(e, "w1").T
            u = x[r] @ expert(e, "w3").T
            out[r] += w[r, k] * (clamped_swiglu(g, u) @ expert(e, "w2").T)
    sg, su, sd = shared("w1"), shared("w3"), shared("w2")
    out += clamped_swiglu(x @ sg.T, x @ su.T) @ sd.T
    print(f"shared expert width {sg.shape[0]} (n_shared={n_shared}, inter={inter})")

    # one local safetensors holding exactly what was fetched
    hdr, blob, off = {}, bytearray(), 0
    for n, (b, dt, sh) in {**gate, **raw}.items():
        hdr[n] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    hj = json.dumps(hdr).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(a.out_weights, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(blob)
    print(f"wrote {a.out_weights} ({(8 + len(hj) + len(blob)) / 1e6:.1f} MB)")

    with open(a.out_ref, "w") as f:
        json.dump(
            {
                "layer": L,
                "rows": a.rows,
                "dim": dim,
                "top_k": top_k,
                "block": a.block,
                "chosen_experts": chosen,
                "top_idx": idx.astype(int).tolist(),
                "top_w": w.astype(float).tolist(),
                "x": x.astype(float).reshape(-1).tolist(),
                "out": out.astype(float).reshape(-1).tolist(),
            },
            f,
        )
    print(f"wrote {a.out_ref}")


if __name__ == "__main__":
    main()
