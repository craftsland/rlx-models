"""Decode real quantized tensors — one per scale layout — as ground truth.

Deliberately uses torch's OWN fp8 decoders and the FP4 table lifted verbatim
from the reference `convert.py`, so this stays independent of `dsv41_quant.rs`
rather than a second copy of it.

    python3 fetch_subset.py --out /tmp/dsv41w/quant.safetensors --what quant
    python3 dump_real_quant.py /tmp/dsv41w/quant.safetensors
"""
import json, os, sys
import torch

# verbatim from inference/convert.py
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
BLOCK = 32
N_SAMPLES = 1500


def decode(codes, scales):
    """`convert.py`'s `cast_e2m1fn_to_e4m3fn` unpacking, plus the scale layout."""
    rows = codes.shape[0]
    if codes.dtype == torch.int8:
        u = codes.view(torch.uint8)
        low, high = u & 0x0F, (u >> 4) & 0x0F
        vals = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(1)
    else:
        vals = codes.float()          # torch decodes e4m3fn itself
    s = scales.float()                # and e8m0fnu
    # one scale row per weight row, or one per BLOCK rows
    rep = 1 if s.shape[0] == rows else BLOCK
    s = s.repeat_interleave(rep, 0)[:rows].repeat_interleave(BLOCK, 1)[:, :vals.shape[1]]
    return vals * s


def main():
    from safetensors import safe_open
    with safe_open(sys.argv[1], framework="pt") as f:
        raw = {k: f.get_tensor(k) for k in f.keys()}

    out = {}
    for name in sorted(k for k in raw if k.endswith(".weight")):
        sk = name[: -len(".weight")] + ".scale"
        assert sk in raw, f"{name} has no scale"
        t = decode(raw[name], raw[sk])
        flat = t.reshape(-1)
        # ~N_SAMPLES values, on an ODD stride so they cannot all land at the same
        # offset within a 32-wide scale block
        stride = max(1, flat.numel() // N_SAMPLES) | 1
        out[name] = {
            "shape": list(t.shape),
            "stored_dtype": str(raw[name].dtype).replace("torch.", ""),
            "scale_shape": list(raw[sk].shape),
            "layout": "row_groups" if raw[sk].shape[0] == t.shape[0] else "tile",
            "absmean": float(flat.abs().mean()),
            "max": float(flat.abs().max()),
            "stride": stride,
            "samples": flat[::stride].tolist(),
        }
        print(f"{name:<40} {tuple(t.shape)} {out[name]['layout']:<10} "
              f"absmean {out[name]['absmean']:.5g}  n_samples {len(out[name]['samples'])}")

    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(here, "..", "..", "tests", "fixtures", "dsv41_real_quant_digest.json")
    json.dump(out, open(dst, "w"), separators=(",", ":"))
    print("wrote", os.path.relpath(dst))


if __name__ == "__main__":
    main()
