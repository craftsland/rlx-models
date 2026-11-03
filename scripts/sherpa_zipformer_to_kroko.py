#!/usr/bin/env python3
"""Convert a sherpa-onnx streaming Zipformer2 (icefall export) into the
directory `rlx-kroko`'s exporter reads, with audio.cpp's Kroko converter.

The converter recovers float weights from dynamically quantized ONNX graphs
(Kroko's community packages and sherpa-onnx `*.int8.onnx` exports are the
same format), but it lists Kroko-L's encoder layers by hand. This sets that
list from the encoder's `num_encoder_layers` metadata and writes `tokens.txt`
from `bpe.model` (sherpa-onnx ships only the SentencePiece model for some
releases).

    python3 -I scripts/sherpa_zipformer_to_kroko.py \\
        --converter ../audio.cpp/tools/community_models/convert_kroko_onnx.py \\
        --encoder ENC.int8.onnx --decoder DEC.int8.onnx --joiner JOIN.int8.onnx \\
        --bpe bpe.model OUT_DIR

The inputs are data: ONNX protobufs and a SentencePiece protobuf, parsed
without running any code from them.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path


def varint(b: bytes, i: int) -> tuple[int, int]:
    v, shift = 0, 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << shift
        if c < 0x80:
            return v, i
        shift += 7


def fields(b: bytes):
    """(field number, wire type, value) of one protobuf message."""
    i = 0
    while i < len(b):
        key, i = varint(b, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i)
        elif wt == 1:
            v, i = b[i : i + 8], i + 8
        elif wt == 2:
            n, i = varint(b, i)
            v, i = b[i : i + n], i + n
        elif wt == 5:
            v, i = b[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield num, wt, v


def sentencepiece_pieces(path: Path) -> list[str]:
    """ModelProto.pieces[*].piece, in id order."""
    pieces = []
    for num, wt, v in fields(path.read_bytes()):
        if num == 1 and wt == 2:
            piece = next((x for n, w, x in fields(v) if n == 1 and w == 2), b"")
            pieces.append(piece.decode("utf-8"))
    return pieces


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--converter", type=Path, required=True)
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--decoder", type=Path, required=True)
    ap.add_argument("--joiner", type=Path, required=True)
    ap.add_argument("--bpe", type=Path, help="bpe.model (if there is no tokens.txt)")
    ap.add_argument("--tokens", type=Path, help="tokens.txt")
    ap.add_argument("output", type=Path)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("convert_kroko_onnx", a.converter)
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)

    import onnx

    meta = {
        p.key: p.value
        for p in onnx.load(a.encoder, load_external_data=False).metadata_props
    }
    layers = [int(x) for x in meta["num_encoder_layers"].split(",")]
    conv.LAYER_PREFIXES = [
        f"encoder.encoders.0.layers.{i}" if s == 0 else f"encoder.encoders.{s}.encoder.layers.{i}"
        for s, n in enumerate(layers)
        for i in range(n)
    ]
    print(f"encoder layers per stack {layers}: {len(conv.LAYER_PREFIXES)} layers")

    # Older icefall exports name nodes by number (`Mul_1305`), and the
    # converter finds the stack downsamplers by name (`/downsample.../Mul`).
    # Find them by structure instead: a Mul by a constant of shape (k, 1, 1)
    # whose values are positive and sum to 1 (the exported softmax of
    # `SimpleDownsample.bias`), and give them the names the converter expects.
    original = conv.recover_zipformer_constants

    def recover(model):
        from onnx import numpy_helper

        init = {i.name: i for i in model.graph.initializer}
        k = 0
        for node in model.graph.node:
            if node.op_type != "Mul" or node.name.startswith("/downsample"):
                continue
            for x in node.input:
                t = init.get(x)
                if t is None or len(t.dims) != 3 or list(t.dims[1:]) != [1, 1] or t.dims[0] < 2:
                    continue
                v = numpy_helper.to_array(t)
                if (v > 0).all() and abs(float(v.sum()) - 1.0) < 1e-4:
                    node.name = f"/downsample_{k}/Mul"
                    k += 1
                    break
        return original(model)

    conv.recover_zipformer_constants = recover

    with tempfile.TemporaryDirectory(prefix="sherpa-zipformer-") as tmp:
        d = Path(tmp)
        for name, src in (("encoder", a.encoder), ("decoder", a.decoder), ("joiner", a.joiner)):
            (d / f"{name}.onnx").symlink_to(src.resolve())
        if a.tokens:
            (d / "tokens.txt").symlink_to(a.tokens.resolve())
        elif a.bpe:
            pieces = sentencepiece_pieces(a.bpe)
            (d / "tokens.txt").write_text(
                "".join(f"{p} {i}\n" for i, p in enumerate(pieces)), encoding="utf-8"
            )
            print(f"tokens.txt from {a.bpe.name}: {len(pieces)} pieces, first {pieces[:4]}")
        else:
            sys.exit("need --tokens or --bpe")
        sys.argv = ["convert_kroko_onnx.py", str(d), str(a.output), "--overwrite"]
        conv.main()
    add_bias_norm_scales(a.encoder, a.output / "model.safetensors")


def add_bias_norm_scales(encoder: Path, weights: Path) -> None:
    """Older exports fold each BiasNorm's `log_scale.exp()` into an unnamed
    scalar constant. BiasNorm is `x * mean((x - bias)^2)^-0.5 * exp(log_scale)`,
    so from the `Sub` that reads `<prefix>.bias`, follow the chain to the `Mul`
    by a one-element constant and store `log(value)` as `<prefix>.log_scale`."""
    import numpy as np
    import onnx
    from onnx import numpy_helper
    from safetensors.numpy import load_file, save_file

    model = onnx.load(encoder, load_external_data=False)
    init = {i.name: i for i in model.graph.initializer}
    users: dict[str, list] = {}
    for node in model.graph.node:
        for x in node.input:
            users.setdefault(x, []).append(node)
    tensors = load_file(str(weights))
    added = 0
    for name in [n for n in init if n.endswith("norm.bias")]:
        key = "encoder." + name.removesuffix(".bias") + ".log_scale"
        if key in tensors:
            continue
        frontier = [n for n in users.get(name, []) if n.op_type == "Sub"]
        found = None
        for _ in range(6):
            nxt = []
            for node in frontier:
                for out in node.output:
                    for user in users.get(out, []):
                        if user.op_type == "Mul":
                            for x in user.input:
                                t = init.get(x)
                                if t is not None and int(np.prod(t.dims)) == 1:
                                    found = float(numpy_helper.to_array(t).reshape(-1)[0])
                        nxt.append(user)
                if found is not None:
                    break
            if found is not None:
                break
            frontier = nxt
        if found is None or found <= 0:
            raise SystemExit(f"cannot recover {key}")
        tensors[key] = np.array([np.log(found)], dtype=np.float32)
        added += 1
    save_file(tensors, str(weights), metadata={"format": "pt"})
    print(f"recovered {added} BiasNorm log_scale values")


if __name__ == "__main__":
    main()
