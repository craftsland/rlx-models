"""Run one REAL layer's attention block through the released reference.

Takes the subset produced by `fetch_subset.py` (~130 MB of the 510 GB
checkpoint) and dumps the attention sublayer output for a seeded hidden state,
so the port can be checked against the reference on actual fp8 weights rather
than synthetic ones.

    python3 fetch_subset.py --out /tmp/dsv41_layer0.safetensors --layer 0
    RLX_REF_NOQUANT=1 python3 dump_real_layer.py /tmp/dsv41_layer0.safetensors
"""
import json, os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RLX_REF_NOQUANT", "1")
import prng
import model as M

SEQ = 160  # > window_size (128), so the sliding window actually evicts


def dequant(codes, scales, block=32):
    """Decode a quantized tensor the way the checkpoint stores it.

    Deliberately uses torch's OWN fp8 decoders rather than re-deriving them, so
    this stays an independent check of `dsv41_quant.rs` instead of a second copy
    of it. The three scale layouts are told apart by the scale's row count: one
    row per weight row (the FP4 experts and the Engram table) or one per
    `block` rows (the FP8 Linears).
    """
    rows, cols = codes.shape
    if codes.dtype == torch.int8:  # FP4 nibble pairs
        tbl = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
        u = codes.view(torch.uint8)
        vals = torch.empty(rows, cols * 2)
        vals[:, 0::2] = tbl[(u & 0x0F).long()]
        vals[:, 1::2] = tbl[((u >> 4) & 0x0F).long()]
        cols *= 2
    else:
        vals = codes.float()
    s = scales.float()
    rep_r = 1 if s.shape[0] == rows else block
    s = s.repeat_interleave(rep_r, 0)[:rows].repeat_interleave(block, 1)[:, :cols]
    return vals * s


def load_subset(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        raw = {k: f.get_tensor(k) for k in f.keys()}
    # a `.scale` is a companion only when its `<stem>.weight` exists — the HC
    # parameters are literally named `hc_attn_scale` and are not quantized
    companion = {
        k for k in raw
        if k.endswith(".scale") and k[: -len(".scale")] + ".weight" in raw
    }
    out, used = {}, set()
    for k, v in raw.items():
        if k in companion:
            continue
        sk = k[: -len(".weight")] + ".scale" if k.endswith(".weight") else None
        if sk in companion:
            out[k] = dequant(v, raw[sk])
            used.add(sk)
        else:
            out[k] = v.float()
    # A scale left unconsumed means a weight was read as raw codes — which is
    # ~450x too large and otherwise looks like a plausible tensor.
    assert used == companion, f"unconsumed scales: {sorted(companion - used)}"
    return out


def main():
    LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    sub = load_subset(sys.argv[1])
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")))
    t = cfg["text_config"]
    torch.set_default_dtype(torch.float32)
    args = M.ModelArgs(
        max_batch_size=1, max_seq_len=SEQ, temperature=0, dtype="bf16", expert_dtype=None,
        vocab_size=t["vocab_size"], dim=t["hidden_size"], moe_inter_dim=t["moe_intermediate_size"],
        n_layers=t["num_hidden_layers"], n_mtp_layers=0, n_heads=t["num_attention_heads"],
        n_routed_experts=t["n_routed_experts"], n_shared_experts=t["n_shared_experts"],
        n_activated_experts=t["num_experts_per_tok"], score_func=t["scoring_func"],
        route_scale=t["routed_scaling_factor"], swiglu_limit=t["swiglu_limit"],
        q_lora_rank=t["q_lora_rank"], head_dim=t["head_dim"], rope_head_dim=t["qk_rope_head_dim"],
        norm_eps=t["rms_norm_eps"], o_groups=t["o_groups"], o_lora_rank=t["o_lora_rank"],
        window_size=t["sliding_window"], compress_ratios=tuple(t["compress_ratios"]),
        kv_source_layers=tuple(t["kv_source_layer_ids"]),
        index_source_layers=tuple(t["index_source_layer_ids"]),
        compress_rope_theta=t["compress_rope_theta"],
        original_seq_len=t["rope_scaling"]["original_max_position_embeddings"],
        rope_theta=t["rope_theta"], rope_factor=t["rope_scaling"]["factor"],
        beta_fast=t["rope_scaling"]["beta_fast"], beta_slow=t["rope_scaling"]["beta_slow"],
        index_n_heads=t["index_n_heads"], index_head_dim=t["index_head_dim"],
        index_topk=t["index_topk"], candidate_source_layer=t["candidate_source_layer_id"],
        candidate_topk_blocks=t["candidate_topk_blocks"],
        candidate_block_size=t["candidate_block_size"], hc_mult=t["hc_mult"],
        hc_sinkhorn_iters=t["hc_sinkhorn_iters"], hc_eps=t["hc_eps"],
    )
    hc, d = args.hc_mult, args.dim
    ratio = args.compress_ratios[LAYER]
    print(f"layer {LAYER}: compress_ratio {ratio}, "
          f"kv_source {LAYER in args.kv_source_layers}, "
          f"index_source {LAYER in args.index_source_layers}")

    attn = M.Attention(LAYER, args)
    norm = M.RMSNorm(d, args.norm_eps)
    lp = f"layers.{LAYER}"
    with torch.no_grad():
        for mod, key in [(attn.wq_a, "attn.wq_a"), (attn.wq_b, "attn.wq_b"),
                         (attn.wkv, "attn.wkv"), (attn.wo_a, "attn.wo_a"),
                         (attn.wo_b, "attn.wo_b")]:
            mod.weight = nn.Parameter(sub[f"{lp}.{key}.weight"])
            mod.scale = None
        attn.q_norm.weight = nn.Parameter(sub[f"{lp}.attn.q_norm.weight"])
        attn.kv_norm.weight = nn.Parameter(sub[f"{lp}.attn.kv_norm.weight"])
        attn.attn_sink = nn.Parameter(sub[f"{lp}.attn.attn_sink"])
        norm.weight = nn.Parameter(sub[f"{lp}.attn_norm.weight"])
        # a KV source carries a compressor, an index source an indexer
        if attn.compressor is not None:
            attn.compressor.wkv.weight = nn.Parameter(sub[f"{lp}.attn.compressor.wkv.weight"])
            attn.compressor.wkv.scale = None
            if ratio > 1:
                attn.compressor.wgate.weight = nn.Parameter(
                    sub[f"{lp}.attn.compressor.wgate.weight"])
                attn.compressor.wgate.scale = None
            attn.compressor.norm.weight = nn.Parameter(
                sub[f"{lp}.attn.compressor.norm.weight"])
        if attn.indexer is not None:
            attn.indexer.wq_b.weight = nn.Parameter(sub[f"{lp}.attn.indexer.wq_b.weight"])
            attn.indexer.wq_b.scale = None
            attn.indexer.weights_proj.weight = nn.Parameter(
                sub[f"{lp}.attn.indexer.weights_proj.weight"])
            attn.indexer.weights_proj.scale = None
            if attn.indexer.owns_k:
                attn.indexer.wk.weight = nn.Parameter(sub[f"{lp}.attn.indexer.wk.weight"])
                attn.indexer.wk.scale = None
                attn.indexer.k_norm.weight = nn.Parameter(
                    sub[f"{lp}.attn.indexer.k_norm.weight"])

    # What layer 0 actually sees: real token embeddings expanded to `hc` copies,
    # with the one-hot initial pre-mix. Driving it with noise instead puts the
    # trained weights far out of distribution and the comparison stops meaning
    # much.
    rows = sub["embed.rows"][:SEQ]
    assert rows.shape == (SEQ, d), rows.shape
    h = rows.reshape(1, SEQ, 1, d).expand(1, SEQ, hc, d).contiguous()
    pre_mix = torch.zeros(1, SEQ, hc)
    pre_mix[:, :, 0] = 1.0

    hc_fn = sub[f"{lp}.hc_attn_fn"]
    hc_scale = sub[f"{lp}.hc_attn_scale"]
    hc_base = sub[f"{lp}.hc_attn_base"]
    with torch.inference_mode():
        # the reference's Block.hc_mixes, inlined (a Block would allocate the MoE)
        xf = h.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + args.norm_eps)
        mixes = torch.nn.functional.linear(xf, hc_fn) * rsqrt
        from kernel import hc_split_sinkhorn
        attn_pre, attn_post, attn_comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc, args.hc_sinkhorn_iters, args.hc_eps)
        xa = torch.sum(pre_mix.unsqueeze(-1) * h.float(), dim=2)
        xa = norm(xa)
        attn_out = attn(xa, 0)

    here = os.path.dirname(os.path.abspath(__file__))
    full = {
        "seq": SEQ, "layer": LAYER,
        "attn_norm_out": xa.reshape(-1).tolist(),
        "attn_out": attn_out.reshape(-1).tolist(),
        "attn_pre": attn_pre.reshape(-1).tolist(),
    }
    beside = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])),
                          f"dsv41_real_layer{LAYER}_ref.json")
    json.dump(full, open(beside, "w"), separators=(",", ":"))

    # A digest small enough to commit: the full tensors are 8 MB of JSON, and the
    # test that wants them already needs the 130 MB weight subset. Row norms plus
    # a fixed stride of samples catch any real regression at 30 KB.
    def digest(t):
        f = t.reshape(t.shape[-2], t.shape[-1]).float()
        return {
            "shape": list(f.shape),
            "absmean": float(f.abs().mean()),
            "max": float(f.abs().max()),
            "row_norms": f.norm(dim=-1).tolist(),
            "stride": 977,
            "samples": f.reshape(-1)[::977].tolist(),
        }

    dst = os.path.join(here, "..", "..", "tests", "fixtures",
                       f"dsv41_real_layer{LAYER}_digest.json")
    json.dump(
        {"seq": SEQ, "layer": LAYER, "compress_ratio": ratio,
         "attn_out": digest(attn_out), "attn_norm_out": digest(xa),
         "attn_pre": attn_pre.reshape(-1).tolist()},
        open(dst, "w"), separators=(",", ":"))
    print("attn_out", tuple(attn_out.shape), "absmean", float(attn_out.abs().mean()))
    print("wrote", beside, "and", os.path.relpath(dst))


if __name__ == "__main__":
    main()
