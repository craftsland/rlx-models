#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# SPDX-License-Identifier: GPL-3.0-only
"""Reference logits for rlx-mistral-large4 on a tiny random checkpoint.

The installed transformers has no `mistral_large4` yet, so this is the forward
pass of `transformers/models/mistral_large4/modular_mistral_large4.py`
rewritten in plain torch (float64, CPU, batch 1). It is written from the HF
code, not from the Rust port, so the two are independent spellings of the
same model. Pieces, by HF class:

  * `MistralLarge4TextRotaryEmbedding` — YaRN `inv_freq` (transformers'
    `_compute_yarn_parameters`, truncate=True) times the learned `exp(alpha)`;
    `attention_factor` multiplies cos/sin.
  * `apply_rotary_pos_emb_interleave` — rotates (x0,x1),(x2,x3)… pairs and
    writes `cat(evens', odds')`.
  * `MistralLarge4TextSoftmaxScaling` — query *= s_h·log(pos+1) + b_h.
  * `MistralLarge4TextIndexer` — LN(x)→wq_b→SSMax, LN(wk x), RoPE on the
    leading `qk_rope_head_dim` dims (RoPE layers only), relu(q·k/√d) weighted
    by `weights_proj(x)/√n_heads`, summed over heads, top-k of the causal
    scores. `"shared"` layers reuse the previous layer's top-k.
  * `MistralLarge4TextAttention` — MLA with no Q-LoRA, NoPE layers widening
    the non-rotary part, KV sharing via `kv_source_layers`, scaling √qk_head.
  * `MistralLarge4TextTopkRouter` / `MoE` — softmax, top-k, renormalized with
    `sum + 1e-20`, times `routed_scaling_factor`, plus the shared expert.
  * `MistralLarge4ForConditionalGeneration` — logit softcapping.

Weights are written under their **GGUF** names in the orientation the GGUF
loader hands rlx (torch `[out, in]`; `attn_k_b` `[h, kv_lora, nope]`,
`attn_v_b` `[h, v, kv_lora]`, expert banks `[E, out, in]`), rounded to f32
first so both sides see identical values.

    python3 scripts/mistral_large4_reference.py /tmp/ml4-ref
    RLX_ML4_REF_DIR=/tmp/ml4-ref cargo test -p rlx-mistral-large4 --test reference_parity

Two configs share the weights: `sparse` (index_topk < seq, so the DSA
selection is real) and `dense` (index_topk = 2048). Top-k ties at the
selection boundary would make the comparison ill-posed, so the seed is bumped
until neither the indexer nor the router has one.
"""
import argparse
import json
import math
import os
import sys

import torch
from safetensors.torch import save_file

torch.set_default_dtype(torch.float64)

N_LAYERS = 8
BASE = dict(
    vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
    num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=4,
    kv_lora_rank=12, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16, head_dim=8,
    n_routed_experts=6, num_experts_per_tok=2, n_shared_experts=1, n_group=1, topk_group=1,
    routed_scaling_factor=1.5, first_k_dense_replace=1,
    index_n_heads=4, index_head_dim=16,
    rms_norm_eps=1e-5, final_logit_softcapping=5.0,
    # the shipped pattern, shrunk: NoPE every 4th layer, a late layer sharing
    # an earlier NoPE layer's KV, and shared indexers on rope and nope layers
    no_rope_layers=[int((i + 1) % 4 != 0) for i in range(N_LAYERS)],
    kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 3],
    indexer_types=["full", "full", "shared", "full", "shared", "full", "full", "shared"],
    mlp_layer_types=["dense"] + ["sparse"] * (N_LAYERS - 1),
    rope_parameters=dict(rope_type="yarn", rope_theta=10000.0, factor=4.0,
                         original_max_position_embeddings=8, beta_fast=32.0, beta_slow=1.0,
                         attention_factor=1.0),
    model_type="mistral_large4_text",
)
INDEXER_EPS = 1e-6


class Tie(Exception):
    pass


def yarn_inv_freq(rp, dim):
    base, factor = rp["rope_theta"], rp["factor"]
    orig = rp["original_max_position_embeddings"]

    def corr_dim(rot):
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(corr_dim(rp["beta_fast"])), 0)
    high = min(math.ceil(corr_dim(rp["beta_slow"])), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2) - low) / (high - low)).clamp(0, 1)
    pos_freqs = base ** (torch.arange(0, dim, 2) / dim)
    extra, inter = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    ef = 1 - ramp
    return inter * (1 - ef) + extra * ef


def make_weights(c, seed):
    g = torch.Generator().manual_seed(seed)

    def rn(*shape, scale=None):
        fan_in = shape[-1]
        s = scale if scale is not None else 1.0 / math.sqrt(fan_in)
        return torch.randn(*shape, generator=g) * s

    def near1(*shape):
        return 1.0 + 0.1 * torch.randn(*shape, generator=g)

    D, H = c["hidden_size"], c["num_attention_heads"]
    kv, qk = c["kv_lora_rank"], c["qk_nope_head_dim"] + c["qk_rope_head_dim"]
    vd, r = c["v_head_dim"], c["qk_rope_head_dim"]
    nh, hd = c["index_n_heads"], c["index_head_dim"]
    E, Im, I, V = c["n_routed_experts"], c["moe_intermediate_size"], c["intermediate_size"], c["vocab_size"]
    w = {
        "token_embd.weight": rn(V, D, scale=1.0),
        "output.weight": rn(V, D),
        "output_norm.weight": near1(D),
        # exp(-alpha), alpha small
        "rope_freqs.weight": torch.exp(-0.2 * torch.randn(r // 2, generator=g)),
    }
    for i in range(c["num_hidden_layers"]):
        b = f"blk.{i}"
        rope_l = r if c["no_rope_layers"][i] else 0
        nope_l = qk - rope_l
        w[f"{b}.attn_norm.weight"] = near1(D)
        w[f"{b}.attn_q.weight"] = rn(H * qk, D)
        if c["kv_source_layers"][i] == i:
            w[f"{b}.attn_kv_a_mqa.weight"] = rn(kv + rope_l, D)
            w[f"{b}.attn_kv_a_norm.weight"] = near1(kv)
        w[f"{b}.attn_k_b.weight"] = rn(H, kv, nope_l, scale=1 / math.sqrt(kv))
        w[f"{b}.attn_v_b.weight"] = rn(H, vd, kv)
        w[f"{b}.attn_output.weight"] = rn(D, H * vd)
        w[f"{b}.attn_ssmax.weight"] = 0.3 * torch.randn(H, generator=g)
        w[f"{b}.attn_ssmax.bias"] = near1(H)
        if c["indexer_types"][i] == "full":
            p = f"{b}.indexer"
            w[f"{p}.q_norm.weight"] = near1(D)
            w[f"{p}.q_norm.bias"] = 0.1 * torch.randn(D, generator=g)
            w[f"{p}.attn_q_b.weight"] = rn(nh * hd, D)
            w[f"{p}.attn_k.weight"] = rn(hd, D)
            w[f"{p}.k_norm.weight"] = near1(hd)
            w[f"{p}.k_norm.bias"] = 0.1 * torch.randn(hd, generator=g)
            w[f"{p}.proj.weight"] = rn(nh, D)
            w[f"{p}.ssmax.weight"] = 0.3 * torch.randn(nh, generator=g)
            w[f"{p}.ssmax.bias"] = near1(nh)
        w[f"{b}.ffn_norm.weight"] = near1(D)
        if c["mlp_layer_types"][i] == "sparse":
            w[f"{b}.ffn_gate_inp.weight"] = rn(E, D, scale=1.0)
            w[f"{b}.ffn_gate_exps.weight"] = rn(E, Im, D)
            w[f"{b}.ffn_up_exps.weight"] = rn(E, Im, D)
            w[f"{b}.ffn_down_exps.weight"] = rn(E, D, Im)
            w[f"{b}.ffn_gate_shexp.weight"] = rn(Im, D)
            w[f"{b}.ffn_up_shexp.weight"] = rn(Im, D)
            w[f"{b}.ffn_down_shexp.weight"] = rn(D, Im)
        else:
            w[f"{b}.ffn_gate.weight"] = rn(I, D)
            w[f"{b}.ffn_up.weight"] = rn(I, D)
            w[f"{b}.ffn_down.weight"] = rn(D, I)
    # round once so rlx (f32) and this script (f64) read identical values
    return {k: v.float().double() for k, v in w.items()}


def rms(x, wt, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * wt


def ln(x, wt, bias, eps):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), wt, bias, eps)


def rot(x, cos, sin):
    # x [S, h, 2·half], cos/sin [S, half]
    c, s = cos[:, None, :], sin[:, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def check_topk_margin(scores, k, what):
    # scores [rows, n]; a tie straddling the k-th place makes top-k ambiguous
    if k >= scores.shape[-1]:
        return
    srt = scores.sort(dim=-1, descending=True).values
    a, b = srt[:, k - 1], srt[:, k]
    finite = torch.isfinite(a) & torch.isfinite(b)
    gap = (a - b).abs()[finite]
    if gap.numel() and gap.min() < 1e-4:
        raise Tie(f"{what}: top-{k} boundary gap {gap.min():.2e}")


def forward(c, w, ids=None, x=None, blocks=None, head=True):
    S = len(ids) if x is None else x.shape[0]
    D, H = c["hidden_size"], c["num_attention_heads"]
    kv, r = c["kv_lora_rank"], c["qk_rope_head_dim"]
    qk, vd = c["qk_nope_head_dim"] + r, c["v_head_dim"]
    nh, hd, topk = c["index_n_heads"], c["index_head_dim"], c["index_topk"]
    eps = c["rms_norm_eps"]
    rp = c["rope_parameters"]

    pos = torch.arange(S, dtype=torch.float64)
    inv = yarn_inv_freq(rp, c["head_dim"]) / w["rope_freqs.weight"]
    ang = pos[:, None] * inv[None, :]
    att = rp.get("attention_factor", 1.0)
    cos, sin = ang.cos() * att, ang.sin() * att
    log_pos = torch.log(pos + 1)[:, None]  # [S, 1]
    causal = torch.ones(S, S, dtype=torch.bool).tril()

    if x is None:
        x = w["token_embd.weight"][torch.tensor(ids)]
    shared_kv = {}
    sel = None
    for i in (blocks if blocks is not None else range(c["num_hidden_layers"])):
        b = f"blk.{i}"
        use_rope = bool(c["no_rope_layers"][i])
        rope_l = r if use_rope else 0
        nope_l = qk - rope_l
        h = rms(x, w[f"{b}.attn_norm.weight"], eps)

        # ── indexer ──
        if c["indexer_types"][i] == "full":
            p = f"{b}.indexer"
            q = ln(h, w[f"{p}.q_norm.weight"], w[f"{p}.q_norm.bias"], INDEXER_EPS) @ w[f"{p}.attn_q_b.weight"].T
            q = q.view(S, nh, hd)
            q = q * (w[f"{p}.ssmax.weight"][None, :] * log_pos + w[f"{p}.ssmax.bias"][None, :])[..., None]
            k = ln(h @ w[f"{p}.attn_k.weight"].T, w[f"{p}.k_norm.weight"], w[f"{p}.k_norm.bias"], INDEXER_EPS)
            k = k[:, None, :]
            if use_rope:
                q = torch.cat([rot(q[..., :r], cos, sin), q[..., r:]], -1)
                k = torch.cat([rot(k[..., :r], cos, sin), k[..., r:]], -1)
            k = k[:, 0, :]
            scores = torch.relu(torch.einsum("qhd,td->qht", q, k) * hd ** -0.5)
            wts = (h @ w[f"{p}.proj.weight"].T) * nh ** -0.5  # [S, nh]
            idx = torch.einsum("qh,qht->qt", wts, scores)
            idx = idx.masked_fill(~causal, float("-inf"))
            kk = min(topk, S)
            check_topk_margin(idx, kk, f"indexer {i}")
            top = idx.topk(kk, dim=-1).indices
            sel = torch.zeros(S, S, dtype=torch.bool).scatter(-1, top, True)
        if sel is None and S <= topk:
            # a run starting at a "shared" layer: below the budget the
            # selection it would reuse is every causal position
            sel = causal
        assert sel is not None

        # ── attention ──
        qs = (h @ w[f"{b}.attn_q.weight"].T).view(S, H, qk)
        q_pass, q_rot = qs[..., :nope_l], qs[..., nope_l:]
        if c["kv_source_layers"][i] != i:
            k_pass, k_rot = shared_kv["rope" if use_rope else "nope"]
        else:
            ckv = h @ w[f"{b}.attn_kv_a_mqa.weight"].T
            k_pass = rms(ckv[:, :kv], w[f"{b}.attn_kv_a_norm.weight"], eps)
            k_rot = ckv[:, kv:][:, None, :]  # [S, 1, rope_l]
            if use_rope:
                k_rot = rot(k_rot, cos, sin)
        if use_rope:
            q_rot = rot(q_rot, cos, sin)
        if any(src == i for src in c["kv_source_layers"][i + 1:]):
            shared_kv["rope" if use_rope else "nope"] = (k_pass, k_rot)
        query = torch.cat([q_pass, q_rot], -1)
        query = query * (w[f"{b}.attn_ssmax.weight"][None, :] * log_pos + w[f"{b}.attn_ssmax.bias"][None, :])[..., None]
        k_nope = torch.einsum("tc,hcn->htn", k_pass, w[f"{b}.attn_k_b.weight"])
        keys = torch.cat([k_nope, k_rot.permute(1, 0, 2).expand(H, S, rope_l)], -1)  # [H, S, qk]
        vals = torch.einsum("hvc,tc->htv", w[f"{b}.attn_v_b.weight"], k_pass)
        logits = torch.einsum("qhd,htd->hqt", query, keys) * qk ** -0.5
        logits = logits.masked_fill(~(causal & sel)[None], float("-inf"))
        out = torch.einsum("hqt,htv->qhv", logits.softmax(-1), vals).reshape(S, H * vd)
        x = x + out @ w[f"{b}.attn_output.weight"].T

        # ── FFN ──
        h = rms(x, w[f"{b}.ffn_norm.weight"], eps)
        if c["mlp_layer_types"][i] == "sparse":
            probs = (h @ w[f"{b}.ffn_gate_inp.weight"].T).softmax(-1)
            K = c["num_experts_per_tok"]
            check_topk_margin(probs, K, f"router {i}")
            top = probs.topk(K, dim=-1)
            tw = top.values / (top.values.sum(-1, keepdim=True) + 1e-20) * c["routed_scaling_factor"]
            y = torch.zeros_like(h)
            for t in range(S):
                for j in range(K):
                    e = int(top.indices[t, j])
                    g_ = h[t] @ w[f"{b}.ffn_gate_exps.weight"][e].T
                    u_ = h[t] @ w[f"{b}.ffn_up_exps.weight"][e].T
                    y[t] += tw[t, j] * ((torch.nn.functional.silu(g_) * u_) @ w[f"{b}.ffn_down_exps.weight"][e].T)
            sh = torch.nn.functional.silu(h @ w[f"{b}.ffn_gate_shexp.weight"].T) * (h @ w[f"{b}.ffn_up_shexp.weight"].T)
            y = y + sh @ w[f"{b}.ffn_down_shexp.weight"].T
        else:
            y = (torch.nn.functional.silu(h @ w[f"{b}.ffn_gate.weight"].T) * (h @ w[f"{b}.ffn_up.weight"].T)) @ w[f"{b}.ffn_down.weight"].T
        x = x + y

    if not head:
        return x
    x = rms(x, w["output_norm.weight"], eps)
    logits = x @ w["output.weight"].T
    cap = c.get("final_logit_softcapping")
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return logits


def gguf_config(meta, n_experts):
    """The text config from `mistral-large4.*` metadata (this script's own read)."""
    k = lambda n: meta[f"mistral-large4.{n}"]
    n = k("block_count")
    qk, r = k("attention.key_length_mla"), k("rope.dimension_count")
    return dict(
        hidden_size=k("embedding_length"), num_attention_heads=k("attention.head_count"),
        kv_lora_rank=k("attention.kv_lora_rank"), qk_nope_head_dim=qk - r, qk_rope_head_dim=r,
        v_head_dim=k("attention.value_length_mla"), head_dim=r,
        index_n_heads=k("attention.indexer.head_count"), index_head_dim=k("attention.indexer.key_length"),
        index_topk=k("attention.indexer.top_k"), rms_norm_eps=k("attention.layer_norm_rms_epsilon"),
        num_hidden_layers=n,
        no_rope_layers=[int(b) for b in k("attention.rope_pattern")],
        kv_source_layers=list(k("attention.kv_source_map")),
        indexer_types=["full" if b else "shared" for b in k("attention.indexer.types")],
        mlp_layer_types=["dense" if i < k("leading_dense_block_count") else "sparse" for i in range(n)],
        num_experts_per_tok=k("expert_used_count"), n_routed_experts=n_experts,
        routed_scaling_factor=k("expert_weights_scale"),
        rope_parameters=dict(rope_type="yarn", rope_theta=k("rope.freq_base"),
                             factor=k("rope.scaling.factor"),
                             original_max_position_embeddings=k("rope.scaling.original_context_length"),
                             beta_fast=k("rope.scaling.yarn_beta_fast"), beta_slow=k("rope.scaling.yarn_beta_slow"),
                             attention_factor=1.0),
    )


def gguf_main(a):
    """Real weights: run `--blocks` of a subset GGUF over a random hidden state."""
    import numpy as np
    from gguf import GGUFReader
    from gguf.quants import dequantize

    class Bank:
        """A routed expert bank, dequantized one expert at a time: 128 real
        experts in f64 are ~100 GB per layer."""

        def __init__(self, t):
            self.t, self.shape = t, [int(d) for d in reversed(t.shape)]

        def __getitem__(self, e):
            arr = dequantize(self.t.data[e], self.t.tensor_type).astype(np.float64)
            return torch.from_numpy(arr.reshape(self.shape[1:]))

    blocks = [int(b) for b in a.blocks.split(",")]
    wanted = lambda n: not n.startswith("blk.") or int(n.split(".")[1]) in blocks
    meta, w = {}, {}
    # `--gguf` may list several parts of one split, comma-separated.
    for path in a.gguf.split(","):
        rd = GGUFReader(path)
        for f in rd.fields.values():
            if not f.name.startswith("mistral-large4."):
                continue
            vals = [f.parts[i] for i in f.data]
            vals = [v.tolist()[0] if v.size == 1 else v.tolist() for v in vals]
            meta[f.name] = vals if len(vals) > 1 or f.name.endswith(("_pattern", "_map", ".types")) else vals[0]
        for t in rd.tensors:
            if not wanted(t.name):
                continue
            if t.name.endswith("_exps.weight"):
                w[t.name] = Bank(t)
                continue
            arr = dequantize(t.data, t.tensor_type).astype(np.float64)
            w[t.name] = torch.from_numpy(arr.reshape([int(d) for d in reversed(t.shape)]))
    n_exp = int(w[next(n for n in w if n.endswith("ffn_gate_inp.weight"))].shape[0])
    c = gguf_config(meta, n_exp)
    c["index_topk"] = a.topk
    for seed in range(a.seed, a.seed + 50):
        g = torch.Generator().manual_seed(seed)
        x0 = torch.randn(a.seq, c["hidden_size"], generator=g).float().double()
        try:
            out = forward(c, w, x=x0, blocks=blocks, head=False)
        except Tie as e:
            print(f"seed {seed}: {e}; retrying", file=sys.stderr)
            continue
        break
    else:
        sys.exit("no tie-free input found")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ref_real.json"), "w") as f:
        json.dump({"gguf": os.path.abspath(a.gguf.split(",")[0]), "blocks": blocks, "seq": a.seq,
                   "index_topk": a.topk, "hidden_in": x0.flatten().tolist(),
                   "hidden_out": out.flatten().tolist()}, f)
    print(f"seed {seed}: blocks {blocks}, |out| = {out.norm():.3e} -> {a.out}/ref_real.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--gguf", help="real-weight mode: a subset from scripts/mistral_large4_subset.py, "
                    "or comma-separated parts of a split (metadata shard first)")
    ap.add_argument("--blocks", default="35,38,39")
    ap.add_argument("--topk", type=int, default=6,
                    help="real-weight mode: index_topk override, < seq so the selection is real")
    ap.add_argument("--seq", type=int, default=9)
    ap.add_argument("--sparse-topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.gguf:
        if a.seq == 9:
            a.seq = 16
        return gguf_main(a)
    os.makedirs(a.out, exist_ok=True)
    configs = {"sparse": dict(BASE, index_topk=a.sparse_topk), "dense": dict(BASE, index_topk=2048)}
    for seed in range(a.seed, a.seed + 200):
        w = make_weights(BASE, seed)
        g = torch.Generator().manual_seed(seed + 1000)
        ids = torch.randint(0, BASE["vocab_size"], (a.seq,), generator=g).tolist()
        try:
            refs = {n: forward(c, w, ids) for n, c in configs.items()}
        except Tie as e:
            print(f"seed {seed}: {e}; retrying", file=sys.stderr)
            continue
        break
    else:
        sys.exit("no tie-free seed found")
    save_file({k: v.float().contiguous() for k, v in w.items()}, os.path.join(a.out, "weights.safetensors"))
    for n, c in configs.items():
        with open(os.path.join(a.out, f"config_{n}.json"), "w") as f:
            json.dump({"model_type": "mistral_large4", "text_config": c}, f, indent=1)
        with open(os.path.join(a.out, f"ref_{n}.json"), "w") as f:
            json.dump({"input_ids": ids, "seq": a.seq, "vocab": BASE["vocab_size"],
                       "logits": refs[n].flatten().tolist()}, f)
    diff = (refs["sparse"] - refs["dense"]).abs().max().item()
    print(f"seed {seed}: wrote {a.out}; sparse vs dense max |Δ| = {diff:.3e} (must be > 0 for the sparse case to mean anything)")


if __name__ == "__main__":
    main()
