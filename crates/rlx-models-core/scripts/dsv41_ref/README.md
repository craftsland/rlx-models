# DeepSeek-V4.1 reference parity harness

Regenerates the fixtures behind
`crates/rlx-models-core/tests/dsv41_reference_parity.rs`.

The released `inference/model.py` runs unmodified on CPU once its tilelang
kernels are replaced. `kernel.py` here is that replacement: a
numerically-identical torch transliteration of `act_quant`, `fp4_act_quant`,
`fp8_gemm`, `fp4_gemm`, `sparse_attn` and `hc_split_sinkhorn`, written from the
prim_funcs rather than approximated.

```sh
./fetch.sh                                  # pull model.py / engram.py / vision.py
RLX_REF_NOQUANT=1 python3 dump.py           # text stack   -> dsv41_ref.json
RLX_REF_NOQUANT=1 python3 dump_vision.py    # ViT+aligner  -> dsv41_vision_ref.json
RLX_REF_NOQUANT=1 python3 dump_dspark.py    # draft head   -> dsv41_dspark_ref.json
```

Needs `torch`, `numpy` and `sympy`. No GPU and no checkpoint: every parameter
comes from `prng.py`, a name-keyed splitmix64 stream the Rust side reproduces
exactly, so the fixtures only have to carry shapes and outputs.

Two switches matter:

- **`RLX_REF_NOQUANT=1`** turns the in-place FP8/FP4 activation round-trips into
  no-ops. Those are precision simulation, not semantics, and the port computes
  the F32-exact value; leaving them on measures the quantization error instead of
  the port's.
- **`dump.py` pins `torch.topk`'s tie order** to lowest-index-wins. Torch leaves
  it unspecified and the Indexer produces exact ties constantly (it rectifies its
  head scores), so without pinning, neither implementation is reproducible, let
  alone comparable. `Op::TopK` uses the same rule.

`dump.py` writes every intermediate it can hook; the committed fixture is a
trimmed copy. Point `RLX_DSV41_REF` at the full one to bisect with
`cargo run -p rlx-models-core --example dsv41_bisect`.

## Real weights, without the 510 GB

A safetensors header gives every tensor's byte range, so a real-weight test only
pays for the tensors it touches. `fetch_subset.py` range-fetches one layer's
attention block plus a slice of `embed.weight` — 130 MB of the 48 shards — into
a single local file:

```sh
mkdir -p /tmp/dsv41w
# layer 0 is sliding-window only; layer 2 is a KV *and* index source, so it runs
# the compressor, the index keys and the YaRN-scaled compressed RoPE
python3 fetch_subset.py --out /tmp/dsv41w/layer0.safetensors --layer 0
python3 fetch_subset.py --out /tmp/dsv41w/layer2.safetensors --layer 2
# one tensor per quantization layout + 64 rows of the 384M-row Engram table
python3 fetch_subset.py --out /tmp/dsv41w/quant.safetensors --what quant --tokenizer

RLX_REF_NOQUANT=1 python3 dump_real_layer.py /tmp/dsv41w/layer0.safetensors 0
RLX_REF_NOQUANT=1 python3 dump_real_layer.py /tmp/dsv41w/layer2.safetensors 2
python3 dump_real_quant.py /tmp/dsv41w/quant.safetensors

RLX_DSV41_WEIGHTS=/tmp/dsv41w cargo test -p rlx-models-core --release \
    --test dsv41_reference_parity
```

About 300 MB in total, and it buys four things:

| test | what real bytes prove |
|---|---|
| `real_layer_attention_matches_reference` | the port's `DsV41Loader` + attention at real geometry, 160 tokens (past `sliding_window`), for a plain **and** a compressed layer — 7.5e-7 relative |
| `dequant_matches_reference_on_real_bytes` | all three scale layouts decode **exactly**: FP8 tiles, FP4 nibble pairs, and the row-wise FP8 Engram table |
| `engram_token_map_matches_the_real_tokenizer` | the normalization collapses the real 129,280-token vocab to exactly **99092**, the size every hash multiplier is derived from |
| `tensor_manifest_matches_the_real_checkpoint` | (no download) all **96,085** tensors match what the port expects, both directions |

`dump_real_layer.py` and `dump_real_quant.py` deliberately decode with **torch's
own** fp8 decoders and the FP4 table lifted verbatim from the reference
`convert.py`, so they stay independent checks of `dsv41_quant.rs` rather than a
second copy of it.

Without `RLX_DSV41_WEIGHTS` each of these checks its committed digest and skips
the rest; the manifest and token-map merge-case tests always run.

## Backends

The reference harness is CPU-only, but the port is not. `--features metal,mlx`
adds those devices to `prefill_matches_reference_on_all_backends`, and

```sh
RLX_DSV41_DEVICE=metal cargo run --release -p rlx-models-core \
    --features metal --example dsv41_bisect
```

compares that backend against CPU at every tap — no reference dump needed, since
CPU is already pinned to the reference. That is how both upstream backend bugs
were found; `dsv41_rope_probe` and `dsv41_topk_probe` then isolated them to a
single op each.
