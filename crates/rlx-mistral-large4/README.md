# rlx-mistral-large4

**Mistral Large 4** (`general.architecture = mistral-large4`, HF
`model_type = mistral_large4`) for RLX — 1 T total / 52 B active.

* Weights: [`mistralai/Mistral-Large-4-1T-A52B`](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)
* GGUF: [`unsloth/Mistral-Large-4-1T-A52B-GGUF`](https://huggingface.co/unsloth/Mistral-Large-4-1T-A52B-GGUF) (233 GB at `UD-IQ1_S`)

## Architecture (text)

72 layers, `hidden = 6144`, `vocab = 204800`, 64 heads of 128.

| Piece | Module | Notes |
|---|---|---|
| MLA without Q-LoRA, `kv_lora 512` | [`attention`](src/attention.rs) | queries scaled per head by SSMax `s·log(pos+1)+b` |
| NoPE every 4th layer | [`attention`](src/attention.rs) | whole head non-rotary there: `attn_k_b` is `[h, 512, 128]` instead of `[h, 512, 64]` |
| Cross-layer KV sharing | [`attention`](src/attention.rs) | layers 39, 43 … 71 have no `attn_kv_a_mqa` and attend to layer 35's latent |
| DSA lightning indexer, top-2048 | [`indexer`](src/indexer.rs) | 28 `"shared"` layers reuse the previous selection; exact causal short-circuit at `seq ≤ 2048` |
| LeRoPE | [`rope`](src/rope.rs) | YaRN (factor 2048 over 512) ÷ learned `rope_freqs.weight` (= `exp(-alpha)`), interleaved pairs |
| 128-expert MoE, top-4 + 1 shared | [`moe`](src/moe.rs) | softmax router, renormalized; layer 0 dense |
| Logit softcap 30 | [`flow`](src/flow.rs) | `c·tanh(logits/c)` |
| Incremental decode | [`decode`](src/decode.rs) | latent-only cache for the 63 owning layers (576 / 512 floats per token) |
| **Disk-paged experts** | [`paged`](src/paged.rs) | whole model, one token per step; routed experts read per token by `rlx_distributed::ExpertPager`, packed |

## Running the whole model (experts paged from disk)

```sh
cargo run -p rlx-mistral-large4 --release --features metal -- \
  --weights …/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf \
  --prompt "…" --tokenizer tokenizer.json   # or --prompt-ids 1,…
  [--max-tokens 32] [--ctx 512] [--cache-gb 8] [--device metal] [--timing]
```

Of the 233 GB at `UD-IQ1_S`, 214 GB are routed expert banks. Those stay on
disk; each token reads the 4 experts it fires per layer (≈ 7 GB/token cold,
less as the `--cache-gb` LRU warms). Resident: ≈ 16.5 GB of packed
attention / router / shared-expert / head weights, 2 GB of f32
`attn_k_b`/`attn_v_b`, and ≈ 8 GB of per-layer expert slot buffers — about
26 GB plus the cache. Shards are mmapped one by one (never merged) and the
token embedding is dequantized one row per token on the host.

Per layer, a step runs two graphs: attention + router (graph A), then — once
the host has paged the fired experts straight into slot-sized packed bank
params — shared + routed experts (graph B). The ids come from the same
in-graph router the resident model uses. Steady state on real blocks
(M-series, page cache warm): CPU 30 / 20 / 109 ms and Metal 16 / 19 / 77 ms
for attention / paging / experts per layer per token — ≈ 8 s/token on Metal
before disk reads; the IQ GEMVs dominate.

Through the generic path: the `mistral-large4` GGUF architecture is in
`rlx_core::model_registry`, and [`MistralLarge4Runner`](src/runner.rs)
(`LmRunner`: `generate`, `predict_logits` with prefix reuse, host-driven
`prefill_logits` / `decode_logits`) is registered as runner `mistral-large4` —
by the facade's builtin table (`rlx-models` feature `mistral-large4`, part of
`runner`), or by `rlx_mistral_large4::register()` for a crate that does not
use the facade. `auto_runner(shard_or_dir)` then builds it; `max_seq` sets the
context (default and maximum 2047), the expert cache gets 8 GB.

Limits: prompts are fed one token per step (exact — decode reproduces
prefill), and the context is capped at `index_topk − 1` = 2047 tokens. No
tokenizer is read from the GGUF; pass the HF `tokenizer.json`.

## Tests

| Command | What it checks |
|---|---|
| `cargo test -p rlx-mistral-large4 --test contract` | the shipped `config.json`, the GGUF metadata and the full 1602-tensor manifest (both directions, every shape) |
| `just test-mistral-large4-parity` | logits vs an independent torch port of `modular_mistral_large4.py` on a tiny checkpoint: RoPE + NoPE layers, KV sharing, shared indexers, **sparse** DSA (`topk 5 < seq 9`), MoE, softcap; forced DSA == causal below the budget; decode == reference |
| `just mistral-large4-real` | blocks 35, 38, 39 of the real `UD-IQ1_S` (range-fetched, ~1.1 GB, 6 of 128 experts) vs gguf-py dequant + the torch port, dense **and** packed weights; DSA made selective with `index_topk = 6`; then the same blocks **paged** token by token |
| `RLX_ML4_REAL_DENSE_DIR=… cargo test -p rlx-mistral-large4 --release --test paged paged_blocks_match_reference` on one real shard | blocks 69 and 70 of `UD-IQ1_M` with **all 128 experts and the real routing**, paged from the downloaded shard: download parts `00001` (metadata, 14 MB) and `00007` (13.6 GB), put `rope_freqs.weight` (range-fetched from part 2 by `scripts/mistral_large4_rope_part.py`) in a one-tensor stand-in part beside them, run `scripts/mistral_large4_reference.py --gguf p1,p7,rope_part --blocks 69,70 --topk 2048 --seq 4 DIR` (experts dequantized lazily, 4.7 GB peak), then the test against `DIR` |
| `cargo test -p rlx-mistral-large4 --test paged` | a tiny GGUF with Q8_0 banks written on the fly: paged decode (no cache, then warm cache) == resident prefill, every expert read from the file ; the header sniff + registry build the runner and its `generate` / `predict_logits` / `prefill_logits`+`decode_logits` agree with stepping the model |

Measured: tiny-checkpoint parity 3e-6..7e-6 max abs; real blocks 2.3e-6
(dequantized) and 2.1e-6 (packed) relative; paged == resident 1.2e-6 (tiny),
paged real blocks vs reference 9e-7 (CPU) / 3e-7 (Metal); real shard, blocks
69–70 with 128 experts: 1.6e-6 (CPU) / 3.8e-7 (Metal).

With real routing the slot banks change on every step, so on CPU the dequant
cache re-expands them each token: experts take ≈ 2.6 s per layer per token,
≈ 1.1 s with `RLX_DEQUANT_CACHE=0` (same result), and ≈ 65 ms on Metal. Use
Metal; the 109 ms CPU figure above came from a 6-expert subset whose slots
rarely changed.

## Not done

* **The whole model has not been run here** — 233 GB does not fit on this
  machine's disks; the paged path is validated on the tiny GGUF and on real
  blocks.
* Batched prefill on the paged path, and overlapping paging with compute.
* Decode past `index_topk` — needs an indexer-key cache; the decode builder
  rejects such a capacity rather than running dense attention.
* The Pixtral vision tower (`mmproj-*.gguf`) and the DSpark MTP draft head.
* Tokenizer (Tekken BPE in the GGUF) and chat template.
