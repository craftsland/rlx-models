# rlx-jina-ocr

[jinaai/jina-ocr-v1](https://huggingface.co/jinaai/jina-ocr-v1) on RLX — a DeepSeek-OCR derivative: SAM-ViT-B + CLIP-L/14-224 "DeepEncoder", a linear `2048 → 1280` projector, and a 3B / 570M-active DeepSeek-V2 MoE decoder, plus a FastMTP draft head.

## Status

**Runs on the shared DeepSeek-OCR stack.** `baidu/Unlimited-OCR` is the same architecture under the same tensor names, so [`rlx-unlimited-ocr`](../rlx-unlimited-ocr/README.md) supplies the vision towers, projector, expert packing and compiled MoE decoder. This crate supplies what differs, and each difference is pinned by a test because every one of them fails *silently* — wrong attention span, wrong RoPE base or a stray BOS all still produce fluent-looking text.

| | Unlimited-OCR | jina-ocr-v1 |
|---|---|---|
| attention | rolling window, 128 | **full causal** (no `sliding_window` key) |
| `rope_theta` | 10 000 (implicit) | **1 000 000** |
| prompt | `<image>document parsing.` + BOS | `<\|User\|>:` chat template, **no BOS** |
| tiling `max_num` | 32 | **9** |
| n-gram guard | 35 / 128 | 35 / **1024** + `<td>`/`</td>` whitelist |
| draft head | — | **FastMTP**, recursive (wired, opt-in; config says K=3, K=1 measures better) |

**Verified on real weights, CPU.** Fed the reference implementation's own `inputs_embeds`, rlx reproduces its greedy output **token for token** (first-token logit 14.877 vs 14.875), and the full pipeline transcribes the bundled fixture page correctly. Getting there surfaced two pre-existing silent bugs in the shared compiled decoder (rank-4 RoPE writing almost nothing — forward *and* backward, across seven backends; the MoE router gathering weights with ONNX `Gather` instead of `take_along_axis`) — see [`rlx-unlimited-ocr`](../rlx-unlimited-ocr/README.md) and the CHANGELOG.

**All five available backends agree.** CPU, Metal, MLX, wgpu and Vulkan return token-identical greedy output on the same inputs. CUDA/ROCm were not checked (not present on this machine).

| precision | cpu | metal | mlx | wgpu | vulkan |
|---|---|---|---|---|---|
| f32 / f16 | exact | exact | exact | capacity | capacity |
| q8_0 / q4_0 | exact | exact | exact | exact | exact |

`capacity` is a loud refusal, not a wrong answer — wgpu will not stripe a 20 GiB activation arena across 4 GiB buffers, and Vulkan reports the 10.3 GiB weight prefix over `maxStorageBufferRange`. Both work at `--lm-precision q8_0`.

Architecture is additionally checked offline: `tests/checkpoint_inventory.rs` validates every tensor name and shape against the published safetensors *headers*, fetched over HTTP range requests and baked into a 16 KB fixture — all 2 722 tensors, including the expert grid, the SAM relative-position tables (which is where the 14-px window and the `[2,5,8,11]` global blocks are observable) and the FastMTP block. `tests/prompt_ids.rs` checks the chat template and prompt-id splice against ids from the checkpoint's own `tokenizer.json`.

## Architecture

| Stage | Module | Notes |
|-------|--------|-------|
| Config | [`config`](src/config.rs) | `config.json` + `processor_config.json`; pins `sliding_window = 0` and `rope_theta = 1e6` |
| Preprocess | [`preprocess`](src/preprocess.rs) | Gundam: 1024 px global view + up to **9** 640 px tiles; `ImageOps.pad` letterbox on 127-grey, `mean = std = 0.5` |
| Prompt | [`prompt`](src/prompt.rs) | `JINA_OCR_CHAT_TEMPLATE` + BOS-less id assembly around the `<image>` marker |
| Vision + decoder | `rlx-unlimited-ocr` | SAM → CLIP (fed SAM features as patch embeds) → concat → projector; pack order **local → global → separator** |
| Decode | [`runner`](src/runner.rs) | Greedy + sliding-window no-repeat-n-gram (35 / 1024, `<td>` whitelisted) |
| Post-process | [`postprocess`](src/postprocess.rs) | `decode_ocr`, `<\|ref\|>…<\|det\|>` → markdown + crop rects (1000-bin coords) |
| Speculation | [`mtp`](src/mtp.rs) | FastMTP draft head + `Drafter` adapter — see below |

Config: `hidden_size=1280`, `num_hidden_layers=12`, `num_attention_heads=10`, `n_routed_experts=64`, `n_shared_experts=2`, `num_experts_per_tok=6`, `moe_intermediate_size=896`, `intermediate_size=6848`, `first_k_dense_replace=1`, `vocab_size=129280`, `sliding_window=0`, `rope_theta=1e6`, `use_mla=false`. Tokens: `IMAGE_TOKEN_ID=128815`, `BOS=0`, `EOS=1`.

## FastMTP

The draft head ([`mtp`](src/mtp.rs)) loads the `mtp_module.heads.0.*` weights and runs the recursive draft forward (`K` from the config, overridable) — `eh_proj([enorm(Emb(t)); hnorm(h)])` → dense block → shared `model.norm` → shared `lm_head` — with its own KV cache.

It **is** wired into decode. The multi-token-with-past graph it needed (`seq = K+1`, `MaskKind::Bias`) now exists in the shared crate as the chunked verify pass, so `JinaOcrRunner::generate_with_mtp` runs a full speculative loop. Output is **byte-identical** to plain decode on the real checkpoint: a draft token is kept only when it equals the target's own greedy pick, which makes speculation a pure speed trade.

**It is opt-in.** Two real defects were found and fixed getting it working: the graph tapped the *post*-norm hidden state where FastMTP wants the pre-norm one (`pre_norm_hidden_states` in `modeling_deepseekv2.py`), and the draft block — a transformer layer with its own KV cache — was never primed over the prompt, so it proposed from two or three rows of context instead of the whole page. Both fail silently: the head keeps emitting valid tokens, just uninformed ones.

The checkpoint ships the draft *weights* but not the module that runs them (the card points at a vLLM plugin), so the remaining choices were settled by measurement rather than read off a reference.

[`examples/mtp_probe.rs`](examples/mtp_probe.rs) scores the head, and prints a control first: the target's own tapped hidden state, through the shared norm and host LM head, must reproduce the tokens the target actually generated. On the bundled page it does, 47/47 — without that, a draft score cannot be told apart from a broken tap.

| measurement | result |
|---|---|
| control (target hidden → its own next token) | **47/47** |
| `eh_proj([enorm(e); hnorm(h)])` | **25.5%** top-1 vs target |
| reversed concat order | 0.0% |
| priming over the prompt | +4pp |
| draft depth `K=1` / `K=2` / `K=3` | 1.26 / 1.32 / **1.32** tokens/round |

The concat order matches DeepSeek-V3's *code*, not its paper — the paper writes `M[RMSNorm(h_i); RMSNorm(Emb(t_{i+1}))]` with the operands the other way round. And the config's `K = 3` is the wrong depth: **step 3 accepts 0.0% of the time**, a full host forward plus a 129280×1280 projection for nothing. `MtpHead::set_steps` overrides it; `K = 1` captures 1.26 of the 1.32 tokens/round available.

**It is a net loss on this model, and the draft head is not the reason.** [`examples/mtp_cost.rs`](examples/mtp_cost.rs) times the three terms that decide it — plain decode step, chunked verify forward, host draft step — *interleaved in one process*, at a real 927-token context, after a warm-up so no timing pays for a graph compile:

```
median over 15 reps, interleaved, at 927 tokens of context

plain decode step             782.8 ms
one host draft step             9.5 ms

K        chunk(K+1)     round cost    tokens/round*   break-even
1           1246.5ms        1256.0ms             1.26        0.79x
2           1252.6ms        1271.5ms             1.32        0.81x
3           1217.3ms        1245.7ms             1.32        0.83x
```

Drafting is under 2% of a round. The chunked verify forward is 1.6×–2.6× a single-token decode step (the ratio moves with machine load; it is never ≈1×), and at 1.26–1.41 tokens/round that lands at 0.47×–0.83×. A 192-token end-to-end run reproduces it independently at 0.62×.

Note the chunk cost is nearly flat in `n` — 1247 / 1253 / 1217 ms for `n = 2 / 3 / 4`. So it is not the extra tokens that cost, it is the `n > 1` path taking a slower route than the single-token one. That is the thing to fix: a chunked forward at parity with a decode step would make `K = 2` a ~1.3× win.

**How not to measure this.** This is a shared machine: identical work returned 88 s, 129 s and 255 s depending on what else was running. Timing the two arms sequentially therefore compares load, not code — that produced an apparent 1.05× speedup that a later run flatly contradicted. Subtracting an estimated prefill cost to recover a per-token figure was no better, since on a 48-token transcription the vision encode plus the 927-token prefill is ~101 s of ~131 s. [`examples/mtp_bench.rs`](examples/mtp_bench.rs) still runs both arms end to end and breaks the time down — useful for the acceptance statistics and for checking the output stayed identical, but read `mtp_cost` for the speed question.

The draft head needs a host-resident f32 copy of `lm_head` (129280×1280, 662 MB) because the draft hidden state never enters the compiled graph; `JinaMtp::host_bytes` reports it. The embedding table is not duplicated — draft steps look rows up through the already-packed weights.

## Setup

```bash
just fetch-jina-ocr
# or: huggingface-cli download jinaai/jina-ocr-v1
# or: cargo run -p rlx-jina-ocr --features hf-download --release -- --download
```

Weights resolve from `RLX_JINA_OCR_DIR` or the Hugging Face cache (`HF_HOME` / `~/.cache/huggingface`). ~6.7 GB, bf16.

## CLI

```bash
just jina-ocr -- --image page.png
just features=all-backends jina-ocr -- --image page.png --device metal --lm-precision q8_0

cargo run -p rlx-jina-ocr --release -- --dry          # config + tensor counts
cargo run -p rlx-jina-ocr --release -- --image page.png --raw   # no markdown rendering
```

`--mode gundam|native`, `--lm-precision f32|f16|bf16|q8_0|q4_0|auto`, `--max-tokens N`, `--prompt TEXT`, `--device auto|cpu|metal|cuda|…`.

## Library

```rust,ignore
use rlx_jina_ocr::{InferenceOptions, JinaOcrSession};

let mut session = JinaOcrSession::open(model_dir, InferenceOptions::for_ocr())?;
let result = session.run_single("page.png")?;
println!("{}", result.markdown);
for crop in &result.crops {
    // crop.pixel_box is in the source image's own pixel space
    println!("{} -> {:?}", crop.alt_text, crop.pixel_box);
}
```

## Debugging

Two examples exist to split "is the vision pack right" from "is the LM right" —
the split that localized both decoder bugs:

```bash
# Projected vision pack for one page, vs the reference's compute_inputs_embeds rows
cargo run -p rlx-jina-ocr --release --example dump_vision -- --image page.png --out /tmp/v.f32

# Decoder only, on caller-supplied inputs_embeds (bypasses the vision tower).
# RLX_JINA_EAGER=1 runs the uncompiled host path; RLX_JINA_DUMP_KV=1 dumps
# per-layer K/V so the first diverging layer names the faulty lowering.
cargo run -p rlx-jina-ocr --release --example lm_from_embeds -- --embeds /tmp/e.f32 --steps 24
```

## Tests

```bash
cargo test -p rlx-jina-ocr                         # offline: 48 unit + 8 inventory + template
RLX_JINA_OCR_DIR=/path/to/tokenizer-only-dir \
  cargo test -p rlx-jina-ocr --test prompt_ids     # id-exact vs tokenizer.json (~10 MB)
just test-jina-ocr-parity                          # needs the 6.7 GB checkpoint
```

## License

GPL-3.0-only, matching the workspace. The **model weights** are CC BY-NC 4.0 — non-commercial use only. That is a constraint on the checkpoint, not on this crate.
