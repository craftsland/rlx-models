// RLX — versatile ML compiler + runtime. GPLv3.
//! The `k`-token chunk step, against the two paths it generalizes.
//!
//! [`build_deepseek_v41_chunk`] has to reduce exactly to prefill when it starts
//! at zero over an empty cache, and exactly to a decode step when `k = 1`. If it
//! does both, and a prompt split into chunks equals the whole prompt, then the
//! window offsets, the compressor's group accounting across a chunk boundary,
//! and the compressed-visibility rule are all right — and those are the three
//! places this is easy to get subtly wrong.
//!
//! There is no reference dump here on purpose: these are *identities* between
//! paths in this port, so they can be checked to the last bit rather than to a
//! tolerance.

use rlx_models_core::dsv41::DeepseekV41Spec;
use rlx_models_core::dsv41_chunk::build_deepseek_v41_chunk;
use rlx_models_core::dsv41_decode::{
    V41ChunkPlan, V41DecodeCache, V41DecodePlan, build_deepseek_v41_decode, names,
};
use rlx_models_core::dsv41_engram::EngramHashPlan;
use rlx_models_core::dsv41_graph::{V41Inputs, build_deepseek_v41_prefill};
use rlx_models_core::weight_loader::SyntheticLoader;
use rlx_runtime::{Device, Session};
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Text-only, with a compressor whose ratio does *not* divide the chunk sizes —
/// so every test here crosses a group boundary rather than landing on one.
fn config(vocab: usize) -> Value {
    json!({
        "model_type": "deepseek_v41",
        "vocab_size": vocab, "hidden_size": 16, "num_hidden_layers": 6,
        "num_attention_heads": 2, "head_dim": 16, "qk_rope_head_dim": 8,
        "q_lora_rank": 8, "o_groups": 1, "o_lora_rank": 4, "hc_mult": 2,
        "compress_ratios": [0, 0, 3, 3, 3, 3],
        "kv_source_layer_ids": [2], "index_source_layer_ids": [2, 4],
        "index_head_dim": 8, "index_n_heads": 1, "index_topk": 2,
        "sliding_window": 5,
        "moe_intermediate_size": 8, "n_routed_experts": 4,
        "num_experts_per_tok": 2, "n_shared_experts": 1,
        "scoring_func": "sigmoid", "norm_topk_prob": true,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "compress_rope_theta": 10000.0,
        "engram_layer_ids": [1], "engram_num_embeddings": [64],
        "engram_max_ngram_size": 2, "engram_vocab_size": vocab,
        "engram_compressed_vocab_size": 16, "engram_n_heads": 1,
        "engram_head_dim": 4, "engram_pad_token_id": 0,
    })
}

fn manifest(spec: &DeepseekV41Spec) -> BTreeMap<String, Vec<usize>> {
    spec.expected_tensors()
        .into_iter()
        .map(|n| (n.name, n.shape))
        .collect()
}

fn opts() -> rlx_runtime::CompileOptions {
    rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    )
}

fn run(
    g: rlx_ir::graph::Graph,
    params: &std::collections::HashMap<String, Vec<f32>>,
    feed: &[(&str, &[f32])],
) -> Vec<Vec<f32>> {
    let mut s = Session::new(Device::Cpu).compile_with(g, &opts());
    for (n, v) in params {
        s.set_param(n, v);
    }
    s.run(feed)
}

struct Fixture {
    spec: DeepseekV41Spec,
    shapes: BTreeMap<String, Vec<usize>>,
    hash: EngramHashPlan,
    map: Vec<u32>,
}

fn fixture() -> Fixture {
    let cfg = config(32);
    let spec = DeepseekV41Spec::from_config(&cfg).expect("config parses");
    let shapes = manifest(&spec);
    let e = spec.engram.as_ref().expect("engram");
    let cv = e.compressed_vocab_size;
    let map: Vec<u32> = (0..spec.vocab_size).map(|i| (i % cv) as u32).collect();
    let hash = EngramHashPlan::new(e, &map).expect("hash plan");
    Fixture {
        spec,
        shapes,
        hash,
        map,
    }
}

impl Fixture {
    fn rows(&self, ids: &[u32], from: usize, len: usize) -> Vec<i64> {
        let c: Vec<u32> = ids.iter().map(|&i| self.map[i as usize]).collect();
        self.hash.hash_ids(&c[from..from + len], None, &c[..from])
    }

    fn loader(&self) -> SyntheticLoader {
        SyntheticLoader::new(self.shapes.clone())
    }
}

fn feed_with<'a>(ids: &'a [f32], cached: &'a [(String, &'a [f32])]) -> Vec<(&'a str, &'a [f32])> {
    let mut f: Vec<(&str, &[f32])> = vec![("input_ids", ids)];
    for (n, v) in cached {
        f.push((n.as_str(), *v));
    }
    f
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let maxd = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        maxd <= 1e-4 * scale,
        "{what}: max |Δ| {maxd:.3e} on a scale of {scale:.3e}"
    );
}

/// A chunk starting at zero over an empty cache *is* a prefill.
#[test]
fn a_chunk_from_zero_equals_prefill() {
    let fx = fixture();
    let ids: Vec<u32> = vec![5, 9, 2, 14, 7, 3, 11];
    let seq = ids.len();
    let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();

    let inputs = V41Inputs {
        engram_rows: fx.rows(&ids, 0, seq),
        ..Default::default()
    };
    let mut l = fx.loader();
    let mut pk = std::collections::HashMap::new();
    let (g, p, _) =
        build_deepseek_v41_prefill(&fx.spec, &mut l, seq, &inputs, &mut pk).expect("prefill");
    let want = run(g, &p, &[("input_ids", idf.as_slice())])[0].clone();

    let plan = V41ChunkPlan::new(&fx.spec, 0, seq);
    assert_eq!(plan.cache_len, 0, "a chunk from zero has nothing cached");
    let mut l = fx.loader();
    let mut pk = std::collections::HashMap::new();
    let (g, p, _) =
        build_deepseek_v41_chunk(&fx.spec, &mut l, &plan, &inputs, &mut pk).expect("chunk");
    let got = run(g, &p, &[("input_ids", idf.as_slice())])[0].clone();
    close(&got, &want, "chunk(0, seq) vs prefill");
}

/// A chunk of one token *is* a decode step, at every position — including the
/// ones where the compressor fires and the ones where it does not.
#[test]
fn a_chunk_of_one_equals_a_decode_step() {
    let fx = fixture();
    let ids: Vec<u32> = vec![5, 9, 2, 14, 7, 3, 11, 1];
    let mut c_dec = V41DecodeCache::new(&fx.spec);
    let mut c_chunk = V41DecodeCache::new(&fx.spec);

    for (pos, &tok) in ids.iter().enumerate() {
        let rows = fx.rows(&ids, pos, 1);
        let inputs = V41Inputs {
            engram_rows: rows.clone(),
            ..Default::default()
        };
        let idf = [tok as f32];

        let dplan = V41DecodePlan::new(&fx.spec, pos);
        let mut l = fx.loader();
        let mut pk = std::collections::HashMap::new();
        let (g, p, dnames) =
            build_deepseek_v41_decode(&fx.spec, &mut l, pos, &inputs, &mut pk).expect("decode");
        let cached = c_dec.step_inputs(&dplan);
        let eng = engram_feed(&fx, &rows);
        let mut f = feed_with(&idf, &cached);
        for (n, v) in &eng {
            f.push((n.as_str(), v.as_slice()));
        }
        let dout = run(g, &p, &f);
        drop(cached);
        c_dec.apply(&dplan, &dnames, &dout).expect("decode apply");

        let cplan = V41ChunkPlan::new(&fx.spec, pos, 1);
        let mut l = fx.loader();
        let mut pk = std::collections::HashMap::new();
        let (g, p, cnames) =
            build_deepseek_v41_chunk(&fx.spec, &mut l, &cplan, &inputs, &mut pk).expect("chunk");
        let cached = c_chunk.chunk_inputs(&cplan);
        let cout = run(g, &p, &feed_with(&idf, &cached));
        drop(cached);
        c_chunk
            .apply_chunk(&cplan, &cnames, &cout)
            .expect("chunk apply");

        close(
            &cout[0],
            &dout[0],
            &format!("chunk(k=1) vs decode at {pos}"),
        );
    }
}

/// Engram rows, split per layer — the chunk bakes them, decode takes them as an
/// input, so the decode side needs the feed built.
fn engram_feed(fx: &Fixture, rows: &[i64]) -> Vec<(String, Vec<f32>)> {
    let e = fx.spec.engram.as_ref().unwrap();
    let cols = e.n_hash_cols();
    e.layer_ids
        .iter()
        .enumerate()
        .map(|(k, &il)| {
            (
                names::engram_rows(il),
                rows[k * cols..(k + 1) * cols]
                    .iter()
                    .map(|&v| v as f32)
                    .collect(),
            )
        })
        .collect()
}

/// A prompt split into chunks equals the prompt in one pass.
///
/// The split sizes are deliberately not multiples of the compressor's ratio of
/// 3, so groups start in one chunk and finish in the next — which is the whole
/// reason the cache needed replace-semantics for a carried partial group.
#[test]
fn chunked_prompt_equals_one_pass() {
    let fx = fixture();
    let ids: Vec<u32> = vec![5, 9, 2, 14, 7, 3, 11, 1, 6, 13, 4];
    let seq = ids.len();
    let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();

    let inputs = V41Inputs {
        engram_rows: fx.rows(&ids, 0, seq),
        ..Default::default()
    };
    let mut l = fx.loader();
    let mut pk = std::collections::HashMap::new();
    let (g, p, _) =
        build_deepseek_v41_prefill(&fx.spec, &mut l, seq, &inputs, &mut pk).expect("prefill");
    let want = run(g, &p, &[("input_ids", idf.as_slice())])[0].clone();
    let vocab = fx.spec.vocab_size;

    // The splits are chosen to cover every shape of the compressor's group
    // arithmetic against a ratio of 3: chunks that end mid-group, chunks that
    // end exactly on a boundary, chunks that consume a carried group exactly
    // (`[4, 2, 5]`, `[2, 4, 5]`), and single-token chunks.
    //
    // Note these do *not* detect a stale group left behind by an
    // exactly-consuming chunk — that heals itself, because the chunk after a
    // boundary declares no group input and then overwrites the entry. The cache
    // length assertions in `speculative_acceptance_commits_the_right_prefix` are
    // what pin that.
    for splits in [
        vec![4usize, 4, 3],
        vec![2, 5, 4],
        vec![1, 1, 9],
        vec![5, 6],
        vec![3, 3, 5],
        vec![6, 5],
        vec![3, 6, 2],
        vec![4, 2, 5],
        vec![2, 4, 5],
        vec![1, 2, 8],
    ] {
        assert_eq!(splits.iter().sum::<usize>(), seq);
        let mut cache = V41DecodeCache::new(&fx.spec);
        let mut start = 0usize;
        let mut last: Vec<f32> = Vec::new();
        for &k in &splits {
            let plan = V41ChunkPlan::new(&fx.spec, start, k);
            let inputs = V41Inputs {
                engram_rows: fx.rows(&ids, start, k),
                ..Default::default()
            };
            let mut l = fx.loader();
            let mut pk = std::collections::HashMap::new();
            let (g, p, cnames) =
                build_deepseek_v41_chunk(&fx.spec, &mut l, &plan, &inputs, &mut pk)
                    .unwrap_or_else(|e| panic!("chunk at {start}+{k}: {e}"));
            let chunk_ids: Vec<f32> = idf[start..start + k].to_vec();
            let cached = cache.chunk_inputs(&plan);
            let out = run(g, &p, &feed_with(&chunk_ids, &cached));
            drop(cached);
            cache.apply_chunk(&plan, &cnames, &out).expect("apply");
            last = out[0][(k - 1) * vocab..].to_vec();
            start += k;
        }
        close(
            &last,
            &want[(seq - 1) * vocab..],
            &format!("chunks {splits:?} vs one pass"),
        );
    }
}
