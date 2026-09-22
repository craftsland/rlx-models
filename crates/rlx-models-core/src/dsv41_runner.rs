// RLX — versatile ML compiler + runtime. GPLv3.
//! A **generate loop** for DeepSeek-V4.1: checkpoint directory in, text out.
//!
//! The graph builders in the sibling modules each produce one graph for one
//! step. This is what drives them: tokenize, run the prompt, sample, feed the
//! sampled token back, and stop on an end token or a budget.
//!
//! Three things make V4.1 awkward to drive, and they are most of what is here:
//!
//! * **The decode graph's shape depends on the position.** How many cached
//!   window rows exist, and which compressors fire, both change from step to
//!   step, so a naive loop recompiles every token. Compiled sessions are cached
//!   under a signature of exactly those two things, which collapses to a handful
//!   of distinct shapes once the window fills.
//! * **The Engram row ids are the host's job.** They are an n-gram hash over the
//!   token history, recomputed each step against the growing prefix.
//! * **The experts do not fit.** [`MoeExecution::Paged`] routes on the host and
//!   pages only the experts a token needs; see [`crate::dsv41_pager`].
//!
//! Weights are read through [`crate::dsv41_weights::StreamingLoader`], never
//! mmapped, so resident memory tracks the working set rather than the checkpoint.

use crate::dsv41::DeepseekV41Spec;
use crate::dsv41_decode::{V41DecodeCache, V41DecodePlan, build_deepseek_v41_decode};
use crate::dsv41_engram::{EngramHashPlan, compress_token_map};
use crate::dsv41_graph::{V41Inputs, build_deepseek_v41_prefill};
use crate::dsv41_pager::ExpertPager;
use crate::dsv41_quant::DEFAULT_BLOCK;
use crate::dsv41_weights::{StreamingLoader, WeightIndex};
#[cfg_attr(not(feature = "tokenizer"), allow(unused_imports))]
use anyhow::{Context, Result, anyhow, bail};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// How the routed experts are supplied to the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoeExecution {
    /// The whole bank is a graph parameter. Simple and fastest when it fits —
    /// which at the released size it does not.
    #[default]
    Resident,
    /// Route on the host, page in only the chosen experts, hand them to the
    /// graph as a gathered bank.
    Paged,
}

/// Runner-wide settings, as opposed to per-request sampling.
#[derive(Debug, Clone)]
pub struct RunnerOptions {
    pub device: Device,
    pub moe: MoeExecution,
    /// Byte budget for the paged expert cache. Ignored when
    /// [`MoeExecution::Resident`].
    pub expert_budget_bytes: u64,
    /// `quantization_config.weight_block_size[0]`; the released checkpoint's
    /// value is [`DEFAULT_BLOCK`].
    pub block: usize,
    /// Cap on distinct compiled decode sessions kept alive.
    pub max_cached_sessions: usize,
    /// Process the prompt in chunks of at most this many tokens.
    ///
    /// `None` runs it in one pass, which builds a graph sized for the whole
    /// prompt — fine for a short one, and a large allocation for a long one.
    /// Chunking trades a graph per chunk for a peak that does not grow with the
    /// prompt. The result is identical either way
    /// ([`crate::dsv41_chunk`] pins that).
    pub prefill_chunk: Option<usize>,
}

impl Default for RunnerOptions {
    fn default() -> Self {
        RunnerOptions {
            device: Device::Cpu,
            moe: MoeExecution::Resident,
            expert_budget_bytes: 8 << 30,
            block: DEFAULT_BLOCK,
            max_cached_sessions: 8,
            prefill_chunk: None,
        }
    }
}

/// One image, already turned into ViT patches.
///
/// The runner does not decode or resize images: turning a JPEG into patches
/// needs PIL's exact `ImageOps.pad` resampling, which this port has no reference
/// dump for, and a near-miss there is silently the wrong picture. The *planning*
/// side is ported and checked —
/// [`crate::dsv41_vision::plan_image_grid`] gives the patch grid to resize to,
/// and [`crate::dsv41_vision::num_image_tokens`] how many prompt slots it costs.
#[derive(Debug, Clone)]
pub struct ImagePatches {
    /// `[n_vit_h · n_vit_w, 3 · patch_size · patch_size]`, row-major over the
    /// grid, each patch laid out `(channel, y, x)`.
    pub patches: Vec<f32>,
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    /// Prompt position of the image's `IMAGE_START` token. The span runs for
    /// [`num_image_tokens`](crate::dsv41_vision::num_image_tokens) positions and
    /// must fit entirely inside the prompt.
    pub at: usize,
}

impl ImagePatches {
    /// The LLM token grid this image's patches reduce to.
    pub fn llm_grid(&self, spec: &DeepseekV41Spec) -> Result<(usize, usize)> {
        let r = spec
            .vision
            .as_ref()
            .ok_or_else(|| anyhow!("deepseek_v41: this checkpoint has no vision tower"))?
            .downsample_ratio
            .max(1);
        Ok((self.n_vit_h.div_ceil(r), self.n_vit_w.div_ceil(r)))
    }

    /// Prompt positions this image occupies, delimiters and newlines included.
    pub fn span_len(&self, spec: &DeepseekV41Spec) -> Result<usize> {
        let (h, w) = self.llm_grid(spec)?;
        Ok(crate::dsv41_vision::num_image_tokens(h, w))
    }
}

/// Per-request sampling.
#[derive(Debug, Clone)]
pub struct SampleOpts {
    pub max_new_tokens: usize,
    /// `<= 0` is greedy.
    pub temperature: f32,
    /// `0` disables.
    pub top_k: usize,
    /// `>= 1` disables.
    pub top_p: f32,
    pub seed: u64,
    /// Token ids that end generation.
    pub stop: Vec<u32>,
}

impl Default for SampleOpts {
    fn default() -> Self {
        SampleOpts {
            max_new_tokens: 64,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            stop: Vec::new(),
        }
    }
}

impl SampleOpts {
    pub fn greedy(max_new_tokens: usize) -> Self {
        SampleOpts {
            max_new_tokens,
            ..Default::default()
        }
    }
}

/// The token vocabulary, as the Engram's n-gram hash needs to see it.
///
/// The hash is keyed on a *compressed* token id, which folds the vocabulary down
/// to `engram_compressed_vocab_size` by merging pieces the memory should not
/// distinguish. That folding is defined over both the raw piece and its decoded
/// text, so a checkpoint with an Engram cannot be driven from ids alone — the
/// caller supplies this, or builds it from `tokenizer.json` with the `tokenizer`
/// feature.
#[derive(Debug, Clone)]
pub struct Vocab {
    /// Raw piece per id, e.g. `\u{120}the`.
    pub pieces: Vec<String>,
    /// Decoded text per id.
    pub decoded: Vec<String>,
}

impl Vocab {
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }
}

/// What one request produced.
#[derive(Debug, Clone)]
pub struct Generation {
    /// The newly generated ids, excluding the prompt.
    pub tokens: Vec<u32>,
    /// Decoded text, empty when the runner has no tokenizer.
    pub text: String,
    pub prompt_tokens: usize,
    /// Set when generation ended on a token in [`SampleOpts::stop`].
    pub stopped_on: Option<u32>,
}

/// What makes two decode steps share a compiled graph.
///
/// Every field of the plan that changes an input's *shape* has to be here, and
/// `len_before`/`len_after` are easy to leave out because they do not appear in
/// the step's control flow — only in the width of the compressed-cache inputs.
/// Omitting them lets two positions with the same `(fires, group_filled)` share
/// a graph built for a shorter cache, which decodes fluently and is wrong.
///
/// Values that change per position without changing a shape must **not** be
/// baked into the graph at all; the Engram's n-gram rows are a graph input for
/// exactly that reason ([`crate::dsv41_decode::names::engram_rows`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StepShape {
    cache_len: usize,
    /// `(source, ratio, len_before, len_after, fires, group_filled)`.
    sources: Vec<(usize, usize, usize, usize, bool, usize)>,
}

impl StepShape {
    fn of(plan: &V41DecodePlan) -> Self {
        StepShape {
            cache_len: plan.cache_len,
            sources: plan
                .sources
                .iter()
                .map(|s| {
                    (
                        s.source_layer,
                        s.ratio,
                        s.len_before,
                        s.len_after,
                        s.fires,
                        s.group_filled,
                    )
                })
                .collect(),
        }
    }

    fn key(&self) -> String {
        format!("{}|{:?}", self.cache_len, self.sources)
    }
}

/// Drives DeepSeek-V4.1 end to end.
pub struct V41Runner {
    spec: DeepseekV41Spec,
    dir: PathBuf,
    opts: RunnerOptions,
    #[cfg(feature = "tokenizer")]
    tokenizer: Option<tokenizers::Tokenizer>,
    /// Vocabulary for the Engram's hash, and for decoding output.
    vocab: Option<Vocab>,
    /// `(hash plan, vocab -> compressed id)`, when the checkpoint has an Engram.
    engram: Option<(EngramHashPlan, Vec<u32>)>,
    pager: Option<std::sync::Arc<ExpertPager>>,
    /// Built lazily on the first paged step, because it compiles a graph per
    /// layer and a resident run never needs it.
    stepper: Option<crate::dsv41_paged::PagedStepper>,
    /// Same, for the batched prompt pass.
    prefiller: Option<crate::dsv41_paged::PagedPrefill>,
    /// The loader the paged path reuses; its index is read once.
    paged_loader: Option<StreamingLoader>,
    sessions: HashMap<StepShape, CompiledGraph>,
    /// Insert order, for evicting the oldest compiled session.
    session_order: Vec<StepShape>,
}

impl V41Runner {
    /// Open a checkpoint directory: `config.json`, `*.safetensors`, and
    /// optionally `tokenizer.json`.
    pub fn open(dir: impl AsRef<Path>, opts: RunnerOptions) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let cfg_path = dir.join("config.json");
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&cfg_path).with_context(|| format!("read {cfg_path:?}"))?,
        )
        .with_context(|| format!("parse {cfg_path:?}"))?;
        let spec = DeepseekV41Spec::from_config(&cfg)?;
        spec.validate()?;

        let pager = match opts.moe {
            MoeExecution::Paged => Some(std::sync::Arc::new(ExpertPager::new(
                WeightIndex::open(&dir)?,
                opts.block,
                opts.expert_budget_bytes,
            ))),
            MoeExecution::Resident => None,
        };

        let mut r = V41Runner {
            spec,
            dir,
            opts,
            #[cfg(feature = "tokenizer")]
            tokenizer: None,
            vocab: None,
            engram: None,
            pager,
            stepper: None,
            prefiller: None,
            paged_loader: None,
            sessions: HashMap::new(),
            session_order: Vec::new(),
        };
        #[cfg(feature = "tokenizer")]
        r.load_tokenizer()?;
        r.engram = r.build_engram()?;
        Ok(r)
    }

    /// Open with a caller-supplied vocabulary, for drivers that tokenize
    /// elsewhere. Always available, feature or not.
    pub fn open_with_vocab(
        dir: impl AsRef<Path>,
        opts: RunnerOptions,
        vocab: Vocab,
    ) -> Result<Self> {
        let mut r = Self::open(dir, opts)?;
        r.set_vocab(vocab)?;
        Ok(r)
    }

    /// Replace the vocabulary and rebuild the Engram hash plan around it.
    pub fn set_vocab(&mut self, vocab: Vocab) -> Result<()> {
        if vocab.pieces.len() != vocab.decoded.len() {
            bail!(
                "deepseek_v41: vocab has {} pieces but {} decoded strings",
                vocab.pieces.len(),
                vocab.decoded.len()
            );
        }
        if vocab.len() < self.spec.vocab_size {
            bail!(
                "deepseek_v41: vocab has {} entries, the model's vocab_size is {}",
                vocab.len(),
                self.spec.vocab_size
            );
        }
        self.vocab = Some(vocab);
        self.engram = self.build_engram()?;
        Ok(())
    }

    /// Read `tokenizer.json` from the checkpoint directory, if present, and take
    /// the vocabulary from it.
    #[cfg(feature = "tokenizer")]
    fn load_tokenizer(&mut self) -> Result<()> {
        let path = self.dir.join("tokenizer.json");
        if !path.exists() {
            return Ok(());
        }
        let tk = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| anyhow!("deepseek_v41: load {path:?}: {e}"))?;
        let n = self.spec.vocab_size;
        let mut pieces = Vec::with_capacity(n);
        let mut decoded = Vec::with_capacity(n);
        for i in 0..n {
            let p = tk.id_to_token(i as u32).unwrap_or_default();
            decoded.push(tk.decode(&[i as u32], false).unwrap_or_else(|_| p.clone()));
            pieces.push(p);
        }
        self.vocab = Some(Vocab { pieces, decoded });
        self.tokenizer = Some(tk);
        Ok(())
    }

    pub fn spec(&self) -> &DeepseekV41Spec {
        &self.spec
    }

    pub fn pager_stats(&self) -> Option<crate::dsv41_pager::PagerStats> {
        self.pager.as_ref().map(|p| p.stats())
    }

    /// The Engram hash plan, built from the vocabulary — `None` until there is
    /// one.
    ///
    /// The n-gram memory is keyed on a *compressed* token id, which folds the
    /// full vocabulary down to `engram_compressed_vocab_size` by merging pieces
    /// that differ only in ways the memory should ignore. That folding reads the
    /// token strings, so a checkpoint with an Engram cannot be driven from ids
    /// alone.
    ///
    /// A missing vocabulary is not reported here: [`Self::open_with_vocab`] opens
    /// first and supplies one second, so failing at open time would make that
    /// impossible. [`Self::check_ready`] raises it at the point it actually
    /// matters.
    fn build_engram(&self) -> Result<Option<(EngramHashPlan, Vec<u32>)>> {
        let (Some(e), Some(v)) = (self.spec.engram.as_ref(), self.vocab.as_ref()) else {
            return Ok(None);
        };
        let (map, _) = compress_token_map(&v.decoded, &v.pieces);
        Ok(Some((EngramHashPlan::new(e, &map)?, map)))
    }

    /// Everything the checkpoint needs that the caller has to supply.
    fn check_ready(&self) -> Result<()> {
        if self.spec.engram.is_some() && self.engram.is_none() {
            bail!(
                "deepseek_v41: this checkpoint has an Engram, which needs a vocabulary — build \
                 with the `tokenizer` feature so `tokenizer.json` is read, or open with \
                 `V41Runner::open_with_vocab`"
            );
        }
        Ok(())
    }

    fn loader(&self) -> Result<StreamingLoader> {
        StreamingLoader::open(&self.dir, self.opts.block)
    }

    fn compile_opts(&self) -> CompileOptions {
        crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            self.opts.device,
        )
    }

    /// The Engram row inputs one decode step needs, split per layer.
    ///
    /// `hash_ids` returns the rows for every Engram layer interleaved; the graph
    /// takes one input per layer.
    fn engram_feed(&self, rows: &[i64]) -> Vec<(String, Vec<f32>)> {
        let Some(e) = &self.spec.engram else {
            return Vec::new();
        };
        let cols = e.n_hash_cols();
        e.layer_ids
            .iter()
            .enumerate()
            .filter_map(|(k, &il)| {
                let base = k * cols;
                rows.get(base..base + cols).map(|r| {
                    (
                        crate::dsv41_decode::names::engram_rows(il),
                        r.iter().map(|&v| v as f32).collect(),
                    )
                })
            })
            .collect()
    }

    /// Engram row ids for `chunk`, given everything before it.
    fn engram_rows(&self, history: &[u32], chunk: &[u32]) -> Vec<i64> {
        let Some((plan, map)) = &self.engram else {
            return Vec::new();
        };
        let c = |ids: &[u32]| -> Vec<u32> {
            ids.iter()
                .map(|&i| map.get(i as usize).copied().unwrap_or(0))
                .collect()
        };
        plan.hash_ids(&c(chunk), None, &c(history))
    }

    /// Run the whole prompt in one batched pass.
    ///
    /// Returns the last position's logits — what the next token is sampled from
    /// — and, when `cache` is given, leaves it exactly where stepping the same
    /// prompt would have: one graph run instead of `seq` of them.
    fn prefill(&mut self, ids: &[u32], cache: Option<&mut V41DecodeCache>) -> Result<Vec<f32>> {
        if ids.is_empty() {
            bail!("deepseek_v41: cannot prefill an empty prompt");
        }
        self.check_ready()?;
        let want_cache = cache.is_some();
        if self.opts.moe == MoeExecution::Paged {
            return self.prefill_paged(ids, cache);
        }
        if let Some(chunk) = self.opts.prefill_chunk.filter(|c| *c > 0 && ids.len() > *c) {
            let Some(c) = cache else {
                bail!(
                    "deepseek_v41: chunked prefill builds its state in the cache, so it needs \
                     one — use `prefill_chunk: None` for a cacheless scoring pass"
                );
            };
            return self.prefill_chunked(ids, chunk, c);
        }
        let inputs = V41Inputs {
            engram_rows: self.engram_rows(&[], ids),
            image_positions: Vec::new(),
            emit_cache: want_cache,
            ..Default::default()
        };
        let mut loader = self.loader()?;
        let mut packed = HashMap::new();
        let (g, params, cache_names) =
            build_deepseek_v41_prefill(&self.spec, &mut loader, ids.len(), &inputs, &mut packed)?;
        let mut sess = Session::new(self.opts.device).compile_with(g, &self.compile_opts());
        for (n, d) in &params {
            sess.set_param(n, d);
        }
        let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
        let out = sess.run(&[("input_ids", idf.as_slice())]);
        let v = self.spec.vocab_size;
        if out[0].len() != ids.len() * v {
            bail!(
                "deepseek_v41: prefill produced {} logits, expected {} x {v}",
                out[0].len(),
                ids.len()
            );
        }
        if let Some(c) = cache {
            // the cache tensors follow `logits` in graph-output order
            let vals: Vec<Vec<f32>> = out[out.len() - cache_names.len()..].to_vec();
            c.prime(&cache_names, &vals)?;
        }
        Ok(out[0][(ids.len() - 1) * v..].to_vec())
    }

    /// Process the prompt `chunk` tokens at a time, threading the cache from one
    /// chunk to the next.
    ///
    /// Each chunk is a `k`-token step at an absolute offset, so the graph is
    /// sized for the chunk rather than the prompt. The final chunk's last row is
    /// the logits the next token is sampled from, exactly as a single pass would
    /// have produced.
    fn prefill_chunked(
        &mut self,
        ids: &[u32],
        chunk: usize,
        cache: &mut V41DecodeCache,
    ) -> Result<Vec<f32>> {
        use crate::dsv41_chunk::build_deepseek_v41_chunk;
        use crate::dsv41_decode::V41ChunkPlan;

        let v = self.spec.vocab_size;
        let mut last = Vec::new();
        let mut start = 0usize;
        while start < ids.len() {
            let k = chunk.min(ids.len() - start);
            let plan = V41ChunkPlan::new(&self.spec, start, k);
            let inputs = V41Inputs {
                engram_rows: self.engram_rows(&ids[..start], &ids[start..start + k]),
                ..Default::default()
            };
            let mut loader = self.loader()?;
            let mut packed = HashMap::new();
            let (g, params, names) =
                build_deepseek_v41_chunk(&self.spec, &mut loader, &plan, &inputs, &mut packed)
                    .with_context(|| format!("build chunk at {start}+{k}"))?;
            let mut sess = Session::new(self.opts.device).compile_with(g, &self.compile_opts());
            for (n, d) in &params {
                sess.set_param(n, d);
            }
            let idf: Vec<f32> = ids[start..start + k].iter().map(|&i| i as f32).collect();
            let cached = cache.chunk_inputs(&plan);
            let mut feed: Vec<(&str, &[f32])> = vec![("input_ids", idf.as_slice())];
            for (n, val) in &cached {
                feed.push((n.as_str(), *val));
            }
            let out = sess.run(&feed);
            drop(cached);
            cache.apply_chunk(&plan, &names, &out)?;
            last = out[0][(k - 1) * v..k * v].to_vec();
            start += k;
        }
        Ok(last)
    }

    /// The paged prompt pass: route every row on the host, page in the union of
    /// experts they name, and hand that to the graph as a gathered bank.
    ///
    /// The union grows with the prompt, so this saves less the longer the prompt
    /// is — see [`crate::dsv41_paged::PagedPrefill`]. Decode is unaffected: its
    /// union stays at `top_k` however long the context gets.
    fn prefill_paged(
        &mut self,
        ids: &[u32],
        cache: Option<&mut V41DecodeCache>,
    ) -> Result<Vec<f32>> {
        if self.paged_loader.is_none() {
            self.paged_loader = Some(self.loader()?);
        }
        if self.prefiller.is_none() {
            let pager = self
                .pager
                .as_ref()
                .ok_or_else(|| anyhow!("deepseek_v41: paged execution without a pager"))?
                .clone();
            let w = self.paged_loader.as_mut().expect("just created");
            self.prefiller = Some(crate::dsv41_paged::PagedPrefill::new(
                &self.spec,
                w,
                pager,
                self.opts.device,
            )?);
        }
        let rows = self.engram_rows(&[], ids);
        let want_cache = cache.is_some();
        let p = self.prefiller.as_mut().expect("built above");
        let w = self.paged_loader.as_mut().expect("built above");
        let out = p.run(w, ids, &rows, want_cache)?;
        if let Some(c) = cache {
            c.prime(&out.cache_names, &out.cache_values)?;
        }
        Ok(out.logits)
    }

    /// The prompt's last-position logits, without touching any cache.
    pub fn prefill_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        self.prefill(ids, None)
    }

    /// One decode step at `pos`, consuming `token` and returning its logits.
    fn step(
        &mut self,
        pos: usize,
        token: u32,
        history: &[u32],
        cache: &mut V41DecodeCache,
    ) -> Result<Vec<f32>> {
        if self.opts.moe == MoeExecution::Paged {
            return self.step_paged(pos, token, history, cache);
        }
        let plan = V41DecodePlan::new(&self.spec, pos);
        let shape = StepShape::of(&plan);
        let names = self.ensure_session(&shape, pos, history, token)?;

        let idf = [token as f32];
        let rows = self.engram_rows(history, &[token]);
        let eng = self.engram_feed(&rows);
        let cached = cache.step_inputs(&plan);
        let mut feed: Vec<(&str, &[f32])> = vec![("input_ids", idf.as_slice())];
        for (n, v) in &eng {
            feed.push((n.as_str(), v.as_slice()));
        }
        for (n, v) in &cached {
            feed.push((n.as_str(), *v));
        }
        let sess = self
            .sessions
            .get_mut(&shape)
            .expect("ensure_session inserted it");
        let out = sess.run(&feed);
        cache.apply(&plan, &names, &out)?;
        Ok(out[0].clone())
    }

    /// The paged path: route on the host and page in only the experts this
    /// token names, instead of putting every layer's bank in the graph.
    fn step_paged(
        &mut self,
        pos: usize,
        token: u32,
        history: &[u32],
        cache: &mut V41DecodeCache,
    ) -> Result<Vec<f32>> {
        let plan = V41DecodePlan::new(&self.spec, pos);
        let rows = self.engram_rows(history, &[token]);
        if self.paged_loader.is_none() {
            self.paged_loader = Some(self.loader()?);
        }
        if self.stepper.is_none() {
            let pager = self
                .pager
                .as_ref()
                .ok_or_else(|| anyhow!("deepseek_v41: paged execution without a pager"))?
                .clone();
            let w = self.paged_loader.as_mut().expect("just created");
            self.stepper = Some(crate::dsv41_paged::PagedStepper::new(
                &self.spec,
                w,
                pager,
                self.opts.device,
            )?);
        }
        let key = StepShape::of(&plan).key();
        let eng = self.engram_feed(&rows);
        let cached: Vec<(String, &[f32])> = cache.step_inputs(&plan);
        let stepper = self.stepper.as_mut().expect("built above");
        let w = self.paged_loader.as_mut().expect("built above");
        let (logits, names, vals) = stepper.step(w, &plan, token, &eng, &cached, &key)?;
        drop(cached);
        cache.apply(&plan, &names, &vals)?;
        Ok(logits)
    }

    /// Compile the decode graph for `shape` if it is not already cached, and
    /// return its output names.
    ///
    /// The names are a property of the graph, so they are recomputed alongside
    /// it; caching them with the session would be the tidier shape but the
    /// builder returns them together and they are only a few strings.
    fn ensure_session(
        &mut self,
        shape: &StepShape,
        pos: usize,
        history: &[u32],
        token: u32,
    ) -> Result<Vec<String>> {
        let inputs = V41Inputs {
            engram_rows: self.engram_rows(history, &[token]),
            image_positions: Vec::new(),
            ..Default::default()
        };
        let mut loader = self.loader()?;
        let mut packed = HashMap::new();
        let (g, params, names) =
            build_deepseek_v41_decode(&self.spec, &mut loader, pos, &inputs, &mut packed)
                .with_context(|| format!("build decode graph at position {pos}"))?;
        if self.sessions.contains_key(shape) {
            return Ok(names);
        }
        let mut sess = Session::new(self.opts.device).compile_with(g, &self.compile_opts());
        for (n, d) in &params {
            sess.set_param(n, d);
        }
        self.sessions.insert(shape.clone(), sess);
        self.session_order.push(shape.clone());
        while self.session_order.len() > self.opts.max_cached_sessions.max(1) {
            let old = self.session_order.remove(0);
            self.sessions.remove(&old);
        }
        Ok(names)
    }

    /// Generate from token ids, returning only the new ids.
    ///
    /// The prompt goes through the batched prefill graph in one pass, which
    /// primes the decode cache; generation then steps from position `len`.
    pub fn generate_ids(&mut self, prompt: &[u32], opts: &SampleOpts) -> Result<Generation> {
        if prompt.is_empty() {
            bail!("deepseek_v41: cannot generate from an empty prompt");
        }
        self.check_ready()?;
        for (i, &t) in prompt.iter().enumerate() {
            if t as usize >= self.spec.vocab_size {
                bail!(
                    "deepseek_v41: prompt token {i} is id {t}, outside the vocabulary of {}",
                    self.spec.vocab_size
                );
            }
        }
        let mut cache = V41DecodeCache::new(&self.spec);
        let mut logits = self.prefill(prompt, Some(&mut cache))?;
        let mut history: Vec<u32> = prompt.to_vec();
        history.reserve(opts.max_new_tokens);

        let mut rng = SplitMix::new(opts.seed);
        let mut out = Vec::new();
        let mut stopped_on = None;
        for _ in 0..opts.max_new_tokens {
            let next = sample(&logits, opts, &mut rng)?;
            if opts.stop.contains(&next) {
                stopped_on = Some(next);
                break;
            }
            out.push(next);
            let pos = history.len();
            logits = self.step(pos, next, &history, &mut cache)?;
            history.push(next);
        }

        let text = self.decode_ids(&out)?;
        Ok(Generation {
            tokens: out,
            text,
            prompt_tokens: prompt.len(),
            stopped_on,
        })
    }

    #[cfg(feature = "tokenizer")]
    fn decode_ids(&self, ids: &[u32]) -> Result<String> {
        let (Some(tk), false) = (&self.tokenizer, ids.is_empty()) else {
            return Ok(String::new());
        };
        tk.decode(ids, true)
            .map_err(|e| anyhow!("deepseek_v41: decode generated ids: {e}"))
    }

    /// Without a tokenizer there is nothing to detokenize with; the ids are
    /// still returned.
    #[cfg(not(feature = "tokenizer"))]
    fn decode_ids(&self, _ids: &[u32]) -> Result<String> {
        Ok(String::new())
    }

    /// Step a prompt one position at a time, the way a runner without a primed
    /// cache has to. Exposed so a test can hold the fast and slow paths side by
    /// side; generation itself always prefills.
    #[doc(hidden)]
    pub fn step_prompt_for_test(
        &mut self,
        ids: &[u32],
        cache: &mut V41DecodeCache,
    ) -> Result<Vec<f32>> {
        self.check_ready()?;
        let mut history: Vec<u32> = Vec::with_capacity(ids.len());
        let mut logits = Vec::new();
        for (pos, &t) in ids.iter().enumerate() {
            logits = self.step(pos, t, &history, cache)?;
            history.push(t);
        }
        Ok(logits)
    }

    /// Run the vision tower and return its language-model embeddings,
    /// `[n_tokens, dim]`.
    pub fn image_embeddings(&mut self, img: &ImagePatches) -> Result<Vec<f32>> {
        let mut loader = self.loader()?;
        let mut packed = HashMap::new();
        let (g, params) = crate::dsv41_vision::build_v41_vision(
            &self.spec,
            &mut loader,
            img.n_vit_h,
            img.n_vit_w,
            &mut packed,
        )?;
        let mut sess = Session::new(self.opts.device).compile_with(g, &self.compile_opts());
        for (n, d) in &params {
            sess.set_param(n, d);
        }
        Ok(sess.run(&[("patches", img.patches.as_slice())])[0].clone())
    }

    /// The prompt's embeddings with each image's span written in.
    ///
    /// The span is **structured**, not a flat run of aligner rows:
    ///
    /// ```text
    ///   IMAGE_START  (IMAGE × n_llm_w  NEWLINE) × n_llm_h  IMAGE_END
    /// ```
    ///
    /// Only the `IMAGE` slots take aligner output; the three delimiters take
    /// their own learned embeddings. Writing the tower's rows contiguously across
    /// the span instead would shift every row after the first line — a model that
    /// still produces fluent text about the wrong picture.
    ///
    /// Also returns the image mask, which the MoE router needs: a position inside
    /// an image span routes through `gate.bias_vl` rather than `gate.bias`.
    fn embed_prompt(
        &mut self,
        ids: &[u32],
        images: &[ImagePatches],
    ) -> Result<(Vec<f32>, Vec<bool>)> {
        use crate::dsv41_vision::{ImageTokenType, image_token_types};

        let d = self.spec.dim;
        let mut loader = self.loader()?;
        let (table, shape) = crate::weight_loader::WeightLoader::take(&mut loader, "embed.weight")?;
        if shape != vec![self.spec.vocab_size, d] {
            bail!(
                "deepseek_v41: embed.weight is {shape:?}, expected [{}, {d}]",
                self.spec.vocab_size
            );
        }
        let mut embeds = vec![0f32; ids.len() * d];
        for (i, &t) in ids.iter().enumerate() {
            let src = t as usize * d;
            embeds[i * d..(i + 1) * d].copy_from_slice(&table[src..src + d]);
        }

        let mut mask = vec![false; ids.len()];
        if images.is_empty() {
            return Ok((embeds, mask));
        }
        let delim = |name: &str, l: &mut StreamingLoader| -> Result<Vec<f32>> {
            let (v, s) = crate::weight_loader::WeightLoader::take(l, name)?;
            if v.len() != d {
                bail!("deepseek_v41: `{name}` is {s:?}, expected [{d}]");
            }
            Ok(v)
        };
        let start = delim("image_start", &mut loader)?;
        let end = delim("image_end", &mut loader)?;
        let newline = delim("image_newline", &mut loader)?;

        for img in images {
            let (n_h, n_w) = img.llm_grid(&self.spec)?;
            let span = crate::dsv41_vision::num_image_tokens(n_h, n_w);
            if img.at + span > ids.len() {
                bail!(
                    "deepseek_v41: an image at position {} needs {span} tokens ({n_h}x{n_w} grid \
                     plus newlines and delimiters) but the prompt is only {} long",
                    img.at,
                    ids.len()
                );
            }
            let rows = self.image_embeddings(img)?;
            let n_rows = rows.len() / d.max(1);
            if n_rows != n_h * n_w {
                bail!("deepseek_v41: the aligner produced {n_rows} rows for a {n_h}x{n_w} grid");
            }
            let mut next = 0usize;
            for (k, t) in image_token_types(n_h, n_w).into_iter().enumerate() {
                let dst = (img.at + k) * d;
                let src: &[f32] = match t {
                    ImageTokenType::Start => &start,
                    ImageTokenType::End => &end,
                    ImageTokenType::NewLine => &newline,
                    ImageTokenType::Image => {
                        let s = &rows[next * d..(next + 1) * d];
                        next += 1;
                        s
                    }
                };
                embeds[dst..dst + d].copy_from_slice(src);
            }
            debug_assert_eq!(next, n_rows, "every aligner row is placed");
            for m in &mut mask[img.at..img.at + span] {
                *m = true;
            }
        }
        Ok((embeds, mask))
    }

    /// Generate with one or more images spliced into the prompt.
    ///
    /// The image positions in `prompt` should be `image_token_id`; their ids are
    /// never looked up, since the tower's rows replace them.
    pub fn generate_with_images(
        &mut self,
        prompt: &[u32],
        images: &[ImagePatches],
        opts: &SampleOpts,
    ) -> Result<Generation> {
        if images.is_empty() {
            return self.generate_ids(prompt, opts);
        }
        if self.spec.vision.is_none() {
            bail!("deepseek_v41: this checkpoint has no vision tower");
        }
        self.check_ready()?;
        let (embeds, mask) = self.embed_prompt(prompt, images)?;
        let mut cache = V41DecodeCache::new(&self.spec);
        let mut logits = self.prefill_embeds(prompt.len(), &embeds, &mask, Some(&mut cache))?;

        let mut history: Vec<u32> = prompt.to_vec();
        let mut rng = SplitMix::new(opts.seed);
        let mut out = Vec::new();
        let mut stopped_on = None;
        for _ in 0..opts.max_new_tokens {
            let next = sample(&logits, opts, &mut rng)?;
            if opts.stop.contains(&next) {
                stopped_on = Some(next);
                break;
            }
            out.push(next);
            let pos = history.len();
            logits = self.step(pos, next, &history, &mut cache)?;
            history.push(next);
        }
        let text = self.decode_ids(&out)?;
        Ok(Generation {
            tokens: out,
            text,
            prompt_tokens: prompt.len(),
            stopped_on,
        })
    }

    /// Batched prefill from precomputed embeddings.
    fn prefill_embeds(
        &mut self,
        seq: usize,
        embeds: &[f32],
        image_positions: &[bool],
        cache: Option<&mut V41DecodeCache>,
    ) -> Result<Vec<f32>> {
        let want_cache = cache.is_some();
        let inputs = V41Inputs {
            engram_rows: Vec::new(),
            image_positions: image_positions.to_vec(),
            emit_cache: want_cache,
            embeds: true,
            ..Default::default()
        };
        // The Engram hashes token ids, which an image position does not have; the
        // reference excludes them from every n-gram, which is what the `alive`
        // mask on `hash_ids` is for.
        let inputs = match &self.engram {
            None => inputs,
            Some(_) => bail!(
                "deepseek_v41: image prompts with an Engram need the n-gram rows computed \
                 against the image mask; not wired yet"
            ),
        };
        let mut loader = self.loader()?;
        let mut packed = HashMap::new();
        let (g, params, cache_names) =
            build_deepseek_v41_prefill(&self.spec, &mut loader, seq, &inputs, &mut packed)?;
        let mut sess = Session::new(self.opts.device).compile_with(g, &self.compile_opts());
        for (n, d) in &params {
            sess.set_param(n, d);
        }
        let out = sess.run(&[("inputs_embeds", embeds)]);
        let v = self.spec.vocab_size;
        if let Some(c) = cache {
            let vals: Vec<Vec<f32>> = out[out.len() - cache_names.len()..].to_vec();
            c.prime(&cache_names, &vals)?;
        }
        Ok(out[0][(seq - 1) * v..].to_vec())
    }

    /// Batched prefill that also primes `cache`.
    #[doc(hidden)]
    pub fn prefill_into_test(
        &mut self,
        ids: &[u32],
        cache: &mut V41DecodeCache,
    ) -> Result<Vec<f32>> {
        self.prefill(ids, Some(cache))
    }

    /// One decode step, for the same side-by-side comparison.
    #[doc(hidden)]
    pub fn step_for_test(
        &mut self,
        pos: usize,
        token: u32,
        history: &[u32],
        cache: &mut V41DecodeCache,
    ) -> Result<Vec<f32>> {
        self.step(pos, token, history, cache)
    }

    /// Batched prefill from precomputed embeddings, for the side-by-side
    /// comparison against the id path.
    #[doc(hidden)]
    pub fn prefill_embeds_for_test(
        &mut self,
        seq: usize,
        embeds: &[f32],
        image_positions: &[bool],
    ) -> Result<Vec<f32>> {
        self.prefill_embeds(seq, embeds, image_positions, None)
    }

    /// Generate with DSpark speculative decoding.
    ///
    /// Greedy only, and deliberately so: greedy speculation is *exact* — every
    /// committed token is one the backbone itself would have produced — so this
    /// returns the same ids as [`Self::generate_ids`] with a greedy `SampleOpts`,
    /// just in fewer backbone passes when the draft head is right. Sampling
    /// speculatively needs the modified rejection rule, which this does not
    /// implement.
    ///
    /// Returns the generation and the acceptance statistics, which are what say
    /// whether drafting is paying for itself on a given model.
    pub fn generate_speculative(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        stop: &[u32],
    ) -> Result<(Generation, crate::dsv41_speculative::SpeculativeStats)> {
        use crate::dsv41_speculative::{SpeculativeDecoder, SpeculativeState, greedy};

        if prompt.is_empty() {
            bail!("deepseek_v41: cannot generate from an empty prompt");
        }
        self.check_ready()?;
        let mut dec = SpeculativeDecoder::new(&self.spec, self.opts.device)?;

        // the prompt pass has to emit `main_hidden`: the draft stages attend over
        // the main model's stream, so their caches are seeded from it
        let mut cache = V41DecodeCache::new(&self.spec);
        let (logits, main_hidden) = self.prefill_with_main_hidden(prompt, &mut cache)?;
        let mut loader = self.loader()?;
        dec.seed(&mut loader, &main_hidden, prompt.len())?;

        let d = self.spec.dim * self.spec.dspark_target_layer_ids.len();
        let n = prompt.len();
        let mut state = SpeculativeState {
            pos: n - 1,
            token: prompt[n - 1],
            main_hidden: main_hidden[(n - 1) * d..n * d].to_vec(),
            next_logits: logits,
        };
        let mut history: Vec<u32> = prompt.to_vec();
        let mut out: Vec<u32> = Vec::new();
        let mut stopped_on = None;

        let spec = self.spec.clone();
        let engram = |hist: &[u32], chunk: &[u32]| -> Vec<i64> {
            let Some((plan, map)) = &self.engram else {
                return Vec::new();
            };
            let c = |ids: &[u32]| -> Vec<u32> {
                ids.iter()
                    .map(|&i| map.get(i as usize).copied().unwrap_or(0))
                    .collect()
            };
            let _ = &spec;
            plan.hash_ids(&c(chunk), None, &c(hist))
        };

        'outer: while out.len() < max_new_tokens {
            let round = dec.round(&mut loader, &mut state, &mut history, &mut cache, &engram)?;
            for t in round.tokens {
                if stop.contains(&t) {
                    stopped_on = Some(t);
                    break 'outer;
                }
                out.push(t);
                if out.len() >= max_new_tokens {
                    break 'outer;
                }
            }
            // `greedy` here only guards against a degenerate round producing
            // nothing, which would spin forever
            let _ = greedy(&state.next_logits);
        }

        let text = self.decode_ids(&out)?;
        Ok((
            Generation {
                tokens: out,
                text,
                prompt_tokens: prompt.len(),
                stopped_on,
            },
            dec.stats(),
        ))
    }

    /// Batched prefill that also returns `main_hidden`, which DSpark needs.
    fn prefill_with_main_hidden(
        &mut self,
        ids: &[u32],
        cache: &mut V41DecodeCache,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let inputs = V41Inputs {
            engram_rows: self.engram_rows(&[], ids),
            emit_main_hidden: true,
            emit_cache: true,
            ..Default::default()
        };
        let mut loader = self.loader()?;
        let mut packed = HashMap::new();
        let (g, params, cache_names) =
            build_deepseek_v41_prefill(&self.spec, &mut loader, ids.len(), &inputs, &mut packed)?;
        let mut sess = Session::new(self.opts.device).compile_with(g, &self.compile_opts());
        for (n, d) in &params {
            sess.set_param(n, d);
        }
        let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
        let out = sess.run(&[("input_ids", idf.as_slice())]);
        let v = self.spec.vocab_size;
        let vals: Vec<Vec<f32>> = out[out.len() - cache_names.len()..].to_vec();
        cache.prime(&cache_names, &vals)?;
        Ok((out[0][(ids.len() - 1) * v..].to_vec(), out[1].clone()))
    }

    /// Prefill returning `main_hidden` too, for driving the speculative decoder
    /// directly.
    #[doc(hidden)]
    pub fn prefill_with_main_hidden_for_test(
        &mut self,
        ids: &[u32],
        cache: &mut V41DecodeCache,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.prefill_with_main_hidden(ids, cache)
    }

    /// Tokenize, generate, detokenize.
    #[cfg(feature = "tokenizer")]
    pub fn generate(&mut self, prompt: &str, opts: &SampleOpts) -> Result<Generation> {
        let tk = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("deepseek_v41: no tokenizer.json in {:?}", self.dir))?;
        let enc = tk
            .encode(prompt, false)
            .map_err(|e| anyhow!("deepseek_v41: encode prompt: {e}"))?;
        let ids = enc.get_ids().to_vec();
        self.generate_ids(&ids, opts)
    }
}

/// splitmix64, so a seed reproduces a run exactly.
struct SplitMix(u64);

impl SplitMix {
    fn new(seed: u64) -> Self {
        SplitMix(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

/// Pick the next token.
///
/// Greedy at `temperature <= 0`, and greedy is *exact*: ties go to the lowest
/// id, so a run is reproducible rather than dependent on iteration order.
fn sample(logits: &[f32], opts: &SampleOpts, rng: &mut SplitMix) -> Result<u32> {
    if logits.is_empty() {
        bail!("deepseek_v41: empty logits");
    }
    if opts.temperature <= 0.0 {
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        return Ok(best as u32);
    }

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    if opts.top_k > 0 {
        idx.truncate(opts.top_k.max(1));
    }

    let m = idx
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - m) / opts.temperature).exp())
        .collect();
    let sum: f32 = p.iter().sum();
    if !(sum.is_finite() && sum > 0.0) {
        // every candidate underflowed; fall back to the most likely one
        return Ok(idx[0] as u32);
    }
    for v in p.iter_mut() {
        *v /= sum;
    }

    if opts.top_p < 1.0 {
        let mut acc = 0f32;
        let mut keep = p.len();
        for (i, &v) in p.iter().enumerate() {
            acc += v;
            if acc >= opts.top_p {
                keep = i + 1;
                break;
            }
        }
        idx.truncate(keep);
        p.truncate(keep);
        let s: f32 = p.iter().sum();
        for v in p.iter_mut() {
            *v /= s;
        }
    }

    let r = rng.next_f32();
    let mut acc = 0f32;
    for (i, &v) in p.iter().enumerate() {
        acc += v;
        if r < acc {
            return Ok(idx[i] as u32);
        }
    }
    Ok(*idx.last().expect("non-empty") as u32)
}
