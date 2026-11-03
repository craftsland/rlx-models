// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! The whole model with its routed experts **paged from disk**.
//!
//! At `UD-IQ1_S` the 71 MoE layers' routed banks are ~215 GB of the 233 GB
//! checkpoint; everything else — attention, routers, shared experts, the
//! head — is ~25 GB packed. So the non-expert weights stay resident (packed,
//! never widened), and each token reads only the `top_k = 4` experts it fires
//! per layer: ~3 × 4 × 8 MB ≈ 100 MB per layer, ~7 GB per token, of which an
//! LRU of `cache_bytes` absorbs whatever repeats.
//!
//! ```text
//!   token ─► host embed (one Q6_K row) ─► x
//!   for each layer:
//!     A_i: attn_norm → MLA (latent cache) → +x → ffn_norm → router
//!          ─► x_mid, h, ids[4], w[4], new latent row
//!     host: ExpertPager reads the 4 fired experts of 3 banks (packed bytes)
//!           straight into B_i's slot-sized bank params
//!     B_i: shared(h) + Σ w_k · expert_slot_k(h) + x_mid ─► x
//!   head: output_norm → output → softcap ─► logits
//! ```
//!
//! ## Why two graphs per layer
//!
//! You cannot page the experts a token fires until the router has run, and the
//! router's input is that layer's attention output. So each MoE layer is split
//! at the router. The ids come from the **same** [`emit_router`] the resident
//! graph uses, run in-graph, so the slots match the resident routing by
//! construction — a host reimplementation of top-k only agrees while it happens
//! to accumulate the same way (see `rlx-glm5next`'s `paged_moe` for the
//! measured margins).
//!
//! ## Shards
//!
//! The published model is a 6-shard split. Each shard is mmapped as its own
//! loader and keys are routed to whichever shard holds them;
//! `GgufFile::from_split_paths` would instead merge every shard into one owned
//! buffer, i.e. read 233 GB into RAM.
//!
//! ## Scope
//!
//! One token per step: a prompt is fed as consecutive steps, which is exact
//! (decode reproduces prefill; `tests/reference_parity.rs`). Context is capped
//! at `index_topk` (2048) for the same reason as [`crate::decode`].
//!
//! [`emit_router`]: crate::moe::emit_router

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::flow_bridge::PackedWeightLoaderSource;
use rlx_core::flow_util::compile_built;
use rlx_core::weight_loader::{GgufLoader, WeightLoader, ggml_type_to_quant_scheme};
use rlx_distributed::{
    BankIx, ExpertPager, GGUF_MOE_BANKS, PagerStats, register_gguf_expert_banks,
};
use rlx_flow::{CompileProfile, GgufPackedBank, GgufPackedLinear, ModelFlow, WeightSource};
use rlx_gguf::{GgmlType, GgufFile};
use rlx_glm5next::common::rms_norm;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::attention::{DecodeKv, emit_attention_decode};
use crate::config::MistralLarge4Config;
use crate::flow::{EMBED_KEY, attn_dims, emit_ffn_site, emit_head, moe_dims};
use crate::moe::{emit_moe_slots, emit_router};
use crate::rope::{ROPE_FREQS_KEY, emit_pos_tables};

/// The three routed banks, in the order this module uses them.
pub const BANKS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

/// Every shard of a (possibly split) GGUF, each mmapped as its own loader.
pub struct ShardedGguf {
    paths: Vec<PathBuf>,
    loaders: Vec<GgufLoader>,
    /// tensor name → (shard, dtype, GGML shape).
    index: HashMap<String, (usize, GgmlType, Vec<usize>)>,
}

impl ShardedGguf {
    /// Open `path` and, when it is one part of a `-0000N-of-0000M` split, all
    /// of its siblings.
    ///
    /// A directory resolves to its single `.gguf`, or to the `-00001-of-`
    /// shard of the split it holds.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = &resolve_shard(path.as_ref())?;
        let paths = match rlx_core::gguf_support::gguf_split_siblings(path)? {
            Some(parts) => parts,
            None => vec![path.to_path_buf()],
        };
        let mut loaders = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        let paths: Vec<PathBuf> = paths
            .into_iter()
            .map(|p| {
                // Unsloth's first part carries metadata only; its data offset
                // lies past the end of the file, which the mmap reader refuses.
                let tensors = GgufFile::header_from_path(&p)
                    .with_context(|| format!("mistral-large4: reading {}", p.display()))?
                    .tensors
                    .len();
                Ok((tensors > 0).then_some(p))
            })
            .filter_map(Result::transpose)
            .collect::<Result<_>>()?;
        if paths.is_empty() {
            bail!(
                "mistral-large4: no tensors in {} or its split siblings",
                path.display()
            );
        }
        for (si, p) in paths.iter().enumerate() {
            let file = GgufFile::from_path_mmap(p)
                .with_context(|| format!("mistral-large4: mmapping {}", p.display()))?;
            for t in file.tensors.values() {
                index.insert(t.name.clone(), (si, t.dtype, t.shape.clone()));
            }
            loaders.push(GgufLoader::from_gguf_file(file));
        }
        Ok(Self {
            paths,
            loaders,
            index,
        })
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub fn has(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    fn loader(&mut self, key: &str) -> Result<&mut GgufLoader> {
        let &(si, _, _) = self
            .index
            .get(key)
            .ok_or_else(|| anyhow!("mistral-large4: no tensor `{key}` in any shard"))?;
        Ok(&mut self.loaders[si])
    }

    /// `(dtype, GGML shape)` of `key`.
    pub fn info(&self, key: &str) -> Option<(GgmlType, &[usize])> {
        self.index.get(key).map(|(_, d, s)| (*d, s.as_slice()))
    }

    /// Release mmapped pages the builders touched.
    fn release(&self) {
        for l in &self.loaders {
            l.release_mapped_pages();
        }
    }
}

/// `path` itself, or the GGUF a directory holds (its first shard if split).
pub fn resolve_shard(path: &Path) -> Result<PathBuf> {
    if !path.is_dir() {
        return Ok(path.to_path_buf());
    }
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(path)
        .with_context(|| format!("mistral-large4: reading {}", path.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
        .filter(|p| {
            // a vision projector or draft head beside the weights is not a shard
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            !n.starts_with("mmproj") && !n.starts_with("dspark")
        })
        .collect();
    ggufs.sort();
    if let Some(first) = ggufs
        .iter()
        .find(|p| p.to_string_lossy().contains("-00001-of-"))
    {
        return Ok(first.clone());
    }
    match ggufs.as_slice() {
        [one] => Ok(one.clone()),
        [] => bail!("mistral-large4: no .gguf in {}", path.display()),
        _ => bail!(
            "mistral-large4: {} holds several unrelated .gguf files; pass one",
            path.display()
        ),
    }
}

/// A [`WeightSource`] over a [`ShardedGguf`] that declares the routed banks
/// at **slot** width (zeros, refilled per step) instead of reading them.
struct SlotSource<'a> {
    shards: &'a mut ShardedGguf,
    /// Experts per bank in the graph (`rows * top_k`).
    slots: usize,
    /// `rope_freqs.weight`, read once: every layer graph loads it, and the
    /// loaders hand each tensor out once.
    rope_freqs: (Vec<f32>, Vec<usize>),
}

fn is_bank(key: &str) -> bool {
    key.ends_with("_exps.weight")
}

impl WeightSource for SlotSource<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        if key == ROPE_FREQS_KEY {
            return Ok(self.rope_freqs.clone());
        }
        if is_bank(key) {
            bail!("mistral-large4: `{key}` is paged; only its packed form can be declared");
        }
        let l = self.shards.loader(key)?;
        PackedWeightLoaderSource(l).take(key, transpose)
    }

    fn has(&self, key: &str) -> bool {
        key == ROPE_FREQS_KEY || self.shards.has(key)
    }

    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        if key == ROPE_FREQS_KEY || is_bank(key) || !self.shards.has(key) {
            return Ok(None);
        }
        let l = self.shards.loader(key)?;
        PackedWeightLoaderSource(l).take_packed(key)
    }

    fn has_packed(&self, key: &str) -> bool {
        self.shards.has(key) && key != ROPE_FREQS_KEY
    }

    fn take_packed_bank(&mut self, key: &str) -> Result<Option<GgufPackedBank>> {
        if !is_bank(key) {
            return Ok(None);
        }
        let (dtype, shape) = self
            .shards
            .info(key)
            .ok_or_else(|| anyhow!("mistral-large4: no bank `{key}`"))?;
        let scheme = ggml_type_to_quant_scheme(dtype).ok_or_else(|| {
            anyhow!("mistral-large4: `{key}` is {dtype:?}; paging needs a quantized bank")
        })?;
        let [in_dim, out_dim, _experts] = shape else {
            bail!("mistral-large4: `{key}` is not a 3-D bank ({shape:?})");
        };
        let per = rlx_gguf::bytes_for_public(dtype, in_dim * out_dim)
            .ok_or_else(|| anyhow!("mistral-large4: `{key}`: unaligned {dtype:?} expert"))?;
        Ok(Some(GgufPackedBank {
            w_q: vec![0u8; self.slots * per],
            scheme,
            num_experts: self.slots,
            in_dim: *in_dim,
            out_dim: *out_dim,
        }))
    }
}

/// Host-side embedding: one dequantized row per token, read from the mmap.
struct HostEmbed {
    file: GgufFile,
    name: String,
    dtype: GgmlType,
    hidden: usize,
    row_bytes: usize,
}

impl HostEmbed {
    fn open(shards: &ShardedGguf, hidden: usize) -> Result<Self> {
        let &(si, dtype, ref shape) = shards
            .index
            .get(EMBED_KEY)
            .ok_or_else(|| anyhow!("mistral-large4: no `{EMBED_KEY}`"))?;
        if shape.first() != Some(&hidden) {
            bail!("mistral-large4: `{EMBED_KEY}` is {shape:?}, expected rows of {hidden}");
        }
        let row_bytes = rlx_gguf::bytes_for_public(dtype, hidden)
            .ok_or_else(|| anyhow!("mistral-large4: embedding rows are not block-aligned"))?;
        Ok(Self {
            file: GgufFile::from_path_mmap(&shards.paths[si])?,
            name: EMBED_KEY.to_string(),
            dtype,
            hidden,
            row_bytes,
        })
    }

    fn row(&self, id: usize) -> Result<Vec<f32>> {
        let t = self
            .file
            .tensors
            .get(&self.name)
            .ok_or_else(|| anyhow!("embedding vanished"))?;
        let all = self.file.tensor_bytes(t)?;
        let lo = id * self.row_bytes;
        let bytes = all
            .get(lo..lo + self.row_bytes)
            .ok_or_else(|| anyhow!("mistral-large4: token id {id} out of range"))?;
        rlx_gguf::dequant_typed(self.dtype, bytes, self.hidden, "token_embd")
    }
}

/// Graph inputs.
const X: &str = "x";
const H: &str = "h";
const W: &str = "w";
const POS: &str = "pos";
const MASK: &str = "mask";
const KV: &str = "kv";

/// One layer's two graphs and where its outputs sit in their packed rows.
struct Layer {
    index: usize,
    attn: CompiledGraph,
    /// `None` on a dense-FFN layer (graph A then finishes the layer).
    experts: Option<Experts>,
    /// Latent row width when the layer owns its KV, 0 otherwise.
    row_width: usize,
}

struct Experts {
    graph: CompiledGraph,
    /// Param names of the three slot banks, in [`BANKS`] order.
    params: [String; 3],
    banks: [BankIx; 3],
    bytes_per_expert: [usize; 3],
}

/// Where a step's time went, summed over layers.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepTimes {
    pub attention: Duration,
    pub page: Duration,
    pub experts: Duration,
    pub head: Duration,
}

/// How to open a [`PagedStack`] / [`PagedMistralLarge4`].
#[derive(Debug, Clone)]
pub struct PagedOptions {
    pub device: Device,
    /// Context capacity in tokens (≤ `index_topk − 1`).
    pub cap: usize,
    /// Pager LRU budget, bytes of packed experts kept in RAM.
    pub cache_bytes: u64,
}

impl Default for PagedOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            cap: 512,
            cache_bytes: 8 << 30,
        }
    }
}

/// A run of layers with paged experts, driven one token at a time over the
/// hidden state. [`PagedMistralLarge4`] wraps it with the embedding and head;
/// on its own it runs the blocks of a range-fetched subset
/// (`scripts/mistral_large4_subset.py`).
pub struct PagedStack {
    cfg: MistralLarge4Config,
    cap: usize,
    pos: usize,
    layers: Vec<Layer>,
    /// Latent cache per owning layer, `[cap * width]`.
    caches: HashMap<usize, Vec<f32>>,
    mask: Vec<f32>,
    pager: Arc<ExpertPager>,
    times: StepTimes,
    whole_uploads: usize,
}

impl PagedStack {
    /// Build graphs for `layers` (increasing; every KV source must be in it).
    pub fn new(
        cfg: &MistralLarge4Config,
        shards: &mut ShardedGguf,
        layers: &[usize],
        opts: &PagedOptions,
    ) -> Result<Self> {
        cfg.validate()?;
        if opts.cap == 0 || !cfg.dsa_is_dense(opts.cap + 1) {
            bail!(
                "mistral-large4: paged context {} must be in 1..{} (index_topk − 1): past \
                 the DSA budget the selection needs an indexer cache that is not emitted",
                opts.cap,
                cfg.index_topk
            );
        }
        if layers.windows(2).any(|w| w[0] >= w[1]) {
            bail!("mistral-large4: layers {layers:?} must be increasing");
        }
        for &i in layers {
            if !cfg.owns_kv(i) && !layers.contains(&cfg.kv_source[i]) {
                bail!(
                    "mistral-large4: layer {i} shares KV with {}, which is not in {layers:?}",
                    cfg.kv_source[i]
                );
            }
        }

        let mut pager = ExpertPager::new(opts.cache_bytes);
        for p in shards.paths().to_vec() {
            register_gguf_expert_banks(&mut pager, &p, &GGUF_MOE_BANKS)?;
        }
        let rope_freqs = {
            let l = shards.loader(ROPE_FREQS_KEY)?;
            l.take(ROPE_FREQS_KEY)?
        };

        let mut built = Vec::with_capacity(layers.len());
        for &i in layers {
            let mut src = SlotSource {
                shards,
                slots: cfg.num_experts_per_tok,
                rope_freqs: rope_freqs.clone(),
            };
            built.push(build_layer(cfg, i, opts, &mut src, &pager)?);
            shards.release();
        }
        let caches = layers
            .iter()
            .filter(|&&i| cfg.owns_kv(i))
            .map(|&i| (i, vec![0f32; opts.cap * cfg.latent_width(i)]))
            .collect();
        let mut mask = vec![0f32; opts.cap + 1];
        mask[opts.cap] = 1.0;
        Ok(Self {
            cfg: cfg.clone(),
            cap: opts.cap,
            pos: 0,
            layers: built,
            caches,
            mask,
            pager: Arc::new(pager),
            times: StepTimes::default(),
            whole_uploads: 0,
        })
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    /// Forget the context (the expert cache is kept: it is model state).
    pub fn reset(&mut self) {
        self.pos = 0;
        for c in self.caches.values_mut() {
            c.fill(0.0);
        }
        self.mask.fill(0.0);
        self.mask[self.cap] = 1.0;
    }

    pub fn pager_stats(&self) -> PagerStats {
        self.pager.stats()
    }

    pub fn times(&self) -> StepTimes {
        self.times
    }

    /// Zero [`Self::times`] — e.g. after the first step, whose runs include
    /// one-off kernel/pipeline compilation.
    pub fn reset_times(&mut self) {
        self.times = StepTimes::default();
    }

    /// Bank uploads that fell back to a whole-buffer write (0 = every expert
    /// went straight into its slot).
    pub fn whole_bank_uploads(&self) -> usize {
        self.whole_uploads
    }

    /// Run every layer for the token at the current position; `x` is its
    /// `[hidden]` input, the residual stream after the last layer comes back.
    pub fn step(&mut self, mut x: Vec<f32>) -> Result<Vec<f32>> {
        let d = self.cfg.hidden_size;
        let k = self.cfg.num_experts_per_tok;
        if x.len() != d {
            bail!(
                "mistral-large4: step input is {} wide, expected {d}",
                x.len()
            );
        }
        if self.pos >= self.cap {
            bail!("mistral-large4: context full ({} tokens)", self.cap);
        }
        let pos = [self.pos as f32];
        let mut rows: HashMap<usize, Vec<f32>> = HashMap::new();

        for li in 0..self.layers.len() {
            let i = self.layers[li].index;
            // KV input: an owning layer reads its cache; a sharing layer reads
            // its source's cache with this step's new source row appended.
            let shared_all;
            let kv: &[f32] = if self.cfg.owns_kv(i) {
                &self.caches[&i]
            } else {
                let src = self.cfg.kv_source[i];
                let mut all = self.caches[&src].clone();
                all.extend_from_slice(&rows[&src]);
                shared_all = all;
                &shared_all
            };
            let t = Instant::now();
            let out = self.layers[li]
                .attn
                .run(&[
                    (X, &x[..]),
                    (POS, &pos[..]),
                    (MASK, &self.mask[..]),
                    (KV, kv),
                ])
                .remove(0);
            self.times.attention += t.elapsed();

            let rw = self.layers[li].row_width;
            let Some(ex) = self.layers[li].experts.as_mut() else {
                // Dense layer: [x_out | row]
                if rw > 0 {
                    rows.insert(i, out[d..d + rw].to_vec());
                }
                x = out[..d].to_vec();
                continue;
            };
            // MoE layer: [x_mid | h | ids | w | row]
            let (x_mid, h) = (&out[..d], &out[d..2 * d]);
            let ids = &out[2 * d..2 * d + k];
            let w = &out[2 * d + k..2 * d + 2 * k];
            if rw > 0 {
                rows.insert(i, out[2 * d + 2 * k..2 * d + 2 * k + rw].to_vec());
            }
            let ids: Vec<usize> = ids
                .iter()
                .map(|&v| {
                    let id = v.round();
                    if !v.is_finite() || id < 0.0 || id as usize >= self.cfg.n_routed_experts {
                        bail!("mistral-large4: layer {i} router returned expert id {v}");
                    }
                    Ok(id as usize)
                })
                .collect::<Result<_>>()?;

            let t = Instant::now();
            for b in 0..3 {
                let per = ex.bytes_per_expert[b];
                let mut partial = true;
                for (slot, &id) in ids.iter().enumerate() {
                    let bytes = self.pager.expert_at(ex.banks[b], id)?;
                    if !ex.graph.set_param_range(&ex.params[b], slot * per, &bytes) {
                        partial = false;
                        break;
                    }
                }
                if !partial {
                    self.whole_uploads += 1;
                    let bytes = self.pager.gather_at(ex.banks[b], &ids)?;
                    ex.graph.set_param_typed(&ex.params[b], &bytes, DType::U8);
                }
            }
            self.times.page += t.elapsed();

            let t = Instant::now();
            x = ex.graph.run(&[(X, x_mid), (H, h), (W, w)]).remove(0);
            self.times.experts += t.elapsed();
        }

        for (i, row) in rows {
            let wdt = row.len();
            let c = self.caches.get_mut(&i).expect("owning layer has a cache");
            c[self.pos * wdt..(self.pos + 1) * wdt].copy_from_slice(&row);
        }
        self.mask[self.pos] = 1.0;
        self.pos += 1;
        Ok(x)
    }
}

/// Build graph A (and B on a MoE layer) for layer `i`.
fn build_layer(
    cfg: &MistralLarge4Config,
    i: usize,
    opts: &PagedOptions,
    src: &mut SlotSource<'_>,
    pager: &ExpertPager,
) -> Result<Layer> {
    let f = DType::F32;
    let (d, k, cap) = (cfg.hidden_size, cfg.num_experts_per_tok, opts.cap);
    let owns = cfg.owns_kv(i);
    let width = cfg.latent_width(i);
    let row_width = if owns { width } else { 0 };
    let moe = cfg.moe_layers[i];
    let out_w = if moe { 2 * d + 2 * k } else { d } + row_width;
    let prefix = format!("blk.{i}");
    let kv_shape = if owns {
        Shape::new(&[1, cap, width], f)
    } else {
        Shape::new(&[cap + 1, width], f)
    };

    let c = cfg.clone();
    let p = prefix.clone();
    let a = ModelFlow::new(format!("ml4_paged_attn{i}"))
        .with_profile(CompileProfile::llama32_prefill())
        .input(X, Shape::new(&[1, d], f))
        .input(POS, Shape::new(&[1, 1], f))
        .input(MASK, Shape::new(&[1, cap + 1], f))
        .input(KV, kv_shape)
        .plugin_named("attn", move |emit, _prev| {
            let x = emit.flow_input(X)?.hir_id();
            let pos = emit.flow_input(POS)?.hir_id();
            let mask = emit.flow_input(MASK)?.hir_id();
            let kv = emit.flow_input(KV)?.hir_id();
            let t = emit_pos_tables(emit, &c, pos, 1)?;
            let normed = rms_norm(emit, &format!("{p}.attn_norm"), x, d, c.rms_norm_eps)?;
            let cache = if owns {
                Ok((kv, mask, cap))
            } else {
                Err(DecodeKv {
                    all: kv,
                    mask,
                    len: cap + 1,
                })
            };
            let (attn, _, row) = emit_attention_decode(
                emit,
                &p,
                normed,
                attn_dims(&c, 1),
                c.rope_dim(i),
                cache,
                &t,
            )?;
            let x_mid = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(x, attn)
            };
            let mut parts = Vec::new();
            if moe {
                let h = rms_norm(emit, &format!("{p}.ffn_norm"), x_mid, d, c.rms_norm_eps)?;
                let (idx, w) = emit_router(emit, &p, h, moe_dims(&c, 1))?;
                parts.extend([x_mid, h, idx, w]);
            } else {
                parts.push(emit_ffn_site(emit, &c, i, x_mid, 1)?);
            }
            parts.extend(row);
            let mut gb = HirMut::new(emit.hir());
            let out = gb.concat_(parts, 1);
            Ok(Some(emit.wrap(out, Shape::new(&[1, out_w], f))))
        })
        .output("packed")
        .build_with(src, None)?;
    let attn = compile_built(a, opts.device)?;

    let experts = if moe {
        let c = cfg.clone();
        let p = prefix.clone();
        let b = ModelFlow::new(format!("ml4_paged_experts{i}"))
            .with_profile(CompileProfile::llama32_prefill())
            .input(X, Shape::new(&[1, d], f))
            .input(H, Shape::new(&[1, d], f))
            .input(W, Shape::new(&[1, k], f))
            .plugin_named("experts", move |emit, _prev| {
                let x_mid = emit.flow_input(X)?.hir_id();
                let h = emit.flow_input(H)?.hir_id();
                let w = emit.flow_input(W)?.hir_id();
                let y = emit_moe_slots(emit, &p, h, w, moe_dims(&c, 1))?;
                let mut gb = HirMut::new(emit.hir());
                let out = gb.add(x_mid, y);
                Ok(Some(emit.wrap(out, Shape::new(&[1, d], f))))
            })
            .output("x_out")
            .build_with(src, None)?;
        let graph = compile_built(b, opts.device)?;
        let mut banks = [BankIx::default(); 3];
        let mut bytes_per_expert = [0usize; 3];
        for (bi, bank) in BANKS.iter().enumerate() {
            let ix = pager.bank_ix(i, bank).ok_or_else(|| {
                anyhow!("mistral-large4: layer {i} bank `{bank}` not in the pager")
            })?;
            let key = format!("{prefix}.{bank}.weight");
            let (dtype, shape) = src.shards.info(&key).ok_or_else(|| anyhow!("no {key}"))?;
            let want = rlx_gguf::bytes_for_public(dtype, shape[0] * shape[1]).unwrap_or(0);
            let got = pager.location(ix).bytes_per_expert;
            if got != want {
                bail!(
                    "mistral-large4: {key}: pager slices {got}-byte experts, graph slots are {want}"
                );
            }
            banks[bi] = ix;
            bytes_per_expert[bi] = got;
        }
        Some(Experts {
            graph,
            // `rlx_glm5next::common::packed_bank` names the U8 blob `{key}\0q`.
            params: BANKS.map(|b| format!("{prefix}.{b}.weight\0q")),
            banks,
            bytes_per_expert,
        })
    } else {
        None
    };
    Ok(Layer {
        index: i,
        attn,
        experts,
        row_width,
    })
}

/// The full model: host embedding, a [`PagedStack`] over all 72 layers, and
/// the head.
pub struct PagedMistralLarge4 {
    pub cfg: MistralLarge4Config,
    embed: HostEmbed,
    stack: PagedStack,
    head: CompiledGraph,
}

impl PagedMistralLarge4 {
    /// Open the checkpoint at `path` (any shard of a split, or a single file).
    ///
    /// Builds and compiles 72 attention graphs, 71 expert graphs and the head;
    /// the non-expert weights land in them packed (~25 GB at `UD-IQ1_S`).
    pub fn open(path: impl AsRef<Path>, opts: &PagedOptions) -> Result<Self> {
        let mut shards = ShardedGguf::open(path)?;
        let cfg = MistralLarge4Config::from_gguf_path(&shards.paths()[0])?;
        let embed = HostEmbed::open(&shards, cfg.hidden_size)?;
        let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
        let stack = PagedStack::new(&cfg, &mut shards, &layers, opts)?;

        let c = cfg.clone();
        let (d, v) = (cfg.hidden_size, cfg.vocab_size);
        let mut src = SlotSource {
            shards: &mut shards,
            slots: 0,
            rope_freqs: (Vec::new(), Vec::new()),
        };
        let head = ModelFlow::new("ml4_paged_head")
            .with_profile(CompileProfile::llama32_prefill())
            .input(X, Shape::new(&[1, d], DType::F32))
            .plugin_named("head", move |emit, _prev| {
                let x = emit.flow_input(X)?.hir_id();
                let y = emit_head(emit, &c, x, 1, true)?;
                Ok(Some(emit.wrap(y, Shape::new(&[1, v], DType::F32))))
            })
            .output("logits")
            .build_with(&mut src, None)?;
        let head = compile_built(head, opts.device)?;
        shards.release();
        Ok(Self {
            cfg,
            embed,
            stack,
            head,
        })
    }

    pub fn stack(&self) -> &PagedStack {
        &self.stack
    }

    pub fn reset(&mut self) {
        self.stack.reset();
    }

    /// Feed `token` at the next position; returns its logits.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        let x = self.embed.row(token as usize)?;
        let x = self.stack.step(x)?;
        let t = Instant::now();
        let logits = self.head.run(&[(X, &x[..])]).remove(0);
        self.stack.times.head += t.elapsed();
        Ok(logits)
    }

    /// Greedy continuation of `prompt` by `max_new` tokens, stopping at `eos`.
    pub fn generate_greedy(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            bail!("mistral-large4: empty prompt");
        }
        let mut logits = Vec::new();
        for &t in prompt {
            logits = self.step(t)?;
        }
        let mut out = Vec::new();
        for _ in 0..max_new {
            let next = argmax(&logits) as u32;
            out.push(next);
            on_token(next);
            if Some(next) == eos || self.stack.position() >= self.stack.cap {
                break;
            }
            logits = self.step(next)?;
        }
        Ok(out)
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}
