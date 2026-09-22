// RLX — versatile ML compiler + runtime. GPLv3.
//! Demand-paging for the **DeepSeek-V4.1** routed experts.
//!
//! The experts are the checkpoint: 40 layers × 384 experts × 3 projections, and
//! a token touches `num_experts_per_tok` of them per layer. Materializing a
//! layer's whole bank costs ~34 GB as f32, so the bank cannot be a graph
//! parameter and the weights cannot all be resident — but the *working set* of
//! one token is `top_k` experts, which is small.
//!
//! This pages at exactly that granularity. Each expert projection is its own
//! checkpoint tensor (unlike the GGUF single-bank layout
//! [`rlx_distributed::ExpertPager`] targets), so the unit of paging is one
//! `layers.{L}.ffn.experts.{E}.{proj}.weight` — read with `pread`, dequantized,
//! transposed into the `[in, out]` orientation [`rlx_ir::op::Op::GroupedMatMul`]
//! wants, and held in an LRU under a byte budget.
//!
//! Locking note: the `pread` and the dequantize happen **outside** the lock. A
//! pager that holds its mutex across the read serializes every miss behind the
//! disk and loses most of the benefit of having a cache at all.

use crate::dsv41::DeepseekV41Spec;
use crate::dsv41_weights::{RawTensor, StreamingLoader, WeightIndex};
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Which of the three expert projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Proj {
    /// SwiGLU gate.
    W1,
    /// Down-projection.
    W2,
    /// SwiGLU up.
    W3,
}

impl Proj {
    pub fn key(self) -> &'static str {
        match self {
            Proj::W1 => "w1",
            Proj::W2 => "w2",
            Proj::W3 => "w3",
        }
    }

    pub const ALL: [Proj; 3] = [Proj::W1, Proj::W2, Proj::W3];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    layer: usize,
    proj: Proj,
    expert: usize,
}

/// Cache counters. `misses` is the number of disk reads actually performed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PagerStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_read: u64,
    pub resident_bytes: u64,
}

impl PagerStats {
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 {
            0.0
        } else {
            self.hits as f64 / n as f64
        }
    }
}

struct Slot {
    /// Held **as stored**. A released-checkpoint expert is 18 MB packed and
    /// 142 MB as f32, so caching the dequantized form would shrink the working
    /// set this budget can hold by 8× — the difference between keeping a decode
    /// step's experts resident and thrashing.
    data: Arc<RawTensor>,
    /// Recency tick; the lowest is evicted first.
    used: u64,
}

/// An LRU over dequantized expert projections, bounded by bytes.
pub struct ExpertPager {
    loader: Mutex<StreamingLoader>,
    budget_bytes: u64,
    resident: Mutex<HashMap<Key, Slot>>,
    tick: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    bytes_read: AtomicU64,
    resident_bytes: AtomicU64,
}

impl ExpertPager {
    /// `budget_bytes` caps the dequantized bytes held resident.
    ///
    /// It is a *soft* floor of one expert: a budget smaller than a single
    /// projection would evict each expert before it could be used, so the pager
    /// always keeps the one it just read.
    pub fn new(index: WeightIndex, block: usize, budget_bytes: u64) -> Self {
        ExpertPager {
            loader: Mutex::new(StreamingLoader::from_index(index, block)),
            budget_bytes,
            resident: Mutex::new(HashMap::new()),
            tick: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            resident_bytes: AtomicU64::new(0),
        }
    }

    pub fn open(dir: &std::path::Path, block: usize, budget_bytes: u64) -> Result<Self> {
        Ok(Self::new(WeightIndex::open(dir)?, block, budget_bytes))
    }

    fn name(spec: &DeepseekV41Spec, layer: usize, proj: Proj, expert: usize) -> String {
        format!(
            "{}.ffn.experts.{expert}.{}.weight",
            spec.layer_prefix(layer),
            proj.key()
        )
    }

    /// One expert projection as `[in, out]` f32.
    ///
    /// Convenience over [`Self::raw`] for callers that want the floats; the
    /// cache itself holds the packed form, so this dequantizes on every call.
    /// [`Self::gather_bank`] is the path that matters for a run.
    pub fn expert(
        &self,
        spec: &DeepseekV41Spec,
        layer: usize,
        proj: Proj,
        expert: usize,
    ) -> Result<Vec<f32>> {
        self.raw(spec, layer, proj, expert)?.to_f32_transposed()
    }

    /// One expert projection as the checkpoint stores it.
    ///
    /// Returns an `Arc` so two rows routing to the same expert share one copy.
    pub fn raw(
        &self,
        spec: &DeepseekV41Spec,
        layer: usize,
        proj: Proj,
        expert: usize,
    ) -> Result<Arc<RawTensor>> {
        let key = Key {
            layer,
            proj,
            expert,
        };
        let now = self.tick.fetch_add(1, Ordering::Relaxed);
        // Fast path: hit. Take the lock only long enough to clone the Arc.
        if let Some(slot) = self.resident.lock().unwrap().get_mut(&key) {
            slot.used = now;
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::clone(&slot.data));
        }
        self.misses.fetch_add(1, Ordering::Relaxed);

        // Slow path: read and dequantize with the cache lock released, so other
        // threads keep hitting while this one waits on the disk.
        let name = Self::name(spec, layer, proj, expert);
        let raw = {
            let mut l = self.loader.lock().unwrap();
            l.fetch_raw(&name)?
        };
        if raw.shape().len() != 2 {
            bail!(
                "deepseek_v41: expert `{name}` is rank {}, expected 2",
                raw.shape().len()
            );
        }
        let bytes = raw.stored_bytes() as u64;
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        let arc = Arc::new(raw);

        let mut res = self.resident.lock().unwrap();
        // A concurrent miss on the same key may have inserted already; keep the
        // existing Arc so every holder sees one allocation.
        if let Some(slot) = res.get_mut(&key) {
            slot.used = now;
            return Ok(Arc::clone(&slot.data));
        }
        res.insert(
            key,
            Slot {
                data: Arc::clone(&arc),
                used: now,
            },
        );
        self.resident_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.evict_to_budget(&mut res, key);
        Ok(arc)
    }

    /// Evict least-recently-used slots until the budget is met, never evicting
    /// `keep` — the entry the caller is about to use.
    fn evict_to_budget(&self, res: &mut HashMap<Key, Slot>, keep: Key) {
        while self.resident_bytes.load(Ordering::Relaxed) > self.budget_bytes && res.len() > 1 {
            let victim = res
                .iter()
                .filter(|(k, _)| **k != keep)
                .min_by_key(|(_, s)| s.used)
                .map(|(k, _)| *k);
            let Some(v) = victim else { break };
            if let Some(s) = res.remove(&v) {
                self.resident_bytes
                    .fetch_sub(s.data.stored_bytes() as u64, Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Stack the given experts into the `[n_sel, in, out]` bank
    /// [`rlx_ir::op::Op::GroupedMatMul`] indexes, in the order supplied.
    ///
    /// The caller's expert ids become *slot* indices 0..n_sel, which is what lets
    /// the graph carry a bank of only the routed experts.
    pub fn gather_bank(
        &self,
        spec: &DeepseekV41Spec,
        layer: usize,
        proj: Proj,
        ids: &[usize],
    ) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        for &e in ids {
            // dequantized here and nowhere else: the transient f32 is only the
            // gathered bank, not the resident cache
            out.extend_from_slice(&self.raw(spec, layer, proj, e)?.to_f32_transposed()?);
        }
        Ok(out)
    }

    /// Warm the cache for experts that are about to be needed.
    pub fn prefetch(&self, spec: &DeepseekV41Spec, layer: usize, ids: &[usize]) {
        for &e in ids {
            for p in Proj::ALL {
                let _ = self.raw(spec, layer, p, e);
            }
        }
    }

    pub fn stats(&self) -> PagerStats {
        PagerStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            resident_bytes: self.resident_bytes.load(Ordering::Relaxed),
        }
    }

    /// Drop every resident expert.
    pub fn clear(&self) {
        self.resident.lock().unwrap().clear();
        self.resident_bytes.store(0, Ordering::Relaxed);
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_loader::SyntheticLoader;
    use safetensors::serialize_to_file;
    use safetensors::tensor::{Dtype as StDtype, TensorView};
    use std::path::PathBuf;

    const N_EXPERTS: usize = 6;
    const OUT: usize = 4;
    const IN: usize = 3;

    fn spec() -> DeepseekV41Spec {
        DeepseekV41Spec::from_config(&serde_json::json!({
            "vocab_size": 16, "hidden_size": IN, "num_hidden_layers": 2,
            "num_attention_heads": 1, "head_dim": 4, "o_lora_rank": 2,
            "n_routed_experts": N_EXPERTS, "moe_intermediate_size": OUT,
        }))
        .unwrap()
    }

    fn name(layer: usize, proj: &str, e: usize) -> String {
        format!("layers.{layer}.ffn.experts.{e}.{proj}.weight")
    }

    /// A checkpoint holding just the expert tensors, stored `[out, in]` as the
    /// real one does.
    fn checkpoint(tag: &str) -> PathBuf {
        let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
        let dir = PathBuf::from(base).join(format!("rlx_dsv41_pager_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut data = Vec::new();
        for layer in 0..2 {
            for proj in ["w1", "w2", "w3"] {
                for e in 0..N_EXPERTS {
                    let n = name(layer, proj, e);
                    let v = SyntheticLoader::values(&n, &[OUT, IN]);
                    data.push((n, bytemuck::cast_slice(&v).to_vec()));
                }
            }
        }
        let views: HashMap<String, TensorView> = data
            .iter()
            .map(|(n, b)| {
                (
                    n.clone(),
                    TensorView::new(StDtype::F32, vec![OUT, IN], b).unwrap(),
                )
            })
            .collect();
        serialize_to_file(&views, None, &dir.join("model.safetensors")).unwrap();
        dir
    }

    fn pager(tag: &str, budget: u64) -> (ExpertPager, DeepseekV41Spec) {
        let dir = checkpoint(tag);
        (ExpertPager::open(&dir, 32, budget).unwrap(), spec())
    }

    /// The pager must hand back the expert **transposed** into `[in, out]`.
    ///
    /// The checkpoint stores `[out, in]` and `Op::GroupedMatMul` indexes
    /// `[E, in, out]`; getting this backwards is not a crash, it is a silently
    /// wrong matmul, so it is worth pinning against the stored bytes directly.
    #[test]
    fn an_expert_comes_back_transposed() {
        let (p, sp) = pager("transpose", 1 << 30);
        let got = p.expert(&sp, 1, Proj::W2, 3).unwrap();
        let stored = SyntheticLoader::values(&name(1, "w2", 3), &[OUT, IN]);
        assert_eq!(got.len(), OUT * IN);
        for i in 0..OUT {
            for j in 0..IN {
                assert_eq!(
                    got[j * OUT + i],
                    stored[i * IN + j],
                    "element ({i}, {j}) is not transposed"
                );
            }
        }
    }

    /// A second read of the same expert must be a cache hit, and must return the
    /// *same allocation* — two rows routing to one expert should not each pay
    /// for a copy of the stored bytes.
    #[test]
    fn a_repeat_read_hits_the_cache_without_copying() {
        let (p, sp) = pager("hit", 1 << 30);
        let a = p.raw(&sp, 0, Proj::W1, 2).unwrap();
        let b = p.raw(&sp, 0, Proj::W1, 2).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "the second read allocated again");
        let s = p.stats();
        assert_eq!((s.hits, s.misses), (1, 1));
        assert_eq!(s.hit_rate(), 0.5);
    }

    /// Under a budget of one expert the pager must evict, keep working, and
    /// report the eviction — the whole point being that it stays inside the
    /// budget rather than growing to the bank.
    #[test]
    fn a_tight_budget_evicts_and_still_returns_correct_weights() {
        let one = (OUT * IN * 4) as u64;
        let (p, sp) = pager("evict", one);
        for e in 0..N_EXPERTS {
            let got = p.expert(&sp, 0, Proj::W1, e).unwrap();
            let stored = SyntheticLoader::values(&name(0, "w1", e), &[OUT, IN]);
            assert_eq!(
                got[0], stored[0],
                "expert {e} came back wrong after eviction"
            );
        }
        let s = p.stats();
        assert_eq!(s.misses, N_EXPERTS as u64, "every read should miss");
        assert!(
            s.evictions > 0,
            "nothing was evicted under a 1-expert budget"
        );
        assert!(
            s.resident_bytes <= one,
            "resident {} exceeds the budget {one}",
            s.resident_bytes
        );
    }

    /// Eviction must never drop the entry the caller is about to use, even when
    /// the budget is smaller than a single expert.
    #[test]
    fn the_entry_just_read_survives_an_impossible_budget() {
        let (p, sp) = pager("keep", 1);
        let got = p.expert(&sp, 0, Proj::W3, 4).unwrap();
        let stored = SyntheticLoader::values(&name(0, "w3", 4), &[OUT, IN]);
        assert_eq!(got[0], stored[0]);
        assert_eq!(p.stats().resident_bytes, (OUT * IN * 4) as u64);
    }

    /// The gathered bank must be the experts concatenated **in the order asked
    /// for**, because the caller's slot indices are positions in that order.
    #[test]
    fn the_gathered_bank_follows_the_requested_order() {
        let (p, sp) = pager("gather", 1 << 30);
        let ids = [5usize, 0, 3];
        let bank = p.gather_bank(&sp, 1, Proj::W1, &ids).unwrap();
        assert_eq!(bank.len(), ids.len() * OUT * IN);
        for (slot, &e) in ids.iter().enumerate() {
            let want = p.expert(&sp, 1, Proj::W1, e).unwrap();
            let got = &bank[slot * OUT * IN..(slot + 1) * OUT * IN];
            assert_eq!(got, want.as_slice(), "slot {slot} does not hold expert {e}");
        }
    }

    /// Layer and projection are part of the key: reading the same expert id from
    /// a different layer must not return the first one's weights.
    #[test]
    fn layer_and_projection_are_part_of_the_cache_key() {
        let (p, sp) = pager("key", 1 << 30);
        let l0 = p.expert(&sp, 0, Proj::W1, 2).unwrap();
        let l1 = p.expert(&sp, 1, Proj::W1, 2).unwrap();
        let w3 = p.expert(&sp, 0, Proj::W3, 2).unwrap();
        assert_ne!(l0.as_slice(), l1.as_slice(), "layers collided");
        assert_ne!(l0.as_slice(), w3.as_slice(), "projections collided");
        assert_eq!(p.stats().misses, 3, "each should be its own entry");
    }

    /// Prefetch warms exactly the experts named, so the reads that follow hit.
    #[test]
    fn prefetch_warms_the_working_set() {
        let (p, sp) = pager("prefetch", 1 << 30);
        p.prefetch(&sp, 0, &[1, 4]);
        let before = p.stats();
        assert_eq!(before.misses, 6, "two experts x three projections");
        for e in [1usize, 4] {
            for proj in Proj::ALL {
                let _ = p.expert(&sp, 0, proj, e).unwrap();
            }
        }
        let after = p.stats();
        assert_eq!(
            after.misses, before.misses,
            "prefetched reads should not miss"
        );
        assert_eq!(after.hits, 6);
    }

    /// The cache must hold experts **as stored**, not dequantized.
    ///
    /// This is the entire reason the pager defers the dequant: a released
    /// expert is 18 MB packed and 142 MB as f32, so a budget that holds a
    /// decode step's working set packed would hold an eighth of it otherwise.
    /// The assertion is on `resident_bytes` rather than on a timing, because
    /// that is the number the budget is enforced against.
    #[test]
    fn the_cache_holds_packed_bytes_not_floats() {
        use crate::dsv41_weights::{CheckpointFormat, write_synthetic_checkpoint_as};

        let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
        let dir = PathBuf::from(base).join("rlx_dsv41_pager_packed");
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = serde_json::json!({
            "model_type": "deepseek_v41",
            "vocab_size": 32, "hidden_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 1, "head_dim": 32, "o_lora_rank": 4,
            "q_lora_rank": 8, "hc_mult": 2, "qk_rope_head_dim": 16,
            "compress_ratios": [0, 0],
            "moe_intermediate_size": 32, "n_routed_experts": 4,
            "num_experts_per_tok": 2, "n_shared_experts": 1,
            "quantization_config": { "weight_block_size": [8, 8], "expert_dtype": "fp4" },
        });
        write_synthetic_checkpoint_as(&dir, &cfg, CheckpointFormat::Quantized).unwrap();
        let sp = DeepseekV41Spec::from_config(&cfg).unwrap();
        let p = ExpertPager::open(&dir, 8, 1 << 30).unwrap();

        let raw = p.raw(&sp, 0, Proj::W1, 0).unwrap();
        let floats = raw.to_f32_transposed().unwrap();
        let packed = raw.stored_bytes();
        let dense = floats.len() * 4;
        // FP4 is a nibble per weight against four bytes, so the ceiling is 8x;
        // the scales eat into it, and at this toy block size of 8 they eat more
        // than at the released 32.
        assert!(
            packed * 4 <= dense,
            "packed {packed} B vs {dense} B as f32 — less than a 4x saving"
        );
        assert_eq!(
            p.stats().resident_bytes as usize,
            packed,
            "the cache is charging the dequantized size, not the stored one"
        );
    }
}
