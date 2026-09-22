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

//! Reading an upscaler checkpoint.
//!
//! Community models ship as bare PyTorch state dicts — `.pth` far more often
//! than `.safetensors` — usually wrapped in one of a handful of trainer-
//! specific keys (`params`, `params_ema`, `state_dict`, …). [`Checkpoint::open`]
//! handles both containers and unwraps the wrapper.
//!
//! # Structural reparameterization happens here, not in the graph
//!
//! SPAN's `Conv3XC` trains as four parallel/serial branches and collapses to a
//! single 3×3 for inference. Checkpoints carry the training-time weights (and,
//! often, a **stale** `eval_conv` from whenever the exporter last ran), so
//! [`fuse_conv3xc`] recomputes the fused kernel from the branches exactly as
//! the reference `update_params()` does. Trusting a checkpoint's stored
//! `eval_conv` is the tempting shortcut and it is wrong: the reference calls
//! `update_params()` on *every* forward, so the branches are the source of
//! truth and `eval_conv` is a cache.

use anyhow::{Context, Result, bail, ensure};
use rlx_core::weight_map::WeightMap;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Trainer wrappers seen in the wild, in the order they are tried.
const WRAPPERS: &[&str] = &[
    "params_ema",
    "params",
    "state_dict",
    "model_state_dict",
    "model",
    "generator",
    "net_g",
    "netG",
];

/// A loaded state dict: tensor name → (row-major f32 data, shape).
#[derive(Clone)]
pub struct Checkpoint {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Checkpoint {
    /// Open a `.pth` / `.pt` / `.bin` (torch.save) or `.safetensors` file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();

        let tensors = match ext.as_str() {
            "safetensors" => {
                let mut wm = WeightMap::from_file(
                    path.to_str()
                        .context("checkpoint path is not valid UTF-8")?,
                )?;
                let keys: Vec<String> = wm.keys().map(|k| k.to_string()).collect();
                let mut out = HashMap::with_capacity(keys.len());
                for k in keys {
                    let v = wm.take(&k)?;
                    out.insert(k, v);
                }
                out
            }
            "pth" | "pt" | "bin" | "ckpt" => {
                let m = rlx_torch_ckpt::PtModel::open(path)
                    .with_context(|| format!("reading {} as a torch checkpoint", path.display()))?;
                let mut out = HashMap::with_capacity(m.len());
                for name in m.names() {
                    let t = m.tensor(&name)?;
                    out.insert(name, (t.data, t.shape));
                }
                out
            }
            other => bail!(
                "unsupported checkpoint extension {other:?} — expected .pth, .pt, .bin, .ckpt or .safetensors"
            ),
        };

        ensure!(
            !tensors.is_empty(),
            "{} contains no tensors",
            path.display()
        );
        Ok(Self { tensors }.normalize())
    }

    /// Build directly from tensors (tests, and the synthetic fixtures).
    pub fn from_tensors(tensors: HashMap<String, (Vec<f32>, Vec<usize>)>) -> Self {
        Self { tensors }.normalize()
    }

    /// Put the tensors into the one layout the rest of the crate expects.
    ///
    /// Every consumer needs this, not just the runner — `identify` and any
    /// direct `detect` call would otherwise see raw keys and fail to recognize
    /// a perfectly good checkpoint.
    fn normalize(mut self) -> Self {
        strip_profiler_buffers(&mut self);
        self = self.unwrap_wrapper();
        esrgan_to_old_arch(&mut self);
        self
    }

    /// Strip a trainer wrapper prefix when *every* key carries it. Requiring
    /// unanimity matters: a checkpoint that legitimately has a `model.` submodule
    /// alongside other top-level keys must not be silently truncated.
    fn unwrap_wrapper(mut self) -> Self {
        let candidates: Vec<String> = WRAPPERS
            .iter()
            .map(|w| format!("{w}."))
            .filter(|p| !self.is_a_sequential_under(p))
            .filter(|p| self.tensors.keys().any(|k| k.starts_with(p.as_str())))
            .collect();

        for prefix in &candidates {
            if self.tensors.len() > 1 && self.tensors.keys().all(|k| k.starts_with(prefix.as_str()))
            {
                return self.strip(prefix);
            }
        }

        // No single wrapper covers everything. The usual reason is that a
        // trainer saved *two* copies of the same network — BasicSR writes both
        // `params` and `params_ema` — so unanimity can never hold and the
        // checkpoint would be rejected despite being perfectly ordinary.
        //
        // Choosing between them is safe only when the keys left behind are a
        // duplicate copy rather than content, which needs both conditions
        // below: at least two distinct wrappers are present, and between them
        // they account for every key. `WRAPPERS` is in preference order and
        // `params_ema` leads it, which is the copy BasicSR evaluates and
        // releases.
        let all_wrapped = self
            .tensors
            .keys()
            .all(|k| candidates.iter().any(|p| k.starts_with(p.as_str())));
        if candidates.len() > 1 && all_wrapped {
            let prefix = candidates[0].clone();
            self.tensors.retain(|k, _| k.starts_with(prefix.as_str()));
            return self.strip(&prefix);
        }
        self
    }

    /// Whether `prefix` fronts a flattened `nn.Sequential` rather than a
    /// trainer wrapper.
    ///
    /// ESRGAN's old layout puts its *whole network* under `model.`, so
    /// unanimity is not enough to call something a wrapper. A wrapper contains
    /// named submodules; a `Sequential` starts with a bare index.
    fn is_a_sequential_under(&self, prefix: &str) -> bool {
        self.tensors.keys().any(|k| {
            k.strip_prefix(prefix)
                .and_then(|r| r.chars().next())
                .is_some_and(|c| c.is_ascii_digit())
        })
    }

    /// Drop `prefix` from every key, then look for another wrapper beneath it.
    ///
    /// Wrappers do not nest in practice, but re-running is free and makes
    /// `params_ema.state_dict.*` behave.
    fn strip(mut self, prefix: &str) -> Self {
        self.tensors = self
            .tensors
            .drain()
            .map(|(k, v)| (k[prefix.len()..].to_string(), v))
            .collect();
        self.unwrap_wrapper()
    }

    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn shape_of(&self, key: &str) -> Option<&[usize]> {
        self.tensors.get(key).map(|(_, s)| s.as_slice())
    }

    pub fn get(&self, key: &str) -> Option<(&[f32], &[usize])> {
        self.tensors
            .get(key)
            .map(|(d, s)| (d.as_slice(), s.as_slice()))
    }

    /// Keys in sorted order — detection walks these, so the order must be
    /// deterministic or a "highest index wins" scan becomes flaky.
    pub fn keys(&self) -> Vec<&str> {
        let mut k: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        k.sort_unstable();
        k
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn insert(&mut self, key: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) {
        self.tensors.insert(key.into(), (data, shape));
    }

    pub fn remove(&mut self, key: &str) {
        self.tensors.remove(key);
    }

    /// Remove a tensor and hand back its contents.
    pub fn take_owned(&mut self, key: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        self.tensors.remove(key)
    }

    pub fn into_weight_map(self) -> WeightMap {
        WeightMap::from_tensors(self.tensors)
    }

    /// A fresh [`WeightMap`] over a copy of the tensors.
    ///
    /// Building a graph *consumes* its weight map (each tensor is moved into
    /// the graph's parameter table), so recompiling for a different tile size
    /// needs a new one.
    pub fn to_weight_map(&self) -> WeightMap {
        WeightMap::from_tensors(self.tensors.clone())
    }

    /// Every `Conv3XC` in the checkpoint, by prefix (e.g. `block_1.c1_r`).
    pub fn conv3xc_prefixes(&self) -> Vec<String> {
        let mut out = BTreeMap::new();
        for k in self.keys() {
            if let Some(p) = k.strip_suffix(".sk.weight") {
                out.insert(p.to_string(), ());
            }
        }
        out.into_keys().collect()
    }
}

// ── Conv3XC reparameterization ──────────────────────────────────────────

/// Reverse the last two axes of an `[a, b, kh, kw]` tensor (`Tensor.flip(2, 3)`).
fn flip_hw(x: &[f32], shape: [usize; 4]) -> Vec<f32> {
    let [a, b, kh, kw] = shape;
    let mut out = vec![0.0; x.len()];
    for i in 0..a {
        for j in 0..b {
            for y in 0..kh {
                for z in 0..kw {
                    let src = ((i * b + j) * kh + y) * kw + z;
                    let dst = ((i * b + j) * kh + (kh - 1 - y)) * kw + (kw - 1 - z);
                    out[dst] = x[src];
                }
            }
        }
    }
    out
}

/// Swap the first two axes (`Tensor.permute(1, 0, 2, 3)`).
fn permute01(x: &[f32], shape: [usize; 4]) -> (Vec<f32>, [usize; 4]) {
    let [a, b, kh, kw] = shape;
    let mut out = vec![0.0; x.len()];
    for i in 0..a {
        for j in 0..b {
            for y in 0..kh {
                for z in 0..kw {
                    let src = ((i * b + j) * kh + y) * kw + z;
                    let dst = ((j * a + i) * kh + y) * kw + z;
                    out[dst] = x[src];
                }
            }
        }
    }
    (out, [b, a, kh, kw])
}

/// Dense stride-1 `F.conv2d` on the host. Operands here are kernels, not
/// images — the largest is a few hundred kilobytes — so a plain loop is right.
fn conv2d_host(
    x: &[f32],
    xs: [usize; 4],
    w: &[f32],
    ws: [usize; 4],
    pad: usize,
) -> Result<(Vec<f32>, [usize; 4])> {
    let [n, c, h, ww] = xs;
    let [o, wc, kh, kw] = ws;
    ensure!(c == wc, "conv2d_host: {c} input channels vs {wc} in kernel");
    let oh = h + 2 * pad - kh + 1;
    let ow = ww + 2 * pad - kw + 1;
    let mut out = vec![0.0f32; n * o * oh * ow];
    for ni in 0..n {
        for oi in 0..o {
            for y in 0..oh {
                for z in 0..ow {
                    let mut acc = 0.0f32;
                    for ci in 0..c {
                        for ky in 0..kh {
                            let sy = y + ky;
                            if sy < pad || sy >= h + pad {
                                continue;
                            }
                            let sy = sy - pad;
                            for kx in 0..kw {
                                let sx = z + kx;
                                if sx < pad || sx >= ww + pad {
                                    continue;
                                }
                                let sx = sx - pad;
                                acc += x[((ni * c + ci) * h + sy) * ww + sx]
                                    * w[((oi * wc + ci) * kh + ky) * kw + kx];
                            }
                        }
                    }
                    out[((ni * o + oi) * oh + y) * ow + z] = acc;
                }
            }
        }
    }
    Ok((out, [n, o, oh, ow]))
}

fn dims4(shape: &[usize], what: &str) -> Result<[usize; 4]> {
    ensure!(shape.len() == 4, "{what} should be rank 4, got {shape:?}");
    Ok([shape[0], shape[1], shape[2], shape[3]])
}

/// Fold one `Conv3XC` into the single 3×3 its `eval_conv` represents.
///
/// Transliterates the reference `update_params()`: the 1×1 → 3×3 → 1×1 chain is
/// composed by convolving the kernels themselves (in the flipped, channel-
/// transposed domain that turns correlation into composition), the biases are
/// propagated through each stage, and the 1×1 skip is zero-padded to 3×3 and
/// added.
pub fn fuse_conv3xc(ck: &mut Checkpoint, prefix: &str) -> Result<()> {
    let get = |ck: &Checkpoint, name: &str| -> Result<(Vec<f32>, Vec<usize>)> {
        let (d, s) = ck
            .get(name)
            .with_context(|| format!("Conv3XC {prefix}: missing {name}"))?;
        Ok((d.to_vec(), s.to_vec()))
    };

    let (w1, w1s) = get(ck, &format!("{prefix}.conv.0.weight"))?;
    let (b1, _) = get(ck, &format!("{prefix}.conv.0.bias"))?;
    let (w2, w2s) = get(ck, &format!("{prefix}.conv.1.weight"))?;
    let (b2, _) = get(ck, &format!("{prefix}.conv.1.bias"))?;
    let (w3, w3s) = get(ck, &format!("{prefix}.conv.2.weight"))?;
    let (b3, _) = get(ck, &format!("{prefix}.conv.2.bias"))?;
    let (skw, skws) = get(ck, &format!("{prefix}.sk.weight"))?;
    let (skb, _) = get(ck, &format!("{prefix}.sk.bias"))?;

    let w1s = dims4(&w1s, "conv.0.weight")?;
    let w2s = dims4(&w2s, "conv.1.weight")?;
    let w3s = dims4(&w3s, "conv.2.weight")?;
    let skws = dims4(&skws, "sk.weight")?;

    // Stage 1∘2: w = flip(permute(conv2d(permute(flip(w1)), w2, pad=2)))
    let a = flip_hw(&w1, w1s);
    let (a, as_) = permute01(&a, w1s);
    let (t, ts) = conv2d_host(&a, as_, &w2, w2s, 2)?;
    let t = flip_hw(&t, ts);
    let (w, ws) = permute01(&t, ts);

    // b = Σ_{c,y,x} w2[:, c, y, x] · b1[c]  +  b2
    let [o2, c2, kh2, kw2] = w2s;
    let mut b = vec![0.0f32; o2];
    for oi in 0..o2 {
        let mut acc = 0.0f32;
        for ci in 0..c2 {
            for y in 0..kh2 {
                for x in 0..kw2 {
                    acc += w2[((oi * c2 + ci) * kh2 + y) * kw2 + x] * b1[ci];
                }
            }
        }
        b[oi] = acc + b2[oi];
    }

    // Stage (1∘2)∘3, the trailing 1×1.
    let a = flip_hw(&w, ws);
    let (a, as_) = permute01(&a, ws);
    let (t, ts) = conv2d_host(&a, as_, &w3, w3s, 0)?;
    let t = flip_hw(&t, ts);
    let (mut wc, wcs) = permute01(&t, ts);

    let [o3, c3, kh3, kw3] = w3s;
    let mut bc = vec![0.0f32; o3];
    for oi in 0..o3 {
        let mut acc = 0.0f32;
        for ci in 0..c3 {
            for y in 0..kh3 {
                for x in 0..kw3 {
                    acc += w3[((oi * c3 + ci) * kh3 + y) * kw3 + x] * b[ci];
                }
            }
        }
        bc[oi] = acc + b3[oi];
    }

    // The 1×1 skip, zero-padded into the centre of a 3×3.
    let [o, c, kh, kw] = wcs;
    ensure!(
        kh == 3 && kw == 3,
        "Conv3XC {prefix}: fused kernel is {kh}×{kw}, expected 3×3"
    );
    ensure!(
        skws[0] == o && skws[1] == c,
        "Conv3XC {prefix}: skip is {:?}, fused branch is [{o}, {c}, 3, 3]",
        skws
    );
    for oi in 0..o {
        for ci in 0..c {
            wc[((oi * c + ci) * 3 + 1) * 3 + 1] += skw[oi * c + ci];
        }
    }
    for oi in 0..o {
        bc[oi] += skb[oi];
    }

    ck.insert(format!("{prefix}.eval_conv.weight"), wc, vec![o, c, 3, 3]);
    ck.insert(format!("{prefix}.eval_conv.bias"), bc, vec![o]);
    for suffix in [
        "conv.0.weight",
        "conv.0.bias",
        "conv.1.weight",
        "conv.1.bias",
        "conv.2.weight",
        "conv.2.bias",
        "sk.weight",
        "sk.bias",
    ] {
        ck.remove(&format!("{prefix}.{suffix}"));
    }
    Ok(())
}

/// Rewrite a new-arch RRDBNet state dict into the old-arch key layout.
///
/// ESRGAN checkpoints exist in three shapes and only differ by naming:
///
/// | | stem | blocks | trunk conv | tail |
/// |---|---|---|---|---|
/// | old (`ESRGAN`) | `model.0` | `model.1.sub.{i}.RDB{j}.conv{k}.0` | `model.1.sub.{nb}` | `model.{n}` |
/// | new (`Real-ESRGAN`) | `conv_first` | `body.{i}.rdb{j}.conv{k}` | `conv_body` | `conv_up{n}`, `conv_hr`, `conv_last` |
/// | BSRGAN / RealSR | `conv_first` | `RRDB_trunk.{i}.RDB{j}.conv{k}` | `trunk_conv` | `upconv{n}`, `HRconv`, `conv_last` |
///
/// Normalizing to one layout here keeps the graph builder from carrying three
/// naming schemes, and the old one is the right target because its flattened
/// indices are what the scale is recovered from.
///
/// A checkpoint already in the old layout is left alone.
/// Drop `thop` profiler buffers.
///
/// A model profiled with `thop` before being saved carries a `total_ops` and a
/// `total_params` scalar on *every* module, registered as real buffers. The
/// released OmniSR weights are the worst case in this crate: 728 tensors of
/// which 545 are these. They are not architecture — they are measurement
/// residue — and leaving them in would defeat the guard that reports tensors a
/// build never read, which is how several real porting bugs here were caught.
fn strip_profiler_buffers(ck: &mut Checkpoint) {
    ck.tensors.retain(|k, _| {
        !(k == "total_ops"
            || k == "total_params"
            || k.ends_with(".total_ops")
            || k.ends_with(".total_params"))
    });
}

pub fn esrgan_to_old_arch(ck: &mut Checkpoint) {
    // `conv_first` alone is *not* an ESRGAN tell: SwinIR, DRCT, DAT and
    // MambaIRv2 all have one. Renaming on that basis would quietly rewrite
    // every Swin-family checkpoint's stem into `model.0`. Require a key only
    // a residual-dense trunk has.
    let is_rrdb =
        ck.contains("body.0.rdb1.conv1.weight") || ck.contains("RRDB_trunk.0.RDB1.conv1.weight");
    if !is_rrdb {
        return;
    }

    let mut renames: Vec<(String, String)> = Vec::new();
    let mut max_upconv = 0usize;

    for key in ck
        .keys()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>()
    {
        let Some((base, suffix)) = key.rsplit_once('.') else {
            continue;
        };
        if !matches!(suffix, "weight" | "bias") {
            continue;
        }
        let new = match base {
            "conv_first" => Some("model.0".to_string()),
            // The trunk convolution's index is `nb`, filled in below once the
            // block count is known.
            "trunk_conv" | "conv_body" => Some("model.1.sub./NB/".to_string()),
            "conv_hr" | "HRconv" => Some("__hr".to_string()),
            "conv_last" => Some("__last".to_string()),
            other => rename_block(other).or_else(|| {
                rename_upconv(other)
                    .inspect(|idx| max_upconv = max_upconv.max(*idx))
                    .map(|idx| format!("model.{idx}"))
            }),
        };
        if let Some(new) = new {
            let renamed = format!("{new}.{suffix}");
            renames.push((key.clone(), renamed));
        }
    }

    // `nb` is one past the highest block index seen.
    let nb = renames
        .iter()
        .filter_map(|(_, new)| {
            new.strip_prefix("model.1.sub.")
                .and_then(|r| r.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
        })
        .max()
        .map_or(0, |m| m + 1);

    for (old_key, new_key) in renames {
        let resolved = new_key
            .replace("/NB/", &nb.to_string())
            .replace("__hr", &format!("model.{}", max_upconv + 2))
            .replace("__last", &format!("model.{}", max_upconv + 4));
        if let Some((data, shape)) = ck.take_owned(&old_key) {
            ck.insert(resolved, data, shape);
        }
    }
}

/// `body.{i}.rdb{j}.conv{k}` / `RRDB_trunk.{i}.RDB{j}.conv{k}` → old layout.
fn rename_block(base: &str) -> Option<String> {
    let rest = base
        .strip_prefix("body.")
        .or_else(|| base.strip_prefix("RRDB_trunk."))?;
    let mut parts = rest.split('.');
    let i: usize = parts.next()?.parse().ok()?;
    let rdb = parts.next()?;
    let j: usize = rdb
        .strip_prefix("rdb")
        .or_else(|| rdb.strip_prefix("RDB"))?
        .parse()
        .ok()?;
    let conv = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    // `conv1x1` (ESRGAN+) has no trailing `.0` in either layout.
    if conv == "conv1x1" {
        return Some(format!("model.1.sub.{i}.RDB{j}.conv1x1"));
    }
    let k: usize = conv.strip_prefix("conv")?.parse().ok()?;
    Some(format!("model.1.sub.{i}.RDB{j}.conv{k}.0"))
}

/// `upconv{n}` / `conv_up{n}` → `model.{3n}`.
fn rename_upconv(base: &str) -> Option<usize> {
    let n: usize = base
        .strip_prefix("upconv")
        .or_else(|| base.strip_prefix("conv_up"))?
        .parse()
        .ok()?;
    Some(n * 3)
}

/// Fuse every `Conv3XC` found in the checkpoint.
pub fn fuse_all_conv3xc(ck: &mut Checkpoint) -> Result<()> {
    for p in ck.conv3xc_prefixes() {
        fuse_conv3xc(ck, &p).with_context(|| format!("fusing Conv3XC {p}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(keys: &[&str]) -> Checkpoint {
        Checkpoint {
            tensors: keys
                .iter()
                .map(|k| (k.to_string(), (vec![0.0f32], vec![1usize])))
                .collect(),
        }
        .unwrap_wrapper()
    }

    /// BasicSR saves the raw and the exponentially-averaged weights side by
    /// side, so no wrapper is unanimous. The EMA copy is the one released for
    /// evaluation, and `WRAPPERS` is ordered to pick it.
    #[test]
    fn dual_params_checkpoints_take_the_ema_copy() {
        let ck = named(&[
            "params.to_feat.weight",
            "params.to_img.0.weight",
            "params_ema.to_feat.weight",
            "params_ema.to_img.0.weight",
        ]);
        assert!(ck.contains("to_feat.weight"));
        assert_eq!(
            ck.len(),
            2,
            "the non-EMA copy should be dropped, not merged"
        );
    }

    /// Dropping keys is only safe when the survivors are a duplicate copy. A
    /// checkpoint that carries real content outside the wrapper must be left
    /// exactly as it is, or a partial network would reach detection and fail
    /// somewhere far less legible.
    #[test]
    fn a_partial_wrapper_is_left_alone() {
        let ck = named(&["params.to_feat.weight", "some_other_head.weight"]);
        assert!(
            ck.contains("params.to_feat.weight"),
            "a non-wrapper sibling must block stripping"
        );
        assert_eq!(ck.len(), 2);
    }

    fn seeded(n: usize, seed: u64) -> Vec<f32> {
        // Deterministic, spread across sign and magnitude — a constant fill
        // would make a transposed kernel indistinguishable from a correct one.
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    /// Run the unfused Conv3XC as the reference's *training* branch does —
    /// pad the input by 1, run the three convolutions with no further padding,
    /// add the 1×1 skip of the unpadded input — and compare against a single
    /// 3×3 with the fused kernel. If the fold is right they agree exactly.
    #[test]
    fn conv3xc_fusion_matches_the_unfused_branches() {
        let (c_in, c_out, gain) = (3usize, 4usize, 2usize);
        let (h, w) = (5usize, 6usize);

        let w1s = [c_in * gain, c_in, 1, 1];
        let w2s = [c_out * gain, c_in * gain, 3, 3];
        let w3s = [c_out, c_out * gain, 1, 1];
        let sks = [c_out, c_in, 1, 1];

        let mut t = HashMap::new();
        let put =
            |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: &str, s: [usize; 4], seed: u64| {
                let n: usize = s.iter().product();
                t.insert(k.to_string(), (seeded(n, seed), s.to_vec()));
            };
        put(&mut t, "c.conv.0.weight", w1s, 1);
        put(&mut t, "c.conv.1.weight", w2s, 2);
        put(&mut t, "c.conv.2.weight", w3s, 3);
        put(&mut t, "c.sk.weight", sks, 4);
        t.insert("c.conv.0.bias".into(), (seeded(w1s[0], 5), vec![w1s[0]]));
        t.insert("c.conv.1.bias".into(), (seeded(w2s[0], 6), vec![w2s[0]]));
        t.insert("c.conv.2.bias".into(), (seeded(w3s[0], 7), vec![w3s[0]]));
        t.insert("c.sk.bias".into(), (seeded(c_out, 8), vec![c_out]));

        let x = seeded(c_in * h * w, 99);

        // Unfused reference path.
        let mut padded = vec![0.0f32; c_in * (h + 2) * (w + 2)];
        for ci in 0..c_in {
            for y in 0..h {
                for z in 0..w {
                    padded[(ci * (h + 2) + y + 1) * (w + 2) + z + 1] = x[(ci * h + y) * w + z];
                }
            }
        }
        let bias_add = |v: &mut [f32], b: &[f32], o: usize, hw: usize| {
            for oi in 0..o {
                for i in 0..hw {
                    v[oi * hw + i] += b[oi];
                }
            }
        };
        let (mut y1, s1) = conv2d_host(
            &padded,
            [1, c_in, h + 2, w + 2],
            &t["c.conv.0.weight"].0,
            w1s,
            0,
        )
        .unwrap();
        bias_add(&mut y1, &t["c.conv.0.bias"].0, s1[1], s1[2] * s1[3]);
        let (mut y2, s2) = conv2d_host(&y1, s1, &t["c.conv.1.weight"].0, w2s, 0).unwrap();
        bias_add(&mut y2, &t["c.conv.1.bias"].0, s2[1], s2[2] * s2[3]);
        let (mut y3, s3) = conv2d_host(&y2, s2, &t["c.conv.2.weight"].0, w3s, 0).unwrap();
        bias_add(&mut y3, &t["c.conv.2.bias"].0, s3[1], s3[2] * s3[3]);
        let (mut sk, sks_out) =
            conv2d_host(&x, [1, c_in, h, w], &t["c.sk.weight"].0, sks, 0).unwrap();
        bias_add(
            &mut sk,
            &t["c.sk.bias"].0,
            sks_out[1],
            sks_out[2] * sks_out[3],
        );
        let want: Vec<f32> = y3.iter().zip(&sk).map(|(a, b)| a + b).collect();

        // Fused path.
        let mut ck = Checkpoint::from_tensors(t);
        fuse_conv3xc(&mut ck, "c").unwrap();
        let (fw, fws) = ck.get("c.eval_conv.weight").unwrap();
        let (fb, _) = ck.get("c.eval_conv.bias").unwrap();
        assert_eq!(fws, [c_out, c_in, 3, 3]);
        let (mut got, gs) = conv2d_host(&x, [1, c_in, h, w], fw, [c_out, c_in, 3, 3], 1).unwrap();
        bias_add(&mut got, fb, gs[1], gs[2] * gs[3]);

        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() < 2e-5,
                "fused {g} vs unfused {w} (Δ {})",
                (g - w).abs()
            );
        }
    }

    #[test]
    fn fusion_consumes_the_branch_tensors() {
        let mut t = HashMap::new();
        for (k, s) in [
            ("c.conv.0.weight", vec![2, 1, 1, 1]),
            ("c.conv.1.weight", vec![2, 2, 3, 3]),
            ("c.conv.2.weight", vec![1, 2, 1, 1]),
            ("c.sk.weight", vec![1, 1, 1, 1]),
        ] {
            let n: usize = s.iter().product();
            t.insert(k.to_string(), (seeded(n, 1), s));
        }
        t.insert("c.conv.0.bias".into(), (vec![0.0; 2], vec![2]));
        t.insert("c.conv.1.bias".into(), (vec![0.0; 2], vec![2]));
        t.insert("c.conv.2.bias".into(), (vec![0.0; 1], vec![1]));
        t.insert("c.sk.bias".into(), (vec![0.0; 1], vec![1]));
        let mut ck = Checkpoint::from_tensors(t);
        fuse_all_conv3xc(&mut ck).unwrap();
        assert!(ck.contains("c.eval_conv.weight"));
        assert!(!ck.contains("c.sk.weight"));
        assert!(!ck.contains("c.conv.1.weight"));
    }

    /// ESRGAN's old layout is a flattened `nn.Sequential` under `model.`, so
    /// the wrapper stripper sees a unanimous `model.` prefix and would eat the
    /// architecture itself — leaving `0.weight`, `1.sub.…` and no recognizable
    /// model at all.
    #[test]
    fn a_flattened_sequential_is_not_mistaken_for_a_wrapper() {
        let mut t = HashMap::new();
        t.insert("model.0.weight".to_string(), (vec![1.0], vec![1]));
        t.insert(
            "model.1.sub.0.RDB1.conv1.0.weight".to_string(),
            (vec![1.0], vec![1]),
        );
        let ck = Checkpoint::from_tensors(t);
        assert!(
            ck.contains("model.0.weight"),
            "the `model.` prefix was eaten"
        );
    }

    /// `conv_first` is shared by SwinIR, DRCT, DAT and MambaIRv2, so it cannot
    /// be the trigger for ESRGAN's rename — doing so rewrote their stems into
    /// `model.0` and made them undetectable.
    #[test]
    fn esrgan_rename_leaves_non_rrdb_checkpoints_alone() {
        let mut t = HashMap::new();
        for k in [
            "conv_first.weight",
            "conv_after_body.weight",
            "conv_last.weight",
            "layers.0.residual_group.blocks.0.attn.qkv.weight",
        ] {
            t.insert(k.to_string(), (vec![1.0], vec![1]));
        }
        let ck = Checkpoint::from_tensors(t);
        assert!(ck.contains("conv_first.weight"));
        assert!(!ck.contains("model.0.weight"));
    }

    /// A wrapper is stripped only when unanimous, so the `params.` form loads
    /// while a checkpoint with a genuine `model.` submodule keeps its names.
    #[test]
    fn wrapper_prefix_is_stripped_only_when_unanimous() {
        let mut t = HashMap::new();
        t.insert("params.body.0.weight".to_string(), (vec![1.0], vec![1]));
        t.insert("params.body.0.bias".to_string(), (vec![1.0], vec![1]));
        let ck = Checkpoint::from_tensors(t);
        assert!(ck.contains("body.0.weight"));

        let mut t = HashMap::new();
        t.insert("model.a".to_string(), (vec![1.0], vec![1]));
        t.insert("head.b".to_string(), (vec![1.0], vec![1]));
        let ck = Checkpoint::from_tensors(t);
        assert!(ck.contains("model.a"));
    }
}
