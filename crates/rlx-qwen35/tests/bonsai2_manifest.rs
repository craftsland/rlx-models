//! `prism-ml/Ternary-Bonsai-2-27B-gguf` checked against its real header.
//!
//! The published model is 5.95 GB, but everything this crate decides at
//! load time — the architecture, the quant type, which weights are
//! stored Hadamard-folded, and how wide each sign vector is — lives in
//! the GGUF header, ahead of the data. So the fixture here is the real
//! header with the two tokenizer arrays stripped: 175 KB carrying the
//! genuine `prism.hadamard.*` block and all 851 real tensor infos.
//! Regenerate with `scripts/bonsai2_manifest.py`.
//!
//! This is the check that the port matches the *published* model rather
//! than matching my reading of the fork's source. A synthetic fixture
//! cannot catch a misread metadata key, a sign vector split at the wrong
//! offset, or a folded-weight list that misses a tensor kind.

use rlx_gguf::{GgmlType, GgufFile};
use rlx_qwen35::Qwen35Config;
use rlx_qwen35::prism_hadamard::{FoldCtx, PrismHadamard};

fn manifest() -> GgufFile {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bonsai2_manifest.gguf"
    );
    GgufFile::header_from_path(path).expect("parse Bonsai 2 manifest fixture")
}

#[test]
fn resolves_ptq1_0_and_not_pestles_g8_0() {
    let f = manifest();
    // Both PrismML and Doses AI's mortar.cpp fork claim ggml type 143.
    // Nothing in the id distinguishes them, so the reader picks a
    // dialect from the file's own metadata; reading these tensors as
    // G8_0 would mis-stride every weight (16 B / 32 elems vs 28 / 128).
    let embd = f.get("token_embd.weight").expect("token_embd.weight");
    assert_eq!(
        embd.dtype,
        GgmlType::PTQ1_0,
        "type 143 resolved as {:?}",
        embd.dtype
    );

    let quant = f
        .tensors
        .values()
        .filter(|t| t.dtype == GgmlType::PTQ1_0)
        .count();
    // 401 folded weights + the rotated token_embd table.
    assert_eq!(quant, 402, "PTQ1_0 tensor count");
}

#[test]
fn config_matches_the_published_model() {
    let cfg = Qwen35Config::from_gguf(&manifest()).expect("parse config");
    assert_eq!(cfg.num_hidden_layers, 64);
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.intermediate_size, 17408);
    assert_eq!(cfg.num_attention_heads, 24);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.full_attention_interval, 4);
    assert_eq!(cfg.ssm_state_size, 128);
    assert_eq!(cfg.ssm_group_count, 16);
    assert_eq!(cfg.ssm_time_step_rank, 48);
    assert_eq!(cfg.ssm_inner_size, 6144);
    assert_eq!(cfg.rope_dim_sections, vec![11, 11, 10, 0]);
    // Embed and LM head are separate tensors here; Bonsai 1 tied them.
    assert!(manifest().get("output.weight").is_some());
}

#[test]
fn hadamard_block_parses_and_covers_every_folded_tensor() {
    let f = manifest();
    let h = PrismHadamard::from_gguf(&f)
        .expect("parse prism.hadamard")
        .expect("Bonsai 2 declares a rotation");

    assert_eq!(h.block_size, 1024);
    assert_eq!(h.folded_count(), 401);
    assert!(h.is_inverse("token_embd.weight"));
    assert!(
        !h.is_folded("token_embd.weight"),
        "the table is inverse-only"
    );
    assert!(h.is_folded("output.weight"), "the LM head is folded");

    // Every folded name must name a real tensor whose input width the
    // block size divides — the check that would fail first if the fold
    // list and the tensor list had drifted apart.
    h.validate_widths(|name| f.get(name).map(|t| t.shape[0]))
        .expect("every folded weight present with a divisible width");

    // The three sign widths in the file are 5120 / 6144 / 17408; ask for
    // each via the public path so a mis-split `sign_values` shows up.
    for w in [5120usize, 6144, 17408] {
        let signs = h
            .signs_for(w)
            .unwrap_or_else(|e| panic!("signs for width {w}: {e}"))
            .unwrap_or_else(|| panic!("width {w} has no sign vector"));
        assert_eq!(signs.len(), w);
        assert!(signs.iter().all(|&s| s == 1.0 || s == -1.0));
        // An all-+1 vector would mean the split landed on padding.
        assert!(signs.iter().any(|&s| s < 0.0), "width {w} signs are all +1");
    }
    assert!(
        h.signs_for(4096).is_err(),
        "an unlisted width must fail loudly, not silently read as identity"
    );
}

#[test]
fn every_ternary_matmul_weight_is_accounted_for() {
    // If the publisher folded a tensor kind this port does not route
    // through the transform, the model still runs and still emits
    // fluent text. So require the two sets to match exactly rather than
    // trusting the fold list to be complete.
    let f = manifest();
    let h = PrismHadamard::from_gguf(&f).unwrap().unwrap();
    let mut unaccounted: Vec<&str> = f
        .tensors
        .values()
        .filter(|t| t.dtype == GgmlType::PTQ1_0)
        .map(|t| t.name.as_str())
        .filter(|n| !h.is_folded(n) && !h.is_inverse(n))
        .collect();
    unaccounted.sort_unstable();
    assert!(
        unaccounted.is_empty(),
        "ternary weights with no declared transform: {unaccounted:?}"
    );
}

#[test]
fn ssm_out_is_the_only_permuted_weight() {
    let f = manifest();
    let cfg = Qwen35Config::from_gguf(&f).unwrap();
    let h = PrismHadamard::from_gguf(&f).unwrap().unwrap();
    let ctx = FoldCtx::new(h.clone(), cfg.ssm_time_step_rank, cfg.ssm_group_count);

    let mut permuted = 0usize;
    let mut folded = 0usize;
    for t in f.tensors.values() {
        if !h.is_folded(&t.name) {
            continue;
        }
        let fold = ctx
            .for_weight(&t.name, t.shape[0])
            .unwrap_or_else(|e| panic!("{}: {e}", t.name))
            .unwrap_or_else(|| panic!("{} declared folded but resolved to None", t.name));
        folded += 1;
        assert_eq!(fold.block_size, 1024);
        assert!(fold.signs.is_some(), "{} lost its sign vector", t.name);
        if let Some(p) = fold.perm {
            assert!(t.name.ends_with(".ssm_out.weight"), "{} permuted", t.name);
            // 48 value heads over 16 groups => 3 repeats of a 128-wide head.
            assert_eq!((p.hd, p.nk, p.rep), (128, 16, 3));
            permuted += 1;
        }
    }
    assert_eq!(folded, 401);
    // One per linear-attention layer: 64 blocks less the 16 full-attention ones.
    assert_eq!(
        permuted, 48,
        "expected one permuted ssm_out per linear layer"
    );
}
