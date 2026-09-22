//! A `prism.hadamard`-folded model must compute exactly what the
//! unfolded model computes.
//!
//! `prism-ml/Ternary-Bonsai-2-27B-gguf` stores its weights in a rotated
//! basis: each weight row is permuted, sign-flipped and Hadamard-rotated
//! offline, and the runtime has to apply the identical transform to the
//! *activation* before every matmul. Four conventions have to agree for
//! that to be the same function:
//!
//!   1. the order of the three steps — permute, then signs, then
//!      rotation (any other order is a different linear map);
//!   2. the rotation's orientation and its `1/√block` normalization;
//!   3. how the feature axis is blocked — `[-1, block]` with each block
//!      contiguous, not strided;
//!   4. the GDN `ssm_out` head reorder, tiled `[rep, nk, hd]` to grouped
//!      `[nk, rep, hd]`.
//!
//! Every one of those is silent when wrong. The shapes all still check
//! out, the model still runs, and — because the rotation is orthogonal —
//! the activations keep their magnitude, so it still emits fluent text,
//! just not this model's text. Eyeballing output cannot catch it. So
//! this test builds the *same model twice* — [`synth_weights_folded`]
//! returns a folded bundle and its algebraically exact dense
//! equivalent — and requires the logits to agree.
//!
//! What this cannot cover is which tensors the publisher actually
//! folded; that is a property of the checkpoint and is read from
//! `prism.hadamard.weight_names` at load.

use rlx_qwen35::synth::{synth_weights_folded, tiny_cfg};
use rlx_runtime::{Device, Session};

const BATCH: usize = 1;
const SEQ: usize = 4;
const INPUT_IDS: [f32; SEQ] = [5.0, 11.0, 2.0, 19.0];

fn logits_on(device: Device, folded: bool) -> Vec<f32> {
    let cfg = tiny_cfg();
    let weights = synth_weights_folded(&cfg, folded);
    let (hir, params, packed) =
        rlx_qwen35::build_qwen35_prefill_flow(&cfg, &weights, BATCH, SEQ, true, false, false)
            .expect("build prefill flow");
    assert!(packed.is_empty(), "synthetic weights should not be packed");

    let mut compiled = Session::new(device)
        .compile_hir(hir)
        .expect("compile prefill");
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    compiled.run(&[("input_ids", &INPUT_IDS[..])]).remove(0)
}

#[test]
fn folded_weights_match_their_unfolded_equivalent() {
    check_device(Device::Cpu);
}

#[test]
fn folded_weights_match_their_unfolded_equivalent_metal() {
    if !cfg!(feature = "metal") {
        eprintln!("skip: built without the metal feature");
        return;
    }
    check_device(Device::Metal);
}

fn check_device(dev: Device) {
    let dense = logits_on(dev, false);
    let folded = logits_on(dev, true);
    assert_eq!(dense.len(), folded.len(), "logit shape changed");

    // Guard the guard: against constant logits "they match" is vacuous.
    let spread = dense.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - dense.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    assert!(
        spread > 1e-3,
        "dense reference logits are nearly constant (spread {spread}); \
         this fixture cannot distinguish a correct rotation"
    );

    let max_abs = dense
        .iter()
        .zip(&folded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Same arithmetic, different association order — the rotation sums
    // `block` terms per element, so this is f32 reassociation, not a
    // modelling difference.
    assert!(
        max_abs < 2e-3,
        "Hadamard-folded projections diverge from the weights they encode: \
         max |Δlogit| = {max_abs} over {} values (logit spread {spread})",
        dense.len()
    );
}

/// The folded bundle must actually take the folded path, or the
/// equivalence test above passes by construction.
#[test]
fn folded_bundle_reports_folded_projections() {
    use rlx_qwen35::{Qwen35LayerFfn, Qwen35TrunkLayer};
    let cfg = tiny_cfg();
    let w = synth_weights_folded(&cfg, true);

    let mut folded = 0usize;
    let mut with_perm = 0usize;
    let mut count = |p: &rlx_qwen35::Proj| {
        if let rlx_qwen35::Proj::Folded(f) = p {
            folded += 1;
            if f.fold.perm.is_some() {
                with_perm += 1;
            }
        }
    };
    for layer in &w.trunk_layers {
        let ffn = match layer {
            Qwen35TrunkLayer::Linear(l) => {
                for p in [&l.attn_qkv, &l.attn_gate, &l.ssm_out] {
                    count(p);
                }
                // The recurrent-state path stays higher precision and
                // unfolded in the real checkpoint.
                assert!(
                    matches!(l.ssm_alpha, rlx_qwen35::Proj::Dense(_)),
                    "ssm_alpha should not be folded"
                );
                assert!(
                    matches!(l.ssm_beta, rlx_qwen35::Proj::Dense(_)),
                    "ssm_beta should not be folded"
                );
                &l.ffn
            }
            Qwen35TrunkLayer::FullAttn(l) => {
                for p in [&l.attn_q_gate, &l.attn_k, &l.attn_v, &l.attn_output] {
                    count(p);
                }
                &l.ffn
            }
        };
        if let Qwen35LayerFfn::Dense { gate, up, down } = ffn {
            for p in [gate, up, down] {
                count(p);
            }
        }
    }

    assert!(folded > 0, "no folded projections in the folded bundle");
    assert!(
        with_perm > 0,
        "no folded projection carried the GDN head permutation — the one \
         part of the transform that is invisible when wrong"
    );
    assert!(
        w.output_fold.is_some(),
        "the LM head reaches the graph outside emit_linear and must carry its own fold"
    );
}

/// The dense half must contain no folded projection, or "the same model
/// twice" is really the same bundle twice.
#[test]
fn dense_bundle_is_actually_unfolded() {
    use rlx_qwen35::Qwen35TrunkLayer;
    let cfg = tiny_cfg();
    let w = synth_weights_folded(&cfg, false);
    assert!(w.output_fold.is_none());
    for layer in &w.trunk_layers {
        let projs: Vec<&rlx_qwen35::Proj> = match layer {
            Qwen35TrunkLayer::Linear(l) => vec![&l.attn_qkv, &l.attn_gate, &l.ssm_out],
            Qwen35TrunkLayer::FullAttn(l) => {
                vec![&l.attn_q_gate, &l.attn_k, &l.attn_v, &l.attn_output]
            }
        };
        for p in projs {
            assert!(
                !matches!(p, rlx_qwen35::Proj::Folded(_)),
                "dense reference bundle carries a folded projection"
            );
        }
    }
}
