use rlx_porcupine::{PorcupineEngine, PorcupineWeights, WakeConfig, score_wav};

fn scores(pcm: &[f32]) -> Vec<f32> {
    let mut eng = PorcupineEngine::new(PorcupineWeights::stub("porcupine"), WakeConfig::default());
    score_wav(&mut eng, pcm)
        .unwrap()
        .iter()
        .map(|s| s.score)
        .collect()
}

fn tone(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.05).sin() * 0.3).collect()
}

/// Deterministic, correctly framed, in-range scores — not merely finite ones.
///
/// `is_finite` is passed by an engine returning all zeros, or garbage, or a
/// different number of frames than the audio implies. These properties are not.
///
/// What is deliberately **not** asserted here is that different audio scores
/// differently. `WakeCnnWeights::stub` sets `fc2_b = -2.0` over a conv stack of
/// ±0.01 weights and zero biases, whose ReLU output is identically zero — so
/// every frame of every input scores exactly `sigmoid(-2) = 0.1192`. That is
/// the stub behaving as built, not the scorer ignoring its input, and asserting
/// sensitivity here would be asserting something false. Input sensitivity
/// belongs in a test with trained weights.
#[test]
fn stub_scores_are_deterministic_and_correctly_framed() {
    let silence = vec![0.0f32; 16_000];
    let voiced = tone(16_000);

    let a = scores(&silence);
    assert!(!a.is_empty(), "no scores produced");
    assert!(a.iter().all(|s| s.is_finite()), "non-finite score");
    assert!(
        a.iter().all(|s| (0.0..=1.0).contains(s)),
        "score outside [0, 1]: {a:?}"
    );
    assert_eq!(a, scores(&silence), "scoring is not deterministic");

    // Framing must follow the audio, not the content.
    let b = scores(&voiced);
    assert_eq!(
        a.len(),
        b.len(),
        "same-length audio produced a different number of frames"
    );
    let half = scores(&silence[..8_000]);
    assert!(
        half.len() < a.len(),
        "half the audio produced {} frames vs {} — framing ignores input length",
        half.len(),
        a.len()
    );
}
