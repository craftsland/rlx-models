use rlx_nanowakeword::{NanoWakeWordEngine, NanoWakeWordWeights, WakeConfig, score_wav};

fn scores(lite: bool, pcm: &[f32]) -> Vec<f32> {
    let w = NanoWakeWordWeights::stub(lite, "hey nano");
    let mut eng = NanoWakeWordEngine::new(w, WakeConfig::default());
    score_wav(&mut eng, pcm)
        .unwrap()
        .iter()
        .map(|s| s.score)
        .collect()
}

type Case = (&'static str, fn(&[f32]) -> Vec<f32>);
const CASES: [Case; 2] = [
    ("lite", |p| scores(true, p)),
    ("full", |p| scores(false, p)),
];

fn tone(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.05).sin() * 0.3).collect()
}

/// Deterministic, correctly framed, in-range scores — not merely finite ones.
///
/// `is_finite` is passed by an engine returning all zeros, or a different number
/// of frames than the audio implies. These properties are not.
///
/// Deliberately **not** asserted: that different audio scores differently.
/// `WakeCnnWeights::stub` sets `fc2_b = -2.0` over a conv stack of ±0.01 weights
/// and zero biases whose ReLU output is identically zero, so every frame of
/// every input scores exactly `sigmoid(-2)`. That is the stub behaving as built,
/// and asserting sensitivity here would assert something false.
#[test]
fn stub_scores_are_deterministic_and_correctly_framed() {
    for (label, mk) in CASES {
        let silence = vec![0.0f32; 16_000];
        let a = mk(&silence);
        assert!(!a.is_empty(), "{label}: no scores produced");
        assert!(a.iter().all(|s| s.is_finite()), "{label}: non-finite score");
        assert!(
            a.iter().all(|s| (0.0..=1.0).contains(s)),
            "{label}: score outside [0, 1]"
        );
        assert_eq!(a, mk(&silence), "{label}: scoring is not deterministic");

        let b = mk(&tone(16_000));
        assert_eq!(a.len(), b.len(), "{label}: framing depends on content");
        let half = mk(&silence[..8_000]);
        assert!(
            half.len() < a.len(),
            "{label}: half the audio gave {} frames vs {} — framing ignores length",
            half.len(),
            a.len()
        );
    }
}
