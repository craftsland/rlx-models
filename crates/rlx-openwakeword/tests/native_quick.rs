use rlx_openwakeword::{
    OpenWakeWordEngine, OpenWakeWordWeights, WakeConfig, WakeEngine, score_wav,
};

fn scores(pcm: &[f32]) -> Vec<f32> {
    let mut eng =
        OpenWakeWordEngine::new(OpenWakeWordWeights::stub("alexa"), WakeConfig::default());
    score_wav(&mut eng, pcm)
        .unwrap()
        .iter()
        .map(|s| s.score)
        .collect()
}

type Case = (&'static str, fn(&[f32]) -> Vec<f32>);
const CASES: [Case; 1] = [("alexa", scores)];

/// `reset()` must actually clear the streaming buffers.
#[test]
fn reset_restores_a_fresh_stream() {
    let mut eng =
        OpenWakeWordEngine::new(OpenWakeWordWeights::stub("alexa"), WakeConfig::default());
    let first = eng.push_pcm(&vec![0.0f32; 1280]).unwrap();
    eng.push_pcm(&vec![0.5f32; 1280 * 3]).unwrap();
    eng.reset();
    let after = eng.push_pcm(&vec![0.0f32; 1280]).unwrap();
    assert_eq!(after.len(), first.len(), "reset changed the framing");
    let a: Vec<f32> = first.iter().map(|s| s.score).collect();
    let b: Vec<f32> = after.iter().map(|s| s.score).collect();
    assert_eq!(a, b, "reset did not restore the initial stream state");
}

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
