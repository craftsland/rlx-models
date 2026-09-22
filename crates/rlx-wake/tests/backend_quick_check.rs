use rlx_runtime::Device;
use rlx_wake::{
    MelConfig, MelFrontend, WakeCnn, WakeCnnConfig, WakeCnnWeights, WakeConfig, WakeEngine,
    WakeStep, available_devices, bench_device_label, bind_streaming_device, resolve_device,
    score_wav, streaming_execution_device,
};

struct ProbeEngine {
    cfg: WakeConfig,
    mel: MelFrontend,
    cnn: WakeCnn,
    samples_seen: usize,
    last_fire_ms: f32,
    device_label: &'static str,
}

impl ProbeEngine {
    fn new(device: Device) -> Self {
        let (_, label) = bind_streaming_device(device).expect("bind device");
        let weights = WakeCnnWeights::stub(WakeCnnConfig::lite());
        Self {
            cfg: WakeConfig::default(),
            mel: MelFrontend::new(MelConfig::default()),
            cnn: WakeCnn::new(weights),
            samples_seen: 0,
            last_fire_ms: -1.0e9,
            device_label: label,
        }
    }
}

impl WakeEngine for ProbeEngine {
    fn push_pcm(&mut self, samples: &[f32]) -> anyhow::Result<Vec<WakeStep>> {
        let frames = self.mel.push(samples);
        let score = self.cnn.push_mel_frames(&frames);
        self.samples_seen += samples.len();
        let t_ms = self.samples_seen as f32 * 1000.0 / 16_000.0;
        let mut fired = score >= self.cfg.threshold;
        if fired && t_ms - self.last_fire_ms < self.cfg.cooldown_ms {
            fired = false;
        }
        if fired {
            self.last_fire_ms = t_ms;
        }
        Ok(vec![WakeStep { score, fired, t_ms }])
    }

    fn reset(&mut self) {
        self.mel.reset();
        self.cnn.reset();
        self.samples_seen = 0;
        self.last_fire_ms = -1.0e9;
    }

    fn config(&self) -> &WakeConfig {
        &self.cfg
    }
}

#[test]
fn cpu_streaming_path() {
    let device = resolve_device("cpu").unwrap();
    assert_eq!(streaming_execution_device(device), Device::Cpu);
    let mut eng = ProbeEngine::new(device);
    let pcm = vec![0.0f32; 16_000];
    let steps = score_wav(&mut eng, &pcm).unwrap();
    assert!(!steps.is_empty());
    assert!(steps.iter().all(|s| s.score.is_finite()));
}

/// Every backend must produce the *same* scores as CPU, not merely finite ones.
///
/// This swept the devices already but only asked each for finite numbers, which
/// a backend returning zeros — or one silently falling back to a different
/// execution path — passes just as easily. The scores are deterministic for a
/// fixed stub and a fixed PCM buffer, so they should agree exactly up to f32
/// reassociation.
#[test]
fn all_available_backends_score_like_cpu() {
    let pcm = vec![0.0f32; 1280 * 4];
    let devices = available_devices();
    assert!(!devices.is_empty(), "expected at least cpu");

    let scores_for = |device: Device| -> (&'static str, Vec<f32>) {
        let (exec, label) = bind_streaming_device(device).unwrap();
        assert_eq!(exec, device);
        assert_eq!(label, bench_device_label(device));
        let mut eng = ProbeEngine::new(device);
        assert_eq!(eng.device_label, label);
        let steps = score_wav(&mut eng, &pcm).unwrap();
        assert!(!steps.is_empty(), "device={label}");
        (label, steps.iter().map(|s| s.score).collect())
    };

    let (_, cpu) = scores_for(Device::Cpu);
    assert!(
        cpu.iter().all(|v| v.is_finite()),
        "CPU reference scores are not finite"
    );
    let scale = cpu.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);

    let mut bad = Vec::new();
    for device in devices {
        if device == Device::Cpu {
            continue;
        }
        let (label, got) = scores_for(device);
        if got.len() != cpu.len() {
            bad.push(format!(
                "  {label}: {} steps, CPU gave {}",
                got.len(),
                cpu.len()
            ));
            continue;
        }
        let rel = got
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / scale;
        if !rel.is_finite() || rel >= 1e-3 {
            bad.push(format!("  {label}: scores differ from CPU by {rel:.6}"));
        }
    }
    assert!(
        bad.is_empty(),
        "wake scores differ from CPU on {} backend(s):\n{}",
        bad.len(),
        bad.join("\n")
    );
}
