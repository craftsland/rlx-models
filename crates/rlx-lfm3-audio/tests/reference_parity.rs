//! Parity with `scripts/lfm3_audio_reference.py` — a float32 torch port of
//! LiquidAI's released modeling code over the bf16 safetensors checkpoint.
//!
//! `tests/fixtures/reference.json` records, for `assets/harness/jfk_16k.wav`:
//! the log-mel frontend, the encoder + projector output, one LM forward,
//! greedy ASR, the depth decoder on fixed inputs, greedy TTS frames (codebooks
//! revealed in index order) and the codec decoding them.
//!
//! The frontend check needs no weights. The rest load `LFM3_AUDIO_GGUF`
//! (default `weights/lm/lfm3-audio-2b-gguf`) — the `F16` LM by default
//! (`LFM3_AUDIO_QUANT` to pick another; tolerances below are for F16, which
//! is the bf16 checkpoint re-encoded) with the mmproj picked by
//! `LFM3_AUDIO_MMPROJ_QUANT` (default F16) — on `LFM3_AUDIO_DEVICE` (default
//! `cpu`), in low-memory mode with `LFM3_AUDIO_LOW_MEMORY=1`, and skip when
//! it is absent.

use std::path::PathBuf;

use rlx_lfm3_audio::{GenerateOptions, Lfm3Audio, RevealOrder, Sampling, Segment, mel, wav};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn reference() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reference.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn ints(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn jfk() -> Vec<f32> {
    wav::read_16k(&root().join("assets/harness/jfk_16k.wav")).unwrap()
}

#[test]
fn log_mel_matches_reference() {
    let r = &reference()["mel"];
    let pcm = jfk();
    let feats = mel::log_mel(&pcm, 128);
    let frames = mel::num_frames(pcm.len());
    assert_eq!(frames, r["shape"][1].as_u64().unwrap() as usize);
    let col = |t: usize| (0..128).map(|m| feats[m * frames + t]).collect::<Vec<_>>();
    let e0 = max_abs(&col(0), &floats(&r["frame0"]));
    let em = max_abs(&col(frames / 2), &floats(&r["frame_mid"]));
    let sum: f64 = feats.iter().map(|&x| x as f64).sum();
    let want = r["sum"].as_f64().unwrap();
    eprintln!("mel: frame0 {e0:.2e}, mid {em:.2e}, sum {sum:.4} vs {want:.4}");
    assert!(e0 < 1e-4 && em < 1e-4);
    assert!((sum - want).abs() / want.abs() < 1e-5);
}

/// The model, and whether its LM is the bf16 checkpoint re-encoded (F16 /
/// BF16: strict bounds) rather than a packed quant (Q8_0, Q4_K_M: the LM
/// stages get quantization-sized bounds), and the same for the mmproj
/// (`LFM3_AUDIO_MMPROJ_QUANT`, default F16) behind the encoder, depth decoder
/// and codec.
fn open() -> Option<(Lfm3Audio, bool, bool)> {
    let dir = rlx_ir::env::read("LFM3_AUDIO_GGUF")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("weights/lm/lfm3-audio-2b-gguf"));
    if !dir.is_dir() {
        eprintln!("skip: {} missing", dir.display());
        return None;
    }
    let quant = rlx_ir::env::read("LFM3_AUDIO_QUANT")
        .ok()
        .unwrap_or_else(|| "F16".into());
    let device = rlx_ir::env::read("LFM3_AUDIO_DEVICE")
        .ok()
        .unwrap_or_else(|| "cpu".into());
    let device = rlx_cli::parse_device(&device).unwrap();
    let mm = rlx_ir::env::read("LFM3_AUDIO_MMPROJ_QUANT")
        .ok()
        .unwrap_or_else(|| "F16".into());
    let strict = matches!(quant.as_str(), "F16" | "BF16");
    let mm_strict = matches!(mm.as_str(), "F16" | "BF16");
    eprintln!("LM {quant} + mmproj {mm} on {device:?}");
    Some((
        {
            let mut m = Lfm3Audio::open_dir(&dir, Some(&quant), Some(&mm), device).unwrap();
            // `LFM3_AUDIO_LOW_MEMORY=1`: one LM graph, stage graphs dropped.
            m.set_low_memory(rlx_ir::env::read("LFM3_AUDIO_LOW_MEMORY").is_ok_and(|v| v == "1"));
            m
        },
        strict,
        mm_strict,
    ))
}

#[test]
fn pipeline_matches_reference() {
    let Some((mut m, strict, mm_strict)) = open() else {
        return;
    };
    let r = reference();
    let h = m.cfg.lm.hidden;
    eprintln!(
        "LM {} layers: {}",
        m.cfg.lm.layers.len(),
        rlx_lfm3_audio::model::layer_summary(&m.cfg)
    );

    // Speech encoder + projector.
    let ra = &r["audio_features"];
    let feats = m.encode_audio(&jfk()).unwrap();
    let n = feats.len() / h;
    assert_eq!(n, ra["shape"][0].as_u64().unwrap() as usize);
    let row0 = &feats[..64];
    let last = &feats[(n - 1) * h..(n - 1) * h + 64];
    let c0 = cosine(row0, &floats(&ra["row0"]));
    let cl = cosine(last, &floats(&ra["row_last"]));
    let rms = (feats.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / feats.len() as f64).sqrt();
    eprintln!(
        "encoder: row0 cos {c0:.6} max|Δ| {:.2e}, last cos {cl:.6}, rms {rms:.5} vs {:.5}",
        max_abs(row0, &floats(&ra["row0"])),
        ra["rms"].as_f64().unwrap()
    );
    let enc_tol = if mm_strict { 0.999 } else { 0.99 };
    assert!(c0 > enc_tol && cl > enc_tol, "encoder output differs");

    // One LM forward.
    let rl = &r["lm"];
    let ids = ints(&rl["ids"]);
    let mut st = m.lm_state(ids.len() + 1).unwrap();
    let emb = m.embed_tokens(&ids).unwrap();
    let (hidden, logits) = m.lm_feed(&mut st, &emb).unwrap();
    let ch = cosine(&hidden[..64], &floats(&rl["hidden_last"]));
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top: Vec<u32> = order[..5].iter().map(|&i| i as u32).collect();
    let top_logits: Vec<f32> = order[..5].iter().map(|&i| logits[i]).collect();
    eprintln!(
        "lm: hidden cos {ch:.6}, top5 {top:?} {top_logits:?} vs {:?} {:?}",
        ints(&rl["top5_ids"]),
        floats(&rl["top5_logits"])
    );
    assert_eq!(top[0], ints(&rl["top5_ids"])[0]);
    // 4-bit LM weights move the hidden state by ~1% (Q4_K_M: cos 0.989).
    let four_bit = rlx_ir::env::read("LFM3_AUDIO_QUANT").is_ok_and(|q| q.starts_with("Q4"));
    assert!(ch > if four_bit { 0.98 } else { 0.999 });
    let logit_tol = if strict { 0.1 } else { 0.5 };
    assert!(max_abs(&top_logits[..1], &floats(&rl["top5_logits"])[..1]) < logit_tol);

    // Greedy ASR from the reference's prompt ids.
    let rs = &r["asr"];
    let prompt = ints(&rs["prompt_ids"]);
    let a = prompt
        .iter()
        .position(|&t| t == m.cfg.tokens.audio)
        .unwrap();
    let b = prompt
        .iter()
        .rposition(|&t| t == m.cfg.tokens.audio)
        .unwrap()
        + 1;
    let segs = vec![
        Segment::Tokens(prompt[..a].to_vec()),
        Segment::Audio(feats.clone()),
        Segment::Tokens(prompt[b..].to_vec()),
    ];
    let mut opts = GenerateOptions::new(&m.cfg);
    opts.text_only = true;
    opts.max_new_tokens = 64;
    let out = m.generate(&segs, &opts, |_| {}, |_| {}).unwrap();
    eprintln!("asr: {:?}", out.tokens);
    assert_eq!(
        out.tokens,
        ints(&rs["generated"]),
        "greedy ASR tokens differ"
    );

    // Depth decoder on fixed inputs.
    let rd = &r["depth"];
    let hid = floats(&rd["hidden"]);
    let codes = ints(&rd["codes"]);
    let masked: Vec<bool> = rd["masked"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    let revealed: Vec<bool> = masked.iter().map(|m| !m).collect();
    let dl = m.depth_logits(&hid, &codes, &revealed).unwrap();
    let cs = m.cfg.codebook_size;
    let mut worst = 0f32;
    let mut argmax = Vec::new();
    for (q, want) in rd["logits_head"].as_array().unwrap().iter().enumerate() {
        let row = &dl[q * cs..(q + 1) * cs];
        worst = worst.max(max_abs(&row[..32], &floats(want)));
        argmax.push(rlx_lfm3_audio::sampling::argmax(row));
    }
    eprintln!("depth: max|Δ| {worst:.2e}, argmax {argmax:?}");
    if mm_strict {
        assert_eq!(argmax, ints(&rd["argmax"]));
        assert!(worst < 2e-2);
    } else {
        let same = argmax
            .iter()
            .zip(ints(&rd["argmax"]))
            .filter(|(a, b)| **a == *b)
            .count();
        assert!(same >= 9, "depth argmax agrees on {same}/12 codebooks");
    }

    // Greedy TTS frames.
    let rt = &r["tts"];
    let want: Vec<Vec<u32>> = rt["frames"].as_array().unwrap().iter().map(ints).collect();
    let mut opts = GenerateOptions::new(&m.cfg);
    opts.speech = Sampling::GREEDY;
    opts.reveal = RevealOrder::Index;
    opts.max_new_tokens = want.len() + 1;
    let segs = vec![Segment::Tokens(ints(&rt["prompt_ids"]))];
    let out = m.generate(&segs, &opts, |_| {}, |_| {}).unwrap();
    let same = out
        .frames
        .iter()
        .zip(&want)
        .take_while(|(a, b)| a == b)
        .count();
    let (agree, total) = out
        .frames
        .iter()
        .zip(&want)
        .flat_map(|(a, b)| a.iter().zip(b))
        .fold((0, 0), |(n, t), (a, b)| (n + (a == b) as usize, t + 1));
    let first = out.frames.first().map_or(0, |f| {
        f.iter().zip(&want[0]).filter(|(a, b)| a == b).count()
    });
    eprintln!(
        "tts: {} frames, first {same} identical to the reference's {}; codes agreeing {agree}/{total}, frame 0 {first}/12",
        out.frames.len(),
        want.len()
    );
    if strict && mm_strict {
        assert!(
            same >= want.len().min(8),
            "greedy TTS diverged at frame {same}"
        );
    } else {
        // Quantized weights move near-tied argmaxes; one changed code then
        // changes every later frame. Require the first frame to stay close.
        assert!(
            first >= 9,
            "greedy TTS frame 0 shares only {first}/12 codes"
        );
    }

    // Codec on the reference's frames.
    let rc = &r["codec"];
    let frames: Vec<Vec<u32>> = rc["codes"].as_array().unwrap().iter().map(ints).collect();
    let pcm = m.decode_frames(&frames).unwrap();
    assert_eq!(pcm.len(), rc["num_samples"].as_u64().unwrap() as usize);
    let head = floats(&rc["head"]);
    let cc = cosine(&pcm[..head.len()], &head);
    let rms = (pcm.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / pcm.len() as f64).sqrt();
    eprintln!(
        "codec: head cos {cc:.6} max|Δ| {:.2e}, rms {rms:.5} vs {:.5}",
        max_abs(&pcm[..head.len()], &head),
        rc["rms"].as_f64().unwrap()
    );
    assert!(cc > if mm_strict { 0.999 } else { 0.98 });

    // FSQ table vs the quantizer's own formula.
    let fsq = m
        .fsq_latents(&(0..16).map(|c| vec![c; 1]).collect::<Vec<_>>())
        .unwrap();
    assert!(
        max_abs(&fsq, &floats(&r["fsq_first16"])) < 1e-6,
        "FSQ codebook differs"
    );
}
