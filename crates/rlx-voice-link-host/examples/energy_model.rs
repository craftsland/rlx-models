// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Average device current for a wearable voice interface, five ways:
//!
//! 1. stream all audio to the phone;
//! 2. energy gate + phone confirmation (what `rlx-voice-link` does);
//! 3. the same with a wake-on-sound microphone (VM3011-class), so the PDM
//!    clock and the gate run only after the mic hears something;
//! 4. ASR on the device's Cortex-M33 with DSP (Citrinet, nRF54H20-class);
//! 5. ASR on a Cortex-M55 + Ethos-U55 (Alif Ensemble-class).
//!
//! Inputs: the duty cycle `wearable_sim --json` measured (how long the radio
//! carried audio, how much of it was speech), the per-frame instruction counts
//! from `rlx-voice-link-bench` and the nRF54H20 firmware, and component
//! figures. Each figure prints with its source; `assumed` marks the ones no
//! datasheet gave, with the range to try. The day is rescaled to `--speech`
//! (fraction of time someone talks to the device) and `--noise` (fraction of
//! time loud non-speech sound is around), keeping the simulation's per-second
//! rates.
//!
//! ```text
//! cargo run -p rlx-voice-link-host --release --example energy_model -- \
//!     --duty duty.json [--speech 0.05] [--noise 0.10] [--battery-mah 100]
//! ```

use anyhow::{Context, Result};

/// One component figure and where it came from.
struct Fig {
    value: f64,
    unit: &'static str,
    what: &'static str,
    source: &'static str,
}

fn fig(value: f64, unit: &'static str, what: &'static str, source: &'static str) -> Fig {
    Fig {
        value,
        unit,
        what,
        source,
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let num = |k: &str, d: f64| -> Result<f64> {
        Ok(get(k).map(|v| v.parse()).transpose()?.unwrap_or(d))
    };
    let duty: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(get("--duty").context("--duty")?)?)?;
    let d = |k: &str| duty[k].as_f64().unwrap_or(0.0);
    let speech_frac = num("--speech", 0.05)?;
    let noise_frac = num("--noise", 0.10)?;
    let battery = num("--battery-mah", 100.0)?;

    // ---- component figures (3.0 V supply) ----
    let v = 3.0;
    let cpu_ua_mhz = fig(
        52.0,
        "µA/MHz",
        "nRF52840 CPU from flash, DC/DC",
        "nRF52840 PS v1.8",
    );
    let tx_ma = fig(
        5.3,
        "mA",
        "nRF52840 radio TX 0 dBm, DC/DC",
        "nRF52840 PS v1.8",
    );
    let rx_ma = fig(6.4, "mA", "nRF52840 radio RX, DC/DC", "nRF52840 PS v1.8");
    let sleep_ua = fig(
        3.0,
        "µA",
        "nRF52840 System ON, RTC, RAM retained",
        "nRF52840 PS v1.8 (~1.5–3.2)",
    );
    let mic_ma = fig(
        0.6,
        "mA",
        "PDM MEMS mic, normal mode (nRF52840 PDM clock ≥1 MHz)",
        "assumed (0.5–1.0); SPH0641LU4H-1 low-power mode is 0.235 mA but needs ≤815 kHz",
    );
    let zpl_ua = fig(
        10.0,
        "µA",
        "VM3011 zero-power listening (clock off)",
        "Vesper VM3011 datasheet",
    );
    let wake_instr = fig(
        3000.0,
        "instr",
        "per 16 ms frame: PDM IRQ, executor wake, queue",
        "assumed (1–5 k)",
    );
    let gate_instr = fig(
        1242.0,
        "instr",
        "gate per frame, Cortex-M4F",
        "measured, rlx-voice-link-bench",
    );
    let adpcm_instr = fig(
        15825.0,
        "instr",
        "ADPCM per frame while sending",
        "measured, rlx-voice-link-bench",
    );
    let pkt_tx_ms = fig(
        1.09,
        "ms",
        "TX per 244-byte notification, 2M PHY (262 B on air + ramp-up)",
        "BLE 5 2M PHY timing",
    );
    let pkt_rx_ms = fig(
        0.25,
        "ms",
        "RX per packet: ack + T_IFS listening",
        "BLE timing, assumed margin",
    );
    let evt_ua_s = fig(
        0.25,
        "µA·s",
        "charge per idle connection event (radio + HFXO + CPU)",
        "assumed (0.15–0.4), Nordic online power profiler class",
    );
    let conn_ms = fig(
        30.0,
        "ms",
        "connection interval chosen by iOS/macOS",
        "Apple accessory design guidelines (15–30 ms)",
    );
    let asr_instr_s = fig(
        2.587e9 / 5.2,
        "instr/s",
        "Citrinet int8 on Cortex-M33 + DSP",
        "measured, rlx-citrinet-nrf54h (QEMU)",
    );
    let m33_ua_mhz = fig(
        30.0,
        "µA/MHz",
        "Cortex-M33 class core, 22 nm",
        "assumed (20–52): Nordic publishes no nRF54H20 µA/MHz",
    );
    let npu_gain = fig(
        33.0,
        "×",
        "energy per inference, Ethos-U55 vs Cortex-M55 alone",
        "Alif Ensemble E3, keyword spotting (vendor benchmark)",
    );

    // ---- rates from the simulation ----
    let sim_s = d("seconds");
    let sim_speech = d("speech_seconds");
    let sim_streamed = d("streamed_seconds");
    let sim_stream_speech = d("streamed_speech_seconds");
    let sim_nonspeech = sim_s - sim_speech;
    // Radio time per second of speech, and false-trigger radio time per
    // second of non-speech, as the simulation measured them. The simulation
    // is a noisy room (distractors in ~30% of its non-speech time); scale the
    // false-trigger rate to `--noise`.
    let stream_per_speech = sim_stream_speech / sim_speech;
    let sim_noise_frac = 0.30;
    let false_per_nonspeech =
        (sim_streamed - sim_stream_speech) / sim_nonspeech * (noise_frac / sim_noise_frac);
    let streamed = speech_frac * stream_per_speech + (1.0 - speech_frac) * false_per_nonspeech;
    let bytes_per_streamed_s = d("bytes") / sim_streamed;
    let pkts_per_streamed_s = d("packets") / sim_streamed;

    let frames_s = 16_000.0 / 256.0;
    let cpu_ma_per_instr_s = cpu_ua_mhz.value / 1e3 / 1e6; // mA per (instr/s), 1 IPC
    let radio_ma_streaming =
        pkts_per_streamed_s * (pkt_tx_ms.value * tx_ma.value + pkt_rx_ms.value * rx_ma.value) / 1e3;
    let conn_idle_ma = evt_ua_s.value / 1e3 / (conn_ms.value / 1e3);
    let sleep_ma = sleep_ua.value / 1e3;

    // 1. stream everything.
    let s1 = mic_ma.value
        + sleep_ma
        + cpu_ma_per_instr_s * frames_s * (wake_instr.value + adpcm_instr.value)
        + radio_ma_streaming
        + conn_idle_ma;
    // 2. gate + phone confirm: mic and gate all day, ADPCM + radio while
    //    streaming.
    let s2 = mic_ma.value
        + sleep_ma
        + cpu_ma_per_instr_s * frames_s * (wake_instr.value + gate_instr.value)
        + streamed * (cpu_ma_per_instr_s * frames_s * adpcm_instr.value + radio_ma_streaming)
        + conn_idle_ma;
    // 3. wake-on-sound mic: PDM and gate only while there is sound (speech,
    //    noise, plus 1 s after each).
    let awake = (speech_frac + noise_frac).min(1.0) * 1.1;
    let s3 = zpl_ua.value / 1e3 * (1.0 - awake)
        + awake
            * (mic_ma.value
                + cpu_ma_per_instr_s * frames_s * (wake_instr.value + gate_instr.value))
        + sleep_ma
        + streamed * (cpu_ma_per_instr_s * frames_s * adpcm_instr.value + radio_ma_streaming)
        + conn_idle_ma;
    // 4. ASR on the M33 during kept audio (gate as in 3 decides when), text
    //    only over the radio.
    let m33_ma_per_instr_s = m33_ua_mhz.value / 1e3 / 1e6;
    let asr_ma = asr_instr_s.value * m33_ma_per_instr_s;
    let s4 = s3 - streamed * radio_ma_streaming + streamed * asr_ma;
    // 5. ASR on an Ethos-U55: the same work at 1/33 the energy.
    let s5 = s3 - streamed * radio_ma_streaming + streamed * asr_ma / npu_gain.value;

    println!("component figures (3.0 V):");
    for f in [
        &cpu_ua_mhz,
        &tx_ma,
        &rx_ma,
        &sleep_ua,
        &mic_ma,
        &zpl_ua,
        &wake_instr,
        &gate_instr,
        &adpcm_instr,
        &pkt_tx_ms,
        &pkt_rx_ms,
        &evt_ua_s,
        &conn_ms,
        &asr_instr_s,
        &m33_ua_mhz,
        &npu_gain,
    ] {
        let (val, unit) = if f.value >= 1e6 {
            (f.value / 1e6, format!("M {}", f.unit))
        } else {
            (f.value, f.unit.to_string())
        };
        println!("  {val:>9.3} {unit:<10} {:<62} [{}]", f.what, f.source);
    }
    println!(
        "\nsimulation: radio carries audio {:.2} s per s of speech, {:.3} s per s of noisy non-speech; \
         {:.0} packets/s, {:.1} kbit/s while streaming",
        stream_per_speech,
        (sim_streamed - sim_stream_speech) / sim_nonspeech,
        pkts_per_streamed_s,
        bytes_per_streamed_s * 8.0 / 1e3
    );
    println!(
        "day: speech {:.0}%, loud non-speech {:.0}% → radio carries audio {:.1}% of the time\n",
        100.0 * speech_frac,
        100.0 * noise_frac,
        100.0 * streamed
    );
    println!(
        "streaming costs {:.2} mA on air; ASR on the M33 {:.1} mA ({:.0} MHz of work); idle connection {:.0} µA\n",
        radio_ma_streaming + cpu_ma_per_instr_s * frames_s * adpcm_instr.value,
        asr_ma,
        asr_instr_s.value / 1e6,
        conn_idle_ma * 1e3
    );
    println!("{:<58} {:>9} {:>12}", "scenario", "avg mA", "hours/battery");
    for (name, ma) in [
        ("1. stream everything to the phone", s1),
        (
            "2. energy gate + phone TEN-VAD confirm (rlx-voice-link)",
            s2,
        ),
        ("3. 2 + wake-on-sound mic (VM3011-class)", s3),
        ("4. 3 but ASR on the device's Cortex-M33 + DSP", s4),
        ("5. 3 but ASR on a Cortex-M55 + Ethos-U55", s5),
    ] {
        println!("{name:<58} {ma:>9.3} {:>12.0}", battery / ma);
    }
    println!(
        "\n(battery {battery:.0} mAh; {:.1} V; ASR scenarios still need the phone for an LM; \
         average current, so peaks such as ASR at {:.0} MHz must fit the core)",
        v,
        asr_instr_s.value / 1e6
    );
    Ok(())
}
