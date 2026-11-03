# rlx-voice-link

Speech from a microphone MCU to a phone or Mac over BLE, where speech
recognition and a local language model run.

```text
nRF52840 + PDM mic ──BLE (ADPCM, 66 kbit/s)──▶ iPhone / Mac
  rlx-voice-link-nrf                              rlx-voice-link-host (Rust CLI)
  (Embassy + TrouBLE)                             RlxVoiceLink (Swift, iOS/macOS)
                                                    → int8 Conformer-CTC ASR
                                                    → LM (rlx-openai, or in-app)
```

| crate | runs on | what |
|---|---|---|
| `rlx-voice-link` | everywhere, `no_std` | IMA-ADPCM, packet framing, `Packetizer` / `Reassembler`, GATT UUIDs |
| [`rlx-voice-link-nrf`](../rlx-voice-link-nrf) | nRF52840 | PDM → voice gate → ADPCM → GATT notifications; hands-free, push-to-talk button as override |
| [`rlx-voice-link-host`](../rlx-voice-link-host) | Mac / Linux | `rlx-voice-link listen` (btleplug) and `simulate` (no radio) |
| [`rlx-voice-link-ffi`](../rlx-voice-link-ffi) | iOS / macOS | C API + Swift package: CoreBluetooth central, receiver, ASR |

The ASR is [`rlx-conformer-ctc-core`](../rlx-conformer-ctc-core): the same
13.9 MB int8 `.rcc` and the same code on the Mac, the iPhone and (for a
device that has to transcribe on its own) an ESP32-class MCU.

## Demo (no hardware)

[`voice_link_demo`](../rlx-voice-link-host/examples/voice_link_demo.rs) runs
the whole path on one machine. A simulated nRF52840 encodes a clip packet by
packet, a lossy radio drops packets at random, the phone side reassembles
and transcribes, and an optional local LM answers. It calls the same
`Packetizer` / `Reassembler` the firmware and the phone use, so it is also
the shortest reference for the API.

```bash
just fetch-conformer-ctc && just export-conformer-ctc-mcu   # ASR model, once
just voice-link-demo                                         # three JFK clips, 3% loss
```

With no `--wav`, the demo sends the three JFK clips in `assets/` as one
conversation: the 1961 inaugural line, a voice-cloned copy of it, and a
cloned "Rust" speech.

```text
voice link demo — device: simulated nRF52840 · radio: 3.0% loss, 476 samples/packet (29.8 ms) · phone: int8 Conformer-CTC · LM: none (pass --lm-url)

[1] assets/harness/jfk_16k.wav (5.20 s)
  device → phone   175 packets, 5 dropped, 42.6 kB (65.6 kbit/s)
  phone ASR        106 ms (49× faster than real time)
  heard            "ask not what your country can do for you ask what you can do for your country"
  (same words as the original audio)

[2] assets/jfk/jfk_voice_clone.wav (5.20 s)
  device → phone   175 packets, 4 dropped, 42.6 kB (65.6 kbit/s)
  phone ASR        94 ms (56× faster than real time)
  heard            "ask not what your country can do for you ask what you can do for your country"
  (same words as the original audio)

[3] assets/jfk/jfk_rust_speech.wav (12.08 s)
  device → phone   407 packets, 17 dropped, 99.1 kB (65.6 kbit/s)
  phone ASR        273 ms (44× faster than real time)
  heard            "my fellow americans to day we choose to write our software in rust not because it is easy but because it is fast safe and free of garbage collection"
  (same words as the original audio)

22.5 s of speech, ASR 0.47 s total, 26 packets lost on the link
```

Each utterance is also transcribed from the original audio. When the link
changed the words, the demo prints both versions; otherwise it prints
"same words".

**Heavier loss, paced like live speech:**

```bash
just voice-link-demo --wav assets/jfk/jfk_rust_speech.wav --loss 15 --seed 7 --realtime
```

At 15% loss the 12 s Rust clip starts to lose words (68 of 407 packets lost,
13.8% WER: "rus", and a dropped "but"). The 5 s "ask not" clips stay
word-exact even at 50%; see [Measured](#measured).

**With a language model.** Serve any checkpoint `rlx-openai` handles in one
terminal, then point the demo at it:

```bash
# terminal 1 — Qwen3-0.6B on Metal
just features=metal openai-serve -- --port 8080 \
    --engine qwen3 --weights /path/to/qwen3-0.6b --model-id qwen3 --device metal
# terminal 2
just voice-link-demo --lm-url http://127.0.0.1:8080/v1/chat/completions --lm-model qwen3
```

```text
[3] assets/jfk/jfk_rust_speech.wav (12.08 s)
  device → phone   407 packets, 17 dropped, 99.1 kB (65.6 kbit/s)
  phone ASR        408 ms (30× faster than real time)
  heard            "my fellow americans to day we choose to write our software in rust not because it is easy but because it is fast safe and free of garbage collection"
  (same words as the original audio)
  LM (173.8 s)       Your statement highlights the value of Rust's features like speed, safety, and free of garbage collection, which make it a strong choice for modern software development.
```

That run used Qwen3-0.6B on a heavily loaded machine (load average 40–90),
so the LM and ASR times are not representative. To the two "ask not" lines
the 0.6B model replied with a generic "Your message is a thoughtful
exchange…"; a larger checkpoint answers better. The demo's system prompt
asks for one or two short sentences; change it with `--system`.

| flag | default | meaning |
|---|---|---|
| `--wav F` | the three JFK clips | clip to send; repeat for several turns (any WAV, resampled to 16 kHz) |
| `--rcc F` | `.cache/conformer-ctc-mcu/small-int8.rcc` | ASR model from `just export-conformer-ctc-mcu` |
| `--loss P` | `3` | percent of audio packets dropped at random |
| `--seed N` | `1` | loss pattern; the same seed gives the same drops |
| `--att N` | `244` | ATT payload bytes (MTU − 3); `20` simulates a link that never exchanged MTU (2,972 packets of 28 samples for the first JFK clip, 91 kbit/s: headers dominate) |
| `--realtime` | off | send packets at the pace the mic produces them |
| `--lm-url`, `--lm-model`, `--system` | — | OpenAI-compatible chat endpoint, model id, system prompt |

Any packet can be lost except the END, so every utterance closes. A lost
START is recovered: the receiver opens the utterance at the next packet,
and the packet index tells it how much silence to put in front. On a real
link the BLE layer retransmits lost packets. When the radio falls behind,
the firmware drops audio frames, which is what `--loss` models.

**From the demo to hardware.** Swap the simulated device for
`just voice-link-nrf-flash` and the simulated phone for
`just voice-link-listen`, which takes the same `--lm-url` flags, or for the
Swift package on an iPhone.

## Wire format

One GATT notification per packet on `7a1e0002-…` (service `7a1e0001-…`):
a 6-byte header (version and START/END flags, the packet's index within the
utterance, the ADPCM predictor and step index the packet starts from) and 4-bit codes, two
samples per byte. At ATT MTU 247 a packet carries 476 samples (29.75 ms).

Each packet carries the codec state it starts from, so it decodes alone: a
lost packet costs its own 30 ms, filled with silence so timing is kept, and
nothing after it. The index restarts at 0 with every utterance, so even a
lost START packet is replaced by the right amount of silence. Writes of `1` / `2` to `7a1e0003-…` start and stop an
utterance (push-to-talk, the same as the device button); `3` (`CMD_REJECT`)
tells the voice gate that what it opened is not speech.

## Low power: when the radio is on

A wearable cannot stream all day, and it cannot afford a real voice detector
either: TEN-VAD costs 1.02 M instructions per 16 ms frame on a Cortex-M4F,
about all of an nRF52840. So the work is split:

1. **Device: `gate::Gate`.** An integer sum of squares per frame against a
   noise floor that adapts only while closed. It opens on 3 loud frames in a
   row (48 ms, so typing and taps do not open it) and sends 304 ms of pre-roll
   first, so the first syllable survives. It closes after 400 ms quiet, after
   30 s, or on a reject. 1,242 instructions per frame on the M4F (0.12% of the
   core).
2. **Phone: `confirm::Confirmer`** (`rlx-voice-link-host`). TEN-VAD on the
   arriving audio. It keeps the utterance after 5 speech frames, or writes
   `CMD_REJECT` if 50 frames pass without them. On a reject the gate backs off
   (4× threshold for 10 s) and takes the rejected sound's quietest frame as its
   new floor, so a fan or a TV stops reopening it.
3. **Phone, before ASR:** pieces under 1 s apart are joined (the gate splits
   sentences at long pauses), and transcripts that are only fillers are dropped.

`examples/wearable_sim.rs` (`just voice-link-wearable-sim`): 24 min of room
noise with 30 LibriSpeech utterances and loud non-speech (door slams, typing,
music, two 30 s fan runs), every frame through the real gate, packetizer,
reassembler, confirmer and int8 ASR. Seeds 1–3:

| | |
|---|---|
| speech in the audio | 241.6 s (16.6%) |
| radio carrying audio | 19–20% of the time; 237.5 s of speech, 52 s not |
| not speech on air | music 32 s (TEN-VAD scores the tones as voice), door 3.2 s, fan 1.1 s, typing 0 s |
| speech kept / utterances missed | 98% / 0 of 30 |
| WER, gated | 2.39–2.54% |
| WER, ASR on each clean utterance | 1.79% |

On the M4F the device's work per 16 ms frame is 1,242 instructions for the
gate (all day) and 15,825 for ADPCM (only while sending), measured by
`rlx-voice-link-bench` (`just voice-link-bench`).

`examples/energy_model.rs` (`just voice-link-energy`) turns that duty cycle
and published component figures into average current; every figure prints with
its source, and `assumed` marks the ones no datasheet gave. A day with speech 5% of
the time and loud non-speech 10%, 100 mAh battery:

| scenario | avg current | hours |
|---|---|---|
| stream everything | 0.92 mA | 109 |
| gate + phone confirm (this crate) | 0.64 mA | 155 |
| + wake-on-sound mic (VM3011-class, 10 µA listening) | 0.14 mA | 715 |
| ASR on the device's Cortex-M33 + DSP instead of streaming | 1.06 mA | 94 |
| ASR on a Cortex-M55 + Ethos-U55 instead of streaming | 0.15 mA | 655 |

With the gate, the microphone dominates: a PDM mic in normal mode draws about
0.6 mA (assumed, 0.5–1.0), while the radio's share of a 5%-speech day is
about 20 µA. A microphone that listens with its clock off is the next 4×.
On-device ASR only pays with an NPU, and then it about breaks even with
streaming; the phone still runs the LM.

## Measured

`rlx-voice-link simulate` and `examples/link_wer.rs` in `rlx-voice-link-host`
run the whole chain except the radio. LibriSpeech dev-clean, 500 utterances /
10,103 words, int8 ASR:

| input | WER |
|---|---|
| original 16-bit PCM | 3.60% |
| over the link (ADPCM, no loss) | 3.76% |
| over the link, 1 packet in 20 lost | 3.75% |

The JFK clips, from `just test-voice-link`
(`rlx-voice-link-host/tests/link.rs`), random loss, seed 7:

| loss | `jfk_16k.wav` (5.2 s) | `jfk_voice_clone.wav` (5.2 s) | `jfk_rust_speech.wav` (12.1 s) |
|---|---|---|---|
| 0% / 5% | exact | exact | exact |
| 15% | exact (31/175 lost) | exact | 13.8% WER (68/407 lost) |
| 30% | exact (58/175) | exact | 6.9% WER (129/407) |
| 50% | exact (83/175) | exact | 34.5% WER (197/407) |

At 0% and 5% loss the tests require the exact transcript (two seeds at 5%).
At 15%, 30% and 50% they require WER at most 20%, 40% and 70%. They also
check that every utterance keeps its full length. WER is not monotonic in
the loss rate, because which words lose packets matters more than how many
packets are lost.

Not yet measured on hardware: the nRF52840 firmware builds (137 KB flash,
58 KB RAM including the gate's pre-roll and a 32-frame queue) but has not been
flashed in this repo, so real-link throughput, loss and current on iOS/macOS
are unverified.

## Run

```bash
just export-conformer-ctc-mcu                      # the ASR model, once
just voice-link-simulate assets/harness/jfk_16k.wav
just voice-link-nrf-flash                          # XIAO nRF52840 Sense over SWD (probe-rs)
just voice-link-listen                             # Mac: Enter = push-to-talk
just voice-link-listen --lm-url http://127.0.0.1:8080/v1/chat/completions
just voice-link-xcframework                        # Swift package binary (iOS + macOS)
```
