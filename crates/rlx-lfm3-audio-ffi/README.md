# rlx-lfm3-audio-ffi

C ABI over [`rlx-lfm3-audio`](../rlx-lfm3-audio/README.md) (LiquidAI
LFM3-Audio-2B): open a model, transcribe 16 kHz audio, synthesize 24 kHz
speech. Built as a static library for Swift / iOS apps; `ios/` is a SwiftUI
app on it. Not published to crates.io (`publish = false`).

## C API

Declared in [`include/rlx_lfm3_audio.h`](include/rlx_lfm3_audio.h):

| function | does |
|---|---|
| `lfm3a_open(lm, mmproj, device, low_memory)` | load the LM GGUF + mmproj GGUF; `device` `"metal"` (NULL), `"cpu"`, `"coreml"`, … |
| `lfm3a_transcribe(m, pcm, n, language, max_tokens)` | 16 kHz mono → text; `language` `"English"` or NULL (model picks); `max_tokens` 0 = 256 |
| `lfm3a_speak(m, text, language, voice, seed, &len)` | text → 24 kHz mono; `language` `"en"`, `voice` `"female"` / `"male"` |
| `lfm3a_free_string`, `lfm3a_free_audio(pcm, len)` | free what the two calls above return |
| `lfm3a_close(m)` | free the model |
| `lfm3a_last_error()` | message of the last failed call on this thread |
| `lfm3a_footprint()` | process physical footprint in bytes (what iOS's memory limit counts) |

Failing calls return NULL and set `lfm3a_last_error`; panics are caught at the
boundary. A handle is not thread-safe: use it from one thread (the app runs
it on a serial queue). `low_memory` keeps one LM graph and drops the encoder
and codec graphs after use — see the memory section of the model README.

## Building the library

```sh
cargo build -p rlx-lfm3-audio-ffi --release --features metal --target aarch64-apple-ios
cargo build -p rlx-lfm3-audio-ffi --release --features metal --target aarch64-apple-ios-sim
```

Set `IPHONEOS_DEPLOYMENT_TARGET` (17.0 here) to match the app. A static
archive carries no link directives, so the app links `-lrlx_lfm3_audio_ffi`
plus `Accelerate`, `Metal`, `MetalPerformanceShaders`,
`MetalPerformanceShadersGraph`, `-lresolv` and `-lc++` (`ios/project.yml`).
The `coreml` feature adds the CoreML backend.

## The iOS app

`ios/` (needs [XcodeGen](https://github.com/yonaskolb/XcodeGen)): Load,
Transcribe (bundled JFK clip), Speak. `ios/run.sh` builds the library and the
app, installs it, copies the Q4_K_M LM + Q8_0 mmproj into the app's Documents
and launches it with `--selftest` (ASR on the JFK clip, TTS, the TTS
transcribed back; timings and footprint on the console):

```sh
just fetch-lfm3-audio Q4_K_M Q8_0                 # the low-memory pair, ~1.4 GB
just lfm3-audio-ios-sim                           # booted simulator
just lfm3-audio-ios <device-id> <team-id>         # device from `xcrun devicectl list devices`, unlocked
```

`run.sh` settings: `WEIGHTS` (directory of the GGUFs), `LM`, `MMPROJ` (file
names), `TEAM` (signing team), `BUNDLE_ID` (default `dev.rlx.lfm3audio`; pick
one your team can sign). Signing is automatic; the app needs no extra
entitlements at its ~2.2 GiB peak.

Outside the self-test, copy the two GGUFs into the app's Documents through
the Files app or Finder; the app picks the Q4_K_M LM and Q8_0 mmproj when
several are present.

Status: on the iOS simulator the JFK transcript is correct, the spoken
sentence transcribes back word for word and the footprint stays ≤ 1.7 GiB;
timings vary between simulator runs (ASR 3–9 s). Not yet run on a device.
