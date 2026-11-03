# rlx-voice-link-ffi

C API (`include/rlx_voice_link.h`) and Swift package (`swift/`) for
[`rlx-voice-link`](../rlx-voice-link) on iOS and macOS:

* `VoiceLinkCentral` — CoreBluetooth central for `rlx-voice-link-nrf`,
  push-to-talk via `startTalking()` / `stopTalking()`;
* `UtteranceReceiver` — notifications → utterance PCM;
* `SpeechRecognizer` — on-device int8 Conformer-CTC from the `.rcc` model;
* `OpenAIChatModel` / `ChatModel` — hand the transcript to an LM.

```bash
crates/rlx-voice-link-ffi/scripts/build-xcframework.sh     # IOS=0 for macOS only
cd crates/rlx-voice-link-ffi/swift
RLX_RCC=…/small-int8.rcc RLX_WAV=…/jfk_16k.wav swift test
```

Verified: `swift test` on macOS and `xcodebuild test` on an iPhone 17 Pro
simulator both transcribe the JFK clip correctly (133 ms / 533 ms); the
package builds for iOS devices. An iOS app needs
`NSBluetoothAlwaysUsageDescription` in Info.plist, and the
`bluetooth-central` background mode to stream in the background.
