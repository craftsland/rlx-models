# rlx-voice-link-host

Mac/Linux receiver for [`rlx-voice-link`](../rlx-voice-link): BLE
(btleplug / CoreBluetooth) → utterance PCM → int8 Conformer-CTC
([`rlx-conformer-ctc-core`](../rlx-conformer-ctc-core)) → optionally an
OpenAI-compatible LM endpoint such as `rlx-openai`.

```bash
rlx-voice-link listen   --rcc small-int8.rcc [--lm-url URL --lm-model NAME --system TEXT]
rlx-voice-link simulate --rcc small-int8.rcc --wav clip.wav [--drop-every N] [--att 244]
cargo run -p rlx-voice-link-host --release --example voice_link_demo -- [--wav F]… [--loss P] [--lm-url URL]
cargo run -p rlx-voice-link-host --release --example link_wer -- \
    --rcc small-int8.rcc --list DIR/list.tsv [--drop-every N]
```

`listen`: Enter toggles recording on the device (push-to-talk from the
Mac); the device button works too. The terminal app needs Bluetooth
permission the first time. `simulate` runs the full chain without a radio.
Build without the radio with `--no-default-features` (drops `ble`).

`examples/voice_link_demo.rs` is the end-to-end demo without hardware
(`just voice-link-demo`), documented in
[`rlx-voice-link`](../rlx-voice-link/README.md#demo-no-hardware).
