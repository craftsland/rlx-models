# rlx-voice-link-nrf

nRF52840 firmware for [`rlx-voice-link`](../rlx-voice-link): PDM microphone
→ IMA-ADPCM → BLE GATT notifications. Embassy, TrouBLE (BLE host) and
Nordic's SoftDevice Controller (`nrf-sdc`), all from crates.io.

| board | feature | mic | start/stop |
|---|---|---|---|
| Seeed XIAO nRF52840 Sense | `board-xiao` (default) | PDM CLK P1.00, DIN P0.16, power P1.10 | button D1 (P0.03 → GND) or the phone |
| Arduino Nano 33 BLE Sense | `board-nano33` | PDM CLK P0.26, DIN P0.25, power P0.17 | the phone (no user button) |

```bash
cd crates/rlx-voice-link-nrf
cargo run --release                                        # flash + RTT logs (probe-rs, SWD)
cargo run --release --no-default-features --features board-nano33
```

Flashing goes over SWD and overwrites the factory UF2 bootloader; the image
starts at 0x0 and needs no SoftDevice (`memory.x`).

* PDM 1.280 MHz / ratio 80 → 16 kHz, +10 dB gain, 16 ms frames.
* After connecting it requests 2M PHY and 251-byte data length; the
  connection interval is left to the central (the peripheral-only controller
  library has no LE Connection Update).
* Packet size follows the negotiated ATT MTU, up to 476 samples.
* Frames are dropped, never blocked on, if the radio falls behind; the
  receiver fills the gap with silence.

Size: 136 KB flash, ~32 KB RAM + a 16 KB controller pool. The pool size is
checked by the controller at startup (`build_sdc` fails if too small). Built
and linted here; **not yet run on a board**.
