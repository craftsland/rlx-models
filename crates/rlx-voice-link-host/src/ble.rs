// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! The radio: find the device by its service UUID, subscribe to audio
//! notifications, reassemble utterances. btleplug uses CoreBluetooth on
//! macOS (the terminal app needs Bluetooth permission the first time).

use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use btleplug::api::{Central, Manager as _, Peripheral as _, ScanFilter, WriteType};
use btleplug::platform::{Manager, Peripheral};
use futures::StreamExt;
use rlx_voice_link::{
    AUDIO_CHAR_UUID, CMD_START, CMD_STOP, CONTROL_CHAR_UUID, Event, Reassembler, SERVICE_UUID,
};
use uuid::Uuid;

/// Longest utterance kept: 30 s at 16 kHz.
pub const MAX_SAMPLES: usize = 30 * 16_000;

fn uuid(s: &str) -> Uuid {
    Uuid::parse_str(s).unwrap_or_else(|_| unreachable!("constant UUID"))
}

async fn find(timeout: Duration) -> Result<Peripheral> {
    let manager = Manager::new().await?;
    let adapter = manager
        .adapters()
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no Bluetooth adapter"))?;
    let service = uuid(SERVICE_UUID);
    adapter
        .start_scan(ScanFilter {
            services: vec![service],
        })
        .await?;
    let deadline = tokio::time::Instant::now() + timeout;
    while tokio::time::Instant::now() < deadline {
        for p in adapter.peripherals().await? {
            if let Some(props) = p.properties().await?
                && props.services.contains(&service)
            {
                adapter.stop_scan().await.ok();
                eprintln!(
                    "[voice-link] found {} ({})",
                    props.local_name.unwrap_or_default(),
                    p.address()
                );
                return Ok(p);
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    bail!("no device advertising {SERVICE_UUID} within {timeout:?}")
}

/// Connect and call `on_utterance` for every completed utterance until the
/// device disconnects. Pressing Enter on stdin toggles start/stop on the
/// device (push-to-talk from the Mac); the device button works too.
pub async fn listen(mut on_utterance: impl FnMut(&[i16], u32)) -> Result<()> {
    let p = find(Duration::from_secs(20)).await?;
    p.connect().await.context("connect")?;
    p.discover_services().await?;
    let chars = p.characteristics();
    let audio = chars
        .iter()
        .find(|c| c.uuid == uuid(AUDIO_CHAR_UUID))
        .cloned()
        .ok_or_else(|| anyhow!("device has no audio characteristic"))?;
    let control = chars
        .iter()
        .find(|c| c.uuid == uuid(CONTROL_CHAR_UUID))
        .cloned();
    p.subscribe(&audio).await?;
    eprintln!("[voice-link] connected; press Enter to start/stop, or use the device button");

    // Enter on stdin → toggle, forwarded to the async side.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    std::thread::spawn(move || {
        let mut line = String::new();
        while std::io::stdin().read_line(&mut line).is_ok_and(|n| n > 0) {
            if tx.send(()).is_err() {
                break;
            }
            line.clear();
        }
    });

    let mut buf = vec![0i16; MAX_SAMPLES];
    let mut rx_audio = Reassembler::new(&mut buf);
    let mut notes = p.notifications().await?;
    let mut talking = false;
    loop {
        tokio::select! {
            n = notes.next() => {
                let Some(n) = n else { break };
                if n.uuid != audio.uuid {
                    continue;
                }
                if let Event::Utterance(_) = rx_audio.push(&n.value) {
                    talking = false;
                    on_utterance(rx_audio.utterance(), rx_audio.lost());
                }
            }
            Some(()) = rx.recv() => {
                if let Some(c) = &control {
                    talking = !talking;
                    let cmd = if talking { CMD_START } else { CMD_STOP };
                    p.write(c, &[cmd], WriteType::WithResponse).await?;
                    eprintln!("[voice-link] {}", if talking { "listening…" } else { "stop" });
                }
            }
        }
    }
    eprintln!("[voice-link] disconnected");
    Ok(())
}
