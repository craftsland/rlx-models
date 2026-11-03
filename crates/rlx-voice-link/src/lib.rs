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

//! A voice link over BLE: a microphone MCU (nRF52840) captures speech and
//! streams it to a phone or Mac, which runs ASR and a language model.
//!
//! The same `no_std` code runs on both ends:
//!
//! * [`Packetizer`] (MCU) — 16 kHz i16 PCM → IMA-ADPCM packets sized for one
//!   GATT notification;
//! * [`Reassembler`] (phone / Mac) — packets → one utterance of PCM, with lost
//!   packets filled with silence of the right length.
//!
//! # Wire format
//!
//! One packet is one GATT notification on [`AUDIO_CHAR_UUID`]:
//!
//! | byte | field |
//! |---|---|
//! | 0 | `version << 4 \| flags` ([`FLAG_START`], [`FLAG_END`]) |
//! | 1 | packet index within the utterance (START = 0), wrapping |
//! | 2–3 | ADPCM predictor at packet start, i16 LE |
//! | 4 | ADPCM step index at packet start |
//! | 5 | reserved, 0 |
//! | 6.. | 4-bit codes, two samples per byte, low nibble first |
//!
//! Every packet carries the codec state it starts from, so it decodes on its
//! own; a lost packet costs its ~30 ms and nothing after it. 16 kHz × 4 bit
//! = 64 kbit/s plus headers, well inside BLE 2M PHY throughput on iOS/macOS.
//!
//! Commands from the phone go to [`CONTROL_CHAR_UUID`] (write): [`CMD_START`]
//! / [`CMD_STOP`] open and close an utterance, the same as the device button;
//! [`CMD_REJECT`] tells the [`gate`] that what it opened is not speech.
//!
//! # Low power
//!
//! The device does not stream all day. [`gate::Gate`] opens an utterance on
//! anything louder than the room (integer energy, about 1,000 instructions per
//! 16 ms frame) and the phone confirms with a real voice detector, rejecting
//! the rest. The radio is on for speech plus a little noise; see
//! `rlx-voice-link-host`'s `wearable_sim` example for the duty cycle.

#![cfg_attr(not(test), no_std)]

pub mod adpcm;
pub mod gate;

/// Wire-format version in the high nibble of byte 0.
pub const VERSION: u8 = 1;
/// First packet of an utterance.
pub const FLAG_START: u8 = 1;
/// Last packet of an utterance (may carry no samples).
pub const FLAG_END: u8 = 2;
/// Header bytes before the ADPCM payload.
pub const HEADER: usize = 6;
/// PCM sample rate on the link.
pub const SAMPLE_RATE: u32 = 16_000;

/// GATT service: `rlx voice link`.
pub const SERVICE_UUID: &str = "7a1e0001-5c2b-4e8a-9f43-72c0b1d6a5e3";
/// Notify: audio packets, device → phone.
pub const AUDIO_CHAR_UUID: &str = "7a1e0002-5c2b-4e8a-9f43-72c0b1d6a5e3";
/// Write: commands, phone → device.
pub const CONTROL_CHAR_UUID: &str = "7a1e0003-5c2b-4e8a-9f43-72c0b1d6a5e3";
/// The service UUID as little-endian bytes, the order BLE stacks take.
pub const SERVICE_UUID_LE: [u8; 16] = uuid_le(0x0001);
/// [`AUDIO_CHAR_UUID`] as little-endian bytes.
pub const AUDIO_CHAR_UUID_LE: [u8; 16] = uuid_le(0x0002);
/// [`CONTROL_CHAR_UUID`] as little-endian bytes.
pub const CONTROL_CHAR_UUID_LE: [u8; 16] = uuid_le(0x0003);
/// Command: begin an utterance.
pub const CMD_START: u8 = 1;
/// Command: end the utterance.
pub const CMD_STOP: u8 = 2;
/// Phone → device: the open utterance is not speech. The device closes it
/// and raises its voice-gate threshold for a while ([`gate::Gate::reject`]).
pub const CMD_REJECT: u8 = 3;

const fn uuid_le(short: u16) -> [u8; 16] {
    // 7a1eXXXX-5c2b-4e8a-9f43-72c0b1d6a5e3, reversed.
    let be: [u8; 16] = [
        0x7a,
        0x1e,
        (short >> 8) as u8,
        short as u8,
        0x5c,
        0x2b,
        0x4e,
        0x8a,
        0x9f,
        0x43,
        0x72,
        0xc0,
        0xb1,
        0xd6,
        0xa5,
        0xe3,
    ];
    let mut le = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        le[i] = be[15 - i];
        i += 1;
    }
    le
}

/// PCM samples that fit one packet for a given ATT payload size
/// (`MTU - 3`). Even, so a packet never splits a byte.
pub const fn samples_per_packet(att_payload: usize) -> usize {
    (att_payload - HEADER) * 2
}

/// Device side: PCM → packets.
#[derive(Debug, Default)]
pub struct Packetizer {
    state: adpcm::State,
    seq: u8,
    started: bool,
}

impl Packetizer {
    /// A packetizer with no utterance open.
    pub const fn new() -> Self {
        Self {
            state: adpcm::State {
                predictor: 0,
                index: 0,
            },
            seq: 0,
            started: false,
        }
    }

    /// Encode `pcm` (even length) as one packet into `out`; `end` closes the
    /// utterance. The first packet after construction or after an `end`
    /// carries [`FLAG_START`]. Returns the packet length.
    ///
    /// # Panics
    /// If `pcm.len()` is odd or `out` is shorter than
    /// `HEADER + pcm.len() / 2`.
    pub fn packet(&mut self, pcm: &[i16], end: bool, out: &mut [u8]) -> usize {
        assert!(pcm.len().is_multiple_of(2) && out.len() >= HEADER + pcm.len() / 2);
        let mut flags = 0;
        if !self.started {
            flags |= FLAG_START;
            self.state = adpcm::State::default();
            // Numbering restarts per utterance, so a receiver that misses the
            // START still knows how many packets came before.
            self.seq = 0;
            self.started = true;
        }
        if end {
            flags |= FLAG_END;
            self.started = false;
        }
        out[0] = (VERSION << 4) | flags;
        out[1] = self.seq;
        out[2..4].copy_from_slice(&self.state.predictor.to_le_bytes());
        out[4] = self.state.index;
        out[5] = 0;
        adpcm::encode(&mut self.state, pcm, &mut out[HEADER..]);
        self.seq = self.seq.wrapping_add(1);
        HEADER + pcm.len() / 2
    }
}

/// What a packet did to the [`Reassembler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// Samples appended to the open utterance.
    Audio,
    /// An utterance of this many samples is complete; read it with
    /// [`Reassembler::utterance`] before the next push.
    Utterance(usize),
    /// Ignored: malformed or of another wire-format version.
    Dropped,
}

/// Phone / Mac side: packets → utterance PCM, in a caller-provided buffer.
pub struct Reassembler<'a> {
    buf: &'a mut [i16],
    len: usize,
    open: bool,
    next_seq: u8,
    last_samples: usize,
    lost: u32,
    truncated: bool,
}

impl<'a> Reassembler<'a> {
    /// Utterances longer than `buf` are cut at its length.
    pub fn new(buf: &'a mut [i16]) -> Self {
        Self {
            buf,
            len: 0,
            open: false,
            next_seq: 0,
            last_samples: 0,
            lost: 0,
            truncated: false,
        }
    }

    /// Packets lost (filled with silence) since construction.
    pub fn lost(&self) -> u32 {
        self.lost
    }

    /// Whether the last utterance hit the buffer's length.
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// The completed utterance after [`Event::Utterance`].
    pub fn utterance(&self) -> &[i16] {
        &self.buf[..self.len]
    }

    /// The utterance received so far, while it is still open (empty
    /// otherwise). For a receiver that judges audio as it arrives, such as a
    /// voice detector deciding whether to send [`CMD_REJECT`].
    pub fn partial(&self) -> &[i16] {
        if self.open {
            &self.buf[..self.len]
        } else {
            &[]
        }
    }

    fn append_silence(&mut self, n: usize) {
        let end = (self.len + n).min(self.buf.len());
        self.truncated |= self.len + n > self.buf.len();
        self.buf[self.len..end].fill(0);
        self.len = end;
    }

    /// Feed one notification.
    pub fn push(&mut self, packet: &[u8]) -> Event {
        if packet.len() < HEADER || packet[0] >> 4 != VERSION {
            return Event::Dropped;
        }
        let flags = packet[0] & 0xf;
        let seq = packet[1];
        let gap = seq.wrapping_sub(self.next_seq) as usize;
        if flags & FLAG_START != 0 {
            self.open = true;
            self.len = 0;
            self.truncated = false;
        } else if !self.open {
            // The START packet was lost, or this receiver joined mid-way:
            // open the utterance here rather than lose all of it. Indices
            // restart at 0 per utterance, so `seq` packets came before this
            // one; they become silence of this packet's length.
            self.open = true;
            self.len = 0;
            self.truncated = false;
            let missing = seq as usize;
            self.lost += missing as u32;
            self.append_silence(missing * (packet.len() - HEADER) * 2);
        } else {
            // Packets lost in between: silence of the last packet's length.
            if gap > 0 && gap < 128 {
                self.lost += gap as u32;
                self.append_silence(gap * self.last_samples);
            }
        }
        self.next_seq = seq.wrapping_add(1);

        let mut state = adpcm::State {
            predictor: i16::from_le_bytes([packet[2], packet[3]]),
            index: packet[4].min(88),
        };
        let data = &packet[HEADER..];
        let n = data.len() * 2;
        let room = (self.buf.len() - self.len) / 2;
        let take = data.len().min(room);
        self.truncated |= take < data.len();
        adpcm::decode(
            &mut state,
            &data[..take],
            &mut self.buf[self.len..self.len + 2 * take],
        );
        self.len += 2 * take;
        if n > 0 {
            self.last_samples = n;
        }
        if flags & FLAG_END != 0 {
            self.open = false;
            return Event::Utterance(self.len);
        }
        Event::Audio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| ((i as f64 * 0.07).sin() * 6000.0) as i16)
            .collect()
    }

    fn send(pcm: &[i16], per: usize) -> Vec<Vec<u8>> {
        let mut p = Packetizer::new();
        let chunks: Vec<&[i16]> = pcm.chunks(per).collect();
        chunks
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut out = vec![0u8; HEADER + per / 2];
                let n = p.packet(c, i + 1 == chunks.len(), &mut out);
                out.truncate(n);
                out
            })
            .collect()
    }

    #[test]
    fn uuid_bytes_match_strings() {
        let hex: String = SERVICE_UUID.chars().filter(|c| *c != '-').collect();
        let be: Vec<u8> = (0..16)
            .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        let mut le = be.clone();
        le.reverse();
        assert_eq!(le, SERVICE_UUID_LE);
        assert_eq!(samples_per_packet(244), 476);
    }

    #[test]
    fn utterance_round_trips() {
        let pcm = tone(16_000);
        let packets = send(&pcm, 476);
        let mut buf = vec![0i16; 32_000];
        let mut r = Reassembler::new(&mut buf);
        let mut got = None;
        for p in &packets {
            if let Event::Utterance(n) = r.push(p) {
                got = Some(n);
            }
        }
        assert_eq!(got, Some(16_000));
        let mut dec = adpcm::State::default();
        let mut whole = vec![0u8; 8000];
        adpcm::encode(&mut dec, &pcm, &mut whole);
        let mut d2 = adpcm::State::default();
        let mut ref_out = vec![0i16; 16_000];
        adpcm::decode(&mut d2, &whole, &mut ref_out);
        assert_eq!(
            r.utterance(),
            &ref_out[..],
            "packetized == continuous stream"
        );
        assert_eq!(r.lost(), 0);
    }

    #[test]
    fn lost_packet_keeps_length_and_later_audio() {
        let pcm = tone(4760);
        let mut packets = send(&pcm, 476);
        packets.remove(3);
        let mut buf = vec![0i16; 10_000];
        let mut r = Reassembler::new(&mut buf);
        let mut n = 0;
        for p in &packets {
            if let Event::Utterance(k) = r.push(p) {
                n = k;
            }
        }
        assert_eq!(n, 4760, "gap filled to the original length");
        assert_eq!(r.lost(), 1);
        assert!(r.utterance()[3 * 476..4 * 476].iter().all(|&s| s == 0));
        // Packet 4 decodes on its own: identical to the lossless decode.
        let full = send(&pcm, 476);
        let mut buf2 = vec![0i16; 10_000];
        let mut r2 = Reassembler::new(&mut buf2);
        for p in &full {
            r2.push(p);
        }
        assert_eq!(r.utterance()[4 * 476..], r2.utterance()[4 * 476..]);
    }

    #[test]
    fn truncates_and_rejects_malformed() {
        let pcm = tone(2000);
        let packets = send(&pcm, 476);
        let mut buf = vec![0i16; 1000];
        let mut r = Reassembler::new(&mut buf);
        let mut last = Event::Dropped;
        for p in &packets {
            last = r.push(p);
        }
        assert_eq!(last, Event::Utterance(1000));
        assert!(r.truncated());
        assert_eq!(
            r.push(&[0x20, 0, 0, 0, 0, 0]),
            Event::Dropped,
            "wrong version"
        );
        assert_eq!(r.push(&[0x10, 0]), Event::Dropped, "short");
    }

    #[test]
    fn lost_start_packet_still_delivers_the_utterance() {
        // Two utterances back to back; the second one's START is lost.
        let mut p = Packetizer::new();
        let a = tone(952);
        let b = tone(1428);
        let mut wire = Vec::new();
        for (pcm, _) in [(&a, 0), (&b, 1)] {
            let chunks: Vec<&[i16]> = pcm.chunks(476).collect();
            for (i, c) in chunks.iter().enumerate() {
                let mut out = vec![0u8; HEADER + 238];
                let n = p.packet(c, i + 1 == chunks.len(), &mut out);
                out.truncate(n);
                wire.push(out);
            }
        }
        // wire: a0(START) a1(END) b0(START) b1 b2(END) — drop b0.
        wire.remove(2);
        let mut buf = vec![0i16; 4000];
        let mut r = Reassembler::new(&mut buf);
        let mut got = Vec::new();
        for w in &wire {
            if let Event::Utterance(n) = r.push(w) {
                got.push(n);
            }
        }
        assert_eq!(
            got,
            vec![952, 1428],
            "length kept, START replaced by silence"
        );
        assert_eq!(r.lost(), 1);
        assert!(r.utterance()[..476].iter().all(|&s| s == 0));
    }

    #[test]
    fn joins_mid_utterance_at_full_length() {
        // A fresh receiver that missed the START still keeps the timing.
        let packets = send(&tone(1428), 476);
        let mut buf = vec![0i16; 4000];
        let mut r = Reassembler::new(&mut buf);
        assert_eq!(r.push(&packets[1]), Event::Audio);
        assert_eq!(r.push(&packets[2]), Event::Utterance(1428));
        assert_eq!(r.lost(), 1);
    }

    #[test]
    fn packet_index_restarts_per_utterance() {
        let mut p = Packetizer::new();
        let mut out = [0u8; HEADER + 2];
        p.packet(&[1, 2], false, &mut out);
        p.packet(&[1, 2], true, &mut out);
        assert_eq!(out[1], 1);
        p.packet(&[1, 2], false, &mut out);
        assert_eq!((out[0] & FLAG_START, out[1]), (FLAG_START, 0));
    }
}
