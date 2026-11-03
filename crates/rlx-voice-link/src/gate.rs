// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Voice gate: decides, one 16 ms frame at a time, when the device sends
//! audio, so the radio is on only while someone may be speaking.
//!
//! It is the cheap first stage of a cascade. The device runs only this: one
//! sum of squares per frame and a few comparisons, integer only, about 1,000
//! instructions per frame. It opens on anything louder than the room. The
//! phone runs a real voice detector on what arrives (`rlx-ten-vad`, far too
//! heavy for an always-on nRF52) and sends [`crate::CMD_REJECT`] when it is not
//! speech. The gate then closes and raises its threshold for a while, so a fan
//! or a TV does not keep the radio busy.
//!
//! * **Noise floor**: the smallest recent frame energy. It drops at once to a
//!   quieter frame and rises slowly (about 2× per 3 s) while the gate is
//!   closed. Speech does not raise it.
//! * **Open**: `open_frames` of the last 3 frames (default all 3, 48 ms of
//!   sustained sound, so clicks and taps do not open it) above
//!   `floor × ratio` and above `abs_min`. The frames before the trigger come out of a pre-roll
//!   ring first, so the first syllable is not lost.
//! * **Close**: `hangover_frames` quiet frames in a row, `max_frames` open,
//!   or a reject from the phone, which also teaches the floor the rejected
//!   sound's level.

/// PCM samples per frame (16 ms at 16 kHz), the nRF52840 PDM DMA block.
pub const FRAME: usize = 256;

/// Gate settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateConfig {
    /// Open when a frame's energy exceeds the floor by this factor, in
    /// sixteenths (`160` = 10×, 10 dB).
    pub ratio_x16: u32,
    /// Energy below this is never speech: digital silence, a muted mic.
    /// Sum of squares over one frame (`256 × rms²`).
    pub abs_min: u64,
    /// Loud frames among the last 3 needed to open.
    pub open_frames: u8,
    /// Quiet frames in a row that close the gate.
    pub hangover_frames: u16,
    /// Longest an utterance stays open, in frames.
    pub max_frames: u32,
    /// How long a reject raises the threshold, in frames.
    pub backoff_frames: u32,
    /// Threshold factor during a back-off (multiplies `ratio_x16`).
    pub backoff_mult: u32,
    /// Floor rise per closed frame: `floor += floor >> rise_shift`.
    pub rise_shift: u32,
}

impl GateConfig {
    /// The defaults, usable in a `static`.
    pub const DEFAULT: Self = Self {
        ratio_x16: 160,
        // rms 60 of 32768 (−55 dBFS).
        abs_min: 256 * 60 * 60,
        // 48 ms of sustained sound: syllables last longer, clicks
        // (typing, taps) do not.
        open_frames: 3,
        // 400 ms: inside a sentence, pauses are shorter.
        hangover_frames: 25,
        // 30 s.
        max_frames: 1875,
        // 10 s at 4× the threshold.
        backoff_frames: 625,
        backoff_mult: 4,
        // +0.4% per frame: about 2× per 2.8 s.
        rise_shift: 8,
    };
}

impl Default for GateConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// What to do with the frame just pushed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Nothing to send.
    Idle,
    /// An utterance opens: send [`Gate::preroll`] (oldest first), then this
    /// frame.
    Open,
    /// Send this frame.
    Send,
    /// Send this frame, then close the utterance.
    Close,
}

/// Counters for duty-cycle accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GateStats {
    /// Frames pushed.
    pub frames: u64,
    /// Frames sent, pre-roll included.
    pub sent: u64,
    /// Utterances opened.
    pub opens: u32,
    /// Utterances the phone rejected.
    pub rejects: u32,
    /// Utterances closed at `max_frames`.
    pub capped: u32,
}

/// The gate, with a pre-roll ring of `PRE` frames (`PRE × 512` bytes).
pub struct Gate<const PRE: usize> {
    cfg: GateConfig,
    floor: u64,
    recent: u8,
    open: bool,
    quiet: u16,
    open_for: u32,
    /// Quietest frame of the open utterance: the background under it.
    open_min: u64,
    backoff: u32,
    ring: [[i16; FRAME]; PRE],
    head: usize,
    filled: usize,
    stats: GateStats,
}

/// Sum of squares of one frame.
#[inline]
pub fn energy(frame: &[i16]) -> u64 {
    frame.iter().map(|&s| (s as i32 * s as i32) as u64).sum()
}

impl<const PRE: usize> Gate<PRE> {
    /// A closed gate. The floor starts high and settles within a few frames
    /// of room noise.
    pub const fn new(cfg: GateConfig) -> Self {
        Self {
            cfg,
            floor: u64::MAX >> 8,
            recent: 0,
            open: false,
            quiet: 0,
            open_for: 0,
            open_min: u64::MAX,
            backoff: 0,
            ring: [[0; FRAME]; PRE],
            head: 0,
            filled: 0,
            stats: GateStats {
                frames: 0,
                sent: 0,
                opens: 0,
                rejects: 0,
                capped: 0,
            },
        }
    }

    /// Settings.
    pub fn config(&self) -> &GateConfig {
        &self.cfg
    }

    /// Counters since construction.
    pub fn stats(&self) -> GateStats {
        self.stats
    }

    /// An utterance is open.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// The current noise floor (sum of squares per frame).
    pub fn floor(&self) -> u64 {
        self.floor
    }

    fn loud(&self, e: u64) -> bool {
        let mult = if self.backoff > 0 {
            self.cfg.backoff_mult as u64
        } else {
            1
        };
        e >= self.cfg.abs_min
            && e.saturating_mul(16) > self.floor.saturating_mul(self.cfg.ratio_x16 as u64 * mult)
    }

    /// Classify the next frame.
    pub fn push(&mut self, frame: &[i16; FRAME]) -> Step {
        self.stats.frames += 1;
        self.backoff = self.backoff.saturating_sub(1);
        let e = energy(frame);
        let loud = self.loud(e);
        if !self.open {
            if e < self.floor {
                self.floor = e.max(1);
            } else {
                self.floor += (self.floor >> self.cfg.rise_shift).max(1);
            }
        }
        self.recent = ((self.recent << 1) | u8::from(loud)) & 0b111;
        if self.open {
            self.open_for += 1;
            self.open_min = self.open_min.min(e);
            self.stats.sent += 1;
            self.quiet = if loud { 0 } else { self.quiet + 1 };
            if self.open_for >= self.cfg.max_frames {
                self.stats.capped += 1;
                // Something loud that never stops: treat it like a reject.
                self.backoff = self.cfg.backoff_frames;
                return self.close();
            }
            if self.quiet >= self.cfg.hangover_frames {
                return self.close();
            }
            return Step::Send;
        }
        if self.recent.count_ones() as u8 >= self.cfg.open_frames {
            self.open = true;
            self.quiet = 0;
            self.open_for = 0;
            self.open_min = e;
            self.stats.opens += 1;
            self.stats.sent += 1 + self.filled as u64;
            return Step::Open;
        }
        // Keep it for the pre-roll.
        if PRE > 0 {
            self.ring[self.head] = *frame;
            self.head = (self.head + 1) % PRE;
            self.filled = (self.filled + 1).min(PRE);
        }
        Step::Idle
    }

    fn close(&mut self) -> Step {
        self.open = false;
        self.filled = 0;
        self.recent = 0;
        Step::Close
    }

    /// The frames before the trigger, oldest first; valid right after
    /// [`Step::Open`].
    pub fn preroll(&self) -> impl Iterator<Item = &[i16; FRAME]> {
        let start = (self.head + PRE - self.filled) % PRE.max(1);
        (0..self.filled).map(move |i| &self.ring[(start + i) % PRE])
    }

    /// The phone says the open utterance is not speech: close, raise the
    /// threshold for `backoff_frames`, and take the quietest frame of what
    /// was rejected as the new noise floor. A steady sound (a fan, a car)
    /// then stays below the threshold instead of reopening the gate as soon
    /// as it closes; when it stops, the floor drops back at once. Returns
    /// whether an utterance was open (the caller then closes it on the wire).
    pub fn reject(&mut self) -> bool {
        self.stats.rejects += 1;
        self.backoff = self.cfg.backoff_frames;
        let was = self.open;
        if was && self.open_min != u64::MAX {
            self.floor = self.floor.max(self.open_min);
        }
        if was {
            self.close();
        }
        was
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(amp: i16, seed: &mut u32) -> [i16; FRAME] {
        core::array::from_fn(|_| {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let r = (*seed >> 16) as i32 % 2001 - 1000;
            (r * amp as i32 / 1000) as i16
        })
    }

    #[test]
    fn opens_on_speech_and_closes_after_hangover() {
        let mut g: Gate<4> = Gate::new(GateConfig::default());
        let mut seed = 1;
        for _ in 0..200 {
            assert_eq!(g.push(&frame(30, &mut seed)), Step::Idle);
        }
        // Loud: opens on the third loud frame in a row, with pre-roll.
        assert_eq!(g.push(&frame(3000, &mut seed)), Step::Idle);
        assert_eq!(g.push(&frame(3000, &mut seed)), Step::Idle);
        assert_eq!(g.push(&frame(3000, &mut seed)), Step::Open);
        assert_eq!(g.preroll().count(), 4);
        for _ in 0..10 {
            assert_eq!(g.push(&frame(3000, &mut seed)), Step::Send);
        }
        let cfg = *g.config();
        let mut steps = 0;
        loop {
            steps += 1;
            match g.push(&frame(30, &mut seed)) {
                Step::Send => {}
                Step::Close => break,
                s => panic!("{s:?}"),
            }
        }
        assert_eq!(steps, cfg.hangover_frames as usize);
        // The floor did not learn the speech.
        assert!(g.floor() < energy(&frame(300, &mut seed)));
    }

    #[test]
    fn reject_backs_off() {
        let mut g: Gate<0> = Gate::new(GateConfig::default());
        let mut seed = 7;
        for _ in 0..200 {
            g.push(&frame(30, &mut seed));
        }
        // 16× the floor's energy: over the 10× threshold, under the 40×
        // back-off threshold.
        g.push(&frame(120, &mut seed));
        g.push(&frame(120, &mut seed));
        assert_eq!(g.push(&frame(120, &mut seed)), Step::Open);
        assert!(g.reject());
        for _ in 0..50 {
            assert_eq!(g.push(&frame(120, &mut seed)), Step::Idle);
        }
        let s = g.stats();
        assert_eq!((s.opens, s.rejects), (1, 1));
    }

    #[test]
    fn digital_silence_never_opens() {
        let mut g: Gate<2> = Gate::new(GateConfig::default());
        for _ in 0..100 {
            assert_eq!(g.push(&[0; FRAME]), Step::Idle);
        }
        assert_eq!(g.push(&[1; FRAME]), Step::Idle);
    }
}
