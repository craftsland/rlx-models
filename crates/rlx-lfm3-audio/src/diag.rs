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

//! Timing traces (`--timing`) and the process footprint.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static TIMING: AtomicBool = AtomicBool::new(false);

/// Print graph compiles and LM feeds with their durations (stderr).
pub fn set_timing(on: bool) {
    TIMING.store(on, Ordering::Relaxed);
}

pub(crate) fn timing() -> bool {
    TIMING.load(Ordering::Relaxed)
}

fn footprint_mib() -> f64 {
    footprint() as f64 / (1u64 << 20) as f64
}

/// Prints its label and lifetime on drop when [`set_timing`] is on.
pub(crate) struct Timer(Option<(String, Instant)>);

impl Timer {
    pub(crate) fn new(label: impl FnOnce() -> String) -> Self {
        Self(timing().then(|| (label(), Instant::now())))
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if let Some((label, t)) = &self.0 {
            eprintln!(
                "[lfm3-audio] {label}: {:.2?} (footprint {:.0} MiB)",
                t.elapsed(),
                footprint_mib()
            );
        }
    }
}

/// With timing on, print `label` and the current footprint.
pub(crate) fn trace(label: &str) {
    if timing() {
        eprintln!(
            "[lfm3-audio]     {label} (footprint {:.0} MiB)",
            footprint_mib()
        );
    }
}

/// This process's physical footprint (`TASK_VM_INFO.phys_footprint`, what
/// iOS's memory limit counts); 0 elsewhere.
pub fn footprint() -> u64 {
    #[cfg(target_vendor = "apple")]
    {
        unsafe extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
        }
        let mut info = [0i32; 128];
        let mut count = info.len() as u32;
        // SAFETY: a buffer of `count` naturals for TASK_VM_INFO (flavor 22);
        // `phys_footprint` is its 19th u64.
        unsafe {
            if task_info(mach_task_self(), 22, info.as_mut_ptr(), &mut count) != 0 {
                return 0;
            }
            *(info.as_ptr() as *const u64).add(18)
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        0
    }
}
