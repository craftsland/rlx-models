// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Instruction counts of the voice link's device-side stages on a
//! Cortex-M4F (the nRF52840's core), per 16 ms frame of the JFK clip:
//!
//! * `gate` — `rlx_voice_link::gate::Gate::push` (what runs all day);
//! * `adpcm` — `Packetizer::packet` (only while an utterance is open);
//! * `tenvad` — TEN-VAD `Vad::process_i16` (what the phone runs instead).
//!
//! Each stage is an `#[inline(never)]` function; profile with
//! `rlx-citrinet-nrf54h/tools/libpcprof.dylib` and divide by the frame count
//! this prints (`just voice-link-bench`).

#![no_std]
#![no_main]

use core::panic::PanicInfo;
use rlx_voice_link::gate::{FRAME, Gate, GateConfig};
use rlx_voice_link::{HEADER, Packetizer, samples_per_packet};

core::arch::global_asm!(
    r#"
    .section .vectors, "a"
    .word _stack_top
    .word reset
    .rept 14
    .word fault
    .endr

    .text
    .thumb_func
    .globl reset
reset:
    ldr  r0, =_stack_top
    msr  msp, r0
    @ CPACR: full access to CP10/CP11 (the FPU) before any Rust runs.
    ldr  r0, =0xE000ED88
    ldr  r1, [r0]
    orr  r1, r1, #(0xF << 20)
    str  r1, [r0]
    dsb
    isb
    @ .data: MRAM image -> RAM.
    ldr  r0, =__data_start
    ldr  r1, =__data_end
    ldr  r2, =__data_load
1:  cmp  r0, r1
    bhs  2f
    ldr  r3, [r2], #4
    str  r3, [r0], #4
    b    1b
2:  ldr  r0, =__bss_start
    ldr  r1, =__bss_end
    movs r2, #0
3:  cmp  r0, r1
    bhs  4f
    str  r2, [r0], #4
    b    3b
4:  bl   rust_main
5:  b    5b

    .thumb_func
    .globl fault
fault:
    b    fault
"#
);

/// Arm semihosting call.
fn semihost(op: u32, arg: usize) -> usize {
    let r: usize;
    // SAFETY: `bkpt 0xab` traps to the debugger/QEMU; `arg` points at a
    // valid parameter block for `op`.
    unsafe {
        core::arch::asm!("bkpt 0xab", inlateout("r0") op as usize => r, in("r1") arg,
            options(nostack));
    }
    r
}

/// Semihosting handle of the console, opened on first use.
static mut CONSOLE: usize = usize::MAX;

fn print(s: &[u8]) {
    // SAFETY: single-threaded bare metal.
    let console = unsafe { &mut *core::ptr::addr_of_mut!(CONSOLE) };
    if *console == usize::MAX {
        // SYS_OPEN ":tt" in mode 4 ("w") is the console's stdout.
        let name = b":tt\0";
        let block = [name.as_ptr() as usize, 4, 3];
        *console = semihost(0x01, block.as_ptr() as usize);
    }
    // SYS_WRITE.
    let block = [*console, s.as_ptr() as usize, s.len()];
    semihost(0x05, block.as_ptr() as usize);
}

fn print_u32(mut v: u32) {
    let mut buf = [0u8; 10];
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    print(&buf[i..]);
}

fn exit(code: u32) -> ! {
    // SYS_EXIT_EXTENDED: ADP_Stopped_ApplicationExit with an exit code.
    let block = [0x20026usize, code as usize];
    semihost(0x20, block.as_ptr() as usize);
    loop {
        core::hint::spin_loop();
    }
}

#[repr(C, align(4))]
struct Aligned<T: ?Sized>(T);

static PCM: &Aligned<[u8]> = &Aligned(*include_bytes!("../data/pcm_i16.bin"));
static mut GATE: Gate<19> = Gate::new(GateConfig::DEFAULT);
static mut VAD: core::mem::MaybeUninit<rlx_ten_vad_core::Vad> = core::mem::MaybeUninit::uninit();

/// Passes over the clip for the cheap stages, so they dominate their counts.
const REPEAT: usize = 8;

#[inline(never)]
fn stage_gate(pcm: &[i16], gate: &mut Gate<19>) -> u32 {
    let mut opens = 0;
    for _ in 0..REPEAT {
        for f in pcm.as_chunks::<FRAME>().0 {
            if gate.push(core::hint::black_box(f)) == rlx_voice_link::gate::Step::Open {
                opens += 1;
            }
        }
    }
    opens
}

#[inline(never)]
fn stage_adpcm(pcm: &[i16]) -> usize {
    let per = samples_per_packet(244);
    let mut out = [0u8; HEADER + 238];
    let mut tx = Packetizer::new();
    let mut bytes = 0;
    for _ in 0..REPEAT {
        for c in pcm.chunks_exact(per) {
            bytes += tx.packet(core::hint::black_box(c), false, &mut out);
        }
    }
    bytes
}

#[inline(never)]
fn stage_tenvad(pcm: &[i16], vad: &mut rlx_ten_vad_core::Vad) -> u32 {
    let mut speech = 0;
    for f in pcm.as_chunks::<{ rlx_ten_vad_core::HOP_SIZE }>().0 {
        if vad.process_i16(core::hint::black_box(f)) > 0.5 {
            speech += 1;
        }
    }
    speech
}

#[unsafe(no_mangle)]
extern "C" fn rust_main() -> ! {
    // SAFETY: 4-aligned, i16 accepts any bit pattern; single-threaded.
    let pcm: &[i16] =
        unsafe { core::slice::from_raw_parts(PCM.0.as_ptr().cast(), PCM.0.len() / 2) };
    let gate = unsafe { &mut *core::ptr::addr_of_mut!(GATE) };
    let vad = unsafe { (*core::ptr::addr_of_mut!(VAD)).write(rlx_ten_vad_core::Vad::new()) };
    let frames = (pcm.len() / FRAME) as u32;
    let opens = stage_gate(pcm, gate);
    let bytes = stage_adpcm(pcm);
    let speech = stage_tenvad(pcm, vad);
    print(b"frames gate ");
    print_u32(frames * REPEAT as u32);
    print(b" (opens ");
    print_u32(opens);
    print(b"), adpcm ");
    print_u32(frames * REPEAT as u32);
    print(b" (");
    print_u32(bytes as u32);
    print(b" B), tenvad ");
    print_u32((pcm.len() / rlx_ten_vad_core::HOP_SIZE) as u32);
    print(b" (speech ");
    print_u32(speech);
    print(b")\n");
    exit(0)
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    print(b"panic\n");
    exit(5)
}
