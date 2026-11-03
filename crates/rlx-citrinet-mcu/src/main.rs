// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Bare-metal RISC-V firmware: NeMo Citrinet CTC speech recognition with no
//! OS, no allocator and no neural accelerator.
//!
//! Runs on QEMU's `virt` board with the ESP32-P4's ISA (`rv32imafc`), so the
//! compute path is what that chip executes; only the memory map and UART
//! differ. It transcribes a clip linked into the image and checks the result
//! against vectors the host produced from the same int8 model
//! (`examples/mcu_vectors.rs` in `rlx-citrinet`): equal token ids *and*
//! an equal hash of the encoder output's bits, so a pass means bit-identical
//! to the host build, not merely the same words.

#![no_std]
#![no_main]

use core::panic::PanicInfo;
use rlx_citrinet_core::{Frontend, Model, scratch_len, transcribe_with_features};

core::arch::global_asm!(
    r#"
    .section .init
    .globl _start
_start:
    la   sp, _stack_top
    // RISC-V resets with mstatus.FS = Off and every float instruction traps;
    // enable the FPU before any Rust runs.
    li   t0, 0x2000
    csrs mstatus, t0
    la   t0, __bss_start
    la   t1, __bss_end
1:  bgeu t0, t1, 2f
    sw   zero, 0(t0)
    addi t0, t0, 4
    j    1b
2:  call rust_main
3:  j    3b
"#
);

const UART: *mut u8 = 0x1000_0000 as *mut u8;
const FINISHER: *mut u32 = 0x0010_0000 as *mut u32;

fn print(s: &[u8]) {
    for &b in s {
        unsafe { UART.write_volatile(b) }
    }
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

fn print_hex(v: u32) {
    let mut buf = *b"0x00000000";
    for i in 0..8 {
        let n = (v >> (28 - 4 * i)) & 0xf;
        buf[2 + i] = if n < 10 {
            b'0' + n as u8
        } else {
            b'a' + n as u8 - 10
        };
    }
    print(&buf);
}

fn exit(code: u32) -> ! {
    // SiFive test finisher: 0x5555 = pass, (code << 16) | 0x3333 = fail.
    let v = if code == 0 {
        0x5555
    } else {
        (code << 16) | 0x3333
    };
    unsafe { FINISHER.write_volatile(v) };
    loop {
        core::hint::spin_loop();
    }
}

#[repr(C, align(4))]
struct Aligned<T: ?Sized>(T);

/// The model; memory-mapped flash on silicon.
static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!(env!("RCT_MODEL")));
static PCM: &Aligned<[u8]> = &Aligned(*include_bytes!("../data/pcm_i16.bin"));
static IDS: &Aligned<[u8]> = &Aligned(*include_bytes!("../data/ids_u16.bin"));
const FEATURES_FNV: &str = include_str!("../data/features_fnv.txt");

/// Scratch; PSRAM on silicon. 4 MiB + 256 KiB covers about 14 s of audio.
static mut FS: [f32; 1 << 20] = [0.0; 1 << 20];
static mut IS: [i8; 1 << 18] = [0; 1 << 18];
static mut FEATS: [f32; 1 << 16] = [0.0; 1 << 16];
static mut OUT: [u16; 1024] = [0; 1024];

fn parse_hex(s: &str) -> u32 {
    let s = s.trim().trim_start_matches("0x");
    let mut v = 0u32;
    for b in s.bytes() {
        let d = match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            _ => 0,
        };
        v = (v << 4) | d as u32;
    }
    v
}

fn mcycle() -> u32 {
    let c: u32;
    unsafe { core::arch::asm!("csrr {0}, mcycle", out(reg) c) };
    c
}

#[unsafe(no_mangle)]
extern "C" fn rust_main() -> ! {
    // SAFETY: single-threaded bare metal; each static is borrowed once.
    let (fs, is, feats, out) = unsafe {
        (
            &mut *core::ptr::addr_of_mut!(FS),
            &mut *core::ptr::addr_of_mut!(IS),
            &mut *core::ptr::addr_of_mut!(FEATS),
            &mut *core::ptr::addr_of_mut!(OUT),
        )
    };
    // SAFETY: the data is 4-aligned and i16/u16 accept any bit pattern.
    let pcm: &[i16] =
        unsafe { core::slice::from_raw_parts(PCM.0.as_ptr().cast(), PCM.0.len() / 2) };
    let want: &[u16] =
        unsafe { core::slice::from_raw_parts(IDS.0.as_ptr().cast(), IDS.0.len() / 2) };

    let model = match Model::parse(&MODEL.0) {
        Ok(m) => m,
        Err(_) => {
            print(b"model parse failed\n");
            exit(1)
        }
    };
    let cfg = *model.config();
    let need = scratch_len(&model, pcm.len());
    print(b"citrinet int8: ");
    print_u32(pcm.len() as u32);
    print(b" samples, model ");
    print_u32(MODEL.0.len() as u32);
    print(b" B, scratch ");
    print_u32(need.bytes() as u32);
    print(b" B\n");
    let t = model.enc_frames(Frontend::frames(&model, pcm.len()));
    if need.f32s > fs.len() || need.i8s > is.len() || t * model.decoder_width() > feats.len() {
        print(b"clip too long for the static scratch\n");
        exit(2)
    }

    let fe = Frontend::new(cfg.n_fft);
    let c0 = mcycle();
    let n = match transcribe_with_features(
        &model,
        &fe,
        pcm,
        fs,
        is,
        out,
        &mut feats[..t * model.decoder_width()],
    ) {
        Ok(n) => n,
        Err(_) => {
            print(b"transcribe failed\n");
            exit(3)
        }
    };
    let c1 = mcycle();

    let mut text = [0u8; 1024];
    let len = model.decode(&out[..n], &mut text);
    print(b"text: ");
    print(&text[..len]);
    print(b"\n");

    let mut h: u32 = 0x811c_9dc5;
    for v in &feats[..t * model.decoder_width()] {
        for b in v.to_bits().to_le_bytes() {
            h = (h ^ b as u32).wrapping_mul(0x0100_0193);
        }
    }
    let ids_ok = &out[..n] == want;
    let feats_ok = h == parse_hex(FEATURES_FNV);
    print(b"ids ");
    print(if ids_ok { b"match" } else { b"MISMATCH" });
    print(b", encoder fnv ");
    print_hex(h);
    print(if feats_ok { b" match" } else { b" MISMATCH" });
    print(b"\nmcycle ");
    print_u32(c1.wrapping_sub(c0));
    print(b" (virt board: wall-clock derived, not a cycle count)\n");
    exit(if ids_ok && feats_ok { 0 } else { 4 })
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    print(b"panic\n");
    exit(5)
}
