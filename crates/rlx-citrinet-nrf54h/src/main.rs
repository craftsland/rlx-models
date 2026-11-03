// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Bare-metal Cortex-M33 firmware for the nRF54H20's application core:
//! NeMo Citrinet CTC speech recognition with no OS, no allocator and no
//! accelerator, inside the chip's memories (see `link.x`):
//!
//! * the 13.3 MB model is read in place from external flash (EXMIF),
//! * code and constants live in the 2 MB MRAM,
//! * every buffer fits the 768 KB of general-purpose RAM.
//!
//! Scratch grows with the utterance, so audio longer than the scratch allows
//! is cut at pauses (`next_cut`) and transcribed piece by piece. The clip,
//! the expected token ids and the expected hash of every piece's encoder
//! output come from the host (`examples/mcu_vectors.rs` in `rlx-citrinet`,
//! run with this firmware's scratch sizes); a pass means bit-identical to the
//! host build.
//!
//! Runs on QEMU's MPS2 AN505 (a Cortex-M33 board). Semihosting carries the
//! output and the exit code.

#![no_std]
#![no_main]

use core::panic::PanicInfo;
use rlx_citrinet_core::{
    Error, Frontend, Model, StageInput, StageOutput, encoder_output, max_samples, next_cut,
    run_stage, scratch_len, split, stage_output,
};

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
    // SYS_EXIT_EXTENDED: ADP_Stopped_ApplicationExit with an exit code.
    let block = [0x20026usize, code as usize];
    semihost(0x20, block.as_ptr() as usize);
    loop {
        core::hint::spin_loop();
    }
}

#[repr(C, align(4))]
struct Aligned<T: ?Sized>(T);

/// The model, in external flash.
#[unsafe(link_section = ".exmif")]
static MODEL: Aligned<[u8; include_bytes!(env!("RCT_MODEL")).len()]> =
    Aligned(*include_bytes!(env!("RCT_MODEL")));
static PCM: &Aligned<[u8]> = &Aligned(*include_bytes!("../data/pcm_i16.bin"));
static IDS: &Aligned<[u8]> = &Aligned(*include_bytes!("../data/ids_u16.bin"));
const FEATURES_FNV: &str = include_str!("../data/features_fnv.txt");
const MAX_SAMPLES: &str = include_str!("../data/max_samples.txt");

/// Scratch, in RAM: 704 KB of f32 + 4 KB of i8. `mcu_vectors` must be run
/// with these sizes (`--f32s 180224 --i8s 4096`).
const F32S: usize = 180_224;
const I8S: usize = 4096;
static mut FS: [f32; F32S] = [0.0; F32S];
static mut IS: [i8; I8S] = [0; I8S];
static mut OUT: [u16; 512] = [0; 512];

fn parse_dec(s: &str) -> usize {
    s.trim()
        .bytes()
        .fold(0, |v, b| v * 10 + (b.wrapping_sub(b'0') as usize % 10))
}

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

/// CMSDK APB UART on the AN505 (UART0, UART1): a stand-in for the SPI or
/// UART between two chips. Polled, no interrupts.
struct Uart(usize);

impl Uart {
    const DATA: usize = 0x00;
    const STATE: usize = 0x04;
    const CTRL: usize = 0x08;
    const BAUDDIV: usize = 0x10;

    fn new(base: usize) -> Self {
        let u = Self(base);
        u.set(Self::BAUDDIV, 16);
        u.set(Self::CTRL, 0b11); // TX and RX enable
        u
    }
    fn set(&self, reg: usize, v: u32) {
        // SAFETY: a device register of the board's UART.
        unsafe { ((self.0 + reg) as *mut u32).write_volatile(v) }
    }
    fn get(&self, reg: usize) -> u32 {
        // SAFETY: a device register of the board's UART.
        unsafe { ((self.0 + reg) as *const u32).read_volatile() }
    }
}

impl split::Link for Uart {
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        for &b in bytes {
            while self.get(Self::STATE) & 1 != 0 {}
            self.set(Self::DATA, b as u32);
        }
        Ok(())
    }
    fn read(&mut self, buf: &mut [u8]) -> Result<(), Error> {
        for b in buf {
            while self.get(Self::STATE) & 2 == 0 {}
            *b = self.get(Self::DATA) as u8;
        }
        Ok(())
    }
}

const UART0: usize = 0x4020_0000;
const UART1: usize = 0x4020_1000;

fn fail(msg: &[u8], code: u32) -> ! {
    print(msg);
    exit(code)
}

fn fnv(mut h: u32, v: &[f32]) -> u32 {
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h = (h ^ b as u32).wrapping_mul(0x0100_0193);
        }
    }
    h
}

/// After a piece: count the ids and hash the encoder output (last stage), or
/// send the activation rows on (other stages).
#[allow(clippy::too_many_arguments)]
fn finish(
    model: &Model<'_>,
    first: bool,
    rows_or_samples: usize,
    result: StageOutput,
    fs: &[f32],
    tx: &mut Option<Uart>,
    seq: u32,
    n: &mut usize,
    h: &mut u32,
) {
    match result {
        StageOutput::Ids(k) => {
            *n += k;
            *h = fnv(*h, encoder_output(model, rows_or_samples, fs));
        }
        StageOutput::Rows(rows) => {
            let w = model.decoder_width();
            let link = tx.get_or_insert_with(|| Uart::new(if first { UART0 } else { UART1 }));
            if split::send_rows(link, seq, rows, w, stage_output(model, rows, fs)).is_err() {
                fail(b"send failed\n", 3)
            }
            print(b"sent ");
            print_u32(rows as u32);
            print(b" rows\n");
        }
    }
}

/// Whole model, or one pipeline stage, chosen by the `.rct` this image was
/// built with: the first stage transcribes the audio and sends activation
/// frames on UART0; a later stage receives them on UART0, and either sends
/// its own on UART1 or, holding the CTC head, checks the result.
#[unsafe(no_mangle)]
extern "C" fn rust_main() -> ! {
    // SAFETY: single-threaded bare metal; each static is borrowed once.
    let (fs, is, out) = unsafe {
        (
            &mut *core::ptr::addr_of_mut!(FS),
            &mut *core::ptr::addr_of_mut!(IS),
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
        Err(_) => fail(b"model parse failed\n", 1),
    };
    let cfg = *model.config();
    let (first, last) = (cfg.first_block == 0, cfg.has_head);
    print(b"citrinet int8 on nRF54H20 app core (Cortex-M33");
    print(if cfg!(feature = "dsp") {
        b" + DSP"
    } else {
        b", portable"
    });
    print(b"): blocks ");
    print_u32(cfg.first_block as u32);
    print(b"..");
    print_u32((cfg.first_block + cfg.n_blocks) as u32);
    print(b" of ");
    print_u32(cfg.total_blocks as u32);
    print(b", model ");
    print_u32(MODEL.0.len() as u32);
    print(b" B in external flash, scratch ");
    print_u32((F32S * 4 + I8S) as u32);
    print(b" B in RAM\n");

    // Pieces are cut where the host cut them (`max_samples.txt`, from the
    // whole model's scratch), so every arrangement checks against the same
    // vectors.
    let max_len = parse_dec(MAX_SAMPLES);
    if first && last && max_len != max_samples(&model, F32S, I8S) {
        fail(
            b"max_samples differs from the host's: regenerate data/\n",
            2,
        )
    }
    let mut h: u32 = 0x811c_9dc5;
    let mut n = 0;
    let mut tx: Option<Uart> = None;
    let mut seq = 0u32;
    if first {
        let fe = Frontend::new(cfg.n_fft);
        let mut start = 0;
        while start < pcm.len() {
            let end = next_cut(pcm, start, max_len, 400);
            let piece = &pcm[start..end];
            let need = scratch_len(&model, piece.len());
            if need.f32s > F32S || need.i8s > I8S {
                fail(b"piece too long for the scratch\n", 2)
            }
            let r = run_stage(&model, StageInput::Pcm(piece, &fe), fs, is, &mut out[n..]);
            let r = r.unwrap_or_else(|_| fail(b"run failed\n", 3));
            finish(
                &model,
                first,
                piece.len(),
                r,
                fs,
                &mut tx,
                seq,
                &mut n,
                &mut h,
            );
            print(b"piece ");
            print_u32(start as u32);
            print(b"..");
            print_u32(end as u32);
            print(b"\n");
            start = end;
            seq += 1;
        }
    } else {
        let mut rx = Uart::new(UART0);
        loop {
            let frame = split::recv_rows(&mut rx, model.input_width(), fs)
                .unwrap_or_else(|_| fail(b"receive failed\n", 3));
            if frame.rows == 0 {
                break;
            }
            let need = scratch_len(&model, frame.rows);
            if need.f32s > F32S || need.i8s > I8S {
                fail(b"frame too long for the scratch\n", 2)
            }
            let r = run_stage::<[i16]>(&model, StageInput::Rows(frame.rows), fs, is, &mut out[n..]);
            let r = r.unwrap_or_else(|_| fail(b"run failed\n", 3));
            print(b"received ");
            print_u32(frame.rows as u32);
            print(b" rows\n");
            finish(
                &model, first, frame.rows, r, fs, &mut tx, frame.seq, &mut n, &mut h,
            );
            seq = frame.seq + 1;
        }
    }
    if !last {
        // An empty frame ends the stream for the next stage.
        let link = tx.get_or_insert_with(|| Uart::new(if first { UART0 } else { UART1 }));
        if split::send_rows(link, seq, 0, model.decoder_width(), &[]).is_err() {
            fail(b"send failed\n", 3)
        }
        print(b"stage done\n");
        exit(0)
    }

    let mut text = [0u8; 1024];
    let len = model.decode(&out[..n], &mut text);
    print(b"text: ");
    print(&text[..len]);
    print(b"\n");
    let ids_ok = &out[..n] == want;
    let feats_ok = h == parse_hex(FEATURES_FNV);
    print(b"ids ");
    print(if ids_ok { b"match" } else { b"MISMATCH" });
    print(b", encoder fnv ");
    print_hex(h);
    print(if feats_ok {
        b" match\n"
    } else {
        b" MISMATCH\n"
    });
    exit(if ids_ok && feats_ok { 0 } else { 4 })
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    print(b"panic\n");
    exit(5)
}
