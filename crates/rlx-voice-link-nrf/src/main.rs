// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! nRF52840 voice link: PDM microphone → IMA-ADPCM → BLE notifications.
//!
//! The device only captures and streams; recognition and the language model
//! run on the phone or Mac (`rlx-voice-link listen`, or the Swift package).
//!
//! Hands-free by default: [`rlx_voice_link::gate::Gate`] opens an utterance on
//! sustained sound above the room's noise floor (an integer sum of squares
//! per 16 ms frame) and sends ~300 ms of pre-roll first. The phone confirms
//! with a voice detector and writes `CMD_REJECT` when it is not speech; the
//! gate then closes and learns that sound as background. The radio carries
//! audio only around speech. The board button or `CMD_START` / `CMD_STOP`
//! still force push-to-talk from either side.
//!
//! * PDM at 1.280 MHz / ratio 80 = 16 kHz mono, 16 ms DMA frames.
//! * Frames go through a 32-deep channel (512 ms, room for the pre-roll) to
//!   the BLE task, which
//!   packs as many samples as the negotiated ATT MTU allows (476 at MTU 247)
//!   into one notification. Frames are dropped, not blocked on, if the radio
//!   falls behind; the receiver fills the gap with silence.
//! * After connecting it asks for 2M PHY and 251-byte data length: about 34
//!   notifications/s and 66 kbit/s, well inside what iOS and macOS sustain.

#![no_std]
#![no_main]

use core::sync::atomic::{AtomicBool, Ordering};

use defmt::{info, unwrap, warn};
use embassy_executor::Spawner;
use embassy_futures::join::join;
use embassy_futures::select::{Either, select};
#[cfg(feature = "board-xiao")]
use embassy_nrf::gpio::{Input, Pull};
use embassy_nrf::gpio::{Level, Output, OutputDrive};
use embassy_nrf::mode::Async;
use embassy_nrf::pdm::{self, Frequency, Pdm, Ratio, SamplerState};
use embassy_nrf::peripherals::{PDM, RNG};
use embassy_nrf::{Peri, bind_interrupts, rng};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel;
use embassy_sync::signal::Signal;
use embassy_time::Timer;
use fixed::types::I7F1;
use nrf_sdc::mpsl::MultiprotocolServiceLayer;
use nrf_sdc::{self as sdc, mpsl};
use rlx_voice_link::gate::{Gate, GateConfig, Step};
use rlx_voice_link::{
    CMD_REJECT, CMD_START, CMD_STOP, HEADER, Packetizer, SERVICE_UUID_LE, samples_per_packet,
};
use static_cell::StaticCell;
use trouble_host::prelude::*;
use {defmt_rtt as _, panic_probe as _};

bind_interrupts!(struct Irqs {
    RNG => rng::InterruptHandler<RNG>;
    PDM => pdm::InterruptHandler<PDM>;
    EGU0_SWI0 => nrf_sdc::mpsl::LowPrioInterruptHandler;
    CLOCK_POWER => nrf_sdc::mpsl::ClockInterruptHandler;
    RADIO => nrf_sdc::mpsl::HighPrioInterruptHandler;
    TIMER0 => nrf_sdc::mpsl::HighPrioInterruptHandler;
    RTC0 => nrf_sdc::mpsl::HighPrioInterruptHandler;
});

/// PDM DMA frame: 16 ms at 16 kHz.
const FRAME: usize = 256;
/// Most samples one notification carries (ATT payload 244).
const MAX_PER_PACKET: usize = samples_per_packet(244);
/// The audio characteristic's declared size: header + 238 bytes of codes.
const AUDIO_LEN: usize = HEADER + MAX_PER_PACKET / 2;

static RECORDING: AtomicBool = AtomicBool::new(false);
/// Push-to-talk (button or `CMD_START`) is on: send every frame, ignore the
/// gate.
static MANUAL: AtomicBool = AtomicBool::new(false);
/// The phone wrote `CMD_REJECT`; the mic task applies it at the next frame.
static REJECT: AtomicBool = AtomicBool::new(false);
/// Frames of pre-roll the gate keeps (304 ms, 9.7 KB).
const PRE: usize = 19;
/// Raised whenever `RECORDING` changes, so the BLE task can close an utterance.
static TOGGLED: Signal<CriticalSectionRawMutex, ()> = Signal::new();
static AUDIO: Channel<CriticalSectionRawMutex, [i16; FRAME], 32> = Channel::new();

/// Push-to-talk on or off.
fn set_manual(on: bool) {
    MANUAL.store(on, Ordering::Release);
    set_recording(on);
}

fn set_recording(on: bool) {
    if RECORDING.swap(on, Ordering::AcqRel) != on {
        info!("recording {}", on);
        TOGGLED.signal(());
    }
}

// ---- GATT ----------------------------------------------------------------

#[gatt_server]
struct Server {
    voice: VoiceService,
}

/// `rlx_voice_link::SERVICE_UUID` and its characteristics. The macro needs
/// literals; the `const` assertions below keep them equal to the crate's.
#[gatt_service(uuid = "7a1e0001-5c2b-4e8a-9f43-72c0b1d6a5e3")]
struct VoiceService {
    /// Audio packets (notify only).
    #[characteristic(uuid = "7a1e0002-5c2b-4e8a-9f43-72c0b1d6a5e3", notify, value = [0u8; AUDIO_LEN])]
    audio: [u8; AUDIO_LEN],
    /// Commands: `CMD_START` / `CMD_STOP` (push-to-talk), `CMD_REJECT` (not
    /// speech: close and back off).
    #[characteristic(
        uuid = "7a1e0003-5c2b-4e8a-9f43-72c0b1d6a5e3",
        write,
        write_without_response
    )]
    control: u8,
}

const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}
const _: () = assert!(same(
    rlx_voice_link::SERVICE_UUID,
    "7a1e0001-5c2b-4e8a-9f43-72c0b1d6a5e3"
));
const _: () = assert!(same(
    rlx_voice_link::AUDIO_CHAR_UUID,
    "7a1e0002-5c2b-4e8a-9f43-72c0b1d6a5e3"
));
const _: () = assert!(same(
    rlx_voice_link::CONTROL_CHAR_UUID,
    "7a1e0003-5c2b-4e8a-9f43-72c0b1d6a5e3"
));

// ---- tasks ---------------------------------------------------------------

#[embassy_executor::task]
async fn mpsl_task(mpsl: &'static MultiprotocolServiceLayer<'static>) -> ! {
    mpsl.run().await
}

fn queue(frame: &[i16; FRAME]) {
    if AUDIO.try_send(*frame).is_err() {
        warn!("audio queue full, frame dropped");
    }
}

/// Continuous PDM capture. Every frame goes through the voice gate; frames
/// are queued for the radio while an utterance is open (or push-to-talk is
/// on).
#[embassy_executor::task]
async fn mic_task(mut pdm: Pdm<'static>) {
    let mut bufs = [[0i16; FRAME]; 2];
    let mut gate: Gate<PRE> = Gate::new(GateConfig::default());
    loop {
        let r = pdm
            .run_task_sampler(&mut bufs, |frame| {
                if MANUAL.load(Ordering::Acquire) {
                    queue(frame);
                    return SamplerState::Sampled;
                }
                if REJECT.swap(false, Ordering::AcqRel) && gate.reject() {
                    set_recording(false);
                }
                match gate.push(frame) {
                    Step::Idle => {}
                    Step::Open => {
                        for f in gate.preroll() {
                            queue(f);
                        }
                        queue(frame);
                        set_recording(true);
                    }
                    Step::Send => queue(frame),
                    Step::Close => {
                        queue(frame);
                        set_recording(false);
                    }
                }
                SamplerState::Sampled
            })
            .await;
        if r.is_err() {
            warn!("pdm sampler error; restarting");
            Timer::after_millis(10).await;
        }
    }
}

/// Board button (to GND): each press toggles recording.
#[cfg(feature = "board-xiao")]
#[embassy_executor::task]
async fn button_task(mut button: Input<'static>) {
    loop {
        button.wait_for_falling_edge().await;
        set_manual(!MANUAL.load(Ordering::Acquire));
        Timer::after_millis(40).await;
        button.wait_for_high().await;
        Timer::after_millis(40).await;
    }
}

async fn ble_task<C: Controller, P: PacketPool>(mut runner: Runner<'_, C, P>) {
    loop {
        if let Err(e) = runner.run().await {
            panic!("[ble] {:?}", defmt::Debug2Format(&e));
        }
    }
}

/// GATT requests until disconnect: control writes start/stop recording.
async fn gatt_events(server: &Server<'_>, conn: &GattConnection<'_, '_, DefaultPacketPool>) {
    let control = server.voice.control.handle;
    let reason = loop {
        match conn.next().await {
            GattConnectionEvent::Disconnected { reason } => break reason,
            GattConnectionEvent::Gatt { event } => {
                if let GattEvent::Write(w) = &event
                    && w.handle() == control
                {
                    w.with_data(|_, data| match data.first() {
                        Some(&CMD_START) => set_manual(true),
                        Some(&CMD_STOP) => set_manual(false),
                        Some(&CMD_REJECT) => REJECT.store(true, Ordering::Release),
                        _ => {}
                    });
                }
                if let Ok(reply) = event.accept() {
                    reply.send().await;
                }
            }
            _ => {}
        }
    };
    info!("disconnected: {:?}", reason);
    set_manual(false);
}

/// Stream recorded audio as notifications until the connection drops.
async fn stream<C>(
    server: &Server<'_>,
    conn: &GattConnection<'_, '_, DefaultPacketPool>,
    stack: &Stack<'_, C, DefaultPacketPool>,
) where
    C: Controller
        + bt_hci::controller::ControllerCmdSync<bt_hci::cmd::le::LeSetDataLength>
        + bt_hci::controller::ControllerCmdSync<bt_hci::cmd::le::LeReadLocalSupportedFeatures>
        + bt_hci::controller::ControllerCmdAsync<bt_hci::cmd::le::LeSetPhy>,
{
    // Throughput: 2M PHY and long packets, best effort — the central decides,
    // and the link works at 1M / 27-byte packets too, just with smaller
    // notifications. The connection interval is left to the central: the
    // peripheral-only controller library has no LE Connection Update, and
    // iOS/macOS choose 15-30 ms, where several notifications fit one event.
    let raw = conn.raw();
    if raw.set_phy(stack, PhyKind::Le2M).await.is_err() {
        warn!("2M PHY refused");
    }
    if raw.update_data_length(stack, 251, 2120).await.is_err() {
        warn!("data length extension refused");
    }

    let audio = server.voice.audio;
    let mut tx = Packetizer::new();
    let mut acc = [0i16; MAX_PER_PACKET];
    let mut fill = 0usize;
    let mut open = false;
    let mut out = [0u8; AUDIO_LEN];
    loop {
        // The MTU can change after connecting (exchange-MTU), so read it per
        // packet. ATT notification payload is MTU - 3.
        let att = (raw.att_mtu() as usize).saturating_sub(3);
        let per = samples_per_packet(att.clamp(HEADER + 2, AUDIO_LEN)).min(MAX_PER_PACKET);
        match select(AUDIO.receive(), TOGGLED.wait()).await {
            Either::First(frame) => {
                open = true;
                let mut rest: &[i16] = &frame;
                while !rest.is_empty() {
                    let take = (per - fill).min(rest.len());
                    acc[fill..fill + take].copy_from_slice(&rest[..take]);
                    fill += take;
                    rest = &rest[take..];
                    if fill == per {
                        let n = tx.packet(&acc[..per], false, &mut out);
                        if audio.notify_raw(conn, &out[..n], false).await.is_err() {
                            return;
                        }
                        fill = 0;
                    }
                }
            }
            Either::Second(()) => {
                if RECORDING.load(Ordering::Acquire) || !open {
                    continue;
                }
                // Recording stopped: drain what the mic already queued,
                // then close the utterance with the remainder.
                while let Ok(frame) = AUDIO.try_receive() {
                    for &s in frame.iter() {
                        if fill == per {
                            let n = tx.packet(&acc[..per], false, &mut out);
                            if audio.notify_raw(conn, &out[..n], false).await.is_err() {
                                return;
                            }
                            fill = 0;
                        }
                        acc[fill] = s;
                        fill += 1;
                    }
                }
                let even = fill & !1;
                let n = tx.packet(&acc[..even], true, &mut out);
                if audio.notify_raw(conn, &out[..n], false).await.is_err() {
                    return;
                }
                fill = 0;
                open = false;
            }
        }
    }
}

async fn advertise<'v, 's, C: Controller>(
    peripheral: &mut Peripheral<'v, C, DefaultPacketPool>,
    server: &'s Server<'v>,
) -> Result<GattConnection<'v, 's, DefaultPacketPool>, BleHostError<C::Error>> {
    let mut adv = [0u8; 31];
    let len = AdStructure::encode_slice(
        &[
            AdStructure::Flags(LE_GENERAL_DISCOVERABLE | BR_EDR_NOT_SUPPORTED),
            AdStructure::CompleteServiceUuids128(&[SERVICE_UUID_LE]),
        ],
        &mut adv[..],
    )?;
    let mut scan = [0u8; 31];
    let scan_len = AdStructure::encode_slice(
        &[AdStructure::CompleteLocalName(b"rlx-voice")],
        &mut scan[..],
    )?;
    let advertiser = peripheral
        .advertise(
            &Default::default(),
            Advertisement::ConnectableScannableUndirected {
                adv_data: &adv[..len],
                scan_data: &scan[..scan_len],
            },
        )
        .await?;
    info!("advertising");
    let conn = advertiser.accept().await?.with_attribute_server(server)?;
    info!("connected");
    Ok(conn)
}

fn build_sdc<'d, const N: usize>(
    p: nrf_sdc::Peripherals<'d>,
    rng: &'d mut rng::Rng<Async>,
    mpsl: &'d MultiprotocolServiceLayer,
    mem: &'d mut sdc::Mem<N>,
) -> Result<nrf_sdc::SoftdeviceController<'d>, nrf_sdc::Error> {
    sdc::Builder::new()?
        .support_adv()
        .support_peripheral()
        .support_dle_peripheral()
        .support_le_2m_phy()
        .support_phy_update_peripheral()
        .peripheral_count(1)?
        .buffer_cfg(
            DefaultPacketPool::MTU as u16,
            DefaultPacketPool::MTU as u16,
            4,
            3,
        )?
        .build(p, rng, mpsl, mem)
}

fn microphone(
    pdm: Peri<'static, PDM>,
    clk: Peri<'static, impl embassy_nrf::gpio::Pin>,
    din: Peri<'static, impl embassy_nrf::gpio::Pin>,
) -> Pdm<'static> {
    // MEMS PDM mics come in quiet; +10 dB keeps speech well above the
    // ADPCM step floor without clipping at conversational distance.
    let config = pdm::Config {
        frequency: Frequency::_1280k,
        ratio: Ratio::Ratio80,
        gain_left: I7F1::from_num(10),
        gain_right: I7F1::from_num(10),
        ..Default::default()
    };
    Pdm::new(pdm, Irqs, clk, din, config)
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_nrf::init(Default::default());

    let mpsl_p =
        mpsl::Peripherals::new(p.RTC0, p.TIMER0, p.TEMP, p.PPI_CH19, p.PPI_CH30, p.PPI_CH31);
    let lfclk_cfg = mpsl::raw::mpsl_clock_lfclk_cfg_t {
        source: mpsl::raw::MPSL_CLOCK_LF_SRC_RC as u8,
        rc_ctiv: mpsl::raw::MPSL_RECOMMENDED_RC_CTIV as u8,
        rc_temp_ctiv: mpsl::raw::MPSL_RECOMMENDED_RC_TEMP_CTIV as u8,
        accuracy_ppm: mpsl::raw::MPSL_DEFAULT_CLOCK_ACCURACY_PPM as u16,
        skip_wait_lfclk_started: mpsl::raw::MPSL_DEFAULT_SKIP_WAIT_LFCLK_STARTED != 0,
    };
    static MPSL: StaticCell<MultiprotocolServiceLayer> = StaticCell::new();
    let mpsl = MPSL.init(unwrap!(mpsl::MultiprotocolServiceLayer::new(
        mpsl_p, Irqs, lfclk_cfg
    )));
    spawner.spawn(unwrap!(mpsl_task(&*mpsl)));

    // Board wiring.
    #[cfg(feature = "board-xiao")]
    let (mic, _mic_power) = {
        let power = Output::new(p.P1_10, Level::High, OutputDrive::Standard);
        Timer::after_millis(10).await;
        let button = Input::new(p.P0_03, Pull::Up);
        spawner.spawn(unwrap!(button_task(button)));
        (microphone(p.PDM, p.P1_00, p.P0_16), power)
    };
    #[cfg(all(feature = "board-nano33", not(feature = "board-xiao")))]
    let (mic, _mic_power) = {
        let power = Output::new(p.P0_17, Level::High, OutputDrive::HighDrive);
        Timer::after_millis(10).await;
        (microphone(p.PDM, p.P0_26, p.P0_25), power)
    };
    spawner.spawn(unwrap!(mic_task(mic)));

    let sdc_p = sdc::Peripherals::new(
        p.PPI_CH17, p.PPI_CH18, p.PPI_CH20, p.PPI_CH21, p.PPI_CH22, p.PPI_CH23, p.PPI_CH24,
        p.PPI_CH25, p.PPI_CH26, p.PPI_CH27, p.PPI_CH28, p.PPI_CH29,
    );
    static RNG_CELL: StaticCell<rng::Rng<Async>> = StaticCell::new();
    let rng = RNG_CELL.init(rng::Rng::new(p.RNG, Irqs));
    static SDC_MEM: StaticCell<sdc::Mem<16384>> = StaticCell::new();
    let sdc_mem = SDC_MEM.init(sdc::Mem::new());
    let sdc = unwrap!(build_sdc(sdc_p, rng, mpsl, sdc_mem));

    // A fixed static random address; replace with one derived from FICR for
    // more than one device in range.
    let address = Address::random([0x52, 0x4c, 0x58, 0x56, 0x4c, 0xc0]);
    let mut resources: HostResources<DefaultPacketPool, 1, 2> = HostResources::new();
    let stack = trouble_host::new(sdc, &mut resources)
        .set_random_address(address)
        .build();
    let runner = stack.runner();
    let mut peripheral = stack.peripheral();
    let server = unwrap!(Server::new_with_config(GapConfig::Peripheral(
        PeripheralConfig {
            name: "rlx-voice",
            appearance: &appearance::UNKNOWN,
        }
    )));

    join(ble_task(runner), async {
        loop {
            match advertise(&mut peripheral, &server).await {
                Ok(conn) => {
                    select(gatt_events(&server, &conn), stream(&server, &conn, &stack)).await;
                }
                Err(e) => {
                    warn!("advertise: {:?}", defmt::Debug2Format(&e));
                    Timer::after_secs(1).await;
                }
            }
        }
    })
    .await;
}
