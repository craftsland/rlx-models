// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Cut Citrinet into pipeline stages for several chips (or cores), one
//! command: export one `.rct` per stage, print the plan, run the stages
//! chained through the wire format they would use on SPI/UART
//! (`rlx_mcu::split`), and check the result is bit-identical to the whole
//! model.
//!
//! ```text
//! cargo run -p rlx-citrinet --release --example mcu_split -- \
//!     --nemo stt_en_citrinet_256_ls.nemo [--stages 2 | --cuts 0,8,23] \
//!     [--wav clip.wav] [--out-dir DIR] [--bits 8|9]
//! ```
//! `--stages N` (default 2) balances compute; `--cuts` sets the block
//! boundaries. `--out-dir` writes `stage<i>of<n>.rct`.

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use rlx_citrinet::mcu::{Blob, Stages, export_rct, export_rct_stages};
use rlx_citrinet_core as core;
use rlx_mcu::split::{MemLink, recv_rows, send_rows};
use rlx_nemo::NemoModel;

fn fnv(h: u32, v: &[f32]) -> u32 {
    v.iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .fold(h, |h, b| (h ^ b as u32).wrapping_mul(0x0100_0193))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    let nemo = NemoModel::open(Path::new(get("--nemo").context("--nemo")?))?;
    let bits: u32 = get("--bits").map(|b| b.parse()).transpose()?.unwrap_or(8);
    let stages = match get("--cuts") {
        Some(c) => Stages::Cuts(
            c.split(',')
                .map(|v| v.trim().parse())
                .collect::<Result<_, _>>()?,
        ),
        None => Stages::Balanced(get("--stages").map(|v| v.parse()).transpose()?.unwrap_or(2)),
    };
    let wav = get("--wav").map_or("assets/harness/jfk_16k.wav", String::as_str);
    let (pcm, rate) = rlx_core::asr_bench::read_wav_mono(Path::new(wav))?;
    ensure!(rate == 16_000, "need a 16 kHz clip");

    let whole = Blob::new(export_rct(&nemo, bits)?);
    let whole = core::Model::parse(whole.bytes()).map_err(|e| anyhow!("{e}"))?;
    let (cuts, blobs) = export_rct_stages(&nemo, bits, &stages)?;
    let blobs: Vec<Blob> = blobs.into_iter().map(Blob::new).collect();
    let models = blobs
        .iter()
        .map(|b| core::Model::parse(b.bytes()).map_err(|e| anyhow!("{e}")))
        .collect::<Result<Vec<_>>>()?;
    let fe = core::Frontend::new(whole.config().n_fft);

    // Whole model.
    let need = core::scratch_len(&whole, pcm.len());
    let (mut fs, mut is) = (vec![0f32; need.f32s], vec![0i8; need.i8s]);
    let mut ids = vec![0u16; pcm.len() / 160 + 1];
    let n = core::transcribe(&whole, &fe, pcm.as_slice(), &mut fs, &mut is, &mut ids)
        .map_err(|e| anyhow!("{e}"))?;
    let want_ids = ids[..n].to_vec();
    let want_hash = fnv(0x811c_9dc5, core::encoder_output(&whole, pcm.len(), &fs));

    // The same audio through the stages, chained over the wire format.
    let mut costs = vec![0u64; whole.config().total_blocks];
    core::block_costs(&whole, pcm.len(), &mut costs);
    let total: u64 = costs.iter().sum();
    let mut link = MemLink::default();
    let mut rows_in = 0;
    let mut got_ids = Vec::new();
    let mut got_hash = 0;
    println!(
        "{} stages over {} blocks, {:.2} s clip:",
        models.len(),
        whole.config().total_blocks,
        pcm.len() as f64 / 16_000.0
    );
    println!("stage  blocks   file MB  compute  scratch KB  sends");
    for (s, m) in models.iter().enumerate() {
        let cfg = *m.config();
        let n_in = if s == 0 { pcm.len() } else { rows_in };
        let need = core::scratch_len(m, n_in);
        let (mut fs, mut is) = (vec![0f32; need.f32s], vec![0i8; need.i8s]);
        let input = if s == 0 {
            core::StageInput::Pcm(pcm.as_slice(), &fe)
        } else {
            let f = recv_rows(
                &mut link,
                m.input_width(),
                core::stage_input(m, rows_in, &mut fs),
            )
            .map_err(|e| anyhow!("stage {s} receive: {e}"))?;
            ensure!(f.rows == rows_in && f.seq == s as u32 - 1);
            core::StageInput::Rows(rows_in)
        };
        let share: u64 = costs[cfg.first_block..cfg.first_block + cfg.n_blocks]
            .iter()
            .sum();
        let sends = match core::run_stage(m, input, &mut fs, &mut is, &mut ids)
            .map_err(|e| anyhow!("stage {s}: {e}"))?
        {
            core::StageOutput::Rows(rows) => {
                let w = m.decoder_width();
                send_rows(
                    &mut link,
                    s as u32,
                    rows,
                    w,
                    core::stage_output(m, rows, &fs),
                )
                .map_err(|e| anyhow!("{e}"))?;
                rows_in = rows;
                format!(
                    "{rows}×{w} f32 = {:.1} KB",
                    (rows * w * 4 + 20) as f64 / 1024.0
                )
            }
            core::StageOutput::Ids(k) => {
                got_ids = ids[..k].to_vec();
                got_hash = fnv(0x811c_9dc5, core::encoder_output(m, rows_in, &fs));
                format!("{k} token ids")
            }
        };
        println!(
            "{s:>5}  {:>2}..{:<2}  {:>7.2}  {:>6.1}%  {:>10.1}  {sends}",
            cfg.first_block,
            cfg.first_block + cfg.n_blocks,
            blobs[s].bytes().len() as f64 / 1e6,
            100.0 * share as f64 / total as f64,
            need.bytes() as f64 / 1024.0,
        );
        if let Some(dir) = get("--out-dir") {
            std::fs::create_dir_all(dir)?;
            std::fs::write(
                format!("{dir}/stage{s}of{}.rct", models.len()),
                blobs[s].bytes(),
            )?;
        }
    }
    let mut text = vec![0u8; got_ids.len() * 32 + 1];
    let len = whole.decode(&got_ids, &mut text);
    println!("text: {}", String::from_utf8_lossy(&text[..len]));
    println!("cuts {cuts:?}");
    ensure!(
        got_ids == want_ids && got_hash == want_hash,
        "pipeline differs from the whole model (ids {} vs {}, encoder fnv {got_hash:#010x} vs {want_hash:#010x})",
        got_ids.len(),
        want_ids.len()
    );
    println!(
        "pipeline == whole model: same {} ids, same encoder bits (fnv {got_hash:#010x})",
        got_ids.len()
    );
    Ok(())
}
