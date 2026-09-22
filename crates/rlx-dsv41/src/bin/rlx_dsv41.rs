// RLX — versatile ML compiler + runtime. GPLv3.
//! `rlx-dsv41` binary entry point.

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    rlx_dsv41::cli::run(&args)
}
