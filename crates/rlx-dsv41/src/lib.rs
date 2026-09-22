// RLX — versatile ML compiler + runtime. GPLv3.
//! **DeepSeek-V4.1-Flash** command line.
//!
//! The model itself — the graphs, the runner, the expert pager — lives in
//! `rlx-models-core`; this is only the CLI around it, kept separate because that
//! crate is the one everything else depends on.
//!
//! ```sh
//! rlx-dsv41 --model /path/to/DeepSeek-V4.1-Flash --prompt "hello" --paged
//! ```

pub mod cli;
