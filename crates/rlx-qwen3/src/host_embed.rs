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

//! Token embeddings gathered on the host from the mmapped GGUF (or read by
//! row from a safetensors checkpoint), for the native packed CPU path
//! (`RLX_QWEN3_HOST_EMBED`).
//!
//! The in-graph embedding is a gather over an F32 copy of `token_embd`
//! (593 MB for Qwen3-0.6B), and every compiled graph — the prefill seed and
//! each decode bucket — holds its own copy to look up one row per decoded
//! token. Here the row is dequantized straight from the packed bytes instead,
//! and the graph takes it as `inputs_embeds`. Same dequantization as the
//! F32 table, so the values are identical.

use anyhow::{Context, Result, ensure};
use rlx_gguf::{GgufFile, GgufTensor};

const TOKEN_EMBD: &str = "token_embd.weight";

pub(crate) struct HostEmbed {
    src: Source,
    hidden: usize,
    vocab: usize,
    row_bytes: usize,
}

enum Source {
    Gguf {
        file: GgufFile,
        tensor: GgufTensor,
    },
    /// `model.embed_tokens.weight` in a safetensors file: rows are read
    /// straight from the file at `offset`, in a float dtype.
    Safetensors {
        file: std::fs::File,
        offset: u64,
        dtype: rlx_ir::DType,
    },
}

const ST_EMBED: &str = "model.embed_tokens.weight";

impl HostEmbed {
    /// `None` when the checkpoint has no `[vocab, hidden]` table this can
    /// slice by row; the caller keeps the in-graph gather then. A `.gguf`
    /// path reads `token_embd`; anything else is a safetensors file or
    /// (sharded) directory.
    pub(crate) fn open(path: &str, hidden: usize) -> Result<Option<Self>> {
        if !path.ends_with(".gguf") {
            return Self::open_safetensors(std::path::Path::new(path), hidden);
        }
        let gguf = path;
        let file = GgufFile::from_path_mmap(gguf).with_context(|| format!("mmap {gguf}"))?;
        let Some(tensor) = file.get(TOKEN_EMBD).cloned() else {
            return Ok(None);
        };
        // GGML order: innermost first.
        let (&[h, vocab], Ok(bytes)) = (tensor.shape.as_slice(), file.tensor_bytes(&tensor)) else {
            return Ok(None);
        };
        if h != hidden || vocab == 0 || !bytes.len().is_multiple_of(vocab) {
            return Ok(None);
        }
        let row_bytes = bytes.len() / vocab;
        Ok(Some(Self {
            src: Source::Gguf { file, tensor },
            hidden,
            vocab,
            row_bytes,
        }))
    }

    fn open_safetensors(path: &std::path::Path, hidden: usize) -> Result<Option<Self>> {
        // A sharded checkpoint names the table's shard in its index.
        let file_path = if path.is_dir() {
            let index = path.join("model.safetensors.index.json");
            match std::fs::read(&index) {
                Ok(raw) => {
                    let v: serde_json::Value =
                        serde_json::from_slice(&raw).with_context(|| format!("parse {index:?}"))?;
                    match v["weight_map"][ST_EMBED].as_str() {
                        Some(shard) => path.join(shard),
                        None => return Ok(None),
                    }
                }
                Err(_) => path.join("model.safetensors"),
            }
        } else {
            path.to_path_buf()
        };
        let mut file =
            std::fs::File::open(&file_path).with_context(|| format!("open {file_path:?}"))?;
        use std::io::Read;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let n = u64::from_le_bytes(len) as usize;
        let mut header = vec![0u8; n];
        file.read_exact(&mut header)?;
        let h: serde_json::Value = serde_json::from_slice(&header)
            .with_context(|| format!("safetensors header of {file_path:?}"))?;
        let t = &h[ST_EMBED];
        let dtype = match t["dtype"].as_str() {
            Some("BF16") => rlx_ir::DType::BF16,
            Some("F16") => rlx_ir::DType::F16,
            Some("F32") => rlx_ir::DType::F32,
            _ => return Ok(None),
        };
        let shape: Vec<usize> = t["shape"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|d| d.as_u64())
                    .map(|d| d as usize)
                    .collect()
            })
            .unwrap_or_default();
        let (&[vocab, h_dim], Some(start)) = (shape.as_slice(), t["data_offsets"][0].as_u64())
        else {
            return Ok(None);
        };
        if h_dim != hidden || vocab == 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            src: Source::Safetensors {
                file,
                offset: 8 + n as u64 + start,
                dtype,
            },
            hidden,
            vocab,
            row_bytes: hidden * dtype.size_bytes(),
        }))
    }

    /// Rows for `ids` (token ids as f32, the generator's input encoding),
    /// concatenated: `[ids.len(), hidden]`.
    pub(crate) fn rows(&self, ids: &[f32]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(ids.len() * self.hidden);
        let mut buf = vec![0u8; self.row_bytes];
        for &id in ids {
            let id = id as usize;
            ensure!(
                id < self.vocab,
                "token id {id} outside vocab {}",
                self.vocab
            );
            match &self.src {
                Source::Gguf { file, tensor } => {
                    let bytes = file.tensor_bytes(tensor)?;
                    let row = &bytes[id * self.row_bytes..(id + 1) * self.row_bytes];
                    out.extend(rlx_gguf::dequant_typed(
                        tensor.dtype,
                        row,
                        self.hidden,
                        TOKEN_EMBD,
                    )?);
                }
                Source::Safetensors {
                    file,
                    offset,
                    dtype,
                } => {
                    read_at(file, &mut buf, offset + (id * self.row_bytes) as u64)?;
                    out.extend(rlx_runtime::backend::widen_bytes_to_f32(&buf, *dtype));
                }
            }
        }
        Ok(out)
    }
}

/// Fill `buf` from `file` at `offset` without moving a shared cursor.
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)?;
    #[cfg(windows)]
    {
        let mut done = 0;
        while done < buf.len() {
            let n = std::os::windows::fs::FileExt::seek_read(
                file,
                &mut buf[done..],
                offset + done as u64,
            )?;
            ensure!(n > 0, "short read of the embedding table");
            done += n;
        }
    }
    Ok(())
}

impl std::fmt::Debug for HostEmbed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dtype = match &self.src {
            Source::Gguf { tensor, .. } => format!("{:?}", tensor.dtype),
            Source::Safetensors { dtype, .. } => format!("{dtype:?}"),
        };
        f.debug_struct("HostEmbed")
            .field("dtype", &dtype)
            .field("vocab", &self.vocab)
            .field("hidden", &self.hidden)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_gguf::{GgmlType, GgufWriter};

    /// Rows gathered from the packed bytes must equal the same rows of the
    /// whole-table dequant the in-graph gather reads — exactly, since both
    /// run the same block decoder.
    #[test]
    fn host_rows_match_full_table_dequant() {
        let (hidden, vocab) = (512, 7);
        let table: Vec<f32> = (0..hidden * vocab)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect();
        let mut w = GgufWriter::new();
        w.add_tensor_bytes(
            TOKEN_EMBD,
            vec![hidden, vocab],
            GgmlType::Q4K,
            rlx_gguf::quantize(&table, GgmlType::Q4K).unwrap(),
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!("rlx-host-embed-{}.gguf", std::process::id()));
        w.write_to_path(&path).unwrap();
        let path = path.to_str().unwrap();

        let he = HostEmbed::open(path, hidden)
            .unwrap()
            .expect("sliceable token_embd");
        let (full, _) = GgufFile::from_path_mmap(path)
            .unwrap()
            .dequant_f32(TOKEN_EMBD)
            .unwrap();
        let got = he.rows(&[3.0, 0.0, 6.0]).unwrap();
        for (i, id) in [3usize, 0, 6].into_iter().enumerate() {
            assert_eq!(
                got[i * hidden..(i + 1) * hidden],
                full[id * hidden..(id + 1) * hidden],
                "row {id}"
            );
        }
        assert!(he.rows(&[vocab as f32]).is_err(), "id past the vocab");
        assert!(
            HostEmbed::open(path, 256).unwrap().is_none(),
            "wrong hidden"
        );
        let _ = std::fs::remove_file(path);
    }

    /// A safetensors table (bf16, as HF checkpoints store it) read by row,
    /// directly or through a sharded directory's index, gives the rows the
    /// whole-table read gives.
    #[test]
    fn safetensors_rows_match_the_table() {
        use safetensors::tensor::{Dtype, TensorView};
        let (hidden, vocab) = (24, 9);
        let bits: Vec<u16> = (0..hidden * vocab)
            .map(|i| ((((i * 37 % 101) as f32 - 50.0) * 0.013).to_bits() >> 16) as u16)
            .collect();
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let pad = vec![0u8; 40];
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path().join("model-00002-of-00002.safetensors");
        safetensors::serialize_to_file(
            [
                ("a.pad", TensorView::new(Dtype::U8, vec![40], &pad).unwrap()),
                (
                    ST_EMBED,
                    TensorView::new(Dtype::BF16, vec![vocab, hidden], &bytes).unwrap(),
                ),
            ],
            None,
            &shard,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            format!(r#"{{"weight_map": {{"{ST_EMBED}": "model-00002-of-00002.safetensors"}}}}"#),
        )
        .unwrap();
        let want = |id: usize| -> Vec<f32> {
            bits[id * hidden..(id + 1) * hidden]
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect()
        };
        for path in [dir.path().to_str().unwrap(), shard.to_str().unwrap()] {
            let he = HostEmbed::open(path, hidden)
                .unwrap()
                .expect("a bf16 table");
            let got = he.rows(&[4.0, 0.0, 8.0]).unwrap();
            assert_eq!(got, [want(4), want(0), want(8)].concat(), "{path}");
            assert!(he.rows(&[vocab as f32]).is_err(), "id past the vocab");
            assert!(HostEmbed::open(path, 8).unwrap().is_none(), "wrong hidden");
        }
    }
}
