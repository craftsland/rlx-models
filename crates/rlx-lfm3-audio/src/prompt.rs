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

//! Prompts of the model card's tasks, as the chat template renders them.
//!
//! `encode` turns template text (special tokens included) into ids; with
//! the `tokenizer` feature `text::Tokenizer::encode` does it.

use anyhow::Result;

use crate::config::Tokens;
use crate::model::Segment;

/// TTS language codes of the GGUF's `tts` template.
pub const TTS_LANGUAGES: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "American English"),
    ("en-us", "American English"),
    ("en-gb", "British English"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("he", "Hebrew"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("es", "Spanish"),
    ("sv", "Swedish"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
];

/// A language code (`en`, `en-gb`, …) as the TTS system prompt names it;
/// anything else is used as given.
pub fn tts_language(code: &str) -> String {
    let lower = code.to_ascii_lowercase();
    TTS_LANGUAGES
        .iter()
        .find(|(c, _)| *c == lower)
        .map(|(_, n)| n.to_string())
        .unwrap_or_else(|| code.to_string())
}

/// `Perform ASR in <language>.` (or `Perform ASR.` without one) over a clip
/// of `audio` encoder embeddings; the reply is text.
pub fn asr(
    tokens: &Tokens,
    encode: &dyn Fn(&str) -> Result<Vec<u32>>,
    language: Option<&str>,
    audio: Vec<f32>,
) -> Result<Vec<Segment>> {
    let system = match language {
        Some(l) => format!("Perform ASR in {l}."),
        None => "Perform ASR.".to_string(),
    };
    let mut head = vec![tokens.bos];
    head.extend(encode(&format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
    ))?);
    Ok(vec![
        Segment::Tokens(head),
        Segment::Audio(audio),
        Segment::Tokens(encode("<|im_end|>\n<|im_start|>assistant\n")?),
    ])
}

/// `Perform TTS in <language> using a <voice> voice.`, the assistant turn
/// prefilled with `<|audio_start|>`; the reply is speech.
pub fn tts(
    tokens: &Tokens,
    encode: &dyn Fn(&str) -> Result<Vec<u32>>,
    text: &str,
    language: &str,
    voice: &str,
) -> Result<Vec<Segment>> {
    let voice = match voice.to_ascii_lowercase().as_str() {
        "male" => "male",
        _ => "female",
    };
    let mut ids = vec![tokens.bos];
    ids.extend(encode(&format!(
        "<|im_start|>system\nPerform TTS in {} using a {voice} voice.<|im_end|>\n\
         <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<|audio_start|>",
        tts_language(language)
    ))?);
    Ok(vec![Segment::Tokens(ids)])
}

/// One user turn (text and/or a clip) under `system`, the assistant turn
/// opened and, when `prefill` is set, started with it (`<|mixed_start|>` for
/// an interleaved spoken reply).
pub fn chat(
    tokens: &Tokens,
    encode: &dyn Fn(&str) -> Result<Vec<u32>>,
    system: Option<&str>,
    user_text: Option<&str>,
    user_audio: Option<Vec<f32>>,
    prefill: Option<&str>,
) -> Result<Vec<Segment>> {
    let mut head = vec![tokens.bos];
    if let Some(s) = system {
        head.extend(encode(&format!("<|im_start|>system\n{s}<|im_end|>\n"))?);
    }
    head.extend(encode("<|im_start|>user\n")?);
    let mut segs = vec![Segment::Tokens(head)];
    if let Some(a) = user_audio {
        segs.push(Segment::Audio(a));
    }
    let mut tail = String::new();
    if let Some(t) = user_text {
        tail.push_str(t);
    }
    tail.push_str("<|im_end|>\n<|im_start|>assistant\n");
    if let Some(p) = prefill {
        tail.push_str(p);
    }
    segs.push(Segment::Tokens(encode(&tail)?));
    Ok(segs)
}
