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

//! Chat-template rendering and prompt-id assembly.
//!
//! Ports `JINA_OCR_CHAT_TEMPLATE` (`chat_template.jinja` /
//! `processing_deepseek_ocr.py`) and `preprocess_prompt_and_image`.
//!
//! Two details are easy to get wrong and silent when wrong:
//!
//! * **No BOS.** `text_encode(..., bos=False)` is what the processor calls, and
//!   `tokenizer_config.json` sets `add_bos_token: false` — the id sequence
//!   starts at `<|User|>:`. Unlimited-OCR's own assembly *does* prepend BOS, so
//!   this crate builds its own ids instead of reusing that path.
//! * **Image tokens replace the `<image>` marker positionally.** The marker is
//!   split out of the rendered text, each side is tokenized independently, and
//!   the placeholder run is spliced between them.

use anyhow::{Result, bail, ensure};

/// `DEFAULT_OCR_PROMPT` in `processing_deepseek_ocr.py`.
pub const DEFAULT_OCR_PROMPT: &str = "Transcribe the provided document image into a clean Markdown format, preserving the natural reading order.";

/// The literal the processor splits on (`DS_OCR_IMG_TOKEN`).
pub const IMAGE_MARKER: &str = "<image>";

const USER_PREFIX: &str = "<|User|>:\n";
const ASSISTANT_PREFIX: &str = "<|Assistant|>:\n";

/// One piece of a message body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Content {
    /// Renders as the literal `<image>` marker.
    Image,
    Text(String),
}

impl Content {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn prefix(self) -> &'static str {
        match self {
            Role::User => USER_PREFIX,
            Role::Assistant => ASSISTANT_PREFIX,
            // The template capitalizes unknown roles; System is lifted out
            // before the loop and never reaches this path.
            Role::System => "<|System|>:\n",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

impl Message {
    pub fn user(content: Vec<Content>) -> Self {
        Self {
            role: Role::User,
            content,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![Content::text(text)],
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::text(text)],
        }
    }
}

/// The single-image OCR conversation `prepare_ocr_inputs` builds.
pub fn ocr_conversation(prompt: &str) -> Vec<Message> {
    vec![Message::user(vec![Content::Image, Content::text(prompt)])]
}

/// Render `messages` through the jina chat template.
///
/// Mirrors the Jinja exactly: a leading system message is lifted out and
/// emitted bare (plus a newline), each remaining message gets its role prefix,
/// text parts are `rstrip`ped, a text part directly after an image part is
/// preceded by a newline, messages are joined by newlines, and
/// `add_generation_prompt` appends `\n<|Assistant|>:\n`.
pub fn apply_chat_template(messages: &[Message], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    let mut rest = messages;

    if let Some(first) = messages.first()
        && first.role == Role::System
    {
        let system = render_plain(&first.content);
        out.push_str(&system);
        if !system.is_empty() {
            out.push('\n');
        }
        rest = &messages[1..];
    }

    for (i, msg) in rest.iter().enumerate() {
        let mut prefix_printed = false;
        let mut last_was_image = false;
        for part in &msg.content {
            let is_image = matches!(part, Content::Image);
            let body = match part {
                Content::Image => IMAGE_MARKER.to_string(),
                Content::Text(t) => t.trim_end().to_string(),
            };
            if last_was_image && !is_image {
                out.push('\n');
            }
            if !prefix_printed && (msg.role == Role::User || !is_image) {
                out.push_str(msg.role.prefix());
                prefix_printed = true;
            }
            out.push_str(&body);
            last_was_image = is_image;
        }
        if i + 1 < rest.len() {
            out.push('\n');
        }
    }

    if add_generation_prompt {
        if !rest.is_empty() {
            out.push('\n');
        }
        out.push_str(ASSISTANT_PREFIX);
    }
    out
}

/// A `content is string` message body: no role prefix handling, no image parts.
fn render_plain(content: &[Content]) -> String {
    content
        .iter()
        .map(|c| match c {
            Content::Image => IMAGE_MARKER.to_string(),
            Content::Text(t) => t.trim_end().to_string(),
        })
        .collect::<Vec<_>>()
        .join("")
}

/// The rendered single-image OCR prompt, ready for [`build_prompt_ids`].
pub fn ocr_prompt_text(prompt: &str) -> String {
    apply_chat_template(&ocr_conversation(prompt), true)
}

/// Splice `image_token_count` placeholder ids into `text` at its single
/// `<image>` marker, tokenizing each side with `encode`.
///
/// `encode` must not add special tokens — the processor calls
/// `tokenizer.encode(text, add_special_tokens=False)` and prepends no BOS.
pub fn build_prompt_ids(
    text: &str,
    image_token_count: usize,
    image_token_id: u32,
    mut encode: impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<Vec<u32>> {
    let chunks: Vec<&str> = text.split(IMAGE_MARKER).collect();
    ensure!(
        chunks.len() == 2,
        "expected exactly one {IMAGE_MARKER} placeholder, found {}",
        chunks.len() - 1
    );
    if image_token_count == 0 {
        bail!("image expands to zero placeholder tokens");
    }
    let pre = encode(chunks[0])?;
    let post = encode(chunks[1])?;

    let mut ids = Vec::with_capacity(pre.len() + image_token_count + post.len());
    ids.extend_from_slice(&pre);
    ids.extend(std::iter::repeat_n(image_token_id, image_token_count));
    ids.extend_from_slice(&post);
    Ok(ids)
}

/// `[image_token_id; n]` positions in `ids`, for asserting prompt layout.
pub fn image_token_positions(ids: &[u32], image_token_id: u32) -> Vec<usize> {
    ids.iter()
        .enumerate()
        .filter(|&(_, &t)| t == image_token_id)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::IMAGE_TOKEN_ID;

    /// The exact string `prepare_ocr_inputs` produces for the default prompt.
    #[test]
    fn ocr_prompt_matches_reference_rendering() {
        let text = ocr_prompt_text(DEFAULT_OCR_PROMPT);
        assert_eq!(
            text,
            format!("<|User|>:\n<image>\n{DEFAULT_OCR_PROMPT}\n<|Assistant|>:\n")
        );
    }

    #[test]
    fn text_after_image_gets_a_newline_but_image_does_not() {
        let msgs = vec![Message::user(vec![
            Content::Image,
            Content::Image,
            Content::text("hi"),
        ])];
        assert_eq!(
            apply_chat_template(&msgs, false),
            "<|User|>:\n<image><image>\nhi"
        );
    }

    #[test]
    fn system_message_is_lifted_out_without_a_role_prefix() {
        let msgs = vec![
            Message::system("You are terse."),
            Message::user(vec![Content::Image, Content::text("go")]),
        ];
        assert_eq!(
            apply_chat_template(&msgs, true),
            "You are terse.\n<|User|>:\n<image>\ngo\n<|Assistant|>:\n"
        );
    }

    #[test]
    fn messages_are_joined_by_newlines() {
        let msgs = vec![
            Message::user(vec![Content::text("q")]),
            Message::assistant("a"),
        ];
        assert_eq!(
            apply_chat_template(&msgs, false),
            "<|User|>:\nq\n<|Assistant|>:\na"
        );
    }

    #[test]
    fn trailing_whitespace_in_text_is_stripped() {
        let msgs = vec![Message::user(vec![Content::text("padded   \n\n")])];
        assert_eq!(apply_chat_template(&msgs, false), "<|User|>:\npadded");
    }

    /// No BOS: the id stream starts with the `<|User|>:` chunk's own tokens.
    #[test]
    fn prompt_ids_have_no_bos_and_splice_image_run() {
        let ids = build_prompt_ids(
            "<|User|>:\n<image>\nGO\n<|Assistant|>:\n",
            4,
            IMAGE_TOKEN_ID,
            |chunk| {
                Ok(match chunk {
                    "<|User|>:\n" => vec![10, 11],
                    "\nGO\n<|Assistant|>:\n" => vec![20, 21, 22],
                    other => panic!("unexpected chunk {other:?}"),
                })
            },
        )
        .expect("build");
        assert_eq!(
            ids,
            vec![
                10,
                11,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                20,
                21,
                22
            ]
        );
        assert_ne!(ids[0], crate::config::BOS_TOKEN_ID);
        assert_eq!(
            image_token_positions(&ids, IMAGE_TOKEN_ID),
            vec![2, 3, 4, 5]
        );
    }

    #[test]
    fn prompt_ids_reject_wrong_marker_count() {
        let enc = |_: &str| Ok(vec![0u32]);
        assert!(build_prompt_ids("no marker", 1, IMAGE_TOKEN_ID, enc).is_err());
        assert!(build_prompt_ids("<image><image>", 1, IMAGE_TOKEN_ID, enc).is_err());
    }

    #[test]
    fn prompt_ids_reject_empty_image_run() {
        assert!(build_prompt_ids("a<image>b", 0, IMAGE_TOKEN_ID, |_| Ok(vec![1])).is_err());
    }
}
