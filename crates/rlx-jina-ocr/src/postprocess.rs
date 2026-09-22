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

//! Output post-processing — `decode_ocr`, `parse_refs`,
//! `extract_markdown_and_crops`, `normalize_markdown_images_to_ref_det`.
//!
//! The model emits grounded spans as
//! `<|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>`, with coordinates
//! normalized into 1000 bins (`0..=999`) against the *original* image size.
//! A span whose label is not a known layout element is an **image**: the label
//! becomes markdown alt text and the box becomes a crop. Known element labels
//! (`title`, `text`, …) are dropped from the rendered markdown.
//!
//! The reference uses two regexes; both are scanned by hand here so the crate
//! stays dependency-free. Each scan documents the backtracking behaviour it
//! reproduces.

/// `<|ref|>` / `<|/ref|>` / `<|det|>` / `<|/det|>` literals.
const REF_OPEN: &str = "<|ref|>";
const REF_CLOSE: &str = "<|/ref|>";
const DET_OPEN: &str = "<|det|>";
const DET_CLOSE: &str = "<|/det|>";

/// Special-token surface forms stripped by `decode_ocr`.
const BOS_TEXT: &str = "<｜begin▁of▁sentence｜>";
const EOS_TEXT: &str = "<｜end▁of▁sentence｜>";
const PAD_TEXT: &str = "<｜▁pad▁｜>";
const IMAGE_TEXT: &str = "<image>";

/// `KNOWN_ELEMENT_TYPES` — labels that name a layout element rather than an
/// image. Compared lowercased.
pub const KNOWN_ELEMENT_TYPES: [&str; 24] = [
    "title",
    "text",
    "paragraph",
    "button",
    "link",
    "icon",
    "header",
    "footer",
    "table",
    "list",
    "code",
    "formula",
    "caption",
    "label",
    "input",
    "checkbox",
    "radio",
    "dropdown",
    "menu",
    "navigation",
    "sidebar",
    "logo",
    "banner",
    "card",
];

/// A grounded span parsed out of the model output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ref {
    /// Text between `<|ref|>` and `<|/ref|>`.
    pub label: String,
    /// Raw text between `<|det|>` and `<|/det|>`.
    pub coords_literal: String,
    /// `[x1, y1, x2, y2]` boxes in 1000-bin space; empty when unparseable.
    pub coords: Vec<[i32; 4]>,
    /// Byte range of the whole `<|ref|>…<|/det|>` span in the source text.
    pub span: (usize, usize),
    /// Label is *not* a known element type, so this span is an image.
    pub is_image: bool,
}

/// `is_image_label(label)`.
pub fn is_image_label(label: &str) -> bool {
    let lower = label.to_lowercase();
    !KNOWN_ELEMENT_TYPES.contains(&lower.as_str())
}

/// `decode_ocr` — trim the prompt echo's special tokens off a decoded string.
///
/// The reference decodes with `skip_special_tokens=False` (an EOS/image-only
/// continuation would otherwise decode to an empty string and lose the
/// diagnostic) and then removes the EOS/PAD/BOS/`<image>` surface forms.
pub fn clean_decoded_text(text: &str) -> String {
    let mut out = text.to_string();
    for tok in [EOS_TEXT, PAD_TEXT, BOS_TEXT, IMAGE_TEXT] {
        out = out.replace(tok, "");
    }
    out.trim().to_string()
}

/// `parse_coords` — `"[[x1,y1,x2,y2], …]"` to boxes; `[]` on any parse failure.
///
/// Values are truncated toward zero, matching Python's `int()` on the floats a
/// model sometimes emits.
pub fn parse_coords(literal: &str) -> Vec<[i32; 4]> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(literal) else {
        return Vec::new();
    };
    let Some(rows) = value.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let Some(items) = row.as_array() else {
            return Vec::new();
        };
        if items.len() != 4 {
            return Vec::new();
        }
        let mut box_ = [0i32; 4];
        for (slot, item) in box_.iter_mut().zip(items) {
            let Some(n) = item.as_f64() else {
                return Vec::new();
            };
            *slot = n.trunc() as i32;
        }
        out.push(box_);
    }
    out
}

/// `parse_refs` — every `<|ref|>…<|/ref|><|det|>…<|/det|>` span, in order.
///
/// Reproduces the regex's leftmost-match-with-lazy-groups behaviour: from each
/// `<|ref|>`, the label ends at the first `<|/ref|>` that is *immediately*
/// followed by `<|det|>`; if no such close tag exists the whole attempt fails
/// and scanning resumes after that `<|ref|>`.
pub fn parse_refs(text: &str) -> Vec<Ref> {
    let mut refs = Vec::new();
    let mut cursor = 0usize;
    while let Some(rel) = text[cursor..].find(REF_OPEN) {
        let start = cursor + rel;
        let label_from = start + REF_OPEN.len();
        let mut matched = false;
        let mut search = label_from;
        while let Some(rel_close) = text[search..].find(REF_CLOSE) {
            let close = search + rel_close;
            let after = close + REF_CLOSE.len();
            if !text[after..].starts_with(DET_OPEN) {
                // Lazy group keeps growing past this close tag.
                search = close + REF_CLOSE.len();
                continue;
            }
            let det_from = after + DET_OPEN.len();
            let Some(rel_det_close) = text[det_from..].find(DET_CLOSE) else {
                break;
            };
            let det_close = det_from + rel_det_close;
            let end = det_close + DET_CLOSE.len();
            let label = text[label_from..close].to_string();
            let coords_literal = text[det_from..det_close].to_string();
            refs.push(Ref {
                coords: parse_coords(&coords_literal),
                is_image: is_image_label(&label),
                label,
                coords_literal,
                span: (start, end),
            });
            cursor = end;
            matched = true;
            break;
        }
        if !matched {
            cursor = start + REF_OPEN.len();
        }
    }
    refs
}

/// `normalize_to_pixel` — 1000-bin coordinate to pixels (`int(v / 999 * dim)`).
pub fn normalize_to_pixel(x: i32, y: i32, width: u32, height: u32) -> (i32, i32) {
    (
        (f64::from(x) / 999.0 * f64::from(width)) as i32,
        (f64::from(y) / 999.0 * f64::from(height)) as i32,
    )
}

/// `coords_to_filename` — `"x1_y1_x2_y2.png"` in 1000-bin space.
pub fn coords_to_filename(b: [i32; 4], ext: &str) -> String {
    format!("{}_{}_{}_{}.{ext}", b[0], b[1], b[2], b[3])
}

/// One image region the output grounded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Crop {
    /// Alt text (the ref label).
    pub alt_text: String,
    /// `x1_y1_x2_y2.png` in 1000-bin space.
    pub filename: String,
    /// Pixel rect in the original image: `[x1, y1, x2, y2]`, `x2 > x1`, `y2 > y1`.
    pub pixel_box: [i32; 4],
    /// 1000-bin rect as emitted by the model.
    pub norm_box: [i32; 4],
}

/// `extract_markdown_and_crops` — render grounded output as markdown.
///
/// Image refs become `![alt](images/x1_y1_x2_y2.png)` and yield a [`Crop`];
/// element refs are dropped. `cleanup` applies the reference's LaTeX
/// substitutions. `width`/`height` are the *original* image dimensions.
pub fn extract_markdown_and_crops(
    text: &str,
    width: u32,
    height: u32,
    cleanup: bool,
) -> (String, Vec<Crop>) {
    let mut md = String::with_capacity(text.len());
    let mut crops = Vec::new();
    let mut last = 0usize;

    for r in parse_refs(text) {
        let (start, end) = r.span;
        md.push_str(&text[last..start]);
        if r.is_image {
            for b in &r.coords {
                let (x1, y1) = normalize_to_pixel(b[0], b[1], width, height);
                let (x2, y2) = normalize_to_pixel(b[2], b[3], width, height);
                if x2 > x1 && y2 > y1 {
                    md.push_str(&format!(
                        "![{}](images/{})",
                        r.label,
                        coords_to_filename(*b, "png")
                    ));
                    crops.push(Crop {
                        alt_text: r.label.clone(),
                        filename: coords_to_filename(*b, "png"),
                        pixel_box: [x1, y1, x2, y2],
                        norm_box: *b,
                    });
                }
            }
        }
        last = end;
    }
    md.push_str(&text[last..]);

    if cleanup {
        md = md.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:");
    }
    (md, crops)
}

/// `normalize_markdown_images_to_ref_det` — rewrite
/// `![alt](…/x1_y1_x2_y2.png)` in an *input* prompt as a ref/det span, so a
/// grounded re-OCR request reaches the model in the form it was trained on.
///
/// The reference regex is
/// `!\[([^\]]*)\]\([^)]*?(\d+)_(\d+)_(\d+)_(\d+)\.(?:png|jpg|jpeg)\)`. Because
/// the four numbers are pinned to the extension, they are always the last four
/// underscore-separated digit runs of the parenthesised path — which is what
/// this scan takes, greedily (so `a/12_3_4_5_6.png` reads `3_4_5_6`, exactly as
/// the lazy prefix plus greedy `\d+` would).
pub fn normalize_markdown_images_to_ref_det(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    while let Some(rel) = text[cursor..].find("![") {
        let start = cursor + rel;
        out.push_str(&text[cursor..start]);
        let alt_from = start + 2;
        let Some(rel_bracket) = text[alt_from..].find(']') else {
            out.push_str(&text[start..]);
            return out;
        };
        let bracket = alt_from + rel_bracket;
        let alt = &text[alt_from..bracket];
        let after = bracket + 1;
        // `[^\]]*` cannot span a `]`, and `\(` must follow immediately.
        if !text[after..].starts_with('(') {
            out.push_str(&text[start..after]);
            cursor = after;
            continue;
        }
        let body_from = after + 1;
        let Some(rel_paren) = text[body_from..].find(')') else {
            out.push_str(&text[start..]);
            return out;
        };
        let paren = body_from + rel_paren;
        let body = &text[body_from..paren];
        match split_coord_path(body) {
            Some(b) => {
                let label = if alt.trim().is_empty() {
                    "image"
                } else {
                    alt.trim()
                };
                out.push_str(&format!(
                    "{REF_OPEN}{label}{REF_CLOSE}{DET_OPEN}[[{},{},{},{}]]{DET_CLOSE}",
                    b[0], b[1], b[2], b[3]
                ));
                cursor = paren + 1;
            }
            None => {
                out.push_str(&text[start..body_from]);
                cursor = body_from;
            }
        }
    }
    out.push_str(&text[cursor..]);
    out
}

/// `…/x1_y1_x2_y2.(png|jpg|jpeg)` to its four numbers.
///
/// `y1`/`x2`/`y2` are whole underscore-separated segments. `x1` is only the
/// *trailing* digit run of the segment before them — the lazy prefix eats
/// whatever comes first, so `images/100_200_300_400` reads `100` while
/// `a/12_3_4_5_6` reads `3` (the prefix takes `a/12_`).
fn split_coord_path(body: &str) -> Option<[i32; 4]> {
    let stem = ["png", "jpg", "jpeg"]
        .iter()
        .find_map(|ext| body.strip_suffix(&format!(".{ext}")))?;
    let parts: Vec<&str> = stem.rsplitn(5, '_').collect();
    if parts.len() < 4 {
        return None;
    }
    let mut nums = [0i32; 4];
    // parts[0..3] are y2, x2, y1 (right to left) — each a whole segment.
    for (slot, part) in nums[1..].iter_mut().rev().zip(&parts[..3]) {
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        *slot = part.parse().ok()?;
    }
    let head = parts[3];
    let digits_from = head
        .bytes()
        .rposition(|b| !b.is_ascii_digit())
        .map_or(0, |i| i + 1);
    let x1 = &head[digits_from..];
    if x1.is_empty() {
        return None;
    }
    nums[0] = x1.parse().ok()?;
    Some(nums)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_decoded_text_strips_specials_and_trims() {
        let raw = format!("  {BOS_TEXT}hello {IMAGE_TEXT}world{EOS_TEXT}{PAD_TEXT}  ");
        assert_eq!(clean_decoded_text(&raw), "hello world");
    }

    #[test]
    fn clean_decoded_text_of_eos_only_is_empty() {
        assert_eq!(clean_decoded_text(EOS_TEXT), "");
    }

    #[test]
    fn parse_refs_reads_label_and_boxes() {
        let text = "a<|ref|>title<|/ref|><|det|>[[10,20,30,40]]<|/det|>b";
        let refs = parse_refs(text);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].label, "title");
        assert_eq!(refs[0].coords, vec![[10, 20, 30, 40]]);
        assert!(!refs[0].is_image, "`title` is a known element type");
        assert_eq!(
            &text[refs[0].span.0..refs[0].span.1],
            &text[1..text.len() - 1]
        );
    }

    #[test]
    fn parse_refs_handles_multiple_boxes_and_unknown_label() {
        let refs = parse_refs("<|ref|>A bar chart<|/ref|><|det|>[[1,2,3,4],[5,6,7,8]]<|/det|>");
        assert_eq!(refs.len(), 1);
        assert!(refs[0].is_image);
        assert_eq!(refs[0].coords, vec![[1, 2, 3, 4], [5, 6, 7, 8]]);
    }

    /// The lazy label group grows past a `<|/ref|>` not followed by `<|det|>`.
    #[test]
    fn parse_refs_skips_unpaired_close_tag() {
        let refs = parse_refs("<|ref|>a<|/ref|>b<|/ref|><|det|>[[1,2,3,4]]<|/det|>");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].label, "a<|/ref|>b");
    }

    #[test]
    fn parse_refs_ignores_unterminated_span() {
        assert!(parse_refs("<|ref|>dangling").is_empty());
        assert!(parse_refs("<|ref|>x<|/ref|><|det|>[[1,2,3,4]]").is_empty());
    }

    #[test]
    fn parse_coords_tolerates_garbage() {
        assert!(parse_coords("not json").is_empty());
        assert!(parse_coords("[[1,2,3]]").is_empty());
        assert_eq!(parse_coords("[[1.9,2,3,4]]"), vec![[1, 2, 3, 4]]);
    }

    #[test]
    fn normalize_to_pixel_uses_999_bins() {
        assert_eq!(normalize_to_pixel(0, 0, 800, 600), (0, 0));
        assert_eq!(normalize_to_pixel(999, 999, 800, 600), (800, 600));
        assert_eq!(normalize_to_pixel(500, 500, 1000, 1000), (500, 500));
    }

    #[test]
    fn markdown_keeps_images_and_drops_elements() {
        let text = "Intro\n<|ref|>title<|/ref|><|det|>[[0,0,999,100]]<|/det|>Body\
                    <|ref|>A chart<|/ref|><|det|>[[100,200,300,400]]<|/det|>End";
        let (md, crops) = extract_markdown_and_crops(text, 1000, 1000, true);
        assert_eq!(md, "Intro\nBody![A chart](images/100_200_300_400.png)End");
        assert_eq!(crops.len(), 1);
        assert_eq!(crops[0].alt_text, "A chart");
        assert_eq!(crops[0].filename, "100_200_300_400.png");
        assert_eq!(crops[0].pixel_box, [100, 200, 300, 400]);
    }

    /// A degenerate box yields neither markdown nor a crop.
    #[test]
    fn markdown_skips_empty_boxes() {
        let (md, crops) = extract_markdown_and_crops(
            "<|ref|>pic<|/ref|><|det|>[[50,50,50,50]]<|/det|>",
            100,
            100,
            false,
        );
        assert_eq!(md, "");
        assert!(crops.is_empty());
    }

    #[test]
    fn markdown_cleanup_replaces_latex_aliases() {
        let (md, _) = extract_markdown_and_crops("a \\coloneqq b \\eqqcolon c", 10, 10, true);
        assert_eq!(md, "a := b =: c");
        let (raw, _) = extract_markdown_and_crops("a \\coloneqq b", 10, 10, false);
        assert_eq!(raw, "a \\coloneqq b");
    }

    #[test]
    fn markdown_image_syntax_becomes_ref_det() {
        assert_eq!(
            normalize_markdown_images_to_ref_det("x ![A chart](images/100_200_300_400.png) y"),
            "x <|ref|>A chart<|/ref|><|det|>[[100,200,300,400]]<|/det|> y"
        );
    }

    #[test]
    fn empty_alt_text_defaults_to_image() {
        assert_eq!(
            normalize_markdown_images_to_ref_det("![](1_2_3_4.jpg)"),
            "<|ref|>image<|/ref|><|det|>[[1,2,3,4]]<|/det|>"
        );
    }

    /// Greedy `\d+` takes the whole final digit run of each field.
    #[test]
    fn coord_path_takes_last_four_groups() {
        assert_eq!(split_coord_path("a/12_3_4_5_6.png"), Some([3, 4, 5, 6]));
        assert_eq!(
            split_coord_path("images/100_200_300_400.jpeg"),
            Some([100, 200, 300, 400])
        );
        assert_eq!(split_coord_path("100_200_300_400.gif"), None);
        assert_eq!(split_coord_path("1_2_3.png"), None);
        assert_eq!(split_coord_path("a_b_c_d.png"), None);
    }

    #[test]
    fn non_coordinate_markdown_images_pass_through() {
        let text = "![alt](https://example.com/pic.png) tail";
        assert_eq!(normalize_markdown_images_to_ref_det(text), text);
    }

    #[test]
    fn plain_prompts_are_untouched() {
        let p = crate::prompt::DEFAULT_OCR_PROMPT;
        assert_eq!(normalize_markdown_images_to_ref_det(p), p);
    }
}
