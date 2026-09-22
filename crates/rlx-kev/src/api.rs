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

//! TypeSafe-compatible `POST /v1/systemone` request / response shapes, mapped
//! onto the single pointer primitive.
//!
//! Mirrors `kev/api.py`:
//!
//! | type     | options fed to the model          | answer |
//! |----------|-----------------------------------|---|
//! | `noul`   | `[false-text, true-text]`         | `p(true)` |
//! | `choice` | `name` or `name: desc`, in order  | argmax + per-name probabilities |
//! | `score`  | ordered level descriptions        | expected level index + legend |
//!
//! Two details are load-bearing and easy to get wrong:
//!
//! * **Option order is part of the request.** Options interact before the
//!   softmax (that is the architecture, not a bug), so `criteria` must keep
//!   its JSON insertion order. `serde_json::Map` only does that with the
//!   `preserve_order` feature, which this workspace does not enable — hence
//!   [`OrderedMap`].
//! * **`render` is a text protocol**, not a debug format. It has to match the
//!   Python character for character or the model sees a different prompt than
//!   it was trained on. See [`render`] for the two places Rust and Python
//!   disagree about how a scalar prints.

use std::fmt::Write as _;

use anyhow::{Result, bail};
use serde::Deserialize;
use serde::de::{Deserializer, MapAccess, Visitor};
use serde_json::Value;

/// Upper bound on options per question (`kev/api.py::MAX_OPTIONS`).
pub const MAX_OPTIONS: usize = 255;

/// An insertion-ordered JSON object.
///
/// `choice` criteria are ordered and the order changes the answer, so we
/// cannot round-trip them through a hash map.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OrderedMap(pub Vec<(String, Value)>);

impl OrderedMap {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(k, _)| k.as_str())
    }
}

impl<'de> Deserialize<'de> for OrderedMap {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OrderedMap;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut m: M,
            ) -> std::result::Result<OrderedMap, M::Error> {
                let mut out = Vec::with_capacity(m.size_hint().unwrap_or(0));
                while let Some((k, v)) = m.next_entry::<String, Value>()? {
                    out.push((k, v));
                }
                Ok(OrderedMap(out))
            }
        }
        d.deserialize_map(V)
    }
}

/// One typed question.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Yes/no. `criteria` optionally describes the `true` / `false` sides.
    Noul {
        instructions: Value,
        #[serde(default)]
        criteria: Option<OrderedMap>,
    },
    /// Named options, each with a description or `null`.
    Choice {
        instructions: Value,
        criteria: OrderedMap,
    },
    /// Ordered levels, lowest first.
    Score {
        instructions: Value,
        criteria: Vec<Value>,
    },
}

impl Question {
    pub fn type_name(&self) -> &'static str {
        match self {
            Question::Noul { .. } => "noul",
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
        }
    }

    fn validate(&self, id: &str) -> Result<()> {
        match self {
            Question::Noul { .. } => Ok(()),
            Question::Choice { criteria, .. } => {
                if !(1..=MAX_OPTIONS).contains(&criteria.len()) {
                    bail!("question {id:?}: criteria must have 1..{MAX_OPTIONS} options");
                }
                Ok(())
            }
            Question::Score { criteria, .. } => {
                if !(2..=MAX_OPTIONS).contains(&criteria.len()) {
                    bail!("question {id:?}: score criteria must have 2..{MAX_OPTIONS} levels");
                }
                Ok(())
            }
        }
    }
}

/// `POST /v1/systemone` body.
#[derive(Debug, Clone, Deserialize)]
pub struct SystemOneRequest {
    /// The content to evaluate: string, object or array.
    pub state: Value,
    #[serde(default = "default_model")]
    pub model: String,
    /// Question id → question. Ids are yours; the model never sees them.
    pub questions: OrderedMap,
}

fn default_model() -> String {
    "kev-latest".into()
}

impl SystemOneRequest {
    /// Parse `questions` into typed values, preserving request order.
    pub fn typed_questions(&self) -> Result<Vec<(String, Question)>> {
        if self.questions.is_empty() {
            bail!("questions must not be empty");
        }
        let mut out = Vec::with_capacity(self.questions.len());
        for (id, raw) in &self.questions.0 {
            let q: Question = serde_json::from_value(raw.clone())
                .map_err(|e| anyhow::anyhow!("question {id:?}: {e}"))?;
            q.validate(id)?;
            out.push((id.clone(), q));
        }
        Ok(out)
    }
}

/// Flatten `string | object | array` into the text the model sees.
///
/// Field names are kept as labels. This has to agree with Python's `str()`
/// on scalars, which differs from Rust's `Display` in exactly two places:
///
/// * `bool` prints `True` / `False`, not `true` / `false`;
/// * a float that happens to be integral prints `3.0`, not `3`. JSON `3` and
///   JSON `3.0` are therefore *different* strings, matching Python's
///   `json.loads` int/float split, which `serde_json::Number` also keeps.
///
/// Very large or very small floats still print in Rust's positional form
/// where Python would use exponent form (`1e+20`); no training record hits
/// that, but it is a known difference rather than an oversight.
pub fn render(v: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Number(n) => render_number(n),
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(|x| {
                let inner = render(x, indent + 1);
                format!("{pad}- {}", inner.trim_start())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(map) => {
            let mut parts = Vec::with_capacity(map.len());
            for (k, x) in map {
                if matches!(x, Value::Object(_) | Value::Array(_)) {
                    parts.push(format!("{pad}{k}:\n{}", render(x, indent + 1)));
                } else {
                    parts.push(format!("{pad}{k}: {}", render(x, 0)));
                }
            }
            parts.join("\n")
        }
    }
}

fn render_number(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    match n.as_f64() {
        Some(f) if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 => format!("{f:.1}"),
        Some(f) => {
            let mut s = String::new();
            let _ = write!(s, "{f}");
            s
        }
        None => n.to_string(),
    }
}

/// `name` alone, or `name: description` when a description is present.
///
/// An empty-string description counts as absent, exactly as in the Python.
pub fn option_text(name: &str, desc: Option<&Value>) -> String {
    match desc {
        None | Some(Value::Null) => name.to_string(),
        Some(Value::String(s)) if s.is_empty() => name.to_string(),
        Some(d) => format!("{name}: {}", render(d, 0)),
    }
}

/// The keys a question's probabilities are reported under, in option order.
pub fn question_keys(q: &Question) -> Vec<String> {
    match q {
        Question::Choice { criteria, .. } => criteria.keys().map(str::to_string).collect(),
        Question::Noul { .. } => vec!["false".into(), "true".into()],
        Question::Score { criteria, .. } => (0..criteria.len()).map(|i| i.to_string()).collect(),
    }
}

/// One question as the encoder sees it: instructions plus rendered options.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordQuestion {
    pub instr: String,
    pub options: Vec<String>,
    /// Ground-truth option index. `0` for inference; set for training.
    pub label: usize,
}

/// One request flattened into the internal record the encoder packs.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub state: String,
    pub questions: Vec<RecordQuestion>,
}

/// Per-question bookkeeping needed to map probabilities back to the response.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionMeta {
    pub id: String,
    pub qtype: String,
    pub keys: Vec<String>,
    /// `score` only: key → level description.
    pub legend: Option<Vec<(String, String)>>,
}

/// Flatten a request into `(record, per-question metadata)`.
pub fn to_record(req: &SystemOneRequest) -> Result<(Record, Vec<QuestionMeta>)> {
    let typed = req.typed_questions()?;
    let mut qs = Vec::with_capacity(typed.len());
    let mut meta = Vec::with_capacity(typed.len());
    for (id, q) in &typed {
        let keys = question_keys(q);
        let (instr, options, legend) = match q {
            Question::Noul {
                instructions,
                criteria,
            } => {
                let c = criteria.clone().unwrap_or_default();
                let opts = vec![
                    option_text("no", c.get("false")),
                    option_text("yes", c.get("true")),
                ];
                (render(instructions, 0), opts, None)
            }
            Question::Choice {
                instructions,
                criteria,
            } => {
                let opts = criteria
                    .0
                    .iter()
                    .map(|(k, v)| option_text(k, Some(v)))
                    .collect();
                (render(instructions, 0), opts, None)
            }
            Question::Score {
                instructions,
                criteria,
            } => {
                let opts: Vec<String> = criteria.iter().map(|x| render(x, 0)).collect();
                let legend = keys.iter().cloned().zip(opts.iter().cloned()).collect();
                (render(instructions, 0), opts, Some(legend))
            }
        };
        qs.push(RecordQuestion {
            instr,
            options,
            label: 0,
        });
        meta.push(QuestionMeta {
            id: id.clone(),
            qtype: q.type_name().to_string(),
            keys,
            legend,
        });
    }
    Ok((
        Record {
            state: render(&req.state, 0),
            questions: qs,
        },
        meta,
    ))
}

/// `(p_max − 1/K) / (1 − 1/K)`; a single option is trivially confident.
pub fn choice_confidence(p: &[f32]) -> f32 {
    let k = p.len() as f32;
    if p.len() == 1 {
        return 1.0;
    }
    let pmax = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (pmax - 1.0 / k) / (1.0 - 1.0 / k)
}

/// `1 − E|level − mode| / (L − 1)`.
///
/// An approximation of TypeSafe's distance-from-the-modal-level statistic,
/// whose exact formula is unpublished — so this is the number kev reports,
/// not a claim about Jev.
pub fn score_confidence(p: &[f32]) -> f32 {
    let l = p.len();
    if l < 2 {
        return 1.0;
    }
    let mode = argmax(p);
    let e: f32 = p
        .iter()
        .enumerate()
        .map(|(i, pi)| pi * (i as f32 - mode as f32).abs())
        .sum();
    1.0 - e / (l - 1) as f32
}

/// First index of the maximum — ties go to the lowest index, as `max()` does
/// in the Python. Tie-breaking direction matters: an MLX `TopK` that broke
/// ties toward the *largest* index has bitten this codebase before.
pub fn argmax(p: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, v) in p.iter().enumerate().skip(1) {
        if *v > p[best] {
            best = i;
        }
    }
    best
}

/// Round to 2 decimals the way Python's `round()` does — ties to even.
///
/// `round(0.125, 2)` is `0.12` in Python and `0.13` under the naive
/// multiply-round-divide, so this is not cosmetic.
pub fn r2(x: f32) -> f32 {
    let scaled = (x as f64) * 100.0;
    (scaled.round_ties_even() / 100.0) as f32
}

/// One answered question.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    Noul {
        noul: f32,
    },
    Choice {
        choice: String,
        confidence: f32,
        probabilities: Vec<(String, f32)>,
    },
    Score {
        score: f32,
        confidence: f32,
        legend: Vec<(String, String)>,
        probabilities: Vec<(String, f32)>,
    },
}

impl Answer {
    /// The response-body JSON for this answer.
    pub fn to_json(&self) -> Value {
        let obj = |pairs: Vec<(String, Value)>| {
            Value::Object(pairs.into_iter().collect::<serde_json::Map<_, _>>())
        };
        let probs = |p: &Vec<(String, f32)>| {
            obj(p
                .iter()
                .map(|(k, v)| (k.clone(), json_f32(*v)))
                .collect::<Vec<_>>())
        };
        match self {
            Answer::Noul { noul } => obj(vec![
                ("type".into(), Value::String("noul".into())),
                ("noul".into(), json_f32(*noul)),
            ]),
            Answer::Choice {
                choice,
                confidence,
                probabilities,
            } => obj(vec![
                ("type".into(), Value::String("choice".into())),
                ("choice".into(), Value::String(choice.clone())),
                ("confidence".into(), json_f32(*confidence)),
                ("probabilities".into(), probs(probabilities)),
            ]),
            Answer::Score {
                score,
                confidence,
                legend,
                probabilities,
            } => obj(vec![
                ("type".into(), Value::String("score".into())),
                ("score".into(), json_f32(*score)),
                (
                    "legend".into(),
                    obj(legend
                        .iter()
                        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                        .collect::<Vec<_>>()),
                ),
                ("probabilities".into(), probs(probabilities)),
                ("confidence".into(), json_f32(*confidence)),
            ]),
        }
    }
}

fn json_f32(v: f32) -> Value {
    serde_json::Number::from_f64(v as f64).map_or(Value::Null, Value::Number)
}

/// Turn per-question probability vectors into typed answers.
pub fn to_answers(probs: &[Vec<f32>], meta: &[QuestionMeta]) -> Result<Vec<(String, Answer)>> {
    if probs.len() != meta.len() {
        bail!(
            "got {} probability vectors for {} questions",
            probs.len(),
            meta.len()
        );
    }
    let mut out = Vec::with_capacity(meta.len());
    for (p, m) in probs.iter().zip(meta) {
        if p.len() != m.keys.len() {
            bail!(
                "question {:?}: {} probabilities for {} options",
                m.id,
                p.len(),
                m.keys.len()
            );
        }
        let answer = match m.qtype.as_str() {
            "noul" => Answer::Noul { noul: r2(p[1]) },
            "choice" => Answer::Choice {
                choice: m.keys[argmax(p)].clone(),
                confidence: r2(choice_confidence(p)),
                probabilities: m.keys.iter().cloned().zip(p.iter().map(|v| r2(*v))).collect(),
            },
            _ => {
                let score: f32 = p.iter().enumerate().map(|(i, pi)| i as f32 * pi).sum();
                Answer::Score {
                    score: r2(score),
                    confidence: r2(score_confidence(p)),
                    legend: m.legend.clone().unwrap_or_default(),
                    probabilities: m
                        .keys
                        .iter()
                        .cloned()
                        .zip(p.iter().map(|v| r2(*v)))
                        .collect(),
                }
            }
        };
        out.push((m.id.clone(), answer));
    }
    Ok(out)
}

/// The `answers` object of the response body.
pub fn answers_json(answers: &[(String, Answer)]) -> Value {
    Value::Object(
        answers
            .iter()
            .map(|(id, a)| (id.clone(), a.to_json()))
            .collect::<serde_json::Map<_, _>>(),
    )
}
