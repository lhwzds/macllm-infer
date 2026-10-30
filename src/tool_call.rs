//! Native single tool-call syntax shared by the pinned Qwen agent profiles.
//!
//! Qwen3 emits one function call as
//! `<tool_call>\n{"name": "<name>", "arguments": <object>}\n</tool_call>`,
//! optionally after a leading `<think>…</think>` reasoning block and visible
//! text. The prompt renderer, the lazy sampling constraint and the HTTP
//! response parser all use this module, so they cannot disagree about where a
//! model-chosen call starts or how its envelope is spelled. Qwen3.5 shares
//! the scanner and streaming splitter, but uses raw XML parameter blocks
//! parsed by [`parse_xml_tool_call`] before schema-directed conversion.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

/// Opening tag of one native call.
pub const TOOL_CALL_OPEN: &str = "<tool_call>";
/// Closing tag of one native call.
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";
/// Bytes that close the arguments envelope and the call.
pub const TOOL_CALL_TAIL: &str = "}\n</tool_call>";

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Canonical bytes between [`TOOL_CALL_OPEN`] and one function's arguments,
/// using the spacing of the official template and its system instructions.
///
/// `encoded_name` must already be a JSON string literal.
#[must_use]
pub fn tool_call_head(encoded_name: &str) -> String {
    format!("\n{{\"name\": {encoded_name}, \"arguments\": ")
}

/// Where a model-chosen call begins in generated assistant text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallScan {
    /// No call has opened. The first `stable` bytes can never become part of
    /// an opening tag or of the whitespace trimmed before one, so they are
    /// safe to stream as content.
    Text { stable: usize },
    /// The opening tag starts at byte `open`. Content is `text[..content_end]`:
    /// everything before the tag except the whitespace separating the two.
    Call { content_end: usize, open: usize },
}

/// Locate the first native call in generated text.
///
/// A leading `<think>` block is reasoning, not an answer: the model may
/// discuss the call syntax there, so the opening tag is recognized only after
/// that block closes. This mirrors the history renderer, which treats text
/// before `</think>` as reasoning. While the output is still undecided (only
/// whitespace, or a prefix of `<think>`), nothing is stable.
#[must_use]
pub fn scan_tool_call(text: &str) -> ToolCallScan {
    let leading = text.len() - text.trim_start().len();
    let rest = &text[leading..];
    let armed_from = if let Some(reasoning) = rest.strip_prefix(THINK_OPEN) {
        match reasoning.find(THINK_CLOSE) {
            Some(end) => leading + THINK_OPEN.len() + end + THINK_CLOSE.len(),
            None => return ToolCallScan::Text { stable: text.len() },
        }
    } else if THINK_OPEN.starts_with(rest) {
        return ToolCallScan::Text { stable: 0 };
    } else {
        0
    };

    let armed = &text[armed_from..];
    if let Some(offset) = armed.find(TOOL_CALL_OPEN) {
        let open = armed_from + offset;
        return ToolCallScan::Call {
            content_end: text[..open].trim_end().len(),
            open,
        };
    }
    // Hold back a suffix that can still grow into the opening tag, together
    // with the whitespace that would be trimmed before it.
    let partial = (1..TOOL_CALL_OPEN.len())
        .rev()
        .find(|&length| armed.ends_with(&TOOL_CALL_OPEN[..length]))
        .unwrap_or(0);
    let held_from = text.len() - partial;
    ToolCallScan::Text {
        stable: armed_from + text[armed_from..held_from].trim_end().len(),
    }
}

/// One parsed call whose name and arguments are not yet checked against the
/// declared tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeToolCall {
    pub name: String,
    pub arguments: Value,
    pub arguments_text: String,
}

/// Parse the text that followed [`TOOL_CALL_OPEN`]: exactly one
/// `{"name": …, "arguments": {…}}` object closed by [`TOOL_CALL_CLOSE`] and
/// nothing after it. The lazy grammar admits only EOS after the closing tag,
/// so trailing text means the call was not produced under that constraint.
pub fn parse_tool_call(after_open: &str) -> Result<NativeToolCall> {
    let body = after_open
        .strip_suffix(TOOL_CALL_CLOSE)
        .context("tool call is not terminated by its closing tag")?;
    let value: Value =
        serde_json::from_str(body.trim()).context("tool call body is not one JSON object")?;
    let object = value
        .as_object()
        .context("tool call body is not one JSON object")?;
    ensure!(
        object.len() == 2,
        "tool call body must contain exactly `name` and `arguments`"
    );
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .context("tool call name must be a string")?;
    let arguments = object
        .get("arguments")
        .filter(|arguments| arguments.is_object())
        .context("tool call arguments must be a JSON object")?;
    let raw: std::collections::HashMap<String, Box<serde_json::value::RawValue>> =
        serde_json::from_str(body.trim())?;
    Ok(NativeToolCall {
        name: name.to_owned(),
        arguments: arguments.clone(),
        arguments_text: raw["arguments"].get().to_owned(),
    })
}

/// Native Qwen3-Coder/Qwen3.5 XML envelope, before schema-directed conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XmlToolCall {
    pub name: String,
    pub parameters: Vec<(String, String)>,
}

/// Parse exactly one Qwen3.5 function block. Values are raw text, not XML:
/// entities are not decoded and only the newline-prefixed parameter closing
/// tag terminates a value. Schema-directed conversion is done by the caller.
pub fn parse_xml_tool_call(after_open: &str) -> Result<XmlToolCall> {
    let rest = after_open
        .strip_prefix("\n<function=")
        .context("tool call must open a function block")?;
    let (name, mut rest) = rest
        .split_once(">\n")
        .context("unterminated function name")?;
    let mut parameters = Vec::new();
    while let Some(parameter) = rest.strip_prefix("<parameter=") {
        let (name, value) = parameter
            .split_once(">\n")
            .context("unterminated parameter name")?;
        let (value, tail) = value
            .split_once("\n</parameter>\n")
            .context("unterminated parameter value")?;
        ensure!(
            !parameters.iter().any(|(seen, _)| seen == name),
            "repeated parameter {name:?}"
        );
        parameters.push((name.to_owned(), value.to_owned()));
        rest = tail;
    }
    ensure!(
        rest == "</function>\n</tool_call>",
        "tool call must end after one closed function block"
    );
    Ok(XmlToolCall {
        name: name.to_owned(),
        parameters,
    })
}

/// How a split stream ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitEnd {
    /// No call opened; the remaining held-back text is ordinary content.
    Content(String),
    /// A call opened; this is the text after its opening tag.
    Call(String),
}

/// Incrementally separates streamed content from a model-chosen call.
///
/// Concatenating every [`Self::push`] result with a final
/// [`SplitEnd::Content`] reproduces the generated text exactly. When a call
/// opens, the streamed content equals the non-streaming content defined by
/// [`ToolCallScan::Call`], and no byte of the opening tag is ever released.
#[derive(Debug, Clone, Default)]
pub struct ToolCallSplitter {
    text: String,
    emitted: usize,
    open: Option<usize>,
}

impl ToolCallSplitter {
    /// Append one decoded fragment and return the content that is now safe
    /// to stream.
    pub fn push(&mut self, fragment: &str) -> String {
        self.text.push_str(fragment);
        if self.open.is_some() {
            return String::new();
        }
        let end = match scan_tool_call(&self.text) {
            ToolCallScan::Text { stable } => stable,
            ToolCallScan::Call { content_end, open } => {
                self.open = Some(open);
                content_end
            }
        };
        if end <= self.emitted {
            return String::new();
        }
        let safe = self.text[self.emitted..end].to_owned();
        self.emitted = end;
        safe
    }

    /// Complete the stream after generation finished.
    #[must_use]
    pub fn finish(self) -> SplitEnd {
        match self.open {
            Some(open) => SplitEnd::Call(self.text[open + TOOL_CALL_OPEN.len()..].to_owned()),
            None => SplitEnd::Content(self.text[self.emitted..].to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const CALL: &str =
        "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>";

    #[test]
    fn scan_finds_calls_after_text_or_reasoning_and_trims_the_separator() {
        assert_eq!(
            scan_tool_call(CALL),
            ToolCallScan::Call {
                content_end: 0,
                open: 0
            }
        );
        let text = format!("Let me look.\n{CALL}");
        assert_eq!(
            scan_tool_call(&text),
            ToolCallScan::Call {
                content_end: "Let me look.".len(),
                open: "Let me look.\n".len()
            }
        );
        let reasoning = "<think>\nI need <tool_call> syntax.\n</think>\n\n";
        let text = format!("{reasoning}{CALL}");
        assert_eq!(
            scan_tool_call(&text),
            ToolCallScan::Call {
                content_end: reasoning.trim_end().len(),
                open: reasoning.len()
            }
        );
    }

    #[test]
    fn scan_never_triggers_inside_an_open_reasoning_block() {
        let text = "<think>\nmaybe <tool_call>\n{";
        assert_eq!(
            scan_tool_call(text),
            ToolCallScan::Text { stable: text.len() }
        );
    }

    #[test]
    fn scan_holds_undecided_prefixes_and_separator_whitespace() {
        for undecided in ["", "\n", "<", "<th", " <think"] {
            assert_eq!(
                scan_tool_call(undecided),
                ToolCallScan::Text { stable: 0 },
                "{undecided:?}"
            );
        }
        assert_eq!(
            scan_tool_call("<to"),
            ToolCallScan::Text { stable: 0 },
            "a prefix of the call tag is held once it cannot open reasoning"
        );
        assert_eq!(
            scan_tool_call("Answer \n<tool_c"),
            ToolCallScan::Text { stable: 6 }
        );
        assert_eq!(
            scan_tool_call("Answer \n\n"),
            ToolCallScan::Text { stable: 6 }
        );
        assert_eq!(scan_tool_call("a < b"), ToolCallScan::Text { stable: 5 });
        let reasoning = "<think>\nr\n</think>\n\n";
        assert_eq!(
            scan_tool_call(reasoning),
            ToolCallScan::Text {
                stable: reasoning.trim_end().len()
            }
        );
    }

    #[test]
    fn parses_only_one_closed_envelope() {
        let after_open = &CALL[TOOL_CALL_OPEN.len()..];
        assert_eq!(
            parse_tool_call(after_open).unwrap(),
            NativeToolCall {
                arguments_text: r#"{"command": "ls"}"#.to_owned(),
                name: "shell".to_owned(),
                arguments: json!({"command": "ls"})
            }
        );
        for invalid in [
            "\n{\"name\": \"shell\", \"arguments\": {}}\n",
            "\n{\"name\": \"shell\", \"arguments\": {}}\n</tool_call>\n",
            "\n{\"name\": \"shell\", \"arguments\": []}\n</tool_call>",
            "\n{\"name\": 1, \"arguments\": {}}\n</tool_call>",
            "\n{\"name\": \"shell\", \"arguments\": {}, \"id\": 1}\n</tool_call>",
            "\n{\"name\": \"shell\"}\n</tool_call>",
            "\nnot json\n</tool_call>",
        ] {
            assert!(parse_tool_call(invalid).is_err(), "{invalid:?}");
        }
    }

    fn split(fragments: &[&str]) -> (String, SplitEnd) {
        let mut splitter = ToolCallSplitter::default();
        let mut streamed = String::new();
        for fragment in fragments {
            streamed.push_str(&splitter.push(fragment));
        }
        (streamed, splitter.finish())
    }

    #[test]
    fn splitter_never_streams_any_part_of_a_split_opening_tag() {
        let fragments = [
            "<think>\nok\n</think>",
            "\n\nI will",
            " run it.\n<",
            "tool",
            "_call>\n{\"name\": \"shell\", ",
            "\"arguments\": {\"command\": \"ls\"}}\n</tool_call>",
        ];
        let (streamed, end) = split(&fragments);
        assert_eq!(streamed, "<think>\nok\n</think>\n\nI will run it.");
        let text = fragments.concat();
        let ToolCallScan::Call { content_end, open } = scan_tool_call(&text) else {
            panic!("expected a call");
        };
        assert_eq!(streamed, text[..content_end]);
        assert_eq!(
            end,
            SplitEnd::Call(text[open + TOOL_CALL_OPEN.len()..].to_owned())
        );
    }

    #[test]
    fn splitter_releases_held_text_when_no_call_opens() {
        let fragments = ["Use a ", "<", "tool", " tag ", "\n", "<tool_c"];
        let (streamed, end) = split(&fragments);
        assert_eq!(streamed, "Use a <tool tag");
        assert_eq!(end, SplitEnd::Content(" \n<tool_c".to_owned()));

        // Every byte-level split of a call yields the same content and call.
        let text = format!("Done?\n{CALL}");
        for cut in 1..text.len() {
            let (streamed, end) = split(&[&text[..cut], &text[cut..]]);
            assert_eq!(streamed, "Done?", "cut at {cut}");
            assert_eq!(end, SplitEnd::Call(CALL[TOOL_CALL_OPEN.len()..].to_owned()));
        }
    }
}
