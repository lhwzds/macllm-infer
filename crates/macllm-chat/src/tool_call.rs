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

/// Opening tag of one reasoning block.
pub const THINK_OPEN: &str = "<think>";
/// Closing tag of one reasoning block.
pub const THINK_CLOSE: &str = "</think>";

/// What the rendered prompt left open for the model to fill.
///
/// The prompt decides how a turn starts. A prompt that ends inside
/// `<think>` leaves the reasoning block open, so everything the model writes
/// is reasoning until it closes the block. A prompt that ends at the
/// assistant turn lets the model open the block itself: it either emits
/// `<think>` before its answer or answers directly, and an answer that never
/// mentions the tags is ordinary content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PromptThinking {
    /// The prompt did not leave a reasoning block open.
    #[default]
    Closed,
    /// The prompt left the reasoning block open for the model to continue.
    Open,
}

/// The reasoning and visible answer of a generated turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnText<'a> {
    /// Thinking text without tags, or `None` if no thinking block opened.
    pub reasoning: Option<&'a str>,
    /// The visible answer.
    pub content: &'a str,
}

/// Split a complete turn with the same prompt-driven machine used by the
/// streaming server. The returned slices borrow the original generated text.
#[must_use]
pub fn split_turn(text: &str, prompt: PromptThinking) -> TurnText<'_> {
    let mut splitter = TurnSplitter::new(true, prompt, false);
    let mut chunks = splitter.push(text);
    let end = splitter.finish();
    chunks.reasoning.push_str(&end.reasoning);
    chunks.content.push_str(&end.content);
    let reasoning = if end.has_reasoning {
        text.find(&chunks.reasoning)
            .and_then(|start| text.get(start..start + chunks.reasoning.len()))
    } else {
        None
    };
    let content = text
        .rfind(&chunks.content)
        .and_then(|start| text.get(start..start + chunks.content.len()))
        .unwrap_or_default();
    TurnText { reasoning, content }
}

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

    scan_call_from(text, armed_from)
}

fn scan_call_from(text: &str, armed_from: usize) -> ToolCallScan {
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

/// The text one streaming step released.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnChunks {
    /// The reasoning text released by this step.
    pub reasoning: String,
    /// The visible text released by this step.
    pub content: String,
}

/// How a streamed turn ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnEnd {
    /// Whether a reasoning block was opened, even if its text is empty.
    pub has_reasoning: bool,
    /// The reasoning text that was still held back when the turn ended.
    pub reasoning: String,
    /// The visible text that was still held back when the turn ended.
    pub content: String,
    /// The text after the opening tag of a call that opened.
    pub call: Option<String>,
}

/// Which part of a generated turn the splitter is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum TurnPhase {
    /// The turn has not decided whether it opens a reasoning block.
    #[default]
    Start,
    /// The turn is inside a reasoning block.
    Thinking,
    /// The reasoning block closed; the visible answer follows.
    Answer,
}

/// Incrementally separates a generated turn into its reasoning block, its
/// visible answer and at most one model-chosen call.
///
/// With reasoning enabled the turn's thinking block is released as reasoning
/// and never as content: neither tag is ever released. With reasoning
/// disabled the turn keeps the legacy behaviour and every byte is content.
///
/// The prompt decides where the turn starts: a prompt that left the
/// reasoning block open starts inside it, and a prompt that did not starts
/// outside it.
///
/// A closed prompt can still be followed by reasoning without an opening
/// tag. Its prefix is held until a closing tag, a tool-call opener, or the
/// end of generation determines its classification. Once a call starts, its
/// contents are never scanned for reasoning tags.
///
/// A native call is only looked for when the request can open one: a turn
/// that cannot carries no call, and its text is never mistaken for one.
///
/// Concatenating the corresponding chunks and final [`TurnEnd`] yields the
/// same fields regardless of input fragmentation. When a call opens, no byte
/// of its opening tag or arguments is released as content.
#[derive(Debug, Clone, Default)]
pub struct TurnSplitter {
    /// Whether the thinking block is split out of the visible answer.
    split_reasoning: bool,
    /// Whether a native call can open in this turn.
    scan_calls: bool,
    text: String,
    phase: TurnPhase,
    /// The offset in `text` where the visible answer starts.
    content_start: usize,
    /// The offset in `text` up to which the visible answer was released.
    content_emitted: usize,
    /// The offset in `text` where the reasoning starts.
    reasoning_start: usize,
    /// The offset in `text` where the reasoning ends.
    reasoning_end: usize,
    /// The offset in `text` up to which the reasoning was released.
    reasoning_emitted: usize,
    /// Whether a reasoning block was opened.
    has_reasoning: bool,
    /// Whether the reasoning released a character that is not a newline.
    reasoning_started: bool,
    /// Whether the model may still repeat the opening tag.
    skip_open_tag: bool,
    /// The offset in `text` of an opened call.
    open: Option<usize>,
    /// Whether the visible answer starts after a closing tag, whose leading
    /// whitespace is trimmed.
    trim_content: bool,
    /// Whether the visible answer already released a non-whitespace byte.
    content_started: bool,
}

impl TurnSplitter {
    /// A splitter for one generated turn.
    ///
    /// `split_reasoning` releases the thinking block as reasoning instead of
    /// content, `prompt` says whether the rendered prompt left the reasoning
    /// block open, and `scan_calls` says whether a native call can open in
    /// this turn.
    #[must_use]
    pub fn new(split_reasoning: bool, prompt: PromptThinking, scan_calls: bool) -> Self {
        let phase = if !split_reasoning {
            TurnPhase::Answer
        } else if prompt == PromptThinking::Open {
            TurnPhase::Thinking
        } else {
            TurnPhase::Start
        };
        Self {
            split_reasoning,
            scan_calls,
            phase,
            skip_open_tag: prompt == PromptThinking::Open,
            has_reasoning: split_reasoning && prompt == PromptThinking::Open,
            ..Self::default()
        }
    }

    /// Whether the prompt or generated text opened a reasoning block.
    #[must_use]
    pub fn has_reasoning(&self) -> bool {
        self.has_reasoning
    }

    /// Append one decoded fragment and return the text that is now safe to
    /// stream.
    pub fn push(&mut self, fragment: &str) -> TurnChunks {
        let mut chunks = TurnChunks::default();
        self.text.push_str(fragment);
        match self.phase {
            TurnPhase::Start => self.push_start(&mut chunks),
            TurnPhase::Thinking => self.push_thinking(&mut chunks),
            TurnPhase::Answer => self.push_answer(&mut chunks),
        }
        self.discard_released();
        chunks
    }

    /// Discard bytes that have already been classified and emitted. Legacy
    /// mode retains the original text because its scanner recognizes a leading
    /// think block itself; deepseek mode only retains undecided suffixes (or
    /// the complete call once its opener has been recognized).
    fn discard_released(&mut self) {
        if !self.split_reasoning {
            return;
        }
        let discard = match self.phase {
            TurnPhase::Start => 0,
            TurnPhase::Thinking if self.skip_open_tag => 0,
            TurnPhase::Thinking => self.reasoning_emitted,
            TurnPhase::Answer => self.open.unwrap_or(self.content_emitted),
        };
        if discard == 0 {
            return;
        }
        self.text.drain(..discard);
        self.reasoning_start = self.reasoning_start.saturating_sub(discard);
        self.reasoning_end = self.reasoning_end.saturating_sub(discard);
        self.reasoning_emitted = self.reasoning_emitted.saturating_sub(discard);
        self.content_start = self.content_start.saturating_sub(discard);
        self.content_emitted = self.content_emitted.saturating_sub(discard);
        if let Some(open) = &mut self.open {
            *open -= discard;
        }
    }

    /// Complete the turn after generation finished.
    #[must_use]
    pub fn finish(mut self) -> TurnEnd {
        let mut chunks = TurnChunks::default();
        match self.phase {
            TurnPhase::Start => {
                if self.split_reasoning {
                    // The turn never opened a thinking block: the held text
                    // is the visible answer.
                    self.phase = TurnPhase::Answer;
                    self.content_start = 0;
                    self.content_emitted = 0;
                    self.push_answer(&mut chunks);
                }
            }
            TurnPhase::Thinking => {
                // The thinking block never closed: the held text is
                // reasoning and the turn has no visible answer.
                let mut from = self.reasoning_emitted;
                if !self.reasoning_started {
                    // The reasoning trims its leading newlines, however many
                    // tokens they arrive in.
                    let region = &self.text[from..];
                    let trimmed = region.trim_start_matches('\n');
                    from += region.len() - trimmed.len();
                }
                chunks
                    .reasoning
                    .push_str(self.text[from..].trim_end_matches('\n'));
            }
            TurnPhase::Answer => {}
        }
        if self.phase == TurnPhase::Answer && self.open.is_none() {
            let end = self.text.len();
            if self.trim_content && !self.content_started {
                // The visible answer starts after the closing tag: its
                // leading whitespace is trimmed even when the turn ends
                // before any visible byte arrives.
                let region = &self.text[self.content_emitted..end];
                let trimmed = region.trim_start();
                self.content_emitted += region.len() - trimmed.len();
            }
            chunks
                .content
                .push_str(&self.text[self.content_emitted..end]);
        }
        TurnEnd {
            has_reasoning: self.has_reasoning,
            reasoning: chunks.reasoning,
            content: chunks.content,
            call: self
                .open
                .map(|open| self.text[open + TOOL_CALL_OPEN.len()..].to_owned()),
        }
    }

    /// Decide whether the turn opens a thinking block.
    ///
    /// The prompt decides where a reasoning block starts. A prompt that left
    /// the block open starts inside it, so the turn never reaches this
    /// phase. A prompt that did not needs the model to open the block
    /// itself: while the text is still the opening tag the turn is
    /// undecided.
    ///
    fn push_start(&mut self, chunks: &mut TurnChunks) {
        let leading = self.text.len() - self.text.trim_start().len();
        let rest = &self.text[leading..];
        if rest.starts_with(THINK_OPEN) {
            // The turn opens a thinking block. A complete tag is recognized
            // before a partial one, so a turn that ends right after the tag
            // never leaks it.
            self.phase = TurnPhase::Thinking;
            self.has_reasoning = true;
            self.reasoning_start = leading + THINK_OPEN.len();
            self.reasoning_end = self.text.len();
            self.skip_open_tag = false;
            self.push_thinking(chunks);
        } else if !rest.is_empty() && !THINK_OPEN.starts_with(rest) {
            // Closing tags do not open reasoning blocks.
            self.phase = TurnPhase::Answer;
            self.push_answer(chunks);
        }
    }

    /// Release the reasoning of an open thinking block.
    fn push_thinking(&mut self, chunks: &mut TurnChunks) {
        if self.skip_open_tag {
            // The prompt left the reasoning block open, so the model may
            // write the opening tag again, after some whitespace. The tag
            // never reaches the client, exactly like the non-streaming
            // split strips it.
            let region = &self.text[self.reasoning_start..];
            let lead = region.len() - region.trim_start().len();
            let after = &region[lead..];
            if after.is_empty() {
                // Only whitespace so far: the tag may still arrive.
                return;
            }
            if after.starts_with(THINK_OPEN) {
                // The model repeated the opening tag: it never reaches the
                // client.
                self.reasoning_start += lead + THINK_OPEN.len();
                self.skip_open_tag = false;
            } else if THINK_OPEN.starts_with(after) {
                // The opening tag may still arrive: hold the reasoning back.
                return;
            } else {
                // The model did not open the block again: the reasoning
                // starts at the beginning of the region.
                self.skip_open_tag = false;
            }
        }
        let region = &self.text[self.reasoning_start..];
        if let Some(end) = region.find(THINK_CLOSE) {
            self.reasoning_end = self.reasoning_start + end;
            self.phase = TurnPhase::Answer;
            self.content_start = self.reasoning_end + THINK_CLOSE.len();
            self.content_emitted = self.content_start;
            self.trim_content = true;
            self.release_reasoning(chunks, true);
            self.push_answer(chunks);
        } else {
            self.reasoning_end = self.text.len();
            self.release_reasoning(chunks, false);
        }
    }

    /// Release the visible answer of the turn.
    fn push_answer(&mut self, chunks: &mut TurnChunks) {
        if self.open.is_some() {
            // The call is open: the rest of the turn is call text.
            return;
        }
        if self.split_reasoning && self.trim_content && !self.content_started {
            // AFTER_THINK discards leading whitespace as it arrives, even
            // when the call scanner would otherwise hold it as a separator.
            let region = &self.text[self.content_emitted..];
            let leading = region.len() - region.trim_start().len();
            self.content_emitted += leading;
            self.content_start = self.content_emitted;
        }
        if !self.scan_calls {
            // The request cannot open a native call: the answer streams as
            // it arrives and no text is mistaken for a call.
            self.release_content(chunks, self.text.len());
            return;
        }
        let content = &self.text[self.content_start..];
        let scan = if self.split_reasoning {
            scan_call_from(content, 0)
        } else {
            scan_tool_call(content)
        };
        match scan {
            ToolCallScan::Call { content_end, open } => {
                self.open = Some(self.content_start + open);
                self.release_content(chunks, self.content_start + content_end);
            }
            ToolCallScan::Text { stable } => {
                let limit = self.content_start + stable;
                self.release_content(chunks, limit);
            }
        }
    }

    /// Release the reasoning text that is no longer undecided.
    fn release_reasoning(&mut self, chunks: &mut TurnChunks, closed: bool) {
        let start = self.reasoning_start;
        let end = self.reasoning_end;
        let mut from = self.reasoning_emitted.max(start);
        if !self.reasoning_started {
            // The reasoning trims its leading newlines, however many tokens
            // they arrive in: they are only released once a character that
            // is not a newline follows them.
            let region = &self.text[from..end];
            let trimmed = region.trim_start_matches('\n');
            from += region.len() - trimmed.len();
        }
        let mut limit = end;
        if !closed {
            // Hold back a suffix that can still grow into the closing tag.
            let region = &self.text[from..limit];
            let partial = (1..THINK_CLOSE.len())
                .rev()
                .find(|&length| region.ends_with(&THINK_CLOSE[..length]))
                .unwrap_or(0);
            limit -= partial;
        }
        // The reasoning trims its trailing newlines.
        let region = &self.text[from..limit];
        limit -= region.len() - region.trim_end_matches('\n').len();
        if limit > from {
            let released = &self.text[from..limit];
            chunks.reasoning.push_str(released);
            if released.chars().any(|character| character != '\n') {
                self.reasoning_started = true;
            }
        }
        self.reasoning_emitted = limit;
    }

    /// Release visible text up to `end`.
    fn release_content(&mut self, chunks: &mut TurnChunks, end: usize) {
        let mut from = self.content_emitted;
        if self.trim_content && !self.content_started {
            // The visible answer starts after the closing tag: its leading
            // whitespace is trimmed.
            let region = &self.text[from..end];
            let trimmed = region.trim_start();
            if trimmed.is_empty() {
                self.content_emitted = end;
                return;
            }
            from += region.len() - trimmed.len();
            self.content_started = true;
        }
        if end > from {
            chunks.content.push_str(&self.text[from..end]);
            self.content_emitted = end;
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

    /// Every way of chunking a turn produces the same fields, including
    /// tool arguments with tags and ambiguous prefixes at generation end.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn prompt_driven_turn_splitter_is_chunk_invariant() {
        let cases = [
            (
                "<think>\nNeed ls.\n</think>\n\nListing.",
                PromptThinking::Closed,
                false,
                "Need ls.",
                "Listing.",
                None,
            ),
            (
                "Need ls.\n</think>\n\nListing.",
                PromptThinking::Open,
                false,
                "Need ls.",
                "Listing.",
                None,
            ),
            (
                "Listing.",
                PromptThinking::Closed,
                false,
                "",
                "Listing.",
                None,
            ),
            (
                "<think>\nstill thinking",
                PromptThinking::Closed,
                false,
                "still thinking",
                "",
                None,
            ),
            (
                "Need ls.\n</think>\n\nListing.",
                PromptThinking::Closed,
                false,
                "",
                "Need ls.\n</think>\n\nListing.",
                None,
            ),
            (
                "\n<think>reason</think>",
                PromptThinking::Open,
                false,
                "reason",
                "",
                None,
            ),
            (
                "mention <tool_call> here</think>  done",
                PromptThinking::Open,
                true,
                "mention <tool_call> here",
                "done",
                None,
            ),
            (
                "<think>mention <tool_call> here</think>  done",
                PromptThinking::Closed,
                true,
                "mention <tool_call> here",
                "done",
                None,
            ),
            (
                "<think>ok</think>  <tool_call>{\"x\":1}</tool_call>",
                PromptThinking::Closed,
                true,
                "ok",
                "",
                Some("{\"x\":1}</tool_call>"),
            ),
            (
                "Need ls.\n</think>\n<tool_call>{\"command\":\"</think>\"}</tool_call>",
                PromptThinking::Closed,
                true,
                "",
                "Need ls.\n</think>",
                Some("{\"command\":\"</think>\"}</tool_call>"),
            ),
            (
                "Look.\n<tool_call>{\"command\":\"</think>\"}</tool_call>",
                PromptThinking::Closed,
                true,
                "",
                "Look.",
                Some("{\"command\":\"</think>\"}</tool_call>"),
            ),
            (
                "a </think> b",
                PromptThinking::Closed,
                true,
                "",
                "a </think> b",
                None,
            ),
            (
                "answer <tool_call>partial",
                PromptThinking::Closed,
                false,
                "",
                "answer <tool_call>partial",
                None,
            ),
            (
                "prefix  <tool_call>partial",
                PromptThinking::Closed,
                true,
                "",
                "prefix",
                Some("partial"),
            ),
            ("  <thi", PromptThinking::Closed, false, "", "  <thi", None),
            ("  \n", PromptThinking::Closed, false, "", "  \n", None),
            ("  \n", PromptThinking::Closed, true, "", "  \n", None),
            (
                "<think>abc\n\n</think>\t",
                PromptThinking::Closed,
                true,
                "abc",
                "",
                None,
            ),
            (
                "é漢<think>z",
                PromptThinking::Closed,
                true,
                "",
                "é漢<think>z",
                None,
            ),
        ];
        for (text, prompt, tools, expected_reasoning, expected_content, expected_call) in cases {
            if !tools {
                let turn = split_turn(text, prompt);
                assert_eq!(
                    turn.reasoning.unwrap_or_default(),
                    expected_reasoning,
                    "{text:?}"
                );
                assert_eq!(turn.content, expected_content, "{text:?}");
            }
            let cuts: Vec<_> = text
                .char_indices()
                .map(|(index, _)| index)
                .chain([text.len()])
                .collect();
            let run = |parts: &[&str]| {
                let mut splitter = TurnSplitter::new(true, prompt, tools);
                let mut reasoning = String::new();
                let mut content = String::new();
                for part in parts {
                    let chunk = splitter.push(part);
                    reasoning.push_str(&chunk.reasoning);
                    content.push_str(&chunk.content);
                }
                let end = splitter.finish();
                reasoning.push_str(&end.reasoning);
                content.push_str(&end.content);
                assert_eq!(reasoning, expected_reasoning, "{text:?} {parts:?}");
                assert_eq!(content, expected_content, "{text:?} {parts:?}");
                assert_eq!(end.call.as_deref(), expected_call, "{text:?} {parts:?}");
            };
            run(&[text]);
            for &cut in &cuts {
                run(&[&text[..cut], &text[cut..]]);
            }
            let chunks: Vec<_> = cuts
                .windows(2)
                .map(|pair| &text[pair[0]..pair[1]])
                .collect();
            run(&chunks);
        }
    }

    #[test]
    fn empty_reasoning_presence_is_chunk_invariant() {
        for (text, prompt, expected) in [
            ("<think>\n</think>Answer", PromptThinking::Closed, true),
            ("</think>Answer", PromptThinking::Open, true),
            ("<think>", PromptThinking::Closed, true),
            ("Answer", PromptThinking::Closed, false),
        ] {
            for cut in 0..=text.len() {
                let mut splitter = TurnSplitter::new(true, prompt, false);
                splitter.push(&text[..cut]);
                splitter.push(&text[cut..]);
                assert_eq!(splitter.finish().has_reasoning, expected, "{text:?}, {cut}");
            }
            assert_eq!(split_turn(text, prompt).reasoning.is_some(), expected);
        }
    }

    #[test]
    fn ordinary_content_streams_before_finish() {
        for tools in [false, true] {
            let mut splitter = TurnSplitter::new(true, PromptThinking::Closed, tools);
            assert_eq!(splitter.push("Hel").content, "Hel");
            assert_eq!(
                splitter.push("lo ").content,
                if tools { "lo" } else { "lo " }
            );
            assert_eq!(
                splitter.push("world").content,
                if tools { " world" } else { "world" }
            );
            assert_eq!(splitter.finish().content, "");
        }
    }

    #[test]
    fn streamed_deepseek_turn_does_not_retain_already_released_text() {
        for (prompt, fragment, reasoning) in [(PromptThinking::Open, "thinking text", true)] {
            let mut splitter = TurnSplitter::new(true, prompt, true);
            for _ in 0..1000 {
                let chunk = splitter.push(fragment);
                if reasoning {
                    assert!(!chunk.reasoning.is_empty());
                } else {
                    assert!(!chunk.content.is_empty());
                }
                assert!(splitter.text.len() <= TOOL_CALL_OPEN.len());
            }
        }
    }

    #[test]
    fn whitespace_after_thinking_is_discarded_incrementally() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Closed, true);
        let chunk = splitter.push("<think>ok</think>");
        assert_eq!(chunk.reasoning, "ok");
        for _ in 0..1000 {
            let chunk = splitter.push(" \n");
            assert!(chunk.content.is_empty());
            assert!(splitter.text.is_empty());
        }
        assert_eq!(splitter.push("answer").content, "answer");
        assert_eq!(splitter.finish().content, "");
    }

    fn split(fragments: &[&str]) -> (String, TurnEnd) {
        let mut splitter = TurnSplitter::new(false, PromptThinking::Closed, true);
        let mut streamed = String::new();
        for fragment in fragments {
            let chunks = splitter.push(fragment);
            streamed.push_str(&chunks.content);
        }
        (streamed, splitter.finish())
    }

    /// The streamed reasoning keeps its own spacing as well.
    #[test]
    fn the_streamed_reasoning_keeps_its_own_spacing() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Open, false);
        let chunks = splitter.push("  Need ls.");
        assert_eq!(chunks.reasoning, "  Need ls.");
        let end = splitter.finish();
        assert_eq!(end.reasoning, "");
    }

    /// A turn that closes the reasoning block the prompt left open reports
    /// the reasoning and never leaks the closing tag.
    #[test]
    fn a_turn_that_closes_an_open_reasoning_block_reports_it() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Open, false);
        let chunks = splitter.push("Need ls.\n</think>\n\nListing.");
        assert_eq!(chunks.reasoning, "Need ls.");
        assert_eq!(chunks.content, "Listing.");
        let end = splitter.finish();
        assert_eq!(end.reasoning, "");
        assert_eq!(end.content, "");
    }

    /// The closing tag of the reasoning block the prompt left open never
    /// reaches the client, not even when it is split across tokens.
    #[test]
    fn a_split_closing_tag_of_an_open_block_never_leaks() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Open, false);
        let mut reasoning = String::new();
        let mut content = String::new();
        for fragment in ["Need ls.\n</thi", "nk>\n\nListing."] {
            let chunks = splitter.push(fragment);
            reasoning.push_str(&chunks.reasoning);
            content.push_str(&chunks.content);
        }
        let end = splitter.finish();
        reasoning.push_str(&end.reasoning);
        content.push_str(&end.content);
        assert!(!reasoning.contains("</think>"), "{reasoning:?}");
        assert!(!content.contains("</think>"), "{content:?}");
        assert_eq!(reasoning, "Need ls.");
        assert_eq!(content, "Listing.");
    }

    #[test]
    fn reasoning_is_never_released_as_content() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Closed, false);
        let mut reasoning = String::new();
        let mut content = String::new();
        for fragment in ["<think>\nNeed ls.\n</thi", "nk>\n\nListing."] {
            let chunks = splitter.push(fragment);
            reasoning.push_str(&chunks.reasoning);
            content.push_str(&chunks.content);
        }
        let end = splitter.finish();
        reasoning.push_str(&end.reasoning);
        content.push_str(&end.content);
        assert_eq!(reasoning, "Need ls.");
        assert_eq!(content, "Listing.");
        assert!(!reasoning.contains("<think>"));
        assert!(!reasoning.contains("</think>"));
    }

    #[test]
    fn a_missing_opening_tag_still_splits_the_reasoning() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Open, false);
        let mut reasoning = String::new();
        let mut content = String::new();
        for fragment in ["Need ls.\n</thi", "nk>\n\nListing."] {
            let chunks = splitter.push(fragment);
            reasoning.push_str(&chunks.reasoning);
            content.push_str(&chunks.content);
        }
        let end = splitter.finish();
        reasoning.push_str(&end.reasoning);
        content.push_str(&end.content);
        assert_eq!(reasoning, "Need ls.");
        assert_eq!(content, "Listing.");
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
            end.call,
            Some(text[open + TOOL_CALL_OPEN.len()..].to_owned())
        );
        assert_eq!(end.content, "");
    }

    #[test]
    fn splitter_releases_held_text_when_no_call_opens() {
        let fragments = ["Use a ", "<", "tool", " tag ", "\n", "<tool_c"];
        let (streamed, end) = split(&fragments);
        assert_eq!(streamed, "Use a <tool tag");
        assert_eq!(end.content, " \n<tool_c".to_owned());
        assert_eq!(end.call, None);

        // Every byte-level split of a call yields the same content and call.
        let text = format!("Done?\n{CALL}");
        for cut in 1..text.len() {
            let (streamed, end) = split(&[&text[..cut], &text[cut..]]);
            assert_eq!(streamed, "Done?", "cut at {cut}");
            assert_eq!(end.call, Some(CALL[TOOL_CALL_OPEN.len()..].to_owned()));
        }
    }

    /// A prompt that left the thinking block open strips the opening tag
    /// the model writes after some whitespace, exactly like the
    /// non-streaming split does.
    #[test]
    fn an_open_prompt_strips_the_opening_tag_after_whitespace() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Open, false);
        let chunks = splitter.push("\n<think>reason</think>");
        assert_eq!(chunks.reasoning, "reason");
        assert_eq!(chunks.content, "");
        let end = splitter.finish();
        assert_eq!(end.reasoning, "");
        assert_eq!(end.content, "");
    }

    /// The opening tag of a repeated block never reaches the client, not
    /// even when it is split across tokens.
    #[test]
    fn an_open_prompt_never_streams_a_split_opening_tag() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Open, false);
        let chunks = splitter.push("\n<thi");
        assert_eq!(chunks.reasoning, "");
        assert_eq!(chunks.content, "");
        let chunks = splitter.push("nk>reason</think>");
        assert_eq!(chunks.reasoning, "reason");
        assert_eq!(chunks.content, "");
        let end = splitter.finish();
        assert_eq!(end.reasoning, "");
        assert_eq!(end.content, "");
    }

    /// A prompt that left the thinking block open streams its reasoning as
    /// it arrives, without waiting for the closing tag.
    #[test]
    fn an_open_prompt_streams_its_reasoning_while_it_is_generated() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Open, false);
        let chunks = splitter.push("Need ls.");
        assert_eq!(chunks.reasoning, "Need ls.");
        assert_eq!(chunks.content, "");
        let chunks = splitter.push("\n</think>\n\nListing.");
        assert_eq!(chunks.reasoning, "");
        assert_eq!(chunks.content, "Listing.");
        let end = splitter.finish();
        assert_eq!(end.reasoning, "");
        assert_eq!(end.content, "");
    }

    /// A turn that cannot open a call never treats the tag as one.
    #[test]
    fn a_turn_without_calls_keeps_the_tag_as_content() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Closed, false);
        let chunks = splitter.push("A tag. <tool_call>");
        assert_eq!(chunks.content, "A tag. <tool_call>");
        let end = splitter.finish();
        assert_eq!(end.content, "");
        assert_eq!(end.call, None);
    }

    /// A turn that ends right after the opening tag never leaks it.
    #[test]
    fn a_turn_that_ends_after_the_opening_tag_never_leaks_it() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Closed, false);
        let chunks = splitter.push("<think>");
        assert_eq!(chunks.reasoning, "");
        assert_eq!(chunks.content, "");
        let end = splitter.finish();
        assert_eq!(end.reasoning, "");
        assert_eq!(end.content, "");
    }

    /// Leading newlines of the reasoning are trimmed however many tokens they
    /// arrive in.
    #[test]
    fn the_leading_newlines_of_the_reasoning_are_trimmed_across_tokens() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Closed, false);
        let mut reasoning = String::new();
        for fragment in ["<think>", "\n", "\n", "Need ls."] {
            let chunks = splitter.push(fragment);
            reasoning.push_str(&chunks.reasoning);
        }
        let end = splitter.finish();
        reasoning.push_str(&end.reasoning);
        assert_eq!(reasoning, "Need ls.");
    }

    /// The reasoning never starts with a newline, however the newlines of the
    /// opening tag arrive.
    #[test]
    fn the_reasoning_never_starts_with_a_newline() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Closed, true);
        let mut reasoning = String::new();
        for fragment in ["<think>", "\n", "\n", "\n", "Need ls."] {
            let chunks = splitter.push(fragment);
            reasoning.push_str(&chunks.reasoning);
            assert!(!chunks.reasoning.starts_with('\n'), "{fragment:?}");
        }
        let end = splitter.finish();
        reasoning.push_str(&end.reasoning);
        assert_eq!(reasoning, "Need ls.");
    }

    #[test]
    fn whitespace_after_the_closing_tag_is_not_content() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Closed, false);
        let mut content = String::new();
        for fragment in ["<think>\nNeed ls.\n</think>\n", "   "] {
            let chunks = splitter.push(fragment);
            content.push_str(&chunks.content);
        }
        let end = splitter.finish();
        content.push_str(&end.content);
        assert_eq!(content, "");
    }

    /// A turn that scans for calls still trims the whitespace that follows
    /// the closing tag: the answer of such a turn is empty, not the
    /// whitespace the model wrote.
    #[test]
    fn whitespace_after_the_closing_tag_is_not_content_when_calls_are_scanned() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Closed, true);
        let mut content = String::new();
        for fragment in ["<think>\nNeed ls.\n</think>\n", "   "] {
            let chunks = splitter.push(fragment);
            content.push_str(&chunks.content);
        }
        let end = splitter.finish();
        content.push_str(&end.content);
        assert_eq!(content, "");
    }

    /// The whitespace that follows the closing tag is trimmed even when the
    /// whole turn arrives in one piece.
    #[test]
    fn a_whole_turn_of_whitespace_after_the_closing_tag_is_empty() {
        let mut splitter = TurnSplitter::new(true, PromptThinking::Closed, true);
        let chunks = splitter.push("<think>\nNeed ls.\n</think>\n   ");
        let end = splitter.finish();
        assert_eq!(format!("{}{}", chunks.content, end.content), "");
        assert_eq!(format!("{}{}", chunks.reasoning, end.reasoning), "Need ls.");
    }
}
