//! Detection and incremental decoding of common template tool-call formats.
//!
//! Detection inspects template text, never the architecture or model name.
//! These parsers decode syntax only; they do not execute calls or constrain
//! generation. Reasoning must be removed by `TurnSplitter` before decoding.
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};

use crate::grammar::CompiledToolCalls;
use crate::tool_call::{NativeToolCall, XmlToolCall};

/// The tool-call convention advertised by a checkpoint's chat template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    Hermes,
    QwenXml,
    Llama3Json,
    MistralJson,
    GraniteJson,
    Pythonic,
}

impl ToolCallFormat {
    /// Special token spellings that delimit this format's calls.
    #[must_use]
    pub fn markers(self) -> &'static [&'static str] {
        match self {
            Self::Hermes => &["<tool_call>", "</tool_call>"],
            Self::QwenXml => &[
                "<tool_call>",
                "</tool_call>",
                "<function=",
                "</function>",
                "<parameter=",
                "</parameter>",
            ],
            Self::Llama3Json => &["<|python_tag|>"],
            Self::MistralJson => &["[TOOL_CALLS]", "[ARGS]", "[CALL_ID]"],
            Self::GraniteJson => &["<|tool_call|>"],
            Self::Pythonic => &["<|tool_call_start|>", "<|tool_call_end|>"],
        }
    }
}

/// First matching rule wins: Pythonic, Mistral, Granite, Qwen XML (both
/// `<tool_call>` and `<function=`), Hermes, then Llama JSON. Llama's untagged
/// form requires the quoted `name`/`parameters` example in a tool prompt;
/// merely using a tool schema's `parameters` variable is insufficient.
#[must_use]
pub fn detect(template: &str) -> Option<ToolCallFormat> {
    use ToolCallFormat::{GraniteJson, Hermes, Llama3Json, MistralJson, Pythonic, QwenXml};
    if template.contains("<|tool_call_start|>") {
        Some(Pythonic)
    } else if template.contains("[TOOL_CALLS]") {
        Some(MistralJson)
    } else if template.contains("<|tool_call|>") {
        Some(GraniteJson)
    } else if template.contains("<tool_call>") && template.contains("<function=") {
        Some(QwenXml)
    } else if template.contains("<tool_call>") {
        let has_key = |keys: &[&str]| {
            keys.iter().any(|key| {
                template.match_indices(key).any(|(pos, _)| {
                    template[pos + key.len()..]
                        .trim_start_matches(|c: char| c.is_ascii_whitespace())
                        .starts_with(':')
                })
            })
        };
        let json_keys =
            has_key(&["\"name\"", "'name'"]) && has_key(&["\"arguments\"", "'arguments'"]);
        let json_form = (template.contains("\"name\":") && template.contains("\"arguments\":"))
            || (json_keys && template.contains("tojson"));
        if json_form && !template.contains("<arg_key>") && !template.contains("<arg_value>") {
            Some(Hermes)
        } else {
            None
        }
    } else if template.contains("<|python_tag|>")
        || (template.contains("\"parameters\"")
            && template.contains("tool")
            && template.contains("\"name\""))
    {
        Some(Llama3Json)
    } else {
        None
    }
}

/// Content and completed calls released by one incremental step.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ParsedChunks {
    pub content: String,
    pub calls: Vec<NativeToolCall>,
}
/// The final withheld content and calls, after checking for malformed input.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ParsedEnd {
    pub content: String,
    pub calls: Vec<NativeToolCall>,
}

/// Incremental scanner over decoded UTF-8 fragments.
///
/// Outside calls, a possible marker prefix is withheld. An untagged Llama
/// answer beginning with an object is buffered until its complete syntax and
/// declared names can be checked. Tagged calls commit immediately. Errors
/// are retained and returned by `finish`. Concatenated output is independent
/// of fragment boundaries.
pub const MAX_TOOL_CALL_BYTES: usize = 256 * 1024;

/// The longest run of leading whitespace held while a Llama answer is
/// classified as an untagged call. A longer run is content, so the buffer
/// never grows without bound, whatever the answer continues with.
const MAX_LEADING_WHITESPACE_BYTES: usize = 1024;
/// Every framed closing-tag candidate followed by more than 1 KiB of
/// whitespace fails closed, identically at every input split.
const MAX_XML_CLOSING_WHITESPACE_BYTES: usize = 1024;
const SLACK: usize =
    "<|tool_call_end|>".len() + "</parameter>".len() + MAX_XML_CLOSING_WHITESPACE_BYTES;

#[derive(Default)]
struct ScanState {
    offset: usize,
    quote: Option<(u8, bool)>,
    escaped: bool,
    depth: usize,
    started: bool,
    xml_last: usize,
    xml_last_non_indent: Option<u8>,
    xml_in_parameter: bool,
    xml_header_end: Option<usize>,
    xml_framed: bool,
    xml_whitespace_overflow: bool,
    prefix_started: bool,
}

pub struct ToolCallParser {
    format: ToolCallFormat,
    pending: String,
    in_call: bool,
    error: Option<anyhow::Error>,
    llama_initial: bool,
    llama_calls: bool,
    declarations: Option<CompiledToolCalls>,
    scan: ScanState,
    untagged: bool,
    llama_separator: bool,
    /// Whether the parser is right after a completed call.
    ///
    /// The whitespace that follows a completed call is held until the next
    /// call or the next text decides what it is. The flag is cleared once
    /// that is decided.
    after_call: bool,
}
impl ToolCallParser {
    #[must_use]
    pub fn new(format: ToolCallFormat) -> Self {
        Self {
            format,
            pending: String::new(),
            in_call: false,
            error: None,
            llama_initial: true,
            llama_calls: false,
            declarations: None,
            scan: ScanState::default(),
            untagged: false,
            llama_separator: false,
            after_call: false,
        }
    }
    /// Attach tool declarations for untagged detection and XML conversion.
    #[must_use]
    pub fn with_declarations(mut self, declarations: CompiledToolCalls) -> Self {
        self.declarations = Some(declarations);
        self
    }
    /// Fail closed and drop the buffered text.
    ///
    /// The buffer keeps at most [`MAX_TOOL_CALL_BYTES`] plus closing-marker slack, so a failed
    /// call releases no partial text either.
    fn fail(&mut self, message: &str) {
        self.error = Some(anyhow::anyhow!("{message}"));
        self.pending.clear();
    }
    fn marker(&self) -> &'static str {
        match self.format {
            ToolCallFormat::Hermes | ToolCallFormat::QwenXml => "<tool_call>",
            ToolCallFormat::Llama3Json => "<|python_tag|>",
            ToolCallFormat::MistralJson => "[TOOL_CALLS]",
            ToolCallFormat::GraniteJson => "<|tool_call|>",
            ToolCallFormat::Pythonic => "<|tool_call_start|>",
        }
    }
    pub fn push(&mut self, text: &str) -> ParsedChunks {
        let mut out = ParsedChunks::default();
        if self.error.is_some() {
            return out;
        }
        // Bound each incremental scan even when a caller supplies a huge fragment.
        let mut rest = text;
        while !rest.is_empty() {
            let mut end = rest.len().min(64 * 1024);
            while !rest.is_char_boundary(end) {
                end -= 1;
            }
            let chunk = self.push_chunk(&rest[..end]);
            out.content.push_str(&chunk.content);
            out.calls.extend(chunk.calls);
            rest = &rest[end..];
            if self.error.is_some() {
                return out;
            }
        }
        if text.is_empty() {
            return self.push_chunk(text);
        }
        out
    }
    fn push_chunk(&mut self, text: &str) -> ParsedChunks {
        let mut out = ParsedChunks::default();
        self.pending.push_str(text);
        if self.untagged {
            if self.pending.len() > MAX_TOOL_CALL_BYTES {
                self.fail("tool call exceeds maximum size");
                return out;
            }
            let marker = self.marker();
            if let Some(pos) = scan_envelope(
                ToolCallFormat::Hermes,
                &self.pending,
                marker,
                &mut self.scan,
            ) {
                out.content.push_str(&self.pending[..pos]);
                self.pending.drain(..pos + marker.len());
                self.untagged = false;
                self.llama_initial = false;
                self.in_call = true;
                self.scan = ScanState::default();
            } else {
                return out;
            }
        }
        loop {
            if !self.in_call {
                let marker = self.marker();
                // Retain leading whitespace while deciding whether this is an untagged call.
                if self.format == ToolCallFormat::Llama3Json && self.llama_initial {
                    while self.scan.offset < self.pending.len()
                        && self.pending.as_bytes()[self.scan.offset].is_ascii_whitespace()
                    {
                        self.scan.offset += 1;
                    }
                    if self.scan.offset > MAX_LEADING_WHITESPACE_BYTES {
                        // The leading whitespace run is longer than the limit:
                        // it is content, not the preamble of an untagged
                        // call. Release it and drop the untagged
                        // classification, wherever the fragment boundary
                        // fell.
                        out.content.push_str(&self.pending[..self.scan.offset]);
                        self.pending.drain(..self.scan.offset);
                        self.scan = ScanState::default();
                        self.llama_initial = false;
                    } else if self.scan.offset == self.pending.len() {
                        // Only whitespace so far. Hold it while the answer may
                        // still be an untagged call.
                        break;
                    } else if self.pending.as_bytes()[self.scan.offset] == b'{' {
                        self.untagged = true;
                        self.scan = ScanState::default();
                        return self.push("");
                    }
                }
                // The whitespace that follows a completed call separates it
                // from the next call. Hold it until the next call or the
                // next text decides what it is.
                if self.after_call && !self.pending.is_empty() {
                    let run = self.pending.len()
                        - self
                            .pending
                            .trim_start_matches(|c: char| c.is_ascii_whitespace())
                            .len();
                    if run > MAX_LEADING_WHITESPACE_BYTES {
                        // The run is longer than the limit: it is content.
                        out.content.push_str(&self.pending[..run]);
                        self.pending.drain(..run);
                        self.after_call = false;
                    } else {
                        let rest = &self.pending[run..];
                        if rest.starts_with(marker) {
                            // The whitespace separates two calls: drop it.
                            self.pending.drain(..run);
                            self.after_call = false;
                        } else if marker.starts_with(rest) {
                            // The next call may still start: hold the
                            // whitespace until the text decides.
                            break;
                        } else {
                            // Text follows the whitespace: both are content.
                            self.after_call = false;
                        }
                    }
                }
                let found = self.pending.find(marker);
                if let Some(pos) = found {
                    out.content.push_str(&self.pending[..pos]);
                    if !self.pending[..pos]
                        .trim_matches(|c: char| c.is_ascii_whitespace())
                        .is_empty()
                    {
                        self.llama_initial = false;
                    }
                    self.pending.drain(..pos);
                    if self.pending.starts_with(marker) {
                        self.pending.drain(..marker.len());
                    }
                    self.in_call = true;
                    self.scan = ScanState::default();
                } else {
                    let keep = (1..marker.len())
                        .rev()
                        .find(|&n| self.pending.ends_with(&marker[..n]))
                        .unwrap_or(0);
                    let end = self.pending.len() - keep;
                    out.content.push_str(&self.pending[..end]);
                    if !self.pending[..end]
                        .trim_matches(|c: char| c.is_ascii_whitespace())
                        .is_empty()
                    {
                        self.llama_initial = false;
                    }
                    self.pending.drain(..end);
                    self.scan.offset = 0;
                    break;
                }
            }
            let end_marker = match self.format {
                ToolCallFormat::Hermes | ToolCallFormat::QwenXml => Some("</tool_call>"),
                ToolCallFormat::Pythonic => Some("<|tool_call_end|>"),
                _ => None,
            };
            if let Some(end_marker) = end_marker {
                let Some(pos) =
                    scan_envelope(self.format, &self.pending, end_marker, &mut self.scan)
                else {
                    if self.scan.xml_whitespace_overflow
                        || self.pending.len() > MAX_TOOL_CALL_BYTES + SLACK
                    {
                        self.fail("tool call exceeds maximum size");
                    }
                    break;
                };
                if pos > MAX_TOOL_CALL_BYTES {
                    self.fail("tool call exceeds maximum size");
                    break;
                }
                let body = self.pending[..pos].to_owned();
                self.pending.drain(..pos + end_marker.len());
                match self.parse_body(&body) {
                    Ok(calls) => {
                        // A call has completed: the whitespace that
                        // follows it separates it from the next one.
                        self.after_call = true;
                        out.calls.extend(calls);
                    }
                    Err(e) => {
                        self.error = Some(e);
                        self.pending.clear();
                        break;
                    }
                }
                self.in_call = false;
                self.scan = ScanState::default();
            } else {
                if self.format == ToolCallFormat::Llama3Json && self.llama_calls {
                    let skipped = self.pending.len()
                        - self
                            .pending
                            .trim_start_matches(|c: char| c.is_ascii_whitespace() || c == ';')
                            .len();
                    if self.pending[..skipped].contains([';', '\n']) {
                        self.llama_separator = true;
                    }
                    self.pending.drain(..skipped);
                    let marker = self.marker();
                    if self.pending.starts_with(marker) {
                        self.pending.drain(..marker.len());
                        // A new marker must be followed by another call.
                        self.llama_calls = false;
                        self.llama_separator = false;
                    } else if !self.pending.is_empty() && marker.starts_with(&self.pending) {
                        break;
                    } else if !self.pending.is_empty() && !self.llama_separator {
                        self.error = Some(anyhow::anyhow!("missing call separator"));
                        break;
                    }
                }
                let Some(end) = scan_json(self.format, &self.pending, &mut self.scan) else {
                    if self.pending.len() > MAX_TOOL_CALL_BYTES {
                        self.fail("tool call exceeds maximum size");
                    }
                    break;
                };
                if end > MAX_TOOL_CALL_BYTES {
                    self.fail("tool call exceeds maximum size");
                    break;
                }
                let body = self.pending[..end].to_owned();
                match self.parse_body(&body) {
                    Ok(calls) => {
                        if self.format != ToolCallFormat::Llama3Json {
                            // A call has completed: the whitespace that
                            // follows it separates it from the next one.
                            self.after_call = true;
                        }
                        out.calls.extend(calls);
                    }
                    Err(error) => {
                        self.error = Some(error);
                        self.pending.clear();
                        break;
                    }
                }
                self.pending.drain(..end);
                self.scan = ScanState::default();
                if self.format == ToolCallFormat::Llama3Json {
                    self.llama_calls = true;
                    self.llama_separator = false;
                } else {
                    self.in_call = false;
                }
            }
        }
        debug_assert!(
            self.pending.len() <= MAX_TOOL_CALL_BYTES + SLACK,
            "the tool-call buffer grew past its limit"
        );
        out
    }
    fn parse_body(&self, body: &str) -> Result<Vec<NativeToolCall>> {
        if self.format == ToolCallFormat::QwenXml
            && let Some(declarations) = &self.declarations
        {
            return Ok(vec![declarations.convert_xml_call(parse_xml_body(body)?)?]);
        }
        parse_body(self.format, body)
    }
    /// Whether the current call is unfinished, rather than already malformed.
    #[must_use]
    pub fn is_incomplete(&self) -> bool {
        if self.error.is_some() || !self.in_call {
            return false;
        }
        match self.format {
            ToolCallFormat::Hermes => {
                envelope_end(self.format, &self.pending, "</tool_call>").is_none()
            }
            ToolCallFormat::QwenXml => xml_is_incomplete(&self.pending),
            ToolCallFormat::Pythonic => {
                envelope_end(self.format, &self.pending, "<|tool_call_end|>").is_none()
            }
            ToolCallFormat::Llama3Json
                if self.llama_calls
                    && self
                        .pending
                        .trim_matches(|c: char| c.is_ascii_whitespace() || c == ';')
                        .is_empty() =>
            {
                false
            }
            _ => {
                let text = if self.format == ToolCallFormat::MistralJson
                    && !self
                        .pending
                        .trim_start_matches(|c: char| c.is_ascii_whitespace())
                        .starts_with('[')
                {
                    let Some((_, args)) = self.pending.split_once("[ARGS]") else {
                        return true;
                    };
                    args
                } else {
                    &self.pending
                };
                serde_json::from_str::<Value>(text).is_err_and(|error| error.is_eof())
            }
        }
    }
    pub fn finish(mut self) -> Result<ParsedEnd> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        if self.in_call {
            if self.format == ToolCallFormat::Llama3Json
                && self.llama_calls
                && self
                    .pending
                    .trim_matches(|c: char| c.is_ascii_whitespace() || c == ';')
                    .is_empty()
            {
                return Ok(ParsedEnd::default());
            }
            ensure!(
                !matches!(
                    self.format,
                    ToolCallFormat::Hermes | ToolCallFormat::QwenXml | ToolCallFormat::Pythonic
                ),
                "unterminated tool call"
            );
            Ok(ParsedEnd {
                content: String::new(),
                calls: self.parse_body(&self.pending)?,
            })
        } else {
            if self.format == ToolCallFormat::Llama3Json
                && self.llama_initial
                && !self
                    .pending
                    .trim_end_matches(|c: char| c.is_ascii_whitespace())
                    .ends_with(';')
                && let Ok(calls) = parse_body(self.format, &self.pending)
                && self
                    .declarations
                    .as_ref()
                    .is_some_and(|d| calls.iter().all(|c| d.contains_name(&c.name)))
            {
                ensure!(
                    self.pending.len() <= MAX_TOOL_CALL_BYTES,
                    "tool call exceeds maximum size"
                );
                return Ok(ParsedEnd {
                    content: String::new(),
                    calls,
                });
            }
            // The whitespace that follows the last call separates it from
            // the end of the turn: it is not content.
            let content =
                if self.after_call && self.pending.bytes().all(|byte| byte.is_ascii_whitespace()) {
                    String::new()
                } else {
                    self.pending
                };
            Ok(ParsedEnd {
                content,
                calls: Vec::new(),
            })
        }
    }
}

// XML values are raw text, so the presence of a closing-call token does not
// imply that the envelope closed. Check whether completing only the current
// structural tags could produce a valid call. Trying overlapping suffixes
// also recognizes generation cut off midway through a closing tag.
fn xml_is_incomplete(text: &str) -> bool {
    if "<function=".starts_with(text.trim_start_matches(|c: char| c.is_ascii_whitespace())) {
        return true;
    }
    // A synthetic space cannot appear in a valid name. Only the particular
    // byte appended by this probe is accepted by the internal parser.
    let probe = " ";
    for ending in [
        "</tool_call>".to_owned(),
        "</function></tool_call>".to_owned(),
        "</parameter></function></tool_call>".to_owned(),
        format!("<parameter={probe}></parameter></function></tool_call>"),
        "></function></tool_call>".to_owned(),
        "></parameter></function></tool_call>".to_owned(),
        "\n</tool_call>".to_owned(),
        "\n</function>\n</tool_call>".to_owned(),
        "\n</parameter>\n</function>\n</tool_call>".to_owned(),
        format!("\n<parameter={probe}>\n\n</parameter>\n</function>\n</tool_call>"),
        ">\n</function>\n</tool_call>".to_owned(),
        ">\n</parameter>\n</function>\n</tool_call>".to_owned(),
        format!("{probe}></parameter></function></tool_call>"),
        format!("{probe}>\n</parameter>\n</function>\n</tool_call>"),
    ] {
        let probe_index = ending.find(probe);
        // A text that already contains the whole ending is complete, not incomplete.
        let overlap_limit = probe_index.unwrap_or(ending.len() - 1);
        for overlap in 0..=overlap_limit {
            if text.ends_with(&ending[..overlap]) {
                let completed = format!("{text}{}", &ending[overlap..]);
                let body = completed.strip_suffix("</tool_call>").unwrap();
                let probe_position = probe_index.map(|index| text.len() + index - overlap);
                if parse_xml_body_with_probe(body, probe_position).is_ok() {
                    return true;
                }
            }
        }
    }
    false
}

// Incremental scanners retain quote and nesting state across fragments.
fn scan_envelope(
    format: ToolCallFormat,
    text: &str,
    marker: &str,
    state: &mut ScanState,
) -> Option<usize> {
    let bytes = text.as_bytes();
    while state.offset < bytes.len() {
        let i = state.offset;
        let byte = bytes[i];
        if format == ToolCallFormat::QwenXml {
            if byte == b'<' && marker.starts_with(&text[i..]) && text.len() - i < marker.len() {
                break;
            }
            if byte == b'<'
                && !state.xml_in_parameter
                && bytes[i..].starts_with(marker.as_bytes())
                && bytes[..state.xml_last].ends_with(b"</function>")
            {
                return Some(i);
            }
            if !state.xml_in_parameter && bytes[..=i].ends_with(b"<parameter=") {
                state.xml_in_parameter = true;
                state.xml_header_end = None;
                state.xml_framed = false;
            } else if state.xml_in_parameter && state.xml_header_end.is_none() && byte == b'>' {
                state.xml_header_end = Some(i + 1);
            } else if state.xml_in_parameter && state.xml_header_end == Some(i) {
                state.xml_framed = byte == b'\n';
            }
            if state.xml_in_parameter && byte == b'<' {
                let tail = &text[i..];
                let close = "</parameter>";
                if close.starts_with(tail) && tail.len() < close.len() {
                    break;
                }
                if let Some(after) = tail.strip_prefix(close) {
                    if state.xml_framed && state.xml_last_non_indent != Some(b'\n') {
                        state.xml_last_non_indent = Some(byte);
                        state.offset += 1;
                        continue;
                    }
                    let next = after.trim_start_matches(|c: char| c.is_ascii_whitespace());
                    if after.len() - next.len() > MAX_XML_CLOSING_WHITESPACE_BYTES {
                        // Leave the candidate unconsumed; the caller fails closed.
                        state.xml_whitespace_overflow = true;
                        return None;
                    }
                    if next.is_empty()
                        || (next.len() < "<parameter=".len() && "<parameter=".starts_with(next))
                        || (next.len() < "</function>".len() && "</function>".starts_with(next))
                    {
                        break;
                    }
                    if next.starts_with("<parameter=") || next.starts_with("</function>") {
                        state.xml_in_parameter = false;
                    }
                }
            }
            if !byte.is_ascii_whitespace() {
                state.xml_last = i + 1;
            }
            if !matches!(byte, b' ' | b'\t') {
                state.xml_last_non_indent = Some(byte);
            }
        } else if let Some((quote, triple)) = state.quote {
            if state.escaped {
                state.escaped = false;
            } else if byte == b'\\' {
                state.escaped = true;
            } else if byte == quote {
                if triple {
                    if bytes.len() - i < 3 {
                        break;
                    }
                    if bytes[i..].starts_with(&[quote; 3]) {
                        state.quote = None;
                        state.offset += 2;
                    }
                } else {
                    state.quote = None;
                }
            }
        } else if byte == b'"' || (format == ToolCallFormat::Pythonic && byte == b'\'') {
            if format == ToolCallFormat::Pythonic && bytes.len() - i < 3 {
                break;
            }
            let triple = format == ToolCallFormat::Pythonic && bytes[i..].starts_with(&[byte; 3]);
            state.quote = Some((byte, triple));
            if triple {
                state.offset += 2;
            }
        } else if byte == b'<' {
            if marker.starts_with(&text[i..]) && text.len() - i < marker.len() {
                break;
            }
            if text[i..].starts_with(marker) {
                return Some(i);
            }
        }
        state.offset += 1;
    }
    None
}

fn scan_json(format: ToolCallFormat, text: &str, state: &mut ScanState) -> Option<usize> {
    let bytes = text.as_bytes();
    while state.offset < bytes.len() {
        let i = state.offset;
        let b = bytes[i];
        if !state.started {
            if b == b'{'
                || b == b'[' && (format != ToolCallFormat::MistralJson || !state.prefix_started)
            {
                state.started = true;
                state.depth = 1;
            } else if !b.is_ascii_whitespace() {
                state.prefix_started = true;
            }
        } else if state.quote.is_some() {
            if state.escaped {
                state.escaped = false;
            } else if b == b'\\' {
                state.escaped = true;
            } else if b == b'"' {
                state.quote = None;
            }
        } else {
            match b {
                b'"' => state.quote = Some((b'"', false)),
                b'{' | b'[' => state.depth += 1,
                b'}' | b']' => {
                    state.depth -= 1;
                    if state.depth == 0 {
                        state.offset += 1;
                        return Some(state.offset);
                    }
                }
                _ => {}
            }
        }
        state.offset += 1;
    }
    None
}

// An envelope token inside a quoted argument is data. Scan bytes because all
// delimiters are ASCII; returned offsets are therefore UTF-8 boundaries.
fn envelope_end(format: ToolCallFormat, text: &str, marker: &str) -> Option<usize> {
    if format == ToolCallFormat::QwenXml {
        // Parameter values are raw text. A closing token inside a value is
        // data unless the preceding function block actually parses.
        return text
            .match_indices(marker)
            .find_map(|(index, _)| parse_xml_body(&text[..index]).ok().map(|_| index));
    }
    let bytes = text.as_bytes();
    let mut quote: Option<(u8, bool)> = None;
    let mut escaped = false;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some((delimiter, triple)) = quote {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == delimiter {
                if !triple {
                    quote = None;
                } else if bytes.get(index..index + 3) == Some(&[delimiter; 3]) {
                    quote = None;
                    index += 2;
                }
            }
        } else if byte == b'"' || (format == ToolCallFormat::Pythonic && byte == b'\'') {
            let triple = format == ToolCallFormat::Pythonic
                && bytes.get(index..index + 3) == Some(&[byte; 3]);
            quote = Some((byte, triple));
            if triple {
                index += 2;
            }
        } else if byte == b'<' && text[index..].starts_with(marker) {
            return Some(index);
        }
        index += 1;
    }
    None
}

// Parse generic XML directly using the same boundaries as the scanner.
fn parse_xml_body(body: &str) -> Result<XmlToolCall> {
    parse_xml_body_with_probe(body, None)
}

fn parse_xml_body_with_probe(body: &str, probe: Option<usize>) -> Result<XmlToolCall> {
    let rest = body
        .trim_matches(|c: char| c.is_ascii_whitespace())
        .strip_prefix("<function=")
        .context("tool call must open a function block")?;
    let (name, mut rest) = rest.split_once('>').context("unterminated function name")?;
    ensure!(valid_xml_name(name), "invalid function name");
    let mut parameters = Vec::new();
    loop {
        rest = rest.trim_start_matches(|c: char| c.is_ascii_whitespace());
        let Some(parameter) = rest.strip_prefix("<parameter=") else {
            break;
        };
        let (name, value) = parameter
            .split_once('>')
            .context("unterminated parameter name")?;
        let synthetic = name.strip_suffix(' ').is_some_and(|prefix| {
            probe == Some(name.as_ptr() as usize - body.as_ptr() as usize + name.len() - 1)
                && (prefix.is_empty() || valid_xml_name(prefix))
        });
        ensure!(valid_xml_name(name) || synthetic, "invalid parameter name");
        let framed = value.starts_with('\n');
        let (value, tail) =
            split_parameter_value(value, framed).context("unterminated parameter value")?;
        let value = if framed {
            value.strip_prefix('\n').unwrap_or(value)
        } else {
            value
        };
        ensure!(
            !parameters.iter().any(|(seen, _)| seen == name),
            "repeated parameter {name:?}"
        );
        parameters.push((name.to_owned(), value.to_owned()));
        rest = tail;
    }
    ensure!(
        rest.trim_matches(|c: char| c.is_ascii_whitespace()) == "</function>",
        "unclosed function block"
    );
    Ok(XmlToolCall {
        name: name.to_owned(),
        parameters,
    })
}

fn valid_xml_name(name: &str) -> bool {
    !name.is_empty()
        && !name.bytes().any(|byte| {
            byte.is_ascii_whitespace() || matches!(byte, b'=' | b'<' | b'>' | b'/' | b'\'' | b'"')
        })
}

/// Split a parameter value from the text that follows it.
///
/// A framed value ends with the newline that separates it from its closing
/// tag; an unframed value ends at the tag itself. The closing tag may be
/// indented: a template that prints an indented block ends the value with
/// spaces or tabs before `</parameter>`. That indentation belongs to the
/// tag, not to the value.
fn split_parameter_value(value: &str, framed: bool) -> Option<(&str, &str)> {
    let close = "</parameter>";
    let mut search = 0;
    loop {
        let index = value[search..].find(close)? + search;
        let (text, tail) = value.split_at(index);
        let tail = &tail[close.len()..];
        let next = tail.trim_start_matches(|c: char| c.is_ascii_whitespace());
        if !next.is_empty() && !next.starts_with("<parameter=") && !next.starts_with("</function>")
        {
            search = index + close.len();
            continue;
        }
        let trimmed = text.trim_end_matches([' ', '\t']);
        if !framed {
            return Some((text, tail));
        }
        if let Some(newline) = trimmed.strip_suffix('\n') {
            return Some((newline, tail));
        }
        search = index + close.len();
    }
}

fn call(value: &Value, key: &str) -> Result<NativeToolCall> {
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .context("tool call name must be a string")?;
    ensure!(!name.is_empty(), "tool call name must not be empty");
    let arguments = value.get(key).context("missing tool call arguments")?;
    let arguments = if let Some(text) = arguments.as_str() {
        serde_json::from_str(text)?
    } else {
        arguments.clone()
    };
    ensure!(
        arguments.is_object(),
        "tool call arguments must be an object"
    );
    Ok(NativeToolCall {
        id: None,
        name: name.to_owned(),
        arguments_text: arguments.to_string(),
        arguments,
    })
}
fn parse_body(format: ToolCallFormat, body: &str) -> Result<Vec<NativeToolCall>> {
    use ToolCallFormat::{GraniteJson, Hermes, Llama3Json, MistralJson, Pythonic, QwenXml};
    match format {
        Hermes => Ok(vec![call(
            &serde_json::from_str::<Value>(body.trim_matches(|c: char| c.is_ascii_whitespace()))?,
            "arguments",
        )?]),
        QwenXml => {
            let xml = parse_xml_body(body)?;
            ensure!(!xml.name.is_empty(), "tool call name must not be empty");
            let mut arguments = Map::new();
            for (name, value) in xml.parameters {
                arguments.insert(name, Value::String(value));
            }
            let arguments = Value::Object(arguments);
            Ok(vec![NativeToolCall {
                id: None,
                name: xml.name,
                arguments_text: arguments.to_string(),
                arguments,
            }])
        }
        Pythonic => PythonParser::new(body).calls(),
        MistralJson
            if body
                .trim_start_matches(|c: char| c.is_ascii_whitespace())
                .starts_with('[') =>
        {
            let values: Vec<Value> =
                serde_json::from_str(body.trim_matches(|c: char| c.is_ascii_whitespace()))?;
            ensure!(!values.is_empty(), "missing tool call");
            values.iter().map(|v| call(v, "arguments")).collect()
        }
        MistralJson => {
            let (name, args) = body
                .trim_matches(|c: char| c.is_ascii_whitespace())
                .split_once("[ARGS]")
                .context("missing [ARGS]")?;
            let (name, id) = name
                .split_once("[CALL_ID]")
                .map_or((name, None), |(name, id)| (name, Some(id)));
            ensure!(
                !name
                    .trim_matches(|c: char| c.is_ascii_whitespace())
                    .is_empty(),
                "missing tool name"
            );
            let arguments: Value =
                serde_json::from_str(args.trim_matches(|c: char| c.is_ascii_whitespace()))?;
            ensure!(arguments.is_object(), "tool arguments must be an object");
            Ok(vec![NativeToolCall {
                name: name
                    .trim_matches(|c: char| c.is_ascii_whitespace())
                    .to_owned(),
                id: id.map(str::to_owned),
                arguments_text: arguments.to_string(),
                arguments,
            }])
        }
        GraniteJson => {
            let values: Vec<Value> =
                serde_json::from_str(body.trim_matches(|c: char| c.is_ascii_whitespace()))?;
            ensure!(!values.is_empty(), "missing tool call");
            values.iter().map(|v| call(v, "arguments")).collect()
        }
        Llama3Json => {
            let mut result = Vec::new();
            let mut rest = body.trim_matches(|c: char| c.is_ascii_whitespace());
            while !rest.is_empty() {
                let mut stream = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
                let value = stream.next().context("missing tool call")??;
                result.push(call(
                    &value,
                    if value.get("parameters").is_some() {
                        "parameters"
                    } else {
                        "arguments"
                    },
                )?);
                let tail = &rest[stream.byte_offset()..];
                let trimmed = tail.trim_start_matches(|c: char| c.is_ascii_whitespace());
                if trimmed.is_empty() {
                    rest = trimmed;
                } else if let Some(next) = trimmed.strip_prefix(';') {
                    rest = next.trim_start_matches(|c: char| c.is_ascii_whitespace());
                } else {
                    ensure!(
                        tail[..tail.len() - trimmed.len()].contains('\n'),
                        "missing call separator"
                    );
                    rest = trimmed;
                }
            }
            ensure!(!result.is_empty(), "missing tool call");
            Ok(result)
        }
    }
}

struct PythonParser<'a> {
    rest: &'a str,
}
impl<'a> PythonParser<'a> {
    fn new(s: &'a str) -> Self {
        Self { rest: s }
    }
    fn ws(&mut self) {
        self.rest = self
            .rest
            .trim_start_matches(|c: char| c.is_ascii_whitespace());
    }
    fn eat(&mut self, s: &str) -> bool {
        self.ws();
        if let Some(r) = self.rest.strip_prefix(s) {
            self.rest = r;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, s: &str) -> Result<()> {
        ensure!(self.eat(s), "expected {s:?}");
        Ok(())
    }
    fn ident(&mut self) -> Result<String> {
        self.ws();
        let n = self
            .rest
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
            .count();
        ensure!(
            n > 0 && !self.rest.as_bytes()[0].is_ascii_digit(),
            "expected identifier"
        );
        let name = self.rest[..n].to_owned();
        self.rest = &self.rest[n..];
        Ok(name)
    }
    fn calls(mut self) -> Result<Vec<NativeToolCall>> {
        self.expect("[")?;
        let mut calls = Vec::new();
        while !self.eat("]") {
            self.ws();
            let n = self
                .rest
                .bytes()
                .take_while(|b| b.is_ascii_alphanumeric() || b"_.-".contains(b))
                .count();
            ensure!(
                n > 0
                    && (self.rest.as_bytes()[0].is_ascii_alphabetic()
                        || self.rest.as_bytes()[0] == b'_'),
                "expected function name"
            );
            let name = self.rest[..n].to_owned();
            self.rest = &self.rest[n..];
            self.expect("(")?;
            let mut args = Map::new();
            while !self.eat(")") {
                let key = self.ident()?;
                self.expect("=")?;
                let value = self.value(0)?;
                ensure!(args.insert(key, value).is_none(), "duplicate argument");
                if !self.eat(",") {
                    self.expect(")")?;
                    break;
                }
            }
            let arguments = Value::Object(args);
            calls.push(NativeToolCall {
                id: None,
                name,
                arguments_text: arguments.to_string(),
                arguments,
            });
            if !self.eat(",") {
                self.expect("]")?;
                break;
            }
        }
        self.ws();
        ensure!(
            self.rest.is_empty() && !calls.is_empty(),
            "invalid call list"
        );
        Ok(calls)
    }
    fn value(&mut self, depth: usize) -> Result<Value> {
        ensure!(depth < 64, "Python literal nesting limit exceeded");
        self.ws();
        if self.eat("True") {
            return Ok(Value::Bool(true));
        }
        if self.eat("False") {
            return Ok(Value::Bool(false));
        }
        if self.eat("None") {
            return Ok(Value::Null);
        }
        if self.rest.starts_with('\'') || self.rest.starts_with('"') {
            return self.string().map(Value::String);
        }
        if self.eat("[") {
            let mut v = Vec::new();
            while !self.eat("]") {
                v.push(self.value(depth + 1)?);
                if !self.eat(",") {
                    self.expect("]")?;
                    break;
                }
            }
            return Ok(Value::Array(v));
        }
        if self.eat("{") {
            let mut m = Map::new();
            while !self.eat("}") {
                let k = self.string()?;
                self.expect(":")?;
                let v = self.value(depth + 1)?;
                m.insert(k, v);
                if !self.eat(",") {
                    self.expect("}")?;
                    break;
                }
            }
            return Ok(Value::Object(m));
        }
        let n = self
            .rest
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || b"-+._".contains(b))
            .count();
        ensure!(n > 0, "invalid literal");
        let number = &self.rest[..n];
        self.rest = &self.rest[n..];
        python_number(number)
    }
    fn string(&mut self) -> Result<String> {
        self.ws();
        let quote = self.rest.chars().next().context("expected string")?;
        ensure!(quote == '\'' || quote == '"', "expected string");
        let triple = self
            .rest
            .starts_with(if quote == '\'' { "\'\'\'" } else { "\"\"\"" });
        self.rest = &self.rest[if triple { 3 } else { 1 }..];
        let mut result = String::new();
        while let Some(c) = self.rest.chars().next() {
            self.rest = &self.rest[c.len_utf8()..];
            if c == quote {
                if !triple {
                    return Ok(result);
                }
                if self.rest.starts_with(&format!("{quote}{quote}")) {
                    self.rest = &self.rest[2..];
                    return Ok(result);
                }
                result.push(c);
                continue;
            }
            if c == '\\' {
                let e = self.rest.chars().next().context("unfinished escape")?;
                self.rest = &self.rest[e.len_utf8()..];
                if e == '\n' {
                    continue;
                }
                let escaped = match e {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    'b' => '\x08',
                    'f' => '\x0c',
                    'a' => '\x07',
                    'v' => '\x0b',
                    'u' | 'U' | 'x' => {
                        let count = match e {
                            'u' => 4,
                            'U' => 8,
                            _ => 2,
                        };
                        let digits = self
                            .rest
                            .get(..count)
                            .context("unfinished unicode escape")?;
                        ensure!(
                            digits.bytes().all(|b| b.is_ascii_hexdigit()),
                            "invalid unicode escape"
                        );
                        let code = u32::from_str_radix(digits, 16)?;
                        self.rest = &self.rest[count..];
                        char::from_u32(code).context("invalid unicode scalar")?
                    }
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    '0'..='7' => {
                        let more = self
                            .rest
                            .bytes()
                            .take(2)
                            .take_while(|b| (b'0'..=b'7').contains(b))
                            .count();
                        let digits = format!("{e}{}", &self.rest[..more]);
                        self.rest = &self.rest[more..];
                        char::from_u32(u32::from_str_radix(&digits, 8)?)
                            .context("invalid octal escape")?
                    }
                    // Python retains the backslash of an unrecognized escape.
                    _ => {
                        result.push('\\');
                        e
                    }
                };
                result.push(escaped);
            } else {
                ensure!(
                    triple || (c != '\n' && c != '\r'),
                    "unescaped newline in string"
                );
                result.push(c);
            }
        }
        bail!("unterminated string")
    }
}

fn python_number(number: &str) -> Result<Value> {
    let unsigned = number.strip_prefix(['+', '-']).unwrap_or(number);
    ensure!(
        !unsigned.is_empty() && !unsigned.starts_with(['+', '-']),
        "invalid numeric literal"
    );
    let radix = if unsigned.starts_with("0x") || unsigned.starts_with("0X") {
        16
    } else if unsigned.starts_with("0o") || unsigned.starts_with("0O") {
        8
    } else if unsigned.starts_with("0b") || unsigned.starts_with("0B") {
        2
    } else {
        10
    };
    let digits = if radix == 10 {
        unsigned
    } else {
        &unsigned[2..]
    };
    for (index, byte) in digits.bytes().enumerate() {
        if byte == b'_' {
            let valid_digit = |b: u8| char::from(b).is_digit(radix);
            ensure!(
                (index > 0 && valid_digit(digits.as_bytes()[index - 1])
                    || index == 0 && radix != 10)
                    && digits
                        .as_bytes()
                        .get(index + 1)
                        .is_some_and(|&b| valid_digit(b)),
                "invalid numeric separator"
            );
        }
    }
    if radix != 10 {
        ensure!(
            digits
                .bytes()
                .all(|b| b == b'_' || char::from(b).is_digit(radix)),
            "invalid integer literal"
        );
        let magnitude = u64::from_str_radix(&digits.replace('_', ""), radix)?;
        if number.starts_with('-') {
            let signed = -i128::from(magnitude);
            return Ok(Value::from(
                i64::try_from(signed).context("integer out of range")?,
            ));
        }
        return Ok(Value::from(magnitude));
    }
    let number = number.replace('_', "");
    let number = number.strip_prefix('+').unwrap_or(&number);
    if !number.contains(['.', 'e', 'E'])
        && number
            .strip_prefix('-')
            .unwrap_or(number)
            .bytes()
            .all(|b| b == b'0')
    {
        return Ok(Value::from(0));
    }
    let exponent_start = number.find(['e', 'E']).unwrap_or(number.len());
    let (mantissa, exponent) = number.split_at(exponent_start);
    let mantissa = if mantissa.starts_with('.') {
        format!("0{mantissa}")
    } else if let Some(rest) = mantissa.strip_prefix("-.") {
        format!("-0.{rest}")
    } else {
        mantissa.to_owned()
    };
    let number = if mantissa.ends_with('.') {
        format!("{mantissa}0{exponent}")
    } else {
        format!("{mantissa}{exponent}")
    };
    if number.contains(['.', 'e', 'E']) {
        // Python permits leading zeroes in floats (e.g. 01.5), while JSON
        // does not. The Rust float parser accepts those literal spellings;
        // JSON numbers still require a finite value.
        let value: f64 = number.parse().context("invalid float literal")?;
        return serde_json::Number::from_f64(value)
            .map(Value::Number)
            .context("float literal is not finite");
    }
    Ok(serde_json::from_str(&number)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(format: ToolCallFormat, fragments: &[&str]) -> Result<ParsedEnd> {
        let mut parser = ToolCallParser::new(format);
        let mut result = ParsedEnd::default();
        for fragment in fragments {
            let chunks = parser.push(fragment);
            result.content.push_str(&chunks.content);
            result.calls.extend(chunks.calls);
        }
        let end = parser.finish()?;
        result.content.push_str(&end.content);
        result.calls.extend(end.calls);
        Ok(result)
    }

    fn assert_all_splits(format: ToolCallFormat, text: &str, expected: &ParsedEnd) {
        for split in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
            assert_eq!(
                decode(format, &[&text[..split], "", &text[split..]]).unwrap(),
                *expected,
                "{format:?}, split {split}"
            );
        }
        let fragments: Vec<_> = text
            .char_indices()
            .map(|(index, c)| &text[index..index + c.len_utf8()])
            .collect();
        assert_eq!(decode(format, &fragments).unwrap(), *expected);
    }

    fn declarations() -> CompiledToolCalls {
        use crate::grammar::CompiledJsonSchema;
        let schema = CompiledJsonSchema::compile_closed(&serde_json::json!({
            "type":"object", "properties":{"x":{"type":"string"},"n":{"type":"integer"}},
            "additionalProperties":false
        }))
        .unwrap();
        CompiledToolCalls::new([(String::from("f"), schema)]).unwrap()
    }

    #[test]
    fn xml_closing_whitespace_is_bounded() {
        use std::time::{Duration, Instant};
        let text = format!(
            "<tool_call><function=f><parameter=x>v</parameter>{}",
            " ".repeat(10 * 1024 * 1024)
        );
        for size in [text.len(), 64 * 1024] {
            let start = Instant::now();
            let mut parser = ToolCallParser::new(ToolCallFormat::QwenXml);
            for chunk in text.as_bytes().chunks(size) {
                parser.push(std::str::from_utf8(chunk).unwrap());
                assert!(parser.pending.len() <= MAX_TOOL_CALL_BYTES + SLACK);
            }
            assert!(
                parser
                    .finish()
                    .unwrap_err()
                    .to_string()
                    .contains("maximum size")
            );
            // Guard against quadratic behaviour, not execution speed.
            assert!(start.elapsed() <= Duration::from_secs(30));
        }
    }

    #[test]
    fn framed_xml_indentation_scans_incrementally() {
        use std::time::{Duration, Instant};
        let text = format!(
            "<tool_call><function=f><parameter=x>\nvalue\n{}</parameter>{}</function></tool_call>",
            " ".repeat(200_000),
            " ".repeat(MAX_XML_CLOSING_WHITESPACE_BYTES),
        );
        let start = Instant::now();
        let fragments: Vec<_> = text
            .as_bytes()
            .iter()
            .map(|byte| std::str::from_utf8(std::slice::from_ref(byte)).unwrap())
            .collect();
        let parsed = decode(ToolCallFormat::QwenXml, &fragments).unwrap();
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].arguments["x"], "value");
        // Guard against quadratic behaviour, not execution speed.
        assert!(start.elapsed() <= Duration::from_secs(30));
    }

    #[test]
    fn xml_unframed_closing_tags_and_whitespace_are_preserved() {
        for (value, expected_value) in [
            ("abc</parameter>def", "abc</parameter>def"),
            ("value\n  ", "value\n  "),
        ] {
            let text = format!(
                "<tool_call><function=f><parameter=x>{value}</parameter></function></tool_call>"
            );
            let expected = decode(ToolCallFormat::QwenXml, &[&text]).unwrap();
            assert_eq!(expected.calls[0].arguments["x"], expected_value);
            assert_all_splits(ToolCallFormat::QwenXml, &text, &expected);
        }
    }

    #[test]
    fn llama_nbsp_is_not_a_separator_or_trailing_whitespace() {
        let call = r#"<|python_tag|>{"name":"f","parameters":{}}"#;
        for text in [format!("{call}\u{a0}{call}"), format!("{call}\u{a0}")] {
            for split in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
                assert!(
                    decode(
                        ToolCallFormat::Llama3Json,
                        &[&text[..split], &text[split..]]
                    )
                    .is_err(),
                    "split {split}"
                );
            }
        }
    }

    #[test]
    fn streaming_size_limit_fails_closed() {
        for (format, prefix) in [
            (
                ToolCallFormat::Llama3Json,
                "{\"name\":\"f\",\"arguments\":{\"x\":\"",
            ),
            (ToolCallFormat::Pythonic, "<|tool_call_start|>[f(x='"),
            (
                ToolCallFormat::GraniteJson,
                "<|tool_call|>[{\"name\":\"f\",\"arguments\":{\"x\":\"",
            ),
        ] {
            let mut parser = ToolCallParser::new(format);
            assert!(parser.push(prefix).content.is_empty());
            for _ in 0..257 {
                let result = parser.push(&"x".repeat(1024));
                assert!(result.content.is_empty());
                assert!(result.calls.is_empty());
            }
            assert!(parser.finish().is_err(), "{format:?}");
        }
    }

    #[test]
    fn regression_buffer_limits_and_marker_splits() {
        use ToolCallFormat::{GraniteJson, Hermes, Llama3Json, MistralJson, Pythonic, QwenXml};
        for format in [
            Hermes,
            QwenXml,
            Llama3Json,
            MistralJson,
            GraniteJson,
            Pythonic,
        ] {
            for size in [1024 * 1024, 64 * 1024] {
                let text = "x".repeat(1024 * 1024);
                let mut parser = ToolCallParser::new(format);
                for chunk in text.as_bytes().chunks(size) {
                    parser.push(std::str::from_utf8(chunk).unwrap());
                    assert!(parser.pending.len() <= MAX_TOOL_CALL_BYTES + 16);
                }
                assert!(parser.finish().is_ok());
            }
        }
        for size in [10 * 1024 * 1024, 64 * 1024] {
            let text = " ".repeat(10 * 1024 * 1024);
            let mut parser = ToolCallParser::new(Llama3Json);
            for chunk in text.as_bytes().chunks(size) {
                parser.push(std::str::from_utf8(chunk).unwrap());
                assert!(parser.pending.len() <= MAX_TOOL_CALL_BYTES + 16);
            }
            assert!(parser.finish().is_ok());
        }
        for size in [2 * 1024 * 1024, 64 * 1024] {
            let text = format!("<tool_call>bad</tool_call>{}", "x".repeat(1024 * 1024));
            let mut parser = ToolCallParser::new(Hermes);
            for chunk in text.as_bytes().chunks(size) {
                parser.push(std::str::from_utf8(chunk).unwrap());
                assert!(parser.pending.len() <= MAX_TOOL_CALL_BYTES + 16);
            }
            assert!(parser.finish().is_err());
        }
        let marker = "<|python_tag|>";
        let text = format!(
            "{}{}{}{{\"name\":\"f\",\"arguments\":{{}}}}",
            " ".repeat(1025),
            "x".repeat(MAX_TOOL_CALL_BYTES + 1),
            marker
        );
        let expected = decode(Llama3Json, &[&text]).unwrap();
        let start = text.find(marker).unwrap();
        for split in start + 1..start + marker.len() {
            assert_eq!(
                decode(Llama3Json, &[&text[..split], &text[split..]]).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn regression_end_marker_limit_and_ascii_whitespace() {
        let body = format!(
            "{{\"name\":\"f\",\"arguments\":{{\"x\":\"{}\"}}}}",
            "x".repeat(
                MAX_TOOL_CALL_BYTES - 5 - "{\"name\":\"f\",\"arguments\":{\"x\":\"\"}}".len()
            )
        );
        assert_eq!(body.len(), MAX_TOOL_CALL_BYTES - 5);
        let text = format!("<tool_call>{body}</tool_call>");
        let expected = decode(ToolCallFormat::Hermes, &[&text]).unwrap();
        let start = text.find("</tool_call>").unwrap();
        for split in start + 1..start + "</tool_call>".len() {
            assert_eq!(
                decode(ToolCallFormat::Hermes, &[&text[..split], &text[split..]]).unwrap(),
                expected
            );
        }
        let text = "\u{a0}{\"name\":\"f\",\"arguments\":{}}";
        let expected = decode(ToolCallFormat::Llama3Json, &[text]).unwrap();
        assert_all_splits(ToolCallFormat::Llama3Json, text, &expected);
    }

    #[test]
    fn xml_values_preserve_nonstructural_whitespace() {
        for (value, expected_value) in [("hello  ", "hello  "), ("\nhello\n\n  ", "hello\n")] {
            let text = format!(
                "<tool_call><function=f><parameter=x>{value}</parameter></function></tool_call>"
            );
            let expected = decode(ToolCallFormat::QwenXml, &[&text]).unwrap();
            assert_eq!(expected.calls[0].arguments["x"], expected_value);
            assert_all_splits(ToolCallFormat::QwenXml, &text, &expected);
        }
    }

    #[test]
    fn xml_parameter_whitespace_variants_stream_at_every_boundary() {
        for text in [
            "<tool_call>\n<function=f>\n<parameter=x>value</parameter></function></tool_call>",
            "<tool_call>\n<function=f>\n<parameter=x>\nvalue\n</parameter>  \n</function> \n</tool_call>",
            "<tool_call> \n<function=f>\n<parameter=x>value</parameter>\n</function>\n</tool_call>",
        ] {
            let expected = decode(ToolCallFormat::QwenXml, &[text]).unwrap();
            assert_eq!(expected.calls[0].arguments["x"], "value");
            assert_all_splits(ToolCallFormat::QwenXml, text, &expected);
        }
    }

    #[test]
    fn xml_closing_tags_accept_indentation() {
        for text in [
            "<tool_call>\n<function=f>\n<parameter=x>\nvalue\n  </parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=f>\n<parameter=x>\nvalue\n\t</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=f>\n<parameter=x>\nvalue\n  </parameter>\n  </function>\n</tool_call>",
            "<tool_call>\n<function=f>\n<parameter=x>\nvalue\n  </parameter>\n</function>\n</tool_call>",
        ] {
            let expected = decode(ToolCallFormat::QwenXml, &[text]).unwrap();
            assert_eq!(expected.calls[0].arguments["x"], "value", "{text}");
            assert_all_splits(ToolCallFormat::QwenXml, text, &expected);
        }
    }

    #[test]
    fn xml_raw_parameter_keeps_embedded_parameter_tags() {
        let text = "<tool_call>\n<function=f>\n<parameter=x>\nraw </parameter><parameter=other> text </function></tool_call>\n</parameter>  \n</function>\n</tool_call>";
        let expected = decode(ToolCallFormat::QwenXml, &[text]).unwrap();
        assert_eq!(
            expected.calls[0].arguments["x"],
            "raw </parameter><parameter=other> text </function></tool_call>"
        );
        assert_all_splits(ToolCallFormat::QwenXml, text, &expected);
    }

    #[test]
    fn regression_format_edges() {
        for (format, text) in [
            (ToolCallFormat::MistralJson, "[TOOL_CALLS][]"),
            (ToolCallFormat::GraniteJson, "<|tool_call|>[]"),
        ] {
            assert!(decode(format, &[text]).is_err());
        }
        let joined = r#"{"name":"f","arguments":{}}{"name":"f","arguments":{}}"#;
        assert!(parse_body(ToolCallFormat::Llama3Json, joined).is_err());
        assert_eq!(
            parse_body(ToolCallFormat::Llama3Json, &joined.replace("}{", "}\n{"))
                .unwrap()
                .len(),
            2
        );
        let parsed = decode(
            ToolCallFormat::MistralJson,
            &["[TOOL_CALLS]get-weather[CALL_ID]abc[ARGS]{}"],
        )
        .unwrap();
        assert_eq!(parsed.calls[0].name, "get-weather");
        assert_eq!(parsed.calls[0].id.as_deref(), Some("abc"));
        let parsed = decode(
            ToolCallFormat::Pythonic,
            &["<|tool_call_start|>[get-weather()]<|tool_call_end|>"],
        )
        .unwrap();
        assert_eq!(parsed.calls[0].name, "get-weather");
        let parsed = decode(
            ToolCallFormat::Hermes,
            &[r#"<tool_call>{"name":"f","arguments":"{\"x\":1}"}</tool_call>"#],
        )
        .unwrap();
        assert_eq!(parsed.calls[0].arguments["x"], 1);
        assert!(
            decode(
                ToolCallFormat::Hermes,
                &[r#"<tool_call>{"name":"f","arguments":"[]"}</tool_call>"#]
            )
            .is_err()
        );
        assert_eq!(
            detect("<tool_call><arg_key>name</arg_key><arg_value>arguments</arg_value>"),
            None
        );
    }

    #[test]
    fn xml_closing_tokens_inside_values_do_not_end_the_call() {
        for text in [
            "<tool_call><function=f><parameter=x>literal </function></tool_call> text</parameter></function></tool_call>",
            "<tool_call>\n<function=f>\n<parameter=x>\nliteral </function></tool_call> text\n</parameter>\n</function>\n</tool_call>",
        ] {
            let expected = decode(ToolCallFormat::QwenXml, &[text]).unwrap();
            assert_eq!(expected.calls.len(), 1);
            assert_all_splits(ToolCallFormat::QwenXml, text, &expected);
        }
    }

    #[test]
    fn large_calls_are_linear_and_bounded() {
        for (format, start, end) in [
            (
                ToolCallFormat::Hermes,
                r#"<tool_call>{"name":"f","arguments":{"x":""#,
                r#""}}</tool_call>"#,
            ),
            (
                ToolCallFormat::MistralJson,
                r#"[TOOL_CALLS]f[ARGS]{"x":""#,
                r#""}"#,
            ),
            (
                ToolCallFormat::QwenXml,
                "<tool_call><function=f><parameter=x>",
                "</parameter></function></tool_call>",
            ),
        ] {
            let text = format!("{start}{}{end}", "a".repeat(200 * 1024));
            let now = std::time::Instant::now();
            let mut parser = ToolCallParser::new(format);
            let mut calls = Vec::new();
            for chunk in text.as_bytes().chunks(1) {
                calls.extend(parser.push(std::str::from_utf8(chunk).unwrap()).calls);
            }
            calls.extend(parser.finish().unwrap().calls);
            assert_eq!(calls.len(), 1);
            assert!(
                now.elapsed() < std::time::Duration::from_secs(20),
                "{format:?}: {:?}",
                now.elapsed()
            );
            let large = format!("{start}{}{end}", "a".repeat(300 * 1024));
            assert!(decode(format, &[&large]).is_err());
        }
    }

    #[test]
    fn untagged_llama_requires_a_complete_declared_call() {
        for text in [
            r#"{"status":"ok"}"#,
            " \n{\"status\":\"ok\"}\n ",
            r#"{"name":"unknown","parameters":{}}"#,
            r#"{"name":123,"parameters":{}}"#,
            r#"{"name":"f","parameters":null}"#,
            r#"{"name":"f","parameters":{}} garbage"#,
            r#"{"name":"f","parameters":{"x":"unfinished"#,
            r#"{"name":"f","parameters":{"x":"猫"}} trailing"#,
            r#"{"name":"f","parameters":{}} {"status":"ok"}"#,
            r#"{"name":"f","parameters":{}} {"name":"unknown","parameters":{}}"#,
            r#"{"name":"f","arguments":{}} trailing"#,
            r#"{"name":"f","arguments":{}};"#,
            r#"{"name":"f","arguments":{"x":"<|python_tag|>"}} trailing"#,
        ] {
            for split in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
                let mut parser = ToolCallParser::new(ToolCallFormat::Llama3Json)
                    .with_declarations(declarations());
                assert_eq!(parser.push(&text[..split]), ParsedChunks::default());
                assert_eq!(parser.push(&text[split..]), ParsedChunks::default());
                let end = parser.finish().unwrap();
                assert_eq!(end.content, text);
                assert!(end.calls.is_empty());
            }
        }
        let text = r#" {"name":"f","arguments":{}}; {"name":"f","parameters":{}} "#;
        for split in 0..=text.len() {
            let mut parser =
                ToolCallParser::new(ToolCallFormat::Llama3Json).with_declarations(declarations());
            assert_eq!(parser.push(&text[..split]), ParsedChunks::default());
            assert_eq!(parser.push(&text[split..]), ParsedChunks::default());
            let end = parser.finish().unwrap();
            assert_eq!(end.calls.len(), 2);
            assert!(end.content.is_empty());
        }
        let mut parser =
            ToolCallParser::new(ToolCallFormat::Llama3Json).with_declarations(declarations());
        let tagged = r#"<|python_tag|>{"name":"f","arguments":{}}"#;
        assert_eq!(parser.push(tagged).calls.len(), 1);
        let mut parser =
            ToolCallParser::new(ToolCallFormat::Llama3Json).with_declarations(declarations());
        let untagged = r#"{"name":"f","arguments":{"x":"<|python_tag|>"}}"#;
        assert!(parser.push(untagged).calls.is_empty());
        assert_eq!(parser.finish().unwrap().calls.len(), 1);
    }

    #[test]
    fn llama_leading_whitespace_stays_bounded_and_becomes_content() {
        let mut parser = ToolCallParser::new(ToolCallFormat::Llama3Json);
        let mut content = String::new();
        let chunk = " ".repeat(64);
        for _ in 0..(MAX_TOOL_CALL_BYTES / chunk.len() + 8) {
            let chunks = parser.push(&chunk);
            content.push_str(&chunks.content);
            assert!(parser.pending.len() <= MAX_TOOL_CALL_BYTES);
        }
        assert!(content.len() >= MAX_TOOL_CALL_BYTES);
        assert!(content.chars().all(|character| character == ' '));
        assert!(parser.finish().unwrap().content.is_empty());
    }

    #[test]
    fn llama_untagged_call_after_whitespace_parses_at_every_split() {
        let text = "  {\"name\":\"f\",\"arguments\":{}}";
        for split in (0..=text.len()).filter(|&index| text.is_char_boundary(index)) {
            let mut parser =
                ToolCallParser::new(ToolCallFormat::Llama3Json).with_declarations(declarations());
            let first = parser.push(&text[..split]);
            let second = parser.push(&text[split..]);
            let end = parser.finish().unwrap();
            assert!(first.content.is_empty(), "split {split}");
            assert!(second.content.is_empty(), "split {split}");
            assert!(end.content.is_empty(), "split {split}");
            assert_eq!(
                first.calls.len() + second.calls.len() + end.calls.len(),
                1,
                "split {split}"
            );
        }
    }

    #[test]
    fn llama_whitespace_over_the_limit_is_content_not_a_call() {
        let whitespace = " ".repeat(MAX_LEADING_WHITESPACE_BYTES + 1);
        let text = format!("{whitespace}{{\"name\":\"f\",\"arguments\":{{}}}}");
        // The over-limit whitespace run is content at every fragment
        // boundary, so the object that follows it is content too.
        for split in 0..=text.len() {
            let mut parser =
                ToolCallParser::new(ToolCallFormat::Llama3Json).with_declarations(declarations());
            let first = parser.push(&text[..split]);
            let second = parser.push(&text[split..]);
            let end = parser.finish().unwrap();
            let content = first.content + &second.content + &end.content;
            assert_eq!(
                first.calls.len() + second.calls.len() + end.calls.len(),
                0,
                "split {split}"
            );
            assert_eq!(content, text, "split {split}");
        }
    }

    #[test]
    fn llama_tagged_call_after_over_limit_whitespace_is_parsed() {
        let whitespace = " ".repeat(MAX_LEADING_WHITESPACE_BYTES + 1);
        let text = format!("{whitespace}<|python_tag|>{{\"name\":\"f\",\"arguments\":{{}}}}");
        // The whitespace is content and the tagged call behind it is still
        // parsed, at every fragment boundary.
        for split in 0..=text.len() {
            let mut parser =
                ToolCallParser::new(ToolCallFormat::Llama3Json).with_declarations(declarations());
            let first = parser.push(&text[..split]);
            let second = parser.push(&text[split..]);
            let end = parser.finish().unwrap();
            let content = first.content + &second.content + &end.content;
            assert_eq!(
                first.calls.len() + second.calls.len() + end.calls.len(),
                1,
                "split {split}"
            );
            assert_eq!(content, whitespace, "split {split}");
        }
    }

    #[test]
    fn xml_scanned_values_at_every_split() {
        for (text, expected_args) in [
            (
                "<tool_call><function=f><parameter=x>abc\n</parameter>\ndef</parameter></function></tool_call>",
                serde_json::json!({"x":"abc\n</parameter>\ndef"}),
            ),
            (
                "<tool_call><function=f><parameter=x>\na\n</parameter> <parameter=y>\nq\n</parameter></function></tool_call>",
                serde_json::json!({"x":"a", "y":"q"}),
            ),
        ] {
            let expected = decode(ToolCallFormat::QwenXml, &[text]).unwrap();
            assert_eq!(expected.calls[0].arguments, expected_args);
            assert_all_splits(ToolCallFormat::QwenXml, text, &expected);
        }
    }

    #[test]
    fn xml_nonstructural_close_over_whitespace_limit_fails_at_every_split() {
        let text = format!(
            "<tool_call><function=f><parameter=x>v</parameter>{}more</parameter></function></tool_call>",
            " ".repeat(MAX_XML_CLOSING_WHITESPACE_BYTES + 1)
        );
        for split in 0..=text.len() {
            assert!(
                decode(ToolCallFormat::QwenXml, &[&text[..split], &text[split..]]).is_err(),
                "split {split}"
            );
        }
    }

    #[test]
    fn xml_names_are_validated_at_every_split() {
        use crate::grammar::CompiledJsonSchema;
        let schema = CompiledJsonSchema::compile_closed(&serde_json::json!({
            "type":"object", "properties":{"city":{"type":"string"}},
            "additionalProperties":false
        }))
        .unwrap();
        let declarations = CompiledToolCalls::new([(String::from("get_weather"), schema)]).unwrap();
        for (body, valid) in [
            ("<function=a b></function>", false),
            ("<function=f=g></function>", false),
            ("<function=f extra=x></function>", false),
            (
                "<function=get_weather><parameter=>v</parameter></function>",
                false,
            ),
            ("<function=f<tool_call></function>", false),
            (
                "<function=get_weather><parameter=city>Paris</parameter></function>",
                true,
            ),
        ] {
            let text = format!("<tool_call>{body}</tool_call>");
            for declared in [false, true] {
                for split in 0..=text.len() {
                    let mut parser = ToolCallParser::new(ToolCallFormat::QwenXml);
                    if declared {
                        parser = parser.with_declarations(declarations.clone());
                    }
                    let a = parser.push(&text[..split]);
                    let b = parser.push(&text[split..]);
                    let result = parser.finish();
                    assert_eq!(
                        result.is_ok(),
                        valid,
                        "{text}, declared {declared}, split {split}"
                    );
                    if valid {
                        let calls = a
                            .calls
                            .into_iter()
                            .chain(b.calls)
                            .chain(result.unwrap().calls)
                            .collect::<Vec<_>>();
                        assert_eq!(calls.len(), 1);
                        assert_eq!(calls[0].name, "get_weather");
                        assert_eq!(calls[0].arguments["city"], "Paris");
                    }
                }
            }
        }
    }

    #[test]
    fn xml_structural_close_over_whitespace_limit_fails_at_every_split() {
        let text = format!(
            "<tool_call><function=f><parameter=x>v</parameter>{}</function></tool_call>",
            " ".repeat(MAX_XML_CLOSING_WHITESPACE_BYTES + 1)
        );
        for split in 0..=text.len() {
            assert!(
                decode(ToolCallFormat::QwenXml, &[&text[..split], &text[split..]]).is_err(),
                "split {split}"
            );
        }
    }

    #[test]
    fn xml_mixed_framing_scanner_rule() {
        let text = "<tool_call>\n<function=f>\n<parameter=x>\na\n</parameter> <parameter=y>z\n</parameter>\n</function>\n</tool_call>";
        let parsed = decode(ToolCallFormat::QwenXml, &[text]).unwrap();
        assert_eq!(
            parsed.calls[0].arguments,
            serde_json::json!({"x":"a", "y":"z\n"})
        );
        assert_all_splits(ToolCallFormat::QwenXml, text, &parsed);
    }

    #[test]
    fn xml_partial_parameter_after_existing_name_is_incomplete() {
        let text = "<tool_call><function=f><parameter=x>b</parameter><parameter=";
        for split in 0..=text.len() {
            let mut parser = ToolCallParser::new(ToolCallFormat::QwenXml);
            parser.push(&text[..split]);
            parser.push(&text[split..]);
            assert!(parser.is_incomplete(), "split {split}");
        }
    }

    #[test]
    fn xml_incomplete_probe_is_bounded_and_avoids_name_suffixes() {
        use std::time::{Duration, Instant};
        let text = format!(
            "<function=f><parameter=prefix__probe0>v</parameter><parameter=prefix{}",
            "x".repeat(200_000)
        );
        let start = Instant::now();
        assert!(xml_is_incomplete(&text));
        assert!(start.elapsed() < Duration::from_secs(30));

        assert!(xml_is_incomplete(
            "<function=f><parameter=prefix__probe0>v</parameter><parameter=prefix"
        ));
    }

    #[test]
    fn xml_incomplete_probe_handles_many_occupied_names() {
        use std::fmt::Write;
        let mut text = String::from("<function=f>");
        for i in 0..1000 {
            write!(text, "<parameter=__probe{i}>v</parameter>").unwrap();
        }
        text.push_str("<parameter=__probe");
        assert!(xml_is_incomplete(&text));
        assert!(!xml_is_incomplete(
            "<function=f><parameter=bad name>v</parameter><parameter="
        ));
    }

    #[test]
    fn xml_complete_call_is_not_incomplete() {
        let complete = "<function=f><parameter=x>v</parameter></function></tool_call>";
        // A text that already contains the whole ending is complete.
        assert!(!xml_is_incomplete(complete));
        // Without the closing marker the same call is still unfinished.
        assert!(xml_is_incomplete(
            "<function=f><parameter=x>v</parameter></function>"
        ));
    }

    #[test]
    fn xml_uses_declared_types() {
        for text in [
            "<tool_call><function=f><parameter=x>12345</parameter><parameter=n>12345</parameter></function></tool_call>",
            "<tool_call>\n<function=f>\n<parameter=x>\n12345\n</parameter>\n<parameter=n>\n12345\n</parameter>\n</function>\n</tool_call>",
        ] {
            for split in 0..=text.len() {
                let mut parser =
                    ToolCallParser::new(ToolCallFormat::QwenXml).with_declarations(declarations());
                let first = parser.push(&text[..split]);
                let second = parser.push(&text[split..]);
                let end = parser.finish().unwrap();
                assert!(first.content.is_empty());
                assert!(second.content.is_empty());
                assert!(end.content.is_empty());
                let calls: Vec<_> = first
                    .calls
                    .into_iter()
                    .chain(second.calls)
                    .chain(end.calls)
                    .collect();
                assert_eq!(calls.len(), 1);
                assert_eq!(
                    calls[0].arguments,
                    serde_json::json!({"x":"12345", "n":12345})
                );
            }
        }
    }

    #[test]
    fn python_triple_quoted_strings() {
        for (text, expected) in [
            (
                "<|tool_call_start|>[f(x='''hello\n<|tool_call_end|> \\' world''')]<|tool_call_end|>",
                "hello\n<|tool_call_end|> ' world",
            ),
            (
                "<|tool_call_start|>[f(x=\"\"\"hello\n<|tool_call_end|> \\\" world\"\"\")]<|tool_call_end|>",
                "hello\n<|tool_call_end|> \" world",
            ),
            (
                r"<|tool_call_start|>[f(x='''one ' two '' \u732b\t\\''')]<|tool_call_end|>",
                "one ' two '' 猫\t\\",
            ),
            (
                "<|tool_call_start|>[f(x='''escaped \\'\\'\\'\ncontinued\\\nline''')]<|tool_call_end|>",
                "escaped '''\ncontinuedline",
            ),
            (
                "<|tool_call_start|>[f(x=\"\"\"\"\"\") ,f(x='''''')]<|tool_call_end|>",
                "",
            ),
        ] {
            let result = decode(ToolCallFormat::Pythonic, &[text]).unwrap();
            assert_eq!(result.calls[0].arguments["x"], expected);
            assert_all_splits(ToolCallFormat::Pythonic, text, &result);
        }
        for text in [
            "<|tool_call_start|>[f(x='''unfinished '')]<|tool_call_end|>",
            "<|tool_call_start|>[f(x=\"\"\"unfinished \"\")]<|tool_call_end|>",
        ] {
            assert!(decode(ToolCallFormat::Pythonic, &[text]).is_err());
        }
    }

    #[test]
    fn llama_ambiguous_content_and_fragmented_markers() {
        let tagged = r#"<|python_tag|>{"name":"f","arguments":{}}"#;
        for split in 0..=tagged.len() {
            let mut parser =
                ToolCallParser::new(ToolCallFormat::Llama3Json).with_declarations(declarations());
            let mut content = parser.push("  ").content;
            let first = parser.push(&tagged[..split]);
            content.push_str(&first.content);
            let second = parser.push(&tagged[split..]);
            content.push_str(&second.content);
            let end = parser.finish().unwrap();
            content.push_str(&end.content);
            assert_eq!(content, "  ");
            assert_eq!(first.calls.len() + second.calls.len() + end.calls.len(), 1);
        }
        let text = format!(r#"{{"ordinary":"<|python_tag|>"}}{tagged}"#);
        for split in 0..=text.len() {
            let mut parser =
                ToolCallParser::new(ToolCallFormat::Llama3Json).with_declarations(declarations());
            let first = parser.push(&text[..split]);
            let second = parser.push(&text[split..]);
            let end = parser.finish().unwrap();
            assert_eq!(
                format!("{}{}{}", first.content, second.content, end.content),
                r#"{"ordinary":"<|python_tag|>"}"#
            );
            assert_eq!(first.calls.len() + second.calls.len() + end.calls.len(), 1);
        }
        let ordinary = format!(r#"{{"text":"{}"}}"#, "x".repeat(300 * 1024));
        let mut parser =
            ToolCallParser::new(ToolCallFormat::Llama3Json).with_declarations(declarations());
        for byte in ordinary.as_bytes() {
            assert!(
                parser
                    .push(std::str::from_utf8(std::slice::from_ref(byte)).unwrap())
                    .content
                    .is_empty()
            );
        }
        assert!(parser.finish().is_err());
        let large_call = format!(
            r#"{{"name":"f","arguments":{{"x":"{}"}}}}"#,
            "x".repeat(300 * 1024)
        );
        let mut parser =
            ToolCallParser::new(ToolCallFormat::Llama3Json).with_declarations(declarations());
        parser.push(&large_call);
        assert!(parser.finish().is_err());
    }

    #[test]
    fn hermes_detection_requires_json_call_style() {
        assert_eq!(
            detect(
                "<tool_call>{{ {'name': call.name, 'arguments': call.arguments} | tojson }}</tool_call>"
            ),
            Some(ToolCallFormat::Hermes)
        );
        assert_eq!(
            detect(r#"<tool_call><arg_key> tool {"name": "x", "parameters": {}}"#),
            None
        );
        assert_eq!(
            detect("<tool_call><arg_key>x</arg_key><arg_value>y</arg_value>"),
            None
        );
    }

    #[test]
    fn detection_uses_markers_and_the_llama_call_prompt() {
        assert_eq!(
            detect(r#"Use a tool with {"name": "function_name", "parameters": {}}"#),
            Some(ToolCallFormat::Llama3Json)
        );
        assert_eq!(detect("{{ tool.name }} {{ tool.parameters }}"), None);
        assert_eq!(detect("<tool_cal> [TOOL_CALL] <|tool_call_start>"), None);
        assert_eq!(
            detect("<tool_call><function=f>"),
            Some(ToolCallFormat::QwenXml)
        );
        assert_eq!(
            detect("<tool_call><function=f> <|tool_call_start|>"),
            Some(ToolCallFormat::Pythonic)
        );
    }

    #[test]
    fn json_variants_and_quoted_markers_are_chunk_invariant() {
        for (format, text, arguments) in [
            (
                ToolCallFormat::MistralJson,
                "lead [TOOL_CALLS]f[ARGS]{\"x\":\"猫\"} tail",
                serde_json::json!({"x":"猫"}),
            ),
            (
                ToolCallFormat::Hermes,
                "lead <tool_call>{\"name\":\"f\",\"arguments\":{\"x\":\"</tool_call>\"}}</tool_call> tail",
                serde_json::json!({"x":"</tool_call>"}),
            ),
            (
                ToolCallFormat::Pythonic,
                "lead <|tool_call_start|>[f(x='<|tool_call_end|>')]<|tool_call_end|> tail",
                serde_json::json!({"x":"<|tool_call_end|>"}),
            ),
        ] {
            let expected = ParsedEnd {
                content: "lead  tail".into(),
                calls: vec![NativeToolCall {
                    id: None,
                    name: "f".into(),
                    arguments_text: arguments.to_string(),
                    arguments,
                }],
            };
            assert_all_splits(format, text, &expected);
        }

        let object = r#"{"name":"f","parameters":{"x":"<|python_tag|>"}}"#;
        for text in [
            format!("<|python_tag|>{object}\n{object}"),
            format!("<|python_tag|>{object};<|python_tag|>{object}"),
        ] {
            let expected = decode(ToolCallFormat::Llama3Json, &[&text]).unwrap();
            assert_eq!(expected.calls.len(), 2);
            assert_eq!(expected.calls[0].arguments["x"], "<|python_tag|>");
            assert_all_splits(ToolCallFormat::Llama3Json, &text, &expected);
        }
    }

    #[test]
    fn compact_xml_and_raw_parameter_markers_are_chunk_invariant() {
        for text in [
            "lead <tool_call><function=f><parameter=x>猫</parameter></function></tool_call> tail",
            "lead <tool_call>\n<function=f>\n<parameter=x>\n猫\n</parameter>\n</function>\n</tool_call> tail",
        ] {
            let expected = decode(ToolCallFormat::QwenXml, &[text]).unwrap();
            assert_eq!(expected.content, "lead  tail");
            assert_eq!(expected.calls.len(), 1);
            assert_eq!(expected.calls[0].arguments, serde_json::json!({"x":"猫"}));
            assert_all_splits(ToolCallFormat::QwenXml, text, &expected);
        }
        for text in [
            "<tool_call><function=f><parameter=x>literal </tool_call> text</parameter></function></tool_call>",
            "<tool_call>\n<function=f>\n<parameter=x>\nliteral </tool_call> text\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=f>\n<parameter=x>\nliteral </parameter> text\n</parameter>\n</function>\n</tool_call>",
        ] {
            let expected = decode(ToolCallFormat::QwenXml, &[text]).unwrap();
            assert_eq!(expected.calls.len(), 1);
            assert!(
                expected.calls[0].arguments["x"]
                    .as_str()
                    .unwrap()
                    .starts_with("literal </")
            );
            assert_all_splits(ToolCallFormat::QwenXml, text, &expected);
        }
    }

    #[test]
    fn content_streams_while_only_a_possible_marker_prefix_is_held() {
        for format in [
            ToolCallFormat::Hermes,
            ToolCallFormat::QwenXml,
            ToolCallFormat::Llama3Json,
            ToolCallFormat::MistralJson,
            ToolCallFormat::GraniteJson,
            ToolCallFormat::Pythonic,
        ] {
            let mut parser = ToolCallParser::new(format);
            let marker = parser.marker();
            assert_eq!(parser.push("visible 猫").content, "visible 猫");
            assert!(parser.push(&marker[..marker.len() - 1]).content.is_empty());
            assert_eq!(
                parser.push("!").content,
                format!("{}!", &marker[..marker.len() - 1])
            );
            assert_eq!(parser.finish().unwrap(), ParsedEnd::default());
            let partial = format!("text {}", &marker[..marker.len() - 1]);
            let expected = ParsedEnd {
                content: partial.clone(),
                calls: vec![],
            };
            assert_all_splits(format, &partial, &expected);
        }
    }

    #[test]
    fn completed_calls_are_released_by_push() {
        for (format, text) in [
            (
                ToolCallFormat::Hermes,
                "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>",
            ),
            (
                ToolCallFormat::QwenXml,
                "<tool_call>\n<function=f>\n</function>\n</tool_call>",
            ),
            (
                ToolCallFormat::Llama3Json,
                "<|python_tag|>{\"name\":\"f\",\"parameters\":{}}",
            ),
            (ToolCallFormat::MistralJson, "[TOOL_CALLS]f[ARGS]{}"),
            (
                ToolCallFormat::GraniteJson,
                "<|tool_call|>[{\"name\":\"f\",\"arguments\":{}}]",
            ),
            (
                ToolCallFormat::Pythonic,
                "<|tool_call_start|>[f()]<|tool_call_end|>",
            ),
        ] {
            let mut parser = ToolCallParser::new(format);
            assert_eq!(parser.push(text).calls.len(), 1, "{format:?}");
            assert_eq!(parser.finish().unwrap(), ParsedEnd::default());
        }
    }
    #[test]
    fn tagged_calls_are_chunk_invariant() {
        for (format, text) in [
            (
                ToolCallFormat::Hermes,
                "hi<tool_call>{\"name\":\"f\",\"arguments\":{\"x\":1}}</tool_call>",
            ),
            (
                ToolCallFormat::Pythonic,
                "hi<|tool_call_start|>[f(x=1)]<|tool_call_end|>",
            ),
        ] {
            for split in 0..=text.len() {
                let mut parser = ToolCallParser::new(format);
                let first = parser.push(&text[..split]);
                let second = parser.push(&text[split..]);
                let end = parser.finish().unwrap();
                assert_eq!(
                    format!("{}{}{}", first.content, second.content, end.content),
                    "hi"
                );
                let calls: Vec<_> = first
                    .calls
                    .into_iter()
                    .chain(second.calls)
                    .chain(end.calls)
                    .collect();
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "f");
                assert_eq!(calls[0].arguments, serde_json::json!({"x":1}));
            }
        }
    }
    #[test]
    fn all_formats_detect_and_parse_at_each_boundary() {
        use ToolCallFormat::{GraniteJson, Hermes, Llama3Json, MistralJson, Pythonic, QwenXml};
        let cases = [
            (
                Hermes,
                "<tool_call> \"name\": \"arguments\":",
                "<tool_call>{\"name\":\"f\",\"arguments\":{\"x\":\"猫\"}}</tool_call>",
            ),
            (
                QwenXml,
                "<tool_call> <function=",
                "<tool_call>\n<function=f>\n<parameter=x>\n猫\n</parameter>\n</function>\n</tool_call>",
            ),
            (
                Llama3Json,
                "<|python_tag|>",
                "<|python_tag|>{\"name\":\"f\",\"parameters\":{\"x\":\"猫\"}}",
            ),
            (
                MistralJson,
                "[TOOL_CALLS]",
                "[TOOL_CALLS][{\"name\":\"f\",\"arguments\":{\"x\":\"猫\"}}]",
            ),
            (
                GraniteJson,
                "<|tool_call|>",
                "<|tool_call|>[{\"name\":\"f\",\"arguments\":{\"x\":\"猫\"}}]",
            ),
            (
                Pythonic,
                "<|tool_call_start|>",
                "<|tool_call_start|>[f(x='猫')]<|tool_call_end|>",
            ),
        ];
        for (format, template, call) in cases {
            assert_eq!(detect(template), Some(format));
            let text = format!("hello {call}");
            for split in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
                let mut parser = ToolCallParser::new(format);
                let a = parser.push(&text[..split]);
                let b = parser.push(&text[split..]);
                let end = parser.finish().unwrap();
                assert_eq!(
                    format!("{}{}{}", a.content, b.content, end.content),
                    "hello "
                );
                let calls: Vec<_> = a
                    .calls
                    .into_iter()
                    .chain(b.calls)
                    .chain(end.calls)
                    .collect();
                assert_eq!(calls.len(), 1, "{format:?} at {split}");
                assert_eq!(calls[0].name, "f");
                assert_eq!(calls[0].arguments, serde_json::json!({"x":"猫"}));
            }
        }
        assert_eq!(detect("ordinary template"), None);
    }
    #[test]
    fn malformed_calls_are_errors() {
        for (format, text) in [
            (ToolCallFormat::Hermes, "<tool_call>{bad}</tool_call>"),
            (ToolCallFormat::QwenXml, "<tool_call>bad</tool_call>"),
            (ToolCallFormat::Llama3Json, "<|python_tag|>{bad}"),
            (ToolCallFormat::MistralJson, "[TOOL_CALLS][bad]"),
            (ToolCallFormat::GraniteJson, "<|tool_call|>[bad]"),
            (
                ToolCallFormat::Pythonic,
                "<|tool_call_start|>[f(x=bad)]<|tool_call_end|>",
            ),
            (ToolCallFormat::MistralJson, "[TOOL_CALLS]f[ARGS]{bad}"),
            (
                ToolCallFormat::Hermes,
                "<tool_call>{\"name\":\"f\",\"arguments\":[]}</tool_call>",
            ),
            (
                ToolCallFormat::Llama3Json,
                "<|python_tag|>{\"name\":\"f\",\"parameters\":null}",
            ),
            (
                ToolCallFormat::Pythonic,
                "<|tool_call_start|>[f(x=__import__('os'))]<|tool_call_end|>",
            ),
            (
                ToolCallFormat::Pythonic,
                "<|tool_call_start|>[f(x=1, x=2)]<|tool_call_end|>",
            ),
        ] {
            for split in 0..=text.len() {
                assert!(
                    decode(format, &[&text[..split], &text[split..]]).is_err(),
                    "{format:?} at {split}"
                );
            }
        }
    }

    #[test]
    fn unterminated_calls_are_errors_and_never_content() {
        for (format, text) in [
            (
                ToolCallFormat::Hermes,
                "<tool_call>{\"name\":\"f\",\"arguments\":{}}",
            ),
            (ToolCallFormat::QwenXml, "<tool_call>\n<function=f>\n"),
            (ToolCallFormat::Llama3Json, "<|python_tag|>{\"name\":\"f\","),
            (ToolCallFormat::MistralJson, "[TOOL_CALLS]f[AR"),
            (ToolCallFormat::GraniteJson, "<|tool_call|>["),
            (
                ToolCallFormat::Pythonic,
                "<|tool_call_start|>[f(x='unfinished",
            ),
        ] {
            let mut parser = ToolCallParser::new(format);
            let chunks = parser.push(text);
            assert_eq!(chunks, ParsedChunks::default());
            assert!(parser.finish().is_err(), "{format:?}");
        }
    }

    #[test]
    fn incomplete_status_is_chunk_invariant_and_ignores_quoted_closing_markers() {
        use ToolCallFormat::{GraniteJson, Hermes, Llama3Json, MistralJson, Pythonic, QwenXml};
        for (format, text, incomplete) in [
            (Hermes, "<tool_call>{\"name\":\"f\",", true),
            (Hermes, "<tool_call>{bad}</tool_call>", false),
            (
                Hermes,
                "<tool_call>{\"name\":\"f\",\"arguments\":{\"x\":\"</tool_call> unfinished",
                true,
            ),
            (QwenXml, "<tool_call><function=f><parameter=x>", true),
            (
                QwenXml,
                "<tool_call><function=f><parameter=x>literal </tool_call> unfinished",
                true,
            ),
            (
                QwenXml,
                "<tool_call>\n<function=f>\n<parameter=x>\nliteral </tool_call>",
                true,
            ),
            (
                QwenXml,
                "<tool_call><function=f><parameter=x>value</par",
                true,
            ),
            (QwenXml, "<tool_call>bad</tool_call>", false),
            (
                QwenXml,
                "<tool_call><function=f><parameter=x>1</parameter><parameter=x>2</parameter></function></tool_call>",
                false,
            ),
            (Llama3Json, "<|python_tag|>{\"name\":\"f\",", true),
            (Llama3Json, "<|python_tag|>{bad}", false),
            (Llama3Json, "{\"name\":\"f\",\"parameters\":{}};\n", false),
            (MistralJson, "[TOOL_CALLS][{\"name\":\"f\",", true),
            (MistralJson, "[TOOL_CALLS]f[AR", true),
            (MistralJson, "[TOOL_CALLS][bad]", false),
            (GraniteJson, "<|tool_call|>[{\"name\":\"f\",", true),
            (GraniteJson, "<|tool_call|>[bad]", false),
            (Pythonic, "<|tool_call_start|>[f(x='unfinished", true),
            (
                Pythonic,
                "<|tool_call_start|>[f(x=bad)]<|tool_call_end|>",
                false,
            ),
            (
                Pythonic,
                "<|tool_call_start|>[f(x='<|tool_call_end|> unfinished",
                true,
            ),
        ] {
            for split in 0..=text.len() {
                let mut parser = ToolCallParser::new(format);
                parser.push(&text[..split]);
                parser.push(&text[split..]);
                assert_eq!(parser.is_incomplete(), incomplete, "{format:?} at {split}");
            }
        }
    }

    #[test]
    fn xml_prefixes_remain_incomplete_until_the_envelope_closes() {
        for text in [
            "<tool_call><function=f><parameter=x>literal </tool_call></parameter></function></tool_call>",
            "<tool_call>\n<function=f>\n<parameter=x>\nliteral </tool_call>\n</parameter>\n</function>\n</tool_call>",
        ] {
            for end in "<tool_call>".len()..text.len() {
                let mut parser = ToolCallParser::new(ToolCallFormat::QwenXml);
                parser.push(&text[..end]);
                assert!(parser.is_incomplete(), "prefix {:?}", &text[..end]);
            }
            let mut parser = ToolCallParser::new(ToolCallFormat::QwenXml);
            assert_eq!(parser.push(text).calls.len(), 1);
            assert!(!parser.is_incomplete());
        }
    }

    #[test]
    fn two_calls_and_text_lookalikes() {
        use ToolCallFormat::{GraniteJson, Hermes, Llama3Json, MistralJson, Pythonic, QwenXml};
        for (format, text) in [
            (
                Hermes,
                "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"g\",\"arguments\":{}}</tool_call>",
            ),
            (
                QwenXml,
                "<tool_call>\n<function=f>\n</function>\n</tool_call><tool_call>\n<function=g>\n</function>\n</tool_call>",
            ),
            (
                Llama3Json,
                "<|python_tag|>{\"name\":\"f\",\"parameters\":{}};{\"name\":\"g\",\"parameters\":{}}",
            ),
            (
                MistralJson,
                "[TOOL_CALLS][{\"name\":\"f\",\"arguments\":{}},{\"name\":\"g\",\"arguments\":{}}]",
            ),
            (
                GraniteJson,
                "<|tool_call|>[{\"name\":\"f\",\"arguments\":{}},{\"name\":\"g\",\"arguments\":{}}]",
            ),
            (Pythonic, "<|tool_call_start|>[f(),g()]<|tool_call_end|>"),
        ] {
            for split in 0..=text.len() {
                let mut parser = ToolCallParser::new(format);
                let a = parser.push(&text[..split]);
                let b = parser.push(&text[split..]);
                let end = parser.finish().unwrap();
                let calls: Vec<_> = a
                    .calls
                    .into_iter()
                    .chain(b.calls)
                    .chain(end.calls)
                    .collect();
                assert_eq!(
                    calls
                        .iter()
                        .map(|call| call.name.as_str())
                        .collect::<Vec<_>>(),
                    vec!["f", "g"],
                    "{format:?} {split}"
                );
            }
            let mut parser = ToolCallParser::new(format);
            let a = parser.push("ordinary <tool_cal or [TOOL_CALL] text");
            let end = parser.finish().unwrap();
            assert_eq!(
                a.content + &end.content,
                "ordinary <tool_cal or [TOOL_CALL] text"
            );
            assert!(a.calls.is_empty() && end.calls.is_empty());
        }
        let mut parser = ToolCallParser::new(MistralJson);
        let a = parser.push("[TOOL_CALLS]f[ARGS]{\"x\":1}");
        let end = parser.finish().unwrap();
        assert_eq!(
            a.calls
                .into_iter()
                .chain(end.calls)
                .next()
                .unwrap()
                .arguments,
            serde_json::json!({"x":1})
        );
    }
    #[test]
    fn python_literals_support_nested_values_without_execution() {
        let mut parser = ToolCallParser::new(ToolCallFormat::Pythonic);
        let chunks = parser.push(r"<|tool_call_start|>[f(a=[1,-2.5,True,False,None], b={'x':'\u732b'}, c=+.5, d=1.e+2, e='can\'t\\\n\t\x61\141')]<|tool_call_end|>");
        let end = parser.finish().unwrap();
        let call = chunks.calls.into_iter().chain(end.calls).next().unwrap();
        assert_eq!(
            call.arguments,
            serde_json::json!({"a":[1,-2.5,true,false,null],"b":{"x":"猫"},"c":0.5,"d":100.0,"e":"can't\\\n\taa"})
        );
    }

    #[test]
    fn python_numbers_accept_literal_forms_and_reject_expressions() {
        for (source, expected) in [
            ("0x_Ff", serde_json::json!(255)),
            ("-0b101", serde_json::json!(-5)),
            ("0o17", serde_json::json!(15)),
            ("1_000", serde_json::json!(1000)),
            ("00", serde_json::json!(0)),
            ("1.e+2", serde_json::json!(100.0)),
            (".5e-1", serde_json::json!(0.05)),
            ("01.5", serde_json::json!(1.5)),
        ] {
            let text = format!("<|tool_call_start|>[f(x={source})]<|tool_call_end|>");
            let parsed = decode(ToolCallFormat::Pythonic, &[&text]).unwrap();
            assert_eq!(parsed.calls[0].arguments["x"], expected);
            assert_all_splits(ToolCallFormat::Pythonic, &text, &parsed);
        }
        for source in [
            "+",
            "--0",
            "--",
            "-+0",
            "+-0",
            "+-1",
            "+-1.5",
            "0x+1",
            "-",
            "1+2",
            "1__2",
            "1_",
            "_1",
            "1e_2",
            "0x__f",
            "012",
            "float('inf')",
        ] {
            let text = format!("<|tool_call_start|>[f(x={source})]<|tool_call_end|>");
            assert!(
                decode(ToolCallFormat::Pythonic, &[&text]).is_err(),
                "{source}"
            );
        }
    }

    /// A call in each format the generic path understands.
    fn sample_calls() -> [(ToolCallFormat, &'static str); 5] {
        [
            (
                ToolCallFormat::Hermes,
                "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>",
            ),
            (
                ToolCallFormat::QwenXml,
                "<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n</function>\n</tool_call>",
            ),
            (
                ToolCallFormat::Pythonic,
                "<|tool_call_start|>[f()]<|tool_call_end|>",
            ),
            (ToolCallFormat::MistralJson, "[TOOL_CALLS]f[ARGS]{}"),
            (
                ToolCallFormat::GraniteJson,
                "<|tool_call|>[{\"name\":\"f\",\"arguments\":{}}]",
            ),
        ]
    }

    #[test]
    fn whitespace_after_a_call_separates_calls_from_content() {
        for (format, call) in sample_calls() {
            // Two calls separated by a newline: the newline separates them.
            let text = format!("{call}\n{call}");
            let expected = decode(format, &[&text]).unwrap();
            assert_eq!(expected.content, "", "{format:?}");
            assert_eq!(expected.calls.len(), 2, "{format:?}");
            assert_all_splits(format, &text, &expected);

            // A call and a trailing newline: the newline separates the call
            // from the end of the turn.
            let text = format!("{call}\n");
            let expected = decode(format, &[&text]).unwrap();
            assert_eq!(expected.content, "", "{format:?}");
            assert_eq!(expected.calls.len(), 1, "{format:?}");
            assert_all_splits(format, &text, &expected);

            // Text after the newline: the newline and the text are content.
            let text = format!("{call}\ndone");
            let expected = decode(format, &[&text]).unwrap();
            assert_eq!(expected.content, "\ndone", "{format:?}");
            assert_eq!(expected.calls.len(), 1, "{format:?}");
            assert_all_splits(format, &text, &expected);

            // A partial marker after the newline: the newline and the
            // partial marker are content, because no call follows them.
            let marker = ToolCallParser::new(format).marker();
            let partial = &marker[..marker.len() - 1];
            let text = format!("{call}\n{partial}");
            let expected = decode(format, &[&text]).unwrap();
            assert_eq!(expected.content, format!("\n{partial}"), "{format:?}");
            assert_eq!(expected.calls.len(), 1, "{format:?}");
            assert_all_splits(format, &text, &expected);
        }
    }

    #[test]
    fn a_whitespace_run_past_the_limit_after_a_call_is_content() {
        for (format, call) in sample_calls() {
            let whitespace = " ".repeat(MAX_LEADING_WHITESPACE_BYTES + 1);
            let text = format!("{call}{whitespace}{call}");
            let expected = decode(format, &[&text]).unwrap();
            assert_eq!(expected.content, whitespace, "{format:?}");
            assert_eq!(expected.calls.len(), 2, "{format:?}");
            assert_all_splits(format, &text, &expected);
        }
    }

    #[test]
    fn llama_whitespace_between_calls_is_never_content() {
        let call = "<|python_tag|>{\"name\":\"f\",\"arguments\":{}}";
        // Two calls separated by a newline: the newline separates them.
        let text = format!("{call}\n{call}");
        let expected = decode(ToolCallFormat::Llama3Json, &[&text]).unwrap();
        assert_eq!(expected.content, "", "llama");
        assert_eq!(expected.calls.len(), 2, "llama");
        assert_all_splits(ToolCallFormat::Llama3Json, &text, &expected);

        // A call and a trailing newline: the newline separates the call
        // from the end of the turn.
        let text = format!("{call}\n");
        let expected = decode(ToolCallFormat::Llama3Json, &[&text]).unwrap();
        assert_eq!(expected.content, "", "llama");
        assert_eq!(expected.calls.len(), 1, "llama");
        assert_all_splits(ToolCallFormat::Llama3Json, &text, &expected);

        // A run of newlines past the limit is still a separator.
        let whitespace = "\n".repeat(MAX_LEADING_WHITESPACE_BYTES + 1);
        let text = format!("{call}{whitespace}{call}");
        let expected = decode(ToolCallFormat::Llama3Json, &[&text]).unwrap();
        assert_eq!(expected.content, "", "llama");
        assert_eq!(expected.calls.len(), 2, "llama");
        assert_all_splits(ToolCallFormat::Llama3Json, &text, &expected);
    }

    #[test]
    fn text_without_calls_is_unchanged() {
        for format in [
            ToolCallFormat::Hermes,
            ToolCallFormat::QwenXml,
            ToolCallFormat::MistralJson,
            ToolCallFormat::GraniteJson,
            ToolCallFormat::Pythonic,
            ToolCallFormat::Llama3Json,
        ] {
            for text in ["  leading", "plain text", "  ", "\n\t "] {
                let expected = decode(format, &[text]).unwrap();
                assert_eq!(expected.content, text, "{format:?} {text:?}");
                assert!(expected.calls.is_empty(), "{format:?} {text:?}");
                assert_all_splits(format, text, &expected);
            }
        }
    }
}
