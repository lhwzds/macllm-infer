//! A generic Jinja chat-template renderer for unpinned checkpoints.
//!
//! Every verified model family keeps its byte-pinned renderer. A checkpoint
//! whose `tokenizer.chat_template` is not one of those programs can still
//! chat: [`JinjaChatTemplate`] compiles the embedded template once and
//! executes it with the environment Hugging Face `apply_chat_template`
//! documents (see `tests/fixtures/chat_templates/README.md`):
//! `trim_blocks`, `lstrip_blocks`, `loopcontrols`, Python string, list and
//! dict methods, `tojson` equal to `json.dumps(ensure_ascii=False)` with the
//! default separators and insertion order, `raise_exception` and
//! `strftime_now`.
//!
//! # The threat model
//!
//! The template travels with the GGUF file, and the GGUF file is the one the
//! operator chose: it is trusted like the model weights it carries. The
//! renderer is therefore not a sandbox for the template. It bounds the work a
//! template can schedule, so that a template which loops forever, recurses
//! forever or writes forever stops with an ordinary error instead of hanging
//! the process: a render fails when it exceeds the fuel budget
//! (`TEMPLATE_FUEL`), the recursion limit (`TEMPLATE_RECURSION_LIMIT`), the
//! size of the rendered prompt (`MAX_RENDERED_BYTES`) or the size of a single
//! `tojson` result (`MAX_JSON_BYTES`), and a source larger than
//! `MAX_TEMPLATE_BYTES` is rejected before it is parsed.
//!
//! The renderer's own `tojson` writes its result while it walks the value, so
//! a lazy value stops at the `tojson` limit instead of being materialized
//! first.
//!
//! These bounds are generic. They do not harden every built-in filter
//! against an allocation bomb: a template can still request a large
//! allocation through a built-in filter, and MiniJinja performs that
//! allocation before the renderer ever sees its result. Only load GGUF files
//! you trust.
//!
//! The module only depends on the chat contracts, so it stays usable from any
//! crate that can depend on `macllm-chat`.

use std::io;

use anyhow::{Context, Result};
use minijinja::value::{Kwargs, Value as JinjaValue, ValueKind};
use minijinja::{AutoEscape, Environment, Error, ErrorKind};

use crate::chat::{ChatMessage, ChatTemplateOptions, ChatToolCall};

/// The single template an instance owns.
const TEMPLATE_NAME: &str = "chat_template";

/// The largest template source that may be compiled, in bytes.
///
/// A checkpoint may embed a megabyte of template text. A larger source is
/// rejected before the parser sees it, so a large file cannot make the
/// compiler allocate for it.
const MAX_TEMPLATE_BYTES: usize = 1024 * 1024;

/// The instruction budget of one render, in MiniJinja's fuel units.
///
/// MiniJinja charges one unit per executed instruction, so the budget bounds
/// the work a template can schedule: an endless loop runs out of fuel and the
/// render fails with an ordinary error.
///
/// The measured maximum over every template and every case in
/// `tests/fixtures/chat_templates` is 798 units across 144 renders, so this
/// budget leaves the fixtures more than a hundred times the headroom they
/// need. A conversation of a thousand messages costs about fifty thousand
/// units, so the budget still holds a very long conversation.
const TEMPLATE_FUEL: u64 = 100_000;

/// The deepest template recursion, in stack frames.
///
/// MiniJinja counts every macro call, block and loop frame against this
/// limit, so a recursive macro fails with an ordinary error instead of
/// exhausting the thread stack.
const TEMPLATE_RECURSION_LIMIT: usize = 64;

/// The largest rendered prompt, in bytes.
///
/// The prompt is written through a capped writer, so a template that writes
/// more than this stops the render with an ordinary error instead of growing
/// the buffer without a bound.
const MAX_RENDERED_BYTES: usize = 8 * 1024 * 1024;

/// The largest `tojson` result, in bytes.
///
/// The result becomes part of the prompt, so a result larger than the prompt
/// limit is never useful to a template. The result is written while the value
/// is walked, so the limit also bounds what a lazy value can ask for.
const MAX_JSON_BYTES: usize = 8 * 1024 * 1024;

/// The largest indentation `tojson` accepts, in spaces or in bytes of the
/// literal indentation text.
///
/// The indentation is repeated for every nesting level of the result, so a
/// large one would multiply the size of the output. The limit is checked
/// before the indentation is built.
const MAX_JSON_INDENT: usize = 16;

/// A compiled Hugging Face-style chat template.
///
/// The template is parsed once by [`JinjaChatTemplate::new`], which fails on
/// a syntax error, and rendered any number of times afterwards.
#[derive(Debug, Clone)]
pub struct JinjaChatTemplate {
    environment: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl JinjaChatTemplate {
    /// Compile `source`, keeping the special tokens the template may print.
    ///
    /// `bos_token` and `eos_token` are the tokenizer's own strings; they are
    /// passed verbatim and empty when a checkpoint does not declare them.
    pub fn new(source: &str, bos_token: &str, eos_token: &str) -> Result<Self> {
        anyhow::ensure!(
            source.len() <= MAX_TEMPLATE_BYTES,
            "the chat template is larger than the {MAX_TEMPLATE_BYTES} byte limit"
        );
        let mut environment = Environment::new();
        environment.set_trim_blocks(true);
        environment.set_lstrip_blocks(true);
        // Chat templates produce plain prompts, never HTML.
        environment.set_auto_escape_callback(|_| AutoEscape::None);
        // Python-compatible `.strip()`, `.split()`, `.startswith()`,
        // `.endswith()`, `.items()`, `.get()` and friends.
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // transformers installs this exact `tojson`; MiniJinja's built-in
        // filter escapes HTML and uses compact separators instead.
        environment.add_filter("tojson", tojson);
        environment.add_function("raise_exception", raise_exception);
        environment.add_function("strftime_now", strftime_now);
        // A template may not run forever: the fuel budget bounds the number of
        // instructions one render may execute.
        environment.set_fuel(Some(TEMPLATE_FUEL));
        // Nor may it recurse without a bound.
        environment.set_recursion_limit(TEMPLATE_RECURSION_LIMIT);
        // The environment owns no loader, so the only template it can reach is
        // the one below: `include` and `import` reach neither the filesystem
        // nor another template.
        environment
            .add_template_owned(TEMPLATE_NAME, source.to_owned())
            .with_context(|| "chat template is not a valid Jinja program")?;
        Ok(Self {
            environment,
            bos_token: bos_token.to_owned(),
            eos_token: eos_token.to_owned(),
        })
    }

    /// Render a conversation through the compiled template.
    ///
    /// `tools` is only visible to the template when it is `Some`, matching
    /// `apply_chat_template`'s omission of the variable. `enable_thinking`
    /// is passed the same way, so a template keeps its own default when the
    /// option is unset.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        options: &ChatTemplateOptions,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let mut context = serde_json::Map::new();
        context.insert(
            "messages".to_owned(),
            serde_json::Value::Array(messages.iter().map(message_value).collect::<Result<_>>()?),
        );
        if let Some(tools) = tools {
            context.insert("tools".to_owned(), serde_json::Value::Array(tools.to_vec()));
        }
        context.insert(
            "add_generation_prompt".to_owned(),
            serde_json::Value::Bool(add_generation_prompt),
        );
        context.insert(
            "bos_token".to_owned(),
            serde_json::Value::String(self.bos_token.clone()),
        );
        context.insert(
            "eos_token".to_owned(),
            serde_json::Value::String(self.eos_token.clone()),
        );
        if let Some(enable_thinking) = options.enable_thinking {
            context.insert(
                "enable_thinking".to_owned(),
                serde_json::Value::Bool(enable_thinking),
            );
        }
        let template = self
            .environment
            .get_template(TEMPLATE_NAME)
            .context("the compiled chat template is missing")?;
        let mut prompt = CappedBuffer::new(
            MAX_RENDERED_BYTES,
            format!(
                "the rendered chat template is larger than the {MAX_RENDERED_BYTES} byte limit"
            ),
        );
        let outcome = template.render_captured_to(serde_json::Value::Object(context), &mut prompt);
        if prompt.exceeded() {
            anyhow::bail!("{}", prompt.message());
        }
        outcome.context("chat template failed to render the conversation")?;
        prompt
            .into_text()
            .context("the rendered chat template is not valid UTF-8")
    }
}

/// A buffer that refuses to grow past a limit.
///
/// MiniJinja cannot limit the size of a render, so the prompt and a `tojson`
/// result go through this buffer: the first write past the cap ends the work
/// with an ordinary error instead of an unbounded allocation.
struct CappedBuffer {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
    message: String,
}

impl CappedBuffer {
    /// A buffer that refuses to grow past `limit` bytes.
    fn new(limit: usize, message: impl Into<String>) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            exceeded: false,
            message: message.into(),
        }
    }

    /// Whether a write went past the cap.
    fn exceeded(&self) -> bool {
        self.exceeded
    }

    /// The message that explains the cap.
    fn message(&self) -> &str {
        &self.message
    }

    /// Whether a write of `len` bytes must be refused; the first write past
    /// the cap sets the flag, so every later write is refused as well.
    fn refuses(&mut self, len: usize) -> bool {
        if self.bytes.len().saturating_add(len) > self.limit {
            self.exceeded = true;
        }
        self.exceeded
    }

    /// Append text to the buffer.
    fn push_str(&mut self, text: &str) -> Result<(), Error> {
        if self.refuses(text.len()) {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                self.message.clone(),
            ));
        }
        self.bytes.extend_from_slice(text.as_bytes());
        Ok(())
    }

    /// The buffer as text.
    fn into_text(self) -> Result<String, std::string::FromUtf8Error> {
        String::from_utf8(self.bytes)
    }
}

impl io::Write for CappedBuffer {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.refuses(buffer.len()) {
            return Err(io::Error::other(self.message.clone()));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// One message as a chat template observes it: `role` and `content` always,
/// the tool and reasoning payload only when it is present.
fn message_value(message: &ChatMessage) -> Result<serde_json::Value> {
    let mut value = serde_json::Map::new();
    value.insert(
        "role".to_owned(),
        serde_json::Value::String(message.role.as_str().to_owned()),
    );
    value.insert(
        "content".to_owned(),
        serde_json::Value::String(message.content.clone()),
    );
    if let Some(name) = &message.name {
        value.insert("name".to_owned(), serde_json::Value::String(name.clone()));
    }
    if let Some(reasoning) = &message.reasoning_content {
        value.insert(
            "reasoning_content".to_owned(),
            serde_json::Value::String(reasoning.clone()),
        );
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        value.insert(
            "tool_call_id".to_owned(),
            serde_json::Value::String(tool_call_id.clone()),
        );
    }
    if let Some(tool_calls) = &message.tool_calls {
        value.insert(
            "tool_calls".to_owned(),
            serde_json::Value::Array(
                tool_calls
                    .iter()
                    .map(tool_call_value)
                    .collect::<Result<_>>()?,
            ),
        );
    }
    Ok(serde_json::Value::Object(value))
}

fn tool_call_value(call: &ChatToolCall) -> Result<serde_json::Value> {
    let arguments = call.function.arguments.to_template_value();
    anyhow::ensure!(
        arguments.is_object(),
        "tool call arguments must encode a JSON object"
    );
    Ok(serde_json::json!({
        "id": call.id,
        "type": call.kind,
        "function": {
            "name": call.function.name,
            "arguments": arguments,
        },
    }))
}

/// `tojson` with `json.dumps` semantics: `ensure_ascii=False`, the default
/// `", "` / `": "` separators and the caller's insertion order.
fn tojson(value: JinjaValue, kwargs: Kwargs) -> Result<JinjaValue, Error> {
    let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    let separators = kwargs
        .get::<Option<JinjaValue>>("separators")?
        .map(|separators| -> Result<(String, String), Error> {
            let pair: Vec<String> = separators
                .try_iter()?
                .map(|item| {
                    item.as_str().map(str::to_owned).ok_or_else(|| {
                        Error::new(
                            ErrorKind::InvalidOperation,
                            "tojson separators must be strings",
                        )
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            match pair.as_slice() {
                [item, key] => Ok((item.clone(), key.clone())),
                _ => Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "tojson separators must be a pair of strings",
                )),
            }
        })
        .transpose()?;
    let indent = kwargs
        .get::<Option<JinjaValue>>("indent")?
        .map(json_indent)
        .transpose()?;
    let separators = separators.unwrap_or_else(|| {
        (
            if indent.is_some() { "," } else { ", " }.to_owned(),
            ": ".to_owned(),
        )
    });
    kwargs.assert_all_used()?;
    let style = JsonStyle {
        indent,
        item_separator: separators.0,
        key_separator: separators.1,
        sort_keys,
        ensure_ascii,
    };
    let mut out = CappedBuffer::new(
        MAX_JSON_BYTES,
        format!("the tojson result is larger than the {MAX_JSON_BYTES} byte limit"),
    );
    write_json_value(&value, &mut out, &style, 0)?;
    let text = out
        .into_text()
        .map_err(|error| Error::new(ErrorKind::BadSerialization, error.to_string()))?;
    Ok(JinjaValue::from_safe_string(text))
}

/// The `json.dumps` options a template may pass to `tojson`.
struct JsonStyle {
    indent: Option<String>,
    item_separator: String,
    key_separator: String,
    sort_keys: bool,
    ensure_ascii: bool,
}

/// Python accepts both a literal indentation string and an integer number of
/// spaces.
///
/// The indentation is repeated for every nesting level of the result, so it
/// is bounded: an integer outside `0..=16` and a literal longer than
/// [`MAX_JSON_INDENT`] bytes are refused before the indentation is built.
fn json_indent(value: JinjaValue) -> Result<String, Error> {
    let refused = |text: String| Error::new(ErrorKind::InvalidOperation, text);
    let out_of_range = || {
        refused(format!(
            "tojson indent must be between 0 and {MAX_JSON_INDENT} spaces"
        ))
    };
    if let Some(indent) = value.as_str() {
        if indent.len() > MAX_JSON_INDENT {
            return Err(refused(format!(
                "the tojson indent is longer than the {MAX_JSON_INDENT} byte limit"
            )));
        }
        return Ok(indent.to_owned());
    }
    // The value is read as it is: turning it into JSON first would
    // materialize a lazy sequence before its type is even known.
    let spaces = match value.kind() {
        ValueKind::Number if value.is_integer() => value.as_i64().ok_or_else(out_of_range)?,
        // In Python, bool is a subclass of int.
        ValueKind::Bool => i64::from(value.is_true()),
        _ => return Err(refused("tojson indent must be an integer or string".into())),
    };
    let spaces = usize::try_from(spaces).map_err(|_| out_of_range())?;
    if spaces > MAX_JSON_INDENT {
        return Err(out_of_range());
    }
    Ok(" ".repeat(spaces))
}

/// Serialize one value the way Python's `json` module does.
///
/// The value is written as it is traversed: nothing but the result itself is
/// held in memory, so a lazy sequence cannot make the renderer allocate.
fn write_json_value(
    value: &JinjaValue,
    out: &mut CappedBuffer,
    style: &JsonStyle,
    level: usize,
) -> Result<(), Error> {
    match value.kind() {
        ValueKind::Undefined | ValueKind::None | ValueKind::Invalid => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => write_json_number(value, out),
        ValueKind::String => {
            write_json_string(value.as_str().unwrap_or_default(), out, style.ensure_ascii)
        }
        // MiniJinja serializes a byte string as the list of its bytes.
        ValueKind::Bytes => write_json_array(
            value
                .as_bytes()
                .unwrap_or_default()
                .iter()
                .map(|byte| JinjaValue::from(*byte)),
            out,
            style,
            level,
        ),
        ValueKind::Seq | ValueKind::Iterable => {
            write_json_array(value.try_iter()?, out, style, level)
        }
        ValueKind::Map => write_json_object(value, out, style, level),
        // A plain object has no JSON shape. MiniJinja writes its string form,
        // and so does the renderer.
        _ => write_json_string(&value.to_string(), out, style.ensure_ascii),
    }
}

/// Write a number the way Python's `json` module does.
fn write_json_number(value: &JinjaValue, out: &mut CappedBuffer) -> Result<(), Error> {
    if value.is_integer() {
        let text = value.as_i64().map(|number| number.to_string()).or_else(|| {
            u64::try_from(value.clone())
                .ok()
                .map(|number| number.to_string())
        });
        if let Some(text) = text {
            return out.push_str(&text);
        }
    }
    let number = f64::try_from(value.clone()).unwrap_or(f64::NAN);
    if !number.is_finite() {
        // `NaN` and the infinities have no JSON form: they become `null`.
        return out.push_str("null");
    }
    // Python pads exponents to two digits and includes their sign.
    let formatted = format!("{number:?}");
    let Some((mantissa, exponent)) = formatted.split_once('e') else {
        return out.push_str(&formatted);
    };
    let exponent: i32 = exponent.parse().map_err(|error: std::num::ParseIntError| {
        Error::new(ErrorKind::BadSerialization, error.to_string())
    })?;
    out.push_str(mantissa)?;
    out.push_str(&format!("e{exponent:+03}"))
}

/// Write a sequence as a JSON array, taking the items as they are needed so
/// a lazy sequence is never held in memory.
fn write_json_array(
    items: impl Iterator<Item = JinjaValue>,
    out: &mut CappedBuffer,
    style: &JsonStyle,
    level: usize,
) -> Result<(), Error> {
    let mut item_written = false;
    out.push_str("[")?;
    for item in items {
        if item_written {
            out.push_str(&style.item_separator)?;
        }
        item_written = true;
        push_indent(out, style, level + 1)?;
        write_json_value(&item, out, style, level + 1)?;
    }
    if item_written {
        push_indent(out, style, level)?;
    }
    out.push_str("]")
}

/// Write a mapping as a JSON object, in insertion order unless the template
/// asks for sorted keys.
fn write_json_object(
    value: &JinjaValue,
    out: &mut CappedBuffer,
    style: &JsonStyle,
    level: usize,
) -> Result<(), Error> {
    let mut entries: Vec<(String, JinjaValue)> = Vec::new();
    for key in value.try_iter()? {
        // Python's `json` module writes dictionary keys as strings.
        let name = match key.kind() {
            ValueKind::String => key.as_str().unwrap_or_default().to_owned(),
            ValueKind::Number if key.is_integer() => key
                .as_i64()
                .map(|number| number.to_string())
                .or_else(|| {
                    u64::try_from(key.clone())
                        .ok()
                        .map(|number| number.to_string())
                })
                .unwrap_or_default(),
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "tojson object keys must be strings",
                ));
            }
        };
        entries.push((name, value.get_item(&key)?));
    }
    if entries.is_empty() {
        return out.push_str("{}");
    }
    if style.sort_keys {
        entries.sort_by(|left, right| left.0.cmp(&right.0));
    }
    out.push_str("{")?;
    for (index, (name, item)) in entries.iter().enumerate() {
        if index > 0 {
            out.push_str(&style.item_separator)?;
        }
        push_indent(out, style, level + 1)?;
        write_json_string(name, out, style.ensure_ascii)?;
        out.push_str(&style.key_separator)?;
        write_json_value(item, out, style, level + 1)?;
    }
    push_indent(out, style, level)?;
    out.push_str("}")
}

/// Write the indentation of one nesting level.
fn push_indent(out: &mut CappedBuffer, style: &JsonStyle, level: usize) -> Result<(), Error> {
    if let Some(indent) = &style.indent {
        out.push_str("\n")?;
        for _ in 0..level {
            out.push_str(indent)?;
        }
    }
    Ok(())
}

/// Python's JSON string escaping: the short escapes, `\u00xx` for the other
/// control characters, and `\uXXXX` for non-ASCII only with `ensure_ascii`.
fn write_json_string(text: &str, out: &mut CappedBuffer, ensure_ascii: bool) -> Result<(), Error> {
    out.push_str("\"")?;
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\"")?,
            '\\' => out.push_str("\\\\")?,
            '\u{8}' => out.push_str("\\b")?,
            '\u{c}' => out.push_str("\\f")?,
            '\n' => out.push_str("\\n")?,
            '\r' => out.push_str("\\r")?,
            '\t' => out.push_str("\\t")?,
            control if control < ' ' => {
                out.push_str(&format!("\\u{code:04x}", code = u32::from(control)))?;
            }
            other if ensure_ascii && other >= '\u{7f}' => {
                let mut buffer = [0; 2];
                for unit in other.encode_utf16(&mut buffer) {
                    out.push_str(&format!("\\u{unit:04x}"))?;
                }
            }
            other => {
                let mut buffer = [0; 4];
                out.push_str(other.encode_utf8(&mut buffer))?;
            }
        }
    }
    out.push_str("\"")
}

/// `raise_exception` turns a template's message into a render error.
fn raise_exception(message: JinjaValue) -> Result<JinjaValue, Error> {
    let message = message
        .as_str()
        .map_or_else(|| message.to_string(), ToOwned::to_owned);
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

/// `strftime_now` uses the host's local wall clock, like Python's
/// `datetime.now().strftime`. The resulting datetime is naive: zone directives
/// are empty even though its calendar fields come from local time.
fn strftime_now(format: &str) -> Result<String, Error> {
    #[cfg(all(
        target_pointer_width = "64",
        any(target_os = "linux", target_os = "macos")
    ))]
    {
        posix_clock::strftime_now(format)
    }
    #[cfg(not(all(
        target_pointer_width = "64",
        any(target_os = "linux", target_os = "macos")
    )))]
    {
        let _ = format;
        Err(Error::new(
            ErrorKind::InvalidOperation,
            "strftime_now requires a 64-bit Linux or macOS host",
        ))
    }
}

// The runtime supports Linux and macOS. Both use this POSIX `tm` layout and
// a signed-long `time_t` on their supported 64-bit architectures. Keep the FFI
// and its ABI assumptions behind this safe, platform-specific wrapper.
#[cfg(all(
    target_pointer_width = "64",
    any(target_os = "linux", target_os = "macos")
))]
mod posix_clock {
    use std::ffi::CString;
    use std::fmt::Write;
    use std::mem::MaybeUninit;
    use std::os::raw::{c_char, c_int, c_long};

    use minijinja::{Error, ErrorKind};

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CalendarTime {
        second: c_int,
        minute: c_int,
        hour: c_int,
        month_day: c_int,
        month: c_int,
        year: c_int,
        week_day: c_int,
        year_day: c_int,
        is_dst: c_int,
        utc_offset: c_long,
        zone: *const c_char,
    }

    unsafe extern "C" {
        fn localtime_r(timestamp: *const c_long, result: *mut CalendarTime) -> *mut CalendarTime;
        fn strftime(
            output: *mut c_char,
            capacity: usize,
            format: *const c_char,
            calendar: *const CalendarTime,
        ) -> usize;
    }

    pub(super) fn strftime_now(format: &str) -> Result<String, Error> {
        let elapsed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))?;
        let timestamp = c_long::try_from(elapsed.as_secs())
            .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))?;
        let mut calendar = MaybeUninit::<CalendarTime>::uninit();
        // SAFETY: both pointers are valid for their declared types, the result
        // has the Linux/macOS `tm` layout, and localtime_r initializes it on
        // success without exposing shared calendar storage to other threads.
        if unsafe { localtime_r(&raw const timestamp, calendar.as_mut_ptr()) }.is_null() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "the local time is outside the supported calendar range",
            ));
        }
        // SAFETY: localtime_r returned non-null and initialized every field.
        let calendar = unsafe { calendar.assume_init() };
        format_time(calendar, elapsed.subsec_micros(), format)
    }

    fn format_time(
        mut calendar: CalendarTime,
        microsecond: u32,
        format: &str,
    ) -> Result<String, Error> {
        // Python's datetime.now() has no tzinfo. Do not expose the timezone
        // attached by localtime_r to an otherwise naive datetime.
        calendar.is_dst = -1;
        calendar.utc_offset = 0;
        calendar.zone = std::ptr::null();
        let mut prepared = String::with_capacity(format.len());
        let mut characters = format.chars().peekable();
        while let Some(character) = characters.next() {
            if character != '%' {
                prepared.push(character);
                continue;
            }
            match characters.next() {
                Some('%') => prepared.push_str("%%"),
                Some('f') => write!(prepared, "{microsecond:06}").expect("write to String"),
                Some('z' | 'Z') => (),
                Some(':') if characters.next_if_eq(&'z').is_some() => (),
                Some(other) => {
                    prepared.push('%');
                    prepared.push(other);
                }
                None => prepared.push('%'),
            }
        }

        // Python permits embedded NULs in the format. Each C format ends at a
        // NUL, so render its pieces separately and retain every delimiter.
        let mut output = String::new();
        for (index, part) in prepared.split('\0').enumerate() {
            if index > 0 {
                output.push('\0');
            }
            output.push_str(&format_part(&calendar, part)?);
        }
        Ok(output)
    }

    fn format_part(calendar: &CalendarTime, format: &str) -> Result<String, Error> {
        // A literal prefix guarantees that even an empty formatted result has
        // a nonzero length, distinguishing it from an insufficient buffer.
        let format = CString::new(format!("@{format}"))
            .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))?;
        let mut capacity = 128;
        loop {
            let mut output = vec![0_u8; capacity];
            // SAFETY: the writable buffer has exactly capacity bytes, format
            // is NUL-terminated, and calendar is a fully initialized C `tm`.
            let written = unsafe {
                strftime(
                    output.as_mut_ptr().cast(),
                    output.len(),
                    format.as_ptr(),
                    calendar,
                )
            };
            if written > 0 {
                return String::from_utf8(output[1..written].to_vec())
                    .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()));
            }
            // Widths in a malformed format can request arbitrary allocations.
            if capacity >= 1024 * 1024 {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "strftime_now output exceeds one MiB or the format is invalid",
                ));
            }
            capacity *= 2;
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn calendar(year: i32, month: i32, day: i32, weekday: i32, ordinal: i32) -> CalendarTime {
            CalendarTime {
                second: 0,
                minute: 0,
                hour: 0,
                month_day: day,
                month: month - 1,
                year: year - 1900,
                week_day: weekday,
                year_day: ordinal - 1,
                is_dst: 0,
                utc_offset: 0,
                zone: std::ptr::null(),
            }
        }

        #[test]
        fn formats_python_standard_directives_and_iso_week_boundaries() -> Result<(), Error> {
            let format = "%Y-%m-%d %w %u %j %U %W %G %V";
            for (calendar, expected) in [
                (
                    calendar(1970, 1, 1, 4, 1),
                    "1970-01-01 4 4 001 00 00 1970 01",
                ),
                (
                    calendar(2021, 1, 1, 5, 1),
                    "2021-01-01 5 5 001 00 00 2020 53",
                ),
                (
                    calendar(2019, 12, 30, 1, 364),
                    "2019-12-30 1 1 364 52 52 2020 01",
                ),
                (
                    calendar(2023, 1, 1, 0, 1),
                    "2023-01-01 0 7 001 01 00 2022 52",
                ),
                (
                    calendar(2024, 1, 1, 1, 1),
                    "2024-01-01 1 1 001 00 01 2024 01",
                ),
                (
                    calendar(2000, 2, 29, 2, 60),
                    "2000-02-29 2 2 060 09 09 2000 09",
                ),
            ] {
                assert_eq!(format_time(calendar, 0, format)?, expected);
            }
            Ok(())
        }

        #[test]
        fn preserves_empty_nul_and_escaped_formats_and_naive_zones() -> Result<(), Error> {
            let calendar = calendar(1970, 1, 1, 4, 1);
            assert_eq!(format_time(calendar, 12_345, "")?, "");
            assert_eq!(
                format_time(calendar, 12_345, "%z|%Z|%:z|%f|%%f")?,
                "|||012345|%f"
            );
            assert_eq!(
                format_time(calendar, 12_345, "a\0%Y\0%f")?,
                "a\x001970\x00012345"
            );
            assert_eq!(
                format_time(calendar, 12_345, "%c|%x|%X|%F|%D|%R|%T|%r|%C|%f|%%|%n%t")?,
                "Thu Jan  1 00:00:00 1970|01/01/70|00:00:00|1970-01-01|01/01/70|00:00|00:00:00|12:00:00 AM|19|012345|%|\n\t"
            );
            assert_eq!(format_time(calendar, 0, &"x".repeat(500))?, "x".repeat(500));
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ChatToolArguments;

    fn tools() -> Vec<serde_json::Value> {
        vec![serde_json::json!({
            "second": 2,
            "first": ["a", "b"],
            "nested": {"key": "值", "html": "<b>&'</b>"},
            "float": 1.5,
            "flag": true,
            "none": null,
        })]
    }

    fn render_template(source: &str, tools: Option<&[serde_json::Value]>) -> Result<String> {
        JinjaChatTemplate::new(source, "<bos>", "<eos>")?.render(
            &[ChatMessage::user("hi")],
            tools,
            &ChatTemplateOptions::default(),
            true,
        )
    }

    #[test]
    fn tojson_uses_python_dumps_separators_unicode_and_key_order() -> Result<()> {
        assert_eq!(
            render_template("{{ tools[0] | tojson }}", Some(&tools()))?,
            "{\"second\": 2, \"first\": [\"a\", \"b\"], \"nested\": {\"key\": \"值\", \"html\": \"<b>&'</b>\"}, \"float\": 1.5, \"flag\": true, \"none\": null}"
        );
        Ok(())
    }

    #[test]
    fn tojson_honours_indent_separators_and_sort_keys() -> Result<()> {
        assert_eq!(
            render_template("{{ tools[0] | tojson(indent=4) }}", Some(&tools()))?,
            "{\n    \"second\": 2,\n    \"first\": [\n        \"a\",\n        \"b\"\n    ],\n    \"nested\": {\n        \"key\": \"值\",\n        \"html\": \"<b>&'</b>\"\n    },\n    \"float\": 1.5,\n    \"flag\": true,\n    \"none\": null\n}"
        );
        assert_eq!(
            render_template("{{ tools[0] | tojson(sort_keys=true) }}", Some(&tools()))?,
            "{\"first\": [\"a\", \"b\"], \"flag\": true, \"float\": 1.5, \"nested\": {\"html\": \"<b>&'</b>\", \"key\": \"值\"}, \"none\": null, \"second\": 2}"
        );
        assert_eq!(
            render_template(
                "{{ tools[0] | tojson(separators=[',', ':'], ensure_ascii=true) }}",
                Some(&tools())
            )?,
            "{\"second\":2,\"first\":[\"a\",\"b\"],\"nested\":{\"key\":\"\\u503c\",\"html\":\"<b>&'</b>\"},\"float\":1.5,\"flag\":true,\"none\":null}"
        );
        Ok(())
    }

    #[test]
    fn tojson_preserves_explicit_separators_when_indenting() -> Result<()> {
        assert_eq!(
            render_template(
                "{{ [1, 2] | tojson(indent=0, separators=[', ', ': ']) }}",
                None
            )?,
            "[\n1, \n2\n]"
        );
        assert_eq!(
            render_template("{{ {'z': 1, 'a': 2} | tojson(separators=['', '']) }}", None)?,
            "{\"z\"1\"a\"2}"
        );
        Ok(())
    }

    #[test]
    fn tojson_accepts_python_string_and_numeric_indentation() -> Result<()> {
        assert_eq!(
            render_template("{{ {'values': [1, 2]} | tojson(indent='\\t') }}", None)?,
            "{\n\t\"values\": [\n\t\t1,\n\t\t2\n\t]\n}"
        );
        assert_eq!(
            render_template("{{ [1, 2] | tojson(indent=0) }}", None)?,
            "[\n1,\n2\n]"
        );
        assert_eq!(
            render_template("{{ [1, 2] | tojson(indent=16) }}", None)?,
            "[\n                1,\n                2\n]"
        );
        assert_eq!(
            render_template("{{ [1, 2] | tojson(indent='') }}", None)?,
            "[\n1,\n2\n]"
        );
        Ok(())
    }

    #[test]
    fn tojson_refuses_an_indentation_outside_the_allowed_range() {
        for indent in ["-2", "17", "1000000", "1.5"] {
            let error =
                render_template(&format!("{{{{ [1, 2] | tojson(indent={indent}) }}}}"), None)
                    .expect_err("an indentation outside the range must fail");
            assert!(
                format!("{error:#}").contains("tojson indent"),
                "{indent}: {error:#}"
            );
        }
    }

    #[test]
    fn tojson_uses_python_float_exponents() -> Result<()> {
        let tools = vec![serde_json::json!([1e-7, 1e20, -0.0, 1.0, 1e-5])];
        assert_eq!(
            render_template("{{ tools[0] | tojson }}", Some(&tools))?,
            "[1e-07, 1e+20, -0.0, 1.0, 1e-05]"
        );
        Ok(())
    }

    #[test]
    fn tojson_escapes_control_characters_like_python() -> Result<()> {
        let template = JinjaChatTemplate::new("{{ messages[0].content | tojson }}", "", "")?;
        let rendered = template.render(
            &[ChatMessage::user("line\nbreak\ttab\u{1}bell")],
            None,
            &ChatTemplateOptions::default(),
            true,
        )?;
        assert_eq!(rendered, "\"line\\nbreak\\ttab\\u0001bell\"");
        Ok(())
    }

    #[test]
    fn raise_exception_fails_the_render() {
        let error = render_template(
            "{% if add_generation_prompt %}{{ raise_exception('unsupported role') }}{% endif %}",
            None,
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("unsupported role"),
            "{error:#}"
        );
    }

    #[test]
    fn a_syntax_error_fails_at_compile_time() {
        let error = JinjaChatTemplate::new("{% for message in messages %}", "", "").unwrap_err();
        assert!(error.to_string().contains("valid Jinja"), "{error}");
        assert!(
            JinjaChatTemplate::new("{{ 'unterminated", "", "")
                .unwrap_err()
                .to_string()
                .contains("valid Jinja")
        );
    }

    #[test]
    fn exposes_special_tokens_and_conditional_options() -> Result<()> {
        let source = "{{ bos_token }}{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}{{ 'gen' if add_generation_prompt }}{{ eos_token }}";
        let template = JinjaChatTemplate::new(source, "<s>", "</s>")?;
        let options = ChatTemplateOptions::default();
        assert_eq!(
            template.render(
                &[ChatMessage::system("be brief"), ChatMessage::user("hello")],
                None,
                &options,
                true
            )?,
            "<s>system:be briefuser:hellogen</s>"
        );
        assert_eq!(
            template.render(&[ChatMessage::user("hello")], None, &options, false)?,
            "<s>user:hello</s>"
        );
        Ok(())
    }

    #[test]
    fn enable_thinking_is_visible_only_when_it_is_set() -> Result<()> {
        let source = "{{ 'explicit' if enable_thinking is defined else 'default' }}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        let unset = ChatTemplateOptions::default();
        assert_eq!(
            template.render(&[ChatMessage::user("hi")], None, &unset, true)?,
            "default"
        );
        for (thinking, expected) in [(false, "explicit"), (true, "explicit")] {
            let options = ChatTemplateOptions {
                enable_thinking: Some(thinking),
            };
            assert_eq!(
                template.render(&[ChatMessage::user("hi")], None, &options, true)?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn tools_are_absent_without_a_declaration() -> Result<()> {
        let template =
            JinjaChatTemplate::new("{{ 'tools' if tools is defined else 'bare' }}", "", "")?;
        let options = ChatTemplateOptions::default();
        assert_eq!(
            template.render(&[ChatMessage::user("hi")], None, &options, true)?,
            "bare"
        );
        assert_eq!(
            template.render(&[ChatMessage::user("hi")], Some(&tools()), &options, true)?,
            "tools"
        );
        Ok(())
    }

    #[test]
    fn renders_tool_history_with_ordered_arguments() -> Result<()> {
        let source = "{% for message in messages %}{{ message.role }}\
{% for call in message.tool_calls or [] %}{{ call.id }}:{{ call.function.name }}\
{{ call.function.arguments | tojson }}{% endfor %}\
{% endfor %}{{ bos_token }}";
        let template = JinjaChatTemplate::new(source, "<bos>", "<eos>")?;
        let rendered = template.render(
            &[
                ChatMessage::user("list files"),
                ChatMessage::assistant("").with_tool_calls(vec![ChatToolCall::new(
                    "call_1",
                    "shell",
                    ChatToolArguments::from_json_text("{\"command\": \"ls\", \"all\": true}"),
                )]),
                ChatMessage::tool("a.txt", "call_1"),
            ],
            None,
            &ChatTemplateOptions::default(),
            true,
        )?;
        assert_eq!(
            rendered,
            "userassistantcall_1:shell{\"command\": \"ls\", \"all\": true}tool<bos>"
        );
        Ok(())
    }

    #[test]
    fn exposes_optional_message_fields_and_normalizes_wire_arguments() -> Result<()> {
        let message: ChatMessage = serde_json::from_str(
            r#"{
            "role": "assistant", "content": "answer", "name": "helper",
            "reasoning_content": "reason", "tool_call_id": "result_1",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "lookup", "arguments": "{\"z\": 1, \"a\": 2}"
            }}]
        }"#,
        )?;
        let template = JinjaChatTemplate::new(
            "{% set m = messages[0] %}{{ m.name }}|{{ m.reasoning_content }}|{{ m.tool_call_id }}|{{ m.tool_calls[0].function.arguments | tojson }}",
            "",
            "",
        )?;
        assert_eq!(
            template.render(&[message], None, &ChatTemplateOptions::default(), true)?,
            "helper|reason|result_1|{\"z\": 1, \"a\": 2}"
        );
        Ok(())
    }

    #[test]
    fn rejects_tool_arguments_that_are_not_objects() -> Result<()> {
        let template = JinjaChatTemplate::new("{{ messages | tojson }}", "", "")?;
        for arguments in ["broken json", "[]", "null", "42"] {
            let message = ChatMessage::assistant("").with_tool_calls(vec![ChatToolCall::new(
                "call_1",
                "lookup",
                ChatToolArguments::from_json_text(arguments),
            )]);
            assert!(
                template
                    .render(&[message], None, &ChatTemplateOptions::default(), true)
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn trims_blocks_and_supports_loop_controls_and_python_container_methods() -> Result<()> {
        let source = "start\n  {% for value in [0, 1, 2, 3] %}\n  {% if value == 0 %}{% continue %}{% endif %}\n  {% if value == 2 %}{% break %}{% endif %}\n{{ value }}\n  {% endfor %}\n{% for key, value in {'z': 1, 'a': 2}.items() %}{{ key }}{{ value }}{% endfor %}|{{ {}.get('fallback', 'yes') }}|{{ [3, 2, 3].count(3) }}";
        assert_eq!(render_template(source, None)?, "start\n1\nz1a2|yes|2");
        Ok(())
    }

    #[test]
    fn python_string_methods_are_available() -> Result<()> {
        let template = JinjaChatTemplate::new(
            "{% set text = messages[0].content %}{{ text.strip().split(',')[0] }}|\
{{ text.strip().startswith('hi') }}|{{ text.endswith('there') }}|{{ text.strip().upper() }}",
            "",
            "",
        )?;
        let rendered = template.render(
            &[ChatMessage::user(" hi, there ")],
            None,
            &ChatTemplateOptions::default(),
            true,
        )?;
        assert_eq!(rendered, "hi|True|False|HI, THERE");
        Ok(())
    }

    #[test]
    fn formats_the_current_time_with_strftime() -> Result<()> {
        let rendered = render_template("{{ strftime_now('%Y-%m-%dT%H:') }}", None)?;
        let year: i64 = rendered[..4].parse().context("a four digit year")?;
        assert!(year >= 2025, "{rendered}");
        assert_eq!(rendered.as_bytes()[4], b'-');

        let microseconds = render_template("{{ strftime_now('%f') }}", None)?;
        assert_eq!(microseconds.len(), 6);
        assert!(microseconds.bytes().all(|byte| byte.is_ascii_digit()));
        assert_eq!(
            render_template("{{ strftime_now('%z|%Z|%:z') }}", None)?,
            "||"
        );
        Ok(())
    }

    #[test]
    fn a_template_that_never_ends_runs_out_of_fuel() -> Result<()> {
        let template = JinjaChatTemplate::new(
            "{% for index in range(100000) %}{% for inner in range(100000) %}{{ inner }}{% endfor %}{% endfor %}",
            "",
            "",
        )?;
        let started = std::time::Instant::now();
        let error = template
            .render(
                &[ChatMessage::user("hi")],
                None,
                &ChatTemplateOptions::default(),
                true,
            )
            .expect_err("the fuel budget must stop the loop");
        assert!(format!("{error:#}").contains("fuel"), "{error:#}");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "the render took {:?}",
            started.elapsed()
        );
        Ok(())
    }

    #[test]
    fn a_prompt_larger_than_the_limit_is_refused() -> Result<()> {
        let template = JinjaChatTemplate::new("{{ 'x' * 9000000 }}", "", "")?;
        let error = template
            .render(
                &[ChatMessage::user("hi")],
                None,
                &ChatTemplateOptions::default(),
                true,
            )
            .expect_err("the prompt limit must stop the render");
        assert!(format!("{error:#}").contains("larger than"), "{error:#}");
        Ok(())
    }

    #[test]
    fn a_tojson_result_larger_than_the_limit_is_refused() -> Result<()> {
        let template = JinjaChatTemplate::new("{{ ('x' * 9000000) | tojson }}", "", "")?;
        let error = template
            .render(
                &[ChatMessage::user("hi")],
                None,
                &ChatTemplateOptions::default(),
                true,
            )
            .expect_err("the tojson limit must stop the render");
        assert!(format!("{error:#}").contains("tojson result"), "{error:#}");
        Ok(())
    }

    #[test]
    fn tojson_writes_the_python_shapes_of_scalars_containers_and_lazy_values() -> Result<()> {
        assert_eq!(render_template("{{ [] | tojson }}", None)?, "[]");
        assert_eq!(render_template("{{ {} | tojson }}", None)?, "{}");
        assert_eq!(
            render_template("{{ [[], {}] | tojson }}", None)?,
            "[[], {}]"
        );
        assert_eq!(render_template("{{ 'text' | tojson }}", None)?, "\"text\"");
        assert_eq!(render_template("{{ none | tojson }}", None)?, "null");
        assert_eq!(render_template("{{ true | tojson }}", None)?, "true");
        assert_eq!(render_template("{{ 42 | tojson }}", None)?, "42");
        assert_eq!(render_template("{{ 1.5 | tojson }}", None)?, "1.5");
        assert_eq!(render_template("{{ (1, 2) | tojson }}", None)?, "[1, 2]");
        // The items of a lazy sequence are written without materializing it.
        assert_eq!(
            render_template("{{ ([1, 2] * 3) | tojson }}", None)?,
            "[1, 2, 1, 2, 1, 2]"
        );
        Ok(())
    }

    #[test]
    fn a_lazy_tojson_input_fails_instead_of_exhausting_memory() -> Result<()> {
        let template = JinjaChatTemplate::new("{{ ([0] * 1000000000) | tojson }}", "", "")?;
        let started = std::time::Instant::now();
        let error = template
            .render(
                &[ChatMessage::user("hi")],
                None,
                &ChatTemplateOptions::default(),
                true,
            )
            .expect_err("the tojson limit must stop the render");
        assert!(format!("{error:#}").contains("tojson result"), "{error:#}");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "the render took {:?}",
            started.elapsed()
        );
        Ok(())
    }

    #[test]
    fn a_recursive_macro_fails_within_the_recursion_limit() -> Result<()> {
        let template = JinjaChatTemplate::new(
            "{% macro recurse(depth) %}{{ recurse(depth + 1) }}{% endmacro %}{{ recurse(0) }}",
            "",
            "",
        )?;
        let error = template
            .render(
                &[ChatMessage::user("hi")],
                None,
                &ChatTemplateOptions::default(),
                true,
            )
            .expect_err("the recursion limit must stop the macro");
        assert!(
            format!("{error:#}").contains("recursion limit"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn a_template_larger_than_the_limit_is_refused() {
        let source = "x".repeat(MAX_TEMPLATE_BYTES + 1);
        let error = JinjaChatTemplate::new(&source, "", "")
            .expect_err("an oversize template must be refused");
        assert!(format!("{error:#}").contains("larger than"), "{error:#}");
    }
}
