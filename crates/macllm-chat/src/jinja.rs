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

/// What the template's own renders show about the `enable_thinking` option.
///
/// The probe of [`JinjaChatTemplate::new`] fills it in: the template honors
/// the option when its source names it and its two renders differ, and its
/// default is the value its render without the option equals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EnableThinkingProbe {
    /// The template honors the option.
    honors: bool,
    /// The default the template applies when the option is left out.
    default: Option<bool>,
}

impl EnableThinkingProbe {
    /// A template that says nothing about the option.
    const fn unknown() -> Self {
        Self {
            honors: false,
            default: None,
        }
    }
}

/// A compiled Hugging Face-style chat template.
///
/// The template is parsed once by [`JinjaChatTemplate::new`], which fails on
/// a syntax error, and rendered any number of times afterwards.
#[derive(Debug, Clone)]
pub struct JinjaChatTemplate {
    environment: Environment<'static>,
    bos_token: String,
    eos_token: String,
    honors_enable_thinking: bool,
    default_enable_thinking: Option<bool>,
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
        // The transformers-only `{% generation %}` markers have no MiniJinja
        // equivalent, so they become ordinary blocks before the parser sees
        // them.
        let program = rewrite_generation_tags(source);
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
            .add_template_owned(TEMPLATE_NAME, program)
            .with_context(|| "chat template is not a valid Jinja program")?;
        let mut template = Self {
            environment,
            bos_token: bos_token.to_owned(),
            eos_token: eos_token.to_owned(),
            honors_enable_thinking: false,
            default_enable_thinking: None,
        };
        // A template that honors the option is the only one that may take it.
        // The answer comes from the template's own source and its own
        // renders, not from a guess about the program: a template that reads
        // the option into a variable of its own still honors it. The same
        // probe shows the default the template applies when the option is
        // left out.
        let probe = template.probe_enable_thinking(source);
        template.honors_enable_thinking = probe.honors;
        template.default_enable_thinking = probe.default;
        Ok(template)
    }

    /// Whether the template honors the `enable_thinking` option.
    ///
    /// The answer comes from the template's own source and its own renders:
    /// the source has to name the option, and the same conversation is
    /// rendered with the option true and with the option false. The template
    /// honors the option when the source names it and both renders succeed
    /// and differ. A request that sets the option is refused when the
    /// template does not honor it, instead of being silently ignored.
    ///
    /// Two limits: the probe renders a single user turn, so a template that
    /// reads the option only on other turns, such as tool turns, is treated
    /// as not honoring it; and a template that names the option without
    /// using it and prints a clock that varies between renders reads as
    /// honoring it.
    #[must_use]
    pub fn honors_enable_thinking(&self) -> bool {
        self.honors_enable_thinking
    }

    /// The default the template applies when the option is left out.
    ///
    /// The answer comes from the template's own renders: the same
    /// conversation is rendered with the option omitted, with the option
    /// true and with the option false. The default is `Some(true)` when the
    /// render without the option equals the render with the option true,
    /// `Some(false)` when it equals the render with the option false, and
    /// `None` when the template does not honor the option, when the render
    /// without the option fails, or when it equals neither of the two.
    ///
    /// Two limits: the probe renders a single user turn, so a template that
    /// reads the option only on other turns, such as tool turns, has no
    /// default here; and a template that prints a clock that varies between
    /// renders usually has no default, but the omitted render can match one
    /// of the two renders when the clock does not change between them.
    #[must_use]
    pub fn default_enable_thinking(&self) -> Option<bool> {
        self.default_enable_thinking
    }

    /// What the template's own source and renders show about the option.
    ///
    /// The source has to name the option, and the probe renders one user
    /// turn with the generation prompt, once with the option true and once
    /// with the option false. The template honors the option when the source
    /// names it and both renders succeed and differ. A render that fails
    /// says nothing about the template, so a template whose probe render
    /// fails does not honor the option.
    ///
    /// The default of the template is the render it makes when the option is
    /// omitted: the default is the value that render equals, and a render
    /// that fails or matches neither of the two renders leaves the default
    /// unknown.
    fn probe_enable_thinking(&self, source: &str) -> EnableThinkingProbe {
        // A template that never names the option cannot honor it, whatever
        // its renders print.
        if !source.contains("enable_thinking") {
            return EnableThinkingProbe::unknown();
        }
        let messages = [ChatMessage::user("hello")];
        let render = |thinking: Option<bool>| {
            // The probe renders without the option check: a render that
            // refuses the option would say nothing about the template.
            self.render_unchecked(
                &messages,
                None,
                ChatTemplateOptions {
                    enable_thinking: thinking,
                },
                true,
                false,
            )
        };
        let (Ok(enabled), Ok(disabled)) = (render(Some(true)), render(Some(false))) else {
            return EnableThinkingProbe::unknown();
        };
        if enabled == disabled {
            return EnableThinkingProbe::unknown();
        }
        // The render without the option is the template's own default.
        let Ok(omitted) = render(None) else {
            return EnableThinkingProbe {
                honors: true,
                default: None,
            };
        };
        let default = if omitted == enabled {
            Some(true)
        } else if omitted == disabled {
            Some(false)
        } else {
            None
        };
        EnableThinkingProbe {
            honors: true,
            default,
        }
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
        self.render_inner(messages, tools, *options, add_generation_prompt, false)
    }

    /// Render an agent prompt, representing disabled tools as Jinja `none`.
    pub fn render_agent(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        options: &ChatTemplateOptions,
        add_generation_prompt: bool,
    ) -> Result<String> {
        self.render_inner(messages, tools, *options, add_generation_prompt, true)
    }

    fn render_inner(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        options: ChatTemplateOptions,
        add_generation_prompt: bool,
        none_tools: bool,
    ) -> Result<String> {
        if options.enable_thinking.is_some() && !self.honors_enable_thinking {
            anyhow::bail!(
                "the chat template does not use `enable_thinking`; the option is not supported"
            );
        }
        self.render_unchecked(messages, tools, options, add_generation_prompt, none_tools)
    }

    /// Render a conversation without the option check.
    ///
    /// The probe of [`Self::honors_enable_thinking`] renders through this
    /// path: a render that went through the check would refuse the option
    /// before the template had a chance to show that it honors it.
    fn render_unchecked(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        options: ChatTemplateOptions,
        add_generation_prompt: bool,
        none_tools: bool,
    ) -> Result<String> {
        let mut context = serde_json::Map::new();
        context.insert(
            "messages".to_owned(),
            serde_json::Value::Array(messages.iter().map(message_value).collect::<Result<_>>()?),
        );
        if let Some(tools) = tools {
            context.insert("tools".to_owned(), serde_json::Value::Array(tools.to_vec()));
        } else if none_tools {
            context.insert("tools".to_owned(), serde_json::Value::Null);
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

/// Rewrite the transformers-only `{% generation %}` markers.
///
/// A transformers template marks the assistant's generated text with a
/// `{% generation %}` block that MiniJinja cannot compile. The block becomes
/// an `if true` block, so its content and its whitespace control survive the
/// rewrite unchanged.
///
/// The rewrite touches neither a comment, nor an expression, nor a raw block:
/// their text is literal. It renames only a tag whose first word is exactly
/// `generation` or `endgeneration`, so a tag that merely starts with
/// `generation` is a different tag.
fn rewrite_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut index = 0;
    while index < source.len() {
        let tail = &source[index..];
        let Some(offset) = tail.find('{') else {
            out.push_str(tail);
            break;
        };
        let start = index + offset;
        out.push_str(&source[index..start]);
        let tail = &source[start..];
        if tail.starts_with("{#") {
            // A comment is literal text.
            let end = comment_end(tail).unwrap_or(tail.len());
            out.push_str(&tail[..end]);
            index = start + end;
            continue;
        }
        if tail.starts_with("{{") {
            // An expression is literal text: a string may hold any tag.
            let end = expression_end(tail).unwrap_or(tail.len());
            out.push_str(&tail[..end]);
            index = start + end;
            continue;
        }
        if tail.starts_with("{%") {
            let Some(end) = tag_end(tail) else {
                out.push_str(tail);
                break;
            };
            let tag = &tail[..end];
            index = start + end;
            match tag_word(tag) {
                "raw" => {
                    // A raw block holds literal text until its end tag.
                    out.push_str(tag);
                    let block = &source[index..];
                    if let Some(end) = raw_block_end(block) {
                        out.push_str(&block[..end]);
                        index += end;
                    } else {
                        out.push_str(block);
                        index = source.len();
                    }
                }
                "generation" => out.push_str(&rewrite_tag(tag, "if true")),
                "endgeneration" => out.push_str(&rewrite_tag(tag, "endif")),
                _ => out.push_str(tag),
            }
            continue;
        }
        // A lone brace is literal text.
        out.push('{');
        index = start + 1;
    }
    out
}

/// The end of the comment that starts `source`, including its `#}`.
fn comment_end(source: &str) -> Option<usize> {
    source[2..].find("#}").map(|end| end + 4)
}

/// The end of the tag that starts `source`, including its closing `%}`.
///
/// A `%}` inside a string literal does not end the tag.
fn tag_end(source: &str) -> Option<usize> {
    construct_end(source, b"%}")
}

/// The end of the expression that starts `source`, including its `}}`.
fn expression_end(source: &str) -> Option<usize> {
    construct_end(source, b"}}")
}

/// The end of the construct that starts `source`, including its closing
/// delimiter.
///
/// A delimiter inside a string literal does not end the construct.
fn construct_end(source: &str, close: &[u8]) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut index = 2;
    let mut quote: Option<u8> = None;
    let mut escaped = false;
    while index + 1 < bytes.len() {
        let byte = bytes[index];
        if let Some(delimiter) = quote {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == delimiter {
                quote = None;
            }
        } else if byte == b'\'' || byte == b'"' {
            quote = Some(byte);
        } else if bytes[index..].starts_with(close) {
            return Some(index + close.len());
        }
        index += 1;
    }
    None
}

/// The end of a raw block: the offset just after its `{% endraw %}` tag.
///
/// The text of a raw block is literal, so only its end tag ends it.
fn raw_block_end(source: &str) -> Option<usize> {
    let mut index = 0;
    while let Some(offset) = source[index..].find("{%") {
        let start = index + offset;
        let end = tag_end(&source[start..])?;
        if tag_word(&source[start..start + end]) == "endraw" {
            return Some(start + end);
        }
        index = start + 2;
    }
    None
}

/// The first word of a tag, without its whitespace control characters.
fn tag_word(tag: &str) -> &str {
    let body = tag
        .strip_prefix("{%")
        .and_then(|body| body.strip_suffix("%}"))
        .unwrap_or(tag);
    let body = body.trim_start_matches(['-', '+']).trim_start();
    // A tag may end with its whitespace-control marker, so `{% generation-%}`
    // names the `generation` tag and not a word called `generation-`.
    let body = match body.strip_suffix('-') {
        Some(stripped) => stripped,
        None => body.strip_suffix('+').unwrap_or(body),
    };
    body.split_whitespace().next().unwrap_or("")
}

/// One tag with a new body, keeping its whitespace control characters.
///
/// The trimming of a tag depends on the `-` and `+` markers around it, so a
/// rewritten tag keeps them: `{%- generation -%}` becomes `{%- if true -%}`.
fn rewrite_tag(tag: &str, replacement: &str) -> String {
    let body = tag
        .strip_prefix("{%")
        .and_then(|body| body.strip_suffix("%}"))
        .unwrap_or(tag);
    let mut leading = "";
    let mut trailing = "";
    let mut rest = body;
    if let Some(first) = rest.chars().next()
        && (first == '-' || first == '+')
    {
        leading = &rest[..first.len_utf8()];
        rest = &rest[first.len_utf8()..];
    }
    let rest = rest.trim_end();
    if let Some(last) = rest.chars().last()
        && (last == '-' || last == '+')
    {
        trailing = &rest[rest.len() - last.len_utf8()..];
    }
    format!("{{%{leading} {replacement} {trailing}%}}")
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
///
/// The Windows host has no POSIX `strftime`, so it renders the same contract
/// with `chrono` for an allowlist of directives and refuses the rest.
fn strftime_now(format: &str) -> Result<String, Error> {
    #[cfg(windows)]
    {
        windows_clock::strftime_now(format)
    }
    #[cfg(all(
        not(windows),
        target_pointer_width = "64",
        any(target_os = "linux", target_os = "macos")
    ))]
    {
        posix_clock::strftime_now(format)
    }
    #[cfg(not(any(
        windows,
        all(
            target_pointer_width = "64",
            any(target_os = "linux", target_os = "macos")
        )
    )))]
    {
        let _ = format;
        Err(Error::new(
            ErrorKind::InvalidOperation,
            "strftime_now requires a 64-bit Linux or macOS host",
        ))
    }
}

// This path covers Linux and macOS; Windows uses `windows_clock`. Both use
// this POSIX `tm` layout and a signed-long `time_t` on their supported 64-bit
// architectures. Keep the FFI and its ABI assumptions behind this safe,
// platform-specific wrapper.
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
        fn renders_the_shared_week_table() -> Result<(), Error> {
            for &(year, month, day, weekday, ordinal, expected) in
                super::super::tests::STRFTIME_WEEK_CASES
            {
                let calendar = calendar(year, month, day, weekday, ordinal);
                assert_eq!(
                    format_time(calendar, 0, super::super::tests::STRFTIME_WEEK_FORMAT)?,
                    expected,
                    "{year}-{month}-{day}"
                );
            }
            Ok(())
        }

        #[test]
        fn renders_the_shared_strftime_table() -> Result<(), Error> {
            let calendar = calendar(1970, 1, 1, 4, 1);
            for (format, expected) in super::super::tests::STRFTIME_CASES {
                assert_eq!(
                    format_time(calendar, 12_345, format)?,
                    *expected,
                    "{format}"
                );
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

// Windows has no POSIX `strftime`, so the same contract is rendered by
// `chrono`, which implements the directives on every supported host. The
// module is compiled on every host under the test build, so the Mac test run
// exercises the Windows path as well.
#[cfg(any(windows, test))]
mod windows_clock {
    use std::fmt::Write;
    use std::iter::Peekable;
    use std::str::Chars;

    use chrono::NaiveDateTime;
    use minijinja::{Error, ErrorKind};

    /// The directives that `chrono` renders for the Windows path.
    ///
    /// The list is an allowlist: every other directive is refused, because a
    /// prompt that silently renders differently from the POSIX host is worse
    /// than a prompt that fails.
    ///
    /// The letters are stored without separators, so the list holds the
    /// directives alone. A separator would let a directive that is not a
    /// letter, such as `% `, reach `chrono`, which renders it as an error
    /// instead of a directive of its own.
    const CHRONO_DIRECTIVES: &str = "aAbBcCdDeFgGhHIjmMnprRStTuUVwWxXyY";

    /// The local wall clock of the host, rendered like the POSIX host does.
    #[cfg(windows)]
    pub(super) fn strftime_now(format: &str) -> Result<String, Error> {
        format_time(chrono::Local::now().naive_local(), format)
    }

    /// Renders one format for one naive local datetime.
    ///
    /// The result follows the POSIX contract: the microseconds have six
    /// digits, the zone directives are empty, and an embedded NUL is kept.
    pub(super) fn format_time(time: NaiveDateTime, format: &str) -> Result<String, Error> {
        let mut output = String::with_capacity(format.len());
        // Python permits embedded NULs in the format, so every part between
        // two NULs is rendered on its own and the delimiters are retained.
        for (index, part) in format.split('\0').enumerate() {
            if index > 0 {
                output.push('\0');
            }
            output.push_str(&format_part(time, part)?);
        }
        Ok(output)
    }

    fn format_part(time: NaiveDateTime, format: &str) -> Result<String, Error> {
        let mut output = String::with_capacity(format.len());
        let mut characters = format.chars().peekable();
        while let Some(character) = characters.next() {
            if character != '%' {
                output.push(character);
                continue;
            }
            let directive = collect_directive(&mut characters);
            match directive.as_str() {
                "%%" => output.push('%'),
                "%f" => write!(output, "{:06}", time.and_utc().timestamp_subsec_micros())
                    .expect("write to String"),
                "%z" | "%Z" | "%:z" => (),
                _ => {
                    if !is_chrono_directive(&directive) {
                        return Err(Error::new(
                            ErrorKind::InvalidOperation,
                            format!("strftime_now cannot render the directive {directive}"),
                        ));
                    }
                    // `chrono` reports a format that it cannot render as an
                    // error, which is refused like any other unknown
                    // directive instead of panicking.
                    write!(output, "{}", time.format(&directive)).map_err(|error| {
                        Error::new(
                            ErrorKind::InvalidOperation,
                            format!(
                                "strftime_now cannot render the directive {directive}: {error}"
                            ),
                        )
                    })?;
                }
            }
        }
        Ok(output)
    }

    /// Collects the directive that follows a `%`, including its flags, its
    /// width and its modifier, so that a refused directive can be named.
    fn collect_directive(characters: &mut Peekable<Chars<'_>>) -> String {
        let mut directive = String::from("%");
        while let Some(&next) = characters.peek() {
            if next.is_ascii_digit() || matches!(next, '-' | '_' | '0' | '^' | '#') {
                directive.push(next);
                characters.next();
            } else {
                break;
            }
        }
        if let Some(':' | 'E' | 'O') = characters.peek() {
            directive.push(characters.next().expect("peeked"));
        }
        if let Some(next) = characters.next() {
            directive.push(next);
        }
        directive
    }

    /// Reports whether one directive is a single letter of the allowlist.
    fn is_chrono_directive(directive: &str) -> bool {
        let mut letters = directive.strip_prefix('%').unwrap_or_default().chars();
        match (letters.next(), letters.next()) {
            (Some(letter), None) => CHRONO_DIRECTIVES.contains(letter),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ChatToolArguments;
    use chrono::Datelike;

    /// The formats and the outputs that every clock path must render.
    ///
    /// The instant is 1970-01-01 00:00:00.012345, which the POSIX path
    /// renders with the fixed calendar of its own tests and the Windows path
    /// renders with `chrono`.
    pub(super) const STRFTIME_CASES: &[(&str, &str)] = &[
        ("%a", "Thu"),
        ("%A", "Thursday"),
        ("%b", "Jan"),
        ("%B", "January"),
        ("%c", "Thu Jan  1 00:00:00 1970"),
        ("%C", "19"),
        ("%d", "01"),
        ("%D", "01/01/70"),
        ("%e", " 1"),
        ("%F", "1970-01-01"),
        ("%g", "70"),
        ("%G", "1970"),
        ("%h", "Jan"),
        ("%H", "00"),
        ("%I", "12"),
        ("%j", "001"),
        ("%m", "01"),
        ("%M", "00"),
        ("%n", "\n"),
        ("%p", "AM"),
        ("%r", "12:00:00 AM"),
        ("%R", "00:00"),
        ("%S", "00"),
        ("%t", "\t"),
        ("%T", "00:00:00"),
        ("%u", "4"),
        ("%U", "00"),
        ("%V", "01"),
        ("%w", "4"),
        ("%W", "00"),
        ("%x", "01/01/70"),
        ("%X", "00:00:00"),
        ("%y", "70"),
        ("%Y", "1970"),
        ("%B %d, %Y", "January 01, 1970"),
        ("%d %b %Y", "01 Jan 1970"),
        ("%d %B %Y", "01 January 1970"),
        ("%f", "012345"),
        ("%z|%Z|%:z", "||"),
        ("%%", "%"),
        ("%%f", "%f"),
        ("a\0%Y\0%f", "a\x001970\x00012345"),
    ];

    /// The format of the shared week table.
    ///
    /// The format holds the calendar fields of one date, the day of the year
    /// and the week numbers that the two clock paths must agree on.
    pub(super) const STRFTIME_WEEK_FORMAT: &str = "%Y-%m-%d %w %u %j %U %W %G %g %V";

    /// The week boundaries that every clock path must render.
    ///
    /// Each row holds the year, the month, the day, the weekday, the ordinal
    /// day of the year and the output that the format above must render for
    /// midnight on that date.
    pub(super) const STRFTIME_WEEK_CASES: &[(i32, i32, i32, i32, i32, &str)] = &[
        (1970, 1, 1, 4, 1, "1970-01-01 4 4 001 00 00 1970 70 01"),
        (2021, 1, 1, 5, 1, "2021-01-01 5 5 001 00 00 2020 20 53"),
        (2019, 12, 30, 1, 364, "2019-12-30 1 1 364 52 52 2020 20 01"),
        (2023, 1, 1, 0, 1, "2023-01-01 0 7 001 01 00 2022 22 52"),
        (2024, 1, 1, 1, 1, "2024-01-01 1 1 001 00 01 2024 24 01"),
        (2000, 2, 29, 2, 60, "2000-02-29 2 2 060 09 09 2000 00 09"),
    ];

    /// The instant that the shared table renders.
    fn epoch_microsecond() -> chrono::NaiveDateTime {
        chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
            .expect("the epoch day exists")
            .and_hms_micro_opt(0, 0, 0, 12_345)
            .expect("the epoch instant exists")
    }

    #[test]
    fn windows_clock_renders_the_shared_strftime_table() -> Result<(), Error> {
        let time = epoch_microsecond();
        for (format, expected) in STRFTIME_CASES {
            assert_eq!(
                windows_clock::format_time(time, format)?,
                *expected,
                "{format}"
            );
        }
        Ok(())
    }

    #[test]
    fn windows_clock_renders_the_shared_week_table() -> Result<(), Error> {
        for &(year, month, day, weekday, ordinal, expected) in STRFTIME_WEEK_CASES {
            let month =
                u32::try_from(month).expect("the month of the shared week table is not negative");
            let day = u32::try_from(day).expect("the day of the shared week table is not negative");
            let date = chrono::NaiveDate::from_ymd_opt(year, month, day).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidOperation,
                    "the date of the shared week table does not exist",
                )
            })?;
            let weekday = u32::try_from(weekday)
                .expect("the weekday of the shared week table is not negative");
            let ordinal = u32::try_from(ordinal)
                .expect("the ordinal of the shared week table is not negative");
            assert_eq!(
                date.weekday().num_days_from_sunday(),
                weekday,
                "{year}-{month}-{day}"
            );
            assert_eq!(date.ordinal(), ordinal, "{year}-{month}-{day}");
            let time = date.and_hms_opt(0, 0, 0).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidOperation,
                    "the instant of the shared week table does not exist",
                )
            })?;
            assert_eq!(
                windows_clock::format_time(time, STRFTIME_WEEK_FORMAT)?,
                expected,
                "{year}-{month}-{day}"
            );
        }
        Ok(())
    }

    #[test]
    fn windows_clock_refuses_the_directives_it_cannot_render() {
        let time = epoch_microsecond();
        for format in [
            "%Q", "%", "abc%", "%-d", "%5Y", "%:", "%:a", "%E", "%O", "% ",
        ] {
            let error = windows_clock::format_time(time, format)
                .expect_err("the directive must be refused");
            assert_eq!(error.kind(), ErrorKind::InvalidOperation, "{format}");
            assert!(
                error
                    .detail()
                    .is_some_and(|detail| detail.contains("cannot render the directive")),
                "{format}"
            );
        }
    }

    #[test]
    fn windows_clock_refuses_a_directive_that_is_not_a_letter() {
        // The allowlist of the Windows clock holds the directive letters alone,
        // so a directive that is not a letter, such as `% `, is refused by the
        // allowlist and is named in the error.
        let time = epoch_microsecond();
        let error = windows_clock::format_time(time, "% ")
            .expect_err("a directive that is not a letter must be refused");
        assert_eq!(error.kind(), ErrorKind::InvalidOperation);
        assert_eq!(
            error.detail(),
            Some("strftime_now cannot render the directive % "),
            "{error:?}"
        );
    }

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
        // The template tells whether the option is there and what it says,
        // so the variable is absent without the option and the two renders
        // with it differ.
        let source = "{{ 'explicit' if enable_thinking is defined else 'default' }}{{ 'on' if enable_thinking is defined and enable_thinking else 'off' }}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        let unset = ChatTemplateOptions::default();
        assert_eq!(
            template.render(&[ChatMessage::user("hi")], None, &unset, true)?,
            "defaultoff"
        );
        for (thinking, expected) in [(false, "explicitoff"), (true, "expliciton")] {
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
    fn a_template_that_only_asks_whether_the_option_is_defined_does_not_honor_it() -> Result<()> {
        // The template asks whether the option is there, not what it says:
        // both renders are the same, so the option changes nothing.
        let source = "{{ 'set' if enable_thinking is defined else 'unset' }}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(!template.honors_enable_thinking(), "{source}");
        let options = ChatTemplateOptions {
            enable_thinking: Some(false),
        };
        let error = template
            .render(&[ChatMessage::user("hi")], None, &options, true)
            .expect_err("the option must be refused");
        assert!(
            format!("{error:#}").contains("enable_thinking"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn a_template_whose_probe_render_fails_does_not_honor_the_option() -> Result<()> {
        // The template raises on one of the two renders: the failed render
        // says nothing about the option, so the template does not honor it.
        for source in [
            "{% if enable_thinking %}{{ raise_exception('thinking is not allowed') }}{% else %}plain{% endif %}",
            "{% if enable_thinking %}plain{% else %}{{ raise_exception('thinking is required') }}{% endif %}",
        ] {
            let template = JinjaChatTemplate::new(source, "", "")?;
            assert!(!template.honors_enable_thinking(), "{source}");
            let options = ChatTemplateOptions {
                enable_thinking: Some(false),
            };
            let error = template
                .render(&[ChatMessage::user("hi")], None, &options, true)
                .expect_err("the option must be refused");
            assert!(
                format!("{error:#}").contains("enable_thinking"),
                "{error:#}"
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

        // Both clock paths print the microseconds and drop the zone
        // directives; the POSIX host asserts them here and the Windows host
        // asserts them in its own test.
        #[cfg(all(
            target_pointer_width = "64",
            any(target_os = "linux", target_os = "macos")
        ))]
        {
            let microseconds = render_template("{{ strftime_now('%f') }}", None)?;
            assert_eq!(microseconds.len(), 6);
            assert!(microseconds.bytes().all(|byte| byte.is_ascii_digit()));
            assert_eq!(
                render_template("{{ strftime_now('%z|%Z|%:z') }}", None)?,
                "||"
            );
        }
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn formats_the_current_time_with_strftime_on_windows() -> Result<()> {
        let rendered = render_template("{{ strftime_now('%Y') }}", None)?;
        assert_eq!(rendered.len(), 4, "{rendered}");
        assert!(
            rendered.bytes().all(|byte| byte.is_ascii_digit()),
            "{rendered}"
        );
        assert!(render_template("{{ strftime_now('%Q') }}", None).is_err());

        // The Windows clock prints the microseconds and drops the zone
        // directives, exactly like the POSIX clock does.
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

    /// The path of the pinned SmolLM3 template, the real template that
    /// carries the transformers-only generation markers.
    const SMOLLM3_TEMPLATE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/chat_templates/templates/pinned/smollm3-official-system-override-text-subset.jinja"
    );

    #[test]
    fn the_generation_markers_of_the_real_template_compile_and_render() -> Result<()> {
        let source = std::fs::read_to_string(SMOLLM3_TEMPLATE)?;
        let template = JinjaChatTemplate::new(&source, "", "")?;
        let options = ChatTemplateOptions::default();
        let rendered = template.render(&[ChatMessage::user("hello")], None, &options, true)?;
        assert!(rendered.contains("hello"), "{rendered}");
        assert!(rendered.contains("assistant"), "{rendered}");
        Ok(())
    }

    #[test]
    fn the_generation_markers_keep_their_whitespace_control() {
        assert_eq!(rewrite_generation_tags("{% generation %}"), "{% if true %}");
        assert_eq!(
            rewrite_generation_tags("{% endgeneration %}"),
            "{% endif %}"
        );
        assert_eq!(
            rewrite_generation_tags("{%- generation -%}"),
            "{%- if true -%}"
        );
        assert_eq!(
            rewrite_generation_tags("{%- endgeneration -%}"),
            "{%- endif -%}"
        );
        assert_eq!(
            rewrite_generation_tags("{%+ generation +%}"),
            "{%+ if true +%}"
        );
        assert_eq!(
            rewrite_generation_tags("{% generation-%}"),
            "{% if true -%}"
        );
        assert_eq!(
            rewrite_generation_tags("{%-generation-%}"),
            "{%- if true -%}"
        );
        assert_eq!(
            rewrite_generation_tags("{%- endgeneration-%}"),
            "{%- endif -%}"
        );
        assert_eq!(
            rewrite_generation_tags("{% generation_x-%}"),
            "{% generation_x-%}"
        );
    }

    #[test]
    fn a_rewritten_marker_renders_like_the_block_it_becomes() -> Result<()> {
        let rewritten =
            JinjaChatTemplate::new("a {%- generation -%}b{%- endgeneration -%} c", "", "")?;
        let explicit = JinjaChatTemplate::new("a {%- if true -%}b{%- endif -%} c", "", "")?;
        let options = ChatTemplateOptions::default();
        let messages = [ChatMessage::user("hi")];
        assert_eq!(
            rewritten.render(&messages, None, &options, true)?,
            explicit.render(&messages, None, &options, true)?
        );
        Ok(())
    }

    #[test]
    fn a_comment_a_raw_block_and_another_tag_keep_their_text() {
        assert_eq!(
            rewrite_generation_tags("{# {% generation %} #}"),
            "{# {% generation %} #}"
        );
        assert_eq!(
            rewrite_generation_tags("{% raw %}{% generation %}{% endgeneration %}{% endraw %}"),
            "{% raw %}{% generation %}{% endgeneration %}{% endraw %}"
        );
        assert_eq!(
            rewrite_generation_tags("{% generation_x %}"),
            "{% generation_x %}"
        );
        assert_eq!(
            rewrite_generation_tags("{% endgeneration_x %}"),
            "{% endgeneration_x %}"
        );
        assert_eq!(
            rewrite_generation_tags("{{ '{% generation %}' }}"),
            "{{ '{% generation %}' }}"
        );
    }

    #[test]
    fn a_template_that_uses_the_option_receives_it() -> Result<()> {
        let source = "{{ enable_thinking }}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(template.honors_enable_thinking());
        let options = ChatTemplateOptions {
            enable_thinking: Some(false),
        };
        assert_eq!(
            template.render(&[ChatMessage::user("hi")], None, &options, true)?,
            "False"
        );
        Ok(())
    }

    #[test]
    fn a_template_that_reads_the_option_into_a_variable_of_its_own_honors_it() -> Result<()> {
        // The static analysis of the program cannot see that this template
        // reads the option: it assigns the target before it visits the
        // expression. The renders of the template show that it honors it.
        let source = "{% set enable_thinking = enable_thinking | default(true) %}{% if enable_thinking %}<think>{% endif %}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(template.honors_enable_thinking(), "{source}");
        let options = ChatTemplateOptions {
            enable_thinking: Some(false),
        };
        assert_eq!(
            template.render(&[ChatMessage::user("hi")], None, &options, true)?,
            ""
        );
        let options = ChatTemplateOptions {
            enable_thinking: Some(true),
        };
        assert_eq!(
            template.render(&[ChatMessage::user("hi")], None, &options, true)?,
            "<think>"
        );
        Ok(())
    }

    #[test]
    fn a_template_that_reads_the_option_in_an_expression_honors_it() -> Result<()> {
        let source = "{{ enable_thinking if enable_thinking is defined else true }}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(template.honors_enable_thinking(), "{source}");
        let options = ChatTemplateOptions {
            enable_thinking: Some(false),
        };
        assert_eq!(
            template.render(&[ChatMessage::user("hi")], None, &options, true)?,
            "False"
        );
        Ok(())
    }

    #[test]
    fn a_template_that_prints_the_same_text_either_way_does_not_honor_the_option() -> Result<()> {
        // The template reads the option, but the renders are the same, so
        // the option changes nothing.
        let source = "{{ 'same' if enable_thinking is defined else 'same' }}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(!template.honors_enable_thinking(), "{source}");
        let options = ChatTemplateOptions {
            enable_thinking: Some(false),
        };
        let error = template
            .render(&[ChatMessage::user("hi")], None, &options, true)
            .expect_err("the option must be refused");
        assert!(
            format!("{error:#}").contains("enable_thinking"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn a_template_that_prints_a_clock_does_not_honor_the_option() -> Result<()> {
        // The template never names the option, so the two renders of the
        // probe may differ by the clock alone. The template does not honor
        // the option, whatever the clock prints.
        let source = "{{ strftime_now('%f') }}{{ messages[0].content }}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(!template.honors_enable_thinking(), "{source}");
        let options = ChatTemplateOptions {
            enable_thinking: Some(false),
        };
        let error = template
            .render(&[ChatMessage::user("hi")], None, &options, true)
            .expect_err("the option must be refused");
        assert!(
            format!("{error:#}").contains("enable_thinking"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn a_template_that_closes_the_block_unless_the_option_is_true_defaults_to_off() -> Result<()> {
        // The template keeps the reasoning block closed unless the option
        // asks for it, so its own default is off.
        let source = "{% if enable_thinking is defined and enable_thinking %}<think>{% endif %}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(template.honors_enable_thinking(), "{source}");
        assert_eq!(template.default_enable_thinking(), Some(false), "{source}");
        Ok(())
    }

    #[test]
    fn a_template_that_keeps_the_block_open_unless_the_option_is_false_defaults_to_on() -> Result<()>
    {
        // The template keeps the reasoning block open unless the option
        // closes it, so its own default is on.
        let source = "{% if enable_thinking is not defined or enable_thinking %}<think>{% endif %}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(template.honors_enable_thinking(), "{source}");
        assert_eq!(template.default_enable_thinking(), Some(true), "{source}");
        Ok(())
    }

    #[test]
    fn a_template_that_defaults_the_option_to_true_reports_it() -> Result<()> {
        // The template names its own default, so the render without the
        // option is the render with the option true.
        let source = "{% if enable_thinking | default(true) %}<think>{% endif %}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(template.honors_enable_thinking(), "{source}");
        assert_eq!(template.default_enable_thinking(), Some(true), "{source}");
        Ok(())
    }

    #[test]
    fn a_template_that_never_names_the_option_has_no_default() -> Result<()> {
        let source = "{{ messages[0].content }}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(!template.honors_enable_thinking(), "{source}");
        assert_eq!(template.default_enable_thinking(), None, "{source}");
        Ok(())
    }

    #[test]
    fn a_template_that_prints_the_same_text_either_way_has_no_default() -> Result<()> {
        // The template reads the option, but the renders are the same, so
        // the option changes nothing and the template has no default.
        let source = "{{ 'same' if enable_thinking is defined else 'same' }}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(!template.honors_enable_thinking(), "{source}");
        assert_eq!(template.default_enable_thinking(), None, "{source}");
        Ok(())
    }

    #[test]
    fn a_template_that_raises_when_the_option_is_omitted_has_no_default() -> Result<()> {
        // The template honors the option, but its render without the option
        // fails, so the default is unknown.
        let source = "{% if enable_thinking is defined %}{{ 'on' if enable_thinking else 'off' }}{% else %}{{ raise_exception('the option is required') }}{% endif %}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(template.honors_enable_thinking(), "{source}");
        assert_eq!(template.default_enable_thinking(), None, "{source}");
        Ok(())
    }

    #[test]
    fn a_template_that_prints_a_third_text_when_the_option_is_omitted_has_no_default() -> Result<()>
    {
        // The template honors the option, but the render without the option
        // is neither of the two renders, so the default is unknown.
        let source = "{% if enable_thinking is defined %}{{ 'on' if enable_thinking else 'off' }}{% else %}unset{% endif %}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(template.honors_enable_thinking(), "{source}");
        assert_eq!(template.default_enable_thinking(), None, "{source}");
        Ok(())
    }

    #[test]
    fn a_template_that_never_uses_the_option_refuses_it() -> Result<()> {
        let source = "{{ messages[0].content }}";
        let template = JinjaChatTemplate::new(source, "", "")?;
        assert!(!template.honors_enable_thinking());
        let options = ChatTemplateOptions {
            enable_thinking: Some(false),
        };
        let error = template
            .render(&[ChatMessage::user("hi")], None, &options, true)
            .expect_err("the option must be refused");
        assert!(
            format!("{error:#}").contains("enable_thinking"),
            "{error:#}"
        );
        // The option the template never reads changes nothing else.
        let unset = ChatTemplateOptions::default();
        assert_eq!(
            template.render(&[ChatMessage::user("hi")], None, &unset, true)?,
            "hi"
        );
        Ok(())
    }

    #[test]
    fn a_mention_of_the_option_in_a_comment_or_a_string_does_not_count() -> Result<()> {
        for source in [
            "{# enable_thinking #}{{ messages[0].content }}",
            "{{ 'enable_thinking' }}{{ messages[0].content }}",
        ] {
            let template = JinjaChatTemplate::new(source, "", "")?;
            assert!(!template.honors_enable_thinking(), "{source}");
        }
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
