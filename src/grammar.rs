//! Fail-closed constrained decoding for strict JSON response formats and
//! native tool calls.
//!
//! The grammar checks a candidate token's complete decoded byte fragment
//! before the sampler may select it. It deliberately does not accept token
//! vocabulary strings: byte-level `BPE` and `SentencePiece` vocabulary entries
//! are not necessarily the text emitted by decoding that token in the current
//! context. Callers must provide the exact, context-sensitive output bytes for
//! every candidate.
//!
//! The supported JSON Schema subset is intentionally small and closed. A
//! schema that contains an unrecognized keyword is rejected at compile time;
//! it never silently degrades to unconstrained JSON. Annotations (`title`,
//! `description`, `default`, `examples`, `$comment`, `$schema`) are accepted
//! and ignored. Object schemas require `additionalProperties: false`.
//! Non-strict tool schemas are compiled closed: their validation keywords
//! that the grammar cannot express (length, pattern, format, numeric bounds,
//! item counts) are accepted but not enforced, while strict schemas reject
//! them. Composition (`anyOf`, `oneOf`, `allOf`, `$ref`) and type arrays are
//! rejected in every mode. `enum` and `const` values are emitted using
//! their canonical `serde_json` spelling, which is a valid (and deterministic)
//! subset of the representations accepted by JSON Schema.
//!
//! Native tool calls wrap the same argument grammar in the Qwen3 call
//! JSON envelope or Qwen3.5's schema-directed XML parameter blocks.
//! [`LazyToolCallConstraint`] leaves ordinary text unconstrained and applies
//! the selected envelope only after the model opens a call.

use std::{
    collections::{BTreeSet, HashSet, VecDeque},
    sync::Arc,
};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};
use tokenizers::{Tokenizer, tokenizer::step_decode_stream};

use crate::{
    sampling::LogitsConstraint,
    tool_call::{
        NativeToolCall, TOOL_CALL_OPEN, TOOL_CALL_TAIL, ToolCallScan, parse_tool_call,
        parse_xml_tool_call, scan_tool_call, tool_call_head,
    },
};

/// Default hard limit for one constrained response.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 1024 * 1024;

const MAX_SCHEMA_DEPTH: usize = 64;
const MAX_SCHEMA_PROPERTIES: usize = 128;
const MAX_ENUM_VALUES: usize = 256;
const MAX_ARRAY_ITEMS: usize = 4096;

/// Whether the current byte prefix can still become a schema-valid JSON value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarStatus {
    /// The prefix is valid but cannot end yet.
    Viable,
    /// The prefix is already a complete valid value. More JSON whitespace (or
    /// a lexical continuation of a number) may still be accepted.
    Complete,
    /// No continuation can turn this byte sequence into a valid value.
    Dead,
}

/// A strict, validated representation of the supported JSON Schema subset.
#[derive(Debug, Clone)]
pub struct CompiledJsonSchema {
    root: SchemaNode,
}

impl CompiledJsonSchema {
    /// Compile a schema, rejecting boolean schemas, unsupported keywords,
    /// open-ended objects, invalid `required` declarations, and values that do
    /// not conform to their own `type` plus `enum`/`const` constraints.
    pub fn compile(schema: &Value) -> Result<Self> {
        Ok(Self {
            root: compile_schema_node(schema, "$", 0, SchemaMode::Strict)?,
        })
    }

    /// Compile a non-strict function schema by closing every object to its
    /// declared properties. `additionalProperties` may be absent or boolean
    /// and `required` may name a subset of the properties. Closing only
    /// narrows the accepted values. Validation keywords the grammar cannot
    /// express (string length, `pattern`, `format`, numeric bounds,
    /// `multipleOf`, item counts) are accepted but not enforced, so generated
    /// arguments can violate them; every other rule of the strict subset
    /// still applies.
    pub fn compile_closed(schema: &Value) -> Result<Self> {
        Ok(Self {
            root: compile_schema_node(schema, "$", 0, SchemaMode::Closed)?,
        })
    }

    /// Whether a complete JSON value conforms to the compiled schema.
    #[must_use]
    pub fn accepts(&self, value: &Value) -> bool {
        node_conforms(&self.root, value)
    }
}

/// How object schemas are admitted into the compiled subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchemaMode {
    /// `OpenAI` strict mode: closed objects whose properties are all required.
    Strict,
    /// A non-strict schema narrowed to its declared properties.
    Closed,
}

/// Result of applying a grammar mask to one logit row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateMask {
    /// Number of vocabulary entries left selectable by the grammar.
    pub allowed_tokens: usize,
    /// Whether an EOS token was eligible because the JSON value is complete.
    pub can_end: bool,
}

/// Request-local incremental JSON Schema grammar.
///
/// Prefix analysis reparses the bounded response buffer. This prioritizes a
/// small, auditable correctness boundary; a future optimized implementation
/// can preserve this API while caching parser states per token block.
#[derive(Debug, Clone)]
pub struct JsonSchemaGrammar {
    schema: CompiledJsonSchema,
    output: Vec<u8>,
    max_output_bytes: usize,
}

/// A concrete sampler constraint backed by a Hugging Face tokenizer.
///
/// The constraint owns the selected token history and a cloneable mirror of
/// `tokenizers::DecodeStream`'s state. Each vocabulary candidate is evaluated
/// by `step_decode_stream` against a cloned current state, so the grammar sees
/// the exact context-sensitive text delta rather than a raw vocabulary entry.
/// Only the selected candidate advances the real decoder and grammar states.
///
/// A non-EOS token that produces `None` or an empty fragment is rejected. This
/// is deliberately conservative: `None` can mean either a skipped special
/// token or the first half of a byte-fallback sequence. Supporting the latter
/// safely would require grammar states over pending non-UTF-8 bytes and
/// multi-token lookahead; this implementation never claims that support.
#[derive(Clone)]
pub struct JsonSchemaTokenConstraint {
    tokenizer: Arc<Tokenizer>,
    eos_token_ids: BTreeSet<u32>,
    selected_token_ids: Vec<u32>,
    decoder: DecoderState,
    grammar: JsonSchemaGrammar,
    finished: bool,
}

#[derive(Debug, Clone, Default)]
struct DecoderState {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl JsonSchemaTokenConstraint {
    /// Construct a constraint from an already compiled schema.
    pub fn new(
        tokenizer: Arc<Tokenizer>,
        eos_token_ids: Vec<u32>,
        schema: CompiledJsonSchema,
    ) -> Result<Self> {
        Self::with_grammar(tokenizer, eos_token_ids, JsonSchemaGrammar::new(schema))
    }

    /// Compile a schema and construct a tokenizer-backed constraint.
    pub fn from_schema(
        tokenizer: Arc<Tokenizer>,
        eos_token_ids: Vec<u32>,
        schema: &Value,
    ) -> Result<Self> {
        Self::new(
            tokenizer,
            eos_token_ids,
            CompiledJsonSchema::compile(schema)?,
        )
    }

    /// Construct from a configured grammar, preserving its output-size limit.
    pub fn with_grammar(
        tokenizer: Arc<Tokenizer>,
        eos_token_ids: Vec<u32>,
        grammar: JsonSchemaGrammar,
    ) -> Result<Self> {
        let eos = validate_eos_token_ids(&tokenizer, eos_token_ids)?;
        Ok(Self {
            tokenizer,
            eos_token_ids: eos,
            selected_token_ids: Vec::new(),
            decoder: DecoderState::default(),
            grammar,
            finished: false,
        })
    }

    #[must_use]
    pub fn grammar(&self) -> &JsonSchemaGrammar {
        &self.grammar
    }

    #[must_use]
    pub fn selected_token_ids(&self) -> &[u32] {
        &self.selected_token_ids
    }

    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.finished
    }

    fn candidate_step(&self, token_id: u32) -> Result<(DecoderState, Option<String>)> {
        decode_candidate(&self.tokenizer, &self.decoder, token_id)
    }

    fn candidate_is_allowed(&self, token_id: u32) -> bool {
        if self.eos_token_ids.contains(&token_id) {
            return self.grammar.can_end();
        }
        if self.grammar.can_end() {
            return false;
        }
        self.candidate_step(token_id)
            .ok()
            .and_then(|(_, fragment)| fragment)
            .filter(|fragment| !fragment.is_empty())
            .is_some_and(|fragment| self.grammar.allows_token_text(&fragment))
    }
}

fn validate_eos_token_ids(tokenizer: &Tokenizer, eos_token_ids: Vec<u32>) -> Result<BTreeSet<u32>> {
    ensure!(
        !eos_token_ids.is_empty(),
        "at least one EOS token ID is required"
    );
    let vocabulary_size = tokenizer.get_vocab_size(true);
    let eos_count = eos_token_ids.len();
    let eos = eos_token_ids.into_iter().collect::<BTreeSet<_>>();
    ensure!(
        eos.len() == eos_count,
        "EOS token ID list contains duplicates"
    );
    for token_id in &eos {
        let token_index = usize::try_from(*token_id).context("EOS token ID exceeds usize")?;
        ensure!(
            token_index < vocabulary_size,
            "EOS token ID {token_id} is outside tokenizer vocabulary size {vocabulary_size}"
        );
    }
    Ok(eos)
}

/// Decode one candidate against a cloned decoder state, returning the state
/// selecting it would produce and its exact context-sensitive text delta.
fn decode_candidate(
    tokenizer: &Tokenizer,
    decoder: &DecoderState,
    token_id: u32,
) -> Result<(DecoderState, Option<String>)> {
    let mut next = decoder.clone();
    let fragment = step_decode_stream(
        tokenizer,
        vec![token_id],
        true,
        &mut next.ids,
        &mut next.prefix,
        &mut next.prefix_index,
    )
    .map_err(|error| anyhow::anyhow!("tokenizer rejected candidate token {token_id}: {error}"))?;
    Ok((next, fragment))
}

impl LogitsConstraint for JsonSchemaTokenConstraint {
    fn mask_logits(&mut self, logits: &mut [f32]) -> Result<()> {
        ensure!(
            !self.finished,
            "cannot sample after the JSON grammar selected EOS"
        );
        for (index, logit) in logits.iter_mut().enumerate() {
            if !logit.is_finite() {
                continue;
            }
            let token_id = u32::try_from(index).context("vocabulary exceeds u32 token IDs")?;
            if !self.candidate_is_allowed(token_id) {
                *logit = f32::NEG_INFINITY;
            }
        }
        ensure!(
            logits.iter().any(|logit| logit.is_finite()),
            "JSON grammar rejected every finite token candidate"
        );
        Ok(())
    }

    fn commit_token(&mut self, token_id: u32) -> Result<()> {
        ensure!(
            !self.finished,
            "cannot commit a token after the JSON grammar selected EOS"
        );
        if self.eos_token_ids.contains(&token_id) {
            ensure!(
                self.grammar.can_end(),
                "cannot commit EOS before a complete schema-valid JSON value"
            );
            self.selected_token_ids.push(token_id);
            self.finished = true;
            return Ok(());
        }
        ensure!(
            !self.grammar.can_end(),
            "only EOS may be committed after the JSON value is complete"
        );

        let (next_decoder, fragment) = self.candidate_step(token_id)?;
        let fragment = fragment
            .filter(|fragment| !fragment.is_empty())
            .with_context(|| {
                format!(
                    "token {token_id} produced no text; pending byte-fallback and skipped special tokens are conservatively rejected"
                )
            })?;
        self.grammar.commit_token_text(&fragment)?;
        self.decoder = next_decoder;
        self.selected_token_ids.push(token_id);
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        self.grammar.finish()
    }

    fn checks_tokens(&self) -> bool {
        true
    }

    fn allows_token(&self, token: u32) -> Result<bool> {
        ensure!(
            !self.finished,
            "cannot sample after the JSON grammar selected EOS"
        );
        Ok(self.candidate_is_allowed(token))
    }
}

impl JsonSchemaGrammar {
    #[must_use]
    pub fn new(schema: CompiledJsonSchema) -> Self {
        Self {
            schema,
            output: Vec::new(),
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        }
    }

    pub fn from_schema(schema: &Value) -> Result<Self> {
        Ok(Self::new(CompiledJsonSchema::compile(schema)?))
    }

    pub fn with_max_output_bytes(mut self, max_output_bytes: usize) -> Result<Self> {
        ensure!(
            max_output_bytes > 0,
            "grammar output limit must be positive"
        );
        self.max_output_bytes = max_output_bytes;
        Ok(self)
    }

    #[must_use]
    pub fn status(&self) -> GrammarStatus {
        analyze_prefix(&self.schema.root, &self.output, self.max_output_bytes)
    }

    #[must_use]
    pub fn can_end(&self) -> bool {
        self.status() == GrammarStatus::Complete
    }

    #[must_use]
    pub fn output_bytes(&self) -> &[u8] {
        &self.output
    }

    #[must_use]
    /// Return the already-validated UTF-8 output.
    ///
    /// # Panics
    ///
    /// Panics only if the internal invariant maintained by
    /// [`Self::commit_token_bytes`] is violated.
    pub fn output_text(&self) -> &str {
        // `commit_token_bytes` rejects invalid UTF-8, so this invariant is
        // maintained at every public mutation boundary.
        std::str::from_utf8(&self.output).expect("grammar output is valid UTF-8")
    }

    /// Check a candidate's exact decoded bytes without mutating grammar state.
    /// Empty fragments are rejected: EOS must be passed separately to
    /// [`Self::mask_candidate_logits`] so it is only enabled at a complete JSON
    /// boundary.
    #[must_use]
    pub fn allows_token_bytes(&self, token_bytes: &[u8]) -> bool {
        if token_bytes.is_empty()
            || self.output.len().saturating_add(token_bytes.len()) > self.max_output_bytes
        {
            return false;
        }
        let mut candidate = self.output.clone();
        candidate.extend_from_slice(token_bytes);
        analyze_prefix(&self.schema.root, &candidate, self.max_output_bytes) != GrammarStatus::Dead
    }

    #[must_use]
    pub fn allows_token_text(&self, token_text: &str) -> bool {
        self.allows_token_bytes(token_text.as_bytes())
    }

    /// Commit a token only after applying the same pre-sampling viability test
    /// used by [`Self::allows_token_bytes`].
    pub fn commit_token_bytes(&mut self, token_bytes: &[u8]) -> Result<GrammarStatus> {
        ensure!(!token_bytes.is_empty(), "grammar tokens must make progress");
        ensure!(
            self.allows_token_bytes(token_bytes),
            "token bytes make the JSON grammar prefix impossible"
        );
        self.output.extend_from_slice(token_bytes);
        Ok(self.status())
    }

    pub fn commit_token_text(&mut self, token_text: &str) -> Result<GrammarStatus> {
        self.commit_token_bytes(token_text.as_bytes())
    }

    /// Ensure generation ended at a complete schema-valid JSON value.
    pub fn finish(&self) -> Result<()> {
        ensure!(
            self.can_end(),
            "constrained generation ended before a complete schema-valid JSON value"
        );
        Ok(())
    }

    pub fn reset(&mut self) {
        self.output.clear();
    }

    /// Mask a vocabulary-sized logit row before temperature, top-k/top-p, or
    /// greedy selection is applied.
    ///
    /// `decode_candidate` must return the exact bytes that selecting `token_id`
    /// would append in the current decoder context. Returning a tokenizer's raw
    /// vocabulary spelling (for example, `\u{2581}` or byte-level surrogate
    /// characters) is unsafe and violates this contract. `None` marks a
    /// non-text special token. EOS IDs are controlled exclusively by
    /// `eos_token_ids` and are allowed only when [`Self::can_end`] is true.
    pub fn mask_candidate_logits<F>(
        &self,
        logits: &mut [f32],
        eos_token_ids: &[u32],
        mut decode_candidate: F,
    ) -> Result<CandidateMask>
    where
        F: FnMut(u32) -> Result<Option<Vec<u8>>>,
    {
        let can_end = self.can_end();
        let eos = eos_token_ids.iter().copied().collect::<HashSet<_>>();
        let mut allowed_tokens = 0;

        for (index, logit) in logits.iter_mut().enumerate() {
            let token_id = u32::try_from(index).context("vocabulary exceeds u32 token IDs")?;
            let allowed = if eos.contains(&token_id) {
                can_end
            } else if can_end {
                false
            } else {
                decode_candidate(token_id)?
                    .as_deref()
                    .is_some_and(|bytes| self.allows_token_bytes(bytes))
            };
            if allowed {
                allowed_tokens += 1;
            } else {
                *logit = f32::NEG_INFINITY;
            }
        }

        ensure!(
            allowed_tokens > 0,
            "JSON grammar rejected every token candidate"
        );
        Ok(CandidateMask {
            allowed_tokens,
            can_end,
        })
    }
}

/// One declared function a model-chosen native call may name.
#[derive(Debug, Clone)]
struct ToolCallSchema {
    name: String,
    /// Canonical envelope bytes from after `<tool_call>` up to the arguments.
    head: Vec<u8>,
    arguments: CompiledJsonSchema,
}

/// The calls a lazily constrained native tool turn may produce: one declared
/// name followed by arguments valid under that function's compiled schema.
#[derive(Debug, Clone)]
pub struct CompiledToolCalls {
    tools: Vec<ToolCallSchema>,
    xml: bool,
}

impl CompiledToolCalls {
    /// Pair each declared function name with its compiled arguments schema.
    pub fn new(tools: impl IntoIterator<Item = (String, CompiledJsonSchema)>) -> Result<Self> {
        let mut names = HashSet::new();
        let mut compiled = Vec::new();
        for (name, arguments) in tools {
            ensure!(!name.is_empty(), "tool names must not be empty");
            ensure!(names.insert(name.clone()), "duplicate tool name {name:?}");
            let encoded = serde_json::to_string(&name).context("serializing tool name")?;
            compiled.push(ToolCallSchema {
                head: tool_call_head(&encoded).into_bytes(),
                name,
                arguments,
            });
        }
        ensure!(
            !compiled.is_empty(),
            "a native tool call grammar requires at least one declared tool"
        );
        Ok(Self {
            tools: compiled,
            xml: false,
        })
    }

    /// Compile the same declarations for Qwen3.5's native XML envelope.
    /// Function and parameter names must be representable as literal tag names.
    pub fn qwen35(tools: impl IntoIterator<Item = (String, CompiledJsonSchema)>) -> Result<Self> {
        let mut calls = Self::new(tools)?;
        for tool in &mut calls.tools {
            ensure!(
                xml_name_is_safe(&tool.name),
                "tool name cannot be represented in XML: {:?}",
                tool.name
            );
            let Some(SchemaKind::Object(object)) = &tool.arguments.root.kind else {
                bail!("Qwen3.5 tool arguments must have type object");
            };
            for property in &object.properties {
                ensure!(
                    xml_name_is_safe(&property.name),
                    "parameter name cannot be represented in XML: {:?}",
                    property.name
                );
            }
            tool.head = format!("\n<function={}>\n", tool.name).into_bytes();
        }
        calls.xml = true;
        Ok(calls)
    }

    /// Parse and revalidate a generated call using this declaration's syntax.
    /// XML values are converted by their declared types; insertion order is
    /// preserved when serializing the compact arguments object.
    pub fn parse_call(&self, after_open: &str) -> Result<NativeToolCall> {
        let call = if self.xml {
            let raw = parse_xml_tool_call(after_open)?;
            let tool = self
                .tools
                .iter()
                .find(|tool| tool.name == raw.name)
                .with_context(|| format!("tool `{}` is not declared", raw.name))?;
            let Some(SchemaKind::Object(object)) = &tool.arguments.root.kind else {
                bail!("XML tool arguments must have type object");
            };
            let mut arguments = Map::new();
            for (name, text) in raw.parameters {
                let property = object
                    .properties
                    .iter()
                    .find(|property| property.name == name)
                    .with_context(|| format!("parameter {name:?} is not declared"))?;
                arguments.insert(name, convert_xml_value(&property.schema, &text)?);
            }
            let arguments = Value::Object(arguments);
            NativeToolCall {
                name: raw.name,
                arguments_text: serde_json::to_string(&arguments)?,
                arguments,
            }
        } else {
            parse_tool_call(after_open)?
        };
        self.validate_call(&call.name, &call.arguments)?;
        Ok(call)
    }

    /// Check a parsed call against the declared names and their schemas.
    /// The serving layer re-validates constrained output with this before
    /// exposing it, so a grammar defect cannot leak an invalid call.
    pub fn validate_call(&self, name: &str, arguments: &Value) -> Result<()> {
        let tool = self
            .tools
            .iter()
            .find(|tool| tool.name == name)
            .with_context(|| format!("tool `{name}` is not declared"))?;
        ensure!(
            arguments.is_object(),
            "tool `{name}` arguments are not a JSON object"
        );
        ensure!(
            tool.arguments.accepts(arguments),
            "tool `{name}` arguments do not satisfy its schema"
        );
        Ok(())
    }
}

/// Request-local grammar for the bytes after a native call's opening tag:
/// `\n{"name": "<declared>", "arguments": <schema-valid object>}\n</tool_call>`.
///
/// The default envelope is the canonical spelling Qwen3 is prompted with.
/// Declarations compiled with [`CompiledToolCalls::qwen35`] instead select
/// `\n<function=NAME>\n` and schema-directed raw parameter blocks. Both
/// envelopes admit exactly one call and no bytes after its closing tag.
#[derive(Debug, Clone)]
pub struct ToolCallGrammar {
    calls: CompiledToolCalls,
    output: Vec<u8>,
    max_output_bytes: usize,
}

impl ToolCallGrammar {
    #[must_use]
    pub fn new(calls: CompiledToolCalls) -> Self {
        Self {
            calls,
            output: Vec::new(),
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        }
    }

    #[must_use]
    pub fn status(&self) -> GrammarStatus {
        self.analyze(&self.output)
    }

    fn analyze(&self, bytes: &[u8]) -> GrammarStatus {
        if self.calls.xml {
            analyze_xml_tool_call(&self.calls.tools, bytes, self.max_output_bytes)
        } else {
            analyze_tool_call(&self.calls.tools, bytes, self.max_output_bytes)
        }
    }

    #[must_use]
    pub fn can_end(&self) -> bool {
        self.status() == GrammarStatus::Complete
    }

    /// Return the already-validated UTF-8 call text after the opening tag.
    ///
    /// # Panics
    ///
    /// Panics only if the UTF-8 invariant kept by [`Self::commit_token_text`]
    /// is violated.
    #[must_use]
    pub fn output_text(&self) -> &str {
        std::str::from_utf8(&self.output).expect("grammar output is valid UTF-8")
    }

    /// Check a candidate's exact decoded text without mutating state.
    #[must_use]
    pub fn allows_token_text(&self, token_text: &str) -> bool {
        if token_text.is_empty()
            || self.output.len().saturating_add(token_text.len()) > self.max_output_bytes
        {
            return false;
        }
        let mut candidate = self.output.clone();
        candidate.extend_from_slice(token_text.as_bytes());
        self.analyze(&candidate) != GrammarStatus::Dead
    }

    pub fn commit_token_text(&mut self, token_text: &str) -> Result<GrammarStatus> {
        ensure!(!token_text.is_empty(), "grammar tokens must make progress");
        ensure!(
            self.allows_token_text(token_text),
            "token text makes the native tool call impossible"
        );
        self.output.extend_from_slice(token_text.as_bytes());
        Ok(self.status())
    }

    /// Ensure generation ended at a complete, closed native call.
    pub fn finish(&self) -> Result<()> {
        ensure!(
            self.can_end(),
            "generation ended before the native tool call was complete"
        );
        Ok(())
    }
}

fn analyze_tool_call(
    tools: &[ToolCallSchema],
    bytes: &[u8],
    max_output_bytes: usize,
) -> GrammarStatus {
    if bytes.len() > max_output_bytes || std::str::from_utf8(bytes).is_err() {
        return GrammarStatus::Dead;
    }
    let mut viable = false;
    let mut complete = false;
    for tool in tools {
        let head = match_terminal(bytes, 0, &tool.head);
        viable |= head.incomplete;
        for after_head in head.complete {
            let arguments = parse_node(&tool.arguments.root, bytes, after_head, 0);
            viable |= arguments.incomplete;
            for arguments_end in arguments.complete {
                let tail = match_terminal(bytes, arguments_end, TOOL_CALL_TAIL.as_bytes());
                viable |= tail.incomplete;
                complete |= tail.complete.contains(&bytes.len());
            }
        }
    }
    if complete {
        GrammarStatus::Complete
    } else if viable {
        GrammarStatus::Viable
    } else {
        GrammarStatus::Dead
    }
}

const XML_PARAMETER_END: &str = "\n</parameter>\n";
const XML_FUNCTION_END: &[u8] = b"</function>\n</tool_call>";

fn xml_name_is_safe(name: &str) -> bool {
    // These are raw template fields, not XML identifiers. Only the field
    // terminator is unrepresentable; punctuation and empty property names
    // otherwise round-trip literally.
    !name.contains(">\n")
}

fn xml_exact_spellings(exact: &ExactValue) -> Vec<String> {
    match &exact.value {
        Value::String(text) => {
            if text.contains("\n</parameter>") {
                Vec::new()
            } else {
                vec![text.clone()]
            }
        }
        Value::Bool(true) => vec!["true".to_owned(), "True".to_owned()],
        Value::Bool(false) => vec!["false".to_owned(), "False".to_owned()],
        Value::Null => vec!["null".to_owned(), "None".to_owned()],
        _ => vec![String::from_utf8(exact.bytes.clone()).expect("serialized JSON is UTF-8")],
    }
}

/// Unlike JSON strings, XML string parameters have no quoting or escaping.
/// The newline-prefixed closing tag is their only reserved byte sequence.
fn convert_xml_value(schema: &SchemaNode, text: &str) -> Result<Value> {
    let value = match schema.kind.as_ref() {
        None => schema
            .exact
            .as_ref()
            .and_then(|values| {
                values.iter().find(|value| {
                    xml_exact_spellings(value)
                        .iter()
                        .any(|spelling| spelling == text)
                })
            })
            .context("parameter does not match enum or const")?
            .value
            .clone(),
        Some(SchemaKind::String) => {
            ensure!(
                !text.contains("\n</parameter>"),
                "string contains a parameter closing tag"
            );
            Value::String(text.to_owned())
        }
        Some(SchemaKind::Boolean) => match text {
            "true" | "True" => Value::Bool(true),
            "false" | "False" => Value::Bool(false),
            _ => bail!("invalid boolean parameter"),
        },
        Some(SchemaKind::Null) => {
            ensure!(matches!(text, "null" | "None"), "invalid null parameter");
            Value::Null
        }
        _ => {
            ensure!(
                parse_xml_value(schema, text.as_bytes(), 0)
                    .complete
                    .contains(&text.len()),
                "invalid JSON parameter"
            );
            serde_json::from_str(text).context("invalid JSON parameter")?
        }
    };
    ensure!(
        node_conforms(schema, &value),
        "parameter does not satisfy its schema"
    );
    Ok(value)
}

fn parse_xml_value(schema: &SchemaNode, input: &[u8], position: usize) -> ParseOutcomes {
    match schema.kind.as_ref() {
        None => {
            let mut outcomes = ParseOutcomes::default();
            for exact in schema.exact.iter().flatten() {
                for spelling in xml_exact_spellings(exact) {
                    outcomes.merge(match_terminal(input, position, spelling.as_bytes()));
                }
            }
            outcomes
        }
        Some(SchemaKind::String) => {
            if let Some(exact) = &schema.exact {
                let mut outcomes = ParseOutcomes::default();
                for candidate in exact {
                    if let Some(text) = candidate.value.as_str()
                        && !text.contains("\n</parameter>")
                    {
                        outcomes.merge(match_terminal(input, position, text.as_bytes()));
                    }
                }
                outcomes
            } else {
                // The full delimiter is handled by analyze_xml_function. In
                // its absence arbitrary raw text can always be extended.
                ParseOutcomes::incomplete()
            }
        }
        Some(SchemaKind::Boolean | SchemaKind::Null) => {
            let candidates = [
                ("true", Value::Bool(true)),
                ("True", Value::Bool(true)),
                ("false", Value::Bool(false)),
                ("False", Value::Bool(false)),
                ("null", Value::Null),
                ("None", Value::Null),
            ];
            let mut outcomes = ParseOutcomes::default();
            for (text, value) in candidates {
                if node_conforms(schema, &value) {
                    outcomes.merge(match_terminal(input, position, text.as_bytes()));
                }
            }
            outcomes
        }
        Some(SchemaKind::Object(_) | SchemaKind::Array(_)) => {
            let mut outcomes = parse_node(schema, input, skip_json_whitespace(input, position), 0);
            // JSON whitespace belongs to VALUE, but the separator's newline
            // must remain available as the start of the parameter tail.
            for end in outcomes.complete.clone() {
                for cursor in end..skip_json_whitespace(input, end) {
                    outcomes.complete.insert(cursor + 1);
                }
            }
            outcomes
        }
        _ => parse_node(schema, input, position, 0),
    }
}

fn analyze_xml_tool_call(tools: &[ToolCallSchema], bytes: &[u8], limit: usize) -> GrammarStatus {
    if bytes.len() > limit || std::str::from_utf8(bytes).is_err() {
        return GrammarStatus::Dead;
    }
    let mut viable = false;
    for tool in tools {
        let head = match_terminal(bytes, 0, &tool.head);
        viable |= head.incomplete;
        for position in head.complete {
            match analyze_xml_function(tool, bytes, position) {
                GrammarStatus::Complete => return GrammarStatus::Complete,
                GrammarStatus::Viable => viable = true,
                GrammarStatus::Dead => (),
            }
        }
    }
    if viable {
        GrammarStatus::Viable
    } else {
        GrammarStatus::Dead
    }
}

fn analyze_xml_function(tool: &ToolCallSchema, bytes: &[u8], mut position: usize) -> GrammarStatus {
    let Some(SchemaKind::Object(object)) = &tool.arguments.root.kind else {
        return GrammarStatus::Dead;
    };
    let mut arguments = Map::new();
    loop {
        if object
            .properties
            .iter()
            .all(|property| !property.required || arguments.contains_key(&property.name))
            && tool.arguments.accepts(&Value::Object(arguments.clone()))
        {
            let tail = match_terminal(bytes, position, XML_FUNCTION_END);
            if tail.complete.contains(&bytes.len()) {
                return GrammarStatus::Complete;
            }
            if tail.incomplete {
                return GrammarStatus::Viable;
            }
        }
        let mut next = None;
        for property in &object.properties {
            if arguments.contains_key(&property.name) {
                continue;
            }
            let head = format!("<parameter={}>\n", property.name);
            let matched = match_terminal(bytes, position, head.as_bytes());
            if matched.incomplete {
                return GrammarStatus::Viable;
            }
            if let Some(&start) = matched.complete.first() {
                next = Some((property, start));
                break;
            }
        }
        let Some((property, start)) = next else {
            return GrammarStatus::Dead;
        };
        let rest = std::str::from_utf8(&bytes[start..]).expect("checked UTF-8");
        if let Some(end) = rest.find("\n</parameter>") {
            let Ok(value) = convert_xml_value(&property.schema, &rest[..end]) else {
                return GrammarStatus::Dead;
            };
            let tail = match_terminal(bytes, start + end, XML_PARAMETER_END.as_bytes());
            if tail.incomplete {
                return GrammarStatus::Viable;
            }
            let Some(&end) = tail.complete.first() else {
                return GrammarStatus::Dead;
            };
            arguments.insert(property.name.clone(), value);
            position = end;
        } else {
            let value = parse_xml_value(&property.schema, bytes, start);
            if value.incomplete
                || value
                    .complete
                    .iter()
                    .any(|&end| match_terminal(bytes, end, XML_PARAMETER_END.as_bytes()).incomplete)
            {
                return GrammarStatus::Viable;
            }
            return GrammarStatus::Dead;
        }
    }
}

/// Tool-choice `auto` sampling constraint for native Qwen3 or Qwen3.5 calls.
///
/// Like llama.cpp's lazy grammars, sampling is unconstrained while the
/// decoded output is ordinary text or reasoning. Once [`scan_tool_call`]
/// sees the opening tag, every candidate is checked against
/// [`ToolCallGrammar`], and after `</tool_call>` completes only EOS remains
/// selectable, so a turn carries at most one call. Text answers are never
/// masked and pay only one incremental decode per sampled token.
///
/// The opening tag itself is sampled without a mask. If the token completing
/// it also carries bytes the envelope rejects, the commit fails and the
/// request errors instead of exposing an unconstrained call. Qwen3 emits the
/// tag as one atomic added token, so this is a defensive boundary.
#[derive(Clone)]
pub struct LazyToolCallConstraint {
    tokenizer: Arc<Tokenizer>,
    eos_token_ids: BTreeSet<u32>,
    decoder: DecoderState,
    /// Decoded output before the call opened, rescanned for the opening tag.
    text: String,
    grammar: ToolCallGrammar,
    call_opened: bool,
    finished: bool,
}

impl LazyToolCallConstraint {
    pub fn new(
        tokenizer: Arc<Tokenizer>,
        eos_token_ids: Vec<u32>,
        calls: CompiledToolCalls,
    ) -> Result<Self> {
        let eos = validate_eos_token_ids(&tokenizer, eos_token_ids)?;
        Ok(Self {
            tokenizer,
            eos_token_ids: eos,
            decoder: DecoderState::default(),
            text: String::new(),
            grammar: ToolCallGrammar::new(calls),
            call_opened: false,
            finished: false,
        })
    }

    #[must_use]
    pub fn grammar(&self) -> &ToolCallGrammar {
        &self.grammar
    }

    /// Whether the decoded output has opened a native call.
    #[must_use]
    pub const fn call_opened(&self) -> bool {
        self.call_opened
    }

    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.finished
    }

    fn call_candidate_is_allowed(&self, token_id: u32) -> bool {
        if self.eos_token_ids.contains(&token_id) {
            return self.grammar.can_end();
        }
        if self.grammar.can_end() {
            return false;
        }
        decode_candidate(&self.tokenizer, &self.decoder, token_id)
            .ok()
            .and_then(|(_, fragment)| fragment)
            .filter(|fragment| !fragment.is_empty())
            .is_some_and(|fragment| self.grammar.allows_token_text(&fragment))
    }
}

impl LogitsConstraint for LazyToolCallConstraint {
    fn mask_logits(&mut self, logits: &mut [f32]) -> Result<()> {
        ensure!(
            !self.finished,
            "cannot sample after the tool-call constraint selected EOS"
        );
        if !self.call_opened {
            return Ok(());
        }
        for (index, logit) in logits.iter_mut().enumerate() {
            if !logit.is_finite() {
                continue;
            }
            let token_id = u32::try_from(index).context("vocabulary exceeds u32 token IDs")?;
            if !self.call_candidate_is_allowed(token_id) {
                *logit = f32::NEG_INFINITY;
            }
        }
        ensure!(
            logits.iter().any(|logit| logit.is_finite()),
            "native tool call grammar rejected every finite token candidate"
        );
        Ok(())
    }

    fn commit_token(&mut self, token_id: u32) -> Result<()> {
        ensure!(
            !self.finished,
            "cannot commit a token after the tool-call constraint selected EOS"
        );
        if self.eos_token_ids.contains(&token_id) {
            ensure!(
                !self.call_opened || self.grammar.can_end(),
                "cannot commit EOS inside an incomplete native tool call"
            );
            self.finished = true;
            return Ok(());
        }

        let (next_decoder, fragment) = decode_candidate(&self.tokenizer, &self.decoder, token_id)?;
        if self.call_opened {
            ensure!(
                !self.grammar.can_end(),
                "only EOS may follow a complete native tool call"
            );
            let fragment = fragment
                .filter(|fragment| !fragment.is_empty())
                .with_context(|| {
                    format!(
                        "token {token_id} produced no text inside a native tool call; pending byte-fallback and skipped special tokens are conservatively rejected"
                    )
                })?;
            self.grammar.commit_token_text(&fragment)?;
            self.decoder = next_decoder;
            return Ok(());
        }

        self.decoder = next_decoder;
        if let Some(fragment) = fragment {
            self.text.push_str(&fragment);
        }
        if let ToolCallScan::Call { open, .. } = scan_tool_call(&self.text) {
            self.call_opened = true;
            let after_open = &self.text[open + TOOL_CALL_OPEN.len()..];
            if !after_open.is_empty() {
                self.grammar.commit_token_text(after_open).context(
                    "text sampled with the tool-call opening tag does not fit the call envelope",
                )?;
            }
        }
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        if self.call_opened {
            self.grammar.finish()
        } else {
            Ok(())
        }
    }

    fn checks_tokens(&self) -> bool {
        true
    }

    fn allows_token(&self, token: u32) -> Result<bool> {
        ensure!(
            !self.finished,
            "cannot sample after the tool-call constraint selected EOS"
        );
        Ok(!self.call_opened || self.call_candidate_is_allowed(token))
    }
}

#[derive(Debug, Clone)]
struct SchemaNode {
    kind: Option<SchemaKind>,
    exact: Option<Vec<ExactValue>>,
}

#[derive(Debug, Clone)]
struct ExactValue {
    value: Value,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
enum SchemaKind {
    Object(ObjectSchema),
    Array(Box<SchemaNode>),
    String,
    Number,
    Integer,
    Boolean,
    Null,
}

#[derive(Debug, Clone)]
struct ObjectSchema {
    properties: Vec<PropertySchema>,
}

#[derive(Debug, Clone)]
struct PropertySchema {
    name: String,
    encoded_name: Vec<u8>,
    schema: SchemaNode,
    required: bool,
}

fn compile_schema_node(
    schema: &Value,
    path: &str,
    depth: usize,
    mode: SchemaMode,
) -> Result<SchemaNode> {
    ensure!(
        depth <= MAX_SCHEMA_DEPTH,
        "schema nesting exceeds {MAX_SCHEMA_DEPTH} at {path}"
    );
    let object = schema.as_object().with_context(|| {
        format!("{path} must be a JSON object; boolean schemas are unsupported")
    })?;

    ensure!(
        !(object.contains_key("enum") && object.contains_key("const")),
        "{path} cannot combine enum and const in the supported subset"
    );

    let type_name = object
        .get("type")
        .map(|value| {
            value
                .as_str()
                .with_context(|| format!("{path}.type must be one supported type string"))
        })
        .transpose()?;

    let kind = match type_name {
        Some("object") => Some(compile_object_schema(object, path, depth, mode)?),
        Some("array") => Some(compile_array_schema(object, path, depth, mode)?),
        Some("string") => {
            let unenforced = UNENFORCED_STRING_KEYWORDS;
            reject_unknown_keywords(object, path, &["type", "enum", "const"], unenforced, mode)?;
            Some(SchemaKind::String)
        }
        Some("number") => {
            let unenforced = UNENFORCED_NUMBER_KEYWORDS;
            reject_unknown_keywords(object, path, &["type", "enum", "const"], unenforced, mode)?;
            Some(SchemaKind::Number)
        }
        Some("integer") => {
            let unenforced = UNENFORCED_NUMBER_KEYWORDS;
            reject_unknown_keywords(object, path, &["type", "enum", "const"], unenforced, mode)?;
            Some(SchemaKind::Integer)
        }
        Some("boolean") => {
            reject_unknown_keywords(object, path, &["type", "enum", "const"], &[], mode)?;
            Some(SchemaKind::Boolean)
        }
        Some("null") => {
            reject_unknown_keywords(object, path, &["type", "enum", "const"], &[], mode)?;
            Some(SchemaKind::Null)
        }
        Some(other) => bail!("{path}.type {other:?} is unsupported"),
        None => {
            reject_unknown_keywords(object, path, &["enum", "const"], &[], mode)?;
            ensure!(
                object.contains_key("enum") || object.contains_key("const"),
                "{path} must declare one supported type, enum, or const"
            );
            None
        }
    };

    let exact = compile_exact_values(object, path)?;
    if let (Some(kind), Some(values)) = (&kind, &exact) {
        for exact_value in values {
            ensure!(
                kind_conforms(kind, &exact_value.value),
                "{} value {} does not conform to {path}.type",
                if object.contains_key("const") {
                    "const"
                } else {
                    "enum"
                },
                exact_value.value
            );
        }
    }

    Ok(SchemaNode { kind, exact })
}

fn compile_object_schema(
    object: &Map<String, Value>,
    path: &str,
    depth: usize,
    mode: SchemaMode,
) -> Result<SchemaKind> {
    reject_unknown_keywords(
        object,
        path,
        &[
            "type",
            "properties",
            "required",
            "additionalProperties",
            "enum",
            "const",
        ],
        &[],
        mode,
    )?;
    match mode {
        SchemaMode::Strict => ensure!(
            object.get("additionalProperties") == Some(&Value::Bool(false)),
            "{path}.additionalProperties must be explicitly false"
        ),
        SchemaMode::Closed => ensure!(
            object
                .get("additionalProperties")
                .is_none_or(Value::is_boolean),
            "{path}.additionalProperties must be absent or a boolean"
        ),
    }
    let properties = object
        .get("properties")
        .and_then(Value::as_object)
        .with_context(|| format!("{path}.properties must be an object"))?;
    ensure!(
        properties.len() <= MAX_SCHEMA_PROPERTIES,
        "{path}.properties exceeds the limit of {MAX_SCHEMA_PROPERTIES}"
    );

    let no_required = Vec::new();
    let required_array = match (object.get("required"), mode) {
        (None, SchemaMode::Closed) => &no_required,
        (required, _) => required
            .and_then(Value::as_array)
            .with_context(|| format!("{path}.required must list every property"))?,
    };
    let mut required = HashSet::new();
    for (index, value) in required_array.iter().enumerate() {
        let name = value
            .as_str()
            .with_context(|| format!("{path}.required[{index}] must be a property-name string"))?;
        ensure!(
            required.insert(name.to_owned()),
            "{path}.required contains duplicate property {name:?}"
        );
    }
    for name in &required {
        ensure!(
            properties.contains_key(name),
            "{path}.required names unknown property {name:?}"
        );
    }
    ensure!(
        mode == SchemaMode::Closed || required.len() == properties.len(),
        "{path}.required must contain every property in strict mode"
    );

    let mut compiled = Vec::with_capacity(properties.len());
    for (name, property_schema) in properties {
        let property_path = format!("{path}.properties[{name:?}]");
        compiled.push(PropertySchema {
            name: name.clone(),
            encoded_name: serde_json::to_vec(name).context("serializing JSON property name")?,
            schema: compile_schema_node(property_schema, &property_path, depth + 1, mode)?,
            required: required.contains(name),
        });
    }
    Ok(SchemaKind::Object(ObjectSchema {
        properties: compiled,
    }))
}

fn compile_array_schema(
    object: &Map<String, Value>,
    path: &str,
    depth: usize,
    mode: SchemaMode,
) -> Result<SchemaKind> {
    reject_unknown_keywords(
        object,
        path,
        &["type", "items", "enum", "const"],
        UNENFORCED_ARRAY_KEYWORDS,
        mode,
    )?;
    let items = object
        .get("items")
        .with_context(|| format!("{path}.items is required for an array schema"))?;
    Ok(SchemaKind::Array(Box::new(compile_schema_node(
        items,
        &format!("{path}.items"),
        depth + 1,
        mode,
    )?)))
}

/// Keywords that only describe a value. Every mode accepts them; they stay
/// in the rendered tool declaration and the grammar ignores them.
const ANNOTATION_KEYWORDS: &[&str] = &[
    "title",
    "description",
    "default",
    "examples",
    "$comment",
    "$schema",
];

/// Validation keywords the grammar cannot enforce. Closed (non-strict)
/// schemas accept them so the model still sees them in the declaration, but
/// generated arguments are not checked against them; strict schemas reject
/// them because strict mode promises conformance.
const UNENFORCED_STRING_KEYWORDS: &[&str] = &["minLength", "maxLength", "pattern", "format"];
const UNENFORCED_NUMBER_KEYWORDS: &[&str] = &[
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
];
const UNENFORCED_ARRAY_KEYWORDS: &[&str] = &["minItems", "maxItems"];

fn reject_unknown_keywords(
    object: &Map<String, Value>,
    path: &str,
    allowed: &[&str],
    unenforced: &[&str],
    mode: SchemaMode,
) -> Result<()> {
    for (keyword, value) in object {
        let keyword = keyword.as_str();
        if allowed.contains(&keyword) {
            continue;
        }
        let well_formed = if ANNOTATION_KEYWORDS.contains(&keyword) {
            match keyword {
                "default" => true,
                "examples" => value.is_array(),
                _ => value.is_string(),
            }
        } else if mode == SchemaMode::Closed && unenforced.contains(&keyword) {
            match keyword {
                "minLength" | "maxLength" | "minItems" | "maxItems" => value.is_u64(),
                "pattern" | "format" => value.is_string(),
                // Draft 4 spells exclusive bounds as booleans.
                "exclusiveMinimum" | "exclusiveMaximum" => value.is_number() || value.is_boolean(),
                _ => value.is_number(),
            }
        } else {
            bail!("unsupported JSON Schema keyword {path}.{keyword}");
        };
        ensure!(well_formed, "{path}.{keyword} has an invalid value");
    }
    Ok(())
}

fn compile_exact_values(
    object: &Map<String, Value>,
    path: &str,
) -> Result<Option<Vec<ExactValue>>> {
    let values = if let Some(value) = object.get("const") {
        vec![value.clone()]
    } else if let Some(value) = object.get("enum") {
        let values = value
            .as_array()
            .with_context(|| format!("{path}.enum must be an array"))?;
        ensure!(!values.is_empty(), "{path}.enum must not be empty");
        ensure!(
            values.len() <= MAX_ENUM_VALUES,
            "{path}.enum exceeds the limit of {MAX_ENUM_VALUES}"
        );
        values.clone()
    } else {
        return Ok(None);
    };

    let mut seen = BTreeSet::new();
    let mut exact = Vec::with_capacity(values.len());
    for mut value in values {
        // `preserve_order` is needed by Qwen3 tool rendering, but exact
        // grammar literals and duplicate detection previously used sorted
        // object keys recursively. Keep those grammar semantics unchanged.
        value.sort_all_objects();
        let bytes = serde_json::to_vec(&value).context("serializing enum/const value")?;
        ensure!(
            seen.insert(bytes.clone()),
            "{path} contains a duplicate enum value"
        );
        exact.push(ExactValue { value, bytes });
    }
    Ok(Some(exact))
}

fn node_conforms(node: &SchemaNode, value: &Value) -> bool {
    if let Some(exact) = &node.exact
        && !exact.iter().any(|candidate| candidate.value == *value)
    {
        return false;
    }
    node.kind
        .as_ref()
        .is_none_or(|kind| kind_conforms(kind, value))
}

fn kind_conforms(kind: &SchemaKind, value: &Value) -> bool {
    match kind {
        SchemaKind::Object(schema) => {
            let Some(object) = value.as_object() else {
                return false;
            };
            if object.len() > schema.properties.len() {
                return false;
            }
            for property in &schema.properties {
                match object.get(&property.name) {
                    Some(value) if node_conforms(&property.schema, value) => {}
                    Some(_) => return false,
                    None if property.required => return false,
                    None => {}
                }
            }
            object.keys().all(|name| {
                schema
                    .properties
                    .iter()
                    .any(|property| property.name == *name)
            })
        }
        SchemaKind::Array(items) => value
            .as_array()
            .is_some_and(|values| values.iter().all(|value| node_conforms(items, value))),
        SchemaKind::String => value.is_string(),
        SchemaKind::Number => value.is_number(),
        SchemaKind::Integer => {
            value.as_i64().is_some()
                || value.as_u64().is_some()
                || value
                    .as_f64()
                    .is_some_and(|number| number.is_finite() && number.fract() == 0.0)
        }
        SchemaKind::Boolean => value.is_boolean(),
        SchemaKind::Null => value.is_null(),
    }
}

#[derive(Debug, Default)]
struct ParseOutcomes {
    complete: BTreeSet<usize>,
    incomplete: bool,
}

impl ParseOutcomes {
    fn completed(position: usize) -> Self {
        Self {
            complete: BTreeSet::from([position]),
            incomplete: false,
        }
    }

    const fn incomplete() -> Self {
        Self {
            complete: BTreeSet::new(),
            incomplete: true,
        }
    }

    fn merge(&mut self, other: Self) {
        self.complete.extend(other.complete);
        self.incomplete |= other.incomplete;
    }
}

fn analyze_prefix(root: &SchemaNode, bytes: &[u8], max_output_bytes: usize) -> GrammarStatus {
    if bytes.len() > max_output_bytes {
        return GrammarStatus::Dead;
    }
    let Ok(input) = std::str::from_utf8(bytes) else {
        // Candidate token fragments must be exact decoded UTF-8, not arbitrary
        // tokenizer vocabulary bytes or a partial code point.
        return GrammarStatus::Dead;
    };
    let input = input.as_bytes();
    let start = skip_json_whitespace(input, 0);
    let outcomes = parse_node(root, input, start, 0);
    let complete = outcomes
        .complete
        .iter()
        .any(|position| skip_json_whitespace(input, *position) == input.len());
    if complete {
        GrammarStatus::Complete
    } else if outcomes.incomplete {
        GrammarStatus::Viable
    } else {
        GrammarStatus::Dead
    }
}

fn parse_node(node: &SchemaNode, input: &[u8], position: usize, depth: usize) -> ParseOutcomes {
    if depth > MAX_SCHEMA_DEPTH {
        return ParseOutcomes::default();
    }
    if let Some(exact) = &node.exact {
        let mut outcomes = ParseOutcomes::default();
        for value in exact {
            outcomes.merge(match_terminal(input, position, &value.bytes));
        }
        return outcomes;
    }
    match node
        .kind
        .as_ref()
        .expect("compiled schema has kind or exact")
    {
        SchemaKind::Object(schema) => parse_object(schema, input, position, depth),
        SchemaKind::Array(items) => parse_array(items, input, position, depth),
        SchemaKind::String => parse_string(input, position),
        SchemaKind::Number => parse_number(input, position, false),
        SchemaKind::Integer => parse_number(input, position, true),
        SchemaKind::Boolean => {
            let mut outcomes = match_terminal(input, position, b"true");
            outcomes.merge(match_terminal(input, position, b"false"));
            outcomes
        }
        SchemaKind::Null => match_terminal(input, position, b"null"),
    }
}

fn match_terminal(input: &[u8], position: usize, terminal: &[u8]) -> ParseOutcomes {
    if position > input.len() {
        return ParseOutcomes::default();
    }
    let available = &input[position..];
    let compared = available.len().min(terminal.len());
    if available[..compared] != terminal[..compared] {
        return ParseOutcomes::default();
    }
    if available.len() < terminal.len() {
        ParseOutcomes::incomplete()
    } else {
        ParseOutcomes::completed(position + terminal.len())
    }
}

fn parse_string(input: &[u8], position: usize) -> ParseOutcomes {
    if position == input.len() {
        return ParseOutcomes::incomplete();
    }
    if input.get(position) != Some(&b'"') {
        return ParseOutcomes::default();
    }

    let mut cursor = position + 1;
    while cursor < input.len() {
        match input[cursor] {
            b'"' => {
                let end = cursor + 1;
                if serde_json::from_slice::<String>(&input[position..end]).is_ok() {
                    return ParseOutcomes::completed(end);
                }
                return ParseOutcomes::default();
            }
            b'\\' => {
                cursor += 1;
                if cursor == input.len() {
                    return ParseOutcomes::incomplete();
                }
                match input[cursor] {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => cursor += 1,
                    b'u' => {
                        let (code_unit, next) = match parse_hex_escape(input, cursor + 1) {
                            HexEscape::Complete(value, next) => (value, next),
                            HexEscape::Incomplete => return ParseOutcomes::incomplete(),
                            HexEscape::Invalid => return ParseOutcomes::default(),
                        };
                        cursor = next;
                        if (0xD800..=0xDBFF).contains(&code_unit) {
                            if input.len() < cursor + 2 {
                                return ParseOutcomes::incomplete();
                            }
                            if input[cursor] != b'\\' || input[cursor + 1] != b'u' {
                                return ParseOutcomes::default();
                            }
                            let (low, next) = match parse_hex_escape(input, cursor + 2) {
                                HexEscape::Complete(value, next) => (value, next),
                                HexEscape::Incomplete => return ParseOutcomes::incomplete(),
                                HexEscape::Invalid => return ParseOutcomes::default(),
                            };
                            if !(0xDC00..=0xDFFF).contains(&low) {
                                return ParseOutcomes::default();
                            }
                            cursor = next;
                        } else if (0xDC00..=0xDFFF).contains(&code_unit) {
                            return ParseOutcomes::default();
                        }
                    }
                    _ => return ParseOutcomes::default(),
                }
            }
            byte if byte < 0x20 => return ParseOutcomes::default(),
            _ => cursor += 1,
        }
    }
    ParseOutcomes::incomplete()
}

enum HexEscape {
    Complete(u16, usize),
    Incomplete,
    Invalid,
}

fn parse_hex_escape(input: &[u8], start: usize) -> HexEscape {
    let Some(digits) = input.get(start..) else {
        return HexEscape::Incomplete;
    };
    let mut value = 0_u16;
    for digit in digits.iter().take(4) {
        let nibble = match digit {
            b'0'..=b'9' => u16::from(*digit - b'0'),
            b'a'..=b'f' => u16::from(*digit - b'a' + 10),
            b'A'..=b'F' => u16::from(*digit - b'A' + 10),
            _ => return HexEscape::Invalid,
        };
        value = value * 16 + nibble;
    }
    if digits.len() < 4 {
        HexEscape::Incomplete
    } else {
        HexEscape::Complete(value, start + 4)
    }
}

fn parse_number(input: &[u8], position: usize, integer_only: bool) -> ParseOutcomes {
    if position == input.len() {
        return ParseOutcomes::incomplete();
    }
    let mut cursor = position;
    if input[cursor] == b'-' {
        cursor += 1;
        if cursor == input.len() {
            return ParseOutcomes::incomplete();
        }
    }

    match input.get(cursor) {
        Some(b'0') => cursor += 1,
        Some(b'1'..=b'9') => {
            cursor += 1;
            while input.get(cursor).is_some_and(u8::is_ascii_digit) {
                cursor += 1;
            }
        }
        _ => return ParseOutcomes::default(),
    }

    if integer_only {
        return ParseOutcomes::completed(cursor);
    }

    if input.get(cursor) == Some(&b'.') {
        cursor += 1;
        if cursor == input.len() {
            return ParseOutcomes::incomplete();
        }
        if !input[cursor].is_ascii_digit() {
            return ParseOutcomes::default();
        }
        while input.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
    }

    if input
        .get(cursor)
        .is_some_and(|byte| matches!(byte, b'e' | b'E'))
    {
        cursor += 1;
        if cursor == input.len() {
            return ParseOutcomes::incomplete();
        }
        if input
            .get(cursor)
            .is_some_and(|byte| matches!(byte, b'+' | b'-'))
        {
            cursor += 1;
            if cursor == input.len() {
                return ParseOutcomes::incomplete();
            }
        }
        if !input[cursor].is_ascii_digit() {
            return ParseOutcomes::default();
        }
        while input.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
    }

    ParseOutcomes::completed(cursor)
}

fn parse_object(
    schema: &ObjectSchema,
    input: &[u8],
    position: usize,
    depth: usize,
) -> ParseOutcomes {
    let opening = match_terminal(input, position, b"{");
    let mut outcomes = ParseOutcomes {
        complete: BTreeSet::new(),
        incomplete: opening.incomplete,
    };
    for after_open in opening.complete {
        let seen = vec![false; schema.properties.len()];
        outcomes.merge(parse_object_head(schema, input, after_open, &seen, depth));
    }
    outcomes
}

fn parse_object_head(
    schema: &ObjectSchema,
    input: &[u8],
    position: usize,
    seen: &[bool],
    depth: usize,
) -> ParseOutcomes {
    let position = skip_json_whitespace(input, position);
    if position == input.len() {
        return ParseOutcomes::incomplete();
    }
    if input[position] == b'}' {
        return if required_satisfied(schema, seen) {
            ParseOutcomes::completed(position + 1)
        } else {
            ParseOutcomes::default()
        };
    }
    parse_object_member(schema, input, position, seen, depth)
}

fn parse_object_member(
    schema: &ObjectSchema,
    input: &[u8],
    position: usize,
    seen: &[bool],
    depth: usize,
) -> ParseOutcomes {
    let mut outcomes = ParseOutcomes::default();
    for (property_index, property) in schema.properties.iter().enumerate() {
        if seen[property_index] {
            continue;
        }
        let key = match_terminal(input, position, &property.encoded_name);
        outcomes.incomplete |= key.incomplete;
        for key_end in key.complete {
            let colon_position = skip_json_whitespace(input, key_end);
            let colon = match_terminal(input, colon_position, b":");
            outcomes.incomplete |= colon.incomplete;
            for after_colon in colon.complete {
                let value_position = skip_json_whitespace(input, after_colon);
                let value = parse_node(&property.schema, input, value_position, depth + 1);
                outcomes.incomplete |= value.incomplete;
                for value_end in value.complete {
                    let mut next_seen = seen.to_vec();
                    next_seen[property_index] = true;
                    outcomes.merge(parse_object_after_value(
                        schema, input, value_end, &next_seen, depth,
                    ));
                }
            }
        }
    }
    outcomes
}

fn parse_object_after_value(
    schema: &ObjectSchema,
    input: &[u8],
    position: usize,
    seen: &[bool],
    depth: usize,
) -> ParseOutcomes {
    let position = skip_json_whitespace(input, position);
    if position == input.len() {
        return ParseOutcomes::incomplete();
    }
    match input[position] {
        b'}' if required_satisfied(schema, seen) => ParseOutcomes::completed(position + 1),
        b',' => {
            let next = skip_json_whitespace(input, position + 1);
            if next == input.len() {
                ParseOutcomes::incomplete()
            } else {
                parse_object_member(schema, input, next, seen, depth)
            }
        }
        _ => ParseOutcomes::default(),
    }
}

fn required_satisfied(schema: &ObjectSchema, seen: &[bool]) -> bool {
    schema
        .properties
        .iter()
        .zip(seen)
        .all(|(property, seen)| !property.required || *seen)
}

fn parse_array(items: &SchemaNode, input: &[u8], position: usize, depth: usize) -> ParseOutcomes {
    let opening = match_terminal(input, position, b"[");
    let mut outcomes = ParseOutcomes {
        complete: BTreeSet::new(),
        incomplete: opening.incomplete,
    };
    for after_open in opening.complete {
        let item_position = skip_json_whitespace(input, after_open);
        if item_position == input.len() {
            outcomes.incomplete = true;
            continue;
        }
        if input[item_position] == b']' {
            outcomes.complete.insert(item_position + 1);
            continue;
        }

        let mut frontier = VecDeque::from([(item_position, 0_usize)]);
        let mut visited = HashSet::new();
        while let Some((position, completed_items)) = frontier.pop_front() {
            if completed_items >= MAX_ARRAY_ITEMS || !visited.insert((position, completed_items)) {
                continue;
            }
            let item = parse_node(items, input, position, depth + 1);
            outcomes.incomplete |= item.incomplete;
            for item_end in item.complete {
                let delimiter = skip_json_whitespace(input, item_end);
                if delimiter == input.len() {
                    outcomes.incomplete = true;
                    continue;
                }
                match input[delimiter] {
                    b']' => {
                        outcomes.complete.insert(delimiter + 1);
                    }
                    b',' => {
                        let next = skip_json_whitespace(input, delimiter + 1);
                        if next == input.len() {
                            outcomes.incomplete = true;
                        } else {
                            frontier.push_back((next, completed_items + 1));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    outcomes
}

fn skip_json_whitespace(input: &[u8], mut position: usize) -> usize {
    while input
        .get(position)
        .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\r'))
    {
        position += 1;
    }
    position
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::json;

    use super::*;

    #[test]
    fn exact_object_literals_keep_sorted_grammar_and_detect_reordered_duplicates() {
        let schema = json!({"const": {"z": {"b": 1, "a": 2}, "a": 0}});
        let exact = compile_exact_values(schema.as_object().unwrap(), "root")
            .unwrap()
            .unwrap();
        assert_eq!(exact[0].bytes, br#"{"a":0,"z":{"a":2,"b":1}}"#);
        let schema = json!({"enum": [{"b": 1, "a": 2}, {"a": 2, "b": 1}]});
        assert!(compile_exact_values(schema.as_object().unwrap(), "root").is_err());
    }

    fn agent_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "ok": {"type": "boolean"},
                "tags": {
                    "type": "array",
                    "items": {"enum": ["rust", "metal"]}
                }
            },
            "required": ["answer", "ok", "tags"],
            "additionalProperties": false
        })
    }

    #[test]
    fn incrementally_accepts_a_schema_valid_object() -> Result<()> {
        let mut grammar = JsonSchemaGrammar::from_schema(&agent_schema())?;
        for fragment in [
            "{",
            "\"answer\"",
            ":",
            "\"yes\"",
            ",",
            "\"tags\"",
            ":[",
            "\"rust\"",
            "]",
            ",\"ok\":true",
            "}",
        ] {
            assert!(grammar.allows_token_text(fragment), "fragment {fragment:?}");
            grammar.commit_token_text(fragment)?;
        }
        assert_eq!(grammar.status(), GrammarStatus::Complete);
        assert_eq!(
            serde_json::from_str::<Value>(grammar.output_text())?,
            json!({"answer":"yes", "tags":["rust"], "ok":true})
        );
        grammar.finish()
    }

    #[test]
    fn object_properties_can_arrive_in_any_order_but_not_twice() -> Result<()> {
        let schema = json!({
            "type": "object",
            "properties": {"a":{"type":"integer"}, "b":{"type":"integer"}},
            "required": ["a", "b"],
            "additionalProperties": false
        });
        for value in [r#"{"b":2,"a":1}"#, r#"{"a":1,"b":2}"#] {
            let mut grammar = JsonSchemaGrammar::from_schema(&schema)?;
            grammar.commit_token_text(value)?;
            grammar.finish()?;
        }
        let grammar = JsonSchemaGrammar::from_schema(&schema)?;
        assert!(!grammar.allows_token_text(r#"{"a":1,"a""#));
        assert!(!grammar.allows_token_text(r#"{"unknown""#));
        Ok(())
    }

    #[test]
    fn required_and_additional_properties_are_enforced_before_completion() -> Result<()> {
        let grammar = JsonSchemaGrammar::from_schema(&agent_schema())?;
        assert!(!grammar.allows_token_text("{}"));
        assert!(!grammar.allows_token_text(r#"{"answer":"x"}"#));
        assert!(!grammar.allows_token_text(r#"{"unknown":null"#));
        Ok(())
    }

    #[test]
    fn enum_and_const_are_canonical_fail_closed_terminals() -> Result<()> {
        let schema = json!({
            "type": "object",
            "properties": {
                "kind": {"enum": ["call", "answer"]},
                "version": {"type":"integer", "const": 1}
            },
            "required": ["kind", "version"],
            "additionalProperties": false
        });
        let mut grammar = JsonSchemaGrammar::from_schema(&schema)?;
        assert!(grammar.allows_token_text(r#"{"kind":"cal"#));
        assert!(!grammar.allows_token_text(r#"{"kind":"other"#));
        grammar.commit_token_text(r#"{"kind":"call","version":1}"#)?;
        grammar.finish()?;

        let grammar = JsonSchemaGrammar::from_schema(&schema)?;
        assert!(!grammar.allows_token_text(r#"{"kind":"call","version":2"#));
        Ok(())
    }

    #[test]
    fn strict_json_string_escapes_and_numbers_are_incremental() -> Result<()> {
        let string_schema = json!({"type":"string"});
        let mut string = JsonSchemaGrammar::from_schema(&string_schema)?;
        for part in ["\"hello ", "\\uD83D", "\\uDE80", "\""] {
            assert!(string.allows_token_text(part));
            string.commit_token_text(part)?;
        }
        string.finish()?;
        assert_eq!(
            serde_json::from_str::<String>(string.output_text())?,
            "hello 🚀"
        );

        let number_schema = json!({"type":"number"});
        let mut number = JsonSchemaGrammar::from_schema(&number_schema)?;
        for part in ["-", "12", ".", "5", "e", "+", "2"] {
            assert!(number.allows_token_text(part), "number fragment {part:?}");
            number.commit_token_text(part)?;
        }
        number.finish()?;

        let integer = JsonSchemaGrammar::from_schema(&json!({"type":"integer"}))?;
        assert!(integer.allows_token_text("12"));
        assert!(!integer.allows_token_text("12.0"));
        Ok(())
    }

    #[test]
    fn malformed_json_prefixes_are_dead() -> Result<()> {
        let string = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
        assert!(!string.allows_token_text("\"bad\n"));
        assert!(!string.allows_token_text("\"\\x"));
        assert!(!string.allows_token_text("\"\\uX"));
        assert!(!string.allows_token_text("\"\\uD83D\\uX"));
        assert!(!string.allows_token_bytes(&[0xF0, 0x9F]));

        let number = JsonSchemaGrammar::from_schema(&json!({"type":"number"}))?;
        assert!(!number.allows_token_text("01"));
        assert!(!number.allows_token_text("1e+X"));
        Ok(())
    }

    #[test]
    fn annotations_are_ignored_and_unenforced_keywords_are_closed_mode_only() -> Result<()> {
        let annotated = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Shell",
            "description": "Run a command.",
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command.", "examples": ["ls"]},
                "timeout": {"type": "integer", "title": "Timeout", "default": 60, "$comment": "s"}
            },
            "required": ["command", "timeout"],
            "additionalProperties": false
        });
        for compile in [
            CompiledJsonSchema::compile,
            CompiledJsonSchema::compile_closed,
        ] {
            let schema = compile(&annotated)?;
            assert!(schema.accepts(&json!({"command": "ls", "timeout": 5})));
            assert!(!schema.accepts(&json!({"command": "ls", "timeout": "5"})));
        }

        let validated = json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "maxLength": 4, "pattern": "^/", "format": "uri"},
                "count": {"type": "integer", "minimum": 1, "maximum": 3, "exclusiveMinimum": 0, "multipleOf": 1},
                "ratio": {"type": "number", "exclusiveMaximum": true},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2}
            },
            "required": ["path"]
        });
        let closed = CompiledJsonSchema::compile_closed(&validated)?;
        // Shown to the model, not enforced: these values violate every bound.
        assert!(closed.accepts(&json!({"path": "too-long", "count": 9, "tags": []})));
        let mut strict = validated;
        strict["additionalProperties"] = json!(false);
        strict["required"] = json!(["path", "count", "ratio", "tags"]);
        let error = CompiledJsonSchema::compile(&strict).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported JSON Schema keyword")
        );

        for malformed in [
            json!({"type": "string", "description": 5}),
            json!({"type": "string", "examples": "ls"}),
            json!({"type": "string", "minLength": -1}),
            json!({"type": "integer", "minimum": "1"}),
            json!({"type": "boolean", "minLength": 1}),
        ] {
            assert!(
                CompiledJsonSchema::compile_closed(&malformed).is_err(),
                "{malformed}"
            );
        }
        // Composition and type arrays stay unsupported in both modes.
        for composed in [
            json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
            json!({"type": ["string", "null"]}),
            json!({"$ref": "#/$defs/x"}),
            json!({"type": "object", "properties": {}, "allOf": []}),
        ] {
            assert!(
                CompiledJsonSchema::compile(&composed).is_err(),
                "{composed}"
            );
            assert!(
                CompiledJsonSchema::compile_closed(&composed).is_err(),
                "{composed}"
            );
        }
        Ok(())
    }

    #[test]
    fn compiler_rejects_every_unsupported_or_open_ended_shape() {
        for schema in [
            json!(true),
            json!({}),
            json!({"type":["string", "null"]}),
            json!({"type":"string", "minLength":1}),
            json!({"type":"array"}),
            json!({"type":"object", "properties":{}}),
            json!({"type":"object", "properties":{}, "additionalProperties":true}),
            json!({
                "type":"object",
                "properties":{},
                "required":["missing"],
                "additionalProperties":false
            }),
            json!({"enum":[]}),
            json!({"type":"integer", "const":"wrong"}),
            json!({"enum":[1], "const":1}),
            json!({"$ref":"#/$defs/value"}),
        ] {
            assert!(
                CompiledJsonSchema::compile(&schema).is_err(),
                "schema should fail closed: {schema}"
            );
        }
    }

    fn shell_schema() -> Value {
        // RestLoop's declaration: non-strict, open, every property required.
        json!({
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        })
    }

    #[test]
    fn closed_compilation_narrows_non_strict_objects_to_declared_properties() -> Result<()> {
        assert!(CompiledJsonSchema::compile(&shell_schema()).is_err());
        let shell = CompiledJsonSchema::compile_closed(&shell_schema())?;
        assert!(shell.accepts(&json!({"command": "ls"})));
        assert!(!shell.accepts(&json!({"command": "ls", "cwd": "/"})));
        assert!(!shell.accepts(&json!({})));
        assert!(!shell.accepts(&json!({"command": 1})));

        let optional = CompiledJsonSchema::compile_closed(&json!({
            "type": "object",
            "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}},
            "required": ["path"],
            "additionalProperties": true
        }))?;
        assert!(optional.accepts(&json!({"path": "a"})));
        assert!(optional.accepts(&json!({"lines": 3, "path": "a"})));
        assert!(!optional.accepts(&json!({"lines": 3})));
        let mut grammar = JsonSchemaGrammar::new(optional);
        grammar.commit_token_text("{\"path\": \"a\"")?;
        assert!(grammar.allows_token_text("}"));
        assert!(grammar.allows_token_text(", \"lines\": 2}"));
        assert!(!grammar.allows_token_text(", \"other\""));

        for schema in [
            json!({"type": "object"}),
            json!({"type": "object", "properties": {}, "additionalProperties": {"type": "string"}}),
            json!({"type": "object", "properties": {"a": {"type": "string", "uniqueItems": true}}}),
            json!({"type": "object", "properties": {}, "required": ["missing"]}),
        ] {
            assert!(
                CompiledJsonSchema::compile_closed(&schema).is_err(),
                "schema should fail closed: {schema}"
            );
        }
        Ok(())
    }

    fn tool_calls() -> Result<CompiledToolCalls> {
        CompiledToolCalls::new([
            (
                "shell".to_owned(),
                CompiledJsonSchema::compile_closed(&shell_schema())?,
            ),
            (
                "shelf".to_owned(),
                CompiledJsonSchema::compile(&json!({
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": false
                }))?,
            ),
        ])
    }

    #[test]
    fn tool_call_grammar_accepts_one_canonical_envelope_for_a_declared_tool() -> Result<()> {
        let mut grammar = ToolCallGrammar::new(tool_calls()?);
        assert_eq!(grammar.status(), GrammarStatus::Viable);
        for fragment in [
            "\n{\"name\": \"",
            "shel",
            "l\", \"arguments\": ",
            "{\"command\": ",
            "\"ls -la\"}",
            "}\n",
            "</tool_call>",
        ] {
            assert_ne!(grammar.commit_token_text(fragment)?, GrammarStatus::Dead);
        }
        assert_eq!(grammar.status(), GrammarStatus::Complete);
        grammar.finish()?;
        assert!(!grammar.allows_token_text("\n"));
        assert!(!grammar.allows_token_text("<tool_call>"));
        assert_eq!(
            grammar.output_text(),
            "\n{\"name\": \"shell\", \"arguments\": {\"command\": \"ls -la\"}}\n</tool_call>"
        );

        let fresh = ToolCallGrammar::new(tool_calls()?);
        assert!(fresh.finish().is_err());
        for dead in [
            "{",
            "\n{\"name\":\"shell\"",
            "\n{\"name\": \"rm\"",
            "\n{\"name\": \"shell\", \"arguments\": {\"cmd\"",
            "\n{\"name\": \"shell\", \"arguments\": {\"command\": 1",
            "\n{\"name\": \"shell\", \"arguments\": {}}",
            "\n{\"name\": \"shelf\", \"arguments\": {\"command\"",
            "\n{\"name\": \"shelf\", \"arguments\": {}}\n</tool_call>\n",
        ] {
            assert!(!fresh.allows_token_text(dead), "{dead:?}");
        }
        assert!(fresh.allows_token_text("\n{\"name\": \"shelf\", \"arguments\": {}}\n</tool_"));
        Ok(())
    }

    #[test]
    fn tool_call_validation_rechecks_names_and_schemas() -> Result<()> {
        let calls = tool_calls()?;
        calls.validate_call("shell", &json!({"command": "ls"}))?;
        calls.validate_call("shelf", &json!({}))?;
        assert!(calls.validate_call("rm", &json!({})).is_err());
        assert!(
            calls
                .validate_call("shell", &json!({"command": 1}))
                .is_err()
        );
        assert!(calls.validate_call("shell", &json!(["ls"])).is_err());
        assert!(CompiledToolCalls::new([]).is_err());
        let schema = CompiledJsonSchema::compile_closed(&shell_schema())?;
        assert!(
            CompiledToolCalls::new([
                ("shell".to_owned(), schema.clone()),
                ("shell".to_owned(), schema)
            ])
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn candidate_logits_are_masked_before_sampling_and_eos_waits_for_completion() -> Result<()> {
        let mut grammar = JsonSchemaGrammar::from_schema(&json!({"type":"boolean"}))?;
        let candidates = [Some(b"t".to_vec()), Some(b"x".to_vec()), None, None];
        let mut logits = vec![9.0, 10.0, 11.0, 12.0];
        let mask = grammar.mask_candidate_logits(&mut logits, &[3], |token_id| {
            Ok(candidates[token_id as usize].clone())
        })?;
        assert_eq!(mask.allowed_tokens, 1);
        assert_eq!(
            logits,
            vec![9.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]
        );

        grammar.commit_token_text("true")?;
        let mut logits = vec![9.0, 10.0, 11.0, 12.0];
        let mask = grammar.mask_candidate_logits(&mut logits, &[3], |token_id| {
            Ok(candidates[token_id as usize].clone())
        })?;
        assert!(mask.can_end);
        assert!(logits[..3].iter().all(|logit| logit.is_infinite()));
        assert_eq!(logits[3].to_bits(), 12.0_f32.to_bits());
        Ok(())
    }

    #[test]
    fn complete_token_is_rejected_when_it_would_violate_schema() -> Result<()> {
        let grammar = JsonSchemaGrammar::from_schema(&json!({"enum":["yes", "no"]}))?;
        assert!(grammar.allows_token_text("\"y"));
        assert!(!grammar.allows_token_text("\"maybe\""));

        let array = JsonSchemaGrammar::from_schema(&json!({
            "type":"array",
            "items":{"type":"boolean"}
        }))?;
        assert!(array.allows_token_text("[true, fal"));
        assert!(!array.allows_token_text("[true, null"));
        Ok(())
    }

    #[test]
    fn output_limit_and_finish_are_fail_closed() -> Result<()> {
        let mut grammar =
            JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?.with_max_output_bytes(4)?;
        grammar.commit_token_text("\"a")?;
        assert!(grammar.finish().is_err());
        assert!(!grammar.allows_token_text("long"));
        grammar.commit_token_text("\"")?;
        assert_eq!(grammar.output_bytes(), b"\"a\"");
        grammar.finish()?;
        grammar.reset();
        assert_eq!(grammar.status(), GrammarStatus::Viable);
        Ok(())
    }
}
