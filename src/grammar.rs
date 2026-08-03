//! Fail-closed constrained decoding for strict JSON response formats.
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
//! it never silently degrades to unconstrained JSON. Object schemas require
//! `additionalProperties: false`. `enum` and `const` values are emitted using
//! their canonical `serde_json` spelling, which is a valid (and deterministic)
//! subset of the representations accepted by JSON Schema.

use std::{
    collections::{BTreeSet, HashSet, VecDeque},
    sync::Arc,
};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};
use tokenizers::{Tokenizer, tokenizer::step_decode_stream};

use crate::sampling::LogitsConstraint;

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
            root: compile_schema_node(schema, "$", 0)?,
        })
    }
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
        let mut next = self.decoder.clone();
        let fragment = step_decode_stream(
            self.tokenizer.as_ref(),
            vec![token_id],
            true,
            &mut next.ids,
            &mut next.prefix,
            &mut next.prefix_index,
        )
        .map_err(|error| {
            anyhow::anyhow!("tokenizer rejected candidate token {token_id}: {error}")
        })?;
        Ok((next, fragment))
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

fn compile_schema_node(schema: &Value, path: &str, depth: usize) -> Result<SchemaNode> {
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
        Some("object") => Some(compile_object_schema(object, path, depth)?),
        Some("array") => Some(compile_array_schema(object, path, depth)?),
        Some("string") => {
            reject_unknown_keywords(object, path, &["type", "enum", "const"])?;
            Some(SchemaKind::String)
        }
        Some("number") => {
            reject_unknown_keywords(object, path, &["type", "enum", "const"])?;
            Some(SchemaKind::Number)
        }
        Some("integer") => {
            reject_unknown_keywords(object, path, &["type", "enum", "const"])?;
            Some(SchemaKind::Integer)
        }
        Some("boolean") => {
            reject_unknown_keywords(object, path, &["type", "enum", "const"])?;
            Some(SchemaKind::Boolean)
        }
        Some("null") => {
            reject_unknown_keywords(object, path, &["type", "enum", "const"])?;
            Some(SchemaKind::Null)
        }
        Some(other) => bail!("{path}.type {other:?} is unsupported"),
        None => {
            reject_unknown_keywords(object, path, &["enum", "const"])?;
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
    )?;
    ensure!(
        object.get("additionalProperties") == Some(&Value::Bool(false)),
        "{path}.additionalProperties must be explicitly false"
    );
    let properties = object
        .get("properties")
        .and_then(Value::as_object)
        .with_context(|| format!("{path}.properties must be an object"))?;
    ensure!(
        properties.len() <= MAX_SCHEMA_PROPERTIES,
        "{path}.properties exceeds the limit of {MAX_SCHEMA_PROPERTIES}"
    );

    let required_array = object
        .get("required")
        .and_then(Value::as_array)
        .with_context(|| format!("{path}.required must list every property"))?;
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
        required.len() == properties.len(),
        "{path}.required must contain every property in strict mode"
    );

    let mut compiled = Vec::with_capacity(properties.len());
    for (name, property_schema) in properties {
        let property_path = format!("{path}.properties[{name:?}]");
        compiled.push(PropertySchema {
            name: name.clone(),
            encoded_name: serde_json::to_vec(name).context("serializing JSON property name")?,
            schema: compile_schema_node(property_schema, &property_path, depth + 1)?,
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
) -> Result<SchemaKind> {
    reject_unknown_keywords(object, path, &["type", "items", "enum", "const"])?;
    let items = object
        .get("items")
        .with_context(|| format!("{path}.items is required for an array schema"))?;
    Ok(SchemaKind::Array(Box::new(compile_schema_node(
        items,
        &format!("{path}.items"),
        depth + 1,
    )?)))
}

fn reject_unknown_keywords(
    object: &Map<String, Value>,
    path: &str,
    allowed: &[&str],
) -> Result<()> {
    for keyword in object.keys() {
        ensure!(
            allowed.contains(&keyword.as_str()),
            "unsupported JSON Schema keyword {path}.{keyword}"
        );
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
    for value in values {
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
