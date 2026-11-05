//! Strict `OpenAI` Chat Completions wire types.
//!
//! This module intentionally implements a small, honest subset of
//! `POST /v1/chat/completions`. Wire messages are validated before they are
//! converted into the runtime's [`ChatMessage`] type so unsupported `OpenAI`
//! features cannot be silently discarded.

use std::{collections::HashSet, fmt};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    chat::{ChatMessage, ChatRole, ChatTemplateOptions, validate_chat_messages},
    finish::FinishReason as GenerationFinishReason,
    grammar::CompiledJsonSchema,
    output::MAX_STOP_SEQUENCES,
};

pub const DEFAULT_MAX_COMPLETION_TOKENS: usize = 128;
pub const DEFAULT_TEMPERATURE: f64 = 1.0;
pub const DEFAULT_TOP_P: f64 = 1.0;

/// Field names of the raw completion response.
///
/// The list belongs to the wire contract, so it lives next to the other
/// protocol types. The serving layer re-exports it and the benchmark harness
/// compares both implementations against it.
pub const RAW_COMPLETION_RESPONSE_FIELDS: [&str; 22] = [
    "index",
    "content",
    "stop",
    "tokens_predicted",
    "tokens_evaluated",
    "truncated",
    "stop_type",
    "generation_settings/seed",
    "generation_settings/temperature",
    "generation_settings/top_k",
    "generation_settings/top_p",
    "generation_settings/repeat_penalty",
    "generation_settings/ignore_eos",
    "generation_settings/stream",
    "generation_settings/max_tokens",
    "generation_settings/n_predict",
    "generation_settings/n_keep",
    "timings/cache_n",
    "timings/prompt_n",
    "timings/prompt_ms",
    "timings/predicted_n",
    "timings/predicted_ms",
];

/// JSON request accepted by the Chat Completions endpoint.
///
/// Fields that are understood but unsupported remain in this wire type so
/// [`Self::validate`] can return an OpenAI-shaped error instead of ignoring
/// them. Truly unknown fields are rejected during JSON deserialization.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    /// Deprecated `OpenAI` alias retained for client compatibility.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub n: Option<usize>,

    // Agent request fields are parsed and validated into an explicit contract.
    // JSON-schema response formats, forced single calls and pinned-template
    // automatic choice use constrained generation. Generic detected formats
    // parse automatic calls and enforce `parallel_tool_calls` after generation.
    #[serde(default)]
    pub tools: Option<Vec<FunctionTool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub stop: Option<StopSequences>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    /// Template keyword options (vLLM/SGLang/llama.cpp convention). Only
    /// `enable_thinking` is understood; any other key is rejected.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, Value>>,
}

impl ChatCompletionRequest {
    /// Deserialize a request while preserving the endpoint's JSON error shape.
    pub fn from_json(body: &[u8]) -> Result<Self, ApiError> {
        serde_json::from_slice(body).map_err(|error| {
            ApiError::invalid_request(
                format!("Invalid JSON request: {error}"),
                None,
                Some("invalid_json".to_owned()),
            )
        })
    }

    /// Validate and normalize the supported subset for one served model.
    ///
    /// `default_max_completion_tokens` is supplied by the server so the
    /// protocol layer does not need to know a model's context policy.
    #[allow(clippy::too_many_lines)]
    pub fn validate(
        self,
        served_model: &str,
        default_max_completion_tokens: usize,
    ) -> Result<ValidatedChatCompletionRequest, ApiError> {
        self.validate_with_generic(served_model, default_max_completion_tokens, false)
    }

    /// Validate with support for multi-call history in generic templates.
    pub fn validate_with_generic(
        self,
        served_model: &str,
        default_max_completion_tokens: usize,
        generic: bool,
    ) -> Result<ValidatedChatCompletionRequest, ApiError> {
        if self.model.trim().is_empty() {
            return Err(invalid_value("model", "model must not be empty"));
        }
        if self.model != served_model {
            return Err(ApiError::invalid_request(
                format!("The model `{}` does not exist", self.model),
                Some("model".to_owned()),
                Some("model_not_found".to_owned()),
            ));
        }
        if default_max_completion_tokens == 0 {
            return Err(ApiError::server_error(
                "server default_max_completion_tokens must be greater than zero",
            ));
        }

        let n = self.n.unwrap_or(1);
        if n != 1 {
            return Err(unsupported_value(
                "n",
                "only n=1 is supported; multiple choices are not implemented",
            ));
        }
        if self.max_completion_tokens.is_some() && self.max_tokens.is_some() {
            return Err(ApiError::invalid_request(
                "max_completion_tokens and max_tokens cannot both be set",
                Some("max_completion_tokens".to_owned()),
                Some("mutually_exclusive".to_owned()),
            ));
        }
        let max_completion_tokens = self
            .max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(default_max_completion_tokens);
        if max_completion_tokens == 0 {
            return Err(invalid_value(
                "max_completion_tokens",
                "max_completion_tokens must be greater than zero",
            ));
        }

        let temperature = self.temperature;
        if temperature.is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value)) {
            return Err(invalid_value(
                "temperature",
                "temperature must be finite and in [0, 2]",
            ));
        }
        let top_p = self.top_p;
        if top_p.is_some_and(|value| !(value.is_finite() && 0.0 < value && value <= 1.0)) {
            return Err(invalid_value("top_p", "top_p must be finite and in (0, 1]"));
        }
        let seed = self
            .seed
            .map(|seed| {
                u64::try_from(seed).map_err(|_| invalid_value("seed", "seed must be non-negative"))
            })
            .transpose()?;

        for (name, value) in [
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ] {
            if value.is_some_and(|value| !value.is_finite() || !(-2.0..=2.0).contains(&value)) {
                return Err(invalid_value(name, "penalty must be finite and in [-2, 2]"));
            }
        }
        reject_unsupported_fields(&self)?;
        let stop = normalize_stop_sequences(self.stop)?;
        let include_usage = match self.stream_options {
            Some(_) if !self.stream => {
                return Err(invalid_value(
                    "stream_options",
                    "stream_options is only valid when stream=true",
                ));
            }
            Some(options) => {
                if options.include_obfuscation == Some(true) {
                    return Err(unsupported_value(
                        "stream_options.include_obfuscation",
                        "stream obfuscation is not implemented; set include_obfuscation=false",
                    ));
                }
                options.include_usage
            }
            None => false,
        };

        let chat_template = normalize_chat_template_kwargs(self.chat_template_kwargs)?;
        let messages = validate_messages(self.messages, generic)?;
        let response_format = self
            .response_format
            .map(validate_response_format)
            .transpose()?;
        if response_format.is_some() && messages.as_tool_aware().is_some() {
            return Err(invalid_value(
                "response_format",
                "response_format is not supported with tool-call history",
            ));
        }
        if response_format.is_some() && !stop.is_empty() {
            return Err(invalid_value(
                "stop",
                "stop cannot be combined with response_format.json_schema because it could terminate before the JSON value is complete",
            ));
        }
        let tools = validate_tools(self.tools.unwrap_or_default())?;
        let tool_choice = normalize_tool_choice(self.tool_choice, &tools)?;
        let parallel_tool_calls =
            normalize_parallel_tool_calls(self.parallel_tool_calls, &tools, &tool_choice)?;
        validate_tool_runtime_subset(
            &tools,
            &tool_choice,
            parallel_tool_calls,
            response_format.as_ref(),
            &stop,
        )?;

        Ok(ValidatedChatCompletionRequest {
            model: self.model,
            messages,
            stream: self.stream,
            include_usage,
            max_completion_tokens,
            temperature,
            top_p,
            presence_penalty: self.presence_penalty.map(|value| value as f32),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0) as f32,
            seed,
            stop,
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls,
            chat_template,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

/// A wire message whose role-dependent fields are normalized during
/// validation. Keeping `content` role-neutral here lets us return precise
/// errors for unsupported multimodal content and permit `null` only on an
/// assistant tool-call message.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<AssistantToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub function_call: Option<Value>,
    #[serde(default)]
    pub refusal: Option<String>,
    #[serde(default)]
    pub audio: Option<Value>,
    /// The reasoning an assistant turn produced. The Qwen3 and Qwen3.5
    /// templates render it as the turn's thinking block.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
    #[serde(default)]
    pub include_obfuscation: Option<bool>,
}

const MAX_FUNCTION_TOOLS: usize = 128;
const MAX_SCHEMA_NAME_CHARS: usize = 64;

/// One supported function tool declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FunctionTool {
    #[serde(rename = "type")]
    pub kind: FunctionToolKind,
    pub function: FunctionDefinition,
    #[serde(skip)]
    pub declaration: Value,
}

// Preserve the complete ordered wire declaration for native templates, while
// retaining typed validation of the fields used by the runtime.
impl<'de> Deserialize<'de> for FunctionTool {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            #[serde(rename = "type")]
            kind: FunctionToolKind,
            function: FunctionDefinition,
        }
        let declaration = Value::deserialize(deserializer)?;
        let fields: Fields =
            serde_json::from_value(declaration.clone()).map_err(serde::de::Error::custom)?;
        Ok(Self {
            kind: fields.kind,
            function: fields.function,
            declaration,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FunctionToolKind {
    #[serde(rename = "function")]
    Function,
}

impl FunctionTool {
    /// Compile the grammar that constrains this function's arguments.
    ///
    /// Annotations (`title`, `description`, `default`, `examples`) are kept
    /// in the rendered declaration and ignored by the grammar. A strict
    /// schema must otherwise be in the closed subset. A non-strict schema
    /// carries no conformance promise in the `OpenAI` contract: it is closed
    /// to its declared properties, and its length, pattern, format, numeric
    /// bound and item-count keywords are shown to the model but not enforced.
    pub fn compile_arguments_schema(&self) -> anyhow::Result<CompiledJsonSchema> {
        if self.function.strict {
            CompiledJsonSchema::compile(&self.function.parameters)
        } else {
            CompiledJsonSchema::compile_closed(&self.function.parameters)
        }
    }
}

/// One function declaration. Non-strict schemas are accepted when they can
/// be closed into the constrained-generation subset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub parameters: Value,
    #[serde(default)]
    pub strict: bool,
}

/// OpenAI-compatible tool selection, normalized after request validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(ToolChoiceMode),
    Function(NamedFunctionToolChoice),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedFunctionToolChoice {
    #[serde(rename = "type")]
    pub kind: FunctionToolKind,
    pub function: NamedFunction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedFunction {
    pub name: String,
}

/// Structured-output format supported by the fail-closed grammar engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResponseFormat {
    JsonSchema { json_schema: JsonSchemaFormat },
}

impl ResponseFormat {
    /// Compile the response grammar using the exact fail-closed subset used by
    /// constrained sampling. Validation calls this before engine admission;
    /// the server calls it again only to move an owned compiled specification
    /// across the worker boundary.
    pub fn compile_json_schema(&self) -> Result<CompiledJsonSchema, ApiError> {
        let Self::JsonSchema { json_schema } = self;
        CompiledJsonSchema::compile(&json_schema.schema).map_err(|error| {
            invalid_value(
                "response_format.json_schema.schema",
                format!("unsupported JSON schema for constrained generation: {error:#}"),
            )
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JsonSchemaFormat {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub schema: Value,
    #[serde(default)]
    pub strict: bool,
}

/// Function call used both in assistant history and future response objects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssistantToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: FunctionToolKind,
    pub function: AssistantFunctionCall,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssistantFunctionCall {
    pub name: String,
    /// `OpenAI` carries arguments as a JSON-encoded string on the wire.
    pub arguments: String,
}

/// One fully validated tool-aware history item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum AgentMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<AssistantToolCall>,
    },
    Tool {
        content: String,
        tool_call_id: String,
    },
}

/// A normal text history remains directly consumable by the current engine;
/// tool histories are preserved without lossy conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidatedMessages {
    Text(Vec<ChatMessage>),
    ToolAware(Vec<AgentMessage>),
}

impl ValidatedMessages {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Text(messages) => messages.len(),
            Self::ToolAware(messages) => messages.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn as_text(&self) -> Option<&[ChatMessage]> {
        match self {
            Self::Text(messages) => Some(messages),
            Self::ToolAware(_) => None,
        }
    }

    #[must_use]
    pub fn as_tool_aware(&self) -> Option<&[AgentMessage]> {
        match self {
            Self::Text(_) => None,
            Self::ToolAware(messages) => Some(messages),
        }
    }

    fn into_text(self) -> Option<Vec<ChatMessage>> {
        match self {
            Self::Text(messages) => Some(messages),
            Self::ToolAware(_) => None,
        }
    }
}

/// Request fields after protocol validation and runtime normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedChatCompletionRequest {
    pub model: String,
    pub messages: ValidatedMessages,
    pub stream: bool,
    pub include_usage: bool,
    pub max_completion_tokens: usize,
    /// `None` when the request omitted it; the server's sampling profile
    /// supplies the value.
    pub temperature: Option<f64>,
    /// `None` when the request omitted it; see `temperature`.
    pub top_p: Option<f64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: f32,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub response_format: Option<ResponseFormat>,
    pub tools: Vec<FunctionTool>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    /// Template options set by this request; unset options use server
    /// defaults, and the model adapter rejects any it cannot honor.
    pub chat_template: ChatTemplateOptions,
}

impl ValidatedChatCompletionRequest {
    /// Convert requests whose message semantics the current text runtime can
    /// faithfully preserve. A JSON response format constrains the generated
    /// text separately; tool declarations and tool-aware histories remain
    /// explicit runtime errors until their execution path is connected.
    pub fn into_text_runtime_messages(self) -> Result<Vec<ChatMessage>, ApiError> {
        if !matches!(self.tool_choice, ToolChoice::Mode(ToolChoiceMode::None)) {
            return Err(agent_runtime_unavailable("tools"));
        }
        self.messages
            .into_text()
            .ok_or_else(|| agent_runtime_unavailable("messages"))
    }

    /// Return the single tool selected by the validated forced-call subset.
    #[must_use]
    pub fn forced_tool(&self) -> Option<&FunctionTool> {
        match &self.tool_choice {
            ToolChoice::Mode(ToolChoiceMode::Required) => self.tools.first(),
            ToolChoice::Function(named) => self
                .tools
                .iter()
                .find(|tool| tool.function.name == named.function.name),
            ToolChoice::Mode(ToolChoiceMode::None | ToolChoiceMode::Auto) => None,
        }
    }
}

fn reject_unsupported_fields(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    if request.logprobs == Some(true) {
        return Err(unsupported_field("logprobs"));
    }
    if request.top_logprobs.is_some() {
        return Err(unsupported_field("top_logprobs"));
    }
    Ok(())
}

fn normalize_chat_template_kwargs(
    kwargs: Option<serde_json::Map<String, Value>>,
) -> Result<ChatTemplateOptions, ApiError> {
    let mut options = ChatTemplateOptions::default();
    for (key, value) in kwargs.unwrap_or_default() {
        let param = format!("chat_template_kwargs.{key}");
        match (key.as_str(), value) {
            ("enable_thinking", Value::Bool(enabled)) => options.enable_thinking = Some(enabled),
            ("enable_thinking", _) => {
                return Err(invalid_value(param, "enable_thinking must be a boolean"));
            }
            _ => {
                return Err(unsupported_value(
                    param,
                    "only chat_template_kwargs.enable_thinking is supported",
                ));
            }
        }
    }
    Ok(options)
}

fn validate_response_format(format: ResponseFormat) -> Result<ResponseFormat, ApiError> {
    match &format {
        ResponseFormat::JsonSchema { json_schema } => {
            validate_schema_name("response_format.json_schema.name", &json_schema.name)?;
            if !json_schema.strict {
                return Err(invalid_value(
                    "response_format.json_schema.strict",
                    "only strict=true JSON schemas are supported",
                ));
            }
            validate_strict_object_schema(
                &json_schema.schema,
                "response_format.json_schema.schema",
            )?;
            format.compile_json_schema()?;
        }
    }
    Ok(format)
}

fn validate_tools(tools: Vec<FunctionTool>) -> Result<Vec<FunctionTool>, ApiError> {
    if tools.len() > MAX_FUNCTION_TOOLS {
        return Err(invalid_value(
            "tools",
            format!("tools supports at most {MAX_FUNCTION_TOOLS} functions"),
        ));
    }

    let mut names = HashSet::with_capacity(tools.len());
    for (index, tool) in tools.iter().enumerate() {
        let base = format!("tools.{index}.function");
        validate_schema_name(format!("{base}.name"), &tool.function.name)?;
        if !names.insert(tool.function.name.as_str()) {
            return Err(invalid_value(
                format!("{base}.name"),
                format!("duplicate tool function `{}`", tool.function.name),
            ));
        }
        if tool.function.strict {
            validate_strict_object_schema(
                &tool.function.parameters,
                &format!("{base}.parameters"),
            )?;
        } else if tool.function.parameters.get("type").and_then(Value::as_str) != Some("object") {
            return Err(invalid_value(
                format!("{base}.parameters.type"),
                "function parameters must have an object root",
            ));
        }
        tool.compile_arguments_schema().map_err(|error| {
            invalid_value(
                format!("{base}.parameters"),
                format!("unsupported JSON schema for constrained tool arguments: {error:#}"),
            )
        })?;
    }
    Ok(tools)
}

fn normalize_tool_choice(
    choice: Option<ToolChoice>,
    tools: &[FunctionTool],
) -> Result<ToolChoice, ApiError> {
    let default_choice = ToolChoice::Mode(if tools.is_empty() {
        ToolChoiceMode::None
    } else {
        ToolChoiceMode::Auto
    });
    let choice = choice.unwrap_or(default_choice);

    match &choice {
        ToolChoice::Mode(ToolChoiceMode::Auto | ToolChoiceMode::Required) if tools.is_empty() => {
            return Err(invalid_value(
                "tool_choice",
                "tool_choice requires at least one declared tool",
            ));
        }
        ToolChoice::Mode(_) => {}
        ToolChoice::Function(named) => {
            validate_schema_name("tool_choice.function.name", &named.function.name)?;
            if !tools
                .iter()
                .any(|tool| tool.function.name == named.function.name)
            {
                return Err(invalid_value(
                    "tool_choice.function.name",
                    format!(
                        "forced tool `{}` is not present in tools",
                        named.function.name
                    ),
                ));
            }
        }
    }
    Ok(choice)
}

/// `parallel_tool_calls` grants permission for several calls in one turn; it
/// never obliges more than one. Automatic choice accepts either value (`true`
/// is the `OpenAI` default): pinned constrained paths emit at most one call,
/// while generic detected formats enforce this permission on parsed calls.
/// Forced and `required` calls keep their explicit single-call contract and
/// require `false`.
fn normalize_parallel_tool_calls(
    requested: Option<bool>,
    tools: &[FunctionTool],
    choice: &ToolChoice,
) -> Result<bool, ApiError> {
    let tools_enabled =
        !tools.is_empty() && !matches!(choice, ToolChoice::Mode(ToolChoiceMode::None));
    if requested == Some(true) && !tools_enabled {
        return Err(invalid_value(
            "parallel_tool_calls",
            "parallel_tool_calls=true requires enabled tools",
        ));
    }
    Ok(tools_enabled && requested.unwrap_or(true))
}

fn validate_tool_runtime_subset(
    tools: &[FunctionTool],
    choice: &ToolChoice,
    parallel_tool_calls: bool,
    response_format: Option<&ResponseFormat>,
    stop: &[String],
) -> Result<(), ApiError> {
    if tools.is_empty() {
        return Ok(());
    }
    if response_format.is_some() {
        return Err(invalid_value(
            "response_format",
            "response_format cannot be combined with declared function tools",
        ));
    }
    if matches!(choice, ToolChoice::Mode(ToolChoiceMode::None)) {
        return Ok(());
    }
    if !stop.is_empty() {
        return Err(invalid_value(
            "stop",
            "stop cannot be combined with enabled tool calls because it could terminate before a call's arguments JSON is complete",
        ));
    }
    match choice {
        // Pinned constrained paths stop after one call. Generic detected
        // formats may return several calls when `parallel_tool_calls` permits.
        ToolChoice::Mode(ToolChoiceMode::Auto) => return Ok(()),
        ToolChoice::Mode(ToolChoiceMode::Required) if tools.len() != 1 => {
            return Err(unsupported_value(
                "tool_choice",
                "tool_choice=required is supported only with exactly one declared tool",
            ));
        }
        ToolChoice::Mode(ToolChoiceMode::Required | ToolChoiceMode::None)
        | ToolChoice::Function(_) => {}
    }
    if parallel_tool_calls {
        return Err(unsupported_value(
            "parallel_tool_calls",
            "the native tool runtime supports exactly one call; set parallel_tool_calls=false",
        ));
    }
    Ok(())
}

fn validate_schema_name(param: impl Into<String>, name: &str) -> Result<(), ApiError> {
    let param = param.into();
    let valid = !name.is_empty()
        && name.chars().count() <= MAX_SCHEMA_NAME_CHARS
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'));
    if !valid {
        return Err(invalid_value(
            param,
            "name must be 1-64 ASCII letters, digits, underscores, or hyphens",
        ));
    }
    Ok(())
}

fn validate_strict_object_schema(schema: &Value, param: &str) -> Result<(), ApiError> {
    let Some(root) = schema.as_object() else {
        return Err(invalid_value(param, "schema must be a JSON object"));
    };
    if root.get("type").and_then(Value::as_str) != Some("object") {
        return Err(invalid_value(
            format!("{param}.type"),
            "strict schemas must have an object root",
        ));
    }
    validate_strict_schema_node(schema, param)
}

fn validate_strict_schema_node(schema: &Value, param: &str) -> Result<(), ApiError> {
    let Some(object) = schema.as_object() else {
        return Err(invalid_value(param, "schema nodes must be JSON objects"));
    };

    if object.get("type").and_then(Value::as_str) == Some("object") {
        if object.get("additionalProperties") != Some(&Value::Bool(false)) {
            return Err(invalid_value(
                format!("{param}.additionalProperties"),
                "strict object schemas require additionalProperties=false",
            ));
        }
        let properties = object
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                invalid_value(
                    format!("{param}.properties"),
                    "strict object schemas require a properties object",
                )
            })?;
        let required = object
            .get("required")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                invalid_value(
                    format!("{param}.required"),
                    "strict object schemas require every property to be required",
                )
            })?;
        let mut required_names = HashSet::with_capacity(required.len());
        for name in required {
            let Some(name) = name.as_str() else {
                return Err(invalid_value(
                    format!("{param}.required"),
                    "required entries must be strings",
                ));
            };
            if !required_names.insert(name) {
                return Err(invalid_value(
                    format!("{param}.required"),
                    format!("required contains duplicate property `{name}`"),
                ));
            }
        }
        let property_names = properties
            .keys()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        if required_names != property_names {
            return Err(invalid_value(
                format!("{param}.required"),
                "strict object schemas require every property exactly once",
            ));
        }
        for (name, property) in properties {
            validate_strict_schema_node(property, &format!("{param}.properties.{name}"))?;
        }
    }

    if let Some(items) = object.get("items") {
        validate_strict_schema_node(items, &format!("{param}.items"))?;
    }
    for keyword in ["$defs", "definitions"] {
        if let Some(definitions) = object.get(keyword) {
            let Some(definitions) = definitions.as_object() else {
                return Err(invalid_value(
                    format!("{param}.{keyword}"),
                    format!("{keyword} must be an object"),
                ));
            };
            for (name, definition) in definitions {
                validate_strict_schema_node(definition, &format!("{param}.{keyword}.{name}"))?;
            }
        }
    }
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(branches) = object.get(keyword) {
            let Some(branches) = branches.as_array() else {
                return Err(invalid_value(
                    format!("{param}.{keyword}"),
                    format!("{keyword} must be an array"),
                ));
            };
            for (index, branch) in branches.iter().enumerate() {
                validate_strict_schema_node(branch, &format!("{param}.{keyword}.{index}"))?;
            }
        }
    }
    Ok(())
}

fn normalize_stop_sequences(stop: Option<StopSequences>) -> Result<Vec<String>, ApiError> {
    let stops = match stop {
        None => Vec::new(),
        Some(StopSequences::One(stop)) => vec![stop],
        Some(StopSequences::Many(stops)) => stops,
    };
    if stops.len() > MAX_STOP_SEQUENCES {
        return Err(invalid_value(
            "stop",
            format!("stop supports at most {MAX_STOP_SEQUENCES} sequences"),
        ));
    }
    if stops.iter().any(String::is_empty) {
        return Err(invalid_value("stop", "stop sequences must not be empty"));
    }
    Ok(stops)
}

fn validate_messages(
    messages: Vec<RequestMessage>,
    generic: bool,
) -> Result<ValidatedMessages, ApiError> {
    if messages.is_empty() {
        return Err(invalid_value(
            "messages",
            "messages must contain at least one message",
        ));
    }

    let tool_aware = messages.iter().any(|message| {
        message.role == "tool"
            || message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
    });
    if tool_aware {
        validate_tool_aware_messages(messages, generic)
    } else {
        validate_text_messages(messages)
    }
}

fn validate_text_messages(messages: Vec<RequestMessage>) -> Result<ValidatedMessages, ApiError> {
    let mut validated = Vec::with_capacity(messages.len());
    for (index, mut message) in messages.into_iter().enumerate() {
        let reasoning = message.reasoning_content.take();
        let role_param = format!("messages.{index}.role");
        let role = match message.role.as_str() {
            "system" => ChatRole::System,
            "user" => ChatRole::User,
            "assistant" => ChatRole::Assistant,
            "tool" | "function" => {
                return Err(unsupported_value(
                    &role_param,
                    "tool and function messages are not supported",
                ));
            }
            "developer" => {
                return Err(unsupported_value(
                    &role_param,
                    "developer messages are not supported by the current chat templates",
                ));
            }
            role => {
                return Err(invalid_value(
                    &role_param,
                    format!("unsupported message role `{role}`"),
                ));
            }
        };

        if message.name.is_some() {
            return Err(unsupported_field(format!("messages.{index}.name")));
        }
        if message.tool_calls.is_some() {
            return Err(unsupported_field(format!("messages.{index}.tool_calls")));
        }
        if message.tool_call_id.is_some() {
            return Err(unsupported_field(format!("messages.{index}.tool_call_id")));
        }
        if message.function_call.is_some() {
            return Err(unsupported_field(format!("messages.{index}.function_call")));
        }
        if message.refusal.is_some() {
            return Err(unsupported_field(format!("messages.{index}.refusal")));
        }
        if message.audio.is_some() {
            return Err(unsupported_field(format!("messages.{index}.audio")));
        }

        if role != ChatRole::Assistant {
            reject_reasoning_content(reasoning.as_deref(), index)?;
        }
        let Some(Value::String(content)) = message.content else {
            return Err(unsupported_value(
                format!("messages.{index}.content"),
                "only string message content is supported; multimodal content is not implemented",
            ));
        };
        let content = if role == ChatRole::Assistant {
            assistant_content_with_reasoning(Some(content), reasoning.as_deref())
                .unwrap_or_default()
        } else {
            content
        };
        validated.push(ChatMessage::new(role, content));
    }

    validate_chat_messages(&validated).map_err(|error| {
        invalid_value(
            "messages",
            format!("messages do not form a supported conversation: {error}"),
        )
    })?;
    Ok(ValidatedMessages::Text(validated))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolConversationState {
    NeedUser,
    NeedAssistant,
    NeedToolResponses,
    MayGenerateAfterTools,
}

// Keeping the role state machine in one function makes every transition and
// terminal condition auditable together.
#[allow(clippy::too_many_lines)]
fn validate_tool_aware_messages(
    messages: Vec<RequestMessage>,
    generic: bool,
) -> Result<ValidatedMessages, ApiError> {
    let mut validated = Vec::with_capacity(messages.len());
    let mut state = ToolConversationState::NeedUser;
    let mut pending_tool_calls = HashSet::<String>::new();
    let mut all_tool_call_ids = HashSet::<String>::new();

    for (index, mut message) in messages.into_iter().enumerate() {
        let reasoning = message.reasoning_content.take();
        reject_common_message_extensions(&message, index)?;
        let role_param = format!("messages.{index}.role");
        match message.role.as_str() {
            "system" => {
                reject_reasoning_content(reasoning.as_deref(), index)?;
                if index != 0 || state != ToolConversationState::NeedUser {
                    return Err(invalid_value(
                        role_param,
                        "a system message is only allowed at the beginning",
                    ));
                }
                reject_tool_fields(&message, index)?;
                let content = require_string_content(message.content, index)?;
                validated.push(AgentMessage::System { content });
            }
            "user" => {
                reject_reasoning_content(reasoning.as_deref(), index)?;
                if state != ToolConversationState::NeedUser {
                    return Err(invalid_value(
                        role_param,
                        "a user message must begin a new assistant turn",
                    ));
                }
                reject_tool_fields(&message, index)?;
                let content = require_string_content(message.content, index)?;
                validated.push(AgentMessage::User { content });
                state = ToolConversationState::NeedAssistant;
            }
            "assistant" => {
                if !matches!(
                    state,
                    ToolConversationState::NeedAssistant
                        | ToolConversationState::MayGenerateAfterTools
                ) {
                    return Err(invalid_value(
                        role_param,
                        "an assistant message must follow a user turn or a complete tool batch",
                    ));
                }
                if message.tool_call_id.is_some() {
                    return Err(invalid_value(
                        format!("messages.{index}.tool_call_id"),
                        "tool_call_id is only valid on tool messages",
                    ));
                }
                let tool_calls_were_present = message.tool_calls.is_some();
                let content = optional_assistant_content(message.content, index)?;
                let content = assistant_content_with_reasoning(content, reasoning.as_deref());
                let tool_calls = message.tool_calls.unwrap_or_default();
                if tool_calls_were_present && tool_calls.is_empty() {
                    return Err(invalid_value(
                        format!("messages.{index}.tool_calls"),
                        "tool_calls must contain at least one call when provided",
                    ));
                }
                if content.is_none() && tool_calls.is_empty() {
                    return Err(invalid_value(
                        format!("messages.{index}.content"),
                        "assistant messages require content or at least one tool call",
                    ));
                }
                if tool_calls.is_empty() {
                    validated.push(AgentMessage::Assistant {
                        content,
                        tool_calls,
                    });
                    state = ToolConversationState::NeedUser;
                    continue;
                }
                if !generic && tool_calls.len() != 1 {
                    return Err(unsupported_value(
                        format!("messages.{index}.tool_calls"),
                        "the native tool history subset supports exactly one call per assistant turn",
                    ));
                }
                // Content before a call (visible text or reasoning) is kept;
                // clients that echo an empty string mean no content.
                let content = content.filter(|content| !content.is_empty());
                pending_tool_calls.clear();
                for (call_index, call) in tool_calls.iter().enumerate() {
                    validate_assistant_tool_call(call, index, call_index)?;
                    if !all_tool_call_ids.insert(call.id.clone()) {
                        return Err(invalid_value(
                            format!("messages.{index}.tool_calls.{call_index}.id"),
                            format!("duplicate tool call id `{}`", call.id),
                        ));
                    }
                    pending_tool_calls.insert(call.id.clone());
                }
                validated.push(AgentMessage::Assistant {
                    content,
                    tool_calls,
                });
                state = ToolConversationState::NeedToolResponses;
            }
            "tool" => {
                reject_reasoning_content(reasoning.as_deref(), index)?;
                if state != ToolConversationState::NeedToolResponses {
                    return Err(invalid_value(
                        role_param,
                        "a tool message must answer the preceding assistant tool calls",
                    ));
                }
                if message.tool_calls.is_some() {
                    return Err(invalid_value(
                        format!("messages.{index}.tool_calls"),
                        "tool messages cannot contain tool_calls",
                    ));
                }
                let tool_call_id = message.tool_call_id.ok_or_else(|| {
                    invalid_value(
                        format!("messages.{index}.tool_call_id"),
                        "tool messages require tool_call_id",
                    )
                })?;
                if tool_call_id.is_empty() || !pending_tool_calls.remove(&tool_call_id) {
                    return Err(invalid_value(
                        format!("messages.{index}.tool_call_id"),
                        "tool_call_id must reference one unresolved preceding tool call",
                    ));
                }
                let content = require_string_content(message.content, index)?;
                validated.push(AgentMessage::Tool {
                    content,
                    tool_call_id,
                });
                if pending_tool_calls.is_empty() {
                    state = ToolConversationState::MayGenerateAfterTools;
                }
            }
            "function" => {
                return Err(unsupported_value(
                    role_param,
                    "legacy function messages are not supported; use role=tool",
                ));
            }
            "developer" => {
                return Err(unsupported_value(
                    role_param,
                    "developer messages are not supported by the current chat templates",
                ));
            }
            role => {
                return Err(invalid_value(
                    role_param,
                    format!("unsupported message role `{role}`"),
                ));
            }
        }
    }

    match state {
        ToolConversationState::NeedAssistant | ToolConversationState::MayGenerateAfterTools => {
            Ok(ValidatedMessages::ToolAware(validated))
        }
        ToolConversationState::NeedToolResponses => Err(invalid_value(
            "messages",
            "every assistant tool call must have one tool response",
        )),
        ToolConversationState::NeedUser => Err(invalid_value(
            "messages",
            "messages must end with a user turn or a complete tool response batch",
        )),
    }
}

/// Render an assistant turn that carries its reasoning separately.
///
/// The Qwen3 and Qwen3.5 templates read `message.reasoning_content` and
/// render the turn exactly as if its content were the thinking block
/// followed by the visible answer.
fn assistant_content_with_reasoning(
    content: Option<String>,
    reasoning: Option<&str>,
) -> Option<String> {
    match reasoning {
        Some(reasoning) => Some(format!(
            "<think>\n{reasoning}\n</think>\n\n{}",
            content.unwrap_or_default()
        )),
        _ => content,
    }
}

/// Reject reasoning history on a message that cannot carry it.
fn reject_reasoning_content(reasoning: Option<&str>, index: usize) -> Result<(), ApiError> {
    if reasoning.is_some_and(|reasoning| !reasoning.is_empty()) {
        return Err(invalid_value(
            format!("messages.{index}.reasoning_content"),
            "reasoning_content is only valid on assistant messages",
        ));
    }
    Ok(())
}

fn reject_common_message_extensions(
    message: &RequestMessage,
    index: usize,
) -> Result<(), ApiError> {
    for (present, field) in [
        (message.name.is_some(), "name"),
        (message.function_call.is_some(), "function_call"),
        (message.refusal.is_some(), "refusal"),
        (message.audio.is_some(), "audio"),
    ] {
        if present {
            return Err(unsupported_field(format!("messages.{index}.{field}")));
        }
    }
    Ok(())
}

fn reject_tool_fields(message: &RequestMessage, index: usize) -> Result<(), ApiError> {
    if message.tool_calls.is_some() {
        return Err(invalid_value(
            format!("messages.{index}.tool_calls"),
            "tool_calls is only valid on assistant messages",
        ));
    }
    if message.tool_call_id.is_some() {
        return Err(invalid_value(
            format!("messages.{index}.tool_call_id"),
            "tool_call_id is only valid on tool messages",
        ));
    }
    Ok(())
}

fn require_string_content(content: Option<Value>, index: usize) -> Result<String, ApiError> {
    match content {
        Some(Value::String(content)) => Ok(content),
        _ => Err(unsupported_value(
            format!("messages.{index}.content"),
            "only string message content is supported; multimodal content is not implemented",
        )),
    }
}

fn optional_assistant_content(
    content: Option<Value>,
    index: usize,
) -> Result<Option<String>, ApiError> {
    match content {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(content)) => Ok(Some(content)),
        Some(_) => Err(unsupported_value(
            format!("messages.{index}.content"),
            "assistant content must be a string or null",
        )),
    }
}

fn validate_assistant_tool_call(
    call: &AssistantToolCall,
    message_index: usize,
    call_index: usize,
) -> Result<(), ApiError> {
    let base = format!("messages.{message_index}.tool_calls.{call_index}");
    if call.id.is_empty() {
        return Err(invalid_value(
            format!("{base}.id"),
            "tool call id must not be empty",
        ));
    }
    validate_schema_name(format!("{base}.function.name"), &call.function.name)?;
    let arguments: Value = serde_json::from_str(&call.function.arguments).map_err(|error| {
        invalid_value(
            format!("{base}.function.arguments"),
            format!("tool arguments must contain valid JSON: {error}"),
        )
    })?;
    if !arguments.is_object() {
        return Err(invalid_value(
            format!("{base}.function.arguments"),
            "tool arguments must encode a JSON object",
        ));
    }
    Ok(())
}

fn invalid_value(param: impl Into<String>, message: impl Into<String>) -> ApiError {
    ApiError::invalid_request(
        message,
        Some(param.into()),
        Some("invalid_value".to_owned()),
    )
}

fn unsupported_field(param: impl Into<String>) -> ApiError {
    let param = param.into();
    ApiError::invalid_request(
        format!("The `{param}` field is not supported"),
        Some(param),
        Some("unsupported_parameter".to_owned()),
    )
}

fn unsupported_value(param: impl Into<String>, message: impl Into<String>) -> ApiError {
    ApiError::invalid_request(
        message,
        Some(param.into()),
        Some("unsupported_value".to_owned()),
    )
}

fn agent_runtime_unavailable(param: impl Into<String>) -> ApiError {
    let param = param.into();
    ApiError::invalid_request(
        format!(
            "The `{param}` contract requires native agent rendering and cannot be lowered into text-only messages"
        ),
        Some(param),
        Some("agent_runtime_required".to_owned()),
    )
}

/// HTTP-independent classification used by the server to choose a status code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiErrorStatus {
    BadRequest,
    InternalServerError,
}

/// OpenAI-compatible error object nested under the top-level `error` key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ApiError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
    #[serde(skip)]
    pub status: ApiErrorStatus,
}

impl ApiError {
    #[must_use]
    pub fn invalid_request(
        message: impl Into<String>,
        param: Option<String>,
        code: Option<String>,
    ) -> Self {
        Self {
            message: message.into(),
            error_type: "invalid_request_error".to_owned(),
            param,
            code,
            status: ApiErrorStatus::BadRequest,
        }
    }

    #[must_use]
    pub fn server_error(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: "server_error".to_owned(),
            param: None,
            code: Some("internal_error".to_owned()),
            status: ApiErrorStatus::InternalServerError,
        }
    }

    #[must_use]
    pub fn into_response(self) -> ApiErrorResponse {
        ApiErrorResponse { error: self }
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ApiError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ApiErrorResponse {
    pub error: ApiError,
}

impl From<ApiError> for ApiErrorResponse {
    fn from(error: ApiError) -> Self {
        Self { error }
    }
}

/// Finish reasons exposed by the supported `OpenAI` response subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatFinishReason {
    Stop,
    Length,
    ToolCalls,
}

impl ChatFinishReason {
    /// Map a completed runtime request to an `OpenAI` finish reason.
    ///
    /// Cooperative cancellation has no faithful `OpenAI` finish reason. A
    /// disconnected stream therefore emits no synthetic terminal chunk.
    #[must_use]
    pub const fn from_generation(reason: GenerationFinishReason) -> Option<Self> {
        match reason {
            GenerationFinishReason::Stop => Some(Self::Stop),
            GenerationFinishReason::Length => Some(Self::Length),
            GenerationFinishReason::Cancelled => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ResponseRole {
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub prompt_tokens_details: PromptTokenDetails,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PromptTokenDetails {
    pub cached_tokens: usize,
}

impl TokenUsage {
    #[must_use]
    pub const fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self::with_cached_tokens(prompt_tokens, completion_tokens, 0)
    }

    #[must_use]
    pub const fn with_cached_tokens(
        prompt_tokens: usize,
        completion_tokens: usize,
        cached_tokens: usize,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: PromptTokenDetails { cached_tokens },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: TokenUsage,
}

impl ChatCompletionResponse {
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        content: impl Into<String>,
        reasoning: Option<String>,
        finish_reason: ChatFinishReason,
        usage: TokenUsage,
    ) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion".to_owned(),
            created,
            model: model.into(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ResponseMessage {
                    role: ResponseRole::Assistant,
                    content: Some(content.into()),
                    reasoning_content: reasoning,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason,
            }],
            usage,
        }
    }

    /// A turn that ends with one call. `content` is the visible text the
    /// model produced before the call, or `None` when there was none.
    #[must_use]
    pub fn tool_call(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        content: Option<String>,
        reasoning: Option<String>,
        call: AssistantToolCall,
        usage: TokenUsage,
    ) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion".to_owned(),
            created,
            model: model.into(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ResponseMessage {
                    role: ResponseRole::Assistant,
                    content,
                    reasoning_content: reasoning,
                    tool_calls: Some(vec![call]),
                },
                logprobs: None,
                finish_reason: ChatFinishReason::ToolCalls,
            }],
            usage,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ResponseMessage,
    pub logprobs: Option<Value>,
    pub finish_reason: ChatFinishReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResponseMessage {
    pub role: ResponseRole,
    pub content: Option<String>,
    /// The reasoning the model produced before its answer. The key is absent
    /// when the turn has no reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<AssistantToolCall>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// `null` on ordinary chunks and populated only by the optional final
    /// usage chunk requested with `stream_options.include_usage`.
    pub usage: Option<TokenUsage>,
}

impl ChatCompletionChunk {
    #[must_use]
    pub fn role(id: impl Into<String>, created: u64, model: impl Into<String>) -> Self {
        Self::single_choice(
            id,
            created,
            model,
            ChunkDelta {
                role: Some(ResponseRole::Assistant),
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            None,
        )
    }

    #[must_use]
    pub fn content(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self::single_choice(
            id,
            created,
            model,
            ChunkDelta {
                role: None,
                content: Some(content.into()),
                reasoning_content: None,
                tool_calls: None,
            },
            None,
        )
    }

    /// A chunk that carries only the model's reasoning.
    #[must_use]
    pub fn reasoning(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        reasoning: impl Into<String>,
    ) -> Self {
        Self::single_choice(
            id,
            created,
            model,
            ChunkDelta {
                role: None,
                content: None,
                reasoning_content: Some(reasoning.into()),
                tool_calls: None,
            },
            None,
        )
    }

    #[must_use]
    pub fn finished(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        finish_reason: ChatFinishReason,
    ) -> Self {
        Self::single_choice(
            id,
            created,
            model,
            ChunkDelta::default(),
            Some(finish_reason),
        )
    }

    #[must_use]
    pub fn usage(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        usage: TokenUsage,
    ) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".to_owned(),
            created,
            model: model.into(),
            choices: Vec::new(),
            usage: Some(usage),
        }
    }

    #[must_use]
    pub fn tool_call_header(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self::single_choice(
            id,
            created,
            model,
            ChunkDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![ChunkToolCall {
                    index: 0,
                    id: Some(call_id.into()),
                    kind: Some(FunctionToolKind::Function),
                    function: ChunkFunctionCall {
                        name: Some(name.into()),
                        arguments: Some(String::new()),
                    },
                }]),
            },
            None,
        )
    }

    #[must_use]
    pub fn tool_arguments(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::single_choice(
            id,
            created,
            model,
            ChunkDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![ChunkToolCall {
                    index: 0,
                    id: None,
                    kind: None,
                    function: ChunkFunctionCall {
                        name: None,
                        arguments: Some(arguments.into()),
                    },
                }]),
            },
            None,
        )
    }

    fn single_choice(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        delta: ChunkDelta,
        finish_reason: Option<ChatFinishReason>,
    ) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".to_owned(),
            created,
            model: model.into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                logprobs: None,
                finish_reason,
            }],
            usage: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub logprobs: Option<Value>,
    pub finish_reason: Option<ChatFinishReason>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<ResponseRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// The reasoning the model produced before its answer. The key is absent
    /// when the chunk carries no reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChunkToolCall {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<FunctionToolKind>,
    pub function: ChunkFunctionCall,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChunkFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    const MODEL: &str = "test-model";

    fn parse_and_validate(value: Value) -> Result<ValidatedChatCompletionRequest, ApiError> {
        ChatCompletionRequest::from_json(&serde_json::to_vec(&value).unwrap())?
            .validate(MODEL, DEFAULT_MAX_COMPLETION_TOKENS)
    }

    fn parse_and_validate_generic(
        value: Value,
    ) -> Result<ValidatedChatCompletionRequest, ApiError> {
        ChatCompletionRequest::from_json(&serde_json::to_vec(&value).unwrap())?
            .validate_with_generic(MODEL, DEFAULT_MAX_COMPLETION_TOKENS, true)
    }

    fn strict_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]}
            },
            "required": ["city", "unit"],
            "additionalProperties": false
        })
    }

    fn function_tool(name: &str) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": "Look up weather",
                "parameters": strict_schema(),
                "strict": true
            }
        })
    }

    #[test]
    fn validates_supported_request_and_normalizes_defaults() {
        let request = parse_and_validate(json!({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "One?"},
                {"role": "assistant", "content": "1"},
                {"role": "user", "content": "Two?"}
            ]
        }))
        .unwrap();

        assert_eq!(request.model, MODEL);
        assert_eq!(request.messages.len(), 4);
        assert_eq!(
            request.messages.as_text().unwrap()[2],
            ChatMessage::assistant("1")
        );
        assert!(!request.stream);
        assert!(!request.include_usage);
        assert_eq!(request.max_completion_tokens, 128);
        assert_eq!(request.temperature, None);
        assert_eq!(request.top_p, None);
        assert_eq!(request.seed, None);
    }

    #[test]
    fn accepts_streaming_sampling_fields_and_deprecated_token_alias() {
        let request = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {"include_usage": true, "include_obfuscation": false},
            "max_tokens": 17,
            "temperature": 0.0,
            "top_p": 0.75,
            "seed": 7,
            "n": 1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "logprobs": false,
            "tools": [],
            "parallel_tool_calls": false
        }))
        .unwrap();

        assert!(request.stream);
        assert!(request.include_usage);
        assert_eq!(request.max_completion_tokens, 17);
        assert_eq!(request.temperature, Some(0.0));
        assert_eq!(request.top_p, Some(0.75));
        assert_eq!(request.seed, Some(7));
    }

    #[test]
    fn reports_json_and_unknown_field_errors_in_openai_shape() {
        let malformed = ChatCompletionRequest::from_json(br#"{"model": }"#).unwrap_err();
        assert_eq!(malformed.code.as_deref(), Some("invalid_json"));

        let unknown = ChatCompletionRequest::from_json(
            &serde_json::to_vec(&json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "magic": true
            }))
            .unwrap(),
        )
        .unwrap_err();
        assert_eq!(unknown.code.as_deref(), Some("invalid_json"));

        let response = serde_json::to_value(unknown.into_response()).unwrap();
        assert_eq!(response["error"]["type"], "invalid_request_error");
        assert_eq!(response["error"]["param"], Value::Null);
        assert_eq!(response["error"]["code"], "invalid_json");
        assert!(
            response["error"]["message"]
                .as_str()
                .unwrap()
                .contains("unknown field `magic`")
        );
    }

    #[test]
    fn rejects_model_choice_count_and_token_budget_errors() {
        for (patch, param, code) in [
            (json!({"model": "missing"}), "model", "model_not_found"),
            (json!({"n": 2}), "n", "unsupported_value"),
            (
                json!({"max_completion_tokens": 2, "max_tokens": 2}),
                "max_completion_tokens",
                "mutually_exclusive",
            ),
            (
                json!({"max_completion_tokens": 0}),
                "max_completion_tokens",
                "invalid_value",
            ),
        ] {
            let mut value = json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}]
            });
            value
                .as_object_mut()
                .unwrap()
                .extend(patch.as_object().unwrap().clone());
            let error = parse_and_validate(value).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(param));
            assert_eq!(error.code.as_deref(), Some(code));
        }
    }

    #[test]
    fn rejects_invalid_sampling_ranges_and_negative_seed() {
        for (field, value) in [
            ("temperature", json!(-0.1)),
            ("temperature", json!(2.1)),
            ("top_p", json!(0.0)),
            ("top_p", json!(1.1)),
            ("seed", json!(-1)),
        ] {
            let mut request = json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}]
            });
            request[field] = value;
            let error = parse_and_validate(request).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(field));
            assert_eq!(error.code.as_deref(), Some("invalid_value"));
        }
    }

    #[test]
    fn rejects_stream_options_without_streaming() {
        let error = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream_options": {"include_usage": true}
        }))
        .unwrap_err();
        assert_eq!(error.param.as_deref(), Some("stream_options"));
    }

    #[test]
    fn rejects_requested_stream_obfuscation_explicitly() {
        let error = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {"include_obfuscation": true}
        }))
        .unwrap_err();
        assert_eq!(
            error.param.as_deref(),
            Some("stream_options.include_obfuscation")
        );
        assert_eq!(error.code.as_deref(), Some("unsupported_value"));
    }

    #[test]
    fn explicitly_rejects_unsupported_request_features() {
        let cases = [("logprobs", json!(true)), ("top_logprobs", json!(2))];

        for (field, value) in cases {
            let mut request = json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}]
            });
            request[field] = value;
            let error = parse_and_validate(request).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(field));
        }
    }

    #[test]
    fn validates_and_preserves_strict_json_schema_contract() {
        let request = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Return weather as JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_response",
                    "description": "Machine-readable weather",
                    "schema": strict_schema(),
                    "strict": true
                }
            }
        }))
        .unwrap();

        let Some(ResponseFormat::JsonSchema { json_schema }) = &request.response_format else {
            panic!("expected a validated JSON schema response format");
        };
        assert_eq!(json_schema.name, "weather_response");
        assert_eq!(json_schema.schema, strict_schema());
        assert_eq!(request.into_text_runtime_messages().unwrap().len(), 1);
    }

    #[test]
    fn rejects_non_strict_or_structurally_open_json_schemas() {
        let cases = [
            (
                json!({
                    "name": "weather",
                    "schema": strict_schema(),
                    "strict": false
                }),
                "response_format.json_schema.strict",
            ),
            (
                json!({
                    "name": "weather",
                    "schema": {"type": "array", "items": {"type": "string"}},
                    "strict": true
                }),
                "response_format.json_schema.schema.type",
            ),
            (
                json!({
                    "name": "weather",
                    "schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": true
                    },
                    "strict": true
                }),
                "response_format.json_schema.schema.additionalProperties",
            ),
            (
                json!({
                    "name": "weather",
                    "schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": [],
                        "additionalProperties": false
                    },
                    "strict": true
                }),
                "response_format.json_schema.schema.required",
            ),
            (
                json!({
                    "name": "weather",
                    "schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "minLength": 1}},
                        "required": ["city"],
                        "additionalProperties": false
                    },
                    "strict": true
                }),
                "response_format.json_schema.schema",
            ),
        ];

        for (json_schema, param) in cases {
            let error = parse_and_validate(json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "response_format": {"type": "json_schema", "json_schema": json_schema}
            }))
            .unwrap_err();
            assert_eq!(error.param.as_deref(), Some(param));
            assert_eq!(error.code.as_deref(), Some("invalid_value"));
        }
    }

    #[test]
    fn rejects_user_stop_with_json_schema_before_engine_admission() {
        let error = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stop": "}",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "schema": strict_schema(),
                    "strict": true
                }
            }
        }))
        .unwrap_err();

        assert_eq!(error.param.as_deref(), Some("stop"));
        assert_eq!(error.code.as_deref(), Some("invalid_value"));
        assert!(error.message.contains("could terminate before"));
    }

    #[test]
    fn validates_forced_single_and_automatic_tool_subsets() {
        // Declared tools default to automatic choice, and `parallel_tool_calls`
        // keeps its OpenAI default: permission, never an obligation.
        let automatic = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [function_tool("get_weather")]
        }))
        .unwrap();
        assert_eq!(
            automatic.tool_choice,
            ToolChoice::Mode(ToolChoiceMode::Auto)
        );
        assert!(automatic.parallel_tool_calls);
        assert!(automatic.forced_tool().is_none());
        for parallel in [true, false] {
            let explicit = parse_and_validate(json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [function_tool("get_weather"), function_tool("get_time")],
                "tool_choice": "auto",
                "parallel_tool_calls": parallel
            }))
            .unwrap();
            assert_eq!(explicit.parallel_tool_calls, parallel);
        }

        let forced = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [function_tool("get_weather")],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"}
            },
            "parallel_tool_calls": false
        }))
        .unwrap();
        assert!(matches!(forced.tool_choice, ToolChoice::Function(_)));
        assert!(!forced.parallel_tool_calls);
        assert_eq!(forced.forced_tool().unwrap().function.name, "get_weather");

        let required = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [function_tool("get_weather")],
            "tool_choice": "required",
            "parallel_tool_calls": false
        }))
        .unwrap();
        assert_eq!(required.forced_tool().unwrap().function.name, "get_weather");

        let text_only = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [function_tool("get_weather")],
            "tool_choice": "none",
            "parallel_tool_calls": false
        }))
        .unwrap();
        assert!(text_only.forced_tool().is_none());
        assert_eq!(text_only.into_text_runtime_messages().unwrap().len(), 1);
    }

    #[test]
    fn rejects_invalid_tool_declarations_choices_and_combinations() {
        let cases = [
            (
                json!({"tool_choice": "auto"}),
                "tool_choice",
                "invalid_value",
            ),
            (
                json!({"parallel_tool_calls": true}),
                "parallel_tool_calls",
                "invalid_value",
            ),
            (
                json!({
                    "tools": [function_tool("get_weather"), function_tool("get_weather")]
                }),
                "tools.1.function.name",
                "invalid_value",
            ),
            (
                json!({
                    "tools": [function_tool("get_weather")],
                    "tool_choice": {"type": "function", "function": {"name": "missing"}}
                }),
                "tool_choice.function.name",
                "invalid_value",
            ),
        ];

        for (patch, param, code) in cases {
            let mut request = json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}]
            });
            request
                .as_object_mut()
                .unwrap()
                .extend(patch.as_object().unwrap().clone());
            let error = parse_and_validate(request).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(param));
            assert_eq!(error.code.as_deref(), Some(code));
        }
    }

    #[test]
    fn rejects_every_tool_mode_outside_the_supported_matrix() {
        let named = json!({
            "type": "function",
            "function": {"name": "get_weather"}
        });
        let cases = [
            (
                json!({
                    "tools": [function_tool("get_weather")],
                    "tool_choice": "auto",
                    "stop": "</tool_call>"
                }),
                "stop",
            ),
            (
                json!({
                    "tools": [function_tool("get_weather"), function_tool("get_time")],
                    "tool_choice": "required",
                    "parallel_tool_calls": false
                }),
                "tool_choice",
            ),
            (
                json!({
                    "tools": [function_tool("get_weather")],
                    "tool_choice": named.clone(),
                    "parallel_tool_calls": true
                }),
                "parallel_tool_calls",
            ),
            (
                json!({
                    "tools": [function_tool("get_weather")],
                    "tool_choice": named.clone(),
                    "parallel_tool_calls": false,
                    "stop": "}"
                }),
                "stop",
            ),
            (
                json!({
                    "tools": [function_tool("get_weather")],
                    "tool_choice": named,
                    "parallel_tool_calls": false,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "weather",
                            "strict": true,
                            "schema": strict_schema()
                        }
                    }
                }),
                "response_format",
            ),
        ];

        for (patch, param) in cases {
            let mut request = json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}]
            });
            request
                .as_object_mut()
                .unwrap()
                .extend(patch.as_object().unwrap().clone());
            let error = parse_and_validate(request).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(param));
        }
    }

    fn restloop_shell_tool() -> Value {
        json!({"type": "function", "function": {
            "name": "shell",
            "description": "Run a shell command and return its exit code and output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }})
    }

    #[test]
    fn accepts_non_strict_tools_by_closing_them_to_declared_properties() {
        let request = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "list files"}],
            "tools": [restloop_shell_tool()]
        }))
        .unwrap();
        assert_eq!(request.tool_choice, ToolChoice::Mode(ToolChoiceMode::Auto));
        let tool = &request.tools[0];
        assert!(!tool.function.strict);
        let schema = tool.compile_arguments_schema().unwrap();
        assert!(schema.accepts(&json!({"command": "ls"})));
        assert!(!schema.accepts(&json!({"command": "ls", "extra": true})));

        for (parameters, param) in [
            (
                json!({"type": "string"}),
                "tools.0.function.parameters.type",
            ),
            (
                json!({
                    "type": "object",
                    "properties": {"command": {"anyOf": [{"type": "string"}, {"type": "null"}]}}
                }),
                "tools.0.function.parameters",
            ),
        ] {
            let mut tool = restloop_shell_tool();
            tool["function"]["parameters"] = parameters;
            let error = parse_and_validate(json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "list files"}],
                "tools": [tool]
            }))
            .unwrap_err();
            assert_eq!(error.param.as_deref(), Some(param));
        }
    }

    #[test]
    fn accepts_only_the_enable_thinking_template_kwarg() {
        let request = |kwargs: Value| {
            parse_and_validate(json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "chat_template_kwargs": kwargs
            }))
        };
        assert_eq!(
            parse_and_validate(json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .unwrap()
            .chat_template,
            ChatTemplateOptions::default()
        );
        assert_eq!(
            request(json!({})).unwrap().chat_template,
            ChatTemplateOptions::default()
        );
        for enabled in [false, true] {
            assert_eq!(
                request(json!({"enable_thinking": enabled}))
                    .unwrap()
                    .chat_template
                    .enable_thinking,
                Some(enabled)
            );
        }
        for (kwargs, param, code) in [
            (
                json!({"enable_thinking": "no"}),
                "chat_template_kwargs.enable_thinking",
                "invalid_value",
            ),
            (
                json!({"enable_thinking": false, "add_generation_prompt": false}),
                "chat_template_kwargs.add_generation_prompt",
                "unsupported_value",
            ),
        ] {
            let error = request(kwargs).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(param));
            assert_eq!(error.code.as_deref(), Some(code));
        }
    }

    #[test]
    fn accepts_annotated_agent_tool_schemas_in_both_modes() {
        // The shape agent SDKs emit: titles and descriptions everywhere,
        // defaults and examples, plus bounds in a non-strict declaration.
        let non_strict = json!({"type": "function", "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "title": "ReadFileArgs",
                "properties": {
                    "path": {"type": "string", "title": "Path", "description": "File path.", "minLength": 1, "examples": ["README.md"]},
                    "offset": {"type": "integer", "description": "First line.", "default": 0, "minimum": 0},
                    "limit": {"type": "integer", "exclusiveMinimum": 0, "maximum": 2000}
                },
                "required": ["path"]
            }
        }});
        let mut strict = json!({"type": "function", "function": {
            "name": "lookup",
            "strict": true,
            "parameters": strict_schema()
        }});
        strict["function"]["parameters"]["properties"]["city"]["description"] = json!("City name.");
        strict["function"]["parameters"]["description"] = json!("Weather lookup.");
        let request = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [non_strict, strict]
        }))
        .unwrap();
        assert_eq!(request.tools.len(), 2);
        // Annotations reach the rendered declaration unchanged.
        assert_eq!(
            request.tools[0].function.parameters["properties"]["path"]["examples"],
            json!(["README.md"])
        );

        let mut strict_bound = json!({"type": "function", "function": {
            "name": "lookup",
            "strict": true,
            "parameters": strict_schema()
        }});
        strict_bound["function"]["parameters"]["properties"]["city"]["maxLength"] = json!(20);
        let error = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [strict_bound]
        }))
        .unwrap_err();
        assert!(error.message.contains("maxLength"));
    }

    #[test]
    fn keeps_visible_content_before_a_historical_tool_call() {
        let history = |content: Value| {
            parse_and_validate(json!({
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "list files"},
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "shell", "arguments": "{\"command\":\"ls\"}"}
                        }]
                    },
                    {"role": "tool", "content": "exit: 0", "tool_call_id": "call_1"}
                ],
                "tools": [restloop_shell_tool()]
            }))
            .unwrap()
        };
        for (content, expected) in [
            (
                json!("<think>\nls\n</think>"),
                Some("<think>\nls\n</think>"),
            ),
            (json!("Listing."), Some("Listing.")),
            (json!(""), None),
            (Value::Null, None),
        ] {
            let request = history(content);
            let AgentMessage::Assistant { content, .. } =
                &request.messages.as_tool_aware().unwrap()[1]
            else {
                panic!("expected an assistant tool call");
            };
            assert_eq!(content.as_deref(), expected);
        }
    }

    #[test]
    fn compiles_tool_argument_schemas_before_runtime_submission() {
        let mut tool = function_tool("get_weather");
        tool["function"]["parameters"]["properties"]["city"]["minLength"] = json!(1);
        let error = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [tool],
            "tool_choice": "required",
            "parallel_tool_calls": false
        }))
        .unwrap_err();
        assert_eq!(error.param.as_deref(), Some("tools.0.function.parameters"));
        assert!(error.message.contains("unsupported JSON schema"));
    }

    #[test]
    fn accepts_and_validates_stop_sequences() {
        let single = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stop": "END"
        }))
        .unwrap();
        assert_eq!(single.stop, ["END"]);

        let many = parse_and_validate(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stop": ["END", "DONE"]
        }))
        .unwrap();
        assert_eq!(many.stop, ["END", "DONE"]);

        for stop in [
            json!(""),
            json!(["ok", ""]),
            json!(["1", "2", "3", "4", "5"]),
        ] {
            let error = parse_and_validate(json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "stop": stop
            }))
            .unwrap_err();
            assert_eq!(error.param.as_deref(), Some("stop"));
            assert_eq!(error.code.as_deref(), Some("invalid_value"));
        }
    }

    #[test]
    fn validates_tool_call_and_tool_response_history_without_losing_fields() {
        let request = parse_and_validate(json!({
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "LA weather"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_la",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{ \"city\" : \"LA\" }"}
                        }
                    ]
                },
                {"role": "tool", "content": "warm", "tool_call_id": "call_la"}
            ]
        }))
        .unwrap();

        let messages = request.messages.as_tool_aware().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(
            messages[2],
            AgentMessage::Tool {
                content: "warm".to_owned(),
                tool_call_id: "call_la".to_owned()
            }
        );
        let AgentMessage::Assistant { tool_calls, .. } = &messages[1] else {
            panic!("expected assistant tool call");
        };
        assert_eq!(tool_calls[0].function.arguments, r#"{ "city" : "LA" }"#);
        assert_eq!(
            request
                .into_text_runtime_messages()
                .unwrap_err()
                .param
                .as_deref(),
            Some("messages")
        );
    }

    #[test]
    fn validates_multiple_historical_calls_and_out_of_order_tool_responses() {
        let calls = json!([
            {
                "id": "call_tokyo", "type": "function",
                "function": {"name": "weather", "arguments": "{\"city\":\"東京\"}"}
            },
            {
                "id": "call_la", "type": "function",
                "function": {"name": "weather", "arguments": "{ \"city\" : \"LA\" }"}
            }
        ]);
        let request = parse_and_validate_generic(json!({
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "Weather in both cities?"},
                {"role": "assistant", "content": "Checking both.", "tool_calls": calls},
                {"role": "tool", "content": "warm", "tool_call_id": "call_la"},
                {"role": "tool", "content": "rain", "tool_call_id": "call_tokyo"}
            ]
        }))
        .unwrap();
        let messages = request.messages.as_tool_aware().unwrap();
        let AgentMessage::Assistant {
            content,
            tool_calls,
        } = &messages[1]
        else {
            panic!("expected assistant tool calls");
        };
        assert_eq!(content.as_deref(), Some("Checking both."));
        assert_eq!(serde_json::to_value(tool_calls).unwrap(), calls);
        assert_eq!(
            messages[2],
            AgentMessage::Tool {
                content: "warm".into(),
                tool_call_id: "call_la".into(),
            }
        );
    }

    #[test]
    fn pinned_history_rejects_multiple_calls_with_original_error() {
        for arguments in ["{}", "not json"] {
            let request: ChatCompletionRequest = serde_json::from_value(json!({
                "model": MODEL,
                "messages": [
                    {"role":"user", "content":"go"},
                    {"role":"assistant", "tool_calls":[
                        {"id":"a", "type":"function", "function":{"name":"f", "arguments":arguments}},
                        {"id":"b", "type":"function", "function":{"name":"f", "arguments":"{}"}}
                    ]}
                ]
            }))
            .unwrap();
            let error = request
                .validate(MODEL, DEFAULT_MAX_COMPLETION_TOKENS)
                .unwrap_err();
            assert_eq!(error.status, ApiErrorStatus::BadRequest);
            assert_eq!(
                serde_json::to_string(&error.into_response()).unwrap(),
                r#"{"error":{"message":"the native tool history subset supports exactly one call per assistant turn","type":"invalid_request_error","param":"messages.1.tool_calls","code":"unsupported_value"}}"#,
                "arguments={arguments}"
            );
        }
    }

    #[test]
    fn rejects_duplicate_or_unknown_responses_in_a_historical_batch() {
        for second_id in ["call_a", "unknown"] {
            let error = parse_and_validate_generic(json!({
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "Weather?"},
                    {"role": "assistant", "tool_calls": [
                        {"id":"call_a", "type":"function", "function":{"name":"weather", "arguments":"{}"}},
                        {"id":"call_b", "type":"function", "function":{"name":"weather", "arguments":"{}"}}
                    ]},
                    {"role": "tool", "content": "result", "tool_call_id": "call_a"},
                    {"role": "tool", "content": "result", "tool_call_id": second_id}
                ]
            }))
            .unwrap_err();
            assert_eq!(error.param.as_deref(), Some("messages.3.tool_call_id"));
            assert_eq!(error.code.as_deref(), Some("invalid_value"));
        }
    }

    #[test]
    fn rejects_orphaned_mismatched_or_incomplete_tool_histories() {
        let cases = [
            (
                json!([{"role": "tool", "content": "result", "tool_call_id": "call_1"}]),
                "messages.0.role",
                "invalid_value",
            ),
            (
                json!([
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"}
                        }]
                    }
                ]),
                "messages",
                "invalid_value",
            ),
            (
                json!([
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"}
                        }]
                    },
                    {"role": "tool", "content": "result", "tool_call_id": "missing"}
                ]),
                "messages.2.tool_call_id",
                "invalid_value",
            ),
            (
                json!([
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "lookup", "arguments": "not json"}
                        }]
                    }
                ]),
                "messages.1.tool_calls.0.function.arguments",
                "invalid_value",
            ),
            (
                json!([
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": "calling",
                        "tool_calls": [
                            {
                                "id": "call_1", "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"}
                            },
                            {
                                "id": "call_2", "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"}
                            }
                        ]
                    }
                ]),
                "messages",
                "invalid_value",
            ),
        ];

        for (messages, param, code) in cases {
            let error = parse_and_validate_generic(json!({"model": MODEL, "messages": messages}))
                .unwrap_err();
            assert_eq!(error.param.as_deref(), Some(param));
            assert_eq!(error.code.as_deref(), Some(code));
        }
    }

    #[test]
    fn rejects_unsupported_message_features_and_multimodal_content() {
        let message_cases = [
            (
                json!({"role": "user", "content": "hello", "name": "alice"}),
                "messages.0.name",
                "unsupported_parameter",
            ),
            (
                json!({"role": "assistant", "content": "", "tool_calls": []}),
                "messages.0.tool_calls",
                "unsupported_parameter",
            ),
            (
                json!({"role": "user", "content": [{"type": "text", "text": "hello"}]}),
                "messages.0.content",
                "unsupported_value",
            ),
        ];

        for (message, param, code) in message_cases {
            let error = parse_and_validate(json!({
                "model": MODEL,
                "messages": [message]
            }))
            .unwrap_err();
            assert_eq!(error.param.as_deref(), Some(param));
            assert_eq!(error.code.as_deref(), Some(code));
        }
    }

    #[test]
    fn rejects_conversations_outside_the_template_subset() {
        for messages in [
            json!([]),
            json!([{"role": "assistant", "content": "hello"}]),
            json!([
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"}
            ]),
            json!([
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"}
            ]),
        ] {
            let error =
                parse_and_validate(json!({"model": MODEL, "messages": messages})).unwrap_err();
            assert_eq!(error.param.as_deref(), Some("messages"));
        }
    }

    #[test]
    fn serializes_non_streaming_response_shape() {
        let response = ChatCompletionResponse::new(
            "chatcmpl-1",
            123,
            MODEL,
            "hello",
            None,
            ChatFinishReason::Stop,
            TokenUsage::new(3, 2),
        );
        assert_eq!(
            serde_json::to_value(response).unwrap(),
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 123,
                "model": MODEL,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "logprobs": null,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 0}
                }
            })
        );
    }

    #[test]
    fn serializes_forced_tool_response_and_buffered_sse_shapes() {
        let call = AssistantToolCall {
            id: "call_1".to_owned(),
            kind: FunctionToolKind::Function,
            function: AssistantFunctionCall {
                name: "get_weather".to_owned(),
                arguments: r#"{"city":"LA"}"#.to_owned(),
            },
        };
        let response = ChatCompletionResponse::tool_call(
            "chatcmpl-1",
            123,
            MODEL,
            None,
            None,
            call.clone(),
            TokenUsage::new(3, 2),
        );
        let response = serde_json::to_value(response).unwrap();
        assert_eq!(response["choices"][0]["message"]["content"], Value::Null);
        assert_eq!(
            response["choices"][0]["message"]["tool_calls"][0]["function"],
            json!({"name": "get_weather", "arguments": "{\"city\":\"LA\"}"})
        );
        assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");

        let with_content = ChatCompletionResponse::tool_call(
            "chatcmpl-1",
            123,
            MODEL,
            Some("Checking.".to_owned()),
            None,
            call,
            TokenUsage::new(3, 2),
        );
        let with_content = serde_json::to_value(with_content).unwrap();
        assert_eq!(
            with_content["choices"][0]["message"]["content"],
            "Checking."
        );
        assert_eq!(
            with_content["choices"][0]["message"]["tool_calls"][0]["type"],
            "function"
        );

        let header = ChatCompletionChunk::tool_call_header(
            "chatcmpl-1",
            123,
            MODEL,
            "call_1",
            "get_weather",
        );
        let arguments =
            ChatCompletionChunk::tool_arguments("chatcmpl-1", 123, MODEL, r#"{"city":"LA"}"#);
        assert_eq!(
            serde_json::to_value(header).unwrap()["choices"][0]["delta"]["tool_calls"][0],
            json!({
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": ""}
            })
        );
        assert_eq!(
            serde_json::to_value(arguments).unwrap()["choices"][0]["delta"]["tool_calls"][0],
            json!({"index": 0, "function": {"arguments": "{\"city\":\"LA\"}"}})
        );
    }

    #[test]
    fn serializes_role_content_finish_and_usage_chunks() {
        let role = ChatCompletionChunk::role("chatcmpl-1", 123, MODEL);
        let content = ChatCompletionChunk::content("chatcmpl-1", 123, MODEL, "hel");
        let finished =
            ChatCompletionChunk::finished("chatcmpl-1", 123, MODEL, ChatFinishReason::Length);
        let usage = ChatCompletionChunk::usage("chatcmpl-1", 123, MODEL, TokenUsage::new(3, 2));

        assert_eq!(role.choices[0].delta.role, Some(ResponseRole::Assistant));
        assert_eq!(content.choices[0].delta.content.as_deref(), Some("hel"));
        assert_eq!(
            serde_json::to_value(finished).unwrap()["choices"][0],
            json!({"index": 0, "delta": {}, "logprobs": null, "finish_reason": "length"})
        );
        assert_eq!(
            serde_json::to_value(usage).unwrap(),
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": MODEL,
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 0}
                }
            })
        );
    }

    #[test]
    fn serializes_cached_prompt_tokens_in_openai_shape() {
        let usage = TokenUsage::with_cached_tokens(16_384, 42, 16_128);
        assert_eq!(
            serde_json::to_value(usage).unwrap(),
            json!({
                "prompt_tokens": 16_384,
                "completion_tokens": 42,
                "total_tokens": 16_426,
                "prompt_tokens_details": {"cached_tokens": 16_128}
            })
        );
    }

    #[test]
    fn maps_only_faithful_generation_finish_reasons() {
        assert_eq!(
            ChatFinishReason::from_generation(GenerationFinishReason::Stop),
            Some(ChatFinishReason::Stop)
        );
        assert_eq!(
            ChatFinishReason::from_generation(GenerationFinishReason::Length),
            Some(ChatFinishReason::Length)
        );
        assert_eq!(
            ChatFinishReason::from_generation(GenerationFinishReason::Cancelled),
            None
        );
    }
}
