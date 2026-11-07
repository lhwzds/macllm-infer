pub mod falcon_h1;
pub mod gemma;
pub mod gemma2;
pub mod gemma3;
pub mod granite_moe;
pub mod lfm2;
pub mod llama;
pub mod qwen2;
pub mod qwen3;
pub mod qwen35;
pub mod qwen3_moe;
pub mod qwen3_tasks;
pub mod smollm3;
#[cfg(test)]
mod template_parity;

use std::{fs, path::Path};

use anyhow::{Context, Result, bail, ensure};
use candle_nn::VarBuilder;
use serde::Deserialize;

use macllm_chat::{
    jinja::JinjaChatTemplate,
    tool_format::{self, ToolCallFormat},
};

use crate::model_weights::{DenseWeightSource, WeightSource};
use crate::{
    agent::{AgentPrompt, AgentTemplateProfile},
    architecture::{CausalLm, DecoderSpec, RuntimeSpec},
    chat::{ChatMessage, ChatRole, ChatTemplateOptions, validate_chat_messages},
};

/// Family-specific code stops at this boundary. It translates configuration,
/// weight naming and prompt conventions into the model runtime.
pub trait ModelAdapter {
    fn family(&self) -> &'static str;
    fn runtime_spec(&self) -> &RuntimeSpec;
    /// Return the shared pre-norm decoder geometry when this family uses it.
    /// Other autoregressive topologies may return `None` and build their own
    /// `CausalLm` implementation.
    fn decoder_spec(&self) -> Option<&DecoderSpec> {
        None
    }
    /// Convenience constructor for dense builders used by Safetensors and tests.
    fn build_dense(&self, weights: VarBuilder<'_>) -> candle_core::Result<Box<dyn CausalLm>> {
        let source = DenseWeightSource::new(weights);
        self.build(&source)
    }

    /// Canonical constructor for dense or block-quantized weights.
    fn build(&self, weights: &dyn WeightSource) -> candle_core::Result<Box<dyn CausalLm>> {
        let model = self.build_model(weights)?;
        if model.runtime_spec() != self.runtime_spec() {
            candle_core::bail!("adapter and model disagree about runtime metadata");
        }
        Ok(model)
    }

    /// Family-specific constructor used by the checked public build path.
    #[doc(hidden)]
    fn build_model(&self, weights: &dyn WeightSource) -> candle_core::Result<Box<dyn CausalLm>>;
    /// Render a validated chat history through a pinned renderer or the
    /// checkpoint's generic Jinja fallback.
    fn format_chat_messages(&self, messages: &[ChatMessage]) -> Result<String>;

    /// Template defaults relevant to serving and sampling.
    fn chat_template_defaults(&self) -> ChatTemplateOptions {
        ChatTemplateOptions::default()
    }

    /// Return a detected generic tool syntax for a non-pinned agent template.
    fn tool_call_format(&self) -> Option<ToolCallFormat> {
        None
    }

    /// Return the exact native agent-template profile verified for this
    /// checkpoint, or `None` when the pinned tool path is unavailable.
    fn agent_template_profile(&self) -> Option<AgentTemplateProfile> {
        None
    }

    /// Render a validated protocol-independent agent prompt.  Adapters must
    /// opt in with a pinned profile or a detected Jinja format; the default
    /// fails closed.
    fn format_agent_prompt(&self, _prompt: &AgentPrompt) -> Result<String> {
        bail!("native agent prompt rendering is unavailable for this model adapter")
    }

    /// Render a chat history with explicit template options. Adapters that
    /// implement none keep this fail-closed default.
    fn format_chat_messages_with(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        options.ensure_default(self.family())?;
        self.format_chat_messages(messages)
    }

    /// Render an agent prompt with explicit template options. Adapters that
    /// implement none keep this fail-closed default.
    fn format_agent_prompt_with(
        &self,
        prompt: &AgentPrompt,
        options: ChatTemplateOptions,
    ) -> Result<String> {
        options.ensure_default(self.family())?;
        self.format_agent_prompt(prompt)
    }

    /// Render and validate the adapter-specific serving startup probe through
    /// the same chat-template path used by real requests.
    fn format_serving_chat_probe(&self) -> Result<String> {
        self.format_chat_messages(&[ChatMessage::user("health check")])
    }

    /// Render a one-turn chat prompt through the same multi-turn template path.
    fn format_chat(&self, system: Option<&str>, user: &str) -> Result<String> {
        let mut messages = Vec::with_capacity(usize::from(system.is_some()) + 1);
        if let Some(system) = system {
            messages.push(ChatMessage::system(system));
        }
        messages.push(ChatMessage::user(user));
        self.format_chat_messages(&messages)
    }

    /// Hand the tokenizer prompt tokens a GGUF checkpoint embeds to the
    /// adapter. The GGUF loader calls this once, right after construction.
    ///
    /// Adapters whose template classification is byte-pinned keep rendering
    /// chat through their own renderer. An adapter that accepts a generic
    /// template compiles the Jinja fallback here for every other checkpoint
    /// program. The GGUF loader wraps the remaining adapters with the shared
    /// fallback instead.
    fn attach_prompt_tokens(&mut self, _tokens: GgufPromptTokens<'_>) -> Result<()> {
        Ok(())
    }

    /// Whether this adapter has a recognized native chat template.
    fn has_pinned_chat_template(&self) -> bool {
        false
    }

    fn validate(&self) -> Result<()> {
        self.runtime_spec().validate()?;
        if let Some(spec) = self.decoder_spec() {
            spec.validate()?;
            ensure!(
                &spec.runtime == self.runtime_spec(),
                "adapter runtime metadata disagrees with its decoder spec"
            );
        }
        Ok(())
    }
}

/// The markers the template-derived reserved list never holds.
///
/// The named list holds the markers of the reasoning and the tool-call
/// formats. They stay ordinary text by decision: text and reasoning that
/// mention them keep working, and user text that contains one tokenizes to
/// the same token the template prints.
///
/// The list applies to the reserved list the template derives, which the chat
/// route uses. The agent route also refuses the tool-call markers of this list
/// on its own: they are part of NATIVE_CONTROL_STRINGS in
/// crates/macllm-chat/src/agent.rs.
const NAMED_MARKERS: &[&str] = &["<think>", "</think>", "<tool_call>", "</tool_call>"];

/// The prompt tokens a GGUF checkpoint embeds for chat rendering: the Jinja
/// program in `tokenizer.chat_template` plus the special tokens it may print.
///
/// The strings are the tokenizer's own; a checkpoint that does not declare a
/// token leaves it `None`, which the generic renderer passes as empty.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GgufPromptTokens<'a> {
    pub chat_template: Option<&'a str>,
    pub bos_token: Option<&'a str>,
    pub eos_token: Option<&'a str>,
    /// The tokenizer's special tokens: the ones it marks special. The
    /// generic path reserves them all.
    pub special_tokens: &'a [String],
    /// The tokenizer's marker-like added tokens: every added token that
    /// looks like a marker.
    ///
    /// The generic path reserves every one of them but two kinds. The
    /// markers of NAMED_MARKERS stay ordinary text by decision. A plain
    /// markup tag stays ordinary text only when there is a valid template,
    /// when the template's own source does not name it and when no plain
    /// render of the template prints it; it is reserved otherwise, so an
    /// invalid template and a template that fails to render its probe keep
    /// it reserved. Every other marker shape is reserved without looking at
    /// the template.
    pub marker_tokens: &'a [String],
    /// The effective policy of the tokenizer: whether it adds its BOS token
    /// to every prompt. The tokenizer owns the policy, so a checkpoint whose
    /// metadata does not name it keeps the default of its own profile.
    pub add_bos_token: bool,
}

/// A checkpoint's embedded template, compiled for the generic renderer.
///
/// Adapters keep this for the templates they cannot verify byte-for-byte, so
/// those checkpoints still chat instead of failing closed. Compilation errors
/// are retained for chat requests, allowing raw-prompt inference to load and
/// build the same adapter without requiring a usable chat template.
#[derive(Debug, Clone)]
pub(crate) struct JinjaChatFallback {
    family: &'static str,
    template: std::result::Result<JinjaChatTemplate, String>,
    format: Option<ToolCallFormat>,
    /// The strings an untrusted request must not carry.
    reserved: Vec<String>,
    /// The tokenizer's BOS string, empty when the checkpoint declares none.
    bos_token: String,
    /// Whether the tokenizer adds its BOS token to every prompt.
    add_bos_token: bool,
}

impl JinjaChatFallback {
    pub(crate) fn tool_call_format(&self) -> Option<ToolCallFormat> {
        self.format
    }

    /// The template defaults the generic path serves.
    ///
    /// The default comes from the template's own renders: a template that
    /// honors the option has a default of its own, and a template that does
    /// not compile has none. The default decides the sampling defaults; the
    /// production renders do not use it, because a render leaves the option
    /// out and the template applies its own default.
    pub(crate) fn chat_template_defaults(&self) -> ChatTemplateOptions {
        ChatTemplateOptions {
            enable_thinking: self
                .template
                .as_ref()
                .ok()
                .and_then(JinjaChatTemplate::default_enable_thinking),
        }
    }
    pub(crate) fn format_agent_prompt(
        &self,
        prompt: &AgentPrompt,
        options: ChatTemplateOptions,
    ) -> Result<String> {
        use crate::agent::{AgentDirective, AgentMessage};
        use crate::chat::{ChatToolArguments, ChatToolCall, ChatToolFunction};
        ensure!(
            self.format.is_some(),
            "unsupported tool-call format in chat template"
        );
        ensure!(
            !matches!(prompt.directive, AgentDirective::ForcedSingle { .. }),
            "forced or required tool choice is unsupported for generic templates"
        );
        prompt.validate_generic()?;
        prompt.reject_generic_controls(&self.reserved)?;
        let messages: Vec<_> = prompt
            .messages
            .iter()
            .map(|message| match message {
                AgentMessage::System { content } => ChatMessage::system(content),
                AgentMessage::User { content } => ChatMessage::user(content),
                AgentMessage::Assistant { content, tool_call } => {
                    let mut message =
                        ChatMessage::assistant(content.as_deref().unwrap_or_default());
                    if let Some(call) = tool_call {
                        message.tool_calls = Some(vec![ChatToolCall {
                            id: call.id.clone(),
                            kind: "function".to_owned(),
                            function: ChatToolFunction {
                                name: call.name.clone(),
                                arguments: ChatToolArguments::from_json_text(&call.arguments),
                            },
                        }]);
                    }
                    message
                }
                AgentMessage::AssistantBatch {
                    content,
                    tool_calls,
                } => {
                    let mut message =
                        ChatMessage::assistant(content.as_deref().unwrap_or_default());
                    message.tool_calls = Some(
                        tool_calls
                            .iter()
                            .map(|call| ChatToolCall {
                                id: call.id.clone(),
                                kind: "function".to_owned(),
                                function: ChatToolFunction {
                                    name: call.name.clone(),
                                    arguments: ChatToolArguments::from_json_text(&call.arguments),
                                },
                            })
                            .collect(),
                    );
                    message
                }
                AgentMessage::Tool {
                    content,
                    tool_call_id,
                } => ChatMessage::tool(content, tool_call_id),
            })
            .collect();
        let tools: Vec<_> = prompt
            .tools
            .iter()
            .filter(|_| matches!(prompt.directive, AgentDirective::Auto))
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                })
            })
            .collect();
        let rendered = self
            .template
            .as_ref()
            .map_err(|error| anyhow::anyhow!("{error}"))?
            .render_agent(
                &messages,
                matches!(prompt.directive, AgentDirective::Auto).then_some(&tools),
                &options,
                true,
            )?;
        Ok(self.with_bos_token(rendered))
    }

    /// The prompt the tokenizer's own BOS policy expects.
    ///
    /// A tokenizer that adds its BOS token to every prompt needs the rendered
    /// prompt to start with it. A render that already carries the token
    /// keeps its bytes, whitespace before the token included: a template may
    /// print whitespace before its own BOS token, and the render carries it
    /// then.
    fn with_bos_token(&self, rendered: String) -> String {
        if !self.add_bos_token || self.bos_token.is_empty() {
            return rendered;
        }
        if rendered.trim_start().starts_with(&self.bos_token) {
            return rendered;
        }
        let mut prompt = String::with_capacity(rendered.len() + self.bos_token.len());
        prompt.push_str(&self.bos_token);
        prompt.push_str(&rendered);
        prompt
    }

    /// Render a validated chat history through the embedded template.
    ///
    /// Ordinary chat omits `tools`; agent prompts pass declarations through
    /// `format_agent_prompt` instead.
    pub(crate) fn format_chat_messages(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        validate_chat_messages(messages)?;
        for message in messages {
            reject_generic_content(&message.content, &self.reserved)?;
        }
        let rendered = self
            .template
            .as_ref()
            .map_err(|error| anyhow::anyhow!("{error}"))?
            .render(messages, None, &options, true)
            .with_context(|| {
                format!(
                    "the {} chat template could not render the conversation",
                    self.family
                )
            })?;
        Ok(self.with_bos_token(rendered))
    }
}

fn reject_generic_content(content: &str, reserved: &[String]) -> Result<()> {
    for control in reserved.iter().map(String::as_str) {
        if !control.is_empty() && content.contains(control) {
            anyhow::bail!("content contains reserved native control string `{control}`");
        }
    }
    Ok(())
}

/// Whether a marker-shaped token is a plain markup tag.
///
/// A plain markup tag is an opening angle bracket, an optional slash, one or
/// more ASCII lowercase letters or digits, and a closing angle bracket.
/// Nothing else belongs to the shape: a name with an underscore, with a bar
/// or with an upper case letter is a marker of its own kind.
fn is_plain_markup_tag(token: &str) -> bool {
    let Some(body) = token.strip_prefix('<') else {
        return false;
    };
    let Some(body) = body.strip_suffix('>') else {
        return false;
    };
    let body = body.strip_prefix('/').unwrap_or(body);
    !body.is_empty()
        && body
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
}

/// The marker-like tokens the generic path reserves.
///
/// A marker-like token is reserved when a client that forges one would write
/// a part of the prompt of its own. Real tokenizers hold ordinary HTML tags
/// as added tokens: Gemma 3 holds dozens of them, a table tag and a code tag
/// among them, and text that contains one must keep working. A token shaped
/// like a plain markup tag is therefore reserved only when the template's
/// source names it or when a plain render of the template prints it. The
/// source counts, so a mention of the tag in a template comment reserves it
/// as well; that fails closed.
///
/// Every other marker shape is reserved without looking at the template, so
/// a template cannot hide one by assembling it from pieces or by printing it
/// only on some turns.
///
/// The markers of the named list are never reserved: that is the decision
/// stated on NAMED_MARKERS.
///
/// The probe renders a plain conversation without a generation prompt: three
/// turns of plain text, and the same conversation with a plain system turn
/// in front of it. A template that refuses a system message cannot print
/// its system markers, so the second render is skipped when it fails.
///
/// A template that cannot render the probe keeps every plain markup tag
/// reserved.
///
/// One limit remains: a turn marker shaped like a plain markup tag, assembled
/// from pieces and printed only on a conditional turn, is not reserved.
fn reserved_marker_tokens(
    template: Option<&JinjaChatTemplate>,
    source: &str,
    marker_tokens: &[String],
) -> Vec<String> {
    let options = ChatTemplateOptions::default();
    let plain = [
        ChatMessage::user("hello"),
        ChatMessage::assistant("hi"),
        ChatMessage::user("again"),
    ];
    let with_system = [
        ChatMessage::system("hello"),
        ChatMessage::user("hello"),
        ChatMessage::assistant("hi"),
        ChatMessage::user("again"),
    ];
    // The renders of the probe, when the template renders them. A template
    // that cannot render the probe keeps every plain markup tag reserved.
    let probe = template.and_then(|template| {
        let first = template.render(&plain, None, &options, false).ok()?;
        // A template that refuses a system message cannot print a system
        // marker, so the second render is skipped when it fails.
        let second = template.render(&with_system, None, &options, false).ok();
        Some((first, second))
    });
    marker_tokens
        .iter()
        .filter(|marker| !NAMED_MARKERS.contains(&marker.as_str()))
        .filter(|marker| {
            // Every marker shape but a plain markup tag is reserved without
            // looking at the template.
            if !is_plain_markup_tag(marker.as_str()) {
                return true;
            }
            // A plain markup tag is reserved when the template's own source
            // names it or a plain render prints it. A template that cannot
            // render the probe keeps it reserved.
            let Some((first, second)) = probe.as_ref() else {
                return true;
            };
            source.contains(marker.as_str())
                || first.contains(marker.as_str())
                || second
                    .as_ref()
                    .is_some_and(|render| render.contains(marker.as_str()))
        })
        .cloned()
        .collect()
}

/// Resolve the chat prompt path of a checkpoint and compile the generic
/// renderer when the template is not one of the byte-pinned programs.
///
/// Adapters use this directly from [`ModelAdapter::attach_prompt_tokens`] or
/// through [`with_jinja_fallback`] with the result of their template
/// classification, so `classify_chat_template` returning `None` no longer
/// disables chat rendering for a checkpoint that embeds a template. A failed
/// compilation is cached just like a successful one and reported at render
/// time; it must never prevent raw-prompt inference.
pub(crate) fn jinja_fallback(
    family: &'static str,
    paths: GenericPaths,
    tokens: GgufPromptTokens<'_>,
) -> Option<JinjaChatFallback> {
    debug_assert!(!paths.chat || paths.agent);
    let source = tokens.chat_template;
    if paths.chat && source.is_none() {
        log_chat_template_path(family, "none (structured chat needs a raw prompt)");
        return None;
    }
    if paths.chat {
        log_chat_template_path(family, "jinja");
    } else {
        log_chat_template_path(family, "pinned");
    }
    if !paths.agent {
        return None;
    }
    let source = source?;
    let template = JinjaChatTemplate::new(
        source,
        tokens.bos_token.unwrap_or_default(),
        tokens.eos_token.unwrap_or_default(),
    )
    .map_err(|error| format!("the {family} chat template is not a valid Jinja program: {error:#}"));
    let format = tool_format::detect(source);
    eprintln!("tool call format: family={family} format={format:?}");
    // A pinned chat path does not serve the default of the template, so the
    // line belongs to the generic chat path alone.
    if paths.chat {
        // The default the template applies when the request leaves the option
        // out: the value the sampling defaults use.
        let thinking = template
            .as_ref()
            .ok()
            .and_then(JinjaChatTemplate::default_enable_thinking);
        eprintln!("thinking default: family={family} default={thinking:?}");
    }
    // The strings the generic path refuses in untrusted content: the
    // tokenizer's special tokens, and the marker-like tokens of the
    // tokenizer that the rule of `reserved_marker_tokens` reserves. That
    // rule keeps the markers of NAMED_MARKERS ordinary, reserves every
    // marker shape but a plain markup tag without looking at the template,
    // and reserves a plain markup tag when the template's own source names
    // it, when a plain render of the template prints it, or when there is
    // no valid template or the probe render fails. A client that forges one
    // of them would write a part of the prompt of its own.
    let markers = reserved_marker_tokens(template.as_ref().ok(), source, tokens.marker_tokens);
    let mut reserved = tokens.special_tokens.to_vec();
    for token in markers {
        if !reserved.contains(&token) {
            reserved.push(token);
        }
    }
    Some(JinjaChatFallback {
        family,
        template,
        format,
        reserved,
        bos_token: tokens.bos_token.unwrap_or_default().to_owned(),
        add_bos_token: tokens.add_bos_token,
    })
}

/// The prompt paths the generic renderer serves for a checkpoint.
///
/// The chat path and the agent path are decided separately: a checkpoint can
/// render chat through a byte-pinned template and still need the generic
/// renderer for its tool calls. The combination `chat: true, agent: false` is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GenericPaths {
    /// The generic renderer renders ordinary chat.
    pub chat: bool,
    /// The generic renderer renders agent prompts and their tool calls.
    pub agent: bool,
}

/// One line per loaded model, so an operator can see which prompt path a
/// checkpoint took.
fn log_chat_template_path(family: &str, path: &str) {
    eprintln!("chat template: family={family} path={path}");
}

/// Add generic chat rendering to adapters with unrecognized GGUF templates.
pub(crate) fn with_jinja_fallback(
    adapter: Box<dyn ModelAdapter>,
    tokens: GgufPromptTokens<'_>,
) -> Box<dyn ModelAdapter> {
    let pinned_chat = adapter.has_pinned_chat_template();
    // The shared fallback renders both paths, so it exists only when the
    // chat template is generic.
    let paths = GenericPaths {
        chat: !pinned_chat,
        agent: !pinned_chat,
    };
    let fallback = jinja_fallback(adapter.family(), paths, tokens);
    if let Some(fallback) = fallback {
        Box::new(FallbackAdapter { adapter, fallback })
    } else {
        adapter
    }
}

struct FallbackAdapter {
    adapter: Box<dyn ModelAdapter>,
    fallback: JinjaChatFallback,
}

impl ModelAdapter for FallbackAdapter {
    fn family(&self) -> &'static str {
        self.adapter.family()
    }
    fn runtime_spec(&self) -> &RuntimeSpec {
        self.adapter.runtime_spec()
    }
    fn decoder_spec(&self) -> Option<&DecoderSpec> {
        self.adapter.decoder_spec()
    }
    fn build_model(&self, weights: &dyn WeightSource) -> candle_core::Result<Box<dyn CausalLm>> {
        self.adapter.build_model(weights)
    }
    fn format_chat_messages(&self, messages: &[ChatMessage]) -> Result<String> {
        // The render leaves the option out, so the template applies its own
        // default on every turn. The default of the template is the value
        // the sampling defaults use.
        self.fallback
            .format_chat_messages(messages, ChatTemplateOptions::default())
    }
    fn format_chat_messages_with(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        self.fallback.format_chat_messages(messages, options)
    }
    fn chat_template_defaults(&self) -> ChatTemplateOptions {
        self.fallback.chat_template_defaults()
    }
    fn attach_prompt_tokens(&mut self, tokens: GgufPromptTokens<'_>) -> Result<()> {
        self.adapter.attach_prompt_tokens(tokens)
    }
    fn validate(&self) -> Result<()> {
        self.adapter.validate()
    }
    fn build_dense(&self, weights: VarBuilder<'_>) -> candle_core::Result<Box<dyn CausalLm>> {
        self.adapter.build_dense(weights)
    }
    fn build(&self, weights: &dyn WeightSource) -> candle_core::Result<Box<dyn CausalLm>> {
        self.adapter.build(weights)
    }
    fn tool_call_format(&self) -> Option<ToolCallFormat> {
        self.fallback.tool_call_format()
    }
    fn agent_template_profile(&self) -> Option<AgentTemplateProfile> {
        self.adapter.agent_template_profile()
    }
    fn format_agent_prompt(&self, prompt: &AgentPrompt) -> Result<String> {
        // The render leaves the option out, so the template applies its own
        // default on every turn. The default of the template is the value
        // the sampling defaults use.
        self.fallback
            .format_agent_prompt(prompt, ChatTemplateOptions::default())
    }
    fn format_agent_prompt_with(
        &self,
        prompt: &AgentPrompt,
        options: ChatTemplateOptions,
    ) -> Result<String> {
        self.fallback.format_agent_prompt(prompt, options)
    }
}

/// Detect a family from `model_type`. Adding a new family only extends this
/// function and its adapter; runtime and generation code stay unchanged.
pub fn detect(config_path: impl AsRef<Path>) -> Result<Box<dyn ModelAdapter>> {
    let path = config_path.as_ref();
    let config: serde_json::Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("invalid config JSON at {}", path.display()))?;
    match config.get("model_type").and_then(serde_json::Value::as_str) {
        Some("llama") => Ok(Box::new(llama::LlamaAdapter::from_config(path)?)),
        Some("lfm2") => Ok(Box::new(lfm2::Lfm2Adapter::from_config(path)?)),
        Some("qwen2") => Ok(Box::new(qwen2::Qwen2Adapter::from_config(path)?)),
        Some("qwen3") => Ok(Box::new(qwen3::Qwen3Adapter::from_config(path)?)),
        Some(model_type) => bail!("unsupported model_type `{model_type}`"),
        None => bail!("config has no string `model_type`"),
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl TokenIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::One(token) => vec![token],
            Self::Many(tokens) => tokens,
        }
    }
}

#[derive(Debug, Deserialize)]
struct GenerationMetadata {
    #[serde(default)]
    eos_token_id: Option<TokenIds>,
}

/// Prefer generation metadata over the model config and preserve first-seen
/// order while removing duplicate stop IDs.
pub(crate) fn load_stop_token_ids(config_path: &Path, fallback: Vec<u32>) -> Result<Vec<u32>> {
    let generation_path = config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("generation_config.json");
    let tokens = if generation_path.is_file() {
        let metadata: GenerationMetadata = serde_json::from_slice(
            &fs::read(&generation_path)
                .with_context(|| format!("failed to read {}", generation_path.display()))?,
        )
        .with_context(|| format!("invalid generation config at {}", generation_path.display()))?;
        metadata.eos_token_id.map_or(fallback, TokenIds::into_vec)
    } else {
        fallback
    };

    let mut unique = Vec::with_capacity(tokens.len());
    for token in tokens {
        if !unique.contains(&token) {
            unique.push(token);
        }
    }
    Ok(unique)
}

pub(crate) fn format_chatml(default_system: &str, system: Option<&str>, user: &str) -> String {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        system.unwrap_or(default_system),
        user
    )
}

pub(crate) fn format_chatml_messages(
    default_system: &str,
    messages: &[ChatMessage],
) -> Result<String> {
    validate_chat_messages(messages)?;
    let (system, turns) = match messages.first() {
        Some(ChatMessage {
            role: ChatRole::System,
            content,
            ..
        }) => (Some(content.as_str()), &messages[1..]),
        _ => (None, messages),
    };

    // Keep the established one-turn renderer as the canonical byte-for-byte
    // path while extending the same ChatML grammar to prior assistant turns.
    if let [message] = turns {
        return Ok(format_chatml(default_system, system, &message.content));
    }

    let mut prompt = format!(
        "<|im_start|>system\n{}<|im_end|>\n",
        system.unwrap_or(default_system)
    );
    for message in turns {
        prompt.push_str("<|im_start|>");
        prompt.push_str(message.role.as_str());
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(prompt)
}

/// Resolve unscaled `RoPE` metadata across legacy Transformers configs and the
/// Transformers 5 `rope_parameters` schema. Non-default scaling is rejected.
pub(crate) fn resolve_default_rope_theta(
    family: &str,
    legacy_theta: Option<f64>,
    parameters: Option<&serde_json::Value>,
) -> Result<f64> {
    let Some(parameters) = parameters else {
        return Ok(legacy_theta.unwrap_or(10_000.0));
    };
    let parameters = parameters
        .as_object()
        .with_context(|| format!("{family} rope_parameters must be an object"))?;
    let rope_type = parameters
        .get("rope_type")
        .map(|value| {
            value
                .as_str()
                .with_context(|| format!("{family} rope_parameters.rope_type must be a string"))
        })
        .transpose()?;
    let legacy_type = parameters
        .get("type")
        .map(|value| {
            value
                .as_str()
                .with_context(|| format!("{family} rope_parameters.type must be a string"))
        })
        .transpose()?;
    if let (Some(rope_type), Some(legacy_type)) = (rope_type, legacy_type) {
        ensure!(
            rope_type == legacy_type,
            "{family} rope_parameters.rope_type disagrees with rope_parameters.type"
        );
    }
    let rope_type = rope_type.or(legacy_type).unwrap_or("default");
    ensure!(
        rope_type == "default",
        "{family} rope_type `{rope_type}` is not supported yet"
    );
    for key in parameters.keys() {
        ensure!(
            matches!(
                key.as_str(),
                "rope_type" | "type" | "rope_theta" | "partial_rotary_factor"
            ),
            "{family} default rope parameter `{key}` is not supported"
        );
    }
    if let Some(partial) = parameters.get("partial_rotary_factor") {
        let partial = partial.as_f64().with_context(|| {
            format!("{family} rope_parameters.partial_rotary_factor must be numeric")
        })?;
        ensure!(
            (partial - 1.0).abs() < f64::EPSILON,
            "{family} partial rotary dimensions are not supported yet"
        );
    }
    let nested_theta = parameters
        .get("rope_theta")
        .map(|value| {
            value
                .as_f64()
                .with_context(|| format!("{family} rope_parameters.rope_theta must be numeric"))
        })
        .transpose()?;
    if let (Some(legacy), Some(nested)) = (legacy_theta, nested_theta) {
        ensure!(
            (legacy - nested).abs() < f64::EPSILON,
            "{family} top-level rope_theta disagrees with rope_parameters.rope_theta"
        );
    }
    Ok(nested_theta.or(legacy_theta).unwrap_or(10_000.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_llama_without_changing_the_runtime_boundary() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let config = temp.path().join("config.json");
        fs::write(
            &config,
            r#"{
              "model_type":"llama","vocab_size":32,"hidden_size":8,
              "intermediate_size":16,"num_hidden_layers":1,
              "num_attention_heads":2,"num_key_value_heads":1,
              "max_position_embeddings":16,"rms_norm_eps":0.00001,
              "hidden_act":"silu","eos_token_id":2
            }"#,
        )?;
        let adapter = detect(config)?;
        assert_eq!(adapter.family(), "llama");
        assert_eq!(adapter.runtime_spec().stop_token_ids, vec![2]);
        assert!(adapter.decoder_spec().is_some());
        Ok(())
    }

    #[test]
    fn resolves_only_default_rope_metadata_across_schema_versions() -> Result<()> {
        assert!((resolve_default_rope_theta("test", None, None)? - 10_000.0).abs() < f64::EPSILON);
        let current = serde_json::json!({"rope_type":"default","rope_theta":12_345.0});
        assert!(
            (resolve_default_rope_theta("test", None, Some(&current))? - 12_345.0).abs()
                < f64::EPSILON
        );
        let scaled = serde_json::json!({"rope_type":"llama3","rope_theta":500_000.0});
        assert!(resolve_default_rope_theta("test", None, Some(&scaled)).is_err());
        let malformed = serde_json::json!({"rope_type":3});
        assert!(resolve_default_rope_theta("test", None, Some(&malformed)).is_err());
        let conflicting = serde_json::json!({"rope_type":"default","type":"llama3"});
        assert!(resolve_default_rope_theta("test", None, Some(&conflicting)).is_err());
        let partial = serde_json::json!({
            "rope_type":"default","rope_theta":10_000.0,"partial_rotary_factor":0.5
        });
        assert!(resolve_default_rope_theta("test", None, Some(&partial)).is_err());
        assert!(resolve_default_rope_theta("test", Some(1.0), Some(&current)).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod generic_agent_tests {
    use super::*;
    use crate::agent::{AgentDirective, AgentMessage, ToolDefinition};

    #[test]
    fn generic_omits_tools_and_reserves_only_the_control_strings() {
        for (source, marker) in [
            ("<tool_call> \"name\": \"arguments\":", "<tool_call>"),
            ("<tool_call> <function=", "<function="),
            ("<|python_tag|>", "<|python_tag|>"),
            ("[TOOL_CALLS]", "[CALL_ID]"),
            ("<|tool_call|>", "<|tool_call|>"),
            ("<|tool_call_start|>", "<|tool_call_end|>"),
        ] {
            let template = format!(
                "{{# {source} #}}{{% if tools is not none %}}present{{% else %}}absent{{% endif %}}"
            );
            let special = vec!["<|endoftext|>".to_owned()];
            let fallback = jinja_fallback(
                "test",
                GenericPaths {
                    chat: true,
                    agent: true,
                },
                GgufPromptTokens {
                    chat_template: Some(&template),
                    special_tokens: &special,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut prompt = AgentPrompt {
                messages: vec![AgentMessage::User {
                    content: "hello".into(),
                }],
                tools: vec![],
                directive: AgentDirective::TextOnly,
            };
            assert_eq!(
                fallback
                    .format_agent_prompt(&prompt, ChatTemplateOptions::default())
                    .unwrap(),
                "absent"
            );
            // A tool-format marker is ordinary text: the generic path
            // reserves only the strings that can forge a turn of the render.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user(marker)],
                        ChatTemplateOptions::default()
                    )
                    .is_ok(),
                "{marker}"
            );
            // The tokenizer's own special tokens stay reserved.
            let reserved = "<|endoftext|>";
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user(reserved)],
                        ChatTemplateOptions::default()
                    )
                    .is_err(),
                "{reserved}"
            );
            prompt.messages = vec![AgentMessage::User {
                content: reserved.into(),
            }];
            assert!(
                fallback
                    .format_agent_prompt(&prompt, ChatTemplateOptions::default())
                    .is_err(),
                "{reserved}"
            );
        }
    }

    #[test]
    fn renders_tools_and_rejects_forced_choice() {
        let source = "<tool_call>{# \"name\": \"arguments\": #}{{ tools | tojson }}{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}";
        let fallback = jinja_fallback(
            "test",
            GenericPaths {
                chat: true,
                agent: true,
            },
            GgufPromptTokens {
                chat_template: Some(source),
                ..Default::default()
            },
        )
        .unwrap();
        let mut prompt = AgentPrompt {
            messages: vec![AgentMessage::User {
                content: "hello".into(),
            }],
            tools: vec![ToolDefinition {
                name: "weather".into(),
                description: Some("forecast".into()),
                parameters: serde_json::json!({"type":"object"}),
                strict: false,
                declaration: None,
            }],
            directive: AgentDirective::Auto,
        };
        let rendered = fallback
            .format_agent_prompt(&prompt, ChatTemplateOptions::default())
            .unwrap();
        assert!(
            rendered.contains("weather")
                && rendered.contains("forecast")
                && rendered.contains("hello")
        );
        // `none` keeps historical calls but does not advertise callable
        // functions for the next turn, even if the caller retains tools.
        prompt.directive = AgentDirective::TextOnly;
        assert_eq!(
            fallback
                .format_agent_prompt(&prompt, ChatTemplateOptions::default())
                .unwrap(),
            "<tool_call>nulluser:hello"
        );
        prompt.directive = AgentDirective::ForcedSingle {
            name: "weather".into(),
        };
        assert!(
            fallback
                .format_agent_prompt(&prompt, ChatTemplateOptions::default())
                .is_err()
        );
    }

    #[test]
    fn generic_history_is_passed_in_hugging_face_message_form() {
        let source = "{# <tool_call> \"name\": \"arguments\": #}{{ {'tools': tools, 'messages': messages, 'generation': add_generation_prompt, 'thinking': enable_thinking} | tojson }}";
        let fallback = jinja_fallback(
            "test",
            GenericPaths {
                chat: true,
                agent: true,
            },
            GgufPromptTokens {
                chat_template: Some(source),
                ..Default::default()
            },
        )
        .unwrap();
        let prompt = AgentPrompt {
            messages: vec![
                AgentMessage::User {
                    content: "weather?".into(),
                },
                AgentMessage::Assistant {
                    content: Some("Checking.".into()),
                    tool_call: Some(crate::agent::ToolCall {
                        id: "call_1".into(),
                        name: "weather".into(),
                        arguments: r#"{"city":"東京"}"#.into(),
                    }),
                },
                AgentMessage::Tool {
                    content: "sunny".into(),
                    tool_call_id: "call_1".into(),
                },
            ],
            tools: vec![ToolDefinition {
                name: "weather".into(),
                description: Some("forecast".into()),
                parameters: serde_json::json!({"type":"object", "properties":{"city":{"type":"string"}}}),
                strict: false,
                declaration: None,
            }],
            directive: AgentDirective::Auto,
        };
        let rendered = fallback
            .format_agent_prompt(
                &prompt,
                ChatTemplateOptions {
                    enable_thinking: Some(false),
                },
            )
            .unwrap();
        let value: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "tools": [{"type":"function", "function": {
                    "name":"weather", "description":"forecast", "parameters":prompt.tools[0].parameters,
                }}],
                "messages": [
                    {"role":"user", "content":"weather?"},
                    {"role":"assistant", "content":"Checking.", "tool_calls":[{
                        "id":"call_1", "type":"function", "function":{"name":"weather", "arguments":{"city":"東京"}},
                    }]},
                    {"role":"tool", "content":"sunny", "tool_call_id":"call_1"},
                ],
                "generation":true, "thinking":false,
            })
        );
    }

    #[test]
    fn generic_batch_history_renders_one_assistant_message() {
        let source = "{# [TOOL_CALLS] #}{{ messages | tojson }}";
        let fallback = jinja_fallback(
            "test",
            GenericPaths {
                chat: true,
                agent: true,
            },
            GgufPromptTokens {
                chat_template: Some(source),
                ..Default::default()
            },
        )
        .unwrap();
        let calls = ["a", "b"]
            .into_iter()
            .map(|id| crate::agent::ToolCall {
                id: id.into(),
                name: "weather".into(),
                arguments: r#"{"city":"東京"}"#.into(),
            })
            .collect();
        let prompt = AgentPrompt {
            messages: vec![
                AgentMessage::User {
                    content: "weather?".into(),
                },
                AgentMessage::AssistantBatch {
                    content: Some("Checking both.".into()),
                    tool_calls: calls,
                },
                AgentMessage::Tool {
                    content: "b".into(),
                    tool_call_id: "b".into(),
                },
                AgentMessage::Tool {
                    content: "a".into(),
                    tool_call_id: "a".into(),
                },
            ],
            tools: vec![],
            directive: AgentDirective::TextOnly,
        };
        let rendered = fallback
            .format_agent_prompt(&prompt, ChatTemplateOptions::default())
            .unwrap();
        let value: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(
            value,
            serde_json::json!([
                {"role":"user", "content":"weather?"},
                {"role":"assistant", "content":"Checking both.", "tool_calls":[
                    {"id":"a", "type":"function", "function":{"name":"weather", "arguments":{"city":"東京"}}},
                    {"id":"b", "type":"function", "function":{"name":"weather", "arguments":{"city":"東京"}}},
                ]},
                {"role":"tool", "content":"b", "tool_call_id":"b"},
                {"role":"tool", "content":"a", "tool_call_id":"a"},
            ])
        );
    }

    /// The reserved control strings and the BOS token of the generic path.
    mod generic_reserved_tests {
        use super::*;
        use crate::agent::{AgentDirective, AgentMessage, ToolCall, ToolDefinition};

        /// A Qwen-style template: the ChatML markers are the tokenizer's special
        /// tokens, and the role marker is a plain token the template prints once
        /// per turn.
        const TEMPLATE: &str = "{% for message in messages %}<|im_start|><start_of_turn>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if tools is not none %}<|tool_call|>{% endif %}";

        /// The fallback of one template with the given control tokens.
        fn compile_fallback(
            template: &str,
            special: &[String],
            markers: &[String],
        ) -> JinjaChatFallback {
            jinja_fallback(
                "test",
                GenericPaths {
                    chat: true,
                    agent: true,
                },
                GgufPromptTokens {
                    chat_template: Some(template),
                    special_tokens: special,
                    marker_tokens: markers,
                    ..Default::default()
                },
            )
            .expect("the fallback compiles")
        }

        #[test]
        fn the_special_tokens_of_the_tokenizer_are_always_reserved() {
            let special = vec!["<|im_start|>".to_owned(), "<|im_end|>".to_owned()];
            let fallback = compile_fallback(TEMPLATE, &special, &[]);
            for content in ["<|im_start|>", "<|im_end|>"] {
                let error = fallback
                    .format_chat_messages(
                        &[ChatMessage::user(content)],
                        ChatTemplateOptions::default(),
                    )
                    .expect_err("a special token must be refused");
                assert!(
                    format!("{error:#}").contains("reserved native control string"),
                    "{error:#}"
                );
            }
        }

        #[test]
        fn a_marker_that_delimits_a_turn_is_reserved() {
            // The role marker is not a special token, but the template prints it
            // once per turn, so a client that forges it would write a turn.
            let markers = vec!["<start_of_turn>".to_owned()];
            let fallback = compile_fallback(TEMPLATE, &[], &markers);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<start_of_turn>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
            // A marker the template never prints is ordinary text.
            let other = vec!["<table>".to_owned()];
            let fallback = compile_fallback(TEMPLATE, &[], &other);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<table>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_ok()
            );
        }

        #[test]
        fn the_tool_format_markers_are_ordinary_text() {
            let special = vec!["<|im_start|>".to_owned(), "<|im_end|>".to_owned()];
            let markers = vec![
                "<think>".to_owned(),
                "</think>".to_owned(),
                "<tool_call>".to_owned(),
            ];
            let fallback = compile_fallback(TEMPLATE, &special, &markers);
            for content in ["<think>", "</think>", "<tool_call>"] {
                assert!(
                    fallback
                        .format_chat_messages(
                            &[ChatMessage::user(content)],
                            ChatTemplateOptions::default(),
                        )
                        .is_ok(),
                    "{content}"
                );
                // The template-derived reserved list holds no tool-format
                // marker, so this check passes; validate_generic applies the
                // fixed list of NATIVE_CONTROL_STRINGS on its own.
                let prompt = AgentPrompt {
                    messages: vec![AgentMessage::User {
                        content: content.into(),
                    }],
                    tools: vec![],
                    directive: AgentDirective::TextOnly,
                };
                assert!(
                    prompt.reject_generic_controls(&fallback.reserved).is_ok(),
                    "{content}"
                );
            }
            // The agent path renders a history that carries the marker in a tool
            // result and in the tool arguments.
            let prompt = AgentPrompt {
                messages: vec![
                    AgentMessage::User {
                        content: "run it".into(),
                    },
                    AgentMessage::Assistant {
                        content: Some(String::new()),
                        tool_call: Some(ToolCall {
                            id: "call_1".into(),
                            name: "shell".into(),
                            arguments: r#"{"command":"<think>"}"#.into(),
                        }),
                    },
                    AgentMessage::Tool {
                        content: "<think>done</think>".into(),
                        tool_call_id: "call_1".into(),
                    },
                ],
                tools: vec![ToolDefinition {
                    name: "shell".into(),
                    description: None,
                    parameters: serde_json::json!({"type": "object"}),
                    strict: false,
                    declaration: None,
                }],
                directive: AgentDirective::TextOnly,
            };
            let rendered = fallback.format_agent_prompt(&prompt, ChatTemplateOptions::default());
            assert!(rendered.is_ok(), "{rendered:?}");
        }

        #[test]
        fn a_template_that_cannot_render_the_probe_keeps_every_marker_reserved() {
            // The template refuses a conversation of more than one turn, so the
            // probe fails and every marker-like token stays reserved.
            let template = "{% if messages | length > 1 %}{{ raise_exception('too many turns') }}{% endif %}{{ messages[0].content }}";
            let markers = vec!["<table>".to_owned()];
            let fallback = compile_fallback(template, &[], &markers);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<table>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
            // A template that renders the probe reserves only its turn delimiters.
            let rendered = compile_fallback(TEMPLATE, &[], &markers);
            assert!(
                rendered
                    .format_chat_messages(
                        &[ChatMessage::user("<table>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_ok()
            );
        }

        #[test]
        fn a_template_that_does_not_compile_keeps_every_plain_markup_tag_reserved() {
            // The template is not a valid Jinja program, so there is no
            // valid template and no probe render: every plain markup tag
            // stays reserved.
            let template = "{% for message in messages %}{{ message.content }";
            let markers = vec!["<table>".to_owned()];
            let fallback = compile_fallback(template, &[], &markers);
            assert!(fallback.reserved.contains(&"<table>".to_owned()));
        }

        #[test]
        fn a_marker_that_only_a_system_turn_prints_is_reserved() {
            // The template prints `<|system|>` for a system turn and a marker
            // for every turn. A client that forges the system marker would
            // write a system turn of its own.
            let template = "{% for message in messages %}{% if message.role == 'system' %}<|system|>{% endif %}<start_of_turn>{{ message.role }}\n{{ message.content }}{% endfor %}";
            let markers = vec!["<|system|>".to_owned(), "<start_of_turn>".to_owned()];
            let fallback = compile_fallback(template, &[], &markers);
            // The system marker is refused in untrusted content.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<|system|>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
            // The marker of every turn stays refused as well.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<start_of_turn>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
            // Ordinary text still renders.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("hello")],
                        ChatTemplateOptions::default(),
                    )
                    .is_ok()
            );
        }

        #[test]
        fn a_marker_a_template_assembles_from_pieces_is_reserved() {
            // The template builds the marker from pieces, so the marker never
            // appears in its own source. The plain render prints it anyway,
            // so a client that forges it would write a turn of its own.
            use tokenizers::{AddedToken, models::bpe::BPE};
            let template = "{% for message in messages %}{{ '<|syste' ~ 'm|>' }}{{ message.content }}{% endfor %}";
            let mut tokenizer = tokenizers::Tokenizer::new(BPE::default());
            tokenizer
                .add_tokens([AddedToken::from("<|system|>", false)])
                .unwrap();
            let control = crate::gguf_model::prompt_control_tokens(&tokenizer);
            // The marker is not a special token of the tokenizer, but it is a
            // marker-like added token, so it is in the marker list.
            assert!(!control.special.contains(&"<|system|>".to_owned()));
            assert!(control.markers.contains(&"<|system|>".to_owned()));
            let fallback = compile_fallback(template, &control.special, &control.markers);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<|system|>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
        }

        #[test]
        fn a_template_that_refuses_a_system_message_keeps_its_turn_delimiter() {
            // The template refuses a system message, so it cannot print a
            // system marker. That failure must not reserve every marker.
            let template = "{% if messages[0].role == 'system' %}{{ raise_exception('no system turn') }}{% endif %}{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ message.content }}{% endfor %}";
            let markers = vec!["<start_of_turn>".to_owned(), "<think>".to_owned()];
            let fallback = compile_fallback(template, &[], &markers);
            // The marker of every turn is still refused.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<start_of_turn>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
            // A marker the template never prints stays ordinary text.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<think>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_ok()
            );
        }

        #[test]
        fn a_marker_that_a_tool_turn_prints_is_reserved() {
            // The template prints `<|tool|>` and `<tool_response>` for a
            // tool turn. Neither marker is a special token of the tokenizer,
            // but the template's own source names them, so a client that
            // forges one would write the tool turn of the template.
            let template = "{% for message in messages %}{% if message.role == 'tool' %}<|tool|><tool_response>{% endif %}<start_of_turn>{{ message.role }}\n{{ message.content }}{% endfor %}";
            let markers = vec!["<|tool|>".to_owned(), "<tool_response>".to_owned()];
            let fallback = compile_fallback(template, &[], &markers);
            // The tool marker is refused in untrusted content.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<|tool|>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
            // The tool response marker of the same setup is refused as well.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<tool_response>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
        }

        #[test]
        fn a_default_system_turn_that_prints_a_marker_is_reserved() {
            // The template prints `<|system|>` for the system turn it makes
            // up itself. The marker is not a special token of the tokenizer,
            // but the template prints it in a plain conversation, so a client
            // that forges it would write the default system turn.
            let template = "{% if messages[0].role != 'system' %}<|system|>default system prompt{% endif %}{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ message.content }}{% endfor %}";
            let markers = vec!["<|system|>".to_owned(), "<start_of_turn>".to_owned()];
            let fallback = compile_fallback(template, &[], &markers);
            // The system marker is refused in untrusted content.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<|system|>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
            // Ordinary text still renders.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("hello")],
                        ChatTemplateOptions::default(),
                    )
                    .is_ok()
            );
        }

        #[test]
        fn a_marker_that_the_generation_prompt_prints_is_reserved() {
            // The template prints `<|gen|>` only when it asks for the next
            // turn. A plain conversation never prints it, but the source
            // names it, so a client that forges it would write the
            // generation prompt of the template.
            let template = "{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ message.content }}{% endfor %}{% if add_generation_prompt %}<|gen|><think>{% endif %}";
            let markers = vec![
                "<|gen|>".to_owned(),
                "<think>".to_owned(),
                "<start_of_turn>".to_owned(),
            ];
            let fallback = compile_fallback(template, &[], &markers);
            // The marker of the generation prompt is refused.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<|gen|>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
            // The reasoning marker of the same template stays plain text: it
            // is on the named list.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<think>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_ok()
            );
            // The marker of every turn is still refused.
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<start_of_turn>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
        }

        #[test]
        fn a_marker_that_the_tools_print_is_reserved() {
            // The template prints `<|tools|>` when tools are declared. A
            // conversation without tools never prints it, but the source
            // names it, so the marker is reserved.
            let template = "{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ message.content }}{% endfor %}{% if tools is not none %}<|tools|>{% endif %}";
            let markers = vec!["<|tools|>".to_owned()];
            let fallback = compile_fallback(template, &[], &markers);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<|tools|>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
        }

        #[test]
        fn the_reasoning_marker_of_a_default_system_prompt_stays_plain_text() {
            // The template prints `<think>` for the system turn it makes up
            // itself. The marker is on the named list, so it stays ordinary
            // text while the tokenizer holds it as a marker-like token.
            let template = "{% if messages[0].role != 'system' %}<think>default system prompt{% endif %}{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ message.content }}{% endfor %}";
            let markers = vec!["<think>".to_owned()];
            let fallback = compile_fallback(template, &[], &markers);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<think>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_ok()
            );
            // A tokenizer that marks the token special reserves it anyway.
            let special = vec!["<think>".to_owned()];
            let fallback = compile_fallback(template, &special, &markers);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<think>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
        }

        #[test]
        fn a_marker_the_template_neither_names_nor_prints_stays_plain_text() {
            // The marker is a marker-like token of the tokenizer, but the
            // template neither names nor prints it, so it is ordinary text.
            let markers = vec!["<table>".to_owned()];
            let fallback = compile_fallback(TEMPLATE, &[], &markers);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<table>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_ok()
            );
        }

        #[test]
        fn a_marker_a_template_assembles_for_a_tool_turn_is_reserved() {
            // The template assembles the marker from pieces and prints it
            // only for a tool turn, so its own source never names it. The
            // marker is not a plain markup tag, so it is reserved without
            // looking at the template.
            let template = "{% for message in messages %}{% if message.role == 'tool' %}{{ '<|to' ~ 'ol|>' }}{% endif %}{{ message.content }}{% endfor %}";
            let markers = vec!["<|tool|>".to_owned()];
            let fallback = compile_fallback(template, &[], &markers);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<|tool|>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
        }

        #[test]
        fn a_marker_of_another_shape_is_reserved_without_the_template() {
            // The template neither names nor prints the marker, and the
            // marker is not a plain markup tag, so it is reserved anyway.
            let template = "{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ message.content }}{% endfor %}";
            let markers = vec!["<tool_response>".to_owned()];
            let fallback = compile_fallback(template, &[], &markers);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<tool_response>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
        }

        #[test]
        fn the_plain_markup_tags_of_the_tokenizer_stay_plain_text() {
            // The template neither names nor prints the tags, so they stay
            // ordinary text: a real tokenizer holds them as added tokens.
            let markers = vec![
                "<table>".to_owned(),
                "</td>".to_owned(),
                "<code>".to_owned(),
            ];
            let fallback = compile_fallback(TEMPLATE, &[], &markers);
            for content in ["<table>", "</td>", "<code>"] {
                assert!(
                    fallback
                        .format_chat_messages(
                            &[ChatMessage::user(content)],
                            ChatTemplateOptions::default(),
                        )
                        .is_ok(),
                    "{content}"
                );
            }
        }

        #[test]
        fn a_plain_markup_tag_the_template_uses_is_reserved() {
            // The source names the tag, so a client that forges it would
            // write a turn of the template.
            let template = "{% for message in messages %}<user>{{ message.role }}\n{{ message.content }}{% endfor %}";
            let markers = vec!["<user>".to_owned()];
            let fallback = compile_fallback(template, &[], &markers);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<user>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
            // A mention of the tag in a comment of the source reserves it as
            // well, even when no render of the template prints it.
            let template = "{# <user> #}{% for message in messages %}{{ message.role }}\n{{ message.content }}{% endfor %}";
            let fallback = compile_fallback(template, &[], &markers);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<user>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
            // The template assembles the tag from pieces, so its source does
            // not name it, but a plain render prints it.
            let template =
                "{% for message in messages %}{{ '<us' ~ 'er>' }}{{ message.content }}{% endfor %}";
            let fallback = compile_fallback(template, &[], &markers);
            assert!(
                fallback
                    .format_chat_messages(
                        &[ChatMessage::user("<user>")],
                        ChatTemplateOptions::default(),
                    )
                    .is_err()
            );
        }

        #[test]
        fn a_plain_markup_tag_has_a_shape_of_its_own() {
            for token in ["<table>", "</td>", "<h1>", "<code>", "<b>"] {
                assert!(is_plain_markup_tag(token), "{token}");
            }
            for token in [
                "<|tool|>",
                "<tool_response>",
                "<start_of_turn>",
                "[PAD]",
                "<|\"|>",
                "<|USER|>",
            ] {
                assert!(!is_plain_markup_tag(token), "{token}");
            }
        }

        #[test]
        fn the_render_carries_the_bos_token_the_tokenizer_policy_asks_for() -> Result<()> {
            let plain = "{{ messages[0].content }}";
            let with_bos = "{{ bos_token }}{{ messages[0].content }}";
            let whitespace = " {{ bos_token }}{{ messages[0].content }}";
            let render = |template: &str, add_bos: bool, bos: Option<&str>| {
                let fallback = jinja_fallback(
                    "test",
                    GenericPaths {
                        chat: true,
                        agent: true,
                    },
                    GgufPromptTokens {
                        chat_template: Some(template),
                        bos_token: bos,
                        add_bos_token: add_bos,
                        ..Default::default()
                    },
                )
                .expect("the fallback compiles");
                fallback.format_chat_messages(
                    &[ChatMessage::user("hi")],
                    ChatTemplateOptions::default(),
                )
            };
            // A tokenizer that adds its BOS token to every prompt gets a render
            // that starts with it.
            assert_eq!(render(plain, true, Some("<bos>"))?, "<bos>hi");
            // A render that already starts with the BOS token keeps its bytes.
            assert_eq!(render(with_bos, true, Some("<bos>"))?, "<bos>hi");
            // A render that carries whitespace before the BOS token keeps its
            // bytes as well: the token is in the render already.
            assert_eq!(render(whitespace, true, Some("<bos>"))?, " <bos>hi");
            // A tokenizer that does not add its BOS token leaves the render alone.
            assert_eq!(render(plain, false, Some("<bos>"))?, "hi");
            // A checkpoint without a BOS string has nothing to prepend.
            assert_eq!(render(plain, true, None)?, "hi");
            Ok(())
        }

        #[test]
        fn the_agent_render_carries_the_same_bos_token() -> Result<()> {
            let fallback = fallback_with_bos();
            let prompt = AgentPrompt {
                messages: vec![AgentMessage::User {
                    content: "hi".into(),
                }],
                tools: vec![],
                directive: AgentDirective::TextOnly,
            };
            assert_eq!(
                fallback.format_agent_prompt(&prompt, ChatTemplateOptions::default())?,
                "<bos>hi"
            );
            Ok(())
        }

        /// A fallback whose tokenizer adds its BOS token to every prompt.
        fn fallback_with_bos() -> JinjaChatFallback {
            jinja_fallback(
                "test",
                GenericPaths {
                    chat: true,
                    agent: true,
                },
                GgufPromptTokens {
                    chat_template: Some(
                        "{{ messages[0].content }}{% if tools is not none %}<|tool_call|>{% endif %}",
                    ),
                    bos_token: Some("<bos>"),
                    add_bos_token: true,
                    ..Default::default()
                },
            )
            .expect("the fallback compiles")
        }
    }

    /// The template options of the generic path.
    mod generic_option_tests {
        use super::*;
        use crate::agent::{AgentDirective, AgentMessage};

        /// The fallback of one template.
        fn compile_fallback(template: &str) -> JinjaChatFallback {
            jinja_fallback(
                "test",
                GenericPaths {
                    chat: true,
                    agent: true,
                },
                GgufPromptTokens {
                    chat_template: Some(template),
                    ..Default::default()
                },
            )
            .expect("the fallback compiles")
        }

        #[test]
        fn the_generic_path_refuses_an_option_its_template_never_reads() -> Result<()> {
            let options = ChatTemplateOptions {
                enable_thinking: Some(false),
            };
            let fallback = compile_fallback("{{ messages[0].content }}");
            let error = fallback
                .format_chat_messages(&[ChatMessage::user("hi")], options)
                .expect_err("the option must be refused");
            assert!(
                format!("{error:#}").contains("enable_thinking"),
                "{error:#}"
            );
            // The agent path refuses the option as well.
            let fallback = compile_fallback(
                "{{ messages[0].content }}{% if tools is not none %}<|tool_call|>{% endif %}",
            );
            let prompt = AgentPrompt {
                messages: vec![AgentMessage::User {
                    content: "hi".into(),
                }],
                tools: vec![],
                directive: AgentDirective::TextOnly,
            };
            let error = fallback
                .format_agent_prompt(&prompt, options)
                .expect_err("the option must be refused");
            assert!(
                format!("{error:#}").contains("enable_thinking"),
                "{error:#}"
            );
            // A template that reads the option receives it.
            let fallback = compile_fallback("{{ enable_thinking }}");
            assert_eq!(
                fallback.format_chat_messages(&[ChatMessage::user("hi")], options)?,
                "False"
            );
            // A request that does not ask for the option is untouched.
            let fallback = compile_fallback("{{ messages[0].content }}");
            assert_eq!(
                fallback.format_chat_messages(
                    &[ChatMessage::user("hi")],
                    ChatTemplateOptions::default()
                )?,
                "hi"
            );
            Ok(())
        }
    }

    /// The thinking default of the generic path.
    mod generic_thinking_default_tests {
        use super::*;
        use crate::models::{
            qwen3::Qwen3Adapter, qwen3_moe::Qwen3MoeAdapter, qwen35::Qwen35Adapter,
        };

        /// The source of one template fixture.
        fn fixture(name: &str) -> String {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/chat_templates/templates")
                .join(format!("{name}.jinja"));
            std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        }

        /// The fallback of one template.
        fn compile_fallback(template: &str) -> JinjaChatFallback {
            jinja_fallback(
                "test",
                GenericPaths {
                    chat: true,
                    agent: true,
                },
                GgufPromptTokens {
                    chat_template: Some(template),
                    ..Default::default()
                },
            )
            .expect("the fallback compiles")
        }

        /// A decoder geometry for one family.
        fn spec(architecture: &str) -> DecoderSpec {
            let config: crate::models::qwen3::Qwen3Config =
                serde_json::from_value(serde_json::json!({
                    "model_type":"qwen3", "vocab_size":128, "hidden_size":64,
                    "intermediate_size":128, "num_hidden_layers":1, "num_attention_heads":4,
                    "num_key_value_heads":2, "head_dim":16, "max_position_embeddings":128,
                    "tie_word_embeddings":true, "rms_norm_eps":0.000_001, "hidden_act":"silu"
                }))
                .expect("the config parses");
            let mut spec = config.to_spec().expect("the config is a decoder spec");
            spec.architecture = architecture.to_owned();
            spec
        }

        /// A template that keeps the reasoning block closed unless the option
        /// asks for it, so its own default is off.
        const DEFAULT_OFF: &str = "{% if enable_thinking is defined and enable_thinking %}<think>{% endif %}{{ messages[0].content }}";

        #[test]
        fn the_real_template_fixtures_report_their_own_default() {
            for (name, expected) in [
                ("qwen35-unsloth", Some(false)),
                ("qwen35-official", Some(true)),
                ("qwen3-current", Some(true)),
                ("qwen3-gguf", Some(true)),
            ] {
                let source = fixture(name);
                let fallback = compile_fallback(&source);
                assert_eq!(
                    fallback.chat_template_defaults().enable_thinking,
                    expected,
                    "{name}"
                );
            }
        }

        #[test]
        fn the_qwen35_fixtures_agree_with_the_pinned_adapter() -> Result<()> {
            for (name, expected) in [
                ("qwen35-official", Some(true)),
                ("qwen35-unsloth", Some(false)),
            ] {
                let source = fixture(name);
                let generic = compile_fallback(&source)
                    .chat_template_defaults()
                    .enable_thinking;
                let pinned = Qwen35Adapter::from_decoder_spec(spec("qwen35"), Some(&source))?
                    .chat_template_defaults()
                    .enable_thinking;
                assert_eq!(generic, expected, "{name}");
                assert_eq!(pinned, expected, "{name}");
                assert_eq!(generic, pinned, "{name}");
            }
            Ok(())
        }

        #[test]
        fn the_adapters_report_the_default_of_their_own_template() -> Result<()> {
            let tokens = GgufPromptTokens {
                chat_template: Some(DEFAULT_OFF),
                ..Default::default()
            };
            // The shared fallback of a checkpoint without a pinned template.
            let fallback = with_jinja_fallback(
                Box::new(Qwen3Adapter::from_decoder_spec(spec("qwen3"), None)?),
                tokens,
            );
            assert_eq!(
                fallback.chat_template_defaults().enable_thinking,
                Some(false)
            );
            // An unpinned Qwen3 adapter.
            let mut adapter = Qwen3Adapter::from_decoder_spec(spec("qwen3"), Some(DEFAULT_OFF))?;
            adapter.attach_prompt_tokens(tokens)?;
            assert_eq!(
                adapter.chat_template_defaults().enable_thinking,
                Some(false)
            );
            // An unpinned Qwen3 MoE adapter.
            let mut adapter =
                Qwen3MoeAdapter::from_decoder_spec(spec("qwen3moe"), Some(DEFAULT_OFF))?;
            adapter.attach_prompt_tokens(tokens)?;
            assert_eq!(
                adapter.chat_template_defaults().enable_thinking,
                Some(false)
            );
            // An unpinned Qwen3.5 adapter.
            let mut adapter = Qwen35Adapter::from_decoder_spec(spec("qwen35"), Some(DEFAULT_OFF))?;
            adapter.attach_prompt_tokens(tokens)?;
            assert_eq!(
                adapter.chat_template_defaults().enable_thinking,
                Some(false)
            );
            Ok(())
        }

        #[test]
        fn the_pinned_text_subset_reports_no_default() -> Result<()> {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/chat_templates/templates/pinned")
                .join("qwen35-official-text-subset.jinja");
            let source = std::fs::read_to_string(&path)?;
            let adapter = Qwen35Adapter::from_decoder_spec(spec("qwen35"), Some(&source))?;
            assert_eq!(adapter.chat_template_defaults().enable_thinking, None);
            Ok(())
        }

        #[test]
        fn a_template_that_does_not_compile_reports_no_default() {
            let fallback = jinja_fallback(
                "test",
                GenericPaths {
                    chat: true,
                    agent: true,
                },
                GgufPromptTokens {
                    chat_template: Some("{% for message in messages %}"),
                    ..Default::default()
                },
            )
            .expect("a failed compilation is kept for the render");
            assert_eq!(fallback.chat_template_defaults().enable_thinking, None);
        }

        #[test]
        fn a_render_without_the_option_leaves_it_out() -> Result<()> {
            // The template reads as default off in the probe, and it shows
            // the difference only with a longer history: the option is
            // defined, so the history prints its marker.
            const SOURCE: &str = "{% if enable_thinking is defined and enable_thinking %}on{% else %}off{% endif %}{% if enable_thinking is defined and messages|length > 1 %}X{% endif %}";
            let fallback = compile_fallback(SOURCE);
            assert_eq!(
                fallback.chat_template_defaults().enable_thinking,
                Some(false)
            );
            // The adapter of a checkpoint without a pinned template renders
            // the request without the option, so the template applies its
            // own default on every turn.
            let adapter = with_jinja_fallback(
                Box::new(Qwen3Adapter::from_decoder_spec(spec("qwen3"), None)?),
                GgufPromptTokens {
                    chat_template: Some(SOURCE),
                    ..Default::default()
                },
            );
            let messages = [ChatMessage::system("be brief"), ChatMessage::user("hello")];
            let rendered = adapter.format_chat_messages(&messages)?;
            assert!(!rendered.contains('X'), "{rendered}");
            assert!(rendered.contains("off"), "{rendered}");
            Ok(())
        }
    }

    #[test]
    fn detects_all_qwen_template_fixtures() {
        for (name, expected) in [
            ("qwen3-current", ToolCallFormat::Hermes),
            ("qwen3-gguf", ToolCallFormat::Hermes),
            ("qwen35-official", ToolCallFormat::QwenXml),
            ("qwen35-unsloth", ToolCallFormat::QwenXml),
        ] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/chat_templates/templates")
                .join(format!("{name}.jinja"));
            assert_eq!(
                tool_format::detect(&std::fs::read_to_string(path).unwrap()),
                Some(expected)
            );
        }
    }
}
