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

use std::{fs, path::Path};

use anyhow::{Context, Result, bail, ensure};
use candle_nn::VarBuilder;
use serde::Deserialize;

use macllm_chat::jinja::JinjaChatTemplate;

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

    /// Return the exact native agent-template profile verified for this
    /// checkpoint, or `None` when tool-aware prompt rendering is unavailable.
    fn agent_template_profile(&self) -> Option<AgentTemplateProfile> {
        None
    }

    /// Render a validated protocol-independent agent prompt.  Adapters must
    /// opt in with a pinned profile; the default fails closed.
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
}

impl JinjaChatFallback {
    /// Render a validated chat history through the embedded template.
    ///
    /// Tool-aware prompts stay on the pinned native profiles; the generic
    /// renderer only serves ordinary text conversations.
    pub(crate) fn format_chat_messages(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        validate_chat_messages(messages)?;
        self.template
            .as_ref()
            .map_err(|error| anyhow::anyhow!("{error}"))?
            .render(messages, None, &options, true)
            .with_context(|| {
                format!(
                    "the {} chat template could not render the conversation",
                    self.family
                )
            })
    }
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
pub(crate) fn jinja_chat_fallback(
    family: &'static str,
    pinned: bool,
    tokens: GgufPromptTokens<'_>,
) -> Option<JinjaChatFallback> {
    if pinned {
        log_chat_template_path(family, "pinned");
        return None;
    }
    let Some(source) = tokens.chat_template else {
        log_chat_template_path(family, "none (structured chat needs a raw prompt)");
        return None;
    };
    let template = JinjaChatTemplate::new(
        source,
        tokens.bos_token.unwrap_or_default(),
        tokens.eos_token.unwrap_or_default(),
    )
    .map_err(|error| format!("the {family} chat template is not a valid Jinja program: {error:#}"));
    log_chat_template_path(family, "jinja");
    Some(JinjaChatFallback { family, template })
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
    let fallback =
        jinja_chat_fallback(adapter.family(), adapter.has_pinned_chat_template(), tokens);
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
        self.fallback
            .format_chat_messages(messages, self.chat_template_defaults())
    }
    fn format_chat_messages_with(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        self.fallback.format_chat_messages(messages, options)
    }
    fn chat_template_defaults(&self) -> ChatTemplateOptions {
        self.adapter.chat_template_defaults()
    }
    fn agent_template_profile(&self) -> Option<AgentTemplateProfile> {
        self.adapter.agent_template_profile()
    }
    fn format_agent_prompt(&self, prompt: &AgentPrompt) -> Result<String> {
        self.adapter.format_agent_prompt(prompt)
    }
    fn format_agent_prompt_with(
        &self,
        prompt: &AgentPrompt,
        options: ChatTemplateOptions,
    ) -> Result<String> {
        self.adapter.format_agent_prompt_with(prompt, options)
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
