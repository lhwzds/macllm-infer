use anyhow::{Result, bail, ensure};

use crate::{
    agent::{AgentPrompt, AgentTemplateProfile},
    architecture::{DecoderSpec, RuntimeSpec},
    chat::{ChatMessage, ChatTemplateOptions},
    decoder::DecoderModel,
    model_weights::WeightSource,
    models::{
        ModelAdapter,
        qwen3::{
            Qwen3ChatTemplate, classify_chat_template, format_qwen3_agent_prompt,
            format_qwen3_chat_messages,
        },
    },
};

/// GGUF-backed Qwen3 routed mixture-of-experts adapter.
///
/// `Qwen3MoE` combines the same explicit-width, Q/K-normalized attention used by
/// dense Qwen3 with the shared decoder's selected-softmax sparse-MoE block.
/// Family-specific code therefore ends at metadata, tensor and prompt
/// translation; execution remains in [`DecoderModel`].
#[derive(Debug, Clone)]
pub struct Qwen3MoeAdapter {
    spec: DecoderSpec,
    chat_template: Option<Qwen3ChatTemplate>,
}

impl Qwen3MoeAdapter {
    /// Construct an adapter from an already validated GGUF decoder spec.
    /// Unknown templates deliberately leave raw-prompt inference available
    /// while disabling structured chat rendering.
    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "qwen3moe",
            "expected a qwen3moe decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        Ok(Self {
            spec,
            chat_template: classify_chat_template(chat_template),
        })
    }
}

impl ModelAdapter for Qwen3MoeAdapter {
    fn family(&self) -> &'static str {
        "qwen3moe"
    }

    fn runtime_spec(&self) -> &RuntimeSpec {
        &self.spec.runtime
    }

    fn decoder_spec(&self) -> Option<&DecoderSpec> {
        Some(&self.spec)
    }

    fn build_model(
        &self,
        weights: &dyn WeightSource,
    ) -> candle_core::Result<Box<dyn crate::architecture::CausalLm>> {
        Ok(Box::new(DecoderModel::load(self.spec.clone(), weights)?))
    }

    fn format_chat_messages(&self, messages: &[ChatMessage]) -> Result<String> {
        self.format_chat_messages_with(messages, ChatTemplateOptions::default())
    }

    fn format_chat_messages_with(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        match self.chat_template {
            Some(Qwen3ChatTemplate::Current | Qwen3ChatTemplate::Gguf) => {
                format_qwen3_chat_messages(messages, options)
            }
            None => bail!(
                "unsupported or missing Qwen3MoE chat template; pass --raw-prompt or use the pinned official Qwen3 template"
            ),
        }
    }

    fn agent_template_profile(&self) -> Option<AgentTemplateProfile> {
        match self.chat_template {
            Some(Qwen3ChatTemplate::Current) => Some(AgentTemplateProfile::Qwen3Current),
            Some(Qwen3ChatTemplate::Gguf) => Some(AgentTemplateProfile::Qwen3Gguf),
            None => None,
        }
    }

    fn format_agent_prompt(&self, prompt: &AgentPrompt) -> Result<String> {
        self.format_agent_prompt_with(prompt, ChatTemplateOptions::default())
    }

    fn format_agent_prompt_with(
        &self,
        prompt: &AgentPrompt,
        options: ChatTemplateOptions,
    ) -> Result<String> {
        let Some(profile) = self.agent_template_profile() else {
            bail!("unsupported or missing Qwen3MoE native agent template profile");
        };
        format_qwen3_agent_prompt(profile, prompt, options)
    }
}
