use anyhow::{Result, bail, ensure};

use crate::{
    agent::{AgentPrompt, AgentTemplateProfile},
    architecture::{DecoderSpec, RuntimeSpec},
    chat::{ChatMessage, ChatTemplateOptions},
    decoder::DecoderModel,
    model_weights::WeightSource,
    models::ModelAdapter,
};

// The prompt renderers live in the chat group so that the serving and
// benchmark layers can use them without depending on the model adapters.
pub use crate::chat_templates::qwen35::{
    OFFICIAL_CHAT_TEMPLATE_LEN, OFFICIAL_CHAT_TEMPLATE_SHA256, OFFICIAL_GGUF_CHAT_TEMPLATE_LEN,
    OFFICIAL_GGUF_CHAT_TEMPLATE_SHA256, Qwen35ChatTemplate, UNSLOTH_GGUF_CHAT_TEMPLATE_LEN,
    UNSLOTH_GGUF_CHAT_TEMPLATE_SHA256, classify_chat_template, format_qwen35_agent_prompt,
    format_qwen35_chat_messages, format_qwen35_plain,
};

// The shared decoder and the adapters read the same weight-name table, so it
// lives in the architecture module and is re-exported here.
pub(crate) use crate::architecture::QWEN35_WEIGHT_NAMES;

/// Adapter for the dense Qwen3.5 text trunk stored in canonical llama.cpp
/// GGUF form.
///
/// Exact template identities are recognized by digest. The two newer
/// identities support text and native tool history; the older GGUF identity
/// remains limited to its verified plain-text subset. Multimodal input is not
/// supported.
#[derive(Debug, Clone)]
pub struct Qwen35Adapter {
    spec: DecoderSpec,
    chat_template: Option<Qwen35ChatTemplate>,
}

impl Qwen35Adapter {
    /// Construct an adapter from the metadata and tensors already validated by
    /// the GGUF loader.
    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "qwen35",
            "expected a qwen35 decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        Ok(Self {
            spec,
            chat_template: classify_chat_template(chat_template),
        })
    }
}

impl ModelAdapter for Qwen35Adapter {
    fn family(&self) -> &'static str {
        "qwen35"
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
        Ok(Box::new(DecoderModel::load_with_weight_names(
            self.spec.clone(),
            weights,
            &QWEN35_WEIGHT_NAMES,
        )?))
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
            Some(Qwen35ChatTemplate::OfficialTextSubset) => {
                options.ensure_default(self.family())?;
                format_qwen35_chat_messages(messages)
            }
            Some(template @ (Qwen35ChatTemplate::Official | Qwen35ChatTemplate::Unsloth)) => {
                format_qwen35_plain(messages, template == Qwen35ChatTemplate::Official, options)
            }
            None => bail!(
                "unsupported or missing Qwen3.5 chat template; use `run --raw-prompt` or a pinned Qwen3.5 template"
            ),
        }
    }

    fn chat_template_defaults(&self) -> ChatTemplateOptions {
        ChatTemplateOptions {
            enable_thinking: match self.chat_template {
                Some(Qwen35ChatTemplate::Official) => Some(true),
                Some(Qwen35ChatTemplate::Unsloth) => Some(false),
                Some(Qwen35ChatTemplate::OfficialTextSubset) | None => None,
            },
        }
    }

    fn agent_template_profile(&self) -> Option<AgentTemplateProfile> {
        match self.chat_template {
            Some(Qwen35ChatTemplate::Official) => Some(AgentTemplateProfile::Qwen35Official),
            Some(Qwen35ChatTemplate::Unsloth) => Some(AgentTemplateProfile::Qwen35Unsloth),
            _ => None,
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
        let profile = self
            .agent_template_profile()
            .ok_or_else(|| anyhow::anyhow!("unsupported Qwen3.5 native agent template"))?;
        format_qwen35_agent_prompt(profile, prompt, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_official_plain_text_default_non_thinking_subset() -> Result<()> {
        let prompt = format_qwen35_chat_messages(&[
            ChatMessage::system("  Be concise.  "),
            ChatMessage::user(" First? "),
            ChatMessage::assistant("<think>old reasoning</think>\n\nFirst. "),
            ChatMessage::user("Second?"),
        ])?;
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nFirst?<|im_end|>\n<|im_start|>assistant\nFirst.<|im_end|>\n<|im_start|>user\nSecond?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            format_qwen35_chat_messages(&[ChatMessage::user("What is the capital of France?")])?,
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        Ok(())
    }

    #[test]
    fn recognizes_exact_new_identities() {
        for (name, expected) in [
            ("official", Qwen35ChatTemplate::Official),
            ("unsloth", Qwen35ChatTemplate::Unsloth),
        ] {
            let path = format!(
                "{}/tests/fixtures/chat_templates/templates/qwen35-{name}.jinja",
                env!("CARGO_MANIFEST_DIR")
            );
            let mut bytes = std::fs::read(path).unwrap();
            assert_eq!(
                classify_chat_template(Some(std::str::from_utf8(&bytes).unwrap())),
                Some(expected)
            );
            bytes[0] ^= 1;
            assert_eq!(
                classify_chat_template(Some(std::str::from_utf8(&bytes).unwrap())),
                None
            );
        }
    }

    #[test]
    fn unknown_template_stays_raw_only() {
        assert_eq!(classify_chat_template(None), None);
        assert_eq!(classify_chat_template(Some("unverified")), None);
    }
}
