use anyhow::{Result, bail, ensure};
use sha2::{Digest, Sha256};

use crate::{
    architecture::{DecoderSpec, RuntimeSpec},
    chat::{ChatMessage, ChatRole, validate_chat_messages},
    decoder::{
        DecoderLayerWeightNames, DecoderModel, DecoderWeightNames, STANDARD_DECODER_WEIGHT_NAMES,
    },
    model_weights::WeightSource,
    models::ModelAdapter,
};

pub(crate) const QWEN35_WEIGHT_NAMES: DecoderWeightNames = DecoderWeightNames {
    layer: DecoderLayerWeightNames {
        mixer_norm: "attn_norm",
        feed_forward_norm: "attn_post_norm",
        gated_delta: Some("linear_attn"),
        ..STANDARD_DECODER_WEIGHT_NAMES.layer
    },
    ..STANDARD_DECODER_WEIGHT_NAMES
};

// `tokenizer.chat_template` embedded in ggml-org/Qwen3.5-0.8B-GGUF at
// revision 8fea620810c4afa23dd6443f999a48574c1611a3.
const OFFICIAL_GGUF_CHAT_TEMPLATE_LEN: usize = 7_755;
const OFFICIAL_GGUF_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x27, 0x3d, 0x8e, 0x0e, 0x68, 0x3b, 0x88, 0x50, 0x71, 0xfb, 0x17, 0xe0, 0x8d, 0x71, 0xe5, 0xf2,
    0xa5, 0xdd, 0xfb, 0x53, 0x09, 0x75, 0x61, 0x81, 0x68, 0x1d, 0xe4, 0xf5, 0xa1, 0x82, 0x2d, 0x80,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen35ChatTemplate {
    OfficialTextSubset,
}

fn classify_chat_template(template: Option<&str>) -> Option<Qwen35ChatTemplate> {
    let template = template?;
    if template.len() != OFFICIAL_GGUF_CHAT_TEMPLATE_LEN {
        return None;
    }
    let digest: [u8; 32] = Sha256::digest(template.as_bytes()).into();
    (digest == OFFICIAL_GGUF_CHAT_TEMPLATE_SHA256).then_some(Qwen35ChatTemplate::OfficialTextSubset)
}

/// Adapter for the dense Qwen3.5 text trunk stored in canonical llama.cpp
/// GGUF form.
///
/// The exact official template is recognized by digest. Its plain-text
/// system/user/assistant subset is rendered locally; the common chat contract
/// cannot express the template's multimodal and tool branches, so those remain
/// rejected before reaching this adapter.
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
        match self.chat_template {
            Some(Qwen35ChatTemplate::OfficialTextSubset) => format_qwen35_chat_messages(messages),
            None => bail!(
                "unsupported or missing Qwen3.5 chat template; use `run --raw-prompt` or the pinned official Qwen3.5 GGUF template"
            ),
        }
    }
}

fn format_qwen35_chat_messages(messages: &[ChatMessage]) -> Result<String> {
    validate_chat_messages(messages)?;
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(message.role.as_str());
        prompt.push('\n');
        let content = message.content.trim();
        if message.role == ChatRole::Assistant {
            // The official template removes historical reasoning before the
            // next user turn. The common contract always ends in user, so all
            // assistant messages reaching this branch are historical.
            prompt.push_str(
                content
                    .rsplit_once("</think>")
                    .map_or(content, |(_, answer)| answer.trim_start_matches('\n')),
            );
        } else {
            prompt.push_str(content);
        }
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    Ok(prompt)
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
    fn unknown_template_stays_raw_only() {
        assert_eq!(classify_chat_template(None), None);
        assert_eq!(classify_chat_template(Some("unverified")), None);
    }
}
