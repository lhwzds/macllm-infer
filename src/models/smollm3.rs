use anyhow::{Result, bail, ensure};
use sha2::{Digest, Sha256};

use crate::{
    architecture::{DecoderSpec, RuntimeSpec},
    chat::{ChatMessage, ChatRole, validate_chat_messages},
    decoder::DecoderModel,
    model_weights::WeightSource,
    models::ModelAdapter,
};

// `tokenizer.chat_template` embedded in ggml-org/SmolLM3-3B-GGUF at revision
// 4965cb60b150737b68a0408c36aeefb65078f894. The template contains a
// process-clock-dependent default system branch, so an exact identity match
// is necessary but not sufficient for local rendering: only its deterministic
// `/system_override` text-role subset is implemented.
const OFFICIAL_GGUF_CHAT_TEMPLATE_LEN: usize = 5_493;
const OFFICIAL_GGUF_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0xb9, 0xb6, 0x6f, 0x04, 0xc6, 0x4f, 0xbb, 0x86, 0x95, 0xcf, 0x5b, 0x35, 0xc3, 0x77, 0x80, 0xef,
    0xd0, 0xb8, 0xe0, 0x82, 0x9f, 0xbf, 0xe3, 0xe3, 0x0f, 0xaf, 0xb9, 0xf4, 0x69, 0xb7, 0xd3, 0x0e,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmolLm3ChatTemplate {
    OfficialSystemOverrideTextSubset,
}

fn classify_chat_template(template: Option<&str>) -> Option<SmolLm3ChatTemplate> {
    let template = template?;
    if template.len() != OFFICIAL_GGUF_CHAT_TEMPLATE_LEN {
        return None;
    }
    let digest: [u8; 32] = Sha256::digest(template.as_bytes()).into();
    classify_chat_template_identity(template.len(), digest)
}

fn classify_chat_template_identity(
    byte_len: usize,
    digest: [u8; 32],
) -> Option<SmolLm3ChatTemplate> {
    (byte_len == OFFICIAL_GGUF_CHAT_TEMPLATE_LEN && digest == OFFICIAL_GGUF_CHAT_TEMPLATE_SHA256)
        .then_some(SmolLm3ChatTemplate::OfficialSystemOverrideTextSubset)
}

/// Adapter for the dense `SmolLM3` text decoder stored in canonical llama.cpp
/// GGUF form.
///
/// Model geometry and tensor validation are lowered into [`DecoderSpec`] by
/// the GGUF loader. This adapter owns the family boundary and the exact,
/// deterministic text-only subset of the pinned tokenizer template.
#[derive(Debug, Clone)]
pub struct SmolLm3Adapter {
    spec: DecoderSpec,
    chat_template: Option<SmolLm3ChatTemplate>,
}

impl SmolLm3Adapter {
    /// Construct an adapter from metadata already validated by the GGUF
    /// loader, which attaches generic Jinja rendering for unknown templates.
    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "smollm3",
            "expected a smollm3 decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        Ok(Self {
            spec,
            chat_template: classify_chat_template(chat_template),
        })
    }
}

impl ModelAdapter for SmolLm3Adapter {
    fn family(&self) -> &'static str {
        "smollm3"
    }

    fn has_pinned_chat_template(&self) -> bool {
        self.chat_template.is_some()
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
        match self.chat_template {
            Some(SmolLm3ChatTemplate::OfficialSystemOverrideTextSubset) => {
                format_smollm3_chat_messages(messages)
            }
            None => bail!(
                "unsupported or missing SmolLM3 chat template; use `run --raw-prompt` or the pinned official SmolLM3 GGUF template"
            ),
        }
    }

    fn format_serving_chat_probe(&self) -> Result<String> {
        self.format_chat_messages(&[
            ChatMessage::system("/system_override/no_think"),
            ChatMessage::user("health check"),
        ])
    }
}

fn format_smollm3_chat_messages(messages: &[ChatMessage]) -> Result<String> {
    validate_chat_messages(messages)?;
    let Some(ChatMessage {
        role: ChatRole::System,
        content: system_message,
        ..
    }) = messages.first()
    else {
        bail!(
            "SmolLM3 structured chat requires an explicit `/system_override` system message because the official default template uses the dynamic current date; use `run --raw-prompt` otherwise"
        );
    };
    ensure!(
        system_message.contains("/system_override"),
        "SmolLM3 structured chat requires `/system_override` because the official default template uses the dynamic current date; use `run --raw-prompt` otherwise"
    );

    // Match the official Jinja ordering exactly: `/no_think` wins when both
    // flags appear, then all occurrences of both mode flags are removed.
    let thinking = if system_message.contains("/no_think") {
        false
    } else {
        // `/think` explicitly selects the same mode as the template default.
        true
    };
    let custom_instructions = system_message
        .replace("/no_think", "")
        .replace("/think", "");
    let custom_instructions = custom_instructions.trim_end();
    let system_override = custom_instructions.replace("/system_override", "");
    let system_override = system_override.trim_end();

    let mut prompt = String::from("<|im_start|>system\n");
    prompt.push_str(system_override);
    prompt.push_str("<|im_end|>\n");
    for message in &messages[1..] {
        prompt.push_str("<|im_start|>");
        prompt.push_str(message.role.as_str());
        prompt.push('\n');
        if message.role == ChatRole::Assistant {
            if !thinking {
                prompt.push_str("<think>\n\n</think>\n");
            }
            prompt.push_str(message.content.trim_start_matches('\n'));
        } else {
            prompt.push_str(&message.content);
        }
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    if !thinking {
        prompt.push_str("<think>\n\n</think>\n");
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::{DecoderLayerSpec, ProjectionBias, RopeLayout, RopeSpec};

    fn tiny_spec() -> DecoderSpec {
        DecoderSpec {
            architecture: "smollm3".into(),
            runtime: RuntimeSpec {
                vocab_size: 32,
                max_context: 64,
                stop_token_ids: vec![1, 2],
            },
            hidden_size: 8,
            intermediate_size: 16,
            num_layers: 1,
            num_query_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            norm_epsilon: 1e-5,
            layer_specs: vec![DecoderLayerSpec::standard(
                RopeSpec {
                    theta: 10_000.0,
                    layout: RopeLayout::HalfSplit,
                    rotary_dim: 4,
                    frequency_factors: None,
                },
                4,
            )],
            embedding_scale: 1.0,
            final_logit_softcap: None,
            final_logit_scale: 1.0,
            projection_bias: ProjectionBias {
                query: false,
                key: false,
                value: false,
                output: false,
            },
            mlp_bias: false,
            tied_embeddings: true,
        }
    }

    #[test]
    fn exposes_smollm3_shared_decoder_boundary_and_rejects_wrong_family() -> Result<()> {
        let adapter = SmolLm3Adapter::from_decoder_spec(tiny_spec(), None)?;
        assert_eq!(adapter.family(), "smollm3");
        assert_eq!(adapter.decoder_spec(), Some(&tiny_spec()));

        let mut wrong = tiny_spec();
        wrong.architecture = "llama".into();
        assert!(SmolLm3Adapter::from_decoder_spec(wrong, None).is_err());
        Ok(())
    }

    #[test]
    fn exact_template_identity_fails_closed_for_absent_modified_or_wrong_length_input() {
        assert_eq!(classify_chat_template(None), None);
        assert_eq!(classify_chat_template(Some("unverified")), None);
        let same_length_but_modified = "x".repeat(OFFICIAL_GGUF_CHAT_TEMPLATE_LEN);
        assert_eq!(
            classify_chat_template(Some(&same_length_but_modified)),
            None
        );
        assert_eq!(OFFICIAL_GGUF_CHAT_TEMPLATE_LEN, 5_493);
        assert_eq!(
            OFFICIAL_GGUF_CHAT_TEMPLATE_SHA256,
            [
                0xb9, 0xb6, 0x6f, 0x04, 0xc6, 0x4f, 0xbb, 0x86, 0x95, 0xcf, 0x5b, 0x35, 0xc3, 0x77,
                0x80, 0xef, 0xd0, 0xb8, 0xe0, 0x82, 0x9f, 0xbf, 0xe3, 0xe3, 0x0f, 0xaf, 0xb9, 0xf4,
                0x69, 0xb7, 0xd3, 0x0e,
            ]
        );
        assert_eq!(
            classify_chat_template_identity(
                OFFICIAL_GGUF_CHAT_TEMPLATE_LEN,
                OFFICIAL_GGUF_CHAT_TEMPLATE_SHA256
            ),
            Some(SmolLm3ChatTemplate::OfficialSystemOverrideTextSubset)
        );
        let mut modified_digest = OFFICIAL_GGUF_CHAT_TEMPLATE_SHA256;
        modified_digest[0] ^= 1;
        assert_eq!(
            classify_chat_template_identity(OFFICIAL_GGUF_CHAT_TEMPLATE_LEN, modified_digest),
            None
        );
    }

    #[test]
    fn renders_deterministic_system_override_thinking_subset() -> Result<()> {
        let prompt = format_smollm3_chat_messages(&[
            ChatMessage::system("Be concise./system_override/think"),
            ChatMessage::user("First?"),
            ChatMessage::assistant("\n<think>old reasoning</think>\nFirst."),
            ChatMessage::user("Second?"),
        ])?;
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nFirst?<|im_end|>\n<|im_start|>assistant\n<think>old reasoning</think>\nFirst.<|im_end|>\n<|im_start|>user\nSecond?<|im_end|>\n<|im_start|>assistant\n"
        );
        Ok(())
    }

    #[test]
    fn renders_no_think_prefix_and_gives_no_think_precedence() -> Result<()> {
        let prompt = format_smollm3_chat_messages(&[
            ChatMessage::system("Be terse./system_override/think/no_think"),
            ChatMessage::user("First?"),
            ChatMessage::assistant("\nFirst."),
            ChatMessage::user("Second?"),
        ])?;
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nFirst?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\nFirst.<|im_end|>\n<|im_start|>user\nSecond?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
        );
        Ok(())
    }

    #[test]
    fn serving_probe_uses_the_deterministic_template_subset() -> Result<()> {
        let adapter = SmolLm3Adapter {
            spec: tiny_spec(),
            chat_template: Some(SmolLm3ChatTemplate::OfficialSystemOverrideTextSubset),
        };
        assert_eq!(
            adapter.format_serving_chat_probe()?,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nhealth check<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
        );
        assert!(
            adapter
                .format_chat_messages(&[ChatMessage::user("health check")])
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_qwen3_template_options_fail_closed() {
        let adapter = SmolLm3Adapter {
            spec: tiny_spec(),
            chat_template: Some(SmolLm3ChatTemplate::OfficialSystemOverrideTextSubset),
        };
        let messages = [
            ChatMessage::system("/system_override"),
            ChatMessage::user("Hi"),
        ];
        for enabled in [false, true] {
            let options = crate::chat::ChatTemplateOptions {
                enable_thinking: Some(enabled),
            };
            let error = adapter
                .format_chat_messages_with(&messages, options)
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("not supported by the smollm3 adapter")
            );
        }
    }

    #[test]
    fn dynamic_date_and_unknown_template_paths_require_raw_prompt() -> Result<()> {
        for messages in [
            vec![ChatMessage::user("Hello")],
            vec![
                ChatMessage::system("Be concise. /think"),
                ChatMessage::user("Hello"),
            ],
        ] {
            let error = format_smollm3_chat_messages(&messages)
                .expect_err("dynamic-date branch must fail closed")
                .to_string();
            assert!(error.contains("dynamic current date"));
            assert!(error.contains("--raw-prompt"));
        }

        let adapter = SmolLm3Adapter::from_decoder_spec(tiny_spec(), Some("unknown"))?;
        let error = adapter
            .format_chat(Some("/system_override"), "Hello")
            .expect_err("unknown template must fail closed")
            .to_string();
        assert!(error.contains("--raw-prompt"));
        Ok(())
    }
}
