use anyhow::{Result, bail, ensure};

use crate::{
    architecture::{DecoderSpec, RuntimeSpec},
    chat::{ChatMessage, ChatRole, validate_chat_messages},
    decoder::DecoderModel,
    model_weights::WeightSource,
    models::ModelAdapter,
};

// Gemma 1 GGUFs exist with both official tokenizer-template revisions: the
// original relies on model tokenization to add BOS, while newer conversions
// embed `bos_token` in Jinja. MacLLM tokenizes rendered chat with automatic
// special-token insertion disabled, so both exact identities lower to one
// renderer that emits `<bos>` exactly once.
const OFFICIAL_CHAT_TEMPLATE: &str = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}";
const OFFICIAL_LEGACY_CHAT_TEMPLATE: &str = "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmaChatTemplate {
    Official,
}

fn classify_chat_template(template: Option<&str>) -> Option<GemmaChatTemplate> {
    matches!(
        template?,
        OFFICIAL_CHAT_TEMPLATE | OFFICIAL_LEGACY_CHAT_TEMPLATE
    )
    .then_some(GemmaChatTemplate::Official)
}

/// GGUF-backed Gemma 1 adapter.
///
/// Gemma's full causal attention, pre-norm decoder, gated GELU feed-forward
/// block, embedding scale and tied output head all lower to the shared
/// [`DecoderSpec`]. This type owns only family identity and prompt semantics.
#[derive(Debug, Clone)]
pub struct GemmaAdapter {
    spec: DecoderSpec,
    chat_template: Option<GemmaChatTemplate>,
}

impl GemmaAdapter {
    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "gemma",
            "expected a gemma decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        Ok(Self {
            spec,
            chat_template: classify_chat_template(chat_template),
        })
    }
}

impl ModelAdapter for GemmaAdapter {
    fn family(&self) -> &'static str {
        "gemma"
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
            Some(GemmaChatTemplate::Official) => format_gemma_chat_messages(messages),
            None => bail!(
                "unsupported or missing Gemma chat template; pass --raw-prompt or use the official Gemma instruction template"
            ),
        }
    }
}

fn format_gemma_chat_messages(messages: &[ChatMessage]) -> Result<String> {
    validate_chat_messages(messages)?;
    ensure!(
        !matches!(messages.first(), Some(message) if message.role == ChatRole::System),
        "Gemma chat does not support a system message"
    );

    let mut prompt = String::from("<bos>");
    for message in messages {
        let role = match message.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "model",
            ChatRole::System => unreachable!("system messages were rejected above"),
            ChatRole::Tool => unreachable!("chat history validation rejected a tool response"),
        };
        prompt.push_str("<start_of_turn>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(message.content.trim());
        prompt.push_str("<end_of_turn>\n");
    }
    prompt.push_str("<start_of_turn>model\n");
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::{
        Activation, AttentionKind, AttentionSpec, DecoderLayerSpec, FeedForwardKind, NormLayout,
        PositionEncodingSpec, ProjectionBias, RopeLayout, RopeSpec, SequenceMixerSpec,
    };

    #[test]
    fn unknown_gguf_template_uses_shared_fallback() -> Result<()> {
        let source = "{{ bos_token }}{{ messages[0].content }}{{ eos_token }}";
        let adapter = GemmaAdapter::from_decoder_spec(tiny_spec(), Some(source))?;
        let adapter = crate::models::with_jinja_fallback(
            Box::new(adapter),
            crate::models::GgufPromptTokens {
                chat_template: Some(source),
                bos_token: Some("<bos>"),
                eos_token: Some("<eos>"),
            },
        );
        assert_eq!(
            adapter.format_chat_messages(&[ChatMessage::user("hello")])?,
            "<bos>hello<eos>"
        );
        assert_eq!(
            adapter.format_chat_messages_with(
                &[ChatMessage::user("hello")],
                crate::chat::ChatTemplateOptions {
                    enable_thinking: Some(false)
                },
            )?,
            "<bos>hello<eos>"
        );
        let prompt = crate::agent::AgentPrompt {
            messages: vec![crate::agent::AgentMessage::User {
                content: "hello".into(),
            }],
            tools: vec![],
            directive: crate::agent::AgentDirective::TextOnly,
        };
        assert!(adapter.agent_template_profile().is_none());
        assert!(adapter.format_agent_prompt(&prompt).is_err());
        assert!(
            adapter
                .format_agent_prompt_with(&prompt, crate::chat::ChatTemplateOptions::default())
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn invalid_jinja_does_not_prevent_adapter_construction() -> Result<()> {
        let source = "{% for message in messages %}";
        let adapter = GemmaAdapter::from_decoder_spec(tiny_spec(), Some(source))?;
        let adapter = crate::models::with_jinja_fallback(
            Box::new(adapter),
            crate::models::GgufPromptTokens {
                chat_template: Some(source),
                ..Default::default()
            },
        );
        adapter.validate()?;
        let error = adapter
            .format_chat_messages(&[ChatMessage::user("hello")])
            .unwrap_err();
        assert!(format!("{error:#}").contains("not a valid Jinja program"));
        Ok(())
    }

    #[test]
    fn recognized_template_keeps_its_pinned_renderer() -> Result<()> {
        let adapter = GemmaAdapter::from_decoder_spec(tiny_spec(), Some(OFFICIAL_CHAT_TEMPLATE))?;
        let messages = [ChatMessage::user(" hello ")];
        let expected = adapter.format_chat_messages(&messages)?;
        let adapter = crate::models::with_jinja_fallback(
            Box::new(adapter),
            crate::models::GgufPromptTokens {
                chat_template: Some(OFFICIAL_CHAT_TEMPLATE),
                bos_token: Some("different BOS"),
                eos_token: Some("different EOS"),
            },
        );
        assert_eq!(adapter.format_chat_messages(&messages)?, expected);
        Ok(())
    }

    fn tiny_spec() -> DecoderSpec {
        DecoderSpec {
            architecture: "gemma".into(),
            runtime: RuntimeSpec {
                vocab_size: 128,
                max_context: 32,
                stop_token_ids: vec![1, 107],
            },
            hidden_size: 8,
            intermediate_size: 16,
            num_layers: 1,
            num_query_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            norm_epsilon: 1e-6,
            layer_specs: vec![DecoderLayerSpec {
                mixer: SequenceMixerSpec::Attention(AttentionSpec {
                    kind: AttentionKind::FullCausal,
                    position_encoding: PositionEncodingSpec::Rope(RopeSpec {
                        theta: 10_000.0,
                        layout: RopeLayout::HalfSplit,
                        rotary_dim: 4,
                        frequency_factors: None,
                    }),
                    qk_norm: false,
                    output_gate: false,
                    scale: 0.5,
                    logit_softcap: None,
                }),
                activation: Activation::GeluTanh,
                norm_layout: NormLayout::PreNorm,
                feed_forward: FeedForwardKind::Dense,
                mixer_residual_scale: 1.0,
                feed_forward_residual_scale: 1.0,
            }],
            embedding_scale: 8.0_f64.sqrt(),
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
    fn accepts_only_the_exact_official_template_and_inserts_model_bos() -> Result<()> {
        for template in [OFFICIAL_CHAT_TEMPLATE, OFFICIAL_LEGACY_CHAT_TEMPLATE] {
            let adapter = GemmaAdapter::from_decoder_spec(tiny_spec(), Some(template))?;
            assert_eq!(adapter.family(), "gemma");
            assert_eq!(
                adapter.format_chat_messages(&[
                    ChatMessage::user("  First question.  "),
                    ChatMessage::assistant(" First answer. "),
                    ChatMessage::user(" Follow-up. "),
                ])?,
                "<bos><start_of_turn>user\nFirst question.<end_of_turn>\n<start_of_turn>model\nFirst answer.<end_of_turn>\n<start_of_turn>user\nFollow-up.<end_of_turn>\n<start_of_turn>model\n"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_system_messages_unknown_templates_and_wrong_family() -> Result<()> {
        let adapter = GemmaAdapter::from_decoder_spec(tiny_spec(), Some(OFFICIAL_CHAT_TEMPLATE))?;
        assert!(adapter.format_chat(Some("Be concise."), "Hello").is_err());

        for template in [None, Some("{{ unknown }}")] {
            let adapter = GemmaAdapter::from_decoder_spec(tiny_spec(), template)?;
            assert!(adapter.format_chat(None, "Hello").is_err());
        }

        let mut wrong = tiny_spec();
        wrong.architecture = "gemma2".into();
        assert!(GemmaAdapter::from_decoder_spec(wrong, None).is_err());
        Ok(())
    }
}
