use anyhow::{Result, bail, ensure};

use crate::{
    architecture::{DecoderSpec, RuntimeSpec},
    chat::{ChatMessage, ChatRole, validate_chat_messages},
    decoder::DecoderModel,
    model_weights::WeightSource,
    models::ModelAdapter,
};

// Exact template embedded by the official Gemma 2 instruction tokenizer and
// canonical llama.cpp GGUF conversions. Keep this identity check strict: raw
// prompt inference remains available for fine-tunes with another template.
const OFFICIAL_CHAT_TEMPLATE: &str = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma2ChatTemplate {
    Official,
}

fn classify_chat_template(template: Option<&str>) -> Option<Gemma2ChatTemplate> {
    (template? == OFFICIAL_CHAT_TEMPLATE).then_some(Gemma2ChatTemplate::Official)
}

/// GGUF-backed Gemma 2 adapter.
///
/// The GGUF loader lowers alternating local/global attention, sandwich norms,
/// soft caps and embedding scaling into the shared [`DecoderSpec`]. This type
/// owns only the family identity and exact prompt convention.
#[derive(Debug, Clone)]
pub struct Gemma2Adapter {
    spec: DecoderSpec,
    chat_template: Option<Gemma2ChatTemplate>,
}

impl Gemma2Adapter {
    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "gemma2",
            "expected a gemma2 decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        Ok(Self {
            spec,
            chat_template: classify_chat_template(chat_template),
        })
    }
}

impl ModelAdapter for Gemma2Adapter {
    fn family(&self) -> &'static str {
        "gemma2"
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
            Some(Gemma2ChatTemplate::Official) => format_gemma2_chat_messages(messages),
            None => bail!(
                "unsupported or missing Gemma 2 chat template; pass --raw-prompt or use the official Gemma 2 instruction template"
            ),
        }
    }
}

fn format_gemma2_chat_messages(messages: &[ChatMessage]) -> Result<String> {
    validate_chat_messages(messages)?;
    ensure!(
        !matches!(messages.first(), Some(message) if message.role == ChatRole::System),
        "Gemma 2 chat does not support a system message"
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

    fn tiny_spec() -> DecoderSpec {
        DecoderSpec {
            architecture: "gemma2".into(),
            runtime: RuntimeSpec {
                vocab_size: 16,
                max_context: 32,
                stop_token_ids: vec![1, 7],
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
                    kind: AttentionKind::SlidingCausal { window: 8 },
                    position_encoding: PositionEncodingSpec::Rope(RopeSpec {
                        theta: 10_000.0,
                        layout: RopeLayout::HalfSplit,
                        rotary_dim: 4,
                        frequency_factors: None,
                    }),
                    qk_norm: false,
                    output_gate: false,
                    scale: 0.5,
                    logit_softcap: Some(50.0),
                }),
                activation: Activation::GeluTanh,
                norm_layout: NormLayout::Sandwich,
                feed_forward: FeedForwardKind::Dense,
                mixer_residual_scale: 1.0,
                feed_forward_residual_scale: 1.0,
            }],
            embedding_scale: 8.0_f64.sqrt(),
            final_logit_softcap: Some(30.0),
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

    /// The pinned template source still classifies to its variant.
    ///
    /// The fixture file holds the exact source of the pinned variant, and the
    /// classifier recognises it by string equality.
    #[test]
    fn the_pinned_template_source_still_classifies_to_its_variant() {
        let text = crate::models::template_parity::pinned_text("gemma2-official.jinja");
        assert_eq!(text, OFFICIAL_CHAT_TEMPLATE);
        assert_eq!(
            classify_chat_template(Some(&text)),
            Some(Gemma2ChatTemplate::Official),
        );
    }

    #[test]
    fn accepts_only_the_exact_official_template_and_renders_turns() -> Result<()> {
        let adapter = Gemma2Adapter::from_decoder_spec(tiny_spec(), Some(OFFICIAL_CHAT_TEMPLATE))?;
        assert_eq!(adapter.family(), "gemma2");
        assert_eq!(
            adapter.format_chat_messages(&[
                ChatMessage::user("  First question.  "),
                ChatMessage::assistant(" First answer. "),
                ChatMessage::user(" Follow-up. "),
            ])?,
            "<bos><start_of_turn>user\nFirst question.<end_of_turn>\n<start_of_turn>model\nFirst answer.<end_of_turn>\n<start_of_turn>user\nFollow-up.<end_of_turn>\n<start_of_turn>model\n"
        );
        Ok(())
    }

    #[test]
    fn rejects_system_messages_unknown_templates_and_wrong_family() -> Result<()> {
        let adapter = Gemma2Adapter::from_decoder_spec(tiny_spec(), Some(OFFICIAL_CHAT_TEMPLATE))?;
        assert!(adapter.format_chat(Some("Be concise."), "Hello").is_err());

        for template in [None, Some("{{ unknown }}")] {
            let adapter = Gemma2Adapter::from_decoder_spec(tiny_spec(), template)?;
            assert!(adapter.format_chat(None, "Hello").is_err());
        }

        let mut wrong = tiny_spec();
        wrong.architecture = "gemma3".into();
        assert!(Gemma2Adapter::from_decoder_spec(wrong, None).is_err());
        Ok(())
    }
}
