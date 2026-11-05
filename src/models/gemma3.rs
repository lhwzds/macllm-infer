use anyhow::{Result, bail, ensure};
use sha2::{Digest, Sha256};

use crate::{
    architecture::{DecoderSpec, RuntimeSpec},
    chat::{ChatMessage, ChatRole, validate_chat_messages},
    decoder::DecoderModel,
    model_weights::WeightSource,
    models::ModelAdapter,
};

const GEMMA3_CHAT_TEMPLATE_LEN: usize = 1_532;
const GEMMA3_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x7d, 0xe1, 0xc5, 0x8e, 0x20, 0x8e, 0xda, 0x46, 0xe9, 0xc7, 0xf8, 0x63, 0x97, 0xdf, 0x37, 0xec,
    0x49, 0x88, 0x3a, 0xee, 0xce, 0x39, 0xfb, 0x96, 0x1e, 0x0a, 0x6b, 0x24, 0x08, 0x8d, 0xd3, 0xc4,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma3ChatTemplate {
    Official,
}

fn classify_chat_template(template: Option<&str>) -> Option<Gemma3ChatTemplate> {
    let template = template?;
    if template.len() != GEMMA3_CHAT_TEMPLATE_LEN {
        return None;
    }
    let digest: [u8; 32] = Sha256::digest(template.as_bytes()).into();
    (digest == GEMMA3_CHAT_TEMPLATE_SHA256).then_some(Gemma3ChatTemplate::Official)
}

/// GGUF-backed Gemma 3 family adapter.
///
/// Gemma 3 geometry and its per-layer execution capabilities are lowered to a
/// format-neutral [`DecoderSpec`] by the GGUF loader. The adapter owns only the
/// family boundary and the prompt convention selected by tokenizer metadata.
#[derive(Debug, Clone)]
pub struct Gemma3Adapter {
    spec: DecoderSpec,
    chat_template: Option<Gemma3ChatTemplate>,
}

impl Gemma3Adapter {
    /// Construct a Gemma 3 adapter from validated GGUF decoder metadata.
    ///
    /// An unknown or absent template deliberately leaves chat formatting
    /// disabled while keeping raw-prompt inference available.
    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "gemma3",
            "expected a gemma3 decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        Ok(Self {
            spec,
            chat_template: classify_chat_template(chat_template),
        })
    }
}

impl ModelAdapter for Gemma3Adapter {
    fn family(&self) -> &'static str {
        "gemma3"
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
            Some(Gemma3ChatTemplate::Official) => format_gemma3_chat_messages(messages),
            None => bail!(
                "unsupported or missing Gemma 3 chat template; pass --raw-prompt or use the pinned official Gemma 3 template"
            ),
        }
    }
}

fn format_gemma3_chat(system: Option<&str>, user: &str) -> String {
    let user = user.trim();
    let content = match system {
        Some(system) => format!("{system}\n\n{user}"),
        None => user.to_owned(),
    };
    format!("<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n")
}

fn format_gemma3_chat_messages(messages: &[ChatMessage]) -> Result<String> {
    validate_chat_messages(messages)?;
    let (system, turns) = match messages.first() {
        Some(ChatMessage {
            role: ChatRole::System,
            content,
            ..
        }) => (Some(content.as_str()), &messages[1..]),
        _ => (None, messages),
    };
    if let [message] = turns {
        return Ok(format_gemma3_chat(system, &message.content));
    }

    let mut prompt = String::from("<bos>");
    for (turn, message) in turns.iter().enumerate() {
        let role = match message.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "model",
            ChatRole::System => unreachable!("chat history validation rejected an inline system"),
            ChatRole::Tool => unreachable!("chat history validation rejected a tool response"),
        };
        prompt.push_str("<start_of_turn>");
        prompt.push_str(role);
        prompt.push('\n');
        if turn == 0
            && let Some(system) = system
        {
            prompt.push_str(system);
            prompt.push_str("\n\n");
        }
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
        PositionEncodingSpec, ProjectionBias, RopeLayout, RopeSpec,
    };

    const OFFICIAL_CHAT_TEMPLATE: &str = r#"{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '

' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '

' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '
' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>
' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model
'}}
{%- endif -%}
"#;

    fn tiny_spec() -> DecoderSpec {
        let rope = RopeSpec {
            theta: 10_000.0,
            layout: RopeLayout::HalfSplit,
            rotary_dim: 4,
            frequency_factors: None,
        };
        DecoderSpec {
            architecture: "gemma3".into(),
            runtime: RuntimeSpec {
                vocab_size: 32,
                max_context: 16,
                stop_token_ids: vec![1],
            },
            hidden_size: 8,
            intermediate_size: 16,
            num_layers: 1,
            num_query_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            norm_epsilon: 1e-6,
            layer_specs: vec![DecoderLayerSpec {
                mixer: crate::architecture::SequenceMixerSpec::Attention(AttentionSpec {
                    kind: AttentionKind::SlidingCausal { window: 8 },
                    position_encoding: PositionEncodingSpec::Rope(rope),
                    qk_norm: true,
                    output_gate: false,
                    scale: 0.5,
                    logit_softcap: None,
                }),
                activation: Activation::GeluTanh,
                norm_layout: NormLayout::Sandwich,
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

    /// The pinned template source still classifies to its variant.
    ///
    /// The Gemma 3 template is pinned by digest, so the fixture holds the
    /// exact bytes whose length and SHA-256 the classifier checks.
    #[test]
    fn the_pinned_template_source_still_classifies_to_its_variant() {
        let text = crate::models::template_parity::pinned_text("gemma3-official.jinja");
        assert_eq!(text, OFFICIAL_CHAT_TEMPLATE);
        assert_eq!(text.len(), GEMMA3_CHAT_TEMPLATE_LEN);
        assert_eq!(
            <[u8; 32]>::from(Sha256::digest(text.as_bytes())),
            GEMMA3_CHAT_TEMPLATE_SHA256,
        );
        assert_eq!(
            classify_chat_template(Some(&text)),
            Some(Gemma3ChatTemplate::Official),
        );
    }

    #[test]
    fn accepts_only_the_pinned_official_template_identity() -> Result<()> {
        assert_eq!(OFFICIAL_CHAT_TEMPLATE.len(), GEMMA3_CHAT_TEMPLATE_LEN);
        assert_eq!(
            <[u8; 32]>::from(Sha256::digest(OFFICIAL_CHAT_TEMPLATE.as_bytes())),
            GEMMA3_CHAT_TEMPLATE_SHA256
        );

        let adapter = Gemma3Adapter::from_decoder_spec(tiny_spec(), Some(OFFICIAL_CHAT_TEMPLATE))?;
        assert_eq!(adapter.family(), "gemma3");
        assert_eq!(
            adapter.format_chat(None, "  Hello  ")?,
            "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        );
        Ok(())
    }

    #[test]
    fn preserves_system_text_and_trims_the_first_user_turn() -> Result<()> {
        let adapter = Gemma3Adapter::from_decoder_spec(tiny_spec(), Some(OFFICIAL_CHAT_TEMPLATE))?;
        assert_eq!(
            adapter.format_chat(Some("  Be concise.\n"), "  Explain Rust.  ")?,
            "<bos><start_of_turn>user\n  Be concise.\n\n\nExplain Rust.<end_of_turn>\n<start_of_turn>model\n"
        );
        Ok(())
    }

    #[test]
    fn renders_exact_multi_turn_gemma_history() -> Result<()> {
        let adapter = Gemma3Adapter::from_decoder_spec(tiny_spec(), Some(OFFICIAL_CHAT_TEMPLATE))?;
        assert_eq!(
            adapter.format_chat_messages(&[
                ChatMessage::system("  Be concise.\n"),
                ChatMessage::user("  First question.  "),
                ChatMessage::assistant("  First answer.  "),
                ChatMessage::user("  Follow-up.  "),
            ])?,
            "<bos><start_of_turn>user\n  Be concise.\n\n\nFirst question.<end_of_turn>\n<start_of_turn>model\nFirst answer.<end_of_turn>\n<start_of_turn>user\nFollow-up.<end_of_turn>\n<start_of_turn>model\n"
        );
        Ok(())
    }

    #[test]
    fn unknown_template_requires_raw_prompt_and_wrong_family_is_rejected() -> Result<()> {
        for template in [None, Some("{{ unknown template }}")] {
            let adapter = Gemma3Adapter::from_decoder_spec(tiny_spec(), template)?;
            assert!(adapter.format_chat(None, "Hello").is_err());
        }
        let same_length_but_changed = OFFICIAL_CHAT_TEMPLATE.replacen("bos_token", "eos_token", 1);
        let adapter =
            Gemma3Adapter::from_decoder_spec(tiny_spec(), Some(same_length_but_changed.as_str()))?;
        assert!(adapter.format_chat(None, "Hello").is_err());

        let mut wrong_family = tiny_spec();
        wrong_family.architecture = "llama".into();
        assert!(Gemma3Adapter::from_decoder_spec(wrong_family, None).is_err());
        Ok(())
    }
}
