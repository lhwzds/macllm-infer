use anyhow::{Result, bail, ensure};
use candle_core::DType;
use sha2::{Digest, Sha256};

use crate::{
    architecture::{DecoderSpec, RuntimeSpec},
    chat::{ChatMessage, ChatRole, validate_chat_messages},
    decoder::DecoderModel,
    model_weights::WeightSource,
    models::ModelAdapter,
};

// Identities of the two confirmed `tokenizer.chat_template` variants embedded
// in Granite 3.1 1B-A400M Instruct GGUFs. Checking both byte length and digest
// keeps raw-prompt inference available for other Granite checkpoints without
// silently applying a merely similar prompt grammar.
const LEGACY_GRANITE_3_1_CHAT_TEMPLATE_LEN: usize = 3_224;
const LEGACY_GRANITE_3_1_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x22, 0xda, 0x30, 0x19, 0x45, 0xac, 0x6b, 0x61, 0x7a, 0x88, 0x35, 0xd4, 0x35, 0xcb, 0xbd, 0xc7,
    0x80, 0xc0, 0xac, 0x08, 0xd6, 0x9c, 0x07, 0xe1, 0xe1, 0x7e, 0x6c, 0x67, 0x99, 0xee, 0xd8, 0x00,
];
const CURRENT_GRANITE_3_1_CHAT_TEMPLATE_LEN: usize = 3_543;
const CURRENT_GRANITE_3_1_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x08, 0x96, 0x2c, 0x2f, 0x15, 0xd5, 0x67, 0x67, 0x85, 0x4b, 0x46, 0xdf, 0xc4, 0x07, 0x0b, 0x37,
    0xf4, 0xc4, 0x43, 0x55, 0x18, 0x33, 0xbb, 0xa6, 0x5b, 0x41, 0x71, 0x91, 0x73, 0x5f, 0x31, 0x87,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraniteMoeChatTemplate {
    /// Original compact GGUF template: fixed default system and spaces between
    /// role records.
    LegacyGranite3_1Instruct,
    /// Current official template: dated default system and newlines between
    /// role records.
    CurrentGranite3_1Instruct,
}

fn classify_chat_template(template: Option<&str>) -> Option<GraniteMoeChatTemplate> {
    let template = template?;
    if !matches!(
        template.len(),
        LEGACY_GRANITE_3_1_CHAT_TEMPLATE_LEN | CURRENT_GRANITE_3_1_CHAT_TEMPLATE_LEN
    ) {
        return None;
    }
    let digest: [u8; 32] = Sha256::digest(template.as_bytes()).into();
    match (template.len(), digest) {
        (LEGACY_GRANITE_3_1_CHAT_TEMPLATE_LEN, LEGACY_GRANITE_3_1_CHAT_TEMPLATE_SHA256) => {
            Some(GraniteMoeChatTemplate::LegacyGranite3_1Instruct)
        }
        (CURRENT_GRANITE_3_1_CHAT_TEMPLATE_LEN, CURRENT_GRANITE_3_1_CHAT_TEMPLATE_SHA256) => {
            Some(GraniteMoeChatTemplate::CurrentGranite3_1Instruct)
        }
        _ => None,
    }
}

/// GGUF-backed IBM Granite sparse-mixture-of-experts adapter.
///
/// GGUF metadata is lowered to a format-neutral [`DecoderSpec`] by the loader.
/// This adapter owns the family boundary, shared decoder construction and the
/// exact text-only subset of the pinned Granite 3.1 Instruct chat template.
#[derive(Debug, Clone)]
pub struct GraniteMoeAdapter {
    spec: DecoderSpec,
    chat_template: Option<GraniteMoeChatTemplate>,
}

impl GraniteMoeAdapter {
    /// Construct an adapter from validated Granite `MoE` GGUF metadata.
    ///
    /// An unknown or absent template deliberately disables chat formatting
    /// while leaving raw-prompt inference available.
    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "granitemoe",
            "expected a granitemoe decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        Ok(Self {
            spec,
            chat_template: classify_chat_template(chat_template),
        })
    }
}

impl ModelAdapter for GraniteMoeAdapter {
    fn family(&self) -> &'static str {
        "granitemoe"
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
        if weights.dtype() == DType::F16 {
            candle_core::bail!(
                "Granite sparse-MoE inference does not support F16 execution because router logits can overflow; use F32 or BF16"
            );
        }
        Ok(Box::new(DecoderModel::load(self.spec.clone(), weights)?))
    }

    fn format_chat_messages(&self, messages: &[ChatMessage]) -> Result<String> {
        match self.chat_template {
            Some(template) => format_granite_3_1_chat_messages(template, messages),
            None => bail!(
                "unsupported or missing Granite MoE chat template; pass --raw-prompt or use the pinned Granite 3.1 Instruct GGUF template"
            ),
        }
    }
}

fn format_granite_3_1_chat_messages(
    template: GraniteMoeChatTemplate,
    messages: &[ChatMessage],
) -> Result<String> {
    validate_chat_messages(messages)?;
    let (system, turns) = match messages.first() {
        Some(ChatMessage {
            role: ChatRole::System,
            content,
            ..
        }) => (content.as_str(), &messages[1..]),
        _ => match template {
            GraniteMoeChatTemplate::LegacyGranite3_1Instruct => (
                "Knowledge Cutoff Date: April 2024. You are Granite, developed by IBM. You are a helpful AI assistant.",
                messages,
            ),
            GraniteMoeChatTemplate::CurrentGranite3_1Instruct => {
                // The current Jinja template calls `strftime_now` when it
                // synthesizes a default system prompt. Requiring an explicit
                // system message keeps this renderer byte-exact without hiding
                // a process clock/time-zone dependency in the adapter.
                bail!(
                    "the current Granite 3.1 chat template requires the current date when no system message is provided; supply an explicit system message or pass --raw-prompt"
                );
            }
        },
    };
    let separator = match template {
        GraniteMoeChatTemplate::LegacyGranite3_1Instruct => " ",
        GraniteMoeChatTemplate::CurrentGranite3_1Instruct => "\n",
    };

    let mut prompt = String::from("<|start_of_role|>system<|end_of_role|>");
    prompt.push_str(system);
    prompt.push_str("<|end_of_text|>");
    prompt.push_str(separator);
    for message in turns {
        prompt.push_str("<|start_of_role|>");
        prompt.push_str(message.role.as_str());
        prompt.push_str("<|end_of_role|>");
        prompt.push_str(&message.content);
        prompt.push_str("<|end_of_text|>");
        prompt.push_str(separator);
    }
    prompt.push_str("<|start_of_role|>assistant<|end_of_role|>");
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::{
        DecoderLayerSpec, FeedForwardKind, ProjectionBias, RopeLayout, RopeSpec, RouterWeighting,
        SparseMoeSpec,
    };
    use crate::model_weights::DenseWeightSource;
    use candle_core::Device;
    use candle_nn::VarBuilder;

    const LEGACY_OFFICIAL_CHAT_TEMPLATE: &str = r#"{%- if messages[0]['role'] == 'system' %}{%- set system_message = messages[0]['content'] %}{%- set loop_messages = messages[1:] %}{%- else %}{%- set system_message = "Knowledge Cutoff Date: April 2024. You are Granite, developed by IBM." %}{%- if tools and documents %}{%- set system_message = system_message + " You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request. Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data." %}{%- elif tools %}{%- set system_message = system_message + " You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request." %}{%- elif documents %}{%- set system_message = system_message + " Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data." %}{%- else %}{%- set system_message = system_message + " You are a helpful AI assistant." %}{%- endif %}{%- if controls and 'citations' in controls and documents %}{%- set system_message = system_message + ' In your response, use the symbols <co> and </co> to indicate when a fact comes from a document in the search result, e.g <co>0</co> for a fact from document 0. Afterwards, list all the citations with their corresponding documents in an ordered list.' %}{%- endif %}{%- if controls and 'hallucinations' in controls and documents %}{%- set system_message = system_message + ' Finally, after the response is written, include a numbered list of sentences from the response that are potentially hallucinated and not based in the documents.' %}{%- endif %}{%- set loop_messages = messages %}{%- endif %}{{- '<|start_of_role|>system<|end_of_role|>' + system_message + '<|end_of_text|> ' }}{%- if tools %}{{- '<|start_of_role|>tools<|end_of_role|>' }}{{- tools | tojson(indent=4) }}{{- '<|end_of_text|> ' }}{%- endif %}{%- if documents %}{{- '<|start_of_role|>documents<|end_of_role|>' }}{%- for document in documents %}{{- 'Document ' + loop.index0 | string + ' ' }}{{- document['text'] }}{%- if not loop.last %}{{- ' '}}{%- endif%}{%- endfor %}{{- '<|end_of_text|> ' }}{%- endif %}{%- for message in loop_messages %}{{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|> ' }}{%- if loop.last and add_generation_prompt %}{{- '<|start_of_role|>assistant' }}{%- if controls %}{{- ' ' + controls | tojson()}}{%- endif %}{{- '<|end_of_role|>' }}{%- endif %}{%- endfor %}"#;

    const CURRENT_OFFICIAL_CHAT_TEMPLATE: &str = concat!(
        r#"{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = "Knowledge Cutoff Date: April 2024.
Today's Date: " + strftime_now('%B %d, %Y') + ".
You are Granite, developed by IBM." %}
    {%- if tools and documents %}
        {%- set system_message = system_message + " You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.

Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data." %}
    {%- elif tools %}
        {%- set system_message = system_message + " You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request." %}
    {%- elif documents %}
        {%- set system_message = system_message + " Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data." %}
    {%- else %}
        {%- set system_message = system_message + " You are a helpful AI assistant." %}"#,
        "    \n",
        r#"    {%- endif %}
    {%- if 'citations' in controls and documents %}
        {%- set system_message = system_message + '

In your response, use the symbols <co> and </co> to indicate when a fact comes from a document in the search result, e.g <co>0</co> for a fact from document 0. Afterwards, list all the citations with their corresponding documents in an ordered list.' %}
    {%- endif %}
    {%- if 'hallucinations' in controls and documents %}
        {%- set system_message = system_message + '

Finally, after the response is written, include a numbered list of sentences from the response that are potentially hallucinated and not based in the documents.' %}
    {%- endif %}
    {%- set loop_messages = messages %}
{%- endif %}
{{- '<|start_of_role|>system<|end_of_role|>' + system_message + '<|end_of_text|>
' }}
{%- if tools %}
    {{- '<|start_of_role|>tools<|end_of_role|>' }}
    {{- tools | tojson(indent=4) }}
    {{- '<|end_of_text|>
' }}
{%- endif %}
{%- if documents %}
    {{- '<|start_of_role|>documents<|end_of_role|>' }}
    {%- for document in documents %}
        {{- 'Document ' + loop.index0 | string + '
' }}
        {{- document['text'] }}
        {%- if not loop.last %}
            {{- '

'}}
        {%- endif%}
    {%- endfor %}
    {{- '<|end_of_text|>
' }}
{%- endif %}
{%- for message in loop_messages %}
    {{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|>
' }}
    {%- if loop.last and add_generation_prompt %}
        {{- '<|start_of_role|>assistant' }}
            {%- if controls %}
                {{- ' ' + controls | tojson()}}
            {%- endif %}
        {{- '<|end_of_role|>' }}
    {%- endif %}
{%- endfor %}"#,
    );

    fn tiny_spec() -> DecoderSpec {
        let head_dim = 4;
        let mut layer = DecoderLayerSpec::standard(
            RopeSpec {
                theta: 1_500_000.0,
                layout: RopeLayout::HalfSplit,
                rotary_dim: head_dim,
                frequency_factors: None,
            },
            head_dim,
        );
        layer
            .attention_mut()
            .expect("standard decoder layer uses attention")
            .scale = 1.0 / 4.0;
        layer.feed_forward = FeedForwardKind::SparseMoe(SparseMoeSpec {
            expert_count: 4,
            experts_per_token: 2,
            expert_intermediate_size: 8,
            router_weighting: RouterWeighting::SelectedSoftmax,
        });
        layer.mixer_residual_scale = 0.22;
        layer.feed_forward_residual_scale = 0.22;
        DecoderSpec {
            architecture: "granitemoe".into(),
            runtime: RuntimeSpec {
                vocab_size: 32,
                max_context: 16,
                stop_token_ids: vec![0],
            },
            hidden_size: 8,
            intermediate_size: 8,
            num_layers: 1,
            num_query_heads: 2,
            num_kv_heads: 1,
            head_dim,
            norm_epsilon: 1e-6,
            layer_specs: vec![layer],
            embedding_scale: 12.0,
            final_logit_softcap: None,
            final_logit_scale: 1.0 / 6.0,
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
    fn accepts_both_confirmed_gguf_template_identities() -> Result<()> {
        assert_eq!(
            LEGACY_OFFICIAL_CHAT_TEMPLATE.len(),
            LEGACY_GRANITE_3_1_CHAT_TEMPLATE_LEN
        );
        assert_eq!(
            <[u8; 32]>::from(Sha256::digest(LEGACY_OFFICIAL_CHAT_TEMPLATE.as_bytes())),
            LEGACY_GRANITE_3_1_CHAT_TEMPLATE_SHA256
        );
        assert_eq!(
            CURRENT_OFFICIAL_CHAT_TEMPLATE.len(),
            CURRENT_GRANITE_3_1_CHAT_TEMPLATE_LEN
        );
        assert_eq!(
            <[u8; 32]>::from(Sha256::digest(CURRENT_OFFICIAL_CHAT_TEMPLATE.as_bytes())),
            CURRENT_GRANITE_3_1_CHAT_TEMPLATE_SHA256
        );

        let current = GraniteMoeAdapter::from_decoder_spec(
            tiny_spec(),
            Some(CURRENT_OFFICIAL_CHAT_TEMPLATE),
        )?;
        assert_eq!(current.family(), "granitemoe");
        assert_eq!(current.runtime_spec().stop_token_ids, vec![0]);
        assert_eq!(
            current.format_chat(Some("Be precise."), "Hello")?,
            "<|start_of_role|>system<|end_of_role|>Be precise.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        );

        let legacy =
            GraniteMoeAdapter::from_decoder_spec(tiny_spec(), Some(LEGACY_OFFICIAL_CHAT_TEMPLATE))?;
        assert_eq!(
            legacy.format_chat(Some("Be precise."), "Hello")?,
            "<|start_of_role|>system<|end_of_role|>Be precise.<|end_of_text|> <|start_of_role|>user<|end_of_role|>Hello<|end_of_text|> <|start_of_role|>assistant<|end_of_role|>"
        );
        assert_eq!(
            legacy.format_chat(None, "Hello")?,
            "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024. You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|> <|start_of_role|>user<|end_of_role|>Hello<|end_of_text|> <|start_of_role|>assistant<|end_of_role|>"
        );
        Ok(())
    }

    #[test]
    fn renders_exact_multi_turn_history_without_trimming_content() -> Result<()> {
        let adapter = GraniteMoeAdapter::from_decoder_spec(
            tiny_spec(),
            Some(CURRENT_OFFICIAL_CHAT_TEMPLATE),
        )?;
        assert_eq!(
            adapter.format_chat_messages(&[
                ChatMessage::system("  Be concise.\n"),
                ChatMessage::user(" First "),
                ChatMessage::assistant(" Answer "),
                ChatMessage::user(" Follow-up "),
            ])?,
            "<|start_of_role|>system<|end_of_role|>  Be concise.\n<|end_of_text|>\n<|start_of_role|>user<|end_of_role|> First <|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|> Answer <|end_of_text|>\n<|start_of_role|>user<|end_of_role|> Follow-up <|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        );
        Ok(())
    }

    #[test]
    fn rejects_dynamic_default_system_unknown_templates_and_wrong_family() -> Result<()> {
        let adapter = GraniteMoeAdapter::from_decoder_spec(
            tiny_spec(),
            Some(CURRENT_OFFICIAL_CHAT_TEMPLATE),
        )?;
        let missing_system = adapter.format_chat(None, "Hello").unwrap_err().to_string();
        assert!(missing_system.contains("current date"));

        for template in [None, Some("{{ unknown template }}")] {
            let adapter = GraniteMoeAdapter::from_decoder_spec(tiny_spec(), template)?;
            assert!(adapter.format_chat(Some("system"), "Hello").is_err());
        }
        let same_length_but_changed =
            CURRENT_OFFICIAL_CHAT_TEMPLATE.replacen("Granite", "granite", 1);
        let adapter = GraniteMoeAdapter::from_decoder_spec(
            tiny_spec(),
            Some(same_length_but_changed.as_str()),
        )?;
        assert!(adapter.format_chat(Some("system"), "Hello").is_err());

        let mut wrong_family = tiny_spec();
        wrong_family.architecture = "llama".into();
        assert!(GraniteMoeAdapter::from_decoder_spec(wrong_family, None).is_err());
        Ok(())
    }

    #[test]
    fn rejects_f16_before_loading_granite_weights() -> Result<()> {
        let adapter = GraniteMoeAdapter::from_decoder_spec(tiny_spec(), None)?;
        let weights = DenseWeightSource::new(VarBuilder::zeros(DType::F16, &Device::Cpu));
        let error = adapter
            .build_model(&weights)
            .err()
            .expect("F16 Granite build should fail before weight loading")
            .to_string();
        assert!(error.contains("does not support F16"));
        Ok(())
    }
}
