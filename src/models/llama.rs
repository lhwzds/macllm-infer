use std::{fs, path::Path};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{
    architecture::{
        DecoderLayerSpec, DecoderSpec, ProjectionBias, RopeLayout, RopeSpec, RuntimeSpec,
        validate_rope_table_shape,
    },
    chat::{ChatMessage, ChatRole, validate_chat_messages},
    decoder::DecoderModel,
    model_weights::WeightSource,
    models::{
        ModelAdapter, format_chatml_messages, load_stop_token_ids, resolve_default_rope_theta,
    },
};

const SMOLLM2_DEFAULT_SYSTEM: &str =
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
const SMOLLM2_CHAT_TEMPLATE: &str = concat!(
    "{% for message in messages %}",
    "{% if loop.first and messages[0]['role'] != 'system' %}",
    "{{ '<|im_start|>system\n",
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}",
    "{% endif %}",
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}",
    "{% endfor %}",
    "{% if add_generation_prompt %}",
    "{{ '<|im_start|>assistant\n' }}",
    "{% endif %}",
);
const LLAMA3_2_FALLBACK_DATE: &str = "26 Jul 2024";
const LLAMA3_2_CHAT_TEMPLATE_LEN: usize = 3_827;
const LLAMA3_2_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x58, 0x16, 0xfc, 0xe1, 0x04, 0x44, 0xe0, 0x3c, 0x2e, 0x9e, 0xe1, 0xef, 0x8a, 0x4a, 0x1e, 0xa6,
    0x1a, 0xe7, 0xe6, 0x9e, 0x43, 0x86, 0x13, 0xf3, 0xb1, 0x7b, 0x69, 0xd0, 0x42, 0x62, 0x23, 0xa4,
];

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum LlamaEosTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl LlamaEosTokenIds {
    fn to_vec(&self) -> Vec<u32> {
        match self {
            Self::One(token) => vec![*token],
            Self::Many(tokens) => tokens.clone(),
        }
    }
}

/// Hugging Face configuration fields that determine standard Llama decoder
/// execution. Unrelated training metadata is intentionally ignored by Serde.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    /// Transformers 5 stores even default `RoPE` settings in this object.
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub rope_interleaved: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    eos_token_id: Option<LlamaEosTokenIds>,
}

impl LlamaConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        serde_json::from_slice(
            &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("invalid Llama config at {}", path.display()))
    }

    pub fn to_spec(&self) -> Result<DecoderSpec> {
        self.to_spec_with_stop_tokens(
            self.eos_token_id
                .as_ref()
                .map_or_else(Vec::new, LlamaEosTokenIds::to_vec),
        )
    }

    fn to_spec_with_stop_tokens(&self, stop_token_ids: Vec<u32>) -> Result<DecoderSpec> {
        ensure!(self.model_type == "llama", "expected model_type=llama");
        ensure!(self.hidden_act == "silu", "only Llama SiLU is supported");
        ensure!(
            !self.rope_interleaved,
            "interleaved Llama RoPE is not supported"
        );
        ensure!(self.hidden_size > 0, "hidden_size must be positive");
        ensure!(
            self.num_attention_heads > 0,
            "num_attention_heads must be positive"
        );
        let head_dim = if let Some(head_dim) = self.head_dim {
            head_dim
        } else {
            ensure!(
                self.hidden_size.is_multiple_of(self.num_attention_heads),
                "hidden_size must be divisible by attention heads when head_dim is absent"
            );
            self.hidden_size / self.num_attention_heads
        };
        // Llama 3 materializes one factor per rotary frequency below. Apply the
        // same bound as the eager RoPE table before allocating that vector.
        validate_rope_table_shape(self.max_position_embeddings, head_dim)?;
        let num_kv_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        let projection_bias = ProjectionBias {
            query: self.attention_bias,
            key: self.attention_bias,
            value: self.attention_bias,
            output: self.attention_bias,
        };
        let rope = resolve_llama_rope(
            self.rope_theta,
            self.rope_scaling.as_ref(),
            self.rope_parameters.as_ref(),
            head_dim,
            self.max_position_embeddings,
        )?;
        let spec = DecoderSpec {
            architecture: "llama".into(),
            runtime: RuntimeSpec {
                vocab_size: self.vocab_size,
                max_context: self.max_position_embeddings,
                stop_token_ids,
            },
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_layers: self.num_hidden_layers,
            num_query_heads: self.num_attention_heads,
            num_kv_heads,
            head_dim,
            norm_epsilon: self.rms_norm_eps,
            layer_specs: vec![DecoderLayerSpec::standard(rope, head_dim); self.num_hidden_layers],
            embedding_scale: 1.0,
            final_logit_softcap: None,
            final_logit_scale: 1.0,
            projection_bias,
            mlp_bias: self.mlp_bias,
            tied_embeddings: self.tie_word_embeddings,
        };
        spec.validate()?;
        Ok(spec)
    }
}

fn resolve_llama_rope(
    legacy_theta: Option<f64>,
    legacy_scaling: Option<&serde_json::Value>,
    parameters: Option<&serde_json::Value>,
    head_dim: usize,
    max_context: usize,
) -> Result<RopeSpec> {
    ensure!(
        legacy_scaling.is_none() || parameters.is_none(),
        "Llama config cannot define both rope_scaling and rope_parameters"
    );
    let Some(raw) = legacy_scaling.or(parameters) else {
        return Ok(RopeSpec {
            theta: legacy_theta.unwrap_or(10_000.0),
            layout: RopeLayout::HalfSplit,
            rotary_dim: head_dim,
            frequency_factors: None,
        });
    };
    let object = raw
        .as_object()
        .context("Llama RoPE parameters must be an object")?;
    let rope_type = rope_parameter_type(object)?;
    if rope_type == "default" {
        let theta = if parameters.is_some() {
            resolve_default_rope_theta("Llama", legacy_theta, Some(raw))?
        } else {
            for key in object.keys() {
                ensure!(
                    matches!(key.as_str(), "rope_type" | "type"),
                    "Llama default rope_scaling parameter `{key}` is not supported"
                );
            }
            legacy_theta.unwrap_or(10_000.0)
        };
        return Ok(RopeSpec {
            theta,
            layout: RopeLayout::HalfSplit,
            rotary_dim: head_dim,
            frequency_factors: None,
        });
    }
    ensure!(
        rope_type == "llama3",
        "Llama rope_type `{rope_type}` is not supported yet"
    );
    resolve_llama3_rope(object, legacy_theta, head_dim, max_context)
}

fn resolve_llama3_rope(
    object: &serde_json::Map<String, serde_json::Value>,
    legacy_theta: Option<f64>,
    head_dim: usize,
    max_context: usize,
) -> Result<RopeSpec> {
    for key in object.keys() {
        ensure!(
            matches!(
                key.as_str(),
                "rope_type"
                    | "type"
                    | "rope_theta"
                    | "partial_rotary_factor"
                    | "factor"
                    | "low_freq_factor"
                    | "high_freq_factor"
                    | "original_max_position_embeddings"
            ),
            "Llama 3 RoPE parameter `{key}` is not supported"
        );
    }
    if let Some(partial) = object.get("partial_rotary_factor") {
        let partial = partial
            .as_f64()
            .context("Llama 3 partial_rotary_factor must be numeric")?;
        ensure!(
            (partial - 1.0).abs() < f64::EPSILON,
            "partial Llama 3 RoPE is not supported"
        );
    }

    let nested_theta = optional_json_f64(object, "rope_theta")?;
    if let (Some(legacy), Some(nested)) = (legacy_theta, nested_theta) {
        ensure!(
            (legacy - nested).abs() < f64::EPSILON,
            "Llama top-level rope_theta disagrees with Llama 3 rope_theta"
        );
    }
    let theta = nested_theta
        .or(legacy_theta)
        .context("Llama 3 rope_theta is missing")?;
    let factor = required_json_f64(object, "factor")?;
    let low_freq_factor = required_json_f64(object, "low_freq_factor")?;
    let high_freq_factor = required_json_f64(object, "high_freq_factor")?;
    let original_max_position_embeddings = object
        .get("original_max_position_embeddings")
        .context("Llama 3 original_max_position_embeddings is missing")?
        .as_u64()
        .context("Llama 3 original_max_position_embeddings must be an unsigned integer")?;
    let original_max_position_embeddings = usize::try_from(original_max_position_embeddings)
        .context("Llama 3 original context length does not fit usize")?;
    ensure!(
        factor.is_finite() && factor >= 1.0,
        "Llama 3 factor must be finite and at least one"
    );
    ensure!(
        low_freq_factor.is_finite() && low_freq_factor > 0.0,
        "Llama 3 low_freq_factor must be finite and positive"
    );
    ensure!(
        high_freq_factor.is_finite() && high_freq_factor > low_freq_factor,
        "Llama 3 high_freq_factor must be finite and greater than low_freq_factor"
    );
    ensure!(
        original_max_position_embeddings > 0 && original_max_position_embeddings < max_context,
        "Llama 3 original context length must be positive and smaller than max_position_embeddings"
    );

    Ok(RopeSpec {
        theta,
        layout: RopeLayout::HalfSplit,
        rotary_dim: head_dim,
        frequency_factors: Some(llama3_frequency_factors(
            theta,
            head_dim,
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        )),
    })
}

fn rope_parameter_type(object: &serde_json::Map<String, serde_json::Value>) -> Result<&str> {
    let current = object
        .get("rope_type")
        .map(|value| value.as_str().context("Llama rope_type must be a string"))
        .transpose()?;
    let legacy = object
        .get("type")
        .map(|value| value.as_str().context("Llama RoPE type must be a string"))
        .transpose()?;
    if let (Some(current), Some(legacy)) = (current, legacy) {
        ensure!(
            current == legacy,
            "Llama rope_type disagrees with legacy type"
        );
    }
    Ok(current.or(legacy).unwrap_or("default"))
}

fn optional_json_f64(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<f64>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_f64()
                .with_context(|| format!("Llama 3 {key} must be numeric"))
        })
        .transpose()
}

fn required_json_f64(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<f64> {
    optional_json_f64(object, key)?.with_context(|| format!("Llama 3 {key} is missing"))
}

fn llama3_frequency_factors(
    theta: f64,
    head_dim: usize,
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    original_max_position_embeddings: usize,
) -> Vec<f32> {
    let theta = theta as f32;
    let factor = factor as f32;
    let low_freq_factor = low_freq_factor as f32;
    let high_freq_factor = high_freq_factor as f32;
    let original_context = original_max_position_embeddings as f32;
    (0..head_dim)
        .step_by(2)
        .map(|index| {
            let inverse_frequency = 1.0 / theta.powf(index as f32 / head_dim as f32);
            let wavelength = 2.0 * std::f32::consts::PI / inverse_frequency;
            let low_freq_wavelength = original_context / low_freq_factor;
            let high_freq_wavelength = original_context / high_freq_factor;
            let effective_ratio = if wavelength < high_freq_wavelength {
                1.0
            } else if wavelength > low_freq_wavelength {
                1.0 / factor
            } else {
                let smooth = (original_context / wavelength - low_freq_factor)
                    / (high_freq_factor - low_freq_factor);
                (1.0 - smooth) / factor + smooth
            };
            1.0 / effective_ratio
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlamaChatTemplate {
    SmolLm2ChatMl,
    Llama3_2Instruct,
}

#[derive(Debug, Deserialize)]
struct TokenizerMetadata {
    #[serde(default)]
    chat_template: Option<serde_json::Value>,
}

fn load_chat_template(config_path: &Path) -> Result<Option<LlamaChatTemplate>> {
    let tokenizer_config = config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("tokenizer_config.json");
    if !tokenizer_config.is_file() {
        return Ok(None);
    }
    let metadata: TokenizerMetadata = serde_json::from_slice(
        &fs::read(&tokenizer_config)
            .with_context(|| format!("failed to read {}", tokenizer_config.display()))?,
    )
    .with_context(|| format!("invalid tokenizer config at {}", tokenizer_config.display()))?;
    let template = metadata.chat_template.as_ref().and_then(|template| {
        template.as_str().or_else(|| {
            template
                .as_object()
                .and_then(|templates| templates.get("default"))
                .and_then(serde_json::Value::as_str)
        })
    });
    Ok(classify_chat_template(template))
}

fn classify_chat_template(template: Option<&str>) -> Option<LlamaChatTemplate> {
    let template = template?;
    if template == SMOLLM2_CHAT_TEMPLATE {
        return Some(LlamaChatTemplate::SmolLm2ChatMl);
    }
    if template.len() == LLAMA3_2_CHAT_TEMPLATE_LEN {
        let digest: [u8; 32] = Sha256::digest(template.as_bytes()).into();
        if digest == LLAMA3_2_CHAT_TEMPLATE_SHA256 {
            return Some(LlamaChatTemplate::Llama3_2Instruct);
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct LlamaAdapter {
    spec: DecoderSpec,
    chat_template: Option<LlamaChatTemplate>,
}

impl LlamaAdapter {
    pub fn from_config(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config = LlamaConfig::from_path(path)?;
        let fallback = config
            .eos_token_id
            .as_ref()
            .map_or_else(Vec::new, LlamaEosTokenIds::to_vec);
        let stop_token_ids = load_stop_token_ids(path, fallback)?;
        Ok(Self {
            spec: config.to_spec_with_stop_tokens(stop_token_ids)?,
            chat_template: load_chat_template(path)?,
        })
    }

    /// Construct the adapter from format-neutral decoder metadata. GGUF keeps
    /// model geometry and the chat template in its own metadata table rather
    /// than in Hugging Face JSON sidecars.
    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "llama",
            "expected a llama decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        let chat_template = classify_chat_template(chat_template);
        Ok(Self {
            spec,
            chat_template,
        })
    }
}

impl ModelAdapter for LlamaAdapter {
    fn family(&self) -> &'static str {
        "llama"
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
            Some(LlamaChatTemplate::SmolLm2ChatMl) => {
                format_chatml_messages(SMOLLM2_DEFAULT_SYSTEM, messages)
            }
            Some(LlamaChatTemplate::Llama3_2Instruct) => format_llama3_2_chat_messages(messages),
            None => bail!(
                "unsupported or missing Llama chat template; pass --raw-prompt or use a supported tokenizer_config.json"
            ),
        }
    }
}

fn format_llama3_2_chat(system: Option<&str>, user: &str) -> String {
    // The pinned Jinja template uses this deterministic fallback when no
    // strftime helper or explicit date_string is supplied by the host.
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
Cutting Knowledge Date: December 2023\nToday Date: {LLAMA3_2_FALLBACK_DATE}\n\n{}\
<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}\
<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        system.map(str::trim).unwrap_or_default(),
        user.trim()
    )
}

fn format_llama3_2_chat_messages(messages: &[ChatMessage]) -> Result<String> {
    validate_chat_messages(messages)?;
    let (system, turns) = match messages.first() {
        Some(ChatMessage {
            role: ChatRole::System,
            content,
        }) => (Some(content.as_str()), &messages[1..]),
        _ => (None, messages),
    };
    if let [message] = turns {
        return Ok(format_llama3_2_chat(system, &message.content));
    }

    let mut prompt = format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
Cutting Knowledge Date: December 2023\nToday Date: {LLAMA3_2_FALLBACK_DATE}\n\n{}\
<|eot_id|>",
        system.map(str::trim).unwrap_or_default()
    );
    for message in turns {
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(message.role.as_str());
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(message.content.trim());
        prompt.push_str("<|eot_id|>");
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMOL_CONFIG: &str = r#"{
      "model_type":"llama","vocab_size":49152,"hidden_size":576,
      "intermediate_size":1536,"num_hidden_layers":30,
      "num_attention_heads":9,"num_key_value_heads":3,
      "max_position_embeddings":8192,"tie_word_embeddings":true,
      "rope_theta":100000.0,"rms_norm_eps":0.00001,"hidden_act":"silu",
      "rope_scaling":null,"rope_interleaved":false,
      "attention_bias":false,"mlp_bias":false,"eos_token_id":2
    }"#;

    #[test]
    fn maps_official_smollm2_geometry() -> Result<()> {
        let config: LlamaConfig = serde_json::from_str(SMOL_CONFIG)?;
        let spec = config.to_spec()?;
        assert_eq!(spec.runtime.vocab_size, 49_152);
        assert_eq!(spec.runtime.stop_token_ids, vec![2]);
        assert_eq!(spec.head_dim, 64);
        assert_eq!(spec.kv_groups(), 3);
        assert_eq!(spec.kv_bytes_per_token(2)?, 23_040);
        assert!(!spec.projection_bias.query);
        assert!(!spec.mlp_bias);
        assert!(spec.tied_embeddings);
        Ok(())
    }

    #[test]
    fn accepts_format_neutral_gguf_geometry_and_known_template() -> Result<()> {
        let config: LlamaConfig = serde_json::from_str(SMOL_CONFIG)?;
        let adapter =
            LlamaAdapter::from_decoder_spec(config.to_spec()?, Some(SMOLLM2_CHAT_TEMPLATE))?;
        assert_eq!(adapter.family(), "llama");
        assert_eq!(
            adapter.format_chat(None, "Hello")?,
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        Ok(())
    }

    #[test]
    fn formats_exact_multi_turn_smollm2_chatml() -> Result<()> {
        let config: LlamaConfig = serde_json::from_str(SMOL_CONFIG)?;
        let adapter =
            LlamaAdapter::from_decoder_spec(config.to_spec()?, Some(SMOLLM2_CHAT_TEMPLATE))?;
        assert_eq!(
            adapter.format_chat_messages(&[
                ChatMessage::user(" First "),
                ChatMessage::assistant(" Answer "),
                ChatMessage::user(" Follow-up "),
            ])?,
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n First <|im_end|>\n<|im_start|>assistant\n Answer <|im_end|>\n<|im_start|>user\n Follow-up <|im_end|>\n<|im_start|>assistant\n"
        );
        Ok(())
    }

    #[test]
    fn applies_hugging_face_defaults_and_explicit_attention_width() -> Result<()> {
        let config: LlamaConfig = serde_json::from_str(
            r#"{
              "model_type":"llama","vocab_size":32,"hidden_size":8,
              "intermediate_size":16,"num_hidden_layers":1,
              "num_attention_heads":2,"head_dim":6,
              "max_position_embeddings":16,"rms_norm_eps":0.00001,
              "hidden_act":"silu","attention_bias":true,"mlp_bias":true
            }"#,
        )?;
        let spec = config.to_spec()?;
        assert_eq!(spec.num_kv_heads, 2);
        assert_eq!(spec.head_dim, 6);
        assert!(
            (spec.layer_specs[0]
                .attention()
                .context("Llama layer is not attention")?
                .rope()
                .context("Llama layer does not use RoPE")?
                .theta
                - 10_000.0)
                .abs()
                < f64::EPSILON
        );
        assert_eq!(spec.num_query_heads * spec.head_dim, 12);
        assert_ne!(spec.num_query_heads * spec.head_dim, spec.hidden_size);
        assert!(spec.projection_bias.output);
        assert!(spec.mlp_bias);
        assert!(!spec.tied_embeddings);
        Ok(())
    }

    #[test]
    fn lowers_llama3_rope_scaling_to_frequency_factors() -> Result<()> {
        let mut config: LlamaConfig = serde_json::from_str(SMOL_CONFIG)?;
        config.hidden_size = 128;
        config.num_attention_heads = 2;
        config.num_key_value_heads = Some(1);
        config.max_position_embeddings = 131_072;
        config.rope_theta = Some(500_000.0);
        config.rope_scaling = Some(serde_json::json!({
            "rope_type":"llama3",
            "factor":32.0,
            "low_freq_factor":1.0,
            "high_freq_factor":4.0,
            "original_max_position_embeddings":8192
        }));
        let legacy = config.to_spec()?;
        let factors = legacy.layer_specs[0]
            .attention()
            .context("Llama layer is not attention")?
            .rope()
            .context("Llama layer does not use RoPE")?
            .frequency_factors
            .as_deref()
            .context("Llama 3 factors are absent")?;
        assert_eq!(factors.len(), 32);
        assert!(
            factors[..15]
                .iter()
                .all(|factor| (*factor - 1.0).abs() < f32::EPSILON)
        );
        for (actual, expected) in factors[15..18]
            .iter()
            .zip([1.651_329_f32, 3.292_263, 9.666_731])
        {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert!(
            factors[18..]
                .iter()
                .all(|factor| (*factor - 32.0).abs() < f32::EPSILON)
        );

        config.rope_scaling = None;
        config.rope_parameters = Some(serde_json::json!({
            "rope_type":"llama3",
            "rope_theta":500_000.0,
            "factor":32.0,
            "low_freq_factor":1.0,
            "high_freq_factor":4.0,
            "original_max_position_embeddings":8192
        }));
        assert_eq!(
            config.to_spec()?.layer_specs[0]
                .attention()
                .context("Llama layer is not attention")?
                .rope()
                .context("Llama layer does not use RoPE")?,
            legacy.layer_specs[0]
                .attention()
                .context("Llama layer is not attention")?
                .rope()
                .context("Llama layer does not use RoPE")?
        );
        Ok(())
    }

    #[test]
    fn rejects_unsupported_or_malformed_rope_semantics() -> Result<()> {
        let mut config: LlamaConfig = serde_json::from_str(SMOL_CONFIG)?;
        config.rope_scaling = Some(serde_json::json!({"rope_type":"yarn","factor":8.0}));
        assert!(config.to_spec().is_err());
        config.rope_scaling = Some(serde_json::json!({
            "rope_type":"llama3","factor":8.0,"low_freq_factor":1.0,
            "high_freq_factor":4.0,"original_max_position_embeddings":0
        }));
        assert!(config.to_spec().is_err());
        config.rope_theta = None;
        config.rope_scaling = Some(serde_json::json!({
            "rope_type":"llama3","factor":8.0,"low_freq_factor":1.0,
            "high_freq_factor":4.0,"original_max_position_embeddings":8192
        }));
        assert!(config.to_spec().is_err());
        config.rope_theta = Some(100_000.0);
        assert!(config.to_spec().is_err());
        config.rope_scaling = None;
        config.rope_parameters = Some(serde_json::json!({
            "rope_type":"llama3","rope_theta":100_000.0,"factor":8.0,
            "low_freq_factor":1.0,"high_freq_factor":4.0,
            "original_max_position_embeddings":4096,"partial_rotary_factor":0.5
        }));
        assert!(config.to_spec().is_err());
        config.rope_parameters = None;
        config.rope_interleaved = true;
        assert!(config.to_spec().is_err());
        Ok(())
    }

    #[test]
    fn rejects_oversized_llama3_rope_before_factor_allocation() -> Result<()> {
        let mut config: LlamaConfig = serde_json::from_str(SMOL_CONFIG)?;
        config.max_position_embeddings = 1;
        config.head_dim = Some(usize::MAX - 1);
        config.rope_theta = Some(500_000.0);
        config.rope_scaling = Some(serde_json::json!({
            "rope_type":"llama3",
            "factor":32.0,
            "low_freq_factor":1.0,
            "high_freq_factor":4.0,
            "original_max_position_embeddings":1
        }));
        let error = config.to_spec().unwrap_err().to_string();
        assert!(error.contains("RoPE table"));
        Ok(())
    }

    #[test]
    fn reads_transformers_five_default_rope_parameters() -> Result<()> {
        let mut config: LlamaConfig = serde_json::from_str(SMOL_CONFIG)?;
        config.rope_theta = None;
        config.rope_parameters = Some(serde_json::json!({
            "rope_type":"default","rope_theta":12_345.0
        }));
        let spec = config.to_spec()?;
        assert!(
            (spec.layer_specs[0]
                .attention()
                .context("Llama layer is not attention")?
                .rope()
                .context("Llama layer does not use RoPE")?
                .theta
                - 12_345.0)
                .abs()
                < f64::EPSILON
        );
        Ok(())
    }

    #[test]
    fn reads_stop_override_and_only_formats_a_recognized_template() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let config_path = temp.path().join("config.json");
        fs::write(&config_path, SMOL_CONFIG)?;
        fs::write(
            temp.path().join("generation_config.json"),
            r#"{"eos_token_id":[3,2,3]}"#,
        )?;
        fs::write(
            temp.path().join("tokenizer_config.json"),
            serde_json::to_vec(&serde_json::json!({
                "chat_template": SMOLLM2_CHAT_TEMPLATE
            }))?,
        )?;
        let adapter = LlamaAdapter::from_config(config_path)?;
        assert_eq!(adapter.runtime_spec().stop_token_ids, vec![3, 2]);
        assert_eq!(
            adapter.format_chat(None, "Hello")?,
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(
            adapter
                .format_chat(Some("Be terse."), "Hello")?
                .contains("Be terse.")
        );
        Ok(())
    }

    #[test]
    fn formats_only_the_exact_llama3_2_no_tools_template() -> Result<()> {
        let config: LlamaConfig = serde_json::from_str(SMOL_CONFIG)?;
        let adapter = LlamaAdapter {
            spec: config.to_spec()?,
            chat_template: Some(LlamaChatTemplate::Llama3_2Instruct),
        };
        assert_eq!(
            adapter.format_chat(None, "  Hello\n")?,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n\
<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello\
<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert_eq!(
            adapter.format_chat(Some("  Be terse.\n"), "  Hello\n")?,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nBe terse.\
<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello\
<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert_eq!(
            adapter.format_chat_messages(&[
                ChatMessage::system("  Be terse.\n"),
                ChatMessage::user("  First question.  "),
                ChatMessage::assistant("  First answer.  "),
                ChatMessage::user("  Follow-up.  "),
            ])?,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nBe terse.\
<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nFirst question.\
<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nFirst answer.\
<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nFollow-up.\
<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );

        let unknown_template = "x".repeat(LLAMA3_2_CHAT_TEMPLATE_LEN);
        let raw_only = LlamaAdapter::from_decoder_spec(config.to_spec()?, Some(&unknown_template))?;
        assert!(raw_only.format_chat(None, "Hello").is_err());
        Ok(())
    }

    #[test]
    fn unknown_llama_chat_template_requires_raw_prompt() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let config_path = temp.path().join("config.json");
        fs::write(&config_path, SMOL_CONFIG)?;
        fs::write(
            temp.path().join("tokenizer_config.json"),
            r#"{"chat_template":{"default":"<different-template>","tool_use":"<tool-template>"}}"#,
        )?;
        let adapter = LlamaAdapter::from_config(config_path)?;
        assert!(adapter.format_chat(None, "Hello").is_err());
        Ok(())
    }
}
