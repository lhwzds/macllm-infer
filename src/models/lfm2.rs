use std::{fs, path::Path};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

use crate::{
    architecture::{
        DecoderLayerSpec, DecoderSpec, MAX_DECODER_LAYERS, ProjectionBias, RopeLayout, RopeSpec,
        RuntimeSpec, SequenceMixerSpec, ShortConvolutionSpec,
    },
    chat::{ChatMessage, validate_chat_messages},
    decoder::{
        AttentionWeightNames, DecoderLayerWeightNames, DecoderModel, DecoderWeightNames,
        DenseFeedForwardWeightNames, STANDARD_DECODER_WEIGHT_NAMES,
    },
    model_weights::WeightSource,
    models::{ModelAdapter, load_stop_token_ids},
};

pub(crate) const LFM2_WEIGHT_NAMES: DecoderWeightNames = DecoderWeightNames {
    final_norm: "embedding_norm",
    layer: DecoderLayerWeightNames {
        mixer_norm: "operator_norm",
        feed_forward_norm: "ffn_norm",
        attention: "self_attn",
        feed_forward: "feed_forward",
        short_convolution: "conv",
        ..STANDARD_DECODER_WEIGHT_NAMES.layer
    },
    attention: AttentionWeightNames {
        output: "out_proj",
        query_norm: "q_layernorm",
        key_norm: "k_layernorm",
        ..STANDARD_DECODER_WEIGHT_NAMES.attention
    },
    dense_feed_forward: DenseFeedForwardWeightNames {
        gate: "w1",
        up: "w3",
        down: "w2",
    },
    ..STANDARD_DECODER_WEIGHT_NAMES
};

// Exact template embedded in LiquidAI/LFM2-350M-GGUF revision
// 8fdc9d526b7ed346b19257551b05816c7912ecc2. The richer current HF template
// includes tool branches that the text-only ChatMessage contract cannot yet
// represent, so it remains available through the raw-prompt path.
const OFFICIAL_GGUF_CHAT_TEMPLATE: &str = r"{{bos_token}}{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}";

#[derive(Debug, Clone, Deserialize)]
pub struct Lfm2Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub block_dim: usize,
    pub block_ff_dim: usize,
    pub block_multiple_of: usize,
    #[serde(default)]
    pub block_auto_adjust_ff_dim: bool,
    #[serde(default = "one")]
    pub block_ffn_dim_multiplier: f64,
    pub block_norm_eps: f64,
    pub norm_eps: f64,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub conv_dim: usize,
    pub conv_dim_out: usize,
    #[serde(default)]
    pub conv_bias: bool,
    #[serde(default)]
    pub block_use_swiglu: bool,
    pub full_attn_idxs: Vec<usize>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

const fn one() -> f64 {
    1.0
}

impl Lfm2Config {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        serde_json::from_slice(
            &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("invalid LFM2 config at {}", path.display()))
    }

    pub fn to_spec(&self) -> Result<DecoderSpec> {
        self.to_spec_with_stop_tokens(self.eos_token_id.into_iter().collect())
    }

    #[allow(clippy::too_many_lines)] // Validation and exact lowering stay together as the adapter contract.
    fn to_spec_with_stop_tokens(&self, stop_token_ids: Vec<u32>) -> Result<DecoderSpec> {
        ensure!(self.model_type == "lfm2", "expected model_type=lfm2");
        ensure!(
            self.num_hidden_layers > 0 && self.num_hidden_layers <= MAX_DECODER_LAYERS,
            "LFM2 layer count {} must be in 1..={MAX_DECODER_LAYERS}",
            self.num_hidden_layers
        );
        ensure!(
            self.hidden_size == self.block_dim
                && self.conv_dim == self.hidden_size
                && self.conv_dim_out == self.hidden_size,
            "LFM2 hidden, block and convolution dimensions must match"
        );
        ensure!(
            self.num_attention_heads == self.num_heads,
            "LFM2 num_attention_heads and num_heads disagree"
        );
        ensure!(
            self.num_attention_heads > 0,
            "LFM2 attention-head count must be positive"
        );
        ensure!(
            self.hidden_size.is_multiple_of(self.num_attention_heads),
            "LFM2 hidden size must be divisible by attention heads"
        );
        ensure!(
            self.block_auto_adjust_ff_dim,
            "LFM2 checkpoints without automatic FFN adjustment are not supported"
        );
        ensure!(
            (self.block_ffn_dim_multiplier - 1.0).abs() < f64::EPSILON,
            "LFM2 FFN multipliers other than 1.0 are not supported"
        );
        ensure!(
            self.block_multiple_of > 0,
            "LFM2 block_multiple_of must be positive"
        );
        ensure!(
            self.block_use_swiglu,
            "only LFM2 SwiGLU feed-forward blocks are supported"
        );
        ensure!(!self.conv_bias, "LFM2 convolution bias is not supported");
        ensure!(
            self.block_norm_eps.to_bits() == self.norm_eps.to_bits(),
            "LFM2 block and model RMSNorm epsilon disagree"
        );

        let adjusted = self
            .block_ff_dim
            .checked_mul(2)
            .context("LFM2 FFN adjustment overflows usize")?
            / 3;
        let intermediate_size = adjusted
            .div_ceil(self.block_multiple_of)
            .checked_mul(self.block_multiple_of)
            .context("LFM2 FFN rounding overflows usize")?;
        let head_dim = self.hidden_size / self.num_attention_heads;
        let rope = RopeSpec {
            theta: self.rope_theta,
            layout: RopeLayout::HalfSplit,
            rotary_dim: head_dim,
            frequency_factors: None,
        };
        let mut attention_layer = DecoderLayerSpec::standard(rope, head_dim);
        attention_layer
            .attention_mut()
            .expect("standard decoder layer uses attention")
            .qk_norm = true;

        let mut attention_schedule = vec![false; self.num_hidden_layers];
        for &index in &self.full_attn_idxs {
            ensure!(
                index < self.num_hidden_layers,
                "LFM2 attention layer index {index} is out of range"
            );
            ensure!(
                !attention_schedule[index],
                "LFM2 attention layer index {index} is duplicated"
            );
            attention_schedule[index] = true;
        }
        ensure!(
            attention_schedule.iter().any(|&attention| attention),
            "LFM2 must declare at least one attention layer"
        );
        let layer_specs = attention_schedule
            .into_iter()
            .map(|attention| {
                if attention {
                    attention_layer.clone()
                } else {
                    let mut layer = attention_layer.clone();
                    layer.mixer = SequenceMixerSpec::ShortConvolution(ShortConvolutionSpec {
                        kernel_size: self.conv_l_cache,
                    });
                    layer
                }
            })
            .collect();
        let spec = DecoderSpec {
            architecture: "lfm2".into(),
            runtime: RuntimeSpec {
                vocab_size: self.vocab_size,
                max_context: self.max_position_embeddings,
                stop_token_ids,
            },
            hidden_size: self.hidden_size,
            intermediate_size,
            num_layers: self.num_hidden_layers,
            num_query_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim,
            norm_epsilon: self.norm_eps,
            layer_specs,
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
        };
        spec.validate()?;
        Ok(spec)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lfm2ChatTemplate {
    OfficialGguf,
}

fn classify_chat_template(template: Option<&str>) -> Option<Lfm2ChatTemplate> {
    (template? == OFFICIAL_GGUF_CHAT_TEMPLATE).then_some(Lfm2ChatTemplate::OfficialGguf)
}

#[derive(Debug, Deserialize)]
struct TokenizerMetadata {
    #[serde(default)]
    chat_template: Option<String>,
}

fn load_chat_template(config_path: &Path) -> Result<Option<Lfm2ChatTemplate>> {
    let path = config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("tokenizer_config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let metadata: TokenizerMetadata = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("invalid tokenizer config at {}", path.display()))?;
    Ok(classify_chat_template(metadata.chat_template.as_deref()))
}

#[derive(Debug)]
pub struct Lfm2Adapter {
    spec: DecoderSpec,
    chat_template: Option<Lfm2ChatTemplate>,
}

impl Lfm2Adapter {
    pub fn from_config(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config = Lfm2Config::from_path(path)?;
        let stop_token_ids = load_stop_token_ids(path, config.eos_token_id.into_iter().collect())?;
        let spec = config.to_spec_with_stop_tokens(stop_token_ids)?;
        let adapter = Self {
            spec,
            chat_template: load_chat_template(path)?,
        };
        adapter.validate()?;
        Ok(adapter)
    }

    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(spec.architecture == "lfm2", "expected LFM2 decoder spec");
        let adapter = Self {
            spec,
            chat_template: classify_chat_template(chat_template),
        };
        adapter.validate()?;
        Ok(adapter)
    }
}

impl ModelAdapter for Lfm2Adapter {
    fn family(&self) -> &'static str {
        "lfm2"
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
        DecoderModel::load_with_weight_names(self.spec.clone(), weights, &LFM2_WEIGHT_NAMES)
            .map(|model| Box::new(model) as Box<dyn crate::architecture::CausalLm>)
    }

    fn format_chat_messages(&self, messages: &[ChatMessage]) -> Result<String> {
        let Some(Lfm2ChatTemplate::OfficialGguf) = self.chat_template else {
            bail!("LFM2 tokenizer chat template is absent or unsupported; use a raw prompt")
        };
        validate_chat_messages(messages)?;
        let mut prompt = String::from("<|startoftext|>");
        for message in messages {
            prompt.push_str("<|im_start|>");
            prompt.push_str(message.role.as_str());
            prompt.push('\n');
            prompt.push_str(&message.content);
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>assistant\n");
        Ok(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFFICIAL_CONFIG: &str = r#"{
      "model_type":"lfm2","vocab_size":65536,"hidden_size":1024,
      "block_dim":1024,"block_ff_dim":6656,"block_multiple_of":256,
      "block_auto_adjust_ff_dim":true,"block_ffn_dim_multiplier":1.0,
      "block_norm_eps":0.00001,"norm_eps":0.00001,
      "num_hidden_layers":16,"num_attention_heads":16,"num_heads":16,
      "num_key_value_heads":8,"max_position_embeddings":128000,
      "rope_theta":1000000.0,"conv_L_cache":3,"conv_dim":1024,
      "conv_dim_out":1024,"conv_bias":false,"block_use_swiglu":true,
      "full_attn_idxs":[2,5,8,10,12,14],"eos_token_id":7
    }"#;

    #[test]
    fn maps_official_hybrid_geometry() -> Result<()> {
        let config: Lfm2Config = serde_json::from_str(OFFICIAL_CONFIG)?;
        let spec = config.to_spec()?;
        assert_eq!(spec.intermediate_size, 4_608);
        assert_eq!(spec.head_dim, 64);
        assert_eq!(spec.runtime.stop_token_ids, vec![7]);
        let attention = spec
            .layer_specs
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| layer.attention().map(|_| index))
            .collect::<Vec<_>>();
        assert_eq!(attention, [2, 5, 8, 10, 12, 14]);
        assert_eq!(
            spec.kv_bytes_per_token(4)?,
            2 * 6 * 8 * 64 * std::mem::size_of::<f32>()
        );
        Ok(())
    }

    #[test]
    fn renders_the_exact_embedded_gguf_text_chat_grammar() -> Result<()> {
        let config: Lfm2Config = serde_json::from_str(OFFICIAL_CONFIG)?;
        let adapter =
            Lfm2Adapter::from_decoder_spec(config.to_spec()?, Some(OFFICIAL_GGUF_CHAT_TEMPLATE))?;
        assert_eq!(
            adapter.format_chat(None, "Hello")?,
            "<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        Ok(())
    }

    #[test]
    fn rejects_oversized_layer_count_before_allocating_the_schedule() -> Result<()> {
        let mut config: Lfm2Config = serde_json::from_str(OFFICIAL_CONFIG)?;
        config.num_hidden_layers = usize::MAX;
        let error = config.to_spec().unwrap_err().to_string();
        assert!(
            error.contains("LFM2 layer count"),
            "unexpected error: {error}"
        );
        Ok(())
    }
}
