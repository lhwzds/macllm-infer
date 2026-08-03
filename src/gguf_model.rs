use std::{path::Path, sync::Arc};

use anyhow::{Context, Result, bail, ensure};
use tokenizers::Tokenizer;

use crate::{
    architecture::{
        Activation, AttentionKind, AttentionSpec, DecoderLayerSpec, DecoderSpec, FeedForwardKind,
        GatedDeltaNetSpec, MAX_DECODER_LAYERS, Mamba2Spec, NormLayout, ParallelAttentionMamba2Spec,
        PositionEncodingSpec, ProjectionBias, RopeLayout, RopeSpec, RouterWeighting, RuntimeSpec,
        SequenceMixerSpec, ShortConvolutionSpec, SparseMoeSpec,
    },
    gguf::{GgufBackendConfig, GgufFile, GgufMetadata, GgufValue, GgufValueType, LlamaQkConfig},
    gguf_tokenizer::{EmbeddedBpeTokenizer, EmbeddedSentencePieceTokenizer},
    models::{
        ModelAdapter, falcon_h1::FalconH1Adapter, gemma::GemmaAdapter, gemma2::Gemma2Adapter,
        gemma3::Gemma3Adapter, lfm2::Lfm2Adapter, llama::LlamaAdapter, qwen2::Qwen2Adapter,
        qwen3::Qwen3Adapter, qwen3_moe::Qwen3MoeAdapter, qwen35::Qwen35Adapter,
        smollm3::SmolLm3Adapter,
    },
};

const MAX_GGUF_CONTEXT_LENGTH: usize = 1 << 20;
const MAX_TOKENIZER_BUILD_INPUT_BYTES: u64 = 64 << 20;
const SUPPORTED_LLAMA_METADATA_KEYS: &[&str] = &[
    "llama.attention.causal",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.attention.key_length",
    "llama.attention.layer_norm_rms_epsilon",
    "llama.attention.sliding_window",
    "llama.attention.value_length",
    "llama.block_count",
    "llama.context_length",
    "llama.embedding_length",
    "llama.expert_count",
    "llama.expert_used_count",
    "llama.expert_weights_scale",
    "llama.feed_forward_length",
    "llama.rope.dimension_count",
    "llama.rope.freq_base",
    "llama.vocab_size",
];
const SUPPORTED_QWEN2_METADATA_KEYS: &[&str] = &[
    "qwen2.attention.causal",
    "qwen2.attention.head_count",
    "qwen2.attention.head_count_kv",
    "qwen2.attention.key_length",
    "qwen2.attention.layer_norm_rms_epsilon",
    "qwen2.attention.sliding_window",
    "qwen2.attention.value_length",
    "qwen2.block_count",
    "qwen2.context_length",
    "qwen2.embedding_length",
    "qwen2.feed_forward_length",
    "qwen2.rope.dimension_count",
    "qwen2.rope.freq_base",
    "qwen2.vocab_size",
];
const SUPPORTED_QWEN3_METADATA_KEYS: &[&str] = &[
    "qwen3.attention.causal",
    "qwen3.attention.head_count",
    "qwen3.attention.head_count_kv",
    "qwen3.attention.key_length",
    "qwen3.attention.layer_norm_rms_epsilon",
    "qwen3.attention.sliding_window",
    "qwen3.attention.value_length",
    "qwen3.block_count",
    "qwen3.context_length",
    "qwen3.embedding_length",
    "qwen3.feed_forward_length",
    "qwen3.rope.dimension_count",
    "qwen3.rope.freq_base",
    "qwen3.vocab_size",
];
const SUPPORTED_SMOLLM3_METADATA_KEYS: &[&str] = &[
    "smollm3.attention.causal",
    "smollm3.attention.head_count",
    "smollm3.attention.head_count_kv",
    "smollm3.attention.key_length",
    "smollm3.attention.layer_norm_rms_epsilon",
    "smollm3.attention.sliding_window",
    "smollm3.attention.value_length",
    "smollm3.block_count",
    "smollm3.context_length",
    "smollm3.embedding_length",
    "smollm3.feed_forward_length",
    "smollm3.rope.dimension_count",
    "smollm3.rope.freq_base",
    "smollm3.vocab_size",
];
const SUPPORTED_QWEN3_MOE_METADATA_KEYS: &[&str] = &[
    "qwen3moe.attention.causal",
    "qwen3moe.attention.head_count",
    "qwen3moe.attention.head_count_kv",
    "qwen3moe.attention.key_length",
    "qwen3moe.attention.layer_norm_rms_epsilon",
    "qwen3moe.attention.sliding_window",
    "qwen3moe.attention.value_length",
    "qwen3moe.block_count",
    "qwen3moe.context_length",
    "qwen3moe.embedding_length",
    "qwen3moe.expert_count",
    "qwen3moe.expert_feed_forward_length",
    "qwen3moe.expert_shared_feed_forward_length",
    "qwen3moe.expert_used_count",
    "qwen3moe.feed_forward_length",
    "qwen3moe.rope.dimension_count",
    "qwen3moe.rope.freq_base",
    "qwen3moe.vocab_size",
];
const SUPPORTED_QWEN35_METADATA_KEYS: &[&str] = &[
    "qwen35.attention.causal",
    "qwen35.attention.head_count",
    "qwen35.attention.head_count_kv",
    "qwen35.attention.key_length",
    "qwen35.attention.layer_norm_rms_epsilon",
    "qwen35.attention.recurrent_layers",
    "qwen35.attention.value_length",
    "qwen35.block_count",
    "qwen35.context_length",
    "qwen35.embedding_length",
    "qwen35.feed_forward_length",
    "qwen35.full_attention_interval",
    "qwen35.nextn_predict_layers",
    "qwen35.rope.dimension_count",
    "qwen35.rope.dimension_sections",
    "qwen35.rope.freq_base",
    "qwen35.ssm.conv_kernel",
    "qwen35.ssm.group_count",
    "qwen35.ssm.inner_size",
    "qwen35.ssm.state_size",
    "qwen35.ssm.time_step_rank",
    "qwen35.vocab_size",
];
const SUPPORTED_GEMMA2_METADATA_KEYS: &[&str] = &[
    "gemma2.attention.causal",
    "gemma2.attention.head_count",
    "gemma2.attention.head_count_kv",
    "gemma2.attention.key_length",
    "gemma2.attention.layer_norm_rms_epsilon",
    "gemma2.attention.sliding_window",
    "gemma2.attention.sliding_window_pattern",
    "gemma2.attention.value_length",
    "gemma2.attn_logit_softcapping",
    "gemma2.block_count",
    "gemma2.context_length",
    "gemma2.embedding_length",
    "gemma2.feed_forward_length",
    "gemma2.final_logit_softcapping",
    "gemma2.rope.freq_base",
    "gemma2.rope.freq_base_swa",
    "gemma2.vocab_size",
];
const SUPPORTED_GEMMA_METADATA_KEYS: &[&str] = &[
    "gemma.attention.causal",
    "gemma.attention.head_count",
    "gemma.attention.head_count_kv",
    "gemma.attention.key_length",
    "gemma.attention.layer_norm_rms_epsilon",
    "gemma.attention.value_length",
    "gemma.block_count",
    "gemma.context_length",
    "gemma.embedding_length",
    "gemma.feed_forward_length",
    "gemma.rope.dimension_count",
    "gemma.rope.freq_base",
    "gemma.vocab_size",
];
const SUPPORTED_GEMMA3_METADATA_KEYS: &[&str] = &[
    "gemma3.attention.head_count",
    "gemma3.attention.head_count_kv",
    "gemma3.attention.key_length",
    "gemma3.attention.layer_norm_rms_epsilon",
    "gemma3.attention.sliding_window",
    "gemma3.attention.sliding_window_pattern",
    "gemma3.attention.value_length",
    "gemma3.block_count",
    "gemma3.context_length",
    "gemma3.embedding_length",
    "gemma3.feed_forward_length",
    "gemma3.final_logit_softcapping",
    "gemma3.rope.freq_base",
    "gemma3.rope.freq_base_swa",
    "gemma3.vocab_size",
];
const SUPPORTED_GRANITE_MOE_METADATA_KEYS: &[&str] = &[
    "granitemoe.attention.causal",
    "granitemoe.attention.head_count",
    "granitemoe.attention.head_count_kv",
    "granitemoe.attention.key_length",
    "granitemoe.attention.layer_norm_rms_epsilon",
    "granitemoe.attention.scale",
    "granitemoe.attention.value_length",
    "granitemoe.block_count",
    "granitemoe.context_length",
    "granitemoe.embedding_length",
    "granitemoe.embedding_scale",
    "granitemoe.expert_count",
    "granitemoe.expert_used_count",
    "granitemoe.feed_forward_length",
    "granitemoe.logit_scale",
    "granitemoe.residual_scale",
    "granitemoe.rope.dimension_count",
    "granitemoe.rope.freq_base",
    "granitemoe.vocab_size",
];
const SUPPORTED_LFM2_METADATA_KEYS: &[&str] = &[
    "lfm2.attention.causal",
    "lfm2.attention.head_count",
    "lfm2.attention.head_count_kv",
    "lfm2.attention.key_length",
    "lfm2.attention.layer_norm_rms_epsilon",
    "lfm2.attention.value_length",
    "lfm2.block_count",
    "lfm2.context_length",
    "lfm2.embedding_length",
    "lfm2.feed_forward_length",
    "lfm2.rope.dimension_count",
    "lfm2.rope.freq_base",
    "lfm2.shortconv.l_cache",
    "lfm2.vocab_size",
];
const SUPPORTED_FALCON_H1_METADATA_KEYS: &[&str] = &[
    "falcon-h1.attention.head_count",
    "falcon-h1.attention.head_count_kv",
    "falcon-h1.attention.key_length",
    "falcon-h1.attention.layer_norm_rms_epsilon",
    "falcon-h1.attention.value_length",
    "falcon-h1.block_count",
    "falcon-h1.context_length",
    "falcon-h1.embedding_length",
    "falcon-h1.feed_forward_length",
    "falcon-h1.rope.freq_base",
    "falcon-h1.ssm.conv_kernel",
    "falcon-h1.ssm.group_count",
    "falcon-h1.ssm.inner_size",
    "falcon-h1.ssm.state_size",
    "falcon-h1.ssm.time_step_rank",
    "falcon-h1.vocab_size",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufArchitecture {
    FalconH1,
    Gemma,
    Gemma2,
    Gemma3,
    GraniteMoe,
    Lfm2,
    Llama,
    Qwen2,
    Qwen3,
    Qwen3Moe,
    Qwen35,
    SmolLm3,
}

/// A logical single- or multi-file GGUF checkpoint translated into the format-neutral runtime
/// contracts used by the rest of `MacLLM Infer`.
#[derive(Debug, Clone)]
pub struct GgufModel {
    file: GgufFile,
    architecture: GgufArchitecture,
    spec: DecoderSpec,
    tokenizer: Arc<Tokenizer>,
    chat_template: Option<String>,
}

impl GgufModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        let architecture = require_string(file.metadata(), "general.architecture")?;
        let (architecture, spec) = match architecture {
            "falcon-h1" => (GgufArchitecture::FalconH1, falcon_h1_spec(&file)?),
            "gemma" => (GgufArchitecture::Gemma, gemma_spec(&file)?),
            "gemma2" => (GgufArchitecture::Gemma2, gemma2_spec(&file)?),
            "gemma3" => (GgufArchitecture::Gemma3, gemma3_spec(&file)?),
            "granitemoe" => (GgufArchitecture::GraniteMoe, granite_moe_spec(&file)?),
            "lfm2" => (GgufArchitecture::Lfm2, lfm2_spec(&file)?),
            "llama" => (GgufArchitecture::Llama, llama_spec(&file)?),
            "qwen2" => (GgufArchitecture::Qwen2, qwen2_spec(&file)?),
            "qwen3" => (GgufArchitecture::Qwen3, qwen3_spec(&file)?),
            "qwen3moe" => (GgufArchitecture::Qwen3Moe, qwen3_moe_spec(&file)?),
            "qwen35" => (GgufArchitecture::Qwen35, qwen35_spec(&file)?),
            "smollm3" => (GgufArchitecture::SmolLm3, smollm3_spec(&file)?),
            architecture => bail!(
                "unsupported GGUF architecture `{architecture}`; supported architectures are falcon-h1, gemma, gemma2, gemma3, granitemoe, lfm2, llama, qwen2, qwen3, qwen3moe, qwen35 and smollm3"
            ),
        };
        validate_decoder_tensors(&file, &spec)?;
        let (tokenizer, chat_template) = embedded_tokenizer(file.metadata(), &spec)?;
        Ok(Self {
            file,
            architecture,
            spec,
            tokenizer: Arc::new(tokenizer),
            chat_template,
        })
    }

    #[must_use]
    pub fn file(&self) -> &GgufFile {
        &self.file
    }

    #[must_use]
    pub fn spec(&self) -> &DecoderSpec {
        &self.spec
    }

    pub fn adapter(&self) -> Result<Box<dyn ModelAdapter>> {
        match self.architecture {
            GgufArchitecture::FalconH1 => Ok(Box::new(FalconH1Adapter::from_decoder_spec(
                self.spec.clone(),
                self.chat_template.as_deref(),
            )?)),
            GgufArchitecture::Gemma => Ok(Box::new(GemmaAdapter::from_decoder_spec(
                self.spec.clone(),
                self.chat_template.as_deref(),
            )?)),
            GgufArchitecture::Gemma2 => Ok(Box::new(Gemma2Adapter::from_decoder_spec(
                self.spec.clone(),
                self.chat_template.as_deref(),
            )?)),
            GgufArchitecture::Gemma3 => Ok(Box::new(Gemma3Adapter::from_decoder_spec(
                self.spec.clone(),
                self.chat_template.as_deref(),
            )?)),
            GgufArchitecture::GraniteMoe => Ok(Box::new(
                crate::models::granite_moe::GraniteMoeAdapter::from_decoder_spec(
                    self.spec.clone(),
                    self.chat_template.as_deref(),
                )?,
            )),
            GgufArchitecture::Lfm2 => Ok(Box::new(Lfm2Adapter::from_decoder_spec(
                self.spec.clone(),
                self.chat_template.as_deref(),
            )?)),
            GgufArchitecture::Llama => Ok(Box::new(LlamaAdapter::from_decoder_spec(
                self.spec.clone(),
                self.chat_template.as_deref(),
            )?)),
            GgufArchitecture::Qwen2 => Ok(Box::new(Qwen2Adapter::from_decoder_spec(
                self.spec.clone(),
                self.chat_template.as_deref(),
            )?)),
            GgufArchitecture::Qwen3 => Ok(Box::new(Qwen3Adapter::from_decoder_spec(
                self.spec.clone(),
                self.chat_template.as_deref(),
            )?)),
            GgufArchitecture::Qwen3Moe => Ok(Box::new(Qwen3MoeAdapter::from_decoder_spec(
                self.spec.clone(),
                self.chat_template.as_deref(),
            )?)),
            GgufArchitecture::Qwen35 => Ok(Box::new(Qwen35Adapter::from_decoder_spec(
                self.spec.clone(),
                self.chat_template.as_deref(),
            )?)),
            GgufArchitecture::SmolLm3 => Ok(Box::new(SmolLm3Adapter::from_decoder_spec(
                self.spec.clone(),
                self.chat_template.as_deref(),
            )?)),
        }
    }

    #[must_use]
    pub fn tokenizer(&self) -> Arc<Tokenizer> {
        Arc::clone(&self.tokenizer)
    }

    pub(crate) fn backend_config(&self) -> GgufBackendConfig {
        GgufBackendConfig {
            map_hf_names: true,
            llama_qk: matches!(
                self.architecture,
                GgufArchitecture::FalconH1
                    | GgufArchitecture::Llama
                    | GgufArchitecture::GraniteMoe
                    | GgufArchitecture::SmolLm3
            )
            .then_some(LlamaQkConfig {
                query_heads: self.spec.num_query_heads,
                kv_heads: self.spec.num_kv_heads,
            }),
            fallback_output_to_embedding: self.spec.tied_embeddings,
        }
    }
}

#[allow(clippy::too_many_lines)] // One strict profile owns the hybrid geometry and tensor-presence contract.
fn falcon_h1_spec(file: &GgufFile) -> Result<DecoderSpec> {
    let metadata = file.metadata();
    validate_supported_falcon_h1_metadata(metadata)?;
    ensure!(
        file.tensor("rope_freqs.weight").is_none(),
        "GGUF Falcon-H1 RoPE frequency factors are not supported yet"
    );

    let vocab_size = falcon_h1_vocab_size(file)?;
    let hidden_size = required_usize(metadata, "falcon-h1.embedding_length")?;
    let intermediate_size = required_usize(metadata, "falcon-h1.feed_forward_length")?;
    let num_layers = required_usize(metadata, "falcon-h1.block_count")?;
    ensure!(
        num_layers > 0 && num_layers <= MAX_DECODER_LAYERS,
        "GGUF Falcon-H1 layer count {num_layers} must be in 1..={MAX_DECODER_LAYERS}"
    );

    let num_query_heads = required_usize(metadata, "falcon-h1.attention.head_count")?;
    let num_kv_heads = required_usize(metadata, "falcon-h1.attention.head_count_kv")?;
    ensure!(
        num_query_heads > 0 && num_kv_heads > 0 && num_query_heads.is_multiple_of(num_kv_heads),
        "GGUF Falcon-H1 requires positive query/KV head counts with query heads divisible by KV heads"
    );
    let head_dim = required_usize(metadata, "falcon-h1.attention.key_length")?;
    ensure!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "GGUF Falcon-H1 attention head width must be positive and even"
    );
    ensure!(
        optional_usize(metadata, "falcon-h1.attention.value_length")?.unwrap_or(head_dim)
            == head_dim,
        "different GGUF Falcon-H1 key/value head widths are not supported"
    );
    ensure!(
        num_query_heads.checked_mul(head_dim) == Some(hidden_size),
        "GGUF Falcon-H1 currently requires query heads * head width to equal embedding length"
    );

    let max_context = required_usize(metadata, "falcon-h1.context_length")?;
    ensure!(
        max_context <= MAX_GGUF_CONTEXT_LENGTH,
        "GGUF Falcon-H1 context length {max_context} exceeds the supported limit {MAX_GGUF_CONTEXT_LENGTH}"
    );
    let norm_epsilon = require_f64(metadata, "falcon-h1.attention.layer_norm_rms_epsilon")?;
    ensure!(
        norm_epsilon.is_finite() && norm_epsilon > 0.0,
        "GGUF Falcon-H1 RMS norm epsilon must be finite and positive"
    );
    let rope_theta = require_f64(metadata, "falcon-h1.rope.freq_base")?;
    ensure!(
        rope_theta.is_finite() && rope_theta > 0.0,
        "GGUF Falcon-H1 RoPE frequency base must be finite and positive"
    );

    let convolution_kernel_size = required_usize(metadata, "falcon-h1.ssm.conv_kernel")?;
    let inner_size = required_usize(metadata, "falcon-h1.ssm.inner_size")?;
    let state_size = required_usize(metadata, "falcon-h1.ssm.state_size")?;
    let num_heads = required_usize(metadata, "falcon-h1.ssm.time_step_rank")?;
    let num_groups = required_usize(metadata, "falcon-h1.ssm.group_count")?;
    ensure!(
        convolution_kernel_size > 0
            && inner_size > 0
            && state_size > 0
            && num_heads > 0
            && num_groups > 0,
        "GGUF Falcon-H1 Mamba2 geometry must be positive"
    );
    ensure!(
        inner_size.is_multiple_of(num_heads),
        "GGUF Falcon-H1 Mamba2 inner size must be divisible by its head count"
    );
    ensure!(
        num_heads.is_multiple_of(num_groups),
        "GGUF Falcon-H1 Mamba2 heads must be divisible by its group count"
    );

    reject_falcon_h1_unmodeled_biases(file)?;
    let projection_bias = ProjectionBias {
        query: consistent_layer_presence(file, "attn_q.bias", num_layers)?,
        key: consistent_layer_presence(file, "attn_k.bias", num_layers)?,
        value: consistent_layer_presence(file, "attn_v.bias", num_layers)?,
        output: consistent_layer_presence(file, "attn_output.bias", num_layers)?,
    };
    let gate_bias = consistent_layer_presence(file, "ffn_gate.bias", num_layers)?;
    let up_bias = consistent_layer_presence(file, "ffn_up.bias", num_layers)?;
    let down_bias = consistent_layer_presence(file, "ffn_down.bias", num_layers)?;
    ensure!(
        gate_bias == up_bias && up_bias == down_bias,
        "the shared decoder requires all Falcon-H1 MLP projections to agree on bias presence"
    );
    let convolution_bias = consistent_layer_presence(file, "ssm_conv1d.bias", num_layers)?;
    let input_projection_bias = consistent_layer_presence(file, "ssm_in.bias", num_layers)?;
    let output_projection_bias = consistent_layer_presence(file, "ssm_out.bias", num_layers)?;
    ensure!(
        !consistent_layer_presence(file, "ssm_norm.weight", num_layers)?,
        "GGUF Falcon-H1 gated Mamba2 RMS normalization is not supported yet"
    );
    ensure!(
        file.tensor("output.bias").is_none(),
        "GGUF output bias is not supported by the shared decoder"
    );
    validate_falcon_h1_tensor_inventory(file, num_layers)?;

    let attention = AttentionSpec {
        kind: AttentionKind::FullCausal,
        position_encoding: PositionEncodingSpec::Rope(RopeSpec {
            theta: rope_theta,
            layout: RopeLayout::HalfSplit,
            rotary_dim: head_dim,
            frequency_factors: None,
        }),
        qk_norm: false,
        output_gate: false,
        scale: 1.0 / (head_dim as f64).sqrt(),
        logit_softcap: None,
    };
    let mamba2 = Mamba2Spec {
        convolution_kernel_size,
        inner_size,
        state_size,
        num_heads,
        num_groups,
        head_dim: inner_size / num_heads,
        convolution_bias,
        input_projection_bias,
        output_projection_bias,
        gated_rms_norm: false,
    };
    let layer = DecoderLayerSpec {
        mixer: SequenceMixerSpec::ParallelAttentionMamba2(ParallelAttentionMamba2Spec {
            attention,
            mamba2,
        }),
        activation: Activation::Silu,
        norm_layout: NormLayout::PreNorm,
        feed_forward: FeedForwardKind::Dense,
        mixer_residual_scale: 1.0,
        feed_forward_residual_scale: 1.0,
    };
    let spec = DecoderSpec {
        architecture: "falcon-h1".into(),
        runtime: RuntimeSpec {
            vocab_size,
            max_context,
            stop_token_ids: optional_u32(metadata, "tokenizer.ggml.eos_token_id")?
                .into_iter()
                .collect(),
        },
        hidden_size,
        intermediate_size,
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        norm_epsilon,
        layer_specs: vec![layer; num_layers],
        embedding_scale: 1.0,
        final_logit_softcap: None,
        final_logit_scale: 1.0,
        projection_bias,
        mlp_bias: gate_bias,
        tied_embeddings: file.tensor("output.weight").is_none(),
    };
    spec.validate()?;
    Ok(spec)
}

fn falcon_h1_vocab_size(file: &GgufFile) -> Result<usize> {
    let metadata = file.metadata();
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    ensure!(
        !tokens.is_empty(),
        "GGUF Falcon-H1 tokenizer vocabulary is empty"
    );
    let embedding = file
        .tensor("token_embd.weight")
        .context("required GGUF tensor `token_embd.weight` is missing")?;
    ensure!(
        embedding.shape.len() == 2 && embedding.shape[0] == tokens.len(),
        "GGUF Falcon-H1 tokenizer has {} tokens but embedding shape is {:?}",
        tokens.len(),
        embedding.shape
    );
    let declared = required_usize(metadata, "falcon-h1.vocab_size")?;
    ensure!(
        declared == tokens.len(),
        "GGUF Falcon-H1 vocabulary metadata declares {declared} tokens, found {}",
        tokens.len()
    );
    Ok(tokens.len())
}

fn reject_falcon_h1_unmodeled_biases(file: &GgufFile) -> Result<()> {
    const ALLOWED_SUFFIXES: &[&str] = &[
        ".attn_q.bias",
        ".attn_k.bias",
        ".attn_v.bias",
        ".attn_output.bias",
        ".ffn_gate.bias",
        ".ffn_up.bias",
        ".ffn_down.bias",
        ".ssm_in.bias",
        ".ssm_conv1d.bias",
        ".ssm_dt.bias",
        ".ssm_out.bias",
    ];
    if let Some(name) = file.tensors().keys().find(|name| {
        name.rsplit_once('.')
            .is_some_and(|(_, suffix)| suffix == "bias")
            && !ALLOWED_SUFFIXES.iter().any(|suffix| name.ends_with(suffix))
    }) {
        bail!("GGUF Falcon-H1 contains unmodeled bias tensor `{name}`");
    }
    Ok(())
}

fn validate_falcon_h1_tensor_inventory(file: &GgufFile, layers: usize) -> Result<()> {
    const GLOBAL_TENSORS: &[&str] = &["token_embd.weight", "output_norm.weight", "output.weight"];
    const LAYER_TENSOR_SUFFIXES: &[&str] = &[
        "attn_norm.weight",
        "ffn_norm",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q.bias",
        "attn_k.bias",
        "attn_v.bias",
        "attn_output.bias",
        "ssm_in.weight",
        "ssm_in.bias",
        "ssm_conv1d.weight",
        "ssm_conv1d.bias",
        "ssm_dt.bias",
        "ssm_a",
        "ssm_d",
        "ssm_out.weight",
        "ssm_out.bias",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "ffn_gate.bias",
        "ffn_up.bias",
        "ffn_down.bias",
    ];

    for name in file.tensors().keys() {
        if GLOBAL_TENSORS.contains(&name.as_str()) {
            continue;
        }
        let Some((layer_text, suffix)) = name
            .strip_prefix("blk.")
            .and_then(|remainder| remainder.split_once('.'))
        else {
            bail!("GGUF Falcon-H1 contains unsupported tensor `{name}`");
        };
        let Ok(layer) = layer_text.parse::<usize>() else {
            bail!("GGUF Falcon-H1 contains unsupported tensor `{name}`");
        };
        ensure!(
            layer_text == layer.to_string(),
            "GGUF Falcon-H1 contains unsupported tensor `{name}`"
        );
        ensure!(
            layer < layers,
            "GGUF Falcon-H1 tensor `{name}` uses layer {layer}, outside the declared {layers}-layer profile"
        );
        ensure!(
            LAYER_TENSOR_SUFFIXES.contains(&suffix),
            "GGUF Falcon-H1 contains unsupported tensor `{name}`"
        );
    }
    Ok(())
}

#[allow(clippy::too_many_lines)] // Metadata validation and hybrid lowering form one fail-closed contract.
fn lfm2_spec(file: &GgufFile) -> Result<DecoderSpec> {
    let metadata = file.metadata();
    validate_supported_lfm2_metadata(metadata)?;
    ensure!(
        file.tensor("rope_freqs.weight").is_none(),
        "GGUF LFM2 rope frequency factors are not supported yet"
    );
    let vocab_size = lfm2_vocab_size(file)?;
    let hidden_size = required_usize(metadata, "lfm2.embedding_length")?;
    let intermediate_size = required_usize(metadata, "lfm2.feed_forward_length")?;
    let num_layers = required_usize(metadata, "lfm2.block_count")?;
    ensure!(
        num_layers > 0 && num_layers <= MAX_DECODER_LAYERS,
        "GGUF LFM2 layer count {num_layers} must be in 1..={MAX_DECODER_LAYERS}"
    );
    reject_lfm2_biases(file)?;
    let num_query_heads = required_usize(metadata, "lfm2.attention.head_count")?;
    ensure!(
        num_query_heads > 0 && hidden_size.is_multiple_of(num_query_heads),
        "GGUF LFM2 embedding length must be divisible by a positive query-head count"
    );
    let kv_heads_by_layer =
        required_layer_usizes(metadata, "lfm2.attention.head_count_kv", num_layers)?;
    let num_kv_heads = kv_heads_by_layer
        .iter()
        .copied()
        .find(|&heads| heads != 0)
        .context("GGUF LFM2 declares no attention layers")?;
    ensure!(
        kv_heads_by_layer
            .iter()
            .all(|&heads| heads == 0 || heads == num_kv_heads),
        "GGUF LFM2 attention layers must use one consistent KV-head count"
    );
    ensure!(
        num_query_heads.is_multiple_of(num_kv_heads),
        "GGUF LFM2 query heads must be divisible by KV heads"
    );
    let inferred_head_dim = hidden_size / num_query_heads;
    let head_dim =
        optional_usize(metadata, "lfm2.attention.key_length")?.unwrap_or(inferred_head_dim);
    ensure!(
        head_dim == inferred_head_dim,
        "GGUF LFM2 key length {head_dim} must equal hidden_size / query_heads ({inferred_head_dim})"
    );
    ensure!(
        optional_usize(metadata, "lfm2.attention.value_length")?.unwrap_or(head_dim) == head_dim,
        "different GGUF LFM2 key/value head widths are not supported"
    );
    ensure!(
        optional_usize(metadata, "lfm2.rope.dimension_count")?.unwrap_or(head_dim) == head_dim,
        "partial GGUF LFM2 RoPE is not supported"
    );
    let max_context = required_usize(metadata, "lfm2.context_length")?;
    ensure!(
        max_context <= MAX_GGUF_CONTEXT_LENGTH,
        "GGUF LFM2 context length {max_context} exceeds the supported limit {MAX_GGUF_CONTEXT_LENGTH}"
    );
    let norm_epsilon = require_f64(metadata, "lfm2.attention.layer_norm_rms_epsilon")?;
    let kernel_size = required_usize(metadata, "lfm2.shortconv.l_cache")?;
    let rope = RopeSpec {
        theta: require_f64(metadata, "lfm2.rope.freq_base")?,
        layout: RopeLayout::HalfSplit,
        rotary_dim: head_dim,
        frequency_factors: None,
    };
    let mut attention_layer = DecoderLayerSpec::standard(rope, head_dim);
    attention_layer
        .attention_mut()
        .expect("standard decoder layer uses attention")
        .qk_norm = true;
    let layer_specs = kv_heads_by_layer
        .into_iter()
        .map(|kv_heads| {
            if kv_heads == 0 {
                let mut layer = attention_layer.clone();
                layer.mixer =
                    SequenceMixerSpec::ShortConvolution(ShortConvolutionSpec { kernel_size });
                layer
            } else {
                attention_layer.clone()
            }
        })
        .collect::<Vec<_>>();
    let spec = DecoderSpec {
        architecture: "lfm2".into(),
        runtime: RuntimeSpec {
            vocab_size,
            max_context,
            stop_token_ids: optional_u32(metadata, "tokenizer.ggml.eos_token_id")?
                .into_iter()
                .collect(),
        },
        hidden_size,
        intermediate_size,
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        norm_epsilon,
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
        tied_embeddings: file.tensor("output.weight").is_none(),
    };
    spec.validate()?;
    Ok(spec)
}

fn lfm2_vocab_size(file: &GgufFile) -> Result<usize> {
    let metadata = file.metadata();
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    ensure!(
        !tokens.is_empty(),
        "GGUF LFM2 tokenizer vocabulary is empty"
    );
    let embedding = file
        .tensor("token_embd.weight")
        .context("required GGUF tensor `token_embd.weight` is missing")?;
    ensure!(
        embedding.shape.len() == 2 && embedding.shape[0] == tokens.len(),
        "GGUF LFM2 tokenizer has {} tokens but embedding shape is {:?}",
        tokens.len(),
        embedding.shape
    );
    if let Some(declared) = optional_usize(metadata, "lfm2.vocab_size")? {
        ensure!(
            declared == tokens.len(),
            "GGUF LFM2 vocabulary metadata declares {declared} tokens, found {}",
            tokens.len()
        );
    }
    Ok(tokens.len())
}

fn reject_lfm2_biases(file: &GgufFile) -> Result<()> {
    if let Some(name) = file.tensors().keys().find(|name| {
        name.rsplit_once('.')
            .is_some_and(|(_, suffix)| suffix == "bias")
    }) {
        bail!("GGUF LFM2 does not support bias tensor `{name}`");
    }
    Ok(())
}

fn granite_moe_spec(file: &GgufFile) -> Result<DecoderSpec> {
    let metadata = file.metadata();
    validate_supported_granite_moe_metadata(metadata)?;
    ensure!(
        file.tensor("rope_freqs.weight").is_none(),
        "GGUF Granite MoE rope frequency factors are not supported yet"
    );
    let geometry = granite_moe_geometry(metadata)?;
    let scales = granite_moe_scales(metadata)?;

    let expert_count = required_usize(metadata, "granitemoe.expert_count")?;
    let experts_per_token = required_usize(metadata, "granitemoe.expert_used_count")?;
    let expert_intermediate_size = required_usize(metadata, "granitemoe.feed_forward_length")?;
    reject_granite_moe_biases_and_shared_experts(file, geometry.num_layers)?;

    let layer = DecoderLayerSpec {
        mixer: SequenceMixerSpec::Attention(AttentionSpec {
            kind: AttentionKind::FullCausal,
            position_encoding: PositionEncodingSpec::Rope(RopeSpec {
                theta: scales.rope_theta,
                layout: RopeLayout::HalfSplit,
                rotary_dim: geometry.head_dim,
                frequency_factors: None,
            }),
            qk_norm: false,
            output_gate: false,
            scale: scales.attention,
            logit_softcap: None,
        }),
        activation: Activation::Silu,
        norm_layout: NormLayout::PreNorm,
        feed_forward: FeedForwardKind::SparseMoe(SparseMoeSpec {
            expert_count,
            experts_per_token,
            expert_intermediate_size,
            router_weighting: RouterWeighting::SelectedSoftmax,
        }),
        mixer_residual_scale: scales.residual,
        feed_forward_residual_scale: scales.residual,
    };
    let spec = DecoderSpec {
        architecture: "granitemoe".into(),
        runtime: RuntimeSpec {
            vocab_size: geometry.vocab_size,
            max_context: geometry.max_context,
            stop_token_ids: granite_moe_stop_token_ids(metadata)?,
        },
        hidden_size: geometry.hidden_size,
        intermediate_size: expert_intermediate_size,
        num_layers: geometry.num_layers,
        num_query_heads: geometry.num_query_heads,
        num_kv_heads: geometry.num_kv_heads,
        head_dim: geometry.head_dim,
        norm_epsilon: geometry.norm_epsilon,
        layer_specs: vec![layer; geometry.num_layers],
        embedding_scale: scales.embedding,
        final_logit_softcap: None,
        final_logit_scale: 1.0 / scales.logit_divisor,
        projection_bias: ProjectionBias {
            query: false,
            key: false,
            value: false,
            output: false,
        },
        mlp_bias: false,
        tied_embeddings: file.tensor("output.weight").is_none(),
    };
    spec.validate()?;
    Ok(spec)
}

struct GraniteMoeGeometry {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_context: usize,
    norm_epsilon: f64,
}

fn granite_moe_geometry(metadata: &GgufMetadata) -> Result<GraniteMoeGeometry> {
    let vocab_size = required_usize(metadata, "granitemoe.vocab_size")?;
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    ensure!(
        tokens.len() == vocab_size,
        "GGUF Granite MoE vocabulary metadata declares {vocab_size} tokens but the tokenizer contains {}",
        tokens.len()
    );
    let hidden_size = required_usize(metadata, "granitemoe.embedding_length")?;
    let num_layers = required_usize(metadata, "granitemoe.block_count")?;
    ensure!(
        num_layers <= MAX_DECODER_LAYERS,
        "GGUF Granite MoE layer count {num_layers} exceeds the supported limit {MAX_DECODER_LAYERS}"
    );
    let num_query_heads = required_usize(metadata, "granitemoe.attention.head_count")?;
    ensure!(
        num_query_heads > 0,
        "Granite MoE attention head count must be positive"
    );
    let num_kv_heads =
        optional_usize(metadata, "granitemoe.attention.head_count_kv")?.unwrap_or(num_query_heads);
    ensure!(
        num_kv_heads > 0,
        "Granite MoE KV head count must be positive"
    );
    let head_dim = if let Some(explicit) =
        optional_usize(metadata, "granitemoe.attention.key_length")?
    {
        explicit
    } else {
        ensure!(
            hidden_size.is_multiple_of(num_query_heads),
            "Granite MoE embedding length must be divisible by query heads when key length is absent"
        );
        hidden_size / num_query_heads
    };
    ensure!(
        optional_usize(metadata, "granitemoe.attention.value_length")?.unwrap_or(head_dim)
            == head_dim,
        "different Granite MoE key/value head widths are not supported"
    );
    let rope_dimension =
        optional_usize(metadata, "granitemoe.rope.dimension_count")?.unwrap_or(head_dim);
    ensure!(
        rope_dimension == head_dim,
        "partial Granite MoE RoPE is not supported: dimension_count={rope_dimension}, head_dim={head_dim}"
    );
    let max_context = required_usize(metadata, "granitemoe.context_length")?;
    ensure!(
        max_context <= MAX_GGUF_CONTEXT_LENGTH,
        "GGUF Granite MoE context length {max_context} exceeds the supported limit {MAX_GGUF_CONTEXT_LENGTH}"
    );
    let norm_epsilon = require_f64(metadata, "granitemoe.attention.layer_norm_rms_epsilon")?;
    ensure!(
        norm_epsilon.is_finite() && norm_epsilon > 0.0,
        "GGUF Granite MoE RMS norm epsilon must be finite and positive"
    );
    Ok(GraniteMoeGeometry {
        vocab_size,
        hidden_size,
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        max_context,
        norm_epsilon,
    })
}

struct GraniteMoeScales {
    rope_theta: f64,
    attention: f64,
    embedding: f64,
    residual: f64,
    logit_divisor: f64,
}

fn granite_moe_scales(metadata: &GgufMetadata) -> Result<GraniteMoeScales> {
    let scales = GraniteMoeScales {
        rope_theta: optional_f64(metadata, "granitemoe.rope.freq_base")?.unwrap_or(10_000.0),
        attention: require_f64(metadata, "granitemoe.attention.scale")?,
        embedding: require_f64(metadata, "granitemoe.embedding_scale")?,
        residual: require_f64(metadata, "granitemoe.residual_scale")?,
        logit_divisor: require_f64(metadata, "granitemoe.logit_scale")?,
    };
    for (name, value) in [
        ("RoPE frequency base", scales.rope_theta),
        ("attention scale", scales.attention),
        ("embedding scale", scales.embedding),
        ("residual scale", scales.residual),
        ("logit divisor", scales.logit_divisor),
    ] {
        ensure!(
            value.is_finite() && value > 0.0,
            "GGUF Granite MoE {name} must be finite and positive"
        );
    }
    Ok(scales)
}

fn granite_moe_stop_token_ids(metadata: &GgufMetadata) -> Result<Vec<u32>> {
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    let mut stop_token_ids = optional_u32(metadata, "tokenizer.ggml.eos_token_id")?
        .into_iter()
        .collect::<Vec<_>>();
    if let Some(index) = tokens
        .iter()
        .position(|token| token.as_str() == Some("<|end_of_text|>"))
    {
        let token_id = u32::try_from(index).context("Granite MoE stop token ID exceeds u32")?;
        if !stop_token_ids.contains(&token_id) {
            stop_token_ids.push(token_id);
        }
    }
    Ok(stop_token_ids)
}

fn reject_granite_moe_biases_and_shared_experts(file: &GgufFile, num_layers: usize) -> Result<()> {
    for suffix in [
        "attn_q.bias",
        "attn_k.bias",
        "attn_v.bias",
        "attn_output.bias",
        "ffn_gate_inp.bias",
        "ffn_gate_exps.bias",
        "ffn_up_exps.bias",
        "ffn_down_exps.bias",
    ] {
        ensure!(
            !consistent_layer_presence(file, suffix, num_layers)?,
            "Granite MoE does not support `{suffix}`"
        );
    }
    ensure!(
        file.tensor("output.bias").is_none(),
        "GGUF output bias is not supported by the shared decoder"
    );
    ensure!(
        !file.tensors().keys().any(|name| name.contains("shexp")),
        "Granite MoE shared experts are not supported yet"
    );
    Ok(())
}

#[allow(clippy::too_many_lines)] // One fail-closed adapter keeps all Gemma geometry and fixed semantics auditable.
fn gemma_spec(file: &GgufFile) -> Result<DecoderSpec> {
    let metadata = file.metadata();
    validate_supported_gemma_metadata(metadata)?;
    ensure!(
        file.tensor("rope_freqs.weight").is_none(),
        "GGUF Gemma RoPE frequency-factor tensors are not supported"
    );

    let vocab_size = gemma_vocab_size(file)?;
    let hidden_size = required_usize(metadata, "gemma.embedding_length")?;
    let intermediate_size = required_usize(metadata, "gemma.feed_forward_length")?;
    let num_layers = required_usize(metadata, "gemma.block_count")?;
    ensure!(
        num_layers > 0 && num_layers <= MAX_DECODER_LAYERS,
        "GGUF Gemma layer count {num_layers} must be in 1..={MAX_DECODER_LAYERS}"
    );

    let num_query_heads = required_usize(metadata, "gemma.attention.head_count")?;
    let num_kv_heads =
        optional_usize(metadata, "gemma.attention.head_count_kv")?.unwrap_or(num_query_heads);
    ensure!(
        num_query_heads > 0 && num_kv_heads > 0 && num_query_heads.is_multiple_of(num_kv_heads),
        "GGUF Gemma requires positive query/KV head counts with query heads divisible by KV heads"
    );
    let head_dim = required_usize(metadata, "gemma.attention.key_length")?;
    ensure!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "GGUF Gemma attention head width must be positive and even"
    );
    ensure!(
        optional_usize(metadata, "gemma.attention.value_length")?.unwrap_or(head_dim) == head_dim,
        "different GGUF Gemma key/value head widths are not supported"
    );
    ensure!(
        num_query_heads.checked_mul(head_dim) == Some(hidden_size),
        "GGUF Gemma requires query heads * head width to equal embedding length"
    );

    let max_context = required_usize(metadata, "gemma.context_length")?;
    ensure!(
        max_context > 0 && max_context <= MAX_GGUF_CONTEXT_LENGTH,
        "GGUF Gemma context length {max_context} must be in 1..={MAX_GGUF_CONTEXT_LENGTH}"
    );
    let norm_epsilon = require_f64(metadata, "gemma.attention.layer_norm_rms_epsilon")?;
    ensure!(
        norm_epsilon.is_finite() && norm_epsilon > 0.0,
        "GGUF Gemma RMS norm epsilon must be finite and positive"
    );
    let rope_theta = optional_f64(metadata, "gemma.rope.freq_base")?.unwrap_or(10_000.0);
    ensure!(
        rope_theta.is_finite() && rope_theta > 0.0,
        "GGUF Gemma RoPE frequency base must be finite and positive"
    );
    let rotary_dim = optional_usize(metadata, "gemma.rope.dimension_count")?.unwrap_or(head_dim);
    ensure!(
        rotary_dim > 0 && rotary_dim <= head_dim && rotary_dim.is_multiple_of(2),
        "GGUF Gemma RoPE dimension must be positive, even and no greater than the head width"
    );
    reject_gemma_incompatible_tensors(file, num_layers)?;

    let layer = DecoderLayerSpec {
        mixer: SequenceMixerSpec::Attention(AttentionSpec {
            kind: AttentionKind::FullCausal,
            position_encoding: PositionEncodingSpec::Rope(RopeSpec {
                theta: rope_theta,
                layout: RopeLayout::HalfSplit,
                rotary_dim,
                frequency_factors: None,
            }),
            qk_norm: false,
            output_gate: false,
            scale: 1.0 / (head_dim as f64).sqrt(),
            logit_softcap: None,
        }),
        activation: Activation::GeluTanh,
        norm_layout: NormLayout::PreNorm,
        feed_forward: FeedForwardKind::Dense,
        mixer_residual_scale: 1.0,
        feed_forward_residual_scale: 1.0,
    };
    let spec = DecoderSpec {
        architecture: "gemma".into(),
        runtime: RuntimeSpec {
            vocab_size,
            max_context,
            stop_token_ids: gemma_stop_token_ids(metadata, "Gemma")?,
        },
        hidden_size,
        intermediate_size,
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        norm_epsilon,
        layer_specs: vec![layer; num_layers],
        embedding_scale: (hidden_size as f64).sqrt(),
        final_logit_softcap: None,
        final_logit_scale: 1.0,
        projection_bias: ProjectionBias {
            query: false,
            key: false,
            value: false,
            output: false,
        },
        mlp_bias: false,
        // llama.cpp's canonical Gemma converter omits lm_head.weight and
        // reuses token_embd.weight for logits.
        tied_embeddings: true,
    };
    spec.validate()?;
    Ok(spec)
}

fn gemma_vocab_size(file: &GgufFile) -> Result<usize> {
    let metadata = file.metadata();
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    ensure!(!tokens.is_empty(), "GGUF tokenizer vocabulary is empty");
    let embedding = file
        .tensor("token_embd.weight")
        .context("required GGUF tensor `token_embd.weight` is missing")?;
    ensure!(
        embedding.shape.len() == 2,
        "GGUF tensor `token_embd.weight` has shape {:?}, expected a rank-2 vocabulary matrix",
        embedding.shape
    );
    ensure!(
        embedding.shape[0] == tokens.len(),
        "GGUF tokenizer vocabulary has {} tokens but `token_embd.weight` has {} rows",
        tokens.len(),
        embedding.shape[0]
    );
    if let Some(declared) = optional_usize(metadata, "gemma.vocab_size")? {
        ensure!(
            declared == tokens.len(),
            "GGUF tokenizer vocabulary has {} tokens but Gemma metadata declares {declared}",
            tokens.len()
        );
    }
    Ok(tokens.len())
}

fn reject_gemma_incompatible_tensors(file: &GgufFile, num_layers: usize) -> Result<()> {
    for suffix in [
        "attn_q.bias",
        "attn_k.bias",
        "attn_v.bias",
        "attn_output.bias",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "post_attention_norm.weight",
        "post_ffw_norm.weight",
        "ffn_gate.bias",
        "ffn_up.bias",
        "ffn_down.bias",
    ] {
        ensure!(
            !consistent_layer_presence(file, suffix, num_layers)?,
            "Gemma does not support `{suffix}`"
        );
    }
    ensure!(
        file.tensor("output.bias").is_none(),
        "GGUF output bias is not supported by the shared decoder"
    );
    ensure!(
        file.tensor("output.weight").is_none(),
        "Gemma requires a tied output head and does not support `output.weight`"
    );
    Ok(())
}

fn gemma2_spec(file: &GgufFile) -> Result<DecoderSpec> {
    let metadata = file.metadata();
    validate_supported_gemma2_metadata(metadata)?;
    ensure!(
        file.tensor("rope_freqs.weight").is_none(),
        "GGUF Gemma 2 RoPE frequency-factor tensors are not supported"
    );

    let vocab_size = gemma2_vocab_size(file)?;
    let hidden_size = required_usize(metadata, "gemma2.embedding_length")?;
    let num_layers = required_usize(metadata, "gemma2.block_count")?;
    ensure!(
        num_layers > 0 && num_layers <= MAX_DECODER_LAYERS,
        "GGUF Gemma 2 layer count {num_layers} must be in 1..={MAX_DECODER_LAYERS}"
    );
    let (num_query_heads, num_kv_heads, head_dim) =
        gemma2_attention_geometry(metadata, hidden_size)?;

    let max_context = required_usize(metadata, "gemma2.context_length")?;
    ensure!(
        max_context > 0 && max_context <= MAX_GGUF_CONTEXT_LENGTH,
        "GGUF Gemma 2 context length {max_context} must be in 1..={MAX_GGUF_CONTEXT_LENGTH}"
    );
    let norm_epsilon = require_f64(metadata, "gemma2.attention.layer_norm_rms_epsilon")?;
    ensure!(
        norm_epsilon.is_finite() && norm_epsilon > 0.0,
        "GGUF Gemma 2 RMS norm epsilon must be finite and positive"
    );
    let layer_specs =
        gemma2_layer_specs(metadata, num_layers, hidden_size, num_query_heads, head_dim)?;
    reject_gemma2_incompatible_tensors(file, num_layers)?;

    let spec = DecoderSpec {
        architecture: "gemma2".into(),
        runtime: RuntimeSpec {
            vocab_size,
            max_context,
            stop_token_ids: gemma2_stop_token_ids(metadata)?,
        },
        hidden_size,
        intermediate_size: required_usize(metadata, "gemma2.feed_forward_length")?,
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        norm_epsilon,
        layer_specs,
        embedding_scale: (hidden_size as f64).sqrt(),
        final_logit_softcap: Some(require_f64(metadata, "gemma2.final_logit_softcapping")?),
        final_logit_scale: 1.0,
        projection_bias: ProjectionBias {
            query: false,
            key: false,
            value: false,
            output: false,
        },
        mlp_bias: false,
        // Canonical Gemma 2 GGUFs always reuse token_embd.weight for logits.
        // Rejecting an independent head above keeps this adapter aligned with
        // llama.cpp instead of silently accepting an unmodeled variant.
        tied_embeddings: true,
    };
    spec.validate()?;
    Ok(spec)
}

fn gemma2_attention_geometry(
    metadata: &GgufMetadata,
    hidden_size: usize,
) -> Result<(usize, usize, usize)> {
    let query_heads = required_usize(metadata, "gemma2.attention.head_count")?;
    ensure!(
        query_heads > 0,
        "GGUF Gemma 2 attention head count must be positive"
    );
    let kv_heads =
        optional_usize(metadata, "gemma2.attention.head_count_kv")?.unwrap_or(query_heads);
    ensure!(
        kv_heads > 0 && query_heads.is_multiple_of(kv_heads),
        "GGUF Gemma 2 requires positive query/KV head counts with query heads divisible by KV heads"
    );
    let head_dim = if let Some(explicit) = optional_usize(metadata, "gemma2.attention.key_length")?
    {
        explicit
    } else {
        ensure!(
            hidden_size.is_multiple_of(query_heads),
            "GGUF Gemma 2 embedding length must be divisible by query heads when key length is absent"
        );
        hidden_size / query_heads
    };
    ensure!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "GGUF Gemma 2 attention head width must be positive and even"
    );
    let value_dim = optional_usize(metadata, "gemma2.attention.value_length")?.unwrap_or(head_dim);
    ensure!(
        value_dim == head_dim,
        "different GGUF Gemma 2 key/value head widths are not supported"
    );
    Ok((query_heads, kv_heads, head_dim))
}

fn gemma2_layer_specs(
    metadata: &GgufMetadata,
    num_layers: usize,
    hidden_size: usize,
    query_heads: usize,
    head_dim: usize,
) -> Result<Vec<DecoderLayerSpec>> {
    let global_theta = optional_f64(metadata, "gemma2.rope.freq_base")?.unwrap_or(10_000.0);
    let local_theta = optional_f64(metadata, "gemma2.rope.freq_base_swa")?.unwrap_or(global_theta);
    ensure!(
        global_theta.is_finite() && global_theta > 0.0,
        "GGUF Gemma 2 global RoPE frequency base must be finite and positive"
    );
    ensure!(
        local_theta.is_finite() && local_theta > 0.0,
        "GGUF Gemma 2 local RoPE frequency base must be finite and positive"
    );
    let window = optional_usize(metadata, "gemma2.attention.sliding_window")?.unwrap_or(4_096);
    ensure!(window > 0, "GGUF Gemma 2 sliding window must be positive");
    let softcap = require_f64(metadata, "gemma2.attn_logit_softcapping")?;
    ensure!(
        softcap.is_finite() && softcap > 0.0,
        "GGUF Gemma 2 attention-logit softcap must be finite and positive"
    );
    let scale = if num_layers == 46 {
        ensure!(
            hidden_size.is_multiple_of(query_heads),
            "GGUF Gemma 2 27B embedding length must be divisible by query heads"
        );
        1.0 / ((hidden_size / query_heads) as f64).sqrt()
    } else {
        1.0 / (head_dim as f64).sqrt()
    };
    Ok(gemma2_sliding_layers(metadata, num_layers)?
        .into_iter()
        .map(|sliding| DecoderLayerSpec {
            mixer: SequenceMixerSpec::Attention(AttentionSpec {
                kind: if sliding {
                    AttentionKind::SlidingCausal { window }
                } else {
                    AttentionKind::FullCausal
                },
                position_encoding: PositionEncodingSpec::Rope(RopeSpec {
                    theta: if sliding { local_theta } else { global_theta },
                    layout: RopeLayout::HalfSplit,
                    rotary_dim: head_dim,
                    frequency_factors: None,
                }),
                qk_norm: false,
                output_gate: false,
                scale,
                logit_softcap: Some(softcap),
            }),
            activation: Activation::GeluTanh,
            norm_layout: NormLayout::Sandwich,
            feed_forward: FeedForwardKind::Dense,
            mixer_residual_scale: 1.0,
            feed_forward_residual_scale: 1.0,
        })
        .collect())
}

fn reject_gemma2_incompatible_tensors(file: &GgufFile, num_layers: usize) -> Result<()> {
    for suffix in [
        "attn_q.bias",
        "attn_k.bias",
        "attn_v.bias",
        "attn_output.bias",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "ffn_gate.bias",
        "ffn_up.bias",
        "ffn_down.bias",
    ] {
        ensure!(
            !consistent_layer_presence(file, suffix, num_layers)?,
            "Gemma 2 does not support `{suffix}`"
        );
    }
    ensure!(
        file.tensor("output.bias").is_none(),
        "GGUF output bias is not supported by the shared decoder"
    );
    ensure!(
        file.tensor("output.weight").is_none(),
        "Gemma 2 requires a tied output head and does not support `output.weight`"
    );
    Ok(())
}

fn gemma2_sliding_layers(metadata: &GgufMetadata, num_layers: usize) -> Result<Vec<bool>> {
    let Some(pattern) = metadata.get("gemma2.attention.sliding_window_pattern") else {
        return Ok((0..num_layers).map(|layer| layer % 2 == 0).collect());
    };
    match pattern {
        GgufValue::U32(period) => {
            let period = usize::try_from(*period)?;
            // Match llama.cpp's set_swa_pattern contract: zero means every
            // layer is sliding, one means every layer is dense, and larger
            // periods use N-1 sliding layers followed by one dense layer.
            if period == 0 {
                Ok(vec![true; num_layers])
            } else {
                Ok((0..num_layers)
                    .map(|layer| layer % period < period - 1)
                    .collect())
            }
        }
        GgufValue::Array {
            element_type: GgufValueType::Bool,
            values,
        } => {
            ensure!(
                values.len() == num_layers,
                "Gemma 2 sliding-window pattern has {} entries for {num_layers} layers",
                values.len()
            );
            values
                .iter()
                .enumerate()
                .map(|(index, value)| match value {
                    GgufValue::Bool(value) => Ok(*value),
                    _ => bail!("Gemma 2 sliding-window pattern element {index} is not BOOL"),
                })
                .collect()
        }
        _ => bail!(
            "GGUF metadata `gemma2.attention.sliding_window_pattern` must be UINT32 or ARRAY<BOOL>"
        ),
    }
}

fn gemma2_vocab_size(file: &GgufFile) -> Result<usize> {
    let metadata = file.metadata();
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    ensure!(!tokens.is_empty(), "GGUF tokenizer vocabulary is empty");
    let embedding = file
        .tensor("token_embd.weight")
        .context("required GGUF tensor `token_embd.weight` is missing")?;
    ensure!(
        embedding.shape.len() == 2,
        "GGUF tensor `token_embd.weight` has shape {:?}, expected a rank-2 vocabulary matrix",
        embedding.shape
    );
    ensure!(
        embedding.shape[0] == tokens.len(),
        "GGUF tokenizer vocabulary has {} tokens but `token_embd.weight` has {} rows",
        tokens.len(),
        embedding.shape[0]
    );
    if let Some(declared) = optional_usize(metadata, "gemma2.vocab_size")? {
        ensure!(
            declared == tokens.len(),
            "GGUF tokenizer vocabulary has {} tokens but Gemma 2 metadata declares {declared}",
            tokens.len()
        );
    }
    Ok(tokens.len())
}

fn gemma2_stop_token_ids(metadata: &GgufMetadata) -> Result<Vec<u32>> {
    gemma_stop_token_ids(metadata, "Gemma 2")
}

fn gemma3_spec(file: &GgufFile) -> Result<DecoderSpec> {
    let metadata = file.metadata();
    validate_supported_gemma3_metadata(metadata)?;

    let vocab_size = gemma3_vocab_size(file)?;
    let hidden_size = required_usize(metadata, "gemma3.embedding_length")?;
    let num_layers = required_usize(metadata, "gemma3.block_count")?;
    ensure!(
        num_layers <= MAX_DECODER_LAYERS,
        "GGUF Gemma 3 layer count {num_layers} exceeds the supported limit {MAX_DECODER_LAYERS}"
    );
    let (num_query_heads, num_kv_heads, head_dim) =
        gemma3_attention_geometry(metadata, hidden_size)?;

    let max_context = required_usize(metadata, "gemma3.context_length")?;
    ensure!(
        max_context <= MAX_GGUF_CONTEXT_LENGTH,
        "GGUF Gemma 3 context length {max_context} exceeds the supported limit {MAX_GGUF_CONTEXT_LENGTH}"
    );
    let norm_epsilon = require_f64(metadata, "gemma3.attention.layer_norm_rms_epsilon")?;
    ensure!(
        norm_epsilon.is_finite() && norm_epsilon > 0.0,
        "GGUF Gemma 3 RMS norm epsilon must be finite and positive"
    );
    let layer_specs =
        gemma3_layer_specs(metadata, num_layers, hidden_size, num_query_heads, head_dim)?;
    reject_gemma3_biases(file, num_layers)?;

    let spec = DecoderSpec {
        architecture: "gemma3".into(),
        runtime: RuntimeSpec {
            vocab_size,
            max_context,
            stop_token_ids: gemma3_stop_token_ids(metadata)?,
        },
        hidden_size,
        intermediate_size: required_usize(metadata, "gemma3.feed_forward_length")?,
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        norm_epsilon,
        layer_specs,
        embedding_scale: (hidden_size as f64).sqrt(),
        final_logit_softcap: optional_f64(metadata, "gemma3.final_logit_softcapping")?,
        final_logit_scale: 1.0,
        projection_bias: ProjectionBias {
            query: false,
            key: false,
            value: false,
            output: false,
        },
        mlp_bias: false,
        tied_embeddings: file.tensor("output.weight").is_none(),
    };
    spec.validate()?;
    Ok(spec)
}

fn gemma3_attention_geometry(
    metadata: &GgufMetadata,
    hidden_size: usize,
) -> Result<(usize, usize, usize)> {
    let query_heads = required_usize(metadata, "gemma3.attention.head_count")?;
    ensure!(
        query_heads > 0,
        "Gemma 3 attention head count must be positive"
    );
    let kv_heads =
        optional_usize(metadata, "gemma3.attention.head_count_kv")?.unwrap_or(query_heads);
    ensure!(kv_heads > 0, "Gemma 3 KV head count must be positive");
    let head_dim = if let Some(explicit) = optional_usize(metadata, "gemma3.attention.key_length")?
    {
        explicit
    } else {
        ensure!(
            hidden_size.is_multiple_of(query_heads),
            "Gemma 3 embedding length must be divisible by query heads when key length is absent"
        );
        hidden_size / query_heads
    };
    let value_dim = optional_usize(metadata, "gemma3.attention.value_length")?.unwrap_or(head_dim);
    ensure!(
        value_dim == head_dim,
        "different Gemma 3 key/value head widths are not supported"
    );
    Ok((query_heads, kv_heads, head_dim))
}

fn gemma3_layer_specs(
    metadata: &GgufMetadata,
    num_layers: usize,
    hidden_size: usize,
    query_heads: usize,
    head_dim: usize,
) -> Result<Vec<DecoderLayerSpec>> {
    let global_theta = optional_f64(metadata, "gemma3.rope.freq_base")?.unwrap_or(1_000_000.0);
    let local_theta = optional_f64(metadata, "gemma3.rope.freq_base_swa")?.unwrap_or(10_000.0);
    ensure!(
        global_theta.is_finite() && global_theta > 0.0,
        "GGUF Gemma 3 global RoPE frequency base must be finite and positive"
    );
    ensure!(
        local_theta.is_finite() && local_theta > 0.0,
        "GGUF Gemma 3 local RoPE frequency base must be finite and positive"
    );
    let window = required_usize(metadata, "gemma3.attention.sliding_window")?;
    ensure!(window > 0, "GGUF Gemma 3 sliding window must be positive");
    let scale = if num_layers == 62 {
        1.0 / ((hidden_size / query_heads) as f64).sqrt()
    } else {
        1.0 / (head_dim as f64).sqrt()
    };
    Ok(gemma3_sliding_layers(metadata, num_layers)?
        .into_iter()
        .map(|sliding| DecoderLayerSpec {
            mixer: SequenceMixerSpec::Attention(AttentionSpec {
                kind: if sliding {
                    AttentionKind::SlidingCausal { window }
                } else {
                    AttentionKind::FullCausal
                },
                position_encoding: PositionEncodingSpec::Rope(RopeSpec {
                    theta: if sliding { local_theta } else { global_theta },
                    layout: RopeLayout::HalfSplit,
                    rotary_dim: head_dim,
                    frequency_factors: None,
                }),
                qk_norm: true,
                output_gate: false,
                scale,
                logit_softcap: None,
            }),
            activation: Activation::GeluTanh,
            norm_layout: NormLayout::Sandwich,
            feed_forward: FeedForwardKind::Dense,
            mixer_residual_scale: 1.0,
            feed_forward_residual_scale: 1.0,
        })
        .collect())
}

fn reject_gemma3_biases(file: &GgufFile, num_layers: usize) -> Result<()> {
    for suffix in [
        "attn_q.bias",
        "attn_k.bias",
        "attn_v.bias",
        "attn_output.bias",
        "ffn_gate.bias",
        "ffn_up.bias",
        "ffn_down.bias",
    ] {
        ensure!(
            !consistent_layer_presence(file, suffix, num_layers)?,
            "Gemma 3 does not support `{suffix}`"
        );
    }
    ensure!(
        file.tensor("output.bias").is_none(),
        "GGUF output bias is not supported by the shared decoder"
    );
    Ok(())
}

fn gemma3_sliding_layers(metadata: &GgufMetadata, num_layers: usize) -> Result<Vec<bool>> {
    let Some(pattern) = metadata.get("gemma3.attention.sliding_window_pattern") else {
        return Ok((0..num_layers).map(|layer| layer % 6 < 5).collect());
    };
    match pattern {
        GgufValue::U32(period) => {
            let period = usize::try_from(*period)?;
            ensure!(period > 0, "Gemma 3 sliding-window period must be positive");
            Ok((0..num_layers)
                .map(|layer| layer % period < period - 1)
                .collect())
        }
        GgufValue::Array {
            element_type: GgufValueType::Bool,
            values,
        } => {
            ensure!(
                values.len() == num_layers,
                "Gemma 3 sliding-window pattern has {} entries for {num_layers} layers",
                values.len()
            );
            values
                .iter()
                .enumerate()
                .map(|(index, value)| match value {
                    GgufValue::Bool(value) => Ok(*value),
                    _ => bail!("Gemma 3 sliding-window pattern element {index} is not BOOL"),
                })
                .collect()
        }
        _ => bail!(
            "GGUF metadata `gemma3.attention.sliding_window_pattern` must be UINT32 or ARRAY<BOOL>"
        ),
    }
}

fn gemma3_vocab_size(file: &GgufFile) -> Result<usize> {
    let metadata = file.metadata();
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    ensure!(!tokens.is_empty(), "GGUF tokenizer vocabulary is empty");
    let embedding = file
        .tensor("token_embd.weight")
        .context("required GGUF tensor `token_embd.weight` is missing")?;
    ensure!(
        embedding.shape.len() == 2,
        "GGUF tensor `token_embd.weight` has shape {:?}, expected a rank-2 vocabulary matrix",
        embedding.shape
    );
    ensure!(
        embedding.shape[0] == tokens.len(),
        "GGUF tokenizer vocabulary has {} tokens but `token_embd.weight` has {} rows",
        tokens.len(),
        embedding.shape[0]
    );
    if let Some(declared) = optional_usize(metadata, "gemma3.vocab_size")? {
        ensure!(
            declared == tokens.len(),
            "GGUF tokenizer vocabulary has {} tokens but Gemma 3 metadata declares {declared}",
            tokens.len()
        );
    }
    Ok(tokens.len())
}

fn gemma3_stop_token_ids(metadata: &GgufMetadata) -> Result<Vec<u32>> {
    gemma_stop_token_ids(metadata, "Gemma 3")
}

fn gemma_stop_token_ids(metadata: &GgufMetadata, family: &str) -> Result<Vec<u32>> {
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    let mut stop_token_ids = optional_u32(metadata, "tokenizer.ggml.eos_token_id")?
        .into_iter()
        .collect::<Vec<_>>();
    if let Some(index) = tokens
        .iter()
        .position(|token| token.as_str() == Some("<end_of_turn>"))
    {
        let token_id = u32::try_from(index)
            .with_context(|| format!("{family} end-of-turn token ID exceeds u32"))?;
        if !stop_token_ids.contains(&token_id) {
            stop_token_ids.push(token_id);
        }
    }
    Ok(stop_token_ids)
}

#[allow(clippy::too_many_lines)] // Geometry, tensor-presence and RoPE validation form one contract.
fn llama_spec(file: &GgufFile) -> Result<DecoderSpec> {
    let metadata = file.metadata();
    validate_supported_llama_metadata(metadata)?;

    let vocab_size = required_usize(metadata, "llama.vocab_size")?;
    let hidden_size = required_usize(metadata, "llama.embedding_length")?;
    let num_layers = required_usize(metadata, "llama.block_count")?;
    ensure!(
        num_layers <= MAX_DECODER_LAYERS,
        "GGUF Llama layer count {num_layers} exceeds the supported limit {MAX_DECODER_LAYERS}"
    );
    let num_query_heads = required_usize(metadata, "llama.attention.head_count")?;
    ensure!(
        num_query_heads > 0,
        "llama attention head count must be positive"
    );
    let num_kv_heads =
        optional_usize(metadata, "llama.attention.head_count_kv")?.unwrap_or(num_query_heads);
    ensure!(num_kv_heads > 0, "llama KV head count must be positive");
    let head_dim = if let Some(explicit) = optional_usize(metadata, "llama.attention.key_length")? {
        explicit
    } else {
        ensure!(
            hidden_size % num_query_heads == 0,
            "llama embedding length must be divisible by query heads when key length is absent"
        );
        hidden_size / num_query_heads
    };
    let value_dim = optional_usize(metadata, "llama.attention.value_length")?.unwrap_or(head_dim);
    ensure!(
        value_dim == head_dim,
        "different Llama key/value head widths are not supported"
    );
    let rope_dimension =
        optional_usize(metadata, "llama.rope.dimension_count")?.unwrap_or(head_dim);
    ensure!(
        rope_dimension == head_dim,
        "partial Llama RoPE is not supported: dimension_count={rope_dimension}, head_dim={head_dim}"
    );
    let rope_frequency_factors = gguf_rope_frequency_factors(file, head_dim)?;

    let max_context = required_usize(metadata, "llama.context_length")?;
    ensure!(
        max_context <= MAX_GGUF_CONTEXT_LENGTH,
        "GGUF Llama context length {max_context} exceeds the supported limit {MAX_GGUF_CONTEXT_LENGTH}"
    );
    let attention_kind = match optional_usize(metadata, "llama.attention.sliding_window")? {
        Some(0) | None => AttentionKind::FullCausal,
        Some(window) => {
            ensure!(
                window <= max_context,
                "GGUF Llama sliding window {window} exceeds context length {max_context}"
            );
            AttentionKind::SlidingCausal { window }
        }
    };
    let norm_epsilon = require_f64(metadata, "llama.attention.layer_norm_rms_epsilon")?;
    ensure!(
        norm_epsilon.is_finite() && norm_epsilon > 0.0,
        "GGUF Llama RMS norm epsilon must be finite and positive"
    );
    let rope_theta = optional_f64(metadata, "llama.rope.freq_base")?.unwrap_or(10_000.0);
    ensure!(
        rope_theta.is_finite() && rope_theta > 0.0,
        "GGUF Llama RoPE frequency base must be finite and positive"
    );

    let projection_bias = ProjectionBias {
        query: consistent_layer_presence(file, "attn_q.bias", num_layers)?,
        key: consistent_layer_presence(file, "attn_k.bias", num_layers)?,
        value: consistent_layer_presence(file, "attn_v.bias", num_layers)?,
        output: consistent_layer_presence(file, "attn_output.bias", num_layers)?,
    };
    let intermediate_size = required_usize(metadata, "llama.feed_forward_length")?;
    let (feed_forward, mlp_bias) = llama_feed_forward_profile(file, num_layers, intermediate_size)?;
    ensure!(
        file.tensor("output.bias").is_none(),
        "GGUF output bias is not supported by the shared decoder"
    );

    let stop_token_ids = llama_stop_token_ids(metadata)?;
    let rope = RopeSpec {
        theta: rope_theta,
        layout: RopeLayout::HalfSplit,
        rotary_dim: head_dim,
        frequency_factors: rope_frequency_factors,
    };
    let mut layer = DecoderLayerSpec::standard(rope, head_dim);
    layer
        .attention_mut()
        .expect("standard Llama layer uses attention")
        .kind = attention_kind;
    layer.feed_forward = feed_forward;
    let spec = DecoderSpec {
        architecture: "llama".into(),
        runtime: RuntimeSpec {
            vocab_size,
            max_context,
            stop_token_ids,
        },
        hidden_size,
        intermediate_size,
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        norm_epsilon,
        layer_specs: vec![layer; num_layers],
        embedding_scale: 1.0,
        final_logit_softcap: None,
        final_logit_scale: 1.0,
        projection_bias,
        mlp_bias,
        tied_embeddings: file.tensor("output.weight").is_none(),
    };
    spec.validate()?;
    Ok(spec)
}

fn llama_feed_forward_profile(
    file: &GgufFile,
    num_layers: usize,
    intermediate_size: usize,
) -> Result<(FeedForwardKind, bool)> {
    let metadata = file.metadata();
    let expert_count = optional_usize(metadata, "llama.expert_count")?.unwrap_or(0);
    let experts_per_token = optional_usize(metadata, "llama.expert_used_count")?.unwrap_or(0);
    let expert_weights_scale = optional_f64(metadata, "llama.expert_weights_scale")?.unwrap_or(1.0);
    ensure!(
        expert_weights_scale.is_finite() && (expert_weights_scale - 1.0).abs() < f64::EPSILON,
        "GGUF Llama expert weight scale {expert_weights_scale} is not supported; only 1 is modeled"
    );

    let expert_tensor = file.tensors().keys().find(|name| {
        name.contains(".ffn_gate_inp.") || name.contains("_exps.") || name.contains("shexp")
    });

    if expert_count == 0 {
        ensure!(
            experts_per_token == 0,
            "GGUF Llama expert_used_count requires a positive expert_count"
        );
        if let Some(name) = expert_tensor {
            bail!(
                "dense Llama contains sparse expert tensor `{name}` without a positive expert_count"
            );
        }
        let gate_bias = consistent_layer_presence(file, "ffn_gate.bias", num_layers)?;
        let up_bias = consistent_layer_presence(file, "ffn_up.bias", num_layers)?;
        let down_bias = consistent_layer_presence(file, "ffn_down.bias", num_layers)?;
        ensure!(
            gate_bias == up_bias && up_bias == down_bias,
            "the shared decoder requires all Llama MLP projections to agree on bias presence"
        );
        return Ok((FeedForwardKind::Dense, gate_bias));
    }

    ensure!(
        experts_per_token > 0 && experts_per_token <= expert_count,
        "GGUF Llama requires expert_used_count in 1..={expert_count}"
    );
    // `.bias` is a case-sensitive GGUF tensor-name suffix, not a filesystem
    // extension; accepting alternate casing would weaken the tensor contract.
    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    if let Some(name) = file
        .tensors()
        .keys()
        .find(|name| name.contains(".ffn_") && name.ends_with(".bias"))
    {
        bail!("sparse-MoE Llama does not support FFN bias tensor `{name}`");
    }
    ensure!(
        !file.tensors().keys().any(|name| name.contains("shexp")),
        "sparse-MoE Llama shared experts are not supported"
    );
    if let Some(name) = file.tensors().keys().find(|name| {
        (name.contains(".ffn_gate_inp.") || name.contains("_exps."))
            && ![
                ".ffn_gate_inp.weight",
                ".ffn_gate_exps.weight",
                ".ffn_up_exps.weight",
                ".ffn_down_exps.weight",
            ]
            .iter()
            .any(|suffix| name.ends_with(suffix))
    }) {
        bail!("sparse-MoE Llama contains unsupported expert tensor `{name}`");
    }
    Ok((
        FeedForwardKind::SparseMoe(SparseMoeSpec {
            expert_count,
            experts_per_token,
            expert_intermediate_size: intermediate_size,
            router_weighting: RouterWeighting::SelectedSoftmax,
        }),
        false,
    ))
}

fn gguf_rope_frequency_factors(file: &GgufFile, head_dim: usize) -> Result<Option<Vec<f32>>> {
    if file.tensor("rope_freqs.weight").is_none() {
        return Ok(None);
    }
    let factors = file.read_f32_vector("rope_freqs.weight")?;
    ensure!(
        factors.len() == head_dim / 2,
        "GGUF `rope_freqs.weight` has {} factors, expected {} for head dimension {head_dim}",
        factors.len(),
        head_dim / 2
    );
    ensure!(
        factors
            .iter()
            .all(|factor| factor.is_finite() && *factor > 0.0),
        "GGUF `rope_freqs.weight` factors must be finite and positive"
    );
    Ok(Some(factors))
}

fn qwen2_spec(file: &GgufFile) -> Result<DecoderSpec> {
    let metadata = file.metadata();
    validate_supported_qwen2_metadata(metadata)?;
    ensure!(
        file.tensor("rope_freqs.weight").is_none(),
        "GGUF Qwen2 rope frequency factors are not supported yet"
    );

    let vocab_size = qwen2_vocab_size(file)?;
    let hidden_size = required_usize(metadata, "qwen2.embedding_length")?;
    let num_layers = required_usize(metadata, "qwen2.block_count")?;
    ensure!(
        num_layers <= MAX_DECODER_LAYERS,
        "GGUF Qwen2 layer count {num_layers} exceeds the supported limit {MAX_DECODER_LAYERS}"
    );
    let num_query_heads = required_usize(metadata, "qwen2.attention.head_count")?;
    ensure!(
        num_query_heads > 0,
        "Qwen2 attention head count must be positive"
    );
    let num_kv_heads =
        optional_usize(metadata, "qwen2.attention.head_count_kv")?.unwrap_or(num_query_heads);
    ensure!(num_kv_heads > 0, "Qwen2 KV head count must be positive");
    let head_dim = if let Some(explicit) = optional_usize(metadata, "qwen2.attention.key_length")? {
        explicit
    } else {
        ensure!(
            hidden_size % num_query_heads == 0,
            "Qwen2 embedding length must be divisible by query heads when key length is absent"
        );
        hidden_size / num_query_heads
    };
    let value_dim = optional_usize(metadata, "qwen2.attention.value_length")?.unwrap_or(head_dim);
    ensure!(
        value_dim == head_dim,
        "different Qwen2 key/value head widths are not supported"
    );
    let rope_dimension =
        optional_usize(metadata, "qwen2.rope.dimension_count")?.unwrap_or(head_dim);
    ensure!(
        rope_dimension == head_dim,
        "partial Qwen2 RoPE is not supported: dimension_count={rope_dimension}, head_dim={head_dim}"
    );

    let max_context = required_usize(metadata, "qwen2.context_length")?;
    ensure!(
        max_context <= MAX_GGUF_CONTEXT_LENGTH,
        "GGUF Qwen2 context length {max_context} exceeds the supported limit {MAX_GGUF_CONTEXT_LENGTH}"
    );
    let norm_epsilon = require_f64(metadata, "qwen2.attention.layer_norm_rms_epsilon")?;
    ensure!(
        norm_epsilon.is_finite() && norm_epsilon > 0.0,
        "GGUF Qwen2 RMS norm epsilon must be finite and positive"
    );
    let rope_theta = optional_f64(metadata, "qwen2.rope.freq_base")?.unwrap_or(10_000.0);
    ensure!(
        rope_theta.is_finite() && rope_theta > 0.0,
        "GGUF Qwen2 RoPE frequency base must be finite and positive"
    );

    validate_qwen2_biases(file, num_layers)?;

    let rope = RopeSpec {
        theta: rope_theta,
        layout: RopeLayout::HalfSplit,
        rotary_dim: head_dim,
        frequency_factors: None,
    };
    let spec = DecoderSpec {
        architecture: "qwen2".into(),
        runtime: RuntimeSpec {
            vocab_size,
            max_context,
            stop_token_ids: qwen2_stop_token_ids(metadata)?,
        },
        hidden_size,
        intermediate_size: required_usize(metadata, "qwen2.feed_forward_length")?,
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        norm_epsilon,
        layer_specs: vec![DecoderLayerSpec::standard(rope, head_dim); num_layers],
        embedding_scale: 1.0,
        final_logit_softcap: None,
        final_logit_scale: 1.0,
        projection_bias: ProjectionBias {
            query: true,
            key: true,
            value: true,
            output: false,
        },
        mlp_bias: false,
        tied_embeddings: file.tensor("output.weight").is_none(),
    };
    spec.validate()?;
    Ok(spec)
}

fn qwen3_spec(file: &GgufFile) -> Result<DecoderSpec> {
    let metadata = file.metadata();
    validate_supported_qwen3_metadata(metadata)?;
    ensure!(
        file.tensor("rope_freqs.weight").is_none(),
        "GGUF Qwen3 rope frequency factors are not supported yet"
    );

    let vocab_size = qwen3_vocab_size(file)?;
    let hidden_size = required_usize(metadata, "qwen3.embedding_length")?;
    let num_layers = required_usize(metadata, "qwen3.block_count")?;
    ensure!(
        num_layers <= MAX_DECODER_LAYERS,
        "GGUF Qwen3 layer count {num_layers} exceeds the supported limit {MAX_DECODER_LAYERS}"
    );
    let num_query_heads = required_usize(metadata, "qwen3.attention.head_count")?;
    ensure!(
        num_query_heads > 0,
        "Qwen3 attention head count must be positive"
    );
    let num_kv_heads =
        optional_usize(metadata, "qwen3.attention.head_count_kv")?.unwrap_or(num_query_heads);
    ensure!(num_kv_heads > 0, "Qwen3 KV head count must be positive");
    // Qwen3-0.6B deliberately has hidden_size / query_heads = 64 while its
    // projection head width is 128, so this field is architectural rather
    // than derivable geometry.
    let head_dim = required_usize(metadata, "qwen3.attention.key_length")?;
    let value_dim = optional_usize(metadata, "qwen3.attention.value_length")?.unwrap_or(head_dim);
    ensure!(
        value_dim == head_dim,
        "different Qwen3 key/value head widths are not supported"
    );
    let rope_dimension =
        optional_usize(metadata, "qwen3.rope.dimension_count")?.unwrap_or(head_dim);
    ensure!(
        rope_dimension == head_dim,
        "partial Qwen3 RoPE is not supported: dimension_count={rope_dimension}, head_dim={head_dim}"
    );

    let max_context = required_usize(metadata, "qwen3.context_length")?;
    ensure!(
        max_context <= MAX_GGUF_CONTEXT_LENGTH,
        "GGUF Qwen3 context length {max_context} exceeds the supported limit {MAX_GGUF_CONTEXT_LENGTH}"
    );
    let norm_epsilon = require_f64(metadata, "qwen3.attention.layer_norm_rms_epsilon")?;
    ensure!(
        norm_epsilon.is_finite() && norm_epsilon > 0.0,
        "GGUF Qwen3 RMS norm epsilon must be finite and positive"
    );
    let rope_theta = optional_f64(metadata, "qwen3.rope.freq_base")?.unwrap_or(10_000.0);
    ensure!(
        rope_theta.is_finite() && rope_theta > 0.0,
        "GGUF Qwen3 RoPE frequency base must be finite and positive"
    );

    validate_qwen3_biases(file, num_layers)?;

    let rope = RopeSpec {
        theta: rope_theta,
        layout: RopeLayout::HalfSplit,
        rotary_dim: head_dim,
        frequency_factors: None,
    };
    let mut layer = DecoderLayerSpec::standard(rope, head_dim);
    layer
        .attention_mut()
        .expect("standard decoder layer uses attention")
        .qk_norm = true;
    let spec = DecoderSpec {
        architecture: "qwen3".into(),
        runtime: RuntimeSpec {
            vocab_size,
            max_context,
            stop_token_ids: qwen_stop_token_ids(metadata, "Qwen3")?,
        },
        hidden_size,
        intermediate_size: required_usize(metadata, "qwen3.feed_forward_length")?,
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        norm_epsilon,
        layer_specs: vec![layer; num_layers],
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
        tied_embeddings: file.tensor("output.weight").is_none(),
    };
    spec.validate()?;
    Ok(spec)
}

/// Lower the current dense `SmolLM3` GGUF contract into the shared decoder.
///
/// `SmolLM3` uses ordinary GQA attention and `SwiGLU` projections, but deliberately
/// omits `RoPE` in every fourth one-based layer. Keeping that schedule in the
/// architecture profile makes `NoPE` explicit instead of encoding it as a
/// degenerate rotary dimension.
#[allow(clippy::too_many_lines)] // Geometry and the fail-closed tensor contract form one profile.
fn smollm3_spec(file: &GgufFile) -> Result<DecoderSpec> {
    let metadata = file.metadata();
    validate_supported_smollm3_metadata(metadata)?;
    ensure!(
        file.tensor("rope_freqs.weight").is_none(),
        "GGUF SmolLM3 rope frequency factors are not supported"
    );

    let vocab_size = smollm3_vocab_size(file)?;
    let hidden_size = required_usize(metadata, "smollm3.embedding_length")?;
    let num_layers = required_usize(metadata, "smollm3.block_count")?;
    ensure!(
        num_layers == 36,
        "the current GGUF SmolLM3 profile requires 36 decoder layers, got {num_layers}"
    );
    ensure!(
        num_layers <= MAX_DECODER_LAYERS,
        "GGUF SmolLM3 layer count {num_layers} exceeds the supported limit {MAX_DECODER_LAYERS}"
    );
    let num_query_heads = required_usize(metadata, "smollm3.attention.head_count")?;
    ensure!(
        num_query_heads > 0,
        "SmolLM3 attention head count must be positive"
    );
    let num_kv_heads =
        optional_usize(metadata, "smollm3.attention.head_count_kv")?.unwrap_or(num_query_heads);
    ensure!(num_kv_heads > 0, "SmolLM3 KV head count must be positive");
    let head_dim = if let Some(explicit) = optional_usize(metadata, "smollm3.attention.key_length")?
    {
        explicit
    } else {
        ensure!(
            hidden_size % num_query_heads == 0,
            "SmolLM3 embedding length must be divisible by query heads when key length is absent"
        );
        hidden_size / num_query_heads
    };
    let value_dim = optional_usize(metadata, "smollm3.attention.value_length")?.unwrap_or(head_dim);
    ensure!(
        value_dim == head_dim,
        "different SmolLM3 key/value head widths are not supported"
    );
    let rope_dimension =
        optional_usize(metadata, "smollm3.rope.dimension_count")?.unwrap_or(head_dim);
    ensure!(
        rope_dimension == head_dim,
        "partial SmolLM3 RoPE is not supported: dimension_count={rope_dimension}, head_dim={head_dim}"
    );

    let max_context = required_usize(metadata, "smollm3.context_length")?;
    ensure!(
        max_context <= MAX_GGUF_CONTEXT_LENGTH,
        "GGUF SmolLM3 context length {max_context} exceeds the supported limit {MAX_GGUF_CONTEXT_LENGTH}"
    );
    let norm_epsilon = require_f64(metadata, "smollm3.attention.layer_norm_rms_epsilon")?;
    ensure!(
        norm_epsilon.is_finite() && norm_epsilon > 0.0,
        "GGUF SmolLM3 RMS norm epsilon must be finite and positive"
    );
    let rope_theta = optional_f64(metadata, "smollm3.rope.freq_base")?.unwrap_or(10_000.0);
    ensure!(
        rope_theta.is_finite() && rope_theta > 0.0,
        "GGUF SmolLM3 RoPE frequency base must be finite and positive"
    );

    validate_smollm3_tensor_contract(file, num_layers)?;

    let rope = RopeSpec {
        theta: rope_theta,
        layout: RopeLayout::HalfSplit,
        rotary_dim: head_dim,
        frequency_factors: None,
    };
    let layer_specs = (0..num_layers)
        .map(|index| {
            let mut layer = DecoderLayerSpec::standard(rope.clone(), head_dim);
            if (index + 1).is_multiple_of(4) {
                layer
                    .attention_mut()
                    .expect("standard SmolLM3 layer uses attention")
                    .position_encoding = PositionEncodingSpec::None;
            }
            layer
        })
        .collect();
    let spec = DecoderSpec {
        architecture: "smollm3".into(),
        runtime: RuntimeSpec {
            vocab_size,
            max_context,
            stop_token_ids: qwen_stop_token_ids(metadata, "SmolLM3")?,
        },
        hidden_size,
        intermediate_size: required_usize(metadata, "smollm3.feed_forward_length")?,
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        norm_epsilon,
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
        tied_embeddings: file.tensor("output.weight").is_none(),
    };
    spec.validate()?;
    Ok(spec)
}

fn validate_smollm3_tensor_contract(file: &GgufFile, num_layers: usize) -> Result<()> {
    const GLOBAL_TENSORS: &[&str] = &["token_embd.weight", "output_norm.weight", "output.weight"];
    const LAYER_SUFFIXES: &[&str] = &[
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ];

    for name in file.tensors().keys() {
        if GLOBAL_TENSORS.contains(&name.as_str()) {
            continue;
        }
        let Some(layer_tensor) = name.strip_prefix("blk.") else {
            bail!("GGUF SmolLM3 contains unsupported tensor `{name}`");
        };
        let Some((layer, suffix)) = layer_tensor.split_once('.') else {
            bail!("GGUF SmolLM3 contains malformed layer tensor `{name}`");
        };
        let layer = layer
            .parse::<usize>()
            .with_context(|| format!("GGUF SmolLM3 tensor `{name}` has an invalid layer index"))?;
        ensure!(
            layer < num_layers,
            "GGUF SmolLM3 tensor `{name}` uses layer {layer}, outside the declared {num_layers}-layer trunk"
        );
        ensure!(
            LAYER_SUFFIXES.contains(&suffix),
            "GGUF SmolLM3 contains unsupported tensor `{name}`"
        );
    }
    Ok(())
}

/// Lower the current llama.cpp `qwen3moe` GGUF contract into the shared
/// Qwen3-attention plus sparse-MoE decoder capabilities.
#[allow(clippy::too_many_lines)] // One strict profile keeps every modeled metadata semantic visible.
fn qwen3_moe_spec(file: &GgufFile) -> Result<DecoderSpec> {
    let metadata = file.metadata();
    validate_supported_qwen3_moe_metadata(metadata)?;
    ensure!(
        file.tensor("rope_freqs.weight").is_none(),
        "GGUF Qwen3MoE rope frequency factors are not supported yet"
    );

    let vocab_size = qwen3_moe_vocab_size(file)?;
    let hidden_size = required_usize(metadata, "qwen3moe.embedding_length")?;
    let num_layers = required_usize(metadata, "qwen3moe.block_count")?;
    ensure!(
        num_layers > 0 && num_layers <= MAX_DECODER_LAYERS,
        "GGUF Qwen3MoE layer count {num_layers} must be in 1..={MAX_DECODER_LAYERS}"
    );
    let num_query_heads = required_usize(metadata, "qwen3moe.attention.head_count")?;
    let num_kv_heads =
        optional_usize(metadata, "qwen3moe.attention.head_count_kv")?.unwrap_or(num_query_heads);
    ensure!(
        num_query_heads > 0 && num_kv_heads > 0 && num_query_heads.is_multiple_of(num_kv_heads),
        "GGUF Qwen3MoE requires positive query/KV head counts with query heads divisible by KV heads"
    );

    // Qwen3MoE projections may be wider than the residual stream, so this is
    // an explicit architecture field rather than hidden_size / query_heads.
    let head_dim = required_usize(metadata, "qwen3moe.attention.key_length")?;
    let value_dim =
        optional_usize(metadata, "qwen3moe.attention.value_length")?.unwrap_or(head_dim);
    ensure!(
        value_dim == head_dim,
        "different Qwen3MoE key/value head widths are not supported"
    );
    let rope_dimension =
        optional_usize(metadata, "qwen3moe.rope.dimension_count")?.unwrap_or(head_dim);
    ensure!(
        rope_dimension == head_dim,
        "partial Qwen3MoE RoPE is not supported: dimension_count={rope_dimension}, head_dim={head_dim}"
    );

    let max_context = required_usize(metadata, "qwen3moe.context_length")?;
    ensure!(
        max_context > 0 && max_context <= MAX_GGUF_CONTEXT_LENGTH,
        "GGUF Qwen3MoE context length {max_context} must be in 1..={MAX_GGUF_CONTEXT_LENGTH}"
    );
    let norm_epsilon = require_f64(metadata, "qwen3moe.attention.layer_norm_rms_epsilon")?;
    ensure!(
        norm_epsilon.is_finite() && norm_epsilon > 0.0,
        "GGUF Qwen3MoE RMS norm epsilon must be finite and positive"
    );
    let rope_theta = optional_f64(metadata, "qwen3moe.rope.freq_base")?.unwrap_or(10_000.0);
    ensure!(
        rope_theta.is_finite() && rope_theta > 0.0,
        "GGUF Qwen3MoE RoPE frequency base must be finite and positive"
    );

    let feed_forward_length = required_usize(metadata, "qwen3moe.feed_forward_length")?;
    ensure!(
        feed_forward_length > 0,
        "GGUF Qwen3MoE feed-forward length must be positive"
    );
    let expert_count = required_usize(metadata, "qwen3moe.expert_count")?;
    let experts_per_token = required_usize(metadata, "qwen3moe.expert_used_count")?;
    ensure!(
        expert_count > 0 && experts_per_token > 0 && experts_per_token <= expert_count,
        "GGUF Qwen3MoE requires a positive expert count and experts-per-token in 1..={expert_count}"
    );
    // This exactly follows llama.cpp: an explicit expert FFN length wins;
    // older files derive it from n_ff / n_expert_used.
    let expert_intermediate_size = match optional_usize(
        metadata,
        "qwen3moe.expert_feed_forward_length",
    )? {
        Some(explicit) if explicit > 0 => explicit,
        Some(0) | None => {
            ensure!(
                feed_forward_length.is_multiple_of(experts_per_token),
                "GGUF Qwen3MoE feed-forward length {feed_forward_length} must be divisible by experts-per-token {experts_per_token} when expert_feed_forward_length is absent or zero"
            );
            feed_forward_length / experts_per_token
        }
        Some(_) => unreachable!("usize values are non-negative"),
    };
    ensure!(
        expert_intermediate_size > 0,
        "GGUF Qwen3MoE expert feed-forward length must be positive"
    );
    reject_qwen3_moe_biases_and_shared_experts(file)?;

    let rope = RopeSpec {
        theta: rope_theta,
        layout: RopeLayout::HalfSplit,
        rotary_dim: head_dim,
        frequency_factors: None,
    };
    let mut layer = DecoderLayerSpec::standard(rope, head_dim);
    layer
        .attention_mut()
        .expect("standard decoder layer uses attention")
        .qk_norm = true;
    layer.feed_forward = FeedForwardKind::SparseMoe(SparseMoeSpec {
        expert_count,
        experts_per_token,
        expert_intermediate_size,
        router_weighting: RouterWeighting::SelectedSoftmax,
    });
    let spec = DecoderSpec {
        architecture: "qwen3moe".into(),
        runtime: RuntimeSpec {
            vocab_size,
            max_context,
            stop_token_ids: qwen_stop_token_ids(metadata, "Qwen3MoE")?,
        },
        hidden_size,
        // Preserve llama.cpp's model-level n_ff separately from n_ff_exp.
        intermediate_size: feed_forward_length,
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        norm_epsilon,
        layer_specs: vec![layer; num_layers],
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
        tied_embeddings: file.tensor("output.weight").is_none(),
    };
    spec.validate()?;
    Ok(spec)
}

#[allow(clippy::too_many_lines)] // Exact hybrid metadata lowering is one fail-closed compatibility contract.
fn qwen35_spec(file: &GgufFile) -> Result<DecoderSpec> {
    let metadata = file.metadata();
    validate_supported_qwen35_metadata(metadata)?;
    ensure!(
        metadata.get("qwen35.nextn_predict_layers").is_none(),
        "GGUF Qwen3.5 NextN/MTP layers are not supported by the text-trunk runtime"
    );
    ensure!(
        file.tensor("rope_freqs.weight").is_none(),
        "GGUF Qwen3.5 rope frequency factors are not supported yet"
    );
    ensure!(
        !file.tensors().keys().any(|name| name.contains(".nextn.")),
        "GGUF Qwen3.5 contains an unsupported NextN/MTP tensor"
    );

    let vocab_size = qwen35_vocab_size(file)?;
    let hidden_size = required_usize(metadata, "qwen35.embedding_length")?;
    let intermediate_size = required_usize(metadata, "qwen35.feed_forward_length")?;
    let num_layers = required_usize(metadata, "qwen35.block_count")?;
    ensure!(
        num_layers > 0 && num_layers <= MAX_DECODER_LAYERS,
        "GGUF Qwen3.5 layer count {num_layers} must be in 1..={MAX_DECODER_LAYERS}"
    );
    reject_qwen35_out_of_range_layers(file, num_layers)?;

    let num_query_heads = required_usize(metadata, "qwen35.attention.head_count")?;
    let num_kv_heads = required_usize(metadata, "qwen35.attention.head_count_kv")?;
    ensure!(
        num_query_heads > 0 && num_kv_heads > 0 && num_query_heads.is_multiple_of(num_kv_heads),
        "GGUF Qwen3.5 requires positive query/KV head counts with query heads divisible by KV heads"
    );
    let head_dim = required_usize(metadata, "qwen35.attention.key_length")?;
    ensure!(
        head_dim > 0,
        "GGUF Qwen3.5 attention head width must be positive"
    );
    ensure!(
        optional_usize(metadata, "qwen35.attention.value_length")?.unwrap_or(head_dim) == head_dim,
        "different GGUF Qwen3.5 full-attention key/value head widths are not supported"
    );
    let rotary_dim = required_usize(metadata, "qwen35.rope.dimension_count")?;
    ensure!(
        rotary_dim > 0 && rotary_dim <= head_dim && rotary_dim.is_multiple_of(2),
        "GGUF Qwen3.5 RoPE dimension {rotary_dim} must be a positive even value no larger than head width {head_dim}"
    );
    let sections = required_nonnegative_usize_array(metadata, "qwen35.rope.dimension_sections")?;
    ensure!(
        sections.len() == 4,
        "Qwen3.5 MRoPE metadata must contain four dimension sections, found {}",
        sections.len()
    );
    ensure!(
        sections.iter().sum::<usize>() == rotary_dim / 2,
        "Qwen3.5 MRoPE sections must cover exactly half the rotary dimensions"
    );

    let max_context = required_usize(metadata, "qwen35.context_length")?;
    ensure!(
        max_context <= MAX_GGUF_CONTEXT_LENGTH,
        "GGUF Qwen3.5 context length {max_context} exceeds the supported limit {MAX_GGUF_CONTEXT_LENGTH}"
    );
    let norm_epsilon = require_f64(metadata, "qwen35.attention.layer_norm_rms_epsilon")?;
    ensure!(
        norm_epsilon.is_finite() && norm_epsilon > 0.0,
        "GGUF Qwen3.5 RMS norm epsilon must be finite and positive"
    );
    let rope_theta = require_f64(metadata, "qwen35.rope.freq_base")?;
    ensure!(
        rope_theta.is_finite() && rope_theta > 0.0,
        "GGUF Qwen3.5 RoPE frequency base must be finite and positive"
    );

    let convolution_kernel_size = required_usize(metadata, "qwen35.ssm.conv_kernel")?;
    let num_key_heads = required_usize(metadata, "qwen35.ssm.group_count")?;
    let num_value_heads = required_usize(metadata, "qwen35.ssm.time_step_rank")?;
    let key_head_dim = required_usize(metadata, "qwen35.ssm.state_size")?;
    let value_width = required_usize(metadata, "qwen35.ssm.inner_size")?;
    ensure!(
        num_value_heads > 0 && value_width.is_multiple_of(num_value_heads),
        "GGUF Qwen3.5 SSM inner size must be divisible by a positive value-head count"
    );
    let value_head_dim = value_width / num_value_heads;
    ensure!(
        num_key_heads > 0 && num_value_heads.is_multiple_of(num_key_heads),
        "GGUF Qwen3.5 value heads must be divisible by positive key heads"
    );
    let delta = GatedDeltaNetSpec {
        convolution_kernel_size,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        qk_norm_epsilon: norm_epsilon,
    };

    let recurrent_layers = qwen35_recurrent_schedule(metadata, num_layers)?;

    reject_qwen35_unmodeled_biases(file)?;
    let rope = RopeSpec {
        theta: rope_theta,
        layout: RopeLayout::HalfSplit,
        rotary_dim,
        frequency_factors: None,
    };
    let mut attention_layer = DecoderLayerSpec::standard(rope, head_dim);
    let attention = attention_layer
        .attention_mut()
        .expect("standard decoder layer uses attention");
    attention.qk_norm = true;
    attention.output_gate = true;
    let layer_specs = recurrent_layers
        .into_iter()
        .map(|recurrent| {
            let mut layer = attention_layer.clone();
            if recurrent {
                layer.mixer = SequenceMixerSpec::GatedDeltaNet(delta);
            }
            layer
        })
        .collect();

    let spec = DecoderSpec {
        architecture: "qwen35".into(),
        runtime: RuntimeSpec {
            vocab_size,
            max_context,
            stop_token_ids: qwen_stop_token_ids(metadata, "Qwen3.5")?,
        },
        hidden_size,
        intermediate_size,
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        norm_epsilon,
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
        tied_embeddings: file.tensor("output.weight").is_none(),
    };
    spec.validate()?;
    Ok(spec)
}

fn qwen35_recurrent_schedule(metadata: &GgufMetadata, layers: usize) -> Result<Vec<bool>> {
    if let Some(values) = metadata.get("qwen35.attention.recurrent_layers") {
        let values = values.as_array().context(
            "GGUF metadata `qwen35.attention.recurrent_layers` must be an array of BOOL",
        )?;
        ensure!(
            values.len() == layers,
            "GGUF metadata `qwen35.attention.recurrent_layers` has {} entries, expected {layers}",
            values.len()
        );
        return values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value.as_bool().with_context(|| {
                    format!(
                        "GGUF metadata `qwen35.attention.recurrent_layers` element {index} is not BOOL"
                    )
                })
            })
            .collect();
    }
    let interval = optional_usize(metadata, "qwen35.full_attention_interval")?.unwrap_or(4);
    ensure!(
        interval > 0,
        "GGUF Qwen3.5 full-attention interval must be positive"
    );
    Ok((0..layers)
        .map(|layer| (layer + 1) % interval != 0)
        .collect())
}

fn reject_qwen35_unmodeled_biases(file: &GgufFile) -> Result<()> {
    if let Some(name) = file.tensors().keys().find(|name| {
        name.rsplit_once('.')
            .is_some_and(|(_, suffix)| suffix == "bias")
            && !name.ends_with(".ssm_dt.bias")
    }) {
        bail!("GGUF Qwen3.5 does not support unmodeled bias tensor `{name}`");
    }
    ensure!(
        file.tensor("output.bias").is_none(),
        "GGUF output bias is not supported by the shared decoder"
    );
    Ok(())
}

fn reject_qwen35_out_of_range_layers(file: &GgufFile, layers: usize) -> Result<()> {
    if let Some((name, index)) = file.tensors().keys().find_map(|name| {
        let (index, _) = name.strip_prefix("blk.")?.split_once('.')?;
        let index = index.parse::<usize>().ok()?;
        (index >= layers).then_some((name, index))
    }) {
        bail!(
            "GGUF Qwen3.5 tensor `{name}` uses layer {index}, outside the declared {layers}-layer text trunk"
        );
    }
    Ok(())
}

fn validate_qwen2_biases(file: &GgufFile, num_layers: usize) -> Result<()> {
    for suffix in ["attn_q.bias", "attn_k.bias", "attn_v.bias"] {
        ensure!(
            consistent_layer_presence(file, suffix, num_layers)?,
            "Qwen2 requires `{suffix}` in every decoder layer"
        );
    }
    for suffix in [
        "attn_output.bias",
        "ffn_gate.bias",
        "ffn_up.bias",
        "ffn_down.bias",
    ] {
        ensure!(
            !consistent_layer_presence(file, suffix, num_layers)?,
            "Qwen2 does not support `{suffix}`"
        );
    }
    ensure!(
        file.tensor("output.bias").is_none(),
        "GGUF output bias is not supported by the shared decoder"
    );
    Ok(())
}

fn validate_qwen3_biases(file: &GgufFile, num_layers: usize) -> Result<()> {
    for suffix in [
        "attn_q.bias",
        "attn_k.bias",
        "attn_v.bias",
        "attn_output.bias",
        "ffn_gate.bias",
        "ffn_up.bias",
        "ffn_down.bias",
    ] {
        ensure!(
            !consistent_layer_presence(file, suffix, num_layers)?,
            "Qwen3 does not support `{suffix}`"
        );
    }
    ensure!(
        file.tensor("output.bias").is_none(),
        "GGUF output bias is not supported by the shared decoder"
    );
    Ok(())
}

fn reject_qwen3_moe_biases_and_shared_experts(file: &GgufFile) -> Result<()> {
    if let Some(name) = file.tensors().keys().find(|name| {
        name.rsplit_once('.')
            .is_some_and(|(_, suffix)| suffix == "bias")
    }) {
        bail!("Qwen3MoE does not support bias tensor `{name}`");
    }
    ensure!(
        !file.tensors().keys().any(|name| name.contains("shexp")),
        "Qwen3MoE shared experts are not supported by this architecture contract"
    );
    Ok(())
}

fn qwen2_vocab_size(file: &GgufFile) -> Result<usize> {
    let metadata = file.metadata();
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    ensure!(!tokens.is_empty(), "GGUF tokenizer vocabulary is empty");
    let embedding = file
        .tensor("token_embd.weight")
        .context("required GGUF tensor `token_embd.weight` is missing")?;
    ensure!(
        embedding.shape.len() == 2,
        "GGUF tensor `token_embd.weight` has shape {:?}, expected a rank-2 vocabulary matrix",
        embedding.shape
    );
    let embedding_vocab_size = embedding.shape[0];
    ensure!(
        embedding_vocab_size == tokens.len(),
        "GGUF tokenizer vocabulary has {} tokens but `token_embd.weight` has {embedding_vocab_size} rows",
        tokens.len()
    );
    if let Some(declared) = optional_usize(metadata, "qwen2.vocab_size")? {
        ensure!(
            declared == tokens.len(),
            "GGUF tokenizer vocabulary has {} tokens but Qwen2 metadata declares {declared}",
            tokens.len()
        );
    }
    Ok(tokens.len())
}

fn qwen3_vocab_size(file: &GgufFile) -> Result<usize> {
    let metadata = file.metadata();
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    ensure!(!tokens.is_empty(), "GGUF tokenizer vocabulary is empty");
    let embedding = file
        .tensor("token_embd.weight")
        .context("required GGUF tensor `token_embd.weight` is missing")?;
    ensure!(
        embedding.shape.len() == 2,
        "GGUF tensor `token_embd.weight` has shape {:?}, expected a rank-2 vocabulary matrix",
        embedding.shape
    );
    let embedding_vocab_size = embedding.shape[0];
    ensure!(
        embedding_vocab_size == tokens.len(),
        "GGUF tokenizer vocabulary has {} tokens but `token_embd.weight` has {embedding_vocab_size} rows",
        tokens.len()
    );
    if let Some(declared) = optional_usize(metadata, "qwen3.vocab_size")? {
        ensure!(
            declared == tokens.len(),
            "GGUF tokenizer vocabulary has {} tokens but Qwen3 metadata declares {declared}",
            tokens.len()
        );
    }
    Ok(tokens.len())
}

fn smollm3_vocab_size(file: &GgufFile) -> Result<usize> {
    let metadata = file.metadata();
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    ensure!(!tokens.is_empty(), "GGUF tokenizer vocabulary is empty");
    let embedding = file
        .tensor("token_embd.weight")
        .context("required GGUF tensor `token_embd.weight` is missing")?;
    ensure!(
        embedding.shape.len() == 2,
        "GGUF tensor `token_embd.weight` has shape {:?}, expected a rank-2 vocabulary matrix",
        embedding.shape
    );
    ensure!(
        embedding.shape[0] == tokens.len(),
        "GGUF tokenizer vocabulary has {} tokens but `token_embd.weight` has {} rows",
        tokens.len(),
        embedding.shape[0]
    );
    if let Some(declared) = optional_usize(metadata, "smollm3.vocab_size")? {
        ensure!(
            declared == tokens.len(),
            "GGUF tokenizer vocabulary has {} tokens but SmolLM3 metadata declares {declared}",
            tokens.len()
        );
    }
    Ok(tokens.len())
}

fn qwen3_moe_vocab_size(file: &GgufFile) -> Result<usize> {
    let metadata = file.metadata();
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    ensure!(!tokens.is_empty(), "GGUF tokenizer vocabulary is empty");
    let embedding = file
        .tensor("token_embd.weight")
        .context("required GGUF tensor `token_embd.weight` is missing")?;
    ensure!(
        embedding.shape.len() == 2,
        "GGUF tensor `token_embd.weight` has shape {:?}, expected a rank-2 vocabulary matrix",
        embedding.shape
    );
    let embedding_vocab_size = embedding.shape[0];
    ensure!(
        embedding_vocab_size == tokens.len(),
        "GGUF tokenizer vocabulary has {} tokens but `token_embd.weight` has {embedding_vocab_size} rows",
        tokens.len()
    );
    if let Some(declared) = optional_usize(metadata, "qwen3moe.vocab_size")? {
        ensure!(
            declared == tokens.len(),
            "GGUF tokenizer vocabulary has {} tokens but Qwen3MoE metadata declares {declared}",
            tokens.len()
        );
    }
    Ok(tokens.len())
}

fn qwen35_vocab_size(file: &GgufFile) -> Result<usize> {
    let metadata = file.metadata();
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    ensure!(!tokens.is_empty(), "GGUF tokenizer vocabulary is empty");
    let embedding = file
        .tensor("token_embd.weight")
        .context("required GGUF tensor `token_embd.weight` is missing")?;
    ensure!(
        embedding.shape.len() == 2,
        "GGUF tensor `token_embd.weight` has shape {:?}, expected a rank-2 vocabulary matrix",
        embedding.shape
    );
    ensure!(
        embedding.shape[0] == tokens.len(),
        "GGUF tokenizer vocabulary has {} tokens but `token_embd.weight` has {} rows",
        tokens.len(),
        embedding.shape[0]
    );
    if let Some(declared) = optional_usize(metadata, "qwen35.vocab_size")? {
        ensure!(
            declared == tokens.len(),
            "GGUF tokenizer vocabulary has {} tokens but Qwen3.5 metadata declares {declared}",
            tokens.len()
        );
    }
    Ok(tokens.len())
}

fn qwen2_stop_token_ids(metadata: &GgufMetadata) -> Result<Vec<u32>> {
    qwen_stop_token_ids(metadata, "Qwen2")
}

fn qwen_stop_token_ids(metadata: &GgufMetadata, family: &str) -> Result<Vec<u32>> {
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    let mut stop_token_ids = optional_u32(metadata, "tokenizer.ggml.eos_token_id")?
        .into_iter()
        .collect::<Vec<_>>();
    for stop_token in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(index) = tokens
            .iter()
            .position(|token| token.as_str() == Some(stop_token))
        {
            let token_id = u32::try_from(index)
                .with_context(|| format!("{family} stop token ID exceeds u32"))?;
            if !stop_token_ids.contains(&token_id) {
                stop_token_ids.push(token_id);
            }
        }
    }
    Ok(stop_token_ids)
}

fn llama_stop_token_ids(metadata: &GgufMetadata) -> Result<Vec<u32>> {
    let tokens = require_array(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?;
    let mut stop_token_ids = Vec::new();
    for (token_name, metadata_key) in [
        ("<|end_of_text|>", None),
        ("<|eom_id|>", Some("tokenizer.ggml.eom_token_id")),
        ("<|eot_id|>", Some("tokenizer.ggml.eot_token_id")),
    ] {
        if let Some(token_id) = metadata_key
            .map(|key| optional_u32(metadata, key))
            .transpose()?
            .flatten()
            .or_else(|| {
                tokens
                    .iter()
                    .position(|token| token.as_str() == Some(token_name))
                    .and_then(|index| u32::try_from(index).ok())
            })
            && !stop_token_ids.contains(&token_id)
        {
            stop_token_ids.push(token_id);
        }
    }
    if let Some(eos_token_id) = optional_u32(metadata, "tokenizer.ggml.eos_token_id")?
        && !stop_token_ids.contains(&eos_token_id)
    {
        stop_token_ids.push(eos_token_id);
    }
    Ok(stop_token_ids)
}

fn validate_supported_gemma3_metadata(metadata: &GgufMetadata) -> Result<()> {
    for (key, _) in metadata.iter() {
        if key.starts_with("gemma3.") {
            ensure!(
                SUPPORTED_GEMMA3_METADATA_KEYS.contains(&key),
                "GGUF Gemma 3 metadata `{key}` is not supported yet"
            );
        }
    }
    Ok(())
}

fn validate_supported_gemma_metadata(metadata: &GgufMetadata) -> Result<()> {
    for (key, _) in metadata.iter() {
        if key.starts_with("gemma.") {
            ensure!(
                SUPPORTED_GEMMA_METADATA_KEYS.contains(&key),
                "GGUF Gemma metadata `{key}` is not supported yet"
            );
        }
    }
    ensure!(
        optional_bool(metadata, "gemma.attention.causal")?.unwrap_or(true),
        "non-causal GGUF Gemma attention is not supported"
    );
    Ok(())
}

fn validate_supported_gemma2_metadata(metadata: &GgufMetadata) -> Result<()> {
    for (key, _) in metadata.iter() {
        if key.starts_with("gemma2.") {
            ensure!(
                SUPPORTED_GEMMA2_METADATA_KEYS.contains(&key),
                "GGUF Gemma 2 metadata `{key}` is not supported yet"
            );
        }
    }
    ensure!(
        optional_bool(metadata, "gemma2.attention.causal")?.unwrap_or(true),
        "non-causal GGUF Gemma 2 attention is not supported"
    );
    Ok(())
}

fn validate_supported_falcon_h1_metadata(metadata: &GgufMetadata) -> Result<()> {
    for (key, _) in metadata.iter() {
        if key.starts_with("falcon-h1.") {
            ensure!(
                SUPPORTED_FALCON_H1_METADATA_KEYS.contains(&key),
                "GGUF Falcon-H1 metadata `{key}` is not supported yet"
            );
        }
    }
    Ok(())
}

fn validate_supported_granite_moe_metadata(metadata: &GgufMetadata) -> Result<()> {
    for (key, _) in metadata.iter() {
        if key.starts_with("granitemoe.") {
            ensure!(
                SUPPORTED_GRANITE_MOE_METADATA_KEYS.contains(&key),
                "GGUF Granite MoE metadata `{key}` is not supported yet"
            );
        }
    }
    ensure!(
        optional_bool(metadata, "granitemoe.attention.causal")?.unwrap_or(true),
        "non-causal GGUF Granite MoE attention is not supported"
    );
    Ok(())
}

fn validate_supported_lfm2_metadata(metadata: &GgufMetadata) -> Result<()> {
    for (key, _) in metadata.iter() {
        if key.starts_with("lfm2.") {
            ensure!(
                SUPPORTED_LFM2_METADATA_KEYS.contains(&key),
                "GGUF LFM2 metadata `{key}` is not supported yet"
            );
        }
    }
    ensure!(
        optional_bool(metadata, "lfm2.attention.causal")?.unwrap_or(true),
        "non-causal GGUF LFM2 attention is not supported"
    );
    Ok(())
}

fn validate_supported_llama_metadata(metadata: &GgufMetadata) -> Result<()> {
    for (key, _) in metadata.iter() {
        if key.starts_with("llama.") {
            ensure!(
                SUPPORTED_LLAMA_METADATA_KEYS.contains(&key),
                "GGUF Llama metadata `{key}` is not supported yet"
            );
        }
    }
    ensure!(
        optional_bool(metadata, "llama.attention.causal")?.unwrap_or(true),
        "non-causal GGUF Llama attention is not supported"
    );
    for key in [
        "llama.attention.sliding_window_pattern",
        "llama.attention.scale",
    ] {
        ensure!(
            metadata.get(key).is_none(),
            "GGUF Llama attention metadata `{key}` is not supported yet"
        );
    }
    Ok(())
}

fn validate_supported_qwen2_metadata(metadata: &GgufMetadata) -> Result<()> {
    for (key, _) in metadata.iter() {
        if key.starts_with("qwen2.") {
            ensure!(
                SUPPORTED_QWEN2_METADATA_KEYS.contains(&key),
                "GGUF Qwen2 metadata `{key}` is not supported yet"
            );
        }
    }
    ensure!(
        optional_bool(metadata, "qwen2.attention.causal")?.unwrap_or(true),
        "non-causal GGUF Qwen2 attention is not supported"
    );
    let sliding_window = optional_usize(metadata, "qwen2.attention.sliding_window")?.unwrap_or(0);
    ensure!(
        sliding_window == 0,
        "sliding-window GGUF Qwen2 attention is not supported"
    );
    Ok(())
}

fn validate_supported_qwen3_metadata(metadata: &GgufMetadata) -> Result<()> {
    for (key, _) in metadata.iter() {
        if key.starts_with("qwen3.") {
            ensure!(
                SUPPORTED_QWEN3_METADATA_KEYS.contains(&key),
                "GGUF Qwen3 metadata `{key}` is not supported yet"
            );
        }
    }
    ensure!(
        optional_bool(metadata, "qwen3.attention.causal")?.unwrap_or(true),
        "non-causal GGUF Qwen3 attention is not supported"
    );
    let sliding_window = optional_usize(metadata, "qwen3.attention.sliding_window")?.unwrap_or(0);
    ensure!(
        sliding_window == 0,
        "sliding-window GGUF Qwen3 attention is not supported"
    );
    Ok(())
}

fn validate_supported_smollm3_metadata(metadata: &GgufMetadata) -> Result<()> {
    for (key, _) in metadata.iter() {
        if key.starts_with("smollm3.") {
            ensure!(
                SUPPORTED_SMOLLM3_METADATA_KEYS.contains(&key),
                "GGUF SmolLM3 metadata `{key}` is not supported yet"
            );
        }
    }
    ensure!(
        optional_bool(metadata, "smollm3.attention.causal")?.unwrap_or(true),
        "non-causal GGUF SmolLM3 attention is not supported"
    );
    let sliding_window = optional_usize(metadata, "smollm3.attention.sliding_window")?.unwrap_or(0);
    ensure!(
        sliding_window == 0,
        "sliding-window GGUF SmolLM3 attention is not supported"
    );
    Ok(())
}

fn validate_supported_qwen3_moe_metadata(metadata: &GgufMetadata) -> Result<()> {
    for (key, _) in metadata.iter() {
        if key.starts_with("qwen3moe.") {
            ensure!(
                SUPPORTED_QWEN3_MOE_METADATA_KEYS.contains(&key),
                "GGUF Qwen3MoE metadata `{key}` is not supported yet"
            );
        }
    }
    ensure!(
        optional_bool(metadata, "qwen3moe.attention.causal")?.unwrap_or(true),
        "non-causal GGUF Qwen3MoE attention is not supported"
    );
    let sliding_window =
        optional_usize(metadata, "qwen3moe.attention.sliding_window")?.unwrap_or(0);
    ensure!(
        sliding_window == 0,
        "sliding-window GGUF Qwen3MoE attention is not supported"
    );
    ensure!(
        optional_usize(metadata, "qwen3moe.expert_shared_feed_forward_length")?.unwrap_or(0) == 0,
        "Qwen3MoE shared experts are not supported by the current llama.cpp architecture contract"
    );
    Ok(())
}

fn validate_supported_qwen35_metadata(metadata: &GgufMetadata) -> Result<()> {
    for (key, _) in metadata.iter() {
        if key.starts_with("qwen35.") {
            ensure!(
                SUPPORTED_QWEN35_METADATA_KEYS.contains(&key),
                "GGUF Qwen3.5 metadata `{key}` is not supported yet"
            );
        }
    }
    ensure!(
        optional_bool(metadata, "qwen35.attention.causal")?.unwrap_or(true),
        "non-causal GGUF Qwen3.5 attention is not supported"
    );
    Ok(())
}

fn embedded_tokenizer(
    metadata: &GgufMetadata,
    spec: &DecoderSpec,
) -> Result<(Tokenizer, Option<String>)> {
    validate_tokenizer_build_budget(metadata)?;
    let tokens = require_string_array(metadata, "tokenizer.ggml.tokens")?;
    ensure!(
        tokens.len() == spec.runtime.vocab_size,
        "GGUF tokenizer vocabulary has {} tokens but model metadata declares {}",
        tokens.len(),
        spec.runtime.vocab_size
    );
    let token_types = require_i32_array(metadata, "tokenizer.ggml.token_type")?;
    for key in [
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.eom_token_id",
        "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
    ] {
        if let Some(token_id) = optional_u32(metadata, key)? {
            ensure!(
                usize::try_from(token_id)
                    .ok()
                    .is_some_and(|id| id < tokens.len()),
                "GGUF metadata `{key}` token ID {token_id} is outside the vocabulary"
            );
        }
    }
    let model = require_string(metadata, "tokenizer.ggml.model")?;
    let pre_tokenizer = require_string(metadata, "tokenizer.ggml.pre")?;
    // SentencePiece/Llama tokenizers prepend the dummy whitespace marker by
    // default. Many older GGUFs, including TinyLlama, omit the explicit key;
    // GPT-2/BPE profiles instead require the existing false default.
    let add_space_prefix =
        optional_bool(metadata, "tokenizer.ggml.add_space_prefix")?.unwrap_or(model == "llama");
    let bos_policy = optional_bool(metadata, "tokenizer.ggml.add_bos_token")?;
    let eos_policy = optional_bool(metadata, "tokenizer.ggml.add_eos_token")?;
    let bos_token_id = optional_u32(metadata, "tokenizer.ggml.bos_token_id")?;
    let eos_token_id = optional_u32(metadata, "tokenizer.ggml.eos_token_id")?;
    let tokenizer = match model {
        "gpt2" => EmbeddedBpeTokenizer {
            model: model.to_owned(),
            pre_tokenizer: pre_tokenizer.to_owned(),
            tokens,
            token_types,
            merges: require_string_array(metadata, "tokenizer.ggml.merges")?,
            add_space_prefix,
            add_bos_token: bos_policy,
            add_eos_token: eos_policy,
            bos_token_id,
            eos_token_id,
        }
        .build()?,
        "llama" => EmbeddedSentencePieceTokenizer {
            model: model.to_owned(),
            pre_tokenizer: pre_tokenizer.to_owned(),
            tokens,
            scores: require_f32_array(metadata, "tokenizer.ggml.scores")?,
            token_types,
            add_space_prefix,
            add_bos_token: bos_policy,
            add_eos_token: eos_policy,
            bos_token_id,
            eos_token_id,
            unknown_token_id: optional_u32(metadata, "tokenizer.ggml.unknown_token_id")?,
        }
        .build()?,
        unsupported => bail!("unsupported embedded tokenizer model `{unsupported}`"),
    };
    let chat_template = optional_string(metadata, "tokenizer.chat_template")?.map(str::to_owned);
    Ok((tokenizer, chat_template))
}

fn validate_tokenizer_build_budget(metadata: &GgufMetadata) -> Result<()> {
    let mut bytes = 0_u64;
    let model = require_string(metadata, "tokenizer.ggml.model")?;
    let string_arrays: &[&str] = if model == "gpt2" {
        &["tokenizer.ggml.tokens", "tokenizer.ggml.merges"]
    } else {
        &["tokenizer.ggml.tokens"]
    };
    for key in string_arrays {
        let value = metadata
            .get(key)
            .with_context(|| format!("required GGUF metadata `{key}` is missing"))?;
        ensure!(
            value.array_element_type() == Some(GgufValueType::String),
            "GGUF metadata `{key}` must be ARRAY<STRING>"
        );
        let values = value.as_array().context("metadata value is not an array")?;
        for (index, value) in values.iter().enumerate() {
            let string = value
                .as_str()
                .with_context(|| format!("GGUF metadata `{key}` element {index} is not STRING"))?;
            bytes = bytes
                .checked_add(u64::try_from(string.len())?)
                .and_then(|value| value.checked_add(32))
                .context("GGUF tokenizer build budget overflow")?;
        }
    }
    let token_types = metadata
        .get("tokenizer.ggml.token_type")
        .context("required GGUF metadata `tokenizer.ggml.token_type` is missing")?
        .as_array()
        .context("GGUF metadata `tokenizer.ggml.token_type` must be an array")?;
    bytes = bytes
        .checked_add(
            u64::try_from(token_types.len())?
                .checked_mul(u64::try_from(std::mem::size_of::<i32>())?)
                .context("GGUF tokenizer type budget overflow")?,
        )
        .context("GGUF tokenizer build budget overflow")?;
    if model == "llama" {
        let scores = require_array(metadata, "tokenizer.ggml.scores", GgufValueType::F32)?;
        bytes = bytes
            .checked_add(
                u64::try_from(scores.len())?
                    .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
                    .context("GGUF tokenizer score budget overflow")?,
            )
            .context("GGUF tokenizer build budget overflow")?;
    }
    ensure!(
        bytes <= MAX_TOKENIZER_BUILD_INPUT_BYTES,
        "GGUF tokenizer requires an estimated {bytes} bytes of build input, exceeding the limit {MAX_TOKENIZER_BUILD_INPUT_BYTES}"
    );
    Ok(())
}

fn validate_decoder_tensors(file: &GgufFile, spec: &DecoderSpec) -> Result<()> {
    expect_shape(
        file,
        "token_embd.weight",
        &[spec.runtime.vocab_size, spec.hidden_size],
    )?;
    let output_norm = if spec.architecture == "lfm2" {
        "token_embd_norm.weight"
    } else {
        "output_norm.weight"
    };
    expect_shape(file, output_norm, &[spec.hidden_size])?;
    if !spec.tied_embeddings {
        expect_shape(
            file,
            "output.weight",
            &[spec.runtime.vocab_size, spec.hidden_size],
        )?;
    }
    let query_width = spec
        .num_query_heads
        .checked_mul(spec.head_dim)
        .context("query projection width overflow")?;
    let kv_width = spec
        .num_kv_heads
        .checked_mul(spec.head_dim)
        .context("KV projection width overflow")?;
    for layer in 0..spec.num_layers {
        validate_decoder_layer_tensors(file, spec, layer, query_width, kv_width)?;
    }
    Ok(())
}

#[allow(clippy::too_many_lines)] // Mixer, FFN and bias topology are validated in one layer-local audit.
fn validate_decoder_layer_tensors(
    file: &GgufFile,
    spec: &DecoderSpec,
    layer: usize,
    query_width: usize,
    kv_width: usize,
) -> Result<()> {
    let prefix = format!("blk.{layer}");
    let layer_spec = &spec.layer_specs[layer];
    let ffn_norm_suffix = match spec.architecture.as_str() {
        "qwen35" => "post_attention_norm.weight",
        "falcon-h1" => "ffn_norm",
        _ => "ffn_norm.weight",
    };
    for suffix in ["attn_norm.weight", ffn_norm_suffix] {
        expect_shape(file, &format!("{prefix}.{suffix}"), &[spec.hidden_size])?;
    }
    if layer_spec.norm_layout == NormLayout::Sandwich {
        for suffix in ["post_attention_norm.weight", "post_ffw_norm.weight"] {
            expect_shape(file, &format!("{prefix}.{suffix}"), &[spec.hidden_size])?;
        }
    }
    match &layer_spec.mixer {
        SequenceMixerSpec::Attention(attention) => {
            if attention.qk_norm {
                for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                    expect_shape(file, &format!("{prefix}.{suffix}"), &[spec.head_dim])?;
                }
            }
            let projected_query_width = query_width
                .checked_mul(if attention.output_gate { 2 } else { 1 })
                .context("query-and-gate projection width overflow")?;
            for (suffix, expected) in [
                ("attn_q.weight", [projected_query_width, spec.hidden_size]),
                ("attn_k.weight", [kv_width, spec.hidden_size]),
                ("attn_v.weight", [kv_width, spec.hidden_size]),
                ("attn_output.weight", [spec.hidden_size, query_width]),
            ] {
                expect_shape(file, &format!("{prefix}.{suffix}"), &expected)?;
            }
            for suffix in [
                "shortconv.conv.weight",
                "shortconv.in_proj.weight",
                "shortconv.out_proj.weight",
                "attn_qkv.weight",
                "attn_gate.weight",
                "ssm_conv1d.weight",
                "ssm_dt.bias",
                "ssm_a",
                "ssm_beta.weight",
                "ssm_alpha.weight",
                "ssm_norm.weight",
                "ssm_out.weight",
            ] {
                ensure!(
                    file.tensor(&format!("{prefix}.{suffix}")).is_none(),
                    "attention layer {layer} contains short-convolution tensor `{suffix}`"
                );
            }
        }
        SequenceMixerSpec::ShortConvolution(shortconv) => {
            let projected = spec
                .hidden_size
                .checked_mul(3)
                .context("short-convolution projection width overflow")?;
            for (suffix, expected) in [
                (
                    "shortconv.conv.weight",
                    vec![spec.hidden_size, shortconv.kernel_size],
                ),
                (
                    "shortconv.in_proj.weight",
                    vec![projected, spec.hidden_size],
                ),
                (
                    "shortconv.out_proj.weight",
                    vec![spec.hidden_size, spec.hidden_size],
                ),
            ] {
                expect_shape(file, &format!("{prefix}.{suffix}"), &expected)?;
            }
            for suffix in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "attn_qkv.weight",
                "attn_gate.weight",
                "ssm_conv1d.weight",
                "ssm_dt.bias",
                "ssm_a",
                "ssm_beta.weight",
                "ssm_alpha.weight",
                "ssm_norm.weight",
                "ssm_out.weight",
            ] {
                ensure!(
                    file.tensor(&format!("{prefix}.{suffix}")).is_none(),
                    "short-convolution layer {layer} contains attention tensor `{suffix}`"
                );
            }
        }
        SequenceMixerSpec::GatedDeltaNet(delta) => {
            let key_width = delta
                .num_key_heads
                .checked_mul(delta.key_head_dim)
                .context("Gated DeltaNet key width overflow")?;
            let value_width = delta
                .num_value_heads
                .checked_mul(delta.value_head_dim)
                .context("Gated DeltaNet value width overflow")?;
            let convolution_width = key_width
                .checked_mul(2)
                .and_then(|width| width.checked_add(value_width))
                .context("Gated DeltaNet convolution width overflow")?;
            for (suffix, expected) in [
                ("attn_qkv.weight", vec![convolution_width, spec.hidden_size]),
                ("attn_gate.weight", vec![value_width, spec.hidden_size]),
                (
                    "ssm_conv1d.weight",
                    vec![convolution_width, delta.convolution_kernel_size],
                ),
                ("ssm_dt.bias", vec![delta.num_value_heads]),
                ("ssm_a", vec![delta.num_value_heads]),
                (
                    "ssm_beta.weight",
                    vec![delta.num_value_heads, spec.hidden_size],
                ),
                (
                    "ssm_alpha.weight",
                    vec![delta.num_value_heads, spec.hidden_size],
                ),
                ("ssm_norm.weight", vec![delta.value_head_dim]),
                ("ssm_out.weight", vec![spec.hidden_size, value_width]),
            ] {
                expect_shape(file, &format!("{prefix}.{suffix}"), &expected)?;
            }
            for suffix in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "shortconv.conv.weight",
                "shortconv.in_proj.weight",
                "shortconv.out_proj.weight",
            ] {
                ensure!(
                    file.tensor(&format!("{prefix}.{suffix}")).is_none(),
                    "Gated DeltaNet layer {layer} contains incompatible tensor `{suffix}`"
                );
            }
        }
        SequenceMixerSpec::ParallelAttentionMamba2(parallel) => {
            let attention = &parallel.attention;
            if attention.qk_norm {
                for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                    expect_shape(file, &format!("{prefix}.{suffix}"), &[spec.head_dim])?;
                }
            }
            let projected_query_width = query_width
                .checked_mul(if attention.output_gate { 2 } else { 1 })
                .context("query-and-gate projection width overflow")?;
            for (suffix, expected) in [
                ("attn_q.weight", [projected_query_width, spec.hidden_size]),
                ("attn_k.weight", [kv_width, spec.hidden_size]),
                ("attn_v.weight", [kv_width, spec.hidden_size]),
                ("attn_output.weight", [spec.hidden_size, query_width]),
            ] {
                expect_shape(file, &format!("{prefix}.{suffix}"), &expected)?;
            }

            let mamba = parallel.mamba2;
            let group_state_width = mamba
                .num_groups
                .checked_mul(mamba.state_size)
                .context("Mamba2 group-state width overflow")?;
            let convolution_width = mamba
                .inner_size
                .checked_add(
                    group_state_width
                        .checked_mul(2)
                        .context("Mamba2 B/C width overflow")?,
                )
                .context("Mamba2 convolution width overflow")?;
            let projection_width = mamba
                .inner_size
                .checked_add(convolution_width)
                .and_then(|width| width.checked_add(mamba.num_heads))
                .context("Mamba2 projection width overflow")?;
            for (suffix, expected) in [
                ("ssm_in.weight", vec![projection_width, spec.hidden_size]),
                (
                    "ssm_conv1d.weight",
                    vec![convolution_width, mamba.convolution_kernel_size],
                ),
                ("ssm_dt.bias", vec![mamba.num_heads]),
                ("ssm_out.weight", vec![spec.hidden_size, mamba.inner_size]),
            ] {
                expect_shape(file, &format!("{prefix}.{suffix}"), &expected)?;
            }
            for suffix in ["ssm_a", "ssm_d"] {
                expect_one_of_shapes(
                    file,
                    &format!("{prefix}.{suffix}"),
                    &[vec![mamba.num_heads], vec![mamba.num_heads, 1]],
                )?;
            }
            if mamba.convolution_bias {
                expect_shape(
                    file,
                    &format!("{prefix}.ssm_conv1d.bias"),
                    &[convolution_width],
                )?;
            }
            if mamba.input_projection_bias {
                expect_shape(file, &format!("{prefix}.ssm_in.bias"), &[projection_width])?;
            }
            if mamba.output_projection_bias {
                expect_shape(file, &format!("{prefix}.ssm_out.bias"), &[spec.hidden_size])?;
            }
            if mamba.gated_rms_norm {
                expect_shape(
                    file,
                    &format!("{prefix}.ssm_norm.weight"),
                    &[mamba.num_groups, mamba.inner_size / mamba.num_groups],
                )?;
            }
            for suffix in [
                "shortconv.conv.weight",
                "shortconv.in_proj.weight",
                "shortconv.out_proj.weight",
                "attn_qkv.weight",
                "attn_gate.weight",
                "ssm_beta.weight",
                "ssm_alpha.weight",
            ] {
                ensure!(
                    file.tensor(&format!("{prefix}.{suffix}")).is_none(),
                    "parallel attention/Mamba2 layer {layer} contains incompatible tensor `{suffix}`"
                );
            }
        }
    }
    match layer_spec.feed_forward {
        FeedForwardKind::Dense => {
            for (suffix, expected) in [
                (
                    "ffn_gate.weight",
                    [spec.intermediate_size, spec.hidden_size],
                ),
                ("ffn_up.weight", [spec.intermediate_size, spec.hidden_size]),
                (
                    "ffn_down.weight",
                    [spec.hidden_size, spec.intermediate_size],
                ),
            ] {
                expect_shape(file, &format!("{prefix}.{suffix}"), &expected)?;
            }
            for suffix in [
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ] {
                ensure!(
                    file.tensor(&format!("{prefix}.{suffix}")).is_none(),
                    "dense layer {layer} contains incompatible sparse-MoE tensor `{suffix}`"
                );
            }
        }
        FeedForwardKind::SparseMoe(moe) => {
            expect_shape(
                file,
                &format!("{prefix}.ffn_gate_inp.weight"),
                &[moe.expert_count, spec.hidden_size],
            )?;
            for suffix in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
                expect_shape(
                    file,
                    &format!("{prefix}.{suffix}"),
                    &[
                        moe.expert_count,
                        moe.expert_intermediate_size,
                        spec.hidden_size,
                    ],
                )?;
            }
            expect_shape(
                file,
                &format!("{prefix}.ffn_down_exps.weight"),
                &[
                    moe.expert_count,
                    spec.hidden_size,
                    moe.expert_intermediate_size,
                ],
            )?;
            for suffix in [
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
                "ffn_gate.bias",
                "ffn_up.bias",
                "ffn_down.bias",
            ] {
                ensure!(
                    file.tensor(&format!("{prefix}.{suffix}")).is_none(),
                    "sparse-MoE layer {layer} contains unsupported dense tensor `{suffix}`"
                );
            }
        }
    }
    if let Some(attention) = layer_spec.mixer.attention() {
        let projected_query_width = if attention.output_gate {
            query_width
                .checked_mul(2)
                .context("query-and-gate bias width overflow")?
        } else {
            query_width
        };
        for (suffix, expected, present) in [
            (
                "attn_q.bias",
                projected_query_width,
                spec.projection_bias.query,
            ),
            ("attn_k.bias", kv_width, spec.projection_bias.key),
            ("attn_v.bias", kv_width, spec.projection_bias.value),
            (
                "attn_output.bias",
                spec.hidden_size,
                spec.projection_bias.output,
            ),
        ] {
            if present {
                expect_shape(file, &format!("{prefix}.{suffix}"), &[expected])?;
            }
        }
    }
    if layer_spec.feed_forward == FeedForwardKind::Dense {
        for (suffix, expected, present) in [
            ("ffn_gate.bias", spec.intermediate_size, spec.mlp_bias),
            ("ffn_up.bias", spec.intermediate_size, spec.mlp_bias),
            ("ffn_down.bias", spec.hidden_size, spec.mlp_bias),
        ] {
            if present {
                expect_shape(file, &format!("{prefix}.{suffix}"), &[expected])?;
            }
        }
    }
    Ok(())
}

fn expect_shape(file: &GgufFile, name: &str, expected: &[usize]) -> Result<()> {
    let tensor = file
        .tensor(name)
        .with_context(|| format!("required GGUF tensor `{name}` is missing"))?;
    ensure!(
        tensor.shape == expected,
        "GGUF tensor `{name}` has shape {:?}, expected {expected:?}",
        tensor.shape
    );
    Ok(())
}

fn expect_one_of_shapes(file: &GgufFile, name: &str, expected: &[Vec<usize>]) -> Result<()> {
    let tensor = file
        .tensor(name)
        .with_context(|| format!("required GGUF tensor `{name}` is missing"))?;
    ensure!(
        expected.contains(&tensor.shape),
        "GGUF tensor `{name}` has shape {:?}, expected one of {expected:?}",
        tensor.shape
    );
    Ok(())
}

fn consistent_layer_presence(file: &GgufFile, suffix: &str, layers: usize) -> Result<bool> {
    if layers == 0 {
        return Ok(false);
    }
    let present = file.tensor(&format!("blk.0.{suffix}")).is_some();
    for layer in 1..layers {
        ensure!(
            file.tensor(&format!("blk.{layer}.{suffix}")).is_some() == present,
            "GGUF tensor `{suffix}` is present in only some decoder layers"
        );
    }
    Ok(present)
}

fn require_string<'a>(metadata: &'a GgufMetadata, key: &str) -> Result<&'a str> {
    let value = metadata
        .get(key)
        .with_context(|| format!("required GGUF metadata `{key}` is missing"))?;
    value.as_str().with_context(|| {
        format!(
            "GGUF metadata `{key}` must be STRING, got {:?}",
            value.value_type()
        )
    })
}

fn optional_string<'a>(metadata: &'a GgufMetadata, key: &str) -> Result<Option<&'a str>> {
    metadata
        .get(key)
        .map(|value| {
            value.as_str().with_context(|| {
                format!(
                    "GGUF metadata `{key}` must be STRING, got {:?}",
                    value.value_type()
                )
            })
        })
        .transpose()
}

fn required_usize(metadata: &GgufMetadata, key: &str) -> Result<usize> {
    optional_usize(metadata, key)?
        .with_context(|| format!("required GGUF metadata `{key}` is missing"))
}

fn optional_usize(metadata: &GgufMetadata, key: &str) -> Result<Option<usize>> {
    optional_u64(metadata, key)?
        .map(|value| usize::try_from(value).with_context(|| format!("`{key}` does not fit usize")))
        .transpose()
}

fn required_layer_usizes(metadata: &GgufMetadata, key: &str, layers: usize) -> Result<Vec<usize>> {
    let value = metadata
        .get(key)
        .with_context(|| format!("required GGUF metadata `{key}` is missing"))?;
    if let Some(values) = value.as_array() {
        ensure!(
            values.len() == layers,
            "GGUF metadata `{key}` has {} layer values, expected {layers}",
            values.len()
        );
        return values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                nonnegative_usize(value).with_context(|| {
                    format!(
                        "GGUF metadata `{key}` element {index} must be a non-negative integer that fits usize"
                    )
                })
            })
            .collect();
    }
    let scalar = nonnegative_usize(value).with_context(|| {
        format!(
            "GGUF metadata `{key}` must be a non-negative integer or an array of non-negative integers"
        )
    })?;
    Ok(vec![scalar; layers])
}

fn required_nonnegative_usize_array(metadata: &GgufMetadata, key: &str) -> Result<Vec<usize>> {
    let values = metadata
        .get(key)
        .and_then(GgufValue::as_array)
        .with_context(|| format!("required GGUF metadata `{key}` must be an integer array"))?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            nonnegative_usize(value).with_context(|| {
                format!(
                    "GGUF metadata `{key}` element {index} must be a non-negative integer that fits usize"
                )
            })
        })
        .collect()
}

fn nonnegative_usize(value: &GgufValue) -> Option<usize> {
    match value {
        GgufValue::U8(value) => Some(usize::from(*value)),
        GgufValue::U16(value) => Some(usize::from(*value)),
        GgufValue::U32(value) => usize::try_from(*value).ok(),
        GgufValue::U64(value) => usize::try_from(*value).ok(),
        GgufValue::I8(value) => usize::try_from(*value).ok(),
        GgufValue::I16(value) => usize::try_from(*value).ok(),
        GgufValue::I32(value) => usize::try_from(*value).ok(),
        GgufValue::I64(value) => usize::try_from(*value).ok(),
        _ => None,
    }
}

fn optional_u32(metadata: &GgufMetadata, key: &str) -> Result<Option<u32>> {
    metadata
        .get(key)
        .map(|value| {
            value.as_u32().with_context(|| {
                format!(
                    "GGUF metadata `{key}` must be UINT32, got {:?}",
                    value.value_type()
                )
            })
        })
        .transpose()
}

fn optional_u64(metadata: &GgufMetadata, key: &str) -> Result<Option<u64>> {
    metadata
        .get(key)
        .map(|value| {
            value.as_u64().with_context(|| {
                format!(
                    "GGUF metadata `{key}` must be an unsigned integer, got {:?}",
                    value.value_type()
                )
            })
        })
        .transpose()
}

fn require_f64(metadata: &GgufMetadata, key: &str) -> Result<f64> {
    optional_f64(metadata, key)?
        .with_context(|| format!("required GGUF metadata `{key}` is missing"))
}

fn optional_f64(metadata: &GgufMetadata, key: &str) -> Result<Option<f64>> {
    metadata
        .get(key)
        .map(|value| {
            value.as_f64().with_context(|| {
                format!(
                    "GGUF metadata `{key}` must be FLOAT32 or FLOAT64, got {:?}",
                    value.value_type()
                )
            })
        })
        .transpose()
}

fn optional_bool(metadata: &GgufMetadata, key: &str) -> Result<Option<bool>> {
    metadata
        .get(key)
        .map(|value| {
            value.as_bool().with_context(|| {
                format!(
                    "GGUF metadata `{key}` must be BOOL, got {:?}",
                    value.value_type()
                )
            })
        })
        .transpose()
}

fn require_string_array(metadata: &GgufMetadata, key: &str) -> Result<Vec<String>> {
    require_array(metadata, key, GgufValueType::String)?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .map(str::to_owned)
                .with_context(|| format!("GGUF metadata `{key}` element {index} is not STRING"))
        })
        .collect()
}

fn require_array<'a>(
    metadata: &'a GgufMetadata,
    key: &str,
    element_type: GgufValueType,
) -> Result<&'a [GgufValue]> {
    let value = metadata
        .get(key)
        .with_context(|| format!("required GGUF metadata `{key}` is missing"))?;
    ensure!(
        value.array_element_type() == Some(element_type),
        "GGUF metadata `{key}` must be ARRAY<{element_type:?}>"
    );
    value.as_array().context("metadata value is not an array")
}

fn require_i32_array(metadata: &GgufMetadata, key: &str) -> Result<Vec<i32>> {
    let value = metadata
        .get(key)
        .with_context(|| format!("required GGUF metadata `{key}` is missing"))?;
    ensure!(
        value.array_element_type() == Some(GgufValueType::I32),
        "GGUF metadata `{key}` must be ARRAY<INT32>"
    );
    value
        .as_array()
        .context("metadata value is not an array")?
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            GgufValue::I32(value) => Ok(*value),
            _ => bail!("GGUF metadata `{key}` element {index} is not INT32"),
        })
        .collect()
}

fn require_f32_array(metadata: &GgufMetadata, key: &str) -> Result<Vec<f32>> {
    require_array(metadata, key, GgufValueType::F32)?
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            GgufValue::F32(value) => Ok(*value),
            _ => bail!("GGUF metadata `{key}` element {index} is not FLOAT32"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write, path::PathBuf};

    use candle_core::{DType, Device};

    use crate::model_weights::GgufWeightSource;

    use super::*;

    #[test]
    fn synthetic_falcon_h1_builds_parallel_attention_mamba2_runtime() -> Result<()> {
        let file = synthetic_falcon_h1_gguf(SyntheticFalconH1Config::default())?;
        let model = GgufModel::open(file.path())?;
        let spec = model.spec();
        assert_eq!(spec.architecture, "falcon-h1");
        assert_eq!(spec.runtime.vocab_size, 5);
        assert_eq!(spec.runtime.stop_token_ids, [1]);
        assert_eq!(spec.hidden_size, 4);
        assert_eq!(spec.intermediate_size, 6);
        assert_eq!(spec.num_layers, 1);
        assert_eq!(spec.num_query_heads, 2);
        assert_eq!(spec.num_kv_heads, 1);
        assert_eq!(spec.head_dim, 2);
        assert_eq!(spec.kv_bytes_per_token(2)?, 8);
        assert_eq!(spec.fixed_sequence_state_bytes(4)?, 96);
        assert!(spec.tied_embeddings);
        let SequenceMixerSpec::ParallelAttentionMamba2(parallel) = &spec.layer_specs[0].mixer
        else {
            panic!("Falcon-H1 must lower to parallel attention/Mamba2");
        };
        assert_eq!(parallel.attention.kind, AttentionKind::FullCausal);
        assert!(
            (parallel
                .attention
                .rope()
                .expect("Falcon-H1 attention uses RoPE")
                .theta
                / 1.0e11
                - 1.0)
                .abs()
                < 1e-6
        );
        assert_eq!(parallel.mamba2.convolution_kernel_size, 3);
        assert_eq!(parallel.mamba2.inner_size, 4);
        assert_eq!(parallel.mamba2.state_size, 2);
        assert_eq!(parallel.mamba2.num_heads, 2);
        assert_eq!(parallel.mamba2.num_groups, 1);
        assert_eq!(parallel.mamba2.head_dim, 2);
        assert!(parallel.mamba2.convolution_bias);
        assert!(!parallel.mamba2.gated_rms_norm);
        assert!(model.backend_config().llama_qk.is_some());
        assert!(model.adapter()?.format_chat(None, "Hello").is_err());

        let encoded = model
            .tokenizer()
            .encode("a", true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(encoded.get_ids(), &[0, 2]);

        // Exercise every canonical Falcon-H1 weight path through the shared decoder.
        // SAFETY: the NamedTempFile remains alive and immutable for the model.
        let backend = unsafe { model.file().backend(model.backend_config()) }?;
        let source = GgufWeightSource::new(backend, DType::F32, Device::Cpu);
        let runtime = model.adapter()?.build(&source)?;
        let mut session = runtime.new_session()?;
        assert_eq!(
            runtime
                .forward_tokens(session.as_mut(), encoded.get_ids(), 0)?
                .dims2()?,
            (1, 5)
        );
        assert_eq!(session.cache_len(), 2);
        Ok(())
    }

    #[test]
    fn synthetic_falcon_h1_rejects_unmodeled_profile_variants() -> Result<()> {
        for (file, expected) in [
            (
                synthetic_falcon_h1_gguf(SyntheticFalconH1Config {
                    omit: Some("blk.0.ssm_d"),
                    ..SyntheticFalconH1Config::default()
                })?,
                "ssm_d",
            ),
            (
                synthetic_falcon_h1_gguf(SyntheticFalconH1Config {
                    wrong_shape: Some(("blk.0.ssm_conv1d.weight", &[7, 3])),
                    ..SyntheticFalconH1Config::default()
                })?,
                "ssm_conv1d.weight",
            ),
            (
                synthetic_falcon_h1_gguf(SyntheticFalconH1Config {
                    extra_tensor: Some(("blk.0.ssm_norm.weight", &[4, 1])),
                    ..SyntheticFalconH1Config::default()
                })?,
                "gated Mamba2 RMS normalization",
            ),
            (
                synthetic_falcon_h1_gguf(SyntheticFalconH1Config {
                    extra_tensor: Some(("blk.0.unmodeled.bias", &[4])),
                    ..SyntheticFalconH1Config::default()
                })?,
                "unmodeled bias tensor",
            ),
            (
                synthetic_falcon_h1_gguf(SyntheticFalconH1Config {
                    extra_tensor: Some(("blk.0.attn_q_norm.weight", &[2])),
                    ..SyntheticFalconH1Config::default()
                })?,
                "unsupported tensor `blk.0.attn_q_norm.weight`",
            ),
            (
                synthetic_falcon_h1_gguf(SyntheticFalconH1Config {
                    extra_tensor: Some(("blk.0.attn_k_norm.weight", &[2])),
                    ..SyntheticFalconH1Config::default()
                })?,
                "unsupported tensor `blk.0.attn_k_norm.weight`",
            ),
            (
                synthetic_falcon_h1_gguf(SyntheticFalconH1Config {
                    extra_tensor: Some(("blk.0.unmodeled.weight", &[4])),
                    ..SyntheticFalconH1Config::default()
                })?,
                "unsupported tensor `blk.0.unmodeled.weight`",
            ),
            (
                synthetic_falcon_h1_gguf(SyntheticFalconH1Config {
                    extra_tensor: Some(("blk.1.attn_q.weight", &[4, 4])),
                    ..SyntheticFalconH1Config::default()
                })?,
                "outside the declared 1-layer profile",
            ),
        ] {
            let error = GgufModel::open(file.path())
                .expect_err("unsupported Falcon-H1 profile must fail closed")
                .to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?} in error, got {error:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn synthetic_standalone_llama_builds_without_sidecars() -> Result<()> {
        let file = synthetic_llama_gguf(None)?;
        let model = GgufModel::open(file.path())?;
        assert_eq!(model.spec().runtime.vocab_size, 4);
        assert_eq!(model.spec().hidden_size, 4);
        assert_eq!(model.spec().intermediate_size, 6);
        assert_eq!(model.spec().num_query_heads, 2);
        assert_eq!(model.spec().num_kv_heads, 1);
        assert_eq!(model.spec().head_dim, 2);
        assert!(model.spec().tied_embeddings);
        let ids = model
            .tokenizer()
            .encode("<|stop|>a12", false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(ids.get_ids(), &[0, 1, 2, 3]);
        assert!(model.adapter()?.format_chat(None, "Hello").is_err());

        let missing = synthetic_llama_gguf(Some("blk.0.attn_q.weight"))?;
        assert!(GgufModel::open(missing.path()).is_err());
        Ok(())
    }

    #[test]
    fn synthetic_llama_composes_sliding_attention_and_generic_sparse_moe() -> Result<()> {
        let file = synthetic_llama_gguf_with(
            None,
            SyntheticLlamaConfig {
                sliding_window: Some(4),
                expert_count: Some(3),
                expert_used_count: Some(2),
                expert_weights_scale: Some(1.0),
                ..SyntheticLlamaConfig::default()
            },
        )?;
        let model = GgufModel::open(file.path())?;
        let spec = model.spec();
        assert_eq!(spec.architecture, "llama");
        assert_eq!(
            spec.layer_specs[0]
                .attention()
                .context("Llama layer is not attention")?
                .kind,
            AttentionKind::SlidingCausal { window: 4 }
        );
        assert_eq!(
            spec.layer_specs[0].feed_forward,
            FeedForwardKind::SparseMoe(SparseMoeSpec {
                expert_count: 3,
                experts_per_token: 2,
                expert_intermediate_size: 6,
                router_weighting: RouterWeighting::SelectedSoftmax,
            })
        );

        // Exercise the rank-three expert banks and sliding session state through
        // the same shared decoder used by dense Llama profiles.
        // SAFETY: the NamedTempFile remains alive and immutable for the model.
        let backend = unsafe { model.file().backend(model.backend_config()) }?;
        let source = GgufWeightSource::new(backend, DType::F32, Device::Cpu);
        let runtime = model.adapter()?.build(&source)?;
        let mut session = runtime.new_session()?;
        let prefix = [1, 2, 1, 2, 1, 2];
        let logits = runtime.forward_tokens(session.as_mut(), &prefix, 0)?;
        assert_eq!(logits.dims2()?, (1, 4));
        assert!(
            logits
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .all(|value| value == 0.0)
        );
        assert_eq!(session.cache_len(), prefix.len());
        let stats = session
            .kv_cache_stats()
            .context("Llama sliding session has no KV diagnostics")?;
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].policy, crate::cache::KvCachePolicy::Sliding);
        assert_eq!(stats[0].logical_len, prefix.len());
        assert_eq!(stats[0].resident_start, 3);
        assert_eq!(stats[0].resident_len, 3);

        let cached = runtime.forward_tokens(session.as_mut(), &[1], prefix.len())?;
        let mut full_session = runtime.new_session()?;
        let full = runtime.forward_tokens(full_session.as_mut(), &[1, 2, 1, 2, 1, 2, 1], 0)?;
        assert_eq!(
            cached.flatten_all()?.to_vec1::<f32>()?,
            full.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn synthetic_llama_zero_sliding_window_keeps_full_attention() -> Result<()> {
        let file = synthetic_llama_gguf_with(
            None,
            SyntheticLlamaConfig {
                sliding_window: Some(0),
                ..SyntheticLlamaConfig::default()
            },
        )?;
        let model = GgufModel::open(file.path())?;
        assert_eq!(
            model.spec().layer_specs[0]
                .attention()
                .context("Llama layer is not attention")?
                .kind,
            AttentionKind::FullCausal
        );
        Ok(())
    }

    #[test]
    fn explicit_key_length_allows_nonstandard_projection_width() -> Result<()> {
        let file = synthetic_llama_gguf_with(
            None,
            SyntheticLlamaConfig {
                hidden_size: 5,
                key_length: Some(2),
                ..SyntheticLlamaConfig::default()
            },
        )?;
        let model = GgufModel::open(file.path())?;
        assert_eq!(model.spec().hidden_size, 5);
        assert_eq!(model.spec().num_query_heads, 2);
        assert_eq!(model.spec().head_dim, 2);
        Ok(())
    }

    #[test]
    fn synthetic_llama_reads_generic_rope_frequency_factors() -> Result<()> {
        let file = synthetic_llama_gguf_with(
            None,
            SyntheticLlamaConfig {
                hidden_size: 8,
                query_heads: 2,
                kv_heads: 1,
                rope_factors: Some(&[1.0, 4.0]),
                ..SyntheticLlamaConfig::default()
            },
        )?;
        let model = GgufModel::open(file.path())?;
        assert_eq!(
            model.spec().layer_specs[0]
                .attention()
                .expect("Llama layer uses attention")
                .rope()
                .expect("Llama layer uses RoPE")
                .frequency_factors
                .as_deref(),
            Some([1.0, 4.0].as_slice())
        );

        for invalid_factors in [
            &[0.0_f32][..],
            &[-1.0][..],
            &[f32::NAN][..],
            &[f32::INFINITY][..],
            &[1.0, 2.0][..],
        ] {
            let invalid = synthetic_llama_gguf_with(
                None,
                SyntheticLlamaConfig {
                    rope_factors: Some(invalid_factors),
                    ..SyntheticLlamaConfig::default()
                },
            )?;
            assert!(GgufModel::open(invalid.path()).is_err());
        }
        Ok(())
    }

    #[test]
    fn synthetic_qwen2_derives_vocab_and_disables_llama_qk_permutation() -> Result<()> {
        let file = synthetic_qwen2_gguf(None, SyntheticLlamaConfig::default(), None)?;
        let model = GgufModel::open(file.path())?;
        assert_eq!(model.spec().architecture, "qwen2");
        assert_eq!(model.spec().runtime.vocab_size, 4);
        assert_eq!(model.spec().runtime.stop_token_ids, vec![0]);
        assert!(model.spec().projection_bias.query);
        assert!(model.spec().projection_bias.key);
        assert!(model.spec().projection_bias.value);
        assert!(!model.spec().projection_bias.output);
        assert!(!model.spec().mlp_bias);
        assert_eq!(model.backend_config().llama_qk, None);
        assert!(model.adapter()?.format_chat(None, "Hello").is_err());
        Ok(())
    }

    #[test]
    fn synthetic_qwen3_preserves_explicit_attention_width_and_qk_norm() -> Result<()> {
        let config = SyntheticLlamaConfig {
            hidden_size: 6,
            key_length: Some(2),
            projection_biases: false,
            qk_norm: true,
            ..SyntheticLlamaConfig::default()
        };
        let file = synthetic_qwen3_gguf(None, config, None)?;
        let model = GgufModel::open(file.path())?;
        let spec = model.spec();
        assert_eq!(spec.architecture, "qwen3");
        assert_eq!(spec.runtime.vocab_size, 4);
        assert_eq!(spec.runtime.stop_token_ids, [0]);
        assert_eq!(spec.hidden_size, 6);
        assert_eq!(spec.head_dim, 2);
        assert_ne!(spec.head_dim, spec.hidden_size / spec.num_query_heads);
        assert_eq!(spec.num_query_heads * spec.head_dim, 4);
        assert_ne!(spec.num_query_heads * spec.head_dim, spec.hidden_size);
        assert!(
            spec.layer_specs
                .iter()
                .all(|layer| layer.attention().is_some_and(|attention| attention.qk_norm))
        );
        assert!(!spec.projection_bias.query);
        assert!(!spec.projection_bias.key);
        assert!(!spec.projection_bias.value);
        assert!(!spec.projection_bias.output);
        assert!(spec.tied_embeddings);
        assert_eq!(model.backend_config().llama_qk, None);

        // Exercise canonical Q/K norm mapping and the real shared decoder.
        // SAFETY: the NamedTempFile remains alive and immutable for the model.
        let backend = unsafe { model.file().backend(model.backend_config()) }?;
        let source = GgufWeightSource::new(backend, DType::F32, Device::Cpu);
        let runtime = model.adapter()?.build(&source)?;
        let mut session = runtime.new_session()?;
        assert_eq!(
            runtime
                .forward_tokens(session.as_mut(), &[1, 2], 0)?
                .dims2()?,
            (1, 4)
        );
        Ok(())
    }

    #[test]
    fn synthetic_smollm3_builds_nope_schedule_and_preserves_cache_parity() -> Result<()> {
        let file = synthetic_smollm3_gguf(SyntheticSmolLm3Config::default())?;
        let model = GgufModel::open(file.path())?;
        let spec = model.spec();
        assert_eq!(spec.architecture, "smollm3");
        assert_eq!(spec.runtime.vocab_size, 4);
        assert_eq!(spec.runtime.stop_token_ids, [0, 1]);
        assert_eq!(spec.num_layers, 36);
        assert_eq!(spec.num_query_heads, 2);
        assert_eq!(spec.num_kv_heads, 1);
        assert_eq!(spec.head_dim, 2);
        assert!(spec.tied_embeddings);
        assert!(matches!(
            &spec.layer_specs[2]
                .attention()
                .context("SmolLM3 layer 2 is not attention")?
                .position_encoding,
            PositionEncodingSpec::Rope(_)
        ));
        assert!(matches!(
            &spec.layer_specs[3]
                .attention()
                .context("SmolLM3 layer 3 is not attention")?
                .position_encoding,
            PositionEncodingSpec::None
        ));
        for (index, layer) in spec.layer_specs.iter().enumerate() {
            let attention = layer
                .attention()
                .context("SmolLM3 layer is not attention")?;
            assert_eq!(
                matches!(&attention.position_encoding, PositionEncodingSpec::None),
                (index + 1).is_multiple_of(4),
                "unexpected SmolLM3 NoPE schedule at layer {index}"
            );
        }
        assert_eq!(
            model.backend_config().llama_qk,
            Some(LlamaQkConfig {
                query_heads: 2,
                kv_heads: 1,
            })
        );
        assert!(model.backend_config().fallback_output_to_embedding);
        assert_eq!(model.adapter()?.family(), "smollm3");

        // Exercise all 36 layers through the same CPU decoder used by real
        // checkpoints, then compare incremental decode with one-shot prefill.
        // SAFETY: the NamedTempFile remains alive and immutable for the model.
        let backend = unsafe { model.file().backend(model.backend_config()) }?;
        let source = GgufWeightSource::new(backend, DType::F32, Device::Cpu);
        let runtime = model.adapter()?.build(&source)?;
        let mut cached_session = runtime.new_session()?;
        let _ = runtime.forward_tokens(cached_session.as_mut(), &[2, 3], 0)?;
        let cached = runtime.forward_tokens(cached_session.as_mut(), &[2], 2)?;
        assert_eq!(cached_session.cache_len(), 3);
        drop(cached_session);
        let mut full_session = runtime.new_session()?;
        let full = runtime.forward_tokens(full_session.as_mut(), &[2, 3, 2], 0)?;
        assert_eq!(
            cached.flatten_all()?.to_vec1::<f32>()?,
            full.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn synthetic_smollm3_rejects_unmodeled_metadata_and_tensors() -> Result<()> {
        for (config, expected) in [
            (
                SyntheticSmolLm3Config {
                    unsupported_metadata: Some("smollm3.rope.scaling.type"),
                    ..SyntheticSmolLm3Config::default()
                },
                "metadata `smollm3.rope.scaling.type` is not supported",
            ),
            (
                SyntheticSmolLm3Config {
                    extra_tensor: Some(("rope_freqs.weight", &[1])),
                    ..SyntheticSmolLm3Config::default()
                },
                "rope frequency factors are not supported",
            ),
            (
                SyntheticSmolLm3Config {
                    extra_tensor: Some(("blk.0.attn_q.bias", &[4])),
                    ..SyntheticSmolLm3Config::default()
                },
                "unsupported tensor `blk.0.attn_q.bias`",
            ),
            (
                SyntheticSmolLm3Config {
                    extra_tensor: Some(("blk.0.attn_rot_embd", &[2])),
                    ..SyntheticSmolLm3Config::default()
                },
                "unsupported tensor `blk.0.attn_rot_embd`",
            ),
        ] {
            let file = synthetic_smollm3_gguf(config)?;
            let error = GgufModel::open(file.path())
                .expect_err("unmodeled SmolLM3 semantics must fail closed")
                .to_string();
            assert!(error.contains(expected), "{error}");
        }
        Ok(())
    }

    #[test]
    fn synthetic_qwen3_rejects_missing_geometry_norms_biases_and_vocab() -> Result<()> {
        let valid = SyntheticLlamaConfig {
            key_length: Some(2),
            projection_biases: false,
            qk_norm: true,
            ..SyntheticLlamaConfig::default()
        };
        let cases = [
            (
                synthetic_qwen3_gguf(
                    None,
                    SyntheticLlamaConfig {
                        key_length: None,
                        projection_biases: false,
                        qk_norm: true,
                        ..SyntheticLlamaConfig::default()
                    },
                    None,
                )?,
                "qwen3.attention.key_length",
            ),
            (
                synthetic_qwen3_gguf(Some("blk.0.attn_q_norm.weight"), valid, None)?,
                "attn_q_norm.weight",
            ),
            (
                synthetic_qwen3_gguf(Some("blk.0.attn_k_norm.weight"), valid, None)?,
                "attn_k_norm.weight",
            ),
            (
                synthetic_qwen3_gguf(
                    None,
                    SyntheticLlamaConfig {
                        projection_biases: true,
                        ..valid
                    },
                    None,
                )?,
                "does not support `attn_q.bias`",
            ),
            (synthetic_qwen3_gguf(None, valid, Some(5))?, "declares 5"),
        ];
        for (file, expected) in cases {
            let error = GgufModel::open(file.path()).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "expected `{expected}` in error, got `{error}`"
            );
        }
        Ok(())
    }

    #[test]
    fn synthetic_qwen3_rejects_unmodeled_attention_semantics() -> Result<()> {
        let valid = SyntheticLlamaConfig {
            key_length: Some(2),
            projection_biases: false,
            qk_norm: true,
            ..SyntheticLlamaConfig::default()
        };
        let cases = [
            (
                synthetic_qwen3_gguf(
                    None,
                    SyntheticLlamaConfig {
                        unsupported_llama_key: Some("llama.unmodeled"),
                        ..valid
                    },
                    None,
                )?,
                "metadata `qwen3.unmodeled`",
            ),
            (
                synthetic_qwen3_gguf(
                    None,
                    SyntheticLlamaConfig {
                        causal: Some(false),
                        ..valid
                    },
                    None,
                )?,
                "non-causal",
            ),
            (
                synthetic_qwen3_gguf(
                    None,
                    SyntheticLlamaConfig {
                        sliding_window: Some(4),
                        ..valid
                    },
                    None,
                )?,
                "sliding-window",
            ),
            (
                synthetic_qwen3_gguf(
                    None,
                    SyntheticLlamaConfig {
                        value_length: Some(4),
                        ..valid
                    },
                    None,
                )?,
                "different Qwen3 key/value head widths",
            ),
        ];
        for (file, expected) in cases {
            let error = GgufModel::open(file.path()).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "expected `{expected}` in error, got `{error}`"
            );
        }
        Ok(())
    }

    #[test]
    fn synthetic_qwen3_rejects_unmodeled_rope_norm_and_bias_semantics() -> Result<()> {
        let valid = SyntheticLlamaConfig {
            key_length: Some(2),
            projection_biases: false,
            qk_norm: true,
            ..SyntheticLlamaConfig::default()
        };
        let cases = [
            (
                synthetic_qwen3_gguf(
                    None,
                    SyntheticLlamaConfig {
                        rope_dimension: Some(1),
                        ..valid
                    },
                    None,
                )?,
                "partial Qwen3 RoPE",
            ),
            (
                synthetic_qwen3_gguf(
                    None,
                    SyntheticLlamaConfig {
                        rope_factors: Some(&[1.0]),
                        ..valid
                    },
                    None,
                )?,
                "rope frequency factors",
            ),
            (
                synthetic_qwen3_gguf(
                    None,
                    SyntheticLlamaConfig {
                        norm_epsilon: 0.0,
                        ..valid
                    },
                    None,
                )?,
                "RMS norm epsilon",
            ),
            (
                synthetic_qwen3_gguf(
                    None,
                    SyntheticLlamaConfig {
                        rope_theta: 0.0,
                        ..valid
                    },
                    None,
                )?,
                "RoPE frequency base",
            ),
        ];
        for (file, expected) in cases {
            let error = GgufModel::open(file.path()).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "expected `{expected}` in error, got `{error}`"
            );
        }

        for bias in [
            "blk.0.attn_k.bias",
            "blk.0.attn_v.bias",
            "blk.0.attn_output.bias",
            "blk.0.ffn_gate.bias",
            "blk.0.ffn_up.bias",
            "blk.0.ffn_down.bias",
            "output.bias",
        ] {
            let file = synthetic_qwen3_gguf(
                None,
                SyntheticLlamaConfig {
                    extra_bias: Some(bias),
                    ..valid
                },
                None,
            )?;
            let error = GgufModel::open(file.path()).unwrap_err().to_string();
            assert!(
                error.contains("bias"),
                "expected `{bias}` to be rejected, got `{error}`"
            );
        }
        Ok(())
    }

    #[test]
    fn synthetic_qwen3_moe_builds_qwen_attention_and_generic_sparse_runtime() -> Result<()> {
        let file = synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig::default())?;
        let model = GgufModel::open(file.path())?;
        let spec = model.spec();
        assert_eq!(spec.architecture, "qwen3moe");
        assert_eq!(spec.runtime.vocab_size, 4);
        assert_eq!(spec.runtime.stop_token_ids, [0]);
        assert_eq!(spec.hidden_size, 6);
        assert_eq!(spec.intermediate_size, 6);
        assert_eq!(spec.num_query_heads, 2);
        assert_eq!(spec.num_kv_heads, 1);
        assert_eq!(spec.head_dim, 2);
        assert_eq!(spec.num_query_heads * spec.head_dim, 4);
        assert_ne!(spec.num_query_heads * spec.head_dim, spec.hidden_size);
        assert!(spec.tied_embeddings);
        let attention = spec.layer_specs[0]
            .attention()
            .context("Qwen3MoE layer is not attention")?;
        assert!(attention.qk_norm);
        assert!((attention.scale - 1.0 / 2.0_f64.sqrt()).abs() < f64::EPSILON);
        assert_eq!(
            spec.layer_specs[0].feed_forward,
            FeedForwardKind::SparseMoe(SparseMoeSpec {
                expert_count: 3,
                experts_per_token: 2,
                expert_intermediate_size: 3,
                router_weighting: RouterWeighting::SelectedSoftmax,
            })
        );
        assert_eq!(model.backend_config().llama_qk, None);
        assert_eq!(model.adapter()?.family(), "qwen3moe");
        assert!(model.adapter()?.format_chat(None, "Hello").is_err());

        // Exercise the canonical rank-three banks, selected routing and the
        // complete shared decoder rather than metadata parsing alone.
        // SAFETY: the NamedTempFile remains alive and immutable for the model.
        let backend = unsafe { model.file().backend(model.backend_config()) }?;
        let source = GgufWeightSource::new(backend, DType::F32, Device::Cpu);
        let runtime = model.adapter()?.build(&source)?;
        let mut session = runtime.new_session()?;
        let logits = runtime.forward_tokens(session.as_mut(), &[1, 2], 0)?;
        assert_eq!(logits.dims2()?, (1, 4));
        assert!(
            logits
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .all(|value| value == 0.0)
        );
        Ok(())
    }

    #[test]
    fn synthetic_qwen3_moe_matches_explicit_and_legacy_expert_width_rules() -> Result<()> {
        for explicit in [None, Some(0), Some(4)] {
            let file = synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
                expert_feed_forward_length: explicit,
                ..SyntheticQwen3MoeConfig::default()
            })?;
            let model = GgufModel::open(file.path())?;
            let FeedForwardKind::SparseMoe(moe) = model.spec().layer_specs[0].feed_forward else {
                panic!("Qwen3MoE layer is not sparse MoE")
            };
            assert_eq!(model.spec().intermediate_size, 6);
            assert_eq!(
                moe.expert_intermediate_size,
                if explicit == Some(4) { 4 } else { 3 }
            );
        }

        let zero_shared_width = synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
            extra_metadata: Some((
                "qwen3moe.expert_shared_feed_forward_length",
                GgufValue::U32(0),
            )),
            ..SyntheticQwen3MoeConfig::default()
        })?;
        assert_eq!(
            GgufModel::open(zero_shared_width.path())?
                .spec()
                .architecture,
            "qwen3moe"
        );
        Ok(())
    }

    #[test]
    fn synthetic_qwen3_moe_rejects_unmodeled_or_inconsistent_contracts() -> Result<()> {
        let cases = [
            (
                synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
                    omit: Some("blk.0.attn_q_norm.weight"),
                    ..SyntheticQwen3MoeConfig::default()
                })?,
                "attn_q_norm.weight",
            ),
            (
                synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
                    omit: Some("blk.0.ffn_gate_inp.weight"),
                    ..SyntheticQwen3MoeConfig::default()
                })?,
                "ffn_gate_inp.weight",
            ),
            (
                synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
                    extra_tensor: Some(("blk.0.ffn_gate.weight", &[4, 6])),
                    ..SyntheticQwen3MoeConfig::default()
                })?,
                "unsupported dense tensor",
            ),
            (
                synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
                    extra_tensor: Some(("blk.0.ffn_gate_inp.bias", &[3])),
                    ..SyntheticQwen3MoeConfig::default()
                })?,
                "bias tensor",
            ),
            (
                synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
                    extra_tensor: Some(("blk.0.ffn_gate_shexp.weight", &[4, 3])),
                    ..SyntheticQwen3MoeConfig::default()
                })?,
                "shared experts",
            ),
            (
                synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
                    extra_metadata: Some(("qwen3moe.unmodeled", GgufValue::U32(1))),
                    ..SyntheticQwen3MoeConfig::default()
                })?,
                "metadata `qwen3moe.unmodeled`",
            ),
            (
                synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
                    extra_metadata: Some((
                        "qwen3moe.expert_shared_feed_forward_length",
                        GgufValue::U32(4),
                    )),
                    ..SyntheticQwen3MoeConfig::default()
                })?,
                "shared experts",
            ),
            (
                synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
                    extra_metadata: Some(("qwen3moe.attention.causal", GgufValue::Bool(false))),
                    ..SyntheticQwen3MoeConfig::default()
                })?,
                "non-causal",
            ),
            (
                synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
                    extra_metadata: Some(("qwen3moe.attention.sliding_window", GgufValue::U32(4))),
                    ..SyntheticQwen3MoeConfig::default()
                })?,
                "sliding-window",
            ),
            (
                synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
                    feed_forward_length: 5,
                    expert_feed_forward_length: None,
                    ..SyntheticQwen3MoeConfig::default()
                })?,
                "must be divisible",
            ),
            (
                synthetic_qwen3_moe_gguf(SyntheticQwen3MoeConfig {
                    experts_per_token: 0,
                    ..SyntheticQwen3MoeConfig::default()
                })?,
                "experts-per-token",
            ),
        ];
        for (file, expected) in cases {
            let error = GgufModel::open(file.path())
                .expect_err("invalid Qwen3MoE profile must fail closed")
                .to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?} in error, got {error:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn synthetic_lfm2_builds_hybrid_runtime_from_signed_layer_schedule() -> Result<()> {
        let file = synthetic_lfm2_gguf(SyntheticLfm2Config::default())?;
        let model = GgufModel::open(file.path())?;
        let spec = model.spec();
        assert_eq!(spec.architecture, "lfm2");
        assert_eq!(spec.runtime.vocab_size, 5);
        assert_eq!(spec.runtime.stop_token_ids, [1]);
        assert_eq!(spec.hidden_size, 4);
        assert_eq!(spec.intermediate_size, 6);
        assert_eq!(spec.num_query_heads, 2);
        assert_eq!(spec.num_kv_heads, 1);
        assert_eq!(spec.head_dim, 2);
        assert_eq!(spec.kv_bytes_per_token(2)?, 8);
        assert!(matches!(
            spec.layer_specs[0].mixer,
            SequenceMixerSpec::ShortConvolution(ShortConvolutionSpec { kernel_size: 3 })
        ));
        assert!(
            spec.layer_specs[1]
                .attention()
                .is_some_and(|attention| attention.qk_norm)
        );
        assert!(spec.tied_embeddings);
        assert_eq!(model.backend_config().llama_qk, None);

        // Exercise the canonical LFM2 tensor mapping and both mixer variants,
        // not only metadata parsing. All-zero tensors make the result exact.
        // SAFETY: the NamedTempFile remains alive and immutable for the model.
        let backend = unsafe { model.file().backend(model.backend_config()) }?;
        let source = GgufWeightSource::new(backend, DType::F32, Device::Cpu);
        let runtime = model.adapter()?.build(&source)?;
        let mut session = runtime.new_session()?;
        let logits = runtime.forward_tokens(session.as_mut(), &[2, 3], 0)?;
        assert_eq!(logits.dims2()?, (1, 5));
        assert!(
            logits
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .all(|value| value == 0.0)
        );
        Ok(())
    }

    #[test]
    fn synthetic_qwen35_builds_gated_delta_and_output_gated_attention() -> Result<()> {
        let file = synthetic_qwen35_gguf(None)?;
        let model = GgufModel::open(file.path())?;
        let spec = model.spec();
        assert_eq!(spec.architecture, "qwen35");
        assert_eq!(spec.runtime.vocab_size, 5);
        assert_eq!(spec.runtime.stop_token_ids, [1, 0]);
        assert_eq!(spec.hidden_size, 8);
        assert_eq!(spec.intermediate_size, 12);
        assert_eq!(spec.num_layers, 4);
        assert_eq!(spec.num_query_heads, 2);
        assert_eq!(spec.num_kv_heads, 1);
        assert_eq!(spec.head_dim, 4);
        for layer in 0..3 {
            assert!(matches!(
                spec.layer_specs[layer].mixer,
                SequenceMixerSpec::GatedDeltaNet(GatedDeltaNetSpec {
                    convolution_kernel_size: 4,
                    num_key_heads: 2,
                    num_value_heads: 2,
                    key_head_dim: 4,
                    value_head_dim: 4,
                    ..
                })
            ));
        }
        let attention = spec.layer_specs[3]
            .attention()
            .context("Qwen3.5 layer 3 is not full attention")?;
        assert!(attention.qk_norm);
        assert!(attention.output_gate);
        assert_eq!(
            attention
                .rope()
                .context("Qwen3.5 attention does not use RoPE")?
                .rotary_dim,
            4
        );
        assert!(spec.tied_embeddings);
        assert_eq!(model.backend_config().llama_qk, None);
        assert!(model.adapter()?.format_chat(None, "Hello").is_err());

        // Exercise every Qwen3.5-specific tensor alias through the actual
        // decoder loader and both mixer variants.
        // SAFETY: the NamedTempFile remains alive and immutable for the model.
        let backend = unsafe { model.file().backend(model.backend_config()) }?;
        let source = GgufWeightSource::new(backend, DType::F32, Device::Cpu);
        let runtime = model.adapter()?.build(&source)?;
        let mut session = runtime.new_session()?;
        let logits = runtime.forward_tokens(session.as_mut(), &[2, 3], 0)?;
        assert_eq!(logits.dims2()?, (1, 5));
        assert!(
            logits
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .all(|value| value == 0.0)
        );
        Ok(())
    }

    #[test]
    fn synthetic_qwen35_rejects_mtp_and_out_of_range_layers() -> Result<()> {
        for (extra_tensor, expected) in [
            ("blk.4.attn_norm.weight", "outside the declared"),
            ("blk.3.nextn.eh_proj.weight", "NextN/MTP"),
        ] {
            let file = synthetic_qwen35_gguf(Some((extra_tensor, &[8][..])))?;
            let error = GgufModel::open(file.path()).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "expected `{expected}` in error, got `{error}`"
            );
        }
        Ok(())
    }

    #[test]
    fn synthetic_lfm2_rejects_invalid_schedule_and_mixer_tensors() -> Result<()> {
        let cases = [
            (
                SyntheticLfm2Config {
                    kv_heads_by_layer: &[0],
                    ..SyntheticLfm2Config::default()
                },
                "has 1 layer values, expected 2",
            ),
            (
                SyntheticLfm2Config {
                    kv_heads_by_layer: &[-1, 1],
                    ..SyntheticLfm2Config::default()
                },
                "element 0 must be a non-negative integer",
            ),
            (
                SyntheticLfm2Config {
                    kv_heads_by_layer: &[0, 0],
                    ..SyntheticLfm2Config::default()
                },
                "declares no attention layers",
            ),
            (
                SyntheticLfm2Config {
                    kv_heads_by_layer: &[1, 2],
                    ..SyntheticLfm2Config::default()
                },
                "one consistent KV-head count",
            ),
            (
                SyntheticLfm2Config {
                    wrong_shape: Some(("blk.0.shortconv.conv.weight", &[4, 4])),
                    ..SyntheticLfm2Config::default()
                },
                "shortconv.conv.weight",
            ),
            (
                SyntheticLfm2Config {
                    extra_tensor: Some(("blk.1.shortconv.conv.weight", &[3, 4])),
                    ..SyntheticLfm2Config::default()
                },
                "attention layer 1 contains short-convolution tensor",
            ),
            (
                SyntheticLfm2Config {
                    extra_tensor: Some(("blk.0.attn_q.weight", &[4, 4])),
                    ..SyntheticLfm2Config::default()
                },
                "short-convolution layer 0 contains attention tensor",
            ),
            (
                SyntheticLfm2Config {
                    omit: Some("blk.1.attn_q_norm.weight"),
                    ..SyntheticLfm2Config::default()
                },
                "attn_q_norm.weight",
            ),
            (
                SyntheticLfm2Config {
                    extra_tensor: Some(("blk.0.shortconv.in_proj.bias", &[12])),
                    ..SyntheticLfm2Config::default()
                },
                "does not support bias tensor",
            ),
            (
                SyntheticLfm2Config {
                    extra_tensor: Some(("blk.1.attn_q.bias", &[4])),
                    ..SyntheticLfm2Config::default()
                },
                "does not support bias tensor",
            ),
            (
                SyntheticLfm2Config {
                    extra_tensor: Some(("blk.0.ffn_gate.bias", &[6])),
                    ..SyntheticLfm2Config::default()
                },
                "does not support bias tensor",
            ),
            (
                SyntheticLfm2Config {
                    extra_tensor: Some(("output.bias", &[5])),
                    ..SyntheticLfm2Config::default()
                },
                "does not support bias tensor",
            ),
        ];
        for (config, expected) in cases {
            let file = synthetic_lfm2_gguf(config)?;
            let error = GgufModel::open(file.path()).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "expected `{expected}` in error, got `{error}`"
            );
        }
        Ok(())
    }

    #[test]
    fn synthetic_lfm2_rejects_unmodeled_rope_frequency_factors() -> Result<()> {
        let file = synthetic_lfm2_gguf(SyntheticLfm2Config {
            extra_tensor: Some(("rope_freqs.weight", &[1])),
            ..SyntheticLfm2Config::default()
        })?;
        let error = GgufModel::open(file.path()).unwrap_err().to_string();
        assert!(
            error.contains("rope frequency factors are not supported"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn synthetic_granite_moe_builds_generic_sparse_runtime() -> Result<()> {
        let file = synthetic_granite_moe_gguf(None)?;
        let model = GgufModel::open(file.path())?;
        let spec = model.spec();
        assert_eq!(spec.architecture, "granitemoe");
        assert_eq!(spec.runtime.vocab_size, 4);
        assert_eq!(spec.runtime.stop_token_ids, [0]);
        assert_eq!(spec.hidden_size, 4);
        assert_eq!(spec.intermediate_size, 3);
        assert_eq!(spec.num_query_heads, 2);
        assert_eq!(spec.num_kv_heads, 1);
        assert_eq!(spec.head_dim, 2);
        assert!((spec.embedding_scale - 12.0).abs() < f64::EPSILON);
        assert!((spec.final_logit_scale - 1.0 / 6.0).abs() < f64::EPSILON);
        assert!(spec.tied_embeddings);
        assert!(
            (spec.layer_specs[0]
                .attention()
                .expect("Granite layer uses attention")
                .scale
                - 0.015_625)
                .abs()
                < f64::EPSILON
        );
        assert!((spec.layer_specs[0].mixer_residual_scale - 0.22).abs() < 1e-7);
        assert!((spec.layer_specs[0].feed_forward_residual_scale - 0.22).abs() < 1e-7);
        assert_eq!(
            spec.layer_specs[0].feed_forward,
            FeedForwardKind::SparseMoe(SparseMoeSpec {
                expert_count: 3,
                experts_per_token: 2,
                expert_intermediate_size: 3,
                router_weighting: RouterWeighting::SelectedSoftmax,
            })
        );
        assert_eq!(
            model.backend_config().llama_qk,
            Some(LlamaQkConfig {
                query_heads: 2,
                kv_heads: 1,
            })
        );
        let ids = model
            .tokenizer()
            .encode("a12", false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(ids.get_ids(), [1, 2, 3]);

        // Exercise the real GGUF WeightSource, dense rank-three bank splitting,
        // routing and the complete shared decoder with an all-zero fixture.
        // SAFETY: the NamedTempFile remains alive and immutable for the model.
        let backend = unsafe { model.file().backend(model.backend_config()) }?;
        let source = GgufWeightSource::new(backend, DType::F32, Device::Cpu);
        let runtime = model.adapter()?.build(&source)?;
        let mut session = runtime.new_session()?;
        let logits = runtime.forward_tokens(session.as_mut(), &[1, 2], 0)?;
        assert_eq!(logits.dims2()?, (1, 4));
        assert!(
            logits
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .all(|value| value == 0.0)
        );

        let missing_router = synthetic_granite_moe_gguf(Some("blk.0.ffn_gate_inp.weight"))?;
        assert!(
            GgufModel::open(missing_router.path())
                .unwrap_err()
                .to_string()
                .contains("ffn_gate_inp.weight")
        );
        Ok(())
    }

    #[test]
    fn synthetic_qwen2_rejects_bad_vocab_bias_and_attention_semantics() -> Result<()> {
        let cases = [
            (
                synthetic_qwen2_gguf(None, SyntheticLlamaConfig::default(), Some(5))?,
                "declares 5",
            ),
            (
                synthetic_qwen2_gguf(
                    Some("blk.0.attn_q.bias"),
                    SyntheticLlamaConfig::default(),
                    None,
                )?,
                "requires `attn_q.bias`",
            ),
            (
                synthetic_qwen2_gguf(
                    None,
                    SyntheticLlamaConfig {
                        sliding_window: Some(8),
                        ..SyntheticLlamaConfig::default()
                    },
                    None,
                )?,
                "sliding-window",
            ),
            (
                synthetic_qwen2_gguf(
                    None,
                    SyntheticLlamaConfig {
                        unsupported_llama_key: Some("llama.rope.scaling.type"),
                        ..SyntheticLlamaConfig::default()
                    },
                    None,
                )?,
                "qwen2.rope.scaling.type",
            ),
            (
                synthetic_qwen2_gguf(
                    None,
                    SyntheticLlamaConfig {
                        rope_factors: Some(&[1.0]),
                        ..SyntheticLlamaConfig::default()
                    },
                    None,
                )?,
                "rope frequency factors",
            ),
        ];
        for (file, expected) in cases {
            let error = GgufModel::open(file.path()).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "expected `{expected}` in error, got `{error}`"
            );
        }
        Ok(())
    }

    #[test]
    #[allow(clippy::too_many_lines)] // Keeping every rejected metadata case together makes the fail-closed table auditable.
    fn rejects_unsupported_or_dangerous_llama_metadata() -> Result<()> {
        let cases = [
            (
                SyntheticLlamaConfig {
                    causal: Some(false),
                    ..SyntheticLlamaConfig::default()
                },
                "non-causal",
            ),
            (
                SyntheticLlamaConfig {
                    sliding_window: Some(17),
                    ..SyntheticLlamaConfig::default()
                },
                "sliding window 17 exceeds context length 16",
            ),
            (
                SyntheticLlamaConfig {
                    sliding_window_pattern: true,
                    ..SyntheticLlamaConfig::default()
                },
                "sliding_window_pattern",
            ),
            (
                SyntheticLlamaConfig {
                    attention_scale: Some(0.5),
                    ..SyntheticLlamaConfig::default()
                },
                "attention.scale",
            ),
            (
                SyntheticLlamaConfig {
                    unsupported_llama_key: Some("llama.rope.scale"),
                    ..SyntheticLlamaConfig::default()
                },
                "llama.rope.scale",
            ),
            (
                SyntheticLlamaConfig {
                    unsupported_llama_key: Some("llama.rope.scale_linear"),
                    ..SyntheticLlamaConfig::default()
                },
                "llama.rope.scale_linear",
            ),
            (
                SyntheticLlamaConfig {
                    unsupported_llama_key: Some("llama.rope.scaling.type"),
                    ..SyntheticLlamaConfig::default()
                },
                "llama.rope.scaling.type",
            ),
            (
                SyntheticLlamaConfig {
                    expert_used_count: Some(1),
                    ..SyntheticLlamaConfig::default()
                },
                "expert_used_count requires a positive expert_count",
            ),
            (
                SyntheticLlamaConfig {
                    expert_count: Some(3),
                    expert_used_count: Some(0),
                    ..SyntheticLlamaConfig::default()
                },
                "expert_used_count in 1..=3",
            ),
            (
                SyntheticLlamaConfig {
                    expert_count: Some(3),
                    expert_used_count: Some(4),
                    ..SyntheticLlamaConfig::default()
                },
                "expert_used_count in 1..=3",
            ),
            (
                SyntheticLlamaConfig {
                    expert_count: Some(3),
                    expert_used_count: Some(2),
                    expert_weights_scale: Some(0.5),
                    ..SyntheticLlamaConfig::default()
                },
                "expert weight scale",
            ),
            (
                SyntheticLlamaConfig {
                    context_length: MAX_GGUF_CONTEXT_LENGTH + 1,
                    ..SyntheticLlamaConfig::default()
                },
                "context length",
            ),
            (
                SyntheticLlamaConfig {
                    block_count: MAX_DECODER_LAYERS + 1,
                    ..SyntheticLlamaConfig::default()
                },
                "layer count",
            ),
            (
                SyntheticLlamaConfig {
                    rope_theta: f32::INFINITY,
                    ..SyntheticLlamaConfig::default()
                },
                "frequency base",
            ),
            (
                SyntheticLlamaConfig {
                    norm_epsilon: f32::NAN,
                    ..SyntheticLlamaConfig::default()
                },
                "norm epsilon",
            ),
            (
                SyntheticLlamaConfig {
                    split_metadata: true,
                    ..SyntheticLlamaConfig::default()
                },
                "split.no",
            ),
        ];
        for (config, expected) in cases {
            assert_synthetic_rejected(config, expected)?;
        }
        Ok(())
    }

    #[test]
    fn synthetic_llama_moe_tensor_contract_fails_closed() -> Result<()> {
        let sparse = SyntheticLlamaConfig {
            expert_count: Some(3),
            expert_used_count: Some(2),
            ..SyntheticLlamaConfig::default()
        };
        let cases = [
            (
                synthetic_llama_gguf_with(Some("blk.0.ffn_gate_inp.weight"), sparse)?,
                "ffn_gate_inp.weight",
            ),
            (
                synthetic_llama_gguf_with(Some("blk.0.ffn_up_exps.weight"), sparse)?,
                "ffn_up_exps.weight",
            ),
            (
                synthetic_llama_gguf_with(
                    None,
                    SyntheticLlamaConfig {
                        wrong_shape: Some(("blk.0.ffn_gate_inp.weight", &[4, 2])),
                        ..sparse
                    },
                )?,
                "expected [3, 4]",
            ),
            (
                synthetic_llama_gguf_with(
                    None,
                    SyntheticLlamaConfig {
                        wrong_shape: Some(("blk.0.ffn_up_exps.weight", &[4, 5, 3])),
                        ..sparse
                    },
                )?,
                "expected [3, 6, 4]",
            ),
            (
                synthetic_llama_gguf_with(
                    None,
                    SyntheticLlamaConfig {
                        extra_tensor: Some(("blk.0.ffn_gate.weight", &[4, 6])),
                        ..sparse
                    },
                )?,
                "unsupported dense tensor `ffn_gate.weight`",
            ),
            (
                synthetic_llama_gguf_with(
                    None,
                    SyntheticLlamaConfig {
                        extra_bias: Some("blk.0.ffn_gate.bias"),
                        ..sparse
                    },
                )?,
                "does not support FFN bias tensor",
            ),
            (
                synthetic_llama_gguf_with(
                    None,
                    SyntheticLlamaConfig {
                        extra_tensor: Some(("blk.0.ffn_gate_shexp.weight", &[4, 6])),
                        ..sparse
                    },
                )?,
                "shared experts are not supported",
            ),
            (
                synthetic_llama_gguf_with(
                    None,
                    SyntheticLlamaConfig {
                        extra_tensor: Some(("blk.0.ffn_gate_up_exps.weight", &[4, 6, 3])),
                        ..sparse
                    },
                )?,
                "unsupported expert tensor",
            ),
            (
                synthetic_llama_gguf_with(
                    None,
                    SyntheticLlamaConfig {
                        extra_tensor: Some(("blk.0.ffn_gate_inp.weight", &[4, 3])),
                        ..SyntheticLlamaConfig::default()
                    },
                )?,
                "without a positive expert_count",
            ),
        ];
        for (file, expected) in cases {
            let error = GgufModel::open(file.path()).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "expected `{expected}` in error, got `{error}`"
            );
        }
        Ok(())
    }

    #[allow(clippy::large_types_passed_by_value)] // Copy fixture values keep table-driven call sites concise.
    fn assert_synthetic_rejected(config: SyntheticLlamaConfig, expected: &str) -> Result<()> {
        let file = synthetic_llama_gguf_with(None, config)?;
        let error = GgufModel::open(file.path()).unwrap_err().to_string();
        assert!(
            error.contains(expected),
            "expected `{expected}` in error, got `{error}`"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires MACLLM_GGUF_MODEL pointing at the fixed SmolLM2 F16 GGUF"]
    fn real_smollm2_builds_standalone_runtime_metadata_and_tokenizer() -> Result<()> {
        let path = PathBuf::from(std::env::var("MACLLM_GGUF_MODEL")?);
        let model = GgufModel::open(path)?;
        assert_eq!(model.file().metadata().len(), 33);
        assert_eq!(model.file().tensors().len(), 272);
        assert_eq!(model.file().data_offset(), 1_785_664);
        assert_eq!(model.spec().runtime.vocab_size, 49_152);
        assert_eq!(model.spec().num_layers, 30);
        assert_eq!(model.spec().head_dim, 64);
        assert_eq!(model.spec().runtime.stop_token_ids, vec![2]);
        assert!(model.spec().tied_embeddings);

        let adapter = model.adapter()?;
        let rendered = adapter.format_chat(None, "Hello")?;
        let ids = model
            .tokenizer()
            .encode(rendered, false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(
            ids.get_ids(),
            &[
                1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018,
                411, 407, 19712, 8182, 2, 198, 1, 4093, 198, 19556, 2, 198, 1, 520, 9531, 198,
            ]
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires MACLLM_QWEN_GGUF_MODEL pointing at the pinned Qwen2.5 GGUF file"]
    fn real_qwen2_5_builds_standalone_runtime_metadata_and_adapter() -> Result<()> {
        let path = PathBuf::from(std::env::var("MACLLM_QWEN_GGUF_MODEL")?);
        let model = GgufModel::open(path)?;
        assert_eq!(model.spec().architecture, "qwen2");
        assert!(model.file().metadata().get("qwen2.vocab_size").is_none());
        assert_eq!(model.spec().runtime.vocab_size, 151_936);
        assert_eq!(model.spec().num_layers, 24);
        assert_eq!(model.spec().hidden_size, 896);
        assert_eq!(model.spec().num_query_heads, 14);
        assert_eq!(model.spec().num_kv_heads, 2);
        assert_eq!(model.spec().head_dim, 64);
        assert_eq!(model.spec().runtime.stop_token_ids, vec![151_645, 151_643]);
        assert_eq!(
            model.file().tensor("token_embd.weight").unwrap().shape,
            [151_936, 896]
        );
        assert_eq!(model.backend_config().llama_qk, None);
        assert_eq!(
            model.chat_template.as_deref(),
            Some(crate::models::qwen2::QWEN2_5_CHAT_TEMPLATE)
        );
        assert_eq!(
            model.adapter()?.format_chat(None, "Hello")?,
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires MACLLM_QWEN3_GGUF_MODEL pointing at the pinned Qwen3 0.6B Q8_0 GGUF file"]
    fn real_qwen3_0_6b_loads_explicit_attention_geometry_tokenizer_and_chat() -> Result<()> {
        let path = PathBuf::from(std::env::var("MACLLM_QWEN3_GGUF_MODEL")?);
        let model = GgufModel::open(path)?;
        let spec = model.spec();
        assert_eq!(model.file().metadata().len(), 28);
        assert_eq!(model.file().tensors().len(), 310);
        assert_eq!(model.file().data_offset(), 5_951_136);
        assert_eq!(spec.architecture, "qwen3");
        assert_eq!(spec.runtime.vocab_size, 151_936);
        assert_eq!(spec.runtime.max_context, 40_960);
        assert_eq!(spec.runtime.stop_token_ids, [151_645, 151_643]);
        assert_eq!(spec.hidden_size, 1_024);
        assert_eq!(spec.intermediate_size, 3_072);
        assert_eq!(spec.num_layers, 28);
        assert_eq!(spec.num_query_heads, 16);
        assert_eq!(spec.num_kv_heads, 8);
        assert_eq!(spec.head_dim, 128);
        assert_eq!(spec.num_query_heads * spec.head_dim, 2_048);
        assert!(
            spec.layer_specs
                .iter()
                .all(|layer| layer.attention().is_some_and(|attention| attention.qk_norm))
        );
        assert!(spec.layer_specs.iter().all(|layer| {
            layer.attention().is_some_and(|attention| {
                attention
                    .rope()
                    .is_some_and(|rope| (rope.theta - 1_000_000.0).abs() < f64::EPSILON)
            })
        }));
        assert!(!spec.projection_bias.query);
        assert!(!spec.projection_bias.key);
        assert!(!spec.projection_bias.value);
        assert!(!spec.projection_bias.output);
        assert!(spec.tied_embeddings);
        assert!(model.file().tensor("output.weight").is_none());
        assert_eq!(
            model.file().tensor("blk.0.attn_q.weight").unwrap().shape,
            [2_048, 1_024]
        );
        assert_eq!(
            model
                .file()
                .tensor("blk.0.attn_q_norm.weight")
                .unwrap()
                .shape,
            [128]
        );
        assert_eq!(model.backend_config().llama_qk, None);

        let rendered = model.adapter()?.format_chat(None, "Hello")?;
        assert_eq!(
            rendered,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        let ids = model
            .tokenizer()
            .encode(rendered, false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(
            ids.get_ids(),
            [151_644, 872, 198, 9_707, 151_645, 198, 151_644, 77_091, 198]
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires MACLLM_QWEN35_GGUF_MODEL pointing at the pinned official Qwen3.5-0.8B Q4_0 GGUF"]
    fn real_qwen35_0_8b_loads_text_trunk_geometry_and_tensor_contract() -> Result<()> {
        let path = PathBuf::from(std::env::var("MACLLM_QWEN35_GGUF_MODEL")?);
        let model = GgufModel::open(path)?;
        let spec = model.spec();
        assert_eq!(model.file().tensors().len(), 320);
        assert_eq!(spec.architecture, "qwen35");
        assert_eq!(spec.runtime.vocab_size, 248_320);
        assert_eq!(spec.runtime.max_context, 262_144);
        assert!(spec.runtime.stop_token_ids.contains(&248_044));
        assert_eq!(spec.hidden_size, 1_024);
        assert_eq!(spec.intermediate_size, 3_584);
        assert_eq!(spec.num_layers, 24);
        assert_eq!(spec.num_query_heads, 8);
        assert_eq!(spec.num_kv_heads, 2);
        assert_eq!(spec.head_dim, 256);
        assert!(spec.tied_embeddings);
        assert!(model.file().tensor("output.weight").is_none());
        assert!(model.file().tensor("blk.24.attn_norm.weight").is_none());

        for (layer, layer_spec) in spec.layer_specs.iter().enumerate() {
            if (layer + 1) % 4 == 0 {
                let attention = layer_spec
                    .attention()
                    .with_context(|| format!("Qwen3.5 layer {layer} is not full attention"))?;
                assert!(attention.qk_norm);
                assert!(attention.output_gate);
                let rope = attention
                    .rope()
                    .context("Qwen3.5 attention does not use RoPE")?;
                assert_eq!(rope.rotary_dim, 64);
                assert!((rope.theta - 10_000_000.0).abs() < f64::EPSILON);
            } else {
                assert!(matches!(
                    layer_spec.mixer,
                    SequenceMixerSpec::GatedDeltaNet(GatedDeltaNetSpec {
                        convolution_kernel_size: 4,
                        num_key_heads: 16,
                        num_value_heads: 16,
                        key_head_dim: 128,
                        value_head_dim: 128,
                        ..
                    })
                ));
            }
        }
        assert_eq!(
            model.file().tensor("blk.0.attn_qkv.weight").unwrap().shape,
            [6_144, 1_024]
        );
        assert_eq!(
            model
                .file()
                .tensor("blk.0.ssm_conv1d.weight")
                .unwrap()
                .shape,
            [6_144, 4]
        );
        assert_eq!(
            model.file().tensor("blk.3.attn_q.weight").unwrap().shape,
            [4_096, 1_024]
        );
        assert_eq!(model.backend_config().llama_qk, None);
        assert_eq!(model.adapter()?.family(), "qwen35");
        assert!(model.adapter()?.format_chat(None, "Hello").is_err());
        Ok(())
    }

    #[test]
    fn synthetic_gemma_reuses_the_shared_full_causal_decoder() -> Result<()> {
        let file = synthetic_gemma_gguf(None)?;
        let model = GgufModel::open(file.path())?;
        let spec = model.spec();
        assert_eq!(spec.architecture, "gemma");
        assert_eq!(spec.runtime.vocab_size, 5);
        assert_eq!(spec.runtime.max_context, 16);
        assert_eq!(spec.runtime.stop_token_ids, [0, 1]);
        assert_eq!(spec.hidden_size, 4);
        assert_eq!(spec.intermediate_size, 6);
        assert_eq!(spec.num_layers, 2);
        assert_eq!(spec.num_query_heads, 2);
        assert_eq!(spec.num_kv_heads, 1);
        assert_eq!(spec.head_dim, 2);
        assert!((spec.embedding_scale - 2.0).abs() < f64::EPSILON);
        assert_eq!(spec.final_logit_softcap, None);
        assert!(spec.tied_embeddings);
        for layer in &spec.layer_specs {
            assert_eq!(layer.activation, Activation::GeluTanh);
            assert_eq!(layer.norm_layout, NormLayout::PreNorm);
            let attention = layer.attention().context("Gemma layer is not attention")?;
            assert_eq!(attention.kind, AttentionKind::FullCausal);
            assert!(!attention.qk_norm);
            assert_eq!(attention.logit_softcap, None);
            let rope = attention.rope().context("Gemma layer does not use RoPE")?;
            assert_eq!(rope.rotary_dim, 2);
            assert!((rope.theta - 10_000.0).abs() < f64::EPSILON);
            assert!((attention.scale - 2.0_f64.sqrt().recip()).abs() < f64::EPSILON);
        }
        assert_eq!(model.backend_config().llama_qk, None);
        assert_eq!(model.adapter()?.family(), "gemma");

        // Exercise every canonical Gemma tensor path through the shared
        // decoder, including tied embeddings and paged full-attention state.
        // SAFETY: the NamedTempFile remains alive and immutable for the model.
        let backend = unsafe { model.file().backend(model.backend_config()) }?;
        let source = GgufWeightSource::new(backend, DType::F32, Device::Cpu);
        let runtime = model.adapter()?.build(&source)?;
        let mut session = runtime.new_session()?;
        assert_eq!(
            runtime
                .forward_tokens(session.as_mut(), &[2, 3], 0)?
                .dims2()?,
            (1, 5)
        );
        assert_eq!(session.cache_len(), 2);
        Ok(())
    }

    #[test]
    fn synthetic_gemma_consumes_optional_rope_metadata() -> Result<()> {
        let file = synthetic_gemma_gguf(Some("custom-rope"))?;
        let model = GgufModel::open(file.path())?;
        for layer in &model.spec().layer_specs {
            let rope = layer
                .attention()
                .context("Gemma layer is not attention")?
                .rope()
                .context("Gemma layer does not use RoPE")?;
            assert_eq!(rope.rotary_dim, 2);
            assert!((rope.theta - 20_000.0).abs() < f64::EPSILON);
        }
        Ok(())
    }

    #[test]
    fn synthetic_gemma_fails_closed_for_unmodeled_contracts() -> Result<()> {
        for (variant, expected) in [
            ("metadata", "metadata `gemma.unmodeled` is not supported"),
            ("noncausal", "non-causal GGUF Gemma attention"),
            (
                "bad-value-width",
                "different GGUF Gemma key/value head widths",
            ),
            ("extra-output", "does not support `output.weight`"),
            (
                "extra-post-norm",
                "does not support `post_attention_norm.weight`",
            ),
            ("extra-q-norm", "does not support `attn_q_norm.weight`"),
            ("extra-bias", "does not support `attn_q.bias`"),
        ] {
            let file = synthetic_gemma_gguf(Some(variant))?;
            let error = GgufModel::open(file.path())
                .expect_err("unmodeled Gemma contract must fail closed")
                .to_string();
            assert!(error.contains(expected), "{variant}: {error}");
        }
        Ok(())
    }

    #[test]
    fn synthetic_gemma2_reuses_sandwich_decoder_and_alternating_attention() -> Result<()> {
        let file = synthetic_gemma2_gguf(None)?;
        let model = GgufModel::open(file.path())?;
        let spec = model.spec();
        assert_eq!(spec.architecture, "gemma2");
        assert_eq!(spec.runtime.vocab_size, 5);
        assert_eq!(spec.runtime.max_context, 16);
        assert_eq!(spec.runtime.stop_token_ids, [0, 1]);
        assert_eq!(spec.hidden_size, 4);
        assert_eq!(spec.intermediate_size, 6);
        assert_eq!(spec.num_layers, 2);
        assert_eq!(spec.num_query_heads, 2);
        assert_eq!(spec.num_kv_heads, 1);
        assert_eq!(spec.head_dim, 2);
        assert!((spec.embedding_scale - 2.0).abs() < f64::EPSILON);
        assert_eq!(spec.final_logit_softcap, Some(30.0));
        assert!(spec.tied_embeddings);
        for (index, layer) in spec.layer_specs.iter().enumerate() {
            assert_eq!(layer.activation, Activation::GeluTanh);
            assert_eq!(layer.norm_layout, NormLayout::Sandwich);
            let attention = layer
                .attention()
                .context("Gemma 2 layer is not attention")?;
            assert!(!attention.qk_norm);
            assert_eq!(attention.logit_softcap, Some(50.0));
            assert!((attention.scale - 2.0_f64.sqrt().recip()).abs() < f64::EPSILON);
            assert_eq!(
                attention.kind,
                if index == 0 {
                    AttentionKind::SlidingCausal { window: 4 }
                } else {
                    AttentionKind::FullCausal
                }
            );
        }
        assert_eq!(model.backend_config().llama_qk, None);
        assert_eq!(model.adapter()?.family(), "gemma2");

        // Exercise the canonical GGUF names through the shared decoder rather
        // than stopping at metadata parsing.
        // SAFETY: the NamedTempFile remains alive and immutable for the model.
        let backend = unsafe { model.file().backend(model.backend_config()) }?;
        let source = GgufWeightSource::new(backend, DType::F32, Device::Cpu);
        let runtime = model.adapter()?.build(&source)?;
        let mut session = runtime.new_session()?;
        assert_eq!(
            runtime
                .forward_tokens(session.as_mut(), &[2, 3], 0)?
                .dims2()?,
            (1, 5)
        );
        assert_eq!(session.cache_len(), 2);
        Ok(())
    }

    #[test]
    fn synthetic_gemma2_rejects_unmodeled_metadata_and_missing_sandwich_norm() -> Result<()> {
        let unknown = synthetic_gemma2_gguf(Some("metadata"))?;
        assert!(
            GgufModel::open(unknown.path())
                .expect_err("unknown Gemma 2 metadata must fail closed")
                .to_string()
                .contains("metadata `gemma2.unmodeled` is not supported")
        );
        let missing = synthetic_gemma2_gguf(Some("blk.0.post_ffw_norm.weight"))?;
        assert!(
            GgufModel::open(missing.path())
                .expect_err("missing Gemma 2 post-FFN norm must fail closed")
                .to_string()
                .contains("blk.0.post_ffw_norm.weight")
        );
        Ok(())
    }

    #[test]
    fn synthetic_gemma2_matches_scalar_swa_pattern_semantics() -> Result<()> {
        for (variant, expected) in [
            (
                "pattern-0",
                [
                    AttentionKind::SlidingCausal { window: 4 },
                    AttentionKind::SlidingCausal { window: 4 },
                ],
            ),
            (
                "pattern-1",
                [AttentionKind::FullCausal, AttentionKind::FullCausal],
            ),
            (
                "pattern-2",
                [
                    AttentionKind::SlidingCausal { window: 4 },
                    AttentionKind::FullCausal,
                ],
            ),
        ] {
            let file = synthetic_gemma2_gguf(Some(variant))?;
            let model = GgufModel::open(file.path())?;
            let actual = model
                .spec()
                .layer_specs
                .iter()
                .map(|layer| layer.attention().expect("Gemma 2 uses attention").kind)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{variant}");
        }
        Ok(())
    }

    #[test]
    fn synthetic_gemma2_rejects_untied_head_and_qk_norm_tensors() -> Result<()> {
        for (variant, expected) in [
            ("extra-output", "does not support `output.weight`"),
            ("extra-q-norm", "does not support `attn_q_norm.weight`"),
            ("extra-k-norm", "does not support `attn_k_norm.weight`"),
        ] {
            let file = synthetic_gemma2_gguf(Some(variant))?;
            let error = GgufModel::open(file.path())
                .expect_err("unmodeled Gemma 2 tensor must fail closed")
                .to_string();
            assert!(error.contains(expected), "{variant}: {error}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires MACLLM_GEMMA2_GGUF_MODEL pointing at a Gemma 2 2B instruction GGUF file"]
    fn real_gemma2_2b_loads_reference_geometry_tokenizer_and_chat() -> Result<()> {
        let path = PathBuf::from(std::env::var("MACLLM_GEMMA2_GGUF_MODEL")?);
        let model = GgufModel::open(path)?;
        let spec = model.spec();
        assert_eq!(model.file().metadata().len(), 39);
        assert_eq!(model.file().tensors().len(), 288);
        assert_eq!(model.file().data_offset(), 6_046_560);
        assert_eq!(spec.architecture, "gemma2");
        assert_eq!(spec.runtime.vocab_size, 256_000);
        assert_eq!(spec.runtime.max_context, 8_192);
        assert_eq!(spec.runtime.stop_token_ids, [1, 107]);
        assert_eq!(spec.hidden_size, 2_304);
        assert_eq!(spec.intermediate_size, 9_216);
        assert_eq!(spec.num_layers, 26);
        assert_eq!(spec.num_query_heads, 8);
        assert_eq!(spec.num_kv_heads, 4);
        assert_eq!(spec.head_dim, 256);
        assert_eq!(spec.final_logit_softcap, Some(30.0));
        assert!(spec.tied_embeddings);
        assert_eq!(model.backend_config().llama_qk, None);
        for (index, layer) in spec.layer_specs.iter().enumerate() {
            let attention = layer
                .attention()
                .context("Gemma 2 layer is not attention")?;
            assert_eq!(
                attention.kind,
                if index % 2 == 0 {
                    AttentionKind::SlidingCausal { window: 4_096 }
                } else {
                    AttentionKind::FullCausal
                }
            );
            assert_eq!(attention.logit_softcap, Some(50.0));
            assert!(!attention.qk_norm);
            assert_eq!(layer.norm_layout, NormLayout::Sandwich);
            assert_eq!(layer.activation, Activation::GeluTanh);
        }
        assert_eq!(
            model.adapter()?.format_chat(None, " Hello ")?,
            "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        );
        assert!(
            model
                .adapter()?
                .format_chat(Some("System"), "Hello")
                .is_err()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires MACLLM_GEMMA3_GGUF_MODEL pointing at the pinned Gemma 3 270M GGUF file"]
    fn real_gemma3_270m_loads_layer_capabilities_tokenizer_and_chat() -> Result<()> {
        let path = PathBuf::from(std::env::var("MACLLM_GEMMA3_GGUF_MODEL")?);
        let model = GgufModel::open(path)?;
        let spec = model.spec();
        assert_eq!(model.file().metadata().len(), 39);
        assert_eq!(model.file().tensors().len(), 236);
        assert_eq!(model.file().data_offset(), 6_527_040);
        assert_eq!(spec.architecture, "gemma3");
        assert_eq!(spec.runtime.vocab_size, 262_144);
        assert_eq!(spec.runtime.max_context, 32_768);
        assert_eq!(spec.runtime.stop_token_ids, [1, 106]);
        assert_eq!(spec.hidden_size, 640);
        assert_eq!(spec.intermediate_size, 2_048);
        assert_eq!(spec.num_layers, 18);
        assert_eq!(spec.num_query_heads, 4);
        assert_eq!(spec.num_kv_heads, 1);
        assert_eq!(spec.head_dim, 256);
        assert!((spec.embedding_scale - 640.0_f64.sqrt()).abs() < f64::EPSILON);
        assert!(spec.tied_embeddings);
        for (index, layer) in spec.layer_specs.iter().enumerate() {
            assert_eq!(layer.activation, Activation::GeluTanh);
            assert_eq!(layer.norm_layout, NormLayout::Sandwich);
            let attention = layer.attention().expect("Gemma layer uses attention");
            assert!(attention.qk_norm);
            if index % 6 < 5 {
                assert_eq!(attention.kind, AttentionKind::SlidingCausal { window: 512 });
                assert!(
                    (attention.rope().expect("Gemma 3 attention uses RoPE").theta - 10_000.0).abs()
                        < f64::EPSILON
                );
            } else {
                assert_eq!(attention.kind, AttentionKind::FullCausal);
                assert!(
                    (attention.rope().expect("Gemma 3 attention uses RoPE").theta - 1_000_000.0)
                        .abs()
                        < f64::EPSILON
                );
            }
        }
        assert_eq!(model.backend_config().llama_qk, None);

        let tokenizer = model.tokenizer();
        for (text, add_special_tokens, expected) in [
            ("hello world", true, &[2, 23_391, 1_902][..]),
            ("hello world", false, &[23_391, 1_902][..]),
            ("café ☃", true, &[2, 123_125, 236_859, 236_743, 246_690][..]),
        ] {
            let encoding = tokenizer
                .encode(text, add_special_tokens)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            assert_eq!(encoding.get_ids(), expected, "token IDs for {text:?}");
        }
        assert_eq!(
            model
                .adapter()?
                .format_chat(Some(" Be terse. "), " Hello ")?,
            "<bos><start_of_turn>user\n Be terse. \n\nHello<end_of_turn>\n<start_of_turn>model\n"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires MACLLM_LLAMA3_GGUF_MODEL pointing at the pinned Llama 3.2 GGUF file"]
    fn real_llama3_2_loads_scaled_runtime_tokenizer_and_chat_contract() -> Result<()> {
        let path = PathBuf::from(std::env::var("MACLLM_LLAMA3_GGUF_MODEL")?);
        let model = GgufModel::open(path)?;
        assert_eq!(model.file().metadata().len(), 35);
        assert_eq!(model.file().tensors().len(), 147);
        assert_eq!(model.file().data_offset(), 7_831_552);
        assert_eq!(model.spec().runtime.vocab_size, 128_256);
        assert_eq!(model.spec().runtime.max_context, 131_072);
        assert_eq!(
            model.spec().runtime.stop_token_ids,
            [128_001, 128_008, 128_009]
        );
        assert_eq!(model.spec().hidden_size, 2_048);
        assert_eq!(model.spec().num_layers, 16);
        assert_eq!(model.spec().num_query_heads, 32);
        assert_eq!(model.spec().num_kv_heads, 8);
        assert_eq!(model.spec().head_dim, 64);
        let factors = model.spec().layer_specs[0]
            .attention()
            .context("real Llama layer is not attention")?
            .rope()
            .context("real Llama layer does not use RoPE")?
            .frequency_factors
            .as_deref()
            .context("real Llama 3.2 RoPE factors are absent")?;
        assert_eq!(factors.len(), 32);
        for (actual, expected) in factors[15..18]
            .iter()
            .zip([1.651_329_f32, 3.292_263, 9.666_731])
        {
            assert!((actual - expected).abs() < 1e-5);
        }

        let tokenizer = model.tokenizer();
        let raw = tokenizer
            .encode("The capital of France is", false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(raw.get_ids(), [791, 6_864, 315, 9_822, 374]);
        let with_bos = tokenizer
            .encode("The capital of France is", true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(with_bos.get_ids(), [128_000, 791, 6_864, 315, 9_822, 374]);
        let unicode = tokenizer
            .encode("hello 1234 café 👋\n", false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(
            unicode.get_ids(),
            [15_339, 220, 4_513, 19, 53_050, 62_904, 233, 198]
        );
        assert_eq!(
            unicode.get_offsets(),
            [
                (0, 5),
                (5, 6),
                (6, 9),
                (9, 10),
                (10, 16),
                (16, 21),
                (17, 21),
                (21, 22),
            ]
        );

        let rendered = model.adapter()?.format_chat(None, "Hello")?;
        let chat = tokenizer
            .encode(rendered, false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(
            chat.get_ids(),
            [
                128_000, 128_006, 9_125, 128_007, 271, 38_766, 1_303, 33_025, 2_696, 25, 6_790,
                220, 2_366, 18, 198, 15_724, 2_696, 25, 220, 1_627, 10_263, 220, 2_366, 19, 271,
                128_009, 128_006, 882, 128_007, 271, 9_906, 128_009, 128_006, 78_191, 128_007, 271,
            ]
        );
        Ok(())
    }

    #[derive(Debug, Clone, Copy)]
    struct SyntheticLlamaConfig {
        hidden_size: usize,
        query_heads: usize,
        kv_heads: usize,
        key_length: Option<usize>,
        value_length: Option<usize>,
        rope_dimension: Option<usize>,
        block_count: usize,
        context_length: usize,
        causal: Option<bool>,
        sliding_window: Option<usize>,
        sliding_window_pattern: bool,
        attention_scale: Option<f32>,
        expert_count: Option<usize>,
        expert_used_count: Option<usize>,
        expert_weights_scale: Option<f32>,
        split_metadata: bool,
        unsupported_llama_key: Option<&'static str>,
        rope_theta: f32,
        rope_factors: Option<&'static [f32]>,
        norm_epsilon: f32,
        projection_biases: bool,
        extra_bias: Option<&'static str>,
        extra_tensor: Option<(&'static str, &'static [usize])>,
        wrong_shape: Option<(&'static str, &'static [usize])>,
        qk_norm: bool,
    }

    impl Default for SyntheticLlamaConfig {
        fn default() -> Self {
            Self {
                hidden_size: 4,
                query_heads: 2,
                kv_heads: 1,
                key_length: None,
                value_length: None,
                rope_dimension: None,
                block_count: 1,
                context_length: 16,
                causal: None,
                sliding_window: None,
                sliding_window_pattern: false,
                attention_scale: None,
                expert_count: None,
                expert_used_count: None,
                expert_weights_scale: None,
                split_metadata: false,
                unsupported_llama_key: None,
                rope_theta: 10_000.0,
                rope_factors: None,
                norm_epsilon: 1e-5,
                projection_biases: true,
                extra_bias: None,
                extra_tensor: None,
                wrong_shape: None,
                qk_norm: false,
            }
        }
    }

    fn synthetic_llama_gguf(omit: Option<&str>) -> Result<tempfile::NamedTempFile> {
        synthetic_llama_gguf_with(omit, SyntheticLlamaConfig::default())
    }

    /// Reuse the structurally identical tiny decoder fixture while replacing
    /// only family metadata. `llama` and `qwen2` have the same byte length, so
    /// the GGUF index remains valid and the parser still exercises real disk
    /// serialization rather than a mocked metadata object.
    #[allow(clippy::large_types_passed_by_value)] // Copy fixture values are consumed while building one temporary file.
    fn synthetic_qwen2_gguf(
        omit: Option<&str>,
        config: SyntheticLlamaConfig,
        declared_vocab_size: Option<u32>,
    ) -> Result<tempfile::NamedTempFile> {
        const VOCAB_KEY: &[u8] = b"qwen2.vocab_size";

        let llama = synthetic_llama_gguf_with(omit, config)?;
        let mut bytes = fs::read(llama.path())?;
        replace_all_same_length(&mut bytes, b"llama", b"qwen2");

        let vocab_key_offset = bytes
            .windows(VOCAB_KEY.len())
            .position(|window| window == VOCAB_KEY)
            .context("synthetic qwen2 vocab key is missing")?;
        if let Some(declared_vocab_size) = declared_vocab_size {
            let value_type_offset = vocab_key_offset + VOCAB_KEY.len();
            ensure!(
                bytes.get(value_type_offset..value_type_offset + 4) == Some(&4_u32.to_le_bytes()),
                "synthetic qwen2 vocab metadata is not UINT32"
            );
            bytes[value_type_offset + 4..value_type_offset + 8]
                .copy_from_slice(&declared_vocab_size.to_le_bytes());
        } else {
            bytes[vocab_key_offset..vocab_key_offset + VOCAB_KEY.len()]
                .copy_from_slice(b"general.vocab_sz");
        }

        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(&bytes)?;
        file.flush()?;
        Ok(file)
    }

    #[allow(clippy::large_types_passed_by_value)] // Copy fixture values are consumed while building one temporary file.
    fn synthetic_qwen3_gguf(
        omit: Option<&str>,
        config: SyntheticLlamaConfig,
        declared_vocab_size: Option<u32>,
    ) -> Result<tempfile::NamedTempFile> {
        const VOCAB_KEY: &[u8] = b"qwen3.vocab_size";

        let llama = synthetic_llama_gguf_with(omit, config)?;
        let mut bytes = fs::read(llama.path())?;
        replace_all_same_length(&mut bytes, b"llama", b"qwen3");

        let vocab_key_offset = bytes
            .windows(VOCAB_KEY.len())
            .position(|window| window == VOCAB_KEY)
            .context("synthetic qwen3 vocab key is missing")?;
        if let Some(declared_vocab_size) = declared_vocab_size {
            let value_type_offset = vocab_key_offset + VOCAB_KEY.len();
            ensure!(
                bytes.get(value_type_offset..value_type_offset + 4) == Some(&4_u32.to_le_bytes()),
                "synthetic qwen3 vocab metadata is not UINT32"
            );
            bytes[value_type_offset + 4..value_type_offset + 8]
                .copy_from_slice(&declared_vocab_size.to_le_bytes());
        } else {
            bytes[vocab_key_offset..vocab_key_offset + VOCAB_KEY.len()]
                .copy_from_slice(b"general.vocab_sz");
        }

        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(&bytes)?;
        file.flush()?;
        Ok(file)
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct SyntheticSmolLm3Config {
        unsupported_metadata: Option<&'static str>,
        extra_tensor: Option<(&'static str, &'static [usize])>,
    }

    #[allow(clippy::too_many_lines)] // The 36-layer schedule is the fixture's architectural proof.
    fn synthetic_smollm3_gguf(config: SyntheticSmolLm3Config) -> Result<tempfile::NamedTempFile> {
        const HIDDEN: usize = 4;
        const INTERMEDIATE: usize = 6;
        const VOCAB: usize = 4;
        const LAYERS: usize = 36;
        const QUERY_WIDTH: usize = 4;
        const KV_WIDTH: usize = 2;

        let mut metadata = vec![
            ("general.architecture", GgufValue::String("smollm3".into())),
            ("smollm3.block_count", GgufValue::U32(LAYERS as u32)),
            ("smollm3.context_length", GgufValue::U32(16)),
            ("smollm3.embedding_length", GgufValue::U32(HIDDEN as u32)),
            (
                "smollm3.feed_forward_length",
                GgufValue::U32(INTERMEDIATE as u32),
            ),
            ("smollm3.attention.head_count", GgufValue::U32(2)),
            ("smollm3.attention.head_count_kv", GgufValue::U32(1)),
            ("smollm3.attention.key_length", GgufValue::U32(2)),
            ("smollm3.attention.value_length", GgufValue::U32(2)),
            ("smollm3.rope.dimension_count", GgufValue::U32(2)),
            ("smollm3.rope.freq_base", GgufValue::F32(5_000_000.0)),
            (
                "smollm3.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-5),
            ),
            ("smollm3.vocab_size", GgufValue::U32(VOCAB as u32)),
            ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
            ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::String("smollm".into())),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: ["<|endoftext|>", "<|im_end|>", "a", "b"]
                        .into_iter()
                        .map(|value| GgufValue::String(value.into()))
                        .collect(),
                },
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array {
                    element_type: GgufValueType::I32,
                    values: [3, 3, 1, 1].into_iter().map(GgufValue::I32).collect(),
                },
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: Vec::new(),
                },
            ),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(0)),
        ];
        if let Some(key) = config.unsupported_metadata {
            metadata.push((key, GgufValue::String("linear".into())));
        }

        // GGUF dimensions are stored in the opposite order from Candle's
        // logical tensor view.
        let mut tensors = vec![
            ("token_embd.weight".to_owned(), vec![HIDDEN, VOCAB]),
            ("output_norm.weight".to_owned(), vec![HIDDEN]),
        ];
        for layer in 0..LAYERS {
            let prefix = format!("blk.{layer}");
            tensors.extend([
                (format!("{prefix}.attn_norm.weight"), vec![HIDDEN]),
                (format!("{prefix}.ffn_norm.weight"), vec![HIDDEN]),
                (format!("{prefix}.attn_q.weight"), vec![HIDDEN, QUERY_WIDTH]),
                (format!("{prefix}.attn_k.weight"), vec![HIDDEN, KV_WIDTH]),
                (format!("{prefix}.attn_v.weight"), vec![HIDDEN, KV_WIDTH]),
                (
                    format!("{prefix}.attn_output.weight"),
                    vec![QUERY_WIDTH, HIDDEN],
                ),
                (
                    format!("{prefix}.ffn_gate.weight"),
                    vec![HIDDEN, INTERMEDIATE],
                ),
                (
                    format!("{prefix}.ffn_up.weight"),
                    vec![HIDDEN, INTERMEDIATE],
                ),
                (
                    format!("{prefix}.ffn_down.weight"),
                    vec![INTERMEDIATE, HIDDEN],
                ),
            ]);
        }
        if let Some((name, dimensions)) = config.extra_tensor {
            tensors.push((name.to_owned(), dimensions.to_vec()));
        }
        let tensor_refs = tensors
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.clone()))
            .collect();
        write_synthetic_gguf(metadata, tensor_refs)
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct SyntheticFalconH1Config {
        omit: Option<&'static str>,
        wrong_shape: Option<(&'static str, &'static [usize])>,
        extra_tensor: Option<(&'static str, &'static [usize])>,
    }

    #[allow(clippy::too_many_lines)] // The complete hybrid tensor inventory is the fixture's purpose.
    fn synthetic_falcon_h1_gguf(
        config: SyntheticFalconH1Config,
    ) -> Result<tempfile::NamedTempFile> {
        const HIDDEN: usize = 4;
        const INTERMEDIATE: usize = 6;
        const VOCAB: usize = 5;
        const QUERY_WIDTH: usize = 4;
        const KV_WIDTH: usize = 2;
        const SSM_INNER: usize = 4;
        const SSM_HEADS: usize = 2;
        const SSM_GROUPS: usize = 1;
        const SSM_STATE: usize = 2;
        const SSM_KERNEL: usize = 3;
        const SSM_CONVOLUTION_WIDTH: usize = SSM_INNER + 2 * SSM_GROUPS * SSM_STATE;
        const SSM_PROJECTION_WIDTH: usize = SSM_INNER + SSM_CONVOLUTION_WIDTH + SSM_HEADS;

        let metadata = vec![
            (
                "general.architecture",
                GgufValue::String("falcon-h1".into()),
            ),
            ("falcon-h1.block_count", GgufValue::U32(1)),
            ("falcon-h1.context_length", GgufValue::U32(16)),
            ("falcon-h1.embedding_length", GgufValue::U32(HIDDEN as u32)),
            (
                "falcon-h1.feed_forward_length",
                GgufValue::U32(INTERMEDIATE as u32),
            ),
            ("falcon-h1.attention.head_count", GgufValue::U32(2)),
            ("falcon-h1.attention.head_count_kv", GgufValue::U32(1)),
            ("falcon-h1.attention.key_length", GgufValue::U32(2)),
            ("falcon-h1.attention.value_length", GgufValue::U32(2)),
            (
                "falcon-h1.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-5),
            ),
            ("falcon-h1.rope.freq_base", GgufValue::F32(1.0e11)),
            (
                "falcon-h1.ssm.conv_kernel",
                GgufValue::U32(SSM_KERNEL as u32),
            ),
            ("falcon-h1.ssm.inner_size", GgufValue::U32(SSM_INNER as u32)),
            ("falcon-h1.ssm.state_size", GgufValue::U32(SSM_STATE as u32)),
            (
                "falcon-h1.ssm.time_step_rank",
                GgufValue::U32(SSM_HEADS as u32),
            ),
            (
                "falcon-h1.ssm.group_count",
                GgufValue::U32(SSM_GROUPS as u32),
            ),
            ("falcon-h1.vocab_size", GgufValue::U32(VOCAB as u32)),
            ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(true)),
            ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::String("falcon-h1".into())),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: ["<|begin_of_text|>", "<|end_of_text|>", "a", "1", "2"]
                        .into_iter()
                        .map(|value| GgufValue::String(value.into()))
                        .collect(),
                },
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array {
                    element_type: GgufValueType::I32,
                    values: [3, 3, 1, 1, 1].into_iter().map(GgufValue::I32).collect(),
                },
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: Vec::new(),
                },
            ),
            ("tokenizer.ggml.bos_token_id", GgufValue::U32(0)),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(1)),
        ];

        // Shapes use GGUF's on-disk dimension order; the parser reverses them.
        let mut tensors = vec![
            ("token_embd.weight", vec![HIDDEN, VOCAB]),
            ("output_norm.weight", vec![HIDDEN]),
            ("blk.0.attn_norm.weight", vec![HIDDEN]),
            ("blk.0.ffn_norm", vec![HIDDEN]),
            ("blk.0.attn_q.weight", vec![HIDDEN, QUERY_WIDTH]),
            ("blk.0.attn_k.weight", vec![HIDDEN, KV_WIDTH]),
            ("blk.0.attn_v.weight", vec![HIDDEN, KV_WIDTH]),
            ("blk.0.attn_output.weight", vec![QUERY_WIDTH, HIDDEN]),
            ("blk.0.ssm_in.weight", vec![HIDDEN, SSM_PROJECTION_WIDTH]),
            (
                "blk.0.ssm_conv1d.weight",
                vec![SSM_KERNEL, SSM_CONVOLUTION_WIDTH],
            ),
            ("blk.0.ssm_conv1d.bias", vec![SSM_CONVOLUTION_WIDTH]),
            ("blk.0.ssm_dt.bias", vec![SSM_HEADS]),
            ("blk.0.ssm_a", vec![1, SSM_HEADS]),
            ("blk.0.ssm_d", vec![1, SSM_HEADS]),
            ("blk.0.ssm_out.weight", vec![SSM_INNER, HIDDEN]),
            ("blk.0.ffn_gate.weight", vec![HIDDEN, INTERMEDIATE]),
            ("blk.0.ffn_up.weight", vec![HIDDEN, INTERMEDIATE]),
            ("blk.0.ffn_down.weight", vec![INTERMEDIATE, HIDDEN]),
        ];
        tensors.retain(|(name, _)| Some(*name) != config.omit);
        if let Some((name, dimensions)) = config.wrong_shape {
            let (_, current) = tensors
                .iter_mut()
                .find(|(candidate, _)| *candidate == name)
                .with_context(|| format!("synthetic Falcon-H1 tensor `{name}` is missing"))?;
            // Config uses runtime order for readable error fixtures.
            *current = dimensions.iter().rev().copied().collect();
        }
        if let Some((name, dimensions)) = config.extra_tensor {
            tensors.push((name, dimensions.iter().rev().copied().collect()));
        }
        write_synthetic_gguf(metadata, tensors)
    }

    #[derive(Debug, Clone, Copy)]
    struct SyntheticLfm2Config {
        kv_heads_by_layer: &'static [i32],
        omit: Option<&'static str>,
        wrong_shape: Option<(&'static str, &'static [usize])>,
        extra_tensor: Option<(&'static str, &'static [usize])>,
    }

    impl Default for SyntheticLfm2Config {
        fn default() -> Self {
            Self {
                kv_heads_by_layer: &[0, 1],
                omit: None,
                wrong_shape: None,
                extra_tensor: None,
            }
        }
    }

    #[allow(clippy::too_many_lines)] // Explicit metadata and shapes keep this hybrid fixture auditable.
    fn synthetic_lfm2_gguf(config: SyntheticLfm2Config) -> Result<tempfile::NamedTempFile> {
        const HIDDEN: usize = 4;
        const INTERMEDIATE: usize = 6;
        const VOCAB: usize = 5;
        const QUERY_WIDTH: usize = 4;
        const KV_WIDTH: usize = 2;
        let metadata = vec![
            ("general.architecture", GgufValue::String("lfm2".into())),
            ("lfm2.block_count", GgufValue::U32(2)),
            ("lfm2.context_length", GgufValue::U32(16)),
            ("lfm2.embedding_length", GgufValue::U32(HIDDEN as u32)),
            (
                "lfm2.feed_forward_length",
                GgufValue::U32(INTERMEDIATE as u32),
            ),
            ("lfm2.attention.head_count", GgufValue::U32(2)),
            (
                "lfm2.attention.head_count_kv",
                GgufValue::Array {
                    element_type: GgufValueType::I32,
                    values: config
                        .kv_heads_by_layer
                        .iter()
                        .copied()
                        .map(GgufValue::I32)
                        .collect(),
                },
            ),
            ("lfm2.attention.key_length", GgufValue::U32(2)),
            ("lfm2.attention.value_length", GgufValue::U32(2)),
            ("lfm2.rope.dimension_count", GgufValue::U32(2)),
            ("lfm2.rope.freq_base", GgufValue::F32(10_000.0)),
            (
                "lfm2.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-5),
            ),
            ("lfm2.shortconv.l_cache", GgufValue::U32(3)),
            ("lfm2.vocab_size", GgufValue::U32(VOCAB as u32)),
            ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(true)),
            ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::String("lfm2".into())),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: ["<|startoftext|>", "<|im_end|>", "a", "1", "2"]
                        .into_iter()
                        .map(|value| GgufValue::String(value.into()))
                        .collect(),
                },
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array {
                    element_type: GgufValueType::I32,
                    values: [3, 3, 1, 1, 1].into_iter().map(GgufValue::I32).collect(),
                },
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: Vec::new(),
                },
            ),
            ("tokenizer.ggml.bos_token_id", GgufValue::U32(0)),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(1)),
        ];
        // GGUF writes dimensions in the opposite order from Candle's tensor
        // view. These are the on-disk dimensions used by real LFM2 files.
        let mut tensors = vec![
            ("token_embd.weight", vec![HIDDEN, VOCAB]),
            ("blk.0.attn_norm.weight", vec![HIDDEN]),
            ("blk.0.ffn_norm.weight", vec![HIDDEN]),
            ("blk.0.shortconv.conv.weight", vec![3, HIDDEN]),
            ("blk.0.shortconv.in_proj.weight", vec![HIDDEN, HIDDEN * 3]),
            ("blk.0.shortconv.out_proj.weight", vec![HIDDEN, HIDDEN]),
            ("blk.0.ffn_gate.weight", vec![HIDDEN, INTERMEDIATE]),
            ("blk.0.ffn_up.weight", vec![HIDDEN, INTERMEDIATE]),
            ("blk.0.ffn_down.weight", vec![INTERMEDIATE, HIDDEN]),
            ("blk.1.attn_norm.weight", vec![HIDDEN]),
            ("blk.1.ffn_norm.weight", vec![HIDDEN]),
            ("blk.1.attn_q.weight", vec![HIDDEN, QUERY_WIDTH]),
            ("blk.1.attn_k.weight", vec![HIDDEN, KV_WIDTH]),
            ("blk.1.attn_v.weight", vec![HIDDEN, KV_WIDTH]),
            ("blk.1.attn_output.weight", vec![QUERY_WIDTH, HIDDEN]),
            ("blk.1.attn_q_norm.weight", vec![2]),
            ("blk.1.attn_k_norm.weight", vec![2]),
            ("blk.1.ffn_gate.weight", vec![HIDDEN, INTERMEDIATE]),
            ("blk.1.ffn_up.weight", vec![HIDDEN, INTERMEDIATE]),
            ("blk.1.ffn_down.weight", vec![INTERMEDIATE, HIDDEN]),
            ("token_embd_norm.weight", vec![HIDDEN]),
        ];
        tensors.retain(|(name, _)| Some(*name) != config.omit);
        if let Some((name, dimensions)) = config.wrong_shape {
            let (_, current) = tensors
                .iter_mut()
                .find(|(candidate, _)| *candidate == name)
                .with_context(|| format!("synthetic LFM2 tensor `{name}` is missing"))?;
            *current = dimensions.to_vec();
        }
        if let Some((name, dimensions)) = config.extra_tensor {
            tensors.push((name, dimensions.to_vec()));
        }

        write_synthetic_gguf(metadata, tensors)
    }

    #[allow(clippy::too_many_lines)] // The complete hybrid tensor inventory is the fixture's purpose.
    fn synthetic_qwen35_gguf(
        extra_tensor: Option<(&str, &[usize])>,
    ) -> Result<tempfile::NamedTempFile> {
        const HIDDEN: usize = 8;
        const INTERMEDIATE: usize = 12;
        const VOCAB: usize = 5;
        const QUERY_WIDTH: usize = 8;
        const KV_WIDTH: usize = 4;
        const KEY_WIDTH: usize = 8;
        const VALUE_WIDTH: usize = 8;
        const CONVOLUTION_WIDTH: usize = KEY_WIDTH * 2 + VALUE_WIDTH;
        let metadata = vec![
            ("general.architecture", GgufValue::String("qwen35".into())),
            ("qwen35.block_count", GgufValue::U32(4)),
            ("qwen35.context_length", GgufValue::U32(16)),
            ("qwen35.embedding_length", GgufValue::U32(HIDDEN as u32)),
            (
                "qwen35.feed_forward_length",
                GgufValue::U32(INTERMEDIATE as u32),
            ),
            ("qwen35.attention.head_count", GgufValue::U32(2)),
            ("qwen35.attention.head_count_kv", GgufValue::U32(1)),
            ("qwen35.attention.key_length", GgufValue::U32(4)),
            ("qwen35.attention.value_length", GgufValue::U32(4)),
            (
                "qwen35.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-6),
            ),
            ("qwen35.full_attention_interval", GgufValue::U32(4)),
            ("qwen35.rope.dimension_count", GgufValue::U32(4)),
            (
                "qwen35.rope.dimension_sections",
                GgufValue::Array {
                    element_type: GgufValueType::I32,
                    values: [1, 1, 0, 0].into_iter().map(GgufValue::I32).collect(),
                },
            ),
            ("qwen35.rope.freq_base", GgufValue::F32(10_000_000.0)),
            ("qwen35.ssm.conv_kernel", GgufValue::U32(4)),
            ("qwen35.ssm.inner_size", GgufValue::U32(VALUE_WIDTH as u32)),
            ("qwen35.ssm.state_size", GgufValue::U32(4)),
            ("qwen35.ssm.time_step_rank", GgufValue::U32(2)),
            ("qwen35.ssm.group_count", GgufValue::U32(2)),
            ("qwen35.vocab_size", GgufValue::U32(VOCAB as u32)),
            ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
            ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::String("qwen35".into())),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: ["<|endoftext|>", "<|im_end|>", "a", "1", "2"]
                        .into_iter()
                        .map(|value| GgufValue::String(value.into()))
                        .collect(),
                },
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array {
                    element_type: GgufValueType::I32,
                    values: [3, 3, 1, 1, 1].into_iter().map(GgufValue::I32).collect(),
                },
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: Vec::new(),
                },
            ),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(1)),
        ];

        // Shapes below use GGUF's on-disk dimension order; the parser reverses
        // them into the runtime's [output, input] convention.
        let mut owned_tensors = vec![
            ("token_embd.weight".to_owned(), vec![HIDDEN, VOCAB]),
            ("output_norm.weight".to_owned(), vec![HIDDEN]),
        ];
        for layer in 0..4 {
            let prefix = format!("blk.{layer}");
            owned_tensors.extend([
                (format!("{prefix}.attn_norm.weight"), vec![HIDDEN]),
                (format!("{prefix}.post_attention_norm.weight"), vec![HIDDEN]),
            ]);
            if layer % 4 == 3 {
                owned_tensors.extend([
                    (
                        format!("{prefix}.attn_q.weight"),
                        vec![HIDDEN, QUERY_WIDTH * 2],
                    ),
                    (format!("{prefix}.attn_k.weight"), vec![HIDDEN, KV_WIDTH]),
                    (format!("{prefix}.attn_v.weight"), vec![HIDDEN, KV_WIDTH]),
                    (
                        format!("{prefix}.attn_output.weight"),
                        vec![QUERY_WIDTH, HIDDEN],
                    ),
                    (format!("{prefix}.attn_q_norm.weight"), vec![4]),
                    (format!("{prefix}.attn_k_norm.weight"), vec![4]),
                ]);
            } else {
                owned_tensors.extend([
                    (
                        format!("{prefix}.attn_qkv.weight"),
                        vec![HIDDEN, CONVOLUTION_WIDTH],
                    ),
                    (
                        format!("{prefix}.attn_gate.weight"),
                        vec![HIDDEN, VALUE_WIDTH],
                    ),
                    (
                        format!("{prefix}.ssm_conv1d.weight"),
                        vec![4, CONVOLUTION_WIDTH],
                    ),
                    (format!("{prefix}.ssm_dt.bias"), vec![2]),
                    (format!("{prefix}.ssm_a"), vec![2]),
                    (format!("{prefix}.ssm_beta.weight"), vec![HIDDEN, 2]),
                    (format!("{prefix}.ssm_alpha.weight"), vec![HIDDEN, 2]),
                    (format!("{prefix}.ssm_norm.weight"), vec![4]),
                    (
                        format!("{prefix}.ssm_out.weight"),
                        vec![VALUE_WIDTH, HIDDEN],
                    ),
                ]);
            }
            owned_tensors.extend([
                (
                    format!("{prefix}.ffn_gate.weight"),
                    vec![HIDDEN, INTERMEDIATE],
                ),
                (
                    format!("{prefix}.ffn_up.weight"),
                    vec![HIDDEN, INTERMEDIATE],
                ),
                (
                    format!("{prefix}.ffn_down.weight"),
                    vec![INTERMEDIATE, HIDDEN],
                ),
            ]);
        }
        if let Some((name, dimensions)) = extra_tensor {
            owned_tensors.push((name.to_owned(), dimensions.to_vec()));
        }
        let tensors = owned_tensors
            .iter()
            .map(|(name, dimensions)| (name.as_str(), dimensions.clone()))
            .collect();
        write_synthetic_gguf(metadata, tensors)
    }

    #[allow(clippy::too_many_lines)] // Explicit invalid variants document the complete fail-closed contract.
    fn synthetic_gemma_gguf(variant: Option<&str>) -> Result<tempfile::NamedTempFile> {
        const HIDDEN: usize = 4;
        const INTERMEDIATE: usize = 6;
        const QUERY_WIDTH: usize = 4;
        const KV_WIDTH: usize = 2;
        const LAYERS: usize = 2;
        const VOCAB: usize = 5;

        let value_width = if variant == Some("bad-value-width") {
            4
        } else {
            KV_WIDTH
        };
        let mut metadata = vec![
            ("general.architecture", GgufValue::String("gemma".into())),
            ("gemma.block_count", GgufValue::U32(LAYERS as u32)),
            ("gemma.context_length", GgufValue::U32(16)),
            ("gemma.embedding_length", GgufValue::U32(HIDDEN as u32)),
            (
                "gemma.feed_forward_length",
                GgufValue::U32(INTERMEDIATE as u32),
            ),
            ("gemma.attention.head_count", GgufValue::U32(2)),
            ("gemma.attention.head_count_kv", GgufValue::U32(1)),
            ("gemma.attention.key_length", GgufValue::U32(2)),
            (
                "gemma.attention.value_length",
                GgufValue::U32(value_width as u32),
            ),
            (
                "gemma.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-6),
            ),
            ("gemma.vocab_size", GgufValue::U32(VOCAB as u32)),
            ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
            ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::String("refact".into())),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: ["<eos>", "<end_of_turn>", "a", "1", "2"]
                        .into_iter()
                        .map(|value| GgufValue::String(value.into()))
                        .collect(),
                },
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array {
                    element_type: GgufValueType::I32,
                    values: [3, 3, 1, 1, 1].into_iter().map(GgufValue::I32).collect(),
                },
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: Vec::new(),
                },
            ),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(0)),
        ];
        match variant {
            Some("metadata") => metadata.push(("gemma.unmodeled", GgufValue::U32(1))),
            Some("noncausal") => {
                metadata.push(("gemma.attention.causal", GgufValue::Bool(false)));
            }
            Some("custom-rope") => {
                metadata.push(("gemma.rope.freq_base", GgufValue::F32(20_000.0)));
                metadata.push(("gemma.rope.dimension_count", GgufValue::U32(2)));
            }
            _ => {}
        }

        let mut tensors = vec![
            ("token_embd.weight".to_owned(), vec![HIDDEN, VOCAB]),
            ("output_norm.weight".to_owned(), vec![HIDDEN]),
        ];
        for layer in 0..LAYERS {
            let prefix = format!("blk.{layer}");
            tensors.extend([
                (format!("{prefix}.attn_norm.weight"), vec![HIDDEN]),
                (format!("{prefix}.ffn_norm.weight"), vec![HIDDEN]),
                (format!("{prefix}.attn_q.weight"), vec![HIDDEN, QUERY_WIDTH]),
                (format!("{prefix}.attn_k.weight"), vec![HIDDEN, KV_WIDTH]),
                (format!("{prefix}.attn_v.weight"), vec![HIDDEN, KV_WIDTH]),
                (
                    format!("{prefix}.attn_output.weight"),
                    vec![QUERY_WIDTH, HIDDEN],
                ),
                (
                    format!("{prefix}.ffn_gate.weight"),
                    vec![HIDDEN, INTERMEDIATE],
                ),
                (
                    format!("{prefix}.ffn_up.weight"),
                    vec![HIDDEN, INTERMEDIATE],
                ),
                (
                    format!("{prefix}.ffn_down.weight"),
                    vec![INTERMEDIATE, HIDDEN],
                ),
            ]);
            match variant {
                Some("extra-post-norm") => {
                    tensors.push((format!("{prefix}.post_attention_norm.weight"), vec![HIDDEN]));
                }
                Some("extra-q-norm") => {
                    tensors.push((format!("{prefix}.attn_q_norm.weight"), vec![2]));
                }
                Some("extra-bias") => {
                    tensors.push((format!("{prefix}.attn_q.bias"), vec![QUERY_WIDTH]));
                }
                _ => {}
            }
        }
        if variant == Some("extra-output") {
            tensors.push(("output.weight".to_owned(), vec![HIDDEN, VOCAB]));
        }
        let tensor_refs = tensors
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.clone()))
            .collect();
        write_synthetic_gguf(metadata, tensor_refs)
    }

    #[allow(clippy::too_many_lines)] // The explicit metadata/tensor variants make fail-closed tests auditable.
    fn synthetic_gemma2_gguf(omit_or_variant: Option<&str>) -> Result<tempfile::NamedTempFile> {
        const HIDDEN: usize = 4;
        const INTERMEDIATE: usize = 6;
        const QUERY_WIDTH: usize = 4;
        const KV_WIDTH: usize = 2;
        const LAYERS: usize = 2;
        const VOCAB: usize = 5;

        let mut metadata = vec![
            ("general.architecture", GgufValue::String("gemma2".into())),
            ("gemma2.block_count", GgufValue::U32(LAYERS as u32)),
            ("gemma2.context_length", GgufValue::U32(16)),
            ("gemma2.embedding_length", GgufValue::U32(HIDDEN as u32)),
            (
                "gemma2.feed_forward_length",
                GgufValue::U32(INTERMEDIATE as u32),
            ),
            ("gemma2.attention.head_count", GgufValue::U32(2)),
            ("gemma2.attention.head_count_kv", GgufValue::U32(1)),
            ("gemma2.attention.key_length", GgufValue::U32(2)),
            ("gemma2.attention.value_length", GgufValue::U32(2)),
            (
                "gemma2.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-6),
            ),
            ("gemma2.attention.sliding_window", GgufValue::U32(4)),
            ("gemma2.attn_logit_softcapping", GgufValue::F32(50.0)),
            ("gemma2.final_logit_softcapping", GgufValue::F32(30.0)),
            ("gemma2.rope.freq_base", GgufValue::F32(10_000.0)),
            ("gemma2.vocab_size", GgufValue::U32(VOCAB as u32)),
            ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
            ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::String("refact".into())),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: ["<eos>", "<end_of_turn>", "a", "1", "2"]
                        .into_iter()
                        .map(|value| GgufValue::String(value.into()))
                        .collect(),
                },
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array {
                    element_type: GgufValueType::I32,
                    values: [3, 3, 1, 1, 1].into_iter().map(GgufValue::I32).collect(),
                },
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: Vec::new(),
                },
            ),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(0)),
        ];
        if omit_or_variant == Some("metadata") {
            metadata.push(("gemma2.unmodeled", GgufValue::U32(1)));
        }
        if let Some(period) = match omit_or_variant {
            Some("pattern-0") => Some(0),
            Some("pattern-1") => Some(1),
            Some("pattern-2") => Some(2),
            _ => None,
        } {
            metadata.push((
                "gemma2.attention.sliding_window_pattern",
                GgufValue::U32(period),
            ));
        }

        let mut tensors = vec![
            ("token_embd.weight".to_owned(), vec![HIDDEN, VOCAB]),
            ("output_norm.weight".to_owned(), vec![HIDDEN]),
        ];
        for layer in 0..LAYERS {
            let prefix = format!("blk.{layer}");
            tensors.extend([
                (format!("{prefix}.attn_norm.weight"), vec![HIDDEN]),
                (format!("{prefix}.ffn_norm.weight"), vec![HIDDEN]),
                (format!("{prefix}.post_attention_norm.weight"), vec![HIDDEN]),
                (format!("{prefix}.post_ffw_norm.weight"), vec![HIDDEN]),
                (format!("{prefix}.attn_q.weight"), vec![HIDDEN, QUERY_WIDTH]),
                (format!("{prefix}.attn_k.weight"), vec![HIDDEN, KV_WIDTH]),
                (format!("{prefix}.attn_v.weight"), vec![HIDDEN, KV_WIDTH]),
                (
                    format!("{prefix}.attn_output.weight"),
                    vec![QUERY_WIDTH, HIDDEN],
                ),
                (
                    format!("{prefix}.ffn_gate.weight"),
                    vec![HIDDEN, INTERMEDIATE],
                ),
                (
                    format!("{prefix}.ffn_up.weight"),
                    vec![HIDDEN, INTERMEDIATE],
                ),
                (
                    format!("{prefix}.ffn_down.weight"),
                    vec![INTERMEDIATE, HIDDEN],
                ),
            ]);
        }
        match omit_or_variant {
            Some("extra-output") => {
                tensors.push(("output.weight".to_owned(), vec![HIDDEN, VOCAB]));
            }
            Some("extra-q-norm") => {
                tensors.push(("blk.0.attn_q_norm.weight".to_owned(), vec![2]));
                tensors.push(("blk.1.attn_q_norm.weight".to_owned(), vec![2]));
            }
            Some("extra-k-norm") => {
                tensors.push(("blk.0.attn_k_norm.weight".to_owned(), vec![2]));
                tensors.push(("blk.1.attn_k_norm.weight".to_owned(), vec![2]));
            }
            Some(omit) if !matches!(omit, "metadata" | "pattern-0" | "pattern-1" | "pattern-2") => {
                tensors.retain(|(name, _)| name != omit);
            }
            _ => {}
        }
        let tensor_refs = tensors
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.clone()))
            .collect();
        write_synthetic_gguf(metadata, tensor_refs)
    }

    fn write_synthetic_gguf(
        metadata: Vec<(&str, GgufValue)>,
        tensors: Vec<(&str, Vec<usize>)>,
    ) -> Result<tempfile::NamedTempFile> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in &metadata {
            push_string(&mut bytes, key);
            bytes.extend_from_slice(&metadata_tag(value).to_le_bytes());
            push_metadata_value(&mut bytes, value);
        }
        let mut offset = 0_u64;
        let mut payloads = Vec::new();
        for (name, dimensions) in &tensors {
            push_string(&mut bytes, name);
            bytes.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
            for dimension in dimensions {
                bytes.extend_from_slice(&u64::try_from(*dimension)?.to_le_bytes());
            }
            bytes.extend_from_slice(&0_u32.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            let payload = vec![0_u8; dimensions.iter().product::<usize>() * 4];
            offset += align_32(u64::try_from(payload.len())?);
            payloads.push(payload);
        }
        bytes.resize(usize::try_from(align_32(u64::try_from(bytes.len())?))?, 0);
        for payload in payloads {
            bytes.extend_from_slice(&payload);
            bytes.resize(usize::try_from(align_32(u64::try_from(bytes.len())?))?, 0);
        }
        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(&bytes)?;
        file.flush()?;
        Ok(file)
    }

    #[allow(clippy::too_many_lines)] // Explicit metadata and shapes keep this GGUF fixture auditable.
    fn synthetic_qwen3_moe_gguf(
        config: SyntheticQwen3MoeConfig,
    ) -> Result<tempfile::NamedTempFile> {
        const HIDDEN: usize = 6;
        const QUERY_WIDTH: usize = 4;
        const EXPERTS: usize = 3;
        let fallback_expert_width = usize::try_from(
            config
                .feed_forward_length
                .checked_div(config.experts_per_token)
                .unwrap_or(1),
        )?
        .max(1);
        let expert_width = config
            .expert_feed_forward_length
            .filter(|width| *width > 0)
            .map(usize::try_from)
            .transpose()?
            .unwrap_or(fallback_expert_width);
        let mut metadata = vec![
            ("general.architecture", GgufValue::String("qwen3moe".into())),
            ("qwen3moe.block_count", GgufValue::U32(1)),
            ("qwen3moe.context_length", GgufValue::U32(16)),
            ("qwen3moe.embedding_length", GgufValue::U32(HIDDEN as u32)),
            (
                "qwen3moe.feed_forward_length",
                GgufValue::U32(config.feed_forward_length),
            ),
            ("qwen3moe.attention.head_count", GgufValue::U32(2)),
            ("qwen3moe.attention.head_count_kv", GgufValue::U32(1)),
            ("qwen3moe.attention.key_length", GgufValue::U32(2)),
            ("qwen3moe.attention.value_length", GgufValue::U32(2)),
            ("qwen3moe.rope.dimension_count", GgufValue::U32(2)),
            ("qwen3moe.rope.freq_base", GgufValue::F32(1_000_000.0)),
            (
                "qwen3moe.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-6),
            ),
            ("qwen3moe.expert_count", GgufValue::U32(config.expert_count)),
            (
                "qwen3moe.expert_used_count",
                GgufValue::U32(config.experts_per_token),
            ),
            ("qwen3moe.vocab_size", GgufValue::U32(4)),
            ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
            ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::String("qwen2".into())),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: ["<|im_end|>", "a", "1", "2"]
                        .into_iter()
                        .map(|value| GgufValue::String(value.into()))
                        .collect(),
                },
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array {
                    element_type: GgufValueType::I32,
                    values: [3, 1, 1, 1].into_iter().map(GgufValue::I32).collect(),
                },
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: Vec::new(),
                },
            ),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(0)),
        ];
        if let Some(expert_width) = config.expert_feed_forward_length {
            metadata.push((
                "qwen3moe.expert_feed_forward_length",
                GgufValue::U32(expert_width),
            ));
        }
        if let Some(extra) = config.extra_metadata {
            metadata.push(extra);
        }

        let mut tensors = vec![
            ("token_embd.weight", vec![HIDDEN, 4]),
            ("blk.0.attn_norm.weight", vec![HIDDEN]),
            ("blk.0.ffn_norm.weight", vec![HIDDEN]),
            ("blk.0.attn_q.weight", vec![HIDDEN, QUERY_WIDTH]),
            ("blk.0.attn_k.weight", vec![HIDDEN, 2]),
            ("blk.0.attn_v.weight", vec![HIDDEN, 2]),
            ("blk.0.attn_output.weight", vec![QUERY_WIDTH, HIDDEN]),
            ("blk.0.attn_q_norm.weight", vec![2]),
            ("blk.0.attn_k_norm.weight", vec![2]),
            ("blk.0.ffn_gate_inp.weight", vec![HIDDEN, EXPERTS]),
            (
                "blk.0.ffn_gate_exps.weight",
                vec![HIDDEN, expert_width, EXPERTS],
            ),
            (
                "blk.0.ffn_up_exps.weight",
                vec![HIDDEN, expert_width, EXPERTS],
            ),
            (
                "blk.0.ffn_down_exps.weight",
                vec![expert_width, HIDDEN, EXPERTS],
            ),
            ("output_norm.weight", vec![HIDDEN]),
        ];
        tensors.retain(|(name, _)| Some(*name) != config.omit);
        if let Some((name, dimensions)) = config.extra_tensor {
            tensors.push((name, dimensions.to_vec()));
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in &metadata {
            push_string(&mut bytes, key);
            bytes.extend_from_slice(&metadata_tag(value).to_le_bytes());
            push_metadata_value(&mut bytes, value);
        }
        let mut offset = 0_u64;
        let mut payloads = Vec::new();
        for (name, dimensions) in &tensors {
            push_string(&mut bytes, name);
            bytes.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
            for dimension in dimensions {
                bytes.extend_from_slice(&u64::try_from(*dimension)?.to_le_bytes());
            }
            bytes.extend_from_slice(&0_u32.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            let payload = vec![0_u8; dimensions.iter().product::<usize>() * 4];
            offset += align_32(u64::try_from(payload.len())?);
            payloads.push(payload);
        }
        bytes.resize(usize::try_from(align_32(u64::try_from(bytes.len())?))?, 0);
        for payload in payloads {
            bytes.extend_from_slice(&payload);
            bytes.resize(usize::try_from(align_32(u64::try_from(bytes.len())?))?, 0);
        }
        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(&bytes)?;
        file.flush()?;
        Ok(file)
    }

    #[derive(Debug)]
    struct SyntheticQwen3MoeConfig {
        omit: Option<&'static str>,
        feed_forward_length: u32,
        expert_count: u32,
        experts_per_token: u32,
        expert_feed_forward_length: Option<u32>,
        extra_metadata: Option<(&'static str, GgufValue)>,
        extra_tensor: Option<(&'static str, &'static [usize])>,
    }

    impl Default for SyntheticQwen3MoeConfig {
        fn default() -> Self {
            Self {
                omit: None,
                feed_forward_length: 6,
                expert_count: 3,
                experts_per_token: 2,
                expert_feed_forward_length: Some(3),
                extra_metadata: None,
                extra_tensor: None,
            }
        }
    }

    #[allow(clippy::too_many_lines)] // Explicit metadata and shapes keep this GGUF fixture auditable.
    fn synthetic_granite_moe_gguf(omit: Option<&str>) -> Result<tempfile::NamedTempFile> {
        const HIDDEN: usize = 4;
        const INTERMEDIATE: usize = 3;
        const EXPERTS: usize = 3;
        let metadata = vec![
            (
                "general.architecture",
                GgufValue::String("granitemoe".into()),
            ),
            ("granitemoe.block_count", GgufValue::U32(1)),
            ("granitemoe.context_length", GgufValue::U32(16)),
            ("granitemoe.embedding_length", GgufValue::U32(HIDDEN as u32)),
            (
                "granitemoe.feed_forward_length",
                GgufValue::U32(INTERMEDIATE as u32),
            ),
            ("granitemoe.attention.head_count", GgufValue::U32(2)),
            ("granitemoe.attention.head_count_kv", GgufValue::U32(1)),
            ("granitemoe.rope.freq_base", GgufValue::F32(1_500_000.0)),
            ("granitemoe.rope.dimension_count", GgufValue::U32(2)),
            (
                "granitemoe.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-6),
            ),
            ("granitemoe.expert_count", GgufValue::U32(EXPERTS as u32)),
            ("granitemoe.expert_used_count", GgufValue::U32(2)),
            ("granitemoe.attention.scale", GgufValue::F32(0.015_625)),
            ("granitemoe.embedding_scale", GgufValue::F32(12.0)),
            ("granitemoe.residual_scale", GgufValue::F32(0.22)),
            ("granitemoe.logit_scale", GgufValue::F32(6.0)),
            ("granitemoe.vocab_size", GgufValue::U32(4)),
            ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
            ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::String("refact".into())),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: ["<|end_of_text|>", "a", "1", "2"]
                        .into_iter()
                        .map(|value| GgufValue::String(value.into()))
                        .collect(),
                },
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array {
                    element_type: GgufValueType::I32,
                    values: [3, 1, 1, 1].into_iter().map(GgufValue::I32).collect(),
                },
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: Vec::new(),
                },
            ),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(0)),
        ];
        let mut tensors = vec![
            ("token_embd.weight", vec![HIDDEN, 4]),
            ("blk.0.attn_norm.weight", vec![HIDDEN]),
            ("blk.0.ffn_norm.weight", vec![HIDDEN]),
            ("blk.0.attn_q.weight", vec![HIDDEN, HIDDEN]),
            ("blk.0.attn_k.weight", vec![HIDDEN, 2]),
            ("blk.0.attn_v.weight", vec![HIDDEN, 2]),
            ("blk.0.attn_output.weight", vec![HIDDEN, HIDDEN]),
            ("blk.0.ffn_gate_inp.weight", vec![HIDDEN, EXPERTS]),
            (
                "blk.0.ffn_gate_exps.weight",
                vec![HIDDEN, INTERMEDIATE, EXPERTS],
            ),
            (
                "blk.0.ffn_up_exps.weight",
                vec![HIDDEN, INTERMEDIATE, EXPERTS],
            ),
            (
                "blk.0.ffn_down_exps.weight",
                vec![INTERMEDIATE, HIDDEN, EXPERTS],
            ),
            ("output_norm.weight", vec![HIDDEN]),
        ];
        tensors.retain(|(name, _)| Some(*name) != omit);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in &metadata {
            push_string(&mut bytes, key);
            bytes.extend_from_slice(&metadata_tag(value).to_le_bytes());
            push_metadata_value(&mut bytes, value);
        }
        let mut offset = 0_u64;
        let mut payloads = Vec::new();
        for (name, dimensions) in &tensors {
            push_string(&mut bytes, name);
            bytes.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
            for dimension in dimensions {
                bytes.extend_from_slice(&u64::try_from(*dimension)?.to_le_bytes());
            }
            bytes.extend_from_slice(&0_u32.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            let payload = vec![0_u8; dimensions.iter().product::<usize>() * 4];
            offset += align_32(u64::try_from(payload.len())?);
            payloads.push(payload);
        }
        bytes.resize(usize::try_from(align_32(u64::try_from(bytes.len())?))?, 0);
        for payload in payloads {
            bytes.extend_from_slice(&payload);
            bytes.resize(usize::try_from(align_32(u64::try_from(bytes.len())?))?, 0);
        }
        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(&bytes)?;
        file.flush()?;
        Ok(file)
    }

    fn replace_all_same_length(bytes: &mut [u8], from: &[u8], to: &[u8]) {
        assert_eq!(from.len(), to.len());
        let mut cursor = 0;
        while let Some(relative) = bytes[cursor..]
            .windows(from.len())
            .position(|window| window == from)
        {
            let start = cursor + relative;
            bytes[start..start + from.len()].copy_from_slice(to);
            cursor = start + from.len();
        }
    }

    // Keeping the fixture's metadata, index, and payload serialization together
    // makes malformed-file cases easier to audit.
    #[allow(clippy::large_types_passed_by_value, clippy::too_many_lines)]
    fn synthetic_llama_gguf_with(
        omit: Option<&str>,
        config: SyntheticLlamaConfig,
    ) -> Result<tempfile::NamedTempFile> {
        let head_dim = config
            .key_length
            .unwrap_or(config.hidden_size / config.query_heads);
        let query_width = config.query_heads * head_dim;
        let kv_width = config.kv_heads * head_dim;
        let mut metadata = vec![
            ("general.architecture", GgufValue::String("llama".into())),
            (
                "llama.block_count",
                GgufValue::U32(u32::try_from(config.block_count)?),
            ),
            (
                "llama.context_length",
                GgufValue::U32(u32::try_from(config.context_length)?),
            ),
            (
                "llama.embedding_length",
                GgufValue::U32(u32::try_from(config.hidden_size)?),
            ),
            ("llama.feed_forward_length", GgufValue::U32(6)),
            (
                "llama.attention.head_count",
                GgufValue::U32(u32::try_from(config.query_heads)?),
            ),
            (
                "llama.attention.head_count_kv",
                GgufValue::U32(u32::try_from(config.kv_heads)?),
            ),
            ("llama.rope.freq_base", GgufValue::F32(config.rope_theta)),
            (
                "llama.attention.layer_norm_rms_epsilon",
                GgufValue::F32(config.norm_epsilon),
            ),
            ("llama.vocab_size", GgufValue::U32(4)),
            (
                "llama.rope.dimension_count",
                GgufValue::U32(u32::try_from(config.rope_dimension.unwrap_or(head_dim))?),
            ),
            ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
            ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::String("smollm".into())),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: ["<|stop|>", "a", "1", "2"]
                        .into_iter()
                        .map(|value| GgufValue::String(value.into()))
                        .collect(),
                },
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array {
                    element_type: GgufValueType::I32,
                    values: [3, 1, 1, 1].into_iter().map(GgufValue::I32).collect(),
                },
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array {
                    element_type: GgufValueType::String,
                    values: Vec::new(),
                },
            ),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(0)),
        ];
        if let Some(key_length) = config.key_length {
            metadata.push((
                "llama.attention.key_length",
                GgufValue::U32(u32::try_from(key_length)?),
            ));
        }
        if let Some(value_length) = config.value_length {
            metadata.push((
                "llama.attention.value_length",
                GgufValue::U32(u32::try_from(value_length)?),
            ));
        }
        if let Some(causal) = config.causal {
            metadata.push(("llama.attention.causal", GgufValue::Bool(causal)));
        }
        if let Some(sliding_window) = config.sliding_window {
            metadata.push((
                "llama.attention.sliding_window",
                GgufValue::U32(u32::try_from(sliding_window)?),
            ));
        }
        if config.sliding_window_pattern {
            metadata.push(("llama.attention.sliding_window_pattern", GgufValue::U32(1)));
        }
        if let Some(scale) = config.attention_scale {
            metadata.push(("llama.attention.scale", GgufValue::F32(scale)));
        }
        if let Some(expert_count) = config.expert_count {
            metadata.push((
                "llama.expert_count",
                GgufValue::U32(u32::try_from(expert_count)?),
            ));
        }
        if let Some(expert_used_count) = config.expert_used_count {
            metadata.push((
                "llama.expert_used_count",
                GgufValue::U32(u32::try_from(expert_used_count)?),
            ));
        }
        if let Some(expert_weights_scale) = config.expert_weights_scale {
            metadata.push((
                "llama.expert_weights_scale",
                GgufValue::F32(expert_weights_scale),
            ));
        }
        if config.split_metadata {
            metadata.push(("split.count", GgufValue::U32(1)));
        }
        if let Some(key) = config.unsupported_llama_key {
            metadata.push((key, GgufValue::F32(2.0)));
        }
        let mut tensors = vec![
            ("token_embd.weight", vec![config.hidden_size, 4]),
            ("blk.0.attn_norm.weight", vec![config.hidden_size]),
            ("blk.0.ffn_norm.weight", vec![config.hidden_size]),
            ("blk.0.attn_q.weight", vec![config.hidden_size, query_width]),
            ("blk.0.attn_k.weight", vec![config.hidden_size, kv_width]),
            ("blk.0.attn_v.weight", vec![config.hidden_size, kv_width]),
            (
                "blk.0.attn_output.weight",
                vec![query_width, config.hidden_size],
            ),
            ("output_norm.weight", vec![config.hidden_size]),
        ];
        if let Some(expert_count) = config.expert_count.filter(|count| *count > 0) {
            tensors.extend([
                (
                    "blk.0.ffn_gate_inp.weight",
                    vec![config.hidden_size, expert_count],
                ),
                (
                    "blk.0.ffn_gate_exps.weight",
                    vec![config.hidden_size, 6, expert_count],
                ),
                (
                    "blk.0.ffn_up_exps.weight",
                    vec![config.hidden_size, 6, expert_count],
                ),
                (
                    "blk.0.ffn_down_exps.weight",
                    vec![6, config.hidden_size, expert_count],
                ),
            ]);
        } else {
            tensors.extend([
                ("blk.0.ffn_gate.weight", vec![config.hidden_size, 6]),
                ("blk.0.ffn_up.weight", vec![config.hidden_size, 6]),
                ("blk.0.ffn_down.weight", vec![6, config.hidden_size]),
            ]);
        }
        if config.projection_biases {
            tensors.extend([
                ("blk.0.attn_q.bias", vec![query_width]),
                ("blk.0.attn_k.bias", vec![kv_width]),
                ("blk.0.attn_v.bias", vec![kv_width]),
            ]);
        }
        if config.qk_norm {
            tensors.extend([
                ("blk.0.attn_q_norm.weight", vec![head_dim]),
                ("blk.0.attn_k_norm.weight", vec![head_dim]),
            ]);
        }
        if let Some(name) = config.extra_bias {
            let length = match name {
                "blk.0.attn_q.bias" => query_width,
                "blk.0.attn_k.bias" | "blk.0.attn_v.bias" => kv_width,
                "blk.0.attn_output.bias" | "blk.0.ffn_down.bias" => config.hidden_size,
                "blk.0.ffn_gate.bias" | "blk.0.ffn_up.bias" => 6,
                "output.bias" => 4,
                _ => bail!("unsupported synthetic extra bias `{name}`"),
            };
            tensors.push((name, vec![length]));
        }
        if let Some(factors) = config.rope_factors {
            tensors.push(("rope_freqs.weight", vec![factors.len()]));
        }
        if let Some((name, dimensions)) = config.extra_tensor {
            tensors.push((name, dimensions.to_vec()));
        }
        if let Some((name, dimensions)) = config.wrong_shape {
            let (_, shape) = tensors
                .iter_mut()
                .find(|(candidate, _)| *candidate == name)
                .with_context(|| format!("synthetic wrong-shape tensor `{name}` is missing"))?;
            *shape = dimensions.to_vec();
        }
        tensors.retain(|(name, _)| Some(*name) != omit);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in &metadata {
            push_string(&mut bytes, key);
            bytes.extend_from_slice(&metadata_tag(value).to_le_bytes());
            push_metadata_value(&mut bytes, value);
        }
        let mut offset = 0_u64;
        let mut payloads = Vec::new();
        for (name, dimensions) in &tensors {
            push_string(&mut bytes, name);
            bytes.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
            for dimension in dimensions {
                bytes.extend_from_slice(&u64::try_from(*dimension)?.to_le_bytes());
            }
            bytes.extend_from_slice(&0_u32.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            let elements = dimensions.iter().product::<usize>();
            let payload = if *name == "rope_freqs.weight" {
                config
                    .rope_factors
                    .context("rope frequency tensor has no fixture values")?
                    .iter()
                    .flat_map(|factor| factor.to_le_bytes())
                    .collect()
            } else {
                vec![0_u8; elements * 4]
            };
            offset += align_32(u64::try_from(payload.len())?);
            payloads.push(payload);
        }
        bytes.resize(usize::try_from(align_32(u64::try_from(bytes.len())?))?, 0);
        for payload in payloads {
            bytes.extend_from_slice(&payload);
            bytes.resize(usize::try_from(align_32(u64::try_from(bytes.len())?))?, 0);
        }
        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(&bytes)?;
        file.flush()?;
        Ok(file)
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn metadata_tag(value: &GgufValue) -> u32 {
        match value.value_type() {
            GgufValueType::U8 => 0,
            GgufValueType::I8 => 1,
            GgufValueType::U16 => 2,
            GgufValueType::I16 => 3,
            GgufValueType::U32 => 4,
            GgufValueType::I32 => 5,
            GgufValueType::F32 => 6,
            GgufValueType::Bool => 7,
            GgufValueType::String => 8,
            GgufValueType::Array => 9,
            GgufValueType::U64 => 10,
            GgufValueType::I64 => 11,
            GgufValueType::F64 => 12,
        }
    }

    fn push_metadata_value(bytes: &mut Vec<u8>, value: &GgufValue) {
        match value {
            GgufValue::U32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            GgufValue::F32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            GgufValue::Bool(value) => bytes.push(u8::from(*value)),
            GgufValue::String(value) => push_string(bytes, value),
            GgufValue::Array {
                element_type,
                values,
            } => {
                let element_tag = match element_type {
                    GgufValueType::String => 8_u32,
                    GgufValueType::I32 => 5_u32,
                    _ => panic!("synthetic fixture only uses string and i32 arrays"),
                };
                bytes.extend_from_slice(&element_tag.to_le_bytes());
                bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in values {
                    push_metadata_value(bytes, value);
                }
            }
            GgufValue::I32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            _ => panic!("unsupported synthetic metadata value"),
        }
    }

    fn align_32(value: u64) -> u64 {
        (value + 31) & !31
    }
}
