use std::{
    any::Any,
    fmt::{self, Debug},
};

use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};

// The decoder grows RoPE tables lazily, but validates the checkpoint's declared
// maximum geometry so an eventual full-context table remains inside this
// allocation budget. KV storage likewise grows only as tokens are processed.
const MAX_ROPE_TABLE_ELEMENTS: usize = 64 << 20;
pub(crate) const MAX_DECODER_LAYERS: usize = 4_096;
const MAX_F32_KV_BYTES_PER_TOKEN: usize = 16 << 20;
pub(crate) const MAX_ROUTED_EXPERTS: usize = 1_024;
const MAX_ROUTED_EXPERT_INSTANCES: usize = 65_536;

pub(crate) fn validate_rope_table_shape(max_context: usize, head_dim: usize) -> Result<()> {
    ensure!(head_dim > 0, "head_dim must be positive");
    ensure!(
        head_dim.is_multiple_of(2),
        "half-split RoPE requires an even head_dim"
    );
    let rope_elements = max_context
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow::anyhow!("RoPE table shape overflows usize"))?;
    ensure!(
        rope_elements <= MAX_ROPE_TABLE_ELEMENTS,
        "decoder RoPE table requires {rope_elements} context-by-head elements, exceeding the current limit {MAX_ROPE_TABLE_ELEMENTS}"
    );
    Ok(())
}

/// Activation used by the feed-forward network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
    /// Tanh-approximated GELU used by Gemma-family gated feed-forward blocks.
    GeluTanh,
}

/// How pairs of head dimensions are rotated by `RoPE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// Rotate the first and second half of a head against one another.
    /// Used by Qwen2, Llama and most Hugging Face decoder models.
    HalfSplit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RopeSpec {
    pub theta: f64,
    pub layout: RopeLayout,
    /// Number of leading dimensions in each attention head transformed by
    /// `RoPE`. Keeping this independent from `head_dim` supports partial-RoPE
    /// decoders without changing KV geometry.
    pub rotary_dim: usize,
    /// Optional per-frequency divisors for the base inverse frequencies.
    /// This is the format-neutral form used by GGUF `rope_freqs.weight` and by
    /// adapters that lower family-specific scaling formulas such as Llama 3.
    pub frequency_factors: Option<Vec<f32>>,
}

/// Positional encoding applied by one attention layer.
///
/// The explicit `None` variant models architectures such as `SmolLM3`, where
/// selected attention layers deliberately omit positional encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionEncodingSpec {
    Rope(RopeSpec),
    None,
}

impl PositionEncodingSpec {
    #[must_use]
    pub fn rope(&self) -> Option<&RopeSpec> {
        match self {
            Self::Rope(rope) => Some(rope),
            Self::None => None,
        }
    }

    #[must_use]
    pub fn rope_mut(&mut self) -> Option<&mut RopeSpec> {
        match self {
            Self::Rope(rope) => Some(rope),
            Self::None => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    FullCausal,
    /// Causal attention restricted to the current token and the preceding
    /// `window - 1` absolute positions.
    SlidingCausal {
        window: usize,
    },
}

/// How normalization is arranged around a decoder block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormLayout {
    /// `x + attn(norm(x))`, followed by `x + mlp(norm(x))`.
    PreNorm,
    /// Gemma-style pre/post normalization around both residual branches.
    Sandwich,
}

/// Feed-forward topology selected independently for each decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedForwardKind {
    Dense,
    /// A router selects a fixed number of SwiGLU/GELU experts per token. The
    /// selected raw router logits are normalized with a softmax before expert
    /// outputs are combined.
    SparseMoe(SparseMoeSpec),
}

/// How selected expert scores are converted into combination weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterWeighting {
    /// Select top-k raw logits, then apply softmax only within that set.
    SelectedSoftmax,
}

/// Geometry for a top-k sparse mixture-of-experts feed-forward block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseMoeSpec {
    pub expert_count: usize,
    pub experts_per_token: usize,
    pub expert_intermediate_size: usize,
    pub router_weighting: RouterWeighting,
}

/// Attention capabilities selected independently for every decoder layer.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionSpec {
    pub kind: AttentionKind,
    pub position_encoding: PositionEncodingSpec,
    pub qk_norm: bool,
    /// Whether each query head is projected as interleaved `[query, gate]`
    /// and the attention result is multiplied by `sigmoid(gate)` before the
    /// output projection.
    pub output_gate: bool,
    /// Multiplicative scale applied to QK scores before masking and softmax.
    pub scale: f64,
    /// Optional tanh cap applied to attention logits before masking.
    pub logit_softcap: Option<f64>,
}

impl AttentionSpec {
    /// Returns this layer's `RoPE` configuration when positional rotation is
    /// enabled.
    #[must_use]
    pub fn rope(&self) -> Option<&RopeSpec> {
        self.position_encoding.rope()
    }

    /// Returns the mutable `RoPE` configuration when positional rotation is
    /// enabled.
    #[must_use]
    pub fn rope_mut(&mut self) -> Option<&mut RopeSpec> {
        self.position_encoding.rope_mut()
    }
}

/// Geometry for a double-gated causal depthwise short convolution.
///
/// The operator keeps the preceding `kernel_size - 1` projected `B * x`
/// vectors as sequence state. This is intentionally an execution-level
/// contract rather than an LFM2 flag so other hybrid decoders can reuse it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortConvolutionSpec {
    pub kernel_size: usize,
}

/// Geometry for a causal Gated `DeltaNet` recurrent mixer.
///
/// Q/K/V are projected together, passed through a depthwise causal
/// convolution and `SiLU`, then consumed by an F32 delta-rule recurrence. The
/// state belongs to a sequence session rather than the immutable model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GatedDeltaNetSpec {
    pub convolution_kernel_size: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub qk_norm_epsilon: f64,
}

/// Geometry and optional tensors for a causal Mamba2 selective state-space
/// branch.
///
/// The convolution tail and recurrent matrix are sequence state.  Keeping
/// this contract independent from Falcon-H1 lets other attention/SSM hybrids
/// reuse the same operator and cache policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mamba2Spec {
    pub convolution_kernel_size: usize,
    pub inner_size: usize,
    pub state_size: usize,
    pub num_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    pub convolution_bias: bool,
    pub input_projection_bias: bool,
    pub output_projection_bias: bool,
    pub gated_rms_norm: bool,
}

/// Two token mixers evaluated from the same normalized layer input and added
/// before the residual connection.
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelAttentionMamba2Spec {
    pub attention: AttentionSpec,
    pub mamba2: Mamba2Spec,
}

/// Token-mixing operator selected independently for every decoder layer.
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceMixerSpec {
    Attention(AttentionSpec),
    ShortConvolution(ShortConvolutionSpec),
    GatedDeltaNet(GatedDeltaNetSpec),
    ParallelAttentionMamba2(ParallelAttentionMamba2Spec),
}

impl SequenceMixerSpec {
    #[must_use]
    pub fn attention(&self) -> Option<&AttentionSpec> {
        match self {
            Self::Attention(attention) => Some(attention),
            Self::ParallelAttentionMamba2(parallel) => Some(&parallel.attention),
            Self::ShortConvolution(_) | Self::GatedDeltaNet(_) => None,
        }
    }

    #[must_use]
    pub fn attention_mut(&mut self) -> Option<&mut AttentionSpec> {
        match self {
            Self::Attention(attention) => Some(attention),
            Self::ParallelAttentionMamba2(parallel) => Some(&mut parallel.attention),
            Self::ShortConvolution(_) | Self::GatedDeltaNet(_) => None,
        }
    }
}

/// Execution semantics that may vary between layers while tensor geometry is
/// shared by the model.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderLayerSpec {
    pub mixer: SequenceMixerSpec,
    pub activation: Activation,
    pub norm_layout: NormLayout,
    pub feed_forward: FeedForwardKind,
    /// Multiplicative scale applied to the token-mixer branch before its
    /// residual addition.
    pub mixer_residual_scale: f64,
    /// Multiplicative scale applied to the feed-forward branch before its
    /// residual addition.
    pub feed_forward_residual_scale: f64,
}

impl DecoderLayerSpec {
    #[must_use]
    pub fn standard(rope: RopeSpec, head_dim: usize) -> Self {
        Self {
            mixer: SequenceMixerSpec::Attention(AttentionSpec {
                kind: AttentionKind::FullCausal,
                position_encoding: PositionEncodingSpec::Rope(rope),
                qk_norm: false,
                output_gate: false,
                scale: 1.0 / (head_dim as f64).sqrt(),
                logit_softcap: None,
            }),
            activation: Activation::Silu,
            norm_layout: NormLayout::PreNorm,
            feed_forward: FeedForwardKind::Dense,
            mixer_residual_scale: 1.0,
            feed_forward_residual_scale: 1.0,
        }
    }

    #[must_use]
    pub fn attention(&self) -> Option<&AttentionSpec> {
        self.mixer.attention()
    }

    #[must_use]
    pub fn attention_mut(&mut self) -> Option<&mut AttentionSpec> {
        self.mixer.attention_mut()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionBias {
    pub query: bool,
    pub key: bool,
    pub value: bool,
    pub output: bool,
}

/// Metadata needed by model-independent generation code.
///
/// This deliberately contains no Transformer topology. A future state-space
/// model or other autoregressive architecture can implement `CausalLm`
/// without inventing attention heads or `RoPE` settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeSpec {
    pub vocab_size: usize,
    pub max_context: usize,
    pub stop_token_ids: Vec<u32>,
}

impl RuntimeSpec {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.vocab_size > 0, "vocab_size must be positive");
        ensure!(self.max_context > 0, "max_context must be positive");
        ensure!(
            u32::try_from(self.max_context).is_ok(),
            "max_context exceeds the supported u32 position range"
        );
        ensure!(
            self.stop_token_ids
                .iter()
                .all(|&token| (token as usize) < self.vocab_size),
            "stop token IDs must be inside the vocabulary"
        );
        Ok(())
    }
}

/// A model-independent description of a decoder-only Transformer.
///
/// Adapters populate this structure from a family-specific `config.json`.
/// The execution engine never needs to know whether the checkpoint was called
/// Qwen, Llama or something else.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderSpec {
    pub architecture: String,
    pub runtime: RuntimeSpec,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_query_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub norm_epsilon: f64,
    pub layer_specs: Vec<DecoderLayerSpec>,
    /// Scale applied immediately after token embedding lookup.
    pub embedding_scale: f64,
    /// Optional tanh cap applied after the language-model head.
    pub final_logit_softcap: Option<f64>,
    /// Multiplicative scale applied after the language-model head and before
    /// any optional final-logit softcap.
    pub final_logit_scale: f64,
    pub projection_bias: ProjectionBias,
    /// Whether all three gated-MLP projections include a bias tensor.
    pub mlp_bias: bool,
    pub tied_embeddings: bool,
}

impl DecoderSpec {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.architecture.is_empty(),
            "architecture cannot be empty"
        );
        self.runtime.validate()?;
        ensure!(self.hidden_size > 0, "hidden_size must be positive");
        ensure!(
            self.intermediate_size > 0,
            "intermediate_size must be positive"
        );
        ensure!(self.num_layers > 0, "num_layers must be positive");
        ensure!(
            self.num_layers <= MAX_DECODER_LAYERS,
            "decoder layer count {} exceeds the supported limit {MAX_DECODER_LAYERS}",
            self.num_layers
        );
        let attention_layers = self
            .layer_specs
            .iter()
            .filter(|layer| layer.mixer.attention().is_some())
            .count();
        if attention_layers != 0 {
            ensure!(self.num_query_heads > 0, "num_query_heads must be positive");
            ensure!(self.num_kv_heads > 0, "num_kv_heads must be positive");
        }
        ensure!(
            self.norm_epsilon.is_finite() && self.norm_epsilon > 0.0,
            "norm_epsilon must be finite and positive"
        );
        ensure!(
            self.layer_specs.len() == self.num_layers,
            "decoder layer-spec count {} does not match layer count {}",
            self.layer_specs.len(),
            self.num_layers
        );
        ensure!(
            self.embedding_scale.is_finite() && self.embedding_scale > 0.0,
            "embedding scale must be finite and positive"
        );
        if let Some(softcap) = self.final_logit_softcap {
            ensure!(
                softcap.is_finite() && softcap > 0.0,
                "final-logit softcap must be finite and positive"
            );
        }
        ensure!(
            self.final_logit_scale.is_finite() && self.final_logit_scale > 0.0,
            "final-logit scale must be finite and positive"
        );
        let mut routed_expert_instances = 0_usize;
        for (index, layer) in self.layer_specs.iter().enumerate() {
            self.validate_layer(index, layer)?;
            if let FeedForwardKind::SparseMoe(moe) = layer.feed_forward {
                routed_expert_instances = routed_expert_instances
                    .checked_add(moe.expert_count)
                    .ok_or_else(|| anyhow::anyhow!("total routed expert count overflows usize"))?;
            }
        }
        ensure!(
            routed_expert_instances <= MAX_ROUTED_EXPERT_INSTANCES,
            "decoder declares {routed_expert_instances} routed expert instances, exceeding the limit {MAX_ROUTED_EXPERT_INSTANCES}"
        );
        if attention_layers != 0 {
            ensure!(
                self.num_query_heads.is_multiple_of(self.num_kv_heads),
                "query heads ({}) must be divisible by KV heads ({})",
                self.num_query_heads,
                self.num_kv_heads
            );
            ensure!(
                self.num_query_heads.checked_mul(self.head_dim).is_some(),
                "query attention width overflows usize"
            );
            ensure!(
                self.num_kv_heads.checked_mul(self.head_dim).is_some(),
                "KV attention width overflows usize"
            );
        }
        let f32_kv_bytes = self.kv_bytes_per_token(std::mem::size_of::<f32>())?;
        ensure!(
            f32_kv_bytes <= MAX_F32_KV_BYTES_PER_TOKEN,
            "F32 KV cache requires {f32_kv_bytes} bytes per token, exceeding the current limit {MAX_F32_KV_BYTES_PER_TOKEN}"
        );
        Ok(())
    }

    #[allow(clippy::too_many_lines)] // Keeps every per-layer invariant in one audit point.
    fn validate_layer(&self, index: usize, layer: &DecoderLayerSpec) -> Result<()> {
        match &layer.mixer {
            SequenceMixerSpec::Attention(attention)
            | SequenceMixerSpec::ParallelAttentionMamba2(ParallelAttentionMamba2Spec {
                attention,
                ..
            }) => {
                if let PositionEncodingSpec::Rope(rope) = &attention.position_encoding {
                    ensure!(
                        rope.rotary_dim > 0
                            && rope.rotary_dim <= self.head_dim
                            && rope.rotary_dim.is_multiple_of(2),
                        "layer {index} RoPE dimension {} must be a positive even value no larger than head_dim {}",
                        rope.rotary_dim,
                        self.head_dim
                    );
                    validate_rope_table_shape(self.runtime.max_context, rope.rotary_dim)?;
                    ensure!(
                        rope.theta.is_finite() && rope.theta > 0.0,
                        "layer {index} RoPE theta must be finite and positive"
                    );
                    if let Some(factors) = &rope.frequency_factors {
                        ensure!(
                            factors.len() == rope.rotary_dim / 2,
                            "layer {index} RoPE frequency factor count {} does not match half the head dimension {}",
                            factors.len(),
                            rope.rotary_dim / 2
                        );
                        ensure!(
                            factors
                                .iter()
                                .all(|factor| factor.is_finite() && *factor > 0.0),
                            "layer {index} RoPE frequency factors must be finite and positive"
                        );
                    }
                }
                ensure!(
                    attention.scale.is_finite() && attention.scale > 0.0,
                    "layer {index} attention scale must be finite and positive"
                );
                if attention.output_gate {
                    self.num_query_heads
                        .checked_mul(self.head_dim)
                        .and_then(|width| width.checked_mul(2))
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "layer {index} gated query projection width overflows usize"
                            )
                        })?;
                }
                if let Some(softcap) = attention.logit_softcap {
                    ensure!(
                        softcap.is_finite() && softcap > 0.0,
                        "layer {index} attention-logit softcap must be finite and positive"
                    );
                }
                if let AttentionKind::SlidingCausal { window } = attention.kind {
                    ensure!(window > 0, "layer {index} sliding window must be positive");
                }
            }
            SequenceMixerSpec::ShortConvolution(shortconv) => {
                ensure!(
                    shortconv.kernel_size > 0,
                    "layer {index} short-convolution kernel size must be positive"
                );
                ensure!(
                    self.hidden_size.checked_mul(3).is_some(),
                    "layer {index} short-convolution projection width overflows usize"
                );
            }
            SequenceMixerSpec::GatedDeltaNet(delta) => {
                ensure!(
                    delta.convolution_kernel_size > 0,
                    "layer {index} Gated DeltaNet convolution kernel must be positive"
                );
                ensure!(
                    delta.num_key_heads > 0 && delta.num_value_heads > 0,
                    "layer {index} Gated DeltaNet head counts must be positive"
                );
                ensure!(
                    delta.num_value_heads.is_multiple_of(delta.num_key_heads),
                    "layer {index} Gated DeltaNet value heads must be divisible by key heads"
                );
                ensure!(
                    delta.key_head_dim > 0 && delta.value_head_dim > 0,
                    "layer {index} Gated DeltaNet head dimensions must be positive"
                );
                ensure!(
                    delta.key_head_dim == delta.value_head_dim,
                    "layer {index} Gated DeltaNet currently requires equal key/value head dimensions"
                );
                ensure!(
                    delta.qk_norm_epsilon.is_finite() && delta.qk_norm_epsilon > 0.0,
                    "layer {index} Gated DeltaNet Q/K norm epsilon must be finite and positive"
                );
                let key_width = delta
                    .num_key_heads
                    .checked_mul(delta.key_head_dim)
                    .ok_or_else(|| {
                        anyhow::anyhow!("layer {index} Gated DeltaNet key width overflows usize")
                    })?;
                let value_width = delta
                    .num_value_heads
                    .checked_mul(delta.value_head_dim)
                    .ok_or_else(|| {
                        anyhow::anyhow!("layer {index} Gated DeltaNet value width overflows usize")
                    })?;
                key_width
                    .checked_mul(2)
                    .and_then(|width| width.checked_add(value_width))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "layer {index} Gated DeltaNet convolution width overflows usize"
                        )
                    })?;
                delta
                    .num_value_heads
                    .checked_mul(delta.key_head_dim)
                    .and_then(|elements| elements.checked_mul(delta.value_head_dim))
                    .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or_else(|| {
                        anyhow::anyhow!("layer {index} Gated DeltaNet state size overflows usize")
                    })?;
            }
        }
        if let SequenceMixerSpec::ParallelAttentionMamba2(parallel) = &layer.mixer {
            let mamba = parallel.mamba2;
            ensure!(
                parallel.attention.kind == AttentionKind::FullCausal,
                "layer {index} parallel attention/Mamba2 currently requires full causal attention"
            );
            ensure!(
                mamba.convolution_kernel_size > 0,
                "layer {index} Mamba2 convolution kernel must be positive"
            );
            ensure!(
                mamba.inner_size > 0
                    && mamba.state_size > 0
                    && mamba.num_heads > 0
                    && mamba.num_groups > 0
                    && mamba.head_dim > 0,
                "layer {index} Mamba2 dimensions and head/group counts must be positive"
            );
            ensure!(
                mamba.inner_size == mamba.num_heads.checked_mul(mamba.head_dim).unwrap_or(0),
                "layer {index} Mamba2 inner size must equal num_heads * head_dim"
            );
            ensure!(
                mamba.num_heads.is_multiple_of(mamba.num_groups),
                "layer {index} Mamba2 heads must be divisible by groups"
            );
            let convolution_width = mamba
                .num_groups
                .checked_mul(mamba.state_size)
                .and_then(|group_state| group_state.checked_mul(2))
                .and_then(|bc_width| bc_width.checked_add(mamba.inner_size))
                .ok_or_else(|| {
                    anyhow::anyhow!("layer {index} Mamba2 convolution width overflows usize")
                })?;
            mamba
                .inner_size
                .checked_add(convolution_width)
                .and_then(|width| width.checked_add(mamba.num_heads))
                .ok_or_else(|| {
                    anyhow::anyhow!("layer {index} Mamba2 input projection width overflows usize")
                })?;
            mamba
                .inner_size
                .checked_mul(mamba.state_size)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| {
                    anyhow::anyhow!("layer {index} Mamba2 recurrent state size overflows usize")
                })?;
        }
        ensure!(
            layer.mixer_residual_scale.is_finite() && layer.mixer_residual_scale > 0.0,
            "layer {index} mixer residual scale must be finite and positive"
        );
        ensure!(
            layer.feed_forward_residual_scale.is_finite()
                && layer.feed_forward_residual_scale > 0.0,
            "layer {index} feed-forward residual scale must be finite and positive"
        );
        if let FeedForwardKind::SparseMoe(moe) = layer.feed_forward {
            ensure!(
                !self.mlp_bias,
                "layer {index} sparse MoE does not support expert projection biases"
            );
            ensure!(
                moe.expert_count > 0 && moe.expert_count <= MAX_ROUTED_EXPERTS,
                "layer {index} expert count {} must be in 1..={MAX_ROUTED_EXPERTS}",
                moe.expert_count
            );
            ensure!(
                moe.experts_per_token > 0 && moe.experts_per_token <= moe.expert_count,
                "layer {index} experts per token {} must be in 1..={}",
                moe.experts_per_token,
                moe.expert_count
            );
            ensure!(
                moe.expert_intermediate_size > 0,
                "layer {index} expert intermediate size must be positive"
            );
            ensure!(
                moe.expert_count
                    .checked_mul(moe.expert_intermediate_size)
                    .and_then(|value| value.checked_mul(self.hidden_size))
                    .is_some(),
                "layer {index} expert weight geometry overflows usize"
            );
        }
        Ok(())
    }

    #[must_use]
    pub fn kv_groups(&self) -> usize {
        self.num_query_heads
            .checked_div(self.num_kv_heads)
            .unwrap_or(0)
    }

    /// Bytes used by K and V for one token across all layers.
    pub fn kv_bytes_per_token(&self, element_size: usize) -> Result<usize> {
        let attention_layers = self
            .layer_specs
            .iter()
            .filter(|layer| layer.mixer.attention().is_some())
            .count();
        2_usize
            .checked_mul(attention_layers)
            .and_then(|value| value.checked_mul(self.num_kv_heads))
            .and_then(|value| value.checked_mul(self.head_dim))
            .and_then(|value| value.checked_mul(element_size))
            .ok_or_else(|| anyhow::anyhow!("KV bytes-per-token calculation overflows usize"))
    }

    /// Fixed per-sequence mixer state retained independently from the
    /// token-proportional attention K/V cache.
    ///
    /// Gated `DeltaNet` recurrent matrices are always F32; convolution tails
    /// and short-convolution history use the activation element size supplied
    /// by the caller.
    pub fn fixed_sequence_state_bytes(&self, activation_element_size: usize) -> Result<usize> {
        ensure!(
            activation_element_size > 0,
            "activation element size must be positive"
        );
        self.layer_specs.iter().try_fold(0_usize, |total, layer| {
            let layer_bytes = match &layer.mixer {
                SequenceMixerSpec::Attention(_) => 0,
                SequenceMixerSpec::ShortConvolution(shortconv) => shortconv
                    .kernel_size
                    .checked_sub(1)
                    .ok_or_else(|| {
                        anyhow::anyhow!("short-convolution kernel size must be positive")
                    })?
                    .checked_mul(self.hidden_size)
                    .and_then(|elements| elements.checked_mul(activation_element_size))
                    .ok_or_else(|| {
                        anyhow::anyhow!("short-convolution sequence state size overflows usize")
                    })?,
                SequenceMixerSpec::GatedDeltaNet(delta) => {
                    let key_width = delta
                        .num_key_heads
                        .checked_mul(delta.key_head_dim)
                        .ok_or_else(|| {
                            anyhow::anyhow!("Gated DeltaNet key width overflows usize")
                        })?;
                    let value_width = delta
                        .num_value_heads
                        .checked_mul(delta.value_head_dim)
                        .ok_or_else(|| {
                            anyhow::anyhow!("Gated DeltaNet value width overflows usize")
                        })?;
                    let convolution_width = key_width
                        .checked_mul(2)
                        .and_then(|width| width.checked_add(value_width))
                        .ok_or_else(|| {
                            anyhow::anyhow!("Gated DeltaNet convolution width overflows usize")
                        })?;
                    let convolution_tail = delta
                        .convolution_kernel_size
                        .checked_sub(1)
                        .ok_or_else(|| {
                            anyhow::anyhow!("Gated DeltaNet convolution kernel must be positive")
                        })?
                        .checked_mul(convolution_width)
                        .and_then(|elements| elements.checked_mul(activation_element_size))
                        .ok_or_else(|| {
                            anyhow::anyhow!("Gated DeltaNet convolution state size overflows usize")
                        })?;
                    let recurrent = delta
                        .num_value_heads
                        .checked_mul(delta.key_head_dim)
                        .and_then(|elements| elements.checked_mul(delta.value_head_dim))
                        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
                        .ok_or_else(|| {
                            anyhow::anyhow!("Gated DeltaNet recurrent state size overflows usize")
                        })?;
                    convolution_tail.checked_add(recurrent).ok_or_else(|| {
                        anyhow::anyhow!("Gated DeltaNet sequence state size overflows usize")
                    })?
                }
                SequenceMixerSpec::ParallelAttentionMamba2(parallel) => {
                    let mamba = parallel.mamba2;
                    let convolution_width = mamba
                        .num_groups
                        .checked_mul(mamba.state_size)
                        .and_then(|group_state| group_state.checked_mul(2))
                        .and_then(|bc_width| bc_width.checked_add(mamba.inner_size))
                        .ok_or_else(|| {
                            anyhow::anyhow!("Mamba2 convolution width overflows usize")
                        })?;
                    let convolution_tail = mamba
                        .convolution_kernel_size
                        .checked_sub(1)
                        .ok_or_else(|| {
                            anyhow::anyhow!("Mamba2 convolution kernel must be positive")
                        })?
                        .checked_mul(convolution_width)
                        .and_then(|elements| elements.checked_mul(activation_element_size))
                        .ok_or_else(|| {
                            anyhow::anyhow!("Mamba2 convolution state size overflows usize")
                        })?;
                    let recurrent = mamba
                        .inner_size
                        .checked_mul(mamba.state_size)
                        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
                        .ok_or_else(|| {
                            anyhow::anyhow!("Mamba2 recurrent state size overflows usize")
                        })?;
                    convolution_tail.checked_add(recurrent).ok_or_else(|| {
                        anyhow::anyhow!("Mamba2 sequence state size overflows usize")
                    })?
                }
            };
            total
                .checked_add(layer_bytes)
                .ok_or_else(|| anyhow::anyhow!("sequence state size overflows usize"))
        })
    }
}

/// Mutable state owned by one autoregressive sequence.
///
/// Model weights and execution topology live on [`CausalLm`]. Keeping cache
/// state behind a separate, type-erased object lets one immutable model serve
/// multiple independent sequences without assigning model-global sequence IDs.
pub trait CausalLmSession: Any + Debug {
    /// Expose the concrete session type to its owning model implementation.
    /// Models must still validate that the session belongs to that particular
    /// model instance before using its state.
    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Reset logical sequence state while retaining reusable allocations.
    fn clear_cache(&mut self);

    /// Record that all previously submitted device work visible to this
    /// session has completed.
    ///
    /// Most callers establish completion through [`CausalLm::read_output_to_cpu`]
    /// or [`CausalLm::synchronize`]. Generation also has a best-effort drop
    /// path that owns only the device and session; bounded shared-cache
    /// implementations use this hook after that direct device drain so retired
    /// pages do not remain quarantined forever. The default is a no-op for
    /// sessions without asynchronous shared storage.
    fn mark_device_synchronized(&mut self) {}

    /// Absolute number of tokens processed by this sequence.
    fn cache_len(&self) -> usize;

    /// Optional per-layer cache diagnostics. Architectures without K/V state
    /// may keep the default `None`.
    fn kv_cache_stats(&self) -> Option<Vec<crate::cache::KvCacheStats>> {
        None
    }
}

/// Result of resetting one session for a new prompt.
///
/// `cached_tokens` is the exact number of leading prompt tokens whose model
/// forward was skipped. Implementations must retain at least one uncached
/// prompt token unless they also restore the logits needed for first-token
/// sampling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheMatch {
    pub cached_tokens: usize,
}

/// Model-wide automatic-prefix-cache counters and physical page sharing.
///
/// Snapshot cost is expressed in KV-block equivalents: paged K/V owners count
/// directly, while immutable recurrent tensors are charged by their logical
/// bytes rounded up to the same block byte size. A recurrent-only topology has
/// no comparable KV geometry, so each endpoint is additionally charged for its
/// logical prefix span. Physical counters are de-duplicated at the shared-pool
/// level, so callers can distinguish logical owners from actual resident pages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheDiagnostics {
    pub hits: u64,
    pub misses: u64,
    pub matched_tokens: u64,
    pub entries: usize,
    pub evictions: u64,
    pub resident_snapshot_blocks: usize,
    pub block_tokens: usize,
    pub physical_live_blocks: usize,
    pub physical_live_references: usize,
    pub physical_shared_blocks: usize,
    pub physical_max_blocks: usize,
}

/// One lane in a packed mixed-sequence model call.
///
/// Every lane keeps its own mutable session and absolute position. The current
/// contract accepts a positive, lane-local token count; implementations pack
/// those ragged token runs without padding or fake cache entries.
pub struct CausalLmBatchInput<'a> {
    pub(crate) session: &'a mut dyn CausalLmSession,
    pub(crate) input_ids: &'a [u32],
    pub(crate) start_pos: usize,
}

/// Host-only description of one sequence's next K/V growth request.
///
/// Serving builds these from immutable generation previews before it creates a
/// fail-closed prepared quantum. A model may reserve shared cache blocks for
/// the whole batch without uploading input IDs or mutating visible sequence
/// state.
pub struct CausalLmKvReservationInput<'a> {
    session: &'a dyn CausalLmSession,
    start_pos: usize,
    input_tokens: usize,
}

impl<'a> CausalLmKvReservationInput<'a> {
    #[must_use]
    pub fn new(session: &'a dyn CausalLmSession, start_pos: usize, input_tokens: usize) -> Self {
        Self {
            session,
            start_pos,
            input_tokens,
        }
    }

    #[must_use]
    pub fn session(&self) -> &dyn CausalLmSession {
        self.session
    }

    #[must_use]
    pub const fn start_pos(&self) -> usize {
        self.start_pos
    }

    #[must_use]
    pub const fn input_tokens(&self) -> usize {
        self.input_tokens
    }
}

/// Opaque host-side ownership of blocks reserved for one future model call.
///
/// Implementations must roll the reservation back from `Drop` if it was never
/// consumed. The token is deliberately model-specific: scheduler code may
/// carry it between preview and submission but cannot inspect physical block
/// IDs or mutate sequence cache state.
pub trait CausalLmKvReservation: Any + Debug {
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

/// Recoverable shared-cache pressure reported before any model submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheExhausted {
    pub requested_blocks: usize,
    pub free_blocks: usize,
    pub retired_blocks: usize,
    pub total_blocks: usize,
    pub block_tokens: usize,
}

impl fmt::Display for KvCacheExhausted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "KV cache requires {} additional blocks, but only {} of {} are free ({} retired, {} tokens/block)",
            self.requested_blocks,
            self.free_blocks,
            self.total_blocks,
            self.retired_blocks,
            self.block_tokens
        )
    }
}

impl std::error::Error for KvCacheExhausted {}

/// Reservation failure classification used at the scheduler boundary.
///
/// `Exhausted` is ordinary, request-scoped pressure and is safe to defer or
/// reject without draining the device. `Fatal` means host validation or
/// physical storage creation failed; callers must treat it like a model
/// submission failure rather than silently retrying.
#[derive(Debug)]
pub enum KvCacheReservationError {
    Exhausted(KvCacheExhausted),
    Fatal(candle_core::Error),
}

impl fmt::Display for KvCacheReservationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted(error) => fmt::Display::fmt(error, formatter),
            Self::Fatal(error) => write!(formatter, "KV cache reservation failed: {error}"),
        }
    }
}

impl std::error::Error for KvCacheReservationError {}

impl From<candle_core::Error> for KvCacheReservationError {
    fn from(error: candle_core::Error) -> Self {
        Self::Fatal(error)
    }
}

/// Model-wide shared-cache capacity snapshot for admission and diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheCapacity {
    pub block_tokens: usize,
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub live_blocks: usize,
    pub retired_blocks: usize,
}

impl<'a> CausalLmBatchInput<'a> {
    #[must_use]
    pub fn new(
        session: &'a mut dyn CausalLmSession,
        input_ids: &'a [u32],
        start_pos: usize,
    ) -> Self {
        Self {
            session,
            input_ids,
            start_pos,
        }
    }

    #[must_use]
    pub fn input_ids(&self) -> &[u32] {
        self.input_ids
    }

    #[must_use]
    pub const fn start_pos(&self) -> usize {
        self.start_pos
    }

    #[must_use]
    pub fn session(&self) -> &dyn CausalLmSession {
        &*self.session
    }

    pub fn session_mut(&mut self) -> &mut dyn CausalLmSession {
        &mut *self.session
    }
}

/// Mixed-sequence execution shape implemented by a model runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalLmBatchCapability {
    /// The model accepts only one sequence per call.
    Unsupported,
    /// Ragged lane token runs are packed into shared model operators without
    /// padding, while attention and KV updates remain sequence-scoped.
    PackedRagged,
}

/// The small immutable-model interface consumed by generation and serving code.
pub trait CausalLm {
    fn runtime_spec(&self) -> &RuntimeSpec;

    /// Install the model-wide shared K/V pool policy before any session is
    /// created. Architectures without full-attention K/V state may ignore it;
    /// stateful implementations must reject reconfiguration after allocation
    /// or session creation rather than invalidating existing block tables.
    fn configure_kv_cache(
        &mut self,
        _config: crate::cache::PagedKvPoolConfig,
    ) -> candle_core::Result<()> {
        Ok(())
    }

    /// Select the attention implementation used for multi-row prefill chunks.
    /// Architectures without full attention may ignore it; the default keeps
    /// the native paged kernel.
    fn configure_prefill_attention(
        &mut self,
        _mode: crate::prefill_attention::PrefillAttentionMode,
    ) -> candle_core::Result<()> {
        Ok(())
    }

    /// Configure automatic prompt-prefix snapshots in logical token units.
    /// Zero disables the feature. Stateful runtimes that cannot yet restore a
    /// complete topology must reject a non-zero request rather than reuse only
    /// a misleading subset of their state.
    fn configure_prefix_cache(&mut self, capacity_tokens: usize) -> candle_core::Result<()> {
        if capacity_tokens != 0 {
            candle_core::bail!("this model topology does not support prefix caching");
        }
        Ok(())
    }

    /// Create empty mutable state for one independent sequence.
    fn new_session(&self) -> candle_core::Result<Box<dyn CausalLmSession>>;

    /// Verify that `session` was created by this exact model instance without
    /// mutating either object. Generation calls this before resetting state so
    /// accidentally crossing model/session owners fails without data loss.
    fn validate_session(&self, session: &dyn CausalLmSession) -> candle_core::Result<()>;

    /// Reset or restore sequence state for an exact prompt prefix.
    ///
    /// The default preserves the historical cold-start behavior. Prefix-cache
    /// capable runtimes override this hook and atomically attach only state
    /// owned by this exact model instance.
    fn reset_session_for_prompt(
        &self,
        session: &mut dyn CausalLmSession,
        _prompt_tokens: &[u32],
    ) -> candle_core::Result<PrefixCacheMatch> {
        self.validate_session(session)?;
        session.clear_cache();
        if session.cache_len() != 0 {
            candle_core::bail!("model session did not become empty before prompt admission");
        }
        Ok(PrefixCacheMatch::default())
    }

    /// Publish a prompt snapshot after the successful logits completion
    /// barrier. Cache admission is best-effort and must not turn an otherwise
    /// successful model forward into a request failure.
    fn publish_prompt_prefix(&self, _session: &dyn CausalLmSession, _prompt_tokens: &[u32]) {}

    /// Return process-local cache/page counters when automatic prefix caching
    /// is configured. The default distinguishes unsupported/disabled (`None`)
    /// from a diagnostics collection failure (`Err`).
    fn prefix_cache_diagnostics(&self) -> candle_core::Result<Option<PrefixCacheDiagnostics>> {
        Ok(None)
    }

    /// Run one sequence's prefill or decode step and return last-token logits
    /// `[1, vocab]`. `input_ids` must have batch size one, contain in-vocabulary
    /// token IDs, and `start_pos` must equal the supplied session's current
    /// absolute cache length. The distinct multi-sequence path below uses a
    /// separate batch contract rather than storing several sequences in one
    /// session.
    fn forward(
        &self,
        session: &mut dyn CausalLmSession,
        input_ids: &Tensor,
        start_pos: usize,
    ) -> candle_core::Result<Tensor>;

    /// Host-token convenience path used by generation and serving. The default
    /// implementation validates IDs before constructing a batch-one tensor;
    /// runtimes may override it to avoid redundant device-side validation.
    fn forward_tokens(
        &self,
        session: &mut dyn CausalLmSession,
        input_ids: &[u32],
        start_pos: usize,
    ) -> candle_core::Result<Tensor> {
        if input_ids
            .iter()
            .any(|&token| token as usize >= self.runtime_spec().vocab_size)
        {
            candle_core::bail!("input contains a token ID outside the model vocabulary");
        }
        let input = Tensor::from_slice(input_ids, (1, input_ids.len()), self.device())?;
        self.forward(session, &input, start_pos)
    }

    /// Run one host-token sequence and return its final-normalized last hidden
    /// state `[1, hidden]`. Validation and session advancement match
    /// [`Self::forward_tokens`]; successful output must be consumed through
    /// [`Self::read_output_to_cpu`]. Models that expose only logits keep the
    /// fail-closed default.
    fn forward_last_hidden_tokens(
        &self,
        _session: &mut dyn CausalLmSession,
        _input_ids: &[u32],
        _start_pos: usize,
    ) -> candle_core::Result<Tensor> {
        candle_core::bail!("last-hidden output is not supported by this model")
    }

    /// Which genuine mixed-sequence execution shape this model implements.
    ///
    /// Returning [`CausalLmBatchCapability::PackedRagged`] means one
    /// [`Self::forward_batch_tokens`] call executes shared model operators over
    /// multiple lanes. A serial loop over [`Self::forward_tokens`] must not
    /// advertise that capability.
    fn mixed_batch_capability(&self) -> CausalLmBatchCapability {
        CausalLmBatchCapability::Unsupported
    }

    /// Current model-wide shared K/V capacity, when the runtime uses a bounded
    /// block pool. A contiguous or otherwise unbounded implementation may keep
    /// the default `None`.
    fn kv_cache_capacity(&self) -> Option<KvCacheCapacity> {
        None
    }

    /// Reserve cache growth for an upcoming mixed-sequence submission.
    ///
    /// This hook is host-side and precedes fail-closed generation preparation.
    /// `Exhausted` therefore guarantees that no input upload, cache write, or
    /// visible sequence mutation occurred. Models without a bounded shared
    /// pool return `Ok(None)` and use their existing forward path.
    fn reserve_batch_kv(
        &self,
        _inputs: &[CausalLmKvReservationInput<'_>],
    ) -> std::result::Result<Option<Box<dyn CausalLmKvReservation>>, KvCacheReservationError> {
        Ok(None)
    }

    /// Execute one ragged token quantum for multiple independent sessions.
    ///
    /// The result is last-token logits shaped `[lanes, vocab]`. Implementations
    /// must validate every lane before mutation. If this call returns an error
    /// after mutation, every participating session's logical cache state must
    /// be restored; physical writes and allocations may remain reusable. An
    /// error surfaced by the later final-logits completion/readback requires
    /// the caller to discard or poison the runtime. The default rejects the
    /// operation instead of hiding a serial fallback behind this API.
    fn forward_batch_tokens(
        &self,
        _inputs: &mut [CausalLmBatchInput<'_>],
    ) -> candle_core::Result<Tensor> {
        candle_core::bail!("mixed-sequence batching is not supported by this model")
    }

    /// Run one packed ragged quantum and return each lane's final-normalized
    /// last hidden state `[lanes, hidden]` in submission order. Validation,
    /// rollback and session advancement match [`Self::forward_batch_tokens`];
    /// successful output must be consumed through [`Self::read_output_to_cpu`].
    /// Models that expose only logits keep the fail-closed default.
    fn forward_batch_last_hidden_tokens(
        &self,
        _inputs: &mut [CausalLmBatchInput<'_>],
    ) -> candle_core::Result<Tensor> {
        candle_core::bail!("batched last-hidden output is not supported by this model")
    }

    /// Execute a batch with the exact host-side K/V reservation obtained from
    /// [`Self::reserve_batch_kv`]. The default only accepts `None`, preserving
    /// compatibility for runtimes whose cache does not require scheduling.
    fn forward_batch_tokens_reserved(
        &self,
        inputs: &mut [CausalLmBatchInput<'_>],
        reservation: Option<Box<dyn CausalLmKvReservation>>,
    ) -> candle_core::Result<Tensor> {
        if reservation.is_some() {
            candle_core::bail!("this model does not accept an external KV cache reservation");
        }
        self.forward_batch_tokens(inputs)
    }

    /// Materialize any model output on the host after a successful forward.
    ///
    /// Returning from this hook is the successful-forward completion barrier:
    /// the output and all model work it depends on must be complete.
    fn read_output_to_cpu(&self, output: &Tensor) -> candle_core::Result<Tensor> {
        output.to_device(&Device::Cpu)
    }

    /// Drain previously submitted work when no successful output completion
    /// barrier is available, such as request cleanup after a forward error.
    /// Runtimes may override this when synchronization is not represented
    /// solely by [`Self::device`].
    fn synchronize(&self) -> candle_core::Result<()> {
        self.device().synchronize()
    }

    fn device(&self) -> &Device;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> DecoderSpec {
        let head_dim = 4;
        let num_layers = 2;
        let rope = RopeSpec {
            theta: 10_000.0,
            layout: RopeLayout::HalfSplit,
            rotary_dim: head_dim,
            frequency_factors: None,
        };
        DecoderSpec {
            architecture: "test".into(),
            runtime: RuntimeSpec {
                vocab_size: 32,
                max_context: 64,
                stop_token_ids: vec![2],
            },
            hidden_size: 16,
            intermediate_size: 32,
            num_layers,
            num_query_heads: 4,
            num_kv_heads: 2,
            head_dim,
            norm_epsilon: 1e-5,
            layer_specs: vec![DecoderLayerSpec::standard(rope, head_dim); num_layers],
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
    fn validates_head_geometry() {
        let mut invalid = spec();
        invalid.num_query_heads = 3;
        assert!(invalid.validate().is_err());
        assert!(spec().validate().is_ok());
    }

    #[test]
    fn accepts_an_independent_lm_head() {
        let mut untied = spec();
        untied.tied_embeddings = false;
        assert!(untied.validate().is_ok());
    }

    #[test]
    fn accepts_attention_width_distinct_from_hidden_size() {
        let mut wider_attention = spec();
        wider_attention.head_dim = 6;
        for layer in &mut wider_attention.layer_specs {
            layer
                .attention_mut()
                .expect("test layer uses attention")
                .scale = 1.0 / 6.0_f64.sqrt();
        }
        assert_eq!(
            wider_attention.num_query_heads * wider_attention.head_dim,
            24
        );
        assert_eq!(wider_attention.hidden_size, 16);
        assert!(wider_attention.validate().is_ok());
    }

    #[test]
    fn rejects_invalid_runtime_metadata_and_odd_rope_heads() {
        let mut invalid_stop = spec();
        invalid_stop.runtime.stop_token_ids = vec![32];
        assert!(invalid_stop.validate().is_err());

        let mut odd_head = spec();
        odd_head.head_dim = 3;
        odd_head.num_query_heads = 5;
        odd_head.hidden_size = 15;
        odd_head.num_kv_heads = 1;
        odd_head.layer_specs.iter_mut().for_each(|layer| {
            layer
                .attention_mut()
                .expect("test layer uses attention")
                .scale = 1.0 / 3.0_f64.sqrt();
        });
        assert!(odd_head.validate().is_err());

        let mut infinite_norm = spec();
        infinite_norm.norm_epsilon = f64::INFINITY;
        assert!(infinite_norm.validate().is_err());
    }

    #[test]
    fn computes_compact_gqa_cache_size() {
        // 2 (K,V) * 2 layers * 2 KV heads * 4 dims * 2-byte F16.
        assert_eq!(spec().kv_bytes_per_token(2).unwrap(), 64);
    }

    #[test]
    fn reports_fixed_recurrent_and_convolution_state_per_sequence() {
        let mut hybrid = spec();
        hybrid.layer_specs[0].mixer = SequenceMixerSpec::GatedDeltaNet(GatedDeltaNetSpec {
            convolution_kernel_size: 4,
            num_key_heads: 2,
            num_value_heads: 4,
            key_head_dim: 2,
            value_head_dim: 2,
            qk_norm_epsilon: 1e-6,
        });
        hybrid.layer_specs[1].mixer =
            SequenceMixerSpec::ShortConvolution(ShortConvolutionSpec { kernel_size: 3 });

        // GDN: 3 * (2 * 2 * 2 + 4 * 2) F16 conv bytes +
        //      4 * 2 * 2 F32 recurrent bytes = 160.
        // Short convolution: 2 * hidden(16) F16 bytes = 64.
        assert_eq!(hybrid.fixed_sequence_state_bytes(2).unwrap(), 224);
        assert_eq!(hybrid.kv_bytes_per_token(2).unwrap(), 0);
        assert!(hybrid.fixed_sequence_state_bytes(0).is_err());
    }

    #[test]
    fn validates_sparse_moe_geometry_and_runtime_scales() {
        let mut valid = spec();
        valid.layer_specs[0].feed_forward = FeedForwardKind::SparseMoe(SparseMoeSpec {
            expert_count: 8,
            experts_per_token: 2,
            expert_intermediate_size: 24,
            router_weighting: RouterWeighting::SelectedSoftmax,
        });
        valid.layer_specs[0].mixer_residual_scale = 0.22;
        valid.layer_specs[0].feed_forward_residual_scale = 0.22;
        valid.embedding_scale = 12.0;
        valid.final_logit_scale = 1.0 / 6.0;
        assert!(valid.validate().is_ok());

        let mut biased = valid.clone();
        biased.mlp_bias = true;
        assert!(biased.validate().is_err());

        for mutate in [
            |moe: &mut SparseMoeSpec| moe.expert_count = 0,
            |moe: &mut SparseMoeSpec| moe.experts_per_token = 0,
            |moe: &mut SparseMoeSpec| moe.experts_per_token = moe.expert_count + 1,
            |moe: &mut SparseMoeSpec| moe.expert_intermediate_size = 0,
        ] {
            let mut invalid = valid.clone();
            let FeedForwardKind::SparseMoe(ref mut moe) = invalid.layer_specs[0].feed_forward
            else {
                unreachable!();
            };
            mutate(moe);
            assert!(invalid.validate().is_err());
        }

        for scale in [0.0, f64::NAN, f64::INFINITY] {
            let mut invalid = valid.clone();
            invalid.layer_specs[0].mixer_residual_scale = scale;
            assert!(invalid.validate().is_err());

            let mut invalid = valid.clone();
            invalid.layer_specs[0].feed_forward_residual_scale = scale;
            assert!(invalid.validate().is_err());

            let mut invalid = valid.clone();
            invalid.final_logit_scale = scale;
            assert!(invalid.validate().is_err());
        }

        let mut too_many = spec();
        too_many.num_layers = 65;
        too_many.layer_specs = vec![
            DecoderLayerSpec {
                feed_forward: FeedForwardKind::SparseMoe(SparseMoeSpec {
                    expert_count: 1_024,
                    experts_per_token: 1,
                    expert_intermediate_size: 1,
                    router_weighting: RouterWeighting::SelectedSoftmax,
                }),
                ..too_many.layer_specs[0].clone()
            };
            too_many.num_layers
        ];
        assert!(too_many.validate().is_err());
    }

    #[test]
    fn bounds_eager_rope_but_allows_long_context_dynamic_cache() {
        let mut oversized_rope = spec();
        oversized_rope.runtime.max_context = 1 << 20;
        oversized_rope.head_dim = 128;
        oversized_rope.num_query_heads = 1;
        oversized_rope.num_kv_heads = 1;
        for layer in &mut oversized_rope.layer_specs {
            let attention = layer.attention_mut().expect("test layer uses attention");
            attention.scale = 1.0 / 128.0_f64.sqrt();
            attention
                .rope_mut()
                .expect("test layer uses RoPE")
                .rotary_dim = 128;
        }
        assert!(
            oversized_rope
                .validate()
                .unwrap_err()
                .to_string()
                .contains("RoPE table")
        );

        let mut oversized_cache = spec();
        oversized_cache.runtime.max_context = 1 << 20;
        oversized_cache.num_layers = 4096;
        oversized_cache.num_query_heads = 1;
        oversized_cache.num_kv_heads = 1;
        oversized_cache.head_dim = 32;
        oversized_cache.layer_specs = vec![
            DecoderLayerSpec::standard(
                RopeSpec {
                    theta: 10_000.0,
                    layout: RopeLayout::HalfSplit,
                    rotary_dim: 32,
                    frequency_factors: None,
                },
                32,
            );
            oversized_cache.num_layers
        ];
        assert!(oversized_cache.validate().is_ok());
        assert_eq!(oversized_cache.kv_bytes_per_token(4).unwrap(), 1 << 20);

        let mut too_many_layers = spec();
        too_many_layers.num_layers = MAX_DECODER_LAYERS + 1;
        too_many_layers.layer_specs = vec![
            DecoderLayerSpec::standard(
                RopeSpec {
                    theta: 10_000.0,
                    layout: RopeLayout::HalfSplit,
                    rotary_dim: too_many_layers.head_dim,
                    frequency_factors: None,
                },
                too_many_layers.head_dim,
            );
            too_many_layers.num_layers
        ];
        assert!(
            too_many_layers
                .validate()
                .unwrap_err()
                .to_string()
                .contains("layer count")
        );

        let mut oversized_per_token = spec();
        oversized_per_token.num_layers = MAX_DECODER_LAYERS;
        oversized_per_token.num_query_heads = 1;
        oversized_per_token.num_kv_heads = 1;
        oversized_per_token.head_dim = 2_048;
        oversized_per_token.layer_specs = vec![
            DecoderLayerSpec::standard(
                RopeSpec {
                    theta: 10_000.0,
                    layout: RopeLayout::HalfSplit,
                    rotary_dim: oversized_per_token.head_dim,
                    frequency_factors: None,
                },
                oversized_per_token.head_dim,
            );
            oversized_per_token.num_layers
        ];
        assert!(
            oversized_per_token
                .validate()
                .unwrap_err()
                .to_string()
                .contains("bytes per token")
        );
    }
}

// ---------------------------------------------------------------------------
// Declarative weight names
// ---------------------------------------------------------------------------

/// Declarative semantic names requested from a [`WeightSource`].
///
/// Architecture adapters may compose a different table without teaching the
/// shared decoder about a model family. Execution topology remains entirely in
/// [`DecoderSpec`] and [`DecoderLayerSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecoderWeightNames {
    pub(crate) model: &'static str,
    pub(crate) embedding: &'static str,
    pub(crate) layers: &'static str,
    pub(crate) final_norm: &'static str,
    pub(crate) lm_head: &'static str,
    pub(crate) layer: DecoderLayerWeightNames,
    pub(crate) attention: AttentionWeightNames,
    pub(crate) dense_feed_forward: DenseFeedForwardWeightNames,
    pub(crate) sparse_moe: SparseMoeWeightNames,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecoderLayerWeightNames {
    pub(crate) mixer_norm: &'static str,
    pub(crate) feed_forward_norm: &'static str,
    pub(crate) pre_feed_forward_norm: &'static str,
    pub(crate) post_attention_norm: &'static str,
    pub(crate) post_feed_forward_norm: &'static str,
    pub(crate) attention: &'static str,
    pub(crate) feed_forward: &'static str,
    pub(crate) short_convolution: &'static str,
    pub(crate) gated_delta: Option<&'static str>,
    pub(crate) parallel_mamba2: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionWeightNames {
    pub(crate) query: &'static str,
    pub(crate) key: &'static str,
    pub(crate) value: &'static str,
    pub(crate) output: &'static str,
    pub(crate) query_norm: &'static str,
    pub(crate) key_norm: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseFeedForwardWeightNames {
    pub(crate) gate: &'static str,
    pub(crate) up: &'static str,
    pub(crate) down: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseMoeWeightNames {
    pub(crate) router: &'static str,
    pub(crate) experts: &'static str,
    pub(crate) gate: &'static str,
    pub(crate) up: &'static str,
    pub(crate) down: &'static str,
}

pub(crate) const STANDARD_DECODER_WEIGHT_NAMES: DecoderWeightNames = DecoderWeightNames {
    model: "model",
    embedding: "embed_tokens",
    layers: "layers",
    final_norm: "norm",
    lm_head: "lm_head",
    layer: DecoderLayerWeightNames {
        mixer_norm: "input_layernorm",
        feed_forward_norm: "post_attention_layernorm",
        pre_feed_forward_norm: "pre_feedforward_layernorm",
        post_attention_norm: "post_attention_norm",
        post_feed_forward_norm: "post_feedforward_layernorm",
        attention: "self_attn",
        feed_forward: "mlp",
        short_convolution: "shortconv",
        gated_delta: None,
        parallel_mamba2: None,
    },
    attention: AttentionWeightNames {
        query: "q_proj",
        key: "k_proj",
        value: "v_proj",
        output: "o_proj",
        query_norm: "q_norm",
        key_norm: "k_norm",
    },
    dense_feed_forward: DenseFeedForwardWeightNames {
        gate: "gate_proj",
        up: "up_proj",
        down: "down_proj",
    },
    sparse_moe: SparseMoeWeightNames {
        router: "router",
        experts: "experts",
        gate: "gate_proj",
        up: "up_proj",
        down: "down_proj",
    },
};

/// Weight names of the Falcon-H1 hybrid attention/Mamba2 trunk.
///
/// The table lives here so that the shared decoder and the model adapters
/// agree on one source of truth without depending on each other.
pub(crate) const FALCON_H1_WEIGHT_NAMES: DecoderWeightNames = DecoderWeightNames {
    layer: DecoderLayerWeightNames {
        mixer_norm: "attn_norm",
        feed_forward_norm: "ffn_norm",
        parallel_mamba2: Some("mamba"),
        ..STANDARD_DECODER_WEIGHT_NAMES.layer
    },
    ..STANDARD_DECODER_WEIGHT_NAMES
};

/// Weight names of the Qwen3.5 gated-delta trunk.
///
/// The table lives here so that the shared decoder and the model adapters
/// agree on one source of truth without depending on each other.
pub(crate) const QWEN35_WEIGHT_NAMES: DecoderWeightNames = DecoderWeightNames {
    layer: DecoderLayerWeightNames {
        mixer_norm: "attn_norm",
        feed_forward_norm: "attn_post_norm",
        gated_delta: Some("linear_attn"),
        ..STANDARD_DECODER_WEIGHT_NAMES.layer
    },
    ..STANDARD_DECODER_WEIGHT_NAMES
};
