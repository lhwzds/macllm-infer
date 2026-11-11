use std::{
    cell::{Cell, RefCell},
    cmp::Ordering,
    rc::Rc,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{RmsNorm, VarBuilder};

use crate::{
    architecture::{
        Activation, AttentionKind, AttentionSpec, CausalLm, CausalLmBatchCapability,
        CausalLmBatchInput, CausalLmKvReservation, CausalLmKvReservationInput, CausalLmSession,
        DecoderLayerSpec, DecoderSpec, FeedForwardKind, GatedDeltaNetSpec, KvCacheCapacity,
        KvCacheExhausted, KvCacheReservationError, Mamba2Spec, NormLayout, PrefixCacheDiagnostics,
        PrefixCacheMatch, RopeLayout, RopeSpec, RouterWeighting, SequenceMixerSpec,
        ShortConvolutionSpec, SparseMoeSpec,
    },
    cache::{
        KvCacheStats, KvSegment, PagedKvBatchReservation, PagedKvBlockTable, PagedKvGeometry,
        PagedKvGrowthRequest, PagedKvPoolConfig, PagedKvPoolError, SharedPagedKvPool,
        SlidingCacheCheckpoint, SlidingKvCache,
    },
    gated_delta::{decay_factors, decode_recurrence, prefill_recurrence},
    model_weights::{
        DenseWeightSource, LinearOp, TokenEmbedding, WeightPath, WeightSource, embedding,
        expert_linears_no_bias, linear_b, linear_no_bias, rms_norm,
    },
    prefill_attention::{PrefillAttentionMode, sdpa_causal_attention, sdpa_supported},
    prefix_cache::TokenPrefixCache,
};

// The declarative weight-name tables live in the architecture module so that
// the decoder and the model adapters share them without a cycle.
pub(crate) use crate::architecture::{
    AttentionWeightNames, DecoderLayerWeightNames, DecoderWeightNames, DenseFeedForwardWeightNames,
    STANDARD_DECODER_WEIGHT_NAMES, SparseMoeWeightNames,
};

#[cfg(test)]
use crate::cache::LayerKvCache;

const SLIDING_QUERY_TILE: usize = 128;
// Recurrent-only decoders have no paged-KV geometry from which to inherit an
// index block size. Keep their hash/admission granularity equal to the shared
// decoder's default page geometry.
const RECURRENT_PREFIX_CACHE_BLOCK_TOKENS: usize = 16;
// The correctness-first router synchronizes one F32 logit matrix to the host.
// Bound that temporary before the matmul; larger prefill batches will move to
// a chunked/device-side router rather than risking an allocator abort.
const MAX_HOST_ROUTER_LOGIT_ELEMENTS: usize = 16 << 20;

#[cfg(test)]
thread_local! {
    static FULL_CAUSAL_MASK_CONSTRUCTIONS: Cell<usize> = const { Cell::new(0) };
    static FORCE_CPU_DIRECT_PAGED_ATTENTION: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
fn reset_full_causal_mask_construction_count() {
    FULL_CAUSAL_MASK_CONSTRUCTIONS.set(0);
}

#[cfg(test)]
fn full_causal_mask_construction_count() -> usize {
    FULL_CAUSAL_MASK_CONSTRUCTIONS.get()
}

#[cfg(test)]
struct ForceCpuDirectPagedAttention {
    previous: bool,
}

#[cfg(test)]
impl ForceCpuDirectPagedAttention {
    fn enable() -> Self {
        Self {
            previous: FORCE_CPU_DIRECT_PAGED_ATTENTION.replace(true),
        }
    }
}

#[cfg(test)]
impl Drop for ForceCpuDirectPagedAttention {
    fn drop(&mut self) {
        FORCE_CPU_DIRECT_PAGED_ATTENTION.set(self.previous);
    }
}

#[derive(Debug)]
struct RotaryEmbedding {
    inv_frequency: Tensor,
    tables: Mutex<Option<RotaryTables>>,
    max_context: usize,
    dtype: DType,
    device: Device,
    layout: RopeLayout,
    rotary_dim: usize,
}

#[derive(Debug)]
struct RotaryTables {
    cos: Tensor,
    sin: Tensor,
    capacity: usize,
}

impl RotaryEmbedding {
    fn new(
        rope: &RopeSpec,
        head_dim: usize,
        max_context: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if rope.rotary_dim == 0 || rope.rotary_dim > head_dim {
            candle_core::bail!(
                "RoPE dimension {} must be in 1..={head_dim}",
                rope.rotary_dim
            );
        }
        let inv_frequency = (0..rope.rotary_dim)
            .step_by(2)
            .enumerate()
            .map(|(frequency_index, index)| {
                let base_frequency = 1.0 / rope.theta.powf(index as f64 / rope.rotary_dim as f64);
                let frequency = match &rope.frequency_factors {
                    Some(factors) => (base_frequency / f64::from(factors[frequency_index])) as f32,
                    None => base_frequency as f32,
                };
                if !frequency.is_finite() || frequency <= 0.0 {
                    candle_core::bail!(
                        "RoPE inverse frequency {frequency_index} is not finite and positive"
                    );
                }
                Ok(frequency)
            })
            .collect::<Result<Vec<_>>>()?;
        let half_dim = inv_frequency.len();
        Ok(Self {
            inv_frequency: Tensor::from_vec(inv_frequency, (1, half_dim), device)?,
            tables: Mutex::new(None),
            max_context,
            dtype,
            device: device.clone(),
            layout: rope.layout,
            rotary_dim: rope.rotary_dim,
        })
    }

    fn tables(&self, required: usize) -> Result<(Tensor, Tensor)> {
        const MINIMUM_TABLE_CAPACITY: usize = 256;

        let mut tables = self
            .tables
            .lock()
            .map_err(|_| candle_core::Error::Msg("RoPE table cache is poisoned".into()))?;
        if let Some(existing) = tables.as_ref().filter(|tables| tables.capacity >= required) {
            return Ok((existing.cos.clone(), existing.sin.clone()));
        }
        let geometric = required
            .checked_next_power_of_two()
            .unwrap_or(self.max_context);
        let capacity = geometric
            .max(MINIMUM_TABLE_CAPACITY.min(self.max_context))
            .min(self.max_context);
        if capacity < required {
            candle_core::bail!(
                "RoPE table capacity {capacity} cannot cover required position {required}"
            );
        }
        let positions = Tensor::arange(0_u32, capacity as u32, &self.device)?
            .to_dtype(DType::F32)?
            .reshape((capacity, 1))?;
        // Angles are computed in F32. Geometric growth keeps model-load memory
        // proportional to positions actually used while retaining O(1) table
        // lookup and the checkpoint's full declared context limit.
        let frequencies = positions.matmul(&self.inv_frequency)?;
        let next = RotaryTables {
            cos: frequencies.cos()?.to_dtype(self.dtype)?,
            sin: frequencies.sin()?.to_dtype(self.dtype)?,
            capacity,
        };
        let result = (next.cos.clone(), next.sin.clone());
        *tables = Some(next);
        Ok(result)
    }

    fn apply(&self, query: &Tensor, key: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let sequence = query.dim(2)?;
        if start_pos + sequence > self.max_context {
            candle_core::bail!(
                "RoPE positions {}..{} exceed context limit {}",
                start_pos,
                start_pos + sequence,
                self.max_context
            );
        }
        let (cos_table, sin_table) = self.tables(start_pos + sequence)?;
        match self.layout {
            RopeLayout::HalfSplit => {
                let cos = cos_table.narrow(0, start_pos, sequence)?.contiguous()?;
                let sin = sin_table.narrow(0, start_pos, sequence)?.contiguous()?;
                let query_width = query.dim(3)?;
                let key_width = key.dim(3)?;
                if self.rotary_dim > query_width || self.rotary_dim > key_width {
                    candle_core::bail!(
                        "RoPE dimension {} exceeds query/key widths {query_width}/{key_width}",
                        self.rotary_dim
                    );
                }
                let query_rotary = query.narrow(3, 0, self.rotary_dim)?.contiguous()?;
                let key_rotary = key.narrow(3, 0, self.rotary_dim)?.contiguous()?;
                let query_rotary = candle_nn::rotary_emb::rope(&query_rotary, &cos, &sin)?;
                let key_rotary = candle_nn::rotary_emb::rope(&key_rotary, &cos, &sin)?;
                let query = if self.rotary_dim == query_width {
                    query_rotary
                } else {
                    Tensor::cat(
                        &[
                            &query_rotary,
                            &query.narrow(3, self.rotary_dim, query_width - self.rotary_dim)?,
                        ],
                        3,
                    )?
                };
                let key = if self.rotary_dim == key_width {
                    key_rotary
                } else {
                    Tensor::cat(
                        &[
                            &key_rotary,
                            &key.narrow(3, self.rotary_dim, key_width - self.rotary_dim)?,
                        ],
                        3,
                    )?
                };
                // Concatenating a rotated prefix with an untouched suffix can
                // leave a strided view that Candle's attention matmul rejects.
                Ok((query.contiguous()?, key.contiguous()?))
            }
        }
    }
}

#[derive(Debug)]
struct DenseFeedForward {
    gate: LinearOp,
    up: LinearOp,
    down: LinearOp,
    activation: Activation,
}

impl DenseFeedForward {
    fn load(
        spec: &DecoderSpec,
        activation: Activation,
        names: DenseFeedForwardWeightNames,
        weights: WeightPath<'_>,
    ) -> Result<Self> {
        Ok(Self {
            gate: linear_b(
                spec.hidden_size,
                spec.intermediate_size,
                spec.mlp_bias,
                weights.pp(names.gate),
            )?,
            up: linear_b(
                spec.hidden_size,
                spec.intermediate_size,
                spec.mlp_bias,
                weights.pp(names.up),
            )?,
            down: linear_b(
                spec.intermediate_size,
                spec.hidden_size,
                spec.mlp_bias,
                weights.pp(names.down),
            )?,
            activation,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(input)?;
        let gate = match self.activation {
            Activation::Silu => gate.silu()?,
            Activation::GeluTanh => gate.gelu()?,
        };
        gate.mul(&self.up.forward(input)?)?.apply(&self.down)
    }
}

#[derive(Debug)]
struct ExpertFeedForward {
    gate: LinearOp,
    up: LinearOp,
    down: LinearOp,
    activation: Activation,
}

impl ExpertFeedForward {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(input)?;
        let gate = match self.activation {
            Activation::Silu => gate.silu()?,
            Activation::GeluTanh => gate.gelu()?,
        };
        gate.mul(&self.up.forward(input)?)?.apply(&self.down)
    }
}

#[derive(Debug)]
struct SparseMoeFeedForward {
    router: LinearOp,
    experts: Vec<ExpertFeedForward>,
    experts_per_token: usize,
    router_weighting: RouterWeighting,
}

struct SelectedRoutes {
    rows: Vec<Vec<u32>>,
    weights: Vec<Vec<f32>>,
}

impl SparseMoeFeedForward {
    fn load(
        spec: &DecoderSpec,
        moe: SparseMoeSpec,
        activation: Activation,
        names: SparseMoeWeightNames,
        weights: WeightPath<'_>,
    ) -> Result<Self> {
        let router = linear_no_bias(spec.hidden_size, moe.expert_count, weights.pp(names.router))?;
        let expert_weights = weights.pp(names.experts);
        let gates = expert_linears_no_bias(
            moe.expert_count,
            spec.hidden_size,
            moe.expert_intermediate_size,
            expert_weights.pp(names.gate),
        )?;
        let ups = expert_linears_no_bias(
            moe.expert_count,
            spec.hidden_size,
            moe.expert_intermediate_size,
            expert_weights.pp(names.up),
        )?;
        let downs = expert_linears_no_bias(
            moe.expert_count,
            moe.expert_intermediate_size,
            spec.hidden_size,
            expert_weights.pp(names.down),
        )?;
        let experts = gates
            .into_iter()
            .zip(ups)
            .zip(downs)
            .map(|((gate, up), down)| ExpertFeedForward {
                gate,
                up,
                down,
                activation,
            })
            .collect::<Vec<_>>();
        if experts.len() != moe.expert_count {
            candle_core::bail!(
                "loaded {} routed experts, expected {}",
                experts.len(),
                moe.expert_count
            );
        }
        Ok(Self {
            router,
            experts,
            experts_per_token: moe.experts_per_token,
            router_weighting: moe.router_weighting,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, sequence, hidden) = input.dims3()?;
        let flat_tokens = batch.checked_mul(sequence).ok_or_else(|| {
            candle_core::Error::Msg("sparse MoE token geometry overflows usize".into())
        })?;
        if flat_tokens.checked_mul(hidden).is_none() {
            candle_core::bail!("sparse MoE input geometry overflows usize");
        }
        u32::try_from(flat_tokens).map_err(|_| {
            candle_core::Error::Msg(format!(
                "sparse MoE token count {flat_tokens} exceeds the u32 routing-index limit"
            ))
        })?;
        let router_elements = flat_tokens.checked_mul(self.experts.len()).ok_or_else(|| {
            candle_core::Error::Msg("sparse MoE router geometry overflows usize".into())
        })?;
        if router_elements > MAX_HOST_ROUTER_LOGIT_ELEMENTS {
            candle_core::bail!(
                "sparse MoE host router requires {router_elements} logits, exceeding the current limit {MAX_HOST_ROUTER_LOGIT_ELEMENTS}; use a shorter prefill until chunked routing is available"
            );
        }
        let flat = input.reshape((flat_tokens, hidden))?;
        let router_logits = self.router.forward(&flat)?.to_dtype(DType::F32)?;
        let routes = match self.router_weighting {
            RouterWeighting::SelectedSoftmax => {
                selected_softmax_routes(&router_logits, self.experts.len(), self.experts_per_token)?
            }
        };

        let mut output = flat.zeros_like()?;
        for (expert, (rows, weights)) in self
            .experts
            .iter()
            .zip(routes.rows.iter().zip(&routes.weights))
        {
            if rows.is_empty() {
                continue;
            }
            let rows = Tensor::new(rows.as_slice(), flat.device())?;
            let weights = Tensor::new(weights.as_slice(), flat.device())?
                .to_dtype(flat.dtype())?
                .reshape((weights.len(), 1))?;
            let selected = flat.index_select(&rows, 0)?;
            let contribution = expert.forward(&selected)?.broadcast_mul(&weights)?;
            output = output.index_add(&rows, &contribution, 0)?;
        }
        output.reshape((batch, sequence, hidden))
    }
}

/// Convert FP32 raw router logits into per-expert token rows and selected-only
/// softmax weights. This correctness-first path intentionally synchronizes the
/// small router output to the host; expert projections remain on the selected
/// execution backend and retain quantized storage.
fn selected_softmax_routes(
    router_logits: &Tensor,
    expert_count: usize,
    experts_per_token: usize,
) -> Result<SelectedRoutes> {
    let (token_count, width) = router_logits.dims2()?;
    if width != expert_count {
        candle_core::bail!("router produced {width} logits per token, expected {expert_count}");
    }
    if experts_per_token == 0 || experts_per_token > expert_count {
        candle_core::bail!("experts per token {experts_per_token} must be in 1..={expert_count}");
    }
    u32::try_from(token_count).map_err(|_| {
        candle_core::Error::Msg(format!(
            "router token count {token_count} exceeds the u32 routing-index limit"
        ))
    })?;
    let route_count = token_count
        .checked_mul(experts_per_token)
        .ok_or_else(|| candle_core::Error::Msg("sparse MoE route count overflows usize".into()))?;
    let logits = router_logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if let Some((index, _)) = logits
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        candle_core::bail!(
            "router produced non-finite logits for token row {}",
            index / expert_count
        );
    }

    let mut selected_experts = Vec::<u16>::new();
    selected_experts
        .try_reserve_exact(route_count)
        .map_err(candle_core::Error::wrap)?;
    let mut counts = vec![0_usize; expert_count];
    let mut candidates = Vec::<usize>::new();
    candidates
        .try_reserve_exact(expert_count)
        .map_err(candle_core::Error::wrap)?;
    for row_logits in logits.chunks_exact(expert_count) {
        candidates.clear();
        candidates.extend(0..expert_count);
        if experts_per_token < expert_count {
            candidates.select_nth_unstable_by(experts_per_token, |&left, &right| {
                compare_router_candidates(row_logits, left, right)
            });
            candidates.truncate(experts_per_token);
        }
        candidates
            .sort_unstable_by(|&left, &right| compare_router_candidates(row_logits, left, right));
        for &expert in &candidates {
            counts[expert] += 1;
            selected_experts.push(u16::try_from(expert).map_err(candle_core::Error::wrap)?);
        }
    }
    let mut route_rows = vec![Vec::new(); expert_count];
    let mut route_weights = vec![Vec::new(); expert_count];
    for ((rows, weights), count) in route_rows.iter_mut().zip(&mut route_weights).zip(counts) {
        rows.try_reserve_exact(count)
            .map_err(candle_core::Error::wrap)?;
        weights
            .try_reserve_exact(count)
            .map_err(candle_core::Error::wrap)?;
    }
    for (row, (row_logits, selected)) in logits
        .chunks_exact(expert_count)
        .zip(selected_experts.chunks_exact(experts_per_token))
        .enumerate()
    {
        let maximum = selected
            .iter()
            .map(|&expert| row_logits[usize::from(expert)])
            .fold(f32::NEG_INFINITY, f32::max);
        let denominator = selected
            .iter()
            .map(|&expert| (row_logits[usize::from(expert)] - maximum).exp())
            .sum::<f32>();
        if !denominator.is_finite() || denominator <= 0.0 {
            candle_core::bail!("router softmax normalization failed for token row {row}");
        }
        let row = u32::try_from(row).map_err(candle_core::Error::wrap)?;
        for &expert in selected {
            let expert = usize::from(expert);
            route_rows[expert].push(row);
            route_weights[expert].push((row_logits[expert] - maximum).exp() / denominator);
        }
    }
    Ok(SelectedRoutes {
        rows: route_rows,
        weights: route_weights,
    })
}

fn compare_router_candidates(logits: &[f32], left: usize, right: usize) -> Ordering {
    if logits[left] > logits[right] {
        Ordering::Less
    } else if logits[left] < logits[right] {
        Ordering::Greater
    } else {
        left.cmp(&right)
    }
}

#[derive(Debug)]
enum FeedForward {
    Dense(DenseFeedForward),
    SparseMoe(SparseMoeFeedForward),
}

impl FeedForward {
    fn load(
        spec: &DecoderSpec,
        layer_spec: &DecoderLayerSpec,
        names: &DecoderWeightNames,
        weights: WeightPath<'_>,
    ) -> Result<Self> {
        match layer_spec.feed_forward {
            FeedForwardKind::Dense => DenseFeedForward::load(
                spec,
                layer_spec.activation,
                names.dense_feed_forward,
                weights,
            )
            .map(Self::Dense),
            FeedForwardKind::SparseMoe(moe) => SparseMoeFeedForward::load(
                spec,
                moe,
                layer_spec.activation,
                names.sparse_moe,
                weights,
            )
            .map(Self::SparseMoe),
        }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(feed_forward) => feed_forward.forward(input),
            Self::SparseMoe(feed_forward) => feed_forward.forward(input),
        }
    }
}

#[derive(Debug)]
struct SelfAttention {
    query: LinearOp,
    key: LinearOp,
    value: LinearOp,
    output: LinearOp,
    query_norm: Option<RmsNorm>,
    key_norm: Option<RmsNorm>,
    rotary: Option<Arc<RotaryEmbedding>>,
    kind: AttentionKind,
    num_query_heads: usize,
    num_kv_heads: usize,
    kv_groups: usize,
    head_dim: usize,
    attention_width: usize,
    scale: f64,
    logit_softcap: Option<f64>,
    output_gate: bool,
}

enum PackedAttentionCache<'reservation, 'cache> {
    Full {
        reservation: &'reservation mut PagedKvBatchReservation,
        layer: usize,
    },
    Sliding(&'reservation mut [&'cache mut SlidingKvCache]),
}

#[derive(Debug)]
struct LazyFullCausalMasks<'a> {
    start_positions: &'a [usize],
    lane_lengths: &'a [usize],
    masks: Vec<Option<Tensor>>,
    dtype: DType,
    device: &'a Device,
}

impl<'a> LazyFullCausalMasks<'a> {
    fn new(
        start_positions: &'a [usize],
        lane_lengths: &'a [usize],
        dtype: DType,
        device: &'a Device,
    ) -> Self {
        Self {
            start_positions,
            lane_lengths,
            masks: std::iter::repeat_with(|| None)
                .take(start_positions.len())
                .collect(),
            dtype,
            device,
        }
    }

    fn lane_count(&self) -> usize {
        self.masks.len()
    }

    fn get_or_build(&mut self, lane: usize) -> Result<Option<&Tensor>> {
        let (&query_start, &query_len) = self
            .start_positions
            .get(lane)
            .zip(self.lane_lengths.get(lane))
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "full-causal mask lane {lane} exceeds {} lanes",
                    self.lane_count()
                ))
            })?;
        if query_len == 1 {
            return Ok(None);
        }
        if self.masks[lane].is_none() {
            let key_len = query_start.checked_add(query_len).ok_or_else(|| {
                candle_core::Error::Msg("full-causal mask length overflow".into())
            })?;
            #[cfg(test)]
            FULL_CAUSAL_MASK_CONSTRUCTIONS
                .set(FULL_CAUSAL_MASK_CONSTRUCTIONS.get().saturating_add(1));
            self.masks[lane] = attention_mask_for_range(
                AttentionKind::FullCausal,
                query_start,
                query_len,
                0,
                key_len,
                self.dtype,
                self.device,
            )?;
        }
        Ok(self.masks[lane].as_ref())
    }
}

fn direct_paged_attention_supported(query: &Tensor) -> bool {
    #[cfg(test)]
    if FORCE_CPU_DIRECT_PAGED_ATTENTION.get() {
        // The CPU custom op is the direct-page numerical oracle. Forcing only
        // this preflight lets unit tests observe the exact native routing
        // branch without requiring an accessible Metal device.
        return matches!(query.device(), Device::Cpu) && query.dtype() == DType::F32;
    }
    crate::paged_attention::metal_attention_supported(query)
}

impl SelfAttention {
    fn load(
        spec: &DecoderSpec,
        attention_spec: &AttentionSpec,
        rotary: Option<Arc<RotaryEmbedding>>,
        names: AttentionWeightNames,
        weights: WeightPath<'_>,
    ) -> Result<Self> {
        let bias = spec.projection_bias;
        let query_norm = attention_spec
            .qk_norm
            .then(|| {
                rms_norm(
                    spec.head_dim,
                    spec.norm_epsilon,
                    weights.pp(names.query_norm),
                )
            })
            .transpose()?;
        let key_norm = attention_spec
            .qk_norm
            .then(|| rms_norm(spec.head_dim, spec.norm_epsilon, weights.pp(names.key_norm)))
            .transpose()?;
        Ok(Self {
            query: linear_b(
                spec.hidden_size,
                spec.num_query_heads
                    * spec.head_dim
                    * if attention_spec.output_gate { 2 } else { 1 },
                bias.query,
                weights.pp(names.query),
            )?,
            key: linear_b(
                spec.hidden_size,
                spec.num_kv_heads * spec.head_dim,
                bias.key,
                weights.pp(names.key),
            )?,
            value: linear_b(
                spec.hidden_size,
                spec.num_kv_heads * spec.head_dim,
                bias.value,
                weights.pp(names.value),
            )?,
            output: linear_b(
                spec.num_query_heads * spec.head_dim,
                spec.hidden_size,
                bias.output,
                weights.pp(names.output),
            )?,
            query_norm,
            key_norm,
            rotary,
            kind: attention_spec.kind,
            num_query_heads: spec.num_query_heads,
            num_kv_heads: spec.num_kv_heads,
            kv_groups: spec.kv_groups(),
            head_dim: spec.head_dim,
            attention_width: spec.num_query_heads * spec.head_dim,
            scale: attention_spec.scale,
            logit_softcap: attention_spec.logit_softcap,
            output_gate: attention_spec.output_gate,
        })
    }

    #[cfg(test)]
    fn forward(
        &self,
        cache: &mut LayerKvCache,
        input: &Tensor,
        mask: Option<&Tensor>,
        start_pos: usize,
    ) -> Result<Tensor> {
        let sequence = input.dim(1)?;
        let (query, key, value, output_gate) = self.project_packed_qkv(input, sequence)?;
        let (query, key) = match &self.rotary {
            Some(rotary) => rotary.apply(&query, &key, start_pos)?,
            // RoPE produces fresh contiguous tensors. Preserve the same
            // matmul contract for NoPE layers instead of forwarding the
            // strided head-first transpose views.
            None => (query.contiguous()?, key.contiguous()?),
        };
        let attention = match (self.kind, cache) {
            (AttentionKind::FullCausal, LayerKvCache::Full(cache)) => {
                let (key, value) = cache.append(&key, &value)?;
                self.attention_values(&query, &key, &value, mask)?
            }
            (AttentionKind::SlidingCausal { window }, LayerKvCache::Sliding(cache)) => {
                let history = cache.history_segments()?;
                let attention =
                    self.sliding_attention(&query, &key, &value, &history, start_pos, window)?;
                cache.append(&key, &value)?;
                attention
            }
            _ => candle_core::bail!("attention kind and test cache policy do not match"),
        };
        let attention = attention
            .transpose(1, 2)?
            .reshape((1, sequence, self.attention_width))?;
        let attention = match output_gate {
            Some(gate) => attention.mul(&candle_nn::ops::sigmoid(&gate)?)?,
            None => attention,
        };
        attention.apply(&self.output)
    }

    #[allow(clippy::too_many_lines)] // Validation and both cache policies stay beside the writes they guard.
    fn forward_packed(
        &self,
        cache: PackedAttentionCache<'_, '_>,
        input: &Tensor,
        full_masks: Option<&mut LazyFullCausalMasks<'_>>,
        start_positions: &[usize],
        lane_lengths: &[usize],
        prefill_attention: PrefillAttentionMode,
    ) -> Result<Tensor> {
        let (container_batch, total_tokens, _) = input.dims3()?;
        if container_batch != 1 {
            candle_core::bail!(
                "packed attention requires batch container size 1, found {container_batch}"
            );
        }
        let lanes = match &cache {
            PackedAttentionCache::Full { reservation, .. } => reservation.lane_count(),
            PackedAttentionCache::Sliding(caches) => caches.len(),
        };
        if lanes == 0 || start_positions.len() != lanes || lane_lengths.len() != lanes {
            candle_core::bail!(
                "packed attention has {lanes} caches, {} start positions and {} lane lengths",
                start_positions.len(),
                lane_lengths.len(),
            );
        }
        if !matches!(
            (self.kind, &cache),
            (AttentionKind::FullCausal, PackedAttentionCache::Full { .. })
                | (
                    AttentionKind::SlidingCausal { .. },
                    PackedAttentionCache::Sliding(_)
                )
        ) {
            candle_core::bail!("attention kind does not match its packed KV cache policy");
        }
        match (&cache, full_masks.as_ref()) {
            (PackedAttentionCache::Full { .. }, Some(masks)) if masks.lane_count() == lanes => {}
            (PackedAttentionCache::Full { .. }, _) => {
                candle_core::bail!("full packed attention requires one lazy mask slot per lane")
            }
            (PackedAttentionCache::Sliding(_), None) => {}
            (PackedAttentionCache::Sliding(_), Some(_)) => {
                candle_core::bail!("sliding packed attention cannot receive full-causal masks")
            }
        }
        let packed_tokens = lane_lengths.iter().try_fold(0_usize, |total, &length| {
            if length == 0 {
                return Err(candle_core::Error::Msg(
                    "packed attention lane cannot be empty".into(),
                ));
            }
            total
                .checked_add(length)
                .ok_or_else(|| candle_core::Error::Msg("packed attention size overflow".into()))
        })?;
        if packed_tokens != total_tokens {
            candle_core::bail!(
                "packed attention lane lengths total {packed_tokens}, expected {total_tokens} tokens"
            );
        }
        let (query, key, value, output_gate) = self.project_packed_qkv(input, total_tokens)?;

        // Cache post-RoPE K and unmodified V at KV-head cardinality. A full
        // attention layer writes every lane before taking one immutable batch
        // view, so supported packed prefill/decode needs one page-table-aware
        // dispatch per layer instead of one dispatch per lane. Unsupported
        // batches fall back as a whole, preserving independent masks and page
        // materialization. Sliding attention keeps its bounded transactional
        // ring and lane-local history.
        let attention = match cache {
            PackedAttentionCache::Full { reservation, layer } => self
                .forward_packed_full_attention(
                    reservation,
                    layer,
                    &query,
                    &key,
                    &value,
                    full_masks.expect("full-attention mask slots were validated"),
                    start_positions,
                    lane_lengths,
                    prefill_attention,
                )?,
            PackedAttentionCache::Sliding(caches) => {
                let AttentionKind::SlidingCausal { window } = self.kind else {
                    unreachable!("packed attention cache policy was validated");
                };
                let mut lane_outputs = Vec::with_capacity(lanes);
                let mut offset = 0_usize;
                for lane in 0..lanes {
                    let lane_length = lane_lengths[lane];
                    let query = query.narrow(2, offset, lane_length)?;
                    let key = key.narrow(2, offset, lane_length)?;
                    let (query, key) = match &self.rotary {
                        Some(rotary) => rotary.apply(&query, &key, start_positions[lane])?,
                        None => (query.contiguous()?, key.contiguous()?),
                    };
                    let value = value.narrow(2, offset, lane_length)?;
                    let cache = &mut *caches[lane];
                    let history = cache.history_segments()?;
                    let attention = self.sliding_attention(
                        &query,
                        &key,
                        &value,
                        &history,
                        start_positions[lane],
                        window,
                    )?;
                    cache.append(&key, &value)?;
                    lane_outputs.push(attention);
                    offset += lane_length;
                }
                Tensor::cat(&lane_outputs, 2)?
            }
        };
        let attention =
            attention
                .transpose(1, 2)?
                .reshape((1, total_tokens, self.attention_width))?;
        let attention = match output_gate {
            Some(gate) => attention.mul(&candle_nn::ops::sigmoid(&gate)?)?,
            None => attention,
        };
        attention.apply(&self.output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_packed_full_attention(
        &self,
        reservation: &mut PagedKvBatchReservation,
        layer: usize,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        full_masks: &mut LazyFullCausalMasks<'_>,
        start_positions: &[usize],
        lane_lengths: &[usize],
        prefill_attention: PrefillAttentionMode,
    ) -> Result<Tensor> {
        let lanes = lane_lengths.len();
        let mut lane_queries = Vec::with_capacity(lanes);
        let mut all_native = true;
        let mut offset = 0_usize;
        for lane in 0..lanes {
            let lane_length = lane_lengths[lane];
            let lane_query = query.narrow(2, offset, lane_length)?;
            let lane_key = key.narrow(2, offset, lane_length)?;
            let (lane_query, lane_key) = match &self.rotary {
                Some(rotary) => rotary.apply(&lane_query, &lane_key, start_positions[lane])?,
                None => (lane_query.contiguous()?, lane_key.contiguous()?),
            };
            let lane_value = value.narrow(2, offset, lane_length)?;
            reservation
                .write_layer(lane, layer, &lane_key, &lane_value)
                .map_err(paged_kv_error)?;
            all_native &= direct_paged_attention_supported(&lane_query);
            lane_queries.push(lane_query);
            offset += lane_length;
        }

        let packed_query = Tensor::cat(&lane_queries, 2)?;
        let fused = prefill_attention.uses_fused_kernel();
        // The fused kernel serves the multi-row chunks it can express. When no
        // chunk qualifies, the batch keeps the default path, including its
        // single dispatch for the whole batch.
        let fused_serves_chunk = fused
            && lane_queries.iter().any(|lane_query| {
                lane_query.dim(2).is_ok_and(|length| length > 1)
                    && sdpa_supported(lane_query, self.logit_softcap)
            });
        // The default path picks one implementation for the whole batch: the
        // native paged kernel when every lane supports it, and the dense
        // fallback otherwise. The fused mode may only replace the multi-row
        // chunks, so the lanes that keep the native path follow that same
        // batch decision instead of picking an implementation of their own.
        let batch_uses_native = all_native
            && crate::paged_attention::metal_batch_attention_supported(&packed_query, lane_lengths);
        // The native paged kernel serves a whole batch in one dispatch. The
        // default mode keeps that dispatch for every batch it supports; the
        // fused mode reserves it for batches without a fused chunk and routes
        // the multi-row chunks to Candle's fused kernel instead.
        if all_native && !fused_serves_chunk {
            let native_attention = {
                let read = reservation
                    .layer_batch_read(layer)
                    .map_err(paged_kv_error)?;
                crate::paged_attention::paged_attention_batch(
                    &packed_query,
                    &read,
                    start_positions,
                    lane_lengths,
                    self.scale,
                    self.logit_softcap,
                )
            }?;
            if let Some(attention) = native_attention {
                return Ok(attention);
            }
        }

        let mut lane_outputs = Vec::with_capacity(lanes);
        let mut offset = 0_usize;
        for (lane, &lane_length) in lane_lengths.iter().enumerate() {
            let lane_query = packed_query.narrow(2, offset, lane_length)?;
            let attention =
                if fused && lane_length > 1 && sdpa_supported(&lane_query, self.logit_softcap) {
                    // The fused kernel reads contiguous K/V, so gather this lane's
                    // history from the paged cache before dispatching it. The
                    // gathered history ends with the chunk, so the kernel's causal
                    // diagonal covers exactly the lane's own prefix.
                    let (key, value) = reservation
                        .materialize_layer(lane, layer)
                        .map_err(paged_kv_error)?;
                    // The cache hands over strided views, but the fused kernel
                    // reads contiguous K/V, so the history is made contiguous
                    // here.
                    sdpa_causal_attention(
                        &lane_query,
                        &key.contiguous()?,
                        &value.contiguous()?,
                        self.scale,
                    )?
                } else if fused && batch_uses_native {
                    // The batch as a whole could have used the native paged
                    // kernel, so this lane keeps it: single-row decode and the
                    // geometries the fused kernel cannot express.
                    self.native_lane_attention(
                        reservation,
                        layer,
                        lane,
                        &lane_query,
                        start_positions[lane],
                        full_masks,
                    )?
                } else {
                    // The batch as a whole cannot use the native paged kernel, so
                    // every lane keeps the dense fallback of the default path.
                    let mask = full_masks.get_or_build(lane)?;
                    self.dense_lane_attention(reservation, layer, lane, &lane_query, mask)?
                };
            lane_outputs.push(attention);
            offset += lane_length;
        }
        Tensor::cat(&lane_outputs, 2)
    }

    /// Native paged attention for one lane, with the dense path as fallback.
    fn native_lane_attention(
        &self,
        reservation: &mut PagedKvBatchReservation,
        layer: usize,
        lane: usize,
        lane_query: &Tensor,
        start_position: usize,
        full_masks: &mut LazyFullCausalMasks<'_>,
    ) -> Result<Tensor> {
        let native = {
            let read = reservation
                .layer_read(lane, layer)
                .map_err(paged_kv_error)?;
            crate::paged_attention::paged_attention(
                lane_query,
                &read,
                start_position,
                self.scale,
                self.logit_softcap,
            )?
        };
        if let Some(attention) = native {
            Ok(attention)
        } else {
            let mask = full_masks.get_or_build(lane)?;
            self.dense_lane_attention(reservation, layer, lane, lane_query, mask)
        }
    }

    /// Dense full-causal attention for one lane over its materialized history.
    fn dense_lane_attention(
        &self,
        reservation: &PagedKvBatchReservation,
        layer: usize,
        lane: usize,
        lane_query: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (key, value) = reservation
            .materialize_layer(lane, layer)
            .map_err(paged_kv_error)?;
        self.attention_values(lane_query, &key, &value, mask)
    }

    fn project_packed_qkv(
        &self,
        input: &Tensor,
        total_tokens: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>)> {
        let projected_query = self.query.forward(input)?;
        let (query, output_gate) = if self.output_gate {
            let projected_query = projected_query.reshape((
                1,
                total_tokens,
                self.num_query_heads,
                self.head_dim * 2,
            ))?;
            let query = projected_query
                .narrow(3, 0, self.head_dim)?
                .force_contiguous()?;
            let gate = projected_query
                .narrow(3, self.head_dim, self.head_dim)?
                .force_contiguous()?
                .reshape((1, total_tokens, self.attention_width))?;
            (query, Some(gate))
        } else {
            (
                projected_query.reshape((1, total_tokens, self.num_query_heads, self.head_dim))?,
                None,
            )
        };
        let key = self.key.forward(input)?.reshape((
            1,
            total_tokens,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let value = self
            .value
            .forward(input)?
            .reshape((1, total_tokens, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let query = match &self.query_norm {
            Some(norm) => query.flatten(0, 2)?.apply(norm)?.reshape((
                1,
                total_tokens,
                self.num_query_heads,
                self.head_dim,
            ))?,
            None => query,
        };
        let key = match &self.key_norm {
            Some(norm) => key.flatten(0, 2)?.apply(norm)?.reshape((
                1,
                total_tokens,
                self.num_kv_heads,
                self.head_dim,
            ))?,
            None => key,
        };
        Ok((
            query.transpose(1, 2)?,
            key.transpose(1, 2)?,
            value,
            output_gate,
        ))
    }

    fn attention_values(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        dense_causal_attention(
            query,
            key,
            value,
            self.kv_groups,
            self.scale,
            self.logit_softcap,
            mask,
        )
    }

    fn sliding_attention(
        &self,
        query: &Tensor,
        current_key: &Tensor,
        current_value: &Tensor,
        history: &[KvSegment],
        start_pos: usize,
        window: usize,
    ) -> Result<Tensor> {
        let sequence = query.dim(2)?;
        let key_start = history
            .first()
            .map_or(start_pos, |segment| segment.absolute_start);
        let mut expected_start = key_start;
        for segment in history {
            if segment.absolute_start != expected_start {
                candle_core::bail!(
                    "sliding KV history is not contiguous at absolute position {}",
                    expected_start
                );
            }
            expected_start = expected_start
                .checked_add(segment.len())
                .ok_or_else(|| candle_core::Error::Msg("sliding KV range overflow".into()))?;
        }
        if expected_start != start_pos {
            candle_core::bail!(
                "sliding KV history ends at {}, expected start_pos {}",
                expected_start,
                start_pos
            );
        }

        let key_end = start_pos
            .checked_add(sequence)
            .ok_or_else(|| candle_core::Error::Msg("sliding attention range overflow".into()))?;
        let mut keys = history
            .iter()
            .map(|segment| &segment.key)
            .collect::<Vec<_>>();
        let mut values = history
            .iter()
            .map(|segment| &segment.value)
            .collect::<Vec<_>>();
        keys.push(current_key);
        values.push(current_value);
        let key = if keys.len() == 1 {
            current_key.clone()
        } else {
            Tensor::cat(&keys, 2)?
        };
        let value = if values.len() == 1 {
            current_value.clone()
        } else {
            Tensor::cat(&values, 2)?
        };
        if key.dim(2)? != key_end - key_start || value.dim(2)? != key_end - key_start {
            candle_core::bail!("sliding K/V materialization has an unexpected length");
        }

        let history_limit = window - 1;
        let mut tiles = Vec::with_capacity(sequence.div_ceil(SLIDING_QUERY_TILE));
        for query_offset in (0..sequence).step_by(SLIDING_QUERY_TILE) {
            let query_len = (sequence - query_offset).min(SLIDING_QUERY_TILE);
            let query_start = start_pos + query_offset;
            let visible_start = query_start.saturating_sub(history_limit).max(key_start);
            let visible_end = (query_start + query_len).min(key_end);
            let visible_len = visible_end - visible_start;
            let query_tile = query.narrow(2, query_offset, query_len)?;
            let key_tile = key.narrow(2, visible_start - key_start, visible_len)?;
            let value_tile = value.narrow(2, visible_start - key_start, visible_len)?;
            let mask = attention_mask_for_range(
                AttentionKind::SlidingCausal { window },
                query_start,
                query_len,
                visible_start,
                visible_len,
                query.dtype(),
                query.device(),
            )?;
            tiles.push(self.attention_values(
                &query_tile,
                &key_tile,
                &value_tile,
                mask.as_ref(),
            )?);
        }
        if tiles.len() == 1 {
            Ok(tiles.pop().expect("one sliding attention tile"))
        } else {
            let tiles = tiles.iter().collect::<Vec<_>>();
            Tensor::cat(&tiles, 2)
        }
    }
}

#[derive(Debug, Clone)]
struct ShortConvolutionCheckpoint {
    tail: Option<Tensor>,
    logical_len: usize,
}

/// Mutable history for one short-convolution layer of one sequence.
///
/// `tail` is chronological `[1, kernel_size - 1, hidden]` projected `B * x`.
/// Tensor handles are immutable, so cloning this checkpoint is sufficient for
/// transactional rollback without copying device buffers.
#[derive(Debug)]
struct ShortConvolutionState {
    tail: Option<Tensor>,
    logical_len: usize,
    history_tokens: usize,
}

impl ShortConvolutionState {
    fn new(kernel_size: usize) -> Self {
        Self {
            tail: None,
            logical_len: 0,
            history_tokens: kernel_size - 1,
        }
    }

    fn checkpoint(&self) -> ShortConvolutionCheckpoint {
        ShortConvolutionCheckpoint {
            tail: self.tail.clone(),
            logical_len: self.logical_len,
        }
    }

    fn restore(&mut self, checkpoint: ShortConvolutionCheckpoint) {
        self.tail = checkpoint.tail;
        self.logical_len = checkpoint.logical_len;
    }

    fn clear(&mut self) {
        self.tail = None;
        self.logical_len = 0;
    }
}

/// Double-gated causal depthwise convolution used by hybrid decoders.
#[derive(Debug)]
struct ShortConvolution {
    input: LinearOp,
    kernel: Tensor,
    output: LinearOp,
    hidden_size: usize,
    kernel_size: usize,
}

impl ShortConvolution {
    fn load(
        model_spec: &DecoderSpec,
        spec: ShortConvolutionSpec,
        weights: WeightPath<'_>,
    ) -> Result<Self> {
        let projected = model_spec.hidden_size.checked_mul(3).ok_or_else(|| {
            candle_core::Error::Msg("short-convolution projection width overflow".into())
        })?;
        let kernel_weights = weights.pp("conv");
        let kernel = match kernel_weights.tensor_unchecked("weight") {
            Ok(kernel) => match kernel.dims() {
                [hidden_size, kernel_size]
                    if *hidden_size == model_spec.hidden_size
                        && *kernel_size == spec.kernel_size =>
                {
                    kernel
                }
                [hidden_size, channels, kernel_size]
                    if *hidden_size == model_spec.hidden_size
                        && *channels == 1
                        && *kernel_size == spec.kernel_size =>
                {
                    kernel.squeeze(1)?
                }
                actual => candle_core::bail!(
                    "short-convolution kernel must have shape [{}, {}] or [{}, 1, {}], found {actual:?}",
                    model_spec.hidden_size,
                    spec.kernel_size,
                    model_spec.hidden_size,
                    spec.kernel_size
                ),
            },
            Err(_) => {
                kernel_weights.tensor((model_spec.hidden_size, spec.kernel_size), "weight")?
            }
        };
        Ok(Self {
            input: linear_no_bias(model_spec.hidden_size, projected, weights.pp("in_proj"))?,
            kernel,
            output: linear_no_bias(
                model_spec.hidden_size,
                model_spec.hidden_size,
                weights.pp("out_proj"),
            )?,
            hidden_size: model_spec.hidden_size,
            kernel_size: spec.kernel_size,
        })
    }

    fn forward_packed(
        &self,
        states: &mut [&mut ShortConvolutionState],
        input: &Tensor,
        lane_lengths: &[usize],
    ) -> Result<Tensor> {
        let (container_batch, total_tokens, hidden_size) = input.dims3()?;
        if container_batch != 1 || hidden_size != self.hidden_size {
            candle_core::bail!(
                "packed short convolution expected [1, tokens, {}], found {:?}",
                self.hidden_size,
                input.dims()
            );
        }
        if states.is_empty() || states.len() != lane_lengths.len() {
            candle_core::bail!(
                "packed short convolution has {} states and {} lane lengths",
                states.len(),
                lane_lengths.len()
            );
        }
        let expected_tokens = lane_lengths.iter().try_fold(0_usize, |total, &length| {
            if length == 0 {
                return Err(candle_core::Error::Msg(
                    "packed short-convolution lane cannot be empty".into(),
                ));
            }
            total.checked_add(length).ok_or_else(|| {
                candle_core::Error::Msg("packed short-convolution size overflow".into())
            })
        })?;
        if expected_tokens != total_tokens {
            candle_core::bail!(
                "packed short-convolution lane lengths total {expected_tokens}, expected {total_tokens} tokens"
            );
        }

        let mut outputs = Vec::with_capacity(states.len());
        let mut offset = 0_usize;
        for (state, &lane_length) in states.iter_mut().zip(lane_lengths) {
            let lane = input.narrow(1, offset, lane_length)?;
            outputs.push(self.forward_lane(state, &lane)?);
            offset += lane_length;
        }
        let outputs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&outputs, 1)
    }

    fn forward_lane(&self, state: &mut ShortConvolutionState, input: &Tensor) -> Result<Tensor> {
        let (_, sequence, hidden_size) = input.dims3()?;
        if hidden_size != self.hidden_size || state.history_tokens != self.kernel_size - 1 {
            candle_core::bail!("short-convolution state geometry does not match its layer");
        }
        let projection = self.input.forward(input)?;
        let b = projection.narrow(2, 0, self.hidden_size)?;
        let c = projection.narrow(2, self.hidden_size, self.hidden_size)?;
        let x = projection.narrow(2, self.hidden_size * 2, self.hidden_size)?;
        let bx = b.mul(&x)?;

        let history_tokens = self.kernel_size - 1;
        let window = if history_tokens == 0 {
            bx.clone()
        } else {
            let history = match &state.tail {
                Some(tail) => {
                    let shape = tail.dims3()?;
                    if shape != (1, history_tokens, self.hidden_size) {
                        candle_core::bail!(
                            "short-convolution tail has shape {:?}, expected [1, {history_tokens}, {}]",
                            tail.dims(),
                            self.hidden_size
                        );
                    }
                    tail.clone()
                }
                None => Tensor::zeros(
                    (1, history_tokens, self.hidden_size),
                    bx.dtype(),
                    bx.device(),
                )?,
            };
            Tensor::cat(&[&history, &bx], 1)?
        };

        // This is depthwise Conv1d cross-correlation, written as K broadcast
        // multiply-adds. It preserves the checkpoint's oldest-to-newest
        // kernel order while avoiding the heavyweight generic Conv1d path for
        // the tiny recurrent kernels used by LFM2.
        let mut convolution: Option<Tensor> = None;
        for kernel_index in 0..self.kernel_size {
            let values = window.narrow(1, kernel_index, sequence)?;
            let weight = self
                .kernel
                .narrow(1, kernel_index, 1)?
                .squeeze(1)?
                .reshape((1, 1, self.hidden_size))?;
            let product = values.broadcast_mul(&weight)?;
            convolution = Some(match convolution {
                Some(sum) => sum.add(&product)?,
                None => product,
            });
        }
        let convolution = convolution.expect("validated short-convolution kernel is non-empty");
        let output = c.mul(&convolution)?.apply(&self.output)?;

        let next_tail = if history_tokens == 0 {
            None
        } else {
            Some(
                window
                    .narrow(1, window.dim(1)? - history_tokens, history_tokens)?
                    // A narrow view with batch size one can still report itself
                    // contiguous while retaining the complete prefill storage.
                    // Recurrent state must own only its fixed K-1 token tail.
                    .force_contiguous()?,
            )
        };
        let logical_len = state
            .logical_len
            .checked_add(sequence)
            .ok_or_else(|| candle_core::Error::Msg("short-convolution length overflow".into()))?;
        state.tail = next_tail;
        state.logical_len = logical_len;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct GatedDeltaNetCheckpoint {
    convolution_tail: Option<Tensor>,
    recurrent: Option<Tensor>,
    logical_len: usize,
}

/// Lane-local state for one Gated `DeltaNet` layer.
///
/// The convolution tail uses the model dtype; the delta-rule matrix is always
/// F32. Forward replaces both tensor handles instead of mutating storage in
/// place, so the common transaction checkpoint can restore a failed batch.
#[derive(Debug)]
struct GatedDeltaNetState {
    convolution_tail: Option<Tensor>,
    recurrent: Option<Tensor>,
    logical_len: usize,
    convolution_history_tokens: usize,
    convolution_width: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
}

impl GatedDeltaNetState {
    fn new(spec: GatedDeltaNetSpec) -> Result<Self> {
        let key_width = spec
            .num_key_heads
            .checked_mul(spec.key_head_dim)
            .ok_or_else(|| candle_core::Error::Msg("Gated DeltaNet key width overflow".into()))?;
        let value_width = spec
            .num_value_heads
            .checked_mul(spec.value_head_dim)
            .ok_or_else(|| candle_core::Error::Msg("Gated DeltaNet value width overflow".into()))?;
        let convolution_width = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
            .ok_or_else(|| {
                candle_core::Error::Msg("Gated DeltaNet convolution width overflow".into())
            })?;
        Ok(Self {
            convolution_tail: None,
            recurrent: None,
            logical_len: 0,
            convolution_history_tokens: spec.convolution_kernel_size - 1,
            convolution_width,
            num_value_heads: spec.num_value_heads,
            key_head_dim: spec.key_head_dim,
            value_head_dim: spec.value_head_dim,
        })
    }

    fn checkpoint(&self) -> GatedDeltaNetCheckpoint {
        GatedDeltaNetCheckpoint {
            convolution_tail: self.convolution_tail.clone(),
            recurrent: self.recurrent.clone(),
            logical_len: self.logical_len,
        }
    }

    fn restore(&mut self, checkpoint: GatedDeltaNetCheckpoint) {
        self.convolution_tail = checkpoint.convolution_tail;
        self.recurrent = checkpoint.recurrent;
        self.logical_len = checkpoint.logical_len;
    }

    fn clear(&mut self) {
        self.convolution_tail = None;
        self.recurrent = None;
        self.logical_len = 0;
    }
}

/// Correctness-first causal Gated `DeltaNet` implementation.
///
/// Packed projections are shared, while convolution and recurrence are split
/// into lane-local token streams. A future optimized chunk kernel can replace
/// `forward_lane` without changing the architecture or session contracts.
#[derive(Debug)]
struct GatedDeltaNet {
    qkv: LinearOp,
    output_gate: LinearOp,
    alpha: LinearOp,
    beta: LinearOp,
    convolution_kernel: Tensor,
    time_bias: Tensor,
    state_decay: Tensor,
    output_norm: RmsNorm,
    output: LinearOp,
    spec: GatedDeltaNetSpec,
    convolution_width: usize,
    key_width: usize,
    value_width: usize,
}

impl GatedDeltaNet {
    fn load(
        model_spec: &DecoderSpec,
        spec: GatedDeltaNetSpec,
        weights: WeightPath<'_>,
    ) -> Result<Self> {
        let key_width = spec
            .num_key_heads
            .checked_mul(spec.key_head_dim)
            .ok_or_else(|| candle_core::Error::Msg("Gated DeltaNet key width overflow".into()))?;
        let value_width = spec
            .num_value_heads
            .checked_mul(spec.value_head_dim)
            .ok_or_else(|| candle_core::Error::Msg("Gated DeltaNet value width overflow".into()))?;
        let convolution_width = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
            .ok_or_else(|| {
                candle_core::Error::Msg("Gated DeltaNet convolution width overflow".into())
            })?;
        let convolution_weights = weights.pp("conv1d");
        let convolution_kernel = match convolution_weights.tensor_unchecked("weight") {
            Ok(kernel) => match kernel.dims() {
                [channels, kernel_size]
                    if *channels == convolution_width
                        && *kernel_size == spec.convolution_kernel_size =>
                {
                    kernel
                }
                [channels, one, kernel_size]
                    if *channels == convolution_width
                        && *one == 1
                        && *kernel_size == spec.convolution_kernel_size =>
                {
                    kernel.squeeze(1)?
                }
                actual => candle_core::bail!(
                    "Gated DeltaNet convolution kernel must have shape [{convolution_width}, {}] or [{convolution_width}, 1, {}], found {actual:?}",
                    spec.convolution_kernel_size,
                    spec.convolution_kernel_size
                ),
            },
            Err(_) => convolution_weights
                .tensor((convolution_width, spec.convolution_kernel_size), "weight")?,
        };
        Ok(Self {
            qkv: linear_no_bias(
                model_spec.hidden_size,
                convolution_width,
                weights.pp("in_proj_qkv"),
            )?,
            output_gate: linear_no_bias(
                model_spec.hidden_size,
                value_width,
                weights.pp("in_proj_z"),
            )?,
            alpha: linear_no_bias(
                model_spec.hidden_size,
                spec.num_value_heads,
                weights.pp("in_proj_a"),
            )?,
            beta: linear_no_bias(
                model_spec.hidden_size,
                spec.num_value_heads,
                weights.pp("in_proj_b"),
            )?,
            convolution_kernel,
            time_bias: weights.tensor((spec.num_value_heads,), "dt_bias")?,
            // GGUF conversion stores `-exp(A_log)` directly at this path.
            state_decay: weights.tensor((spec.num_value_heads,), "A_log")?,
            output_norm: rms_norm(
                spec.value_head_dim,
                model_spec.norm_epsilon,
                weights.pp("norm"),
            )?,
            output: linear_no_bias(value_width, model_spec.hidden_size, weights.pp("out_proj"))?,
            spec,
            convolution_width,
            key_width,
            value_width,
        })
    }

    fn forward_packed(
        &self,
        states: &mut [&mut GatedDeltaNetState],
        input: &Tensor,
        lane_lengths: &[usize],
    ) -> Result<Tensor> {
        let (container_batch, total_tokens, _) = input.dims3()?;
        if container_batch != 1 || states.is_empty() || states.len() != lane_lengths.len() {
            candle_core::bail!(
                "packed Gated DeltaNet has batch {container_batch}, {} states and {} lane lengths",
                states.len(),
                lane_lengths.len()
            );
        }
        let expected_tokens = lane_lengths.iter().try_fold(0_usize, |total, &length| {
            if length == 0 {
                return Err(candle_core::Error::Msg(
                    "packed Gated DeltaNet lane cannot be empty".into(),
                ));
            }
            total.checked_add(length).ok_or_else(|| {
                candle_core::Error::Msg("packed Gated DeltaNet size overflow".into())
            })
        })?;
        if expected_tokens != total_tokens {
            candle_core::bail!(
                "packed Gated DeltaNet lane lengths total {expected_tokens}, expected {total_tokens} tokens"
            );
        }

        let qkv = self.qkv.forward(input)?;
        let output_gate = self.output_gate.forward(input)?;
        let alpha = self.alpha.forward(input)?;
        let beta = self.beta.forward(input)?;
        let mut outputs = Vec::with_capacity(states.len());
        let mut offset = 0_usize;
        for (state, &lane_length) in states.iter_mut().zip(lane_lengths) {
            outputs.push(self.forward_lane(
                state,
                &qkv.narrow(1, offset, lane_length)?,
                &output_gate.narrow(1, offset, lane_length)?,
                &alpha.narrow(1, offset, lane_length)?,
                &beta.narrow(1, offset, lane_length)?,
            )?);
            offset += lane_length;
        }
        let outputs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&outputs, 1)
    }

    #[allow(clippy::too_many_lines)] // The scalar recurrence remains linear and source-auditable.
    fn forward_lane(
        &self,
        state: &mut GatedDeltaNetState,
        qkv: &Tensor,
        output_gate: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        let (_, sequence, convolution_width) = qkv.dims3()?;
        if convolution_width != self.convolution_width
            || state.convolution_width != self.convolution_width
            || state.convolution_history_tokens != self.spec.convolution_kernel_size - 1
            || state.num_value_heads != self.spec.num_value_heads
            || state.key_head_dim != self.spec.key_head_dim
            || state.value_head_dim != self.spec.value_head_dim
        {
            candle_core::bail!("Gated DeltaNet state geometry does not match its layer");
        }

        let history_tokens = self.spec.convolution_kernel_size - 1;
        let convolution_window = if history_tokens == 0 {
            qkv.clone()
        } else {
            let history = match &state.convolution_tail {
                Some(tail) => {
                    let expected = (1, history_tokens, self.convolution_width);
                    if tail.dims3()? != expected {
                        candle_core::bail!(
                            "Gated DeltaNet convolution tail has shape {:?}, expected {expected:?}",
                            tail.dims()
                        );
                    }
                    tail.clone()
                }
                None => Tensor::zeros(
                    (1, history_tokens, self.convolution_width),
                    qkv.dtype(),
                    qkv.device(),
                )?,
            };
            Tensor::cat(&[&history, qkv], 1)?
        };
        let mut convolution: Option<Tensor> = None;
        for kernel_index in 0..self.spec.convolution_kernel_size {
            let values = convolution_window.narrow(1, kernel_index, sequence)?;
            let weight = self
                .convolution_kernel
                .narrow(1, kernel_index, 1)?
                .squeeze(1)?
                .reshape((1, 1, self.convolution_width))?;
            let product = values.broadcast_mul(&weight)?;
            convolution = Some(match convolution {
                Some(sum) => sum.add(&product)?,
                None => product,
            });
        }
        let convolution = convolution
            .expect("validated Gated DeltaNet convolution kernel is non-empty")
            .silu()?;

        let q = convolution.narrow(2, 0, self.key_width)?.reshape((
            sequence,
            self.spec.num_key_heads,
            self.spec.key_head_dim,
        ))?;
        let k = convolution
            .narrow(2, self.key_width, self.key_width)?
            .reshape((sequence, self.spec.num_key_heads, self.spec.key_head_dim))?;
        let v = convolution
            .narrow(2, self.key_width * 2, self.value_width)?
            .reshape((
                sequence,
                self.spec.num_value_heads,
                self.spec.value_head_dim,
            ))?
            .to_dtype(DType::F32)?;
        let repeat = self.spec.num_value_heads / self.spec.num_key_heads;
        // These transforms have no recurrence dependency. Computing them for
        // the complete lane avoids repeating Metal dispatches inside the host
        // token loop during prefill.
        let query = repeat_heads_tiled_dim(&q.to_dtype(DType::F32)?, repeat, 1)?;
        let query = l2_normalize_last(&query, self.spec.qk_norm_epsilon)?
            .affine(1.0 / (self.spec.key_head_dim as f64).sqrt(), 0.0)?;
        let key = repeat_heads_tiled_dim(&k.to_dtype(DType::F32)?, repeat, 1)?;
        let key = l2_normalize_last(&key, self.spec.qk_norm_epsilon)?;
        let alpha = alpha
            .reshape((sequence, self.spec.num_value_heads))?
            .to_dtype(DType::F32)?;
        let beta = candle_nn::ops::sigmoid(
            &beta
                .reshape((sequence, self.spec.num_value_heads))?
                .to_dtype(DType::F32)?,
        )?;
        let time_bias = self.time_bias.to_dtype(DType::F32)?;
        let state_decay = self.state_decay.to_dtype(DType::F32)?;
        let decay = match decay_factors(&alpha, &time_bias, &state_decay)? {
            Some(decay) => decay,
            None => softplus(&alpha.broadcast_add(&time_bias)?)?
                .broadcast_mul(&state_decay)?
                .exp()?,
        };

        let mut recurrent = match &state.recurrent {
            Some(recurrent) => {
                let expected = (
                    self.spec.num_value_heads,
                    self.spec.key_head_dim,
                    self.spec.value_head_dim,
                );
                if recurrent.dims3()? != expected || recurrent.dtype() != DType::F32 {
                    candle_core::bail!(
                        "Gated DeltaNet recurrent state has shape {:?}/{:?}, expected {expected:?}/F32",
                        recurrent.dims(),
                        recurrent.dtype()
                    );
                }
                recurrent.clone()
            }
            None => Tensor::zeros(
                (
                    self.spec.num_value_heads,
                    self.spec.key_head_dim,
                    self.spec.value_head_dim,
                ),
                DType::F32,
                qkv.device(),
            )?,
        };
        let mut token_outputs = Vec::with_capacity(sequence);
        let mut fused_recurrence = false;
        #[cfg(feature = "metal")]
        let use_batched_contractions = matches!(qkv.device(), Device::Metal(_));
        #[cfg(not(feature = "metal"))]
        let use_batched_contractions = false;
        if sequence == 1 {
            let token_query = query.squeeze(0)?;
            let token_key = key.squeeze(0)?;
            let token_value = v.squeeze(0)?;
            let token_decay = decay.squeeze(0)?;
            let token_beta = beta.squeeze(0)?;
            if let Some((token_output, next_recurrent)) = decode_recurrence(
                &token_query,
                &token_key,
                &token_value,
                &token_decay,
                &token_beta,
                &recurrent,
            )? {
                recurrent = next_recurrent;
                token_outputs.push(token_output.unsqueeze(0)?);
                fused_recurrence = true;
            }
        } else if let Some((outputs, next_recurrent)) =
            prefill_recurrence(&query, &key, &v, &decay, &beta, &recurrent)?
        {
            recurrent = next_recurrent;
            token_outputs.push(outputs);
            fused_recurrence = true;
        }
        // Unsupported devices and dtypes retain the same auditable tensor path.
        if token_outputs.is_empty() {
            for token in 0..sequence {
                let query = query.narrow(0, token, 1)?.squeeze(0)?;
                let key = key.narrow(0, token, 1)?.squeeze(0)?;
                let value = v.narrow(0, token, 1)?.squeeze(0)?;
                let token_decay = decay.narrow(0, token, 1)?.squeeze(0)?.reshape((
                    self.spec.num_value_heads,
                    1,
                    1,
                ))?;
                recurrent = recurrent.broadcast_mul(&token_decay)?;

                let key_column = key.unsqueeze(2)?;
                let memory = if use_batched_contractions {
                    // Metal batched contractions avoid materializing a full
                    // recurrent-state broadcast merely to reduce it again.
                    key.unsqueeze(1)?.matmul(&recurrent)?.squeeze(1)?
                } else {
                    // Preserve the original reduction order as the CPU
                    // numerical reference for long recurrent checkpoints.
                    recurrent.broadcast_mul(&key_column)?.sum(1)?
                };
                let beta = beta
                    .narrow(0, token, 1)?
                    .squeeze(0)?
                    .reshape((self.spec.num_value_heads, 1))?;
                let delta = value.sub(&memory)?.broadcast_mul(&beta)?;
                let update = if use_batched_contractions {
                    key_column.matmul(&delta.unsqueeze(1)?)?
                } else {
                    key_column.broadcast_mul(&delta.unsqueeze(1)?)?
                };
                recurrent = recurrent.add(&update)?;

                let token_output = if use_batched_contractions {
                    query.unsqueeze(1)?.matmul(&recurrent)?.squeeze(1)?
                } else {
                    recurrent.broadcast_mul(&query.unsqueeze(2)?)?.sum(1)?
                };
                token_outputs.push(token_output.unsqueeze(0)?);
            }
        }
        let token_outputs = if token_outputs.len() == 1 {
            token_outputs
                .pop()
                .expect("one Gated DeltaNet output was just counted")
        } else {
            let outputs = token_outputs.iter().collect::<Vec<_>>();
            Tensor::cat(&outputs, 0)?
        };
        let output = token_outputs
            .to_dtype(qkv.dtype())?
            .apply(&self.output_norm)?;
        let z = output_gate.reshape((
            sequence,
            self.spec.num_value_heads,
            self.spec.value_head_dim,
        ))?;
        let output = output
            .mul(&z.silu()?)?
            .reshape((1, sequence, self.value_width))?
            .apply(&self.output)?;

        let next_tail = if history_tokens == 0 {
            None
        } else {
            Some(
                convolution_window
                    .narrow(
                        1,
                        convolution_window.dim(1)? - history_tokens,
                        history_tokens,
                    )?
                    .force_contiguous()?,
            )
        };
        state.logical_len = state
            .logical_len
            .checked_add(sequence)
            .ok_or_else(|| candle_core::Error::Msg("Gated DeltaNet length overflow".into()))?;
        state.convolution_tail = next_tail;
        // The fused result stores `[state | token outputs]` in one allocation.
        // Detach the state prefix because Candle rounds the combined allocation
        // to a power of two: Qwen3.5's 1 MiB state plus even one output token
        // otherwise retains a 2 MiB buffer for every recurrent layer.
        state.recurrent = Some(if fused_recurrence {
            recurrent.force_contiguous()?
        } else {
            recurrent.contiguous()?
        });
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct Mamba2Checkpoint {
    convolution_tail: Option<Tensor>,
    recurrent: Option<Tensor>,
    logical_len: usize,
}

/// Lane-local Mamba2 state. Tensor handles are replaced after a successful
/// step so the decoder's common checkpoint can roll a failed packed batch back
/// without copying the recurrent matrix eagerly.
#[derive(Debug)]
struct Mamba2State {
    convolution_tail: Option<Tensor>,
    recurrent: Option<Tensor>,
    logical_len: usize,
    convolution_history_tokens: usize,
    convolution_width: usize,
    num_heads: usize,
    head_dim: usize,
    state_size: usize,
}

impl Mamba2State {
    fn new(spec: Mamba2Spec) -> Result<Self> {
        let convolution_width = mamba2_convolution_width(spec)?;
        Ok(Self {
            convolution_tail: None,
            recurrent: None,
            logical_len: 0,
            convolution_history_tokens: spec.convolution_kernel_size - 1,
            convolution_width,
            num_heads: spec.num_heads,
            head_dim: spec.head_dim,
            state_size: spec.state_size,
        })
    }

    fn checkpoint(&self) -> Mamba2Checkpoint {
        Mamba2Checkpoint {
            convolution_tail: self.convolution_tail.clone(),
            recurrent: self.recurrent.clone(),
            logical_len: self.logical_len,
        }
    }

    fn restore(&mut self, checkpoint: Mamba2Checkpoint) {
        self.convolution_tail = checkpoint.convolution_tail;
        self.recurrent = checkpoint.recurrent;
        self.logical_len = checkpoint.logical_len;
    }

    fn clear(&mut self) {
        self.convolution_tail = None;
        self.recurrent = None;
        self.logical_len = 0;
    }
}

/// Mamba2 selective state-space branch with a readable tensor reference path.
///
/// Projections and the causal depthwise convolution operate on a complete
/// packed lane. Eligible contiguous F32 Apple-Metal inputs use one fused,
/// token-serial scan operation; CPU and unsupported layouts retain the
/// primitive recurrence as the numerical reference.
#[derive(Debug)]
struct Mamba2 {
    input: LinearOp,
    convolution_kernel: Tensor,
    convolution_bias: Option<Tensor>,
    time_bias: Tensor,
    state_decay: Tensor,
    skip: Tensor,
    gated_norm_weight: Option<Tensor>,
    output: LinearOp,
    norm_epsilon: f64,
    spec: Mamba2Spec,
    convolution_width: usize,
    projection_width: usize,
}

impl Mamba2 {
    fn load(model_spec: &DecoderSpec, spec: Mamba2Spec, weights: WeightPath<'_>) -> Result<Self> {
        let convolution_width = mamba2_convolution_width(spec)?;
        let projection_width = spec
            .inner_size
            .checked_add(convolution_width)
            .and_then(|width| width.checked_add(spec.num_heads))
            .ok_or_else(|| candle_core::Error::Msg("Mamba2 projection width overflow".into()))?;
        let convolution = weights.pp("conv1d");
        let convolution_kernel = match convolution.tensor_unchecked("weight") {
            Ok(kernel) => match kernel.dims() {
                [channels, kernel_size]
                    if *channels == convolution_width
                        && *kernel_size == spec.convolution_kernel_size =>
                {
                    kernel
                }
                [channels, one, kernel_size]
                    if *channels == convolution_width
                        && *one == 1
                        && *kernel_size == spec.convolution_kernel_size =>
                {
                    kernel.squeeze(1)?
                }
                actual => candle_core::bail!(
                    "Mamba2 convolution kernel must have shape [{convolution_width}, {}] or [{convolution_width}, 1, {}], found {actual:?}",
                    spec.convolution_kernel_size,
                    spec.convolution_kernel_size
                ),
            },
            Err(_) => {
                convolution.tensor((convolution_width, spec.convolution_kernel_size), "weight")?
            }
        };
        let convolution_bias = spec
            .convolution_bias
            .then(|| convolution.tensor((convolution_width,), "bias"))
            .transpose()?;
        let group_width = spec.inner_size / spec.num_groups;
        let gated_norm_weight = spec
            .gated_rms_norm
            .then(|| {
                weights
                    .pp("norm")
                    .tensor((spec.num_groups, group_width), "weight")
            })
            .transpose()?;
        Ok(Self {
            input: linear_b(
                model_spec.hidden_size,
                projection_width,
                spec.input_projection_bias,
                weights.pp("in_proj"),
            )?,
            convolution_kernel,
            convolution_bias,
            time_bias: weights.tensor((spec.num_heads,), "dt_bias")?,
            state_decay: mamba2_head_vector(&weights, "A_log", spec.num_heads)?,
            skip: mamba2_head_vector(&weights, "D", spec.num_heads)?,
            gated_norm_weight,
            output: linear_b(
                spec.inner_size,
                model_spec.hidden_size,
                spec.output_projection_bias,
                weights.pp("out_proj"),
            )?,
            norm_epsilon: model_spec.norm_epsilon,
            spec,
            convolution_width,
            projection_width,
        })
    }

    fn forward_packed(
        &self,
        states: &mut [&mut Mamba2State],
        input: &Tensor,
        lane_lengths: &[usize],
    ) -> Result<Tensor> {
        let (container_batch, total_tokens, _) = input.dims3()?;
        if container_batch != 1 || states.is_empty() || states.len() != lane_lengths.len() {
            candle_core::bail!(
                "packed Mamba2 has batch {container_batch}, {} states and {} lane lengths",
                states.len(),
                lane_lengths.len()
            );
        }
        let expected_tokens = lane_lengths.iter().try_fold(0_usize, |total, &length| {
            if length == 0 {
                return Err(candle_core::Error::Msg(
                    "packed Mamba2 lane cannot be empty".into(),
                ));
            }
            total
                .checked_add(length)
                .ok_or_else(|| candle_core::Error::Msg("packed Mamba2 size overflow".into()))
        })?;
        if expected_tokens != total_tokens {
            candle_core::bail!(
                "packed Mamba2 lane lengths total {expected_tokens}, expected {total_tokens} tokens"
            );
        }

        let projected = self.input.forward(input)?;
        if projected.dim(2)? != self.projection_width {
            candle_core::bail!("Mamba2 input projection returned an unexpected width");
        }
        let mut outputs = Vec::with_capacity(states.len());
        let mut offset = 0_usize;
        for (state, &lane_length) in states.iter_mut().zip(lane_lengths) {
            outputs.push(self.forward_lane(state, &projected.narrow(1, offset, lane_length)?)?);
            offset += lane_length;
        }
        let outputs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&outputs, 1)
    }

    #[allow(clippy::too_many_lines)] // Keep the selective recurrence source-auditable.
    fn forward_lane(&self, state: &mut Mamba2State, projected: &Tensor) -> Result<Tensor> {
        let (_, sequence, width) = projected.dims3()?;
        if sequence == 0
            || width != self.projection_width
            || state.convolution_history_tokens != self.spec.convolution_kernel_size - 1
            || state.convolution_width != self.convolution_width
            || state.num_heads != self.spec.num_heads
            || state.head_dim != self.spec.head_dim
            || state.state_size != self.spec.state_size
        {
            candle_core::bail!("Mamba2 state geometry does not match its layer");
        }

        let gate = projected.narrow(2, 0, self.spec.inner_size)?;
        let xbc = projected.narrow(2, self.spec.inner_size, self.convolution_width)?;
        let dt = projected.narrow(
            2,
            self.spec.inner_size + self.convolution_width,
            self.spec.num_heads,
        )?;
        let history_tokens = self.spec.convolution_kernel_size - 1;
        let convolution_window = if history_tokens == 0 {
            xbc
        } else {
            let history = match &state.convolution_tail {
                Some(tail) => {
                    let expected = (1, history_tokens, self.convolution_width);
                    if tail.dims3()? != expected {
                        candle_core::bail!(
                            "Mamba2 convolution tail has shape {:?}, expected {expected:?}",
                            tail.dims()
                        );
                    }
                    tail.clone()
                }
                None => Tensor::zeros(
                    (1, history_tokens, self.convolution_width),
                    projected.dtype(),
                    projected.device(),
                )?,
            };
            Tensor::cat(&[&history, &xbc], 1)?
        };
        let mut convolution: Option<Tensor> = None;
        for kernel_index in 0..self.spec.convolution_kernel_size {
            let values = convolution_window.narrow(1, kernel_index, sequence)?;
            let weight = self
                .convolution_kernel
                .narrow(1, kernel_index, 1)?
                .squeeze(1)?
                .reshape((1, 1, self.convolution_width))?;
            let product = values.broadcast_mul(&weight)?;
            convolution = Some(match convolution {
                Some(sum) => sum.add(&product)?,
                None => product,
            });
        }
        let mut convolution =
            convolution.expect("validated Mamba2 convolution kernel is non-empty");
        if let Some(bias) = &self.convolution_bias {
            convolution =
                convolution.broadcast_add(&bias.reshape((1, 1, self.convolution_width))?)?;
        }
        let convolution = convolution.silu()?;

        // Keep the complete post-convolution `[x | B | C]` projection
        // contiguous. The Metal scan reads the canonical slices directly,
        // avoiding three materializations before every recurrent layer.
        let convolution_f32 = convolution
            .reshape((sequence, self.convolution_width))?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let raw_dt = dt
            .reshape((sequence, self.spec.num_heads))?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let time_bias = self.time_bias.to_dtype(DType::F32)?.contiguous()?;
        let state_decay_vector = self.state_decay.to_dtype(DType::F32)?.contiguous()?;
        let skip_vector = self.skip.to_dtype(DType::F32)?.contiguous()?;
        let mut recurrent = match &state.recurrent {
            Some(recurrent) => {
                let expected = (
                    self.spec.num_heads,
                    self.spec.head_dim,
                    self.spec.state_size,
                );
                if recurrent.dims3()? != expected || recurrent.dtype() != DType::F32 {
                    candle_core::bail!(
                        "Mamba2 recurrent state has shape {:?}/{:?}, expected {expected:?}/F32",
                        recurrent.dims(),
                        recurrent.dtype()
                    );
                }
                recurrent.clone()
            }
            None => Tensor::zeros(
                (
                    self.spec.num_heads,
                    self.spec.head_dim,
                    self.spec.state_size,
                ),
                DType::F32,
                projected.device(),
            )?,
        };
        let (mut output, fused_recurrence) = if let Some((output, next_recurrent)) =
            crate::mamba2::scan(
                &convolution_f32,
                &raw_dt,
                &time_bias,
                &state_decay_vector,
                &skip_vector,
                &recurrent,
            )? {
            recurrent = next_recurrent;
            (output, true)
        } else {
            let x = convolution_f32
                .narrow(1, 0, self.spec.inner_size)?
                .reshape((sequence, self.spec.num_heads, self.spec.head_dim))?;
            let group_state_width = self.spec.num_groups * self.spec.state_size;
            let b = convolution_f32
                .narrow(1, self.spec.inner_size, group_state_width)?
                .reshape((sequence, self.spec.num_groups, self.spec.state_size))?;
            let c = convolution_f32
                .narrow(
                    1,
                    self.spec.inner_size + group_state_width,
                    group_state_width,
                )?
                .reshape((sequence, self.spec.num_groups, self.spec.state_size))?;
            let dt = softplus(&raw_dt.broadcast_add(&time_bias)?)?;
            let state_decay = state_decay_vector.reshape((self.spec.num_heads, 1, 1))?;
            let skip = skip_vector.reshape((self.spec.num_heads, 1))?;
            let heads_per_group = self.spec.num_heads / self.spec.num_groups;
            let mut token_outputs = Vec::with_capacity(sequence);
            for token in 0..sequence {
                let token_x = x.narrow(0, token, 1)?.squeeze(0)?;
                let token_dt =
                    dt.narrow(0, token, 1)?
                        .squeeze(0)?
                        .reshape((self.spec.num_heads, 1, 1))?;
                let token_b =
                    repeat_mamba_groups(&b.narrow(0, token, 1)?.squeeze(0)?, heads_per_group)?;
                let token_c =
                    repeat_mamba_groups(&c.narrow(0, token, 1)?.squeeze(0)?, heads_per_group)?;
                let decay = token_dt.broadcast_mul(&state_decay)?.exp()?;
                recurrent = recurrent.broadcast_mul(&decay)?;
                let discretized_b = token_b.unsqueeze(1)?.broadcast_mul(&token_dt)?;
                let update = token_x.unsqueeze(2)?.broadcast_mul(&discretized_b)?;
                recurrent = recurrent.add(&update)?;
                let output = recurrent
                    .broadcast_mul(&token_c.unsqueeze(1)?)?
                    .sum(2)?
                    .add(&token_x.broadcast_mul(&skip)?)?;
                token_outputs.push(output.unsqueeze(0)?);
            }
            let outputs = token_outputs.iter().collect::<Vec<_>>();
            (Tensor::cat(&outputs, 0)?, false)
        };
        let gate = gate
            .reshape((sequence, self.spec.num_heads, self.spec.head_dim))?
            .to_dtype(DType::F32)?
            .silu()?;
        output = output.mul(&gate)?;
        if let Some(weight) = &self.gated_norm_weight {
            let group_width = self.spec.inner_size / self.spec.num_groups;
            let grouped = output.reshape((sequence, self.spec.num_groups, group_width))?;
            let denominator = grouped
                .sqr()?
                .sum_keepdim(2)?
                .affine(1.0 / group_width as f64, self.norm_epsilon)?
                .sqrt()?;
            output = grouped
                .broadcast_div(&denominator)?
                .broadcast_mul(&weight.to_dtype(DType::F32)?)?
                .reshape((sequence, self.spec.inner_size))?;
        }
        let output = output
            .to_dtype(projected.dtype())?
            .reshape((1, sequence, self.spec.inner_size))?
            .apply(&self.output)?;

        let next_tail = if history_tokens == 0 {
            None
        } else {
            Some(
                convolution_window
                    .narrow(
                        1,
                        convolution_window.dim(1)? - history_tokens,
                        history_tokens,
                    )?
                    .force_contiguous()?,
            )
        };
        state.logical_len = state
            .logical_len
            .checked_add(sequence)
            .ok_or_else(|| candle_core::Error::Msg("Mamba2 length overflow".into()))?;
        state.convolution_tail = next_tail;
        // A fused result packs `[final state | token outputs]`. Detach the
        // persistent prefix only after every downstream operation succeeds so
        // rollback keeps the old immutable handle and no session retains the
        // larger combined allocation.
        state.recurrent = Some(if fused_recurrence {
            recurrent.force_contiguous()?
        } else {
            recurrent.contiguous()?
        });
        Ok(output)
    }
}

fn mamba2_convolution_width(spec: Mamba2Spec) -> Result<usize> {
    spec.num_groups
        .checked_mul(spec.state_size)
        .and_then(|group_state| group_state.checked_mul(2))
        .and_then(|bc| bc.checked_add(spec.inner_size))
        .ok_or_else(|| candle_core::Error::Msg("Mamba2 convolution width overflow".into()))
}

fn mamba2_head_vector(weights: &WeightPath<'_>, leaf: &str, heads: usize) -> Result<Tensor> {
    let tensor = weights.tensor_unchecked(leaf)?;
    match tensor.dims() {
        [actual] if *actual == heads => Ok(tensor),
        [actual, one] if *actual == heads && *one == 1 => tensor.squeeze(1),
        actual => candle_core::bail!(
            "Mamba2 `{leaf}` must have shape [{heads}] or [{heads}, 1], found {actual:?}"
        ),
    }
}

fn repeat_mamba_groups(input: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(input.clone());
    }
    let groups = input.dim(0)?;
    let mut repeated = Vec::with_capacity(groups * repeats);
    for group in 0..groups {
        let value = input.narrow(0, group, 1)?;
        repeated.extend(std::iter::repeat_n(value, repeats));
    }
    let values = repeated.iter().collect::<Vec<_>>();
    Tensor::cat(&values, 0)
}

fn repeat_heads_tiled_dim(input: &Tensor, repeats: usize, dimension: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(input.clone());
    }
    // Canonical Qwen3.5 GGUFs reorder V-side heads from HF's grouped layout
    // to tiled layout. Match that conversion by repeating the complete K-head
    // block: [K0, K1] -> [K0, K1, K0, K1].
    let inputs = std::iter::repeat_n(input, repeats).collect::<Vec<_>>();
    Tensor::cat(&inputs, dimension)
}

fn l2_normalize_last(input: &Tensor, epsilon: f64) -> Result<Tensor> {
    let last = input.rank() - 1;
    let denominator = input
        .sqr()?
        .sum_keepdim(last)?
        .affine(1.0, epsilon)?
        .sqrt()?;
    input.broadcast_div(&denominator)
}

fn softplus(input: &Tensor) -> Result<Tensor> {
    let positive = input.maximum(&input.zeros_like()?)?;
    let correction = input
        .abs()?
        .affine(-1.0, 0.0)?
        .exp()?
        .affine(1.0, 1.0)?
        .log()?;
    positive.add(&correction)
}

#[derive(Debug)]
enum SequenceMixer {
    Attention(SelfAttention),
    ShortConvolution(ShortConvolution),
    GatedDeltaNet(GatedDeltaNet),
    ParallelAttentionMamba2 {
        attention: SelfAttention,
        mamba2: Mamba2,
    },
}

#[derive(Debug)]
struct DecoderLayer {
    mixer_norm: RmsNorm,
    mixer: SequenceMixer,
    ffn_norm: RmsNorm,
    feed_forward: FeedForward,
    post_attention_norm: Option<RmsNorm>,
    post_ffn_norm: Option<RmsNorm>,
    mixer_residual_scale: f64,
    feed_forward_residual_scale: f64,
}

impl DecoderLayer {
    fn load_mixer(
        spec: &DecoderSpec,
        layer_spec: &DecoderLayerSpec,
        rotary: Option<Arc<RotaryEmbedding>>,
        names: &DecoderWeightNames,
        weights: &WeightPath<'_>,
    ) -> Result<SequenceMixer> {
        match &layer_spec.mixer {
            SequenceMixerSpec::Attention(attention) => {
                Ok(SequenceMixer::Attention(SelfAttention::load(
                    spec,
                    attention,
                    rotary,
                    names.attention,
                    weights.pp(names.layer.attention),
                )?))
            }
            SequenceMixerSpec::ShortConvolution(shortconv) => {
                if rotary.is_some() {
                    candle_core::bail!("short-convolution layer unexpectedly received RoPE state");
                }
                Ok(SequenceMixer::ShortConvolution(ShortConvolution::load(
                    spec,
                    *shortconv,
                    weights.pp(names.layer.short_convolution),
                )?))
            }
            SequenceMixerSpec::GatedDeltaNet(delta) => {
                if rotary.is_some() {
                    candle_core::bail!("Gated DeltaNet layer unexpectedly received RoPE state");
                }
                let path = names.layer.gated_delta.ok_or_else(|| {
                    candle_core::Error::Msg(
                        "Gated DeltaNet weights require a declared tensor path".into(),
                    )
                })?;
                Ok(SequenceMixer::GatedDeltaNet(GatedDeltaNet::load(
                    spec,
                    *delta,
                    weights.pp(path),
                )?))
            }
            SequenceMixerSpec::ParallelAttentionMamba2(parallel) => {
                let mamba_path = names.layer.parallel_mamba2.ok_or_else(|| {
                    candle_core::Error::Msg(
                        "parallel attention/Mamba2 weights require a declared Mamba2 tensor path"
                            .into(),
                    )
                })?;
                Ok(SequenceMixer::ParallelAttentionMamba2 {
                    attention: SelfAttention::load(
                        spec,
                        &parallel.attention,
                        rotary,
                        names.attention,
                        weights.pp(names.layer.attention),
                    )?,
                    mamba2: Mamba2::load(spec, parallel.mamba2, weights.pp(mamba_path))?,
                })
            }
        }
    }

    fn load(
        spec: &DecoderSpec,
        layer_spec: &DecoderLayerSpec,
        rotary: Option<Arc<RotaryEmbedding>>,
        names: &DecoderWeightNames,
        weights: WeightPath<'_>,
    ) -> Result<Self> {
        let (ffn_norm_path, post_attention_norm, post_ffn_norm) = match layer_spec.norm_layout {
            NormLayout::PreNorm => (names.layer.feed_forward_norm, None, None),
            NormLayout::Sandwich => (
                names.layer.pre_feed_forward_norm,
                Some(rms_norm(
                    spec.hidden_size,
                    spec.norm_epsilon,
                    weights.pp(names.layer.post_attention_norm),
                )?),
                Some(rms_norm(
                    spec.hidden_size,
                    spec.norm_epsilon,
                    weights.pp(names.layer.post_feed_forward_norm),
                )?),
            ),
        };
        let mixer = Self::load_mixer(spec, layer_spec, rotary, names, &weights)?;
        Ok(Self {
            mixer_norm: rms_norm(
                spec.hidden_size,
                spec.norm_epsilon,
                weights.pp(names.layer.mixer_norm),
            )?,
            mixer,
            ffn_norm: rms_norm(
                spec.hidden_size,
                spec.norm_epsilon,
                weights.pp(ffn_norm_path),
            )?,
            feed_forward: FeedForward::load(
                spec,
                layer_spec,
                names,
                weights.pp(names.layer.feed_forward),
            )?,
            post_attention_norm,
            post_ffn_norm,
            mixer_residual_scale: layer_spec.mixer_residual_scale,
            feed_forward_residual_scale: layer_spec.feed_forward_residual_scale,
        })
    }

    fn forward_attention_packed(
        &self,
        cache: PackedAttentionCache<'_, '_>,
        input: &Tensor,
        full_masks: Option<&mut LazyFullCausalMasks<'_>>,
        start_positions: &[usize],
        lane_lengths: &[usize],
        prefill_attention: PrefillAttentionMode,
    ) -> Result<Tensor> {
        let normalized = self.mixer_norm.forward(input)?;
        let SequenceMixer::Attention(attention) = &self.mixer else {
            candle_core::bail!("short-convolution layer received an attention cache");
        };
        let mixed = attention.forward_packed(
            cache,
            &normalized,
            full_masks,
            start_positions,
            lane_lengths,
            prefill_attention,
        )?;
        self.finish_mixer(input, mixed)
    }

    fn forward_short_convolution_packed(
        &self,
        states: &mut [&mut ShortConvolutionState],
        input: &Tensor,
        lane_lengths: &[usize],
    ) -> Result<Tensor> {
        let normalized = self.mixer_norm.forward(input)?;
        let SequenceMixer::ShortConvolution(shortconv) = &self.mixer else {
            candle_core::bail!("attention layer received short-convolution state");
        };
        let mixed = shortconv.forward_packed(states, &normalized, lane_lengths)?;
        self.finish_mixer(input, mixed)
    }

    fn forward_gated_delta_net_packed(
        &self,
        states: &mut [&mut GatedDeltaNetState],
        input: &Tensor,
        lane_lengths: &[usize],
    ) -> Result<Tensor> {
        let normalized = self.mixer_norm.forward(input)?;
        let SequenceMixer::GatedDeltaNet(delta) = &self.mixer else {
            candle_core::bail!("non-Gated-DeltaNet layer received recurrent matrix state");
        };
        let mixed = delta.forward_packed(states, &normalized, lane_lengths)?;
        self.finish_mixer(input, mixed)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_parallel_attention_mamba2_packed(
        &self,
        cache: PackedAttentionCache<'_, '_>,
        states: &mut [&mut Mamba2State],
        input: &Tensor,
        full_masks: Option<&mut LazyFullCausalMasks<'_>>,
        start_positions: &[usize],
        lane_lengths: &[usize],
        prefill_attention: PrefillAttentionMode,
    ) -> Result<Tensor> {
        let normalized = self.mixer_norm.forward(input)?;
        let SequenceMixer::ParallelAttentionMamba2 { attention, mamba2 } = &self.mixer else {
            candle_core::bail!("non-parallel layer received attention and Mamba2 state");
        };
        let attention = attention.forward_packed(
            cache,
            &normalized,
            full_masks,
            start_positions,
            lane_lengths,
            prefill_attention,
        )?;
        let recurrent = mamba2.forward_packed(states, &normalized, lane_lengths)?;
        self.finish_mixer(input, attention.add(&recurrent)?)
    }

    fn finish_mixer(&self, input: &Tensor, mixed: Tensor) -> Result<Tensor> {
        let mixed = match &self.post_attention_norm {
            Some(norm) => mixed.apply(norm)?,
            None => mixed,
        };
        let mixed = mixed.affine(self.mixer_residual_scale, 0.0)?;
        let hidden = input.add(&mixed)?;
        let feed_forward = self.ffn_norm.forward(&hidden)?;
        let feed_forward = self.feed_forward.forward(&feed_forward)?;
        let feed_forward = match &self.post_ffn_norm {
            Some(norm) => feed_forward.apply(norm)?,
            None => feed_forward,
        };
        let feed_forward = feed_forward.affine(self.feed_forward_residual_scale, 0.0)?;
        hidden.add(&feed_forward)
    }
}

/// Shared pre-norm decoder implementation used by all model adapters.
#[derive(Debug)]
struct DecoderModelIdentity {
    session_created: Cell<bool>,
}

#[derive(Debug)]
struct DecoderSessionIdentity;

#[derive(Debug)]
enum LayerSequenceState {
    /// Full-attention K/V is stored in the session's shared paged block table.
    FullAttention,
    SlidingAttention(SlidingKvCache),
    ShortConvolution(ShortConvolutionState),
    GatedDeltaNet(GatedDeltaNetState),
    ParallelAttentionMamba2(Mamba2State),
}

#[derive(Debug, Clone)]
enum LayerSequenceCheckpoint {
    FullAttention,
    SlidingAttention(SlidingCacheCheckpoint),
    ShortConvolution(ShortConvolutionCheckpoint),
    GatedDeltaNet(GatedDeltaNetCheckpoint),
    ParallelAttentionMamba2(Mamba2Checkpoint),
}

impl LayerSequenceCheckpoint {
    fn retained_tensor_bytes(&self) -> Option<usize> {
        let tensor_bytes = |tensor: Option<&Tensor>| {
            tensor.map_or(Some(0), |tensor| {
                tensor
                    .elem_count()
                    .checked_mul(tensor.dtype().size_in_bytes())
            })
        };
        match self {
            Self::FullAttention | Self::SlidingAttention(_) => Some(0),
            Self::ShortConvolution(checkpoint) => tensor_bytes(checkpoint.tail.as_ref()),
            Self::GatedDeltaNet(checkpoint) => tensor_bytes(checkpoint.convolution_tail.as_ref())?
                .checked_add(tensor_bytes(checkpoint.recurrent.as_ref())?),
            Self::ParallelAttentionMamba2(checkpoint) => {
                tensor_bytes(checkpoint.convolution_tail.as_ref())?
                    .checked_add(tensor_bytes(checkpoint.recurrent.as_ref())?)
            }
        }
    }
}

impl LayerSequenceState {
    fn logical_len(&self, full_len: Option<usize>) -> Result<usize> {
        match self {
            Self::FullAttention => full_len.ok_or_else(|| {
                candle_core::Error::Msg("full-attention layer has no block table".into())
            }),
            Self::SlidingAttention(cache) => Ok(cache.logical_len()),
            Self::ShortConvolution(state) => Ok(state.logical_len),
            Self::GatedDeltaNet(state) => Ok(state.logical_len),
            Self::ParallelAttentionMamba2(state) => {
                let full_len = full_len.ok_or_else(|| {
                    candle_core::Error::Msg(
                        "parallel attention/Mamba2 layer has no block table".into(),
                    )
                })?;
                if full_len != state.logical_len {
                    candle_core::bail!(
                        "parallel attention length {full_len} disagrees with Mamba2 length {}",
                        state.logical_len
                    );
                }
                Ok(full_len)
            }
        }
    }

    fn checkpoint(&self) -> LayerSequenceCheckpoint {
        match self {
            Self::FullAttention => LayerSequenceCheckpoint::FullAttention,
            Self::SlidingAttention(cache) => {
                LayerSequenceCheckpoint::SlidingAttention(cache.checkpoint())
            }
            Self::ShortConvolution(state) => {
                LayerSequenceCheckpoint::ShortConvolution(state.checkpoint())
            }
            Self::GatedDeltaNet(state) => {
                LayerSequenceCheckpoint::GatedDeltaNet(state.checkpoint())
            }
            Self::ParallelAttentionMamba2(state) => {
                LayerSequenceCheckpoint::ParallelAttentionMamba2(state.checkpoint())
            }
        }
    }

    fn restore(&mut self, checkpoint: LayerSequenceCheckpoint) {
        match (self, checkpoint) {
            (Self::FullAttention, LayerSequenceCheckpoint::FullAttention) => {}
            (Self::SlidingAttention(cache), LayerSequenceCheckpoint::SlidingAttention(saved)) => {
                cache.restore(saved);
            }
            (Self::ShortConvolution(state), LayerSequenceCheckpoint::ShortConvolution(saved)) => {
                state.restore(saved);
            }
            (Self::GatedDeltaNet(state), LayerSequenceCheckpoint::GatedDeltaNet(saved)) => {
                state.restore(saved);
            }
            (
                Self::ParallelAttentionMamba2(state),
                LayerSequenceCheckpoint::ParallelAttentionMamba2(saved),
            ) => state.restore(saved),
            _ => unreachable!("decoder state checkpoint topology changed"),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::FullAttention => {}
            Self::SlidingAttention(cache) => cache.clear(),
            Self::ShortConvolution(state) => state.clear(),
            Self::GatedDeltaNet(state) => state.clear(),
            Self::ParallelAttentionMamba2(state) => state.clear(),
        }
    }
}

/// Mutable sequence state for exactly one sequence of one `DecoderModel`.
#[derive(Debug)]
struct DecoderSession {
    owner: Rc<DecoderModelIdentity>,
    identity: Rc<DecoderSessionIdentity>,
    paged_pool: Option<SharedPagedKvPool>,
    full_table: Option<PagedKvBlockTable>,
    layer_states: Vec<LayerSequenceState>,
    /// Authoritative absolute token count, independent of the mix of KV and
    /// recurrent state used by individual layers.
    logical_len: usize,
}

#[derive(Debug)]
struct DecoderReservationLane {
    session: Rc<DecoderSessionIdentity>,
    start_pos: usize,
    input_tokens: usize,
}

#[derive(Debug)]
struct DecoderKvReservation {
    owner: Rc<DecoderModelIdentity>,
    lanes: Vec<DecoderReservationLane>,
    paged: PagedKvBatchReservation,
}

/// Complete immutable model state at one exact token endpoint.
///
/// Paged K/V pages are shared through their normal reference counts. Recurrent
/// and convolution operators replace tensor handles after every successful
/// forward, so cloning their checkpoint handles gives snapshot/session copy-on-
/// write semantics without an eager device copy.
#[derive(Debug)]
struct DecoderPrefixSnapshot {
    full_table: Option<PagedKvBlockTable>,
    layer_states: Vec<LayerSequenceCheckpoint>,
    logical_len: usize,
}

impl DecoderPrefixSnapshot {
    fn validate_restore(&self, session: &DecoderSession, restore_len: usize) -> Result<()> {
        if restore_len > self.logical_len || self.layer_states.len() != session.layer_states.len() {
            candle_core::bail!("prefix snapshot geometry does not match the destination session");
        }
        match (&self.full_table, &session.full_table) {
            (Some(source), Some(destination))
                if source.logical_len() == self.logical_len && destination.logical_len() == 0 => {}
            (None, None) => {}
            _ => candle_core::bail!(
                "prefix snapshot paged-KV topology does not match the destination session"
            ),
        }
        for (layer, (state, checkpoint)) in session
            .layer_states
            .iter()
            .zip(&self.layer_states)
            .enumerate()
        {
            let valid = match (state, checkpoint) {
                (LayerSequenceState::FullAttention, LayerSequenceCheckpoint::FullAttention) => true,
                (
                    LayerSequenceState::ShortConvolution(_),
                    LayerSequenceCheckpoint::ShortConvolution(saved),
                ) => saved.logical_len == restore_len && self.logical_len == restore_len,
                (
                    LayerSequenceState::GatedDeltaNet(_),
                    LayerSequenceCheckpoint::GatedDeltaNet(saved),
                ) => saved.logical_len == restore_len && self.logical_len == restore_len,
                (
                    LayerSequenceState::ParallelAttentionMamba2(_),
                    LayerSequenceCheckpoint::ParallelAttentionMamba2(saved),
                ) => saved.logical_len == restore_len && self.logical_len == restore_len,
                // Sliding attention mutates its ring allocation in place and
                // is rejected when the cache is configured.
                _ => false,
            };
            if !valid {
                candle_core::bail!(
                    "prefix snapshot state for decoder layer {layer} cannot restore token endpoint {restore_len}"
                );
            }
        }
        Ok(())
    }

    fn restore_into(&self, session: &mut DecoderSession, restore_len: usize) -> Result<()> {
        self.validate_restore(session, restore_len)?;
        if let (Some(source), Some(destination)) = (&self.full_table, &mut session.full_table) {
            destination
                .attach_prefix(source, restore_len)
                .map_err(paged_kv_error)?;
        }
        // All fallible validation and KV retention completed above. Recurrent
        // restoration only swaps immutable tensor handles, keeping admission
        // atomic from the destination session's perspective.
        for (state, checkpoint) in session.layer_states.iter_mut().zip(&self.layer_states) {
            state.restore(checkpoint.clone());
        }
        session.logical_len = restore_len;
        Ok(())
    }
}

#[derive(Debug)]
struct DecoderPrefixCache {
    index: TokenPrefixCache<DecoderPrefixSnapshot>,
    /// Bytes represented by one index cost unit. With paged attention this is
    /// exactly one physical KV block; recurrent-only models use one complete
    /// recurrent snapshot as the unit.
    cost_block_bytes: usize,
}

impl CausalLmKvReservation for DecoderKvReservation {
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecoderTerminal {
    Logits,
    LastHidden,
}

#[derive(Debug)]
pub struct DecoderModel {
    identity: Rc<DecoderModelIdentity>,
    spec: DecoderSpec,
    embedding: TokenEmbedding,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    lm_head: LinearOp,
    dtype: DType,
    device: Device,
    full_layer_indices: Vec<Option<usize>>,
    paged_pool: Option<SharedPagedKvPool>,
    prefix_cache: Option<RefCell<DecoderPrefixCache>>,
    /// Attention implementation used for multi-row prefill chunks.
    prefill_attention: PrefillAttentionMode,
}

impl DecoderModel {
    pub fn load_dense(spec: DecoderSpec, weights: VarBuilder<'_>) -> Result<Self> {
        let source = DenseWeightSource::new(weights);
        Self::load(spec, &source)
    }

    pub fn load(spec: DecoderSpec, source: &dyn WeightSource) -> Result<Self> {
        Self::load_with_weight_names(spec, source, &STANDARD_DECODER_WEIGHT_NAMES)
    }

    #[allow(clippy::too_many_lines)] // One audited load path keeps every shared tensor boundary visible.
    pub(crate) fn load_with_weight_names(
        spec: DecoderSpec,
        source: &dyn WeightSource,
        names: &DecoderWeightNames,
    ) -> Result<Self> {
        spec.validate().map_err(candle_core::Error::wrap)?;
        let weights = WeightPath::root(source);
        let qualified_embedding = format!("{}.{}.weight", names.model, names.embedding);
        let root_embedding = format!("{}.weight", names.embedding);
        let model_weights = match (
            source.contains_tensor(&qualified_embedding),
            source.contains_tensor(&root_embedding),
        ) {
            (Some(false), Some(true)) => weights.clone(),
            _ => weights.pp(names.model),
        };
        let embedding = embedding(
            spec.runtime.vocab_size,
            spec.hidden_size,
            model_weights.pp(names.embedding),
        )?;
        // Most families use one RoPE configuration, while Gemma 3 uses a
        // second base and SmolLM3 deliberately has NoPE layers. Equal RoPE
        // specs share their precomputed tables instead of multiplying
        // context-sized storage by the number of decoder layers.
        let mut rotary_tables: Vec<(RopeSpec, Arc<RotaryEmbedding>)> = Vec::new();
        let mut layers = Vec::with_capacity(spec.num_layers);
        for (index, layer_spec) in spec.layer_specs.iter().enumerate() {
            let rotary = if let Some(rope) = layer_spec
                .mixer
                .attention()
                .and_then(|attention| attention.position_encoding.rope())
            {
                let rotary = if let Some((_, rotary)) =
                    rotary_tables.iter().find(|(existing, _)| existing == rope)
                {
                    Arc::clone(rotary)
                } else {
                    let rotary = Arc::new(RotaryEmbedding::new(
                        rope,
                        spec.head_dim,
                        spec.runtime.max_context,
                        source.dtype(),
                        source.device(),
                    )?);
                    rotary_tables.push((rope.clone(), Arc::clone(&rotary)));
                    rotary
                };
                Some(rotary)
            } else {
                None
            };
            layers.push(DecoderLayer::load(
                &spec,
                layer_spec,
                rotary,
                names,
                model_weights.pp(names.layers).pp(index),
            )?);
        }
        let final_norm = rms_norm(
            spec.hidden_size,
            spec.norm_epsilon,
            model_weights.pp(names.final_norm),
        )?;
        let lm_head = if spec.tied_embeddings {
            embedding.tied_linear()
        } else {
            linear_no_bias(
                spec.hidden_size,
                spec.runtime.vocab_size,
                weights.pp(names.lm_head),
            )?
        };
        let mut next_full_layer = 0_usize;
        let full_layer_indices = spec
            .layer_specs
            .iter()
            .map(|layer| {
                if layer
                    .mixer
                    .attention()
                    .is_some_and(|attention| attention.kind == AttentionKind::FullCausal)
                {
                    let index = next_full_layer;
                    next_full_layer += 1;
                    Some(index)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        // Direct library construction and runtime loading share the same
        // one-model-context default budget. Serving may install an explicit
        // policy through `configure_kv_cache` before creating sessions.
        let paged_pool = if next_full_layer == 0 {
            None
        } else {
            let blocks_per_context = spec.runtime.max_context.div_ceil(16);
            let config =
                PagedKvPoolConfig::new(16, 8, blocks_per_context).map_err(paged_kv_error)?;
            Some(Self::new_paged_pool(
                &spec,
                next_full_layer,
                config,
                source.dtype(),
                source.device(),
            )?)
        };
        Ok(Self {
            identity: Rc::new(DecoderModelIdentity {
                session_created: Cell::new(false),
            }),
            spec,
            embedding,
            layers,
            final_norm,
            lm_head,
            dtype: source.dtype(),
            device: source.device().clone(),
            full_layer_indices,
            paged_pool,
            prefix_cache: None,
            prefill_attention: PrefillAttentionMode::default(),
        })
    }

    fn new_paged_pool(
        spec: &DecoderSpec,
        full_layers: usize,
        config: PagedKvPoolConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<SharedPagedKvPool> {
        let geometry =
            PagedKvGeometry::new(full_layers, spec.num_kv_heads, spec.head_dim, dtype, device)
                .map_err(paged_kv_error)?;
        SharedPagedKvPool::new(config, geometry).map_err(paged_kv_error)
    }

    #[cfg(test)]
    fn attention_mask(
        &self,
        kind: AttentionKind,
        query_len: usize,
        start_pos: usize,
    ) -> Result<Option<Tensor>> {
        let key_end = start_pos
            .checked_add(query_len)
            .ok_or_else(|| candle_core::Error::Msg("attention-mask length overflow".into()))?;
        let key_start = match kind {
            AttentionKind::FullCausal => 0,
            AttentionKind::SlidingCausal { window } => start_pos.saturating_sub(window - 1),
        };
        attention_mask_for_range(
            kind,
            start_pos,
            query_len,
            key_start,
            key_end - key_start,
            self.dtype,
            &self.device,
        )
    }

    #[allow(clippy::too_many_lines)] // The packed layer loop keeps cache and mask policy visibly aligned.
    fn forward_inner_batch(
        &self,
        sessions: &mut [&mut DecoderSession],
        mut paged_cache: Option<&mut PagedKvBatchReservation>,
        input_ids: &Tensor,
        start_positions: &[usize],
        lane_lengths: &[usize],
        terminal: DecoderTerminal,
    ) -> Result<Tensor> {
        let (container_batch, total_tokens) = input_ids.dims2()?;
        if container_batch != 1 {
            candle_core::bail!(
                "packed decoder input requires batch container size 1, found {container_batch}"
            );
        }
        let lanes = sessions.len();
        if lanes == 0 || start_positions.len() != lanes || lane_lengths.len() != lanes {
            candle_core::bail!(
                "packed decoder has {lanes} sessions, {} start positions and {} lane lengths",
                start_positions.len(),
                lane_lengths.len()
            );
        }
        let packed_tokens = lane_lengths.iter().try_fold(0_usize, |total, &length| {
            if length == 0 {
                return Err(candle_core::Error::Msg(
                    "packed decoder lane cannot be empty".into(),
                ));
            }
            total
                .checked_add(length)
                .ok_or_else(|| candle_core::Error::Msg("packed decoder size overflow".into()))
        })?;
        if packed_tokens != total_tokens {
            candle_core::bail!(
                "packed decoder lane lengths total {packed_tokens}, expected {total_tokens} tokens"
            );
        }
        let has_full_attention = self.spec.layer_specs.iter().any(|layer| {
            layer
                .mixer
                .attention()
                .is_some_and(|attention| attention.kind == AttentionKind::FullCausal)
        });
        let mut full_masks = has_full_attention.then(|| {
            LazyFullCausalMasks::new(start_positions, lane_lengths, self.dtype, &self.device)
        });

        let mut hidden = self
            .embedding
            .forward_validated(input_ids)?
            .affine(self.spec.embedding_scale, 0.0)?;
        for (layer_index, (layer, layer_spec)) in
            self.layers.iter().zip(&self.spec.layer_specs).enumerate()
        {
            match &layer_spec.mixer {
                SequenceMixerSpec::Attention(AttentionSpec {
                    kind: AttentionKind::FullCausal,
                    ..
                }) => {
                    let masks = full_masks.as_mut().ok_or_else(|| {
                        candle_core::Error::Msg(
                            "decoder full-attention layer has no lazy mask cache".into(),
                        )
                    })?;
                    let reservation = paged_cache.as_deref_mut().ok_or_else(|| {
                        candle_core::Error::Msg(
                            "decoder full-attention layer has no paged reservation".into(),
                        )
                    })?;
                    hidden = layer.forward_attention_packed(
                        PackedAttentionCache::Full {
                            reservation,
                            layer: self.full_layer_indices[layer_index].ok_or_else(|| {
                                candle_core::Error::Msg(format!(
                                    "decoder full layer {layer_index} has no paged layer index"
                                ))
                            })?,
                        },
                        &hidden,
                        Some(masks),
                        start_positions,
                        lane_lengths,
                        self.prefill_attention,
                    )?;
                }
                SequenceMixerSpec::Attention(AttentionSpec {
                    kind: AttentionKind::SlidingCausal { .. },
                    ..
                }) => {
                    let mut sliding_caches = Vec::with_capacity(lanes);
                    for session in sessions.iter_mut() {
                        let LayerSequenceState::SlidingAttention(cache) =
                            &mut session.layer_states[layer_index]
                        else {
                            candle_core::bail!(
                                "decoder layer {layer_index} is missing its sliding cache"
                            );
                        };
                        sliding_caches.push(cache);
                    }
                    hidden = layer.forward_attention_packed(
                        PackedAttentionCache::Sliding(&mut sliding_caches),
                        &hidden,
                        None,
                        start_positions,
                        lane_lengths,
                        self.prefill_attention,
                    )?;
                }
                SequenceMixerSpec::ShortConvolution(_) => {
                    let mut shortconv_states = Vec::with_capacity(lanes);
                    for session in sessions.iter_mut() {
                        let LayerSequenceState::ShortConvolution(state) =
                            &mut session.layer_states[layer_index]
                        else {
                            candle_core::bail!(
                                "decoder layer {layer_index} is missing its short-convolution state"
                            );
                        };
                        shortconv_states.push(state);
                    }
                    hidden = layer.forward_short_convolution_packed(
                        &mut shortconv_states,
                        &hidden,
                        lane_lengths,
                    )?;
                }
                SequenceMixerSpec::GatedDeltaNet(_) => {
                    let mut delta_states = Vec::with_capacity(lanes);
                    for session in sessions.iter_mut() {
                        let LayerSequenceState::GatedDeltaNet(state) =
                            &mut session.layer_states[layer_index]
                        else {
                            candle_core::bail!(
                                "decoder layer {layer_index} is missing its Gated DeltaNet state"
                            );
                        };
                        delta_states.push(state);
                    }
                    hidden = layer.forward_gated_delta_net_packed(
                        &mut delta_states,
                        &hidden,
                        lane_lengths,
                    )?;
                }
                SequenceMixerSpec::ParallelAttentionMamba2(_) => {
                    let masks = full_masks.as_mut().ok_or_else(|| {
                        candle_core::Error::Msg(
                            "parallel attention/Mamba2 layer has no lazy mask cache".into(),
                        )
                    })?;
                    let mut mamba_states = Vec::with_capacity(lanes);
                    for session in sessions.iter_mut() {
                        let LayerSequenceState::ParallelAttentionMamba2(state) =
                            &mut session.layer_states[layer_index]
                        else {
                            candle_core::bail!(
                                "decoder layer {layer_index} is missing its parallel Mamba2 state"
                            );
                        };
                        mamba_states.push(state);
                    }
                    let reservation = paged_cache.as_deref_mut().ok_or_else(|| {
                        candle_core::Error::Msg(
                            "parallel attention/Mamba2 layer has no paged reservation".into(),
                        )
                    })?;
                    hidden = layer.forward_parallel_attention_mamba2_packed(
                        PackedAttentionCache::Full {
                            reservation,
                            layer: self.full_layer_indices[layer_index].ok_or_else(|| {
                                candle_core::Error::Msg(format!(
                                    "decoder parallel layer {layer_index} has no paged layer index"
                                ))
                            })?,
                        },
                        &mut mamba_states,
                        &hidden,
                        Some(masks),
                        start_positions,
                        lane_lengths,
                        self.prefill_attention,
                    )?;
                }
            }
        }
        let mut end_indices = Vec::with_capacity(lanes);
        let mut offset = 0_usize;
        for &lane_length in lane_lengths {
            offset = offset
                .checked_add(lane_length)
                .ok_or_else(|| candle_core::Error::Msg("packed output offset overflow".into()))?;
            end_indices.push(u32::try_from(offset - 1).map_err(candle_core::Error::wrap)?);
        }
        let end_indices = Tensor::new(end_indices.as_slice(), &self.device)?;
        let last_hidden = hidden
            .index_select(&end_indices, 1)?
            .apply(&self.final_norm)?
            .squeeze(0)?;
        match terminal {
            DecoderTerminal::LastHidden => Ok(last_hidden),
            DecoderTerminal::Logits => {
                let logits = last_hidden
                    .apply(&self.lm_head)?
                    .affine(self.spec.final_logit_scale, 0.0)?;
                match self.spec.final_logit_softcap {
                    Some(softcap) => tanh_softcap(&logits, softcap),
                    None => Ok(logits),
                }
            }
        }
    }
}

impl CausalLm for DecoderModel {
    fn runtime_spec(&self) -> &crate::architecture::RuntimeSpec {
        &self.spec.runtime
    }

    fn configure_kv_cache(&mut self, config: PagedKvPoolConfig) -> Result<()> {
        let full_layers = self.full_layer_indices.iter().flatten().count();
        if full_layers == 0 {
            return Ok(());
        }
        if self.identity.session_created.get() {
            candle_core::bail!("paged KV cache cannot be reconfigured after creating a session");
        }
        self.paged_pool = Some(Self::new_paged_pool(
            &self.spec,
            full_layers,
            config,
            self.dtype,
            &self.device,
        )?);
        Ok(())
    }

    fn configure_prefill_attention(&mut self, mode: PrefillAttentionMode) -> Result<()> {
        if self.identity.session_created.get() {
            candle_core::bail!("prefill attention cannot be reconfigured after creating a session");
        }
        self.prefill_attention = mode;
        Ok(())
    }

    fn configure_prefix_cache(&mut self, capacity_tokens: usize) -> Result<()> {
        if self.identity.session_created.get() {
            candle_core::bail!(
                "automatic prefix cache cannot be reconfigured after creating a session"
            );
        }
        if capacity_tokens == 0 {
            self.prefix_cache = None;
            return Ok(());
        }
        self.validate_prefix_cache_topology()?;
        let (block_tokens, budget_blocks, cost_block_bytes) =
            if let Some(pool) = self.paged_pool.as_ref() {
                let stats = pool.stats().map_err(paged_kv_error)?;
                (
                    stats.block_tokens,
                    capacity_tokens
                        .div_ceil(stats.block_tokens)
                        .min(stats.max_blocks),
                    stats.block_bytes,
                )
            } else {
                // Recurrent state is fixed-size rather than token-scaled. Its
                // actual tensor bytes are one cost unit; admission additionally
                // charges the logical prefix span below so many short endpoints
                // cannot multiply a model-sized state without bound.
                (
                    RECURRENT_PREFIX_CACHE_BLOCK_TOKENS,
                    capacity_tokens.div_ceil(RECURRENT_PREFIX_CACHE_BLOCK_TOKENS),
                    self.recurrent_snapshot_bytes()?.max(1),
                )
            };
        let index = TokenPrefixCache::new(block_tokens, budget_blocks)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        self.prefix_cache = Some(RefCell::new(DecoderPrefixCache {
            index,
            cost_block_bytes,
        }));
        Ok(())
    }

    fn new_session(&self) -> Result<Box<dyn CausalLmSession>> {
        let layer_states = self
            .spec
            .layer_specs
            .iter()
            .map(|layer| match &layer.mixer {
                SequenceMixerSpec::Attention(AttentionSpec {
                    kind: AttentionKind::FullCausal,
                    ..
                }) => Ok(LayerSequenceState::FullAttention),
                SequenceMixerSpec::Attention(AttentionSpec {
                    kind: AttentionKind::SlidingCausal { window },
                    ..
                }) => SlidingKvCache::new(*window, self.spec.runtime.max_context)
                    .map(LayerSequenceState::SlidingAttention),
                SequenceMixerSpec::ShortConvolution(shortconv) => {
                    Ok(LayerSequenceState::ShortConvolution(
                        ShortConvolutionState::new(shortconv.kernel_size),
                    ))
                }
                SequenceMixerSpec::GatedDeltaNet(delta) => {
                    GatedDeltaNetState::new(*delta).map(LayerSequenceState::GatedDeltaNet)
                }
                SequenceMixerSpec::ParallelAttentionMamba2(parallel) => {
                    Mamba2State::new(parallel.mamba2)
                        .map(LayerSequenceState::ParallelAttentionMamba2)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let full_table = self
            .paged_pool
            .as_ref()
            .map(|pool| PagedKvBlockTable::new(pool.clone(), self.spec.runtime.max_context))
            .transpose()
            .map_err(paged_kv_error)?;
        self.identity.session_created.set(true);
        Ok(Box::new(DecoderSession {
            owner: Rc::clone(&self.identity),
            identity: Rc::new(DecoderSessionIdentity),
            paged_pool: self.paged_pool.clone(),
            full_table,
            layer_states,
            logical_len: 0,
        }))
    }

    fn validate_session(&self, session: &dyn CausalLmSession) -> Result<()> {
        let session = session
            .as_any()
            .downcast_ref::<DecoderSession>()
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "session type is incompatible with the shared decoder".into(),
                )
            })?;
        if !Rc::ptr_eq(&self.identity, &session.owner) {
            candle_core::bail!("decoder session belongs to a different model instance");
        }
        Ok(())
    }

    fn reset_session_for_prompt(
        &self,
        session: &mut dyn CausalLmSession,
        prompt_tokens: &[u32],
    ) -> Result<PrefixCacheMatch> {
        let session = self.session_mut(session)?;
        session.clear_cache();
        if session.logical_len != 0
            || session
                .full_table
                .as_ref()
                .is_some_and(|table| table.logical_len() != 0)
        {
            candle_core::bail!("decoder session did not become empty before prompt admission");
        }
        let Some(prefix_cache) = &self.prefix_cache else {
            return Ok(PrefixCacheMatch::default());
        };
        let exact_snapshot = self.prefix_cache_requires_exact_snapshots();
        let mut prefix_cache = prefix_cache.try_borrow_mut().map_err(|_| {
            candle_core::Error::Msg("automatic prefix cache is already borrowed".into())
        })?;
        let hit = if exact_snapshot {
            prefix_cache
                .index
                .lookup(prompt_tokens, prompt_tokens.len())
        } else {
            prefix_cache
                .index
                .lookup_aligned_common_prefix(prompt_tokens, prompt_tokens.len())
        };
        let Some(hit) = hit else {
            return Ok(PrefixCacheMatch::default());
        };
        hit.snapshot.restore_into(session, hit.matched_tokens)?;
        Ok(PrefixCacheMatch {
            cached_tokens: hit.matched_tokens,
        })
    }

    fn publish_prompt_prefix(&self, session: &dyn CausalLmSession, prompt_tokens: &[u32]) {
        let Some(prefix_cache) = &self.prefix_cache else {
            return;
        };
        let Ok(session) = self.session_ref(session) else {
            return;
        };
        let exact_snapshot = self.prefix_cache_requires_exact_snapshots();
        let Ok(mut prefix_cache) = prefix_cache.try_borrow_mut() else {
            return;
        };
        let block_tokens = prefix_cache.index.block_tokens();
        let snapshot_tokens = if exact_snapshot {
            prompt_tokens.len()
        } else {
            prompt_tokens.len() / block_tokens * block_tokens
        };
        if snapshot_tokens == 0
            || session.logical_len < snapshot_tokens
            || (exact_snapshot && session.logical_len != snapshot_tokens)
        {
            return;
        }
        let full_table = match session.full_table.as_ref() {
            Some(table) => match table.fork_prefix(snapshot_tokens) {
                Ok(snapshot) => Some(snapshot),
                Err(_) => return,
            },
            None => None,
        };
        let layer_states = if exact_snapshot {
            session
                .layer_states
                .iter()
                .map(LayerSequenceState::checkpoint)
                .collect::<Vec<_>>()
        } else {
            // A longer dense KV snapshot can be truncated at a block boundary;
            // dense layers carry no independent sequence state.
            vec![LayerSequenceCheckpoint::FullAttention; session.layer_states.len()]
        };
        let Some(recurrent_bytes) = layer_states.iter().try_fold(0_usize, |total, state| {
            total.checked_add(state.retained_tensor_bytes()?)
        }) else {
            return;
        };
        let kv_blocks = full_table
            .as_ref()
            .map_or(0, PagedKvBlockTable::owned_blocks);
        let recurrent_blocks = recurrent_bytes.div_ceil(prefix_cache.cost_block_bytes);
        let logical_blocks = snapshot_tokens.div_ceil(block_tokens);
        let cost = if full_table.is_some() {
            let Some(cost) = kv_blocks.checked_add(recurrent_blocks) else {
                return;
            };
            cost
        } else {
            recurrent_blocks.max(logical_blocks)
        };
        let snapshot = DecoderPrefixSnapshot {
            full_table,
            layer_states,
            logical_len: snapshot_tokens,
        };
        let _ = prefix_cache.index.insert(
            prompt_tokens[..snapshot_tokens].to_vec(),
            snapshot,
            cost.max(1),
        );
    }

    fn prefix_cache_diagnostics(&self) -> Result<Option<PrefixCacheDiagnostics>> {
        let Some(prefix_cache) = &self.prefix_cache else {
            return Ok(None);
        };
        let prefix_cache = prefix_cache.try_borrow().map_err(|_| {
            candle_core::Error::Msg("automatic prefix cache is already borrowed".into())
        })?;
        let index = prefix_cache.index.stats();
        let pool = self
            .paged_pool
            .as_ref()
            .map(SharedPagedKvPool::stats)
            .transpose()
            .map_err(paged_kv_error)?;
        Ok(Some(PrefixCacheDiagnostics {
            hits: index.hits,
            misses: index.misses,
            matched_tokens: index.matched_tokens,
            entries: index.entries,
            evictions: index.evictions,
            resident_snapshot_blocks: index.cost,
            block_tokens: prefix_cache.index.block_tokens(),
            physical_live_blocks: pool.map_or(0, |pool| pool.live_blocks),
            physical_live_references: pool.map_or(0, |pool| pool.live_references),
            physical_shared_blocks: pool.map_or(0, |pool| pool.shared_blocks),
            physical_max_blocks: pool.map_or(0, |pool| pool.max_blocks),
        }))
    }

    fn forward(
        &self,
        session: &mut dyn CausalLmSession,
        input_ids: &Tensor,
        start_pos: usize,
    ) -> Result<Tensor> {
        let session = self.session_mut(session)?;
        self.embedding.validate_ids(input_ids)?;
        let (batch, sequence) = input_ids.dims2()?;
        if batch != 1 {
            candle_core::bail!(
                "one decoder session requires batch size 1, received batch size {batch}"
            );
        }
        self.validate_session_geometry(session, sequence, start_pos)?;
        let reservation = self.reserve_single(session, start_pos, sequence)?;
        self.execute_packed(
            &mut [session],
            input_ids,
            &[start_pos],
            &[sequence],
            reservation,
            DecoderTerminal::Logits,
        )
    }

    fn forward_tokens(
        &self,
        session: &mut dyn CausalLmSession,
        input_ids: &[u32],
        start_pos: usize,
    ) -> Result<Tensor> {
        self.forward_host_tokens(session, input_ids, start_pos, DecoderTerminal::Logits)
    }

    fn forward_last_hidden_tokens(
        &self,
        session: &mut dyn CausalLmSession,
        input_ids: &[u32],
        start_pos: usize,
    ) -> Result<Tensor> {
        self.forward_host_tokens(session, input_ids, start_pos, DecoderTerminal::LastHidden)
    }

    fn mixed_batch_capability(&self) -> CausalLmBatchCapability {
        CausalLmBatchCapability::PackedRagged
    }

    fn kv_cache_capacity(&self) -> Option<KvCacheCapacity> {
        let pool = self.paged_pool.as_ref()?;
        let stats = pool.stats().ok()?;
        Some(KvCacheCapacity {
            block_tokens: stats.block_tokens,
            total_blocks: stats.max_blocks,
            free_blocks: stats
                .free_blocks
                .saturating_add(stats.max_blocks.saturating_sub(stats.allocated_blocks)),
            live_blocks: stats.live_blocks,
            retired_blocks: stats.retired_blocks,
        })
    }

    fn reserve_batch_kv(
        &self,
        inputs: &[CausalLmKvReservationInput<'_>],
    ) -> std::result::Result<Option<Box<dyn CausalLmKvReservation>>, KvCacheReservationError> {
        if inputs.is_empty() {
            return Err(KvCacheReservationError::Fatal(candle_core::Error::Msg(
                "paged KV reservation batch cannot be empty".into(),
            )));
        }
        let mut sessions = Vec::with_capacity(inputs.len());
        for input in inputs {
            let session = self.session_ref(input.session())?;
            self.validate_session_geometry(session, input.input_tokens(), input.start_pos())?;
            sessions.push(session);
        }
        let Some(pool) = self.paged_pool.as_ref() else {
            return Ok(None);
        };
        let mut requests = Vec::with_capacity(inputs.len());
        let mut lanes = Vec::with_capacity(inputs.len());
        for (session, input) in sessions.into_iter().zip(inputs) {
            let table = session.full_table.as_ref().ok_or_else(|| {
                KvCacheReservationError::Fatal(candle_core::Error::Msg(
                    "decoder session is missing its full-attention block table".into(),
                ))
            })?;
            requests.push(PagedKvGrowthRequest::new(table, input.input_tokens()));
            lanes.push(DecoderReservationLane {
                session: Rc::clone(&session.identity),
                start_pos: input.start_pos(),
                input_tokens: input.input_tokens(),
            });
        }
        let paged = loop {
            match pool.reserve_batch(&requests) {
                Ok(reservation) => break reservation,
                Err(error @ PagedKvPoolError::Exhausted { .. }) => {
                    let Some(prefix_cache) = &self.prefix_cache else {
                        return Err(self.classify_reservation_error(error));
                    };
                    let evicted = prefix_cache
                        .try_borrow_mut()
                        .map_err(|_| {
                            KvCacheReservationError::Fatal(candle_core::Error::Msg(
                                "automatic prefix cache is already borrowed".into(),
                            ))
                        })?
                        .index
                        .evict_lru();
                    let Some(evicted) = evicted else {
                        return Err(self.classify_reservation_error(error));
                    };
                    drop(evicted);
                }
                Err(error) => return Err(self.classify_reservation_error(error)),
            }
        };
        Ok(Some(Box::new(DecoderKvReservation {
            owner: Rc::clone(&self.identity),
            lanes,
            paged,
        })))
    }

    fn forward_batch_tokens(&self, inputs: &mut [CausalLmBatchInput<'_>]) -> Result<Tensor> {
        self.forward_host_batch(inputs, DecoderTerminal::Logits)
    }

    fn forward_batch_last_hidden_tokens(
        &self,
        inputs: &mut [CausalLmBatchInput<'_>],
    ) -> Result<Tensor> {
        self.forward_host_batch(inputs, DecoderTerminal::LastHidden)
    }

    fn forward_batch_tokens_reserved(
        &self,
        inputs: &mut [CausalLmBatchInput<'_>],
        reservation: Option<Box<dyn CausalLmKvReservation>>,
    ) -> Result<Tensor> {
        self.forward_host_batch_reserved(inputs, reservation, DecoderTerminal::Logits)
    }

    fn read_output_to_cpu(&self, logits: &Tensor) -> Result<Tensor> {
        let output = logits.to_device(&Device::Cpu)?;
        self.mark_paged_completed()?;
        Ok(output)
    }

    fn synchronize(&self) -> Result<()> {
        self.device.synchronize()?;
        self.mark_paged_completed()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl CausalLmSession for DecoderSession {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn clear_cache(&mut self) {
        if let Some(table) = &mut self.full_table {
            let old_len = table.logical_len();
            table.clear();
            if old_len != 0 && table.logical_len() != 0 {
                // An owned reservation keeps the table locked. Preserve hybrid
                // layer agreement instead of clearing only the sliding rings.
                return;
            }
        }
        for state in &mut self.layer_states {
            state.clear();
        }
        self.logical_len = 0;
    }

    fn mark_device_synchronized(&mut self) {
        let _ = mark_pool_completed(self.paged_pool.as_ref());
    }

    fn cache_len(&self) -> usize {
        self.logical_len
    }

    fn kv_cache_stats(&self) -> Option<Vec<KvCacheStats>> {
        let mut stats = Vec::new();
        for state in &self.layer_states {
            match state {
                LayerSequenceState::FullAttention
                | LayerSequenceState::ParallelAttentionMamba2(_) => {
                    stats.push(self.full_table.as_ref()?.stats().ok()?);
                }
                LayerSequenceState::SlidingAttention(cache) => stats.push(cache.stats()),
                LayerSequenceState::ShortConvolution(_) | LayerSequenceState::GatedDeltaNet(_) => {}
            }
        }
        (!stats.is_empty()).then_some(stats)
    }
}

impl DecoderModel {
    fn session_ref<'a>(&self, session: &'a dyn CausalLmSession) -> Result<&'a DecoderSession> {
        let session = session
            .as_any()
            .downcast_ref::<DecoderSession>()
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "session type is incompatible with the shared decoder".into(),
                )
            })?;
        if !Rc::ptr_eq(&self.identity, &session.owner) {
            candle_core::bail!("decoder session belongs to a different model instance");
        }
        Ok(session)
    }

    fn session_mut<'a>(
        &self,
        session: &'a mut dyn CausalLmSession,
    ) -> Result<&'a mut DecoderSession> {
        let session = session
            .as_any_mut()
            .downcast_mut::<DecoderSession>()
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "session type is incompatible with the shared decoder".into(),
                )
            })?;
        if !Rc::ptr_eq(&self.identity, &session.owner) {
            candle_core::bail!("decoder session belongs to a different model instance");
        }
        Ok(session)
    }

    fn validate_session_geometry(
        &self,
        session: &DecoderSession,
        sequence: usize,
        start_pos: usize,
    ) -> Result<()> {
        if sequence == 0 {
            candle_core::bail!("input sequence cannot be empty");
        }
        if session.layer_states.len() != self.layers.len() {
            candle_core::bail!(
                "decoder session has {} layer states, expected {}",
                session.layer_states.len(),
                self.layers.len()
            );
        }
        if session.logical_len != start_pos {
            candle_core::bail!(
                "start_pos {start_pos} does not match decoder sequence length {}",
                session.logical_len
            );
        }
        let full_len = match (&self.paged_pool, &session.full_table) {
            (Some(_), Some(table)) => Some(table.logical_len()),
            (None, None) => None,
            _ => candle_core::bail!(
                "decoder session full-attention block table does not match its model"
            ),
        };
        let mut state_lengths = Vec::with_capacity(self.layers.len());
        for (layer_index, (layer, state)) in self
            .spec
            .layer_specs
            .iter()
            .zip(&session.layer_states)
            .enumerate()
        {
            match (&layer.mixer, state) {
                (
                    SequenceMixerSpec::Attention(AttentionSpec {
                        kind: AttentionKind::FullCausal,
                        ..
                    }),
                    LayerSequenceState::FullAttention,
                )
                | (
                    SequenceMixerSpec::Attention(AttentionSpec {
                        kind: AttentionKind::SlidingCausal { .. },
                        ..
                    }),
                    LayerSequenceState::SlidingAttention(_),
                )
                | (
                    SequenceMixerSpec::ShortConvolution(_),
                    LayerSequenceState::ShortConvolution(_),
                )
                | (SequenceMixerSpec::GatedDeltaNet(_), LayerSequenceState::GatedDeltaNet(_))
                | (
                    SequenceMixerSpec::ParallelAttentionMamba2(_),
                    LayerSequenceState::ParallelAttentionMamba2(_),
                ) => {
                    state_lengths.push(state.logical_len(full_len).map_err(|error| {
                        candle_core::Error::Msg(format!(
                            "decoder layer {layer_index} has invalid sequence state: {error}"
                        ))
                    })?);
                }
                _ => candle_core::bail!(
                    "decoder layer {layer_index} state policy does not match its mixer kind"
                ),
            }
        }
        if state_lengths.iter().any(|&len| len != start_pos) {
            candle_core::bail!(
                "start_pos {} does not match per-layer sequence-state lengths {:?}",
                start_pos,
                state_lengths
            );
        }
        let request_length = start_pos
            .checked_add(sequence)
            .ok_or_else(|| candle_core::Error::Msg("request length overflow".into()))?;
        if request_length > self.spec.runtime.max_context {
            candle_core::bail!(
                "request length {} exceeds context limit {}",
                request_length,
                self.spec.runtime.max_context
            );
        }
        Ok(())
    }

    fn classify_reservation_error(&self, error: PagedKvPoolError) -> KvCacheReservationError {
        let PagedKvPoolError::Exhausted {
            requested_blocks, ..
        } = error
        else {
            return KvCacheReservationError::Fatal(paged_kv_error(error));
        };
        let Some(pool) = self.paged_pool.as_ref() else {
            return KvCacheReservationError::Fatal(candle_core::Error::Msg(
                "paged KV exhaustion was reported without a model pool".into(),
            ));
        };
        match pool.stats() {
            Ok(stats) => KvCacheReservationError::Exhausted(KvCacheExhausted {
                requested_blocks,
                free_blocks: stats
                    .free_blocks
                    .saturating_add(stats.max_blocks.saturating_sub(stats.allocated_blocks)),
                retired_blocks: stats.retired_blocks,
                total_blocks: stats.max_blocks,
                block_tokens: stats.block_tokens,
            }),
            Err(error) => KvCacheReservationError::Fatal(paged_kv_error(error)),
        }
    }

    fn reserve_single(
        &self,
        session: &DecoderSession,
        start_pos: usize,
        input_tokens: usize,
    ) -> Result<Option<DecoderKvReservation>> {
        let request = [CausalLmKvReservationInput::new(
            session,
            start_pos,
            input_tokens,
        )];
        let reservation = self
            .reserve_batch_kv(&request)
            .map_err(kv_reservation_error)?;
        self.take_decoder_reservation(reservation)
    }

    fn take_decoder_reservation(
        &self,
        reservation: Option<Box<dyn CausalLmKvReservation>>,
    ) -> Result<Option<DecoderKvReservation>> {
        match (&self.paged_pool, reservation) {
            (None, None) => Ok(None),
            (None, Some(_)) => {
                candle_core::bail!("sliding-only decoder does not accept a KV reservation")
            }
            (Some(_), None) => {
                candle_core::bail!("full-attention decoder requires a paged KV reservation")
            }
            (Some(_), Some(reservation)) => reservation
                .into_any()
                .downcast::<DecoderKvReservation>()
                .map(|reservation| Some(*reservation))
                .map_err(|_| {
                    candle_core::Error::Msg(
                        "KV reservation type is incompatible with the shared decoder".into(),
                    )
                }),
        }
    }

    fn validate_decoder_reservation(
        &self,
        reservation: Option<&DecoderKvReservation>,
        sessions: &[&DecoderSession],
        start_positions: &[usize],
        lane_lengths: &[usize],
    ) -> Result<()> {
        let Some(reservation) = reservation else {
            if self.paged_pool.is_some() {
                candle_core::bail!("full-attention decoder requires a paged KV reservation");
            }
            return Ok(());
        };
        if self.paged_pool.is_none() || !Rc::ptr_eq(&self.identity, &reservation.owner) {
            candle_core::bail!("paged KV reservation belongs to a different model instance");
        }
        if reservation.lanes.len() != sessions.len()
            || start_positions.len() != sessions.len()
            || lane_lengths.len() != sessions.len()
            || reservation.paged.lane_count() != sessions.len()
        {
            candle_core::bail!(
                "paged KV reservation lane geometry does not match the submitted batch"
            );
        }
        for (lane, (((expected, session), &start_pos), &input_tokens)) in reservation
            .lanes
            .iter()
            .zip(sessions)
            .zip(start_positions)
            .zip(lane_lengths)
            .enumerate()
        {
            let reserved_old = reservation
                .paged
                .lane_old_len(lane)
                .map_err(paged_kv_error)?;
            let reserved_new = reservation
                .paged
                .lane_new_len(lane)
                .map_err(paged_kv_error)?;
            if !Rc::ptr_eq(&expected.session, &session.identity)
                || expected.start_pos != start_pos
                || expected.input_tokens != input_tokens
                || reserved_old != start_pos
                || reserved_new
                    != start_pos.checked_add(input_tokens).ok_or_else(|| {
                        candle_core::Error::Msg("paged reservation range overflows usize".into())
                    })?
            {
                candle_core::bail!(
                    "paged KV reservation lane {lane} does not match session, order, start or token count"
                );
            }
        }
        Ok(())
    }

    fn sequence_checkpoints(
        sessions: &[&mut DecoderSession],
    ) -> Vec<(usize, Vec<LayerSequenceCheckpoint>)> {
        sessions
            .iter()
            .map(|session| {
                (
                    session.logical_len,
                    session
                        .layer_states
                        .iter()
                        .map(LayerSequenceState::checkpoint)
                        .collect(),
                )
            })
            .collect()
    }

    fn restore_sequence_states(
        sessions: &mut [&mut DecoderSession],
        checkpoints: Vec<(usize, Vec<LayerSequenceCheckpoint>)>,
    ) {
        for (session, (logical_len, lane_checkpoints)) in sessions.iter_mut().zip(checkpoints) {
            for (state, checkpoint) in session.layer_states.iter_mut().zip(lane_checkpoints) {
                state.restore(checkpoint);
            }
            session.logical_len = logical_len;
        }
    }

    fn execute_packed(
        &self,
        sessions: &mut [&mut DecoderSession],
        input_ids: &Tensor,
        start_positions: &[usize],
        lane_lengths: &[usize],
        mut reservation: Option<DecoderKvReservation>,
        terminal: DecoderTerminal,
    ) -> Result<Tensor> {
        let immutable_sessions = sessions
            .iter()
            .map(|session| &**session)
            .collect::<Vec<_>>();
        self.validate_decoder_reservation(
            reservation.as_ref(),
            &immutable_sessions,
            start_positions,
            lane_lengths,
        )?;
        drop(immutable_sessions);

        let checkpoints = Self::sequence_checkpoints(sessions);
        let output = match self.forward_inner_batch(
            sessions,
            reservation
                .as_mut()
                .map(|reservation| &mut reservation.paged),
            input_ids,
            start_positions,
            lane_lengths,
            terminal,
        ) {
            Ok(output) => output,
            Err(error) => {
                Self::restore_sequence_states(sessions, checkpoints);
                return Err(error);
            }
        };

        let completion = if let Some(reservation) = reservation {
            let mut tables = sessions
                .iter_mut()
                .map(|session| {
                    session
                        .full_table
                        .as_mut()
                        .expect("reserved decoder session has a full block table")
                })
                .collect::<Vec<_>>();
            match reservation.paged.commit(&mut tables) {
                Ok(epoch) => Some(epoch),
                Err(error) => {
                    Self::restore_sequence_states(sessions, checkpoints);
                    return Err(paged_kv_error(error));
                }
            }
        } else {
            None
        };
        for ((session, &start_pos), &lane_length) in
            sessions.iter_mut().zip(start_positions).zip(lane_lengths)
        {
            session.logical_len = start_pos.checked_add(lane_length).ok_or_else(|| {
                candle_core::Error::Msg("decoder sequence length overflow".into())
            })?;
        }
        if matches!(&self.device, Device::Cpu)
            && let Some(epoch) = completion
        {
            self.paged_pool
                .as_ref()
                .expect("paged completion has a model pool")
                .mark_completed_through(epoch)
                .map_err(paged_kv_error)?;
        }
        Ok(output)
    }

    fn forward_host_batch_reserved(
        &self,
        inputs: &mut [CausalLmBatchInput<'_>],
        reservation: Option<Box<dyn CausalLmKvReservation>>,
        terminal: DecoderTerminal,
    ) -> Result<Tensor> {
        let reservation = self.take_decoder_reservation(reservation)?;
        if inputs.is_empty() {
            candle_core::bail!("mixed-sequence batch cannot be empty");
        }
        let mut total_tokens = 0_usize;
        let mut lane_lengths = Vec::with_capacity(inputs.len());
        let mut start_positions = Vec::with_capacity(inputs.len());
        let mut immutable_sessions = Vec::with_capacity(inputs.len());
        for input in inputs.iter() {
            let lane_length = input.input_ids().len();
            if input
                .input_ids()
                .iter()
                .any(|&token| token as usize >= self.spec.runtime.vocab_size)
            {
                candle_core::bail!("input contains a token ID outside the model vocabulary");
            }
            let session = self.session_ref(input.session())?;
            self.validate_session_geometry(session, lane_length, input.start_pos())?;
            total_tokens = total_tokens
                .checked_add(lane_length)
                .ok_or_else(|| candle_core::Error::Msg("packed token count overflow".into()))?;
            lane_lengths.push(lane_length);
            start_positions.push(input.start_pos());
            immutable_sessions.push(session);
        }
        self.validate_decoder_reservation(
            reservation.as_ref(),
            &immutable_sessions,
            &start_positions,
            &lane_lengths,
        )?;
        drop(immutable_sessions);
        u32::try_from(total_tokens).map_err(|_| {
            candle_core::Error::Msg(format!(
                "packed token count {total_tokens} exceeds the u32 indexing limit"
            ))
        })?;

        let mut packed_tokens = Vec::new();
        packed_tokens
            .try_reserve_exact(total_tokens)
            .map_err(candle_core::Error::wrap)?;
        for input in inputs.iter() {
            packed_tokens.extend_from_slice(input.input_ids());
        }
        let mut sessions = Vec::with_capacity(inputs.len());
        for input in inputs.iter_mut() {
            sessions.push(self.session_mut(input.session_mut())?);
        }
        // Upload only after reservation and every lane/session/token check.
        let input_ids = Tensor::from_vec(packed_tokens, (1, total_tokens), &self.device)?;
        self.execute_packed(
            &mut sessions,
            &input_ids,
            &start_positions,
            &lane_lengths,
            reservation,
            terminal,
        )
    }

    fn mark_paged_completed(&self) -> Result<()> {
        mark_pool_completed(self.paged_pool.as_ref()).map_err(paged_kv_error)
    }
}

impl DecoderModel {
    fn forward_host_tokens(
        &self,
        session: &mut dyn CausalLmSession,
        input_ids: &[u32],
        start_pos: usize,
        terminal: DecoderTerminal,
    ) -> Result<Tensor> {
        let session = self.session_mut(session)?;
        if input_ids
            .iter()
            .any(|&token| token as usize >= self.spec.runtime.vocab_size)
        {
            candle_core::bail!("input contains a token ID outside the model vocabulary");
        }
        self.validate_session_geometry(session, input_ids.len(), start_pos)?;
        let reservation = self.reserve_single(session, start_pos, input_ids.len())?;
        // Host-only validation and capacity reservation precede the upload.
        let input = Tensor::from_slice(input_ids, (1, input_ids.len()), &self.device)?;
        self.execute_packed(
            &mut [session],
            &input,
            &[start_pos],
            &[input_ids.len()],
            reservation,
            terminal,
        )
    }

    fn forward_host_batch(
        &self,
        inputs: &mut [CausalLmBatchInput<'_>],
        terminal: DecoderTerminal,
    ) -> Result<Tensor> {
        if inputs.is_empty() {
            candle_core::bail!("mixed-sequence batch cannot be empty");
        }
        for input in inputs.iter() {
            if input.input_ids().is_empty() {
                candle_core::bail!("mixed-sequence batch lane cannot be empty");
            }
            if input
                .input_ids()
                .iter()
                .any(|&token| token as usize >= self.spec.runtime.vocab_size)
            {
                candle_core::bail!("input contains a token ID outside the model vocabulary");
            }
        }
        let requests = inputs
            .iter()
            .map(|input| {
                CausalLmKvReservationInput::new(
                    input.session(),
                    input.start_pos(),
                    input.input_ids().len(),
                )
            })
            .collect::<Vec<_>>();
        let reservation = self
            .reserve_batch_kv(&requests)
            .map_err(kv_reservation_error)?;
        self.forward_host_batch_reserved(inputs, reservation, terminal)
    }

    fn prefix_cache_requires_exact_snapshots(&self) -> bool {
        self.spec.layer_specs.iter().any(|layer| {
            matches!(
                &layer.mixer,
                SequenceMixerSpec::ShortConvolution(_)
                    | SequenceMixerSpec::GatedDeltaNet(_)
                    | SequenceMixerSpec::ParallelAttentionMamba2(_)
            )
        })
    }

    fn validate_prefix_cache_topology(&self) -> Result<()> {
        for (layer, spec) in self.spec.layer_specs.iter().enumerate() {
            let supported = match &spec.mixer {
                SequenceMixerSpec::Attention(attention) => {
                    attention.kind == AttentionKind::FullCausal
                }
                SequenceMixerSpec::ParallelAttentionMamba2(parallel) => {
                    parallel.attention.kind == AttentionKind::FullCausal
                }
                SequenceMixerSpec::ShortConvolution(_) | SequenceMixerSpec::GatedDeltaNet(_) => {
                    true
                }
            };
            if !supported {
                candle_core::bail!(
                    "automatic prefix cache cannot snapshot layer {layer}: sliding-attention storage is mutable and has no immutable fork"
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)] // Every recurrent topology's byte geometry is audited here.
    fn recurrent_snapshot_bytes(&self) -> Result<usize> {
        let activation_bytes = self.dtype.size_in_bytes();
        let mut total = 0_usize;
        for layer in &self.spec.layer_specs {
            let bytes = match &layer.mixer {
                SequenceMixerSpec::Attention(_) => 0,
                SequenceMixerSpec::ShortConvolution(spec) => spec
                    .kernel_size
                    .saturating_sub(1)
                    .checked_mul(self.spec.hidden_size)
                    .and_then(|elements| elements.checked_mul(activation_bytes))
                    .ok_or_else(|| {
                        candle_core::Error::Msg(
                            "short-convolution prefix snapshot size overflows usize".into(),
                        )
                    })?,
                SequenceMixerSpec::GatedDeltaNet(spec) => {
                    let key_width = spec
                        .num_key_heads
                        .checked_mul(spec.key_head_dim)
                        .ok_or_else(|| {
                            candle_core::Error::Msg(
                                "Gated DeltaNet prefix key width overflows usize".into(),
                            )
                        })?;
                    let value_width = spec
                        .num_value_heads
                        .checked_mul(spec.value_head_dim)
                        .ok_or_else(|| {
                            candle_core::Error::Msg(
                                "Gated DeltaNet prefix value width overflows usize".into(),
                            )
                        })?;
                    let convolution_width = key_width
                        .checked_mul(2)
                        .and_then(|width| width.checked_add(value_width))
                        .ok_or_else(|| {
                            candle_core::Error::Msg(
                                "Gated DeltaNet prefix convolution width overflows usize".into(),
                            )
                        })?;
                    let tail = spec
                        .convolution_kernel_size
                        .saturating_sub(1)
                        .checked_mul(convolution_width)
                        .and_then(|elements| elements.checked_mul(activation_bytes))
                        .ok_or_else(|| {
                            candle_core::Error::Msg(
                                "Gated DeltaNet prefix convolution size overflows usize".into(),
                            )
                        })?;
                    let recurrent = spec
                        .num_value_heads
                        .checked_mul(spec.key_head_dim)
                        .and_then(|elements| elements.checked_mul(spec.value_head_dim))
                        .and_then(|elements| elements.checked_mul(DType::F32.size_in_bytes()))
                        .ok_or_else(|| {
                            candle_core::Error::Msg(
                                "Gated DeltaNet prefix recurrent size overflows usize".into(),
                            )
                        })?;
                    tail.checked_add(recurrent).ok_or_else(|| {
                        candle_core::Error::Msg(
                            "Gated DeltaNet prefix snapshot size overflows usize".into(),
                        )
                    })?
                }
                SequenceMixerSpec::ParallelAttentionMamba2(parallel) => {
                    let spec = parallel.mamba2;
                    let convolution_width = mamba2_convolution_width(spec)?;
                    let tail = spec
                        .convolution_kernel_size
                        .saturating_sub(1)
                        .checked_mul(convolution_width)
                        .and_then(|elements| elements.checked_mul(activation_bytes))
                        .ok_or_else(|| {
                            candle_core::Error::Msg(
                                "Mamba2 prefix convolution size overflows usize".into(),
                            )
                        })?;
                    let recurrent = spec
                        .num_heads
                        .checked_mul(spec.head_dim)
                        .and_then(|elements| elements.checked_mul(spec.state_size))
                        .and_then(|elements| elements.checked_mul(DType::F32.size_in_bytes()))
                        .ok_or_else(|| {
                            candle_core::Error::Msg(
                                "Mamba2 prefix recurrent size overflows usize".into(),
                            )
                        })?;
                    tail.checked_add(recurrent).ok_or_else(|| {
                        candle_core::Error::Msg(
                            "Mamba2 prefix snapshot size overflows usize".into(),
                        )
                    })?
                }
            };
            total = total.checked_add(bytes).ok_or_else(|| {
                candle_core::Error::Msg("prefix recurrent-state size overflows usize".into())
            })?;
        }
        Ok(total)
    }
}

fn mark_pool_completed(pool: Option<&SharedPagedKvPool>) -> crate::cache::PagedKvResult<()> {
    let Some(pool) = pool else {
        return Ok(());
    };
    let submitted = pool.stats()?.submitted_epoch;
    pool.mark_completed_through(submitted)
}

fn paged_kv_error(error: PagedKvPoolError) -> candle_core::Error {
    candle_core::Error::Msg(error.to_string())
}

fn kv_reservation_error(error: KvCacheReservationError) -> candle_core::Error {
    match error {
        KvCacheReservationError::Exhausted(error) => candle_core::Error::Msg(error.to_string()),
        KvCacheReservationError::Fatal(error) => error,
    }
}

pub(crate) fn attention_mask_for_range(
    kind: AttentionKind,
    query_start: usize,
    query_len: usize,
    key_start: usize,
    key_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    let _ = query_start
        .checked_add(query_len)
        .ok_or_else(|| candle_core::Error::Msg("attention query range overflow".into()))?;
    let _ = key_start
        .checked_add(key_len)
        .ok_or_else(|| candle_core::Error::Msg("attention key range overflow".into()))?;
    let mut has_masked_positions = false;
    let values = (0..query_len)
        .flat_map(|query| {
            let query_position = query_start + query;
            (0..key_len).map(move |key| {
                let key_position = key_start + key;
                let future = key_position > query_position;
                let outside_window = match kind {
                    AttentionKind::FullCausal => false,
                    AttentionKind::SlidingCausal { window } => {
                        query_position.saturating_sub(key_position) >= window
                    }
                };
                if future || outside_window {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .inspect(|value| has_masked_positions |= value.is_infinite())
        .collect::<Vec<_>>();
    if !has_masked_positions {
        return Ok(None);
    }
    let mask = Tensor::from_vec(values, (query_len, key_len), device)?;
    Ok(Some(mask.to_dtype(dtype)?.unsqueeze(0)?.unsqueeze(0)?))
}

/// Dense full-causal attention over a materialized K/V history.
///
/// This is the reference attention path: the query heads are grouped by the
/// KV head they read, so the K/V history is never expanded, the scores are
/// scaled and optionally softcapped, the additive mask is applied and the
/// softmax reduces in F32 before the value matmul.
///
/// Grouped-query attention keeps the K/V history at `num_kv_heads` and
/// gathers the query heads instead: query head `h` reads KV head
/// `h / kv_groups`, the pairing the expanded K/V used to encode. The query
/// rows are therefore reshaped to `(batch, kv_heads, kv_groups * query_len,
/// head_dim)` and the scores are computed against the compact K/V.
pub(crate) fn dense_causal_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    kv_groups: usize,
    scale: f64,
    logit_softcap: Option<f64>,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (batch, query_heads, query_len, head_dim) = query.dims4()?;
    let (key_batch, kv_heads, key_len, key_head_dim) = key.dims4()?;
    let (value_batch, value_heads, value_len, value_head_dim) = value.dims4()?;
    if kv_groups == 0 || query_heads != kv_heads * kv_groups {
        candle_core::bail!(
            "grouped attention expects {kv_groups} query heads per KV head, got \
             {query_heads} query heads and {kv_heads} KV heads"
        );
    }
    if key_batch != batch
        || value_batch != batch
        || value_heads != kv_heads
        || value_len != key_len
        || key_head_dim != head_dim
        || value_head_dim != head_dim
    {
        candle_core::bail!(
            "grouped attention expects matching query, key and value shapes, got \
             {batch}x{query_heads}x{query_len}x{head_dim}, {key_batch}x{kv_heads}x{key_len}x\
             {key_head_dim} and {value_batch}x{value_heads}x{value_len}x{value_head_dim}"
        );
    }
    // The matmul reads the K/V history one head block at a time. The
    // compact cache views already have that layout, so the history is not
    // copied; a view that interleaves the heads is copied once.
    let key = if matmul_ready_kv(key, batch, kv_heads, key_len, head_dim) {
        key.clone()
    } else {
        key.contiguous()?
    };
    let value = if matmul_ready_kv(value, batch, kv_heads, key_len, head_dim) {
        value.clone()
    } else {
        value.contiguous()?
    };
    // Group the query heads by their KV head. The head order is the one the
    // expanded K/V used to produce, so query head `h` still reads KV head
    // `h / kv_groups`.
    let grouped_query = query.reshape((batch, kv_heads, kv_groups * query_len, head_dim))?;
    let scores = grouped_query
        .matmul(&key.transpose(2, 3)?)?
        .affine(scale, 0.0)?;
    let scores = match logit_softcap {
        Some(softcap) => tanh_softcap(&scores, softcap)?,
        None => scores,
    };
    let scores = match mask {
        Some(mask) => {
            // The mask is per query row, so it broadcasts over the group axis
            // the grouped query introduced.
            let scores = scores.reshape((batch, kv_heads, kv_groups, query_len, key_len))?;
            let mask = grouped_mask(mask)?;
            scores.broadcast_add(&mask)?.reshape((
                batch,
                kv_heads,
                kv_groups * query_len,
                key_len,
            ))?
        }
        None => scores,
    };
    // Match reference implementations: reduction in F32, model dtype after.
    let score_dtype = scores.dtype();
    let probabilities =
        candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(score_dtype)?;
    let output = probabilities.matmul(&value)?;
    output.reshape((batch, query_heads, query_len, head_dim))
}

/// Whether the attention matmul can read a K/V history as it is.
///
/// The matmul reads one head block at a time, so the last two axes of a
/// block must be contiguous and the heads must be evenly spaced. The
/// compact cache views satisfy both; a view that interleaves the heads
/// does not and is copied once.
fn matmul_ready_kv(
    tensor: &Tensor,
    batch: usize,
    heads: usize,
    sequence: usize,
    head_dim: usize,
) -> bool {
    let strides = tensor.stride();
    let (batch_stride, head_stride, sequence_stride, dim_stride) =
        (strides[0], strides[1], strides[2], strides[3]);
    let block = sequence * head_dim;
    // The block of one head is either head-major or sequence-major. Those are
    // the two layouts the matmul reads without a copy.
    let contiguous_block = (dim_stride == 1 && sequence_stride == head_dim)
        || (sequence_stride == 1 && dim_stride == sequence);
    // The head blocks are evenly spaced, so one batch element starts
    // `heads` blocks after the previous one.
    let spaced = head_stride >= block && (batch == 1 || batch_stride == head_stride * heads);
    contiguous_block && spaced
}

/// Reshape an additive mask so it broadcasts over the group axis of the
/// grouped scores `(batch, kv_heads, groups, query_len, key_len)`.
///
/// The mask keeps its own leading dimensions and gains a group axis of one
/// right before the query and key axes.
fn grouped_mask(mask: &Tensor) -> Result<Tensor> {
    let dims = mask.dims();
    match dims {
        [query_len, key_len] => mask.reshape((1_usize, 1, 1, *query_len, *key_len)),
        [leading, query_len, key_len] => mask.reshape((*leading, 1_usize, 1, *query_len, *key_len)),
        [batch, leading, query_len, key_len] => {
            mask.reshape((*batch, *leading, 1_usize, *query_len, *key_len))
        }
        _ => candle_core::bail!("attention mask with {} axes is not supported", dims.len()),
    }
}

fn tanh_softcap(input: &Tensor, softcap: f64) -> Result<Tensor> {
    input
        .affine(1.0 / softcap, 0.0)?
        .tanh()?
        .affine(softcap, 0.0)
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, collections::HashMap};

    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;

    use crate::{
        architecture::{PositionEncodingSpec, ProjectionBias, RopeSpec},
        cache::{KvCache, KvCachePolicy},
        generation::{
            FinishReason, GenerationConfig, generate, generate_with_control, run_generation_request,
        },
        sampling::{Sampler, SamplingConfig},
    };

    use super::*;

    fn tiny_spec() -> DecoderSpec {
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
            runtime: crate::architecture::RuntimeSpec {
                vocab_size: 32,
                max_context: 16,
                stop_token_ids: vec![2],
            },
            hidden_size: 8,
            intermediate_size: 16,
            num_layers,
            num_query_heads: 2,
            num_kv_heads: 1,
            head_dim,
            norm_epsilon: 1e-5,
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
            tied_embeddings: true,
        }
    }

    /// A decoder geometry whose head dimension Candle's fused SDPA kernel
    /// instantiates, so the fused prefill path can be compared with the
    /// native paged path.
    fn tiny_fused_spec() -> DecoderSpec {
        let head_dim = 64;
        let num_layers = 2;
        let rope = RopeSpec {
            theta: 10_000.0,
            layout: RopeLayout::HalfSplit,
            rotary_dim: head_dim,
            frequency_factors: None,
        };
        DecoderSpec {
            architecture: "test".into(),
            runtime: crate::architecture::RuntimeSpec {
                vocab_size: 32,
                max_context: 64,
                stop_token_ids: vec![2],
            },
            hidden_size: 4 * head_dim,
            intermediate_size: 8 * head_dim,
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
                query: true,
                key: true,
                value: true,
                output: false,
            },
            mlp_bias: false,
            tied_embeddings: true,
        }
    }

    fn tiny_hybrid_spec() -> DecoderSpec {
        let mut spec = tiny_spec();
        spec.runtime.max_context = 32;
        spec.layer_specs[0].mixer =
            SequenceMixerSpec::ShortConvolution(ShortConvolutionSpec { kernel_size: 3 });
        spec
    }

    fn new_session(model: &DecoderModel) -> Result<Box<dyn CausalLmSession>> {
        model.new_session()
    }

    fn configure_multi_session_test_pool(model: &mut DecoderModel) -> Result<()> {
        let config =
            PagedKvPoolConfig::new(1, 8, model.spec.runtime.max_context).map_err(paged_kv_error)?;
        model.configure_kv_cache(config)
    }

    fn cache_stats(session: &dyn CausalLmSession) -> Vec<KvCacheStats> {
        session
            .kv_cache_stats()
            .expect("shared decoder sessions expose per-layer KV stats")
    }

    fn logical_cache_state(session: &dyn CausalLmSession) -> Vec<(usize, usize, usize)> {
        cache_stats(session)
            .into_iter()
            .map(|stats| (stats.logical_len, stats.resident_start, stats.resident_len))
            .collect()
    }

    fn decoder_session(session: &dyn CausalLmSession) -> &DecoderSession {
        session
            .as_any()
            .downcast_ref::<DecoderSession>()
            .expect("shared decoder creates DecoderSession state")
    }

    #[test]
    fn automatic_prefix_cache_restores_an_aligned_dense_prefix_with_logit_parity() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        model.configure_kv_cache(PagedKvPoolConfig::new(2, 4, 16).map_err(paged_kv_error)?)?;
        model.configure_prefix_cache(16)?;

        let stored_prompt = [1_u32, 2, 3, 4, 5, 6, 7, 8];
        let mut source = new_session(&model)?;
        assert_eq!(
            model
                .reset_session_for_prompt(source.as_mut(), &stored_prompt)?
                .cached_tokens,
            0
        );
        let logits = model.forward_tokens(source.as_mut(), &stored_prompt, 0)?;
        let _ = model.read_output_to_cpu(&logits)?;
        model.publish_prompt_prefix(source.as_ref(), &stored_prompt);

        let divergent_prompt = [1_u32, 2, 3, 4, 9];
        let mut reused = new_session(&model)?;
        let hit = model.reset_session_for_prompt(reused.as_mut(), &divergent_prompt)?;
        assert_eq!(hit.cached_tokens, 4);
        assert_eq!(reused.cache_len(), 4);
        let cached = model.forward_tokens(reused.as_mut(), &[9], 4)?;

        let control =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut control_session = new_session(&control)?;
        let cold = control.forward_tokens(control_session.as_mut(), &divergent_prompt, 0)?;
        let difference = max_difference(&cached, &cold)?;
        assert!(
            difference < 1e-5,
            "restored prefix/cold logits differed by {difference}"
        );

        let cache_stats = model
            .prefix_cache
            .as_ref()
            .expect("prefix cache is configured")
            .borrow()
            .index
            .stats();
        assert_eq!(cache_stats.hits, 1);
        assert_eq!(cache_stats.misses, 1);
        assert_eq!(cache_stats.matched_tokens, 4);
        let pool_stats = model
            .paged_pool
            .as_ref()
            .expect("dense decoder has a paged pool")
            .stats()
            .map_err(paged_kv_error)?;
        assert_eq!(pool_stats.shared_blocks, 4);
        assert_eq!(pool_stats.live_blocks, 5);
        assert_eq!(pool_stats.live_references, 11);
        let diagnostics = model
            .prefix_cache_diagnostics()?
            .expect("configured prefix cache exposes diagnostics");
        assert_eq!(diagnostics.hits, 1);
        assert_eq!(diagnostics.matched_tokens, 4);
        assert_eq!(diagnostics.resident_snapshot_blocks, 4);
        assert_eq!(diagnostics.block_tokens, 2);
        assert_eq!(diagnostics.physical_shared_blocks, 4);
        assert_eq!(diagnostics.physical_live_references, 11);
        Ok(())
    }

    #[test]
    fn automatic_prefix_cache_evicts_before_reporting_physical_exhaustion() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        model.configure_kv_cache(PagedKvPoolConfig::new(2, 3, 3).map_err(paged_kv_error)?)?;
        model.configure_prefix_cache(4)?;

        let cached_prompt = [1_u32, 2, 3, 4];
        let mut source = new_session(&model)?;
        let logits = model.forward_tokens(source.as_mut(), &cached_prompt, 0)?;
        let _ = model.read_output_to_cpu(&logits)?;
        model.publish_prompt_prefix(source.as_ref(), &cached_prompt);
        source.clear_cache();

        let mut pressure = new_session(&model)?;
        let uncached_prompt = [9_u32, 10, 11, 12, 13];
        assert_eq!(
            model
                .reset_session_for_prompt(pressure.as_mut(), &uncached_prompt)?
                .cached_tokens,
            0
        );
        let logits = model.forward_tokens(pressure.as_mut(), &uncached_prompt, 0)?;
        assert_eq!(logits.dims2()?, (1, 32));
        let stats = model
            .prefix_cache
            .as_ref()
            .expect("prefix cache is configured")
            .borrow()
            .index
            .stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.evictions, 1);
        Ok(())
    }

    #[test]
    fn automatic_prefix_cache_restores_shortconv_and_kv_with_partial_page_cow() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_hybrid_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        model.configure_kv_cache(PagedKvPoolConfig::new(2, 4, 32).map_err(paged_kv_error)?)?;
        model.configure_prefix_cache(32)?;

        let stored_prompt = [1_u32, 2, 3, 4, 5, 6];
        let mut source = new_session(&model)?;
        let logits = model.forward_tokens(source.as_mut(), &stored_prompt[..3], 0)?;
        let _ = model.read_output_to_cpu(&logits)?;
        let checkpoint_tail = shortconv_state(source.as_ref(), 0)
            .tail
            .as_ref()
            .expect("prefix creates a convolution tail")
            .clone();
        model.publish_prompt_prefix(source.as_ref(), &stored_prompt[..3]);

        // Advancing from a three-token snapshot exercises both immutable
        // recurrent handles and paged-KV COW for the shared partial page.
        let logits = model.forward_tokens(source.as_mut(), &stored_prompt[3..], 3)?;
        let _ = model.read_output_to_cpu(&logits)?;
        model.publish_prompt_prefix(source.as_ref(), &stored_prompt);

        let divergent_prompt = [1_u32, 2, 3, 9];
        let mut reused = new_session(&model)?;
        let hit = model.reset_session_for_prompt(reused.as_mut(), &divergent_prompt)?;
        assert_eq!(hit.cached_tokens, 3);
        assert!(
            max_difference(
                shortconv_state(reused.as_ref(), 0)
                    .tail
                    .as_ref()
                    .expect("joint snapshot restores the convolution tail"),
                &checkpoint_tail,
            )? < f32::EPSILON
        );
        let cached = model.forward_tokens(reused.as_mut(), &[9], 3)?;

        let control =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut cold_session = new_session(&control)?;
        let cold = control.forward_tokens(cold_session.as_mut(), &divergent_prompt, 0)?;
        assert!(max_difference(&cached, &cold)? < 1e-5);

        let diagnostics = model
            .prefix_cache_diagnostics()?
            .expect("hybrid prefix cache exposes diagnostics");
        assert_eq!(diagnostics.hits, 1);
        assert_eq!(diagnostics.matched_tokens, 3);
        assert_eq!(diagnostics.entries, 2);
        // Prefixes of three and six tokens own two and three KV pages, plus
        // one KV-block-equivalent short-convolution checkpoint each.
        assert_eq!(diagnostics.resident_snapshot_blocks, 7);
        assert!(diagnostics.physical_shared_blocks > 0);
        Ok(())
    }

    #[test]
    fn automatic_prefix_cache_rejects_mutable_sliding_state() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        attention_spec_mut(&mut spec, 0).kind = AttentionKind::SlidingCausal { window: 4 };
        let mut model = DecoderModel::load_dense(spec, VarBuilder::zeros(DType::F32, &device))?;
        let error = model
            .configure_prefix_cache(16)
            .expect_err("mutable sliding rings cannot be immutable prefix snapshots");
        assert!(
            error
                .to_string()
                .contains("sliding-attention storage is mutable")
        );
        Ok(())
    }

    #[test]
    fn last_hidden_projects_to_the_existing_logits() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.final_logit_scale = 0.75;
        spec.final_logit_softcap = Some(2.5);
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors, DType::F32, &device),
        )?;
        configure_multi_session_test_pool(&mut model)?;
        let tokens = [1_u32, 7, 3, 9];

        let mut hidden_session = new_session(&model)?;
        let hidden = model.forward_last_hidden_tokens(hidden_session.as_mut(), &tokens, 0)?;
        let hidden = model.read_output_to_cpu(&hidden)?;
        assert_eq!(hidden.dims2()?, (1, spec.hidden_size));

        let mut logits_session = new_session(&model)?;
        let logits = model.forward_tokens(logits_session.as_mut(), &tokens, 0)?;
        let projected = hidden
            .apply(&model.lm_head)?
            .affine(spec.final_logit_scale, 0.0)?;
        let expected = tanh_softcap(
            &projected,
            spec.final_logit_softcap.expect("test enables softcap"),
        )?;

        assert!(max_difference(&expected, &logits)? < 1e-6);
        assert_eq!(hidden_session.cache_len(), tokens.len());
        assert_eq!(logits_session.cache_len(), tokens.len());
        Ok(())
    }

    #[test]
    fn packed_ragged_last_hidden_matches_serial_lane_order() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors, DType::F32, &device),
        )?;
        model.configure_kv_cache(
            PagedKvPoolConfig::new(2, 16, spec.runtime.max_context).map_err(paged_kv_error)?,
        )?;
        let first_tokens = [1_u32, 2, 3];
        let second_tokens = [11_u32];
        let mut first = new_session(&model)?;
        let mut second = new_session(&model)?;

        let batched = {
            let mut inputs = [
                CausalLmBatchInput::new(first.as_mut(), &first_tokens, 0),
                CausalLmBatchInput::new(second.as_mut(), &second_tokens, 0),
            ];
            model.forward_batch_last_hidden_tokens(&mut inputs)?
        };
        assert_eq!(batched.dims2()?, (2, spec.hidden_size));

        let mut first_serial = new_session(&model)?;
        let mut second_serial = new_session(&model)?;
        let first_expected =
            model.forward_last_hidden_tokens(first_serial.as_mut(), &first_tokens, 0)?;
        let second_expected =
            model.forward_last_hidden_tokens(second_serial.as_mut(), &second_tokens, 0)?;

        assert!(max_difference(&first_expected, &second_expected)? > 1e-5);
        assert!(max_difference(&batched.narrow(0, 0, 1)?, &first_expected)? < 1e-6);
        assert!(max_difference(&batched.narrow(0, 1, 1)?, &second_expected)? < 1e-6);
        assert_eq!(first.cache_len(), first_tokens.len());
        assert_eq!(second.cache_len(), second_tokens.len());
        Ok(())
    }

    #[test]
    fn last_hidden_bypasses_the_language_model_head() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors, DType::F32, &device),
        )?;
        model.lm_head = LinearOp::Dense(candle_nn::Linear::new(
            Tensor::zeros(
                (spec.runtime.vocab_size, spec.hidden_size + 1),
                DType::F32,
                &device,
            )?,
            None,
        ));
        let tokens = [1_u32, 2, 3];

        let mut hidden_session = new_session(&model)?;
        let hidden = model.forward_last_hidden_tokens(hidden_session.as_mut(), &tokens, 0)?;
        assert_eq!(hidden.dims2()?, (1, spec.hidden_size));

        let mut logits_session = new_session(&model)?;
        assert!(
            model
                .forward_tokens(logits_session.as_mut(), &tokens, 0)
                .is_err()
        );
        assert_eq!(logits_session.cache_len(), 0);
        Ok(())
    }

    #[test]
    fn last_hidden_finalize_failure_rolls_back_every_ragged_lane() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors, DType::F32, &device),
        )?;
        model.configure_kv_cache(
            PagedKvPoolConfig::new(2, 8, spec.runtime.max_context).map_err(paged_kv_error)?,
        )?;
        model.final_norm = RmsNorm::new(
            Tensor::ones(spec.hidden_size + 1, DType::F32, &device)?,
            spec.norm_epsilon,
        );
        let first_tokens = [1_u32, 2, 3];
        let second_tokens = [11_u32];
        let mut first = new_session(&model)?;
        let mut second = new_session(&model)?;

        let error = {
            let mut inputs = [
                CausalLmBatchInput::new(first.as_mut(), &first_tokens, 0),
                CausalLmBatchInput::new(second.as_mut(), &second_tokens, 0),
            ];
            model
                .forward_batch_last_hidden_tokens(&mut inputs)
                .expect_err("an invalid final norm must fail before commit")
        };
        assert!(!error.to_string().is_empty());
        assert_eq!(first.cache_len(), 0);
        assert_eq!(second.cache_len(), 0);

        model.final_norm = RmsNorm::new(
            Tensor::ones(spec.hidden_size, DType::F32, &device)?,
            spec.norm_epsilon,
        );
        let retried = {
            let mut inputs = [
                CausalLmBatchInput::new(first.as_mut(), &first_tokens, 0),
                CausalLmBatchInput::new(second.as_mut(), &second_tokens, 0),
            ];
            model.forward_batch_last_hidden_tokens(&mut inputs)?
        };
        assert_eq!(retried.dims2()?, (2, spec.hidden_size));
        assert_eq!(first.cache_len(), first_tokens.len());
        assert_eq!(second.cache_len(), second_tokens.len());
        Ok(())
    }

    #[test]
    fn automatic_prefix_cache_restores_gated_delta_joint_snapshot() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_gated_delta_decoder_spec();
        let tensors = tiny_gated_delta_decoder_tensors(&spec, &device)?;
        let mut model = load_tiny_gated_delta_decoder(spec.clone(), tensors.clone(), &device)?;
        // This synthetic Qwen3.5 trunk is recurrent-only, proving that the
        // snapshot contract is not accidentally dependent on a KV table.
        model.configure_prefix_cache(32)?;

        let prefix = [1_u32, 2, 3];
        let mut source = new_session(&model)?;
        let _ = model.forward_tokens(source.as_mut(), &prefix, 0)?;
        let checkpoint_tail = gated_delta_state(source.as_ref(), 0)
            .convolution_tail
            .as_ref()
            .expect("prefix creates a Gated DeltaNet convolution tail")
            .clone();
        let checkpoint_recurrent = gated_delta_state(source.as_ref(), 0)
            .recurrent
            .as_ref()
            .expect("prefix creates a Gated DeltaNet recurrence")
            .clone();
        model.publish_prompt_prefix(source.as_ref(), &prefix);
        let _ = model.forward_tokens(source.as_mut(), &[4], prefix.len())?;

        let prompt = [1_u32, 2, 3, 5];
        let mut reused = new_session(&model)?;
        assert_eq!(
            model
                .reset_session_for_prompt(reused.as_mut(), &prompt)?
                .cached_tokens,
            prefix.len()
        );
        let restored = gated_delta_state(reused.as_ref(), 0);
        assert!(
            max_difference(
                restored
                    .convolution_tail
                    .as_ref()
                    .expect("snapshot restores convolution state"),
                &checkpoint_tail,
            )? < f32::EPSILON
        );
        assert!(
            max_difference(
                restored
                    .recurrent
                    .as_ref()
                    .expect("snapshot restores recurrent state"),
                &checkpoint_recurrent,
            )? < f32::EPSILON
        );
        let cached = model.forward_tokens(reused.as_mut(), &[5], prefix.len())?;

        let control = load_tiny_gated_delta_decoder(spec, tensors, &device)?;
        let mut cold_session = new_session(&control)?;
        let cold = control.forward_tokens(cold_session.as_mut(), &prompt, 0)?;
        assert!(max_difference(&cached, &cold)? < 1e-5);
        let diagnostics = model
            .prefix_cache_diagnostics()?
            .expect("recurrent-only prefix cache exposes diagnostics");
        assert_eq!(diagnostics.resident_snapshot_blocks, 1);
        assert_eq!(diagnostics.physical_live_blocks, 0);
        Ok(())
    }

    #[test]
    fn recurrent_only_prefix_budget_charges_logical_span_and_evicts_lru() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_gated_delta_decoder_spec();
        spec.runtime.max_context = 32;
        let tensors = tiny_gated_delta_decoder_tensors(&spec, &device)?;
        let mut model = load_tiny_gated_delta_decoder(spec, tensors, &device)?;
        // Two 16-token cost units: a one-token endpoint costs one, while a
        // seventeen-token endpoint costs two and must evict the former.
        model.configure_prefix_cache(32)?;

        let tokens = [1_u32, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5];
        let mut source = new_session(&model)?;
        let _ = model.forward_tokens(source.as_mut(), &tokens[..1], 0)?;
        model.publish_prompt_prefix(source.as_ref(), &tokens[..1]);
        let first = model
            .prefix_cache
            .as_ref()
            .expect("prefix cache is configured")
            .borrow()
            .index
            .stats();
        assert_eq!(first.entries, 1);
        assert_eq!(first.cost, 1);

        let _ = model.forward_tokens(source.as_mut(), &tokens[1..], 1)?;
        model.publish_prompt_prefix(source.as_ref(), &tokens);
        let final_stats = model
            .prefix_cache
            .as_ref()
            .expect("prefix cache is configured")
            .borrow()
            .index
            .stats();
        assert_eq!(final_stats.entries, 1);
        assert_eq!(final_stats.cost, 2);
        assert_eq!(final_stats.evictions, 1);
        Ok(())
    }

    #[test]
    fn automatic_prefix_cache_restores_parallel_mamba2_and_attention_snapshot() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_parallel_mamba2_decoder_spec();
        let tensors = tiny_parallel_mamba2_decoder_tensors(&spec, &device)?;
        let mut model = load_tiny_parallel_mamba2_decoder(spec.clone(), tensors.clone(), &device)?;
        model.configure_kv_cache(PagedKvPoolConfig::new(2, 4, 32).map_err(paged_kv_error)?)?;
        model.configure_prefix_cache(16)?;

        let prefix = [1_u32, 2, 3];
        let mut source = new_session(&model)?;
        let logits = model.forward_tokens(source.as_mut(), &prefix, 0)?;
        let _ = model.read_output_to_cpu(&logits)?;
        let checkpoint_tail = mamba2_state(source.as_ref(), 0)
            .convolution_tail
            .as_ref()
            .expect("prefix creates a Mamba2 convolution tail")
            .clone();
        let checkpoint_recurrent = mamba2_state(source.as_ref(), 0)
            .recurrent
            .as_ref()
            .expect("prefix creates a Mamba2 recurrence")
            .clone();
        model.publish_prompt_prefix(source.as_ref(), &prefix);
        let _ = model.forward_tokens(source.as_mut(), &[4], prefix.len())?;

        let prompt = [1_u32, 2, 3, 5];
        let mut reused = new_session(&model)?;
        assert_eq!(
            model
                .reset_session_for_prompt(reused.as_mut(), &prompt)?
                .cached_tokens,
            prefix.len()
        );
        let restored = mamba2_state(reused.as_ref(), 0);
        assert!(
            max_difference(
                restored
                    .convolution_tail
                    .as_ref()
                    .expect("snapshot restores Mamba2 convolution state"),
                &checkpoint_tail,
            )? < f32::EPSILON
        );
        assert!(
            max_difference(
                restored
                    .recurrent
                    .as_ref()
                    .expect("snapshot restores Mamba2 recurrence"),
                &checkpoint_recurrent,
            )? < f32::EPSILON
        );
        let cached = model.forward_tokens(reused.as_mut(), &[5], prefix.len())?;

        let control = load_tiny_parallel_mamba2_decoder(spec, tensors, &device)?;
        let mut cold_session = new_session(&control)?;
        let cold = control.forward_tokens(cold_session.as_mut(), &prompt, 0)?;
        assert!(max_difference(&cached, &cold)? < 1e-5);
        let diagnostics = model
            .prefix_cache_diagnostics()?
            .expect("parallel Mamba2 cache exposes diagnostics");
        assert_eq!(diagnostics.hits, 1);
        assert!(diagnostics.resident_snapshot_blocks > 2);
        assert!(diagnostics.physical_shared_blocks > 0);
        Ok(())
    }

    fn shortconv_state(session: &dyn CausalLmSession, layer: usize) -> &ShortConvolutionState {
        let LayerSequenceState::ShortConvolution(state) =
            &decoder_session(session).layer_states[layer]
        else {
            panic!("test layer uses short convolution")
        };
        state
    }

    fn attention_spec_mut(spec: &mut DecoderSpec, layer: usize) -> &mut AttentionSpec {
        spec.layer_specs[layer]
            .attention_mut()
            .expect("test layer uses attention")
    }

    fn layer_attention(model: &DecoderModel, layer: usize) -> &SelfAttention {
        let SequenceMixer::Attention(attention) = &model.layers[layer].mixer else {
            panic!("test layer uses attention")
        };
        attention
    }

    fn layer_attention_mut(model: &mut DecoderModel, layer: usize) -> &mut SelfAttention {
        let SequenceMixer::Attention(attention) = &mut model.layers[layer].mixer else {
            panic!("test layer uses attention")
        };
        attention
    }

    fn patterned_tensor(
        shape: &[usize],
        seed: usize,
        scale: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let elements = shape.iter().product::<usize>();
        let values = (0..elements)
            .map(|index| {
                let centered = ((index * 13 + seed * 7) % 29) as f32 - 14.0;
                centered * scale
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, device)
    }

    #[allow(clippy::too_many_lines)] // Explicit tensor names keep the fixture loader-realistic.
    fn patterned_tiny_tensors(
        spec: &DecoderSpec,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".into(),
            patterned_tensor(
                &[spec.runtime.vocab_size, spec.hidden_size],
                1,
                0.015,
                device,
            )?,
        );
        tensors.insert(
            "model.norm.weight".into(),
            Tensor::ones(spec.hidden_size, DType::F32, device)?,
        );
        for layer in 0..spec.num_layers {
            let norm_names: &[&str] = match spec.layer_specs[layer].norm_layout {
                NormLayout::PreNorm => &["input_layernorm", "post_attention_layernorm"],
                NormLayout::Sandwich => &[
                    "input_layernorm",
                    "pre_feedforward_layernorm",
                    "post_attention_norm",
                    "post_feedforward_layernorm",
                ],
            };
            for name in norm_names {
                tensors.insert(
                    format!("model.layers.{layer}.{name}.weight"),
                    Tensor::ones(spec.hidden_size, DType::F32, device)?,
                );
            }
            for (seed, name, rows, bias) in [
                (
                    10,
                    "self_attn.q_proj",
                    spec.num_query_heads * spec.head_dim,
                    spec.projection_bias.query,
                ),
                (
                    20,
                    "self_attn.k_proj",
                    spec.num_kv_heads * spec.head_dim,
                    spec.projection_bias.key,
                ),
                (
                    30,
                    "self_attn.v_proj",
                    spec.num_kv_heads * spec.head_dim,
                    spec.projection_bias.value,
                ),
            ] {
                tensors.insert(
                    format!("model.layers.{layer}.{name}.weight"),
                    patterned_tensor(&[rows, spec.hidden_size], seed + layer, 0.01, device)?,
                );
                if bias {
                    tensors.insert(
                        format!("model.layers.{layer}.{name}.bias"),
                        patterned_tensor(&[rows], seed + layer + 1, 0.002, device)?,
                    );
                }
            }
            tensors.insert(
                format!("model.layers.{layer}.self_attn.o_proj.weight"),
                patterned_tensor(
                    &[spec.hidden_size, spec.num_query_heads * spec.head_dim],
                    40 + layer,
                    0.01,
                    device,
                )?,
            );
            if spec.projection_bias.output {
                tensors.insert(
                    format!("model.layers.{layer}.self_attn.o_proj.bias"),
                    patterned_tensor(&[spec.hidden_size], 41 + layer, 0.002, device)?,
                );
            }
            if spec.layer_specs[layer]
                .attention()
                .is_some_and(|attention| attention.qk_norm)
            {
                for (seed, name) in [(42, "q_norm"), (43, "k_norm")] {
                    tensors.insert(
                        format!("model.layers.{layer}.self_attn.{name}.weight"),
                        patterned_tensor(&[spec.head_dim], seed + layer, 0.05, device)?,
                    );
                }
            }
            if let SequenceMixerSpec::ShortConvolution(shortconv) = &spec.layer_specs[layer].mixer {
                tensors.insert(
                    format!("model.layers.{layer}.shortconv.in_proj.weight"),
                    patterned_tensor(
                        &[spec.hidden_size * 3, spec.hidden_size],
                        44 + layer,
                        0.02,
                        device,
                    )?,
                );
                tensors.insert(
                    format!("model.layers.{layer}.shortconv.conv.weight"),
                    patterned_tensor(
                        &[spec.hidden_size, shortconv.kernel_size],
                        45 + layer,
                        0.03,
                        device,
                    )?,
                );
                tensors.insert(
                    format!("model.layers.{layer}.shortconv.out_proj.weight"),
                    patterned_tensor(
                        &[spec.hidden_size, spec.hidden_size],
                        46 + layer,
                        0.02,
                        device,
                    )?,
                );
            }
            for (seed, name, rows, columns) in [
                (
                    50,
                    "mlp.gate_proj",
                    spec.intermediate_size,
                    spec.hidden_size,
                ),
                (60, "mlp.up_proj", spec.intermediate_size, spec.hidden_size),
                (
                    70,
                    "mlp.down_proj",
                    spec.hidden_size,
                    spec.intermediate_size,
                ),
            ] {
                tensors.insert(
                    format!("model.layers.{layer}.{name}.weight"),
                    patterned_tensor(&[rows, columns], seed + layer, 0.008, device)?,
                );
                if spec.mlp_bias {
                    tensors.insert(
                        format!("model.layers.{layer}.{name}.bias"),
                        patterned_tensor(&[rows], seed + layer + 1, 0.002, device)?,
                    );
                }
            }
        }
        if !spec.tied_embeddings {
            tensors.insert(
                "lm_head.weight".into(),
                patterned_tensor(
                    &[spec.runtime.vocab_size, spec.hidden_size],
                    90,
                    0.015,
                    device,
                )?,
            );
        }
        Ok(tensors)
    }

    fn without_model_namespace(tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        tensors
            .into_iter()
            .map(|(name, tensor)| {
                (
                    name.strip_prefix("model.").unwrap_or(&name).to_owned(),
                    tensor,
                )
            })
            .collect()
    }

    #[test]
    fn dense_decoder_selects_one_adapter_declared_namespace() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();

        let root = without_model_namespace(patterned_tiny_tensors(&spec, &device)?);
        DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(root, DType::F32, &device),
        )?;

        let mut custom_root = without_model_namespace(patterned_tiny_tensors(&spec, &device)?);
        let embedding = custom_root
            .remove("embed_tokens.weight")
            .expect("patterned fixture has an embedding");
        custom_root.insert("tokens.weight".into(), embedding);
        let source =
            DenseWeightSource::new(VarBuilder::from_tensors(custom_root, DType::F32, &device));
        let custom_names = DecoderWeightNames {
            model: "decoder",
            embedding: "tokens",
            ..STANDARD_DECODER_WEIGHT_NAMES
        };
        DecoderModel::load_with_weight_names(spec.clone(), &source, &custom_names)?;

        let mut root_embedding_with_model_layers = patterned_tiny_tensors(&spec, &device)?;
        let embedding = root_embedding_with_model_layers
            .remove("model.embed_tokens.weight")
            .expect("patterned fixture has an embedding");
        root_embedding_with_model_layers.insert("embed_tokens.weight".into(), embedding);
        assert!(
            DecoderModel::load_dense(
                spec.clone(),
                VarBuilder::from_tensors(root_embedding_with_model_layers, DType::F32, &device,),
            )
            .is_err(),
            "a root decoder must not mix in model.* layers"
        );

        let mut model_embedding_with_root_layers =
            without_model_namespace(patterned_tiny_tensors(&spec, &device)?);
        let embedding = model_embedding_with_root_layers
            .remove("embed_tokens.weight")
            .expect("patterned fixture has an embedding");
        model_embedding_with_root_layers.insert("model.embed_tokens.weight".into(), embedding);
        assert!(
            DecoderModel::load_dense(
                spec,
                VarBuilder::from_tensors(model_embedding_with_root_layers, DType::F32, &device,),
            )
            .is_err(),
            "a model.* decoder must not mix in root layers"
        );
        Ok(())
    }

    fn tiny_gated_delta_spec() -> GatedDeltaNetSpec {
        GatedDeltaNetSpec {
            convolution_kernel_size: 3,
            num_key_heads: 1,
            num_value_heads: 1,
            key_head_dim: 2,
            value_head_dim: 2,
            qk_norm_epsilon: 1e-6,
        }
    }

    fn tiny_gated_delta_mixer(device: &Device) -> Result<GatedDeltaNet> {
        let spec = tiny_gated_delta_spec();
        Ok(GatedDeltaNet {
            qkv: LinearOp::Dense(candle_nn::Linear::new(
                Tensor::from_vec(
                    vec![
                        1.0_f32, 0.0, // q0
                        0.0, 1.0, // q1
                        0.5, -0.25, // k0
                        0.25, 0.75, // k1
                        0.8, 0.1, // v0
                        -0.2, 0.6, // v1
                    ],
                    (6, 2),
                    device,
                )?,
                None,
            )),
            output_gate: LinearOp::Dense(candle_nn::Linear::new(
                Tensor::eye(2, DType::F32, device)?,
                None,
            )),
            alpha: LinearOp::Dense(candle_nn::Linear::new(
                Tensor::from_vec(vec![0.2_f32, -0.1], (1, 2), device)?,
                None,
            )),
            beta: LinearOp::Dense(candle_nn::Linear::new(
                Tensor::from_vec(vec![-0.3_f32, 0.4], (1, 2), device)?,
                None,
            )),
            convolution_kernel: Tensor::from_vec(
                vec![
                    0.2_f32, -0.1, 1.0, // q0
                    0.1, 0.3, 0.8, // q1
                    0.25, -0.2, 0.9, // k0
                    -0.15, 0.2, 1.1, // k1
                    0.3, 0.1, 0.7, // v0
                    -0.2, 0.25, 0.85, // v1
                ],
                (6, 3),
                device,
            )?,
            time_bias: Tensor::new(&[0.1_f32], device)?,
            state_decay: Tensor::new(&[-0.5_f32], device)?,
            output_norm: RmsNorm::new(Tensor::new(&[1.1_f32, 0.9], device)?, 1e-5),
            output: LinearOp::Dense(candle_nn::Linear::new(
                Tensor::from_vec(vec![0.7_f32, -0.2, 0.1, 0.8], (2, 2), device)?,
                None,
            )),
            spec,
            convolution_width: 6,
            key_width: 2,
            value_width: 2,
        })
    }

    fn tiny_gated_delta_input(device: &Device) -> Result<Tensor> {
        Tensor::new(
            &[[[0.25_f32, -0.5], [0.75, 0.1], [-0.2, 0.6], [0.4, -0.3]]],
            device,
        )
    }

    fn tiny_mamba2_spec() -> Mamba2Spec {
        Mamba2Spec {
            convolution_kernel_size: 2,
            inner_size: 1,
            state_size: 1,
            num_heads: 1,
            num_groups: 1,
            head_dim: 1,
            convolution_bias: true,
            input_projection_bias: false,
            output_projection_bias: false,
            gated_rms_norm: false,
        }
    }

    fn tiny_mamba2_mixer(device: &Device) -> Result<Mamba2> {
        let spec = tiny_mamba2_spec();
        Ok(Mamba2 {
            input: LinearOp::Dense(candle_nn::Linear::new(
                // Canonical projection order: [z, x, B, C, dt].
                Tensor::new(&[[0.5_f32], [1.0], [0.5], [-0.25], [0.2]], device)?,
                None,
            )),
            // Each row is [older, current]. Distinct coefficients make a
            // reversal of the causal convolution observable in the golden.
            convolution_kernel: Tensor::new(&[[0.25_f32, 1.0], [-0.5, 0.75], [0.6, -1.2]], device)?,
            convolution_bias: Some(Tensor::new(&[0.1_f32, -0.2, 0.05], device)?),
            time_bias: Tensor::new(&[0.15_f32], device)?,
            state_decay: Tensor::new(&[-0.7_f32], device)?,
            skip: Tensor::new(&[0.3_f32], device)?,
            gated_norm_weight: None,
            output: LinearOp::Dense(candle_nn::Linear::new(
                Tensor::new(&[[1.1_f32]], device)?,
                None,
            )),
            norm_epsilon: 1e-5,
            spec,
            convolution_width: 3,
            projection_width: 5,
        })
    }

    fn tiny_mamba2_projected_input(device: &Device) -> Result<Tensor> {
        Tensor::new(
            &[[
                [0.5_f32, 1.0, 0.5, -0.25, 0.2],
                [-0.4, -0.5, 0.75, 0.4, -0.1],
                [0.8, 0.25, -0.6, 0.9, 0.35],
            ]],
            device,
        )
    }

    fn tiny_parallel_mamba2_decoder_spec() -> DecoderSpec {
        let mut spec = tiny_spec();
        spec.architecture = "tiny_falcon_h1".into();
        let attention = spec.layer_specs[0]
            .attention()
            .expect("tiny base layer uses attention")
            .clone();
        let layer = DecoderLayerSpec {
            mixer: SequenceMixerSpec::ParallelAttentionMamba2(
                crate::architecture::ParallelAttentionMamba2Spec {
                    attention,
                    mamba2: tiny_mamba2_spec(),
                },
            ),
            activation: Activation::Silu,
            norm_layout: NormLayout::PreNorm,
            feed_forward: FeedForwardKind::Dense,
            mixer_residual_scale: 1.0,
            feed_forward_residual_scale: 1.0,
        };
        spec.layer_specs = vec![layer; spec.num_layers];
        spec
    }

    fn tiny_parallel_mamba2_decoder_tensors(
        spec: &DecoderSpec,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let mut tensors = patterned_tiny_tensors(spec, device)?;
        for layer in 0..spec.num_layers {
            for (standard, falcon) in [
                ("input_layernorm", "attn_norm"),
                ("post_attention_layernorm", "ffn_norm"),
            ] {
                let standard = format!("model.layers.{layer}.{standard}.weight");
                let tensor = tensors
                    .remove(&standard)
                    .expect("patterned fixture creates both layer norms");
                tensors.insert(format!("model.layers.{layer}.{falcon}.weight"), tensor);
            }
            let SequenceMixerSpec::ParallelAttentionMamba2(parallel) =
                &spec.layer_specs[layer].mixer
            else {
                continue;
            };
            let mamba = parallel.mamba2;
            let convolution_width = mamba2_convolution_width(mamba)?;
            let projection_width = mamba.inner_size + convolution_width + mamba.num_heads;
            let prefix = format!("model.layers.{layer}.mamba");
            tensors.insert(
                format!("{prefix}.in_proj.weight"),
                patterned_tensor(
                    &[projection_width, spec.hidden_size],
                    110 + layer,
                    0.025,
                    device,
                )?,
            );
            tensors.insert(
                format!("{prefix}.conv1d.weight"),
                patterned_tensor(
                    &[convolution_width, mamba.convolution_kernel_size],
                    120 + layer,
                    0.08,
                    device,
                )?,
            );
            tensors.insert(
                format!("{prefix}.conv1d.bias"),
                patterned_tensor(&[convolution_width], 130 + layer, 0.01, device)?,
            );
            tensors.insert(
                format!("{prefix}.dt_bias"),
                patterned_tensor(&[mamba.num_heads], 140 + layer, 0.02, device)?,
            );
            tensors.insert(
                format!("{prefix}.A_log"),
                Tensor::from_vec(
                    (0..mamba.num_heads)
                        .map(|head| -0.4_f32 - head as f32 * 0.1)
                        .collect::<Vec<_>>(),
                    mamba.num_heads,
                    device,
                )?,
            );
            tensors.insert(
                format!("{prefix}.D"),
                Tensor::ones(mamba.num_heads, DType::F32, device)?,
            );
            tensors.insert(
                format!("{prefix}.out_proj.weight"),
                patterned_tensor(
                    &[spec.hidden_size, mamba.inner_size],
                    150 + layer,
                    0.04,
                    device,
                )?,
            );
        }
        Ok(tensors)
    }

    fn load_tiny_parallel_mamba2_decoder(
        spec: DecoderSpec,
        tensors: HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<DecoderModel> {
        let source = DenseWeightSource::new(VarBuilder::from_tensors(tensors, DType::F32, device));
        DecoderModel::load_with_weight_names(
            spec,
            &source,
            &crate::architecture::FALCON_H1_WEIGHT_NAMES,
        )
    }

    fn mamba2_state(session: &dyn CausalLmSession, layer: usize) -> &Mamba2State {
        let LayerSequenceState::ParallelAttentionMamba2(state) =
            &decoder_session(session).layer_states[layer]
        else {
            panic!("test layer uses parallel attention/Mamba2")
        };
        state
    }

    fn layer_dense_feed_forward_mut(
        model: &mut DecoderModel,
        layer: usize,
    ) -> &mut DenseFeedForward {
        let FeedForward::Dense(feed_forward) = &mut model.layers[layer].feed_forward else {
            panic!("test layer uses a dense feed-forward block")
        };
        feed_forward
    }

    fn tiny_gated_delta_decoder_spec() -> DecoderSpec {
        let delta = tiny_gated_delta_spec();
        let layer = DecoderLayerSpec {
            mixer: SequenceMixerSpec::GatedDeltaNet(delta),
            activation: Activation::Silu,
            norm_layout: NormLayout::PreNorm,
            feed_forward: FeedForwardKind::Dense,
            mixer_residual_scale: 1.0,
            feed_forward_residual_scale: 1.0,
        };
        DecoderSpec {
            architecture: "tiny_qwen35_gated_delta".into(),
            runtime: crate::architecture::RuntimeSpec {
                vocab_size: 8,
                max_context: 16,
                stop_token_ids: vec![7],
            },
            hidden_size: 2,
            intermediate_size: 4,
            num_layers: 2,
            num_query_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            norm_epsilon: 1e-5,
            layer_specs: vec![layer; 2],
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

    fn tiny_gated_delta_decoder_tensors(
        spec: &DecoderSpec,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::from([
            (
                "model.embed_tokens.weight".into(),
                patterned_tensor(
                    &[spec.runtime.vocab_size, spec.hidden_size],
                    101,
                    0.08,
                    device,
                )?,
            ),
            (
                "model.norm.weight".into(),
                Tensor::ones(spec.hidden_size, DType::F32, device)?,
            ),
        ]);
        let mixer = tiny_gated_delta_mixer(device)?;
        let qkv = match &mixer.qkv {
            LinearOp::Dense(linear) => linear.weight().clone(),
            LinearOp::Quantized { .. } => unreachable!("tiny fixture uses dense weights"),
            #[cfg(feature = "metal")]
            LinearOp::FileBackedQuantized { .. } => {
                unreachable!("tiny fixture uses dense weights")
            }
        };
        let output_gate = match &mixer.output_gate {
            LinearOp::Dense(linear) => linear.weight().clone(),
            LinearOp::Quantized { .. } => unreachable!("tiny fixture uses dense weights"),
            #[cfg(feature = "metal")]
            LinearOp::FileBackedQuantized { .. } => {
                unreachable!("tiny fixture uses dense weights")
            }
        };
        let alpha = match &mixer.alpha {
            LinearOp::Dense(linear) => linear.weight().clone(),
            LinearOp::Quantized { .. } => unreachable!("tiny fixture uses dense weights"),
            #[cfg(feature = "metal")]
            LinearOp::FileBackedQuantized { .. } => {
                unreachable!("tiny fixture uses dense weights")
            }
        };
        let beta = match &mixer.beta {
            LinearOp::Dense(linear) => linear.weight().clone(),
            LinearOp::Quantized { .. } => unreachable!("tiny fixture uses dense weights"),
            #[cfg(feature = "metal")]
            LinearOp::FileBackedQuantized { .. } => {
                unreachable!("tiny fixture uses dense weights")
            }
        };
        let output = match &mixer.output {
            LinearOp::Dense(linear) => linear.weight().clone(),
            LinearOp::Quantized { .. } => unreachable!("tiny fixture uses dense weights"),
            #[cfg(feature = "metal")]
            LinearOp::FileBackedQuantized { .. } => {
                unreachable!("tiny fixture uses dense weights")
            }
        };
        for layer in 0..spec.num_layers {
            let prefix = format!("model.layers.{layer}");
            for norm in ["attn_norm", "attn_post_norm"] {
                tensors.insert(
                    format!("{prefix}.{norm}.weight"),
                    Tensor::ones(spec.hidden_size, DType::F32, device)?,
                );
            }
            for (name, tensor) in [
                ("in_proj_qkv.weight", qkv.clone()),
                ("in_proj_z.weight", output_gate.clone()),
                ("in_proj_a.weight", alpha.clone()),
                ("in_proj_b.weight", beta.clone()),
                ("conv1d.weight", mixer.convolution_kernel.clone()),
                ("dt_bias", mixer.time_bias.clone()),
                ("A_log", mixer.state_decay.clone()),
                ("norm.weight", Tensor::new(&[1.1_f32, 0.9], device)?),
                ("out_proj.weight", output.clone()),
            ] {
                tensors.insert(format!("{prefix}.linear_attn.{name}"), tensor);
            }
            for (name, shape) in [
                (
                    "gate_proj.weight",
                    (spec.intermediate_size, spec.hidden_size),
                ),
                ("up_proj.weight", (spec.intermediate_size, spec.hidden_size)),
                (
                    "down_proj.weight",
                    (spec.hidden_size, spec.intermediate_size),
                ),
            ] {
                tensors.insert(
                    format!("{prefix}.mlp.{name}"),
                    Tensor::zeros(shape, DType::F32, device)?,
                );
            }
        }
        Ok(tensors)
    }

    fn load_tiny_gated_delta_decoder(
        spec: DecoderSpec,
        tensors: HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<DecoderModel> {
        let source = DenseWeightSource::new(VarBuilder::from_tensors(tensors, DType::F32, device));
        DecoderModel::load_with_weight_names(
            spec,
            &source,
            &crate::architecture::QWEN35_WEIGHT_NAMES,
        )
    }

    fn gated_delta_state(session: &dyn CausalLmSession, layer: usize) -> &GatedDeltaNetState {
        let LayerSequenceState::GatedDeltaNet(state) =
            &decoder_session(session).layer_states[layer]
        else {
            panic!("test layer uses Gated DeltaNet")
        };
        state
    }

    fn layer_gated_delta_mut(model: &mut DecoderModel, layer: usize) -> &mut GatedDeltaNet {
        let SequenceMixer::GatedDeltaNet(delta) = &mut model.layers[layer].mixer else {
            panic!("test layer uses Gated DeltaNet")
        };
        delta
    }

    fn max_difference(left: &Tensor, right: &Tensor) -> Result<f32> {
        left.sub(right)?.abs()?.max_all()?.to_scalar::<f32>()
    }

    /// The fused prefill path must agree with the native paged path on the
    /// same weights, inputs and cache state.
    #[test]
    fn fused_prefill_matches_native_prefill_on_metal() -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Metal fused-prefill comparison: {error}");
                return Ok(());
            }
        };
        let spec = tiny_fused_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let pool = PagedKvPoolConfig::new(16, 4, 64).map_err(paged_kv_error)?;

        let mut native = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        native.configure_kv_cache(pool)?;
        native.configure_prefill_attention(PrefillAttentionMode::Native)?;
        let mut fused = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors, DType::F32, &device),
        )?;
        fused.configure_kv_cache(pool)?;
        fused.configure_prefill_attention(PrefillAttentionMode::Sdpa)?;

        // A chunk that fills whole kernel query tiles uses the in-kernel
        // causal diagonal; a shorter chunk uses the explicit additive mask.
        for prompt_len in [17_usize, 32] {
            let prompt = (0..prompt_len)
                .map(|token| (token % 29 + 1) as u32)
                .collect::<Vec<_>>();
            let mut native_session = new_session(&native)?;
            let native_logits = native.forward_tokens(native_session.as_mut(), &prompt, 0)?;
            let mut fused_session = new_session(&fused)?;
            let fused_logits = fused.forward_tokens(fused_session.as_mut(), &prompt, 0)?;
            let difference = max_difference(&native_logits, &fused_logits)?;
            assert!(
                difference < 1e-4,
                "fused and native prefill logits differed by {difference} for {prompt_len} tokens"
            );
        }
        Ok(())
    }

    /// A prefill chunk that is longer than the native paged kernel's
    /// 128-row limit must still use the fused kernel, and agree with the
    /// dense fallback of the default path.
    #[test]
    fn fused_prefill_serves_chunks_longer_than_the_native_kernel() -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Metal fused-prefill comparison: {error}");
                return Ok(());
            }
        };
        let mut spec = tiny_fused_spec();
        spec.runtime.max_context = 512;
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let pool = PagedKvPoolConfig::new(16, 4, 64).map_err(paged_kv_error)?;

        let mut native = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        native.configure_kv_cache(pool)?;
        native.configure_prefill_attention(PrefillAttentionMode::Native)?;
        let mut fused =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        fused.configure_kv_cache(pool)?;
        fused.configure_prefill_attention(PrefillAttentionMode::Sdpa)?;

        // 300 query rows exceed the native paged kernel's limit, so the
        // default path serves the chunk with the dense fallback while the
        // fused mode serves it with Candle's fused kernel.
        let prompt_len = 300_usize;
        let prompt = (0..prompt_len)
            .map(|token| (token % 29 + 1) as u32)
            .collect::<Vec<_>>();
        let mut native_session = new_session(&native)?;
        reset_full_causal_mask_construction_count();
        let native_logits = native.forward_tokens(native_session.as_mut(), &prompt, 0)?;
        let native_masks = full_causal_mask_construction_count();
        let mut fused_session = new_session(&fused)?;
        reset_full_causal_mask_construction_count();
        let fused_logits = fused.forward_tokens(fused_session.as_mut(), &prompt, 0)?;
        let fused_masks = full_causal_mask_construction_count();
        let difference = max_difference(&native_logits, &fused_logits)?;
        assert!(
            difference < 1e-4,
            "fused and native prefill logits differed by {difference} for {prompt_len} tokens"
        );
        // The dense fallback of the default path builds one causal mask for
        // the chunk, while the fused kernel builds its own mask and never asks
        // the decoder for one.
        assert_eq!(
            native_masks, 1,
            "the default path builds a causal mask for a chunk the native kernel cannot serve"
        );
        assert_eq!(
            fused_masks, 0,
            "the fused kernel must serve a chunk longer than the native kernel's limit"
        );
        Ok(())
    }

    /// The fused prefill path must not move a single-row decode lane off the
    /// attention path the default mode picks for the batch as a whole.
    ///
    /// The default path chooses one implementation per batch: the native
    /// paged kernel when the whole batch supports it, and the dense fallback
    /// otherwise. The fused mode may only replace the multi-row chunks, so
    /// every other lane keeps the batch's decision.
    #[test]
    fn fused_prefill_keeps_the_decode_lane_on_the_batch_decision() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut native = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        configure_multi_session_test_pool(&mut native)?;
        let mut fused =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        configure_multi_session_test_pool(&mut fused)?;
        fused.configure_prefill_attention(PrefillAttentionMode::Sdpa)?;

        let mut native_first = new_session(&native)?;
        let mut native_second = new_session(&native)?;
        let mut fused_first = new_session(&fused)?;
        let mut fused_second = new_session(&fused)?;
        let first_tokens = [1_u32, 2, 3];
        let second_tokens = [4_u32];

        crate::cache::reset_paged_batch_materialization_count();
        let native_logits = {
            let mut inputs = [
                CausalLmBatchInput::new(native_first.as_mut(), &first_tokens, 0),
                CausalLmBatchInput::new(native_second.as_mut(), &second_tokens, 0),
            ];
            native.forward_batch_tokens(&mut inputs)?
        };
        let native_materializations = crate::cache::paged_batch_materialization_count();

        crate::cache::reset_paged_batch_materialization_count();
        let fused_logits = {
            let mut inputs = [
                CausalLmBatchInput::new(fused_first.as_mut(), &first_tokens, 0),
                CausalLmBatchInput::new(fused_second.as_mut(), &second_tokens, 0),
            ];
            fused.forward_batch_tokens(&mut inputs)?
        };
        let fused_materializations = crate::cache::paged_batch_materialization_count();

        // The CPU backend has no fused kernel, so the mixed batch keeps the
        // dense fallback of the default path for every lane, including the
        // single-row decode lane.
        assert!(
            native_materializations > 0,
            "the default path materializes the history of every lane"
        );
        assert_eq!(
            fused_materializations, native_materializations,
            "the fused path must not move the decode lane off the batch decision"
        );
        assert!(max_difference(&native_logits, &fused_logits)? < 1e-5);
        Ok(())
    }

    /// A batch that the native paged kernel can serve as a whole must keep
    /// its single-row decode lane on the native kernel in the fused mode.
    ///
    /// The default path serves such a batch with one native batch dispatch.
    /// The fused mode replaces the multi-row chunk with Candle's fused
    /// kernel, so the decode lane falls back to the native kernel for its own
    /// lane. Both must produce the same logits, and the decode lane must not
    /// materialize the history that the native kernel reads directly.
    #[test]
    fn fused_prefill_keeps_the_native_decode_lane_on_the_native_kernel() -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Metal native-lane comparison: {error}");
                return Ok(());
            }
        };
        let mut spec = tiny_fused_spec();
        spec.runtime.max_context = 512;
        let layers = spec.layer_specs.len();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let pool = PagedKvPoolConfig::new(16, 4, 64).map_err(paged_kv_error)?;

        let mut native = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        native.configure_kv_cache(pool)?;
        native.configure_prefill_attention(PrefillAttentionMode::Native)?;
        let mut fused =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        fused.configure_kv_cache(pool)?;
        fused.configure_prefill_attention(PrefillAttentionMode::Sdpa)?;

        let mut native_chunk = new_session(&native)?;
        let mut native_decode = new_session(&native)?;
        let mut fused_chunk = new_session(&fused)?;
        let mut fused_decode = new_session(&fused)?;

        // The decode lane keeps a short prefix, so its single row is a real
        // decode step over a non-empty history.
        let decode_prefix = [5_u32, 6, 7];
        native.forward_batch_tokens(&mut [CausalLmBatchInput::new(
            native_decode.as_mut(),
            &decode_prefix,
            0,
        )])?;
        fused.forward_batch_tokens(&mut [CausalLmBatchInput::new(
            fused_decode.as_mut(),
            &decode_prefix,
            0,
        )])?;

        // A chunk of whole kernel query tiles stays eligible for the native
        // paged kernel, so the batch as a whole keeps the native decision.
        let chunk_tokens: Vec<u32> = (0_u32..32).map(|token| token % 29 + 1).collect();
        let decode_tokens = [8_u32];
        let decode_start = decode_prefix.len();

        crate::cache::reset_paged_batch_materialization_count();
        let native_logits = {
            let mut inputs = [
                CausalLmBatchInput::new(native_chunk.as_mut(), &chunk_tokens, 0),
                CausalLmBatchInput::new(native_decode.as_mut(), &decode_tokens, decode_start),
            ];
            native.forward_batch_tokens(&mut inputs)?
        };
        let native_materializations = crate::cache::paged_batch_materialization_count();

        crate::cache::reset_paged_batch_materialization_count();
        let fused_logits = {
            let mut inputs = [
                CausalLmBatchInput::new(fused_chunk.as_mut(), &chunk_tokens, 0),
                CausalLmBatchInput::new(fused_decode.as_mut(), &decode_tokens, decode_start),
            ];
            fused.forward_batch_tokens(&mut inputs)?
        };
        let fused_materializations = crate::cache::paged_batch_materialization_count();

        // The default path serves the whole batch with one native dispatch,
        // which reads the paged cache without materializing the history.
        assert_eq!(
            native_materializations, 0,
            "the native batch dispatch must not materialize the history"
        );
        // The fused mode materializes the history of the chunk it serves with
        // the fused kernel, once per layer, and keeps the decode lane on the
        // native kernel, which never materializes the history.
        assert_eq!(
            fused_materializations, layers,
            "the fused path must serve the chunk and keep the decode lane native"
        );
        assert!(
            max_difference(&native_logits, &fused_logits)? < 1e-4,
            "the fused and default batches must agree"
        );
        Ok(())
    }

    /// A Metal batch that mixes a long prefill chunk with a single-row decode
    /// lane must keep the decode lane on the attention path the default mode
    /// picks for the batch as a whole.
    ///
    /// The long chunk exceeds the native paged kernel's 128-row limit, so the
    /// default path cannot serve the batch with the native kernel and falls
    /// back to the dense path for every lane. The fused mode replaces the long
    /// chunk with Candle's fused kernel, but the decode lane must keep the
    /// dense fallback of the batch decision instead of moving to the native
    /// paged kernel.
    #[test]
    fn fused_prefill_keeps_the_decode_lane_on_the_batch_decision_on_metal() -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Metal mixed-batch comparison: {error}");
                return Ok(());
            }
        };
        let mut spec = tiny_fused_spec();
        spec.runtime.max_context = 512;
        let layers = spec.layer_specs.len();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let pool = PagedKvPoolConfig::new(16, 4, 64).map_err(paged_kv_error)?;

        let mut native = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        native.configure_kv_cache(pool)?;
        native.configure_prefill_attention(PrefillAttentionMode::Native)?;
        let mut fused =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        fused.configure_kv_cache(pool)?;
        fused.configure_prefill_attention(PrefillAttentionMode::Sdpa)?;

        let mut native_long = new_session(&native)?;
        let mut native_decode = new_session(&native)?;
        let mut fused_long = new_session(&fused)?;
        let mut fused_decode = new_session(&fused)?;

        // The decode lane keeps a short prefix, so its single row is a real
        // decode step over a non-empty history.
        let decode_prefix = [5_u32, 6, 7];
        native.forward_batch_tokens(&mut [CausalLmBatchInput::new(
            native_decode.as_mut(),
            &decode_prefix,
            0,
        )])?;
        fused.forward_batch_tokens(&mut [CausalLmBatchInput::new(
            fused_decode.as_mut(),
            &decode_prefix,
            0,
        )])?;

        // The long lane exceeds the native paged kernel's row limit, so the
        // batch as a whole cannot use the native kernel.
        let long_tokens: Vec<u32> = (0_u32..300).map(|token| token % 29 + 1).collect();
        let decode_tokens = [8_u32];
        let decode_start = decode_prefix.len();

        crate::cache::reset_paged_batch_materialization_count();
        reset_full_causal_mask_construction_count();
        let native_logits = {
            let mut inputs = [
                CausalLmBatchInput::new(native_long.as_mut(), &long_tokens, 0),
                CausalLmBatchInput::new(native_decode.as_mut(), &decode_tokens, decode_start),
            ];
            native.forward_batch_tokens(&mut inputs)?
        };
        let native_materializations = crate::cache::paged_batch_materialization_count();
        let native_masks = full_causal_mask_construction_count();

        crate::cache::reset_paged_batch_materialization_count();
        reset_full_causal_mask_construction_count();
        let fused_logits = {
            let mut inputs = [
                CausalLmBatchInput::new(fused_long.as_mut(), &long_tokens, 0),
                CausalLmBatchInput::new(fused_decode.as_mut(), &decode_tokens, decode_start),
            ];
            fused.forward_batch_tokens(&mut inputs)?
        };
        let fused_materializations = crate::cache::paged_batch_materialization_count();
        let fused_masks = full_causal_mask_construction_count();

        // The default path cannot serve the batch with the native paged
        // kernel, so it materializes the history of both lanes in every layer.
        assert_eq!(
            native_materializations,
            2 * layers,
            "the default path materializes every lane of a batch the native kernel cannot serve"
        );
        // The fused mode replaces the long chunk with the fused kernel, which
        // materializes the same history once, and the decode lane keeps the
        // dense fallback of the batch decision.
        assert_eq!(
            fused_materializations, native_materializations,
            "the fused path must not move the decode lane off the batch decision"
        );
        // The long chunk is served by Candle's fused kernel, which builds its
        // own mask, while the default path asks the decoder for a causal mask.
        assert_eq!(
            native_masks, 1,
            "the default path builds a causal mask for the long chunk"
        );
        assert_eq!(
            fused_masks, 0,
            "the fused kernel must serve the long chunk of the mixed batch"
        );
        assert!(
            max_difference(&native_logits, &fused_logits)? < 1e-4,
            "the fused and default batches must agree"
        );
        Ok(())
    }

    #[test]
    fn short_convolution_matches_hand_calculation_and_chunked_state() -> Result<()> {
        let device = Device::Cpu;
        let input_weight = Tensor::from_vec(
            vec![
                1.0_f32, 0.0, 0.0, 1.0, // B
                1.0, 0.0, 0.0, 1.0, // C
                1.0, 0.0, 0.0, 1.0, // x
            ],
            (6, 2),
            &device,
        )?;
        let mixer = ShortConvolution {
            input: LinearOp::Dense(candle_nn::Linear::new(input_weight, None)),
            kernel: Tensor::from_vec(vec![1.0_f32, 10.0, 100.0, -1.0, 0.5, 2.0], (2, 3), &device)?,
            output: LinearOp::Dense(candle_nn::Linear::new(
                Tensor::eye(2, DType::F32, &device)?,
                None,
            )),
            hidden_size: 2,
            kernel_size: 3,
        };
        let input = Tensor::new(&[[[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]]], &device)?;

        let mut one_shot_state = ShortConvolutionState::new(3);
        let one_shot = mixer.forward_lane(&mut one_shot_state, &input)?;
        assert_eq!(
            one_shot.to_vec3::<f32>()?,
            vec![vec![
                vec![100.0, 16.0],
                vec![2730.0, 136.0],
                vec![12_955.0, 456.0]
            ]]
        );
        assert_eq!(one_shot_state.logical_len, 3);
        let tail = one_shot_state.tail.as_ref().expect("shortconv tail");
        let (tail_storage, _) = tail.storage_and_layout();
        match &*tail_storage {
            candle_core::Storage::Cpu(candle_core::CpuStorage::F32(values)) => {
                assert_eq!(
                    values.len(),
                    4,
                    "tail must not retain the full prefill window"
                );
            }
            storage => panic!("unexpected shortconv tail storage {storage:?}"),
        }
        assert_eq!(
            one_shot_state
                .tail
                .as_ref()
                .expect("K=3 keeps two history tokens")
                .to_vec3::<f32>()?,
            vec![vec![vec![9.0, 16.0], vec![25.0, 36.0]]]
        );

        let mut chunked_state = ShortConvolutionState::new(3);
        let first = mixer.forward_lane(&mut chunked_state, &input.narrow(1, 0, 1)?)?;
        let second = mixer.forward_lane(&mut chunked_state, &input.narrow(1, 1, 2)?)?;
        let chunked = Tensor::cat(&[&first, &second], 1)?;
        assert!(max_difference(&one_shot, &chunked)? < f32::EPSILON);
        assert_eq!(chunked_state.logical_len, 3);
        assert!(
            max_difference(
                one_shot_state.tail.as_ref().expect("one-shot tail"),
                chunked_state.tail.as_ref().expect("chunked tail"),
            )? < f32::EPSILON
        );
        Ok(())
    }

    #[test]
    fn short_convolution_normalizes_the_official_hf_kernel_shape() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_hybrid_spec();
        let kernel = patterned_tensor(&[spec.hidden_size, 3], 91, 0.03, &device)?;
        let tensors = HashMap::from([
            (
                "conv.in_proj.weight".into(),
                Tensor::zeros(
                    (spec.hidden_size * 3, spec.hidden_size),
                    DType::F32,
                    &device,
                )?,
            ),
            ("conv.conv.weight".into(), kernel.unsqueeze(1)?),
            (
                "conv.out_proj.weight".into(),
                Tensor::eye(spec.hidden_size, DType::F32, &device)?,
            ),
        ]);
        let source = DenseWeightSource::new(VarBuilder::from_tensors(tensors, DType::F32, &device));
        let mixer = ShortConvolution::load(
            &spec,
            ShortConvolutionSpec { kernel_size: 3 },
            WeightPath::root(&source).pp("conv"),
        )?;
        assert_eq!(mixer.kernel.dims(), &[spec.hidden_size, 3]);
        assert!(max_difference(&mixer.kernel, &kernel)? < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn hybrid_shortconv_prefill_chunking_decode_and_clear_are_equivalent() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_hybrid_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let tokens = [1_u32, 2, 3, 4, 5, 6, 7];

        let one_shot = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut one_shot_session = new_session(&one_shot)?;
        let expected = one_shot.forward_tokens(one_shot_session.as_mut(), &tokens, 0)?;
        assert_eq!(one_shot_session.cache_len(), tokens.len());
        assert_eq!(
            shortconv_state(one_shot_session.as_ref(), 0).logical_len,
            tokens.len()
        );
        // KV diagnostics describe only the one attention layer, not the
        // recurrent layer that deliberately owns no K/V pages.
        assert_eq!(cache_stats(one_shot_session.as_ref()).len(), 1);

        let chunked = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut chunked_session = new_session(&chunked)?;
        let mut offset = 0;
        let mut chunked_logits = None;
        for length in [1, 2, 4] {
            chunked_logits = Some(chunked.forward_tokens(
                chunked_session.as_mut(),
                &tokens[offset..offset + length],
                offset,
            )?);
            offset += length;
        }
        assert!(max_difference(&expected, &chunked_logits.expect("last chunk logits"))? < 1e-5);

        let tokenwise = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut tokenwise_session = new_session(&tokenwise)?;
        let mut tokenwise_logits = None;
        for (position, token) in tokens.iter().copied().enumerate() {
            tokenwise_logits =
                Some(tokenwise.forward_tokens(tokenwise_session.as_mut(), &[token], position)?);
        }
        assert!(
            max_difference(&expected, &tokenwise_logits.expect("last tokenwise logits"),)? < 1e-5
        );

        chunked_session.clear_cache();
        assert_eq!(chunked_session.cache_len(), 0);
        assert_eq!(shortconv_state(chunked_session.as_ref(), 0).logical_len, 0);
        let reused = chunked.forward_tokens(chunked_session.as_mut(), &tokens, 0)?;
        assert!(max_difference(&expected, &reused)? < 1e-5);
        Ok(())
    }

    #[test]
    fn packed_ragged_shortconv_lanes_match_serial_without_state_bleed() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_hybrid_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        configure_multi_session_test_pool(&mut model)?;
        let mut first = new_session(&model)?;
        let mut second = new_session(&model)?;
        let mut first_serial = new_session(&model)?;
        let mut second_serial = new_session(&model)?;

        let first_prefix = [1_u32, 2];
        let second_prefix = [11_u32, 12, 13];
        let prefix_batch = {
            let mut inputs = [
                CausalLmBatchInput::new(first.as_mut(), &first_prefix, 0),
                CausalLmBatchInput::new(second.as_mut(), &second_prefix, 0),
            ];
            model.forward_batch_tokens(&mut inputs)?
        };
        let first_prefix_serial = model.forward_tokens(first_serial.as_mut(), &first_prefix, 0)?;
        let second_prefix_serial =
            model.forward_tokens(second_serial.as_mut(), &second_prefix, 0)?;
        assert!(max_difference(&prefix_batch.narrow(0, 0, 1)?, &first_prefix_serial)? < 1e-5);
        assert!(max_difference(&prefix_batch.narrow(0, 1, 1)?, &second_prefix_serial)? < 1e-5);

        let first_suffix = [3_u32, 4, 5];
        let second_suffix = [14_u32];
        let suffix_batch = {
            let mut inputs = [
                CausalLmBatchInput::new(first.as_mut(), &first_suffix, first_prefix.len()),
                CausalLmBatchInput::new(second.as_mut(), &second_suffix, second_prefix.len()),
            ];
            model.forward_batch_tokens(&mut inputs)?
        };
        let first_expected =
            model.forward_tokens(first_serial.as_mut(), &first_suffix, first_prefix.len())?;
        let second_expected =
            model.forward_tokens(second_serial.as_mut(), &second_suffix, second_prefix.len())?;
        assert!(max_difference(&suffix_batch.narrow(0, 0, 1)?, &first_expected)? < 1e-5);
        assert!(max_difference(&suffix_batch.narrow(0, 1, 1)?, &second_expected)? < 1e-5);
        assert_eq!(first.cache_len(), first_prefix.len() + first_suffix.len());
        assert_eq!(
            second.cache_len(),
            second_prefix.len() + second_suffix.len()
        );
        assert!(
            max_difference(
                shortconv_state(first.as_ref(), 0)
                    .tail
                    .as_ref()
                    .expect("first tail"),
                shortconv_state(first_serial.as_ref(), 0)
                    .tail
                    .as_ref()
                    .expect("first serial tail"),
            )? < 1e-6
        );
        assert!(
            max_difference(
                shortconv_state(second.as_ref(), 0)
                    .tail
                    .as_ref()
                    .expect("second tail"),
                shortconv_state(second_serial.as_ref(), 0)
                    .tail
                    .as_ref()
                    .expect("second serial tail"),
            )? < 1e-6
        );
        Ok(())
    }

    #[test]
    fn attention_failure_rolls_shortconv_state_back_before_retry() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_hybrid_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let prefix = [1_u32, 2, 3];
        let suffix = [4_u32, 5];

        let control = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut control_session = new_session(&control)?;
        let _ = control.forward_tokens(control_session.as_mut(), &prefix, 0)?;
        let expected = control.forward_tokens(control_session.as_mut(), &suffix, prefix.len())?;

        let mut model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut session = new_session(&model)?;
        let _ = model.forward_tokens(session.as_mut(), &prefix, 0)?;
        let checkpoint_tail = shortconv_state(session.as_ref(), 0)
            .tail
            .as_ref()
            .expect("prefix creates tail")
            .clone();
        let output = layer_attention(&model, 1).output.clone();
        layer_attention_mut(&mut model, 1).output = LinearOp::Dense(candle_nn::Linear::new(
            Tensor::zeros((1, 1), DType::F32, &device)?,
            None,
        ));
        assert!(
            model
                .forward_tokens(session.as_mut(), &suffix, prefix.len())
                .is_err()
        );
        assert_eq!(session.cache_len(), prefix.len());
        assert_eq!(
            shortconv_state(session.as_ref(), 0).logical_len,
            prefix.len()
        );
        assert!(
            max_difference(
                shortconv_state(session.as_ref(), 0)
                    .tail
                    .as_ref()
                    .expect("rollback restores tail"),
                &checkpoint_tail,
            )? < f32::EPSILON
        );

        layer_attention_mut(&mut model, 1).output = output;
        let actual = model.forward_tokens(session.as_mut(), &suffix, prefix.len())?;
        assert!(max_difference(&actual, &expected)? < 1e-5);
        assert_eq!(
            shortconv_state(session.as_ref(), 0).logical_len,
            prefix.len() + suffix.len()
        );
        Ok(())
    }

    #[test]
    #[allow(clippy::too_many_lines)] // Explicit golden state keeps recurrence order auditable.
    fn gated_delta_matches_scalar_golden_across_prefill_chunking_and_decode() -> Result<()> {
        let device = Device::Cpu;
        let mixer = tiny_gated_delta_mixer(&device)?;
        let input = tiny_gated_delta_input(&device)?;

        let mut one_shot_state = GatedDeltaNetState::new(tiny_gated_delta_spec())?;
        let one_shot = mixer.forward_lane(
            &mut one_shot_state,
            &mixer.qkv.forward(&input)?,
            &mixer.output_gate.forward(&input)?,
            &mixer.alpha.forward(&input)?,
            &mixer.beta.forward(&input)?,
        )?;
        // Independent scalar delta-rule calculation for this fixture. Keeping
        // the numbers explicit catches recurrence order, convolution
        // orientation, normalization, output gating, and projection changes.
        let golden = Tensor::new(
            &[[
                [0.016_996_983_f32, 0.183_911_86],
                [0.500_141_9, 0.046_489_283],
                [-0.109_404_06, 0.389_642_54],
                [0.232_875_17, 0.080_105_25],
            ]],
            &device,
        )?;
        assert!(max_difference(&one_shot, &golden)? < 2e-5);
        assert_eq!(one_shot_state.logical_len, 4);
        let expected_tail = Tensor::new(
            &[[
                [-0.2_f32, 0.6, -0.25, 0.4, -0.1, 0.4],
                [0.4, -0.3, 0.275, -0.125, 0.29, -0.26],
            ]],
            &device,
        )?;
        assert!(
            max_difference(
                one_shot_state
                    .convolution_tail
                    .as_ref()
                    .expect("K=3 retains two projected tokens"),
                &expected_tail,
            )? < 2e-6
        );
        let expected_recurrent = Tensor::new(
            &[[
                [0.127_428_08_f32, -0.041_561_37],
                [0.013_819_901, 0.081_209_71],
            ]],
            &device,
        )?;
        assert!(
            max_difference(
                one_shot_state
                    .recurrent
                    .as_ref()
                    .expect("prefill creates recurrent matrix state"),
                &expected_recurrent,
            )? < 2e-5
        );

        let mut chunked_state = GatedDeltaNetState::new(tiny_gated_delta_spec())?;
        let mut chunked_outputs = Vec::new();
        for (offset, length) in [(0, 1), (1, 3)] {
            let chunk = input.narrow(1, offset, length)?;
            chunked_outputs.push(mixer.forward_lane(
                &mut chunked_state,
                &mixer.qkv.forward(&chunk)?,
                &mixer.output_gate.forward(&chunk)?,
                &mixer.alpha.forward(&chunk)?,
                &mixer.beta.forward(&chunk)?,
            )?);
        }
        let chunked_outputs = chunked_outputs.iter().collect::<Vec<_>>();
        let chunked = Tensor::cat(&chunked_outputs, 1)?;
        assert!(max_difference(&one_shot, &chunked)? < 2e-6);

        let mut token_state = GatedDeltaNetState::new(tiny_gated_delta_spec())?;
        let mut token_outputs = Vec::new();
        for token in 0..input.dim(1)? {
            let chunk = input.narrow(1, token, 1)?;
            token_outputs.push(mixer.forward_lane(
                &mut token_state,
                &mixer.qkv.forward(&chunk)?,
                &mixer.output_gate.forward(&chunk)?,
                &mixer.alpha.forward(&chunk)?,
                &mixer.beta.forward(&chunk)?,
            )?);
        }
        let token_outputs = token_outputs.iter().collect::<Vec<_>>();
        let tokenwise = Tensor::cat(&token_outputs, 1)?;
        assert!(max_difference(&one_shot, &tokenwise)? < 2e-6);
        assert_eq!(chunked_state.logical_len, one_shot_state.logical_len);
        assert_eq!(token_state.logical_len, one_shot_state.logical_len);
        for state in [&chunked_state, &token_state] {
            assert!(
                max_difference(
                    state
                        .convolution_tail
                        .as_ref()
                        .expect("streaming state retains convolution tail"),
                    one_shot_state
                        .convolution_tail
                        .as_ref()
                        .expect("one-shot convolution tail"),
                )? < 2e-6
            );
            assert!(
                max_difference(
                    state
                        .recurrent
                        .as_ref()
                        .expect("streaming state retains recurrence"),
                    one_shot_state
                        .recurrent
                        .as_ref()
                        .expect("one-shot recurrence"),
                )? < 2e-6
            );
        }
        Ok(())
    }

    #[test]
    fn packed_ragged_gated_delta_lanes_match_serial_without_state_bleed() -> Result<()> {
        let device = Device::Cpu;
        let mixer = tiny_gated_delta_mixer(&device)?;
        let first_input = tiny_gated_delta_input(&device)?;
        let second_input = Tensor::new(&[[[-0.6_f32, 0.2], [0.1, 0.9], [0.5, -0.4]]], &device)?;
        let mut packed_first = GatedDeltaNetState::new(tiny_gated_delta_spec())?;
        let mut packed_second = GatedDeltaNetState::new(tiny_gated_delta_spec())?;
        let mut serial_first = GatedDeltaNetState::new(tiny_gated_delta_spec())?;
        let mut serial_second = GatedDeltaNetState::new(tiny_gated_delta_spec())?;

        for (first_offset, first_length, second_offset, second_length) in
            [(0, 1, 0, 2), (1, 3, 2, 1)]
        {
            let first = first_input.narrow(1, first_offset, first_length)?;
            let second = second_input.narrow(1, second_offset, second_length)?;
            let packed_input = Tensor::cat(&[&first, &second], 1)?;
            let packed = mixer.forward_packed(
                &mut [&mut packed_first, &mut packed_second],
                &packed_input,
                &[first_length, second_length],
            )?;
            let first_serial = mixer.forward_lane(
                &mut serial_first,
                &mixer.qkv.forward(&first)?,
                &mixer.output_gate.forward(&first)?,
                &mixer.alpha.forward(&first)?,
                &mixer.beta.forward(&first)?,
            )?;
            let second_serial = mixer.forward_lane(
                &mut serial_second,
                &mixer.qkv.forward(&second)?,
                &mixer.output_gate.forward(&second)?,
                &mixer.alpha.forward(&second)?,
                &mixer.beta.forward(&second)?,
            )?;
            let serial = Tensor::cat(&[&first_serial, &second_serial], 1)?;
            assert!(max_difference(&packed, &serial)? < 2e-6);
        }

        assert_eq!(packed_first.logical_len, 4);
        assert_eq!(packed_second.logical_len, 3);
        for (packed, serial) in [
            (&packed_first, &serial_first),
            (&packed_second, &serial_second),
        ] {
            assert_eq!(packed.logical_len, serial.logical_len);
            assert!(
                max_difference(
                    packed
                        .convolution_tail
                        .as_ref()
                        .expect("packed lane convolution tail"),
                    serial
                        .convolution_tail
                        .as_ref()
                        .expect("serial lane convolution tail"),
                )? < 2e-6
            );
            assert!(
                max_difference(
                    packed.recurrent.as_ref().expect("packed lane recurrence"),
                    serial.recurrent.as_ref().expect("serial lane recurrence"),
                )? < 2e-6
            );
        }
        Ok(())
    }

    #[test]
    #[allow(clippy::too_many_lines)] // Explicit golden state keeps recurrence order auditable.
    fn mamba2_matches_scalar_golden_across_prefill_chunking_and_decode() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_mamba2_spec();
        let mixer = tiny_mamba2_mixer(&device)?;
        let projected = tiny_mamba2_projected_input(&device)?;

        let mut one_shot_state = Mamba2State::new(spec)?;
        assert_eq!(one_shot_state.convolution_history_tokens, 1);
        assert_eq!(one_shot_state.convolution_width, 3);
        assert_eq!(one_shot_state.num_heads, 1);
        assert_eq!(one_shot_state.head_dim, 1);
        assert_eq!(one_shot_state.state_size, 1);
        assert_eq!(one_shot_state.logical_len, 0);
        let one_shot = mixer.forward_lane(&mut one_shot_state, &projected)?;

        // Independent scalar recurrence for this fixture:
        //   conv_t = old * kernel[0] + current * kernel[1] + bias
        //   delta_t = softplus(dt_t + 0.15)
        //   S_t = exp(-0.7 * delta_t) * S_(t-1) + delta_t * x_t * B_t
        //   y_t = 1.1 * (S_t * C_t + 0.3 * x_t) * silu(z_t)
        // where x/B/C are each silu(conv_t). Keeping the resulting numbers
        // explicit catches projection order and convolution orientation as
        // well as discretization, recurrence order, skip, gate and output.
        let golden = Tensor::new(
            &[[[0.089_636_93_f32], [0.005_108_623], [0.024_774_795]]],
            &device,
        )?;
        assert!(max_difference(&one_shot, &golden)? < 2e-6);
        assert_eq!(one_shot_state.logical_len, 3);
        let expected_tail = Tensor::new(&[[[0.25_f32, -0.6, 0.9]]], &device)?;
        assert!(
            max_difference(
                one_shot_state
                    .convolution_tail
                    .as_ref()
                    .expect("K=2 retains the final raw x/B/C projection"),
                &expected_tail,
            )? < 2e-6
        );
        let expected_recurrent = Tensor::new(&[[[-0.013_268_862_f32]]], &device)?;
        assert!(
            max_difference(
                one_shot_state
                    .recurrent
                    .as_ref()
                    .expect("prefill creates recurrent state"),
                &expected_recurrent,
            )? < 2e-6
        );

        let mut chunked_state = Mamba2State::new(spec)?;
        let mut chunked_outputs = Vec::new();
        for (offset, length) in [(0, 1), (1, 2)] {
            chunked_outputs.push(
                mixer.forward_lane(&mut chunked_state, &projected.narrow(1, offset, length)?)?,
            );
        }
        let chunked_outputs = chunked_outputs.iter().collect::<Vec<_>>();
        let chunked = Tensor::cat(&chunked_outputs, 1)?;
        assert!(max_difference(&one_shot, &chunked)? < 2e-6);

        let mut token_state = Mamba2State::new(spec)?;
        let mut token_outputs = Vec::new();
        for token in 0..projected.dim(1)? {
            token_outputs
                .push(mixer.forward_lane(&mut token_state, &projected.narrow(1, token, 1)?)?);
        }
        let token_outputs = token_outputs.iter().collect::<Vec<_>>();
        let tokenwise = Tensor::cat(&token_outputs, 1)?;
        assert!(max_difference(&one_shot, &tokenwise)? < 2e-6);

        for state in [&chunked_state, &token_state] {
            assert_eq!(state.logical_len, one_shot_state.logical_len);
            assert!(
                max_difference(
                    state
                        .convolution_tail
                        .as_ref()
                        .expect("streaming state retains convolution tail"),
                    one_shot_state
                        .convolution_tail
                        .as_ref()
                        .expect("one-shot convolution tail"),
                )? < 2e-6
            );
            assert!(
                max_difference(
                    state
                        .recurrent
                        .as_ref()
                        .expect("streaming state retains recurrence"),
                    one_shot_state
                        .recurrent
                        .as_ref()
                        .expect("one-shot recurrence"),
                )? < 2e-6
            );
        }

        let mut restored_state = Mamba2State::new(spec)?;
        let _ = mixer.forward_lane(&mut restored_state, &projected.narrow(1, 0, 1)?)?;
        let checkpoint = restored_state.checkpoint();
        let expected_suffix =
            mixer.forward_lane(&mut restored_state, &projected.narrow(1, 1, 2)?)?;
        restored_state.restore(checkpoint);
        assert_eq!(restored_state.logical_len, 1);
        let restored_suffix =
            mixer.forward_lane(&mut restored_state, &projected.narrow(1, 1, 2)?)?;
        assert!(max_difference(&expected_suffix, &restored_suffix)? < f32::EPSILON);
        restored_state.clear();
        assert_eq!(restored_state.logical_len, 0);
        assert!(restored_state.convolution_tail.is_none());
        assert!(restored_state.recurrent.is_none());
        // Clear resets sequence data without changing the state's fixed
        // geometry, so the allocation contract remains reusable.
        assert_eq!(restored_state.convolution_history_tokens, 1);
        assert_eq!(restored_state.convolution_width, 3);
        assert_eq!(restored_state.num_heads, 1);
        assert_eq!(restored_state.head_dim, 1);
        assert_eq!(restored_state.state_size, 1);
        Ok(())
    }

    #[test]
    fn packed_ragged_mamba2_lanes_match_serial_without_state_bleed() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_mamba2_spec();
        let mixer = tiny_mamba2_mixer(&device)?;
        let first_input = Tensor::new(&[[[1.0_f32], [-0.5], [0.25], [0.8]]], &device)?;
        let second_input = Tensor::new(&[[[-0.7_f32], [0.4], [1.2]]], &device)?;
        let mut packed_first = Mamba2State::new(spec)?;
        let mut packed_second = Mamba2State::new(spec)?;
        let mut serial_first = Mamba2State::new(spec)?;
        let mut serial_second = Mamba2State::new(spec)?;

        for (first_offset, first_length, second_offset, second_length) in
            [(0, 1, 0, 2), (1, 3, 2, 1)]
        {
            let first = first_input.narrow(1, first_offset, first_length)?;
            let second = second_input.narrow(1, second_offset, second_length)?;
            let packed_input = Tensor::cat(&[&first, &second], 1)?;
            let packed = mixer.forward_packed(
                &mut [&mut packed_first, &mut packed_second],
                &packed_input,
                &[first_length, second_length],
            )?;
            let first_serial =
                mixer.forward_lane(&mut serial_first, &mixer.input.forward(&first)?)?;
            let second_serial =
                mixer.forward_lane(&mut serial_second, &mixer.input.forward(&second)?)?;
            let serial = Tensor::cat(&[&first_serial, &second_serial], 1)?;
            assert!(max_difference(&packed, &serial)? < 2e-6);
        }

        assert_eq!(packed_first.logical_len, 4);
        assert_eq!(packed_second.logical_len, 3);
        for (packed, serial) in [
            (&packed_first, &serial_first),
            (&packed_second, &serial_second),
        ] {
            assert_eq!(packed.logical_len, serial.logical_len);
            assert!(
                max_difference(
                    packed
                        .convolution_tail
                        .as_ref()
                        .expect("packed lane convolution tail"),
                    serial
                        .convolution_tail
                        .as_ref()
                        .expect("serial lane convolution tail"),
                )? < 2e-6
            );
            assert!(
                max_difference(
                    packed.recurrent.as_ref().expect("packed lane recurrence"),
                    serial.recurrent.as_ref().expect("serial lane recurrence"),
                )? < 2e-6
            );
        }
        Ok(())
    }

    #[test]
    #[allow(clippy::too_many_lines)] // Every transactional state component is asserted explicitly.
    fn failed_parallel_attention_mamba2_forward_rolls_kv_and_recurrence_back() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_parallel_mamba2_decoder_spec();
        let tensors = tiny_parallel_mamba2_decoder_tensors(&spec, &device)?;
        let prefix = [1_u32, 2, 3];
        let suffix = [4_u32, 5];

        // This independent model is both the success reference and proof that
        // the suffix changes every state component exercised by the failure.
        let reference = load_tiny_parallel_mamba2_decoder(spec.clone(), tensors.clone(), &device)?;
        let mut reference_session = new_session(&reference)?;
        let _ = reference.forward_tokens(reference_session.as_mut(), &prefix, 0)?;

        let mut model = load_tiny_parallel_mamba2_decoder(spec, tensors, &device)?;
        let mut session = new_session(&model)?;
        let _ = model.forward_tokens(session.as_mut(), &prefix, 0)?;
        let checkpoint_stats = cache_stats(session.as_ref());
        let checkpoint_table_len = decoder_session(session.as_ref())
            .full_table
            .as_ref()
            .expect("parallel attention owns a paged block table")
            .logical_len();
        let checkpoints = (0..model.layers.len())
            .map(|layer| {
                let state = mamba2_state(session.as_ref(), layer);
                (
                    state
                        .convolution_tail
                        .as_ref()
                        .expect("prefix creates a Mamba2 convolution tail")
                        .clone(),
                    state
                        .recurrent
                        .as_ref()
                        .expect("prefix creates a Mamba2 recurrence")
                        .clone(),
                    state.logical_len,
                )
            })
            .collect::<Vec<_>>();

        let expected =
            reference.forward_tokens(reference_session.as_mut(), &suffix, prefix.len())?;
        assert!(
            cache_stats(reference_session.as_ref())
                .iter()
                .all(|stats| stats.logical_len == prefix.len() + suffix.len())
        );
        for (layer, (checkpoint_tail, checkpoint_recurrent, _)) in checkpoints.iter().enumerate() {
            let advanced = mamba2_state(reference_session.as_ref(), layer);
            assert!(
                max_difference(
                    advanced
                        .convolution_tail
                        .as_ref()
                        .expect("reference suffix advances convolution state"),
                    checkpoint_tail,
                )? > 1e-8,
                "layer {layer} fixture must change its convolution tail"
            );
            assert!(
                max_difference(
                    advanced
                        .recurrent
                        .as_ref()
                        .expect("reference suffix advances recurrent state"),
                    checkpoint_recurrent,
                )? > 1e-8,
                "layer {layer} fixture must change its recurrence"
            );
        }

        // The invalid second-layer FFN projection is downstream of that
        // layer's attention write and Mamba2 update. The outer transaction
        // must discard the staged K/V and restore both recurrent handles for
        // every layer before returning this shape error.
        let down = layer_dense_feed_forward_mut(&mut model, 1).down.clone();
        layer_dense_feed_forward_mut(&mut model, 1).down = LinearOp::Dense(candle_nn::Linear::new(
            Tensor::zeros((1, 1), DType::F32, &device)?,
            None,
        ));
        assert!(
            model
                .forward_tokens(session.as_mut(), &suffix, prefix.len())
                .is_err()
        );
        assert_eq!(session.cache_len(), prefix.len());
        assert_eq!(cache_stats(session.as_ref()), checkpoint_stats);
        assert_eq!(
            decoder_session(session.as_ref())
                .full_table
                .as_ref()
                .expect("parallel attention retains its paged block table")
                .logical_len(),
            checkpoint_table_len
        );
        for (layer, (checkpoint_tail, checkpoint_recurrent, checkpoint_len)) in
            checkpoints.iter().enumerate()
        {
            let restored = mamba2_state(session.as_ref(), layer);
            assert_eq!(restored.logical_len, *checkpoint_len);
            assert!(
                max_difference(
                    restored
                        .convolution_tail
                        .as_ref()
                        .expect("rollback restores Mamba2 convolution tail"),
                    checkpoint_tail,
                )? < f32::EPSILON
            );
            assert!(
                max_difference(
                    restored
                        .recurrent
                        .as_ref()
                        .expect("rollback restores Mamba2 recurrence"),
                    checkpoint_recurrent,
                )? < f32::EPSILON
            );
        }

        layer_dense_feed_forward_mut(&mut model, 1).down = down;
        let retried = model.forward_tokens(session.as_mut(), &suffix, prefix.len())?;
        assert!(max_difference(&retried, &expected)? < 1e-5);
        assert_eq!(
            cache_stats(session.as_ref()),
            cache_stats(reference_session.as_ref())
        );
        for layer in 0..model.layers.len() {
            let retried = mamba2_state(session.as_ref(), layer);
            let expected = mamba2_state(reference_session.as_ref(), layer);
            assert_eq!(retried.logical_len, expected.logical_len);
            assert!(
                max_difference(
                    retried
                        .convolution_tail
                        .as_ref()
                        .expect("retry creates Mamba2 convolution tail"),
                    expected
                        .convolution_tail
                        .as_ref()
                        .expect("reference has Mamba2 convolution tail"),
                )? < 2e-6
            );
            assert!(
                max_difference(
                    retried
                        .recurrent
                        .as_ref()
                        .expect("retry creates Mamba2 recurrence"),
                    expected
                        .recurrent
                        .as_ref()
                        .expect("reference has Mamba2 recurrence"),
                )? < 2e-6
            );
        }
        Ok(())
    }

    #[test]
    fn later_layer_failure_rolls_gated_delta_state_back_before_retry() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_gated_delta_decoder_spec();
        let tensors = tiny_gated_delta_decoder_tensors(&spec, &device)?;
        let prefix = [1_u32, 2, 3];
        let suffix = [4_u32, 5];

        let control = load_tiny_gated_delta_decoder(spec.clone(), tensors.clone(), &device)?;
        let mut control_session = new_session(&control)?;
        let _ = control.forward_tokens(control_session.as_mut(), &prefix, 0)?;
        let expected = control.forward_tokens(control_session.as_mut(), &suffix, prefix.len())?;

        let mut model = load_tiny_gated_delta_decoder(spec, tensors, &device)?;
        let mut session = new_session(&model)?;
        let _ = model.forward_tokens(session.as_mut(), &prefix, 0)?;
        let checkpoint_tail = gated_delta_state(session.as_ref(), 0)
            .convolution_tail
            .as_ref()
            .expect("prefix creates convolution tail")
            .clone();
        let checkpoint_recurrent = gated_delta_state(session.as_ref(), 0)
            .recurrent
            .as_ref()
            .expect("prefix creates recurrence")
            .clone();
        let checkpoint_logical_len = gated_delta_state(session.as_ref(), 0).logical_len;

        let later_output = layer_gated_delta_mut(&mut model, 1).output.clone();
        layer_gated_delta_mut(&mut model, 1).output = LinearOp::Dense(candle_nn::Linear::new(
            Tensor::zeros((1, 1), DType::F32, &device)?,
            None,
        ));
        assert!(
            model
                .forward_tokens(session.as_mut(), &suffix, prefix.len())
                .is_err()
        );
        assert_eq!(session.cache_len(), prefix.len());
        let restored = gated_delta_state(session.as_ref(), 0);
        assert_eq!(restored.logical_len, checkpoint_logical_len);
        assert!(
            max_difference(
                restored
                    .convolution_tail
                    .as_ref()
                    .expect("rollback restores convolution tail"),
                &checkpoint_tail,
            )? < f32::EPSILON
        );
        assert!(
            max_difference(
                restored
                    .recurrent
                    .as_ref()
                    .expect("rollback restores recurrent matrix"),
                &checkpoint_recurrent,
            )? < f32::EPSILON
        );

        layer_gated_delta_mut(&mut model, 1).output = later_output;
        let actual = model.forward_tokens(session.as_mut(), &suffix, prefix.len())?;
        assert!(max_difference(&actual, &expected)? < 2e-6);
        assert_eq!(session.cache_len(), prefix.len() + suffix.len());
        for layer in 0..2 {
            let retried = gated_delta_state(session.as_ref(), layer);
            let fresh = gated_delta_state(control_session.as_ref(), layer);
            assert_eq!(retried.logical_len, fresh.logical_len);
            assert!(
                max_difference(
                    retried
                        .convolution_tail
                        .as_ref()
                        .expect("retried convolution tail"),
                    fresh
                        .convolution_tail
                        .as_ref()
                        .expect("fresh convolution tail"),
                )? < 2e-6
            );
            assert!(
                max_difference(
                    retried.recurrent.as_ref().expect("retried recurrence"),
                    fresh.recurrent.as_ref().expect("fresh recurrence"),
                )? < 2e-6
            );
        }
        Ok(())
    }

    fn biased_golden_tensors(device: &Device) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".into(),
            Tensor::zeros((2, 2), DType::F32, device)?,
        );
        for name in [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ] {
            tensors.insert(name.into(), Tensor::ones(2, DType::F32, device)?);
        }
        for (name, shape) in [
            ("self_attn.q_proj.weight", (4, 2)),
            ("self_attn.k_proj.weight", (2, 2)),
            ("self_attn.v_proj.weight", (2, 2)),
            ("mlp.gate_proj.weight", (2, 2)),
            ("mlp.up_proj.weight", (2, 2)),
        ] {
            tensors.insert(
                format!("model.layers.0.{name}"),
                Tensor::zeros(shape, DType::F32, device)?,
            );
        }
        for (name, values) in [
            ("self_attn.k_proj.bias", vec![0.2, -0.3]),
            ("self_attn.v_proj.bias", vec![1.0, -1.0]),
            ("self_attn.o_proj.bias", vec![0.5, 0.25]),
            ("mlp.gate_proj.bias", vec![0.5, -0.5]),
            ("mlp.up_proj.bias", vec![2.0, 3.0]),
            ("mlp.down_proj.bias", vec![0.25, 0.5]),
        ] {
            tensors.insert(
                format!("model.layers.0.{name}"),
                Tensor::from_vec(values, 2, device)?,
            );
        }
        tensors.insert(
            "model.layers.0.self_attn.q_proj.bias".into(),
            Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], 4, device)?,
        );
        tensors.insert(
            "model.layers.0.self_attn.o_proj.weight".into(),
            Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], (2, 4), device)?,
        );
        for name in ["mlp.down_proj.weight", "lm_head.weight"] {
            let name = if name == "lm_head.weight" {
                name.into()
            } else {
                format!("model.layers.0.{name}")
            };
            tensors.insert(name, Tensor::eye(2, DType::F32, device)?);
        }
        Ok(tensors)
    }

    /// Deterministic pseudo-random values in `[-1, 1)`.
    fn pseudo_random_values(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (state >> 33) as f32 / (1_u64 << 31) as f32;
                unit * 2.0 - 1.0
            })
            .collect()
    }

    /// A causal mask over the last `query_len` positions of the history.
    fn causal_mask(
        query_len: usize,
        key_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let first_visible = key_len.saturating_sub(query_len);
        let values = (0..query_len)
            .flat_map(|query| {
                let query_position = first_visible + query;
                (0..key_len).map(move |key| {
                    if key > query_position {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(values, (query_len, key_len), device)?;
        mask.to_dtype(dtype)?.unsqueeze(0)?.unsqueeze(0)
    }

    /// Expand KV heads for the attention matmul only.
    ///
    /// This is the formula the grouped attention replaced, kept as the
    /// reference implementation of the tests.
    fn expand_kv_heads(input: &Tensor, groups: usize) -> Result<Tensor> {
        if groups == 1 {
            return Ok(input.clone());
        }
        let (batch, kv_heads, sequence, head_dim) = input.dims4()?;
        input
            .unsqueeze(2)?
            .expand((batch, kv_heads, groups, sequence, head_dim))?
            .reshape((batch, kv_heads * groups, sequence, head_dim))
    }

    /// The expand-based attention the grouped path replaced.
    fn expanded_causal_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        kv_groups: usize,
        scale: f64,
        logit_softcap: Option<f64>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let key = expand_kv_heads(key, kv_groups)?.contiguous()?;
        let value = expand_kv_heads(value, kv_groups)?.contiguous()?;
        let scores = query.matmul(&key.transpose(2, 3)?)?.affine(scale, 0.0)?;
        let scores = match logit_softcap {
            Some(softcap) => tanh_softcap(&scores, softcap)?,
            None => scores,
        };
        let scores = match mask {
            Some(mask) => scores.broadcast_add(mask)?,
            None => scores,
        };
        // Match reference implementations: reduction in F32, model dtype after.
        let score_dtype = scores.dtype();
        let probabilities = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
            .to_dtype(score_dtype)?;
        probabilities.matmul(&value)
    }

    /// Query head `h` reads KV head `h / kv_groups`, the pairing the expanded
    /// K/V used to encode.
    #[test]
    fn grouped_attention_maps_adjacent_query_groups() -> Result<()> {
        let device = Device::Cpu;
        // Two KV heads with distinct keys and values, four query heads in
        // two groups of two.
        let key = Tensor::new(&[[[[1.0_f32], [0.0]], [[0.0], [1.0]]]], &device)?;
        let value = Tensor::new(&[[[[1.0_f32], [10.0]], [[-1.0], [-10.0]]]], &device)?;
        let query = Tensor::new(&[[[[1.0_f32]], [[1.0]], [[1.0]], [[1.0]]]], &device)?;
        let output = dense_causal_attention(&query, &key, &value, 2, 1.0, None, None)?;
        let rows = output.flatten_all()?.to_vec1::<f32>()?;
        // The first KV head scores `[1, 0]` and the second one `[0, 1]`, so
        // the first group averages `1` and `10` with the weights of the first
        // softmax and the second group averages `-1` and `-10` with the
        // weights of the second one.
        let first = 0.731_058_6_f32 * 1.0 + 0.268_941_4_f32 * 10.0;
        let second = -0.268_941_4_f32 - 0.731_058_6_f32 * 10.0;
        for (index, expected) in [first, first, second, second].into_iter().enumerate() {
            assert!(
                (rows[index] - expected).abs() <= 1e-4,
                "query head {index} produced {} instead of {expected}",
                rows[index]
            );
        }
        Ok(())
    }

    /// The device the comparison runs on for one dtype.
    ///
    /// The CPU backend only supports F16 matmuls without the accelerate
    /// feature, so the F16 case moves to Metal when the host has it.
    fn comparison_device(dtype: DType) -> Option<Device> {
        if dtype == DType::F32 {
            return Some(Device::Cpu);
        }
        #[cfg(feature = "metal")]
        if let Ok(device) = Device::new_metal(0) {
            return Some(device);
        }
        // Probe the CPU backend: the accelerate build has no F16 matmul.
        let probe = Tensor::ones((2, 2), DType::F16, &Device::Cpu).ok()?;
        probe.matmul(&probe).ok().map(|_| Device::Cpu)
    }

    /// The grouped attention must agree with the expand-based formula it
    /// replaced, for every group count, query length, softcap and dtype.
    #[test]
    fn grouped_attention_matches_the_expanded_reference() -> Result<()> {
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 8;
        const KEY_LEN: usize = 6;
        for dtype in [DType::F32, DType::F16] {
            let Some(device) = comparison_device(dtype) else {
                eprintln!("skipping the {dtype:?} comparison: no F16 matmul backend");
                continue;
            };
            for groups in [1_usize, 2, 4] {
                for (query_len, masked) in [(1_usize, false), (5, true)] {
                    for softcap in [None, Some(30.0_f64)] {
                        let query_heads = KV_HEADS * groups;
                        let query = Tensor::from_vec(
                            pseudo_random_values(query_heads * query_len * HEAD_DIM, 13),
                            (1, query_heads, query_len, HEAD_DIM),
                            &device,
                        )?
                        .to_dtype(dtype)?;
                        let key = Tensor::from_vec(
                            pseudo_random_values(KV_HEADS * KEY_LEN * HEAD_DIM, 29),
                            (1, KV_HEADS, KEY_LEN, HEAD_DIM),
                            &device,
                        )?
                        .to_dtype(dtype)?;
                        let value = Tensor::from_vec(
                            pseudo_random_values(KV_HEADS * KEY_LEN * HEAD_DIM, 47),
                            (1, KV_HEADS, KEY_LEN, HEAD_DIM),
                            &device,
                        )?
                        .to_dtype(dtype)?;
                        let mask = if masked {
                            Some(causal_mask(query_len, KEY_LEN, dtype, &device)?)
                        } else {
                            None
                        };
                        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
                        let grouped = dense_causal_attention(
                            &query,
                            &key,
                            &value,
                            groups,
                            scale,
                            softcap,
                            mask.as_ref(),
                        )?;
                        let expanded = expanded_causal_attention(
                            &query,
                            &key,
                            &value,
                            groups,
                            scale,
                            softcap,
                            mask.as_ref(),
                        )?;
                        let tolerance = if dtype == DType::F32 { 1e-5 } else { 2e-3 };
                        let difference = max_difference(
                            &grouped.to_dtype(DType::F32)?,
                            &expanded.to_dtype(DType::F32)?,
                        )?;
                        assert!(
                            difference <= tolerance,
                            "groups={groups} query_len={query_len} softcap={softcap:?} \
                             {dtype:?} differed by {difference} (tolerance {tolerance})"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// The KV cache hands over strided K/V views, so the grouped attention
    /// must read them without a contiguous copy of the history.
    #[test]
    fn grouped_attention_reads_strided_kv_views() -> Result<()> {
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const KEY_LEN: usize = 5;
        const CAPACITY: usize = 9;
        let device = Device::Cpu;
        for groups in [1_usize, 2, 4] {
            let query_heads = KV_HEADS * groups;
            let query = Tensor::from_vec(
                pseudo_random_values(query_heads * HEAD_DIM, 71),
                (1, query_heads, 1, HEAD_DIM),
                &device,
            )?;
            let key = Tensor::from_vec(
                pseudo_random_values(KV_HEADS * CAPACITY * HEAD_DIM, 83),
                (1, KV_HEADS, CAPACITY, HEAD_DIM),
                &device,
            )?
            .narrow(2, 2, KEY_LEN)?;
            let value = Tensor::from_vec(
                pseudo_random_values(KV_HEADS * CAPACITY * HEAD_DIM, 97),
                (1, KV_HEADS, CAPACITY, HEAD_DIM),
                &device,
            )?
            .narrow(2, 2, KEY_LEN)?;
            assert!(!key.is_contiguous(), "the test needs a strided K/V view");
            let scale = 1.0 / (HEAD_DIM as f64).sqrt();
            let grouped = dense_causal_attention(&query, &key, &value, groups, scale, None, None)?;
            let expanded =
                expanded_causal_attention(&query, &key, &value, groups, scale, None, None)?;
            let difference = max_difference(&grouped, &expanded)?;
            assert!(
                difference <= 1e-5,
                "groups={groups} differed by {difference} on a strided K/V view"
            );
        }
        Ok(())
    }

    /// The dense attention must read the one-block history of the paged cache
    /// as it is: the strided view and its contiguous copy agree exactly.
    #[test]
    fn dense_attention_matches_the_contiguous_reference_for_one_block_kv() -> Result<()> {
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const BLOCK_TOKENS: usize = 8;
        const HISTORY: usize = 5;
        let device = Device::Cpu;
        for groups in [1_usize, 2, 4] {
            let query_heads = KV_HEADS * groups;
            let query = Tensor::from_vec(
                pseudo_random_values(query_heads * HEAD_DIM, 101),
                (1, query_heads, 1, HEAD_DIM),
                &device,
            )?;
            // The cache view of one block: the head blocks of a buffer that
            // holds more tokens than the history uses.
            let key = Tensor::from_vec(
                pseudo_random_values(KV_HEADS * BLOCK_TOKENS * HEAD_DIM, 103),
                (1, KV_HEADS, BLOCK_TOKENS, HEAD_DIM),
                &device,
            )?
            .narrow(2, 0, HISTORY)?;
            let value = Tensor::from_vec(
                pseudo_random_values(KV_HEADS * BLOCK_TOKENS * HEAD_DIM, 107),
                (1, KV_HEADS, BLOCK_TOKENS, HEAD_DIM),
                &device,
            )?
            .narrow(2, 0, HISTORY)?;
            assert!(!key.is_contiguous(), "the test needs a strided K/V view");
            assert!(!value.is_contiguous(), "the test needs a strided V view");
            let scale = 1.0 / (HEAD_DIM as f64).sqrt();
            let strided = dense_causal_attention(&query, &key, &value, groups, scale, None, None)?;
            let contiguous = dense_causal_attention(
                &query,
                &key.contiguous()?,
                &value.contiguous()?,
                groups,
                scale,
                None,
                None,
            )?;
            let difference = max_difference(&strided, &contiguous)?;
            assert!(
                difference == 0.0,
                "groups={groups} differed by {difference} on a one-block history"
            );
        }
        Ok(())
    }

    /// The compact K/V history must reach the matmul as it is, so the
    /// grouped attention never copies the history it reads.
    #[test]
    fn compact_kv_history_reaches_the_matmul_without_a_copy() -> Result<()> {
        const KV_HEADS: usize = 4;
        const HEAD_DIM: usize = 8;
        const CAPACITY: usize = 16;
        const LENGTH: usize = 9;
        let device = Device::Cpu;
        // The cache view of one layer: the head blocks of a buffer that
        // holds more tokens than the history uses.
        let cached = Tensor::zeros((1, KV_HEADS, CAPACITY, HEAD_DIM), DType::F32, &device)?
            .narrow(2, 0, LENGTH)?;
        assert!(!cached.is_contiguous(), "the test needs a strided K/V view");
        assert!(matmul_ready_kv(&cached, 1, KV_HEADS, LENGTH, HEAD_DIM));
        // A view whose head dimension is interleaved with another one cannot
        // be read one head block at a time.
        let interleaved = Tensor::zeros((1, KV_HEADS, LENGTH, HEAD_DIM * 2), DType::F32, &device)?
            .narrow(3, 0, HEAD_DIM)?;
        assert!(!matmul_ready_kv(
            &interleaved,
            1,
            KV_HEADS,
            LENGTH,
            HEAD_DIM
        ));
        Ok(())
    }

    #[test]
    fn partial_rope_rotates_only_the_configured_prefix() -> Result<()> {
        let device = Device::Cpu;
        let rotary = RotaryEmbedding::new(
            &RopeSpec {
                theta: 10_000.0,
                layout: RopeLayout::HalfSplit,
                rotary_dim: 2,
                frequency_factors: None,
            },
            4,
            8,
            DType::F32,
            &device,
        )?;
        let query = Tensor::new(
            &[[[[1.0_f32, 2.0, 30.0, 40.0], [3.0, 4.0, 50.0, 60.0]]]],
            &device,
        )?;
        let key = Tensor::new(
            &[[[[5.0_f32, 6.0, 70.0, 80.0], [7.0, 8.0, 90.0, 100.0]]]],
            &device,
        )?;
        let (rotated_query, rotated_key) = rotary.apply(&query, &key, 1)?;

        assert!(
            max_difference(&rotated_query.narrow(3, 2, 2)?, &query.narrow(3, 2, 2)?)?
                < f32::EPSILON
        );
        assert!(
            max_difference(&rotated_key.narrow(3, 2, 2)?, &key.narrow(3, 2, 2)?)? < f32::EPSILON
        );
        assert!(max_difference(&rotated_query.narrow(3, 0, 2)?, &query.narrow(3, 0, 2)?)? > 1e-3);
        assert!(max_difference(&rotated_key.narrow(3, 0, 2)?, &key.narrow(3, 0, 2)?)? > 1e-3);
        Ok(())
    }

    #[test]
    fn gated_attention_splits_each_interleaved_head_and_gates_attention_values() -> Result<()> {
        let device = Device::Cpu;
        let rotary = Arc::new(RotaryEmbedding::new(
            &RopeSpec {
                theta: 10_000.0,
                layout: RopeLayout::HalfSplit,
                rotary_dim: 2,
                frequency_factors: None,
            },
            2,
            4,
            DType::F32,
            &device,
        )?);
        let attention = SelfAttention {
            query: LinearOp::Dense(candle_nn::Linear::new(
                // Per head: [q0, q1, gate0, gate1].
                Tensor::from_vec(
                    vec![1.0_f32, 2.0, 0.0, -1.0, 5.0, 6.0, 1.0, 0.5],
                    (8, 1),
                    &device,
                )?,
                None,
            )),
            key: LinearOp::Dense(candle_nn::Linear::new(
                Tensor::ones((4, 1), DType::F32, &device)?,
                None,
            )),
            value: LinearOp::Dense(candle_nn::Linear::new(
                Tensor::from_vec(vec![2.0_f32, 4.0, 6.0, 8.0], (4, 1), &device)?,
                None,
            )),
            output: LinearOp::Dense(candle_nn::Linear::new(
                Tensor::eye(4, DType::F32, &device)?,
                None,
            )),
            query_norm: None,
            key_norm: None,
            rotary: Some(rotary),
            kind: AttentionKind::FullCausal,
            num_query_heads: 2,
            num_kv_heads: 2,
            kv_groups: 1,
            head_dim: 2,
            attention_width: 4,
            scale: 1.0 / 2.0_f64.sqrt(),
            logit_softcap: None,
            output_gate: true,
        };
        let input = Tensor::new(&[[[1.0_f32]]], &device)?;
        let (query, _, _, gate) = attention.project_packed_qkv(&input, 1)?;
        assert_eq!(
            query.flatten_all()?.to_vec1::<f32>()?,
            vec![1.0, 2.0, 5.0, 6.0]
        );
        assert_eq!(
            gate.expect("gated attention returns its projected gate")
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![0.0, -1.0, 1.0, 0.5]
        );

        let mut cache = LayerKvCache::Full(KvCache::new(4));
        let actual = attention.forward(&mut cache, &input, None, 0)?;
        let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());
        let expected = Tensor::new(
            &[[[
                2.0_f32 * sigmoid(0.0),
                4.0 * sigmoid(-1.0),
                6.0 * sigmoid(1.0),
                8.0 * sigmoid(0.5),
            ]]],
            &device,
        )?;
        assert!(max_difference(&actual, &expected)? < 2e-6);
        Ok(())
    }

    #[test]
    fn gated_delta_repeats_key_heads_for_gguf_tiled_value_order() -> Result<()> {
        let input = Tensor::new(&[[1_f32, 2.0], [3.0, 4.0]], &Device::Cpu)?;
        let expanded = repeat_heads_tiled_dim(&input, 2, 0)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(expanded, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn selected_softmax_routes_only_normalizes_top_k_with_stable_ties() -> Result<()> {
        let logits = Tensor::new(
            &[[1.0_f32, 3.0, 2.0, -5.0], [4.0_f32, 4.0, 0.0, -1.0]],
            &Device::Cpu,
        )?;
        let routes = selected_softmax_routes(&logits, 4, 2)?;

        // Token zero selects experts 1 and 2. Token one has a tie, resolved in
        // favor of the lower expert IDs 0 then 1.
        assert_eq!(routes.rows, vec![vec![1], vec![0, 1], vec![0], vec![]]);
        let logistic = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        assert!((routes.weights[0][0] - 0.5).abs() < 1e-7);
        assert!((routes.weights[1][0] - logistic).abs() < 1e-7);
        assert!((routes.weights[1][1] - 0.5).abs() < 1e-7);
        assert!((routes.weights[2][0] - (1.0 - logistic)).abs() < 1e-7);
        assert!(routes.weights[3].is_empty());
        Ok(())
    }

    #[test]
    fn selected_softmax_routes_rejects_invalid_or_non_finite_logits() -> Result<()> {
        let logits = Tensor::new(&[[1.0_f32, f32::NAN]], &Device::Cpu)?;
        assert!(selected_softmax_routes(&logits, 2, 1).is_err());
        let logits = Tensor::new(&[[1.0_f32, 2.0]], &Device::Cpu)?;
        assert!(selected_softmax_routes(&logits, 3, 1).is_err());
        assert!(selected_softmax_routes(&logits, 2, 0).is_err());
        assert!(selected_softmax_routes(&logits, 2, 3).is_err());
        Ok(())
    }

    #[test]
    fn selected_softmax_routes_treats_signed_zero_as_a_tie_and_supports_k_equal_e() -> Result<()> {
        let signed_zero = Tensor::new(&[[0.0_f32, -0.0]], &Device::Cpu)?;
        let routes = selected_softmax_routes(&signed_zero, 2, 1)?;
        assert_eq!(routes.rows, [vec![0], vec![]]);

        let all = Tensor::new(&[[3.0_f32, 1.0, 2.0, 0.0]], &Device::Cpu)?;
        let routes = selected_softmax_routes(&all, 4, 4)?;
        assert!(routes.rows.iter().all(|rows| rows == &[0]));
        let total = routes.weights.iter().map(|weights| weights[0]).sum::<f32>();
        assert!((total - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn sparse_moe_dense_bank_routes_and_index_adds_exact_outputs() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.hidden_size = 2;
        let moe = SparseMoeSpec {
            expert_count: 3,
            experts_per_token: 2,
            expert_intermediate_size: 1,
            router_weighting: RouterWeighting::SelectedSoftmax,
        };
        let inverse_silu_one = 1.0_f32 + (-1.0_f32).exp();
        let tensors = HashMap::from([
            (
                "model.layers.0.mlp.router.weight".into(),
                Tensor::from_vec(
                    vec![
                        4.0_f32.ln(),
                        0.0,
                        3.0_f32.ln(),
                        2.0_f32.ln(),
                        0.0,
                        6.0_f32.ln(),
                    ],
                    (3, 2),
                    &device,
                )?,
            ),
            (
                "model.layers.0.mlp.experts.gate_proj.weight".into(),
                Tensor::ones((3, 1, 2), DType::F32, &device)?,
            ),
            (
                "model.layers.0.mlp.experts.up_proj.weight".into(),
                Tensor::ones((3, 1, 2), DType::F32, &device)?,
            ),
            (
                "model.layers.0.mlp.experts.down_proj.weight".into(),
                Tensor::from_vec(
                    vec![
                        inverse_silu_one,
                        0.0,
                        0.0,
                        2.0 * inverse_silu_one,
                        3.0 * inverse_silu_one,
                        3.0 * inverse_silu_one,
                    ],
                    (3, 2, 1),
                    &device,
                )?,
            ),
        ]);
        let source = DenseWeightSource::new(VarBuilder::from_tensors(tensors, DType::F32, &device));
        let feed_forward = SparseMoeFeedForward::load(
            &spec,
            moe,
            Activation::Silu,
            STANDARD_DECODER_WEIGHT_NAMES.sparse_moe,
            WeightPath::root(&source).pp("model.layers.0.mlp"),
        )?;
        let input = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], (1, 2, 2), &device)?;
        let actual = feed_forward
            .forward(&input)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = [4.0 / 7.0, 6.0 / 7.0, 9.0 / 4.0, 11.0 / 4.0];
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "MoE output {index} was {actual}, expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    #[allow(clippy::too_many_lines)] // Explicit weights make the scale ordering auditable.
    fn sparse_moe_decoder_applies_runtime_scales_and_preserves_cache_parity() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.runtime.vocab_size = 2;
        spec.runtime.max_context = 4;
        spec.runtime.stop_token_ids = vec![1];
        spec.hidden_size = 2;
        spec.intermediate_size = 1;
        spec.num_layers = 1;
        spec.num_query_heads = 1;
        spec.num_kv_heads = 1;
        spec.head_dim = 2;
        spec.layer_specs.truncate(1);
        let attention = attention_spec_mut(&mut spec, 0);
        attention.scale = 1.0 / 2.0_f64.sqrt();
        attention
            .rope_mut()
            .expect("test attention uses RoPE")
            .rotary_dim = 2;
        spec.layer_specs[0].mixer_residual_scale = 0.5;
        spec.layer_specs[0].feed_forward_residual_scale = 0.25;
        spec.layer_specs[0].feed_forward = FeedForwardKind::SparseMoe(SparseMoeSpec {
            expert_count: 1,
            experts_per_token: 1,
            expert_intermediate_size: 1,
            router_weighting: RouterWeighting::SelectedSoftmax,
        });
        spec.embedding_scale = 2.0;
        spec.final_logit_scale = 0.2;
        spec.projection_bias = ProjectionBias {
            query: false,
            key: false,
            value: false,
            output: false,
        };
        spec.mlp_bias = false;
        spec.tied_embeddings = false;

        let tensors = HashMap::from([
            (
                "model.embed_tokens.weight".into(),
                Tensor::eye(2, DType::F32, &device)?,
            ),
            (
                "model.norm.weight".into(),
                Tensor::ones(2, DType::F32, &device)?,
            ),
            (
                "model.layers.0.input_layernorm.weight".into(),
                Tensor::ones(2, DType::F32, &device)?,
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".into(),
                Tensor::ones(2, DType::F32, &device)?,
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".into(),
                Tensor::zeros((2, 2), DType::F32, &device)?,
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".into(),
                Tensor::zeros((2, 2), DType::F32, &device)?,
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".into(),
                Tensor::from_vec(vec![0.0_f32, 0.0, 1.0, 0.0], (2, 2), &device)?,
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".into(),
                Tensor::eye(2, DType::F32, &device)?,
            ),
            (
                "model.layers.0.mlp.router.weight".into(),
                Tensor::zeros((1, 2), DType::F32, &device)?,
            ),
            (
                "model.layers.0.mlp.experts.gate_proj.weight".into(),
                Tensor::from_vec(vec![1.0_f32, 0.0], (1, 1, 2), &device)?,
            ),
            (
                "model.layers.0.mlp.experts.up_proj.weight".into(),
                Tensor::from_vec(vec![0.0_f32, 1.0], (1, 1, 2), &device)?,
            ),
            (
                "model.layers.0.mlp.experts.down_proj.weight".into(),
                Tensor::from_vec(vec![1.0_f32, -1.0], (1, 2, 1), &device)?,
            ),
            (
                "lm_head.weight".into(),
                Tensor::eye(2, DType::F32, &device)?,
            ),
        ]);
        let model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut model_session = new_session(&model)?;
        let actual = model
            .forward_tokens(model_session.as_mut(), &[0], 0)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let rms = |values: [f32; 2]| {
            let denominator =
                (values[0].powi(2).midpoint(values[1].powi(2)) + spec.norm_epsilon as f32).sqrt();
            [values[0] / denominator, values[1] / denominator]
        };
        let embedded = [2.0_f32, 0.0];
        let attention_input = rms(embedded);
        let after_attention = [embedded[0], embedded[1] + 0.5 * attention_input[0]];
        let feed_forward_input = rms(after_attention);
        let gate = feed_forward_input[0];
        let routed = gate / (1.0 + (-gate).exp()) * feed_forward_input[1];
        let after_feed_forward = [
            after_attention[0] + 0.25 * routed,
            after_attention[1] - 0.25 * routed,
        ];
        let normalized = rms(after_feed_forward);
        let expected = [normalized[0] * 0.2, normalized[1] * 0.2];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let cached = model.forward_tokens(model_session.as_mut(), &[1], 1)?;
        let fresh = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut fresh_session = new_session(&fresh)?;
        let full = fresh.forward_tokens(fresh_session.as_mut(), &[0, 1], 0)?;
        let difference = max_difference(&cached, &full)?;
        assert!(
            difference < 1e-5,
            "MoE cache/full difference was {difference}"
        );

        let mut invalid_tensors = tensors;
        invalid_tensors.insert(
            "model.layers.0.mlp.router.weight".into(),
            Tensor::new(&[[f32::NAN, 0.0]], &device)?,
        );
        let failing = DecoderModel::load_dense(
            spec,
            VarBuilder::from_tensors(invalid_tensors, DType::F32, &device),
        )?;
        let mut failing_session = new_session(&failing)?;
        assert!(
            failing
                .forward_tokens(failing_session.as_mut(), &[0], 0)
                .is_err()
        );
        assert_eq!(failing_session.cache_len(), 0);
        Ok(())
    }

    #[test]
    fn rejects_rope_scaling_that_underflows_an_inverse_frequency() {
        let mut spec = tiny_spec();
        spec.head_dim = 4;
        attention_spec_mut(&mut spec, 0)
            .rope_mut()
            .expect("test attention uses RoPE")
            .theta = f64::MAX;
        attention_spec_mut(&mut spec, 0)
            .rope_mut()
            .expect("test attention uses RoPE")
            .frequency_factors = Some(vec![1.0, f32::MAX]);
        spec.layer_specs[1] = spec.layer_specs[0].clone();
        spec.validate().unwrap();
        let error = RotaryEmbedding::new(
            spec.layer_specs[0]
                .attention()
                .expect("test layer uses attention")
                .rope()
                .expect("test attention uses RoPE"),
            spec.head_dim,
            spec.runtime.max_context,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("inverse frequency 1"));
    }

    #[test]
    fn prefill_then_decode_advances_cache() -> Result<()> {
        let device = Device::Cpu;
        let weights = VarBuilder::zeros(DType::F32, &device);
        let model = DecoderModel::load_dense(tiny_spec(), weights)?;
        let mut session = new_session(&model)?;
        let error = model
            .forward_tokens(session.as_mut(), &[32], 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside the model vocabulary"));
        assert_eq!(session.cache_len(), 0);
        let prompt = Tensor::new(&[[1_u32, 2, 3]], &device)?;
        let logits = model.forward(session.as_mut(), &prompt, 0)?;
        assert_eq!(logits.dims2()?, (1, 32));
        assert_eq!(session.cache_len(), 3);
        let next = Tensor::new(&[[4_u32]], &device)?;
        let _ = model.forward(session.as_mut(), &next, 3)?;
        assert_eq!(session.cache_len(), 4);
        assert!(model.forward(session.as_mut(), &next, 0).is_err());
        session.clear_cache();
        assert_eq!(session.cache_len(), 0);
        Ok(())
    }

    #[test]
    fn mixed_same_start_ragged_batch_matches_serial_lanes() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        configure_multi_session_test_pool(&mut model)?;
        assert_eq!(
            model.mixed_batch_capability(),
            CausalLmBatchCapability::PackedRagged
        );
        let mut mixed_first = new_session(&model)?;
        let mut mixed_second = new_session(&model)?;
        let mut serial_first = new_session(&model)?;
        let mut serial_second = new_session(&model)?;
        let first_tokens = [1_u32, 2, 3];
        let second_tokens = [4_u32];

        let mixed = {
            let mut inputs = [
                CausalLmBatchInput::new(mixed_first.as_mut(), &first_tokens, 0),
                CausalLmBatchInput::new(mixed_second.as_mut(), &second_tokens, 0),
            ];
            model.forward_batch_tokens(&mut inputs)?
        };
        let first_expected = model.forward_tokens(serial_first.as_mut(), &first_tokens, 0)?;
        let second_expected = model.forward_tokens(serial_second.as_mut(), &second_tokens, 0)?;

        assert_eq!(mixed.dims2()?, (2, model.spec.runtime.vocab_size));
        assert!(max_difference(&first_expected, &second_expected)? > 1e-5);
        assert!(max_difference(&mixed.narrow(0, 0, 1)?, &first_expected)? < 1e-5);
        assert!(max_difference(&mixed.narrow(0, 1, 1)?, &second_expected)? < 1e-5);
        assert_eq!(mixed_first.cache_len(), first_tokens.len());
        assert_eq!(mixed_second.cache_len(), second_tokens.len());
        Ok(())
    }

    #[test]
    fn paged_full_cache_matches_chunked_execution_at_15_16_17_tokens() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.runtime.max_context = 32;
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let tokens = (1_u32..=17).collect::<Vec<_>>();

        let one_shot = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut one_shot_session = new_session(&one_shot)?;
        let expected = one_shot.forward_tokens(one_shot_session.as_mut(), &tokens, 0)?;
        for stats in cache_stats(one_shot_session.as_ref()) {
            assert_eq!(stats.policy, KvCachePolicy::Paged);
            assert_eq!(stats.logical_len, 17);
            assert_eq!(stats.resident_len, 17);
            assert_eq!(stats.allocated_capacity, 32);
        }

        let chunked =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut chunked_session = new_session(&chunked)?;
        let _ = chunked.forward_tokens(chunked_session.as_mut(), &tokens[..15], 0)?;
        assert!(
            cache_stats(chunked_session.as_ref())
                .iter()
                .all(|stats| stats.logical_len == 15 && stats.allocated_capacity == 16)
        );
        let _ = chunked.forward_tokens(chunked_session.as_mut(), &tokens[15..16], 15)?;
        assert!(
            cache_stats(chunked_session.as_ref())
                .iter()
                .all(|stats| stats.logical_len == 16 && stats.allocated_capacity == 16)
        );
        let actual = chunked.forward_tokens(chunked_session.as_mut(), &tokens[16..], 16)?;
        assert!(
            cache_stats(chunked_session.as_ref())
                .iter()
                .all(|stats| stats.logical_len == 17 && stats.allocated_capacity == 32)
        );
        assert!(max_difference(&actual, &expected)? < 1e-4);
        Ok(())
    }

    #[test]
    fn full_causal_mask_is_lazy_shared_and_direct_pages_skip_materialization() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_parallel_mamba2_decoder_spec();
        spec.layer_specs[0] = tiny_spec().layer_specs[0].clone();
        let tensors = tiny_parallel_mamba2_decoder_tensors(&spec, &device)?;
        let full_layers = spec.num_layers;

        let single = load_tiny_parallel_mamba2_decoder(spec.clone(), tensors.clone(), &device)?;
        let mut single_session = new_session(&single)?;
        reset_full_causal_mask_construction_count();
        crate::cache::reset_paged_batch_materialization_count();
        let _ = single.forward_tokens(single_session.as_mut(), &[1], 0)?;
        assert_eq!(
            full_causal_mask_construction_count(),
            0,
            "one-token decode/prefill must not allocate a causal mask"
        );
        assert_eq!(
            crate::cache::paged_batch_materialization_count(),
            full_layers,
            "the CPU fallback still materializes each full-attention layer"
        );

        let mut fallback =
            load_tiny_parallel_mamba2_decoder(spec.clone(), tensors.clone(), &device)?;
        configure_multi_session_test_pool(&mut fallback)?;
        let mut fallback_session = new_session(&fallback)?;
        reset_full_causal_mask_construction_count();
        crate::cache::reset_paged_batch_materialization_count();
        let expected = fallback.forward_tokens(fallback_session.as_mut(), &[1, 2, 3, 4], 0)?;
        assert_eq!(
            full_causal_mask_construction_count(),
            1,
            "one lane must reuse one mask across standard and parallel-attention layers"
        );
        assert_eq!(
            crate::cache::paged_batch_materialization_count(),
            full_layers
        );

        let mut first_lane = new_session(&fallback)?;
        let mut second_lane = new_session(&fallback)?;
        let first_tokens = [5_u32, 6, 7];
        let second_tokens = [8_u32, 9];
        reset_full_causal_mask_construction_count();
        crate::cache::reset_paged_batch_materialization_count();
        let mut inputs = [
            CausalLmBatchInput::new(first_lane.as_mut(), &first_tokens, 0),
            CausalLmBatchInput::new(second_lane.as_mut(), &second_tokens, 0),
        ];
        let _ = fallback.forward_batch_tokens(&mut inputs)?;
        assert_eq!(
            full_causal_mask_construction_count(),
            2,
            "each fallback lane must build one independently shaped mask"
        );
        assert_eq!(
            crate::cache::paged_batch_materialization_count(),
            full_layers * 2
        );

        let direct = load_tiny_parallel_mamba2_decoder(spec, tensors, &device)?;
        let mut direct_session = new_session(&direct)?;
        reset_full_causal_mask_construction_count();
        crate::cache::reset_paged_batch_materialization_count();
        let direct_guard = ForceCpuDirectPagedAttention::enable();
        let actual = direct.forward_tokens(direct_session.as_mut(), &[1, 2, 3, 4], 0)?;
        drop(direct_guard);
        assert_eq!(full_causal_mask_construction_count(), 0);
        assert_eq!(crate::cache::paged_batch_materialization_count(), 0);
        assert!(max_difference(&actual, &expected)? < 1e-5);
        Ok(())
    }

    #[test]
    fn packed_direct_pages_batch_preserves_ragged_causal_boundaries() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        configure_multi_session_test_pool(&mut model)?;
        let mut packed_first = new_session(&model)?;
        let mut packed_second = new_session(&model)?;
        let mut serial_first = new_session(&model)?;
        let mut serial_second = new_session(&model)?;
        let first_prefix = [1_u32, 2, 3];
        let second_prefix = [4_u32];
        for session in [&mut packed_first, &mut serial_first] {
            let _ = model.forward_tokens(session.as_mut(), &first_prefix, 0)?;
        }
        for session in [&mut packed_second, &mut serial_second] {
            let _ = model.forward_tokens(session.as_mut(), &second_prefix, 0)?;
        }

        let first_suffix = [5_u32];
        let second_suffix = [6_u32, 7];
        reset_full_causal_mask_construction_count();
        crate::cache::reset_paged_batch_materialization_count();
        let direct_guard = ForceCpuDirectPagedAttention::enable();
        let packed = {
            let mut inputs = [
                CausalLmBatchInput::new(packed_first.as_mut(), &first_suffix, first_prefix.len()),
                CausalLmBatchInput::new(
                    packed_second.as_mut(),
                    &second_suffix,
                    second_prefix.len(),
                ),
            ];
            model.forward_batch_tokens(&mut inputs)?
        };
        drop(direct_guard);
        assert_eq!(full_causal_mask_construction_count(), 0);
        assert_eq!(crate::cache::paged_batch_materialization_count(), 0);

        let first_expected =
            model.forward_tokens(serial_first.as_mut(), &first_suffix, first_prefix.len())?;
        let second_expected =
            model.forward_tokens(serial_second.as_mut(), &second_suffix, second_prefix.len())?;
        assert!(max_difference(&packed.narrow(0, 0, 1)?, &first_expected)? < 1e-5);
        assert!(max_difference(&packed.narrow(0, 1, 1)?, &second_expected)? < 1e-5);
        assert_eq!(packed_first.cache_len(), 4);
        assert_eq!(packed_second.cache_len(), 3);
        Ok(())
    }

    #[test]
    fn external_paged_reservation_rejects_swapped_lanes_and_unlocks_tables() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        configure_multi_session_test_pool(&mut model)?;
        let mut first = new_session(&model)?;
        let mut second = new_session(&model)?;
        let reservation = model
            .reserve_batch_kv(&[
                CausalLmKvReservationInput::new(first.as_ref(), 0, 1),
                CausalLmKvReservationInput::new(second.as_ref(), 0, 1),
            ])
            .map_err(kv_reservation_error)?
            .expect("full decoder returns an owned reservation");
        let first_token = [1_u32];
        let second_token = [2_u32];
        let error = {
            let mut swapped = [
                CausalLmBatchInput::new(second.as_mut(), &second_token, 0),
                CausalLmBatchInput::new(first.as_mut(), &first_token, 0),
            ];
            model
                .forward_batch_tokens_reserved(&mut swapped, Some(reservation))
                .unwrap_err()
        };
        assert!(
            error
                .to_string()
                .contains("session, order, start or token count")
        );
        assert_eq!(first.cache_len(), 0);
        assert_eq!(second.cache_len(), 0);

        let logits = {
            let mut ordered = [
                CausalLmBatchInput::new(first.as_mut(), &first_token, 0),
                CausalLmBatchInput::new(second.as_mut(), &second_token, 0),
            ];
            model.forward_batch_tokens(&mut ordered)?
        };
        assert_eq!(logits.dims2()?, (2, model.spec.runtime.vocab_size));
        assert_eq!(first.cache_len(), 1);
        assert_eq!(second.cache_len(), 1);
        Ok(())
    }

    #[test]
    fn completed_clear_releases_a_single_block_pool_for_a_peer() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        model.configure_kv_cache(PagedKvPoolConfig::new(16, 1, 1).map_err(paged_kv_error)?)?;
        let mut first = new_session(&model)?;
        let mut second = new_session(&model)?;
        let _ = model.forward_tokens(first.as_mut(), &[1], 0)?;
        assert_eq!(
            model.kv_cache_capacity().map(|stats| stats.live_blocks),
            Some(1)
        );
        let error = model
            .forward_tokens(second.as_mut(), &[2], 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("only 0 of 1 are free"));
        assert_eq!(second.cache_len(), 0);

        first.clear_cache();
        let capacity = model
            .kv_cache_capacity()
            .expect("model exposes pool capacity");
        assert_eq!(capacity.live_blocks, 0);
        assert_eq!(capacity.free_blocks, 1);
        let _ = model.forward_tokens(second.as_mut(), &[2], 0)?;
        assert_eq!(second.cache_len(), 1);
        Ok(())
    }

    #[test]
    fn mixed_ragged_batch_with_different_starts_matches_serial_lanes() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        configure_multi_session_test_pool(&mut model)?;
        let mut mixed_first = new_session(&model)?;
        let mut mixed_second = new_session(&model)?;
        let mut serial_first = new_session(&model)?;
        let mut serial_second = new_session(&model)?;
        let first_prefix = [1_u32, 2, 3];
        let second_prefix = [4_u32];
        for session in [&mut mixed_first, &mut serial_first] {
            let _ = model.forward_tokens(session.as_mut(), &first_prefix, 0)?;
        }
        for session in [&mut mixed_second, &mut serial_second] {
            let _ = model.forward_tokens(session.as_mut(), &second_prefix, 0)?;
        }
        let first_suffix = [5_u32];
        let second_suffix = [6_u32, 7];

        let mixed = {
            let mut inputs = [
                CausalLmBatchInput::new(mixed_first.as_mut(), &first_suffix, first_prefix.len()),
                CausalLmBatchInput::new(mixed_second.as_mut(), &second_suffix, second_prefix.len()),
            ];
            model.forward_batch_tokens(&mut inputs)?
        };
        let first_expected =
            model.forward_tokens(serial_first.as_mut(), &first_suffix, first_prefix.len())?;
        let second_expected =
            model.forward_tokens(serial_second.as_mut(), &second_suffix, second_prefix.len())?;

        assert!(max_difference(&mixed.narrow(0, 0, 1)?, &first_expected)? < 1e-5);
        assert!(max_difference(&mixed.narrow(0, 1, 1)?, &second_expected)? < 1e-5);
        assert_eq!(mixed_first.cache_len(), 4);
        assert_eq!(mixed_second.cache_len(), 3);
        Ok(())
    }

    #[test]
    fn mixed_batch_prevalidation_is_atomic_across_lanes() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let foreign =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut first = new_session(&model)?;
        let mut foreign_session = new_session(&foreign)?;
        let _ = model.forward_tokens(first.as_mut(), &[1], 0)?;
        let before = logical_cache_state(first.as_ref());
        let first_suffix = [2_u32];
        let foreign_tokens = [3_u32];

        let error = {
            let mut inputs = [
                CausalLmBatchInput::new(first.as_mut(), &first_suffix, 1),
                CausalLmBatchInput::new(foreign_session.as_mut(), &foreign_tokens, 0),
            ];
            model.forward_batch_tokens(&mut inputs).unwrap_err()
        };
        assert!(error.to_string().contains("different model instance"));
        assert_eq!(logical_cache_state(first.as_ref()), before);

        let mut second = new_session(&model)?;
        let invalid = [model.spec.runtime.vocab_size as u32];
        let error = {
            let mut inputs = [
                CausalLmBatchInput::new(first.as_mut(), &first_suffix, 1),
                CausalLmBatchInput::new(second.as_mut(), &invalid, 0),
            ];
            model.forward_batch_tokens(&mut inputs).unwrap_err()
        };
        assert!(error.to_string().contains("outside the model vocabulary"));
        assert_eq!(logical_cache_state(first.as_ref()), before);
        assert_eq!(second.cache_len(), 0);

        let second_tokens = [3_u32];
        let error = {
            let mut inputs = [
                CausalLmBatchInput::new(first.as_mut(), &first_suffix, 1),
                CausalLmBatchInput::new(second.as_mut(), &second_tokens, 1),
            ];
            model.forward_batch_tokens(&mut inputs).unwrap_err()
        };
        assert!(
            error
                .to_string()
                .contains("does not match decoder sequence length")
        );
        assert_eq!(logical_cache_state(first.as_ref()), before);
        assert_eq!(second.cache_len(), 0);
        Ok(())
    }

    #[test]
    fn failed_mixed_batch_rolls_back_every_lane_and_retries() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        attention_spec_mut(&mut spec, 1).kind = AttentionKind::SlidingCausal { window: 3 };
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        configure_multi_session_test_pool(&mut model)?;
        let mut first = new_session(&model)?;
        let mut second = new_session(&model)?;
        let _ = model.forward_tokens(first.as_mut(), &[1, 2, 3], 0)?;
        let _ = model.forward_tokens(first.as_mut(), &[4], 3)?;
        let _ = model.forward_tokens(second.as_mut(), &[5], 0)?;
        let first_before = logical_cache_state(first.as_ref());
        let second_before = logical_cache_state(second.as_ref());
        let valid_head = model.lm_head.clone();
        model.lm_head = LinearOp::Dense(candle_nn::Linear::new(
            Tensor::zeros((1, 1), DType::F32, &device)?,
            None,
        ));
        let first_suffix = [6_u32, 7];
        let second_suffix = [8_u32];

        let failed = {
            let mut inputs = [
                CausalLmBatchInput::new(first.as_mut(), &first_suffix, 4),
                CausalLmBatchInput::new(second.as_mut(), &second_suffix, 1),
            ];
            model.forward_batch_tokens(&mut inputs)
        };
        assert!(failed.is_err());
        assert_eq!(logical_cache_state(first.as_ref()), first_before);
        assert_eq!(logical_cache_state(second.as_ref()), second_before);

        model.lm_head = valid_head;
        let retried = {
            let mut inputs = [
                CausalLmBatchInput::new(first.as_mut(), &first_suffix, 4),
                CausalLmBatchInput::new(second.as_mut(), &second_suffix, 1),
            ];
            model.forward_batch_tokens(&mut inputs)?
        };
        let mut first_control = new_session(&model)?;
        let mut second_control = new_session(&model)?;
        let _ = model.forward_tokens(first_control.as_mut(), &[1, 2, 3], 0)?;
        let _ = model.forward_tokens(first_control.as_mut(), &[4], 3)?;
        let _ = model.forward_tokens(second_control.as_mut(), &[5], 0)?;
        let first_expected = model.forward_tokens(first_control.as_mut(), &first_suffix, 4)?;
        let second_expected = model.forward_tokens(second_control.as_mut(), &second_suffix, 1)?;

        assert!(max_difference(&retried.narrow(0, 0, 1)?, &first_expected)? < 1e-5);
        assert!(max_difference(&retried.narrow(0, 1, 1)?, &second_expected)? < 1e-5);
        assert_eq!(
            logical_cache_state(first.as_ref()),
            logical_cache_state(first_control.as_ref())
        );
        assert_eq!(
            logical_cache_state(second.as_ref()),
            logical_cache_state(second_control.as_ref())
        );
        Ok(())
    }

    #[test]
    fn hybrid_sliding_then_full_failure_rolls_back_both_and_retries() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.runtime.max_context = 32;
        attention_spec_mut(&mut spec, 0).kind = AttentionKind::SlidingCausal { window: 4 };
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let prefix = (1_u32..=15).collect::<Vec<_>>();
        let suffix = [16_u32, 17];

        let control = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut control_session = new_session(&control)?;
        let _ = control.forward_tokens(control_session.as_mut(), &prefix, 0)?;
        let expected = control.forward_tokens(control_session.as_mut(), &suffix, prefix.len())?;

        let mut model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut session = new_session(&model)?;
        let _ = model.forward_tokens(session.as_mut(), &prefix, 0)?;
        let checkpoint = cache_stats(session.as_ref());
        assert_eq!(checkpoint[0].resident_len, 3);
        assert_eq!(checkpoint[1].allocated_capacity, 16);

        let valid_output = layer_attention(&model, 1).output.clone();
        layer_attention_mut(&mut model, 1).output = LinearOp::Dense(candle_nn::Linear::new(
            Tensor::zeros((1, 1), DType::F32, &device)?,
            None,
        ));
        assert!(
            model
                .forward_tokens(session.as_mut(), &suffix, prefix.len())
                .is_err()
        );
        assert_eq!(session.cache_len(), prefix.len());
        assert_eq!(cache_stats(session.as_ref()), checkpoint);
        model.synchronize()?;

        layer_attention_mut(&mut model, 1).output = valid_output;
        let actual = model.forward_tokens(session.as_mut(), &suffix, prefix.len())?;
        assert!(max_difference(&actual, &expected)? < 1e-4);
        assert_eq!(
            logical_cache_state(session.as_ref()),
            logical_cache_state(control_session.as_ref())
        );
        Ok(())
    }

    #[test]
    fn one_session_rejects_a_multi_sequence_tensor_without_mutation() -> Result<()> {
        let device = Device::Cpu;
        let model = DecoderModel::load_dense(tiny_spec(), VarBuilder::zeros(DType::F32, &device))?;
        let mut session = new_session(&model)?;
        let input = Tensor::new(&[[1_u32, 2], [3, 4]], &device)?;

        let error = model
            .forward(session.as_mut(), &input, 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires batch size 1"));
        assert_eq!(session.cache_len(), 0);

        let logits = model.forward_tokens(session.as_mut(), &[1, 2], 0)?;
        assert_eq!(logits.dims2()?, (1, 32));
        assert_eq!(session.cache_len(), 2);
        Ok(())
    }

    #[test]
    fn interleaved_sessions_are_independent_and_drop_does_not_touch_peers() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let mut model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        configure_multi_session_test_pool(&mut model)?;
        let mut first = new_session(&model)?;
        let mut second = new_session(&model)?;
        let mut first_fresh = new_session(&model)?;
        let mut second_fresh = new_session(&model)?;

        let _ = model.forward_tokens(first.as_mut(), &[1, 2], 0)?;
        let _ = model.forward_tokens(second.as_mut(), &[5], 0)?;
        let first_logits = model.forward_tokens(first.as_mut(), &[3], 2)?;
        let second_logits = model.forward_tokens(second.as_mut(), &[6, 7], 1)?;

        let first_expected = model.forward_tokens(first_fresh.as_mut(), &[1, 2, 3], 0)?;
        let second_expected = model.forward_tokens(second_fresh.as_mut(), &[5, 6, 7], 0)?;
        assert!(max_difference(&first_logits, &first_expected)? < 1e-4);
        assert!(max_difference(&second_logits, &second_expected)? < 1e-4);
        assert_eq!(first.cache_len(), 3);
        assert_eq!(second.cache_len(), 3);

        let second_stats = cache_stats(second.as_ref());
        first.clear_cache();
        assert_eq!(first.cache_len(), 0);
        assert_eq!(cache_stats(second.as_ref()), second_stats);
        drop(first);

        let mut replacement = new_session(&model)?;
        assert_eq!(replacement.cache_len(), 0);
        let _ = model.forward_tokens(replacement.as_mut(), &[8], 0)?;
        assert_eq!(replacement.cache_len(), 1);
        assert_eq!(second.cache_len(), 3);
        Ok(())
    }

    #[test]
    fn decoder_rejects_a_session_from_another_model_without_mutating_it() -> Result<()> {
        let device = Device::Cpu;
        let spec = tiny_spec();
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let first = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let second =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut session = new_session(&first)?;

        let error = second
            .forward_tokens(session.as_mut(), &[1], 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("different model instance"));
        assert_eq!(session.cache_len(), 0);

        let logits = first.forward_tokens(session.as_mut(), &[1], 0)?;
        assert_eq!(logits.dims2()?, (1, 32));
        assert_eq!(session.cache_len(), 1);
        Ok(())
    }

    #[test]
    fn sliding_mask_applies_during_prefill_and_single_token_decode() -> Result<()> {
        let device = Device::Cpu;
        let model = DecoderModel::load_dense(tiny_spec(), VarBuilder::zeros(DType::F32, &device))?;
        let kind = AttentionKind::SlidingCausal { window: 2 };

        assert!(model.attention_mask(kind, 1, 1)?.is_none());
        assert!(
            model.attention_mask(kind, 1, 2)?.is_none(),
            "the bounded key range physically excludes expired positions"
        );
        let decode = attention_mask_for_range(kind, 2, 1, 0, 3, DType::F32, &device)?
            .expect("an untrimmed key range must mask the expired position")
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(decode[0].is_infinite() && decode[0].is_sign_negative());
        assert_eq!(&decode[1..], &[0.0, 0.0]);

        let prefill = model
            .attention_mask(kind, 3, 0)?
            .expect("multi-token prefill must have a mask")
            .reshape((3, 3))?
            .to_vec2::<f32>()?;
        assert!(prefill[0][0].abs() < f32::EPSILON);
        assert!(prefill[0][1].is_sign_negative());
        assert_eq!(&prefill[1][..2], &[0.0, 0.0]);
        assert!(prefill[1][2].is_sign_negative());
        assert!(prefill[2][0].is_sign_negative());
        assert_eq!(&prefill[2][1..], &[0.0, 0.0]);

        let gemma_window = AttentionKind::SlidingCausal { window: 512 };
        assert!(model.attention_mask(gemma_window, 1, 511)?.is_none());
        assert!(model.attention_mask(gemma_window, 1, 512)?.is_none());
        let at_boundary =
            attention_mask_for_range(gemma_window, 512, 1, 0, 513, DType::F32, &device)?
                .expect("position 512 must reject position zero in an untrimmed range")
                .flatten_all()?
                .to_vec1::<f32>()?;
        assert!(at_boundary[0].is_sign_negative());
        assert!(at_boundary[1..].iter().all(|value| *value == 0.0));
        assert!(model.attention_mask(gemma_window, 1, 513)?.is_none());
        let past_boundary =
            attention_mask_for_range(gemma_window, 513, 1, 0, 514, DType::F32, &device)?
                .expect("position 513 must reject two old positions in an untrimmed range")
                .flatten_all()?
                .to_vec1::<f32>()?;
        assert!(
            past_boundary[..2]
                .iter()
                .all(|value| value.is_sign_negative())
        );
        assert!(past_boundary[2..].iter().all(|value| *value == 0.0));

        let absolute_range = attention_mask_for_range(
            AttentionKind::SlidingCausal { window: 4 },
            6,
            2,
            3,
            5,
            DType::F32,
            &device,
        )?
        .expect("the tile contains both a future key and one expired key")
        .reshape((2, 5))?
        .to_vec2::<f32>()?;
        assert_eq!(&absolute_range[0][..4], &[0.0, 0.0, 0.0, 0.0]);
        assert!(absolute_range[0][4].is_sign_negative());
        assert!(absolute_range[1][0].is_sign_negative());
        assert_eq!(&absolute_range[1][1..], &[0.0, 0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn mixed_sliding_and_full_layers_match_one_shot_and_chunked_prefill() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.runtime.max_context = 32;
        attention_spec_mut(&mut spec, 0).kind = AttentionKind::SlidingCausal { window: 4 };
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let tokens = (1_u32..=11).collect::<Vec<_>>();

        let one_shot = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut one_shot_session = new_session(&one_shot)?;
        let expected = one_shot.forward_tokens(one_shot_session.as_mut(), &tokens, 0)?;
        let stats = cache_stats(one_shot_session.as_ref());
        assert_eq!(stats[0].logical_len, 11);
        assert_eq!(stats[0].resident_start, 8);
        assert_eq!(stats[0].resident_len, 3);
        assert_eq!(stats[0].allocated_capacity, 6);
        assert_eq!(stats[0].physical_capacity, 6);
        assert_eq!(stats[1].logical_len, 11);
        assert_eq!(stats[1].resident_start, 0);
        assert_eq!(stats[1].resident_len, 11);

        let chunked = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut chunked_session = new_session(&chunked)?;
        let mut start = 0;
        let mut actual = None;
        for len in [3, 2, 6] {
            actual = Some(chunked.forward_tokens(
                chunked_session.as_mut(),
                &tokens[start..start + len],
                start,
            )?);
            start += len;
        }
        let difference = max_difference(&actual.expect("last chunk logits"), &expected)?;
        assert!(
            difference < 1e-4,
            "chunked/one-shot difference was {difference}"
        );
        assert_eq!(cache_stats(chunked_session.as_ref()), stats);

        let tokenwise =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut tokenwise_session = new_session(&tokenwise)?;
        let mut tokenwise_logits = None;
        for (position, token) in tokens.iter().copied().enumerate() {
            tokenwise_logits =
                Some(tokenwise.forward_tokens(tokenwise_session.as_mut(), &[token], position)?);
        }
        let difference = max_difference(&tokenwise_logits.expect("last token logits"), &expected)?;
        assert!(
            difference < 1e-4,
            "tokenwise/one-shot difference was {difference}"
        );
        assert_eq!(cache_stats(tokenwise_session.as_ref()), stats);
        Ok(())
    }

    #[test]
    fn gemma2_sandwich_sliding_window_is_observable_across_cache_boundary() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.architecture = "gemma2".into();
        spec.runtime.max_context = 32;
        spec.embedding_scale = (spec.hidden_size as f64).sqrt();
        spec.final_logit_softcap = Some(30.0);
        for (index, layer) in spec.layer_specs.iter_mut().enumerate() {
            layer.activation = Activation::GeluTanh;
            layer.norm_layout = NormLayout::Sandwich;
            let attention = layer
                .mixer
                .attention_mut()
                .expect("tiny Gemma 2 layer uses attention");
            attention.kind = if index == 0 {
                AttentionKind::SlidingCausal { window: 4 }
            } else {
                AttentionKind::FullCausal
            };
            attention.logit_softcap = Some(50.0);
        }
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let tokens = (1_u32..=11).collect::<Vec<_>>();

        let one_shot = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut one_shot_session = new_session(&one_shot)?;
        let expected = one_shot.forward_tokens(one_shot_session.as_mut(), &tokens, 0)?;
        let stats = cache_stats(one_shot_session.as_ref());
        assert_eq!(stats[0].resident_start, 8);
        assert_eq!(stats[0].resident_len, 3);
        assert_eq!(stats[1].resident_start, 0);
        assert_eq!(stats[1].resident_len, 11);

        let chunked = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut chunked_session = new_session(&chunked)?;
        let mut start = 0;
        let mut actual = None;
        for len in [3, 2, 6] {
            actual = Some(chunked.forward_tokens(
                chunked_session.as_mut(),
                &tokens[start..start + len],
                start,
            )?);
            start += len;
        }
        let difference = max_difference(&actual.expect("last chunk logits"), &expected)?;
        assert!(difference < 1e-4, "Gemma 2 chunked drift was {difference}");
        assert_eq!(cache_stats(chunked_session.as_ref()), stats);

        let mut full_spec = spec;
        attention_spec_mut(&mut full_spec, 0).kind = AttentionKind::FullCausal;
        let full = DecoderModel::load_dense(
            full_spec,
            VarBuilder::from_tensors(tensors, DType::F32, &device),
        )?;
        let mut full_session = new_session(&full)?;
        let full_logits = full.forward_tokens(full_session.as_mut(), &tokens, 0)?;
        let difference = max_difference(&full_logits, &expected)?;
        assert!(
            difference > 1e-7,
            "the tiny Gemma 2 fixture did not make the window observable"
        );
        Ok(())
    }

    #[test]
    fn long_sliding_prefill_tiles_queries_and_keeps_storage_bounded() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.runtime.max_context = 300;
        spec.num_layers = 1;
        spec.layer_specs.truncate(1);
        attention_spec_mut(&mut spec, 0).kind = AttentionKind::SlidingCausal { window: 4 };
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let tokens = (0..257)
            .map(|index| (index % (spec.runtime.vocab_size - 1) + 1) as u32)
            .collect::<Vec<_>>();

        let one_shot = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut one_shot_session = new_session(&one_shot)?;
        let expected = one_shot.forward_tokens(one_shot_session.as_mut(), &tokens, 0)?;
        let stats = cache_stats(one_shot_session.as_ref());
        assert_eq!(stats[0].logical_len, 257);
        assert_eq!(stats[0].resident_start, 254);
        assert_eq!(stats[0].resident_len, 3);
        assert_eq!(stats[0].allocated_capacity, 6);

        let chunked =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut chunked_session = new_session(&chunked)?;
        let mut start = 0;
        let mut actual = None;
        for len in [3, 2, 127, 125] {
            actual = Some(chunked.forward_tokens(
                chunked_session.as_mut(),
                &tokens[start..start + len],
                start,
            )?);
            start += len;
        }
        assert_eq!(start, tokens.len());
        let difference = max_difference(&actual.expect("last chunk logits"), &expected)?;
        assert!(
            difference < 1e-4,
            "long chunked/one-shot difference was {difference}"
        );
        assert_eq!(cache_stats(chunked_session.as_ref()), stats);
        Ok(())
    }

    #[test]
    fn window_two_model_and_generation_keep_absolute_cache_progress() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.runtime.max_context = 32;
        spec.num_layers = 1;
        spec.layer_specs.truncate(1);
        attention_spec_mut(&mut spec, 0).kind = AttentionKind::SlidingCausal { window: 2 };
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let tokens = (1_u32..=7).collect::<Vec<_>>();

        let one_shot = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut one_shot_session = new_session(&one_shot)?;
        let expected = one_shot.forward_tokens(one_shot_session.as_mut(), &tokens, 0)?;
        let stats = cache_stats(one_shot_session.as_ref());
        assert_eq!(stats[0].logical_len, 7);
        assert_eq!(stats[0].resident_start, 6);
        assert_eq!(stats[0].resident_len, 1);
        assert_eq!(stats[0].allocated_capacity, 2);

        let tokenwise = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut tokenwise_session = new_session(&tokenwise)?;
        let mut actual = None;
        for (position, token) in tokens.iter().copied().enumerate() {
            actual =
                Some(tokenwise.forward_tokens(tokenwise_session.as_mut(), &[token], position)?);
        }
        let difference = max_difference(&actual.expect("last token logits"), &expected)?;
        assert!(
            difference < 1e-4,
            "W=2 tokenwise/one-shot difference was {difference}"
        );

        let generated_model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut generated_session = new_session(&generated_model)?;
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: 0.0,
            repetition_penalty: 1.0,
            seed: 42,
            ..SamplingConfig::default()
        })?;
        let mut generated = Vec::new();
        let metrics = generate(
            &generated_model,
            generated_session.as_mut(),
            &[1, 2, 3],
            &mut sampler,
            GenerationConfig {
                max_new_tokens: 5,
                ignore_stop_tokens: true,
            },
            |token| {
                generated.push(token);
                Ok(())
            },
        )?;
        assert_eq!(generated.len(), 5);
        assert_eq!(metrics.decode_steps, 4);
        assert_eq!(generated_session.cache_len(), 7);
        let stats = cache_stats(generated_session.as_ref());
        assert_eq!(stats[0].logical_len, 7);
        assert_eq!(stats[0].resident_len, 1);
        assert_eq!(stats[0].allocated_capacity, 2);
        Ok(())
    }

    #[test]
    #[allow(clippy::too_many_lines)] // One scenario locks wrap, cleanup, allocation reuse and fresh parity.
    fn cancelled_request_clears_wrapped_mixed_cache_for_fresh_reuse() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.runtime.max_context = 32;
        attention_spec_mut(&mut spec, 0).kind = AttentionKind::SlidingCausal { window: 4 };
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut session = new_session(&model)?;
        let cancelled = Cell::new(false);
        let visible_tokens = Cell::new(0_usize);
        let metrics = generate_with_control(
            &model,
            session.as_mut(),
            &[1, 3, 4, 5, 6],
            &mut Sampler::new(SamplingConfig {
                temperature: 0.0,
                repetition_penalty: 1.0,
                seed: 42,
                ..SamplingConfig::default()
            })?,
            GenerationConfig {
                max_new_tokens: 6,
                ignore_stop_tokens: true,
            },
            || cancelled.get(),
            |_| {
                let count = visible_tokens.get() + 1;
                visible_tokens.set(count);
                if count == 5 {
                    cancelled.set(true);
                }
                Ok(())
            },
        )?;
        assert_eq!(metrics.finish_reason, FinishReason::Cancelled);
        assert_eq!(metrics.generated_tokens, 5);
        assert_eq!(metrics.decode_steps, 4);
        assert_eq!(session.cache_len(), 9);
        let state = decoder_session(session.as_ref());
        let LayerSequenceState::SlidingAttention(sliding) = &state.layer_states[0] else {
            panic!("first layer must use sliding attention");
        };
        assert_eq!(sliding.history_segments()?.len(), 2);
        assert_eq!(
            state
                .full_table
                .as_ref()
                .map(PagedKvBlockTable::logical_len),
            Some(9)
        );

        session.clear_cache();
        let cleared = cache_stats(session.as_ref());
        assert!(
            cleared
                .iter()
                .all(|stats| stats.logical_len == 0 && stats.resident_len == 0)
        );
        assert!(cleared[0].allocated_capacity > 0);
        assert_eq!(cleared[1].allocated_capacity, 0);
        let allocated = cleared
            .iter()
            .map(|stats| stats.allocated_capacity)
            .collect::<Vec<_>>();

        cancelled.set(false);
        visible_tokens.set(0);
        let metrics = run_generation_request(
            &model,
            session.as_mut(),
            &[1, 3, 4, 5, 6],
            &mut Sampler::new(SamplingConfig {
                temperature: 0.0,
                repetition_penalty: 1.0,
                seed: 42,
                ..SamplingConfig::default()
            })?,
            GenerationConfig {
                max_new_tokens: 6,
                ignore_stop_tokens: true,
            },
            || cancelled.get(),
            |_| {
                let count = visible_tokens.get() + 1;
                visible_tokens.set(count);
                if count == 5 {
                    cancelled.set(true);
                }
                Ok(())
            },
        )?;
        assert_eq!(metrics.finish_reason, FinishReason::Cancelled);
        assert_eq!(metrics.generated_tokens, 5);
        assert_eq!(metrics.decode_steps, 4);
        assert!(cache_stats(session.as_ref()).iter().zip(allocated).all(
            |(stats, allocated_capacity)| {
                stats.logical_len == 0
                    && stats.resident_len == 0
                    && stats.allocated_capacity == allocated_capacity
            }
        ));

        let prompt = [7_u32, 8, 9];
        let config = GenerationConfig {
            max_new_tokens: 5,
            ignore_stop_tokens: true,
        };
        let mut reused_tokens = Vec::new();
        let _ = run_generation_request(
            &model,
            session.as_mut(),
            &prompt,
            &mut Sampler::new(SamplingConfig {
                temperature: 0.0,
                repetition_penalty: 1.0,
                seed: 7,
                ..SamplingConfig::default()
            })?,
            config,
            || false,
            |token| {
                reused_tokens.push(token);
                Ok(())
            },
        )?;

        let fresh =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut fresh_session = new_session(&fresh)?;
        let mut fresh_tokens = Vec::new();
        let _ = run_generation_request(
            &fresh,
            fresh_session.as_mut(),
            &prompt,
            &mut Sampler::new(SamplingConfig {
                temperature: 0.0,
                repetition_penalty: 1.0,
                seed: 7,
                ..SamplingConfig::default()
            })?,
            config,
            || false,
            |token| {
                fresh_tokens.push(token);
                Ok(())
            },
        )?;
        assert_eq!(reused_tokens, fresh_tokens);
        Ok(())
    }

    #[test]
    fn failed_forward_after_ring_wrap_restores_contents_for_retry() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.runtime.max_context = 32;
        for layer in &mut spec.layer_specs {
            layer
                .attention_mut()
                .expect("test layer uses attention")
                .kind = AttentionKind::SlidingCausal { window: 4 };
        }
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let prefix = [1_u32, 2, 3, 4, 5];
        let retry = [6_u32, 7, 8];

        let control = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut control_session = new_session(&control)?;
        let _ = control.forward_tokens(control_session.as_mut(), &prefix[..3], 0)?;
        let _ = control.forward_tokens(control_session.as_mut(), &prefix[3..], 3)?;
        let expected = control.forward_tokens(control_session.as_mut(), &retry, prefix.len())?;

        let mut model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut session = new_session(&model)?;
        let _ = model.forward_tokens(session.as_mut(), &prefix[..3], 0)?;
        let _ = model.forward_tokens(session.as_mut(), &prefix[3..], 3)?;
        let mut peer_session = new_session(&model)?;
        let _ = model.forward_tokens(peer_session.as_mut(), &[9, 10], 0)?;
        let peer_checkpoint_stats = cache_stats(peer_session.as_ref());
        let checkpoint_stats = cache_stats(session.as_ref());
        assert!(checkpoint_stats.iter().all(|stats| stats.resident_len == 3));
        let output = layer_attention(&model, 1).output.clone();
        layer_attention_mut(&mut model, 1).output = LinearOp::Dense(candle_nn::Linear::new(
            Tensor::zeros((1, 1), DType::F32, &device)?,
            None,
        ));
        assert!(
            model
                .forward_tokens(session.as_mut(), &retry, prefix.len())
                .is_err()
        );
        assert_eq!(session.cache_len(), prefix.len());
        assert_eq!(cache_stats(session.as_ref()), checkpoint_stats);
        assert_eq!(cache_stats(peer_session.as_ref()), peer_checkpoint_stats);

        layer_attention_mut(&mut model, 1).output = output;
        let peer_logits = model.forward_tokens(peer_session.as_mut(), &[11], 2)?;
        assert_eq!(peer_logits.dims2()?, (1, 32));
        assert_eq!(peer_session.cache_len(), 3);
        let actual = model.forward_tokens(session.as_mut(), &retry, prefix.len())?;
        let difference = max_difference(&actual, &expected)?;
        assert!(
            difference < 1e-4,
            "retry/control difference was {difference}"
        );
        assert_eq!(
            cache_stats(session.as_ref()),
            cache_stats(control_session.as_ref())
        );

        // The successful retry leaves live slots split across the ring tail
        // and head. Clearing that state must behave like a fresh model while
        // retaining the six allocated physical slots for reuse.
        session.clear_cache();
        assert!(cache_stats(session.as_ref()).iter().all(|stats| {
            stats.logical_len == 0 && stats.resident_len == 0 && stats.allocated_capacity == 6
        }));
        let restart = [11_u32, 12];
        let restarted = model.forward_tokens(session.as_mut(), &restart, 0)?;
        let fresh =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut fresh_session = new_session(&fresh)?;
        let fresh_logits = fresh.forward_tokens(fresh_session.as_mut(), &restart, 0)?;
        let difference = max_difference(&restarted, &fresh_logits)?;
        assert!(difference < 1e-4, "clear/fresh difference was {difference}");
        Ok(())
    }

    #[test]
    fn attention_width_can_differ_from_residual_width() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.num_layers = 1;
        spec.head_dim = 6;
        spec.layer_specs.truncate(1);
        attention_spec_mut(&mut spec, 0).scale = 1.0 / 6.0_f64.sqrt();
        let model = DecoderModel::load_dense(spec, VarBuilder::zeros(DType::F32, &device))?;
        let mut session = new_session(&model)?;
        let logits = model.forward(session.as_mut(), &Tensor::new(&[[1_u32, 2]], &device)?, 0)?;
        assert_eq!(logits.dims2()?, (1, 32));
        Ok(())
    }

    #[test]
    fn nope_attention_skips_rotary_state_and_preserves_cache_parity() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.layer_specs[1]
            .attention_mut()
            .expect("test layer uses attention")
            .position_encoding = PositionEncodingSpec::None;
        let tensors = patterned_tiny_tensors(&spec, &device)?;

        let cached_model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        assert!(layer_attention(&cached_model, 0).rotary.is_some());
        assert!(layer_attention(&cached_model, 1).rotary.is_none());
        let mut cached_session = new_session(&cached_model)?;
        let _ = cached_model.forward_tokens(cached_session.as_mut(), &[1, 2, 3], 0)?;
        let cached = cached_model.forward_tokens(cached_session.as_mut(), &[4], 3)?;

        let full_model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut full_session = new_session(&full_model)?;
        let full = full_model.forward_tokens(full_session.as_mut(), &[1, 2, 3, 4], 0)?;
        let difference = max_difference(&cached, &full)?;
        assert!(
            difference < 1e-5,
            "NoPE cache/full difference was {difference}"
        );
        Ok(())
    }

    #[test]
    fn per_head_qk_norm_keeps_prefill_and_decode_on_the_same_semantics() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.num_layers = 1;
        spec.layer_specs.truncate(1);
        attention_spec_mut(&mut spec, 0).qk_norm = true;
        let tensors = patterned_tiny_tensors(&spec, &device)?;

        let cached_model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        )?;
        let mut cached_session = new_session(&cached_model)?;
        let _ = cached_model.forward_tokens(cached_session.as_mut(), &[1, 2, 3], 0)?;
        let cached = cached_model.forward_tokens(cached_session.as_mut(), &[4], 3)?;

        let full_model =
            DecoderModel::load_dense(spec, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let mut full_session = new_session(&full_model)?;
        let full = full_model.forward_tokens(full_session.as_mut(), &[1, 2, 3, 4], 0)?;
        let difference = max_difference(&cached, &full)?;
        assert!(
            difference < 1e-5,
            "Q/K-normalized cache/full difference was {difference}"
        );
        Ok(())
    }

    #[test]
    fn per_head_qk_norm_is_applied_before_rope() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.num_layers = 1;
        spec.num_kv_heads = 2;
        spec.layer_specs.truncate(1);
        attention_spec_mut(&mut spec, 0).qk_norm = true;
        let tensors = patterned_tiny_tensors(&spec, &device)?;
        let model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(tensors, DType::F32, &device),
        )?;
        let input = patterned_tensor(&[1, 2, spec.hidden_size], 99, 0.1, &device)?;

        let (expected, norm_after_rope) = {
            let attention = layer_attention(&model, 0);
            let (batch, sequence, _) = input.dims3()?;
            let query = attention.query.forward(&input)?.reshape((
                batch,
                sequence,
                attention.num_query_heads,
                attention.head_dim,
            ))?;
            let key = attention.key.forward(&input)?.reshape((
                batch,
                sequence,
                attention.num_kv_heads,
                attention.head_dim,
            ))?;
            let value = attention
                .value
                .forward(&input)?
                .reshape((batch, sequence, attention.num_kv_heads, attention.head_dim))?
                .transpose(1, 2)?;
            let query_norm = attention.query_norm.as_ref().expect("Q norm is loaded");
            let key_norm = attention.key_norm.as_ref().expect("K norm is loaded");

            let normalized_query = query
                .flatten(0, 2)?
                .apply(query_norm)?
                .reshape((
                    batch,
                    sequence,
                    attention.num_query_heads,
                    attention.head_dim,
                ))?
                .transpose(1, 2)?;
            let normalized_key = key
                .flatten(0, 2)?
                .apply(key_norm)?
                .reshape((batch, sequence, attention.num_kv_heads, attention.head_dim))?
                .transpose(1, 2)?;
            let rotary = attention.rotary.as_ref().expect("test attention uses RoPE");
            let (normalized_query, normalized_key) =
                rotary.apply(&normalized_query, &normalized_key, 0)?;
            let expected = attention
                .attention_values(&normalized_query, &normalized_key, &value, None)?
                .transpose(1, 2)?
                .reshape((batch, sequence, attention.attention_width))?
                .apply(&attention.output)?;

            let (rotated_query, rotated_key) =
                rotary.apply(&query.transpose(1, 2)?, &key.transpose(1, 2)?, 0)?;
            let rotated_query = rotated_query.flatten(0, 2)?.apply(query_norm)?.reshape((
                batch,
                attention.num_query_heads,
                sequence,
                attention.head_dim,
            ))?;
            let rotated_key = rotated_key.flatten(0, 2)?.apply(key_norm)?.reshape((
                batch,
                attention.num_kv_heads,
                sequence,
                attention.head_dim,
            ))?;
            let norm_after_rope = attention
                .attention_values(&rotated_query, &rotated_key, &value, None)?
                .transpose(1, 2)?
                .reshape((batch, sequence, attention.attention_width))?
                .apply(&attention.output)?;
            (expected, norm_after_rope)
        };

        let mut cache = LayerKvCache::Full(KvCache::new(spec.runtime.max_context));
        let actual = layer_attention(&model, 0).forward(&mut cache, &input, None, 0)?;
        assert!(max_difference(&actual, &expected)? < 1e-5);
        assert!(
            max_difference(&actual, &norm_after_rope)? > 1e-4,
            "fixture must distinguish Q/K norm before versus after RoPE"
        );
        Ok(())
    }

    #[test]
    fn nonstandard_attention_width_and_nonzero_biases_match_golden() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.runtime.vocab_size = 2;
        spec.runtime.max_context = 4;
        spec.runtime.stop_token_ids = vec![1];
        spec.hidden_size = 2;
        spec.intermediate_size = 2;
        spec.num_layers = 1;
        spec.num_query_heads = 2;
        spec.num_kv_heads = 1;
        spec.head_dim = 2;
        spec.layer_specs.truncate(1);
        let attention = attention_spec_mut(&mut spec, 0);
        attention.scale = 1.0 / 2.0_f64.sqrt();
        attention
            .rope_mut()
            .expect("test attention uses RoPE")
            .rotary_dim = 2;
        spec.projection_bias = ProjectionBias {
            query: true,
            key: true,
            value: true,
            output: true,
        };
        spec.mlp_bias = true;
        spec.tied_embeddings = false;

        let model = DecoderModel::load_dense(
            spec.clone(),
            VarBuilder::from_tensors(biased_golden_tensors(&device)?, DType::F32, &device),
        )?;
        let mut session = new_session(&model)?;
        let logits = model
            .forward(session.as_mut(), &Tensor::new(&[[0_u32]], &device)?, 0)?
            .to_vec2::<f32>()?
            .remove(0);

        let silu = |value: f32| value / (1.0 + (-value).exp());
        let hidden = [1.5 + silu(0.5) * 2.0 + 0.25, -0.75 + silu(-0.5) * 3.0 + 0.5];
        let rms = (hidden[0].powi(2).midpoint(hidden[1].powi(2)) + spec.norm_epsilon as f32).sqrt();
        let expected = [hidden[0] / rms, hidden[1] / rms];
        for (actual, expected) in logits.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn failed_layer_rolls_every_cache_back() -> Result<()> {
        let device = Device::Cpu;
        let weights = VarBuilder::zeros(DType::F32, &device);
        let mut model = DecoderModel::load_dense(tiny_spec(), weights)?;
        // Force a shape error after the second layer has appended its K/V.
        let invalid_weight = Tensor::zeros((1, 1), DType::F32, &device)?;
        layer_attention_mut(&mut model, 1).output =
            LinearOp::Dense(candle_nn::Linear::new(invalid_weight, None));
        let mut session = new_session(&model)?;
        let prompt = Tensor::new(&[[1_u32, 2, 3]], &device)?;
        assert!(model.forward(session.as_mut(), &prompt, 0).is_err());
        assert_eq!(session.cache_len(), 0);
        assert!(
            decoder_session(session.as_ref())
                .layer_states
                .iter()
                .all(|state| state.logical_len(Some(0)).is_ok_and(|length| length == 0))
        );
        assert_eq!(
            decoder_session(session.as_ref())
                .full_table
                .as_ref()
                .map(PagedKvBlockTable::logical_len),
            Some(0)
        );
        Ok(())
    }

    #[test]
    fn untied_lm_head_and_mlp_biases_load() -> Result<()> {
        let device = Device::Cpu;
        let mut spec = tiny_spec();
        spec.runtime.vocab_size = 4;
        spec.runtime.max_context = 4;
        spec.runtime.stop_token_ids = vec![2];
        spec.hidden_size = 4;
        spec.intermediate_size = 8;
        spec.num_layers = 1;
        spec.num_query_heads = 1;
        spec.num_kv_heads = 1;
        spec.head_dim = 4;
        spec.layer_specs.truncate(1);
        attention_spec_mut(&mut spec, 0).scale = 0.5;
        spec.projection_bias = ProjectionBias {
            query: false,
            key: false,
            value: false,
            output: false,
        };
        spec.mlp_bias = true;
        spec.tied_embeddings = false;

        let mut tensors = HashMap::new();
        let mut embedding = vec![0.0_f32; 16];
        embedding[4] = 1.0; // token 1 embeds along hidden dimension 0.
        tensors.insert(
            "model.embed_tokens.weight".into(),
            Tensor::from_vec(embedding, (4, 4), &device)?,
        );
        tensors.insert(
            "model.norm.weight".into(),
            Tensor::ones(4, DType::F32, &device)?,
        );
        for name in ["input_layernorm", "post_attention_layernorm"] {
            tensors.insert(
                format!("model.layers.0.{name}.weight"),
                Tensor::ones(4, DType::F32, &device)?,
            );
        }
        for (name, shape) in [
            ("self_attn.q_proj", (4, 4)),
            ("self_attn.k_proj", (4, 4)),
            ("self_attn.v_proj", (4, 4)),
            ("self_attn.o_proj", (4, 4)),
            ("mlp.gate_proj", (8, 4)),
            ("mlp.up_proj", (8, 4)),
            ("mlp.down_proj", (4, 8)),
        ] {
            tensors.insert(
                format!("model.layers.0.{name}.weight"),
                Tensor::zeros(shape, DType::F32, &device)?,
            );
        }
        for (name, length) in [
            ("mlp.gate_proj.bias", 8),
            ("mlp.up_proj.bias", 8),
            ("mlp.down_proj.bias", 4),
        ] {
            tensors.insert(
                format!("model.layers.0.{name}"),
                Tensor::zeros(length, DType::F32, &device)?,
            );
        }
        let mut lm_head = vec![0.0_f32; 16];
        lm_head[12] = 1.0; // independent head maps hidden dimension 0 to token 3.
        tensors.insert(
            "lm_head.weight".into(),
            Tensor::from_vec(lm_head, (4, 4), &device)?,
        );
        let weights = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = DecoderModel::load_dense(spec, weights)?;
        let mut session = new_session(&model)?;
        let input = Tensor::new(&[[1_u32]], &device)?;
        let logits = model
            .forward(session.as_mut(), &input, 0)?
            .to_vec2::<f32>()?
            .remove(0);
        let predicted = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map_or(0, |(index, _)| index);
        assert_eq!(predicted, 3);
        Ok(())
    }
}
