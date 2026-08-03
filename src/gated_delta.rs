use candle_core::{CpuStorage, CustomOp3, DType, Device, Layout, Result, Shape, Storage, Tensor};

/// Compute `exp(softplus(alpha + time_bias) * state_decay)` with one backend
/// operation.
///
/// Gated recurrent models evaluate this small control tensor for every layer
/// and decode token. Expressing the stable softplus formula as primitive
/// Candle operations launches many tiny Metal kernels, so this operation keeps
/// the architecture contract unchanged while giving CPU and Metal a compact
/// execution boundary. Unsupported layouts return `None` for the readable
/// tensor fallback.
pub(crate) fn decay_factors(
    alpha: &Tensor,
    time_bias: &Tensor,
    state_decay: &Tensor,
) -> Result<Option<Tensor>> {
    let (tokens, heads) = alpha.dims2()?;
    if tokens == 0 || heads == 0 {
        candle_core::bail!("Gated DeltaNet decay requires non-empty token and head dimensions");
    }
    if time_bias.dims1()? != heads || state_decay.dims1()? != heads {
        candle_core::bail!(
            "Gated DeltaNet decay has alpha shape {:?}, time bias {:?} and state decay {:?}",
            alpha.dims(),
            time_bias.dims(),
            state_decay.dims()
        );
    }
    if alpha.device().location() != time_bias.device().location()
        || alpha.device().location() != state_decay.device().location()
    {
        candle_core::bail!("Gated DeltaNet decay tensors are on different devices");
    }
    if alpha.dtype() != DType::F32
        || time_bias.dtype() != DType::F32
        || state_decay.dtype() != DType::F32
        || !alpha.is_contiguous()
        || !time_bias.is_contiguous()
        || !state_decay.is_contiguous()
    {
        return Ok(None);
    }

    let supported = match alpha.device() {
        // Keep primitive Candle operations as the CPU numerical oracle.
        Device::Cpu => false,
        #[cfg(feature = "metal")]
        Device::Metal(device) => metal::device_supported(device),
        #[allow(unreachable_patterns)]
        _ => false,
    };
    if !supported {
        return Ok(None);
    }

    alpha
        .apply_op3_no_bwd(time_bias, state_decay, &GatedDeltaDecayOp { heads })
        .map(Some)
}

struct GatedDeltaDecayOp {
    heads: usize,
}

impl CustomOp3 for GatedDeltaDecayOp {
    fn name(&self) -> &'static str {
        "gated-delta-decay"
    }

    fn cpu_fwd(
        &self,
        alpha: &CpuStorage,
        alpha_layout: &Layout,
        time_bias: &CpuStorage,
        time_bias_layout: &Layout,
        state_decay: &CpuStorage,
        state_decay_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(alpha) = alpha else {
            candle_core::bail!("CPU Gated DeltaNet decay requires F32 alpha");
        };
        let CpuStorage::F32(time_bias) = time_bias else {
            candle_core::bail!("CPU Gated DeltaNet decay requires F32 time bias");
        };
        let CpuStorage::F32(state_decay) = state_decay else {
            candle_core::bail!("CPU Gated DeltaNet decay requires F32 state decay");
        };
        if !alpha_layout.is_contiguous()
            || !time_bias_layout.is_contiguous()
            || !state_decay_layout.is_contiguous()
        {
            candle_core::bail!("CPU Gated DeltaNet decay requires contiguous inputs");
        }

        let count = alpha_layout.shape().elem_count();
        let alpha_start = alpha_layout.start_offset();
        let bias_start = time_bias_layout.start_offset();
        let decay_start = state_decay_layout.start_offset();
        let mut output = Vec::with_capacity(count);
        for index in 0..count {
            let head = index % self.heads;
            let shifted = alpha[alpha_start + index] + time_bias[bias_start + head];
            let softplus = shifted.max(0.0) + (-shifted.abs()).exp().ln_1p();
            output.push((softplus * state_decay[decay_start + head]).exp());
        }
        Ok((CpuStorage::F32(output), alpha_layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        alpha: &candle_core::MetalStorage,
        alpha_layout: &Layout,
        time_bias: &candle_core::MetalStorage,
        time_bias_layout: &Layout,
        state_decay: &candle_core::MetalStorage,
        state_decay_layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::forward_decay(
            self,
            alpha,
            alpha_layout,
            time_bias,
            time_bias_layout,
            state_decay,
            state_decay_layout,
        )
    }
}

/// Apply one autoregressive Gated `DeltaNet` recurrence step.
///
/// The operation returns a new state rather than mutating the input so the
/// decoder's handle-based transaction checkpoints remain valid if a later
/// layer fails. Query and key are already normalized, query already includes
/// its scale, and `decay`/`beta` are already transformed controls.
pub(crate) fn decode_recurrence(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    recurrent: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let (heads, key_dim) = query.dims2()?;
    if heads == 0 || key_dim == 0 || key.dims2()? != (heads, key_dim) {
        candle_core::bail!("Gated DeltaNet decode has incompatible query/key geometry");
    }
    let (value_heads, value_dim) = value.dims2()?;
    if value_heads != heads
        || value_dim == 0
        || decay.dims1()? != heads
        || beta.dims1()? != heads
        || recurrent.dims3()? != (heads, key_dim, value_dim)
    {
        candle_core::bail!(
            "Gated DeltaNet decode has query {:?}, key {:?}, value {:?}, decay {:?}, beta {:?} and state {:?}",
            query.dims(),
            key.dims(),
            value.dims(),
            decay.dims(),
            beta.dims(),
            recurrent.dims()
        );
    }
    for tensor in [key, value, decay, beta, recurrent] {
        if tensor.device().location() != query.device().location() {
            candle_core::bail!("Gated DeltaNet decode tensors are on different devices");
        }
    }
    if [query, key, value, decay, beta, recurrent]
        .iter()
        .any(|tensor| tensor.dtype() != DType::F32 || !tensor.is_contiguous())
    {
        return Ok(None);
    }

    let supported = match query.device() {
        // Preserve the tensor implementation as the CPU numerical reference;
        // the fused loop uses a different accumulation order that can amplify
        // through a long recurrent checkpoint even when one step is close.
        Device::Cpu => false,
        #[cfg(feature = "metal")]
        Device::Metal(device) => metal::device_supported(device),
        #[allow(unreachable_patterns)]
        _ => false,
    };
    if !supported {
        return Ok(None);
    }

    let state_elements = heads
        .checked_mul(key_dim)
        .and_then(|elements| elements.checked_mul(value_dim))
        .ok_or_else(|| candle_core::Error::Msg("Gated DeltaNet state size overflow".into()))?;
    let output_elements = heads
        .checked_mul(value_dim)
        .ok_or_else(|| candle_core::Error::Msg("Gated DeltaNet output size overflow".into()))?;
    let combined_elements = state_elements
        .checked_add(output_elements)
        .ok_or_else(|| candle_core::Error::Msg("Gated DeltaNet result size overflow".into()))?;
    let combined = query.apply_op3_no_bwd(
        key,
        recurrent,
        &GatedDeltaRecurrenceOp {
            value,
            decay,
            beta,
            tokens: 1,
            heads,
            key_dim,
            value_dim,
            state_elements,
            combined_elements,
        },
    )?;
    let next_state = combined
        .narrow(0, 0, state_elements)?
        .reshape((heads, key_dim, value_dim))?;
    let output = combined
        .narrow(0, state_elements, output_elements)?
        .reshape((heads, value_dim))?;
    Ok(Some((output, next_state)))
}

/// Apply a complete prefill lane of the autoregressive Gated `DeltaNet`
/// recurrence with one backend operation.
///
/// The fused backend still walks tokens in causal order, but it keeps the
/// evolving state inside one result buffer instead of materializing several
/// full recurrent-state tensors per token. The input state remains immutable,
/// preserving the decoder transaction's rollback semantics.
pub(crate) fn prefill_recurrence(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    recurrent: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let (tokens, heads, key_dim) = query.dims3()?;
    if tokens == 0 || heads == 0 || key_dim == 0 || key.dims3()? != (tokens, heads, key_dim) {
        candle_core::bail!("Gated DeltaNet prefill has incompatible query/key geometry");
    }
    let (value_tokens, value_heads, value_dim) = value.dims3()?;
    if value_tokens != tokens
        || value_heads != heads
        || value_dim == 0
        || decay.dims2()? != (tokens, heads)
        || beta.dims2()? != (tokens, heads)
        || recurrent.dims3()? != (heads, key_dim, value_dim)
    {
        candle_core::bail!(
            "Gated DeltaNet prefill has query {:?}, key {:?}, value {:?}, decay {:?}, beta {:?} and state {:?}",
            query.dims(),
            key.dims(),
            value.dims(),
            decay.dims(),
            beta.dims(),
            recurrent.dims()
        );
    }
    for tensor in [key, value, decay, beta, recurrent] {
        if tensor.device().location() != query.device().location() {
            candle_core::bail!("Gated DeltaNet prefill tensors are on different devices");
        }
    }
    if [query, key, value, decay, beta, recurrent]
        .iter()
        .any(|tensor| tensor.dtype() != DType::F32 || !tensor.is_contiguous())
    {
        return Ok(None);
    }

    let supported = match query.device() {
        // Keep the primitive tensor loop as the CPU numerical oracle.
        Device::Cpu => false,
        #[cfg(feature = "metal")]
        Device::Metal(device) => metal::device_supported(device),
        #[allow(unreachable_patterns)]
        _ => false,
    };
    if !supported {
        return Ok(None);
    }

    let state_elements = heads
        .checked_mul(key_dim)
        .and_then(|elements| elements.checked_mul(value_dim))
        .ok_or_else(|| candle_core::Error::Msg("Gated DeltaNet state size overflow".into()))?;
    let output_elements = tokens
        .checked_mul(heads)
        .and_then(|elements| elements.checked_mul(value_dim))
        .ok_or_else(|| candle_core::Error::Msg("Gated DeltaNet output size overflow".into()))?;
    let combined_elements = state_elements
        .checked_add(output_elements)
        .ok_or_else(|| candle_core::Error::Msg("Gated DeltaNet result size overflow".into()))?;
    let combined = query.apply_op3_no_bwd(
        key,
        recurrent,
        &GatedDeltaRecurrenceOp {
            value,
            decay,
            beta,
            tokens,
            heads,
            key_dim,
            value_dim,
            state_elements,
            combined_elements,
        },
    )?;
    let next_state = combined
        .narrow(0, 0, state_elements)?
        .reshape((heads, key_dim, value_dim))?;
    let output = combined
        .narrow(0, state_elements, output_elements)?
        .reshape((tokens, heads, value_dim))?;
    Ok(Some((output, next_state)))
}

struct GatedDeltaRecurrenceOp<'a> {
    value: &'a Tensor,
    decay: &'a Tensor,
    beta: &'a Tensor,
    tokens: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
    state_elements: usize,
    combined_elements: usize,
}

impl CustomOp3 for GatedDeltaRecurrenceOp<'_> {
    fn name(&self) -> &'static str {
        "gated-delta-decode-recurrence"
    }

    #[allow(clippy::too_many_lines)]
    fn cpu_fwd(
        &self,
        query: &CpuStorage,
        query_layout: &Layout,
        key: &CpuStorage,
        key_layout: &Layout,
        recurrent: &CpuStorage,
        recurrent_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(query) = query else {
            candle_core::bail!("CPU Gated DeltaNet decode requires F32 query");
        };
        let CpuStorage::F32(key) = key else {
            candle_core::bail!("CPU Gated DeltaNet decode requires F32 key");
        };
        let CpuStorage::F32(recurrent) = recurrent else {
            candle_core::bail!("CPU Gated DeltaNet decode requires F32 state");
        };
        if !query_layout.is_contiguous()
            || !key_layout.is_contiguous()
            || !recurrent_layout.is_contiguous()
        {
            candle_core::bail!("CPU Gated DeltaNet decode requires contiguous inputs");
        }

        let (value_storage, value_layout) = self.value.storage_and_layout();
        let (decay_storage, decay_layout) = self.decay.storage_and_layout();
        let (beta_storage, beta_layout) = self.beta.storage_and_layout();
        let Storage::Cpu(CpuStorage::F32(value)) = &*value_storage else {
            candle_core::bail!("CPU Gated DeltaNet decode requires F32 value");
        };
        let Storage::Cpu(CpuStorage::F32(decay)) = &*decay_storage else {
            candle_core::bail!("CPU Gated DeltaNet decode requires F32 decay");
        };
        let Storage::Cpu(CpuStorage::F32(beta)) = &*beta_storage else {
            candle_core::bail!("CPU Gated DeltaNet decode requires F32 beta");
        };
        if !value_layout.is_contiguous()
            || !decay_layout.is_contiguous()
            || !beta_layout.is_contiguous()
        {
            candle_core::bail!("CPU Gated DeltaNet decode requires contiguous controls");
        }

        let query_start = query_layout.start_offset();
        let key_start = key_layout.start_offset();
        let value_start = value_layout.start_offset();
        let decay_start = decay_layout.start_offset();
        let beta_start = beta_layout.start_offset();
        let recurrent_start = recurrent_layout.start_offset();
        let mut output = vec![0.0_f32; self.combined_elements];
        for head in 0..self.heads {
            let head_state = head * self.key_dim * self.value_dim;
            for value_index in 0..self.value_dim {
                for token in 0..self.tokens {
                    let token_head = token * self.heads + head;
                    let head_key = token_head * self.key_dim;
                    let head_value = token_head * self.value_dim;
                    let head_decay = decay[decay_start + token_head];
                    let head_beta = beta[beta_start + token_head];
                    let mut memory = 0.0_f32;
                    for key_index in 0..self.key_dim {
                        let state_index = head_state + key_index * self.value_dim + value_index;
                        let previous = if token == 0 {
                            recurrent[recurrent_start + state_index]
                        } else {
                            output[state_index]
                        };
                        memory += previous * head_decay * key[key_start + head_key + key_index];
                    }
                    let delta =
                        (value[value_start + head_value + value_index] - memory) * head_beta;
                    let mut projected = 0.0_f32;
                    for key_index in 0..self.key_dim {
                        let state_index = head_state + key_index * self.value_dim + value_index;
                        let previous = if token == 0 {
                            recurrent[recurrent_start + state_index]
                        } else {
                            output[state_index]
                        };
                        let next =
                            previous * head_decay + key[key_start + head_key + key_index] * delta;
                        output[state_index] = next;
                        projected += next * query[query_start + head_key + key_index];
                    }
                    output[self.state_elements + head_value + value_index] = projected;
                }
            }
        }
        Ok((
            CpuStorage::F32(output),
            Shape::from_dims(&[self.combined_elements]),
        ))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        query: &candle_core::MetalStorage,
        query_layout: &Layout,
        key: &candle_core::MetalStorage,
        key_layout: &Layout,
        recurrent: &candle_core::MetalStorage,
        recurrent_layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::forward_recurrence(
            self,
            query,
            query_layout,
            key,
            key_layout,
            recurrent,
            recurrent_layout,
        )
    }
}

#[cfg(feature = "metal")]
mod metal {
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex, OnceLock},
    };

    use candle_core::{DType, Layout, MetalStorage, Result, Shape, Storage};
    use candle_metal_kernels::metal::{ComputePipeline, Library};
    use objc2_metal::{MTLDevice, MTLGPUFamily, MTLSize};

    use super::{GatedDeltaDecayOp, GatedDeltaRecurrenceOp};

    const SOURCE: &str = include_str!("metal/gated_delta_decay.metal");

    struct Pipelines {
        _library: Library,
        decay_f32: ComputePipeline,
        recurrence_f32: ComputePipeline,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct DecayParams {
        count: u32,
        heads: u32,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct RecurrenceParams {
        tokens: u32,
        heads: u32,
        key_dim: u32,
        value_dim: u32,
        state_elements: u32,
    }

    pub(super) fn device_supported(device: &candle_core::MetalDevice) -> bool {
        device
            .metal_device()
            .as_ref()
            .supportsFamily(MTLGPUFamily::Apple1)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_decay(
        op: &GatedDeltaDecayOp,
        alpha: &MetalStorage,
        alpha_layout: &Layout,
        time_bias: &MetalStorage,
        time_bias_layout: &Layout,
        state_decay: &MetalStorage,
        state_decay_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        let device = alpha.device();
        let pipelines = pipelines(device)?;
        if pipelines.decay_f32.max_total_threads_per_threadgroup() < 256 {
            candle_core::bail!("Gated DeltaNet decay requires 256 Metal threads");
        }
        let output_count = alpha_layout.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_count, DType::F32)
            .with_label("gated-delta-decay-output")
            .build()?;
        let params = DecayParams {
            count: u32::try_from(output_count).map_err(|_| {
                candle_core::Error::Msg("Gated DeltaNet decay size exceeds U32".into())
            })?,
            heads: u32::try_from(op.heads).map_err(|_| {
                candle_core::Error::Msg("Gated DeltaNet head count exceeds U32".into())
            })?,
        };

        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.decay_f32);
        let encoder = encoder.as_ref();
        encoder.set_input_buffer(
            0,
            Some(alpha.buffer()),
            alpha_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(time_bias.buffer()),
            time_bias_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(state_decay.buffer()),
            state_decay_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &params);
        encoder.dispatch_threads(
            MTLSize {
                width: output_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: output_count.min(256),
                height: 1,
                depth: 1,
            },
        );

        Ok((
            MetalStorage::new(output, device.clone(), output_count, DType::F32),
            alpha_layout.shape().clone(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_recurrence(
        op: &GatedDeltaRecurrenceOp<'_>,
        query: &MetalStorage,
        query_layout: &Layout,
        key: &MetalStorage,
        key_layout: &Layout,
        recurrent: &MetalStorage,
        recurrent_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        let device = query.device();
        let pipelines = pipelines(device)?;
        if pipelines.recurrence_f32.max_total_threads_per_threadgroup() < 256 {
            candle_core::bail!("Gated DeltaNet recurrence requires 256 Metal threads");
        }
        let (value_storage, value_layout) = op.value.storage_and_layout();
        let (decay_storage, decay_layout) = op.decay.storage_and_layout();
        let (beta_storage, beta_layout) = op.beta.storage_and_layout();
        let Storage::Metal(value) = &*value_storage else {
            candle_core::bail!("Metal Gated DeltaNet recurrence received non-Metal value");
        };
        let Storage::Metal(decay) = &*decay_storage else {
            candle_core::bail!("Metal Gated DeltaNet recurrence received non-Metal decay");
        };
        let Storage::Metal(beta) = &*beta_storage else {
            candle_core::bail!("Metal Gated DeltaNet recurrence received non-Metal beta");
        };
        let output_count = op.combined_elements;
        let output = device
            .new_buffer_builder()
            .with_size_for(output_count, DType::F32)
            .with_label("gated-delta-recurrence-output")
            .build()?;
        let params = RecurrenceParams {
            tokens: as_u32(op.tokens, "token count")?,
            heads: as_u32(op.heads, "head count")?,
            key_dim: as_u32(op.key_dim, "key dimension")?,
            value_dim: as_u32(op.value_dim, "value dimension")?,
            state_elements: as_u32(op.state_elements, "state size")?,
        };

        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.recurrence_f32);
        let encoder = encoder.as_ref();
        encoder.set_input_buffer(
            0,
            Some(query.buffer()),
            query_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(key.buffer()),
            key_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(recurrent.buffer()),
            recurrent_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            3,
            Some(value.buffer()),
            value_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            4,
            Some(decay.buffer()),
            decay_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            5,
            Some(beta.buffer()),
            beta_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(6, Some(&output), 0);
        encoder.set_bytes(7, &params);
        encoder.insert_memory_barrier();
        encoder.dispatch_thread_groups(
            MTLSize {
                width: op.heads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.insert_memory_barrier();

        Ok((
            MetalStorage::new(output, device.clone(), output_count, DType::F32),
            Shape::from_dims(&[output_count]),
        ))
    }

    fn pipelines(device: &candle_core::MetalDevice) -> Result<Arc<Pipelines>> {
        static CACHE: OnceLock<Mutex<HashMap<u64, Arc<Pipelines>>>> = OnceLock::new();
        let registry_id = device.metal_device().registry_id();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache.lock().map_err(|_| {
            candle_core::Error::Msg("Gated DeltaNet Metal pipeline cache poisoned".into())
        })?;
        if let Some(pipelines) = cache.get(&registry_id) {
            return Ok(Arc::clone(pipelines));
        }
        let library = device
            .metal_device()
            .new_library_with_source(SOURCE, None)
            .map_err(candle_core::Error::wrap)?;
        let decay_function = library
            .get_function("gated_delta_decay_f32", None)
            .map_err(candle_core::Error::wrap)?;
        let recurrence_function = library
            .get_function("gated_delta_decode_recurrence_f32", None)
            .map_err(candle_core::Error::wrap)?;
        let pipelines = Arc::new(Pipelines {
            decay_f32: device
                .metal_device()
                .new_compute_pipeline_state_with_function(&decay_function)
                .map_err(candle_core::Error::wrap)?,
            recurrence_f32: device
                .metal_device()
                .new_compute_pipeline_state_with_function(&recurrence_function)
                .map_err(candle_core::Error::wrap)?,
            _library: library,
        });
        cache.insert(registry_id, Arc::clone(&pipelines));
        Ok(pipelines)
    }

    fn as_u32(value: usize, label: &str) -> Result<u32> {
        u32::try_from(value)
            .map_err(|_| candle_core::Error::Msg(format!("Gated DeltaNet {label} exceeds U32")))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "metal")]
    use candle_core::Shape;
    use candle_core::{Device, Result, Tensor};

    use super::{GatedDeltaDecayOp, GatedDeltaRecurrenceOp};
    #[cfg(feature = "metal")]
    use super::{decay_factors, decode_recurrence, prefill_recurrence};

    #[cfg(feature = "metal")]
    fn offset_tensor(values: &[f32], shape: &[usize], device: &Device) -> Result<Tensor> {
        const PREFIX: usize = 7;
        let mut storage = vec![f32::NAN; PREFIX];
        storage.extend_from_slice(values);
        Tensor::from_vec(storage, PREFIX + values.len(), device)?
            .narrow(0, PREFIX, values.len())?
            .reshape(Shape::from_dims(shape))
    }

    #[cfg(feature = "metal")]
    fn assert_nonzero_offset(tensor: &Tensor) {
        let (_storage, layout) = tensor.storage_and_layout();
        assert!(layout.start_offset() > 0);
    }

    #[cfg(feature = "metal")]
    fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error <= tolerance,
            "{label} max error {max_error} exceeds {tolerance}"
        );
    }

    #[test]
    fn cpu_decay_matches_stable_scalar_formula() -> Result<()> {
        let device = Device::Cpu;
        let alpha = Tensor::from_vec(vec![-3.0_f32, 1.5, 20.0, -20.0], (2, 2), &device)?;
        let bias = Tensor::from_vec(vec![0.25_f32, -0.5], 2, &device)?;
        let state_decay = Tensor::from_vec(vec![-0.75_f32, -0.125], 2, &device)?;
        let actual = alpha
            .apply_op3_no_bwd(&bias, &state_decay, &GatedDeltaDecayOp { heads: 2 })?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let alpha = [-3.0_f32, 1.5, 20.0, -20.0];
        let bias = [0.25_f32, -0.5];
        let state_decay = [-0.75_f32, -0.125];
        for (index, actual) in actual.into_iter().enumerate() {
            let head = index % 2;
            let shifted = alpha[index] + bias[head];
            let softplus = shifted.max(0.0) + (-shifted.abs()).exp().ln_1p();
            let expected = (softplus * state_decay[head]).exp();
            assert!((actual - expected).abs() <= 1e-7);
        }
        Ok(())
    }

    #[test]
    fn cpu_recurrence_composite_output_matches_scalar_golden() -> Result<()> {
        let device = Device::Cpu;
        let query = Tensor::from_vec(vec![0.5_f32, -1.0], (1, 2), &device)?;
        let key = Tensor::from_vec(vec![0.25_f32, 0.75], (1, 2), &device)?;
        let value = Tensor::from_vec(vec![1.0_f32, 2.0], (1, 2), &device)?;
        let decay = Tensor::from_vec(vec![0.8_f32], 1, &device)?;
        let beta = Tensor::from_vec(vec![0.5_f32], 1, &device)?;
        let recurrent = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (1, 2, 2), &device)?;
        let combined = query.apply_op3_no_bwd(
            &key,
            &recurrent,
            &GatedDeltaRecurrenceOp {
                value: &value,
                decay: &decay,
                beta: &beta,
                tokens: 1,
                heads: 1,
                key_dim: 2,
                value_dim: 2,
                state_elements: 4,
                combined_elements: 6,
            },
        )?;
        let actual = combined.to_vec1::<f32>()?;
        let expected = [0.675_f32, 1.5, 2.025, 2.9, -1.6875, -2.15];
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6);
        }
        Ok(())
    }

    #[test]
    fn cpu_prefill_recurrence_matches_sequential_steps() -> Result<()> {
        let device = Device::Cpu;
        let query = Tensor::from_vec(vec![0.5_f32, -1.0, 0.25, 0.75], (2, 1, 2), &device)?;
        let key = Tensor::from_vec(vec![0.25_f32, 0.75, -0.5, 0.5], (2, 1, 2), &device)?;
        let value = Tensor::from_vec(vec![1.0_f32, 2.0, -1.0, 0.5], (2, 1, 2), &device)?;
        let decay = Tensor::from_vec(vec![0.8_f32, 0.9], (2, 1), &device)?;
        let beta = Tensor::from_vec(vec![0.5_f32, 0.25], (2, 1), &device)?;
        let recurrent = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (1, 2, 2), &device)?;
        let state_elements = 4;
        let one_shot = query.apply_op3_no_bwd(
            &key,
            &recurrent,
            &GatedDeltaRecurrenceOp {
                value: &value,
                decay: &decay,
                beta: &beta,
                tokens: 2,
                heads: 1,
                key_dim: 2,
                value_dim: 2,
                state_elements,
                combined_elements: state_elements + 4,
            },
        )?;

        let mut sequential_state = recurrent;
        let mut sequential_outputs = Vec::new();
        for token in 0..2 {
            let token_query = query.narrow(0, token, 1)?.reshape((1, 2))?;
            let token_key = key.narrow(0, token, 1)?.reshape((1, 2))?;
            let token_value = value.narrow(0, token, 1)?.reshape((1, 2))?;
            let token_decay = decay.narrow(0, token, 1)?.reshape(1)?;
            let token_beta = beta.narrow(0, token, 1)?.reshape(1)?;
            let combined = token_query.apply_op3_no_bwd(
                &token_key,
                &sequential_state,
                &GatedDeltaRecurrenceOp {
                    value: &token_value,
                    decay: &token_decay,
                    beta: &token_beta,
                    tokens: 1,
                    heads: 1,
                    key_dim: 2,
                    value_dim: 2,
                    state_elements,
                    combined_elements: state_elements + 2,
                },
            )?;
            sequential_state = combined.narrow(0, 0, state_elements)?.reshape((1, 2, 2))?;
            sequential_outputs.push(combined.narrow(0, state_elements, 2)?);
        }
        let sequential_outputs = sequential_outputs.iter().collect::<Vec<_>>();
        let sequential_outputs = Tensor::cat(&sequential_outputs, 0)?;
        let expected = Tensor::cat(&[&sequential_state.flatten_all()?, &sequential_outputs], 0)?;
        let actual = one_shot.to_vec1::<f32>()?;
        let expected = expected.to_vec1::<f32>()?;
        assert_eq!(actual, expected, "CPU prefill composition");
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an accessible Apple Metal device"]
    #[allow(clippy::too_many_lines)]
    fn metal_fused_ops_match_offset_cpu_oracle_and_preserve_retry_state() -> Result<()> {
        let cpu = Device::Cpu;
        let metal = Device::new_metal(0)?;

        let heads = 3;
        let alpha_values = (0..12)
            .map(|index| (index as f32 - 5.0) * 0.2)
            .collect::<Vec<_>>();
        let bias_values = [-0.3_f32, 0.1, 0.4];
        let state_decay_values = [-0.7_f32, -0.2, -1.1];
        let cpu_alpha = offset_tensor(&alpha_values, &[4, heads], &cpu)?;
        let cpu_bias = offset_tensor(&bias_values, &[heads], &cpu)?;
        let cpu_state_decay = offset_tensor(&state_decay_values, &[heads], &cpu)?;
        let expected_decay = cpu_alpha
            .apply_op3_no_bwd(&cpu_bias, &cpu_state_decay, &GatedDeltaDecayOp { heads })?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let metal_alpha = offset_tensor(&alpha_values, &[4, heads], &metal)?;
        let metal_bias = offset_tensor(&bias_values, &[heads], &metal)?;
        let metal_state_decay = offset_tensor(&state_decay_values, &[heads], &metal)?;
        for tensor in [&metal_alpha, &metal_bias, &metal_state_decay] {
            assert_nonzero_offset(tensor);
        }
        let actual_decay = decay_factors(&metal_alpha, &metal_bias, &metal_state_decay)?
            .expect("Apple Metal should use the fused decay kernel")
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close("decay", &actual_decay, &expected_decay, 2e-6);

        let key_dim = 5;
        let value_dim = 300;
        let state_elements = heads * key_dim * value_dim;
        let output_elements = heads * value_dim;
        let query_values = (0..heads * key_dim)
            .map(|index| ((index * 17 % 23) as f32 - 11.0) * 0.025)
            .collect::<Vec<_>>();
        let key_values = (0..heads * key_dim)
            .map(|index| ((index * 7 % 19) as f32 - 9.0) * 0.03)
            .collect::<Vec<_>>();
        let value_values = (0..output_elements)
            .map(|index| ((index * 13 % 37) as f32 - 18.0) * 0.01)
            .collect::<Vec<_>>();
        let decay_values = [0.91_f32, 0.83, 0.77];
        let beta_values = [0.2_f32, 0.6, 0.9];
        let recurrent_values = (0..state_elements)
            .map(|index| ((index * 29 % 101) as f32 - 50.0) * 0.001)
            .collect::<Vec<_>>();

        let cpu_query = offset_tensor(&query_values, &[heads, key_dim], &cpu)?;
        let cpu_key = offset_tensor(&key_values, &[heads, key_dim], &cpu)?;
        let cpu_value = offset_tensor(&value_values, &[heads, value_dim], &cpu)?;
        let cpu_decay = offset_tensor(&decay_values, &[heads], &cpu)?;
        let cpu_beta = offset_tensor(&beta_values, &[heads], &cpu)?;
        let cpu_recurrent = offset_tensor(&recurrent_values, &[heads, key_dim, value_dim], &cpu)?;
        let expected = cpu_query
            .apply_op3_no_bwd(
                &cpu_key,
                &cpu_recurrent,
                &GatedDeltaRecurrenceOp {
                    value: &cpu_value,
                    decay: &cpu_decay,
                    beta: &cpu_beta,
                    tokens: 1,
                    heads,
                    key_dim,
                    value_dim,
                    state_elements,
                    combined_elements: state_elements + output_elements,
                },
            )?
            .to_vec1::<f32>()?;
        let (expected_state, expected_output) = expected.split_at(state_elements);

        let metal_query = offset_tensor(&query_values, &[heads, key_dim], &metal)?;
        let metal_key = offset_tensor(&key_values, &[heads, key_dim], &metal)?;
        let metal_value = offset_tensor(&value_values, &[heads, value_dim], &metal)?;
        let metal_decay = offset_tensor(&decay_values, &[heads], &metal)?;
        let metal_beta = offset_tensor(&beta_values, &[heads], &metal)?;
        let metal_recurrent =
            offset_tensor(&recurrent_values, &[heads, key_dim, value_dim], &metal)?;
        for tensor in [
            &metal_query,
            &metal_key,
            &metal_value,
            &metal_decay,
            &metal_beta,
            &metal_recurrent,
        ] {
            assert_nonzero_offset(tensor);
        }
        let original_state = metal_recurrent.flatten_all()?.to_vec1::<f32>()?;
        let first = decode_recurrence(
            &metal_query,
            &metal_key,
            &metal_value,
            &metal_decay,
            &metal_beta,
            &metal_recurrent,
        )?
        .expect("Apple Metal should use the fused recurrence kernel");
        let retry = decode_recurrence(
            &metal_query,
            &metal_key,
            &metal_value,
            &metal_decay,
            &metal_beta,
            &metal_recurrent,
        )?
        .expect("retry should reuse the original immutable state");

        let first_output = first.0.flatten_all()?.to_vec1::<f32>()?;
        let first_state = first.1.flatten_all()?.to_vec1::<f32>()?;
        let retry_output = retry.0.flatten_all()?.to_vec1::<f32>()?;
        let retry_state = retry.1.flatten_all()?.to_vec1::<f32>()?;
        let state_after_retry = metal_recurrent.flatten_all()?.to_vec1::<f32>()?;
        assert_close("recurrence output", &first_output, expected_output, 2e-5);
        assert_close("recurrence state", &first_state, expected_state, 2e-5);
        assert_close("retry output", &retry_output, &first_output, 0.0);
        assert_close("retry state", &retry_state, &first_state, 0.0);
        assert_close("original state", &state_after_retry, &original_state, 0.0);

        let tokens = 4;
        let prefill_query_values = (0..tokens * heads * key_dim)
            .map(|index| ((index * 17 % 31) as f32 - 15.0) * 0.015)
            .collect::<Vec<_>>();
        let prefill_key_values = (0..tokens * heads * key_dim)
            .map(|index| ((index * 11 % 29) as f32 - 14.0) * 0.017)
            .collect::<Vec<_>>();
        let prefill_value_values = (0..tokens * output_elements)
            .map(|index| ((index * 13 % 41) as f32 - 20.0) * 0.008)
            .collect::<Vec<_>>();
        let prefill_decay_values = (0..tokens * heads)
            .map(|index| 0.72_f32 + (index % heads) as f32 * 0.06)
            .collect::<Vec<_>>();
        let prefill_beta_values = (0..tokens * heads)
            .map(|index| 0.15_f32 + (index % 5) as f32 * 0.12)
            .collect::<Vec<_>>();

        let cpu_query = offset_tensor(&prefill_query_values, &[tokens, heads, key_dim], &cpu)?;
        let cpu_key = offset_tensor(&prefill_key_values, &[tokens, heads, key_dim], &cpu)?;
        let cpu_value = offset_tensor(&prefill_value_values, &[tokens, heads, value_dim], &cpu)?;
        let cpu_decay = offset_tensor(&prefill_decay_values, &[tokens, heads], &cpu)?;
        let cpu_beta = offset_tensor(&prefill_beta_values, &[tokens, heads], &cpu)?;
        let cpu_recurrent = offset_tensor(&recurrent_values, &[heads, key_dim, value_dim], &cpu)?;
        let prefill_output_elements = tokens * output_elements;
        let expected = cpu_query
            .apply_op3_no_bwd(
                &cpu_key,
                &cpu_recurrent,
                &GatedDeltaRecurrenceOp {
                    value: &cpu_value,
                    decay: &cpu_decay,
                    beta: &cpu_beta,
                    tokens,
                    heads,
                    key_dim,
                    value_dim,
                    state_elements,
                    combined_elements: state_elements + prefill_output_elements,
                },
            )?
            .to_vec1::<f32>()?;
        let (expected_state, expected_output) = expected.split_at(state_elements);

        let metal_query = offset_tensor(&prefill_query_values, &[tokens, heads, key_dim], &metal)?;
        let metal_key = offset_tensor(&prefill_key_values, &[tokens, heads, key_dim], &metal)?;
        let metal_value =
            offset_tensor(&prefill_value_values, &[tokens, heads, value_dim], &metal)?;
        let metal_decay = offset_tensor(&prefill_decay_values, &[tokens, heads], &metal)?;
        let metal_beta = offset_tensor(&prefill_beta_values, &[tokens, heads], &metal)?;
        let metal_recurrent =
            offset_tensor(&recurrent_values, &[heads, key_dim, value_dim], &metal)?;
        for tensor in [
            &metal_query,
            &metal_key,
            &metal_value,
            &metal_decay,
            &metal_beta,
            &metal_recurrent,
        ] {
            assert_nonzero_offset(tensor);
        }
        let original_state = metal_recurrent.flatten_all()?.to_vec1::<f32>()?;
        let first = prefill_recurrence(
            &metal_query,
            &metal_key,
            &metal_value,
            &metal_decay,
            &metal_beta,
            &metal_recurrent,
        )?
        .expect("Apple Metal should use the fused prefill recurrence kernel");
        let retry = prefill_recurrence(
            &metal_query,
            &metal_key,
            &metal_value,
            &metal_decay,
            &metal_beta,
            &metal_recurrent,
        )?
        .expect("prefill retry should reuse the original immutable state");
        let first_output = first.0.flatten_all()?.to_vec1::<f32>()?;
        let first_state = first.1.flatten_all()?.to_vec1::<f32>()?;
        let retry_output = retry.0.flatten_all()?.to_vec1::<f32>()?;
        let retry_state = retry.1.flatten_all()?.to_vec1::<f32>()?;
        let state_after_retry = metal_recurrent.flatten_all()?.to_vec1::<f32>()?;
        assert_close("prefill output", &first_output, expected_output, 3e-5);
        assert_close("prefill state", &first_state, expected_state, 3e-5);
        assert_close("prefill retry output", &retry_output, &first_output, 0.0);
        assert_close("prefill retry state", &retry_state, &first_state, 0.0);
        assert_close(
            "prefill original state",
            &state_after_retry,
            &original_state,
            0.0,
        );
        Ok(())
    }
}
