use candle_core::{CpuStorage, CustomOp3, DType, Device, Layout, Result, Shape, Storage, Tensor};

/// Run the selective Mamba2 recurrence as one backend operation.
///
/// `xbc` is the complete contiguous, convolved projection
/// `[S, H * P + 2 * G * N]`, laid out as `[x | B | C]`. The remaining inputs
/// are raw `dt[S, H]`, `dt_bias[H]`, canonical negative decay `a[H]`, skip
/// weight `d[H]`, and immutable initial state `[H, P, N]`. Group geometry is
/// derived from the full projection width, and heads are divided evenly into
/// contiguous groups.
///
/// The Metal implementation computes stable `softplus(dt + dt_bias)` and
/// returns token output `[S, H, P]` plus final state `[H, P, N]` as views into
/// one allocation. Unsupported devices, dtypes, or layouts return `None` so
/// the decoder can use its readable tensor fallback.
#[doc(hidden)]
pub fn scan(
    xbc: &Tensor,
    raw_dt: &Tensor,
    dt_bias: &Tensor,
    a: &Tensor,
    d: &Tensor,
    initial_state: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let geometry = ScanGeometry::validate(xbc, raw_dt, dt_bias, a, d, initial_state)?;
    if [xbc, raw_dt, dt_bias, a, d, initial_state]
        .iter()
        .any(|tensor| tensor.dtype() != DType::F32 || !tensor.is_contiguous())
    {
        return Ok(None);
    }

    let supported = match xbc.device() {
        // The primitive decoder path remains the CPU fallback and numerical
        // oracle. This boundary only replaces many Metal tensor launches.
        Device::Cpu => false,
        #[cfg(feature = "metal")]
        Device::Metal(device) => metal::device_supported(device),
        #[allow(unreachable_patterns)]
        _ => false,
    };
    if !supported {
        return Ok(None);
    }

    apply_scan(xbc, raw_dt, dt_bias, a, d, initial_state, geometry).map(Some)
}

#[derive(Clone, Copy, Debug)]
struct ScanGeometry {
    tokens: usize,
    heads: usize,
    head_dim: usize,
    state_size: usize,
    heads_per_group: usize,
    inner_size: usize,
    group_state_width: usize,
    xbc_width: usize,
    state_elements: usize,
    output_elements: usize,
    combined_elements: usize,
}

impl ScanGeometry {
    fn validate(
        xbc: &Tensor,
        raw_dt: &Tensor,
        dt_bias: &Tensor,
        a: &Tensor,
        d: &Tensor,
        initial_state: &Tensor,
    ) -> Result<Self> {
        let (tokens, xbc_width) = xbc.dims2()?;
        let (heads, head_dim, state_size) = initial_state.dims3()?;
        if tokens == 0 || heads == 0 || head_dim == 0 || state_size == 0 {
            candle_core::bail!("Mamba2 scan requires non-empty dimensions");
        }
        if raw_dt.dims2()? != (tokens, heads)
            || dt_bias.dims1()? != heads
            || a.dims1()? != heads
            || d.dims1()? != heads
        {
            candle_core::bail!(
                "Mamba2 scan has xBC {:?}, dt {:?}, dt bias {:?}, A {:?}, D {:?}, and state {:?}",
                xbc.dims(),
                raw_dt.dims(),
                dt_bias.dims(),
                a.dims(),
                d.dims(),
                initial_state.dims()
            );
        }
        for tensor in [raw_dt, dt_bias, a, d, initial_state] {
            if !tensor.device().same_device(xbc.device()) {
                candle_core::bail!("Mamba2 scan tensors are on different devices");
            }
        }

        let inner_size = heads
            .checked_mul(head_dim)
            .ok_or_else(|| candle_core::Error::Msg("Mamba2 scan inner size overflow".into()))?;
        let bc_width = xbc_width.checked_sub(inner_size).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "Mamba2 xBC width {xbc_width} is smaller than inner size {inner_size}"
            ))
        })?;
        let one_group_bc = state_size
            .checked_mul(2)
            .ok_or_else(|| candle_core::Error::Msg("Mamba2 scan B/C width overflow".into()))?;
        if bc_width == 0 || bc_width % one_group_bc != 0 {
            candle_core::bail!(
                "Mamba2 xBC width {xbc_width} cannot encode paired B/C states of width {state_size}"
            );
        }
        let groups = bc_width / one_group_bc;
        if groups == 0 || heads % groups != 0 {
            candle_core::bail!(
                "Mamba2 scan head count {heads} is not divisible by inferred group count {groups}"
            );
        }
        let group_state_width = groups
            .checked_mul(state_size)
            .ok_or_else(|| candle_core::Error::Msg("Mamba2 scan group state overflow".into()))?;
        let expected_width = inner_size
            .checked_add(group_state_width.checked_mul(2).ok_or_else(|| {
                candle_core::Error::Msg("Mamba2 scan projection width overflow".into())
            })?)
            .ok_or_else(|| {
                candle_core::Error::Msg("Mamba2 scan projection width overflow".into())
            })?;
        if xbc_width != expected_width {
            candle_core::bail!(
                "Mamba2 xBC width is {xbc_width}, expected {expected_width} from inferred geometry"
            );
        }

        let state_elements = inner_size
            .checked_mul(state_size)
            .ok_or_else(|| candle_core::Error::Msg("Mamba2 scan state size overflow".into()))?;
        let output_elements = tokens
            .checked_mul(inner_size)
            .ok_or_else(|| candle_core::Error::Msg("Mamba2 scan output size overflow".into()))?;
        let combined_elements = state_elements
            .checked_add(output_elements)
            .ok_or_else(|| candle_core::Error::Msg("Mamba2 scan result size overflow".into()))?;
        Ok(Self {
            tokens,
            heads,
            head_dim,
            state_size,
            heads_per_group: heads / groups,
            inner_size,
            group_state_width,
            xbc_width,
            state_elements,
            output_elements,
            combined_elements,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_scan(
    xbc: &Tensor,
    raw_dt: &Tensor,
    dt_bias: &Tensor,
    a: &Tensor,
    d: &Tensor,
    initial_state: &Tensor,
    geometry: ScanGeometry,
) -> Result<(Tensor, Tensor)> {
    let combined = xbc.apply_op3_no_bwd(
        raw_dt,
        initial_state,
        &Mamba2ScanOp {
            dt_bias,
            a,
            d,
            geometry,
        },
    )?;
    let final_state = combined.narrow(0, 0, geometry.state_elements)?.reshape((
        geometry.heads,
        geometry.head_dim,
        geometry.state_size,
    ))?;
    let output = combined
        .narrow(0, geometry.state_elements, geometry.output_elements)?
        .reshape((geometry.tokens, geometry.heads, geometry.head_dim))?;
    Ok((output, final_state))
}

struct Mamba2ScanOp<'a> {
    dt_bias: &'a Tensor,
    a: &'a Tensor,
    d: &'a Tensor,
    geometry: ScanGeometry,
}

impl CustomOp3 for Mamba2ScanOp<'_> {
    fn name(&self) -> &'static str {
        "mamba2-scan"
    }

    #[allow(clippy::too_many_lines)]
    fn cpu_fwd(
        &self,
        xbc: &CpuStorage,
        xbc_layout: &Layout,
        raw_dt: &CpuStorage,
        raw_dt_layout: &Layout,
        initial_state: &CpuStorage,
        initial_state_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(xbc) = xbc else {
            candle_core::bail!("CPU Mamba2 scan requires F32 xBC");
        };
        let CpuStorage::F32(raw_dt) = raw_dt else {
            candle_core::bail!("CPU Mamba2 scan requires F32 dt");
        };
        let CpuStorage::F32(initial_state) = initial_state else {
            candle_core::bail!("CPU Mamba2 scan requires F32 initial state");
        };
        if !xbc_layout.is_contiguous()
            || !raw_dt_layout.is_contiguous()
            || !initial_state_layout.is_contiguous()
        {
            candle_core::bail!("CPU Mamba2 scan requires contiguous direct inputs");
        }

        let (dt_bias_storage, dt_bias_layout) = self.dt_bias.storage_and_layout();
        let (a_storage, a_layout) = self.a.storage_and_layout();
        let (d_storage, d_layout) = self.d.storage_and_layout();
        let Storage::Cpu(CpuStorage::F32(dt_bias)) = &*dt_bias_storage else {
            candle_core::bail!("CPU Mamba2 scan requires F32 dt bias");
        };
        let Storage::Cpu(CpuStorage::F32(a)) = &*a_storage else {
            candle_core::bail!("CPU Mamba2 scan requires F32 A");
        };
        let Storage::Cpu(CpuStorage::F32(d)) = &*d_storage else {
            candle_core::bail!("CPU Mamba2 scan requires F32 D");
        };
        if !dt_bias_layout.is_contiguous() || !a_layout.is_contiguous() || !d_layout.is_contiguous()
        {
            candle_core::bail!("CPU Mamba2 scan requires contiguous captured inputs");
        }

        let geometry = self.geometry;
        let xbc_start = xbc_layout.start_offset();
        let raw_dt_start = raw_dt_layout.start_offset();
        let dt_bias_start = dt_bias_layout.start_offset();
        let a_start = a_layout.start_offset();
        let skip_start = d_layout.start_offset();
        let state_start = initial_state_layout.start_offset();
        let mut combined = vec![0.0_f32; geometry.combined_elements];
        combined[..geometry.state_elements]
            .copy_from_slice(&initial_state[state_start..state_start + geometry.state_elements]);

        for token in 0..geometry.tokens {
            let projection_base = xbc_start + token * geometry.xbc_width;
            for head in 0..geometry.heads {
                let token_head = token * geometry.heads + head;
                let shifted = raw_dt[raw_dt_start + token_head] + dt_bias[dt_bias_start + head];
                let delta = shifted.max(0.0) + (-shifted.abs()).exp().ln_1p();
                let decay = (delta * a[a_start + head]).exp();
                let group = head / geometry.heads_per_group;
                let b_base = projection_base + geometry.inner_size + group * geometry.state_size;
                let c_base = projection_base
                    + geometry.inner_size
                    + geometry.group_state_width
                    + group * geometry.state_size;
                for position in 0..geometry.head_dim {
                    let lane = head * geometry.head_dim + position;
                    let value = xbc[projection_base + lane];
                    let state_base = lane * geometry.state_size;
                    let mut projected = 0.0_f32;
                    for state_index in 0..geometry.state_size {
                        let recurrent_index = state_base + state_index;
                        let next = decay * combined[recurrent_index]
                            + delta * value * xbc[b_base + state_index];
                        combined[recurrent_index] = next;
                        projected += next * xbc[c_base + state_index];
                    }
                    let output_index = token * geometry.inner_size + lane;
                    combined[geometry.state_elements + output_index] =
                        projected + d[skip_start + head] * value;
                }
            }
        }

        Ok((
            CpuStorage::F32(combined),
            Shape::from_dims(&[geometry.combined_elements]),
        ))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        xbc: &candle_core::MetalStorage,
        xbc_layout: &Layout,
        raw_dt: &candle_core::MetalStorage,
        raw_dt_layout: &Layout,
        initial_state: &candle_core::MetalStorage,
        initial_state_layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::forward(
            self,
            xbc,
            xbc_layout,
            raw_dt,
            raw_dt_layout,
            initial_state,
            initial_state_layout,
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

    use super::Mamba2ScanOp;

    const SOURCE: &str = include_str!("metal/mamba2_scan.metal");

    struct Pipelines {
        _library: Library,
        f32: ComputePipeline,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct ScanParams {
        tokens: u32,
        heads: u32,
        head_dim: u32,
        state_size: u32,
        heads_per_group: u32,
        inner_size: u32,
        group_state_width: u32,
        xbc_width: u32,
        state_elements: u32,
        threadgroup_width: u32,
    }

    pub(super) fn device_supported(device: &candle_core::MetalDevice) -> bool {
        device
            .metal_device()
            .as_ref()
            .supportsFamily(MTLGPUFamily::Apple1)
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn forward(
        op: &Mamba2ScanOp<'_>,
        xbc: &MetalStorage,
        xbc_layout: &Layout,
        raw_dt: &MetalStorage,
        raw_dt_layout: &Layout,
        initial_state: &MetalStorage,
        initial_state_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        let device = xbc.device();
        let pipelines = pipelines(device)?;
        let (dt_bias_storage, dt_bias_layout) = op.dt_bias.storage_and_layout();
        let (a_storage, a_layout) = op.a.storage_and_layout();
        let (d_storage, d_layout) = op.d.storage_and_layout();
        let Storage::Metal(dt_bias) = &*dt_bias_storage else {
            candle_core::bail!("Metal Mamba2 scan received non-Metal dt bias");
        };
        let Storage::Metal(a) = &*a_storage else {
            candle_core::bail!("Metal Mamba2 scan received non-Metal A");
        };
        let Storage::Metal(d) = &*d_storage else {
            candle_core::bail!("Metal Mamba2 scan received non-Metal D");
        };
        let geometry = op.geometry;
        let threadgroup_width = geometry
            .state_size
            .checked_next_power_of_two()
            .unwrap_or(256)
            .min(256);
        if pipelines.f32.max_total_threads_per_threadgroup() < threadgroup_width {
            candle_core::bail!(
                "Mamba2 scan requires {threadgroup_width} Metal threads per threadgroup"
            );
        }
        let output = device
            .new_buffer_builder()
            .with_size_for(geometry.combined_elements, DType::F32)
            .with_label("mamba2-scan-output-and-state")
            .build()?;
        let params = ScanParams {
            tokens: as_u32(geometry.tokens, "token count")?,
            heads: as_u32(geometry.heads, "head count")?,
            head_dim: as_u32(geometry.head_dim, "head dimension")?,
            state_size: as_u32(geometry.state_size, "state size")?,
            heads_per_group: as_u32(geometry.heads_per_group, "heads per group")?,
            inner_size: as_u32(geometry.inner_size, "inner size")?,
            group_state_width: as_u32(geometry.group_state_width, "group state width")?,
            xbc_width: as_u32(geometry.xbc_width, "xBC width")?,
            state_elements: as_u32(geometry.state_elements, "state element count")?,
            threadgroup_width: as_u32(threadgroup_width, "threadgroup width")?,
        };

        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.f32);
        let encoder = encoder.as_ref();
        encoder.set_input_buffer(
            0,
            Some(xbc.buffer()),
            xbc_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(raw_dt.buffer()),
            raw_dt_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(initial_state.buffer()),
            initial_state_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            3,
            Some(dt_bias.buffer()),
            dt_bias_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            4,
            Some(a.buffer()),
            a_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            5,
            Some(d.buffer()),
            d_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(6, Some(&output), 0);
        encoder.set_bytes(7, &params);
        let lane_count = geometry.inner_size;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: lane_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threadgroup_width,
                height: 1,
                depth: 1,
            },
        );

        Ok((
            MetalStorage::new(
                output,
                device.clone(),
                geometry.combined_elements,
                DType::F32,
            ),
            Shape::from_dims(&[geometry.combined_elements]),
        ))
    }

    fn pipelines(device: &candle_core::MetalDevice) -> Result<Arc<Pipelines>> {
        static CACHE: OnceLock<Mutex<HashMap<u64, Arc<Pipelines>>>> = OnceLock::new();
        let registry_id = device.metal_device().registry_id();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache
            .lock()
            .map_err(|_| candle_core::Error::Msg("Mamba2 Metal pipeline cache poisoned".into()))?;
        if let Some(pipelines) = cache.get(&registry_id) {
            return Ok(Arc::clone(pipelines));
        }
        let library = device
            .metal_device()
            .new_library_with_source(SOURCE, None)
            .map_err(candle_core::Error::wrap)?;
        let function = library
            .get_function("mamba2_scan_f32", None)
            .map_err(candle_core::Error::wrap)?;
        let pipelines = Arc::new(Pipelines {
            f32: device
                .metal_device()
                .new_compute_pipeline_state_with_function(&function)
                .map_err(candle_core::Error::wrap)?,
            _library: library,
        });
        cache.insert(registry_id, Arc::clone(&pipelines));
        Ok(pipelines)
    }

    fn as_u32(value: usize, label: &str) -> Result<u32> {
        u32::try_from(value)
            .map_err(|_| candle_core::Error::Msg(format!("Mamba2 scan {label} exceeds U32")))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Shape, Tensor};

    use super::{ScanGeometry, apply_scan, scan};

    const TOKENS: usize = 3;
    const HEADS: usize = 4;
    const HEAD_DIM: usize = 2;
    const GROUPS: usize = 2;
    const STATE_SIZE: usize = 3;
    const INNER_SIZE: usize = HEADS * HEAD_DIM;
    const GROUP_STATE_WIDTH: usize = GROUPS * STATE_SIZE;
    const XBC_WIDTH: usize = INNER_SIZE + 2 * GROUP_STATE_WIDTH;
    const PREFIX: usize = 7;

    struct Values {
        xbc: Vec<f32>,
        raw_dt: Vec<f32>,
        dt_bias: Vec<f32>,
        a: Vec<f32>,
        d: Vec<f32>,
        initial_state: Vec<f32>,
    }

    struct Inputs {
        xbc: Tensor,
        raw_dt: Tensor,
        dt_bias: Tensor,
        a: Tensor,
        d: Tensor,
        initial_state: Tensor,
    }

    fn fixture_values() -> Values {
        let mut xbc = Vec::with_capacity(TOKENS * XBC_WIDTH);
        for token in 0..TOKENS {
            xbc.extend(
                (0..INNER_SIZE).map(|index| ((token * 11 + index * 7) % 19) as f32 * 0.07 - 0.5),
            );
            xbc.extend(
                (0..GROUP_STATE_WIDTH)
                    .map(|index| ((token * 5 + index * 3) % 13) as f32 * 0.04 - 0.2),
            );
            xbc.extend(
                (0..GROUP_STATE_WIDTH)
                    .map(|index| ((token * 7 + index * 5) % 17) as f32 * 0.03 - 0.18),
            );
        }
        Values {
            xbc,
            // Extreme values lock the stable softplus formula: exp(100)
            // would overflow in the naive implementation.
            raw_dt: vec![
                100.0, -100.0, 0.2, -0.7, -1.3, 0.9, 1.4, -0.2, 0.0, 0.4, -0.8, 1.1,
            ],
            dt_bias: vec![-0.15, 0.25, -0.4, 0.05],
            a: vec![-0.03, -0.2, -0.45, -0.1],
            d: vec![0.2, -0.3, 0.5, 0.1],
            initial_state: (0..INNER_SIZE * STATE_SIZE)
                .map(|index| (index as f32 - 11.0) * 0.015)
                .collect(),
        }
    }

    fn offset_tensor(values: &[f32], shape: &[usize], device: &Device) -> Result<Tensor> {
        let mut storage = vec![f32::NAN; PREFIX];
        storage.extend_from_slice(values);
        Tensor::from_vec(storage, PREFIX + values.len(), device)?
            .narrow(0, PREFIX, values.len())?
            .reshape(Shape::from_dims(shape))
    }

    fn inputs(values: &Values, device: &Device) -> Result<Inputs> {
        Ok(Inputs {
            xbc: offset_tensor(&values.xbc, &[TOKENS, XBC_WIDTH], device)?,
            raw_dt: offset_tensor(&values.raw_dt, &[TOKENS, HEADS], device)?,
            dt_bias: offset_tensor(&values.dt_bias, &[HEADS], device)?,
            a: offset_tensor(&values.a, &[HEADS], device)?,
            d: offset_tensor(&values.d, &[HEADS], device)?,
            initial_state: offset_tensor(
                &values.initial_state,
                &[HEADS, HEAD_DIM, STATE_SIZE],
                device,
            )?,
        })
    }

    fn assert_nonzero_offsets(inputs: &Inputs) {
        for tensor in [
            &inputs.xbc,
            &inputs.raw_dt,
            &inputs.dt_bias,
            &inputs.a,
            &inputs.d,
            &inputs.initial_state,
        ] {
            let (_storage, layout) = tensor.storage_and_layout();
            assert!(layout.start_offset() > 0);
        }
    }

    fn oracle(inputs: &Inputs) -> Result<(Tensor, Tensor)> {
        let geometry = ScanGeometry::validate(
            &inputs.xbc,
            &inputs.raw_dt,
            &inputs.dt_bias,
            &inputs.a,
            &inputs.d,
            &inputs.initial_state,
        )?;
        apply_scan(
            &inputs.xbc,
            &inputs.raw_dt,
            &inputs.dt_bias,
            &inputs.a,
            &inputs.d,
            &inputs.initial_state,
            geometry,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_scan(
        xbc: &[f32],
        raw_dt: &[f32],
        dt_bias: &[f32],
        a: &[f32],
        d: &[f32],
        initial_state: &[f32],
        tokens: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut state = initial_state.to_vec();
        let mut output = vec![0.0_f32; tokens * INNER_SIZE];
        let heads_per_group = HEADS / GROUPS;
        for token in 0..tokens {
            let projection_base = token * XBC_WIDTH;
            for head in 0..HEADS {
                let shifted = raw_dt[token * HEADS + head] + dt_bias[head];
                let delta = shifted.max(0.0) + (-shifted.abs()).exp().ln_1p();
                let decay = (delta * a[head]).exp();
                let group = head / heads_per_group;
                let b_base = projection_base + INNER_SIZE + group * STATE_SIZE;
                let c_base = projection_base + INNER_SIZE + GROUP_STATE_WIDTH + group * STATE_SIZE;
                for position in 0..HEAD_DIM {
                    let lane = head * HEAD_DIM + position;
                    let value = xbc[projection_base + lane];
                    let state_base = lane * STATE_SIZE;
                    let mut projected = 0.0_f32;
                    for state_index in 0..STATE_SIZE {
                        let index = state_base + state_index;
                        state[index] =
                            decay * state[index] + delta * value * xbc[b_base + state_index];
                        projected += state[index] * xbc[c_base + state_index];
                    }
                    output[token * INNER_SIZE + lane] = projected + d[head] * value;
                }
            }
        }
        (output, state)
    }

    fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        let mut max_error = 0.0_f32;
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                actual.is_finite(),
                "{label}[{index}] is not finite: {actual}"
            );
            assert!(
                expected.is_finite(),
                "{label} expected[{index}] is not finite: {expected}"
            );
            max_error = max_error.max((actual - expected).abs());
        }
        assert!(
            max_error <= tolerance,
            "{label} max error {max_error} exceeds {tolerance}"
        );
    }

    #[test]
    fn cpu_oracle_handles_offsets_chunking_decode_and_final_state() -> Result<()> {
        let values = fixture_values();
        let cpu = Device::Cpu;
        let inputs = inputs(&values, &cpu)?;
        assert_nonzero_offsets(&inputs);
        assert!(
            scan(
                &inputs.xbc,
                &inputs.raw_dt,
                &inputs.dt_bias,
                &inputs.a,
                &inputs.d,
                &inputs.initial_state,
            )?
            .is_none(),
            "CPU must leave execution to the primitive decoder fallback"
        );

        let (full_output, full_state) = oracle(&inputs)?;
        let (expected_output, expected_state) = scalar_scan(
            &values.xbc,
            &values.raw_dt,
            &values.dt_bias,
            &values.a,
            &values.d,
            &values.initial_state,
            TOKENS,
        );
        assert_close(
            "CPU oracle output",
            &full_output.flatten_all()?.to_vec1::<f32>()?,
            &expected_output,
            1e-7,
        );
        assert_close(
            "CPU oracle final state",
            &full_state.flatten_all()?.to_vec1::<f32>()?,
            &expected_state,
            1e-7,
        );

        let prefix_inputs = Inputs {
            xbc: inputs.xbc.narrow(0, 0, 2)?,
            raw_dt: inputs.raw_dt.narrow(0, 0, 2)?,
            dt_bias: inputs.dt_bias.clone(),
            a: inputs.a.clone(),
            d: inputs.d.clone(),
            initial_state: inputs.initial_state.clone(),
        };
        let (prefix_output, prefix_state) = oracle(&prefix_inputs)?;
        let decode_inputs = Inputs {
            // The final chunk is exactly one decode token and retains a
            // non-zero source-layout offset.
            xbc: inputs.xbc.narrow(0, 2, 1)?,
            raw_dt: inputs.raw_dt.narrow(0, 2, 1)?,
            dt_bias: inputs.dt_bias.clone(),
            a: inputs.a.clone(),
            d: inputs.d.clone(),
            initial_state: prefix_state,
        };
        let (decode_output, chunked_state) = oracle(&decode_inputs)?;
        let chunked_output = Tensor::cat(&[&prefix_output, &decode_output], 0)?;
        assert_close(
            "CPU chunked output",
            &chunked_output.flatten_all()?.to_vec1::<f32>()?,
            &expected_output,
            1e-7,
        );
        assert_close(
            "CPU chunked final state",
            &chunked_state.flatten_all()?.to_vec1::<f32>()?,
            &expected_state,
            1e-7,
        );
        assert_close(
            "immutable initial state",
            &inputs.initial_state.flatten_all()?.to_vec1::<f32>()?,
            &values.initial_state,
            0.0,
        );
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an accessible Apple Metal device"]
    fn metal_scan_matches_offset_cpu_oracle_for_prefill_decode_and_state() -> Result<()> {
        let values = fixture_values();
        let cpu_inputs = inputs(&values, &Device::Cpu)?;
        let (expected_output, expected_state) = oracle(&cpu_inputs)?;
        let expected_output = expected_output.flatten_all()?.to_vec1::<f32>()?;
        let expected_state = expected_state.flatten_all()?.to_vec1::<f32>()?;

        let metal = Device::new_metal(0)?;
        let metal_inputs = inputs(&values, &metal)?;
        assert_nonzero_offsets(&metal_inputs);
        let original_state = metal_inputs.initial_state.flatten_all()?.to_vec1::<f32>()?;
        let (output, final_state) = scan(
            &metal_inputs.xbc,
            &metal_inputs.raw_dt,
            &metal_inputs.dt_bias,
            &metal_inputs.a,
            &metal_inputs.d,
            &metal_inputs.initial_state,
        )?
        .expect("Apple Metal should use the fused Mamba2 scan");
        assert_close(
            "Metal prefill output",
            &output.flatten_all()?.to_vec1::<f32>()?,
            &expected_output,
            3e-5,
        );
        assert_close(
            "Metal prefill state",
            &final_state.flatten_all()?.to_vec1::<f32>()?,
            &expected_state,
            3e-5,
        );

        let prefix = scan(
            &metal_inputs.xbc.narrow(0, 0, 2)?,
            &metal_inputs.raw_dt.narrow(0, 0, 2)?,
            &metal_inputs.dt_bias,
            &metal_inputs.a,
            &metal_inputs.d,
            &metal_inputs.initial_state,
        )?
        .expect("Metal should fuse a multi-token prefix");
        let decode = scan(
            &metal_inputs.xbc.narrow(0, 2, 1)?,
            &metal_inputs.raw_dt.narrow(0, 2, 1)?,
            &metal_inputs.dt_bias,
            &metal_inputs.a,
            &metal_inputs.d,
            &prefix.1,
        )?
        .expect("Metal should fuse a one-token decode");
        let chunked_output = Tensor::cat(&[&prefix.0, &decode.0], 0)?;
        assert_close(
            "Metal chunked output",
            &chunked_output.flatten_all()?.to_vec1::<f32>()?,
            &expected_output,
            3e-5,
        );
        assert_close(
            "Metal chunked state",
            &decode.1.flatten_all()?.to_vec1::<f32>()?,
            &expected_state,
            3e-5,
        );
        assert_close(
            "Metal immutable initial state",
            &metal_inputs.initial_state.flatten_all()?.to_vec1::<f32>()?,
            &original_state,
            0.0,
        );
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an accessible Apple Metal device"]
    fn metal_scan_strides_state_larger_than_one_threadgroup() -> Result<()> {
        const LARGE_STATE: usize = 300;
        const LARGE_WIDTH: usize = 1 + 2 * LARGE_STATE;
        let cpu = Device::Cpu;
        let metal = Device::new_metal(0)?;
        let xbc_values = (0..LARGE_WIDTH)
            .map(|index| (index % 31) as f32 * 0.004 - 0.05)
            .collect::<Vec<_>>();
        let state_values = (0..LARGE_STATE)
            .map(|index| (index % 23) as f32 * 0.003 - 0.02)
            .collect::<Vec<_>>();

        let make = |device: &Device| -> Result<Inputs> {
            Ok(Inputs {
                xbc: offset_tensor(&xbc_values, &[1, LARGE_WIDTH], device)?,
                raw_dt: offset_tensor(&[0.35], &[1, 1], device)?,
                dt_bias: offset_tensor(&[-0.1], &[1], device)?,
                a: offset_tensor(&[-0.2], &[1], device)?,
                d: offset_tensor(&[0.4], &[1], device)?,
                initial_state: offset_tensor(&state_values, &[1, 1, LARGE_STATE], device)?,
            })
        };
        let (expected_output, expected_state) = oracle(&make(&cpu)?)?;
        let metal_inputs = make(&metal)?;
        let (actual_output, actual_state) = scan(
            &metal_inputs.xbc,
            &metal_inputs.raw_dt,
            &metal_inputs.dt_bias,
            &metal_inputs.a,
            &metal_inputs.d,
            &metal_inputs.initial_state,
        )?
        .expect("Metal should stride N=300 across a capped 256-thread group");
        assert_close(
            "large-state output",
            &actual_output.flatten_all()?.to_vec1::<f32>()?,
            &expected_output.flatten_all()?.to_vec1::<f32>()?,
            3e-5,
        );
        assert_close(
            "large-state final state",
            &actual_state.flatten_all()?.to_vec1::<f32>()?,
            &expected_state.flatten_all()?.to_vec1::<f32>()?,
            3e-5,
        );
        Ok(())
    }
}
