//! File-backed GGUF quantized weights for Metal.
//!
//! Candle's ordinary quantized loader copies every weight into an anonymous
//! `MTLBuffer`. On unified-memory Macs that makes the whole model dirty memory.
//! This module instead wraps one private, read-only file mapping per GGUF shard
//! in an external shared Metal buffer and binds each tensor at its file offset.

use std::{
    ffi::c_void,
    fs::File,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use block2::RcBlock;
use candle_core::{
    CpuStorage, CustomOp1, DType, Device as CandleDevice, Error, Layout, MetalDevice, MetalStorage,
    Result, Shape, Tensor, backend::BackendStorage, quantized::GgmlDType,
};
use candle_metal_kernels::{
    Kernels, MetalKernelError, Output, RESOURCE_OPTIONS, debug_group,
    metal::{Buffer, ComputeCommandEncoder, Device},
    set_params,
    source::Source,
    utils::EncoderProvider,
};
use memmap2::MmapOptions;
use objc2_metal::{MTLDevice as _, MTLSize};

/// A single GGUF shard exposed to Metal without copying its bytes.
#[derive(Debug)]
pub(crate) struct FileBackedMetalShard {
    buffer: Arc<Buffer>,
    device: MetalDevice,
    file_len: usize,
}

impl FileBackedMetalShard {
    /// Attempt to wrap a complete shard in a no-copy Metal buffer.
    ///
    /// `Ok(None)` means the device cannot represent this shard and callers
    /// should use the ordinary copied quantized storage path.
    pub(crate) fn try_new(
        file: &File,
        file_len: u64,
        device: &MetalDevice,
    ) -> Result<Option<Self>> {
        if file_len == 0 {
            return Ok(None);
        }
        let file_len = usize::try_from(file_len)
            .map_err(|_| Error::Msg("GGUF shard length does not fit usize".into()))?;
        let page_size = system_page_size()?;
        let buffer_len = file_len
            .checked_add(page_size - 1)
            .map(|length| length / page_size * page_size)
            .ok_or_else(|| Error::Msg("page-rounded GGUF shard length overflows".into()))?;
        if buffer_len > device.device().as_ref().maxBufferLength() {
            return Ok(None);
        }

        // MAP_PRIVATE keeps the checkpoint file authoritative and makes clean
        // pages reclaimable. Mapping from offset zero guarantees a page-aligned
        // base address; the kernel rounds the final VM page even when EOF does
        // not land on a page boundary.
        // SAFETY: the caller's GGUF contract keeps the underlying inode
        // immutable and untruncated for every model using this mapping.
        let mapping = unsafe { MmapOptions::new().len(buffer_len).map_copy_read_only(file) }
            .map_err(|error| Error::Msg(format!("failed to map GGUF shard for Metal: {error}")))?;
        let pointer = NonNull::new(mapping.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| Error::Msg("GGUF mapping returned a null pointer".into()))?;
        if !(pointer.as_ptr() as usize).is_multiple_of(page_size) {
            return Err(Error::Msg(format!(
                "GGUF mapping is not aligned to the {page_size}-byte VM page size"
            )));
        }

        // The Metal object can be retained by an in-flight command buffer after
        // the Rust weight object is dropped. Let Metal's final deallocator own
        // the mmap so `munmap` cannot race GPU reads. The callback is specified
        // to run once; the Option also makes a duplicate callback harmless.
        let mapping = Mutex::new(Some(mapping));
        let deallocator: RcBlock<dyn Fn(NonNull<c_void>, usize)> = RcBlock::new(
            move |_pointer: NonNull<c_void>, _length: usize| match mapping.lock() {
                Ok(mut mapping) => drop(mapping.take()),
                Err(poisoned) => drop(poisoned.into_inner().take()),
            },
        );
        // SAFETY: `pointer` is page aligned, `buffer_len` covers whole VM pages
        // in one mapping, the captured mmap outlives the returned MTLBuffer, and
        // the callback's captured Mutex is safe to invoke on any Metal thread.
        let raw = unsafe {
            device
                .device()
                .as_ref()
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    pointer,
                    buffer_len,
                    RESOURCE_OPTIONS,
                    Some(&deallocator),
                )
        };
        let Some(raw) = raw else {
            return Ok(None);
        };
        let buffer = Buffer::new(raw);
        buffer.set_label("macllm_file_backed_gguf_shard");
        Ok(Some(Self {
            buffer: Arc::new(buffer),
            device: device.clone(),
            file_len,
        }))
    }
}

fn system_page_size() -> Result<usize> {
    // SAFETY: sysconf has no pointer arguments and `_SC_PAGESIZE` is supported
    // by every Unix target on which Candle's Metal backend can run.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(Error::Msg("failed to query the VM page size".into()));
    }
    usize::try_from(page_size).map_err(|_| Error::Msg("VM page size does not fit usize".into()))
}

/// A rank-two block-quantized tensor residing at an offset in a shard buffer.
#[derive(Debug)]
pub struct FileBackedQuantizedTensor {
    shard: Arc<FileBackedMetalShard>,
    dtype: GgmlDType,
    shape: Shape,
    byte_offset: usize,
    byte_len: usize,
}

impl FileBackedQuantizedTensor {
    pub(crate) fn new(
        shard: Arc<FileBackedMetalShard>,
        dtype: GgmlDType,
        shape: impl Into<Shape>,
        byte_offset: u64,
        byte_len: usize,
    ) -> Result<Self> {
        let shape = shape.into();
        if shape.rank() != 2 {
            return Err(Error::Msg(format!(
                "file-backed quantized tensor must be rank two, found {shape:?}"
            )));
        }
        let byte_offset = usize::try_from(byte_offset)
            .map_err(|_| Error::Msg("quantized tensor offset does not fit usize".into()))?;
        let byte_end = byte_offset
            .checked_add(byte_len)
            .ok_or_else(|| Error::Msg("quantized tensor byte range overflows".into()))?;
        if byte_end > shard.file_len {
            return Err(Error::Msg(format!(
                "quantized tensor range {byte_offset}..{byte_end} exceeds its {}-byte shard",
                shard.file_len
            )));
        }
        if !byte_offset.is_multiple_of(2) {
            return Err(Error::Msg(format!(
                "quantized tensor offset {byte_offset} is not two-byte aligned"
            )));
        }
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(dtype.block_size()) {
            return Err(Error::Msg(format!(
                "quantized tensor element count {elem_count} is not divisible by {:?} block size {}",
                dtype,
                dtype.block_size()
            )));
        }
        let expected_len = elem_count
            .checked_div(dtype.block_size())
            .and_then(|blocks| blocks.checked_mul(dtype.type_size()))
            .ok_or_else(|| Error::Msg("quantized tensor storage length overflows".into()))?;
        if byte_len != expected_len {
            return Err(Error::Msg(format!(
                "quantized tensor has {byte_len} bytes, expected {expected_len} for shape {shape:?} and {dtype:?}"
            )));
        }
        Ok(Self {
            shard,
            dtype,
            shape,
            byte_offset,
            byte_len,
        })
    }

    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }

    pub(crate) fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub(crate) fn matmul(&self, input: &Tensor) -> Result<Tensor> {
        input.apply_op1_no_bwd(&FileBackedMatMul { weight: self })
    }

    pub(crate) fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        let (_, hidden) = self.shape.dims2()?;
        let mut output_shape = ids.dims().to_vec();
        output_shape.push(hidden);
        let ids = ids
            .to_device(&CandleDevice::Metal(self.shard.device.clone()))?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .contiguous()?;
        ids.apply_op1_no_bwd(&FileBackedEmbedding { weight: self })?
            .reshape(output_shape)
    }

    #[cfg(test)]
    pub(crate) fn byte_range(&self) -> std::ops::Range<usize> {
        self.byte_offset..self.byte_offset + self.byte_len
    }
}

struct FileBackedMatMul<'a> {
    weight: &'a FileBackedQuantizedTensor,
}

impl CustomOp1 for FileBackedMatMul<'_> {
    fn name(&self) -> &'static str {
        "file-backed-quantized-matmul"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        Err(Error::Msg(
            "file-backed quantized weights require Metal".into(),
        ))
    }

    #[allow(clippy::too_many_lines)]
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let weight = self.weight;
        if storage.device().id() != weight.shard.device.id() {
            return Err(Error::Msg(
                "file-backed weight and input use different Metal devices".into(),
            ));
        }
        if storage.dtype() != DType::F32 {
            return Err(Error::Msg(format!(
                "file-backed quantized matmul expects F32 input, found {:?}",
                storage.dtype()
            )));
        }
        if !layout.is_contiguous() {
            return Err(Error::Msg(format!(
                "file-backed quantized matmul requires contiguous input, found {layout:?}"
            )));
        }
        let source_shape = layout.shape();
        if source_shape.rank() < 2 {
            return Err(Error::Msg(format!(
                "file-backed quantized matmul input has rank {}",
                source_shape.rank()
            )));
        }
        let (output_rows, input_columns) = weight.shape.dims2()?;
        let mut output_dims = source_shape.dims().to_vec();
        if output_dims.pop() != Some(input_columns) {
            return Err(Error::Msg(format!(
                "input shape {source_shape:?} is incompatible with weight shape {:?}",
                weight.shape
            )));
        }
        output_dims.push(output_rows);
        let output_shape = Shape::from(output_dims);

        let device = storage.device().clone();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_shape.elem_count(), DType::F32)
            .with_label("file_backed_qmatmul")
            .build()?;
        let encoder = device.command_encoder()?;

        if source_shape.dim(source_shape.rank() - 2)? == 1 {
            let batch_rows = match source_shape.rank() {
                3 => source_shape.dims()[0] * source_shape.dims()[1],
                2 => source_shape.dims()[0],
                rank => {
                    return Err(Error::Msg(format!(
                        "file-backed quantized matrix-vector input has unsupported rank {rank}"
                    )));
                }
            };
            for batch in 0..batch_rows {
                call_quantized_matmul_mv_t_offset(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    weight.dtype.into(),
                    (1, 1, output_rows, input_columns),
                    storage.buffer(),
                    (layout.start_offset() + batch * input_columns)
                        * storage.dtype().size_in_bytes(),
                    &weight.shard.buffer,
                    weight.byte_offset,
                    batch * output_rows * DType::F32.size_in_bytes(),
                    &output,
                )
                .map_err(candle_core::MetalError::from)?;
            }
        } else {
            if source_shape.rank() > 4 {
                return Err(Error::Msg(format!(
                    "file-backed quantized matmul input rank {} exceeds four",
                    source_shape.rank()
                )));
            }
            let weight_layout = Layout::contiguous(
                [
                    vec![1; 4 - weight.shape.rank()],
                    weight.shape.dims().to_vec(),
                ]
                .concat(),
            );
            let weight_stride = weight_layout
                .stride()
                .iter()
                .map(|stride| {
                    stride
                        .checked_mul(weight.dtype.type_size())
                        .map(|bytes| bytes / weight.dtype.block_size())
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| Error::Msg("quantized Metal weight stride overflows".into()))?;
            let input_layout = Layout::contiguous(
                [
                    vec![1; 4 - source_shape.rank()],
                    source_shape.dims().to_vec(),
                ]
                .concat(),
            );
            let input_stride = input_layout
                .stride()
                .iter()
                .map(|stride| stride * DType::F32.size_in_bytes())
                .collect::<Vec<_>>();
            call_quantized_matmul_mm_t_offset(
                device.device(),
                &encoder,
                device.kernels(),
                weight.dtype.into(),
                weight_layout.dims(),
                &weight_stride,
                &weight.shard.buffer,
                weight.byte_offset,
                input_layout.dims(),
                &input_stride,
                storage.buffer(),
                layout.start_offset() * storage.dtype().size_in_bytes(),
                output_shape.dims(),
                0,
                &output,
            )
            .map_err(candle_core::MetalError::from)?;
        }

        drop(encoder);
        Ok((
            MetalStorage::new(output, device, output_shape.elem_count(), DType::F32),
            output_shape,
        ))
    }
}

struct FileBackedEmbedding<'a> {
    weight: &'a FileBackedQuantizedTensor,
}

impl CustomOp1 for FileBackedEmbedding<'_> {
    fn name(&self) -> &'static str {
        "file-backed-quantized-embedding"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        Err(Error::Msg(
            "file-backed quantized weights require Metal".into(),
        ))
    }

    fn metal_fwd(&self, ids: &MetalStorage, ids_layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let weight = self.weight;
        if ids.device().id() != weight.shard.device.id() {
            return Err(Error::Msg(
                "file-backed embedding and ids use different Metal devices".into(),
            ));
        }
        if ids.dtype() != DType::U32 || !ids_layout.is_contiguous() {
            return Err(Error::Msg(
                "file-backed quantized embedding requires contiguous U32 ids".into(),
            ));
        }
        let (rows, hidden) = weight.shape.dims2()?;
        if !hidden.is_multiple_of(weight.dtype.block_size()) {
            return Err(Error::Msg(format!(
                "embedding width {hidden} is not divisible by {:?} block size {}",
                weight.dtype,
                weight.dtype.block_size()
            )));
        }
        let expected_len = rows * hidden * weight.dtype.type_size() / weight.dtype.block_size();
        if weight.byte_len != expected_len {
            return Err(Error::Msg(format!(
                "file-backed embedding has {} bytes, expected {expected_len}",
                weight.byte_len
            )));
        }
        let ids_len = ids_layout.shape().elem_count();
        let device = ids.device().clone();
        let output = device
            .new_buffer_builder()
            .with_size_for(ids_len * hidden, DType::F32)
            .with_label("file_backed_qembedding")
            .build()?;
        let encoder = device.command_encoder()?;
        call_quantized_get_rows_offset(
            device.device(),
            &encoder,
            device.kernels(),
            weight.dtype.into(),
            hidden,
            hidden * weight.dtype.type_size() / weight.dtype.block_size(),
            ids_len,
            &weight.shard.buffer,
            weight.byte_offset,
            ids.buffer(),
            ids_layout.start_offset() * DType::U32.size_in_bytes(),
            &output,
        )
        .map_err(candle_core::MetalError::from)?;
        drop(encoder);
        let shape = Shape::from((ids_len, hidden));
        Ok((
            MetalStorage::new(output, device, shape.elem_count(), DType::F32),
            shape,
        ))
    }
}

// Candle 0.11 exposes the GGML kernels, but its wrappers always bind weights
// at byte offset zero. These three wrappers intentionally mirror Candle's
// argument layout while adding one validated weight/source offset.

#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn call_quantized_matmul_mv_t_offset(
    device: &Device,
    encoder_provider: impl EncoderProvider,
    kernels: &Kernels,
    dtype: candle_metal_kernels::GgmlDType,
    (batch, rows, output_columns, input_columns): (usize, usize, usize, usize),
    input: &Buffer,
    input_offset: usize,
    weight: &Buffer,
    weight_offset: usize,
    output_offset: usize,
    output: &Buffer,
) -> std::result::Result<(), MetalKernelError> {
    let ne00 = input_columns as i64;
    let ne01 = output_columns as i64;
    let ne02 = batch as i64;
    let ne03 = 1_i64;
    let ne10 = input_columns as i64;
    let ne11 = rows as i64;
    let ne12 = batch as i64;
    let ne13 = 1_i64;
    let ne0 = output_columns as i64;
    let ne1 = rows as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    let (thread_width, thread_height, row_alignment) = match dtype {
        candle_metal_kernels::GgmlDType::Q4_0
        | candle_metal_kernels::GgmlDType::Q4_1
        | candle_metal_kernels::GgmlDType::Q5_0
        | candle_metal_kernels::GgmlDType::Q5_1
        | candle_metal_kernels::GgmlDType::Q8_0
        | candle_metal_kernels::GgmlDType::Q8_1 => (8, 8, 8),
        candle_metal_kernels::GgmlDType::Q2K
        | candle_metal_kernels::GgmlDType::Q3K
        | candle_metal_kernels::GgmlDType::Q5K => (2, 32, 4),
        candle_metal_kernels::GgmlDType::Q4K => (4, 8, 4),
        candle_metal_kernels::GgmlDType::Q6K => (2, 32, 2),
        candle_metal_kernels::GgmlDType::F16
        | candle_metal_kernels::GgmlDType::BF16
        | candle_metal_kernels::GgmlDType::Q8K
        | candle_metal_kernels::GgmlDType::F32 => (32, 1, 8),
    };
    let kernel_name = match dtype {
        candle_metal_kernels::GgmlDType::Q4_0 => "kernel_mul_mv_q4_0_f32",
        candle_metal_kernels::GgmlDType::Q4_1 => "kernel_mul_mv_q4_1_f32",
        candle_metal_kernels::GgmlDType::Q5_0 => "kernel_mul_mv_q5_0_f32",
        candle_metal_kernels::GgmlDType::Q5_1 => "kernel_mul_mv_q5_1_f32",
        candle_metal_kernels::GgmlDType::Q8_0 => "kernel_mul_mv_q8_0_f32",
        candle_metal_kernels::GgmlDType::Q8_1 => "kernel_mul_mv_q8_1_f32",
        candle_metal_kernels::GgmlDType::Q2K => "kernel_mul_mv_q2_K_f32",
        candle_metal_kernels::GgmlDType::Q3K => "kernel_mul_mv_q3_K_f32",
        candle_metal_kernels::GgmlDType::Q4K => "kernel_mul_mv_q4_K_f32",
        candle_metal_kernels::GgmlDType::Q5K => "kernel_mul_mv_q5_K_f32",
        candle_metal_kernels::GgmlDType::Q6K => "kernel_mul_mv_q6_K_f32",
        candle_metal_kernels::GgmlDType::Q8K => "kernel_mul_mv_q8_K_f32",
        candle_metal_kernels::GgmlDType::F16 => "kernel_mul_mv_f16_f32",
        candle_metal_kernels::GgmlDType::BF16 => "kernel_mul_mv_bf16_f32",
        candle_metal_kernels::GgmlDType::F32 => "kernel_mul_mv_f32_f32",
    };
    let pipeline = kernels.load_pipeline(device, Source::Quantized, kernel_name)?;
    let encoder = encoder_provider.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "file_qmm_mv {kernel_name} B={batch} M={rows} K={input_columns} N={output_columns}"
    );
    set_params!(
        encoder,
        (
            (weight, weight_offset),
            (input, input_offset),
            Output::with_offset(output, output_offset),
            ne00,
            ne01,
            ne02,
            0_i64,
            0_i64,
            0_i64,
            ne10,
            ne11,
            ne12,
            0_i64,
            0_i64,
            0_i64,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: output_columns.div_ceil(row_alignment),
            height: rows,
            depth: (ne12 * ne13) as usize,
        },
        MTLSize {
            width: thread_width,
            height: thread_height,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn call_quantized_matmul_mm_t_offset(
    device: &Device,
    encoder_provider: impl EncoderProvider,
    kernels: &Kernels,
    dtype: candle_metal_kernels::GgmlDType,
    weight_shape: &[usize],
    weight_stride: &[usize],
    weight: &Buffer,
    weight_offset: usize,
    input_shape: &[usize],
    input_stride: &[usize],
    input: &Buffer,
    input_offset: usize,
    output_shape: &[usize],
    output_offset: usize,
    output: &Buffer,
) -> std::result::Result<(), MetalKernelError> {
    let ne00 = weight_shape[weight_shape.len() - 1] as i64;
    let ne01 = weight_shape[weight_shape.len() - 2] as i64;
    let ne02 = weight_shape[weight_shape.len() - 3] as i64;
    let ne03 = weight_shape[weight_shape.len() - 4] as i64;
    let nb01 = weight_stride[weight_stride.len() - 2] as i64;
    let nb02 = weight_stride[weight_stride.len() - 3] as i64;
    let nb03 = weight_stride[weight_stride.len() - 4] as i64;
    let ne11 = input_shape[input_shape.len() - 2] as i64;
    let ne12 = input_shape[input_shape.len() - 3] as i64;
    let ne13 = input_shape[input_shape.len() - 4] as i64;
    let nb10 = input_stride[input_stride.len() - 1] as i64;
    let nb11 = input_stride[input_stride.len() - 2] as i64;
    let nb12 = input_stride[input_stride.len() - 3] as i64;
    let nb13 = input_stride[input_stride.len() - 4] as i64;
    let ne0 = output_shape[output_shape.len() - 1] as i64;
    let ne1 = output_shape[output_shape.len() - 2] as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    let kernel_name = match dtype {
        candle_metal_kernels::GgmlDType::Q4_0 => "kernel_mul_mm_q4_0_f32",
        candle_metal_kernels::GgmlDType::Q4_1 => "kernel_mul_mm_q4_1_f32",
        candle_metal_kernels::GgmlDType::Q5_0 => "kernel_mul_mm_q5_0_f32",
        candle_metal_kernels::GgmlDType::Q5_1 => "kernel_mul_mm_q5_1_f32",
        candle_metal_kernels::GgmlDType::Q8_0 => "kernel_mul_mm_q8_0_f32",
        candle_metal_kernels::GgmlDType::Q2K => "kernel_mul_mm_q2_K_f32",
        candle_metal_kernels::GgmlDType::Q3K => "kernel_mul_mm_q3_K_f32",
        candle_metal_kernels::GgmlDType::Q4K => "kernel_mul_mm_q4_K_f32",
        candle_metal_kernels::GgmlDType::Q5K => "kernel_mul_mm_q5_K_f32",
        candle_metal_kernels::GgmlDType::Q6K => "kernel_mul_mm_q6_K_f32",
        candle_metal_kernels::GgmlDType::F16 => "kernel_mul_mm_f16_f32",
        candle_metal_kernels::GgmlDType::BF16 => "kernel_mul_mm_bf16_f32",
        candle_metal_kernels::GgmlDType::F32 => "kernel_mul_mm_f32_f32",
        candle_metal_kernels::GgmlDType::Q8_1 => {
            return Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "qmatmul"));
        }
        candle_metal_kernels::GgmlDType::Q8K => {
            return Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "qmatmul"));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Quantized, kernel_name)?;
    let encoder = encoder_provider.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "file_qmm_mm {kernel_name} M={ne11} K={ne00} N={ne01}"
    );
    set_params!(
        encoder,
        (
            (weight, weight_offset),
            (input, input_offset),
            Output::with_offset(output, output_offset),
            ne00,
            ne02,
            nb01,
            nb02,
            nb03,
            ne12,
            nb10,
            nb11,
            nb12,
            nb13,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.set_threadgroup_memory_length(0, 8192);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: (ne11 as usize).div_ceil(32),
            height: (ne01 as usize).div_ceil(64),
            depth: (ne12 * ne13) as usize,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn call_quantized_get_rows_offset(
    device: &Device,
    encoder_provider: impl EncoderProvider,
    kernels: &Kernels,
    dtype: candle_metal_kernels::GgmlDType,
    hidden_size: usize,
    row_stride: usize,
    ids_len: usize,
    weight: &Buffer,
    weight_offset: usize,
    ids: &Buffer,
    ids_offset: usize,
    output: &Buffer,
) -> std::result::Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        candle_metal_kernels::GgmlDType::F32 => "kernel_get_rows_f32",
        candle_metal_kernels::GgmlDType::F16 => "kernel_get_rows_f16",
        candle_metal_kernels::GgmlDType::BF16 => "kernel_get_rows_bf16",
        candle_metal_kernels::GgmlDType::Q4_0 => "kernel_get_rows_q4_0",
        candle_metal_kernels::GgmlDType::Q4_1 => "kernel_get_rows_q4_1",
        candle_metal_kernels::GgmlDType::Q5_0 => "kernel_get_rows_q5_0",
        candle_metal_kernels::GgmlDType::Q5_1 => "kernel_get_rows_q5_1",
        candle_metal_kernels::GgmlDType::Q8_0 => "kernel_get_rows_q8_0",
        candle_metal_kernels::GgmlDType::Q2K => "kernel_get_rows_q2_K",
        candle_metal_kernels::GgmlDType::Q3K => "kernel_get_rows_q3_K",
        candle_metal_kernels::GgmlDType::Q4K => "kernel_get_rows_q4_K",
        candle_metal_kernels::GgmlDType::Q5K => "kernel_get_rows_q5_K",
        candle_metal_kernels::GgmlDType::Q6K => "kernel_get_rows_q6_K",
        candle_metal_kernels::GgmlDType::Q8_1 => {
            return Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "get_rows"));
        }
        candle_metal_kernels::GgmlDType::Q8K => {
            return Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "get_rows"));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Quantized, kernel_name)?;
    let encoder = encoder_provider.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "file_qget_rows {kernel_name} ids={ids_len} hidden={hidden_size}"
    );
    set_params!(
        encoder,
        (
            (weight, weight_offset),
            (ids, ids_offset),
            Output::new(output),
            hidden_size as i64,
            row_stride as u64,
            0_u64,
            ids_len as i64,
            core::mem::size_of::<u32>() as u64,
            0_u64,
            (hidden_size * core::mem::size_of::<f32>()) as u64,
            0_u64
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: ids_len,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
