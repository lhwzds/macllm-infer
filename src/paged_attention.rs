use candle_core::{CpuStorage, CustomOp2, DType, Device, Layout, Result, Shape, Storage, Tensor};

use crate::cache::{PagedKvLayerBatchRead, PagedKvLayerRead, PagedKvSlabSnapshot};

#[cfg(feature = "metal")]
const MAX_METAL_SLABS: usize = 1024;
#[cfg(feature = "metal")]
const MAX_METAL_HEAD_DIM: usize = 256;
#[cfg(feature = "metal")]
const MAX_METAL_QUERY_LEN: usize = 128;
#[cfg(feature = "metal")]
const MAX_METAL_LANES: usize = 128;

/// Cheap native-path preflight that does not allocate or upload a page table.
#[cfg(feature = "metal")]
pub(crate) fn metal_attention_supported(query: &Tensor) -> bool {
    let Ok((batch, _, query_len, head_dim)) = query.dims4() else {
        return false;
    };
    let Device::Metal(device) = query.device() else {
        return false;
    };
    batch == 1
        && (1..=MAX_METAL_QUERY_LEN).contains(&query_len)
        && head_dim <= MAX_METAL_HEAD_DIM
        && matches!(query.dtype(), DType::F32 | DType::F16)
        && metal::device_supported(device)
}

/// Cheap native-path preflight for a packed batch. The Metal grid's X axis is
/// the largest per-lane query chunk, so the packed token count itself is not
/// subject to `MAX_METAL_QUERY_LEN`.
#[cfg(feature = "metal")]
pub(crate) fn metal_batch_attention_supported(query: &Tensor, lane_lengths: &[usize]) -> bool {
    let Ok((batch, _, _, head_dim)) = query.dims4() else {
        return false;
    };
    let Device::Metal(device) = query.device() else {
        return false;
    };
    batch == 1
        && !lane_lengths.is_empty()
        && lane_lengths.len() <= MAX_METAL_LANES
        && lane_lengths
            .iter()
            .all(|&length| (1..=MAX_METAL_QUERY_LEN).contains(&length))
        && head_dim <= MAX_METAL_HEAD_DIM
        && matches!(query.dtype(), DType::F32 | DType::F16)
        && metal::device_supported(device)
}

#[cfg(not(feature = "metal"))]
pub(crate) fn metal_batch_attention_supported(_query: &Tensor, _lane_lengths: &[usize]) -> bool {
    false
}

#[cfg(not(feature = "metal"))]
pub(crate) fn metal_attention_supported(_query: &Tensor) -> bool {
    false
}

/// Run full-causal attention directly over the paged K/V pool.
///
/// The immutable read already contains the current query chunk. Query row
/// `row` may observe exactly `query_start + row + 1` keys, so fresh prefill,
/// chunked prefill, and one-token decode share the same direct-page operation.
///
/// Unsupported device/dtype/shape combinations return `None` so the decoder
/// can retain the materialized implementation as a correctness fallback.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn paged_attention(
    query: &Tensor,
    read: &PagedKvLayerRead<'_>,
    query_start: usize,
    scale: f64,
    softcap: Option<f64>,
) -> Result<Option<Tensor>> {
    let Ok((batch, query_heads, query_len, _)) = query.dims4() else {
        return Ok(None);
    };
    if batch != 1 || query_len == 0 {
        return Ok(None);
    }
    validate_read(query, read, query_start, scale, softcap)?;

    let supported = match query.device() {
        Device::Cpu => query.dtype() == DType::F32,
        #[cfg(feature = "metal")]
        Device::Metal(_) => {
            metal_attention_supported(query)
                && read.slabs().len() <= MAX_METAL_SLABS
                && read.page_table().is_contiguous()
                && read
                    .slabs()
                    .iter()
                    .all(|slab| slab.key().is_contiguous() && slab.value().is_contiguous())
        }
        #[allow(unreachable_patterns)]
        _ => false,
    };
    if !supported {
        return Ok(None);
    }

    let query = query.contiguous()?;
    let op = PagedAttentionOp {
        read,
        query_heads,
        query_len,
        query_start,
        scale: scale as f32,
        softcap: softcap.map_or(0.0, |value| value as f32),
    };
    query.apply_op2_no_bwd(read.page_table(), &op).map(Some)
}

/// Run full-causal attention for a packed, ragged request batch in one custom
/// operation. Query tokens are concatenated along axis 2 in lane order. Each
/// lane still owns an independent page range and causal frontier.
///
/// Unsupported device/dtype/shape combinations return `None`, matching the
/// single-lane API's materialized-attention fallback contract.
pub(crate) fn paged_attention_batch(
    query: &Tensor,
    read: &PagedKvLayerBatchRead<'_>,
    start_positions: &[usize],
    lane_lengths: &[usize],
    scale: f64,
    softcap: Option<f64>,
) -> Result<Option<Tensor>> {
    let Ok((batch, query_heads, total_query_len, _)) = query.dims4() else {
        return Ok(None);
    };
    if batch != 1 || total_query_len == 0 {
        return Ok(None);
    }
    validate_batch_read(query, read, start_positions, lane_lengths, scale, softcap)?;

    let supported = match query.device() {
        Device::Cpu => query.dtype() == DType::F32,
        #[cfg(feature = "metal")]
        Device::Metal(_) => {
            metal_batch_attention_supported(query, lane_lengths)
                && read.slabs().len() <= MAX_METAL_SLABS
                && read.page_table().is_contiguous()
                && read
                    .slabs()
                    .iter()
                    .all(|slab| slab.key().is_contiguous() && slab.value().is_contiguous())
        }
        #[allow(unreachable_patterns)]
        _ => false,
    };
    if !supported {
        return Ok(None);
    }

    let query = query.contiguous()?;
    let op = BatchedPagedAttentionOp {
        read,
        query_heads,
        total_query_len,
        start_positions,
        lane_lengths,
        scale: scale as f32,
        softcap: softcap.map_or(0.0, |value| value as f32),
    };
    query.apply_op2_no_bwd(read.page_table(), &op).map(Some)
}

#[allow(clippy::too_many_lines)]
fn validate_batch_read(
    query: &Tensor,
    read: &PagedKvLayerBatchRead<'_>,
    start_positions: &[usize],
    lane_lengths: &[usize],
    scale: f64,
    softcap: Option<f64>,
) -> Result<()> {
    let (_, query_heads, total_query_len, head_dim) = query.dims4()?;
    let lane_count = read.lane_count();
    if lane_count == 0 {
        candle_core::bail!("batched paged attention requires at least one lane");
    }
    for (label, length) in [
        ("start-position", start_positions.len()),
        ("query-length", lane_lengths.len()),
        ("page-offset", read.lane_page_offsets().len()),
        ("page-count", read.lane_page_counts().len()),
        ("logical-length", read.lane_logical_lens().len()),
    ] {
        if length != lane_count {
            candle_core::bail!(
                "batched paged attention has {length} {label} entries for {lane_count} lanes"
            );
        }
    }
    if query.dtype() != read.dtype() {
        candle_core::bail!(
            "batched paged attention query dtype {:?} does not match K/V dtype {:?}",
            query.dtype(),
            read.dtype()
        );
    }
    if query.device().location() != read.page_table().device().location() {
        candle_core::bail!("batched paged attention query and page table are on different devices");
    }
    if head_dim != read.head_dim() {
        candle_core::bail!(
            "batched paged attention query head dimension {head_dim} does not match K/V head dimension {}",
            read.head_dim()
        );
    }
    if query_heads == 0 || read.kv_heads() == 0 || !query_heads.is_multiple_of(read.kv_heads()) {
        candle_core::bail!(
            "batched paged attention query heads {query_heads} are not divisible by {} K/V heads",
            read.kv_heads()
        );
    }
    if read.block_tokens() == 0 {
        candle_core::bail!("batched paged attention requires positive-size blocks");
    }
    if read.layer() >= read.full_layers() {
        candle_core::bail!(
            "batched paged attention layer {} is outside {} full layers",
            read.layer(),
            read.full_layers()
        );
    }
    if read.slabs().is_empty() {
        candle_core::bail!("batched paged attention has no physical slabs");
    }
    let (table_rows, table_width) = read.page_table().dims2()?;
    if read.page_table().dtype() != DType::U32 || table_width != 2 {
        candle_core::bail!("batched paged attention page table must be U32 with shape [pages, 2]");
    }

    let mut packed_query_len = 0_usize;
    for lane in 0..lane_count {
        let query_len = lane_lengths[lane];
        if query_len == 0 {
            candle_core::bail!("batched paged attention lane {lane} has no query tokens");
        }
        let logical_len = read.lane_logical_lens()[lane];
        if logical_len == 0 {
            candle_core::bail!("batched paged attention lane {lane} has empty K/V history");
        }
        let query_end = start_positions[lane]
            .checked_add(query_len)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "batched paged attention lane {lane} query range overflows usize"
                ))
            })?;
        if query_end != logical_len {
            candle_core::bail!(
                "batched paged attention lane {lane} query range {}..{query_end} does not end at K/V logical length {logical_len}",
                start_positions[lane]
            );
        }
        let expected_pages = logical_len.div_ceil(read.block_tokens());
        let page_count = read.lane_page_counts()[lane];
        if page_count != expected_pages {
            candle_core::bail!(
                "batched paged attention lane {lane} has {page_count} pages for logical length {logical_len} and block size {}",
                read.block_tokens()
            );
        }
        let page_end = read.lane_page_offsets()[lane]
            .checked_add(page_count)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "batched paged attention lane {lane} page range overflows usize"
                ))
            })?;
        if page_end > table_rows {
            candle_core::bail!(
                "batched paged attention lane {lane} page range {page_end} exceeds table length {table_rows}"
            );
        }
        packed_query_len = packed_query_len.checked_add(query_len).ok_or_else(|| {
            candle_core::Error::Msg(
                "batched paged attention packed query length overflows usize".into(),
            )
        })?;

        // These fields are serialized into Metal descriptors. Validate them
        // on every backend so oversized fixtures cannot behave differently on
        // CPU and Metal.
        for (label, value) in [
            ("query start", start_positions[lane]),
            ("query length", query_len),
            ("logical length", logical_len),
            ("page offset", read.lane_page_offsets()[lane]),
            ("page count", page_count),
        ] {
            ensure_u32(value, label)?;
        }
    }
    if packed_query_len != total_query_len {
        candle_core::bail!(
            "batched paged attention lane query lengths total {packed_query_len}, but query contains {total_query_len} tokens"
        );
    }
    for (label, value) in [
        ("packed query length", total_query_len),
        ("lane count", lane_count),
        ("block tokens", read.block_tokens()),
        ("full layers", read.full_layers()),
        ("layer", read.layer()),
        ("K/V heads", read.kv_heads()),
        ("query heads", query_heads),
        ("head dimension", read.head_dim()),
        ("slab count", read.slabs().len()),
    ] {
        ensure_u32(value, label)?;
    }
    validate_slab_snapshot(
        query,
        read.slabs(),
        read.full_layers(),
        read.kv_heads(),
        read.block_tokens(),
        read.head_dim(),
        read.dtype(),
        "batched paged attention",
    )?;
    validate_scale(scale, softcap, "batched paged attention")
}

#[cfg_attr(not(test), allow(dead_code))]
fn validate_read(
    query: &Tensor,
    read: &PagedKvLayerRead<'_>,
    query_start: usize,
    scale: f64,
    softcap: Option<f64>,
) -> Result<()> {
    let (_, query_heads, query_len, head_dim) = query.dims4()?;
    if query.dtype() != read.dtype() {
        candle_core::bail!(
            "paged attention query dtype {:?} does not match K/V dtype {:?}",
            query.dtype(),
            read.dtype()
        );
    }
    if query.device().location() != read.page_table().device().location() {
        candle_core::bail!("paged attention query and page table are on different devices");
    }
    if head_dim != read.head_dim() {
        candle_core::bail!(
            "paged attention query head dimension {head_dim} does not match K/V head dimension {}",
            read.head_dim()
        );
    }
    if query_heads == 0 || read.kv_heads() == 0 || !query_heads.is_multiple_of(read.kv_heads()) {
        candle_core::bail!(
            "paged attention query heads {query_heads} are not divisible by {} K/V heads",
            read.kv_heads()
        );
    }
    if read.logical_len() == 0 || read.block_tokens() == 0 {
        candle_core::bail!("paged attention requires non-empty K/V history and positive blocks");
    }
    let query_end = query_start
        .checked_add(query_len)
        .ok_or_else(|| candle_core::Error::Msg("paged attention query range overflow".into()))?;
    if query_end != read.logical_len() {
        candle_core::bail!(
            "paged attention query range {query_start}..{query_end} does not end at K/V logical length {}",
            read.logical_len()
        );
    }
    if read.layer() >= read.full_layers() {
        candle_core::bail!(
            "paged attention layer {} is outside {} full layers",
            read.layer(),
            read.full_layers()
        );
    }
    let expected_pages = read.logical_len().div_ceil(read.block_tokens());
    if read.page_count() != expected_pages {
        candle_core::bail!(
            "paged attention has {} pages for logical length {} and block size {}",
            read.page_count(),
            read.logical_len(),
            read.block_tokens()
        );
    }
    if read.slabs().is_empty() {
        candle_core::bail!("paged attention has no physical slabs");
    }
    let (table_rows, table_width) = read.page_table().dims2()?;
    if read.page_table().dtype() != DType::U32 || table_width != 2 {
        candle_core::bail!("paged attention page table must be U32 with shape [pages, 2]");
    }
    let table_end = read
        .page_offset()
        .checked_add(read.page_count())
        .ok_or_else(|| candle_core::Error::Msg("paged attention page range overflow".into()))?;
    if table_end > table_rows {
        candle_core::bail!(
            "paged attention page range {table_end} exceeds table length {table_rows}"
        );
    }
    validate_slab_snapshot(
        query,
        read.slabs(),
        read.full_layers(),
        read.kv_heads(),
        read.block_tokens(),
        read.head_dim(),
        read.dtype(),
        "paged attention",
    )?;
    validate_scale(scale, softcap, "paged attention")
}

#[allow(clippy::too_many_arguments)]
fn validate_slab_snapshot(
    query: &Tensor,
    slabs: &[PagedKvSlabSnapshot],
    full_layers: usize,
    kv_heads: usize,
    block_tokens: usize,
    head_dim: usize,
    dtype: DType,
    operation: &str,
) -> Result<()> {
    for (index, slab) in slabs.iter().enumerate() {
        if slab.key().dims() != slab.value().dims() {
            candle_core::bail!("{operation} slab {index} has mismatched K/V shapes");
        }
        let dims = slab.key().dims();
        if dims.len() != 5
            || dims[0] == 0
            || dims[1] != full_layers
            || dims[2] != kv_heads
            || dims[3] != block_tokens
            || dims[4] != head_dim
        {
            candle_core::bail!(
                "{operation} slab {index} has incompatible shape {:?}",
                slab.key().shape()
            );
        }
        if slab.key().dtype() != dtype
            || slab.value().dtype() != dtype
            || slab.key().device().location() != query.device().location()
            || slab.value().device().location() != query.device().location()
        {
            candle_core::bail!("{operation} slab {index} has incompatible dtype or device");
        }
    }
    Ok(())
}

fn validate_scale(scale: f64, softcap: Option<f64>, operation: &str) -> Result<()> {
    let scale = scale as f32;
    if !scale.is_finite() || scale <= 0.0 {
        candle_core::bail!("{operation} scale must be finite and positive");
    }
    if let Some(softcap) = softcap {
        let softcap = softcap as f32;
        if !softcap.is_finite() || softcap <= 0.0 {
            candle_core::bail!("{operation} softcap must be finite and positive");
        }
    }
    Ok(())
}

fn ensure_u32(value: usize, label: &str) -> Result<()> {
    u32::try_from(value)
        .map(|_| ())
        .map_err(|_| candle_core::Error::Msg(format!("paged attention {label} exceeds U32")))
}

#[cfg_attr(not(test), allow(dead_code))]
struct PagedAttentionOp<'a> {
    read: &'a PagedKvLayerRead<'a>,
    query_heads: usize,
    query_len: usize,
    query_start: usize,
    scale: f32,
    /// Zero disables softcapping.
    softcap: f32,
}

impl CustomOp2 for PagedAttentionOp<'_> {
    fn name(&self) -> &'static str {
        "paged-attention"
    }

    fn cpu_fwd(
        &self,
        query: &CpuStorage,
        query_layout: &Layout,
        page_table: &CpuStorage,
        table_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(query) = query else {
            candle_core::bail!("CPU paged decode only supports F32 queries");
        };
        let CpuStorage::U32(page_table) = page_table else {
            candle_core::bail!("CPU paged decode requires a U32 page table");
        };
        let slabs = cpu_slabs(self.read.slabs())?;
        let head_dim = self.read.head_dim();
        let groups = self.query_heads / self.read.kv_heads();
        let mut output = vec![0.0_f32; self.query_heads * self.query_len * head_dim];

        for query_head in 0..self.query_heads {
            let kv_head = query_head / groups;
            for query_row in 0..self.query_len {
                let visible_len = self.query_start + query_row + 1;
                let mut maximum = f32::NEG_INFINITY;
                let mut normalizer = 0.0_f32;
                let mut accumulator = vec![0.0_f32; head_dim];
                for position in 0..visible_len {
                    let page = position / self.read.block_tokens();
                    let in_page = position % self.read.block_tokens();
                    let table_row = self.read.page_offset() + page;
                    let slab_index =
                        page_table[strided_offset(table_layout, &[table_row, 0])?] as usize;
                    let slab_block =
                        page_table[strided_offset(table_layout, &[table_row, 1])?] as usize;
                    let slab = slabs.get(slab_index).ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "paged attention page {page} references slab {slab_index}, but only {} slabs exist",
                            slabs.len()
                        ))
                    })?;

                    let mut score = 0.0_f32;
                    for dimension in 0..head_dim {
                        let query_index =
                            strided_offset(query_layout, &[0, query_head, query_row, dimension])?;
                        let key_index = strided_offset(
                            &slab.key_layout,
                            &[slab_block, self.read.layer(), kv_head, in_page, dimension],
                        )?;
                        score += query[query_index] * slab.key[key_index];
                    }
                    score *= self.scale;
                    if self.softcap != 0.0 {
                        score = (score / self.softcap).tanh() * self.softcap;
                    }

                    let next_maximum = maximum.max(score);
                    let old_weight = if maximum.is_finite() {
                        (maximum - next_maximum).exp()
                    } else {
                        0.0
                    };
                    let new_weight = (score - next_maximum).exp();
                    for (dimension, accumulated) in accumulator.iter_mut().enumerate() {
                        let value_index = strided_offset(
                            &slab.value_layout,
                            &[slab_block, self.read.layer(), kv_head, in_page, dimension],
                        )?;
                        *accumulated =
                            *accumulated * old_weight + slab.value[value_index] * new_weight;
                    }
                    normalizer = normalizer * old_weight + new_weight;
                    maximum = next_maximum;
                }
                let destination = (query_head * self.query_len + query_row) * head_dim;
                for (dimension, accumulated) in accumulator.into_iter().enumerate() {
                    output[destination + dimension] = accumulated / normalizer;
                }
            }
        }
        Ok((CpuStorage::F32(output), query_layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        query: &candle_core::MetalStorage,
        query_layout: &Layout,
        page_table: &candle_core::MetalStorage,
        table_layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::forward(self, query, query_layout, page_table, table_layout)
    }
}

struct BatchedPagedAttentionOp<'a> {
    read: &'a PagedKvLayerBatchRead<'a>,
    query_heads: usize,
    total_query_len: usize,
    start_positions: &'a [usize],
    lane_lengths: &'a [usize],
    scale: f32,
    /// Zero disables softcapping.
    softcap: f32,
}

impl CustomOp2 for BatchedPagedAttentionOp<'_> {
    fn name(&self) -> &'static str {
        "batched-paged-attention"
    }

    fn cpu_fwd(
        &self,
        query: &CpuStorage,
        query_layout: &Layout,
        page_table: &CpuStorage,
        table_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(query) = query else {
            candle_core::bail!("CPU batched paged attention only supports F32 queries");
        };
        let CpuStorage::U32(page_table) = page_table else {
            candle_core::bail!("CPU batched paged attention requires a U32 page table");
        };
        let slabs = cpu_slabs(self.read.slabs())?;
        let head_dim = self.read.head_dim();
        let groups = self.query_heads / self.read.kv_heads();
        let mut output = vec![0.0_f32; self.query_heads * self.total_query_len * head_dim];
        let mut query_offset = 0_usize;

        for lane in 0..self.read.lane_count() {
            let query_len = self.lane_lengths[lane];
            let page_offset = self.read.lane_page_offsets()[lane];
            let page_count = self.read.lane_page_counts()[lane];
            for query_head in 0..self.query_heads {
                let kv_head = query_head / groups;
                for lane_query_row in 0..query_len {
                    let query_row = query_offset + lane_query_row;
                    let visible_len = self.start_positions[lane] + lane_query_row + 1;
                    let mut maximum = f32::NEG_INFINITY;
                    let mut normalizer = 0.0_f32;
                    let mut accumulator = vec![0.0_f32; head_dim];
                    for position in 0..visible_len {
                        let page = position / self.read.block_tokens();
                        if page >= page_count {
                            candle_core::bail!(
                                "batched paged attention lane {lane} position {position} exceeds its page range"
                            );
                        }
                        let in_page = position % self.read.block_tokens();
                        let table_row = page_offset + page;
                        let slab_index =
                            page_table[strided_offset(table_layout, &[table_row, 0])?] as usize;
                        let slab_block =
                            page_table[strided_offset(table_layout, &[table_row, 1])?] as usize;
                        let slab = slabs.get(slab_index).ok_or_else(|| {
                            candle_core::Error::Msg(format!(
                                "batched paged attention lane {lane} page {page} references slab {slab_index}, but only {} slabs exist",
                                slabs.len()
                            ))
                        })?;

                        let mut score = 0.0_f32;
                        for dimension in 0..head_dim {
                            let query_index = strided_offset(
                                query_layout,
                                &[0, query_head, query_row, dimension],
                            )?;
                            let key_index = strided_offset(
                                &slab.key_layout,
                                &[slab_block, self.read.layer(), kv_head, in_page, dimension],
                            )?;
                            score += query[query_index] * slab.key[key_index];
                        }
                        score *= self.scale;
                        if self.softcap != 0.0 {
                            score = (score / self.softcap).tanh() * self.softcap;
                        }

                        let next_maximum = maximum.max(score);
                        let old_weight = if maximum.is_finite() {
                            (maximum - next_maximum).exp()
                        } else {
                            0.0
                        };
                        let new_weight = (score - next_maximum).exp();
                        for (dimension, accumulated) in accumulator.iter_mut().enumerate() {
                            let value_index = strided_offset(
                                &slab.value_layout,
                                &[slab_block, self.read.layer(), kv_head, in_page, dimension],
                            )?;
                            *accumulated =
                                *accumulated * old_weight + slab.value[value_index] * new_weight;
                        }
                        normalizer = normalizer * old_weight + new_weight;
                        maximum = next_maximum;
                    }
                    let destination = (query_head * self.total_query_len + query_row) * head_dim;
                    for (dimension, accumulated) in accumulator.into_iter().enumerate() {
                        output[destination + dimension] = accumulated / normalizer;
                    }
                }
            }
            query_offset += query_len;
        }
        Ok((CpuStorage::F32(output), query_layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        query: &candle_core::MetalStorage,
        query_layout: &Layout,
        page_table: &candle_core::MetalStorage,
        table_layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::forward_batch(self, query, query_layout, page_table, table_layout)
    }
}

struct CpuSlab {
    key: Vec<f32>,
    value: Vec<f32>,
    key_layout: Layout,
    value_layout: Layout,
}

fn cpu_slabs(snapshots: &[PagedKvSlabSnapshot]) -> Result<Vec<CpuSlab>> {
    let mut slabs = Vec::with_capacity(snapshots.len());
    for slab in snapshots {
        let (key_guard, key_layout) = slab.key().storage_and_layout();
        let (value_guard, value_layout) = slab.value().storage_and_layout();
        let key = match &*key_guard {
            Storage::Cpu(CpuStorage::F32(values)) => values.clone(),
            _ => candle_core::bail!("CPU paged decode only supports F32 K/V slabs"),
        };
        let value = match &*value_guard {
            Storage::Cpu(CpuStorage::F32(values)) => values.clone(),
            _ => candle_core::bail!("CPU paged decode only supports F32 K/V slabs"),
        };
        slabs.push(CpuSlab {
            key,
            value,
            key_layout: key_layout.clone(),
            value_layout: value_layout.clone(),
        });
    }
    Ok(slabs)
}

fn strided_offset(layout: &Layout, indices: &[usize]) -> Result<usize> {
    if indices.len() != layout.dims().len() {
        candle_core::bail!(
            "paged attention received {} indices for rank {} storage",
            indices.len(),
            layout.dims().len()
        );
    }
    indices
        .iter()
        .zip(layout.dims())
        .zip(layout.stride())
        .try_fold(
            layout.start_offset(),
            |offset, ((&index, &dim), &stride)| {
                if index >= dim {
                    return Err(candle_core::Error::Msg(format!(
                        "paged attention index {index} exceeds dimension {dim}"
                    )));
                }
                index
                    .checked_mul(stride)
                    .and_then(|delta| offset.checked_add(delta))
                    .ok_or_else(|| candle_core::Error::Msg("paged attention index overflow".into()))
            },
        )
}

#[cfg(feature = "metal")]
mod metal {
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex, OnceLock},
    };

    use candle_core::{DType, Layout, MetalStorage, Result, Shape, Storage};
    use candle_metal_kernels::{
        RESOURCE_OPTIONS,
        metal::{ComputePipeline, Library},
    };
    use objc2_metal::{
        MTLArgumentBuffersTier, MTLArgumentEncoder, MTLComputePipelineState, MTLDevice,
        MTLFunction, MTLGPUFamily, MTLSize,
    };

    use super::{BatchedPagedAttentionOp, MAX_METAL_SLABS, PagedAttentionOp};

    const SOURCE: &str = include_str!("metal/paged_decode.metal");
    const ARGUMENT_BUFFER_INDEX: usize = 2;
    const GQA4_GROUPS: usize = 4;
    const GQA4_MAX_HEAD_DIM: usize = 128;
    const GQA4_MIN_QUERY_LEN: usize = 4;
    const GQA4_THREADS: usize = 128;
    const GQA4_KEY_GROUPS: usize = GQA4_THREADS / 32;

    #[cfg(test)]
    static TEST_BATCH_KERNEL_OVERRIDE: std::sync::atomic::AtomicU8 =
        std::sync::atomic::AtomicU8::new(0);

    #[cfg(test)]
    pub(super) struct TestBatchKernelOverride(u8);

    #[cfg(test)]
    impl Drop for TestBatchKernelOverride {
        fn drop(&mut self) {
            TEST_BATCH_KERNEL_OVERRIDE.store(self.0, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[cfg(test)]
    pub(super) fn override_batch_kernel(use_gqa4: bool) -> TestBatchKernelOverride {
        let route = if use_gqa4 { 2 } else { 1 };
        let previous = TEST_BATCH_KERNEL_OVERRIDE.swap(route, std::sync::atomic::Ordering::SeqCst);
        TestBatchKernelOverride(previous)
    }

    #[derive(Debug)]
    struct Pipelines {
        library: Library,
        #[cfg_attr(not(test), allow(dead_code))]
        f32: ComputePipeline,
        #[cfg_attr(not(test), allow(dead_code))]
        f16: ComputePipeline,
        batch_f32: ComputePipeline,
        batch_f16: ComputePipeline,
        batch_gqa4_f32: ComputePipeline,
        batch_gqa4_f16: ComputePipeline,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    #[cfg_attr(not(test), allow(dead_code))]
    struct Params {
        logical_len: u32,
        query_start: u32,
        query_len: u32,
        page_offset: u32,
        page_count: u32,
        block_tokens: u32,
        full_layers: u32,
        layer: u32,
        kv_heads: u32,
        query_heads: u32,
        head_dim: u32,
        groups: u32,
        slab_count: u32,
        scale: f32,
        softcap: f32,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct BatchParams {
        query_len: u32,
        block_tokens: u32,
        full_layers: u32,
        layer: u32,
        kv_heads: u32,
        query_heads: u32,
        head_dim: u32,
        groups: u32,
        slab_count: u32,
        lane_count: u32,
        max_query_len: u32,
        scale: f32,
        softcap: f32,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct LaneDescriptor {
        logical_len: u32,
        query_start: u32,
        query_offset: u32,
        query_len: u32,
        page_offset: u32,
        page_count: u32,
    }

    pub(super) fn device_supported(device: &candle_core::MetalDevice) -> bool {
        let device = device.metal_device().as_ref();
        device.supportsFamily(MTLGPUFamily::Apple1)
            && device.argumentBuffersSupport() == MTLArgumentBuffersTier::Tier2
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn forward(
        op: &PagedAttentionOp<'_>,
        query: &MetalStorage,
        query_layout: &Layout,
        page_table: &MetalStorage,
        table_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        let device = query.device();
        let pipelines = pipelines(device)?;
        let (kernel_name, pipeline) = match query.dtype() {
            DType::F32 => ("paged_attention_f32", &pipelines.f32),
            DType::F16 => ("paged_attention_f16", &pipelines.f16),
            dtype => candle_core::bail!("unsupported Metal paged-attention dtype {dtype:?}"),
        };
        validate_pipeline(pipeline, 256)?;

        let function = pipelines
            .library
            .get_function(kernel_name, None)
            .map_err(candle_core::Error::wrap)?;
        // SAFETY: buffer index 2 is the argument-buffer parameter in both
        // kernels compiled from `paged_decode.metal`.
        let argument_encoder = unsafe {
            function
                .as_ref()
                .newArgumentEncoderWithBufferIndex(ARGUMENT_BUFFER_INDEX)
        };
        let argument_buffer = device
            .metal_device()
            .new_buffer(argument_encoder.encodedLength(), RESOURCE_OPTIONS)
            .map_err(candle_core::Error::wrap)?;
        // SAFETY: the shared argument buffer has exactly the encoded length
        // requested by this function's argument encoder and remains alive
        // through command encoding.
        unsafe {
            argument_encoder.setArgumentBuffer_offset(Some(argument_buffer.as_ref()), 0);
        }
        for (index, slab) in op.read.slabs().iter().enumerate() {
            let (key_storage, _) = slab.key().storage_and_layout();
            let (value_storage, _) = slab.value().storage_and_layout();
            let (Storage::Metal(key), Storage::Metal(value)) = (&*key_storage, &*value_storage)
            else {
                candle_core::bail!("paged Metal attention received a non-Metal slab");
            };
            // SAFETY: the MSL argument-buffer arrays each have
            // `MAX_METAL_SLABS` entries and validation bounds `index` below it.
            unsafe {
                argument_encoder.setBuffer_offset_atIndex(Some(key.buffer().as_ref()), 0, index);
                argument_encoder.setBuffer_offset_atIndex(
                    Some(value.buffer().as_ref()),
                    0,
                    MAX_METAL_SLABS + index,
                );
            }
        }

        let output_count = query_layout.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_count, query.dtype())
            .with_label("paged-attention-output")
            .build()?;
        let params = Params {
            logical_len: as_u32(op.read.logical_len(), "logical length")?,
            query_start: as_u32(op.query_start, "query start")?,
            query_len: as_u32(op.query_len, "query length")?,
            page_offset: as_u32(op.read.page_offset(), "page offset")?,
            page_count: as_u32(op.read.page_count(), "page count")?,
            block_tokens: as_u32(op.read.block_tokens(), "block tokens")?,
            full_layers: as_u32(op.read.full_layers(), "full layers")?,
            layer: as_u32(op.read.layer(), "layer")?,
            kv_heads: as_u32(op.read.kv_heads(), "K/V heads")?,
            query_heads: as_u32(op.query_heads, "query heads")?,
            head_dim: as_u32(op.read.head_dim(), "head dimension")?,
            groups: as_u32(op.query_heads / op.read.kv_heads(), "GQA groups")?,
            slab_count: as_u32(op.read.slabs().len(), "slab count")?,
            scale: op.scale,
            softcap: op.softcap,
        };

        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(pipeline);
        let encoder = encoder.as_ref();
        encoder.set_input_buffer(
            0,
            Some(query.buffer()),
            query_layout.start_offset() * query.dtype().size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(page_table.buffer()),
            table_layout.start_offset() * page_table.dtype().size_in_bytes(),
        );
        encoder.set_input_buffer(ARGUMENT_BUFFER_INDEX, Some(&argument_buffer), 0);
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &params);
        // Slab resources are reached indirectly and therefore are invisible to
        // Candle's buffer tracker. Conservative barriers order the preceding
        // K/V writes and any later reuse around this dispatch.
        encoder.insert_memory_barrier();
        encoder.dispatch_thread_groups(
            MTLSize {
                width: op.query_len,
                height: op.query_heads,
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
            MetalStorage::new(output, device.clone(), output_count, query.dtype()),
            query_layout.shape().clone(),
        ))
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn forward_batch(
        op: &BatchedPagedAttentionOp<'_>,
        query: &MetalStorage,
        query_layout: &Layout,
        page_table: &MetalStorage,
        table_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        let device = query.device();
        let pipelines = pipelines(device)?;
        let (legacy_name, legacy_pipeline, gqa4_name, gqa4_pipeline) = match query.dtype() {
            DType::F32 => (
                "batched_paged_attention_f32",
                &pipelines.batch_f32,
                "batched_paged_attention_gqa4_f32",
                &pipelines.batch_gqa4_f32,
            ),
            DType::F16 => (
                "batched_paged_attention_f16",
                &pipelines.batch_f16,
                "batched_paged_attention_gqa4_f16",
                &pipelines.batch_gqa4_f16,
            ),
            dtype => {
                candle_core::bail!("unsupported Metal batched paged-attention dtype {dtype:?}")
            }
        };
        let gqa4_scratch_bytes = gqa4_scratch_bytes(op.read.head_dim())?;
        let gqa4_shape_supported = op.query_heads / op.read.kv_heads() == GQA4_GROUPS
            && op.read.head_dim() <= GQA4_MAX_HEAD_DIM
            && op
                .lane_lengths
                .iter()
                .all(|&query_len| query_len >= GQA4_MIN_QUERY_LEN)
            && supports_dynamic_threadgroup_memory(device, gqa4_pipeline, gqa4_scratch_bytes);
        let use_gqa4 = select_gqa4_route(gqa4_shape_supported);
        let (kernel_name, pipeline, threads, grid_heads, scratch_bytes) = if use_gqa4 {
            (
                gqa4_name,
                gqa4_pipeline,
                GQA4_THREADS,
                op.read.kv_heads(),
                gqa4_scratch_bytes,
            )
        } else {
            (legacy_name, legacy_pipeline, 256, op.query_heads, 0)
        };
        validate_pipeline(pipeline, threads)?;

        let function = pipelines
            .library
            .get_function(kernel_name, None)
            .map_err(candle_core::Error::wrap)?;
        // SAFETY: buffer index 2 is the argument-buffer parameter in both
        // batched kernels compiled from `paged_decode.metal`.
        let argument_encoder = unsafe {
            function
                .as_ref()
                .newArgumentEncoderWithBufferIndex(ARGUMENT_BUFFER_INDEX)
        };
        let argument_buffer = device
            .metal_device()
            .new_buffer(argument_encoder.encodedLength(), RESOURCE_OPTIONS)
            .map_err(candle_core::Error::wrap)?;
        // SAFETY: the shared argument buffer has exactly the encoded length
        // requested by this function's argument encoder and remains alive
        // through command encoding.
        unsafe {
            argument_encoder.setArgumentBuffer_offset(Some(argument_buffer.as_ref()), 0);
        }
        for (index, slab) in op.read.slabs().iter().enumerate() {
            let (key_storage, _) = slab.key().storage_and_layout();
            let (value_storage, _) = slab.value().storage_and_layout();
            let (Storage::Metal(key), Storage::Metal(value)) = (&*key_storage, &*value_storage)
            else {
                candle_core::bail!("batched paged Metal attention received a non-Metal slab");
            };
            // SAFETY: the MSL argument-buffer arrays each have
            // `MAX_METAL_SLABS` entries and validation bounds `index` below it.
            unsafe {
                argument_encoder.setBuffer_offset_atIndex(Some(key.buffer().as_ref()), 0, index);
                argument_encoder.setBuffer_offset_atIndex(
                    Some(value.buffer().as_ref()),
                    0,
                    MAX_METAL_SLABS + index,
                );
            }
        }

        let mut query_offset = 0_usize;
        let mut lane_descriptors = Vec::with_capacity(op.read.lane_count());
        for lane in 0..op.read.lane_count() {
            let query_len = op.lane_lengths[lane];
            lane_descriptors.push(LaneDescriptor {
                logical_len: as_u32(op.read.lane_logical_lens()[lane], "lane logical length")?,
                query_start: as_u32(op.start_positions[lane], "lane query start")?,
                query_offset: as_u32(query_offset, "lane query offset")?,
                query_len: as_u32(query_len, "lane query length")?,
                page_offset: as_u32(op.read.lane_page_offsets()[lane], "lane page offset")?,
                page_count: as_u32(op.read.lane_page_counts()[lane], "lane page count")?,
            });
            query_offset = query_offset.checked_add(query_len).ok_or_else(|| {
                candle_core::Error::Msg(
                    "batched paged attention query offsets overflow usize".into(),
                )
            })?;
        }
        let max_query_len = op
            .lane_lengths
            .iter()
            .copied()
            .max()
            .ok_or_else(|| candle_core::Error::Msg("empty paged attention batch".into()))?;
        let params = BatchParams {
            query_len: as_u32(op.total_query_len, "packed query length")?,
            block_tokens: as_u32(op.read.block_tokens(), "block tokens")?,
            full_layers: as_u32(op.read.full_layers(), "full layers")?,
            layer: as_u32(op.read.layer(), "layer")?,
            kv_heads: as_u32(op.read.kv_heads(), "K/V heads")?,
            query_heads: as_u32(op.query_heads, "query heads")?,
            head_dim: as_u32(op.read.head_dim(), "head dimension")?,
            groups: as_u32(op.query_heads / op.read.kv_heads(), "GQA groups")?,
            slab_count: as_u32(op.read.slabs().len(), "slab count")?,
            lane_count: as_u32(op.read.lane_count(), "lane count")?,
            max_query_len: as_u32(max_query_len, "maximum lane query length")?,
            scale: op.scale,
            softcap: op.softcap,
        };

        let output_count = query_layout.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_count, query.dtype())
            .with_label("batched-paged-attention-output")
            .build()?;

        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(pipeline);
        let encoder = encoder.as_ref();
        encoder.set_input_buffer(
            0,
            Some(query.buffer()),
            query_layout.start_offset() * query.dtype().size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(page_table.buffer()),
            table_layout.start_offset() * page_table.dtype().size_in_bytes(),
        );
        encoder.set_input_buffer(ARGUMENT_BUFFER_INDEX, Some(&argument_buffer), 0);
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &params);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of_val(lane_descriptors.as_slice()),
            lane_descriptors.as_ptr().cast(),
        );
        if scratch_bytes > 0 {
            encoder.set_threadgroup_memory_length(0, scratch_bytes);
        }
        // One shared slab binding services every lane. Conservative barriers
        // expose preceding K/V writes and protect later allocator reuse.
        encoder.insert_memory_barrier();
        encoder.dispatch_thread_groups(
            MTLSize {
                width: max_query_len,
                height: grid_heads,
                depth: op.read.lane_count(),
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        encoder.insert_memory_barrier();

        Ok((
            MetalStorage::new(output, device.clone(), output_count, query.dtype()),
            query_layout.shape().clone(),
        ))
    }

    fn pipelines(device: &candle_core::MetalDevice) -> Result<Arc<Pipelines>> {
        static CACHE: OnceLock<Mutex<HashMap<u64, Arc<Pipelines>>>> = OnceLock::new();
        let registry_id = device.metal_device().registry_id();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache
            .lock()
            .map_err(|_| candle_core::Error::Msg("paged Metal pipeline cache poisoned".into()))?;
        if let Some(pipelines) = cache.get(&registry_id) {
            return Ok(Arc::clone(pipelines));
        }
        let library = device
            .metal_device()
            .new_library_with_source(SOURCE, None)
            .map_err(candle_core::Error::wrap)?;
        let f32_function = library
            .get_function("paged_attention_f32", None)
            .map_err(candle_core::Error::wrap)?;
        let f16_function = library
            .get_function("paged_attention_f16", None)
            .map_err(candle_core::Error::wrap)?;
        let batch_f32_function = library
            .get_function("batched_paged_attention_f32", None)
            .map_err(candle_core::Error::wrap)?;
        let batch_f16_function = library
            .get_function("batched_paged_attention_f16", None)
            .map_err(candle_core::Error::wrap)?;
        let batch_gqa4_f32_function = library
            .get_function("batched_paged_attention_gqa4_f32", None)
            .map_err(candle_core::Error::wrap)?;
        let batch_gqa4_f16_function = library
            .get_function("batched_paged_attention_gqa4_f16", None)
            .map_err(candle_core::Error::wrap)?;
        let pipelines = Arc::new(Pipelines {
            f32: device
                .metal_device()
                .new_compute_pipeline_state_with_function(&f32_function)
                .map_err(candle_core::Error::wrap)?,
            f16: device
                .metal_device()
                .new_compute_pipeline_state_with_function(&f16_function)
                .map_err(candle_core::Error::wrap)?,
            batch_f32: device
                .metal_device()
                .new_compute_pipeline_state_with_function(&batch_f32_function)
                .map_err(candle_core::Error::wrap)?,
            batch_f16: device
                .metal_device()
                .new_compute_pipeline_state_with_function(&batch_f16_function)
                .map_err(candle_core::Error::wrap)?,
            batch_gqa4_f32: device
                .metal_device()
                .new_compute_pipeline_state_with_function(&batch_gqa4_f32_function)
                .map_err(candle_core::Error::wrap)?,
            batch_gqa4_f16: device
                .metal_device()
                .new_compute_pipeline_state_with_function(&batch_gqa4_f16_function)
                .map_err(candle_core::Error::wrap)?,
            library,
        });
        cache.insert(registry_id, Arc::clone(&pipelines));
        Ok(pipelines)
    }

    fn validate_pipeline(pipeline: &ComputePipeline, required_threads: usize) -> Result<()> {
        if pipeline.max_total_threads_per_threadgroup() < required_threads {
            candle_core::bail!(
                "paged Metal attention requires a {required_threads}-thread threadgroup"
            );
        }
        if pipeline.as_ref().threadExecutionWidth() != 32 {
            candle_core::bail!("paged Metal attention requires a SIMD execution width of 32");
        }
        Ok(())
    }

    fn gqa4_scratch_bytes(head_dim: usize) -> Result<usize> {
        GQA4_KEY_GROUPS
            .checked_mul(GQA4_GROUPS)
            .and_then(|entries| entries.checked_mul(head_dim.checked_add(2)?))
            .and_then(|floats| floats.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "GQA4 paged-attention threadgroup memory size overflow".into(),
                )
            })
    }

    fn supports_dynamic_threadgroup_memory(
        device: &candle_core::MetalDevice,
        pipeline: &ComputePipeline,
        dynamic_bytes: usize,
    ) -> bool {
        pipeline
            .as_ref()
            .staticThreadgroupMemoryLength()
            .checked_add(dynamic_bytes)
            .is_some_and(|required| {
                required <= device.metal_device().as_ref().maxThreadgroupMemoryLength()
            })
    }

    fn select_gqa4_route(shape_supported: bool) -> bool {
        #[cfg(test)]
        {
            match TEST_BATCH_KERNEL_OVERRIDE.load(std::sync::atomic::Ordering::SeqCst) {
                1 => false,
                _ => shape_supported,
            }
        }
        #[cfg(not(test))]
        {
            shape_supported
        }
    }

    fn as_u32(value: usize, label: &str) -> Result<u32> {
        u32::try_from(value).map_err(|_| {
            candle_core::Error::Msg(format!("paged attention {label} exceeds the U32 range"))
        })
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use crate::cache::{
        PagedKvBlockTable, PagedKvGeometry, PagedKvGrowthRequest, PagedKvPoolConfig,
        SharedPagedKvPool,
    };

    use super::{paged_attention, paged_attention_batch};

    #[test]
    fn cpu_paged_decode_row_matches_materialized_gqa_across_slabs() -> Result<()> {
        let device = Device::Cpu;
        let pool = SharedPagedKvPool::new(
            PagedKvPoolConfig::new(2, 1, 4).map_err(candle_core::Error::wrap)?,
            PagedKvGeometry::new(1, 2, 2, DType::F32, &device).map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let mut table =
            PagedKvBlockTable::new(pool.clone(), 8).map_err(candle_core::Error::wrap)?;
        let mut reservation = pool
            .reserve_batch(&[PagedKvGrowthRequest::new(&table, 3)])
            .map_err(candle_core::Error::wrap)?;
        let key = Tensor::from_vec(
            vec![
                1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, // K/V head 0
                -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, // K/V head 1
            ],
            (1, 2, 3, 2),
            &device,
        )?;
        let value = Tensor::from_vec(
            vec![
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, // K/V head 0
                -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, // K/V head 1
            ],
            (1, 2, 3, 2),
            &device,
        )?;
        reservation
            .write_layer(0, 0, &key, &value)
            .map_err(candle_core::Error::wrap)?;
        let (materialized_key, materialized_value) = reservation
            .materialize_layer(0, 0)
            .map_err(candle_core::Error::wrap)?;
        let read = reservation
            .layer_read(0, 0)
            .map_err(candle_core::Error::wrap)?;
        let query = Tensor::from_vec(
            vec![1.0_f32, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
            (1, 4, 1, 2),
            &device,
        )?;

        for softcap in [None, Some(1.5)] {
            let actual = paged_attention(&query, &read, 2, 0.5, softcap)?
                .expect("CPU F32 paged decode is supported");
            let expected = dense_causal_attention(
                &query,
                &materialized_key,
                &materialized_value,
                2,
                2,
                0.5,
                softcap,
            )?;
            let difference = actual
                .sub(&expected)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(difference <= 1e-6, "maximum difference was {difference}");
        }
        let _ = reservation
            .commit(&mut [&mut table])
            .map_err(candle_core::Error::wrap)?;
        Ok(())
    }

    #[test]
    fn cpu_ragged_batch_matches_serial_direct_page_attention() -> Result<()> {
        const KV_HEADS: usize = 2;
        const QUERY_HEADS: usize = 4;
        const HEAD_DIM: usize = 3;
        let device = Device::Cpu;
        let lane_lengths = [2_usize, 5, 3];
        let start_positions = [0_usize; 3];
        let pool = SharedPagedKvPool::new(
            PagedKvPoolConfig::new(2, 2, 12).map_err(candle_core::Error::wrap)?,
            PagedKvGeometry::new(1, KV_HEADS, HEAD_DIM, DType::F32, &device)
                .map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let tables = [
            PagedKvBlockTable::new(pool.clone(), 8).map_err(candle_core::Error::wrap)?,
            PagedKvBlockTable::new(pool.clone(), 8).map_err(candle_core::Error::wrap)?,
            PagedKvBlockTable::new(pool.clone(), 8).map_err(candle_core::Error::wrap)?,
        ];
        let mut reservation = pool
            .reserve_batch(&[
                PagedKvGrowthRequest::new(&tables[0], lane_lengths[0]),
                PagedKvGrowthRequest::new(&tables[1], lane_lengths[1]),
                PagedKvGrowthRequest::new(&tables[2], lane_lengths[2]),
            ])
            .map_err(candle_core::Error::wrap)?;

        let mut queries = Vec::new();
        for (lane, &query_len) in lane_lengths.iter().enumerate() {
            let phase = lane as f32 * 0.375 + 0.25;
            let key = Tensor::from_vec(
                (0..KV_HEADS * query_len * HEAD_DIM)
                    .map(|index| (index as f32 * 0.1375 + phase).sin())
                    .collect::<Vec<_>>(),
                (1, KV_HEADS, query_len, HEAD_DIM),
                &device,
            )?;
            let value = Tensor::from_vec(
                (0..KV_HEADS * query_len * HEAD_DIM)
                    .map(|index| (index as f32 * 0.0875 + phase).cos())
                    .collect::<Vec<_>>(),
                (1, KV_HEADS, query_len, HEAD_DIM),
                &device,
            )?;
            reservation
                .write_layer(lane, 0, &key, &value)
                .map_err(candle_core::Error::wrap)?;
            queries.push(Tensor::from_vec(
                (0..QUERY_HEADS * query_len * HEAD_DIM)
                    .map(|index| (index as f32 * 0.1125 + phase * 0.5).cos())
                    .collect::<Vec<_>>(),
                (1, QUERY_HEADS, query_len, HEAD_DIM),
                &device,
            )?);
        }

        let mut serial_outputs = Vec::new();
        for lane in 0..lane_lengths.len() {
            let read = reservation
                .layer_read(lane, 0)
                .map_err(candle_core::Error::wrap)?;
            serial_outputs.push(
                paged_attention(
                    &queries[lane],
                    &read,
                    start_positions[lane],
                    0.625,
                    Some(1.7),
                )?
                .expect("CPU F32 serial paged attention is supported"),
            );
        }
        let expected = Tensor::cat(&serial_outputs, 2)?;
        let packed_query = Tensor::cat(&queries, 2)?;
        let batch_read = reservation
            .layer_batch_read(0)
            .map_err(candle_core::Error::wrap)?;
        let actual = paged_attention_batch(
            &packed_query,
            &batch_read,
            &start_positions,
            &lane_lengths,
            0.625,
            Some(1.7),
        )?
        .expect("CPU F32 batched paged attention is supported");
        let difference = actual
            .sub(&expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(difference <= 1e-6, "maximum difference was {difference}");
        Ok(())
    }

    #[test]
    fn cpu_direct_pages_match_fresh_chunked_prefill_and_decode() -> Result<()> {
        let device = Device::Cpu;
        let pool = SharedPagedKvPool::new(
            PagedKvPoolConfig::new(2, 1, 8).map_err(candle_core::Error::wrap)?,
            PagedKvGeometry::new(1, 2, 3, DType::F32, &device).map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let mut table =
            PagedKvBlockTable::new(pool.clone(), 16).map_err(candle_core::Error::wrap)?;

        run_cpu_append_case(&pool, &mut table, 3, 0.25, None)?;
        run_cpu_append_case(&pool, &mut table, 4, 0.75, Some(1.25))?;
        run_cpu_append_case(&pool, &mut table, 1, 1.25, Some(1.25))?;
        assert_eq!(table.logical_len(), 8);
        Ok(())
    }

    fn run_cpu_append_case(
        pool: &SharedPagedKvPool,
        table: &mut PagedKvBlockTable,
        query_len: usize,
        phase: f32,
        softcap: Option<f64>,
    ) -> Result<()> {
        const KV_HEADS: usize = 2;
        const QUERY_HEADS: usize = 4;
        const HEAD_DIM: usize = 3;

        let query_start = table.logical_len();
        let device = Device::Cpu;
        let mut reservation = pool
            .reserve_batch(&[PagedKvGrowthRequest::new(table, query_len)])
            .map_err(candle_core::Error::wrap)?;
        let key = Tensor::from_vec(
            (0..KV_HEADS * query_len * HEAD_DIM)
                .map(|index| (index as f32 * 0.1375 + phase).sin())
                .collect::<Vec<_>>(),
            (1, KV_HEADS, query_len, HEAD_DIM),
            &device,
        )?;
        let value = Tensor::from_vec(
            (0..KV_HEADS * query_len * HEAD_DIM)
                .map(|index| (index as f32 * 0.0875 + phase).cos())
                .collect::<Vec<_>>(),
            (1, KV_HEADS, query_len, HEAD_DIM),
            &device,
        )?;
        reservation
            .write_layer(0, 0, &key, &value)
            .map_err(candle_core::Error::wrap)?;
        let (materialized_key, materialized_value) = reservation
            .materialize_layer(0, 0)
            .map_err(candle_core::Error::wrap)?;
        let read = reservation
            .layer_read(0, 0)
            .map_err(candle_core::Error::wrap)?;
        let query = Tensor::from_vec(
            (0..QUERY_HEADS * query_len * HEAD_DIM)
                .map(|index| (index as f32 * 0.1125 + phase * 0.5).cos())
                .collect::<Vec<_>>(),
            (1, QUERY_HEADS, query_len, HEAD_DIM),
            &device,
        )?;

        let actual = paged_attention(&query, &read, query_start, 0.625, softcap)?
            .expect("CPU F32 direct-page attention is supported");
        let expected = dense_causal_attention(
            &query,
            &materialized_key,
            &materialized_value,
            QUERY_HEADS / KV_HEADS,
            query_start,
            0.625,
            softcap,
        )?;
        let difference = actual
            .sub(&expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            difference <= 1e-6,
            "query range {query_start}..{} maximum difference was {difference}",
            query_start + query_len
        );
        let _ = reservation
            .commit(&mut [&mut *table])
            .map_err(candle_core::Error::wrap)?;
        Ok(())
    }

    #[test]
    fn unsupported_cpu_shape_and_dtype_return_none() -> Result<()> {
        let device = Device::Cpu;
        let pool = SharedPagedKvPool::new(
            PagedKvPoolConfig::new(2, 1, 2).map_err(candle_core::Error::wrap)?,
            PagedKvGeometry::new(1, 1, 2, DType::F32, &device).map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let table = PagedKvBlockTable::new(pool.clone(), 4).map_err(candle_core::Error::wrap)?;
        let mut reservation = pool
            .reserve_batch(&[PagedKvGrowthRequest::new(&table, 2)])
            .map_err(candle_core::Error::wrap)?;
        let kv = Tensor::zeros((1, 1, 2, 2), DType::F32, &device)?;
        reservation
            .write_layer(0, 0, &kv, &kv)
            .map_err(candle_core::Error::wrap)?;
        let read = reservation
            .layer_read(0, 0)
            .map_err(candle_core::Error::wrap)?;
        let query = Tensor::zeros((2, 1, 1, 2), DType::F32, &device)?;
        assert!(paged_attention(&query, &read, 1, 1.0, None)?.is_none());

        let pool = SharedPagedKvPool::new(
            PagedKvPoolConfig::new(2, 1, 1).map_err(candle_core::Error::wrap)?,
            PagedKvGeometry::new(1, 1, 2, DType::BF16, &device)
                .map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let table = PagedKvBlockTable::new(pool.clone(), 2).map_err(candle_core::Error::wrap)?;
        let mut reservation = pool
            .reserve_batch(&[PagedKvGrowthRequest::new(&table, 1)])
            .map_err(candle_core::Error::wrap)?;
        let kv = Tensor::zeros((1, 1, 1, 2), DType::BF16, &device)?;
        reservation
            .write_layer(0, 0, &kv, &kv)
            .map_err(candle_core::Error::wrap)?;
        let read = reservation
            .layer_read(0, 0)
            .map_err(candle_core::Error::wrap)?;
        assert!(paged_attention(&kv, &read, 0, 1.0, None)?.is_none());
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an Apple Metal device"]
    fn metal_f32_and_f16_match_dense_attention() -> Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16] {
            metal_case(dtype, &device)?;
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an Apple Metal device"]
    #[allow(clippy::too_many_lines)]
    fn metal_gqa4_d128_f16_matches_legacy_batch() -> Result<()> {
        const LANES: usize = 2;
        const HISTORY_LEN: usize = 257;
        const QUERY_LEN: usize = 16;
        const LOGICAL_LEN: usize = HISTORY_LEN + QUERY_LEN;
        const BLOCK_TOKENS: usize = 16;
        const KV_HEADS: usize = 4;
        const QUERY_HEADS: usize = 16;
        const HEAD_DIM: usize = 128;

        let device = Device::new_metal(0)?;
        let blocks_per_lane = LOGICAL_LEN.div_ceil(BLOCK_TOKENS);
        let pool = SharedPagedKvPool::new(
            PagedKvPoolConfig::new(BLOCK_TOKENS, 8, LANES * blocks_per_lane)
                .map_err(candle_core::Error::wrap)?,
            PagedKvGeometry::new(1, KV_HEADS, HEAD_DIM, DType::F16, &device)
                .map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let tables = (0..LANES)
            .map(|_| {
                PagedKvBlockTable::new(pool.clone(), LOGICAL_LEN).map_err(candle_core::Error::wrap)
            })
            .collect::<Result<Vec<_>>>()?;
        let requests = tables
            .iter()
            .map(|table| PagedKvGrowthRequest::new(table, LOGICAL_LEN))
            .collect::<Vec<_>>();
        let mut reservation = pool
            .reserve_batch(&requests)
            .map_err(candle_core::Error::wrap)?;
        let mut queries = Vec::with_capacity(LANES);
        for lane in 0..LANES {
            let phase = lane as f32 * 0.375 + 0.25;
            let key = Tensor::from_vec(
                (0..KV_HEADS * LOGICAL_LEN * HEAD_DIM)
                    .map(|index| ((index % 4093) as f32 * 0.003_906_25 + phase).sin())
                    .collect::<Vec<_>>(),
                (1, KV_HEADS, LOGICAL_LEN, HEAD_DIM),
                &device,
            )?
            .to_dtype(DType::F16)?;
            let value = Tensor::from_vec(
                (0..KV_HEADS * LOGICAL_LEN * HEAD_DIM)
                    .map(|index| ((index % 4091) as f32 * 0.002_929_687_5 + phase).cos())
                    .collect::<Vec<_>>(),
                (1, KV_HEADS, LOGICAL_LEN, HEAD_DIM),
                &device,
            )?
            .to_dtype(DType::F16)?;
            reservation
                .write_layer(lane, 0, &key, &value)
                .map_err(candle_core::Error::wrap)?;
            queries.push(
                Tensor::from_vec(
                    (0..QUERY_HEADS * QUERY_LEN * HEAD_DIM)
                        .map(|index| (index as f32 * 0.003_125 + phase * 0.5).cos())
                        .collect::<Vec<_>>(),
                    (1, QUERY_HEADS, QUERY_LEN, HEAD_DIM),
                    &device,
                )?
                .to_dtype(DType::F16)?,
            );
        }

        let packed_query = Tensor::cat(&queries, 2)?;
        let batch_read = reservation
            .layer_batch_read(0)
            .map_err(candle_core::Error::wrap)?;
        let start_positions = [HISTORY_LEN; LANES];
        let lane_lengths = [QUERY_LEN; LANES];
        let expected = {
            let _route = super::metal::override_batch_kernel(false);
            paged_attention_batch(
                &packed_query,
                &batch_read,
                &start_positions,
                &lane_lengths,
                1.0 / (HEAD_DIM as f64).sqrt(),
                Some(2.0),
            )?
            .expect("legacy Metal batch kernel is supported")
        };
        let actual = {
            let _route = super::metal::override_batch_kernel(true);
            paged_attention_batch(
                &packed_query,
                &batch_read,
                &start_positions,
                &lane_lengths,
                1.0 / (HEAD_DIM as f64).sqrt(),
                Some(2.0),
            )?
            .expect("GQA4 Metal batch kernel is supported")
        };
        let difference = actual
            .to_dtype(DType::F32)?
            .sub(&expected.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        eprintln!("F16 GQA4-vs-legacy D128 maximum difference: {difference:.8}");
        assert!(
            difference <= 0.003,
            "F16 GQA4 maximum difference {difference} exceeded 0.003"
        );
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an Apple Metal device and allocates a 16K-token K/V history"]
    fn metal_long_multi_slab_history_matches_dense_attention() -> Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16] {
            metal_long_history_case(dtype, &device)?;
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an Apple Metal device and allocates eight 16K-token K/V histories"]
    #[allow(clippy::too_many_lines)]
    fn metal_batched_eight_lane_16k_ragged_gqa4_matches_legacy() -> Result<()> {
        const LANES: usize = 8;
        const HISTORY_LEN: usize = 16 * 1024;
        const LANE_LENGTHS: [usize; LANES] = [30, 28, 29, 29, 28, 28, 34, 30];
        const BLOCK_TOKENS: usize = 16;
        const SLAB_BLOCKS: usize = 64;
        const KV_HEADS: usize = 4;
        const QUERY_HEADS: usize = 16;
        const HEAD_DIM: usize = 128;

        let device = Device::new_metal(0)?;
        let logical_lens = LANE_LENGTHS.map(|query_len| HISTORY_LEN + query_len);
        let total_blocks = logical_lens
            .iter()
            .map(|logical_len| logical_len.div_ceil(BLOCK_TOKENS))
            .sum();
        let pool = SharedPagedKvPool::new(
            PagedKvPoolConfig::new(BLOCK_TOKENS, SLAB_BLOCKS, total_blocks)
                .map_err(candle_core::Error::wrap)?,
            PagedKvGeometry::new(1, KV_HEADS, HEAD_DIM, DType::F16, &device)
                .map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let tables = (0..LANES)
            .map(|lane| {
                PagedKvBlockTable::new(pool.clone(), logical_lens[lane])
                    .map_err(candle_core::Error::wrap)
            })
            .collect::<Result<Vec<_>>>()?;
        let requests = tables
            .iter()
            .zip(logical_lens)
            .map(|(table, logical_len)| PagedKvGrowthRequest::new(table, logical_len))
            .collect::<Vec<_>>();
        let mut reservation = pool
            .reserve_batch(&requests)
            .map_err(candle_core::Error::wrap)?;

        let maximum_logical_len = *logical_lens
            .iter()
            .max()
            .expect("the fixed eight-lane fixture is non-empty");
        let key = Tensor::from_vec(
            (0..KV_HEADS * maximum_logical_len * HEAD_DIM)
                .map(|index| ((index % 65_521) as f32 * 0.000_976_562_5 + 0.25).sin())
                .collect::<Vec<_>>(),
            (1, KV_HEADS, maximum_logical_len, HEAD_DIM),
            &device,
        )?
        .to_dtype(DType::F16)?;
        let value = Tensor::from_vec(
            (0..KV_HEADS * maximum_logical_len * HEAD_DIM)
                .map(|index| ((index % 65_519) as f32 * 0.000_732_421_9 + 0.75).cos())
                .collect::<Vec<_>>(),
            (1, KV_HEADS, maximum_logical_len, HEAD_DIM),
            &device,
        )?
        .to_dtype(DType::F16)?;
        let mut queries = Vec::with_capacity(LANES);
        for (lane, (&query_len, &logical_len)) in
            LANE_LENGTHS.iter().zip(logical_lens.iter()).enumerate()
        {
            let phase = lane as f32 * 0.1875 + 0.25;
            reservation
                .write_layer(
                    lane,
                    0,
                    &key.narrow(2, 0, logical_len)?,
                    &value.narrow(2, 0, logical_len)?,
                )
                .map_err(candle_core::Error::wrap)?;
            queries.push(
                Tensor::from_vec(
                    (0..QUERY_HEADS * query_len * HEAD_DIM)
                        .map(|index| (index as f32 * 0.015_625 + phase).cos())
                        .collect::<Vec<_>>(),
                    (1, QUERY_HEADS, query_len, HEAD_DIM),
                    &device,
                )?
                .to_dtype(DType::F16)?,
            );
        }

        let start_positions = [HISTORY_LEN; LANES];
        let packed_query = Tensor::cat(&queries, 2)?;
        let batch_read = reservation
            .layer_batch_read(0)
            .map_err(candle_core::Error::wrap)?;
        let expected = {
            let _route = super::metal::override_batch_kernel(false);
            paged_attention_batch(
                &packed_query,
                &batch_read,
                &start_positions,
                &LANE_LENGTHS,
                0.125,
                Some(2.0),
            )?
            .expect("legacy Metal F16 batched paged attention is supported")
        };
        let actual = {
            let _route = super::metal::override_batch_kernel(true);
            paged_attention_batch(
                &packed_query,
                &batch_read,
                &start_positions,
                &LANE_LENGTHS,
                0.125,
                Some(2.0),
            )?
            .expect("GQA4 Metal F16 batched paged attention is supported")
        };
        let difference = actual
            .to_dtype(DType::F32)?
            .sub(&expected.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        eprintln!("F16 eight-lane 16K ragged GQA4-vs-legacy maximum difference: {difference:.8}");
        assert!(
            difference <= 0.003,
            "F16 eight-lane 16K ragged maximum difference {difference} exceeded 0.003"
        );
        Ok(())
    }

    /// Benchmark-only microprofile for the exact packed-attention geometry used
    /// by the 8-way 16K prefix demo. This intentionally excludes projection,
    /// rotary embedding, K/V scatter, MLP, and logits work: it measures one
    /// synchronized `paged_attention_batch` layer at qlen 1 and 16.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "benchmark-only: requires Apple Metal and allocates eight 16K-token K/V histories"]
    fn metal_profile_batched_eight_lane_16k_attention() -> Result<()> {
        const WARMUP_RUNS: usize = 3;
        const MEASURED_RUNS: usize = 7;
        const AGENT_SUFFIX_LENGTHS: [usize; 8] = [30, 28, 29, 29, 28, 28, 34, 30];

        let device = Device::new_metal(0)?;
        for (label, lane_lengths) in [
            ("q1", [1_usize; 8]),
            ("q16", [16_usize; 8]),
            ("agent-suffix", AGENT_SUFFIX_LENGTHS),
        ] {
            let routes = metal_profile_batched_eight_lane_16k_case(
                &device,
                lane_lengths,
                WARMUP_RUNS,
                MEASURED_RUNS,
            )?;
            for (route, samples) in routes {
                let mut sorted = samples.clone();
                sorted.sort_by(f64::total_cmp);
                let p50_ms = sorted[sorted.len() / 2];
                eprintln!(
                    "METAL_PAGED_ATTENTION_PROFILE case={label} route={route} lane_lengths={lane_lengths:?} packed_query_len={} lanes=8 history=16384 kv_heads=4 query_heads=16 head_dim=128 dtype=F16 warmups={WARMUP_RUNS} samples_ms={samples:?} p50_ms={p50_ms:.6}",
                    lane_lengths.iter().sum::<usize>()
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_lines)]
    fn metal_profile_batched_eight_lane_16k_case(
        device: &Device,
        lane_lengths: [usize; 8],
        warmup_runs: usize,
        measured_runs: usize,
    ) -> Result<Vec<(&'static str, Vec<f64>)>> {
        const LANES: usize = 8;
        const HISTORY_LEN: usize = 16 * 1024;
        const BLOCK_TOKENS: usize = 16;
        const SLAB_BLOCKS: usize = 64;
        const KV_HEADS: usize = 4;
        const QUERY_HEADS: usize = 16;
        const HEAD_DIM: usize = 128;

        let logical_lens = lane_lengths.map(|query_len| HISTORY_LEN + query_len);
        let total_blocks = logical_lens
            .iter()
            .map(|logical_len| logical_len.div_ceil(BLOCK_TOKENS))
            .sum();
        let pool = SharedPagedKvPool::new(
            PagedKvPoolConfig::new(BLOCK_TOKENS, SLAB_BLOCKS, total_blocks)
                .map_err(candle_core::Error::wrap)?,
            PagedKvGeometry::new(1, KV_HEADS, HEAD_DIM, DType::F16, device)
                .map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let tables = (0..LANES)
            .map(|lane| {
                PagedKvBlockTable::new(pool.clone(), logical_lens[lane])
                    .map_err(candle_core::Error::wrap)
            })
            .collect::<Result<Vec<_>>>()?;
        let requests = tables
            .iter()
            .zip(logical_lens)
            .map(|(table, logical_len)| PagedKvGrowthRequest::new(table, logical_len))
            .collect::<Vec<_>>();
        let mut reservation = pool
            .reserve_batch(&requests)
            .map_err(candle_core::Error::wrap)?;

        // Reusing zero-filled tensors avoids host fixture generation overhead;
        // the kernel still performs the same page traversal, dot products,
        // online softmax, and value accumulation as non-zero model data.
        for (lane, &logical_len) in logical_lens.iter().enumerate() {
            let key = Tensor::zeros((1, KV_HEADS, logical_len, HEAD_DIM), DType::F16, device)?;
            let value = Tensor::zeros((1, KV_HEADS, logical_len, HEAD_DIM), DType::F16, device)?;
            reservation
                .write_layer(lane, 0, &key, &value)
                .map_err(candle_core::Error::wrap)?;
        }
        let packed_query_len = lane_lengths.iter().sum::<usize>();
        let packed_query = Tensor::zeros(
            (1, QUERY_HEADS, packed_query_len, HEAD_DIM),
            DType::F16,
            device,
        )?;
        let batch_read = reservation
            .layer_batch_read(0)
            .map_err(candle_core::Error::wrap)?;
        let start_positions = [HISTORY_LEN; LANES];

        // Drain all slab allocation and scatter work before warmup/timing.
        device.synchronize()?;
        let route_names: &[(&str, bool)] = if lane_lengths.iter().all(|&length| length >= 4) {
            &[("legacy", false), ("gqa4", true)]
        } else {
            &[("legacy", false)]
        };
        let mut route_samples = Vec::with_capacity(route_names.len());
        for &(route, use_gqa4) in route_names {
            let _route = super::metal::override_batch_kernel(use_gqa4);
            for _ in 0..warmup_runs {
                let output = paged_attention_batch(
                    &packed_query,
                    &batch_read,
                    &start_positions,
                    &lane_lengths,
                    1.0 / (HEAD_DIM as f64).sqrt(),
                    None,
                )?
                .expect("Metal F16 batched 16K attention is supported");
                std::hint::black_box(&output);
                device.synchronize()?;
            }

            let mut samples = Vec::with_capacity(measured_runs);
            for run in 0..measured_runs {
                device.synchronize()?;
                let started = std::time::Instant::now();
                let output = paged_attention_batch(
                    &packed_query,
                    &batch_read,
                    &start_positions,
                    &lane_lengths,
                    1.0 / (HEAD_DIM as f64).sqrt(),
                    None,
                )?
                .expect("Metal F16 batched 16K attention is supported");
                std::hint::black_box(&output);
                device.synchronize()?;
                let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
                eprintln!(
                    "METAL_PAGED_ATTENTION_SAMPLE route={route} lane_lengths={lane_lengths:?} run={} elapsed_ms={elapsed_ms:.6}",
                    run + 1
                );
                samples.push(elapsed_ms);
            }
            route_samples.push((route, samples));
        }
        Ok(route_samples)
    }

    #[cfg(feature = "metal")]
    fn metal_case(dtype: DType, device: &Device) -> Result<()> {
        const FIRST_LEN: usize = 9;
        const SECOND_LEN: usize = 11;
        const QUERY_LEN: usize = 5;
        const QUERY_START: usize = SECOND_LEN - QUERY_LEN;

        let pool = SharedPagedKvPool::new(
            PagedKvPoolConfig::new(2, 2, 12).map_err(candle_core::Error::wrap)?,
            PagedKvGeometry::new(1, 1, 64, dtype, device).map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let first = PagedKvBlockTable::new(pool.clone(), 32).map_err(candle_core::Error::wrap)?;
        let second = PagedKvBlockTable::new(pool.clone(), 32).map_err(candle_core::Error::wrap)?;
        let mut reservation = pool
            .reserve_batch(&[
                PagedKvGrowthRequest::new(&first, FIRST_LEN),
                PagedKvGrowthRequest::new(&second, SECOND_LEN),
            ])
            .map_err(candle_core::Error::wrap)?;
        for (lane, len, phase) in [(0, FIRST_LEN, 0.25_f32), (1, SECOND_LEN, 0.75_f32)] {
            let key_values = (0..len * 64)
                .map(|index| (index as f32 * 0.031_25 + phase).sin())
                .collect::<Vec<_>>();
            let value_values = (0..len * 64)
                .map(|index| (index as f32 * 0.023_437_5 + phase).cos())
                .collect::<Vec<_>>();
            let key = Tensor::from_vec(key_values, (1, 1, len, 64), device)?.to_dtype(dtype)?;
            let value = Tensor::from_vec(value_values, (1, 1, len, 64), device)?.to_dtype(dtype)?;
            reservation
                .write_layer(lane, 0, &key, &value)
                .map_err(candle_core::Error::wrap)?;
        }
        let (materialized_key, materialized_value) = reservation
            .materialize_layer(1, 0)
            .map_err(candle_core::Error::wrap)?;
        let read = reservation
            .layer_read(1, 0)
            .map_err(candle_core::Error::wrap)?;
        assert_eq!(read.page_offset(), FIRST_LEN.div_ceil(2));
        assert_eq!(read.logical_len(), SECOND_LEN);
        let query_values = (0..4 * QUERY_LEN * 64)
            .map(|index| (index as f32 * 0.015_625).cos())
            .collect::<Vec<_>>();
        let query =
            Tensor::from_vec(query_values, (1, 4, QUERY_LEN, 64), device)?.to_dtype(dtype)?;
        let actual = paged_attention(&query, &read, QUERY_START, 0.125, Some(2.0))?
            .expect("Metal F32/F16 paged prefill is supported");
        let expected = dense_causal_attention(
            &query,
            &materialized_key,
            &materialized_value,
            4,
            QUERY_START,
            0.125,
            Some(2.0),
        )?;
        let difference = actual
            .to_dtype(DType::F32)?
            .sub(&expected.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let tolerance = if dtype == DType::F16 { 0.003 } else { 0.000_02 };
        eprintln!("{dtype:?} short-history maximum difference: {difference:.8}");
        assert!(
            difference <= tolerance,
            "{dtype:?} maximum difference {difference} exceeded {tolerance}"
        );
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn metal_long_history_case(dtype: DType, device: &Device) -> Result<()> {
        const HISTORY_LEN: usize = 16 * 1024;
        const BLOCK_TOKENS: usize = 16;
        const SLAB_BLOCKS: usize = 64;
        const KV_HEADS: usize = 2;
        const QUERY_HEADS: usize = 4;
        const HEAD_DIM: usize = 64;

        let blocks = HISTORY_LEN.div_ceil(BLOCK_TOKENS);
        let pool = SharedPagedKvPool::new(
            PagedKvPoolConfig::new(BLOCK_TOKENS, SLAB_BLOCKS, blocks)
                .map_err(candle_core::Error::wrap)?,
            PagedKvGeometry::new(1, KV_HEADS, HEAD_DIM, dtype, device)
                .map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let table =
            PagedKvBlockTable::new(pool.clone(), HISTORY_LEN).map_err(candle_core::Error::wrap)?;
        let mut reservation = pool
            .reserve_batch(&[PagedKvGrowthRequest::new(&table, HISTORY_LEN)])
            .map_err(candle_core::Error::wrap)?;
        let key = Tensor::from_vec(
            (0..KV_HEADS * HISTORY_LEN * HEAD_DIM)
                .map(|index| ((index % 65_521) as f32 * 0.000_976_562_5 + 0.25).sin())
                .collect::<Vec<_>>(),
            (1, KV_HEADS, HISTORY_LEN, HEAD_DIM),
            device,
        )?
        .to_dtype(dtype)?;
        let value = Tensor::from_vec(
            (0..KV_HEADS * HISTORY_LEN * HEAD_DIM)
                .map(|index| ((index % 65_519) as f32 * 0.000_732_421_9 + 0.75).cos())
                .collect::<Vec<_>>(),
            (1, KV_HEADS, HISTORY_LEN, HEAD_DIM),
            device,
        )?
        .to_dtype(dtype)?;
        reservation
            .write_layer(0, 0, &key, &value)
            .map_err(candle_core::Error::wrap)?;
        let (materialized_key, materialized_value) = reservation
            .materialize_layer(0, 0)
            .map_err(candle_core::Error::wrap)?;
        let read = reservation
            .layer_read(0, 0)
            .map_err(candle_core::Error::wrap)?;
        assert_eq!(read.logical_len(), HISTORY_LEN);
        assert_eq!(read.page_count(), blocks);
        assert!(
            read.slabs().len() > 1,
            "long-history fixture must span multiple physical slabs"
        );

        for (query_len, softcap) in [(1, None), (3, Some(2.0))] {
            let query_start = HISTORY_LEN - query_len;
            let query = Tensor::from_vec(
                (0..QUERY_HEADS * query_len * HEAD_DIM)
                    .map(|index| (index as f32 * 0.015_625 + query_len as f32 * 0.125).cos())
                    .collect::<Vec<_>>(),
                (1, QUERY_HEADS, query_len, HEAD_DIM),
                device,
            )?
            .to_dtype(dtype)?;
            let actual = paged_attention(&query, &read, query_start, 0.125, softcap)?
                .expect("16K Metal F32/F16 paged attention is supported");
            let expected = dense_causal_attention(
                &query,
                &materialized_key,
                &materialized_value,
                QUERY_HEADS / KV_HEADS,
                query_start,
                0.125,
                softcap,
            )?;
            let difference = actual
                .to_dtype(DType::F32)?
                .sub(&expected.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let tolerance = if dtype == DType::F16 { 0.006 } else { 0.000_1 };
            eprintln!("{dtype:?} 16K qlen={query_len} maximum difference: {difference:.8}");
            assert!(
                difference <= tolerance,
                "{dtype:?} 16K qlen={query_len} maximum difference {difference} exceeded {tolerance}"
            );
        }
        Ok(())
    }

    fn dense_causal_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        groups: usize,
        query_start: usize,
        scale: f64,
        softcap: Option<f64>,
    ) -> Result<Tensor> {
        let key = repeat_kv(key, groups)?.contiguous()?;
        let value = repeat_kv(value, groups)?.contiguous()?;
        let mut scores = query.matmul(&key.transpose(2, 3)?)?.affine(scale, 0.0)?;
        if let Some(softcap) = softcap {
            scores = scores
                .affine(1.0 / softcap, 0.0)?
                .tanh()?
                .affine(softcap, 0.0)?;
        }
        let query_len = query.dim(2)?;
        let key_len = key.dim(2)?;
        if query_start.checked_add(query_len) != Some(key_len) {
            candle_core::bail!(
                "dense reference query range {query_start}..{} does not end at key length {key_len}",
                query_start.saturating_add(query_len)
            );
        }
        let mask = (0..query_len)
            .flat_map(|row| {
                (0..key_len).map(move |position| {
                    if position > query_start + row {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(mask, (1, 1, query_len, key_len), query.device())?
            .to_dtype(scores.dtype())?;
        scores = scores.broadcast_add(&mask)?;
        let dtype = scores.dtype();
        let probabilities =
            candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(dtype)?;
        probabilities.matmul(&value)
    }

    fn repeat_kv(input: &Tensor, groups: usize) -> Result<Tensor> {
        if groups == 1 {
            return Ok(input.clone());
        }
        let (batch, heads, sequence, head_dim) = input.dims4()?;
        input
            .unsqueeze(2)?
            .expand((batch, heads, groups, sequence, head_dim))?
            .reshape((batch, heads * groups, sequence, head_dim))
    }
}
