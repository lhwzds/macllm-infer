//! Optional fused prefill attention through Candle's built-in SDPA kernel.
//!
//! The native paged Metal kernel reads K/V directly from the paged cache and
//! never materializes the history. Candle's fused `sdpa` kernel instead needs
//! contiguous K/V, so this module gathers one lane's history from the paged
//! cache and dispatches the fused kernel for multi-row query chunks. The
//! switch exists so both paths can be measured against real models.

use candle_core::{DType, Device, Result, Tensor};
use clap::ValueEnum;

/// Head dimensions instantiated by Candle's fused SDPA kernel.
const SDPA_HEAD_DIMS: [usize; 8] = [32, 64, 72, 80, 96, 128, 256, 512];

/// Widest F32 head the fused kernel can instantiate within its 32 KB
/// threadgroup-memory budget. A 32-row F32 query tile with a 256-wide head
/// does not fit, so the pipeline cannot be created at all.
const SDPA_F32_MAX_HEAD_DIM: usize = 128;

/// Query rows per fused-kernel query tile.
const SDPA_QUERY_TILE: usize = 32;

/// Query rows per fused-kernel query tile of the 512-wide head variant.
const SDPA_WIDE_QUERY_TILE: usize = 8;

/// Key rows per fused-kernel key tile of the narrow heads.
const SDPA_NARROW_KEY_TILE: usize = 32;

/// Key rows per fused-kernel key tile of the wide heads.
const SDPA_WIDE_KEY_TILE: usize = 16;

/// Additive mask value that removes a key from the softmax. It is finite in
/// F32, so the F16 mask becomes a negative infinity and the kernel's
/// exponential underflows every masked score to zero.
const MASK_NEGATIVE: f64 = -1.0e30;

/// Select the attention implementation used for multi-row prefill chunks.
///
/// Single-token decode always keeps the native paged kernel; only chunks with
/// more than one query row can reach the fused kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum PrefillAttentionMode {
    /// Native paged Metal kernel, with the dense Candle path as its fallback.
    Native,
    /// Candle's fused SDPA kernel over the materialized K/V history.
    #[default]
    Sdpa,
}

impl PrefillAttentionMode {
    /// Report whether this mode dispatches multi-row chunks to the fused
    /// kernel.
    #[must_use]
    pub fn uses_fused_kernel(self) -> bool {
        matches!(self, Self::Sdpa)
    }
}

/// Report whether Candle's fused SDPA kernel can serve this query chunk.
///
/// The kernel is Metal-only, requires a GQA-compatible head layout and cannot
/// express logit softcapping. Every unsupported case keeps the caller's
/// existing attention path.
#[must_use]
pub fn sdpa_supported(query: &Tensor, softcap: Option<f64>) -> bool {
    if softcap.is_some() {
        return false;
    }
    let Ok((batch, query_heads, query_len, head_dim)) = query.dims4() else {
        return false;
    };
    if batch != 1 || query_len < 2 || query_heads == 0 {
        return false;
    }
    if !SDPA_HEAD_DIMS.contains(&head_dim) {
        return false;
    }
    if query.dtype() == DType::F32 && head_dim > SDPA_F32_MAX_HEAD_DIM {
        return false;
    }
    if !matches!(query.dtype(), DType::F16 | DType::F32 | DType::BF16) {
        return false;
    }
    match query.device() {
        #[cfg(feature = "metal")]
        Device::Metal(_) => true,
        _ => false,
    }
}

/// Run one lane's full-causal attention through Candle's fused SDPA kernel.
///
/// `key` and `value` hold the lane's whole history (`0..key_len`), while
/// `query` holds the trailing `query_len` rows of that history. The kernel
/// derives the query offset from the key length, so its causal diagonal is
/// aligned to the bottom-right of the (query, key) rectangle: query row `i`
/// observes key positions `0..=key_len - query_len + i`.
///
/// The kernel's own diagonal has two limits. It applies the causal test to
/// the trailing key tiles of every query tile, which is exact only when the
/// chunk fills whole query tiles. It also derives the number of key tiles it
/// walks from the *padded* query length, so a chunk that does not fill whole
/// query tiles makes the kernel walk one key tile past the end of the K/V
/// buffer and read memory outside the history.
///
/// Every geometry that is not a whole number of query tiles is therefore
/// served with an explicit additive mask instead: with a mask the kernel
/// walks exactly the key tiles of the buffer, and the mask encodes the same
/// bottom-right diagonal.
pub fn sdpa_causal_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    let (_, query_heads, query_len, head_dim) = query.dims4()?;
    let (_, key_heads, key_len, key_head_dim) = key.dims4()?;
    let (_, _, value_len, value_head_dim) = value.dims4()?;
    if key_len < query_len {
        candle_core::bail!(
            "sdpa key length {key_len} cannot cover a query chunk of {query_len} rows"
        );
    }
    if key_head_dim != head_dim || value_head_dim != head_dim || value_len != key_len {
        candle_core::bail!("sdpa key, value and query must share one head dimension");
    }
    if key_heads == 0 || query_heads % key_heads != 0 {
        candle_core::bail!("sdpa query heads must be a multiple of the key heads");
    }
    if key.dtype() != query.dtype() || value.dtype() != query.dtype() {
        candle_core::bail!("sdpa query, key and value must share one dtype");
    }
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    // The in-kernel diagonal is only exact for whole query tiles, and the
    // kernel must never walk past the last key tile of the K/V buffer.
    let in_kernel_causal = in_kernel_causal_is_exact(query_len, head_dim)
        && sdpa_kernel_reads_within_key_buffer(query_len, key_len, head_dim);
    let (mask, do_causal) = if in_kernel_causal {
        (None, true)
    } else {
        let mask = causal_mask(
            query_heads,
            query_len,
            key_len,
            query.dtype(),
            query.device(),
        )?;
        (Some(mask), false)
    };
    candle_nn::ops::sdpa(
        &query,
        &key,
        &value,
        mask.as_ref(),
        do_causal,
        scale as f32,
        1.0,
    )
}

/// Report whether the kernel's in-kernel causal diagonal is exact for this
/// geometry.
///
/// The kernel pads the query chunk to whole query tiles and derives its key
/// tile walk from that padded length, so the diagonal is exact only when the
/// chunk fills whole query tiles. The caller must also keep the kernel
/// inside the K/V buffer, which [`sdpa_kernel_reads_within_key_buffer`]
/// reports.
#[must_use]
pub fn in_kernel_causal_is_exact(query_len: usize, head_dim: usize) -> bool {
    query_len.is_multiple_of(sdpa_query_tile(head_dim))
}

/// Query rows per fused-kernel query tile for one head dimension.
#[must_use]
pub fn sdpa_query_tile(head_dim: usize) -> usize {
    if head_dim == 512 {
        SDPA_WIDE_QUERY_TILE
    } else {
        SDPA_QUERY_TILE
    }
}

/// Key rows per fused-kernel key tile for one head dimension.
#[must_use]
pub fn sdpa_key_tile(head_dim: usize) -> usize {
    if head_dim == 512 {
        SDPA_WIDE_QUERY_TILE
    } else if head_dim < 128 {
        SDPA_NARROW_KEY_TILE
    } else {
        SDPA_WIDE_KEY_TILE
    }
}

/// Report how many key tiles the fused kernel walks for one query chunk.
///
/// The kernel dispatches one threadgroup per query tile and walks the history
/// in key tiles. The last query tile walks the widest range, up to the end
/// of the history, so a chunk that does not fill whole query tiles makes the
/// kernel read one key tile past the end of the K/V buffer.
#[must_use]
pub fn sdpa_kernel_key_tiles(query_len: usize, key_len: usize, head_dim: usize) -> usize {
    let query_tile = sdpa_query_tile(head_dim);
    let key_tile = sdpa_key_tile(head_dim);
    let padded_query = query_len.div_ceil(query_tile) * query_tile;
    let first_visible = key_len.saturating_sub(query_len);
    first_visible
        .saturating_add(padded_query)
        .div_ceil(key_tile)
}

/// Report whether the fused kernel stays inside the K/V buffer.
///
/// The kernel's key-tile walk must not reach past the last key tile of the
/// buffer, otherwise it reads memory that does not belong to the history.
#[must_use]
pub fn sdpa_kernel_reads_within_key_buffer(
    query_len: usize,
    key_len: usize,
    head_dim: usize,
) -> bool {
    sdpa_kernel_key_tiles(query_len, key_len, head_dim) <= key_len.div_ceil(sdpa_key_tile(head_dim))
}

/// Build the additive causal mask for the trailing `query_len` rows of a
/// `key_len`-long history.
///
/// The mask is built on the device so a long prefill never pays for a host
/// mask and a host-to-device copy per layer. The fused kernel indexes the
/// mask by query head, so one shared mask is broadcast across heads instead
/// of being copied per head.
fn causal_mask(
    query_heads: usize,
    query_len: usize,
    key_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let first_visible = key_len - query_len;
    let shape = (query_len, key_len);
    let key_positions = Tensor::arange(0_u32, key_len as u32, device)?
        .unsqueeze(0)?
        .broadcast_as(shape)?;
    let query_positions = Tensor::arange(first_visible as u32, key_len as u32, device)?
        .unsqueeze(1)?
        .broadcast_as(shape)?;
    // A key is hidden from a query row when it starts after that row, so the
    // diagonal runs along the bottom-right of the (query, key) rectangle.
    let hidden = key_positions.gt(&query_positions)?;
    let mask = hidden
        .to_dtype(DType::F32)?
        .affine(MASK_NEGATIVE, 0.0)?
        .to_dtype(dtype)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    if query_heads == 1 {
        Ok(mask)
    } else {
        mask.expand((1, query_heads, query_len, key_len))
    }
}

#[cfg(test)]
mod support_tests {
    use super::*;
    use candle_core::Device;

    /// A query chunk the fused kernel cannot serve must keep the caller's
    /// existing attention path.
    #[test]
    fn unsupported_queries_are_rejected_before_dispatch() {
        let device = Device::Cpu;
        let query = Tensor::zeros((1, 4, 8, 128), DType::F32, &device).expect("query");
        assert!(
            !sdpa_supported(&query, Some(30.0)),
            "softcapping is unsupported"
        );

        let single_row = Tensor::zeros((1, 4, 1, 128), DType::F32, &device).expect("query");
        assert!(
            !sdpa_supported(&single_row, None),
            "decode keeps the native path"
        );

        let odd_head = Tensor::zeros((1, 4, 8, 96 + 1), DType::F32, &device).expect("query");
        assert!(
            !sdpa_supported(&odd_head, None),
            "unknown head dimensions are unsupported"
        );

        let wide_f32 = Tensor::zeros((1, 4, 8, 256), DType::F32, &device).expect("query");
        assert!(
            !sdpa_supported(&wide_f32, None),
            "wide F32 heads do not fit"
        );

        assert!(
            !sdpa_supported(&query, None),
            "the CPU backend has no fused kernel"
        );
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::architecture::AttentionKind;
    use crate::cache::{
        PagedKvBlockTable, PagedKvGeometry, PagedKvGrowthRequest, PagedKvPoolConfig,
        SharedPagedKvPool,
    };
    use crate::decoder::{attention_mask_for_range, dense_causal_attention};

    /// Return the first Metal device, or `None` when this host has none.
    fn metal_device() -> Option<Device> {
        match Device::new_metal(0) {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping Metal prefill-attention test: {error}");
                None
            }
        }
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

    /// Largest absolute difference between two tensors.
    fn max_difference(left: &Tensor, right: &Tensor) -> Result<f32> {
        left.to_dtype(DType::F32)?
            .sub(&right.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()
    }

    /// The three attention results of one geometry.
    struct Geometry {
        /// Result of the fused SDPA kernel.
        fused: Tensor,
        /// Result of the native paged Metal kernel, when it serves the chunk.
        native: Option<Tensor>,
        /// Result of the dense fallback over the materialized history.
        dense: Tensor,
    }

    /// Write random K/V into a paged cache and compute the fused, native and
    /// dense results for one geometry.
    ///
    /// Returns `None` when the fused kernel cannot serve this geometry.
    fn geometry_results(
        device: &Device,
        dtype: DType,
        head_dim: usize,
        kv_heads: usize,
        query_heads: usize,
        history: usize,
        query_len: usize,
    ) -> Result<Option<Geometry>> {
        const BLOCK_TOKENS: usize = 16;
        let total = history + query_len;
        let pool = SharedPagedKvPool::new(
            PagedKvPoolConfig::new(BLOCK_TOKENS, 8, total.div_ceil(BLOCK_TOKENS) + 1)
                .map_err(candle_core::Error::wrap)?,
            PagedKvGeometry::new(1, kv_heads, head_dim, dtype, device)
                .map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let mut table =
            PagedKvBlockTable::new(pool.clone(), total).map_err(candle_core::Error::wrap)?;

        // Append the history first so the measured chunk is written on top of
        // a non-empty lane, exactly like a chunked prefill.
        if history > 0 {
            let mut reservation = pool
                .reserve_batch(&[PagedKvGrowthRequest::new(&table, history)])
                .map_err(candle_core::Error::wrap)?;
            let key = Tensor::from_vec(
                pseudo_random_values(kv_heads * history * head_dim, 11),
                (1, kv_heads, history, head_dim),
                device,
            )?
            .to_dtype(dtype)?;
            let value = Tensor::from_vec(
                pseudo_random_values(kv_heads * history * head_dim, 29),
                (1, kv_heads, history, head_dim),
                device,
            )?
            .to_dtype(dtype)?;
            reservation
                .write_layer(0, 0, &key, &value)
                .map_err(candle_core::Error::wrap)?;
            reservation
                .commit(&mut [&mut table])
                .map_err(candle_core::Error::wrap)?;
        }

        let mut reservation = pool
            .reserve_batch(&[PagedKvGrowthRequest::new(&table, query_len)])
            .map_err(candle_core::Error::wrap)?;
        let key = Tensor::from_vec(
            pseudo_random_values(kv_heads * query_len * head_dim, 43),
            (1, kv_heads, query_len, head_dim),
            device,
        )?
        .to_dtype(dtype)?;
        let value = Tensor::from_vec(
            pseudo_random_values(kv_heads * query_len * head_dim, 59),
            (1, kv_heads, query_len, head_dim),
            device,
        )?
        .to_dtype(dtype)?;
        reservation
            .write_layer(0, 0, &key, &value)
            .map_err(candle_core::Error::wrap)?;
        let (materialized_key, materialized_value) = reservation
            .materialize_layer(0, 0)
            .map_err(candle_core::Error::wrap)?;
        assert_eq!(materialized_key.dim(2)?, total);
        assert_eq!(materialized_value.dim(2)?, total);
        let query = Tensor::from_vec(
            pseudo_random_values(query_heads * query_len * head_dim, 71),
            (1, query_heads, query_len, head_dim),
            device,
        )?
        .to_dtype(dtype)?;
        if !sdpa_supported(&query, None) {
            return Ok(None);
        }
        let scale = 1.0 / (head_dim as f64).sqrt();
        let fused = sdpa_causal_attention(&query, &materialized_key, &materialized_value, scale)?;
        // The native paged kernel reads the history straight from the paged
        // cache, so it is the reference for the chunks it accepts. The
        // decoder's own preflight decides which chunks those are.
        let native = if crate::paged_attention::metal_attention_supported(&query) {
            let read = reservation
                .layer_read(0, 0)
                .map_err(candle_core::Error::wrap)?;
            crate::paged_attention::paged_attention(&query, &read, history, scale, None)?
        } else {
            None
        };
        // The dense path is the mathematical reference implementation: the
        // materialized history, the host-built causal mask and the F32
        // softmax.
        let mask = attention_mask_for_range(
            AttentionKind::FullCausal,
            history,
            query_len,
            0,
            total,
            dtype,
            device,
        )?;
        let dense = dense_causal_attention(
            &query,
            &materialized_key,
            &materialized_value,
            query_heads / kv_heads,
            scale,
            None,
            mask.as_ref(),
        )?;
        Ok(Some(Geometry {
            fused,
            native,
            dense,
        }))
    }

    /// The expected output of a zero-query attention over unit value rows.
    ///
    /// Every score is identical, so each output row is the mean of the value
    /// rows it may observe. `first_visible` is the absolute position of the
    /// first query row, and the diagonal grows by one row per query row.
    fn expected_mean_over_identity(
        query_len: usize,
        head_dim: usize,
        first_visible: usize,
    ) -> Vec<f32> {
        let mut values = vec![0.0_f32; query_len * head_dim];
        for row in 0..query_len {
            let visible = first_visible + row + 1;
            for column in 0..head_dim {
                if column < visible {
                    values[row * head_dim + column] = 1.0 / visible as f32;
                }
            }
        }
        values
    }

    /// Candle's `do_causal` mask must align the causal diagonal to the
    /// bottom-right of the (query, key) rectangle, so a trailing query chunk
    /// observes the whole history it was appended to.
    ///
    /// The probe fills whole kernel query tiles and whole kernel key tiles.
    /// A partial query tile makes the kernel walk one key tile past the end of
    /// the K/V buffer, so its result cannot decide the alignment.

    #[test]
    fn metal_sdpa_causal_diagonal_is_bottom_right() -> Result<()> {
        const QUERY_LEN: usize = 32;
        const KEY_LEN: usize = 64;
        const HEAD_DIM: usize = 64;
        let Some(device) = metal_device() else {
            return Ok(());
        };
        assert!(
            sdpa_kernel_reads_within_key_buffer(QUERY_LEN, KEY_LEN, HEAD_DIM),
            "the causal probe geometry must stay inside the K/V buffer"
        );
        assert!(
            in_kernel_causal_is_exact(QUERY_LEN, HEAD_DIM),
            "the causal probe geometry must fill whole query tiles"
        );
        // Zero queries make every score identical, so each output row is the
        // mean of the value rows it may observe. Distinct unit vectors keep
        // the bottom-right and top-left diagonals trivially distinguishable.
        let query = Tensor::zeros((1, 1, QUERY_LEN, HEAD_DIM), DType::F32, &device)?;
        let key = Tensor::zeros((1, 1, KEY_LEN, HEAD_DIM), DType::F32, &device)?;
        let mut values = vec![0.0_f32; KEY_LEN * HEAD_DIM];
        for row in 0..KEY_LEN {
            values[row * HEAD_DIM + row] = 1.0;
        }
        let value = Tensor::from_vec(values, (1, 1, KEY_LEN, HEAD_DIM), &device)?;
        let causal = candle_nn::ops::sdpa(&query, &key, &value, None, true, 1.0, 1.0)?;
        let mask = causal_mask(1, QUERY_LEN, KEY_LEN, DType::F32, &device)?;
        let masked = candle_nn::ops::sdpa(&query, &key, &value, Some(&mask), false, 1.0, 1.0)?;
        let difference = max_difference(&causal, &masked)?;
        assert!(
            difference <= 1e-6,
            "Candle's do_causal mask must align the causal diagonal to the bottom-right, \
             but it differed from the explicit bottom-right mask by {difference}"
        );
        // The explicit bottom-right mask must produce the mean over the
        // history the chunk was appended to, and the top-left diagonal must
        // produce a different answer.
        let bottom_right = expected_mean_over_identity(QUERY_LEN, HEAD_DIM, KEY_LEN - QUERY_LEN);
        let bottom_right = Tensor::from_vec(bottom_right, (1, 1, QUERY_LEN, HEAD_DIM), &device)?;
        assert!(max_difference(&masked, &bottom_right)? <= 1e-6);
        let top_left = expected_mean_over_identity(QUERY_LEN, HEAD_DIM, 0);
        let top_left = Tensor::from_vec(top_left, (1, 1, QUERY_LEN, HEAD_DIM), &device)?;
        assert!(
            max_difference(&causal, &top_left)? > 1e-3,
            "the causal probe must distinguish the bottom-right diagonal from the top-left one"
        );
        Ok(())
    }

    /// A query chunk that does not fill whole kernel query tiles must use the
    /// explicit additive mask, because the in-kernel diagonal would make the
    /// kernel read past the end of the K/V buffer.
    #[test]
    fn metal_sdpa_incomplete_tiles_use_the_explicit_mask() -> Result<()> {
        const QUERY_LEN: usize = 17;
        const KEY_LEN: usize = 64;
        const HEAD_DIM: usize = 64;
        let Some(device) = metal_device() else {
            return Ok(());
        };
        assert!(
            !in_kernel_causal_is_exact(QUERY_LEN, HEAD_DIM),
            "a partial query tile cannot use the in-kernel diagonal"
        );
        assert!(
            !sdpa_kernel_reads_within_key_buffer(QUERY_LEN, KEY_LEN, HEAD_DIM),
            "a partial query tile walks one key tile past the K/V buffer"
        );
        assert!(
            !sdpa_kernel_reads_within_key_buffer(2, 4, HEAD_DIM),
            "a two-row chunk over a four-row history walks one key tile past the K/V buffer"
        );
        let query = Tensor::from_vec(
            pseudo_random_values(QUERY_LEN * HEAD_DIM, 13),
            (1, 1, QUERY_LEN, HEAD_DIM),
            &device,
        )?;
        let key = Tensor::from_vec(
            pseudo_random_values(KEY_LEN * HEAD_DIM, 17),
            (1, 1, KEY_LEN, HEAD_DIM),
            &device,
        )?;
        let value = Tensor::from_vec(
            pseudo_random_values(KEY_LEN * HEAD_DIM, 19),
            (1, 1, KEY_LEN, HEAD_DIM),
            &device,
        )?;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let actual = sdpa_causal_attention(&query, &key, &value, scale)?;
        let mask = causal_mask(1, QUERY_LEN, KEY_LEN, DType::F32, &device)?;
        let expected = dense_causal_attention(&query, &key, &value, 1, scale, None, Some(&mask))?;
        assert!(
            max_difference(&actual, &expected)? <= 1e-6,
            "the explicit mask must reproduce the dense causal attention"
        );
        Ok(())
    }

    #[test]
    fn metal_sdpa_prefill_matches_reference_paths_across_geometries() -> Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        for dtype in [DType::F16, DType::F32] {
            for head_dim in [128_usize, 256] {
                let mut compared = 0_usize;
                let mut native_compared = 0_usize;
                for (kv_heads, query_heads) in [(1_usize, 1_usize), (1, 4), (1, 8)] {
                    for query_len in [2_usize, 17, 128, 300] {
                        for history in [0_usize, 5, 333] {
                            let Some(geometry) = geometry_results(
                                &device,
                                dtype,
                                head_dim,
                                kv_heads,
                                query_heads,
                                history,
                                query_len,
                            )?
                            else {
                                continue;
                            };
                            compared += 1;
                            let tolerance = if dtype == DType::F16 { 2e-3 } else { 1e-4 };
                            // Chunks of at most 128 query rows are served by
                            // the native paged Metal kernel, longer chunks
                            // fall back to the dense path.
                            let reference = match &geometry.native {
                                Some(native) => {
                                    native_compared += 1;
                                    native
                                }
                                None => &geometry.dense,
                            };
                            let difference = max_difference(&geometry.fused, reference)?;
                            assert!(
                                difference <= tolerance,
                                "{dtype:?} head_dim={head_dim} kv_heads={kv_heads} \
                                 query_heads={query_heads} query_len={query_len} history={history} \
                                 differed by {difference} (tolerance {tolerance})"
                            );
                            // The dense path is the mathematical reference for
                            // every geometry.
                            let dense_difference =
                                max_difference(&geometry.fused, &geometry.dense)?;
                            assert!(
                                dense_difference <= tolerance,
                                "{dtype:?} head_dim={head_dim} kv_heads={kv_heads} \
                                 query_heads={query_heads} query_len={query_len} history={history} \
                                 differed by {dense_difference} from the dense path \
                                 (tolerance {tolerance})"
                            );
                        }
                    }
                }
                if dtype == DType::F32 && head_dim > SDPA_F32_MAX_HEAD_DIM {
                    // The fused kernel cannot instantiate an F32 256-wide head.
                    assert_eq!(compared, 0);
                    assert_eq!(native_compared, 0);
                } else {
                    assert_eq!(compared, 36);
                    // Three of the four chunk lengths fit the native kernel.
                    assert_eq!(native_compared, 27);
                }
            }
        }
        Ok(())
    }
}
