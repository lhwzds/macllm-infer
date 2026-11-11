use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use candle_core::DType;
use tokenizers::Tokenizer;

use crate::{
    architecture::CausalLm,
    backend::{DTypeKind, DeviceKind},
    cache::PagedKvPoolConfig,
    models::ModelAdapter,
    prefill_attention::PrefillAttentionMode,
    weights::{ModelFiles, ModelFormat},
};

/// Model-loading options shared by the CLI and long-lived serving workers.
#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    pub model: PathBuf,
    pub device: DeviceKind,
    pub dtype: Option<DTypeKind>,
    /// Tokens per logical full-attention K/V block.
    pub kv_cache_block_tokens: usize,
    /// Logical blocks grouped into one pair of physical K/V slab tensors.
    pub kv_cache_slab_blocks: usize,
    /// Model-wide token capacity shared by all sequences. `None` uses one
    /// declared model context across the entire pool, independent of the
    /// serving active-sequence limit.
    pub kv_cache_capacity_tokens: Option<usize>,
    /// Maximum resident automatic-prefix snapshot cost in logical tokens.
    /// Zero keeps blocking run/bench commands cold by default.
    pub prefix_cache_capacity_tokens: usize,
    /// Attention implementation used for multi-row prefill chunks.
    pub prefill_attention: PrefillAttentionMode,
}

impl ModelLoadConfig {
    pub const DEFAULT_KV_CACHE_BLOCK_TOKENS: usize = 16;
    pub const DEFAULT_KV_CACHE_SLAB_BLOCKS: usize = 8;
    /// CUDA has no paged-attention kernel, so every decode step takes the
    /// dense fallback and copies each K/V block once per layer per step.
    /// Larger blocks keep that copy count down; see
    /// `bench/2026-11-10-cuda-kv-blocks.md` in the work folder.
    ///
    /// The prefix cache reuses whole blocks, so on CUDA a prompt shorter
    /// than one block gets no prefix reuse, and every live sequence holds at
    /// least one block.
    pub const CUDA_KV_CACHE_BLOCK_TOKENS: usize = 256;
    /// CUDA has no paged-attention kernel, so every decode step takes the
    /// dense fallback and copies each K/V block once per layer per step.
    /// A slab of four 256-token blocks pools the allocation without raising
    /// the per-step copy count; see `bench/2026-11-10-cuda-kv-blocks.md` in
    /// the work folder.
    pub const CUDA_KV_CACHE_SLAB_BLOCKS: usize = 4;
    pub const DEFAULT_SERVER_PREFIX_CACHE_CAPACITY_TOKENS: usize = 32 * 1024;

    /// The K/V block and slab geometry that suits `device`.
    ///
    /// CPU and Metal keep the small blocks of the paged Metal kernel, while
    /// CUDA uses the larger blocks of its dense fallback.
    #[must_use]
    pub fn default_kv_cache_geometry(device: DeviceKind) -> (usize, usize) {
        match device {
            DeviceKind::Cuda => (
                Self::CUDA_KV_CACHE_BLOCK_TOKENS,
                Self::CUDA_KV_CACHE_SLAB_BLOCKS,
            ),
            DeviceKind::Cpu | DeviceKind::Metal => (
                Self::DEFAULT_KV_CACHE_BLOCK_TOKENS,
                Self::DEFAULT_KV_CACHE_SLAB_BLOCKS,
            ),
        }
    }

    #[must_use]
    pub fn new(model: impl Into<PathBuf>, device: DeviceKind, dtype: Option<DTypeKind>) -> Self {
        let (kv_cache_block_tokens, kv_cache_slab_blocks) = Self::default_kv_cache_geometry(device);
        Self {
            model: model.into(),
            device,
            dtype,
            kv_cache_block_tokens,
            kv_cache_slab_blocks,
            kv_cache_capacity_tokens: None,
            prefix_cache_capacity_tokens: 0,
            prefill_attention: PrefillAttentionMode::default(),
        }
    }
}

/// A fully loaded runtime. Construct it on the thread that will own and use
/// the model; neither the model nor its backend is required to be `Send`.
pub struct LoadedRuntime {
    pub adapter: Box<dyn ModelAdapter>,
    pub tokenizer: Arc<Tokenizer>,
    pub model: Box<dyn CausalLm>,
    pub dtype: DTypeKind,
}

impl LoadedRuntime {
    /// Load the tokenizer, adapter and weights exactly once.
    pub fn load(config: &ModelLoadConfig) -> Result<Self> {
        let files = ModelFiles::from_dir(&config.model)?;
        let dtype = resolve_dtype(config.device, config.dtype, &files)?;
        let adapter = files.adapter()?;
        adapter.validate()?;
        let tokenizer = files.tokenizer()?;
        // Embedded GGUF token IDs are validated against the model vocabulary
        // while the tokenizer is built. Avoid cloning its complete BPE vocab
        // merely to rediscover that maximum here. External tokenizer.json
        // files retain the stronger independent maximum-ID check.
        let tokenizer_is_compatible = match files.format() {
            ModelFormat::Gguf => tokenizer.get_vocab_size(true) > 0,
            ModelFormat::Safetensors => tokenizer
                .get_vocab(true)
                .values()
                .copied()
                .max()
                .is_some_and(|token| (token as usize) < adapter.runtime_spec().vocab_size),
        };
        anyhow::ensure!(
            tokenizer_is_compatible,
            "tokenizer vocabulary is empty or contains IDs outside the model vocabulary"
        );
        let cache_config = paged_kv_pool_config(config, adapter.runtime_spec().max_context)?;
        let device = config.device.create()?;
        // SAFETY: the runtime contract treats model files as immutable while
        // this process owns tensors mapped from them.
        let weights = unsafe { files.weight_source(DType::from(dtype), &device)? };
        let mut model = adapter.build(weights.as_ref())?;
        drop(weights);
        drop(files);
        model.configure_kv_cache(cache_config)?;
        model.configure_prefix_cache(config.prefix_cache_capacity_tokens)?;
        model.configure_prefill_attention(config.prefill_attention)?;
        release_loader_allocator_pages();
        Ok(Self {
            adapter,
            tokenizer,
            model,
            dtype,
        })
    }

    /// Tokenize an already rendered prompt using the loaded tokenizer.
    pub fn encode(&self, prompt: impl Into<String>, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(prompt.into(), add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| anyhow::anyhow!(error.to_string()))
    }
}

/// Return free pages left by GGUF metadata and tokenizer construction to
/// macOS. The active tokenizer, model tensors and file-backed Metal mappings
/// remain allocated; this only asks the default malloc zone to purge empty
/// spans after the one-time loader objects have been dropped.
#[cfg(target_os = "macos")]
fn release_loader_allocator_pages() {
    unsafe extern "C" {
        fn malloc_zone_pressure_relief(zone: *mut libc::malloc_zone_t, goal: usize) -> usize;
    }

    // SAFETY: `malloc_default_zone` returns the process-owned default zone and
    // a zero goal requests best-effort release without invalidating live
    // allocations. The API is available on every supported macOS version.
    let _released = unsafe { malloc_zone_pressure_relief(libc::malloc_default_zone(), 0) };
}

#[cfg(not(target_os = "macos"))]
fn release_loader_allocator_pages() {}

fn paged_kv_pool_config(
    config: &ModelLoadConfig,
    default_capacity_tokens: usize,
) -> Result<PagedKvPoolConfig> {
    anyhow::ensure!(
        config.kv_cache_block_tokens > 0,
        "KV cache block size must be positive"
    );
    anyhow::ensure!(
        config.kv_cache_slab_blocks > 0,
        "KV cache slab size must be positive"
    );
    let capacity_tokens = config
        .kv_cache_capacity_tokens
        .unwrap_or(default_capacity_tokens);
    anyhow::ensure!(
        capacity_tokens > 0,
        "KV cache token capacity must be positive"
    );
    let max_blocks = capacity_tokens.div_ceil(config.kv_cache_block_tokens);
    PagedKvPoolConfig::new(
        config.kv_cache_block_tokens,
        config.kv_cache_slab_blocks,
        max_blocks,
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}

fn resolve_dtype(
    device: DeviceKind,
    requested: Option<DTypeKind>,
    files: &ModelFiles,
) -> Result<DTypeKind> {
    let dtype = requested.unwrap_or_else(|| {
        if device == DeviceKind::Cpu || files.has_quantized_tensors() {
            DTypeKind::F32
        } else {
            DTypeKind::F16
        }
    });
    device.validate_dtype(dtype)?;
    Ok(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_load_config_uses_one_context_shared_pool_defaults() {
        let config = ModelLoadConfig::new("model", DeviceKind::Cpu, None);

        assert_eq!(
            config.kv_cache_block_tokens,
            ModelLoadConfig::DEFAULT_KV_CACHE_BLOCK_TOKENS
        );
        assert_eq!(
            config.kv_cache_slab_blocks,
            ModelLoadConfig::DEFAULT_KV_CACHE_SLAB_BLOCKS
        );
        assert_eq!(config.kv_cache_capacity_tokens, None);
        assert_eq!(config.prefix_cache_capacity_tokens, 0);

        let pool = paged_kv_pool_config(&config, 4096).unwrap();
        assert_eq!(pool.block_tokens(), 16);
        assert_eq!(pool.slab_blocks(), 8);
        assert_eq!(pool.max_blocks(), 256);
    }

    #[test]
    fn kv_cache_geometry_follows_the_device() {
        assert_eq!(
            ModelLoadConfig::default_kv_cache_geometry(DeviceKind::Cpu),
            (
                ModelLoadConfig::DEFAULT_KV_CACHE_BLOCK_TOKENS,
                ModelLoadConfig::DEFAULT_KV_CACHE_SLAB_BLOCKS
            )
        );
        assert_eq!(
            ModelLoadConfig::default_kv_cache_geometry(DeviceKind::Metal),
            (
                ModelLoadConfig::DEFAULT_KV_CACHE_BLOCK_TOKENS,
                ModelLoadConfig::DEFAULT_KV_CACHE_SLAB_BLOCKS
            )
        );
        assert_eq!(
            ModelLoadConfig::default_kv_cache_geometry(DeviceKind::Cuda),
            (
                ModelLoadConfig::CUDA_KV_CACHE_BLOCK_TOKENS,
                ModelLoadConfig::CUDA_KV_CACHE_SLAB_BLOCKS
            )
        );

        let cuda = ModelLoadConfig::new("model", DeviceKind::Cuda, None);
        assert_eq!(
            cuda.kv_cache_block_tokens,
            ModelLoadConfig::CUDA_KV_CACHE_BLOCK_TOKENS
        );
        assert_eq!(
            cuda.kv_cache_slab_blocks,
            ModelLoadConfig::CUDA_KV_CACHE_SLAB_BLOCKS
        );
    }

    #[test]
    fn explicit_kv_capacity_rounds_up_to_whole_blocks() {
        let mut config = ModelLoadConfig::new("model", DeviceKind::Cpu, None);
        config.kv_cache_block_tokens = 16;
        config.kv_cache_slab_blocks = 4;
        config.kv_cache_capacity_tokens = Some(33);

        let pool = paged_kv_pool_config(&config, 4096).unwrap();
        assert_eq!(pool.block_tokens(), 16);
        assert_eq!(pool.slab_blocks(), 4);
        assert_eq!(pool.max_blocks(), 3);
    }

    #[test]
    fn zero_kv_pool_dimensions_are_rejected_before_loading_weights() {
        let mut config = ModelLoadConfig::new("model", DeviceKind::Cpu, None);
        config.kv_cache_block_tokens = 0;
        assert!(
            paged_kv_pool_config(&config, 4096)
                .unwrap_err()
                .to_string()
                .contains("block size must be positive")
        );

        config.kv_cache_block_tokens = 16;
        config.kv_cache_slab_blocks = 0;
        assert!(
            paged_kv_pool_config(&config, 4096)
                .unwrap_err()
                .to_string()
                .contains("slab size must be positive")
        );

        config.kv_cache_slab_blocks = 8;
        config.kv_cache_capacity_tokens = Some(0);
        assert!(
            paged_kv_pool_config(&config, 4096)
                .unwrap_err()
                .to_string()
                .contains("token capacity must be positive")
        );
    }
}
