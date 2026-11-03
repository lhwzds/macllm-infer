//! Custom Candle kernels for `MacLLM Infer`.
//!
//! The crate owns the paged-attention, gated-delta and Mamba2 operations. The
//! Metal sources live next to the Rust code that embeds them.

pub mod gated_delta;
pub mod mamba2;
pub mod paged_attention;

// The paged-attention kernels read the paged K/V allocator through the
// crate-root path they used before the crate split.
#[doc(hidden)]
pub use macllm_kv::cache;
