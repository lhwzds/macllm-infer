//! K/V cache and prefix index for `MacLLM Infer`.
//!
//! The crate owns the paged K/V allocator that the decoder shares between
//! sessions and the request-prefix index that makes a repeated prompt cheap.

pub mod cache;
pub mod prefix_cache;
