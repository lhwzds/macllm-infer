//! `MacLLM Infer` is an Apple Silicon-first Rust LLM inference runtime.
//!
//! The runtime deliberately separates three concerns:
//! - [`architecture`]: topology-free runtime metadata, shared decoder
//!   specifications and model contracts.
//! - [`models`]: thin adapters that translate a checkpoint family into a spec.
//! - [`decoder`]: the shared Transformer execution path.

pub mod architecture;
pub mod backend;
// The benchmark harness and the model adapters keep their long, auditable
// steps and their toolchain names in one place, so the two lints that police
// their size and their documentation are allowed for these modules.
#[allow(clippy::too_many_lines, clippy::doc_markdown)]
pub mod benchmark;
pub mod decoder;
pub mod embedding;
pub mod engine;
#[cfg(feature = "metal")]
mod file_backed_quant;
pub mod generation;
pub mod gguf;
pub mod gguf_model;
pub mod gguf_tokenizer;
pub mod hub;
pub mod model_weights;
#[allow(clippy::too_many_lines, clippy::doc_markdown)]
pub mod models;
pub mod prefill_attention;
pub mod rerank;
pub mod runtime;
pub mod sampling;
pub mod server;
pub(crate) mod snapshot;
pub mod token_stream;
pub mod weights;

// The chat group, the K/V cache and the custom kernels live in their own
// crates now. The re-exports keep every existing path working.
pub use macllm_chat::{
    agent, chat, chat_templates, constraint, finish, grammar, openai, output, tool_call,
};
#[doc(hidden)]
pub use macllm_kernels::{gated_delta, mamba2, paged_attention};
pub use macllm_kv::{cache, prefix_cache};
