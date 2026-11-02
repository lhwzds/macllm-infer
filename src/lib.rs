//! `MacLLM Infer` is an Apple Silicon-first Rust LLM inference runtime.
//!
//! The runtime deliberately separates three concerns:
//! - [`architecture`]: topology-free runtime metadata, shared decoder
//!   specifications and model contracts.
//! - [`models`]: thin adapters that translate a checkpoint family into a spec.
//! - [`decoder`]: the shared Transformer execution path.

pub mod agent;
pub mod architecture;
pub mod backend;
// The benchmark harness and the model adapters keep their long, auditable
// steps and their toolchain names in one place, so the two lints that police
// their size and their documentation are allowed for these modules.
#[allow(clippy::too_many_lines, clippy::doc_markdown)]
pub mod benchmark;
pub mod cache;
pub mod chat;
pub mod decoder;
pub mod embedding;
pub mod engine;
#[cfg(feature = "metal")]
mod file_backed_quant;
mod gated_delta;
pub mod generation;
pub mod gguf;
pub mod gguf_model;
pub mod gguf_tokenizer;
pub mod grammar;
pub mod hub;
mod mamba2;
pub mod model_weights;
#[allow(clippy::too_many_lines, clippy::doc_markdown)]
pub mod models;
pub mod openai;
pub mod output;
mod paged_attention;
pub mod prefix_cache;
pub mod rerank;
pub mod runtime;
pub mod sampling;
pub mod server;
pub mod token_stream;
pub mod tool_call;
pub mod weights;
