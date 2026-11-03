//! Byte-exact chat and agent prompt renderers for the pinned model families.
//!
//! The renderers live in the chat group because they only need the protocol
//! independent chat and agent contracts. The model adapters reuse them, so
//! the served prompt bytes stay identical no matter which crate owns the
//! adapter in a later workspace layout.

pub mod qwen3;
pub mod qwen35;
