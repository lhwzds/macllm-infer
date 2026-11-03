//! Protocol-independent chat contracts for `MacLLM Infer`.
//!
//! The crate owns the wire-independent vocabulary of the serving stack: the
//! chat history, the native tool-call syntax, the constrained-decoding
//! grammar, the `OpenAI` wire types, the agent prompt contracts, the output
//! filters, the logits-constraint trait, the finish-reason vocabulary and the
//! byte-exact chat templates.
//!
//! It deliberately depends on neither the tensor runtime nor the generation
//! runtime, so every consumer may depend on it.

pub mod agent;
pub mod chat;
pub mod chat_templates;
pub mod constraint;
pub mod finish;
pub mod grammar;
pub mod openai;
pub mod output;
pub mod tool_call;
