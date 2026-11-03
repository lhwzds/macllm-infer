//! Observable reason why a generation loop stopped producing tokens.
//!
//! The reason is part of the chat group's wire contract: the OpenAI
//! compatibility layer maps it onto the protocol's finish reasons, so it
//! must not depend on the generation runtime.

/// Observable reason why a generation loop stopped producing tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The configured output-token budget was exhausted.
    Length,
    /// The model emitted one of its configured stop-token IDs.
    Stop,
    /// The caller requested cooperative cancellation between model steps.
    Cancelled,
}
