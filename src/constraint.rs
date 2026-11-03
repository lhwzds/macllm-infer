//! Request-local pre-sampling constraints.
//!
//! A constraint masks disallowed vocabulary entries before temperature and
//! top-k/top-p are applied, then advances its state only for the selected
//! token. This keeps structured output enforcement inside sampling rather
//! than validating an already-emitted string.
//!
//! The trait lives in the chat group so that grammar, tool-call and
//! serving-side constraints can be written without depending on the
//! generation runtime.

use anyhow::{Result, bail};

/// Request-local pre-sampling constraint.
///
/// Implementations mask disallowed vocabulary entries before temperature and
/// top-k/top-p are applied, then advance their state only for the selected
/// token. This keeps structured output enforcement inside sampling rather
/// than validating an already-emitted string.
pub trait LogitsConstraint {
    fn mask_logits(&mut self, logits: &mut [f32]) -> Result<()>;
    fn commit_token(&mut self, token: u32) -> Result<()>;

    /// Validate a non-cancelled generation boundary.
    ///
    /// Stateful constraints use this to reject a token-budget or caller stop
    /// that would otherwise expose an incomplete structured value. Plain
    /// token filters may keep the default no-op implementation.
    fn finish(&self) -> Result<()> {
        Ok(())
    }

    /// Validate a generation that ended at the token budget.
    ///
    /// A constraint that guards a tool call may be cut short by the budget:
    /// the serving layer then reports the length limit and never exposes the
    /// incomplete call. Constraints that must produce a complete value keep
    /// the default behaviour and reject an incomplete one.
    fn finish_at_length_boundary(&self) -> Result<()> {
        self.finish()
    }

    /// Whether [`Self::allows_token`] is implemented. The sampler then checks
    /// the few tokens it could select before paying for a full mask.
    fn checks_tokens(&self) -> bool {
        false
    }

    /// Whether [`Self::mask_logits`] would leave this token's finite logit
    /// selectable, without changing any state. Implementations must agree
    /// exactly with `mask_logits`, including failing when it would fail.
    fn allows_token(&self, _token: u32) -> Result<bool> {
        bail!("this sampling constraint only supports full logit masks")
    }
}
