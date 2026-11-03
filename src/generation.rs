use std::{
    num::NonZeroUsize,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    architecture::{CausalLm, CausalLmBatchInput, CausalLmKvReservationInput, CausalLmSession},
    constraint::LogitsConstraint,
    sampling::Sampler,
};
use anyhow::{Context, Result, ensure};

pub use crate::finish::FinishReason;

#[derive(Debug, Clone, Copy)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    /// Keep sampling through stop tokens. Benchmarks enable this so every run
    /// executes the same number of model steps.
    pub ignore_stop_tokens: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            ignore_stop_tokens: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationMetrics {
    pub prompt_tokens: usize,
    /// Leading prompt tokens restored without executing model forward.
    pub cached_prompt_tokens: usize,
    /// Exact wall time spent looking up and attaching the reusable prompt
    /// prefix before the first model quantum is prepared.
    pub prefix_lookup_attach_time: Duration,
    pub generated_tokens: usize,
    /// Number of single-token decode quanta this sequence joined after prefill.
    pub decode_steps: usize,
    pub time_to_first_token: Duration,
    /// Prefill model time before the packed forward that produces first-token
    /// logits. This excludes prefix lookup/attachment and the first-token
    /// forward itself.
    pub suffix_prefill_time: Duration,
    /// Model time for the prefill forward that produces first-token logits.
    pub first_token_model_time: Duration,
    /// Compatibility aggregate equal to `suffix_prefill_time +
    /// first_token_model_time`.
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub total_time: Duration,
    pub finish_reason: FinishReason,
}

/// Request-local control returned after observing one sampled token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenControl {
    Continue,
    /// Count and retain this sampled token, then finish without another model
    /// forward. Text stop filters use this after suppressing a matched stop.
    Stop,
}

impl GenerationMetrics {
    #[must_use]
    pub const fn processed_prompt_tokens(&self) -> usize {
        self.prompt_tokens.saturating_sub(self.cached_prompt_tokens)
    }

    #[must_use]
    pub fn prefill_tokens_per_second(&self) -> f64 {
        rate(self.processed_prompt_tokens(), self.prefill_time)
    }

    /// Total logical prompt tokens divided by the actual prefill duration.
    /// This captures the user-visible benefit of prefix reuse without
    /// mislabeling it as kernel throughput.
    #[must_use]
    pub fn effective_prefill_tokens_per_second(&self) -> f64 {
        rate(self.prompt_tokens, self.prefill_time)
    }

    #[must_use]
    pub fn decode_tokens_per_second(&self) -> f64 {
        rate(self.decode_steps, self.decode_time)
    }

    /// Visible output tokens divided by total wall-clock generation time,
    /// including prefill, sampling and the token callback.
    #[must_use]
    pub fn output_tokens_per_second(&self) -> f64 {
        rate(self.generated_tokens, self.total_time)
    }
}

/// One resumable generation quantum.
///
/// A call to [`GenerationSequence::step`] performs at most one model forward
/// and invokes the token callback at most once. The last visible token carries
/// its final metrics in the same quantum; stop and cancellation can finish a
/// sequence without producing a visible token. An intermediate budgeted
/// prefill step has input progress but neither a token nor final metrics.
#[derive(Debug, Clone)]
pub struct GenerationStep {
    input_tokens: usize,
    token: Option<u32>,
    metrics: Option<GenerationMetrics>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationStatus {
    Running,
    Finished,
    Failed,
}

impl GenerationStep {
    fn advanced(input_tokens: usize) -> Self {
        Self {
            input_tokens,
            token: None,
            metrics: None,
        }
    }

    fn emitted(input_tokens: usize, token: u32, metrics: Option<GenerationMetrics>) -> Self {
        Self {
            input_tokens,
            token: Some(token),
            metrics,
        }
    }

    fn finished(input_tokens: usize, metrics: GenerationMetrics) -> Self {
        Self {
            input_tokens,
            token: None,
            metrics: Some(metrics),
        }
    }

    #[must_use]
    pub const fn input_tokens(&self) -> usize {
        self.input_tokens
    }

    #[must_use]
    pub const fn token(&self) -> Option<u32> {
        self.token
    }

    #[must_use]
    pub fn metrics(&self) -> Option<&GenerationMetrics> {
        self.metrics.as_ref()
    }

    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.metrics().is_some()
    }
}

/// A single independently resumable autoregressive sequence.
///
/// The sequence owns its session and sampler so scheduler code cannot resume
/// its generation history against another same-model session or RNG stream.
/// Model weights remain external and immutable, allowing several sequences to
/// be prepared independently and then executed in one mixed tensor batch.
#[must_use = "generation does not start until the sequence is stepped"]
pub struct GenerationSequence {
    core: GenerationCore,
    device: candle_core::Device,
    session: Option<Box<dyn CausalLmSession>>,
    sampler: Sampler,
    logits_constraint: Option<Box<dyn LogitsConstraint>>,
    publish_prompt_prefix: bool,
    drop_cleanup: SequenceDropCleanup,
    #[cfg(test)]
    drop_drain_attempts: Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceDropCleanup {
    DrainAndClear,
    Discard,
}

impl GenerationSequence {
    pub fn new<M>(
        model: &M,
        prompt_tokens: Vec<u32>,
        sampler: Sampler,
        config: GenerationConfig,
    ) -> Result<Self>
    where
        M: CausalLm + ?Sized,
    {
        let session = model.new_session()?;
        Self::with_session(model, session, prompt_tokens, sampler, config)
    }

    pub(crate) fn with_session<M>(
        model: &M,
        mut session: Box<dyn CausalLmSession>,
        prompt_tokens: Vec<u32>,
        sampler: Sampler,
        config: GenerationConfig,
    ) -> Result<Self>
    where
        M: CausalLm + ?Sized,
    {
        let core = GenerationCore::new(model, session.as_mut(), prompt_tokens, config)?;
        Ok(Self {
            core,
            device: model.device().clone(),
            session: Some(session),
            sampler,
            logits_constraint: None,
            publish_prompt_prefix: true,
            drop_cleanup: SequenceDropCleanup::DrainAndClear,
            #[cfg(test)]
            drop_drain_attempts: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    /// Install one request-local pre-sampling constraint before the first
    /// model quantum. Constraints cannot be swapped after generation starts.
    pub fn set_logits_constraint(&mut self, constraint: Box<dyn LogitsConstraint>) -> Result<()> {
        ensure!(
            self.core.next_quantum == 0 && self.core.in_flight.is_none(),
            "logits constraint must be installed before generation starts"
        );
        ensure!(
            self.logits_constraint.is_none(),
            "generation sequence already has a logits constraint"
        );
        ensure!(
            self.core.config.max_new_tokens > 0,
            "logits constraints require a positive output-token budget"
        );
        self.logits_constraint = Some(constraint);
        Ok(())
    }

    /// Control whether completed prompt blocks are offered to the automatic
    /// prefix cache. Serving requests enable publication by default; repeated
    /// benchmark waves can disable it so task-specific suffixes do not change
    /// the prefix matched by later waves.
    pub(crate) fn set_publish_prompt_prefix(&mut self, publish: bool) -> Result<()> {
        ensure!(
            self.core.next_quantum == 0 && self.core.in_flight.is_none(),
            "prompt-prefix publication policy must be set before generation starts"
        );
        self.publish_prompt_prefix = publish;
        Ok(())
    }

    /// Advance this sequence by one resumable model quantum.
    ///
    /// Every call must use the model instance that created the sequence. A
    /// returned error is a sticky terminal failure: the sequence cannot be
    /// resumed and its session allocation cannot be reused. Cancellation is
    /// instead a successful terminal step with [`FinishReason::Cancelled`].
    pub fn step<M, F, C>(
        &mut self,
        model: &M,
        should_cancel: C,
        on_token: F,
    ) -> Result<GenerationStep>
    where
        M: CausalLm + ?Sized,
        F: FnMut(u32) -> Result<()>,
        C: FnMut() -> bool,
    {
        self.step_with_budget(model, usize::MAX, should_cancel, on_token)
    }

    /// Advance using a positive `input_token_budget`.
    ///
    /// Decode consumes exactly one input token; a long prefill may yield
    /// without sampling until its final prompt chunk has completed. The model
    /// owner and sticky-error contract from [`Self::step`] also applies. A
    /// zero budget is rejected before the quantum begins and leaves the
    /// sequence runnable.
    pub fn step_with_budget<M, F, C>(
        &mut self,
        model: &M,
        input_token_budget: usize,
        mut should_cancel: C,
        on_token: F,
    ) -> Result<GenerationStep>
    where
        M: CausalLm + ?Sized,
        F: FnMut(u32) -> Result<()>,
        C: FnMut() -> bool,
    {
        ensure!(
            input_token_budget > 0,
            "generation step input-token budget must be positive"
        );
        let preparation =
            self.prepare_step_with_budget(model, input_token_budget, &mut should_cancel)?;
        let prepared = match preparation {
            GenerationPreparation::Complete(step) => return Ok(step),
            GenerationPreparation::Forward(prepared) => prepared,
        };
        let model_start = Instant::now();
        let logits = {
            let session = self
                .session
                .as_deref_mut()
                .ok_or_else(|| anyhow::anyhow!("generation sequence session is unavailable"))?;
            model.forward_tokens(session, prepared.input_ids(), prepared.start_pos())?
        };
        let logits = logits.to_dtype(candle_core::DType::F32)?;
        let logits = model.read_output_to_cpu(&logits)?;
        let model_time = model_start.elapsed();
        self.validate_prepared_forward(&prepared)?;
        self.complete_prepared(
            model,
            prepared,
            logits,
            model_time,
            &mut should_cancel,
            on_token,
        )
    }

    pub(crate) fn prepare_step_with_budget<M, C>(
        &mut self,
        model: &M,
        input_token_budget: usize,
        should_cancel: C,
    ) -> Result<GenerationPreparation>
    where
        M: CausalLm + ?Sized,
        C: FnMut() -> bool,
    {
        ensure!(
            input_token_budget > 0,
            "generation step input-token budget must be positive"
        );
        let session = self
            .session
            .as_deref_mut()
            .ok_or_else(|| anyhow::anyhow!("generation sequence session is unavailable"))?;
        self.core
            .prepare(model, session, input_token_budget, should_cancel)
    }

    pub(crate) fn preview_step_with_budget<M>(
        &self,
        model: &M,
        input_token_budget: usize,
    ) -> Result<GenerationPreview>
    where
        M: CausalLm + ?Sized,
    {
        ensure!(
            input_token_budget > 0,
            "generation preview input-token budget must be positive"
        );
        let session = self
            .session
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("generation sequence session is unavailable"))?;
        self.core.preview(model, session, input_token_budget)
    }

    /// Consume a previously observed preview only after the scheduler has
    /// obtained any required host-side K/V reservation.
    pub(crate) fn prepare_step_from_preview<M, C>(
        &mut self,
        model: &M,
        preview: &GenerationForwardPreview,
        input_token_budget: usize,
        should_cancel: C,
    ) -> Result<GenerationPreparation>
    where
        M: CausalLm + ?Sized,
        C: FnMut() -> bool,
    {
        let session = self
            .session
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("generation sequence session is unavailable"))?;
        self.core
            .validate_preview(model, session, preview, input_token_budget)?;
        let preparation =
            self.prepare_step_with_budget(model, input_token_budget, should_cancel)?;
        if let GenerationPreparation::Forward(prepared) = &preparation
            && (prepared.start_pos() != preview.start_pos()
                || prepared.input_tokens() != preview.input_tokens()
                || prepared.next_cache_len != preview.next_cache_len())
        {
            self.core.invalidate_in_flight();
            anyhow::bail!("generation state changed after its K/V reservation preview");
        }
        Ok(preparation)
    }

    pub(crate) fn kv_reservation_input<'a>(
        &'a self,
        preview: &GenerationForwardPreview,
    ) -> Result<CausalLmKvReservationInput<'a>> {
        let session = self
            .session
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("generation sequence session is unavailable"))?;
        self.core.validate_preview_geometry(session, preview)?;
        Ok(CausalLmKvReservationInput::new(
            session,
            preview.start_pos(),
            preview.input_tokens(),
        ))
    }

    pub(crate) fn prepared_batch_input<'a>(
        &'a mut self,
        prepared: &'a PreparedGeneration,
    ) -> Result<CausalLmBatchInput<'a>> {
        let Some(session) = self.session.as_deref() else {
            self.core.invalidate_in_flight();
            anyhow::bail!("generation sequence session is unavailable");
        };
        self.core.validate_prepared_before(prepared, session)?;
        let session = self
            .session
            .as_deref_mut()
            .ok_or_else(|| anyhow::anyhow!("generation sequence session is unavailable"))?;
        Ok(CausalLmBatchInput::new(
            session,
            prepared.input_ids(),
            prepared.start_pos(),
        ))
    }

    pub(crate) fn validate_prepared_forward(
        &mut self,
        prepared: &PreparedGeneration,
    ) -> Result<()> {
        let Some(session) = self.session.as_deref() else {
            self.core.invalidate_in_flight();
            anyhow::bail!("generation sequence session is unavailable");
        };
        self.core.validate_prepared_after(prepared, session)
    }

    pub(crate) fn complete_prepared<M, F, C>(
        &mut self,
        model: &M,
        prepared: PreparedGeneration,
        logits: candle_core::Tensor,
        model_time: Duration,
        should_cancel: C,
        on_token: F,
    ) -> Result<GenerationStep>
    where
        M: CausalLm + ?Sized,
        F: FnMut(u32) -> Result<()>,
        C: FnMut() -> bool,
    {
        let mut on_token = on_token;
        self.complete_prepared_controlled(
            model,
            prepared,
            logits,
            model_time,
            should_cancel,
            |token, _n_decoded| {
                on_token(token)?;
                Ok(TokenControl::Continue)
            },
        )
    }

    pub(crate) fn complete_prepared_controlled<M, F, C>(
        &mut self,
        model: &M,
        prepared: PreparedGeneration,
        logits: candle_core::Tensor,
        model_time: Duration,
        should_cancel: C,
        on_token: F,
    ) -> Result<GenerationStep>
    where
        M: CausalLm + ?Sized,
        F: FnMut(u32, usize) -> Result<TokenControl>,
        C: FnMut() -> bool,
    {
        let Some(session) = self.session.as_deref() else {
            self.core.invalidate_in_flight();
            anyhow::bail!("generation sequence session is unavailable");
        };
        if let Err(error) = model.validate_session(session) {
            self.core.invalidate_in_flight();
            return Err(error.into());
        }
        self.core.validate_prepared_after(&prepared, session)?;
        if self.publish_prompt_prefix
            && let Some(prefix_tokens) = prepared.completed_prefill_tokens(self.core.prompt_tokens)
        {
            model.publish_prompt_prefix(session, &self.core.history[..prefix_tokens]);
        }
        self.core.complete(
            model,
            &mut self.sampler,
            self.logits_constraint.as_deref_mut(),
            prepared,
            GenerationForwardOutput { logits, model_time },
            should_cancel,
            on_token,
        )
    }

    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.status() == GenerationStatus::Finished
    }

    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.status() != GenerationStatus::Running
    }

    #[must_use]
    pub fn status(&self) -> GenerationStatus {
        self.core.status()
    }

    #[must_use]
    pub fn metrics(&self) -> Option<&GenerationMetrics> {
        self.core.metrics()
    }

    #[must_use]
    pub(crate) const fn prefix_lookup_attach_time(&self) -> Duration {
        self.core.prefix_lookup_attach_time
    }

    #[must_use]
    pub fn cache_len(&self) -> usize {
        self.session_ref().cache_len()
    }

    /// Drain device work, clear logical state and return this sequence's
    /// reusable session allocation. A drain failure drops the session instead
    /// of exposing potentially unsafe-to-reuse backend state. Failed
    /// sequences are drained and cleared but their allocations are discarded.
    pub fn reclaim_session<M>(mut self, model: &M) -> Result<Box<dyn CausalLmSession>>
    where
        M: CausalLm + ?Sized,
    {
        self.reclaim_session_inner(model, true)
    }

    /// Clear and return a session after the caller has completed the model
    /// work since this sequence's most recent forward. Serving uses this after
    /// one whole-batch final-logits readback so terminal lanes do not each
    /// drain the same device again. Once entered, validation failure or cleanup
    /// panic discards the session without a fallback drain or cleanup retry.
    pub(crate) fn reclaim_session_after_completion_barrier<M>(
        mut self,
        model: &M,
    ) -> Result<Box<dyn CausalLmSession>>
    where
        M: CausalLm + ?Sized,
    {
        // The caller has already established quiescence for this sequence.
        // From this point onward, even validation failure or unwinding from
        // cache cleanup must only discard the session: retrying a device drain
        // would violate the shared completion-barrier contract.
        self.drop_cleanup = SequenceDropCleanup::Discard;
        self.reclaim_session_inner(model, false)
    }

    /// Discard this sequence during runtime poisoning without issuing another
    /// device synchronization or making its session reusable.
    pub(crate) fn discard_session_without_synchronize(mut self) {
        self.drop_cleanup = SequenceDropCleanup::Discard;
        drop(self.session.take());
    }

    fn reclaim_session_inner<M>(
        &mut self,
        model: &M,
        synchronize: bool,
    ) -> Result<Box<dyn CausalLmSession>>
    where
        M: CausalLm + ?Sized,
    {
        model.validate_session(self.session_ref())?;
        let failed = self.status() == GenerationStatus::Failed;
        let drain = if synchronize {
            model.synchronize()
        } else {
            Ok(())
        };
        if let Some(session) = self.session.as_deref_mut() {
            if drain.is_ok() && synchronize {
                session.mark_device_synchronized();
            }
            session.clear_cache();
        }
        if let Err(error) = drain {
            drop(self.session.take());
            return Err(error.into());
        }
        if failed {
            drop(self.session.take());
            anyhow::bail!("a failed generation sequence cannot return a reusable session");
        }
        self.session
            .take()
            .ok_or_else(|| anyhow::anyhow!("generation sequence session is unavailable"))
    }

    fn session_ref(&self) -> &dyn CausalLmSession {
        self.session
            .as_deref()
            .expect("a live generation sequence must own a session")
    }

    fn drain_and_clear(&mut self) -> candle_core::Result<()> {
        if self.session.is_none() {
            return Ok(());
        }
        #[cfg(test)]
        self.drop_drain_attempts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let drain = self.device.synchronize();
        if let Some(session) = self.session.as_deref_mut() {
            if drain.is_ok() {
                session.mark_device_synchronized();
            }
            session.clear_cache();
        }
        drain
    }
}

impl Drop for GenerationSequence {
    fn drop(&mut self) {
        if self.drop_cleanup == SequenceDropCleanup::Discard {
            drop(self.session.take());
            return;
        }
        // A forward may enqueue work before returning an error. The destructor
        // cannot report cleanup failure, so it drains best-effort and always
        // discards the logical cache instead of exposing the allocation again.
        let _ = catch_unwind(AssertUnwindSafe(|| self.drain_and_clear()));
    }
}

#[derive(Debug)]
enum GenerationPhase {
    Prefill { cursor: usize },
    Decode(u32),
    Finished(GenerationMetrics),
    Failed,
}

#[derive(Debug)]
struct GenerationCore {
    identity: Arc<()>,
    next_quantum: u64,
    in_flight: Option<u64>,
    config: GenerationConfig,
    history: Vec<u32>,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    prefix_lookup_attach_time: Duration,
    generated_tokens: usize,
    decode_steps: usize,
    total_start: Instant,
    time_to_first_token: Duration,
    suffix_prefill_time: Duration,
    first_token_model_time: Duration,
    prefill_time: Duration,
    decode_time: Duration,
    phase: GenerationPhase,
}

pub(crate) enum GenerationPreparation {
    Complete(GenerationStep),
    Forward(PreparedGeneration),
}

/// Immutable host-side view of the next generation quantum.
///
/// A scheduler may use this to reserve bounded shared K/V capacity before a
/// fail-closed [`PreparedGeneration`] exists. Repeated previews do not consume
/// a quantum nonce, change the phase, or make the session unsafe to reuse.
#[derive(Debug, Clone)]
pub(crate) enum GenerationPreview {
    Complete,
    Forward(GenerationForwardPreview),
}

#[derive(Debug, Clone)]
pub(crate) struct GenerationForwardPreview {
    owner: Arc<()>,
    quantum: u64,
    input_token_budget: usize,
    start_pos: usize,
    input_tokens: usize,
    next_cache_len: usize,
}

impl GenerationForwardPreview {
    pub(crate) const fn start_pos(&self) -> usize {
        self.start_pos
    }

    pub(crate) const fn input_tokens(&self) -> usize {
        self.input_tokens
    }

    pub(crate) const fn next_cache_len(&self) -> usize {
        self.next_cache_len
    }
}

pub(crate) struct PreparedGeneration {
    owner: Arc<()>,
    quantum: u64,
    phase: GenerationPhase,
    input_ids: Vec<u32>,
    start_pos: usize,
    next_cache_len: usize,
    ready_to_sample: bool,
}

struct GenerationForwardOutput {
    logits: candle_core::Tensor,
    model_time: Duration,
}

impl PreparedGeneration {
    pub(crate) fn input_ids(&self) -> &[u32] {
        &self.input_ids
    }

    pub(crate) const fn input_tokens(&self) -> usize {
        self.input_ids.len()
    }

    pub(crate) const fn start_pos(&self) -> usize {
        self.start_pos
    }

    pub(crate) fn phase_kind(&self) -> PreparedGenerationPhase {
        match (&self.phase, self.ready_to_sample) {
            (GenerationPhase::Prefill { .. }, false) => PreparedGenerationPhase::SuffixPrefill,
            (GenerationPhase::Prefill { .. }, true) => PreparedGenerationPhase::FirstToken,
            (GenerationPhase::Decode(_), _) => PreparedGenerationPhase::Decode,
            (GenerationPhase::Finished(_) | GenerationPhase::Failed, _) => {
                unreachable!("terminal generation phase cannot be prepared")
            }
        }
    }

    fn completed_prefill_tokens(&self, prompt_tokens: usize) -> Option<usize> {
        (matches!(self.phase, GenerationPhase::Prefill { .. })
            && self.next_cache_len > 0
            && self.next_cache_len <= prompt_tokens)
            .then_some(self.next_cache_len)
    }
}

/// Host-side phase classification for one prepared lane. It is deliberately
/// independent of backend tensors so benchmark tracing can describe a packed
/// forward without extending device lifetimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreparedGenerationPhase {
    SuffixPrefill,
    FirstToken,
    Decode,
}

impl GenerationCore {
    fn new<M>(
        model: &M,
        session: &mut dyn CausalLmSession,
        prompt_tokens: Vec<u32>,
        config: GenerationConfig,
    ) -> Result<Self>
    where
        M: CausalLm + ?Sized,
    {
        validate_request(model, session, &prompt_tokens, config)?;
        let total_start = Instant::now();
        let prefix_lookup_start = Instant::now();
        let prefix = model.reset_session_for_prompt(session, &prompt_tokens)?;
        let prefix_lookup_attach_time = prefix_lookup_start.elapsed();
        ensure!(
            prefix.cached_tokens < prompt_tokens.len(),
            "prefix cache must leave at least one prompt token for first-token logits"
        );
        ensure!(
            session.cache_len() == prefix.cached_tokens,
            "prefix cache restored {} tokens but session length is {}",
            prefix.cached_tokens,
            session.cache_len()
        );
        Ok(Self {
            identity: Arc::new(()),
            next_quantum: 0,
            in_flight: None,
            prompt_tokens: prompt_tokens.len(),
            cached_prompt_tokens: prefix.cached_tokens,
            prefix_lookup_attach_time,
            history: prompt_tokens,
            config,
            generated_tokens: 0,
            decode_steps: 0,
            total_start,
            time_to_first_token: Duration::ZERO,
            suffix_prefill_time: Duration::ZERO,
            first_token_model_time: Duration::ZERO,
            prefill_time: Duration::ZERO,
            decode_time: Duration::ZERO,
            phase: GenerationPhase::Prefill {
                cursor: prefix.cached_tokens,
            },
        })
    }

    fn step<M, F, C>(
        &mut self,
        model: &M,
        session: &mut dyn CausalLmSession,
        sampler: &mut Sampler,
        input_token_budget: usize,
        mut should_cancel: C,
        mut on_token: F,
    ) -> Result<GenerationStep>
    where
        M: CausalLm + ?Sized,
        F: FnMut(u32) -> Result<()>,
        C: FnMut() -> bool,
    {
        let preparation = self.prepare(model, session, input_token_budget, &mut should_cancel)?;
        let prepared = match preparation {
            GenerationPreparation::Complete(step) => return Ok(step),
            GenerationPreparation::Forward(prepared) => prepared,
        };
        let model_start = Instant::now();
        let logits = model.forward_tokens(session, prepared.input_ids(), prepared.start_pos())?;
        let logits = logits.to_dtype(candle_core::DType::F32)?;
        let logits = model.read_output_to_cpu(&logits)?;
        let model_time = model_start.elapsed();
        self.validate_prepared_after(&prepared, session)?;
        if let Some(prefix_tokens) = prepared.completed_prefill_tokens(self.prompt_tokens) {
            model.publish_prompt_prefix(session, &self.history[..prefix_tokens]);
        }
        self.complete(
            model,
            sampler,
            None,
            prepared,
            GenerationForwardOutput { logits, model_time },
            &mut should_cancel,
            |token, _n_decoded| {
                on_token(token)?;
                Ok(TokenControl::Continue)
            },
        )
    }

    fn preview<M>(
        &self,
        model: &M,
        session: &dyn CausalLmSession,
        input_token_budget: usize,
    ) -> Result<GenerationPreview>
    where
        M: CausalLm + ?Sized,
    {
        ensure!(
            input_token_budget > 0,
            "generation preview input-token budget must be positive"
        );
        ensure!(
            self.in_flight.is_none(),
            "generation sequence already has an in-flight quantum"
        );
        match &self.phase {
            GenerationPhase::Finished(_) => return Ok(GenerationPreview::Complete),
            GenerationPhase::Failed => {
                anyhow::bail!("generation sequence cannot resume after a failed step");
            }
            GenerationPhase::Prefill { .. } | GenerationPhase::Decode(_) => {}
        }
        model.validate_session(session)?;
        if self.config.max_new_tokens == 0 {
            return Ok(GenerationPreview::Complete);
        }

        let (start_pos, input_tokens) = match &self.phase {
            GenerationPhase::Prefill { cursor } => {
                let remaining = self
                    .prompt_tokens
                    .checked_sub(*cursor)
                    .ok_or_else(|| anyhow::anyhow!("generation prefill cursor is invalid"))?;
                ensure!(
                    remaining > 0,
                    "generation prefill cursor is already complete"
                );
                (*cursor, remaining.min(input_token_budget))
            }
            GenerationPhase::Decode(_) => {
                let position = self
                    .prompt_tokens
                    .checked_add(self.generated_tokens.saturating_sub(1))
                    .ok_or_else(|| anyhow::anyhow!("generation cache position overflow"))?;
                (position, 1)
            }
            GenerationPhase::Finished(_) | GenerationPhase::Failed => unreachable!(),
        };
        ensure!(
            session.cache_len() == start_pos,
            "generation session cache length {} does not match expected position {}",
            session.cache_len(),
            start_pos
        );
        let next_cache_len = start_pos
            .checked_add(input_tokens)
            .ok_or_else(|| anyhow::anyhow!("generation cache length overflow"))?;
        Ok(GenerationPreview::Forward(GenerationForwardPreview {
            owner: Arc::clone(&self.identity),
            quantum: self.next_quantum,
            input_token_budget,
            start_pos,
            input_tokens,
            next_cache_len,
        }))
    }

    fn validate_preview_geometry(
        &self,
        session: &dyn CausalLmSession,
        preview: &GenerationForwardPreview,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.identity, &preview.owner),
            "generation K/V preview belongs to another sequence"
        );
        ensure!(
            self.in_flight.is_none(),
            "generation sequence already has an in-flight quantum"
        );
        ensure!(
            self.next_quantum == preview.quantum,
            "generation K/V preview is no longer current"
        );
        ensure!(
            session.cache_len() == preview.start_pos,
            "generation session cache length {} does not match previewed position {}",
            session.cache_len(),
            preview.start_pos
        );
        ensure!(
            preview.start_pos.checked_add(preview.input_tokens) == Some(preview.next_cache_len),
            "generation K/V preview has inconsistent cache geometry"
        );
        Ok(())
    }

    fn validate_preview<M>(
        &self,
        model: &M,
        session: &dyn CausalLmSession,
        preview: &GenerationForwardPreview,
        input_token_budget: usize,
    ) -> Result<()>
    where
        M: CausalLm + ?Sized,
    {
        self.validate_preview_geometry(session, preview)?;
        ensure!(
            input_token_budget == preview.input_token_budget,
            "generation K/V preview was created for a different input-token budget"
        );
        let current = self.preview(model, session, input_token_budget)?;
        let GenerationPreview::Forward(current) = current else {
            anyhow::bail!("generation state no longer requires the previewed forward");
        };
        ensure!(
            current.start_pos == preview.start_pos
                && current.input_tokens == preview.input_tokens
                && current.next_cache_len == preview.next_cache_len,
            "generation state changed after its K/V reservation preview"
        );
        Ok(())
    }

    fn prepare<M, C>(
        &mut self,
        model: &M,
        session: &mut dyn CausalLmSession,
        input_token_budget: usize,
        mut should_cancel: C,
    ) -> Result<GenerationPreparation>
    where
        M: CausalLm + ?Sized,
        C: FnMut() -> bool,
    {
        ensure!(
            input_token_budget > 0,
            "generation step input-token budget must be positive"
        );
        // A prepared quantum is fail-closed: until its opaque plan is consumed
        // successfully, the sequence cannot be resumed or its session reused.
        let phase = std::mem::replace(&mut self.phase, GenerationPhase::Failed);
        match &phase {
            GenerationPhase::Finished(metrics) => {
                self.in_flight = None;
                self.phase = GenerationPhase::Finished(metrics.clone());
                return Ok(GenerationPreparation::Complete(GenerationStep::finished(
                    0,
                    metrics.clone(),
                )));
            }
            GenerationPhase::Failed => {
                self.in_flight = None;
                anyhow::bail!("generation sequence cannot resume after a failed step");
            }
            GenerationPhase::Prefill { .. } | GenerationPhase::Decode(_) => {}
        }

        model.validate_session(session)?;
        if self.config.max_new_tokens == 0 {
            let metrics = self.finish(FinishReason::Length);
            return Ok(GenerationPreparation::Complete(GenerationStep::finished(
                0, metrics,
            )));
        }
        if should_cancel() {
            let metrics = self.finish(FinishReason::Cancelled);
            return Ok(GenerationPreparation::Complete(GenerationStep::finished(
                0, metrics,
            )));
        }

        let (start_pos, input_ids) = match &phase {
            GenerationPhase::Prefill { cursor } => {
                let remaining = self
                    .prompt_tokens
                    .checked_sub(*cursor)
                    .ok_or_else(|| anyhow::anyhow!("generation prefill cursor is invalid"))?;
                ensure!(
                    remaining > 0,
                    "generation prefill cursor is already complete"
                );
                let input_tokens = remaining.min(input_token_budget);
                let end = cursor
                    .checked_add(input_tokens)
                    .ok_or_else(|| anyhow::anyhow!("generation prefill range overflow"))?;
                (*cursor, self.history[*cursor..end].to_vec())
            }
            GenerationPhase::Decode(token) => {
                let position = self
                    .prompt_tokens
                    .checked_add(self.generated_tokens.saturating_sub(1))
                    .ok_or_else(|| anyhow::anyhow!("generation cache position overflow"))?;
                (position, vec![*token])
            }
            GenerationPhase::Finished(_) | GenerationPhase::Failed => unreachable!(),
        };
        ensure!(
            session.cache_len() == start_pos,
            "generation session cache length {} does not match expected position {}",
            session.cache_len(),
            start_pos
        );
        let next_cache_len = start_pos
            .checked_add(input_ids.len())
            .ok_or_else(|| anyhow::anyhow!("generation cache length overflow"))?;
        let ready_to_sample = !matches!(phase, GenerationPhase::Prefill { .. })
            || next_cache_len == self.prompt_tokens;
        let quantum = self.next_quantum;
        self.next_quantum = self
            .next_quantum
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("generation quantum identity overflow"))?;
        self.in_flight = Some(quantum);
        Ok(GenerationPreparation::Forward(PreparedGeneration {
            owner: Arc::clone(&self.identity),
            quantum,
            phase,
            input_ids,
            start_pos,
            next_cache_len,
            ready_to_sample,
        }))
    }

    fn invalidate_in_flight(&mut self) {
        self.in_flight = None;
    }

    fn validate_prepared_owner(&mut self, prepared: &PreparedGeneration) -> Result<()> {
        if !Arc::ptr_eq(&self.identity, &prepared.owner) {
            self.invalidate_in_flight();
            anyhow::bail!("prepared generation quantum belongs to another sequence");
        }
        if !matches!(self.phase, GenerationPhase::Failed) {
            self.invalidate_in_flight();
            anyhow::bail!("generation sequence does not have an in-flight quantum");
        }
        if self.in_flight != Some(prepared.quantum) {
            self.invalidate_in_flight();
            anyhow::bail!("prepared generation quantum is no longer current");
        }
        Ok(())
    }

    fn validate_prepared_before(
        &mut self,
        prepared: &PreparedGeneration,
        session: &dyn CausalLmSession,
    ) -> Result<()> {
        self.validate_prepared_owner(prepared)?;
        if session.cache_len() != prepared.start_pos {
            self.invalidate_in_flight();
            anyhow::bail!(
                "generation session cache length {} does not match prepared position {}",
                session.cache_len(),
                prepared.start_pos
            );
        }
        Ok(())
    }

    fn validate_prepared_after(
        &mut self,
        prepared: &PreparedGeneration,
        session: &dyn CausalLmSession,
    ) -> Result<()> {
        self.validate_prepared_owner(prepared)?;
        if session.cache_len() != prepared.next_cache_len {
            self.invalidate_in_flight();
            anyhow::bail!(
                "model advanced generation cache to {}, expected {}",
                session.cache_len(),
                prepared.next_cache_len
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn complete<M, F, C>(
        &mut self,
        model: &M,
        sampler: &mut Sampler,
        mut constraint: Option<&mut (dyn LogitsConstraint + 'static)>,
        prepared: PreparedGeneration,
        output: GenerationForwardOutput,
        mut should_cancel: C,
        mut on_token: F,
    ) -> Result<GenerationStep>
    where
        M: CausalLm + ?Sized,
        F: FnMut(u32, usize) -> Result<TokenControl>,
        C: FnMut() -> bool,
    {
        self.validate_prepared_owner(&prepared)?;
        let consumed_quantum = self.in_flight.take();
        debug_assert_eq!(consumed_quantum, Some(prepared.quantum));
        let PreparedGeneration {
            phase,
            input_ids,
            next_cache_len,
            ready_to_sample,
            ..
        } = prepared;
        let consumed_tokens = input_ids.len();
        match phase {
            GenerationPhase::Prefill { .. } if ready_to_sample => {
                self.first_token_model_time += output.model_time;
                self.prefill_time += output.model_time;
            }
            GenerationPhase::Prefill { .. } => {
                self.suffix_prefill_time += output.model_time;
                self.prefill_time += output.model_time;
            }
            GenerationPhase::Decode(_) => {
                self.decode_time += output.model_time;
                self.decode_steps += 1;
            }
            GenerationPhase::Finished(_) | GenerationPhase::Failed => unreachable!(),
        }

        if should_cancel() {
            let metrics = self.finish(FinishReason::Cancelled);
            return Ok(GenerationStep::finished(consumed_tokens, metrics));
        }
        if !ready_to_sample {
            self.phase = GenerationPhase::Prefill {
                cursor: next_cache_len,
            };
            return Ok(GenerationStep::advanced(consumed_tokens));
        }
        let token = sampler.sample_with_generated(
            &output.logits,
            &self.history,
            &self.history[self.history.len() - self.generated_tokens..],
            constraint.as_deref_mut(),
        )?;
        if should_cancel() {
            let metrics = self.finish(FinishReason::Cancelled);
            return Ok(GenerationStep::finished(consumed_tokens, metrics));
        }
        if !self.config.ignore_stop_tokens && model.runtime_spec().stop_token_ids.contains(&token) {
            finish_constraint(&constraint)?;
            let metrics = self.finish(FinishReason::Stop);
            return Ok(GenerationStep::finished(consumed_tokens, metrics));
        }
        let reaches_token_budget = self.generated_tokens + 1 == self.config.max_new_tokens;
        if reaches_token_budget {
            // Validate before invoking the caller's visible-token callback so
            // an incomplete structured value cannot escape at a length
            // boundary and only then turn into an error. A tool call that the
            // budget cut short is the one exception: the serving layer reports
            // the length limit and never exposes the incomplete call.
            finish_constraint_at_length(&constraint)?;
        }
        if self.generated_tokens == 0 {
            self.time_to_first_token = self.total_start.elapsed();
        }
        let n_decoded = self
            .generated_tokens
            .checked_add(1)
            .context("generated token count overflowed")?;
        let control = on_token(token, n_decoded)?;
        self.generated_tokens = n_decoded;
        self.history.push(token);
        if control == TokenControl::Stop {
            finish_constraint(&constraint)?;
            let metrics = self.finish(FinishReason::Stop);
            return Ok(GenerationStep::emitted(
                consumed_tokens,
                token,
                Some(metrics),
            ));
        }
        if reaches_token_budget {
            let metrics = self.finish(FinishReason::Length);
            return Ok(GenerationStep::emitted(
                consumed_tokens,
                token,
                Some(metrics),
            ));
        }
        self.phase = GenerationPhase::Decode(token);
        Ok(GenerationStep::emitted(consumed_tokens, token, None))
    }

    fn finish(&mut self, finish_reason: FinishReason) -> GenerationMetrics {
        self.in_flight = None;
        let metrics = GenerationMetrics {
            prompt_tokens: self.prompt_tokens,
            cached_prompt_tokens: self.cached_prompt_tokens,
            prefix_lookup_attach_time: self.prefix_lookup_attach_time,
            generated_tokens: self.generated_tokens,
            decode_steps: self.decode_steps,
            time_to_first_token: self.time_to_first_token,
            suffix_prefill_time: self.suffix_prefill_time,
            first_token_model_time: self.first_token_model_time,
            prefill_time: self.prefill_time,
            decode_time: self.decode_time,
            total_time: self.total_start.elapsed(),
            finish_reason,
        };
        self.phase = GenerationPhase::Finished(metrics.clone());
        metrics
    }

    fn metrics(&self) -> Option<&GenerationMetrics> {
        match &self.phase {
            GenerationPhase::Finished(metrics) => Some(metrics),
            GenerationPhase::Prefill { .. }
            | GenerationPhase::Decode(_)
            | GenerationPhase::Failed => None,
        }
    }

    fn status(&self) -> GenerationStatus {
        match &self.phase {
            GenerationPhase::Prefill { .. } | GenerationPhase::Decode(_) => {
                GenerationStatus::Running
            }
            GenerationPhase::Finished(_) => GenerationStatus::Finished,
            GenerationPhase::Failed => GenerationStatus::Failed,
        }
    }
}

fn validate_request<M>(
    model: &M,
    session: &dyn CausalLmSession,
    prompt_tokens: &[u32],
    config: GenerationConfig,
) -> Result<()>
where
    M: CausalLm + ?Sized,
{
    ensure!(
        !prompt_tokens.is_empty(),
        "prompt must contain at least one token"
    );
    let required_cache_len = prompt_tokens
        .len()
        .checked_add(config.max_new_tokens.saturating_sub(1))
        .ok_or_else(|| anyhow::anyhow!("requested context length overflow"))?;
    ensure!(
        required_cache_len <= model.runtime_spec().max_context,
        "prompt plus requested output exceeds model context limit"
    );
    ensure!(
        prompt_tokens
            .iter()
            .all(|&token| (token as usize) < model.runtime_spec().vocab_size),
        "input contains a token ID outside the model vocabulary"
    );
    model.validate_session(session)?;
    Ok(())
}

fn finish_constraint(constraint: &Option<&mut (dyn LogitsConstraint + 'static)>) -> Result<()> {
    if let Some(constraint) = constraint.as_deref() {
        constraint.finish()?;
    }
    Ok(())
}

/// Validate a constraint at the token budget boundary.
///
/// A tool call that the budget cut short is not an error: the serving layer
/// reports the length limit and drops the incomplete call.
fn finish_constraint_at_length(
    constraint: &Option<&mut (dyn LogitsConstraint + 'static)>,
) -> Result<()> {
    if let Some(constraint) = constraint.as_deref() {
        constraint.finish_at_length_boundary()?;
    }
    Ok(())
}

fn rate(tokens: usize, duration: Duration) -> f64 {
    if duration.is_zero() {
        0.0
    } else {
        tokens as f64 / duration.as_secs_f64()
    }
}

pub fn generate<M, F>(
    model: &M,
    session: &mut dyn CausalLmSession,
    prompt_tokens: &[u32],
    sampler: &mut Sampler,
    config: GenerationConfig,
    on_token: F,
) -> Result<GenerationMetrics>
where
    M: CausalLm + ?Sized,
    F: FnMut(u32) -> Result<()>,
{
    generate_with_control(
        model,
        session,
        prompt_tokens,
        sampler,
        config,
        || false,
        on_token,
    )
}

/// Generate tokens while limiting each prefill forward to at most
/// `prefill_token_quantum` prompt tokens.
///
/// Decode forwards still consume one input token at a time. Intermediate
/// prefill chunks only advance model and cache state; sampling and the token
/// callback begin after the final prompt chunk completes.
pub fn generate_with_prefill_token_quantum<M, F>(
    model: &M,
    session: &mut dyn CausalLmSession,
    prompt_tokens: &[u32],
    sampler: &mut Sampler,
    config: GenerationConfig,
    prefill_token_quantum: NonZeroUsize,
    on_token: F,
) -> Result<GenerationMetrics>
where
    M: CausalLm + ?Sized,
    F: FnMut(u32) -> Result<()>,
{
    generate_with_control_and_prefill_token_quantum(
        model,
        session,
        prompt_tokens,
        sampler,
        BlockingGenerationConfig {
            generation: config,
            prefill_token_quantum,
        },
        || false,
        on_token,
    )
}

/// Generate tokens while polling `should_cancel` between completed model
/// quanta. In-flight kernels are allowed to finish before cancellation is
/// observed; no partially submitted model step is interrupted.
pub fn generate_with_control<M, F, C>(
    model: &M,
    session: &mut dyn CausalLmSession,
    prompt_tokens: &[u32],
    sampler: &mut Sampler,
    config: GenerationConfig,
    should_cancel: C,
    on_token: F,
) -> Result<GenerationMetrics>
where
    M: CausalLm + ?Sized,
    F: FnMut(u32) -> Result<()>,
    C: FnMut() -> bool,
{
    generate_with_control_and_prefill_token_quantum(
        model,
        session,
        prompt_tokens,
        sampler,
        BlockingGenerationConfig {
            generation: config,
            prefill_token_quantum: NonZeroUsize::MAX,
        },
        should_cancel,
        on_token,
    )
}

#[derive(Debug, Clone, Copy)]
struct BlockingGenerationConfig {
    generation: GenerationConfig,
    prefill_token_quantum: NonZeroUsize,
}

fn generate_with_control_and_prefill_token_quantum<M, F, C>(
    model: &M,
    session: &mut dyn CausalLmSession,
    prompt_tokens: &[u32],
    sampler: &mut Sampler,
    config: BlockingGenerationConfig,
    mut should_cancel: C,
    mut on_token: F,
) -> Result<GenerationMetrics>
where
    M: CausalLm + ?Sized,
    F: FnMut(u32) -> Result<()>,
    C: FnMut() -> bool,
{
    let mut core = GenerationCore::new(model, session, prompt_tokens.to_vec(), config.generation)?;
    loop {
        let step = core.step(
            model,
            session,
            sampler,
            config.prefill_token_quantum.get(),
            &mut should_cancel,
            &mut on_token,
        )?;
        if let Some(metrics) = step.metrics {
            return Ok(metrics);
        }
    }
}

/// Run one isolated request and always clear its logical K/V state before
/// returning. Cancellation and callback errors are reusable request outcomes;
/// callers must still treat model or device errors as potentially fatal because
/// clearing metadata cannot prove an asynchronous backend remains healthy.
pub fn run_generation_request<M, F, C>(
    model: &M,
    session: &mut dyn CausalLmSession,
    prompt_tokens: &[u32],
    sampler: &mut Sampler,
    config: GenerationConfig,
    should_cancel: C,
    on_token: F,
) -> Result<GenerationMetrics>
where
    M: CausalLm + ?Sized,
    F: FnMut(u32) -> Result<()>,
    C: FnMut() -> bool,
{
    model.validate_session(session)?;
    let mut lease = SessionLease {
        session,
        clear_on_drop: true,
    };
    let result = generate_with_control(
        model,
        &mut *lease.session,
        prompt_tokens,
        sampler,
        config,
        should_cancel,
        on_token,
    );
    // A forward may enqueue device work before returning an error. Drain on
    // the owner thread before making the cache allocation reusable.
    let drain = model.synchronize();
    let clear = lease.clear_cache();
    drop(lease);
    let result = match (result, drain) {
        (Ok(metrics), Ok(())) => Ok(metrics),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(drain_error)) => Err(anyhow::anyhow!(
            "request cleanup failed while synchronizing the device: {drain_error}"
        )),
        (Err(error), Err(drain_error)) => Err(anyhow::anyhow!(
            "{error}; request cleanup also failed while synchronizing the device: {drain_error}"
        )),
    };
    match (result, clear) {
        (Ok(metrics), Ok(())) => Ok(metrics),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(())) => Err(anyhow::anyhow!(
            "request cleanup panicked while clearing the session cache"
        )),
        (Err(error), Err(())) => Err(anyhow::anyhow!(
            "{error}; request cleanup also panicked while clearing the session cache"
        )),
    }
}

struct SessionLease<'a> {
    session: &'a mut dyn CausalLmSession,
    clear_on_drop: bool,
}

impl SessionLease<'_> {
    fn clear_cache(&mut self) -> std::result::Result<(), ()> {
        self.clear_on_drop = false;
        catch_unwind(AssertUnwindSafe(|| self.session.clear_cache())).map_err(|_| ())
    }
}

impl Drop for SessionLease<'_> {
    fn drop(&mut self) {
        if self.clear_on_drop {
            self.clear_on_drop = false;
            let _ = catch_unwind(AssertUnwindSafe(|| self.session.clear_cache()));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        any::Any,
        cell::RefCell,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };

    use candle_core::{Device, Tensor};

    use crate::architecture::{PrefixCacheMatch, RuntimeSpec};
    use crate::sampling::SamplingConfig;

    use super::*;

    struct FakeModel {
        owner: Arc<()>,
        spec: RuntimeSpec,
        device: Device,
        output_logits: Vec<Vec<f32>>,
        cached_prefix_tokens: usize,
        published_prefixes: RefCell<Vec<Vec<u32>>>,
    }

    #[derive(Debug)]
    struct FakeSession {
        owner: Arc<()>,
        cache_len: usize,
        calls: usize,
        forwards: Vec<(Vec<u32>, usize)>,
        clear_count: Arc<AtomicUsize>,
        synchronized_count: Arc<AtomicUsize>,
        drop_count: Arc<AtomicUsize>,
        panic_on_clear: Arc<AtomicBool>,
    }

    impl Drop for FakeSession {
        fn drop(&mut self) {
            self.drop_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl CausalLmSession for FakeSession {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn clear_cache(&mut self) {
            assert!(
                !self.panic_on_clear.swap(false, Ordering::Relaxed),
                "injected generation cache cleanup panic"
            );
            self.clear_count.fetch_add(1, Ordering::Relaxed);
            self.cache_len = 0;
            self.calls = 0;
            self.forwards.clear();
        }

        fn mark_device_synchronized(&mut self) {
            self.synchronized_count.fetch_add(1, Ordering::Relaxed);
        }

        fn cache_len(&self) -> usize {
            self.cache_len
        }
    }

    impl FakeModel {
        fn new(max_context: usize, output_tokens: Vec<u32>) -> Self {
            let output_logits = output_tokens
                .into_iter()
                .map(|token| {
                    let mut logits = vec![0.0_f32; 4];
                    logits[token as usize] = 10.0;
                    logits
                })
                .collect();
            Self::with_logits(max_context, output_logits)
        }

        fn with_logits(max_context: usize, output_logits: Vec<Vec<f32>>) -> Self {
            assert!(
                output_logits.iter().all(|logits| logits.len() == 4),
                "fake logits must match the four-token vocabulary"
            );
            Self {
                owner: Arc::new(()),
                spec: RuntimeSpec {
                    vocab_size: 4,
                    max_context,
                    stop_token_ids: vec![3],
                },
                device: Device::Cpu,
                output_logits,
                cached_prefix_tokens: 0,
                published_prefixes: RefCell::new(Vec::new()),
            }
        }

        fn with_cached_prefix(mut self, cached_prefix_tokens: usize) -> Self {
            self.cached_prefix_tokens = cached_prefix_tokens;
            self
        }
    }

    impl CausalLm for FakeModel {
        fn runtime_spec(&self) -> &RuntimeSpec {
            &self.spec
        }

        fn new_session(&self) -> candle_core::Result<Box<dyn CausalLmSession>> {
            Ok(Box::new(FakeSession {
                owner: Arc::clone(&self.owner),
                cache_len: 0,
                calls: 0,
                forwards: Vec::new(),
                clear_count: Arc::new(AtomicUsize::new(0)),
                synchronized_count: Arc::new(AtomicUsize::new(0)),
                drop_count: Arc::new(AtomicUsize::new(0)),
                panic_on_clear: Arc::new(AtomicBool::new(false)),
            }))
        }

        fn validate_session(&self, session: &dyn CausalLmSession) -> candle_core::Result<()> {
            let session = session
                .as_any()
                .downcast_ref::<FakeSession>()
                .ok_or_else(|| candle_core::Error::Msg("wrong fake session type".into()))?;
            if !Arc::ptr_eq(&self.owner, &session.owner) {
                candle_core::bail!("fake session belongs to another model");
            }
            Ok(())
        }

        fn reset_session_for_prompt(
            &self,
            session: &mut dyn CausalLmSession,
            prompt_tokens: &[u32],
        ) -> candle_core::Result<PrefixCacheMatch> {
            self.validate_session(session)?;
            if self.cached_prefix_tokens >= prompt_tokens.len() {
                candle_core::bail!("fake prefix cache must leave one prompt token");
            }
            session.clear_cache();
            let session = session
                .as_any_mut()
                .downcast_mut::<FakeSession>()
                .ok_or_else(|| candle_core::Error::Msg("wrong fake session type".into()))?;
            session.cache_len = self.cached_prefix_tokens;
            Ok(PrefixCacheMatch {
                cached_tokens: self.cached_prefix_tokens,
            })
        }

        fn forward(
            &self,
            session: &mut dyn CausalLmSession,
            input_ids: &Tensor,
            start_pos: usize,
        ) -> candle_core::Result<Tensor> {
            self.validate_session(session)?;
            let session = session
                .as_any_mut()
                .downcast_mut::<FakeSession>()
                .ok_or_else(|| candle_core::Error::Msg("wrong fake session type".into()))?;
            if start_pos != session.cache_len {
                candle_core::bail!("fake cache position mismatch");
            }
            let tokens = input_ids.to_vec2::<u32>()?.remove(0);
            session.forwards.push((tokens, start_pos));
            session.cache_len += input_ids.dim(1)?;
            let logits = self
                .output_logits
                .get(session.calls)
                .cloned()
                .unwrap_or_else(|| vec![10.0_f32, 0.0, 0.0, 0.0]);
            session.calls += 1;
            Tensor::from_vec(logits, (1, self.spec.vocab_size), &self.device)
        }

        fn publish_prompt_prefix(&self, _session: &dyn CausalLmSession, prompt_tokens: &[u32]) {
            self.published_prefixes
                .borrow_mut()
                .push(prompt_tokens.to_vec());
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    fn fake_session(session: &dyn CausalLmSession) -> &FakeSession {
        session
            .as_any()
            .downcast_ref::<FakeSession>()
            .expect("fake model must create fake sessions")
    }

    fn greedy() -> Sampler {
        Sampler::new(SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        })
        .expect("greedy test configuration is valid")
    }

    struct TerminalConstraint {
        allowed_token: u32,
        terminal_is_valid: bool,
        finish_calls: Arc<AtomicUsize>,
    }

    impl LogitsConstraint for TerminalConstraint {
        fn mask_logits(&mut self, logits: &mut [f32]) -> Result<()> {
            for (index, logit) in logits.iter_mut().enumerate() {
                if index != self.allowed_token as usize {
                    *logit = f32::NEG_INFINITY;
                }
            }
            Ok(())
        }

        fn commit_token(&mut self, token: u32) -> Result<()> {
            ensure!(token == self.allowed_token, "unexpected constrained token");
            Ok(())
        }

        fn finish(&self) -> Result<()> {
            self.finish_calls.fetch_add(1, Ordering::Relaxed);
            ensure!(
                self.terminal_is_valid,
                "injected incomplete structured value"
            );
            Ok(())
        }
    }

    #[test]
    fn constraint_validates_length_boundary_before_exposing_the_last_token() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let finish_calls = Arc::new(AtomicUsize::new(0));
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: true,
            },
        )?;
        sequence.set_logits_constraint(Box::new(TerminalConstraint {
            allowed_token: 1,
            terminal_is_valid: false,
            finish_calls: Arc::clone(&finish_calls),
        }))?;
        let mut visible_tokens = Vec::new();

        let error = sequence
            .step(
                &model,
                || false,
                |token| {
                    visible_tokens.push(token);
                    Ok(())
                },
            )
            .unwrap_err();

        assert!(error.to_string().contains("incomplete structured value"));
        assert!(visible_tokens.is_empty());
        assert_eq!(finish_calls.load(Ordering::Relaxed), 1);
        assert_eq!(sequence.status(), GenerationStatus::Failed);
        Ok(())
    }

    /// A constraint whose value the token budget may cut short.
    struct LengthTolerantConstraint {
        allowed_token: u32,
    }

    impl LogitsConstraint for LengthTolerantConstraint {
        fn mask_logits(&mut self, logits: &mut [f32]) -> Result<()> {
            for (index, logit) in logits.iter_mut().enumerate() {
                if index != self.allowed_token as usize {
                    *logit = f32::NEG_INFINITY;
                }
            }
            Ok(())
        }

        fn commit_token(&mut self, token: u32) -> Result<()> {
            ensure!(token == self.allowed_token, "unexpected constrained token");
            Ok(())
        }

        fn finish(&self) -> Result<()> {
            anyhow::bail!("the constrained value is incomplete")
        }

        fn finish_at_length_boundary(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_value_cut_short_by_the_budget_finishes_at_its_length_limit() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: true,
            },
        )?;
        sequence.set_logits_constraint(Box::new(LengthTolerantConstraint { allowed_token: 1 }))?;
        let mut visible_tokens = Vec::new();

        let step = sequence.step(
            &model,
            || false,
            |token| {
                visible_tokens.push(token);
                Ok(())
            },
        )?;

        assert_eq!(visible_tokens, [1]);
        assert_eq!(
            step.metrics().map(|metrics| metrics.finish_reason),
            Some(FinishReason::Length)
        );
        Ok(())
    }

    #[test]
    fn constraint_accepts_a_complete_length_boundary() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let finish_calls = Arc::new(AtomicUsize::new(0));
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: true,
            },
        )?;
        sequence.set_logits_constraint(Box::new(TerminalConstraint {
            allowed_token: 1,
            terminal_is_valid: true,
            finish_calls: Arc::clone(&finish_calls),
        }))?;
        let mut visible_tokens = Vec::new();

        let step = sequence.step(
            &model,
            || false,
            |token| {
                visible_tokens.push(token);
                Ok(())
            },
        )?;

        assert_eq!(visible_tokens, [1]);
        assert_eq!(finish_calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            step.metrics().map(|metrics| metrics.finish_reason),
            Some(FinishReason::Length)
        );
        Ok(())
    }

    #[test]
    fn every_successful_prefill_quantum_publishes_its_exact_endpoint() -> Result<()> {
        let model = FakeModel::new(16, vec![0, 0, 1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0, 1, 2, 0, 1],
            greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: true,
            },
        )?;

        while !sequence.is_finished() {
            let _step = sequence.step_with_budget(&model, 2, || false, |_| Ok(()))?;
        }

        assert_eq!(
            model.published_prefixes.into_inner(),
            [vec![0, 1], vec![0, 1, 2, 0], vec![0, 1, 2, 0, 1]]
        );
        Ok(())
    }

    #[test]
    fn resumable_sequence_can_disable_prompt_prefix_publication() -> Result<()> {
        let model = FakeModel::new(16, vec![0, 0, 1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0, 1, 2, 0, 1],
            greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: true,
            },
        )?;
        sequence.set_publish_prompt_prefix(false)?;

        while !sequence.is_finished() {
            let _step = sequence.step_with_budget(&model, 2, || false, |_| Ok(()))?;
        }

        assert!(model.published_prefixes.into_inner().is_empty());
        Ok(())
    }

    #[test]
    fn metrics_split_prefill_without_double_counting_first_token_forward() -> Result<()> {
        let model = FakeModel::new(16, vec![0, 1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0, 1, 2, 0, 1],
            greedy(),
            GenerationConfig {
                max_new_tokens: 2,
                ignore_stop_tokens: true,
            },
        )?;

        while !sequence.is_finished() {
            let _step = sequence.step_with_budget(&model, 2, || false, |_| Ok(()))?;
        }
        let metrics = sequence.metrics().expect("sequence must finish");

        assert_eq!(
            metrics.prefill_time,
            metrics.suffix_prefill_time + metrics.first_token_model_time
        );
        assert_eq!(metrics.decode_steps, 1);
        assert!(metrics.prefix_lookup_attach_time <= metrics.total_time);
        Ok(())
    }

    #[test]
    fn eos_after_decode_counts_the_model_step() -> Result<()> {
        let model = FakeModel::new(8, vec![1, 3]);
        let mut session = model.new_session()?;
        let metrics = generate(
            &model,
            session.as_mut(),
            &[0],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 3,
                ignore_stop_tokens: false,
            },
            |_| Ok(()),
        )?;
        assert_eq!(metrics.generated_tokens, 1);
        assert_eq!(metrics.decode_steps, 1);
        assert_eq!(metrics.finish_reason, FinishReason::Stop);
        Ok(())
    }

    #[test]
    fn cached_prefix_skips_only_restored_prompt_tokens() -> Result<()> {
        let model = FakeModel::new(8, vec![1]).with_cached_prefix(2);
        let mut session = model.new_session()?;
        let metrics = generate(
            &model,
            session.as_mut(),
            &[0, 1, 2],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: true,
            },
            |_| Ok(()),
        )?;

        assert_eq!(metrics.prompt_tokens, 3);
        assert_eq!(metrics.cached_prompt_tokens, 2);
        assert_eq!(metrics.processed_prompt_tokens(), 1);
        assert_eq!(fake_session(session.as_ref()).forwards, [(vec![2], 2)]);
        Ok(())
    }

    #[test]
    fn no_visible_token_has_no_ttft() -> Result<()> {
        let stopped = FakeModel::new(8, vec![3]);
        let mut stopped_session = stopped.new_session()?;
        let metrics = generate(
            &stopped,
            stopped_session.as_mut(),
            &[0],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 3,
                ignore_stop_tokens: false,
            },
            |_| Ok(()),
        )?;
        assert_eq!(metrics.generated_tokens, 0);
        assert_eq!(metrics.time_to_first_token, Duration::ZERO);
        assert_eq!(metrics.finish_reason, FinishReason::Stop);

        let disabled = FakeModel::new(8, vec![1]);
        let mut disabled_session = disabled.new_session()?;
        let metrics = generate(
            &disabled,
            disabled_session.as_mut(),
            &[0],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 0,
                ignore_stop_tokens: false,
            },
            |_| Ok(()),
        )?;
        assert_eq!(metrics.generated_tokens, 0);
        assert_eq!(metrics.time_to_first_token, Duration::ZERO);
        assert_eq!(metrics.finish_reason, FinishReason::Length);
        assert_eq!(fake_session(disabled_session.as_ref()).calls, 0);
        assert_eq!(disabled_session.cache_len(), 0);
        Ok(())
    }

    #[test]
    fn host_token_path_rejects_out_of_vocabulary_ids_before_forward() {
        let model = FakeModel::new(8, vec![1]);
        let mut session = model.new_session().unwrap();
        let error = generate(
            &model,
            session.as_mut(),
            &[4],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ..GenerationConfig::default()
            },
            |_| Ok(()),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("outside the model vocabulary"));
        assert_eq!(fake_session(session.as_ref()).calls, 0);
    }

    #[test]
    fn one_token_is_legal_at_full_context() -> Result<()> {
        let model = FakeModel::new(2, vec![1]);
        let mut session = model.new_session()?;
        let metrics = generate(
            &model,
            session.as_mut(),
            &[0, 1],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
            |_| Ok(()),
        )?;
        assert_eq!(metrics.generated_tokens, 1);
        assert_eq!(metrics.decode_steps, 0);
        Ok(())
    }

    #[test]
    fn two_tokens_are_rejected_at_full_context() {
        let model = FakeModel::new(2, vec![1, 2]);
        let mut session = model.new_session().unwrap();
        let result = generate(
            &model,
            session.as_mut(),
            &[0, 1],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 2,
                ignore_stop_tokens: false,
            },
            |_| Ok(()),
        );
        assert!(result.is_err());
    }

    #[test]
    fn benchmark_mode_ignores_stop_tokens() -> Result<()> {
        let model = FakeModel::new(8, vec![3, 3, 3]);
        let mut session = model.new_session()?;
        let metrics = generate(
            &model,
            session.as_mut(),
            &[0],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 3,
                ignore_stop_tokens: true,
            },
            |_| Ok(()),
        )?;
        assert_eq!(metrics.generated_tokens, 3);
        assert_eq!(metrics.decode_steps, 2);
        assert_eq!(metrics.finish_reason, FinishReason::Length);
        Ok(())
    }

    #[test]
    fn cancellation_before_prefill_skips_model_work() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let mut session = model.new_session()?;
        let metrics = generate_with_control(
            &model,
            session.as_mut(),
            &[0],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
            || true,
            |_| Ok(()),
        )?;
        assert_eq!(metrics.finish_reason, FinishReason::Cancelled);
        assert_eq!(metrics.generated_tokens, 0);
        assert_eq!(fake_session(session.as_ref()).calls, 0);
        assert_eq!(session.cache_len(), 0);
        Ok(())
    }

    #[test]
    fn resumable_steps_match_blocking_generation_and_one_forward_per_quantum() -> Result<()> {
        let model = FakeModel::new(8, vec![1, 2, 3]);
        let config = GenerationConfig {
            max_new_tokens: 4,
            ignore_stop_tokens: false,
        };
        let mut sequence = GenerationSequence::new(&model, vec![0, 0], greedy(), config)?;
        let mut stepped_tokens = Vec::new();

        let first = sequence.step(
            &model,
            || false,
            |token| {
                stepped_tokens.push(token);
                Ok(())
            },
        )?;
        assert_eq!(first.input_tokens(), 2);
        assert_eq!(first.token(), Some(1));
        assert!(!first.is_finished());
        assert_eq!(fake_session(sequence.session_ref()).calls, 1);

        let second = sequence.step(
            &model,
            || false,
            |token| {
                stepped_tokens.push(token);
                Ok(())
            },
        )?;
        assert_eq!(second.input_tokens(), 1);
        assert_eq!(second.token(), Some(2));
        assert!(!second.is_finished());
        assert_eq!(fake_session(sequence.session_ref()).calls, 2);

        let terminal = sequence.step(
            &model,
            || false,
            |token| {
                stepped_tokens.push(token);
                Ok(())
            },
        )?;
        assert_eq!(terminal.input_tokens(), 1);
        assert_eq!(terminal.token(), None, "stop token must not be visible");
        let stepped_metrics = terminal
            .metrics()
            .expect("stop token finishes the sequence")
            .clone();
        assert_eq!(fake_session(sequence.session_ref()).calls, 3);
        assert_eq!(
            fake_session(sequence.session_ref()).forwards,
            [(vec![0, 0], 0), (vec![1], 2), (vec![2], 3),]
        );
        assert_eq!(stepped_tokens, [1, 2]);
        assert_eq!(sequence.cache_len(), 4);
        assert_eq!(stepped_metrics.prompt_tokens, 2);
        assert_eq!(stepped_metrics.generated_tokens, 2);
        assert_eq!(stepped_metrics.decode_steps, 2);
        assert_eq!(stepped_metrics.finish_reason, FinishReason::Stop);

        let mut blocking_session = model.new_session()?;
        let mut blocking_tokens = Vec::new();
        let blocking_metrics = generate(
            &model,
            blocking_session.as_mut(),
            &[0, 0],
            &mut greedy(),
            config,
            |token| {
                blocking_tokens.push(token);
                Ok(())
            },
        )?;
        assert_eq!(blocking_tokens, stepped_tokens);
        assert_eq!(
            blocking_metrics.prompt_tokens,
            stepped_metrics.prompt_tokens
        );
        assert_eq!(
            blocking_metrics.generated_tokens,
            stepped_metrics.generated_tokens
        );
        assert_eq!(blocking_metrics.decode_steps, stepped_metrics.decode_steps);
        assert_eq!(
            blocking_metrics.finish_reason,
            stepped_metrics.finish_reason
        );
        Ok(())
    }

    #[test]
    fn final_visible_token_finishes_in_the_same_quantum_and_is_sticky() -> Result<()> {
        let model = FakeModel::new(8, vec![1, 2]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 2,
                ignore_stop_tokens: false,
            },
        )?;
        let first = sequence.step(&model, || false, |_| Ok(()))?;
        assert_eq!(first.token(), Some(1));
        assert!(!first.is_finished());
        let terminal = sequence.step(&model, || false, |_| Ok(()))?;
        assert_eq!(terminal.token(), Some(2));
        let metrics = terminal
            .metrics()
            .expect("the final visible token must carry length metrics");
        assert_eq!(metrics.generated_tokens, 2);
        assert_eq!(metrics.decode_steps, 1);
        assert_eq!(metrics.finish_reason, FinishReason::Length);
        assert_eq!(sequence.cache_len(), 2);

        let calls = fake_session(sequence.session_ref()).calls;
        let repeated = sequence.step(
            &model,
            || false,
            |_| -> Result<()> { panic!("a finished sequence must not invoke its callback") },
        )?;
        assert_eq!(repeated.token(), None);
        assert!(repeated.is_finished());
        assert_eq!(fake_session(sequence.session_ref()).calls, calls);
        assert_eq!(sequence.status(), GenerationStatus::Finished);
        assert!(sequence.is_terminal());
        let synchronized_count =
            Arc::clone(&fake_session(sequence.session_ref()).synchronized_count);
        let reclaimed = sequence.reclaim_session(&model)?;
        assert_eq!(synchronized_count.load(Ordering::Relaxed), 1);
        assert_eq!(reclaimed.cache_len(), 0);
        assert_eq!(fake_session(reclaimed.as_ref()).calls, 0);
        Ok(())
    }

    #[test]
    fn zero_budget_step_finishes_without_model_work() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 0,
                ignore_stop_tokens: false,
            },
        )?;
        let step = sequence.step(
            &model,
            || true,
            |_| -> Result<()> { panic!("zero budget cannot produce a token") },
        )?;
        let metrics = step.metrics().expect("zero budget is terminal");
        assert_eq!(metrics.finish_reason, FinishReason::Length);
        assert_eq!(metrics.generated_tokens, 0);
        assert_eq!(metrics.decode_steps, 0);
        assert_eq!(metrics.time_to_first_token, Duration::ZERO);
        assert_eq!(fake_session(sequence.session_ref()).calls, 0);
        assert_eq!(sequence.cache_len(), 0);
        Ok(())
    }

    #[test]
    fn cancellation_between_quanta_skips_the_next_decode() -> Result<()> {
        let model = FakeModel::new(8, vec![1, 2]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 3,
                ignore_stop_tokens: false,
            },
        )?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let signal = Arc::clone(&cancelled);
        let first = sequence.step(
            &model,
            || false,
            move |_| {
                signal.store(true, Ordering::Release);
                Ok(())
            },
        )?;
        assert_eq!(first.token(), Some(1));
        assert!(!first.is_finished());
        assert_eq!(fake_session(sequence.session_ref()).calls, 1);

        let check = Arc::clone(&cancelled);
        let terminal = sequence.step(
            &model,
            move || check.load(Ordering::Acquire),
            |_| -> Result<()> { panic!("cancelled sequence cannot emit another token") },
        )?;
        let metrics = terminal.metrics().expect("cancellation is terminal");
        assert_eq!(metrics.finish_reason, FinishReason::Cancelled);
        assert_eq!(metrics.generated_tokens, 1);
        assert_eq!(metrics.decode_steps, 0);
        assert_eq!(fake_session(sequence.session_ref()).calls, 1);
        assert_eq!(sequence.cache_len(), 1);
        Ok(())
    }

    #[test]
    fn token_budget_chunks_prefill_without_sampling_intermediate_logits() -> Result<()> {
        let model = FakeModel::new(8, vec![1, 1, 1, 2]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0, 0, 0, 0, 0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 2,
                ignore_stop_tokens: false,
            },
        )?;
        let mut tokens = Vec::new();

        for expected_cache in [2, 4] {
            let step = sequence.step_with_budget(
                &model,
                2,
                || false,
                |token| {
                    tokens.push(token);
                    Ok(())
                },
            )?;
            assert_eq!(step.input_tokens(), 2);
            assert_eq!(step.token(), None);
            assert!(!step.is_finished());
            assert_eq!(sequence.cache_len(), expected_cache);
        }

        let final_prefill = sequence.step_with_budget(
            &model,
            2,
            || false,
            |token| {
                tokens.push(token);
                Ok(())
            },
        )?;
        assert_eq!(final_prefill.input_tokens(), 1);
        assert_eq!(final_prefill.token(), Some(1));
        assert!(!final_prefill.is_finished());
        assert_eq!(sequence.cache_len(), 5);

        let decode = sequence.step_with_budget(
            &model,
            1,
            || false,
            |token| {
                tokens.push(token);
                Ok(())
            },
        )?;
        assert_eq!(decode.input_tokens(), 1);
        assert_eq!(decode.token(), Some(2));
        assert!(decode.is_finished());
        assert_eq!(tokens, [1, 2]);
        assert_eq!(sequence.cache_len(), 6);
        assert_eq!(
            fake_session(sequence.session_ref()).forwards,
            [(vec![0, 0], 0), (vec![0, 0], 2), (vec![0], 4), (vec![1], 5),]
        );
        let metrics = decode.metrics().expect("second token exhausts the budget");
        assert_eq!(metrics.prompt_tokens, 5);
        assert_eq!(metrics.generated_tokens, 2);
        assert_eq!(metrics.decode_steps, 1);
        Ok(())
    }

    #[test]
    fn blocking_quantum_chunks_prefill_and_matches_one_shot_generation() -> Result<()> {
        let model = FakeModel::new(8, vec![1, 1, 1]);
        let prompt = [0, 0, 0, 0, 0];
        let config = GenerationConfig {
            max_new_tokens: 1,
            ignore_stop_tokens: false,
        };

        let mut chunked_session = model.new_session()?;
        let mut chunked_tokens = Vec::new();
        let chunked = generate_with_prefill_token_quantum(
            &model,
            chunked_session.as_mut(),
            &prompt,
            &mut greedy(),
            config,
            NonZeroUsize::new(2).expect("the test quantum is positive"),
            |token| {
                chunked_tokens.push(token);
                Ok(())
            },
        )?;
        assert_eq!(chunked_tokens, [1]);
        assert_eq!(
            fake_session(chunked_session.as_ref()).forwards,
            [(vec![0, 0], 0), (vec![0, 0], 2), (vec![0], 4),]
        );

        let mut one_shot_session = model.new_session()?;
        let mut one_shot_tokens = Vec::new();
        let one_shot = generate(
            &model,
            one_shot_session.as_mut(),
            &prompt,
            &mut greedy(),
            config,
            |token| {
                one_shot_tokens.push(token);
                Ok(())
            },
        )?;
        assert_eq!(one_shot_tokens, chunked_tokens);
        assert_eq!(
            fake_session(one_shot_session.as_ref()).forwards,
            [(prompt.to_vec(), 0)]
        );
        assert_eq!(chunked.prompt_tokens, one_shot.prompt_tokens);
        assert_eq!(chunked.generated_tokens, one_shot.generated_tokens);
        assert_eq!(chunked.decode_steps, one_shot.decode_steps);
        assert_eq!(chunked.finish_reason, one_shot.finish_reason);
        Ok(())
    }

    #[test]
    fn resumable_steps_preserve_sampler_rng_and_history() -> Result<()> {
        let scripted_logits = vec![vec![0.2_f32, 2.2, 2.0, -2.0]; 4];
        let sampling = SamplingConfig {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.7,
            seed: 73,
        };
        let prompt = vec![0_u32];
        let logits = Tensor::from_vec(scripted_logits[0].clone(), (1, 4), &Device::Cpu)?;

        let mut control_sampler = Sampler::new(sampling)?;
        let mut control_history = prompt.clone();
        let mut expected_tokens = Vec::new();
        for _ in 0..scripted_logits.len() {
            let token = control_sampler.sample(&logits, &control_history)?;
            expected_tokens.push(token);
            control_history.push(token);
        }
        let reset_tokens = (0..scripted_logits.len())
            .map(|_| Sampler::new(sampling)?.sample(&logits, &prompt))
            .collect::<Result<Vec<_>>>()?;
        assert_ne!(
            expected_tokens, reset_tokens,
            "fixture must distinguish persistent RNG/history from per-step reset"
        );

        let model = FakeModel::with_logits(8, scripted_logits);
        let mut sequence = GenerationSequence::new(
            &model,
            prompt,
            Sampler::new(sampling)?,
            GenerationConfig {
                max_new_tokens: 4,
                ignore_stop_tokens: true,
            },
        )?;
        let mut stepped_tokens = Vec::new();
        while !sequence.is_terminal() {
            sequence.step(
                &model,
                || false,
                |token| {
                    stepped_tokens.push(token);
                    Ok(())
                },
            )?;
        }
        assert_eq!(stepped_tokens, expected_tokens);
        Ok(())
    }

    #[test]
    fn generated_penalties_are_sequence_local_across_interleaved_steps() -> Result<()> {
        let model = FakeModel::with_logits(8, vec![vec![2.0, 1.0, 0.0, -1.0]; 3]);
        let make_sequence = || -> Result<GenerationSequence> {
            let mut sampler = Sampler::new(SamplingConfig {
                temperature: 0.0,
                repetition_penalty: 1.0,
                ..SamplingConfig::default()
            })?;
            sampler.set_penalties(1.5, 0.0)?;
            GenerationSequence::new(
                &model,
                vec![0],
                sampler,
                GenerationConfig {
                    max_new_tokens: 3,
                    ignore_stop_tokens: true,
                },
            )
        };
        let mut first = make_sequence()?;
        let mut second = make_sequence()?;
        let (mut a, mut b) = (Vec::new(), Vec::new());
        while !first.is_terminal() || !second.is_terminal() {
            if !first.is_terminal() {
                first.step(
                    &model,
                    || false,
                    |token| {
                        a.push(token);
                        Ok(())
                    },
                )?;
            }
            if !second.is_terminal() {
                second.step(
                    &model,
                    || false,
                    |token| {
                        b.push(token);
                        Ok(())
                    },
                )?;
            }
        }
        assert_eq!(a, vec![0, 1, 0]);
        assert_eq!(a, b);
        let mut fresh = make_sequence()?;
        let mut c = Vec::new();
        while !fresh.is_terminal() {
            fresh.step(
                &model,
                || false,
                |token| {
                    c.push(token);
                    Ok(())
                },
            )?;
        }
        assert_eq!(a, c);
        Ok(())
    }

    #[test]
    fn invalid_step_budget_does_not_poison_a_running_sequence() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
        )?;
        let error = sequence
            .step_with_budget(&model, 0, || false, |_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("budget must be positive"));
        assert_eq!(sequence.status(), GenerationStatus::Running);
        assert_eq!(fake_session(sequence.session_ref()).calls, 0);
        assert!(
            sequence
                .step_with_budget(&model, 1, || false, |_| Ok(()))?
                .is_finished()
        );
        Ok(())
    }

    #[test]
    fn two_owned_sequences_interleave_without_crossing_sessions() -> Result<()> {
        let model = FakeModel::new(8, vec![1, 2]);
        let config = GenerationConfig {
            max_new_tokens: 2,
            ignore_stop_tokens: false,
        };
        let mut first = GenerationSequence::new(&model, vec![0], greedy(), config)?;
        let mut second = GenerationSequence::new(&model, vec![0, 0], greedy(), config)?;
        let mut first_tokens = Vec::new();
        let mut second_tokens = Vec::new();

        assert!(
            !first
                .step(
                    &model,
                    || false,
                    |token| {
                        first_tokens.push(token);
                        Ok(())
                    }
                )?
                .is_finished()
        );
        assert!(
            !second
                .step(
                    &model,
                    || false,
                    |token| {
                        second_tokens.push(token);
                        Ok(())
                    }
                )?
                .is_finished()
        );
        assert!(
            first
                .step(
                    &model,
                    || false,
                    |token| {
                        first_tokens.push(token);
                        Ok(())
                    }
                )?
                .is_finished()
        );
        assert!(
            second
                .step(
                    &model,
                    || false,
                    |token| {
                        second_tokens.push(token);
                        Ok(())
                    }
                )?
                .is_finished()
        );

        assert_eq!(first_tokens, [1, 2]);
        assert_eq!(second_tokens, [1, 2]);
        assert_eq!(first.cache_len(), 2);
        assert_eq!(second.cache_len(), 3);
        assert_eq!(
            fake_session(first.session_ref()).forwards,
            [((vec![0]), 0), (vec![1], 1)]
        );
        assert_eq!(
            fake_session(second.session_ref()).forwards,
            [((vec![0, 0]), 0), (vec![1], 2)]
        );
        Ok(())
    }

    #[test]
    fn failed_resumable_step_is_terminal_without_replaying_a_forward() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 2,
                ignore_stop_tokens: false,
            },
        )?;
        let error = sequence
            .step(&model, || false, |_| anyhow::bail!("sink failed"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("sink failed"));
        assert_eq!(fake_session(sequence.session_ref()).calls, 1);
        let retry = sequence
            .step(&model, || false, |_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(retry.contains("cannot resume"));
        assert_eq!(fake_session(sequence.session_ref()).calls, 1);
        assert_eq!(sequence.status(), GenerationStatus::Failed);
        assert!(sequence.is_terminal());
        assert!(!sequence.is_finished());
        let clear_count = Arc::clone(&fake_session(sequence.session_ref()).clear_count);
        let reclaim = sequence.reclaim_session(&model).unwrap_err().to_string();
        assert!(reclaim.contains("cannot return a reusable session"));
        assert_eq!(clear_count.load(Ordering::Relaxed), 2);
        Ok(())
    }

    #[test]
    fn repeated_generation_previews_do_not_consume_or_fail_close_the_quantum() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0, 1, 2],
            greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
        )?;

        let GenerationPreview::Forward(first_preview) =
            sequence.preview_step_with_budget(&model, 2)?
        else {
            panic!("running sequence must preview a forward");
        };
        let GenerationPreview::Forward(second_preview) =
            sequence.preview_step_with_budget(&model, 2)?
        else {
            panic!("running sequence must preview a forward");
        };
        assert_eq!(first_preview.start_pos(), 0);
        assert_eq!(first_preview.input_tokens(), 2);
        assert_eq!(first_preview.next_cache_len(), 2);
        assert!(Arc::ptr_eq(&first_preview.owner, &second_preview.owner));
        assert_eq!(first_preview.quantum, second_preview.quantum);
        assert_eq!(first_preview.start_pos(), second_preview.start_pos());
        assert_eq!(first_preview.input_tokens(), second_preview.input_tokens());
        assert_eq!(sequence.status(), GenerationStatus::Running);
        assert_eq!(sequence.cache_len(), 0);

        let prepared =
            match sequence.prepare_step_from_preview(&model, &first_preview, 2, || false)? {
                GenerationPreparation::Forward(prepared) => prepared,
                GenerationPreparation::Complete(_) => panic!("previewed forward completed early"),
            };
        assert_eq!(prepared.input_ids(), &[0, 1]);
        assert_eq!(prepared.start_pos(), 0);
        Ok(())
    }

    #[test]
    fn cancellation_after_kv_preview_completes_without_an_in_flight_plan() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
        )?;
        let GenerationPreview::Forward(preview) = sequence.preview_step_with_budget(&model, 1)?
        else {
            panic!("running sequence must preview a forward");
        };
        let completed = sequence.prepare_step_from_preview(&model, &preview, 1, || true)?;
        let GenerationPreparation::Complete(step) = completed else {
            panic!("cancellation must not create a prepared forward");
        };
        assert_eq!(
            step.metrics().map(|metrics| metrics.finish_reason),
            Some(FinishReason::Cancelled)
        );
        assert_eq!(sequence.status(), GenerationStatus::Finished);
        Ok(())
    }

    #[test]
    fn foreign_or_wrong_budget_preview_is_rejected_before_fail_closed_preparation() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let config = GenerationConfig {
            max_new_tokens: 1,
            ignore_stop_tokens: false,
        };
        let first = GenerationSequence::new(&model, vec![0, 1], greedy(), config)?;
        let mut second = GenerationSequence::new(&model, vec![0, 1], greedy(), config)?;
        let GenerationPreview::Forward(foreign) = first.preview_step_with_budget(&model, 1)? else {
            panic!("running sequence must preview a forward");
        };

        let Err(error) = second.kv_reservation_input(&foreign) else {
            panic!("foreign preview must not expose a reservation input");
        };
        let error = error.to_string();
        assert!(error.contains("belongs to another sequence"));
        let Err(error) = second.prepare_step_from_preview(&model, &foreign, 1, || false) else {
            panic!("foreign preview must not prepare a quantum");
        };
        let error = error.to_string();
        assert!(error.contains("belongs to another sequence"));
        assert_eq!(second.status(), GenerationStatus::Running);

        let GenerationPreview::Forward(own) = second.preview_step_with_budget(&model, 1)? else {
            panic!("running sequence must preview a forward");
        };
        let Err(error) = second.prepare_step_from_preview(&model, &own, 2, || false) else {
            panic!("a preview must be bound to its token budget");
        };
        let error = error.to_string();
        assert!(error.contains("different input-token budget"));
        assert_eq!(second.status(), GenerationStatus::Running);

        let prepared = second.prepare_step_from_preview(&model, &own, 1, || false)?;
        assert!(matches!(prepared, GenerationPreparation::Forward(_)));
        Ok(())
    }

    #[test]
    fn prepared_quantum_identity_and_drop_are_fail_closed() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let config = GenerationConfig {
            max_new_tokens: 1,
            ignore_stop_tokens: false,
        };
        let mut first = GenerationSequence::new(&model, vec![0], greedy(), config)?;
        let mut second = GenerationSequence::new(&model, vec![1], greedy(), config)?;
        let first_plan = match first.prepare_step_with_budget(&model, 1, || false)? {
            GenerationPreparation::Forward(prepared) => prepared,
            GenerationPreparation::Complete(_) => panic!("live sequence must require a forward"),
        };
        let second_plan = match second.prepare_step_with_budget(&model, 1, || false)? {
            GenerationPreparation::Forward(prepared) => prepared,
            GenerationPreparation::Complete(_) => panic!("live sequence must require a forward"),
        };

        let Err(error) = first.prepared_batch_input(&second_plan) else {
            panic!("a prepared plan must not cross sequence identities");
        };
        assert!(error.to_string().contains("belongs to another sequence"));
        let Err(error) = first.prepared_batch_input(&first_plan) else {
            panic!("a plan API error must invalidate the original in-flight plan");
        };
        assert!(error.to_string().contains("no longer current"));
        assert_eq!(first.status(), GenerationStatus::Failed);
        assert_eq!(second.status(), GenerationStatus::Failed);
        drop(first_plan);
        drop(second_plan);
        let error = first
            .step(&model, || false, |_| Ok(()))
            .expect_err("dropping an in-flight plan must not make it resumable");
        assert!(
            error
                .to_string()
                .contains("cannot resume after a failed step")
        );
        Ok(())
    }

    #[test]
    fn prepared_completion_rejects_a_different_model_and_invalidates_the_plan() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let other_model = FakeModel::new(8, vec![3]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
        )?;
        let prepared = match sequence.prepare_step_with_budget(&model, 1, || false)? {
            GenerationPreparation::Forward(prepared) => prepared,
            GenerationPreparation::Complete(_) => panic!("live sequence must require a forward"),
        };
        let logits = {
            let session = sequence
                .session
                .as_deref_mut()
                .expect("live sequence owns its session");
            model.forward_tokens(session, prepared.input_ids(), prepared.start_pos())?
        };
        let error = sequence
            .complete_prepared(
                &other_model,
                prepared,
                logits,
                Duration::ZERO,
                || false,
                |_| Ok(()),
            )
            .expect_err("a different model must not complete the prepared quantum");
        assert!(error.to_string().contains("belongs to another model"));
        assert_eq!(sequence.status(), GenerationStatus::Failed);
        let retry = sequence
            .step(&model, || false, |_| Ok(()))
            .expect_err("a rejected completion must remain fail-closed");
        assert!(retry.to_string().contains("cannot resume"));
        Ok(())
    }

    #[test]
    fn dropping_a_failed_sequence_drains_and_clears_its_session() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 2,
                ignore_stop_tokens: false,
            },
        )?;
        let clear_count = Arc::clone(&fake_session(sequence.session_ref()).clear_count);
        let synchronized_count =
            Arc::clone(&fake_session(sequence.session_ref()).synchronized_count);
        let drop_drain_attempts = Arc::clone(&sequence.drop_drain_attempts);
        assert_eq!(clear_count.load(Ordering::Relaxed), 1);
        let error = sequence
            .step(&model, || false, |_| anyhow::bail!("sink failed"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("sink failed"));
        drop(sequence);
        assert_eq!(clear_count.load(Ordering::Relaxed), 2);
        assert_eq!(synchronized_count.load(Ordering::Relaxed), 1);
        assert_eq!(drop_drain_attempts.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn quiescent_reclaim_returns_a_cleared_reusable_session_without_a_drop_drain() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
        )?;
        let step = sequence.step(&model, || false, |_| Ok(()))?;
        assert!(step.metrics().is_some());

        let clear_count = Arc::clone(&fake_session(sequence.session_ref()).clear_count);
        let drop_count = Arc::clone(&fake_session(sequence.session_ref()).drop_count);
        let synchronized_count =
            Arc::clone(&fake_session(sequence.session_ref()).synchronized_count);
        let drop_drain_attempts = Arc::clone(&sequence.drop_drain_attempts);
        let session = sequence.reclaim_session_after_completion_barrier(&model)?;

        assert_eq!(drop_drain_attempts.load(Ordering::Relaxed), 0);
        assert_eq!(synchronized_count.load(Ordering::Relaxed), 0);
        assert_eq!(clear_count.load(Ordering::Relaxed), 2);
        assert_eq!(drop_count.load(Ordering::Relaxed), 0);
        assert_eq!(session.cache_len(), 0);
        model.validate_session(session.as_ref())?;
        drop(session);
        assert_eq!(drop_count.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn quiescent_reclaim_wrong_model_discards_without_a_drop_drain_or_clear() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let other_model = FakeModel::new(8, vec![1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
        )?;
        let step = sequence.step(&model, || false, |_| Ok(()))?;
        assert!(step.metrics().is_some());

        let clear_count = Arc::clone(&fake_session(sequence.session_ref()).clear_count);
        let drop_count = Arc::clone(&fake_session(sequence.session_ref()).drop_count);
        let drop_drain_attempts = Arc::clone(&sequence.drop_drain_attempts);
        let error = sequence
            .reclaim_session_after_completion_barrier(&other_model)
            .expect_err("a foreign model must not receive a reusable session")
            .to_string();

        assert!(error.contains("belongs to another model"));
        assert_eq!(drop_drain_attempts.load(Ordering::Relaxed), 0);
        assert_eq!(clear_count.load(Ordering::Relaxed), 1);
        assert_eq!(drop_count.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn quiescent_reclaim_clear_panic_discards_without_a_drop_drain_or_retry() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let mut sequence = GenerationSequence::new(
            &model,
            vec![0],
            greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
        )?;
        let step = sequence.step(&model, || false, |_| Ok(()))?;
        assert!(step.metrics().is_some());

        let session = fake_session(sequence.session_ref());
        let clear_count = Arc::clone(&session.clear_count);
        let drop_count = Arc::clone(&session.drop_count);
        session.panic_on_clear.store(true, Ordering::Relaxed);
        let drop_drain_attempts = Arc::clone(&sequence.drop_drain_attempts);
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = sequence.reclaim_session_after_completion_barrier(&model);
        }));

        assert!(panic.is_err());
        assert_eq!(drop_drain_attempts.load(Ordering::Relaxed), 0);
        assert_eq!(clear_count.load(Ordering::Relaxed), 1);
        assert_eq!(drop_count.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn isolated_request_cancels_after_a_visible_token_and_reuses_model() -> Result<()> {
        let model = FakeModel::new(8, vec![1, 2]);
        let mut session = model.new_session()?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancellation_check = Arc::clone(&cancelled);
        let cancellation_signal = Arc::clone(&cancelled);
        let metrics = run_generation_request(
            &model,
            session.as_mut(),
            &[0],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 3,
                ignore_stop_tokens: false,
            },
            move || cancellation_check.load(Ordering::Acquire),
            move |_| {
                cancellation_signal.store(true, Ordering::Release);
                Ok(())
            },
        )?;
        assert_eq!(metrics.finish_reason, FinishReason::Cancelled);
        assert_eq!(metrics.generated_tokens, 1);
        assert_eq!(metrics.decode_steps, 0);
        assert_eq!(session.cache_len(), 0);
        assert_eq!(fake_session(session.as_ref()).calls, 0);

        let next = run_generation_request(
            &model,
            session.as_mut(),
            &[0],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
            || false,
            |_| Ok(()),
        )?;
        assert_eq!(next.finish_reason, FinishReason::Length);
        assert_eq!(next.generated_tokens, 1);
        assert_eq!(session.cache_len(), 0);
        Ok(())
    }

    #[test]
    fn isolated_request_clears_cache_after_callback_error() {
        let model = FakeModel::new(8, vec![1]);
        let mut session = model.new_session().unwrap();
        let error = run_generation_request(
            &model,
            session.as_mut(),
            &[0],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
            || false,
            |_| anyhow::bail!("client disconnected"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("client disconnected"));
        assert_eq!(session.cache_len(), 0);
        assert_eq!(fake_session(session.as_ref()).calls, 0);
    }

    #[test]
    fn isolated_request_lease_clears_cache_during_unwind() {
        let model = FakeModel::new(8, vec![1]);
        let mut session = model.new_session().unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = run_generation_request(
                &model,
                session.as_mut(),
                &[0],
                &mut greedy(),
                GenerationConfig {
                    max_new_tokens: 1,
                    ignore_stop_tokens: false,
                },
                || false,
                |_| panic!("response sink panicked"),
            );
        }));
        assert!(panic.is_err());
        assert_eq!(session.cache_len(), 0);
        assert_eq!(fake_session(session.as_ref()).calls, 0);
    }

    #[test]
    fn isolated_request_reports_a_normal_path_cache_cleanup_panic() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let mut session = model.new_session()?;
        let panic_on_clear = Arc::clone(&fake_session(session.as_ref()).panic_on_clear);
        let error = run_generation_request(
            &model,
            session.as_mut(),
            &[0],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
            || false,
            |_| {
                panic_on_clear.store(true, Ordering::Relaxed);
                Ok(())
            },
        )
        .expect_err("a cache cleanup panic must become a request error")
        .to_string();
        assert!(error.contains("panicked while clearing the session cache"));
        session.clear_cache();
        assert_eq!(session.cache_len(), 0);
        Ok(())
    }

    #[test]
    fn isolated_request_cleanup_panic_does_not_double_panic_during_unwind() -> Result<()> {
        let model = FakeModel::new(8, vec![1]);
        let mut session = model.new_session()?;
        let panic_on_clear = Arc::clone(&fake_session(session.as_ref()).panic_on_clear);
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = run_generation_request(
                &model,
                session.as_mut(),
                &[0],
                &mut greedy(),
                GenerationConfig {
                    max_new_tokens: 1,
                    ignore_stop_tokens: false,
                },
                || false,
                |_| {
                    panic_on_clear.store(true, Ordering::Relaxed);
                    panic!("response sink panicked");
                },
            );
        }));
        assert!(panic.is_err());
        session.clear_cache();
        assert_eq!(session.cache_len(), 0);
        Ok(())
    }

    #[test]
    fn foreign_session_is_rejected_before_generation_cleanup_mutates_it() -> Result<()> {
        let owner = FakeModel::new(8, vec![1]);
        let other = FakeModel::new(8, vec![1]);
        let mut session = owner.new_session()?;
        let input = Tensor::new(&[[0_u32]], owner.device())?;
        let _ = owner.forward(session.as_mut(), &input, 0)?;
        assert_eq!(session.cache_len(), 1);

        let error = run_generation_request(
            &other,
            session.as_mut(),
            &[0],
            &mut greedy(),
            GenerationConfig {
                max_new_tokens: 1,
                ignore_stop_tokens: false,
            },
            || false,
            |_| Ok(()),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("belongs to another model"));
        assert_eq!(session.cache_len(), 1);
        assert_eq!(fake_session(session.as_ref()).calls, 1);
        Ok(())
    }
}
