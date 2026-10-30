use std::{
    collections::VecDeque,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU8, Ordering},
        mpsc as std_mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use tokio::sync::{mpsc, oneshot};

use crate::{
    agent::AgentPrompt,
    architecture::{
        CausalLm, CausalLmBatchCapability, CausalLmKvReservation, CausalLmSession,
        KvCacheExhausted, KvCacheReservationError, PrefixCacheDiagnostics,
    },
    chat::{ChatMessage, ChatTemplateOptions},
    generation::{
        FinishReason, GenerationConfig, GenerationForwardPreview, GenerationMetrics,
        GenerationPreparation, GenerationPreview, GenerationSequence, GenerationStep,
        PreparedGeneration, PreparedGenerationPhase, TokenControl,
    },
    grammar::{
        CompiledJsonSchema, CompiledToolCalls, JsonSchemaTokenConstraint, LazyToolCallConstraint,
    },
    output::StopFilter,
    runtime::{LoadedRuntime, ModelLoadConfig},
    sampling::{LogitsConstraint, Sampler, SamplingConfig},
    token_stream::TokenStream,
};

/// Runtime identity exposed by the serving layer without sharing model state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineMetadata {
    pub served_model: String,
    pub family: String,
    pub max_context: usize,
    pub device: String,
    pub dtype: String,
    /// Template options the worker applies to requests that leave them
    /// unset, so protocol defaults can follow the effective template mode.
    pub chat_template_defaults: ChatTemplateOptions,
}

/// Worker-owned automatic-prefix-cache diagnostics mirrored for read-only callers.
///
/// Reading this snapshot never calls into the model. `current == None` is
/// intentionally fail-closed: `observation` distinguishes a model without
/// diagnostics from a collection error (or a snapshot that has not yet been
/// sampled). Peak counters survive later eviction, reclamation, or collection
/// failure so a short-lived sharing maximum remains observable after a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnginePrefixCacheSnapshot {
    pub current: Option<PrefixCacheDiagnostics>,
    pub peak_physical_live_blocks: usize,
    pub peak_physical_live_references: usize,
    pub peak_physical_shared_blocks: usize,
    /// Largest active-sequence count reached by an admission observation.
    pub peak_concurrent_active_sequences: usize,
    /// First whole diagnostic captured when that sequence count was reached.
    /// Keeping the first sample prevents later forwards or cache publication
    /// at the same concurrency from contaminating admission-time evidence.
    pub peak_concurrent: Option<PrefixCacheDiagnostics>,
    pub active_sequences: usize,
    pub observation: Option<String>,
}

impl Default for EnginePrefixCacheSnapshot {
    fn default() -> Self {
        Self {
            current: None,
            peak_physical_live_blocks: 0,
            peak_physical_live_references: 0,
            peak_physical_shared_blocks: 0,
            peak_concurrent_active_sequences: 0,
            peak_concurrent: None,
            active_sequences: 0,
            observation: Some("prefix-cache diagnostics have not been sampled".to_owned()),
        }
    }
}

/// One protocol-shaped request submitted for worker-side model preflight.
#[derive(Debug, Clone)]
pub enum EngineInput {
    /// Render and tokenize a validated conversation with the model adapter.
    Chat(Vec<ChatMessage>),
    /// Render a normal text conversation and constrain only its generated
    /// assistant content. Keeping this distinct from tool-aware input avoids
    /// silently changing ordinary text requests.
    ConstrainedChat {
        messages: Vec<ChatMessage>,
        constraint: EngineConstraintSpec,
    },
    /// Render a protocol-independent prompt through a pinned native model
    /// profile. Forced calls carry a tool-arguments constraint, automatic
    /// tool choice carries a lazy native tool-call constraint, and completed
    /// tool history may also request an ordinary text turn without one.
    Agent {
        prompt: AgentPrompt,
        constraint: Option<EngineConstraintSpec>,
    },
    /// Tokenize an already rendered prompt without applying a chat template.
    RawPrompt(String),
    /// Use an already verified token sequence without template or tokenizer work.
    TokenIds(Vec<u32>),
}

impl EngineInput {
    fn constraint(&self) -> Option<&EngineConstraintSpec> {
        match self {
            Self::ConstrainedChat { constraint, .. } => Some(constraint),
            Self::Agent { constraint, .. } => constraint.as_ref(),
            Self::Chat(_) | Self::RawPrompt(_) | Self::TokenIds(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineConstraintPurpose {
    ResponseFormat,
    ToolArguments,
    /// Tool choice `auto`: text stays free until the model opens a call.
    ToolCall,
}

/// The grammar a request-local constraint enforces.
#[derive(Debug, Clone)]
pub enum EngineConstraintGrammar {
    /// The whole generated text is one schema-valid JSON value.
    JsonSchema(CompiledJsonSchema),
    /// At most one native call, constrained only after it opens.
    LazyToolCall(CompiledToolCalls),
}

/// Cloneable, already-validated request-local sampling constraint.
#[derive(Debug, Clone)]
pub struct EngineConstraintSpec {
    pub purpose: EngineConstraintPurpose,
    pub grammar: EngineConstraintGrammar,
}

impl EngineConstraintSpec {
    #[must_use]
    pub const fn response_format(schema: CompiledJsonSchema) -> Self {
        Self {
            purpose: EngineConstraintPurpose::ResponseFormat,
            grammar: EngineConstraintGrammar::JsonSchema(schema),
        }
    }

    #[must_use]
    pub const fn tool_arguments(schema: CompiledJsonSchema) -> Self {
        Self {
            purpose: EngineConstraintPurpose::ToolArguments,
            grammar: EngineConstraintGrammar::JsonSchema(schema),
        }
    }

    #[must_use]
    pub const fn tool_call(calls: CompiledToolCalls) -> Self {
        Self {
            purpose: EngineConstraintPurpose::ToolCall,
            grammar: EngineConstraintGrammar::LazyToolCall(calls),
        }
    }

    const fn label(&self) -> &'static str {
        match self.purpose {
            EngineConstraintPurpose::ResponseFormat => "response_format.json_schema",
            EngineConstraintPurpose::ToolArguments => "tool arguments JSON schema",
            EngineConstraintPurpose::ToolCall => "native tool call",
        }
    }
}

/// One protocol-shaped request submitted for worker-side model preflight.
#[derive(Debug, Clone)]
pub struct EngineRequest {
    pub input: EngineInput,
    pub sampling: SamplingConfig,
    /// Kept on EngineRequest rather than SamplingConfig to avoid changing unrelated sampling literals.
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub max_new_tokens: usize,
    pub ignore_stop_tokens: bool,
    pub stop: Vec<String>,
    /// Template options the request set explicitly. Unset options fall back
    /// to [`EngineConfig::chat_template_defaults`]; raw inputs bypass the
    /// template and therefore reject explicit options.
    pub chat_template: ChatTemplateOptions,
}

impl EngineRequest {
    /// Build the sampler with all request-local sampling parameters.
    pub(crate) fn sampler(&self) -> Result<Sampler> {
        let mut sampler = Sampler::new(self.sampling)?;
        sampler.set_penalties(self.presence_penalty, self.frequency_penalty)?;
        Ok(sampler)
    }
}

/// Per-request serving behavior that does not change prompt or sampling
/// semantics. Ordinary requests publish completed prompt blocks so later
/// requests may reuse them. Controlled benchmark waves disable publication to
/// keep their seeded prefix invariant across repeated measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineRequestPolicy {
    pub publish_prompt_prefix: bool,
}

impl EngineRequestPolicy {
    #[must_use]
    pub const fn benchmark_measured() -> Self {
        Self {
            publish_prompt_prefix: false,
        }
    }
}

impl Default for EngineRequestPolicy {
    fn default() -> Self {
        Self {
            publish_prompt_prefix: true,
        }
    }
}

/// Ordered worker events. Text is already tokenizer-decoded and therefore
/// safe to send without sharing tokenizer state with HTTP tasks. `token` is
/// present when a sampled token made the fragment available; terminal decoder
/// and stop-filter flushes carry `None` instead of duplicating token evidence.
#[derive(Debug)]
pub enum EngineEvent {
    Text {
        content: String,
        token: Option<EngineTokenEvidence>,
    },
    Finished(GenerationMetrics),
    Failed {
        message: String,
        kind: EngineFailureKind,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineTokenEvidence {
    pub token_id: u32,
    pub n_decoded: usize,
    pub prompt_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineFailureKind {
    InvalidRequest,
    ResourceExhausted,
    Internal,
    Fatal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnginePreflightError {
    pub message: String,
    pub kind: EngineFailureKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EngineStatus {
    Starting = 0,
    Ready = 1,
    Busy = 2,
    Poisoned = 3,
    Stopped = 4,
}

impl EngineStatus {
    fn from_atomic(value: u8) -> Self {
        match value {
            0 => Self::Starting,
            1 => Self::Ready,
            2 => Self::Busy,
            3 => Self::Poisoned,
            _ => Self::Stopped,
        }
    }

    #[must_use]
    pub const fn is_healthy(self) -> bool {
        matches!(self, Self::Ready | Self::Busy)
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model: ModelLoadConfig,
    pub served_model: String,
    /// Maximum requests waiting for an active sequence slot.
    pub queue_capacity: usize,
    /// Maximum independently stateful sequences resident on the model worker.
    pub max_active_sequences: usize,
    /// Positive total model-input token budget for one scheduler round.
    pub scheduler_token_budget: usize,
    /// Positive per-sequence prefill quantum; cannot exceed the round budget.
    pub prefill_token_quantum: usize,
    /// Capacity of each request-local worker event channel.
    pub event_capacity: usize,
    /// Continuous time a request may fail to accept its next event.
    pub event_send_timeout: Duration,
    /// Capture batch-only host timing sidecars. Disabled by default so the
    /// production scheduler does not allocate trace records or inspect lane
    /// phases on its hot path.
    pub capture_benchmark_trace: bool,
    /// Server-wide chat-template options for templated requests. A model
    /// whose adapter cannot honor them fails to start.
    pub chat_template_defaults: ChatTemplateOptions,
}

impl EngineConfig {
    #[must_use]
    pub fn new(model: ModelLoadConfig, served_model: impl Into<String>) -> Self {
        Self {
            model,
            served_model: served_model.into(),
            queue_capacity: 8,
            max_active_sequences: 2,
            scheduler_token_budget: 256,
            prefill_token_quantum: 128,
            event_capacity: 16,
            event_send_timeout: Duration::from_secs(30),
            capture_benchmark_trace: false,
            chat_template_defaults: ChatTemplateOptions::default(),
        }
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.queue_capacity > 0, "queue capacity must be positive");
        anyhow::ensure!(
            self.max_active_sequences > 0,
            "maximum active sequences must be positive"
        );
        anyhow::ensure!(
            self.scheduler_token_budget > 0,
            "scheduler token budget must be positive"
        );
        anyhow::ensure!(
            self.prefill_token_quantum > 0,
            "prefill token quantum must be positive"
        );
        anyhow::ensure!(
            self.prefill_token_quantum <= self.scheduler_token_budget,
            "prefill token quantum cannot exceed the scheduler token budget"
        );
        anyhow::ensure!(self.event_capacity > 0, "event capacity must be positive");
        anyhow::ensure!(
            !self.event_send_timeout.is_zero(),
            "event send timeout must be positive"
        );
        anyhow::ensure!(
            Instant::now()
                .checked_add(self.event_send_timeout)
                .is_some(),
            "event send timeout exceeds the platform limit"
        );
        anyhow::ensure!(
            !self.served_model.trim().is_empty(),
            "served model name must not be empty"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    Full,
    Unavailable(EngineStatus),
    Closed,
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => formatter.write_str("inference queue is full"),
            Self::Unavailable(status) => {
                write!(formatter, "inference worker is unavailable ({status:?})")
            }
            Self::Closed => formatter.write_str("inference worker has stopped"),
        }
    }
}

impl std::error::Error for SubmitError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchSubmitError {
    Empty,
    TooLarge { requested: usize, maximum: usize },
    Submit(SubmitError),
}

impl std::fmt::Display for BatchSubmitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => formatter.write_str("inference batch must not be empty"),
            Self::TooLarge { requested, maximum } => write!(
                formatter,
                "inference batch contains {requested} requests, but the atomic admission limit is {maximum}"
            ),
            Self::Submit(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for BatchSubmitError {}

/// Whole-packed-forward classification used by benchmark traces. A model
/// duration is assigned to exactly one variant; mixed-phase calls are never
/// split or double-counted across more convenient buckets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkForwardKind {
    SuffixPrefill,
    FirstToken,
    Decode,
    Mixed,
}

/// One logical lane participating in an observed packed model call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkForwardLane {
    /// Index within this trace's submitted batch. `None` means the honest
    /// packed boundary also contained a request outside the traced batch.
    pub member_index: Option<usize>,
    /// A lane is never `Mixed`; that variant belongs to the packed record.
    pub phase: BenchmarkForwardKind,
    pub start_pos: usize,
    pub input_tokens: usize,
}

/// Raw duration and geometry for one completed packed model call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkPackedForwardRecord {
    pub kind: BenchmarkForwardKind,
    pub lane_count: usize,
    pub input_tokens: usize,
    pub model_time: Duration,
    pub lanes: Vec<BenchmarkForwardLane>,
}

/// Complete, serialization-friendly timing sidecar for one atomic batch.
/// All timestamps have been reduced to durations; no worker-owned `Instant`
/// escapes through the public interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineBatchTrace {
    pub expected_members: usize,
    pub admitted_members: usize,
    pub terminal_members: usize,
    pub rejected_members: usize,
    pub queue_wait: Duration,
    pub admission_exclusive: Duration,
    pub prefix_lookup_attach_time: Duration,
    pub diagnostics_time: Duration,
    pub admission_diagnostics: Option<PrefixCacheDiagnostics>,
    pub admission_active_sequences: usize,
    pub suffix_prefill_time: Duration,
    pub first_token_model_time: Duration,
    pub decode_time: Duration,
    pub mixed_forward_time: Duration,
    pub packed_forwards: Vec<BenchmarkPackedForwardRecord>,
}

impl EngineBatchTrace {
    fn empty(expected_members: usize) -> Self {
        Self {
            expected_members,
            admitted_members: 0,
            terminal_members: 0,
            rejected_members: 0,
            queue_wait: Duration::ZERO,
            admission_exclusive: Duration::ZERO,
            prefix_lookup_attach_time: Duration::ZERO,
            diagnostics_time: Duration::ZERO,
            admission_diagnostics: None,
            admission_active_sequences: 0,
            suffix_prefill_time: Duration::ZERO,
            first_token_model_time: Duration::ZERO,
            decode_time: Duration::ZERO,
            mixed_forward_time: Duration::ZERO,
            packed_forwards: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct EngineBatchTraceState {
    trace: EngineBatchTrace,
    released_at: Option<Instant>,
    admission_started_at: Option<Instant>,
    admission_complete: bool,
}

/// Cloneable reader for a worker-owned benchmark trace. `try_read` remains
/// empty until admission has completed and every submitted member has reached
/// a terminal outcome, including preflight rejection and cancellation.
#[derive(Debug, Clone)]
pub struct EngineBatchTraceHandle {
    shared: Arc<Mutex<EngineBatchTraceState>>,
}

impl EngineBatchTraceHandle {
    fn new(expected_members: usize) -> Self {
        Self {
            shared: Arc::new(Mutex::new(EngineBatchTraceState {
                trace: EngineBatchTrace::empty(expected_members),
                released_at: None,
                admission_started_at: None,
                admission_complete: false,
            })),
        }
    }

    #[must_use]
    pub fn try_read(&self) -> Option<EngineBatchTrace> {
        let state = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (state.admission_complete && state.trace.terminal_members == state.trace.expected_members)
            .then(|| state.trace.clone())
    }

    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.try_read().is_some()
    }

    fn set_released_at(&self, released_at: Instant) {
        self.shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .released_at = Some(released_at);
    }

    fn begin_admission(&self, started_at: Instant) {
        let mut state = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(state.admission_started_at.is_none());
        state.trace.queue_wait = state.released_at.map_or(Duration::ZERO, |released_at| {
            started_at.saturating_duration_since(released_at)
        });
        state.admission_started_at = Some(started_at);
    }

    fn record_admitted(&self, prefix_lookup_attach_time: Duration) {
        let mut state = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.trace.admitted_members += 1;
        state.trace.prefix_lookup_attach_time += prefix_lookup_attach_time;
    }

    fn finish_admission(
        &self,
        finished_at: Instant,
        diagnostics_time: Duration,
        admission_diagnostics: Option<PrefixCacheDiagnostics>,
        admission_active_sequences: usize,
    ) {
        let mut state = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let wall_time = state.admission_started_at.map_or(Duration::ZERO, |start| {
            finished_at.saturating_duration_since(start)
        });
        state.trace.diagnostics_time = diagnostics_time;
        state.trace.admission_diagnostics = admission_diagnostics;
        state.trace.admission_active_sequences = admission_active_sequences;
        state.trace.admission_exclusive = wall_time
            .saturating_sub(state.trace.prefix_lookup_attach_time)
            .saturating_sub(diagnostics_time);
        state.admission_complete = true;
    }

    fn record_packed_forward(&self, record: BenchmarkPackedForwardRecord) {
        let mut state = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match record.kind {
            BenchmarkForwardKind::SuffixPrefill => {
                state.trace.suffix_prefill_time += record.model_time;
            }
            BenchmarkForwardKind::FirstToken => {
                state.trace.first_token_model_time += record.model_time;
            }
            BenchmarkForwardKind::Decode => state.trace.decode_time += record.model_time,
            BenchmarkForwardKind::Mixed => {
                state.trace.mixed_forward_time += record.model_time;
            }
        }
        state.trace.packed_forwards.push(record);
    }

    fn mark_terminal(&self, rejected: bool) {
        let mut state = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.trace.terminal_members += 1;
        state.trace.rejected_members += usize::from(rejected);
        debug_assert!(state.trace.terminal_members <= state.trace.expected_members);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchAdmissionState {
    Pending,
    Ready,
    Aborted,
}

/// Shared release barrier for one contiguous batch of queued requests.
///
/// The worker may dequeue the first job while the caller is still sending its
/// peers, but it cannot begin admission until every send has succeeded. The
/// condition variable keeps this a blocking worker-side boundary rather than
/// a timing-sensitive spin loop.
struct BatchAdmissionGate {
    expected_jobs: usize,
    state: Mutex<BatchAdmissionState>,
    changed: Condvar,
}

impl BatchAdmissionGate {
    fn pending(expected_jobs: usize) -> Self {
        debug_assert!(expected_jobs > 0);
        Self {
            expected_jobs,
            state: Mutex::new(BatchAdmissionState::Pending),
            changed: Condvar::new(),
        }
    }

    fn ready_at(&self, released_at: Instant) -> Instant {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *state == BatchAdmissionState::Pending {
            *state = BatchAdmissionState::Ready;
            self.changed.notify_all();
        }
        released_at
    }

    fn abort(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *state == BatchAdmissionState::Pending {
            *state = BatchAdmissionState::Aborted;
            self.changed.notify_all();
        }
    }

    fn wait_until_released(&self) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while *state == BatchAdmissionState::Pending {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        *state == BatchAdmissionState::Ready
    }

    #[cfg(test)]
    fn state(&self) -> BatchAdmissionState {
        *self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Abort-on-drop guard for a batch that has not reached its release point.
///
/// This covers ordinary send errors as well as unwinding between sends, so a
/// worker waiting on the first queued job cannot be stranded indefinitely.
struct ArmedBatchAdmissionGate {
    gate: Option<Arc<BatchAdmissionGate>>,
}

impl ArmedBatchAdmissionGate {
    fn new(gate: Arc<BatchAdmissionGate>) -> Self {
        Self { gate: Some(gate) }
    }

    fn release_at(mut self, released_at: Instant) -> Instant {
        self.gate
            .take()
            .expect("an armed batch admission gate is present")
            .ready_at(released_at)
    }
}

impl Drop for ArmedBatchAdmissionGate {
    fn drop(&mut self) {
        if let Some(gate) = self.gate.take() {
            gate.abort();
        }
    }
}

struct BatchTraceMember {
    handle: EngineBatchTraceHandle,
    member_index: usize,
    terminal_recorded: bool,
}

impl BatchTraceMember {
    fn mark_terminal(&mut self, rejected: bool) {
        if !self.terminal_recorded {
            self.handle.mark_terminal(rejected);
            self.terminal_recorded = true;
        }
    }
}

struct InferenceJob {
    request: EngineRequest,
    request_policy: EngineRequestPolicy,
    events: mpsc::Sender<EngineEvent>,
    cancelled: Arc<AtomicBool>,
    preflight: Option<oneshot::Sender<std::result::Result<(), EnginePreflightError>>>,
    terminal_failure: Arc<Mutex<Option<EnginePreflightError>>>,
    event_send_timeout: Duration,
    batch_admission: Option<Arc<BatchAdmissionGate>>,
    batch_trace: Option<BatchTraceMember>,
}

impl Drop for InferenceJob {
    fn drop(&mut self) {
        if let Some(member) = self.batch_trace.as_mut() {
            member.mark_terminal(false);
        }
    }
}

/// Tickets returned by one atomically released batch submission.
///
/// `released_at` is captured immediately before the ready signal. Request
/// construction and queue staging therefore remain outside measurements that
/// deliberately use this common batch-release boundary.
pub struct EngineBatchSubmission {
    tickets: Vec<EngineTicket>,
    released_at: Instant,
    benchmark_trace: Option<EngineBatchTraceHandle>,
}

impl EngineBatchSubmission {
    #[must_use]
    pub fn released_at(&self) -> Instant {
        self.released_at
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tickets.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tickets.is_empty()
    }

    #[must_use]
    pub fn into_tickets(self) -> Vec<EngineTicket> {
        self.tickets
    }

    #[must_use]
    pub fn benchmark_trace(&self) -> Option<EngineBatchTraceHandle> {
        self.benchmark_trace.clone()
    }

    #[must_use]
    pub fn into_parts(self) -> (Vec<EngineTicket>, Option<EngineBatchTraceHandle>) {
        (self.tickets, self.benchmark_trace)
    }
}

/// Cloneable admission handle for one dedicated, model-owning worker thread.
#[derive(Clone)]
pub struct EngineHandle {
    jobs: mpsc::Sender<InferenceJob>,
    submit_lock: Arc<Mutex<()>>,
    status: Arc<AtomicU8>,
    metadata: EngineMetadata,
    prefix_cache_snapshot: Arc<Mutex<EnginePrefixCacheSnapshot>>,
    event_capacity: usize,
    event_send_timeout: Duration,
    maximum_atomic_batch: usize,
    capture_benchmark_trace: bool,
}

impl EngineHandle {
    /// Start a worker and wait until its model is either fully loaded or fails.
    /// The model is constructed inside the worker and never crosses threads.
    pub fn start(config: EngineConfig) -> Result<Self> {
        config.validate()?;

        let (jobs, receiver) = mpsc::channel(config.queue_capacity);
        let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
        let status = Arc::new(AtomicU8::new(EngineStatus::Starting as u8));
        let worker_status = Arc::clone(&status);
        let prefix_cache_snapshot = Arc::new(Mutex::new(EnginePrefixCacheSnapshot::default()));
        let worker_prefix_cache_snapshot = Arc::clone(&prefix_cache_snapshot);
        let worker_model = config.model.clone();
        let served_model = config.served_model.clone();
        let scheduler = SchedulerConfig {
            max_active_sequences: config.max_active_sequences,
            scheduler_token_budget: config.scheduler_token_budget,
            prefill_token_quantum: config.prefill_token_quantum,
            chat_template_defaults: config.chat_template_defaults,
        };
        thread::Builder::new()
            .name("macllm-inference".to_owned())
            .spawn(move || {
                worker_main(
                    worker_model,
                    served_model,
                    receiver,
                    worker_status,
                    ready_tx,
                    scheduler,
                    worker_prefix_cache_snapshot,
                );
            })
            .context("failed to spawn inference worker")?;

        let metadata = ready_rx
            .recv()
            .context("inference worker stopped during model loading")?
            .map_err(anyhow::Error::msg)?;
        Ok(Self {
            jobs,
            submit_lock: Arc::new(Mutex::new(())),
            status,
            metadata,
            prefix_cache_snapshot,
            event_capacity: config.event_capacity,
            event_send_timeout: config.event_send_timeout,
            maximum_atomic_batch: config.queue_capacity.min(config.max_active_sequences),
            capture_benchmark_trace: config.capture_benchmark_trace,
        })
    }

    #[must_use]
    pub fn metadata(&self) -> &EngineMetadata {
        &self.metadata
    }

    #[must_use]
    pub fn status(&self) -> EngineStatus {
        EngineStatus::from_atomic(self.status.load(Ordering::Acquire))
    }

    /// Return the latest worker-published prefix-cache snapshot.
    ///
    /// This is a host-only clone of bounded counters; it never enters the
    /// model or blocks the inference worker on device work.
    #[must_use]
    pub fn prefix_cache_snapshot(&self) -> EnginePrefixCacheSnapshot {
        self.prefix_cache_snapshot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Enqueue a request in the bounded waiting queue.
    ///
    /// Success does not yet reserve an active sequence slot; callers use
    /// [`EngineTicket::wait_for_preflight`] to wait for admission.
    pub fn submit(&self, request: EngineRequest) -> Result<EngineTicket, SubmitError> {
        self.submit_with_policy(request, EngineRequestPolicy::default())
    }

    /// Enqueue one request with an explicit prompt-prefix publication policy.
    pub fn submit_with_policy(
        &self,
        request: EngineRequest,
        request_policy: EngineRequestPolicy,
    ) -> Result<EngineTicket, SubmitError> {
        let _submit_guard = self
            .submit_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let status = self.status();
        if !status.is_healthy() {
            return Err(SubmitError::Unavailable(status));
        }
        let (events, receiver) = mpsc::channel(self.event_capacity);
        let (preflight, readiness) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let terminal_failure = Arc::new(Mutex::new(None));
        let job = InferenceJob {
            request,
            request_policy,
            events,
            cancelled: Arc::clone(&cancelled),
            preflight: Some(preflight),
            terminal_failure: Arc::clone(&terminal_failure),
            event_send_timeout: self.event_send_timeout,
            batch_admission: None,
            batch_trace: None,
        };
        match self.jobs.try_send(job) {
            Ok(()) => Ok(EngineTicket {
                receiver,
                cancelled,
                preflight: Some(readiness),
                terminal_failure,
            }),
            Err(mpsc::error::TrySendError::Full(_)) => Err(SubmitError::Full),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(SubmitError::Closed),
        }
    }

    /// Enqueue and release one admission-atomic request batch.
    ///
    /// All jobs share a worker-side gate. The first dequeued job cannot enter
    /// model preflight until every peer send succeeds, and the scheduler then
    /// drains the contiguous group into its available slots before the first
    /// forward. A partial send aborts the gate and every already-queued member
    /// is discarded without model work.
    pub fn submit_batch(
        &self,
        requests: Vec<EngineRequest>,
    ) -> Result<EngineBatchSubmission, BatchSubmitError> {
        self.submit_batch_with_policy(
            requests
                .into_iter()
                .map(|request| (request, EngineRequestPolicy::default()))
                .collect(),
        )
    }

    /// Enqueue an admission-atomic batch with a request-local publication
    /// policy for every member.
    pub fn submit_batch_with_policy(
        &self,
        requests: Vec<(EngineRequest, EngineRequestPolicy)>,
    ) -> Result<EngineBatchSubmission, BatchSubmitError> {
        if requests.is_empty() {
            return Err(BatchSubmitError::Empty);
        }
        if requests.len() > self.maximum_atomic_batch {
            return Err(BatchSubmitError::TooLarge {
                requested: requests.len(),
                maximum: self.maximum_atomic_batch,
            });
        }
        let _submit_guard = self
            .submit_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let status = self.status();
        if !status.is_healthy() {
            return Err(BatchSubmitError::Submit(SubmitError::Unavailable(status)));
        }

        let gate = Arc::new(BatchAdmissionGate::pending(requests.len()));
        let armed_gate = ArmedBatchAdmissionGate::new(Arc::clone(&gate));
        let benchmark_trace = self
            .capture_benchmark_trace
            .then(|| EngineBatchTraceHandle::new(requests.len()));
        let mut tickets = Vec::with_capacity(requests.len());
        for (member_index, (request, request_policy)) in requests.into_iter().enumerate() {
            let (events, receiver) = mpsc::channel(self.event_capacity);
            let (preflight, readiness) = oneshot::channel();
            let cancelled = Arc::new(AtomicBool::new(false));
            let terminal_failure = Arc::new(Mutex::new(None));
            let job = InferenceJob {
                request,
                request_policy,
                events,
                cancelled: Arc::clone(&cancelled),
                preflight: Some(preflight),
                terminal_failure: Arc::clone(&terminal_failure),
                event_send_timeout: self.event_send_timeout,
                batch_admission: Some(Arc::clone(&gate)),
                batch_trace: benchmark_trace.as_ref().map(|handle| BatchTraceMember {
                    handle: handle.clone(),
                    member_index,
                    terminal_recorded: false,
                }),
            };
            match self.jobs.try_send(job) {
                Ok(()) => tickets.push(EngineTicket {
                    receiver,
                    cancelled,
                    preflight: Some(readiness),
                    terminal_failure,
                }),
                Err(mpsc::error::TrySendError::Full(_)) => {
                    return Err(BatchSubmitError::Submit(SubmitError::Full));
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    return Err(BatchSubmitError::Submit(SubmitError::Closed));
                }
            }
        }
        let released_at = Instant::now();
        if let Some(trace) = &benchmark_trace {
            trace.set_released_at(released_at);
        }
        let gate_released_at = armed_gate.release_at(released_at);
        debug_assert_eq!(gate_released_at, released_at);
        Ok(EngineBatchSubmission {
            tickets,
            released_at,
            benchmark_trace,
        })
    }
}

/// A request-local event stream. Dropping it is the network-disconnect
/// cancellation path; only the worker is allowed to mutate or clear model KV.
pub struct EngineTicket {
    receiver: mpsc::Receiver<EngineEvent>,
    cancelled: Arc<AtomicBool>,
    preflight: Option<oneshot::Receiver<std::result::Result<(), EnginePreflightError>>>,
    terminal_failure: Arc<Mutex<Option<EnginePreflightError>>>,
}

impl EngineTicket {
    #[cfg(test)]
    pub(crate) fn channel(
        capacity: usize,
        result: std::result::Result<(), EnginePreflightError>,
    ) -> (mpsc::Sender<EngineEvent>, Arc<AtomicBool>, Self) {
        let (events, receiver) = mpsc::channel(capacity);
        let cancelled = Arc::new(AtomicBool::new(false));
        let terminal_failure = Arc::new(Mutex::new(None));
        let (preflight, readiness) = oneshot::channel();
        let _ = preflight.send(result);
        (
            events,
            Arc::clone(&cancelled),
            Self {
                receiver,
                cancelled,
                preflight: Some(readiness),
                terminal_failure,
            },
        )
    }

    /// Wait until the worker has rendered and tokenized the prompt, validated
    /// its context/sampling contract and reserved an active sequence session.
    pub async fn wait_for_preflight(&mut self) -> std::result::Result<(), EnginePreflightError> {
        let Some(preflight) = self.preflight.take() else {
            return Ok(());
        };
        preflight.await.unwrap_or_else(|_| {
            Err(EnginePreflightError {
                message: "inference worker stopped during request preflight".to_owned(),
                kind: EngineFailureKind::Fatal,
            })
        })
    }

    pub async fn recv(&mut self) -> Option<EngineEvent> {
        match self.receiver.recv().await {
            Some(event) => Some(event),
            None => self
                .terminal_failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .map(|failure| EngineEvent::Failed {
                    message: failure.message,
                    kind: failure.kind,
                }),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

impl Drop for EngineTicket {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[derive(Debug, Clone, Copy)]
struct SchedulerConfig {
    max_active_sequences: usize,
    scheduler_token_budget: usize,
    prefill_token_quantum: usize,
    chat_template_defaults: ChatTemplateOptions,
}

fn sample_prefix_cache_snapshot(
    model: &dyn CausalLm,
    shared: &Arc<Mutex<EnginePrefixCacheSnapshot>>,
    active_sequences: usize,
    observation: &'static str,
) -> Option<PrefixCacheDiagnostics> {
    // Diagnostics are deliberately outside the scheduler's fatal path. Even
    // an incorrect implementation that panics must not poison healthy model
    // inference merely because a demo requested observability.
    let result = catch_unwind(AssertUnwindSafe(|| model.prefix_cache_diagnostics()));
    let mut snapshot = shared
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    snapshot.active_sequences = active_sequences;
    match result {
        Ok(Ok(Some(current))) => {
            snapshot.peak_physical_live_blocks = snapshot
                .peak_physical_live_blocks
                .max(current.physical_live_blocks);
            snapshot.peak_physical_live_references = snapshot
                .peak_physical_live_references
                .max(current.physical_live_references);
            snapshot.peak_physical_shared_blocks = snapshot
                .peak_physical_shared_blocks
                .max(current.physical_shared_blocks);
            let replace_concurrent = active_sequences > snapshot.peak_concurrent_active_sequences
                || (active_sequences == snapshot.peak_concurrent_active_sequences
                    && snapshot.peak_concurrent.is_none());
            if replace_concurrent {
                snapshot.peak_concurrent_active_sequences = active_sequences;
                snapshot.peak_concurrent = Some(current);
            }
            snapshot.current = Some(current);
            snapshot.observation = None;
            Some(current)
        }
        Ok(Ok(None)) => {
            snapshot.current = None;
            snapshot.observation = Some(format!(
                "{observation}: loaded model does not expose prefix-cache diagnostics"
            ));
            None
        }
        Ok(Err(error)) => {
            snapshot.current = None;
            snapshot.observation = Some(format!(
                "{observation}: failed to collect prefix-cache diagnostics: {error}"
            ));
            None
        }
        Err(_) => {
            snapshot.current = None;
            snapshot.observation = Some(format!(
                "{observation}: prefix-cache diagnostics collection panicked"
            ));
            None
        }
    }
}

fn refresh_prefix_cache_snapshot(
    model: &dyn CausalLm,
    shared: &Arc<Mutex<EnginePrefixCacheSnapshot>>,
    active_sequences: usize,
    observation: &'static str,
) {
    let _ = sample_prefix_cache_snapshot(model, shared, active_sequences, observation);
}

fn worker_main(
    model: ModelLoadConfig,
    served_model: String,
    jobs: mpsc::Receiver<InferenceJob>,
    status: Arc<AtomicU8>,
    ready: std_mpsc::SyncSender<std::result::Result<EngineMetadata, String>>,
    scheduler_config: SchedulerConfig,
    prefix_cache_snapshot: Arc<Mutex<EnginePrefixCacheSnapshot>>,
) {
    let runtime = match LoadedRuntime::load(&model) {
        Ok(runtime) => runtime,
        Err(error) => {
            status.store(EngineStatus::Stopped as u8, Ordering::Release);
            let _ = ready.send(Err(format!("failed to load model: {error:#}")));
            return;
        }
    };
    refresh_prefix_cache_snapshot(
        runtime.model.as_ref(),
        &prefix_cache_snapshot,
        0,
        "model loaded",
    );
    if runtime.model.mixed_batch_capability() != CausalLmBatchCapability::PackedRagged {
        status.store(EngineStatus::Stopped as u8, Ordering::Release);
        let _ = ready.send(Err(
            "loaded model does not support packed-ragged serving batches".to_owned(),
        ));
        return;
    }
    let metadata = EngineMetadata {
        served_model,
        family: runtime.adapter.family().to_owned(),
        max_context: runtime.model.runtime_spec().max_context,
        device: if runtime.model.device().is_metal() {
            "metal"
        } else {
            "cpu"
        }
        .to_owned(),
        dtype: format!("{:?}", runtime.dtype).to_ascii_lowercase(),
        chat_template_defaults: scheduler_config
            .chat_template_defaults
            .or(runtime.adapter.chat_template_defaults()),
    };
    let rendered = match runtime.adapter.format_serving_chat_probe() {
        Ok(rendered) => rendered,
        Err(error) => {
            status.store(EngineStatus::Stopped as u8, Ordering::Release);
            let _ = ready.send(Err(format!(
                "model has no supported serving chat template: {error:#}"
            )));
            return;
        }
    };
    if !scheduler_config.chat_template_defaults.is_default()
        && let Err(error) = runtime.adapter.format_chat_messages_with(
            &[ChatMessage::user("health check")],
            scheduler_config.chat_template_defaults,
        )
    {
        status.store(EngineStatus::Stopped as u8, Ordering::Release);
        let _ = ready.send(Err(format!(
            "model cannot honor the server chat template options: {error:#}"
        )));
        return;
    }
    match runtime.encode(rendered, false) {
        Ok(tokens) if !tokens.is_empty() => {}
        Ok(_) => {
            status.store(EngineStatus::Stopped as u8, Ordering::Release);
            let _ = ready.send(Err(
                "model chat template tokenizer preflight produced no tokens".to_owned(),
            ));
            return;
        }
        Err(error) => {
            status.store(EngineStatus::Stopped as u8, Ordering::Release);
            let _ = ready.send(Err(format!(
                "model chat template failed tokenizer preflight: {error:#}"
            )));
            return;
        }
    }
    let mut sessions = Vec::with_capacity(scheduler_config.max_active_sequences);
    for index in 0..scheduler_config.max_active_sequences {
        let session = match runtime.model.new_session() {
            Ok(session) => session,
            Err(error) => {
                status.store(EngineStatus::Stopped as u8, Ordering::Release);
                let _ = ready.send(Err(format!(
                    "failed to create serving inference session {} of {}: {error:#}",
                    index + 1,
                    scheduler_config.max_active_sequences
                )));
                return;
            }
        };
        if let Err(error) = runtime.model.validate_session(session.as_ref()) {
            status.store(EngineStatus::Stopped as u8, Ordering::Release);
            let _ = ready.send(Err(format!(
                "failed to validate serving inference session {} of {}: {error:#}",
                index + 1,
                scheduler_config.max_active_sequences
            )));
            return;
        }
        sessions.push(session);
    }
    status.store(EngineStatus::Ready as u8, Ordering::Release);
    if ready.send(Ok(metadata)).is_err() {
        status.store(EngineStatus::Stopped as u8, Ordering::Release);
        return;
    }

    ServingScheduler::new(
        &runtime,
        jobs,
        status,
        sessions,
        scheduler_config,
        prefix_cache_snapshot,
    )
    .run();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivePhase {
    Generating,
    Delivering,
}

struct ActiveSequence<'a> {
    job: InferenceJob,
    sequence: Option<GenerationSequence>,
    reclaimed_session: Option<Box<dyn CausalLmSession>>,
    prompt_tokens: usize,
    stream: TokenStream<'a>,
    stop_filter: StopFilter,
    phase: ActivePhase,
    pending: VecDeque<EngineEvent>,
    blocked_since: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlushState {
    Drained,
    Blocked,
    Abandoned,
}

struct FlushResult {
    state: FlushState,
    sent: usize,
}

impl ActiveSequence<'_> {
    fn flush_pending(&mut self, now: Instant) -> FlushResult {
        if self.job.cancelled.load(Ordering::Acquire) || self.job.events.is_closed() {
            self.job.cancelled.store(true, Ordering::Release);
            self.pending.clear();
            return FlushResult {
                state: FlushState::Abandoned,
                sent: 0,
            };
        }

        let mut sent = 0;
        while let Some(event) = self.pending.pop_front() {
            match self.job.events.try_send(event) {
                Ok(()) => {
                    sent += 1;
                    // A client that accepted an event is making progress. If
                    // the following event is full, it gets a fresh timeout.
                    self.blocked_since = None;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    self.job.cancelled.store(true, Ordering::Release);
                    self.pending.clear();
                    return FlushResult {
                        state: FlushState::Abandoned,
                        sent,
                    };
                }
                Err(mpsc::error::TrySendError::Full(event)) => {
                    self.pending.push_front(event);
                    let blocked_since = self.blocked_since.get_or_insert(now);
                    if now.saturating_duration_since(*blocked_since) >= self.job.event_send_timeout
                    {
                        self.job.cancelled.store(true, Ordering::Release);
                        self.pending.clear();
                        return FlushResult {
                            state: FlushState::Abandoned,
                            sent,
                        };
                    }
                    return FlushResult {
                        state: FlushState::Blocked,
                        sent,
                    };
                }
            }
        }
        self.blocked_since = None;
        FlushResult {
            state: FlushState::Drained,
            sent,
        }
    }
}

struct DriveResult {
    made_progress: bool,
    input_tokens: usize,
    visits: usize,
}

enum VisitAction {
    Keep,
    Remove,
    Complete {
        step: GenerationStep,
        stream_error: Option<String>,
    },
    Preview {
        preview: GenerationForwardPreview,
        input_token_budget: usize,
    },
    Forward(Option<PreparedGeneration>),
}

struct CollectedVisits {
    actions: Vec<VisitAction>,
    input_tokens: usize,
    made_progress: bool,
}

struct ReservedVisits {
    reservation: Option<Box<dyn CausalLmKvReservation>>,
    input_tokens: usize,
    made_progress: bool,
}

struct PendingBenchmarkForward {
    handle: EngineBatchTraceHandle,
    kind: BenchmarkForwardKind,
    input_tokens: usize,
    lanes: Vec<BenchmarkForwardLane>,
}

struct SubmittedBatchOutput {
    logits: candle_core::Tensor,
    model_start: Instant,
    benchmark_traces: Vec<PendingBenchmarkForward>,
}

struct CompletedBatchOutput {
    rows: Vec<candle_core::Tensor>,
    model_time: Duration,
}

const fn benchmark_forward_kind(phase: PreparedGenerationPhase) -> BenchmarkForwardKind {
    match phase {
        PreparedGenerationPhase::SuffixPrefill => BenchmarkForwardKind::SuffixPrefill,
        PreparedGenerationPhase::FirstToken => BenchmarkForwardKind::FirstToken,
        PreparedGenerationPhase::Decode => BenchmarkForwardKind::Decode,
    }
}

struct ServingScheduler<'a> {
    runtime: &'a LoadedRuntime,
    jobs: mpsc::Receiver<InferenceJob>,
    status: Arc<AtomicU8>,
    sessions: Vec<Box<dyn CausalLmSession>>,
    active: VecDeque<ActiveSequence<'a>>,
    pending_atomic_batch: Option<Vec<InferenceJob>>,
    config: SchedulerConfig,
    jobs_closed: bool,
    round_budget_remaining: usize,
    round_visits_remaining: usize,
    forward_needs_drain: bool,
    prefix_cache_snapshot: Arc<Mutex<EnginePrefixCacheSnapshot>>,
}

impl<'a> ServingScheduler<'a> {
    fn new(
        runtime: &'a LoadedRuntime,
        jobs: mpsc::Receiver<InferenceJob>,
        status: Arc<AtomicU8>,
        sessions: Vec<Box<dyn CausalLmSession>>,
        config: SchedulerConfig,
        prefix_cache_snapshot: Arc<Mutex<EnginePrefixCacheSnapshot>>,
    ) -> Self {
        debug_assert_eq!(sessions.len(), config.max_active_sequences);
        Self {
            runtime,
            jobs,
            status,
            sessions,
            active: VecDeque::with_capacity(config.max_active_sequences),
            pending_atomic_batch: None,
            config,
            jobs_closed: false,
            round_budget_remaining: 0,
            round_visits_remaining: 0,
            forward_needs_drain: false,
            prefix_cache_snapshot,
        }
    }

    fn refresh_prefix_cache_snapshot(&self, observation: &'static str) {
        refresh_prefix_cache_snapshot(
            self.runtime.model.as_ref(),
            &self.prefix_cache_snapshot,
            self.active.len(),
            observation,
        );
    }

    fn publish_active_sequence_count(&self) {
        self.prefix_cache_snapshot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active_sequences = self.active.len();
    }

    fn run(&mut self) {
        loop {
            if self.active.is_empty() && self.pending_atomic_batch.is_some() {
                let admission = catch_unwind(AssertUnwindSafe(|| {
                    self.admit_pending_atomic_batch_if_available()
                }));
                match admission {
                    Ok(Ok(_)) => {}
                    Ok(Err(error)) => {
                        self.poison(format!("inference worker failed: {error:#}"));
                        return;
                    }
                    Err(_) => {
                        self.poison("inference worker panicked during batch admission".to_owned());
                        return;
                    }
                }
            }
            if self.active.is_empty() {
                if self.jobs_closed {
                    self.status
                        .store(EngineStatus::Stopped as u8, Ordering::Release);
                    return;
                }
                self.status
                    .store(EngineStatus::Ready as u8, Ordering::Release);
                let Some(job) = self.jobs.blocking_recv() else {
                    self.jobs_closed = true;
                    continue;
                };
                self.status
                    .store(EngineStatus::Busy as u8, Ordering::Release);
                let admission = catch_unwind(AssertUnwindSafe(|| self.admit_received(job)));
                match admission {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        self.poison(format!("inference worker failed: {error:#}"));
                        return;
                    }
                    Err(_) => {
                        self.poison(
                            "inference worker panicked during request admission".to_owned(),
                        );
                        return;
                    }
                }
                if self.active.is_empty() {
                    continue;
                }
            }

            self.status
                .store(EngineStatus::Busy as u8, Ordering::Release);
            let drive = catch_unwind(AssertUnwindSafe(|| self.drive_once(Instant::now())));
            let made_progress = match drive {
                Ok(Ok(progress)) => progress,
                Ok(Err(error)) => {
                    self.poison(format!("inference worker failed: {error:#}"));
                    return;
                }
                Err(_) => {
                    self.poison("inference worker panicked and was poisoned".to_owned());
                    return;
                }
            };
            if !made_progress {
                thread::park_timeout(Duration::from_millis(1));
            }
        }
    }

    fn admit(&mut self, job: InferenceJob) -> Result<()> {
        self.admit_with_diagnostic_refresh(job, true)
    }

    #[allow(clippy::too_many_lines)]
    fn admit_with_diagnostic_refresh(
        &mut self,
        mut job: InferenceJob,
        refresh_diagnostics: bool,
    ) -> Result<()> {
        anyhow::ensure!(
            self.active.len() < self.config.max_active_sequences,
            "scheduler attempted to exceed the active sequence limit"
        );
        if job
            .batch_admission
            .as_ref()
            .is_some_and(|gate| !gate.wait_until_released())
        {
            job.cancelled.store(true, Ordering::Release);
            return Ok(());
        }
        if job.cancelled.load(Ordering::Acquire) || job.events.is_closed() {
            job.cancelled.store(true, Ordering::Release);
            return Ok(());
        }
        let stop_filter = match StopFilter::new(job.request.stop.clone()) {
            Ok(filter) => filter,
            Err(error) => {
                reject_preflight(
                    &mut job,
                    EngineFailureKind::InvalidRequest,
                    error.to_string(),
                );
                return Ok(());
            }
        };
        let constraint_spec = job.request.input.constraint().cloned();
        let constraint_label = constraint_spec.as_ref().map(EngineConstraintSpec::label);
        let Some((prompt_tokens, sampler)) =
            prepare_request(self.runtime, &mut job, self.config.chat_template_defaults)
        else {
            return Ok(());
        };
        let logits_constraint = match initialize_logits_constraint(self.runtime, constraint_spec) {
            Ok(constraint) => constraint,
            Err(error) => {
                reject_preflight(
                    &mut job,
                    EngineFailureKind::InvalidRequest,
                    format!(
                        "failed to initialize {} constraint: {error:#}",
                        constraint_label.unwrap_or("JSON schema")
                    ),
                );
                return Ok(());
            }
        };
        if let Some(message) = kv_capacity_rejection(
            self.runtime,
            prompt_tokens.len(),
            job.request.max_new_tokens,
        ) {
            reject_preflight(&mut job, EngineFailureKind::ResourceExhausted, message);
            return Ok(());
        }
        if job.cancelled.load(Ordering::Acquire) || job.events.is_closed() {
            job.cancelled.store(true, Ordering::Release);
            return Ok(());
        }
        let session = self
            .sessions
            .pop()
            .context("active slot has no reusable inference session")?;
        let prompt_token_count = prompt_tokens.len();
        let mut sequence = GenerationSequence::with_session(
            self.runtime.model.as_ref(),
            session,
            prompt_tokens,
            sampler,
            GenerationConfig {
                max_new_tokens: job.request.max_new_tokens,
                ignore_stop_tokens: job.request.ignore_stop_tokens,
            },
        )?;
        sequence.set_publish_prompt_prefix(job.request_policy.publish_prompt_prefix)?;
        if let Some(constraint) = logits_constraint
            && let Err(error) = sequence.set_logits_constraint(constraint)
        {
            let session = sequence.reclaim_session(self.runtime.model.as_ref())?;
            self.sessions.push(session);
            if refresh_diagnostics {
                self.refresh_prefix_cache_snapshot("rejected request session reclaimed");
            }
            reject_preflight(
                &mut job,
                EngineFailureKind::InvalidRequest,
                format!(
                    "invalid {} request: {error:#}",
                    constraint_label.unwrap_or("JSON schema")
                ),
            );
            return Ok(());
        }
        if job.cancelled.load(Ordering::Acquire) || job.events.is_closed() {
            job.cancelled.store(true, Ordering::Release);
            let session = sequence.reclaim_session(self.runtime.model.as_ref())?;
            self.sessions.push(session);
            if refresh_diagnostics {
                self.refresh_prefix_cache_snapshot("cancelled request session reclaimed");
            }
            return Ok(());
        }
        if let Some(preflight) = job.preflight.take()
            && preflight.send(Ok(())).is_err()
        {
            job.cancelled.store(true, Ordering::Release);
            let session = sequence.reclaim_session(self.runtime.model.as_ref())?;
            self.sessions.push(session);
            if refresh_diagnostics {
                self.refresh_prefix_cache_snapshot("abandoned request session reclaimed");
            }
            return Ok(());
        }
        let prefix_lookup_attach_time = sequence.prefix_lookup_attach_time();
        let batch_trace = job.batch_trace.as_ref().map(|member| member.handle.clone());
        self.active.push_back(ActiveSequence {
            job,
            sequence: Some(sequence),
            reclaimed_session: None,
            prompt_tokens: prompt_token_count,
            stream: TokenStream::new(&self.runtime.tokenizer),
            stop_filter,
            phase: ActivePhase::Generating,
            pending: VecDeque::with_capacity(3),
            blocked_since: None,
        });
        if let Some(trace) = batch_trace {
            trace.record_admitted(prefix_lookup_attach_time);
        }
        if refresh_diagnostics {
            self.refresh_prefix_cache_snapshot("prefix attached during request admission");
        }
        Ok(())
    }

    fn try_admit_available(&mut self) -> Result<bool> {
        let mut admitted = self.admit_pending_atomic_batch_if_available()?;
        if self.pending_atomic_batch.is_some() {
            return Ok(admitted);
        }
        while self.active.len() < self.config.max_active_sequences && !self.jobs_closed {
            match self.jobs.try_recv() {
                Ok(job) => {
                    self.admit_received(job)?;
                    admitted = true;
                    if self.pending_atomic_batch.is_some() {
                        break;
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.jobs_closed = true;
                }
            }
        }
        Ok(admitted)
    }

    /// Resolve a batch release before any member reaches model preflight.
    /// Once ready, every peer is already in the channel and contiguous because
    /// all handle submission paths share the same submit lock.
    fn admit_received(&mut self, job: InferenceJob) -> Result<()> {
        let Some(gate) = job.batch_admission.clone() else {
            return self.admit(job);
        };
        if !gate.wait_until_released() {
            job.cancelled.store(true, Ordering::Release);
            return Ok(());
        }

        let mut batch = Vec::with_capacity(gate.expected_jobs);
        batch.push(job);
        while batch.len() < gate.expected_jobs {
            let peer = match self.jobs.try_recv() {
                Ok(peer) => peer,
                Err(mpsc::error::TryRecvError::Empty) => {
                    anyhow::bail!(
                        "released atomic batch exposed only {} of {} queued jobs",
                        batch.len(),
                        gate.expected_jobs
                    );
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.jobs_closed = true;
                    anyhow::bail!(
                        "atomic batch channel closed after {} of {} queued jobs",
                        batch.len(),
                        gate.expected_jobs
                    );
                }
            };
            anyhow::ensure!(
                peer.batch_admission
                    .as_ref()
                    .is_some_and(|peer_gate| Arc::ptr_eq(peer_gate, &gate)),
                "atomic batch jobs were not contiguous in the inference queue"
            );
            batch.push(peer);
        }
        anyhow::ensure!(
            self.pending_atomic_batch.is_none(),
            "scheduler collected a second atomic batch while one was pending"
        );
        self.pending_atomic_batch = Some(batch);
        let _ = self.admit_pending_atomic_batch_if_available()?;
        Ok(())
    }

    /// Admit the whole released group only when every member can occupy an
    /// active slot. Existing work may continue while the group waits, but no
    /// member of the group can reach a forward early.
    fn admit_pending_atomic_batch_if_available(&mut self) -> Result<bool> {
        let Some(batch_len) = self.pending_atomic_batch.as_ref().map(Vec::len) else {
            return Ok(false);
        };
        let available = self
            .config
            .max_active_sequences
            .checked_sub(self.active.len())
            .context("active sequence count exceeds the scheduler limit")?;
        if batch_len > available {
            return Ok(false);
        }
        let batch = self
            .pending_atomic_batch
            .take()
            .context("pending atomic batch disappeared before admission")?;
        let benchmark_trace = batch
            .iter()
            .find_map(|job| job.batch_trace.as_ref().map(|member| member.handle.clone()));
        if let Some(trace) = &benchmark_trace {
            anyhow::ensure!(
                batch.iter().all(|job| {
                    job.batch_trace
                        .as_ref()
                        .is_some_and(|member| Arc::ptr_eq(&member.handle.shared, &trace.shared))
                }),
                "atomic batch contains inconsistent benchmark trace handles"
            );
            trace.begin_admission(Instant::now());
        }
        for job in batch {
            self.admit_with_diagnostic_refresh(job, false)?;
        }
        let diagnostics_start = Instant::now();
        let admission_diagnostics = sample_prefix_cache_snapshot(
            self.runtime.model.as_ref(),
            &self.prefix_cache_snapshot,
            self.active.len(),
            "atomic batch prefix attachment complete",
        );
        let diagnostics_time = diagnostics_start.elapsed();
        if let Some(trace) = benchmark_trace {
            trace.finish_admission(
                Instant::now(),
                diagnostics_time,
                admission_diagnostics,
                self.active.len(),
            );
        }
        Ok(true)
    }

    fn drive_once(&mut self, now: Instant) -> Result<bool> {
        let admitted = self.try_admit_available()?;
        if self.active.is_empty() {
            return Ok(admitted);
        }
        if self.round_visits_remaining == 0 || self.round_budget_remaining == 0 {
            self.round_visits_remaining = self.active.len();
            self.round_budget_remaining = self.config.scheduler_token_budget;
        }

        let result = self.drive_batch(now)?;
        anyhow::ensure!(
            result.visits <= self.round_visits_remaining,
            "scheduler visited more sequences than remained in the round"
        );
        self.round_visits_remaining -= result.visits;
        anyhow::ensure!(
            result.input_tokens <= self.round_budget_remaining,
            "generation batch exceeded the remaining scheduler token budget"
        );
        self.round_budget_remaining -= result.input_tokens;
        if self.active.is_empty() {
            self.round_visits_remaining = 0;
            self.round_budget_remaining = 0;
        }
        Ok(admitted || result.made_progress)
    }

    fn drive_batch(&mut self, now: Instant) -> Result<DriveResult> {
        let mut collected = self.collect_visits(now)?;
        let visits = collected.actions.len();
        self.reclaim_collected_removals(&collected.actions)?;
        if collected
            .actions
            .iter()
            .any(|action| matches!(action, VisitAction::Remove))
        {
            self.refresh_prefix_cache_snapshot("completed request session reclaimed");
        }
        let reserved = self.reserve_and_prepare_visits(&mut collected.actions)?;
        collected.input_tokens = reserved.input_tokens;
        collected.made_progress |= reserved.made_progress;
        collected.made_progress |=
            self.execute_prepared_visits(&mut collected.actions, reserved.reservation)?;
        collected.made_progress |= self.finalize_visits(collected.actions, now)?;
        Ok(DriveResult {
            made_progress: collected.made_progress,
            input_tokens: collected.input_tokens,
            visits,
        })
    }

    /// Reclaim lanes that were already known to be abandoned or removable
    /// when visits were collected. Their prior model work is quiescent at this
    /// iteration boundary, so releasing them before peer submission makes the
    /// reclaimed capacity available to the next batch without flushing any
    /// newly produced peer events.
    fn reclaim_collected_removals(&mut self, actions: &[VisitAction]) -> Result<()> {
        let model = self.runtime.model.as_ref();
        for (slot, action) in self.active.iter_mut().take(actions.len()).zip(actions) {
            if matches!(action, VisitAction::Remove) {
                Self::reclaim_slot_sequence_after_completion_barrier(model, slot)?;
            }
        }
        Ok(())
    }

    fn collect_visits(&mut self, now: Instant) -> Result<CollectedVisits> {
        let model = self.runtime.model.as_ref();
        let visit_limit = self.round_visits_remaining.min(self.active.len());
        let mut actions = Vec::with_capacity(visit_limit);
        let mut token_budget = self.round_budget_remaining;
        let mut input_tokens = 0_usize;
        let mut made_progress = false;

        for index in 0..visit_limit {
            if token_budget == 0 {
                break;
            }
            let slot = self
                .active
                .get_mut(index)
                .expect("visited scheduler index must exist");
            let initial_flush = slot.flush_pending(now);
            made_progress |= initial_flush.sent > 0;
            match initial_flush.state {
                FlushState::Abandoned => {
                    actions.push(VisitAction::Remove);
                    made_progress = true;
                    continue;
                }
                FlushState::Blocked => {
                    actions.push(VisitAction::Keep);
                    continue;
                }
                FlushState::Drained => {}
            }
            if slot.phase == ActivePhase::Delivering {
                actions.push(VisitAction::Remove);
                made_progress = true;
                continue;
            }

            let step_budget = self.config.prefill_token_quantum.min(token_budget);
            anyhow::ensure!(step_budget > 0, "scheduler produced an empty model quantum");
            let cancelled = Arc::clone(&slot.job.cancelled);
            let events = slot.job.events.clone();
            let sequence = slot
                .sequence
                .as_mut()
                .context("generating slot has no live sequence")?;
            if cancelled.load(Ordering::Acquire) || events.is_closed() {
                let preparation = sequence.prepare_step_with_budget(model, step_budget, || true)?;
                let GenerationPreparation::Complete(step) = preparation else {
                    anyhow::bail!("cancelled generation unexpectedly prepared a forward");
                };
                actions.push(VisitAction::Complete {
                    step,
                    stream_error: None,
                });
                made_progress = true;
                continue;
            }
            match sequence.preview_step_with_budget(model, step_budget)? {
                GenerationPreview::Complete => {
                    let preparation =
                        sequence.prepare_step_with_budget(model, step_budget, || false)?;
                    let GenerationPreparation::Complete(step) = preparation else {
                        anyhow::bail!("completed generation preview prepared a forward");
                    };
                    actions.push(VisitAction::Complete {
                        step,
                        stream_error: None,
                    });
                    made_progress = true;
                }
                GenerationPreview::Forward(preview) => {
                    anyhow::ensure!(
                        preview.input_tokens() <= step_budget,
                        "generation lane exceeded its previewed input-token quantum"
                    );
                    token_budget -= preview.input_tokens();
                    input_tokens = input_tokens
                        .checked_add(preview.input_tokens())
                        .context("generation batch token count overflow")?;
                    actions.push(VisitAction::Preview {
                        preview,
                        input_token_budget: step_budget,
                    });
                }
            }
        }
        Ok(CollectedVisits {
            actions,
            input_tokens,
            made_progress,
        })
    }

    fn reserve_and_prepare_visits(
        &mut self,
        actions: &mut [VisitAction],
    ) -> Result<ReservedVisits> {
        let candidate_indices = actions
            .iter()
            .enumerate()
            .filter_map(|(index, action)| {
                matches!(action, VisitAction::Preview { .. }).then_some(index)
            })
            .collect::<Vec<_>>();
        if candidate_indices.is_empty() {
            return Ok(ReservedVisits {
                reservation: None,
                input_tokens: 0,
                made_progress: false,
            });
        }

        let mut selected = candidate_indices.len();
        let reservation = loop {
            let reservation_result = {
                let mut inputs = Vec::with_capacity(selected);
                for &index in &candidate_indices[..selected] {
                    let VisitAction::Preview { preview, .. } = &actions[index] else {
                        unreachable!("candidate index must still contain a preview");
                    };
                    let sequence = self
                        .active
                        .get(index)
                        .and_then(|slot| slot.sequence.as_ref())
                        .context("previewed slot has no live sequence")?;
                    inputs.push(sequence.kv_reservation_input(preview)?);
                }
                // A panic or backend allocation failure may leave submitted
                // storage initialization work. Ordinary Exhausted and success
                // reset this flag before returning to the scheduler.
                self.forward_needs_drain = true;
                let result = self.runtime.model.reserve_batch_kv(&inputs);
                drop(inputs);
                result
            };
            match reservation_result {
                // A successful reservation may lazily create backend slab
                // tensors. Keep the cleanup obligation armed through
                // generation preparation and model submission; the final
                // logits completion barrier clears it.
                Ok(reservation) => break reservation,
                Err(KvCacheReservationError::Exhausted(exhausted)) if selected > 1 => {
                    self.forward_needs_drain = false;
                    selected -= 1;
                    let _ = exhausted;
                }
                Err(KvCacheReservationError::Exhausted(exhausted)) => {
                    self.forward_needs_drain = false;
                    return self.fail_kv_exhausted_visit(actions, &candidate_indices, exhausted);
                }
                Err(KvCacheReservationError::Fatal(error)) => return Err(error.into()),
            }
        };

        for &index in &candidate_indices[selected..] {
            actions[index] = VisitAction::Keep;
        }
        let input_tokens = self.prepare_reserved_visits(actions, &candidate_indices[..selected])?;
        Ok(ReservedVisits {
            reservation,
            input_tokens,
            made_progress: false,
        })
    }

    fn prepare_reserved_visits(
        &mut self,
        actions: &mut [VisitAction],
        selected_indices: &[usize],
    ) -> Result<usize> {
        let mut input_tokens = 0_usize;
        for &index in selected_indices {
            let (preview, input_token_budget) = match &actions[index] {
                VisitAction::Preview {
                    preview,
                    input_token_budget,
                } => (preview.clone(), *input_token_budget),
                _ => unreachable!("selected candidate must still contain a preview"),
            };
            input_tokens = input_tokens
                .checked_add(preview.input_tokens())
                .context("reserved generation batch token count overflow")?;
            let sequence = self
                .active
                .get_mut(index)
                .and_then(|slot| slot.sequence.as_mut())
                .context("reserved slot has no live sequence")?;
            let preparation = sequence.prepare_step_from_preview(
                self.runtime.model.as_ref(),
                &preview,
                input_token_budget,
                || false,
            )?;
            let GenerationPreparation::Forward(prepared) = preparation else {
                anyhow::bail!("reserved generation preview completed before submission");
            };
            actions[index] = VisitAction::Forward(Some(prepared));
        }
        Ok(input_tokens)
    }

    fn fail_kv_exhausted_visit(
        &mut self,
        actions: &mut [VisitAction],
        candidate_indices: &[usize],
        exhausted: KvCacheExhausted,
    ) -> Result<ReservedVisits> {
        let failed_index = candidate_indices[0];
        for &index in candidate_indices {
            actions[index] = VisitAction::Keep;
        }
        let slot = self
            .active
            .get_mut(failed_index)
            .context("KV-exhausted slot disappeared")?;
        slot.pending.push_back(EngineEvent::Failed {
            message: exhausted.to_string(),
            kind: EngineFailureKind::ResourceExhausted,
        });
        slot.phase = ActivePhase::Delivering;
        Self::reclaim_slot_sequence_after_completion_barrier(self.runtime.model.as_ref(), slot)?;
        self.refresh_prefix_cache_snapshot("KV-exhausted request session reclaimed");
        Ok(ReservedVisits {
            reservation: None,
            input_tokens: 0,
            made_progress: true,
        })
    }

    fn execute_prepared_visits(
        &mut self,
        actions: &mut [VisitAction],
        reservation: Option<Box<dyn CausalLmKvReservation>>,
    ) -> Result<bool> {
        let visits = actions.len();
        let lane_count = actions
            .iter()
            .filter(|action| matches!(action, VisitAction::Forward(Some(_))))
            .count();
        if lane_count == 0 {
            drop(reservation);
            return Ok(false);
        }
        let submitted = self.submit_mixed_forward(actions, lane_count, reservation)?;
        anyhow::ensure!(
            submitted.logits.dims2()? == (lane_count, self.runtime.model.runtime_spec().vocab_size),
            "mixed generation logits have shape {:?}, expected [{lane_count}, {}]",
            submitted.logits.dims(),
            self.runtime.model.runtime_spec().vocab_size
        );

        // Validate every lane before sampling or invoking any request
        // callback. One bad cache/postcondition poisons the whole batch
        // without exposing a partial row commit.
        for (slot, action) in self.active.iter_mut().take(visits).zip(actions.iter()) {
            if let VisitAction::Forward(Some(prepared)) = action {
                slot.sequence
                    .as_mut()
                    .context("prepared slot has no live sequence")?
                    .validate_prepared_forward(prepared)?;
            }
        }
        let output = self.complete_mixed_forward(submitted, lane_count)?;
        self.complete_mixed_rows(actions, output)?;
        Ok(true)
    }

    fn submit_mixed_forward(
        &mut self,
        actions: &[VisitAction],
        lane_count: usize,
        reservation: Option<Box<dyn CausalLmKvReservation>>,
    ) -> Result<SubmittedBatchOutput> {
        let benchmark_traces = self.collect_pending_benchmark_forwards(actions);
        let model = self.runtime.model.as_ref();
        let mut batch_inputs = Vec::with_capacity(lane_count);
        for (slot, action) in self.active.iter_mut().take(actions.len()).zip(actions) {
            if let VisitAction::Forward(Some(prepared)) = action {
                let sequence = slot
                    .sequence
                    .as_mut()
                    .context("prepared slot has no live sequence")?;
                batch_inputs.push(sequence.prepared_batch_input(prepared)?);
            }
        }
        anyhow::ensure!(
            batch_inputs.len() == lane_count,
            "scheduler lost a prepared generation lane"
        );
        let model_start = Instant::now();
        self.forward_needs_drain = true;
        let logits = model.forward_batch_tokens_reserved(&mut batch_inputs, reservation)?;
        drop(batch_inputs);
        Ok(SubmittedBatchOutput {
            logits,
            model_start,
            benchmark_traces,
        })
    }

    fn collect_pending_benchmark_forwards(
        &self,
        actions: &[VisitAction],
    ) -> Vec<PendingBenchmarkForward> {
        struct ObservedLane {
            owner: Option<(EngineBatchTraceHandle, usize)>,
            phase: BenchmarkForwardKind,
            start_pos: usize,
            input_tokens: usize,
        }

        if !self
            .active
            .iter()
            .take(actions.len())
            .any(|slot| slot.job.batch_trace.is_some())
        {
            return Vec::new();
        }

        let mut observed = Vec::new();
        let mut handles: Vec<EngineBatchTraceHandle> = Vec::new();
        for (slot, action) in self.active.iter().take(actions.len()).zip(actions) {
            let VisitAction::Forward(Some(prepared)) = action else {
                continue;
            };
            let owner = slot.job.batch_trace.as_ref().map(|member| {
                let handle = member.handle.clone();
                if !handles
                    .iter()
                    .any(|current| Arc::ptr_eq(&current.shared, &handle.shared))
                {
                    handles.push(handle.clone());
                }
                (handle, member.member_index)
            });
            observed.push(ObservedLane {
                owner,
                phase: benchmark_forward_kind(prepared.phase_kind()),
                start_pos: prepared.start_pos(),
                input_tokens: prepared.input_tokens(),
            });
        }
        let input_tokens = observed.iter().map(|lane| lane.input_tokens).sum();
        let kind = observed
            .first()
            .map_or(BenchmarkForwardKind::Mixed, |first| {
                if observed.iter().all(|lane| lane.phase == first.phase) {
                    first.phase
                } else {
                    BenchmarkForwardKind::Mixed
                }
            });
        handles
            .into_iter()
            .map(|handle| PendingBenchmarkForward {
                lanes: observed
                    .iter()
                    .map(|lane| BenchmarkForwardLane {
                        member_index: lane.owner.as_ref().and_then(|(owner, member_index)| {
                            Arc::ptr_eq(&owner.shared, &handle.shared).then_some(*member_index)
                        }),
                        phase: lane.phase,
                        start_pos: lane.start_pos,
                        input_tokens: lane.input_tokens,
                    })
                    .collect(),
                handle,
                kind,
                input_tokens,
            })
            .collect()
    }

    fn complete_mixed_forward(
        &mut self,
        submitted: SubmittedBatchOutput,
        lane_count: usize,
    ) -> Result<CompletedBatchOutput> {
        let SubmittedBatchOutput {
            logits,
            model_start,
            benchmark_traces,
        } = submitted;
        let logits = logits.to_dtype(candle_core::DType::F32)?;
        // The whole-batch host readback is the successful completion barrier.
        // Once entered, a failure must poison without retrying a second drain.
        self.forward_needs_drain = false;
        let logits = self.runtime.model.read_output_to_cpu(&logits)?;
        let model_time = model_start.elapsed();
        for pending in benchmark_traces {
            pending
                .handle
                .record_packed_forward(BenchmarkPackedForwardRecord {
                    kind: pending.kind,
                    lane_count,
                    input_tokens: pending.input_tokens,
                    model_time,
                    lanes: pending.lanes,
                });
        }
        let rows = (0..lane_count)
            .map(|row| logits.narrow(0, row, 1))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(CompletedBatchOutput { rows, model_time })
    }

    fn complete_mixed_rows(
        &mut self,
        actions: &mut [VisitAction],
        output: CompletedBatchOutput,
    ) -> Result<()> {
        let model = self.runtime.model.as_ref();
        let mut rows = output.rows.into_iter();
        let mut published_prompt_prefix = false;
        let mut replaced_failed_session = false;
        for (slot, action) in self.active.iter_mut().take(actions.len()).zip(actions) {
            let VisitAction::Forward(prepared) = action else {
                continue;
            };
            let prepared = prepared
                .take()
                .context("prepared generation lane was already consumed")?;
            published_prompt_prefix |= slot.job.request_policy.publish_prompt_prefix
                && prepared
                    .start_pos()
                    .checked_add(prepared.input_tokens())
                    .is_some_and(|next_cache_len| next_cache_len <= slot.prompt_tokens);
            let row = rows
                .next()
                .context("mixed generation output row is missing")?;
            let ActiveSequence {
                job,
                sequence,
                prompt_tokens,
                stream,
                stop_filter,
                pending,
                ..
            } = slot;
            let cancelled = Arc::clone(&job.cancelled);
            let events = job.events.clone();
            let mut stream_error = None;
            let completion = sequence
                .as_mut()
                .context("prepared slot has no live sequence")?
                .complete_prepared_controlled(
                    model,
                    prepared,
                    row,
                    output.model_time,
                    move || cancelled.load(Ordering::Acquire) || events.is_closed(),
                    |token, n_decoded| {
                        match stream.push(token) {
                            Ok(Some(fragment)) => {
                                let output = stop_filter.push(&fragment);
                                pending.push_back(EngineEvent::Text {
                                    content: output.text,
                                    token: Some(EngineTokenEvidence {
                                        token_id: token,
                                        n_decoded,
                                        prompt_tokens: *prompt_tokens,
                                    }),
                                });
                                if output.matched {
                                    return Ok(TokenControl::Stop);
                                }
                            }
                            Ok(None) => {}
                            Err(error) => {
                                stream_error = Some(format!(
                                    "tokenizer failed while streaming generated text: {error:#}"
                                ));
                            }
                        }
                        Ok(TokenControl::Continue)
                    },
                );
            match completion {
                Ok(step) => *action = VisitAction::Complete { step, stream_error },
                Err(error) if job.request.input.constraint().is_some() => {
                    let label = job
                        .request
                        .input
                        .constraint()
                        .map_or("JSON schema", EngineConstraintSpec::label);
                    pending.push_back(EngineEvent::Failed {
                        message: format!("{label} generation failed: {error:#}"),
                        kind: EngineFailureKind::InvalidRequest,
                    });
                    slot.phase = ActivePhase::Delivering;
                    Self::replace_failed_slot_session_after_completion_barrier(model, slot)?;
                    replaced_failed_session = true;
                    *action = VisitAction::Keep;
                }
                Err(error) => return Err(error),
            }
        }
        anyhow::ensure!(
            rows.next().is_none(),
            "mixed generation produced an extra output row"
        );
        if published_prompt_prefix {
            self.refresh_prefix_cache_snapshot("prompt prefix published");
        } else if replaced_failed_session {
            self.refresh_prefix_cache_snapshot("failed constrained request session replaced");
        }
        Ok(())
    }

    fn finalize_visits(&mut self, mut actions: Vec<VisitAction>, now: Instant) -> Result<bool> {
        let model = self.runtime.model.as_ref();
        let visits = actions.len();
        let mut made_progress = false;
        let mut reclaimed_session = false;
        for (slot, action) in self.active.iter_mut().take(visits).zip(&mut actions) {
            let was_reclaimed = slot.reclaimed_session.is_some();
            let completed = std::mem::replace(action, VisitAction::Keep);
            match completed {
                VisitAction::Complete { step, stream_error } => {
                    if Self::apply_slot_step(model, slot, &step, stream_error)? {
                        *action = VisitAction::Remove;
                    }
                }
                other => *action = other,
            }
            reclaimed_session |= !was_reclaimed && slot.reclaimed_session.is_some();
        }

        // Collected removals were reclaimed before model submission. Completion
        // and post-forward cancellation can add more removals here; reclaiming
        // is idempotent while all slots remain scheduler-owned and before any
        // peer exposes this batch's newly pending events.
        for (slot, action) in self.active.iter_mut().take(visits).zip(&actions) {
            if matches!(action, VisitAction::Remove) {
                let was_reclaimed = slot.reclaimed_session.is_some();
                Self::reclaim_slot_sequence_after_completion_barrier(model, slot)?;
                reclaimed_session |= !was_reclaimed && slot.reclaimed_session.is_some();
            }
        }
        if reclaimed_session {
            self.refresh_prefix_cache_snapshot("completed request session reclaimed");
        }

        let observed_after_batch = Instant::now();
        let post_batch_now = observed_after_batch.max(now);
        for (slot, action) in self.active.iter_mut().take(visits).zip(&mut actions) {
            if matches!(action, VisitAction::Remove) {
                continue;
            }
            let final_flush = slot.flush_pending(post_batch_now);
            made_progress |= final_flush.sent > 0;
            if final_flush.state == FlushState::Drained && slot.phase == ActivePhase::Delivering {
                *action = VisitAction::Remove;
            } else if final_flush.state == FlushState::Abandoned {
                made_progress = true;
                // A running channel can close during this flush. Defer its
                // fallible session cleanup to the next scheduler visit so no
                // peer event is followed by a same-batch cleanup failure.
                if slot.reclaimed_session.is_some() {
                    *action = VisitAction::Remove;
                } else {
                    *action = VisitAction::Keep;
                }
            }
        }

        // Every removal is already reclaimed. No fallible model/session call
        // is allowed after final event flushing begins.
        for (slot, action) in self.active.iter_mut().take(visits).zip(&actions) {
            if matches!(action, VisitAction::Remove) {
                anyhow::ensure!(
                    slot.reclaimed_session.is_some(),
                    "removed slot has no reclaimed session"
                );
            }
        }

        for action in actions {
            let mut slot = self
                .active
                .pop_front()
                .context("scheduler lost a visited active slot")?;
            if matches!(action, VisitAction::Remove) {
                let session = slot
                    .reclaimed_session
                    .take()
                    .expect("removed slot was validated before leaving the active queue");
                self.sessions.push(session);
            } else {
                self.active.push_back(slot);
            }
        }
        debug_assert_eq!(
            self.sessions.len() + self.active.len(),
            self.config.max_active_sequences
        );
        self.publish_active_sequence_count();
        Ok(made_progress)
    }

    /// Apply request-local decoding and terminal state. `true` marks the slot
    /// for removal after every lane in the model batch has committed.
    fn apply_slot_step(
        model: &dyn CausalLm,
        slot: &mut ActiveSequence<'_>,
        step: &GenerationStep,
        stream_error: Option<String>,
    ) -> Result<bool> {
        if let Some(error) = stream_error {
            slot.pending.push_back(EngineEvent::Failed {
                message: error,
                kind: EngineFailureKind::Internal,
            });
            Self::reclaim_slot_sequence_after_completion_barrier(model, slot)?;
            slot.phase = ActivePhase::Delivering;
            return Ok(false);
        }
        let Some(metrics) = step.metrics().cloned() else {
            return Ok(false);
        };
        if metrics.finish_reason == FinishReason::Cancelled
            || slot.job.cancelled.load(Ordering::Acquire)
            || slot.job.events.is_closed()
        {
            return Ok(true);
        }
        match slot.stream.finish() {
            Ok(Some(fragment)) => {
                let output = slot.stop_filter.push(&fragment);
                slot.pending.push_back(EngineEvent::Text {
                    content: output.text,
                    token: None,
                });
            }
            Ok(None) => {}
            Err(error) => slot.pending.push_back(EngineEvent::Failed {
                message: format!("tokenizer failed while finalizing generated text: {error:#}"),
                kind: EngineFailureKind::Internal,
            }),
        }
        if !matches!(slot.pending.back(), Some(EngineEvent::Failed { .. })) {
            let output = slot.stop_filter.finish();
            if !output.text.is_empty() {
                slot.pending.push_back(EngineEvent::Text {
                    content: output.text,
                    token: None,
                });
            }
        }
        if !matches!(slot.pending.back(), Some(EngineEvent::Failed { .. })) {
            slot.pending.push_back(EngineEvent::Finished(metrics));
        }
        Self::reclaim_slot_sequence_after_completion_barrier(model, slot)?;
        slot.phase = ActivePhase::Delivering;
        Ok(false)
    }

    /// Clear a terminal/request-local sequence after the scheduler's shared
    /// final-logits completion barrier. The allocation remains owned by the
    /// slot until its bounded outbox has drained or been abandoned.
    fn reclaim_slot_sequence_after_completion_barrier(
        model: &dyn CausalLm,
        slot: &mut ActiveSequence<'_>,
    ) -> Result<()> {
        if slot.reclaimed_session.is_some() {
            return Ok(());
        }
        let sequence = slot
            .sequence
            .take()
            .context("active slot has no generation sequence to reclaim")?;
        let session = sequence.reclaim_session_after_completion_barrier(model)?;
        slot.reclaimed_session = Some(session);
        Ok(())
    }

    /// A failed constrained sequence cannot return its allocation to the
    /// reusable pool. The whole-batch logits readback already established the
    /// completion barrier, so discard that allocation without another device
    /// drain and replace it before the request leaves the active queue.
    fn replace_failed_slot_session_after_completion_barrier(
        model: &dyn CausalLm,
        slot: &mut ActiveSequence<'_>,
    ) -> Result<()> {
        anyhow::ensure!(
            slot.reclaimed_session.is_none(),
            "failed constrained slot already owns a reclaimed session"
        );
        let sequence = slot
            .sequence
            .take()
            .context("failed constrained slot has no generation sequence")?;
        sequence.discard_session_without_synchronize();
        let replacement = model
            .new_session()
            .context("failed to replace a request-local constrained generation session")?;
        model
            .validate_session(replacement.as_ref())
            .context("replacement constrained generation session is invalid")?;
        slot.reclaimed_session = Some(replacement);
        Ok(())
    }

    fn poison(&mut self, mut message: String) {
        self.status
            .store(EngineStatus::Poisoned as u8, Ordering::Release);
        self.jobs.close();
        if std::mem::take(&mut self.forward_needs_drain) {
            match catch_unwind(AssertUnwindSafe(|| self.runtime.model.synchronize())) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    message = format!(
                        "{message}; forward-error cleanup synchronization also failed: {error}"
                    );
                }
                Err(_) => {
                    message.push_str("; forward-error cleanup synchronization panicked");
                }
            }
        }
        for slot in &mut self.active {
            slot.pending.clear();
            let failure = EnginePreflightError {
                message: message.clone(),
                kind: EngineFailureKind::Fatal,
            };
            if matches!(
                slot.job.events.try_send(EngineEvent::Failed {
                    message: failure.message.clone(),
                    kind: failure.kind,
                }),
                Err(mpsc::error::TrySendError::Full(_))
            ) {
                *slot
                    .job
                    .terminal_failure
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(failure);
            }
            if let Some(sequence) = slot.sequence.take() {
                sequence.discard_session_without_synchronize();
            }
            drop(slot.reclaimed_session.take());
        }
        self.active.clear();
        self.sessions.clear();
        if let Some(batch) = self.pending_atomic_batch.take() {
            for mut job in batch {
                reject_preflight(&mut job, EngineFailureKind::Fatal, message.clone());
            }
        }
        while let Ok(mut job) = self.jobs.try_recv() {
            reject_preflight(&mut job, EngineFailureKind::Fatal, message.clone());
        }
    }
}

fn initialize_logits_constraint(
    runtime: &LoadedRuntime,
    spec: Option<EngineConstraintSpec>,
) -> Result<Option<Box<dyn LogitsConstraint>>> {
    let Some(spec) = spec else {
        return Ok(None);
    };
    let tokenizer = Arc::clone(&runtime.tokenizer);
    let stop_token_ids = runtime.model.runtime_spec().stop_token_ids.clone();
    Ok(Some(match spec.grammar {
        EngineConstraintGrammar::JsonSchema(schema) => Box::new(JsonSchemaTokenConstraint::new(
            tokenizer,
            stop_token_ids,
            schema,
        )?),
        EngineConstraintGrammar::LazyToolCall(calls) => Box::new(LazyToolCallConstraint::new(
            tokenizer,
            stop_token_ids,
            calls,
        )?),
    }))
}

fn kv_capacity_rejection(
    runtime: &LoadedRuntime,
    prompt_tokens: usize,
    max_new_tokens: usize,
) -> Option<String> {
    let capacity = runtime.model.kv_cache_capacity()?;
    let maximum_cached_tokens = if max_new_tokens == 0 {
        Some(0)
    } else {
        prompt_tokens.checked_add(max_new_tokens - 1)
    };
    let total_tokens = capacity.total_blocks.checked_mul(capacity.block_tokens);
    let permanently_too_large = match (maximum_cached_tokens, total_tokens) {
        (Some(required), Some(total)) => required > total,
        _ => true,
    };
    permanently_too_large.then(|| {
        format!(
            "request requires up to {} cached tokens, exceeding the configured shared KV capacity of {} blocks × {} tokens",
            maximum_cached_tokens.map_or("overflow".to_owned(), |value| value.to_string()),
            capacity.total_blocks,
            capacity.block_tokens
        )
    })
}

#[allow(clippy::too_many_lines)]
fn prepare_request(
    runtime: &LoadedRuntime,
    job: &mut InferenceJob,
    chat_template_defaults: ChatTemplateOptions,
) -> Option<(Vec<u32>, Sampler)> {
    let chat_template = job.request.chat_template.or(chat_template_defaults);
    if matches!(
        &job.request.input,
        EngineInput::RawPrompt(_) | EngineInput::TokenIds(_)
    ) && !job.request.chat_template.is_default()
    {
        reject_preflight(
            job,
            EngineFailureKind::InvalidRequest,
            "chat template options require a templated chat or agent request".to_owned(),
        );
        return None;
    }
    let prompt_tokens = match &job.request.input {
        EngineInput::Chat(messages) | EngineInput::ConstrainedChat { messages, .. } => {
            let rendered = match runtime
                .adapter
                .format_chat_messages_with(messages, chat_template)
            {
                Ok(rendered) => rendered,
                Err(error) => {
                    reject_preflight(
                        job,
                        EngineFailureKind::InvalidRequest,
                        format!("failed to render chat prompt: {error:#}"),
                    );
                    return None;
                }
            };
            match runtime.encode(rendered, false) {
                Ok(tokens) => tokens,
                Err(error) => {
                    reject_preflight(
                        job,
                        EngineFailureKind::Internal,
                        format!("failed to tokenize chat prompt: {error:#}"),
                    );
                    return None;
                }
            }
        }
        EngineInput::Agent { prompt, .. } => {
            let rendered = match runtime
                .adapter
                .format_agent_prompt_with(prompt, chat_template)
            {
                Ok(rendered) => rendered,
                Err(error) => {
                    reject_preflight(
                        job,
                        EngineFailureKind::InvalidRequest,
                        format!("failed to render native agent prompt: {error:#}"),
                    );
                    return None;
                }
            };
            match runtime.encode(rendered, false) {
                Ok(tokens) => tokens,
                Err(error) => {
                    reject_preflight(
                        job,
                        EngineFailureKind::Internal,
                        format!("failed to tokenize native agent prompt: {error:#}"),
                    );
                    return None;
                }
            }
        }
        EngineInput::RawPrompt(prompt) => {
            if prompt.is_empty() {
                reject_preflight(
                    job,
                    EngineFailureKind::InvalidRequest,
                    "raw prompt must not be empty".to_owned(),
                );
                return None;
            }
            match runtime.encode(prompt.clone(), false) {
                Ok(tokens) => tokens,
                Err(error) => {
                    reject_preflight(
                        job,
                        EngineFailureKind::Internal,
                        format!("failed to tokenize raw prompt: {error:#}"),
                    );
                    return None;
                }
            }
        }
        EngineInput::TokenIds(tokens) => {
            if tokens.is_empty() {
                reject_preflight(
                    job,
                    EngineFailureKind::InvalidRequest,
                    "raw token prompt must not be empty".to_owned(),
                );
                return None;
            }
            if let Some((index, token)) = tokens
                .iter()
                .copied()
                .enumerate()
                .find(|(_, token)| (*token as usize) >= runtime.model.runtime_spec().vocab_size)
            {
                reject_preflight(
                    job,
                    EngineFailureKind::InvalidRequest,
                    format!(
                        "raw prompt token {token} at index {index} exceeds model vocabulary {}",
                        runtime.model.runtime_spec().vocab_size
                    ),
                );
                return None;
            }
            tokens.clone()
        }
    };
    let max_context = runtime.model.runtime_spec().max_context;
    let required_context = prompt_tokens
        .len()
        .checked_add(job.request.max_new_tokens.saturating_sub(1));
    if prompt_tokens.is_empty() || required_context.is_none_or(|required| required > max_context) {
        // Lead with OpenAI's wording so agent clients that recognize context
        // overflow by provider phrases can shorten their history and retry.
        reject_preflight(
            job,
            EngineFailureKind::InvalidRequest,
            format!(
                "This model's maximum context length is {max_context} tokens; the prompt ({} tokens) plus requested output ({} tokens) exceeds it",
                prompt_tokens.len(),
                job.request.max_new_tokens
            ),
        );
        return None;
    }
    let sampler = match job.request.sampler() {
        Ok(sampler) => sampler,
        Err(error) => {
            reject_preflight(
                job,
                EngineFailureKind::InvalidRequest,
                format!("invalid sampling parameters: {error:#}"),
            );
            return None;
        }
    };
    Some((prompt_tokens, sampler))
}

fn reject_preflight(job: &mut InferenceJob, kind: EngineFailureKind, message: String) {
    if let Some(member) = job.batch_trace.as_mut() {
        member.mark_terminal(true);
    }
    if let Some(preflight) = job.preflight.take() {
        let _ = preflight.send(Err(EnginePreflightError { message, kind }));
    } else {
        let _ = job.events.try_send(EngineEvent::Failed { message, kind });
    }
}

#[cfg(test)]
mod tests {
    use std::{
        any::Any,
        sync::{Mutex, atomic::AtomicUsize},
    };

    use candle_core::{Device, Tensor};
    use serde_json::json;
    use tokenizers::{
        Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };

    use crate::{
        architecture::{
            CausalLm, CausalLmBatchCapability, CausalLmBatchInput, CausalLmKvReservationInput,
            KvCacheCapacity, KvCacheExhausted, RuntimeSpec,
        },
        backend::DTypeKind,
        model_weights::WeightSource,
        models::ModelAdapter,
    };

    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ForwardTrace {
        session_id: usize,
        label: u32,
        input_ids: Vec<u32>,
        start_pos: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum InvocationTrace {
        Single(ForwardTrace),
        Packed(Vec<ForwardTrace>),
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum LifecycleTrace {
        Clear(usize),
        PackedForward(Vec<usize>),
    }

    #[derive(Debug, Default)]
    struct FakeState {
        next_session_id: AtomicUsize,
        trace: Mutex<Vec<InvocationTrace>>,
        lifecycle: Mutex<Vec<LifecycleTrace>>,
        cleared_sessions: Mutex<Vec<usize>>,
        panic_clear_session: Mutex<Option<usize>>,
        fail_at: Mutex<Option<(u32, usize)>>,
        panic_at: Mutex<Option<(u32, usize)>>,
        fail_after_submit_at: Mutex<Option<(u32, usize)>>,
        panic_after_submit_at: Mutex<Option<(u32, usize)>>,
        completion_barrier_calls: AtomicUsize,
        fail_completion_barrier_at: Mutex<Option<usize>>,
        cleanup_drain_calls: AtomicUsize,
        bad_batch_shape: AtomicBool,
        corrupt_cache_label: Mutex<Option<u32>>,
        cancel_on_completion_barrier: Mutex<Option<Arc<AtomicBool>>>,
        reservation_token_limit: Mutex<Option<usize>>,
        reservation_attempts: Mutex<Vec<Vec<usize>>>,
        prefix_cache_diagnostics:
            Mutex<Option<std::result::Result<PrefixCacheDiagnostics, String>>>,
        prefix_cache_diagnostics_calls: AtomicUsize,
        panic_prefix_cache_diagnostics: AtomicBool,
    }

    #[derive(Debug)]
    struct FakeKvReservation {
        lanes: usize,
    }

    impl CausalLmKvReservation for FakeKvReservation {
        fn into_any(self: Box<Self>) -> Box<dyn Any> {
            self
        }
    }

    struct FakeAdapter {
        spec: RuntimeSpec,
    }

    impl ModelAdapter for FakeAdapter {
        fn family(&self) -> &'static str {
            "scheduler-fake"
        }

        fn runtime_spec(&self) -> &RuntimeSpec {
            &self.spec
        }

        fn build_model(
            &self,
            _weights: &dyn WeightSource,
        ) -> candle_core::Result<Box<dyn CausalLm>> {
            candle_core::bail!("scheduler fake adapter does not load weights")
        }

        fn format_chat_messages(&self, messages: &[ChatMessage]) -> Result<String> {
            Ok(messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join(" "))
        }
    }

    #[derive(Debug)]
    struct FakeSession {
        owner: Arc<()>,
        state: Arc<FakeState>,
        id: usize,
        label: Option<u32>,
        cache_len: usize,
    }

    impl CausalLmSession for FakeSession {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn clear_cache(&mut self) {
            let mut panic_session = self
                .state
                .panic_clear_session
                .lock()
                .expect("clear panic lock is healthy");
            if *panic_session == Some(self.id) {
                *panic_session = None;
                drop(panic_session);
                panic!("injected scheduler cache cleanup panic");
            }
            drop(panic_session);
            self.state
                .cleared_sessions
                .lock()
                .expect("clear trace lock is healthy")
                .push(self.id);
            self.state
                .lifecycle
                .lock()
                .expect("lifecycle trace lock is healthy")
                .push(LifecycleTrace::Clear(self.id));
            self.label = None;
            self.cache_len = 0;
        }

        fn cache_len(&self) -> usize {
            self.cache_len
        }
    }

    struct FakeModel {
        owner: Arc<()>,
        state: Arc<FakeState>,
        spec: RuntimeSpec,
        device: Device,
    }

    impl FakeModel {
        fn output_token(label: u32) -> usize {
            if label == 2 { 4 } else { 3 }
        }

        fn check_injected_failure(&self, lane: &ForwardTrace) -> candle_core::Result<()> {
            if self
                .state
                .panic_at
                .lock()
                .expect("panic lock is healthy")
                .is_some_and(|failure| failure == (lane.label, lane.start_pos))
            {
                panic!("injected scheduler model panic");
            }
            if self
                .state
                .fail_at
                .lock()
                .expect("failure lock is healthy")
                .is_some_and(|failure| failure == (lane.label, lane.start_pos))
            {
                candle_core::bail!("injected scheduler model failure");
            }
            Ok(())
        }

        fn check_injected_post_submit(&self, lane: &ForwardTrace) -> candle_core::Result<()> {
            if self
                .state
                .panic_after_submit_at
                .lock()
                .expect("post-submit panic lock is healthy")
                .is_some_and(|failure| failure == (lane.label, lane.start_pos))
            {
                panic!("injected scheduler model panic after submission");
            }
            if self
                .state
                .fail_after_submit_at
                .lock()
                .expect("post-submit failure lock is healthy")
                .is_some_and(|failure| failure == (lane.label, lane.start_pos))
            {
                candle_core::bail!("injected scheduler model failure after submission");
            }
            Ok(())
        }
    }

    impl CausalLm for FakeModel {
        fn runtime_spec(&self) -> &RuntimeSpec {
            &self.spec
        }

        fn new_session(&self) -> candle_core::Result<Box<dyn CausalLmSession>> {
            Ok(Box::new(FakeSession {
                owner: Arc::clone(&self.owner),
                state: Arc::clone(&self.state),
                id: self.state.next_session_id.fetch_add(1, Ordering::Relaxed),
                label: None,
                cache_len: 0,
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
            if session.cache_len != start_pos {
                candle_core::bail!("fake cache position mismatch");
            }
            let input = input_ids.to_vec2::<u32>()?.remove(0);
            let label = session
                .label
                .or_else(|| input.first().copied())
                .unwrap_or(0);
            let lane = ForwardTrace {
                session_id: session.id,
                label,
                input_ids: input,
                start_pos,
            };
            self.state
                .trace
                .lock()
                .expect("forward trace lock is healthy")
                .push(InvocationTrace::Single(lane.clone()));
            self.check_injected_failure(&lane)?;
            session.label = Some(label);
            session.cache_len += lane.input_ids.len();
            let mut logits = vec![0.0_f32; self.spec.vocab_size];
            logits[Self::output_token(label)] = 10.0;
            Tensor::from_vec(logits, (1, self.spec.vocab_size), &self.device)
        }

        fn mixed_batch_capability(&self) -> CausalLmBatchCapability {
            CausalLmBatchCapability::PackedRagged
        }

        fn kv_cache_capacity(&self) -> Option<KvCacheCapacity> {
            self.state
                .reservation_token_limit
                .lock()
                .expect("reservation limit lock is healthy")
                .map(|limit| KvCacheCapacity {
                    block_tokens: 1,
                    total_blocks: self.spec.max_context,
                    free_blocks: limit,
                    live_blocks: 0,
                    retired_blocks: 0,
                })
        }

        fn prefix_cache_diagnostics(&self) -> candle_core::Result<Option<PrefixCacheDiagnostics>> {
            self.state
                .prefix_cache_diagnostics_calls
                .fetch_add(1, Ordering::Relaxed);
            assert!(
                !self
                    .state
                    .panic_prefix_cache_diagnostics
                    .load(Ordering::Relaxed),
                "injected prefix-cache diagnostics panic"
            );
            match &*self
                .state
                .prefix_cache_diagnostics
                .lock()
                .expect("prefix-cache diagnostics lock is healthy")
            {
                Some(Ok(diagnostics)) => Ok(Some(*diagnostics)),
                Some(Err(message)) => Err(candle_core::Error::Msg(message.clone())),
                None => Ok(None),
            }
        }

        fn reserve_batch_kv(
            &self,
            inputs: &[CausalLmKvReservationInput<'_>],
        ) -> std::result::Result<Option<Box<dyn CausalLmKvReservation>>, KvCacheReservationError>
        {
            let Some(limit) = *self
                .state
                .reservation_token_limit
                .lock()
                .expect("reservation limit lock is healthy")
            else {
                return Ok(None);
            };
            let token_counts = inputs
                .iter()
                .map(CausalLmKvReservationInput::input_tokens)
                .collect::<Vec<_>>();
            self.state
                .reservation_attempts
                .lock()
                .expect("reservation trace lock is healthy")
                .push(token_counts.clone());
            let requested = token_counts.iter().sum::<usize>();
            if requested > limit {
                return Err(KvCacheReservationError::Exhausted(KvCacheExhausted {
                    requested_blocks: requested,
                    free_blocks: limit,
                    retired_blocks: 0,
                    total_blocks: self.spec.max_context,
                    block_tokens: 1,
                }));
            }
            for input in inputs {
                self.validate_session(input.session())
                    .map_err(KvCacheReservationError::Fatal)?;
                if input.input_tokens() == 0 || input.session().cache_len() != input.start_pos() {
                    return Err(KvCacheReservationError::Fatal(candle_core::Error::Msg(
                        "invalid fake KV reservation geometry".into(),
                    )));
                }
            }
            Ok(Some(Box::new(FakeKvReservation {
                lanes: inputs.len(),
            })))
        }

        fn forward_batch_tokens(
            &self,
            inputs: &mut [CausalLmBatchInput<'_>],
        ) -> candle_core::Result<Tensor> {
            if inputs.is_empty() {
                candle_core::bail!("fake mixed batch cannot be empty");
            }
            let mut lanes = Vec::with_capacity(inputs.len());
            for input in inputs.iter() {
                self.validate_session(input.session())?;
                let session = input
                    .session()
                    .as_any()
                    .downcast_ref::<FakeSession>()
                    .ok_or_else(|| candle_core::Error::Msg("wrong fake session type".into()))?;
                if session.cache_len != input.start_pos() {
                    candle_core::bail!("fake cache position mismatch");
                }
                let input_ids = input.input_ids().to_vec();
                let label = session
                    .label
                    .or_else(|| input_ids.first().copied())
                    .unwrap_or(0);
                lanes.push(ForwardTrace {
                    session_id: session.id,
                    label,
                    input_ids,
                    start_pos: input.start_pos(),
                });
            }
            self.state
                .trace
                .lock()
                .expect("forward trace lock is healthy")
                .push(InvocationTrace::Packed(lanes.clone()));
            self.state
                .lifecycle
                .lock()
                .expect("lifecycle trace lock is healthy")
                .push(LifecycleTrace::PackedForward(
                    lanes.iter().map(|lane| lane.session_id).collect(),
                ));
            for lane in &lanes {
                self.check_injected_failure(lane)?;
            }

            for (input, lane) in inputs.iter_mut().zip(&lanes) {
                let session = input
                    .session_mut()
                    .as_any_mut()
                    .downcast_mut::<FakeSession>()
                    .ok_or_else(|| candle_core::Error::Msg("wrong fake session type".into()))?;
                session.label = Some(lane.label);
                session.cache_len += lane.input_ids.len();
                if self
                    .state
                    .corrupt_cache_label
                    .lock()
                    .expect("cache corruption lock is healthy")
                    .is_some_and(|label| label == lane.label)
                {
                    session.cache_len += 1;
                }
            }
            for lane in &lanes {
                self.check_injected_post_submit(lane)?;
            }
            let columns = if self.state.bad_batch_shape.load(Ordering::Relaxed) {
                self.spec.vocab_size - 1
            } else {
                self.spec.vocab_size
            };
            let mut logits = vec![0.0_f32; lanes.len() * columns];
            for (row, lane) in lanes.iter().enumerate() {
                logits[row * columns + Self::output_token(lane.label)] = 10.0;
            }
            Tensor::from_vec(logits, (lanes.len(), columns), &self.device)
        }

        fn forward_batch_tokens_reserved(
            &self,
            inputs: &mut [CausalLmBatchInput<'_>],
            reservation: Option<Box<dyn CausalLmKvReservation>>,
        ) -> candle_core::Result<Tensor> {
            let configured = self
                .state
                .reservation_token_limit
                .lock()
                .expect("reservation limit lock is healthy")
                .is_some();
            if configured {
                let reservation = reservation.ok_or_else(|| {
                    candle_core::Error::Msg("fake forward lost its KV reservation".into())
                })?;
                let reservation = reservation
                    .into_any()
                    .downcast::<FakeKvReservation>()
                    .map_err(|_| candle_core::Error::Msg("wrong fake KV reservation".into()))?;
                if reservation.lanes != inputs.len() {
                    candle_core::bail!("fake KV reservation lane count changed");
                }
            } else if reservation.is_some() {
                candle_core::bail!("unexpected fake KV reservation");
            }
            self.forward_batch_tokens(inputs)
        }

        fn read_output_to_cpu(&self, logits: &Tensor) -> candle_core::Result<Tensor> {
            let attempt = self
                .state
                .completion_barrier_calls
                .fetch_add(1, Ordering::Relaxed)
                + 1;
            if self
                .state
                .fail_completion_barrier_at
                .lock()
                .expect("completion failure lock is healthy")
                .is_some_and(|failure| failure == attempt)
            {
                candle_core::bail!("injected scheduler completion barrier failure");
            }
            if let Some(cancelled) = self
                .state
                .cancel_on_completion_barrier
                .lock()
                .expect("completion cancellation lock is healthy")
                .take()
            {
                cancelled.store(true, Ordering::Release);
            }
            logits.to_device(&Device::Cpu)
        }

        fn synchronize(&self) -> candle_core::Result<()> {
            self.state
                .cleanup_drain_calls
                .fetch_add(1, Ordering::Relaxed);
            self.device.synchronize()
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    fn fake_runtime() -> Result<(LoadedRuntime, Arc<FakeState>)> {
        let spec = RuntimeSpec {
            vocab_size: 6,
            max_context: 64,
            stop_token_ids: vec![5],
        };
        let vocab = [
            ("[UNK]".to_owned(), 0),
            ("a".to_owned(), 1),
            ("b".to_owned(), 2),
            ("x".to_owned(), 3),
            ("y".to_owned(), 4),
            ("<eos>".to_owned(), 5),
        ]
        .into_iter()
        .collect();
        let tokenizer_model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_owned())
            .build()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(tokenizer_model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let state = Arc::new(FakeState::default());
        let runtime = LoadedRuntime {
            adapter: Box::new(FakeAdapter { spec: spec.clone() }),
            tokenizer: Arc::new(tokenizer),
            model: Box::new(FakeModel {
                owner: Arc::new(()),
                state: Arc::clone(&state),
                spec,
                device: Device::Cpu,
            }),
            dtype: DTypeKind::F32,
        };
        Ok((runtime, state))
    }

    #[test]
    fn generic_output_readback_uses_the_existing_completion_barrier() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let output = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu)?;

        let materialized = runtime.model.read_output_to_cpu(&output)?;

        assert_eq!(materialized.dims2()?, (2, 3));
        assert_eq!(completion_barrier_calls(&state), 1);
        Ok(())
    }

    fn scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_active_sequences: 2,
            scheduler_token_budget: 4,
            prefill_token_quantum: 2,
            chat_template_defaults: ChatTemplateOptions::default(),
        }
    }

    fn scheduler(
        runtime: &LoadedRuntime,
        jobs: mpsc::Receiver<InferenceJob>,
        status: Arc<AtomicU8>,
        config: SchedulerConfig,
    ) -> Result<ServingScheduler<'_>> {
        scheduler_with_prefix_cache_snapshot(
            runtime,
            jobs,
            status,
            config,
            Arc::new(Mutex::new(EnginePrefixCacheSnapshot::default())),
        )
    }

    fn scheduler_with_prefix_cache_snapshot(
        runtime: &LoadedRuntime,
        jobs: mpsc::Receiver<InferenceJob>,
        status: Arc<AtomicU8>,
        config: SchedulerConfig,
        prefix_cache_snapshot: Arc<Mutex<EnginePrefixCacheSnapshot>>,
    ) -> Result<ServingScheduler<'_>> {
        let sessions = (0..config.max_active_sequences)
            .map(|_| runtime.model.new_session())
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(ServingScheduler::new(
            runtime,
            jobs,
            status,
            sessions,
            config,
            prefix_cache_snapshot,
        ))
    }

    fn job(
        prompt: &str,
        max_new_tokens: usize,
        event_capacity: usize,
        event_send_timeout: Duration,
    ) -> (InferenceJob, EngineTicket) {
        let (events, receiver) = mpsc::channel(event_capacity);
        let (preflight, readiness) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let terminal_failure = Arc::new(Mutex::new(None));
        (
            InferenceJob {
                request: EngineRequest {
                    input: EngineInput::Chat(vec![ChatMessage::user(prompt)]),
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    sampling: SamplingConfig {
                        temperature: 0.0,
                        ..SamplingConfig::default()
                    },
                    max_new_tokens,
                    ignore_stop_tokens: false,
                    stop: Vec::new(),
                    chat_template: ChatTemplateOptions::default(),
                },
                request_policy: EngineRequestPolicy::default(),
                events,
                cancelled: Arc::clone(&cancelled),
                preflight: Some(preflight),
                terminal_failure: Arc::clone(&terminal_failure),
                event_send_timeout,
                batch_admission: None,
                batch_trace: None,
            },
            EngineTicket {
                receiver,
                cancelled,
                preflight: Some(readiness),
                terminal_failure,
            },
        )
    }

    fn preflight_is_ready(ticket: &mut EngineTicket) -> bool {
        matches!(
            ticket
                .preflight
                .as_mut()
                .expect("test ticket has preflight")
                .try_recv(),
            Ok(Ok(()))
        )
    }

    fn trace(state: &FakeState) -> Vec<ForwardTrace> {
        state
            .trace
            .lock()
            .expect("forward trace lock is healthy")
            .iter()
            .flat_map(|invocation| match invocation {
                InvocationTrace::Single(lane) => std::slice::from_ref(lane),
                InvocationTrace::Packed(lanes) => lanes.as_slice(),
            })
            .cloned()
            .collect()
    }

    fn invocations(state: &FakeState) -> Vec<InvocationTrace> {
        state
            .trace
            .lock()
            .expect("forward trace lock is healthy")
            .clone()
    }

    fn reset_lifecycle(state: &FakeState) {
        state
            .lifecycle
            .lock()
            .expect("lifecycle trace lock is healthy")
            .clear();
    }

    fn lifecycle(state: &FakeState) -> Vec<LifecycleTrace> {
        state
            .lifecycle
            .lock()
            .expect("lifecycle trace lock is healthy")
            .clone()
    }

    fn set_reservation_limit(state: &FakeState, limit: usize) {
        *state
            .reservation_token_limit
            .lock()
            .expect("reservation limit lock is healthy") = Some(limit);
        state
            .reservation_attempts
            .lock()
            .expect("reservation trace lock is healthy")
            .clear();
    }

    fn reservation_attempts(state: &FakeState) -> Vec<Vec<usize>> {
        state
            .reservation_attempts
            .lock()
            .expect("reservation trace lock is healthy")
            .clone()
    }

    fn completion_barrier_calls(state: &FakeState) -> usize {
        state.completion_barrier_calls.load(Ordering::Relaxed)
    }

    fn cleanup_drain_calls(state: &FakeState) -> usize {
        state.cleanup_drain_calls.load(Ordering::Relaxed)
    }

    fn cleared_sessions(state: &FakeState) -> Vec<usize> {
        state
            .cleared_sessions
            .lock()
            .expect("clear trace lock is healthy")
            .clone()
    }

    fn received_fatal(ticket: &mut EngineTicket) -> bool {
        loop {
            match ticket.receiver.try_recv() {
                Ok(EngineEvent::Failed {
                    kind: EngineFailureKind::Fatal,
                    ..
                }) => return true,
                Ok(_) => {}
                Err(_) => return false,
            }
        }
    }

    fn received_finished(ticket: &mut EngineTicket) -> bool {
        loop {
            match ticket.receiver.try_recv() {
                Ok(EngineEvent::Finished(_)) => return true,
                Ok(_) => {}
                Err(_) => return false,
            }
        }
    }

    fn collect_success(ticket: &mut EngineTicket) -> (String, GenerationMetrics) {
        let mut text = String::new();
        let mut metrics = None;
        loop {
            match ticket.receiver.try_recv() {
                Ok(EngineEvent::Text {
                    content: fragment, ..
                }) => text.push_str(&fragment),
                Ok(EngineEvent::Finished(result)) => metrics = Some(result),
                Ok(EngineEvent::Failed { message, kind }) => {
                    panic!("expected successful request, received {kind:?}: {message}");
                }
                Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {
                    break;
                }
            }
        }
        (
            text,
            metrics.expect("successful request must include final metrics"),
        )
    }

    fn request() -> EngineRequest {
        EngineRequest {
            input: EngineInput::Chat(vec![ChatMessage::user("hello")]),
            sampling: SamplingConfig::default(),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            max_new_tokens: 1,
            ignore_stop_tokens: false,
            stop: Vec::new(),
            chat_template: ChatTemplateOptions::default(),
        }
    }

    fn prefix_cache_diagnostics(
        physical_live_blocks: usize,
        physical_live_references: usize,
        physical_shared_blocks: usize,
    ) -> PrefixCacheDiagnostics {
        PrefixCacheDiagnostics {
            physical_live_blocks,
            physical_live_references,
            physical_shared_blocks,
            ..PrefixCacheDiagnostics::default()
        }
    }

    #[test]
    fn prefix_cache_snapshot_preserves_physical_peaks_and_fails_closed() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let shared = Arc::new(Mutex::new(EnginePrefixCacheSnapshot::default()));
        *state
            .prefix_cache_diagnostics
            .lock()
            .expect("prefix-cache diagnostics lock is healthy") =
            Some(Ok(prefix_cache_diagnostics(4, 12, 3)));

        refresh_prefix_cache_snapshot(runtime.model.as_ref(), &shared, 8, "peak batch");
        let peak = shared
            .lock()
            .expect("prefix-cache snapshot lock is healthy")
            .clone();
        assert_eq!(peak.current, Some(prefix_cache_diagnostics(4, 12, 3)));
        assert_eq!(peak.peak_physical_live_blocks, 4);
        assert_eq!(peak.peak_physical_live_references, 12);
        assert_eq!(peak.peak_physical_shared_blocks, 3);
        assert_eq!(peak.peak_concurrent_active_sequences, 8);
        assert_eq!(peak.peak_concurrent, peak.current);
        assert_eq!(peak.active_sequences, 8);
        assert_eq!(peak.observation, None);

        *state
            .prefix_cache_diagnostics
            .lock()
            .expect("prefix-cache diagnostics lock is healthy") =
            Some(Ok(prefix_cache_diagnostics(5, 20, 4)));
        refresh_prefix_cache_snapshot(
            runtime.model.as_ref(),
            &shared,
            8,
            "later work at the same concurrency",
        );
        let same_concurrency = shared
            .lock()
            .expect("prefix-cache snapshot lock is healthy")
            .clone();
        assert_eq!(
            same_concurrency.current,
            Some(prefix_cache_diagnostics(5, 20, 4))
        );
        assert_eq!(same_concurrency.peak_physical_live_references, 20);
        assert_eq!(same_concurrency.peak_concurrent, peak.current);

        *state
            .prefix_cache_diagnostics
            .lock()
            .expect("prefix-cache diagnostics lock is healthy") =
            Some(Ok(prefix_cache_diagnostics(1, 1, 0)));
        refresh_prefix_cache_snapshot(runtime.model.as_ref(), &shared, 0, "batch reclaimed");
        let reclaimed = shared
            .lock()
            .expect("prefix-cache snapshot lock is healthy")
            .clone();
        assert_eq!(reclaimed.current, Some(prefix_cache_diagnostics(1, 1, 0)));
        assert_eq!(reclaimed.peak_physical_live_blocks, 5);
        assert_eq!(reclaimed.peak_physical_live_references, 20);
        assert_eq!(reclaimed.peak_physical_shared_blocks, 4);
        assert_eq!(reclaimed.peak_concurrent_active_sequences, 8);
        assert_eq!(reclaimed.peak_concurrent, peak.current);
        assert_eq!(reclaimed.active_sequences, 0);

        *state
            .prefix_cache_diagnostics
            .lock()
            .expect("prefix-cache diagnostics lock is healthy") =
            Some(Err("injected diagnostics failure".to_owned()));
        refresh_prefix_cache_snapshot(runtime.model.as_ref(), &shared, 0, "after failure");
        let failed = shared
            .lock()
            .expect("prefix-cache snapshot lock is healthy")
            .clone();
        assert_eq!(failed.current, None);
        assert_eq!(failed.peak_physical_live_references, 20);
        assert!(
            failed
                .observation
                .as_deref()
                .is_some_and(|message| message.contains("injected diagnostics failure"))
        );

        state
            .panic_prefix_cache_diagnostics
            .store(true, Ordering::Relaxed);
        refresh_prefix_cache_snapshot(runtime.model.as_ref(), &shared, 0, "after panic");
        let panicked = shared
            .lock()
            .expect("prefix-cache snapshot lock is healthy")
            .clone();
        assert_eq!(panicked.current, None);
        assert_eq!(panicked.peak_physical_live_references, 20);
        assert!(
            panicked
                .observation
                .as_deref()
                .is_some_and(|message| message.contains("collection panicked"))
        );
        Ok(())
    }

    #[test]
    fn scheduler_mirrors_admission_commit_and_reclaim_without_model_errors_becoming_fatal()
    -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (_jobs, receiver) = mpsc::channel(1);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let shared = Arc::new(Mutex::new(EnginePrefixCacheSnapshot::default()));
        *state
            .prefix_cache_diagnostics
            .lock()
            .expect("prefix-cache diagnostics lock is healthy") =
            Some(Ok(prefix_cache_diagnostics(2, 5, 2)));
        let mut scheduler = scheduler_with_prefix_cache_snapshot(
            &runtime,
            receiver,
            Arc::clone(&status),
            scheduler_config(),
            Arc::clone(&shared),
        )?;
        let (request, mut ticket) = job("a", 1, 8, Duration::from_secs(1));

        scheduler.admit(request)?;
        assert!(preflight_is_ready(&mut ticket));
        let admitted = shared
            .lock()
            .expect("prefix-cache snapshot lock is healthy")
            .clone();
        assert_eq!(admitted.active_sequences, 1);
        assert_eq!(admitted.current, Some(prefix_cache_diagnostics(2, 5, 2)));

        *state
            .prefix_cache_diagnostics
            .lock()
            .expect("prefix-cache diagnostics lock is healthy") =
            Some(Err("diagnostics unavailable during generation".to_owned()));
        scheduler.drive_once(Instant::now())?;

        assert!(received_finished(&mut ticket));
        assert!(scheduler.active.is_empty());
        assert_eq!(scheduler.sessions.len(), 2);
        assert_eq!(
            EngineStatus::from_atomic(status.load(Ordering::Acquire)),
            EngineStatus::Ready
        );
        let completed = shared
            .lock()
            .expect("prefix-cache snapshot lock is healthy")
            .clone();
        assert_eq!(completed.current, None);
        assert_eq!(completed.active_sequences, 0);
        assert_eq!(completed.peak_physical_live_references, 5);
        assert!(
            completed.observation.as_deref().is_some_and(
                |message| message.contains("diagnostics unavailable during generation")
            )
        );
        assert!(state.prefix_cache_diagnostics_calls.load(Ordering::Relaxed) >= 3);
        Ok(())
    }

    #[test]
    fn raw_token_input_bypasses_chat_rendering_and_preserves_ids() -> Result<()> {
        let (runtime, _state) = fake_runtime()?;
        let (mut raw_job, _ticket) =
            job("text that must not be used", 2, 4, Duration::from_secs(1));
        raw_job.request.input = EngineInput::TokenIds(vec![4, 0, 2]);
        raw_job.request.ignore_stop_tokens = true;

        let (tokens, _sampler) =
            prepare_request(&runtime, &mut raw_job, ChatTemplateOptions::default())
                .context("raw-token preflight failed")?;

        assert_eq!(tokens, vec![4, 0, 2]);
        assert!(raw_job.request.ignore_stop_tokens);
        Ok(())
    }

    #[test]
    fn raw_prompt_input_bypasses_chat_rendering_and_uses_the_tokenizer() -> Result<()> {
        let (runtime, _state) = fake_runtime()?;
        let (mut raw_job, _ticket) =
            job("text that must not be used", 2, 4, Duration::from_secs(1));
        raw_job.request.input = EngineInput::RawPrompt("a b".to_owned());
        raw_job.request.ignore_stop_tokens = true;

        let (tokens, _sampler) =
            prepare_request(&runtime, &mut raw_job, ChatTemplateOptions::default())
                .context("raw-prompt preflight failed")?;

        assert_eq!(tokens, vec![1, 2]);
        assert!(raw_job.request.ignore_stop_tokens);
        Ok(())
    }

    #[test]
    fn raw_prompt_input_rejects_empty_text() -> Result<()> {
        let (runtime, _state) = fake_runtime()?;
        let (mut raw_job, mut ticket) = job("unused", 1, 4, Duration::from_secs(1));
        raw_job.request.input = EngineInput::RawPrompt(String::new());

        assert!(prepare_request(&runtime, &mut raw_job, ChatTemplateOptions::default()).is_none());
        let error = ticket
            .preflight
            .as_mut()
            .context("raw-prompt test ticket lost preflight")?
            .try_recv()
            .context("raw-prompt rejection was not delivered")?
            .expect_err("an empty raw prompt must fail preflight");
        assert!(error.message.contains("raw prompt must not be empty"));
        assert_eq!(error.kind, EngineFailureKind::InvalidRequest);
        Ok(())
    }

    #[test]
    fn chat_template_options_fail_closed_without_adapter_support() -> Result<()> {
        let (runtime, _state) = fake_runtime()?;
        let disabled = ChatTemplateOptions {
            enable_thinking: Some(false),
        };
        let rejected = |job: &mut InferenceJob, ticket: &mut EngineTicket, defaults| {
            assert!(prepare_request(&runtime, job, defaults).is_none());
            ticket
                .preflight
                .as_mut()
                .expect("test ticket keeps preflight")
                .try_recv()
                .expect("rejection is delivered")
                .expect_err("options must fail preflight")
        };

        // The fake adapter implements no template options, whether they come
        // from the request or from server-wide defaults.
        let (mut chat_job, mut ticket) = job("hello", 1, 4, Duration::from_secs(1));
        chat_job.request.chat_template = disabled;
        let error = rejected(&mut chat_job, &mut ticket, ChatTemplateOptions::default());
        assert!(error.message.contains("enable_thinking"));
        let (mut chat_job, mut ticket) = job("hello", 1, 4, Duration::from_secs(1));
        let error = rejected(&mut chat_job, &mut ticket, disabled);
        assert!(error.message.contains("enable_thinking"));

        // Raw inputs bypass templates: explicit options are an error, while
        // server defaults do not apply to them.
        let (mut raw_job, mut ticket) = job("unused", 1, 4, Duration::from_secs(1));
        raw_job.request.input = EngineInput::RawPrompt("a b".to_owned());
        raw_job.request.chat_template = disabled;
        let error = rejected(&mut raw_job, &mut ticket, ChatTemplateOptions::default());
        assert!(error.message.contains("templated"));
        let (mut raw_job, _ticket) = job("unused", 1, 4, Duration::from_secs(1));
        raw_job.request.input = EngineInput::RawPrompt("a b".to_owned());
        assert!(prepare_request(&runtime, &mut raw_job, disabled).is_some());
        Ok(())
    }

    #[test]
    fn context_overflow_uses_openai_wording_with_token_detail() -> Result<()> {
        let (runtime, _state) = fake_runtime()?;
        let max_context = runtime.model.runtime_spec().max_context;
        let requested = max_context + 1;
        let (mut long_job, mut ticket) = job("hello", requested, 4, Duration::from_secs(1));
        assert!(prepare_request(&runtime, &mut long_job, ChatTemplateOptions::default()).is_none());
        let error = ticket
            .preflight
            .as_mut()
            .context("test ticket lost preflight")?
            .try_recv()
            .context("rejection was not delivered")?
            .expect_err("an oversized request must fail preflight");
        assert_eq!(error.kind, EngineFailureKind::InvalidRequest);
        assert!(error.message.starts_with(&format!(
            "This model's maximum context length is {max_context} tokens; the prompt ("
        )));
        assert!(
            error
                .message
                .contains(&format!("requested output ({requested} tokens)"))
        );
        Ok(())
    }

    #[test]
    fn raw_token_input_rejects_empty_and_out_of_vocabulary_prompts() -> Result<()> {
        let (runtime, _state) = fake_runtime()?;

        for (tokens, expected) in [
            (Vec::new(), "must not be empty"),
            (vec![1, 6], "token 6 at index 1 exceeds model vocabulary 6"),
        ] {
            let (mut raw_job, mut ticket) = job("unused", 1, 4, Duration::from_secs(1));
            raw_job.request.input = EngineInput::TokenIds(tokens);
            assert!(
                prepare_request(&runtime, &mut raw_job, ChatTemplateOptions::default()).is_none()
            );
            let error = ticket
                .preflight
                .as_mut()
                .context("raw-token test ticket lost preflight")?
                .try_recv()
                .context("raw-token rejection was not delivered")?
                .expect_err("invalid raw tokens must fail preflight");
            assert!(error.message.contains(expected), "{}", error.message);
            assert_eq!(error.kind, EngineFailureKind::InvalidRequest);
        }
        Ok(())
    }

    #[test]
    fn status_health_is_explicit() {
        assert!(!EngineStatus::Starting.is_healthy());
        assert!(EngineStatus::Ready.is_healthy());
        assert!(EngineStatus::Busy.is_healthy());
        assert!(!EngineStatus::Poisoned.is_healthy());
        assert!(!EngineStatus::Stopped.is_healthy());
    }

    #[test]
    fn scheduler_limits_must_be_positive_and_internally_consistent() {
        let base = EngineConfig::new(
            ModelLoadConfig::new("unused", crate::backend::DeviceKind::Cpu, None),
            "test-model",
        );
        assert!(base.validate().is_ok());

        let mut config = base.clone();
        config.max_active_sequences = 0;
        assert!(config.validate().is_err());
        let mut config = base.clone();
        config.scheduler_token_budget = 0;
        assert!(config.validate().is_err());
        let mut config = base.clone();
        config.prefill_token_quantum = 0;
        assert!(config.validate().is_err());
        let mut config = base;
        config.prefill_token_quantum = config.scheduler_token_budget + 1;
        assert!(config.validate().is_err());
    }

    #[tokio::test]
    async fn dropping_ticket_sets_sticky_cancellation() {
        let (_events, receiver) = mpsc::channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let ticket = EngineTicket {
            receiver,
            cancelled: Arc::clone(&cancelled),
            preflight: None,
            terminal_failure: Arc::new(Mutex::new(None)),
        };
        drop(ticket);
        assert!(cancelled.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn admission_distinguishes_full_closed_and_poisoned() {
        let (jobs, receiver) = mpsc::channel(1);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let handle = EngineHandle {
            jobs,
            submit_lock: Arc::new(Mutex::new(())),
            status: Arc::clone(&status),
            metadata: EngineMetadata {
                served_model: "test-model".to_owned(),
                family: "test".to_owned(),
                max_context: 8,
                device: "cpu".to_owned(),
                dtype: "f32".to_owned(),
                chat_template_defaults: ChatTemplateOptions::default(),
            },
            prefix_cache_snapshot: Arc::new(Mutex::new(EnginePrefixCacheSnapshot::default())),
            event_capacity: 1,
            event_send_timeout: Duration::from_millis(1),
            maximum_atomic_batch: 1,
            capture_benchmark_trace: false,
        };
        assert_eq!(
            handle.prefix_cache_snapshot(),
            EnginePrefixCacheSnapshot::default()
        );

        let first = handle.submit(request()).expect("first job fits");
        assert!(matches!(handle.submit(request()), Err(SubmitError::Full)));
        drop(receiver);
        assert!(matches!(handle.submit(request()), Err(SubmitError::Closed)));
        drop(first);

        status.store(EngineStatus::Poisoned as u8, Ordering::Release);
        assert!(matches!(
            handle.submit(request()),
            Err(SubmitError::Unavailable(EngineStatus::Poisoned))
        ));
    }

    #[test]
    fn atomic_batch_is_fully_admitted_before_its_first_forward() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (jobs, receiver) = mpsc::channel(3);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let handle = EngineHandle {
            jobs,
            submit_lock: Arc::new(Mutex::new(())),
            status: Arc::clone(&status),
            metadata: EngineMetadata {
                served_model: "test-model".to_owned(),
                family: "test".to_owned(),
                max_context: 8,
                device: "cpu".to_owned(),
                dtype: "f32".to_owned(),
                chat_template_defaults: ChatTemplateOptions::default(),
            },
            prefix_cache_snapshot: Arc::new(Mutex::new(EnginePrefixCacheSnapshot::default())),
            event_capacity: 8,
            event_send_timeout: Duration::from_secs(1),
            maximum_atomic_batch: 3,
            capture_benchmark_trace: true,
        };
        let submission = handle
            .submit_batch_with_policy(vec![
                (request(), EngineRequestPolicy::benchmark_measured()),
                (request(), EngineRequestPolicy::benchmark_measured()),
                (request(), EngineRequestPolicy::benchmark_measured()),
            ])
            .expect("three-request batch fits atomically");
        assert_eq!(submission.len(), 3);
        let batch_trace = submission
            .benchmark_trace()
            .expect("trace capture is enabled");
        let mut tickets = submission.into_tickets();
        let config = SchedulerConfig {
            max_active_sequences: 3,
            scheduler_token_budget: 3,
            prefill_token_quantum: 1,
            chat_template_defaults: ChatTemplateOptions::default(),
        };
        let mut scheduler = scheduler(&runtime, receiver, status, config)?;

        scheduler.drive_once(Instant::now())?;

        assert!(tickets.iter_mut().all(preflight_is_ready));
        assert!(matches!(
            invocations(&state).as_slice(),
            [InvocationTrace::Packed(lanes)] if lanes.len() == 3
        ));
        let trace = batch_trace
            .try_read()
            .expect("all three one-token requests are terminal");
        assert_eq!(trace.expected_members, 3);
        assert_eq!(trace.admitted_members, 3);
        assert_eq!(trace.terminal_members, 3);
        assert_eq!(trace.rejected_members, 0);
        assert_eq!(trace.admission_active_sequences, 3);
        assert_eq!(trace.packed_forwards.len(), 1);
        assert_eq!(
            trace.packed_forwards[0].kind,
            BenchmarkForwardKind::FirstToken
        );
        assert_eq!(trace.packed_forwards[0].lane_count, 3);
        assert!(
            trace.packed_forwards[0]
                .lanes
                .iter()
                .all(|lane| lane.member_index.is_some())
        );
        assert_eq!(
            trace.first_token_model_time,
            trace.packed_forwards[0].model_time
        );
        assert_eq!(trace.suffix_prefill_time, Duration::ZERO);
        assert_eq!(trace.decode_time, Duration::ZERO);
        assert_eq!(trace.mixed_forward_time, Duration::ZERO);
        assert_eq!(
            state.prefix_cache_diagnostics_calls.load(Ordering::Relaxed),
            2,
            "atomic admission samples once; measured requests skip publish-time sampling and completion reclaims once"
        );
        Ok(())
    }

    #[test]
    fn benchmark_trace_retains_each_packed_decode_forward() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (jobs, receiver) = mpsc::channel(2);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let handle = EngineHandle {
            jobs,
            submit_lock: Arc::new(Mutex::new(())),
            status: Arc::clone(&status),
            metadata: EngineMetadata {
                served_model: "test-model".to_owned(),
                family: "test".to_owned(),
                max_context: 8,
                device: "cpu".to_owned(),
                dtype: "f32".to_owned(),
                chat_template_defaults: ChatTemplateOptions::default(),
            },
            prefix_cache_snapshot: Arc::new(Mutex::new(EnginePrefixCacheSnapshot::default())),
            event_capacity: 8,
            event_send_timeout: Duration::from_secs(1),
            maximum_atomic_batch: 2,
            capture_benchmark_trace: true,
        };
        let mut first = request();
        first.max_new_tokens = 2;
        let mut second = request();
        second.max_new_tokens = 2;
        let submission = handle.submit_batch_with_policy(vec![
            (first, EngineRequestPolicy::benchmark_measured()),
            (second, EngineRequestPolicy::benchmark_measured()),
        ])?;
        let trace_handle = submission
            .benchmark_trace()
            .expect("trace capture is enabled");
        let _tickets = submission.into_tickets();
        let config = SchedulerConfig {
            max_active_sequences: 2,
            scheduler_token_budget: 2,
            prefill_token_quantum: 1,
            chat_template_defaults: ChatTemplateOptions::default(),
        };
        let mut scheduler = scheduler(&runtime, receiver, status, config)?;

        scheduler.drive_once(Instant::now())?;
        assert!(trace_handle.try_read().is_none());
        assert_eq!(
            state.prefix_cache_diagnostics_calls.load(Ordering::Relaxed),
            1,
            "the whole atomic batch must produce one first-admission sample"
        );
        scheduler.drive_once(Instant::now())?;

        let trace = trace_handle
            .try_read()
            .expect("two-token requests are terminal after one decode step");
        assert_eq!(
            trace
                .packed_forwards
                .iter()
                .map(|record| record.kind)
                .collect::<Vec<_>>(),
            [
                BenchmarkForwardKind::FirstToken,
                BenchmarkForwardKind::Decode
            ]
        );
        assert!(
            trace
                .packed_forwards
                .iter()
                .all(|record| record.lane_count == 2 && record.input_tokens == 2)
        );
        assert_eq!(trace.decode_time, trace.packed_forwards[1].model_time);
        assert_eq!(trace.mixed_forward_time, Duration::ZERO);
        Ok(())
    }

    #[test]
    fn partial_atomic_batch_send_aborts_queued_members_without_waiting() {
        let (jobs, mut receiver) = mpsc::channel(2);
        let (blocker, _blocker_ticket) = job("a", 1, 1, Duration::from_secs(1));
        jobs.try_send(blocker).expect("blocker occupies one slot");
        let handle = EngineHandle {
            jobs,
            submit_lock: Arc::new(Mutex::new(())),
            status: Arc::new(AtomicU8::new(EngineStatus::Ready as u8)),
            metadata: EngineMetadata {
                served_model: "test-model".to_owned(),
                family: "test".to_owned(),
                max_context: 8,
                device: "cpu".to_owned(),
                dtype: "f32".to_owned(),
                chat_template_defaults: ChatTemplateOptions::default(),
            },
            prefix_cache_snapshot: Arc::new(Mutex::new(EnginePrefixCacheSnapshot::default())),
            event_capacity: 1,
            event_send_timeout: Duration::from_secs(1),
            maximum_atomic_batch: 2,
            capture_benchmark_trace: false,
        };

        assert!(matches!(
            handle.submit_batch(vec![request(), request()]),
            Err(BatchSubmitError::Submit(SubmitError::Full))
        ));
        let _blocker = receiver.try_recv().expect("blocker remains first");
        let partial = receiver
            .try_recv()
            .expect("first batch member was queued before the full error");
        let gate = partial
            .batch_admission
            .clone()
            .expect("partial batch member retains its admission gate");
        assert_eq!(gate.state(), BatchAdmissionState::Aborted);
        assert!(!gate.wait_until_released());
        assert!(partial.cancelled.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn poisoned_backpressured_ticket_preserves_the_fatal_reason() -> Result<()> {
        let (runtime, _state) = fake_runtime()?;
        let (_jobs, receiver) = mpsc::channel(1);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let mut scheduler = scheduler(&runtime, receiver, status, scheduler_config())?;
        let (request, mut ticket) = job("a", 1, 1, Duration::from_secs(1));
        scheduler.admit(request)?;
        assert!(preflight_is_ready(&mut ticket));
        scheduler.active[0]
            .job
            .events
            .try_send(EngineEvent::Text {
                content: "already buffered".to_owned(),
                token: Some(EngineTokenEvidence {
                    token_id: 3,
                    n_decoded: 1,
                    prompt_tokens: 1,
                }),
            })
            .expect("one event fits before poison");

        scheduler.poison("specific device failure".to_owned());

        assert!(matches!(
            ticket.recv().await,
            Some(EngineEvent::Text { .. })
        ));
        let Some(EngineEvent::Failed { message, kind }) = ticket.recv().await else {
            panic!("a full event channel must retain the fatal terminal reason");
        };
        assert_eq!(kind, EngineFailureKind::Fatal);
        assert_eq!(message, "specific device failure");
        assert!(ticket.recv().await.is_none());
        Ok(())
    }

    #[test]
    fn scheduler_batches_ragged_prefill_and_decode_with_stable_rows() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (jobs, receiver) = mpsc::channel(4);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let mut scheduler = scheduler(&runtime, receiver, status, scheduler_config())?;
        let (long, mut long_ticket) = job("a b x y a b", 1, 8, Duration::from_secs(1));
        let (short, mut short_ticket) = job("b", 2, 8, Duration::from_secs(1));
        jobs.try_send(long).expect("long request fits");
        jobs.try_send(short).expect("short request fits");

        let now = Instant::now();
        scheduler.drive_once(now)?;
        assert!(preflight_is_ready(&mut long_ticket));
        assert!(preflight_is_ready(&mut short_ticket));
        scheduler.drive_once(now)?;
        scheduler.drive_once(now)?;

        let trace = trace(&state);
        assert_eq!(
            trace.iter().map(|step| step.label).collect::<Vec<_>>(),
            [1, 2, 1, 2, 1]
        );
        assert!(trace.iter().all(|step| step.input_ids.len() <= 2));
        let invocations = invocations(&state);
        assert_eq!(invocations.len(), 3);
        assert!(matches!(
            &invocations[0],
            InvocationTrace::Packed(lanes)
                if lanes.iter().map(|lane| (lane.label, lane.start_pos, lane.input_ids.as_slice())).collect::<Vec<_>>()
                    == [(1, 0, [1, 2].as_slice()), (2, 0, [2].as_slice())]
        ));
        assert!(matches!(
            &invocations[1],
            InvocationTrace::Packed(lanes)
                if lanes.iter().map(|lane| (lane.label, lane.start_pos, lane.input_ids.as_slice())).collect::<Vec<_>>()
                    == [(1, 2, [3, 4].as_slice()), (2, 1, [4].as_slice())]
        ));
        assert!(matches!(
            &invocations[2],
            InvocationTrace::Packed(lanes)
                if lanes.iter().map(|lane| (lane.label, lane.start_pos, lane.input_ids.as_slice())).collect::<Vec<_>>()
                    == [(1, 4, [1, 2].as_slice())]
        ));
        assert!(
            invocations
                .iter()
                .all(|invocation| !matches!(invocation, InvocationTrace::Single(_)))
        );
        assert_eq!(completion_barrier_calls(&state), invocations.len());
        assert_eq!(cleanup_drain_calls(&state), 0);

        let (long_text, long_metrics) = collect_success(&mut long_ticket);
        assert_eq!(long_text.replace(' ', ""), "x");
        assert_eq!(long_metrics.prompt_tokens, 6);
        assert_eq!(long_metrics.generated_tokens, 1);
        assert_eq!(long_metrics.decode_steps, 0);
        let (short_text, short_metrics) = collect_success(&mut short_ticket);
        assert_eq!(short_text.replace(' ', ""), "yy");
        assert_eq!(short_metrics.prompt_tokens, 1);
        assert_eq!(short_metrics.generated_tokens, 2);
        assert_eq!(short_metrics.decode_steps, 1);
        assert!(scheduler.active.is_empty());
        assert_eq!(scheduler.sessions.len(), 2);
        Ok(())
    }

    #[test]
    fn request_stop_is_suppressed_and_finishes_without_another_decode() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (_jobs, receiver) = mpsc::channel(1);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let mut scheduler = scheduler(&runtime, receiver, status, scheduler_config())?;
        let (mut request, mut ticket) = job("a", 4, 8, Duration::from_secs(1));
        request.request.stop = vec!["x".to_owned()];
        scheduler.admit(request)?;
        assert!(preflight_is_ready(&mut ticket));

        scheduler.drive_once(Instant::now())?;
        let EngineEvent::Text { content, token } = ticket
            .receiver
            .try_recv()
            .expect("matched stop still emits sampled-token evidence")
        else {
            panic!("sampled token must precede terminal metrics");
        };
        assert!(content.is_empty());
        assert_eq!(
            token,
            Some(EngineTokenEvidence {
                token_id: 3,
                n_decoded: 1,
                prompt_tokens: 1,
            })
        );
        let EngineEvent::Finished(metrics) = ticket
            .receiver
            .try_recv()
            .expect("stopped request emits terminal metrics")
        else {
            panic!("sampled-token evidence must be followed by terminal metrics");
        };
        assert_eq!(metrics.generated_tokens, 1);
        assert_eq!(metrics.decode_steps, 0);
        assert_eq!(metrics.finish_reason, FinishReason::Stop);
        assert_eq!(invocations(&state).len(), 1);
        assert!(scheduler.active.is_empty());
        Ok(())
    }

    #[test]
    fn scheduler_fills_available_slots_before_the_first_iteration() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (jobs, receiver) = mpsc::channel(3);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let config = SchedulerConfig {
            max_active_sequences: 3,
            scheduler_token_budget: 3,
            prefill_token_quantum: 1,
            chat_template_defaults: ChatTemplateOptions::default(),
        };
        let mut scheduler = scheduler(&runtime, receiver, status, config)?;
        let (first, mut first_ticket) = job("a", 1, 8, Duration::from_secs(1));
        let (second, mut second_ticket) = job("b", 1, 8, Duration::from_secs(1));
        let (third, mut third_ticket) = job("x", 1, 8, Duration::from_secs(1));
        jobs.try_send(first).expect("first request fits");
        jobs.try_send(second).expect("second request fits");
        jobs.try_send(third).expect("third request fits");

        scheduler.drive_once(Instant::now())?;

        assert!(preflight_is_ready(&mut first_ticket));
        assert!(preflight_is_ready(&mut second_ticket));
        assert!(preflight_is_ready(&mut third_ticket));
        assert!(matches!(
            invocations(&state).as_slice(),
            [InvocationTrace::Packed(lanes)]
                if lanes.iter().map(|lane| lane.label).collect::<Vec<_>>() == [1, 2, 3]
        ));
        assert_eq!(completion_barrier_calls(&state), 1);
        assert_eq!(cleanup_drain_calls(&state), 0);
        Ok(())
    }

    #[test]
    fn kv_pressure_shrinks_the_batch_before_generation_becomes_in_flight() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        set_reservation_limit(&state, 2);
        let (_jobs, receiver) = mpsc::channel(1);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let config = SchedulerConfig {
            max_active_sequences: 3,
            scheduler_token_budget: 3,
            prefill_token_quantum: 1,
            chat_template_defaults: ChatTemplateOptions::default(),
        };
        let mut scheduler = scheduler(&runtime, receiver, status, config)?;
        let (first, mut first_ticket) = job("a", 1, 8, Duration::from_secs(1));
        let (second, mut second_ticket) = job("b", 1, 8, Duration::from_secs(1));
        let (third, mut third_ticket) = job("x", 1, 8, Duration::from_secs(1));
        scheduler.admit(first)?;
        scheduler.admit(second)?;
        scheduler.admit(third)?;
        assert!(preflight_is_ready(&mut first_ticket));
        assert!(preflight_is_ready(&mut second_ticket));
        assert!(preflight_is_ready(&mut third_ticket));

        scheduler.drive_once(Instant::now())?;
        assert_eq!(reservation_attempts(&state), [vec![1, 1, 1], vec![1, 1]]);
        assert!(matches!(
            invocations(&state).as_slice(),
            [InvocationTrace::Packed(lanes)] if lanes.len() == 2
        ));
        assert_eq!(scheduler.active.len(), 1);
        assert!(
            !scheduler.active[0]
                .sequence
                .as_ref()
                .expect("deferred lane keeps its generation sequence")
                .is_terminal()
        );

        scheduler.drive_once(Instant::now())?;
        assert_eq!(
            reservation_attempts(&state),
            [vec![1, 1, 1], vec![1, 1], vec![1]]
        );
        assert_eq!(invocations(&state).len(), 2);
        let _ = collect_success(&mut first_ticket);
        let _ = collect_success(&mut second_ticket);
        let _ = collect_success(&mut third_ticket);
        assert_eq!(completion_barrier_calls(&state), 2);
        assert_eq!(cleanup_drain_calls(&state), 0);
        assert!(scheduler.active.is_empty());
        Ok(())
    }

    #[test]
    fn irreducible_kv_pressure_fails_only_that_request_without_model_submission() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        set_reservation_limit(&state, 0);
        let (_jobs, receiver) = mpsc::channel(1);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let config = SchedulerConfig {
            max_active_sequences: 1,
            scheduler_token_budget: 1,
            prefill_token_quantum: 1,
            chat_template_defaults: ChatTemplateOptions::default(),
        };
        let mut scheduler = scheduler(&runtime, receiver, status, config)?;
        let (request, mut ticket) = job("a", 1, 8, Duration::from_secs(1));
        scheduler.admit(request)?;
        assert!(preflight_is_ready(&mut ticket));

        scheduler.drive_once(Instant::now())?;

        assert_eq!(reservation_attempts(&state), [vec![1]]);
        assert!(invocations(&state).is_empty());
        let Some(EngineEvent::Failed { message, kind }) = ticket.receiver.try_recv().ok() else {
            panic!("KV pressure must produce one request-local failure");
        };
        assert_eq!(kind, EngineFailureKind::ResourceExhausted);
        assert!(message.contains("KV cache requires"));
        assert_eq!(completion_barrier_calls(&state), 0);
        assert_eq!(cleanup_drain_calls(&state), 0);
        assert!(scheduler.active.is_empty());
        assert_eq!(scheduler.sessions.len(), 1);
        Ok(())
    }

    #[test]
    fn cancellation_after_shared_completion_drops_only_its_lane() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (_jobs, receiver) = mpsc::channel(1);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let mut scheduler = scheduler(&runtime, receiver, status, scheduler_config())?;
        let (cancelled, mut cancelled_ticket) = job("a", 1, 8, Duration::from_secs(1));
        let (peer, mut peer_ticket) = job("b", 1, 8, Duration::from_secs(1));
        scheduler.admit(cancelled)?;
        scheduler.admit(peer)?;
        assert!(preflight_is_ready(&mut cancelled_ticket));
        assert!(preflight_is_ready(&mut peer_ticket));
        *state
            .cancel_on_completion_barrier
            .lock()
            .expect("completion cancellation lock is healthy") =
            Some(Arc::clone(&cancelled_ticket.cancelled));

        scheduler.drive_once(Instant::now())?;

        assert!(cancelled_ticket.cancelled.load(Ordering::Acquire));
        assert!(cancelled_ticket.receiver.try_recv().is_err());
        let (peer_text, peer_metrics) = collect_success(&mut peer_ticket);
        assert_eq!(peer_text.replace(' ', ""), "y");
        assert_eq!(peer_metrics.generated_tokens, 1);
        assert!(matches!(
            invocations(&state).as_slice(),
            [InvocationTrace::Packed(lanes)]
                if lanes.iter().map(|lane| lane.label).collect::<Vec<_>>() == [1, 2]
        ));
        assert_eq!(completion_barrier_calls(&state), 1);
        assert_eq!(cleanup_drain_calls(&state), 0);
        assert!(scheduler.active.is_empty());
        assert_eq!(scheduler.sessions.len(), 2);
        Ok(())
    }

    #[test]
    fn budget_smaller_than_active_count_preserves_the_round_robin_cursor() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (_jobs, receiver) = mpsc::channel(1);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let config = SchedulerConfig {
            max_active_sequences: 3,
            scheduler_token_budget: 2,
            prefill_token_quantum: 1,
            chat_template_defaults: ChatTemplateOptions::default(),
        };
        let mut scheduler = scheduler(&runtime, receiver, status, config)?;
        let (first, mut first_ticket) = job("a a a a", 1, 8, Duration::from_secs(1));
        let (second, mut second_ticket) = job("b b b b", 1, 8, Duration::from_secs(1));
        let (third, mut third_ticket) = job("y y y y", 1, 8, Duration::from_secs(1));
        scheduler.admit(first)?;
        scheduler.admit(second)?;
        scheduler.admit(third)?;
        assert!(preflight_is_ready(&mut first_ticket));
        assert!(preflight_is_ready(&mut second_ticket));
        assert!(preflight_is_ready(&mut third_ticket));

        let now = Instant::now();
        for _ in 0..3 {
            scheduler.drive_once(now)?;
        }
        let trace = trace(&state);
        assert_eq!(
            trace.iter().map(|step| step.label).collect::<Vec<_>>(),
            [1, 2, 4, 1, 2, 4]
        );
        assert!(trace.iter().all(|step| step.input_ids.len() == 1));
        assert_eq!(
            invocations(&state)
                .into_iter()
                .map(|invocation| match invocation {
                    InvocationTrace::Packed(lanes) => {
                        lanes.into_iter().map(|lane| lane.label).collect::<Vec<_>>()
                    }
                    InvocationTrace::Single(_) => panic!("scheduler used a serial model call"),
                })
                .collect::<Vec<_>>(),
            [vec![1, 2], vec![4, 1], vec![2, 4]]
        );
        assert_eq!(completion_barrier_calls(&state), 3);
        assert_eq!(cleanup_drain_calls(&state), 0);
        Ok(())
    }

    #[test]
    fn cancelled_queued_request_is_skipped_before_preflight_and_model_work() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (jobs, receiver) = mpsc::channel(4);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let config = SchedulerConfig {
            max_active_sequences: 1,
            scheduler_token_budget: 1,
            prefill_token_quantum: 1,
            chat_template_defaults: ChatTemplateOptions::default(),
        };
        let mut scheduler = scheduler(&runtime, receiver, status, config)?;
        let (active, active_ticket) = job("a a a a", 1, 8, Duration::from_secs(1));
        let (cancelled, mut cancelled_ticket) = job("b b", 1, 8, Duration::from_secs(1));
        let (healthy, mut healthy_ticket) = job("y", 1, 8, Duration::from_secs(1));
        jobs.try_send(active).expect("active request fits");
        jobs.try_send(cancelled).expect("cancelled request fits");
        jobs.try_send(healthy).expect("healthy request fits");

        let now = Instant::now();
        scheduler.drive_once(now)?;
        cancelled_ticket.cancel();
        active_ticket.cancel();
        scheduler.drive_once(now)?;
        assert!(scheduler.active.is_empty());
        scheduler.drive_once(now)?;
        assert!(scheduler.active.is_empty());
        assert!(matches!(
            cancelled_ticket
                .preflight
                .as_mut()
                .expect("cancelled ticket has preflight")
                .try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
        assert!(!trace(&state).iter().any(|step| step.label == 2));

        scheduler.drive_once(now)?;
        assert!(preflight_is_ready(&mut healthy_ticket));
        assert!(trace(&state).iter().any(|step| step.label == 4));
        Ok(())
    }

    #[test]
    fn active_limit_delays_preflight_until_a_slot_is_reclaimed() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (jobs, receiver) = mpsc::channel(4);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let mut scheduler = scheduler(&runtime, receiver, status, scheduler_config())?;
        let (first, mut first_ticket) = job("a a a a", 3, 8, Duration::from_secs(1));
        let (second, mut second_ticket) = job("b b b b", 3, 8, Duration::from_secs(1));
        let (third, mut third_ticket) = job("y b", 1, 8, Duration::from_secs(1));
        jobs.try_send(first).expect("first request fits");
        jobs.try_send(second).expect("second request fits");
        jobs.try_send(third).expect("third request fits");

        let now = Instant::now();
        scheduler.drive_once(now)?;
        scheduler.drive_once(now)?;
        let first_session = trace(&state)
            .first()
            .expect("first request executes a prefill quantum")
            .session_id;
        assert!(preflight_is_ready(&mut first_ticket));
        assert!(preflight_is_ready(&mut second_ticket));
        assert!(matches!(
            third_ticket
                .preflight
                .as_mut()
                .expect("third ticket has preflight")
                .try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert_eq!(scheduler.active.len(), 2);
        assert_eq!(scheduler.sessions.len(), 0);
        let clears_before_cancel = cleared_sessions(&state);

        first_ticket.cancel();
        for _ in 0..4 {
            scheduler.drive_once(now)?;
            if scheduler.active.len() == 1 {
                break;
            }
        }
        assert_eq!(scheduler.active.len(), 1);
        assert_eq!(scheduler.sessions.len(), 1);
        let clears_after_cancel = cleared_sessions(&state);
        assert_eq!(clears_after_cancel.len(), clears_before_cancel.len() + 1);
        assert_eq!(clears_after_cancel.last(), Some(&first_session));
        assert!(matches!(
            third_ticket
                .preflight
                .as_mut()
                .expect("third ticket has preflight")
                .try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));

        scheduler.drive_once(now)?;
        assert!(preflight_is_ready(&mut third_ticket));
        assert!(scheduler.active.len() <= 2);
        assert_eq!(scheduler.active.len() + scheduler.sessions.len(), 2);
        for _ in 0..4 {
            scheduler.drive_once(now)?;
            if trace(&state).iter().any(|step| step.label == 4) {
                break;
            }
        }
        let reused = trace(&state)
            .into_iter()
            .find(|step| step.label == 4)
            .expect("third request executes after admission");
        assert_eq!(reused.session_id, first_session);
        Ok(())
    }

    #[test]
    fn backpressured_slot_is_skipped_while_its_peer_finishes() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (jobs, receiver) = mpsc::channel(4);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let mut config = scheduler_config();
        config.prefill_token_quantum = 1;
        let mut scheduler = scheduler(&runtime, receiver, status, config)?;
        let (slow, mut slow_ticket) = job("a", 3, 1, Duration::from_millis(10));
        let (peer, mut peer_ticket) = job("b", 2, 8, Duration::from_millis(10));
        jobs.try_send(slow).expect("slow request fits");
        jobs.try_send(peer).expect("peer request fits");

        let now = Instant::now();
        scheduler.drive_once(now)?;
        scheduler.drive_once(now)?;
        scheduler.drive_once(now)?;
        scheduler.drive_once(now)?;
        scheduler.drive_once(now)?;

        let before = trace(&state)
            .into_iter()
            .filter(|step| step.label == 1)
            .count();
        assert_eq!(before, 2, "slow slot must pause with one pending event");
        let first_peer_event = peer_ticket
            .receiver
            .try_recv()
            .expect("peer emits its first token");
        let second_peer_event = peer_ticket
            .receiver
            .try_recv()
            .expect("peer emits its second token");
        assert!(matches!(
            first_peer_event,
            EngineEvent::Text {
                token: Some(EngineTokenEvidence {
                    token_id: 4,
                    n_decoded: 1,
                    prompt_tokens: 1,
                }),
                ..
            }
        ));
        assert!(matches!(
            second_peer_event,
            EngineEvent::Text {
                token: Some(EngineTokenEvidence {
                    token_id: 4,
                    n_decoded: 2,
                    prompt_tokens: 1,
                }),
                ..
            }
        ));
        assert!(matches!(
            peer_ticket.receiver.try_recv(),
            Ok(EngineEvent::Finished(_))
        ));

        let first_blocked_since = scheduler
            .active
            .front()
            .expect("slow slot remains active")
            .blocked_since
            .expect("slow slot records its first full event queue");
        scheduler.drive_once(first_blocked_since + Duration::from_millis(5))?;
        let after = trace(&state)
            .into_iter()
            .filter(|step| step.label == 1)
            .count();
        assert_eq!(
            after, before,
            "blocked slot must not execute another forward"
        );
        assert_eq!(
            scheduler
                .active
                .front()
                .expect("slow slot remains active")
                .blocked_since,
            Some(first_blocked_since),
            "polling a full queue must not reset its timeout"
        );
        assert!(slow_ticket.receiver.try_recv().is_ok());
        let resumed_at = first_blocked_since + Duration::from_millis(5);
        scheduler.drive_once(resumed_at)?;
        assert!(trace(&state).iter().filter(|step| step.label == 1).count() > before);
        assert_eq!(
            scheduler
                .active
                .front()
                .expect("slow slot remains active")
                .blocked_since,
            Some(resumed_at),
            "a successful send must reset the following event's timeout"
        );
        scheduler.drive_once(resumed_at + Duration::from_millis(9))?;
        assert!(!slow_ticket.cancelled.load(Ordering::Acquire));
        scheduler.drive_once(resumed_at + Duration::from_millis(11))?;
        assert!(slow_ticket.cancelled.load(Ordering::Acquire));
        assert!(scheduler.active.is_empty());
        assert_eq!(scheduler.sessions.len(), 2);
        Ok(())
    }

    #[test]
    fn blocked_middle_slot_is_filtered_without_shifting_packed_rows() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (_jobs, receiver) = mpsc::channel(3);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let config = SchedulerConfig {
            max_active_sequences: 3,
            scheduler_token_budget: 3,
            prefill_token_quantum: 1,
            chat_template_defaults: ChatTemplateOptions::default(),
        };
        let mut scheduler = scheduler(&runtime, receiver, status, config)?;
        let (first, _first_ticket) = job("a", 3, 8, Duration::from_secs(1));
        let (blocked, _blocked_ticket) = job("b", 3, 1, Duration::from_secs(1));
        let (third, _third_ticket) = job("x", 3, 8, Duration::from_secs(1));
        scheduler.admit(first)?;
        scheduler.admit(blocked)?;
        scheduler.admit(third)?;

        let now = Instant::now();
        scheduler.drive_once(now)?;
        scheduler.drive_once(now)?;
        scheduler.drive_once(now)?;

        let invocations = invocations(&state);
        assert!(matches!(
            invocations.as_slice(),
            [
                InvocationTrace::Packed(first),
                InvocationTrace::Packed(second),
                InvocationTrace::Packed(third),
            ] if first.iter().map(|lane| lane.label).collect::<Vec<_>>() == [1, 2, 3]
                && second.iter().map(|lane| lane.label).collect::<Vec<_>>() == [1, 2, 3]
                && third.iter().map(|lane| (lane.label, lane.start_pos, lane.input_ids.as_slice())).collect::<Vec<_>>()
                    == [(1, 2, [3].as_slice()), (3, 2, [3].as_slice())]
        ));
        assert_eq!(
            trace(&state).iter().filter(|lane| lane.label == 2).count(),
            2,
            "the blocked middle slot must not execute in the third iteration"
        );
        assert_eq!(completion_barrier_calls(&state), 3);
        assert_eq!(cleanup_drain_calls(&state), 0);
        Ok(())
    }

    #[test]
    fn cancelled_prefix_slot_is_removed_before_peer_row_mapping() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (_jobs, receiver) = mpsc::channel(3);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let config = SchedulerConfig {
            max_active_sequences: 3,
            scheduler_token_budget: 3,
            prefill_token_quantum: 1,
            chat_template_defaults: ChatTemplateOptions::default(),
        };
        let mut scheduler = scheduler(&runtime, receiver, status, config)?;
        let (cancelled, mut cancelled_ticket) = job("a", 2, 8, Duration::from_secs(1));
        let (second, mut second_ticket) = job("b", 1, 8, Duration::from_secs(1));
        let (third, mut third_ticket) = job("x", 1, 8, Duration::from_secs(1));
        scheduler.admit(cancelled)?;
        scheduler.admit(second)?;
        scheduler.admit(third)?;
        assert!(preflight_is_ready(&mut cancelled_ticket));
        assert!(preflight_is_ready(&mut second_ticket));
        assert!(preflight_is_ready(&mut third_ticket));
        reset_lifecycle(&state);
        cancelled_ticket.cancel();

        scheduler.drive_once(Instant::now())?;

        assert!(cancelled_ticket.receiver.try_recv().is_err());
        let invocations = invocations(&state);
        assert!(matches!(
            invocations.as_slice(),
            [InvocationTrace::Packed(lanes)]
                if lanes.iter().map(|lane| (lane.label, lane.start_pos, lane.input_ids.as_slice())).collect::<Vec<_>>()
                    == [(2, 0, [2].as_slice()), (3, 0, [3].as_slice())]
        ));
        let lifecycle = lifecycle(&state);
        let [
            LifecycleTrace::Clear(cancelled_session),
            LifecycleTrace::PackedForward(peer_sessions),
            ..,
        ] = lifecycle.as_slice()
        else {
            panic!("cancelled prefix must be reclaimed before peer forward: {lifecycle:?}");
        };
        let InvocationTrace::Packed(peer_lanes) = &invocations[0] else {
            unreachable!("the only invocation is packed");
        };
        assert_eq!(
            peer_sessions,
            &peer_lanes
                .iter()
                .map(|lane| lane.session_id)
                .collect::<Vec<_>>()
        );
        assert!(!peer_sessions.contains(cancelled_session));
        let (second_text, _) = collect_success(&mut second_ticket);
        let (third_text, _) = collect_success(&mut third_ticket);
        assert_eq!(second_text.replace(' ', ""), "y");
        assert_eq!(third_text.replace(' ', ""), "x");
        assert_eq!(completion_barrier_calls(&state), 1);
        assert_eq!(cleanup_drain_calls(&state), 0);
        Ok(())
    }

    #[test]
    fn request_local_preflight_failure_does_not_poison_the_worker() -> Result<()> {
        let (runtime, _state) = fake_runtime()?;
        let (jobs, receiver) = mpsc::channel(4);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let mut scheduler = scheduler(&runtime, receiver, Arc::clone(&status), scheduler_config())?;
        let (mut invalid, mut invalid_ticket) = job("a", 1, 8, Duration::from_secs(1));
        invalid.request.sampling.top_p = 0.0;
        let (healthy, mut healthy_ticket) = job("b", 1, 8, Duration::from_secs(1));
        jobs.try_send(invalid).expect("invalid request fits");
        jobs.try_send(healthy).expect("healthy request fits");
        drop(jobs);

        scheduler.run();

        let invalid_error = invalid_ticket
            .preflight
            .as_mut()
            .expect("invalid ticket has preflight")
            .try_recv()
            .expect("invalid request receives preflight")
            .expect_err("invalid sampling must be rejected");
        assert_eq!(invalid_error.kind, EngineFailureKind::InvalidRequest);
        assert!(preflight_is_ready(&mut healthy_ticket));
        assert!(received_finished(&mut healthy_ticket));
        assert_eq!(
            EngineStatus::from_atomic(status.load(Ordering::Acquire)),
            EngineStatus::Stopped
        );
        assert_eq!(scheduler.sessions.len(), 2);
        Ok(())
    }

    #[test]
    fn incomplete_json_schema_request_replaces_its_session_without_poisoning_engine() -> Result<()>
    {
        let (mut runtime, state) = fake_runtime()?;
        // Label `a` emits token 3. Decode it as an object opener so the
        // grammar accepts the sampled token, then rejects the max-token
        // boundary because the JSON value is still incomplete. Label `b`
        // continues to emit the ordinary token 4 for the healthy follow-up.
        let vocab = [
            ("[UNK]".to_owned(), 0),
            ("a".to_owned(), 1),
            ("b".to_owned(), 2),
            ("{".to_owned(), 3),
            ("y".to_owned(), 4),
            ("<eos>".to_owned(), 5),
        ]
        .into_iter()
        .collect();
        let tokenizer_model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_owned())
            .build()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(tokenizer_model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        runtime.tokenizer = Arc::new(tokenizer);

        let (jobs, receiver) = mpsc::channel(2);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let config = SchedulerConfig {
            max_active_sequences: 1,
            scheduler_token_budget: 1,
            prefill_token_quantum: 1,
            chat_template_defaults: ChatTemplateOptions::default(),
        };
        let mut scheduler = scheduler(&runtime, receiver, Arc::clone(&status), config)?;
        let (mut constrained, mut constrained_ticket) = job("a", 1, 8, Duration::from_secs(1));
        constrained.request.input = EngineInput::ConstrainedChat {
            messages: vec![ChatMessage::user("a")],
            constraint: EngineConstraintSpec::response_format(CompiledJsonSchema::compile(
                &json!({
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": false
                }),
            )?),
        };
        let (healthy, mut healthy_ticket) = job("b", 1, 8, Duration::from_secs(1));
        jobs.try_send(constrained)
            .expect("constrained request fits");
        jobs.try_send(healthy).expect("healthy request fits");
        drop(jobs);

        scheduler.run();

        assert!(preflight_is_ready(&mut constrained_ticket));
        let Some(EngineEvent::Failed { message, kind }) =
            constrained_ticket.receiver.try_recv().ok()
        else {
            panic!("incomplete JSON must produce a request-local failure");
        };
        assert_eq!(kind, EngineFailureKind::InvalidRequest);
        assert!(message.contains("response_format.json_schema generation failed"));
        assert!(message.contains("complete schema-valid JSON value"));
        assert!(preflight_is_ready(&mut healthy_ticket));
        assert!(received_finished(&mut healthy_ticket));
        assert_eq!(
            EngineStatus::from_atomic(status.load(Ordering::Acquire)),
            EngineStatus::Stopped
        );
        assert_eq!(scheduler.sessions.len(), 1);
        assert_eq!(state.next_session_id.load(Ordering::Relaxed), 2);
        assert_eq!(cleanup_drain_calls(&state), 0);
        Ok(())
    }

    #[test]
    fn scheduler_panic_is_caught_and_fanned_out_as_fatal() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        *state
            .panic_after_submit_at
            .lock()
            .expect("post-submit panic lock is healthy") = Some((1, 0));
        let (jobs, receiver) = mpsc::channel(4);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let mut scheduler = scheduler(&runtime, receiver, Arc::clone(&status), scheduler_config())?;
        let (panicking, mut panicking_ticket) = job("a a", 2, 8, Duration::from_secs(1));
        let (peer, mut peer_ticket) = job("b b", 2, 8, Duration::from_secs(1));
        jobs.try_send(panicking).expect("panicking request fits");
        jobs.try_send(peer).expect("peer request fits");
        drop(jobs);

        scheduler.run();

        assert!(preflight_is_ready(&mut panicking_ticket));
        assert!(preflight_is_ready(&mut peer_ticket));
        assert!(received_fatal(&mut panicking_ticket));
        assert!(received_fatal(&mut peer_ticket));
        assert!(matches!(
            invocations(&state).as_slice(),
            [InvocationTrace::Packed(lanes)] if lanes.len() == 2
        ));
        assert_eq!(completion_barrier_calls(&state), 0);
        assert_eq!(cleanup_drain_calls(&state), 1);
        assert_eq!(
            EngineStatus::from_atomic(status.load(Ordering::Acquire)),
            EngineStatus::Poisoned
        );
        assert!(scheduler.active.is_empty());
        assert!(scheduler.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn cleanup_panic_keeps_all_slots_for_fatal_fanout_before_peer_flush() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        let (_jobs, receiver) = mpsc::channel(2);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let mut scheduler = scheduler(&runtime, receiver, Arc::clone(&status), scheduler_config())?;
        let (cancelled, mut cancelled_ticket) = job("a", 1, 8, Duration::from_secs(1));
        let (peer, mut peer_ticket) = job("b", 1, 8, Duration::from_secs(1));
        scheduler.admit(cancelled)?;
        scheduler.admit(peer)?;
        assert!(preflight_is_ready(&mut cancelled_ticket));
        assert!(preflight_is_ready(&mut peer_ticket));
        reset_lifecycle(&state);
        cancelled_ticket.cancel();
        *state
            .panic_clear_session
            .lock()
            .expect("clear panic lock is healthy") = Some(1);

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = scheduler.drive_once(Instant::now());
        }));
        assert!(panic.is_err());
        assert_eq!(scheduler.active.len(), 2);
        assert!(scheduler.active[0].sequence.is_none());
        assert!(scheduler.active[0].reclaimed_session.is_none());
        assert!(scheduler.active[1].sequence.is_some());
        assert!(invocations(&state).is_empty());
        assert!(lifecycle(&state).is_empty());
        assert_eq!(completion_barrier_calls(&state), 0);
        assert_eq!(cleanup_drain_calls(&state), 0);
        assert!(cancelled_ticket.receiver.try_recv().is_err());
        assert!(peer_ticket.receiver.try_recv().is_err());

        scheduler.poison("cleanup panicked".to_owned());
        assert!(received_fatal(&mut cancelled_ticket));
        assert!(received_fatal(&mut peer_ticket));
        assert_eq!(
            EngineStatus::from_atomic(status.load(Ordering::Acquire)),
            EngineStatus::Poisoned
        );
        assert!(scheduler.active.is_empty());
        assert!(scheduler.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn shared_completion_failure_poison_is_atomic_and_not_retried() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        *state
            .fail_completion_barrier_at
            .lock()
            .expect("completion failure lock is healthy") = Some(1);
        let (jobs, receiver) = mpsc::channel(2);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let mut scheduler = scheduler(&runtime, receiver, Arc::clone(&status), scheduler_config())?;
        let (first, mut first_ticket) = job("a", 1, 8, Duration::from_secs(1));
        let (second, mut second_ticket) = job("b", 1, 8, Duration::from_secs(1));
        let (queued, mut queued_ticket) = job("a b", 1, 8, Duration::from_secs(1));
        scheduler.admit(first)?;
        scheduler.admit(second)?;
        jobs.try_send(queued).expect("queued request fits");
        assert!(preflight_is_ready(&mut first_ticket));
        assert!(preflight_is_ready(&mut second_ticket));

        let error = scheduler
            .drive_once(Instant::now())
            .expect_err("shared completion barrier must fail");
        assert!(
            error
                .to_string()
                .contains("injected scheduler completion barrier failure")
        );
        assert!(first_ticket.receiver.try_recv().is_err());
        assert!(second_ticket.receiver.try_recv().is_err());
        scheduler.poison(format!("inference worker failed: {error:#}"));

        assert_eq!(completion_barrier_calls(&state), 1);
        assert_eq!(cleanup_drain_calls(&state), 0);
        assert!(matches!(
            invocations(&state).as_slice(),
            [InvocationTrace::Packed(lanes)] if lanes.len() == 2
        ));
        assert!(received_fatal(&mut first_ticket));
        assert!(received_fatal(&mut second_ticket));
        let queued_error = queued_ticket
            .preflight
            .as_mut()
            .expect("queued ticket has preflight")
            .try_recv()
            .expect("queued request receives fatal preflight")
            .expect_err("queued request must fail");
        assert_eq!(queued_error.kind, EngineFailureKind::Fatal);
        assert_eq!(
            EngineStatus::from_atomic(status.load(Ordering::Acquire)),
            EngineStatus::Poisoned
        );
        assert!(scheduler.active.is_empty());
        assert!(scheduler.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn mixed_shape_and_cache_barriers_prevent_partial_row_commit() -> Result<()> {
        for corrupt_cache in [false, true] {
            let (runtime, state) = fake_runtime()?;
            if corrupt_cache {
                *state
                    .corrupt_cache_label
                    .lock()
                    .expect("cache corruption lock is healthy") = Some(2);
            } else {
                state.bad_batch_shape.store(true, Ordering::Relaxed);
            }
            let (_jobs, receiver) = mpsc::channel(1);
            let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
            let mut scheduler =
                scheduler(&runtime, receiver, Arc::clone(&status), scheduler_config())?;
            let (first, mut first_ticket) = job("a", 1, 8, Duration::from_secs(1));
            let (second, mut second_ticket) = job("b", 1, 8, Duration::from_secs(1));
            scheduler.admit(first)?;
            scheduler.admit(second)?;
            assert!(preflight_is_ready(&mut first_ticket));
            assert!(preflight_is_ready(&mut second_ticket));

            let error = scheduler
                .drive_once(Instant::now())
                .expect_err("mixed completion barrier must reject the injected violation");
            if corrupt_cache {
                assert!(
                    error
                        .to_string()
                        .contains("model advanced generation cache")
                );
            } else {
                assert!(
                    error
                        .to_string()
                        .contains("mixed generation logits have shape")
                );
            }
            assert!(first_ticket.receiver.try_recv().is_err());
            assert!(second_ticket.receiver.try_recv().is_err());
            scheduler.poison(format!("inference worker failed: {error:#}"));

            assert_eq!(completion_barrier_calls(&state), 0);
            assert_eq!(cleanup_drain_calls(&state), 1);
            assert!(received_fatal(&mut first_ticket));
            assert!(received_fatal(&mut second_ticket));
            assert_eq!(
                EngineStatus::from_atomic(status.load(Ordering::Acquire)),
                EngineStatus::Poisoned
            );
            assert!(scheduler.active.is_empty());
            assert!(scheduler.sessions.is_empty());
        }
        Ok(())
    }

    #[test]
    fn fatal_model_error_poisons_active_and_queued_requests() -> Result<()> {
        let (runtime, state) = fake_runtime()?;
        *state
            .fail_after_submit_at
            .lock()
            .expect("post-submit failure lock is healthy") = Some((1, 0));
        let (jobs, receiver) = mpsc::channel(4);
        let status = Arc::new(AtomicU8::new(EngineStatus::Ready as u8));
        let mut scheduler = scheduler(&runtime, receiver, Arc::clone(&status), scheduler_config())?;
        let (failing, mut failing_ticket) = job("a a", 2, 8, Duration::from_secs(1));
        let (peer, mut peer_ticket) = job("b b", 2, 8, Duration::from_secs(1));
        let (queued, mut queued_ticket) = job("a b", 1, 8, Duration::from_secs(1));
        scheduler.admit(failing)?;
        scheduler.admit(peer)?;
        jobs.try_send(queued).expect("queued request fits");
        assert!(preflight_is_ready(&mut failing_ticket));
        assert!(preflight_is_ready(&mut peer_ticket));

        let now = Instant::now();
        let error = scheduler
            .drive_once(now)
            .expect_err("a post-submission forward failure is fatal");
        assert!(
            error
                .to_string()
                .contains("injected scheduler model failure after submission")
        );
        assert!(failing_ticket.receiver.try_recv().is_err());
        assert!(peer_ticket.receiver.try_recv().is_err());
        assert!(matches!(
            invocations(&state).as_slice(),
            [InvocationTrace::Packed(lanes)]
                if lanes.iter().map(|lane| (lane.label, lane.start_pos, lane.input_ids.as_slice())).collect::<Vec<_>>()
                    == [(1, 0, [1, 1].as_slice()), (2, 0, [2, 2].as_slice())]
        ));
        assert_eq!(completion_barrier_calls(&state), 0);
        scheduler.poison(format!("inference worker failed: {error:#}"));
        assert_eq!(cleanup_drain_calls(&state), 1);

        assert_eq!(
            EngineStatus::from_atomic(status.load(Ordering::Acquire)),
            EngineStatus::Poisoned
        );
        assert!(received_fatal(&mut failing_ticket));
        assert!(received_fatal(&mut peer_ticket));
        let queued_error = queued_ticket
            .preflight
            .as_mut()
            .expect("queued ticket has preflight")
            .try_recv()
            .expect("queued request receives fatal preflight")
            .expect_err("queued request must fail");
        assert_eq!(queued_error.kind, EngineFailureKind::Fatal);
        assert!(scheduler.active.is_empty());
        assert!(scheduler.sessions.is_empty());
        Ok(())
    }
}
