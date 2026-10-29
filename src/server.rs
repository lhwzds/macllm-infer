use std::{
    collections::VecDeque,
    convert::Infallible,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures_util::stream::{self, Stream};
use serde::{Deserialize, Serialize};

use crate::{
    agent::{
        AgentDirective, AgentMessage as RuntimeAgentMessage, AgentPrompt,
        ToolCall as RuntimeToolCall, ToolDefinition as RuntimeToolDefinition,
    },
    chat::ChatRole,
    engine::{
        EngineConfig, EngineConstraintSpec, EngineEvent, EngineFailureKind, EngineHandle,
        EngineInput, EngineMetadata, EngineRequest, EngineRequestPolicy, EngineStatus,
        EngineTicket, SubmitError,
    },
    generation::{FinishReason, GenerationMetrics},
    grammar::{CompiledJsonSchema, CompiledToolCalls},
    openai::{
        AgentMessage as OpenAiAgentMessage, ApiError, AssistantFunctionCall, AssistantToolCall,
        ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatFinishReason,
        FunctionTool, FunctionToolKind, TokenUsage, ToolChoice, ToolChoiceMode,
        ValidatedChatCompletionRequest, ValidatedMessages,
    },
    sampling::SamplingConfig,
    tool_call::{
        SplitEnd, TOOL_CALL_OPEN, ToolCallScan, ToolCallSplitter, parse_tool_call, scan_tool_call,
    },
};

static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);
pub const RAW_COMPLETION_RESPONSE_FIELDS: [&str; 22] = [
    "index",
    "content",
    "stop",
    "tokens_predicted",
    "tokens_evaluated",
    "truncated",
    "stop_type",
    "generation_settings/seed",
    "generation_settings/temperature",
    "generation_settings/top_k",
    "generation_settings/top_p",
    "generation_settings/repeat_penalty",
    "generation_settings/ignore_eos",
    "generation_settings/stream",
    "generation_settings/max_tokens",
    "generation_settings/n_predict",
    "generation_settings/n_keep",
    "timings/cache_n",
    "timings/prompt_n",
    "timings/prompt_ms",
    "timings/predicted_n",
    "timings/predicted_ms",
];

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind: SocketAddr,
    pub engine: EngineConfig,
    pub options: ServerOptions,
}

/// Protocol defaults applied by the HTTP layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerOptions {
    /// Completion budget for Chat Completions requests that set neither
    /// `max_completion_tokens` nor `max_tokens`. It must be positive and fit
    /// the model context; the server checks this at startup.
    pub default_max_completion_tokens: usize,
    /// Sampling values for fields a request omits.
    pub sampling_profile: SamplingProfile,
}

impl ServerOptions {
    pub const DEFAULT_MAX_COMPLETION_TOKENS: usize = 128;
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            default_max_completion_tokens: Self::DEFAULT_MAX_COMPLETION_TOKENS,
            sampling_profile: SamplingProfile::default(),
        }
    }
}

/// Sampling values used for fields a request omits. Request values always
/// win. `top_k` is not a Chat Completions field, so it always comes from
/// here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
}

/// Named server sampling defaults.
///
/// A profile rather than per-field flags: the Qwen3 recommendation depends
/// on each request's effective thinking mode (its `chat_template_kwargs`, or
/// the server's `--disable-thinking`), which one fixed value per field
/// cannot express.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SamplingProfile {
    /// The `OpenAI` defaults: temperature 1, top_p 1, no top-k.
    #[default]
    OpenAi,
    /// The Qwen3 model card's recommendation: temperature 0.6, top_p 0.95,
    /// top_k 20 when thinking; 0.7, 0.8, 20 when not.
    Qwen3,
}

impl SamplingProfile {
    #[must_use]
    pub const fn defaults(self, thinking: bool) -> SamplingDefaults {
        match (self, thinking) {
            (Self::OpenAi, _) => SamplingDefaults {
                temperature: crate::openai::DEFAULT_TEMPERATURE,
                top_p: crate::openai::DEFAULT_TOP_P,
                top_k: 0,
            },
            (Self::Qwen3, true) => SamplingDefaults {
                temperature: 0.6,
                top_p: 0.95,
                top_k: 20,
            },
            (Self::Qwen3, false) => SamplingDefaults {
                temperature: 0.7,
                top_p: 0.8,
                top_k: 20,
            },
        }
    }
}

/// Minimal interface used by the HTTP layer. Production uses one
/// [`EngineHandle`]; the trait also makes the wire contract independently
/// testable without loading a checkpoint.
pub trait InferenceEngine: Send + Sync {
    fn metadata(&self) -> EngineMetadata;
    fn status(&self) -> EngineStatus;
    fn submit(&self, request: EngineRequest) -> std::result::Result<EngineTicket, SubmitError>;

    fn submit_with_policy(
        &self,
        request: EngineRequest,
        _policy: EngineRequestPolicy,
    ) -> std::result::Result<EngineTicket, SubmitError> {
        self.submit(request)
    }
}

impl InferenceEngine for EngineHandle {
    fn metadata(&self) -> EngineMetadata {
        EngineHandle::metadata(self).clone()
    }

    fn status(&self) -> EngineStatus {
        EngineHandle::status(self)
    }

    fn submit(&self, request: EngineRequest) -> std::result::Result<EngineTicket, SubmitError> {
        EngineHandle::submit(self, request)
    }

    fn submit_with_policy(
        &self,
        request: EngineRequest,
        policy: EngineRequestPolicy,
    ) -> std::result::Result<EngineTicket, SubmitError> {
        EngineHandle::submit_with_policy(self, request, policy)
    }
}

#[derive(Clone)]
struct AppState {
    engine: Arc<dyn InferenceEngine>,
    options: ServerOptions,
}

/// Build the strict Chat Completions and raw benchmark router around one engine.
pub fn router(engine: Arc<dyn InferenceEngine>) -> Router {
    router_with_options(engine, ServerOptions::default())
}

/// Build the router with explicit protocol defaults.
pub fn router_with_options(engine: Arc<dyn InferenceEngine>, options: ServerOptions) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/completion", post(raw_completion))
        .with_state(AppState { engine, options })
}

/// Load one model, bind the listener and serve until Ctrl-C.
pub async fn serve(config: ServerConfig) -> Result<()> {
    let engine = EngineHandle::start(config.engine)?;
    let max_context = engine.metadata().max_context;
    let default_max_completion_tokens = config.options.default_max_completion_tokens;
    anyhow::ensure!(
        (1..=max_context).contains(&default_max_completion_tokens),
        "default max completion tokens must be in 1..={max_context} for this model, got {default_max_completion_tokens}"
    );
    let listener = tokio::net::TcpListener::bind(config.bind)
        .await
        .with_context(|| format!("failed to bind inference server to {}", config.bind))?;
    let address = listener.local_addr()?;
    eprintln!(
        "serving model `{}` at http://{address}",
        engine.metadata().served_model
    );
    axum::serve(
        listener,
        router_with_options(Arc::new(engine), config.options),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .context("inference HTTP server failed")
}

async fn shutdown_signal() {
    if tokio::signal::ctrl_c().await.is_err() {
        std::future::pending::<()>().await;
    }
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    worker: &'static str,
    model: String,
    device: String,
    dtype: String,
}

async fn health(State(state): State<AppState>) -> Response {
    let status = state.engine.status();
    let metadata = state.engine.metadata();
    let body = HealthResponse {
        status: if status.is_healthy() { "ok" } else { "error" },
        worker: status_label(status),
        model: metadata.served_model,
        device: metadata.device,
        dtype: metadata.dtype,
    };
    let http_status = if status.is_healthy() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (http_status, Json(body)).into_response()
}

const fn status_label(status: EngineStatus) -> &'static str {
    match status {
        EngineStatus::Starting => "starting",
        EngineStatus::Ready => "ready",
        EngineStatus::Busy => "busy",
        EngineStatus::Poisoned => "poisoned",
        EngineStatus::Stopped => "stopped",
    }
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

async fn models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelObject {
            id: state.engine.metadata().served_model,
            object: "model",
            created: 0,
            owned_by: "macllm-infer",
        }],
    })
}

/// Shared, fail-closed subset accepted by both benchmark servers.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCompletionRequest {
    prompt: String,
    n_predict: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
    cache_prompt: bool,
    stream: bool,
    return_tokens: bool,
    id_slot: usize,
    ignore_eos: bool,
    repeat_penalty: f64,
    response_fields: Vec<String>,
}

impl RawCompletionRequest {
    fn validate(
        self,
        max_context: usize,
    ) -> std::result::Result<(EngineRequest, RawGenerationSettings, EngineRequestPolicy), ApiError>
    {
        if self.prompt.is_empty() {
            return Err(ApiError::invalid_request(
                "prompt must not be empty",
                Some("prompt".to_owned()),
                Some("invalid_value".to_owned()),
            ));
        }
        if self.n_predict == 0 || self.n_predict > max_context {
            return Err(ApiError::invalid_request(
                format!("n_predict must be in 1..={max_context}"),
                Some("n_predict".to_owned()),
                Some("invalid_value".to_owned()),
            ));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(ApiError::invalid_request(
                "temperature must be finite and non-negative",
                Some("temperature".to_owned()),
                Some("invalid_value".to_owned()),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(ApiError::invalid_request(
                "top_p must be finite and in (0, 1]",
                Some("top_p".to_owned()),
                Some("invalid_value".to_owned()),
            ));
        }
        if self.id_slot >= 8 {
            return Err(ApiError::invalid_request(
                "id_slot must be in 0..8",
                Some("id_slot".to_owned()),
                Some("invalid_value".to_owned()),
            ));
        }
        if !self.ignore_eos {
            return Err(ApiError::invalid_request(
                "ignore_eos must be true for fixed-length benchmark completions",
                Some("ignore_eos".to_owned()),
                Some("unsupported_value".to_owned()),
            ));
        }
        if !self.return_tokens {
            return Err(ApiError::invalid_request(
                "return_tokens must be true for benchmark token-parity evidence",
                Some("return_tokens".to_owned()),
                Some("unsupported_value".to_owned()),
            ));
        }
        if self.repeat_penalty.to_bits() != 1.0_f64.to_bits() {
            return Err(ApiError::invalid_request(
                "repeat_penalty must be 1.0 for the shared benchmark contract",
                Some("repeat_penalty".to_owned()),
                Some("unsupported_value".to_owned()),
            ));
        }
        if !self
            .response_fields
            .iter()
            .map(String::as_str)
            .eq(RAW_COMPLETION_RESPONSE_FIELDS)
        {
            return Err(ApiError::invalid_request(
                "response_fields must exactly match the benchmark response projection",
                Some("response_fields".to_owned()),
                Some("unsupported_value".to_owned()),
            ));
        }

        let settings = RawGenerationSettings {
            seed: self.seed,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repeat_penalty: 1.0,
            ignore_eos: true,
            stream: self.stream,
            max_tokens: self.n_predict,
            n_predict: self.n_predict,
            n_keep: 0,
        };
        let policy = if self.cache_prompt {
            EngineRequestPolicy::default()
        } else {
            EngineRequestPolicy::benchmark_measured()
        };
        let request = EngineRequest {
            input: EngineInput::RawPrompt(self.prompt),
            sampling: SamplingConfig {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                repetition_penalty: 1.0,
                seed: self.seed,
            },
            max_new_tokens: self.n_predict,
            ignore_stop_tokens: true,
            stop: Vec::new(),
            chat_template: crate::chat::ChatTemplateOptions::default(),
        };
        Ok((request, settings, policy))
    }
}

#[derive(Debug, Clone, Serialize)]
struct RawGenerationSettings {
    seed: u64,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repeat_penalty: f64,
    ignore_eos: bool,
    stream: bool,
    max_tokens: usize,
    n_predict: usize,
    n_keep: usize,
}

#[derive(Debug, Serialize)]
struct RawCompletionEvent {
    index: usize,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens: Option<Vec<u32>>,
    stop: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    id_slot: Option<isize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_predicted: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_evaluated: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncated: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_type: Option<&'static str>,
    #[serde(
        rename = "generation_settings/seed",
        skip_serializing_if = "Option::is_none"
    )]
    generation_seed: Option<u64>,
    #[serde(
        rename = "generation_settings/temperature",
        skip_serializing_if = "Option::is_none"
    )]
    generation_temperature: Option<f64>,
    #[serde(
        rename = "generation_settings/top_k",
        skip_serializing_if = "Option::is_none"
    )]
    generation_top_k: Option<usize>,
    #[serde(
        rename = "generation_settings/top_p",
        skip_serializing_if = "Option::is_none"
    )]
    generation_top_p: Option<f64>,
    #[serde(
        rename = "generation_settings/repeat_penalty",
        skip_serializing_if = "Option::is_none"
    )]
    generation_repeat_penalty: Option<f64>,
    #[serde(
        rename = "generation_settings/ignore_eos",
        skip_serializing_if = "Option::is_none"
    )]
    generation_ignore_eos: Option<bool>,
    #[serde(
        rename = "generation_settings/stream",
        skip_serializing_if = "Option::is_none"
    )]
    generation_stream: Option<bool>,
    #[serde(
        rename = "generation_settings/max_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    generation_max_tokens: Option<usize>,
    #[serde(
        rename = "generation_settings/n_predict",
        skip_serializing_if = "Option::is_none"
    )]
    generation_n_predict: Option<usize>,
    #[serde(
        rename = "generation_settings/n_keep",
        skip_serializing_if = "Option::is_none"
    )]
    generation_n_keep: Option<usize>,
    #[serde(rename = "timings/cache_n", skip_serializing_if = "Option::is_none")]
    timing_cache_n: Option<usize>,
    #[serde(rename = "timings/prompt_n", skip_serializing_if = "Option::is_none")]
    timing_prompt_n: Option<usize>,
    #[serde(rename = "timings/prompt_ms", skip_serializing_if = "Option::is_none")]
    timing_prompt_ms: Option<f64>,
    #[serde(
        rename = "timings/predicted_n",
        skip_serializing_if = "Option::is_none"
    )]
    timing_predicted_n: Option<usize>,
    #[serde(
        rename = "timings/predicted_ms",
        skip_serializing_if = "Option::is_none"
    )]
    timing_predicted_ms: Option<f64>,
}

impl RawCompletionEvent {
    fn fragment(content: String, token_id: u32, n_decoded: usize, prompt_tokens: usize) -> Self {
        Self {
            index: 0,
            content,
            tokens: Some(vec![token_id]),
            stop: false,
            id_slot: Some(-1),
            tokens_predicted: Some(n_decoded),
            tokens_evaluated: Some(prompt_tokens),
            truncated: None,
            stop_type: None,
            generation_seed: None,
            generation_temperature: None,
            generation_top_k: None,
            generation_top_p: None,
            generation_repeat_penalty: None,
            generation_ignore_eos: None,
            generation_stream: None,
            generation_max_tokens: None,
            generation_n_predict: None,
            generation_n_keep: None,
            timing_cache_n: None,
            timing_prompt_n: None,
            timing_prompt_ms: None,
            timing_predicted_n: None,
            timing_predicted_ms: None,
        }
    }

    fn terminal(
        content: String,
        settings: RawGenerationSettings,
        metrics: &GenerationMetrics,
    ) -> Self {
        let stop_type = match metrics.finish_reason {
            FinishReason::Length => "limit",
            FinishReason::Stop => "stop",
            FinishReason::Cancelled => "cancelled",
        };
        Self {
            index: 0,
            content,
            tokens: None,
            stop: true,
            id_slot: None,
            tokens_predicted: Some(metrics.generated_tokens),
            tokens_evaluated: Some(metrics.prompt_tokens),
            truncated: Some(false),
            stop_type: Some(stop_type),
            generation_seed: Some(settings.seed),
            generation_temperature: Some(settings.temperature),
            generation_top_k: Some(settings.top_k),
            generation_top_p: Some(settings.top_p),
            generation_repeat_penalty: Some(settings.repeat_penalty),
            generation_ignore_eos: Some(settings.ignore_eos),
            generation_stream: Some(settings.stream),
            generation_max_tokens: Some(settings.max_tokens),
            generation_n_predict: Some(settings.n_predict),
            generation_n_keep: Some(settings.n_keep),
            timing_cache_n: Some(metrics.cached_prompt_tokens),
            timing_prompt_n: Some(metrics.processed_prompt_tokens()),
            timing_prompt_ms: Some(duration_millis(metrics.prefill_time)),
            timing_predicted_n: Some(metrics.generated_tokens),
            timing_predicted_ms: Some(duration_millis(
                metrics
                    .total_time
                    .saturating_sub(metrics.time_to_first_token),
            )),
        }
    }
}

fn duration_millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

async fn raw_completion(State(state): State<AppState>, body: Bytes) -> Response {
    let request = match serde_json::from_slice::<RawCompletionRequest>(&body) {
        Ok(request) => request,
        Err(error) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                ApiError::invalid_request(
                    format!("invalid /completion request: {error}"),
                    None,
                    Some("invalid_json".to_owned()),
                ),
            );
        }
    };
    let stream_response = request.stream;
    let (engine_request, settings, policy) =
        match request.validate(state.engine.metadata().max_context) {
            Ok(request) => request,
            Err(error) => return api_error(status_for_api_error(&error), error),
        };
    let mut ticket = match state.engine.submit_with_policy(engine_request, policy) {
        Ok(ticket) => ticket,
        Err(error) => return submit_error(error),
    };
    if let Err(error) = ticket.wait_for_preflight().await {
        return engine_failure_response(error.kind, error.message);
    }

    if stream_response {
        raw_streaming_response(ticket, settings).into_response()
    } else {
        raw_non_streaming_response(ticket, settings).await
    }
}

async fn raw_non_streaming_response(
    mut ticket: EngineTicket,
    settings: RawGenerationSettings,
) -> Response {
    let mut content = String::new();
    while let Some(event) = ticket.recv().await {
        match event {
            EngineEvent::Text {
                content: fragment, ..
            } => content.push_str(&fragment),
            EngineEvent::Finished(metrics) => {
                if let Err(error) = validate_raw_completion_metrics(&settings, &metrics) {
                    return api_error(StatusCode::INTERNAL_SERVER_ERROR, error);
                }
                return (
                    StatusCode::OK,
                    Json(RawCompletionEvent::terminal(content, settings, &metrics)),
                )
                    .into_response();
            }
            EngineEvent::Failed { message, kind } => {
                return engine_failure_response(kind, message);
            }
        }
    }
    api_error(
        StatusCode::SERVICE_UNAVAILABLE,
        ApiError::server_error("inference worker stopped before completing the request"),
    )
}

fn raw_streaming_response(
    ticket: EngineTicket,
    settings: RawGenerationSettings,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let state = RawStreamState {
        ticket,
        settings,
        terminal: false,
        last_n_decoded: 0,
        prompt_tokens: None,
    };
    Sse::new(stream::unfold(state, next_raw_stream_event))
}

struct RawStreamState {
    ticket: EngineTicket,
    settings: RawGenerationSettings,
    terminal: bool,
    last_n_decoded: usize,
    prompt_tokens: Option<usize>,
}

async fn next_raw_stream_event(
    mut state: RawStreamState,
) -> Option<(std::result::Result<Event, Infallible>, RawStreamState)> {
    if state.terminal {
        return None;
    }
    let event = match state.ticket.recv().await {
        Some(EngineEvent::Text {
            content,
            token:
                Some(crate::engine::EngineTokenEvidence {
                    token_id,
                    n_decoded,
                    prompt_tokens,
                }),
        }) => {
            let valid_ordinal =
                n_decoded > state.last_n_decoded && n_decoded <= state.settings.n_predict;
            let valid_prompt = state
                .prompt_tokens
                .is_none_or(|expected| expected == prompt_tokens);
            if !valid_ordinal || !valid_prompt {
                state.terminal = true;
                json_event(
                    &ApiError::server_error(
                        "inference worker emitted inconsistent raw token metadata",
                    )
                    .into_response(),
                )
            } else {
                state.last_n_decoded = n_decoded;
                state.prompt_tokens = Some(prompt_tokens);
                json_event(&RawCompletionEvent::fragment(
                    content,
                    token_id,
                    n_decoded,
                    prompt_tokens,
                ))
            }
        }
        Some(EngineEvent::Text { token: None, .. }) => {
            state.terminal = true;
            json_event(
                &ApiError::server_error(
                    "inference worker emitted raw text without sampled-token evidence",
                )
                .into_response(),
            )
        }
        Some(EngineEvent::Finished(metrics)) => {
            state.terminal = true;
            match validate_raw_completion_metrics(&state.settings, &metrics).and_then(|()| {
                if state.last_n_decoded == metrics.generated_tokens
                    && state.prompt_tokens == Some(metrics.prompt_tokens)
                {
                    Ok(())
                } else {
                    Err(ApiError::server_error(
                        "inference worker raw token metadata did not reconcile with terminal metrics",
                    ))
                }
            }) {
                Ok(()) => json_event(&RawCompletionEvent::terminal(
                    String::new(),
                    state.settings.clone(),
                    &metrics,
                )),
                Err(error) => json_event(&error.into_response()),
            }
        }
        Some(EngineEvent::Failed { message, kind }) => {
            state.terminal = true;
            json_event(&engine_failure_error(kind, message).into_response())
        }
        None => {
            state.terminal = true;
            json_event(
                &ApiError::server_error("inference worker stopped before completing the request")
                    .into_response(),
            )
        }
    };
    Some((Ok(event), state))
}

fn validate_raw_completion_metrics(
    settings: &RawGenerationSettings,
    metrics: &GenerationMetrics,
) -> std::result::Result<(), ApiError> {
    if metrics.finish_reason != FinishReason::Length
        || metrics.generated_tokens != settings.n_predict
    {
        return Err(ApiError::server_error(format!(
            "raw completion ended after {} tokens with {:?}; expected the fixed {}-token limit",
            metrics.generated_tokens, metrics.finish_reason, settings.n_predict
        )));
    }
    Ok(())
}

fn engine_failure_error(kind: EngineFailureKind, message: String) -> ApiError {
    match kind {
        EngineFailureKind::InvalidRequest => {
            ApiError::invalid_request(message, None, Some("invalid_request".to_owned()))
        }
        EngineFailureKind::ResourceExhausted
        | EngineFailureKind::Internal
        | EngineFailureKind::Fatal => ApiError::server_error(message),
    }
}

fn engine_failure_response(kind: EngineFailureKind, message: String) -> Response {
    let status = match kind {
        EngineFailureKind::InvalidRequest => StatusCode::BAD_REQUEST,
        EngineFailureKind::ResourceExhausted => StatusCode::SERVICE_UNAVAILABLE,
        EngineFailureKind::Internal | EngineFailureKind::Fatal => StatusCode::INTERNAL_SERVER_ERROR,
    };
    api_error(status, engine_failure_error(kind, message))
}

async fn chat_completions(State(state): State<AppState>, body: Bytes) -> Response {
    let request = match ChatCompletionRequest::from_json(&body) {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error),
    };
    let metadata = state.engine.metadata();
    let request = match request.validate(
        &metadata.served_model,
        state.options.default_max_completion_tokens,
    ) {
        Ok(request) => request,
        Err(error) => return api_error(status_for_api_error(&error), error),
    };
    if request.max_completion_tokens > metadata.max_context {
        return api_error(
            StatusCode::BAD_REQUEST,
            ApiError::invalid_request(
                format!(
                    "max_completion_tokens exceeds model context limit ({})",
                    metadata.max_context
                ),
                Some("max_completion_tokens".to_owned()),
                Some("invalid_value".to_owned()),
            ),
        );
    }

    let stream_response = request.stream;
    let include_usage = request.include_usage;
    let identity = RequestIdentity::new(request.model.clone());
    // Templates think unless a request or the server disables it.
    let thinking = request
        .chat_template
        .or(metadata.chat_template_defaults)
        .enable_thinking
        != Some(false);
    let sampling = state.options.sampling_profile.defaults(thinking);
    let (engine_request, output) = match to_engine_request(request, identity.default_seed, sampling)
    {
        Ok(lowered) => lowered,
        Err(error) => return api_error(status_for_api_error(&error), error),
    };
    let mut ticket = match state.engine.submit(engine_request) {
        Ok(ticket) => ticket,
        Err(error) => return submit_error(error),
    };
    if let Err(error) = ticket.wait_for_preflight().await {
        let status = match error.kind {
            EngineFailureKind::InvalidRequest => StatusCode::BAD_REQUEST,
            EngineFailureKind::ResourceExhausted => StatusCode::SERVICE_UNAVAILABLE,
            EngineFailureKind::Internal | EngineFailureKind::Fatal => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        let error = match error.kind {
            EngineFailureKind::InvalidRequest => {
                ApiError::invalid_request(error.message, None, Some("invalid_request".to_owned()))
            }
            EngineFailureKind::ResourceExhausted
            | EngineFailureKind::Internal
            | EngineFailureKind::Fatal => ApiError::server_error(error.message),
        };
        return api_error(status, error);
    }

    if stream_response {
        streaming_response(ticket, identity, include_usage, output).into_response()
    } else {
        non_streaming_response(ticket, identity, output).await
    }
}

/// How the HTTP layer turns generated text into one assistant message.
/// It is fixed while lowering the request, so the response parser always
/// matches what the engine was asked to generate.
enum AssistantOutput {
    /// Generated text is the message content.
    Text,
    /// The prompt opened one named call; generated text is its arguments.
    ForcedCall { name: String },
    /// Generated text is content that may end in one native call. The lazy
    /// grammar constrained the call; it is parsed and re-validated against
    /// these declarations before any part of it is exposed.
    AutoCall { calls: CompiledToolCalls },
}

fn to_engine_request(
    request: ValidatedChatCompletionRequest,
    request_default_seed: u64,
    defaults: SamplingDefaults,
) -> std::result::Result<(EngineRequest, AssistantOutput), ApiError> {
    let sampling = SamplingConfig {
        temperature: request.temperature.unwrap_or(defaults.temperature),
        top_k: defaults.top_k,
        top_p: request.top_p.unwrap_or(defaults.top_p),
        repetition_penalty: 1.0,
        seed: request.seed.unwrap_or(request_default_seed),
    };
    let max_new_tokens = request.max_completion_tokens;
    let stop = request.stop.clone();
    let chat_template = request.chat_template;
    let response_constraint = request
        .response_format
        .as_ref()
        .map(|format| {
            format
                .compile_json_schema()
                .map(EngineConstraintSpec::response_format)
        })
        .transpose()?;
    let forced_tool = request.forced_tool().cloned();
    let automatic = matches!(request.tool_choice, ToolChoice::Mode(ToolChoiceMode::Auto));
    let (input, output) = if let Some(tool) = forced_tool {
        let schema = compile_tool_arguments(&tool)?;
        let name = tool.function.name;
        let prompt = AgentPrompt {
            messages: lower_agent_messages(request.messages)?,
            tools: request
                .tools
                .into_iter()
                .filter(|candidate| candidate.function.name == name)
                .map(runtime_tool_definition)
                .collect(),
            directive: AgentDirective::ForcedSingle { name: name.clone() },
        };
        (
            EngineInput::Agent {
                prompt,
                constraint: Some(EngineConstraintSpec::tool_arguments(schema)),
            },
            AssistantOutput::ForcedCall { name },
        )
    } else if automatic {
        let calls = CompiledToolCalls::new(
            request
                .tools
                .iter()
                .map(|tool| Ok((tool.function.name.clone(), compile_tool_arguments(tool)?)))
                .collect::<std::result::Result<Vec<_>, ApiError>>()?,
        )
        .map_err(|error| {
            ApiError::invalid_request(
                format!("invalid native tool declarations: {error:#}"),
                Some("tools".to_owned()),
                Some("invalid_value".to_owned()),
            )
        })?;
        let prompt = AgentPrompt {
            messages: lower_agent_messages(request.messages)?,
            tools: request
                .tools
                .into_iter()
                .map(runtime_tool_definition)
                .collect(),
            directive: AgentDirective::Auto,
        };
        (
            EngineInput::Agent {
                prompt,
                constraint: Some(EngineConstraintSpec::tool_call(calls.clone())),
            },
            AssistantOutput::AutoCall { calls },
        )
    } else {
        let input = match request.messages {
            ValidatedMessages::Text(messages) => match response_constraint {
                Some(constraint) => EngineInput::ConstrainedChat {
                    messages,
                    constraint,
                },
                None => EngineInput::Chat(messages),
            },
            messages @ ValidatedMessages::ToolAware(_) => EngineInput::Agent {
                prompt: AgentPrompt {
                    messages: lower_agent_messages(messages)?,
                    tools: Vec::new(),
                    directive: AgentDirective::TextOnly,
                },
                constraint: None,
            },
        };
        (input, AssistantOutput::Text)
    };
    Ok((
        EngineRequest {
            input,
            sampling,
            max_new_tokens,
            ignore_stop_tokens: false,
            stop,
            chat_template,
        },
        output,
    ))
}

fn compile_tool_arguments(
    tool: &FunctionTool,
) -> std::result::Result<CompiledJsonSchema, ApiError> {
    tool.compile_arguments_schema().map_err(|error| {
        ApiError::invalid_request(
            format!("unsupported JSON schema for constrained tool arguments: {error:#}"),
            Some("tools".to_owned()),
            Some("invalid_value".to_owned()),
        )
    })
}

fn runtime_tool_definition(tool: FunctionTool) -> RuntimeToolDefinition {
    RuntimeToolDefinition {
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters,
        strict: tool.function.strict,
    }
}

fn lower_agent_messages(
    messages: ValidatedMessages,
) -> std::result::Result<Vec<RuntimeAgentMessage>, ApiError> {
    match messages {
        ValidatedMessages::Text(messages) => Ok(messages
            .into_iter()
            .map(|message| match message.role {
                ChatRole::System => RuntimeAgentMessage::System {
                    content: message.content,
                },
                ChatRole::User => RuntimeAgentMessage::User {
                    content: message.content,
                },
                ChatRole::Assistant => RuntimeAgentMessage::Assistant {
                    content: Some(message.content),
                    tool_call: None,
                },
            })
            .collect()),
        ValidatedMessages::ToolAware(messages) => messages
            .into_iter()
            .map(|message| match message {
                OpenAiAgentMessage::System { content } => {
                    Ok(RuntimeAgentMessage::System { content })
                }
                OpenAiAgentMessage::User { content } => Ok(RuntimeAgentMessage::User { content }),
                OpenAiAgentMessage::Assistant {
                    content,
                    mut tool_calls,
                } => {
                    if tool_calls.len() > 1 {
                        return Err(ApiError::invalid_request(
                            "the native tool history subset supports one call per assistant turn",
                            Some("messages".to_owned()),
                            Some("unsupported_value".to_owned()),
                        ));
                    }
                    let tool_call = tool_calls.pop().map(|call| RuntimeToolCall {
                        id: call.id,
                        name: call.function.name,
                        arguments: call.function.arguments,
                    });
                    Ok(RuntimeAgentMessage::Assistant { content, tool_call })
                }
                OpenAiAgentMessage::Tool {
                    content,
                    tool_call_id,
                } => Ok(RuntimeAgentMessage::Tool {
                    content,
                    tool_call_id,
                }),
            })
            .collect(),
    }
}

#[derive(Clone)]
struct RequestIdentity {
    id: String,
    created: u64,
    model: String,
    default_seed: u64,
}

impl RequestIdentity {
    fn new(model: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let created = now.as_secs();
        let sequence = REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let nanos = now.as_nanos();
        let default_seed =
            (nanos as u64) ^ ((nanos >> u64::BITS) as u64) ^ sequence.rotate_left(u64::BITS / 2);
        Self {
            id: format!("chatcmpl-{created:x}{sequence:x}"),
            created,
            model,
            default_seed,
        }
    }
}

async fn non_streaming_response(
    mut ticket: EngineTicket,
    identity: RequestIdentity,
    output: AssistantOutput,
) -> Response {
    let mut content = String::new();
    while let Some(event) = ticket.recv().await {
        match event {
            EngineEvent::Text {
                content: fragment, ..
            } => content.push_str(&fragment),
            EngineEvent::Finished(metrics) => {
                if matches!(output, AssistantOutput::ForcedCall { .. })
                    && metrics.finish_reason != crate::generation::FinishReason::Stop
                {
                    return api_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        ApiError::server_error(
                            "forced tool generation reached its length limit before a stop token",
                        ),
                    );
                }
                let Some(finish_reason) = ChatFinishReason::from_generation(metrics.finish_reason)
                else {
                    return api_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        ApiError::server_error("request was cancelled before completion"),
                    );
                };
                let usage = TokenUsage::with_cached_tokens(
                    metrics.prompt_tokens,
                    metrics.generated_tokens,
                    metrics.cached_prompt_tokens,
                );
                let call = match &output {
                    AssistantOutput::Text => Ok(None),
                    AssistantOutput::ForcedCall { name } => {
                        canonical_arguments(&content).map(|arguments| {
                            Some((
                                None,
                                AssistantFunctionCall {
                                    name: name.clone(),
                                    arguments,
                                },
                            ))
                        })
                    }
                    AssistantOutput::AutoCall { calls } => split_auto_call(&content, calls),
                };
                let response = match call {
                    Ok(Some((content, function))) => ChatCompletionResponse::tool_call(
                        identity.id.clone(),
                        identity.created,
                        identity.model,
                        content,
                        AssistantToolCall {
                            id: tool_call_id(&identity.id),
                            kind: FunctionToolKind::Function,
                            function,
                        },
                        usage,
                    ),
                    Ok(None) => ChatCompletionResponse::new(
                        identity.id,
                        identity.created,
                        identity.model,
                        content,
                        finish_reason,
                        usage,
                    ),
                    Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, error),
                };
                return (StatusCode::OK, Json(response)).into_response();
            }
            EngineEvent::Failed { message, kind } => {
                let (status, error) = match kind {
                    EngineFailureKind::InvalidRequest => (
                        StatusCode::BAD_REQUEST,
                        ApiError::invalid_request(
                            message,
                            None,
                            Some("invalid_request".to_owned()),
                        ),
                    ),
                    EngineFailureKind::ResourceExhausted => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        ApiError::server_error(message),
                    ),
                    EngineFailureKind::Internal | EngineFailureKind::Fatal => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        ApiError::server_error(message),
                    ),
                };
                return api_error(status, error);
            }
        }
    }
    api_error(
        StatusCode::SERVICE_UNAVAILABLE,
        ApiError::server_error("inference worker stopped before completing the request"),
    )
}

fn tool_call_id(request_id: &str) -> String {
    format!(
        "call_{}",
        request_id.strip_prefix("chatcmpl-").unwrap_or(request_id)
    )
}

/// Separate the visible content of an automatic turn from the one native
/// call it may end with. Content is the text before the opening tag without
/// the separating whitespace, or `None` when nothing precedes the call. Any
/// opened call that fails to parse or validate is a server error: it is never
/// returned as content, where a client would mistake it for a final answer.
fn split_auto_call(
    text: &str,
    calls: &CompiledToolCalls,
) -> std::result::Result<Option<(Option<String>, AssistantFunctionCall)>, ApiError> {
    match scan_tool_call(text) {
        ToolCallScan::Text { .. } => Ok(None),
        ToolCallScan::Call { content_end, open } => {
            let function = auto_call_function(&text[open + TOOL_CALL_OPEN.len()..], calls)?;
            let content = (content_end > 0).then(|| text[..content_end].to_owned());
            Ok(Some((content, function)))
        }
    }
}

fn auto_call_function(
    after_open: &str,
    calls: &CompiledToolCalls,
) -> std::result::Result<AssistantFunctionCall, ApiError> {
    let call = parse_tool_call(after_open)
        .and_then(|call| {
            calls.validate_call(&call.name, &call.arguments)?;
            Ok(call)
        })
        .map_err(|error| {
            ApiError::server_error(format!(
                "model produced an invalid native tool call: {error:#}"
            ))
        })?;
    let arguments = serde_json::to_string(&call.arguments).map_err(|error| {
        ApiError::server_error(format!("failed to serialize tool arguments: {error}"))
    })?;
    Ok(AssistantFunctionCall {
        name: call.name,
        arguments,
    })
}

fn canonical_arguments(arguments: &str) -> std::result::Result<String, ApiError> {
    let value: serde_json::Value = serde_json::from_str(arguments).map_err(|error| {
        ApiError::server_error(format!(
            "constrained tool arguments were not valid JSON: {error}"
        ))
    })?;
    if !value.is_object() {
        return Err(ApiError::server_error(
            "constrained tool arguments were not a JSON object",
        ));
    }
    serde_json::to_string(&value).map_err(|error| {
        ApiError::server_error(format!(
            "failed to serialize constrained tool arguments: {error}"
        ))
    })
}

fn streaming_response(
    ticket: EngineTicket,
    identity: RequestIdentity,
    include_usage: bool,
    output: AssistantOutput,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let state = StreamState {
        ticket,
        identity,
        include_usage,
        role_sent: false,
        terminal: false,
        pending: VecDeque::new(),
        output,
        buffered_tool_arguments: String::new(),
        splitter: ToolCallSplitter::default(),
    };
    Sse::new(stream::unfold(state, next_stream_event)).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

struct StreamState {
    ticket: EngineTicket,
    identity: RequestIdentity,
    include_usage: bool,
    role_sent: bool,
    terminal: bool,
    pending: VecDeque<Event>,
    output: AssistantOutput,
    /// Forced calls stream nothing until their arguments are complete.
    buffered_tool_arguments: String,
    /// Automatic calls stream content and withhold everything from the
    /// opening tag on until the call has been validated.
    splitter: ToolCallSplitter,
}

impl StreamState {
    fn chunk_event(&self, build: impl FnOnce(String, u64, String) -> ChatCompletionChunk) -> Event {
        json_event(&build(
            self.identity.id.clone(),
            self.identity.created,
            self.identity.model.clone(),
        ))
    }

    /// Queue one validated call as `OpenAI` tool-call deltas: the header with
    /// id and name, the complete arguments, then `finish_reason=tool_calls`.
    fn push_tool_call(&mut self, function: AssistantFunctionCall) {
        let call_id = tool_call_id(&self.identity.id);
        let header = self.chunk_event(|id, created, model| {
            ChatCompletionChunk::tool_call_header(id, created, model, call_id, function.name)
        });
        let arguments = self.chunk_event(|id, created, model| {
            ChatCompletionChunk::tool_arguments(id, created, model, function.arguments)
        });
        let finished = self.chunk_event(|id, created, model| {
            ChatCompletionChunk::finished(id, created, model, ChatFinishReason::ToolCalls)
        });
        self.pending.extend([header, arguments, finished]);
    }
}

#[allow(clippy::too_many_lines)]
async fn next_stream_event(
    mut state: StreamState,
) -> Option<(std::result::Result<Event, Infallible>, StreamState)> {
    if !state.role_sent {
        state.role_sent = true;
        if !matches!(state.output, AssistantOutput::ForcedCall { .. }) {
            let chunk = ChatCompletionChunk::role(
                state.identity.id.clone(),
                state.identity.created,
                state.identity.model.clone(),
            );
            return Some((Ok(json_event(&chunk)), state));
        }
    }

    loop {
        if let Some(event) = state.pending.pop_front() {
            return Some((Ok(event), state));
        }
        if state.terminal {
            return None;
        }

        match state.ticket.recv().await {
            Some(EngineEvent::Text {
                content: fragment, ..
            }) => {
                let fragment = match &state.output {
                    AssistantOutput::Text => fragment,
                    AssistantOutput::ForcedCall { .. } => {
                        state.buffered_tool_arguments.push_str(&fragment);
                        continue;
                    }
                    AssistantOutput::AutoCall { .. } => {
                        let safe = state.splitter.push(&fragment);
                        if safe.is_empty() {
                            continue;
                        }
                        safe
                    }
                };
                let chunk = ChatCompletionChunk::content(
                    state.identity.id.clone(),
                    state.identity.created,
                    state.identity.model.clone(),
                    fragment,
                );
                return Some((Ok(json_event(&chunk)), state));
            }
            Some(EngineEvent::Finished(metrics)) => {
                let reason = ChatFinishReason::from_generation(metrics.finish_reason);
                match std::mem::replace(&mut state.output, AssistantOutput::Text) {
                    AssistantOutput::ForcedCall { name } => {
                        if metrics.finish_reason == crate::generation::FinishReason::Stop {
                            match canonical_arguments(&state.buffered_tool_arguments) {
                                Ok(arguments) => {
                                    let role = state.chunk_event(ChatCompletionChunk::role);
                                    state.pending.push_back(role);
                                    state.push_tool_call(AssistantFunctionCall { name, arguments });
                                }
                                Err(error) => {
                                    state.pending.push_back(json_event(&error.into_response()));
                                }
                            }
                        } else {
                            state.pending.push_back(json_event(
                                &ApiError::server_error(
                                    "forced tool generation reached its length limit before a stop token",
                                )
                                .into_response(),
                            ));
                        }
                    }
                    // A cancelled generation exposes neither held text nor a call.
                    AssistantOutput::AutoCall { calls } => {
                        if let Some(reason) = reason {
                            match std::mem::take(&mut state.splitter).finish() {
                                SplitEnd::Content(rest) => {
                                    if !rest.is_empty() {
                                        let content = state.chunk_event(|id, created, model| {
                                            ChatCompletionChunk::content(id, created, model, rest)
                                        });
                                        state.pending.push_back(content);
                                    }
                                    let finished = state.chunk_event(|id, created, model| {
                                        ChatCompletionChunk::finished(id, created, model, reason)
                                    });
                                    state.pending.push_back(finished);
                                }
                                SplitEnd::Call(after_open) => {
                                    match auto_call_function(&after_open, &calls) {
                                        Ok(function) => state.push_tool_call(function),
                                        Err(error) => state
                                            .pending
                                            .push_back(json_event(&error.into_response())),
                                    }
                                }
                            }
                        }
                    }
                    AssistantOutput::Text => {
                        if let Some(reason) = reason {
                            let finished = state.chunk_event(|id, created, model| {
                                ChatCompletionChunk::finished(id, created, model, reason)
                            });
                            state.pending.push_back(finished);
                        }
                    }
                }
                if state.include_usage {
                    let usage = TokenUsage::with_cached_tokens(
                        metrics.prompt_tokens,
                        metrics.generated_tokens,
                        metrics.cached_prompt_tokens,
                    );
                    state
                        .pending
                        .push_back(json_event(&ChatCompletionChunk::usage(
                            state.identity.id.clone(),
                            state.identity.created,
                            state.identity.model.clone(),
                            usage,
                        )));
                }
                state.pending.push_back(Event::default().data("[DONE]"));
                state.terminal = true;
            }
            Some(EngineEvent::Failed { message, kind }) => {
                let error = match kind {
                    EngineFailureKind::InvalidRequest => {
                        ApiError::invalid_request(message, None, Some("invalid_request".to_owned()))
                    }
                    EngineFailureKind::ResourceExhausted
                    | EngineFailureKind::Internal
                    | EngineFailureKind::Fatal => ApiError::server_error(message),
                };
                state.pending.push_back(json_event(&error.into_response()));
                state.pending.push_back(Event::default().data("[DONE]"));
                state.terminal = true;
            }
            None => {
                state.pending.push_back(json_event(
                    &ApiError::server_error(
                        "inference worker stopped before completing the request",
                    )
                    .into_response(),
                ));
                state.pending.push_back(Event::default().data("[DONE]"));
                state.terminal = true;
            }
        }
    }
}

fn json_event(value: &impl Serialize) -> Event {
    match serde_json::to_string(value) {
        Ok(json) => Event::default().data(json),
        Err(_) => Event::default().data(
            r#"{"error":{"message":"response serialization failed","type":"server_error","param":null,"code":"internal_error"}}"#,
        ),
    }
}

fn submit_error(error: SubmitError) -> Response {
    match error {
        SubmitError::Full => api_error(
            StatusCode::TOO_MANY_REQUESTS,
            ApiError::server_error("inference queue is full"),
        ),
        SubmitError::Unavailable(_) | SubmitError::Closed => api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            ApiError::server_error(error.to_string()),
        ),
    }
}

fn status_for_api_error(error: &ApiError) -> StatusCode {
    match error.status {
        crate::openai::ApiErrorStatus::BadRequest => StatusCode::BAD_REQUEST,
        crate::openai::ApiErrorStatus::InternalServerError => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn api_error(status: StatusCode, error: ApiError) -> Response {
    (status, Json(error.into_response())).into_response()
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize},
    };

    use axum::{
        body::Body,
        http::{Request, header},
    };
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use crate::engine::EnginePreflightError;
    use crate::generation::{FinishReason, GenerationMetrics};

    use super::*;

    struct FakeEngine {
        metadata: EngineMetadata,
        status: EngineStatus,
        events: Mutex<Vec<EngineEvent>>,
        submit_error: Option<SubmitError>,
        preflight_error: Option<EnginePreflightError>,
        last_cancelled: Mutex<Option<Arc<AtomicBool>>>,
        last_request: Mutex<Option<EngineRequest>>,
        last_policy: Mutex<Option<EngineRequestPolicy>>,
        submit_count: AtomicUsize,
    }

    impl FakeEngine {
        fn healthy(events: Vec<EngineEvent>) -> Self {
            Self {
                metadata: EngineMetadata {
                    served_model: "test-model".to_owned(),
                    family: "test".to_owned(),
                    max_context: 128,
                    device: "cpu".to_owned(),
                    dtype: "f32".to_owned(),
                    chat_template_defaults: crate::chat::ChatTemplateOptions::default(),
                },
                status: EngineStatus::Ready,
                events: Mutex::new(events),
                submit_error: None,
                preflight_error: None,
                last_cancelled: Mutex::new(None),
                last_request: Mutex::new(None),
                last_policy: Mutex::new(None),
                submit_count: AtomicUsize::new(0),
            }
        }
    }

    impl InferenceEngine for FakeEngine {
        fn metadata(&self) -> EngineMetadata {
            self.metadata.clone()
        }

        fn status(&self) -> EngineStatus {
            self.status
        }

        fn submit(&self, request: EngineRequest) -> std::result::Result<EngineTicket, SubmitError> {
            self.submit_with_policy(request, EngineRequestPolicy::default())
        }

        fn submit_with_policy(
            &self,
            request: EngineRequest,
            policy: EngineRequestPolicy,
        ) -> std::result::Result<EngineTicket, SubmitError> {
            self.submit_count.fetch_add(1, Ordering::Relaxed);
            if let Some(error) = self.submit_error {
                return Err(error);
            }
            *self.last_request.lock().unwrap() = Some(request);
            *self.last_policy.lock().unwrap() = Some(policy);
            let events = std::mem::take(&mut *self.events.lock().unwrap());
            let preflight = self.preflight_error.clone().map_or(Ok(()), Err);
            let (sender, cancelled, ticket) = EngineTicket::channel(events.len().max(1), preflight);
            *self.last_cancelled.lock().unwrap() = Some(cancelled);
            for event in events {
                sender.try_send(event).unwrap();
            }
            drop(sender);
            Ok(ticket)
        }
    }

    fn metrics(reason: FinishReason) -> GenerationMetrics {
        GenerationMetrics {
            prompt_tokens: 3,
            cached_prompt_tokens: 0,
            prefix_lookup_attach_time: Duration::ZERO,
            generated_tokens: 2,
            decode_steps: 1,
            time_to_first_token: Duration::ZERO,
            suffix_prefill_time: Duration::ZERO,
            first_token_model_time: Duration::ZERO,
            prefill_time: Duration::ZERO,
            decode_time: Duration::ZERO,
            total_time: Duration::ZERO,
            finish_reason: reason,
        }
    }

    fn text_event(
        content: impl Into<String>,
        token_id: u32,
        n_decoded: usize,
        prompt_tokens: usize,
    ) -> EngineEvent {
        EngineEvent::Text {
            content: content.into(),
            token: Some(crate::engine::EngineTokenEvidence {
                token_id,
                n_decoded,
                prompt_tokens,
            }),
        }
    }

    #[test]
    fn request_seed_is_local_unless_the_client_supplies_one() {
        let validated = |seed: Option<i64>| {
            let mut value = json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}]
            });
            if let Some(seed) = seed {
                value["seed"] = json!(seed);
            }
            ChatCompletionRequest::from_json(&serde_json::to_vec(&value).unwrap())
                .unwrap()
                .validate("test-model", 16)
                .unwrap()
        };

        let (implicit, _) =
            to_engine_request(validated(None), 91, SamplingProfile::OpenAi.defaults(true)).unwrap();
        assert_eq!(implicit.sampling.seed, 91);
        assert!(matches!(implicit.input, EngineInput::Chat(_)));
        assert!(!implicit.ignore_stop_tokens);
        assert_eq!(
            to_engine_request(
                validated(Some(7)),
                91,
                SamplingProfile::OpenAi.defaults(true)
            )
            .unwrap()
            .0
            .sampling
            .seed,
            7
        );

        let structured = ChatCompletionRequest::from_json(
            &serde_json::to_vec(&json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "return JSON"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": true,
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": false
                        }
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap()
        .validate("test-model", 16)
        .unwrap();
        let (structured, _) =
            to_engine_request(structured, 91, SamplingProfile::OpenAi.defaults(true)).unwrap();
        assert!(matches!(
            structured.input,
            EngineInput::ConstrainedChat {
                constraint: EngineConstraintSpec {
                    purpose: crate::engine::EngineConstraintPurpose::ResponseFormat,
                    ..
                },
                ..
            }
        ));
    }

    fn request(streaming: bool, include_usage: bool) -> Request<Body> {
        let stream_options = if include_usage {
            json!({"include_usage": true})
        } else {
            Value::Null
        };
        let mut value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": streaming
        });
        if include_usage {
            value["stream_options"] = stream_options;
        }
        Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&value).unwrap()))
            .unwrap()
    }

    fn raw_request(streaming: bool, cache_prompt: bool) -> Request<Body> {
        Request::post("/completion")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "prompt": "a b",
                    "n_predict": 2,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "seed": 7,
                    "cache_prompt": cache_prompt,
                    "stream": streaming,
                    "return_tokens": true,
                    "id_slot": 3,
                    "ignore_eos": true,
                    "repeat_penalty": 1.0,
                    "response_fields": RAW_COMPLETION_RESPONSE_FIELDS
                }))
                .unwrap(),
            ))
            .unwrap()
    }

    fn structured_request(streaming: bool) -> Request<Body> {
        Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "return JSON"}],
                    "stream": streaming,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer",
                            "strict": true,
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                                "additionalProperties": false
                            }
                        }
                    }
                })
                .to_string(),
            ))
            .unwrap()
    }

    fn forced_tool_request(streaming: bool, include_usage: bool) -> Request<Body> {
        let mut value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "look up AAPL"}],
            "stream": streaming,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a symbol",
                    "parameters": {
                        "type": "object",
                        "properties": {"symbol": {"type": "string"}},
                        "required": ["symbol"],
                        "additionalProperties": false
                    },
                    "strict": true
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup"}
            },
            "parallel_tool_calls": false
        });
        if include_usage {
            value["stream_options"] = json!({"include_usage": true});
        }
        Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(value.to_string()))
            .unwrap()
    }

    async fn body_json(response: Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_and_models_report_the_single_loaded_runtime() {
        let app = router(Arc::new(FakeEngine::healthy(Vec::new())));
        let health = app
            .clone()
            .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        let health = body_json(health).await;
        assert_eq!(health["worker"], "ready");
        assert_eq!(health["device"], "cpu");
        assert_eq!(health["dtype"], "f32");

        let models = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(body_json(models).await["data"][0]["id"], "test-model");
    }

    #[tokio::test]
    async fn raw_completion_uses_the_untemplated_fixed_length_contract() {
        let mut finished = metrics(FinishReason::Length);
        finished.cached_prompt_tokens = 2;
        finished.time_to_first_token = Duration::from_millis(3);
        finished.prefill_time = Duration::from_millis(2);
        finished.total_time = Duration::from_millis(4);
        let engine = Arc::new(FakeEngine::healthy(vec![
            text_event("hel", 11, 1, 3),
            text_event("lo", 12, 2, 3),
            EngineEvent::Finished(finished),
        ]));
        let response = router(engine.clone())
            .oneshot(raw_request(false, true))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["content"], "hello");
        assert!(body.get("tokens").is_none());
        assert!(body.get("prompt").is_none());
        assert_eq!(
            body.as_object()
                .expect("raw completion response must be an object")
                .len(),
            RAW_COMPLETION_RESPONSE_FIELDS.len()
        );
        assert_eq!(body["stop"], true);
        assert_eq!(body["tokens_predicted"], 2);
        assert_eq!(body["tokens_evaluated"], 3);
        assert_eq!(body["stop_type"], "limit");
        assert_eq!(body["generation_settings/ignore_eos"], true);
        assert_eq!(body["generation_settings/repeat_penalty"], 1.0);
        assert_eq!(body["generation_settings/n_predict"], 2);
        assert_eq!(body["timings/cache_n"], 2);
        assert_eq!(body["timings/prompt_n"], 1);
        assert_eq!(body["timings/predicted_n"], 2);
        assert_eq!(body["timings/predicted_ms"], 1.0);

        let submitted = engine.last_request.lock().unwrap();
        let submitted = submitted.as_ref().expect("raw request was not submitted");
        assert!(matches!(
            &submitted.input,
            EngineInput::RawPrompt(prompt) if prompt == "a b"
        ));
        assert!(submitted.ignore_stop_tokens);
        assert!(submitted.stop.is_empty());
        assert_eq!(submitted.max_new_tokens, 2);
        assert_eq!(
            submitted.sampling.repetition_penalty.to_bits(),
            1.0_f32.to_bits()
        );
        assert_eq!(submitted.sampling.seed, 7);
        assert!(
            engine
                .last_policy
                .lock()
                .unwrap()
                .expect("raw request policy was not recorded")
                .publish_prompt_prefix
        );
    }

    #[tokio::test]
    async fn raw_completion_streams_visible_text_then_exact_terminal_evidence() {
        let mut finished = metrics(FinishReason::Length);
        finished.cached_prompt_tokens = 1;
        finished.time_to_first_token = Duration::from_millis(3);
        finished.prefill_time = Duration::from_millis(2);
        finished.total_time = Duration::from_millis(4);
        let engine = Arc::new(FakeEngine::healthy(vec![
            text_event("hello", 42, 2, 3),
            EngineEvent::Finished(finished),
        ]));
        let response = router(engine.clone())
            .oneshot(raw_request(true, false))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let body = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        let fragment = body.find("\"content\":\"hello\"").unwrap();
        let terminal = body.find("\"tokens_predicted\":2").unwrap();
        assert!(fragment < terminal);
        assert!(body.contains("\"tokens_evaluated\":3"));
        assert!(body.contains("\"timings/cache_n\":1"));
        assert!(body.contains("\"timings/prompt_n\":2"));
        assert!(body.contains("\"timings/predicted_ms\":1.0"));
        assert!(body.contains("\"stop_type\":\"limit\""));
        assert!(body.contains("\"tokens\":[42]"));
        assert!(body.contains("\"id_slot\":-1"));
        assert!(!body.contains("[DONE]"));
        assert!(!body.contains("keep-alive"));
        assert!(
            !engine
                .last_policy
                .lock()
                .unwrap()
                .expect("raw request policy was not recorded")
                .publish_prompt_prefix
        );
    }

    #[tokio::test]
    async fn raw_completion_stream_fails_closed_without_token_evidence() {
        let engine = Arc::new(FakeEngine::healthy(vec![
            EngineEvent::Text {
                content: "tail".to_owned(),
                token: None,
            },
            EngineEvent::Finished(metrics(FinishReason::Length)),
        ]));
        let response = router(engine)
            .oneshot(raw_request(true, false))
            .await
            .unwrap();
        let body = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains("without sampled-token evidence"));
        assert!(!body.contains("\"stop_type\":\"limit\""));
    }

    #[tokio::test]
    async fn raw_completion_stream_reconciles_monotonic_token_metadata() {
        let cases = [
            (
                vec![
                    text_event("a", 11, 1, 3),
                    text_event("b", 12, 1, 3),
                    EngineEvent::Finished(metrics(FinishReason::Length)),
                ],
                "inconsistent raw token metadata",
            ),
            (
                vec![
                    text_event("a", 11, 1, 3),
                    text_event("b", 12, 2, 4),
                    EngineEvent::Finished(metrics(FinishReason::Length)),
                ],
                "inconsistent raw token metadata",
            ),
            (
                vec![
                    text_event("a", 11, 1, 3),
                    EngineEvent::Finished(metrics(FinishReason::Length)),
                ],
                "did not reconcile with terminal metrics",
            ),
        ];

        for (events, expected_error) in cases {
            let response = router(Arc::new(FakeEngine::healthy(events)))
                .oneshot(raw_request(true, false))
                .await
                .unwrap();
            let body = String::from_utf8(
                response
                    .into_body()
                    .collect()
                    .await
                    .unwrap()
                    .to_bytes()
                    .to_vec(),
            )
            .unwrap();
            assert!(body.contains(expected_error), "{body}");
            assert!(!body.contains("\"stop_type\":\"limit\""), "{body}");
        }
    }

    #[test]
    fn raw_completion_rejects_unknown_or_non_fixed_contract_fields() {
        let base = json!({
            "prompt": "a b",
            "n_predict": 2,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": 7,
            "cache_prompt": true,
            "stream": true,
            "return_tokens": true,
            "id_slot": 3,
            "ignore_eos": true,
            "repeat_penalty": 1.0,
            "response_fields": RAW_COMPLETION_RESPONSE_FIELDS
        });

        let mut unknown = base.clone();
        unknown["unexpected"] = json!(true);
        let error = serde_json::from_value::<RawCompletionRequest>(unknown).unwrap_err();
        assert!(error.to_string().contains("unknown field"));

        for (field, value, expected_param, expected_code) in [
            ("id_slot", json!(8), "id_slot", "invalid_value"),
            (
                "ignore_eos",
                json!(false),
                "ignore_eos",
                "unsupported_value",
            ),
            (
                "return_tokens",
                json!(false),
                "return_tokens",
                "unsupported_value",
            ),
            (
                "repeat_penalty",
                json!(1.1),
                "repeat_penalty",
                "unsupported_value",
            ),
        ] {
            let mut value_with_error = base.clone();
            value_with_error[field] = value;
            let error = serde_json::from_value::<RawCompletionRequest>(value_with_error)
                .unwrap()
                .validate(128)
                .unwrap_err();
            assert_eq!(error.param.as_deref(), Some(expected_param));
            assert_eq!(error.code.as_deref(), Some(expected_code));
        }
    }

    #[test]
    fn raw_completion_requires_the_exact_requested_token_boundary() {
        let request = serde_json::from_value::<RawCompletionRequest>(json!({
            "prompt": "a b",
            "n_predict": 32,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": 7,
            "cache_prompt": true,
            "stream": true,
            "return_tokens": true,
            "id_slot": 0,
            "ignore_eos": true,
            "repeat_penalty": 1.0,
            "response_fields": RAW_COMPLETION_RESPONSE_FIELDS
        }))
        .unwrap();
        let (_engine_request, settings, _policy) = request.validate(128).unwrap();
        let mut finished = metrics(FinishReason::Length);
        finished.generated_tokens = 31;

        assert!(validate_raw_completion_metrics(&settings, &finished).is_err());
        finished.generated_tokens = 32;
        assert!(validate_raw_completion_metrics(&settings, &finished).is_ok());
        finished.finish_reason = FinishReason::Stop;
        assert!(validate_raw_completion_metrics(&settings, &finished).is_err());
    }

    #[tokio::test]
    async fn non_streaming_response_aggregates_worker_text_and_usage() {
        let app = router(Arc::new(FakeEngine::healthy(vec![
            text_event("hel", 11, 1, 3),
            text_event("lo", 12, 2, 3),
            EngineEvent::Finished(metrics(FinishReason::Stop)),
        ])));
        let response = app.oneshot(request(false, false)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["choices"][0]["message"]["content"], "hello");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["total_tokens"], 5);
        assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    }

    #[tokio::test]
    async fn json_schema_content_is_preserved_in_non_streaming_and_sse_responses() {
        let json_content = r#"{"answer":"ok"}"#;
        let non_streaming = router(Arc::new(FakeEngine::healthy(vec![
            text_event(json_content, 11, 1, 3),
            EngineEvent::Finished(metrics(FinishReason::Stop)),
        ])))
        .oneshot(structured_request(false))
        .await
        .unwrap();
        assert_eq!(non_streaming.status(), StatusCode::OK);
        assert_eq!(
            body_json(non_streaming).await["choices"][0]["message"]["content"],
            json_content
        );

        let streaming = router(Arc::new(FakeEngine::healthy(vec![
            text_event(json_content, 11, 1, 3),
            EngineEvent::Finished(metrics(FinishReason::Stop)),
        ])))
        .oneshot(structured_request(true))
        .await
        .unwrap();
        assert_eq!(streaming.status(), StatusCode::OK);
        let body = String::from_utf8(
            streaming
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains(r#""content":"{\"answer\":\"ok\"}""#));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn sse_emits_role_content_finish_usage_and_done_in_order() {
        let app = router(Arc::new(FakeEngine::healthy(vec![
            text_event("hello", 11, 1, 3),
            EngineEvent::Finished(metrics(FinishReason::Length)),
        ])));
        let response = app.oneshot(request(true, true)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let body = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        let role = body.find("\"role\":\"assistant\"").unwrap();
        let content = body.find("\"content\":\"hello\"").unwrap();
        let finish = body.find("\"finish_reason\":\"length\"").unwrap();
        let usage = body.find("\"total_tokens\":5").unwrap();
        let done = body.find("data: [DONE]").unwrap();
        assert!(role < content && content < finish && finish < usage && usage < done);
    }

    #[tokio::test]
    async fn unsupported_tool_modes_never_submit_and_full_queue_is_shaped() {
        let engine = Arc::new(FakeEngine::healthy(Vec::new()));
        let app = router(engine.clone());
        let tool = |name: &str| {
            json!({
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": false
                    },
                    "strict": true
                }
            })
        };
        let bad = Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [tool("lookup"), tool("clock")],
                    "tool_choice": "required",
                    "parallel_tool_calls": false
                })
                .to_string(),
            ))
            .unwrap();
        let response = app.oneshot(bad).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            body_json(response).await["error"]["code"],
            "unsupported_value"
        );
        assert_eq!(engine.submit_count.load(Ordering::Relaxed), 0);

        let mut full = FakeEngine::healthy(Vec::new());
        full.submit_error = Some(SubmitError::Full);
        let response = router(Arc::new(full))
            .oneshot(request(false, false))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(body_json(response).await.get("error").is_some());
    }

    #[test]
    fn forced_named_tool_lowers_to_one_native_definition_and_arguments_constraint() {
        let request = ChatCompletionRequest::from_json(
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": false
                            },
                            "strict": true
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "clock",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                                "additionalProperties": false
                            },
                            "strict": true
                        }
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "weather"}},
                "parallel_tool_calls": false
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap()
        .validate("test-model", 16)
        .unwrap();
        let (request, _) =
            to_engine_request(request, 7, SamplingProfile::OpenAi.defaults(true)).unwrap();
        let EngineInput::Agent {
            prompt,
            constraint: Some(constraint),
        } = request.input
        else {
            panic!("expected constrained native agent input");
        };
        assert_eq!(
            constraint.purpose,
            crate::engine::EngineConstraintPurpose::ToolArguments
        );
        assert_eq!(prompt.tools.len(), 1);
        assert_eq!(prompt.tools[0].name, "weather");
        assert_eq!(
            prompt.directive,
            AgentDirective::ForcedSingle {
                name: "weather".to_owned()
            }
        );
    }

    #[test]
    fn one_tool_call_and_response_history_lowers_without_losing_identity() {
        let body = json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "look up AAPL"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{ \"symbol\" : \"AAPL\" }"
                        }
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "123"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"symbol": {"type": "string"}},
                        "required": ["symbol"],
                        "additionalProperties": false
                    },
                    "strict": true
                }
            }],
            "tool_choice": "required",
            "parallel_tool_calls": false
        });
        let request = ChatCompletionRequest::from_json(body.to_string().as_bytes())
            .unwrap()
            .validate("test-model", 16)
            .unwrap();
        let (request, _) =
            to_engine_request(request, 7, SamplingProfile::OpenAi.defaults(true)).unwrap();
        let EngineInput::Agent { prompt, .. } = request.input else {
            panic!("expected native agent history");
        };
        let RuntimeAgentMessage::Assistant {
            content: None,
            tool_call: Some(call),
        } = &prompt.messages[1]
        else {
            panic!("expected one assistant tool call");
        };
        assert_eq!(call.id, "call_1");
        assert_eq!(call.name, "lookup");
        assert_eq!(call.arguments, r#"{"symbol":"AAPL"}"#);
        assert_eq!(
            prompt.messages[2],
            RuntimeAgentMessage::Tool {
                content: "123".to_owned(),
                tool_call_id: "call_1".to_owned()
            }
        );
    }

    #[tokio::test]
    async fn forced_tool_non_streaming_returns_only_a_canonical_tool_call() {
        let response = router(Arc::new(FakeEngine::healthy(vec![
            text_event("{ \"symbol\" :", 11, 1, 3),
            text_event(" \"AAPL\" }", 12, 2, 3),
            EngineEvent::Finished(metrics(FinishReason::Stop)),
        ])))
        .oneshot(forced_tool_request(false, false))
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["choices"][0]["message"]["content"], Value::Null);
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"],
            json!({"name": "lookup", "arguments": "{\"symbol\":\"AAPL\"}"})
        );
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn forced_tool_sse_buffers_until_success_and_never_leaks_failed_arguments() {
        let success = router(Arc::new(FakeEngine::healthy(vec![
            text_event("{\"symbol\":", 11, 1, 3),
            text_event("\"AAPL\"}", 12, 2, 3),
            EngineEvent::Finished(metrics(FinishReason::Stop)),
        ])))
        .oneshot(forced_tool_request(true, true))
        .await
        .unwrap();
        let success = String::from_utf8(
            success
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        let role = success.find("\"role\":\"assistant\"").unwrap();
        let header = success.find("\"name\":\"lookup\"").unwrap();
        let arguments = success.find("{\\\"symbol\\\":\\\"AAPL\\\"}").unwrap();
        let finish = success.find("\"finish_reason\":\"tool_calls\"").unwrap();
        let usage = success.find("\"total_tokens\":5").unwrap();
        let done = success.find("data: [DONE]").unwrap();
        assert!(role < header && header < arguments && arguments < finish && finish < usage);
        assert!(usage < done);
        assert!(!success.contains("\"content\""));

        let failed = router(Arc::new(FakeEngine::healthy(vec![
            text_event("{\"symbol\":\"SECRET", 11, 1, 3),
            EngineEvent::Failed {
                message: "incomplete tool arguments".to_owned(),
                kind: EngineFailureKind::InvalidRequest,
            },
        ])))
        .oneshot(forced_tool_request(true, false))
        .await
        .unwrap();
        let failed = String::from_utf8(
            failed
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(failed.contains("incomplete tool arguments"));
        assert!(!failed.contains("SECRET"));
        assert!(!failed.contains("tool_calls"));
    }

    #[tokio::test]
    async fn forced_tool_length_boundary_returns_no_tool_call() {
        for streaming in [false, true] {
            let response = router(Arc::new(FakeEngine::healthy(vec![
                text_event("{\"symbol\":\"AAPL\"}", 11, 1, 3),
                EngineEvent::Finished(metrics(FinishReason::Length)),
            ])))
            .oneshot(forced_tool_request(streaming, false))
            .await
            .unwrap();
            if streaming {
                let body = String::from_utf8(
                    response
                        .into_body()
                        .collect()
                        .await
                        .unwrap()
                        .to_bytes()
                        .to_vec(),
                )
                .unwrap();
                assert!(body.contains("length limit"));
                assert!(!body.contains("tool_calls"));
                assert!(!body.contains("AAPL"));
            } else {
                assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                let body = body_json(response).await;
                assert!(
                    body["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains("length limit")
                );
                assert!(body.get("choices").is_none());
            }
        }
    }

    /// RestLoop's only tool: non-strict, with no `additionalProperties`.
    fn shell_tool() -> Value {
        json!({"type": "function", "function": {
            "name": "shell",
            "description": "Run a shell command and return its exit code and output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }})
    }

    fn chat_request(body: &Value) -> Request<Body> {
        Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn auto_request(streaming: bool) -> Request<Body> {
        chat_request(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "list files"}],
            "tools": [shell_tool()],
            "stream": streaming,
            "stream_options": if streaming { json!({"include_usage": true}) } else { Value::Null }
        }))
    }

    async fn body_text(response: Response) -> String {
        String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap()
    }

    /// Decode every JSON `data:` event of an SSE body, in order.
    fn sse_json(body: &str) -> Vec<Value> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str(data).unwrap())
            .collect()
    }

    fn streamed_content(events: &[Value]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
            .map(str::to_owned)
            .collect()
    }

    const SHELL_CALL: &str =
        "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>";

    fn fragments(text: &str, sizes: &[usize]) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        let mut rest = text;
        for (index, size) in sizes.iter().enumerate() {
            let (head, tail) = rest.split_at((*size).min(rest.len()));
            events.push(text_event(head, 11, index + 1, 3));
            rest = tail;
        }
        if !rest.is_empty() {
            events.push(text_event(rest, 11, sizes.len() + 1, 3));
        }
        events
    }

    #[test]
    fn automatic_choice_lowers_every_tool_and_a_lazy_tool_call_constraint() {
        let mut clock = shell_tool();
        clock["function"]["name"] = json!("clock");
        let request = ChatCompletionRequest::from_json(
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "list files"}],
                "tools": [shell_tool(), clock]
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap()
        .validate("test-model", 16)
        .unwrap();
        let (request, output) =
            to_engine_request(request, 7, SamplingProfile::OpenAi.defaults(true)).unwrap();
        let EngineInput::Agent {
            prompt,
            constraint: Some(constraint),
        } = request.input
        else {
            panic!("expected constrained native agent input");
        };
        assert_eq!(
            constraint.purpose,
            crate::engine::EngineConstraintPurpose::ToolCall
        );
        assert_eq!(prompt.directive, AgentDirective::Auto);
        assert_eq!(
            prompt
                .tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            ["shell", "clock"]
        );
        let AssistantOutput::AutoCall { calls } = output else {
            panic!("expected an automatic call response contract");
        };
        calls
            .validate_call("clock", &json!({"command": "date"}))
            .unwrap();
        assert!(request.stop.is_empty());
    }

    #[tokio::test]
    async fn server_defaults_and_template_kwargs_reach_the_engine_request() {
        let engine = Arc::new(FakeEngine::healthy(vec![EngineEvent::Finished(metrics(
            FinishReason::Stop,
        ))]));
        let options = ServerOptions {
            default_max_completion_tokens: 64,
            sampling_profile: SamplingProfile::OpenAi,
        };
        let response = router_with_options(engine.clone(), options)
            .oneshot(chat_request(&json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "chat_template_kwargs": {"enable_thinking": false}
            })))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let request = engine.last_request.lock().unwrap().take().unwrap();
        assert_eq!(request.max_new_tokens, 64);
        assert_eq!(request.chat_template.enable_thinking, Some(false));
        assert!(matches!(request.input, EngineInput::Chat(_)));

        // Explicit budgets still win, and the agent path carries the kwargs.
        let engine = Arc::new(FakeEngine::healthy(vec![EngineEvent::Finished(metrics(
            FinishReason::Stop,
        ))]));
        let response = router_with_options(engine.clone(), options)
            .oneshot(chat_request(&json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "list files"}],
                "tools": [shell_tool()],
                "max_tokens": 7,
                "chat_template_kwargs": {"enable_thinking": true}
            })))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let request = engine.last_request.lock().unwrap().take().unwrap();
        assert_eq!(request.max_new_tokens, 7);
        assert_eq!(request.chat_template.enable_thinking, Some(true));
        assert!(matches!(request.input, EngineInput::Agent { .. }));
        assert_eq!(ServerOptions::default().default_max_completion_tokens, 128);
    }

    #[tokio::test]
    async fn sampling_profile_fills_only_omitted_fields_by_thinking_mode() {
        let sampled = |body: Value, server_disables_thinking: bool, profile| async move {
            let mut fake =
                FakeEngine::healthy(vec![EngineEvent::Finished(metrics(FinishReason::Stop))]);
            if server_disables_thinking {
                fake.metadata.chat_template_defaults.enable_thinking = Some(false);
            }
            let engine = Arc::new(fake);
            let options = ServerOptions {
                sampling_profile: profile,
                ..ServerOptions::default()
            };
            let response = router_with_options(engine.clone(), options)
                .oneshot(chat_request(&body))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let sampling = engine.last_request.lock().unwrap().take().unwrap().sampling;
            (sampling.temperature, sampling.top_p, sampling.top_k)
        };
        let base = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "list files"}],
            "tools": [shell_tool()]
        });
        let with = |patch: Value| {
            let mut body = base.clone();
            body.as_object_mut()
                .unwrap()
                .extend(patch.as_object().unwrap().clone());
            body
        };

        // The OpenAI profile keeps today's behavior.
        assert_eq!(
            sampled(base.clone(), false, SamplingProfile::OpenAi).await,
            (1.0, 1.0, 0)
        );
        // Qwen3: thinking by default, non-thinking when the server or the
        // request disables it; a request can also re-enable it.
        let qwen3 = SamplingProfile::Qwen3;
        assert_eq!(sampled(base.clone(), false, qwen3).await, (0.6, 0.95, 20));
        assert_eq!(sampled(base.clone(), true, qwen3).await, (0.7, 0.8, 20));
        let kwargs =
            |enabled: bool| with(json!({"chat_template_kwargs": {"enable_thinking": enabled}}));
        assert_eq!(sampled(kwargs(false), false, qwen3).await, (0.7, 0.8, 20));
        assert_eq!(sampled(kwargs(true), true, qwen3).await, (0.6, 0.95, 20));
        // Request values always win, field by field.
        assert_eq!(
            sampled(with(json!({"temperature": 0.0})), true, qwen3).await,
            (0.0, 0.8, 20)
        );
        assert_eq!(
            sampled(with(json!({"temperature": 1.2, "top_p": 0.5})), true, qwen3).await,
            (1.2, 0.5, 20)
        );
    }

    #[tokio::test]
    async fn auto_non_streaming_separates_content_from_one_validated_call() {
        let reasoning = "<think>\nNeed ls.\n</think>\n\nListing.\n";
        let cases = [
            (
                format!("{reasoning}{SHELL_CALL}"),
                json!("<think>\nNeed ls.\n</think>\n\nListing."),
            ),
            (format!("\n{SHELL_CALL}"), Value::Null),
        ];
        for (text, content) in cases {
            let response = router(Arc::new(FakeEngine::healthy({
                let mut events = fragments(&text, &[7, 30, 5]);
                events.push(EngineEvent::Finished(metrics(FinishReason::Stop)));
                events
            })))
            .oneshot(auto_request(false))
            .await
            .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = body_json(response).await;
            let message = &body["choices"][0]["message"];
            assert_eq!(message["content"], content);
            let call = &message["tool_calls"][0];
            assert!(call["id"].as_str().unwrap().starts_with("call_"));
            assert_eq!(call["type"], "function");
            assert_eq!(
                call["function"],
                json!({"name": "shell", "arguments": "{\"command\":\"ls\"}"})
            );
            assert_eq!(message["tool_calls"].as_array().unwrap().len(), 1);
            assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        }

        // An opening tag outside reasoning always starts a call. The lazy
        // grammar would have constrained what follows; text that bypassed it
        // is an error, never content a client could mistake for an answer.
        let response = router(Arc::new(FakeEngine::healthy(vec![
            text_event("Use <tool_call> tags", 11, 1, 3),
            EngineEvent::Finished(metrics(FinishReason::Stop)),
        ])))
        .oneshot(auto_request(false))
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        // Reasoning may spell the tag, and a held partial tag is released as
        // content when generation ends without a call.
        let response = router(Arc::new(FakeEngine::healthy(vec![
            text_event(
                "<think>\nWhat is <tool_call>?\n</think>\n\nA tag. <tool",
                11,
                1,
                3,
            ),
            EngineEvent::Finished(metrics(FinishReason::Length)),
        ])))
        .oneshot(auto_request(false))
        .await
        .unwrap();
        let body = body_json(response).await;
        assert_eq!(
            body["choices"][0]["message"]["content"],
            "<think>\nWhat is <tool_call>?\n</think>\n\nA tag. <tool"
        );
        assert!(body["choices"][0]["message"].get("tool_calls").is_none());
        assert_eq!(body["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn auto_calls_that_fail_validation_are_errors_never_content() {
        let invalid = [
            "Sure.\n<tool_call>\n{\"name\": \"rm\", \"arguments\": {\"path\": \"SECRET\"}}\n</tool_call>",
            "Sure.\n<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": 1, \"x\": \"SECRET\"}}\n</tool_call>",
            "Sure.\n<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": \"SECRET\"}}",
        ];
        for text in invalid {
            for streaming in [false, true] {
                let response = router(Arc::new(FakeEngine::healthy(vec![
                    text_event(text, 11, 1, 3),
                    EngineEvent::Finished(metrics(FinishReason::Stop)),
                ])))
                .oneshot(auto_request(streaming))
                .await
                .unwrap();
                if streaming {
                    let body = body_text(response).await;
                    let events = sse_json(&body);
                    assert_eq!(streamed_content(&events).concat(), "Sure.");
                    assert!(body.contains("invalid native tool call"));
                    assert!(!body.contains("SECRET"));
                    assert!(!body.contains("tool_calls"));
                    assert!(body.contains("data: [DONE]"));
                } else {
                    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                    let body = body_json(response).await;
                    assert!(body.get("choices").is_none());
                    assert!(!body.to_string().contains("SECRET"));
                }
            }
        }
    }

    #[tokio::test]
    async fn auto_sse_streams_content_then_tool_call_deltas_without_leaking_the_tag() {
        let text = format!("<think>\nok\n</think>\n\nListing.\n{SHELL_CALL}");
        // Split the opening tag across several worker fragments.
        let opening = text.find("<tool_call>").unwrap();
        let mut events = fragments(&text, &[opening - 3, 4, 3, 4, 9]);
        events.push(EngineEvent::Finished(metrics(FinishReason::Stop)));
        let response = router(Arc::new(FakeEngine::healthy(events)))
            .oneshot(auto_request(true))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_text(response).await;
        let events = sse_json(&body);

        assert_eq!(events[0]["choices"][0]["delta"]["role"], "assistant");
        let content = streamed_content(&events);
        assert_eq!(content.concat(), "<think>\nok\n</think>\n\nListing.");
        assert!(content.iter().all(|delta| !delta.contains("tool")));

        let header = events
            .iter()
            .position(|event| event["choices"][0]["delta"]["tool_calls"][0]["id"].is_string())
            .unwrap();
        let header_call = &events[header]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(header_call["index"], 0);
        assert_eq!(header_call["type"], "function");
        assert_eq!(header_call["function"]["name"], "shell");
        assert_eq!(
            events[header + 1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            "{\"command\":\"ls\"}"
        );
        assert_eq!(
            events[header + 2]["choices"][0]["finish_reason"],
            "tool_calls"
        );
        assert_eq!(events[header + 3]["usage"]["total_tokens"], 5);
        assert!(body.trim_end().ends_with("data: [DONE]"));
        let last_content = events
            .iter()
            .rposition(|event| event["choices"][0]["delta"]["content"].is_string())
            .unwrap();
        assert!(last_content < header);
    }

    #[tokio::test]
    async fn auto_sse_releases_held_text_when_no_call_opens() {
        let response = router(Arc::new(FakeEngine::healthy(vec![
            text_event("Done", 11, 1, 3),
            text_event(" \n", 12, 2, 3),
            text_event("<tool", 13, 3, 3),
            EngineEvent::Finished(metrics(FinishReason::Length)),
        ])))
        .oneshot(auto_request(true))
        .await
        .unwrap();
        let body = body_text(response).await;
        let events = sse_json(&body);
        assert_eq!(streamed_content(&events), ["Done", " \n<tool"]);
        assert!(body.contains("\"finish_reason\":\"length\""));
        assert!(!body.contains("tool_calls"));
    }

    /// Apply RestLoop's bookkeeping to one reply: keep the assistant message
    /// with `content` as a string and any `tool_calls`, then answer each call.
    fn restloop_record(messages: &mut Vec<Value>, reply: &Value, output: &str) -> bool {
        let message = &reply["choices"][0]["message"];
        let text = message["content"].as_str().unwrap_or("");
        let calls = message["tool_calls"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut keep = json!({"role": "assistant", "content": text});
        if let Some(reasoning) = message.get("reasoning_content").filter(|r| !r.is_null()) {
            keep["reasoning_content"] = reasoning.clone();
        }
        if !calls.is_empty() {
            keep["tool_calls"] = json!(calls);
        }
        messages.push(keep);
        for call in &calls {
            messages.push(json!({"role": "tool", "tool_call_id": call["id"], "content": output}));
        }
        !calls.is_empty()
    }

    async fn restloop_step(messages: &[Value], generated: &str) -> (Value, String) {
        let engine = Arc::new(FakeEngine::healthy(vec![
            text_event(generated, 11, 1, 3),
            EngineEvent::Finished(metrics(FinishReason::Stop)),
        ]));
        let response = router(engine.clone())
            .oneshot(chat_request(&json!({
                "model": "test-model",
                "messages": messages,
                "tools": [shell_tool()]
            })))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let request = engine.last_request.lock().unwrap().take().unwrap();
        let EngineInput::Agent { prompt, constraint } = request.input else {
            panic!("RestLoop requests use the native agent renderer");
        };
        assert_eq!(prompt.directive, AgentDirective::Auto);
        assert!(constraint.is_some());
        let rendered = crate::models::qwen3::format_qwen3_agent_prompt(
            crate::agent::AgentTemplateProfile::Qwen3Current,
            &prompt,
            request.chat_template,
        )
        .unwrap();
        (body_json(response).await, rendered)
    }

    #[tokio::test]
    async fn restloop_multi_step_requests_round_trip_through_auto_tool_choice() {
        let mut messages = vec![
            json!({"role": "system", "content": "You are a coding agent."}),
            json!({"role": "user", "content": "What does a.txt say?"}),
        ];

        // Turn 1: reasoning, then a call.
        let (reply, rendered) = restloop_step(
            &messages,
            &format!("<think>\nList first.\n</think>\n\n{SHELL_CALL}"),
        )
        .await;
        assert!(rendered.ends_with(
            "<|im_start|>user\nWhat does a.txt say?<|im_end|>\n<|im_start|>assistant\n"
        ));
        assert!(restloop_record(&mut messages, &reply, "exit: 0\na.txt\n"));
        assert_eq!(messages[2]["content"], "<think>\nList first.\n</think>");

        // Turn 2: the replayed call reproduces the model's own turn, then a
        // second call with visible text before it.
        let (reply, rendered) = restloop_step(
            &messages,
            "Reading it.\n<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": \"cat a.txt\"}}\n</tool_call>",
        )
        .await;
        assert!(rendered.ends_with(concat!(
            "<|im_start|>assistant\n<think>\nList first.\n</think>\n\n",
            "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\":\"ls\"}}\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\nexit: 0\na.txt\n\n</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n"
        )));
        assert_eq!(reply["choices"][0]["message"]["content"], "Reading it.");
        assert!(restloop_record(&mut messages, &reply, "exit: 0\nhi\n"));

        // Turn 3: the final answer has no call.
        let (reply, rendered) =
            restloop_step(&messages, "<think>\nDone.\n</think>\n\na.txt says hi.").await;
        assert!(rendered.contains(concat!(
            "<|im_start|>assistant\nReading it.\n<tool_call>\n",
            "{\"name\": \"shell\", \"arguments\": {\"command\":\"cat a.txt\"}}\n</tool_call><|im_end|>\n"
        )));
        assert_eq!(reply["choices"][0]["finish_reason"], "stop");
        assert!(!restloop_record(&mut messages, &reply, ""));

        // An interactive follow-up continues the same conversation.
        messages.push(json!({"role": "user", "content": "Thanks."}));
        let (reply, rendered) = restloop_step(&messages, "You're welcome.").await;
        assert!(
            rendered.contains("a.txt says hi.<|im_end|>\n<|im_start|>user\nThanks.<|im_end|>\n")
        );
        assert_eq!(reply["choices"][0]["message"]["content"], "You're welcome.");
        assert_eq!(messages.len(), 8);
    }

    #[tokio::test]
    async fn poisoned_health_and_unavailable_admission_return_503() {
        let mut fake = FakeEngine::healthy(Vec::new());
        fake.status = EngineStatus::Poisoned;
        fake.submit_error = Some(SubmitError::Unavailable(EngineStatus::Poisoned));
        let engine = Arc::new(fake);
        let app = router(engine);

        let health = app
            .clone()
            .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::SERVICE_UNAVAILABLE);
        let health = body_json(health).await;
        assert_eq!(health["status"], "error");
        assert_eq!(health["worker"], "poisoned");

        let response = app.oneshot(request(false, false)).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body_json(response).await["error"]["type"], "server_error");
    }

    #[tokio::test]
    async fn malformed_json_never_reaches_the_engine() {
        let engine = Arc::new(FakeEngine::healthy(Vec::new()));
        let request = Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"model": }"#))
            .unwrap();
        let response = router(engine.clone()).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "invalid_json");
        assert_eq!(engine.submit_count.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn unsupported_json_schema_never_reaches_the_engine() {
        let engine = Arc::new(FakeEngine::healthy(Vec::new()));
        let request = Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "return JSON"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer",
                            "strict": true,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string", "minLength": 1}
                                },
                                "required": ["answer"],
                                "additionalProperties": false
                            }
                        }
                    }
                })
                .to_string(),
            ))
            .unwrap();
        let response = router(engine.clone()).oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body["error"]["param"], "response_format.json_schema.schema");
        assert_eq!(body["error"]["code"], "invalid_value");
        assert_eq!(engine.submit_count.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn streaming_preflight_failure_returns_http_error_before_sse_starts() {
        let mut fake = FakeEngine::healthy(Vec::new());
        fake.preflight_error = Some(EnginePreflightError {
            message: "prompt exceeds context".to_owned(),
            kind: EngineFailureKind::InvalidRequest,
        });
        let response = router(Arc::new(fake))
            .oneshot(request(true, false))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        let body = body_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "prompt exceeds context");
    }

    #[tokio::test]
    async fn resource_exhausted_preflight_returns_503_before_any_response_mode_starts() {
        for streaming in [false, true] {
            let mut fake = FakeEngine::healthy(Vec::new());
            fake.preflight_error = Some(EnginePreflightError {
                message: "request exceeds shared KV capacity".to_owned(),
                kind: EngineFailureKind::ResourceExhausted,
            });
            let response = router(Arc::new(fake))
                .oneshot(request(streaming, false))
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
            let body = body_json(response).await;
            assert_eq!(body["error"]["type"], "server_error");
            assert_eq!(
                body["error"]["message"],
                "request exceeds shared KV capacity"
            );
        }
    }

    #[tokio::test]
    async fn worker_failures_preserve_request_vs_fatal_error_types() {
        for (kind, status, error_type) in [
            (
                EngineFailureKind::InvalidRequest,
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                EngineFailureKind::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
            ),
            (
                EngineFailureKind::Fatal,
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
            ),
        ] {
            let app = router(Arc::new(FakeEngine::healthy(vec![EngineEvent::Failed {
                message: "failed".to_owned(),
                kind,
            }])));
            let response = app.oneshot(request(false, false)).await.unwrap();
            assert_eq!(response.status(), status);
            assert_eq!(body_json(response).await["error"]["type"], error_type);
        }

        let response = router(Arc::new(FakeEngine::healthy(Vec::new())))
            .oneshot(request(false, false))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn non_streaming_resource_exhaustion_is_a_request_scoped_503() {
        let app = router(Arc::new(FakeEngine::healthy(vec![EngineEvent::Failed {
            message: "shared KV pool is full".to_owned(),
            kind: EngineFailureKind::ResourceExhausted,
        }])));
        let response = app.oneshot(request(false, false)).await.unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = body_json(response).await;
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["message"], "shared KV pool is full");
    }

    #[tokio::test]
    async fn sse_worker_failure_emits_error_then_done() {
        let app = router(Arc::new(FakeEngine::healthy(vec![EngineEvent::Failed {
            message: "invalid prompt".to_owned(),
            kind: EngineFailureKind::InvalidRequest,
        }])));
        let response = app.oneshot(request(true, false)).await.unwrap();
        let body = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        let error = body.find("invalid_request_error").unwrap();
        let done = body.find("data: [DONE]").unwrap();
        assert!(error < done);
        assert!(!body.contains("\"finish_reason\":\"stop\""));
    }

    #[tokio::test]
    async fn sse_resource_exhaustion_emits_server_error_then_done() {
        let app = router(Arc::new(FakeEngine::healthy(vec![EngineEvent::Failed {
            message: "shared KV pool is full".to_owned(),
            kind: EngineFailureKind::ResourceExhausted,
        }])));
        let response = app.oneshot(request(true, false)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        let error = body.find("\"type\":\"server_error\"").unwrap();
        let message = body.find("shared KV pool is full").unwrap();
        let done = body.find("data: [DONE]").unwrap();
        assert!(error < done && message < done);
        assert!(!body.contains("invalid_request_error"));
    }

    #[tokio::test]
    async fn dropping_a_started_sse_body_cancels_the_worker_ticket() {
        let engine = Arc::new(FakeEngine::healthy(vec![
            text_event("hello", 11, 1, 3),
            EngineEvent::Finished(metrics(FinishReason::Stop)),
        ]));
        let app = router(engine.clone());
        let response = app.oneshot(request(true, false)).await.unwrap();
        let mut body = response.into_body();
        let mut received = String::new();
        while !received.contains("\"content\":\"hello\"") {
            let frame = body.frame().await.unwrap().unwrap();
            if let Some(data) = frame.data_ref() {
                received.push_str(std::str::from_utf8(data).unwrap());
            }
        }
        drop(body);
        assert!(
            engine
                .last_cancelled
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .load(Ordering::Acquire)
        );
    }
}
