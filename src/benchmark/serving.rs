//! Externally observed `MacLLM` serving benchmark execution.
//!
//! Requests enter through the same bounded scheduler used by the HTTP server,
//! but the benchmark submits the manifest's already verified token IDs. This
//! keeps chat-template and tokenizer work from silently changing the workload.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use futures_util::{StreamExt, stream::FuturesUnordered};
use serde::Serialize;

use crate::{
    backend::{DTypeKind, DeviceKind},
    engine::{
        EngineConfig, EngineEvent, EngineHandle, EngineInput, EngineRequest, EngineTicket,
        SubmitError,
    },
    generation::{FinishReason, GenerationMetrics},
    runtime::ModelLoadConfig,
    sampling::SamplingConfig,
};

use super::{
    BenchmarkDType, BenchmarkDevice, ImplementationKind, MEMORY_SAMPLING_PERIOD_MS, MemoryResult,
    PrefixCacheMode, ProcessMemoryObservation, ProcessMemoryPeak, ProcessMemorySampler,
    REQUIRED_CONCURRENCY, RequestSample, ServingResult, VerifiedScoreboard, aggregate_throughput,
    current_process_memory, summarize, token_ids_sha256,
};

/// `MacLLM` serving metrics and process-scoped memory evidence from one engine.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MacllmServingExecution {
    pub measurement_boundary: String,
    pub loaded_memory: ProcessMemoryObservation,
    pub serving_results: Vec<ServingResult>,
    pub memory_results: Vec<MemoryResult>,
}

/// Build the deterministic scheduler/model configuration used at one measured
/// concurrency. V1 serving evidence is deliberately cold: cache-enabled modes
/// need additional manifest fields and reset semantics before they are valid.
pub fn build_macllm_serving_engine_config(
    verified: &VerifiedScoreboard,
    concurrency: usize,
) -> Result<EngineConfig> {
    verified.manifest.validate()?;
    ensure!(
        REQUIRED_CONCURRENCY.contains(&concurrency),
        "MacLLM serving concurrency must be one of {REQUIRED_CONCURRENCY:?}"
    );
    ensure_verified_prompt(verified)?;
    ensure!(
        verified.manifest.workload.prefix_cache == PrefixCacheMode::Disabled,
        "MacLLM serving scoreboard v1 requires prefix_cache=disabled"
    );

    let runtime = &verified.manifest.runtime;
    let mut model = ModelLoadConfig::new(
        verified.manifest.model.path.clone(),
        match runtime.device {
            BenchmarkDevice::Cpu => DeviceKind::Cpu,
            BenchmarkDevice::Metal => DeviceKind::Metal,
        },
        Some(match runtime.dtype {
            BenchmarkDType::F32 => DTypeKind::F32,
            BenchmarkDType::F16 => DTypeKind::F16,
            BenchmarkDType::Bf16 => DTypeKind::Bf16,
        }),
    );
    model.kv_cache_block_tokens = runtime.kv_cache_block_tokens;
    model.kv_cache_slab_blocks = runtime.kv_cache_slab_blocks;
    model.kv_cache_capacity_tokens = Some(runtime.kv_cache_capacity_tokens);
    model.prefix_cache_capacity_tokens = 0;

    ensure_shared_kv_capacity(verified, concurrency)?;
    let scheduler_token_budget = runtime
        .prefill_token_quantum
        .checked_mul(concurrency)
        .context("MacLLM serving scheduler token budget overflow")?;
    let event_capacity = verified
        .manifest
        .workload
        .decode_steps
        .checked_add(2)
        .context("MacLLM serving event capacity overflow")?;
    let served_model = format!(
        "{}@{}",
        verified.manifest.model.repository, verified.manifest.model.revision
    );
    let mut config = EngineConfig::new(model, served_model);
    config.queue_capacity = concurrency;
    config.max_active_sequences = concurrency;
    config.scheduler_token_budget = scheduler_token_budget;
    config.prefill_token_quantum = runtime.prefill_token_quantum;
    // One visible fragment per token, one tokenizer flush, and one terminal
    // event can be retained without applying benchmark-client backpressure.
    config.event_capacity = event_capacity;
    Ok(config)
}

/// Load one eight-slot engine and measure complete request waves at widths
/// 1/4/8. Every request timer starts immediately before `EngineHandle::submit`;
/// TTFT ends at the first externally received `Text`, and latency at `Finished`.
pub async fn run_macllm_serving(verified: &VerifiedScoreboard) -> Result<MacllmServingExecution> {
    verified.manifest.validate()?;
    ensure_verified_prompt(verified)?;
    ensure!(
        verified.manifest.workload.prefix_cache == PrefixCacheMode::Disabled,
        "MacLLM serving scoreboard v1 requires prefix_cache=disabled"
    );

    let maximum_concurrency = *REQUIRED_CONCURRENCY
        .last()
        .context("required serving concurrency list is empty")?;
    let config = build_macllm_serving_engine_config(verified, maximum_concurrency)?;
    let handle = EngineHandle::start(config).context("failed to start MacLLM serving engine")?;
    ensure!(
        handle.metadata().family == verified.manifest.model.architecture,
        "MacLLM serving adapter family changed after scoreboard verification"
    );
    let loaded_memory = current_process_memory()?;
    let spec = ServingRunSpec::from_verified(verified);
    let mut serving_results = Vec::with_capacity(REQUIRED_CONCURRENCY.len());
    let mut memory_results = Vec::with_capacity(REQUIRED_CONCURRENCY.len());
    for concurrency in REQUIRED_CONCURRENCY {
        let observed =
            run_engine_width_inner(&handle, &spec, concurrency, Some(std::process::id()))
                .await
                .with_context(|| {
                    format!("MacLLM serving concurrency {concurrency} benchmark failed")
                })?;
        let peak = observed
            .memory_peak
            .context("MacLLM serving memory sampler returned no peak")?;
        serving_results.push(observed.result);
        memory_results.push(MemoryResult {
            implementation: ImplementationKind::Macllm,
            concurrency,
            loaded_steady_bytes: loaded_memory.physical_footprint_bytes,
            peak_physical_footprint_bytes: peak.physical_footprint_bytes,
            peak_resident_bytes: peak.resident_bytes,
            sampling_period_ms: MEMORY_SAMPLING_PERIOD_MS,
            measurement_boundary: concat!(
                "current-state proc_pid_rusage samples for the scoreboard CLI process; loaded ",
                "steady is sampled after one eight-slot engine loads and before serving warmups; ",
                "the width peak starts after that width's warmup waves and ends after its final ",
                "measured wave; macOS lifetime maximum is intentionally excluded; widths reuse ",
                "the same loaded engine"
            )
            .to_owned(),
        });
    }
    Ok(MacllmServingExecution {
        measurement_boundary: concat!(
            "one in-process eight-slot MacLLM engine; exact manifest token IDs enter the bounded ",
            "scheduler directly; complete report execution runs this stage before the MacLLM ",
            "kernel stage so model memory from that later stage cannot contaminate loaded steady"
        )
        .to_owned(),
        loaded_memory,
        serving_results,
        memory_results,
    })
}

#[derive(Debug, Clone)]
struct ServingRunSpec {
    prompt_token_ids: Vec<u32>,
    completion_tokens: usize,
    sampling: SamplingConfig,
    ignore_stop_tokens: bool,
    warmup_waves: usize,
    measured_requests: usize,
}

impl ServingRunSpec {
    fn from_verified(verified: &VerifiedScoreboard) -> Self {
        let workload = &verified.manifest.workload;
        Self {
            prompt_token_ids: verified.prompt_token_ids.clone(),
            completion_tokens: workload.decode_steps,
            sampling: SamplingConfig {
                temperature: workload.sampling.temperature,
                top_k: workload.sampling.top_k,
                top_p: workload.sampling.top_p,
                repetition_penalty: workload.sampling.repetition_penalty,
                seed: workload.sampling.seed,
            },
            ignore_stop_tokens: workload.sampling.ignore_stop_tokens,
            warmup_waves: workload.warmup_waves,
            measured_requests: workload.samples_per_concurrency,
        }
    }

    fn request(&self) -> EngineRequest {
        EngineRequest {
            input: EngineInput::TokenIds(self.prompt_token_ids.clone()),
            sampling: self.sampling,
            max_new_tokens: self.completion_tokens,
            ignore_stop_tokens: self.ignore_stop_tokens,
            stop: Vec::new(),
            chat_template: crate::chat::ChatTemplateOptions::default(),
        }
    }
}

trait ServingSubmitter {
    fn submit_serving(
        &self,
        request: EngineRequest,
    ) -> std::result::Result<EngineTicket, SubmitError>;
}

impl ServingSubmitter for EngineHandle {
    fn submit_serving(
        &self,
        request: EngineRequest,
    ) -> std::result::Result<EngineTicket, SubmitError> {
        self.submit(request)
    }
}

#[derive(Debug)]
struct ObservedRequest {
    request: usize,
    time_to_first_token: Duration,
    latency: Duration,
    completion_tokens: usize,
    completed_at: Instant,
}

#[derive(Debug)]
struct ObservedWave {
    submitted_at: Instant,
    completed_at: Instant,
    requests: Vec<ObservedRequest>,
}

#[cfg(test)]
async fn run_engine_width<S: ServingSubmitter>(
    submitter: &S,
    spec: &ServingRunSpec,
    concurrency: usize,
) -> Result<ServingResult> {
    Ok(run_engine_width_inner(submitter, spec, concurrency, None)
        .await?
        .result)
}

struct ServingWidthExecution {
    result: ServingResult,
    memory_peak: Option<ProcessMemoryPeak>,
}

async fn run_engine_width_inner<S: ServingSubmitter>(
    submitter: &S,
    spec: &ServingRunSpec,
    concurrency: usize,
    memory_pid: Option<u32>,
) -> Result<ServingWidthExecution> {
    ensure!(concurrency > 0, "serving concurrency must be positive");
    ensure!(
        spec.measured_requests.is_multiple_of(concurrency),
        "serving request count must form complete concurrency waves"
    );

    for warmup in 0..spec.warmup_waves {
        let _discarded = execute_wave(submitter, spec, concurrency, warmup * concurrency).await?;
    }
    let memory_sampler = memory_pid.map(ProcessMemorySampler::start).transpose()?;

    let wave_count = spec.measured_requests / concurrency;
    let mut samples = Vec::with_capacity(spec.measured_requests);
    let mut measurement_start = None;
    let mut measurement_end = None;
    for wave in 0..wave_count {
        let first_request = wave
            .checked_mul(concurrency)
            .context("serving request ID overflow")?;
        let observed = execute_wave(submitter, spec, concurrency, first_request).await?;
        measurement_start.get_or_insert(observed.submitted_at);
        measurement_end = Some(observed.completed_at);
        for request in observed.requests {
            samples.push(RequestSample {
                wave,
                request: request.request,
                prompt_tokens: spec.prompt_token_ids.len(),
                completion_tokens: request.completion_tokens,
                time_to_first_token_ms: millis(request.time_to_first_token),
                latency_ms: millis(request.latency),
            });
        }
    }

    let measurement_start = measurement_start.context("serving run retained no start time")?;
    let measurement_end = measurement_end.context("serving run retained no completion time")?;
    let wall_time = measurement_end
        .checked_duration_since(measurement_start)
        .context("serving completion preceded first submission")?;
    ensure!(!wall_time.is_zero(), "serving wall time must be positive");
    let total_completion_tokens = samples.iter().try_fold(0_u64, |total, sample| {
        total
            .checked_add(sample.completion_tokens as u64)
            .context("serving completion-token total overflow")
    })?;
    let ttft = samples
        .iter()
        .map(|sample| sample.time_to_first_token_ms)
        .collect::<Vec<_>>();
    let latency = samples
        .iter()
        .map(|sample| sample.latency_ms)
        .collect::<Vec<_>>();

    let result = ServingResult {
        implementation: ImplementationKind::Macllm,
        concurrency,
        request_count: samples.len(),
        total_completion_tokens,
        wall_time_ms: millis(wall_time),
        aggregate_output_tokens_per_second: aggregate_throughput(
            total_completion_tokens,
            wall_time,
        )?,
        time_to_first_token_ms: summarize(&ttft)?,
        latency_ms: summarize(&latency)?,
        samples,
    };
    let memory_peak = memory_sampler
        .map(ProcessMemorySampler::finish)
        .transpose()?;
    Ok(ServingWidthExecution {
        result,
        memory_peak,
    })
}

async fn execute_wave<S: ServingSubmitter>(
    submitter: &S,
    spec: &ServingRunSpec,
    concurrency: usize,
    first_request: usize,
) -> Result<ObservedWave> {
    let mut pending = FuturesUnordered::new();
    let mut first_submission = None;
    for offset in 0..concurrency {
        let request_id = first_request
            .checked_add(offset)
            .context("serving request ID overflow")?;
        let submitted_at = Instant::now();
        first_submission.get_or_insert(submitted_at);
        let ticket = submitter
            .submit_serving(spec.request())
            .map_err(|error| anyhow::anyhow!("request {request_id} submission failed: {error}"))?;
        pending.push(observe_ticket(
            ticket,
            submitted_at,
            request_id,
            spec.prompt_token_ids.len(),
            spec.completion_tokens,
        ));
    }

    let mut requests = Vec::with_capacity(concurrency);
    let mut completed_at = None;
    while let Some(observation) = pending.next().await {
        let observation = observation?;
        completed_at = Some(
            completed_at.map_or(observation.completed_at, |current: Instant| {
                current.max(observation.completed_at)
            }),
        );
        requests.push(observation);
    }
    requests.sort_unstable_by_key(|request| request.request);
    ensure!(
        requests.len() == concurrency,
        "serving wave lost request observations"
    );
    Ok(ObservedWave {
        submitted_at: first_submission.context("serving wave submitted no requests")?,
        completed_at: completed_at.context("serving wave completed no requests")?,
        requests,
    })
}

async fn observe_ticket(
    mut ticket: EngineTicket,
    submitted_at: Instant,
    request: usize,
    expected_prompt_tokens: usize,
    expected_completion_tokens: usize,
) -> Result<ObservedRequest> {
    ticket.wait_for_preflight().await.map_err(|error| {
        anyhow::anyhow!(
            "request {request} preflight failed ({:?}): {}",
            error.kind,
            error.message
        )
    })?;
    let mut time_to_first_token = None;
    loop {
        match ticket.recv().await {
            Some(EngineEvent::Text { .. }) => {
                time_to_first_token.get_or_insert_with(|| submitted_at.elapsed());
            }
            Some(EngineEvent::Finished(metrics)) => {
                validate_finished_metrics(
                    request,
                    &metrics,
                    expected_prompt_tokens,
                    expected_completion_tokens,
                )?;
                let completed_at = Instant::now();
                let latency = completed_at
                    .checked_duration_since(submitted_at)
                    .context("serving completion preceded request submission")?;
                let time_to_first_token = time_to_first_token.with_context(|| {
                    format!("request {request} finished without an externally visible Text event")
                })?;
                ensure!(
                    latency >= time_to_first_token,
                    "request {request} latency is shorter than TTFT"
                );
                return Ok(ObservedRequest {
                    request,
                    time_to_first_token,
                    latency,
                    completion_tokens: metrics.generated_tokens,
                    completed_at,
                });
            }
            Some(EngineEvent::Failed { message, kind }) => {
                bail!("request {request} failed ({kind:?}): {message}");
            }
            None => bail!("request {request} event stream closed before Finished"),
        }
    }
}

fn validate_finished_metrics(
    request: usize,
    metrics: &GenerationMetrics,
    expected_prompt_tokens: usize,
    expected_completion_tokens: usize,
) -> Result<()> {
    ensure!(
        metrics.prompt_tokens == expected_prompt_tokens,
        "request {request} engine prompt count changed: expected {expected_prompt_tokens}, found {}",
        metrics.prompt_tokens
    );
    ensure!(
        metrics.cached_prompt_tokens == 0,
        "request {request} unexpectedly restored {} cached prompt tokens in a cold scoreboard",
        metrics.cached_prompt_tokens
    );
    ensure!(
        metrics.generated_tokens == expected_completion_tokens,
        "request {request} emitted {} completion tokens; expected {expected_completion_tokens}",
        metrics.generated_tokens
    );
    ensure!(
        metrics.finish_reason == FinishReason::Length,
        "request {request} ended with {:?}; fixed-length scoreboard requires Length",
        metrics.finish_reason
    );
    Ok(())
}

fn ensure_verified_prompt(verified: &VerifiedScoreboard) -> Result<()> {
    ensure!(
        verified.prompt_token_ids.len() == verified.manifest.workload.prompt.token_count,
        "verified scoreboard prompt token count changed before serving execution"
    );
    ensure!(
        token_ids_sha256(&verified.prompt_token_ids)
            == verified.manifest.workload.prompt.token_ids_sha256,
        "verified scoreboard prompt token fingerprint changed before serving execution"
    );
    Ok(())
}

fn ensure_shared_kv_capacity(verified: &VerifiedScoreboard, concurrency: usize) -> Result<()> {
    let runtime = &verified.manifest.runtime;
    let maximum_cached_per_request = verified
        .prompt_token_ids
        .len()
        .checked_add(verified.manifest.workload.decode_steps.saturating_sub(1))
        .context("MacLLM serving request KV requirement overflow")?;
    let blocks_per_request = maximum_cached_per_request.div_ceil(runtime.kv_cache_block_tokens);
    let required_blocks = blocks_per_request
        .checked_mul(concurrency)
        .context("MacLLM serving shared KV requirement overflow")?;
    let available_blocks = runtime
        .kv_cache_capacity_tokens
        .div_ceil(runtime.kv_cache_block_tokens);
    ensure!(
        required_blocks <= available_blocks,
        "MacLLM serving concurrency {concurrency} requires {required_blocks} KV blocks, but the manifest provides {available_blocks}"
    );
    Ok(())
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use tokio::sync::mpsc;

    use super::*;

    struct FakeSubmitter {
        expected_prompt: Vec<u32>,
        prompt_tokens: usize,
        completion_tokens: usize,
        submissions: AtomicUsize,
        senders: Mutex<VecDeque<mpsc::Sender<EngineEvent>>>,
    }

    impl FakeSubmitter {
        fn new(prompt: Vec<u32>, completion_tokens: usize) -> Self {
            Self {
                prompt_tokens: prompt.len(),
                expected_prompt: prompt,
                completion_tokens,
                submissions: AtomicUsize::new(0),
                senders: Mutex::new(VecDeque::new()),
            }
        }
    }

    impl ServingSubmitter for FakeSubmitter {
        fn submit_serving(
            &self,
            request: EngineRequest,
        ) -> std::result::Result<EngineTicket, SubmitError> {
            let EngineInput::TokenIds(tokens) = request.input else {
                panic!("serving benchmark submitted a chat request")
            };
            assert_eq!(tokens, self.expected_prompt);
            assert!(request.ignore_stop_tokens);
            assert_eq!(request.max_new_tokens, self.completion_tokens);
            assert!(request.stop.is_empty());
            self.submissions.fetch_add(1, Ordering::Relaxed);

            let (sender, _cancelled, ticket) = EngineTicket::channel(4, Ok(()));
            sender
                .try_send(EngineEvent::Text {
                    content: "x".to_owned(),
                    token: Some(crate::engine::EngineTokenEvidence {
                        token_id: 1,
                        n_decoded: 1,
                        prompt_tokens: self.prompt_tokens,
                    }),
                })
                .expect("fake Text event fits");
            sender
                .try_send(EngineEvent::Finished(GenerationMetrics {
                    prompt_tokens: self.prompt_tokens,
                    cached_prompt_tokens: 0,
                    prefix_lookup_attach_time: Duration::ZERO,
                    generated_tokens: self.completion_tokens,
                    decode_steps: self.completion_tokens.saturating_sub(1),
                    time_to_first_token: Duration::ZERO,
                    suffix_prefill_time: Duration::ZERO,
                    first_token_model_time: Duration::ZERO,
                    prefill_time: Duration::ZERO,
                    decode_time: Duration::ZERO,
                    total_time: Duration::ZERO,
                    finish_reason: FinishReason::Length,
                }))
                .expect("fake Finished event fits");
            self.senders
                .lock()
                .expect("fake sender lock is healthy")
                .push_back(sender);
            Ok(ticket)
        }
    }

    #[tokio::test]
    async fn complete_waves_use_exact_raw_tokens_and_strict_aggregates() -> Result<()> {
        let spec = ServingRunSpec {
            prompt_token_ids: vec![7, 8, 9],
            completion_tokens: 5,
            sampling: SamplingConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
                seed: 42,
            },
            ignore_stop_tokens: true,
            warmup_waves: 1,
            measured_requests: 8,
        };
        let submitter = FakeSubmitter::new(spec.prompt_token_ids.clone(), spec.completion_tokens);

        let result = run_engine_width(&submitter, &spec, 4).await?;

        assert_eq!(submitter.submissions.load(Ordering::Relaxed), 12);
        assert_eq!(result.implementation, ImplementationKind::Macllm);
        assert_eq!(result.concurrency, 4);
        assert_eq!(result.request_count, 8);
        assert_eq!(result.total_completion_tokens, 40);
        assert_eq!(result.samples.len(), 8);
        assert_eq!(
            result
                .samples
                .iter()
                .filter(|sample| sample.wave == 0)
                .count(),
            4
        );
        assert_eq!(
            result
                .samples
                .iter()
                .filter(|sample| sample.wave == 1)
                .count(),
            4
        );
        assert_eq!(result.time_to_first_token_ms.sample_count, 8);
        assert_eq!(result.latency_ms.sample_count, 8);
        assert!(result.wall_time_ms > 0.0);
        assert!(result.aggregate_output_tokens_per_second > 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn missing_text_event_fails_closed_instead_of_faking_ttft() {
        let (sender, _cancelled, ticket) = EngineTicket::channel(2, Ok(()));
        sender
            .try_send(EngineEvent::Finished(GenerationMetrics {
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
                finish_reason: FinishReason::Length,
            }))
            .unwrap();
        let error = observe_ticket(ticket, Instant::now(), 0, 3, 2)
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("without an externally visible Text")
        );
    }
}
