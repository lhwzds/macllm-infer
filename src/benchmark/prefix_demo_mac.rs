//! Cache-enabled `MacLLM` execution for the deterministic 16K-prefix demo.
//!
//! This is deliberately separate from the cold scoreboard. It warms one exact
//! prompt prefix, submits all eight measured chats before draining any ticket,
//! and reports request-level timings plus worker-owned cache diagnostics.

use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use futures_util::{StreamExt, stream::FuturesUnordered};
use serde::Serialize;

use crate::{
    architecture::PrefixCacheDiagnostics,
    backend::{DTypeKind, DeviceKind},
    engine::{
        BenchmarkForwardKind, EngineBatchTrace, EngineConfig, EngineEvent, EngineHandle,
        EngineInput, EnginePrefixCacheSnapshot, EngineRequest, EngineRequestPolicy, EngineTicket,
    },
    generation::{FinishReason, GenerationMetrics},
    runtime::ModelLoadConfig,
    sampling::SamplingConfig,
    weights::ModelFiles,
};

use super::{
    DistributionSummary,
    prefix_demo::{
        DEFAULT_SHARED_PREFIX_TOKENS, MaterializedPrefixDemo, MaterializedPrefixPrompt,
        PREFIX_DEMO_TASK_COUNT, PreparedPrefixDemo, PreparedPrefixPrompt,
    },
    provenance::{sha256_bytes, token_ids_sha256, verify_file},
    summarize,
};

pub const MAC_PREFIX_DEMO_CONCURRENCY: usize = 8;
pub const MAC_PREFIX_DEMO_DECODE_TOKENS: usize = 32;
pub const MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET: usize = 1_024;
pub const MAC_PREFIX_DEMO_CACHE_CAPACITY_TOKENS: usize = 65_536;
/// Minimum number of independent eight-request waves required for percentile evidence.
pub const MAC_PREFIX_DEMO_MIN_MEASURED_WAVES: usize = 20;
/// Default measured-wave count for the public latency-sprint demo.
pub const MAC_PREFIX_DEMO_DEFAULT_MEASURED_WAVES: usize = 20;

const CACHE_QUIESCENCE_TIMEOUT: Duration = Duration::from_secs(2);
const BENCHMARK_TRACE_COMPLETION_TIMEOUT: Duration = Duration::from_secs(2);

/// Model and cache settings for one cache-enabled demo engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacPrefixDemoRunConfig {
    pub model_path: PathBuf,
    pub served_model: String,
    pub prefill_token_quantum: usize,
    pub kv_cache_block_tokens: usize,
    pub kv_cache_slab_blocks: usize,
    pub kv_cache_capacity_tokens: usize,
    pub prefix_cache_capacity_tokens: usize,
    pub measured_waves: usize,
}

impl MacPrefixDemoRunConfig {
    #[must_use]
    pub fn new(
        model_path: impl Into<PathBuf>,
        served_model: impl Into<String>,
        prefill_token_quantum: usize,
    ) -> Self {
        Self {
            model_path: model_path.into(),
            served_model: served_model.into(),
            prefill_token_quantum,
            kv_cache_block_tokens: ModelLoadConfig::DEFAULT_KV_CACHE_BLOCK_TOKENS,
            kv_cache_slab_blocks: ModelLoadConfig::DEFAULT_KV_CACHE_SLAB_BLOCKS,
            kv_cache_capacity_tokens: MAC_PREFIX_DEMO_CACHE_CAPACITY_TOKENS,
            prefix_cache_capacity_tokens: MAC_PREFIX_DEMO_CACHE_CAPACITY_TOKENS,
            measured_waves: MAC_PREFIX_DEMO_DEFAULT_MEASURED_WAVES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MacPrefixDemoConfigEvidence {
    pub model_path: PathBuf,
    pub served_model: String,
    pub device: String,
    pub dtype: String,
    pub queue_capacity: usize,
    pub max_active_sequences: usize,
    pub scheduler_token_budget: usize,
    pub prefill_token_quantum: usize,
    pub event_capacity: usize,
    pub kv_cache_block_tokens: usize,
    pub kv_cache_slab_blocks: usize,
    pub kv_cache_capacity_tokens: usize,
    pub prefix_cache_capacity_tokens: usize,
    pub seed_completion_tokens: usize,
    pub measured_completion_tokens: usize,
    pub measured_request_count: usize,
    pub measured_waves: usize,
    pub exact_cache_endpoint_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MacPrefixDemoIdentityEvidence {
    pub prepared_schema_version: u32,
    pub model_family: String,
    pub model_filename: String,
    pub model_sha256: String,
    pub repository_file_count: usize,
    pub repository_bundle_sha256: String,
    pub repository_payload_encoding: String,
    pub encoded_repository_bundle_sha256: String,
    pub selected_bundle_byte_count: u64,
    pub selected_bundle_char_count: usize,
    pub selected_bundle_sha256: String,
    pub common_prefix_token_ids_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixDemoUsageEvidence {
    pub prompt_tokens: usize,
    pub cached_prompt_tokens: usize,
    pub completion_tokens: usize,
    pub decode_steps: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixDemoRequestEvidence {
    pub name: String,
    pub task_index: Option<usize>,
    pub rendered_prompt_sha256: String,
    pub prompt_token_ids_sha256: String,
    pub text: String,
    pub time_to_first_token_ms: f64,
    pub latency_ms: f64,
    pub usage: PrefixDemoUsageEvidence,
    pub finish_reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixDemoAggregateEvidence {
    pub request_count: usize,
    pub total_completion_tokens: u64,
    pub wall_time_ms: f64,
    pub aggregate_output_tokens_per_second: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixCacheDiagnosticsEvidence {
    pub hits: u64,
    pub misses: u64,
    pub matched_tokens: u64,
    pub entries: usize,
    pub evictions: u64,
    pub resident_snapshot_blocks: usize,
    pub block_tokens: usize,
    pub physical_live_blocks: usize,
    pub physical_live_references: usize,
    pub physical_shared_blocks: usize,
    pub physical_max_blocks: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixCacheSnapshotEvidence {
    pub current: PrefixCacheDiagnosticsEvidence,
    pub active_sequences: usize,
    pub observation: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixCachePeakEvidence {
    pub active_sequences: usize,
    pub resident_snapshot_blocks: usize,
    pub block_tokens: usize,
    pub physical_live_blocks: usize,
    pub physical_live_references: usize,
    pub physical_shared_blocks: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixCacheCounterDeltaEvidence {
    pub hits: u64,
    pub misses: u64,
    pub matched_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MacPrefixDemoCacheEvidence {
    pub before_wave: PrefixCacheSnapshotEvidence,
    pub after_wave: PrefixCacheSnapshotEvidence,
    pub eight_way_first_admission_snapshot: PrefixCachePeakEvidence,
    pub measured_counter_delta: PrefixCacheCounterDeltaEvidence,
    pub active_request_prefix_page_references: usize,
    pub active_request_prefix_page_fanout: f64,
    pub active_request_prefix_reuse_coverage: f64,
    pub new_physical_live_blocks_at_admission: usize,
    pub atomic_pool_owner_density: f64,
    pub physical_reuse_boundary: String,
}

/// Non-overlapping worker-side sampled spans for one eight-request wave.
///
/// Packed model forwards are recorded once for the whole wave, never copied to
/// every request. `single_step_decode_ms` therefore contains one sample per
/// packed decode quantum rather than eight correlated copies. These spans are
/// intentionally not a complete decomposition of external wall time: scheduler
/// preparation, sampling, token decoding, and event delivery remain outside
/// the named buckets.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MacPrefixDemoPhaseTimingEvidence {
    pub queue_admission_ms: f64,
    pub prefix_lookup_attach_ms: f64,
    pub cache_diagnostics_ms: f64,
    pub suffix_prefill_ms: f64,
    pub first_token_model_forward_ms: f64,
    pub single_step_decode_ms: Vec<f64>,
    pub single_step_decode_total_ms: f64,
    pub boundary: String,
}

/// Raw evidence for one consecutive eight-request wave.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MacPrefixDemoWaveEvidence {
    pub wave_index: usize,
    pub requests: Vec<PrefixDemoRequestEvidence>,
    pub aggregate: PrefixDemoAggregateEvidence,
    pub cache: MacPrefixDemoCacheEvidence,
    pub phases: MacPrefixDemoPhaseTimingEvidence,
}

/// Wave-level percentile evidence recomputed from all retained raw waves.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MacPrefixDemoDistributionEvidence {
    pub wave_max_time_to_first_token_ms_samples: Vec<f64>,
    pub wave_max_time_to_first_token_ms: DistributionSummary,
    pub wave_wall_time_ms_samples: Vec<f64>,
    pub wave_wall_time_ms: DistributionSummary,
    pub wave_aggregate_output_tokens_per_second_samples: Vec<f64>,
    pub wave_aggregate_output_tokens_per_second: DistributionSummary,
    pub queue_admission_ms_samples: Vec<f64>,
    pub queue_admission_ms: DistributionSummary,
    pub prefix_lookup_attach_ms_samples: Vec<f64>,
    pub prefix_lookup_attach_ms: DistributionSummary,
    pub cache_diagnostics_ms_samples: Vec<f64>,
    pub cache_diagnostics_ms: DistributionSummary,
    pub suffix_prefill_ms_samples: Vec<f64>,
    pub suffix_prefill_ms: DistributionSummary,
    pub first_token_model_forward_ms_samples: Vec<f64>,
    pub first_token_model_forward_ms: DistributionSummary,
    pub single_step_decode_total_ms_samples: Vec<f64>,
    pub single_step_decode_total_ms: DistributionSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MacPrefixDemoExecution {
    pub implementation: String,
    pub measurement_boundary: String,
    pub configuration: MacPrefixDemoConfigEvidence,
    pub identity: MacPrefixDemoIdentityEvidence,
    pub seed: PrefixDemoRequestEvidence,
    pub cold_cache: PrefixCacheSnapshotEvidence,
    pub after_seed_cache: PrefixCacheSnapshotEvidence,
    pub waves: Vec<MacPrefixDemoWaveEvidence>,
    pub distributions: MacPrefixDemoDistributionEvidence,
}

/// Run one cold seed followed by consecutive eight-request exact-16K waves.
pub async fn run_macllm_prefix_demo(
    config: &MacPrefixDemoRunConfig,
    prepared: &PreparedPrefixDemo,
    materialized: &MaterializedPrefixDemo,
) -> Result<MacPrefixDemoExecution> {
    prepared.validate()?;
    validate_materialized_demo(prepared, materialized)?;
    validate_run_config(config, prepared)?;
    let (engine_config, config_evidence) = build_engine_config(config, prepared)?;
    let handle = EngineHandle::start(engine_config)
        .context("failed to start cache-enabled MacLLM prefix-demo engine")?;
    validate_loaded_engine(&handle, prepared)?;

    let cold = require_quiescent_snapshot(&handle, "cold engine").await?;
    validate_cold_cache(&cold)?;

    let seed = run_seed(&handle, &prepared.seed, &materialized.seed).await?;
    let after_seed = require_quiescent_snapshot(&handle, "after seed").await?;
    validate_seed_cache(&cold, &after_seed, prepared.requested_shared_tokens)?;

    let mut waves = Vec::with_capacity(config.measured_waves);
    for wave_index in 0..config.measured_waves {
        let before_wave = require_quiescent_snapshot(&handle, "before measured wave").await?;
        let measured = run_measured_batch(&handle, prepared, materialized).await?;
        let measured_end = measured
            .requests
            .iter()
            .map(|request| request.completed_at)
            .max()
            .context("measured prefix-demo batch retained no completion time")?;
        let after_wave = require_quiescent_snapshot(&handle, "after measured wave").await?;
        let aggregate =
            aggregate_requests(&measured.requests, measured.batch_released_at, measured_end)?;
        let counter_delta = validate_measured_cache_delta(
            &before_wave,
            &after_wave,
            prepared.requested_shared_tokens,
            measured.requests.len(),
        )?;
        let page_reuse = validate_eight_way_page_reuse(
            &before_wave,
            &measured.trace,
            prepared.requested_shared_tokens,
            prepared.kv_cache_block_tokens,
        )?;
        let phases = phase_timing_evidence(&measured.trace)?;
        waves.push(MacPrefixDemoWaveEvidence {
            wave_index,
            requests: measured
                .requests
                .into_iter()
                .map(ObservedRequest::into_evidence)
                .collect(),
            aggregate,
            cache: MacPrefixDemoCacheEvidence {
                before_wave: snapshot_evidence(&before_wave, "before measured wave")?,
                after_wave: snapshot_evidence(&after_wave, "after measured wave")?,
                eight_way_first_admission_snapshot: page_reuse.first_admission,
                measured_counter_delta: counter_delta,
                active_request_prefix_page_references: page_reuse
                    .active_request_prefix_page_references,
                active_request_prefix_page_fanout: page_reuse
                    .active_request_prefix_page_fanout,
                active_request_prefix_reuse_coverage: page_reuse
                    .active_request_prefix_reuse_coverage,
                new_physical_live_blocks_at_admission: page_reuse
                    .new_physical_live_blocks_at_admission,
                atomic_pool_owner_density: page_reuse.atomic_pool_owner_density,
                physical_reuse_boundary: concat!(
                    "this wave's single post-attach, pre-forward diagnostic is differenced against ",
                    "its immediately preceding quiescent cache snapshot. The reference delta must ",
                    "equal eight times the 1,024 prefix pages while the physical-live-block delta ",
                    "remains zero. measured requests cannot publish new prompt snapshots, preserving ",
                    "the exact 16K endpoint across consecutive waves"
                )
                .to_owned(),
            },
            phases,
        });
    }
    verify_prepared_model(config, prepared)
        .context("prefix-demo model changed during MacLLM execution")?;
    let distributions = derive_macllm_distributions(&waves)?;

    Ok(MacPrefixDemoExecution {
        implementation: "macllm".to_owned(),
        measurement_boundary: concat!(
            "one Metal/F32 EngineHandle consuming the verified materialized token IDs directly; ",
            "model load, prompt preparation, token-input construction, the cold seed, and cache ",
            "quiescence checks and staging all eight queue sends are excluded from each measured wave; ",
            "all request timers share the instant immediately before the atomic admission gate is ",
            "released, after every send succeeds; TTFT ends at the first externally received ",
            "non-empty Text event, latency ends at Finished, and aggregate wall time spans that ",
            "batch release through the last Finished; measured requests do not publish prompt ",
            "snapshots, so all consecutive waves retain the exact seeded 16K endpoint; worker-side ",
            "benchmark phases are non-overlapping and packed forward time is recorded once per wave"
        )
        .to_owned(),
        configuration: config_evidence,
        identity: identity_evidence(prepared),
        seed: seed.into_evidence(),
        cold_cache: snapshot_evidence(&cold, "cold engine")?,
        after_seed_cache: snapshot_evidence(&after_seed, "after seed")?,
        waves,
        distributions,
    })
}

fn build_engine_config(
    config: &MacPrefixDemoRunConfig,
    prepared: &PreparedPrefixDemo,
) -> Result<(EngineConfig, MacPrefixDemoConfigEvidence)> {
    let event_capacity = MAC_PREFIX_DEMO_DECODE_TOKENS
        .checked_add(2)
        .context("prefix-demo event capacity overflow")?;
    let mut model = ModelLoadConfig::new(
        config.model_path.clone(),
        DeviceKind::Metal,
        Some(DTypeKind::F32),
    );
    model.kv_cache_block_tokens = config.kv_cache_block_tokens;
    model.kv_cache_slab_blocks = config.kv_cache_slab_blocks;
    model.kv_cache_capacity_tokens = Some(config.kv_cache_capacity_tokens);
    model.prefix_cache_capacity_tokens = config.prefix_cache_capacity_tokens;

    let mut engine = EngineConfig::new(model, config.served_model.clone());
    engine.queue_capacity = MAC_PREFIX_DEMO_CONCURRENCY;
    engine.max_active_sequences = MAC_PREFIX_DEMO_CONCURRENCY;
    engine.scheduler_token_budget = MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET;
    engine.prefill_token_quantum = config.prefill_token_quantum;
    engine.event_capacity = event_capacity;
    engine.capture_benchmark_trace = true;

    let evidence = MacPrefixDemoConfigEvidence {
        model_path: config.model_path.clone(),
        served_model: config.served_model.clone(),
        device: "metal".to_owned(),
        dtype: "f32".to_owned(),
        queue_capacity: engine.queue_capacity,
        max_active_sequences: engine.max_active_sequences,
        scheduler_token_budget: engine.scheduler_token_budget,
        prefill_token_quantum: engine.prefill_token_quantum,
        event_capacity,
        kv_cache_block_tokens: config.kv_cache_block_tokens,
        kv_cache_slab_blocks: config.kv_cache_slab_blocks,
        kv_cache_capacity_tokens: config.kv_cache_capacity_tokens,
        prefix_cache_capacity_tokens: config.prefix_cache_capacity_tokens,
        seed_completion_tokens: 1,
        measured_completion_tokens: MAC_PREFIX_DEMO_DECODE_TOKENS,
        measured_request_count: PREFIX_DEMO_TASK_COUNT,
        measured_waves: config.measured_waves,
        exact_cache_endpoint_tokens: prepared.requested_shared_tokens,
    };
    Ok((engine, evidence))
}

fn validate_run_config(
    config: &MacPrefixDemoRunConfig,
    prepared: &PreparedPrefixDemo,
) -> Result<()> {
    ensure!(
        prepared.requested_shared_tokens == DEFAULT_SHARED_PREFIX_TOKENS,
        "MacLLM public prefix demo requires an exact {DEFAULT_SHARED_PREFIX_TOKENS}-token endpoint"
    );
    ensure!(
        config.prefill_token_quantum == prepared.prefill_quantum,
        "execution prefill quantum differs from prepared prompt evidence"
    );
    ensure!(
        config.prefill_token_quantum > 0
            && config.prefill_token_quantum <= MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET,
        "prefix-demo prefill quantum must be in 1..={MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET}"
    );
    ensure!(
        prepared
            .requested_shared_tokens
            .is_multiple_of(config.prefill_token_quantum),
        "cache endpoint must align to the prefill publication quantum"
    );
    ensure!(
        config.kv_cache_block_tokens > 0 && config.kv_cache_slab_blocks > 0,
        "prefix-demo KV block and slab sizes must be positive"
    );
    ensure!(
        config.kv_cache_block_tokens == prepared.kv_cache_block_tokens,
        "execution KV block size differs from prepared LCP evidence"
    );
    ensure!(
        !config.served_model.trim().is_empty(),
        "prefix-demo served model name must not be empty"
    );
    ensure!(
        config.measured_waves > 0,
        "prefix-demo execution requires at least one measured wave"
    );
    let minimum_prefix_capacity = prepared
        .requested_shared_tokens
        .checked_mul(4)
        .context("minimum prefix-cache capacity overflow")?;
    ensure!(
        config.prefix_cache_capacity_tokens >= minimum_prefix_capacity,
        "prefix cache requires at least {minimum_prefix_capacity} tokens to retain the exact endpoint and nearby published snapshots"
    );
    verify_prepared_model(config, prepared)?;
    ensure_kv_capacity(config, prepared)
}

fn verify_prepared_model(
    config: &MacPrefixDemoRunConfig,
    prepared: &PreparedPrefixDemo,
) -> Result<()> {
    let files = ModelFiles::from_dir(&config.model_path)
        .context("failed to open prefix-demo model for identity verification")?;
    ensure!(
        files.weight_file_count() == 1,
        "prefix demo requires one exact model weight file"
    );
    let current = verify_file(
        files.weight_paths()[0],
        prepared.model_file.size_bytes,
        &prepared.model_file.sha256,
    )?;
    ensure!(
        current == prepared.model_file,
        "execution model path or filename differs from prepared evidence"
    );
    Ok(())
}

fn ensure_kv_capacity(
    config: &MacPrefixDemoRunConfig,
    prepared: &PreparedPrefixDemo,
) -> Result<()> {
    let block_tokens = config.kv_cache_block_tokens;
    let mut required_blocks = prepared.seed.prompt_token_count.div_ceil(block_tokens);
    for task in &prepared.tasks {
        ensure!(
            task.prompt_token_count > prepared.requested_shared_tokens,
            "task prompt must retain at least one uncached token after the shared endpoint"
        );
        let maximum_logical_len = task
            .prompt_token_count
            .checked_add(MAC_PREFIX_DEMO_DECODE_TOKENS.saturating_sub(1))
            .context("task KV length overflow")?;
        let private_tokens = maximum_logical_len
            .checked_sub(prepared.requested_shared_tokens)
            .context("task prompt is shorter than the cache endpoint")?;
        required_blocks = required_blocks
            .checked_add(private_tokens.div_ceil(block_tokens))
            .context("eight-request KV requirement overflow")?;
    }
    let available_blocks = config.kv_cache_capacity_tokens.div_ceil(block_tokens);
    ensure!(
        required_blocks <= available_blocks,
        "prefix demo requires {required_blocks} physical KV blocks, but configuration provides {available_blocks}"
    );
    Ok(())
}

fn validate_loaded_engine(handle: &EngineHandle, prepared: &PreparedPrefixDemo) -> Result<()> {
    ensure!(
        handle.metadata().family == prepared.model_family,
        "loaded model family differs from prepared prefix-demo evidence"
    );
    for prompt in std::iter::once(&prepared.seed).chain(&prepared.tasks) {
        let required_context = prompt
            .prompt_token_count
            .checked_add(MAC_PREFIX_DEMO_DECODE_TOKENS)
            .context("prefix-demo context requirement overflow")?;
        ensure!(
            required_context <= handle.metadata().max_context,
            "prompt `{}` requires {required_context} context tokens, but model declares {}",
            prompt.name,
            handle.metadata().max_context
        );
    }
    Ok(())
}

fn validate_materialized_demo(
    prepared: &PreparedPrefixDemo,
    materialized: &MaterializedPrefixDemo,
) -> Result<()> {
    ensure!(
        materialized.tasks.len() == prepared.tasks.len()
            && materialized.tasks.len() == PREFIX_DEMO_TASK_COUNT,
        "materialized prefix demo must contain exactly {PREFIX_DEMO_TASK_COUNT} tasks in prepared order"
    );
    validate_materialized_prompt(&prepared.seed, &materialized.seed)?;

    let shared_tokens = materialized
        .seed
        .token_ids
        .get(..prepared.requested_shared_tokens)
        .context("materialized seed is shorter than the prepared cache endpoint")?;
    ensure!(
        token_ids_sha256(shared_tokens) == prepared.common_prefix_token_ids_sha256,
        "materialized seed common-prefix digest differs from prepared evidence"
    );

    for (expected, actual) in prepared.tasks.iter().zip(&materialized.tasks) {
        validate_materialized_prompt(expected, actual)?;
        ensure!(
            longest_common_prefix(&materialized.seed.token_ids, &actual.token_ids)
                == prepared.requested_shared_tokens,
            "materialized prompt `{}` does not preserve the exact prepared token LCP",
            expected.name
        );
        ensure!(
            actual.token_ids.get(..prepared.requested_shared_tokens) == Some(shared_tokens),
            "materialized prompt `{}` changed inside the prepared common prefix",
            expected.name
        );
    }
    Ok(())
}

fn validate_materialized_prompt(
    expected: &PreparedPrefixPrompt,
    actual: &MaterializedPrefixPrompt,
) -> Result<()> {
    ensure!(
        actual.name == expected.name,
        "materialized prompt order or name changed: expected `{}`, found `{}`",
        expected.name,
        actual.name
    );
    ensure!(
        actual.rendered_prompt.len() == expected.rendered_prompt_byte_count
            && sha256_bytes(actual.rendered_prompt.as_bytes()) == expected.rendered_prompt_sha256,
        "materialized rendered prompt `{}` differs from prepared byte evidence",
        expected.name
    );
    ensure!(
        actual.token_ids.len() == expected.prompt_token_count
            && token_ids_sha256(&actual.token_ids) == expected.prompt_token_ids_sha256,
        "materialized token IDs for `{}` differ from prepared token evidence",
        expected.name
    );
    Ok(())
}

fn longest_common_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

async fn run_seed(
    handle: &EngineHandle,
    seed: &PreparedPrefixPrompt,
    materialized: &MaterializedPrefixPrompt,
) -> Result<ObservedRequest> {
    let (request, submitted_token_ids_sha256) = token_request(&materialized.token_ids, 1)?;
    let submitted_at = Instant::now();
    let ticket = handle
        .submit(request)
        .map_err(|error| anyhow::anyhow!("seed submission failed: {error}"))?;
    observe_ticket(ticket, submitted_at, seed, submitted_token_ids_sha256, 0, 1)
        .await
        .context("cache-warming seed failed")
}

async fn run_measured_batch(
    handle: &EngineHandle,
    prepared: &PreparedPrefixDemo,
    materialized: &MaterializedPrefixDemo,
) -> Result<ObservedBatch> {
    let mut requests = Vec::with_capacity(MAC_PREFIX_DEMO_CONCURRENCY);
    let mut submitted_token_digests = Vec::with_capacity(MAC_PREFIX_DEMO_CONCURRENCY);
    for materialized_task in &materialized.tasks {
        let (request, submitted_token_ids_sha256) =
            token_request(&materialized_task.token_ids, MAC_PREFIX_DEMO_DECODE_TOKENS)?;
        requests.push(request);
        submitted_token_digests.push(submitted_token_ids_sha256);
    }
    ensure!(
        requests.len() == MAC_PREFIX_DEMO_CONCURRENCY,
        "prefix-demo batch did not construct exactly eight requests"
    );
    let submission = handle
        .submit_batch_with_policy(
            requests
                .into_iter()
                .map(|request| (request, EngineRequestPolicy::benchmark_measured()))
                .collect(),
        )
        .map_err(|error| anyhow::anyhow!("eight-request atomic submission failed: {error}"))?;
    ensure!(
        submission.len() == MAC_PREFIX_DEMO_CONCURRENCY,
        "prefix-demo atomic submission did not retain all eight tickets"
    );
    let batch_released_at = submission.released_at();
    let (tickets, trace_handle) = submission.into_parts();
    let trace_handle = trace_handle.context(
        "prefix-demo engine did not enable its benchmark-only atomic-batch timing sidecar",
    )?;

    let mut pending = FuturesUnordered::new();
    for ((task, submitted_token_ids_sha256), ticket) in prepared
        .tasks
        .iter()
        .zip(submitted_token_digests)
        .zip(tickets)
    {
        pending.push(observe_ticket(
            ticket,
            batch_released_at,
            task,
            submitted_token_ids_sha256,
            prepared.requested_shared_tokens,
            MAC_PREFIX_DEMO_DECODE_TOKENS,
        ));
    }
    ensure!(
        pending.len() == MAC_PREFIX_DEMO_CONCURRENCY,
        "prefix-demo batch did not submit all eight requests before draining"
    );

    let mut observed = Vec::with_capacity(MAC_PREFIX_DEMO_CONCURRENCY);
    while let Some(result) = pending.next().await {
        observed.push(result?);
    }
    observed.sort_unstable_by_key(|request| request.task_index);
    ensure!(
        observed.len() == MAC_PREFIX_DEMO_CONCURRENCY,
        "prefix-demo batch lost a request observation"
    );
    let trace = wait_for_complete_trace(&trace_handle).await?;
    Ok(ObservedBatch {
        batch_released_at,
        requests: observed,
        trace,
    })
}

async fn wait_for_complete_trace(
    trace_handle: &crate::engine::EngineBatchTraceHandle,
) -> Result<EngineBatchTrace> {
    let deadline = Instant::now()
        .checked_add(BENCHMARK_TRACE_COMPLETION_TIMEOUT)
        .context("benchmark trace completion deadline overflow")?;
    loop {
        if let Some(trace) = trace_handle.try_read() {
            return Ok(trace);
        }
        ensure!(
            Instant::now() < deadline,
            "prefix-demo atomic-batch timing sidecar did not become complete after every request finished"
        );
        // `Finished` is sent before the worker removes and drops its active
        // slot. Let that final scheduler turn publish the terminal sidecar
        // state instead of racing it with the event observers.
        tokio::task::yield_now().await;
    }
}

fn token_request(token_ids: &[u32], max_new_tokens: usize) -> Result<(EngineRequest, String)> {
    ensure!(!token_ids.is_empty(), "prefix-demo token input is empty");
    let input_token_ids = token_ids.to_vec();
    let submitted_token_ids_sha256 = token_ids_sha256(&input_token_ids);
    Ok((
        EngineRequest {
            input: EngineInput::TokenIds(input_token_ids),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            sampling: SamplingConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
                seed: 42,
            },
            max_new_tokens,
            ignore_stop_tokens: true,
            stop: Vec::new(),
            chat_template: crate::chat::ChatTemplateOptions::default(),
        },
        submitted_token_ids_sha256,
    ))
}

#[derive(Debug)]
struct ObservedRequest {
    name: String,
    task_index: Option<usize>,
    rendered_prompt_sha256: String,
    prompt_token_ids_sha256: String,
    text: String,
    time_to_first_token: Duration,
    latency: Duration,
    usage: PrefixDemoUsageEvidence,
    completed_at: Instant,
}

#[derive(Debug)]
struct ObservedBatch {
    batch_released_at: Instant,
    requests: Vec<ObservedRequest>,
    trace: EngineBatchTrace,
}

impl ObservedRequest {
    fn into_evidence(self) -> PrefixDemoRequestEvidence {
        PrefixDemoRequestEvidence {
            name: self.name,
            task_index: self.task_index,
            rendered_prompt_sha256: self.rendered_prompt_sha256,
            prompt_token_ids_sha256: self.prompt_token_ids_sha256,
            text: self.text,
            time_to_first_token_ms: millis(self.time_to_first_token),
            latency_ms: millis(self.latency),
            usage: self.usage,
            finish_reason: "length".to_owned(),
        }
    }
}

async fn observe_ticket(
    mut ticket: EngineTicket,
    submitted_at: Instant,
    expected: &PreparedPrefixPrompt,
    submitted_token_ids_sha256: String,
    expected_cached_tokens: usize,
    expected_completion_tokens: usize,
) -> Result<ObservedRequest> {
    ensure!(
        submitted_token_ids_sha256 == expected.prompt_token_ids_sha256,
        "request `{}` submitted token digest differs from prepared evidence",
        expected.name
    );
    ticket.wait_for_preflight().await.map_err(|error| {
        anyhow::anyhow!(
            "request `{}` preflight failed ({:?}): {}",
            expected.name,
            error.kind,
            error.message
        )
    })?;
    let mut text = String::new();
    let mut time_to_first_token = None;
    loop {
        match ticket.recv().await {
            Some(EngineEvent::Text {
                content: fragment, ..
            }) => {
                ensure!(
                    !fragment.is_empty(),
                    "engine emitted an empty Text fragment"
                );
                time_to_first_token.get_or_insert_with(|| submitted_at.elapsed());
                text.push_str(&fragment);
            }
            Some(EngineEvent::Finished(metrics)) => {
                validate_metrics(
                    &expected.name,
                    &metrics,
                    expected.prompt_token_count,
                    expected_cached_tokens,
                    expected_completion_tokens,
                )?;
                let completed_at = Instant::now();
                let latency = completed_at
                    .checked_duration_since(submitted_at)
                    .context("request completion preceded submission")?;
                let time_to_first_token = time_to_first_token.with_context(|| {
                    format!("request `{}` finished without visible text", expected.name)
                })?;
                ensure!(
                    latency >= time_to_first_token,
                    "request latency is shorter than TTFT"
                );
                return Ok(ObservedRequest {
                    name: expected.name.clone(),
                    task_index: expected.task_index,
                    rendered_prompt_sha256: expected.rendered_prompt_sha256.clone(),
                    prompt_token_ids_sha256: submitted_token_ids_sha256,
                    text,
                    time_to_first_token,
                    latency,
                    usage: PrefixDemoUsageEvidence {
                        prompt_tokens: metrics.prompt_tokens,
                        cached_prompt_tokens: metrics.cached_prompt_tokens,
                        completion_tokens: metrics.generated_tokens,
                        decode_steps: metrics.decode_steps,
                    },
                    completed_at,
                });
            }
            Some(EngineEvent::Failed { message, kind }) => {
                bail!("request `{}` failed ({kind:?}): {message}", expected.name);
            }
            None => bail!("request `{}` closed before Finished", expected.name),
        }
    }
}

fn validate_metrics(
    name: &str,
    metrics: &GenerationMetrics,
    expected_prompt_tokens: usize,
    expected_cached_tokens: usize,
    expected_completion_tokens: usize,
) -> Result<()> {
    ensure!(
        metrics.prompt_tokens == expected_prompt_tokens,
        "request `{name}` prompt count changed: expected {expected_prompt_tokens}, found {}",
        metrics.prompt_tokens
    );
    ensure!(
        metrics.cached_prompt_tokens == expected_cached_tokens,
        "request `{name}` restored {} cached tokens; expected {expected_cached_tokens}",
        metrics.cached_prompt_tokens
    );
    ensure!(
        metrics.generated_tokens == expected_completion_tokens,
        "request `{name}` emitted {} tokens; expected {expected_completion_tokens}",
        metrics.generated_tokens
    );
    ensure!(
        metrics.finish_reason == FinishReason::Length,
        "request `{name}` ended with {:?}; fixed-length demo requires Length",
        metrics.finish_reason
    );
    Ok(())
}

async fn require_quiescent_snapshot(
    handle: &EngineHandle,
    stage: &'static str,
) -> Result<EnginePrefixCacheSnapshot> {
    let deadline = Instant::now()
        .checked_add(CACHE_QUIESCENCE_TIMEOUT)
        .context("cache quiescence deadline overflow")?;
    loop {
        let snapshot = handle.prefix_cache_snapshot();
        validate_snapshot(&snapshot, stage)?;
        if snapshot.active_sequences == 0 {
            return Ok(snapshot);
        }
        ensure!(
            Instant::now() < deadline,
            "{stage} cache diagnostics did not become quiescent"
        );
        tokio::task::yield_now().await;
    }
}

fn validate_snapshot(
    snapshot: &EnginePrefixCacheSnapshot,
    stage: &str,
) -> Result<PrefixCacheDiagnostics> {
    ensure!(
        snapshot.observation.is_none(),
        "{stage} cache diagnostics are unavailable: {}",
        snapshot
            .observation
            .as_deref()
            .unwrap_or("unknown observation")
    );
    let current = snapshot
        .current
        .with_context(|| format!("{stage} cache diagnostics have no current sample"))?;
    ensure!(current.block_tokens > 0, "{stage} cache block size is zero");
    ensure!(
        current.physical_live_blocks <= current.physical_max_blocks,
        "{stage} live physical blocks exceed the configured pool"
    );
    ensure!(
        current.physical_shared_blocks <= current.physical_live_blocks,
        "{stage} shared blocks exceed live blocks"
    );
    ensure!(
        current.physical_live_references >= current.physical_live_blocks,
        "{stage} physical references are fewer than live blocks"
    );
    Ok(current)
}

fn validate_cold_cache(snapshot: &EnginePrefixCacheSnapshot) -> Result<()> {
    ensure!(
        snapshot.active_sequences == 0,
        "cold engine has active sequences"
    );
    let current = validate_snapshot(snapshot, "cold engine")?;
    ensure!(
        current.hits == 0
            && current.misses == 0
            && current.matched_tokens == 0
            && current.entries == 0
            && current.evictions == 0
            && current.resident_snapshot_blocks == 0
            && current.physical_live_blocks == 0
            && current.physical_live_references == 0
            && current.physical_shared_blocks == 0,
        "new prefix-demo engine did not expose a cold empty cache"
    );
    Ok(())
}

fn validate_seed_cache(
    cold: &EnginePrefixCacheSnapshot,
    after_seed: &EnginePrefixCacheSnapshot,
    endpoint_tokens: usize,
) -> Result<()> {
    let cold = validate_snapshot(cold, "cold engine")?;
    let seed = validate_snapshot(after_seed, "after seed")?;
    ensure!(
        after_seed.active_sequences == 0,
        "seed left an active sequence"
    );
    ensure!(seed.entries > 0, "seed published no cache entries");
    ensure!(
        seed.resident_snapshot_blocks > 0 && seed.physical_live_blocks > 0,
        "seed published no resident physical cache pages"
    );
    ensure!(
        endpoint_tokens.is_multiple_of(seed.block_tokens),
        "prepared cache endpoint is not cache-block aligned"
    );
    ensure!(
        seed.hits == cold.hits,
        "cold seed unexpectedly hit the cache"
    );
    ensure!(
        seed.matched_tokens == cold.matched_tokens,
        "cold seed unexpectedly restored cached tokens"
    );
    let expected_misses = cold
        .misses
        .checked_add(1)
        .context("seed cache-miss counter overflow")?;
    ensure!(
        seed.misses == expected_misses,
        "cold seed must add exactly one cache miss"
    );
    Ok(())
}

fn validate_measured_cache_delta(
    before: &EnginePrefixCacheSnapshot,
    after: &EnginePrefixCacheSnapshot,
    endpoint_tokens: usize,
    request_count: usize,
) -> Result<PrefixCacheCounterDeltaEvidence> {
    let before = validate_snapshot(before, "before measured batch")?;
    let after_current = validate_snapshot(after, "after measured batch")?;
    ensure!(
        after.active_sequences == 0,
        "measured batch left active sequences"
    );
    let requests = u64::try_from(request_count).context("request count exceeds u64")?;
    let endpoint = u64::try_from(endpoint_tokens).context("cache endpoint exceeds u64")?;
    let expected_matched = endpoint
        .checked_mul(requests)
        .context("expected matched-token counter delta overflow")?;
    let expected_hits = before
        .hits
        .checked_add(requests)
        .context("expected cache-hit counter overflow")?;
    let expected_total_matched = before
        .matched_tokens
        .checked_add(expected_matched)
        .context("expected matched-token counter overflow")?;
    ensure!(
        after_current.hits == expected_hits,
        "measured batch cache hits changed by {}, expected {requests}",
        after_current.hits.saturating_sub(before.hits)
    );
    ensure!(
        after_current.misses == before.misses,
        "a measured request missed the warmed cache endpoint"
    );
    ensure!(
        after_current.matched_tokens == expected_total_matched,
        "measured batch matched-token total is not request_count * endpoint"
    );
    Ok(PrefixCacheCounterDeltaEvidence {
        hits: after_current
            .hits
            .checked_sub(before.hits)
            .context("cache hit counter regressed")?,
        misses: after_current
            .misses
            .checked_sub(before.misses)
            .context("cache miss counter regressed")?,
        matched_tokens: after_current
            .matched_tokens
            .checked_sub(before.matched_tokens)
            .context("matched-token counter regressed")?,
    })
}

#[derive(Debug, Clone, Copy)]
struct ValidatedPageReuse {
    first_admission: PrefixCachePeakEvidence,
    active_request_prefix_page_references: usize,
    active_request_prefix_page_fanout: f64,
    active_request_prefix_reuse_coverage: f64,
    new_physical_live_blocks_at_admission: usize,
    atomic_pool_owner_density: f64,
}

fn validate_eight_way_page_reuse(
    before_wave: &EnginePrefixCacheSnapshot,
    trace: &EngineBatchTrace,
    requested_shared_tokens: usize,
    prepared_block_tokens: usize,
) -> Result<ValidatedPageReuse> {
    let baseline = validate_snapshot(before_wave, "before-wave cache")?;
    ensure!(
        trace.expected_members == MAC_PREFIX_DEMO_CONCURRENCY
            && trace.admitted_members == MAC_PREFIX_DEMO_CONCURRENCY
            && trace.terminal_members == MAC_PREFIX_DEMO_CONCURRENCY
            && trace.rejected_members == 0
            && trace.admission_active_sequences == MAC_PREFIX_DEMO_CONCURRENCY,
        "engine benchmark sidecar did not retain one complete eight-way admission"
    );
    let concurrent = trace
        .admission_diagnostics
        .context("engine retained no post-attach, pre-forward prefix-cache diagnostic")?;
    ensure!(
        concurrent.physical_live_blocks > 0
            && concurrent.physical_live_references > concurrent.physical_live_blocks
            && concurrent.physical_shared_blocks > 0,
        "measured batch did not expose physical page sharing"
    );
    ensure!(
        concurrent.block_tokens == prepared_block_tokens
            && requested_shared_tokens.is_multiple_of(concurrent.block_tokens),
        "first-admission diagnostic block geometry differs from prepared evidence"
    );
    let prefix_blocks = requested_shared_tokens / concurrent.block_tokens;
    let expected_request_references = prefix_blocks
        .checked_mul(MAC_PREFIX_DEMO_CONCURRENCY)
        .context("expected active-request prefix references overflow")?;
    let active_request_prefix_page_references = concurrent
        .physical_live_references
        .checked_sub(baseline.physical_live_references)
        .context("physical page references regressed between seed and eight-way admission")?;
    ensure!(
        active_request_prefix_page_references == expected_request_references,
        "eight-way admission added {active_request_prefix_page_references} physical page references; expected exactly {expected_request_references}"
    );
    let new_physical_live_blocks_at_admission = concurrent
        .physical_live_blocks
        .checked_sub(baseline.physical_live_blocks)
        .context("physical live blocks regressed between seed and eight-way admission")?;
    ensure!(
        new_physical_live_blocks_at_admission == 0,
        "eight-way prefix attachment allocated {new_physical_live_blocks_at_admission} new physical blocks instead of reusing the warmed pages"
    );
    ensure!(
        concurrent.physical_shared_blocks >= prefix_blocks,
        "first-admission diagnostic exposes only {} shared physical blocks; expected at least {prefix_blocks}",
        concurrent.physical_shared_blocks
    );
    let active_request_prefix_page_fanout = finite_ratio(
        active_request_prefix_page_references,
        prefix_blocks,
        "active-request prefix-page fan-out",
    )?;
    let active_request_prefix_reuse_coverage = finite_ratio(
        active_request_prefix_page_references,
        expected_request_references,
        "active-request prefix reuse coverage",
    )?;
    ensure!(
        (active_request_prefix_page_fanout - MAC_PREFIX_DEMO_CONCURRENCY as f64).abs()
            <= f64::EPSILON
            && (active_request_prefix_reuse_coverage - 1.0).abs() <= f64::EPSILON,
        "eight-way prefix reuse ratios are inconsistent with exact physical reference counts"
    );
    let atomic_pool_owner_density = finite_ratio(
        concurrent.physical_live_references,
        concurrent.physical_live_blocks,
        "atomic pool owner density",
    )?;
    let first_admission = PrefixCachePeakEvidence {
        active_sequences: trace.admission_active_sequences,
        resident_snapshot_blocks: concurrent.resident_snapshot_blocks,
        block_tokens: concurrent.block_tokens,
        physical_live_blocks: concurrent.physical_live_blocks,
        physical_live_references: concurrent.physical_live_references,
        physical_shared_blocks: concurrent.physical_shared_blocks,
    };
    Ok(ValidatedPageReuse {
        first_admission,
        active_request_prefix_page_references,
        active_request_prefix_page_fanout,
        active_request_prefix_reuse_coverage,
        new_physical_live_blocks_at_admission,
        atomic_pool_owner_density,
    })
}

fn finite_ratio(numerator: usize, denominator: usize, label: &str) -> Result<f64> {
    ensure!(denominator > 0, "cannot divide {label} by zero");
    let ratio = numerator as f64 / denominator as f64;
    ensure!(ratio.is_finite() && ratio >= 0.0, "invalid {label} ratio");
    Ok(ratio)
}

fn phase_timing_evidence(trace: &EngineBatchTrace) -> Result<MacPrefixDemoPhaseTimingEvidence> {
    ensure!(
        trace.expected_members == MAC_PREFIX_DEMO_CONCURRENCY
            && trace.admitted_members == MAC_PREFIX_DEMO_CONCURRENCY
            && trace.terminal_members == MAC_PREFIX_DEMO_CONCURRENCY
            && trace.rejected_members == 0,
        "benchmark timing sidecar does not describe one successful eight-request wave"
    );
    ensure!(
        trace.mixed_forward_time.is_zero()
            && trace
                .packed_forwards
                .iter()
                .all(|forward| forward.kind != BenchmarkForwardKind::Mixed),
        "a packed forward mixed benchmark phases and cannot be assigned without double-counting"
    );

    let mut decode_steps = Vec::with_capacity(MAC_PREFIX_DEMO_DECODE_TOKENS - 1);
    for forward in &trace.packed_forwards {
        ensure!(
            forward.lane_count == MAC_PREFIX_DEMO_CONCURRENCY
                && forward.lanes.len() == MAC_PREFIX_DEMO_CONCURRENCY
                && forward.input_tokens
                    == forward
                        .lanes
                        .iter()
                        .try_fold(0_usize, |total, lane| total.checked_add(lane.input_tokens))
                        .context("packed forward input-token count overflow")?
                && forward
                    .lanes
                    .iter()
                    .enumerate()
                    .all(|(member, lane)| lane.member_index == Some(member)
                        && lane.phase == forward.kind),
            "benchmark packed-forward geometry is not one stable eight-lane demo batch"
        );
        if forward.kind == BenchmarkForwardKind::Decode {
            decode_steps.push(millis(forward.model_time));
        }
    }
    ensure!(
        decode_steps.len() == MAC_PREFIX_DEMO_DECODE_TOKENS - 1,
        "benchmark trace retained {} packed decode quanta; expected {}",
        decode_steps.len(),
        MAC_PREFIX_DEMO_DECODE_TOKENS - 1
    );
    let decode_total_ms = millis(trace.decode_time);
    ensure!(
        approximately_equal_millis(decode_total_ms, decode_steps.iter().sum()),
        "packed decode total differs from its raw forward durations"
    );
    let queue_admission = trace
        .queue_wait
        .checked_add(trace.admission_exclusive)
        .context("queue/admission duration overflow")?;
    Ok(MacPrefixDemoPhaseTimingEvidence {
        queue_admission_ms: millis(queue_admission),
        prefix_lookup_attach_ms: millis(trace.prefix_lookup_attach_time),
        cache_diagnostics_ms: millis(trace.diagnostics_time),
        suffix_prefill_ms: millis(trace.suffix_prefill_time),
        first_token_model_forward_ms: millis(trace.first_token_model_time),
        single_step_decode_ms: decode_steps,
        single_step_decode_total_ms: decode_total_ms,
        boundary: concat!(
            "queue/admission is queue_wait plus admission-exclusive host time; prefix lookup/attach ",
            "is the summed reset-session span; cache diagnostics is one post-attach, pre-forward ",
            "snapshot; remaining suffix prefill contains only non-terminal prefill forwards; ",
            "the first-token-producing model forward is the complete terminal packed prefill forward, ",
            "including projections, attention, MLP, output conversion, and logits readback, and may ",
            "contain the remaining suffix; single-step decode retains one duration per packed eight-lane ",
            "decode forward. Every packed model duration belongs to exactly one bucket. These mutually ",
            "exclusive sampled spans are not an exhaustive decomposition of external wave wall time"
        )
        .to_owned(),
    })
}

fn approximately_equal_millis(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= scale * 1.0e-9
}

pub(crate) fn derive_macllm_distributions(
    waves: &[MacPrefixDemoWaveEvidence],
) -> Result<MacPrefixDemoDistributionEvidence> {
    ensure!(
        !waves.is_empty(),
        "MacLLM execution retains no measured waves"
    );
    let mut wave_max_ttft = Vec::with_capacity(waves.len());
    let mut wave_wall = Vec::with_capacity(waves.len());
    let mut wave_tps = Vec::with_capacity(waves.len());
    let mut queue_admission = Vec::with_capacity(waves.len());
    let mut prefix_lookup_attach = Vec::with_capacity(waves.len());
    let mut cache_diagnostics = Vec::with_capacity(waves.len());
    let mut suffix_prefill = Vec::with_capacity(waves.len());
    let mut first_token_model_forward = Vec::with_capacity(waves.len());
    let mut single_step_decode_total = Vec::with_capacity(waves.len());

    for (expected_index, wave) in waves.iter().enumerate() {
        ensure!(
            wave.wave_index == expected_index,
            "MacLLM measured waves are not retained in contiguous execution order"
        );
        ensure!(
            wave.requests.len() == MAC_PREFIX_DEMO_CONCURRENCY,
            "MacLLM wave {expected_index} does not retain exactly eight requests"
        );
        let max_ttft = wave
            .requests
            .iter()
            .map(|request| request.time_to_first_token_ms)
            .max_by(f64::total_cmp)
            .context("MacLLM wave has no TTFT samples")?;
        wave_max_ttft.push(max_ttft);
        wave_wall.push(wave.aggregate.wall_time_ms);
        wave_tps.push(wave.aggregate.aggregate_output_tokens_per_second);
        queue_admission.push(wave.phases.queue_admission_ms);
        prefix_lookup_attach.push(wave.phases.prefix_lookup_attach_ms);
        cache_diagnostics.push(wave.phases.cache_diagnostics_ms);
        suffix_prefill.push(wave.phases.suffix_prefill_ms);
        first_token_model_forward.push(wave.phases.first_token_model_forward_ms);
        single_step_decode_total.push(wave.phases.single_step_decode_total_ms);
    }

    Ok(MacPrefixDemoDistributionEvidence {
        wave_max_time_to_first_token_ms: summarize(&wave_max_ttft)?,
        wave_max_time_to_first_token_ms_samples: wave_max_ttft,
        wave_wall_time_ms: summarize(&wave_wall)?,
        wave_wall_time_ms_samples: wave_wall,
        wave_aggregate_output_tokens_per_second: summarize(&wave_tps)?,
        wave_aggregate_output_tokens_per_second_samples: wave_tps,
        queue_admission_ms: summarize(&queue_admission)?,
        queue_admission_ms_samples: queue_admission,
        prefix_lookup_attach_ms: summarize(&prefix_lookup_attach)?,
        prefix_lookup_attach_ms_samples: prefix_lookup_attach,
        cache_diagnostics_ms: summarize(&cache_diagnostics)?,
        cache_diagnostics_ms_samples: cache_diagnostics,
        suffix_prefill_ms: summarize(&suffix_prefill)?,
        suffix_prefill_ms_samples: suffix_prefill,
        first_token_model_forward_ms: summarize(&first_token_model_forward)?,
        first_token_model_forward_ms_samples: first_token_model_forward,
        single_step_decode_total_ms: summarize(&single_step_decode_total)?,
        single_step_decode_total_ms_samples: single_step_decode_total,
    })
}

fn aggregate_requests(
    requests: &[ObservedRequest],
    batch_release: Instant,
    last_completion: Instant,
) -> Result<PrefixDemoAggregateEvidence> {
    ensure!(
        requests.len() == MAC_PREFIX_DEMO_CONCURRENCY,
        "aggregate requires exactly eight request samples"
    );
    let wall_time = last_completion
        .checked_duration_since(batch_release)
        .context("last completion preceded atomic batch release")?;
    ensure!(!wall_time.is_zero(), "measured batch wall time is zero");
    let total_completion_tokens = requests.iter().try_fold(0_u64, |total, request| {
        total
            .checked_add(
                u64::try_from(request.usage.completion_tokens)
                    .context("completion token count exceeds u64")?,
            )
            .context("aggregate completion-token total overflow")
    })?;
    let rate = total_completion_tokens as f64 / wall_time.as_secs_f64();
    ensure!(
        rate.is_finite() && rate > 0.0,
        "invalid aggregate output throughput"
    );
    Ok(PrefixDemoAggregateEvidence {
        request_count: requests.len(),
        total_completion_tokens,
        wall_time_ms: millis(wall_time),
        aggregate_output_tokens_per_second: rate,
    })
}

fn snapshot_evidence(
    snapshot: &EnginePrefixCacheSnapshot,
    stage: &str,
) -> Result<PrefixCacheSnapshotEvidence> {
    Ok(PrefixCacheSnapshotEvidence {
        current: validate_snapshot(snapshot, stage)?.into(),
        active_sequences: snapshot.active_sequences,
        observation: snapshot.observation.clone(),
    })
}

impl From<PrefixCacheDiagnostics> for PrefixCacheDiagnosticsEvidence {
    fn from(value: PrefixCacheDiagnostics) -> Self {
        Self {
            hits: value.hits,
            misses: value.misses,
            matched_tokens: value.matched_tokens,
            entries: value.entries,
            evictions: value.evictions,
            resident_snapshot_blocks: value.resident_snapshot_blocks,
            block_tokens: value.block_tokens,
            physical_live_blocks: value.physical_live_blocks,
            physical_live_references: value.physical_live_references,
            physical_shared_blocks: value.physical_shared_blocks,
            physical_max_blocks: value.physical_max_blocks,
        }
    }
}

fn identity_evidence(prepared: &PreparedPrefixDemo) -> MacPrefixDemoIdentityEvidence {
    MacPrefixDemoIdentityEvidence {
        prepared_schema_version: prepared.schema_version,
        model_family: prepared.model_family.clone(),
        model_filename: prepared.model_file.filename.clone(),
        model_sha256: prepared.model_file.sha256.clone(),
        repository_file_count: prepared.repository_bundle.file_count,
        repository_bundle_sha256: prepared.repository_bundle.sha256.clone(),
        repository_payload_encoding: prepared.repository_payload_encoding.clone(),
        encoded_repository_bundle_sha256: prepared.encoded_repository_bundle_sha256.clone(),
        selected_bundle_byte_count: prepared.selected_bundle_byte_count,
        selected_bundle_char_count: prepared.selected_bundle_char_count,
        selected_bundle_sha256: prepared.selected_bundle_sha256.clone(),
        common_prefix_token_ids_sha256: prepared.common_prefix_token_ids_sha256.clone(),
    }
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real-device development gate for profiling one `MacLLM` wave without
    /// launching llama-server or constructing a publishable comparison.
    ///
    /// The model and repository default to this checkout's pinned fixture and
    /// can be overridden with `MACLLM_PREFIX_PROBE_MODEL` and
    /// `MACLLM_PREFIX_PROBE_REPOSITORY`.
    #[tokio::test]
    #[ignore = "requires the pinned local model and an Apple Metal device"]
    async fn real_model_one_wave_phase_probe() -> Result<()> {
        let model_path = std::env::var_os("MACLLM_PREFIX_PROBE_MODEL")
            .map_or_else(|| PathBuf::from("models/smollm3-3b-q4_k_m"), PathBuf::from);
        let repository = std::env::var_os("MACLLM_PREFIX_PROBE_REPOSITORY")
            .map_or_else(|| PathBuf::from("."), PathBuf::from);
        let prepared = super::super::prepare_default_prefix_demo(&model_path, &repository)?;
        let materialized =
            super::super::materialize_prefix_demo(&model_path, &repository, &prepared)?;
        let mut config = MacPrefixDemoRunConfig::new(
            &model_path,
            "smollm3-3b-q4_k_m-phase-probe",
            prepared.prefill_quantum,
        );
        config.kv_cache_block_tokens = prepared.kv_cache_block_tokens;
        config.measured_waves = 1;

        let execution = run_macllm_prefix_demo(&config, &prepared, &materialized).await?;
        let wave = execution
            .waves
            .first()
            .context("one-wave probe returned no measured wave")?;
        let wave_max_ttft_ms = wave
            .requests
            .iter()
            .map(|request| request.time_to_first_token_ms)
            .max_by(f64::total_cmp)
            .context("one-wave probe returned no TTFT samples")?;
        println!(
            "MACLLM_PREFIX_PROBE_RESULT={}",
            serde_json::json!({
                "wave_max_ttft_ms": wave_max_ttft_ms,
                "aggregate_output_tokens_per_second": wave.aggregate.aggregate_output_tokens_per_second,
                "phases": wave.phases,
                "cache": {
                    "hits": wave.cache.measured_counter_delta.hits,
                    "misses": wave.cache.measured_counter_delta.misses,
                    "matched_tokens": wave.cache.measured_counter_delta.matched_tokens,
                    "active_request_prefix_page_references": wave.cache.active_request_prefix_page_references,
                    "new_physical_live_blocks_at_admission": wave.cache.new_physical_live_blocks_at_admission,
                }
            })
        );
        Ok(())
    }

    fn prompt_identity_fixture() -> (PreparedPrefixPrompt, MaterializedPrefixPrompt) {
        let rendered_prompt = "rendered".to_owned();
        let token_ids = vec![4, 2, 9, 1];
        let expected = PreparedPrefixPrompt {
            name: "seed".to_owned(),
            task_index: None,
            messages: vec![crate::chat::ChatMessage::user("fixture")],
            rendered_prompt_byte_count: rendered_prompt.len(),
            rendered_prompt_sha256: sha256_bytes(rendered_prompt.as_bytes()),
            prompt_token_count: token_ids.len(),
            prompt_token_ids_sha256: token_ids_sha256(&token_ids),
            lcp_with_seed_tokens: None,
            cache_endpoint_tokens: 3,
            common_prefix_token_ids_sha256: token_ids_sha256(&token_ids[..3]),
        };
        let materialized = MaterializedPrefixPrompt {
            name: expected.name.clone(),
            rendered_prompt,
            token_ids,
        };
        (expected, materialized)
    }

    fn metrics(cached: usize, finish_reason: FinishReason) -> GenerationMetrics {
        GenerationMetrics {
            prompt_tokens: 16_400,
            cached_prompt_tokens: cached,
            prefix_lookup_attach_time: Duration::from_millis(1),
            generated_tokens: MAC_PREFIX_DEMO_DECODE_TOKENS,
            decode_steps: MAC_PREFIX_DEMO_DECODE_TOKENS - 1,
            time_to_first_token: Duration::from_millis(2),
            suffix_prefill_time: Duration::ZERO,
            first_token_model_time: Duration::from_millis(1),
            prefill_time: Duration::from_millis(1),
            decode_time: Duration::from_millis(5),
            total_time: Duration::from_millis(8),
            finish_reason,
        }
    }

    fn snapshot(hits: u64, misses: u64, matched_tokens: u64) -> EnginePrefixCacheSnapshot {
        EnginePrefixCacheSnapshot {
            current: Some(PrefixCacheDiagnostics {
                hits,
                misses,
                matched_tokens,
                entries: 2,
                evictions: 0,
                resident_snapshot_blocks: 1_024,
                block_tokens: 16,
                physical_live_blocks: 1_100,
                physical_live_references: 8_200,
                physical_shared_blocks: 1_024,
                physical_max_blocks: 4_096,
            }),
            peak_physical_live_blocks: 1_100,
            peak_physical_live_references: 8_200,
            peak_physical_shared_blocks: 1_024,
            peak_concurrent_active_sequences: MAC_PREFIX_DEMO_CONCURRENCY,
            peak_concurrent: Some(PrefixCacheDiagnostics {
                hits,
                misses,
                matched_tokens,
                entries: 2,
                evictions: 0,
                resident_snapshot_blocks: 1_024,
                block_tokens: 16,
                physical_live_blocks: 1_100,
                physical_live_references: 8_200,
                physical_shared_blocks: 1_024,
                physical_max_blocks: 4_096,
            }),
            active_sequences: 0,
            observation: None,
        }
    }

    #[test]
    fn metrics_require_exact_prompt_cache_budget_and_length_finish() -> Result<()> {
        validate_metrics(
            "task-1",
            &metrics(DEFAULT_SHARED_PREFIX_TOKENS, FinishReason::Length),
            16_400,
            DEFAULT_SHARED_PREFIX_TOKENS,
            MAC_PREFIX_DEMO_DECODE_TOKENS,
        )?;
        assert!(
            validate_metrics(
                "task-1",
                &metrics(DEFAULT_SHARED_PREFIX_TOKENS - 1, FinishReason::Length),
                16_400,
                DEFAULT_SHARED_PREFIX_TOKENS,
                MAC_PREFIX_DEMO_DECODE_TOKENS,
            )
            .is_err()
        );
        assert!(
            validate_metrics(
                "task-1",
                &metrics(DEFAULT_SHARED_PREFIX_TOKENS, FinishReason::Stop),
                16_400,
                DEFAULT_SHARED_PREFIX_TOKENS,
                MAC_PREFIX_DEMO_DECODE_TOKENS,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn materialized_prompt_identity_rejects_render_token_and_order_drift() -> Result<()> {
        let (expected, materialized) = prompt_identity_fixture();
        validate_materialized_prompt(&expected, &materialized)?;

        let mut rendered_drift = materialized.clone();
        rendered_drift.rendered_prompt = "renderXd".to_owned();
        assert!(validate_materialized_prompt(&expected, &rendered_drift).is_err());

        let mut token_drift = materialized.clone();
        token_drift.token_ids[2] += 1;
        assert!(validate_materialized_prompt(&expected, &token_drift).is_err());

        let mut order_drift = materialized;
        order_drift.name = "task-1".to_owned();
        assert!(validate_materialized_prompt(&expected, &order_drift).is_err());
        Ok(())
    }

    #[test]
    fn token_request_submits_exact_ids_and_hashes_the_owned_input() -> Result<()> {
        let token_ids = vec![7, 11, 13, 17];
        let (request, submitted_digest) = token_request(&token_ids, 5)?;
        assert_eq!(submitted_digest, token_ids_sha256(&token_ids));
        match request.input {
            EngineInput::TokenIds(submitted) => assert_eq!(submitted, token_ids),
            other => panic!("expected raw token input, found {other:?}"),
        }
        assert!(token_request(&[], 1).is_err());
        Ok(())
    }

    #[test]
    fn cache_delta_is_exact_and_overflow_fails_closed() -> Result<()> {
        let before = snapshot(0, 1, 0);
        let after = snapshot(8, 1, 8 * DEFAULT_SHARED_PREFIX_TOKENS as u64);
        let delta = validate_measured_cache_delta(
            &before,
            &after,
            DEFAULT_SHARED_PREFIX_TOKENS,
            PREFIX_DEMO_TASK_COUNT,
        )?;
        assert_eq!(delta.hits, 8);
        assert_eq!(delta.misses, 0);
        assert_eq!(delta.matched_tokens, 131_072);

        let saturated = snapshot(u64::MAX, 1, 0);
        assert!(
            validate_measured_cache_delta(
                &saturated,
                &saturated,
                DEFAULT_SHARED_PREFIX_TOKENS,
                PREFIX_DEMO_TASK_COUNT,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn page_reuse_is_an_admission_delta_not_pool_owner_density() -> Result<()> {
        let baseline = snapshot(0, 1, 0);
        let mut admitted = snapshot(8, 1, 8 * DEFAULT_SHARED_PREFIX_TOKENS as u64);
        let first_admission = admitted
            .peak_concurrent
            .as_mut()
            .expect("test snapshot has a first-admission diagnostic");
        first_admission.physical_live_references +=
            MAC_PREFIX_DEMO_CONCURRENCY * (DEFAULT_SHARED_PREFIX_TOKENS / 16);
        let mut trace = EngineBatchTrace {
            expected_members: MAC_PREFIX_DEMO_CONCURRENCY,
            admitted_members: MAC_PREFIX_DEMO_CONCURRENCY,
            terminal_members: MAC_PREFIX_DEMO_CONCURRENCY,
            rejected_members: 0,
            queue_wait: Duration::ZERO,
            admission_exclusive: Duration::ZERO,
            prefix_lookup_attach_time: Duration::ZERO,
            diagnostics_time: Duration::ZERO,
            admission_diagnostics: admitted.peak_concurrent,
            admission_active_sequences: MAC_PREFIX_DEMO_CONCURRENCY,
            suffix_prefill_time: Duration::ZERO,
            first_token_model_time: Duration::ZERO,
            decode_time: Duration::ZERO,
            mixed_forward_time: Duration::ZERO,
            packed_forwards: Vec::new(),
        };
        let evidence =
            validate_eight_way_page_reuse(&baseline, &trace, DEFAULT_SHARED_PREFIX_TOKENS, 16)?;
        assert_eq!(evidence.active_request_prefix_page_references, 8_192);
        assert!((evidence.active_request_prefix_page_fanout - 8.0).abs() < f64::EPSILON);
        assert!((evidence.active_request_prefix_reuse_coverage - 1.0).abs() < f64::EPSILON);
        assert_eq!(evidence.new_physical_live_blocks_at_admission, 0);
        assert!(evidence.atomic_pool_owner_density > 8.0);

        trace
            .admission_diagnostics
            .as_mut()
            .expect("test snapshot has a first-admission diagnostic")
            .physical_live_blocks += 1;
        assert!(
            validate_eight_way_page_reuse(&baseline, &trace, DEFAULT_SHARED_PREFIX_TOKENS, 16,)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn unavailable_diagnostics_fail_closed() {
        let unavailable = EnginePrefixCacheSnapshot::default();
        assert!(validate_snapshot(&unavailable, "test").is_err());
    }

    #[test]
    fn twenty_wave_distributions_are_recomputed_from_wave_level_samples() -> Result<()> {
        let snapshot = snapshot(0, 1, 0);
        let snapshot = snapshot_evidence(&snapshot, "fixture")?;
        let mut waves = (0..MAC_PREFIX_DEMO_MIN_MEASURED_WAVES)
            .map(|wave_index| {
                let sample = (wave_index + 1) as f64;
                MacPrefixDemoWaveEvidence {
                    wave_index,
                    requests: (0..MAC_PREFIX_DEMO_CONCURRENCY)
                        .map(|task| PrefixDemoRequestEvidence {
                            name: format!("task_{task}"),
                            task_index: Some(task + 1),
                            rendered_prompt_sha256: "a".repeat(64),
                            prompt_token_ids_sha256: "b".repeat(64),
                            text: "ok".to_owned(),
                            time_to_first_token_ms: sample,
                            latency_ms: sample + 1.0,
                            usage: PrefixDemoUsageEvidence {
                                prompt_tokens: 16_400,
                                cached_prompt_tokens: DEFAULT_SHARED_PREFIX_TOKENS,
                                completion_tokens: MAC_PREFIX_DEMO_DECODE_TOKENS,
                                decode_steps: MAC_PREFIX_DEMO_DECODE_TOKENS - 1,
                            },
                            finish_reason: "length".to_owned(),
                        })
                        .collect(),
                    aggregate: PrefixDemoAggregateEvidence {
                        request_count: MAC_PREFIX_DEMO_CONCURRENCY,
                        total_completion_tokens: 256,
                        wall_time_ms: sample,
                        aggregate_output_tokens_per_second: sample,
                    },
                    cache: MacPrefixDemoCacheEvidence {
                        before_wave: snapshot.clone(),
                        after_wave: snapshot.clone(),
                        eight_way_first_admission_snapshot: PrefixCachePeakEvidence {
                            active_sequences: 8,
                            resident_snapshot_blocks: 1_024,
                            block_tokens: 16,
                            physical_live_blocks: 1_024,
                            physical_live_references: 9_216,
                            physical_shared_blocks: 1_024,
                        },
                        measured_counter_delta: PrefixCacheCounterDeltaEvidence {
                            hits: 8,
                            misses: 0,
                            matched_tokens: 131_072,
                        },
                        active_request_prefix_page_references: 8_192,
                        active_request_prefix_page_fanout: 8.0,
                        active_request_prefix_reuse_coverage: 1.0,
                        new_physical_live_blocks_at_admission: 0,
                        atomic_pool_owner_density: 9.0,
                        physical_reuse_boundary: "fixture".to_owned(),
                    },
                    phases: MacPrefixDemoPhaseTimingEvidence {
                        queue_admission_ms: sample,
                        prefix_lookup_attach_ms: sample,
                        cache_diagnostics_ms: sample,
                        suffix_prefill_ms: sample,
                        first_token_model_forward_ms: sample,
                        single_step_decode_ms: vec![sample; MAC_PREFIX_DEMO_DECODE_TOKENS - 1],
                        single_step_decode_total_ms: sample,
                        boundary: "fixture".to_owned(),
                    },
                }
            })
            .collect::<Vec<_>>();

        let evidence = derive_macllm_distributions(&waves)?;
        assert_eq!(evidence.wave_max_time_to_first_token_ms.sample_count, 20);
        assert!((evidence.wave_max_time_to_first_token_ms.p50 - 10.0).abs() <= f64::EPSILON);
        assert!((evidence.wave_max_time_to_first_token_ms.p95 - 19.0).abs() <= f64::EPSILON);
        waves[10].wave_index = 99;
        assert!(derive_macllm_distributions(&waves).is_err());
        Ok(())
    }
}
