//! Cache-enabled `llama-server` evidence for the prefix-cache demo.
//!
//! This is deliberately separate from the cold-cache scoreboard runner. It
//! starts one pinned server, verifies every rendered prompt through
//! `/tokenize`, explicitly seeds all eight fixed slots, then retains at least
//! twenty measured eight-request waves. Every measured request is pinned back
//! to the slot seeded for it, and every completed wave is followed by a seed
//! restore before the next wave. A pair of unmeasured replay waves proves that
//! greedy completion token IDs are reproducible. External request timings and
//! terminal server counters remain separate: in particular, `cache_n` is not
//! evidence of physical KV-page sharing and `predicted_ms` is not a pure
//! decode-kernel measurement.
//!
//! The small HTTP/chunked/SSE transport below mirrors the fail-closed parser in
//! `benchmark::llama_server`. It is copied here because that transport is
//! private and this demo must remain independently wireable without changing
//! the public cold-scoreboard contract.

use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, Read, Write},
    net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use super::prefix_demo_mac::MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET;
use super::{
    BenchmarkCommand, FileProvenance, ImplementationKind, LlamaCppVersion, LlamaFlashAttention,
    ModelProvenance, VerifiedScoreboard, aggregate_throughput, token_ids_sha256,
};

/// Number of simultaneous coding-agent requests in the final demo.
pub const PREFIX_DEMO_CONCURRENCY: usize = 8;
/// Minimum number of retained llama.cpp waves required for percentile evidence.
pub const LLAMA_PREFIX_DEMO_MIN_MEASURED_WAVES: usize = 20;
/// Default retained llama.cpp wave count for a fair warm-prefix comparison.
pub const LLAMA_PREFIX_DEMO_DEFAULT_MEASURED_WAVES: usize = 20;
/// Extra per-slot space kept outside the prompt and fixed decode boundary.
pub const PREFIX_DEMO_CONTEXT_HEADROOM_TOKENS: usize = 256;
/// `llama-server` same-slot KV-shift reuse is disabled for this comparison.
pub const PREFIX_DEMO_CACHE_REUSE_TOKENS: usize = 0;

const LOOPBACK: Ipv4Addr = Ipv4Addr::LOCALHOST;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_mins(5);
const HEALTH_TIMEOUT: Duration = Duration::from_mins(2);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(50);
const PORT_CLOSE_TIMEOUT: Duration = Duration::from_secs(5);
const PORT_POLL_INTERVAL: Duration = Duration::from_millis(20);
const PORT_PROBE_TIMEOUT: Duration = Duration::from_millis(100);
const MAX_HTTP_LINE_BYTES: usize = 16 * 1024;
const MAX_HTTP_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const MAX_HTTP_BODY_BYTES: usize = 32 * 1024 * 1024;
pub const LLAMA_CONTEXT_ALIGNMENT_PER_SLOT: usize = 256;

/// One already-rendered native-model prompt and its locally computed IDs.
#[derive(Debug, Clone, Copy)]
pub struct LlamaPrefixDemoPrompt<'a> {
    pub rendered_prompt: &'a str,
    pub token_ids: &'a [u32],
}

/// Immutable binary, source, and model identity used by the demo process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaPrefixDemoProvenance {
    pub llama_cpp_version: LlamaCppVersion,
    pub llama_cpp_full_commit: String,
    pub llama_server: FileProvenance,
    pub model: ModelProvenance,
}

/// Auditable total/per-slot context calculation for eight independent slots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaPrefixDemoContextGeometry {
    pub slots: usize,
    pub maximum_prompt_tokens: usize,
    pub decode_tokens: usize,
    pub headroom_tokens_per_slot: usize,
    pub minimum_tokens_per_slot: usize,
    pub minimum_total_context_tokens: usize,
    pub configured_total_context_tokens: usize,
    pub configured_tokens_per_slot: usize,
    pub manifest_kv_capacity_tokens: usize,
    pub cache_reuse_chunk_tokens: usize,
    pub logical_batch_tokens: usize,
    pub physical_ubatch_tokens: usize,
}

/// Exact tokenizer comparison performed against the running server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaPrefixDemoTokenizerEvidence {
    pub label: String,
    pub token_count: usize,
    pub expected_token_ids_sha256: String,
    pub observed_token_ids_sha256: String,
    pub exact: bool,
    pub add_special: bool,
}

/// Explicit generation/request policy sent to `/completion`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaPrefixDemoRequestPolicy {
    pub decode_tokens: usize,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub seed: u64,
    pub ignore_eos: bool,
    pub stream: bool,
    pub return_tokens: bool,
    pub cache_prompt: bool,
    pub n_keep: usize,
}

/// Terminal counters and timings reported by `llama-server` itself.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaPrefixDemoTerminalEvidence {
    /// Logical prompt tokens reported as reused by `llama-server`.
    pub server_reported_cache_n: usize,
    /// Prompt tokens reported as evaluated for this request.
    pub server_reported_prompt_n: usize,
    pub server_reported_prompt_ms: f64,
    pub server_reported_prompt_tokens_per_second: f64,
    pub server_reported_predicted_n: usize,
    pub server_reported_predicted_ms: f64,
    pub server_reported_predicted_tokens_per_second: f64,
}

/// One externally timed seed or eight-way request.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaPrefixDemoRequestEvidence {
    pub label: String,
    pub request: Option<usize>,
    /// Explicit llama-server slot pinned in the submitted JSON body.
    pub slot_id: usize,
    pub prompt_tokens: usize,
    pub prompt_token_ids_sha256: String,
    /// Authoritative visible content from the deterministic non-stream replay.
    pub content: String,
    /// Content observed in the measured streaming wave before replay attachment.
    pub streamed_content: String,
    pub streamed_content_matches_replay: bool,
    pub content_source: String,
    pub completion_tokens: usize,
    pub completion_token_ids: Vec<u32>,
    pub completion_token_ids_sha256: String,
    pub completion_token_source: String,
    pub external_time_to_first_token_ms: f64,
    pub external_latency_ms: f64,
    pub terminal: LlamaPrefixDemoTerminalEvidence,
}

/// Non-percentile range for one small demo wave.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixDemoTimingSummary {
    pub sample_count: usize,
    pub min: f64,
    pub mean: f64,
    pub max: f64,
}

impl PrefixDemoTimingSummary {
    fn from_samples(samples: &[f64]) -> Result<Self> {
        ensure!(
            !samples.is_empty()
                && samples
                    .iter()
                    .all(|sample| sample.is_finite() && *sample >= 0.0),
            "prefix-demo timing samples must be finite, non-negative, and non-empty"
        );
        let min = samples
            .iter()
            .copied()
            .min_by(f64::total_cmp)
            .context("prefix-demo timing samples have no minimum")?;
        let max = samples
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .context("prefix-demo timing samples have no maximum")?;
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        Ok(Self {
            sample_count: samples.len(),
            min,
            mean,
            max,
        })
    }

    /// Recompute the non-percentile summary from retained request samples.
    pub fn verify(&self, samples: &[f64]) -> Result<()> {
        let expected = Self::from_samples(samples)?;
        ensure!(
            self.sample_count == expected.sample_count
                && approximately_equal(self.min, expected.min)
                && approximately_equal(self.mean, expected.mean)
                && approximately_equal(self.max, expected.max),
            "prefix-demo timing summary differs from retained request samples"
        );
        Ok(())
    }
}

/// Aggregate of exactly one simultaneous eight-request wave.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaPrefixDemoWaveEvidence {
    pub wave_index: usize,
    pub concurrency: usize,
    pub request_count: usize,
    pub total_completion_tokens: u64,
    pub external_wall_time_ms: f64,
    pub aggregate_output_tokens_per_second: f64,
    pub external_time_to_first_token_ms: PrefixDemoTimingSummary,
    pub external_latency_ms: PrefixDemoTimingSummary,
    pub requests: Vec<LlamaPrefixDemoRequestEvidence>,
    pub cache: LlamaPrefixDemoCacheObservation,
}

/// Exact-token replay evidence, deliberately excluded from measured throughput.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaPrefixDemoDeterminismEvidence {
    pub replay_concurrency: usize,
    pub replay_excluded_from_performance_measurement: bool,
    pub exact_per_request: bool,
    pub first_nonstream_replay_token_ids_sha256: Vec<String>,
    pub second_nonstream_replay_token_ids_sha256: Vec<String>,
}

/// Fail-closed same-slot warm-cache proof from one measured wave.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaPrefixDemoCacheObservation {
    pub exact_common_prefix_tokens: usize,
    pub server_reported_cache_n_by_request: Vec<usize>,
    pub server_reported_prompt_n_by_request: Vec<usize>,
    pub expected_prompt_n_by_request: Vec<usize>,
    pub requests_satisfying_exact_contract: usize,
    pub minimum_server_reported_cache_n: usize,
    pub maximum_server_reported_cache_n: usize,
    pub interpretation: String,
}

/// Complete cache-on comparison evidence from one isolated child process.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaPrefixDemoExecution {
    pub schema_version: u32,
    pub implementation: ImplementationKind,
    pub provenance: LlamaPrefixDemoProvenance,
    pub command: Vec<String>,
    pub base_url: String,
    pub health_wait_ms: f64,
    pub port_closed_after_shutdown: bool,
    pub context: LlamaPrefixDemoContextGeometry,
    pub request_policy: LlamaPrefixDemoRequestPolicy,
    pub tokenizer_parity: Vec<LlamaPrefixDemoTokenizerEvidence>,
    pub slot_seeds: Vec<LlamaPrefixDemoRequestEvidence>,
    pub measured_waves: Vec<LlamaPrefixDemoWaveEvidence>,
    pub determinism: LlamaPrefixDemoDeterminismEvidence,
    pub measurement_boundary: String,
    pub warnings: Vec<String>,
}

/// Build the exact cache-enabled server command without launching a process.
pub fn build_llama_prefix_demo_command(
    verified: &VerifiedScoreboard,
    seed: LlamaPrefixDemoPrompt<'_>,
    requests: &[LlamaPrefixDemoPrompt<'_>; PREFIX_DEMO_CONCURRENCY],
    decode_steps: usize,
    expected_cache_endpoint: usize,
    port: u16,
) -> Result<BenchmarkCommand> {
    let plan = plan_demo(
        verified,
        seed,
        requests,
        decode_steps,
        expected_cache_endpoint,
        LLAMA_PREFIX_DEMO_DEFAULT_MEASURED_WAVES,
        port,
    )?;
    Ok(plan.command)
}

/// Run eight cold slot seeds, retained warm waves, and two token-only replays.
///
/// Successful return also guarantees that the child was killed/reaped and its
/// loopback port stopped accepting connections.
#[allow(clippy::too_many_lines)]
pub fn run_llama_prefix_demo(
    verified: &VerifiedScoreboard,
    seed: LlamaPrefixDemoPrompt<'_>,
    requests: &[LlamaPrefixDemoPrompt<'_>; PREFIX_DEMO_CONCURRENCY],
    decode_steps: usize,
    expected_cache_endpoint: usize,
    measured_waves: usize,
    port: u16,
) -> Result<LlamaPrefixDemoExecution> {
    let plan = plan_demo(
        verified,
        seed,
        requests,
        decode_steps,
        expected_cache_endpoint,
        measured_waves,
        port,
    )?;
    let command_argv = plan.command.report_argv()?;
    let address = SocketAddr::V4(SocketAddrV4::new(LOOPBACK, port));
    ensure_port_is_closed(address)?;

    let mut server = LlamaServerProcess::spawn(&plan.command)?;
    let run_result = (|| {
        let health_wait = server.wait_until_ready(address)?;
        let tokenizer_parity = verify_all_tokenizers(address, &plan)?;

        let slot_seeds = seed_all_slots(address, &plan.slot_seeds)?;

        // Correctness replays are deliberately outside every measured timing
        // boundary. Restore all slots between them so both start from the same
        // exact seed state as every retained measured wave.
        let first_replay = execute_nonstream_wave(address, &plan.requests)?;
        restore_seed_slots(address, &plan.slot_seeds)?;
        let second_replay = execute_nonstream_wave(address, &plan.requests)?;
        let determinism = verify_replay(&first_replay, &second_replay)?;
        restore_seed_slots(address, &plan.slot_seeds)?;

        let mut measured_waves = Vec::with_capacity(plan.measured_waves);
        for wave_index in 0..plan.measured_waves {
            let mut measured = execute_wave(address, &plan.requests)?;
            attach_replay_tokens(&mut measured.requests, &first_replay)?;
            let wave = aggregate_wave(wave_index, measured, plan.expected_cache_endpoint)?;
            // A successful restore request processes the exact endpoint seed
            // and leaves that slot ready for the next task-specific suffix.
            // Failure to restore any one slot aborts the entire run.
            restore_seed_slots(address, &plan.slot_seeds)?;
            measured_waves.push(wave);
        }
        server.ensure_running()?;

        Ok::<_, anyhow::Error>(RunEvidence {
            health_wait,
            tokenizer_parity,
            slot_seeds,
            measured_waves,
            determinism,
        })
    })();

    // Always request deterministic shutdown, even when tokenization or a wave
    // failed. Drop remains a second kill/reap guard for early process errors.
    let stop_result = server.stop();
    let port_result = wait_until_port_closed(address);
    let run = finish_run_and_cleanup(run_result, stop_result, port_result)?;

    let mut warnings = vec![
        concat!(
            "each retained measured request is fixed to its explicitly seeded id_slot and must ",
            "report exact cache_n/prompt_n accounting, but those counters remain logical evidence; ",
            "they do not prove physical KV-page sharing, page reference counts, or proportional ",
            "memory savings"
        )
        .to_owned(),
        concat!(
            "--cache-reuse 0 explicitly disables llama-server same-slot KV-shift reuse, so ",
            "cache_n remains logical server evidence and is not mixed with configured shifted-chunk reuse"
        )
        .to_owned(),
        concat!(
            "terminal prompt_ms and predicted_ms are server-reported serving timings; predicted_ms ",
            "is not a pure single-token decode-forward kernel boundary"
        )
        .to_owned(),
        concat!(
            "external TTFT starts immediately before each loopback TCP connection and ends when ",
            "the first streamed token event is parsed; external latency ends at the terminal SSE event"
        )
        .to_owned(),
        concat!(
            "input identity is established by exact /tokenize token-ID parity before execution, ",
            "plus terminal evaluated-token accounting and truncated=false; llama-server's ",
            "implementation-derived prompt echo is not treated as a byte-for-byte transport echo"
        )
        .to_owned(),
        concat!(
            "two non-stream exact-token replays are performed before the measured waves and are ",
            "excluded from throughput, TTFT, and latency aggregation; measured completion token ",
            "IDs come from those replays because streaming may buffer incomplete UTF-8"
        )
        .to_owned(),
    ];
    let streaming_content_mismatches = run
        .measured_waves
        .iter()
        .flat_map(|wave| &wave.requests)
        .filter(|request| !request.streamed_content_matches_replay)
        .count()
        + run
            .slot_seeds
            .iter()
            .filter(|seed| !seed.streamed_content_matches_replay)
            .count();
    if streaming_content_mismatches > 0 {
        warnings.push(format!(
            "llama-server streaming content differed from deterministic non-stream replay content for {streaming_content_mismatches} request(s); both strings are retained, performance uses only the streaming waves, and measured output/token evidence uses the two exact non-stream replays"
        ));
    }

    Ok(LlamaPrefixDemoExecution {
        schema_version: 2,
        implementation: ImplementationKind::LlamaCpp,
        provenance: LlamaPrefixDemoProvenance {
            llama_cpp_version: verified.llama_cpp_version.clone(),
            llama_cpp_full_commit: verified.llama_cpp_commit.clone(),
            llama_server: verified.llama_server.clone(),
            model: verified.model.clone(),
        },
        command: command_argv,
        base_url: format!("http://{address}"),
        health_wait_ms: millis(run.health_wait),
        port_closed_after_shutdown: true,
        context: plan.context,
        request_policy: plan.request_policy,
        tokenizer_parity: run.tokenizer_parity,
        slot_seeds: run.slot_seeds,
        measured_waves: run.measured_waves,
        determinism: run.determinism,
        measurement_boundary: concat!(
            "one isolated pinned llama-server child with eight non-unified slots, continuous ",
            "batching, a 1024-token logical batch matching MacLLM's scheduler budget, a 128-token ",
            "physical ubatch matching its per-sequence prefill quantum, prompt caching, global ",
            "cache RAM, idle-slot cache, and automatic slot similarity disabled, and same-slot ",
            "KV-shift reuse disabled with --cache-reuse 0; all eight id_slot values are cold-seeded separately with the exact ",
            "prefix endpoint, every measured request is fixed to its corresponding slot, and all ",
            "slots are restored to the seed after every retained wave; aggregate throughput spans ",
            "the earliest actual request submission through the last terminal SSE event; exact ",
            "cache_n and prompt_n equality is a hard gate for every request in every wave"
        )
        .to_owned(),
        warnings,
    })
}

struct DemoPlan {
    command: BenchmarkCommand,
    context: LlamaPrefixDemoContextGeometry,
    request_policy: LlamaPrefixDemoRequestPolicy,
    slot_seeds: [CompletionSpec; PREFIX_DEMO_CONCURRENCY],
    requests: [CompletionSpec; PREFIX_DEMO_CONCURRENCY],
    expected_cache_endpoint: usize,
    measured_waves: usize,
}

fn plan_demo(
    verified: &VerifiedScoreboard,
    seed: LlamaPrefixDemoPrompt<'_>,
    requests: &[LlamaPrefixDemoPrompt<'_>; PREFIX_DEMO_CONCURRENCY],
    decode_steps: usize,
    expected_cache_endpoint: usize,
    measured_waves: usize,
    port: u16,
) -> Result<DemoPlan> {
    verified.manifest.validate()?;
    verified
        .llama_cpp_version
        .verify(&verified.manifest.llama_cpp, &verified.llama_cpp_commit)?;
    ensure!(port > 0, "llama-server prefix-demo port must be non-zero");
    ensure!(
        decode_steps > 0,
        "prefix-demo decode_steps must be positive"
    );
    ensure!(
        expected_cache_endpoint == super::prefix_demo::DEFAULT_SHARED_PREFIX_TOKENS,
        "fair llama.cpp warm-prefix comparison requires the exact {}-token endpoint; got \
         {expected_cache_endpoint}",
        super::prefix_demo::DEFAULT_SHARED_PREFIX_TOKENS
    );
    validate_measured_wave_count(measured_waves)?;
    ensure!(
        verified.manifest.runtime.prefill_token_quantum
            == super::prefix_demo::DEFAULT_PREFILL_QUANTUM,
        "prefix demo requires a {}-token physical prefill quantum",
        super::prefix_demo::DEFAULT_PREFILL_QUANTUM
    );

    let maximum_prompt_tokens = validate_prompt_set(seed, requests, expected_cache_endpoint)?;
    let context = context_geometry(
        maximum_prompt_tokens,
        decode_steps,
        verified.manifest.runtime.kv_cache_capacity_tokens,
        MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET,
        verified.manifest.runtime.prefill_token_quantum,
    )?;
    let threads = verified
        .host
        .performance_cores
        .or(verified.host.total_cores)
        .context("verified host has no CPU core count for pinned llama-server threads")?;
    ensure!(threads > 0, "llama-server thread count must be positive");

    let arguments = llama_server_arguments(
        &verified.model.file.canonical_path,
        &verified.manifest.llama_cpp,
        context.configured_total_context_tokens,
        context.logical_batch_tokens,
        context.physical_ubatch_tokens,
        threads,
        port,
    )?;
    let command = BenchmarkCommand {
        program: verified.llama_server.canonical_path.clone(),
        arguments,
    };
    let sampling = ExpectedSampling::from_verified(verified)?;
    let request_policy = sampling.request_policy(decode_steps);
    let slot_seed_results: [Result<CompletionSpec>; PREFIX_DEMO_CONCURRENCY] =
        std::array::from_fn(|slot_id| {
            CompletionSpec::new(format!("seed_slot_{slot_id}"), seed, 1, slot_id, sampling)
        });
    let slot_seeds = slot_seed_results
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("prefix demo must retain exactly eight slot seed specs"))?;
    let requests = requests
        .iter()
        .copied()
        .enumerate()
        .map(|(request, prompt)| {
            CompletionSpec::new(
                format!("request_{request}"),
                prompt,
                decode_steps,
                request,
                sampling,
            )
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("prefix demo must retain exactly eight request specs"))?;
    Ok(DemoPlan {
        command,
        context,
        request_policy,
        slot_seeds,
        requests,
        expected_cache_endpoint,
        measured_waves,
    })
}

fn validate_measured_wave_count(measured_waves: usize) -> Result<()> {
    ensure!(
        measured_waves >= LLAMA_PREFIX_DEMO_MIN_MEASURED_WAVES,
        "llama.cpp warm-prefix comparison requires at least \
         {LLAMA_PREFIX_DEMO_MIN_MEASURED_WAVES} retained measured waves; got {measured_waves}"
    );
    Ok(())
}

fn validate_prompt_set(
    seed: LlamaPrefixDemoPrompt<'_>,
    requests: &[LlamaPrefixDemoPrompt<'_>; PREFIX_DEMO_CONCURRENCY],
    expected_cache_endpoint: usize,
) -> Result<usize> {
    ensure!(
        expected_cache_endpoint > 0,
        "expected cache endpoint must be positive"
    );
    validate_prompt("seed", seed)?;
    ensure!(
        seed.token_ids.len() == expected_cache_endpoint,
        "each fixed llama-server slot must be seeded at the exact cache endpoint: seed has {} \
         tokens, expected {expected_cache_endpoint}",
        seed.token_ids.len()
    );
    let mut maximum_prompt_tokens = seed.token_ids.len();
    let mut common_prefix = seed.token_ids.to_vec();
    for (request, prompt) in requests.iter().copied().enumerate() {
        validate_prompt(&format!("request_{request}"), prompt)?;
        ensure!(
            prompt.token_ids.len() > expected_cache_endpoint,
            "request {request} has no task suffix after the expected cache endpoint"
        );
        let shared = common_prefix_len(&common_prefix, prompt.token_ids);
        common_prefix.truncate(shared);
        maximum_prompt_tokens = maximum_prompt_tokens.max(prompt.token_ids.len());
    }
    ensure!(
        common_prefix.len() == expected_cache_endpoint,
        "seed plus eight prompts have an exact common prefix of {} tokens; expected {}",
        common_prefix.len(),
        expected_cache_endpoint
    );
    Ok(maximum_prompt_tokens)
}

fn validate_prompt(label: &str, prompt: LlamaPrefixDemoPrompt<'_>) -> Result<()> {
    ensure!(
        !prompt.rendered_prompt.is_empty(),
        "{label} rendered prompt cannot be empty"
    );
    ensure!(
        !prompt.token_ids.is_empty(),
        "{label} token IDs cannot be empty"
    );
    Ok(())
}

fn common_prefix_len(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn context_geometry(
    maximum_prompt_tokens: usize,
    decode_steps: usize,
    manifest_kv_capacity_tokens: usize,
    logical_batch_tokens: usize,
    physical_ubatch_tokens: usize,
) -> Result<LlamaPrefixDemoContextGeometry> {
    ensure!(maximum_prompt_tokens > 0, "maximum prompt must be positive");
    ensure!(decode_steps > 0, "decode_steps must be positive");
    ensure!(
        manifest_kv_capacity_tokens > 0,
        "manifest KV capacity must be positive"
    );
    ensure!(
        logical_batch_tokens > 0
            && physical_ubatch_tokens > 0
            && logical_batch_tokens >= physical_ubatch_tokens,
        "llama-server logical batch must be positive and at least its physical ubatch"
    );
    let minimum_tokens_per_slot = maximum_prompt_tokens
        .checked_add(decode_steps)
        .and_then(|tokens| tokens.checked_add(PREFIX_DEMO_CONTEXT_HEADROOM_TOKENS))
        .context("prefix-demo per-slot context requirement overflow")?;
    let minimum_total_context_tokens = minimum_tokens_per_slot
        .checked_mul(PREFIX_DEMO_CONCURRENCY)
        .context("prefix-demo total context requirement overflow")?;
    let requested_total = manifest_kv_capacity_tokens.max(minimum_total_context_tokens);
    let total_alignment = LLAMA_CONTEXT_ALIGNMENT_PER_SLOT
        .checked_mul(PREFIX_DEMO_CONCURRENCY)
        .context("prefix-demo context alignment overflow")?;
    let configured_total_context_tokens = round_up(requested_total, total_alignment)
        .context("prefix-demo total context alignment overflow")?;
    let configured_tokens_per_slot = configured_total_context_tokens / PREFIX_DEMO_CONCURRENCY;
    ensure!(
        configured_tokens_per_slot >= minimum_tokens_per_slot,
        "configured per-slot context is smaller than the request requirement"
    );
    Ok(LlamaPrefixDemoContextGeometry {
        slots: PREFIX_DEMO_CONCURRENCY,
        maximum_prompt_tokens,
        decode_tokens: decode_steps,
        headroom_tokens_per_slot: PREFIX_DEMO_CONTEXT_HEADROOM_TOKENS,
        minimum_tokens_per_slot,
        minimum_total_context_tokens,
        configured_total_context_tokens,
        configured_tokens_per_slot,
        manifest_kv_capacity_tokens,
        cache_reuse_chunk_tokens: PREFIX_DEMO_CACHE_REUSE_TOKENS,
        logical_batch_tokens,
        physical_ubatch_tokens,
    })
}

fn round_up(value: usize, alignment: usize) -> Option<usize> {
    let remainder = value.checked_rem(alignment)?;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

fn llama_server_arguments(
    model: &Path,
    llama: &super::LlamaCppRequirement,
    context_tokens: usize,
    logical_batch_tokens: usize,
    physical_ubatch_tokens: usize,
    threads: usize,
    port: u16,
) -> Result<Vec<String>> {
    ensure!(
        logical_batch_tokens > 0
            && physical_ubatch_tokens > 0
            && logical_batch_tokens >= physical_ubatch_tokens,
        "llama-server logical batch must be positive and at least its physical ubatch"
    );
    let model = model
        .to_str()
        .with_context(|| format!("verified model path is not UTF-8: {}", model.display()))?;
    let flash_attention = match llama.flash_attention {
        LlamaFlashAttention::On => "on",
        LlamaFlashAttention::Off => "off",
        LlamaFlashAttention::Auto => "auto",
    };
    let mmap = if llama.mmap { "--mmap" } else { "--no-mmap" };
    Ok(vec![
        "--model".to_owned(),
        model.to_owned(),
        "--host".to_owned(),
        LOOPBACK.to_string(),
        "--port".to_owned(),
        port.to_string(),
        "--parallel".to_owned(),
        PREFIX_DEMO_CONCURRENCY.to_string(),
        "--ctx-size".to_owned(),
        context_tokens.to_string(),
        "--batch-size".to_owned(),
        logical_batch_tokens.to_string(),
        "--ubatch-size".to_owned(),
        physical_ubatch_tokens.to_string(),
        "--threads".to_owned(),
        threads.to_string(),
        "--threads-batch".to_owned(),
        threads.to_string(),
        "--threads-http".to_owned(),
        PREFIX_DEMO_CONCURRENCY.to_string(),
        "--n-gpu-layers".to_owned(),
        llama.gpu_layers.to_string(),
        "--flash-attn".to_owned(),
        flash_attention.to_owned(),
        "--cache-type-k".to_owned(),
        llama.cache_type_k.clone(),
        "--cache-type-v".to_owned(),
        llama.cache_type_v.clone(),
        mmap.to_owned(),
        "--fit".to_owned(),
        "off".to_owned(),
        "--no-kv-unified".to_owned(),
        "--cont-batching".to_owned(),
        "--cache-prompt".to_owned(),
        "--cache-reuse".to_owned(),
        PREFIX_DEMO_CACHE_REUSE_TOKENS.to_string(),
        "--cache-ram".to_owned(),
        "0".to_owned(),
        "--no-cache-idle-slots".to_owned(),
        "--slot-prompt-similarity".to_owned(),
        "0".to_owned(),
        "--no-ui".to_owned(),
        "--metrics".to_owned(),
        "--no-slots".to_owned(),
        "--reasoning".to_owned(),
        "off".to_owned(),
        "--sse-ping-interval".to_owned(),
        "-1".to_owned(),
        "--timeout".to_owned(),
        DEFAULT_REQUEST_TIMEOUT.as_secs().to_string(),
        "--offline".to_owned(),
        "--log-disable".to_owned(),
    ])
}

#[derive(Debug, Clone, Copy)]
struct ExpectedSampling {
    seed: u64,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repeat_penalty: f64,
}

impl ExpectedSampling {
    fn from_verified(verified: &VerifiedScoreboard) -> Result<Self> {
        let sampling = verified.manifest.workload.sampling;
        ensure!(
            sampling.temperature.to_bits() == 0.0_f64.to_bits()
                && sampling.top_k == 0
                && sampling.top_p.to_bits() == 1.0_f64.to_bits()
                && sampling.repetition_penalty.to_bits() == 1.0_f32.to_bits()
                && sampling.ignore_stop_tokens,
            "prefix demo requires the manifest's exact greedy, ignore-stop sampling policy"
        );
        Ok(Self {
            seed: sampling.seed,
            temperature: sampling.temperature,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
            repeat_penalty: f64::from(sampling.repetition_penalty),
        })
    }

    fn request_policy(self, decode_steps: usize) -> LlamaPrefixDemoRequestPolicy {
        LlamaPrefixDemoRequestPolicy {
            decode_tokens: decode_steps,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repeat_penalty,
            seed: self.seed,
            ignore_eos: true,
            stream: true,
            return_tokens: true,
            cache_prompt: true,
            n_keep: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct CompletionSpec {
    label: String,
    slot_id: usize,
    prompt: String,
    prompt_token_ids: Vec<u32>,
    completion_tokens: usize,
    body: Arc<Vec<u8>>,
    nonstream_body: Arc<Vec<u8>>,
    sampling: ExpectedSampling,
}

impl CompletionSpec {
    fn new(
        label: String,
        prompt: LlamaPrefixDemoPrompt<'_>,
        completion_tokens: usize,
        slot_id: usize,
        sampling: ExpectedSampling,
    ) -> Result<Self> {
        ensure!(
            slot_id < PREFIX_DEMO_CONCURRENCY,
            "llama-server slot ID {slot_id} is outside the configured eight-slot range"
        );
        let request = |stream| CompletionRequest {
            prompt: prompt.rendered_prompt,
            id_slot: slot_id,
            n_predict: completion_tokens,
            temperature: sampling.temperature,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
            min_p: 0.0,
            top_n_sigma: -1.0,
            xtc_probability: 0.0,
            typical_p: 1.0,
            repeat_last_n: 0,
            repeat_penalty: sampling.repeat_penalty,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            dry_multiplier: 0.0,
            mirostat: 0,
            seed: sampling.seed,
            ignore_eos: true,
            stream,
            return_tokens: true,
            cache_prompt: true,
            n_keep: 0,
            n_probs: 0,
            timings_per_token: false,
            post_sampling_probs: false,
            backend_sampling: false,
        };
        Ok(Self {
            label,
            slot_id,
            prompt: prompt.rendered_prompt.to_owned(),
            prompt_token_ids: prompt.token_ids.to_vec(),
            completion_tokens,
            body: Arc::new(serde_json::to_vec(&request(true))?),
            nonstream_body: Arc::new(serde_json::to_vec(&request(false))?),
            sampling,
        })
    }
}

#[derive(Serialize)]
struct CompletionRequest<'a> {
    prompt: &'a str,
    id_slot: usize,
    n_predict: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    min_p: f64,
    top_n_sigma: f64,
    xtc_probability: f64,
    typical_p: f64,
    repeat_last_n: usize,
    repeat_penalty: f64,
    presence_penalty: f64,
    frequency_penalty: f64,
    dry_multiplier: f64,
    mirostat: usize,
    seed: u64,
    ignore_eos: bool,
    stream: bool,
    return_tokens: bool,
    cache_prompt: bool,
    n_keep: usize,
    n_probs: usize,
    timings_per_token: bool,
    post_sampling_probs: bool,
    backend_sampling: bool,
}

struct RunEvidence {
    health_wait: Duration,
    tokenizer_parity: Vec<LlamaPrefixDemoTokenizerEvidence>,
    slot_seeds: Vec<LlamaPrefixDemoRequestEvidence>,
    measured_waves: Vec<LlamaPrefixDemoWaveEvidence>,
    determinism: LlamaPrefixDemoDeterminismEvidence,
}

fn finish_run_and_cleanup<T>(
    run_result: Result<T>,
    stop_result: Result<()>,
    port_result: Result<()>,
) -> Result<T> {
    let mut failures = Vec::new();
    let run = match run_result {
        Ok(run) => Some(run),
        Err(error) => {
            failures.push(format!("run failed: {error:#}"));
            None
        }
    };
    if let Err(error) = stop_result {
        failures.push(format!("child stop/reap failed: {error:#}"));
    }
    if let Err(error) = port_result {
        failures.push(format!("port-close verification failed: {error:#}"));
    }
    if !failures.is_empty() {
        bail!(
            "llama-server prefix demo run or cleanup failed: {}",
            failures.join("; ")
        );
    }
    run.context("successful llama-server prefix demo retained no run evidence")
}

#[derive(Debug)]
struct ObservedCompletion {
    request: usize,
    label: String,
    prompt_tokens: usize,
    prompt_token_ids_sha256: String,
    submitted_at: Instant,
    completed_at: Instant,
    time_to_first_token: Duration,
    latency: Duration,
    content: String,
    streamed_content: Option<String>,
    token_ids: Vec<u32>,
    timings: WireTimings,
}

#[derive(Debug)]
struct ObservedWave {
    completed_at: Instant,
    requests: Vec<ObservedCompletion>,
}

#[derive(Debug)]
struct ObservedTokenCompletion {
    request: usize,
    label: String,
    content: String,
    token_ids: Vec<u32>,
}

fn seed_all_slots(
    address: SocketAddr,
    specs: &[CompletionSpec; PREFIX_DEMO_CONCURRENCY],
) -> Result<Vec<LlamaPrefixDemoRequestEvidence>> {
    let mut evidence = Vec::with_capacity(PREFIX_DEMO_CONCURRENCY);
    for (slot_id, spec) in specs.iter().enumerate() {
        ensure!(
            spec.slot_id == slot_id,
            "seed spec {} targets slot {} instead of fixed slot {slot_id}",
            spec.label,
            spec.slot_id
        );
        let mut observed = run_completion_indexed(address, spec, slot_id)?;
        ensure!(
            observed.timings.cache_n == 0
                && observed.timings.prompt_n == spec.prompt_token_ids.len(),
            "fresh llama-server slot {slot_id} seed reported cache_n={} and prompt_n={}; expected \
             cache_n=0 and prompt_n={}",
            observed.timings.cache_n,
            observed.timings.prompt_n,
            spec.prompt_token_ids.len()
        );
        let replay = run_nonstream_completion(address, spec, slot_id)?;
        attach_replay(&mut observed, &replay)?;
        evidence.push(request_evidence(
            format!("seed_slot_{slot_id}"),
            None,
            observed,
        )?);
    }
    ensure!(
        evidence.len() == PREFIX_DEMO_CONCURRENCY,
        "cold seeding did not retain evidence for all eight slots"
    );
    Ok(evidence)
}

fn restore_seed_slots(
    address: SocketAddr,
    specs: &[CompletionSpec; PREFIX_DEMO_CONCURRENCY],
) -> Result<()> {
    for (slot_id, spec) in specs.iter().enumerate() {
        ensure!(
            spec.slot_id == slot_id,
            "restore spec {} targets slot {} instead of fixed slot {slot_id}",
            spec.label,
            spec.slot_id
        );
        let restored = run_nonstream_completion(address, spec, slot_id).with_context(|| {
            format!("failed to restore exact seed in llama-server slot {slot_id}")
        })?;
        ensure!(
            restored.request == slot_id && restored.label == spec.label,
            "llama-server seed restore changed slot identity"
        );
    }
    Ok(())
}

fn execute_wave(
    address: SocketAddr,
    specs: &[CompletionSpec; PREFIX_DEMO_CONCURRENCY],
) -> Result<ObservedWave> {
    let barrier = Arc::new(Barrier::new(PREFIX_DEMO_CONCURRENCY + 1));
    let mut workers = Vec::with_capacity(PREFIX_DEMO_CONCURRENCY);
    for (request, spec) in specs.iter().cloned().enumerate() {
        ensure!(
            spec.slot_id == request,
            "request spec {} targets slot {} instead of fixed slot {request}",
            spec.label,
            spec.slot_id
        );
        let worker_barrier = Arc::clone(&barrier);
        workers.push(thread::spawn(move || {
            worker_barrier.wait();
            run_completion_indexed(address, &spec, request)
        }));
    }
    barrier.wait();

    let mut requests = Vec::with_capacity(PREFIX_DEMO_CONCURRENCY);
    let mut first_error = None;
    for worker in workers {
        let result = worker
            .join()
            .map_err(|_| anyhow::anyhow!("llama-server prefix-demo worker panicked"))
            .and_then(|result| result);
        match result {
            Ok(request) => requests.push(request),
            Err(error) if first_error.is_none() => first_error = Some(error),
            Err(_error) => {}
        }
    }
    if let Some(error) = first_error {
        return Err(error);
    }
    requests.sort_unstable_by_key(|request| request.request);
    let completed_at = requests
        .iter()
        .map(|request| request.completed_at)
        .max()
        .context("llama-server prefix-demo wave completed no requests")?;
    Ok(ObservedWave {
        completed_at,
        requests,
    })
}

fn execute_nonstream_wave(
    address: SocketAddr,
    specs: &[CompletionSpec; PREFIX_DEMO_CONCURRENCY],
) -> Result<Vec<ObservedTokenCompletion>> {
    let barrier = Arc::new(Barrier::new(PREFIX_DEMO_CONCURRENCY + 1));
    let mut workers = Vec::with_capacity(PREFIX_DEMO_CONCURRENCY);
    for (request, spec) in specs.iter().cloned().enumerate() {
        ensure!(
            spec.slot_id == request,
            "replay spec {} targets slot {} instead of fixed slot {request}",
            spec.label,
            spec.slot_id
        );
        let worker_barrier = Arc::clone(&barrier);
        workers.push(thread::spawn(move || {
            worker_barrier.wait();
            run_nonstream_completion(address, &spec, request)
        }));
    }
    barrier.wait();
    let mut completions = Vec::with_capacity(PREFIX_DEMO_CONCURRENCY);
    let mut first_error = None;
    for worker in workers {
        let result = worker
            .join()
            .map_err(|_| anyhow::anyhow!("llama-server token replay worker panicked"))
            .and_then(|result| result);
        match result {
            Ok(completion) => completions.push(completion),
            Err(error) if first_error.is_none() => first_error = Some(error),
            Err(_error) => {}
        }
    }
    if let Some(error) = first_error {
        return Err(error);
    }
    completions.sort_unstable_by_key(|completion| completion.request);
    ensure!(
        completions.len() == PREFIX_DEMO_CONCURRENCY,
        "non-stream token replay lost one or more requests"
    );
    Ok(completions)
}

fn run_nonstream_completion(
    address: SocketAddr,
    spec: &CompletionSpec,
    request: usize,
) -> Result<ObservedTokenCompletion> {
    let response = buffered_request(
        address,
        "POST",
        "/completion",
        Some(spec.nonstream_body.as_slice()),
        DEFAULT_REQUEST_TIMEOUT,
    )?;
    ensure!(
        response.status == 200,
        "{} non-stream /completion returned HTTP {}",
        spec.label,
        response.status
    );
    let event: WireCompletionEvent = serde_json::from_slice(&response.body)
        .context("invalid llama-server non-stream completion JSON")?;
    ensure!(event.index == 0, "llama-server returned a non-zero index");
    ensure!(event.stop, "non-stream replay omitted its terminal result");
    ensure!(
        event.tokens_evaluated == spec.prompt_token_ids.len(),
        "{} replay evaluated {} prompt tokens; expected {}",
        spec.label,
        event.tokens_evaluated,
        spec.prompt_token_ids.len()
    );
    ensure!(
        event.tokens_predicted == spec.completion_tokens
            && event.tokens.len() == spec.completion_tokens,
        "{} non-stream replay did not return every generated token ID",
        spec.label
    );
    ensure!(
        event.truncated == Some(false) && event.stop_type.as_deref() == Some("limit"),
        "{} non-stream replay truncated or changed the fixed completion boundary",
        spec.label
    );
    event
        .generation_settings
        .context("non-stream replay omitted generation_settings")?
        .validate(spec, false)?;
    event
        .timings
        .context("non-stream replay omitted timings")?
        .validate(spec)?;
    Ok(ObservedTokenCompletion {
        request,
        label: spec.label.clone(),
        content: event.content,
        token_ids: event.tokens,
    })
}

fn attach_replay_tokens(
    measured: &mut [ObservedCompletion],
    replay: &[ObservedTokenCompletion],
) -> Result<()> {
    ensure!(
        measured.len() == PREFIX_DEMO_CONCURRENCY && replay.len() == PREFIX_DEMO_CONCURRENCY,
        "token attachment requires one complete measured and replay wave"
    );
    for (measured, replay) in measured.iter_mut().zip(replay) {
        attach_replay(measured, replay)?;
    }
    Ok(())
}

fn attach_replay(
    measured: &mut ObservedCompletion,
    replay: &ObservedTokenCompletion,
) -> Result<()> {
    ensure!(
        measured.request == replay.request && measured.label == replay.label,
        "token replay request identity changed"
    );
    ensure!(
        measured.streamed_content.is_none() && measured.token_ids.is_empty(),
        "token replay was attached more than once"
    );
    measured.streamed_content = Some(std::mem::replace(
        &mut measured.content,
        replay.content.clone(),
    ));
    measured.token_ids.clone_from(&replay.token_ids);
    Ok(())
}

fn run_completion_indexed(
    address: SocketAddr,
    spec: &CompletionSpec,
    request: usize,
) -> Result<ObservedCompletion> {
    let submitted_at = Instant::now();
    let stream = connect(address, DEFAULT_REQUEST_TIMEOUT)?;
    let mut reader = BufReader::new(stream);
    write_request(
        reader.get_mut(),
        address,
        "POST",
        "/completion",
        Some(spec.body.as_slice()),
        "text/event-stream",
    )?;
    let head = read_http_head(&mut reader)?;
    ensure!(
        head.status == 200,
        "{} /completion returned HTTP {}",
        spec.label,
        head.status
    );
    ensure!(
        head.headers
            .get("content-type")
            .is_some_and(|value| value.starts_with("text/event-stream")),
        "{} /completion did not return text/event-stream",
        spec.label
    );
    ensure!(
        header_has_chunked_encoding(&head.headers),
        "{} /completion was not HTTP chunked",
        spec.label
    );

    let mut decoder = SseDecoder::default();
    let mut observer = CompletionObserver::default();
    loop {
        let Some(chunk) = read_http_chunk(&mut reader)? else {
            bail!(
                "{} chunked stream ended before its terminal event",
                spec.label
            );
        };
        let arrived_at = Instant::now();
        for data in decoder.push(&chunk)? {
            observer.observe(&data, arrived_at, submitted_at, spec)?;
            if observer.terminal.is_some() {
                let completed_at = Instant::now();
                let latency = completed_at
                    .checked_duration_since(submitted_at)
                    .context("llama-server completion preceded request submission")?;
                return observer.finish(request, submitted_at, latency, completed_at, spec);
            }
        }
    }
}

#[derive(Default)]
struct CompletionObserver {
    last_tokens_predicted: usize,
    content: String,
    time_to_first_token: Option<Duration>,
    terminal: Option<WireTimings>,
}

impl CompletionObserver {
    #[allow(clippy::too_many_lines)]
    fn observe(
        &mut self,
        data: &str,
        arrived_at: Instant,
        submitted_at: Instant,
        spec: &CompletionSpec,
    ) -> Result<()> {
        ensure!(
            data != "[DONE]",
            "llama-server sent [DONE] before terminal timings"
        );
        ensure!(
            self.terminal.is_none(),
            "llama-server sent data after terminal event"
        );
        let event: WireCompletionEvent =
            serde_json::from_str(data).context("invalid llama-server completion SSE JSON")?;
        ensure!(event.index == 0, "llama-server returned a non-zero index");
        ensure!(
            event.tokens_evaluated == spec.prompt_token_ids.len(),
            "{} event evaluated {} prompt tokens; expected {}",
            spec.label,
            event.tokens_evaluated,
            spec.prompt_token_ids.len()
        );
        if event.stop {
            ensure!(
                event.tokens.is_empty(),
                "terminal llama-server event emitted token IDs"
            );
            ensure!(
                event.tokens_predicted == spec.completion_tokens
                    && event.tokens_predicted >= self.last_tokens_predicted,
                "terminal predicted count differs from the fixed completion boundary"
            );
            ensure!(
                event.truncated == Some(false),
                "llama-server truncated a prefix-demo prompt"
            );
            ensure!(
                event.stop_type.as_deref() == Some("limit"),
                "llama-server did not stop at the fixed token limit"
            );
            let settings = event
                .generation_settings
                .context("terminal event omitted generation_settings")?;
            settings.validate(spec, true)?;
            let timings = event.timings.context("terminal event omitted timings")?;
            timings.validate(spec)?;
            self.terminal = Some(timings);
            return Ok(());
        }

        ensure!(
            event.timings.is_none(),
            "non-terminal llama-server event included timings"
        );
        ensure!(
            !event.content.is_empty() || !event.tokens.is_empty(),
            "non-terminal llama-server event had no visible output"
        );
        if self.time_to_first_token.is_none() {
            self.time_to_first_token = Some(
                arrived_at
                    .checked_duration_since(submitted_at)
                    .context("first llama-server token preceded request submission")?,
            );
        }
        self.content.push_str(&event.content);
        ensure!(
            event.tokens_predicted > self.last_tokens_predicted,
            "streamed tokens_predicted did not advance"
        );
        ensure!(
            event.tokens_predicted <= spec.completion_tokens,
            "llama-server exceeded the completion token budget"
        );
        self.last_tokens_predicted = event.tokens_predicted;
        Ok(())
    }

    fn finish(
        self,
        request: usize,
        submitted_at: Instant,
        latency: Duration,
        completed_at: Instant,
        spec: &CompletionSpec,
    ) -> Result<ObservedCompletion> {
        let time_to_first_token = self
            .time_to_first_token
            .with_context(|| format!("{} finished without a streamed token", spec.label))?;
        ensure!(
            latency >= time_to_first_token,
            "{} latency is shorter than TTFT",
            spec.label
        );
        Ok(ObservedCompletion {
            request,
            label: spec.label.clone(),
            prompt_tokens: spec.prompt_token_ids.len(),
            prompt_token_ids_sha256: token_ids_sha256(&spec.prompt_token_ids),
            submitted_at,
            completed_at,
            time_to_first_token,
            latency,
            content: self.content,
            streamed_content: None,
            token_ids: Vec::new(),
            timings: self
                .terminal
                .context("request finished without terminal server timings")?,
        })
    }
}

fn request_evidence(
    label: String,
    request: Option<usize>,
    observed: ObservedCompletion,
) -> Result<LlamaPrefixDemoRequestEvidence> {
    if let Some(request) = request {
        ensure!(
            request == observed.request,
            "request evidence index {request} differs from fixed slot {}",
            observed.request
        );
    }
    let streamed_content = observed
        .streamed_content
        .context("request evidence has no measured streaming content")?;
    let streamed_content_matches_replay = streamed_content == observed.content;
    let completion_token_ids_sha256 = token_ids_sha256(&observed.token_ids);
    Ok(LlamaPrefixDemoRequestEvidence {
        label,
        request,
        slot_id: observed.request,
        prompt_tokens: observed.prompt_tokens,
        prompt_token_ids_sha256: observed.prompt_token_ids_sha256,
        content: observed.content,
        streamed_content,
        streamed_content_matches_replay,
        content_source: concat!(
            "unmeasured non-stream replay; the measured streaming wave is retained separately ",
            "because incomplete UTF-8 may be buffered or omitted"
        )
        .to_owned(),
        completion_tokens: observed.token_ids.len(),
        completion_token_ids: observed.token_ids,
        completion_token_ids_sha256,
        completion_token_source:
            "unmeasured non-stream return_tokens replay; streaming can buffer incomplete UTF-8"
                .to_owned(),
        external_time_to_first_token_ms: millis(observed.time_to_first_token),
        external_latency_ms: millis(observed.latency),
        terminal: LlamaPrefixDemoTerminalEvidence {
            server_reported_cache_n: observed.timings.cache_n,
            server_reported_prompt_n: observed.timings.prompt_n,
            server_reported_prompt_ms: observed.timings.prompt_ms,
            server_reported_prompt_tokens_per_second: observed.timings.prompt_per_second,
            server_reported_predicted_n: observed.timings.predicted_n,
            server_reported_predicted_ms: observed.timings.predicted_ms,
            server_reported_predicted_tokens_per_second: observed.timings.predicted_per_second,
        },
    })
}

fn aggregate_wave(
    wave_index: usize,
    wave: ObservedWave,
    expected_cache_endpoint: usize,
) -> Result<LlamaPrefixDemoWaveEvidence> {
    ensure!(
        wave.requests.len() == PREFIX_DEMO_CONCURRENCY,
        "prefix-demo wave must retain exactly eight requests"
    );
    let earliest_submitted_at = wave
        .requests
        .iter()
        .map(|request| request.submitted_at)
        .min()
        .context("prefix-demo wave retained no request submissions")?;
    let wall_time = wave
        .completed_at
        .checked_duration_since(earliest_submitted_at)
        .context("prefix-demo wave completed before its earliest request submission")?;
    ensure!(
        !wall_time.is_zero(),
        "prefix-demo wall time must be positive"
    );
    let total_completion_tokens = wave.requests.iter().try_fold(0_u64, |total, request| {
        total
            .checked_add(request.token_ids.len() as u64)
            .context("prefix-demo completion-token total overflow")
    })?;
    let ttft = wave
        .requests
        .iter()
        .map(|request| millis(request.time_to_first_token))
        .collect::<Vec<_>>();
    let latency = wave
        .requests
        .iter()
        .map(|request| millis(request.latency))
        .collect::<Vec<_>>();
    let requests = wave
        .requests
        .into_iter()
        .map(|observed| {
            let request = observed.request;
            let label = observed.label.clone();
            request_evidence(label, Some(request), observed)
        })
        .collect::<Result<Vec<_>>>()?;
    let cache = cache_observation(&requests, expected_cache_endpoint)
        .with_context(|| format!("llama.cpp measured wave {wave_index} is not exact-cache warm"))?;
    Ok(LlamaPrefixDemoWaveEvidence {
        wave_index,
        concurrency: PREFIX_DEMO_CONCURRENCY,
        request_count: PREFIX_DEMO_CONCURRENCY,
        total_completion_tokens,
        external_wall_time_ms: millis(wall_time),
        aggregate_output_tokens_per_second: aggregate_throughput(
            total_completion_tokens,
            wall_time,
        )?,
        external_time_to_first_token_ms: PrefixDemoTimingSummary::from_samples(&ttft)?,
        external_latency_ms: PrefixDemoTimingSummary::from_samples(&latency)?,
        requests,
        cache,
    })
}

fn verify_replay(
    first: &[ObservedTokenCompletion],
    second: &[ObservedTokenCompletion],
) -> Result<LlamaPrefixDemoDeterminismEvidence> {
    ensure!(
        first.len() == PREFIX_DEMO_CONCURRENCY && second.len() == PREFIX_DEMO_CONCURRENCY,
        "determinism verification requires two complete non-stream eight-request replays"
    );
    let mut first_digests = Vec::with_capacity(PREFIX_DEMO_CONCURRENCY);
    let mut second_digests = Vec::with_capacity(PREFIX_DEMO_CONCURRENCY);
    for (first, second) in first.iter().zip(second) {
        ensure!(
            first.request == second.request && first.label == second.label,
            "determinism replay request identity changed"
        );
        ensure!(
            first.token_ids == second.token_ids,
            "greedy llama-server completion token IDs changed for request {}",
            first.request
        );
        first_digests.push(token_ids_sha256(&first.token_ids));
        second_digests.push(token_ids_sha256(&second.token_ids));
    }
    Ok(LlamaPrefixDemoDeterminismEvidence {
        replay_concurrency: PREFIX_DEMO_CONCURRENCY,
        replay_excluded_from_performance_measurement: true,
        exact_per_request: true,
        first_nonstream_replay_token_ids_sha256: first_digests,
        second_nonstream_replay_token_ids_sha256: second_digests,
    })
}

fn cache_observation(
    requests: &[LlamaPrefixDemoRequestEvidence],
    expected_cache_endpoint: usize,
) -> Result<LlamaPrefixDemoCacheObservation> {
    ensure!(
        requests.len() == PREFIX_DEMO_CONCURRENCY,
        "cache observation requires exactly eight requests"
    );
    for (request_index, request) in requests.iter().enumerate() {
        ensure!(
            request.request == Some(request_index) && request.slot_id == request_index,
            "cache observation request {request_index} is not bound to fixed slot {request_index}"
        );
    }
    let cache_n = requests
        .iter()
        .map(|request| request.terminal.server_reported_cache_n)
        .collect::<Vec<_>>();
    let prompt_n = requests
        .iter()
        .map(|request| request.terminal.server_reported_prompt_n)
        .collect::<Vec<_>>();
    let expected_prompt_n = requests
        .iter()
        .map(|request| {
            request
                .prompt_tokens
                .checked_sub(expected_cache_endpoint)
                .context("measured request is shorter than the exact cache endpoint")
        })
        .collect::<Result<Vec<_>>>()?;
    let requests_satisfying_exact_contract = requests
        .iter()
        .zip(&expected_prompt_n)
        .filter(|(request, expected_prompt_n)| {
            request.terminal.server_reported_cache_n == expected_cache_endpoint
                && request.terminal.server_reported_prompt_n == **expected_prompt_n
        })
        .count();
    ensure!(
        requests_satisfying_exact_contract == PREFIX_DEMO_CONCURRENCY,
        "exact warm-prefix contract failed: expected every request to report \
         cache_n={expected_cache_endpoint} and prompt_n=prompt_tokens-{expected_cache_endpoint}; \
         cache_n={cache_n:?}, prompt_n={prompt_n:?}, \
         expected_prompt_n={expected_prompt_n:?}",
    );
    let minimum = cache_n
        .iter()
        .copied()
        .min()
        .context("cache observation retained no requests")?;
    let maximum = cache_n
        .iter()
        .copied()
        .max()
        .context("cache observation retained no requests")?;
    Ok(LlamaPrefixDemoCacheObservation {
        exact_common_prefix_tokens: expected_cache_endpoint,
        requests_satisfying_exact_contract,
        minimum_server_reported_cache_n: minimum,
        maximum_server_reported_cache_n: maximum,
        server_reported_cache_n_by_request: cache_n,
        server_reported_prompt_n_by_request: prompt_n,
        expected_prompt_n_by_request: expected_prompt_n,
        interpretation: concat!(
            "every request is fixed to the id_slot seeded for it and this wave is retained only ",
            "after cache_n equals the exact shared endpoint and prompt_n equals the task suffix ",
            "length for all eight requests. This proves logical same-slot warm reuse only: global ",
            "cache RAM, idle-slot cache, and KV shifting are disabled, and cache_n is not converted ",
            "into MacLLM page references, shared-block counts, or physical memory savings."
        )
        .to_owned(),
    })
}

#[derive(Deserialize)]
struct WireCompletionEvent {
    index: usize,
    #[serde(default)]
    content: String,
    #[serde(default)]
    tokens: Vec<u32>,
    #[serde(default)]
    stop: bool,
    tokens_predicted: usize,
    tokens_evaluated: usize,
    #[serde(default)]
    truncated: Option<bool>,
    #[serde(default)]
    stop_type: Option<String>,
    #[serde(default)]
    generation_settings: Option<WireGenerationSettings>,
    #[serde(default)]
    timings: Option<WireTimings>,
}

#[derive(Deserialize)]
struct WireGenerationSettings {
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
    #[serde(default)]
    cache_prompt: Option<bool>,
}

impl WireGenerationSettings {
    fn validate(&self, spec: &CompletionSpec, expected_stream: bool) -> Result<()> {
        ensure!(
            self.seed == spec.sampling.seed,
            "llama-server changed the greedy seed"
        );
        ensure!(
            self.temperature.to_bits() == spec.sampling.temperature.to_bits()
                && self.top_k == spec.sampling.top_k
                && self.top_p.to_bits() == spec.sampling.top_p.to_bits()
                && self.repeat_penalty.to_bits() == spec.sampling.repeat_penalty.to_bits(),
            "llama-server generation settings differ from the manifest"
        );
        ensure!(self.ignore_eos, "llama-server did not ignore stop tokens");
        ensure!(
            self.stream == expected_stream,
            "llama-server changed the requested streaming mode"
        );
        ensure!(
            self.max_tokens == spec.completion_tokens
                && self.n_predict == spec.completion_tokens
                && self.n_keep == 0,
            "llama-server changed the fixed completion boundary"
        );
        ensure!(
            self.cache_prompt.unwrap_or(true),
            "llama-server terminal settings disabled prompt caching"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct WireTimings {
    cache_n: usize,
    prompt_n: usize,
    prompt_ms: f64,
    prompt_per_second: f64,
    predicted_n: usize,
    predicted_ms: f64,
    predicted_per_second: f64,
}

impl WireTimings {
    fn validate(&self, spec: &CompletionSpec) -> Result<()> {
        let accounted_prompt = self
            .cache_n
            .checked_add(self.prompt_n)
            .context("llama-server prompt timing count overflow")?;
        ensure!(
            accounted_prompt == spec.prompt_token_ids.len(),
            "llama-server cache_n + prompt_n differs from the verified prompt length"
        );
        ensure!(
            self.predicted_n == spec.completion_tokens,
            "llama-server predicted_n differs from the fixed token budget"
        );
        ensure!(
            positive_finite(self.prompt_ms)
                && positive_finite(self.prompt_per_second)
                && positive_finite(self.predicted_ms)
                && positive_finite(self.predicted_per_second),
            "llama-server emitted invalid terminal timings"
        );
        ensure!(
            approximately_equal(
                self.prompt_per_second,
                self.prompt_n as f64 * 1_000.0 / self.prompt_ms,
            ) && approximately_equal(
                self.predicted_per_second,
                self.predicted_n as f64 * 1_000.0 / self.predicted_ms,
            ),
            "llama-server timing rates do not match their raw durations"
        );
        Ok(())
    }
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= 1.0e-6 * scale
}

struct LlamaServerProcess {
    child: Option<Child>,
}

impl LlamaServerProcess {
    fn spawn(command: &BenchmarkCommand) -> Result<Self> {
        let child = Command::new(&command.program)
            .args(&command.arguments)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to launch {}", command.program.display()))?;
        Ok(Self { child: Some(child) })
    }

    fn ensure_running(&mut self) -> Result<()> {
        let child = self
            .child
            .as_mut()
            .context("llama-server child is absent")?;
        if let Some(status) = child.try_wait().context("failed to poll llama-server")? {
            bail!("llama-server exited unexpectedly with {status}");
        }
        Ok(())
    }

    fn wait_until_ready(&mut self, address: SocketAddr) -> Result<Duration> {
        let started_at = Instant::now();
        loop {
            self.ensure_running()?;
            if health_is_ready(address).unwrap_or(false) {
                return Ok(started_at.elapsed());
            }
            ensure!(
                started_at.elapsed() < HEALTH_TIMEOUT,
                "llama-server did not become healthy within {} seconds",
                HEALTH_TIMEOUT.as_secs()
            );
            thread::sleep(HEALTH_POLL_INTERVAL);
        }
    }

    fn stop(mut self) -> Result<()> {
        let child = self
            .child
            .as_mut()
            .context("llama-server child is absent")?;
        if child
            .try_wait()
            .context("failed to poll llama-server before shutdown")?
            .is_none()
        {
            child.kill().context("failed to stop llama-server")?;
            let _status = child.wait().context("failed to reap llama-server")?;
        }
        self.child.take();
        Ok(())
    }
}

impl Drop for LlamaServerProcess {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ignored = child.kill();
            let _ignored = child.wait();
        }
    }
}

fn ensure_port_is_closed(address: SocketAddr) -> Result<()> {
    ensure!(
        TcpStream::connect_timeout(&address, PORT_PROBE_TIMEOUT).is_err(),
        "refusing to launch prefix demo because {address} is already accepting connections"
    );
    Ok(())
}

fn wait_until_port_closed(address: SocketAddr) -> Result<()> {
    let started_at = Instant::now();
    loop {
        if TcpStream::connect_timeout(&address, PORT_PROBE_TIMEOUT).is_err() {
            return Ok(());
        }
        ensure!(
            started_at.elapsed() < PORT_CLOSE_TIMEOUT,
            "llama-server port {address} remained open after child shutdown"
        );
        thread::sleep(PORT_POLL_INTERVAL);
    }
}

#[derive(Deserialize)]
struct HealthResponse {
    status: String,
}

fn health_is_ready(address: SocketAddr) -> Result<bool> {
    let response = buffered_request(address, "GET", "/health", None, Duration::from_secs(1))?;
    if response.status != 200 {
        return Ok(false);
    }
    let health: HealthResponse =
        serde_json::from_slice(&response.body).context("invalid llama-server health JSON")?;
    Ok(health.status == "ok")
}

#[derive(Deserialize)]
struct TokenizeResponse {
    tokens: Vec<u32>,
}

fn verify_all_tokenizers(
    address: SocketAddr,
    plan: &DemoPlan,
) -> Result<Vec<LlamaPrefixDemoTokenizerEvidence>> {
    let mut evidence = Vec::with_capacity(PREFIX_DEMO_CONCURRENCY + 1);
    evidence.push(verify_tokenizer(address, &plan.slot_seeds[0])?);
    for request in &plan.requests {
        evidence.push(verify_tokenizer(address, request)?);
    }
    Ok(evidence)
}

fn verify_tokenizer(
    address: SocketAddr,
    spec: &CompletionSpec,
) -> Result<LlamaPrefixDemoTokenizerEvidence> {
    let body = serde_json::to_vec(&serde_json::json!({
        "content": spec.prompt,
        "add_special": false,
    }))?;
    let response = buffered_request(
        address,
        "POST",
        "/tokenize",
        Some(&body),
        DEFAULT_REQUEST_TIMEOUT,
    )?;
    ensure!(
        response.status == 200,
        "{} /tokenize returned HTTP {}: {}",
        spec.label,
        response.status,
        String::from_utf8_lossy(&response.body)
    );
    let response: TokenizeResponse =
        serde_json::from_slice(&response.body).context("invalid llama-server /tokenize JSON")?;
    let expected_digest = token_ids_sha256(&spec.prompt_token_ids);
    let observed_digest = token_ids_sha256(&response.tokens);
    ensure!(
        response.tokens == spec.prompt_token_ids,
        "{} llama-server tokenizer IDs differ from MacLLM rendered-prompt IDs: expected {} tokens/{expected_digest}, observed {} tokens/{observed_digest}",
        spec.label,
        spec.prompt_token_ids.len(),
        response.tokens.len()
    );
    Ok(LlamaPrefixDemoTokenizerEvidence {
        label: spec.label.clone(),
        token_count: spec.prompt_token_ids.len(),
        expected_token_ids_sha256: expected_digest,
        observed_token_ids_sha256: observed_digest,
        exact: true,
        add_special: false,
    })
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>> {
        ensure!(
            self.buffer.len().saturating_add(bytes.len()) <= MAX_HTTP_BODY_BYTES,
            "llama-server SSE buffer exceeded its safety limit"
        );
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some((boundary, separator_len)) = sse_boundary(&self.buffer) {
            let raw = self.buffer[..boundary].to_vec();
            self.buffer.drain(..boundary + separator_len);
            let raw = std::str::from_utf8(&raw).context("llama-server SSE was not UTF-8")?;
            let mut data_lines = Vec::new();
            for line in raw.lines() {
                let line = line.trim_end_matches('\r');
                if let Some(data) = line.strip_prefix("data:") {
                    data_lines.push(data.strip_prefix(' ').unwrap_or(data));
                }
            }
            if !data_lines.is_empty() {
                events.push(data_lines.join("\n"));
            }
        }
        Ok(events)
    }
}

fn sse_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    for index in 0..buffer.len().saturating_sub(1) {
        if buffer[index] != b'\n' {
            continue;
        }
        if buffer[index + 1] == b'\n' {
            return Some((index, 2));
        }
        if buffer[index + 1] == b'\r' && buffer.get(index + 2).is_some_and(|byte| *byte == b'\n') {
            return Some((index, 3));
        }
    }
    None
}

struct BufferedHttpResponse {
    status: u16,
    body: Vec<u8>,
}

fn buffered_request(
    address: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    timeout: Duration,
) -> Result<BufferedHttpResponse> {
    let stream = connect(address, timeout)?;
    let mut reader = BufReader::new(stream);
    write_request(
        reader.get_mut(),
        address,
        method,
        path,
        body,
        "application/json",
    )?;
    let head = read_http_head(&mut reader)?;
    let body = read_http_body(&mut reader, &head.headers)?;
    Ok(BufferedHttpResponse {
        status: head.status,
        body,
    })
}

// Shared HTTP-timing reuse boundary: `connect` + `write_request` +
// `read_http_head` + `read_http_chunk` + `SseDecoder` contain no llama.cpp
// model semantics. `CompletionObserver` and `WireCompletionEvent` above are
// intentionally llama-server-specific validators. A Mac `/completion` runner
// can reuse/extract this framing layer while retaining its own terminal schema.
fn connect(address: SocketAddr, timeout: Duration) -> Result<TcpStream> {
    let stream = TcpStream::connect_timeout(&address, timeout)
        .with_context(|| format!("failed to connect to llama-server at {address}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .context("failed to set llama-server read timeout")?;
    stream
        .set_write_timeout(Some(timeout))
        .context("failed to set llama-server write timeout")?;
    stream
        .set_nodelay(true)
        .context("failed to disable Nagle for llama-server prefix-demo client")?;
    Ok(stream)
}

fn write_request(
    stream: &mut TcpStream,
    address: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    accept: &str,
) -> Result<()> {
    let body = body.unwrap_or_default();
    let content_type = if body.is_empty() {
        ""
    } else {
        "Content-Type: application/json\r\n"
    };
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nAccept: {accept}\r\n{content_type}Connection: close\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream
        .write_all(head.as_bytes())
        .context("failed to write llama-server HTTP request head")?;
    stream
        .write_all(body)
        .context("failed to write llama-server HTTP request body")?;
    stream
        .flush()
        .context("failed to flush llama-server HTTP request")
}

struct HttpHead {
    status: u16,
    headers: BTreeMap<String, String>,
}

fn read_http_head<R: BufRead>(reader: &mut R) -> Result<HttpHead> {
    let status_line = read_http_line(reader)?.context("llama-server returned no HTTP status")?;
    let mut parts = status_line.split_whitespace();
    ensure!(
        parts
            .next()
            .is_some_and(|version| version.starts_with("HTTP/1.")),
        "llama-server returned an unsupported HTTP status line"
    );
    let status = parts
        .next()
        .context("llama-server HTTP status code is absent")?
        .parse::<u16>()
        .context("llama-server HTTP status code is invalid")?;
    let mut headers = BTreeMap::new();
    loop {
        let line = read_http_line(reader)?.context("llama-server HTTP headers ended early")?;
        if line.is_empty() {
            break;
        }
        let (name, value) = line
            .split_once(':')
            .context("llama-server returned a malformed HTTP header")?;
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim().to_owned();
        headers
            .entry(name)
            .and_modify(|existing: &mut String| {
                existing.push(',');
                existing.push_str(&value);
            })
            .or_insert(value);
    }
    Ok(HttpHead { status, headers })
}

fn read_http_line<R: BufRead>(reader: &mut R) -> Result<Option<String>> {
    let mut line = String::new();
    let bytes = reader
        .read_line(&mut line)
        .context("failed to read llama-server HTTP line")?;
    if bytes == 0 {
        return Ok(None);
    }
    ensure!(
        bytes <= MAX_HTTP_LINE_BYTES,
        "llama-server HTTP line exceeded its safety limit"
    );
    while line.ends_with('\n') || line.ends_with('\r') {
        line.pop();
    }
    Ok(Some(line))
}

fn header_has_chunked_encoding(headers: &BTreeMap<String, String>) -> bool {
    headers.get("transfer-encoding").is_some_and(|value| {
        value
            .split(',')
            .any(|encoding| encoding.trim().eq_ignore_ascii_case("chunked"))
    })
}

fn read_http_body<R: BufRead>(
    reader: &mut R,
    headers: &BTreeMap<String, String>,
) -> Result<Vec<u8>> {
    if header_has_chunked_encoding(headers) {
        let mut body = Vec::new();
        while let Some(chunk) = read_http_chunk(reader)? {
            ensure!(
                body.len().saturating_add(chunk.len()) <= MAX_HTTP_BODY_BYTES,
                "llama-server HTTP body exceeded its safety limit"
            );
            body.extend_from_slice(&chunk);
        }
        return Ok(body);
    }
    if let Some(length) = headers.get("content-length") {
        let length = length
            .parse::<usize>()
            .context("llama-server Content-Length is invalid")?;
        ensure!(
            length <= MAX_HTTP_BODY_BYTES,
            "llama-server HTTP body is too large"
        );
        let mut body = vec![0; length];
        reader
            .read_exact(&mut body)
            .context("llama-server HTTP body ended early")?;
        return Ok(body);
    }
    let mut body = Vec::new();
    reader
        .take(MAX_HTTP_BODY_BYTES as u64 + 1)
        .read_to_end(&mut body)
        .context("failed to read llama-server close-delimited HTTP body")?;
    ensure!(
        body.len() <= MAX_HTTP_BODY_BYTES,
        "llama-server HTTP body is too large"
    );
    Ok(body)
}

fn read_http_chunk<R: BufRead>(reader: &mut R) -> Result<Option<Vec<u8>>> {
    let size_line = read_http_line(reader)?.context("llama-server chunk stream ended early")?;
    let size_hex = size_line.split(';').next().unwrap_or_default().trim();
    ensure!(
        !size_hex.is_empty(),
        "llama-server returned an empty HTTP chunk size"
    );
    let size = usize::from_str_radix(size_hex, 16)
        .context("llama-server returned an invalid HTTP chunk size")?;
    ensure!(
        size <= MAX_HTTP_CHUNK_BYTES,
        "llama-server HTTP chunk exceeded its safety limit"
    );
    if size == 0 {
        loop {
            let trailer =
                read_http_line(reader)?.context("llama-server chunk trailers ended early")?;
            if trailer.is_empty() {
                break;
            }
        }
        return Ok(None);
    }
    let mut chunk = vec![0; size];
    reader
        .read_exact(&mut chunk)
        .context("llama-server HTTP chunk ended early")?;
    let mut terminator = [0_u8; 2];
    reader
        .read_exact(&mut terminator)
        .context("llama-server HTTP chunk terminator ended early")?;
    ensure!(
        terminator == *b"\r\n",
        "llama-server HTTP chunk did not end in CRLF"
    );
    Ok(Some(chunk))
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn llama_requirement() -> super::super::LlamaCppRequirement {
        super::super::LlamaCppRequirement {
            binary_directory: "/tools".into(),
            source_directory: None,
            build: 10_050,
            commit: "b15ca938ad00aa6b3ee6c2edda7363fd02826b18".to_owned(),
            gpu_layers: 99,
            flash_attention: LlamaFlashAttention::Auto,
            cache_type_k: "f16".to_owned(),
            cache_type_v: "f16".to_owned(),
            mmap: true,
        }
    }

    fn exact_cache_request(
        slot_id: usize,
        prompt_tokens: usize,
        cache_n: usize,
        prompt_n: usize,
    ) -> LlamaPrefixDemoRequestEvidence {
        LlamaPrefixDemoRequestEvidence {
            label: format!("request_{slot_id}"),
            request: Some(slot_id),
            slot_id,
            prompt_tokens,
            prompt_token_ids_sha256: format!("prompt-{slot_id}"),
            content: format!("completion-{slot_id}"),
            streamed_content: format!("completion-{slot_id}"),
            streamed_content_matches_replay: true,
            content_source: "test replay".to_owned(),
            completion_tokens: 32,
            completion_token_ids: vec![slot_id as u32; 32],
            completion_token_ids_sha256: format!("completion-{slot_id}"),
            completion_token_source: "test replay".to_owned(),
            external_time_to_first_token_ms: 1.0,
            external_latency_ms: 2.0,
            terminal: LlamaPrefixDemoTerminalEvidence {
                server_reported_cache_n: cache_n,
                server_reported_prompt_n: prompt_n,
                server_reported_prompt_ms: 1.0,
                server_reported_prompt_tokens_per_second: prompt_n as f64 * 1_000.0,
                server_reported_predicted_n: 32,
                server_reported_predicted_ms: 1.0,
                server_reported_predicted_tokens_per_second: 32_000.0,
            },
        }
    }

    #[test]
    fn command_pins_cache_on_eight_slot_continuous_batching() -> Result<()> {
        let arguments = llama_server_arguments(
            Path::new("/models/SmolLM3-Q4_K_M.gguf"),
            &llama_requirement(),
            133_120,
            1_024,
            128,
            10,
            18_084,
        )?;
        let joined = arguments.join(" ");
        assert!(joined.contains("--parallel 8 --ctx-size 133120"));
        assert!(joined.contains("--batch-size 1024 --ubatch-size 128"));
        assert!(joined.contains("--no-kv-unified --cont-batching"));
        assert!(joined.contains("--cache-prompt --cache-reuse 0"));
        assert!(joined.contains("--cache-ram 0 --no-cache-idle-slots"));
        assert!(joined.contains("--slot-prompt-similarity 0"));
        assert!(!joined.contains("--no-cache-prompt"));
        assert!(joined.contains("--n-gpu-layers 99 --flash-attn auto"));
        Ok(())
    }

    #[test]
    fn context_is_total_for_all_slots_and_grows_past_manifest_capacity() -> Result<()> {
        let geometry = context_geometry(16_420, 32, 65_536, 1_024, 128)?;
        assert_eq!(geometry.minimum_tokens_per_slot, 16_708);
        assert_eq!(geometry.minimum_total_context_tokens, 133_664);
        assert_eq!(geometry.configured_total_context_tokens, 135_168);
        assert_eq!(geometry.configured_tokens_per_slot, 16_896);
        assert_eq!(geometry.cache_reuse_chunk_tokens, 0);
        assert_eq!(geometry.logical_batch_tokens, 1_024);
        assert_eq!(geometry.physical_ubatch_tokens, 128);
        assert!(
            geometry.configured_total_context_tokens
                >= PREFIX_DEMO_CONCURRENCY * (16_420 + 32 + PREFIX_DEMO_CONTEXT_HEADROOM_TOKENS)
        );

        let manifest_larger = context_geometry(128, 32, 65_539, 1_024, 128)?;
        assert_eq!(manifest_larger.configured_total_context_tokens, 67_584);
        assert_eq!(manifest_larger.configured_tokens_per_slot, 8_448);
        assert_eq!(manifest_larger.configured_tokens_per_slot % 256, 0);
        Ok(())
    }

    #[test]
    fn context_and_prefix_validation_fail_closed() {
        assert!(context_geometry(usize::MAX, 1, 1, 1_024, 128).is_err());
        assert!(context_geometry(1, usize::MAX, 1, 1_024, 128).is_err());
        assert!(context_geometry(1, 1, 1, 127, 128).is_err());

        let common = vec![7_u32; 127];
        let seed = LlamaPrefixDemoPrompt {
            rendered_prompt: "seed",
            token_ids: &common,
        };
        let request_tokens = std::array::from_fn::<_, PREFIX_DEMO_CONCURRENCY, _>(|index| {
            let mut tokens = common.clone();
            tokens.push(index as u32);
            tokens
        });
        let requests = std::array::from_fn(|index| LlamaPrefixDemoPrompt {
            rendered_prompt: "request",
            token_ids: &request_tokens[index],
        });
        assert!(validate_prompt_set(seed, &requests, 0).is_err());
        assert!(validate_prompt_set(seed, &requests, 126).is_err());
        assert!(validate_prompt_set(seed, &requests, 127).is_ok());
        let mut wrong_tokens = request_tokens.clone();
        wrong_tokens[7][0] = 8;
        let wrong = std::array::from_fn(|index| LlamaPrefixDemoPrompt {
            rendered_prompt: "request",
            token_ids: &wrong_tokens[index],
        });
        assert!(validate_prompt_set(seed, &wrong, 127).is_err());
    }

    #[test]
    fn fair_comparison_requires_at_least_twenty_retained_waves() {
        assert!(validate_measured_wave_count(0).is_err());
        assert!(validate_measured_wave_count(LLAMA_PREFIX_DEMO_MIN_MEASURED_WAVES - 1).is_err());
        assert!(validate_measured_wave_count(LLAMA_PREFIX_DEMO_MIN_MEASURED_WAVES).is_ok());
        assert!(validate_measured_wave_count(37).is_ok());
    }

    #[test]
    fn completion_bodies_pin_the_requested_slot() -> Result<()> {
        let prompt = LlamaPrefixDemoPrompt {
            rendered_prompt: "prompt",
            token_ids: &[1, 2, 3],
        };
        let sampling = ExpectedSampling {
            seed: 42,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
        };
        let spec = CompletionSpec::new("request_6".to_owned(), prompt, 32, 6, sampling)?;
        for (body, stream) in [
            (spec.body.as_slice(), true),
            (spec.nonstream_body.as_slice(), false),
        ] {
            let body: serde_json::Value = serde_json::from_slice(body)?;
            assert_eq!(body["id_slot"], 6);
            assert_eq!(body["n_predict"], 32);
            assert_eq!(body["stream"], stream);
            assert_eq!(body["cache_prompt"], true);
        }
        assert!(
            CompletionSpec::new(
                "bad".to_owned(),
                prompt,
                1,
                PREFIX_DEMO_CONCURRENCY,
                sampling,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn exact_cache_contract_rejects_one_bad_request() -> Result<()> {
        let mut requests = (0..PREFIX_DEMO_CONCURRENCY)
            .map(|slot_id| exact_cache_request(slot_id, 16_400 + slot_id, 16_384, 16 + slot_id))
            .collect::<Vec<_>>();
        let exact = cache_observation(&requests, 16_384)?;
        assert_eq!(
            exact.requests_satisfying_exact_contract,
            PREFIX_DEMO_CONCURRENCY
        );
        assert_eq!(
            exact.server_reported_cache_n_by_request,
            vec![16_384; PREFIX_DEMO_CONCURRENCY]
        );

        requests[3].terminal.server_reported_cache_n = 16_383;
        requests[3].terminal.server_reported_prompt_n += 1;
        assert!(cache_observation(&requests, 16_384).is_err());

        requests[3].terminal.server_reported_cache_n = 16_384;
        assert!(cache_observation(&requests, 16_384).is_err());
        Ok(())
    }

    #[test]
    fn chunked_http_and_split_sse_are_parsed_fail_closed() -> Result<()> {
        let wire = concat!(
            "8;kind=a\r\n",
            "data: {\"\r\n",
            "7\r\n",
            "x\":1}\n\n\r\n",
            "0\r\n",
            "X-End: yes\r\n",
            "\r\n"
        );
        let mut reader = Cursor::new(wire.as_bytes());
        let first = read_http_chunk(&mut reader)?.context("first chunk missing")?;
        let second = read_http_chunk(&mut reader)?.context("second chunk missing")?;
        assert!(read_http_chunk(&mut reader)?.is_none());
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(&first)?.is_empty());
        assert_eq!(decoder.push(&second)?, vec!["{\"x\":1}"]);

        let oversized = format!("{}\n", "x".repeat(MAX_HTTP_LINE_BYTES + 1));
        assert!(read_http_line(&mut Cursor::new(oversized)).is_err());
        Ok(())
    }

    #[test]
    fn terminal_counts_accept_cache_but_require_exact_prompt_accounting() -> Result<()> {
        let prompt = LlamaPrefixDemoPrompt {
            rendered_prompt: "prompt",
            token_ids: &[1, 2, 3, 4, 5],
        };
        let spec = CompletionSpec::new(
            "request_0".to_owned(),
            prompt,
            7,
            0,
            ExpectedSampling {
                seed: 42,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repeat_penalty: 1.0,
            },
        )?;
        let valid = WireTimings {
            cache_n: 4,
            prompt_n: 1,
            prompt_ms: 10.0,
            prompt_per_second: 100.0,
            predicted_n: 7,
            predicted_ms: 70.0,
            predicted_per_second: 100.0,
        };
        valid.validate(&spec)?;
        let mut invalid = valid.clone();
        invalid.cache_n = 5;
        assert!(invalid.validate(&spec).is_err());
        invalid = valid;
        invalid.predicted_n = 6;
        assert!(invalid.validate(&spec).is_err());
        Ok(())
    }

    #[test]
    fn run_and_cleanup_failures_are_reported_together() {
        let error = finish_run_and_cleanup::<()>(
            Err(anyhow::anyhow!("request wave broke")),
            Err(anyhow::anyhow!("child would not reap")),
            Err(anyhow::anyhow!("port remained open")),
        )
        .expect_err("all three failures must reject the demo");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("run failed: request wave broke"));
        assert!(rendered.contains("child stop/reap failed: child would not reap"));
        assert!(rendered.contains("port-close verification failed: port remained open"));
    }

    #[test]
    fn replay_retains_stream_content_without_treating_it_as_token_truth() -> Result<()> {
        let now = Instant::now();
        let mut measured = ObservedCompletion {
            request: 1,
            label: "request_1".to_owned(),
            prompt_tokens: 16_400,
            prompt_token_ids_sha256: "prompt".to_owned(),
            submitted_at: now,
            completed_at: now,
            time_to_first_token: Duration::from_millis(1),
            latency: Duration::from_millis(2),
            content: "streamed replacement character".to_owned(),
            streamed_content: None,
            token_ids: Vec::new(),
            timings: WireTimings {
                cache_n: 16_384,
                prompt_n: 16,
                prompt_ms: 1.0,
                prompt_per_second: 16_000.0,
                predicted_n: 2,
                predicted_ms: 1.0,
                predicted_per_second: 2_000.0,
            },
        };
        let replay = ObservedTokenCompletion {
            request: 1,
            label: "request_1".to_owned(),
            content: "replay text".to_owned(),
            token_ids: vec![7, 8],
        };
        attach_replay(&mut measured, &replay)?;
        assert_eq!(measured.content, "replay text");
        assert_eq!(
            measured.streamed_content.as_deref(),
            Some("streamed replacement character")
        );
        assert_eq!(measured.token_ids, vec![7, 8]);
        assert!(attach_replay(&mut measured, &replay).is_err());
        Ok(())
    }

    #[test]
    fn wave_aggregation_uses_earliest_actual_submission_to_last_completion() -> Result<()> {
        let earliest_submitted_at = Instant::now();
        let requests = (0..PREFIX_DEMO_CONCURRENCY)
            .map(|request| ObservedCompletion {
                request,
                label: format!("request_{request}"),
                prompt_tokens: 16_400 + request,
                prompt_token_ids_sha256: format!("prompt-{request}"),
                submitted_at: earliest_submitted_at
                    + Duration::from_millis(if request == 3 { 0 } else { request as u64 + 1 }),
                completed_at: earliest_submitted_at + Duration::from_millis(100 + request as u64),
                time_to_first_token: Duration::from_millis(10 + request as u64),
                latency: Duration::from_millis(90 + request as u64),
                content: format!("completion-{request}"),
                streamed_content: Some(format!("completion-{request}")),
                token_ids: vec![request as u32; 32],
                timings: WireTimings {
                    cache_n: 16_384,
                    prompt_n: 16 + request,
                    prompt_ms: 1.0,
                    prompt_per_second: (16 + request) as f64 * 1_000.0,
                    predicted_n: 32,
                    predicted_ms: 32.0,
                    predicted_per_second: 1_000.0,
                },
            })
            .collect();
        let evidence = aggregate_wave(
            3,
            ObservedWave {
                completed_at: earliest_submitted_at + Duration::from_millis(107),
                requests,
            },
            16_384,
        )?;
        assert_eq!(evidence.wave_index, 3);
        assert_eq!(evidence.request_count, 8);
        assert_eq!(evidence.total_completion_tokens, 256);
        assert!((evidence.external_wall_time_ms - 107.0).abs() < f64::EPSILON);
        assert!((evidence.aggregate_output_tokens_per_second - (256.0 / 0.107)).abs() < 1.0e-9);
        assert_eq!(evidence.external_time_to_first_token_ms.sample_count, 8);
        assert!((evidence.external_time_to_first_token_ms.min - 10.0).abs() < f64::EPSILON);
        assert!((evidence.external_time_to_first_token_ms.mean - 13.5).abs() < f64::EPSILON);
        assert!((evidence.external_time_to_first_token_ms.max - 17.0).abs() < f64::EPSILON);
        Ok(())
    }
}
