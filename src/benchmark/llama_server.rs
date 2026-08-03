//! Reproducible `llama-server` HTTP serving evidence.
//!
//! The runner launches the already verified binary with an explicit execution
//! policy, verifies the manifest prompt through `/tokenize`, and measures the
//! streaming `/completion` boundary. External TTFT/latency and the server's
//! own prompt/predicted timings are retained separately so neither is passed
//! off as the other.

use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, Read, Write},
    net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream},
    process::{Child, Command, Stdio},
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use super::{
    BenchmarkCommand, DistributionSummary, ImplementationKind, LlamaFlashAttention,
    MEMORY_SAMPLING_PERIOD_MS, MemoryResult, PrefixCacheMode, ProcessMemoryObservation,
    ProcessMemorySampler, REQUIRED_CONCURRENCY, RequestSample, ServingResult, VerifiedScoreboard,
    aggregate_throughput, process_memory, summarize, token_ids_sha256,
};

const LOOPBACK: Ipv4Addr = Ipv4Addr::LOCALHOST;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_mins(5);
const HEALTH_TIMEOUT: Duration = Duration::from_mins(2);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(50);
const MAX_HTTP_LINE_BYTES: usize = 16 * 1024;
const MAX_HTTP_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const MAX_HTTP_BODY_BYTES: usize = 32 * 1024 * 1024;

/// Raw server-side timing evidence for one externally observed request.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaServerRequestSample {
    pub wave: usize,
    pub request: usize,
    pub token_ids: Vec<u32>,
    pub cache_tokens: usize,
    pub prompt_tokens: usize,
    pub prompt_ms: f64,
    pub prompt_tokens_per_second: f64,
    pub predicted_tokens: usize,
    pub predicted_ms: f64,
    pub predicted_tokens_per_second: f64,
}

/// External serving evidence and internal timings for one concurrency width.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaServerWidthEvidence {
    pub serving: ServingResult,
    pub server_prompt_ms: DistributionSummary,
    pub server_prompt_tokens_per_second: DistributionSummary,
    pub server_predicted_ms: DistributionSummary,
    pub server_predicted_tokens_per_second: DistributionSummary,
    pub server_samples: Vec<LlamaServerRequestSample>,
}

/// Complete output of one pinned `llama-server` benchmark process.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaServerExecution {
    pub implementation: ImplementationKind,
    pub measurement_boundary: String,
    pub command: Vec<String>,
    pub base_url: String,
    pub health_wait_ms: f64,
    pub loaded_memory: ProcessMemoryObservation,
    pub tokenizer_token_ids: Vec<u32>,
    pub parity_token_ids: Vec<u32>,
    pub widths: Vec<LlamaServerWidthEvidence>,
    pub memory_results: Vec<MemoryResult>,
}

/// Build the exact long-lived server command used for all 1/4/8 measurements.
pub fn build_llama_server_command(
    verified: &VerifiedScoreboard,
    port: u16,
) -> Result<BenchmarkCommand> {
    verified.manifest.validate()?;
    ensure!(port > 0, "llama-server benchmark port must be non-zero");
    ensure_verified_prompt(verified)?;
    ensure!(
        verified.manifest.workload.prefix_cache == PrefixCacheMode::Disabled,
        "llama-server scoreboard v1 requires prefix_cache=disabled"
    );
    let threads = verified
        .host
        .performance_cores
        .or(verified.host.total_cores)
        .context("verified host has no CPU core count for pinned llama-server threads")?;
    ensure!(threads > 0, "llama-server thread count must be positive");
    let maximum_parallel = *REQUIRED_CONCURRENCY
        .last()
        .context("required concurrency list is empty")?;
    let per_slot_context = verified.manifest.runtime.kv_cache_capacity_tokens / maximum_parallel;
    let required_context = verified
        .prompt_token_ids
        .len()
        .checked_add(verified.manifest.workload.decode_steps)
        .context("llama-server request context requirement overflow")?;
    ensure!(
        required_context <= per_slot_context,
        "llama-server has {per_slot_context} context tokens per slot but the workload requires {required_context}"
    );

    Ok(BenchmarkCommand {
        program: verified.llama_server.canonical_path.clone(),
        arguments: llama_server_arguments(
            &verified.model.file.canonical_path,
            &verified.manifest.llama_cpp,
            verified.manifest.runtime.kv_cache_capacity_tokens,
            verified.manifest.runtime.prefill_token_quantum,
            threads,
            port,
        )?,
    })
}

/// Launch one verified server, measure complete waves at 1/4/8, then stop it.
pub fn run_llama_server_serving(
    verified: &VerifiedScoreboard,
    port: u16,
) -> Result<LlamaServerExecution> {
    let command = build_llama_server_command(verified, port)?;
    let report_command = command.report_argv()?;
    let address = SocketAddr::V4(SocketAddrV4::new(LOOPBACK, port));
    let mut server = LlamaServerProcess::spawn(&command)?;
    let health_wait = server.wait_until_ready(address)?;
    let tokenizer_token_ids = tokenize_prompt(address, verified)?;
    ensure!(
        tokenizer_token_ids == verified.prompt_token_ids,
        "llama-server tokenizer IDs differ from the manifest-verified MacLLM tokenizer IDs"
    );
    let server_pid = server.pid()?;
    let loaded_memory = process_memory(server_pid)?;

    let spec = Arc::new(LlamaServerRunSpec::from_verified(verified)?);
    let mut widths = Vec::with_capacity(REQUIRED_CONCURRENCY.len());
    let mut memory_results = Vec::with_capacity(REQUIRED_CONCURRENCY.len());
    let mut parity_token_ids = None;
    for concurrency in REQUIRED_CONCURRENCY {
        server.ensure_running()?;
        let (width, width_parity, memory) = run_width(
            address,
            Arc::clone(&spec),
            concurrency,
            server_pid,
            loaded_memory.physical_footprint_bytes,
        )
        .with_context(|| format!("llama-server concurrency {concurrency} benchmark failed"))?;
        if let Some(expected) = &parity_token_ids {
            ensure!(
                width_parity == *expected,
                "greedy llama-server token IDs changed at concurrency {concurrency}"
            );
        } else {
            parity_token_ids = Some(width_parity);
        }
        widths.push(width);
        memory_results.push(memory);
    }
    server.ensure_running()?;
    server.stop()?;

    Ok(LlamaServerExecution {
        implementation: ImplementationKind::LlamaCpp,
        measurement_boundary: concat!(
            "one long-lived pinned llama-server process with eight slots and prompt caching disabled; ",
            "external TTFT starts immediately before the loopback TCP connection and ends when the ",
            "first streamed token event is parsed; external latency ends at the terminal SSE event; ",
            "server prompt/predicted timings are retained verbatim from that terminal event"
        )
        .to_owned(),
        command: report_command,
        base_url: format!("http://{address}"),
        health_wait_ms: millis(health_wait),
        loaded_memory,
        tokenizer_token_ids,
        parity_token_ids: parity_token_ids.context("llama-server retained no parity tokens")?,
        widths,
        memory_results,
    })
}

fn llama_server_arguments(
    model: &std::path::Path,
    llama: &super::LlamaCppRequirement,
    context_tokens: usize,
    batch_tokens: usize,
    threads: usize,
    port: u16,
) -> Result<Vec<String>> {
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
        REQUIRED_CONCURRENCY
            .last()
            .context("required concurrency list is empty")?
            .to_string(),
        "--ctx-size".to_owned(),
        context_tokens.to_string(),
        "--batch-size".to_owned(),
        batch_tokens.to_string(),
        "--ubatch-size".to_owned(),
        batch_tokens.to_string(),
        "--threads".to_owned(),
        threads.to_string(),
        "--threads-batch".to_owned(),
        threads.to_string(),
        "--threads-http".to_owned(),
        REQUIRED_CONCURRENCY
            .last()
            .context("required concurrency list is empty")?
            .to_string(),
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
        "--cont-batching".to_owned(),
        "--no-cache-prompt".to_owned(),
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

    fn pid(&self) -> Result<u32> {
        Ok(self
            .child
            .as_ref()
            .context("llama-server child is absent")?
            .id())
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
        let mut child = self.child.take().context("llama-server child is absent")?;
        if child
            .try_wait()
            .context("failed to poll llama-server before shutdown")?
            .is_none()
        {
            child.kill().context("failed to stop llama-server")?;
            let _status = child.wait().context("failed to reap llama-server")?;
        }
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

fn tokenize_prompt(address: SocketAddr, verified: &VerifiedScoreboard) -> Result<Vec<u32>> {
    let body = serde_json::to_vec(&serde_json::json!({
        "content": verified.prompt_text,
        "add_special": verified.manifest.workload.prompt.add_special_tokens,
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
        "llama-server /tokenize returned HTTP {}: {}",
        response.status,
        String::from_utf8_lossy(&response.body)
    );
    let response: TokenizeResponse =
        serde_json::from_slice(&response.body).context("invalid llama-server /tokenize JSON")?;
    Ok(response.tokens)
}

#[derive(Debug, Clone)]
struct LlamaServerRunSpec {
    prompt: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    warmup_waves: usize,
    measured_requests: usize,
    completion_body: Arc<Vec<u8>>,
    sampling: ExpectedSampling,
}

impl LlamaServerRunSpec {
    fn from_verified(verified: &VerifiedScoreboard) -> Result<Self> {
        ensure_verified_prompt(verified)?;
        let workload = &verified.manifest.workload;
        let sampling = ExpectedSampling {
            seed: workload.sampling.seed,
            temperature: workload.sampling.temperature,
            top_k: workload.sampling.top_k,
            top_p: workload.sampling.top_p,
            repeat_penalty: f64::from(workload.sampling.repetition_penalty),
        };
        let request = CompletionRequest {
            prompt: &verified.prompt_text,
            n_predict: workload.decode_steps,
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
            ignore_eos: workload.sampling.ignore_stop_tokens,
            stream: true,
            return_tokens: true,
            cache_prompt: false,
            n_keep: 0,
            n_probs: 0,
            timings_per_token: false,
            post_sampling_probs: false,
            backend_sampling: false,
        };
        Ok(Self {
            prompt: verified.prompt_text.clone(),
            prompt_tokens: verified.prompt_token_ids.len(),
            completion_tokens: workload.decode_steps,
            warmup_waves: workload.warmup_waves,
            measured_requests: workload.samples_per_concurrency,
            completion_body: Arc::new(serde_json::to_vec(&request)?),
            sampling,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct ExpectedSampling {
    seed: u64,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repeat_penalty: f64,
}

#[derive(Serialize)]
struct CompletionRequest<'a> {
    prompt: &'a str,
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

#[derive(Debug)]
struct ObservedCompletion {
    request: usize,
    time_to_first_token: Duration,
    latency: Duration,
    completed_at: Instant,
    token_ids: Vec<u32>,
    timings: WireTimings,
}

#[derive(Debug)]
struct ObservedWave {
    released_at: Instant,
    completed_at: Instant,
    requests: Vec<ObservedCompletion>,
}

#[allow(clippy::too_many_lines)]
fn run_width(
    address: SocketAddr,
    spec: Arc<LlamaServerRunSpec>,
    concurrency: usize,
    server_pid: u32,
    loaded_steady_bytes: Option<u64>,
) -> Result<(LlamaServerWidthEvidence, Vec<u32>, MemoryResult)> {
    ensure!(concurrency > 0, "serving concurrency must be positive");
    ensure!(
        spec.measured_requests.is_multiple_of(concurrency),
        "serving request count must form complete concurrency waves"
    );
    let mut parity_token_ids = None;
    for warmup in 0..spec.warmup_waves {
        let observed = execute_wave(
            address,
            Arc::clone(&spec),
            concurrency,
            warmup * concurrency,
        )?;
        retain_parity(&mut parity_token_ids, &observed.requests, "warmup")?;
    }
    let memory_sampler = ProcessMemorySampler::start(server_pid)?;

    let wave_count = spec.measured_requests / concurrency;
    let mut external_samples = Vec::with_capacity(spec.measured_requests);
    let mut server_samples = Vec::with_capacity(spec.measured_requests);
    let mut measurement_start = None;
    let mut measurement_end = None;
    for wave in 0..wave_count {
        let first_request = wave
            .checked_mul(concurrency)
            .context("llama-server request ID overflow")?;
        let observed = execute_wave(address, Arc::clone(&spec), concurrency, first_request)?;
        retain_parity(&mut parity_token_ids, &observed.requests, "measured")?;
        measurement_start.get_or_insert(observed.released_at);
        measurement_end = Some(observed.completed_at);
        for request in observed.requests {
            external_samples.push(RequestSample {
                wave,
                request: request.request,
                prompt_tokens: spec.prompt_tokens,
                completion_tokens: request.token_ids.len(),
                time_to_first_token_ms: millis(request.time_to_first_token),
                latency_ms: millis(request.latency),
            });
            server_samples.push(LlamaServerRequestSample {
                wave,
                request: request.request,
                token_ids: request.token_ids,
                cache_tokens: request.timings.cache_n,
                prompt_tokens: request.timings.prompt_n,
                prompt_ms: request.timings.prompt_ms,
                prompt_tokens_per_second: request.timings.prompt_per_second,
                predicted_tokens: request.timings.predicted_n,
                predicted_ms: request.timings.predicted_ms,
                predicted_tokens_per_second: request.timings.predicted_per_second,
            });
        }
    }
    let measurement_start = measurement_start.context("serving run retained no start time")?;
    let measurement_end = measurement_end.context("serving run retained no completion time")?;
    let wall_time = measurement_end
        .checked_duration_since(measurement_start)
        .context("serving completion preceded first wave release")?;
    ensure!(!wall_time.is_zero(), "serving wall time must be positive");
    let total_completion_tokens = external_samples.iter().try_fold(0_u64, |total, sample| {
        total
            .checked_add(sample.completion_tokens as u64)
            .context("serving completion-token total overflow")
    })?;
    let ttft = external_samples
        .iter()
        .map(|sample| sample.time_to_first_token_ms)
        .collect::<Vec<_>>();
    let latency = external_samples
        .iter()
        .map(|sample| sample.latency_ms)
        .collect::<Vec<_>>();
    let prompt_ms = server_samples
        .iter()
        .map(|sample| sample.prompt_ms)
        .collect::<Vec<_>>();
    let prompt_tps = server_samples
        .iter()
        .map(|sample| sample.prompt_tokens_per_second)
        .collect::<Vec<_>>();
    let predicted_ms = server_samples
        .iter()
        .map(|sample| sample.predicted_ms)
        .collect::<Vec<_>>();
    let predicted_tps = server_samples
        .iter()
        .map(|sample| sample.predicted_tokens_per_second)
        .collect::<Vec<_>>();
    let serving = ServingResult {
        implementation: ImplementationKind::LlamaCpp,
        concurrency,
        request_count: external_samples.len(),
        total_completion_tokens,
        wall_time_ms: millis(wall_time),
        aggregate_output_tokens_per_second: aggregate_throughput(
            total_completion_tokens,
            wall_time,
        )?,
        time_to_first_token_ms: summarize(&ttft)?,
        latency_ms: summarize(&latency)?,
        samples: external_samples,
    };
    let memory_peak = memory_sampler.finish()?;
    let memory = MemoryResult {
        implementation: ImplementationKind::LlamaCpp,
        concurrency,
        loaded_steady_bytes,
        peak_physical_footprint_bytes: memory_peak.physical_footprint_bytes,
        peak_resident_bytes: memory_peak.resident_bytes,
        sampling_period_ms: MEMORY_SAMPLING_PERIOD_MS,
        measurement_boundary: concat!(
            "current-state proc_pid_rusage samples for the isolated pinned llama-server child; ",
            "loaded steady is sampled after health and exact tokenizer preflight; the width peak ",
            "starts after that width's warmup waves and ends after its final measured wave; ",
            "macOS lifetime maximum is intentionally excluded; widths reuse the same child"
        )
        .to_owned(),
    };
    Ok((
        LlamaServerWidthEvidence {
            serving,
            server_prompt_ms: summarize(&prompt_ms)?,
            server_prompt_tokens_per_second: summarize(&prompt_tps)?,
            server_predicted_ms: summarize(&predicted_ms)?,
            server_predicted_tokens_per_second: summarize(&predicted_tps)?,
            server_samples,
        },
        parity_token_ids.context("llama-server width retained no parity token IDs")?,
        memory,
    ))
}

fn retain_parity(
    parity: &mut Option<Vec<u32>>,
    requests: &[ObservedCompletion],
    phase: &str,
) -> Result<()> {
    for request in requests {
        if let Some(expected) = parity.as_ref() {
            ensure!(
                request.token_ids == *expected,
                "greedy llama-server token IDs changed in {phase} request {}",
                request.request
            );
        } else {
            *parity = Some(request.token_ids.clone());
        }
    }
    Ok(())
}

fn execute_wave(
    address: SocketAddr,
    spec: Arc<LlamaServerRunSpec>,
    concurrency: usize,
    first_request: usize,
) -> Result<ObservedWave> {
    let barrier = Arc::new(Barrier::new(concurrency + 1));
    let mut workers = Vec::with_capacity(concurrency);
    for offset in 0..concurrency {
        let request = first_request
            .checked_add(offset)
            .context("llama-server request ID overflow")?;
        let worker_barrier = Arc::clone(&barrier);
        let worker_spec = Arc::clone(&spec);
        workers.push(thread::spawn(move || {
            worker_barrier.wait();
            run_completion(address, &worker_spec, request)
        }));
    }
    let released_at = Instant::now();
    barrier.wait();

    let mut requests = Vec::with_capacity(concurrency);
    let mut first_error = None;
    for worker in workers {
        let result = worker
            .join()
            .map_err(|_| anyhow::anyhow!("llama-server request worker panicked"))
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
        .context("llama-server wave completed no requests")?;
    Ok(ObservedWave {
        released_at,
        completed_at,
        requests,
    })
}

fn run_completion(
    address: SocketAddr,
    spec: &LlamaServerRunSpec,
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
        Some(spec.completion_body.as_slice()),
        "text/event-stream",
    )?;
    let head = read_http_head(&mut reader)?;
    ensure!(
        head.status == 200,
        "request {request} /completion returned HTTP {}",
        head.status
    );
    ensure!(
        head.headers
            .get("content-type")
            .is_some_and(|value| value.starts_with("text/event-stream")),
        "request {request} /completion did not return text/event-stream"
    );
    ensure!(
        header_has_chunked_encoding(&head.headers),
        "request {request} /completion was not HTTP chunked"
    );

    let mut decoder = SseDecoder::default();
    let mut observer = CompletionObserver::default();
    loop {
        let Some(chunk) = read_http_chunk(&mut reader)? else {
            bail!("request {request} chunked stream ended before its terminal event");
        };
        let arrived_at = Instant::now();
        for data in decoder.push(&chunk)? {
            observer.observe(&data, arrived_at, submitted_at, spec)?;
            if observer.terminal.is_some() {
                let completed_at = Instant::now();
                let latency = completed_at
                    .checked_duration_since(submitted_at)
                    .context("llama-server completion preceded request submission")?;
                return observer.finish(request, latency, completed_at, spec);
            }
        }
    }
}

#[derive(Default)]
struct CompletionObserver {
    token_ids: Vec<u32>,
    time_to_first_token: Option<Duration>,
    terminal: Option<WireTimings>,
}

impl CompletionObserver {
    fn observe(
        &mut self,
        data: &str,
        arrived_at: Instant,
        submitted_at: Instant,
        spec: &LlamaServerRunSpec,
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
        ensure!(
            event.index == 0,
            "llama-server returned a non-zero completion index"
        );
        ensure!(
            event.tokens_evaluated == spec.prompt_tokens,
            "llama-server event evaluated {} prompt tokens; expected {}",
            event.tokens_evaluated,
            spec.prompt_tokens
        );
        if event.stop {
            ensure!(
                event.tokens.is_empty(),
                "terminal llama-server event emitted token IDs"
            );
            ensure!(
                event.tokens_predicted == self.token_ids.len(),
                "terminal llama-server predicted count differs from streamed token IDs"
            );
            ensure!(
                event.prompt.as_deref() == Some(spec.prompt.as_str()),
                "terminal llama-server prompt differs from the verified prompt"
            );
            ensure!(
                event.truncated == Some(false),
                "llama-server truncated the prompt"
            );
            ensure!(
                event.stop_type.as_deref() == Some("limit"),
                "llama-server did not stop at the fixed token limit"
            );
            let settings = event
                .generation_settings
                .context("terminal llama-server event omitted generation_settings")?;
            settings.validate(spec)?;
            let timings = event
                .timings
                .context("terminal llama-server event omitted timings")?;
            timings.validate(spec)?;
            self.terminal = Some(timings);
            return Ok(());
        }

        ensure!(
            event.timings.is_none(),
            "non-terminal llama-server event included timings"
        );
        ensure!(
            !event.tokens.is_empty(),
            "non-terminal llama-server event had no token IDs"
        );
        if self.time_to_first_token.is_none() {
            self.time_to_first_token = Some(
                arrived_at
                    .checked_duration_since(submitted_at)
                    .context("first llama-server token preceded request submission")?,
            );
        }
        self.token_ids.extend(event.tokens);
        ensure!(
            event.tokens_predicted == self.token_ids.len(),
            "llama-server streamed token count differs from tokens_predicted"
        );
        ensure!(
            self.token_ids.len() <= spec.completion_tokens,
            "llama-server exceeded the completion token budget"
        );
        Ok(())
    }

    fn finish(
        self,
        request: usize,
        latency: Duration,
        completed_at: Instant,
        spec: &LlamaServerRunSpec,
    ) -> Result<ObservedCompletion> {
        ensure!(
            self.token_ids.len() == spec.completion_tokens,
            "request {request} emitted {} completion tokens; expected {}",
            self.token_ids.len(),
            spec.completion_tokens
        );
        let time_to_first_token = self
            .time_to_first_token
            .with_context(|| format!("request {request} finished without a streamed token"))?;
        ensure!(
            latency >= time_to_first_token,
            "request {request} latency is shorter than TTFT"
        );
        Ok(ObservedCompletion {
            request,
            time_to_first_token,
            latency,
            completed_at,
            token_ids: self.token_ids,
            timings: self
                .terminal
                .context("request finished without terminal server timings")?,
        })
    }
}

#[derive(Deserialize)]
struct WireCompletionEvent {
    index: usize,
    #[serde(default)]
    tokens: Vec<u32>,
    #[serde(default)]
    stop: bool,
    tokens_predicted: usize,
    tokens_evaluated: usize,
    #[serde(default)]
    prompt: Option<String>,
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
}

impl WireGenerationSettings {
    fn validate(&self, spec: &LlamaServerRunSpec) -> Result<()> {
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
        ensure!(self.stream, "llama-server did not retain streaming mode");
        ensure!(
            self.max_tokens == spec.completion_tokens
                && self.n_predict == spec.completion_tokens
                && self.n_keep == 0,
            "llama-server changed the fixed completion boundary"
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
    fn validate(&self, spec: &LlamaServerRunSpec) -> Result<()> {
        ensure!(
            self.cache_n == 0,
            "llama-server reused cached prompt tokens"
        );
        ensure!(
            self.prompt_n == spec.prompt_tokens,
            "llama-server timing prompt count differs from the manifest"
        );
        ensure!(
            self.predicted_n == spec.completion_tokens,
            "llama-server timing predicted count differs from the token budget"
        );
        ensure!(
            positive_finite(self.prompt_ms)
                && positive_finite(self.prompt_per_second)
                && positive_finite(self.predicted_ms)
                && positive_finite(self.predicted_per_second),
            "llama-server emitted invalid internal timings"
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
        .context("failed to disable Nagle for llama-server benchmark client")?;
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

fn ensure_verified_prompt(verified: &VerifiedScoreboard) -> Result<()> {
    ensure!(
        verified.prompt_token_ids.len() == verified.manifest.workload.prompt.token_count,
        "verified scoreboard prompt token count changed before llama-server execution"
    );
    ensure!(
        token_ids_sha256(&verified.prompt_token_ids)
            == verified.manifest.workload.prompt.token_ids_sha256,
        "verified scoreboard prompt token fingerprint changed before llama-server execution"
    );
    Ok(())
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn command_arguments_pin_serving_geometry_and_disable_prompt_cache() -> Result<()> {
        let llama = super::super::LlamaCppRequirement {
            binary_directory: "/tools".into(),
            source_directory: None,
            build: 10_050,
            commit: "b15ca938ad00aa6b3ee6c2edda7363fd02826b18".to_owned(),
            gpu_layers: 99,
            flash_attention: LlamaFlashAttention::Auto,
            cache_type_k: "f16".to_owned(),
            cache_type_v: "f16".to_owned(),
            mmap: true,
        };
        let arguments = llama_server_arguments(
            std::path::Path::new("/models/SmolLM3-Q4_K_M.gguf"),
            &llama,
            65_536,
            128,
            10,
            18_081,
        )?;
        let joined = arguments.join(" ");
        assert!(joined.contains("--parallel 8"));
        assert!(joined.contains("--ctx-size 65536"));
        assert!(joined.contains("--batch-size 128 --ubatch-size 128"));
        assert!(joined.contains("--threads 10 --threads-batch 10 --threads-http 8"));
        assert!(joined.contains("--n-gpu-layers 99 --flash-attn auto"));
        assert!(joined.contains("--cache-type-k f16 --cache-type-v f16 --mmap"));
        assert!(joined.contains("--no-cache-prompt"));
        Ok(())
    }

    #[test]
    fn chunk_decoder_handles_extensions_trailers_and_split_sse_events() -> Result<()> {
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
        Ok(())
    }

    #[test]
    fn http_parser_accepts_chunked_json_and_rejects_oversized_lines() -> Result<()> {
        let wire = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n";
        let mut reader = Cursor::new(wire);
        let head = read_http_head(&mut reader)?;
        assert_eq!(head.status, 200);
        assert_eq!(read_http_body(&mut reader, &head.headers)?, b"{}");

        let oversized = format!("{}\n", "x".repeat(MAX_HTTP_LINE_BYTES + 1));
        assert!(read_http_line(&mut Cursor::new(oversized)).is_err());
        Ok(())
    }

    #[test]
    fn terminal_timing_validation_is_fail_closed() -> Result<()> {
        let spec = LlamaServerRunSpec {
            prompt: "2 + 2 =".to_owned(),
            prompt_tokens: 5,
            completion_tokens: 7,
            warmup_waves: 1,
            measured_requests: 40,
            completion_body: Arc::new(Vec::new()),
            sampling: ExpectedSampling {
                seed: 42,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repeat_penalty: 1.0,
            },
        };
        let valid = WireTimings {
            cache_n: 0,
            prompt_n: 5,
            prompt_ms: 100.0,
            prompt_per_second: 50.0,
            predicted_n: 7,
            predicted_ms: 70.0,
            predicted_per_second: 100.0,
        };
        valid.validate(&spec)?;
        let mut cached = valid.clone();
        cached.cache_n = 1;
        assert!(cached.validate(&spec).is_err());
        let mut wrong_count = valid;
        wrong_count.predicted_n = 6;
        assert!(wrong_count.validate(&spec).is_err());
        let rounded = WireTimings {
            cache_n: 0,
            prompt_n: 5,
            prompt_ms: 130.109,
            prompt_per_second: 38.429_317,
            predicted_n: 7,
            predicted_ms: 67.385,
            predicted_per_second: 103.880_686,
        };
        rounded.validate(&spec)?;
        Ok(())
    }

    #[test]
    fn completion_observer_retains_exact_tokens_and_terminal_evidence() -> Result<()> {
        let spec = LlamaServerRunSpec {
            prompt: "2 + 2 =".to_owned(),
            prompt_tokens: 5,
            completion_tokens: 7,
            warmup_waves: 1,
            measured_requests: 40,
            completion_body: Arc::new(Vec::new()),
            sampling: ExpectedSampling {
                seed: 42,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repeat_penalty: 1.0,
            },
        };
        let submitted_at = Instant::now();
        let mut observer = CompletionObserver::default();
        observer.observe(
            &serde_json::json!({
                "index": 0,
                "tokens": [220, 19, 271, 16, 13, 578, 5219],
                "stop": false,
                "tokens_predicted": 7,
                "tokens_evaluated": 5
            })
            .to_string(),
            submitted_at + Duration::from_millis(10),
            submitted_at,
            &spec,
        )?;
        observer.observe(
            &serde_json::json!({
                "index": 0,
                "tokens": [],
                "stop": true,
                "tokens_predicted": 7,
                "tokens_evaluated": 5,
                "prompt": "2 + 2 =",
                "truncated": false,
                "stop_type": "limit",
                "generation_settings": {
                    "seed": 42,
                    "temperature": 0.0,
                    "top_k": 0,
                    "top_p": 1.0,
                    "repeat_penalty": 1.0,
                    "ignore_eos": true,
                    "stream": true,
                    "max_tokens": 7,
                    "n_predict": 7,
                    "n_keep": 0
                },
                "timings": {
                    "cache_n": 0,
                    "prompt_n": 5,
                    "prompt_ms": 100.0,
                    "prompt_per_second": 50.0,
                    "predicted_n": 7,
                    "predicted_ms": 70.0,
                    "predicted_per_second": 100.0
                }
            })
            .to_string(),
            submitted_at + Duration::from_millis(20),
            submitted_at,
            &spec,
        )?;
        let completion = observer.finish(
            3,
            Duration::from_millis(20),
            submitted_at + Duration::from_millis(20),
            &spec,
        )?;
        assert_eq!(completion.request, 3);
        assert_eq!(completion.time_to_first_token, Duration::from_millis(10));
        assert_eq!(completion.token_ids, vec![220, 19, 271, 16, 13, 578, 5219]);
        Ok(())
    }
}
