//! Strict, same-transport warm-prefix comparison between `MacLLM` and llama.cpp.
//!
//! This lane intentionally does not replace the physical paged-KV demo.  The
//! physical demo proves `MacLLM`'s page/refcount behavior with an internal atomic
//! batch trace.  This module answers a different question: when both products
//! receive the same raw prompts through the same loopback HTTP/SSE client, with
//! the same model, precision, sampling, concurrency, cache endpoint, output
//! length, and retained wave count, what serving latency and throughput do
//! users observe?

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{Arc, Condvar, Mutex},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::server::RAW_COMPLETION_RESPONSE_FIELDS;

use super::{
    BenchmarkCommand, DistributionSummary, FileProvenance, FileProvenanceGuard, ImplementationKind,
    ImplementationProvenance, LLAMA_CONTEXT_ALIGNMENT_PER_SLOT, LlamaCppVersion,
    MAC_PREFIX_DEMO_CACHE_CAPACITY_TOKENS, MAC_PREFIX_DEMO_CONCURRENCY,
    MAC_PREFIX_DEMO_DECODE_TOKENS, MAC_PREFIX_DEMO_MIN_MEASURED_WAVES,
    MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET, MacllmProvenanceGuard, MaterializedPrefixDemo,
    ModelProvenance, PREFIX_DEMO_CACHE_REUSE_TOKENS, PREFIX_DEMO_CONTEXT_HEADROOM_TOKENS,
    PreparedPrefixDemo, SanitizedHost, VerifiedScoreboard, aggregate_throughput,
    build_llama_prefix_demo_command,
    prefix_demo_report::{commit_artifact_pair, normalize_artifact_pair, prepare_atomic_file},
    schema::{
        is_lower_hex, validate_file_provenance, validate_implementation_provenance,
        validate_model_provenance,
    },
    sha256_bytes, summarize, token_ids_sha256,
};

pub const FAIR_PREFIX_REPORT_SCHEMA_VERSION: u32 = 1;
const FAIR_TRANSPORT: &str = "loopback HTTP/1.1 POST /completion with chunked text/event-stream";
const FAIR_PROCESS_ORDER: [&str; 8] = [
    "llama.cpp",
    "macllm",
    "macllm",
    "llama.cpp",
    "macllm",
    "llama.cpp",
    "llama.cpp",
    "macllm",
];
const FAIR_PAIRED_PROCESS_POSITIONS: [(usize, usize); 4] = [(0, 1), (3, 2), (5, 4), (6, 7)];
const FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION: usize = 4;
const FAIR_WARMUP_WAVES_PER_LIFETIME: usize = 5;
const FAIR_STEADY_STATE_WINDOW_WAVES: usize = 3;
const FAIR_STEADY_STATE_MAX_CV: f64 = 0.10;
const FAIR_STEADY_STATE_MAX_RELATIVE_TREND: f64 = 0.10;
const FAIR_POST_SEED_STABILIZATION: Duration = Duration::from_secs(30);
const LOOPBACK: Ipv4Addr = Ipv4Addr::LOCALHOST;
const REQUEST_TIMEOUT: Duration = Duration::from_mins(5);
const HEALTH_TIMEOUT: Duration = Duration::from_mins(2);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(50);
const PORT_CLOSE_TIMEOUT: Duration = Duration::from_secs(5);
const PORT_POLL_INTERVAL: Duration = Duration::from_millis(20);
const PORT_PROBE_TIMEOUT: Duration = Duration::from_millis(100);
const MAX_HTTP_LINE_BYTES: usize = 16 * 1024;
const MAX_HTTP_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const MAX_HTTP_BODY_BYTES: usize = 32 * 1024 * 1024;
const FAIR_CHILD_PATH: &str = "/usr/bin:/bin:/usr/sbin:/sbin";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThreadStartState {
    Waiting,
    Released,
    Cancelled,
}

#[derive(Debug)]
struct ThreadStartGate {
    state: Mutex<ThreadStartState>,
    changed: Condvar,
}

impl ThreadStartGate {
    fn new() -> Self {
        Self {
            state: Mutex::new(ThreadStartState::Waiting),
            changed: Condvar::new(),
        }
    }

    fn wait(&self) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while *state == ThreadStartState::Waiting {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        ensure!(
            *state == ThreadStartState::Released,
            "fair-prefix client start was cancelled"
        );
        Ok(())
    }

    fn transition(&self, next: ThreadStartState) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *state = next;
        self.changed.notify_all();
    }

    fn release(&self) {
        self.transition(ThreadStartState::Released);
    }

    fn cancel(&self) {
        self.transition(ThreadStartState::Cancelled);
    }
}

/// Everything that must be identical before a speed ratio can be published.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixContract {
    pub comparison_scope: String,
    pub exact_shared_prefix_tokens: usize,
    pub macllm_prefix_cache_block_tokens: usize,
    pub macllm_kv_cache_slab_blocks: usize,
    pub macllm_kv_cache_capacity_tokens: usize,
    pub macllm_prefill_token_quantum: usize,
    pub macllm_scheduler_token_budget: usize,
    pub concurrency: usize,
    pub completion_tokens_per_request: usize,
    pub measured_waves_per_implementation: usize,
    pub post_seed_stabilization_ms: u64,
    pub warmup_waves_per_lifetime: usize,
    pub steady_state_window_waves: usize,
    pub steady_state_max_cv_bits: u64,
    pub steady_state_max_relative_trend_bits: u64,
    pub temperature_bits: u64,
    pub top_k: usize,
    pub top_p_bits: u64,
    pub repetition_penalty_bits: u64,
    pub seed: u64,
    pub ignore_eos: bool,
    pub cache_prompt: bool,
    pub stream: bool,
    pub return_tokens: bool,
    pub macllm_kv_dtype: String,
    pub llama_cpp_k_cache_dtype: String,
    pub llama_cpp_v_cache_dtype: String,
    pub llama_cpp_context_tokens: usize,
    pub llama_cpp_batch_tokens: usize,
    pub llama_cpp_ubatch_tokens: usize,
    pub llama_cpp_threads: usize,
    pub llama_cpp_gpu_layers: i32,
    pub llama_cpp_flash_attention: String,
    pub llama_cpp_mmap: bool,
    pub loopback_port: u16,
    pub transport: String,
    pub child_environment: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixTerminalEvidence {
    pub cached_prompt_tokens: usize,
    pub evaluated_prompt_tokens: usize,
    pub generated_tokens: usize,
    pub server_prompt_ms: f64,
    pub server_predicted_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixRequestEvidence {
    pub request_index: usize,
    pub slot_id: usize,
    pub prompt_tokens: usize,
    pub prompt_token_ids_sha256: String,
    pub streamed_content: String,
    pub streamed_token_ids: Vec<u32>,
    pub submitted_offset_ms: f64,
    pub first_token_offset_ms: f64,
    pub completed_offset_ms: f64,
    pub external_time_to_first_token_ms: f64,
    pub external_latency_ms: f64,
    pub terminal: FairPrefixTerminalEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixWaveEvidence {
    pub wave_index: usize,
    pub process_sequence_index: usize,
    pub request_count: usize,
    pub total_completion_tokens: u64,
    pub external_wall_time_ms: f64,
    pub aggregate_output_tokens_per_second: f64,
    pub exact_cache_hit_requests: usize,
    pub requests: Vec<FairPrefixRequestEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixDistributionEvidence {
    pub aggregate_output_tokens_per_second_samples: Vec<f64>,
    pub aggregate_output_tokens_per_second: DistributionSummary,
    pub wave_max_time_to_first_token_ms_samples: Vec<f64>,
    pub wave_max_time_to_first_token_ms: DistributionSummary,
    pub wave_max_latency_ms_samples: Vec<f64>,
    pub wave_max_latency_ms: DistributionSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixReplayEvidence {
    pub prompt_token_ids_sha256: Vec<String>,
    pub content: Vec<String>,
    pub content_sha256: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixRuntimeAttestation {
    pub source: String,
    pub process_command: Option<Vec<String>>,
    pub served_model: Option<String>,
    pub device: String,
    pub kv_cache_k_dtype: String,
    pub kv_cache_v_dtype: String,
    pub model_layers_offloaded: Option<usize>,
    pub model_layers_total: Option<usize>,
    pub flash_attention_enabled: Option<bool>,
    pub parallel_slots: Option<usize>,
    pub context_tokens: Option<usize>,
    pub batch_tokens: Option<usize>,
    pub ubatch_tokens: Option<usize>,
    pub evidence_lines: Vec<String>,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixImplementationRun {
    pub implementation: String,
    pub process_command: Vec<String>,
    pub isolated_os_process: bool,
    pub base_url: String,
    pub process_sequence_indices: Vec<usize>,
    pub health_wait_ms_samples: Vec<f64>,
    pub post_seed_stabilization_ms_samples: Vec<f64>,
    pub warmup_waves_per_lifetime: usize,
    pub warmup_aggregate_output_tokens_per_second_samples: Vec<f64>,
    pub runtime_attestations: Vec<FairPrefixRuntimeAttestation>,
    pub measurement_boundary: String,
    pub tokenizer_exact: bool,
    pub cache_prompt_publication: bool,
    pub seed_requests: usize,
    pub restore_requests: usize,
    pub conditioning_prompt_token_ids_sha256: Vec<String>,
    pub conditioning_lcp_tokens: Vec<usize>,
    pub waves: Vec<FairPrefixWaveEvidence>,
    pub distributions: FairPrefixDistributionEvidence,
    pub lifetime_replays: Vec<FairPrefixReplayEvidence>,
    pub replay: FairPrefixReplayEvidence,
    pub port_closed_after_shutdown: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixEligibility {
    pub eligible: bool,
    pub model_identity_parity: bool,
    pub precision_parity: bool,
    pub transport_parity: bool,
    pub process_isolation_parity: bool,
    pub runtime_attestation_gate: bool,
    pub counterbalanced_execution_order: bool,
    pub stabilization_parity: bool,
    pub warmup_parity: bool,
    pub steady_state_gate: bool,
    pub conditioning_parity: bool,
    pub cache_policy_parity: bool,
    pub input_token_parity: bool,
    pub output_content_parity: bool,
    pub output_token_id_parity: bool,
    pub sample_count_parity: bool,
    pub cache_state_parity: bool,
    pub fixed_completion_parity: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixPairEvidence {
    pub pair_index: usize,
    pub process_order: Vec<String>,
    pub macllm_process_sequence_index: usize,
    pub llama_cpp_process_sequence_index: usize,
    pub retained_waves_per_implementation: usize,
    pub macllm_phase_output_tokens_per_second: f64,
    pub llama_cpp_phase_output_tokens_per_second: f64,
    pub macllm_to_llama_cpp_phase_throughput_ratio: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixComparison {
    pub macllm_p50_aggregate_output_tokens_per_second: f64,
    pub llama_cpp_p50_aggregate_output_tokens_per_second: f64,
    pub macllm_p50_wave_max_time_to_first_token_ms: f64,
    pub llama_cpp_p50_wave_max_time_to_first_token_ms: f64,
    pub paired_process_blocks: Vec<FairPrefixPairEvidence>,
    pub macllm_to_llama_cpp_paired_geomean_throughput_ratio: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixReport {
    pub schema_version: u32,
    pub generated_unix_ms: u128,
    pub profile: String,
    pub host: SanitizedHost,
    pub model: ModelProvenance,
    pub macllm_provenance: ImplementationProvenance,
    pub llama_cpp_server: FileProvenance,
    pub llama_cpp_version: LlamaCppVersion,
    pub llama_cpp_full_commit: String,
    pub llama_cpp_control_attestation: FairPrefixRuntimeAttestation,
    pub prepared: PreparedPrefixDemo,
    pub contract: FairPrefixContract,
    pub execution_order: Vec<String>,
    pub macllm: FairPrefixImplementationRun,
    pub llama_cpp: FairPrefixImplementationRun,
    pub eligibility: FairPrefixEligibility,
    pub comparison: FairPrefixComparison,
    pub warnings: Vec<String>,
}

impl FairPrefixReport {
    /// Recompute every derived statistic and comparison gate from raw waves.
    #[allow(clippy::too_many_lines)] // Binary provenance and every derived comparison remain one audit point.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == FAIR_PREFIX_REPORT_SCHEMA_VERSION,
            "unsupported fair-prefix report schema {}",
            self.schema_version
        );
        ensure!(
            self.generated_unix_ms > 0,
            "fair-prefix report timestamp is zero"
        );
        ensure!(
            !self.profile.trim().is_empty(),
            "fair-prefix profile cannot be empty"
        );
        ensure!(
            self.warnings == canonical_warnings(),
            "fair-prefix canonical scope and uncertainty warnings changed"
        );
        self.host.validate()?;
        validate_model_provenance(&self.model)?;
        validate_implementation_provenance(&self.macllm_provenance)?;
        ensure!(
            self.macllm_provenance.implementation == ImplementationKind::Macllm
                && self.macllm_provenance.dirty == Some(false),
            "fair-prefix publication requires clean MacLLM provenance"
        );
        ensure!(
            self.macllm.implementation == EndpointKind::Macllm.label()
                && self.llama_cpp.implementation == EndpointKind::LlamaCpp.label(),
            "fair-prefix implementation fields are not bound to their report roles"
        );
        ensure!(
            self.execution_order == FAIR_PROCESS_ORDER.map(str::to_owned),
            "fair-prefix report did not retain the required AB/BA/BA/AB execution order"
        );
        validate_file_provenance(&self.llama_cpp_server)?;
        ensure!(
            self.llama_cpp_version.build > 0
                && (7..=40).contains(&self.llama_cpp_version.commit_abbreviation.len())
                && is_lower_hex(
                    &self.llama_cpp_version.commit_abbreviation,
                    self.llama_cpp_version.commit_abbreviation.len()
                )
                && is_lower_hex(&self.llama_cpp_full_commit, 40)
                && self
                    .llama_cpp_full_commit
                    .starts_with(&self.llama_cpp_version.commit_abbreviation),
            "fair-prefix llama.cpp build/source provenance is invalid"
        );
        let macllm_executable = self
            .macllm_provenance
            .executable
            .canonical_path
            .to_str()
            .context("MacLLM executable path is not UTF-8")?;
        ensure!(
            self.macllm.process_command.first().map(String::as_str) == Some(macllm_executable),
            "fair-prefix MacLLM phases did not spawn the provenance-pinned executable"
        );
        let llama_cpp_executable = self
            .llama_cpp_server
            .canonical_path
            .to_str()
            .context("llama.cpp server executable path is not UTF-8")?;
        ensure!(
            self.llama_cpp.process_command.first().map(String::as_str)
                == Some(llama_cpp_executable),
            "fair-prefix llama.cpp phases did not spawn the provenance-pinned executable"
        );
        self.prepared.validate()?;
        ensure!(
            self.prepared.model_file == self.model.file,
            "fair-prefix prepared model differs from report provenance"
        );
        validate_contract(&self.contract, &self.prepared, &self.host)?;
        validate_launch_commands(self)?;
        validate_runtime_attestation(
            &self.llama_cpp_control_attestation,
            EndpointKind::LlamaCpp.label(),
            &self.profile,
            &self.contract,
        )?;
        validate_implementation_run(&self.macllm, &self.contract, &self.prepared, &self.profile)?;
        validate_implementation_run(
            &self.llama_cpp,
            &self.contract,
            &self.prepared,
            &self.profile,
        )?;
        let expected_eligibility = derive_eligibility(
            &self.contract,
            &self.model,
            &self.prepared,
            &self.profile,
            &self.execution_order,
            &self.macllm,
            &self.llama_cpp,
            &self.llama_cpp_control_attestation,
        );
        ensure!(
            self.eligibility == expected_eligibility,
            "fair-prefix comparison eligibility differs from raw evidence"
        );
        let expected_comparison =
            derive_comparison(&self.eligibility, &self.macllm, &self.llama_cpp)?;
        ensure!(
            approximately_equal(
                self.comparison
                    .macllm_p50_aggregate_output_tokens_per_second,
                expected_comparison.macllm_p50_aggregate_output_tokens_per_second,
            ) && approximately_equal(
                self.comparison
                    .llama_cpp_p50_aggregate_output_tokens_per_second,
                expected_comparison.llama_cpp_p50_aggregate_output_tokens_per_second,
            ) && approximately_equal(
                self.comparison.macllm_p50_wave_max_time_to_first_token_ms,
                expected_comparison.macllm_p50_wave_max_time_to_first_token_ms,
            ) && approximately_equal(
                self.comparison
                    .llama_cpp_p50_wave_max_time_to_first_token_ms,
                expected_comparison.llama_cpp_p50_wave_max_time_to_first_token_ms,
            ) && optional_approximately_equal(
                self.comparison
                    .macllm_to_llama_cpp_paired_geomean_throughput_ratio,
                expected_comparison.macllm_to_llama_cpp_paired_geomean_throughput_ratio,
            ) && self.comparison.paired_process_blocks.len()
                == expected_comparison.paired_process_blocks.len()
                && self
                    .comparison
                    .paired_process_blocks
                    .iter()
                    .zip(&expected_comparison.paired_process_blocks)
                    .all(|(actual, expected)| {
                        actual.pair_index == expected.pair_index
                            && actual.process_order == expected.process_order
                            && actual.macllm_process_sequence_index
                                == expected.macllm_process_sequence_index
                            && actual.llama_cpp_process_sequence_index
                                == expected.llama_cpp_process_sequence_index
                            && actual.retained_waves_per_implementation
                                == expected.retained_waves_per_implementation
                            && approximately_equal(
                                actual.macllm_phase_output_tokens_per_second,
                                expected.macllm_phase_output_tokens_per_second,
                            )
                            && approximately_equal(
                                actual.llama_cpp_phase_output_tokens_per_second,
                                expected.llama_cpp_phase_output_tokens_per_second,
                            )
                            && optional_approximately_equal(
                                actual.macllm_to_llama_cpp_phase_throughput_ratio,
                                expected.macllm_to_llama_cpp_phase_throughput_ratio,
                            )
                    }),
            "fair-prefix comparison differs from raw evidence"
        );
        ensure!(
            self.eligibility.eligible
                == self
                    .comparison
                    .macllm_to_llama_cpp_paired_geomean_throughput_ratio
                    .is_some(),
            "throughput ratio presence must exactly match comparison eligibility"
        );
        ensure!(
            self.comparison.paired_process_blocks.iter().all(|pair| {
                pair.macllm_to_llama_cpp_phase_throughput_ratio.is_some()
                    == self.eligibility.eligible
            }),
            "paired ratio presence must exactly match comparison eligibility"
        );
        Ok(())
    }
}

/// Run the same raw-prompt HTTP workload against isolated llama.cpp and
/// `MacLLM` servers.  Model load, health checks, cache seeding, slot restoration,
/// and correctness replays are excluded from retained performance waves.
#[allow(clippy::too_many_lines)] // Keep the complete counterbalanced process order in one audit point.
pub async fn run_fair_prefix_benchmark(
    verified: &VerifiedScoreboard,
    provenance_guard: MacllmProvenanceGuard,
    prepared: PreparedPrefixDemo,
    materialized: &MaterializedPrefixDemo,
    measured_waves: usize,
    port: u16,
) -> Result<FairPrefixReport> {
    validate_inputs(verified, &prepared, materialized, measured_waves, port)?;
    let contract = build_contract(verified, &prepared, measured_waves, port)?;
    let address = SocketAddr::V4(SocketAddrV4::new(LOOPBACK, port));
    ensure_port_is_closed(address)?;
    let specs = FairPromptSet::new(materialized, &contract)?;
    let mut llama_executable_guard = FileProvenanceGuard::capture(&verified.llama_server)
        .context("failed to pin llama-server executable provenance")?;
    let mut model_file_guard = FileProvenanceGuard::capture(&verified.model.file)
        .context("failed to pin model-file provenance")?;
    let waves_per_lifetime =
        contract.measured_waves_per_implementation / FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION;

    let mut llama_command = build_llama_prefix_demo_command(
        verified,
        super::LlamaPrefixDemoPrompt {
            rendered_prompt: &materialized.cache_seed.rendered_prompt,
            token_ids: &materialized.cache_seed.token_ids,
        },
        &std::array::from_fn(|index| super::LlamaPrefixDemoPrompt {
            rendered_prompt: &materialized.tasks[index].rendered_prompt,
            token_ids: &materialized.tasks[index].token_ids,
        }),
        contract.completion_tokens_per_request,
        contract.exact_shared_prefix_tokens,
        port,
    )?;
    pin_llama_measurement_device(&mut llama_command)?;
    let llama_attestation_command = build_llama_attestation_command(&llama_command)?;
    let macllm_command = build_fair_macllm_command(verified, address, &contract)?;
    let mut llama_phases = Vec::with_capacity(FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION);
    let mut macllm_phases = Vec::with_capacity(FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION);
    for (process_sequence_index, implementation) in FAIR_PROCESS_ORDER.iter().enumerate() {
        if *implementation == EndpointKind::LlamaCpp.label() {
            let wave_index_start = llama_phases
                .len()
                .checked_mul(waves_per_lifetime)
                .context("fair-prefix llama.cpp wave index overflow")?;
            llama_phases.push(
                run_llama_endpoint(
                    verified,
                    &llama_command,
                    address,
                    &specs,
                    &contract,
                    process_sequence_index,
                    wave_index_start,
                    waves_per_lifetime,
                    &mut llama_executable_guard,
                    &mut model_file_guard,
                )
                .with_context(|| {
                    format!(
                        "fair-prefix llama.cpp process sequence {process_sequence_index} failed"
                    )
                })?,
            );
        } else {
            ensure!(
                *implementation == EndpointKind::Macllm.label(),
                "fair-prefix process order contains an unknown implementation"
            );
            let wave_index_start = macllm_phases
                .len()
                .checked_mul(waves_per_lifetime)
                .context("fair-prefix MacLLM wave index overflow")?;
            macllm_phases.push(
                run_macllm_endpoint(
                    &macllm_command,
                    address,
                    &verified.manifest.name,
                    &specs,
                    &contract,
                    process_sequence_index,
                    wave_index_start,
                    waves_per_lifetime,
                    &mut model_file_guard,
                )
                .with_context(|| {
                    format!("fair-prefix MacLLM process sequence {process_sequence_index} failed")
                })?,
            );
        }
    }
    let llama_runtime_attestation = run_llama_attestation_endpoint(
        &llama_attestation_command,
        address,
        &contract,
        &mut llama_executable_guard,
        &mut model_file_guard,
    )
    .context("separate unmeasured llama.cpp backend attestation failed")?;
    let llama_cpp = combine_implementation_runs(EndpointKind::LlamaCpp, llama_phases, &contract)?;
    let macllm = combine_implementation_runs(EndpointKind::Macllm, macllm_phases, &contract)?;
    let llama_cpp_server = llama_executable_guard
        .finish()
        .context("llama-server executable changed during fair-prefix execution")?;
    let guarded_model_file = model_file_guard
        .finish()
        .context("model file changed during fair-prefix execution")?;
    ensure!(
        guarded_model_file == verified.model.file,
        "guarded model file differs from report model provenance"
    );
    let macllm_provenance = provenance_guard
        .finish()
        .context("MacLLM executable/source provenance changed during fair-prefix execution")?;

    let execution_order = FAIR_PROCESS_ORDER.map(str::to_owned).to_vec();
    let eligibility = derive_eligibility(
        &contract,
        &verified.model,
        &prepared,
        &verified.manifest.name,
        &execution_order,
        &macllm,
        &llama_cpp,
        &llama_runtime_attestation,
    );
    let comparison = derive_comparison(&eligibility, &macllm, &llama_cpp)?;
    let report = FairPrefixReport {
        schema_version: FAIR_PREFIX_REPORT_SCHEMA_VERSION,
        generated_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock precedes Unix epoch")?
            .as_millis(),
        profile: verified.manifest.name.clone(),
        host: verified.host.clone(),
        model: verified.model.clone(),
        macllm_provenance,
        llama_cpp_server,
        llama_cpp_version: verified.llama_cpp_version.clone(),
        llama_cpp_full_commit: verified.llama_cpp_commit.clone(),
        llama_cpp_control_attestation: llama_runtime_attestation,
        prepared,
        contract,
        execution_order,
        macllm,
        llama_cpp,
        eligibility,
        comparison,
        warnings: canonical_warnings(),
    };
    report.validate()?;
    Ok(report)
}

fn canonical_warnings() -> Vec<String> {
    vec![
        concat!(
            "this is a controlled exact-slot cache lane, not a product-default or independently ",
            "best-tuned product comparison. llama.cpp uses the same 1024/128 logical/physical ",
            "batch geometry as MacLLM and disables global prompt-cache RAM and unified K/V; a ",
            "future best-tuned lane must select each implementation's configuration on a separate ",
            "calibration set, freeze it, and measure only on held-out waves"
        )
        .to_owned(),
        concat!(
            "the two implementations run in four deterministic paired blocks with AB/BA/BA/AB ",
            "order, a 30-second post-seed stabilization interval, and five excluded warm-up ",
            "waves per isolated process lifetime with a pre-registered CV/trend gate. The reported ",
            "ratio is only a four-pair point estimate: it is the geometric mean of paired phase ",
            "throughputs, where each phase divides all retained completion tokens by summed external ",
            "wall time. It is not an uncertainty-qualified faster claim or a long-term SLA"
        )
        .to_owned(),
        concat!(
            "the runner does not record or gate the power source, Low Power Mode, thermal pressure ",
            "or clock state, or background load. Counterbalancing, cooldown, and the warm-up ",
            "CV/trend gate mitigate observed order and drift effects but do not prove equivalent ",
            "host conditions; the result remains one controlled-run point estimate"
        )
        .to_owned(),
        concat!(
            "both sides receive the same eight initial seed bodies and the same eight rotated-task ",
            "conditioning bodies at identical points. Each conditioning prompt has an exact-16K ",
            "LCP with the next measured task, so both servers report cache_n=16K and perform the ",
            "same suffix work before the next wave. Initial seed compute is not symmetric: ",
            "llama.cpp cold-prefills all eight slots while MacLLM can reuse 16,368 tokens on ",
            "seven seeds. Those requests are excluded; the subsequent fixed cooldown and five ",
            "identical warm-up waves must pass the pre-registered steady-state gate"
        )
        .to_owned(),
        concat!(
            "measured llama.cpp processes run with logging disabled. Backend evidence comes from one ",
            "separate unmeasured verbosity-4 process launched after all measured phases with identical ",
            "binary, model, and non-logging argv; response_fields removes the full prompt and unrelated ",
            "terminal metadata from every llama.cpp terminal response"
        )
        .to_owned(),
        concat!(
            "cache_n is treated only as a logical exact-prefix contract. This serving lane ",
            "does not claim that llama.cpp and MacLLM use the same physical KV layout; use ",
            "the separate physical prefix demo for MacLLM page/refcount evidence"
        )
        .to_owned(),
        concat!(
            "greedy correctness uses complete non-stream response content plus the 32 generated ",
            "token IDs retained from every measured stream; token IDs must remain identical across ",
            "waves and between implementations for each task"
        )
        .to_owned(),
        concat!(
            "the Homebrew llama-server executable is pinned, but this report does not yet hash the ",
            "non-system dynamic-library closure; do not treat it as supply-chain-complete provenance"
        )
        .to_owned(),
    ]
}

/// Validate and write a readable JSON/Markdown artifact pair.
pub fn write_fair_prefix_artifacts(
    report: &FairPrefixReport,
    json_path: &Path,
    markdown_path: &Path,
) -> Result<()> {
    report.validate()?;
    let (json_path, markdown_path) = normalize_artifact_pair(json_path, markdown_path)?;
    let json = serialized_fair_prefix_json(report)?;
    let markdown = render_fair_prefix_markdown_with_json_sha(report, &sha256_bytes(&json))?;
    let json_temp = prepare_atomic_file(&json_path, &json)?;
    let markdown_temp = prepare_atomic_file(&markdown_path, markdown.as_bytes())?;
    commit_artifact_pair(json_temp, &json_path, markdown_temp, &markdown_path)
}

pub fn render_fair_prefix_markdown(report: &FairPrefixReport) -> Result<String> {
    report.validate()?;
    let json = serialized_fair_prefix_json(report)?;
    render_fair_prefix_markdown_with_json_sha(report, &sha256_bytes(&json))
}

fn serialized_fair_prefix_json(report: &FairPrefixReport) -> Result<Vec<u8>> {
    report.validate()?;
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    Ok(json)
}

#[allow(clippy::too_many_lines)] // Keep the compact human report in the same order as the JSON gates.
fn render_fair_prefix_markdown_with_json_sha(
    report: &FairPrefixReport,
    json_sha256: &str,
) -> Result<String> {
    let mut output = String::new();
    output.push_str("# MacLLM vs llama.cpp controlled warm-prefix benchmark\n\n");
    writeln!(
        output,
        "> Authoritative JSON companion SHA-256: `{json_sha256}`.\n"
    )?;
    writeln!(output, "- Profile: `{}`", report.profile)?;
    writeln!(
        output,
        "- Scope: `{}`; this is not a product-default or independently best-tuned comparison",
        report.contract.comparison_scope
    )?;
    writeln!(
        output,
        "- Workload: exact {}-token prefix, {} concurrent requests, {} output tokens/request",
        report.contract.exact_shared_prefix_tokens,
        report.contract.concurrency,
        report.contract.completion_tokens_per_request
    )?;
    writeln!(
        output,
        "- Samples: {} retained waves per implementation across {} process lifetimes",
        report.contract.measured_waves_per_implementation,
        FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
    )?;
    writeln!(
        output,
        "- Process order: `{}`",
        report.execution_order.join(" -> ")
    )?;
    writeln!(
        output,
        "- Warm-up: {} excluded waves per process lifetime; final {} must satisfy CV <= {:.0}% and relative trend <= {:.0}%",
        report.macllm.warmup_waves_per_lifetime,
        report.contract.steady_state_window_waves,
        f64::from_bits(report.contract.steady_state_max_cv_bits) * 100.0,
        f64::from_bits(report.contract.steady_state_max_relative_trend_bits) * 100.0
    )?;
    writeln!(
        output,
        "- Post-seed stabilization: {} ms per process lifetime",
        report.contract.post_seed_stabilization_ms
    )?;
    writeln!(
        output,
        "- Isolation: fresh OS process per lifetime; {} excluded seed and {} excluded restore requests per implementation",
        report.macllm.seed_requests, report.macllm.restore_requests
    )?;
    writeln!(output, "- Transport: {}", report.contract.transport)?;
    let child_environment = report
        .contract
        .child_environment
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        output,
        "- Child environment: inherited environment cleared; `{child_environment}`"
    )?;
    writeln!(
        output,
        "- KV precision: MacLLM {}, llama.cpp K={} V={}\n",
        report.contract.macllm_kv_dtype,
        report.contract.llama_cpp_k_cache_dtype,
        report.contract.llama_cpp_v_cache_dtype
    )?;
    output.push_str("| Runtime | aggregate TPS p50 | aggregate TPS p95 | wave-max TTFT p50 | wave-max latency p50 |\n");
    output.push_str("|---|---:|---:|---:|---:|\n");
    for run in [&report.macllm, &report.llama_cpp] {
        writeln!(
            output,
            "| {} | {:.3} | {:.3} | {:.3} ms | {:.3} ms |",
            run.implementation,
            run.distributions.aggregate_output_tokens_per_second.p50,
            run.distributions.aggregate_output_tokens_per_second.p95,
            run.distributions.wave_max_time_to_first_token_ms.p50,
            run.distributions.wave_max_latency_ms.p50,
        )?;
    }
    output.push('\n');
    output.push_str("| Pair | order | MacLLM phase TPS | llama.cpp phase TPS | paired ratio |\n");
    output.push_str("|---:|---|---:|---:|---:|\n");
    for pair in &report.comparison.paired_process_blocks {
        let ratio = pair
            .macllm_to_llama_cpp_phase_throughput_ratio
            .map_or_else(|| "withheld".to_owned(), |ratio| format!("{ratio:.3}x"));
        writeln!(
            output,
            "| {} | {} | {:.3} | {:.3} | {} |",
            pair.pair_index + 1,
            pair.process_order.join(" -> "),
            pair.macllm_phase_output_tokens_per_second,
            pair.llama_cpp_phase_output_tokens_per_second,
            ratio
        )?;
    }
    output.push('\n');
    if let Some(ratio) = report
        .comparison
        .macllm_to_llama_cpp_paired_geomean_throughput_ratio
    {
        writeln!(
            output,
            "Comparison eligible: **the four-pair point estimate is {ratio:.3}x MacLLM/llama.cpp** \
             by the geometric mean of paired phase-throughput ratios for this exact contract. \
             No uncertainty-qualified faster claim is made.\n"
        )?;
    } else {
        output
            .push_str("**Not comparison-eligible. No speed ratio is published.**\n\nReasons:\n\n");
        for reason in &report.eligibility.reasons {
            writeln!(output, "- {reason}")?;
        }
        output.push('\n');
    }
    output.push_str("## Boundaries\n\n");
    for warning in &report.warnings {
        writeln!(output, "- {warning}")?;
    }
    Ok(output)
}

#[allow(clippy::too_many_lines)] // The dedicated manifest's complete executable contract is one fail-closed gate.
fn validate_inputs(
    verified: &VerifiedScoreboard,
    prepared: &PreparedPrefixDemo,
    materialized: &MaterializedPrefixDemo,
    measured_waves: usize,
    port: u16,
) -> Result<()> {
    verified.manifest.validate()?;
    prepared.validate()?;
    let fair_manifest = verified
        .manifest
        .fair_prefix
        .as_ref()
        .context("fair-prefix benchmark requires the dedicated manifest declaration")?;
    ensure!(port > 0, "fair-prefix benchmark port must be non-zero");
    ensure!(
        measured_waves >= MAC_PREFIX_DEMO_MIN_MEASURED_WAVES,
        "fair-prefix benchmark requires at least {MAC_PREFIX_DEMO_MIN_MEASURED_WAVES} measured waves per implementation"
    );
    ensure!(
        measured_waves.is_multiple_of(FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION),
        "fair-prefix measured waves must split evenly across four counterbalanced process lifetimes"
    );
    ensure!(
        prepared.requested_shared_tokens == 16_384,
        "fair-prefix benchmark requires an exact 16,384-token endpoint"
    );
    ensure!(
        fair_manifest.prompt_source == "prepared_repository_bundle"
            && fair_manifest.exact_shared_prefix_tokens == prepared.requested_shared_tokens
            && fair_manifest.concurrency == MAC_PREFIX_DEMO_CONCURRENCY
            && fair_manifest.completion_tokens_per_request == MAC_PREFIX_DEMO_DECODE_TOKENS
            && fair_manifest.measured_waves_per_implementation == measured_waves
            && fair_manifest.process_lifetimes_per_implementation
                == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
            && fair_manifest.process_order == FAIR_PROCESS_ORDER
            && fair_manifest.warmup_waves_per_lifetime == FAIR_WARMUP_WAVES_PER_LIFETIME
            && fair_manifest.post_seed_stabilization_ms
                == u64::try_from(FAIR_POST_SEED_STABILIZATION.as_millis())
                    .context("fair-prefix stabilization duration does not fit u64")?
            && fair_manifest.cache_prompt,
        "dedicated fair-prefix manifest declaration differs from the executable contract"
    );
    ensure!(
        materialized.tasks.len() == MAC_PREFIX_DEMO_CONCURRENCY
            && prepared.tasks.len() == MAC_PREFIX_DEMO_CONCURRENCY,
        "fair-prefix benchmark requires exactly eight tasks"
    );
    ensure!(
        materialized.seed.token_ids.len() == prepared.seed.prompt_token_count
            && token_ids_sha256(&materialized.seed.token_ids)
                == prepared.seed.prompt_token_ids_sha256,
        "materialized fair-prefix seed differs from prepared token evidence"
    );
    ensure!(
        materialized.cache_seed.token_ids.len() == prepared.requested_shared_tokens
            && token_ids_sha256(&materialized.cache_seed.token_ids)
                == prepared.common_prefix_token_ids_sha256
            && !materialized.cache_seed.rendered_prompt.is_empty()
            && materialized
                .seed
                .rendered_prompt
                .starts_with(&materialized.cache_seed.rendered_prompt),
        "materialized fair-prefix cache seed is not the prepared exact 16K endpoint"
    );
    for (index, (actual, expected)) in materialized.tasks.iter().zip(&prepared.tasks).enumerate() {
        ensure!(
            actual.token_ids.len() == expected.prompt_token_count
                && token_ids_sha256(&actual.token_ids) == expected.prompt_token_ids_sha256
                && sha256_bytes(actual.rendered_prompt.as_bytes())
                    == expected.rendered_prompt_sha256,
            "materialized fair-prefix task {index} differs from prepared evidence"
        );
        ensure!(
            expected.lcp_with_seed_tokens == Some(prepared.requested_shared_tokens),
            "fair-prefix task {index} does not have the exact prepared 16K LCP"
        );
        ensure!(
            actual.token_ids.get(..prepared.requested_shared_tokens)
                == Some(materialized.cache_seed.token_ids.as_slice())
                && actual
                    .rendered_prompt
                    .starts_with(&materialized.cache_seed.rendered_prompt),
            "fair-prefix task {index} does not retain the raw exact cache seed"
        );
    }
    ensure!(
        verified.manifest.runtime.device == super::BenchmarkDevice::Metal,
        "fair-prefix benchmark currently requires the Metal runtime"
    );
    ensure!(
        verified.manifest.runtime.dtype == super::BenchmarkDType::F32,
        "fair-prefix benchmark currently requires MacLLM F32"
    );
    ensure!(
        verified.manifest.runtime.kv_cache_block_tokens == prepared.kv_cache_block_tokens,
        "fair-prefix runtime and prepared prompt disagree on the MacLLM cache block size"
    );
    ensure!(
        verified.manifest.runtime.kv_cache_capacity_tokens >= MAC_PREFIX_DEMO_CACHE_CAPACITY_TOKENS,
        "fair-prefix MacLLM physical K/V pool must hold at least {MAC_PREFIX_DEMO_CACHE_CAPACITY_TOKENS} tokens"
    );
    ensure!(
        verified.manifest.workload.decode_steps == MAC_PREFIX_DEMO_DECODE_TOKENS
            && verified.manifest.workload.warmup_waves == FAIR_WARMUP_WAVES_PER_LIFETIME
            && verified.manifest.workload.samples_per_concurrency
                == measured_waves
                    .checked_mul(MAC_PREFIX_DEMO_CONCURRENCY)
                    .context("fair-prefix manifest sample count overflow")?
            && verified.manifest.workload.prefix_cache == super::PrefixCacheMode::Warm,
        "fair-prefix manifest workload must declare 32-token completions, five warm-up waves, {} retained requests, and warm prefix cache",
        measured_waves * MAC_PREFIX_DEMO_CONCURRENCY
    );
    ensure!(
        verified
            .manifest
            .llama_cpp
            .cache_type_k
            .eq_ignore_ascii_case("f32")
            && verified
                .manifest
                .llama_cpp
                .cache_type_v
                .eq_ignore_ascii_case("f32"),
        "fair-prefix matched-precision lane requires llama.cpp F32 K/V cache"
    );
    Ok(())
}

fn build_contract(
    verified: &VerifiedScoreboard,
    prepared: &PreparedPrefixDemo,
    measured_waves: usize,
    port: u16,
) -> Result<FairPrefixContract> {
    let sampling = verified.manifest.workload.sampling;
    let runtime = &verified.manifest.runtime;
    let llama = &verified.manifest.llama_cpp;
    let llama_cpp_threads = verified
        .host
        .performance_cores
        .or(verified.host.total_cores)
        .context("verified host has no CPU core count for pinned llama-server threads")?;
    let llama_cpp_flash_attention = match llama.flash_attention {
        super::LlamaFlashAttention::On => "on",
        super::LlamaFlashAttention::Off => "off",
        super::LlamaFlashAttention::Auto => "auto",
    };
    let contract = FairPrefixContract {
        comparison_scope: "controlled_exact_slot_cache_lane".to_owned(),
        exact_shared_prefix_tokens: prepared.requested_shared_tokens,
        macllm_prefix_cache_block_tokens: prepared.kv_cache_block_tokens,
        macllm_kv_cache_slab_blocks: runtime.kv_cache_slab_blocks,
        macllm_kv_cache_capacity_tokens: runtime
            .kv_cache_capacity_tokens
            .max(MAC_PREFIX_DEMO_CACHE_CAPACITY_TOKENS),
        macllm_prefill_token_quantum: runtime.prefill_token_quantum,
        macllm_scheduler_token_budget: MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET,
        concurrency: MAC_PREFIX_DEMO_CONCURRENCY,
        completion_tokens_per_request: MAC_PREFIX_DEMO_DECODE_TOKENS,
        measured_waves_per_implementation: measured_waves,
        post_seed_stabilization_ms: u64::try_from(FAIR_POST_SEED_STABILIZATION.as_millis())
            .context("fair-prefix stabilization duration does not fit u64")?,
        warmup_waves_per_lifetime: FAIR_WARMUP_WAVES_PER_LIFETIME,
        steady_state_window_waves: FAIR_STEADY_STATE_WINDOW_WAVES,
        steady_state_max_cv_bits: FAIR_STEADY_STATE_MAX_CV.to_bits(),
        steady_state_max_relative_trend_bits: FAIR_STEADY_STATE_MAX_RELATIVE_TREND.to_bits(),
        temperature_bits: sampling.temperature.to_bits(),
        top_k: sampling.top_k,
        top_p_bits: sampling.top_p.to_bits(),
        repetition_penalty_bits: f64::from(sampling.repetition_penalty).to_bits(),
        seed: sampling.seed,
        ignore_eos: sampling.ignore_stop_tokens,
        cache_prompt: true,
        stream: true,
        return_tokens: true,
        macllm_kv_dtype: "f32".to_owned(),
        llama_cpp_k_cache_dtype: verified
            .manifest
            .llama_cpp
            .cache_type_k
            .to_ascii_lowercase(),
        llama_cpp_v_cache_dtype: verified
            .manifest
            .llama_cpp
            .cache_type_v
            .to_ascii_lowercase(),
        llama_cpp_context_tokens: expected_llama_context_tokens(
            prepared,
            runtime.kv_cache_capacity_tokens,
        )?,
        llama_cpp_batch_tokens: MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET,
        llama_cpp_ubatch_tokens: runtime.prefill_token_quantum,
        llama_cpp_threads,
        llama_cpp_gpu_layers: llama.gpu_layers,
        llama_cpp_flash_attention: llama_cpp_flash_attention.to_owned(),
        llama_cpp_mmap: llama.mmap,
        loopback_port: port,
        transport: FAIR_TRANSPORT.to_owned(),
        child_environment: fair_child_environment(),
    };
    validate_contract(&contract, prepared, &verified.host)?;
    Ok(contract)
}

fn expected_llama_context_tokens(
    prepared: &PreparedPrefixDemo,
    manifest_kv_capacity_tokens: usize,
) -> Result<usize> {
    let maximum_prompt_tokens = prepared
        .tasks
        .iter()
        .map(|task| task.prompt_token_count)
        .max()
        .context("fair-prefix prepared evidence has no measured tasks")?;
    let minimum_tokens_per_slot = maximum_prompt_tokens
        .checked_add(MAC_PREFIX_DEMO_DECODE_TOKENS)
        .and_then(|tokens| tokens.checked_add(PREFIX_DEMO_CONTEXT_HEADROOM_TOKENS))
        .context("fair-prefix llama.cpp per-slot context requirement overflow")?;
    let minimum_total_context_tokens = minimum_tokens_per_slot
        .checked_mul(MAC_PREFIX_DEMO_CONCURRENCY)
        .context("fair-prefix llama.cpp total context requirement overflow")?;
    let requested_total = manifest_kv_capacity_tokens.max(minimum_total_context_tokens);
    let total_alignment = LLAMA_CONTEXT_ALIGNMENT_PER_SLOT
        .checked_mul(MAC_PREFIX_DEMO_CONCURRENCY)
        .context("fair-prefix llama.cpp context alignment overflow")?;
    round_up_to(requested_total, total_alignment)
        .context("fair-prefix llama.cpp total context alignment overflow")
}

fn round_up_to(value: usize, alignment: usize) -> Option<usize> {
    let remainder = value.checked_rem(alignment)?;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

// The contract is intentionally validated in one place so the executable
// benchmark and the persisted attestation cannot silently drift apart.
#[allow(clippy::too_many_lines)]
fn validate_contract(
    contract: &FairPrefixContract,
    prepared: &PreparedPrefixDemo,
    host: &SanitizedHost,
) -> Result<()> {
    let expected_llama_cpp_threads = host
        .performance_cores
        .or(host.total_cores)
        .context("fair-prefix host has no CPU core count for pinned llama-server threads")?;
    ensure!(
        contract.comparison_scope == "controlled_exact_slot_cache_lane",
        "fair-prefix comparison scope changed"
    );
    ensure!(
        contract.exact_shared_prefix_tokens == prepared.requested_shared_tokens
            && contract.exact_shared_prefix_tokens == 16_384,
        "fair-prefix contract lost the exact 16K endpoint"
    );
    ensure!(
        contract.macllm_prefix_cache_block_tokens == prepared.kv_cache_block_tokens
            && contract.macllm_prefix_cache_block_tokens > 0
            && contract
                .exact_shared_prefix_tokens
                .is_multiple_of(contract.macllm_prefix_cache_block_tokens),
        "fair-prefix MacLLM prefix-cache block geometry changed"
    );
    ensure!(
        contract.macllm_kv_cache_slab_blocks > 0
            && contract.macllm_kv_cache_capacity_tokens >= MAC_PREFIX_DEMO_CACHE_CAPACITY_TOKENS
            && contract
                .macllm_kv_cache_capacity_tokens
                .is_multiple_of(contract.macllm_prefix_cache_block_tokens)
            && contract.macllm_prefill_token_quantum == prepared.prefill_quantum
            && contract.macllm_scheduler_token_budget == MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET
            && contract.macllm_prefill_token_quantum <= contract.macllm_scheduler_token_budget,
        "fair-prefix MacLLM launch geometry changed"
    );
    ensure!(
        contract.concurrency == MAC_PREFIX_DEMO_CONCURRENCY
            && contract.completion_tokens_per_request == MAC_PREFIX_DEMO_DECODE_TOKENS,
        "fair-prefix concurrency or completion boundary changed"
    );
    ensure!(
        contract.measured_waves_per_implementation >= MAC_PREFIX_DEMO_MIN_MEASURED_WAVES,
        "fair-prefix contract retained too few measured waves"
    );
    ensure!(
        contract
            .measured_waves_per_implementation
            .is_multiple_of(FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION),
        "fair-prefix measured waves do not split evenly across counterbalanced lifetimes"
    );
    ensure!(
        contract.post_seed_stabilization_ms
            == u64::try_from(FAIR_POST_SEED_STABILIZATION.as_millis())
                .context("fair-prefix stabilization duration does not fit u64")?,
        "fair-prefix post-seed stabilization interval changed"
    );
    ensure!(
        contract.warmup_waves_per_lifetime == FAIR_WARMUP_WAVES_PER_LIFETIME
            && contract.steady_state_window_waves == FAIR_STEADY_STATE_WINDOW_WAVES
            && contract.steady_state_window_waves <= contract.warmup_waves_per_lifetime
            && contract.steady_state_max_cv_bits == FAIR_STEADY_STATE_MAX_CV.to_bits()
            && contract.steady_state_max_relative_trend_bits
                == FAIR_STEADY_STATE_MAX_RELATIVE_TREND.to_bits(),
        "fair-prefix warm-up steady-state gate changed"
    );
    ensure!(
        contract.temperature_bits == 0.0_f64.to_bits()
            && contract.top_k == 0
            && contract.top_p_bits == 1.0_f64.to_bits()
            && contract.repetition_penalty_bits == 1.0_f64.to_bits()
            && contract.seed == 42
            && contract.ignore_eos
            && contract.cache_prompt
            && contract.stream
            && contract.return_tokens,
        "fair-prefix contract is not fixed greedy generation"
    );
    ensure!(
        contract.macllm_kv_dtype.eq_ignore_ascii_case("f32")
            && contract
                .llama_cpp_k_cache_dtype
                .eq_ignore_ascii_case(&contract.macllm_kv_dtype)
            && contract
                .llama_cpp_v_cache_dtype
                .eq_ignore_ascii_case(&contract.macllm_kv_dtype),
        "fair-prefix K/V precision is not matched"
    );
    ensure!(
        contract.llama_cpp_context_tokens
            == expected_llama_context_tokens(prepared, contract.macllm_kv_cache_capacity_tokens,)?
            && contract.llama_cpp_batch_tokens == contract.macllm_scheduler_token_budget
            && contract.llama_cpp_ubatch_tokens == contract.macllm_prefill_token_quantum
            && expected_llama_cpp_threads > 0
            && contract.llama_cpp_threads == expected_llama_cpp_threads
            && contract.llama_cpp_gpu_layers == 99
            && contract.llama_cpp_flash_attention == "on"
            && contract.loopback_port > 0,
        "fair-prefix llama.cpp launch geometry changed"
    );
    ensure!(
        contract.transport == FAIR_TRANSPORT
            && contract.child_environment == fair_child_environment(),
        "fair-prefix transport or child-process environment boundary changed"
    );
    Ok(())
}

fn validate_launch_commands(report: &FairPrefixReport) -> Result<()> {
    let expected_base_url = format!("http://{}:{}", LOOPBACK, report.contract.loopback_port);
    ensure!(
        report.macllm.base_url == expected_base_url
            && report.llama_cpp.base_url == expected_base_url,
        "fair-prefix implementations did not use the contract loopback address"
    );
    let expected_llama_command = expected_llama_cpp_process_command(report)?;
    ensure!(
        report.macllm.process_command == expected_macllm_process_command(report)?
            && report.llama_cpp.process_command == expected_llama_command,
        "fair-prefix launch argv differs from the model, precision, cache, batching, or transport contract"
    );
    let expected_attestation_command = attestation_argv_from_measured(expected_llama_command)?;
    ensure!(
        report
            .llama_cpp_control_attestation
            .process_command
            .as_ref()
            == Some(&expected_attestation_command),
        "llama.cpp runtime attestation command differs from the measured command beyond logging"
    );
    Ok(())
}

fn expected_macllm_process_command(report: &FairPrefixReport) -> Result<Vec<String>> {
    let contract = &report.contract;
    let executable = report
        .macllm_provenance
        .executable
        .canonical_path
        .to_str()
        .context("MacLLM executable path is not UTF-8")?;
    let model = report
        .model
        .file
        .canonical_path
        .to_str()
        .context("fair-prefix model path is not UTF-8")?;
    Ok(vec![
        executable.to_owned(),
        "serve".to_owned(),
        "--model".to_owned(),
        model.to_owned(),
        "--device".to_owned(),
        "metal".to_owned(),
        "--dtype".to_owned(),
        contract.macllm_kv_dtype.clone(),
        "--kv-cache-block-tokens".to_owned(),
        contract.macllm_prefix_cache_block_tokens.to_string(),
        "--kv-cache-slab-blocks".to_owned(),
        contract.macllm_kv_cache_slab_blocks.to_string(),
        "--kv-cache-capacity-tokens".to_owned(),
        contract.macllm_kv_cache_capacity_tokens.to_string(),
        "--prefix-cache-capacity-tokens".to_owned(),
        contract.exact_shared_prefix_tokens.to_string(),
        "--host".to_owned(),
        LOOPBACK.to_string(),
        "--port".to_owned(),
        contract.loopback_port.to_string(),
        "--served-model-name".to_owned(),
        report.profile.clone(),
        "--queue-capacity".to_owned(),
        contract.concurrency.to_string(),
        "--max-active-sequences".to_owned(),
        contract.concurrency.to_string(),
        "--scheduler-token-budget".to_owned(),
        contract.macllm_scheduler_token_budget.to_string(),
        "--prefill-token-quantum".to_owned(),
        contract.macllm_prefill_token_quantum.to_string(),
        "--event-capacity".to_owned(),
        contract
            .completion_tokens_per_request
            .checked_add(2)
            .context("fair-prefix event capacity overflow")?
            .to_string(),
        "--event-send-timeout-seconds".to_owned(),
        REQUEST_TIMEOUT.as_secs().to_string(),
    ])
}

fn expected_llama_cpp_process_command(report: &FairPrefixReport) -> Result<Vec<String>> {
    let contract = &report.contract;
    let executable = report
        .llama_cpp_server
        .canonical_path
        .to_str()
        .context("llama.cpp executable path is not UTF-8")?;
    let model = report
        .model
        .file
        .canonical_path
        .to_str()
        .context("fair-prefix model path is not UTF-8")?;
    let mmap = if contract.llama_cpp_mmap {
        "--mmap"
    } else {
        "--no-mmap"
    };
    Ok(vec![
        executable.to_owned(),
        "--model".to_owned(),
        model.to_owned(),
        "--host".to_owned(),
        LOOPBACK.to_string(),
        "--port".to_owned(),
        contract.loopback_port.to_string(),
        "--parallel".to_owned(),
        contract.concurrency.to_string(),
        "--ctx-size".to_owned(),
        contract.llama_cpp_context_tokens.to_string(),
        "--batch-size".to_owned(),
        contract.llama_cpp_batch_tokens.to_string(),
        "--ubatch-size".to_owned(),
        contract.llama_cpp_ubatch_tokens.to_string(),
        "--threads".to_owned(),
        contract.llama_cpp_threads.to_string(),
        "--threads-batch".to_owned(),
        contract.llama_cpp_threads.to_string(),
        "--threads-http".to_owned(),
        contract.concurrency.to_string(),
        "--n-gpu-layers".to_owned(),
        contract.llama_cpp_gpu_layers.to_string(),
        "--flash-attn".to_owned(),
        contract.llama_cpp_flash_attention.clone(),
        "--cache-type-k".to_owned(),
        contract.llama_cpp_k_cache_dtype.clone(),
        "--cache-type-v".to_owned(),
        contract.llama_cpp_v_cache_dtype.clone(),
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
        REQUEST_TIMEOUT.as_secs().to_string(),
        "--offline".to_owned(),
        "--log-disable".to_owned(),
        "--device".to_owned(),
        "MTL0".to_owned(),
    ])
}

#[derive(Debug, Clone)]
struct FairPromptSpec {
    label: String,
    prompt: String,
    token_ids: Vec<u32>,
    prompt_tokens: usize,
    prompt_token_ids_sha256: String,
    slot_id: usize,
}

#[derive(Debug, Clone)]
struct FairPromptSet {
    seed: FairPromptSpec,
    tasks: [FairPromptSpec; MAC_PREFIX_DEMO_CONCURRENCY],
    conditioning_lcp_tokens: Vec<usize>,
}

impl FairPromptSet {
    fn new(materialized: &MaterializedPrefixDemo, contract: &FairPrefixContract) -> Result<Self> {
        let seed = FairPromptSpec {
            label: "seed".to_owned(),
            prompt: materialized.cache_seed.rendered_prompt.clone(),
            token_ids: materialized.cache_seed.token_ids.clone(),
            prompt_tokens: materialized.cache_seed.token_ids.len(),
            prompt_token_ids_sha256: token_ids_sha256(&materialized.cache_seed.token_ids),
            slot_id: 0,
        };
        ensure!(
            seed.prompt_tokens == contract.exact_shared_prefix_tokens,
            "fair-prefix seed is not exactly the cache endpoint"
        );
        let tasks: [FairPromptSpec; MAC_PREFIX_DEMO_CONCURRENCY] = materialized
            .tasks
            .iter()
            .enumerate()
            .map(|(slot_id, task)| FairPromptSpec {
                label: format!("request_{slot_id}"),
                prompt: task.rendered_prompt.clone(),
                token_ids: task.token_ids.clone(),
                prompt_tokens: task.token_ids.len(),
                prompt_token_ids_sha256: token_ids_sha256(&task.token_ids),
                slot_id,
            })
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| anyhow::anyhow!("fair-prefix task set must contain eight prompts"))?;
        let conditioning_lcp_tokens = (0..MAC_PREFIX_DEMO_CONCURRENCY)
            .map(|slot_id| {
            let conditioning = &tasks[(slot_id + 1) % MAC_PREFIX_DEMO_CONCURRENCY];
            let measured = &tasks[slot_id];
            ensure!(
                conditioning
                    .token_ids
                    .get(..seed.prompt_tokens)
                    == Some(seed.token_ids.as_slice())
                    && measured.token_ids.get(..seed.prompt_tokens) == Some(seed.token_ids.as_slice()),
                "fair-prefix task pair for slot {slot_id} does not begin at the exact cache seed"
            );
            let lcp = conditioning
                .token_ids
                .iter()
                .zip(&measured.token_ids)
                .take_while(|(left, right)| left == right)
                .count();
            ensure!(
                lcp == contract.exact_shared_prefix_tokens,
                "fair-prefix conditioning prompt for slot {slot_id} does not diverge exactly at 16K"
            );
            Ok(lcp)
        })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            seed,
            tasks,
            conditioning_lcp_tokens,
        })
    }

    fn slot_seeds(&self) -> [FairPromptSpec; MAC_PREFIX_DEMO_CONCURRENCY] {
        std::array::from_fn(|slot_id| {
            let mut seed = self.seed.clone();
            seed.label = format!("seed_slot_{slot_id}");
            seed.slot_id = slot_id;
            seed
        })
    }

    fn conditioning_prompts(&self) -> [FairPromptSpec; MAC_PREFIX_DEMO_CONCURRENCY] {
        std::array::from_fn(|slot_id| {
            let source = (slot_id + 1) % MAC_PREFIX_DEMO_CONCURRENCY;
            let mut conditioning = self.tasks[source].clone();
            conditioning.label = format!("restore_slot_{slot_id}_from_task_{source}");
            conditioning.slot_id = slot_id;
            conditioning
        })
    }

    fn conditioning_prompt_token_ids_sha256(&self) -> Vec<String> {
        (0..MAC_PREFIX_DEMO_CONCURRENCY)
            .map(|slot_id| {
                self.tasks[(slot_id + 1) % MAC_PREFIX_DEMO_CONCURRENCY]
                    .prompt_token_ids_sha256
                    .clone()
            })
            .collect()
    }
}

#[allow(clippy::too_many_lines)] // Every ratio eligibility invariant remains visible in one validator.
fn validate_implementation_run(
    run: &FairPrefixImplementationRun,
    contract: &FairPrefixContract,
    prepared: &PreparedPrefixDemo,
    profile: &str,
) -> Result<()> {
    ensure!(
        run.implementation == EndpointKind::LlamaCpp.label()
            || run.implementation == EndpointKind::Macllm.label(),
        "unsupported fair-prefix implementation {}",
        run.implementation
    );
    let expected_runtime_attestations =
        usize::from(run.implementation == EndpointKind::Macllm.label())
            * FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION;
    let unique_process_sequence_indices = run
        .process_sequence_indices
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let expected_process_sequence_indices = if run.implementation == EndpointKind::Macllm.label() {
        [1, 2, 4, 7]
    } else {
        [0, 3, 5, 6]
    };
    ensure!(
        run.process_sequence_indices.len() == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
            && unique_process_sequence_indices.len() == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
            && run.process_sequence_indices == expected_process_sequence_indices
            && run.health_wait_ms_samples.len() == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
            && run.post_seed_stabilization_ms_samples.len()
                == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
            && run
                .health_wait_ms_samples
                .iter()
                .all(|sample| sample.is_finite() && *sample >= 0.0)
            && run.post_seed_stabilization_ms_samples.iter().all(|sample| {
                sample.is_finite() && *sample >= contract.post_seed_stabilization_ms as f64
            })
            && run.warmup_waves_per_lifetime == FAIR_WARMUP_WAVES_PER_LIFETIME
            && run.warmup_aggregate_output_tokens_per_second_samples.len()
                == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION * FAIR_WARMUP_WAVES_PER_LIFETIME
            && run.runtime_attestations.len() == expected_runtime_attestations
            && run.isolated_os_process
            && run.cache_prompt_publication
            && run.port_closed_after_shutdown,
        "{} did not retain {} clean, equally stabilized and warmed process lifetimes",
        run.implementation,
        FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
    );
    for (process_sequence_index, samples) in run.process_sequence_indices.iter().zip(
        run.warmup_aggregate_output_tokens_per_second_samples
            .chunks_exact(FAIR_WARMUP_WAVES_PER_LIFETIME),
    ) {
        validate_steady_state_samples(
            samples,
            contract,
            &run.implementation,
            *process_sequence_index,
        )?;
    }
    for attestation in &run.runtime_attestations {
        validate_runtime_attestation(attestation, &run.implementation, profile, contract)?;
    }
    let expected_seed_requests = FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION * contract.concurrency;
    ensure!(
        run.seed_requests == expected_seed_requests,
        "{} retained the wrong number of excluded seed requests",
        run.implementation
    );
    let expected_waves_per_lifetime =
        contract.measured_waves_per_implementation / FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION;
    let expected_restore_requests = FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
        .checked_mul(
            expected_waves_per_lifetime
                .checked_add(FAIR_WARMUP_WAVES_PER_LIFETIME)
                .context("fair-prefix restored wave count overflow")?,
        )
        .and_then(|restored_waves| restored_waves.checked_mul(contract.concurrency))
        .context("fair-prefix restore request count overflow")?;
    ensure!(
        run.restore_requests == expected_restore_requests,
        "{} retained the wrong number of excluded restore requests",
        run.implementation
    );
    let expected_conditioning_digests = (0..contract.concurrency)
        .map(|slot_id| {
            prepared.tasks[(slot_id + 1) % contract.concurrency]
                .prompt_token_ids_sha256
                .clone()
        })
        .collect::<Vec<_>>();
    ensure!(
        run.conditioning_prompt_token_ids_sha256 == expected_conditioning_digests,
        "{} retained the wrong rotated conditioning prompt identities",
        run.implementation
    );
    ensure!(
        run.conditioning_lcp_tokens.len() == contract.concurrency
            && run
                .conditioning_lcp_tokens
                .iter()
                .all(|lcp| *lcp == contract.exact_shared_prefix_tokens),
        "{} conditioning prompts do not diverge from the next measured prompts at exactly {} tokens",
        run.implementation,
        contract.exact_shared_prefix_tokens
    );
    for process_sequence_index in &run.process_sequence_indices {
        ensure!(
            run.waves
                .iter()
                .filter(|wave| { wave.process_sequence_index == *process_sequence_index })
                .count()
                == expected_waves_per_lifetime,
            "{} process sequence index {} retained the wrong number of waves",
            run.implementation,
            process_sequence_index
        );
    }
    ensure!(
        run.measurement_boundary.contains(FAIR_TRANSPORT),
        "{} measurement boundary is not the shared transport",
        run.implementation
    );
    ensure!(
        run.tokenizer_exact,
        "{} did not prove exact tokenizer input parity",
        run.implementation
    );
    ensure!(
        run.waves.len() == contract.measured_waves_per_implementation,
        "{} retained {} waves; expected {}",
        run.implementation,
        run.waves.len(),
        contract.measured_waves_per_implementation
    );
    for (wave_index, wave) in run.waves.iter().enumerate() {
        let expected_process_sequence_index =
            run.process_sequence_indices[wave_index / expected_waves_per_lifetime];
        ensure!(
            wave.wave_index == wave_index
                && wave.process_sequence_index == expected_process_sequence_index
                && wave.request_count == contract.concurrency
                && wave.requests.len() == contract.concurrency
                && wave.exact_cache_hit_requests == contract.concurrency,
            "{} wave {wave_index} does not satisfy the complete eight-way contract",
            run.implementation
        );
        let expected_total = u64::try_from(
            contract
                .concurrency
                .checked_mul(contract.completion_tokens_per_request)
                .context("fair-prefix completion total overflow")?,
        )?;
        ensure!(
            wave.requests.iter().all(|request| {
                request.submitted_offset_ms.is_finite()
                    && request.submitted_offset_ms >= 0.0
                    && request.first_token_offset_ms.is_finite()
                    && request.first_token_offset_ms >= 0.0
                    && request.completed_offset_ms.is_finite()
                    && request.completed_offset_ms >= 0.0
            }),
            "{} wave {wave_index} contains invalid relative timestamps",
            run.implementation
        );
        let earliest_submission = wave
            .requests
            .iter()
            .map(|request| request.submitted_offset_ms)
            .min_by(f64::total_cmp)
            .context("fair-prefix wave contains no submission offsets")?;
        let latest_completion = wave
            .requests
            .iter()
            .map(|request| request.completed_offset_ms)
            .max_by(f64::total_cmp)
            .context("fair-prefix wave contains no completion offsets")?;
        let reconstructed_wall_time = latest_completion - earliest_submission;
        ensure!(
            wave.total_completion_tokens == expected_total
                && positive_finite(wave.external_wall_time_ms)
                && positive_finite(wave.aggregate_output_tokens_per_second)
                && approximately_equal(earliest_submission, 0.0)
                && approximately_equal(wave.external_wall_time_ms, reconstructed_wall_time)
                && approximately_equal(
                    wave.aggregate_output_tokens_per_second,
                    expected_total as f64 * 1_000.0 / wave.external_wall_time_ms,
                ),
            "{} wave {wave_index} aggregate differs from retained requests",
            run.implementation
        );
        for (request_index, request) in wave.requests.iter().enumerate() {
            let expected = &prepared.tasks[request_index];
            ensure!(
                request.request_index == request_index
                    && request.slot_id == request_index
                    && request.prompt_tokens == expected.prompt_token_count
                    && request.prompt_token_ids_sha256 == expected.prompt_token_ids_sha256
                    && request.streamed_token_ids.len() == contract.completion_tokens_per_request
                    && run
                        .waves
                        .first()
                        .and_then(|reference| { reference.requests.get(request_index) })
                        .is_some_and(|reference| {
                            request.streamed_token_ids == reference.streamed_token_ids
                        })
                    && request.terminal.cached_prompt_tokens == contract.exact_shared_prefix_tokens
                    && request.terminal.evaluated_prompt_tokens
                        == expected.prompt_token_count - contract.exact_shared_prefix_tokens
                    && request.terminal.generated_tokens == contract.completion_tokens_per_request
                    && request.terminal.server_prompt_ms.is_finite()
                    && request.terminal.server_prompt_ms >= 0.0
                    && request.terminal.server_predicted_ms.is_finite()
                    && request.terminal.server_predicted_ms >= 0.0
                    && request.submitted_offset_ms <= request.first_token_offset_ms
                    && request.first_token_offset_ms <= request.completed_offset_ms
                    && approximately_equal(
                        request.external_time_to_first_token_ms,
                        request.first_token_offset_ms - request.submitted_offset_ms,
                    )
                    && approximately_equal(
                        request.external_latency_ms,
                        request.completed_offset_ms - request.submitted_offset_ms,
                    )
                    && positive_finite(request.external_time_to_first_token_ms)
                    && positive_finite(request.external_latency_ms)
                    && request.external_latency_ms >= request.external_time_to_first_token_ms,
                "{} wave {wave_index} request {request_index} violates the fixed request contract",
                run.implementation
            );
        }
    }
    ensure!(
        run.lifetime_replays.len() == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
            && run
                .lifetime_replays
                .iter()
                .all(|replay| replay == &run.replay)
            && run.replay.content.len() == contract.concurrency
            && run.replay.content_sha256.len() == contract.concurrency
            && run.replay.prompt_token_ids_sha256.len() == contract.concurrency,
        "{} did not retain four identical lifetime replays with eight tasks",
        run.implementation
    );
    for index in 0..contract.concurrency {
        ensure!(
            run.replay.prompt_token_ids_sha256[index]
                == prepared.tasks[index].prompt_token_ids_sha256
                && sha256_bytes(run.replay.content[index].as_bytes())
                    == run.replay.content_sha256[index],
            "{} replay identity differs for task {index}",
            run.implementation
        );
    }
    for wave in &run.waves {
        for request in &wave.requests {
            ensure!(
                request.streamed_content == run.replay.content[request.request_index],
                "{} wave {} streamed content differs from its excluded correctness replay for request {}",
                run.implementation,
                wave.wave_index,
                request.request_index
            );
        }
    }
    validate_distributions(&run.distributions, &run.waves)
}

fn validate_distributions(
    distributions: &FairPrefixDistributionEvidence,
    waves: &[FairPrefixWaveEvidence],
) -> Result<()> {
    let throughput = waves
        .iter()
        .map(|wave| wave.aggregate_output_tokens_per_second)
        .collect::<Vec<_>>();
    let ttft = waves
        .iter()
        .map(|wave| {
            wave.requests
                .iter()
                .map(|request| request.external_time_to_first_token_ms)
                .max_by(f64::total_cmp)
                .context("fair-prefix wave contains no TTFT samples")
        })
        .collect::<Result<Vec<_>>>()?;
    let latency = waves
        .iter()
        .map(|wave| {
            wave.requests
                .iter()
                .map(|request| request.external_latency_ms)
                .max_by(f64::total_cmp)
                .context("fair-prefix wave contains no latency samples")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        distributions.aggregate_output_tokens_per_second_samples == throughput
            && distributions.wave_max_time_to_first_token_ms_samples == ttft
            && distributions.wave_max_latency_ms_samples == latency,
        "fair-prefix raw distribution samples differ from waves"
    );
    ensure_distribution_equal(
        &distributions.aggregate_output_tokens_per_second,
        &summarize(&throughput)?,
        "aggregate throughput",
    )?;
    ensure_distribution_equal(
        &distributions.wave_max_time_to_first_token_ms,
        &summarize(&ttft)?,
        "wave-max TTFT",
    )?;
    ensure_distribution_equal(
        &distributions.wave_max_latency_ms,
        &summarize(&latency)?,
        "wave-max latency",
    )
}

fn ensure_distribution_equal(
    actual: &DistributionSummary,
    expected: &DistributionSummary,
    label: &str,
) -> Result<()> {
    ensure!(
        actual.sample_count == expected.sample_count
            && approximately_equal(actual.mean, expected.mean)
            && approximately_equal(actual.p50, expected.p50)
            && approximately_equal(actual.p95, expected.p95)
            && approximately_equal(actual.max, expected.max),
        "fair-prefix {label} distribution differs from raw samples"
    );
    Ok(())
}

fn build_distributions(waves: &[FairPrefixWaveEvidence]) -> Result<FairPrefixDistributionEvidence> {
    let throughput = waves
        .iter()
        .map(|wave| wave.aggregate_output_tokens_per_second)
        .collect::<Vec<_>>();
    let ttft = waves
        .iter()
        .map(|wave| {
            wave.requests
                .iter()
                .map(|request| request.external_time_to_first_token_ms)
                .max_by(f64::total_cmp)
                .context("fair-prefix wave contains no TTFT samples")
        })
        .collect::<Result<Vec<_>>>()?;
    let latency = waves
        .iter()
        .map(|wave| {
            wave.requests
                .iter()
                .map(|request| request.external_latency_ms)
                .max_by(f64::total_cmp)
                .context("fair-prefix wave contains no latency samples")
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(FairPrefixDistributionEvidence {
        aggregate_output_tokens_per_second: summarize(&throughput)?,
        aggregate_output_tokens_per_second_samples: throughput,
        wave_max_time_to_first_token_ms: summarize(&ttft)?,
        wave_max_time_to_first_token_ms_samples: ttft,
        wave_max_latency_ms: summarize(&latency)?,
        wave_max_latency_ms_samples: latency,
    })
}

#[allow(clippy::too_many_lines)] // The fail-closed comparison gates are intentionally one explicit table.
#[allow(clippy::too_many_arguments)] // Every independent eligibility input remains explicit at the audit boundary.
fn derive_eligibility(
    contract: &FairPrefixContract,
    model: &ModelProvenance,
    prepared: &PreparedPrefixDemo,
    profile: &str,
    execution_order: &[String],
    macllm: &FairPrefixImplementationRun,
    llama_cpp: &FairPrefixImplementationRun,
    llama_cpp_control_attestation: &FairPrefixRuntimeAttestation,
) -> FairPrefixEligibility {
    let model_identity_parity = model.file == prepared.model_file;
    let precision_parity = contract
        .macllm_kv_dtype
        .eq_ignore_ascii_case(&contract.llama_cpp_k_cache_dtype)
        && contract
            .macllm_kv_dtype
            .eq_ignore_ascii_case(&contract.llama_cpp_v_cache_dtype);
    let transport_parity = macllm.measurement_boundary.contains(FAIR_TRANSPORT)
        && llama_cpp.measurement_boundary.contains(FAIR_TRANSPORT);
    let process_isolation_parity = macllm.isolated_os_process
        && llama_cpp.isolated_os_process
        && macllm.port_closed_after_shutdown
        && llama_cpp.port_closed_after_shutdown;
    let runtime_attestation_gate = macllm.runtime_attestations.len()
        == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
        && macllm.runtime_attestations.iter().all(|attestation| {
            validate_runtime_attestation(attestation, &macllm.implementation, profile, contract)
                .is_ok()
        })
        && llama_cpp.runtime_attestations.is_empty()
        && validate_runtime_attestation(
            llama_cpp_control_attestation,
            EndpointKind::LlamaCpp.label(),
            profile,
            contract,
        )
        .is_ok();
    let expected_execution_order = FAIR_PROCESS_ORDER.map(str::to_owned);
    let counterbalanced_execution_order = execution_order == expected_execution_order
        && llama_cpp.process_sequence_indices == [0, 3, 5, 6]
        && macllm.process_sequence_indices == [1, 2, 4, 7];
    let stabilization_parity = [macllm, llama_cpp].iter().all(|run| {
        run.post_seed_stabilization_ms_samples.len() == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
            && run.post_seed_stabilization_ms_samples.iter().all(|sample| {
                sample.is_finite() && *sample >= contract.post_seed_stabilization_ms as f64
            })
    });
    let warmup_parity = macllm.warmup_waves_per_lifetime == FAIR_WARMUP_WAVES_PER_LIFETIME
        && llama_cpp.warmup_waves_per_lifetime == FAIR_WARMUP_WAVES_PER_LIFETIME
        && macllm.health_wait_ms_samples.len() == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
        && llama_cpp.health_wait_ms_samples.len() == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
        && macllm
            .warmup_aggregate_output_tokens_per_second_samples
            .len()
            == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION * FAIR_WARMUP_WAVES_PER_LIFETIME
        && llama_cpp
            .warmup_aggregate_output_tokens_per_second_samples
            .len()
            == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION * FAIR_WARMUP_WAVES_PER_LIFETIME;
    let steady_state_gate = [macllm, llama_cpp].iter().all(|run| {
        run.process_sequence_indices
            .iter()
            .zip(
                run.warmup_aggregate_output_tokens_per_second_samples
                    .chunks_exact(FAIR_WARMUP_WAVES_PER_LIFETIME),
            )
            .all(|(process_sequence_index, samples)| {
                validate_steady_state_samples(
                    samples,
                    contract,
                    &run.implementation,
                    *process_sequence_index,
                )
                .is_ok()
            })
    });
    let expected_restore_requests = contract
        .measured_waves_per_implementation
        .checked_add(FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION * FAIR_WARMUP_WAVES_PER_LIFETIME)
        .and_then(|restored_waves| restored_waves.checked_mul(contract.concurrency));
    let conditioning_parity = expected_restore_requests.is_some_and(|expected| {
        macllm.seed_requests == llama_cpp.seed_requests
            && macllm.seed_requests
                == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION * contract.concurrency
            && macllm.restore_requests == expected
            && llama_cpp.restore_requests == expected
            && macllm.conditioning_prompt_token_ids_sha256
                == llama_cpp.conditioning_prompt_token_ids_sha256
            && macllm.conditioning_lcp_tokens == llama_cpp.conditioning_lcp_tokens
            && macllm.conditioning_lcp_tokens.len() == contract.concurrency
            && macllm
                .conditioning_lcp_tokens
                .iter()
                .all(|lcp| *lcp == contract.exact_shared_prefix_tokens)
    });
    let cache_policy_parity = contract.cache_prompt
        && macllm.cache_prompt_publication
        && llama_cpp.cache_prompt_publication;
    let input_token_parity = macllm.replay.prompt_token_ids_sha256
        == llama_cpp.replay.prompt_token_ids_sha256
        && macllm.replay.prompt_token_ids_sha256
            == prepared
                .tasks
                .iter()
                .map(|task| task.prompt_token_ids_sha256.clone())
                .collect::<Vec<_>>();
    let output_content_parity = macllm.replay.content == llama_cpp.replay.content
        && macllm.replay.content_sha256 == llama_cpp.replay.content_sha256;
    let output_token_id_parity = macllm.waves.len() == llama_cpp.waves.len()
        && macllm
            .waves
            .iter()
            .zip(&llama_cpp.waves)
            .all(|(macllm_wave, llama_cpp_wave)| {
                macllm_wave.requests.len() == llama_cpp_wave.requests.len()
                    && macllm_wave
                        .requests
                        .iter()
                        .zip(&llama_cpp_wave.requests)
                        .all(|(macllm_request, llama_cpp_request)| {
                            macllm_request.request_index == llama_cpp_request.request_index
                                && macllm_request.streamed_token_ids
                                    == llama_cpp_request.streamed_token_ids
                        })
            });
    let sample_count_parity = macllm.waves.len() == llama_cpp.waves.len()
        && macllm.waves.len() == contract.measured_waves_per_implementation
        && macllm.waves.len() >= MAC_PREFIX_DEMO_MIN_MEASURED_WAVES;
    let cache_state_parity = [&macllm.waves, &llama_cpp.waves].iter().all(|waves| {
        waves.len() == contract.measured_waves_per_implementation
            && waves
                .iter()
                .all(|wave| wave.exact_cache_hit_requests == contract.concurrency)
    });
    let fixed_completion_parity = [&macllm.waves, &llama_cpp.waves].iter().all(|waves| {
        waves.iter().all(|wave| {
            wave.requests.iter().all(|request| {
                request.terminal.generated_tokens == contract.completion_tokens_per_request
            })
        })
    });
    let mut reasons = Vec::new();
    for (condition, reason) in [
        (model_identity_parity, "model file identity is not shared"),
        (precision_parity, "K/V cache precision is not matched"),
        (
            transport_parity,
            "external transport/timing boundary is not shared",
        ),
        (
            process_isolation_parity,
            "both implementations did not run in fresh isolated OS processes",
        ),
        (
            runtime_attestation_gate,
            "MacLLM process health attestations or the separate llama.cpp control attestation are invalid",
        ),
        (
            counterbalanced_execution_order,
            "process execution order is not the required four-block AB/BA counterbalance",
        ),
        (
            stabilization_parity,
            "process lifetimes did not receive the same post-seed stabilization interval",
        ),
        (
            warmup_parity,
            "process lifetimes did not receive the same excluded warm-up waves",
        ),
        (
            steady_state_gate,
            "one or more process lifetimes failed the pre-registered warm-up CV/trend gate",
        ),
        (
            conditioning_parity,
            "seed/restore counts, rotated prompt identities, or exact conditioning LCPs differ",
        ),
        (
            cache_policy_parity,
            "cache_prompt lookup/publication policy is not shared",
        ),
        (
            input_token_parity,
            "exact prompt token digests are not identical",
        ),
        (
            output_content_parity,
            "deterministic replay output content differs",
        ),
        (
            output_token_id_parity,
            "measured generated token IDs differ between implementations",
        ),
        (
            sample_count_parity,
            "retained wave counts are unequal or below 20",
        ),
        (
            cache_state_parity,
            "one or more waves failed the exact 8/8 warm-cache contract",
        ),
        (
            fixed_completion_parity,
            "one or more requests missed the fixed output-token boundary",
        ),
    ] {
        if !condition {
            reasons.push(reason.to_owned());
        }
    }
    FairPrefixEligibility {
        eligible: reasons.is_empty(),
        model_identity_parity,
        precision_parity,
        transport_parity,
        process_isolation_parity,
        runtime_attestation_gate,
        counterbalanced_execution_order,
        stabilization_parity,
        warmup_parity,
        steady_state_gate,
        conditioning_parity,
        cache_policy_parity,
        input_token_parity,
        output_content_parity,
        output_token_id_parity,
        sample_count_parity,
        cache_state_parity,
        fixed_completion_parity,
        reasons,
    }
}

fn derive_comparison(
    eligibility: &FairPrefixEligibility,
    macllm: &FairPrefixImplementationRun,
    llama_cpp: &FairPrefixImplementationRun,
) -> Result<FairPrefixComparison> {
    let mac_tps = macllm.distributions.aggregate_output_tokens_per_second.p50;
    let llama_tps = llama_cpp
        .distributions
        .aggregate_output_tokens_per_second
        .p50;
    ensure!(
        positive_finite(mac_tps) && positive_finite(llama_tps),
        "fair-prefix throughput must be positive"
    );
    let waves_per_lifetime = macllm.waves.len() / FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION;
    let mut paired_process_blocks = Vec::with_capacity(FAIR_PAIRED_PROCESS_POSITIONS.len());
    let mut eligible_pair_ratios = Vec::with_capacity(FAIR_PAIRED_PROCESS_POSITIONS.len());
    for (pair_index, &(llama_position, mac_position)) in
        FAIR_PAIRED_PROCESS_POSITIONS.iter().enumerate()
    {
        let llama_pair_tps =
            process_phase_throughput(llama_cpp, llama_position, waves_per_lifetime)?;
        let mac_pair_tps = process_phase_throughput(macllm, mac_position, waves_per_lifetime)?;
        let raw_ratio = mac_pair_tps / llama_pair_tps;
        ensure!(
            positive_finite(raw_ratio),
            "fair-prefix paired throughput ratio must be positive"
        );
        if eligibility.eligible {
            eligible_pair_ratios.push(raw_ratio);
        }
        let process_order = if llama_position < mac_position {
            vec![
                EndpointKind::LlamaCpp.label().to_owned(),
                EndpointKind::Macllm.label().to_owned(),
            ]
        } else {
            vec![
                EndpointKind::Macllm.label().to_owned(),
                EndpointKind::LlamaCpp.label().to_owned(),
            ]
        };
        paired_process_blocks.push(FairPrefixPairEvidence {
            pair_index,
            process_order,
            macllm_process_sequence_index: mac_position,
            llama_cpp_process_sequence_index: llama_position,
            retained_waves_per_implementation: waves_per_lifetime,
            macllm_phase_output_tokens_per_second: mac_pair_tps,
            llama_cpp_phase_output_tokens_per_second: llama_pair_tps,
            macllm_to_llama_cpp_phase_throughput_ratio: eligibility.eligible.then_some(raw_ratio),
        });
    }
    let paired_geomean_ratio = if eligibility.eligible {
        let mean_log = eligible_pair_ratios
            .iter()
            .map(|ratio| ratio.ln())
            .sum::<f64>()
            / eligible_pair_ratios.len() as f64;
        let ratio = mean_log.exp();
        ensure!(
            positive_finite(ratio),
            "eligible fair-prefix paired geometric-mean ratio must be positive"
        );
        Some(ratio)
    } else {
        None
    };
    Ok(FairPrefixComparison {
        macllm_p50_aggregate_output_tokens_per_second: mac_tps,
        llama_cpp_p50_aggregate_output_tokens_per_second: llama_tps,
        macllm_p50_wave_max_time_to_first_token_ms: macllm
            .distributions
            .wave_max_time_to_first_token_ms
            .p50,
        llama_cpp_p50_wave_max_time_to_first_token_ms: llama_cpp
            .distributions
            .wave_max_time_to_first_token_ms
            .p50,
        paired_process_blocks,
        macllm_to_llama_cpp_paired_geomean_throughput_ratio: paired_geomean_ratio,
    })
}

fn process_phase_throughput(
    run: &FairPrefixImplementationRun,
    process_sequence_index: usize,
    expected_waves: usize,
) -> Result<f64> {
    phase_throughput(
        &run.waves,
        &run.implementation,
        process_sequence_index,
        expected_waves,
    )
}

fn phase_throughput(
    all_waves: &[FairPrefixWaveEvidence],
    implementation: &str,
    process_sequence_index: usize,
    expected_waves: usize,
) -> Result<f64> {
    let waves = all_waves
        .iter()
        .filter(|wave| wave.process_sequence_index == process_sequence_index)
        .collect::<Vec<_>>();
    ensure!(
        waves.len() == expected_waves,
        "{} process sequence {} retained {} waves; expected {}",
        implementation,
        process_sequence_index,
        waves.len(),
        expected_waves
    );
    let total_tokens = waves.iter().try_fold(0_u64, |total, wave| {
        total
            .checked_add(wave.total_completion_tokens)
            .context("fair-prefix phase token count overflow")
    })?;
    let total_wall_time_ms = waves
        .iter()
        .map(|wave| wave.external_wall_time_ms)
        .sum::<f64>();
    ensure!(
        total_tokens > 0 && positive_finite(total_wall_time_ms),
        "fair-prefix phase requires positive tokens and wall time"
    );
    let throughput = total_tokens as f64 * 1_000.0 / total_wall_time_ms;
    ensure!(
        positive_finite(throughput),
        "fair-prefix phase throughput must be positive"
    );
    Ok(throughput)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndpointKind {
    LlamaCpp,
    Macllm,
}

impl EndpointKind {
    const fn label(self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama.cpp",
            Self::Macllm => "macllm",
        }
    }

    const fn health_path(self) -> &'static str {
        match self {
            Self::LlamaCpp => "/health",
            Self::Macllm => "/healthz",
        }
    }
}

struct EndpointWorkloadEvidence {
    waves: Vec<FairPrefixWaveEvidence>,
    replay: FairPrefixReplayEvidence,
    seed_requests: usize,
    restore_requests: usize,
    post_seed_stabilization: Duration,
    warmup_waves: usize,
    warmup_aggregate_output_tokens_per_second_samples: Vec<f64>,
}

#[allow(clippy::too_many_arguments)] // Mirrors the Mac endpoint's explicit isolated-phase boundary.
fn run_llama_endpoint(
    verified: &VerifiedScoreboard,
    command: &BenchmarkCommand,
    address: SocketAddr,
    specs: &FairPromptSet,
    contract: &FairPrefixContract,
    process_sequence_index: usize,
    wave_index_start: usize,
    wave_count: usize,
    executable_guard: &mut FileProvenanceGuard,
    model_guard: &mut FileProvenanceGuard,
) -> Result<FairPrefixImplementationRun> {
    let argv = command.report_argv()?;
    executable_guard.verify_path()?;
    model_guard.verify_path()?;
    let mut child = ChildGuard::spawn(command)?;
    executable_guard.verify_path()?;
    model_guard.verify_path()?;
    let run_result = (|| {
        let health_wait = child.wait_until_healthy(address, EndpointKind::LlamaCpp)?;
        model_guard.verify_path()?;
        verify_llama_tokenizer(address, specs)?;
        let evidence = run_endpoint_workload(
            address,
            specs,
            contract,
            EndpointKind::LlamaCpp,
            process_sequence_index,
            wave_index_start,
            wave_count,
        )?;
        child.ensure_running()?;
        executable_guard.verify_path()?;
        model_guard.verify_path()?;
        let distributions = build_distributions(&evidence.waves)?;
        Ok::<_, anyhow::Error>((health_wait, evidence, distributions))
    })();
    let stop_result = child.stop();
    let provenance_result = executable_guard.verify_path();
    let model_provenance_result = model_guard.verify_path();
    let close_result = wait_until_port_closed(address);
    let (health_wait, evidence, distributions) = finish_with_cleanup(
        run_result,
        vec![
            ("llama.cpp shutdown", stop_result),
            ("llama.cpp executable provenance", provenance_result),
            ("model-file provenance", model_provenance_result),
            ("port-close verification", close_result),
        ],
    )?;
    ensure!(
        verified.llama_server.canonical_path == command.program,
        "fair-prefix llama command did not use the verified server binary"
    );
    Ok(FairPrefixImplementationRun {
        implementation: EndpointKind::LlamaCpp.label().to_owned(),
        process_command: argv,
        isolated_os_process: true,
        base_url: format!("http://{address}"),
        process_sequence_indices: vec![process_sequence_index],
        health_wait_ms_samples: vec![millis(health_wait)],
        post_seed_stabilization_ms_samples: vec![millis(evidence.post_seed_stabilization)],
        warmup_waves_per_lifetime: evidence.warmup_waves,
        warmup_aggregate_output_tokens_per_second_samples: evidence
            .warmup_aggregate_output_tokens_per_second_samples,
        runtime_attestations: Vec::new(),
        measurement_boundary: format!(
            "{FAIR_TRANSPORT}; one phase of a paired AB/BA sequence using an isolated pinned \
             llama-server process; all eight requests cross \
             one client barrier; shared wall time begins at the earliest actual request \
             submission and ends at the last parsed terminal SSE event; health, tokenization, seeding, identical \
             eight-request rotated-task conditioning after every warm-up/measured wave, five warm-up waves, \
             and non-stream correctness replay are excluded"
        ),
        tokenizer_exact: true,
        cache_prompt_publication: true,
        seed_requests: evidence.seed_requests,
        restore_requests: evidence.restore_requests,
        conditioning_prompt_token_ids_sha256: specs.conditioning_prompt_token_ids_sha256(),
        conditioning_lcp_tokens: specs.conditioning_lcp_tokens.clone(),
        waves: evidence.waves,
        distributions,
        lifetime_replays: vec![evidence.replay.clone()],
        replay: evidence.replay,
        port_closed_after_shutdown: true,
    })
}

#[allow(clippy::too_many_arguments)] // Mirrors the llama endpoint's explicit isolated-phase boundary.
fn run_macllm_endpoint(
    command: &BenchmarkCommand,
    address: SocketAddr,
    profile: &str,
    specs: &FairPromptSet,
    contract: &FairPrefixContract,
    process_sequence_index: usize,
    wave_index_start: usize,
    wave_count: usize,
    model_guard: &mut FileProvenanceGuard,
) -> Result<FairPrefixImplementationRun> {
    ensure_port_is_closed(address)?;
    let argv = command.report_argv()?;
    model_guard.verify_path()?;
    let mut child = ChildGuard::spawn(command)?;
    model_guard.verify_path()?;
    let run_result = (|| {
        let health_wait = child.wait_until_healthy(address, EndpointKind::Macllm)?;
        model_guard.verify_path()?;
        let runtime_attestation = read_macllm_runtime_attestation(address, profile, contract)?;
        let evidence = run_endpoint_workload(
            address,
            specs,
            contract,
            EndpointKind::Macllm,
            process_sequence_index,
            wave_index_start,
            wave_count,
        )?;
        child.ensure_running()?;
        model_guard.verify_path()?;
        let distributions = build_distributions(&evidence.waves)?;
        Ok::<_, anyhow::Error>((health_wait, runtime_attestation, evidence, distributions))
    })();
    let stop_result = child.stop();
    let model_provenance_result = model_guard.verify_path();
    let close_result = wait_until_port_closed(address);
    let (health_wait, runtime_attestation, evidence, distributions) = finish_with_cleanup(
        run_result,
        vec![
            ("MacLLM process shutdown", stop_result),
            ("model-file provenance", model_provenance_result),
            ("port-close verification", close_result),
        ],
    )?;

    Ok(FairPrefixImplementationRun {
        implementation: EndpointKind::Macllm.label().to_owned(),
        process_command: argv,
        isolated_os_process: true,
        base_url: format!("http://{address}"),
        process_sequence_indices: vec![process_sequence_index],
        health_wait_ms_samples: vec![millis(health_wait)],
        post_seed_stabilization_ms_samples: vec![millis(evidence.post_seed_stabilization)],
        warmup_waves_per_lifetime: evidence.warmup_waves,
        warmup_aggregate_output_tokens_per_second_samples: evidence
            .warmup_aggregate_output_tokens_per_second_samples,
        runtime_attestations: vec![runtime_attestation],
        measurement_boundary: format!(
            "{FAIR_TRANSPORT}; one phase of a paired AB/BA sequence using an isolated pinned \
             MacLLM serve process; all \
             eight requests cross the same client barrier used for llama.cpp; shared wall time \
             begins at the earliest actual request submission and ends at the last parsed terminal SSE \
             event; model load, health, seeding, identical eight-request rotated-task conditioning \
             after every warm-up/measured wave, five warm-up waves, and non-stream correctness replay are excluded"
        ),
        tokenizer_exact: true,
        cache_prompt_publication: true,
        seed_requests: evidence.seed_requests,
        restore_requests: evidence.restore_requests,
        conditioning_prompt_token_ids_sha256: specs.conditioning_prompt_token_ids_sha256(),
        conditioning_lcp_tokens: specs.conditioning_lcp_tokens.clone(),
        waves: evidence.waves,
        distributions,
        lifetime_replays: vec![evidence.replay.clone()],
        replay: evidence.replay,
        port_closed_after_shutdown: true,
    })
}

fn run_llama_attestation_endpoint(
    command: &BenchmarkCommand,
    address: SocketAddr,
    contract: &FairPrefixContract,
    executable_guard: &mut FileProvenanceGuard,
    model_guard: &mut FileProvenanceGuard,
) -> Result<FairPrefixRuntimeAttestation> {
    ensure_port_is_closed(address)?;
    let argv = command.report_argv()?;
    executable_guard.verify_path()?;
    model_guard.verify_path()?;
    let mut child = ChildGuard::spawn(command)?;
    executable_guard.verify_path()?;
    model_guard.verify_path()?;
    let run_result = (|| {
        child.wait_until_healthy(address, EndpointKind::LlamaCpp)?;
        model_guard.verify_path()?;
        let attestation = wait_for_llama_runtime_attestation(&mut child, contract, &argv)?;
        child.ensure_running()?;
        executable_guard.verify_path()?;
        model_guard.verify_path()?;
        Ok::<_, anyhow::Error>(attestation)
    })();
    let stop_result = child.stop();
    let executable_result = executable_guard.verify_path();
    let model_result = model_guard.verify_path();
    let close_result = wait_until_port_closed(address);
    finish_with_cleanup(
        run_result,
        vec![
            ("llama.cpp attestation shutdown", stop_result),
            (
                "llama.cpp attestation executable provenance",
                executable_result,
            ),
            ("llama.cpp attestation model provenance", model_result),
            ("attestation port-close verification", close_result),
        ],
    )
}

fn wait_for_llama_runtime_attestation(
    child: &mut ChildGuard,
    contract: &FairPrefixContract,
    process_command: &[String],
) -> Result<FairPrefixRuntimeAttestation> {
    let started_at = Instant::now();
    loop {
        let stderr = child.stderr_text()?;
        let parse_error =
            match read_llama_cpp_runtime_attestation(&stderr, contract, process_command) {
                Ok(attestation) => return Ok(attestation),
                Err(error) => error,
            };
        child.ensure_running()?;
        ensure!(
            started_at.elapsed() < HEALTH_TIMEOUT,
            "llama.cpp startup attestation did not become complete within {} seconds: {}",
            HEALTH_TIMEOUT.as_secs(),
            format_args!("{parse_error:#}")
        );
        thread::sleep(HEALTH_POLL_INTERVAL);
    }
}

#[allow(clippy::too_many_lines)] // Phase identity and evidence concatenation remain one auditable boundary.
fn combine_implementation_runs(
    kind: EndpointKind,
    phases: Vec<FairPrefixImplementationRun>,
    contract: &FairPrefixContract,
) -> Result<FairPrefixImplementationRun> {
    ensure!(
        phases.len() == FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION,
        "{} requires exactly {} process lifetimes",
        kind.label(),
        FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
    );
    let first = phases
        .first()
        .context("fair-prefix implementation retained no process phases")?;
    let expected_phase_attestations = usize::from(kind == EndpointKind::Macllm);
    ensure!(
        phases.iter().all(|phase| {
            phase.implementation == kind.label()
                && phase.process_command == first.process_command
                && phase.isolated_os_process
                && phase.base_url == first.base_url
                && phase.tokenizer_exact
                && phase.cache_prompt_publication
                && phase.conditioning_prompt_token_ids_sha256
                    == first.conditioning_prompt_token_ids_sha256
                && phase.conditioning_lcp_tokens == first.conditioning_lcp_tokens
                && phase.lifetime_replays.len() == 1
                && phase.lifetime_replays.first() == Some(&phase.replay)
                && phase.replay == first.replay
                && phase.port_closed_after_shutdown
                && phase.warmup_waves_per_lifetime == FAIR_WARMUP_WAVES_PER_LIFETIME
                && phase
                    .warmup_aggregate_output_tokens_per_second_samples
                    .len()
                    == FAIR_WARMUP_WAVES_PER_LIFETIME
                && phase.runtime_attestations.len() == expected_phase_attestations
        }),
        "{} process lifetimes differ in identity, replay, warm-up, or cleanup",
        kind.label()
    );
    let process_sequence_indices = phases
        .iter()
        .flat_map(|phase| phase.process_sequence_indices.iter().copied())
        .collect::<Vec<_>>();
    let health_wait_ms_samples = phases
        .iter()
        .flat_map(|phase| phase.health_wait_ms_samples.iter().copied())
        .collect::<Vec<_>>();
    let post_seed_stabilization_ms_samples = phases
        .iter()
        .flat_map(|phase| phase.post_seed_stabilization_ms_samples.iter().copied())
        .collect::<Vec<_>>();
    let warmup_aggregate_output_tokens_per_second_samples = phases
        .iter()
        .flat_map(|phase| {
            phase
                .warmup_aggregate_output_tokens_per_second_samples
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    let runtime_attestations = phases
        .iter()
        .flat_map(|phase| phase.runtime_attestations.iter().cloned())
        .collect::<Vec<_>>();
    let seed_requests = phases.iter().try_fold(0_usize, |total, phase| {
        total
            .checked_add(phase.seed_requests)
            .context("fair-prefix seed request count overflow")
    })?;
    let restore_requests = phases.iter().try_fold(0_usize, |total, phase| {
        total
            .checked_add(phase.restore_requests)
            .context("fair-prefix restore request count overflow")
    })?;
    let mut waves = phases
        .iter()
        .flat_map(|phase| phase.waves.iter().cloned())
        .collect::<Vec<_>>();
    waves.sort_unstable_by_key(|wave| wave.wave_index);
    ensure!(
        waves.len() == contract.measured_waves_per_implementation,
        "{} combined process lifetimes retained the wrong number of waves",
        kind.label()
    );
    let distributions = build_distributions(&waves)?;
    let lifetime_replays = phases
        .iter()
        .map(|phase| phase.replay.clone())
        .collect::<Vec<_>>();
    Ok(FairPrefixImplementationRun {
        implementation: kind.label().to_owned(),
        process_command: first.process_command.clone(),
        isolated_os_process: true,
        base_url: first.base_url.clone(),
        process_sequence_indices,
        health_wait_ms_samples,
        post_seed_stabilization_ms_samples,
        warmup_waves_per_lifetime: FAIR_WARMUP_WAVES_PER_LIFETIME,
        warmup_aggregate_output_tokens_per_second_samples,
        runtime_attestations,
        measurement_boundary: format!(
            "{FAIR_TRANSPORT}; four isolated process lifetimes in four paired AB/BA blocks; \
             each lifetime excludes model load, health, cache seeding, five identical warm-up \
             waves that must pass the fixed CV/trend gate, identical eight-request rotated-task conditioning after every warm-up/measured wave, and \
             correctness replay; retained waves use one eight-client barrier and time \
             from the earliest actual request submission through the last parsed terminal SSE event"
        ),
        tokenizer_exact: true,
        cache_prompt_publication: true,
        seed_requests,
        restore_requests,
        conditioning_prompt_token_ids_sha256: first.conditioning_prompt_token_ids_sha256.clone(),
        conditioning_lcp_tokens: first.conditioning_lcp_tokens.clone(),
        waves,
        distributions,
        lifetime_replays,
        replay: first.replay.clone(),
        port_closed_after_shutdown: true,
    })
}

fn pin_llama_measurement_device(command: &mut BenchmarkCommand) -> Result<()> {
    let positions = command
        .arguments
        .iter()
        .enumerate()
        .filter_map(|(index, argument)| (argument == "--log-disable").then_some(index))
        .collect::<Vec<_>>();
    ensure!(
        positions.len() == 1,
        "fair-prefix measured llama command must contain exactly one --log-disable"
    );
    ensure!(
        !command
            .arguments
            .iter()
            .any(|argument| argument == "--device" || argument == "-dev"),
        "fair-prefix base llama command unexpectedly pins a device"
    );
    command
        .arguments
        .extend(["--device".to_owned(), "MTL0".to_owned()]);
    Ok(())
}

fn build_llama_attestation_command(measured: &BenchmarkCommand) -> Result<BenchmarkCommand> {
    let mut command = measured.clone();
    replace_log_disable_with_trace(&mut command.arguments)?;
    Ok(command)
}

fn attestation_argv_from_measured(mut measured: Vec<String>) -> Result<Vec<String>> {
    replace_log_disable_with_trace(&mut measured)?;
    Ok(measured)
}

fn replace_log_disable_with_trace(arguments: &mut Vec<String>) -> Result<()> {
    ensure!(
        arguments
            .iter()
            .filter(|argument| argument.as_str() == "--log-disable")
            .count()
            == 1,
        "fair-prefix measured llama command contains duplicate --log-disable"
    );
    let log_disable = arguments
        .iter()
        .position(|argument| argument == "--log-disable")
        .context("fair-prefix measured llama command omitted --log-disable")?;
    arguments.splice(
        log_disable..=log_disable,
        ["-lv".to_owned(), "4".to_owned()],
    );
    Ok(())
}

fn build_fair_macllm_command(
    verified: &VerifiedScoreboard,
    address: SocketAddr,
    contract: &FairPrefixContract,
) -> Result<BenchmarkCommand> {
    // The physical pool must hold the eight active suffixes, but the resident
    // snapshot budget is exactly one 16K endpoint. Both servers therefore run
    // `cache_prompt=true` while longer task-specific publications cannot
    // replace the shared endpoint between retained waves.
    let prefix_cache_capacity_tokens = contract.exact_shared_prefix_tokens;
    let executable =
        std::env::current_exe().context("failed to resolve the fair-prefix MacLLM executable")?;
    let executable = fs::canonicalize(&executable).with_context(|| {
        format!(
            "failed to canonicalize the fair-prefix MacLLM executable {}",
            executable.display()
        )
    })?;
    Ok(BenchmarkCommand {
        program: executable,
        arguments: vec![
            "serve".to_owned(),
            "--model".to_owned(),
            verified.model.file.canonical_path.display().to_string(),
            "--device".to_owned(),
            "metal".to_owned(),
            "--dtype".to_owned(),
            "f32".to_owned(),
            "--kv-cache-block-tokens".to_owned(),
            contract.macllm_prefix_cache_block_tokens.to_string(),
            "--kv-cache-slab-blocks".to_owned(),
            contract.macllm_kv_cache_slab_blocks.to_string(),
            "--kv-cache-capacity-tokens".to_owned(),
            contract.macllm_kv_cache_capacity_tokens.to_string(),
            "--prefix-cache-capacity-tokens".to_owned(),
            prefix_cache_capacity_tokens.to_string(),
            "--host".to_owned(),
            address.ip().to_string(),
            "--port".to_owned(),
            address.port().to_string(),
            "--served-model-name".to_owned(),
            verified.manifest.name.clone(),
            "--queue-capacity".to_owned(),
            MAC_PREFIX_DEMO_CONCURRENCY.to_string(),
            "--max-active-sequences".to_owned(),
            MAC_PREFIX_DEMO_CONCURRENCY.to_string(),
            "--scheduler-token-budget".to_owned(),
            contract.macllm_scheduler_token_budget.to_string(),
            "--prefill-token-quantum".to_owned(),
            contract.macllm_prefill_token_quantum.to_string(),
            "--event-capacity".to_owned(),
            (MAC_PREFIX_DEMO_DECODE_TOKENS + 2).to_string(),
            "--event-send-timeout-seconds".to_owned(),
            REQUEST_TIMEOUT.as_secs().to_string(),
        ],
    })
}

#[allow(clippy::too_many_lines)] // Seed, stabilize, warm, measure, restore, and replay form one phase contract.
fn run_endpoint_workload(
    address: SocketAddr,
    specs: &FairPromptSet,
    contract: &FairPrefixContract,
    kind: EndpointKind,
    process_sequence_index: usize,
    wave_index_start: usize,
    wave_count: usize,
) -> Result<EndpointWorkloadEvidence> {
    ensure!(
        process_sequence_index < FAIR_PROCESS_ORDER.len()
            && FAIR_PROCESS_ORDER[process_sequence_index] == kind.label(),
        "{} was scheduled at the wrong paired process position {}",
        kind.label(),
        process_sequence_index
    );
    ensure!(
        wave_count
            == contract.measured_waves_per_implementation
                / FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION,
        "{} process lifetime retained the wrong wave count",
        kind.label()
    );
    let slot_seeds = specs.slot_seeds();
    let conditioning_prompts = specs.conditioning_prompts();
    for (seed_index, seed) in slot_seeds.iter().enumerate() {
        let observed = run_nonstream_completion(address, seed, 1, contract)?;
        let expected_cached_tokens = match kind {
            EndpointKind::LlamaCpp => 0,
            EndpointKind::Macllm if seed_index == 0 => 0,
            EndpointKind::Macllm => macllm_exact_seed_reuse_tokens(contract)?,
        };
        let expected_evaluated_tokens = seed
            .prompt_tokens
            .checked_sub(expected_cached_tokens)
            .context("fair-prefix seed cache count exceeds its prompt length")?;
        ensure!(
            observed.terminal.cached_prompt_tokens == expected_cached_tokens
                && observed.terminal.evaluated_prompt_tokens == expected_evaluated_tokens
                && observed.terminal.generated_tokens == 1,
            "{} seed for slot {} did not establish the expected exact endpoint",
            kind.label(),
            seed.slot_id
        );
    }

    let stabilization_started_at = Instant::now();
    thread::sleep(FAIR_POST_SEED_STABILIZATION);
    let post_seed_stabilization = stabilization_started_at.elapsed();
    let mut warmup_aggregate_output_tokens_per_second_samples =
        Vec::with_capacity(FAIR_WARMUP_WAVES_PER_LIFETIME);
    for warmup_index in 0..FAIR_WARMUP_WAVES_PER_LIFETIME {
        let warmup = run_streaming_wave(
            address,
            &specs.tasks,
            contract,
            wave_index_start,
            process_sequence_index,
        )?;
        ensure!(
            warmup.exact_cache_hit_requests == contract.concurrency
                && warmup
                    .requests
                    .iter()
                    .all(|request| request.terminal.generated_tokens
                        == contract.completion_tokens_per_request),
            "{} process sequence {} failed excluded exact-cache warm-up wave {}",
            kind.label(),
            process_sequence_index,
            warmup_index
        );
        warmup_aggregate_output_tokens_per_second_samples
            .push(warmup.aggregate_output_tokens_per_second);
        restore_endpoint_slots(address, &conditioning_prompts, contract, kind)?;
    }
    validate_steady_state_samples(
        &warmup_aggregate_output_tokens_per_second_samples,
        contract,
        kind.label(),
        process_sequence_index,
    )?;

    let mut waves = Vec::with_capacity(wave_count);
    for offset in 0..wave_count {
        let wave_index = wave_index_start
            .checked_add(offset)
            .context("fair-prefix wave index overflow")?;
        waves.push(run_streaming_wave(
            address,
            &specs.tasks,
            contract,
            wave_index,
            process_sequence_index,
        )?);
        restore_endpoint_slots(address, &conditioning_prompts, contract, kind)?;
    }

    let replay_requests = run_nonstream_wave(
        address,
        &specs.tasks,
        contract.completion_tokens_per_request,
        contract,
    )?;
    let replay = FairPrefixReplayEvidence {
        prompt_token_ids_sha256: replay_requests
            .iter()
            .map(|request| request.prompt_token_ids_sha256.clone())
            .collect(),
        content: replay_requests
            .iter()
            .map(|request| request.content.clone())
            .collect(),
        content_sha256: replay_requests
            .iter()
            .map(|request| sha256_bytes(request.content.as_bytes()))
            .collect(),
    };
    for wave in &waves {
        for request in &wave.requests {
            ensure!(
                request.streamed_content == replay.content[request.request_index],
                "{} wave {} streamed content differs from its excluded correctness replay for request {}",
                kind.label(),
                wave.wave_index,
                request.request_index
            );
        }
    }
    Ok(EndpointWorkloadEvidence {
        waves,
        replay,
        seed_requests: slot_seeds.len(),
        restore_requests: wave_count
            .checked_add(FAIR_WARMUP_WAVES_PER_LIFETIME)
            .and_then(|restored_waves| restored_waves.checked_mul(slot_seeds.len()))
            .context("fair-prefix restore request count overflow")?,
        post_seed_stabilization,
        warmup_waves: FAIR_WARMUP_WAVES_PER_LIFETIME,
        warmup_aggregate_output_tokens_per_second_samples,
    })
}

fn validate_steady_state_samples(
    samples: &[f64],
    contract: &FairPrefixContract,
    implementation: &str,
    process_sequence_index: usize,
) -> Result<()> {
    ensure!(
        samples.len() == contract.warmup_waves_per_lifetime
            && contract.steady_state_window_waves > 1
            && contract.steady_state_window_waves <= samples.len(),
        "{implementation} process sequence {process_sequence_index} retained the wrong warm-up window"
    );
    let window = &samples[samples.len() - contract.steady_state_window_waves..];
    ensure!(
        window
            .iter()
            .all(|sample| sample.is_finite() && *sample > 0.0),
        "{implementation} process sequence {process_sequence_index} has invalid warm-up throughput"
    );
    let mean = window.iter().sum::<f64>() / window.len() as f64;
    let variance = window
        .iter()
        .map(|sample| {
            let delta = *sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / window.len() as f64;
    let coefficient_of_variation = variance.sqrt() / mean;
    let relative_trend = (window[window.len() - 1] - window[0]).abs() / mean;
    let max_cv = f64::from_bits(contract.steady_state_max_cv_bits);
    let max_relative_trend = f64::from_bits(contract.steady_state_max_relative_trend_bits);
    ensure!(
        coefficient_of_variation <= max_cv && relative_trend <= max_relative_trend,
        "{implementation} process sequence {process_sequence_index} did not reach the pre-registered steady-state gate: CV {coefficient_of_variation:.4} > {max_cv:.4} or relative trend {relative_trend:.4} > {max_relative_trend:.4}"
    );
    Ok(())
}

fn restore_endpoint_slots(
    address: SocketAddr,
    conditioning_prompts: &[FairPromptSpec; MAC_PREFIX_DEMO_CONCURRENCY],
    contract: &FairPrefixContract,
    kind: EndpointKind,
) -> Result<()> {
    for conditioning in conditioning_prompts {
        let observed = run_nonstream_completion(address, conditioning, 1, contract)?;
        let expected_evaluated_tokens = conditioning
            .prompt_tokens
            .checked_sub(contract.exact_shared_prefix_tokens)
            .context("fair-prefix conditioning prompt is shorter than the exact cache endpoint")?;
        ensure!(
            observed.terminal.cached_prompt_tokens == contract.exact_shared_prefix_tokens
                && observed.terminal.evaluated_prompt_tokens == expected_evaluated_tokens
                && observed.terminal.generated_tokens == 1,
            "{} slot {} did not restore the exact seed endpoint",
            kind.label(),
            conditioning.slot_id
        );
    }
    Ok(())
}

fn macllm_exact_seed_reuse_tokens(contract: &FairPrefixContract) -> Result<usize> {
    contract
        .exact_shared_prefix_tokens
        .checked_sub(contract.macllm_prefix_cache_block_tokens)
        .context("MacLLM exact seed is shorter than one prefix-cache block")
}

fn finish_with_cleanup<T>(run: Result<T>, cleanup: Vec<(&'static str, Result<()>)>) -> Result<T> {
    let mut failures = Vec::new();
    let value = match run {
        Ok(value) => Some(value),
        Err(error) => {
            failures.push(format!("run failed: {error:#}"));
            None
        }
    };
    for (label, result) in cleanup {
        if let Err(error) = result {
            failures.push(format!("{label} failed: {error:#}"));
        }
    }
    if !failures.is_empty() {
        bail!("fair-prefix execution failed: {}", failures.join("; "));
    }
    value.context("successful fair-prefix execution retained no evidence")
}

#[derive(Debug, Serialize)]
struct FairCompletionRequest<'a> {
    prompt: &'a str,
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
    response_fields: &'static [&'static str],
}

fn completion_body(
    spec: &FairPromptSpec,
    completion_tokens: usize,
    stream: bool,
    contract: &FairPrefixContract,
) -> Result<Vec<u8>> {
    serde_json::to_vec(&FairCompletionRequest {
        prompt: &spec.prompt,
        n_predict: completion_tokens,
        temperature: f64::from_bits(contract.temperature_bits),
        top_k: contract.top_k,
        top_p: f64::from_bits(contract.top_p_bits),
        seed: contract.seed,
        cache_prompt: contract.cache_prompt,
        stream,
        return_tokens: contract.return_tokens,
        id_slot: spec.slot_id,
        ignore_eos: contract.ignore_eos,
        repeat_penalty: f64::from_bits(contract.repetition_penalty_bits),
        response_fields: &RAW_COMPLETION_RESPONSE_FIELDS,
    })
    .context("failed to serialize fair-prefix completion request")
}

#[derive(Debug)]
struct ObservedFairCompletion {
    request_index: usize,
    prompt_token_ids_sha256: String,
    content: String,
    token_ids: Vec<u32>,
    submitted_at: Instant,
    first_token_at: Instant,
    completed_at: Instant,
    time_to_first_token: Duration,
    latency: Duration,
    terminal: FairPrefixTerminalEvidence,
}

// Keeping submission, first-token, and completion timestamps together makes
// the reconstructed concurrent-wave wall clock independently auditable.
#[allow(clippy::too_many_lines)]
fn run_streaming_wave(
    address: SocketAddr,
    specs: &[FairPromptSpec; MAC_PREFIX_DEMO_CONCURRENCY],
    contract: &FairPrefixContract,
    wave_index: usize,
    process_sequence_index: usize,
) -> Result<FairPrefixWaveEvidence> {
    let prepared = specs
        .iter()
        .cloned()
        .enumerate()
        .map(|(request_index, spec)| {
            let body = completion_body(
                &spec,
                contract.completion_tokens_per_request,
                true,
                contract,
            )?;
            Ok((request_index, spec, body))
        })
        .collect::<Result<Vec<_>>>()?;
    let start_gate = Arc::new(ThreadStartGate::new());
    let mut workers = Vec::with_capacity(MAC_PREFIX_DEMO_CONCURRENCY);
    for (request_index, spec, body) in prepared {
        let worker_gate = Arc::clone(&start_gate);
        let contract = contract.clone();
        let worker = thread::Builder::new()
            .name(format!("fair-stream-{request_index}"))
            .spawn(move || {
                worker_gate.wait()?;
                run_streaming_completion(address, &spec, request_index, &body, &contract)
            });
        match worker {
            Ok(worker) => workers.push(worker),
            Err(error) => {
                start_gate.cancel();
                let _ignored = join_completion_workers(workers);
                return Err(error).context("failed to spawn fair-prefix streaming worker");
            }
        }
    }
    start_gate.release();
    let mut observed = join_completion_workers(workers)?;
    observed.sort_unstable_by_key(|request| request.request_index);
    ensure!(
        observed.len() == contract.concurrency,
        "fair-prefix streaming wave lost one or more requests"
    );
    let completed_at = observed
        .iter()
        .map(|request| request.completed_at)
        .max()
        .context("fair-prefix streaming wave retained no terminal time")?;
    let submitted_at = observed
        .iter()
        .map(|request| request.submitted_at)
        .min()
        .context("fair-prefix streaming wave retained no submission time")?;
    let wall_time = completed_at
        .checked_duration_since(submitted_at)
        .context("fair-prefix wave completed before its earliest request submission")?;
    ensure!(
        !wall_time.is_zero(),
        "fair-prefix external wall time must be positive"
    );
    let total_completion_tokens = observed.iter().try_fold(0_u64, |total, request| {
        total
            .checked_add(u64::try_from(request.terminal.generated_tokens)?)
            .context("fair-prefix completion total overflow")
    })?;
    let exact_cache_hit_requests = observed
        .iter()
        .filter(|request| {
            request.terminal.cached_prompt_tokens == contract.exact_shared_prefix_tokens
        })
        .count();
    let requests = observed
        .into_iter()
        .zip(specs)
        .map(|(observed, spec)| {
            let submitted_offset = observed
                .submitted_at
                .checked_duration_since(submitted_at)
                .context("fair-prefix request submission preceded wave origin")?;
            let first_token_offset = observed
                .first_token_at
                .checked_duration_since(submitted_at)
                .context("fair-prefix first token preceded wave origin")?;
            let completed_offset = observed
                .completed_at
                .checked_duration_since(submitted_at)
                .context("fair-prefix completion preceded wave origin")?;
            Ok(FairPrefixRequestEvidence {
                request_index: observed.request_index,
                slot_id: spec.slot_id,
                prompt_tokens: spec.prompt_tokens,
                prompt_token_ids_sha256: observed.prompt_token_ids_sha256,
                streamed_content: observed.content,
                streamed_token_ids: observed.token_ids,
                submitted_offset_ms: millis(submitted_offset),
                first_token_offset_ms: millis(first_token_offset),
                completed_offset_ms: millis(completed_offset),
                external_time_to_first_token_ms: millis(observed.time_to_first_token),
                external_latency_ms: millis(observed.latency),
                terminal: observed.terminal,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(FairPrefixWaveEvidence {
        wave_index,
        process_sequence_index,
        request_count: contract.concurrency,
        total_completion_tokens,
        external_wall_time_ms: millis(wall_time),
        aggregate_output_tokens_per_second: aggregate_throughput(
            total_completion_tokens,
            wall_time,
        )?,
        exact_cache_hit_requests,
        requests,
    })
}

fn join_completion_workers(
    workers: Vec<thread::JoinHandle<Result<ObservedFairCompletion>>>,
) -> Result<Vec<ObservedFairCompletion>> {
    let mut observed = Vec::with_capacity(workers.len());
    let mut first_error = None;
    for worker in workers {
        let result = worker
            .join()
            .map_err(|_| anyhow::anyhow!("fair-prefix HTTP worker panicked"))
            .and_then(|result| result);
        match result {
            Ok(request) => observed.push(request),
            Err(error) if first_error.is_none() => first_error = Some(error),
            Err(_error) => {}
        }
    }
    if let Some(error) = first_error {
        return Err(error);
    }
    Ok(observed)
}

#[allow(clippy::too_many_lines)] // Keep the external HTTP/SSE timing boundary source-auditable.
fn run_streaming_completion(
    address: SocketAddr,
    spec: &FairPromptSpec,
    request_index: usize,
    body: &[u8],
    contract: &FairPrefixContract,
) -> Result<ObservedFairCompletion> {
    let submitted_at = Instant::now();
    let deadline = submitted_at
        .checked_add(REQUEST_TIMEOUT)
        .context("fair-prefix request deadline overflow")?;
    let stream = DeadlineStream::connect_until(address, deadline)?;
    let mut reader = BufReader::new(stream);
    write_request(
        reader.get_mut(),
        address,
        "POST",
        "/completion",
        Some(body),
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
    let mut content = String::new();
    let mut token_ids = Vec::with_capacity(contract.completion_tokens_per_request);
    let mut first_token_at = None;
    let mut last_n_decoded = 0;
    loop {
        let chunk = read_http_chunk(&mut reader)?
            .with_context(|| format!("{} stream ended before terminal evidence", spec.label))?;
        let arrived_at = Instant::now();
        ensure!(
            arrived_at <= deadline,
            "{} exceeded the absolute request deadline",
            spec.label
        );
        for data in decoder.push(&chunk)? {
            ensure!(
                data != "[DONE]",
                "{} ended before terminal evidence",
                spec.label
            );
            let event = match serde_json::from_str::<WireSsePayload>(&data)
                .with_context(|| format!("{} emitted invalid SSE JSON", spec.label))?
            {
                WireSsePayload::Completion(event) => event,
                WireSsePayload::Error(response) => {
                    bail!(
                        "{} stream failed ({}{}): {}",
                        spec.label,
                        response.error.error_type,
                        response
                            .error
                            .code
                            .as_deref()
                            .map_or_else(String::new, |code| format!("/{code}")),
                        response.error.message
                    );
                }
            };
            ensure!(event.index == 0, "{} returned a non-zero index", spec.label);
            if !event.stop {
                let (n_decoded, token_id) = validate_wire_fragment(
                    &event,
                    spec,
                    last_n_decoded,
                    contract.completion_tokens_per_request,
                )?;
                last_n_decoded = n_decoded;
                token_ids.push(token_id);
            }
            if !event.content.is_empty() {
                first_token_at.get_or_insert(arrived_at);
                content.push_str(&event.content);
            }
            if event.stop {
                let terminal = validate_wire_terminal(&event, spec, contract, true)?;
                ensure!(
                    last_n_decoded == terminal.generated_tokens
                        && token_ids.len() == terminal.generated_tokens,
                    "{} streamed token ordinals did not reconcile with terminal output count",
                    spec.label
                );
                let completed_at = Instant::now();
                let latency = completed_at
                    .checked_duration_since(submitted_at)
                    .context("fair-prefix completion preceded request submission")?;
                let first_token_at = first_token_at.with_context(|| {
                    format!("{} produced no visible streamed content", spec.label)
                })?;
                let time_to_first_token = first_token_at
                    .checked_duration_since(submitted_at)
                    .with_context(|| {
                        format!("{} first token preceded request submission", spec.label)
                    })?;
                ensure!(
                    latency >= time_to_first_token,
                    "{} latency is shorter than TTFT",
                    spec.label
                );
                return Ok(ObservedFairCompletion {
                    request_index,
                    prompt_token_ids_sha256: spec.prompt_token_ids_sha256.clone(),
                    content,
                    token_ids,
                    submitted_at,
                    first_token_at,
                    completed_at,
                    time_to_first_token,
                    latency,
                    terminal,
                });
            }
        }
    }
}

fn run_nonstream_wave(
    address: SocketAddr,
    specs: &[FairPromptSpec; MAC_PREFIX_DEMO_CONCURRENCY],
    completion_tokens: usize,
    contract: &FairPrefixContract,
) -> Result<Vec<ObservedFairCompletion>> {
    let prepared = specs
        .iter()
        .cloned()
        .enumerate()
        .map(|(request_index, spec)| {
            let body = completion_body(&spec, completion_tokens, false, contract)?;
            Ok((request_index, spec, body))
        })
        .collect::<Result<Vec<_>>>()?;
    let start_gate = Arc::new(ThreadStartGate::new());
    let mut workers = Vec::with_capacity(MAC_PREFIX_DEMO_CONCURRENCY);
    for (request_index, spec, body) in prepared {
        let worker_gate = Arc::clone(&start_gate);
        let contract = contract.clone();
        let worker = thread::Builder::new()
            .name(format!("fair-replay-{request_index}"))
            .spawn(move || {
                worker_gate.wait()?;
                let mut observed = run_nonstream_completion_with_body(
                    address,
                    &spec,
                    completion_tokens,
                    &contract,
                    &body,
                )?;
                observed.request_index = request_index;
                Ok(observed)
            });
        match worker {
            Ok(worker) => workers.push(worker),
            Err(error) => {
                start_gate.cancel();
                let _ignored = join_completion_workers(workers);
                return Err(error).context("failed to spawn fair-prefix replay worker");
            }
        }
    }
    start_gate.release();
    let mut observed = join_completion_workers(workers)?;
    observed.sort_unstable_by_key(|request| request.request_index);
    ensure!(
        observed.len() == contract.concurrency,
        "fair-prefix non-stream replay lost one or more requests"
    );
    Ok(observed)
}

fn run_nonstream_completion(
    address: SocketAddr,
    spec: &FairPromptSpec,
    completion_tokens: usize,
    contract: &FairPrefixContract,
) -> Result<ObservedFairCompletion> {
    let body = completion_body(spec, completion_tokens, false, contract)?;
    run_nonstream_completion_with_body(address, spec, completion_tokens, contract, &body)
}

fn run_nonstream_completion_with_body(
    address: SocketAddr,
    spec: &FairPromptSpec,
    completion_tokens: usize,
    contract: &FairPrefixContract,
    body: &[u8],
) -> Result<ObservedFairCompletion> {
    let submitted_at = Instant::now();
    let response = buffered_request(address, "POST", "/completion", Some(body), REQUEST_TIMEOUT)?;
    ensure!(
        response.status == 200,
        "{} non-stream /completion returned HTTP {}: {}",
        spec.label,
        response.status,
        String::from_utf8_lossy(&response.body)
    );
    let event: WireCompletionEvent = serde_json::from_slice(&response.body)
        .with_context(|| format!("{} emitted invalid non-stream JSON", spec.label))?;
    let terminal =
        validate_wire_terminal_for_tokens(&event, spec, contract, false, completion_tokens)?;
    let completed_at = Instant::now();
    let latency = completed_at
        .checked_duration_since(submitted_at)
        .context("fair-prefix non-stream completion preceded submission")?;
    Ok(ObservedFairCompletion {
        request_index: spec.slot_id,
        prompt_token_ids_sha256: spec.prompt_token_ids_sha256.clone(),
        content: event.content,
        token_ids: Vec::new(),
        submitted_at,
        first_token_at: completed_at,
        completed_at,
        time_to_first_token: latency,
        latency,
        terminal,
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCompletionEvent {
    index: usize,
    #[serde(default)]
    content: String,
    #[serde(default)]
    tokens: Option<Vec<u32>>,
    #[serde(default)]
    stop: bool,
    #[serde(default)]
    id_slot: Option<isize>,
    #[serde(default)]
    tokens_predicted: Option<usize>,
    #[serde(default)]
    tokens_evaluated: Option<usize>,
    #[serde(default)]
    truncated: Option<bool>,
    #[serde(default)]
    stop_type: Option<String>,
    #[serde(default, rename = "generation_settings/seed")]
    generation_seed: Option<u64>,
    #[serde(default, rename = "generation_settings/temperature")]
    generation_temperature: Option<f64>,
    #[serde(default, rename = "generation_settings/top_k")]
    generation_top_k: Option<usize>,
    #[serde(default, rename = "generation_settings/top_p")]
    generation_top_p: Option<f64>,
    #[serde(default, rename = "generation_settings/repeat_penalty")]
    generation_repeat_penalty: Option<f64>,
    #[serde(default, rename = "generation_settings/ignore_eos")]
    generation_ignore_eos: Option<bool>,
    #[serde(default, rename = "generation_settings/stream")]
    generation_stream: Option<bool>,
    #[serde(default, rename = "generation_settings/max_tokens")]
    generation_max_tokens: Option<usize>,
    #[serde(default, rename = "generation_settings/n_predict")]
    generation_n_predict: Option<usize>,
    #[serde(default, rename = "generation_settings/n_keep")]
    generation_n_keep: Option<usize>,
    #[serde(default, rename = "timings/cache_n")]
    timing_cache_n: Option<usize>,
    #[serde(default, rename = "timings/prompt_n")]
    timing_prompt_n: Option<usize>,
    #[serde(default, rename = "timings/prompt_ms")]
    timing_prompt_ms: Option<f64>,
    #[serde(default, rename = "timings/predicted_n")]
    timing_predicted_n: Option<usize>,
    #[serde(default, rename = "timings/predicted_ms")]
    timing_predicted_ms: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WireSsePayload {
    Completion(Box<WireCompletionEvent>),
    Error(WireErrorResponse),
}

#[derive(Debug, Deserialize)]
struct WireErrorResponse {
    error: WireError,
}

#[derive(Debug, Deserialize)]
struct WireError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    #[serde(default)]
    code: Option<String>,
}

fn validate_wire_fragment(
    event: &WireCompletionEvent,
    spec: &FairPromptSpec,
    last_n_decoded: usize,
    completion_tokens: usize,
) -> Result<(usize, u32)> {
    let tokens = event
        .tokens
        .as_deref()
        .context("streamed fragment omitted sampled token ID")?;
    let n_decoded = event
        .tokens_predicted
        .context("streamed fragment omitted tokens_predicted")?;
    let expected_n_decoded = last_n_decoded
        .checked_add(1)
        .context("streamed token ordinal overflowed")?;
    ensure!(
        event.index == 0
            && !event.stop
            && tokens.len() == 1
            && event.id_slot == Some(-1)
            && n_decoded == expected_n_decoded
            && n_decoded <= completion_tokens
            && event.tokens_evaluated == Some(spec.prompt_tokens)
            && event.truncated.is_none()
            && event.stop_type.is_none()
            && event.generation_seed.is_none()
            && event.generation_temperature.is_none()
            && event.generation_top_k.is_none()
            && event.generation_top_p.is_none()
            && event.generation_repeat_penalty.is_none()
            && event.generation_ignore_eos.is_none()
            && event.generation_stream.is_none()
            && event.generation_max_tokens.is_none()
            && event.generation_n_predict.is_none()
            && event.generation_n_keep.is_none()
            && event.timing_cache_n.is_none()
            && event.timing_prompt_n.is_none()
            && event.timing_prompt_ms.is_none()
            && event.timing_predicted_n.is_none()
            && event.timing_predicted_ms.is_none(),
        "{} streamed fragment differs from the shared projected response schema",
        spec.label
    );
    Ok((n_decoded, tokens[0]))
}

fn validate_wire_terminal(
    event: &WireCompletionEvent,
    spec: &FairPromptSpec,
    contract: &FairPrefixContract,
    expected_stream: bool,
) -> Result<FairPrefixTerminalEvidence> {
    validate_wire_terminal_for_tokens(
        event,
        spec,
        contract,
        expected_stream,
        contract.completion_tokens_per_request,
    )
}

fn validate_wire_terminal_for_tokens(
    event: &WireCompletionEvent,
    spec: &FairPromptSpec,
    contract: &FairPrefixContract,
    expected_stream: bool,
    completion_tokens: usize,
) -> Result<FairPrefixTerminalEvidence> {
    ensure!(event.index == 0, "{} returned a non-zero index", spec.label);
    ensure!(event.stop, "{} omitted terminal stop=true", spec.label);
    ensure!(
        event.tokens.is_none() && event.id_slot.is_none(),
        "{} terminal response escaped the shared response_fields projection",
        spec.label
    );
    ensure!(
        event.tokens_predicted == Some(completion_tokens)
            && event.tokens_evaluated == Some(spec.prompt_tokens)
            && event.truncated == Some(false)
            && event.stop_type.as_deref() == Some("limit"),
        "{} terminal token accounting or stop boundary changed",
        spec.label
    );
    ensure!(
        event.generation_seed == Some(contract.seed)
            && event
                .generation_temperature
                .is_some_and(|value| value.to_bits() == contract.temperature_bits)
            && event.generation_top_k == Some(contract.top_k)
            && event
                .generation_top_p
                .is_some_and(|value| value.to_bits() == contract.top_p_bits)
            && event
                .generation_repeat_penalty
                .is_some_and(|value| value.to_bits() == contract.repetition_penalty_bits)
            && event.generation_ignore_eos == Some(contract.ignore_eos)
            && event.generation_stream == Some(expected_stream)
            && event.generation_max_tokens == Some(completion_tokens)
            && event.generation_n_predict == Some(completion_tokens)
            && event.generation_n_keep == Some(0),
        "{} terminal generation settings differ from the shared contract",
        spec.label
    );
    ensure!(
        event
            .timing_cache_n
            .zip(event.timing_prompt_n)
            .and_then(|(cache, prompt)| cache.checked_add(prompt))
            .is_some_and(|tokens| tokens == spec.prompt_tokens)
            && event.timing_predicted_n == Some(completion_tokens)
            && event
                .timing_prompt_ms
                .is_some_and(|milliseconds| milliseconds.is_finite() && milliseconds >= 0.0)
            && event
                .timing_predicted_ms
                .is_some_and(|milliseconds| milliseconds.is_finite() && milliseconds >= 0.0),
        "{} terminal timings violate prompt or completion accounting",
        spec.label
    );
    Ok(FairPrefixTerminalEvidence {
        cached_prompt_tokens: event
            .timing_cache_n
            .context("validated terminal omitted timings/cache_n")?,
        evaluated_prompt_tokens: event
            .timing_prompt_n
            .context("validated terminal omitted timings/prompt_n")?,
        generated_tokens: event
            .timing_predicted_n
            .context("validated terminal omitted timings/predicted_n")?,
        server_prompt_ms: event
            .timing_prompt_ms
            .context("validated terminal omitted timings/prompt_ms")?,
        server_predicted_ms: event
            .timing_predicted_ms
            .context("validated terminal omitted timings/predicted_ms")?,
    })
}

#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: String,
    #[serde(default)]
    worker: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    dtype: Option<String>,
}

fn health_is_ready(address: SocketAddr, kind: EndpointKind) -> Result<bool> {
    let response = buffered_request(
        address,
        "GET",
        kind.health_path(),
        None,
        Duration::from_secs(1),
    )?;
    if response.status != 200 {
        return Ok(false);
    }
    let health: HealthResponse =
        serde_json::from_slice(&response.body).context("invalid benchmark health JSON")?;
    Ok(health.status == "ok")
}

fn read_macllm_runtime_attestation(
    address: SocketAddr,
    profile: &str,
    contract: &FairPrefixContract,
) -> Result<FairPrefixRuntimeAttestation> {
    let response = buffered_request(
        address,
        "GET",
        EndpointKind::Macllm.health_path(),
        None,
        Duration::from_secs(1),
    )?;
    ensure!(
        response.status == 200,
        "MacLLM runtime attestation health request returned HTTP {}",
        response.status
    );
    let health: HealthResponse =
        serde_json::from_slice(&response.body).context("invalid MacLLM health attestation JSON")?;
    let worker = health
        .worker
        .context("MacLLM health attestation omitted worker status")?;
    let served_model = health
        .model
        .context("MacLLM health attestation omitted served model")?;
    let device = health
        .device
        .context("MacLLM health attestation omitted device")?;
    let dtype = health
        .dtype
        .context("MacLLM health attestation omitted dtype")?;
    ensure!(
        health.status == "ok"
            && worker == "ready"
            && served_model == profile
            && device == "metal"
            && dtype.eq_ignore_ascii_case(&contract.macllm_kv_dtype),
        "MacLLM health attestation differs from the fair-prefix runtime contract"
    );
    let evidence_lines = vec![format!(
        "GET /healthz status={} worker={worker} model={served_model} device={device} dtype={dtype}",
        health.status
    )];
    Ok(FairPrefixRuntimeAttestation {
        source: "MacLLM GET /healthz after successful EngineHandle model load".to_owned(),
        process_command: None,
        served_model: Some(served_model),
        device,
        kv_cache_k_dtype: dtype.clone(),
        kv_cache_v_dtype: dtype,
        model_layers_offloaded: None,
        model_layers_total: None,
        flash_attention_enabled: None,
        parallel_slots: None,
        context_tokens: None,
        batch_tokens: None,
        ubatch_tokens: None,
        evidence_sha256: sha256_bytes(evidence_lines.join("\n").as_bytes()),
        evidence_lines,
    })
}

fn read_llama_cpp_runtime_attestation(
    startup_log: &str,
    contract: &FairPrefixContract,
    process_command: &[String],
) -> Result<FairPrefixRuntimeAttestation> {
    let device_line = required_log_line(
        startup_log,
        "llama_prepare_model_devices: using device MTL0",
        "Metal device selection",
    )?;
    let offload_line = required_log_line(
        startup_log,
        "load_tensors: offloaded ",
        "full model offload",
    )?;
    let parallel_line = required_log_line(
        startup_log,
        "llama_context: n_seq_max",
        "parallel slot count",
    )?;
    let context_line = required_log_line(
        startup_log,
        "llama_context: n_ctx         =",
        "context size",
    )?;
    let batch_line =
        required_log_line(startup_log, "llama_context: n_batch", "logical batch size")?;
    let ubatch_line = required_log_line(
        startup_log,
        "llama_context: n_ubatch",
        "physical batch size",
    )?;
    let kv_device_line = required_log_line(
        startup_log,
        "llama_kv_cache:       MTL0 KV buffer size",
        "Metal K/V cache allocation",
    )?;
    let kv_line = required_log_line(
        startup_log,
        "llama_kv_cache: size =",
        "K/V cache allocation",
    )?;
    let flash_line = required_flash_attention_log_line(startup_log)?;

    let (model_layers_offloaded, model_layers_total) = parse_offloaded_layers(&offload_line)?;
    let parallel_slots = parse_log_usize(&parallel_line, "n_seq_max")?;
    let context_tokens = parse_log_usize(&context_line, "n_ctx")?;
    let batch_tokens = parse_log_usize(&batch_line, "n_batch")?;
    let ubatch_tokens = parse_log_usize(&ubatch_line, "n_ubatch")?;
    ensure!(
        model_layers_offloaded == model_layers_total
            && model_layers_total > 0
            && parallel_slots == contract.concurrency
            && context_tokens == contract.llama_cpp_context_tokens
            && batch_tokens == contract.llama_cpp_batch_tokens
            && ubatch_tokens == contract.llama_cpp_ubatch_tokens
            && kv_line.contains(&format!("K ({}):", contract.llama_cpp_k_cache_dtype))
            && kv_line.contains(&format!("V ({}):", contract.llama_cpp_v_cache_dtype)),
        "llama.cpp startup log differs from the Metal/offload/F32 K/V launch contract"
    );
    let evidence_lines = vec![
        device_line,
        offload_line,
        parallel_line,
        context_line,
        batch_line,
        ubatch_line,
        kv_device_line,
        kv_line,
        flash_line,
    ];
    Ok(FairPrefixRuntimeAttestation {
        source: "separate unmeasured llama-server verbosity-4 startup attestation captured after all measured phases".to_owned(),
        process_command: Some(process_command.to_vec()),
        served_model: None,
        device: "MTL0".to_owned(),
        kv_cache_k_dtype: contract.llama_cpp_k_cache_dtype.clone(),
        kv_cache_v_dtype: contract.llama_cpp_v_cache_dtype.clone(),
        model_layers_offloaded: Some(model_layers_offloaded),
        model_layers_total: Some(model_layers_total),
        flash_attention_enabled: Some(true),
        parallel_slots: Some(parallel_slots),
        context_tokens: Some(context_tokens),
        batch_tokens: Some(batch_tokens),
        ubatch_tokens: Some(ubatch_tokens),
        evidence_sha256: sha256_bytes(evidence_lines.join("\n").as_bytes()),
        evidence_lines,
    })
}

fn validate_runtime_attestation(
    attestation: &FairPrefixRuntimeAttestation,
    implementation: &str,
    profile: &str,
    contract: &FairPrefixContract,
) -> Result<()> {
    ensure!(
        !attestation.evidence_lines.is_empty()
            && attestation.evidence_sha256
                == sha256_bytes(attestation.evidence_lines.join("\n").as_bytes()),
        "{implementation} runtime attestation evidence digest is invalid"
    );
    if implementation == EndpointKind::LlamaCpp.label() {
        let process_command = attestation
            .process_command
            .as_deref()
            .context("llama.cpp runtime attestation omitted its isolated process command")?;
        let expected = read_llama_cpp_runtime_attestation(
            &attestation.evidence_lines.join("\n"),
            contract,
            process_command,
        )?;
        ensure!(
            *attestation == expected,
            "llama.cpp structured runtime attestation differs from its startup log evidence"
        );
        return Ok(());
    }
    ensure!(
        implementation == EndpointKind::Macllm.label(),
        "unsupported runtime attestation implementation {implementation}"
    );
    let expected_line = format!(
        "GET /healthz status=ok worker=ready model={profile} device=metal dtype={}",
        contract.macllm_kv_dtype
    );
    ensure!(
        attestation.source == "MacLLM GET /healthz after successful EngineHandle model load"
            && attestation.process_command.is_none()
            && attestation.served_model.as_deref() == Some(profile)
            && attestation.device == "metal"
            && attestation
                .kv_cache_k_dtype
                .eq_ignore_ascii_case(&contract.macllm_kv_dtype)
            && attestation
                .kv_cache_v_dtype
                .eq_ignore_ascii_case(&contract.macllm_kv_dtype)
            && attestation.model_layers_offloaded.is_none()
            && attestation.model_layers_total.is_none()
            && attestation.flash_attention_enabled.is_none()
            && attestation.parallel_slots.is_none()
            && attestation.context_tokens.is_none()
            && attestation.batch_tokens.is_none()
            && attestation.ubatch_tokens.is_none()
            && attestation.evidence_lines == [expected_line],
        "MacLLM structured runtime attestation differs from its health evidence"
    );
    Ok(())
}

fn required_log_line(startup_log: &str, marker: &str, label: &str) -> Result<String> {
    startup_log
        .lines()
        .find(|line| line.contains(marker))
        .map(str::trim)
        .map(str::to_owned)
        .with_context(|| format!("llama.cpp startup log omitted {label}"))
}

fn required_flash_attention_log_line(startup_log: &str) -> Result<String> {
    startup_log
        .lines()
        .find(|line| {
            line.contains("resolve_fused_ops: Flash Attention enabled")
                || (line.contains("llama_context: flash_attn")
                    && line
                        .split_once('=')
                        .is_some_and(|(_, value)| value.trim() == "enabled"))
        })
        .map(str::trim)
        .map(str::to_owned)
        .context("llama.cpp startup log omitted an enabled Flash Attention state")
}

fn parse_log_usize(line: &str, field: &str) -> Result<usize> {
    let (_, value) = line
        .split_once('=')
        .with_context(|| format!("llama.cpp startup log {field} has no value"))?;
    value
        .split_whitespace()
        .next()
        .context("llama.cpp startup log value is empty")?
        .parse::<usize>()
        .with_context(|| format!("llama.cpp startup log {field} is not an integer"))
}

fn parse_offloaded_layers(line: &str) -> Result<(usize, usize)> {
    let (_, suffix) = line
        .split_once("offloaded ")
        .context("llama.cpp startup offload line has no count")?;
    let ratio = suffix
        .split_whitespace()
        .next()
        .context("llama.cpp startup offload count is empty")?;
    let (offloaded, total) = ratio
        .split_once('/')
        .context("llama.cpp startup offload count is not a ratio")?;
    Ok((
        offloaded
            .parse::<usize>()
            .context("llama.cpp offloaded layer count is invalid")?,
        total
            .parse::<usize>()
            .context("llama.cpp total layer count is invalid")?,
    ))
}

#[derive(Debug, Deserialize)]
struct TokenizeResponse {
    tokens: Vec<u32>,
}

fn verify_llama_tokenizer(address: SocketAddr, specs: &FairPromptSet) -> Result<()> {
    verify_llama_prompt_tokens(address, &specs.seed)?;
    for spec in &specs.tasks {
        verify_llama_prompt_tokens(address, spec)?;
    }
    Ok(())
}

fn verify_llama_prompt_tokens(address: SocketAddr, spec: &FairPromptSpec) -> Result<()> {
    let body = serde_json::to_vec(&serde_json::json!({
        "content": spec.prompt,
        "add_special": false,
    }))?;
    let response = buffered_request(address, "POST", "/tokenize", Some(&body), REQUEST_TIMEOUT)?;
    ensure!(
        response.status == 200,
        "{} /tokenize returned HTTP {}: {}",
        spec.label,
        response.status,
        String::from_utf8_lossy(&response.body)
    );
    let response: TokenizeResponse =
        serde_json::from_slice(&response.body).context("invalid llama.cpp /tokenize JSON")?;
    ensure!(
        response.tokens.len() == spec.prompt_tokens
            && token_ids_sha256(&response.tokens) == spec.prompt_token_ids_sha256,
        "{} llama.cpp tokenizer differs from the prepared MacLLM token evidence",
        spec.label
    );
    Ok(())
}

struct ChildGuard {
    child: Option<Child>,
    stderr_log: tempfile::NamedTempFile,
}

impl ChildGuard {
    fn spawn(command: &BenchmarkCommand) -> Result<Self> {
        let stderr_log =
            tempfile::NamedTempFile::new().context("failed to create benchmark child log")?;
        let stderr_writer = stderr_log
            .reopen()
            .context("failed to open benchmark child log for writing")?;
        let mut child_command = Command::new(&command.program);
        child_command
            .args(&command.arguments)
            .env_clear()
            .envs(fair_child_environment())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr_writer));
        let child = child_command
            .spawn()
            .with_context(|| format!("failed to launch {}", command.program.display()))?;
        Ok(Self {
            child: Some(child),
            stderr_log,
        })
    }

    fn stderr_text(&self) -> Result<String> {
        let bytes = fs::read(self.stderr_log.path())
            .context("failed to read benchmark child startup log")?;
        String::from_utf8(bytes).context("benchmark child startup log is not UTF-8")
    }

    fn ensure_running(&mut self) -> Result<()> {
        let child = self.child.as_mut().context("benchmark child is absent")?;
        if let Some(status) = child.try_wait().context("failed to poll benchmark child")? {
            bail!("benchmark child exited unexpectedly with {status}");
        }
        Ok(())
    }

    fn wait_until_healthy(&mut self, address: SocketAddr, kind: EndpointKind) -> Result<Duration> {
        let started_at = Instant::now();
        loop {
            self.ensure_running()?;
            if health_is_ready(address, kind).unwrap_or(false) {
                return Ok(started_at.elapsed());
            }
            ensure!(
                started_at.elapsed() < HEALTH_TIMEOUT,
                "{} did not become healthy within {} seconds",
                kind.label(),
                HEALTH_TIMEOUT.as_secs()
            );
            thread::sleep(HEALTH_POLL_INTERVAL);
        }
    }

    fn stop(mut self) -> Result<()> {
        let child = self.child.as_mut().context("benchmark child is absent")?;
        if child
            .try_wait()
            .context("failed to poll benchmark child before shutdown")?
            .is_none()
        {
            child.kill().context("failed to stop benchmark child")?;
            let _status = child.wait().context("failed to reap benchmark child")?;
        }
        self.child.take();
        Ok(())
    }
}

fn fair_child_environment() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("LANG".to_owned(), "C".to_owned()),
        ("LC_ALL".to_owned(), "C".to_owned()),
        ("PATH".to_owned(), FAIR_CHILD_PATH.to_owned()),
    ])
}

impl Drop for ChildGuard {
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
        "refusing to launch fair-prefix benchmark because {address} is already accepting connections"
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
            "fair-prefix server port {address} remained open after shutdown"
        );
        thread::sleep(PORT_POLL_INTERVAL);
    }
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>> {
        ensure!(
            self.buffer.len().saturating_add(bytes.len()) <= MAX_HTTP_BODY_BYTES,
            "fair-prefix SSE buffer exceeded its safety limit"
        );
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some((boundary, separator_len)) = sse_boundary(&self.buffer) {
            let raw = self.buffer[..boundary].to_vec();
            self.buffer.drain(..boundary + separator_len);
            let raw = std::str::from_utf8(&raw).context("fair-prefix SSE was not UTF-8")?;
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
    let deadline = Instant::now()
        .checked_add(timeout)
        .context("fair-prefix buffered request deadline overflow")?;
    let stream = DeadlineStream::connect_until(address, deadline)?;
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

struct DeadlineStream {
    stream: TcpStream,
    deadline: Instant,
}

impl DeadlineStream {
    fn connect_until(address: SocketAddr, deadline: Instant) -> Result<Self> {
        let timeout = deadline
            .checked_duration_since(Instant::now())
            .context("fair-prefix request deadline expired before connect")?;
        let stream = TcpStream::connect_timeout(&address, timeout)
            .with_context(|| format!("failed to connect to fair-prefix server at {address}"))?;
        stream
            .set_nodelay(true)
            .context("failed to disable Nagle for fair-prefix client")?;
        Ok(Self { stream, deadline })
    }

    fn remaining(&self) -> std::io::Result<Duration> {
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "fair-prefix absolute request deadline expired",
                )
            })
    }
}

impl Read for DeadlineStream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.stream.set_read_timeout(Some(self.remaining()?))?;
        let bytes = self.stream.read(buffer)?;
        self.remaining()?;
        Ok(bytes)
    }
}

impl Write for DeadlineStream {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.stream.set_write_timeout(Some(self.remaining()?))?;
        let bytes = self.stream.write(buffer)?;
        self.remaining()?;
        Ok(bytes)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.set_write_timeout(Some(self.remaining()?))?;
        self.stream.flush()?;
        self.remaining().map(|_| ())
    }
}

fn write_request(
    stream: &mut impl Write,
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
        .context("failed to write fair-prefix HTTP request head")?;
    stream
        .write_all(body)
        .context("failed to write fair-prefix HTTP request body")?;
    stream
        .flush()
        .context("failed to flush fair-prefix HTTP request")
}

struct HttpHead {
    status: u16,
    headers: BTreeMap<String, String>,
}

fn read_http_head<R: BufRead>(reader: &mut R) -> Result<HttpHead> {
    let status_line = read_http_line(reader)?.context("fair-prefix server returned no status")?;
    let mut parts = status_line.split_whitespace();
    ensure!(
        parts
            .next()
            .is_some_and(|version| version.starts_with("HTTP/1.")),
        "fair-prefix server returned an unsupported status line"
    );
    let status = parts
        .next()
        .context("fair-prefix HTTP status code is absent")?
        .parse::<u16>()
        .context("fair-prefix HTTP status code is invalid")?;
    let mut headers = BTreeMap::new();
    loop {
        let line = read_http_line(reader)?.context("fair-prefix headers ended early")?;
        if line.is_empty() {
            break;
        }
        let (name, value) = line
            .split_once(':')
            .context("fair-prefix server returned a malformed HTTP header")?;
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
        .context("failed to read fair-prefix HTTP line")?;
    if bytes == 0 {
        return Ok(None);
    }
    ensure!(
        bytes <= MAX_HTTP_LINE_BYTES,
        "fair-prefix HTTP line exceeded its safety limit"
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
                "fair-prefix HTTP body exceeded its safety limit"
            );
            body.extend_from_slice(&chunk);
        }
        return Ok(body);
    }
    if let Some(length) = headers.get("content-length") {
        let length = length
            .parse::<usize>()
            .context("fair-prefix Content-Length is invalid")?;
        ensure!(
            length <= MAX_HTTP_BODY_BYTES,
            "fair-prefix HTTP body is too large"
        );
        let mut body = vec![0; length];
        reader
            .read_exact(&mut body)
            .context("fair-prefix HTTP body ended early")?;
        return Ok(body);
    }
    let mut body = Vec::new();
    reader
        .take(MAX_HTTP_BODY_BYTES as u64 + 1)
        .read_to_end(&mut body)
        .context("failed to read fair-prefix close-delimited body")?;
    ensure!(
        body.len() <= MAX_HTTP_BODY_BYTES,
        "fair-prefix HTTP body is too large"
    );
    Ok(body)
}

fn read_http_chunk<R: BufRead>(reader: &mut R) -> Result<Option<Vec<u8>>> {
    let size_line = read_http_line(reader)?.context("fair-prefix chunk stream ended early")?;
    let size_hex = size_line.split(';').next().unwrap_or_default().trim();
    ensure!(
        !size_hex.is_empty(),
        "fair-prefix server returned an empty chunk size"
    );
    let size = usize::from_str_radix(size_hex, 16)
        .context("fair-prefix server returned an invalid chunk size")?;
    ensure!(
        size <= MAX_HTTP_CHUNK_BYTES,
        "fair-prefix HTTP chunk exceeded its safety limit"
    );
    if size == 0 {
        loop {
            let trailer =
                read_http_line(reader)?.context("fair-prefix chunk trailers ended early")?;
            if trailer.is_empty() {
                break;
            }
        }
        return Ok(None);
    }
    let mut chunk = vec![0; size];
    reader
        .read_exact(&mut chunk)
        .context("fair-prefix HTTP chunk ended early")?;
    let mut terminator = [0_u8; 2];
    reader
        .read_exact(&mut terminator)
        .context("fair-prefix chunk terminator ended early")?;
    ensure!(
        terminator == *b"\r\n",
        "fair-prefix HTTP chunk did not end in CRLF"
    );
    Ok(Some(chunk))
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= 1.0e-6 * scale
}

fn optional_approximately_equal(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => approximately_equal(left, right),
        (None, None) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Cursor, path::PathBuf};

    use serde_json::json;

    use super::*;
    use crate::{
        benchmark::{
            MaterializedPrefixPrompt, PREFIX_DEMO_SCHEMA_VERSION, PreparedPrefixPrompt,
            REPOSITORY_PAYLOAD_ENCODING, RepositoryBundleEvidence,
        },
        chat::ChatMessage,
    };

    const FIXTURE_SYSTEM_INSTRUCTION: &str = concat!(
        "/system_override/no_think\n",
        "You are MacLLM's local coding agent. Use only the repository bundle prefix below. ",
        "Cite concrete file paths, do not invent missing code, and keep the answer concise. ",
        "The deterministic bundle may end at a UTF-8 boundary inside the final file.\n\n",
        "REPOSITORY BUNDLE PREFIX\n",
    );
    const FIXTURE_TASKS: [&str; MAC_PREFIX_DEMO_CONCURRENCY] = [
        "A: In src/agent.rs, list the fail-closed request rules enforced by validate_agent_request.",
        "B: In src/agent.rs, trace how assistant tool-call history is rendered and validated.",
        "C: In src/architecture.rs, explain DecoderSpec::validate invariants for heads and layers.",
        "D: In src/architecture.rs, compare the represented full, sliding, and recurrent state semantics.",
        "E: In src/architecture.rs, explain PrefixCacheDiagnostics counters and physical owner meaning.",
        "F: In src/backend.rs, explain how device and dtype selection fail closed on unsupported builds.",
        "G: In src/benchmark/assemble.rs, explain how MacllmProvenanceGuard pins executable and source identity.",
        "H: In src/benchmark/assemble.rs, audit one capture-time check that rejects mixed provenance.",
    ];
    const FIXTURE_SEED_TASK: &str =
        "0: From Cargo.toml, state the package purpose, default features, and release tuning.";

    fn contract() -> FairPrefixContract {
        FairPrefixContract {
            comparison_scope: "controlled_exact_slot_cache_lane".to_owned(),
            exact_shared_prefix_tokens: 16_384,
            macllm_prefix_cache_block_tokens: 16,
            macllm_kv_cache_slab_blocks: 8,
            macllm_kv_cache_capacity_tokens: 65_536,
            macllm_prefill_token_quantum: 128,
            macllm_scheduler_token_budget: 1_024,
            concurrency: 8,
            completion_tokens_per_request: 32,
            measured_waves_per_implementation: 20,
            post_seed_stabilization_ms: 30_000,
            warmup_waves_per_lifetime: 5,
            steady_state_window_waves: 3,
            steady_state_max_cv_bits: 0.10_f64.to_bits(),
            steady_state_max_relative_trend_bits: 0.10_f64.to_bits(),
            temperature_bits: 0.0_f64.to_bits(),
            top_k: 0,
            top_p_bits: 1.0_f64.to_bits(),
            repetition_penalty_bits: 1.0_f64.to_bits(),
            seed: 42,
            ignore_eos: true,
            cache_prompt: true,
            stream: true,
            return_tokens: true,
            macllm_kv_dtype: "f32".to_owned(),
            llama_cpp_k_cache_dtype: "f32".to_owned(),
            llama_cpp_v_cache_dtype: "f32".to_owned(),
            llama_cpp_context_tokens: 135_168,
            llama_cpp_batch_tokens: 1_024,
            llama_cpp_ubatch_tokens: 128,
            llama_cpp_threads: 10,
            llama_cpp_gpu_layers: 99,
            llama_cpp_flash_attention: "on".to_owned(),
            llama_cpp_mmap: true,
            loopback_port: 18_086,
            transport: FAIR_TRANSPORT.to_owned(),
            child_environment: fair_child_environment(),
        }
    }

    fn prompt() -> FairPromptSpec {
        FairPromptSpec {
            label: "request_3".to_owned(),
            prompt: "hello".to_owned(),
            token_ids: Vec::new(),
            prompt_tokens: 16_400,
            prompt_token_ids_sha256: "a".repeat(64),
            slot_id: 3,
        }
    }

    #[test]
    fn fair_prompt_set_uses_the_exact_materialized_cache_seed() -> Result<()> {
        let common = (0..16_384)
            .map(|token| u32::try_from(token).expect("fixture token fits u32"))
            .collect::<Vec<_>>();
        let cache_seed = MaterializedPrefixPrompt {
            name: "cache_seed".to_owned(),
            rendered_prompt: "shared raw prefix".to_owned(),
            token_ids: common.clone(),
        };
        let mut reference_ids = common.clone();
        reference_ids.extend([90_001, 90_002]);
        let seed = MaterializedPrefixPrompt {
            name: "seed".to_owned(),
            rendered_prompt: "shared raw prefix reference task".to_owned(),
            token_ids: reference_ids,
        };
        let tasks = (0..MAC_PREFIX_DEMO_CONCURRENCY)
            .map(|index| {
                let mut token_ids = common.clone();
                token_ids.push(100_000 + u32::try_from(index).expect("fixture index fits u32"));
                MaterializedPrefixPrompt {
                    name: format!("task_{index}"),
                    rendered_prompt: format!("shared raw prefix task {index}"),
                    token_ids,
                }
            })
            .collect();
        let materialized = MaterializedPrefixDemo {
            seed,
            cache_seed,
            tasks,
        };

        let prompts = FairPromptSet::new(&materialized, &contract())?;
        assert_eq!(prompts.seed.prompt, "shared raw prefix");
        assert_eq!(prompts.seed.prompt_tokens, 16_384);
        assert_eq!(prompts.conditioning_lcp_tokens, vec![16_384; 8]);
        Ok(())
    }

    fn fixture_file(path: &str, filename: &str, digest_byte: char) -> FileProvenance {
        FileProvenance {
            canonical_path: PathBuf::from(path),
            filename: filename.to_owned(),
            size_bytes: 1,
            sha256: digest_byte.to_string().repeat(64),
        }
    }

    fn fixture_selected_bundle() -> String {
        concat!(
            "--- FILE: Cargo.toml ---\n",
            "[package]\nname = \"fixture\"\n",
            "--- FILE: src/agent.rs ---\n",
            "pub fn validate_agent_request() {}\n",
            "--- FILE: src/architecture.rs ---\n",
            "pub struct DecoderSpec;\n",
            "--- FILE: src/backend.rs ---\n",
            "pub fn select_backend() {}\n",
            "--- FILE: src/benchmark/assemble.rs ---\n",
            "pub struct MacllmProvenanceGuard;\n",
        )
        .to_owned()
    }

    fn fixture_prepared_prompt(
        name: String,
        task_index: Option<usize>,
        system: &str,
        user: &str,
        prompt_token_ids_sha256: String,
        common_prefix_token_ids_sha256: &str,
    ) -> PreparedPrefixPrompt {
        PreparedPrefixPrompt {
            name,
            task_index,
            messages: vec![
                ChatMessage::system(system.to_owned()),
                ChatMessage::user(user.to_owned()),
            ],
            rendered_prompt_byte_count: system.len() + user.len(),
            rendered_prompt_sha256: prompt_token_ids_sha256.clone(),
            prompt_token_count: 16_400,
            prompt_token_ids_sha256,
            lcp_with_seed_tokens: task_index.map(|_| 16_384),
            cache_endpoint_tokens: 16_384,
            common_prefix_token_ids_sha256: common_prefix_token_ids_sha256.to_owned(),
        }
    }

    fn fixture_prepared(model_file: FileProvenance) -> PreparedPrefixDemo {
        let selected_bundle = fixture_selected_bundle();
        let system = format!("{FIXTURE_SYSTEM_INSTRUCTION}{selected_bundle}");
        let common_prefix_token_ids_sha256 = "f".repeat(64);
        let seed = fixture_prepared_prompt(
            "seed".to_owned(),
            None,
            &system,
            FIXTURE_SEED_TASK,
            format!("{:064x}", 100),
            &common_prefix_token_ids_sha256,
        );
        let tasks = FIXTURE_TASKS
            .iter()
            .enumerate()
            .map(|(offset, task)| {
                let task_index = offset + 1;
                fixture_prepared_prompt(
                    format!("task_{task_index}"),
                    Some(task_index),
                    &system,
                    task,
                    format!("{task_index:064x}"),
                    &common_prefix_token_ids_sha256,
                )
            })
            .collect();
        let selected_bundle_sha256 = sha256_bytes(selected_bundle.as_bytes());
        PreparedPrefixDemo {
            schema_version: PREFIX_DEMO_SCHEMA_VERSION,
            model_family: "smollm3".to_owned(),
            model_file,
            repository_bundle: RepositoryBundleEvidence {
                file_count: 5,
                byte_count: u64::try_from(selected_bundle.len()).unwrap(),
                sha256: selected_bundle_sha256.clone(),
                files: [
                    "Cargo.toml",
                    "src/agent.rs",
                    "src/architecture.rs",
                    "src/backend.rs",
                    "src/benchmark/assemble.rs",
                ]
                .map(str::to_owned)
                .to_vec(),
            },
            repository_payload_encoding: REPOSITORY_PAYLOAD_ENCODING.to_owned(),
            encoded_repository_bundle_byte_count: u64::try_from(selected_bundle.len()).unwrap(),
            encoded_repository_bundle_sha256: selected_bundle_sha256.clone(),
            selected_bundle_byte_count: u64::try_from(selected_bundle.len()).unwrap(),
            selected_bundle_char_count: selected_bundle.chars().count(),
            selected_bundle_sha256,
            requested_shared_tokens: 16_384,
            prefill_quantum: 128,
            kv_cache_block_tokens: 16,
            common_prefix_token_ids_sha256,
            seed,
            tasks,
        }
    }

    fn fixture_replay(prepared: &PreparedPrefixDemo) -> FairPrefixReplayEvidence {
        let content = (0..MAC_PREFIX_DEMO_CONCURRENCY)
            .map(|request_index| format!("fixture answer {request_index}"))
            .collect::<Vec<_>>();
        FairPrefixReplayEvidence {
            prompt_token_ids_sha256: prepared
                .tasks
                .iter()
                .map(|task| task.prompt_token_ids_sha256.clone())
                .collect(),
            content_sha256: content
                .iter()
                .map(|value| sha256_bytes(value.as_bytes()))
                .collect(),
            content,
        }
    }

    fn fixture_macllm_attestation(
        profile: &str,
        contract: &FairPrefixContract,
    ) -> FairPrefixRuntimeAttestation {
        let evidence_lines = vec![format!(
            "GET /healthz status=ok worker=ready model={profile} device=metal dtype={}",
            contract.macllm_kv_dtype
        )];
        FairPrefixRuntimeAttestation {
            source: "MacLLM GET /healthz after successful EngineHandle model load".to_owned(),
            process_command: None,
            served_model: Some(profile.to_owned()),
            device: "metal".to_owned(),
            kv_cache_k_dtype: contract.macllm_kv_dtype.clone(),
            kv_cache_v_dtype: contract.macllm_kv_dtype.clone(),
            model_layers_offloaded: None,
            model_layers_total: None,
            flash_attention_enabled: None,
            parallel_slots: None,
            context_tokens: None,
            batch_tokens: None,
            ubatch_tokens: None,
            evidence_sha256: sha256_bytes(evidence_lines.join("\n").as_bytes()),
            evidence_lines,
        }
    }

    fn fixture_llama_attestation(
        contract: &FairPrefixContract,
        process_command: &[String],
    ) -> FairPrefixRuntimeAttestation {
        let log = "\
I llama_prepare_model_devices: using device MTL0 (Apple M4 Pro)
I load_tensors: offloaded 37/37 layers to GPU
I llama_context: n_seq_max     = 8
I llama_context: n_ctx         = 135168
I llama_context: n_batch       = 1024
I llama_context: n_ubatch      = 128
I llama_kv_cache:       MTL0 KV buffer size = 19008.00 MiB
I llama_kv_cache: size = 19008.00 MiB, K (f32): 9504.00 MiB, V (f32): 9504.00 MiB
I resolve_fused_ops: Flash Attention enabled
";
        read_llama_cpp_runtime_attestation(log, contract, process_command).unwrap()
    }

    #[allow(clippy::too_many_lines)] // Keeping the complete synthetic run in one helper makes cross-field invariants auditable.
    fn fixture_run(
        kind: EndpointKind,
        prepared: &PreparedPrefixDemo,
        contract: &FairPrefixContract,
        profile: &str,
        process_sequence_indices: [usize; FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION],
        wall_time_ms: f64,
    ) -> FairPrefixImplementationRun {
        let replay = fixture_replay(prepared);
        let total_completion_tokens =
            u64::try_from(contract.concurrency * contract.completion_tokens_per_request).unwrap();
        let waves = (0..contract.measured_waves_per_implementation)
            .map(|wave_index| {
                let requests = prepared
                    .tasks
                    .iter()
                    .enumerate()
                    .map(|(request_index, task)| FairPrefixRequestEvidence {
                        request_index,
                        slot_id: request_index,
                        prompt_tokens: task.prompt_token_count,
                        prompt_token_ids_sha256: task.prompt_token_ids_sha256.clone(),
                        streamed_content: replay.content[request_index].clone(),
                        streamed_token_ids: (0..contract.completion_tokens_per_request)
                            .map(|token_index| {
                                u32::try_from(1_000 + request_index * 100 + token_index).unwrap()
                            })
                            .collect(),
                        submitted_offset_ms: 0.0,
                        first_token_offset_ms: wall_time_ms / 10.0,
                        completed_offset_ms: wall_time_ms,
                        external_time_to_first_token_ms: wall_time_ms / 10.0,
                        external_latency_ms: wall_time_ms,
                        terminal: FairPrefixTerminalEvidence {
                            cached_prompt_tokens: contract.exact_shared_prefix_tokens,
                            evaluated_prompt_tokens: task.prompt_token_count
                                - contract.exact_shared_prefix_tokens,
                            generated_tokens: contract.completion_tokens_per_request,
                            server_prompt_ms: 1.0,
                            server_predicted_ms: 2.0,
                        },
                    })
                    .collect();
                FairPrefixWaveEvidence {
                    wave_index,
                    process_sequence_index: process_sequence_indices
                        [wave_index / (contract.measured_waves_per_implementation / 4)],
                    request_count: contract.concurrency,
                    total_completion_tokens,
                    external_wall_time_ms: wall_time_ms,
                    aggregate_output_tokens_per_second: total_completion_tokens as f64 * 1_000.0
                        / wall_time_ms,
                    exact_cache_hit_requests: contract.concurrency,
                    requests,
                }
            })
            .collect::<Vec<_>>();
        let runtime_attestations = if kind == EndpointKind::Macllm {
            vec![
                fixture_macllm_attestation(profile, contract);
                FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
            ]
        } else {
            Vec::new()
        };
        FairPrefixImplementationRun {
            implementation: kind.label().to_owned(),
            process_command: Vec::new(),
            isolated_os_process: true,
            base_url: format!("http://{}:{}", LOOPBACK, contract.loopback_port),
            process_sequence_indices: process_sequence_indices.to_vec(),
            health_wait_ms_samples: vec![1.0; FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION],
            post_seed_stabilization_ms_samples: vec![
                contract.post_seed_stabilization_ms as f64;
                FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
            ],
            warmup_waves_per_lifetime: FAIR_WARMUP_WAVES_PER_LIFETIME,
            warmup_aggregate_output_tokens_per_second_samples: vec![
                100.0;
                FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
                    * FAIR_WARMUP_WAVES_PER_LIFETIME
            ],
            runtime_attestations,
            measurement_boundary: FAIR_TRANSPORT.to_owned(),
            tokenizer_exact: true,
            cache_prompt_publication: true,
            seed_requests: FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION * contract.concurrency,
            restore_requests: FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
                * (contract.measured_waves_per_implementation
                    / FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION
                    + FAIR_WARMUP_WAVES_PER_LIFETIME)
                * contract.concurrency,
            conditioning_prompt_token_ids_sha256: (0..contract.concurrency)
                .map(|slot_id| {
                    prepared.tasks[(slot_id + 1) % contract.concurrency]
                        .prompt_token_ids_sha256
                        .clone()
                })
                .collect(),
            conditioning_lcp_tokens: vec![
                contract.exact_shared_prefix_tokens;
                contract.concurrency
            ],
            distributions: build_distributions(&waves).unwrap(),
            waves,
            lifetime_replays: vec![replay.clone(); FAIR_PROCESS_LIFETIMES_PER_IMPLEMENTATION],
            replay,
            port_closed_after_shutdown: true,
        }
    }

    #[allow(clippy::too_many_lines)] // The fixture mirrors the complete persisted publication boundary.
    fn synthetic_valid_report() -> FairPrefixReport {
        let profile = "fixture-smollm3".to_owned();
        let model_file = fixture_file("/fixtures/model.gguf", "model.gguf", 'a');
        let model = ModelProvenance {
            file: model_file.clone(),
            snapshot: None,
            architecture: "smollm3".to_owned(),
            quantization: "Q4_K_M".to_owned(),
        };
        let prepared = fixture_prepared(model_file);
        let contract = contract();
        let host = SanitizedHost {
            machine_name: "Mac".to_owned(),
            machine_model: "Mac16,7".to_owned(),
            chip: "Apple M4 Pro".to_owned(),
            memory_bytes: 24 * 1024 * 1024 * 1024,
            total_cores: Some(12),
            performance_cores: Some(10),
            efficiency_cores: Some(2),
            os_version: "15.5".to_owned(),
            os_build: "24F74".to_owned(),
            architecture: "arm64".to_owned(),
        };
        let macllm = fixture_run(
            EndpointKind::Macllm,
            &prepared,
            &contract,
            &profile,
            [1, 2, 4, 7],
            100.0,
        );
        let llama_cpp = fixture_run(
            EndpointKind::LlamaCpp,
            &prepared,
            &contract,
            &profile,
            [0, 3, 5, 6],
            200.0,
        );
        let control_attestation = fixture_llama_attestation(&contract, &[]);
        let execution_order = FAIR_PROCESS_ORDER.map(str::to_owned).to_vec();
        let eligibility = derive_eligibility(
            &contract,
            &model,
            &prepared,
            &profile,
            &execution_order,
            &macllm,
            &llama_cpp,
            &control_attestation,
        );
        let comparison = derive_comparison(&eligibility, &macllm, &llama_cpp).unwrap();
        let mut report = FairPrefixReport {
            schema_version: FAIR_PREFIX_REPORT_SCHEMA_VERSION,
            generated_unix_ms: 1,
            profile,
            host,
            model,
            macllm_provenance: ImplementationProvenance {
                implementation: ImplementationKind::Macllm,
                version: "0.1.0-test".to_owned(),
                commit: "b".repeat(40),
                executable: fixture_file("/fixtures/macllm", "macllm", 'b'),
                dirty: Some(false),
            },
            llama_cpp_server: fixture_file("/fixtures/llama-server", "llama-server", 'c'),
            llama_cpp_version: LlamaCppVersion {
                build: 10_050,
                commit_abbreviation: "d".repeat(7),
                compiler_and_target: None,
            },
            llama_cpp_full_commit: "d".repeat(40),
            llama_cpp_control_attestation: control_attestation,
            prepared,
            contract,
            execution_order,
            macllm,
            llama_cpp,
            eligibility,
            comparison,
            warnings: canonical_warnings(),
        };
        report.macllm.process_command = expected_macllm_process_command(&report).unwrap();
        report.llama_cpp.process_command = expected_llama_cpp_process_command(&report).unwrap();
        let control_command =
            attestation_argv_from_measured(report.llama_cpp.process_command.clone()).unwrap();
        report.llama_cpp_control_attestation =
            fixture_llama_attestation(&report.contract, &control_command);
        report.eligibility = derive_eligibility(
            &report.contract,
            &report.model,
            &report.prepared,
            &report.profile,
            &report.execution_order,
            &report.macllm,
            &report.llama_cpp,
            &report.llama_cpp_control_attestation,
        );
        report.comparison =
            derive_comparison(&report.eligibility, &report.macllm, &report.llama_cpp).unwrap();
        report
    }

    #[test]
    fn synthetic_report_passes_complete_validation() {
        synthetic_valid_report().validate().unwrap();
    }

    #[test]
    fn report_validation_rejects_timing_offset_and_wall_drift() {
        let mut wall_drift = synthetic_valid_report();
        wall_drift.macllm.waves[0].external_wall_time_ms += 1.0;
        assert!(wall_drift.validate().is_err());

        let mut offset_drift = synthetic_valid_report();
        offset_drift.llama_cpp.waves[0].requests[0].completed_offset_ms += 1.0;
        assert!(offset_drift.validate().is_err());
    }

    #[test]
    fn report_validation_rejects_process_block_rebinding() {
        let mut report = synthetic_valid_report();
        report.macllm.waves[5].process_sequence_index = 1;
        assert!(report.validate().is_err());
    }

    #[test]
    fn report_validation_rejects_cross_implementation_token_id_drift() {
        let mut report = synthetic_valid_report();
        for wave in &mut report.llama_cpp.waves {
            wave.requests[0].streamed_token_ids[0] += 1;
        }
        assert!(report.validate().is_err());
    }

    #[test]
    fn report_validation_rejects_host_thread_drift() {
        let mut report = synthetic_valid_report();
        report.host.total_cores = Some(10);
        report.host.performance_cores = Some(8);
        report.host.efficiency_cores = Some(2);
        assert!(report.validate().is_err());
    }

    #[test]
    fn report_validation_rejects_control_attestation_argv_drift() {
        let mut report = synthetic_valid_report();
        report
            .llama_cpp_control_attestation
            .process_command
            .as_mut()
            .unwrap()
            .push("--unexpected".to_owned());
        assert!(report.validate().is_err());
    }

    #[test]
    fn report_validation_rejects_missing_lifetime_replay() {
        let mut report = synthetic_valid_report();
        report.macllm.lifetime_replays.pop();
        assert!(report.validate().is_err());
    }

    #[test]
    fn report_validation_rejects_paired_ratio_and_geomean_drift() {
        let mut pair_drift = synthetic_valid_report();
        pair_drift.comparison.paired_process_blocks[0].macllm_to_llama_cpp_phase_throughput_ratio =
            Some(3.0);
        assert!(pair_drift.validate().is_err());

        let mut geomean_drift = synthetic_valid_report();
        geomean_drift
            .comparison
            .macllm_to_llama_cpp_paired_geomean_throughput_ratio = Some(3.0);
        assert!(geomean_drift.validate().is_err());
    }

    #[test]
    fn shared_request_body_contains_only_the_strict_cross_server_contract() {
        let body = completion_body(&prompt(), 32, true, &contract()).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            body,
            json!({
                "prompt": "hello",
                "n_predict": 32,
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 1.0,
                "seed": 42,
                "cache_prompt": true,
                "stream": true,
                "return_tokens": true,
                "id_slot": 3,
                "ignore_eos": true,
                "repeat_penalty": 1.0,
                "response_fields": [
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
                    "timings/predicted_ms"
                ]
            })
        );
        assert_eq!(macllm_exact_seed_reuse_tokens(&contract()).unwrap(), 16_368);
    }

    #[test]
    fn streamed_fragment_validator_requires_real_monotonic_token_accounting() {
        let value = json!({
            "index": 0,
            "content": "x",
            "tokens": [123],
            "stop": false,
            "id_slot": -1,
            "tokens_predicted": 1,
            "tokens_evaluated": 16400
        });
        let mut last_n_decoded = 0;
        for n_decoded in 1..=32 {
            let mut current = value.clone();
            current["tokens"] = json!([122 + u32::try_from(n_decoded).unwrap()]);
            current["tokens_predicted"] = json!(n_decoded);
            let current: WireCompletionEvent = serde_json::from_value(current).unwrap();
            assert_eq!(
                validate_wire_fragment(&current, &prompt(), last_n_decoded, 32).unwrap(),
                (n_decoded, 122 + u32::try_from(n_decoded).unwrap())
            );
            last_n_decoded = n_decoded;
        }

        let event: WireCompletionEvent = serde_json::from_value(value.clone()).unwrap();
        assert!(validate_wire_fragment(&event, &prompt(), 1, 32).is_err());

        let mut skipped_ordinal = value.clone();
        skipped_ordinal["tokens_predicted"] = json!(3);
        let skipped_ordinal = serde_json::from_value(skipped_ordinal).unwrap();
        assert!(validate_wire_fragment(&skipped_ordinal, &prompt(), 1, 32).is_err());

        let mut prompt_drift = value.clone();
        prompt_drift["tokens_predicted"] = json!(2);
        prompt_drift["tokens_evaluated"] = json!(16_399);
        let prompt_drift = serde_json::from_value(prompt_drift).unwrap();
        assert!(validate_wire_fragment(&prompt_drift, &prompt(), 1, 32).is_err());

        let mut past_budget = value;
        past_budget["tokens_predicted"] = json!(33);
        let past_budget = serde_json::from_value(past_budget).unwrap();
        assert!(validate_wire_fragment(&past_budget, &prompt(), 1, 32).is_err());
    }

    #[test]
    fn terminal_validator_requires_exact_cache_prompt_and_output_accounting() {
        let value = json!({
            "index": 0,
            "content": "",
            "stop": true,
            "tokens_predicted": 32,
            "tokens_evaluated": 16400,
            "truncated": false,
            "stop_type": "limit",
            "generation_settings/seed": 42,
            "generation_settings/temperature": 0.0,
            "generation_settings/top_k": 0,
            "generation_settings/top_p": 1.0,
            "generation_settings/repeat_penalty": 1.0,
            "generation_settings/ignore_eos": true,
            "generation_settings/stream": true,
            "generation_settings/max_tokens": 32,
            "generation_settings/n_predict": 32,
            "generation_settings/n_keep": 0,
            "timings/cache_n": 16384,
            "timings/prompt_n": 16,
            "timings/prompt_ms": 1.0,
            "timings/predicted_n": 32,
            "timings/predicted_ms": 2.0
        });
        let event: WireCompletionEvent = serde_json::from_value(value.clone()).unwrap();
        let terminal = validate_wire_terminal(&event, &prompt(), &contract(), true).unwrap();
        assert_eq!(terminal.cached_prompt_tokens, 16_384);
        assert_eq!(terminal.evaluated_prompt_tokens, 16);
        assert_eq!(terminal.generated_tokens, 32);

        let mut zero_decode_time = value.clone();
        zero_decode_time["timings/predicted_ms"] = json!(0.0);
        let zero_decode_time = serde_json::from_value(zero_decode_time).unwrap();
        assert!(validate_wire_terminal(&zero_decode_time, &prompt(), &contract(), true).is_ok());

        let mut leaked = value.clone();
        leaked["prompt"] = json!("a projected response must never echo the 16K prompt");
        assert!(
            serde_json::from_value::<WireCompletionEvent>(leaked).is_err(),
            "unknown terminal fields must fail closed"
        );

        let mut bad = value;
        bad["timings/cache_n"] = json!(16_383);
        let bad: WireCompletionEvent = serde_json::from_value(bad).unwrap();
        assert!(validate_wire_terminal(&bad, &prompt(), &contract(), true).is_err());
    }

    #[test]
    fn shared_chunk_and_sse_framing_handles_split_events() {
        let mut decoder = SseDecoder::default();
        assert!(
            decoder
                .push(b"data: {\"index\":0,\"content\":\"he")
                .unwrap()
                .is_empty()
        );
        let events = decoder
            .push(b"llo\",\"stop\":false}\r\n\r\ndata: [DONE]\n\n")
            .unwrap();
        assert_eq!(
            events,
            vec![
                "{\"index\":0,\"content\":\"hello\",\"stop\":false}",
                "[DONE]"
            ]
        );

        let mut input = Cursor::new(b"5\r\nhello\r\n0\r\nX-Test: yes\r\n\r\n");
        assert_eq!(read_http_chunk(&mut input).unwrap().unwrap(), b"hello");
        assert!(read_http_chunk(&mut input).unwrap().is_none());
    }

    #[test]
    fn streaming_payload_distinguishes_completion_events_from_server_errors() {
        let completion =
            serde_json::from_str::<WireSsePayload>(r#"{"index":0,"content":"ok","stop":false}"#)
                .unwrap();
        assert!(matches!(completion, WireSsePayload::Completion(_)));

        let error = serde_json::from_str::<WireSsePayload>(
            r#"{"error":{"message":"worker failed","type":"server_error","param":null,"code":"internal_error"}}"#,
        )
        .unwrap();
        let WireSsePayload::Error(error) = error else {
            panic!("server error was parsed as a completion event");
        };
        assert_eq!(error.error.message, "worker failed");
        assert_eq!(error.error.error_type, "server_error");
        assert_eq!(error.error.code.as_deref(), Some("internal_error"));
    }

    #[test]
    fn llama_startup_log_attests_metal_full_offload_f32_kv_and_flash_attention() {
        let log = "\
I llama_prepare_model_devices: using device MTL0 (Apple M4 Pro)
I load_tensors: offloaded 37/37 layers to GPU
I llama_context: n_seq_max     = 8
I llama_context: n_ctx         = 135168
I llama_context: n_batch       = 1024
I llama_context: n_ubatch      = 128
I llama_kv_cache:       MTL0 KV buffer size = 19008.00 MiB
I llama_kv_cache: size = 19008.00 MiB, K (f32): 9504.00 MiB, V (f32): 9504.00 MiB
I resolve_fused_ops: Flash Attention enabled
";
        let command = vec!["llama-server".to_owned(), "-lv".to_owned(), "4".to_owned()];
        let attestation = read_llama_cpp_runtime_attestation(log, &contract(), &command).unwrap();
        assert_eq!(attestation.device, "MTL0");
        assert_eq!(attestation.model_layers_offloaded, Some(37));
        assert_eq!(attestation.model_layers_total, Some(37));
        assert_eq!(attestation.flash_attention_enabled, Some(true));
        assert!(
            validate_runtime_attestation(&attestation, "llama.cpp", "fixture", &contract()).is_ok()
        );
    }

    #[test]
    fn llama_startup_log_accepts_current_context_flash_attention_line() {
        let log = "\
I llama_prepare_model_devices: using device MTL0 (Apple M4 Pro)
I load_tensors: offloaded 37/37 layers to GPU
I llama_context: n_seq_max     = 8
I llama_context: n_ctx         = 135168
I llama_context: n_batch       = 1024
I llama_context: n_ubatch      = 128
I llama_context: flash_attn    = enabled
I llama_kv_cache:       MTL0 KV buffer size = 19008.00 MiB
I llama_kv_cache: size = 19008.00 MiB, K (f32): 9504.00 MiB, V (f32): 9504.00 MiB
";
        let command = vec!["llama-server".to_owned(), "-lv".to_owned(), "4".to_owned()];
        let attestation = read_llama_cpp_runtime_attestation(log, &contract(), &command).unwrap();
        assert_eq!(
            attestation.evidence_lines.last().map(String::as_str),
            Some("I llama_context: flash_attn    = enabled")
        );
        assert_eq!(attestation.flash_attention_enabled, Some(true));
    }

    #[test]
    fn llama_startup_log_rejects_disabled_context_flash_attention() {
        let log = "\
I llama_prepare_model_devices: using device MTL0 (Apple M4 Pro)
I load_tensors: offloaded 37/37 layers to GPU
I llama_context: n_seq_max     = 8
I llama_context: n_ctx         = 135168
I llama_context: n_batch       = 1024
I llama_context: n_ubatch      = 128
I llama_context: flash_attn    = disabled
I llama_kv_cache:       MTL0 KV buffer size = 19008.00 MiB
I llama_kv_cache: size = 19008.00 MiB, K (f32): 9504.00 MiB, V (f32): 9504.00 MiB
";
        let command = vec!["llama-server".to_owned(), "-lv".to_owned(), "4".to_owned()];
        let error = read_llama_cpp_runtime_attestation(log, &contract(), &command).unwrap_err();
        assert!(
            format!("{error:#}").contains("enabled Flash Attention state"),
            "{error:#}"
        );
    }

    #[test]
    fn steady_state_gate_rejects_noisy_or_trending_final_window() {
        let stable = [50.0, 70.0, 100.0, 102.0, 101.0];
        assert!(validate_steady_state_samples(&stable, &contract(), "fixture", 0).is_ok());

        let noisy = [50.0, 70.0, 100.0, 140.0, 80.0];
        assert!(validate_steady_state_samples(&noisy, &contract(), "fixture", 0).is_err());
    }

    #[test]
    fn phase_throughput_uses_total_tokens_over_summed_wall_time() {
        let wave =
            |wave_index, total_completion_tokens, external_wall_time_ms| FairPrefixWaveEvidence {
                wave_index,
                process_sequence_index: 3,
                request_count: 8,
                total_completion_tokens,
                external_wall_time_ms,
                aggregate_output_tokens_per_second: total_completion_tokens as f64 * 1_000.0
                    / external_wall_time_ms,
                exact_cache_hit_requests: 8,
                requests: Vec::new(),
            };
        let waves = [wave(0, 100, 1_000.0), wave(1, 100, 3_000.0)];
        let throughput = phase_throughput(&waves, "fixture", 3, 2).unwrap();
        assert!(approximately_equal(throughput, 50.0));
    }
}
