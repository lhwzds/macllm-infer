//! Versioned, fail-closed schemas for reproducible benchmark inputs and output.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use super::{host::SanitizedHost, stats::DistributionSummary};

/// Current on-disk manifest and report schema version.
pub const SCOREBOARD_SCHEMA_VERSION: u32 = 1;
/// A percentile needs at least twenty observations before it is reported.
pub const MIN_PERCENTILE_SAMPLES: usize = 20;
/// The first public scoreboard always measures these serving widths.
pub const REQUIRED_CONCURRENCY: [usize; 3] = [1, 4, 8];
/// Canonical seed used even though greedy argmax does not consume randomness.
pub const GREEDY_SEED: u64 = 42;

/// Complete, immutable description of one scoreboard run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreboardManifest {
    pub schema_version: u32,
    pub name: String,
    pub host: HostRequirement,
    pub model: ModelRequirement,
    pub llama_cpp: LlamaCppRequirement,
    pub workload: WorkloadSpec,
    pub runtime: RuntimeBenchmarkSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fair_prefix: Option<FairPrefixManifestSpec>,
}

impl ScoreboardManifest {
    /// Reject incomplete, ambiguous, or silently upgraded benchmark inputs.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == SCOREBOARD_SCHEMA_VERSION,
            "unsupported scoreboard manifest schema version {}; expected {SCOREBOARD_SCHEMA_VERSION}",
            self.schema_version
        );
        ensure!(
            !self.name.trim().is_empty(),
            "scoreboard name cannot be empty"
        );
        self.host.validate()?;
        self.model.validate()?;
        self.llama_cpp.validate()?;
        self.workload.validate()?;
        self.runtime.validate()?;
        if let Some(fair_prefix) = &self.fair_prefix {
            fair_prefix.validate()?;
        }
        Ok(())
    }
}

/// Hardware identity required by a benchmark profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostRequirement {
    pub machine_model: String,
    pub chip: String,
    pub memory_bytes: u64,
    pub architecture: String,
}

impl HostRequirement {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.machine_model.trim().is_empty(),
            "host machine_model cannot be empty"
        );
        ensure!(!self.chip.trim().is_empty(), "host chip cannot be empty");
        ensure!(self.memory_bytes > 0, "host memory_bytes must be positive");
        ensure!(
            !self.architecture.trim().is_empty(),
            "host architecture cannot be empty"
        );
        Ok(())
    }

    /// Verify that the current sanitized host is exactly the declared machine.
    pub fn verify(&self, actual: &SanitizedHost) -> Result<()> {
        ensure!(
            actual.machine_model == self.machine_model,
            "host model mismatch: expected {}, found {}",
            self.machine_model,
            actual.machine_model
        );
        ensure!(
            actual.chip == self.chip,
            "host chip mismatch: expected {}, found {}",
            self.chip,
            actual.chip
        );
        ensure!(
            actual.memory_bytes == self.memory_bytes,
            "host memory mismatch: expected {} bytes, found {} bytes",
            self.memory_bytes,
            actual.memory_bytes
        );
        ensure!(
            actual.architecture == self.architecture,
            "host architecture mismatch: expected {}, found {}",
            self.architecture,
            actual.architecture
        );
        Ok(())
    }
}

/// Expected identity of the exact model bytes used by both implementations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRequirement {
    pub path: PathBuf,
    pub repository: String,
    pub revision: String,
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub architecture: String,
    pub quantization: String,
}

impl ModelRequirement {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.path.as_os_str().is_empty(),
            "model path cannot be empty"
        );
        ensure!(
            self.repository
                .split_once('/')
                .is_some_and(|(owner, name)| {
                    !owner.trim().is_empty() && !name.trim().is_empty()
                }),
            "model repository must use owner/name form"
        );
        ensure!(
            is_lower_hex(&self.revision, 40),
            "model revision must be a lowercase 40-character commit hash"
        );
        ensure!(
            !self.filename.trim().is_empty()
                && Path::new(&self.filename)
                    .file_name()
                    .and_then(|name| name.to_str())
                    == Some(self.filename.as_str()),
            "model filename must be one basename"
        );
        ensure!(
            self.filename.to_ascii_lowercase().ends_with(".gguf"),
            "scoreboard model filename must end in .gguf"
        );
        ensure!(self.size_bytes > 0, "model size_bytes must be positive");
        ensure!(
            is_lower_hex(&self.sha256, 64),
            "model sha256 must be a lowercase 64-character digest"
        );
        ensure!(
            !self.architecture.trim().is_empty(),
            "model architecture cannot be empty"
        );
        ensure!(
            !self.quantization.trim().is_empty(),
            "model quantization cannot be empty"
        );
        Ok(())
    }
}

/// Pinned `llama.cpp` toolchain and execution-policy requirements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaCppRequirement {
    pub binary_directory: PathBuf,
    /// Optional clean Git checkout used to build a local CMake toolchain.
    /// When omitted, the benchmark requires a Homebrew installation receipt.
    #[serde(default)]
    pub source_directory: Option<PathBuf>,
    pub build: u32,
    pub commit: String,
    pub gpu_layers: i32,
    pub flash_attention: LlamaFlashAttention,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub mmap: bool,
}

impl LlamaCppRequirement {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.binary_directory.as_os_str().is_empty(),
            "llama.cpp binary_directory cannot be empty"
        );
        ensure!(
            self.source_directory
                .as_ref()
                .is_none_or(|directory| !directory.as_os_str().is_empty()),
            "llama.cpp source_directory cannot be empty when provided"
        );
        ensure!(self.build > 0, "llama.cpp build must be positive");
        ensure!(
            is_lower_hex(&self.commit, 40),
            "llama.cpp commit must be a lowercase 40-character hash"
        );
        ensure!(
            !self.cache_type_k.trim().is_empty() && !self.cache_type_v.trim().is_empty(),
            "llama.cpp K/V cache types cannot be empty"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlamaFlashAttention {
    On,
    Off,
    Auto,
}

/// Prompt, generation, sampling, and statistical shape shared by both sides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadSpec {
    pub prompt: PromptRequirement,
    /// Timed single-token decode forwards. Kernel runs emit one additional,
    /// unreported token so this count is not confused with first-token
    /// prefill sampling; parity and serving still retain exactly this many
    /// completion token IDs. Stop tokens cannot shorten either boundary.
    pub decode_steps: usize,
    pub sampling: GreedySamplingSpec,
    pub concurrency: Vec<usize>,
    pub warmup_waves: usize,
    pub samples_per_concurrency: usize,
    pub prefix_cache: PrefixCacheMode,
}

/// Dedicated workload declaration for the generated exact-prefix repository
/// prompts used by `demo fair-prefix`. The generic scoreboard prompt remains a
/// separate compatibility workload and is never substituted silently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairPrefixManifestSpec {
    pub prompt_source: String,
    pub exact_shared_prefix_tokens: usize,
    pub concurrency: usize,
    pub completion_tokens_per_request: usize,
    pub measured_waves_per_implementation: usize,
    pub process_lifetimes_per_implementation: usize,
    pub process_order: Vec<String>,
    pub warmup_waves_per_lifetime: usize,
    pub post_seed_stabilization_ms: u64,
    pub cache_prompt: bool,
}

impl FairPrefixManifestSpec {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.prompt_source == "prepared_repository_bundle",
            "fair-prefix prompt_source must be prepared_repository_bundle"
        );
        ensure!(
            self.exact_shared_prefix_tokens > 0
                && self.concurrency > 0
                && self.completion_tokens_per_request > 0
                && self.measured_waves_per_implementation >= MIN_PERCENTILE_SAMPLES
                && self.process_lifetimes_per_implementation > 0
                && self.warmup_waves_per_lifetime > 0
                && self.post_seed_stabilization_ms > 0
                && self.cache_prompt,
            "fair-prefix manifest contains a zero, undersampled, or disabled-cache contract"
        );
        ensure!(
            self.measured_waves_per_implementation
                .is_multiple_of(self.process_lifetimes_per_implementation),
            "fair-prefix waves must split evenly across process lifetimes"
        );
        ensure!(
            self.process_order.len()
                == self
                    .process_lifetimes_per_implementation
                    .checked_mul(2)
                    .ok_or_else(|| anyhow::anyhow!("fair-prefix process count overflow"))?,
            "fair-prefix process order must contain one lifetime per implementation in every pair"
        );
        ensure!(
            self.process_order
                .iter()
                .all(|implementation| implementation == "macllm" || implementation == "llama.cpp"),
            "fair-prefix process order contains an unknown implementation"
        );
        ensure!(
            self.process_order.chunks_exact(2).all(|pair| {
                (pair[0] == "macllm" && pair[1] == "llama.cpp")
                    || (pair[0] == "llama.cpp" && pair[1] == "macllm")
            }),
            "each fair-prefix process pair must contain one MacLLM and one llama.cpp lifetime"
        );
        Ok(())
    }
}

impl WorkloadSpec {
    fn validate(&self) -> Result<()> {
        self.prompt.validate()?;
        ensure!(self.decode_steps > 0, "decode_steps must be positive");
        self.sampling.validate()?;
        ensure!(
            self.concurrency.as_slice() == REQUIRED_CONCURRENCY,
            "scoreboard concurrency must be exactly [1, 4, 8]"
        );
        ensure!(self.warmup_waves > 0, "warmup_waves must be positive");
        ensure!(
            self.samples_per_concurrency >= MIN_PERCENTILE_SAMPLES,
            "samples_per_concurrency must be at least {MIN_PERCENTILE_SAMPLES}"
        );
        ensure!(
            REQUIRED_CONCURRENCY
                .into_iter()
                .all(|width| self.samples_per_concurrency.is_multiple_of(width)),
            "samples_per_concurrency must form complete waves for concurrency 1, 4, and 8"
        );
        Ok(())
    }
}

/// Immutable prompt bytes and their independently verified tokenization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptRequirement {
    pub path: PathBuf,
    pub sha256: String,
    pub token_count: usize,
    /// SHA-256 over every token ID encoded as four little-endian bytes.
    pub token_ids_sha256: String,
    pub add_special_tokens: bool,
    /// Remove exactly one final LF from the verified UTF-8 file before
    /// tokenization. This makes the text-file transport newline explicit
    /// instead of silently changing the benchmark prompt.
    #[serde(default)]
    pub strip_final_newline: bool,
}

impl PromptRequirement {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.path.as_os_str().is_empty(),
            "prompt path cannot be empty"
        );
        ensure!(
            is_lower_hex(&self.sha256, 64),
            "prompt sha256 must be a lowercase 64-character digest"
        );
        ensure!(self.token_count > 0, "prompt token_count must be positive");
        ensure!(
            is_lower_hex(&self.token_ids_sha256, 64),
            "prompt token_ids_sha256 must be a lowercase 64-character digest"
        );
        Ok(())
    }

    pub(crate) fn prompt_text(&self, mut verified_file_text: String) -> Result<String> {
        if self.strip_final_newline {
            ensure!(
                verified_file_text.ends_with('\n'),
                "prompt strip_final_newline=true requires one final LF byte"
            );
            ensure!(
                !verified_file_text.ends_with("\r\n"),
                "prompt strip_final_newline supports LF, not CRLF"
            );
            verified_file_text.pop();
        }
        Ok(verified_file_text)
    }
}

/// Deliberately redundant greedy settings prevent implicit sampler defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GreedySamplingSpec {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub ignore_stop_tokens: bool,
}

impl GreedySamplingSpec {
    fn validate(self) -> Result<()> {
        ensure!(
            self.temperature.to_bits() == 0.0_f64.to_bits(),
            "greedy parity requires temperature=0"
        );
        ensure!(self.top_k == 0, "greedy parity requires top_k=0");
        ensure!(
            self.top_p.to_bits() == 1.0_f64.to_bits(),
            "greedy parity requires top_p=1"
        );
        ensure!(
            self.repetition_penalty.to_bits() == 1.0_f32.to_bits(),
            "greedy parity requires repetition_penalty=1"
        );
        ensure!(
            self.seed == GREEDY_SEED,
            "greedy parity requires seed={GREEDY_SEED}"
        );
        ensure!(
            self.ignore_stop_tokens,
            "greedy parity requires ignore_stop_tokens=true"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixCacheMode {
    Disabled,
    Cold,
    Warm,
}

/// Explicit `MacLLM` execution settings recorded beside `llama.cpp` settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBenchmarkSpec {
    pub device: BenchmarkDevice,
    pub dtype: BenchmarkDType,
    pub prefill_token_quantum: usize,
    pub kv_cache_block_tokens: usize,
    pub kv_cache_slab_blocks: usize,
    pub kv_cache_capacity_tokens: usize,
}

impl RuntimeBenchmarkSpec {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.prefill_token_quantum > 0,
            "prefill_token_quantum must be positive"
        );
        ensure!(
            self.kv_cache_block_tokens > 0,
            "kv_cache_block_tokens must be positive"
        );
        ensure!(
            self.kv_cache_slab_blocks > 0,
            "kv_cache_slab_blocks must be positive"
        );
        ensure!(
            self.kv_cache_capacity_tokens > 0,
            "kv_cache_capacity_tokens must be positive"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BenchmarkDevice {
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BenchmarkDType {
    F32,
    F16,
    Bf16,
}

/// Cryptographic identity of one regular file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileProvenance {
    pub canonical_path: PathBuf,
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

/// Download marker provenance, when the fixture came from `macllm download`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotProvenance {
    pub repository: String,
    pub revision: String,
    pub requested_revision: Option<String>,
    pub complete: bool,
}

/// Fully verified model identity embedded into every report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProvenance {
    pub file: FileProvenance,
    pub snapshot: Option<SnapshotProvenance>,
    pub architecture: String,
    pub quantization: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplementationKind {
    Macllm,
    LlamaCpp,
}

/// Executable and source identity of one measured implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImplementationProvenance {
    pub implementation: ImplementationKind,
    pub version: String,
    pub commit: String,
    pub executable: FileProvenance,
    pub dirty: Option<bool>,
}

/// Complete single-request `MacLLM` timing, separate from HTTP serving.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelResult {
    pub implementation: ImplementationKind,
    pub measurement_boundary: String,
    pub command: Vec<String>,
    pub prompt_tokens: usize,
    pub decode_steps: usize,
    pub prefill_tokens_per_second_samples: Vec<f64>,
    pub decode_tokens_per_second_samples: Vec<f64>,
    pub time_to_first_token_ms_samples: Vec<f64>,
    pub prefill_tokens_per_second: DistributionSummary,
    pub decode_tokens_per_second: DistributionSummary,
    pub time_to_first_token_ms: DistributionSummary,
}

/// Throughput-only kernel aggregates emitted by pinned `llama-bench` runs.
///
/// This deliberately has no TTFT field: `llama-bench` uses synthetic token
/// IDs and does not expose a request time-to-first-token boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaBenchThroughputResult {
    pub implementation: ImplementationKind,
    pub measurement_boundary: String,
    pub command: Vec<String>,
    pub prompt_tokens: usize,
    pub decode_steps: usize,
    pub prefill_tokens_per_second_samples: Vec<f64>,
    pub decode_tokens_per_second_samples: Vec<f64>,
    pub prefill_tokens_per_second: DistributionSummary,
    pub decode_tokens_per_second: DistributionSummary,
}

/// One externally observed serving request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestSample {
    pub wave: usize,
    pub request: usize,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub time_to_first_token_ms: f64,
    pub latency_ms: f64,
}

/// Aggregate and raw evidence for one implementation/concurrency pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServingResult {
    pub implementation: ImplementationKind,
    pub concurrency: usize,
    pub request_count: usize,
    pub total_completion_tokens: u64,
    pub wall_time_ms: f64,
    pub aggregate_output_tokens_per_second: f64,
    pub time_to_first_token_ms: DistributionSummary,
    pub latency_ms: DistributionSummary,
    pub samples: Vec<RequestSample>,
}

/// Raw internal timing evidence from one `llama-server` request.
///
/// These values come from the terminal server event and are not external
/// request latency or TTFT measurements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerTimingSample {
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

/// Aggregate and raw internal `llama-server` timings for one concurrency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerTimingResult {
    pub implementation: ImplementationKind,
    pub concurrency: usize,
    pub measurement_boundary: String,
    pub command: Vec<String>,
    pub prompt_ms: DistributionSummary,
    pub prompt_tokens_per_second: DistributionSummary,
    pub predicted_ms: DistributionSummary,
    pub predicted_tokens_per_second: DistributionSummary,
    pub samples: Vec<ServerTimingSample>,
}

/// Process-scoped memory measurements for one implementation/concurrency pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryResult {
    pub implementation: ImplementationKind,
    pub concurrency: usize,
    pub measurement_boundary: String,
    pub loaded_steady_bytes: Option<u64>,
    pub peak_physical_footprint_bytes: Option<u64>,
    pub peak_resident_bytes: Option<u64>,
    pub sampling_period_ms: u64,
}

/// Full-sequence greedy-token comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParityResult {
    pub prompt_tokens: usize,
    pub prompt_token_ids_sha256: String,
    pub macllm_token_ids: Vec<u32>,
    pub llama_cpp_token_ids: Vec<u32>,
    pub exact: bool,
    pub first_mismatch: Option<usize>,
}

/// Self-contained report; raw samples and full commands remain auditable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreboardReport {
    pub schema_version: u32,
    pub run_id: String,
    pub recorded_at_utc: String,
    pub host: SanitizedHost,
    pub model: ModelProvenance,
    pub implementations: Vec<ImplementationProvenance>,
    pub workload: WorkloadSpec,
    pub kernel_results: Vec<KernelResult>,
    pub llama_bench_throughput: LlamaBenchThroughputResult,
    pub serving_results: Vec<ServingResult>,
    pub server_timing_results: Vec<ServerTimingResult>,
    pub memory_results: Vec<MemoryResult>,
    pub parity: ParityResult,
    pub warnings: Vec<String>,
}

impl ScoreboardReport {
    /// Validate report invariants before serialization or Markdown rendering.
    // Keep the complete publication gate in one linear audit trail.
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == SCOREBOARD_SCHEMA_VERSION,
            "unsupported scoreboard report schema version {}; expected {SCOREBOARD_SCHEMA_VERSION}",
            self.schema_version
        );
        ensure!(
            !self.run_id.trim().is_empty(),
            "report run_id cannot be empty"
        );
        ensure!(
            !self.recorded_at_utc.trim().is_empty(),
            "report recorded_at_utc cannot be empty"
        );
        self.host.validate()?;
        validate_model_provenance(&self.model)?;
        self.workload.validate()?;
        ensure!(
            self.implementations.len() == 2,
            "report must contain exactly MacLLM and llama.cpp provenance"
        );

        let mut implementations = BTreeSet::new();
        for implementation in &self.implementations {
            validate_implementation_provenance(implementation)?;
            ensure!(
                implementations.insert(implementation.implementation),
                "report contains duplicate implementation provenance"
            );
        }
        ensure!(
            implementations
                == BTreeSet::from([ImplementationKind::Macllm, ImplementationKind::LlamaCpp]),
            "report must identify both MacLLM and llama.cpp"
        );

        ensure!(
            self.kernel_results.len() == 1,
            "report must contain exactly one complete MacLLM kernel result"
        );
        for result in &self.kernel_results {
            ensure!(
                result.implementation == ImplementationKind::Macllm,
                "complete kernel result must be MacLLM; llama-bench has no TTFT"
            );
            validate_measurement_boundary(&result.measurement_boundary, "kernel")?;
            validate_command(&result.command, "kernel")?;
            ensure!(
                result.prompt_tokens == self.workload.prompt.token_count,
                "kernel prompt token count does not match the workload"
            );
            ensure!(
                result.decode_steps == self.workload.decode_steps,
                "kernel completion length does not match the workload"
            );
            for (summary, samples) in [
                (
                    &result.prefill_tokens_per_second,
                    result.prefill_tokens_per_second_samples.as_slice(),
                ),
                (
                    &result.decode_tokens_per_second,
                    result.decode_tokens_per_second_samples.as_slice(),
                ),
                (
                    &result.time_to_first_token_ms,
                    result.time_to_first_token_ms_samples.as_slice(),
                ),
            ] {
                summary.verify(samples)?;
                ensure!(
                    summary.sample_count == self.workload.samples_per_concurrency,
                    "kernel summary sample count does not match the workload"
                );
            }
        }

        ensure!(
            self.llama_bench_throughput.implementation == ImplementationKind::LlamaCpp,
            "llama-bench throughput result must identify llama.cpp"
        );
        validate_measurement_boundary(
            &self.llama_bench_throughput.measurement_boundary,
            "llama-bench throughput",
        )?;
        validate_command(
            &self.llama_bench_throughput.command,
            "llama-bench throughput",
        )?;
        ensure!(
            self.llama_bench_throughput.prompt_tokens == self.workload.prompt.token_count,
            "llama-bench prompt token count does not match the workload"
        );
        ensure!(
            self.llama_bench_throughput.decode_steps == self.workload.decode_steps,
            "llama-bench decode length does not match the workload"
        );
        for (summary, samples) in [
            (
                &self.llama_bench_throughput.prefill_tokens_per_second,
                self.llama_bench_throughput
                    .prefill_tokens_per_second_samples
                    .as_slice(),
            ),
            (
                &self.llama_bench_throughput.decode_tokens_per_second,
                self.llama_bench_throughput
                    .decode_tokens_per_second_samples
                    .as_slice(),
            ),
        ] {
            ensure!(
                samples.iter().all(|sample| *sample > 0.0),
                "llama-bench throughput samples must be positive"
            );
            summary.verify(samples)?;
            ensure!(
                summary.sample_count == self.workload.samples_per_concurrency,
                "llama-bench summary sample count does not match the workload"
            );
        }

        ensure!(
            self.serving_results.len() == implementations.len() * REQUIRED_CONCURRENCY.len(),
            "report must contain every implementation/concurrency serving result"
        );
        let mut serving_pairs = BTreeSet::new();
        for result in &self.serving_results {
            ensure!(
                REQUIRED_CONCURRENCY.contains(&result.concurrency),
                "serving result has unsupported concurrency {}",
                result.concurrency
            );
            ensure!(
                implementations.contains(&result.implementation),
                "serving result names an unknown implementation"
            );
            ensure!(
                serving_pairs.insert((result.implementation, result.concurrency)),
                "report contains duplicate serving results"
            );
            ensure!(
                result.request_count == result.samples.len(),
                "serving result request_count does not match raw samples"
            );
            ensure!(
                result.request_count == self.workload.samples_per_concurrency,
                "serving result request count does not match the workload"
            );
            ensure!(
                result.request_count.is_multiple_of(result.concurrency),
                "serving result does not contain complete request waves"
            );
            ensure!(
                result.wall_time_ms.is_finite() && result.wall_time_ms > 0.0,
                "serving result wall time must be finite and positive"
            );
            ensure!(
                result.aggregate_output_tokens_per_second.is_finite()
                    && result.aggregate_output_tokens_per_second > 0.0,
                "serving aggregate throughput must be finite and positive"
            );
            validate_serving_evidence(result, &self.workload)?;
        }
        let expected_pairs = implementations
            .iter()
            .flat_map(|implementation| {
                REQUIRED_CONCURRENCY
                    .into_iter()
                    .map(|concurrency| (*implementation, concurrency))
            })
            .collect::<BTreeSet<_>>();
        ensure!(
            serving_pairs == expected_pairs,
            "serving results do not cover the required concurrency matrix"
        );

        ensure!(
            self.memory_results.len() == expected_pairs.len(),
            "report must contain every implementation/concurrency memory result"
        );
        let mut memory_pairs = BTreeSet::new();
        for result in &self.memory_results {
            ensure!(
                expected_pairs.contains(&(result.implementation, result.concurrency)),
                "memory result names an unsupported implementation/concurrency pair"
            );
            ensure!(
                memory_pairs.insert((result.implementation, result.concurrency)),
                "report contains duplicate memory results"
            );
            validate_measurement_boundary(&result.measurement_boundary, "memory")?;
            ensure!(
                result.loaded_steady_bytes.is_some()
                    && result.peak_physical_footprint_bytes.is_some()
                    && result.peak_resident_bytes.is_some(),
                "memory result must contain loaded steady, peak physical footprint, and peak resident measurements"
            );
            ensure!(
                [
                    result.loaded_steady_bytes,
                    result.peak_physical_footprint_bytes,
                    result.peak_resident_bytes,
                ]
                .into_iter()
                .flatten()
                .all(|bytes| bytes > 0),
                "memory measurements must be positive when reported"
            );
            ensure!(
                result.sampling_period_ms > 0,
                "memory sampling period must be positive"
            );
        }
        ensure!(
            memory_pairs == expected_pairs,
            "memory results do not cover the required concurrency matrix"
        );

        ensure!(
            self.parity.prompt_tokens == self.workload.prompt.token_count,
            "parity prompt token count does not match the workload"
        );
        ensure!(
            self.parity.prompt_token_ids_sha256 == self.workload.prompt.token_ids_sha256,
            "parity prompt token fingerprint does not match the workload"
        );
        ensure!(
            self.parity.macllm_token_ids.len() == self.workload.decode_steps
                && self.parity.llama_cpp_token_ids.len() == self.workload.decode_steps,
            "parity outputs do not match the required completion length"
        );
        ensure!(
            self.parity.exact == (self.parity.macllm_token_ids == self.parity.llama_cpp_token_ids),
            "parity exact flag disagrees with recorded token IDs"
        );
        ensure!(
            self.parity.first_mismatch
                == first_mismatch(
                    &self.parity.macllm_token_ids,
                    &self.parity.llama_cpp_token_ids
                ),
            "parity first_mismatch disagrees with recorded token IDs"
        );

        ensure!(
            self.server_timing_results.len() == REQUIRED_CONCURRENCY.len(),
            "report must contain every llama-server internal timing width"
        );
        let mut server_timing_widths = BTreeSet::new();
        for result in &self.server_timing_results {
            ensure!(
                result.implementation == ImplementationKind::LlamaCpp,
                "server internal timing result must identify llama.cpp"
            );
            ensure!(
                REQUIRED_CONCURRENCY.contains(&result.concurrency),
                "server internal timing result has unsupported concurrency {}",
                result.concurrency
            );
            ensure!(
                server_timing_widths.insert(result.concurrency),
                "report contains duplicate server internal timing widths"
            );
            validate_measurement_boundary(&result.measurement_boundary, "server internal timing")?;
            validate_command(&result.command, "server internal timing")?;
            let serving = self
                .serving_results
                .iter()
                .find(|serving| {
                    serving.implementation == ImplementationKind::LlamaCpp
                        && serving.concurrency == result.concurrency
                })
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "server internal timing has no matching llama.cpp serving result"
                    )
                })?;
            validate_server_timing_evidence(result, serving, &self.workload, &self.parity)?;
        }
        ensure!(
            server_timing_widths == BTreeSet::from(REQUIRED_CONCURRENCY),
            "server internal timings do not cover concurrency 1, 4, and 8"
        );
        ensure!(
            self.warnings
                .iter()
                .all(|warning| !warning.trim().is_empty()),
            "report warnings cannot be empty strings"
        );
        Ok(())
    }
}

fn validate_measurement_boundary(boundary: &str, kind: &str) -> Result<()> {
    ensure!(
        !boundary.trim().is_empty(),
        "{kind} measurement boundary cannot be empty"
    );
    Ok(())
}

fn validate_command(command: &[String], kind: &str) -> Result<()> {
    ensure!(!command.is_empty(), "{kind} command cannot be empty");
    ensure!(
        command.iter().all(|argument| !argument.is_empty()),
        "{kind} command contains an empty argument"
    );
    Ok(())
}

fn validate_server_timing_evidence(
    result: &ServerTimingResult,
    serving: &ServingResult,
    workload: &WorkloadSpec,
    parity: &ParityResult,
) -> Result<()> {
    ensure!(
        result.samples.len() == workload.samples_per_concurrency,
        "server internal timing sample count does not match the workload"
    );
    ensure!(
        result.samples.len() == serving.samples.len(),
        "server internal timing sample count does not match external serving evidence"
    );

    let external_requests = serving
        .samples
        .iter()
        .map(|sample| (sample.wave, sample.request))
        .collect::<BTreeSet<_>>();
    let mut internal_requests = BTreeSet::new();
    let mut request_ids = BTreeSet::new();
    let mut waves = BTreeMap::<usize, usize>::new();
    let mut prompt_ms = Vec::with_capacity(result.samples.len());
    let mut prompt_tps = Vec::with_capacity(result.samples.len());
    let mut predicted_ms = Vec::with_capacity(result.samples.len());
    let mut predicted_tps = Vec::with_capacity(result.samples.len());
    for sample in &result.samples {
        ensure!(
            internal_requests.insert((sample.wave, sample.request)),
            "server internal timing contains a duplicate wave/request pair"
        );
        ensure!(
            request_ids.insert(sample.request),
            "server internal timing contains a duplicate request ID"
        );
        ensure!(
            sample.cache_tokens == 0,
            "server internal timing reports cached prompt tokens"
        );
        ensure!(
            sample.prompt_tokens == workload.prompt.token_count,
            "server internal timing prompt count does not match the workload"
        );
        ensure!(
            sample.predicted_tokens == workload.decode_steps,
            "server internal timing predicted count does not match the workload"
        );
        ensure!(
            sample.token_ids == parity.llama_cpp_token_ids,
            "server internal timing token IDs do not match llama.cpp parity evidence"
        );
        ensure!(
            [
                sample.prompt_ms,
                sample.prompt_tokens_per_second,
                sample.predicted_ms,
                sample.predicted_tokens_per_second,
            ]
            .into_iter()
            .all(|value| value.is_finite() && value > 0.0),
            "server internal timing values must be finite and positive"
        );
        ensure!(
            approximately_equal_reported_rate(
                sample.prompt_tokens_per_second,
                sample.prompt_tokens as f64 * 1_000.0 / sample.prompt_ms,
            ) && approximately_equal_reported_rate(
                sample.predicted_tokens_per_second,
                sample.predicted_tokens as f64 * 1_000.0 / sample.predicted_ms,
            ),
            "server internal timing rates do not match raw durations"
        );
        *waves.entry(sample.wave).or_default() += 1;
        prompt_ms.push(sample.prompt_ms);
        prompt_tps.push(sample.prompt_tokens_per_second);
        predicted_ms.push(sample.predicted_ms);
        predicted_tps.push(sample.predicted_tokens_per_second);
    }
    ensure!(
        internal_requests == external_requests,
        "server internal timing requests do not match external serving evidence"
    );
    ensure!(
        waves.len() == result.samples.len() / result.concurrency
            && waves.values().all(|count| *count == result.concurrency),
        "server internal timing samples do not form complete concurrency waves"
    );
    for (summary, samples) in [
        (&result.prompt_ms, prompt_ms.as_slice()),
        (&result.prompt_tokens_per_second, prompt_tps.as_slice()),
        (&result.predicted_ms, predicted_ms.as_slice()),
        (
            &result.predicted_tokens_per_second,
            predicted_tps.as_slice(),
        ),
    ] {
        summary.verify(samples)?;
        ensure!(
            summary.sample_count == workload.samples_per_concurrency,
            "server internal timing summary sample count does not match the workload"
        );
    }
    Ok(())
}

pub(crate) fn validate_model_provenance(model: &ModelProvenance) -> Result<()> {
    validate_file_provenance(&model.file)?;
    ensure!(
        !model.architecture.trim().is_empty(),
        "model architecture cannot be empty"
    );
    ensure!(
        !model.quantization.trim().is_empty(),
        "model quantization cannot be empty"
    );
    if let Some(snapshot) = &model.snapshot {
        ensure!(
            snapshot
                .repository
                .split_once('/')
                .is_some_and(|(owner, name)| {
                    !owner.trim().is_empty() && !name.trim().is_empty()
                }),
            "model snapshot repository is invalid"
        );
        ensure!(
            is_lower_hex(&snapshot.revision, 40),
            "model snapshot revision is not a full lowercase commit hash"
        );
        ensure!(snapshot.complete, "model snapshot is incomplete");
    }
    Ok(())
}

pub(crate) fn validate_implementation_provenance(
    implementation: &ImplementationProvenance,
) -> Result<()> {
    ensure!(
        !implementation.version.trim().is_empty(),
        "implementation version cannot be empty"
    );
    ensure!(
        is_lower_hex(&implementation.commit, 40),
        "implementation commit must be a full lowercase commit hash"
    );
    validate_file_provenance(&implementation.executable)
}

pub(crate) fn validate_file_provenance(file: &FileProvenance) -> Result<()> {
    ensure!(
        !file.canonical_path.as_os_str().is_empty(),
        "provenance path cannot be empty"
    );
    ensure!(
        !file.filename.trim().is_empty()
            && Path::new(&file.filename)
                .file_name()
                .and_then(|name| name.to_str())
                == Some(file.filename.as_str()),
        "provenance filename must be one basename"
    );
    ensure!(file.size_bytes > 0, "provenance file size must be positive");
    ensure!(
        is_lower_hex(&file.sha256, 64),
        "provenance SHA-256 must be a lowercase 64-character digest"
    );
    Ok(())
}

fn validate_serving_evidence(result: &ServingResult, workload: &WorkloadSpec) -> Result<()> {
    let mut completion_tokens = 0_u64;
    let mut waves = BTreeMap::<usize, usize>::new();
    let mut request_ids = BTreeSet::new();
    let mut ttft = Vec::with_capacity(result.samples.len());
    let mut latency = Vec::with_capacity(result.samples.len());
    for sample in &result.samples {
        ensure!(
            request_ids.insert(sample.request),
            "serving result contains a duplicate request ID"
        );
        ensure!(
            sample.prompt_tokens == workload.prompt.token_count,
            "serving sample prompt token count does not match the workload"
        );
        ensure!(
            sample.completion_tokens == workload.decode_steps,
            "serving sample completion length does not match the workload"
        );
        ensure!(
            sample.time_to_first_token_ms.is_finite()
                && sample.time_to_first_token_ms >= 0.0
                && sample.latency_ms.is_finite()
                && sample.latency_ms >= sample.time_to_first_token_ms,
            "serving sample timings are invalid"
        );
        completion_tokens = completion_tokens
            .checked_add(sample.completion_tokens as u64)
            .ok_or_else(|| anyhow::anyhow!("serving completion-token total overflow"))?;
        *waves.entry(sample.wave).or_default() += 1;
        ttft.push(sample.time_to_first_token_ms);
        latency.push(sample.latency_ms);
    }
    ensure!(
        waves.len() == result.request_count / result.concurrency
            && waves.values().all(|count| *count == result.concurrency),
        "serving samples do not form complete concurrency waves"
    );
    ensure!(
        completion_tokens == result.total_completion_tokens,
        "serving completion-token total does not match raw samples"
    );
    result.time_to_first_token_ms.verify(&ttft)?;
    result.latency_ms.verify(&latency)?;
    let calculated_throughput = completion_tokens as f64 * 1000.0 / result.wall_time_ms;
    ensure!(
        approximately_equal(
            calculated_throughput,
            result.aggregate_output_tokens_per_second
        ),
        "serving aggregate throughput does not match raw totals"
    );
    Ok(())
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= f64::EPSILON * scale * 16.0
}

fn approximately_equal_reported_rate(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= 1.0e-6 * scale
}

pub(crate) fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[must_use]
pub fn first_mismatch(left: &[u32], right: &[u32]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest() -> ScoreboardManifest {
        ScoreboardManifest {
            schema_version: SCOREBOARD_SCHEMA_VERSION,
            name: "smollm3-m4pro".to_owned(),
            host: HostRequirement {
                machine_model: "Mac16,7".to_owned(),
                chip: "Apple M4 Pro".to_owned(),
                memory_bytes: 48 * 1024 * 1024 * 1024,
                architecture: "arm64".to_owned(),
            },
            model: ModelRequirement {
                path: "models/smollm3/SmolLM3-Q4_K_M.gguf".into(),
                repository: "ggml-org/SmolLM3-3B-GGUF".to_owned(),
                revision: "4965cb60b150737b68a0408c36aeefb65078f894".to_owned(),
                filename: "SmolLM3-Q4_K_M.gguf".to_owned(),
                size_bytes: 1_915_305_312,
                sha256: "8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e"
                    .to_owned(),
                architecture: "smollm3".to_owned(),
                quantization: "Q4_K_M".to_owned(),
            },
            llama_cpp: LlamaCppRequirement {
                binary_directory: "/opt/llama/bin".into(),
                source_directory: None,
                build: 10_050,
                commit: "b15ca938ad00aa6b3ee6c2edda7363fd02826b18".to_owned(),
                gpu_layers: 99,
                flash_attention: LlamaFlashAttention::Auto,
                cache_type_k: "f16".to_owned(),
                cache_type_v: "f16".to_owned(),
                mmap: true,
            },
            workload: WorkloadSpec {
                prompt: PromptRequirement {
                    path: "benchmarks/prompts/smollm3.txt".into(),
                    sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .to_owned(),
                    token_count: 512,
                    token_ids_sha256:
                        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                            .to_owned(),
                    add_special_tokens: false,
                    strip_final_newline: false,
                },
                decode_steps: 128,
                sampling: GreedySamplingSpec {
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 1.0,
                    repetition_penalty: 1.0,
                    seed: GREEDY_SEED,
                    ignore_stop_tokens: true,
                },
                concurrency: REQUIRED_CONCURRENCY.to_vec(),
                warmup_waves: 1,
                samples_per_concurrency: 32,
                prefix_cache: PrefixCacheMode::Disabled,
            },
            runtime: RuntimeBenchmarkSpec {
                device: BenchmarkDevice::Metal,
                dtype: BenchmarkDType::F32,
                prefill_token_quantum: 128,
                kv_cache_block_tokens: 16,
                kv_cache_slab_blocks: 8,
                kv_cache_capacity_tokens: 8192,
            },
            fair_prefix: None,
        }
    }

    #[test]
    fn valid_manifest_is_accepted() {
        valid_manifest().validate().unwrap();
    }

    #[test]
    fn fair_prefix_manifest_requires_balanced_isolated_process_pairs() {
        let mut fair = FairPrefixManifestSpec {
            prompt_source: "prepared_repository_bundle".to_owned(),
            exact_shared_prefix_tokens: 16_384,
            concurrency: 8,
            completion_tokens_per_request: 32,
            measured_waves_per_implementation: 20,
            process_lifetimes_per_implementation: 4,
            process_order: [
                "llama.cpp",
                "macllm",
                "macllm",
                "llama.cpp",
                "macllm",
                "llama.cpp",
                "llama.cpp",
                "macllm",
            ]
            .map(str::to_owned)
            .to_vec(),
            warmup_waves_per_lifetime: 1,
            post_seed_stabilization_ms: 30_000,
            cache_prompt: true,
        };
        fair.validate().unwrap();

        fair.process_order[1] = "llama.cpp".to_owned();
        assert!(fair.validate().is_err());
    }

    #[test]
    fn prompt_transport_newline_is_only_removed_when_declared() {
        let mut prompt = valid_manifest().workload.prompt;
        assert_eq!(
            prompt.prompt_text("2 + 2 =\n".to_owned()).unwrap(),
            "2 + 2 =\n"
        );

        prompt.strip_final_newline = true;
        assert_eq!(
            prompt.prompt_text("2 + 2 =\n".to_owned()).unwrap(),
            "2 + 2 ="
        );
        assert!(prompt.prompt_text("2 + 2 =".to_owned()).is_err());
        assert!(prompt.prompt_text("2 + 2 =\r\n".to_owned()).is_err());
    }

    #[test]
    fn manifest_rejects_schema_drift_and_unknown_fields() {
        let mut manifest = valid_manifest();
        manifest.schema_version += 1;
        assert!(manifest.validate().is_err());

        let mut value = serde_json::to_value(valid_manifest()).unwrap();
        value["future_field"] = serde_json::json!(true);
        assert!(serde_json::from_value::<ScoreboardManifest>(value).is_err());
    }

    #[test]
    fn workload_requires_exact_concurrency_and_complete_p95_waves() {
        let mut manifest = valid_manifest();
        manifest.workload.concurrency = vec![1, 2, 4, 8];
        assert!(manifest.validate().is_err());

        manifest.workload.concurrency = REQUIRED_CONCURRENCY.to_vec();
        manifest.workload.samples_per_concurrency = MIN_PERCENTILE_SAMPLES - 1;
        assert!(manifest.validate().is_err());

        manifest.workload.samples_per_concurrency = 20;
        assert!(manifest.validate().is_err());

        manifest.workload.samples_per_concurrency = 24;
        manifest.validate().unwrap();
    }

    #[test]
    fn workload_rejects_non_greedy_or_early_stop_settings() {
        let mut manifest = valid_manifest();
        manifest.workload.sampling.temperature = 0.1;
        assert!(manifest.validate().is_err());

        manifest.workload.sampling.temperature = 0.0;
        manifest.workload.sampling.repetition_penalty = 1.1;
        assert!(manifest.validate().is_err());

        manifest.workload.sampling.repetition_penalty = 1.0;
        manifest.workload.sampling.ignore_stop_tokens = false;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn model_and_llama_identities_must_be_full_lowercase_hashes() {
        let mut manifest = valid_manifest();
        manifest.model.sha256.make_ascii_uppercase();
        assert!(manifest.validate().is_err());

        let mut manifest = valid_manifest();
        manifest.llama_cpp.commit.truncate(8);
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn first_mismatch_handles_values_and_different_lengths() {
        assert_eq!(first_mismatch(&[1, 2, 3], &[1, 9, 3]), Some(1));
        assert_eq!(first_mismatch(&[1, 2], &[1, 2, 3]), Some(2));
        assert_eq!(first_mismatch(&[1, 2], &[1, 2]), None);
    }
}
