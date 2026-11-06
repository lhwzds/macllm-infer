//! Validated JSON and Markdown artifacts for the exact-16K prefix-cache demo.

use std::{
    fmt::Write as _,
    fs,
    io::Write as _,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use super::prefix_demo_mac::{MAC_PREFIX_DEMO_MIN_MEASURED_WAVES, derive_macllm_distributions};
use super::schema::{validate_implementation_provenance, validate_model_provenance};
use super::stats::summarize;
use super::{
    DEFAULT_SHARED_PREFIX_TOKENS, ImplementationKind, ImplementationProvenance,
    LlamaPrefixDemoExecution, MAC_PREFIX_DEMO_CONCURRENCY, MAC_PREFIX_DEMO_DECODE_TOKENS,
    MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET, MacPrefixDemoExecution, ModelProvenance,
    PREFIX_DEMO_CONCURRENCY, PREFIX_DEMO_PREFILL_QUANTUM, PREFIX_DEMO_TASK_COUNT,
    PreparedPrefixDemo, SanitizedHost, VerifiedScoreboard, sha256_bytes, token_ids_sha256,
};

pub const PREFIX_DEMO_REPORT_SCHEMA_VERSION: u32 = 4;

const CACHE_SEMANTICS_BOUNDARY: &str = concat!(
    "Every MacLLM wave retains one post-attach, pre-forward atomic physical KV-page/refcount sample. ",
    "The active-request delta is computed against the quiescent warmed seed and must show ",
    "eight references per 16-token prefix page with zero newly allocated live blocks. ",
    "llama.cpp cache_n is retained only as a logical server counter and is never converted ",
    "into MacLLM page references, byte savings, or a physical-reuse ratio."
);

/// Values derived solely from the two complete execution evidence trees.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixDemoComparisonEligibility {
    pub eligible: bool,
    pub minimum_measured_waves_per_implementation: usize,
    pub macllm_measured_waves: usize,
    pub llama_cpp_measured_waves: usize,
    pub macllm_exact_prefix_cache_hit_requests_by_wave: Vec<usize>,
    pub llama_cpp_exact_prefix_cache_hit_requests_by_wave: Vec<usize>,
    pub exact_input_token_digest_parity: bool,
    pub exact_completion_token_count_parity: bool,
    pub exact_cross_implementation_output_parity: bool,
    pub deterministic_greedy_replay_parity: bool,
    pub llama_cpp_measured_stream_matches_replay: bool,
    pub macllm_kv_cache_dtype: String,
    pub llama_cpp_k_cache_dtype: Option<String>,
    pub llama_cpp_v_cache_dtype: Option<String>,
    pub kv_cache_dtype_parity: bool,
    pub transport_parity: bool,
    pub reasons: Vec<String>,
}

/// Values derived solely from the two complete execution evidence trees.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixDemoComparison {
    pub exact_common_prefix_tokens: usize,
    pub concurrency: usize,
    pub completion_tokens_per_request: usize,
    /// Representative-wave `MacLLM` measured text versus llama.cpp's separate
    /// non-stream correctness replay. Eligibility checks every retained wave.
    pub exact_correctness_replay_output_match_per_request: Vec<bool>,
    pub exact_correctness_replay_output_match: bool,
    pub llama_cpp_stream_matches_correctness_replay_per_request: Vec<bool>,
    pub llama_cpp_stream_matches_correctness_replay_requests: usize,
    /// Cache-hit count in the representative `MacLLM` wave used for the
    /// one-wave llama.cpp comparison. Every retained `MacLLM` wave is validated
    /// independently against the same invariant.
    pub macllm_representative_wave_cache_hit_requests: usize,
    pub macllm_representative_wave_matched_prefix_tokens: u64,
    pub macllm_first_admission_shared_blocks: usize,
    pub macllm_first_admission_page_references: usize,
    pub macllm_first_admission_physical_blocks: usize,
    pub macllm_first_admission_resident_snapshot_blocks: usize,
    pub macllm_active_request_prefix_page_references: usize,
    pub macllm_active_request_prefix_page_fanout: f64,
    pub macllm_active_request_prefix_reuse_coverage: f64,
    pub macllm_new_physical_live_blocks_at_admission: usize,
    pub macllm_atomic_pool_owner_density: f64,
    pub llama_cpp_requests_reporting_at_least_endpoint: usize,
    pub llama_cpp_minimum_server_reported_cache_n: usize,
    pub llama_cpp_maximum_server_reported_cache_n: usize,
    /// `MacLLM` p50 across retained wave-level aggregate-throughput samples.
    pub macllm_aggregate_output_tokens_per_second: f64,
    /// llama.cpp p50 across retained measured wave-level throughput samples.
    pub llama_cpp_aggregate_output_tokens_per_second: f64,
    pub eligibility: PrefixDemoComparisonEligibility,
    /// Present only when the strict comparison eligibility contract passes.
    pub macllm_to_llama_cpp_aggregate_throughput_ratio: Option<f64>,
    pub cache_semantics_boundary: String,
}

/// Complete, self-validating evidence for one isolated MacLLM/llama.cpp run.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixDemoReport {
    pub schema_version: u32,
    pub recorded_at_unix_millis: u64,
    pub profile: String,
    pub manifest_path: PathBuf,
    pub host: SanitizedHost,
    pub model: ModelProvenance,
    pub macllm_implementation: ImplementationProvenance,
    pub prepared: PreparedPrefixDemo,
    pub macllm: MacPrefixDemoExecution,
    pub llama_cpp: LlamaPrefixDemoExecution,
    pub comparison: PrefixDemoComparison,
    pub warnings: Vec<String>,
}

impl PrefixDemoReport {
    /// Recompute all publication-critical summaries from raw evidence.
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == PREFIX_DEMO_REPORT_SCHEMA_VERSION,
            "unsupported prefix-demo report schema {}",
            self.schema_version
        );
        ensure!(
            self.recorded_at_unix_millis > 0,
            "prefix-demo timestamp is zero"
        );
        ensure!(!self.profile.trim().is_empty(), "profile cannot be empty");
        ensure!(
            self.manifest_path.is_absolute(),
            "prefix-demo manifest path must be absolute"
        );
        self.host.validate()?;
        validate_model_provenance(&self.model)?;
        validate_implementation_provenance(&self.macllm_implementation)?;
        ensure!(
            self.macllm_implementation.implementation == ImplementationKind::Macllm,
            "prefix-demo executable provenance is not MacLLM"
        );
        ensure!(
            self.macllm_implementation.dirty == Some(false),
            "publishable prefix-demo evidence requires a clean MacLLM source commit"
        );

        self.prepared.validate()?;
        ensure!(
            self.prepared.requested_shared_tokens == DEFAULT_SHARED_PREFIX_TOKENS,
            "public prefix demo must retain the exact 16K endpoint"
        );
        ensure!(
            self.prepared.model_file == self.model.file,
            "prepared model file differs from report provenance"
        );
        validate_execution_identity(self)?;
        validate_macllm_execution(self)?;
        validate_llama_execution(self)?;
        validate_request_identity(self)?;
        validate_comparison(self)?;
        Ok(())
    }
}

/// Assemble and immediately validate a publishable comparison report.
pub fn assemble_prefix_demo_report(
    verified: &VerifiedScoreboard,
    macllm_implementation: ImplementationProvenance,
    prepared: PreparedPrefixDemo,
    macllm: MacPrefixDemoExecution,
    llama_cpp: LlamaPrefixDemoExecution,
) -> Result<PrefixDemoReport> {
    verified.manifest.validate()?;
    ensure!(
        verified.manifest_path.is_absolute(),
        "verified manifest path must be absolute"
    );
    ensure!(
        prepared.model_file == verified.model.file,
        "prepared prefix-demo model differs from verified manifest model"
    );
    ensure!(
        llama_cpp.provenance.llama_cpp_version == verified.llama_cpp_version
            && llama_cpp.provenance.llama_cpp_full_commit == verified.llama_cpp_commit
            && llama_cpp.provenance.llama_server == verified.llama_server
            && llama_cpp.provenance.model == verified.model,
        "llama.cpp execution provenance differs from verified manifest inputs"
    );
    validate_implementation_provenance(&macllm_implementation)?;
    ensure!(
        macllm_implementation.implementation == ImplementationKind::Macllm
            && macllm_implementation.dirty == Some(false),
        "prefix-demo publication requires one clean MacLLM executable/source identity"
    );

    let comparison = derive_comparison(&prepared, &macllm, &llama_cpp)?;
    let llama_measured_requests = llama_cpp
        .measured_waves
        .len()
        .checked_mul(PREFIX_DEMO_TASK_COUNT)
        .context("llama.cpp measured request count overflow")?;
    let stream_matches = llama_cpp
        .measured_waves
        .iter()
        .flat_map(|wave| &wave.requests)
        .filter(|request| request.streamed_content == request.content)
        .count();
    let stream_mismatches = llama_measured_requests
        .checked_sub(stream_matches)
        .context("stream/replay match count exceeds measured request count")?;
    let correctness_mismatches = comparison
        .exact_correctness_replay_output_match_per_request
        .iter()
        .filter(|exact| !**exact)
        .count();
    let mut warnings = vec![
        format!(
            "MacLLM percentiles use {} retained consecutive eight-request waves; llama.cpp percentiles use {} retained consecutive eight-request waves.",
            macllm.waves.len(),
            llama_cpp.measured_waves.len()
        ),
        concat!(
            "MacLLM output correctness uses representative wave 0 after every retained MacLLM ",
            "wave is proven identical. llama.cpp correctness uses its separate deterministic non-stream ",
            "correctness replay; it is not a same-wave token-parity claim."
        )
        .to_owned(),
    ];
    if stream_mismatches > 0 {
        warnings.push(format!(
            "llama.cpp measured streaming text differed from its correctness replay for {stream_mismatches}/{llama_measured_requests} requests; both values remain in the JSON"
        ));
    }
    if correctness_mismatches > 0 {
        warnings.push(format!(
            "MacLLM measured text differed from llama.cpp's deterministic correctness replay for {correctness_mismatches}/{PREFIX_DEMO_TASK_COUNT} requests; the report preserves each false comparison instead of suppressing the artifact"
        ));
    }
    let recorded_at_unix_millis = u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system time precedes the Unix epoch")?
            .as_millis(),
    )
    .context("prefix-demo timestamp exceeds u64")?;
    let report = PrefixDemoReport {
        schema_version: PREFIX_DEMO_REPORT_SCHEMA_VERSION,
        recorded_at_unix_millis,
        profile: verified.manifest.name.clone(),
        manifest_path: verified.manifest_path.clone(),
        host: verified.host.clone(),
        model: verified.model.clone(),
        macllm_implementation,
        prepared,
        macllm,
        llama_cpp,
        comparison,
        warnings,
    };
    report.validate()?;
    Ok(report)
}

/// Render a concise view; the JSON remains the authoritative evidence artifact.
pub fn render_prefix_demo_markdown(report: &PrefixDemoReport) -> Result<String> {
    report.validate()?;
    let json = serialized_json(report)?;
    render_prefix_demo_markdown_with_json_sha(report, &sha256_bytes(&json))
}

#[allow(clippy::too_many_lines)]
fn render_prefix_demo_markdown_with_json_sha(
    report: &PrefixDemoReport,
    json_sha256: &str,
) -> Result<String> {
    let comparison = &report.comparison;
    let reference_wave = report
        .macllm
        .waves
        .first()
        .context("MacLLM report has no representative wave")?;
    let distributions = &report.macllm.distributions;
    let llama_aggregate_distribution = summarize(
        &report
            .llama_cpp
            .measured_waves
            .iter()
            .map(|wave| wave.aggregate_output_tokens_per_second)
            .collect::<Vec<_>>(),
    )?;
    let llama_wave_max_ttft_distribution = summarize(
        &report
            .llama_cpp
            .measured_waves
            .iter()
            .map(|wave| wave.external_time_to_first_token_ms.max)
            .collect::<Vec<_>>(),
    )?;
    let correctness_matches = comparison
        .exact_correctness_replay_output_match_per_request
        .iter()
        .filter(|exact| **exact)
        .count();
    let mut output = String::new();
    output.push_str("# MacLLM exact-16K coding-agent prefix demo\n\n");
    writeln!(
        output,
        "> JSON companion SHA-256: `{json_sha256}`. MacLLM percentiles use {} retained eight-request waves; llama.cpp percentiles use {} retained eight-request waves plus separate non-stream correctness replays.\n",
        report.macllm.waves.len(),
        report.llama_cpp.measured_waves.len()
    )?;
    output.push_str("## Provenance\n\n");
    output.push_str("| Field | Value |\n|---|---|\n");
    writeln!(output, "| Profile | `{}` |", markdown_cell(&report.profile))?;
    writeln!(output, "| Host | {} |", markdown_cell(&report.host.chip))?;
    writeln!(
        output,
        "| Model | `{}` / `{}` |",
        markdown_cell(&report.model.file.filename),
        report.model.file.sha256
    )?;
    writeln!(
        output,
        "| MacLLM source | `{}` (clean) |",
        report.macllm_implementation.commit
    )?;
    writeln!(
        output,
        "| MacLLM executable | `{}` |",
        report.macllm_implementation.executable.sha256
    )?;
    writeln!(
        output,
        "| llama.cpp source | `{}` |",
        report.llama_cpp.provenance.llama_cpp_full_commit
    )?;

    output.push_str("\n## Result\n\n");
    output.push_str("| Evidence | MacLLM | llama.cpp |\n|---|---:|---:|\n");
    writeln!(
        output,
        "| Exact shared prefix | {} tokens | {} tokens |",
        comparison.exact_common_prefix_tokens, comparison.exact_common_prefix_tokens
    )?;
    writeln!(
        output,
        "| Simultaneous requests | {} | {} |",
        comparison.concurrency, comparison.concurrency
    )?;
    writeln!(
        output,
        "| Aggregate output | p50 {:.3} / p95 {:.3} tok/s | p50 {:.3} / p95 {:.3} tok/s |",
        distributions.wave_aggregate_output_tokens_per_second.p50,
        distributions.wave_aggregate_output_tokens_per_second.p95,
        llama_aggregate_distribution.p50,
        llama_aggregate_distribution.p95
    )?;
    writeln!(
        output,
        "| Wave-max TTFT | p50 {:.3} / p95 {:.3} ms | p50 {:.3} / p95 {:.3} ms |",
        distributions.wave_max_time_to_first_token_ms.p50,
        distributions.wave_max_time_to_first_token_ms.p95,
        llama_wave_max_ttft_distribution.p50,
        llama_wave_max_ttft_distribution.p95
    )?;
    writeln!(
        output,
        "| Representative-wave requests reporting endpoint reuse | {} physical cache hits | {} logical cache_n observations |",
        comparison.macllm_representative_wave_cache_hit_requests,
        comparison.llama_cpp_requests_reporting_at_least_endpoint
    )?;
    writeln!(
        output,
        "| Active-request physical prefix fan-out | {:.3}x | not exposed |",
        comparison.macllm_active_request_prefix_page_fanout
    )?;
    writeln!(
        output,
        "| MacLLM measured text = llama correctness replay | {correctness_matches}/{PREFIX_DEMO_TASK_COUNT} | {correctness_matches}/{PREFIX_DEMO_TASK_COUNT} |"
    )?;
    writeln!(
        output,
        "| llama measured stream = llama correctness replay | n/a | {}/{} |",
        comparison.llama_cpp_stream_matches_correctness_replay_requests, PREFIX_DEMO_TASK_COUNT
    )?;

    output.push_str("\n## Physical cache evidence (MacLLM only)\n\n");
    output.push_str("| Field | Value |\n|---|---:|\n");
    writeln!(
        output,
        "| Consecutive waves proving the same exact endpoint | {} |",
        report.macllm.waves.len()
    )?;
    writeln!(
        output,
        "| First-admission active sequences | {} |",
        reference_wave
            .cache
            .eight_way_first_admission_snapshot
            .active_sequences
    )?;
    writeln!(
        output,
        "| First-admission physical live blocks | {} |",
        comparison.macllm_first_admission_physical_blocks
    )?;
    writeln!(
        output,
        "| First-admission physical live references | {} |",
        comparison.macllm_first_admission_page_references
    )?;
    writeln!(
        output,
        "| First-admission physically shared blocks | {} |",
        comparison.macllm_first_admission_shared_blocks
    )?;
    writeln!(
        output,
        "| Active-request prefix page references | {} |",
        comparison.macllm_active_request_prefix_page_references
    )?;
    writeln!(
        output,
        "| Active-request prefix reuse coverage | {:.3}% |",
        comparison.macllm_active_request_prefix_reuse_coverage * 100.0
    )?;
    writeln!(
        output,
        "| New physical live blocks at admission | {} |",
        comparison.macllm_new_physical_live_blocks_at_admission
    )?;
    writeln!(
        output,
        "| Atomic pool owner density (diagnostic only) | {:.3}x |",
        comparison.macllm_atomic_pool_owner_density
    )?;

    output.push_str("\n## MacLLM benchmark-only phases\n\n");
    output.push_str("Each percentile has one total per retained eight-request wave. Packed model time is recorded once per forward, not once per lane. These mutually exclusive sampled spans are not an exhaustive decomposition of external wall time.\n\n");
    output.push_str("| Exclusive phase | p50 (ms) | p95 (ms) |\n|---|---:|---:|\n");
    for (label, summary) in [
        ("Queue/admission", &distributions.queue_admission_ms),
        (
            "Prefix lookup/attach",
            &distributions.prefix_lookup_attach_ms,
        ),
        ("Cache diagnostics", &distributions.cache_diagnostics_ms),
        ("Remaining suffix prefill", &distributions.suffix_prefill_ms),
        (
            "First-token-producing model forward",
            &distributions.first_token_model_forward_ms,
        ),
        (
            "Single-step decode total",
            &distributions.single_step_decode_total_ms,
        ),
    ] {
        writeln!(
            output,
            "| {label} | {:.3} | {:.3} |",
            summary.p50, summary.p95
        )?;
    }

    output.push_str("\n## Comparison boundary\n\n");
    writeln!(output, "{}\n", comparison.cache_semantics_boundary)?;
    render_comparison_eligibility(
        &mut output,
        &comparison.eligibility,
        comparison.macllm_to_llama_cpp_aggregate_throughput_ratio,
    )?;
    output.push_str("## Warnings\n\n");
    for warning in report.warnings.iter().chain(&report.llama_cpp.warnings) {
        writeln!(output, "- {warning}")?;
    }
    Ok(output)
}

fn render_comparison_eligibility(
    output: &mut String,
    eligibility: &PrefixDemoComparisonEligibility,
    ratio: Option<f64>,
) -> Result<()> {
    if eligibility.eligible {
        let ratio = ratio.context("eligible comparison is missing its throughput ratio")?;
        writeln!(
            output,
            "Eligible aggregate-throughput comparison ({} measured waves per implementation): **{ratio:.3}x**.\n",
            eligibility.macllm_measured_waves
        )?;
        return Ok(());
    }

    ensure!(
        ratio.is_none(),
        "ineligible comparison unexpectedly contains a throughput ratio"
    );
    output.push_str(
        "**Not comparison-eligible.** No aggregate-throughput speed ratio is published.\n\n",
    );
    output.push_str("Eligibility reasons:\n\n");
    for reason in &eligibility.reasons {
        writeln!(output, "- {reason}")?;
    }
    output.push('\n');
    Ok(())
}

/// Serialize a validated, hash-linked JSON/Markdown pair in one directory.
pub fn write_prefix_demo_artifacts(
    report: &PrefixDemoReport,
    json_path: impl AsRef<Path>,
    markdown_path: impl AsRef<Path>,
) -> Result<()> {
    report.validate()?;
    let (json_path, markdown_path) =
        normalize_artifact_pair(json_path.as_ref(), markdown_path.as_ref())?;
    let json = serialized_json(report)?;
    let markdown = render_prefix_demo_markdown_with_json_sha(report, &sha256_bytes(&json))?;
    let json_temp = prepare_atomic_file(&json_path, &json)?;
    let markdown_temp = prepare_atomic_file(&markdown_path, markdown.as_bytes())?;
    commit_artifact_pair(json_temp, &json_path, markdown_temp, &markdown_path)
}

fn validate_execution_identity(report: &PrefixDemoReport) -> Result<()> {
    ensure!(
        report.macllm.implementation == "macllm",
        "MacLLM evidence has the wrong implementation"
    );
    ensure!(
        report.llama_cpp.schema_version == 2
            && report.llama_cpp.implementation == ImplementationKind::LlamaCpp,
        "llama.cpp evidence has the wrong schema or implementation"
    );
    ensure!(
        report.llama_cpp.provenance.model == report.model,
        "llama.cpp execution model differs from report provenance"
    );
    let llama_provenance = ImplementationProvenance {
        implementation: ImplementationKind::LlamaCpp,
        version: format!(
            "llama.cpp build {}",
            report.llama_cpp.provenance.llama_cpp_version.build
        ),
        commit: report.llama_cpp.provenance.llama_cpp_full_commit.clone(),
        executable: report.llama_cpp.provenance.llama_server.clone(),
        dirty: None,
    };
    validate_implementation_provenance(&llama_provenance)?;
    ensure!(
        report.llama_cpp.port_closed_after_shutdown,
        "llama-server port was not proven closed after the demo"
    );

    let identity = &report.macllm.identity;
    let prepared = &report.prepared;
    ensure!(
        identity.prepared_schema_version == prepared.schema_version
            && identity.model_family == prepared.model_family
            && identity.model_filename == prepared.model_file.filename
            && identity.model_sha256 == prepared.model_file.sha256
            && identity.repository_file_count == prepared.repository_bundle.file_count
            && identity.repository_bundle_sha256 == prepared.repository_bundle.sha256
            && identity.repository_payload_encoding == prepared.repository_payload_encoding
            && identity.encoded_repository_bundle_sha256
                == prepared.encoded_repository_bundle_sha256
            && identity.selected_bundle_byte_count == prepared.selected_bundle_byte_count
            && identity.selected_bundle_char_count == prepared.selected_bundle_char_count
            && identity.selected_bundle_sha256 == prepared.selected_bundle_sha256
            && identity.common_prefix_token_ids_sha256 == prepared.common_prefix_token_ids_sha256,
        "MacLLM execution identity differs from prepared evidence"
    );
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn validate_macllm_execution(report: &PrefixDemoReport) -> Result<()> {
    let prepared = &report.prepared;
    let macllm = &report.macllm;
    let config = &macllm.configuration;
    ensure!(
        config.device == "metal"
            && config.dtype == "f32"
            && config.queue_capacity == MAC_PREFIX_DEMO_CONCURRENCY
            && config.max_active_sequences == MAC_PREFIX_DEMO_CONCURRENCY
            && config.scheduler_token_budget == MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET
            && config.prefill_token_quantum == PREFIX_DEMO_PREFILL_QUANTUM
            && config.kv_cache_block_tokens == prepared.kv_cache_block_tokens
            && config.seed_completion_tokens == 1
            && config.measured_request_count == PREFIX_DEMO_TASK_COUNT
            && config.measured_completion_tokens == MAC_PREFIX_DEMO_DECODE_TOKENS
            && config.exact_cache_endpoint_tokens == prepared.requested_shared_tokens
            && config.measured_waves == macllm.waves.len()
            && config.measured_waves >= MAC_PREFIX_DEMO_MIN_MEASURED_WAVES,
        "MacLLM execution configuration differs from the public demo geometry"
    );
    ensure!(
        macllm.cold_cache.active_sequences == 0
            && macllm.after_seed_cache.active_sequences == 0
            && macllm.cold_cache.observation.is_none()
            && macllm.after_seed_cache.observation.is_none(),
        "MacLLM cold/seed cache evidence is unavailable or not quiescent"
    );
    ensure!(
        macllm.cold_cache.current.hits == 0
            && macllm.cold_cache.current.misses == 0
            && macllm.cold_cache.current.matched_tokens == 0
            && macllm.after_seed_cache.current.misses == 1,
        "MacLLM seed evidence does not begin from one cold cache miss"
    );
    ensure!(
        macllm.distributions == derive_macllm_distributions(&macllm.waves)?,
        "MacLLM wave distributions are not the exact derivation of retained raw waves"
    );

    let endpoint_total = u64::try_from(prepared.requested_shared_tokens)?
        .checked_mul(u64::try_from(PREFIX_DEMO_TASK_COUNT)?)
        .context("expected matched-token count overflow")?;
    let prefix_blocks = prepared
        .requested_shared_tokens
        .checked_div(prepared.kv_cache_block_tokens)
        .context("prefix-cache block size is zero")?;
    ensure!(
        prepared
            .requested_shared_tokens
            .is_multiple_of(prepared.kv_cache_block_tokens),
        "exact prefix endpoint is not block aligned"
    );
    let expected_active_references = prefix_blocks
        .checked_mul(MAC_PREFIX_DEMO_CONCURRENCY)
        .context("active prefix reference count overflow")?;
    let reference = macllm
        .waves
        .first()
        .context("MacLLM execution has no representative wave")?;
    for (wave_index, wave) in macllm.waves.iter().enumerate() {
        ensure!(
            wave.wave_index == wave_index && wave.requests.len() == PREFIX_DEMO_TASK_COUNT,
            "MacLLM wave {wave_index} index or request cardinality is invalid"
        );
        let cache = &wave.cache;
        let expected_before = if wave_index == 0 {
            &macllm.after_seed_cache
        } else {
            &macllm.waves[wave_index - 1].cache.after_wave
        };
        ensure!(
            cache.before_wave.active_sequences == 0
                && cache.after_wave.active_sequences == 0
                && cache.before_wave.observation.is_none()
                && cache.after_wave.observation.is_none()
                && &cache.before_wave == expected_before,
            "MacLLM wave {wave_index} cache snapshots are unavailable, not quiescent, or not consecutive"
        );
        let before = cache.before_wave.current;
        let after = cache.after_wave.current;
        let expected_hits = after
            .hits
            .checked_sub(before.hits)
            .context("MacLLM cache hit counter regressed")?;
        let expected_misses = after
            .misses
            .checked_sub(before.misses)
            .context("MacLLM cache miss counter regressed")?;
        let expected_matched = after
            .matched_tokens
            .checked_sub(before.matched_tokens)
            .context("MacLLM matched-token counter regressed")?;
        ensure!(
            cache.measured_counter_delta.hits == expected_hits
                && cache.measured_counter_delta.misses == expected_misses
                && cache.measured_counter_delta.matched_tokens == expected_matched
                && expected_hits == PREFIX_DEMO_TASK_COUNT as u64
                && expected_misses == 0
                && expected_matched == endpoint_total,
            "MacLLM wave {wave_index} did not prove eight exact 16K hits and zero misses"
        );
        ensure!(
            before.entries == after.entries
                && before.evictions == after.evictions
                && before.resident_snapshot_blocks == after.resident_snapshot_blocks
                && before.block_tokens == after.block_tokens
                && before.physical_live_blocks == after.physical_live_blocks
                && before.physical_live_references == after.physical_live_references
                && before.physical_shared_blocks == after.physical_shared_blocks
                && before.physical_max_blocks == after.physical_max_blocks,
            "MacLLM wave {wave_index} mutated the seeded prefix-cache residency despite its no-publication policy"
        );

        let peak = cache.eight_way_first_admission_snapshot;
        let observed_reference_delta = peak
            .physical_live_references
            .checked_sub(before.physical_live_references)
            .context("first-admission physical reference count regressed")?;
        let observed_block_delta = peak
            .physical_live_blocks
            .checked_sub(before.physical_live_blocks)
            .context("first-admission physical block count regressed")?;
        ensure!(
            peak.active_sequences == MAC_PREFIX_DEMO_CONCURRENCY
                && peak.block_tokens == prepared.kv_cache_block_tokens
                && observed_reference_delta == expected_active_references
                && observed_block_delta == 0
                && cache.active_request_prefix_page_references == observed_reference_delta
                && cache.new_physical_live_blocks_at_admission == observed_block_delta
                && peak.physical_shared_blocks >= prefix_blocks,
            "MacLLM wave {wave_index} first-admission evidence does not prove exact eight-way page reuse"
        );
        ensure!(
            approximately_equal(
                cache.active_request_prefix_page_fanout,
                finite_positive_ratio(observed_reference_delta as f64, prefix_blocks as f64)?,
            ) && approximately_equal(
                cache.active_request_prefix_reuse_coverage,
                finite_positive_ratio(
                    observed_reference_delta as f64,
                    expected_active_references as f64,
                )?,
            ) && approximately_equal(
                cache.atomic_pool_owner_density,
                finite_positive_ratio(
                    peak.physical_live_references as f64,
                    peak.physical_live_blocks as f64,
                )?,
            ),
            "MacLLM wave {wave_index} physical-reuse ratios differ from exact page counts"
        );
        validate_phase_timings(wave_index, &wave.phases)?;
        validate_aggregate(
            wave.aggregate.request_count,
            wave.aggregate.total_completion_tokens,
            wave.aggregate.wall_time_ms,
            wave.aggregate.aggregate_output_tokens_per_second,
            &wave
                .requests
                .iter()
                .map(|request| request.usage.completion_tokens)
                .collect::<Vec<_>>(),
            "MacLLM",
        )?;
        if wave_index > 0 {
            ensure!(
                wave.requests
                    .iter()
                    .zip(&reference.requests)
                    .all(|(request, expected)| same_request_result(request, expected)),
                "MacLLM wave {wave_index} output or request identity differs from representative wave 0"
            );
        }
    }
    Ok(())
}

fn validate_phase_timings(
    wave_index: usize,
    phases: &super::prefix_demo_mac::MacPrefixDemoPhaseTimingEvidence,
) -> Result<()> {
    ensure!(
        [
            phases.queue_admission_ms,
            phases.prefix_lookup_attach_ms,
            phases.cache_diagnostics_ms,
            phases.suffix_prefill_ms,
            phases.first_token_model_forward_ms,
            phases.single_step_decode_total_ms,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
            && phases.first_token_model_forward_ms > 0.0
            && phases.single_step_decode_ms.len() == MAC_PREFIX_DEMO_DECODE_TOKENS - 1
            && phases
                .single_step_decode_ms
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && approximately_equal(
                phases.single_step_decode_total_ms,
                phases.single_step_decode_ms.iter().sum(),
            )
            && !phases.boundary.trim().is_empty(),
        "MacLLM wave {wave_index} benchmark-only phase evidence is invalid or overlapping"
    );
    Ok(())
}

fn same_request_result(
    request: &super::prefix_demo_mac::PrefixDemoRequestEvidence,
    expected: &super::prefix_demo_mac::PrefixDemoRequestEvidence,
) -> bool {
    request.name == expected.name
        && request.task_index == expected.task_index
        && request.rendered_prompt_sha256 == expected.rendered_prompt_sha256
        && request.prompt_token_ids_sha256 == expected.prompt_token_ids_sha256
        && request.text == expected.text
        && request.usage == expected.usage
        && request.finish_reason == expected.finish_reason
}

fn validate_llama_execution(report: &PrefixDemoReport) -> Result<()> {
    let llama = &report.llama_cpp;
    ensure!(
        llama.measured_waves.len() >= MAC_PREFIX_DEMO_MIN_MEASURED_WAVES,
        "llama.cpp execution has fewer than {MAC_PREFIX_DEMO_MIN_MEASURED_WAVES} measured waves"
    );
    ensure!(
        llama.context.slots == PREFIX_DEMO_CONCURRENCY
            && llama.context.decode_tokens == MAC_PREFIX_DEMO_DECODE_TOKENS
            && llama.context.logical_batch_tokens == MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET
            && llama.context.physical_ubatch_tokens == PREFIX_DEMO_PREFILL_QUANTUM,
        "llama.cpp batch/context geometry differs from the matched demo budget"
    );
    ensure!(
        llama.request_policy.decode_tokens == MAC_PREFIX_DEMO_DECODE_TOKENS
            && llama.request_policy.temperature == 0.0
            && llama.request_policy.top_k == 0
            && approximately_equal(llama.request_policy.top_p, 1.0)
            && approximately_equal(llama.request_policy.repetition_penalty, 1.0)
            && llama.request_policy.seed == 42
            && llama.request_policy.ignore_eos
            && llama.request_policy.stream
            && llama.request_policy.return_tokens
            && llama.request_policy.cache_prompt
            && llama.request_policy.n_keep == 0,
        "llama.cpp request policy differs from the deterministic demo policy"
    );
    ensure!(
        llama.tokenizer_parity.len() == PREFIX_DEMO_TASK_COUNT + 1
            && llama.tokenizer_parity.iter().all(|entry| {
                entry.exact
                    && !entry.add_special
                    && entry.expected_token_ids_sha256 == entry.observed_token_ids_sha256
            }),
        "llama.cpp tokenizer parity is incomplete or inexact"
    );
    ensure!(
        llama.determinism.replay_concurrency == PREFIX_DEMO_CONCURRENCY
            && llama
                .determinism
                .replay_excluded_from_performance_measurement
            && llama.determinism.exact_per_request
            && llama.determinism.first_nonstream_replay_token_ids_sha256
                == llama.determinism.second_nonstream_replay_token_ids_sha256
            && llama
                .determinism
                .first_nonstream_replay_token_ids_sha256
                .len()
                == PREFIX_DEMO_TASK_COUNT,
        "llama.cpp correctness replay is not exact or is mixed into performance timing"
    );
    for (wave_index, wave) in llama.measured_waves.iter().enumerate() {
        ensure!(
            wave.wave_index == wave_index
                && wave.concurrency == PREFIX_DEMO_CONCURRENCY
                && wave.request_count == PREFIX_DEMO_TASK_COUNT
                && wave.requests.len() == PREFIX_DEMO_TASK_COUNT,
            "llama.cpp measured wave {wave_index} is not exactly eight-way or has the wrong index"
        );
        validate_aggregate(
            wave.request_count,
            wave.total_completion_tokens,
            wave.external_wall_time_ms,
            wave.aggregate_output_tokens_per_second,
            &wave
                .requests
                .iter()
                .map(|request| request.completion_tokens)
                .collect::<Vec<_>>(),
            "llama.cpp",
        )?;
        let ttft = wave
            .requests
            .iter()
            .map(|request| request.external_time_to_first_token_ms)
            .collect::<Vec<_>>();
        let latency = wave
            .requests
            .iter()
            .map(|request| request.external_latency_ms)
            .collect::<Vec<_>>();
        wave.external_time_to_first_token_ms.verify(&ttft)?;
        wave.external_latency_ms.verify(&latency)?;
        validate_llama_cache_observation(report, wave_index, wave)?;
    }
    Ok(())
}

fn validate_llama_cache_observation(
    report: &PrefixDemoReport,
    wave_index: usize,
    wave: &super::LlamaPrefixDemoWaveEvidence,
) -> Result<()> {
    let observation = &wave.cache;
    let cache_n = wave
        .requests
        .iter()
        .map(|request| request.terminal.server_reported_cache_n)
        .collect::<Vec<_>>();
    let prompt_n = wave
        .requests
        .iter()
        .map(|request| request.terminal.server_reported_prompt_n)
        .collect::<Vec<_>>();
    let expected_prompt_n = wave
        .requests
        .iter()
        .map(|request| {
            request
                .prompt_tokens
                .checked_sub(report.prepared.requested_shared_tokens)
                .context("llama.cpp request is shorter than the exact cache endpoint")
        })
        .collect::<Result<Vec<_>>>()?;
    let expected_min = cache_n
        .iter()
        .copied()
        .min()
        .context("no llama cache_n values")?;
    let expected_max = cache_n
        .iter()
        .copied()
        .max()
        .context("no llama cache_n values")?;
    let expected_exact = wave
        .requests
        .iter()
        .zip(&expected_prompt_n)
        .filter(|(request, expected_prompt_n)| {
            request.terminal.server_reported_cache_n == report.prepared.requested_shared_tokens
                && request.terminal.server_reported_prompt_n == **expected_prompt_n
        })
        .count();
    ensure!(
        observation.exact_common_prefix_tokens == report.prepared.requested_shared_tokens
            && observation.server_reported_cache_n_by_request == cache_n
            && observation.server_reported_prompt_n_by_request == prompt_n
            && observation.expected_prompt_n_by_request == expected_prompt_n
            && observation.requests_satisfying_exact_contract == expected_exact
            && expected_exact == PREFIX_DEMO_TASK_COUNT
            && observation.minimum_server_reported_cache_n == expected_min
            && observation.maximum_server_reported_cache_n == expected_max,
        "llama.cpp wave {wave_index} cache observation differs from exact request terminal counters"
    );
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn validate_request_identity(report: &PrefixDemoReport) -> Result<()> {
    let reference_wave = report
        .macllm
        .waves
        .first()
        .context("MacLLM execution has no representative wave")?;
    ensure!(
        report.prepared.tasks.len() == PREFIX_DEMO_TASK_COUNT
            && reference_wave.requests.len() == PREFIX_DEMO_TASK_COUNT
            && report.llama_cpp.slot_seeds.len() == PREFIX_DEMO_CONCURRENCY
            && report.llama_cpp.tokenizer_parity.len() == PREFIX_DEMO_TASK_COUNT + 1,
        "seed/task evidence cardinality differs between implementations"
    );

    let prepared_prompts = std::iter::once(&report.prepared.seed)
        .chain(&report.prepared.tasks)
        .collect::<Vec<_>>();
    for (offset, (prepared, tokenizer)) in prepared_prompts
        .iter()
        .zip(&report.llama_cpp.tokenizer_parity)
        .enumerate()
    {
        let llama_label = if offset == 0 {
            "seed".to_owned()
        } else {
            format!("request_{}", offset - 1)
        };
        ensure!(
            tokenizer.token_count == prepared.prompt_token_count
                && tokenizer.expected_token_ids_sha256 == prepared.prompt_token_ids_sha256
                && tokenizer.observed_token_ids_sha256 == prepared.prompt_token_ids_sha256
                && tokenizer.label == llama_label,
            "prompt {offset} tokenizer identity differs from prepared evidence"
        );
    }

    let mac_seed = &report.macllm.seed;
    ensure!(
        mac_seed.task_index.is_none()
            && mac_seed.name == report.prepared.seed.name
            && mac_seed.rendered_prompt_sha256 == report.prepared.seed.rendered_prompt_sha256
            && mac_seed.prompt_token_ids_sha256 == report.prepared.seed.prompt_token_ids_sha256
            && mac_seed.usage.prompt_tokens == report.prepared.seed.prompt_token_count
            && mac_seed.usage.cached_prompt_tokens == 0
            && mac_seed.usage.completion_tokens == 1
            && mac_seed.usage.decode_steps == 0
            && mac_seed.finish_reason == "length"
            && mac_seed.time_to_first_token_ms.is_finite()
            && mac_seed.time_to_first_token_ms > 0.0
            && mac_seed.latency_ms.is_finite()
            && mac_seed.latency_ms >= mac_seed.time_to_first_token_ms,
        "MacLLM seed identity or usage differs from prepared evidence"
    );
    for (request_index, (prepared, mac)) in report
        .prepared
        .tasks
        .iter()
        .zip(&reference_wave.requests)
        .enumerate()
    {
        ensure!(
            prepared.task_index == Some(request_index + 1)
                && mac.task_index == prepared.task_index
                && mac.name == prepared.name
                && mac.rendered_prompt_sha256 == prepared.rendered_prompt_sha256
                && mac.prompt_token_ids_sha256 == prepared.prompt_token_ids_sha256
                && mac.usage.prompt_tokens == prepared.prompt_token_count
                && mac.usage.cached_prompt_tokens == report.prepared.requested_shared_tokens
                && mac.usage.completion_tokens == MAC_PREFIX_DEMO_DECODE_TOKENS
                && mac.usage.decode_steps == mac.usage.completion_tokens.saturating_sub(1)
                && mac.finish_reason == "length"
                && mac.time_to_first_token_ms.is_finite()
                && mac.time_to_first_token_ms > 0.0
                && mac.latency_ms.is_finite()
                && mac.latency_ms >= mac.time_to_first_token_ms,
            "MacLLM task {request_index} identity or usage differs from prepared evidence"
        );
    }

    for (slot_id, seed) in report.llama_cpp.slot_seeds.iter().enumerate() {
        ensure!(
            seed.request.is_none()
                && seed.slot_id == slot_id
                && seed.label == format!("seed_slot_{slot_id}")
                && seed.prompt_tokens == report.prepared.seed.prompt_token_count
                && seed.prompt_token_ids_sha256 == report.prepared.seed.prompt_token_ids_sha256
                && seed.completion_tokens == 1
                && seed.completion_token_ids.len() == 1
                && seed.completion_token_ids_sha256 == token_ids_sha256(&seed.completion_token_ids)
                && seed.streamed_content_matches_replay == (seed.streamed_content == seed.content)
                && !seed.content_source.trim().is_empty()
                && !seed.completion_token_source.trim().is_empty()
                && seed.external_time_to_first_token_ms.is_finite()
                && seed.external_time_to_first_token_ms > 0.0
                && seed.external_latency_ms.is_finite()
                && seed.external_latency_ms >= seed.external_time_to_first_token_ms
                && seed.terminal.server_reported_cache_n == 0
                && seed.terminal.server_reported_prompt_n
                    == report.prepared.seed.prompt_token_count
                && seed.terminal.server_reported_predicted_n == 1,
            "llama.cpp slot seed {slot_id} identity differs from prepared evidence"
        );
    }

    for (wave_index, wave) in report.llama_cpp.measured_waves.iter().enumerate() {
        for (request_index, (prepared, llama)) in
            report.prepared.tasks.iter().zip(&wave.requests).enumerate()
        {
            let replay_digest = &report
                .llama_cpp
                .determinism
                .first_nonstream_replay_token_ids_sha256[request_index];
            let terminal = &llama.terminal;
            ensure!(
                llama.request == Some(request_index)
                    && llama.slot_id == request_index
                    && llama.label == format!("request_{request_index}")
                    && llama.prompt_tokens == prepared.prompt_token_count
                    && llama.prompt_token_ids_sha256 == prepared.prompt_token_ids_sha256
                    && llama.completion_tokens == MAC_PREFIX_DEMO_DECODE_TOKENS
                    && llama.streamed_content_matches_replay
                        == (llama.streamed_content == llama.content)
                    && !llama.content_source.trim().is_empty()
                    && !llama.completion_token_source.trim().is_empty()
                    && llama.external_time_to_first_token_ms.is_finite()
                    && llama.external_time_to_first_token_ms > 0.0
                    && llama.external_latency_ms.is_finite()
                    && llama.external_latency_ms >= llama.external_time_to_first_token_ms,
                "llama.cpp wave {wave_index} request {request_index} identity or timing is invalid"
            );
            ensure!(
                terminal
                    .server_reported_cache_n
                    .checked_add(terminal.server_reported_prompt_n)
                    == Some(prepared.prompt_token_count)
                    && terminal.server_reported_predicted_n == llama.completion_tokens
                    && terminal.server_reported_prompt_ms.is_finite()
                    && terminal.server_reported_prompt_ms > 0.0
                    && terminal.server_reported_predicted_ms.is_finite()
                    && terminal.server_reported_predicted_ms > 0.0
                    && approximately_equal(
                        terminal.server_reported_prompt_tokens_per_second,
                        terminal.server_reported_prompt_n as f64 * 1_000.0
                            / terminal.server_reported_prompt_ms,
                    )
                    && approximately_equal(
                        terminal.server_reported_predicted_tokens_per_second,
                        terminal.server_reported_predicted_n as f64 * 1_000.0
                            / terminal.server_reported_predicted_ms,
                    ),
                "llama.cpp wave {wave_index} request {request_index} terminal counters are invalid"
            );
            ensure!(
                llama.completion_tokens == llama.completion_token_ids.len()
                    && llama.completion_token_ids_sha256
                        == token_ids_sha256(&llama.completion_token_ids)
                    && replay_digest == &llama.completion_token_ids_sha256,
                "llama.cpp wave {wave_index} request {request_index} replay tokens are inconsistent"
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn derive_comparison(
    prepared: &PreparedPrefixDemo,
    macllm: &MacPrefixDemoExecution,
    llama_cpp: &LlamaPrefixDemoExecution,
) -> Result<PrefixDemoComparison> {
    let reference_wave = macllm
        .waves
        .first()
        .context("cannot derive comparison without a representative MacLLM wave")?;
    let llama_reference_wave = llama_cpp
        .measured_waves
        .first()
        .context("cannot derive comparison without a representative llama.cpp wave")?;
    ensure!(
        reference_wave.requests.len() == PREFIX_DEMO_TASK_COUNT
            && llama_reference_wave.requests.len() == PREFIX_DEMO_TASK_COUNT,
        "cannot derive comparison without exactly eight requests per implementation"
    );
    let correctness_matches = macllm
        .waves
        .first()
        .expect("representative wave was checked")
        .requests
        .iter()
        .zip(&llama_reference_wave.requests)
        .map(|(mac, llama)| mac.text == llama.content)
        .collect::<Vec<_>>();
    let exact_correctness_replay_output_match = correctness_matches.iter().all(|exact| *exact);
    let stream_matches = llama_reference_wave
        .requests
        .iter()
        .map(|request| request.streamed_content == request.content)
        .collect::<Vec<_>>();
    let stream_match_count = stream_matches.iter().filter(|exact| **exact).count();
    let eligibility = derive_comparison_eligibility(ComparisonEligibilityFacts {
        macllm_measured_waves: macllm.waves.len(),
        llama_cpp_measured_waves: llama_cpp.measured_waves.len(),
        macllm_exact_prefix_cache_hit_requests_by_wave: macllm
            .waves
            .iter()
            .map(|wave| {
                wave.requests
                    .iter()
                    .filter(|request| {
                        request.usage.cached_prompt_tokens == prepared.requested_shared_tokens
                    })
                    .count()
            })
            .collect(),
        llama_cpp_exact_prefix_cache_hit_requests_by_wave: llama_cpp
            .measured_waves
            .iter()
            .map(|wave| wave.cache.requests_satisfying_exact_contract)
            .collect(),
        exact_input_token_digest_parity: has_exact_input_parity(prepared, macllm, llama_cpp),
        exact_completion_token_count_parity: has_exact_completion_token_count_parity(
            macllm, llama_cpp,
        ),
        exact_cross_implementation_output_parity: has_exact_cross_output_parity(macllm, llama_cpp),
        deterministic_greedy_replay_parity: has_deterministic_greedy_replay(llama_cpp),
        llama_cpp_measured_stream_matches_replay: llama_cpp
            .measured_waves
            .iter()
            .flat_map(|wave| &wave.requests)
            .all(|request| request.streamed_content == request.content),
        macllm_kv_cache_dtype: macllm.configuration.dtype.clone(),
        llama_cpp_k_cache_dtype: command_option_value(&llama_cpp.command, "--cache-type-k")
            .map(str::to_owned),
        llama_cpp_v_cache_dtype: command_option_value(&llama_cpp.command, "--cache-type-v")
            .map(str::to_owned),
        transport_parity: has_matched_transport_boundary(macllm, llama_cpp),
    });
    let cache = &reference_wave.cache;
    let peak = cache.eight_way_first_admission_snapshot;
    let mac_tps = macllm
        .distributions
        .wave_aggregate_output_tokens_per_second
        .p50;
    let llama_tps = summarize(
        &llama_cpp
            .measured_waves
            .iter()
            .map(|wave| wave.aggregate_output_tokens_per_second)
            .collect::<Vec<_>>(),
    )?
    .p50;
    let throughput_ratio = gated_throughput_ratio(&eligibility, mac_tps, llama_tps)?;
    Ok(PrefixDemoComparison {
        exact_common_prefix_tokens: prepared.requested_shared_tokens,
        concurrency: PREFIX_DEMO_TASK_COUNT,
        completion_tokens_per_request: MAC_PREFIX_DEMO_DECODE_TOKENS,
        exact_correctness_replay_output_match_per_request: correctness_matches,
        exact_correctness_replay_output_match,
        llama_cpp_stream_matches_correctness_replay_per_request: stream_matches,
        llama_cpp_stream_matches_correctness_replay_requests: stream_match_count,
        macllm_representative_wave_cache_hit_requests: usize::try_from(
            cache.measured_counter_delta.hits,
        )
        .context("MacLLM cache-hit count exceeds usize")?,
        macllm_representative_wave_matched_prefix_tokens: cache
            .measured_counter_delta
            .matched_tokens,
        macllm_first_admission_shared_blocks: peak.physical_shared_blocks,
        macllm_first_admission_page_references: peak.physical_live_references,
        macllm_first_admission_physical_blocks: peak.physical_live_blocks,
        macllm_first_admission_resident_snapshot_blocks: peak.resident_snapshot_blocks,
        macllm_active_request_prefix_page_references: cache.active_request_prefix_page_references,
        macllm_active_request_prefix_page_fanout: cache.active_request_prefix_page_fanout,
        macllm_active_request_prefix_reuse_coverage: cache.active_request_prefix_reuse_coverage,
        macllm_new_physical_live_blocks_at_admission: cache.new_physical_live_blocks_at_admission,
        macllm_atomic_pool_owner_density: cache.atomic_pool_owner_density,
        llama_cpp_requests_reporting_at_least_endpoint: llama_reference_wave
            .cache
            .requests_satisfying_exact_contract,
        llama_cpp_minimum_server_reported_cache_n: llama_reference_wave
            .cache
            .minimum_server_reported_cache_n,
        llama_cpp_maximum_server_reported_cache_n: llama_reference_wave
            .cache
            .maximum_server_reported_cache_n,
        macllm_aggregate_output_tokens_per_second: mac_tps,
        llama_cpp_aggregate_output_tokens_per_second: llama_tps,
        eligibility,
        macllm_to_llama_cpp_aggregate_throughput_ratio: throughput_ratio,
        cache_semantics_boundary: CACHE_SEMANTICS_BOUNDARY.to_owned(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ComparisonEligibilityFacts {
    macllm_measured_waves: usize,
    llama_cpp_measured_waves: usize,
    macllm_exact_prefix_cache_hit_requests_by_wave: Vec<usize>,
    llama_cpp_exact_prefix_cache_hit_requests_by_wave: Vec<usize>,
    exact_input_token_digest_parity: bool,
    exact_completion_token_count_parity: bool,
    exact_cross_implementation_output_parity: bool,
    deterministic_greedy_replay_parity: bool,
    llama_cpp_measured_stream_matches_replay: bool,
    macllm_kv_cache_dtype: String,
    llama_cpp_k_cache_dtype: Option<String>,
    llama_cpp_v_cache_dtype: Option<String>,
    transport_parity: bool,
}

#[allow(clippy::too_many_lines)]
fn derive_comparison_eligibility(
    facts: ComparisonEligibilityFacts,
) -> PrefixDemoComparisonEligibility {
    let mut reasons = Vec::new();
    let kv_cache_dtype_parity = facts
        .llama_cpp_k_cache_dtype
        .as_deref()
        .zip(facts.llama_cpp_v_cache_dtype.as_deref())
        .is_some_and(|(key, value)| {
            key.eq_ignore_ascii_case(&facts.macllm_kv_cache_dtype)
                && value.eq_ignore_ascii_case(&facts.macllm_kv_cache_dtype)
        });
    if facts.macllm_measured_waves != facts.llama_cpp_measured_waves {
        reasons.push(format!(
            "measured wave counts differ: MacLLM has {} and llama.cpp has {}; equal wave counts are required",
            facts.macllm_measured_waves, facts.llama_cpp_measured_waves
        ));
    }
    if facts.macllm_measured_waves < MAC_PREFIX_DEMO_MIN_MEASURED_WAVES {
        reasons.push(format!(
            "MacLLM has {} measured waves; at least {} are required",
            facts.macllm_measured_waves, MAC_PREFIX_DEMO_MIN_MEASURED_WAVES
        ));
    }
    if facts.llama_cpp_measured_waves < MAC_PREFIX_DEMO_MIN_MEASURED_WAVES {
        reasons.push(format!(
            "llama.cpp has {} measured waves; at least {} are required",
            facts.llama_cpp_measured_waves, MAC_PREFIX_DEMO_MIN_MEASURED_WAVES
        ));
    }
    append_cache_eligibility_reason(
        &mut reasons,
        "MacLLM",
        facts.macllm_measured_waves,
        &facts.macllm_exact_prefix_cache_hit_requests_by_wave,
    );
    append_cache_eligibility_reason(
        &mut reasons,
        "llama.cpp",
        facts.llama_cpp_measured_waves,
        &facts.llama_cpp_exact_prefix_cache_hit_requests_by_wave,
    );
    if !facts.exact_input_token_digest_parity {
        reasons.push(
            "exact input token-digest parity is not proven for every seed and measured request"
                .to_owned(),
        );
    }
    if !facts.exact_completion_token_count_parity {
        reasons.push(
            "completion-token counts are not identical across implementations and measured waves"
                .to_owned(),
        );
    }
    if !facts.exact_cross_implementation_output_parity {
        reasons.push(
            "MacLLM measured output does not exactly match llama.cpp correctness-replay output for every request"
                .to_owned(),
        );
    }
    if !facts.deterministic_greedy_replay_parity {
        reasons.push(
            "the deterministic greedy replay contract is not proven by retained llama.cpp evidence"
                .to_owned(),
        );
    }
    if !facts.llama_cpp_measured_stream_matches_replay {
        reasons.push(
            "llama.cpp measured streaming output does not exactly match its correctness replay for every request"
                .to_owned(),
        );
    }
    if !kv_cache_dtype_parity {
        reasons.push(format!(
            "KV-cache dtype differs or is unproven: MacLLM={}, llama.cpp K={}, llama.cpp V={}",
            facts.macllm_kv_cache_dtype,
            facts
                .llama_cpp_k_cache_dtype
                .as_deref()
                .unwrap_or("unknown"),
            facts
                .llama_cpp_v_cache_dtype
                .as_deref()
                .unwrap_or("unknown")
        ));
    }
    if !facts.transport_parity {
        reasons.push(
            "transport boundaries differ: MacLLM uses direct EngineHandle execution while llama.cpp uses loopback HTTP streaming"
                .to_owned(),
        );
    }

    PrefixDemoComparisonEligibility {
        eligible: reasons.is_empty(),
        minimum_measured_waves_per_implementation: MAC_PREFIX_DEMO_MIN_MEASURED_WAVES,
        macllm_measured_waves: facts.macllm_measured_waves,
        llama_cpp_measured_waves: facts.llama_cpp_measured_waves,
        macllm_exact_prefix_cache_hit_requests_by_wave: facts
            .macllm_exact_prefix_cache_hit_requests_by_wave,
        llama_cpp_exact_prefix_cache_hit_requests_by_wave: facts
            .llama_cpp_exact_prefix_cache_hit_requests_by_wave,
        exact_input_token_digest_parity: facts.exact_input_token_digest_parity,
        exact_completion_token_count_parity: facts.exact_completion_token_count_parity,
        exact_cross_implementation_output_parity: facts.exact_cross_implementation_output_parity,
        deterministic_greedy_replay_parity: facts.deterministic_greedy_replay_parity,
        llama_cpp_measured_stream_matches_replay: facts.llama_cpp_measured_stream_matches_replay,
        macllm_kv_cache_dtype: facts.macllm_kv_cache_dtype,
        llama_cpp_k_cache_dtype: facts.llama_cpp_k_cache_dtype,
        llama_cpp_v_cache_dtype: facts.llama_cpp_v_cache_dtype,
        kv_cache_dtype_parity,
        transport_parity: facts.transport_parity,
        reasons,
    }
}

fn append_cache_eligibility_reason(
    reasons: &mut Vec<String>,
    implementation: &str,
    measured_waves: usize,
    exact_hit_requests_by_wave: &[usize],
) {
    if exact_hit_requests_by_wave.len() != measured_waves {
        reasons.push(format!(
            "{implementation} exact-prefix cache evidence covers {} of {measured_waves} measured waves",
            exact_hit_requests_by_wave.len()
        ));
        return;
    }
    let incomplete = exact_hit_requests_by_wave
        .iter()
        .enumerate()
        .filter(|(_, hits)| **hits != PREFIX_DEMO_TASK_COUNT)
        .map(|(wave, hits)| format!("wave {wave}: {hits}/{PREFIX_DEMO_TASK_COUNT}"))
        .collect::<Vec<_>>();
    if !incomplete.is_empty() {
        reasons.push(format!(
            "{implementation} exact-prefix cache hits are not 8/8 on every measured wave ({})",
            incomplete.join(", ")
        ));
    }
}

fn has_exact_input_parity(
    prepared: &PreparedPrefixDemo,
    macllm: &MacPrefixDemoExecution,
    llama_cpp: &LlamaPrefixDemoExecution,
) -> bool {
    let Some(reference_wave) = macllm.waves.first() else {
        return false;
    };
    if prepared.tasks.len() != PREFIX_DEMO_TASK_COUNT
        || reference_wave.requests.len() != prepared.tasks.len()
        || llama_cpp.slot_seeds.len() != PREFIX_DEMO_CONCURRENCY
        || llama_cpp.tokenizer_parity.len() != prepared.tasks.len() + 1
    {
        return false;
    }

    macllm.seed.rendered_prompt_sha256 == prepared.seed.rendered_prompt_sha256
        && macllm.seed.prompt_token_ids_sha256 == prepared.seed.prompt_token_ids_sha256
        && llama_cpp.slot_seeds.iter().all(|seed| {
            seed.prompt_tokens == prepared.seed.prompt_token_count
                && seed.prompt_token_ids_sha256 == prepared.seed.prompt_token_ids_sha256
        })
        && std::iter::once(&prepared.seed)
            .chain(&prepared.tasks)
            .zip(&llama_cpp.tokenizer_parity)
            .all(|(prepared, tokenizer)| {
                tokenizer.exact
                    && tokenizer.token_count == prepared.prompt_token_count
                    && tokenizer.expected_token_ids_sha256 == prepared.prompt_token_ids_sha256
                    && tokenizer.observed_token_ids_sha256 == prepared.prompt_token_ids_sha256
            })
        && macllm.waves.iter().all(|wave| {
            wave.requests.len() == prepared.tasks.len()
                && wave
                    .requests
                    .iter()
                    .zip(&prepared.tasks)
                    .all(|(request, prepared)| {
                        request.rendered_prompt_sha256 == prepared.rendered_prompt_sha256
                            && request.prompt_token_ids_sha256 == prepared.prompt_token_ids_sha256
                    })
        })
        && llama_cpp.measured_waves.iter().all(|wave| {
            wave.requests.len() == prepared.tasks.len()
                && wave
                    .requests
                    .iter()
                    .zip(&prepared.tasks)
                    .all(|(request, prepared)| {
                        request.prompt_tokens == prepared.prompt_token_count
                            && request.prompt_token_ids_sha256 == prepared.prompt_token_ids_sha256
                    })
        })
}

fn has_exact_completion_token_count_parity(
    macllm: &MacPrefixDemoExecution,
    llama_cpp: &LlamaPrefixDemoExecution,
) -> bool {
    macllm.waves.iter().all(|wave| {
        wave.requests.len() == PREFIX_DEMO_TASK_COUNT
            && wave
                .requests
                .iter()
                .all(|request| request.usage.completion_tokens == MAC_PREFIX_DEMO_DECODE_TOKENS)
    }) && llama_cpp.measured_waves.iter().all(|wave| {
        wave.requests.len() == PREFIX_DEMO_TASK_COUNT
            && wave
                .requests
                .iter()
                .all(|request| request.completion_tokens == MAC_PREFIX_DEMO_DECODE_TOKENS)
    })
}

fn has_exact_cross_output_parity(
    macllm: &MacPrefixDemoExecution,
    llama_cpp: &LlamaPrefixDemoExecution,
) -> bool {
    let Some(mac_reference) = macllm.waves.first() else {
        return false;
    };
    if mac_reference.requests.len() != PREFIX_DEMO_TASK_COUNT {
        return false;
    }
    macllm.waves.iter().all(|wave| {
        wave.requests.len() == mac_reference.requests.len()
            && wave
                .requests
                .iter()
                .zip(&mac_reference.requests)
                .all(|(request, reference)| request.text == reference.text)
    }) && llama_cpp.measured_waves.iter().all(|wave| {
        wave.requests.len() == mac_reference.requests.len()
            && wave
                .requests
                .iter()
                .zip(&mac_reference.requests)
                .all(|(request, reference)| request.content == reference.text)
    })
}

fn has_deterministic_greedy_replay(llama_cpp: &LlamaPrefixDemoExecution) -> bool {
    let policy = &llama_cpp.request_policy;
    let determinism = &llama_cpp.determinism;
    policy.temperature == 0.0
        && policy.top_k == 0
        && approximately_equal(policy.top_p, 1.0)
        && approximately_equal(policy.repetition_penalty, 1.0)
        && determinism.replay_excluded_from_performance_measurement
        && determinism.exact_per_request
        && determinism.first_nonstream_replay_token_ids_sha256
            == determinism.second_nonstream_replay_token_ids_sha256
        && determinism.first_nonstream_replay_token_ids_sha256.len() == PREFIX_DEMO_TASK_COUNT
        && llama_cpp.measured_waves.iter().all(|wave| {
            wave.requests.len() == PREFIX_DEMO_TASK_COUNT
                && wave
                    .requests
                    .iter()
                    .zip(&determinism.first_nonstream_replay_token_ids_sha256)
                    .all(|(request, replay_digest)| {
                        request.completion_token_ids.len() == request.completion_tokens
                            && request.completion_token_ids_sha256 == *replay_digest
                            && request.completion_token_ids_sha256
                                == token_ids_sha256(&request.completion_token_ids)
                    })
        })
}

fn command_option_value<'a>(command: &'a [String], option: &str) -> Option<&'a str> {
    command
        .windows(2)
        .find(|pair| pair[0] == option)
        .map(|pair| pair[1].as_str())
}

fn has_matched_transport_boundary(
    _macllm: &MacPrefixDemoExecution,
    _llama_cpp: &LlamaPrefixDemoExecution,
) -> bool {
    // Schema v4's MacLLM timing evidence is captured at the direct
    // `EngineHandle` boundary, while llama.cpp is externally timed through
    // loopback HTTP streaming. Fail closed until both execution evidence
    // schemas expose and retain the same transport boundary.
    false
}

fn gated_throughput_ratio(
    eligibility: &PrefixDemoComparisonEligibility,
    macllm_tokens_per_second: f64,
    llama_cpp_tokens_per_second: f64,
) -> Result<Option<f64>> {
    if !eligibility.eligible {
        return Ok(None);
    }
    finite_positive_ratio(macllm_tokens_per_second, llama_cpp_tokens_per_second).map(Some)
}

fn validate_comparison(report: &PrefixDemoReport) -> Result<()> {
    let derived = derive_comparison(&report.prepared, &report.macllm, &report.llama_cpp)?;
    ensure!(
        report.comparison == derived,
        "prefix-demo comparison is not the exact derivation of raw execution evidence"
    );
    Ok(())
}

fn validate_aggregate(
    request_count: usize,
    total_completion_tokens: u64,
    wall_time_ms: f64,
    tokens_per_second: f64,
    completion_tokens: &[usize],
    implementation: &str,
) -> Result<()> {
    let expected_total = completion_tokens.iter().try_fold(0_u64, |total, tokens| {
        total.checked_add(u64::try_from(*tokens).ok()?)
    });
    ensure!(
        request_count == completion_tokens.len()
            && expected_total == Some(total_completion_tokens)
            && wall_time_ms.is_finite()
            && wall_time_ms > 0.0,
        "{implementation} aggregate cardinality, token total, or wall time is invalid"
    );
    let expected_tps = total_completion_tokens as f64 * 1_000.0 / wall_time_ms;
    ensure!(
        approximately_equal(tokens_per_second, expected_tps),
        "{implementation} aggregate throughput differs from its raw token/time evidence"
    );
    Ok(())
}

fn finite_positive_ratio(numerator: f64, denominator: f64) -> Result<f64> {
    ensure!(
        numerator.is_finite() && numerator > 0.0,
        "ratio numerator must be finite and positive"
    );
    ensure!(
        denominator.is_finite() && denominator > 0.0,
        "ratio denominator must be finite and positive"
    );
    let ratio = numerator / denominator;
    ensure!(ratio.is_finite() && ratio > 0.0, "ratio is invalid");
    Ok(ratio)
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= scale * 1.0e-6
}

fn serialized_json(report: &PrefixDemoReport) -> Result<Vec<u8>> {
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    Ok(json)
}

fn markdown_cell(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('`', "\\`")
        .replace(['\r', '\n'], " ")
}

pub(crate) fn normalize_artifact_pair(
    json_path: &Path,
    markdown_path: &Path,
) -> Result<(PathBuf, PathBuf)> {
    let json_path = normalize_destination(json_path)?;
    let markdown_path = normalize_destination(markdown_path)?;
    ensure!(
        json_path
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("json")
            && markdown_path
                .extension()
                .and_then(|extension| extension.to_str())
                == Some("md"),
        "prefix-demo outputs require distinct .json and .md filenames"
    );
    ensure!(
        json_path != markdown_path,
        "prefix-demo JSON and Markdown output paths resolve to the same target"
    );
    ensure!(
        json_path.parent() == markdown_path.parent(),
        "prefix-demo JSON and Markdown must share one output directory for paired publication"
    );
    // A pair of paths that resolve to the same file is refused on every host.
    ensure!(
        !output_paths_are_same_file(&json_path, &markdown_path)?,
        "prefix-demo JSON and Markdown output paths are hard-link aliases"
    );
    Ok((json_path, markdown_path))
}

/// Reports whether two output paths name the same file.
///
/// The check fails closed: a target that exists but cannot be compared is an
/// error rather than a pair that passes. A target that does not exist yet is
/// not the same file as anything, so it passes.
fn output_paths_are_same_file(a: &Path, b: &Path) -> Result<bool> {
    let metadata_a = match fs::metadata(a) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to inspect output target {}", a.display()));
        }
    };
    let metadata_b = match fs::metadata(b) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to inspect output target {}", b.display()));
        }
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        match (metadata_a, metadata_b) {
            (Some(metadata_a), Some(metadata_b)) => {
                Ok(metadata_a.dev() == metadata_b.dev() && metadata_a.ino() == metadata_b.ino())
            }
            _ => Ok(false),
        }
    }
    #[cfg(not(unix))]
    {
        // The metadata only proves that both targets exist here; the
        // comparison itself opens them.
        if metadata_a.is_none() || metadata_b.is_none() {
            return Ok(false);
        }
        let same_file = same_file::is_same_file(a, b).with_context(|| {
            format!(
                "failed to compare output targets {} and {}",
                a.display(),
                b.display()
            )
        })?;
        Ok(same_file)
    }
}

fn normalize_destination(path: &Path) -> Result<PathBuf> {
    let filename = path
        .file_name()
        .filter(|name| !name.is_empty())
        .context("prefix-demo output path has no filename")?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent).with_context(|| {
        format!(
            "prefix-demo output directory does not exist: {}",
            parent.display()
        )
    })?;
    ensure!(
        parent.is_dir(),
        "prefix-demo output parent is not a directory"
    );
    let normalized = parent.join(filename);
    match fs::symlink_metadata(&normalized) {
        Ok(metadata) => ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "prefix-demo output target must be a regular non-symlink file: {}",
            normalized.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!("failed to inspect output target {}", normalized.display())
            });
        }
    }
    Ok(normalized)
}

pub(crate) fn prepare_atomic_file(path: &Path, bytes: &[u8]) -> Result<tempfile::NamedTempFile> {
    let parent = path.parent().context("normalized output has no parent")?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent).with_context(|| {
        format!(
            "failed to create temporary prefix-demo artifact in {}",
            parent.display()
        )
    })?;
    temporary
        .write_all(bytes)
        .with_context(|| format!("failed to write temporary artifact for {}", path.display()))?;
    temporary
        .flush()
        .with_context(|| format!("failed to flush temporary artifact for {}", path.display()))?;
    temporary
        .as_file()
        .sync_all()
        .with_context(|| format!("failed to sync temporary artifact for {}", path.display()))?;
    Ok(temporary)
}

pub(crate) fn commit_artifact_pair(
    json_temp: tempfile::NamedTempFile,
    json_path: &Path,
    markdown_temp: tempfile::NamedTempFile,
    markdown_path: &Path,
) -> Result<()> {
    let json_backup = backup_existing(json_path)?;
    let markdown_backup = match backup_existing(markdown_path) {
        Ok(backup) => backup,
        Err(error) => {
            restore_target(json_path, json_backup.as_deref())?;
            return Err(error);
        }
    };

    if let Err(error) = json_temp.persist(json_path) {
        restore_target(json_path, json_backup.as_deref())?;
        restore_target(markdown_path, markdown_backup.as_deref())?;
        return Err(error.error).context("failed to publish prefix-demo JSON");
    }
    if let Err(error) = markdown_temp.persist(markdown_path) {
        restore_target(json_path, json_backup.as_deref())?;
        restore_target(markdown_path, markdown_backup.as_deref())?;
        return Err(error.error).context("failed to publish prefix-demo Markdown");
    }

    remove_backup(json_backup)?;
    remove_backup(markdown_backup)?;
    // Windows cannot open a directory with `File::open`, so the artifact
    // directory is only synced on the hosts that can open one.
    #[cfg(unix)]
    {
        let parent = json_path
            .parent()
            .context("published artifacts have no parent")?;
        fs::File::open(parent)
            .with_context(|| format!("failed to open artifact directory {}", parent.display()))?
            .sync_all()
            .with_context(|| format!("failed to sync artifact directory {}", parent.display()))?;
    }
    Ok(())
}

fn backup_existing(target: &Path) -> Result<Option<PathBuf>> {
    match fs::symlink_metadata(target) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("failed to inspect existing artifact"),
    }
    let parent = target.parent().context("artifact target has no parent")?;
    let placeholder = tempfile::NamedTempFile::new_in(parent)?;
    let temporary_path = placeholder.into_temp_path();
    let backup = temporary_path.to_path_buf();
    temporary_path.close()?;
    fs::rename(target, &backup).with_context(|| {
        format!(
            "failed to stage existing artifact {} for replacement",
            target.display()
        )
    })?;
    Ok(Some(backup))
}

fn restore_target(target: &Path, backup: Option<&Path>) -> Result<()> {
    match fs::remove_file(target) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("failed to remove partial artifact"),
    }
    if let Some(backup) = backup {
        fs::rename(backup, target)
            .with_context(|| format!("failed to restore previous artifact {}", target.display()))?;
    }
    Ok(())
}

fn remove_backup(backup: Option<PathBuf>) -> Result<()> {
    let Some(backup) = backup else {
        return Ok(());
    };
    fs::remove_file(&backup)
        .with_context(|| format!("failed to remove artifact backup {}", backup.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comparison_eligible_facts() -> ComparisonEligibilityFacts {
        ComparisonEligibilityFacts {
            macllm_measured_waves: MAC_PREFIX_DEMO_MIN_MEASURED_WAVES,
            llama_cpp_measured_waves: MAC_PREFIX_DEMO_MIN_MEASURED_WAVES,
            macllm_exact_prefix_cache_hit_requests_by_wave: vec![
                PREFIX_DEMO_TASK_COUNT;
                MAC_PREFIX_DEMO_MIN_MEASURED_WAVES
            ],
            llama_cpp_exact_prefix_cache_hit_requests_by_wave: vec![
                PREFIX_DEMO_TASK_COUNT;
                MAC_PREFIX_DEMO_MIN_MEASURED_WAVES
            ],
            exact_input_token_digest_parity: true,
            exact_completion_token_count_parity: true,
            exact_cross_implementation_output_parity: true,
            deterministic_greedy_replay_parity: true,
            llama_cpp_measured_stream_matches_replay: true,
            macllm_kv_cache_dtype: "f16".to_owned(),
            llama_cpp_k_cache_dtype: Some("f16".to_owned()),
            llama_cpp_v_cache_dtype: Some("f16".to_owned()),
            transport_parity: true,
        }
    }

    #[test]
    fn ratios_fail_closed() -> Result<()> {
        assert!((finite_positive_ratio(8.0, 4.0)? - 2.0).abs() < f64::EPSILON);
        assert!(finite_positive_ratio(0.0, 4.0).is_err());
        assert!(finite_positive_ratio(4.0, 0.0).is_err());
        assert!(finite_positive_ratio(f64::NAN, 4.0).is_err());
        Ok(())
    }

    #[test]
    fn throughput_ratio_requires_every_fair_comparison_gate() -> Result<()> {
        let eligible = derive_comparison_eligibility(comparison_eligible_facts());
        assert!(eligible.eligible);
        assert!(eligible.reasons.is_empty());
        assert_eq!(gated_throughput_ratio(&eligible, 8.0, 4.0)?, Some(2.0));

        let mut facts = comparison_eligible_facts();
        facts.llama_cpp_measured_waves = 1;
        facts.llama_cpp_exact_prefix_cache_hit_requests_by_wave = vec![7];
        facts.exact_input_token_digest_parity = false;
        facts.exact_completion_token_count_parity = false;
        facts.exact_cross_implementation_output_parity = false;
        facts.deterministic_greedy_replay_parity = false;
        facts.llama_cpp_measured_stream_matches_replay = false;
        facts.llama_cpp_k_cache_dtype = Some("f16".to_owned());
        facts.llama_cpp_v_cache_dtype = Some("f16".to_owned());
        facts.macllm_kv_cache_dtype = "f32".to_owned();
        facts.transport_parity = false;
        let ineligible = derive_comparison_eligibility(facts);

        assert!(!ineligible.eligible);
        assert_eq!(gated_throughput_ratio(&ineligible, 8.0, 4.0)?, None);
        for expected in [
            "wave counts differ",
            "at least 20",
            "cache hits are not 8/8",
            "input token-digest parity",
            "completion-token counts",
            "does not exactly match",
            "deterministic greedy replay",
            "streaming output",
            "KV-cache dtype differs",
            "transport boundaries differ",
        ] {
            assert!(
                ineligible
                    .reasons
                    .iter()
                    .any(|reason| reason.contains(expected)),
                "missing reason containing {expected:?}: {:?}",
                ineligible.reasons
            );
        }
        Ok(())
    }

    #[test]
    fn ineligible_markdown_suppresses_speed_ratio() -> Result<()> {
        let mut facts = comparison_eligible_facts();
        facts.llama_cpp_measured_waves = 1;
        facts.llama_cpp_exact_prefix_cache_hit_requests_by_wave = vec![1];
        let eligibility = derive_comparison_eligibility(facts);
        let mut markdown = String::new();
        render_comparison_eligibility(&mut markdown, &eligibility, None)?;

        assert!(markdown.contains("Not comparison-eligible"));
        assert!(markdown.contains("No aggregate-throughput speed ratio is published"));
        assert!(markdown.contains("Eligibility reasons"));
        assert!(!markdown.contains("x faster"));
        assert!(!markdown.contains("**2.000x**"));
        Ok(())
    }

    #[test]
    fn cache_gate_requires_one_exact_eight_of_eight_proof_per_wave() {
        let mut facts = comparison_eligible_facts();
        facts
            .llama_cpp_exact_prefix_cache_hit_requests_by_wave
            .pop();
        let missing_wave = derive_comparison_eligibility(facts);
        assert!(!missing_wave.eligible);
        assert!(
            missing_wave
                .reasons
                .iter()
                .any(|reason| reason.contains("covers 19 of 20 measured waves"))
        );
    }

    #[test]
    fn markdown_cells_cannot_break_tables_or_code_spans() {
        assert_eq!(markdown_cell("a|b`c\nd"), "a\\|b\\`c d");
    }

    #[test]
    fn phase_validation_rejects_missing_or_double_counted_decode_evidence() -> Result<()> {
        let mut phases = super::super::MacPrefixDemoPhaseTimingEvidence {
            queue_admission_ms: 1.0,
            prefix_lookup_attach_ms: 2.0,
            cache_diagnostics_ms: 0.5,
            suffix_prefill_ms: 0.0,
            first_token_model_forward_ms: 4.0,
            single_step_decode_ms: vec![3.0; MAC_PREFIX_DEMO_DECODE_TOKENS - 1],
            single_step_decode_total_ms: 3.0 * (MAC_PREFIX_DEMO_DECODE_TOKENS - 1) as f64,
            boundary: "exclusive fixture".to_owned(),
        };
        validate_phase_timings(0, &phases)?;
        phases.single_step_decode_total_ms += 1.0;
        assert!(validate_phase_timings(0, &phases).is_err());
        phases.single_step_decode_total_ms -= 1.0;
        phases.single_step_decode_ms.pop();
        assert!(validate_phase_timings(0, &phases).is_err());
        Ok(())
    }

    #[test]
    fn artifact_paths_are_normalized_and_aliases_rejected() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let json = directory.path().join("report.json");
        let alias = directory.path().join(".").join("report.json");
        assert!(normalize_artifact_pair(&json, &alias).is_err());

        fs::write(&json, b"existing")?;
        let hard_link = directory.path().join("report.md");
        fs::hard_link(&json, &hard_link)?;
        assert!(normalize_artifact_pair(&json, &hard_link).is_err());
        Ok(())
    }

    #[test]
    fn artifact_alias_checks_fail_closed() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let json = directory.path().join("report.json");
        let markdown = directory.path().join("report.md");

        // Two targets that do not exist yet are not aliases of each other.
        assert!(normalize_artifact_pair(&json, &markdown).is_ok());

        // Two distinct existing files are not aliases of each other.
        fs::write(&json, b"json")?;
        fs::write(&markdown, b"markdown")?;
        assert!(normalize_artifact_pair(&json, &markdown).is_ok());

        // A hard-linked pair is refused.
        let hard_link = directory.path().join("alias.md");
        fs::hard_link(&json, &hard_link)?;
        let error = normalize_artifact_pair(&json, &hard_link)
            .expect_err("a hard-linked pair must be refused");
        assert!(
            format!("{error:#}").contains("hard-link aliases"),
            "{error:#}"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_hard_linked_pair_is_refused_without_read_permission() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir()?;
        let json = directory.path().join("report.json");
        let markdown = directory.path().join("report.md");
        fs::write(&json, b"json")?;
        fs::hard_link(&json, &markdown)?;
        // The identity of a file does not need read permission, so the pair is
        // still refused when neither name can be opened.
        fs::set_permissions(&json, fs::Permissions::from_mode(0o000))?;
        fs::set_permissions(&markdown, fs::Permissions::from_mode(0o000))?;
        let error = normalize_artifact_pair(&json, &markdown)
            .expect_err("a hard-linked pair must be refused");
        assert!(
            format!("{error:#}").contains("hard-link aliases"),
            "{error:#}"
        );
        // Restore the mode before the temporary directory is dropped.
        fs::set_permissions(&json, fs::Permissions::from_mode(0o600))?;
        fs::set_permissions(&markdown, fs::Permissions::from_mode(0o600))?;
        Ok(())
    }

    #[cfg(not(unix))]
    #[test]
    fn an_existing_pair_that_cannot_be_opened_is_refused() -> Result<()> {
        use std::os::windows::fs::OpenOptionsExt;

        let directory = tempfile::tempdir()?;
        let json = directory.path().join("report.json");
        let markdown = directory.path().join("report.md");
        fs::write(&json, b"json")?;
        fs::write(&markdown, b"markdown")?;
        // The held file is open for reading and writing and shares nothing, so
        // the comparison of the two targets fails and the pair is refused.
        let held = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .share_mode(0)
            .open(&json)?;
        assert!(normalize_artifact_pair(&json, &markdown).is_err());
        drop(held);
        Ok(())
    }

    #[test]
    fn paired_commit_replaces_both_outputs() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let json = directory.path().join("report.json");
        let markdown = directory.path().join("report.md");
        fs::write(&json, b"old-json")?;
        fs::write(&markdown, b"old-markdown")?;
        let json_temp = prepare_atomic_file(&json, b"new-json")?;
        let markdown_temp = prepare_atomic_file(&markdown, b"new-markdown")?;
        commit_artifact_pair(json_temp, &json, markdown_temp, &markdown)?;
        assert_eq!(fs::read(&json)?, b"new-json");
        assert_eq!(fs::read(&markdown)?, b"new-markdown");
        Ok(())
    }
}
