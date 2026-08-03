//! Deterministic Markdown rendering for a validated scoreboard report.

use std::fmt::Write as _;

use anyhow::Result;

use super::{
    schema::{ImplementationKind, ScoreboardReport},
    stats::DistributionSummary,
};

/// Render a human-readable companion to the authoritative JSON report.
// Keep section order explicit so JSON-to-Markdown reviews remain mechanical.
#[allow(clippy::too_many_lines)]
pub fn render_markdown(report: &ScoreboardReport) -> Result<String> {
    report.validate()?;

    let mut output = String::new();
    output.push_str("# MacLLM reproducible scoreboard\n\n");
    output.push_str(
        "> This page is a view of the versioned JSON report. Raw request samples and exact \
commands remain authoritative in that JSON artifact.\n\n",
    );

    output.push_str("## Run identity\n\n");
    output.push_str("| Field | Value |\n|---|---|\n");
    table_row(&mut output, "Run ID", &report.run_id);
    table_row(&mut output, "Recorded at (UTC)", &report.recorded_at_utc);
    table_row(
        &mut output,
        "Schema version",
        &report.schema_version.to_string(),
    );

    output.push_str("\n## Sanitized host\n\n");
    output.push_str("| Field | Value |\n|---|---|\n");
    table_row(&mut output, "Machine name", &report.host.machine_name);
    table_row(&mut output, "Machine model", &report.host.machine_model);
    table_row(&mut output, "Chip", &report.host.chip);
    table_row(
        &mut output,
        "Memory",
        &format_bytes(report.host.memory_bytes),
    );
    table_row(
        &mut output,
        "CPU cores",
        &format_optional(report.host.total_cores),
    );
    table_row(
        &mut output,
        "Performance cores",
        &format_optional(report.host.performance_cores),
    );
    table_row(
        &mut output,
        "Efficiency cores",
        &format_optional(report.host.efficiency_cores),
    );
    table_row(
        &mut output,
        "macOS",
        &format!("{} ({})", report.host.os_version, report.host.os_build),
    );
    table_row(&mut output, "Architecture", &report.host.architecture);

    output.push_str("\n## Model provenance\n\n");
    output.push_str("| Field | Value |\n|---|---|\n");
    table_row(&mut output, "File", &report.model.file.filename);
    table_row(
        &mut output,
        "Canonical path",
        &report.model.file.canonical_path.display().to_string(),
    );
    table_row(
        &mut output,
        "File size",
        &format_bytes(report.model.file.size_bytes),
    );
    table_row(&mut output, "SHA-256", &report.model.file.sha256);
    table_row(&mut output, "Architecture", &report.model.architecture);
    table_row(&mut output, "Quantization", &report.model.quantization);
    if let Some(snapshot) = &report.model.snapshot {
        table_row(&mut output, "Repository", &snapshot.repository);
        table_row(&mut output, "Revision", &snapshot.revision);
        if let Some(requested) = &snapshot.requested_revision {
            table_row(&mut output, "Requested revision", requested);
        }
    }

    output.push_str("\n## Implementations\n\n");
    output.push_str(
        "| Implementation | Version | Commit | Executable SHA-256 | Dirty |\n\
         |---|---|---|---|---|\n",
    );
    for implementation in &report.implementations {
        let dirty = implementation
            .dirty
            .map_or_else(|| "unknown".to_owned(), |value| value.to_string());
        writeln!(
            output,
            "| {} | {} | {} | {} | {} |",
            implementation_label(implementation.implementation),
            markdown_cell(&implementation.version),
            markdown_cell(&implementation.commit),
            markdown_cell(&implementation.executable.sha256),
            dirty
        )
        .expect("writing to a String cannot fail");
    }

    output.push_str("\n## Workload\n\n");
    output.push_str("| Field | Value |\n|---|---|\n");
    table_row(
        &mut output,
        "Prompt path",
        &report.workload.prompt.path.display().to_string(),
    );
    table_row(
        &mut output,
        "Prompt SHA-256",
        &report.workload.prompt.sha256,
    );
    table_row(
        &mut output,
        "Prompt tokens",
        &report.workload.prompt.token_count.to_string(),
    );
    table_row(
        &mut output,
        "Prompt token-ID SHA-256",
        &report.workload.prompt.token_ids_sha256,
    );
    table_row(
        &mut output,
        "Completion tokens",
        &report.workload.decode_steps.to_string(),
    );
    table_row(
        &mut output,
        "Concurrency",
        &format!("{:?}", report.workload.concurrency),
    );
    table_row(
        &mut output,
        "Warm-up waves",
        &report.workload.warmup_waves.to_string(),
    );
    table_row(
        &mut output,
        "Measured requests per concurrency",
        &report.workload.samples_per_concurrency.to_string(),
    );
    table_row(
        &mut output,
        "Greedy sampler",
        &format!(
            "temperature={}, top_k={}, top_p={}, repetition_penalty={}, seed={}, ignore_stop_tokens={}",
            report.workload.sampling.temperature,
            report.workload.sampling.top_k,
            report.workload.sampling.top_p,
            report.workload.sampling.repetition_penalty,
            report.workload.sampling.seed,
            report.workload.sampling.ignore_stop_tokens
        ),
    );
    table_row(
        &mut output,
        "Prefix cache",
        &format!("{:?}", report.workload.prefix_cache).to_ascii_lowercase(),
    );

    output.push_str("\n## Greedy token parity\n\n");
    output.push_str("| Field | Value |\n|---|---|\n");
    table_row(&mut output, "Exact match", &report.parity.exact.to_string());
    table_row(
        &mut output,
        "First mismatch",
        &report
            .parity
            .first_mismatch
            .map_or_else(|| "none".to_owned(), |index| index.to_string()),
    );
    table_row(
        &mut output,
        "MacLLM token IDs",
        &format!("{:?}", report.parity.macllm_token_ids),
    );
    table_row(
        &mut output,
        "llama.cpp token IDs",
        &format!("{:?}", report.parity.llama_cpp_token_ids),
    );

    output.push_str("\n## Complete kernel measurements\n\n");
    output.push_str(
        "> This complete boundary is available only for MacLLM and includes prefill, decode, and TTFT.\n\n",
    );
    output.push_str(
        "| Implementation | Metric | Samples | Mean | p50 | p95 | Max | Unit |\n\
         |---|---|---:|---:|---:|---:|---:|---|\n",
    );
    for result in &report.kernel_results {
        summary_row(
            &mut output,
            result.implementation,
            "Prefill",
            &result.prefill_tokens_per_second,
            "tokens/s",
        );
        summary_row(
            &mut output,
            result.implementation,
            "Decode",
            &result.decode_tokens_per_second,
            "tokens/s",
        );
        summary_row(
            &mut output,
            result.implementation,
            "Time to first token",
            &result.time_to_first_token_ms,
            "ms",
        );
    }

    output.push_str("\n### Exact kernel commands\n\n");
    for result in &report.kernel_results {
        writeln!(
            output,
            "- {} ({}): <code>{}</code>",
            implementation_label(result.implementation),
            markdown_cell(&result.measurement_boundary),
            html_code(&serde_json::to_string(&result.command)?)
        )
        .expect("writing to a String cannot fail");
    }

    output.push_str("\n## llama-bench throughput-only kernel measurements\n\n");
    output.push_str(
        "> `llama-bench` reports prefill/decode throughput over synthetic token IDs. It does not report TTFT or execute the manifest prompt bytes.\n\n",
    );
    output.push_str(
        "| Implementation | Metric | Samples | Mean | p50 | p95 | Max | Unit |\n\
         |---|---|---:|---:|---:|---:|---:|---|\n",
    );
    summary_row(
        &mut output,
        report.llama_bench_throughput.implementation,
        "Prefill",
        &report.llama_bench_throughput.prefill_tokens_per_second,
        "tokens/s",
    );
    summary_row(
        &mut output,
        report.llama_bench_throughput.implementation,
        "Decode",
        &report.llama_bench_throughput.decode_tokens_per_second,
        "tokens/s",
    );
    output.push_str("\n### Exact llama-bench command\n\n");
    writeln!(
        output,
        "- {} ({}): <code>{}</code>",
        implementation_label(report.llama_bench_throughput.implementation),
        markdown_cell(&report.llama_bench_throughput.measurement_boundary),
        html_code(&serde_json::to_string(
            &report.llama_bench_throughput.command
        )?)
    )
    .expect("writing to a String cannot fail");

    output.push_str("\n## Serving measurements\n\n");
    output.push_str(
        "| Implementation | Concurrency | Requests | Aggregate output | TTFT p50 | TTFT p95 | Latency p50 | Latency p95 | Wall time |\n\
         |---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for result in &report.serving_results {
        writeln!(
            output,
            "| {} | {} | {} | {:.3} tok/s | {:.3} ms | {:.3} ms | {:.3} ms | {:.3} ms | {:.3} ms |",
            implementation_label(result.implementation),
            result.concurrency,
            result.request_count,
            result.aggregate_output_tokens_per_second,
            result.time_to_first_token_ms.p50,
            result.time_to_first_token_ms.p95,
            result.latency_ms.p50,
            result.latency_ms.p95,
            result.wall_time_ms
        )
        .expect("writing to a String cannot fail");
    }

    output.push_str("\n## llama-server internal serving timings\n\n");
    output.push_str(
        "> These are timings reported by the terminal `llama-server` event. They are not external TTFT or request latency.\n\n",
    );
    output.push_str(
        "| Concurrency | Metric | Samples | Mean | p50 | p95 | Max | Unit |\n\
         |---:|---|---:|---:|---:|---:|---:|---|\n",
    );
    for result in &report.server_timing_results {
        server_timing_summary_row(
            &mut output,
            result.concurrency,
            "Prompt time",
            &result.prompt_ms,
            "ms",
        );
        server_timing_summary_row(
            &mut output,
            result.concurrency,
            "Prompt throughput",
            &result.prompt_tokens_per_second,
            "tokens/s",
        );
        server_timing_summary_row(
            &mut output,
            result.concurrency,
            "Predicted time",
            &result.predicted_ms,
            "ms",
        );
        server_timing_summary_row(
            &mut output,
            result.concurrency,
            "Predicted throughput",
            &result.predicted_tokens_per_second,
            "tokens/s",
        );
    }
    output.push_str("\n### Exact llama-server commands and boundaries\n\n");
    for result in &report.server_timing_results {
        writeln!(
            output,
            "- concurrency {} ({}): <code>{}</code>",
            result.concurrency,
            markdown_cell(&result.measurement_boundary),
            html_code(&serde_json::to_string(&result.command)?)
        )
        .expect("writing to a String cannot fail");
    }

    output.push_str("\n## Process memory\n\n");
    output.push_str(
        "| Implementation | Concurrency | Measurement boundary | Loaded steady | Peak physical footprint | Peak resident | Sample period |\n\
         |---|---:|---|---:|---:|---:|---:|\n",
    );
    for result in &report.memory_results {
        writeln!(
            output,
            "| {} | {} | {} | {} | {} | {} | {} ms |",
            implementation_label(result.implementation),
            result.concurrency,
            markdown_cell(&result.measurement_boundary),
            format_optional_bytes(result.loaded_steady_bytes),
            format_optional_bytes(result.peak_physical_footprint_bytes),
            format_optional_bytes(result.peak_resident_bytes),
            result.sampling_period_ms
        )
        .expect("writing to a String cannot fail");
    }

    output.push_str("\n## Warnings\n\n");
    if report.warnings.is_empty() {
        output.push_str("None.\n");
    } else {
        for warning in &report.warnings {
            writeln!(output, "- {}", markdown_cell(warning))
                .expect("writing to a String cannot fail");
        }
    }

    Ok(output)
}

fn summary_row(
    output: &mut String,
    implementation: ImplementationKind,
    metric: &str,
    summary: &DistributionSummary,
    unit: &str,
) {
    writeln!(
        output,
        "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {} |",
        implementation_label(implementation),
        metric,
        summary.sample_count,
        summary.mean,
        summary.p50,
        summary.p95,
        summary.max,
        unit
    )
    .expect("writing to a String cannot fail");
}

fn server_timing_summary_row(
    output: &mut String,
    concurrency: usize,
    metric: &str,
    summary: &DistributionSummary,
    unit: &str,
) {
    writeln!(
        output,
        "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {} |",
        concurrency,
        metric,
        summary.sample_count,
        summary.mean,
        summary.p50,
        summary.p95,
        summary.max,
        unit
    )
    .expect("writing to a String cannot fail");
}

const fn implementation_label(implementation: ImplementationKind) -> &'static str {
    match implementation {
        ImplementationKind::Macllm => "MacLLM",
        ImplementationKind::LlamaCpp => "llama.cpp",
    }
}

fn table_row(output: &mut String, field: &str, value: &str) {
    writeln!(
        output,
        "| {} | {} |",
        markdown_cell(field),
        markdown_cell(value)
    )
    .expect("writing to a String cannot fail");
}

fn markdown_cell(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace(['\r', '\n'], "<br>")
}

fn html_code(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn format_optional<T: ToString>(value: Option<T>) -> String {
    value.map_or_else(|| "not reported".to_owned(), |value| value.to_string())
}

fn format_optional_bytes(value: Option<u64>) -> String {
    value.map_or_else(|| "not reported".to_owned(), format_bytes)
}

fn format_bytes(bytes: u64) -> String {
    const MIB: f64 = 1_048_576.0;
    format!("{bytes} B ({:.2} MiB)", bytes as f64 / MIB)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::schema::{
        LlamaBenchThroughputResult, ServerTimingResult, ServerTimingSample,
    };
    use crate::benchmark::{
        FileProvenance, GREEDY_SEED, GreedySamplingSpec, ImplementationProvenance, KernelResult,
        MemoryResult, ModelProvenance, ParityResult, PrefixCacheMode, PromptRequirement,
        REQUIRED_CONCURRENCY, RequestSample, SCOREBOARD_SCHEMA_VERSION, SanitizedHost,
        ServingResult, WorkloadSpec, summarize,
    };

    fn provenance(filename: &str, digest_character: char) -> FileProvenance {
        FileProvenance {
            canonical_path: format!("/tmp/{filename}").into(),
            filename: filename.to_owned(),
            size_bytes: 1,
            sha256: digest_character.to_string().repeat(64),
        }
    }

    fn workload() -> WorkloadSpec {
        WorkloadSpec {
            prompt: PromptRequirement {
                path: "/tmp/prompt.txt".into(),
                sha256: "c".repeat(64),
                token_count: 16,
                token_ids_sha256: "d".repeat(64),
                add_special_tokens: false,
                strip_final_newline: false,
            },
            decode_steps: 8,
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
            samples_per_concurrency: 24,
            prefix_cache: PrefixCacheMode::Disabled,
        }
    }

    fn kernel_result(implementation: ImplementationKind) -> KernelResult {
        let prefill = vec![100.0; 24];
        let decode = vec![20.0; 24];
        let ttft = vec![10.0; 24];
        KernelResult {
            implementation,
            measurement_boundary: "model forward".to_owned(),
            command: vec!["benchmark".to_owned(), "--json".to_owned()],
            prompt_tokens: 16,
            decode_steps: 8,
            prefill_tokens_per_second: summarize(&prefill).unwrap(),
            decode_tokens_per_second: summarize(&decode).unwrap(),
            time_to_first_token_ms: summarize(&ttft).unwrap(),
            prefill_tokens_per_second_samples: prefill,
            decode_tokens_per_second_samples: decode,
            time_to_first_token_ms_samples: ttft,
        }
    }

    fn serving_result(implementation: ImplementationKind, concurrency: usize) -> ServingResult {
        let samples = (0..24)
            .map(|request| RequestSample {
                wave: request / concurrency,
                request,
                prompt_tokens: 16,
                completion_tokens: 8,
                time_to_first_token_ms: 10.0,
                latency_ms: 20.0,
            })
            .collect::<Vec<_>>();
        let ttft = samples
            .iter()
            .map(|sample| sample.time_to_first_token_ms)
            .collect::<Vec<_>>();
        let latency = samples
            .iter()
            .map(|sample| sample.latency_ms)
            .collect::<Vec<_>>();
        ServingResult {
            implementation,
            concurrency,
            request_count: samples.len(),
            total_completion_tokens: 192,
            wall_time_ms: 1000.0,
            aggregate_output_tokens_per_second: 192.0,
            time_to_first_token_ms: summarize(&ttft).unwrap(),
            latency_ms: summarize(&latency).unwrap(),
            samples,
        }
    }

    fn llama_bench_throughput() -> LlamaBenchThroughputResult {
        let prefill = vec![90.0; 24];
        let decode = vec![18.0; 24];
        LlamaBenchThroughputResult {
            implementation: ImplementationKind::LlamaCpp,
            measurement_boundary: "llama-bench synthetic pp/tg; no TTFT".to_owned(),
            command: vec![
                "llama-bench".to_owned(),
                "--output".to_owned(),
                "json".to_owned(),
            ],
            prompt_tokens: 16,
            decode_steps: 8,
            prefill_tokens_per_second: summarize(&prefill).unwrap(),
            decode_tokens_per_second: summarize(&decode).unwrap(),
            prefill_tokens_per_second_samples: prefill,
            decode_tokens_per_second_samples: decode,
        }
    }

    fn server_timing_result(concurrency: usize) -> ServerTimingResult {
        let samples = (0..24)
            .map(|request| ServerTimingSample {
                wave: request / concurrency,
                request,
                token_ids: vec![1; 8],
                cache_tokens: 0,
                prompt_tokens: 16,
                prompt_ms: 5.0,
                prompt_tokens_per_second: 3200.0,
                predicted_tokens: 8,
                predicted_ms: 10.0,
                predicted_tokens_per_second: 800.0,
            })
            .collect::<Vec<_>>();
        let prompt_ms = samples
            .iter()
            .map(|sample| sample.prompt_ms)
            .collect::<Vec<_>>();
        let prompt_tps = samples
            .iter()
            .map(|sample| sample.prompt_tokens_per_second)
            .collect::<Vec<_>>();
        let predicted_ms = samples
            .iter()
            .map(|sample| sample.predicted_ms)
            .collect::<Vec<_>>();
        let predicted_tps = samples
            .iter()
            .map(|sample| sample.predicted_tokens_per_second)
            .collect::<Vec<_>>();
        ServerTimingResult {
            implementation: ImplementationKind::LlamaCpp,
            concurrency,
            measurement_boundary: "terminal llama-server timings; not external TTFT".to_owned(),
            command: vec!["llama-server".to_owned(), "--no-cache-prompt".to_owned()],
            prompt_ms: summarize(&prompt_ms).unwrap(),
            prompt_tokens_per_second: summarize(&prompt_tps).unwrap(),
            predicted_ms: summarize(&predicted_ms).unwrap(),
            predicted_tokens_per_second: summarize(&predicted_tps).unwrap(),
            samples,
        }
    }

    fn valid_report() -> ScoreboardReport {
        let implementations = [ImplementationKind::Macllm, ImplementationKind::LlamaCpp];
        ScoreboardReport {
            schema_version: SCOREBOARD_SCHEMA_VERSION,
            run_id: "scoreboard-test".to_owned(),
            recorded_at_utc: "2026-07-21T12:00:00Z".to_owned(),
            host: SanitizedHost {
                machine_name: "MacBook Pro".to_owned(),
                machine_model: "Mac16,7".to_owned(),
                chip: "Apple M4 Pro".to_owned(),
                memory_bytes: 48 * 1024 * 1024 * 1024,
                total_cores: Some(14),
                performance_cores: Some(10),
                efficiency_cores: Some(4),
                os_version: "26.0".to_owned(),
                os_build: "25A1".to_owned(),
                architecture: "arm64".to_owned(),
            },
            model: ModelProvenance {
                file: provenance("model.gguf", 'e'),
                snapshot: None,
                architecture: "smollm3".to_owned(),
                quantization: "Q4_K_M".to_owned(),
            },
            implementations: implementations
                .into_iter()
                .enumerate()
                .map(|(index, implementation)| ImplementationProvenance {
                    implementation,
                    version: "1".to_owned(),
                    commit: if index == 0 {
                        "a".repeat(40)
                    } else {
                        "b".repeat(40)
                    },
                    executable: provenance(&format!("implementation-{index}"), 'f'),
                    dirty: Some(false),
                })
                .collect(),
            workload: workload(),
            kernel_results: vec![kernel_result(ImplementationKind::Macllm)],
            llama_bench_throughput: llama_bench_throughput(),
            serving_results: implementations
                .into_iter()
                .flat_map(|implementation| {
                    REQUIRED_CONCURRENCY
                        .into_iter()
                        .map(move |concurrency| serving_result(implementation, concurrency))
                })
                .collect(),
            server_timing_results: REQUIRED_CONCURRENCY
                .into_iter()
                .map(server_timing_result)
                .collect(),
            memory_results: implementations
                .into_iter()
                .flat_map(|implementation| {
                    REQUIRED_CONCURRENCY
                        .into_iter()
                        .map(move |concurrency| MemoryResult {
                            implementation,
                            concurrency,
                            measurement_boundary: "sampled process memory".to_owned(),
                            loaded_steady_bytes: Some(1024),
                            peak_physical_footprint_bytes: Some(2048),
                            peak_resident_bytes: Some(1536),
                            sampling_period_ms: 10,
                        })
                })
                .collect(),
            parity: ParityResult {
                prompt_tokens: 16,
                prompt_token_ids_sha256: "d".repeat(64),
                macllm_token_ids: vec![1; 8],
                llama_cpp_token_ids: vec![1; 8],
                exact: true,
                first_mismatch: None,
            },
            warnings: Vec::new(),
        }
    }

    #[test]
    fn markdown_cells_cannot_break_tables() {
        assert_eq!(markdown_cell("a|b\nc\\d"), "a\\|b<br>c\\\\d");
    }

    #[test]
    fn command_text_is_safe_inside_html_code() {
        assert_eq!(html_code("<arg>&value>"), "&lt;arg&gt;&amp;value&gt;");
    }

    #[test]
    fn byte_values_keep_the_exact_count() {
        assert_eq!(format_bytes(1_048_576), "1048576 B (1.00 MiB)");
        assert_eq!(format_optional_bytes(None), "not reported");
    }

    #[test]
    fn complete_report_is_validated_before_rendering() -> Result<()> {
        let report = valid_report();
        let markdown = render_markdown(&report)?;
        assert!(markdown.contains("# MacLLM reproducible scoreboard"));
        assert!(markdown.contains("| MacLLM |"));
        assert!(markdown.contains("| llama.cpp |"));
        assert!(markdown.contains("llama-bench throughput-only kernel measurements"));
        assert!(markdown.contains("llama-server internal serving timings"));
        assert!(markdown.contains("sampled process memory"));

        let mut corrupted = report;
        corrupted.serving_results[0].aggregate_output_tokens_per_second = 1.0;
        assert!(render_markdown(&corrupted).is_err());
        Ok(())
    }

    #[test]
    fn validated_json_and_markdown_artifacts_round_trip() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let json = directory.path().join("scoreboard.json");
        let markdown = directory.path().join("scoreboard.md");
        crate::benchmark::write_scoreboard_artifacts(&valid_report(), &json, &markdown)?;

        let reparsed: ScoreboardReport = serde_json::from_slice(&std::fs::read(json)?)?;
        reparsed.validate()?;
        let rendered = std::fs::read_to_string(markdown)?;
        assert!(rendered.contains("llama-bench throughput-only kernel measurements"));
        assert!(rendered.contains("llama-server internal serving timings"));
        Ok(())
    }

    #[test]
    fn report_rejects_cross_boundary_evidence_drift() {
        let mut wrong_kernel = valid_report();
        wrong_kernel.kernel_results[0].implementation = ImplementationKind::LlamaCpp;
        assert!(wrong_kernel.validate().is_err());

        let mut wrong_llama_bench = valid_report();
        wrong_llama_bench
            .llama_bench_throughput
            .prefill_tokens_per_second
            .mean = 1.0;
        assert!(wrong_llama_bench.validate().is_err());

        let mut cached_server_prompt = valid_report();
        cached_server_prompt.server_timing_results[0].samples[0].cache_tokens = 1;
        assert!(cached_server_prompt.validate().is_err());

        let mut wrong_server_summary = valid_report();
        wrong_server_summary.server_timing_results[0]
            .predicted_tokens_per_second
            .mean = 1.0;
        assert!(wrong_server_summary.validate().is_err());

        let mut wrong_server_tokens = valid_report();
        wrong_server_tokens.server_timing_results[0].samples[0].token_ids[0] = 9;
        assert!(wrong_server_tokens.validate().is_err());

        let mut missing_memory_boundary = valid_report();
        missing_memory_boundary.memory_results[0]
            .measurement_boundary
            .clear();
        assert!(missing_memory_boundary.validate().is_err());

        let mut incomplete_memory = valid_report();
        incomplete_memory.memory_results[0].peak_resident_bytes = None;
        assert!(incomplete_memory.validate().is_err());
    }

    #[test]
    fn llama_bench_schema_cannot_claim_ttft() {
        let mut value = serde_json::to_value(llama_bench_throughput()).unwrap();
        value["time_to_first_token_ms"] = serde_json::json!({
            "sample_count": 24,
            "mean": 1.0,
            "p50": 1.0,
            "p95": 1.0,
            "max": 1.0
        });
        assert!(serde_json::from_value::<LlamaBenchThroughputResult>(value).is_err());
    }
}
