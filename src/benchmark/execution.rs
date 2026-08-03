//! Auditable single-request benchmark execution.
//!
//! `MacLLM` exposes the three timing boundaries needed by [`KernelResult`].
//! `llama-bench` deliberately remains partial: its JSON output contains
//! aggregate prompt-processing and token-generation rates, but no request
//! TTFT or generated token IDs. This module keeps that missing evidence
//! explicit instead of manufacturing values from unrelated timers.

use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

use crate::{
    backend::{DTypeKind, DeviceKind},
    generation::{FinishReason, GenerationConfig, generate_with_prefill_token_quantum},
    runtime::{LoadedRuntime, ModelLoadConfig},
    sampling::{Sampler, SamplingConfig},
};

use super::{
    BenchmarkDType, BenchmarkDevice, ImplementationKind, KernelResult, LlamaBenchKind,
    LlamaBenchRecord, LlamaFlashAttention, PrefixCacheMode, VerifiedScoreboard,
    current_process_memory, parse_llama_bench_json, summarize,
};

pub use super::memory::ProcessMemoryObservation;

/// Exact program and arguments used for one external benchmark invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkCommand {
    pub program: PathBuf,
    pub arguments: Vec<String>,
}

impl BenchmarkCommand {
    /// Convert the executable and every argument into the report's argv form.
    pub fn report_argv(&self) -> Result<Vec<String>> {
        let program = utf8_path(&self.program, "benchmark executable")?;
        let mut argv = Vec::with_capacity(self.arguments.len() + 1);
        argv.push(program.to_owned());
        argv.extend(self.arguments.iter().cloned());
        Ok(argv)
    }

    fn execute(&self) -> Result<std::process::Output> {
        Command::new(&self.program)
            .args(&self.arguments)
            .output()
            .with_context(|| format!("failed to execute {}", self.program.display()))
    }
}

/// Complete `MacLLM` kernel evidence plus data needed by later parity/memory
/// report stages.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MacllmKernelExecution {
    pub result: KernelResult,
    /// The first `workload.decode_steps` completion token IDs. One additional
    /// token is generated solely so the runtime executes exactly that many
    /// single-token decode forwards.
    pub parity_token_ids: Vec<u32>,
    pub loaded_memory: ProcessMemoryObservation,
    pub final_memory: ProcessMemoryObservation,
}

/// Honest subset of evidence emitted by repeated `llama-bench -r 1` runs.
///
/// This is intentionally not a [`KernelResult`]: `llama-bench` does not emit
/// TTFT or generated token IDs and uses synthetic token IDs rather than the
/// manifest's prompt bytes.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaBenchKernelEvidence {
    pub implementation: ImplementationKind,
    pub measurement_boundary: String,
    pub command: Vec<String>,
    pub prompt_tokens: usize,
    pub decode_steps: usize,
    pub prefill_tokens_per_second_samples: Vec<f64>,
    pub decode_tokens_per_second_samples: Vec<f64>,
    pub records: Vec<LlamaBenchRecord>,
}

impl LlamaBenchKernelEvidence {
    /// Refuse to erase the fact that `llama-bench` has no TTFT samples.
    pub fn try_into_kernel_result(self) -> Result<KernelResult> {
        bail!(
            "llama-bench JSON has no per-request time-to-first-token samples; \
             a complete KernelResult requires a separate truthful boundary/schema"
        )
    }
}

#[derive(Debug)]
struct MacllmSample {
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    time_to_first_token_ms: f64,
    parity_token_ids: Vec<u32>,
}

/// Load the verified model once, discard the declared warmups, then retain one
/// raw sample per single-request generation.
pub fn run_macllm_kernel(verified: &VerifiedScoreboard) -> Result<MacllmKernelExecution> {
    ensure!(
        verified.manifest.workload.prefix_cache == PrefixCacheMode::Disabled,
        "MacLLM kernel scoreboard currently requires prefix_cache=disabled"
    );
    let config = macllm_model_load_config(verified);
    let runtime = LoadedRuntime::load(&config).context("failed to load MacLLM benchmark model")?;
    ensure!(
        runtime.adapter.family() == verified.manifest.model.architecture,
        "loaded MacLLM adapter family changed after scoreboard verification"
    );
    let encoded = runtime.encode(
        verified.prompt_text.clone(),
        verified.manifest.workload.prompt.add_special_tokens,
    )?;
    ensure!(
        encoded == verified.prompt_token_ids,
        "MacLLM prompt tokenization changed after scoreboard verification"
    );
    let loaded_memory = current_process_memory()?;

    for _ in 0..verified.manifest.workload.warmup_waves {
        let _sample = run_macllm_sample(&runtime, verified)?;
    }

    let sample_count = verified.manifest.workload.samples_per_concurrency;
    let mut prefill_samples = Vec::with_capacity(sample_count);
    let mut decode_samples = Vec::with_capacity(sample_count);
    let mut ttft_samples = Vec::with_capacity(sample_count);
    let mut parity_token_ids: Option<Vec<u32>> = None;
    for sample_index in 0..sample_count {
        let sample = run_macllm_sample(&runtime, verified)
            .with_context(|| format!("MacLLM kernel sample {} failed", sample_index + 1))?;
        if let Some(expected) = &parity_token_ids {
            ensure!(
                sample.parity_token_ids == *expected,
                "greedy MacLLM token IDs changed at kernel sample {}",
                sample_index + 1
            );
        } else {
            parity_token_ids = Some(sample.parity_token_ids.clone());
        }
        prefill_samples.push(sample.prefill_tokens_per_second);
        decode_samples.push(sample.decode_tokens_per_second);
        ttft_samples.push(sample.time_to_first_token_ms);
    }
    let final_memory = current_process_memory()?;
    let parity_token_ids = parity_token_ids.context("MacLLM kernel retained no token evidence")?;

    let result = KernelResult {
        implementation: ImplementationKind::Macllm,
        measurement_boundary: concat!(
            "in-process runtime timers: model loading, prompt tokenization and session allocation ",
            "excluded; prefill TPS counts manifest prompt tokens over model prefill forwards; ",
            "TTFT starts before cold session reset and ends after greedy sampling; decode TPS counts ",
            "exactly workload.decode_steps single-token model forwards, requiring one unreported ",
            "extra completion token"
        )
        .to_owned(),
        command: macllm_report_argv(verified)?,
        prompt_tokens: verified.prompt_token_ids.len(),
        decode_steps: verified.manifest.workload.decode_steps,
        prefill_tokens_per_second: summarize(&prefill_samples)?,
        decode_tokens_per_second: summarize(&decode_samples)?,
        time_to_first_token_ms: summarize(&ttft_samples)?,
        prefill_tokens_per_second_samples: prefill_samples,
        decode_tokens_per_second_samples: decode_samples,
        time_to_first_token_ms_samples: ttft_samples,
    };

    Ok(MacllmKernelExecution {
        result,
        parity_token_ids,
        loaded_memory,
        final_memory,
    })
}

fn run_macllm_sample(
    runtime: &LoadedRuntime,
    verified: &VerifiedScoreboard,
) -> Result<MacllmSample> {
    let workload = &verified.manifest.workload;
    let mut session = runtime.model.new_session()?;
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: workload.sampling.temperature,
        top_k: workload.sampling.top_k,
        top_p: workload.sampling.top_p,
        repetition_penalty: workload.sampling.repetition_penalty,
        seed: workload.sampling.seed,
    })?;
    let timed_token_count = workload
        .decode_steps
        .checked_add(1)
        .context("kernel timed token count overflow")?;
    let prefill_quantum = NonZeroUsize::new(verified.manifest.runtime.prefill_token_quantum)
        .context("benchmark prefill token quantum is zero")?;
    let mut token_ids = Vec::with_capacity(timed_token_count);
    let metrics = generate_with_prefill_token_quantum(
        runtime.model.as_ref(),
        session.as_mut(),
        &verified.prompt_token_ids,
        &mut sampler,
        GenerationConfig {
            max_new_tokens: timed_token_count,
            ignore_stop_tokens: workload.sampling.ignore_stop_tokens,
        },
        prefill_quantum,
        |token| {
            token_ids.push(token);
            Ok(())
        },
    )?;
    ensure!(
        metrics.finish_reason == FinishReason::Length,
        "MacLLM kernel did not finish at the fixed token budget"
    );
    ensure!(
        metrics.prompt_tokens == verified.prompt_token_ids.len()
            && metrics.cached_prompt_tokens == 0,
        "MacLLM kernel did not execute the complete cold prompt"
    );
    ensure!(
        metrics.decode_steps == workload.decode_steps,
        "MacLLM executed {} decode forwards; expected {}",
        metrics.decode_steps,
        workload.decode_steps
    );
    ensure!(
        token_ids.len() == timed_token_count,
        "MacLLM emitted {} timed tokens; expected {timed_token_count}",
        token_ids.len()
    );
    token_ids.truncate(workload.decode_steps);
    Ok(MacllmSample {
        prefill_tokens_per_second: metrics.prefill_tokens_per_second(),
        decode_tokens_per_second: metrics.decode_tokens_per_second(),
        time_to_first_token_ms: metrics.time_to_first_token.as_secs_f64() * 1_000.0,
        parity_token_ids: token_ids,
    })
}

fn macllm_model_load_config(verified: &VerifiedScoreboard) -> ModelLoadConfig {
    let runtime = &verified.manifest.runtime;
    let mut config = ModelLoadConfig::new(
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
    config.kv_cache_block_tokens = runtime.kv_cache_block_tokens;
    config.kv_cache_slab_blocks = runtime.kv_cache_slab_blocks;
    config.kv_cache_capacity_tokens = Some(runtime.kv_cache_capacity_tokens);
    config.prefix_cache_capacity_tokens = 0;
    config
}

fn macllm_report_argv(verified: &VerifiedScoreboard) -> Result<Vec<String>> {
    Ok(vec![
        "in-process:macllm_infer::benchmark::run_macllm_kernel".to_owned(),
        "--manifest".to_owned(),
        utf8_path(&verified.manifest_path, "scoreboard manifest")?.to_owned(),
        "--warmup-runs".to_owned(),
        verified.manifest.workload.warmup_waves.to_string(),
        "--samples".to_owned(),
        verified
            .manifest
            .workload
            .samples_per_concurrency
            .to_string(),
        "--timed-decode-forwards".to_owned(),
        verified.manifest.workload.decode_steps.to_string(),
    ])
}

/// Build the exact isolated `llama-bench` command used once per raw sample.
/// The tool performs its own single warmup because `--no-warmup` is omitted.
pub fn build_llama_bench_command(verified: &VerifiedScoreboard) -> Result<BenchmarkCommand> {
    ensure!(
        verified.manifest.workload.prefix_cache == PrefixCacheMode::Disabled,
        "llama-bench kernel scoreboard currently requires prefix_cache=disabled"
    );
    ensure!(
        verified.manifest.workload.warmup_waves == 1,
        "llama-bench exposes only an on/off internal warmup; reproducible v1 execution requires warmup_waves=1"
    );
    let threads = verified
        .host
        .performance_cores
        .or(verified.host.total_cores)
        .context("verified host has no CPU core count for pinned llama-bench threads")?;
    ensure!(threads > 0, "llama-bench thread count must be positive");
    let llama = &verified.manifest.llama_cpp;
    let flash_attention = match llama.flash_attention {
        LlamaFlashAttention::On => "on",
        LlamaFlashAttention::Off => "off",
        LlamaFlashAttention::Auto => "auto",
    };
    let batch = verified.manifest.runtime.prefill_token_quantum;
    Ok(BenchmarkCommand {
        program: verified.llama_bench.canonical_path.clone(),
        arguments: vec![
            "--model".to_owned(),
            utf8_path(&verified.model.file.canonical_path, "verified model")?.to_owned(),
            "--n-prompt".to_owned(),
            verified.prompt_token_ids.len().to_string(),
            "--n-gen".to_owned(),
            verified.manifest.workload.decode_steps.to_string(),
            "--batch-size".to_owned(),
            batch.to_string(),
            "--ubatch-size".to_owned(),
            batch.to_string(),
            "--threads".to_owned(),
            threads.to_string(),
            "--n-gpu-layers".to_owned(),
            llama.gpu_layers.to_string(),
            "--flash-attn".to_owned(),
            flash_attention.to_owned(),
            "--cache-type-k".to_owned(),
            llama.cache_type_k.clone(),
            "--cache-type-v".to_owned(),
            llama.cache_type_v.clone(),
            "--mmap".to_owned(),
            u8::from(llama.mmap).to_string(),
            "--repetitions".to_owned(),
            "1".to_owned(),
            "--output".to_owned(),
            "json".to_owned(),
        ],
    })
}

/// Run one isolated `llama-bench -r 1` process per retained raw sample.
///
/// The returned evidence can publish truthful pp/tg distributions, but cannot
/// pass the current complete [`KernelResult`] schema because TTFT is absent.
pub fn run_llama_bench_kernel_evidence(
    verified: &VerifiedScoreboard,
) -> Result<LlamaBenchKernelEvidence> {
    let command = build_llama_bench_command(verified)?;
    let sample_count = verified.manifest.workload.samples_per_concurrency;
    let mut prefill_samples = Vec::with_capacity(sample_count);
    let mut decode_samples = Vec::with_capacity(sample_count);
    let mut records = Vec::with_capacity(sample_count * 2);
    for sample_index in 0..sample_count {
        let output = command
            .execute()
            .with_context(|| format!("llama-bench kernel sample {} failed", sample_index + 1))?;
        ensure!(
            output.status.success(),
            "llama-bench sample {} exited with {}: {}",
            sample_index + 1,
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
        let stdout =
            std::str::from_utf8(&output.stdout).context("llama-bench stdout was not UTF-8")?;
        let sample_records = parse_llama_bench_json(stdout)?;
        let (prefill, decode) = validate_llama_bench_sample(verified, &sample_records)?;
        prefill_samples.push(prefill.average_tokens_per_second);
        decode_samples.push(decode.average_tokens_per_second);
        records.extend(sample_records);
    }
    Ok(LlamaBenchKernelEvidence {
        implementation: ImplementationKind::LlamaCpp,
        measurement_boundary: concat!(
            "llama-bench isolated -r 1 kernel aggregates: model load excluded by llama-bench; ",
            "one internal warmup per retained process; pp/tg use synthetic token IDs and do not ",
            "measure the manifest prompt bytes; llama-bench emits no TTFT or generated token IDs"
        )
        .to_owned(),
        command: command.report_argv()?,
        prompt_tokens: verified.prompt_token_ids.len(),
        decode_steps: verified.manifest.workload.decode_steps,
        prefill_tokens_per_second_samples: prefill_samples,
        decode_tokens_per_second_samples: decode_samples,
        records,
    })
}

/// Strict compatibility entry point for callers expecting a complete result.
/// It fails before expensive work because the required TTFT evidence does not
/// exist in `llama-bench` output.
pub fn run_llama_cpp_kernel(_verified: &VerifiedScoreboard) -> Result<KernelResult> {
    bail!(
        "cannot construct llama.cpp KernelResult from llama-bench: JSON has pp/tg rates but no per-request TTFT; call run_llama_bench_kernel_evidence for the truthful subset"
    )
}

fn validate_llama_bench_sample<'a>(
    verified: &VerifiedScoreboard,
    records: &'a [LlamaBenchRecord],
) -> Result<(&'a LlamaBenchRecord, &'a LlamaBenchRecord)> {
    ensure!(
        records.len() == 2,
        "llama-bench sample must contain exactly one prefill and one decode record; found {}",
        records.len()
    );
    let prefill = records
        .iter()
        .find(|record| record.kind() == LlamaBenchKind::Prefill)
        .context("llama-bench sample has no pure prefill record")?;
    let decode = records
        .iter()
        .find(|record| record.kind() == LlamaBenchKind::Decode)
        .context("llama-bench sample has no pure decode record")?;
    ensure!(
        records.iter().all(|record| matches!(
            record.kind(),
            LlamaBenchKind::Prefill | LlamaBenchKind::Decode
        )),
        "llama-bench sample contains a mixed or empty workload"
    );
    for record in records {
        verify_llama_bench_record_identity(verified, record)?;
    }
    ensure!(
        prefill.prompt_tokens == verified.prompt_token_ids.len() && prefill.generation_tokens == 0,
        "llama-bench prefill shape does not match the manifest"
    );
    ensure!(
        decode.prompt_tokens == 0
            && decode.generation_tokens == verified.manifest.workload.decode_steps,
        "llama-bench decode shape does not match the manifest"
    );
    Ok((prefill, decode))
}

fn verify_llama_bench_record_identity(
    verified: &VerifiedScoreboard,
    record: &LlamaBenchRecord,
) -> Result<()> {
    let llama = &verified.manifest.llama_cpp;
    ensure!(
        record.build_number == llama.build && llama.commit.starts_with(&record.build_commit),
        "llama-bench record build identity does not match the verified binary"
    );
    ensure!(
        Path::new(&record.model_filename).file_name()
            == verified.model.file.canonical_path.file_name(),
        "llama-bench record model filename does not match the verified GGUF"
    );
    // `llama-bench` calls this field `model_size`, but it is the decoded
    // tensor/model payload size rather than the complete GGUF file length.
    // Exact file identity is already enforced by `VerifiedScoreboard` SHA-256.
    let threads = verified
        .host
        .performance_cores
        .or(verified.host.total_cores)
        .context("verified host has no CPU core count")?;
    let batch = verified.manifest.runtime.prefill_token_quantum;
    ensure!(
        record.n_batch == batch && record.n_ubatch == batch && record.n_threads == threads,
        "llama-bench record execution geometry differs from the pinned command"
    );
    ensure!(
        record.n_gpu_layers == llama.gpu_layers && record.mmap == llama.mmap,
        "llama-bench record offload/mmap policy differs from the manifest"
    );
    ensure!(
        record.flash_attention == llama.flash_attention,
        "llama-bench flash-attention state differs from the manifest"
    );
    Ok(())
}

fn utf8_path<'a>(path: &'a Path, label: &str) -> Result<&'a str> {
    path.as_os_str()
        .to_str()
        .with_context(|| format!("{label} path is not valid UTF-8: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{
        FileProvenance, GreedySamplingSpec, HostRequirement, LlamaCppRequirement, LlamaCppVersion,
        ModelProvenance, ModelRequirement, PromptRequirement, RuntimeBenchmarkSpec, SanitizedHost,
        ScoreboardManifest, WorkloadSpec,
    };

    #[allow(clippy::too_many_lines)]
    fn verified_fixture() -> VerifiedScoreboard {
        let model_file = FileProvenance {
            canonical_path: PathBuf::from("/models/fixture.gguf"),
            filename: "fixture.gguf".to_owned(),
            size_bytes: 123,
            sha256: "a".repeat(64),
        };
        VerifiedScoreboard {
            manifest_path: PathBuf::from("/profiles/fixture.json"),
            manifest: ScoreboardManifest {
                schema_version: 1,
                name: "fixture".to_owned(),
                host: HostRequirement {
                    machine_model: "Mac1,1".to_owned(),
                    chip: "Apple Test".to_owned(),
                    memory_bytes: 1024,
                    architecture: "aarch64".to_owned(),
                },
                model: ModelRequirement {
                    path: PathBuf::from("/models"),
                    repository: "owner/model".to_owned(),
                    revision: "b".repeat(40),
                    filename: "fixture.gguf".to_owned(),
                    size_bytes: 123,
                    sha256: "a".repeat(64),
                    architecture: "fixture".to_owned(),
                    quantization: "Q4_K_M".to_owned(),
                },
                llama_cpp: LlamaCppRequirement {
                    binary_directory: PathBuf::from("/tools"),
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
                        path: PathBuf::from("/prompts/fixture.txt"),
                        sha256: "c".repeat(64),
                        token_count: 5,
                        token_ids_sha256: "d".repeat(64),
                        add_special_tokens: false,
                        strip_final_newline: false,
                    },
                    decode_steps: 7,
                    sampling: GreedySamplingSpec {
                        temperature: 0.0,
                        top_k: 0,
                        top_p: 1.0,
                        repetition_penalty: 1.0,
                        seed: 42,
                        ignore_stop_tokens: true,
                    },
                    concurrency: vec![1, 4, 8],
                    warmup_waves: 1,
                    samples_per_concurrency: 40,
                    prefix_cache: PrefixCacheMode::Disabled,
                },
                runtime: RuntimeBenchmarkSpec {
                    device: BenchmarkDevice::Metal,
                    dtype: BenchmarkDType::F32,
                    prefill_token_quantum: 128,
                    kv_cache_block_tokens: 16,
                    kv_cache_slab_blocks: 8,
                    kv_cache_capacity_tokens: 65_536,
                },
                fair_prefix: None,
            },
            host: SanitizedHost {
                machine_name: "Mac".to_owned(),
                machine_model: "Mac1,1".to_owned(),
                chip: "Apple Test".to_owned(),
                memory_bytes: 1024,
                total_cores: Some(12),
                performance_cores: Some(8),
                efficiency_cores: Some(4),
                os_version: "1.0".to_owned(),
                os_build: "A1".to_owned(),
                architecture: "aarch64".to_owned(),
            },
            model: ModelProvenance {
                file: model_file,
                snapshot: None,
                architecture: "fixture".to_owned(),
                quantization: "Q4_K_M".to_owned(),
            },
            prompt: FileProvenance {
                canonical_path: PathBuf::from("/prompts/fixture.txt"),
                filename: "fixture.txt".to_owned(),
                size_bytes: 4,
                sha256: "c".repeat(64),
            },
            prompt_text: "test".to_owned(),
            prompt_token_ids: vec![1, 2, 3, 4, 5],
            llama_cpp_version: LlamaCppVersion {
                build: 10_050,
                commit_abbreviation: "b15ca938a".to_owned(),
                compiler_and_target: None,
            },
            llama_cpp_commit: "b15ca938ad00aa6b3ee6c2edda7363fd02826b18".to_owned(),
            llama_bench: FileProvenance {
                canonical_path: PathBuf::from("/tools/llama-bench"),
                filename: "llama-bench".to_owned(),
                size_bytes: 100,
                sha256: "e".repeat(64),
            },
            llama_server: FileProvenance {
                canonical_path: PathBuf::from("/tools/llama-server"),
                filename: "llama-server".to_owned(),
                size_bytes: 100,
                sha256: "f".repeat(64),
            },
        }
    }

    fn bench_record(kind: LlamaBenchKind) -> LlamaBenchRecord {
        let (prompt_tokens, generation_tokens, rate) = match kind {
            LlamaBenchKind::Prefill => (5, 0, 100.0),
            LlamaBenchKind::Decode => (0, 7, 20.0),
            LlamaBenchKind::Empty | LlamaBenchKind::Mixed => unreachable!(),
        };
        LlamaBenchRecord {
            build_commit: "b15ca938a".to_owned(),
            build_number: 10_050,
            model_filename: "/models/fixture.gguf".to_owned(),
            model_type: "fixture".to_owned(),
            model_size_bytes: 123,
            model_parameters: 42,
            backend: "Metal".to_owned(),
            n_batch: 128,
            n_ubatch: 128,
            n_threads: 8,
            n_gpu_layers: 99,
            flash_attention: LlamaFlashAttention::Auto,
            mmap: true,
            prompt_tokens,
            generation_tokens,
            average_nanoseconds: 1_000,
            standard_deviation_nanoseconds: 0,
            average_tokens_per_second: rate,
            standard_deviation_tokens_per_second: 0.0,
        }
    }

    #[test]
    fn llama_bench_command_pins_every_execution_policy() -> Result<()> {
        let command = build_llama_bench_command(&verified_fixture())?;
        assert_eq!(command.program, Path::new("/tools/llama-bench"));
        assert_eq!(
            command.arguments,
            [
                "--model",
                "/models/fixture.gguf",
                "--n-prompt",
                "5",
                "--n-gen",
                "7",
                "--batch-size",
                "128",
                "--ubatch-size",
                "128",
                "--threads",
                "8",
                "--n-gpu-layers",
                "99",
                "--flash-attn",
                "auto",
                "--cache-type-k",
                "f16",
                "--cache-type-v",
                "f16",
                "--mmap",
                "1",
                "--repetitions",
                "1",
                "--output",
                "json",
            ]
        );
        Ok(())
    }

    #[test]
    fn llama_sample_requires_exact_prefill_decode_and_provenance() -> Result<()> {
        let verified = verified_fixture();
        let records = vec![
            bench_record(LlamaBenchKind::Prefill),
            bench_record(LlamaBenchKind::Decode),
        ];
        let (prefill, decode) = validate_llama_bench_sample(&verified, &records)?;
        assert!((prefill.average_tokens_per_second - 100.0).abs() <= f64::EPSILON);
        assert!((decode.average_tokens_per_second - 20.0).abs() <= f64::EPSILON);

        let mut wrong = records;
        wrong[1].model_filename = "/models/wrong.gguf".to_owned();
        assert!(validate_llama_bench_sample(&verified, &wrong).is_err());
        Ok(())
    }

    #[test]
    fn complete_llama_kernel_result_fails_closed_without_ttft() {
        let error = run_llama_cpp_kernel(&verified_fixture()).unwrap_err();
        assert!(error.to_string().contains("no per-request TTFT"));
    }

    #[test]
    fn macllm_load_config_copies_manifest_geometry_and_disables_prefixes() {
        let verified = verified_fixture();
        let config = macllm_model_load_config(&verified);
        assert_eq!(config.model, Path::new("/models"));
        assert_eq!(config.device, DeviceKind::Metal);
        assert_eq!(config.dtype, Some(DTypeKind::F32));
        assert_eq!(config.kv_cache_block_tokens, 16);
        assert_eq!(config.kv_cache_slab_blocks, 8);
        assert_eq!(config.kv_cache_capacity_tokens, Some(65_536));
        assert_eq!(config.prefix_cache_capacity_tokens, 0);
    }
}
