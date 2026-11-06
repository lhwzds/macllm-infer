//! Reproducible, fail-closed `MacLLM` versus `llama.cpp` scoreboard primitives.

pub mod assemble;
pub mod execution;
pub mod fair_prefix;
pub(crate) mod file_stamp;
pub mod host;
pub mod llama_cpp;
pub mod llama_server;
pub mod memory;
pub mod prefix_demo;
pub mod prefix_demo_llama;
pub mod prefix_demo_mac;
pub mod prefix_demo_report;
pub mod provenance;
pub mod report;
pub mod runner;
pub mod schema;
pub mod serving;
pub mod stats;
pub mod task;

pub use assemble::{
    FileProvenanceGuard, MacllmProvenanceGuard, assemble_scoreboard_report,
    run_complete_scoreboard, write_scoreboard_artifacts,
};
pub use execution::{
    BenchmarkCommand, LlamaBenchKernelEvidence, MacllmKernelExecution, build_llama_bench_command,
    run_llama_bench_kernel_evidence, run_llama_cpp_kernel, run_macllm_kernel,
};
pub use fair_prefix::{
    FAIR_PREFIX_REPORT_SCHEMA_VERSION, FairPrefixComparison, FairPrefixContract,
    FairPrefixDistributionEvidence, FairPrefixEligibility, FairPrefixImplementationRun,
    FairPrefixPairEvidence, FairPrefixReplayEvidence, FairPrefixReport, FairPrefixRequestEvidence,
    FairPrefixRuntimeAttestation, FairPrefixTerminalEvidence, FairPrefixWaveEvidence,
    render_fair_prefix_markdown, run_fair_prefix_benchmark, write_fair_prefix_artifacts,
};
pub use host::{SanitizedHost, collect_macos_host, parse_system_profiler_hardware};
pub use llama_cpp::{
    LlamaBenchKind, LlamaBenchRecord, LlamaCppVersion, collect_llama_cpp_version,
    parse_homebrew_scm_revision, parse_llama_bench_json, parse_llama_bench_jsonl,
    parse_llama_cpp_version,
};
pub use llama_server::{
    LlamaServerExecution, LlamaServerRequestSample, LlamaServerWidthEvidence,
    build_llama_server_command, run_llama_server_serving,
};
pub use memory::{
    MEMORY_SAMPLING_PERIOD_MS, ProcessMemoryObservation, ProcessMemoryPeak, ProcessMemorySampler,
    current_process_memory, process_memory,
};
pub use prefix_demo::{
    DEFAULT_KV_CACHE_BLOCK_TOKENS as PREFIX_DEMO_KV_CACHE_BLOCK_TOKENS,
    DEFAULT_PREFILL_QUANTUM as PREFIX_DEMO_PREFILL_QUANTUM, DEFAULT_SHARED_PREFIX_TOKENS,
    MaterializedPrefixDemo, MaterializedPrefixPrompt, PREFIX_DEMO_SCHEMA_VERSION,
    PREFIX_DEMO_TASK_COUNT, PreparedPrefixDemo, PreparedPrefixPrompt, QuantizedPrefixSelection,
    REPOSITORY_PAYLOAD_ENCODING, RepositoryBundle, RepositoryBundleEvidence,
    build_repository_bundle, encode_repository_payload, materialize_prefix_demo,
    prepare_default_prefix_demo, prepare_prefix_demo, search_quantized_prefix_boundary,
};
pub use prefix_demo_llama::{
    LLAMA_CONTEXT_ALIGNMENT_PER_SLOT, LLAMA_PREFIX_DEMO_DEFAULT_MEASURED_WAVES,
    LLAMA_PREFIX_DEMO_MIN_MEASURED_WAVES, LlamaPrefixDemoCacheObservation,
    LlamaPrefixDemoContextGeometry, LlamaPrefixDemoDeterminismEvidence, LlamaPrefixDemoExecution,
    LlamaPrefixDemoPrompt, LlamaPrefixDemoProvenance, LlamaPrefixDemoRequestEvidence,
    LlamaPrefixDemoRequestPolicy, LlamaPrefixDemoTerminalEvidence,
    LlamaPrefixDemoTokenizerEvidence, LlamaPrefixDemoWaveEvidence, PREFIX_DEMO_CACHE_REUSE_TOKENS,
    PREFIX_DEMO_CONCURRENCY, PREFIX_DEMO_CONTEXT_HEADROOM_TOKENS, PrefixDemoTimingSummary,
    build_llama_prefix_demo_command, run_llama_prefix_demo,
};
pub use prefix_demo_mac::{
    MAC_PREFIX_DEMO_CACHE_CAPACITY_TOKENS, MAC_PREFIX_DEMO_CONCURRENCY,
    MAC_PREFIX_DEMO_DECODE_TOKENS, MAC_PREFIX_DEMO_DEFAULT_MEASURED_WAVES,
    MAC_PREFIX_DEMO_MIN_MEASURED_WAVES, MAC_PREFIX_DEMO_SCHEDULER_TOKEN_BUDGET,
    MacPrefixDemoCacheEvidence, MacPrefixDemoConfigEvidence, MacPrefixDemoDistributionEvidence,
    MacPrefixDemoExecution, MacPrefixDemoIdentityEvidence, MacPrefixDemoPhaseTimingEvidence,
    MacPrefixDemoRunConfig, MacPrefixDemoWaveEvidence, PrefixCacheCounterDeltaEvidence,
    PrefixCacheDiagnosticsEvidence, PrefixCachePeakEvidence, PrefixCacheSnapshotEvidence,
    PrefixDemoAggregateEvidence, PrefixDemoRequestEvidence, PrefixDemoUsageEvidence,
    run_macllm_prefix_demo,
};
pub use prefix_demo_report::{
    PREFIX_DEMO_REPORT_SCHEMA_VERSION, PrefixDemoComparison, PrefixDemoComparisonEligibility,
    PrefixDemoReport, assemble_prefix_demo_report, render_prefix_demo_markdown,
    write_prefix_demo_artifacts,
};
pub use provenance::{
    file_provenance, read_snapshot_provenance, sha256_bytes, token_ids_sha256, verify_file,
    verify_model_provenance, verify_prompt_file,
};
pub use report::render_markdown;
pub use runner::{ScoreboardVerificationSummary, VerifiedScoreboard, verify_scoreboard_manifest};
pub use schema::{
    BenchmarkDType, BenchmarkDevice, FairPrefixManifestSpec, FileProvenance, GREEDY_SEED,
    GreedySamplingSpec, HostRequirement, ImplementationKind, ImplementationProvenance,
    KernelResult, LlamaBenchThroughputResult, LlamaCppRequirement, LlamaFlashAttention,
    MIN_PERCENTILE_SAMPLES, MemoryResult, ModelProvenance, ModelRequirement, ParityResult,
    PrefixCacheMode, PromptRequirement, REQUIRED_CONCURRENCY, RequestSample, RuntimeBenchmarkSpec,
    SCOREBOARD_SCHEMA_VERSION, ScoreboardManifest, ScoreboardReport, ServerTimingResult,
    ServerTimingSample, ServingResult, SnapshotProvenance, WorkloadSpec, first_mismatch,
};
pub use serving::{MacllmServingExecution, build_macllm_serving_engine_config, run_macllm_serving};
pub use stats::{DistributionSummary, aggregate_throughput, nearest_rank, summarize};
pub use task::{
    TASK_BENCHMARK_SCHEMA_VERSION, TaskBenchmarkGuard, TaskBenchmarkReport, TaskEmbeddingInputType,
    TaskRepeat, TaskRuntimeConfig,
};
