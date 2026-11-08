use std::{
    io::{self, Write},
    net::{IpAddr, SocketAddr},
    num::NonZeroUsize,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum, builder::TypedValueParser};
use macllm_infer::{
    backend::{DTypeKind, DeviceKind},
    benchmark::{
        LlamaPrefixDemoPrompt, MAC_PREFIX_DEMO_CACHE_CAPACITY_TOKENS, MacPrefixDemoRunConfig,
        MacllmProvenanceGuard, MaterializedPrefixDemo, PREFIX_DEMO_CONCURRENCY, TaskBenchmarkGuard,
        TaskBenchmarkReport, TaskEmbeddingInputType, TaskRepeat, TaskRuntimeConfig,
        assemble_prefix_demo_report, materialize_prefix_demo, prepare_default_prefix_demo,
        run_complete_scoreboard, run_fair_prefix_benchmark, run_llama_bench_kernel_evidence,
        run_llama_prefix_demo, run_llama_server_serving, run_macllm_kernel, run_macllm_prefix_demo,
        run_macllm_serving, verify_scoreboard_manifest, write_fair_prefix_artifacts,
        write_prefix_demo_artifacts, write_scoreboard_artifacts,
    },
    embedding::embed_token_batch,
    generation::{GenerationConfig, GenerationMetrics, generate_with_prefill_token_quantum},
    hub::{DownloadSelection, download_model},
    models::qwen3_tasks::{Qwen3EmbeddingProfile, Qwen3RerankerProfile},
    prefill_attention::PrefillAttentionMode,
    rerank::{RerankScore, rerank_token_batch},
    runtime::{LoadedRuntime, ModelLoadConfig},
    sampling::{Sampler, SamplingConfig},
    server::{ReasoningFormat, SamplingProfile, ServerConfig, ServerOptions, serve},
    token_stream::TokenStream,
    weights::{ModelFiles, ModelFormat},
};
use serde::Serialize;

/// Default `--device` value for this build.
///
/// Builds with the `metal` feature select the Metal backend. Builds with the
/// `cuda` feature but without Metal select the CUDA backend. Builds without
/// either select the CPU backend, because such a binary cannot create a Metal
/// or CUDA device and would otherwise fail while loading the model.
const DEFAULT_DEVICE: &str = if cfg!(feature = "metal") {
    "metal"
} else if cfg!(feature = "cuda") {
    "cuda"
} else {
    "cpu"
};

#[derive(Debug, Parser)]
#[command(name = "macllm")]
// The public command name is fixed so that Windows reports `macllm` instead
// of the executable file name `macllm.exe`.
#[command(bin_name = "macllm")]
#[command(about = "An Apple Silicon-first Rust LLM inference runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Download a supported checkpoint from Hugging Face.
    Download {
        #[arg(default_value = "Qwen/Qwen2.5-0.5B-Instruct")]
        repo: String,
        /// Optional branch, tag or commit hash. Defaults to the repository main revision.
        #[arg(long)]
        revision: Option<String>,
        #[arg(short, long, default_value = "models/qwen2.5-0.5b-instruct")]
        output: PathBuf,
        /// Download this root-level GGUF file, or every shard when it is a canonical first shard.
        #[arg(long, value_name = "FILENAME")]
        gguf: Option<String>,
    },
    /// Parse configuration and report runtime geometry without loading weights.
    Inspect {
        #[arg(short, long)]
        model: PathBuf,
    },
    /// Generate text and stream decoded tokens to stdout.
    Run(RunArgs),
    /// Produce normalized Qwen3 embeddings as JSON.
    Embed(EmbedArgs),
    /// Score documents against one query with a Qwen3 reranker.
    Rerank(RerankArgs),
    /// Run release-only local diagnostics for generate, embed, and rerank.
    Bench(BenchArgs),
    /// Verify or run a manifest-pinned `MacLLM` versus `llama.cpp` scoreboard.
    Scoreboard(ScoreboardArgs),
    /// Run product demonstrations with fail-closed, publishable evidence.
    Demo(DemoArgs),
    /// Serve one model through a strict text-only Chat Completions-compatible subset.
    Serve(ServeArgs),
}

/// Parse the `--device` selector.
///
/// The accepted spellings are listed explicitly so that the command line
/// keeps the same values and help output, while the parsing itself stays
/// with the backend types.
fn device_kind_parser() -> impl TypedValueParser<Value = DeviceKind> {
    clap::builder::PossibleValuesParser::new(DeviceKind::all())
        .try_map(|value: String| value.parse::<DeviceKind>())
}

/// Parse the `--dtype` selector. See [`device_kind_parser`].
fn dtype_kind_parser() -> impl TypedValueParser<Value = DTypeKind> {
    clap::builder::PossibleValuesParser::new(DTypeKind::all())
        .try_map(|value: String| value.parse::<DTypeKind>())
}

#[derive(Debug, Clone, Args)]
struct RuntimeArgs {
    #[arg(short, long)]
    model: PathBuf,
    #[arg(
        long,
        value_parser = device_kind_parser(),
        default_value = DEFAULT_DEVICE
    )]
    device: DeviceKind,
    /// Activation dtype. Defaults to F32 for CPU or quantized GGUF, F16 otherwise.
    #[arg(long, value_parser = dtype_kind_parser())]
    dtype: Option<DTypeKind>,
    /// Tokens per logical full-attention K/V block.
    #[arg(
        long,
        default_value_t = ModelLoadConfig::DEFAULT_KV_CACHE_BLOCK_TOKENS
    )]
    kv_cache_block_tokens: usize,
    /// Logical K/V blocks stored in each physical slab allocation.
    #[arg(
        long,
        default_value_t = ModelLoadConfig::DEFAULT_KV_CACHE_SLAB_BLOCKS
    )]
    kv_cache_slab_blocks: usize,
    /// Total full-attention K/V token capacity shared across all sequences.
    /// Defaults to one model context, independent of active request count;
    /// explicit values are rounded up to a whole logical block.
    #[arg(long)]
    kv_cache_capacity_tokens: Option<usize>,
    /// Attention implementation for multi-row prefill chunks. `native` keeps
    /// the paged Metal kernel and its dense fallback; `sdpa` routes multi-row
    /// chunks through Candle's fused SDPA kernel over the materialized K/V
    /// history. Single-token decode always keeps the native kernel.
    #[arg(long, value_enum, default_value_t = PrefillAttentionMode::Sdpa)]
    prefill_attention: PrefillAttentionMode,
}

#[derive(Debug, Clone, Args)]
struct PromptArgs {
    #[arg(short, long)]
    prompt: String,
    #[arg(long)]
    system: Option<String>,
    /// Bypass the family chat template and tokenize the prompt verbatim.
    #[arg(long)]
    raw_prompt: bool,
    /// Ask the tokenizer profile to add configured BOS/EOS tokens.
    /// Leave this off for rendered chat templates that already contain BOS.
    #[arg(long)]
    add_special_tokens: bool,
}

#[derive(Debug, Clone, Args)]
struct SamplingArgs {
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    /// Set to zero for greedy argmax decoding.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,
    #[arg(long, default_value_t = 20)]
    top_k: usize,
    #[arg(long, default_value_t = 0.8)]
    top_p: f64,
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Debug, Clone, Args)]
struct BlockingPrefillArgs {
    /// Maximum prompt tokens per blocking prefill forward. By default, Metal
    /// uses 128 except for explicit BF16; CPU and BF16 use one-shot prefill.
    #[arg(long, value_name = "TOKENS")]
    prefill_token_quantum: Option<NonZeroUsize>,
}

impl SamplingArgs {
    fn sampler(&self, seed_offset: u64) -> Result<Sampler> {
        Sampler::new(SamplingConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed.wrapping_add(seed_offset),
        })
    }
}

#[derive(Debug, Args)]
struct RunArgs {
    #[command(flatten)]
    runtime: RuntimeArgs,
    #[command(flatten)]
    prompt: PromptArgs,
    #[command(flatten)]
    sampling: SamplingArgs,
    #[command(flatten)]
    blocking_prefill: BlockingPrefillArgs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum EmbeddingInputType {
    Query,
    Document,
}

#[derive(Debug, Args)]
struct EmbedArgs {
    #[command(flatten)]
    runtime: RuntimeArgs,
    /// Apply the checkpoint's exact query or document input semantics.
    #[arg(long, value_enum)]
    input_type: EmbeddingInputType,
    /// Text to embed. Repeat this flag to submit an ordered batch.
    #[arg(long, required = true)]
    input: Vec<String>,
    /// Override the default retrieval instruction for query inputs.
    #[arg(long)]
    instruction: Option<String>,
}

#[derive(Debug, Args)]
struct RerankArgs {
    #[command(flatten)]
    runtime: RuntimeArgs,
    #[arg(long)]
    query: String,
    /// Candidate document. Repeat this flag to submit an ordered batch.
    #[arg(long, required = true)]
    document: Vec<String>,
    /// Override the checkpoint's default retrieval instruction.
    #[arg(long)]
    instruction: Option<String>,
}

#[derive(Debug, Serialize)]
struct EmbeddingOutput {
    task: &'static str,
    input_type: EmbeddingInputType,
    dimension: usize,
    data: Vec<EmbeddingOutputRow>,
}

#[derive(Debug, Serialize)]
struct EmbeddingOutputRow {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct RerankOutput {
    task: &'static str,
    data: Vec<RerankOutputRow>,
}

#[derive(Debug, Serialize)]
struct RerankOutputRow {
    index: usize,
    margin: f32,
    yes_probability: f32,
}

struct PreparedEmbeddingTask {
    input_type: EmbeddingInputType,
    dimension: usize,
    token_batches: Vec<Vec<u32>>,
}

struct PreparedRerankTask {
    token_batches: Vec<Vec<u32>>,
    no_token_id: u32,
    yes_token_id: u32,
}

#[derive(Debug, Args)]
struct BenchArgs {
    #[command(subcommand)]
    command: BenchCommand,
}

#[derive(Debug, Subcommand)]
enum BenchCommand {
    /// Measure deterministic greedy text generation.
    Generate(BenchGenerateArgs),
    /// Measure one packed Qwen3 embedding batch.
    Embed(BenchEmbeddingArgs),
    /// Measure one packed Qwen3 reranking batch.
    Rerank(BenchRerankArgs),
}

#[derive(Debug, Clone, Args)]
struct BenchRepeatArgs {
    #[arg(long, default_value_t = 20)]
    runs: usize,
    /// Unreported runs used to compile and warm Metal kernels.
    #[arg(long, default_value_t = 1)]
    warmup_runs: usize,
}

const MAX_BENCHMARK_RUNS: usize = 10_000;

#[derive(Debug, Args)]
struct BenchGenerateArgs {
    #[command(flatten)]
    runtime: RuntimeArgs,
    #[command(flatten)]
    prompt: PromptArgs,
    #[command(flatten)]
    blocking_prefill: BlockingPrefillArgs,
    #[command(flatten)]
    repeat: BenchRepeatArgs,
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
}

#[derive(Debug, Args)]
struct BenchEmbeddingArgs {
    #[command(flatten)]
    task: EmbedArgs,
    #[command(flatten)]
    repeat: BenchRepeatArgs,
}

#[derive(Debug, Args)]
struct BenchRerankArgs {
    #[command(flatten)]
    task: RerankArgs,
    #[command(flatten)]
    repeat: BenchRepeatArgs,
}

#[derive(Debug, Args)]
struct ScoreboardArgs {
    #[command(subcommand)]
    command: ScoreboardCommand,
}

#[derive(Debug, Subcommand)]
enum ScoreboardCommand {
    /// Fail closed unless host, model, prompt tokens and `llama.cpp` all match.
    Verify {
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Measure `MacLLM` single-request prefill, decode and `TTFT` evidence.
    MacllmKernel {
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Measure the truthful pp/tg subset emitted by pinned llama-bench.
    LlamaBench {
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Measure pinned llama-server HTTP serving and greedy-token evidence.
    LlamaServer {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, default_value_t = 18_081)]
        port: u16,
    },
    /// Run every evidence stage and write validated JSON plus Markdown.
    Report {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, default_value_t = 18_081)]
        port: u16,
        #[arg(long, value_name = "PATH")]
        json: PathBuf,
        #[arg(long, value_name = "PATH")]
        markdown: PathBuf,
    },
    /// Measure `MacLLM` scheduler serving at the manifest's 1/4/8 widths.
    MacllmServing {
        #[arg(long)]
        manifest: PathBuf,
    },
}

#[derive(Debug, Args)]
struct DemoArgs {
    #[command(subcommand)]
    command: DemoCommand,
}

#[derive(Debug, Subcommand)]
enum DemoCommand {
    /// Compare one exact-16K, eight-way coding-agent prefix across both runtimes.
    PrefixCache {
        /// Scoreboard manifest pinning host, model bytes and llama.cpp binaries.
        #[arg(long)]
        manifest: PathBuf,
        /// Repository whose Cargo.toml and src/**/*.rs become the shared system prefix.
        #[arg(long, default_value = ".")]
        repository: PathBuf,
        #[arg(long, default_value_t = 18_085)]
        port: u16,
        #[arg(long, default_value = "prefix-cache-demo.json", value_name = "PATH")]
        json: PathBuf,
        #[arg(long, default_value = "prefix-cache-demo.md", value_name = "PATH")]
        markdown: PathBuf,
    },
    /// Fair same-HTTP, same-precision, 20-wave exact-16K serving comparison.
    FairPrefix {
        /// Fair-prefix manifest; K/V cache precision must match `MacLLM` F32.
        #[arg(long)]
        manifest: PathBuf,
        /// Repository whose deterministic bundle becomes the shared prefix.
        #[arg(long, default_value = ".")]
        repository: PathBuf,
        #[arg(long, default_value_t = 18_086)]
        port: u16,
        #[arg(
            long,
            default_value = "fair-prefix-benchmark.json",
            value_name = "PATH"
        )]
        json: PathBuf,
        #[arg(long, default_value = "fair-prefix-benchmark.md", value_name = "PATH")]
        markdown: PathBuf,
    },
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[command(flatten)]
    runtime: RuntimeArgs,
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Model ID exposed to OpenAI-compatible clients. Defaults to the path name.
    #[arg(long)]
    served_model_name: Option<String>,
    #[arg(long, default_value_t = 8)]
    queue_capacity: usize,
    /// Maximum independently stateful requests rotated on the model thread.
    #[arg(long, default_value_t = 2)]
    max_active_sequences: usize,
    /// Total model-input tokens available in one persistent scheduler round.
    #[arg(long, default_value_t = 256)]
    scheduler_token_budget: usize,
    /// Maximum prompt tokens processed for one sequence in a single turn.
    #[arg(long, default_value_t = 128)]
    prefill_token_quantum: usize,
    #[arg(long, default_value_t = 16)]
    event_capacity: usize,
    /// Cancel a request if its client stops draining events for this long.
    #[arg(long, default_value_t = 30)]
    event_send_timeout_seconds: u64,
    /// Resident automatic prompt-prefix cache budget. Set to zero to disable.
    #[arg(
        long,
        default_value_t = ModelLoadConfig::DEFAULT_SERVER_PREFIX_CACHE_CAPACITY_TOKENS
    )]
    prefix_cache_capacity_tokens: usize,
    /// Completion budget for chat requests that omit `max_completion_tokens`
    /// and `max_tokens`. Must fit the model context.
    #[arg(long, default_value_t = NonZeroUsize::new(ServerOptions::DEFAULT_MAX_COMPLETION_TOKENS).unwrap())]
    default_max_completion_tokens: NonZeroUsize,
    /// Render chat and tool prompts with `enable_thinking=false` unless a
    /// request sets `chat_template_kwargs.enable_thinking`. Needs a chat
    /// template that uses `enable_thinking`; startup fails otherwise.
    #[arg(long)]
    disable_thinking: bool,
    /// Sampling defaults for omitted temperature, `top_p` and `presence_penalty`
    /// (`top_k` always comes from the profile). `qwen3` and `qwen35` apply their
    /// model card settings for each request's effective thinking mode.
    #[arg(long, value_enum, default_value_t = SamplingProfileArg::Openai)]
    sampling_profile: SamplingProfileArg,
    /// Report the thinking text of a model that reasons. `deepseek` returns it
    /// separately as `reasoning_content`, like llama.cpp and vLLM; `none`
    /// keeps the thinking block inside the answer.
    #[arg(long, value_enum, default_value_t = ReasoningFormatArg::Deepseek)]
    reasoning_format: ReasoningFormatArg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SamplingProfileArg {
    /// Temperature 1, `top_p` 1, no top-k.
    Openai,
    /// Thinking 0.6 / `top_p` 0.95 / `top_k` 20; non-thinking 0.7 / 0.8 / 20.
    Qwen3,
    /// Qwen3.5 defaults, including non-thinking presence penalty 1.5.
    Qwen35,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ReasoningFormatArg {
    /// Report the thinking block as `reasoning_content`.
    Deepseek,
    /// Keep the thinking block inside the answer.
    None,
}

impl From<ReasoningFormatArg> for ReasoningFormat {
    fn from(arg: ReasoningFormatArg) -> Self {
        match arg {
            ReasoningFormatArg::Deepseek => ReasoningFormat::Deepseek,
            ReasoningFormatArg::None => ReasoningFormat::None,
        }
    }
}

impl From<SamplingProfileArg> for SamplingProfile {
    fn from(profile: SamplingProfileArg) -> Self {
        match profile {
            SamplingProfileArg::Openai => Self::OpenAi,
            SamplingProfileArg::Qwen3 => Self::Qwen3,
            SamplingProfileArg::Qwen35 => Self::Qwen35,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Download {
            repo,
            revision,
            output,
            gguf,
        } => {
            println!("downloading {repo} to {}", output.display());
            let selection = gguf.map_or(DownloadSelection::Safetensors, DownloadSelection::Gguf);
            let path = download_model(&repo, revision.as_deref(), &output, &selection)?;
            println!("model ready at {}", path.display());
        }
        Command::Inspect { model } => inspect(model)?,
        Command::Run(args) => run(args)?,
        Command::Embed(args) => embed(args)?,
        Command::Rerank(args) => rerank(args)?,
        Command::Bench(args) => bench(args)?,
        Command::Scoreboard(args) => scoreboard(args).await?,
        Command::Demo(args) => demo(args).await?,
        Command::Serve(args) => serve_command(args).await?,
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
async fn demo(args: DemoArgs) -> Result<()> {
    match args.command {
        DemoCommand::PrefixCache {
            manifest,
            repository,
            port,
            json,
            markdown,
        } => {
            ensure_release_demo_execution()?;
            let provenance_guard = MacllmProvenanceGuard::capture_publishable()
                .context("failed to pin clean MacLLM executable/source provenance")?;
            let verified = verify_scoreboard_manifest(&manifest)?;
            let prepared = prepare_default_prefix_demo(&verified.manifest.model.path, &repository)
                .context("failed to prepare the exact-16K repository prefix")?;
            let materialized =
                materialize_prefix_demo(&verified.manifest.model.path, &repository, &prepared)
                    .context("failed to rematerialize prepared prompt evidence")?;

            ensure!(
                materialized.tasks.len() == PREFIX_DEMO_CONCURRENCY,
                "materialized prefix demo does not contain eight tasks"
            );
            let llama_seed = llama_prefix_demo_cache_seed(&materialized);
            let llama_tasks: [LlamaPrefixDemoPrompt<'_>; PREFIX_DEMO_CONCURRENCY] =
                std::array::from_fn(|index| LlamaPrefixDemoPrompt {
                    rendered_prompt: &materialized.tasks[index].rendered_prompt,
                    token_ids: &materialized.tasks[index].token_ids,
                });
            let llama_cpp = run_llama_prefix_demo(
                &verified,
                llama_seed,
                &llama_tasks,
                macllm_infer::benchmark::MAC_PREFIX_DEMO_DECODE_TOKENS,
                prepared.requested_shared_tokens,
                macllm_infer::benchmark::LLAMA_PREFIX_DEMO_DEFAULT_MEASURED_WAVES,
                port,
            )
            .context("llama.cpp exact-16K prefix demo failed")?;

            let mut macllm_config = MacPrefixDemoRunConfig::new(
                verified.manifest.model.path.clone(),
                verified.manifest.name.clone(),
                prepared.prefill_quantum,
            );
            macllm_config.kv_cache_block_tokens = verified.manifest.runtime.kv_cache_block_tokens;
            macllm_config.kv_cache_slab_blocks = verified.manifest.runtime.kv_cache_slab_blocks;
            macllm_config.kv_cache_capacity_tokens = verified
                .manifest
                .runtime
                .kv_cache_capacity_tokens
                .max(MAC_PREFIX_DEMO_CACHE_CAPACITY_TOKENS);
            macllm_config.prefix_cache_capacity_tokens = MAC_PREFIX_DEMO_CACHE_CAPACITY_TOKENS;
            let macllm = run_macllm_prefix_demo(&macllm_config, &prepared, &materialized)
                .await
                .context("MacLLM exact-16K prefix demo failed")?;

            let reverified = verify_scoreboard_manifest(&manifest)
                .context("post-run manifest verification failed")?;
            ensure!(
                reverified.manifest == verified.manifest
                    && reverified.host == verified.host
                    && reverified.model == verified.model
                    && reverified.llama_cpp_version == verified.llama_cpp_version
                    && reverified.llama_cpp_commit == verified.llama_cpp_commit
                    && reverified.llama_server == verified.llama_server
                    && reverified.llama_bench == verified.llama_bench,
                "a manifest-pinned input changed during the prefix demo"
            );
            let final_materialized =
                materialize_prefix_demo(&reverified.manifest.model.path, &repository, &prepared)
                    .context("post-run prompt materialization failed")?;
            ensure!(
                final_materialized == materialized,
                "repository, template, tokenizer, or prompt tokens changed during the prefix demo"
            );

            let macllm_implementation = provenance_guard
                .finish()
                .context("MacLLM executable/source provenance changed during the demo")?;
            let report = assemble_prefix_demo_report(
                &reverified,
                macllm_implementation,
                prepared,
                macllm,
                llama_cpp,
            )?;
            write_prefix_demo_artifacts(&report, &json, &markdown)?;
            println!("prefix-demo JSON: {}", json.display());
            println!("prefix-demo Markdown: {}", markdown.display());
        }
        DemoCommand::FairPrefix {
            manifest,
            repository,
            port,
            json,
            markdown,
        } => {
            ensure_release_demo_execution()?;
            let provenance_guard = MacllmProvenanceGuard::capture_publishable()
                .context("failed to pin clean MacLLM executable/source provenance")?;
            let verified = verify_scoreboard_manifest(&manifest)?;
            let prepared = prepare_default_prefix_demo(&verified.manifest.model.path, &repository)
                .context("failed to prepare the exact-16K repository prefix")?;
            let materialized =
                materialize_prefix_demo(&verified.manifest.model.path, &repository, &prepared)
                    .context("failed to materialize fair-prefix prompts")?;
            let report = run_fair_prefix_benchmark(
                &verified,
                provenance_guard,
                prepared,
                &materialized,
                macllm_infer::benchmark::MAC_PREFIX_DEMO_DEFAULT_MEASURED_WAVES,
                port,
            )
            .await?;

            let reverified = verify_scoreboard_manifest(&manifest)
                .context("post-run fair-prefix manifest verification failed")?;
            ensure!(
                reverified.manifest == verified.manifest
                    && reverified.host == verified.host
                    && reverified.model == verified.model
                    && reverified.llama_cpp_version == verified.llama_cpp_version
                    && reverified.llama_cpp_commit == verified.llama_cpp_commit
                    && reverified.llama_server == verified.llama_server,
                "a manifest-pinned input changed during the fair-prefix benchmark"
            );
            let final_prepared =
                prepare_default_prefix_demo(&reverified.manifest.model.path, &repository)
                    .context("failed to re-prepare fair-prefix evidence")?;
            let final_materialized = materialize_prefix_demo(
                &reverified.manifest.model.path,
                &repository,
                &final_prepared,
            )
            .context("failed to rematerialize fair-prefix prompts")?;
            ensure!(
                final_prepared == report.prepared && final_materialized == materialized,
                "repository, template, tokenizer, or prompt tokens changed during the fair-prefix benchmark"
            );
            write_fair_prefix_artifacts(&report, &json, &markdown)?;
            println!("fair-prefix JSON: {}", json.display());
            println!("fair-prefix Markdown: {}", markdown.display());
        }
    }
    Ok(())
}

fn llama_prefix_demo_cache_seed(
    materialized: &MaterializedPrefixDemo,
) -> LlamaPrefixDemoPrompt<'_> {
    LlamaPrefixDemoPrompt {
        rendered_prompt: &materialized.cache_seed.rendered_prompt,
        token_ids: &materialized.cache_seed.token_ids,
    }
}

async fn scoreboard(args: ScoreboardArgs) -> Result<()> {
    match args.command {
        ScoreboardCommand::Verify { manifest } => {
            let verified = verify_scoreboard_manifest(manifest)?;
            println!("{}", serde_json::to_string_pretty(&verified.summary())?);
        }
        ScoreboardCommand::MacllmKernel { manifest } => {
            ensure_release_scoreboard_execution()?;
            let verified = verify_scoreboard_manifest(manifest)?;
            let evidence = run_macllm_kernel(&verified)?;
            println!("{}", serde_json::to_string_pretty(&evidence)?);
        }
        ScoreboardCommand::LlamaBench { manifest } => {
            ensure_release_scoreboard_execution()?;
            let verified = verify_scoreboard_manifest(manifest)?;
            let evidence = run_llama_bench_kernel_evidence(&verified)?;
            println!("{}", serde_json::to_string_pretty(&evidence)?);
        }
        ScoreboardCommand::LlamaServer { manifest, port } => {
            ensure_release_scoreboard_execution()?;
            let verified = verify_scoreboard_manifest(manifest)?;
            let evidence = run_llama_server_serving(&verified, port)?;
            println!("{}", serde_json::to_string_pretty(&evidence)?);
        }
        ScoreboardCommand::Report {
            manifest,
            port,
            json,
            markdown,
        } => {
            ensure_release_scoreboard_execution()?;
            let verified = verify_scoreboard_manifest(manifest)?;
            let report = run_complete_scoreboard(&verified, port).await?;
            write_scoreboard_artifacts(&report, &json, &markdown)?;
            println!("scoreboard JSON: {}", json.display());
            println!("scoreboard Markdown: {}", markdown.display());
        }
        ScoreboardCommand::MacllmServing { manifest } => {
            ensure_release_scoreboard_execution()?;
            let verified = verify_scoreboard_manifest(manifest)?;
            let evidence = run_macllm_serving(&verified).await?;
            println!("{}", serde_json::to_string_pretty(&evidence)?);
        }
    }
    Ok(())
}

fn ensure_release_scoreboard_execution() -> Result<()> {
    ensure_optimized_measurement_execution(
        "scoreboard",
        "cargo run --release -- scoreboard ...",
        cfg!(debug_assertions),
        env!("MACLLM_BUILD_PROFILE"),
        env!("MACLLM_BUILD_OPT_LEVEL"),
    )
}

fn ensure_release_demo_execution() -> Result<()> {
    ensure_optimized_measurement_execution(
        "prefix-demo",
        "cargo run --release -- demo prefix-cache ...",
        cfg!(debug_assertions),
        env!("MACLLM_BUILD_PROFILE"),
        env!("MACLLM_BUILD_OPT_LEVEL"),
    )
}

fn ensure_release_benchmark_execution() -> Result<()> {
    ensure_optimized_measurement_execution(
        "benchmark",
        "cargo run --release -- bench <generate|embed|rerank> ...",
        cfg!(debug_assertions),
        env!("MACLLM_BUILD_PROFILE"),
        env!("MACLLM_BUILD_OPT_LEVEL"),
    )
}

fn ensure_optimized_measurement_execution(
    measurement: &str,
    command: &str,
    debug_assertions: bool,
    profile: &str,
    opt_level: &str,
) -> Result<()> {
    ensure!(
        !debug_assertions
            && !profile.trim().is_empty()
            && profile != "unknown"
            && matches!(opt_level, "1" | "2" | "3" | "s" | "z"),
        "{measurement} measurements require an optimized release build; rerun with `{command}`"
    );
    Ok(())
}

fn inspect(model_dir: PathBuf) -> Result<()> {
    let files = ModelFiles::from_dir(&model_dir)?;
    let adapter = files.adapter()?;
    adapter.validate()?;
    let runtime = adapter.runtime_spec();
    println!("family:               {}", adapter.family());
    if let Some(spec) = adapter.decoder_spec() {
        println!("architecture:         {}", spec.architecture);
        println!("layers:               {}", spec.num_layers);
        println!("hidden size:          {}", spec.hidden_size);
        println!(
            "query / KV heads:     {} / {}",
            spec.num_query_heads, spec.num_kv_heads
        );
        println!("head dimension:       {}", spec.head_dim);
        let f16_kv_bytes = spec.kv_bytes_per_token(2)?;
        println!(
            "F16 KV bytes/token:   {} ({:.2} MiB at 4K)",
            f16_kv_bytes,
            f16_kv_bytes as f64 * 4096.0 / 1024.0 / 1024.0
        );
        let f32_fixed_state = spec.fixed_sequence_state_bytes(4)?;
        if f32_fixed_state > 0 {
            println!(
                "F32 fixed state/seq: {} ({:.2} MiB)",
                f32_fixed_state,
                f32_fixed_state as f64 / 1024.0 / 1024.0
            );
        }
    }
    println!("maximum context:      {}", runtime.max_context);
    println!("vocabulary:           {}", runtime.vocab_size);
    println!("stop token IDs:       {:?}", runtime.stop_token_ids);
    println!("model format:         {}", files.format());
    println!("weight files:         {}", files.weight_file_count());
    if let Some(tensors) = files.tensor_count() {
        println!("indexed tensors:      {tensors}");
        println!("quantized tensors:    {}", files.quantized_tensor_count());
        let tensor_types = files
            .gguf_tensor_type_counts()
            .into_iter()
            .map(|(dtype, count)| format!("{}={count}", dtype.label()))
            .collect::<Vec<_>>()
            .join(", ");
        println!("tensor types:         {tensor_types}");
    }
    Ok(())
}

fn run(args: RunArgs) -> Result<()> {
    let (runtime, prompt_tokens) = load_prompt_runtime(&args.runtime, &args.prompt)?;
    let prefill_token_quantum = resolve_blocking_prefill_token_quantum(
        &args.runtime,
        args.blocking_prefill.prefill_token_quantum,
    );
    eprintln!(
        "loaded {} on {:?}; prompt={} tokens",
        runtime.adapter.family(),
        args.runtime.device,
        prompt_tokens.len()
    );
    let mut stream = TokenStream::new(&runtime.tokenizer);
    let mut sampler = args.sampling.sampler(0)?;
    let mut session = runtime.model.new_session()?;
    let metrics = generate_with_prefill_token_quantum(
        runtime.model.as_ref(),
        session.as_mut(),
        &prompt_tokens,
        &mut sampler,
        GenerationConfig {
            max_new_tokens: args.sampling.max_new_tokens,
            ignore_stop_tokens: false,
        },
        prefill_token_quantum,
        |token| {
            if let Some(fragment) = stream.push(token)? {
                print!("{fragment}");
                io::stdout().flush()?;
            }
            Ok(())
        },
    )?;
    if let Some(fragment) = stream.finish()? {
        print!("{fragment}");
    }
    println!();
    print_metrics(&metrics);
    Ok(())
}

fn embed(args: EmbedArgs) -> Result<()> {
    let prepared = prepare_embedding_task(&args)?;
    let runtime = LoadedRuntime::load(&model_load_config(&args.runtime))?;
    let embeddings = embed_token_batch(runtime.model.as_ref(), &prepared.token_batches)?;
    let output = build_embedding_output(prepared.input_type, prepared.dimension, embeddings)?;
    write_json_stdout(&output)
}

fn prepare_embedding_task(args: &EmbedArgs) -> Result<PreparedEmbeddingTask> {
    ensure!(
        args.input_type == EmbeddingInputType::Query || args.instruction.is_none(),
        "--instruction is only valid with --input-type query"
    );
    for (index, input) in args.input.iter().enumerate() {
        ensure!(
            !input.trim().is_empty(),
            "embedding input {index} must not be empty"
        );
    }
    if let Some(instruction) = args.instruction.as_deref() {
        ensure!(
            !instruction.trim().is_empty(),
            "embedding instruction must not be empty"
        );
    }
    let profile = Qwen3EmbeddingProfile::from_model_dir(&args.runtime.model)
        .context("model is not a supported Qwen3 embedding checkpoint")?;
    let tokenizer = validate_task_checkpoint(&args.runtime.model, "embedding", |tokenizer| {
        profile.validate_tokenizer(tokenizer)
    })?;
    let token_batches = args
        .input
        .iter()
        .enumerate()
        .map(|(index, input)| {
            match args.input_type {
                EmbeddingInputType::Query => {
                    profile.tokenize_query(&tokenizer, input, args.instruction.as_deref())
                }
                EmbeddingInputType::Document => profile.tokenize_document(&tokenizer, input),
            }
            .with_context(|| format!("embedding input {index} is invalid"))
        })
        .collect::<Result<Vec<_>>>()?;
    drop(tokenizer);
    Ok(PreparedEmbeddingTask {
        input_type: args.input_type,
        dimension: profile.dimension(),
        token_batches,
    })
}

fn rerank(args: RerankArgs) -> Result<()> {
    let prepared = prepare_rerank_task(&args)?;
    let runtime = LoadedRuntime::load(&model_load_config(&args.runtime))?;
    let scores = rerank_token_batch(
        runtime.model.as_ref(),
        &prepared.token_batches,
        prepared.no_token_id,
        prepared.yes_token_id,
    )?;
    let output = build_rerank_output(scores, args.document.len())?;
    write_json_stdout(&output)
}

fn prepare_rerank_task(args: &RerankArgs) -> Result<PreparedRerankTask> {
    ensure!(
        !args.query.trim().is_empty(),
        "reranker query must not be empty"
    );
    for (index, document) in args.document.iter().enumerate() {
        ensure!(
            !document.trim().is_empty(),
            "reranker document {index} must not be empty"
        );
    }
    if let Some(instruction) = args.instruction.as_deref() {
        ensure!(
            !instruction.trim().is_empty(),
            "reranker instruction must not be empty"
        );
    }
    let profile = Qwen3RerankerProfile::from_model_dir(&args.runtime.model)
        .context("model is not a supported Qwen3 reranker checkpoint")?;
    let tokenizer = validate_task_checkpoint(&args.runtime.model, "reranking", |tokenizer| {
        profile.validate_tokenizer(tokenizer)
    })?;
    let token_batches = args
        .document
        .iter()
        .enumerate()
        .map(|(index, document)| {
            profile
                .tokenize_pair(
                    &tokenizer,
                    &args.query,
                    document,
                    args.instruction.as_deref(),
                )
                .with_context(|| format!("reranker document {index} is invalid"))
        })
        .collect::<Result<Vec<_>>>()?;
    drop(tokenizer);
    Ok(PreparedRerankTask {
        token_batches,
        no_token_id: profile.no_token_id(),
        yes_token_id: profile.yes_token_id(),
    })
}

fn validate_task_checkpoint(
    model: &std::path::Path,
    task: &str,
    validate_tokenizer: impl FnOnce(&tokenizers::Tokenizer) -> Result<()>,
) -> Result<std::sync::Arc<tokenizers::Tokenizer>> {
    let files = ModelFiles::from_dir(model)?;
    ensure!(
        files.format() == ModelFormat::Safetensors,
        "Qwen3 {task} requires a Safetensors checkpoint"
    );
    let tokenizer = files.tokenizer()?;
    validate_tokenizer(&tokenizer)?;
    Ok(tokenizer)
}

fn build_embedding_output(
    input_type: EmbeddingInputType,
    dimension: usize,
    embeddings: Vec<Vec<f32>>,
) -> Result<EmbeddingOutput> {
    ensure!(dimension > 0, "embedding dimension must be positive");
    for (index, embedding) in embeddings.iter().enumerate() {
        ensure!(
            embedding.len() == dimension,
            "embedding row {index} has dimension {}, expected {dimension}",
            embedding.len()
        );
    }
    Ok(EmbeddingOutput {
        task: "embed",
        input_type,
        dimension,
        data: embeddings
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| EmbeddingOutputRow { index, embedding })
            .collect(),
    })
}

fn build_rerank_output(scores: Vec<RerankScore>, expected: usize) -> Result<RerankOutput> {
    ensure!(
        scores.len() == expected,
        "reranker returned {} rows for {expected} documents",
        scores.len()
    );
    Ok(RerankOutput {
        task: "rerank",
        data: scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| RerankOutputRow {
                index,
                margin: score.margin,
                yes_probability: score.yes_probability,
            })
            .collect(),
    })
}

fn write_json_stdout(value: &impl Serialize) -> Result<()> {
    let mut output = serde_json::to_string_pretty(value)?;
    output.push('\n');
    io::stdout().lock().write_all(output.as_bytes())?;
    Ok(())
}

fn bench(args: BenchArgs) -> Result<()> {
    match &args.command {
        BenchCommand::Generate(args) => {
            validate_bench_repeat(&args.repeat)?;
            ensure!(
                args.max_new_tokens > 0,
                "generation benchmark requires at least one output token"
            );
        }
        BenchCommand::Embed(args) => validate_bench_repeat(&args.repeat)?,
        BenchCommand::Rerank(args) => validate_bench_repeat(&args.repeat)?,
    }
    ensure_release_benchmark_execution()?;
    match args.command {
        BenchCommand::Generate(args) => bench_generate(args),
        BenchCommand::Embed(args) => bench_embedding(args),
        BenchCommand::Rerank(args) => bench_rerank(args),
    }
}

fn validate_bench_repeat(args: &BenchRepeatArgs) -> Result<()> {
    ensure!(args.runs > 0, "benchmark runs must be positive");
    ensure!(
        args.runs <= MAX_BENCHMARK_RUNS,
        "benchmark run count is too large; maximum is {MAX_BENCHMARK_RUNS}"
    );
    ensure!(
        args.warmup_runs <= MAX_BENCHMARK_RUNS,
        "benchmark warmup count is too large; maximum is {MAX_BENCHMARK_RUNS}"
    );
    Ok(())
}

fn bench_generate(args: BenchGenerateArgs) -> Result<()> {
    let mut load_config = model_load_config(&args.runtime);
    let report_guard = TaskBenchmarkGuard::capture(&load_config)?;
    report_guard.model_path().clone_into(&mut load_config.model);
    let (runtime, prompt_tokens) = load_prompt_runtime_from_config(&load_config, &args.prompt)?;
    report_guard.verify_checkpoint()?;
    let prefill_token_quantum = resolve_blocking_prefill_token_quantum(
        &args.runtime,
        args.blocking_prefill.prefill_token_quantum,
    );
    eprintln!(
        "generate benchmark excludes model loading, tokenization, and session allocation; includes prefix reset, prefill, decode, greedy sampling, and token collection"
    );
    eprintln!(
        "workload: prompt_tokens={}, output_tokens={}, warmups={}, runs={}, greedy=true",
        prompt_tokens.len(),
        args.max_new_tokens,
        args.repeat.warmup_runs,
        args.repeat.runs
    );
    let mut session = runtime.model.new_session()?;
    let samples = collect_deterministic_samples("generate", &args.repeat, || {
        generate_benchmark_sample(
            &runtime,
            session.as_mut(),
            &prompt_tokens,
            prefill_token_quantum,
            args.max_new_tokens,
        )
    })?;
    let runtime_config =
        TaskRuntimeConfig::capture(&runtime, &load_config, Some(prefill_token_quantum.get()))?;
    let common = report_guard.finish(
        runtime_config,
        std::slice::from_ref(&prompt_tokens),
        TaskRepeat::new(args.repeat.warmup_runs, args.repeat.runs),
    )?;
    let report = TaskBenchmarkReport::generate(common, args.max_new_tokens, samples)?;
    write_json_stdout(&report)
}

fn generate_benchmark_sample(
    runtime: &LoadedRuntime,
    session: &mut dyn macllm_infer::architecture::CausalLmSession,
    prompt_tokens: &[u32],
    prefill_token_quantum: NonZeroUsize,
    max_new_tokens: usize,
) -> Result<(GenerationMetrics, Vec<u32>)> {
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 42,
    })?;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(max_new_tokens)
        .context("generation benchmark output length is too large")?;
    let metrics = generate_with_prefill_token_quantum(
        runtime.model.as_ref(),
        session,
        prompt_tokens,
        &mut sampler,
        GenerationConfig {
            max_new_tokens,
            ignore_stop_tokens: true,
        },
        prefill_token_quantum,
        |token| {
            tokens.push(token);
            Ok(())
        },
    )?;
    ensure!(
        metrics.generated_tokens == max_new_tokens && tokens.len() == max_new_tokens,
        "generation benchmark did not produce its fixed output length"
    );
    Ok((metrics, tokens))
}

fn bench_embedding(mut args: BenchEmbeddingArgs) -> Result<()> {
    let mut load_config = model_load_config(&args.task.runtime);
    let report_guard = TaskBenchmarkGuard::capture(&load_config)?;
    report_guard.model_path().clone_into(&mut load_config.model);
    args.task.runtime.model.clone_from(&load_config.model);
    let prepared = prepare_embedding_task(&args.task)?;
    let runtime = LoadedRuntime::load(&load_config)?;
    report_guard.verify_checkpoint()?;
    eprintln!(
        "embed benchmark excludes model loading and tokenization; includes session allocation, packed forward, CPU readback, and L2 normalization"
    );
    eprintln!(
        "workload: token_lengths={:?}, warmups={}, runs={}",
        token_lengths(&prepared.token_batches),
        args.repeat.warmup_runs,
        args.repeat.runs
    );
    let samples = collect_deterministic_samples("embed", &args.repeat, || {
        let started_at = Instant::now();
        let output = embed_token_batch(runtime.model.as_ref(), &prepared.token_batches)?;
        Ok((started_at.elapsed(), output))
    })?;
    let runtime_config = TaskRuntimeConfig::capture(&runtime, &load_config, None)?;
    let common = report_guard.finish(
        runtime_config,
        &prepared.token_batches,
        TaskRepeat::new(args.repeat.warmup_runs, args.repeat.runs),
    )?;
    let input_type = match prepared.input_type {
        EmbeddingInputType::Query => TaskEmbeddingInputType::Query,
        EmbeddingInputType::Document => TaskEmbeddingInputType::Document,
    };
    let report = TaskBenchmarkReport::embed(common, input_type, prepared.dimension, samples)?;
    write_json_stdout(&report)
}

fn bench_rerank(mut args: BenchRerankArgs) -> Result<()> {
    let mut load_config = model_load_config(&args.task.runtime);
    let report_guard = TaskBenchmarkGuard::capture(&load_config)?;
    report_guard.model_path().clone_into(&mut load_config.model);
    args.task.runtime.model.clone_from(&load_config.model);
    let prepared = prepare_rerank_task(&args.task)?;
    let runtime = LoadedRuntime::load(&load_config)?;
    report_guard.verify_checkpoint()?;
    eprintln!(
        "rerank benchmark excludes model loading and tokenization; includes session allocation, packed forward, selected-logit CPU readback, and sigmoid scoring"
    );
    eprintln!(
        "workload: token_lengths={:?}, warmups={}, runs={}",
        token_lengths(&prepared.token_batches),
        args.repeat.warmup_runs,
        args.repeat.runs
    );
    let samples = collect_deterministic_samples("rerank", &args.repeat, || {
        let started_at = Instant::now();
        let output = rerank_token_batch(
            runtime.model.as_ref(),
            &prepared.token_batches,
            prepared.no_token_id,
            prepared.yes_token_id,
        )?;
        Ok((started_at.elapsed(), output))
    })?;
    let runtime_config = TaskRuntimeConfig::capture(&runtime, &load_config, None)?;
    let common = report_guard.finish(
        runtime_config,
        &prepared.token_batches,
        TaskRepeat::new(args.repeat.warmup_runs, args.repeat.runs),
    )?;
    let report =
        TaskBenchmarkReport::rerank(common, prepared.no_token_id, prepared.yes_token_id, samples)?;
    write_json_stdout(&report)
}

fn collect_deterministic_samples<O: Clone + PartialEq, M>(
    task: &str,
    repeat: &BenchRepeatArgs,
    mut execute: impl FnMut() -> Result<(M, O)>,
) -> Result<Vec<(M, O)>> {
    validate_bench_repeat(repeat)?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(repeat.runs)
        .context("benchmark run count is too large")?;
    let mut expected = None;
    for warmup in 0..repeat.warmup_runs {
        let (_, output) =
            execute().with_context(|| format!("{task} warmup {} failed", warmup + 1))?;
        retain_deterministic_output(&mut expected, &output, task, "warmup", warmup)?;
    }
    for run in 0..repeat.runs {
        let (measurement, output) =
            execute().with_context(|| format!("{task} run {} failed", run + 1))?;
        retain_deterministic_output(&mut expected, &output, task, "measured run", run)?;
        samples.push((measurement, output));
    }
    Ok(samples)
}

fn retain_deterministic_output<T: Clone + PartialEq>(
    expected: &mut Option<T>,
    output: &T,
    task: &str,
    phase: &str,
    index: usize,
) -> Result<()> {
    if let Some(expected) = expected {
        ensure!(
            expected == output,
            "{task} benchmark output changed during {phase} {}",
            index + 1
        );
    } else {
        *expected = Some(output.clone());
    }
    Ok(())
}

fn token_lengths(token_batches: &[Vec<u32>]) -> Vec<usize> {
    token_batches.iter().map(Vec::len).collect()
}

fn load_prompt_runtime(
    runtime_args: &RuntimeArgs,
    prompt: &PromptArgs,
) -> Result<(LoadedRuntime, Vec<u32>)> {
    load_prompt_runtime_from_config(&model_load_config(runtime_args), prompt)
}

fn load_prompt_runtime_from_config(
    config: &ModelLoadConfig,
    prompt: &PromptArgs,
) -> Result<(LoadedRuntime, Vec<u32>)> {
    let runtime = LoadedRuntime::load(config)?;
    let rendered = if prompt.raw_prompt {
        prompt.prompt.clone()
    } else {
        runtime
            .adapter
            .format_chat(prompt.system.as_deref(), &prompt.prompt)?
    };
    let prompt_tokens = runtime.encode(rendered, prompt.add_special_tokens)?;
    Ok((runtime, prompt_tokens))
}

fn model_load_config(args: &RuntimeArgs) -> ModelLoadConfig {
    let mut config = ModelLoadConfig::new(args.model.clone(), args.device, args.dtype);
    config.kv_cache_block_tokens = args.kv_cache_block_tokens;
    config.kv_cache_slab_blocks = args.kv_cache_slab_blocks;
    config.kv_cache_capacity_tokens = args.kv_cache_capacity_tokens;
    config.prefill_attention = args.prefill_attention;
    config
}

fn resolve_blocking_prefill_token_quantum(
    runtime: &RuntimeArgs,
    requested: Option<NonZeroUsize>,
) -> NonZeroUsize {
    requested.unwrap_or_else(|| {
        if runtime.device == DeviceKind::Metal && runtime.dtype != Some(DTypeKind::Bf16) {
            NonZeroUsize::new(128).expect("the automatic Metal prefill quantum is positive")
        } else {
            NonZeroUsize::MAX
        }
    })
}

async fn serve_command(args: ServeArgs) -> Result<()> {
    let served_model = args
        .served_model_name
        .unwrap_or_else(|| default_served_model_name(&args.runtime.model));
    let mut model = model_load_config(&args.runtime);
    model.prefix_cache_capacity_tokens = args.prefix_cache_capacity_tokens;
    let mut engine = macllm_infer::engine::EngineConfig::new(model, served_model);
    engine.queue_capacity = args.queue_capacity;
    engine.max_active_sequences = args.max_active_sequences;
    engine.scheduler_token_budget = args.scheduler_token_budget;
    engine.prefill_token_quantum = args.prefill_token_quantum;
    engine.event_capacity = args.event_capacity;
    engine.event_send_timeout = Duration::from_secs(args.event_send_timeout_seconds);
    if args.disable_thinking {
        engine.chat_template_defaults.enable_thinking = Some(false);
    }
    serve(ServerConfig {
        bind: SocketAddr::new(args.host, args.port),
        engine,
        options: ServerOptions {
            default_max_completion_tokens: args.default_max_completion_tokens.get(),
            sampling_profile: args.sampling_profile.into(),
            reasoning_format: args.reasoning_format.into(),
        },
    })
    .await
}

fn default_served_model_name(path: &std::path::Path) -> String {
    path.file_name()
        .and_then(std::ffi::OsStr::to_str)
        .filter(|name| !name.is_empty())
        .unwrap_or("macllm-model")
        .to_owned()
}

fn print_metrics(metrics: &GenerationMetrics) {
    eprintln!(
        "prompt={} cached={} processed={} output={} TTFT={:.1}ms prefill={:.1} tok/s effective-prefill={:.1} tok/s model-decode={:.1} tok/s e2e-output={:.1} tok/s total={:.1}ms",
        metrics.prompt_tokens,
        metrics.cached_prompt_tokens,
        metrics.processed_prompt_tokens(),
        metrics.generated_tokens,
        metrics.time_to_first_token.as_secs_f64() * 1000.0,
        metrics.prefill_tokens_per_second(),
        metrics.effective_prefill_tokens_per_second(),
        metrics.decode_tokens_per_second(),
        metrics.output_tokens_per_second(),
        metrics.total_time.as_secs_f64() * 1000.0,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use macllm_infer::benchmark::MaterializedPrefixPrompt;

    fn run_args(cli: Cli) -> RunArgs {
        match cli.command {
            Command::Run(args) => args,
            _ => panic!("expected run command"),
        }
    }

    fn bench_generate_args(cli: Cli) -> BenchGenerateArgs {
        match cli.command {
            Command::Bench(BenchArgs {
                command: BenchCommand::Generate(args),
            }) => args,
            _ => panic!("expected generate benchmark command"),
        }
    }

    fn bench_embedding_args(cli: Cli) -> BenchEmbeddingArgs {
        match cli.command {
            Command::Bench(BenchArgs {
                command: BenchCommand::Embed(args),
            }) => args,
            _ => panic!("expected embedding benchmark command"),
        }
    }

    fn bench_rerank_args(cli: Cli) -> BenchRerankArgs {
        match cli.command {
            Command::Bench(BenchArgs {
                command: BenchCommand::Rerank(args),
            }) => args,
            _ => panic!("expected rerank benchmark command"),
        }
    }

    fn embed_args(cli: Cli) -> EmbedArgs {
        match cli.command {
            Command::Embed(args) => args,
            _ => panic!("expected embed command"),
        }
    }

    fn rerank_args(cli: Cli) -> RerankArgs {
        match cli.command {
            Command::Rerank(args) => args,
            _ => panic!("expected rerank command"),
        }
    }

    fn scoreboard_args(cli: Cli) -> ScoreboardArgs {
        match cli.command {
            Command::Scoreboard(args) => args,
            _ => panic!("expected scoreboard command"),
        }
    }

    fn demo_args(cli: Cli) -> DemoArgs {
        match cli.command {
            Command::Demo(args) => args,
            _ => panic!("expected demo command"),
        }
    }

    fn serve_runtime(cli: Cli) -> RuntimeArgs {
        match cli.command {
            Command::Serve(args) => args.runtime,
            _ => panic!("expected serve command"),
        }
    }

    fn serve_args(cli: Cli) -> ServeArgs {
        match cli.command {
            Command::Serve(args) => args,
            _ => panic!("expected serve command"),
        }
    }

    #[test]
    fn cli_uses_shared_kv_pool_defaults() {
        let runtime = serve_runtime(
            Cli::try_parse_from(["macllm", "serve", "--model", "model-dir"]).unwrap(),
        );
        let config = model_load_config(&runtime);

        assert_eq!(
            config.kv_cache_block_tokens,
            ModelLoadConfig::DEFAULT_KV_CACHE_BLOCK_TOKENS
        );
        assert_eq!(
            config.kv_cache_slab_blocks,
            ModelLoadConfig::DEFAULT_KV_CACHE_SLAB_BLOCKS
        );
        assert_eq!(config.kv_cache_capacity_tokens, None);
    }

    #[test]
    fn serve_enables_automatic_prefix_cache_with_an_explicit_disable_switch() {
        let defaults =
            serve_args(Cli::try_parse_from(["macllm", "serve", "--model", "model-dir"]).unwrap());
        assert_eq!(
            defaults.prefix_cache_capacity_tokens,
            ModelLoadConfig::DEFAULT_SERVER_PREFIX_CACHE_CAPACITY_TOKENS
        );

        let disabled = serve_args(
            Cli::try_parse_from([
                "macllm",
                "serve",
                "--model",
                "model-dir",
                "--prefix-cache-capacity-tokens",
                "0",
            ])
            .unwrap(),
        );
        assert_eq!(disabled.prefix_cache_capacity_tokens, 0);
    }

    #[test]
    fn serve_keeps_protocol_defaults_unless_flags_change_them() {
        let defaults =
            serve_args(Cli::try_parse_from(["macllm", "serve", "--model", "model-dir"]).unwrap());
        assert_eq!(defaults.default_max_completion_tokens.get(), 128);
        assert!(!defaults.disable_thinking);
        assert_eq!(defaults.sampling_profile, SamplingProfileArg::Openai);
        assert_eq!(defaults.reasoning_format, ReasoningFormatArg::Deepseek);

        // The thinking block is reported separately unless the server is
        // told to keep it inside the answer.
        let legacy = serve_args(
            Cli::try_parse_from([
                "macllm",
                "serve",
                "--model",
                "model-dir",
                "--reasoning-format",
                "none",
            ])
            .unwrap(),
        );
        assert_eq!(legacy.reasoning_format, ReasoningFormatArg::None);
        assert_eq!(
            ReasoningFormat::from(legacy.reasoning_format),
            ReasoningFormat::None
        );

        let agent = serve_args(
            Cli::try_parse_from([
                "macllm",
                "serve",
                "--model",
                "model-dir",
                "--default-max-completion-tokens",
                "4096",
                "--disable-thinking",
                "--sampling-profile",
                "qwen3",
            ])
            .unwrap(),
        );
        assert_eq!(agent.default_max_completion_tokens.get(), 4_096);
        assert!(agent.disable_thinking);
        assert_eq!(agent.sampling_profile, SamplingProfileArg::Qwen3);
        let qwen35 = serve_args(
            Cli::try_parse_from([
                "macllm",
                "serve",
                "--model",
                "model-dir",
                "--sampling-profile",
                "qwen35",
            ])
            .unwrap(),
        );
        assert_eq!(qwen35.sampling_profile, SamplingProfileArg::Qwen35);
        assert!(
            Cli::try_parse_from([
                "macllm",
                "serve",
                "--model",
                "model-dir",
                "--default-max-completion-tokens",
                "0",
            ])
            .is_err()
        );
    }

    #[test]
    fn cli_propagates_explicit_shared_kv_pool_options() {
        let runtime = serve_runtime(
            Cli::try_parse_from([
                "macllm",
                "serve",
                "--model",
                "model-dir",
                "--kv-cache-block-tokens",
                "32",
                "--kv-cache-slab-blocks",
                "5",
                "--kv-cache-capacity-tokens",
                "8193",
            ])
            .unwrap(),
        );
        let config = model_load_config(&runtime);

        assert_eq!(config.kv_cache_block_tokens, 32);
        assert_eq!(config.kv_cache_slab_blocks, 5);
        assert_eq!(config.kv_cache_capacity_tokens, Some(8193));
    }

    #[test]
    fn blocking_commands_default_prefill_quantum_to_auto() {
        let run = run_args(
            Cli::try_parse_from(["macllm", "run", "--model", "model-dir", "--prompt", "hello"])
                .unwrap(),
        );
        let bench = bench_generate_args(
            Cli::try_parse_from([
                "macllm",
                "bench",
                "generate",
                "--model",
                "model-dir",
                "--prompt",
                "hello",
            ])
            .unwrap(),
        );

        assert_eq!(run.blocking_prefill.prefill_token_quantum, None);
        assert_eq!(bench.blocking_prefill.prefill_token_quantum, None);
    }

    #[test]
    fn task_commands_require_explicit_inputs_and_preserve_batch_order() {
        assert!(
            Cli::try_parse_from([
                "macllm",
                "embed",
                "--model",
                "model-dir",
                "--input-type",
                "query",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "macllm",
                "rerank",
                "--model",
                "model-dir",
                "--query",
                "question",
            ])
            .is_err()
        );

        let embed = embed_args(
            Cli::try_parse_from([
                "macllm",
                "embed",
                "--model",
                "model-dir",
                "--input-type",
                "query",
                "--input",
                "first",
                "--input",
                "second",
                "--instruction",
                "retrieve code",
            ])
            .unwrap(),
        );
        assert_eq!(embed.input_type, EmbeddingInputType::Query);
        assert_eq!(embed.input, ["first", "second"]);
        assert_eq!(embed.instruction.as_deref(), Some("retrieve code"));

        let rerank = rerank_args(
            Cli::try_parse_from([
                "macllm",
                "rerank",
                "--model",
                "model-dir",
                "--query",
                "question",
                "--document",
                "first",
                "--document",
                "second",
            ])
            .unwrap(),
        );
        assert_eq!(rerank.query, "question");
        assert_eq!(rerank.document, ["first", "second"]);
        let expected_device = if cfg!(feature = "metal") {
            DeviceKind::Metal
        } else if cfg!(feature = "cuda") {
            DeviceKind::Cuda
        } else {
            DeviceKind::Cpu
        };
        assert_eq!(rerank.runtime.device, expected_device);
    }

    #[test]
    fn task_benchmarks_are_explicit_batched_and_greedy() {
        assert!(
            Cli::try_parse_from([
                "macllm",
                "bench",
                "--model",
                "model-dir",
                "--prompt",
                "legacy",
            ])
            .is_err()
        );
        let generate = bench_generate_args(
            Cli::try_parse_from([
                "macllm",
                "bench",
                "generate",
                "--model",
                "model-dir",
                "--prompt",
                "hello",
            ])
            .unwrap(),
        );
        assert_eq!(generate.max_new_tokens, 128);
        assert_eq!(generate.repeat.warmup_runs, 1);
        assert_eq!(generate.repeat.runs, 20);
        assert!(
            Cli::try_parse_from([
                "macllm",
                "bench",
                "generate",
                "--model",
                "model-dir",
                "--prompt",
                "hello",
                "--temperature",
                "0.7",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "macllm",
                "bench",
                "generate",
                "--model",
                "model-dir",
                "--prompt",
                "hello",
                "--seed",
                "7",
            ])
            .is_err()
        );

        let embed = bench_embedding_args(
            Cli::try_parse_from([
                "macllm",
                "bench",
                "embed",
                "--model",
                "model-dir",
                "--input-type",
                "query",
                "--input",
                "first",
                "--input",
                "second",
                "--runs",
                "7",
            ])
            .unwrap(),
        );
        assert_eq!(embed.task.input, ["first", "second"]);
        assert_eq!(embed.repeat.runs, 7);

        let rerank = bench_rerank_args(
            Cli::try_parse_from([
                "macllm",
                "bench",
                "rerank",
                "--model",
                "model-dir",
                "--query",
                "question",
                "--document",
                "first",
                "--document",
                "second",
                "--warmup-runs",
                "3",
            ])
            .unwrap(),
        );
        assert_eq!(rerank.task.document, ["first", "second"]);
        assert_eq!(rerank.repeat.warmup_runs, 3);
    }

    #[test]
    fn deterministic_task_measurement_excludes_warmups_and_rejects_drift() -> Result<()> {
        let mut calls = 0;
        let repeat = BenchRepeatArgs {
            runs: 3,
            warmup_runs: 2,
        };
        let samples = collect_deterministic_samples("embed", &repeat, || {
            calls += 1;
            Ok(((), vec![1_u32, 2]))
        })?;
        assert_eq!(calls, 5);
        assert_eq!(samples.len(), 3);
        assert!(samples.iter().all(|((), output)| output == &[1, 2]));

        let mut calls = 0;
        let repeat = BenchRepeatArgs {
            runs: 2,
            warmup_runs: 0,
        };
        let error = collect_deterministic_samples("rerank", &repeat, || {
            calls += 1;
            Ok(((), calls))
        })
        .unwrap_err();
        assert!(error.to_string().contains("output changed"));
        assert!(
            collect_deterministic_samples(
                "embed",
                &BenchRepeatArgs {
                    runs: 0,
                    warmup_runs: 1,
                },
                || Ok(((), 1)),
            )
            .is_err()
        );
        assert!(
            collect_deterministic_samples(
                "embed",
                &BenchRepeatArgs {
                    runs: usize::MAX,
                    warmup_runs: 0,
                },
                || Ok(((), 1)),
            )
            .unwrap_err()
            .to_string()
            .contains("run count is too large")
        );
        Ok(())
    }

    #[test]
    fn benchmark_counts_are_rejected_before_runtime_loading() {
        assert!(
            validate_bench_repeat(&BenchRepeatArgs {
                runs: 0,
                warmup_runs: 1,
            })
            .is_err()
        );
        assert!(
            validate_bench_repeat(&BenchRepeatArgs {
                runs: 1,
                warmup_runs: usize::MAX,
            })
            .is_err()
        );
        let args = bench_generate_args(
            Cli::try_parse_from([
                "macllm",
                "bench",
                "generate",
                "--model",
                "missing-model",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "0",
            ])
            .unwrap(),
        );
        let error = bench(BenchArgs {
            command: BenchCommand::Generate(args),
        })
        .unwrap_err();
        assert!(error.to_string().contains("at least one output token"));
    }

    #[test]
    fn task_text_errors_are_rejected_before_model_loading() {
        let args = embed_args(
            Cli::try_parse_from([
                "macllm",
                "embed",
                "--model",
                "missing-model",
                "--input-type",
                "document",
                "--input",
                "document",
                "--instruction",
                "query-only",
            ])
            .unwrap(),
        );
        assert!(
            embed(args)
                .unwrap_err()
                .to_string()
                .contains("only valid with --input-type query")
        );

        let blank_embedding = embed_args(
            Cli::try_parse_from([
                "macllm",
                "embed",
                "--model",
                "missing-model",
                "--input-type",
                "query",
                "--input",
                " ",
            ])
            .unwrap(),
        );
        assert_eq!(
            embed(blank_embedding).unwrap_err().to_string(),
            "embedding input 0 must not be empty"
        );

        let blank_document = rerank_args(
            Cli::try_parse_from([
                "macllm",
                "rerank",
                "--model",
                "missing-model",
                "--query",
                "question",
                "--document",
                " ",
            ])
            .unwrap(),
        );
        assert_eq!(
            rerank(blank_document).unwrap_err().to_string(),
            "reranker document 0 must not be empty"
        );
    }

    #[test]
    fn task_json_outputs_are_stable_indexed_and_do_not_echo_inputs() -> Result<()> {
        let embedding = build_embedding_output(
            EmbeddingInputType::Document,
            2,
            vec![vec![0.5, -0.5], vec![1.0, 0.0]],
        )?;
        assert_eq!(
            serde_json::to_value(embedding)?,
            serde_json::json!({
                "task": "embed",
                "input_type": "document",
                "dimension": 2,
                "data": [
                    {"index": 0, "embedding": [0.5, -0.5]},
                    {"index": 1, "embedding": [1.0, 0.0]}
                ]
            })
        );
        assert!(
            build_embedding_output(EmbeddingInputType::Query, 3, vec![vec![1.0, 0.0]]).is_err()
        );

        let reranked = build_rerank_output(
            vec![
                RerankScore {
                    margin: 2.0,
                    yes_probability: 0.875,
                },
                RerankScore {
                    margin: -1.0,
                    yes_probability: 0.25,
                },
            ],
            2,
        )?;
        assert_eq!(
            serde_json::to_value(reranked)?,
            serde_json::json!({
                "task": "rerank",
                "data": [
                    {"index": 0, "margin": 2.0, "yes_probability": 0.875},
                    {"index": 1, "margin": -1.0, "yes_probability": 0.25}
                ]
            })
        );
        assert!(build_rerank_output(Vec::new(), 1).is_err());
        Ok(())
    }

    #[test]
    fn scoreboard_verify_requires_an_explicit_manifest() {
        let args = scoreboard_args(
            Cli::try_parse_from([
                "macllm",
                "scoreboard",
                "verify",
                "--manifest",
                "benchmarks/profile.json",
            ])
            .unwrap(),
        );
        assert!(matches!(
            args.command,
            ScoreboardCommand::Verify { manifest }
                if manifest == std::path::Path::new("benchmarks/profile.json")
        ));
        assert!(Cli::try_parse_from(["macllm", "scoreboard", "verify"]).is_err());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn scoreboard_measurement_commands_parse_their_explicit_manifest() {
        let manifest = std::path::Path::new("benchmarks/profile.json");

        let macllm_kernel = scoreboard_args(
            Cli::try_parse_from([
                "macllm",
                "scoreboard",
                "macllm-kernel",
                "--manifest",
                "benchmarks/profile.json",
            ])
            .unwrap(),
        );
        assert!(matches!(
            macllm_kernel.command,
            ScoreboardCommand::MacllmKernel { manifest: path } if path == manifest
        ));

        let llama_bench = scoreboard_args(
            Cli::try_parse_from([
                "macllm",
                "scoreboard",
                "llama-bench",
                "--manifest",
                "benchmarks/profile.json",
            ])
            .unwrap(),
        );
        assert!(matches!(
            llama_bench.command,
            ScoreboardCommand::LlamaBench { manifest: path } if path == manifest
        ));

        let llama_server = scoreboard_args(
            Cli::try_parse_from([
                "macllm",
                "scoreboard",
                "llama-server",
                "--manifest",
                "benchmarks/profile.json",
                "--port",
                "19000",
            ])
            .unwrap(),
        );
        assert!(matches!(
            llama_server.command,
            ScoreboardCommand::LlamaServer { manifest: path, port: 19_000 }
                if path == manifest
        ));
        let llama_server_default = scoreboard_args(
            Cli::try_parse_from([
                "macllm",
                "scoreboard",
                "llama-server",
                "--manifest",
                "benchmarks/profile.json",
            ])
            .unwrap(),
        );
        assert!(matches!(
            llama_server_default.command,
            ScoreboardCommand::LlamaServer { manifest: path, port: 18_081 }
                if path == manifest
        ));

        let macllm_serving = scoreboard_args(
            Cli::try_parse_from([
                "macllm",
                "scoreboard",
                "macllm-serving",
                "--manifest",
                "benchmarks/profile.json",
            ])
            .unwrap(),
        );
        assert!(matches!(
            macllm_serving.command,
            ScoreboardCommand::MacllmServing { manifest: path } if path == manifest
        ));

        let report = scoreboard_args(
            Cli::try_parse_from([
                "macllm",
                "scoreboard",
                "report",
                "--manifest",
                "benchmarks/profile.json",
                "--json",
                "out/report.json",
                "--markdown",
                "out/report.md",
            ])
            .unwrap(),
        );
        assert!(matches!(
            report.command,
            ScoreboardCommand::Report {
                manifest: path,
                port: 18_081,
                json,
                markdown,
            } if path == manifest
                && json == std::path::Path::new("out/report.json")
                && markdown == std::path::Path::new("out/report.md")
        ));

        for command in [
            "macllm-kernel",
            "llama-bench",
            "llama-server",
            "macllm-serving",
            "report",
        ] {
            assert!(Cli::try_parse_from(["macllm", "scoreboard", command]).is_err());
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn scoreboard_measurements_fail_closed_in_debug_builds() {
        let error = ensure_release_scoreboard_execution().unwrap_err();
        assert!(error.to_string().contains("cargo run --release"));
    }

    #[test]
    fn prefix_cache_demos_have_explicit_manifests_and_distinct_defaults() {
        let args = demo_args(
            Cli::try_parse_from([
                "macllm",
                "demo",
                "prefix-cache",
                "--manifest",
                "benchmarks/manifests/profile.json",
            ])
            .unwrap(),
        );
        assert!(matches!(
            args.command,
            DemoCommand::PrefixCache {
                manifest,
                repository,
                port: 18_085,
                json,
                markdown,
            } if manifest == std::path::Path::new("benchmarks/manifests/profile.json")
                && repository == std::path::Path::new(".")
                && json == std::path::Path::new("prefix-cache-demo.json")
                && markdown == std::path::Path::new("prefix-cache-demo.md")
        ));
        assert!(Cli::try_parse_from(["macllm", "demo", "prefix-cache"]).is_err());

        let fair = demo_args(
            Cli::try_parse_from([
                "macllm",
                "demo",
                "fair-prefix",
                "--manifest",
                "benchmarks/manifests/fair.json",
            ])
            .unwrap(),
        );
        assert!(matches!(
            fair.command,
            DemoCommand::FairPrefix {
                manifest,
                repository,
                port: 18_086,
                json,
                markdown,
            } if manifest == std::path::Path::new("benchmarks/manifests/fair.json")
                && repository == std::path::Path::new(".")
                && json == std::path::Path::new("fair-prefix-benchmark.json")
                && markdown == std::path::Path::new("fair-prefix-benchmark.md")
        ));
        assert!(Cli::try_parse_from(["macllm", "demo", "fair-prefix"]).is_err());
    }

    #[test]
    fn prefix_cache_demo_warms_llama_with_the_materialized_cache_seed() {
        let materialized = MaterializedPrefixDemo {
            seed: MaterializedPrefixPrompt {
                name: "long-reference-task".to_owned(),
                rendered_prompt: "16.4K task prompt".to_owned(),
                token_ids: vec![1, 2, 3],
            },
            cache_seed: MaterializedPrefixPrompt {
                name: "exact-cache-seed".to_owned(),
                rendered_prompt: "exact 16K raw prefix".to_owned(),
                token_ids: vec![1, 2],
            },
            tasks: Vec::new(),
        };

        let seed = llama_prefix_demo_cache_seed(&materialized);
        assert_eq!(seed.rendered_prompt, "exact 16K raw prefix");
        assert_eq!(seed.token_ids, [1, 2]);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn prefix_cache_demo_fails_closed_in_debug_builds() {
        let error = ensure_release_demo_execution().unwrap_err();
        assert!(error.to_string().contains("cargo run --release"));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn task_benchmarks_fail_closed_in_debug_builds() {
        let error = ensure_release_benchmark_execution().unwrap_err();
        assert!(error.to_string().contains("bench <generate|embed|rerank>"));
    }

    #[test]
    fn measurement_gate_rejects_unoptimized_custom_profiles() {
        assert!(
            ensure_optimized_measurement_execution(
                "bench",
                "release command",
                false,
                "release",
                "0"
            )
            .is_err()
        );
        assert!(
            ensure_optimized_measurement_execution(
                "bench",
                "release command",
                false,
                "release",
                "unknown",
            )
            .is_err()
        );
        assert!(
            ensure_optimized_measurement_execution(
                "bench",
                "release command",
                true,
                "release",
                "3"
            )
            .is_err()
        );
        assert!(
            ensure_optimized_measurement_execution(
                "bench",
                "release command",
                false,
                "custom",
                "1"
            )
            .is_ok()
        );
        assert!(
            ensure_optimized_measurement_execution("bench", "release command", false, "", "3")
                .is_err()
        );
        assert!(
            ensure_optimized_measurement_execution(
                "bench",
                "release command",
                false,
                "unknown",
                "3",
            )
            .is_err()
        );
        assert!(
            ensure_optimized_measurement_execution(
                "bench",
                "release command",
                false,
                "custom",
                "fast",
            )
            .is_err()
        );
    }

    #[test]
    fn blocking_commands_accept_explicit_prefill_quantum_above_auto_default() {
        let expected = NonZeroUsize::new(513).unwrap();
        let run = run_args(
            Cli::try_parse_from([
                "macllm",
                "run",
                "--model",
                "model-dir",
                "--prompt",
                "hello",
                "--prefill-token-quantum",
                "513",
            ])
            .unwrap(),
        );
        let bench = bench_generate_args(
            Cli::try_parse_from([
                "macllm",
                "bench",
                "generate",
                "--model",
                "model-dir",
                "--prompt",
                "hello",
                "--prefill-token-quantum",
                "513",
            ])
            .unwrap(),
        );

        assert_eq!(run.blocking_prefill.prefill_token_quantum, Some(expected));
        assert_eq!(bench.blocking_prefill.prefill_token_quantum, Some(expected));
        assert_eq!(
            resolve_blocking_prefill_token_quantum(
                &run.runtime,
                run.blocking_prefill.prefill_token_quantum,
            ),
            expected
        );
    }

    #[test]
    fn blocking_and_serving_commands_select_the_prefill_attention_path() {
        let native = run_args(
            Cli::try_parse_from(["macllm", "run", "--model", "model-dir", "--prompt", "hello"])
                .unwrap(),
        );
        assert_eq!(native.runtime.prefill_attention, PrefillAttentionMode::Sdpa);
        assert_eq!(
            model_load_config(&native.runtime).prefill_attention,
            PrefillAttentionMode::Sdpa
        );

        let fused_run = run_args(
            Cli::try_parse_from([
                "macllm",
                "run",
                "--model",
                "model-dir",
                "--prompt",
                "hello",
                "--prefill-attention",
                "sdpa",
            ])
            .unwrap(),
        );
        assert_eq!(
            fused_run.runtime.prefill_attention,
            PrefillAttentionMode::Sdpa
        );
        assert_eq!(
            model_load_config(&fused_run.runtime).prefill_attention,
            PrefillAttentionMode::Sdpa
        );

        let fused_bench = bench_generate_args(
            Cli::try_parse_from([
                "macllm",
                "bench",
                "generate",
                "--model",
                "model-dir",
                "--prompt",
                "hello",
                "--prefill-attention",
                "sdpa",
            ])
            .unwrap(),
        );
        assert_eq!(
            fused_bench.runtime.prefill_attention,
            PrefillAttentionMode::Sdpa
        );

        let fused_serve = serve_args(
            Cli::try_parse_from([
                "macllm",
                "serve",
                "--model",
                "model-dir",
                "--prefill-attention",
                "sdpa",
            ])
            .unwrap(),
        );
        assert_eq!(
            fused_serve.runtime.prefill_attention,
            PrefillAttentionMode::Sdpa
        );
        assert_eq!(
            model_load_config(&fused_serve.runtime).prefill_attention,
            PrefillAttentionMode::Sdpa
        );
    }

    #[test]
    fn blocking_commands_reject_zero_prefill_quantum() {
        for arguments in [
            vec![
                "macllm",
                "run",
                "--model",
                "model-dir",
                "--prompt",
                "hello",
                "--prefill-token-quantum",
                "0",
            ],
            vec![
                "macllm",
                "bench",
                "generate",
                "--model",
                "model-dir",
                "--prompt",
                "hello",
                "--prefill-token-quantum",
                "0",
            ],
        ] {
            let error = Cli::try_parse_from(arguments).unwrap_err();
            assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
        }
    }

    #[test]
    fn automatic_blocking_prefill_quantum_is_metal_and_dtype_aware() {
        let metal = run_args(
            Cli::try_parse_from([
                "macllm",
                "run",
                "--model",
                "model-dir",
                "--device",
                "metal",
                "--prompt",
                "hello",
            ])
            .unwrap(),
        );
        let cpu = run_args(
            Cli::try_parse_from([
                "macllm",
                "run",
                "--model",
                "model-dir",
                "--device",
                "cpu",
                "--prompt",
                "hello",
            ])
            .unwrap(),
        );
        let metal_bf16 = run_args(
            Cli::try_parse_from([
                "macllm",
                "run",
                "--model",
                "model-dir",
                "--device",
                "metal",
                "--dtype",
                "bf16",
                "--prompt",
                "hello",
            ])
            .unwrap(),
        );
        let default = run_args(
            Cli::try_parse_from(["macllm", "run", "--model", "model-dir", "--prompt", "hello"])
                .unwrap(),
        );

        assert_eq!(
            resolve_blocking_prefill_token_quantum(&metal.runtime, None),
            NonZeroUsize::new(128).unwrap()
        );
        assert_eq!(
            resolve_blocking_prefill_token_quantum(&cpu.runtime, None),
            NonZeroUsize::MAX
        );
        assert_eq!(
            resolve_blocking_prefill_token_quantum(&metal_bf16.runtime, None),
            NonZeroUsize::MAX
        );
        // The default device follows the feature selection of this build, so
        // the automatic quantum follows it as well.
        let expected_device = if cfg!(feature = "metal") {
            DeviceKind::Metal
        } else if cfg!(feature = "cuda") {
            DeviceKind::Cuda
        } else {
            DeviceKind::Cpu
        };
        assert_eq!(default.runtime.device, expected_device);
        let expected_default = if expected_device == DeviceKind::Metal {
            NonZeroUsize::new(128).unwrap()
        } else {
            NonZeroUsize::MAX
        };
        assert_eq!(
            resolve_blocking_prefill_token_quantum(&default.runtime, None),
            expected_default
        );
    }
}
