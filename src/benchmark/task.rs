//! Reproducible local diagnostics for generate, embed, and rerank.

use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    backend::{DTypeKind, DeviceKind},
    generation::{FinishReason, GenerationMetrics},
    rerank::RerankScore,
    runtime::{LoadedRuntime, ModelLoadConfig},
    weights::{ModelFiles, ModelFormat},
};

use super::file_stamp::file_stamp;
use super::provenance::parse_snapshot_provenance;
use super::{
    FileProvenance, FileProvenanceGuard, GREEDY_SEED, GreedySamplingSpec, MacllmProvenanceGuard,
    SanitizedHost, SnapshotProvenance, collect_macos_host, file_provenance, token_ids_sha256,
};

pub const TASK_BENCHMARK_SCHEMA_VERSION: u32 = 1;
const TASK_BENCHMARK_ARTIFACT: &str = "macllm.task_benchmark";
const GENERATE_BOUNDARY: &str =
    "cold session reset, prefill, decode, greedy sampling, output collection";
const EMBED_BOUNDARY: &str = "session allocation, packed forward, CPU readback, L2 normalization";
const RERANK_BOUNDARY: &str =
    "label tensor, session allocation, packed forward, selected-logit readback, sigmoid scoring";
const SNAPSHOT_MARKER_MAX_BYTES: usize = 64 * 1024;
const TASK_SIDECARS: [&str; 9] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "modules.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
    "1_LogitScore/config.json",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum TaskBenchmarkScope {
    LocalDiagnostic,
}

#[derive(Debug, Clone, Serialize)]
struct TaskBuildProvenance {
    implementation: String,
    package_version: String,
    git_commit: String,
    dirty: bool,
    executable: LocalFileDigest,
    cargo_profile: String,
    opt_level: String,
    debug_assertions: bool,
}

impl TaskBuildProvenance {
    fn from_guarded(implementation: super::ImplementationProvenance) -> Result<Self> {
        Ok(Self {
            implementation: "macllm".to_owned(),
            package_version: implementation.version,
            git_commit: implementation.commit,
            dirty: implementation
                .dirty
                .context("MacLLM build provenance has no dirty-state attestation")?,
            executable: LocalFileDigest::capture(&implementation.executable, None)?,
            cargo_profile: env!("MACLLM_BUILD_PROFILE").to_owned(),
            opt_level: env!("MACLLM_BUILD_OPT_LEVEL").to_owned(),
            debug_assertions: cfg!(debug_assertions),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct LocalFileDigest {
    path: String,
    size_bytes: u64,
    sha256: String,
}

impl LocalFileDigest {
    fn capture(file: &FileProvenance, base: Option<&Path>) -> Result<Self> {
        let path = match base {
            Some(base) => file
                .canonical_path
                .strip_prefix(base)
                .context("checkpoint file is outside its model root")?
                .to_str()
                .context("checkpoint relative path is not valid UTF-8")?
                .to_owned(),
            None => file.filename.clone(),
        };
        Ok(Self {
            path,
            size_bytes: file.size_bytes,
            sha256: file.sha256.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct TaskCheckpointProvenance {
    format: String,
    snapshot: Option<SnapshotProvenance>,
    weights: Vec<LocalFileDigest>,
    sidecars: Vec<LocalFileDigest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TaskCheckpointDirectoryStamp {
    path: PathBuf,
    device: u64,
    inode: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TaskCheckpointLayout {
    inventory: Vec<PathBuf>,
    directory_stamps: Vec<TaskCheckpointDirectoryStamp>,
}

#[derive(Debug)]
struct TaskCheckpointGuard {
    model: PathBuf,
    base: PathBuf,
    selected_files: Vec<PathBuf>,
    layout: TaskCheckpointLayout,
    files: Vec<FileProvenanceGuard>,
}

impl TaskCheckpointProvenance {
    fn capture(model: impl AsRef<Path>) -> Result<(Self, TaskCheckpointGuard)> {
        let model = canonical_task_model_path(model.as_ref())?;
        let directory_input = model.is_dir();
        let base = if directory_input {
            model.clone()
        } else {
            model
                .parent()
                .context("standalone model file has no parent directory")?
                .to_owned()
        };
        let initial_layout = checkpoint_layout(&base, &model, &[])?;
        let files = ModelFiles::from_dir(&model)?;
        let weight_files = files
            .weight_paths()
            .into_iter()
            .map(|path| checkpoint_file_provenance(&base, path))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            !weight_files.is_empty(),
            "task benchmark checkpoint has no weights"
        );
        let selected_weights = weight_files
            .iter()
            .map(|file| checkpoint_relative_path(&base, &file.canonical_path))
            .collect::<Result<Vec<_>>>()?;
        let mut sidecar_paths = Vec::new();
        if files.format() == ModelFormat::Safetensors {
            for relative in TASK_SIDECARS {
                let relative = PathBuf::from(relative);
                if initial_layout.inventory.contains(&relative) {
                    sidecar_paths.push(base.join(relative));
                }
            }
        }
        let marker_relative = PathBuf::from(".macllm-revision");
        let has_marker = directory_input && initial_layout.inventory.contains(&marker_relative);
        if has_marker {
            sidecar_paths.push(base.join(&marker_relative));
        }
        let sidecar_files = sidecar_paths
            .iter()
            .map(|path| checkpoint_file_provenance(&base, path))
            .collect::<Result<Vec<_>>>()?;
        let weights = weight_files
            .iter()
            .map(|file| LocalFileDigest::capture(file, Some(&base)))
            .collect::<Result<Vec<_>>>()?;
        let sidecars = sidecar_files
            .iter()
            .map(|file| LocalFileDigest::capture(file, Some(&base)))
            .collect::<Result<Vec<_>>>()?;
        let layout = checkpoint_layout(&base, &model, &selected_weights)?;
        ensure!(
            layout.inventory == initial_layout.inventory,
            "task benchmark checkpoint inventory changed while files were selected"
        );
        ensure!(
            initial_layout
                .directory_stamps
                .iter()
                .all(|stamp| layout.directory_stamps.contains(stamp)),
            "task benchmark checkpoint namespace changed while files were selected"
        );
        let mut guards = weight_files
            .iter()
            .chain(&sidecar_files)
            .map(FileProvenanceGuard::capture)
            .collect::<Result<Vec<_>>>()?;
        let snapshot = if has_marker {
            let marker = guards
                .last_mut()
                .context("snapshot marker has no checkpoint file guard")?;
            parse_snapshot_provenance(&marker.read_pinned_bytes(SNAPSHOT_MARKER_MAX_BYTES)?)
                .map(Some)?
        } else {
            None
        };
        if let Some(snapshot) = &snapshot {
            ensure!(snapshot.complete, "model snapshot marker is incomplete");
        }
        let refreshed = ModelFiles::from_dir(&model)?;
        let refreshed_weights = refreshed
            .weight_paths()
            .into_iter()
            .map(|path| resolve_checkpoint_file(&base, path))
            .map(|path| path.and_then(|path| checkpoint_relative_path(&base, &path)))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            refreshed.format() == files.format()
                && refreshed_weights == selected_weights
                && checkpoint_layout(&base, &model, &selected_weights)? == layout,
            "task benchmark checkpoint changed while provenance was captured"
        );
        let guard = TaskCheckpointGuard {
            model,
            base,
            selected_files: selected_weights,
            layout,
            files: guards,
        };
        Ok((
            Self {
                format: files.format().to_string(),
                snapshot,
                weights,
                sidecars,
            },
            guard,
        ))
    }
}

impl TaskCheckpointGuard {
    fn verify(&self) -> Result<()> {
        ensure!(
            checkpoint_layout(&self.base, &self.model, &self.selected_files)? == self.layout,
            "task benchmark checkpoint layout changed"
        );
        for file in &self.files {
            file.verify_identity()?;
        }
        ensure!(
            checkpoint_layout(&self.base, &self.model, &self.selected_files)? == self.layout,
            "task benchmark checkpoint changed while it was verified"
        );
        Ok(())
    }

    fn finish(self) -> Result<()> {
        ensure!(
            checkpoint_layout(&self.base, &self.model, &self.selected_files)? == self.layout,
            "task benchmark checkpoint layout changed"
        );
        for file in self.files {
            let _ = file.finish()?;
        }
        ensure!(
            checkpoint_layout(&self.base, &self.model, &self.selected_files)? == self.layout,
            "task benchmark checkpoint changed while it was finalized"
        );
        Ok(())
    }
}

fn canonical_task_model_path(model: &Path) -> Result<PathBuf> {
    fs::canonicalize(model)
        .with_context(|| format!("failed to resolve task benchmark model {}", model.display()))
}

fn checkpoint_relative_path(base: &Path, path: &Path) -> Result<PathBuf> {
    let relative = path
        .strip_prefix(base)
        .context("checkpoint file is outside its model root")?;
    ensure!(
        relative.components().next().is_some()
            && relative
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "checkpoint path is not a normalized relative file path: {}",
        path.display()
    );
    Ok(relative.to_owned())
}

fn resolve_checkpoint_file(base: &Path, requested: &Path) -> Result<PathBuf> {
    let relative = checkpoint_relative_path(base, requested)?;
    let mut current = base.to_owned();
    let component_count = relative.components().count();
    for (index, component) in relative.components().enumerate() {
        let Component::Normal(component) = component else {
            anyhow::bail!("checkpoint path contains a non-normal component");
        };
        current.push(component);
        let metadata = fs::symlink_metadata(&current)
            .with_context(|| format!("failed to inspect checkpoint path {}", current.display()))?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "task benchmark checkpoint rejects symlink {}",
            relative.display()
        );
        if index + 1 == component_count {
            ensure!(
                metadata.file_type().is_file(),
                "task benchmark checkpoint path is not a regular file: {}",
                relative.display()
            );
        } else {
            ensure!(
                metadata.file_type().is_dir(),
                "task benchmark checkpoint parent is not a directory: {}",
                relative.display()
            );
        }
    }
    let canonical = fs::canonicalize(&current)
        .with_context(|| format!("failed to resolve checkpoint file {}", current.display()))?;
    ensure!(
        canonical.starts_with(base),
        "checkpoint file resolves outside its model root: {}",
        relative.display()
    );
    Ok(canonical)
}

fn checkpoint_file_provenance(base: &Path, path: &Path) -> Result<FileProvenance> {
    let canonical = resolve_checkpoint_file(base, path)?;
    let provenance = file_provenance(path)?;
    ensure!(
        provenance.canonical_path == canonical,
        "checkpoint path changed while provenance was captured"
    );
    Ok(provenance)
}

fn checkpoint_layout(
    base: &Path,
    model: &Path,
    selected_files: &[PathBuf],
) -> Result<TaskCheckpointLayout> {
    let mut inventory = BTreeSet::new();
    let directory_input = model.is_dir();
    if directory_input {
        for entry in fs::read_dir(base)
            .with_context(|| format!("failed to enumerate checkpoint root {}", base.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "gguf" || extension == "safetensors")
            {
                let relative = checkpoint_relative_path(base, &path)?;
                let _ = resolve_checkpoint_file(base, &path)?;
                inventory.insert(relative);
            }
        }
        for relative in TASK_SIDECARS.into_iter().chain([".macllm-revision"]) {
            let path = base.join(relative);
            if checkpoint_entry_exists(&path)? {
                let _ = resolve_checkpoint_file(base, &path)?;
                inventory.insert(PathBuf::from(relative));
            }
        }
    } else {
        ensure!(model.is_file(), "task benchmark model path changed");
        let _ = resolve_checkpoint_file(base, model)?;
        inventory.insert(checkpoint_relative_path(base, model)?);
    }
    let inventory = inventory.into_iter().collect::<Vec<_>>();
    let mut directories = BTreeSet::from([base.to_owned()]);
    for relative in selected_files {
        let selected = base.join(relative);
        if let Some(parent) = selected.parent() {
            directories.extend(
                parent
                    .ancestors()
                    .take_while(|ancestor| ancestor.starts_with(base))
                    .map(Path::to_owned),
            );
        }
    }
    if directory_input {
        for relative in TASK_SIDECARS {
            let candidate = base.join(relative);
            for parent in candidate
                .ancestors()
                .skip(1)
                .take_while(|parent| parent.starts_with(base))
            {
                if checkpoint_entry_exists(parent)? {
                    directories.extend(
                        parent
                            .ancestors()
                            .take_while(|ancestor| ancestor.starts_with(base))
                            .map(Path::to_owned),
                    );
                    break;
                }
            }
        }
    }
    let directory_stamps = directories
        .into_iter()
        .map(|path| checkpoint_directory_stamp(&path))
        .collect::<Result<Vec<_>>>()?;
    Ok(TaskCheckpointLayout {
        inventory,
        directory_stamps,
    })
}

fn checkpoint_entry_exists(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error)
            .with_context(|| format!("failed to inspect checkpoint entry {}", path.display())),
    }
}

fn checkpoint_directory_stamp(path: &Path) -> Result<TaskCheckpointDirectoryStamp> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect checkpoint directory {}", path.display()))?;
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "checkpoint directory must be a real directory: {}",
        path.display()
    );
    ensure!(
        fs::canonicalize(path)? == path,
        "checkpoint directory path is not canonical: {}",
        path.display()
    );
    let stamp = file_stamp(&metadata);
    Ok(TaskCheckpointDirectoryStamp {
        path: path.to_owned(),
        device: stamp.device,
        inode: stamp.inode,
        modified_seconds: stamp.modified_seconds,
        modified_nanoseconds: stamp.modified_nanoseconds,
        changed_seconds: stamp.changed_seconds,
        changed_nanoseconds: stamp.changed_nanoseconds,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskRuntimeConfig {
    family: String,
    device: String,
    dtype: String,
    kv_cache_block_tokens: usize,
    kv_cache_slab_blocks: usize,
    requested_kv_cache_capacity_tokens: Option<usize>,
    effective_kv_cache_capacity_tokens: usize,
    prefix_cache_capacity_tokens: usize,
    prefill_token_quantum: Option<usize>,
}

impl TaskRuntimeConfig {
    pub fn capture(
        runtime: &LoadedRuntime,
        config: &ModelLoadConfig,
        prefill_token_quantum: Option<usize>,
    ) -> Result<Self> {
        let family = runtime.adapter.family().to_owned();
        ensure!(!family.trim().is_empty(), "model family cannot be empty");
        ensure!(
            config.kv_cache_block_tokens > 0
                && config.kv_cache_slab_blocks > 0
                && config
                    .kv_cache_capacity_tokens
                    .is_none_or(|tokens| tokens > 0)
                && config.prefix_cache_capacity_tokens == 0
                && prefill_token_quantum.is_none_or(|tokens| tokens > 0),
            "task benchmark runtime configuration is invalid or enables prefix caching"
        );
        let capacity = runtime
            .model
            .kv_cache_capacity()
            .context("task benchmark model does not expose bounded KV cache capacity")?;
        ensure!(
            capacity.block_tokens == config.kv_cache_block_tokens,
            "runtime KV block size disagrees with the benchmark configuration"
        );
        Ok(Self {
            family,
            device: match config.device {
                DeviceKind::Cpu => "cpu",
                DeviceKind::Metal => "metal",
            }
            .to_owned(),
            dtype: match runtime.dtype {
                DTypeKind::F32 => "f32",
                DTypeKind::F16 => "f16",
                DTypeKind::Bf16 => "bf16",
            }
            .to_owned(),
            kv_cache_block_tokens: config.kv_cache_block_tokens,
            kv_cache_slab_blocks: config.kv_cache_slab_blocks,
            requested_kv_cache_capacity_tokens: config.kv_cache_capacity_tokens,
            effective_kv_cache_capacity_tokens: capacity
                .block_tokens
                .checked_mul(capacity.total_blocks)
                .context("runtime KV token capacity overflow")?,
            prefix_cache_capacity_tokens: config.prefix_cache_capacity_tokens,
            prefill_token_quantum,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct TaskInputFingerprint {
    lane_token_lengths: Vec<usize>,
    batch_token_ids_sha256: String,
}

impl TaskInputFingerprint {
    fn capture(lanes: &[Vec<u32>]) -> Result<Self> {
        ensure!(!lanes.is_empty(), "task benchmark input batch is empty");
        ensure!(
            lanes.iter().all(|lane| !lane.is_empty()),
            "task benchmark input contains an empty lane"
        );
        let mut digest = Sha256::new();
        digest.update(b"macllm.task_input.v1\0");
        digest.update(u64::try_from(lanes.len())?.to_le_bytes());
        for lane in lanes {
            digest.update(u64::try_from(lane.len())?.to_le_bytes());
            for token in lane {
                digest.update(token.to_le_bytes());
            }
        }
        Ok(Self {
            lane_token_lengths: lanes.iter().map(Vec::len).collect(),
            batch_token_ids_sha256: format!("{:x}", digest.finalize()),
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct TaskRepeat {
    warmup_runs: usize,
    measured_runs: usize,
}

impl TaskRepeat {
    #[must_use]
    pub const fn new(warmup_runs: usize, measured_runs: usize) -> Self {
        Self {
            warmup_runs,
            measured_runs,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct TaskOutputFingerprint {
    sha256: String,
}

impl TaskOutputFingerprint {
    #[must_use]
    pub fn generated_tokens(tokens: &[u32]) -> Self {
        Self {
            sha256: token_ids_sha256(tokens),
        }
    }

    #[must_use]
    pub fn embeddings(rows: &[Vec<f32>]) -> Self {
        Self {
            sha256: f32_rows_sha256(rows.iter().map(Vec::as_slice)),
        }
    }

    #[must_use]
    pub fn rerank(scores: &[RerankScore]) -> Self {
        Self {
            sha256: f32_rows_sha256(
                scores
                    .iter()
                    .map(|score| [score.margin, score.yes_probability]),
            ),
        }
    }
}

fn f32_rows_sha256<'a, I, R>(rows: I) -> String
where
    I: IntoIterator<Item = R>,
    R: AsRef<[f32]> + 'a,
{
    let rows = rows.into_iter().collect::<Vec<_>>();
    let mut digest = Sha256::new();
    digest.update(b"macllm.task_output.f32.v1\0");
    digest.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        let row = row.as_ref();
        digest.update((row.len() as u64).to_le_bytes());
        for value in row {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskBenchmarkCommon {
    artifact: String,
    schema_version: u32,
    scope: TaskBenchmarkScope,
    generated_unix_ms: u64,
    host: SanitizedHost,
    build: TaskBuildProvenance,
    checkpoint: TaskCheckpointProvenance,
    runtime: TaskRuntimeConfig,
    input: TaskInputFingerprint,
    repeat: TaskRepeat,
}

#[derive(Debug)]
pub struct TaskBenchmarkGuard {
    generated_unix_ms: u64,
    host: SanitizedHost,
    build: MacllmProvenanceGuard,
    checkpoint: TaskCheckpointProvenance,
    checkpoint_guard: TaskCheckpointGuard,
}

impl TaskBenchmarkGuard {
    pub fn capture(config: &ModelLoadConfig) -> Result<Self> {
        ensure!(
            !cfg!(debug_assertions)
                && matches!(env!("MACLLM_BUILD_OPT_LEVEL"), "1" | "2" | "3" | "s" | "z"),
            "task benchmarks require an optimized build"
        );
        // The host is collected first, so a host that cannot publish a
        // benchmark refuses before the model is hashed.
        let host = collect_macos_host()?;
        let build = MacllmProvenanceGuard::capture_publishable()?;
        let (checkpoint, checkpoint_guard) = TaskCheckpointProvenance::capture(&config.model)?;
        Ok(Self {
            generated_unix_ms: u64::try_from(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .context("system time precedes the Unix epoch")?
                    .as_millis(),
            )?,
            host,
            build,
            checkpoint,
            checkpoint_guard,
        })
    }

    /// Canonical model path sealed by this guard. Benchmark preparation and
    /// runtime loading must both use this path rather than re-resolving the
    /// caller's original path.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.checkpoint_guard.model
    }

    /// Recheck checkpoint layout and open-file identities after loading and
    /// outside the measured task boundary. [`Self::finish`] performs the final
    /// full content rehash. The canonical model root's ancestors are a trusted
    /// namespace boundary rather than an actively hostile filesystem.
    pub fn verify_checkpoint(&self) -> Result<()> {
        self.checkpoint_guard.verify()
    }

    pub fn finish(
        self,
        runtime: TaskRuntimeConfig,
        input: &[Vec<u32>],
        repeat: TaskRepeat,
    ) -> Result<TaskBenchmarkCommon> {
        ensure!(repeat.measured_runs > 0, "measured runs must be positive");
        self.checkpoint_guard.finish()?;
        let build = TaskBuildProvenance::from_guarded(self.build.finish()?)?;
        Ok(TaskBenchmarkCommon {
            artifact: TASK_BENCHMARK_ARTIFACT.to_owned(),
            schema_version: TASK_BENCHMARK_SCHEMA_VERSION,
            scope: TaskBenchmarkScope::LocalDiagnostic,
            generated_unix_ms: self.generated_unix_ms,
            host: self.host,
            build,
            checkpoint: self.checkpoint,
            runtime,
            input: TaskInputFingerprint::capture(input)?,
            repeat,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskEmbeddingInputType {
    Query,
    Document,
}

#[derive(Debug, Clone, Serialize)]
struct GenerateBenchmarkConfig {
    measurement_boundary: &'static str,
    max_new_tokens: usize,
    sampling: GreedySamplingSpec,
}

#[derive(Debug, Clone, Serialize)]
struct EmbedBenchmarkConfig {
    measurement_boundary: &'static str,
    input_type: TaskEmbeddingInputType,
    dimension: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RerankBenchmarkConfig {
    measurement_boundary: &'static str,
    no_token_id: u32,
    yes_token_id: u32,
}

#[derive(Debug, Clone, Serialize)]
struct GenerateSample {
    run: usize,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    processed_prompt_tokens: usize,
    output_tokens: usize,
    decode_steps: usize,
    prefix_lookup_attach_ns: u64,
    suffix_prefill_ns: u64,
    first_token_model_ns: u64,
    prefill_ns: u64,
    decode_ns: u64,
    time_to_first_token_ns: u64,
    total_ns: u64,
}

impl GenerateSample {
    fn capture(
        run: usize,
        metrics: GenerationMetrics,
        expected_prompt_tokens: usize,
        expected_output_tokens: usize,
    ) -> Result<Self> {
        ensure!(
            metrics.finish_reason == FinishReason::Length
                && metrics.prompt_tokens == expected_prompt_tokens
                && metrics.cached_prompt_tokens == 0
                && metrics.processed_prompt_tokens() == expected_prompt_tokens
                && metrics.generated_tokens == expected_output_tokens
                && metrics.decode_steps == expected_output_tokens.saturating_sub(1),
            "generation benchmark sample {run} violates the fixed cold workload"
        );
        Ok(Self {
            run,
            prompt_tokens: metrics.prompt_tokens,
            cached_prompt_tokens: metrics.cached_prompt_tokens,
            processed_prompt_tokens: metrics.processed_prompt_tokens(),
            output_tokens: metrics.generated_tokens,
            decode_steps: metrics.decode_steps,
            prefix_lookup_attach_ns: duration_ns(metrics.prefix_lookup_attach_time)?,
            suffix_prefill_ns: duration_ns(metrics.suffix_prefill_time)?,
            first_token_model_ns: duration_ns(metrics.first_token_model_time)?,
            prefill_ns: duration_ns(metrics.prefill_time)?,
            decode_ns: duration_ns(metrics.decode_time)?,
            time_to_first_token_ns: duration_ns(metrics.time_to_first_token)?,
            total_ns: duration_ns(metrics.total_time)?,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct LatencySample {
    run: usize,
    latency_ns: u64,
}

impl LatencySample {
    fn capture(run: usize, latency: Duration) -> Result<Self> {
        let latency_ns = duration_ns(latency)?;
        ensure!(latency_ns > 0, "task benchmark latency must be positive");
        Ok(Self { run, latency_ns })
    }
}

fn duration_ns(duration: Duration) -> Result<u64> {
    u64::try_from(duration.as_nanos()).context("task benchmark duration exceeds u64 nanoseconds")
}

#[derive(Debug, Clone, Serialize)]
#[serde(transparent)]
pub struct TaskBenchmarkReport(TaskBenchmarkKind);

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "task", rename_all = "snake_case")]
enum TaskBenchmarkKind {
    Generate {
        common: TaskBenchmarkCommon,
        config: GenerateBenchmarkConfig,
        deterministic_output_sha256: String,
        samples: Vec<GenerateSample>,
    },
    Embed {
        common: TaskBenchmarkCommon,
        config: EmbedBenchmarkConfig,
        deterministic_output_sha256: String,
        samples: Vec<LatencySample>,
    },
    Rerank {
        common: TaskBenchmarkCommon,
        config: RerankBenchmarkConfig,
        deterministic_output_sha256: String,
        samples: Vec<LatencySample>,
    },
}

impl TaskBenchmarkReport {
    pub fn generate(
        common: TaskBenchmarkCommon,
        max_new_tokens: usize,
        runs: Vec<(GenerationMetrics, Vec<u32>)>,
    ) -> Result<Self> {
        ensure!(
            max_new_tokens > 0,
            "generation token count must be positive"
        );
        ensure_sample_count(common.repeat, runs.len())?;
        let prompt_tokens = common
            .input
            .lane_token_lengths
            .first()
            .copied()
            .filter(|_| common.input.lane_token_lengths.len() == 1)
            .context("generation benchmark requires exactly one input lane")?;
        let mut output_sha256 = None;
        let samples = runs
            .into_iter()
            .enumerate()
            .map(|(run, (metrics, output))| {
                ensure!(
                    output.len() == max_new_tokens,
                    "generation output length changed during measured run {}",
                    run + 1
                );
                retain_output_digest(
                    &mut output_sha256,
                    TaskOutputFingerprint::generated_tokens(&output).sha256,
                    run,
                )?;
                GenerateSample::capture(run + 1, metrics, prompt_tokens, max_new_tokens)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(TaskBenchmarkKind::Generate {
            common,
            config: GenerateBenchmarkConfig {
                measurement_boundary: GENERATE_BOUNDARY,
                max_new_tokens,
                sampling: fixed_greedy_sampling(),
            },
            deterministic_output_sha256: output_sha256.context("generation output is missing")?,
            samples,
        }))
    }

    pub fn embed(
        common: TaskBenchmarkCommon,
        input_type: TaskEmbeddingInputType,
        dimension: usize,
        runs: Vec<(Duration, Vec<Vec<f32>>)>,
    ) -> Result<Self> {
        ensure!(dimension > 0, "embedding dimension must be positive");
        ensure_sample_count(common.repeat, runs.len())?;
        let mut output_sha256 = None;
        let samples = runs
            .into_iter()
            .enumerate()
            .map(|(run, (latency, embeddings))| {
                ensure!(
                    embeddings.len() == common.input.lane_token_lengths.len()
                        && embeddings.iter().all(|row| row.len() == dimension),
                    "embedding output shape changed during measured run {}",
                    run + 1
                );
                retain_output_digest(
                    &mut output_sha256,
                    TaskOutputFingerprint::embeddings(&embeddings).sha256,
                    run,
                )?;
                LatencySample::capture(run + 1, latency)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(TaskBenchmarkKind::Embed {
            samples,
            common,
            config: EmbedBenchmarkConfig {
                measurement_boundary: EMBED_BOUNDARY,
                input_type,
                dimension,
            },
            deterministic_output_sha256: output_sha256.context("embedding output is missing")?,
        }))
    }

    pub fn rerank(
        common: TaskBenchmarkCommon,
        no_token_id: u32,
        yes_token_id: u32,
        runs: Vec<(Duration, Vec<RerankScore>)>,
    ) -> Result<Self> {
        ensure!(
            no_token_id != yes_token_id,
            "rerank label token IDs must differ"
        );
        ensure_sample_count(common.repeat, runs.len())?;
        let mut output_sha256 = None;
        let samples = runs
            .into_iter()
            .enumerate()
            .map(|(run, (latency, scores))| {
                ensure!(
                    scores.len() == common.input.lane_token_lengths.len(),
                    "rerank output count changed during measured run {}",
                    run + 1
                );
                retain_output_digest(
                    &mut output_sha256,
                    TaskOutputFingerprint::rerank(&scores).sha256,
                    run,
                )?;
                LatencySample::capture(run + 1, latency)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(TaskBenchmarkKind::Rerank {
            samples,
            common,
            config: RerankBenchmarkConfig {
                measurement_boundary: RERANK_BOUNDARY,
                no_token_id,
                yes_token_id,
            },
            deterministic_output_sha256: output_sha256.context("rerank output is missing")?,
        }))
    }
}

fn fixed_greedy_sampling() -> GreedySamplingSpec {
    GreedySamplingSpec {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: GREEDY_SEED,
        ignore_stop_tokens: true,
    }
}

fn ensure_sample_count(repeat: TaskRepeat, samples: usize) -> Result<()> {
    ensure!(
        samples == repeat.measured_runs,
        "task benchmark sample count does not match measured runs"
    );
    Ok(())
}

fn retain_output_digest(expected: &mut Option<String>, digest: String, run: usize) -> Result<()> {
    if let Some(expected) = expected {
        ensure!(
            *expected == digest,
            "task benchmark output changed during measured run {}",
            run + 1
        );
    } else {
        *expected = Some(digest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    fn common(lanes: &[Vec<u32>]) -> Result<TaskBenchmarkCommon> {
        Ok(TaskBenchmarkCommon {
            artifact: TASK_BENCHMARK_ARTIFACT.to_owned(),
            schema_version: TASK_BENCHMARK_SCHEMA_VERSION,
            scope: TaskBenchmarkScope::LocalDiagnostic,
            generated_unix_ms: 1,
            host: SanitizedHost {
                machine_name: "Mac".to_owned(),
                machine_model: "Mac1,1".to_owned(),
                chip: "Apple Test".to_owned(),
                memory_bytes: 1,
                total_cores: Some(2),
                performance_cores: Some(1),
                efficiency_cores: Some(1),
                os_version: "1".to_owned(),
                os_build: "A".to_owned(),
                architecture: "arm64".to_owned(),
            },
            build: TaskBuildProvenance {
                implementation: "macllm".to_owned(),
                package_version: "0.1.0".to_owned(),
                git_commit: "0".repeat(40),
                dirty: false,
                executable: LocalFileDigest {
                    path: "macllm".to_owned(),
                    size_bytes: 1,
                    sha256: "0".repeat(64),
                },
                cargo_profile: "release".to_owned(),
                opt_level: "3".to_owned(),
                debug_assertions: false,
            },
            checkpoint: TaskCheckpointProvenance {
                format: "safetensors".to_owned(),
                snapshot: None,
                weights: vec![LocalFileDigest {
                    path: "model.safetensors".to_owned(),
                    size_bytes: 1,
                    sha256: "1".repeat(64),
                }],
                sidecars: Vec::new(),
            },
            runtime: TaskRuntimeConfig {
                family: "qwen3".to_owned(),
                device: "metal".to_owned(),
                dtype: "f16".to_owned(),
                kv_cache_block_tokens: 16,
                kv_cache_slab_blocks: 8,
                requested_kv_cache_capacity_tokens: None,
                effective_kv_cache_capacity_tokens: 1024,
                prefix_cache_capacity_tokens: 0,
                prefill_token_quantum: None,
            },
            input: TaskInputFingerprint::capture(lanes)?,
            repeat: TaskRepeat::new(1, 1),
        })
    }

    fn generation_metrics(cached_prompt_tokens: usize) -> GenerationMetrics {
        GenerationMetrics {
            prompt_tokens: 2,
            cached_prompt_tokens,
            prefix_lookup_attach_time: Duration::from_nanos(1),
            generated_tokens: 2,
            decode_steps: 1,
            time_to_first_token: Duration::from_nanos(6),
            suffix_prefill_time: Duration::from_nanos(3),
            first_token_model_time: Duration::from_nanos(2),
            prefill_time: Duration::from_nanos(5),
            decode_time: Duration::from_nanos(4),
            total_time: Duration::from_nanos(10),
            finish_reason: FinishReason::Length,
        }
    }

    fn write_task_checkpoint(root: &Path) -> Result<()> {
        fs::write(root.join("config.json"), b"config")?;
        fs::write(root.join("tokenizer.json"), b"tokenizer")?;
        fs::write(root.join("model.safetensors"), b"weights")?;
        fs::write(
            root.join(".macllm-revision"),
            br#"{
              "repo_id":"owner/model",
              "revision":"0123456789abcdef0123456789abcdef01234567",
              "complete":true
            }"#,
        )?;
        Ok(())
    }

    #[test]
    fn token_batch_digest_preserves_lane_boundaries_and_order() -> Result<()> {
        let left = TaskInputFingerprint::capture(&[vec![1], vec![2, 3]])?;
        let right = TaskInputFingerprint::capture(&[vec![1, 2], vec![3]])?;
        let reordered = TaskInputFingerprint::capture(&[vec![2, 3], vec![1]])?;
        assert_ne!(left.batch_token_ids_sha256, right.batch_token_ids_sha256);
        assert_ne!(
            left.batch_token_ids_sha256,
            reordered.batch_token_ids_sha256
        );
        assert_eq!(left.lane_token_lengths, [1, 2]);
        Ok(())
    }

    #[test]
    fn output_digests_preserve_float_bits_and_order() {
        let embeddings = TaskOutputFingerprint::embeddings(&[vec![1.0, -0.0], vec![2.0]]);
        let changed_sign = TaskOutputFingerprint::embeddings(&[vec![1.0, 0.0], vec![2.0]]);
        let reordered = TaskOutputFingerprint::embeddings(&[vec![2.0], vec![1.0, -0.0]]);
        assert_ne!(embeddings.sha256, changed_sign.sha256);
        assert_ne!(embeddings.sha256, reordered.sha256);

        let score = RerankScore {
            margin: 1.0,
            yes_probability: 0.75,
        };
        let changed = RerankScore {
            yes_probability: 0.5,
            ..score
        };
        assert_ne!(
            TaskOutputFingerprint::rerank(&[score]).sha256,
            TaskOutputFingerprint::rerank(&[changed]).sha256
        );
    }

    #[test]
    fn checkpoint_capture_hashes_weights_sidecars_and_revision() -> Result<()> {
        let directory = tempfile::tempdir()?;
        write_task_checkpoint(directory.path())?;

        let (provenance, guard) = TaskCheckpointProvenance::capture(directory.path())?;
        guard.finish()?;
        assert_eq!(provenance.format, "safetensors");
        assert_eq!(provenance.weights[0].path, "model.safetensors");
        assert_eq!(provenance.sidecars.len(), 3);
        assert_eq!(
            provenance
                .snapshot
                .as_ref()
                .map(|value| value.revision.as_str()),
            Some("0123456789abcdef0123456789abcdef01234567")
        );
        assert_eq!(
            provenance.weights[0].sha256,
            crate::benchmark::sha256_bytes(b"weights")
        );
        let json = serde_json::to_string(&provenance)?;
        assert!(!json.contains(directory.path().to_string_lossy().as_ref()));
        assert!(!json.contains("canonical_path"));
        Ok(())
    }

    #[test]
    fn standalone_model_does_not_inherit_parent_snapshot() -> Result<()> {
        let directory = tempfile::tempdir()?;
        fs::write(directory.path().join(".macllm-revision"), b"unrelated")?;
        fs::write(directory.path().join("model.gguf"), b"weights")?;
        let base = fs::canonicalize(directory.path())?;
        let model = base.join("model.gguf");
        let layout = checkpoint_layout(&base, &model, &[])?;
        assert_eq!(layout.inventory, [PathBuf::from("model.gguf")]);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn snapshot_parse_is_bound_to_the_captured_marker_digest() -> Result<()> {
        let directory = tempfile::tempdir()?;
        write_task_checkpoint(directory.path())?;
        let (provenance, guard) = TaskCheckpointProvenance::capture(directory.path())?;
        let snapshot = provenance
            .snapshot
            .context("guarded marker did not produce snapshot provenance")?;
        assert_eq!(snapshot.repository, "owner/model");
        let replacement = directory.path().join("replacement-marker");
        fs::write(
            &replacement,
            br#"{
              "repo_id":"other/model",
              "revision":"fedcba9876543210fedcba9876543210fedcba98",
              "complete":true
            }"#,
        )?;
        fs::rename(&replacement, directory.path().join(".macllm-revision"))?;
        assert!(guard.verify().is_err());
        Ok(())
    }

    #[test]
    fn checkpoint_guard_rejects_new_candidate_files() -> Result<()> {
        let directory = tempfile::tempdir()?;
        write_task_checkpoint(directory.path())?;
        let (_, guard) = TaskCheckpointProvenance::capture(directory.path())?;
        let added = directory.path().join("untracked.safetensors");
        fs::write(&added, b"other")?;
        fs::remove_file(&added)?;
        assert!(guard.verify().is_err());
        Ok(())
    }

    #[test]
    fn checkpoint_guard_rejects_transient_nested_sidecar() -> Result<()> {
        let directory = tempfile::tempdir()?;
        write_task_checkpoint(directory.path())?;
        let pooling = directory.path().join("1_Pooling");
        fs::create_dir(&pooling)?;
        let (_, guard) = TaskCheckpointProvenance::capture(directory.path())?;
        let sidecar = pooling.join("config.json");
        fs::write(&sidecar, b"temporary")?;
        fs::remove_file(sidecar)?;
        assert!(guard.verify().is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_capture_rejects_dangling_sidecar_entries() -> Result<()> {
        use std::os::unix::fs::symlink;

        let sidecar = tempfile::tempdir()?;
        write_task_checkpoint(sidecar.path())?;
        symlink("missing", sidecar.path().join("tokenizer_config.json"))?;
        assert!(TaskCheckpointProvenance::capture(sidecar.path()).is_err());

        let parent = tempfile::tempdir()?;
        write_task_checkpoint(parent.path())?;
        symlink("missing", parent.path().join("1_Pooling"))?;
        assert!(TaskCheckpointProvenance::capture(parent.path()).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_guard_tracks_nested_selected_weight_parent() -> Result<()> {
        let directory = tempfile::tempdir()?;
        write_task_checkpoint(directory.path())?;
        fs::remove_file(directory.path().join("model.safetensors"))?;
        let shards = directory.path().join("shards");
        let parts = shards.join("parts");
        fs::create_dir_all(&parts)?;
        let weight = parts.join("model-00001-of-00001.safetensors");
        fs::write(&weight, b"weights")?;
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"model.weight":"shards/parts/model-00001-of-00001.safetensors"}}"#,
        )?;
        let (_, guard) = TaskCheckpointProvenance::capture(directory.path())?;
        let saved = shards.join("saved-parts");
        fs::rename(&parts, &saved)?;
        fs::create_dir(&parts)?;
        fs::write(
            parts.join("model-00001-of-00001.safetensors"),
            b"replacement",
        )?;
        fs::rename(&parts, shards.join("replacement-parts"))?;
        fs::rename(saved, parts)?;
        assert!(guard.verify().is_err());
        Ok(())
    }

    #[cfg(not(unix))]
    #[test]
    fn checkpoint_guard_keeps_the_nested_weight_parent_in_place() -> Result<()> {
        let directory = tempfile::tempdir()?;
        write_task_checkpoint(directory.path())?;
        fs::remove_file(directory.path().join("model.safetensors"))?;
        let shards = directory.path().join("shards");
        let parts = shards.join("parts");
        fs::create_dir_all(&parts)?;
        let weight = parts.join("model-00001-of-00001.safetensors");
        fs::write(&weight, b"weights")?;
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"model.weight":"shards/parts/model-00001-of-00001.safetensors"}}"#,
        )?;
        let (_, guard) = TaskCheckpointProvenance::capture(directory.path())?;
        // The guard keeps the nested weight open, and Windows refuses to
        // rename a directory that holds an open file. The tracked parent
        // directory therefore stays where it was captured, so the checkpoint
        // layout is unchanged while the guard is alive.
        assert!(fs::rename(&parts, shards.join("saved-parts")).is_err());
        guard.verify()?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_guard_pins_root_and_rejects_weight_symlink_retarget() -> Result<()> {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir()?;
        let checkpoint = directory.path().join("checkpoint");
        fs::create_dir(&checkpoint)?;
        write_task_checkpoint(&checkpoint)?;
        let other_checkpoint = directory.path().join("other-checkpoint");
        fs::create_dir(&other_checkpoint)?;
        write_task_checkpoint(&other_checkpoint)?;
        let root_link = directory.path().join("checkpoint-link");
        symlink(&checkpoint, &root_link)?;
        let (_, guard) = TaskCheckpointProvenance::capture(&root_link)?;
        assert_eq!(guard.model, fs::canonicalize(&checkpoint)?);
        fs::remove_file(&root_link)?;
        symlink(&other_checkpoint, &root_link)?;
        guard.verify()?;

        let replacement = directory.path().join("replacement.safetensors");
        fs::write(&replacement, b"replacement")?;
        let weight = checkpoint.join("model.safetensors");
        fs::remove_file(&weight)?;
        symlink(&replacement, &weight)?;
        assert!(guard.verify().is_err());
        Ok(())
    }

    #[test]
    fn latency_samples_reject_zero_duration() {
        assert!(LatencySample::capture(1, Duration::ZERO).is_err());
        assert!(LatencySample::capture(1, Duration::from_nanos(1)).is_ok());
    }

    #[test]
    fn task_constructors_bind_output_shapes_and_cold_generation() -> Result<()> {
        let generation = TaskBenchmarkReport::generate(
            common(&[vec![1, 2]])?,
            2,
            vec![(generation_metrics(0), vec![3, 4])],
        )?;
        let json = serde_json::to_string(&generation)?;
        assert!(json.contains("\"task\":\"generate\""));
        assert!(json.contains(GENERATE_BOUNDARY));
        assert!(!json.contains("canonical_path"));
        assert!(!json.contains("/Users/"));
        assert!(
            TaskBenchmarkReport::generate(
                common(&[vec![1, 2]])?,
                2,
                vec![(generation_metrics(1), vec![3, 4])],
            )
            .is_err()
        );

        let embedding_common = common(&[vec![1], vec![2]])?;
        assert!(
            TaskBenchmarkReport::embed(
                embedding_common.clone(),
                TaskEmbeddingInputType::Query,
                2,
                vec![(Duration::from_nanos(1), vec![vec![1.0, 0.0]])],
            )
            .is_err()
        );
        TaskBenchmarkReport::embed(
            embedding_common,
            TaskEmbeddingInputType::Query,
            2,
            vec![(
                Duration::from_nanos(1),
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            )],
        )?;

        let rerank_common = common(&[vec![1], vec![2]])?;
        assert!(
            TaskBenchmarkReport::rerank(
                rerank_common,
                1,
                2,
                vec![(
                    Duration::from_nanos(1),
                    vec![RerankScore {
                        margin: 1.0,
                        yes_probability: 0.75,
                    }],
                )],
            )
            .is_err()
        );

        let mut drift = common(&[vec![1]])?;
        drift.repeat = TaskRepeat::new(0, 2);
        assert!(
            TaskBenchmarkReport::embed(
                drift,
                TaskEmbeddingInputType::Document,
                1,
                vec![
                    (Duration::from_nanos(1), vec![vec![1.0]]),
                    (Duration::from_nanos(2), vec![vec![2.0]]),
                ],
            )
            .is_err()
        );
        Ok(())
    }
}
