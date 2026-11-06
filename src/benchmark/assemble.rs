//! Fail-closed assembly and artifact writing for a complete scoreboard run.

use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};

use super::file_stamp::file_stamp;
use super::{
    FileProvenance, ImplementationKind, ImplementationProvenance, LlamaBenchKernelEvidence,
    LlamaBenchThroughputResult, LlamaServerExecution, MacllmKernelExecution,
    MacllmServingExecution, ParityResult, SCOREBOARD_SCHEMA_VERSION, ScoreboardReport,
    ServerTimingResult, ServerTimingSample, VerifiedScoreboard, file_provenance, first_mismatch,
    render_markdown, run_llama_bench_kernel_evidence, run_llama_server_serving, run_macllm_kernel,
    run_macllm_serving, schema::is_lower_hex, summarize,
};

const EXECUTABLE_HASH_BUFFER_BYTES: usize = 1 << 20;
const BUILD_GIT_COMMIT: &str = env!("MACLLM_BUILD_GIT_COMMIT");
const BUILD_GIT_DIRTY: &str = env!("MACLLM_BUILD_GIT_DIRTY");

/// Pins the running `MacLLM` executable and Git source identity for a long demo.
///
/// The declaration intentionally precedes the generic file guard below: the
/// deterministic coding-agent repository prefix must carry the concrete
/// provenance type that its tasks discuss before its exact-16K boundary. Rust
/// permits the fields to reference the helper types declared later in this
/// module, so this does not change the runtime representation or behavior.
/// The guard keeps the executable open, so replacing `target/release/macllm`
/// during a run cannot make a report silently describe a different binary.
/// Its embedded build commit must also equal the clean runtime source `HEAD`.
/// [`Self::finish`] re-hashes that open file, verifies that the executable path
/// still resolves to the same file, and requires the source commit and clean
/// worktree state to remain unchanged.
#[derive(Debug)]
pub struct MacllmProvenanceGuard {
    executable_file: File,
    executable_identity: OpenFileIdentity,
    initial: ImplementationProvenance,
    repository: PathBuf,
    build_source_identity: BuildGitSourceIdentity,
    source_identity: GitSourceIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OpenFileIdentity {
    device: u64,
    inode: u64,
    size_bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl OpenFileIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Result<Self> {
        ensure!(metadata.is_file(), "guarded path is not a regular file");
        let stamp = file_stamp(metadata);
        Ok(Self {
            device: stamp.device,
            inode: stamp.inode,
            size_bytes: metadata.len(),
            modified_seconds: stamp.modified_seconds,
            modified_nanoseconds: stamp.modified_nanoseconds,
            changed_seconds: stamp.changed_seconds,
            changed_nanoseconds: stamp.changed_nanoseconds,
        })
    }
}

/// Pins one expected regular-file identity across a long-running benchmark.
///
/// Capture and every subsequent verification compare the caller-provided
/// provenance with both an open file descriptor and the current path. The
/// descriptor keeps the original inode observable even if the path is replaced
/// while the benchmark is running.
#[derive(Debug)]
pub struct FileProvenanceGuard {
    guarded_file: File,
    file_identity: OpenFileIdentity,
    initial: FileProvenance,
}

impl FileProvenanceGuard {
    /// Capture and pin an already-verified regular-file provenance.
    pub fn capture(expected: &FileProvenance) -> Result<Self> {
        ensure!(
            is_lower_hex(&expected.sha256, 64),
            "expected file SHA-256 must be a lowercase 64-character digest"
        );
        let canonical_path = fs::canonicalize(&expected.canonical_path).with_context(|| {
            format!(
                "failed to resolve expected file {}",
                expected.canonical_path.display()
            )
        })?;
        ensure!(
            canonical_path == expected.canonical_path,
            "expected file provenance path is not canonical: {}",
            expected.canonical_path.display()
        );

        let path_identity = executable_path_identity(&canonical_path)?;
        let mut guarded_file = File::open(&canonical_path)
            .with_context(|| format!("failed to open guarded file {}", canonical_path.display()))?;
        let (opened, file_identity) =
            provenance_from_open_file(&mut guarded_file, &canonical_path)?;
        ensure!(
            path_identity == file_identity,
            "guarded file path changed while provenance was captured"
        );
        ensure!(
            opened == *expected,
            "guarded file does not match its expected provenance"
        );
        ensure!(
            executable_path_identity(&canonical_path)? == file_identity,
            "guarded file path changed while provenance was captured"
        );

        Ok(Self {
            guarded_file,
            file_identity,
            initial: expected.clone(),
        })
    }

    /// Verify that the held file and current path still name the captured file.
    ///
    /// This is intended to run immediately before and after each process spawn
    /// and after each benchmark phase.
    pub fn verify_path(&mut self) -> Result<()> {
        self.verify_identity()?;
        self.verify_held_file()?;
        self.verify_identity()
    }

    /// Verify the held descriptor and current path identities without
    /// re-reading a potentially multi-gigabyte file. Content is fully rehashed
    /// by [`Self::verify_path`] at the final publication boundary.
    pub(super) fn verify_identity(&self) -> Result<()> {
        let expected_path = self.initial.canonical_path.clone();
        let resolved_path = fs::canonicalize(&expected_path).with_context(|| {
            format!(
                "failed to re-resolve guarded file {}",
                expected_path.display()
            )
        })?;
        ensure!(
            resolved_path == expected_path,
            "guarded file path no longer resolves to its captured canonical path"
        );

        self.verify_held_identity()?;
        ensure!(
            executable_path_identity(&expected_path)? == self.file_identity,
            "guarded file path was replaced or changed during benchmark execution"
        );

        let path_file = File::open(&expected_path).with_context(|| {
            format!("failed to reopen guarded file {}", expected_path.display())
        })?;
        let path_identity = OpenFileIdentity::from_metadata(
            &path_file
                .metadata()
                .context("failed to inspect reopened guarded file")?,
        )?;
        ensure!(
            path_identity == self.file_identity,
            "guarded file path no longer matches its captured identity"
        );
        ensure!(
            executable_path_identity(&expected_path)? == self.file_identity,
            "guarded file path was replaced or changed during benchmark execution"
        );
        self.verify_held_identity()
    }

    /// Read a small file from the already pinned descriptor and verify that
    /// the returned bytes have the captured size and digest. This is used for
    /// metadata whose parsed value must come from exactly the same file
    /// identity recorded in benchmark provenance.
    pub(super) fn read_pinned_bytes(&mut self, limit: usize) -> Result<Vec<u8>> {
        self.verify_held_file()?;
        ensure!(
            self.initial.size_bytes <= u64::try_from(limit)?,
            "guarded file exceeds the pinned-read size limit"
        );
        self.guarded_file
            .seek(SeekFrom::Start(0))
            .context("failed to rewind guarded file")?;
        let mut bytes = Vec::with_capacity(usize::try_from(self.initial.size_bytes)?);
        (&mut self.guarded_file)
            .take(u64::try_from(limit)?.saturating_add(1))
            .read_to_end(&mut bytes)
            .context("failed to read guarded file")?;
        ensure!(
            u64::try_from(bytes.len())? == self.initial.size_bytes
                && format!("{:x}", Sha256::digest(&bytes)) == self.initial.sha256,
            "guarded bytes disagree with captured provenance"
        );
        self.verify_held_file()?;
        Ok(bytes)
    }

    /// Perform one final verification and return the original provenance.
    pub fn finish(mut self) -> Result<FileProvenance> {
        self.verify_path()?;
        Ok(self.initial)
    }

    fn verify_held_file(&mut self) -> Result<()> {
        let (held, identity) =
            provenance_from_open_file(&mut self.guarded_file, &self.initial.canonical_path)?;
        ensure!(
            identity == self.file_identity && held == self.initial,
            "open guarded file changed during benchmark execution"
        );
        Ok(())
    }

    fn verify_held_identity(&self) -> Result<()> {
        ensure!(
            OpenFileIdentity::from_metadata(
                &self
                    .guarded_file
                    .metadata()
                    .context("failed to inspect open guarded file")?,
            )? == self.file_identity,
            "open guarded file identity changed during benchmark execution"
        );
        Ok(())
    }
}

fn executable_path_identity(path: &Path) -> Result<OpenFileIdentity> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect executable {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "executable path is not a regular file: {}",
        path.display()
    );
    OpenFileIdentity::from_metadata(&metadata)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GitSourceIdentity {
    commit: String,
    dirty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BuildGitSourceIdentity {
    commit: String,
    dirty: Option<bool>,
}

impl MacllmProvenanceGuard {
    /// Capture publishable provenance before a prefix-demo execution begins.
    ///
    /// Publication is fail-closed: a dirty source worktree is rejected instead
    /// of being represented only by a warning in the final artifact.
    pub fn capture_publishable() -> Result<Self> {
        let executable = std::env::current_exe().context("failed to resolve MacLLM executable")?;
        Self::capture(
            &executable,
            Path::new(env!("CARGO_MANIFEST_DIR")),
            true,
            embedded_build_source_identity(),
        )
    }

    fn capture(
        executable: &Path,
        repository: &Path,
        publishable: bool,
        build_source_identity: BuildGitSourceIdentity,
    ) -> Result<Self> {
        let repository = fs::canonicalize(repository).with_context(|| {
            format!(
                "failed to resolve MacLLM source repository {}",
                repository.display()
            )
        })?;
        let source_identity = collect_git_source_identity(&repository)?;
        if publishable {
            validate_publishable_build_identity(&build_source_identity, &source_identity)?;
        }

        let canonical_path = fs::canonicalize(executable).with_context(|| {
            format!(
                "failed to resolve MacLLM executable {}",
                executable.display()
            )
        })?;
        let path_metadata = fs::symlink_metadata(&canonical_path).with_context(|| {
            format!(
                "failed to inspect MacLLM executable {}",
                canonical_path.display()
            )
        })?;
        ensure!(
            path_metadata.file_type().is_file() && !path_metadata.file_type().is_symlink(),
            "MacLLM executable path is not a regular file: {}",
            canonical_path.display()
        );
        let mut executable_file = File::open(&canonical_path).with_context(|| {
            format!(
                "failed to open MacLLM executable {}",
                canonical_path.display()
            )
        })?;
        let (executable, executable_identity) =
            provenance_from_open_file(&mut executable_file, &canonical_path)?;
        ensure!(
            OpenFileIdentity::from_metadata(&path_metadata)? == executable_identity,
            "MacLLM executable path changed while provenance was captured"
        );
        let final_path_metadata = fs::symlink_metadata(&canonical_path).with_context(|| {
            format!(
                "failed to re-inspect MacLLM executable {}",
                canonical_path.display()
            )
        })?;
        ensure!(
            OpenFileIdentity::from_metadata(&final_path_metadata)? == executable_identity,
            "MacLLM executable path changed while provenance was captured"
        );

        let initial = ImplementationProvenance {
            implementation: ImplementationKind::Macllm,
            version: env!("CARGO_PKG_VERSION").to_owned(),
            commit: build_source_identity.commit.clone(),
            executable,
            dirty: build_source_identity.dirty,
        };
        Ok(Self {
            executable_file,
            executable_identity,
            initial,
            repository,
            build_source_identity,
            source_identity,
        })
    }

    /// Finish a publishable run and return its initial immutable provenance.
    ///
    /// This consumes the guard so a caller cannot accidentally validate once,
    /// continue measuring, and then publish stale provenance.
    pub fn finish(self) -> Result<ImplementationProvenance> {
        let current_executable =
            std::env::current_exe().context("failed to re-resolve MacLLM executable")?;
        let current_executable = fs::canonicalize(&current_executable).with_context(|| {
            format!(
                "failed to re-resolve MacLLM executable {}",
                current_executable.display()
            )
        })?;
        ensure!(
            current_executable == self.initial.executable.canonical_path,
            "the running MacLLM executable path changed during benchmark execution"
        );
        self.finish_against_path(&current_executable)
    }

    fn finish_against_path(mut self, executable_path: &Path) -> Result<ImplementationProvenance> {
        validate_publishable_build_identity(&self.build_source_identity, &self.source_identity)?;
        ensure!(
            self.initial.commit == self.build_source_identity.commit
                && self.initial.dirty == self.build_source_identity.dirty,
            "publishable MacLLM benchmark provenance lost its build-time source identity"
        );
        let (held_executable, held_identity) = provenance_from_open_file(
            &mut self.executable_file,
            &self.initial.executable.canonical_path,
        )?;
        ensure!(
            held_identity == self.executable_identity && held_executable == self.initial.executable,
            "the open MacLLM executable changed during benchmark execution"
        );

        let path_metadata = fs::symlink_metadata(executable_path).with_context(|| {
            format!(
                "failed to re-inspect MacLLM executable {}",
                executable_path.display()
            )
        })?;
        ensure!(
            path_metadata.file_type().is_file() && !path_metadata.file_type().is_symlink(),
            "MacLLM executable path no longer names a regular file: {}",
            executable_path.display()
        );
        ensure!(
            OpenFileIdentity::from_metadata(&path_metadata)? == self.executable_identity,
            "the MacLLM executable path was replaced during benchmark execution"
        );
        let path_provenance = file_provenance(executable_path)?;
        ensure!(
            path_provenance == self.initial.executable,
            "the MacLLM executable path content changed during benchmark execution"
        );
        let final_path_metadata = fs::symlink_metadata(executable_path).with_context(|| {
            format!(
                "failed to finish inspecting MacLLM executable {}",
                executable_path.display()
            )
        })?;
        ensure!(
            OpenFileIdentity::from_metadata(&final_path_metadata)? == self.executable_identity,
            "the MacLLM executable path was replaced during benchmark execution"
        );

        let final_source_identity = collect_git_source_identity(&self.repository)?;
        ensure!(
            final_source_identity == self.source_identity,
            "the MacLLM source commit or dirty state changed during benchmark execution"
        );
        validate_publishable_build_identity(&self.build_source_identity, &final_source_identity)?;
        Ok(self.initial)
    }
}

fn embedded_build_source_identity() -> BuildGitSourceIdentity {
    BuildGitSourceIdentity {
        commit: BUILD_GIT_COMMIT.to_owned(),
        dirty: match BUILD_GIT_DIRTY {
            "false" => Some(false),
            "true" => Some(true),
            _ => None,
        },
    }
}

fn validate_publishable_build_identity(
    build: &BuildGitSourceIdentity,
    runtime: &GitSourceIdentity,
) -> Result<()> {
    ensure!(
        is_lower_hex(&build.commit, 40),
        "publishable MacLLM benchmark provenance requires a 40-character lowercase build-time Git commit"
    );
    ensure!(
        build.dirty == Some(false),
        "publishable MacLLM benchmark provenance requires a clean build-time source worktree"
    );
    ensure!(
        !runtime.dirty,
        "publishable MacLLM benchmark provenance requires a clean source worktree"
    );
    ensure!(
        build.commit == runtime.commit,
        "the MacLLM build-time Git commit does not match the runtime source HEAD"
    );
    Ok(())
}

fn provenance_from_open_file(
    file: &mut File,
    canonical_path: &Path,
) -> Result<(FileProvenance, OpenFileIdentity)> {
    let before = OpenFileIdentity::from_metadata(
        &file
            .metadata()
            .context("failed to inspect open executable")?,
    )?;
    ensure!(
        before.size_bytes > 0,
        "executable cannot be empty: {}",
        canonical_path.display()
    );
    let first_sha256 = sha256_open_file(file)?;
    let between = OpenFileIdentity::from_metadata(
        &file
            .metadata()
            .context("failed to re-inspect open executable")?,
    )?;
    ensure!(
        before == between,
        "open executable metadata changed while hashing"
    );
    let second_sha256 = sha256_open_file(file)?;
    let after = OpenFileIdentity::from_metadata(
        &file
            .metadata()
            .context("failed to finish inspecting open executable")?,
    )?;
    ensure!(
        between == after && first_sha256 == second_sha256,
        "open executable changed while hashing"
    );
    let filename = canonical_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("executable file name is not valid UTF-8")?
        .to_owned();
    Ok((
        FileProvenance {
            canonical_path: canonical_path.to_owned(),
            filename,
            size_bytes: after.size_bytes,
            sha256: second_sha256,
        },
        after,
    ))
}

fn sha256_open_file(file: &mut File) -> Result<String> {
    file.seek(SeekFrom::Start(0))
        .context("failed to rewind open executable")?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; EXECUTABLE_HASH_BUFFER_BYTES];
    loop {
        let read = file
            .read(&mut buffer)
            .context("failed to hash open executable")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_git_source_identity(repository: &Path) -> Result<GitSourceIdentity> {
    let first_commit = git_stdout(repository, &["rev-parse", "HEAD"])?;
    let first_status = git_status(repository)?;
    let second_commit = git_stdout(repository, &["rev-parse", "HEAD"])?;
    let second_status = git_status(repository)?;
    ensure!(
        first_commit == second_commit && first_status == second_status,
        "MacLLM Git source changed while provenance was captured"
    );
    Ok(GitSourceIdentity {
        commit: first_commit,
        dirty: !first_status.is_empty(),
    })
}

fn git_status(repository: &Path) -> Result<Vec<u8>> {
    let status = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(["status", "--porcelain=v1", "--untracked-files=all"])
        .output()
        .context("failed to inspect MacLLM source worktree")?;
    ensure!(
        status.status.success(),
        "git status failed for MacLLM source"
    );
    Ok(status.stdout)
}

/// Run every truthful evidence stage and validate the assembled report.
///
/// `MacLLM` serving deliberately runs before the in-process kernel benchmark so
/// the report process starts its loaded-steady serving memory boundary before
/// any second model load or kernel-lifetime peak can contaminate it.
pub async fn run_complete_scoreboard(
    verified: &VerifiedScoreboard,
    llama_server_port: u16,
) -> Result<ScoreboardReport> {
    let macllm_serving = run_macllm_serving(verified)
        .await
        .context("MacLLM serving and memory evidence failed")?;
    let macllm_kernel =
        run_macllm_kernel(verified).context("MacLLM kernel and parity evidence failed")?;
    let llama_bench = run_llama_bench_kernel_evidence(verified)
        .context("llama-bench throughput-only evidence failed")?;
    let llama_server = run_llama_server_serving(verified, llama_server_port)
        .context("llama-server serving, timing, memory, or parity evidence failed")?;
    assemble_scoreboard_report(
        verified,
        macllm_kernel,
        macllm_serving,
        llama_bench,
        llama_server,
    )
}

/// Normalize independently measured executions into the publication schema.
#[allow(clippy::too_many_lines)]
pub fn assemble_scoreboard_report(
    verified: &VerifiedScoreboard,
    macllm_kernel: MacllmKernelExecution,
    macllm_serving: MacllmServingExecution,
    llama_bench: LlamaBenchKernelEvidence,
    llama_server: LlamaServerExecution,
) -> Result<ScoreboardReport> {
    verified.manifest.validate()?;
    ensure!(
        macllm_kernel.result.implementation == ImplementationKind::Macllm,
        "MacLLM kernel evidence has the wrong implementation"
    );
    ensure!(
        llama_bench.implementation == ImplementationKind::LlamaCpp,
        "llama-bench evidence has the wrong implementation"
    );
    ensure!(
        llama_server.implementation == ImplementationKind::LlamaCpp,
        "llama-server evidence has the wrong implementation"
    );

    let llama_bench_throughput = LlamaBenchThroughputResult {
        implementation: llama_bench.implementation,
        measurement_boundary: llama_bench.measurement_boundary,
        command: llama_bench.command,
        prompt_tokens: llama_bench.prompt_tokens,
        decode_steps: llama_bench.decode_steps,
        prefill_tokens_per_second: summarize(&llama_bench.prefill_tokens_per_second_samples)?,
        decode_tokens_per_second: summarize(&llama_bench.decode_tokens_per_second_samples)?,
        prefill_tokens_per_second_samples: llama_bench.prefill_tokens_per_second_samples,
        decode_tokens_per_second_samples: llama_bench.decode_tokens_per_second_samples,
    };

    let mut serving_results = macllm_serving.serving_results;
    let mut server_timing_results = Vec::with_capacity(llama_server.widths.len());
    for width in llama_server.widths {
        let concurrency = width.serving.concurrency;
        serving_results.push(width.serving);
        server_timing_results.push(ServerTimingResult {
            implementation: ImplementationKind::LlamaCpp,
            concurrency,
            measurement_boundary: concat!(
                "terminal llama-server SSE timings for the same externally measured requests; ",
                "prompt_* is server prompt processing and predicted_* is server serving-generation ",
                "timing; predicted_* is not a pure single-token decode-forward KernelResult"
            )
            .to_owned(),
            command: llama_server.command.clone(),
            prompt_ms: width.server_prompt_ms,
            prompt_tokens_per_second: width.server_prompt_tokens_per_second,
            predicted_ms: width.server_predicted_ms,
            predicted_tokens_per_second: width.server_predicted_tokens_per_second,
            samples: width
                .server_samples
                .into_iter()
                .map(|sample| ServerTimingSample {
                    wave: sample.wave,
                    request: sample.request,
                    token_ids: sample.token_ids,
                    cache_tokens: sample.cache_tokens,
                    prompt_tokens: sample.prompt_tokens,
                    prompt_ms: sample.prompt_ms,
                    prompt_tokens_per_second: sample.prompt_tokens_per_second,
                    predicted_tokens: sample.predicted_tokens,
                    predicted_ms: sample.predicted_ms,
                    predicted_tokens_per_second: sample.predicted_tokens_per_second,
                })
                .collect(),
        });
    }

    let mut memory_results = macllm_serving.memory_results;
    memory_results.extend(llama_server.memory_results);
    let parity = ParityResult {
        prompt_tokens: verified.prompt_token_ids.len(),
        prompt_token_ids_sha256: verified.manifest.workload.prompt.token_ids_sha256.clone(),
        exact: macllm_kernel.parity_token_ids == llama_server.parity_token_ids,
        first_mismatch: first_mismatch(
            &macllm_kernel.parity_token_ids,
            &llama_server.parity_token_ids,
        ),
        macllm_token_ids: macllm_kernel.parity_token_ids,
        llama_cpp_token_ids: llama_server.parity_token_ids,
    };

    let (recorded_at_utc, run_id) = run_identity(&verified.manifest.name)?;
    let (macllm_provenance, macllm_dirty) = macllm_implementation_provenance()?;
    let llama_cpp_provenance = ImplementationProvenance {
        implementation: ImplementationKind::LlamaCpp,
        version: format!("llama.cpp build {}", verified.llama_cpp_version.build),
        commit: verified.llama_cpp_commit.clone(),
        executable: verified.llama_server.clone(),
        dirty: None,
    };
    let mut warnings = vec![
        concat!(
            "llama-bench uses synthetic token IDs and emits no per-request TTFT; its pp/tg ",
            "evidence is published only as llama_bench_throughput, not as a complete KernelResult."
        )
        .to_owned(),
        concat!(
            "llama-server predicted timings are terminal serving-generation timings and are ",
            "published only as server_timing_results, not as pure decode-forward kernel evidence."
        )
        .to_owned(),
        concat!(
            "Memory peaks are fixed-period current-state proc_pid_rusage samples; macOS lifetime ",
            "maximum values are excluded so earlier widths cannot be relabeled as later peaks."
        )
        .to_owned(),
    ];
    if macllm_dirty {
        warnings.push(
            "The MacLLM source worktree was dirty when the executable provenance was collected."
                .to_owned(),
        );
    }
    if !parity.exact {
        warnings.push(format!(
            "Greedy token parity failed at token index {:?}.",
            parity.first_mismatch
        ));
    }

    let report = ScoreboardReport {
        schema_version: SCOREBOARD_SCHEMA_VERSION,
        run_id,
        recorded_at_utc,
        host: verified.host.clone(),
        model: verified.model.clone(),
        implementations: vec![macllm_provenance, llama_cpp_provenance],
        workload: verified.manifest.workload.clone(),
        kernel_results: vec![macllm_kernel.result],
        llama_bench_throughput,
        serving_results,
        server_timing_results,
        memory_results,
        parity,
        warnings,
    };
    report.validate()?;
    Ok(report)
}

/// Serialize validated JSON and Markdown to two explicit artifact paths.
pub fn write_scoreboard_artifacts(
    report: &ScoreboardReport,
    json_path: impl AsRef<Path>,
    markdown_path: impl AsRef<Path>,
) -> Result<()> {
    report.validate()?;
    let json_path = json_path.as_ref();
    let markdown_path = markdown_path.as_ref();
    ensure!(
        json_path != markdown_path,
        "scoreboard JSON and Markdown output paths must differ"
    );
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    let markdown = render_markdown(report)?;

    let json_temp = prepare_atomic_file(json_path, &json)?;
    let markdown_temp = prepare_atomic_file(markdown_path, markdown.as_bytes())?;
    json_temp
        .persist(json_path)
        .with_context(|| format!("failed to persist scoreboard JSON {}", json_path.display()))?;
    markdown_temp.persist(markdown_path).with_context(|| {
        format!(
            "failed to persist scoreboard Markdown {}",
            markdown_path.display()
        )
    })?;
    Ok(())
}

fn prepare_atomic_file(path: &Path, bytes: &[u8]) -> Result<tempfile::NamedTempFile> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    ensure!(
        parent.is_dir(),
        "scoreboard output directory does not exist: {}",
        parent.display()
    );
    let mut temporary = tempfile::NamedTempFile::new_in(parent).with_context(|| {
        format!(
            "failed to create temporary scoreboard artifact in {}",
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

pub(super) fn macllm_implementation_provenance() -> Result<(ImplementationProvenance, bool)> {
    let executable = std::env::current_exe().context("failed to resolve MacLLM executable")?;
    let executable = file_provenance(executable)?;
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
    let build = embedded_build_source_identity();
    let runtime = collect_git_source_identity(repository)?;
    ensure!(
        is_lower_hex(&build.commit, 40),
        "MacLLM executable has no valid build-time Git commit provenance"
    );
    let build_dirty = build
        .dirty
        .context("MacLLM executable has no build-time Git dirty-state provenance")?;
    ensure!(
        build.commit == runtime.commit,
        "the MacLLM build-time Git commit does not match the runtime source HEAD"
    );
    let dirty = build_dirty || runtime.dirty;
    Ok((
        ImplementationProvenance {
            implementation: ImplementationKind::Macllm,
            version: env!("CARGO_PKG_VERSION").to_owned(),
            commit: build.commit,
            executable,
            dirty: Some(dirty),
        },
        dirty,
    ))
}

fn git_stdout(repository: &Path, arguments: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .context("failed to execute git for MacLLM provenance")?;
    ensure!(output.status.success(), "git provenance command failed");
    let value = std::str::from_utf8(&output.stdout)
        .context("git provenance output was not UTF-8")?
        .trim();
    ensure!(!value.is_empty(), "git provenance output was empty");
    Ok(value.to_owned())
}

fn run_identity(profile: &str) -> Result<(String, String)> {
    // The timestamp is formatted in-process so that every host reports the
    // same UTC text without depending on an external `date` command.
    let recorded_at_utc = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    ensure!(!recorded_at_utc.is_empty(), "UTC date output was empty");
    let epoch_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time precedes the Unix epoch")?
        .as_millis();
    Ok((recorded_at_utc, format!("{profile}-{epoch_millis}")))
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::*;

    fn run_git_for_test(repository: &Path, arguments: &[&str]) -> Result<()> {
        let output = Command::new("git")
            .arg("-C")
            .arg(repository)
            .args(arguments)
            .output()
            .context("failed to execute fixture git command")?;
        ensure!(
            output.status.success(),
            "fixture git command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(())
    }

    fn clean_provenance_fixture() -> Result<(tempfile::TempDir, PathBuf, PathBuf)> {
        let root = tempfile::tempdir()?;
        let repository = root.path().join("repository");
        fs::create_dir(&repository)?;
        run_git_for_test(&repository, &["init", "--quiet"])?;
        run_git_for_test(&repository, &["config", "user.name", "MacLLM Test"])?;
        run_git_for_test(
            &repository,
            &["config", "user.email", "macllm-test@example.invalid"],
        )?;
        run_git_for_test(&repository, &["config", "commit.gpgsign", "false"])?;
        fs::write(repository.join("source.rs"), b"pub fn source() {}\n")?;
        run_git_for_test(&repository, &["add", "source.rs"])?;
        run_git_for_test(&repository, &["commit", "--quiet", "-m", "initial"])?;

        let executable = root.path().join("macllm");
        fs::write(&executable, b"immutable macllm executable fixture\n")?;
        Ok((root, repository, executable))
    }

    fn build_identity_for_repository(repository: &Path) -> Result<BuildGitSourceIdentity> {
        let source = collect_git_source_identity(repository)?;
        Ok(BuildGitSourceIdentity {
            commit: source.commit,
            dirty: Some(source.dirty),
        })
    }

    #[test]
    fn atomic_outputs_require_distinct_existing_destinations() {
        let missing = PathBuf::from("/definitely/missing/scoreboard.json");
        assert!(prepare_atomic_file(&missing, b"{}").is_err());
    }

    #[test]
    fn current_executable_and_git_source_have_publishable_provenance() -> Result<()> {
        let (implementation, _dirty) = macllm_implementation_provenance()?;
        assert_eq!(implementation.implementation, ImplementationKind::Macllm);
        assert_eq!(implementation.commit.len(), 40);
        assert!(implementation.executable.size_bytes > 0);
        Ok(())
    }

    #[test]
    fn provenance_guard_returns_the_initial_clean_identity() -> Result<()> {
        let (_root, repository, executable) = clean_provenance_fixture()?;
        let expected_executable = file_provenance(&executable)?;
        let expected_commit = git_stdout(&repository, &["rev-parse", "HEAD"])?;
        let build = build_identity_for_repository(&repository)?;
        let guard = MacllmProvenanceGuard::capture(&executable, &repository, true, build)?;

        let provenance = guard.finish_against_path(&expected_executable.canonical_path)?;

        assert_eq!(provenance.executable, expected_executable);
        assert_eq!(provenance.commit, expected_commit);
        assert_eq!(provenance.dirty, Some(false));
        Ok(())
    }

    #[test]
    fn file_provenance_guard_returns_the_captured_identity() -> Result<()> {
        let root = tempfile::tempdir()?;
        let executable = root.path().join("llama-server");
        fs::write(&executable, b"immutable executable fixture\n")?;
        let expected = file_provenance(&executable)?;
        let mut guard = FileProvenanceGuard::capture(&expected)?;

        guard.verify_path()?;
        assert_eq!(guard.finish()?, expected);
        Ok(())
    }

    #[test]
    fn file_provenance_guard_rejects_path_replacement() -> Result<()> {
        let root = tempfile::tempdir()?;
        let executable = root.path().join("llama-server");
        fs::write(&executable, b"immutable executable fixture\n")?;
        let expected = file_provenance(&executable)?;
        let mut guard = FileProvenanceGuard::capture(&expected)?;
        let replacement = root.path().join("replacement");
        fs::write(&replacement, b"replacement executable fixture\n")?;
        fs::rename(&replacement, &executable)?;

        let error = guard
            .verify_path()
            .expect_err("replaced executable path must be rejected");

        assert!(error.to_string().contains("file"));
        Ok(())
    }

    #[test]
    fn publishable_provenance_rejects_a_dirty_source_at_capture() -> Result<()> {
        let (_root, repository, executable) = clean_provenance_fixture()?;
        let build = build_identity_for_repository(&repository)?;
        fs::write(
            repository.join("source.rs"),
            b"changed during preparation\n",
        )?;

        let error = MacllmProvenanceGuard::capture(&executable, &repository, true, build)
            .expect_err("dirty source must be rejected");

        assert!(
            error
                .to_string()
                .contains("requires a clean source worktree")
        );
        Ok(())
    }

    #[test]
    fn publishable_provenance_rejects_a_dirty_build_identity() -> Result<()> {
        let (_root, repository, executable) = clean_provenance_fixture()?;
        let mut build = build_identity_for_repository(&repository)?;
        build.dirty = Some(true);

        let error = MacllmProvenanceGuard::capture(&executable, &repository, true, build)
            .expect_err("dirty build-time source must be rejected");

        assert!(
            error
                .to_string()
                .contains("clean build-time source worktree")
        );
        Ok(())
    }

    #[test]
    fn publishable_provenance_rejects_an_unknown_build_commit() -> Result<()> {
        let (_root, repository, executable) = clean_provenance_fixture()?;
        let mut build = build_identity_for_repository(&repository)?;
        build.commit = "unknown".to_owned();

        let error = MacllmProvenanceGuard::capture(&executable, &repository, true, build)
            .expect_err("unknown build-time source commit must be rejected");

        assert!(error.to_string().contains("40-character lowercase"));
        Ok(())
    }

    #[test]
    fn publishable_provenance_rejects_a_build_commit_mismatch() -> Result<()> {
        let (_root, repository, executable) = clean_provenance_fixture()?;
        let mut build = build_identity_for_repository(&repository)?;
        build.commit = if build.commit.bytes().all(|byte| byte == b'0') {
            "1111111111111111111111111111111111111111".to_owned()
        } else {
            "0000000000000000000000000000000000000000".to_owned()
        };

        let error = MacllmProvenanceGuard::capture(&executable, &repository, true, build)
            .expect_err("a stale binary commit must be rejected");

        assert!(
            error
                .to_string()
                .contains("does not match the runtime source HEAD")
        );
        Ok(())
    }

    #[test]
    fn provenance_guard_rejects_executable_path_replacement() -> Result<()> {
        let (_root, repository, executable) = clean_provenance_fixture()?;
        let build = build_identity_for_repository(&repository)?;
        let guard = MacllmProvenanceGuard::capture(&executable, &repository, true, build)?;
        let replacement = executable.with_extension("replacement");
        fs::write(&replacement, b"replacement macllm executable fixture\n")?;
        fs::rename(replacement, &executable)?;

        let error = guard
            .finish_against_path(&executable)
            .expect_err("replaced executable path must be rejected");

        assert!(error.to_string().contains("MacLLM executable"));
        Ok(())
    }

    #[test]
    fn provenance_guard_rejects_a_source_commit_change() -> Result<()> {
        let (_root, repository, executable) = clean_provenance_fixture()?;
        let build = build_identity_for_repository(&repository)?;
        let guard = MacllmProvenanceGuard::capture(&executable, &repository, true, build)?;
        fs::write(
            repository.join("source.rs"),
            b"pub fn changed_source() {}\n",
        )?;
        run_git_for_test(&repository, &["add", "source.rs"])?;
        run_git_for_test(&repository, &["commit", "--quiet", "-m", "changed"])?;

        let error = guard
            .finish_against_path(&executable)
            .expect_err("changed source commit must be rejected");

        assert!(
            error
                .to_string()
                .contains("source commit or dirty state changed")
        );
        Ok(())
    }

    #[test]
    fn provenance_guard_rejects_a_worktree_that_becomes_dirty() -> Result<()> {
        let (_root, repository, executable) = clean_provenance_fixture()?;
        let build = build_identity_for_repository(&repository)?;
        let guard = MacllmProvenanceGuard::capture(&executable, &repository, true, build)?;
        fs::write(repository.join("source.rs"), b"uncommitted source change\n")?;

        let error = guard
            .finish_against_path(&executable)
            .expect_err("dirty state change must be rejected");

        assert!(
            error
                .to_string()
                .contains("source commit or dirty state changed")
        );
        Ok(())
    }

    #[test]
    fn run_identity_is_nonempty_and_utc() -> Result<()> {
        let (timestamp, run_id) = run_identity("profile")?;
        assert!(timestamp.ends_with('Z'));
        assert!(run_id.starts_with("profile-"));
        Ok(())
    }
}
