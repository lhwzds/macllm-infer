//! Manifest loading and fail-closed scoreboard preflight.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::weights::ModelFiles;

use super::{
    FileProvenance, LlamaCppVersion, ModelProvenance, SanitizedHost, ScoreboardManifest,
    collect_llama_cpp_version, collect_macos_host, file_provenance, parse_homebrew_scm_revision,
    token_ids_sha256, verify_model_provenance, verify_prompt_file,
};

/// All immutable identities required before a benchmark may execute.
#[derive(Debug)]
pub struct VerifiedScoreboard {
    pub manifest_path: PathBuf,
    pub manifest: ScoreboardManifest,
    pub host: SanitizedHost,
    pub model: ModelProvenance,
    pub prompt: FileProvenance,
    pub prompt_text: String,
    pub prompt_token_ids: Vec<u32>,
    pub llama_cpp_version: LlamaCppVersion,
    pub llama_cpp_commit: String,
    pub llama_bench: FileProvenance,
    pub llama_server: FileProvenance,
}

/// Serializable, privacy-safe summary printed by `scoreboard verify`.
#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreboardVerificationSummary<'a> {
    pub manifest: &'a Path,
    pub profile: &'a str,
    pub host: &'a SanitizedHost,
    pub model: &'a ModelProvenance,
    pub prompt: &'a FileProvenance,
    pub prompt_tokens: usize,
    pub prompt_token_ids_sha256: String,
    pub llama_cpp_version: &'a LlamaCppVersion,
    pub llama_cpp_commit: &'a str,
    pub llama_bench: &'a FileProvenance,
    pub llama_server: &'a FileProvenance,
}

impl VerifiedScoreboard {
    #[must_use]
    pub fn summary(&self) -> ScoreboardVerificationSummary<'_> {
        ScoreboardVerificationSummary {
            manifest: &self.manifest_path,
            profile: &self.manifest.name,
            host: &self.host,
            model: &self.model,
            prompt: &self.prompt,
            prompt_tokens: self.prompt_token_ids.len(),
            prompt_token_ids_sha256: token_ids_sha256(&self.prompt_token_ids),
            llama_cpp_version: &self.llama_cpp_version,
            llama_cpp_commit: &self.llama_cpp_commit,
            llama_bench: &self.llama_bench,
            llama_server: &self.llama_server,
        }
    }
}

/// Load a manifest, resolve its paths relative to the manifest, and verify all
/// immutable local inputs without loading model weights or running a benchmark.
pub fn verify_scoreboard_manifest(path: impl AsRef<Path>) -> Result<VerifiedScoreboard> {
    let requested = path.as_ref();
    let manifest_path = fs::canonicalize(requested).with_context(|| {
        format!(
            "failed to resolve scoreboard manifest {}",
            requested.display()
        )
    })?;
    let bytes = fs::read(&manifest_path).with_context(|| {
        format!(
            "failed to read scoreboard manifest {}",
            manifest_path.display()
        )
    })?;
    let mut manifest: ScoreboardManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid scoreboard manifest {}", manifest_path.display()))?;
    let base = manifest_path
        .parent()
        .context("scoreboard manifest has no parent directory")?;
    resolve_manifest_paths(&mut manifest, base);
    manifest.validate()?;

    let host = collect_macos_host()?;
    manifest.host.verify(&host)?;
    let model = verify_model_provenance(&manifest.model)?;
    let prompt = verify_prompt_file(&manifest.workload.prompt)?;
    let prompt_file_text = fs::read_to_string(&prompt.canonical_path).with_context(|| {
        format!(
            "benchmark prompt is not UTF-8: {}",
            prompt.canonical_path.display()
        )
    })?;
    let prompt_text = manifest
        .workload
        .prompt
        .prompt_text(prompt_file_text)
        .context("failed to apply the declared prompt text transform")?;

    let files = ModelFiles::from_dir(&manifest.model.path)?;
    let adapter = files.adapter()?;
    adapter.validate()?;
    ensure!(
        adapter.family() == manifest.model.architecture,
        "model architecture mismatch: manifest expects {}, runtime detected {}",
        manifest.model.architecture,
        adapter.family()
    );
    ensure!(
        files.weight_file_count() == 1,
        "scoreboard v1 requires one exact GGUF file; found {} shards",
        files.weight_file_count()
    );
    let tokenizer = files.tokenizer()?;
    let encoding = tokenizer
        .encode(
            prompt_text.clone(),
            manifest.workload.prompt.add_special_tokens,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let prompt_token_ids = encoding.get_ids().to_vec();
    ensure!(
        prompt_token_ids.len() == manifest.workload.prompt.token_count,
        "prompt token count mismatch: expected {}, found {}",
        manifest.workload.prompt.token_count,
        prompt_token_ids.len()
    );
    let actual_token_digest = token_ids_sha256(&prompt_token_ids);
    ensure!(
        actual_token_digest == manifest.workload.prompt.token_ids_sha256,
        "prompt token-ID SHA-256 mismatch: expected {}, found {actual_token_digest}",
        manifest.workload.prompt.token_ids_sha256
    );

    let llama_server_path = manifest.llama_cpp.binary_directory.join("llama-server");
    let llama_bench_path = manifest.llama_cpp.binary_directory.join("llama-bench");
    let llama_server = file_provenance(&llama_server_path)?;
    let llama_bench = file_provenance(&llama_bench_path)?;
    if manifest.llama_cpp.source_directory.is_some() {
        ensure!(
            installation_directory_for_binary(&llama_server.canonical_path)?
                == installation_directory_for_binary(&llama_bench.canonical_path)?,
            "local llama.cpp server and bench binaries are not from one CMake build directory"
        );
    }
    let llama_cpp_version = collect_llama_cpp_version(&llama_server.canonical_path)?;
    let llama_cpp_commit = attest_llama_cpp_commit(
        &llama_server.canonical_path,
        manifest.llama_cpp.source_directory.as_deref(),
        manifest.runtime.device == super::BenchmarkDevice::Metal,
    )?;
    llama_cpp_version.verify(&manifest.llama_cpp, &llama_cpp_commit)?;

    Ok(VerifiedScoreboard {
        manifest_path,
        manifest,
        host,
        model,
        prompt,
        prompt_text,
        prompt_token_ids,
        llama_cpp_version,
        llama_cpp_commit,
        llama_bench,
        llama_server,
    })
}

fn resolve_manifest_paths(manifest: &mut ScoreboardManifest, base: &Path) {
    if manifest.model.path.is_relative() {
        manifest.model.path = base.join(&manifest.model.path);
    }
    if manifest.workload.prompt.path.is_relative() {
        manifest.workload.prompt.path = base.join(&manifest.workload.prompt.path);
    }
    if manifest.llama_cpp.binary_directory.is_relative() {
        manifest.llama_cpp.binary_directory = base.join(&manifest.llama_cpp.binary_directory);
    }
    if let Some(source_directory) = &mut manifest.llama_cpp.source_directory
        && source_directory.is_relative()
    {
        *source_directory = base.join(&*source_directory);
    }
}

fn attest_llama_cpp_commit(
    binary: &Path,
    source_directory: Option<&Path>,
    require_metal: bool,
) -> Result<String> {
    if let Some(source_directory) = source_directory {
        return attest_local_cmake_source(binary, source_directory, require_metal);
    }
    let receipt = homebrew_receipt_for_binary(binary)?;
    let receipt_bytes = fs::read(&receipt)
        .with_context(|| format!("failed to read Homebrew receipt {}", receipt.display()))?;
    parse_homebrew_scm_revision(&receipt_bytes)
}

fn homebrew_receipt_for_binary(binary: &Path) -> Result<PathBuf> {
    let installation = installation_directory_for_binary(binary)?;
    let receipt = installation.join("INSTALL_RECEIPT.json");
    ensure!(
        receipt.is_file(),
        "llama.cpp full commit attestation is missing: {}",
        receipt.display()
    );
    Ok(receipt)
}

fn attest_local_cmake_source(
    binary: &Path,
    source_directory: &Path,
    require_metal: bool,
) -> Result<String> {
    let source_directory = fs::canonicalize(source_directory).with_context(|| {
        format!(
            "failed to resolve declared local llama.cpp source directory {}",
            source_directory.display()
        )
    })?;
    ensure!(
        source_directory.is_dir(),
        "declared local llama.cpp source is not a directory: {}",
        source_directory.display()
    );
    let installation = installation_directory_for_binary(binary)?;
    let cache_path = installation.join("CMakeCache.txt");
    let cache = fs::read_to_string(&cache_path).with_context(|| {
        format!(
            "failed to read local llama.cpp CMake cache {}",
            cache_path.display()
        )
    })?;
    validate_local_cmake_options(&cache, require_metal)?;
    let configured_source = cmake_home_directory(&cache)?;
    let configured_source = fs::canonicalize(&configured_source).with_context(|| {
        format!(
            "failed to resolve CMake-recorded llama.cpp source directory {}",
            configured_source.display()
        )
    })?;
    ensure!(
        configured_source == source_directory,
        "local llama.cpp binary was configured from {}, not the manifest-declared source {}",
        configured_source.display(),
        source_directory.display()
    );

    let git_root = local_git_output(&source_directory, &["rev-parse", "--show-toplevel"])?;
    let git_root =
        fs::canonicalize(git_root.trim()).context("failed to resolve local llama.cpp Git root")?;
    ensure!(
        git_root == source_directory,
        "manifest local llama.cpp source must be its Git checkout root"
    );
    let status = local_git_output(
        &source_directory,
        &[
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
    )?;
    ensure!(
        status.trim().is_empty(),
        "manifest local llama.cpp source checkout is dirty"
    );
    let commit = local_git_output(&source_directory, &["rev-parse", "HEAD"])?;
    let commit = commit.trim();
    ensure!(
        super::schema::is_lower_hex(commit, 40),
        "local llama.cpp Git HEAD is not a full lowercase commit hash"
    );
    Ok(commit.to_owned())
}

fn installation_directory_for_binary(binary: &Path) -> Result<PathBuf> {
    let canonical = fs::canonicalize(binary)
        .with_context(|| format!("failed to resolve llama.cpp binary {}", binary.display()))?;
    let bin_directory = canonical
        .parent()
        .context("llama.cpp binary has no parent directory")?;
    bin_directory
        .parent()
        .map(Path::to_path_buf)
        .context("llama.cpp binary is not inside an installation directory")
}

fn cmake_home_directory(cache: &str) -> Result<PathBuf> {
    let value = cmake_cache_value(cache, "CMAKE_HOME_DIRECTORY", "INTERNAL")?;
    ensure!(
        !value.trim().is_empty(),
        "local llama.cpp CMake cache has an empty CMAKE_HOME_DIRECTORY"
    );
    Ok(PathBuf::from(value))
}

fn validate_local_cmake_options(cache: &str, require_metal: bool) -> Result<()> {
    ensure!(
        cmake_cache_value(cache, "CMAKE_BUILD_TYPE", "STRING")? == "Release",
        "local llama.cpp CMake build type must be Release"
    );
    if require_metal {
        ensure!(
            cmake_cache_value(cache, "GGML_METAL", "BOOL")? == "ON",
            "local llama.cpp CMake build does not enable GGML_METAL for a Metal benchmark"
        );
    }
    Ok(())
}

fn cmake_cache_value<'a>(cache: &'a str, key: &str, value_type: &str) -> Result<&'a str> {
    let prefix = format!("{key}:{value_type}=");
    let mut values = cache.lines().filter_map(|line| line.strip_prefix(&prefix));
    let value = values
        .next()
        .with_context(|| format!("local llama.cpp CMake cache has no {key}:{value_type}"))?;
    ensure!(
        values.next().is_none(),
        "local llama.cpp CMake cache has duplicate {key}:{value_type} entries"
    );
    Ok(value)
}

fn local_git_output(source_directory: &Path, arguments: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(source_directory)
        .args(arguments)
        .output()
        .with_context(|| format!("failed to execute Git in {}", source_directory.display()))?;
    ensure!(
        output.status.success(),
        "Git command failed in {}: {}",
        source_directory.display(),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    String::from_utf8(output.stdout).context("Git command stdout was not UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_git(directory: &Path, arguments: &[&str]) -> Result<String> {
        let output = Command::new("git")
            .arg("-C")
            .arg(directory)
            .args(arguments)
            .output()?;
        ensure!(
            output.status.success(),
            "test Git command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).context("test Git stdout was not UTF-8")
    }

    #[test]
    fn relative_manifest_paths_are_resolved_from_the_manifest_directory() {
        let mut manifest: ScoreboardManifest = serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "name": "test",
            "host": {"machine_model":"Mac1,1","chip":"Apple Test","memory_bytes":1,"architecture":"aarch64"},
            "model": {"path":"../model","repository":"o/r","revision":"0000000000000000000000000000000000000000","filename":"m.gguf","size_bytes":1,"sha256":"0000000000000000000000000000000000000000000000000000000000000000","architecture":"test","quantization":"Q4"},
            "llama_cpp": {"binary_directory":"../bin","source_directory":"../source","build":1,"commit":"0000000000000000000000000000000000000000","gpu_layers":0,"flash_attention":"off","cache_type_k":"f16","cache_type_v":"f16","mmap":true},
            "workload": {"prompt":{"path":"prompt.txt","sha256":"0000000000000000000000000000000000000000000000000000000000000000","token_count":1,"token_ids_sha256":"0000000000000000000000000000000000000000000000000000000000000000","add_special_tokens":false},"decode_steps":1,"sampling":{"temperature":0.0,"top_k":0,"top_p":1.0,"repetition_penalty":1.0,"seed":42,"ignore_stop_tokens":true},"concurrency":[1,4,8],"warmup_waves":1,"samples_per_concurrency":40,"prefix_cache":"disabled"},
            "runtime": {"device":"cpu","dtype":"f32","prefill_token_quantum":128,"kv_cache_block_tokens":16,"kv_cache_slab_blocks":8,"kv_cache_capacity_tokens":4096}
        })).unwrap();
        resolve_manifest_paths(&mut manifest, Path::new("/tmp/profile"));
        assert_eq!(manifest.model.path, Path::new("/tmp/profile/../model"));
        assert_eq!(
            manifest.workload.prompt.path,
            Path::new("/tmp/profile/prompt.txt")
        );
        assert_eq!(
            manifest.llama_cpp.binary_directory,
            Path::new("/tmp/profile/../bin")
        );
        assert_eq!(
            manifest.llama_cpp.source_directory.as_deref(),
            Some(Path::new("/tmp/profile/../source"))
        );
    }

    #[test]
    fn cmake_cache_requires_a_nonempty_source_root() {
        assert_eq!(
            cmake_home_directory("CMAKE_HOME_DIRECTORY:INTERNAL=/tmp/llama.cpp\n").unwrap(),
            PathBuf::from("/tmp/llama.cpp")
        );
        assert!(cmake_home_directory("OTHER:INTERNAL=/tmp/llama.cpp\n").is_err());
        assert!(cmake_home_directory("CMAKE_HOME_DIRECTORY:INTERNAL=\n").is_err());
    }

    #[test]
    fn local_cmake_options_require_exact_release_and_metal_entries() {
        let cache = "CMAKE_BUILD_TYPE:STRING=Release\nGGML_METAL:BOOL=ON\n";
        validate_local_cmake_options(cache, true).unwrap();
        validate_local_cmake_options(cache, false).unwrap();
        assert!(
            validate_local_cmake_options(
                "# CMAKE_BUILD_TYPE:STRING=Release\nGGML_METAL:BOOL=ON\n",
                true
            )
            .is_err()
        );
        assert!(
            validate_local_cmake_options(
                "CMAKE_BUILD_TYPE:STRING=Release\nGGML_METAL:BOOL=OFF\n",
                true
            )
            .is_err()
        );
        assert!(
            validate_local_cmake_options(
                "CMAKE_BUILD_TYPE:STRING=Debug\nGGML_METAL:BOOL=ON\n",
                false
            )
            .is_err()
        );
    }

    #[test]
    fn local_cmake_attestation_requires_the_declared_clean_git_root() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let source = temp.path().join("llama.cpp");
        fs::create_dir(&source)?;
        test_git(temp.path(), &["init", "-q", source.to_str().unwrap()])?;
        test_git(
            &source,
            &["config", "user.email", "benchmark@example.invalid"],
        )?;
        test_git(&source, &["config", "user.name", "Benchmark Test"])?;
        fs::write(source.join("CMakeLists.txt"), "project(llama_fixture)\n")?;
        test_git(&source, &["add", "CMakeLists.txt"])?;
        test_git(&source, &["commit", "-qm", "fixture"])?;
        let expected_commit = test_git(&source, &["rev-parse", "HEAD"])?;

        let build = temp.path().join("build");
        let binary_directory = build.join("bin");
        fs::create_dir_all(&binary_directory)?;
        let source = fs::canonicalize(source)?;
        fs::write(
            build.join("CMakeCache.txt"),
            format!(
                "CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_HOME_DIRECTORY:INTERNAL={}\n",
                source.display()
            ),
        )?;
        let server = binary_directory.join("llama-server");
        fs::write(&server, "fixture")?;

        assert_eq!(
            attest_local_cmake_source(&server, &source, false)?,
            expected_commit.trim()
        );
        fs::write(
            build.join("CMakeCache.txt"),
            "CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_HOME_DIRECTORY:INTERNAL=/not/the/source\n",
        )?;
        assert!(attest_local_cmake_source(&server, &source, false).is_err());
        fs::write(
            build.join("CMakeCache.txt"),
            format!(
                "CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_HOME_DIRECTORY:INTERNAL={}\n",
                source.display()
            ),
        )?;
        fs::write(source.join("untracked.txt"), "dirty")?;
        assert!(attest_local_cmake_source(&server, &source, false).is_err());
        Ok(())
    }
}
