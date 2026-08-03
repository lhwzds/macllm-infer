//! Cryptographic file, prompt-token, and model-snapshot provenance.

use std::{
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::schema::{
    FileProvenance, ModelProvenance, ModelRequirement, PromptRequirement, SnapshotProvenance,
    is_lower_hex,
};

const SNAPSHOT_MARKER: &str = ".macllm-revision";
const HASH_BUFFER_BYTES: usize = 1 << 20;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotMarker {
    repo_id: String,
    revision: String,
    #[serde(default)]
    requested_revision: Option<String>,
    complete: bool,
}

/// Resolve a path to one regular file and compute its stable SHA-256 identity.
pub fn file_provenance(path: impl AsRef<Path>) -> Result<FileProvenance> {
    let requested = path.as_ref();
    let canonical_path = fs::canonicalize(requested)
        .with_context(|| format!("failed to resolve benchmark file {}", requested.display()))?;
    let metadata = fs::symlink_metadata(&canonical_path).with_context(|| {
        format!(
            "failed to inspect benchmark file {}",
            canonical_path.display()
        )
    })?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "benchmark path is not a regular file: {}",
        canonical_path.display()
    );
    let filename = canonical_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("benchmark file name is not valid UTF-8")?
        .to_owned();
    let size_bytes = metadata.len();
    let sha256 = sha256_file(&canonical_path)?;
    let final_size = fs::metadata(&canonical_path)
        .with_context(|| format!("failed to recheck {}", canonical_path.display()))?
        .len();
    ensure!(
        final_size == size_bytes,
        "benchmark file size changed while hashing: {}",
        canonical_path.display()
    );
    Ok(FileProvenance {
        canonical_path,
        filename,
        size_bytes,
        sha256,
    })
}

/// Verify an arbitrary file against immutable size and digest expectations.
pub fn verify_file(
    path: impl AsRef<Path>,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<FileProvenance> {
    ensure!(
        is_lower_hex(expected_sha256, 64),
        "expected SHA-256 must be a lowercase 64-character digest"
    );
    let provenance = file_provenance(path)?;
    ensure!(
        provenance.size_bytes == expected_size,
        "file size mismatch for {}: expected {expected_size}, found {}",
        provenance.canonical_path.display(),
        provenance.size_bytes
    );
    ensure!(
        provenance.sha256 == expected_sha256,
        "file SHA-256 mismatch for {}",
        provenance.canonical_path.display()
    );
    Ok(provenance)
}

/// Verify the model file, download marker, and manifest identity together.
pub fn verify_model_provenance(requirement: &ModelRequirement) -> Result<ModelProvenance> {
    let model_path = resolve_model_path(&requirement.path, &requirement.filename)?;
    let file = verify_file(model_path, requirement.size_bytes, &requirement.sha256)?;
    ensure!(
        file.filename == requirement.filename,
        "model filename mismatch: expected {}, found {}",
        requirement.filename,
        file.filename
    );
    let model_directory = file
        .canonical_path
        .parent()
        .context("model file has no parent directory")?;
    let snapshot = read_snapshot_provenance(model_directory)?;
    ensure!(snapshot.complete, "model snapshot marker is incomplete");
    ensure!(
        snapshot.repository == requirement.repository,
        "model repository mismatch: expected {}, found {}",
        requirement.repository,
        snapshot.repository
    );
    ensure!(
        snapshot.revision == requirement.revision,
        "model revision mismatch: expected {}, found {}",
        requirement.revision,
        snapshot.revision
    );
    Ok(ModelProvenance {
        file,
        snapshot: Some(snapshot),
        architecture: requirement.architecture.clone(),
        quantization: requirement.quantization.clone(),
    })
}

/// Verify the prompt bytes; token-count and token-ID checks occur after tokenization.
pub fn verify_prompt_file(requirement: &PromptRequirement) -> Result<FileProvenance> {
    let metadata = fs::metadata(&requirement.path).with_context(|| {
        format!(
            "failed to inspect prompt file {}",
            requirement.path.display()
        )
    })?;
    verify_file(&requirement.path, metadata.len(), &requirement.sha256)
}

/// Read and validate a completed `.macllm-revision` marker.
pub fn read_snapshot_provenance(directory: impl AsRef<Path>) -> Result<SnapshotProvenance> {
    let path = directory.as_ref().join(SNAPSHOT_MARKER);
    let metadata = fs::symlink_metadata(&path)
        .with_context(|| format!("missing model snapshot marker {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "model snapshot marker must be a regular file: {}",
        path.display()
    );
    let bytes = fs::read(&path)
        .with_context(|| format!("failed to read model snapshot marker {}", path.display()))?;
    parse_snapshot_provenance(&bytes)
        .with_context(|| format!("invalid model snapshot marker {}", path.display()))
}

/// Parse snapshot identity from bytes that a caller has already tied to a
/// guarded file digest. Keeping parsing separate avoids reading a marker once
/// for JSON and again for provenance under a different file identity.
pub(super) fn parse_snapshot_provenance(bytes: &[u8]) -> Result<SnapshotProvenance> {
    let marker: SnapshotMarker =
        serde_json::from_slice(bytes).context("invalid model snapshot marker JSON")?;
    ensure!(
        marker
            .repo_id
            .split_once('/')
            .is_some_and(|(owner, name)| { !owner.trim().is_empty() && !name.trim().is_empty() }),
        "model snapshot marker repository is invalid"
    );
    ensure!(
        is_lower_hex(&marker.revision, 40),
        "model snapshot marker revision is not a full lowercase commit hash"
    );
    if let Some(requested) = &marker.requested_revision {
        ensure!(
            !requested.trim().is_empty(),
            "model snapshot requested revision cannot be empty"
        );
    }
    Ok(SnapshotProvenance {
        repository: marker.repo_id,
        revision: marker.revision,
        requested_revision: marker.requested_revision,
        complete: marker.complete,
    })
}

/// SHA-256 over bytes in memory.
#[must_use]
pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// SHA-256 over token IDs serialized as canonical little-endian `u32` values.
#[must_use]
pub fn token_ids_sha256(token_ids: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in token_ids {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(HASH_BUFFER_BYTES, file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("failed to hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn resolve_model_path(configured: &Path, expected_filename: &str) -> Result<PathBuf> {
    if configured.is_file() {
        return Ok(configured.to_owned());
    }
    ensure!(
        configured.is_dir(),
        "model path is not a file or directory: {}",
        configured.display()
    );
    let candidate = configured.join(expected_filename);
    ensure!(
        candidate.is_file(),
        "model directory does not contain expected GGUF {}",
        candidate.display()
    );
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn file_hash_and_token_fingerprint_are_stable() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("prompt.txt");
        fs::write(&path, b"hello benchmark")?;
        let expected = sha256_bytes(b"hello benchmark");
        let provenance = verify_file(&path, 15, &expected)?;
        assert_eq!(provenance.filename, "prompt.txt");
        assert_eq!(provenance.sha256, expected);
        assert_eq!(
            token_ids_sha256(&[0x0102_0304, 5]),
            sha256_bytes(&[4, 3, 2, 1, 5, 0, 0, 0])
        );
        Ok(())
    }

    #[test]
    fn changed_size_or_digest_is_rejected() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model.gguf");
        fs::write(&path, b"fixture")?;
        assert!(verify_file(&path, 6, &sha256_bytes(b"fixture")).is_err());
        assert!(verify_file(&path, 7, &sha256_bytes(b"different")).is_err());
        Ok(())
    }

    #[test]
    fn snapshot_marker_is_strict_and_sanitized() -> Result<()> {
        let directory = tempfile::tempdir()?;
        fs::write(
            directory.path().join(SNAPSHOT_MARKER),
            br#"{
              "repo_id":"owner/model",
              "revision":"0123456789abcdef0123456789abcdef01234567",
              "requested_revision":"main",
              "complete":true
            }"#,
        )?;
        let marker = read_snapshot_provenance(directory.path())?;
        assert_eq!(marker.repository, "owner/model");
        assert!(marker.complete);

        fs::write(
            directory.path().join(SNAPSHOT_MARKER),
            br#"{
              "repo_id":"owner/model",
              "revision":"0123456789abcdef0123456789abcdef01234567",
              "complete":true,
              "unrecognized":1
            }"#,
        )?;
        assert!(read_snapshot_provenance(directory.path()).is_err());
        Ok(())
    }

    #[test]
    fn model_provenance_requires_exact_bytes_and_marker() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let filename = "fixture.gguf";
        let bytes = b"exact gguf bytes";
        fs::write(directory.path().join(filename), bytes)?;
        fs::write(
            directory.path().join(SNAPSHOT_MARKER),
            br#"{
              "repo_id":"owner/model",
              "revision":"0123456789abcdef0123456789abcdef01234567",
              "complete":true
            }"#,
        )?;
        let requirement = ModelRequirement {
            path: directory.path().to_path_buf(),
            repository: "owner/model".to_owned(),
            revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            filename: filename.to_owned(),
            size_bytes: bytes.len() as u64,
            sha256: sha256_bytes(bytes),
            architecture: "fixture".to_owned(),
            quantization: "Q4_0".to_owned(),
        };
        let provenance = verify_model_provenance(&requirement)?;
        assert_eq!(provenance.file.filename, filename);
        assert_eq!(
            provenance
                .snapshot
                .as_ref()
                .map(|marker| marker.repository.as_str()),
            Some("owner/model")
        );
        Ok(())
    }
}
