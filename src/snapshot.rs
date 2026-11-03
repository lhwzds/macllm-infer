//! Model snapshot markers.
//!
//! A downloaded snapshot records the resolved revision and whether the
//! download completed in a small marker file next to the weights. The
//! download hub writes the marker and the model loader verifies it, so the
//! marker logic lives in its own module that both sides can use without
//! depending on each other.

use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

/// Name of the marker file written next to a downloaded snapshot.
pub(crate) const REVISION_MARKER: &str = ".macllm-revision";

/// Contents of the snapshot marker file.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SnapshotMarker {
    pub(crate) repo_id: String,
    /// Resolved immutable Hugging Face commit hash.
    pub(crate) revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) requested_revision: Option<String>,
    pub(crate) complete: bool,
}

/// Return the marker path of a snapshot directory, if the directory carries a
/// regular marker file.
pub(crate) fn checked_marker_path(destination: &Path) -> Result<Option<PathBuf>> {
    let marker_path = destination.join(REVISION_MARKER);
    let metadata = match fs::symlink_metadata(&marker_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to inspect {}", marker_path.display()));
        }
    };
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "model snapshot marker must be a regular file: {}",
        marker_path.display()
    );
    Ok(Some(marker_path))
}

/// Read the marker of a snapshot directory.
pub(crate) fn read_snapshot_marker(path: &Path) -> Result<SnapshotMarker> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("invalid model snapshot marker {}", path.display()))
}

/// Record the resolved revision of a snapshot directory.
pub(crate) fn record_snapshot_marker(
    destination: &Path,
    repo_id: &str,
    resolved_revision: &str,
    requested_revision: Option<&str>,
    complete: bool,
) -> Result<()> {
    fs::create_dir_all(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    let marker = SnapshotMarker {
        repo_id: repo_id.to_owned(),
        revision: resolved_revision.to_owned(),
        requested_revision: requested_revision.map(str::to_owned),
        complete,
    };
    fs::write(
        destination.join(REVISION_MARKER),
        serde_json::to_vec_pretty(&marker)?,
    )
    .context("failed to record model snapshot revision")
}

/// Reject a snapshot whose download did not complete.
pub(crate) fn ensure_snapshot_complete(destination: &Path) -> Result<()> {
    let Some(marker_path) = checked_marker_path(destination)? else {
        return Ok(());
    };
    let marker = read_snapshot_marker(&marker_path)?;
    ensure!(
        marker.complete,
        "model download {}@{} is incomplete; rerun `macllm download` with the same repository and output",
        marker.repo_id,
        marker.revision
    );
    Ok(())
}
