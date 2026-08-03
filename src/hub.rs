use std::{
    ffi::OsStr,
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use hf_hub::{HFClientSync, split_id};
use serde::{Deserialize, Serialize};

use crate::gguf::{MAX_GGUF_SHARDS, gguf_split_filename, parse_gguf_split_filename};

const REVISION_MARKER: &str = ".macllm-revision";
const MAX_GGUF_FILENAME_BYTES: usize = 255;
const SAFETENSORS_ROOT_FILES: [&str; 8] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "modules.json",
    "config_sentence_transformers.json",
];
const SAFETENSORS_NESTED_FILES: [&str; 2] = ["1_Pooling/config.json", "1_LogitScore/config.json"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadSelection {
    Safetensors,
    /// Download one exact root-level GGUF file or one exact canonical shard
    /// family. Requiring its first filename avoids fetching other quantizations.
    Gguf(String),
}

impl DownloadSelection {
    fn allow_patterns(&self) -> Result<Vec<String>> {
        match self {
            Self::Safetensors => Ok(SAFETENSORS_ROOT_FILES
                .into_iter()
                .chain(["model-*.safetensors"])
                .chain(SAFETENSORS_NESTED_FILES)
                .map(str::to_owned)
                .collect()),
            Self::Gguf(filename) => selected_gguf_filenames(filename),
        }
    }

    fn selects(&self, name: &str) -> bool {
        match self {
            Self::Safetensors => {
                SAFETENSORS_ROOT_FILES.contains(&name)
                    || (name.starts_with("model-") && name.ends_with(".safetensors"))
            }
            Self::Gguf(filename) => selects_gguf_filename(filename, name),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SnapshotMarker {
    repo_id: String,
    /// Resolved immutable Hugging Face commit hash.
    revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    requested_revision: Option<String>,
    complete: bool,
}

/// Download a clean model snapshot and only replace `destination` after every
/// required file has arrived and passed the normal model-file validation.
pub fn download_model(
    repo_id: &str,
    revision: Option<&str>,
    destination: impl AsRef<Path>,
    selection: &DownloadSelection,
) -> Result<PathBuf> {
    let (owner, name) = split_id(repo_id);
    ensure!(!owner.is_empty(), "model id must be in owner/name form");
    ensure!(!name.is_empty(), "model name cannot be empty");
    let destination = destination.as_ref().to_path_buf();
    validate_output_path(&destination)?;
    validate_destination(&destination, repo_id)?;

    let parent = destination_parent(&destination);
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create model parent {}", parent.display()))?;
    let staging_root = tempfile::Builder::new()
        .prefix(".macllm-download-")
        .tempdir_in(parent)
        .with_context(|| format!("failed to create staging directory in {}", parent.display()))?;
    let staging = staging_root.path().join("snapshot");
    fs::create_dir(&staging)
        .with_context(|| format!("failed to create staging snapshot {}", staging.display()))?;
    record_snapshot_marker(
        &staging,
        repo_id,
        revision.unwrap_or("main"),
        revision,
        false,
    )?;

    let client = HFClientSync::new().context("failed to create Hugging Face client")?;
    let cache_snapshot = client
        .model(owner, name)
        .snapshot_download()
        .maybe_revision(revision)
        .allow_patterns(selection.allow_patterns()?)
        .send()
        .context("failed to download model snapshot")?;
    let resolved_revision = resolved_revision_from_cache_path(&cache_snapshot)?;
    copy_snapshot_files(&cache_snapshot, &staging, selection)?;
    record_snapshot_marker(&staging, repo_id, &resolved_revision, revision, true)?;
    crate::weights::ModelFiles::from_dir(&staging)
        .context("downloaded snapshot does not contain a structurally complete model file set")?;

    // The destination may have changed while the network request was running.
    validate_destination(&destination, repo_id)?;
    replace_destination(&staging, &destination, repo_id)?;
    Ok(destination)
}

fn validate_gguf_filename(filename: &str) -> Result<()> {
    let path = Path::new(filename);
    ensure!(!filename.is_empty(), "GGUF filename cannot be empty");
    ensure!(
        filename.len() <= MAX_GGUF_FILENAME_BYTES,
        "GGUF filename exceeds the {MAX_GGUF_FILENAME_BYTES}-byte filesystem limit"
    );
    ensure!(
        path.components().count() == 1 && path.file_name() == Some(OsStr::new(filename)),
        "GGUF filename must be one root-level filename, not a path"
    );
    ensure!(
        filename
            .bytes()
            .all(|byte| { byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-') }),
        "GGUF filename may contain only ASCII letters, digits, '.', '_', and '-'"
    );
    ensure!(
        path.extension()
            .is_some_and(|extension| extension == "gguf"),
        "GGUF filename must end in .gguf"
    );
    Ok(())
}

fn selected_gguf_filenames(filename: &str) -> Result<Vec<String>> {
    validate_gguf_filename(filename)?;
    let Some(split) = parse_gguf_split_filename(filename) else {
        return Ok(vec![filename.to_owned()]);
    };
    ensure!(
        split.index == 0,
        "select the canonical first GGUF shard, `{}`",
        gguf_split_filename(&split.prefix, 0, split.count)
    );
    ensure!(
        split.count <= MAX_GGUF_SHARDS,
        "GGUF filename declares {} shards, exceeding the supported safety limit {MAX_GGUF_SHARDS}",
        split.count
    );
    let mut filenames = Vec::new();
    filenames
        .try_reserve_exact(usize::from(split.count))
        .context("failed to reserve GGUF download selection")?;
    for index in 0..split.count {
        filenames.push(gguf_split_filename(&split.prefix, index, split.count));
    }
    Ok(filenames)
}

fn selects_gguf_filename(requested: &str, candidate: &str) -> bool {
    let Some(requested_split) = parse_gguf_split_filename(requested) else {
        return candidate == requested;
    };
    parse_gguf_split_filename(candidate).is_some_and(|candidate_split| {
        candidate_split.prefix == requested_split.prefix
            && candidate_split.count == requested_split.count
            && candidate_split.index < requested_split.count
    })
}

fn validate_output_path(destination: &Path) -> Result<()> {
    let name = destination
        .file_name()
        .context("model output must name a directory, not a filesystem root")?;
    ensure!(
        name != OsStr::new(".") && name != OsStr::new(".."),
        "model output must name a concrete directory"
    );
    Ok(())
}

fn destination_parent(destination: &Path) -> &Path {
    destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn validate_destination(destination: &Path, repo_id: &str) -> Result<()> {
    let metadata = match fs::symlink_metadata(destination) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to inspect {}", destination.display()));
        }
    };
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "model destination must be a real directory: {}",
        destination.display()
    );
    if let Some(marker_path) = checked_marker_path(destination)? {
        let marker = read_snapshot_marker(&marker_path)?;
        ensure!(
            marker.repo_id == repo_id,
            "{} contains a snapshot of {}, not {}; choose another output directory",
            destination.display(),
            marker.repo_id,
            repo_id
        );
        return Ok(());
    }
    let has_files = fs::read_dir(destination)
        .with_context(|| format!("failed to read {}", destination.display()))?
        .next()
        .transpose()?
        .is_some();
    ensure!(
        !has_files,
        "refusing to replace non-empty untracked directory {}; choose an empty output directory",
        destination.display()
    );
    Ok(())
}

fn checked_marker_path(destination: &Path) -> Result<Option<PathBuf>> {
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

fn resolved_revision_from_cache_path(snapshot: &Path) -> Result<String> {
    let revision = snapshot
        .file_name()
        .and_then(OsStr::to_str)
        .context("Hugging Face returned a snapshot path without a commit hash")?;
    ensure!(
        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "Hugging Face returned an invalid resolved commit hash `{revision}`"
    );
    Ok(revision.to_ascii_lowercase())
}

fn copy_snapshot_files(
    source: &Path,
    destination: &Path,
    selection: &DownloadSelection,
) -> Result<()> {
    let mut copied = 0_usize;
    for entry in
        fs::read_dir(source).with_context(|| format!("failed to read {}", source.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name_text) = name.to_str() else {
            continue;
        };
        if !selection.selects(name_text) {
            continue;
        }
        let source_file = entry.path();
        ensure!(
            source_file.is_file(),
            "snapshot entry is not a regular file: {}",
            source_file.display()
        );
        fs::copy(&source_file, destination.join(&name)).with_context(|| {
            format!(
                "failed to copy snapshot file {} into staging",
                source_file.display()
            )
        })?;
        copied += 1;
    }
    if matches!(selection, DownloadSelection::Safetensors) {
        for relative in SAFETENSORS_NESTED_FILES {
            if copy_optional_snapshot_file(source, destination, relative)? {
                copied += 1;
            }
        }
    }
    ensure!(
        copied > 0,
        "Hugging Face snapshot contained no selected model files"
    );
    if let DownloadSelection::Gguf(filename) = selection {
        let expected = selected_gguf_filenames(filename)?.len();
        ensure!(
            copied == expected,
            "Hugging Face snapshot contained {copied} of {expected} required GGUF files"
        );
    }
    Ok(())
}

fn copy_optional_snapshot_file(source: &Path, destination: &Path, relative: &str) -> Result<bool> {
    let source_file = source.join(relative);
    if !source_file.exists() {
        return Ok(false);
    }
    ensure!(
        source_file.is_file(),
        "snapshot entry is not a regular file: {}",
        source_file.display()
    );
    let destination_file = destination.join(relative);
    let parent = destination_file
        .parent()
        .expect("selected nested snapshot file has a parent");
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create snapshot directory {}", parent.display()))?;
    fs::copy(&source_file, &destination_file).with_context(|| {
        format!(
            "failed to copy snapshot file {} into staging",
            source_file.display()
        )
    })?;
    Ok(true)
}

fn replace_destination(staging: &Path, destination: &Path, repo_id: &str) -> Result<()> {
    match fs::symlink_metadata(destination) {
        Err(error) if error.kind() == ErrorKind::NotFound => {
            fs::rename(staging, destination).with_context(|| {
                format!(
                    "failed to install model snapshot at {}",
                    destination.display()
                )
            })?;
            return Ok(());
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to inspect {}", destination.display()));
        }
        Ok(metadata) => ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "model destination must remain a real directory: {}",
            destination.display()
        ),
    }

    let parent = destination_parent(destination);
    let backup_root = tempfile::Builder::new()
        .prefix(".macllm-backup-")
        .tempdir_in(parent)
        .with_context(|| format!("failed to create backup directory in {}", parent.display()))?;
    let backup = backup_root.path().join("previous");
    fs::rename(destination, &backup).with_context(|| {
        format!(
            "failed to move previous snapshot {} into a temporary backup",
            destination.display()
        )
    })?;
    if let Err(validation_error) = validate_destination(&backup, repo_id) {
        if let Err(restore_error) = fs::rename(&backup, destination) {
            let kept_backup = backup_root.keep().join("previous");
            bail!(
                "destination changed ownership during installation ({validation_error}) and could not be restored ({restore_error}); previous data remains at {}",
                kept_backup.display()
            );
        }
        return Err(validation_error)
            .context("destination changed ownership while the snapshot was downloading");
    }
    if let Err(install_error) = fs::rename(staging, destination) {
        if let Err(restore_error) = fs::rename(&backup, destination) {
            let kept_backup = backup_root.keep().join("previous");
            bail!(
                "failed to install new snapshot ({install_error}) and restore the previous snapshot ({restore_error}); previous data remains at {}",
                kept_backup.display()
            );
        }
        return Err(install_error).with_context(|| {
            format!(
                "failed to install model snapshot at {}; previous snapshot was restored",
                destination.display()
            )
        });
    }
    // Dropping the backup only happens after the new complete snapshot is in place.
    drop(backup_root);
    Ok(())
}

fn record_snapshot_marker(
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

fn read_snapshot_marker(path: &Path) -> Result<SnapshotMarker> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("invalid model snapshot marker {}", path.display()))
}

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

#[cfg(test)]
mod tests {
    use super::*;

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn safetensors_download_selects_exact_task_sidecars() -> Result<()> {
        let selection = DownloadSelection::Safetensors;
        let patterns = selection.allow_patterns()?;
        assert_eq!(
            patterns,
            [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "modules.json",
                "config_sentence_transformers.json",
                "model-*.safetensors",
                "1_Pooling/config.json",
                "1_LogitScore/config.json",
            ]
        );
        assert!(selection.selects("model.safetensors"));
        assert!(selection.selects("model-00001-of-00002.safetensors"));
        assert!(selection.selects("modules.json"));
        assert!(!selection.selects("README.md"));
        Ok(())
    }

    #[test]
    fn safetensors_snapshot_copy_preserves_only_selected_nested_sidecars() -> Result<()> {
        let source = tempfile::tempdir()?;
        let destination = tempfile::tempdir()?;
        for relative in [
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "modules.json",
            "config_sentence_transformers.json",
            "1_Pooling/config.json",
            "1_LogitScore/config.json",
            "unknown/config.json",
        ] {
            let path = source.path().join(relative);
            fs::create_dir_all(path.parent().expect("fixture path has a parent"))?;
            fs::write(path, relative)?;
        }

        copy_snapshot_files(
            source.path(),
            destination.path(),
            &DownloadSelection::Safetensors,
        )?;

        for relative in [
            "modules.json",
            "config_sentence_transformers.json",
            "1_Pooling/config.json",
            "1_LogitScore/config.json",
        ] {
            assert_eq!(
                fs::read_to_string(destination.path().join(relative))?,
                relative
            );
        }
        assert!(!destination.path().join("unknown/config.json").exists());
        Ok(())
    }

    #[test]
    fn safetensors_snapshot_rejects_a_non_file_selected_sidecar() -> Result<()> {
        let source = tempfile::tempdir()?;
        let destination = tempfile::tempdir()?;
        fs::write(source.path().join("model.safetensors"), b"weights")?;
        fs::create_dir_all(source.path().join("1_Pooling/config.json"))?;

        let error = copy_snapshot_files(
            source.path(),
            destination.path(),
            &DownloadSelection::Safetensors,
        )
        .expect_err("a selected sidecar directory must be rejected");
        assert!(error.to_string().contains("not a regular file"), "{error}");
        Ok(())
    }

    #[test]
    fn safetensors_snapshot_keeps_task_sidecars_optional_without_empty_directories() -> Result<()> {
        let source = tempfile::tempdir()?;
        let destination = tempfile::tempdir()?;
        fs::write(source.path().join("model.safetensors"), b"weights")?;

        copy_snapshot_files(
            source.path(),
            destination.path(),
            &DownloadSelection::Safetensors,
        )?;

        assert!(!destination.path().join("1_Pooling").exists());
        assert!(!destination.path().join("1_LogitScore").exists());
        Ok(())
    }

    #[test]
    fn gguf_download_selects_one_exact_file_or_canonical_shard_family() -> Result<()> {
        let selection = DownloadSelection::Gguf("model-f16.gguf".into());
        assert_eq!(selection.allow_patterns()?, vec!["model-f16.gguf"]);
        assert!(selection.selects("model-f16.gguf"));
        assert!(!selection.selects("model-q4.gguf"));

        let shards = DownloadSelection::Gguf("model-q4-00001-of-00003.gguf".into());
        assert_eq!(
            shards.allow_patterns()?,
            [
                "model-q4-00001-of-00003.gguf",
                "model-q4-00002-of-00003.gguf",
                "model-q4-00003-of-00003.gguf",
            ]
        );
        assert!(shards.selects("model-q4-00002-of-00003.gguf"));
        assert!(!shards.selects("model-f16-00002-of-00003.gguf"));
        assert!(!shards.selects("model-q4-00002-of-00004.gguf"));

        for invalid_first in [
            "model-q4-00002-of-00003.gguf",
            "model-q4-00001-of-00129.gguf",
        ] {
            assert!(
                DownloadSelection::Gguf(invalid_first.into())
                    .allow_patterns()
                    .is_err()
            );
        }

        for invalid in [
            "",
            "model.bin",
            "../model.gguf",
            "nested/model.gguf",
            "*.gguf",
            "model?.gguf",
            "{f16,q4}.gguf",
            "model[1].gguf",
            r"model\variant.gguf",
        ] {
            assert!(
                DownloadSelection::Gguf(invalid.into())
                    .allow_patterns()
                    .is_err()
            );
        }
        let oversized_name = format!("{}.gguf", "a".repeat(MAX_GGUF_FILENAME_BYTES));
        assert!(
            DownloadSelection::Gguf(oversized_name)
                .allow_patterns()
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn gguf_snapshot_copy_requires_every_exact_shard_and_excludes_other_families() -> Result<()> {
        let source = tempfile::tempdir()?;
        let incomplete = tempfile::tempdir()?;
        let selection = DownloadSelection::Gguf("model-q4-00001-of-00003.gguf".into());
        fs::write(source.path().join("model-q4-00001-of-00003.gguf"), b"one")?;
        let error = copy_snapshot_files(source.path(), incomplete.path(), &selection)
            .unwrap_err()
            .to_string();
        assert!(error.contains("1 of 3 required"), "{error}");

        fs::write(source.path().join("model-q4-00002-of-00003.gguf"), b"two")?;
        fs::write(source.path().join("model-q4-00003-of-00003.gguf"), b"three")?;
        fs::write(
            source.path().join("model-f16-00002-of-00003.gguf"),
            b"other",
        )?;
        let complete = tempfile::tempdir()?;
        copy_snapshot_files(source.path(), complete.path(), &selection)?;
        assert_eq!(fs::read_dir(complete.path())?.count(), 3);
        assert!(
            !complete
                .path()
                .join("model-f16-00002-of-00003.gguf")
                .exists()
        );
        Ok(())
    }

    #[test]
    fn validates_resolved_revisions_and_snapshot_markers() -> Result<()> {
        let cache = Path::new("cache/snapshots").join(COMMIT);
        assert_eq!(resolved_revision_from_cache_path(&cache)?, COMMIT);
        assert!(resolved_revision_from_cache_path(Path::new("cache/snapshots/main")).is_err());

        let temp = tempfile::tempdir()?;
        let destination = temp.path().join("model");
        record_snapshot_marker(&destination, "owner/model", COMMIT, None, false)?;
        validate_destination(&destination, "owner/model")?;
        assert!(ensure_snapshot_complete(&destination).is_err());
        record_snapshot_marker(&destination, "owner/model", COMMIT, None, true)?;
        ensure_snapshot_complete(&destination)?;
        let marker = read_snapshot_marker(&destination.join(REVISION_MARKER))?;
        assert_eq!(marker.revision, COMMIT);
        assert!(marker.requested_revision.is_none());
        assert!(marker.complete);
        assert!(validate_destination(&destination, "other/model").is_err());
        Ok(())
    }

    #[test]
    fn clean_replacement_removes_stale_snapshot_files() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let destination = temp.path().join("model");
        record_snapshot_marker(&destination, "owner/model", COMMIT, Some("main"), true)?;
        fs::write(destination.join("model.safetensors.index.json"), "stale")?;

        let staging = temp.path().join("new-snapshot");
        record_snapshot_marker(&staging, "owner/model", COMMIT, Some("main"), true)?;
        fs::write(staging.join("model.safetensors"), "new")?;
        replace_destination(&staging, &destination, "owner/model")?;

        assert_eq!(
            fs::read_to_string(destination.join("model.safetensors"))?,
            "new"
        );
        assert!(!destination.join("model.safetensors.index.json").exists());
        assert!(!staging.exists());
        Ok(())
    }

    #[test]
    fn replacement_restores_a_destination_with_changed_ownership() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let destination = temp.path().join("model");
        record_snapshot_marker(&destination, "other/model", COMMIT, None, true)?;
        fs::write(destination.join("old.txt"), "old")?;
        let staging = temp.path().join("new-snapshot");
        record_snapshot_marker(&staging, "owner/model", COMMIT, None, true)?;
        fs::write(staging.join("new.txt"), "new")?;

        assert!(replace_destination(&staging, &destination, "owner/model").is_err());
        assert_eq!(fs::read_to_string(destination.join("old.txt"))?, "old");
        assert!(!destination.join("new.txt").exists());
        assert!(staging.exists());
        Ok(())
    }

    #[test]
    fn refuses_nonempty_untracked_or_broad_destinations() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let destination = temp.path().join("model");
        fs::create_dir(&destination)?;
        fs::write(destination.join("notes.txt"), "user data")?;
        assert!(validate_destination(&destination, "owner/model").is_err());
        assert!(validate_output_path(Path::new("/")).is_err());
        assert!(validate_output_path(Path::new("..")).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn refuses_dangling_destinations_and_symlink_markers() -> Result<()> {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir()?;
        let destination = temp.path().join("model");
        symlink(temp.path().join("missing"), &destination)?;
        assert!(validate_destination(&destination, "owner/model").is_err());

        fs::remove_file(&destination)?;
        fs::create_dir(&destination)?;
        let external_marker = temp.path().join("external-marker");
        fs::write(
            &external_marker,
            serde_json::to_vec(&SnapshotMarker {
                repo_id: "owner/model".into(),
                revision: COMMIT.into(),
                requested_revision: None,
                complete: true,
            })?,
        )?;
        symlink(&external_marker, destination.join(REVISION_MARKER))?;
        assert!(validate_destination(&destination, "owner/model").is_err());
        assert!(ensure_snapshot_complete(&destination).is_err());
        Ok(())
    }
}
