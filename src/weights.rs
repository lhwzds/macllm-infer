use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{
    gguf::{GgufTensorType, MAX_GGUF_SHARDS, gguf_split_filename, parse_gguf_split_filename},
    gguf_model::GgufModel,
    model_weights::{DenseWeightSource, GgufWeightSource, WeightSource},
    models::{ModelAdapter, detect},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Safetensors,
    Gguf,
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Safetensors => formatter.write_str("safetensors"),
            Self::Gguf => formatter.write_str("gguf"),
        }
    }
}

#[derive(Debug)]
struct SafetensorsFiles {
    config: PathBuf,
    tokenizer: PathBuf,
    weights: Vec<PathBuf>,
}

#[derive(Debug)]
enum ModelSource {
    Safetensors(SafetensorsFiles),
    Gguf(Box<GgufModel>),
}

/// A validated model source. The CLI and runtime use the same methods whether
/// the checkpoint came from Hugging Face Safetensors sidecars or a logical
/// single- or multi-file GGUF checkpoint.
#[derive(Debug)]
pub struct ModelFiles {
    pub root: PathBuf,
    source: ModelSource,
}

impl ModelFiles {
    /// Open either a model directory or a standalone `.gguf` file.
    pub fn from_dir(root: impl AsRef<Path>) -> Result<Self> {
        let input = root.as_ref();
        if input.is_file() {
            ensure!(
                input
                    .extension()
                    .is_some_and(|extension| extension == "gguf"),
                "standalone model file must end in .gguf: {}",
                input.display()
            );
            return Ok(Self {
                root: input.to_path_buf(),
                source: ModelSource::Gguf(Box::new(GgufModel::open(input)?)),
            });
        }
        ensure!(
            input.is_dir(),
            "model path is not a directory or GGUF file: {}",
            input.display()
        );
        let root = input.to_path_buf();
        crate::hub::ensure_snapshot_complete(&root)?;

        let gguf_files = discover_gguf(&root)?;
        let safetensors = discover_safetensors(&root)?;
        if !gguf_files.is_empty() {
            ensure!(
                safetensors.is_empty(),
                "model directory mixes GGUF and Safetensors weights; select one source"
            );
            let gguf = select_gguf_entrypoint(&gguf_files)?;
            return Ok(Self {
                root,
                source: ModelSource::Gguf(Box::new(GgufModel::open(gguf)?)),
            });
        }

        let config = required_file(&root, "config.json")?;
        let tokenizer = required_file(&root, "tokenizer.json")?;
        let weights = safetensors;
        if weights.is_empty() {
            bail!(
                "no .safetensors or .gguf weights found in {}",
                root.display()
            );
        }
        Ok(Self {
            root,
            source: ModelSource::Safetensors(SafetensorsFiles {
                config,
                tokenizer,
                weights,
            }),
        })
    }

    #[must_use]
    pub fn format(&self) -> ModelFormat {
        match self.source {
            ModelSource::Safetensors(_) => ModelFormat::Safetensors,
            ModelSource::Gguf(_) => ModelFormat::Gguf,
        }
    }

    #[must_use]
    pub fn weight_file_count(&self) -> usize {
        match &self.source {
            ModelSource::Safetensors(files) => files.weights.len(),
            ModelSource::Gguf(model) => model.file().shard_count(),
        }
    }

    /// Return every physical weight file in deterministic checkpoint order.
    ///
    /// GGUF split checkpoints include every shard; Safetensors follow the
    /// canonical index/discovery order already validated at load time.
    #[must_use]
    pub fn weight_paths(&self) -> Vec<&Path> {
        match &self.source {
            ModelSource::Safetensors(files) => files.weights.iter().map(PathBuf::as_path).collect(),
            ModelSource::Gguf(model) => model.file().shard_paths().collect(),
        }
    }

    #[must_use]
    pub fn tensor_count(&self) -> Option<usize> {
        match &self.source {
            ModelSource::Safetensors(_) => None,
            ModelSource::Gguf(model) => Some(model.file().tensors().len()),
        }
    }

    #[must_use]
    pub fn quantized_tensor_count(&self) -> usize {
        match &self.source {
            ModelSource::Safetensors(_) => 0,
            ModelSource::Gguf(model) => model
                .file()
                .tensors()
                .values()
                .filter(|tensor| tensor.dtype.is_quantized())
                .count(),
        }
    }

    #[must_use]
    pub fn has_quantized_tensors(&self) -> bool {
        self.quantized_tensor_count() > 0
    }

    #[must_use]
    pub fn has_gguf_tensor_type(&self, dtype: GgufTensorType) -> bool {
        match &self.source {
            ModelSource::Safetensors(_) => false,
            ModelSource::Gguf(model) => model
                .file()
                .tensors()
                .values()
                .any(|tensor| tensor.dtype == dtype),
        }
    }

    #[must_use]
    pub fn gguf_tensor_type_counts(&self) -> Vec<(GgufTensorType, usize)> {
        let ModelSource::Gguf(model) = &self.source else {
            return Vec::new();
        };
        GgufTensorType::ALL
            .into_iter()
            .filter_map(|dtype| {
                let count = model
                    .file()
                    .tensors()
                    .values()
                    .filter(|tensor| tensor.dtype == dtype)
                    .count();
                (count > 0).then_some((dtype, count))
            })
            .collect()
    }

    pub fn config_path(&self) -> Result<&Path> {
        match &self.source {
            ModelSource::Safetensors(files) => Ok(&files.config),
            ModelSource::Gguf(_) => bail!("GGUF checkpoints have no config.json sidecar"),
        }
    }

    pub fn tokenizer_path(&self) -> Result<&Path> {
        match &self.source {
            ModelSource::Safetensors(files) => Ok(&files.tokenizer),
            ModelSource::Gguf(_) => bail!("GGUF checkpoints have no tokenizer.json sidecar"),
        }
    }

    pub fn safetensor_weights(&self) -> Result<&[PathBuf]> {
        match &self.source {
            ModelSource::Safetensors(files) => Ok(&files.weights),
            ModelSource::Gguf(_) => bail!("model source is GGUF, not Safetensors"),
        }
    }

    pub fn adapter(&self) -> Result<Box<dyn ModelAdapter>> {
        match &self.source {
            ModelSource::Safetensors(files) => detect(&files.config),
            ModelSource::Gguf(model) => model.adapter(),
        }
    }

    pub fn tokenizer(&self) -> Result<Arc<Tokenizer>> {
        match &self.source {
            ModelSource::Safetensors(files) => Tokenizer::from_file(&files.tokenizer)
                .map(Arc::new)
                .map_err(|error| anyhow::anyhow!(error.to_string()))
                .with_context(|| format!("failed to load {}", files.tokenizer.display())),
            ModelSource::Gguf(model) => Ok(model.tokenizer()),
        }
    }

    /// Create the shared decoder's format-neutral weight source. Quantized GGUF
    /// matrices remain compressed; dense Safetensors keep the existing mmap path.
    ///
    /// # Safety
    ///
    /// Checkpoint contents must remain unmodified and untruncated while tensors
    /// or models built from this source remain alive. Safetensors paths must not
    /// be replaced during mapping; GGUF retains its original opened handle, so
    /// replacing that path after parsing is safe.
    pub unsafe fn weight_source(
        &self,
        dtype: DType,
        device: &Device,
    ) -> Result<Box<dyn WeightSource>> {
        match &self.source {
            ModelSource::Safetensors(files) => {
                // SAFETY: the caller upholds Candle's mmap requirement.
                let weights =
                    unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, dtype, device) }
                        .context("failed to mmap safetensors checkpoint")?;
                Ok(Box::new(DenseWeightSource::new(weights)))
            }
            ModelSource::Gguf(model) => {
                // SAFETY: the caller upholds the GGUF file immutability contract.
                let backend = unsafe { model.file().backend(model.backend_config()) }?;
                Ok(Box::new(GgufWeightSource::new(
                    backend,
                    dtype,
                    device.clone(),
                )))
            }
        }
    }

    /// Validation-only source that expands GGUF block matrices after parsing.
    /// This provides a same-file dense reference for quantized kernel parity.
    ///
    /// # Safety
    ///
    /// The checkpoint immutability requirements are identical to
    /// [`Self::weight_source`].
    pub unsafe fn dequantized_weight_source(
        &self,
        dtype: DType,
        device: &Device,
    ) -> Result<Box<dyn WeightSource>> {
        match &self.source {
            ModelSource::Safetensors(_) => {
                // SAFETY: this simply delegates the same caller contract.
                unsafe { self.weight_source(dtype, device) }
            }
            ModelSource::Gguf(model) => {
                // SAFETY: the caller upholds the GGUF file immutability contract.
                let backend = unsafe { model.file().backend(model.backend_config()) }?;
                Ok(Box::new(GgufWeightSource::new_dequantizing(
                    backend,
                    dtype,
                    device.clone(),
                )))
            }
        }
    }
}

fn required_file(root: &Path, name: &str) -> Result<PathBuf> {
    let path = root.join(name);
    if !path.is_file() {
        bail!("required model file is missing: {}", path.display());
    }
    Ok(path)
}

fn discover_gguf(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(root)
        .with_context(|| format!("failed to read model directory {}", root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "gguf") {
            continue;
        }
        if fs::metadata(&path)
            .with_context(|| format!("failed to stat GGUF candidate {}", path.display()))?
            .is_file()
        {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn select_gguf_entrypoint(files: &[PathBuf]) -> Result<&Path> {
    let [only] = files else {
        let first = files
            .first()
            .context("cannot select a GGUF checkpoint from an empty file list")?;
        let filename = first
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| format!("GGUF filename must be UTF-8: {}", first.display()))?;
        let split = parse_gguf_split_filename(filename).with_context(
            || "model directory contains multiple unrelated GGUF files; select one checkpoint",
        )?;
        ensure!(
            split.index == 0,
            "model directory has GGUF split files but no canonical first shard"
        );
        ensure!(
            split.count <= MAX_GGUF_SHARDS,
            "GGUF declares {} shards, exceeding the supported safety limit {MAX_GGUF_SHARDS}",
            split.count
        );
        ensure!(
            usize::from(split.count) == files.len(),
            "model directory does not contain exactly one complete GGUF shard set: expected {}, found {} GGUF files",
            split.count,
            files.len()
        );
        let parent = first.parent().unwrap_or_else(|| Path::new(""));
        for (index, actual) in files.iter().enumerate() {
            let index = u16::try_from(index).context("GGUF shard index does not fit u16")?;
            let expected = parent.join(gguf_split_filename(&split.prefix, index, split.count));
            ensure!(
                actual == &expected,
                "model directory GGUF shard set is incomplete or contains an unrelated file: expected {}, found {}",
                expected.display(),
                actual.display()
            );
        }
        return Ok(first.as_path());
    };
    Ok(only.as_path())
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

fn discover_safetensors(root: &Path) -> Result<Vec<PathBuf>> {
    let index_path = root.join("model.safetensors.index.json");
    if index_path.is_file() {
        let index: SafetensorsIndex = serde_json::from_slice(
            &fs::read(&index_path)
                .with_context(|| format!("failed to read {}", index_path.display()))?,
        )
        .with_context(|| format!("invalid {}", index_path.display()))?;
        let shards: BTreeSet<_> = index.weight_map.into_values().collect();
        return shards
            .into_iter()
            .map(|name| required_file(root, &name))
            .collect();
    }

    let mut files = fs::read_dir(root)
        .with_context(|| format!("failed to read model directory {}", root.display()))?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn index_controls_shard_order_and_deduplicates() -> Result<()> {
        let temp = tempfile::tempdir()?;
        fs::write(temp.path().join("config.json"), "{}")?;
        fs::write(temp.path().join("tokenizer.json"), "{}")?;
        fs::write(temp.path().join("a.safetensors"), "")?;
        fs::write(temp.path().join("b.safetensors"), "")?;
        fs::write(
            temp.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"b.safetensors","y":"a.safetensors","z":"b.safetensors"}}"#,
        )?;
        let files = ModelFiles::from_dir(temp.path())?;
        let weights = files.safetensor_weights()?;
        assert_eq!(weights.len(), 2);
        assert!(weights[0].ends_with("a.safetensors"));
        assert!(weights[1].ends_with("b.safetensors"));
        assert_eq!(
            files.weight_paths(),
            [weights[0].as_path(), weights[1].as_path()]
        );
        assert_eq!(files.format(), ModelFormat::Safetensors);
        Ok(())
    }

    #[test]
    fn rejects_ambiguous_model_directories() -> Result<()> {
        let temp = tempfile::tempdir()?;
        fs::write(
            temp.path().join("model.gguf"),
            b"not parsed because ambiguity wins",
        )?;
        fs::write(temp.path().join("model.safetensors"), "")?;
        assert!(ModelFiles::from_dir(temp.path()).is_err());

        let second = tempfile::tempdir()?;
        fs::write(second.path().join("a.gguf"), "")?;
        fs::write(second.path().join("b.gguf"), "")?;
        assert!(ModelFiles::from_dir(second.path()).is_err());
        Ok(())
    }

    #[test]
    fn selects_exactly_one_complete_canonical_gguf_shard_set() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let files = (1..=3)
            .map(|index| {
                let path = temp
                    .path()
                    .join(format!("model-q4-0000{index}-of-00003.gguf"));
                fs::write(&path, [])?;
                Ok(path)
            })
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(select_gguf_entrypoint(&files)?, files[0]);

        assert!(select_gguf_entrypoint(&files[..2]).is_err());
        let mut mixed = files.clone();
        mixed[2] = temp.path().join("other.gguf");
        assert!(select_gguf_entrypoint(&mixed).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn directory_discovery_follows_symlinked_regular_gguf_files() -> Result<()> {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir()?;
        let target = tempfile::NamedTempFile::new()?;
        let link = directory.path().join("model.gguf");
        symlink(target.path(), &link)?;
        symlink("missing-target", directory.path().join("unrelated.txt"))?;
        assert_eq!(discover_gguf(directory.path())?, [link]);
        Ok(())
    }
}
