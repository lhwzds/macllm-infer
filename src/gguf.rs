//! Minimal, strict GGUF v3 support for decoder checkpoints.
//!
//! Parsing is intentionally independent from Candle's quantized GGUF reader. The
//! index is read with ordinary buffered I/O; only tensor payloads are memory
//! mapped, behind the same file-immutability contract as Candle's safetensors
//! mmap backend.

use std::{
    borrow::Cow,
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{self, BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(feature = "metal")]
use std::{collections::HashMap, sync::Mutex};

use anyhow::{Context, Result, bail, ensure};
use candle_core::{
    DType, Device, Error as CandleError, Shape, Tensor,
    quantized::{GgmlDType, QStorage, QTensor},
};
use candle_nn::var_builder::SimpleBackend;
use memmap2::{Mmap, MmapOptions};

use crate::architecture::MAX_ROUTED_EXPERTS;
#[cfg(feature = "metal")]
use crate::file_backed_quant::{FileBackedMetalShard, FileBackedQuantizedTensor};

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;
// These implementation limits are intentionally below the format's theoretical
// range so a valid-looking sparse file cannot request multi-gigabyte allocations.
// They remain far above known tokenizer vocabularies and model tensor counts.
const MAX_STRING_LENGTH: u64 = 16 << 20;
const MAX_COLLECTION_ENTRIES: u64 = 1 << 20;
const MAX_DECODED_ALLOCATION_BYTES: u64 = 256 << 20;
// Conservatively covers map nodes, Vec headers, duplicated tensor names, and
// small per-entry allocations that are not represented by on-disk bytes.
const INDEX_ALLOCATION_BYTES_PER_ENTRY: u64 = 256;
const MAX_METADATA_KEY_LENGTH: usize = 65_535;
const MAX_TENSOR_NAME_LENGTH: usize = 63;
const MAX_TENSOR_DIMS: u32 = 4;
const MAX_AUXILIARY_F32_ELEMENTS: u64 = 1 << 20;
// Retaining one immutable handle per shard preserves path-replacement safety.
// Leave substantial headroom under the common 256-descriptor process limit.
pub(crate) const MAX_GGUF_SHARDS: u16 = 128;
const SPLIT_NO_KEY: &str = "split.no";
const SPLIT_COUNT_KEY: &str = "split.count";
const SPLIT_TENSORS_COUNT_KEY: &str = "split.tensors.count";

/// A GGUF metadata value tag as encoded on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl GgufValueType {
    fn from_tag(tag: u32) -> Result<Self> {
        match tag {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _ => bail!("unknown GGUF metadata value type {tag}"),
        }
    }

    fn minimum_disk_size(self) -> u64 {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 | Self::String => 8,
            Self::Array => 12,
        }
    }
}

/// A decoded GGUF metadata value.
///
/// Arrays retain their element tag even when empty. Elements are owned once by
/// the metadata index and can be traversed through [`GgufValue::as_array`]
/// without copying large tokenizer tables.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: GgufValueType,
        values: Vec<GgufValue>,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    #[must_use]
    pub fn value_type(&self) -> GgufValueType {
        match self {
            Self::U8(_) => GgufValueType::U8,
            Self::I8(_) => GgufValueType::I8,
            Self::U16(_) => GgufValueType::U16,
            Self::I16(_) => GgufValueType::I16,
            Self::U32(_) => GgufValueType::U32,
            Self::I32(_) => GgufValueType::I32,
            Self::F32(_) => GgufValueType::F32,
            Self::Bool(_) => GgufValueType::Bool,
            Self::String(_) => GgufValueType::String,
            Self::Array { .. } => GgufValueType::Array,
            Self::U64(_) => GgufValueType::U64,
            Self::I64(_) => GgufValueType::I64,
            Self::F64(_) => GgufValueType::F64,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(value) => Some(*value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(value) => Some(u64::from(*value)),
            Self::U16(value) => Some(u64::from(*value)),
            Self::U32(value) => Some(u64::from(*value)),
            Self::U64(value) => Some(*value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(value) => Some(*value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F32(value) => Some(f64::from(*value)),
            Self::F64(value) => Some(*value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array { values, .. } => Some(values),
            _ => None,
        }
    }

    #[must_use]
    pub fn array_element_type(&self) -> Option<GgufValueType> {
        match self {
            Self::Array { element_type, .. } => Some(*element_type),
            _ => None,
        }
    }
}

/// Immutable GGUF key-value metadata.
#[derive(Debug, Clone, Default)]
pub struct GgufMetadata {
    values: BTreeMap<String, GgufValue>,
}

impl GgufMetadata {
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[must_use]
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.values.get(key)
    }

    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &GgufValue)> {
        self.values.iter().map(|(key, value)| (key.as_str(), value))
    }

    #[must_use]
    pub fn string(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(GgufValue::as_str)
    }

    #[must_use]
    pub fn u32(&self, key: &str) -> Option<u32> {
        self.get(key).and_then(GgufValue::as_u32)
    }

    #[must_use]
    pub fn u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(GgufValue::as_u64)
    }

    #[must_use]
    pub fn f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(GgufValue::as_f32)
    }

    #[must_use]
    pub fn f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(GgufValue::as_f64)
    }

    #[must_use]
    pub fn bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(GgufValue::as_bool)
    }

    #[must_use]
    pub fn array(&self, key: &str) -> Option<&[GgufValue]> {
        self.get(key).and_then(GgufValue::as_array)
    }
}

/// The GGML storage types supported by the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl GgufTensorType {
    pub const ALL: [Self; 12] = [
        Self::F32,
        Self::F16,
        Self::Bf16,
        Self::Q4_0,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
        }
    }

    fn from_tag(tag: u32) -> Result<Self> {
        match tag {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            30 => Ok(Self::Bf16),
            _ => bail!(
                "unsupported GGML tensor type {tag}; this runtime currently supports F32 (0), F16 (1), Q4_0 (2), Q5_0 (6), Q5_1 (7), Q8_0 (8), Q2_K (10), Q3_K (11), Q4_K (12), Q5_K (13), Q6_K (14), and BF16 (30)"
            ),
        }
    }

    #[must_use]
    pub fn candle_dtype(self) -> Option<DType> {
        match self {
            Self::F32 => Some(DType::F32),
            Self::F16 => Some(DType::F16),
            Self::Bf16 => Some(DType::BF16),
            Self::Q4_0
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K => None,
        }
    }

    #[must_use]
    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Q4_0
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q2K
                | Self::Q3K
                | Self::Q4K
                | Self::Q5K
                | Self::Q6K
        )
    }

    fn block_size(self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    fn type_size(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q4_0 => 18,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    fn ggml_dtype(self) -> GgmlDType {
        match self {
            Self::F32 => GgmlDType::F32,
            Self::F16 => GgmlDType::F16,
            Self::Bf16 => GgmlDType::BF16,
            Self::Q4_0 => GgmlDType::Q4_0,
            Self::Q5_0 => GgmlDType::Q5_0,
            Self::Q5_1 => GgmlDType::Q5_1,
            Self::Q8_0 => GgmlDType::Q8_0,
            Self::Q2K => GgmlDType::Q2K,
            Self::Q3K => GgmlDType::Q3K,
            Self::Q4K => GgmlDType::Q4K,
            Self::Q5K => GgmlDType::Q5K,
            Self::Q6K => GgmlDType::Q6K,
        }
    }

    fn byte_len(self, shape: &[usize], name: &str) -> Result<u64> {
        let row_width = shape
            .last()
            .copied()
            .with_context(|| format!("tensor '{name}' has no dimensions"))?;
        let block_size =
            usize::try_from(self.block_size()).context("block size does not fit usize")?;
        ensure!(
            row_width.is_multiple_of(block_size),
            "tensor '{name}' row width {row_width} is not divisible by {self:?} block size {block_size}"
        );
        ensure!(
            !matches!(self, Self::Q5_0 | Self::Q5_1) || (row_width / block_size).is_multiple_of(2),
            "tensor '{name}' row width {row_width} is unsupported for {self:?}; Candle CPU kernels require paired 32-element blocks (a multiple of 64)"
        );
        let elements = checked_element_count(shape, name)?;
        usize::try_from(elements)
            .with_context(|| format!("tensor '{name}' element count does not fit usize"))?;
        checked_mul(
            elements / self.block_size(),
            self.type_size(),
            "tensor byte length",
        )
        .with_context(|| format!("tensor '{name}' byte length overflow"))
    }
}

/// A validated GGUF tensor descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// Candle/PyTorch dimension order. GGML dimensions are reversed while reading.
    pub shape: Vec<usize>,
    pub dtype: GgufTensorType,
    /// Byte offset relative to the beginning of the aligned tensor data section.
    pub offset: u64,
    /// Unpadded tensor payload length.
    pub byte_len: u64,
}

impl TensorInfo {
    pub fn element_count(&self) -> Result<u64> {
        checked_element_count(&self.shape, &self.name)
    }

    pub fn absolute_offset(&self, tensor_data_offset: u64) -> Result<u64> {
        tensor_data_offset
            .checked_add(self.offset)
            .with_context(|| format!("absolute offset overflow for tensor '{}'", self.name))
    }
}

/// Canonical llama.cpp split filename fields. `index` is zero-based, matching
/// the on-disk `split.no` value rather than the one-based filename component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GgufSplitName {
    pub prefix: String,
    pub index: u16,
    pub count: u16,
}

/// Parse `<prefix>-00001-of-000NN.gguf` without accepting abbreviated or
/// non-canonical numeric fields. This is shared by local discovery and the
/// download selector so both boundaries identify exactly the same shard set.
#[must_use]
pub(crate) fn parse_gguf_split_filename(filename: &str) -> Option<GgufSplitName> {
    let stem = filename.strip_suffix(".gguf")?;
    let (indexed_prefix, count_text) = stem.rsplit_once("-of-")?;
    let (prefix, index_text) = indexed_prefix.rsplit_once('-')?;
    if prefix.is_empty()
        || index_text.len() != 5
        || count_text.len() != 5
        || !index_text.bytes().all(|byte| byte.is_ascii_digit())
        || !count_text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let file_index = index_text.parse::<u16>().ok()?;
    let count = count_text.parse::<u16>().ok()?;
    if file_index == 0 || count == 0 || file_index > count {
        return None;
    }
    Some(GgufSplitName {
        prefix: prefix.to_owned(),
        index: file_index - 1,
        count,
    })
}

#[must_use]
pub(crate) fn gguf_split_filename(prefix: &str, index: u16, count: u16) -> String {
    format!("{prefix}-{:05}-of-{count:05}.gguf", u32::from(index) + 1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GgufSplitMetadata {
    index: u16,
    count: u16,
    tensor_count: usize,
}

#[derive(Debug, Clone)]
struct GgufShard {
    path: PathBuf,
    file: Arc<File>,
    alignment: u64,
    tensor_data_offset: u64,
    file_len: u64,
}

struct ParsedGgufShard {
    shard: GgufShard,
    metadata: GgufMetadata,
    tensors: BTreeMap<String, TensorInfo>,
}

/// Public location of one tensor inside a logical GGUF checkpoint.
///
/// Tensor offsets restart at zero in every split file. Callers inspecting raw
/// bytes should use this location instead of combining [`GgufFile::data_offset`]
/// with an arbitrary tensor from a later shard.
#[derive(Debug, Clone, Copy)]
pub struct GgufTensorLocation<'a> {
    pub shard_index: usize,
    pub path: &'a Path,
    pub alignment: u64,
    pub data_offset: u64,
    pub tensor: &'a TensorInfo,
}

impl GgufTensorLocation<'_> {
    pub fn absolute_offset(&self) -> Result<u64> {
        self.tensor.absolute_offset(self.data_offset)
    }
}

/// Parsed GGUF v3 metadata and tensor index. A split checkpoint is exposed as
/// one logical file while each tensor retains its owning shard internally.
#[derive(Debug, Clone)]
pub struct GgufFile {
    shards: Arc<Vec<GgufShard>>,
    version: u32,
    metadata: Arc<GgufMetadata>,
    tensors: Arc<BTreeMap<String, TensorInfo>>,
    tensor_shards: Arc<BTreeMap<String, usize>>,
}

impl GgufFile {
    /// Parse and validate one logical GGUF v3 checkpoint without mapping tensor
    /// payloads. A canonical first shard automatically discovers its siblings.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let primary = open_gguf_shard(&path)?;
        let split = parse_split_metadata(&primary.metadata)?;
        let shard_paths = checkpoint_shard_paths(&path, split)?;

        let expected_split = split;
        let primary_metadata = primary.metadata;
        let mut shards = Vec::new();
        shards
            .try_reserve_exact(shard_paths.len())
            .context("failed to reserve GGUF shard table")?;
        let mut tensors = BTreeMap::new();
        let mut tensor_shards = BTreeMap::new();

        validate_aggregate_tensor_count(
            0,
            primary.tensors.len(),
            expected_split.map(|split| split.tensor_count),
        )?;
        merge_parsed_shard(
            primary.shard,
            primary.tensors,
            0,
            &mut shards,
            &mut tensors,
            &mut tensor_shards,
        )?;
        for (shard_index, shard_path) in shard_paths.iter().enumerate().skip(1) {
            let parsed = open_gguf_shard(shard_path).with_context(|| {
                format!(
                    "failed to load GGUF shard {} of {}",
                    shard_index + 1,
                    shard_paths.len()
                )
            })?;
            let actual = parse_split_metadata(&parsed.metadata)?.with_context(|| {
                format!(
                    "GGUF shard {} is missing required split metadata",
                    shard_path.display()
                )
            })?;
            let expected = expected_split.context("multiple GGUF paths require split metadata")?;
            ensure!(
                usize::from(actual.index) == shard_index,
                "GGUF shard {} declares split.no={}, expected {shard_index}",
                shard_path.display(),
                actual.index
            );
            ensure!(
                actual.count == expected.count,
                "GGUF shard {} declares split.count={}, expected {}",
                shard_path.display(),
                actual.count,
                expected.count
            );
            ensure!(
                actual.tensor_count == expected.tensor_count,
                "GGUF shard {} declares split.tensors.count={}, expected {}",
                shard_path.display(),
                actual.tensor_count,
                expected.tensor_count
            );
            validate_secondary_metadata(&primary_metadata, &parsed.metadata, shard_path)?;
            validate_aggregate_tensor_count(
                tensors.len(),
                parsed.tensors.len(),
                Some(expected.tensor_count),
            )?;
            merge_parsed_shard(
                parsed.shard,
                parsed.tensors,
                shard_index,
                &mut shards,
                &mut tensors,
                &mut tensor_shards,
            )?;
        }

        if let Some(split) = expected_split {
            ensure!(
                split.tensor_count == tensors.len(),
                "GGUF split checkpoint declares {} tensors, but {} unique tensors were loaded",
                split.tensor_count,
                tensors.len()
            );
        }
        validate_quantization_metadata(&primary_metadata, &tensors)?;
        Ok(Self {
            shards: Arc::new(shards),
            version: GGUF_VERSION,
            metadata: Arc::new(primary_metadata),
            tensors: Arc::new(tensors),
            tensor_shards: Arc::new(tensor_shards),
        })
    }

    #[must_use]
    /// Return the canonical first shard path. Use [`Self::tensor_location`] for
    /// the physical file that owns a particular tensor.
    pub fn path(&self) -> &Path {
        &self.shards[0].path
    }

    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Return every physical shard path in canonical split order.
    ///
    /// A reproducible benchmark must fingerprint the complete logical
    /// checkpoint rather than only the first GGUF shard.
    #[must_use]
    pub fn shard_paths(&self) -> impl ExactSizeIterator<Item = &Path> {
        self.shards.iter().map(|shard| shard.path.as_path())
    }

    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }

    #[must_use]
    /// Return the canonical first shard alignment.
    pub fn alignment(&self) -> u64 {
        self.shards[0].alignment
    }

    #[must_use]
    /// Return the canonical first shard data offset. Tensor offsets in later
    /// shards must instead be resolved through [`Self::tensor_location`].
    pub fn data_offset(&self) -> u64 {
        self.shards[0].tensor_data_offset
    }

    #[must_use]
    pub fn metadata(&self) -> &GgufMetadata {
        self.metadata.as_ref()
    }

    #[must_use]
    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        self.tensors.as_ref()
    }

    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    #[must_use]
    pub fn tensor_location(&self, name: &str) -> Option<GgufTensorLocation<'_>> {
        let tensor = self.tensors.get(name)?;
        let shard_index = *self.tensor_shards.get(name)?;
        let shard = self.shards.get(shard_index)?;
        Some(GgufTensorLocation {
            shard_index,
            path: &shard.path,
            alignment: shard.alignment,
            data_offset: shard.tensor_data_offset,
            tensor,
        })
    }

    /// Read a small rank-one F32 tensor into owned host memory.
    ///
    /// Architecture adapters use this for execution metadata represented as a
    /// GGUF tensor, such as `rope_freqs.weight`. Model matrices continue to use
    /// the mmap backend.
    pub(crate) fn read_f32_vector(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensor(name)
            .with_context(|| format!("required GGUF tensor `{name}` is missing"))?;
        ensure!(
            info.dtype == GgufTensorType::F32,
            "GGUF auxiliary tensor `{name}` must be F32, found {:?}",
            info.dtype
        );
        ensure!(
            info.shape.len() == 1,
            "GGUF auxiliary tensor `{name}` must be rank one, found shape {:?}",
            info.shape
        );
        let elements = info.element_count()?;
        ensure!(
            elements <= MAX_AUXILIARY_F32_ELEMENTS,
            "GGUF auxiliary tensor `{name}` has {elements} elements, exceeding the limit {MAX_AUXILIARY_F32_ELEMENTS}"
        );
        let expected_bytes = elements
            .checked_mul(4)
            .context("GGUF auxiliary tensor byte length overflow")?;
        ensure!(
            info.byte_len == expected_bytes,
            "GGUF auxiliary tensor `{name}` has {} bytes, expected {expected_bytes}",
            info.byte_len
        );

        let byte_len = usize::try_from(info.byte_len)
            .context("GGUF auxiliary tensor byte length does not fit usize")?;
        let mut bytes = vec![0_u8; byte_len];
        let location = self
            .tensor_location(name)
            .with_context(|| format!("GGUF tensor `{name}` has no owning shard"))?;
        let shard = &self.shards[location.shard_index];
        read_exact_at(shard.file.as_ref(), &mut bytes, location.absolute_offset()?)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunks are four bytes")))
            .collect())
    }

    /// Create a memory-mapped Candle backend.
    ///
    /// # Safety
    ///
    /// The checkpoint files must not have been modified in place since this
    /// [`GgufFile`] was parsed, and their underlying files must remain
    /// unmodified and untruncated while the returned backend or any model
    /// loaded from it remains alive. Replacing or unlinking a path is safe
    /// because retained handles and mappings continue to reference the
    /// original inode. Metal may keep quantized weights file-backed for the
    /// complete model lifetime.
    pub unsafe fn backend(&self, config: GgufBackendConfig) -> Result<GgufBackend> {
        let mut backend_shards = Vec::new();
        backend_shards
            .try_reserve_exact(self.shards.len())
            .context("failed to reserve GGUF backend shard table")?;
        let mut tensor_data_offsets = Vec::new();
        tensor_data_offsets
            .try_reserve_exact(self.shards.len())
            .context("failed to reserve GGUF data-offset table")?;
        for shard in self.shards.iter() {
            let current_len = shard
                .file
                .metadata()
                .with_context(|| format!("failed to stat GGUF file {}", shard.path.display()))?
                .len();
            ensure!(
                current_len == shard.file_len,
                "GGUF file size changed after parsing for {}: expected {}, found {current_len}",
                shard.path.display(),
                shard.file_len
            );
            backend_shards.push(BackendShard {
                file: Arc::clone(&shard.file),
                file_len: current_len,
                #[cfg(feature = "metal")]
                file_backed_metal: Mutex::new(HashMap::new()),
            });
            tensor_data_offsets.push(shard.tensor_data_offset);
        }
        Ok(GgufBackend {
            shards: Arc::new(backend_shards),
            tensor_data_offsets: Arc::new(tensor_data_offsets),
            tensors: Arc::clone(&self.tensors),
            tensor_shards: Arc::clone(&self.tensor_shards),
            config,
        })
    }
}

fn checkpoint_shard_paths(path: &Path, split: Option<GgufSplitMetadata>) -> Result<Vec<PathBuf>> {
    let filename_split = path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(parse_gguf_split_filename);
    if split.is_none_or(|metadata| metadata.count <= 1)
        && filename_split.is_some_and(|filename| filename.count > 1)
    {
        bail!(
            "GGUF filename {} declares an active shard family, but its metadata does not",
            path.display()
        );
    }
    let Some(split) = split.filter(|split| split.count > 1) else {
        return Ok(vec![path.to_path_buf()]);
    };
    ensure!(
        split.index == 0,
        "GGUF split {} has split.no={}; load the first shard instead",
        path.display(),
        split.index
    );
    ensure!(
        split.count <= MAX_GGUF_SHARDS,
        "GGUF declares {} shards, exceeding the supported safety limit {MAX_GGUF_SHARDS}",
        split.count
    );
    expected_split_paths(path, split.count)
}

fn open_gguf_shard(path: &Path) -> Result<ParsedGgufShard> {
    let file = open_regular_gguf_file(path)?;
    let file_len = file
        .metadata()
        .with_context(|| format!("failed to stat GGUF file {}", path.display()))?
        .len();
    let parsed = parse_gguf(BufReader::new(&file), file_len)
        .with_context(|| format!("invalid GGUF file {}", path.display()))?;
    Ok(ParsedGgufShard {
        shard: GgufShard {
            path: path.to_path_buf(),
            file: Arc::new(file),
            alignment: parsed.alignment,
            tensor_data_offset: parsed.tensor_data_offset,
            file_len,
        },
        metadata: parsed.metadata,
        tensors: parsed.tensors,
    })
}

fn open_regular_gguf_file(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        // Prevent a canonical sibling replaced by a FIFO from blocking the
        // loader before it can verify the opened handle's file type.
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .with_context(|| format!("failed to open GGUF file {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to stat GGUF file {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file(),
        "GGUF shard must be a regular file: {}",
        path.display()
    );
    Ok(file)
}

fn merge_parsed_shard(
    shard: GgufShard,
    shard_tensors: BTreeMap<String, TensorInfo>,
    shard_index: usize,
    shards: &mut Vec<GgufShard>,
    tensors: &mut BTreeMap<String, TensorInfo>,
    tensor_shards: &mut BTreeMap<String, usize>,
) -> Result<()> {
    for (name, info) in shard_tensors {
        ensure!(
            !tensors.contains_key(&name),
            "GGUF tensor `{name}` is duplicated across split files"
        );
        tensor_shards.insert(name.clone(), shard_index);
        tensors.insert(name, info);
    }
    shards.push(shard);
    Ok(())
}

fn validate_aggregate_tensor_count(
    current: usize,
    incoming: usize,
    declared_total: Option<usize>,
) -> Result<()> {
    let next = current
        .checked_add(incoming)
        .context("GGUF aggregate tensor count overflow")?;
    ensure!(
        u64::try_from(next).unwrap_or(u64::MAX) <= MAX_COLLECTION_ENTRIES,
        "GGUF checkpoint contains more than {MAX_COLLECTION_ENTRIES} tensors"
    );
    if let Some(declared_total) = declared_total {
        ensure!(
            next <= declared_total,
            "GGUF split checkpoint contains at least {next} tensors, exceeding declared split.tensors.count={declared_total}"
        );
    }
    Ok(())
}

fn parse_split_metadata(metadata: &GgufMetadata) -> Result<Option<GgufSplitMetadata>> {
    let has_split_metadata = metadata.iter().any(|(key, _)| key.starts_with("split."));
    if !has_split_metadata {
        return Ok(None);
    }
    let index = required_split_u16(metadata, SPLIT_NO_KEY)?;
    let count = required_split_u16(metadata, SPLIT_COUNT_KEY)?;
    let tensor_count = match metadata.get(SPLIT_TENSORS_COUNT_KEY) {
        Some(GgufValue::I32(value)) => usize::try_from(*value).with_context(|| {
            format!("GGUF metadata `{SPLIT_TENSORS_COUNT_KEY}` must be non-negative")
        })?,
        Some(value) => bail!(
            "GGUF metadata `{SPLIT_TENSORS_COUNT_KEY}` must be INT32, got {:?}",
            value.value_type()
        ),
        None => bail!("required GGUF metadata `{SPLIT_TENSORS_COUNT_KEY}` is missing"),
    };
    if count == 0 {
        ensure!(
            index == 0,
            "merged GGUF metadata `{SPLIT_NO_KEY}` must be zero when split.count=0"
        );
    } else {
        ensure!(
            index < count,
            "GGUF metadata `{SPLIT_NO_KEY}`={index} is outside split.count={count}"
        );
    }
    ensure!(
        u64::try_from(tensor_count).unwrap_or(u64::MAX) <= MAX_COLLECTION_ENTRIES,
        "GGUF split checkpoint declares {tensor_count} tensors, exceeding the limit {MAX_COLLECTION_ENTRIES}"
    );
    Ok(Some(GgufSplitMetadata {
        index,
        count,
        tensor_count,
    }))
}

fn validate_secondary_metadata(
    primary: &GgufMetadata,
    secondary: &GgufMetadata,
    path: &Path,
) -> Result<()> {
    for (key, value) in secondary.iter() {
        if key.starts_with("split.") || key == "general.alignment" {
            continue;
        }
        let primary_value = primary.get(key).with_context(|| {
            format!(
                "GGUF subsidiary shard {} contains metadata `{key}` absent from the first shard",
                path.display()
            )
        })?;
        ensure!(
            primary_value == value,
            "GGUF subsidiary shard {} has conflicting metadata `{key}`",
            path.display()
        );
    }
    Ok(())
}

fn validate_quantization_metadata(
    metadata: &GgufMetadata,
    tensors: &BTreeMap<String, TensorInfo>,
) -> Result<()> {
    if tensors.values().any(|tensor| tensor.dtype.is_quantized()) {
        match metadata.get("general.quantization_version") {
            Some(GgufValue::U32(2)) => {}
            Some(GgufValue::U32(version)) => {
                bail!("unsupported GGUF quantization version {version}; expected version 2")
            }
            Some(value) => bail!(
                "general.quantization_version must be U32, found {:?}",
                value.value_type()
            ),
            None => bail!("quantized GGUF tensors require general.quantization_version"),
        }
    }
    Ok(())
}

fn required_split_u16(metadata: &GgufMetadata, key: &str) -> Result<u16> {
    match metadata.get(key) {
        Some(GgufValue::U16(value)) => Ok(*value),
        Some(value) => bail!(
            "GGUF metadata `{key}` must be UINT16, got {:?}",
            value.value_type()
        ),
        None => bail!("required GGUF metadata `{key}` is missing"),
    }
}

fn expected_split_paths(first: &Path, count: u16) -> Result<Vec<PathBuf>> {
    let filename = first
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("GGUF split filename must be UTF-8: {}", first.display()))?;
    let parsed = parse_gguf_split_filename(filename).with_context(|| {
        format!(
            "GGUF split filename must match `<prefix>-00001-of-{count:05}.gguf`: {}",
            first.display()
        )
    })?;
    ensure!(
        parsed.index == 0 && parsed.count == count,
        "GGUF first shard filename `{filename}` does not match split.no=0 and split.count={count}"
    );
    let parent = first.parent().unwrap_or_else(|| Path::new(""));
    let mut paths = Vec::new();
    paths
        .try_reserve_exact(usize::from(count))
        .context("failed to reserve GGUF shard paths")?;
    for index in 0..count {
        paths.push(parent.join(gguf_split_filename(&parsed.prefix, index, count)));
    }
    Ok(paths)
}

#[cfg(unix)]
fn read_exact_at(file: &File, bytes: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;

    file.read_exact_at(bytes, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut bytes: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;

    while !bytes.is_empty() {
        let read = file.seek_read(bytes, offset)?;
        if read == 0 {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        bytes = &mut bytes[read..];
    }
    Ok(())
}

/// Head counts needed to undo llama.cpp's Q/K conversion permutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaQkConfig {
    pub query_heads: usize,
    pub kv_heads: usize,
}

/// GGUF tensor routing and compatibility options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufBackendConfig {
    /// Translate Hugging Face decoder paths to canonical GGUF tensor names.
    pub map_hf_names: bool,
    /// Undo llama.cpp's Llama Q/K permutation. Qwen2 must leave this `None`.
    pub llama_qk: Option<LlamaQkConfig>,
    /// Route a missing `output.weight` to `token_embd.weight` for tied models.
    pub fallback_output_to_embedding: bool,
}

impl Default for GgufBackendConfig {
    fn default() -> Self {
        Self {
            map_hf_names: true,
            llama_qk: None,
            fallback_output_to_embedding: false,
        }
    }
}

/// A memory-mapped Candle backend for validated GGUF tensors.
pub struct GgufBackend {
    shards: Arc<Vec<BackendShard>>,
    tensor_data_offsets: Arc<Vec<u64>>,
    tensors: Arc<BTreeMap<String, TensorInfo>>,
    tensor_shards: Arc<BTreeMap<String, usize>>,
    config: GgufBackendConfig,
}

struct BackendShard {
    file: Arc<File>,
    file_len: u64,
    #[cfg(feature = "metal")]
    file_backed_metal:
        Mutex<HashMap<candle_core::metal_backend::DeviceId, Option<Arc<FileBackedMetalShard>>>>,
}

struct TensorPayloadLocation<'a> {
    resolved_name: String,
    info: &'a TensorInfo,
    shard_index: usize,
    file_offset: u64,
}

struct RawTensorPayload<'a> {
    resolved_name: String,
    info: &'a TensorInfo,
    mapping: Mmap,
}

impl RawTensorPayload<'_> {
    fn bytes(&self) -> &[u8] {
        self.mapping.as_ref()
    }
}

/// A matrix loaded without erasing whether its GGUF payload is block-quantized.
#[derive(Debug, Clone)]
pub enum GgufMatrix {
    Dense(Tensor),
    Quantized(Arc<QTensor>),
    #[cfg(feature = "metal")]
    FileBackedQuantized(Arc<FileBackedQuantizedTensor>),
}

impl std::fmt::Debug for GgufBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GgufBackend")
            .field(
                "source_bytes",
                &self.shards.iter().map(|shard| shard.file_len).sum::<u64>(),
            )
            .field("shard_count", &self.shards.len())
            .field("tensor_data_offsets", &self.tensor_data_offsets)
            .field("tensor_count", &self.tensors.len())
            .field("tensor_shard_routes", &self.tensor_shards.len())
            .field("config", &self.config)
            .finish()
    }
}

impl GgufBackend {
    fn resolve_name(&self, requested: &str) -> String {
        let mapped = if self.config.map_hf_names {
            hf_to_gguf_name(requested).unwrap_or_else(|| requested.to_owned())
        } else {
            requested.to_owned()
        };
        if let Some(without_weight) = mapped.strip_suffix(".ffn_norm.weight") {
            let without_weight = format!("{without_weight}.ffn_norm");
            if !self.tensors.contains_key(&mapped) && self.tensors.contains_key(&without_weight) {
                return without_weight;
            }
        }
        if mapped == "output.weight"
            && self.config.fallback_output_to_embedding
            && !self.tensors.contains_key(&mapped)
        {
            "token_embd.weight".to_owned()
        } else {
            mapped
        }
    }

    fn locate_tensor<'a>(
        &'a self,
        requested: &str,
    ) -> candle_core::Result<TensorPayloadLocation<'a>> {
        let name = self.resolve_name(requested);
        let info = self
            .tensors
            .get(&name)
            .ok_or_else(|| CandleError::CannotFindTensor {
                path: requested.to_owned(),
            })?;
        let shard_index = *self.tensor_shards.get(&name).ok_or_else(|| {
            CandleError::Msg(format!("tensor '{}' has no owning GGUF shard", info.name))
        })?;
        if self.shards.get(shard_index).is_none() {
            return Err(CandleError::Msg(format!(
                "tensor '{}' references missing GGUF shard {shard_index}",
                info.name
            )));
        }
        let tensor_data_offset = *self.tensor_data_offsets.get(shard_index).ok_or_else(|| {
            CandleError::Msg(format!(
                "GGUF shard {shard_index} has no tensor data offset"
            ))
        })?;
        let file_offset = tensor_data_offset.checked_add(info.offset).ok_or_else(|| {
            CandleError::Msg(format!("tensor '{}' mmap start overflows", info.name))
        })?;
        Ok(TensorPayloadLocation {
            resolved_name: name,
            info,
            shard_index,
            file_offset,
        })
    }

    fn map_payload_range(
        &self,
        location: &TensorPayloadLocation<'_>,
        relative_offset: u64,
        byte_len: usize,
    ) -> candle_core::Result<Mmap> {
        let shard = self.shards.get(location.shard_index).ok_or_else(|| {
            CandleError::Msg(format!(
                "tensor '{}' references missing GGUF shard {}",
                location.info.name, location.shard_index
            ))
        })?;
        let byte_len_u64 = u64::try_from(byte_len).map_err(|_| {
            CandleError::Msg(format!(
                "tensor '{}' byte length does not fit u64",
                location.info.name
            ))
        })?;
        let relative_end = relative_offset.checked_add(byte_len_u64).ok_or_else(|| {
            CandleError::Msg(format!(
                "tensor '{}' relative mmap range overflows",
                location.info.name
            ))
        })?;
        if relative_end > location.info.byte_len {
            candle_core::bail!(
                "tensor '{}' relative range {relative_offset}..{relative_end} exceeds its {}-byte payload",
                location.info.name,
                location.info.byte_len
            );
        }
        let start = location
            .file_offset
            .checked_add(relative_offset)
            .ok_or_else(|| {
                CandleError::Msg(format!(
                    "tensor '{}' mmap start overflows",
                    location.info.name
                ))
            })?;
        let end = start.checked_add(byte_len_u64).ok_or_else(|| {
            CandleError::Msg(format!(
                "tensor '{}' mmap range overflow",
                location.info.name
            ))
        })?;
        if end > shard.file_len {
            candle_core::bail!(
                "tensor '{}' range {start}..{end} is outside its {}-byte GGUF shard",
                location.info.name,
                shard.file_len
            );
        }
        // Map only this tensor's payload. CPU and Metal constructors copy the
        // bytes into owned storage before returning, so dropping the payload
        // after a load immediately unmaps cold source pages instead of keeping
        // an entire GGUF resident alongside the completed model.
        // SAFETY: backend construction requires every retained file handle to
        // remain immutable and untruncated for the backend lifetime.
        unsafe {
            MmapOptions::new()
                .offset(start)
                .len(byte_len)
                .map(shard.file.as_ref())
        }
        .map_err(|error| {
            CandleError::Msg(format!(
                "failed to mmap tensor '{}' from GGUF shard {} at {start}..{end}: {error}",
                location.info.name, location.shard_index
            ))
        })
    }

    fn raw_tensor<'a>(&'a self, requested: &str) -> candle_core::Result<RawTensorPayload<'a>> {
        let location = self.locate_tensor(requested)?;
        let byte_len = usize::try_from(location.info.byte_len).map_err(|_| {
            CandleError::Msg(format!(
                "tensor '{}' byte length does not fit usize",
                location.info.name
            ))
        })?;
        let mapping = self.map_payload_range(&location, 0, byte_len)?;
        Ok(RawTensorPayload {
            resolved_name: location.resolved_name,
            info: location.info,
            mapping,
        })
    }

    fn llama_qk_heads(&self, name: &str) -> Option<usize> {
        let llama = self.config.llama_qk?;
        if is_attention_projection(name, "attn_q") {
            Some(llama.query_heads)
        } else if is_attention_projection(name, "attn_k") {
            Some(llama.kv_heads)
        } else {
            None
        }
    }

    #[cfg(feature = "metal")]
    fn file_backed_metal_shard(
        &self,
        shard_index: usize,
        device: &candle_core::MetalDevice,
    ) -> candle_core::Result<Option<Arc<FileBackedMetalShard>>> {
        let shard = self.shards.get(shard_index).ok_or_else(|| {
            CandleError::Msg(format!(
                "missing GGUF shard {shard_index} for Metal mapping"
            ))
        })?;
        let mut mappings = shard.file_backed_metal.lock().map_err(|_| {
            CandleError::Msg(format!(
                "GGUF shard {shard_index} Metal mapping cache is poisoned"
            ))
        })?;
        if let Some(mapping) = mappings.get(&device.id()) {
            return Ok(mapping.clone());
        }
        let mapping = FileBackedMetalShard::try_new(shard.file.as_ref(), shard.file_len, device)?
            .map(Arc::new);
        mappings.insert(device.id(), mapping.clone());
        Ok(mapping)
    }

    pub(crate) fn load_tensor(
        &self,
        requested: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let payload = self.raw_tensor(requested)?;
        self.load_dense_payload(payload, dtype, device)
    }

    fn load_dense_payload(
        &self,
        payload: RawTensorPayload<'_>,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let qk_heads = self.llama_qk_heads(payload.resolved_name.as_str());
        let info = payload.info;
        let storage_dtype = info.dtype.candle_dtype().ok_or_else(|| {
            CandleError::Msg(format!(
                "tensor '{}' is {:?}; a dense tensor was requested",
                info.name, info.dtype
            ))
        })?;
        let mut tensor =
            Tensor::from_raw_buffer(payload.bytes(), storage_dtype, &info.shape, device)?;
        // CPU and Metal constructors synchronously copy the source bytes. Do
        // not keep the temporary mapping alive through later permutation or
        // dtype-conversion buffers.
        drop(payload);
        if let Some(heads) = qk_heads {
            tensor = inverse_permute_llama_qk(tensor, heads)?;
        }
        tensor.to_dtype(dtype)
    }

    pub(crate) fn load_matrix(
        &self,
        requested: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<GgufMatrix> {
        const QUANT_BLOCK_ALIGNMENT: usize = 2;

        let payload = self.raw_tensor(requested)?;
        let info = payload.info;
        if !info.dtype.is_quantized() {
            return self
                .load_dense_payload(payload, dtype, device)
                .map(GgufMatrix::Dense);
        }
        let qk_heads = self.llama_qk_heads(payload.resolved_name.as_str());
        let raw = payload.bytes();
        if info.shape.len() != 2 {
            candle_core::bail!(
                "quantized tensor '{}' must be a rank-2 matrix, found shape {:?}",
                info.name,
                info.shape
            );
        }
        let expected_len = usize::try_from(info.byte_len).map_err(|_| {
            CandleError::Msg(format!(
                "tensor '{}' byte length does not fit usize",
                info.name
            ))
        })?;
        if raw.len() != expected_len {
            candle_core::bail!(
                "tensor '{}' payload length mismatch: expected {expected_len}, found {}",
                info.name,
                raw.len()
            );
        }
        let type_size = usize::try_from(info.dtype.type_size()).map_err(CandleError::wrap)?;
        if !raw.len().is_multiple_of(type_size) {
            candle_core::bail!(
                "tensor '{}' payload length {} is not divisible by {:?} type size {type_size}",
                info.name,
                raw.len(),
                info.dtype
            );
        }

        let permuted = qk_heads
            .map(|heads| inverse_permute_quantized_rows(raw, &info.shape, heads))
            .transpose()?;
        let load_storage = |data: &[u8]| -> candle_core::Result<QStorage> {
            if !(data.as_ptr() as usize).is_multiple_of(QUANT_BLOCK_ALIGNMENT) {
                candle_core::bail!(
                    "tensor '{}' payload address is not {QUANT_BLOCK_ALIGNMENT}-byte aligned",
                    info.name
                );
            }
            QStorage::from_data(Cow::Borrowed(data), device, info.dtype.ggml_dtype())
        };
        let storage = if let Some(permuted) = permuted {
            // The row permutation owns its bytes, so the file mapping can be
            // released before the next host-to-device copy.
            drop(payload);
            load_storage(permuted.as_bytes())?
        } else {
            let storage = load_storage(raw)?;
            drop(payload);
            storage
        };
        let tensor = QTensor::new(storage, Shape::from_dims(&info.shape))?;
        Ok(GgufMatrix::Quantized(Arc::new(tensor)))
    }

    /// Preserve a block-quantized matrix and prefer clean file-backed Metal
    /// pages when the shard and tensor meet the no-copy constraints.
    pub(crate) fn load_matrix_preserving(
        &self,
        requested: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<GgufMatrix> {
        #[cfg(feature = "metal")]
        if let Device::Metal(metal) = device {
            let location = self.locate_tensor(requested)?;
            let info = location.info;
            if info.dtype.is_quantized()
                && self
                    .llama_qk_heads(location.resolved_name.as_str())
                    .is_none()
            {
                let byte_len = usize::try_from(info.byte_len).map_err(|_| {
                    CandleError::Msg(format!(
                        "tensor '{}' byte length does not fit usize",
                        info.name
                    ))
                })?;
                if let Some(shard) = self.file_backed_metal_shard(location.shard_index, metal)? {
                    let tensor = FileBackedQuantizedTensor::new(
                        shard,
                        info.dtype.ggml_dtype(),
                        Shape::from_dims(&info.shape),
                        location.file_offset,
                        byte_len,
                    )?;
                    return Ok(GgufMatrix::FileBackedQuantized(Arc::new(tensor)));
                }
            }
        }
        self.load_matrix(requested, dtype, device)
    }

    /// Load a rank-three `[experts, rows, columns]` tensor as independent
    /// rank-two matrices. GGUF stores the expert axis outermost, so each
    /// expert occupies one contiguous payload slice. Keeping those slices as
    /// separate `QTensor`s lets the Apple backends reuse the existing
    /// rank-two quantized matmul path without expanding the whole expert bank.
    pub(crate) fn load_matrix_bank(
        &self,
        requested: &str,
        expected: &Shape,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Vec<GgufMatrix>> {
        const QUANT_BLOCK_ALIGNMENT: usize = 2;

        let location = self.locate_tensor(requested)?;
        let info = location.info;
        let &[expert_count, rows, columns] = info.shape.as_slice() else {
            candle_core::bail!(
                "expert tensor '{}' must have shape [experts, rows, columns], found {:?}",
                info.name,
                info.shape
            );
        };
        if expert_count == 0 {
            candle_core::bail!("expert tensor '{}' has no experts", info.name);
        }
        if info.shape.as_slice() != expected.dims() {
            return Err(CandleError::UnexpectedShape {
                msg: format!("shape mismatch for {requested}"),
                expected: expected.clone(),
                got: Shape::from_dims(&info.shape),
            });
        }
        if expert_count > MAX_ROUTED_EXPERTS {
            candle_core::bail!(
                "expert tensor '{}' declares {expert_count} experts, exceeding the limit {MAX_ROUTED_EXPERTS}",
                info.name
            );
        }

        let expert_byte_len = usize::try_from(
            info.dtype
                .byte_len(&[rows, columns], &info.name)
                .map_err(CandleError::wrap)?,
        )
        .map_err(|_| CandleError::Msg(format!("expert slice for '{}' is too large", info.name)))?;
        let bank_byte_len = expert_byte_len.checked_mul(expert_count).ok_or_else(|| {
            CandleError::Msg(format!(
                "expert payload length overflows for '{}'",
                info.name
            ))
        })?;
        let bank_byte_len_u64 = u64::try_from(bank_byte_len).map_err(|_| {
            CandleError::Msg(format!(
                "expert payload length for '{}' does not fit u64",
                info.name
            ))
        })?;
        if bank_byte_len_u64 != info.byte_len {
            candle_core::bail!(
                "expert tensor '{}' has {} payload bytes, expected {bank_byte_len}",
                info.name,
                info.byte_len
            );
        }

        let storage_dtype = if info.dtype.is_quantized() {
            None
        } else {
            Some(info.dtype.candle_dtype().ok_or_else(|| {
                CandleError::Msg(format!(
                    "expert tensor '{}' has unsupported dense dtype {:?}",
                    info.name, info.dtype
                ))
            })?)
        };

        let mut bank = Vec::new();
        bank.try_reserve_exact(expert_count)
            .map_err(CandleError::wrap)?;
        for expert in 0..expert_count {
            let relative_offset = expert
                .checked_mul(expert_byte_len)
                .and_then(|offset| u64::try_from(offset).ok())
                .ok_or_else(|| {
                    CandleError::Msg(format!(
                        "expert {expert} byte offset overflows for '{}'",
                        info.name
                    ))
                })?;
            let mapping = self.map_payload_range(&location, relative_offset, expert_byte_len)?;
            if let Some(storage_dtype) = storage_dtype {
                let tensor = Tensor::from_raw_buffer(
                    mapping.as_ref(),
                    storage_dtype,
                    &[rows, columns],
                    device,
                )?;
                drop(mapping);
                bank.push(GgufMatrix::Dense(tensor.to_dtype(dtype)?));
            } else {
                let data = mapping.as_ref();
                if !(data.as_ptr() as usize).is_multiple_of(QUANT_BLOCK_ALIGNMENT) {
                    candle_core::bail!(
                        "expert {expert} of tensor '{}' is not {QUANT_BLOCK_ALIGNMENT}-byte aligned",
                        info.name
                    );
                }
                let storage =
                    QStorage::from_data(Cow::Borrowed(data), device, info.dtype.ggml_dtype())?;
                drop(mapping);
                let tensor = QTensor::new(storage, (rows, columns))?;
                bank.push(GgufMatrix::Quantized(Arc::new(tensor)));
            }
        }
        Ok(bank)
    }

    /// Preserve each expert slice while sharing one no-copy shard buffer.
    pub(crate) fn load_matrix_bank_preserving(
        &self,
        requested: &str,
        expected: &Shape,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Vec<GgufMatrix>> {
        #[cfg(feature = "metal")]
        if let Device::Metal(metal) = device {
            let location = self.locate_tensor(requested)?;
            let info = location.info;
            if info.dtype.is_quantized() {
                let &[expert_count, rows, columns] = info.shape.as_slice() else {
                    candle_core::bail!(
                        "expert tensor '{}' must have shape [experts, rows, columns], found {:?}",
                        info.name,
                        info.shape
                    );
                };
                if info.shape.as_slice() != expected.dims() {
                    return Err(CandleError::UnexpectedShape {
                        msg: format!("shape mismatch for {requested}"),
                        expected: expected.clone(),
                        got: Shape::from_dims(&info.shape),
                    });
                }
                if expert_count == 0 || expert_count > MAX_ROUTED_EXPERTS {
                    candle_core::bail!(
                        "expert tensor '{}' declares invalid expert count {expert_count}",
                        info.name
                    );
                }
                let expert_byte_len = usize::try_from(
                    info.dtype
                        .byte_len(&[rows, columns], &info.name)
                        .map_err(CandleError::wrap)?,
                )
                .map_err(|_| {
                    CandleError::Msg(format!("expert slice for '{}' is too large", info.name))
                })?;
                let bank_byte_len = expert_byte_len.checked_mul(expert_count).ok_or_else(|| {
                    CandleError::Msg(format!(
                        "expert payload length overflows for '{}'",
                        info.name
                    ))
                })?;
                if u64::try_from(bank_byte_len).ok() != Some(info.byte_len) {
                    candle_core::bail!(
                        "expert tensor '{}' has {} payload bytes, expected {bank_byte_len}",
                        info.name,
                        info.byte_len
                    );
                }
                if let Some(shard) = self.file_backed_metal_shard(location.shard_index, metal)? {
                    let mut bank = Vec::new();
                    bank.try_reserve_exact(expert_count)
                        .map_err(CandleError::wrap)?;
                    for expert in 0..expert_count {
                        let relative_offset =
                            expert.checked_mul(expert_byte_len).ok_or_else(|| {
                                CandleError::Msg(format!(
                                    "expert {expert} byte offset overflows for '{}'",
                                    info.name
                                ))
                            })?;
                        let absolute_offset = location
                            .file_offset
                            .checked_add(u64::try_from(relative_offset).map_err(CandleError::wrap)?)
                            .ok_or_else(|| {
                                CandleError::Msg(format!(
                                    "expert {expert} absolute offset overflows for '{}'",
                                    info.name
                                ))
                            })?;
                        let tensor = FileBackedQuantizedTensor::new(
                            Arc::clone(&shard),
                            info.dtype.ggml_dtype(),
                            (rows, columns),
                            absolute_offset,
                            expert_byte_len,
                        )?;
                        bank.push(GgufMatrix::FileBackedQuantized(Arc::new(tensor)));
                    }
                    return Ok(bank);
                }
            }
        }
        self.load_matrix_bank(requested, expected, dtype, device)
    }
}

impl SimpleBackend for GgufBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load_tensor(name, dtype, device)?;
        if tensor.shape() != &shape {
            return Err(CandleError::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: shape,
                got: tensor.shape().clone(),
            });
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        self.load_tensor(name, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(&self.resolve_name(name))
    }
}

/// Map the decoder paths requested by the shared Hugging Face-style runtime to
/// canonical GGUF tensor names.
#[must_use]
pub fn hf_to_gguf_name(name: &str) -> Option<String> {
    let fixed = match name {
        "model.embed_tokens.weight" => Some("token_embd.weight"),
        "model.norm.weight" => Some("output_norm.weight"),
        "model.embedding_norm.weight" => Some("token_embd_norm.weight"),
        "lm_head.weight" => Some("output.weight"),
        _ => None,
    };
    if let Some(mapped) = fixed {
        return Some(mapped.to_owned());
    }

    let layer = name.strip_prefix("model.layers.")?;
    let (index, suffix) = layer.split_once('.')?;
    let index = index.parse::<usize>().ok()?;
    let suffix = match suffix {
        "input_layernorm.weight" | "operator_norm.weight" | "attn_norm.weight" => {
            "attn_norm.weight"
        }
        "post_attention_layernorm.weight"
        | "pre_feedforward_layernorm.weight"
        | "ffn_norm.weight" => "ffn_norm.weight",
        "attn_post_norm.weight" | "post_attention_norm.weight" => "post_attention_norm.weight",
        "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.q_proj.bias" => "attn_q.bias",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.k_proj.bias" => "attn_k.bias",
        "self_attn.q_norm.weight" | "self_attn.q_layernorm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" | "self_attn.k_layernorm.weight" => "attn_k_norm.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.v_proj.bias" => "attn_v.bias",
        "self_attn.o_proj.weight" | "self_attn.out_proj.weight" => "attn_output.weight",
        "self_attn.o_proj.bias" => "attn_output.bias",
        "mlp.gate_proj.weight" | "feed_forward.w1.weight" => "ffn_gate.weight",
        "mlp.gate_proj.bias" => "ffn_gate.bias",
        "mlp.up_proj.weight" | "feed_forward.w3.weight" => "ffn_up.weight",
        "mlp.up_proj.bias" => "ffn_up.bias",
        "mlp.down_proj.weight" | "feed_forward.w2.weight" => "ffn_down.weight",
        "mlp.down_proj.bias" => "ffn_down.bias",
        "mlp.router.weight" => "ffn_gate_inp.weight",
        "mlp.experts.gate_proj.weight" => "ffn_gate_exps.weight",
        "mlp.experts.up_proj.weight" => "ffn_up_exps.weight",
        "mlp.experts.down_proj.weight" => "ffn_down_exps.weight",
        "conv.in_proj.weight" | "shortconv.in_proj.weight" => "shortconv.in_proj.weight",
        "conv.conv.weight" | "shortconv.conv.weight" => "shortconv.conv.weight",
        "conv.out_proj.weight" | "shortconv.out_proj.weight" => "shortconv.out_proj.weight",
        "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
        "linear_attn.in_proj_z.weight" => "attn_gate.weight",
        "linear_attn.conv1d.weight" | "mamba.conv1d.weight" => "ssm_conv1d.weight",
        "linear_attn.dt_bias" | "linear_attn.dt_proj.bias" | "mamba.dt_bias" => "ssm_dt.bias",
        "linear_attn.A_log" | "mamba.A_log" => "ssm_a",
        "linear_attn.in_proj_b.weight" => "ssm_beta.weight",
        "linear_attn.in_proj_a.weight" => "ssm_alpha.weight",
        "linear_attn.norm.weight" | "mamba.norm.weight" => "ssm_norm.weight",
        "linear_attn.out_proj.weight" | "mamba.out_proj.weight" => "ssm_out.weight",
        "mamba.in_proj.weight" => "ssm_in.weight",
        "mamba.in_proj.bias" => "ssm_in.bias",
        "mamba.conv1d.bias" => "ssm_conv1d.bias",
        "mamba.D" => "ssm_d",
        "mamba.out_proj.bias" => "ssm_out.bias",
        _ => return None,
    };
    Some(format!("blk.{index}.{suffix}"))
}

/// Undo the permutation used by llama.cpp when converting Llama Q/K weights or
/// biases from Hugging Face to GGUF.
pub fn inverse_permute_llama_qk(tensor: Tensor, heads: usize) -> candle_core::Result<Tensor> {
    let original = tensor.dims().to_vec();
    if original.is_empty() || heads == 0 || !original[0].is_multiple_of(heads) {
        candle_core::bail!(
            "cannot inverse-permute Llama Q/K shape {:?} across {heads} heads",
            original
        );
    }
    let head_dim = original[0] / heads;
    if !head_dim.is_multiple_of(2) {
        candle_core::bail!("Llama Q/K head dimension {head_dim} must be even");
    }
    let mut grouped = Vec::with_capacity(original.len() + 2);
    grouped.extend([heads, head_dim / 2, 2]);
    grouped.extend_from_slice(&original[1..]);
    tensor
        .reshape(Shape::from_dims(&grouped))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape(Shape::from_dims(&original))
}

/// Backing storage used when compressed rows must be reordered before Candle
/// interprets them as block structs. `Vec<u8>` only guarantees byte alignment;
/// `u64` keeps the temporary buffer safely aligned without changing its bytes.
struct AlignedBytes {
    words: Vec<u64>,
    len: usize,
}

impl AlignedBytes {
    fn try_zeroed(len: usize) -> candle_core::Result<Self> {
        let word_count = len.div_ceil(std::mem::size_of::<u64>());
        let mut words = Vec::new();
        words.try_reserve_exact(word_count).map_err(|error| {
            CandleError::Msg(format!(
                "failed to allocate {len} bytes for compressed Q/K row permutation: {error}"
            ))
        })?;
        words.resize(word_count, 0);
        Ok(Self { words, len })
    }

    fn as_bytes(&self) -> &[u8] {
        &bytemuck::cast_slice(&self.words)[..self.len]
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut bytemuck::cast_slice_mut(&mut self.words)[..self.len]
    }
}

fn inverse_permute_quantized_rows(
    raw: &[u8],
    shape: &[usize],
    heads: usize,
) -> candle_core::Result<AlignedBytes> {
    if shape.len() != 2 || heads == 0 || !shape[0].is_multiple_of(heads) {
        candle_core::bail!(
            "cannot inverse-permute compressed Llama Q/K shape {shape:?} across {heads} heads"
        );
    }
    let rows = shape[0];
    let head_dim = rows / heads;
    if !head_dim.is_multiple_of(2) {
        candle_core::bail!("Llama Q/K head dimension {head_dim} must be even");
    }
    if !raw.len().is_multiple_of(rows) {
        candle_core::bail!(
            "compressed Llama Q/K payload length {} is not divisible by {rows} rows",
            raw.len()
        );
    }
    let row_bytes = raw.len() / rows;
    let half = head_dim / 2;
    let mut output = AlignedBytes::try_zeroed(raw.len())?;
    for head in 0..heads {
        for pair in 0..half {
            for lane in 0..2 {
                let source_row = head * head_dim + pair * 2 + lane;
                let destination_row = head * head_dim + lane * half + pair;
                let source = source_row * row_bytes..(source_row + 1) * row_bytes;
                let destination = destination_row * row_bytes..(destination_row + 1) * row_bytes;
                output.as_bytes_mut()[destination].copy_from_slice(&raw[source]);
            }
        }
    }
    Ok(output)
}

fn is_attention_projection(name: &str, projection: &str) -> bool {
    name.split('.').any(|segment| segment == projection)
        && matches!(name.rsplit('.').next(), Some("weight" | "bias"))
}

#[derive(Debug)]
struct ParsedGguf {
    alignment: u64,
    tensor_data_offset: u64,
    metadata: GgufMetadata,
    tensors: BTreeMap<String, TensorInfo>,
}

struct BoundedReader<R> {
    inner: R,
    file_len: u64,
    position: u64,
    decoded_allocation_bytes: u64,
    decoded_allocation_limit: u64,
}

impl<R: Read> BoundedReader<R> {
    fn new(inner: R, file_len: u64) -> Self {
        Self::with_allocation_limit(inner, file_len, MAX_DECODED_ALLOCATION_BYTES)
    }

    fn with_allocation_limit(inner: R, file_len: u64, decoded_allocation_limit: u64) -> Self {
        Self {
            inner,
            file_len,
            position: 0,
            decoded_allocation_bytes: 0,
            decoded_allocation_limit,
        }
    }

    fn charge_decoded_allocation(&mut self, bytes: u64, label: &str) -> Result<()> {
        let next = self
            .decoded_allocation_bytes
            .checked_add(bytes)
            .with_context(|| format!("decoded allocation accounting overflow for {label}"))?;
        ensure!(
            next <= self.decoded_allocation_limit,
            "GGUF decoded allocation budget exceeded while reserving {label}: \
             requested {bytes} bytes with {} of {} bytes already charged",
            self.decoded_allocation_bytes,
            self.decoded_allocation_limit
        );
        self.decoded_allocation_bytes = next;
        Ok(())
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn remaining(&self) -> u64 {
        self.file_len.saturating_sub(self.position)
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        ensure!(
            u64::try_from(N).unwrap_or(u64::MAX) <= self.remaining(),
            "unexpected end of GGUF file at byte {} while reading {N} bytes",
            self.position
        );
        let mut bytes = [0_u8; N];
        self.inner.read_exact(&mut bytes)?;
        self.position = self
            .position
            .checked_add(u64::try_from(N).context("read size does not fit u64")?)
            .context("GGUF reader position overflow")?;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes::<1>()?[0])
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(i8::from_le_bytes(self.bytes::<1>()?))
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes()?))
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.bytes()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.bytes()?))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.bytes()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.bytes()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        ensure!(
            len <= MAX_STRING_LENGTH,
            "GGUF string length {len} exceeds limit {MAX_STRING_LENGTH}"
        );
        ensure!(
            len <= self.remaining(),
            "GGUF string length {len} exceeds {} remaining bytes",
            self.remaining()
        );
        self.charge_decoded_allocation(len, "string storage")?;
        let len = usize::try_from(len).context("GGUF string length does not fit usize")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .context("failed to reserve GGUF string storage")?;
        bytes.resize(len, 0);
        self.inner.read_exact(&mut bytes)?;
        self.position = self
            .position
            .checked_add(u64::try_from(len).context("string length does not fit u64")?)
            .context("GGUF reader position overflow")?;
        String::from_utf8(bytes).context("GGUF string is not valid UTF-8")
    }
}

// Keeping the three on-disk sections in one linear routine makes cursor and
// bounds invariants auditable next to each read.
#[allow(clippy::too_many_lines)]
fn parse_gguf<R: Read>(reader: R, file_len: u64) -> Result<ParsedGguf> {
    let mut reader = BoundedReader::new(reader, file_len);
    ensure!(reader.bytes::<4>()? == GGUF_MAGIC, "bad GGUF magic");
    let version_bytes = reader.bytes::<4>()?;
    let version = u32::from_le_bytes(version_bytes);
    if u32::from_be_bytes(version_bytes) == GGUF_VERSION && version != GGUF_VERSION {
        bail!("big-endian GGUF v3 is not supported");
    }
    ensure!(
        version == GGUF_VERSION,
        "unsupported GGUF version {version}; expected version {GGUF_VERSION}"
    );
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    validate_count(tensor_count, "tensor")?;
    validate_count(metadata_count, "metadata")?;
    let minimum_header = checked_add(
        checked_mul(metadata_count, 13, "metadata header minimum")?,
        checked_mul(tensor_count, 24, "tensor header minimum")?,
        "combined header minimum",
    )?;
    ensure!(
        minimum_header <= reader.remaining(),
        "declared tensor/metadata counts need at least {minimum_header} bytes, but only {} remain",
        reader.remaining()
    );
    reader.charge_decoded_allocation(
        checked_mul(
            metadata_count,
            INDEX_ALLOCATION_BYTES_PER_ENTRY,
            "metadata index allocation",
        )?,
        "metadata index entries",
    )?;
    reader.charge_decoded_allocation(
        checked_mul(
            tensor_count,
            INDEX_ALLOCATION_BYTES_PER_ENTRY,
            "tensor index allocation",
        )?,
        "tensor index entries",
    )?;

    let metadata_capacity =
        usize::try_from(metadata_count).context("GGUF metadata count does not fit usize")?;
    let mut metadata_values = BTreeMap::new();
    // BTreeMap has no fallible reserve API, so the decoded-allocation budget
    // above conservatively charges its nodes before inserting entries.
    for _ in 0..metadata_capacity {
        let key = reader.string()?;
        validate_metadata_key(&key)?;
        let value_type = GgufValueType::from_tag(reader.u32()?)?;
        let value = read_value(&mut reader, value_type, false)?;
        ensure!(
            !metadata_values.contains_key(&key),
            "duplicate GGUF metadata key '{key}'"
        );
        metadata_values.insert(key, value);
    }
    let metadata = GgufMetadata {
        values: metadata_values,
    };

    let tensor_capacity =
        usize::try_from(tensor_count).context("GGUF tensor count does not fit usize")?;
    let mut tensors = BTreeMap::new();
    let mut tensor_order = Vec::new();
    tensor_order
        .try_reserve_exact(tensor_capacity)
        .context("failed to reserve GGUF tensor index")?;
    for _ in 0..tensor_capacity {
        let name = reader.string()?;
        reader.charge_decoded_allocation(
            u64::try_from(name.len())?
                .checked_mul(2)
                .context("tensor name clone allocation overflow")?,
            "tensor name clones",
        )?;
        ensure!(!name.is_empty(), "GGUF tensor name cannot be empty");
        ensure!(
            name.len() <= MAX_TENSOR_NAME_LENGTH,
            "GGUF tensor name '{name}' is {} bytes; maximum is {MAX_TENSOR_NAME_LENGTH}",
            name.len()
        );
        let rank = reader.u32()?;
        ensure!(
            (1..=MAX_TENSOR_DIMS).contains(&rank),
            "tensor '{name}' has rank {rank}; supported ranks are 1..={MAX_TENSOR_DIMS}"
        );
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(usize::try_from(rank).context("tensor rank does not fit usize")?)
            .context("failed to reserve GGUF tensor dimensions")?;
        for _ in 0..rank {
            let dimension = reader.u64()?;
            ensure!(dimension > 0, "tensor '{name}' has a zero-sized dimension");
            shape.push(
                usize::try_from(dimension)
                    .with_context(|| format!("tensor '{name}' dimension does not fit usize"))?,
            );
        }
        shape.reverse();
        let dtype = GgufTensorType::from_tag(reader.u32()?)?;
        let offset = reader.u64()?;
        let byte_len = dtype.byte_len(&shape, &name)?;
        let info = TensorInfo {
            name: name.clone(),
            shape,
            dtype,
            offset,
            byte_len,
        };
        ensure!(
            tensors.insert(name.clone(), info).is_none(),
            "duplicate GGUF tensor name '{name}'"
        );
        tensor_order.push(name);
    }

    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(GgufValue::U32(value)) => u64::from(*value),
        Some(value) => bail!(
            "general.alignment must be U32, found {:?}",
            value.value_type()
        ),
    };
    ensure!(
        alignment >= 8 && alignment % 8 == 0,
        "GGUF alignment {alignment} must be a multiple of 8 bytes"
    );
    let tensor_data_offset = if tensor_count == 0 {
        reader.position()
    } else {
        align_up(reader.position(), alignment).context("GGUF data offset overflow")?
    };
    ensure!(
        tensor_data_offset <= file_len,
        "aligned GGUF tensor data offset {tensor_data_offset} exceeds file length {file_len}"
    );

    let mut expected_offset = 0_u64;
    for name in tensor_order {
        let info = tensors
            .get(&name)
            .with_context(|| format!("internal error: missing tensor '{name}'"))?;
        ensure!(
            info.offset == expected_offset,
            "tensor '{}' has non-canonical offset {}, expected {expected_offset}",
            info.name,
            info.offset
        );
        let start = checked_add(tensor_data_offset, info.offset, "tensor absolute offset")?;
        let end = checked_add(start, info.byte_len, "tensor end offset")?;
        ensure!(
            end <= file_len,
            "tensor '{}' ends at byte {end}, past file length {file_len}",
            info.name
        );
        let padded = align_up(info.byte_len, alignment)
            .with_context(|| format!("padded byte length overflow for tensor '{}'", info.name))?;
        expected_offset = checked_add(expected_offset, padded, "packed tensor data length")?;
    }
    let packed_end = checked_add(tensor_data_offset, expected_offset, "GGUF packed data end")?;
    ensure!(
        packed_end <= file_len,
        "GGUF packed tensor data ends at byte {packed_end}, past file length {file_len}"
    );

    Ok(ParsedGguf {
        alignment,
        tensor_data_offset,
        metadata,
        tensors,
    })
}

fn read_value<R: Read>(
    reader: &mut BoundedReader<R>,
    value_type: GgufValueType,
    inside_array: bool,
) -> Result<GgufValue> {
    match value_type {
        GgufValueType::U8 => Ok(GgufValue::U8(reader.u8()?)),
        GgufValueType::I8 => Ok(GgufValue::I8(reader.i8()?)),
        GgufValueType::U16 => Ok(GgufValue::U16(reader.u16()?)),
        GgufValueType::I16 => Ok(GgufValue::I16(reader.i16()?)),
        GgufValueType::U32 => Ok(GgufValue::U32(reader.u32()?)),
        GgufValueType::I32 => Ok(GgufValue::I32(reader.i32()?)),
        GgufValueType::F32 => Ok(GgufValue::F32(reader.f32()?)),
        GgufValueType::Bool => match reader.u8()? {
            0 => Ok(GgufValue::Bool(false)),
            1 => Ok(GgufValue::Bool(true)),
            value => bail!("invalid GGUF boolean byte {value}; expected 0 or 1"),
        },
        GgufValueType::String => Ok(GgufValue::String(reader.string()?)),
        GgufValueType::Array => {
            ensure!(
                !inside_array,
                "nested GGUF metadata arrays are not supported"
            );
            let element_type = GgufValueType::from_tag(reader.u32()?)?;
            ensure!(
                element_type != GgufValueType::Array,
                "nested GGUF metadata arrays are not supported"
            );
            let len = reader.u64()?;
            validate_count(len, "array element")?;
            let minimum = checked_mul(
                len,
                element_type.minimum_disk_size(),
                "array minimum byte length",
            )?;
            ensure!(
                minimum <= reader.remaining(),
                "GGUF array of {len} {:?} values needs at least {minimum} bytes, but only {} remain",
                element_type,
                reader.remaining()
            );
            reader.charge_decoded_allocation(
                checked_mul(
                    len,
                    u64::try_from(std::mem::size_of::<GgufValue>())
                        .context("GgufValue size does not fit u64")?,
                    "decoded array allocation",
                )?,
                "metadata array elements",
            )?;
            let len = usize::try_from(len).context("GGUF array length does not fit usize")?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(len)
                .context("failed to reserve GGUF array storage")?;
            for _ in 0..len {
                values.push(read_value(reader, element_type, true)?);
            }
            Ok(GgufValue::Array {
                element_type,
                values,
            })
        }
        GgufValueType::U64 => Ok(GgufValue::U64(reader.u64()?)),
        GgufValueType::I64 => Ok(GgufValue::I64(reader.i64()?)),
        GgufValueType::F64 => Ok(GgufValue::F64(reader.f64()?)),
    }
}

fn validate_count(count: u64, label: &str) -> Result<()> {
    ensure!(
        count <= MAX_COLLECTION_ENTRIES,
        "GGUF {label} count {count} exceeds limit {MAX_COLLECTION_ENTRIES}"
    );
    Ok(())
}

fn validate_metadata_key(key: &str) -> Result<()> {
    ensure!(!key.is_empty(), "GGUF metadata key cannot be empty");
    ensure!(
        key.len() <= MAX_METADATA_KEY_LENGTH,
        "GGUF metadata key is {} bytes; maximum is {MAX_METADATA_KEY_LENGTH}",
        key.len()
    );
    ensure!(key.is_ascii(), "GGUF metadata key '{key}' is not ASCII");
    ensure!(
        key.split('.').all(|segment| {
            !segment.is_empty()
                && segment.bytes().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'_' | b'-')
                })
        }),
        "GGUF metadata key '{key}' is not a dot-separated lowercase identifier"
    );
    Ok(())
}

fn checked_element_count(shape: &[usize], name: &str) -> Result<u64> {
    shape.iter().try_fold(1_u64, |elements, dimension| {
        let dimension = u64::try_from(*dimension)
            .with_context(|| format!("tensor '{name}' dimension does not fit u64"))?;
        checked_mul(elements, dimension, "tensor element count")
            .with_context(|| format!("tensor '{name}' element count overflow"))
    })
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_add(right)
        .with_context(|| format!("{label} overflow: {left} + {right}"))
}

fn checked_mul(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_mul(right)
        .with_context(|| format!("{label} overflow: {left} * {right}"))
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    if alignment == 0 {
        return None;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[derive(Clone)]
    struct TestTensor {
        name: &'static str,
        ggml_shape: Vec<u64>,
        dtype: u32,
        data: Vec<u8>,
        offset: Option<u64>,
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn value_tag(value: &GgufValue) -> u32 {
        match value.value_type() {
            GgufValueType::U8 => 0,
            GgufValueType::I8 => 1,
            GgufValueType::U16 => 2,
            GgufValueType::I16 => 3,
            GgufValueType::U32 => 4,
            GgufValueType::I32 => 5,
            GgufValueType::F32 => 6,
            GgufValueType::Bool => 7,
            GgufValueType::String => 8,
            GgufValueType::Array => 9,
            GgufValueType::U64 => 10,
            GgufValueType::I64 => 11,
            GgufValueType::F64 => 12,
        }
    }

    fn type_tag(value_type: GgufValueType) -> u32 {
        match value_type {
            GgufValueType::U8 => 0,
            GgufValueType::I8 => 1,
            GgufValueType::U16 => 2,
            GgufValueType::I16 => 3,
            GgufValueType::U32 => 4,
            GgufValueType::I32 => 5,
            GgufValueType::F32 => 6,
            GgufValueType::Bool => 7,
            GgufValueType::String => 8,
            GgufValueType::Array => 9,
            GgufValueType::U64 => 10,
            GgufValueType::I64 => 11,
            GgufValueType::F64 => 12,
        }
    }

    fn push_value(bytes: &mut Vec<u8>, value: &GgufValue) {
        match value {
            GgufValue::U8(value) => bytes.push(*value),
            GgufValue::I8(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            GgufValue::U16(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            GgufValue::I16(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            GgufValue::U32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            GgufValue::I32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            GgufValue::F32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            GgufValue::Bool(value) => bytes.push(u8::from(*value)),
            GgufValue::String(value) => push_string(bytes, value),
            GgufValue::Array {
                element_type,
                values,
            } => {
                bytes.extend_from_slice(&type_tag(*element_type).to_le_bytes());
                bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in values {
                    push_value(bytes, value);
                }
            }
            GgufValue::U64(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            GgufValue::I64(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            GgufValue::F64(value) => bytes.extend_from_slice(&value.to_le_bytes()),
        }
    }

    fn make_gguf(metadata: &[(&str, GgufValue)], tensors: &[TestTensor]) -> Vec<u8> {
        let alignment = metadata
            .iter()
            .find_map(|(key, value)| {
                (*key == "general.alignment")
                    .then(|| value.as_u32().map_or(DEFAULT_ALIGNMENT, u64::from))
            })
            .unwrap_or(DEFAULT_ALIGNMENT);
        let mut packed_offset = 0_u64;
        let mut offsets = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            let offset = tensor.offset.unwrap_or(packed_offset);
            offsets.push(offset);
            packed_offset = offset + align_up(tensor.data.len() as u64, alignment).unwrap();
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC);
        bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in metadata {
            push_string(&mut bytes, key);
            bytes.extend_from_slice(&value_tag(value).to_le_bytes());
            push_value(&mut bytes, value);
        }
        for (tensor, offset) in tensors.iter().zip(offsets) {
            push_string(&mut bytes, tensor.name);
            bytes.extend_from_slice(&(tensor.ggml_shape.len() as u32).to_le_bytes());
            for dimension in &tensor.ggml_shape {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
            bytes.extend_from_slice(&tensor.dtype.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        let data_offset = align_up(bytes.len() as u64, alignment).unwrap() as usize;
        bytes.resize(data_offset, 0);
        for tensor in tensors {
            bytes.extend_from_slice(&tensor.data);
            let padded = align_up(bytes.len() as u64, alignment).unwrap() as usize;
            bytes.resize(padded, 0);
        }
        bytes
    }

    fn open_bytes(bytes: &[u8]) -> Result<(tempfile::NamedTempFile, GgufFile)> {
        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(bytes)?;
        file.flush()?;
        let gguf = GgufFile::open(file.path())?;
        Ok((file, gguf))
    }

    fn split_metadata(index: u16, count: u16, tensors: i32) -> Vec<(&'static str, GgufValue)> {
        vec![
            (SPLIT_NO_KEY, GgufValue::U16(index)),
            (SPLIT_COUNT_KEY, GgufValue::U16(count)),
            (SPLIT_TENSORS_COUNT_KEY, GgufValue::I32(tensors)),
        ]
    }

    fn write_split(
        root: &Path,
        prefix: &str,
        index: u16,
        count: u16,
        metadata: &[(&str, GgufValue)],
        tensors: &[TestTensor],
    ) -> Result<PathBuf> {
        let path = root.join(gguf_split_filename(prefix, index, count));
        std::fs::write(&path, make_gguf(metadata, tensors))?;
        Ok(path)
    }

    #[test]
    fn parses_only_canonical_split_filenames() {
        assert_eq!(
            parse_gguf_split_filename("model-q4_k_m-00002-of-00012.gguf"),
            Some(GgufSplitName {
                prefix: "model-q4_k_m".into(),
                index: 1,
                count: 12,
            })
        );
        for invalid in [
            "model-00000-of-00002.gguf",
            "model-00003-of-00002.gguf",
            "model-1-of-2.gguf",
            "model-00001-of-00000.gguf",
            "-00001-of-00002.gguf",
            "model-00001-of-00002.bin",
        ] {
            assert!(
                parse_gguf_split_filename(invalid).is_none(),
                "unexpectedly accepted {invalid}"
            );
        }
    }

    #[test]
    fn aggregates_split_metadata_tensors_offsets_and_mappings() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let mut primary_metadata = split_metadata(0, 3, 2);
        primary_metadata.push(("general.name", GgufValue::String("split fixture".into())));
        let first = write_split(directory.path(), "model", 0, 3, &primary_metadata, &[])?;

        let mut second_metadata = split_metadata(1, 3, 2);
        second_metadata.push(("general.alignment", GgufValue::U32(24)));
        write_split(
            directory.path(),
            "model",
            1,
            3,
            &second_metadata,
            &[TestTensor {
                name: "rope_freqs.weight",
                ggml_shape: vec![3],
                dtype: 0,
                data: f32_bytes(&[1.0, 2.0, 3.0]),
                offset: None,
            }],
        )?;

        let mut third_metadata = split_metadata(2, 3, 2);
        third_metadata.push(("general.alignment", GgufValue::U32(64)));
        write_split(
            directory.path(),
            "model",
            2,
            3,
            &third_metadata,
            &[TestTensor {
                name: "token_embd.weight",
                ggml_shape: vec![2, 1],
                dtype: 0,
                data: f32_bytes(&[7.0, 8.0]),
                offset: None,
            }],
        )?;

        let gguf = GgufFile::open(&first)?;
        assert_eq!(gguf.shard_count(), 3);
        assert_eq!(gguf.tensors().len(), 2);
        assert_eq!(
            gguf.metadata().string("general.name"),
            Some("split fixture")
        );
        assert_eq!(gguf.read_f32_vector("rope_freqs.weight")?, [1.0, 2.0, 3.0]);
        let embedding_location = gguf.tensor_location("token_embd.weight").unwrap();
        assert_eq!(embedding_location.shard_index, 2);
        assert_eq!(embedding_location.alignment, 64);
        assert!(
            embedding_location
                .path
                .ends_with("model-00003-of-00003.gguf")
        );
        assert_eq!(
            embedding_location.absolute_offset()?,
            embedding_location.data_offset
        );

        // SAFETY: every temporary shard remains alive and immutable.
        let backend = unsafe { gguf.backend(GgufBackendConfig::default()) }?;
        let embedding =
            backend.get_unchecked("model.embed_tokens.weight", DType::F32, &Device::Cpu)?;
        assert_eq!(embedding.to_vec2::<f32>()?, vec![vec![7.0, 8.0]]);
        Ok(())
    }

    #[test]
    fn accepts_quantized_tensors_when_only_first_shard_has_model_metadata() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let mut primary_metadata = split_metadata(0, 2, 1);
        primary_metadata.push(("general.quantization_version", GgufValue::U32(2)));
        let first = write_split(directory.path(), "quant", 0, 2, &primary_metadata, &[])?;
        write_split(
            directory.path(),
            "quant",
            1,
            2,
            &split_metadata(1, 2, 1),
            &[TestTensor {
                name: "token_embd.weight",
                ggml_shape: vec![32, 1],
                dtype: 8,
                data: vec![0; 34],
                offset: None,
            }],
        )?;
        let gguf = GgufFile::open(first)?;
        assert_eq!(gguf.shard_count(), 2);
        assert!(
            gguf.tensor("token_embd.weight")
                .unwrap()
                .dtype
                .is_quantized()
        );
        Ok(())
    }

    #[test]
    fn accepts_llama_cpp_merged_split_count_zero_sentinel() -> Result<()> {
        let metadata = split_metadata(0, 0, 1);
        let (_file, gguf) = open_bytes(&make_gguf(
            &metadata,
            &[TestTensor {
                name: "token_embd.weight",
                ggml_shape: vec![1],
                dtype: 0,
                data: f32_bytes(&[3.0]),
                offset: None,
            }],
        ))?;
        assert_eq!(gguf.shard_count(), 1);
        assert_eq!(gguf.tensors().len(), 1);
        Ok(())
    }

    #[test]
    fn rejects_incomplete_inconsistent_or_duplicated_split_sets() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let tensor_a = || TestTensor {
            name: "tensor_a",
            ggml_shape: vec![1],
            dtype: 0,
            data: f32_bytes(&[1.0]),
            offset: None,
        };
        let tensor_b = || TestTensor {
            name: "tensor_b",
            ggml_shape: vec![1],
            dtype: 0,
            data: f32_bytes(&[2.0]),
            offset: None,
        };

        let missing = directory.path().join("missing");
        std::fs::create_dir(&missing)?;
        let first = write_split(
            &missing,
            "model",
            0,
            3,
            &split_metadata(0, 3, 2),
            &[tensor_a()],
        )?;
        write_split(
            &missing,
            "model",
            2,
            3,
            &split_metadata(2, 3, 2),
            &[tensor_b()],
        )?;
        let error = format!("{:#}", GgufFile::open(first).unwrap_err());
        assert!(error.contains("shard 2 of 3"), "{error}");

        let duplicate = directory.path().join("duplicate");
        std::fs::create_dir(&duplicate)?;
        let first = write_split(
            &duplicate,
            "model",
            0,
            2,
            &split_metadata(0, 2, 2),
            &[tensor_a()],
        )?;
        write_split(
            &duplicate,
            "model",
            1,
            2,
            &split_metadata(1, 2, 2),
            &[tensor_a()],
        )?;
        let error = GgufFile::open(first).unwrap_err().to_string();
        assert!(error.contains("duplicated across split files"), "{error}");

        let wrong_total = directory.path().join("wrong-total");
        std::fs::create_dir(&wrong_total)?;
        let first = write_split(
            &wrong_total,
            "model",
            0,
            2,
            &split_metadata(0, 2, 3),
            &[tensor_a()],
        )?;
        write_split(
            &wrong_total,
            "model",
            1,
            2,
            &split_metadata(1, 2, 3),
            &[tensor_b()],
        )?;
        let error = GgufFile::open(first).unwrap_err().to_string();
        assert!(error.contains("declares 3 tensors"), "{error}");

        let inconsistent = directory.path().join("inconsistent");
        std::fs::create_dir(&inconsistent)?;
        let first = write_split(
            &inconsistent,
            "model",
            0,
            2,
            &split_metadata(0, 2, 2),
            &[tensor_a()],
        )?;
        write_split(
            &inconsistent,
            "model",
            1,
            2,
            &split_metadata(1, 2, 9),
            &[tensor_b()],
        )?;
        let error = GgufFile::open(first).unwrap_err().to_string();
        assert!(error.contains("split.tensors.count=9"), "{error}");
        Ok(())
    }

    #[test]
    fn rejects_bad_split_types_nonfirst_entrypoints_and_changed_shards() -> Result<()> {
        let bad_type = tempfile::tempdir()?;
        let metadata = vec![
            (SPLIT_NO_KEY, GgufValue::U16(0)),
            (SPLIT_COUNT_KEY, GgufValue::U32(2)),
            (SPLIT_TENSORS_COUNT_KEY, GgufValue::I32(0)),
        ];
        let path = bad_type.path().join("model-00001-of-00002.gguf");
        std::fs::write(&path, make_gguf(&metadata, &[]))?;
        let error = GgufFile::open(path).unwrap_err().to_string();
        assert!(
            error.contains("split.count") && error.contains("UINT16"),
            "{error}"
        );

        let nonfirst = tempfile::tempdir()?;
        let second = write_split(
            nonfirst.path(),
            "model",
            1,
            2,
            &split_metadata(1, 2, 0),
            &[],
        )?;
        let error = GgufFile::open(second).unwrap_err().to_string();
        assert!(error.contains("load the first shard"), "{error}");

        let changed = tempfile::tempdir()?;
        let first = write_split(changed.path(), "model", 0, 2, &split_metadata(0, 2, 1), &[])?;
        let second = write_split(
            changed.path(),
            "model",
            1,
            2,
            &split_metadata(1, 2, 1),
            &[TestTensor {
                name: "tensor_a",
                ggml_shape: vec![1],
                dtype: 0,
                data: f32_bytes(&[1.0]),
                offset: None,
            }],
        )?;
        let gguf = GgufFile::open(first)?;
        std::fs::OpenOptions::new()
            .write(true)
            .open(second)?
            .set_len(1)?;
        // SAFETY: no mmap can be created after the retained second-shard handle
        // observes truncation, so this must fail before exposing tensor bytes.
        let error = unsafe { gguf.backend(GgufBackendConfig::default()) }
            .unwrap_err()
            .to_string();
        assert!(error.contains("size changed after parsing"), "{error}");
        Ok(())
    }

    #[test]
    fn rejects_negative_oversized_missing_and_filename_mismatched_split_metadata() -> Result<()> {
        let negative = tempfile::tempdir()?;
        let metadata = split_metadata(0, 1, -1);
        let path = negative.path().join("model.gguf");
        std::fs::write(&path, make_gguf(&metadata, &[]))?;
        let error = GgufFile::open(path).unwrap_err().to_string();
        assert!(error.contains("must be non-negative"), "{error}");

        let bad_merged_index = tempfile::tempdir()?;
        let path = bad_merged_index.path().join("merged.gguf");
        std::fs::write(&path, make_gguf(&split_metadata(1, 0, 0), &[]))?;
        let error = GgufFile::open(path).unwrap_err().to_string();
        assert!(error.contains("must be zero"), "{error}");

        let oversized = tempfile::tempdir()?;
        let path = write_split(
            oversized.path(),
            "model",
            0,
            MAX_GGUF_SHARDS + 1,
            &split_metadata(0, MAX_GGUF_SHARDS + 1, 0),
            &[],
        )?;
        let error = GgufFile::open(path).unwrap_err().to_string();
        assert!(error.contains("safety limit"), "{error}");

        let mismatch = tempfile::tempdir()?;
        let path = mismatch.path().join("model-00001-of-00003.gguf");
        std::fs::write(&path, make_gguf(&split_metadata(0, 2, 0), &[]))?;
        let error = GgufFile::open(path).unwrap_err().to_string();
        assert!(error.contains("does not match split.no=0"), "{error}");

        let missing_key = tempfile::tempdir()?;
        let first = write_split(
            missing_key.path(),
            "model",
            0,
            2,
            &split_metadata(0, 2, 0),
            &[],
        )?;
        let subsidiary_metadata = vec![
            (SPLIT_NO_KEY, GgufValue::U16(1)),
            (SPLIT_COUNT_KEY, GgufValue::U16(2)),
        ];
        write_split(missing_key.path(), "model", 1, 2, &subsidiary_metadata, &[])?;
        let error = format!("{:#}", GgufFile::open(first).unwrap_err());
        assert!(error.contains(SPLIT_TENSORS_COUNT_KEY), "{error}");

        let false_family = tempfile::tempdir()?;
        let first = false_family.path().join("model-00001-of-00002.gguf");
        std::fs::write(&first, make_gguf(&split_metadata(0, 0, 0), &[]))?;
        std::fs::write(
            false_family.path().join("model-00002-of-00002.gguf"),
            make_gguf(&split_metadata(1, 2, 0), &[]),
        )?;
        let error = GgufFile::open(first).unwrap_err().to_string();
        assert!(
            error.contains("filename") && error.contains("metadata"),
            "{error}"
        );

        let fail_fast = tempfile::tempdir()?;
        let first = write_split(
            fail_fast.path(),
            "model",
            0,
            3,
            &split_metadata(0, 3, 1),
            &[TestTensor {
                name: "tensor_a",
                ggml_shape: vec![1],
                dtype: 0,
                data: f32_bytes(&[1.0]),
                offset: None,
            }],
        )?;
        write_split(
            fail_fast.path(),
            "model",
            1,
            3,
            &split_metadata(1, 3, 1),
            &[TestTensor {
                name: "tensor_b",
                ggml_shape: vec![1],
                dtype: 0,
                data: f32_bytes(&[2.0]),
                offset: None,
            }],
        )?;
        // Shard 3 is intentionally absent: exceeding the declared global
        // tensor count must fail before the loader attempts to open it.
        let error = GgufFile::open(first).unwrap_err().to_string();
        assert!(error.contains("exceeding declared"), "{error}");
        Ok(())
    }

    #[test]
    fn rejects_inconsistent_subsidiary_split_and_model_metadata() -> Result<()> {
        let wrong_index = tempfile::tempdir()?;
        let first = write_split(
            wrong_index.path(),
            "model",
            0,
            2,
            &split_metadata(0, 2, 0),
            &[],
        )?;
        write_split(
            wrong_index.path(),
            "model",
            1,
            2,
            &split_metadata(0, 2, 0),
            &[],
        )?;
        let error = GgufFile::open(first).unwrap_err().to_string();
        assert!(
            error.contains("split.no=0") && error.contains("expected 1"),
            "{error}"
        );

        let wrong_count = tempfile::tempdir()?;
        let first = write_split(
            wrong_count.path(),
            "model",
            0,
            2,
            &split_metadata(0, 2, 0),
            &[],
        )?;
        write_split(
            wrong_count.path(),
            "model",
            1,
            2,
            &split_metadata(1, 3, 0),
            &[],
        )?;
        let error = GgufFile::open(first).unwrap_err().to_string();
        assert!(
            error.contains("split.count=3") && error.contains("expected 2"),
            "{error}"
        );

        let absent_primary_key = tempfile::tempdir()?;
        let first = write_split(
            absent_primary_key.path(),
            "model",
            0,
            2,
            &split_metadata(0, 2, 0),
            &[],
        )?;
        let mut secondary = split_metadata(1, 2, 0);
        secondary.push(("general.architecture", GgufValue::String("llama".into())));
        write_split(absent_primary_key.path(), "model", 1, 2, &secondary, &[])?;
        let error = GgufFile::open(first).unwrap_err().to_string();
        assert!(
            error.contains("general.architecture") && error.contains("absent from the first"),
            "{error}"
        );

        let conflicting_key = tempfile::tempdir()?;
        let mut primary = split_metadata(0, 2, 0);
        primary.push(("tokenizer.ggml.model", GgufValue::String("gpt2".into())));
        let first = write_split(conflicting_key.path(), "model", 0, 2, &primary, &[])?;
        let mut secondary = split_metadata(1, 2, 0);
        secondary.push(("tokenizer.ggml.model", GgufValue::String("llama".into())));
        write_split(conflicting_key.path(), "model", 1, 2, &secondary, &[])?;
        let error = GgufFile::open(first).unwrap_err().to_string();
        assert!(
            error.contains("tokenizer.ggml.model") && error.contains("conflicting"),
            "{error}"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn split_backend_retains_each_parsed_file_handle_after_path_replacement() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let first = write_split(
            directory.path(),
            "model",
            0,
            2,
            &split_metadata(0, 2, 1),
            &[],
        )?;
        let tensor = |values: &[f32]| TestTensor {
            name: "token_embd.weight",
            ggml_shape: vec![2, 1],
            dtype: 0,
            data: f32_bytes(values),
            offset: None,
        };
        let second = write_split(
            directory.path(),
            "model",
            1,
            2,
            &split_metadata(1, 2, 1),
            &[tensor(&[1.0, 2.0])],
        )?;
        let gguf = GgufFile::open(first)?;
        let replacement = directory.path().join("replacement.gguf");
        std::fs::write(
            &replacement,
            make_gguf(&split_metadata(1, 2, 1), &[tensor(&[9.0, 8.0])]),
        )?;
        assert_eq!(
            std::fs::metadata(&replacement)?.len(),
            std::fs::metadata(&second)?.len()
        );
        std::fs::rename(replacement, second)?;

        // SAFETY: replacement changes the directory entry, not the immutable
        // retained file handle parsed for shard 1. Backend construction must
        // clone that handle instead of reopening the replacement path.
        let backend = unsafe { gguf.backend(GgufBackendConfig::default()) }?;
        drop(gguf);
        let loaded =
            backend.get_unchecked("model.embed_tokens.weight", DType::F32, &Device::Cpu)?;
        assert_eq!(loaded.to_vec2::<f32>()?, vec![vec![1.0, 2.0]]);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn rejects_fifo_shards_without_blocking() -> Result<()> {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};

        let directory = tempfile::tempdir()?;
        let fifo = directory.path().join("model.gguf");
        let fifo_path = CString::new(fifo.as_os_str().as_bytes())?;
        // SAFETY: the C string is NUL-terminated and names a fresh temporary path.
        let status = unsafe { libc::mkfifo(fifo_path.as_ptr(), 0o600) };
        ensure!(
            status == 0,
            "failed to create FIFO fixture: {}",
            std::io::Error::last_os_error()
        );
        let error = open_regular_gguf_file(&fifo).unwrap_err().to_string();
        assert!(error.contains("regular file"), "{error}");
        Ok(())
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn reads_small_f32_architecture_vectors_without_mapping_model_weights() -> Result<()> {
        let values = [1.0_f32, 1.651_329, 32.0];
        let bytes = make_gguf(
            &[],
            &[TestTensor {
                name: "rope_freqs.weight",
                ggml_shape: vec![values.len() as u64],
                dtype: 0,
                data: f32_bytes(&values),
                offset: None,
            }],
        );
        let (_file, gguf) = open_bytes(&bytes)?;
        assert_eq!(gguf.read_f32_vector("rope_freqs.weight")?, values);
        Ok(())
    }

    #[test]
    fn rejects_non_f32_or_non_vector_architecture_tensors() -> Result<()> {
        let wrong_dtype = make_gguf(
            &[],
            &[TestTensor {
                name: "rope_freqs.weight",
                ggml_shape: vec![2],
                dtype: 1,
                data: vec![0; 4],
                offset: None,
            }],
        );
        let (_file, gguf) = open_bytes(&wrong_dtype)?;
        assert!(
            gguf.read_f32_vector("rope_freqs.weight")
                .unwrap_err()
                .to_string()
                .contains("must be F32")
        );

        let wrong_rank = make_gguf(
            &[],
            &[TestTensor {
                name: "rope_freqs.weight",
                ggml_shape: vec![1, 2],
                dtype: 0,
                data: f32_bytes(&[1.0, 2.0]),
                offset: None,
            }],
        );
        let (_file, gguf) = open_bytes(&wrong_rank)?;
        assert!(
            gguf.read_f32_vector("rope_freqs.weight")
                .unwrap_err()
                .to_string()
                .contains("must be rank one")
        );
        Ok(())
    }

    fn metadata_value_offset(key: &str) -> usize {
        // Fixed v3 header, key length, key bytes, then the u32 value tag.
        24 + 8 + key.len() + 4
    }

    fn error_text(bytes: &[u8]) -> String {
        format!("{:#}", open_bytes(bytes).unwrap_err())
    }

    #[test]
    fn decoded_allocation_budget_is_cumulative_for_strings_and_arrays() -> Result<()> {
        let mut encoded_string = Vec::new();
        push_string(&mut encoded_string, "hello");
        let mut string_reader = BoundedReader::with_allocation_limit(
            encoded_string.as_slice(),
            u64::try_from(encoded_string.len())?,
            4,
        );
        assert!(
            format!("{:#}", string_reader.string().unwrap_err())
                .contains("decoded allocation budget")
        );

        let mut encoded_arrays = Vec::new();
        for values in [[1_u8, 2], [3, 4]] {
            encoded_arrays.extend_from_slice(&0_u32.to_le_bytes());
            encoded_arrays.extend_from_slice(&2_u64.to_le_bytes());
            encoded_arrays.extend_from_slice(&values);
        }
        let element_bytes = u64::try_from(std::mem::size_of::<GgufValue>())?;
        let mut array_reader = BoundedReader::with_allocation_limit(
            encoded_arrays.as_slice(),
            u64::try_from(encoded_arrays.len())?,
            element_bytes * 3,
        );
        let first = read_value(&mut array_reader, GgufValueType::Array, false)?;
        assert_eq!(first.as_array().map(<[GgufValue]>::len), Some(2));
        assert!(
            format!(
                "{:#}",
                read_value(&mut array_reader, GgufValueType::Array, false).unwrap_err()
            )
            .contains("decoded allocation budget")
        );
        Ok(())
    }

    #[test]
    fn reads_every_metadata_scalar_type_and_typed_arrays() -> Result<()> {
        let metadata = vec![
            ("u8", GgufValue::U8(8)),
            ("i8", GgufValue::I8(-8)),
            ("u16", GgufValue::U16(16)),
            ("i16", GgufValue::I16(-16)),
            ("u32", GgufValue::U32(32)),
            ("i32", GgufValue::I32(-32)),
            ("f32", GgufValue::F32(3.25)),
            ("bool", GgufValue::Bool(true)),
            ("string", GgufValue::String("token".into())),
            (
                "array",
                GgufValue::Array {
                    element_type: GgufValueType::I32,
                    values: vec![GgufValue::I32(1), GgufValue::I32(-2)],
                },
            ),
            ("u64", GgufValue::U64(64)),
            ("i64", GgufValue::I64(-64)),
            ("f64", GgufValue::F64(6.5)),
        ];
        let (_file, gguf) = open_bytes(&make_gguf(&metadata, &[]))?;
        assert_eq!(gguf.metadata().len(), 13);
        assert_eq!(gguf.metadata().string("string"), Some("token"));
        assert_eq!(gguf.metadata().u64("u8"), Some(8));
        assert_eq!(gguf.metadata().f64("f64"), Some(6.5));
        assert_eq!(
            gguf.metadata().array("array"),
            Some(&[GgufValue::I32(1), GgufValue::I32(-2)][..])
        );
        assert_eq!(
            gguf.metadata()
                .get("array")
                .and_then(GgufValue::array_element_type),
            Some(GgufValueType::I32)
        );
        Ok(())
    }

    #[test]
    fn accepts_spec_compliant_non_power_of_two_alignment() -> Result<()> {
        let tensor = TestTensor {
            name: "token_embd.weight",
            ggml_shape: vec![1],
            dtype: 0,
            data: f32_bytes(&[7.]),
            offset: None,
        };
        let bytes = make_gguf(&[("general.alignment", GgufValue::U32(24))], &[tensor]);
        let (_file, gguf) = open_bytes(&bytes)?;
        assert_eq!(gguf.alignment(), 24);
        // SAFETY: the NamedTempFile remains alive and immutable.
        let backend = unsafe { gguf.backend(GgufBackendConfig::default()) }?;
        let loaded =
            backend.get_unchecked("model.embed_tokens.weight", DType::F32, &Device::Cpu)?;
        assert_eq!(loaded.to_vec1::<f32>()?, vec![7.]);
        Ok(())
    }

    #[test]
    fn reverses_dimensions_and_loads_f32_f16_bf16_through_mmap() -> Result<()> {
        let tensors = vec![
            TestTensor {
                name: "token_embd.weight",
                ggml_shape: vec![2, 3],
                dtype: 0,
                data: f32_bytes(&[1., 2., 3., 4., 5., 6.]),
                offset: None,
            },
            TestTensor {
                name: "blk.0.attn_q.weight",
                ggml_shape: vec![2, 2],
                dtype: 1,
                data: [0x3c00_u16, 0x4000, 0x4200, 0x4400]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
                offset: None,
            },
            TestTensor {
                name: "output_norm.weight",
                ggml_shape: vec![2],
                dtype: 30,
                data: [0x3f80_u16, 0x4000]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
                offset: None,
            },
        ];
        let (_file, gguf) = open_bytes(&make_gguf(&[], &tensors))?;
        assert_eq!(gguf.tensor("token_embd.weight").unwrap().shape, [3, 2]);
        assert_eq!(gguf.tensor("blk.0.attn_q.weight").unwrap().offset, 32);

        // SAFETY: the NamedTempFile is kept alive and immutable for the backend lifetime.
        let backend = unsafe { gguf.backend(GgufBackendConfig::default()) }?;
        let embedding =
            backend.get_unchecked("model.embed_tokens.weight", DType::F32, &Device::Cpu)?;
        assert_eq!(embedding.dims(), &[3, 2]);
        assert_eq!(
            embedding.to_vec2::<f32>()?,
            vec![vec![1., 2.], vec![3., 4.], vec![5., 6.]]
        );
        let query = backend.get_unchecked(
            "model.layers.0.self_attn.q_proj.weight",
            DType::F32,
            &Device::Cpu,
        )?;
        assert_eq!(query.to_vec2::<f32>()?, vec![vec![1., 2.], vec![3., 4.]]);
        let norm = backend.get_unchecked("model.norm.weight", DType::F32, &Device::Cpu)?;
        assert_eq!(norm.to_vec1::<f32>()?, vec![1., 2.]);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn temporary_tensor_mappings_support_adjacent_and_concurrent_reads() -> Result<()> {
        let tensors = [
            TestTensor {
                name: "first",
                ggml_shape: vec![4],
                dtype: 0,
                data: f32_bytes(&[1., 2., 3., 4.]),
                offset: None,
            },
            TestTensor {
                name: "second",
                ggml_shape: vec![4],
                dtype: 0,
                data: f32_bytes(&[5., 6., 7., 8.]),
                offset: None,
            },
        ];
        let (_file, gguf) = open_bytes(&make_gguf(&[], &tensors))?;
        let config = GgufBackendConfig {
            map_hf_names: false,
            ..GgufBackendConfig::default()
        };
        // SAFETY: the NamedTempFile remains alive and immutable for all
        // temporary backend mappings in this test.
        let backend = Arc::new(unsafe { gguf.backend(config) }?);

        // Both tiny tensors occupy the same operating-system page. Each load
        // maps only its logical byte range and drops that temporary mapping
        // after copying into owned tensor storage.
        assert_eq!(
            backend
                .load_tensor("second", DType::F32, &Device::Cpu)?
                .to_vec1::<f32>()?,
            vec![5., 6., 7., 8.]
        );
        assert_eq!(
            backend
                .load_tensor("first", DType::F32, &Device::Cpu)?
                .to_vec1::<f32>()?,
            vec![1., 2., 3., 4.]
        );

        let first_payload = backend.raw_tensor("first")?;
        let second_payload = backend.raw_tensor("second")?;
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let (first_bytes, second_bytes) = std::thread::scope(|scope| {
            let first_barrier = Arc::clone(&barrier);
            let first = scope.spawn(move || {
                first_barrier.wait();
                first_payload.bytes().to_vec()
            });
            let second_barrier = Arc::clone(&barrier);
            let second = scope.spawn(move || {
                second_barrier.wait();
                second_payload.bytes().to_vec()
            });
            (
                first.join().expect("first mapping reader panicked"),
                second.join().expect("second mapping reader panicked"),
            )
        });
        assert_eq!(first_bytes, f32_bytes(&[1., 2., 3., 4.]));
        assert_eq!(second_bytes, f32_bytes(&[5., 6., 7., 8.]));

        std::thread::scope(|scope| -> Result<()> {
            let first_backend = Arc::clone(&backend);
            let first_reader = scope.spawn(move || -> Result<()> {
                for _ in 0..64 {
                    ensure!(
                        first_backend
                            .load_tensor("first", DType::F32, &Device::Cpu)?
                            .to_vec1::<f32>()?
                            == [1., 2., 3., 4.]
                    );
                }
                Ok(())
            });
            let second_backend = Arc::clone(&backend);
            let second_reader = scope.spawn(move || -> Result<()> {
                for _ in 0..64 {
                    ensure!(
                        second_backend
                            .load_tensor("second", DType::F32, &Device::Cpu)?
                            .to_vec1::<f32>()?
                            == [5., 6., 7., 8.]
                    );
                }
                Ok(())
            });
            first_reader.join().expect("first reader panicked")?;
            second_reader.join().expect("second reader panicked")?;
            Ok(())
        })?;
        Ok(())
    }

    fn assert_loaded_storage_outlives_backend_and_source(device: &Device) -> Result<()> {
        let mut quantized = vec![0x00, 0x3c]; // F16 scale 1.0.
        quantized.extend(std::iter::repeat_n(2_u8, 32));
        let tensors = [
            TestTensor {
                name: "dense",
                ggml_shape: vec![2],
                dtype: 0,
                data: f32_bytes(&[3., 4.]),
                offset: None,
            },
            TestTensor {
                name: "quantized",
                ggml_shape: vec![32, 1],
                dtype: 8,
                data: quantized,
                offset: None,
            },
        ];
        let metadata = [("general.quantization_version", GgufValue::U32(2))];
        let (file, gguf) = open_bytes(&make_gguf(&metadata, &tensors))?;
        let source_path = file.path().to_path_buf();
        let config = GgufBackendConfig {
            map_hf_names: false,
            ..GgufBackendConfig::default()
        };
        // SAFETY: the temporary file remains immutable until all loads finish.
        let backend = unsafe { gguf.backend(config) }?;
        let dense = backend.load_tensor("dense", DType::F32, device)?;
        let GgufMatrix::Quantized(quantized) =
            backend.load_matrix("quantized", DType::F32, device)?
        else {
            panic!("Q8_0 matrix was unexpectedly expanded")
        };

        drop(backend);
        drop(gguf);
        drop(file);
        assert!(!source_path.exists());
        assert_eq!(dense.to_vec1::<f32>()?, vec![3., 4.]);
        assert!(
            quantized
                .dequantize(device)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .all(|value| (value - 2.0).abs() <= f32::EPSILON)
        );
        Ok(())
    }

    #[test]
    fn cpu_loaded_storage_outlives_backend_and_source_file() -> Result<()> {
        assert_loaded_storage_outlives_backend_and_source(&Device::Cpu)
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an accessible Apple Metal device"]
    fn metal_loaded_storage_outlives_backend_and_source_file() -> Result<()> {
        assert_loaded_storage_outlives_backend_and_source(&Device::new_metal(0)?)
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an accessible Apple Metal device"]
    fn file_backed_metal_weight_uses_nonzero_offset_for_mv_mm_embedding_and_outlives_file()
    -> Result<()> {
        let mut data = Vec::with_capacity(68);
        for quantized_value in [1_u8, 2_u8] {
            data.extend_from_slice(&[0x00, 0x3c]); // F16 scale 1.0.
            data.extend(std::iter::repeat_n(quantized_value, 32));
        }
        let encoded = make_gguf(
            &[("general.quantization_version", GgufValue::U32(2))],
            &[TestTensor {
                name: "weight",
                ggml_shape: vec![32, 2],
                dtype: 8,
                data,
                offset: None,
            }],
        );
        let (file, gguf) = open_bytes(&encoded)?;
        let source_path = file.path().to_path_buf();
        let config = GgufBackendConfig {
            map_hf_names: false,
            ..GgufBackendConfig::default()
        };
        // SAFETY: the source inode remains immutable; the mapped Metal buffer
        // retains its private mapping after these parser objects are dropped.
        let backend = unsafe { gguf.backend(config) }?;
        let expected_offset = backend.locate_tensor("weight")?.file_offset as usize;
        let metal = Device::new_metal(0)?;
        let GgufMatrix::FileBackedQuantized(weight) =
            backend.load_matrix_preserving("weight", DType::F32, &metal)?
        else {
            panic!("eligible Q8_0 Metal matrix did not select file-backed storage")
        };
        assert_eq!(weight.dtype(), GgmlDType::Q8_0);
        assert_eq!(weight.shape().dims(), &[2, 32]);
        assert_eq!(weight.byte_range(), expected_offset..expected_offset + 68);

        drop(backend);
        drop(gguf);
        drop(file);
        assert!(!source_path.exists());

        let vector_input = Tensor::ones((1, 1, 32), DType::F32, &metal)?;
        let mv = weight
            .matmul(&vector_input)?
            .to_device(&Device::Cpu)?
            .to_vec3::<f32>()?;
        assert_eq!(mv, vec![vec![vec![32.0, 64.0]]]);

        let matrix_input = Tensor::from_vec(
            [vec![1.0_f32; 32], vec![0.5_f32; 32]].concat(),
            (1, 2, 32),
            &metal,
        )?;
        let mm = weight
            .matmul(&matrix_input)?
            .to_device(&Device::Cpu)?
            .to_vec3::<f32>()?;
        assert_eq!(mm, vec![vec![vec![32.0, 64.0], vec![16.0, 32.0]]]);

        let embedded = weight.embedding(&Tensor::new(&[[1_u32, 0]], &metal)?)?;
        // Drop the final Rust owner immediately after encoding. The command
        // buffer must retain the external Metal buffer, whose deallocator in
        // turn owns the mmap until GPU completion.
        drop(weight);
        let embedded = embedded.to_device(&Device::Cpu)?.to_vec3::<f32>()?;
        assert_eq!(embedded[0][0], vec![2.0; 32]);
        assert_eq!(embedded[0][1], vec![1.0; 32]);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn backend_uses_the_parsed_file_handle_after_path_replacement() -> Result<()> {
        let tensor = |values: &[f32]| TestTensor {
            name: "token_embd.weight",
            ggml_shape: vec![2, 1],
            dtype: 0,
            data: f32_bytes(values),
            offset: None,
        };
        let original = make_gguf(&[], &[tensor(&[1., 2.])]);
        let replacement = make_gguf(&[], &[tensor(&[9., 8.])]);
        assert_eq!(original.len(), replacement.len());

        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model.gguf");
        std::fs::write(&path, original)?;
        let gguf = GgufFile::open(&path)?;
        let replacement_path = directory.path().join("replacement.gguf");
        std::fs::write(&replacement_path, replacement)?;
        std::fs::rename(replacement_path, &path)?;

        // SAFETY: the parsed file handle itself remains immutable; only its old
        // directory entry was atomically replaced. Backend construction must
        // clone that handle instead of reopening the replacement path.
        let backend = unsafe { gguf.backend(GgufBackendConfig::default()) }?;
        drop(gguf);
        let loaded =
            backend.get_unchecked("model.embed_tokens.weight", DType::F32, &Device::Cpu)?;
        assert_eq!(loaded.to_vec2::<f32>()?, vec![vec![1., 2.]]);
        Ok(())
    }

    #[test]
    fn maps_all_shared_decoder_paths() {
        assert_eq!(
            hf_to_gguf_name("model.layers.12.self_attn.o_proj.weight").as_deref(),
            Some("blk.12.attn_output.weight")
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.3.mlp.down_proj.bias").as_deref(),
            Some("blk.3.ffn_down.bias")
        );
        assert_eq!(
            hf_to_gguf_name("lm_head.weight").as_deref(),
            Some("output.weight")
        );
        for (runtime, gguf) in [
            (
                "model.layers.2.self_attn.q_norm.weight",
                "blk.2.attn_q_norm.weight",
            ),
            (
                "model.layers.2.self_attn.k_norm.weight",
                "blk.2.attn_k_norm.weight",
            ),
            (
                "model.layers.2.pre_feedforward_layernorm.weight",
                "blk.2.ffn_norm.weight",
            ),
            (
                "model.layers.2.post_attention_norm.weight",
                "blk.2.post_attention_norm.weight",
            ),
            (
                "model.layers.2.post_feedforward_layernorm.weight",
                "blk.2.post_ffw_norm.weight",
            ),
            (
                "model.layers.2.attn_post_norm.weight",
                "blk.2.post_attention_norm.weight",
            ),
            (
                "model.layers.2.linear_attn.in_proj_qkv.weight",
                "blk.2.attn_qkv.weight",
            ),
            (
                "model.layers.2.linear_attn.in_proj_z.weight",
                "blk.2.attn_gate.weight",
            ),
            (
                "model.layers.2.linear_attn.conv1d.weight",
                "blk.2.ssm_conv1d.weight",
            ),
            ("model.layers.2.linear_attn.dt_bias", "blk.2.ssm_dt.bias"),
            ("model.layers.2.linear_attn.A_log", "blk.2.ssm_a"),
            (
                "model.layers.2.linear_attn.in_proj_b.weight",
                "blk.2.ssm_beta.weight",
            ),
            (
                "model.layers.2.linear_attn.in_proj_a.weight",
                "blk.2.ssm_alpha.weight",
            ),
            (
                "model.layers.2.linear_attn.norm.weight",
                "blk.2.ssm_norm.weight",
            ),
            (
                "model.layers.2.linear_attn.out_proj.weight",
                "blk.2.ssm_out.weight",
            ),
        ] {
            assert_eq!(hf_to_gguf_name(runtime).as_deref(), Some(gguf));
        }
        assert!(hf_to_gguf_name("model.layers.x.mlp.up_proj.weight").is_none());
    }

    #[test]
    fn inverse_permutates_llama_query_weight_and_backend_applies_it() -> Result<()> {
        let tensor = Tensor::from_vec(
            (0..8).map(|value| value as f32).collect(),
            (8, 1),
            &Device::Cpu,
        )?;
        let permuted = inverse_permute_llama_qk(tensor, 2)?;
        assert_eq!(
            permuted.to_vec2::<f32>()?,
            vec![
                vec![0.],
                vec![2.],
                vec![1.],
                vec![3.],
                vec![4.],
                vec![6.],
                vec![5.],
                vec![7.]
            ]
        );

        let tensors = [
            TestTensor {
                name: "blk.0.attn_q.weight",
                ggml_shape: vec![1, 8],
                dtype: 0,
                data: f32_bytes(&(0..8).map(|value| value as f32).collect::<Vec<_>>()),
                offset: None,
            },
            TestTensor {
                name: "blk.0.attn_k.bias",
                ggml_shape: vec![4],
                dtype: 0,
                data: f32_bytes(&[10., 11., 12., 13.]),
                offset: None,
            },
        ];
        let (_file, gguf) = open_bytes(&make_gguf(&[], &tensors))?;
        let config = GgufBackendConfig {
            llama_qk: Some(LlamaQkConfig {
                query_heads: 2,
                kv_heads: 1,
            }),
            ..GgufBackendConfig::default()
        };
        // SAFETY: the NamedTempFile is kept alive and immutable for the backend lifetime.
        let backend = unsafe { gguf.backend(config) }?;
        let loaded = backend.get_unchecked(
            "model.layers.0.self_attn.q_proj.weight",
            DType::F32,
            &Device::Cpu,
        )?;
        assert_eq!(loaded.to_vec2::<f32>()?, permuted.to_vec2::<f32>()?);
        let key_bias = backend.get_unchecked(
            "model.layers.0.self_attn.k_proj.bias",
            DType::F32,
            &Device::Cpu,
        )?;
        assert_eq!(key_bias.to_vec1::<f32>()?, vec![10., 12., 11., 13.]);
        Ok(())
    }

    #[test]
    fn tied_output_can_fall_back_to_embedding() -> Result<()> {
        let tensors = [TestTensor {
            name: "token_embd.weight",
            ggml_shape: vec![2, 1],
            dtype: 0,
            data: f32_bytes(&[1., 2.]),
            offset: None,
        }];
        let (_file, gguf) = open_bytes(&make_gguf(&[], &tensors))?;
        let config = GgufBackendConfig {
            fallback_output_to_embedding: true,
            ..GgufBackendConfig::default()
        };
        // SAFETY: the NamedTempFile is kept alive and immutable for the backend lifetime.
        let backend = unsafe { gguf.backend(config) }?;
        assert!(backend.contains_tensor("lm_head.weight"));
        let output = backend.get_unchecked("lm_head.weight", DType::F32, &Device::Cpu)?;
        assert_eq!(output.to_vec2::<f32>()?, vec![vec![1., 2.]]);
        Ok(())
    }

    #[test]
    fn rejects_bad_magic_version_and_header_counts() {
        let valid = make_gguf(&[], &[]);

        let mut bad_magic = valid.clone();
        bad_magic[0] = b'X';
        assert!(error_text(&bad_magic).contains("bad GGUF magic"));

        let mut bad_version = valid.clone();
        bad_version[4..8].copy_from_slice(&2_u32.to_le_bytes());
        assert!(error_text(&bad_version).contains("version 2"));

        let mut big_endian = valid.clone();
        big_endian[4..8].copy_from_slice(&GGUF_VERSION.to_be_bytes());
        assert!(error_text(&big_endian).contains("big-endian GGUF v3"));

        let mut huge_count = valid;
        huge_count[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(error_text(&huge_count).contains("tensor count"));
    }

    #[test]
    fn rejects_duplicate_metadata_and_tensor_names() {
        let duplicate_metadata = make_gguf(
            &[("same", GgufValue::U32(1)), ("same", GgufValue::U32(2))],
            &[],
        );
        assert!(error_text(&duplicate_metadata).contains("duplicate GGUF metadata"));

        let tensor = TestTensor {
            name: "same",
            ggml_shape: vec![1],
            dtype: 0,
            data: f32_bytes(&[1.]),
            offset: None,
        };
        let duplicate_tensors = make_gguf(&[], &[tensor.clone(), tensor]);
        assert!(error_text(&duplicate_tensors).contains("duplicate GGUF tensor"));
    }

    #[test]
    fn rejects_invalid_bool_unknown_type_and_nested_array() {
        let mut invalid_bool = make_gguf(&[("bool", GgufValue::Bool(true))], &[]);
        invalid_bool[metadata_value_offset("bool")] = 2;
        assert!(error_text(&invalid_bool).contains("boolean byte 2"));

        let mut unknown_type = make_gguf(&[("value", GgufValue::U32(1))], &[]);
        let tag_offset = 24 + 8 + "value".len();
        unknown_type[tag_offset..tag_offset + 4].copy_from_slice(&99_u32.to_le_bytes());
        assert!(error_text(&unknown_type).contains("value type 99"));

        let nested = GgufValue::Array {
            element_type: GgufValueType::Array,
            values: vec![GgufValue::Array {
                element_type: GgufValueType::U8,
                values: vec![],
            }],
        };
        assert!(error_text(&make_gguf(&[("nested", nested)], &[])).contains("nested GGUF"));
    }

    #[test]
    fn rejects_invalid_alignment_noncanonical_offsets_and_truncation() {
        let mut invalid_alignment = make_gguf(&[("general.alignment", GgufValue::U32(4))], &[]);
        let offset = metadata_value_offset("general.alignment");
        invalid_alignment[offset..offset + 4].copy_from_slice(&3_u32.to_le_bytes());
        assert!(error_text(&invalid_alignment).contains("multiple of 8"));

        let tensors = [
            TestTensor {
                name: "a",
                ggml_shape: vec![1],
                dtype: 0,
                data: f32_bytes(&[1.]),
                offset: Some(0),
            },
            TestTensor {
                name: "b",
                ggml_shape: vec![1],
                dtype: 0,
                data: f32_bytes(&[2.]),
                offset: Some(0),
            },
        ];
        assert!(error_text(&make_gguf(&[], &tensors)).contains("non-canonical offset"));

        let mut truncated = make_gguf(&[], &tensors[..1]);
        truncated.truncate(truncated.len() - 32);
        assert!(open_bytes(&truncated).is_err());
    }

    #[test]
    fn rejects_unsupported_tensor_types_and_shape_overflow() {
        let unsupported = [TestTensor {
            name: "q4_1",
            ggml_shape: vec![32],
            dtype: 3,
            data: vec![0; 20],
            offset: None,
        }];
        let error = error_text(&make_gguf(&[], &unsupported));
        assert!(error.contains("unsupported GGML tensor type 3"));
        assert!(error.contains("Q2_K (10)"));
        assert!(error.contains("Q3_K (11)"));

        let overflow = [TestTensor {
            name: "huge",
            ggml_shape: vec![u64::MAX, 2],
            dtype: 0,
            data: vec![],
            offset: None,
        }];
        assert!(open_bytes(&make_gguf(&[], &overflow)).is_err());
    }

    #[test]
    fn q2k_and_q3k_type_contract_matches_ggml() -> Result<()> {
        assert_eq!(GgufTensorType::ALL.len(), 12);
        for (tensor_type, tag, label, type_size, ggml_dtype) in [
            (GgufTensorType::Q2K, 10, "Q2_K", 84, GgmlDType::Q2K),
            (GgufTensorType::Q3K, 11, "Q3_K", 110, GgmlDType::Q3K),
        ] {
            assert!(GgufTensorType::ALL.contains(&tensor_type));
            assert_eq!(GgufTensorType::from_tag(tag)?, tensor_type);
            assert_eq!(tensor_type.label(), label);
            assert!(tensor_type.is_quantized());
            assert_eq!(tensor_type.candle_dtype(), None);
            assert_eq!(tensor_type.block_size(), 256);
            assert_eq!(tensor_type.type_size(), type_size);
            assert_eq!(tensor_type.ggml_dtype(), ggml_dtype);
        }
        Ok(())
    }

    #[test]
    fn parses_and_dequantizes_q4_0_and_q8_0_golden_blocks() -> Result<()> {
        let q4 = vec![
            0x00, 0x30, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b,
            0x3c, 0x2d, 0x1e, 0x0f,
        ];
        let q8 = vec![
            0x08, 0x20, 0x81, 0xc0, 0xe0, 0xf0, 0xf3, 0xff, 0x00, 0x01, 0x0d, 0x10, 0x20, 0x40,
            0x7f, 0x5f, 0xa1, 0x2a, 0xd6, 0x19, 0xe7, 0x03, 0xfd, 0x72, 0x8e, 0x4c, 0xb4, 0x33,
            0xcd, 0x06, 0xfa, 0x01, 0xff, 0x00,
        ];
        let tensors = [
            TestTensor {
                name: "q4",
                ggml_shape: vec![32],
                dtype: 2,
                data: q4,
                offset: None,
            },
            TestTensor {
                name: "q8",
                ggml_shape: vec![32],
                dtype: 8,
                data: q8,
                offset: None,
            },
        ];
        let metadata = [("general.quantization_version", GgufValue::U32(2))];
        let (_file, gguf) = open_bytes(&make_gguf(&metadata, &tensors))?;
        assert_eq!(gguf.tensor("q4").unwrap().byte_len, 18);
        assert_eq!(gguf.tensor("q8").unwrap().byte_len, 34);
        // The general matrix loader intentionally requires rank two; construct
        // the same blocks as one-row matrices for the QTensor golden below.
        let matrix_tensors = [
            TestTensor {
                name: "q4",
                ggml_shape: vec![32, 1],
                dtype: 2,
                data: tensors[0].data.clone(),
                offset: None,
            },
            TestTensor {
                name: "q8",
                ggml_shape: vec![32, 1],
                dtype: 8,
                data: tensors[1].data.clone(),
                offset: None,
            },
        ];
        let (_file, gguf) = open_bytes(&make_gguf(&metadata, &matrix_tensors))?;
        // SAFETY: the NamedTempFile remains alive and immutable.
        let backend = unsafe { gguf.backend(GgufBackendConfig::default()) }?;
        let GgufMatrix::Quantized(q4) = backend.load_matrix("q4", DType::F32, &Device::Cpu)? else {
            panic!("Q4_0 was unexpectedly expanded")
        };
        assert_eq!(q4.dtype(), GgmlDType::Q4_0);
        assert_eq!(q4.storage_size_in_bytes(), 18);
        let expected_q4 = (0..16)
            .map(|index| -1.0 + index as f32 * 0.125)
            .chain((0..16).map(|index| 0.875 - index as f32 * 0.125))
            .collect::<Vec<_>>();
        assert_eq!(
            q4.dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            expected_q4
        );

        let GgufMatrix::Quantized(q8) = backend.load_matrix("q8", DType::F32, &Device::Cpu)? else {
            panic!("Q8_0 was unexpectedly expanded")
        };
        assert_eq!(q8.dtype(), GgmlDType::Q8_0);
        assert_eq!(q8.storage_size_in_bytes(), 34);
        let signed = [
            -127_i8, -64, -32, -16, -13, -1, 0, 1, 13, 16, 32, 64, 127, 95, -95, 42, -42, 25, -25,
            3, -3, 114, -114, 76, -76, 51, -51, 6, -6, 1, -1, 0,
        ];
        let expected_q8 = signed
            .into_iter()
            .map(|value| f32::from(value) * 0.007_873_535)
            .collect::<Vec<_>>();
        assert_eq!(
            q8.dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            expected_q8
        );
        Ok(())
    }

    #[test]
    fn loads_dense_expert_bank_from_independent_temporary_mappings() -> Result<()> {
        let tensors = [TestTensor {
            name: "experts",
            ggml_shape: vec![2, 1, 2],
            dtype: 0,
            data: f32_bytes(&[1., 2., 3., 4.]),
            offset: None,
        }];
        let (_file, gguf) = open_bytes(&make_gguf(&[], &tensors))?;
        // SAFETY: the NamedTempFile remains alive and immutable.
        let backend = unsafe { gguf.backend(GgufBackendConfig::default()) }?;
        let bank = backend.load_matrix_bank(
            "experts",
            &Shape::from((2, 1, 2)),
            DType::F32,
            &Device::Cpu,
        )?;
        assert_eq!(bank.len(), 2);
        for (matrix, expected) in bank.into_iter().zip([[1., 2.], [3., 4.]]) {
            let GgufMatrix::Dense(matrix) = matrix else {
                panic!("F32 expert was unexpectedly quantized")
            };
            assert_eq!(matrix.to_vec2::<f32>()?, vec![expected.to_vec()]);
        }
        Ok(())
    }

    #[test]
    fn loads_quantized_rank_three_expert_bank_without_expansion() -> Result<()> {
        let mut data = Vec::with_capacity(68);
        for value in [1_u8, 2_u8] {
            // Q8_0 block: F16 scale followed by 32 signed quantized values.
            data.extend_from_slice(&[0x00, 0x3c]);
            data.extend(std::iter::repeat_n(value, 32));
        }
        let metadata = [("general.quantization_version", GgufValue::U32(2))];
        let tensors = [TestTensor {
            name: "experts",
            ggml_shape: vec![32, 1, 2],
            dtype: 8,
            data,
            offset: None,
        }];
        let (_file, gguf) = open_bytes(&make_gguf(&metadata, &tensors))?;
        assert_eq!(gguf.tensor("experts").unwrap().shape, vec![2, 1, 32]);
        // SAFETY: the NamedTempFile remains alive and immutable.
        let backend = unsafe { gguf.backend(GgufBackendConfig::default()) }?;
        assert!(
            backend
                .load_matrix_bank(
                    "experts",
                    &Shape::from((3, 1, 32)),
                    DType::F32,
                    &Device::Cpu,
                )
                .unwrap_err()
                .to_string()
                .contains("shape mismatch")
        );
        let bank = backend.load_matrix_bank(
            "experts",
            &Shape::from((2, 1, 32)),
            DType::F32,
            &Device::Cpu,
        )?;
        assert_eq!(bank.len(), 2);
        for (expert, matrix) in bank.into_iter().enumerate() {
            let GgufMatrix::Quantized(matrix) = matrix else {
                panic!("expert bank was unexpectedly expanded")
            };
            assert_eq!(matrix.dtype(), GgmlDType::Q8_0);
            assert_eq!(matrix.shape().dims(), &[1, 32]);
            assert_eq!(matrix.storage_size_in_bytes(), 34);
            let values = matrix
                .dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(
                values
                    .into_iter()
                    .all(|value| (value - (expert + 1) as f32).abs() < f32::EPSILON)
            );
        }

        let rank_two = [TestTensor {
            name: "matrix",
            ggml_shape: vec![32, 2],
            dtype: 8,
            data: vec![0; 68],
            offset: None,
        }];
        let (_file, gguf) = open_bytes(&make_gguf(&metadata, &rank_two))?;
        // SAFETY: the NamedTempFile remains alive and immutable.
        let backend = unsafe { gguf.backend(GgufBackendConfig::default()) }?;
        assert!(
            backend
                .load_matrix_bank("matrix", &Shape::from((2, 1, 32)), DType::F32, &Device::Cpu,)
                .unwrap_err()
                .to_string()
                .contains("[experts, rows, columns]")
        );

        let oversized_experts = 1_025_usize;
        let oversized = [TestTensor {
            name: "experts",
            ggml_shape: vec![32, 1, oversized_experts as u64],
            dtype: 8,
            data: vec![0; 34 * oversized_experts],
            offset: None,
        }];
        let (_file, gguf) = open_bytes(&make_gguf(&metadata, &oversized))?;
        // SAFETY: the NamedTempFile remains alive and immutable.
        let backend = unsafe { gguf.backend(GgufBackendConfig::default()) }?;
        assert!(
            backend
                .load_matrix_bank(
                    "experts",
                    &Shape::from((oversized_experts, 1, 32)),
                    DType::F32,
                    &Device::Cpu,
                )
                .unwrap_err()
                .to_string()
                .contains("exceeding the limit")
        );
        Ok(())
    }

    #[test]
    fn loads_q2k_q3k_expert_banks_as_validated_compressed_slices() -> Result<()> {
        let metadata = [("general.quantization_version", GgufValue::U32(2))];
        for (name, tag, dtype, type_size) in [
            ("q2k_experts", 10, GgmlDType::Q2K, 84_usize),
            ("q3k_experts", 11, GgmlDType::Q3K, 110_usize),
        ] {
            let mut data = vec![0; type_size * 2];
            // Keep both blocks numerically zero (`d=0`) while making their
            // payloads distinct, so the assertion below proves that expert 1
            // starts at the second slab rather than reusing expert 0.
            data[type_size] = 1;
            let tensors = [TestTensor {
                name,
                ggml_shape: vec![256, 1, 2],
                dtype: tag,
                data: data.clone(),
                offset: None,
            }];
            let encoded = make_gguf(&metadata, &tensors);
            let (_file, gguf) = open_bytes(&encoded)?;
            let info = gguf.tensor(name).unwrap();
            assert_eq!(info.shape, vec![2, 1, 256]);
            assert_eq!(info.byte_len, (type_size * 2) as u64);

            // SAFETY: the NamedTempFile remains alive and immutable.
            let backend = unsafe { gguf.backend(GgufBackendConfig::default()) }?;
            let bank = backend.load_matrix_bank(
                name,
                &Shape::from((2, 1, 256)),
                DType::F32,
                &Device::Cpu,
            )?;
            assert_eq!(bank.len(), 2);
            for (expert, matrix) in bank.into_iter().enumerate() {
                let GgufMatrix::Quantized(matrix) = matrix else {
                    panic!("{dtype:?} expert was unexpectedly expanded")
                };
                assert_eq!(matrix.dtype(), dtype);
                assert_eq!(matrix.shape().dims(), &[1, 256]);
                assert_eq!(matrix.storage_size_in_bytes(), type_size);
                assert_eq!(
                    matrix.data()?.as_ref(),
                    &data[expert * type_size..(expert + 1) * type_size]
                );
                assert!(
                    matrix
                        .dequantize(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .into_iter()
                        .all(|value| value == 0.0)
                );
            }

            // The bank loader independently validates that a rank-three
            // descriptor contains an exact whole number of expert slices.
            let mut malformed_tensors = backend.tensors.as_ref().clone();
            malformed_tensors.get_mut(name).unwrap().byte_len -= 2;
            let malformed = GgufBackend {
                shards: Arc::clone(&backend.shards),
                tensor_data_offsets: Arc::clone(&backend.tensor_data_offsets),
                tensors: Arc::new(malformed_tensors),
                tensor_shards: Arc::clone(&backend.tensor_shards),
                config: backend.config,
            };
            let error = malformed
                .load_matrix_bank(name, &Shape::from((2, 1, 256)), DType::F32, &Device::Cpu)
                .unwrap_err()
                .to_string();
            assert!(error.contains("payload bytes"), "{error}");

            // The index validator also rejects a real file whose final expert
            // payload is one byte short, excluding ordinary alignment padding.
            let padding = usize::try_from(
                align_up(data.len() as u64, DEFAULT_ALIGNMENT)
                    .context("test payload alignment overflow")?
                    - data.len() as u64,
            )?;
            let mut truncated = encoded;
            truncated.truncate(truncated.len() - padding - 1);
            let error = error_text(&truncated);
            assert!(error.contains("past file length"), "{error}");
        }
        Ok(())
    }

    #[test]
    fn parses_and_keeps_k_quant_and_mixed_fallback_blocks_compressed() -> Result<()> {
        let tensors = [
            TestTensor {
                name: "q2k",
                ggml_shape: vec![256, 1],
                dtype: 10,
                data: vec![0; 84],
                offset: None,
            },
            TestTensor {
                name: "q3k",
                ggml_shape: vec![256, 1],
                dtype: 11,
                data: vec![0; 110],
                offset: None,
            },
            TestTensor {
                name: "q4k",
                ggml_shape: vec![256, 1],
                dtype: 12,
                data: vec![0; 144],
                offset: None,
            },
            TestTensor {
                name: "q6k",
                ggml_shape: vec![256, 1],
                dtype: 14,
                data: vec![0; 210],
                offset: None,
            },
            TestTensor {
                name: "q5_0",
                ggml_shape: vec![64, 1],
                dtype: 6,
                data: vec![0; 44],
                offset: None,
            },
            TestTensor {
                name: "q5_1",
                ggml_shape: vec![64, 1],
                dtype: 7,
                data: vec![0; 48],
                offset: None,
            },
            TestTensor {
                name: "q5k",
                ggml_shape: vec![256, 1],
                dtype: 13,
                data: vec![0; 176],
                offset: None,
            },
        ];
        let metadata = [("general.quantization_version", GgufValue::U32(2))];
        let (_file, gguf) = open_bytes(&make_gguf(&metadata, &tensors))?;
        assert_eq!(gguf.tensor("q2k").unwrap().byte_len, 84);
        assert_eq!(gguf.tensor("q3k").unwrap().byte_len, 110);
        assert_eq!(gguf.tensor("q4k").unwrap().byte_len, 144);
        assert_eq!(gguf.tensor("q6k").unwrap().byte_len, 210);
        assert_eq!(gguf.tensor("q5_0").unwrap().byte_len, 44);
        assert_eq!(gguf.tensor("q5_1").unwrap().byte_len, 48);
        assert_eq!(gguf.tensor("q5k").unwrap().byte_len, 176);

        // SAFETY: the NamedTempFile remains alive and immutable.
        let backend = unsafe { gguf.backend(GgufBackendConfig::default()) }?;
        for (name, dtype, storage_bytes, elements) in [
            ("q2k", GgmlDType::Q2K, 84, 256),
            ("q3k", GgmlDType::Q3K, 110, 256),
            ("q4k", GgmlDType::Q4K, 144, 256),
            ("q6k", GgmlDType::Q6K, 210, 256),
            ("q5_0", GgmlDType::Q5_0, 44, 64),
            ("q5_1", GgmlDType::Q5_1, 48, 64),
            ("q5k", GgmlDType::Q5K, 176, 256),
        ] {
            let GgufMatrix::Quantized(tensor) =
                backend.load_matrix(name, DType::F32, &Device::Cpu)?
            else {
                panic!("{dtype:?} was unexpectedly expanded")
            };
            assert_eq!(tensor.dtype(), dtype);
            assert_eq!(tensor.storage_size_in_bytes(), storage_bytes);
            let values = tensor
                .dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(values.len(), elements);
            assert!(values.into_iter().all(|value| value == 0.0));
        }
        Ok(())
    }

    #[test]
    fn quantized_rows_must_not_cross_row_boundaries_and_require_version() {
        let invalid_width = [TestTensor {
            name: "q8",
            ggml_shape: vec![16, 2],
            dtype: 8,
            data: vec![0; 34],
            offset: None,
        }];
        let metadata = [("general.quantization_version", GgufValue::U32(2))];
        assert!(error_text(&make_gguf(&metadata, &invalid_width)).contains("row width 16"));

        for (name, dtype, data_len) in [("q2k", 10, 84), ("q3k", 11, 110), ("q4k", 12, 144)] {
            let invalid_k_width = [TestTensor {
                name,
                ggml_shape: vec![128, 2],
                dtype,
                data: vec![0; data_len],
                offset: None,
            }];
            let error = error_text(&make_gguf(&metadata, &invalid_k_width));
            assert!(error.contains("row width 128"), "{error}");
        }

        for (name, dtype, block_bytes) in [("q5_0", 6, 22), ("q5_1", 7, 24)] {
            let invalid_q5_pairing = [TestTensor {
                name,
                ggml_shape: vec![32, 2],
                dtype,
                data: vec![0; block_bytes * 2],
                offset: None,
            }];
            let error = error_text(&make_gguf(&metadata, &invalid_q5_pairing));
            assert!(error.contains("paired 32-element blocks"), "{error}");
        }

        let missing_version = [TestTensor {
            name: "q8",
            ggml_shape: vec![32],
            dtype: 8,
            data: vec![0; 34],
            offset: None,
        }];
        assert!(
            error_text(&make_gguf(&[], &missing_version))
                .contains("require general.quantization_version")
        );
    }

    #[test]
    fn inverse_permutates_compressed_rows_across_heads_and_blocks() -> Result<()> {
        let mut q8_data = Vec::new();
        for value in 1_i8..=8 {
            for _ in 0..2 {
                q8_data.extend_from_slice(&0x3c00_u16.to_le_bytes()); // f16(1.0)
                q8_data.extend(std::iter::repeat_n(value.cast_unsigned(), 32));
            }
        }

        let mut q4_data = Vec::new();
        for code in 0_u8..8 {
            for _ in 0..2 {
                q4_data.extend_from_slice(&0x3c00_u16.to_le_bytes()); // f16(1.0)
                q4_data.extend(std::iter::repeat_n(code | (code << 4), 16));
            }
        }

        for (tag, dtype, data, expected, storage_bytes) in [
            (
                8,
                GgmlDType::Q8_0,
                q8_data,
                vec![1., 3., 2., 4., 5., 7., 6., 8.],
                8 * 2 * 34,
            ),
            (
                2,
                GgmlDType::Q4_0,
                q4_data,
                vec![-8., -6., -7., -5., -4., -2., -3., -1.],
                8 * 2 * 18,
            ),
        ] {
            let tensor = [TestTensor {
                name: "blk.0.attn_q.weight",
                ggml_shape: vec![64, 8],
                dtype: tag,
                data,
                offset: None,
            }];
            let metadata = [("general.quantization_version", GgufValue::U32(2))];
            let (_file, gguf) = open_bytes(&make_gguf(&metadata, &tensor))?;
            // SAFETY: the NamedTempFile remains alive and immutable.
            let backend = unsafe {
                gguf.backend(GgufBackendConfig {
                    llama_qk: Some(LlamaQkConfig {
                        query_heads: 2,
                        kv_heads: 1,
                    }),
                    ..GgufBackendConfig::default()
                })
            }?;
            let GgufMatrix::Quantized(query) = backend.load_matrix(
                "model.layers.0.self_attn.q_proj.weight",
                DType::F32,
                &Device::Cpu,
            )?
            else {
                panic!("{dtype:?} query was unexpectedly expanded")
            };
            assert_eq!(query.dtype(), dtype);
            assert_eq!(query.storage_size_in_bytes(), storage_bytes);
            let rows = query.dequantize(&Device::Cpu)?.to_vec2::<f32>()?;
            assert_eq!(rows.iter().map(|row| row[0]).collect::<Vec<_>>(), expected);
        }
        Ok(())
    }

    #[test]
    fn rejects_noncanonical_keys_names_ranks_and_extents() {
        assert!(
            error_text(&make_gguf(&[("Bad-Key", GgufValue::U32(1))], &[]))
                .contains("dot-separated lowercase identifier")
        );

        let long_name = [TestTensor {
            name: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ggml_shape: vec![1],
            dtype: 0,
            data: f32_bytes(&[1.]),
            offset: None,
        }];
        assert!(error_text(&make_gguf(&[], &long_name)).contains("maximum is 63"));

        let scalar = [TestTensor {
            name: "scalar",
            ggml_shape: vec![],
            dtype: 0,
            data: f32_bytes(&[1.]),
            offset: None,
        }];
        assert!(error_text(&make_gguf(&[], &scalar)).contains("supported ranks"));

        let zero_extent = [TestTensor {
            name: "empty",
            ggml_shape: vec![0],
            dtype: 0,
            data: vec![],
            offset: None,
        }];
        assert!(error_text(&make_gguf(&[], &zero_extent)).contains("zero-sized dimension"));
    }
}
