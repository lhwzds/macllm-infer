//! Format-neutral model-weight loading and dense/quantized decoder operators.

use std::sync::Arc;

use candle_core::{
    DType, Device, Error, Module, Result, Shape, Tensor,
    quantized::{GgmlDType, QMatMul, QTensor},
};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder};

#[cfg(feature = "metal")]
use crate::file_backed_quant::FileBackedQuantizedTensor;
use crate::gguf::{GgufBackend, GgufMatrix};

/// A matrix whose compressed representation is preserved through model build.
#[derive(Debug, Clone)]
pub enum MatrixWeight {
    Dense(Tensor),
    Quantized(Arc<QTensor>),
    #[cfg(feature = "metal")]
    FileBackedQuantized(Arc<FileBackedQuantizedTensor>),
}

impl MatrixWeight {
    fn shape(&self) -> &Shape {
        match self {
            Self::Dense(tensor) => tensor.shape(),
            Self::Quantized(tensor) => tensor.shape(),
            #[cfg(feature = "metal")]
            Self::FileBackedQuantized(tensor) => tensor.shape(),
        }
    }
}

/// The only loading interface used by the shared decoder.
pub trait WeightSource {
    fn dtype(&self) -> DType;
    fn device(&self) -> &Device;
    /// Report whether an exact tensor name exists when the source can inspect
    /// its namespace without loading the tensor. `None` keeps callers on their
    /// canonical path instead of guessing aliases.
    fn contains_tensor(&self, _name: &str) -> Option<bool> {
        None
    }
    fn tensor(&self, shape: Shape, name: &str) -> Result<Tensor>;

    /// Load a dense tensor before its accepted shape has been normalized.
    /// Consumers must validate the result. Sources that cannot inspect stored
    /// shapes may keep the default and use an exact-shape fallback instead.
    fn tensor_unchecked(&self, name: &str) -> Result<Tensor> {
        Err(Error::Msg(format!(
            "weight source cannot load `{name}` without an expected shape"
        )))
    }

    fn matrix(&self, shape: Shape, name: &str) -> Result<MatrixWeight>;

    /// Load a rank-three `[experts, rows, columns]` weight as independent
    /// matrices. Dense sources use tensor slices; GGUF overrides this method to
    /// preserve compressed expert slices.
    fn matrix_bank(&self, shape: Shape, name: &str) -> Result<Vec<MatrixWeight>> {
        let &[expert_count, rows, columns] = shape.dims() else {
            candle_core::bail!(
                "matrix bank `{name}` must have shape [experts, rows, columns], found {shape:?}"
            );
        };
        let tensor = self.tensor(shape, name)?;
        (0..expert_count)
            .map(|expert| {
                let matrix = tensor.narrow(0, expert, 1)?.squeeze(0)?.contiguous()?;
                ensure_shape(name, &Shape::from((rows, columns)), matrix.shape())?;
                Ok(MatrixWeight::Dense(matrix))
            })
            .collect()
    }
}

/// Compatibility wrapper for Safetensors, tests, and other dense builders.
pub struct DenseWeightSource<'a> {
    weights: VarBuilder<'a>,
}

impl<'a> DenseWeightSource<'a> {
    #[must_use]
    pub fn new(weights: VarBuilder<'a>) -> Self {
        Self { weights }
    }
}

impl WeightSource for DenseWeightSource<'_> {
    fn dtype(&self) -> DType {
        self.weights.dtype()
    }

    fn device(&self) -> &Device {
        self.weights.device()
    }

    fn contains_tensor(&self, name: &str) -> Option<bool> {
        Some(self.weights.contains_tensor(name))
    }

    fn tensor(&self, shape: Shape, name: &str) -> Result<Tensor> {
        self.weights.get(shape, name)
    }

    fn tensor_unchecked(&self, name: &str) -> Result<Tensor> {
        self.weights.get_unchecked(name)
    }

    fn matrix(&self, shape: Shape, name: &str) -> Result<MatrixWeight> {
        self.tensor(shape, name).map(MatrixWeight::Dense)
    }
}

/// GGUF source backed by the project's validated parser and mmap index.
pub struct GgufWeightSource {
    backend: GgufBackend,
    dtype: DType,
    device: Device,
    preserve_quantized: bool,
}

impl GgufWeightSource {
    #[must_use]
    pub fn new(backend: GgufBackend, dtype: DType, device: Device) -> Self {
        Self {
            backend,
            dtype,
            device,
            preserve_quantized: true,
        }
    }

    #[must_use]
    pub fn new_dequantizing(backend: GgufBackend, dtype: DType, device: Device) -> Self {
        Self {
            backend,
            dtype,
            device,
            preserve_quantized: false,
        }
    }
}

impl WeightSource for GgufWeightSource {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn tensor(&self, shape: Shape, name: &str) -> Result<Tensor> {
        let tensor = self.backend.load_tensor(name, self.dtype, &self.device)?;
        ensure_shape(name, &shape, tensor.shape())?;
        Ok(tensor)
    }

    fn tensor_unchecked(&self, name: &str) -> Result<Tensor> {
        self.backend.load_tensor(name, self.dtype, &self.device)
    }

    fn matrix(&self, shape: Shape, name: &str) -> Result<MatrixWeight> {
        let loaded = if self.preserve_quantized {
            self.backend
                .load_matrix_preserving(name, self.dtype, &self.device)?
        } else {
            self.backend.load_matrix(name, self.dtype, &self.device)?
        };
        let matrix = match loaded {
            GgufMatrix::Dense(tensor) => MatrixWeight::Dense(tensor),
            GgufMatrix::Quantized(tensor) if self.preserve_quantized => {
                MatrixWeight::Quantized(tensor)
            }
            GgufMatrix::Quantized(tensor) => {
                MatrixWeight::Dense(tensor.dequantize(&self.device)?.to_dtype(self.dtype)?)
            }
            #[cfg(feature = "metal")]
            GgufMatrix::FileBackedQuantized(tensor) => MatrixWeight::FileBackedQuantized(tensor),
        };
        ensure_shape(name, &shape, matrix.shape())?;
        Ok(matrix)
    }

    fn matrix_bank(&self, shape: Shape, name: &str) -> Result<Vec<MatrixWeight>> {
        let &[expert_count, rows, columns] = shape.dims() else {
            candle_core::bail!(
                "matrix bank `{name}` must have shape [experts, rows, columns], found {shape:?}"
            );
        };
        let bank = if self.preserve_quantized {
            self.backend
                .load_matrix_bank_preserving(name, &shape, self.dtype, &self.device)?
        } else {
            self.backend
                .load_matrix_bank(name, &shape, self.dtype, &self.device)?
        };
        if bank.len() != expert_count {
            candle_core::bail!(
                "matrix bank `{name}` returned {} experts, expected {expert_count}",
                bank.len()
            );
        }
        let expected = Shape::from((rows, columns));
        bank.into_iter()
            .map(|matrix| {
                let matrix = match matrix {
                    GgufMatrix::Dense(tensor) => MatrixWeight::Dense(tensor),
                    GgufMatrix::Quantized(tensor) if self.preserve_quantized => {
                        MatrixWeight::Quantized(tensor)
                    }
                    GgufMatrix::Quantized(tensor) => {
                        MatrixWeight::Dense(tensor.dequantize(&self.device)?.to_dtype(self.dtype)?)
                    }
                    #[cfg(feature = "metal")]
                    GgufMatrix::FileBackedQuantized(tensor) => {
                        MatrixWeight::FileBackedQuantized(tensor)
                    }
                };
                ensure_shape(name, &expected, matrix.shape())?;
                Ok(matrix)
            })
            .collect()
    }
}

fn ensure_shape(name: &str, expected: &Shape, actual: &Shape) -> Result<()> {
    if expected != actual {
        return Err(Error::UnexpectedShape {
            msg: format!("shape mismatch for {name}"),
            expected: expected.clone(),
            got: actual.clone(),
        });
    }
    Ok(())
}

/// A path cursor over a borrowed weight source, analogous to `VarBuilder::pp`.
#[derive(Clone)]
pub struct WeightPath<'a> {
    source: &'a dyn WeightSource,
    prefix: String,
}

impl<'a> WeightPath<'a> {
    #[must_use]
    pub fn root(source: &'a dyn WeightSource) -> Self {
        Self {
            source,
            prefix: String::new(),
        }
    }

    #[must_use]
    pub fn pp(&self, segment: impl std::fmt::Display) -> Self {
        let segment = segment.to_string();
        let prefix = if self.prefix.is_empty() {
            segment
        } else {
            format!("{}.{segment}", self.prefix)
        };
        Self {
            source: self.source,
            prefix,
        }
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.source.dtype()
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        self.source.device()
    }

    fn name(&self, leaf: &str) -> String {
        if self.prefix.is_empty() {
            leaf.to_owned()
        } else {
            format!("{}.{leaf}", self.prefix)
        }
    }

    pub(crate) fn tensor(&self, shape: impl Into<Shape>, leaf: &str) -> Result<Tensor> {
        self.source.tensor(shape.into(), &self.name(leaf))
    }

    pub(crate) fn tensor_unchecked(&self, leaf: &str) -> Result<Tensor> {
        self.source.tensor_unchecked(&self.name(leaf))
    }

    fn matrix(&self, shape: impl Into<Shape>, leaf: &str) -> Result<MatrixWeight> {
        self.source.matrix(shape.into(), &self.name(leaf))
    }

    fn matrix_bank(&self, shape: impl Into<Shape>, leaf: &str) -> Result<Vec<MatrixWeight>> {
        self.source.matrix_bank(shape.into(), &self.name(leaf))
    }
}

/// Decoder projection that dispatches to dense GEMM or a compressed `QMatMul`.
#[derive(Debug, Clone)]
pub enum LinearOp {
    Dense(Linear),
    Quantized {
        weight: QMatMul,
        bias: Option<Tensor>,
    },
    #[cfg(feature = "metal")]
    FileBackedQuantized {
        weight: Arc<FileBackedQuantizedTensor>,
        bias: Option<Tensor>,
    },
}

impl LinearOp {
    fn new(weight: MatrixWeight, bias: Option<Tensor>) -> Self {
        match weight {
            MatrixWeight::Dense(weight) => Self::Dense(Linear::new(weight, bias)),
            MatrixWeight::Quantized(weight) => Self::Quantized {
                // Constructing this variant directly makes preservation of the
                // compressed representation independent of environment flags.
                weight: QMatMul::QTensor(weight),
                bias,
            },
            #[cfg(feature = "metal")]
            MatrixWeight::FileBackedQuantized(weight) => Self::FileBackedQuantized { weight, bias },
        }
    }

    #[must_use]
    pub fn is_quantized(&self) -> bool {
        match self {
            Self::Dense(_) => false,
            Self::Quantized { .. } => true,
            #[cfg(feature = "metal")]
            Self::FileBackedQuantized { .. } => true,
        }
    }
}

impl Module for LinearOp {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(input),
            Self::Quantized { weight, bias } => {
                // Candle 0.11's Metal quantized kernels consume F32. Keeping
                // this conversion at the operator boundary also makes an F16
                // dense runtime safe when it encounters a quantized matrix.
                let output_dtype = input.dtype();
                let input = input.to_dtype(DType::F32)?.contiguous()?;
                let output = if lower_bit_k_quant_needs_metal_mm(&input, weight)? {
                    // Candle 0.11 dispatches Q2_K Metal matrix-vector work in
                    // four-row groups even though its shader consumes eight,
                    // which can corrupt adjacent buffers during one-token
                    // decode. Q3_K only has the analogous tail hazard when its
                    // output row count is not divisible by four. Its compressed
                    // matrix-matrix kernels are safe, so duplicate the row to
                    // select MM and retain a contiguous first result.
                    let row_axis = input.rank() - 2;
                    Tensor::cat(&[&input, &input], row_axis)?
                        .apply(weight)?
                        .narrow(row_axis, 0, 1)?
                        .contiguous()?
                } else {
                    input.apply(weight)?
                }
                .to_dtype(output_dtype)?;
                match bias {
                    Some(bias) => output.broadcast_add(bias),
                    None => Ok(output),
                }
            }
            #[cfg(feature = "metal")]
            Self::FileBackedQuantized { weight, bias } => {
                let output_dtype = input.dtype();
                let input = input.to_dtype(DType::F32)?.contiguous()?;
                let output = if lower_bit_k_quant_needs_metal_mm_for(
                    &input,
                    weight.dtype(),
                    weight.shape().dim(0)?,
                )? {
                    let row_axis = input.rank() - 2;
                    weight
                        .matmul(&Tensor::cat(&[&input, &input], row_axis)?)?
                        .narrow(row_axis, 0, 1)?
                        .contiguous()?
                } else {
                    weight.matmul(&input)?
                }
                .to_dtype(output_dtype)?;
                match bias {
                    Some(bias) => output.broadcast_add(bias),
                    None => Ok(output),
                }
            }
        }
    }
}

fn lower_bit_k_quant_needs_metal_mm(input: &Tensor, weight: &QMatMul) -> Result<bool> {
    if !input.device().is_metal() || input.rank() < 2 {
        return Ok(false);
    }
    let QMatMul::QTensor(weight) = weight else {
        return Ok(false);
    };
    if input.dim(input.rank() - 2)? != 1 {
        return Ok(false);
    }
    let (output_rows, _) = weight.shape().dims2()?;
    lower_bit_k_quant_needs_metal_mm_for(input, weight.dtype(), output_rows)
}

fn lower_bit_k_quant_needs_metal_mm_for(
    input: &Tensor,
    dtype: GgmlDType,
    output_rows: usize,
) -> Result<bool> {
    if !input.device().is_metal() || input.rank() < 2 {
        return Ok(false);
    }
    if input.dim(input.rank() - 2)? != 1 {
        return Ok(false);
    }
    Ok(dtype == GgmlDType::Q2K || (dtype == GgmlDType::Q3K && !output_rows.is_multiple_of(4)))
}

/// Token lookup that preserves a quantized vocabulary matrix and can share it
/// with a tied output projection.
#[derive(Debug, Clone)]
pub enum TokenEmbedding {
    Dense(Embedding),
    Quantized {
        weight: Arc<QTensor>,
        output_dtype: DType,
    },
    #[cfg(feature = "metal")]
    FileBackedQuantized {
        weight: Arc<FileBackedQuantizedTensor>,
        output_dtype: DType,
    },
}

impl TokenEmbedding {
    pub(crate) fn tied_linear(&self) -> LinearOp {
        match self {
            Self::Dense(embedding) => {
                LinearOp::Dense(Linear::new(embedding.embeddings().clone(), None))
            }
            Self::Quantized { weight, .. } => LinearOp::Quantized {
                weight: QMatMul::QTensor(Arc::clone(weight)),
                bias: None,
            },
            #[cfg(feature = "metal")]
            Self::FileBackedQuantized { weight, .. } => LinearOp::FileBackedQuantized {
                weight: Arc::clone(weight),
                bias: None,
            },
        }
    }

    pub(crate) fn validate_ids(&self, input_ids: &Tensor) -> Result<()> {
        if input_ids.dtype() != DType::U32 {
            candle_core::bail!(
                "token embedding expects U32 ids, found {:?}",
                input_ids.dtype()
            );
        }
        let vocab_size = match self {
            Self::Dense(embedding) => embedding.embeddings().dim(0)?,
            Self::Quantized { weight, .. } => weight.shape().dim(0)?,
            #[cfg(feature = "metal")]
            Self::FileBackedQuantized { weight, .. } => weight.shape().dim(0)?,
        };
        let max_id = input_ids.max_all()?.to_scalar::<u32>()?;
        if max_id as usize >= vocab_size {
            candle_core::bail!(
                "token id {max_id} is outside embedding vocabulary of {vocab_size} rows"
            );
        }
        Ok(())
    }

    pub(crate) fn forward_validated(&self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(embedding) => embedding.forward(input_ids),
            Self::Quantized {
                weight,
                output_dtype,
            } => weight.embedding(input_ids)?.to_dtype(*output_dtype),
            #[cfg(feature = "metal")]
            Self::FileBackedQuantized {
                weight,
                output_dtype,
            } => weight.embedding(input_ids)?.to_dtype(*output_dtype),
        }
    }
}

impl Module for TokenEmbedding {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Candle 0.11's quantized Metal get-rows kernel does not receive the
        // vocabulary row count, so an unchecked id could read past the weight
        // buffer. Validate before dispatching that kernel.
        self.validate_ids(input_ids)?;
        self.forward_validated(input_ids)
    }
}

pub fn embedding(vocab: usize, hidden: usize, path: WeightPath<'_>) -> Result<TokenEmbedding> {
    match path.matrix((vocab, hidden), "weight")? {
        MatrixWeight::Dense(weight) => Ok(TokenEmbedding::Dense(Embedding::new(weight, hidden))),
        MatrixWeight::Quantized(weight) => Ok(TokenEmbedding::Quantized {
            weight,
            output_dtype: path.dtype(),
        }),
        #[cfg(feature = "metal")]
        MatrixWeight::FileBackedQuantized(weight) => Ok(TokenEmbedding::FileBackedQuantized {
            weight,
            output_dtype: path.dtype(),
        }),
    }
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    path: WeightPath<'_>,
) -> Result<LinearOp> {
    let bias = if bias {
        Some(path.tensor(out_dim, "bias")?)
    } else {
        None
    };
    Ok(LinearOp::new(
        path.matrix((out_dim, in_dim), "weight")?,
        bias,
    ))
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, path: WeightPath<'_>) -> Result<LinearOp> {
    linear_b(in_dim, out_dim, false, path)
}

pub fn expert_linears_no_bias(
    expert_count: usize,
    in_dim: usize,
    out_dim: usize,
    path: WeightPath<'_>,
) -> Result<Vec<LinearOp>> {
    path.matrix_bank((expert_count, out_dim, in_dim), "weight")?
        .into_iter()
        .map(|weight| Ok(LinearOp::new(weight, None)))
        .collect()
}

pub fn rms_norm(size: usize, epsilon: f64, path: WeightPath<'_>) -> Result<RmsNorm> {
    Ok(RmsNorm::new(path.tensor(size, "weight")?, epsilon))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::quantized::GgmlDType;

    use super::*;

    #[test]
    fn dense_source_reports_and_loads_only_exact_names() -> Result<()> {
        let device = Device::Cpu;
        let root = Tensor::new(&[[1_f32, 2.]], &device)?;
        let source = DenseWeightSource::new(VarBuilder::from_tensors(
            HashMap::from([
                ("embed_tokens.weight".to_owned(), root.clone()),
                (
                    "lm_head.weight".to_owned(),
                    Tensor::new(&[[5_f32, 6.]], &device)?,
                ),
            ]),
            DType::F32,
            &device,
        ));

        assert_eq!(source.contains_tensor("embed_tokens.weight"), Some(true));
        assert_eq!(
            source.contains_tensor("model.embed_tokens.weight"),
            Some(false)
        );
        assert_eq!(
            source
                .tensor(Shape::from((1, 2)), "embed_tokens.weight")?
                .to_vec2::<f32>()?,
            vec![vec![1., 2.]]
        );
        assert_eq!(
            source
                .tensor_unchecked("embed_tokens.weight")?
                .to_vec2::<f32>()?,
            vec![vec![1., 2.]]
        );
        assert_eq!(
            source
                .tensor(Shape::from((1, 2)), "lm_head.weight")?
                .to_vec2::<f32>()?,
            vec![vec![5., 6.]]
        );
        assert!(
            source
                .tensor(Shape::from((1, 2)), "model.embed_tokens.weight")
                .is_err(),
            "the generic source must not invent decoder aliases"
        );
        Ok(())
    }

    #[test]
    fn supported_quantized_linear_ops_match_same_weight_dense_references() -> Result<()> {
        let device = Device::Cpu;
        let bias = Tensor::new(&[0.25_f32, -0.5], &device)?;

        for dtype in [
            GgmlDType::Q8_0,
            GgmlDType::Q4_0,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            // Candle's CPU Q5_0 dot product consumes pairs of blocks. Two
            // blocks also exercise row-stride handling for every format.
            let width = dtype.block_size() * 2;
            let values = (0..2 * width)
                .map(|index| ((index * 17 % 41) as f32 - 20.0) / 20.0)
                .collect::<Vec<_>>();
            let source = Tensor::from_vec(values, (2, width), &device)?;
            let input = Tensor::from_vec(
                (0..3 * width)
                    .map(|index| ((index * 13 % 29) as f32 - 14.0) / 14.0)
                    .collect::<Vec<_>>(),
                (1, 3, width),
                &device,
            )?;
            let weight = Arc::new(QTensor::quantize(&source, dtype)?);
            let quantized = LinearOp::new(
                MatrixWeight::Quantized(Arc::clone(&weight)),
                Some(bias.clone()),
            );
            assert!(quantized.is_quantized());
            let dense = LinearOp::new(
                MatrixWeight::Dense(weight.dequantize(&device)?),
                Some(bias.clone()),
            );
            let actual = quantized.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
            let expected = dense.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
            let difference = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);
            let tolerance = match dtype {
                // The lower-bit K-quants accumulate a little more rounding
                // error in Candle's direct dot-product path than the formats
                // below, even against their own dequantized weights.
                GgmlDType::Q2K => 0.06,
                _ => 0.05,
            };
            assert!(
                difference < tolerance,
                "{dtype:?} direct matmul differed from its dense reference by {difference} (tolerance {tolerance})"
            );
        }
        Ok(())
    }

    #[test]
    fn supported_quantized_embeddings_and_tied_heads_share_compressed_storage() -> Result<()> {
        let device = Device::Cpu;
        for dtype in [
            GgmlDType::Q8_0,
            GgmlDType::Q4_0,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            let width = dtype.block_size() * 2;
            let values = (0..4 * width)
                .map(|index| ((index * 11 % 31) as f32 - 15.0) / 15.0)
                .collect::<Vec<_>>();
            let source = Tensor::from_vec(values, (4, width), &device)?;
            let weight = Arc::new(QTensor::quantize(&source, dtype)?);
            let embedding = TokenEmbedding::Quantized {
                weight: Arc::clone(&weight),
                output_dtype: DType::F32,
            };
            let references_before = Arc::strong_count(&weight);
            let tied = embedding.tied_linear();
            assert!(tied.is_quantized());
            assert_eq!(Arc::strong_count(&weight), references_before + 1);

            let ids = Tensor::new(&[[1_u32, 3]], &device)?;
            let embedded = embedding.forward(&ids)?;
            let actual = embedded.to_vec3::<f32>()?;
            let dense_weight = weight.dequantize(&device)?;
            let dense = dense_weight.to_vec2::<f32>()?;
            assert_eq!(actual[0], vec![dense[1].clone(), dense[3].clone()]);

            let tied_logits = tied.forward(&embedded)?.flatten_all()?.to_vec1::<f32>()?;
            let dense_head = Linear::new(dense_weight, None);
            let dense_logits = dense_head
                .forward(&embedded)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let difference = tied_logits
                .iter()
                .zip(dense_logits)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);
            // QMatMul requantizes the activation while the dense path does
            // not; the exact embedding-row equality above separately guards
            // get_rows and shared-weight orientation.
            let tolerance = match dtype {
                GgmlDType::Q2K => 0.25,
                GgmlDType::Q3K => 0.2,
                _ => 0.15,
            };
            assert!(
                difference < tolerance,
                "{dtype:?} tied head differed from its dense reference by {difference} (tolerance {tolerance})"
            );

            let invalid = Tensor::new(&[[4_u32]], &device)?;
            let error = embedding.forward(&invalid).unwrap_err().to_string();
            assert!(error.contains("outside embedding vocabulary"));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an accessible Apple Metal device"]
    fn lower_bit_k_quant_single_row_metal_dispatch_is_safe_and_accurate() -> Result<()> {
        let cpu = Device::Cpu;
        let metal = Device::new_metal(0)?;
        for (dtype, output_rows, expects_workaround) in [
            (GgmlDType::Q2K, 4, true),
            (GgmlDType::Q3K, 4, false),
            (GgmlDType::Q3K, 2, true),
        ] {
            let width = dtype.block_size() * 2;
            let source = Tensor::from_vec(
                (0..output_rows * width)
                    .map(|index| ((index * 17 % 41) as f32 - 20.0) / 20.0)
                    .collect::<Vec<_>>(),
                (output_rows, width),
                &cpu,
            )?;
            let input = Tensor::from_vec(
                (0..2 * width)
                    .map(|index| ((((index % width) * 13) % 29) as f32 - 14.0) / 14.0)
                    .collect::<Vec<_>>(),
                (2, 1, width),
                &cpu,
            )?
            .to_device(&metal)?;
            let weight = Arc::new(QTensor::quantize_onto(&source, dtype, &metal)?);
            let op = LinearOp::new(MatrixWeight::Quantized(Arc::clone(&weight)), None);
            let qmatmul = QMatMul::QTensor(Arc::clone(&weight));
            assert_eq!(
                lower_bit_k_quant_needs_metal_mm(&input, &qmatmul)?,
                expects_workaround,
                "unexpected {dtype:?} dispatch for {output_rows} output rows"
            );

            let actual = op.forward(&input)?;
            assert_eq!(actual.dims3()?, (2, 1, output_rows));
            assert!(actual.is_contiguous());
            let actual = actual.to_device(&cpu)?.to_vec3::<f32>()?;
            assert!(
                actual
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|value| value.is_finite()),
                "{dtype:?} single-row Metal output contained a non-finite value"
            );
            assert_eq!(
                actual[0], actual[1],
                "{dtype:?} duplicate batch rows produced different output"
            );

            let dense_weight = weight.dequantize(&cpu)?;
            let expected = input
                .to_device(&cpu)?
                .reshape((2, width))?
                .matmul(&dense_weight.t()?)?
                .reshape((2, 1, output_rows))?
                .to_vec3::<f32>()?;
            let difference = actual
                .iter()
                .flatten()
                .flatten()
                .zip(expected.iter().flatten().flatten())
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                difference < 0.01,
                "{dtype:?} Metal output differed from its dense reference by {difference}"
            );
        }
        Ok(())
    }
}
