//! Cross-format parity gates for fixed `SmolLM2` Safetensors and GGUF files.
//!
//! `MACLLM_LLAMA_MODEL_DIR=models/smollm2-135m-instruct \
//!  MACLLM_GGUF_MODEL=models/smollm2-135m-instruct-gguf \
//!  cargo test --release --test gguf_reference -- --ignored --nocapture`

use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use candle_core::{
    DType, Device, Module, Tensor,
    quantized::{GgmlDType, QTensor, gguf_file},
};
use candle_transformers::models::quantized_llama;
use macllm_infer::{
    generation::{GenerationConfig, generate},
    gguf::{GgufFile, GgufTensorType},
    model_weights::TokenEmbedding,
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};

const MAX_F16_CROSS_FORMAT_DIFFERENCE: f32 = 0.05;
const MAX_Q8_CROSS_FORMAT_DIFFERENCE: f32 = 0.75;
const MAX_Q4_CROSS_FORMAT_DIFFERENCE: f32 = 6.5;
const MAX_QUANTIZED_KERNEL_DIFFERENCE: f32 = 0.02;
const MAX_CANDLE_SAME_SHAPE_DIFFERENCE: f32 = 0.001;
const MAX_CANDLE_SAME_SHAPE_RELATIVE_RMS: f64 = 0.0001;
const MAX_CANDLE_FRESH_FULL_DIFFERENCE: f32 = 0.03;
const MAX_CANDLE_FRESH_FULL_RELATIVE_RMS: f64 = 0.002;
// Candle's process-global Metal resource registry and these multi-model gates
// must not be initialized concurrently inside one test binary.
static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires an accessible Apple Metal device"]
fn supported_quantized_metal_embeddings_match_dequantized_rows() -> anyhow::Result<()> {
    let _metal_guard = METAL_TEST_LOCK
        .lock()
        .expect("GGUF Metal test lock poisoned");
    let cpu = Device::Cpu;
    let metal = Device::new_metal(0)?;
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
        let source = Tensor::from_vec(values, (4, width), &cpu)?;
        let weight = Arc::new(QTensor::quantize_onto(&source, dtype, &metal)?);
        let embedding = TokenEmbedding::Quantized {
            weight: Arc::clone(&weight),
            output_dtype: DType::F32,
        };
        let ids = Tensor::new(&[[1_u32, 3]], &metal)?;
        let actual = embedding.forward(&ids)?.to_device(&cpu)?.to_vec3::<f32>()?;
        let dense = weight.dequantize(&cpu)?.to_vec2::<f32>()?;
        let expected = [dense[1].clone(), dense[3].clone()];
        let difference = actual[0]
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            difference < 1e-4,
            "{dtype:?} Metal get_rows differed from dequantized rows by {difference}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires both fixed SmolLM2 checkpoints and an accessible Apple Metal device"]
fn gguf_matches_safetensors_tokenizer_prefill_and_cached_decode() -> anyhow::Result<()> {
    let _metal_guard = METAL_TEST_LOCK
        .lock()
        .expect("GGUF Metal test lock poisoned");
    let safetensors =
        ModelFiles::from_dir(PathBuf::from(std::env::var("MACLLM_LLAMA_MODEL_DIR")?))?;
    let gguf = ModelFiles::from_dir(PathBuf::from(std::env::var("MACLLM_GGUF_MODEL")?))?;
    assert_same_model_contract(&safetensors, &gguf)?;

    let device = Device::new_metal(0)?;
    let dtype = if gguf.has_quantized_tensors() {
        DType::F32
    } else {
        DType::F16
    };
    let is_q4 = gguf.has_gguf_tensor_type(GgufTensorType::Q4_0)
        || gguf.has_gguf_tensor_type(GgufTensorType::Q4K);
    let tolerance = if is_q4 {
        MAX_Q4_CROSS_FORMAT_DIFFERENCE
    } else if gguf.has_quantized_tensors() {
        MAX_Q8_CROSS_FORMAT_DIFFERENCE
    } else {
        MAX_F16_CROSS_FORMAT_DIFFERENCE
    };
    let input = Tensor::new(&[[1_u32, 100, 200, 300]], &device)?;
    let (safetensors_prefill, safetensors_cached, next_token) =
        prefill_and_decode(&safetensors, &input, &device, dtype, None)?;
    let (gguf_prefill, gguf_cached, _) =
        prefill_and_decode(&gguf, &input, &device, dtype, Some(next_token))?;

    if !is_q4 {
        assert_eq!(argmax(&gguf_prefill), argmax(&safetensors_prefill));
    }
    assert_logits_close(
        "GGUF/Safetensors prefill",
        &gguf_prefill,
        &safetensors_prefill,
        tolerance,
    );
    if !is_q4 {
        assert_eq!(argmax(&gguf_cached), argmax(&safetensors_cached));
    }
    assert_logits_close(
        "GGUF/Safetensors cached decode",
        &gguf_cached,
        &safetensors_cached,
        tolerance,
    );

    let combined = Tensor::new(&[[1_u32, 100, 200, 300, next_token]], &device)?;
    let safetensors_full = fresh_forward(&safetensors, &combined, &device, dtype)?;
    let gguf_full = fresh_forward(&gguf, &combined, &device, dtype)?;
    assert_logits_close("GGUF cache/full", &gguf_cached, &gguf_full, tolerance);
    assert_logits_close(
        "GGUF/Safetensors full",
        &gguf_full,
        &safetensors_full,
        tolerance,
    );
    Ok(())
}

fn assert_same_model_contract(safetensors: &ModelFiles, gguf: &ModelFiles) -> anyhow::Result<()> {
    let safetensors_adapter = safetensors.adapter()?;
    let gguf_adapter = gguf.adapter()?;
    let safetensors_spec = safetensors_adapter
        .decoder_spec()
        .expect("SmolLM2 uses the shared decoder");
    let gguf_spec = gguf_adapter
        .decoder_spec()
        .expect("SmolLM2 GGUF uses the shared decoder");
    assert_eq!(gguf_spec.runtime, safetensors_spec.runtime);
    assert_eq!(gguf_spec.hidden_size, safetensors_spec.hidden_size);
    assert_eq!(
        gguf_spec.intermediate_size,
        safetensors_spec.intermediate_size
    );
    assert_eq!(gguf_spec.num_layers, safetensors_spec.num_layers);
    assert_eq!(gguf_spec.num_query_heads, safetensors_spec.num_query_heads);
    assert_eq!(gguf_spec.num_kv_heads, safetensors_spec.num_kv_heads);
    assert_eq!(gguf_spec.head_dim, safetensors_spec.head_dim);
    assert_eq!(gguf_spec.tied_embeddings, safetensors_spec.tied_embeddings);
    assert!((gguf_spec.norm_epsilon - safetensors_spec.norm_epsilon).abs() < 1e-10);
    assert!(
        (gguf_spec.layer_specs[0]
            .attention()
            .expect("GGUF Llama layer uses attention")
            .rope()
            .expect("GGUF Llama layer uses RoPE")
            .theta
            - safetensors_spec.layer_specs[0]
                .attention()
                .expect("Safetensors Llama layer uses attention")
                .rope()
                .expect("Safetensors Llama layer uses RoPE")
                .theta)
            .abs()
            < f64::EPSILON
    );

    let safetensors_tokenizer = safetensors.tokenizer()?;
    let gguf_tokenizer = gguf.tokenizer()?;
    let rendered = safetensors
        .adapter()?
        .format_chat(None, "What is the capital of France?")?;
    assert_eq!(
        rendered,
        gguf.adapter()?
            .format_chat(None, "What is the capital of France?")?
    );
    for text in [
        "Hello",
        "The answer is 12345.",
        "café 🦀 — Rust",
        "<|im_start|>assistant\nParis<|im_end|>",
        rendered.as_str(),
    ] {
        let expected = safetensors_tokenizer
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let actual = gguf_tokenizer
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(
            actual.get_ids(),
            expected.get_ids(),
            "token IDs for {text:?}"
        );
        let expected_decoded = safetensors_tokenizer
            .decode(expected.get_ids(), false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let actual_decoded = gguf_tokenizer
            .decode(actual.get_ids(), false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(
            actual_decoded, expected_decoded,
            "decoded text for {text:?}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a fixed quantized Llama-family GGUF and an accessible Apple Metal device"]
fn quantized_gguf_matches_same_file_dense_reference() -> anyhow::Result<()> {
    let _metal_guard = METAL_TEST_LOCK
        .lock()
        .expect("GGUF Metal test lock poisoned");
    let model_path = PathBuf::from(std::env::var("MACLLM_GGUF_MODEL")?);
    let files = ModelFiles::from_dir(&model_path)?;
    if !files.has_quantized_tensors() {
        eprintln!("same-file quantized reference skipped for a dense fixture");
        return Ok(());
    }
    assert_pinned_q4_k_m_distribution(&model_path)?;
    let has_q2k = files.has_gguf_tensor_type(GgufTensorType::Q2K);
    let has_q3k = files.has_gguf_tensor_type(GgufTensorType::Q3K);
    let same_file_tolerance = if has_q2k {
        0.75
    } else if has_q3k {
        0.4
    } else {
        MAX_QUANTIZED_KERNEL_DIFFERENCE
    };
    // Low-bit direct matmul requantizes activations at every layer. Its
    // cumulative result can legitimately choose another near-tied token than
    // a one-off fully dequantized graph; the independent same-quant oracle and
    // fixed llama.cpp greedy gate below provide the discrete checks instead.
    let require_dense_argmax = !(has_q2k || has_q3k);
    let adapter = files.adapter()?;
    let device = Device::new_metal(0)?;

    // SAFETY: the parity gate treats the fixed checkpoint as immutable.
    let quantized_weights = unsafe { files.weight_source(DType::F32, &device)? };
    let quantized = adapter.build(quantized_weights.as_ref())?;
    let mut quantized_session = quantized.new_session()?;
    drop(quantized_weights);
    // SAFETY: this uses the same immutable file but expands matrices only for a
    // validation reference; the production path above remains compressed.
    let dense_weights = unsafe { files.dequantized_weight_source(DType::F32, &device)? };
    let dense = adapter.build(dense_weights.as_ref())?;
    let mut dense_session = dense.new_session()?;
    drop(dense_weights);

    let input = Tensor::new(&[[1_u32, 100, 200, 300]], &device)?;
    let quantized_prefill = logits(quantized.forward(quantized_session.as_mut(), &input, 0)?)?;
    let dense_prefill = logits(dense.forward(dense_session.as_mut(), &input, 0)?)?;
    let prefill_relative_rms = assert_logits_close(
        "quantized/dense same-file prefill",
        &quantized_prefill,
        &dense_prefill,
        same_file_tolerance,
    );
    if !require_dense_argmax {
        assert!(
            prefill_relative_rms <= 0.05,
            "low-bit prefill relative RMS {prefill_relative_rms} exceeds 0.05"
        );
    }
    if require_dense_argmax {
        assert_eq!(argmax(&quantized_prefill), argmax(&dense_prefill));
    }

    let next_token = argmax(&quantized_prefill) as u32;
    let next = Tensor::new(&[[next_token]], &device)?;
    let quantized_cached = logits(quantized.forward(quantized_session.as_mut(), &next, 4)?)?;
    let dense_cached = logits(dense.forward(dense_session.as_mut(), &next, 4)?)?;
    let cached_relative_rms = assert_logits_close(
        "quantized/dense same-file cached decode",
        &quantized_cached,
        &dense_cached,
        same_file_tolerance,
    );
    if !require_dense_argmax {
        assert!(
            cached_relative_rms <= 0.05,
            "low-bit cached relative RMS {cached_relative_rms} exceeds 0.05"
        );
    }
    if require_dense_argmax {
        assert_eq!(argmax(&quantized_cached), argmax(&dense_cached));
    }

    let combined = Tensor::new(&[[1_u32, 100, 200, 300, next_token]], &device)?;
    // SAFETY: construct a fresh compressed model for cache/full consistency.
    let fresh_weights = unsafe { files.weight_source(DType::F32, &device)? };
    let fresh_quantized = adapter.build(fresh_weights.as_ref())?;
    let mut fresh_quantized_session = fresh_quantized.new_session()?;
    drop(fresh_weights);
    let quantized_full =
        logits(fresh_quantized.forward(fresh_quantized_session.as_mut(), &combined, 0)?)?;
    assert_eq!(argmax(&quantized_cached), argmax(&quantized_full));
    let cache_full_relative_rms = assert_logits_close(
        "quantized cache/full",
        &quantized_cached,
        &quantized_full,
        if require_dense_argmax { 0.05 } else { 0.15 },
    );
    if !require_dense_argmax {
        assert!(
            cache_full_relative_rms <= 0.01,
            "low-bit cache/full relative RMS {cache_full_relative_rms} exceeds 0.01"
        );
    }

    drop(fresh_quantized);
    drop(quantized);
    drop(dense);
    // SAFETY: exercise the documented explicit-F16 Metal path with the same
    // immutable quantized fixture after releasing the larger F32 references.
    let f16_weights = unsafe { files.weight_source(DType::F16, &device)? };
    let f16_model = adapter.build(f16_weights.as_ref())?;
    let mut f16_session = f16_model.new_session()?;
    drop(f16_weights);
    let f16_logits = f16_model.forward(f16_session.as_mut(), &input, 0)?;
    assert_eq!(f16_logits.dims2()?, (1, adapter.runtime_spec().vocab_size));
    Ok(())
}

#[test]
#[ignore = "requires a fixed Q4_0 or Q4_K_M SmolLM2 GGUF and Apple Metal"]
fn q4_greedy_sequences_match_recorded_llama_cpp_goldens() -> anyhow::Result<()> {
    let _metal_guard = METAL_TEST_LOCK
        .lock()
        .expect("GGUF Metal test lock poisoned");
    let model_path = PathBuf::from(std::env::var("MACLLM_GGUF_MODEL")?);
    let files = ModelFiles::from_dir(&model_path)?;
    let expected_text = if files.has_gguf_tensor_type(GgufTensorType::Q4K) {
        assert_pinned_q4_k_m_distribution(&model_path)?;
        " Paris. Paris is a city that is"
    } else if files.has_gguf_tensor_type(GgufTensorType::Q4_0)
        && !files.has_gguf_tensor_type(GgufTensorType::Q6K)
    {
        " the capital of the United States. The"
    } else {
        eprintln!("Q4 llama.cpp golden skipped for a non-Q4 fixture");
        return Ok(());
    };
    let tokenizer = files.tokenizer()?;
    let prompt = tokenizer
        .encode("The capital of France is", false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let expected = tokenizer
        .encode(expected_text, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(expected.get_ids().len(), 8);

    let adapter = files.adapter()?;
    let device = Device::new_metal(0)?;
    // SAFETY: the golden gate treats the pinned checkpoint as immutable.
    let weights = unsafe { files.weight_source(DType::F32, &device)? };
    let model = adapter.build(weights.as_ref())?;
    let mut session = model.new_session()?;
    drop(weights);
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..SamplingConfig::default()
    })?;
    let mut generated = Vec::new();
    generate(
        model.as_ref(),
        session.as_mut(),
        prompt.get_ids(),
        &mut sampler,
        GenerationConfig {
            max_new_tokens: 8,
            ignore_stop_tokens: true,
        },
        |token| {
            generated.push(token);
            Ok(())
        },
    )?;
    assert_eq!(generated, expected.get_ids());
    Ok(())
}

#[test]
#[ignore = "requires a fixed quantized Llama-family GGUF and an accessible Apple Metal device"]
fn gguf_matches_independent_candle_quantized_llama() -> anyhow::Result<()> {
    let _metal_guard = METAL_TEST_LOCK
        .lock()
        .expect("GGUF Metal test lock poisoned");
    let model_path = PathBuf::from(std::env::var("MACLLM_GGUF_MODEL")?);
    let files = ModelFiles::from_dir(&model_path)?;
    if !files.has_quantized_tensors() {
        eprintln!("quantized Candle reference skipped for a dense fixture");
        return Ok(());
    }
    let requires_fresh_full_oracle =
        candle_metal_mv_requires_fresh_full_oracle(&model_path, &files)?;
    let tokenizer = files.tokenizer()?;
    let prompt = tokenizer
        .encode("The capital of France is", false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let device = Device::new_metal(0)?;

    let adapter = files.adapter()?;
    // SAFETY: the parity gate treats the pinned checkpoint as immutable.
    let weights = unsafe { files.weight_source(DType::F32, &device)? };
    let actual_model = adapter.build(weights.as_ref())?;
    let mut actual_session = actual_model.new_session()?;
    drop(weights);

    let mut reader = File::open(single_gguf_path(&model_path)?)?;
    let content = gguf_file::Content::read(&mut reader)?;
    let mut reference_model =
        quantized_llama::ModelWeights::from_gguf(content, &mut reader, &device)?;
    let input = Tensor::from_vec(
        prompt.get_ids().to_vec(),
        (1, prompt.get_ids().len()),
        &device,
    )?;
    let actual_prefill = logits(actual_model.forward(actual_session.as_mut(), &input, 0)?)?;
    let reference_prefill = logits(reference_model.forward(&input, 0)?)?;
    let prefill_relative_rms = assert_logits_close(
        "MacLLM/Candle quantized prefill",
        &actual_prefill,
        &reference_prefill,
        MAX_CANDLE_SAME_SHAPE_DIFFERENCE,
    );
    assert!(
        prefill_relative_rms <= MAX_CANDLE_SAME_SHAPE_RELATIVE_RMS,
        "MacLLM/Candle prefill relative RMS {prefill_relative_rms} exceeds {MAX_CANDLE_SAME_SHAPE_RELATIVE_RMS}"
    );
    assert_eq!(argmax(&actual_prefill), argmax(&reference_prefill));

    let next_token = argmax(&reference_prefill) as u32;
    let next = Tensor::new(&[[next_token]], &device)?;
    let actual_cached =
        logits(actual_model.forward(actual_session.as_mut(), &next, prompt.get_ids().len())?)?;
    if requires_fresh_full_oracle {
        // Candle 0.11 over-dispatches every Q2_K Metal matrix-vector kernel;
        // Q3_K has the same unsafe tail only when output rows are not divisible
        // by four. MacLLM selects compressed MM for those shapes, so use a
        // fresh full-sequence Candle MM forward as the independent oracle.
        let mut combined_ids = prompt.get_ids().to_vec();
        combined_ids.push(next_token);
        let combined = Tensor::from_vec(combined_ids.clone(), (1, combined_ids.len()), &device)?;
        let mut reader = File::open(single_gguf_path(&model_path)?)?;
        let content = gguf_file::Content::read(&mut reader)?;
        let mut fresh_reference =
            quantized_llama::ModelWeights::from_gguf(content, &mut reader, &device)?;
        let reference_full = logits(fresh_reference.forward(&combined, 0)?)?;
        let relative_rms = assert_logits_close(
            "MacLLM cached/Candle fresh-full lower-bit K-quant",
            &actual_cached,
            &reference_full,
            MAX_CANDLE_FRESH_FULL_DIFFERENCE,
        );
        assert!(
            relative_rms <= MAX_CANDLE_FRESH_FULL_RELATIVE_RMS,
            "lower-bit cached/Candle full relative RMS {relative_rms} exceeds {MAX_CANDLE_FRESH_FULL_RELATIVE_RMS}"
        );
        assert_eq!(argmax(&actual_cached), argmax(&reference_full));
    } else {
        let reference_cached = logits(reference_model.forward(&next, prompt.get_ids().len())?)?;
        let cached_relative_rms = assert_logits_close(
            "MacLLM/Candle quantized cached",
            &actual_cached,
            &reference_cached,
            MAX_CANDLE_SAME_SHAPE_DIFFERENCE,
        );
        assert!(
            cached_relative_rms <= MAX_CANDLE_SAME_SHAPE_RELATIVE_RMS,
            "MacLLM/Candle cached relative RMS {cached_relative_rms} exceeds {MAX_CANDLE_SAME_SHAPE_RELATIVE_RMS}"
        );
        assert_eq!(argmax(&actual_cached), argmax(&reference_cached));
    }
    Ok(())
}

fn candle_metal_mv_requires_fresh_full_oracle(
    model_path: &Path,
    files: &ModelFiles,
) -> anyhow::Result<bool> {
    if files.has_gguf_tensor_type(GgufTensorType::Q2K) {
        return Ok(true);
    }
    if !files.has_gguf_tensor_type(GgufTensorType::Q3K) {
        return Ok(false);
    }
    let file = GgufFile::open(single_gguf_path(model_path)?)?;
    Ok(file.tensors().values().any(|tensor| {
        tensor.dtype == GgufTensorType::Q3K
            && tensor.shape.len() == 2
            && !tensor.shape[0].is_multiple_of(4)
    }))
}

fn single_gguf_path(model_path: &Path) -> anyhow::Result<PathBuf> {
    if model_path.is_file() {
        return Ok(model_path.to_owned());
    }
    let paths = fs::read_dir(model_path)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "gguf")
        })
        .collect::<Vec<_>>();
    anyhow::ensure!(paths.len() == 1, "expected exactly one GGUF fixture");
    Ok(paths[0].clone())
}

fn assert_pinned_q4_k_m_distribution(model_path: &Path) -> anyhow::Result<()> {
    let file = GgufFile::open(single_gguf_path(model_path)?)?;
    // Several lower-bit recipes also contain Q4_K fallback tensors. The
    // historical SmolLM2 Q4_K_M fixture is distinguished by its legacy Q5_0
    // rows; do not apply that file-specific distribution to every mixed GGUF.
    if !file
        .tensors()
        .values()
        .any(|tensor| tensor.dtype == GgufTensorType::Q4K)
        || !file
            .tensors()
            .values()
            .any(|tensor| tensor.dtype == GgufTensorType::Q5_0)
    {
        return Ok(());
    }
    let expected = [
        (GgufTensorType::F32, 61),
        (GgufTensorType::Q5_0, 160),
        (GgufTensorType::Q5_1, 13),
        (GgufTensorType::Q8_0, 8),
        (GgufTensorType::Q4K, 16),
        (GgufTensorType::Q5K, 6),
        (GgufTensorType::Q6K, 8),
    ];
    assert_eq!(file.tensors().len(), 272);
    for (dtype, count) in expected {
        assert_eq!(
            file.tensors()
                .values()
                .filter(|tensor| tensor.dtype == dtype)
                .count(),
            count,
            "unexpected {dtype:?} tensor count in the pinned Q4_K_M fixture"
        );
    }
    Ok(())
}

fn prefill_and_decode(
    files: &ModelFiles,
    input: &Tensor,
    device: &Device,
    dtype: DType,
    forced_next_token: Option<u32>,
) -> anyhow::Result<(Vec<f32>, Vec<f32>, u32)> {
    let adapter = files.adapter()?;
    // SAFETY: the parity gate treats both fixed checkpoints as immutable.
    let weights = unsafe { files.weight_source(dtype, device)? };
    let model = adapter.build(weights.as_ref())?;
    let mut session = model.new_session()?;
    let prefill = logits(model.forward(session.as_mut(), input, 0)?)?;
    let next_token = forced_next_token.unwrap_or(argmax(&prefill) as u32);
    let next = Tensor::new(&[[next_token]], device)?;
    let cached = logits(model.forward(session.as_mut(), &next, 4)?)?;
    Ok((prefill, cached, next_token))
}

fn fresh_forward(
    files: &ModelFiles,
    input: &Tensor,
    device: &Device,
    dtype: DType,
) -> anyhow::Result<Vec<f32>> {
    let adapter = files.adapter()?;
    // SAFETY: the parity gate treats both fixed checkpoints as immutable.
    let weights = unsafe { files.weight_source(dtype, device)? };
    let model = adapter.build(weights.as_ref())?;
    let mut session = model.new_session()?;
    Ok(logits(model.forward(session.as_mut(), input, 0)?)?)
}

fn logits(tensor: Tensor) -> candle_core::Result<Vec<f32>> {
    Ok(tensor.to_dtype(DType::F32)?.to_vec2::<f32>()?.remove(0))
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

fn assert_logits_close(label: &str, left: &[f32], right: &[f32], tolerance: f32) -> f64 {
    assert!(!left.is_empty(), "{label} produced no logits");
    assert_eq!(left.len(), right.len(), "{label} logits length mismatch");
    let left_non_finite = left.iter().filter(|value| !value.is_finite()).count();
    let right_non_finite = right.iter().filter(|value| !value.is_finite()).count();
    assert_eq!(
        (left_non_finite, right_non_finite),
        (0, 0),
        "{label} produced non-finite logits: left={left_non_finite}, right={right_non_finite}"
    );
    let (index, difference) = left
        .iter()
        .zip(right)
        .enumerate()
        .map(|(index, (left, right))| (index, (left - right).abs()))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("logits are non-empty");
    let squared_error = left
        .iter()
        .zip(right)
        .map(|(left, right)| f64::from(left - right).powi(2))
        .sum::<f64>();
    let squared_reference = right
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>();
    let relative_rms = (squared_error / squared_reference.max(f64::EPSILON)).sqrt();
    eprintln!(
        "{label} max abs difference: {difference} at index {index}; relative RMS: {relative_rms}"
    );
    assert!(
        difference <= tolerance,
        "{label} max abs difference {difference} exceeds {tolerance} at index {index}"
    );
    relative_rms
}
