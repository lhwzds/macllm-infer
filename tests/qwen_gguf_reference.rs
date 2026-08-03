//! Real-checkpoint Qwen2.5 standalone GGUF verification.
//!
//! These gates are ignored because they require the fixed ~429 MB `Q4_0` GGUF,
//! the ~988 MB Safetensors reference for cross-format checks, and Apple Metal:
//!
//! ```text
//! MACLLM_QWEN_MODEL_DIR=models/qwen2.5-0.5b-instruct \
//! MACLLM_QWEN_GGUF_MODEL=models/qwen2.5-0.5b-instruct-q4_0 \
//! cargo test --release --test qwen_gguf_reference -- --ignored --nocapture
//! ```

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use candle_transformers::models::quantized_qwen2;
use macllm_infer::{
    generation::{GenerationConfig, generate},
    gguf::{GgufFile, GgufTensorType},
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};

const RAW_PROMPT_IDS: [u32; 5] = [785, 6722, 315, 9625, 374];
const LLAMA_CPP_GOLDEN_IDS: [u32; 8] = [12_095, 11, 323, 432, 374, 279, 6722, 315];
const FIRST_UNUSED_TOKEN_ID: u32 = 151_665;
const MAX_QUANTIZED_REFERENCE_DIFFERENCE: f32 = 0.05;
const MAX_CACHE_FULL_DIFFERENCE: f32 = 0.1;
const MAX_Q4_CROSS_FORMAT_DIFFERENCE: f32 = 6.5;

#[test]
#[ignore = "requires fixed Qwen2.5 Safetensors and Q4_0 GGUF fixtures"]
fn qwen_gguf_metadata_and_tokenizer_match_safetensors() -> anyhow::Result<()> {
    let (safetensors, gguf) = fixtures()?;
    assert_pinned_qwen_fixture(&gguf.root)?;

    let safetensors_adapter = safetensors.adapter()?;
    let gguf_adapter = gguf.adapter()?;
    let safetensors_spec = safetensors_adapter
        .decoder_spec()
        .expect("Qwen2.5 uses the shared decoder");
    let gguf_spec = gguf_adapter
        .decoder_spec()
        .expect("Qwen2.5 GGUF uses the shared decoder");
    let mut expected = safetensors_spec.clone();
    // The source checkpoint ties its output head, while the pinned quantizer
    // materialized an independent Q8_0 output.weight. Runtime storage must
    // follow the file that is actually loaded.
    expected.tied_embeddings = false;
    let norm_epsilon_difference = (gguf_spec.norm_epsilon - expected.norm_epsilon).abs();
    assert!(norm_epsilon_difference < 1e-12);
    // GGUF serializes this field as F32 while the JSON sidecar carries a
    // decimal F64, so compare the rest of the contract structurally.
    expected.norm_epsilon = gguf_spec.norm_epsilon;
    assert_eq!(gguf_spec, &expected);

    let safetensors_tokenizer = safetensors.tokenizer()?;
    let gguf_tokenizer = gguf.tokenizer()?;
    let prompt = safetensors_adapter.format_chat(None, "What is the capital of France?")?;
    assert_eq!(
        prompt,
        gguf_adapter.format_chat(None, "What is the capital of France?")?
    );
    for text in [
        "The capital of France is",
        "I've been 'TOLD he's there; 12345",
        "café 🦀 — Rust\n第二行",
        "cafe\u{301}",
        "<|im_start|>assistant\nParis<|im_end|>",
        "<tool_call>{\"name\":\"weather\"}</tool_call>",
        prompt.as_str(),
    ] {
        assert_tokenizer_parity(&safetensors_tokenizer, &gguf_tokenizer, text)?;
    }
    for skip_special in [false, true] {
        let expected = safetensors_tokenizer
            .decode(&[FIRST_UNUSED_TOKEN_ID], skip_special)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let actual = gguf_tokenizer
            .decode(&[FIRST_UNUSED_TOKEN_ID], skip_special)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(actual, expected, "UNUSED token decode parity");
        assert!(actual.is_empty(), "UNUSED token must decode to no text");
    }
    let raw = gguf_tokenizer
        .encode("The capital of France is", false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(raw.get_ids(), RAW_PROMPT_IDS);
    Ok(())
}

#[test]
#[ignore = "requires fixed Qwen2.5 Safetensors and Q4_0 GGUF fixtures plus Apple Metal"]
fn qwen_q4_gguf_matches_safetensors_quantization_quality() -> anyhow::Result<()> {
    let (safetensors, gguf) = fixtures()?;
    let device = Device::new_metal(0)?;
    let input = Tensor::new(&[RAW_PROMPT_IDS], &device)?;

    let safetensors_model = build(&safetensors, DType::F16, &device)?;
    let gguf_model = build(&gguf, DType::F32, &device)?;
    let mut safetensors_session = safetensors_model.new_session()?;
    let mut gguf_session = gguf_model.new_session()?;
    let safetensors_prefill =
        logits(safetensors_model.forward(safetensors_session.as_mut(), &input, 0)?)?;
    let gguf_prefill = logits(gguf_model.forward(gguf_session.as_mut(), &input, 0)?)?;
    assert_logits_close(
        "Qwen GGUF/Safetensors prefill",
        &gguf_prefill,
        &safetensors_prefill,
        MAX_Q4_CROSS_FORMAT_DIFFERENCE,
    );
    assert_eq!(argmax(&gguf_prefill), argmax(&safetensors_prefill));

    let next_token = argmax(&safetensors_prefill) as u32;
    let next = Tensor::new(&[[next_token]], &device)?;
    let safetensors_cached = logits(safetensors_model.forward(
        safetensors_session.as_mut(),
        &next,
        RAW_PROMPT_IDS.len(),
    )?)?;
    let gguf_cached =
        logits(gguf_model.forward(gguf_session.as_mut(), &next, RAW_PROMPT_IDS.len())?)?;
    assert_logits_close(
        "Qwen GGUF/Safetensors cached",
        &gguf_cached,
        &safetensors_cached,
        MAX_Q4_CROSS_FORMAT_DIFFERENCE,
    );
    // The second-token distribution contains a quantized near-tie (token 11
    // versus 13). The same-file Candle gate and llama.cpp sequence golden are
    // the correctness requirements; cross-precision argmax identity is not.
    Ok(())
}

#[test]
#[ignore = "requires the fixed Qwen2.5 Q4_0 GGUF fixture and Apple Metal"]
fn qwen_gguf_matches_independent_candle_quantized_qwen2() -> anyhow::Result<()> {
    let gguf = gguf_fixture()?;
    assert_pinned_qwen_fixture(&gguf.root)?;
    let device = Device::new_metal(0)?;
    let input = Tensor::new(&[RAW_PROMPT_IDS], &device)?;
    let actual = build(&gguf, DType::F32, &device)?;
    let mut actual_session = actual.new_session()?;

    let mut reader = File::open(single_gguf_path(&gguf.root)?)?;
    let content = gguf_file::Content::read(&mut reader)?;
    let mut reference = quantized_qwen2::ModelWeights::from_gguf(content, &mut reader, &device)?;

    let actual_prefill = logits(actual.forward(actual_session.as_mut(), &input, 0)?)?;
    let reference_prefill = logits(reference.forward(&input, 0)?)?;
    assert_logits_close(
        "MacLLM/Candle Qwen quantized prefill",
        &actual_prefill,
        &reference_prefill,
        MAX_QUANTIZED_REFERENCE_DIFFERENCE,
    );
    assert_eq!(argmax(&actual_prefill), argmax(&reference_prefill));

    let next_token = argmax(&reference_prefill) as u32;
    let next = Tensor::new(&[[next_token]], &device)?;
    let actual_cached =
        logits(actual.forward(actual_session.as_mut(), &next, RAW_PROMPT_IDS.len())?)?;
    let reference_cached = logits(reference.forward(&next, RAW_PROMPT_IDS.len())?)?;
    assert_logits_close(
        "MacLLM/Candle Qwen quantized cached",
        &actual_cached,
        &reference_cached,
        MAX_QUANTIZED_REFERENCE_DIFFERENCE,
    );
    assert_eq!(argmax(&actual_cached), argmax(&reference_cached));

    let mut actual_full_session = actual.new_session()?;
    reference.clear_kv_cache();
    let mut combined_ids = RAW_PROMPT_IDS.to_vec();
    combined_ids.push(next_token);
    let combined = Tensor::from_vec(combined_ids, (1, RAW_PROMPT_IDS.len() + 1), &device)?;
    let actual_full = logits(actual.forward(actual_full_session.as_mut(), &combined, 0)?)?;
    let reference_full = logits(reference.forward(&combined, 0)?)?;
    assert_logits_close(
        "MacLLM Qwen cache/full",
        &actual_cached,
        &actual_full,
        MAX_CACHE_FULL_DIFFERENCE,
    );
    assert_logits_close(
        "Candle Qwen cache/full",
        &reference_cached,
        &reference_full,
        MAX_CACHE_FULL_DIFFERENCE,
    );
    assert_logits_close(
        "MacLLM/Candle Qwen quantized full",
        &actual_full,
        &reference_full,
        MAX_QUANTIZED_REFERENCE_DIFFERENCE,
    );
    Ok(())
}

#[test]
#[ignore = "requires the fixed Qwen2.5 Q4_0 GGUF fixture and Apple Metal"]
fn qwen_q4_greedy_sequence_matches_recorded_llama_cpp_golden() -> anyhow::Result<()> {
    let gguf = gguf_fixture()?;
    let tokenizer = gguf.tokenizer()?;
    let prompt = tokenizer
        .encode("The capital of France is", false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(prompt.get_ids(), RAW_PROMPT_IDS);
    let expected = tokenizer
        .encode(" Paris, and it is the capital of", false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(expected.get_ids(), LLAMA_CPP_GOLDEN_IDS);

    let device = Device::new_metal(0)?;
    let model = build(&gguf, DType::F32, &device)?;
    let mut session = model.new_session()?;
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
            max_new_tokens: LLAMA_CPP_GOLDEN_IDS.len(),
            ignore_stop_tokens: true,
        },
        |token| {
            generated.push(token);
            Ok(())
        },
    )?;
    assert_eq!(generated, LLAMA_CPP_GOLDEN_IDS);
    Ok(())
}

fn fixtures() -> anyhow::Result<(ModelFiles, ModelFiles)> {
    let safetensors = ModelFiles::from_dir(PathBuf::from(std::env::var("MACLLM_QWEN_MODEL_DIR")?))?;
    Ok((safetensors, gguf_fixture()?))
}

fn gguf_fixture() -> anyhow::Result<ModelFiles> {
    ModelFiles::from_dir(PathBuf::from(std::env::var("MACLLM_QWEN_GGUF_MODEL")?))
}

fn build(
    files: &ModelFiles,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<Box<dyn macllm_infer::architecture::CausalLm>> {
    let adapter = files.adapter()?;
    // SAFETY: the parity gate treats each pinned checkpoint as immutable.
    let weights = unsafe { files.weight_source(dtype, device)? };
    let model = adapter.build(weights.as_ref())?;
    drop(weights);
    Ok(model)
}

fn assert_tokenizer_parity(
    expected: &tokenizers::Tokenizer,
    actual: &tokenizers::Tokenizer,
    text: &str,
) -> anyhow::Result<()> {
    let expected_encoding = expected
        .encode(text, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let actual_encoding = actual
        .encode(text, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(
        actual_encoding.get_ids(),
        expected_encoding.get_ids(),
        "token IDs for {text:?}"
    );
    assert_eq!(
        actual_encoding.get_offsets(),
        expected_encoding.get_offsets(),
        "token offsets for {text:?}"
    );
    for skip_special in [false, true] {
        assert_eq!(
            actual
                .decode(actual_encoding.get_ids(), skip_special)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
            expected
                .decode(expected_encoding.get_ids(), skip_special)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
            "decoded text for {text:?}, skip_special={skip_special}"
        );
    }
    Ok(())
}

fn assert_pinned_qwen_fixture(model_path: &Path) -> anyhow::Result<()> {
    let file = GgufFile::open(single_gguf_path(model_path)?)?;
    assert_eq!(file.metadata().len(), 26);
    assert_eq!(file.tensors().len(), 291);
    assert_eq!(file.data_offset(), 5_947_744);
    for (dtype, expected) in [
        (GgufTensorType::F32, 121),
        (GgufTensorType::Q4_0, 169),
        (GgufTensorType::Q8_0, 1),
    ] {
        let actual = file
            .tensors()
            .values()
            .filter(|tensor| tensor.dtype == dtype)
            .count();
        assert_eq!(actual, expected, "unexpected {dtype:?} tensor count");
    }
    Ok(())
}

fn single_gguf_path(model_path: &Path) -> anyhow::Result<PathBuf> {
    if model_path.is_file() {
        return Ok(model_path.to_owned());
    }
    let paths = std::fs::read_dir(model_path)?
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

fn assert_logits_close(label: &str, left: &[f32], right: &[f32], tolerance: f32) {
    assert!(!left.is_empty(), "{label} produced no logits");
    assert_eq!(left.len(), right.len(), "{label} logits length mismatch");
    assert!(left.iter().all(|value| value.is_finite()));
    assert!(right.iter().all(|value| value.is_finite()));
    let (index, difference) = left
        .iter()
        .zip(right)
        .enumerate()
        .map(|(index, (left, right))| (index, (left - right).abs()))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("logits are non-empty");
    eprintln!("{label} max abs difference: {difference} at index {index}");
    assert!(
        difference <= tolerance,
        "{label} max abs difference {difference} exceeded {tolerance} at index {index}"
    );
}
