//! Real-checkpoint parity gate against Candle's independent Qwen2 implementation.
//!
//! Kept ignored because it requires the ~1 GB checkpoint and an accessible GPU:
//! `MACLLM_MODEL_DIR=models/qwen2.5-0.5b-instruct cargo test --test qwen_reference -- --ignored`

use std::{fs, path::PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2 as reference;
use macllm_infer::{
    models::{ModelAdapter, qwen2::Qwen2Adapter},
    token_stream::TokenStream,
    weights::ModelFiles,
};
use tokenizers::Tokenizer;

const MAX_ABS_DIFFERENCE: f32 = 0.1;

#[test]
#[ignore = "requires MACLLM_MODEL_DIR and an accessible Apple Metal device"]
fn next_token_matches_independent_qwen_implementation() -> anyhow::Result<()> {
    let root = PathBuf::from(std::env::var("MACLLM_MODEL_DIR")?);
    let files = ModelFiles::from_dir(&root)?;
    let config: reference::Config = serde_json::from_slice(&fs::read(files.config_path()?)?)?;
    let adapter = Qwen2Adapter::from_config(files.config_path()?)?;
    let device = Device::new_metal(0)?;
    let input = Tensor::new(&[[1_u32, 2, 3, 4]], &device)?;

    let reference_logits = {
        // SAFETY: the test treats checkpoint files as immutable.
        let weights = unsafe {
            VarBuilder::from_mmaped_safetensors(files.safetensor_weights()?, DType::F16, &device)?
        };
        let mut model = reference::ModelForCausalLM::new(&config, weights)?;
        model
            .forward(&input, 0)?
            .squeeze(1)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?
            .remove(0)
    };

    let macllm_logits = {
        // SAFETY: the test treats checkpoint files as immutable.
        let weights = unsafe { files.weight_source(DType::F16, &device)? };
        let model = adapter.build(weights.as_ref())?;
        let mut session = model.new_session()?;
        model
            .forward(session.as_mut(), &input, 0)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?
            .remove(0)
    };

    let reference_token = argmax(&reference_logits);
    let macllm_token = argmax(&macllm_logits);
    assert_eq!(macllm_token, reference_token);

    let max_absolute_difference = max_abs_difference(&reference_logits, &macllm_logits);
    eprintln!("prefill max abs difference: {max_absolute_difference}");
    assert!(
        max_absolute_difference <= MAX_ABS_DIFFERENCE,
        "last-token logits diverged: max abs diff={max_absolute_difference}"
    );

    // Cache semantics gate: prefill + one-token decode must match a fresh full
    // forward over the same combined sequence.
    let cached_logits = {
        // SAFETY: the test treats checkpoint files as immutable.
        let weights = unsafe { files.weight_source(DType::F16, &device)? };
        let model = adapter.build(weights.as_ref())?;
        let mut session = model.new_session()?;
        let _ = model.forward(session.as_mut(), &input, 0)?;
        let next = Tensor::new(&[[reference_token as u32]], &device)?;
        model
            .forward(session.as_mut(), &next, 4)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?
            .remove(0)
    };
    let full_logits = {
        // SAFETY: the test treats checkpoint files as immutable.
        let weights = unsafe { files.weight_source(DType::F16, &device)? };
        let model = adapter.build(weights.as_ref())?;
        let mut session = model.new_session()?;
        let combined = Tensor::new(&[[1_u32, 2, 3, 4, reference_token as u32]], &device)?;
        model
            .forward(session.as_mut(), &combined, 0)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?
            .remove(0)
    };
    let reference_full_logits = {
        // SAFETY: the test treats checkpoint files as immutable.
        let weights = unsafe {
            VarBuilder::from_mmaped_safetensors(files.safetensor_weights()?, DType::F16, &device)?
        };
        let mut model = reference::ModelForCausalLM::new(&config, weights)?;
        let combined = Tensor::new(&[[1_u32, 2, 3, 4, reference_token as u32]], &device)?;
        model
            .forward(&combined, 0)?
            .squeeze(1)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?
            .remove(0)
    };
    assert_eq!(argmax(&cached_logits), argmax(&full_logits));
    let cache_difference = max_abs_difference(&cached_logits, &full_logits);
    eprintln!("cached vs MacLLM full max abs difference: {cache_difference}");
    assert!(
        cache_difference <= MAX_ABS_DIFFERENCE,
        "cached decode differs from full forward: max abs diff={cache_difference}"
    );
    assert_eq!(argmax(&cached_logits), argmax(&reference_full_logits));
    let reference_cache_difference = max_abs_difference(&cached_logits, &reference_full_logits);
    eprintln!("cached vs independent full max abs difference: {reference_cache_difference}");
    assert!(
        reference_cache_difference <= MAX_ABS_DIFFERENCE,
        "cached decode differs from independent full forward: max abs diff={reference_cache_difference}"
    );
    Ok(())
}

#[test]
#[ignore = "requires the official Qwen tokenizer via MACLLM_MODEL_DIR"]
fn byte_fallback_waits_for_complete_utf8() -> anyhow::Result<()> {
    let root = PathBuf::from(std::env::var("MACLLM_MODEL_DIR")?);
    let files = ModelFiles::from_dir(&root)?;
    let tokenizer = Tokenizer::from_file(files.tokenizer_path()?)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let mut stream = TokenStream::new(&tokenizer);

    // In this tokenizer, these two byte-fallback tokens form `¡`. Emitting the
    // first replacement character would permanently corrupt streamed output.
    assert_eq!(stream.push(126)?, None);
    assert_eq!(stream.push(94)?.as_deref(), Some("¡"));
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
    assert!(!left.is_empty(), "logits cannot be empty");
    assert_eq!(left.len(), right.len(), "logits length mismatch");
    assert!(
        left.iter().all(|value| value.is_finite()),
        "left logits contain NaN or infinity"
    );
    assert!(
        right.iter().all(|value| value.is_finite()),
        "right logits contain NaN or infinity"
    );
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}
