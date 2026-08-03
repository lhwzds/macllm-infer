//! Llama-family parity gates against Candle's independent implementation.
//!
//! The synthetic F32 gate runs by default. The real Metal gate requires the
//! public `SmolLM2` checkpoint:
//! `MACLLM_LLAMA_MODEL_DIR=models/smollm2-135m-instruct cargo test --release --test llama_reference -- --ignored`

use std::{collections::HashMap, fs, path::PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as reference;
use macllm_infer::{
    generation::{GenerationConfig, GenerationSequence, generate},
    models::{ModelAdapter, llama::LlamaAdapter},
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use tokenizers::Tokenizer;

const SYNTHETIC_CONFIG: &str = r#"{
  "model_type":"llama","vocab_size":17,"hidden_size":8,
  "intermediate_size":12,"num_hidden_layers":2,
  "num_attention_heads":2,"num_key_value_heads":1,
  "max_position_embeddings":64,"tie_word_embeddings":false,
  "rope_theta":10000.0,"rms_norm_eps":0.00001,"hidden_act":"silu",
  "rope_scaling":null,"rope_interleaved":false,
  "attention_bias":false,"mlp_bias":false,"bos_token_id":1,
  "eos_token_id":[2,3]
}"#;

const SYNTHETIC_LLAMA3_CONFIG: &str = r#"{
  "model_type":"llama","vocab_size":17,"hidden_size":8,
  "intermediate_size":12,"num_hidden_layers":2,
  "num_attention_heads":2,"num_key_value_heads":1,
  "max_position_embeddings":64,"tie_word_embeddings":false,
  "rope_theta":500000.0,"rms_norm_eps":0.00001,"hidden_act":"silu",
  "rope_scaling":{"rope_type":"llama3","factor":32.0,
    "low_freq_factor":1.0,"high_freq_factor":4.0,
    "original_max_position_embeddings":8},
  "rope_interleaved":false,"attention_bias":false,"mlp_bias":false,
  "bos_token_id":1,"eos_token_id":[2,3]
}"#;

#[test]
fn synthetic_llama_matches_independent_reference() -> anyhow::Result<()> {
    assert_synthetic_llama_parity("unscaled", SYNTHETIC_CONFIG)
}

#[test]
fn synthetic_llama3_scaling_and_cache_growth_match_independent_reference() -> anyhow::Result<()> {
    assert_synthetic_llama_parity("Llama 3 scaled", SYNTHETIC_LLAMA3_CONFIG)
}

#[test]
fn one_immutable_model_keeps_interleaved_sessions_isolated() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let config_path = temp.path().join("config.json");
    fs::write(&config_path, SYNTHETIC_CONFIG)?;
    let adapter = LlamaAdapter::from_config(&config_path)?;
    let device = Device::Cpu;
    let tensors = synthetic_weights(&device)?;
    let model = adapter.build_dense(VarBuilder::from_tensors(
        tensors.clone(),
        DType::F32,
        &device,
    ))?;

    let prompt_a = [1_u32, 5, 7];
    let prompt_b = [1_u32, 9, 4, 8];
    let mut session_a = model.new_session()?;
    let mut session_b = model.new_session()?;
    let prefill_a = model.forward(session_a.as_mut(), &Tensor::new(&[prompt_a], &device)?, 0)?;
    let prefill_b = model.forward(session_b.as_mut(), &Tensor::new(&[prompt_b], &device)?, 0)?;
    assert_eq!(session_a.cache_len(), prompt_a.len());
    assert_eq!(session_b.cache_len(), prompt_b.len());

    let next_a = argmax(&logits(prefill_a)?) as u32;
    let next_b = argmax(&logits(prefill_b)?) as u32;
    let cached_a = logits(model.forward(
        session_a.as_mut(),
        &Tensor::new(&[[next_a]], &device)?,
        prompt_a.len(),
    )?)?;
    assert_eq!(session_a.cache_len(), prompt_a.len() + 1);
    assert_eq!(session_b.cache_len(), prompt_b.len());
    let cached_b = logits(model.forward(
        session_b.as_mut(),
        &Tensor::new(&[[next_b]], &device)?,
        prompt_b.len(),
    )?)?;

    let mut expected_session = model.new_session()?;
    let mut full_a = prompt_a.to_vec();
    full_a.push(next_a);
    let full_a = logits(model.forward(
        expected_session.as_mut(),
        &Tensor::from_slice(&full_a, (1, full_a.len()), &device)?,
        0,
    )?)?;
    let mut control_session = model.new_session()?;
    let mut full_b = prompt_b.to_vec();
    full_b.push(next_b);
    let full_b = logits(model.forward(
        control_session.as_mut(),
        &Tensor::from_slice(&full_b, (1, full_b.len()), &device)?,
        0,
    )?)?;
    assert_logits_close("interleaved session A", &cached_a, &full_a, 1e-4);
    assert_logits_close("interleaved session B", &cached_b, &full_b, 1e-4);

    let other = adapter.build_dense(VarBuilder::from_tensors(tensors, DType::F32, &device))?;
    let mut foreign_session = model.new_session()?;
    let error = other
        .forward(
            foreign_session.as_mut(),
            &Tensor::new(&[[1_u32]], &device)?,
            0,
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("belongs to a different model instance"));
    assert_eq!(foreign_session.cache_len(), 0);
    Ok(())
}

#[test]
fn budgeted_generation_prefill_matches_the_blocking_driver() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let config_path = temp.path().join("config.json");
    fs::write(&config_path, SYNTHETIC_CONFIG)?;
    let adapter = LlamaAdapter::from_config(&config_path)?;
    let device = Device::Cpu;
    let model = adapter.build_dense(VarBuilder::from_tensors(
        synthetic_weights(&device)?,
        DType::F32,
        &device,
    ))?;
    let prompt = vec![1_u32, 5, 7, 9, 4];
    let generation = GenerationConfig {
        max_new_tokens: 2,
        ignore_stop_tokens: true,
    };
    let sampling = SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..SamplingConfig::default()
    };

    let mut one_shot_session = model.new_session()?;
    let one_shot_logits = logits(model.forward_tokens(one_shot_session.as_mut(), &prompt, 0)?)?;
    let mut chunked_session = model.new_session()?;
    let _ = model.forward_tokens(chunked_session.as_mut(), &prompt[0..2], 0)?;
    let _ = model.forward_tokens(chunked_session.as_mut(), &prompt[2..4], 2)?;
    let chunked_logits =
        logits(model.forward_tokens(chunked_session.as_mut(), &prompt[4..5], 4)?)?;
    assert_logits_close(
        "budgeted versus one-shot prefill",
        &chunked_logits,
        &one_shot_logits,
        1e-4,
    );

    let mut sequence = GenerationSequence::new(
        model.as_ref(),
        prompt.clone(),
        Sampler::new(sampling)?,
        generation,
    )?;
    let mut stepped_tokens = Vec::new();
    let mut scheduled_inputs = Vec::new();
    loop {
        let step = sequence.step_with_budget(
            model.as_ref(),
            2,
            || false,
            |token| {
                stepped_tokens.push(token);
                Ok(())
            },
        )?;
        scheduled_inputs.push(step.input_tokens());
        if step.is_finished() {
            break;
        }
    }
    assert_eq!(scheduled_inputs, [2, 2, 1, 1]);

    let mut blocking_session = model.new_session()?;
    let mut blocking_tokens = Vec::new();
    let blocking = generate(
        model.as_ref(),
        blocking_session.as_mut(),
        &prompt,
        &mut Sampler::new(sampling)?,
        generation,
        |token| {
            blocking_tokens.push(token);
            Ok(())
        },
    )?;
    assert_eq!(stepped_tokens, blocking_tokens);
    assert_eq!(stepped_tokens.len(), 2);
    let stepped = sequence
        .metrics()
        .expect("budgeted sequence must finish after two visible tokens");
    assert_eq!(stepped.prompt_tokens, blocking.prompt_tokens);
    assert_eq!(stepped.generated_tokens, blocking.generated_tokens);
    assert_eq!(stepped.decode_steps, blocking.decode_steps);
    assert_eq!(stepped.finish_reason, blocking.finish_reason);
    assert_eq!(stepped.decode_steps, 1);
    assert_eq!(sequence.cache_len(), blocking_session.cache_len());
    assert_eq!(sequence.cache_len(), prompt.len() + 1);
    Ok(())
}

fn assert_synthetic_llama_parity(label: &str, config_json: &str) -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let config_path = temp.path().join("config.json");
    fs::write(&config_path, config_json)?;
    let adapter = LlamaAdapter::from_config(&config_path)?;
    let raw_config: reference::LlamaConfig = serde_json::from_str(config_json)?;
    let config = raw_config.into_config(false);
    let device = Device::Cpu;
    let tensors = synthetic_weights(&device)?;
    let prompt_tokens = [1_u32, 5, 7, 9, 4, 8, 6, 3, 10, 11, 12, 13, 14, 15, 16, 2];
    let input = Tensor::from_vec(prompt_tokens.to_vec(), (1, prompt_tokens.len()), &device)?;

    let reference_model = reference::Llama::load(
        VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        &config,
    )?;
    let mut reference_cache = reference::Cache::new(true, DType::F32, &config, &device)?;
    let reference_prefill = logits(reference_model.forward(&input, 0, &mut reference_cache)?)?;

    let macllm = adapter.build_dense(VarBuilder::from_tensors(
        tensors.clone(),
        DType::F32,
        &device,
    ))?;
    let mut macllm_session = macllm.new_session()?;
    let macllm_prefill = logits(macllm.forward(macllm_session.as_mut(), &input, 0)?)?;
    assert_logits_close(
        &format!("{label} prefill"),
        &macllm_prefill,
        &reference_prefill,
        1e-4,
    );

    let next_token = argmax(&reference_prefill) as u32;
    let next = Tensor::new(&[[next_token]], &device)?;
    let reference_cached =
        logits(reference_model.forward(&next, prompt_tokens.len(), &mut reference_cache)?)?;
    let macllm_cached =
        logits(macllm.forward(macllm_session.as_mut(), &next, prompt_tokens.len())?)?;
    assert_logits_close(
        &format!("{label} cached decode"),
        &macllm_cached,
        &reference_cached,
        1e-4,
    );

    let mut combined_tokens = prompt_tokens.to_vec();
    combined_tokens.push(next_token);
    let combined = Tensor::from_vec(combined_tokens.clone(), (1, combined_tokens.len()), &device)?;
    let fresh_reference = reference::Llama::load(
        VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
        &config,
    )?;
    let mut fresh_reference_cache = reference::Cache::new(false, DType::F32, &config, &device)?;
    let reference_full =
        logits(fresh_reference.forward(&combined, 0, &mut fresh_reference_cache)?)?;
    let fresh_macllm =
        adapter.build_dense(VarBuilder::from_tensors(tensors, DType::F32, &device))?;
    let mut fresh_macllm_session = fresh_macllm.new_session()?;
    let macllm_full = logits(fresh_macllm.forward(fresh_macllm_session.as_mut(), &combined, 0)?)?;
    assert_logits_close(
        &format!("{label} full"),
        &macllm_full,
        &reference_full,
        1e-4,
    );
    assert_logits_close(
        &format!("{label} cache/full"),
        &macllm_cached,
        &macllm_full,
        1e-4,
    );
    Ok(())
}

#[test]
#[ignore = "requires MACLLM_LLAMA_MODEL_DIR and an accessible Apple Metal device"]
fn real_smollm2_matches_independent_llama() -> anyhow::Result<()> {
    let root = PathBuf::from(std::env::var("MACLLM_LLAMA_MODEL_DIR")?);
    let files = ModelFiles::from_dir(&root)?;
    let raw_config: reference::LlamaConfig =
        serde_json::from_slice(&fs::read(files.config_path()?)?)?;
    let config = raw_config.into_config(false);
    let adapter = LlamaAdapter::from_config(files.config_path()?)?;
    let tokenizer = Tokenizer::from_file(files.tokenizer_path()?)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let rendered = adapter.format_chat(None, "Hello")?;
    let chat_ids = tokenizer
        .encode(rendered, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(
        chat_ids.get_ids(),
        &[
            1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411,
            407, 19712, 8182, 2, 198, 1, 4093, 198, 19556, 2, 198, 1, 520, 9531, 198,
        ]
    );
    let device = Device::new_metal(0)?;
    let input = Tensor::new(&[[1_u32, 100, 200, 300]], &device)?;

    let (reference_prefill, reference_cached, next_token) = {
        // SAFETY: the test treats checkpoint files as immutable.
        let weights = unsafe {
            VarBuilder::from_mmaped_safetensors(files.safetensor_weights()?, DType::F16, &device)?
        };
        let model = reference::Llama::load(weights, &config)?;
        let mut cache = reference::Cache::new(true, DType::F16, &config, &device)?;
        let prefill = logits(model.forward(&input, 0, &mut cache)?)?;
        let next_token = argmax(&prefill) as u32;
        let next = Tensor::new(&[[next_token]], &device)?;
        let cached = logits(model.forward(&next, 4, &mut cache)?)?;
        (prefill, cached, next_token)
    };

    let (macllm_prefill, macllm_cached) = {
        // SAFETY: the test treats checkpoint files as immutable.
        let weights = unsafe { files.weight_source(DType::F16, &device)? };
        let model = adapter.build(weights.as_ref())?;
        let mut session = model.new_session()?;
        let prefill = logits(model.forward(session.as_mut(), &input, 0)?)?;
        let next = Tensor::new(&[[next_token]], &device)?;
        let cached = logits(model.forward(session.as_mut(), &next, 4)?)?;
        (prefill, cached)
    };

    assert_eq!(argmax(&macllm_prefill), argmax(&reference_prefill));
    assert_logits_close(
        "real SmolLM2 prefill",
        &macllm_prefill,
        &reference_prefill,
        0.1,
    );
    assert_eq!(argmax(&macllm_cached), argmax(&reference_cached));
    assert_logits_close(
        "real SmolLM2 cached decode",
        &macllm_cached,
        &reference_cached,
        0.1,
    );

    let combined = Tensor::new(&[[1_u32, 100, 200, 300, next_token]], &device)?;
    let reference_full = {
        // SAFETY: the test treats checkpoint files as immutable.
        let weights = unsafe {
            VarBuilder::from_mmaped_safetensors(files.safetensor_weights()?, DType::F16, &device)?
        };
        let model = reference::Llama::load(weights, &config)?;
        let mut cache = reference::Cache::new(false, DType::F16, &config, &device)?;
        logits(model.forward(&combined, 0, &mut cache)?)?
    };
    let macllm_full = {
        // SAFETY: the test treats checkpoint files as immutable.
        let weights = unsafe { files.weight_source(DType::F16, &device)? };
        let model = adapter.build(weights.as_ref())?;
        let mut session = model.new_session()?;
        logits(model.forward(session.as_mut(), &combined, 0)?)?
    };
    assert_logits_close("real SmolLM2 full", &macllm_full, &reference_full, 0.1);
    assert_logits_close("real SmolLM2 cache/full", &macllm_cached, &macllm_full, 0.1);
    Ok(())
}

fn synthetic_weights(device: &Device) -> candle_core::Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "model.embed_tokens.weight".into(),
        patterned_matrix(17, 8, 1, 0.01, device)?,
    );
    tensors.insert("model.norm.weight".into(), norm_vector(8, 1, device)?);
    tensors.insert(
        "lm_head.weight".into(),
        patterned_matrix(17, 8, 2, 0.01, device)?,
    );
    for layer in 0..2 {
        tensors.insert(
            format!("model.layers.{layer}.input_layernorm.weight"),
            norm_vector(8, 10 + layer, device)?,
        );
        tensors.insert(
            format!("model.layers.{layer}.post_attention_layernorm.weight"),
            norm_vector(8, 20 + layer, device)?,
        );
        for (name, rows, cols, seed) in [
            ("self_attn.q_proj.weight", 8, 8, 30),
            ("self_attn.k_proj.weight", 4, 8, 31),
            ("self_attn.v_proj.weight", 4, 8, 32),
            ("self_attn.o_proj.weight", 8, 8, 33),
            ("mlp.gate_proj.weight", 12, 8, 34),
            ("mlp.up_proj.weight", 12, 8, 35),
            ("mlp.down_proj.weight", 8, 12, 36),
        ] {
            tensors.insert(
                format!("model.layers.{layer}.{name}"),
                patterned_matrix(rows, cols, seed + layer * 17, 0.005, device)?,
            );
        }
    }
    Ok(tensors)
}

fn patterned_matrix(
    rows: usize,
    columns: usize,
    seed: usize,
    scale: f32,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let values = (0..rows * columns)
        .map(|index| {
            let centered = ((index * 37 + seed * 19) % 101) as f32 - 50.0;
            centered * scale / 50.0
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (rows, columns), device)
}

fn norm_vector(length: usize, seed: usize, device: &Device) -> candle_core::Result<Tensor> {
    let values = (0..length)
        .map(|index| 1.0 + (((index + seed) % 7) as f32 - 3.0) * 0.01)
        .collect::<Vec<_>>();
    Tensor::from_vec(values, length, device)
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
    assert!(!left.is_empty(), "{label} produced empty logits");
    assert_eq!(left.len(), right.len(), "{label} logits length mismatch");
    assert!(
        left.iter().all(|value| value.is_finite()),
        "{label} left logits contain NaN or infinity"
    );
    assert!(
        right.iter().all(|value| value.is_finite()),
        "{label} right logits contain NaN or infinity"
    );
    let (max_index, difference) = left
        .iter()
        .zip(right)
        .enumerate()
        .map(|(index, (left, right))| (index, (left - right).abs()))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("logits are non-empty");
    eprintln!("{label} max abs difference: {difference} at index {max_index}");
    assert!(
        difference <= tolerance,
        "{label} diverged at index {max_index}: max abs difference {difference} exceeds {tolerance}"
    );
}
