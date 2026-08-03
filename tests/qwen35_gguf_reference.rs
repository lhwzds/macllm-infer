//! Real Qwen3.5 hybrid-runtime gates against a pinned official GGUF and
//! llama.cpp greedy golden.
//!
//! Fixture: `ggml-org/Qwen3.5-0.8B-GGUF` revision
//! `8fea620810c4afa23dd6443f999a48574c1611a3`, official `Q4_0`.
//!
//! Run with:
//! `MACLLM_QWEN35_GGUF_MODEL=models/qwen3.5-0.8b-q4_0 \
//!  cargo test --release --test qwen35_gguf_reference -- --ignored --nocapture`

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, Device, Tensor};
use macllm_infer::{
    architecture::{CausalLm, CausalLmBatchCapability, CausalLmBatchInput, SequenceMixerSpec},
    generation::{GenerationConfig, generate},
    gguf::GgufTensorType,
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use sha2::{Digest, Sha256};

const FILENAME: &str = "Qwen3.5-0.8B-Q4_0.gguf";
const FIXTURE_SIZE: u64 = 563_036_064;
const FIXTURE_SHA256: &str = "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf";
const RAW_PROMPT: &str = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const RAW_PROMPT_IDS: &[u32] = &[
    248_045, 846, 198, 3_710, 369, 279, 6_511, 314, 9_338, 30, 248_046, 198, 248_045, 74_455, 198,
    248_068, 271, 248_069, 271,
];
const COMBINING_MARK_TEXT: &str = "a\u{301}\u{327} क\u{93c} العربية\u{650}";
const COMBINING_MARK_IDS: &[u32] = &[64, 52_033, 136, 100, 46_215, 150_502, 171_405, 50_930];
// Captured from the pinned file with llama.cpp b10050 (b15ca938a) on both CPU
// and Metal using greedy sampling, repeat penalty 1, seed 1 and context 256.
const LLAMA_CPP_GREEDY_IDS: &[u32] = &[
    760, 6_511, 314, 9_338, 369, 2_972, 57_590, 159_034, 271, 2_064, 369, 279, 7_526, 3_177, 303,
    4_357, 321, 279, 10_407, 314, 279, 8_323, 2_936, 11, 3_055, 440, 279, 6_511, 314, 279, 5_364,
    314,
];

const MAX_CPU_ABSOLUTE_DIFFERENCE: f32 = 0.3;
const MAX_CPU_RELATIVE_RMS: f64 = 0.02;
const MAX_METAL_ABSOLUTE_DIFFERENCE: f32 = 0.05;
const MAX_METAL_RELATIVE_RMS: f64 = 0.005;

static PINNED_FIXTURE_CHECK: OnceLock<(PathBuf, Result<(), String>)> = OnceLock::new();
static INFERENCE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires the pinned official Qwen3.5-0.8B Q4_0 GGUF"]
fn qwen35_metadata_tokenizer_and_hybrid_schedule_are_pinned() -> anyhow::Result<()> {
    let files = fixture()?;
    assert_eq!(files.weight_file_count(), 1);
    assert_eq!(files.tensor_count(), Some(320));
    assert_eq!(
        files.gguf_tensor_type_counts(),
        [
            (GgufTensorType::F32, 133),
            (GgufTensorType::Q4_0, 186),
            (GgufTensorType::Q8_0, 1),
        ]
    );

    let adapter = files.adapter()?;
    let spec = adapter
        .decoder_spec()
        .expect("Qwen3.5 uses the shared decoder");
    assert_eq!(adapter.family(), "qwen35");
    assert_eq!(spec.runtime.vocab_size, 248_320);
    assert_eq!(spec.runtime.max_context, 262_144);
    assert_eq!(spec.runtime.stop_token_ids, [248_046, 248_044]);
    assert_eq!(spec.hidden_size, 1_024);
    assert_eq!(spec.intermediate_size, 3_584);
    assert_eq!(spec.num_layers, 24);
    assert_eq!(spec.num_query_heads, 8);
    assert_eq!(spec.num_kv_heads, 2);
    assert_eq!(spec.head_dim, 256);
    assert_eq!(spec.kv_bytes_per_token(2)?, 12_288);
    assert!(spec.tied_embeddings);

    let full_attention_layers = spec
        .layer_specs
        .iter()
        .enumerate()
        .filter_map(|(index, layer)| layer.attention().map(|_| index))
        .collect::<Vec<_>>();
    assert_eq!(full_attention_layers, [3, 7, 11, 15, 19, 23]);
    for (index, layer) in spec.layer_specs.iter().enumerate() {
        match &layer.mixer {
            SequenceMixerSpec::Attention(attention) => {
                assert!(full_attention_layers.contains(&index));
                assert!(attention.qk_norm);
                assert!(attention.output_gate);
                let rope = attention.rope().expect("Qwen3.5 attention uses RoPE");
                assert_eq!(rope.rotary_dim, 64);
                assert!((rope.theta - 10_000_000.0).abs() < f64::EPSILON);
            }
            SequenceMixerSpec::GatedDeltaNet(delta) => {
                assert!(!full_attention_layers.contains(&index));
                assert_eq!(delta.convolution_kernel_size, 4);
                assert_eq!(delta.num_key_heads, 16);
                assert_eq!(delta.num_value_heads, 16);
                assert_eq!(delta.key_head_dim, 128);
                assert_eq!(delta.value_head_dim, 128);
                assert!((delta.qk_norm_epsilon - 1e-6).abs() < 1e-12);
            }
            SequenceMixerSpec::ShortConvolution(_) => {
                panic!("Qwen3.5 must not lower a Gated DeltaNet layer to short convolution");
            }
            SequenceMixerSpec::ParallelAttentionMamba2(_) => {
                panic!("Qwen3.5 must not lower a layer to parallel attention plus Mamba2");
            }
        }
    }

    let tokenizer = files.tokenizer()?;
    let encoded = tokenizer
        .encode(RAW_PROMPT, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(encoded.get_ids(), RAW_PROMPT_IDS);
    let combining_marks = tokenizer
        .encode(COMBINING_MARK_TEXT, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(combining_marks.get_ids(), COMBINING_MARK_IDS);
    assert_eq!(
        adapter.format_chat(None, "What is the capital of France?")?,
        RAW_PROMPT
    );
    Ok(())
}

#[test]
#[ignore = "requires the pinned official Qwen3.5-0.8B Q4_0 GGUF"]
fn qwen35_cpu_matches_llama_cpp_and_state_semantics() -> anyhow::Result<()> {
    assert_backend(Device::Cpu, "CPU")
}

#[test]
#[ignore = "requires the pinned Qwen3.5 GGUF and Apple Metal"]
fn qwen35_metal_matches_llama_cpp_and_state_semantics() -> anyhow::Result<()> {
    assert_backend(Device::new_metal(0)?, "Metal")
}

fn assert_backend(device: Device, label: &str) -> anyhow::Result<()> {
    let _guard = INFERENCE_TEST_LOCK
        .lock()
        .expect("Qwen3.5 inference test lock poisoned");
    let files = fixture()?;
    let model = build(&files, &device)?;
    assert_greedy_golden(model.as_ref(), label)?;
    assert_chunk_cache_clear_semantics(model.as_ref(), label)?;
    assert_packed_ragged_semantics(model.as_ref(), label)?;
    Ok(())
}

fn assert_greedy_golden(model: &dyn CausalLm, label: &str) -> anyhow::Result<()> {
    let mut session = model.new_session()?;
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: 1,
        ..SamplingConfig::default()
    })?;
    let mut generated = Vec::new();
    generate(
        model,
        session.as_mut(),
        RAW_PROMPT_IDS,
        &mut sampler,
        GenerationConfig {
            max_new_tokens: LLAMA_CPP_GREEDY_IDS.len(),
            ignore_stop_tokens: true,
        },
        |token| {
            generated.push(token);
            Ok(())
        },
    )?;
    assert_eq!(generated, LLAMA_CPP_GREEDY_IDS, "{label} llama.cpp golden");
    Ok(())
}

fn assert_chunk_cache_clear_semantics(model: &dyn CausalLm, label: &str) -> anyhow::Result<()> {
    let split = 7;
    let mut one_shot = model.new_session()?;
    let one_shot_logits = model.forward_tokens(one_shot.as_mut(), RAW_PROMPT_IDS, 0)?;
    let one_shot_logits = model.read_output_to_cpu(&one_shot_logits)?;
    assert_eq!(argmax(&one_shot_logits)?, LLAMA_CPP_GREEDY_IDS[0]);

    let mut chunked = model.new_session()?;
    let _ = model.forward_tokens(chunked.as_mut(), &RAW_PROMPT_IDS[..split], 0)?;
    let chunked_logits = model.forward_tokens(chunked.as_mut(), &RAW_PROMPT_IDS[split..], split)?;
    let chunked_logits = model.read_output_to_cpu(&chunked_logits)?;
    assert_logits_close(label, "chunked/one-shot", &chunked_logits, &one_shot_logits)?;
    assert_eq!(chunked.cache_len(), RAW_PROMPT_IDS.len());

    let cached_logits = model.forward_tokens(
        chunked.as_mut(),
        &[LLAMA_CPP_GREEDY_IDS[0]],
        RAW_PROMPT_IDS.len(),
    )?;
    let cached_logits = model.read_output_to_cpu(&cached_logits)?;
    let mut combined = RAW_PROMPT_IDS.to_vec();
    combined.push(LLAMA_CPP_GREEDY_IDS[0]);
    let mut fresh = model.new_session()?;
    let full_logits = model.forward_tokens(fresh.as_mut(), &combined, 0)?;
    let full_logits = model.read_output_to_cpu(&full_logits)?;
    assert_logits_close(label, "cached/fresh", &cached_logits, &full_logits)?;
    assert_eq!(argmax(&cached_logits)?, LLAMA_CPP_GREEDY_IDS[1]);

    chunked.clear_cache();
    assert_eq!(chunked.cache_len(), 0);
    let restarted = model.forward_tokens(chunked.as_mut(), RAW_PROMPT_IDS, 0)?;
    let restarted = model.read_output_to_cpu(&restarted)?;
    assert_logits_close(label, "clear/fresh", &restarted, &one_shot_logits)?;
    Ok(())
}

fn assert_packed_ragged_semantics(model: &dyn CausalLm, label: &str) -> anyhow::Result<()> {
    assert_eq!(
        model.mixed_batch_capability(),
        CausalLmBatchCapability::PackedRagged
    );
    let first = &RAW_PROMPT_IDS[..7];
    let second = &RAW_PROMPT_IDS[7..12];
    let mut mixed_first = model.new_session()?;
    let mut mixed_second = model.new_session()?;
    let mut serial_first = model.new_session()?;
    let mut serial_second = model.new_session()?;
    let mixed = {
        let mut inputs = [
            CausalLmBatchInput::new(mixed_first.as_mut(), first, 0),
            CausalLmBatchInput::new(mixed_second.as_mut(), second, 0),
        ];
        model.forward_batch_tokens(&mut inputs)?
    };
    let mixed = model.read_output_to_cpu(&mixed)?;
    let serial_first_logits = model.forward_tokens(serial_first.as_mut(), first, 0)?;
    let serial_first_logits = model.read_output_to_cpu(&serial_first_logits)?;
    let serial_second_logits = model.forward_tokens(serial_second.as_mut(), second, 0)?;
    let serial_second_logits = model.read_output_to_cpu(&serial_second_logits)?;
    assert_row_close(label, 0, &mixed, &serial_first_logits)?;
    assert_row_close(label, 1, &mixed, &serial_second_logits)?;

    let first_suffix = &RAW_PROMPT_IDS[12..15];
    let second_suffix = &RAW_PROMPT_IDS[15..];
    let mixed = {
        let mut inputs = [
            CausalLmBatchInput::new(mixed_first.as_mut(), first_suffix, first.len()),
            CausalLmBatchInput::new(mixed_second.as_mut(), second_suffix, second.len()),
        ];
        model.forward_batch_tokens(&mut inputs)?
    };
    let mixed = model.read_output_to_cpu(&mixed)?;
    let serial_first_logits =
        model.forward_tokens(serial_first.as_mut(), first_suffix, first.len())?;
    let serial_first_logits = model.read_output_to_cpu(&serial_first_logits)?;
    let serial_second_logits =
        model.forward_tokens(serial_second.as_mut(), second_suffix, second.len())?;
    let serial_second_logits = model.read_output_to_cpu(&serial_second_logits)?;
    assert_row_close(label, 0, &mixed, &serial_first_logits)?;
    assert_row_close(label, 1, &mixed, &serial_second_logits)?;
    assert_eq!(mixed_first.cache_len(), serial_first.cache_len());
    assert_eq!(mixed_second.cache_len(), serial_second.cache_len());
    Ok(())
}

fn assert_row_close(
    label: &str,
    row: usize,
    mixed: &Tensor,
    expected: &Tensor,
) -> anyhow::Result<()> {
    let actual = mixed.narrow(0, row, 1)?;
    assert_logits_close(label, &format!("packed row {row}"), &actual, expected)
}

fn assert_logits_close(
    label: &str,
    comparison: &str,
    actual: &Tensor,
    expected: &Tensor,
) -> anyhow::Result<()> {
    anyhow::ensure!(actual.dims() == expected.dims());
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    let mut maximum = 0.0_f32;
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(&expected) {
        anyhow::ensure!(actual.is_finite() && expected.is_finite());
        let difference = actual - expected;
        maximum = maximum.max(difference.abs());
        squared_error += f64::from(difference).powi(2);
        squared_reference += f64::from(expected).powi(2);
    }
    let relative_rms = (squared_error / squared_reference.max(f64::EPSILON)).sqrt();
    let (maximum_limit, relative_rms_limit) = if label == "CPU" {
        (MAX_CPU_ABSOLUTE_DIFFERENCE, MAX_CPU_RELATIVE_RMS)
    } else {
        (MAX_METAL_ABSOLUTE_DIFFERENCE, MAX_METAL_RELATIVE_RMS)
    };
    eprintln!("Qwen3.5 {label} {comparison}: max abs {maximum}, relative RMS {relative_rms}");
    anyhow::ensure!(
        maximum <= maximum_limit,
        "{label} {comparison} maximum difference {maximum} exceeded {maximum_limit}"
    );
    anyhow::ensure!(
        relative_rms <= relative_rms_limit,
        "{label} {comparison} relative RMS {relative_rms} exceeded {relative_rms_limit}"
    );
    anyhow::ensure!(argmax_values(&actual) == argmax_values(&expected));
    Ok(())
}

fn argmax(tensor: &Tensor) -> anyhow::Result<u32> {
    Ok(argmax_values(&tensor.flatten_all()?.to_vec1::<f32>()?) as u32)
}

fn argmax_values(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}

fn fixture() -> anyhow::Result<ModelFiles> {
    let configured = PathBuf::from(std::env::var("MACLLM_QWEN35_GGUF_MODEL")?);
    let path = single_gguf_path(&configured)?;
    let (checked_path, result) = PINNED_FIXTURE_CHECK.get_or_init(|| {
        (
            path.clone(),
            validate_pinned_fixture(&path).map_err(|error| error.to_string()),
        )
    });
    anyhow::ensure!(
        checked_path == &path,
        "Qwen3.5 fixture changed within one test process"
    );
    if let Err(error) = result {
        anyhow::bail!(error.clone());
    }
    ModelFiles::from_dir(path)
}

fn single_gguf_path(configured: &Path) -> anyhow::Result<PathBuf> {
    if configured.is_file() {
        return Ok(configured.to_owned());
    }
    let paths = fs::read_dir(configured)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "gguf")
        })
        .collect::<Vec<_>>();
    anyhow::ensure!(
        paths.len() == 1,
        "expected exactly one Qwen3.5 GGUF fixture"
    );
    Ok(paths[0].clone())
}

fn validate_pinned_fixture(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(FILENAME),
        "unexpected Qwen3.5 fixture filename"
    );
    anyhow::ensure!(
        fs::metadata(path)?.len() == FIXTURE_SIZE,
        "unexpected Qwen3.5 fixture size"
    );
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    anyhow::ensure!(
        format!("{:x}", hasher.finalize()) == FIXTURE_SHA256,
        "unexpected Qwen3.5 fixture SHA-256"
    );
    Ok(())
}

fn build(files: &ModelFiles, device: &Device) -> anyhow::Result<Box<dyn CausalLm>> {
    let adapter = files.adapter()?;
    // SAFETY: the parity gate validates and then treats the pinned checkpoint
    // as immutable for the lifetime of the mmap-backed model.
    let weights = unsafe { files.weight_source(DType::F32, device)? };
    let model = adapter.build(weights.as_ref())?;
    drop(weights);
    Ok(model)
}
