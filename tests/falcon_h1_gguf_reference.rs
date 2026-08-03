//! Real Falcon-H1 hybrid-runtime gates against a pinned official GGUF and
//! llama.cpp greedy golden.
//!
//! Fixture: `tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF` revision
//! `578d134817c19bc48991ccff9a14709d664261cc`, official `Q8_0`.
//!
//! Run with:
//! `MACLLM_FALCON_H1_GGUF_MODEL=models/Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf \
//!  cargo test --release --test falcon_h1_gguf_reference -- --ignored --nocapture`

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, Device, Tensor};
use macllm_infer::{
    architecture::{
        AttentionKind, CausalLm, CausalLmBatchCapability, CausalLmBatchInput, SequenceMixerSpec,
    },
    generation::{GenerationConfig, generate},
    gguf::GgufTensorType,
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use sha2::{Digest, Sha256};

const FILENAME: &str = "Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf";
const FIXTURE_SIZE: u64 = 98_353_792;
const FIXTURE_SHA256: &str = "e60034381f4a0e448a5e6fdab29d7b40768a587c9a56e4cf85285455ba65d476";
const RAW_PROMPT: &str = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
const RAW_PROMPT_IDS: &[u32] = &[17, 227, 2_101, 709, 12_698, 228, 709, 227, 963, 10_259, 709];
// Captured from the pinned file with llama.cpp b10050 (b15ca938a) on CPU,
// using greedy sampling, repeat penalty 1, seed 1 and the raw prompt above.
const LLAMA_CPP_GREEDY_IDS: &[u32] = &[12_698, 524, 2_530, 1_097, 855, 15_427, 1_138, 4_765];
const LLAMA_CPP_GREEDY_TEXT: &str = "Hello! How can I assist you today";

const MAX_CPU_ABSOLUTE_DIFFERENCE: f32 = 0.3;
const MAX_CPU_RELATIVE_RMS: f64 = 0.02;
const MAX_METAL_ABSOLUTE_DIFFERENCE: f32 = 0.05;
const MAX_METAL_RELATIVE_RMS: f64 = 0.005;

static PINNED_FIXTURE_CHECK: OnceLock<(PathBuf, Result<(), String>)> = OnceLock::new();
static INFERENCE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires the pinned official Falcon-H1 Tiny 90M Q8_0 GGUF"]
fn falcon_h1_metadata_tokenizer_and_parallel_mamba2_are_pinned() -> anyhow::Result<()> {
    let files = fixture()?;
    assert_eq!(files.weight_file_count(), 1);
    assert_eq!(files.tensor_count(), Some(386));
    assert_eq!(
        files.gguf_tensor_type_counts(),
        [(GgufTensorType::F32, 169), (GgufTensorType::Q8_0, 217),]
    );

    let adapter = files.adapter()?;
    let spec = adapter
        .decoder_spec()
        .expect("Falcon-H1 uses the shared decoder");
    assert_eq!(adapter.family(), "falcon-h1");
    assert_eq!(spec.runtime.vocab_size, 32_768);
    assert_eq!(spec.runtime.max_context, 262_144);
    assert_eq!(spec.runtime.stop_token_ids, [11]);
    assert_eq!(spec.hidden_size, 512);
    assert_eq!(spec.intermediate_size, 768);
    assert_eq!(spec.num_layers, 24);
    assert_eq!(spec.num_query_heads, 8);
    assert_eq!(spec.num_kv_heads, 2);
    assert_eq!(spec.head_dim, 64);
    assert_eq!(spec.kv_bytes_per_token(2)?, 12_288);
    assert_eq!(spec.fixed_sequence_state_bytes(4)?, 4_976_640);
    assert!(spec.tied_embeddings);

    for (index, layer) in spec.layer_specs.iter().enumerate() {
        let SequenceMixerSpec::ParallelAttentionMamba2(parallel) = &layer.mixer else {
            panic!("Falcon-H1 layer {index} is not parallel attention/Mamba2");
        };
        assert_eq!(parallel.attention.kind, AttentionKind::FullCausal);
        assert!(!parallel.attention.qk_norm);
        assert!(!parallel.attention.output_gate);
        let rope = parallel
            .attention
            .rope()
            .expect("Falcon-H1 attention uses RoPE");
        assert_eq!(rope.rotary_dim, 64);
        assert!((rope.theta / 100_000_000_000.0 - 1.0).abs() < 1e-6);

        let mamba2 = parallel.mamba2;
        assert_eq!(mamba2.convolution_kernel_size, 4);
        assert_eq!(mamba2.inner_size, 768);
        assert_eq!(mamba2.state_size, 64);
        assert_eq!(mamba2.num_heads, 24);
        assert_eq!(mamba2.num_groups, 1);
        assert_eq!(mamba2.head_dim, 32);
        assert!(mamba2.convolution_bias);
        assert!(!mamba2.input_projection_bias);
        assert!(!mamba2.output_projection_bias);
        assert!(!mamba2.gated_rms_norm);
    }

    let tokenizer = files.tokenizer()?;
    let encoded = tokenizer
        .encode(RAW_PROMPT, true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(encoded.get_ids(), RAW_PROMPT_IDS);
    let decoded = tokenizer
        .decode(LLAMA_CPP_GREEDY_IDS, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(decoded, LLAMA_CPP_GREEDY_TEXT);
    assert!(adapter.format_chat(None, "Hello").is_err());
    Ok(())
}

#[test]
#[ignore = "requires the pinned official Falcon-H1 Tiny 90M Q8_0 GGUF"]
fn falcon_h1_cpu_matches_llama_cpp_and_state_semantics() -> anyhow::Result<()> {
    assert_backend(Device::Cpu, "CPU")
}

#[test]
#[ignore = "requires the pinned Falcon-H1 GGUF and Apple Metal"]
fn falcon_h1_metal_matches_llama_cpp_and_state_semantics() -> anyhow::Result<()> {
    assert_backend(Device::new_metal(0)?, "Metal")
}

fn assert_backend(device: Device, label: &str) -> anyhow::Result<()> {
    let _guard = INFERENCE_TEST_LOCK
        .lock()
        .expect("Falcon-H1 inference test lock poisoned");
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
    let split = 6;
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
    let fresh_logits = model.forward_tokens(fresh.as_mut(), &combined, 0)?;
    let fresh_logits = model.read_output_to_cpu(&fresh_logits)?;
    assert_logits_close(label, "cached/fresh", &cached_logits, &fresh_logits)?;
    assert_eq!(argmax(&cached_logits)?, LLAMA_CPP_GREEDY_IDS[1]);

    chunked.clear_cache();
    assert_eq!(chunked.cache_len(), 0);
    let restarted_logits = model.forward_tokens(chunked.as_mut(), RAW_PROMPT_IDS, 0)?;
    let restarted_logits = model.read_output_to_cpu(&restarted_logits)?;
    assert_logits_close(label, "clear/fresh", &restarted_logits, &one_shot_logits)?;
    Ok(())
}

fn assert_packed_ragged_semantics(model: &dyn CausalLm, label: &str) -> anyhow::Result<()> {
    assert_eq!(
        model.mixed_batch_capability(),
        CausalLmBatchCapability::PackedRagged
    );
    let first = &RAW_PROMPT_IDS[..4];
    let second = &RAW_PROMPT_IDS[4..7];
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

    let first_suffix = &RAW_PROMPT_IDS[7..9];
    let second_suffix = &RAW_PROMPT_IDS[9..];
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
    eprintln!("Falcon-H1 {label} {comparison}: max abs {maximum}, relative RMS {relative_rms}");
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
    let configured = PathBuf::from(std::env::var("MACLLM_FALCON_H1_GGUF_MODEL")?);
    let path = single_gguf_path(&configured)?;
    let (checked_path, result) = PINNED_FIXTURE_CHECK.get_or_init(|| {
        (
            path.clone(),
            validate_pinned_fixture(&path).map_err(|error| error.to_string()),
        )
    });
    anyhow::ensure!(
        checked_path == &path,
        "Falcon-H1 fixture changed within one test process"
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
        "expected exactly one Falcon-H1 GGUF fixture"
    );
    Ok(paths[0].clone())
}

fn validate_pinned_fixture(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(FILENAME),
        "unexpected Falcon-H1 fixture filename"
    );
    anyhow::ensure!(
        fs::metadata(path)?.len() == FIXTURE_SIZE,
        "unexpected Falcon-H1 fixture size"
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
        "unexpected Falcon-H1 fixture SHA-256"
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
