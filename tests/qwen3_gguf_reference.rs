//! Real Qwen3 GGUF gates against Candle and a pinned llama.cpp golden.
//!
//! Fixture: `Qwen/Qwen3-0.6B-GGUF` at
//! `23749fefcc72300e3a2ad315e1317431b06b590a`, official `Q8_0` SHA-256
//! `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`.
//!
//! Run with:
//! `MACLLM_QWEN3_GGUF_MODEL=models/qwen3-0.6b-q8_0 \
//!  cargo test --release --test qwen3_gguf_reference -- --ignored --nocapture`

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use candle_transformers::models::quantized_qwen3;
use macllm_infer::{
    architecture::CausalLm,
    generation::{GenerationConfig, generate},
    gguf::GgufTensorType,
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use sha2::{Digest, Sha256};

const FILENAME: &str = "Qwen3-0.6B-Q8_0.gguf";
const FIXTURE_SIZE: u64 = 639_446_688;
const FIXTURE_SHA256: &str = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031";
const RAW_PROMPT: &str = "2 + 2 =";
const RAW_PROMPT_IDS: &[u32] = &[17, 488, 220, 17, 284];
const CHAT_PROMPT_IDS: &[u32] = &[
    151_644, 872, 198, 17, 488, 220, 17, 284, 151_645, 198, 151_644, 77_091, 198,
];
// Captured twice on both CPU and Metal with llama.cpp b10050 (b15ca938a):
// `llama-completion -m FILE -no-cnv -dev none -p '2 + 2 =' -n 8
//  --temp 0 --repeat-penalty 1 --seed 42 --no-display-prompt --no-perf
//  --simple-io -lv 2`; replace `-dev none` with `-ngl all` for Metal.
const LLAMA_CPP_GREEDY_IDS: &[u32] = &[220, 19, 11, 773, 279, 2_629, 374, 220];
const CHAT_LLAMA_CPP_GREEDY_IDS: &[u32] = &[151_667, 198, 32_313, 11, 773, 358, 1_184, 311];
// Candle uses fused CPU flash attention while the shared runtime uses standard
// tensor attention. Q8 matrix-vector versus matrix-matrix reductions also
// accumulate differently across 28 layers, so CPU keeps a bounded numeric
// envelope plus exact argmax and llama.cpp-token requirements. Both Metal
// implementations take the same kernels and therefore use a tighter gate.
const MAX_CPU_REFERENCE_DIFFERENCE: f32 = 0.25;
const MAX_CPU_REFERENCE_RELATIVE_RMS: f64 = 0.03;
const MAX_METAL_REFERENCE_DIFFERENCE: f32 = 0.001;
const MAX_METAL_REFERENCE_RELATIVE_RMS: f64 = 0.0001;
const MAX_CPU_MAC_CACHE_FULL_DIFFERENCE: f32 = 0.2;
const MAX_CPU_MAC_CACHE_FULL_RELATIVE_RMS: f64 = 0.02;
const MAX_METAL_MAC_CACHE_FULL_DIFFERENCE: f32 = 0.02;
const MAX_METAL_MAC_CACHE_FULL_RELATIVE_RMS: f64 = 0.001;
const MAX_CANDLE_CACHE_FULL_DIFFERENCE: f32 = 0.01;
const MAX_CANDLE_CACHE_FULL_RELATIVE_RMS: f64 = 0.001;

static PINNED_FIXTURE_CHECK: OnceLock<(PathBuf, Result<(), String>)> = OnceLock::new();
// Candle's Metal registry and the ~640 MB model should not be initialized by
// multiple ignored tests concurrently in the same process.
static INFERENCE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires the pinned official Qwen3 0.6B Q8_0 GGUF"]
fn qwen3_metadata_tokenizer_and_chat_contract_are_pinned() -> anyhow::Result<()> {
    let files = fixture()?;
    assert_eq!(files.weight_file_count(), 1);
    assert_eq!(files.tensor_count(), Some(310));
    assert_eq!(
        files.gguf_tensor_type_counts(),
        [(GgufTensorType::F32, 113), (GgufTensorType::Q8_0, 197),]
    );

    let adapter = files.adapter()?;
    let spec = adapter
        .decoder_spec()
        .expect("Qwen3 uses the shared decoder");
    assert_eq!(adapter.family(), "qwen3");
    assert_eq!(spec.runtime.vocab_size, 151_936);
    assert_eq!(spec.runtime.max_context, 40_960);
    assert_eq!(spec.runtime.stop_token_ids, [151_645, 151_643]);
    assert_eq!(spec.hidden_size, 1_024);
    assert_eq!(spec.intermediate_size, 3_072);
    assert_eq!(spec.num_layers, 28);
    assert_eq!(spec.num_query_heads, 16);
    assert_eq!(spec.num_kv_heads, 8);
    assert_eq!(spec.head_dim, 128);
    assert_eq!(spec.num_query_heads * spec.head_dim, 2_048);
    assert_ne!(spec.num_query_heads * spec.head_dim, spec.hidden_size);
    assert!(
        spec.layer_specs
            .iter()
            .all(|layer| layer.attention().is_some_and(|attention| attention.qk_norm))
    );
    assert!(spec.tied_embeddings);

    let tokenizer = files.tokenizer()?;
    let raw = tokenizer
        .encode(RAW_PROMPT, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(raw.get_ids(), RAW_PROMPT_IDS);
    let chat = adapter.format_chat(None, RAW_PROMPT)?;
    assert_eq!(
        chat,
        "<|im_start|>user\n2 + 2 =<|im_end|>\n<|im_start|>assistant\n"
    );
    let chat = tokenizer
        .encode(chat, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(chat.get_ids(), CHAT_PROMPT_IDS);
    Ok(())
}

#[test]
#[ignore = "requires the pinned Qwen3 Q8_0 GGUF"]
fn qwen3_cpu_matches_candle_llama_cpp_and_cache_full() -> anyhow::Result<()> {
    assert_backend(Device::Cpu, "CPU")
}

#[test]
#[ignore = "requires the pinned Qwen3 Q8_0 GGUF and Apple Metal"]
fn qwen3_metal_matches_candle_llama_cpp_and_cache_full() -> anyhow::Result<()> {
    assert_backend(Device::new_metal(0)?, "Metal")
}

fn assert_backend(device: Device, label: &str) -> anyhow::Result<()> {
    let _guard = INFERENCE_TEST_LOCK
        .lock()
        .expect("Qwen3 inference test lock poisoned");
    let files = fixture()?;
    let model_path = single_gguf_path(&configured_fixture()?)?;
    let actual = build(&files, &device)?;
    let mut reference = build_reference(&model_path, &device)?;
    assert_logit_and_cache_gates(actual.as_ref(), &mut reference, &device, label)?;

    drop(reference);
    drop(actual);
    assert_greedy_goldens(&files, &device, label)
}

fn build_reference(
    model_path: &Path,
    device: &Device,
) -> anyhow::Result<quantized_qwen3::ModelWeights> {
    let mut reader = File::open(model_path)?;
    let mut content = gguf_file::Content::read(&mut reader)?;
    // Candle otherwise defaults this GGUF to F16 because the standard file
    // does not carry its private `general.dtype` hint. F32 is required on CPU
    // and gives both backends the same activation precision on Metal.
    content
        .metadata
        .insert("general.dtype".into(), gguf_file::Value::U32(0));
    quantized_qwen3::ModelWeights::from_gguf(content, &mut reader, device)
        .map_err(anyhow::Error::from)
}

fn assert_logit_and_cache_gates(
    actual: &dyn CausalLm,
    reference: &mut quantized_qwen3::ModelWeights,
    device: &Device,
    label: &str,
) -> anyhow::Result<()> {
    let (reference_difference, reference_relative_rms, cache_difference, cache_relative_rms) =
        backend_thresholds(label);

    let input = Tensor::from_vec(RAW_PROMPT_IDS.to_vec(), (1, RAW_PROMPT_IDS.len()), device)?;
    let mut actual_cached_session = actual.new_session()?;
    let actual_prefill =
        actual.forward_tokens(actual_cached_session.as_mut(), RAW_PROMPT_IDS, 0)?;
    let reference_prefill = reference.forward(&input, 0)?;
    assert_reference_logits(
        label,
        "prefill",
        &actual_prefill,
        &reference_prefill,
        reference_difference,
        reference_relative_rms,
    )?;
    assert_eq!(argmax(&actual_prefill)?, LLAMA_CPP_GREEDY_IDS[0]);

    let next_token = LLAMA_CPP_GREEDY_IDS[0];
    let actual_cached = actual
        .forward_tokens(
            actual_cached_session.as_mut(),
            &[next_token],
            RAW_PROMPT_IDS.len(),
        )?
        .to_device(&Device::Cpu)?;
    let next = Tensor::new(&[[next_token]], device)?;
    let reference_cached = reference
        .forward(&next, RAW_PROMPT_IDS.len())?
        .to_device(&Device::Cpu)?;
    assert_reference_logits(
        label,
        "cached",
        &actual_cached,
        &reference_cached,
        reference_difference,
        reference_relative_rms,
    )?;
    assert_eq!(argmax(&actual_cached)?, LLAMA_CPP_GREEDY_IDS[1]);

    reference.clear_kv_cache();
    let mut combined = RAW_PROMPT_IDS.to_vec();
    combined.push(next_token);
    let mut actual_full_session = actual.new_session()?;
    let actual_full = actual
        .forward_tokens(actual_full_session.as_mut(), &combined, 0)?
        .to_device(&Device::Cpu)?;
    let combined_input = Tensor::from_vec(combined.clone(), (1, combined.len()), device)?;
    let reference_full = reference
        .forward(&combined_input, 0)?
        .to_device(&Device::Cpu)?;
    assert_reference_logits(
        label,
        "fresh-full",
        &actual_full,
        &reference_full,
        reference_difference,
        reference_relative_rms,
    )?;

    assert_cache_full(
        &format!("MacLLM {label}"),
        &actual_cached,
        &actual_full,
        cache_difference,
        cache_relative_rms,
    )?;
    assert_cache_full(
        &format!("Candle {label}"),
        &reference_cached,
        &reference_full,
        MAX_CANDLE_CACHE_FULL_DIFFERENCE,
        MAX_CANDLE_CACHE_FULL_RELATIVE_RMS,
    )?;
    Ok(())
}

fn backend_thresholds(label: &str) -> (f32, f64, f32, f64) {
    if label == "CPU" {
        (
            MAX_CPU_REFERENCE_DIFFERENCE,
            MAX_CPU_REFERENCE_RELATIVE_RMS,
            MAX_CPU_MAC_CACHE_FULL_DIFFERENCE,
            MAX_CPU_MAC_CACHE_FULL_RELATIVE_RMS,
        )
    } else {
        (
            MAX_METAL_REFERENCE_DIFFERENCE,
            MAX_METAL_REFERENCE_RELATIVE_RMS,
            MAX_METAL_MAC_CACHE_FULL_DIFFERENCE,
            MAX_METAL_MAC_CACHE_FULL_RELATIVE_RMS,
        )
    }
}

fn assert_greedy_goldens(files: &ModelFiles, device: &Device, label: &str) -> anyhow::Result<()> {
    let greedy = build(files, device)?;
    let generated = generate_greedy(greedy.as_ref(), RAW_PROMPT_IDS, LLAMA_CPP_GREEDY_IDS.len())?;
    assert_eq!(generated, LLAMA_CPP_GREEDY_IDS, "{label} llama.cpp golden");

    let generated = generate_greedy(
        greedy.as_ref(),
        CHAT_PROMPT_IDS,
        CHAT_LLAMA_CPP_GREEDY_IDS.len(),
    )?;
    assert_eq!(
        generated, CHAT_LLAMA_CPP_GREEDY_IDS,
        "{label} Qwen3 thinking-template golden"
    );
    Ok(())
}

fn generate_greedy(
    model: &dyn CausalLm,
    prompt: &[u32],
    max_new_tokens: usize,
) -> anyhow::Result<Vec<u32>> {
    let mut session = model.new_session()?;
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: 42,
        ..SamplingConfig::default()
    })?;
    let mut generated = Vec::new();
    generate(
        model,
        session.as_mut(),
        prompt,
        &mut sampler,
        GenerationConfig {
            max_new_tokens,
            ignore_stop_tokens: true,
        },
        |token| {
            generated.push(token);
            Ok(())
        },
    )?;
    Ok(generated)
}

fn assert_reference_logits(
    backend: &str,
    phase: &str,
    actual: &Tensor,
    reference: &Tensor,
    max_difference: f32,
    max_relative_rms: f64,
) -> anyhow::Result<()> {
    let (difference, relative_rms) = logit_difference_stats(actual, reference)?;
    eprintln!(
        "MacLLM/Candle Qwen3 {backend} {phase}: max abs {difference}; relative RMS {relative_rms}"
    );
    anyhow::ensure!(
        difference <= max_difference,
        "MacLLM/Candle {backend} {phase} max difference {difference} exceeds {max_difference}"
    );
    anyhow::ensure!(
        relative_rms <= max_relative_rms,
        "MacLLM/Candle {backend} {phase} relative RMS {relative_rms} exceeds {max_relative_rms}"
    );
    anyhow::ensure!(
        argmax(actual)? == argmax(reference)?,
        "MacLLM/Candle {backend} {phase} argmax differs"
    );
    Ok(())
}

fn assert_cache_full(
    label: &str,
    cached: &Tensor,
    full: &Tensor,
    max_difference: f32,
    max_relative_rms: f64,
) -> anyhow::Result<()> {
    let (difference, relative_rms) = logit_difference_stats(cached, full)?;
    eprintln!("{label} cache/full: max abs {difference}; relative RMS {relative_rms}");
    anyhow::ensure!(
        difference <= max_difference,
        "{label} cache/full max difference {difference} exceeds {max_difference}"
    );
    anyhow::ensure!(
        relative_rms <= max_relative_rms,
        "{label} cache/full relative RMS {relative_rms} exceeds {max_relative_rms}"
    );
    anyhow::ensure!(
        argmax(cached)? == argmax(full)?,
        "{label} cache/full argmax differs"
    );
    Ok(())
}

fn fixture() -> anyhow::Result<ModelFiles> {
    let configured = configured_fixture()?;
    let path = single_gguf_path(&configured)?;
    let (checked_path, result) = PINNED_FIXTURE_CHECK.get_or_init(|| {
        (
            path.clone(),
            validate_pinned_fixture(&path).map_err(|error| error.to_string()),
        )
    });
    anyhow::ensure!(
        checked_path == &path,
        "Qwen3 fixture changed within one test process"
    );
    if let Err(error) = result {
        anyhow::bail!(error.clone());
    }
    ModelFiles::from_dir(path)
}

fn configured_fixture() -> anyhow::Result<PathBuf> {
    Ok(PathBuf::from(std::env::var("MACLLM_QWEN3_GGUF_MODEL")?))
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
    anyhow::ensure!(paths.len() == 1, "expected exactly one Qwen3 GGUF fixture");
    Ok(paths[0].clone())
}

fn validate_pinned_fixture(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(FILENAME),
        "unexpected Qwen3 fixture filename"
    );
    anyhow::ensure!(
        fs::metadata(path)?.len() == FIXTURE_SIZE,
        "unexpected Qwen3 fixture size"
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
        "unexpected Qwen3 fixture SHA-256"
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

fn argmax(logits: &Tensor) -> candle_core::Result<u32> {
    logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()
}

fn logit_difference_stats(left: &Tensor, right: &Tensor) -> candle_core::Result<(f32, f64)> {
    let left = left
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let right = right
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(!left.is_empty());
    assert_eq!(left.len(), right.len());
    let mut max_difference = 0.0_f32;
    let mut squared_difference = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    for (&left, &right) in left.iter().zip(&right) {
        assert!(left.is_finite() && right.is_finite());
        let difference = (left - right).abs();
        max_difference = max_difference.max(difference);
        squared_difference += f64::from(difference).powi(2);
        squared_reference += f64::from(right).powi(2);
    }
    Ok((
        max_difference,
        (squared_difference / squared_reference.max(f64::EPSILON)).sqrt(),
    ))
}
