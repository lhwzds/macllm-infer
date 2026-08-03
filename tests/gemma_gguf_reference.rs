//! Real Gemma 1 GGUF gates against a pinned public fixture and llama.cpp.
//!
//! Fixture: `tensorblock/gemma-2b-it-GGUF` at
//! `a624a84e8ed8b33a324cff961d0d3e33d2ada43b`, `Q2_K`.
//!
//! Run with:
//! `MACLLM_GEMMA_GGUF_MODEL=models/gemma-2b-it-q2_k \
//!  cargo test --release --test gemma_gguf_reference -- --ignored --nocapture`

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, Device, Tensor};
use macllm_infer::{
    architecture::{Activation, AttentionKind, CausalLm, NormLayout, RopeLayout},
    generation::{GenerationConfig, generate},
    gguf::GgufTensorType,
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use sha2::{Digest, Sha256};

const FIXTURE_URL: &str = "https://huggingface.co/tensorblock/gemma-2b-it-GGUF/resolve/a624a84e8ed8b33a324cff961d0d3e33d2ada43b/gemma-2b-it-Q2_K.gguf";
const FILENAME: &str = "gemma-2b-it-Q2_K.gguf";
const FIXTURE_SIZE: u64 = 1_157_924_128;
const FIXTURE_SHA256: &str = "0fd8ac13b7cba9bb1813f5dd3366a3970cbb27b515f4a9e6d2970beaae5450ba";

const USER_PROMPT: &str = "The capital of France is";
const RAW_CHAT_PROMPT: &str =
    "<start_of_turn>user\nThe capital of France is<end_of_turn>\n<start_of_turn>model\n";
const FORMATTED_CHAT_PROMPT: &str =
    "<bos><start_of_turn>user\nThe capital of France is<end_of_turn>\n<start_of_turn>model\n";
const PROMPT_IDS: &[u32] = &[
    2, 106, 1_645, 108, 651, 6_037, 576, 6_081, 603, 107, 108, 106, 2_516, 108,
];

// Captured with llama.cpp build 10050 (`b15ca938a`) on CPU:
//
// llama-completion -m gemma-2b-it-Q2_K.gguf -p "$RAW_CHAT_PROMPT" \
//   -n 8 -no-cnv --no-display-prompt --temp 0 --repeat-penalty 1 \
//   --no-warmup --simple-io --device none -ngl 0 --no-op-offload \
//   --no-kv-offload --fit off --special
//
// The exact decoded continuation is `The capital of France is Paris. It`.
const LLAMA_CPP_GREEDY_IDS: &[u32] = &[651, 6_037, 576, 6_081, 603, 7_127, 235_265, 1_165];

const MAX_CPU_CACHE_FULL_DIFFERENCE: f32 = 2.0;
const MAX_CPU_CACHE_FULL_RELATIVE_RMS: f64 = 0.02;
const MAX_METAL_CACHE_FULL_DIFFERENCE: f32 = 0.1;
const MAX_METAL_CACHE_FULL_RELATIVE_RMS: f64 = 0.001;

static PINNED_FIXTURE_CHECK: OnceLock<(PathBuf, Result<(), String>)> = OnceLock::new();
static INFERENCE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires the pinned Gemma 1 2B Q2_K GGUF"]
fn gemma_metadata_tokenizer_and_chat_contract_are_pinned() -> anyhow::Result<()> {
    let files = fixture()?;
    assert_eq!(files.weight_file_count(), 1);
    assert_eq!(files.tensor_count(), Some(164));
    assert_eq!(
        files.gguf_tensor_type_counts(),
        [
            (GgufTensorType::F32, 37),
            (GgufTensorType::Q2K, 72),
            (GgufTensorType::Q3K, 36),
            (GgufTensorType::Q4K, 18),
            (GgufTensorType::Q6K, 1),
        ]
    );

    let adapter = files.adapter()?;
    let spec = adapter
        .decoder_spec()
        .expect("Gemma 1 uses the shared decoder");
    assert_eq!(adapter.family(), "gemma");
    assert_eq!(spec.architecture, "gemma");
    assert_eq!(spec.runtime.vocab_size, 256_000);
    assert_eq!(spec.runtime.max_context, 8_192);
    assert_eq!(spec.runtime.stop_token_ids, [1, 107]);
    assert_eq!(spec.hidden_size, 2_048);
    assert_eq!(spec.intermediate_size, 16_384);
    assert_eq!(spec.num_layers, 18);
    assert_eq!(spec.num_query_heads, 8);
    assert_eq!(spec.num_kv_heads, 1);
    assert_eq!(spec.head_dim, 256);
    assert!((spec.norm_epsilon - 1e-6).abs() < 1e-12);
    assert!((spec.embedding_scale - 2_048.0_f64.sqrt()).abs() < f64::EPSILON);
    assert_eq!(spec.final_logit_softcap, None);
    assert!(spec.tied_embeddings);

    for (index, layer) in spec.layer_specs.iter().enumerate() {
        assert_eq!(layer.activation, Activation::GeluTanh, "layer {index}");
        assert_eq!(layer.norm_layout, NormLayout::PreNorm, "layer {index}");
        let attention = layer
            .attention()
            .expect("every Gemma 1 layer uses attention");
        assert_eq!(attention.kind, AttentionKind::FullCausal, "layer {index}");
        assert!(!attention.qk_norm, "layer {index}");
        assert!((attention.scale - 0.0625).abs() < f64::EPSILON);
        assert_eq!(attention.logit_softcap, None);
        let rope = attention.rope().expect("Gemma attention uses RoPE");
        assert!((rope.theta - 10_000.0).abs() < f64::EPSILON);
        assert_eq!(rope.layout, RopeLayout::HalfSplit);
        assert_eq!(rope.rotary_dim, 256);
    }

    let tokenizer = files.tokenizer()?;
    let raw = tokenizer
        .encode(RAW_CHAT_PROMPT, true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(raw.get_ids(), PROMPT_IDS);

    let chat = adapter.format_chat(None, USER_PROMPT)?;
    assert_eq!(chat, FORMATTED_CHAT_PROMPT);
    let chat = tokenizer
        .encode(chat, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(chat.get_ids(), PROMPT_IDS);
    assert!(adapter.format_chat(Some("System"), USER_PROMPT).is_err());
    Ok(())
}

#[test]
#[ignore = "requires the pinned Gemma 1 2B Q2_K GGUF"]
fn gemma_cpu_cache_matches_fresh_full_forward_and_llama_cpp_golden() -> anyhow::Result<()> {
    assert_backend(Device::Cpu, "CPU")
}

#[test]
#[ignore = "requires the pinned Gemma 1 2B Q2_K GGUF and Apple Metal"]
fn gemma_metal_cache_matches_fresh_full_forward_and_llama_cpp_golden() -> anyhow::Result<()> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        assert_backend(Device::new_metal(0)?, "Metal")
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        eprintln!("skipping Gemma 1 Metal gate outside a Metal-enabled macOS build");
        Ok(())
    }
}

fn assert_backend(device: Device, label: &str) -> anyhow::Result<()> {
    let _guard = INFERENCE_TEST_LOCK
        .lock()
        .expect("Gemma 1 inference test lock poisoned");
    let files = fixture()?;
    let model = build(&files, &device)?;

    let mut cached_session = model.new_session()?;
    let prefill = model.forward_tokens(cached_session.as_mut(), PROMPT_IDS, 0)?;
    let next_token = argmax(&prefill)?;
    assert_eq!(
        next_token, LLAMA_CPP_GREEDY_IDS[0],
        "{label} prefill golden"
    );

    let cached = model
        .forward_tokens(cached_session.as_mut(), &[next_token], PROMPT_IDS.len())?
        .to_device(&Device::Cpu)?;
    let mut combined = PROMPT_IDS.to_vec();
    combined.push(next_token);
    let mut full_session = model.new_session()?;
    let full = model
        .forward_tokens(full_session.as_mut(), &combined, 0)?
        .to_device(&Device::Cpu)?;

    assert_eq!(
        argmax(&cached)?,
        argmax(&full)?,
        "{label} cache/full argmax"
    );
    assert_eq!(
        argmax(&cached)?,
        LLAMA_CPP_GREEDY_IDS[1],
        "{label} decode golden"
    );
    let (difference, relative_rms) = logit_difference_stats(&cached, &full)?;
    let (max_difference, max_relative_rms) = backend_thresholds(label);
    eprintln!("Gemma 1 {label} cache/full: max abs {difference}; relative RMS {relative_rms}");
    anyhow::ensure!(
        difference <= max_difference,
        "Gemma 1 {label} cache/full max difference {difference} exceeds {max_difference}"
    );
    anyhow::ensure!(
        relative_rms <= max_relative_rms,
        "Gemma 1 {label} cache/full relative RMS {relative_rms} exceeds {max_relative_rms}"
    );

    let generated = generate_greedy(model.as_ref(), PROMPT_IDS, LLAMA_CPP_GREEDY_IDS.len())?;
    assert_eq!(
        generated, LLAMA_CPP_GREEDY_IDS,
        "{label} llama.cpp Gemma 1 golden"
    );
    Ok(())
}

fn backend_thresholds(label: &str) -> (f32, f64) {
    if label == "CPU" {
        (
            MAX_CPU_CACHE_FULL_DIFFERENCE,
            MAX_CPU_CACHE_FULL_RELATIVE_RMS,
        )
    } else {
        (
            MAX_METAL_CACHE_FULL_DIFFERENCE,
            MAX_METAL_CACHE_FULL_RELATIVE_RMS,
        )
    }
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

fn fixture() -> anyhow::Result<ModelFiles> {
    let configured = configured_fixture();
    let path = single_gguf_path(&configured)?;
    let (checked_path, result) = PINNED_FIXTURE_CHECK.get_or_init(|| {
        (
            path.clone(),
            validate_pinned_fixture(&path).map_err(|error| error.to_string()),
        )
    });
    anyhow::ensure!(
        checked_path == &path,
        "Gemma 1 fixture changed within one test process"
    );
    if let Err(error) = result {
        anyhow::bail!(error.clone());
    }
    ModelFiles::from_dir(path)
}

fn configured_fixture() -> PathBuf {
    std::env::var_os("MACLLM_GEMMA_GGUF_MODEL")
        .map_or_else(|| PathBuf::from("models/gemma-2b-it-q2_k"), PathBuf::from)
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
        "expected exactly one Gemma 1 GGUF fixture"
    );
    Ok(paths[0].clone())
}

fn validate_pinned_fixture(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(FILENAME),
        "unexpected Gemma 1 fixture filename; download {FIXTURE_URL}"
    );
    anyhow::ensure!(
        fs::metadata(path)?.len() == FIXTURE_SIZE,
        "unexpected Gemma 1 fixture size; download {FIXTURE_URL}"
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
        "unexpected Gemma 1 fixture SHA-256; download {FIXTURE_URL}"
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
