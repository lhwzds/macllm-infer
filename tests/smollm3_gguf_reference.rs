//! Real `SmolLM3` GGUF gates against a pinned official fixture and llama.cpp.
//!
//! Fixture: `ggml-org/SmolLM3-3B-GGUF` revision
//! `4965cb60b150737b68a0408c36aeefb65078f894`, official `Q4_K_M`.
//!
//! Run with the repository-local fixture:
//! `cargo test --release --test smollm3_gguf_reference -- --ignored --nocapture`
//! or override it with `MACLLM_SMOLLM3_GGUF_MODEL`.

use std::{
    fs::{self, File},
    io::Read,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, Device, Tensor};
use macllm_infer::{
    architecture::{CausalLm, PositionEncodingSpec, RopeLayout},
    generation::{GenerationConfig, generate, generate_with_prefill_token_quantum},
    gguf::GgufTensorType,
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use sha2::{Digest, Sha256};

const FILENAME: &str = "SmolLM3-Q4_K_M.gguf";
const FIXTURE_SIZE: u64 = 1_915_305_312;
const FIXTURE_SHA256: &str = "8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e";
const RAW_PROMPT: &str = "2 + 2 =";
const RAW_PROMPT_IDS: &[u32] = &[17, 489, 220, 17, 284];
const UNICODE_TEXT: &str = "café 👋 1234567";
const UNICODE_IDS: &[u32] = &[936, 59_958, 62_904, 233, 220, 4_513, 10_961, 22];
const SYSTEM_MESSAGE: &str = "/system_override/no_think";
const CHAT_PROMPT: &str = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n2 + 2 =<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n";
const CHAT_PROMPT_IDS: &[u32] = &[
    128_011, 9_125, 198, 128_012, 198, 128_011, 882, 198, 17, 489, 220, 17, 284, 128_012, 198,
    128_011, 78_191, 198, 128_002, 271, 128_003, 198,
];

// Captured with llama.cpp b10050 from the pinned file on both CPU and Metal:
// `llama-completion -m FILE -no-cnv -dev none -p '2 + 2 =' -n 8
//  --temp 0 --repeat-penalty 1 --seed 42 --no-display-prompt --no-perf
//  --simple-io -lv 2`; replace `-dev none` with `-ngl all` for Metal.
const LLAMA_CPP_GREEDY_IDS: &[u32] = &[220, 19, 271, 16, 13, 578, 5_219, 220];
// The deterministic `/system_override/no_think` chat repeats the expression
// before answering it; these eight tokens decode to `2 + 2 = 4\n\n`.
const CHAT_LLAMA_CPP_GREEDY_IDS: &[u32] = &[17, 489, 220, 17, 284, 220, 19, 271];

// Quantized matrix-vector and matrix-matrix reductions can accumulate in a
// different order. Exact next-token identity remains mandatory while this
// numeric envelope catches material cache/full regressions.
const MAX_CPU_CACHE_FULL_DIFFERENCE: f32 = 2.5;
const MAX_METAL_CACHE_FULL_DIFFERENCE: f32 = 0.2;

static PINNED_FIXTURE_CHECK: OnceLock<(PathBuf, Result<(), String>)> = OnceLock::new();
static INFERENCE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires the pinned official SmolLM3-3B Q4_K_M GGUF"]
fn smollm3_metadata_tokenizer_chat_and_nope_schedule_are_pinned() -> anyhow::Result<()> {
    let files = fixture()?;
    assert_eq!(files.weight_file_count(), 1);
    assert_eq!(files.tensor_count(), Some(326));
    assert_eq!(
        files.gguf_tensor_type_counts(),
        [
            (GgufTensorType::F32, 73),
            (GgufTensorType::Q4K, 216),
            (GgufTensorType::Q6K, 37),
        ]
    );

    let adapter = files.adapter()?;
    let spec = adapter
        .decoder_spec()
        .expect("SmolLM3 uses the shared decoder");
    assert_eq!(adapter.family(), "smollm3");
    assert_eq!(spec.runtime.vocab_size, 128_256);
    assert_eq!(spec.runtime.max_context, 65_536);
    assert_eq!(spec.runtime.stop_token_ids, [128_012]);
    assert_eq!(spec.hidden_size, 2_048);
    assert_eq!(spec.intermediate_size, 11_008);
    assert_eq!(spec.num_layers, 36);
    assert_eq!(spec.num_query_heads, 16);
    assert_eq!(spec.num_kv_heads, 4);
    assert_eq!(spec.head_dim, 128);
    assert!(spec.tied_embeddings);

    for (index, layer) in spec.layer_specs.iter().enumerate() {
        let attention = layer
            .attention()
            .expect("every SmolLM3 decoder layer uses attention");
        if (index + 1).is_multiple_of(4) {
            assert!(
                matches!(attention.position_encoding, PositionEncodingSpec::None),
                "zero-based layer {index} must omit positional encoding"
            );
        } else {
            let PositionEncodingSpec::Rope(rope) = &attention.position_encoding else {
                panic!("zero-based layer {index} must use RoPE");
            };
            assert_eq!(rope.layout, RopeLayout::HalfSplit);
            assert_eq!(rope.rotary_dim, 128);
            assert!((rope.theta - 5_000_000.0).abs() < f64::EPSILON);
            assert!(rope.frequency_factors.is_none());
        }
    }

    let tokenizer = files.tokenizer()?;
    let raw = tokenizer
        .encode(RAW_PROMPT, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(raw.get_ids(), RAW_PROMPT_IDS);
    let unicode = tokenizer
        .encode(UNICODE_TEXT, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(unicode.get_ids(), UNICODE_IDS);

    let chat = adapter.format_chat(Some(SYSTEM_MESSAGE), RAW_PROMPT)?;
    assert_eq!(chat, CHAT_PROMPT);
    let chat = tokenizer
        .encode(chat, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(chat.get_ids(), CHAT_PROMPT_IDS);
    Ok(())
}

#[test]
#[ignore = "requires the pinned official SmolLM3-3B Q4_K_M GGUF"]
fn smollm3_cpu_matches_llama_cpp_and_cache_full() -> anyhow::Result<()> {
    assert_backend(Device::Cpu, "CPU")
}

#[test]
#[ignore = "requires the pinned SmolLM3 GGUF and Apple Metal"]
fn smollm3_metal_matches_llama_cpp_and_cache_full() -> anyhow::Result<()> {
    assert_backend(Device::new_metal(0)?, "Metal")
}

fn assert_backend(device: Device, label: &str) -> anyhow::Result<()> {
    let _guard = INFERENCE_TEST_LOCK
        .lock()
        .expect("SmolLM3 inference test lock poisoned");
    let files = fixture()?;
    let model = build(&files, &device)?;
    assert_cache_full(model.as_ref(), label)?;

    let generated = generate_greedy(model.as_ref(), RAW_PROMPT_IDS, LLAMA_CPP_GREEDY_IDS.len())?;
    assert_eq!(generated, LLAMA_CPP_GREEDY_IDS, "{label} llama.cpp golden");

    let generated = generate_greedy(
        model.as_ref(),
        CHAT_PROMPT_IDS,
        CHAT_LLAMA_CPP_GREEDY_IDS.len(),
    )?;
    assert_eq!(
        generated, CHAT_LLAMA_CPP_GREEDY_IDS,
        "{label} deterministic-chat llama.cpp golden"
    );

    let long_prompt = RAW_PROMPT_IDS.repeat(30);
    assert!(long_prompt.len() > 128);
    let one_shot = generate_greedy(model.as_ref(), &long_prompt, 1)?;
    let bounded = generate_greedy_with_quantum(
        model.as_ref(),
        &long_prompt,
        1,
        NonZeroUsize::new(128).expect("the real-gate prefill quantum is positive"),
    )?;
    assert_eq!(
        bounded, one_shot,
        "{label} bounded prefill must preserve the greedy token"
    );
    Ok(())
}

fn assert_cache_full(model: &dyn CausalLm, label: &str) -> anyhow::Result<()> {
    let mut cached_session = model.new_session()?;
    let prefill = model.forward_tokens(cached_session.as_mut(), RAW_PROMPT_IDS, 0)?;
    let prefill = model.read_output_to_cpu(&prefill)?;
    assert_eq!(argmax(&prefill)?, LLAMA_CPP_GREEDY_IDS[0]);

    let next_token = LLAMA_CPP_GREEDY_IDS[0];
    let cached =
        model.forward_tokens(cached_session.as_mut(), &[next_token], RAW_PROMPT_IDS.len())?;
    let cached = model.read_output_to_cpu(&cached)?;
    assert_eq!(argmax(&cached)?, LLAMA_CPP_GREEDY_IDS[1]);

    let mut combined = RAW_PROMPT_IDS.to_vec();
    combined.push(next_token);
    let mut fresh_session = model.new_session()?;
    let fresh = model.forward_tokens(fresh_session.as_mut(), &combined, 0)?;
    let fresh = model.read_output_to_cpu(&fresh)?;
    assert_eq!(argmax(&fresh)?, LLAMA_CPP_GREEDY_IDS[1]);

    let difference = max_abs_difference(&cached, &fresh)?;
    let maximum = if label == "CPU" {
        MAX_CPU_CACHE_FULL_DIFFERENCE
    } else {
        MAX_METAL_CACHE_FULL_DIFFERENCE
    };
    eprintln!("SmolLM3 {label} cache/full max abs difference: {difference}");
    anyhow::ensure!(
        difference <= maximum,
        "SmolLM3 {label} cache/full max difference {difference} exceeds {maximum}"
    );
    Ok(())
}

fn generate_greedy(
    model: &dyn CausalLm,
    prompt: &[u32],
    max_new_tokens: usize,
) -> anyhow::Result<Vec<u32>> {
    generate_greedy_with_optional_quantum(model, prompt, max_new_tokens, None)
}

fn generate_greedy_with_quantum(
    model: &dyn CausalLm,
    prompt: &[u32],
    max_new_tokens: usize,
    prefill_token_quantum: NonZeroUsize,
) -> anyhow::Result<Vec<u32>> {
    generate_greedy_with_optional_quantum(
        model,
        prompt,
        max_new_tokens,
        Some(prefill_token_quantum),
    )
}

fn generate_greedy_with_optional_quantum(
    model: &dyn CausalLm,
    prompt: &[u32],
    max_new_tokens: usize,
    prefill_token_quantum: Option<NonZeroUsize>,
) -> anyhow::Result<Vec<u32>> {
    let mut session = model.new_session()?;
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: 42,
        ..SamplingConfig::default()
    })?;
    let mut generated = Vec::new();
    let config = GenerationConfig {
        max_new_tokens,
        ignore_stop_tokens: true,
    };
    if let Some(prefill_token_quantum) = prefill_token_quantum {
        generate_with_prefill_token_quantum(
            model,
            session.as_mut(),
            prompt,
            &mut sampler,
            config,
            prefill_token_quantum,
            |token| {
                generated.push(token);
                Ok(())
            },
        )?;
    } else {
        generate(
            model,
            session.as_mut(),
            prompt,
            &mut sampler,
            config,
            |token| {
                generated.push(token);
                Ok(())
            },
        )?;
    }
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
        "SmolLM3 fixture changed within one test process"
    );
    if let Err(error) = result {
        anyhow::bail!(error.clone());
    }
    ModelFiles::from_dir(path)
}

fn configured_fixture() -> PathBuf {
    std::env::var_os("MACLLM_SMOLLM3_GGUF_MODEL").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/smollm3-3b-q4_k_m"),
        PathBuf::from,
    )
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
        "expected exactly one SmolLM3 GGUF fixture"
    );
    Ok(paths[0].clone())
}

fn validate_pinned_fixture(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(FILENAME),
        "unexpected SmolLM3 fixture filename"
    );
    anyhow::ensure!(
        fs::metadata(path)?.len() == FIXTURE_SIZE,
        "unexpected SmolLM3 fixture size"
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
        "unexpected SmolLM3 fixture SHA-256"
    );
    Ok(())
}

fn build(files: &ModelFiles, device: &Device) -> anyhow::Result<Box<dyn CausalLm>> {
    let adapter = files.adapter()?;
    // SAFETY: the gate validates and then treats the pinned checkpoint as
    // immutable for the lifetime of the mmap-backed model.
    let weights = unsafe { files.weight_source(DType::F32, device)? };
    let model = adapter.build(weights.as_ref())?;
    drop(weights);
    Ok(model)
}

fn argmax(logits: &Tensor) -> candle_core::Result<u32> {
    logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()
}

fn max_abs_difference(left: &Tensor, right: &Tensor) -> candle_core::Result<f32> {
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
    let mut maximum = 0.0_f32;
    for (&left, &right) in left.iter().zip(&right) {
        assert!(left.is_finite() && right.is_finite());
        maximum = maximum.max((left - right).abs());
    }
    Ok(maximum)
}
