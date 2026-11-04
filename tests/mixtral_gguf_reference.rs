//! Real Mixtral-compatible GGUF gates against a pinned llama.cpp golden.
//!
//! Fixture: `tensorblock/TinyMistral-6x248M-GGUF` at
//! `901d1b8ef25b280bf91524a0a5e917b0d56d6417`, `Q4_K_M`.
//!
//! Run with:
//! `MACLLM_MIXTRAL_GGUF_MODEL=/path/to/TinyMistral-6x248M-Q4_K_M.gguf \
//!  cargo test --release --test mixtral_gguf_reference -- --ignored --nocapture`

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, Device, Tensor};
use macllm_infer::{
    architecture::{AttentionKind, CausalLm, FeedForwardKind, RouterWeighting},
    generation::{GenerationConfig, generate},
    gguf::{GgufFile, GgufTensorType},
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use sha2::{Digest, Sha256};

const FIXTURE_URL: &str = "https://huggingface.co/tensorblock/TinyMistral-6x248M-GGUF/resolve/901d1b8ef25b280bf91524a0a5e917b0d56d6417/TinyMistral-6x248M-Q4_K_M.gguf";
const FILENAME: &str = "TinyMistral-6x248M-Q4_K_M.gguf";
const FIXTURE_SIZE: u64 = 613_081_248;
const FIXTURE_SHA256: &str = "1675eb7c8dbe427264bacaf092fdc6fb83949b5ef088b4986554465e46f97d72";

const RAW_PROMPT: &str = "2 + 2 =";
const RAW_PROMPT_IDS: &[u32] = &[32_000, 28_705, 28_750, 648, 28_705, 28_750, 327];
const RAW_PROMPT_IDS_WITHOUT_BOS: &[u32] = &[28_705, 28_750, 648, 28_705, 28_750, 327];

// Captured with llama.cpp build 10050 (`b15ca938a`) on CPU. Flash attention
// is pinned on because this quantized checkpoint produces a different greedy
// sequence through build 10050's non-flash reduction path:
//
// llama-completion -m TinyMistral-6x248M-Q4_K_M.gguf \
//   --device none -ngl 0 -fa on -c 128 --no-conversation \
//   --no-warmup --simple-io -p '2 + 2 =' -n 8 --temp 0 \
//   --repeat-penalty 1 --seed 42 --ignore-eos --no-display-prompt \
//   --no-perf --log-colors off --color off --log-verbosity 1
//
// The decoded continuation is ` 1-2-3-3`.
const LLAMA_CPP_GREEDY_IDS: &[u32] = &[
    28_705, 28_740, 28_733, 28_750, 28_733, 28_770, 28_733, 28_770,
];
// The embedded tokenizer's standalone decoder removes the initial metaspace;
// llama.cpp's streamed bytes retain it as documented above.
const MACLLM_DECODED_GREEDY_TEXT: &str = "1-2-3-3";

// Quantized matrix-vector decode and matrix-matrix full-forward kernels are
// not expected to be bit-identical. These bounds are calibrated by the
// ignored release gates below; CPU retains exact cache/full argmax, while both
// backends retain exact cached decode and complete greedy-token gates.
const MAX_CPU_CACHE_FULL_DIFFERENCE: f32 = 0.2;
const MAX_CPU_CACHE_FULL_RELATIVE_RMS: f64 = 0.01;
const MAX_METAL_CACHE_FULL_DIFFERENCE: f32 = 0.02;
const MAX_METAL_CACHE_FULL_RELATIVE_RMS: f64 = 0.001;

static PINNED_FIXTURE_CHECK: OnceLock<(PathBuf, Result<(), String>)> = OnceLock::new();
// Loading two copies of this 585 MiB checkpoint concurrently is needlessly
// expensive, and Candle's Metal registry is process-global.
static INFERENCE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires the pinned TinyMistral 6x248M Q4_K_M GGUF"]
fn mixtral_metadata_tokenizer_and_raw_only_contract_are_pinned() -> anyhow::Result<()> {
    let files = fixture()?;
    let path = single_gguf_path(&configured_fixture()?)?;
    let gguf = GgufFile::open(path)?;
    assert_eq!(gguf.metadata().len(), 60);
    assert_eq!(
        gguf.metadata().string("general.architecture"),
        Some("llama")
    );
    assert_eq!(
        gguf.metadata().string("tokenizer.ggml.model"),
        Some("llama")
    );
    assert_eq!(
        gguf.metadata().string("tokenizer.ggml.pre"),
        Some("default")
    );
    assert_eq!(
        gguf.metadata().u32("tokenizer.ggml.bos_token_id"),
        Some(32_000)
    );
    assert_eq!(
        gguf.metadata().u32("tokenizer.ggml.eos_token_id"),
        Some(32_001)
    );
    assert_eq!(
        gguf.metadata().bool("tokenizer.ggml.add_bos_token"),
        Some(true)
    );
    assert!(gguf.metadata().get("tokenizer.chat_template").is_none());

    assert_eq!(files.weight_file_count(), 1);
    assert_eq!(files.tensor_count(), Some(123));
    assert_eq!(
        files.gguf_tensor_type_counts(),
        [
            (GgufTensorType::F32, 37),
            (GgufTensorType::Q4K, 73),
            (GgufTensorType::Q6K, 13),
        ]
    );

    let adapter = files.adapter()?;
    let spec = adapter
        .decoder_spec()
        .expect("Mixtral-compatible GGUF uses the shared decoder");
    assert_eq!(adapter.family(), "llama");
    assert_eq!(spec.architecture, "llama");
    assert_eq!(spec.runtime.vocab_size, 32_005);
    assert_eq!(spec.runtime.max_context, 32_768);
    assert_eq!(spec.runtime.stop_token_ids, [32_001]);
    assert_eq!(spec.hidden_size, 1_024);
    assert_eq!(spec.intermediate_size, 4_096);
    assert_eq!(spec.num_layers, 12);
    assert_eq!(spec.num_query_heads, 32);
    assert_eq!(spec.num_kv_heads, 8);
    assert_eq!(spec.head_dim, 32);
    assert_eq!(spec.num_query_heads * spec.head_dim, spec.hidden_size);
    assert!(!spec.tied_embeddings);
    for (index, layer) in spec.layer_specs.iter().enumerate() {
        let attention = layer
            .attention()
            .expect("every TinyMistral layer uses attention");
        assert_eq!(attention.kind, AttentionKind::FullCausal, "layer {index}");
        let FeedForwardKind::SparseMoe(moe) = layer.feed_forward else {
            panic!("TinyMistral layer {index} is not sparse MoE")
        };
        assert_eq!(moe.expert_count, 6, "layer {index}");
        assert_eq!(moe.experts_per_token, 2, "layer {index}");
        assert_eq!(moe.expert_intermediate_size, 4_096, "layer {index}");
        assert_eq!(
            moe.router_weighting,
            RouterWeighting::SelectedSoftmax,
            "layer {index}"
        );
    }

    let tokenizer = files.tokenizer()?;
    let with_bos = tokenizer
        .encode(RAW_PROMPT, true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(with_bos.get_ids(), RAW_PROMPT_IDS);
    let without_bos = tokenizer
        .encode(RAW_PROMPT, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(without_bos.get_ids(), RAW_PROMPT_IDS_WITHOUT_BOS);
    let decoded = tokenizer
        .decode(LLAMA_CPP_GREEDY_IDS, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(decoded, MACLLM_DECODED_GREEDY_TEXT);
    assert!(adapter.format_chat(None, RAW_PROMPT).is_err());
    Ok(())
}

#[test]
#[ignore = "requires the pinned TinyMistral 6x248M Q4_K_M GGUF"]
fn mixtral_cpu_matches_llama_cpp_and_cache_full() -> anyhow::Result<()> {
    assert_backend(Device::Cpu, "CPU")
}

#[test]
#[ignore = "requires the pinned TinyMistral GGUF and Apple Metal"]
fn mixtral_metal_matches_llama_cpp_and_cache_full() {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let device = Device::new_metal(0).expect("Metal device");
        assert_backend(device, "Metal").expect("Mixtral Metal cache parity");
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        eprintln!("skipping Mixtral-compatible Metal gate outside a Metal-enabled macOS build");
    }
}

fn assert_backend(device: Device, label: &str) -> anyhow::Result<()> {
    let _guard = INFERENCE_TEST_LOCK
        .lock()
        .expect("Mixtral-compatible inference test lock poisoned");
    let files = fixture()?;
    let model = build(&files, &device)?;

    let mut cached_session = model.new_session()?;
    let prefill = model.forward_tokens(cached_session.as_mut(), RAW_PROMPT_IDS, 0)?;
    let next_token = argmax(&prefill)?;
    assert_eq!(
        next_token, LLAMA_CPP_GREEDY_IDS[0],
        "{label} prefill golden"
    );

    let cached = model
        .forward_tokens(cached_session.as_mut(), &[next_token], RAW_PROMPT_IDS.len())?
        .to_device(&Device::Cpu)?;
    let mut combined = RAW_PROMPT_IDS.to_vec();
    combined.push(next_token);
    let mut full_session = model.new_session()?;
    let full = model
        .forward_tokens(full_session.as_mut(), &combined, 0)?
        .to_device(&Device::Cpu)?;

    let (difference, relative_rms, diagnostics) = logit_difference_stats(&cached, &full)?;
    let (max_difference, max_relative_rms) = backend_thresholds(label);
    let cached_argmax = argmax(&cached)?;
    let full_argmax = argmax(&full)?;
    eprintln!(
        "Mixtral-compatible {label} cache/full diagnostics: {diagnostics}; \
         cached_argmax={cached_argmax}; full_argmax={full_argmax}"
    );
    // CPU's Q4_K MV/MM paths select the same winner. On Metal this fixture's
    // top two logits are close enough that the measured 0.0121 MV/MM reduction
    // drift flips fresh-full argmax from 28740 to 28750. Metal still validates
    // the complete full-vocabulary logit envelope below, the cached decode
    // golden here, and the full eight-token cached greedy golden afterwards.
    if label == "CPU" {
        assert_eq!(cached_argmax, full_argmax, "CPU cache/full argmax");
    }
    assert_eq!(
        cached_argmax, LLAMA_CPP_GREEDY_IDS[1],
        "{label} decode golden"
    );
    anyhow::ensure!(
        difference <= max_difference,
        "Mixtral-compatible {label} cache/full max difference {difference} exceeds {max_difference}"
    );
    anyhow::ensure!(
        relative_rms <= max_relative_rms,
        "Mixtral-compatible {label} cache/full relative RMS {relative_rms} exceeds {max_relative_rms}"
    );

    let generated = generate_greedy(model.as_ref(), RAW_PROMPT_IDS, LLAMA_CPP_GREEDY_IDS.len())?;
    assert_eq!(
        generated, LLAMA_CPP_GREEDY_IDS,
        "{label} llama.cpp Mixtral-compatible golden"
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
        "Mixtral-compatible fixture changed within one test process"
    );
    if let Err(error) = result {
        anyhow::bail!(error.clone());
    }
    ModelFiles::from_dir(path)
}

fn configured_fixture() -> anyhow::Result<PathBuf> {
    Ok(PathBuf::from(std::env::var("MACLLM_MIXTRAL_GGUF_MODEL")?))
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
        "expected exactly one TinyMistral GGUF fixture"
    );
    Ok(paths[0].clone())
}

fn validate_pinned_fixture(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(FILENAME),
        "unexpected Mixtral-compatible fixture filename; download {FIXTURE_URL}"
    );
    anyhow::ensure!(
        fs::metadata(path)?.len() == FIXTURE_SIZE,
        "unexpected Mixtral-compatible fixture size; download {FIXTURE_URL}"
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
        "unexpected Mixtral-compatible fixture SHA-256; download {FIXTURE_URL}"
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

fn logit_difference_stats(
    cached: &Tensor,
    full: &Tensor,
) -> candle_core::Result<(f32, f64, String)> {
    let cached = cached
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let full = full
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(!cached.is_empty());
    assert_eq!(cached.len(), full.len());

    let mut max_difference = (0_usize, 0.0_f32);
    let mut squared_difference = 0.0_f64;
    let mut squared_scale = 0.0_f64;
    for (index, (&cached, &full)) in cached.iter().zip(&full).enumerate() {
        assert!(cached.is_finite() && full.is_finite());
        let difference = (cached - full).abs();
        if difference > max_difference.1 {
            max_difference = (index, difference);
        }
        squared_difference += f64::from(difference).powi(2);
        squared_scale += f64::from(cached.abs().max(full.abs())).powi(2);
    }
    let rms_difference = (squared_difference / cached.len() as f64).sqrt();
    let rms_scale = (squared_scale / cached.len() as f64).sqrt();
    let relative_rms = rms_difference / rms_scale.max(f64::EPSILON);
    Ok((
        max_difference.1,
        relative_rms,
        format!(
            "max_abs={} at token {} (cached={}, full={}); rms_diff={rms_difference}; relative_rms={relative_rms}",
            max_difference.1, max_difference.0, cached[max_difference.0], full[max_difference.0],
        ),
    ))
}
