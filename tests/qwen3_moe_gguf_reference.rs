//! Real `Qwen3MoE` GGUF gates against a pinned llama.cpp golden.
//!
//! Fixture: `mradermacher/loggenix-moe-0.12B-A0.08B-e5-lr5e4-b4-3060-GGUF`
//! at `6ff146973adf7da06424c7adf023bdd238760efd`, `Q8_0` SHA-256
//! `2f87bbe14ba4f02b4de2b7227777ed821fec013f64cb907b9d6e836aeda0cbf7`.
//!
//! Run with:
//! `MACLLM_QWEN3_MOE_GGUF_MODEL=/path/to/loggenix-moe.Q8_0.gguf \
//!  cargo test --release --test qwen3_moe_gguf_reference -- --ignored --nocapture`

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, Device};
use macllm_infer::{
    architecture::{CausalLm, FeedForwardKind, RouterWeighting},
    generation::{GenerationConfig, generate},
    gguf::GgufTensorType,
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use sha2::{Digest, Sha256};

const FIXTURE_SIZE: u64 = 147_811_520;
const FIXTURE_SHA256: &str = "2f87bbe14ba4f02b4de2b7227777ed821fec013f64cb907b9d6e836aeda0cbf7";
const RAW_PROMPT: &str = "2 + 2 =";
const RAW_PROMPT_IDS: &[u32] = &[17, 488, 220, 17, 284];
const CHAT_PROMPT_IDS: &[u32] = &[
    151_644, 872, 198, 17, 488, 220, 17, 284, 151_645, 198, 151_644, 77_091, 198,
];
// Captured twice with llama.cpp b10050 (b15ca938a) on CPU using a fixed raw
// prompt, greedy sampling and ignored stop tokens. The decoded bytes are
// `1 \\  \\{,2 `.
const LLAMA_CPP_GREEDY_IDS: &[u32] = &[16, 1_124, 220, 1_124, 90, 11, 17, 220];
// This Q8 fixture was exact on CPU. Metal's matrix-vector versus batched
// reductions differed by 0.00154 max / 0.000095 relative RMS, so both gates
// retain a small platform-stable envelope rather than asserting bit identity.
const MAX_CACHE_FULL_DIFFERENCE: f32 = 0.01;
const MAX_CACHE_FULL_RELATIVE_RMS: f64 = 0.001;

static PINNED_FIXTURE_CHECK: OnceLock<(PathBuf, Result<(), String>)> = OnceLock::new();
// Candle's process-global Metal registry and the same mmapped file should not
// be initialized concurrently by the two ignored inference gates.
static INFERENCE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires the pinned Qwen3MoE Q8_0 GGUF"]
fn qwen3_moe_metadata_tokenizer_and_chat_contract_are_pinned() -> anyhow::Result<()> {
    let files = fixture()?;
    assert_eq!(files.weight_file_count(), 1);
    assert_eq!(files.tensor_count(), Some(98));
    assert_eq!(
        files.gguf_tensor_type_counts(),
        [(GgufTensorType::F32, 41), (GgufTensorType::Q8_0, 57)]
    );

    let adapter = files.adapter()?;
    let spec = adapter
        .decoder_spec()
        .expect("Qwen3MoE uses the shared decoder");
    assert_eq!(adapter.family(), "qwen3moe");
    assert_eq!(spec.architecture, "qwen3moe");
    assert_eq!(spec.runtime.vocab_size, 151_936);
    assert_eq!(spec.runtime.max_context, 1_024);
    assert_eq!(spec.runtime.stop_token_ids, [151_645, 151_643]);
    assert_eq!(spec.hidden_size, 512);
    assert_eq!(spec.intermediate_size, 768);
    assert_eq!(spec.num_layers, 8);
    assert_eq!(spec.num_query_heads, 8);
    assert_eq!(spec.num_kv_heads, 2);
    assert_eq!(spec.head_dim, 64);
    assert_eq!(spec.num_query_heads * spec.head_dim, spec.hidden_size);
    assert!((spec.norm_epsilon - 1e-6).abs() < 1e-12);
    assert!(spec.tied_embeddings);
    for layer in &spec.layer_specs {
        let attention = layer.attention().expect("Qwen3MoE layer uses attention");
        assert!(attention.qk_norm);
        assert!((attention.scale - 0.125).abs() < f64::EPSILON);
        let FeedForwardKind::SparseMoe(moe) = layer.feed_forward else {
            panic!("Qwen3MoE layer is not sparse MoE")
        };
        assert_eq!(moe.expert_count, 8);
        assert_eq!(moe.experts_per_token, 2);
        assert_eq!(moe.expert_intermediate_size, 512);
        assert_eq!(moe.router_weighting, RouterWeighting::SelectedSoftmax);
    }

    let tokenizer = files.tokenizer()?;
    let raw = tokenizer
        .encode(RAW_PROMPT, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(raw.get_ids(), RAW_PROMPT_IDS);
    let rendered = adapter.format_chat(None, RAW_PROMPT)?;
    assert_eq!(
        rendered,
        "<|im_start|>user\n2 + 2 =<|im_end|>\n<|im_start|>assistant\n"
    );
    let chat = tokenizer
        .encode(rendered, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(chat.get_ids(), CHAT_PROMPT_IDS);
    Ok(())
}

#[test]
#[ignore = "requires the pinned Qwen3MoE Q8_0 GGUF"]
fn qwen3_moe_cpu_matches_llama_cpp_and_cache_full() -> anyhow::Result<()> {
    assert_backend(Device::Cpu, "CPU")
}

#[test]
#[ignore = "requires the pinned Qwen3MoE Q8_0 GGUF and Apple Metal"]
fn qwen3_moe_metal_matches_llama_cpp_and_cache_full() -> anyhow::Result<()> {
    assert_backend(Device::new_metal(0)?, "Metal")
}

fn assert_backend(device: Device, label: &str) -> anyhow::Result<()> {
    let _guard = INFERENCE_TEST_LOCK
        .lock()
        .expect("Qwen3MoE inference test lock poisoned");
    let files = fixture()?;
    let model = build(&files, &device)?;

    let mut cached_session = model.new_session()?;
    let prefill = model.forward_tokens(cached_session.as_mut(), RAW_PROMPT_IDS, 0)?;
    assert_eq!(argmax(&prefill)?, LLAMA_CPP_GREEDY_IDS[0]);
    let cached = model
        .forward_tokens(
            cached_session.as_mut(),
            &[LLAMA_CPP_GREEDY_IDS[0]],
            RAW_PROMPT_IDS.len(),
        )?
        .to_device(&Device::Cpu)?;
    assert_eq!(argmax(&cached)?, LLAMA_CPP_GREEDY_IDS[1]);

    let mut combined = RAW_PROMPT_IDS.to_vec();
    combined.push(LLAMA_CPP_GREEDY_IDS[0]);
    let mut full_session = model.new_session()?;
    let full = model
        .forward_tokens(full_session.as_mut(), &combined, 0)?
        .to_device(&Device::Cpu)?;
    assert_eq!(argmax(&full)?, LLAMA_CPP_GREEDY_IDS[1]);
    let (difference, relative_rms, diagnostics) = logit_difference_stats(&cached, &full)?;
    eprintln!("Qwen3MoE {label} cache/full diagnostics: {diagnostics}");
    assert!(
        difference <= MAX_CACHE_FULL_DIFFERENCE,
        "Qwen3MoE {label} cached/full logits differed by {difference}"
    );
    assert!(
        relative_rms <= MAX_CACHE_FULL_RELATIVE_RMS,
        "Qwen3MoE {label} cached/full relative RMS difference was {relative_rms}"
    );

    let mut session = model.new_session()?;
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: 42,
        ..SamplingConfig::default()
    })?;
    let mut generated = Vec::new();
    generate(
        model.as_ref(),
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
    assert_eq!(generated, LLAMA_CPP_GREEDY_IDS, "{label} greedy sequence");
    Ok(())
}

fn fixture() -> anyhow::Result<ModelFiles> {
    let configured = PathBuf::from(std::env::var("MACLLM_QWEN3_MOE_GGUF_MODEL")?);
    let path = single_gguf_path(&configured)?;
    let (checked_path, result) = PINNED_FIXTURE_CHECK.get_or_init(|| {
        (
            path.clone(),
            validate_pinned_fixture(&path).map_err(|error| error.to_string()),
        )
    });
    anyhow::ensure!(
        checked_path == &path,
        "Qwen3MoE fixture changed within one test process"
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
        "expected exactly one Qwen3MoE GGUF fixture"
    );
    Ok(paths[0].clone())
}

fn validate_pinned_fixture(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        fs::metadata(path)?.len() == FIXTURE_SIZE,
        "unexpected Qwen3MoE fixture size"
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
        "unexpected Qwen3MoE fixture SHA-256"
    );
    Ok(())
}

fn build(files: &ModelFiles, device: &Device) -> anyhow::Result<Box<dyn CausalLm>> {
    let adapter = files.adapter()?;
    // SAFETY: the parity gate treats the pinned checkpoint as immutable.
    let weights = unsafe { files.weight_source(DType::F32, device)? };
    let model = adapter.build(weights.as_ref())?;
    drop(weights);
    Ok(model)
}

fn logit_difference_stats(
    cached: &candle_core::Tensor,
    full: &candle_core::Tensor,
) -> candle_core::Result<(f32, f64, String)> {
    let cached = cached.flatten_all()?.to_vec1::<f32>()?;
    let full = full.flatten_all()?.to_vec1::<f32>()?;
    let mut max_difference = (0_usize, 0.0_f32);
    let mut squared_difference = 0.0_f64;
    let mut squared_scale = 0.0_f64;
    for (index, (&cached, &full)) in cached.iter().zip(&full).enumerate() {
        let difference = (cached - full).abs();
        if difference > max_difference.1 {
            max_difference = (index, difference);
        }
        squared_difference += f64::from(difference).powi(2);
        squared_scale += f64::from(cached.abs().max(full.abs())).powi(2);
    }
    let rms_difference = (squared_difference / cached.len() as f64).sqrt();
    let rms_scale = (squared_scale / cached.len() as f64).sqrt();
    let relative_rms = rms_difference / rms_scale;
    Ok((
        max_difference.1,
        relative_rms,
        format!(
            "max_abs={} at token {} (cached={}, full={}); rms_diff={rms_difference}; relative_rms={relative_rms}",
            max_difference.1, max_difference.0, cached[max_difference.0], full[max_difference.0],
        ),
    ))
}

fn argmax(logits: &candle_core::Tensor) -> candle_core::Result<u32> {
    logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()
}
