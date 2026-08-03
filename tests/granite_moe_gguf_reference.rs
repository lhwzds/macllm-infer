//! Real Granite 3.1 sparse-MoE GGUF gates against a pinned llama.cpp golden.
//!
//! Fixture:
//! `bartowski/granite-3.1-1b-a400m-instruct-GGUF` at
//! `fc9aa102f8ef5fc48fedc232660049638cdcc9f0`, `Q4_K_M` SHA-256
//! `685976bef39437327bb608bd936ef20633ead4f9ae7b17b90da639f43d4aabcc`.
//!
//! Run with:
//! `MACLLM_GRANITE_MOE_GGUF_MODEL=/path/to/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf \
//!  cargo test --release --test granite_moe_gguf_reference -- --ignored --nocapture`

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, Device};
use macllm_infer::{
    architecture::{CausalLm, FeedForwardKind, RouterWeighting},
    generation::{FinishReason, GenerationConfig, generate},
    gguf::GgufTensorType,
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use sha2::{Digest, Sha256};

const CHAT_PROMPT: &str = "<|start_of_role|>system<|end_of_role|>You are a concise assistant.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Reply with OK.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
const CHAT_PROMPT_IDS: &[u32] = &[
    49_152, 2_946, 49_153, 4_282, 884, 312, 30_836, 47_330, 32, 0, 203, 49_152, 496, 49_153, 9_259,
    623, 9_199, 32, 0, 203, 49_152, 17_594, 49_153,
];
const LLAMA_CPP_GREEDY_IDS: &[u32] = &[2_950, 32];
const FIXTURE_SIZE: u64 = 821_847_648;
const FIXTURE_SHA256: &str = "685976bef39437327bb608bd936ef20633ead4f9ae7b17b90da639f43d4aabcc";
static PINNED_FIXTURE_CHECK: OnceLock<(PathBuf, Result<(), String>)> = OnceLock::new();
// Candle's process-global Metal resource registry is not safe to initialize
// concurrently from two tests in the same binary.
static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());
// Q4_K/Q6_K quantized matmul takes different batched and single-row paths. The
// routed top-k graph amplifies their rounding more than a dense block does, so
// lock both the worst outlier and aggregate drift rather than argmax alone.
const MAX_CACHE_FULL_DIFFERENCE: f32 = 0.75;
const MAX_CACHE_FULL_RELATIVE_RMS: f64 = 0.03;

#[test]
#[ignore = "requires the pinned Granite 3.1 Q4_K_M GGUF"]
fn granite_moe_metadata_tokenizer_and_chat_contract_are_pinned() -> anyhow::Result<()> {
    let files = fixture()?;
    assert_eq!(files.weight_file_count(), 1);
    assert_eq!(files.tensor_count(), Some(242));
    assert_eq!(
        files.gguf_tensor_type_counts(),
        [
            (GgufTensorType::F32, 73),
            (GgufTensorType::Q4K, 144),
            (GgufTensorType::Q6K, 25),
        ]
    );
    let adapter = files.adapter()?;
    let spec = adapter
        .decoder_spec()
        .expect("Granite MoE uses the shared decoder");
    assert_eq!(adapter.family(), "granitemoe");
    assert_eq!(spec.runtime.vocab_size, 49_155);
    assert_eq!(spec.runtime.max_context, 131_072);
    assert_eq!(spec.runtime.stop_token_ids, [0]);
    assert_eq!(spec.hidden_size, 1_024);
    assert_eq!(spec.intermediate_size, 512);
    assert_eq!(spec.num_layers, 24);
    assert_eq!(spec.num_query_heads, 16);
    assert_eq!(spec.num_kv_heads, 8);
    assert_eq!(spec.head_dim, 64);
    assert!((spec.embedding_scale - 12.0).abs() < f64::EPSILON);
    assert!((spec.final_logit_scale - 1.0 / 6.0).abs() < f64::EPSILON);
    assert!((spec.norm_epsilon - 1e-6).abs() < 1e-12);
    for layer in &spec.layer_specs {
        assert!(
            (layer
                .attention()
                .expect("Granite layer uses attention")
                .scale
                - 0.015_625)
                .abs()
                < f64::EPSILON
        );
        assert!((layer.mixer_residual_scale - 0.22).abs() < 1e-7);
        assert!((layer.feed_forward_residual_scale - 0.22).abs() < 1e-7);
        let FeedForwardKind::SparseMoe(moe) = layer.feed_forward else {
            panic!("Granite layer is not sparse MoE")
        };
        assert_eq!(moe.expert_count, 32);
        assert_eq!(moe.experts_per_token, 8);
        assert_eq!(moe.expert_intermediate_size, 512);
        assert_eq!(moe.router_weighting, RouterWeighting::SelectedSoftmax);
    }

    assert_eq!(documented_prompt_ids(&files)?, CHAT_PROMPT_IDS);
    Ok(())
}

#[test]
#[ignore = "requires the pinned Granite 3.1 Q4_K_M GGUF and Apple Metal"]
fn granite_moe_q4_k_m_matches_recorded_llama_cpp_greedy_sequence() -> anyhow::Result<()> {
    let _metal_guard = METAL_TEST_LOCK
        .lock()
        .expect("Granite Metal test lock poisoned");
    let files = fixture()?;
    let prompt = documented_prompt_ids(&files)?;

    let device = Device::new_metal(0)?;
    let model = build(&files, &device)?;
    let mut session = model.new_session()?;
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: 42,
        ..SamplingConfig::default()
    })?;
    let mut generated = Vec::new();
    let metrics = generate(
        model.as_ref(),
        session.as_mut(),
        &prompt,
        &mut sampler,
        GenerationConfig {
            max_new_tokens: 4,
            ignore_stop_tokens: false,
        },
        |token| {
            generated.push(token);
            Ok(())
        },
    )?;
    assert_eq!(generated, LLAMA_CPP_GREEDY_IDS);
    assert_eq!(metrics.finish_reason, FinishReason::Stop);
    Ok(())
}

#[test]
#[ignore = "requires the pinned Granite 3.1 Q4_K_M GGUF and Apple Metal"]
fn granite_moe_cached_decode_matches_fresh_full_forward() -> anyhow::Result<()> {
    let _metal_guard = METAL_TEST_LOCK
        .lock()
        .expect("Granite Metal test lock poisoned");
    let files = fixture()?;
    let prompt = documented_prompt_ids(&files)?;
    let device = Device::new_metal(0)?;
    let model = build(&files, &device)?;
    let mut cached_session = model.new_session()?;
    let prefill = model.forward_tokens(cached_session.as_mut(), &prompt, 0)?;
    assert_eq!(argmax(&prefill)?, LLAMA_CPP_GREEDY_IDS[0]);
    let cached = model
        .forward_tokens(
            cached_session.as_mut(),
            &[LLAMA_CPP_GREEDY_IDS[0]],
            prompt.len(),
        )?
        .to_device(&Device::Cpu)?;
    assert_eq!(argmax(&cached)?, LLAMA_CPP_GREEDY_IDS[1]);

    let mut combined = prompt;
    combined.push(LLAMA_CPP_GREEDY_IDS[0]);
    let mut full_session = model.new_session()?;
    let full = model
        .forward_tokens(full_session.as_mut(), &combined, 0)?
        .to_device(&Device::Cpu)?;
    assert_eq!(argmax(&full)?, LLAMA_CPP_GREEDY_IDS[1]);
    let (difference, relative_rms, diagnostics) = logit_difference_stats(&cached, &full)?;
    eprintln!("Granite cache/full diagnostics: {diagnostics}");
    assert!(
        difference <= MAX_CACHE_FULL_DIFFERENCE,
        "Granite MoE cached/full logits differed by {difference}"
    );
    assert!(
        relative_rms <= MAX_CACHE_FULL_RELATIVE_RMS,
        "Granite MoE cached/full relative RMS difference was {relative_rms}"
    );
    Ok(())
}

fn fixture() -> anyhow::Result<ModelFiles> {
    let configured = PathBuf::from(std::env::var("MACLLM_GRANITE_MOE_GGUF_MODEL")?);
    let path = single_gguf_path(&configured)?;
    let (checked_path, result) = PINNED_FIXTURE_CHECK.get_or_init(|| {
        (
            path.clone(),
            validate_pinned_fixture(&path).map_err(|error| error.to_string()),
        )
    });
    anyhow::ensure!(
        checked_path == &path,
        "Granite fixture changed within one test process"
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
        "expected exactly one Granite GGUF fixture"
    );
    Ok(paths[0].clone())
}

fn validate_pinned_fixture(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        fs::metadata(path)?.len() == FIXTURE_SIZE,
        "unexpected Granite fixture size"
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
        "unexpected Granite fixture SHA-256"
    );
    Ok(())
}

fn documented_prompt_ids(files: &ModelFiles) -> anyhow::Result<Vec<u32>> {
    let prompt = files
        .adapter()?
        .format_chat(Some("You are a concise assistant."), "Reply with OK.")?;
    assert_eq!(prompt, CHAT_PROMPT);
    let encoding = files
        .tokenizer()?
        .encode(prompt, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(encoding.get_ids(), CHAT_PROMPT_IDS);
    Ok(encoding.get_ids().to_vec())
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
    let (cached_top, cached_margin) = top_token_and_margin(&cached);
    let (full_top, full_margin) = top_token_and_margin(&full);
    let relative_rms = rms_difference / rms_scale;
    Ok((
        max_difference.1,
        relative_rms,
        format!(
            "max_abs={} at token {} (cached={}, full={}); rms_diff={rms_difference}; relative_rms={relative_rms}; cached_top={cached_top} margin={cached_margin}; full_top={full_top} margin={full_margin}",
            max_difference.1, max_difference.0, cached[max_difference.0], full[max_difference.0],
        ),
    ))
}

fn top_token_and_margin(logits: &[f32]) -> (usize, f32) {
    let mut best = (usize::MAX, f32::NEG_INFINITY);
    let mut second = f32::NEG_INFINITY;
    for (token, &logit) in logits.iter().enumerate() {
        if logit > best.1 {
            second = best.1;
            best = (token, logit);
        } else if logit > second {
            second = logit;
        }
    }
    (best.0, best.1 - second)
}

fn argmax(logits: &candle_core::Tensor) -> candle_core::Result<u32> {
    logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()
}
