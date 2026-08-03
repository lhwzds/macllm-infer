//! Real `Q2_K/Q3_K` GGUF gates against pinned llama.cpp greedy sequences.
//!
//! Fixtures come from `tensorblock/TinyLlama-1.1B-Chat-v1.0-GGUF` at
//! `917a6091308fa80582a38686dea7c995c1932078`.
//!
//! Run with:
//! `MACLLM_TINYLLAMA_Q2_GGUF_MODEL=models/tinyllama-1.1b-chat-q2_k \
//!  MACLLM_TINYLLAMA_Q3_GGUF_MODEL=models/tinyllama-1.1b-chat-q3_k_m \
//!  cargo test --release --test tinyllama_lower_bit_gguf_reference -- --ignored --nocapture`

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, Device, Tensor};
use macllm_infer::{
    architecture::CausalLm,
    generation::{GenerationConfig, generate},
    gguf::GgufTensorType,
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use sha2::{Digest, Sha256};

const REPOSITORY: &str = "tensorblock/TinyLlama-1.1B-Chat-v1.0-GGUF";
const REVISION: &str = "917a6091308fa80582a38686dea7c995c1932078";
const PROMPT: &str = "2 + 2 =";
const PROMPT_IDS_WITH_BOS: &[u32] = &[1, 29_871, 29_906, 718, 29_871, 29_906, 353];

struct FixtureSpec {
    environment: &'static str,
    filename: &'static str,
    size: u64,
    sha256: &'static str,
    tensor_types: &'static [(GgufTensorType, usize)],
    llama_cpp_greedy_ids: &'static [u32],
    max_cache_full_difference: f32,
    max_cache_full_relative_rms: f64,
}

const Q2_SPEC: FixtureSpec = FixtureSpec {
    environment: "MACLLM_TINYLLAMA_Q2_GGUF_MODEL",
    filename: "TinyLlama-1.1B-Chat-v1.0-Q2_K.gguf",
    size: 432_131_616,
    sha256: "fcc8b527716d3493b622a09f699a8cd0174d92a6ca402f1ff6998b49a5ee7200",
    tensor_types: &[
        (GgufTensorType::F32, 45),
        (GgufTensorType::Q2K, 89),
        (GgufTensorType::Q3K, 44),
        (GgufTensorType::Q4K, 22),
        (GgufTensorType::Q6K, 1),
    ],
    llama_cpp_greedy_ids: &[29_871, 29_946, 13, 13],
    max_cache_full_difference: 0.05,
    max_cache_full_relative_rms: 0.003,
};

const Q3_SPEC: FixtureSpec = FixtureSpec {
    environment: "MACLLM_TINYLLAMA_Q3_GGUF_MODEL",
    filename: "TinyLlama-1.1B-Chat-v1.0-Q3_K_M.gguf",
    size: 548_404_768,
    sha256: "a7f5d8894019a758fdd516d4847a4a1fec5b18321d367b593d6d996a115b7bd7",
    tensor_types: &[
        (GgufTensorType::F32, 45),
        (GgufTensorType::Q3K, 89),
        (GgufTensorType::Q4K, 63),
        (GgufTensorType::Q5K, 3),
        (GgufTensorType::Q6K, 1),
    ],
    llama_cpp_greedy_ids: &[29_871, 29_946, 13, 29_899],
    max_cache_full_difference: 0.15,
    max_cache_full_relative_rms: 0.01,
};

static Q2_FIXTURE_CHECK: OnceLock<(PathBuf, Result<(), String>)> = OnceLock::new();
static Q3_FIXTURE_CHECK: OnceLock<(PathBuf, Result<(), String>)> = OnceLock::new();
// Candle's process-global Metal resource registry is not safe to initialize
// concurrently from two tests in the same binary.
static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires the pinned TinyLlama Q2_K and Q3_K_M GGUF fixtures"]
fn tinyllama_lower_bit_metadata_and_tokenizer_contracts_are_pinned() -> anyhow::Result<()> {
    for (spec, check) in [(&Q2_SPEC, &Q2_FIXTURE_CHECK), (&Q3_SPEC, &Q3_FIXTURE_CHECK)] {
        let files = fixture(spec, check)?;
        assert_eq!(files.weight_file_count(), 1);
        assert_eq!(files.tensor_count(), Some(201));
        assert_eq!(files.gguf_tensor_type_counts(), spec.tensor_types);

        let adapter = files.adapter()?;
        let model = adapter
            .decoder_spec()
            .expect("TinyLlama uses the shared decoder");
        assert_eq!(adapter.family(), "llama");
        assert_eq!(model.runtime.vocab_size, 32_000);
        assert_eq!(model.runtime.max_context, 2_048);
        assert_eq!(model.runtime.stop_token_ids, [2]);
        assert_eq!(model.hidden_size, 2_048);
        assert_eq!(model.intermediate_size, 5_632);
        assert_eq!(model.num_layers, 22);
        assert_eq!(model.num_query_heads, 32);
        assert_eq!(model.num_kv_heads, 4);
        assert_eq!(model.head_dim, 64);

        let prompt = prompt_ids(&files)?;
        assert_eq!(prompt, PROMPT_IDS_WITH_BOS);
    }
    Ok(())
}

#[test]
#[ignore = "requires the pinned TinyLlama Q2_K GGUF and Apple Metal"]
fn tinyllama_q2_k_matches_llama_cpp_and_cache_full() -> anyhow::Result<()> {
    assert_greedy_and_cache_full(&Q2_SPEC, &Q2_FIXTURE_CHECK)
}

#[test]
#[ignore = "requires the pinned TinyLlama Q3_K_M GGUF and Apple Metal"]
fn tinyllama_q3_k_m_matches_llama_cpp_and_cache_full() -> anyhow::Result<()> {
    assert_greedy_and_cache_full(&Q3_SPEC, &Q3_FIXTURE_CHECK)
}

fn assert_greedy_and_cache_full(
    spec: &FixtureSpec,
    check: &'static OnceLock<(PathBuf, Result<(), String>)>,
) -> anyhow::Result<()> {
    let _metal_guard = METAL_TEST_LOCK
        .lock()
        .expect("TinyLlama Metal test lock poisoned");
    let files = fixture(spec, check)?;
    let prompt = prompt_ids(&files)?;
    let device = Device::new_metal(0)?;
    let model = build(&files, &device)?;
    let mut generation_session = model.new_session()?;
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: 42,
        ..SamplingConfig::default()
    })?;
    let mut generated = Vec::new();
    generate(
        model.as_ref(),
        generation_session.as_mut(),
        &prompt,
        &mut sampler,
        GenerationConfig {
            max_new_tokens: spec.llama_cpp_greedy_ids.len(),
            ignore_stop_tokens: true,
        },
        |token| {
            generated.push(token);
            Ok(())
        },
    )?;
    assert_eq!(generated, spec.llama_cpp_greedy_ids);

    let mut cached_session = model.new_session()?;
    let prefill = model.forward_tokens(cached_session.as_mut(), &prompt, 0)?;
    assert_eq!(argmax(&prefill)?, spec.llama_cpp_greedy_ids[0]);
    let cached = model
        .forward_tokens(
            cached_session.as_mut(),
            &[spec.llama_cpp_greedy_ids[0]],
            prompt.len(),
        )?
        .to_device(&Device::Cpu)?;
    assert_eq!(argmax(&cached)?, spec.llama_cpp_greedy_ids[1]);

    let mut combined = prompt;
    combined.push(spec.llama_cpp_greedy_ids[0]);
    let mut full_session = model.new_session()?;
    let full = model
        .forward_tokens(full_session.as_mut(), &combined, 0)?
        .to_device(&Device::Cpu)?;
    assert_eq!(argmax(&full)?, spec.llama_cpp_greedy_ids[1]);
    let (difference, relative_rms) = logit_difference_stats(&cached, &full)?;
    eprintln!(
        "{} cache/full max abs difference: {difference}; relative RMS: {relative_rms}",
        spec.filename
    );
    assert!(
        difference <= spec.max_cache_full_difference,
        "{} cache/full max difference {difference} exceeds {}",
        spec.filename,
        spec.max_cache_full_difference
    );
    assert!(
        relative_rms <= spec.max_cache_full_relative_rms,
        "{} cache/full relative RMS {relative_rms} exceeds {}",
        spec.filename,
        spec.max_cache_full_relative_rms
    );
    Ok(())
}

fn fixture(
    spec: &FixtureSpec,
    check: &'static OnceLock<(PathBuf, Result<(), String>)>,
) -> anyhow::Result<ModelFiles> {
    let configured = PathBuf::from(std::env::var(spec.environment)?);
    validate_revision_marker(&configured)?;
    let path = single_gguf_path(&configured)?;
    let (checked_path, result) = check.get_or_init(|| {
        (
            path.clone(),
            validate_pinned_fixture(&path, spec).map_err(|error| error.to_string()),
        )
    });
    anyhow::ensure!(
        checked_path == &path,
        "{} fixture changed within one test process",
        spec.filename
    );
    if let Err(error) = result {
        anyhow::bail!(error.clone());
    }
    ModelFiles::from_dir(path)
}

fn validate_revision_marker(configured: &Path) -> anyhow::Result<()> {
    if configured.is_file() {
        return Ok(());
    }
    let marker: serde_json::Value =
        serde_json::from_slice(&fs::read(configured.join(".macllm-revision"))?)?;
    anyhow::ensure!(
        marker["repo_id"] == REPOSITORY,
        "unexpected fixture repository"
    );
    anyhow::ensure!(
        marker["revision"] == REVISION,
        "unexpected fixture revision"
    );
    anyhow::ensure!(marker["complete"] == true, "fixture download is incomplete");
    Ok(())
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
        "expected exactly one TinyLlama GGUF fixture"
    );
    Ok(paths[0].clone())
}

fn validate_pinned_fixture(path: &Path, spec: &FixtureSpec) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(spec.filename),
        "unexpected TinyLlama fixture filename"
    );
    anyhow::ensure!(
        fs::metadata(path)?.len() == spec.size,
        "unexpected {} size",
        spec.filename
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
        format!("{:x}", hasher.finalize()) == spec.sha256,
        "unexpected {} SHA-256",
        spec.filename
    );
    Ok(())
}

fn prompt_ids(files: &ModelFiles) -> anyhow::Result<Vec<u32>> {
    let prompt = files
        .tokenizer()?
        .encode(PROMPT, true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(prompt.get_ids().to_vec())
}

fn build(files: &ModelFiles, device: &Device) -> anyhow::Result<Box<dyn CausalLm>> {
    let adapter = files.adapter()?;
    // SAFETY: the parity gate treats the pinned checkpoint as immutable.
    let weights = unsafe { files.weight_source(DType::F32, device)? };
    let model = adapter.build(weights.as_ref())?;
    drop(weights);
    Ok(model)
}

fn argmax(logits: &Tensor) -> candle_core::Result<u32> {
    logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()
}

fn logit_difference_stats(cached: &Tensor, full: &Tensor) -> candle_core::Result<(f32, f64)> {
    let cached = cached.flatten_all()?.to_vec1::<f32>()?;
    let full = full.flatten_all()?.to_vec1::<f32>()?;
    assert!(!cached.is_empty());
    assert_eq!(cached.len(), full.len());
    let mut max_difference = 0.0_f32;
    let mut squared_difference = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    for (&cached, &full) in cached.iter().zip(&full) {
        assert!(cached.is_finite() && full.is_finite());
        let difference = (cached - full).abs();
        max_difference = max_difference.max(difference);
        squared_difference += f64::from(difference).powi(2);
        squared_reference += f64::from(full).powi(2);
    }
    Ok((
        max_difference,
        (squared_difference / squared_reference.max(f64::EPSILON)).sqrt(),
    ))
}
