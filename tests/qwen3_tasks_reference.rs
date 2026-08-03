//! Pinned Qwen3 task gates against Candle's independent graph and Qwen's
//! published examples.
//!
//! Run both gates with:
//! `MACLLM_QWEN3_EMBEDDING_MODEL_DIR=/path/to/embedding \
//!  MACLLM_QWEN3_RERANKER_MODEL_DIR=/path/to/reranker \
//!  cargo test --release --test qwen3_tasks_reference -- --ignored --nocapture --test-threads=1`

use std::{
    ffi::OsStr,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::Mutex,
};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3 as reference;
use macllm_infer::{
    backend::{DTypeKind, DeviceKind},
    embedding::embed_token_batch,
    models::qwen3_tasks::{Qwen3EmbeddingProfile, Qwen3RerankerProfile},
    rerank::rerank_token_batch,
    runtime::{LoadedRuntime, ModelLoadConfig},
    weights::{ModelFiles, ModelFormat},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const EMBEDDING_REPO: &str = "Qwen/Qwen3-Embedding-0.6B";
const EMBEDDING_REVISION: &str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3";
const EMBEDDING_WEIGHTS_SHA256: &str =
    "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd";
const EMBEDDING_SIDECARS: [(&str, &str); 7] = [
    (
        "config.json",
        "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
    ),
    (
        "tokenizer.json",
        "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    ),
    (
        "tokenizer_config.json",
        "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    ),
    (
        "generation_config.json",
        "28396d421a2108acce96383f6a7de78008f7f1b17f807958f3c14c51dbfb65fb",
    ),
    (
        "modules.json",
        "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    ),
    (
        "1_Pooling/config.json",
        "37bf193fa101f19101bfad9c31d3eb0f786e247b7b1e5cb7f007d730eed1ddbd",
    ),
    (
        "config_sentence_transformers.json",
        "10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961",
    ),
];

const RERANKER_REPO: &str = "Qwen/Qwen3-Reranker-0.6B";
const RERANKER_REVISION: &str = "e61197ed45024b0ed8a2d74b80b4d909f1255473";
const RERANKER_WEIGHTS_SHA256: &str =
    "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b";
const RERANKER_SIDECARS: [(&str, &str); 7] = [
    (
        "config.json",
        "d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0",
    ),
    (
        "tokenizer.json",
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    (
        "tokenizer_config.json",
        "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    ),
    (
        "generation_config.json",
        "81051cd3f6e77013827148d0b8a6ead93f8ac390d5ab805f849199f0af6a08db",
    ),
    (
        "modules.json",
        "6f13b6b4a89e577b591b2077bca40c67c26541a6740a8809267cb474f90806a9",
    ),
    (
        "1_LogitScore/config.json",
        "73e3156450564d8a98b7e47bcf5aace0f29600828b51937da545571e84db3ff3",
    ),
    (
        "config_sentence_transformers.json",
        "6a153d6696f78fd588c1c728967f0b773ea869d3c6028f151ce71ebe49140762",
    ),
];

const QUERY: &str = "What is the capital of China?";
const DOCUMENTS: [&str; 2] = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
];
const OFFICIAL_EMBEDDING_SIMILARITIES: [f32; 4] =
    [0.764_556_8, 0.141_425_09, 0.135_497_36, 0.599_954_96];
const OFFICIAL_RERANKER_MARGINS: [f32; 2] = [7.625, -11.375];

const MAX_EMBEDDING_COSINE_ERROR: f32 = 0.001;
const MAX_SIMILARITY_PARITY_ERROR: f32 = 0.005;
const MAX_OFFICIAL_SIMILARITY_ERROR: f32 = 0.01;
const MAX_RERANKER_MARGIN_PARITY_ERROR: f32 = 0.05;
// The published raw margins were captured with PyTorch/CrossEncoder. The
// pinned Metal F16 graph preserves their sign and ordering but accumulates the
// long negative example differently; same-device Candle parity stays tighter.
const MAX_OFFICIAL_RERANKER_MARGIN_ERRORS: [f32; 2] = [0.1, 0.9];
const MAX_RERANKER_PROBABILITY_ERROR: f32 = 0.000_001;

static INFERENCE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RevisionMarker {
    repo_id: String,
    revision: String,
    requested_revision: Option<String>,
    complete: bool,
}

#[test]
#[ignore = "requires the pinned Qwen3 Embedding checkpoint and Apple Metal"]
fn qwen3_embedding_matches_candle_and_official_example() -> Result<()> {
    let _guard = INFERENCE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let root = pinned_fixture(
        "MACLLM_QWEN3_EMBEDDING_MODEL_DIR",
        EMBEDDING_REPO,
        EMBEDDING_REVISION,
        EMBEDDING_WEIGHTS_SHA256,
        &EMBEDDING_SIDECARS,
    )?;
    let files = ModelFiles::from_dir(&root)?;
    let profile = Qwen3EmbeddingProfile::from_model_dir(&root)?;
    ensure!(profile.dimension() == 1_024);
    ensure!(profile.max_tokens() == 8_192);
    let tokenizer = files.tokenizer()?;
    profile.validate_tokenizer(&tokenizer)?;

    let queries = [QUERY, "Explain gravity"];
    let mut token_batches = queries
        .iter()
        .map(|query| profile.tokenize_query(&tokenizer, query, None))
        .collect::<Result<Vec<_>>>()?;
    token_batches.extend(
        DOCUMENTS
            .iter()
            .map(|document| profile.tokenize_document(&tokenizer, document))
            .collect::<Result<Vec<_>>>()?,
    );
    drop(tokenizer);
    drop(files);

    let actual = {
        let runtime = load_macllm(&root)?;
        let rows = embed_token_batch(runtime.model.as_ref(), &token_batches)?;
        runtime.model.synchronize()?;
        rows
    };
    let expected = candle_embeddings(&root, &token_batches)?;

    ensure!(actual.len() == 4 && expected.len() == 4);
    for (index, (actual_row, expected_row)) in actual.iter().zip(&expected).enumerate() {
        ensure!(actual_row.len() == profile.dimension());
        assert_near(
            &format!("embedding lane {index} norm"),
            dot(actual_row, actual_row)?.sqrt(),
            1.0,
            0.000_01,
        )?;
        assert_near(
            &format!("embedding lane {index} Candle cosine"),
            dot(actual_row, expected_row)?,
            1.0,
            MAX_EMBEDDING_COSINE_ERROR,
        )?;
    }

    let actual_similarities = embedding_similarity_matrix(&actual)?;
    let candle_similarities = embedding_similarity_matrix(&expected)?;
    for index in 0..4 {
        assert_near(
            &format!("embedding similarity {index} Candle parity"),
            actual_similarities[index],
            candle_similarities[index],
            MAX_SIMILARITY_PARITY_ERROR,
        )?;
        assert_near(
            &format!("embedding similarity {index} official golden"),
            actual_similarities[index],
            OFFICIAL_EMBEDDING_SIMILARITIES[index],
            MAX_OFFICIAL_SIMILARITY_ERROR,
        )?;
    }
    ensure!(actual_similarities[0] > actual_similarities[1]);
    ensure!(actual_similarities[3] > actual_similarities[2]);
    eprintln!("Qwen3 embedding similarities: {actual_similarities:?}");
    Ok(())
}

#[test]
#[ignore = "requires the pinned Qwen3 Reranker checkpoint and Apple Metal"]
fn qwen3_reranker_matches_candle_and_official_example() -> Result<()> {
    let _guard = INFERENCE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let root = pinned_fixture(
        "MACLLM_QWEN3_RERANKER_MODEL_DIR",
        RERANKER_REPO,
        RERANKER_REVISION,
        RERANKER_WEIGHTS_SHA256,
        &RERANKER_SIDECARS,
    )?;
    let files = ModelFiles::from_dir(&root)?;
    let profile = Qwen3RerankerProfile::from_model_dir(&root)?;
    ensure!(profile.no_token_id() == 2_152);
    ensure!(profile.yes_token_id() == 9_693);
    ensure!(profile.max_tokens() == 8_192);
    let tokenizer = files.tokenizer()?;
    profile.validate_tokenizer(&tokenizer)?;
    let token_batches = DOCUMENTS
        .iter()
        .map(|document| profile.tokenize_pair(&tokenizer, QUERY, document, None))
        .collect::<Result<Vec<_>>>()?;
    drop(tokenizer);
    drop(files);

    let actual = {
        let runtime = load_macllm(&root)?;
        let scores = rerank_token_batch(
            runtime.model.as_ref(),
            &token_batches,
            profile.no_token_id(),
            profile.yes_token_id(),
        )?;
        runtime.model.synchronize()?;
        scores
    };
    let candle = candle_reranker_margins(
        &root,
        &token_batches,
        profile.no_token_id(),
        profile.yes_token_id(),
    )?;

    ensure!(actual.len() == 2 && candle.len() == 2);
    eprintln!(
        "Qwen3 reranker margins: MacLLM=[{}, {}], Candle={candle:?}",
        actual[0].margin, actual[1].margin
    );
    for index in 0..2 {
        assert_near(
            &format!("reranker margin {index} Candle parity"),
            actual[index].margin,
            candle[index],
            MAX_RERANKER_MARGIN_PARITY_ERROR,
        )?;
        assert_near(
            &format!("reranker margin {index} official golden"),
            actual[index].margin,
            OFFICIAL_RERANKER_MARGINS[index],
            MAX_OFFICIAL_RERANKER_MARGIN_ERRORS[index],
        )?;
        assert_near(
            &format!("reranker probability {index} matches its margin"),
            actual[index].yes_probability,
            stable_sigmoid(actual[index].margin),
            MAX_RERANKER_PROBABILITY_ERROR,
        )?;
    }
    ensure!(actual[0].margin > 0.0 && actual[1].margin < 0.0);
    ensure!(actual[0].margin > actual[1].margin);
    Ok(())
}

fn load_macllm(root: &Path) -> Result<LoadedRuntime> {
    let mut config = ModelLoadConfig::new(root, DeviceKind::Metal, Some(DTypeKind::F16));
    config.kv_cache_capacity_tokens = Some(8_192);
    LoadedRuntime::load(&config)
}

fn candle_embeddings(root: &Path, token_batches: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
    let files = ModelFiles::from_dir(root)?;
    let config: reference::Config = serde_json::from_slice(&fs::read(files.config_path()?)?)?;
    let device = Device::new_metal(0)?;
    // SAFETY: the pinned fixture is hashed before use and remains immutable for this test.
    let weights = unsafe {
        VarBuilder::from_mmaped_safetensors(files.safetensor_weights()?, DType::F16, &device)?
    };
    ensure!(weights.contains_tensor("embed_tokens.weight"));
    ensure!(!weights.contains_tensor("model.embed_tokens.weight"));
    let weights = weights.rename_f(|name| name.strip_prefix("model.").unwrap_or(name).to_owned());
    ensure!(weights.contains_tensor("model.embed_tokens.weight"));
    let pristine = reference::Model::new(&config, weights)?;
    let mut rows = Vec::with_capacity(token_batches.len());
    for tokens in token_batches {
        ensure!(!tokens.is_empty());
        let input = Tensor::from_slice(tokens, (1, tokens.len()), &device)?;
        let mut lane = pristine.clone();
        let mut row = lane
            .forward(&input, 0)?
            .narrow(1, tokens.len() - 1, 1)?
            .squeeze(1)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        normalize(&mut row)?;
        rows.push(row);
    }
    device.synchronize()?;
    Ok(rows)
}

fn candle_reranker_margins(
    root: &Path,
    token_batches: &[Vec<u32>],
    no_token_id: u32,
    yes_token_id: u32,
) -> Result<Vec<f32>> {
    let files = ModelFiles::from_dir(root)?;
    let config: reference::Config = serde_json::from_slice(&fs::read(files.config_path()?)?)?;
    let device = Device::new_metal(0)?;
    // SAFETY: the pinned fixture is hashed before use and remains immutable for this test.
    let weights = unsafe {
        VarBuilder::from_mmaped_safetensors(files.safetensor_weights()?, DType::F16, &device)?
    };
    let pristine = reference::ModelForCausalLM::new(&config, weights)?;
    let labels = Tensor::new(&[no_token_id, yes_token_id], &device)?;
    let mut margins = Vec::with_capacity(token_batches.len());
    for tokens in token_batches {
        ensure!(!tokens.is_empty());
        let input = Tensor::from_slice(tokens, (1, tokens.len()), &device)?;
        let mut lane = pristine.clone();
        let row = lane
            .forward(&input, 0)?
            .squeeze(1)?
            .index_select(&labels, 1)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?
            .remove(0);
        ensure!(row.len() == 2);
        margins.push(row[1] - row[0]);
    }
    device.synchronize()?;
    Ok(margins)
}

fn pinned_fixture(
    environment: &str,
    repo: &str,
    revision: &str,
    weights_sha256: &str,
    sidecars: &[(&str, &str)],
) -> Result<PathBuf> {
    let root = PathBuf::from(
        std::env::var_os(environment).with_context(|| format!("{environment} is not set"))?,
    );
    ensure!(root.is_dir(), "{} is not a model directory", root.display());

    let marker_path = root.join(".macllm-revision");
    ensure_regular_file(&marker_path)?;
    let marker: RevisionMarker = serde_json::from_slice(&fs::read(&marker_path)?)?;
    ensure!(
        marker.repo_id == repo,
        "snapshot repository identity mismatch"
    );
    ensure!(marker.revision == revision, "snapshot revision mismatch");
    ensure!(
        marker.requested_revision.as_deref() == Some(revision),
        "snapshot requested revision mismatch"
    );
    ensure!(marker.complete, "snapshot marker is incomplete");

    let files = ModelFiles::from_dir(&root)?;
    ensure!(files.format() == ModelFormat::Safetensors);
    let weights = files.safetensor_weights()?;
    ensure!(weights.len() == 1);
    ensure!(weights[0].file_name() == Some(OsStr::new("model.safetensors")));
    ensure_regular_file(&weights[0])?;
    ensure!(
        sha256(&weights[0])? == weights_sha256,
        "pinned model.safetensors SHA-256 mismatch"
    );
    for &(relative, expected_sha256) in sidecars {
        let path = root.join(relative);
        ensure_regular_file(&path)?;
        ensure!(
            sha256(&path)? == expected_sha256,
            "pinned sidecar SHA-256 mismatch: {relative}"
        );
    }
    Ok(root)
}

fn ensure_regular_file(path: &Path) -> Result<()> {
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_file(),
        "fixture file must be a regular non-symlink: {}",
        path.display()
    );
    Ok(())
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn normalize(row: &mut [f32]) -> Result<()> {
    let norm = row
        .iter()
        .try_fold(0.0_f64, |sum, &value| {
            ensure!(value.is_finite());
            Ok::<_, anyhow::Error>(sum + f64::from(value).powi(2))
        })?
        .sqrt();
    ensure!(norm.is_finite() && norm > 0.0);
    for value in row {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> Result<f32> {
    ensure!(left.len() == right.len());
    Ok(left
        .iter()
        .zip(right)
        .map(|(&left, &right)| f64::from(left) * f64::from(right))
        .sum::<f64>() as f32)
}

fn embedding_similarity_matrix(rows: &[Vec<f32>]) -> Result<[f32; 4]> {
    ensure!(rows.len() == 4);
    Ok([
        dot(&rows[0], &rows[2])?,
        dot(&rows[0], &rows[3])?,
        dot(&rows[1], &rows[2])?,
        dot(&rows[1], &rows[3])?,
    ])
}

fn stable_sigmoid(value: f32) -> f32 {
    let value = f64::from(value);
    if value >= 0.0 {
        (1.0 / (1.0 + (-value).exp())) as f32
    } else {
        let exponent = value.exp();
        (exponent / (1.0 + exponent)) as f32
    }
}

fn assert_near(label: &str, actual: f32, expected: f32, tolerance: f32) -> Result<()> {
    ensure!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "{label}: actual={actual}, expected={expected}, tolerance={tolerance}"
    );
    Ok(())
}
