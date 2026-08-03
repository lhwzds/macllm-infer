//! Real LFM2 hybrid-runtime gates against a pinned official GGUF and llama.cpp.
//!
//! Fixture: `LiquidAI/LFM2-350M-GGUF` revision
//! `8fdc9d526b7ed346b19257551b05816c7912ecc2`, official `Q4_K_M`.
//!
//! Run with:
//! `MACLLM_LFM2_GGUF_MODEL=models/lfm2-350m-q4_k_m \
//!  cargo test --release --test lfm2_gguf_reference -- --ignored --nocapture`

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, Device};
use macllm_infer::{
    architecture::{CausalLm, SequenceMixerSpec, ShortConvolutionSpec},
    generation::{GenerationConfig, generate},
    gguf::GgufTensorType,
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use sha2::{Digest, Sha256};

const FILENAME: &str = "LFM2-350M-Q4_K_M.gguf";
const FIXTURE_SIZE: u64 = 229_309_376;
const FIXTURE_SHA256: &str = "a4d000c7064bd3b2e42c6845836286a899a4e79cf1791da1a6797b58d575957d";
const RAW_PROMPT: &str = "What is the capital of France?";
const RAW_PROMPT_IDS: &[u32] = &[1, 3493, 856, 779, 5706, 803, 4481, 540];
const CHAT_PROMPT_IDS: &[u32] = &[
    1, 6, 6423, 708, 3493, 856, 779, 5706, 803, 4481, 540, 7, 708, 6, 64_015, 708,
];
// Captured from the same pinned file with llama.cpp b10050 (b15ca938a):
// `llama-completion -m FILE -dev none -c 256 -n 12 --no-warmup
//  --simple-io --no-display-prompt --no-perf --temp 0
//  --repeat-penalty 1 --seed 42 --no-conversation -p RENDERED_WITHOUT_BOS`.
const LLAMA_CPP_CPU_CHAT_GREEDY_IDS: &[u32] = &[
    1098, 5706, 803, 4481, 856, 5242, 523, 1311, 856, 768, 2938, 4665,
];
// Replacing `-dev none` with `-ngl all` selects the same first eleven tokens,
// then Metal's quantized reductions choose `city` instead of `political`.
const LLAMA_CPP_METAL_CHAT_GREEDY_IDS: &[u32] = &[
    1098, 5706, 803, 4481, 856, 5242, 523, 1311, 856, 768, 2938, 3771,
];

static PINNED_FIXTURE_CHECK: OnceLock<(PathBuf, Result<(), String>)> = OnceLock::new();
static INFERENCE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires the pinned official LFM2-350M Q4_K_M GGUF"]
fn lfm2_metadata_tokenizer_chat_and_hybrid_schedule_are_pinned() -> anyhow::Result<()> {
    let files = fixture()?;
    assert_eq!(files.weight_file_count(), 1);
    assert_eq!(files.tensor_count(), Some(148));
    assert_eq!(
        files.gguf_tensor_type_counts(),
        [
            (GgufTensorType::F32, 55),
            (GgufTensorType::Q4K, 82),
            (GgufTensorType::Q6K, 11),
        ]
    );

    let adapter = files.adapter()?;
    let spec = adapter
        .decoder_spec()
        .expect("LFM2 uses the shared decoder");
    assert_eq!(adapter.family(), "lfm2");
    assert_eq!(spec.runtime.vocab_size, 65_536);
    assert_eq!(spec.runtime.max_context, 128_000);
    assert_eq!(spec.runtime.stop_token_ids, [7]);
    assert_eq!(spec.hidden_size, 1_024);
    assert_eq!(spec.intermediate_size, 4_608);
    assert_eq!(spec.num_layers, 16);
    assert_eq!(spec.num_query_heads, 16);
    assert_eq!(spec.num_kv_heads, 8);
    assert_eq!(spec.head_dim, 64);
    assert_eq!(spec.kv_bytes_per_token(2)?, 12_288);
    let attention_layers = spec
        .layer_specs
        .iter()
        .enumerate()
        .filter_map(|(index, layer)| layer.attention().map(|_| index))
        .collect::<Vec<_>>();
    assert_eq!(attention_layers, [2, 5, 8, 10, 12, 14]);
    for (index, layer) in spec.layer_specs.iter().enumerate() {
        if attention_layers.contains(&index) {
            assert!(layer.attention().is_some_and(|attention| attention.qk_norm));
        } else {
            assert!(matches!(
                layer.mixer,
                SequenceMixerSpec::ShortConvolution(ShortConvolutionSpec { kernel_size: 3 })
            ));
        }
    }
    assert!(spec.tied_embeddings);

    let tokenizer = files.tokenizer()?;
    let raw = tokenizer
        .encode(RAW_PROMPT, true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(raw.get_ids(), RAW_PROMPT_IDS);
    let chat = adapter.format_chat(None, RAW_PROMPT)?;
    assert_eq!(
        chat,
        "<|startoftext|><|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    );
    let chat = tokenizer
        .encode(chat, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(chat.get_ids(), CHAT_PROMPT_IDS);
    Ok(())
}

#[test]
#[ignore = "requires the pinned official LFM2-350M Q4_K_M GGUF"]
fn lfm2_cpu_matches_llama_cpp_greedy_tokens() -> anyhow::Result<()> {
    assert_greedy_golden(Device::Cpu, "CPU")
}

#[test]
#[ignore = "requires the pinned LFM2 GGUF and Apple Metal"]
fn lfm2_metal_matches_llama_cpp_greedy_tokens() -> anyhow::Result<()> {
    assert_greedy_golden(Device::new_metal(0)?, "Metal")
}

fn assert_greedy_golden(device: Device, label: &str) -> anyhow::Result<()> {
    let _guard = INFERENCE_TEST_LOCK
        .lock()
        .expect("LFM2 inference test lock poisoned");
    let files = fixture()?;
    let model = build(&files, &device)?;
    let expected = if label == "CPU" {
        LLAMA_CPP_CPU_CHAT_GREEDY_IDS
    } else {
        LLAMA_CPP_METAL_CHAT_GREEDY_IDS
    };
    let generated = generate_greedy(model.as_ref(), CHAT_PROMPT_IDS, expected.len())?;
    assert_eq!(generated, expected, "{label} llama.cpp LFM2 golden");
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
        "LFM2 fixture changed within one test process"
    );
    if let Err(error) = result {
        anyhow::bail!(error.clone());
    }
    ModelFiles::from_dir(path)
}

fn configured_fixture() -> anyhow::Result<PathBuf> {
    Ok(PathBuf::from(std::env::var("MACLLM_LFM2_GGUF_MODEL")?))
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
    anyhow::ensure!(paths.len() == 1, "expected exactly one LFM2 GGUF fixture");
    Ok(paths[0].clone())
}

fn validate_pinned_fixture(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(FILENAME),
        "unexpected LFM2 fixture filename"
    );
    anyhow::ensure!(
        fs::metadata(path)?.len() == FIXTURE_SIZE,
        "unexpected LFM2 fixture size"
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
        "unexpected LFM2 fixture SHA-256"
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
