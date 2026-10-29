//! Real Qwen3-4B GGUF gates against a pinned official fixture and llama.cpp.
//!
//! Fixture: `Qwen/Qwen3-4B-GGUF` revision
//! `bc640142c66e1fdd12af0bd68f40445458f3869b`, official `Q4_K_M`, fetched with
//! `macllm download Qwen/Qwen3-4B-GGUF --revision bc640142c66e1fdd12af0bd68f40445458f3869b
//!  --gguf Qwen3-4B-Q4_K_M.gguf -o models/qwen3-4b-q4_k_m`.
//!
//! Run with the repository-local fixture:
//! `cargo test --release --test qwen3_4b_gguf_reference -- --ignored --nocapture --test-threads=1`
//! or override it with `MACLLM_QWEN3_4B_GGUF_MODEL`.

use std::{
    fs::{self, File},
    io::Read,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, Device, Tensor};
use macllm_infer::{
    agent::AgentTemplateProfile,
    architecture::{CausalLm, PositionEncodingSpec, RopeLayout},
    generation::{GenerationConfig, generate, generate_with_prefill_token_quantum},
    gguf::{GgufFile, GgufTensorType},
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use sha2::{Digest, Sha256};

const FILENAME: &str = "Qwen3-4B-Q4_K_M.gguf";
const FIXTURE_SIZE: u64 = 2_497_280_256;
const FIXTURE_SHA256: &str = "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5";
// The embedded `tokenizer.chat_template` is the GGUF-generation Qwen3 Jinja
// program that MacLLM already recognizes as `Qwen3Gguf`. It renders the same
// bytes as the newer Qwen/Qwen3-4B tokenizer_config.json template (4,168 bytes,
// SHA-256 a55ee1b1...) for this text-only prompt.
const CHAT_TEMPLATE_LEN: usize = 4_100;
const CHAT_TEMPLATE_SHA256: &str =
    "57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361";
const RAW_PROMPT: &str = "2 + 2 =";
const RAW_PROMPT_IDS: &[u32] = &[17, 488, 220, 17, 284];
const UNICODE_TEXT: &str = "café 👋 1234567";
const UNICODE_IDS: &[u32] = &[924, 58_858, 61_804, 233, 220, 16, 17, 18, 19, 20, 21, 22];
// MacLLM renders the official template's default generation prompt. The
// official template appends this empty reasoning block when it is rendered
// with `enable_thinking=false`; llama-server `/apply-template` and Jinja2 both
// produced exactly `CHAT_PROMPT` from the embedded template.
const DEFAULT_CHAT_PROMPT: &str = "<|im_start|>user\n2 + 2 =<|im_end|>\n<|im_start|>assistant\n";
const NO_THINKING_SUFFIX: &str = "<think>\n\n</think>\n\n";
const CHAT_PROMPT: &str =
    "<|im_start|>user\n2 + 2 =<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const CHAT_PROMPT_IDS: &[u32] = &[
    151_644, 872, 198, 17, 488, 220, 17, 284, 151_645, 198, 151_644, 77_091, 198, 151_667, 271,
    151_668, 271,
];

// Captured twice on both CPU and Metal with Homebrew llama.cpp 0.7.0
// (build b11669, commit 620d2225c) from the pinned file, sending the token IDs
// above to llama-server `/completion` with `n_predict: 8, temperature: 0,
// repeat_penalty: 1, seed: 42, cache_prompt: false, return_tokens: true`.
// CPU: `llama-server -m FILE -dev none -ngl 0 -c 4096 -np 1 --jinja --seed 42`;
// Metal: replace `-dev none -ngl 0` with `-ngl 99`. `llama-completion -m FILE
// -no-cnv -dev none -p '2 + 2 =' -n 8 --temp 0 --repeat-penalty 1 --seed 42
// --no-display-prompt --no-perf --simple-io -lv 2` prints the same text.
// The raw continuation decodes to ` 4, 2 + 2`. Its fourth token is a narrow
// greedy choice (llama.cpp log-probability margin 0.11 on CPU, 0.19 on Metal).
const LLAMA_CPP_GREEDY_IDS: &[u32] = &[220, 19, 11, 220, 17, 488, 220, 17];
// Thinking disabled, the model restates the expression and answers `2 + 2 = 4.`.
const CHAT_LLAMA_CPP_GREEDY_IDS: &[u32] = &[17, 488, 220, 17, 284, 220, 19, 13];

// Quantized matrix-vector and matrix-matrix reductions can accumulate in a
// different order. Exact next-token identity remains mandatory while this
// numeric envelope catches material cache/full regressions. On an M4 the
// differences were 0.3076 (CPU) and 0.0416 (Metal), bit-identical across
// three runs; the bounds leave roughly 3x headroom.
const MAX_CPU_CACHE_FULL_DIFFERENCE: f32 = 1.0;
const MAX_METAL_CACHE_FULL_DIFFERENCE: f32 = 0.15;

static PINNED_FIXTURE_CHECK: OnceLock<(PathBuf, Result<(), String>)> = OnceLock::new();
// Two copies of this 2.3 GiB checkpoint should not be resident at once, and
// Candle's Metal registry is process-global.
static INFERENCE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires the pinned official Qwen3-4B Q4_K_M GGUF"]
fn qwen3_4b_metadata_tensors_tokenizer_and_chat_are_pinned() -> anyhow::Result<()> {
    let files = fixture()?;
    assert_eq!(files.weight_file_count(), 1);
    assert_eq!(files.tensor_count(), Some(398));
    assert_eq!(
        files.gguf_tensor_type_counts(),
        [
            (GgufTensorType::F32, 145),
            (GgufTensorType::Q4K, 216),
            (GgufTensorType::Q6K, 37),
        ]
    );

    let gguf = GgufFile::open(single_gguf_path(&configured_fixture())?)?;
    let metadata = gguf.metadata();
    assert_eq!(gguf.version(), 3);
    assert_eq!(metadata.len(), 28);
    assert_eq!(metadata.string("general.architecture"), Some("qwen3"));
    // llama.cpp LLAMA_FTYPE_MOSTLY_Q4_K_M.
    assert_eq!(metadata.u32("general.file_type"), Some(15));
    assert_eq!(metadata.string("tokenizer.ggml.model"), Some("gpt2"));
    assert_eq!(metadata.string("tokenizer.ggml.pre"), Some("qwen2"));
    assert_eq!(metadata.u32("tokenizer.ggml.bos_token_id"), Some(151_643));
    assert_eq!(metadata.u32("tokenizer.ggml.eos_token_id"), Some(151_645));
    assert_eq!(
        metadata.u32("tokenizer.ggml.padding_token_id"),
        Some(151_643)
    );
    assert_eq!(metadata.bool("tokenizer.ggml.add_bos_token"), Some(false));
    let template = metadata
        .string("tokenizer.chat_template")
        .expect("the official GGUF embeds its chat template");
    assert_eq!(template.len(), CHAT_TEMPLATE_LEN);
    assert_eq!(
        format!("{:x}", Sha256::digest(template.as_bytes())),
        CHAT_TEMPLATE_SHA256
    );
    assert_tensor_inventory(&gguf);

    let adapter = files.adapter()?;
    let spec = adapter
        .decoder_spec()
        .expect("Qwen3 uses the shared decoder");
    assert_eq!(adapter.family(), "qwen3");
    assert_eq!(
        adapter.agent_template_profile(),
        Some(AgentTemplateProfile::Qwen3Gguf)
    );
    assert_eq!(spec.runtime.vocab_size, 151_936);
    assert_eq!(spec.runtime.max_context, 40_960);
    assert_eq!(spec.runtime.stop_token_ids, [151_645, 151_643]);
    assert_eq!(spec.hidden_size, 2_560);
    assert_eq!(spec.intermediate_size, 9_728);
    assert_eq!(spec.num_layers, 36);
    assert_eq!(spec.num_query_heads, 32);
    assert_eq!(spec.num_kv_heads, 8);
    assert_eq!(spec.head_dim, 128);
    assert_eq!(spec.num_query_heads * spec.head_dim, 4_096);
    assert_ne!(spec.num_query_heads * spec.head_dim, spec.hidden_size);
    assert!((spec.norm_epsilon - 1e-6).abs() < 1e-12);
    assert!(spec.tied_embeddings);
    assert_eq!(spec.layer_specs.len(), 36);
    for (index, layer) in spec.layer_specs.iter().enumerate() {
        let attention = layer
            .attention()
            .expect("every Qwen3 decoder layer uses attention");
        assert!(attention.qk_norm, "layer {index} must use QK norm");
        let PositionEncodingSpec::Rope(rope) = &attention.position_encoding else {
            panic!("layer {index} must use RoPE");
        };
        assert_eq!(rope.layout, RopeLayout::HalfSplit);
        assert_eq!(rope.rotary_dim, 128);
        assert!((rope.theta - 1_000_000.0).abs() < f64::EPSILON);
        assert!(rope.frequency_factors.is_none());
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

    let default_chat = adapter.format_chat(None, RAW_PROMPT)?;
    assert_eq!(default_chat, DEFAULT_CHAT_PROMPT);
    let chat = format!("{default_chat}{NO_THINKING_SUFFIX}");
    assert_eq!(chat, CHAT_PROMPT);
    let chat = tokenizer
        .encode(chat, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(chat.get_ids(), CHAT_PROMPT_IDS);
    Ok(())
}

/// Pin every tensor's name, Candle-order shape, and quantization. `Q4_K_M`
/// promotes `attn_v` and `ffn_down` to `Q6_K` with llama.cpp's
/// `use_more_bits` schedule; the output head is tied to `token_embd`.
fn assert_tensor_inventory(gguf: &GgufFile) {
    let tensors = gguf.tensors();
    assert_eq!(tensors.len(), 398);
    let expect = |name: &str, shape: &[usize], dtype: GgufTensorType| {
        let tensor = tensors
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"));
        assert_eq!(tensor.shape, shape, "{name} shape");
        assert_eq!(tensor.dtype, dtype, "{name} type");
    };
    expect("token_embd.weight", &[151_936, 2_560], GgufTensorType::Q6K);
    expect("output_norm.weight", &[2_560], GgufTensorType::F32);
    assert!(gguf.tensor("output.weight").is_none());

    let layers = 36;
    let mut expected = 2;
    for layer in 0..layers {
        let more_bits =
            layer < layers / 8 || layer >= 7 * layers / 8 || (layer - layers / 8) % 3 == 2;
        let promoted = if more_bits {
            GgufTensorType::Q6K
        } else {
            GgufTensorType::Q4K
        };
        let prefix = format!("blk.{layer}");
        expect(
            &format!("{prefix}.attn_norm.weight"),
            &[2_560],
            GgufTensorType::F32,
        );
        expect(
            &format!("{prefix}.attn_q.weight"),
            &[4_096, 2_560],
            GgufTensorType::Q4K,
        );
        expect(
            &format!("{prefix}.attn_k.weight"),
            &[1_024, 2_560],
            GgufTensorType::Q4K,
        );
        expect(
            &format!("{prefix}.attn_v.weight"),
            &[1_024, 2_560],
            promoted,
        );
        expect(
            &format!("{prefix}.attn_q_norm.weight"),
            &[128],
            GgufTensorType::F32,
        );
        expect(
            &format!("{prefix}.attn_k_norm.weight"),
            &[128],
            GgufTensorType::F32,
        );
        expect(
            &format!("{prefix}.attn_output.weight"),
            &[2_560, 4_096],
            GgufTensorType::Q4K,
        );
        expect(
            &format!("{prefix}.ffn_norm.weight"),
            &[2_560],
            GgufTensorType::F32,
        );
        expect(
            &format!("{prefix}.ffn_gate.weight"),
            &[9_728, 2_560],
            GgufTensorType::Q4K,
        );
        expect(
            &format!("{prefix}.ffn_up.weight"),
            &[9_728, 2_560],
            GgufTensorType::Q4K,
        );
        expect(
            &format!("{prefix}.ffn_down.weight"),
            &[2_560, 9_728],
            promoted,
        );
        expected += 11;
    }
    assert_eq!(expected, tensors.len(), "unexpected extra tensors");
}

#[test]
#[ignore = "requires the pinned official Qwen3-4B Q4_K_M GGUF"]
fn qwen3_4b_cpu_matches_llama_cpp_and_cache_full() -> anyhow::Result<()> {
    assert_backend(Device::Cpu, "CPU")
}

#[test]
#[ignore = "requires the pinned Qwen3-4B Q4_K_M GGUF and Apple Metal"]
fn qwen3_4b_metal_matches_llama_cpp_and_cache_full() -> anyhow::Result<()> {
    assert_backend(Device::new_metal(0)?, "Metal")
}

fn assert_backend(device: Device, label: &str) -> anyhow::Result<()> {
    let _guard = INFERENCE_TEST_LOCK
        .lock()
        .expect("Qwen3-4B inference test lock poisoned");
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
        "{label} thinking-disabled chat llama.cpp golden"
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
    eprintln!("Qwen3-4B {label} cache/full max abs difference: {difference}");
    anyhow::ensure!(
        difference <= maximum,
        "Qwen3-4B {label} cache/full max difference {difference} exceeds {maximum}"
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
        "Qwen3-4B fixture changed within one test process"
    );
    if let Err(error) = result {
        anyhow::bail!(error.clone());
    }
    ModelFiles::from_dir(path)
}

fn configured_fixture() -> PathBuf {
    std::env::var_os("MACLLM_QWEN3_4B_GGUF_MODEL").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/qwen3-4b-q4_k_m"),
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
        "expected exactly one Qwen3-4B GGUF fixture"
    );
    Ok(paths[0].clone())
}

fn validate_pinned_fixture(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(FILENAME),
        "unexpected Qwen3-4B fixture filename"
    );
    anyhow::ensure!(
        fs::metadata(path)?.len() == FIXTURE_SIZE,
        "unexpected Qwen3-4B fixture size"
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
        "unexpected Qwen3-4B fixture SHA-256"
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
