//! Real Gemma 3 GGUF parity gates against pinned llama.cpp goldens.
//!
//! The constants below pin the reproducible 782-token post-window capture.
//!
//! Run with:
//! `MACLLM_GEMMA3_GGUF_MODEL=models/gemma-3-270m-it-qat-q4_0 \
//!  cargo test --release --test gemma3_gguf_reference -- --ignored --nocapture`

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use macllm_infer::{
    architecture::{CausalLm, CausalLmSession},
    cache::KvCachePolicy,
    generation::{GenerationConfig, generate},
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};

const PROMPT_IDS_WITH_BOS: &[u32] = &[2, 818, 5_279, 529, 7_001, 563];
const LLAMA_CPP_GREEDY_IDS: &[u32] = &[9_079, 236_761, 108, 1_018, 20_416, 236_764, 9_079, 563];
const LLAMA_CPP_LONG_PREFIX_GREEDY_IDS: &[u32] = &[107, 818, 5_279, 529];
const LONG_PREFIX_FRAGMENT: &str =
    "The capital of France is Paris. Rust inference uses absolute positions. ";
const LONG_PREFIX_TOKEN_COUNT: usize = 782;
const LLAMA_CPP_LONG_PREFIX_FNV64: u64 = 9_948_813_959_984_910_743;
const MAX_CACHE_FULL_DIFFERENCE: f32 = 0.1;

#[test]
#[ignore = "requires the pinned Gemma 3 270M Q4_0 GGUF and Apple Metal"]
fn gemma3_q4_0_matches_recorded_llama_cpp_greedy_sequence() -> anyhow::Result<()> {
    let files = fixture()?;
    let tokenizer = files.tokenizer()?;
    let prompt = tokenizer
        .encode("The capital of France is", true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(prompt.get_ids(), PROMPT_IDS_WITH_BOS);
    let expected = tokenizer
        .encode(" Paris.\n\n**Therefore, Paris is", false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(expected.get_ids(), LLAMA_CPP_GREEDY_IDS);

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
    generate(
        model.as_ref(),
        session.as_mut(),
        prompt.get_ids(),
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
    assert_eq!(generated, LLAMA_CPP_GREEDY_IDS);
    Ok(())
}

#[test]
#[ignore = "requires the pinned Gemma 3 270M Q4_0 GGUF and Apple Metal"]
fn gemma3_q4_0_matches_llama_cpp_after_sliding_window() -> anyhow::Result<()> {
    let files = fixture()?;
    let tokenizer = files.tokenizer()?;
    let prompt = tokenizer
        .encode(LONG_PREFIX_FRAGMENT.repeat(60), true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(prompt.get_ids().len(), LONG_PREFIX_TOKEN_COUNT);
    assert_eq!(
        token_id_fingerprint(prompt.get_ids()),
        LLAMA_CPP_LONG_PREFIX_FNV64
    );
    let expected = tokenizer
        .encode("\nThe capital of", false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(expected.get_ids(), LLAMA_CPP_LONG_PREFIX_GREEDY_IDS);

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
    generate(
        model.as_ref(),
        session.as_mut(),
        prompt.get_ids(),
        &mut sampler,
        GenerationConfig {
            max_new_tokens: LLAMA_CPP_LONG_PREFIX_GREEDY_IDS.len(),
            ignore_stop_tokens: true,
        },
        |token| {
            generated.push(token);
            Ok(())
        },
    )?;
    assert_eq!(generated, LLAMA_CPP_LONG_PREFIX_GREEDY_IDS);
    Ok(())
}

#[test]
#[ignore = "requires the pinned Gemma 3 270M Q4_0 GGUF and Apple Metal"]
fn gemma3_cached_decode_matches_full_forward() -> anyhow::Result<()> {
    let files = fixture()?;
    let device = Device::new_metal(0)?;
    let prompt = Tensor::from_slice(PROMPT_IDS_WITH_BOS, (1, PROMPT_IDS_WITH_BOS.len()), &device)?;
    let model = build(&files, &device)?;
    let mut cached_session = model.new_session()?;
    let prefill = model.forward(cached_session.as_mut(), &prompt, 0)?;
    assert_eq!(argmax(&prefill)?, LLAMA_CPP_GREEDY_IDS[0]);

    let next = Tensor::new(&[[LLAMA_CPP_GREEDY_IDS[0]]], &device)?;
    let cached = model.forward(cached_session.as_mut(), &next, PROMPT_IDS_WITH_BOS.len())?;
    let mut combined = PROMPT_IDS_WITH_BOS.to_vec();
    combined.push(LLAMA_CPP_GREEDY_IDS[0]);
    let full_input = Tensor::from_vec(combined, (1, PROMPT_IDS_WITH_BOS.len() + 1), &device)?;
    let mut full_session = model.new_session()?;
    let full = model.forward(full_session.as_mut(), &full_input, 0)?;

    assert_eq!(argmax(&cached)?, LLAMA_CPP_GREEDY_IDS[1]);
    assert_eq!(argmax(&full)?, LLAMA_CPP_GREEDY_IDS[1]);
    let difference = cached.sub(&full)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(
        difference <= MAX_CACHE_FULL_DIFFERENCE,
        "Gemma 3 cached/full logits differed by {difference}"
    );
    Ok(())
}

#[test]
#[ignore = "requires the pinned Gemma 3 270M Q4_0 GGUF and Apple Metal"]
fn gemma3_cached_decode_matches_full_forward_across_sliding_boundary() -> anyhow::Result<()> {
    const SLIDING_WINDOW: usize = 512;

    let files = fixture()?;
    let device = Device::new_metal(0)?;
    let prompt_ids = vec![818_u32; SLIDING_WINDOW];
    let prompt = Tensor::from_slice(&prompt_ids, (1, prompt_ids.len()), &device)?;
    let model = build(&files, &device)?;
    let mut cached_session = model.new_session()?;
    let prefill = model.forward(cached_session.as_mut(), &prompt, 0)?;
    assert_gemma_cache_geometry(cached_session.as_ref(), SLIDING_WINDOW)?;
    let next_token = argmax(&prefill)?;

    let next = Tensor::new(&[[next_token]], &device)?;
    let cached = model.forward(cached_session.as_mut(), &next, prompt_ids.len())?;
    assert_gemma_cache_geometry(cached_session.as_ref(), SLIDING_WINDOW + 1)?;
    let mut combined_ids = prompt_ids;
    combined_ids.push(next_token);
    let combined = Tensor::from_vec(combined_ids, (1, SLIDING_WINDOW + 1), &device)?;
    let mut full_session = model.new_session()?;
    let full = model.forward(full_session.as_mut(), &combined, 0)?;

    assert_eq!(argmax(&cached)?, argmax(&full)?);
    let difference = cached.sub(&full)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(
        difference <= MAX_CACHE_FULL_DIFFERENCE,
        "Gemma 3 boundary cached/full logits differed by {difference}"
    );

    // Cross the same boundary inside a multi-token chunk. The local layer must
    // keep the transient position 511 K/V available to position 512 before it
    // publishes only the newest 511 positions to the bounded ring.
    let prefix_ids = vec![818_u32; SLIDING_WINDOW - 1];
    let mut chunked_session = model.new_session()?;
    let _ = model.forward_tokens(chunked_session.as_mut(), &prefix_ids, 0)?;
    assert_gemma_cache_geometry(chunked_session.as_ref(), SLIDING_WINDOW - 1)?;
    let chunked = model.forward_tokens(chunked_session.as_mut(), &[818, 818], prefix_ids.len())?;
    assert_gemma_cache_geometry(chunked_session.as_ref(), SLIDING_WINDOW + 1)?;
    let full_ids = vec![818_u32; SLIDING_WINDOW + 1];
    let mut chunk_full_session = model.new_session()?;
    let full = model.forward_tokens(chunk_full_session.as_mut(), &full_ids, 0)?;
    assert_eq!(argmax(&chunked)?, argmax(&full)?);
    let difference = chunked.sub(&full)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(
        difference <= MAX_CACHE_FULL_DIFFERENCE,
        "Gemma 3 [511]+[2]/full logits differed by {difference}"
    );
    Ok(())
}

#[test]
#[ignore = "requires the pinned Gemma 3 270M Q4_0 GGUF and Apple Metal"]
fn gemma3_long_chunked_prefill_matches_one_shot_and_bounds_local_storage() -> anyhow::Result<()> {
    const SLIDING_WINDOW: usize = 512;
    const HISTORY: usize = SLIDING_WINDOW - 1;
    const RING_CAPACITY: usize = 2 * HISTORY;
    const LONG_SEQUENCE: usize = 2 * SLIDING_WINDOW + 1;

    let files = fixture()?;
    let device = Device::new_metal(0)?;
    let ids = vec![818_u32; LONG_SEQUENCE];

    let model = build(&files, &device)?;
    let mut one_shot_session = model.new_session()?;
    let one_shot = model.forward_tokens(one_shot_session.as_mut(), &ids, 0)?;

    let mut chunked_session = model.new_session()?;
    let mut start = 0;
    let mut chunked = None;
    for len in [HISTORY, 2, SLIDING_WINDOW] {
        chunked = Some(model.forward_tokens(
            chunked_session.as_mut(),
            &ids[start..start + len],
            start,
        )?);
        start += len;
    }
    assert_eq!(start, LONG_SEQUENCE);
    let chunked = chunked.expect("last chunk logits");
    assert_eq!(argmax(&chunked)?, argmax(&one_shot)?);
    let difference = chunked
        .sub(&one_shot)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        difference <= MAX_CACHE_FULL_DIFFERENCE,
        "Gemma 3 long chunked/one-shot logits differed by {difference}"
    );

    let stats = chunked_session
        .kv_cache_stats()
        .ok_or_else(|| anyhow::anyhow!("Gemma 3 runtime did not expose KV cache stats"))?;
    assert_eq!(stats.len(), 18);
    let local = stats
        .iter()
        .filter(|stats| stats.policy == KvCachePolicy::Sliding)
        .collect::<Vec<_>>();
    let global = stats
        .iter()
        .filter(|stats| stats.policy == KvCachePolicy::Full)
        .collect::<Vec<_>>();
    assert_eq!(local.len(), 15);
    assert_eq!(global.len(), 3);
    assert!(local.iter().all(|stats| {
        stats.logical_len == LONG_SEQUENCE
            && stats.resident_start == LONG_SEQUENCE - HISTORY
            && stats.resident_len == HISTORY
            && stats.allocated_capacity == RING_CAPACITY
            && stats.allocated_bytes == RING_CAPACITY * 2 * 256 * std::mem::size_of::<f32>()
    }));
    assert!(global.iter().all(|stats| {
        stats.logical_len == LONG_SEQUENCE
            && stats.resident_start == 0
            && stats.resident_len == LONG_SEQUENCE
            && stats.allocated_capacity >= LONG_SEQUENCE
            && stats.allocated_bytes >= LONG_SEQUENCE * 2 * 256 * std::mem::size_of::<f32>()
    }));
    Ok(())
}

fn fixture() -> anyhow::Result<ModelFiles> {
    ModelFiles::from_dir(PathBuf::from(std::env::var("MACLLM_GEMMA3_GGUF_MODEL")?))
}

fn assert_gemma_cache_geometry(
    session: &dyn CausalLmSession,
    logical_len: usize,
) -> anyhow::Result<()> {
    const HISTORY: usize = 511;
    const RING_CAPACITY: usize = 2 * HISTORY;

    let stats = session
        .kv_cache_stats()
        .ok_or_else(|| anyhow::anyhow!("Gemma 3 runtime did not expose KV cache stats"))?;
    assert_eq!(stats.len(), 18);
    for (layer, stats) in stats.iter().enumerate() {
        assert_eq!(stats.logical_len, logical_len, "layer {layer}");
        if layer % 6 == 5 {
            assert_eq!(stats.policy, KvCachePolicy::Paged, "layer {layer}");
            assert_eq!(stats.resident_start, 0, "layer {layer}");
            assert_eq!(stats.resident_len, logical_len, "layer {layer}");
        } else {
            assert_eq!(stats.policy, KvCachePolicy::Sliding, "layer {layer}");
            assert_eq!(
                stats.resident_start,
                logical_len.saturating_sub(HISTORY),
                "layer {layer}"
            );
            assert_eq!(
                stats.resident_len,
                logical_len.min(HISTORY),
                "layer {layer}"
            );
            assert_eq!(stats.allocated_capacity, RING_CAPACITY, "layer {layer}");
        }
    }
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

fn argmax(logits: &Tensor) -> candle_core::Result<u32> {
    logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()
}

fn token_id_fingerprint(ids: &[u32]) -> u64 {
    ids.iter()
        .flat_map(|id| id.to_le_bytes())
        .fold(14_695_981_039_346_656_037, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
        })
}
