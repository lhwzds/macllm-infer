//! Real Llama 3.2 GGUF parity gates against a pinned llama.cpp golden.
//!
//! Run with:
//! `MACLLM_LLAMA3_GGUF_MODEL=models/llama3.2-1b-instruct-q4-k-m/Llama-3.2-1B-Instruct-Q4_K_M.gguf cargo test --release --test llama3_gguf_reference -- --ignored`

use std::path::PathBuf;

use candle_core::{DType, Device};
use macllm_infer::{
    generation::{GenerationConfig, generate},
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};

const PROMPT_IDS_WITH_BOS: &[u32] = &[128_000, 791, 6_864, 315, 9_822, 374];
const LLAMA_CPP_GREEDY_IDS: &[u32] = &[12_366, 13, 578, 469, 3_168, 301, 22_703, 374];
const LLAMA_CPP_LONG_PREFIX_IDS: &[u32] = &[578, 6_864, 315, 9_822, 374, 578, 6_864, 315];

#[test]
#[ignore = "requires the pinned Llama 3.2 Q4_K_M GGUF and Apple Metal"]
fn llama3_q4_k_m_matches_recorded_llama_cpp_greedy_sequence() -> anyhow::Result<()> {
    let files = ModelFiles::from_dir(PathBuf::from(std::env::var("MACLLM_LLAMA3_GGUF_MODEL")?))?;
    let tokenizer = files.tokenizer()?;
    let prompt = tokenizer
        .encode("The capital of France is", true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(prompt.get_ids(), PROMPT_IDS_WITH_BOS);
    let expected = tokenizer
        .encode(" Paris. The Eiffel Tower is", false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(expected.get_ids(), LLAMA_CPP_GREEDY_IDS);

    let adapter = files.adapter()?;
    let device = Device::new_metal(0)?;
    // SAFETY: the parity gate treats the pinned checkpoint as immutable.
    let weights = unsafe { files.weight_source(DType::F32, &device)? };
    let model = adapter.build(weights.as_ref())?;
    drop(weights);
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

    let mut long_session = model.new_session()?;
    let long_prompt = "The capital of France is ".repeat(64);
    let long_encoding = tokenizer
        .encode(long_prompt, true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    assert_eq!(long_encoding.get_ids().len(), 322);
    let mut long_sampler = Sampler::new(SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: 42,
        ..SamplingConfig::default()
    })?;
    let mut long_generated = Vec::new();
    generate(
        model.as_ref(),
        long_session.as_mut(),
        long_encoding.get_ids(),
        &mut long_sampler,
        GenerationConfig {
            max_new_tokens: LLAMA_CPP_LONG_PREFIX_IDS.len(),
            ignore_stop_tokens: true,
        },
        |token| {
            long_generated.push(token);
            Ok(())
        },
    )?;
    assert_eq!(long_generated, LLAMA_CPP_LONG_PREFIX_IDS);
    Ok(())
}
