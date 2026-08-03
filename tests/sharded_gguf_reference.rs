//! Real canonical sharded-GGUF gate against a pinned llama.cpp greedy golden.
//!
//! Build the fixture from the pinned Qwen3 `Q8_0` file with llama.cpp b10050:
//! `mkdir -p /tmp/qwen3-q8-shards && llama-gguf-split --split-max-tensors 28 \
//!  Qwen3-0.6B-Q8_0.gguf /tmp/qwen3-q8-shards/qwen3-q8`.
//! Then run:
//! `MACLLM_SHARDED_GGUF_DIR=/tmp/qwen3-q8-shards \
//!  cargo test --release --test sharded_gguf_reference -- --ignored --nocapture`

use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use macllm_infer::{
    backend::{DTypeKind, DeviceKind},
    generation::{GenerationConfig, generate},
    gguf::GgufTensorType,
    runtime::{LoadedRuntime, ModelLoadConfig},
    sampling::{Sampler, SamplingConfig},
    weights::ModelFiles,
};
use sha2::{Digest, Sha256};

const RAW_PROMPT_IDS: &[u32] = &[17, 488, 220, 17, 284];
// llama.cpp b10050/b15ca938a, CPU and Metal, temperature zero.
const LLAMA_CPP_GREEDY_IDS: &[u32] = &[220, 19, 11, 773, 279, 2_629, 374, 220];

struct ShardSpec {
    filename: &'static str,
    size: u64,
    sha256: &'static str,
}

// Produced from the pinned 639,446,688-byte Qwen3 Q8_0 checkpoint with
// llama.cpp b10050 and `--split-max-tensors 28`.
const SHARDS: &[ShardSpec] = &[
    ShardSpec {
        filename: "qwen3-q8-00001-of-00012.gguf",
        size: 208_033_760,
        sha256: "8b6e64cf2cccb5f89cc9d3730c28ca7c516263808350cb716fa78d136c3cfee4",
    },
    ShardSpec {
        filename: "qwen3-q8-00002-of-00012.gguf",
        size: 43_475_168,
        sha256: "1b7a4dcad056fd41a7885562b6ecd4e002e13c4260b9ff5c319c1b0b716e049a",
    },
    ShardSpec {
        filename: "qwen3-q8-00003-of-00012.gguf",
        size: 42_361_056,
        sha256: "07a56a3f36e8876d000fa8ce27a0459eab1f74b8c565ca765c9e769642aa12df",
    },
    ShardSpec {
        filename: "qwen3-q8-00004-of-00012.gguf",
        size: 44_589_280,
        sha256: "1da6ee6ad0d23a670c7f6ac21e17b82c98d9f23caaf380b6ffd6033852c5589c",
    },
    ShardSpec {
        filename: "qwen3-q8-00005-of-00012.gguf",
        size: 39_019_264,
        sha256: "4dc022eda638b8c311f7fb2687353304bf06f8a0799473ea721db07a359efa38",
    },
    ShardSpec {
        filename: "qwen3-q8-00006-of-00012.gguf",
        size: 45_702_912,
        sha256: "ff886b8fe4b4501b9fb031bda49b88bda96034278817096e3d20fd1750e3707b",
    },
    ShardSpec {
        filename: "qwen3-q8-00007-of-00012.gguf",
        size: 39_019_264,
        sha256: "ab9342f4ebb0e40c4abd42e247da5681a216140aefae8997755725e3c4cb589f",
    },
    ShardSpec {
        filename: "qwen3-q8-00008-of-00012.gguf",
        size: 44_589_312,
        sha256: "a9ee8d3b34fae637119c0b2ed51026309be9eb7064cc487b73616e52bf73b5e0",
    },
    ShardSpec {
        filename: "qwen3-q8-00009-of-00012.gguf",
        size: 42_361_088,
        sha256: "e76945ff6791373bd0979a412b006c9ed0e34644edba8771002cb70031578854",
    },
    ShardSpec {
        filename: "qwen3-q8-00010-of-00012.gguf",
        size: 41_251_040,
        sha256: "e1b7a4fd8f7052b3a2b0b1eff90d2dc6ab03abe68d3be9329667786ffcdf5bc0",
    },
    ShardSpec {
        filename: "qwen3-q8-00011-of-00012.gguf",
        size: 45_699_328,
        sha256: "32b3941fe3d0b295b7f41e280a02f72191053987d5cf130a06f9c25ba2cc23b4",
    },
    ShardSpec {
        filename: "qwen3-q8-00012-of-00012.gguf",
        size: 3_346_656,
        sha256: "ffc4f0f33ff5a1823f0da67f050c94a5d90d5d14280ffe6d637a60c3c2882042",
    },
];

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024].into_boxed_slice();
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[test]
#[ignore = "requires the pinned Qwen3 0.6B Q8_0 split into 12 canonical GGUF shards"]
fn qwen3_shards_load_every_tensor_and_match_llama_cpp_greedy() -> anyhow::Result<()> {
    let directory = PathBuf::from(std::env::var("MACLLM_SHARDED_GGUF_DIR")?);
    anyhow::ensure!(
        directory.is_dir(),
        "MACLLM_SHARDED_GGUF_DIR must name the dedicated shard directory"
    );
    for shard in SHARDS {
        let path = directory.join(shard.filename);
        assert_eq!(std::fs::metadata(&path)?.len(), shard.size, "{path:?}");
        assert_eq!(sha256_file(&path)?, shard.sha256, "{path:?}");
    }

    let files = ModelFiles::from_dir(&directory)?;
    assert_eq!(files.weight_file_count(), 12);
    assert_eq!(files.tensor_count(), Some(310));
    assert_eq!(
        files.gguf_tensor_type_counts(),
        [(GgufTensorType::F32, 113), (GgufTensorType::Q8_0, 197)]
    );

    let runtime = LoadedRuntime::load(&ModelLoadConfig::new(
        directory,
        DeviceKind::Cpu,
        Some(DTypeKind::F32),
    ))?;
    let prompt = runtime.encode("2 + 2 =", false)?;
    assert_eq!(prompt, RAW_PROMPT_IDS);
    let mut session = runtime.model.new_session()?;
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 42,
    })?;
    let mut generated = Vec::new();
    generate(
        runtime.model.as_ref(),
        session.as_mut(),
        &prompt,
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
