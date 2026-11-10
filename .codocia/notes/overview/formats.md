---
title: Supported formats and architectures
order: 10
evidence:
  - src/weights.rs:58-115 #729a297ff30e
  - src/snapshot.rs:83-96 #b285a82f8992
  - src/models/mod.rs:698-714 #d1fbf6cba622
  - src/gguf_model.rs:316-335 #9cbd64313536
  - src/gguf.rs:283-297 #f65a66077752
---

`ModelFiles::from_dir` accepts a standalone `.gguf` file or a directory. A directory holding both GGUF and Safetensors weights is rejected, and so is a directory whose `macllm download` marker says the download did not complete. Otherwise GGUF wins discovery, and canonical GGUF shard sets are opened as one logical file.

The two formats do not cover the same architectures:

| Format | Dispatch key | Architectures |
| --- | --- | --- |
| GGUF v3 | `general.architecture` in `GgufModel::open` | `falcon-h1`, `gemma`, `gemma2`, `gemma3`, `granitemoe`, `lfm2`, `llama`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `smollm3` |
| Safetensors | `model_type` in `config.json` (`detect`) | `llama`, `lfm2`, `qwen2`, `qwen3` |

The 12-architecture list therefore applies to GGUF only. A Safetensors directory also needs `config.json` and `tokenizer.json`; GGUF uses its embedded tokenizer. GGUF tensor types accepted by `GgufTensorType` are F32, F16, BF16, Q4_0, Q5_0, Q5_1, Q8_0 and Q2_K through Q6_K.
