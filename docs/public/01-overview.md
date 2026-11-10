---
title: Overview and Support Matrix
---

# Overview and Support Matrix

<sub>**English** · [中文](zh/01-overview.md)</sub>

MacLLM Infer is an Apple Silicon-first Rust workspace that runs text-only decoder LLMs through Candle. The main crate is `macllm-infer` (binary `macllm`); chat and tool-call handling, the K/V cache and the custom kernels live in the `macllm-chat`, `macllm-kv` and `macllm-kernels` crates. Metal and Accelerate are default features; the optional `cuda` feature adds NVIDIA GPUs through `--device cuda`. One shared decoder executes every supported family; family code is limited to thin adapters that translate a checkpoint's configuration, weight names and prompt template into a format-neutral spec. The same loaded model serves blocking CLI generation, Qwen3 embedding and reranking, and a long-lived HTTP worker. `llama.cpp` is used as the reference for GGUF parsing, greedy-token correctness and performance, never as a linked dependency.

This page states exactly what the code accepts. See [Command-Line Usage](02-cli.md) for usage, [How It Works](04-how-it-works.md) for the design, and [Code structure](structure.md) for the module map.

<sub>Sources: [Cargo.toml:1-22](../../Cargo.toml#L1-L22) · [Cargo.toml:46-71](../../Cargo.toml#L46-L71) · [lib.rs:1-7](../../src/lib.rs#L1-L7) · [lib.rs:40-45](../../src/lib.rs#L40-L45) · [backend.rs:34-58](../../src/backend.rs#L34-L58) · +1</sub>

## Supported formats and architectures

`ModelFiles::from_dir` accepts a standalone `.gguf` file or a directory. A directory holding both GGUF and Safetensors weights is rejected, and so is a directory whose `macllm download` marker says the download did not complete. Otherwise GGUF wins discovery, and canonical GGUF shard sets are opened as one logical file.

The two formats do not cover the same architectures:

| Format | Dispatch key | Architectures |
| --- | --- | --- |
| GGUF v3 | `general.architecture` in `GgufModel::open` | `falcon-h1`, `gemma`, `gemma2`, `gemma3`, `granitemoe`, `lfm2`, `llama`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `smollm3` |
| Safetensors | `model_type` in `config.json` (`detect`) | `llama`, `lfm2`, `qwen2`, `qwen3` |

The 12-architecture list therefore applies to GGUF only. A Safetensors directory also needs `config.json` and `tokenizer.json`; GGUF uses its embedded tokenizer. GGUF tensor types accepted by `GgufTensorType` are F32, F16, BF16, Q4_0, Q5_0, Q5_1, Q8_0 and Q2_K through Q6_K.

<sub>Sources: [weights.rs:58-115](../../src/weights.rs#L58-L115) · [snapshot.rs:83-96](../../src/snapshot.rs#L83-L96) · [mod.rs:698-714](../../src/models/mod.rs#L698-L714) · [gguf_model.rs:316-335](../../src/gguf_model.rs#L316-L335) · [gguf.rs:283-297](../../src/gguf.rs#L283-L297)</sub>

## Current limits at a glance

Support is checkpoint-specific and fails closed. The boundaries a new user hits first:

- Embedding and reranking accept only Safetensors Qwen3 task checkpoints (`validate_task_checkpoint`) and exist only as the `embed`/`rerank` CLI commands; the HTTP router has no embedding or rerank route. See [Command-Line Usage](02-cli.md).
- The server exposes `/healthz`, `/v1/models`, `/v1/chat/completions` and a benchmark `/completion` route. Which tool calls a model can make depends on its chat template. See [HTTP Server and API Subset](03-http-api.md).
- The automatic prefix cache and sliding-window attention layers are mutually exclusive: enabling the cache on a model with a sliding layer fails at load. See [How It Works](04-how-it-works.md).
- The CPU backend requires `--dtype f32`.
- Correctness claims come from pinned per-checkpoint reference tests, for example Gemma 1 2B Q2_K and SmolLM3-3B Q4_K_M. See [Benchmarks, Gates and Contributing](05-benchmarks-and-testing.md).

<sub>Sources: [main.rs:1010-1023](../../src/main.rs#L1010-L1023) · [server.rs:224-228](../../src/server.rs#L224-L228) · [decoder.rs:4331-4352](../../src/decoder.rs#L4331-L4352) · [backend.rs:61-71](../../src/backend.rs#L61-L71)</sub>
