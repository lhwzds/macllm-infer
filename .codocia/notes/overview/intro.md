---
title:
order: 0
evidence:
  - Cargo.toml:1-22 #af4a0972e7de
  - Cargo.toml:46-71 #0ceee3ff92c2
  - src/lib.rs:1-7 #6b4901cd87d9
  - src/lib.rs:40-45 #4f98110284b0
  - src/backend.rs:34-58 #71c8d601d78e
  - src/benchmark/mod.rs:1 #a7d44c0687ef
---

MacLLM Infer is an Apple Silicon-first Rust workspace that runs text-only decoder LLMs through Candle. The main crate is `macllm-infer` (binary `macllm`); chat and tool-call handling, the K/V cache and the custom kernels live in the `macllm-chat`, `macllm-kv` and `macllm-kernels` crates. Metal and Accelerate are default features; the optional `cuda` feature adds NVIDIA GPUs through `--device cuda`. One shared decoder executes every supported family; family code is limited to thin adapters that translate a checkpoint's configuration, weight names and prompt template into a format-neutral spec. The same loaded model serves blocking CLI generation, Qwen3 embedding and reranking, and a long-lived HTTP worker. `llama.cpp` is used as the reference for GGUF parsing, greedy-token correctness and performance, never as a linked dependency.

This page states exactly what the code accepts. See [[cli]] for usage, [[how-it-works]] for the design, and [[structure]] for the module map.
