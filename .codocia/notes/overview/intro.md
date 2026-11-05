---
title: 
order: 0
evidence:
  - Cargo.toml:1-22 #1556960645b6
  - src/lib.rs:1-7 #6b4901cd87d9
  - src/benchmark/mod.rs:1 #a7d44c0687ef
---

MacLLM Infer is a single Rust crate (`macllm-infer`, binary `macllm`) that runs text-only decoder LLMs on Apple Silicon through Candle, with Metal and Accelerate as default features. One shared decoder executes every supported family; family code is limited to thin adapters that translate a checkpoint's configuration, weight names and prompt template into a format-neutral spec. The same loaded model serves blocking CLI generation, Qwen3 embedding and reranking, and a long-lived HTTP worker. `llama.cpp` is used as the reference for GGUF parsing, greedy-token correctness and performance, never as a linked dependency.

This page states exactly what the code accepts. See [[cli]] for usage, [[how-it-works]] for the design, and [[structure]] for the module map.
