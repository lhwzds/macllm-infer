---
title: 
order: 0
evidence:
  - src/benchmark/mod.rs:1 #a7d44c0687ef
  - tests/gemma_gguf_reference.rs:1-8 #552695cb68a6
  - src/main.rs:663-708 #2528a795880f
---

Claims in this project rest on two kinds of evidence. Checkpoint-pinned reference tests prove token-level correctness against `llama.cpp` or Candle. Manifest-pinned benchmarks under `macllm scoreboard`, `macllm demo` and `macllm bench` measure performance. Every measurement command refuses to run in an unoptimized build (`ensure_optimized_measurement_execution`), so use `--release`.
