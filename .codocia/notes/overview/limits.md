---
title: Current limits at a glance
order: 20
evidence:
  - src/main.rs:1010-1023 #9d94b14b4067
  - src/server.rs:224-228 #a624776b0e03
  - src/decoder.rs:4331-4352 #b05eaac38643
  - src/backend.rs:61-71 #304892e4bac1
---

Support is checkpoint-specific and fails closed. The boundaries a new user hits first:

- Embedding and reranking accept only Safetensors Qwen3 task checkpoints (`validate_task_checkpoint`) and exist only as the `embed`/`rerank` CLI commands; the HTTP router has no embedding or rerank route. See [[cli]].
- The server exposes `/healthz`, `/v1/models`, `/v1/chat/completions` and a benchmark `/completion` route. Which tool calls a model can make depends on its chat template. See [[http-api]].
- The automatic prefix cache and sliding-window attention layers are mutually exclusive: enabling the cache on a model with a sliding layer fails at load. See [[how-it-works]].
- The CPU backend requires `--dtype f32`.
- Correctness claims come from pinned per-checkpoint reference tests, for example Gemma 1 2B Q2_K and SmolLM3-3B Q4_K_M. See [[benchmarks-and-testing]].
