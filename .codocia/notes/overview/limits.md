---
title: Current limits at a glance
order: 20
evidence:
  - src/main.rs:910-923 #9d94b14b4067
  - src/server.rs:123-131 #5eca8bab7288
  - src/openai.rs:626-674 #ac6f356d63ec
  - src/decoder.rs:4319-4340 #b05eaac38643
  - src/backend.rs:30-37 #02796efe9930
---

Support is checkpoint-specific and fails closed. The boundaries a new user hits first:

- Embedding and reranking accept only Safetensors Qwen3 task checkpoints (`validate_task_checkpoint`) and exist only as the `embed`/`rerank` CLI commands; the HTTP router has no embedding or rerank route. See [[cli]].
- The server exposes `/healthz`, `/v1/models`, `/v1/chat/completions` and a benchmark `/completion` route. `tool_choice=auto` is rejected as not implemented; only one forced tool call is supported. See [[http-api]].
- The automatic prefix cache and sliding-window attention layers are mutually exclusive: enabling the cache on a model with a sliding layer fails at load. See [[how-it-works]].
- The CPU backend requires `--dtype f32`.
- Correctness claims come from pinned per-checkpoint reference tests, for example Gemma 1 2B Q2_K and SmolLM3-3B Q4_K_M. See [[benchmarks-and-testing]].
