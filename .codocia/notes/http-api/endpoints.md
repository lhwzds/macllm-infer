---
title: Endpoints
order: 10
evidence:
  - src/server.rs:123-131 #5eca8bab7288
  - src/server.rs:156-185 #255fb6a64ee9
  - src/server.rs:219-236 #fffbe5c6bd80
  - src/server.rs:1265-1276 #32f315b33eaf
  - src/server.rs:1088-1096 #395941996be3
  - src/server.rs:1220-1252 #c5ffb42219c9
---

| Route | Purpose |
| --- | --- |
| `GET /healthz` | `status`, `worker` (`starting`, `ready`, `busy`, `poisoned`, `stopped`), `model`, `device`, `dtype`; 503 when unhealthy |
| `GET /v1/models` | the single served model |
| `POST /v1/chat/completions` | strict Chat Completions subset, JSON or SSE |
| `POST /completion` | untemplated, llama-server-shaped route used by the benchmarks |

`/completion` requires every field of `RawCompletionRequest` (`prompt`, `n_predict`, `temperature`, `top_k`, `top_p`, `seed`, `cache_prompt`, `stream`, `return_tokens`, `id_slot`, `ignore_eos`, `repeat_penalty`, `response_fields`) and rejects unknown ones, so both servers in a benchmark get an identical contract.

A full queue returns 429; a stopped or poisoned worker returns 503. Streams send a keep-alive comment every 15 seconds and end with `data: [DONE]`.
