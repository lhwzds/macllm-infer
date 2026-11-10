---
title: Endpoints
order: 10
evidence:
  - src/server.rs:224-228 #a624776b0e03
  - src/server.rs:264-299 #fcffd8d65d1d
  - src/engine.rs:1287-1292 #07014befda92
  - src/backend.rs:74-88 #690f00ee1dce
  - src/server.rs:327-344 #fffbe5c6bd80
  - src/server.rs:2192-2203 #32f315b33eaf
  - src/server.rs:1820-1824 #fca595c0bb0c
  - src/server.rs:2153-2178 #c0c0d77980f0
---

| Route | Purpose |
| --- | --- |
| `GET /healthz` | `status`, `worker` (`starting`, `ready`, `busy`, `poisoned`, `stopped`), `model`, `device` (the live backend: `metal`, `cuda` or `cpu`), `dtype`; 503 when unhealthy |
| `GET /v1/models` | the single served model |
| `POST /v1/chat/completions` | strict Chat Completions subset, JSON or SSE |
| `POST /completion` | untemplated, llama-server-shaped route used by the benchmarks |

`/completion` requires every field of `RawCompletionRequest` (`prompt`, `n_predict`, `temperature`, `top_k`, `top_p`, `seed`, `cache_prompt`, `stream`, `return_tokens`, `id_slot`, `ignore_eos`, `repeat_penalty`, `response_fields`) and rejects unknown ones, so both servers in a benchmark get an identical contract.

A full queue returns 429; a stopped or poisoned worker returns 503. Streams send a keep-alive comment every 15 seconds and end with `data: [DONE]`.
