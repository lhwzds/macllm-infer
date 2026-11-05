---
title: 
order: 0
evidence:
  - src/main.rs:396-429 #b435123341fb
  - src/main.rs:1257-1283 #1942481a767b
  - src/server.rs:123-150 #90f5e50370c0
---

`macllm serve` loads one model on a dedicated worker thread and serves it over HTTP/1 with axum until Ctrl-C. It binds to loopback by default and has no authentication or TLS of its own. Embedding and reranking are not served; use the CLI ([[cli]]).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8080` | bind address |
| `--served-model-name` | model path's file name | ID clients must send as `model` |
| `--queue-capacity` | `8` | requests waiting for an active slot |
| `--max-active-sequences` | `2` | sequences rotated on the worker |
| `--scheduler-token-budget` | `256` | model-input tokens per scheduler round |
| `--prefill-token-quantum` | `128` | prompt tokens per sequence per turn |
| `--event-capacity` | `16` | per-request event channel size |
| `--event-send-timeout-seconds` | `30` | cancel a request whose client stops reading |
| `--prefix-cache-capacity-tokens` | `32768` | prefix-cache budget; `0` disables it |

The shared model flags (`--model`, `--device`, `--dtype`, `--kv-cache-*`) apply as in [[cli]].
