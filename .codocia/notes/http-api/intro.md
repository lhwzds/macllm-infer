---
title:
order: 0
evidence:
  - src/main.rs:440-491 #988cb7bb8ee6
  - src/main.rs:1357-1392 #b2c6600af857
  - src/server.rs:142-170 #a1cae55a2bea
  - src/server.rs:232-262 #9cd7bd3fbc78
---

`macllm serve` loads one model on a dedicated worker thread and serves it over HTTP/1 with axum until Ctrl-C. It binds to loopback by default and has no authentication or TLS of its own; embedding and reranking stay CLI-only.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8080` | bind address |
| `--served-model-name` | model path's file name | ID clients send as `model` |
| `--queue-capacity` | `8` | requests waiting for an active slot |
| `--max-active-sequences` | `2` | sequences rotated on the worker |
| `--scheduler-token-budget` | `256` | model-input tokens per scheduler round |
| `--prefill-token-quantum` | `128` | prompt tokens per sequence per turn |
| `--event-capacity` | `16` | per-request event channel size |
| `--event-send-timeout-seconds` | `30` | cancel a request whose client stops reading |
| `--prefix-cache-capacity-tokens` | `32768` | prefix-cache budget; `0` disables it |
| `--default-max-completion-tokens` | `128` | budget of chat requests that set none; must fit the model context |
| `--disable-thinking` | off | `enable_thinking=false` unless a request sets it; the template must use the option |
| `--sampling-profile` | `openai` | omitted sampling fields: `openai` (1.0 / 1.0, no top-k), or `qwen3` / `qwen35` model-card values per thinking mode |
| `--reasoning-format` | `deepseek` | return thinking as `reasoning_content`; `none` keeps it in the answer |

The shared model flags (`--model`, `--device`, `--dtype`, `--kv-cache-*`) apply as in [[cli]].
