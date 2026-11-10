---
title: HTTP Server and API Subset
---

# HTTP Server and API Subset

<sub>**English** · [中文](zh/03-http-api.md)</sub>

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

The shared model flags (`--model`, `--device`, `--dtype`, `--kv-cache-*`) apply as in [Command-Line Usage](02-cli.md).

<sub>Sources: [main.rs:440-491](../../src/main.rs#L440-L491) · [main.rs:1357-1392](../../src/main.rs#L1357-L1392) · [server.rs:142-170](../../src/server.rs#L142-L170) · [server.rs:232-262](../../src/server.rs#L232-L262)</sub>

## Endpoints

| Route | Purpose |
| --- | --- |
| `GET /healthz` | `status`, `worker` (`starting`, `ready`, `busy`, `poisoned`, `stopped`), `model`, `device` (the live backend: `metal`, `cuda` or `cpu`), `dtype`; 503 when unhealthy |
| `GET /v1/models` | the single served model |
| `POST /v1/chat/completions` | strict Chat Completions subset, JSON or SSE |
| `POST /completion` | untemplated, llama-server-shaped route used by the benchmarks |

`/completion` requires every field of `RawCompletionRequest` (`prompt`, `n_predict`, `temperature`, `top_k`, `top_p`, `seed`, `cache_prompt`, `stream`, `return_tokens`, `id_slot`, `ignore_eos`, `repeat_penalty`, `response_fields`) and rejects unknown ones, so both servers in a benchmark get an identical contract.

A full queue returns 429; a stopped or poisoned worker returns 503. Streams send a keep-alive comment every 15 seconds and end with `data: [DONE]`.

<sub>Sources: [server.rs:224-228](../../src/server.rs#L224-L228) · [server.rs:264-299](../../src/server.rs#L264-L299) · [engine.rs:1287-1292](../../src/engine.rs#L1287-L1292) · [backend.rs:74-88](../../src/backend.rs#L74-L88) · [server.rs:327-344](../../src/server.rs#L327-L344) · +3</sub>

## What /v1/chat/completions accepts

Unknown JSON fields are a 400, and known-but-unsupported fields return an OpenAI-shaped error instead of being dropped.

- `model` must equal the served model name; `n` must be 1.
- `max_completion_tokens` (or the deprecated `max_tokens`, never both) defaults to `--default-max-completion-tokens` and may not exceed the model context.
- `temperature` in [0, 2] and `top_p` in (0, 1] default to the `--sampling-profile` values, which also fix `top_k`; `presence_penalty` and `frequency_penalty` take [-2, 2]; `seed` must be non-negative. Repetition penalty is fixed to 1.0.
- `stop` takes a string or up to 4 strings.
- `stream` with `stream_options.include_usage`; `include_obfuscation=true` is rejected.
- `chat_template_kwargs` accepts only a boolean `enable_thinking`.
- Roles: `system`, `user`, `assistant`, and `tool` in tool-call histories (string content and a `tool_call_id` answering a call of the preceding assistant turn). That turn holds exactly one call, or several on generic templates with a detected tool-call format. Assistant `reasoning_content` is accepted only for the Qwen3, Qwen3MoE and Qwen3.5 model families, whichever template renders them. `function`, `developer`, `name`, multimodal content, `logprobs` and `top_logprobs` are rejected.

`usage.prompt_tokens_details.cached_tokens` reports how many prompt tokens came from the prefix cache.

<sub>Sources: [openai.rs:54-108](../../crates/macllm-chat/src/openai.rs#L54-L108) · [openai.rs:136-230](../../crates/macllm-chat/src/openai.rs#L136-L230) · [openai.rs:600-630](../../crates/macllm-chat/src/openai.rs#L600-L630) · [output.rs:4](../../crates/macllm-chat/src/output.rs#L4) · [openai.rs:937-1035](../../crates/macllm-chat/src/openai.rs#L937-L1035) · +6</sub>

## Structured output and tool calls

The constraint grammar in `crates/macllm-chat/src/grammar.rs` checks each candidate token's decoded bytes against a closed JSON Schema subset. Unknown schema keywords are rejected, and object schemas need `additionalProperties: false`.

- `response_format` takes only `{"type":"json_schema","json_schema":{...,"strict":true}}`. It cannot be combined with `stop`, tools or tool-call history.
- Tools: up to 128 uniquely named functions. With tools, `tool_choice` defaults to `auto`; `none`, a named function, or `"required"` with exactly one tool are also accepted. `stop` is not allowed while tools are enabled.
- `auto` on a pinned Qwen3, Qwen3MoE or Qwen3.5 template: the answer may end in one call, constrained by a lazy grammar. On a generic GGUF template, the call format is detected from the template (Hermes, Qwen XML, Llama 3 JSON, Mistral JSON, Granite JSON, Pythonic); every parsed call is checked against its declaration; only the first is returned unless `parallel_tool_calls` allows more.
- Forced calls need `parallel_tool_calls=false` and generate the arguments under the schema. They work on pinned Qwen3/Qwen3MoE profiles and generic Hermes templates; Qwen3.5 refuses them, as does a generic template whose prompt opens a reasoning block.
- Other adapters fail closed in `format_agent_prompt`.

<sub>Sources: [grammar.rs:1-20](../../crates/macllm-chat/src/grammar.rs#L1-L20) · [openai.rs:232-247](../../crates/macllm-chat/src/openai.rs#L232-L247) · [openai.rs:632-650](../../crates/macllm-chat/src/openai.rs#L632-L650) · [openai.rs:652-689](../../crates/macllm-chat/src/openai.rs#L652-L689) · [openai.rs:691-795](../../crates/macllm-chat/src/openai.rs#L691-L795) · +13</sub>

## Startup checks and the prefix-cache flag

`EngineHandle::start` returns only after the worker has loaded the model, so `serve` fails before binding if any check fails:

1. The model must support packed ragged batches.
2. The adapter must render a health-check chat prompt through its pinned or generic template, and that prompt must tokenize to at least one token. Checkpoints without a usable template can be used with `run --raw-prompt` but not served.
3. A server-wide `--disable-thinking` must be honoured by the template.
4. All `--max-active-sequences` sessions are created up front.

`serve` then checks that `--default-max-completion-tokens` fits the model context.

`serve` enables the automatic prefix cache by default (32768 tokens), and the cache cannot coexist with sliding-window attention ([How It Works](04-how-it-works.md)). A model with any `SlidingCausal` layer therefore fails to load with "sliding-attention storage is mutable". Gemma 2 and Gemma 3 normally interleave such layers, and GGUF Llama gets them when `llama.attention.sliding_window` is non-zero. Serve those models with `--prefix-cache-capacity-tokens 0`.

<sub>Sources: [engine.rs:863-867](../../src/engine.rs#L863-L867) · [engine.rs:1224-1239](../../src/engine.rs#L1224-L1239) · [engine.rs:1269-1360](../../src/engine.rs#L1269-L1360) · [mod.rs:115-117](../../src/models/mod.rs#L115-L117) · [server.rs:233-240](../../src/server.rs#L233-L240) · +6</sub>
