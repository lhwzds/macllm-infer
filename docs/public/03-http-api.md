---
title: HTTP Server and API Subset
---

# HTTP Server and API Subset

<sub>**English** · [中文](zh/03-http-api.md)</sub>

`macllm serve` loads one model on a dedicated worker thread and serves it over HTTP/1 with axum until Ctrl-C. It binds to loopback by default and has no authentication or TLS of its own. Embedding and reranking are not served; use the CLI ([Command-Line Usage](02-cli.md)).

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

The shared model flags (`--model`, `--device`, `--dtype`, `--kv-cache-*`) apply as in [Command-Line Usage](02-cli.md).

<sub>Sources: [main.rs:396-429](../../src/main.rs#L396-L429) · [main.rs:1257-1283](../../src/main.rs#L1257-L1283) · [server.rs:123-150](../../src/server.rs#L123-L150)</sub>

## Endpoints

| Route | Purpose |
| --- | --- |
| `GET /healthz` | `status`, `worker` (`starting`, `ready`, `busy`, `poisoned`, `stopped`), `model`, `device`, `dtype`; 503 when unhealthy |
| `GET /v1/models` | the single served model |
| `POST /v1/chat/completions` | strict Chat Completions subset, JSON or SSE |
| `POST /completion` | untemplated, llama-server-shaped route used by the benchmarks |

`/completion` requires every field of `RawCompletionRequest` (`prompt`, `n_predict`, `temperature`, `top_k`, `top_p`, `seed`, `cache_prompt`, `stream`, `return_tokens`, `id_slot`, `ignore_eos`, `repeat_penalty`, `response_fields`) and rejects unknown ones, so both servers in a benchmark get an identical contract.

A full queue returns 429; a stopped or poisoned worker returns 503. Streams send a keep-alive comment every 15 seconds and end with `data: [DONE]`.

<sub>Sources: [server.rs:123-131](../../src/server.rs#L123-L131) · [server.rs:156-185](../../src/server.rs#L156-L185) · [server.rs:219-236](../../src/server.rs#L219-L236) · [server.rs:1265-1276](../../src/server.rs#L1265-L1276) · [server.rs:1088-1096](../../src/server.rs#L1088-L1096) · +1</sub>

## What /v1/chat/completions accepts

Unknown JSON fields are a 400, and known-but-unsupported fields return an OpenAI-shaped error instead of being dropped.

- `model` must equal the served model name; `n` must be 1.
- `max_completion_tokens` (or the deprecated `max_tokens`, never both) defaults to 128 and may not exceed the model context.
- `temperature` defaults to 1.0 within [0, 2]; `top_p` defaults to 1.0 within (0, 1]; `seed` must be non-negative. The server fixes `top_k` to 0 and repetition penalty to 1.0.
- `stop` takes a string or up to 4 strings.
- `stream` with `stream_options.include_usage`; `include_obfuscation=true` is rejected.
- Roles: `system`, `user`, `assistant`. `developer`, `name`, multimodal content, `logprobs`, `top_logprobs` and non-zero `frequency_penalty`/`presence_penalty` are rejected.

`usage.prompt_tokens_details.cached_tokens` reports how many prompt tokens came from the prefix cache.

<sub>Sources: [openai.rs:19-72](../../src/openai.rs#L19-L72) · [openai.rs:90-170](../../src/openai.rs#L90-L170) · [openai.rs:489-515](../../src/openai.rs#L489-L515) · [openai.rs:815-870](../../src/openai.rs#L815-L870) · [server.rs:793-802](../../src/server.rs#L793-L802) · +4</sub>

## Structured output and tool calls

Both features run through the constrained-decoding grammar in `src/grammar.rs`, which checks each candidate token's decoded bytes against a closed JSON Schema subset. Unknown schema keywords are rejected, and object schemas need `additionalProperties: false`.

- `response_format` takes only `{"type":"json_schema","json_schema":{...,"strict":true}}`. It cannot be combined with `stop`, tools or tool-call history.
- Tools: `tool_choice=auto` is not implemented and returns an error. The default for a request that declares tools is `auto`, so clients must set `tool_choice` explicitly. Accepted: a named function, `"required"` with exactly one declared tool, or `"none"`. Set `parallel_tool_calls=false`; `stop` is not allowed with a forced call. The forced call's arguments are generated under its parameters schema.
- Native tool prompts exist only for Qwen3 and Qwen3MoE adapters whose chat template hash matches a pinned profile. Other adapters fail closed in `format_agent_prompt`.

<sub>Sources: [openai.rs:517-530](../../src/openai.rs#L517-L530) · [openai.rs:572-674](../../src/openai.rs#L572-L674) · [server.rs:803-830](../../src/server.rs#L803-L830) · [grammar.rs:1-15](../../src/grammar.rs#L1-L15) · [qwen3.rs:264-277](../../src/models/qwen3.rs#L264-L277) · +3</sub>

## Startup checks and the prefix-cache flag

`EngineHandle::start` returns only after the worker has loaded the model, so `serve` fails before binding if any check fails:

1. The model must support packed ragged batches.
2. The adapter must render a health-check chat prompt through its recognised template, and that prompt must tokenize to at least one token. Checkpoints without a supported template can be used with `run --raw-prompt` but not served.
3. All `--max-active-sequences` sessions are created up front.

`serve` enables the automatic prefix cache by default (32768 tokens), and the cache cannot coexist with sliding-window attention ([How It Works](04-how-it-works.md)). A model with any `SlidingCausal` layer therefore fails to load with "sliding-attention storage is mutable". Gemma 2 and Gemma 3 normally interleave such layers, and GGUF Llama gets them when `llama.attention.sliding_window` is non-zero. Serve those models with `--prefix-cache-capacity-tokens 0`.

<sub>Sources: [engine.rs:1141-1245](../../src/engine.rs#L1141-L1245) · [runtime.rs:94-97](../../src/runtime.rs#L94-L97) · [decoder.rs:3359-3369](../../src/decoder.rs#L3359-L3369) · [decoder.rs:4319-4340](../../src/decoder.rs#L4319-L4340) · [gguf_model.rs:1322-1325](../../src/gguf_model.rs#L1322-L1325) · +3</sub>
