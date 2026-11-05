---
title: macllm-infer — All pages
---

# macllm-infer

<sub>**English** · [中文](zh/all.md)</sub>

These docs go one level below the README of MacLLM Infer, a Rust, Apple
Silicon-first, text-only LLM runtime. They state what the code at this commit
actually supports (which formats and architectures, which CLI flags and HTTP
fields, which defaults), how a request flows from model loading through the
serving scheduler, paged K/V cache and prefix cache to the Metal kernels, and
how the pinned reference tests and benchmark manifests gate correctness and
performance claims. They describe current capabilities only.

<a id="overview"></a>

## Overview and Support Matrix

MacLLM Infer is a single Rust crate (`macllm-infer`, binary `macllm`) that runs text-only decoder LLMs on Apple Silicon through Candle, with Metal and Accelerate as default features. One shared decoder executes every supported family; family code is limited to thin adapters that translate a checkpoint's configuration, weight names and prompt template into a format-neutral spec. The same loaded model serves blocking CLI generation, Qwen3 embedding and reranking, and a long-lived HTTP worker. `llama.cpp` is used as the reference for GGUF parsing, greedy-token correctness and performance, never as a linked dependency.

This page states exactly what the code accepts. See [Command-Line Usage](#cli) for usage, [How It Works](#how-it-works) for the design, and [Code structure](#structure) for the module map.

<sub>Sources: [Cargo.toml:1-22](../../Cargo.toml#L1-L22) · [lib.rs:1-7](../../src/lib.rs#L1-L7) · [mod.rs:1](../../src/benchmark/mod.rs#L1)</sub>

### Supported formats and architectures

`ModelFiles::from_dir` accepts a standalone `.gguf` file or a directory. A directory holding both GGUF and Safetensors weights is rejected; GGUF wins discovery otherwise, and canonical GGUF shard sets are opened as one logical file.

The two formats do not cover the same architectures:

| Format | Dispatch key | Architectures |
| --- | --- | --- |
| GGUF v3 | `general.architecture` in `GgufModel::open` | `falcon-h1`, `gemma`, `gemma2`, `gemma3`, `granitemoe`, `lfm2`, `llama`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `smollm3` |
| Safetensors | `model_type` in `config.json` (`detect`) | `llama`, `lfm2`, `qwen2`, `qwen3` |

The 12-architecture list therefore applies to GGUF only. A Safetensors directory also needs `config.json` and `tokenizer.json`; GGUF uses its embedded tokenizer. GGUF tensor types accepted by `GgufTensorType` are F32, F16, BF16, Q4_0, Q5_0, Q5_1, Q8_0 and Q2_K through Q6_K.

<sub>Sources: [weights.rs:58-115](../../src/weights.rs#L58-L115) · [mod.rs:102-118](../../src/models/mod.rs#L102-L118) · [gguf_model.rs:271-300](../../src/gguf_model.rs#L271-L300) · [gguf.rs:283-296](../../src/gguf.rs#L283-L296)</sub>

### Current limits at a glance

Support is checkpoint-specific and fails closed. The boundaries a new user hits first:

- Embedding and reranking accept only Safetensors Qwen3 task checkpoints (`validate_task_checkpoint`) and exist only as the `embed`/`rerank` CLI commands; the HTTP router has no embedding or rerank route. See [Command-Line Usage](#cli).
- The server exposes `/healthz`, `/v1/models`, `/v1/chat/completions` and a benchmark `/completion` route. `tool_choice=auto` is rejected as not implemented; only one forced tool call is supported. See [HTTP Server and API Subset](#http-api).
- The automatic prefix cache and sliding-window attention layers are mutually exclusive: enabling the cache on a model with a sliding layer fails at load. See [How It Works](#how-it-works).
- The CPU backend requires `--dtype f32`.
- Correctness claims come from pinned per-checkpoint reference tests, for example Gemma 1 2B Q2_K and SmolLM3-3B Q4_K_M. See [Benchmarks, Gates and Contributing](#benchmarks-and-testing).

<sub>Sources: [main.rs:910-923](../../src/main.rs#L910-L923) · [server.rs:123-131](../../src/server.rs#L123-L131) · [openai.rs:626-674](../../src/openai.rs#L626-L674) · [decoder.rs:4319-4340](../../src/decoder.rs#L4319-L4340) · [backend.rs:30-37](../../src/backend.rs#L30-L37)</sub>

<a id="cli"></a>

## Command-Line Usage

The `macllm` binary has nine subcommands: `download`, `inspect`, `run`, `embed`, `rerank`, `bench`, `scoreboard`, `demo` and `serve`. All flags below are copied from the clap definitions in `src/main.rs`. There are no runtime environment variables; the `MACLLM_*` variables in the code only point the ignored reference tests at fixture files (see [Benchmarks, Gates and Contributing](#benchmarks-and-testing)). `serve` is covered in [HTTP Server and API Subset](#http-api); `bench`, `scoreboard` and `demo` in [Benchmarks, Gates and Contributing](#benchmarks-and-testing).

<sub>Sources: [main.rs:44-78](../../src/main.rs#L44-L78)</sub>

### Model, device and K/V flags

Every model-loading command shares `RuntimeArgs`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m, --model` | required | `.gguf` file or model directory |
| `--device` | `metal` | `metal` or `cpu` |
| `--dtype` | see below | `f32`, `f16`, `bf16` activations |
| `--kv-cache-block-tokens` | `16` | tokens per logical K/V block |
| `--kv-cache-slab-blocks` | `8` | blocks per physical slab allocation |
| `--kv-cache-capacity-tokens` | one model context | total K/V tokens shared by all sequences, rounded up to a block |

`resolve_dtype` picks `f32` on CPU or when the GGUF has quantized tensors, `f16` otherwise. The CPU/Accelerate backend rejects anything but `f32`.

<sub>Sources: [main.rs:80-106](../../src/main.rs#L80-L106) · [runtime.rs:34-50](../../src/runtime.rs#L34-L50) · [runtime.rs:162-175](../../src/runtime.rs#L162-L175) · [backend.rs:30-37](../../src/backend.rs#L30-L37)</sub>

### Generating text with run

`run` renders `--prompt` (plus optional `--system`) through the adapter's chat template, tokenizes it, and streams decoded text to stdout. Adapters render only templates they recognise and otherwise fail; `--raw-prompt` bypasses the template and tokenizes the prompt verbatim. `--add-special-tokens` asks the tokenizer to add BOS/EOS; leave it off for templates that already contain BOS.

Sampling defaults: `--max-new-tokens 128`, `--temperature 0.7` (0 means greedy argmax), `--top-k 20`, `--top-p 0.8`, `--repetition-penalty 1.1`, `--seed 42`.

`--prefill-token-quantum` caps prompt tokens per prefill forward. Unset, Metal uses 128 unless `--dtype bf16`; CPU and BF16 prefill in one shot. After generation, a metrics line on stderr reports prompt, cached and processed tokens, TTFT, prefill and decode tokens per second.

<sub>Sources: [main.rs:108-160](../../src/main.rs#L108-L160) · [main.rs:1220-1255](../../src/main.rs#L1220-L1255) · [main.rs:1285-1300](../../src/main.rs#L1285-L1300)</sub>

### Embedding and reranking

`embed` and `rerank` work only with Safetensors Qwen3 task checkpoints. `Qwen3EmbeddingProfile` and `Qwen3RerankerProfile` check the sentence-transformers files (`modules.json`, `config_sentence_transformers.json`, and the pooling or logit-score module config), the EOS token, last-token pooling or distinct yes/no label IDs; `validate_task_checkpoint` then requires the Safetensors format. These tasks are CLI-only.

- `embed --input-type query|document --input TEXT [--input TEXT ...] [--instruction TEXT]`; `--instruction` is valid only for queries.
- `rerank --query TEXT --document TEXT [--document TEXT ...] [--instruction TEXT]`.

Inputs longer than `min(max_position_embeddings, 8192)` tokens are rejected, not truncated. Output is pretty-printed JSON on stdout, ordered by input `index` and without echoing inputs. Embedding rows carry a unit-length `embedding`; rerank rows carry `margin` (`yes_logit - no_logit`) and `yes_probability` (its sigmoid).

<sub>Sources: [main.rs:172-232](../../src/main.rs#L172-L232) · [main.rs:808-850](../../src/main.rs#L808-L850) · [main.rs:865-923](../../src/main.rs#L865-L923) · [main.rs:925-968](../../src/main.rs#L925-L968) · [qwen3_tasks.rs:1-40](../../src/models/qwen3_tasks.rs#L1-L40) · +5</sub>

### Downloading and inspecting checkpoints

`download [REPO]` (default `Qwen/Qwen2.5-0.5B-Instruct`, output `-o models/qwen2.5-0.5b-instruct`, optional `--revision`) fetches a fixed Safetensors allow-list: config, generation and tokenizer JSON, `model*.safetensors` plus index, and the sentence-transformers sidecars. `--gguf FILENAME` instead fetches one root-level GGUF, or every shard when given the canonical first shard. Files land in a temporary staging directory, must pass `ModelFiles::from_dir`, and only then replace the destination. A `.macllm-revision` marker records the resolved commit.

`inspect -m PATH` parses metadata without loading weights and prints family, layer geometry, F16 K/V bytes per token, fixed recurrent state per sequence, maximum context, vocabulary, stop token IDs, format and GGUF tensor-type counts.

<sub>Sources: [main.rs:46-62](../../src/main.rs#L46-L62) · [hub.rs:14-47](../../src/hub.rs#L14-L47) · [hub.rs:70-120](../../src/hub.rs#L70-L120) · [main.rs:710-757](../../src/main.rs#L710-L757)</sub>

<a id="http-api"></a>

## HTTP Server and API Subset

`macllm serve` loads one model on a dedicated worker thread and serves it over HTTP/1 with axum until Ctrl-C. It binds to loopback by default and has no authentication or TLS of its own. Embedding and reranking are not served; use the CLI ([Command-Line Usage](#cli)).

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

The shared model flags (`--model`, `--device`, `--dtype`, `--kv-cache-*`) apply as in [Command-Line Usage](#cli).

<sub>Sources: [main.rs:396-429](../../src/main.rs#L396-L429) · [main.rs:1257-1283](../../src/main.rs#L1257-L1283) · [server.rs:123-150](../../src/server.rs#L123-L150)</sub>

### Endpoints

| Route | Purpose |
| --- | --- |
| `GET /healthz` | `status`, `worker` (`starting`, `ready`, `busy`, `poisoned`, `stopped`), `model`, `device`, `dtype`; 503 when unhealthy |
| `GET /v1/models` | the single served model |
| `POST /v1/chat/completions` | strict Chat Completions subset, JSON or SSE |
| `POST /completion` | untemplated, llama-server-shaped route used by the benchmarks |

`/completion` requires every field of `RawCompletionRequest` (`prompt`, `n_predict`, `temperature`, `top_k`, `top_p`, `seed`, `cache_prompt`, `stream`, `return_tokens`, `id_slot`, `ignore_eos`, `repeat_penalty`, `response_fields`) and rejects unknown ones, so both servers in a benchmark get an identical contract.

A full queue returns 429; a stopped or poisoned worker returns 503. Streams send a keep-alive comment every 15 seconds and end with `data: [DONE]`.

<sub>Sources: [server.rs:123-131](../../src/server.rs#L123-L131) · [server.rs:156-185](../../src/server.rs#L156-L185) · [server.rs:219-236](../../src/server.rs#L219-L236) · [server.rs:1265-1276](../../src/server.rs#L1265-L1276) · [server.rs:1088-1096](../../src/server.rs#L1088-L1096) · +1</sub>

### What /v1/chat/completions accepts

Unknown JSON fields are a 400, and known-but-unsupported fields return an OpenAI-shaped error instead of being dropped.

- `model` must equal the served model name; `n` must be 1.
- `max_completion_tokens` (or the deprecated `max_tokens`, never both) defaults to 128 and may not exceed the model context.
- `temperature` defaults to 1.0 within [0, 2]; `top_p` defaults to 1.0 within (0, 1]; `seed` must be non-negative. The server fixes `top_k` to 0 and repetition penalty to 1.0.
- `stop` takes a string or up to 4 strings.
- `stream` with `stream_options.include_usage`; `include_obfuscation=true` is rejected.
- Roles: `system`, `user`, `assistant`, and `tool` in tool-call histories (string content and a `tool_call_id` answering the preceding assistant turn's single tool call). `function`, `developer`, `name`, multimodal content, `logprobs`, `top_logprobs` and non-zero `frequency_penalty`/`presence_penalty` are rejected.

`usage.prompt_tokens_details.cached_tokens` reports how many prompt tokens came from the prefix cache.

<sub>Sources: [openai.rs:19-72](../../src/openai.rs#L19-L72) · [openai.rs:90-170](../../src/openai.rs#L90-L170) · [openai.rs:489-515](../../src/openai.rs#L489-L515) · [openai.rs:815-870](../../src/openai.rs#L815-L870) · [server.rs:793-802](../../src/server.rs#L793-L802) · +5</sub>

### Structured output and tool calls

Both features run through the constrained-decoding grammar in `src/grammar.rs`, which checks each candidate token's decoded bytes against a closed JSON Schema subset. Unknown schema keywords are rejected, and object schemas need `additionalProperties: false`.

- `response_format` takes only `{"type":"json_schema","json_schema":{...,"strict":true}}`. It cannot be combined with `stop`, tools or tool-call history.
- Tools: `tool_choice=auto` is not implemented and returns an error. The default for a request that declares tools is `auto`, so clients must set `tool_choice` explicitly. Accepted: a named function, `"required"` with exactly one declared tool, or `"none"`. Set `parallel_tool_calls=false`; `stop` is not allowed with a forced call. The forced call's arguments are generated under its parameters schema.
- Native tool prompts exist only for Qwen3 and Qwen3MoE adapters whose chat template hash matches a pinned profile. Other adapters fail closed in `format_agent_prompt`.

<sub>Sources: [openai.rs:517-530](../../src/openai.rs#L517-L530) · [openai.rs:572-674](../../src/openai.rs#L572-L674) · [server.rs:803-830](../../src/server.rs#L803-L830) · [grammar.rs:1-15](../../src/grammar.rs#L1-L15) · [qwen3.rs:264-277](../../src/models/qwen3.rs#L264-L277) · +3</sub>

### Startup checks and the prefix-cache flag

`EngineHandle::start` returns only after the worker has loaded the model, so `serve` fails before binding if any check fails:

1. The model must support packed ragged batches.
2. The adapter must render a health-check chat prompt through its recognised template, and that prompt must tokenize to at least one token. Checkpoints without a supported template can be used with `run --raw-prompt` but not served.
3. All `--max-active-sequences` sessions are created up front.

`serve` enables the automatic prefix cache by default (32768 tokens), and the cache cannot coexist with sliding-window attention ([How It Works](#how-it-works)). A model with any `SlidingCausal` layer therefore fails to load with "sliding-attention storage is mutable". Gemma 2 and Gemma 3 normally interleave such layers, and GGUF Llama gets them when `llama.attention.sliding_window` is non-zero. Serve those models with `--prefix-cache-capacity-tokens 0`.

<sub>Sources: [engine.rs:1141-1245](../../src/engine.rs#L1141-L1245) · [runtime.rs:94-97](../../src/runtime.rs#L94-L97) · [decoder.rs:3359-3369](../../src/decoder.rs#L3359-L3369) · [decoder.rs:4319-4340](../../src/decoder.rs#L4319-L4340) · [gguf_model.rs:1322-1325](../../src/gguf_model.rs#L1322-L1325) · +3</sub>

<a id="how-it-works"></a>

## How It Works

Every command goes through `LoadedRuntime::load`: open the files, choose an adapter, build the shared decoder from a `WeightSource`, then configure the paged K/V pool and the prefix cache. Blocking commands call the model directly. `serve` builds the same runtime inside one worker thread that owns the model for its whole life.

```mermaid
flowchart LR
  F["ModelFiles<br/>GGUF or Safetensors"] --> A["ModelAdapter<br/>src/models/*"]
  A --> S["DecoderSpec"]
  S --> D["DecoderModel<br/>src/decoder.rs"]
  W["WeightSource<br/>mmap / quantized"] --> D
  D --> KV["Paged K/V pool<br/>+ prefix cache"]
  H["axum handlers<br/>src/server.rs"] -->|EngineRequest| E["Worker thread<br/>ServingScheduler"]
  E --> D
  C["CLI run/embed/rerank"] --> D
```

<sub>Sources: [lib.rs:1-7](../../src/lib.rs#L1-L7) · [runtime.rs:62-103](../../src/runtime.rs#L62-L103) · [engine.rs:773-805](../../src/engine.rs#L773-L805)</sub>

### Adapters and the shared decoder

`ModelAdapter` is the only place family code lives. An adapter turns a config or GGUF metadata into a `RuntimeSpec` and a `DecoderSpec`, maps weight names, and renders chat or agent prompts. Every adapter's `build_model` returns a `DecoderModel`, so attention, caching, batching and sampling are written once.

A `DecoderSpec` is a list of per-layer specs. Each layer picks a `SequenceMixerSpec`: `Attention` (full or sliding causal, RoPE or NoPE), `ShortConvolution` (LFM2), `GatedDeltaNet` (Qwen3.5) or `ParallelAttentionMamba2` (Falcon-H1), plus a dense or sparse-MoE feed-forward. Hybrid models are therefore just layer lists, and the decoder implements the `CausalLm` trait the rest of the runtime uses.

`ModelAdapter::build` checks that the built model's runtime metadata matches the adapter's, and `validate` checks that the spec is internally consistent before any weight is touched.

<sub>Sources: [mod.rs:28-56](../../src/models/mod.rs#L28-L56) · [mod.rs:102-118](../../src/models/mod.rs#L102-L118) · [architecture.rs:224-229](../../src/architecture.rs#L224-L229) · [architecture.rs:1038-1064](../../src/architecture.rs#L1038-L1064) · [gemma.rs:71-76](../../src/models/gemma.rs#L71-L76) · +4</sub>

### Loading weights without copying them

`ModelFiles::weight_source` memory-maps Safetensors through Candle's `VarBuilder`. GGUF goes through the project's own parser and an mmap-backed `GgufWeightSource`, and block-quantized matrices stay compressed through model build and run as `QMatMul`. With the `metal` feature, `file_backed_quant` wraps each GGUF shard's read-only mapping in one shared Metal buffer and binds each tensor at its file offset. Candle's stock loader would copy every weight into an anonymous buffer, making the model dirty unified memory.

Model files must therefore stay unmodified and untruncated while a model built from them is alive (the loader is `unsafe`); GGUF keeps its opened handle, so replacing its path after parsing is safe. After loading, macOS is asked to return the pages freed by tokenizer and metadata construction (`malloc_zone_pressure_relief`).

<sub>Sources: [weights.rs:240-275](../../src/weights.rs#L240-L275) · [file_backed_quant.rs:1-6](../../src/file_backed_quant.rs#L1-L6) · [model_weights.rs:1-40](../../src/model_weights.rs#L1-L40) · [model_weights.rs:305-306](../../src/model_weights.rs#L305-L306) · [runtime.rs:113-128](../../src/runtime.rs#L113-L128)</sub>

### The serving scheduler

The runtime does not require models to be `Send`, so `EngineHandle::start` builds the model inside a thread named `macllm-inference` and never moves it. HTTP handlers send `InferenceJob`s over a bounded channel (`--queue-capacity`) and get back an `EngineTicket`, a request-local event stream. Dropping the ticket is how a client disconnect cancels a request.

The worker runs continuous batching in rounds. In each round it visits up to `--max-active-sequences` active sequences. Each sequence gets at most `--prefill-token-quantum` input tokens, and the round stops at `--scheduler-token-budget`. All visited lanes run as one packed ragged forward. Decode lanes cost one token, so long prompts are chunked and cannot starve running generations. A lane whose client stops draining events for `--event-send-timeout-seconds` is cancelled.

A panic, or an error escaping admission or the shared `drive_once` step (a failed forward), poisons the worker (`/healthz` reports `poisoned`) and fails active and queued requests. Request-local failures (invalid sampling, one lane's KV-cache exhaustion, incomplete JSON-schema output) fail only that request (`request_local_preflight_failure_does_not_poison_the_worker`).

<sub>Sources: [runtime.rs:52-55](../../src/runtime.rs#L52-L55) · [engine.rs:773-822](../../src/engine.rs#L773-L822) · [engine.rs:1452-1522](../../src/engine.rs#L1452-L1522) · [engine.rs:1795-1822](../../src/engine.rs#L1795-L1822) · [engine.rs:1860-1945](../../src/engine.rs#L1860-L1945) · +6</sub>

### K/V cache layout

Each decoder layer gets a `KvCachePolicy`:

- `Paged`: full-attention layers share one model-wide pool. Its blocks are `--kv-cache-block-tokens` long, allocated `--kv-cache-slab-blocks` at a time, up to `--kv-cache-capacity-tokens` in total (by default one model context for all sequences together, not per sequence). Each sequence holds a `PagedKvBlockTable` of block IDs, and one block stores K and V for every full-attention layer.
- `Sliding`: local-attention layers keep a per-sequence ring of `2 * (W - 1)` slots. A rollback restores three integers instead of copying tensors.
- `Full`: a grow-on-demand contiguous cache.

Batch growth is transactional. A `PagedKvBatchReservation` reserves blocks for every lane before a forward, and dropping it without commit rolls all of them back. Blocks can be shared between tables by reference counting; appending to a shared partial tail block copies it first (copy-on-write).

<sub>Sources: [cache.rs:116-122](../../src/cache.rs#L116-L122) · [cache.rs:162-166](../../src/cache.rs#L162-L166) · [cache.rs:1401-1410](../../src/cache.rs#L1401-L1410) · [cache.rs:1450-1460](../../src/cache.rs#L1450-L1460) · [cache.rs:2392-2397](../../src/cache.rs#L2392-L2397) · +3</sub>

### Automatic prefix cache

`TokenPrefixCache` is a cost-bounded LRU of model-state snapshots keyed by exact token IDs. Hash blocks only narrow the candidates; every hit is confirmed by a token-by-token comparison from position zero. A snapshot is stored only at each inserted prompt endpoint, never at every block, so a set of progressively longer prompts does not retain a quadratic amount of state.

After prefill, `publish_prompt_prefix` stores the prompt. Attention-only models store it truncated to a block boundary as a forked block table, which shares pages by reference count. Before the next prompt, `reset_session_for_prompt` restores the longest match into the session and reports it as `cached_tokens`.

Recurrent layers (ShortConv, Gated DeltaNet, Mamba2) need exact-endpoint snapshots of their state, because recurrent state cannot be truncated. Sliding-window rings are mutable and cannot be forked, so `validate_prefix_cache_topology` rejects any sliding layer. The cache and sliding-window attention are therefore mutually exclusive, and a non-zero capacity on such a model is a load error, not a silent no-op. Benchmark waves can opt out of publishing through `EngineRequestPolicy`.

<sub>Sources: [prefix_cache.rs:1-7](../../src/prefix_cache.rs#L1-L7) · [prefix_cache.rs:116-122](../../src/prefix_cache.rs#L116-L122) · [decoder.rs:3461-3499](../../src/decoder.rs#L3461-L3499) · [decoder.rs:3501-3540](../../src/decoder.rs#L3501-L3540) · [decoder.rs:4308-4340](../../src/decoder.rs#L4308-L4340) · +2</sub>

### Custom Metal kernels are compiled at runtime

Most tensor operations come from Candle. The project adds three Metal sources in `src/metal/`: `paged_decode.metal` (paged attention that reads K/V through a block page table, including a batched GQA-4 variant), `gated_delta_decay.metal` (Gated DeltaNet) and `mamba2_scan.metal` (Mamba2 scan). Each is embedded with `include_str!` and compiled from source by `new_library_with_source` the first time a device needs it. The pipelines are cached per Metal device `registry_id`.

`build.rs` does not touch these kernels. It only embeds `MACLLM_BUILD_PROFILE`, `MACLLM_BUILD_OPT_LEVEL`, `MACLLM_BUILD_GIT_COMMIT` and `MACLLM_BUILD_GIT_DIRTY` for benchmark provenance. A kernel compile error therefore appears at run time, on first use, and benchmarks use `--warmup-runs` to keep that compile cost out of measurements.

The paged kernel is used only when its preflight passes: batch 1, at most 128 query tokens per lane and 128 lanes, head dimension at most 256, and F32 or F16. Otherwise it returns `None` and the decoder falls back to materialized attention.

<sub>Sources: [paged_attention.rs:790](../../src/paged_attention.rs#L790) · [paged_attention.rs:1195-1226](../../src/paged_attention.rs#L1195-L1226) · [paged_attention.rs:14-47](../../src/paged_attention.rs#L14-L47) · [mamba2.rs:331](../../src/mamba2.rs#L331) · [mamba2.rs:480-491](../../src/mamba2.rs#L480-L491) · +9</sub>

<a id="benchmarks-and-testing"></a>

## Benchmarks, Gates and Contributing

Claims in this project rest on two kinds of evidence. Checkpoint-pinned reference tests prove token-level correctness against `llama.cpp` or Candle. Manifest-pinned benchmarks under `macllm scoreboard`, `macllm demo` and `macllm bench` measure performance. Every measurement command refuses to run in an unoptimized build (`ensure_optimized_measurement_execution`), so use `--release`.

<sub>Sources: [mod.rs:1](../../src/benchmark/mod.rs#L1) · [gemma_gguf_reference.rs:1-8](../../tests/gemma_gguf_reference.rs#L1-L8) · [main.rs:663-708](../../src/main.rs#L663-L708)</sub>

### Pinned reference gates

Real-model tests are `#[ignore]`d and need their fixture files. The Gemma 1 gate is typical. It pins the public file by size and SHA-256 and checks the prompt token IDs. On CPU and on Metal, it asserts that cached decode matches a fresh full forward (same argmax, maximum-difference and relative-RMS logit bounds) and that greedy generation reproduces `llama.cpp`'s token IDs. Other gates compare against `llama.cpp` goldens, Candle's independent implementations, or both.

| Architecture | Pinned fixture |
| --- | --- |
| `gemma` | Gemma 1 2B-it Q2_K (`tensorblock/gemma-2b-it-GGUF`) |
| `gemma2` / `gemma3` | Gemma 2 2B-it Q4_K_M / Gemma 3 270M-it QAT Q4_0 |
| `smollm3` | SmolLM3-3B Q4_K_M (official) |
| `llama` | SmolLM2-135M, Llama 3.2 1B Q4_K_M, TinyLlama Q2_K and Q3_K_M, TinyMistral-6x248M MoE |
| `qwen2` / `qwen3` / `qwen35` | Qwen2.5-0.5B Q4_0 / Qwen3-0.6B Q8_0 (also as 12 shards) / Qwen3.5-0.8B Q4_0 |
| `qwen3moe` / `granitemoe` | loggenix-moe Q8_0 / Granite 3.1 1B-A400M Q4_K_M |
| `lfm2` / `falcon-h1` | LFM2-350M Q4_K_M / Falcon-H1-Tiny-90M Q8_0 |
| tasks | Qwen3 Embedding and Reranker checkpoints vs Candle and Qwen's examples |

Run one with its fixture variable, for example `MACLLM_GEMMA_GGUF_MODEL=models/gemma-2b-it-q2_k cargo test --release --test gemma_gguf_reference -- --ignored --nocapture`.

<sub>Sources: [gemma_gguf_reference.rs:1-8](../../tests/gemma_gguf_reference.rs#L1-L8) · [gemma_gguf_reference.rs:27-54](../../tests/gemma_gguf_reference.rs#L27-L54) · [gemma_gguf_reference.rs:148-201](../../tests/gemma_gguf_reference.rs#L148-L201) · [gemma_gguf_reference.rs:290-312](../../tests/gemma_gguf_reference.rs#L290-L312) · [smollm3_gguf_reference.rs:1-8](../../tests/smollm3_gguf_reference.rs#L1-L8) · +16</sub>

### Scoreboard manifests

A scoreboard manifest (`schema_version` 1, see `benchmarks/manifests/`) pins six things: `host` (machine model, chip, memory, CPU architecture); `model` (path relative to the manifest, Hugging Face repository, revision, filename, size, SHA-256, architecture, quantization); `llama_cpp` (binary directory, build number, commit, GPU layers, flash attention, K/V cache types, mmap); `workload` (prompt file with its SHA-256, token count and token-ID SHA-256, decode steps, greedy sampling, concurrency widths 1/4/8, warmups, samples, prefix-cache mode); and `runtime` (device, dtype, prefill quantum, K/V settings).

`scoreboard verify --manifest PATH` loads no weights. It checks the host, the model bytes, the detected architecture, a single GGUF file, the prompt token IDs, and the commit of the `llama-server`/`llama-bench` binaries, then prints a JSON summary. Any mismatch fails. The measuring subcommands run `verify` first: `macllm-kernel`, `llama-bench`, `llama-server` (`--port`, default 18081), `macllm-serving`, and `report`, which runs every stage and writes `--json` and `--markdown`.

<sub>Sources: [main.rs:304-350](../../src/main.rs#L304-L350) · [runner.rs:72-175](../../src/benchmark/runner.rs#L72-L175) · [smollm3-3b-q4_k_m-m4pro.json](../../benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json) · [main.rs:616-661](../../src/main.rs#L616-L661)</sub>

### Prefix-cache demos and local bench

`demo prefix-cache` and `demo fair-prefix` turn a repository's `Cargo.toml` and `src/**/*.rs` (`--repository`, default `.`) into an exact 16K-token shared prefix and send it as eight concurrent requests to both runtimes. They default to ports 18085 and 18086 and write `prefix-cache-demo.{json,md}` or `fair-prefix-benchmark.{json,md}`. Results are point estimates for that host and run.

`fair-prefix` also requires the same HTTP transport and F32 K/V precision on both sides. `main.rs` passes `MAC_PREFIX_DEMO_DEFAULT_MEASURED_WAVES` (20) as its measured waves per implementation. `validate_inputs` requires at least 20 waves, a count that splits evenly across 4 counterbalanced process lifetimes, and a manifest `fair_prefix` block that matches exactly. That block must declare 5 warm-up waves per lifetime (`FAIR_WARMUP_WAVES_PER_LIFETIME`), 32-token completions, a warm prefix cache, and `samples_per_concurrency` equal to waves times 8 (160). A 3-wave steady-state window (`FAIR_STEADY_STATE_WINDOW_WAVES`) checks the warm-ups.

`bench generate|embed|rerank` is a local diagnostic without a manifest: `--runs` (default 20) measured repetitions after `--warmup-runs` (default 1), greedy decoding, and rejection of any run whose output drifts. `scripts/bench-llama-cpp.sh MODEL.gguf` runs plain `llama-bench` and reads `PROMPT_TOKENS` (512), `OUTPUT_TOKENS` (128) and `RUNS` (3).

<sub>Sources: [main.rs:352-394](../../src/main.rs#L352-L394) · [main.rs:560-573](../../src/main.rs#L560-L573) · [prefix_demo_mac.rs:39-46](../../src/benchmark/prefix_demo_mac.rs#L39-L46) · [fair_prefix.rs:59-61](../../src/benchmark/fair_prefix.rs#L59-L61) · [fair_prefix.rs:912-956](../../src/benchmark/fair_prefix.rs#L912-L956) · +5</sub>

### Adding a model family

A new family touches only its adapter and a dispatch arm. The adapter boundary is explained in [How It Works](#how-it-works), and format dispatch in [Overview and Support Matrix](#overview).

1. Add `src/models/<family>.rs` implementing `ModelAdapter`: `family`, `runtime_spec`, `decoder_spec`, `build_model` (normally `DecoderModel::load`, or `load_with_weight_names` for non-standard tensor names) and `format_chat_messages`.
2. For GGUF, add a `<family>_spec` builder with an explicit list of accepted metadata keys, plus match arms in `GgufModel::open` and `GgufModel::adapter`. For Safetensors, add a `model_type` arm in `detect`.
3. Recognise chat templates by exact identity rather than executing Jinja. The Qwen3 adapter, for example, accepts only templates whose byte length and SHA-256 match a pinned constant. Implement `agent_template_profile` only once a native tool template is verified.
4. Add an ignored gate under `tests/` that pins a public fixture by SHA-256 and compares greedy tokens with `llama.cpp`.

If a new layer type needs its own mixer, extend `SequenceMixerSpec` and the decoder, and say whether the prefix cache can snapshot it.

<sub>Sources: [mod.rs:57-100](../../src/models/mod.rs#L57-L100) · [gguf_model.rs:24-43](../../src/gguf_model.rs#L24-L43) · [gguf_model.rs:312-365](../../src/gguf_model.rs#L312-L365) · [qwen3.rs:140-168](../../src/models/qwen3.rs#L140-L168) · [qwen3.rs:246-251](../../src/models/qwen3.rs#L246-L251) · +2</sub>

<a id="structure"></a>

## Code structure

Generated by Codocia from the code on every build: areas, each area's own description, import dependencies between areas, and the pages that explain them.

In scope: 96 files, 92,879 lines; excluded `tests/fixtures/**`, `Cargo.lock`, `LICENSE-MIT`, `README_CN.md`, `.gitignore`

### Files

**Repository root**

- (root files) — 304 lines, 3 files. An Apple Silicon-first Rust LLM inference runtime\
  Hub files: `build.rs`\
  Explained in: [Overview and Support Matrix](#overview) (1), [How It Works](#how-it-works) (1)
- `benchmarks/` — 144 lines, 3 files\
  Explained in: [Benchmarks, Gates and Contributing](#benchmarks-and-testing) (2)
- `scripts/` — 25 lines, 1 file\
  Hub files: `bench-llama-cpp.sh`\
  Explained in: [Benchmarks, Gates and Contributing](#benchmarks-and-testing) (1)
- `tests/` — 7,137 lines, 23 files\
  Explained in: [Benchmarks, Gates and Contributing](#benchmarks-and-testing) (2)

**`src/`**

- (files) — 56,303 lines, 30 files. `MacLLM Infer` is an Apple Silicon-first Rust LLM inference runtime.\
  Hub files: `lib.rs` `architecture.rs` `gguf.rs`\
  Explained in: [Overview and Support Matrix](#overview) (3), [Command-Line Usage](#cli) (5), [HTTP Server and API Subset](#http-api) (5), [How It Works](#how-it-works) (7), [Benchmarks, Gates and Contributing](#benchmarks-and-testing) (4)
- `benchmark/` — 23,014 lines, 19 files. Reproducible, fail-closed `MacLLM` versus `llama.cpp` scoreboard primitives.\
  Hub files: `schema.rs` `mod.rs` `prefix_demo_mac.rs`\
  Explained in: [Overview and Support Matrix](#overview) (1), [Benchmarks, Gates and Contributing](#benchmarks-and-testing) (3)
- `metal/` — 652 lines, 3 files\
  Explained in: [How It Works](#how-it-works) (1)
- `models/` — 5,300 lines, 14 files\
  Hub files: `qwen3.rs` `llama.rs` `qwen3_tasks.rs`\
  Explained in: [Overview and Support Matrix](#overview) (1), [Command-Line Usage](#cli) (1), [HTTP Server and API Subset](#http-api) (1), [How It Works](#how-it-works) (1), [Benchmarks, Gates and Contributing](#benchmarks-and-testing) (1)


### Dependencies

A → B means files in A import B; the number counts file-level imports. Showing the strongest 4 of 4 edges.

```mermaid
flowchart LR
  a3["src/*"]
  a4["src/benchmark/"]
  a6["src/models/"]
  a7["tests/"]
  a7 -->|22| a3
  a6 -->|8| a3
  a4 -->|2| a3
  a3 -->|1| a4
```
