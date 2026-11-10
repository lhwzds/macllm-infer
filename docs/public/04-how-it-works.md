---
title: How It Works
---

# How It Works

<sub>**English** · [中文](zh/04-how-it-works.md)</sub>

Every command goes through `LoadedRuntime::load`: open the files, choose an adapter, build the shared decoder from a `WeightSource`, then configure the paged K/V pool, the prefix cache and the prefill attention mode. Blocking commands call the model directly. `serve` builds the same runtime inside one worker thread that owns the model for its whole life.

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

<sub>Sources: [lib.rs:1-7](../../src/lib.rs#L1-L7) · [runtime.rs:66-103](../../src/runtime.rs#L66-L103) · [engine.rs:849-862](../../src/engine.rs#L849-L862)</sub>

## Adapters and the shared decoder

`ModelAdapter` is the only place family code lives. An adapter turns a config or GGUF metadata into a `RuntimeSpec` and a `DecoderSpec`, maps weight names, and renders chat or agent prompts. Unpinned GGUF templates go to a generic Jinja renderer (`attach_prompt_tokens` for Qwen3/Qwen3MoE/Qwen3.5, `with_jinja_fallback` for the rest, Qwen2 included). Every adapter's `build_model` returns a `DecoderModel`, so attention, caching, batching and sampling are written once.

A `DecoderSpec` is a list of per-layer specs. Each layer picks a `SequenceMixerSpec`: `Attention` (full or sliding causal, RoPE or NoPE), `ShortConvolution` (LFM2), `GatedDeltaNet` (Qwen3.5) or `ParallelAttentionMamba2` (Falcon-H1), plus a dense or sparse-MoE feed-forward. Hybrid models are just layer lists; the decoder implements the `CausalLm` trait the rest of the runtime uses.

`ModelAdapter::build` checks the built model's runtime metadata against the adapter's, and `validate` checks the spec's consistency before any weight is touched.

<sub>Sources: [mod.rs:35-66](../../src/models/mod.rs#L35-L66) · [mod.rs:128-155](../../src/models/mod.rs#L128-L155) · [mod.rs:609-627](../../src/models/mod.rs#L609-L627) · [gguf_model.rs:365-387](../../src/gguf_model.rs#L365-L387) · [mod.rs:698-714](../../src/models/mod.rs#L698-L714) · +7</sub>

## Loading weights without copying them

`ModelFiles::weight_source` memory-maps Safetensors through Candle's `VarBuilder`. GGUF goes through the project's own parser and an mmap-backed `GgufWeightSource`, and block-quantized matrices stay compressed through model build and run as `QMatMul`. With the `metal` feature, `file_backed_quant` wraps each GGUF shard's read-only mapping in one shared Metal buffer and binds each tensor at its file offset. Candle's stock loader would copy every weight into an anonymous buffer, making the model dirty unified memory.

Model files must therefore stay unmodified and untruncated while a model built from them is alive (the loader is `unsafe`); GGUF keeps its opened handle, so replacing its path after parsing is safe. After loading, macOS is asked to return the pages freed by tokenizer and metadata construction (`malloc_zone_pressure_relief`).

<sub>Sources: [weights.rs:240-275](../../src/weights.rs#L240-L275) · [file_backed_quant.rs:1-6](../../src/file_backed_quant.rs#L1-L6) · [model_weights.rs:1-40](../../src/model_weights.rs#L1-L40) · [model_weights.rs:305-306](../../src/model_weights.rs#L305-L306) · [runtime.rs:118-133](../../src/runtime.rs#L118-L133)</sub>

## The serving scheduler

Models need not be `Send`: `EngineHandle::start` builds the model inside the `macllm-inference` thread and never moves it. HTTP handlers send `InferenceJob`s over a bounded channel (`--queue-capacity`) and get back an `EngineTicket`, a request-local event stream. Dropping the ticket cancels the request.

The worker runs continuous batching in rounds. In each round it visits up to `--max-active-sequences` active sequences. Each sequence gets at most `--prefill-token-quantum` input tokens, and the round stops at `--scheduler-token-budget`. All visited lanes run as one packed ragged forward. Decode lanes cost one token, so long prompts cannot starve running generations. A lane whose client stops draining events for `--event-send-timeout-seconds` is cancelled.

A panic, or an error escaping admission or the shared `drive_once` step (a failed forward), poisons the worker (`/healthz` reports `poisoned`) and fails active and queued requests. Request-local failures (invalid sampling, one lane's KV-cache exhaustion, incomplete JSON-schema output) fail only that request.

<sub>Sources: [runtime.rs:56-59](../../src/runtime.rs#L56-L59) · [engine.rs:829-869](../../src/engine.rs#L829-L869) · [engine.rs:1578-1648](../../src/engine.rs#L1578-L1648) · [engine.rs:1938-1965](../../src/engine.rs#L1938-L1965) · [engine.rs:2003-2088](../../src/engine.rs#L2003-L2088) · +6</sub>

## K/V cache layout

Each decoder layer gets a `KvCachePolicy`:

- `Paged`: full-attention layers share one model-wide pool. Its blocks are `--kv-cache-block-tokens` long, allocated `--kv-cache-slab-blocks` at a time, up to `--kv-cache-capacity-tokens` in total (by default one model context for all sequences together, not per sequence). Each sequence holds a `PagedKvBlockTable` of block IDs, and one block stores K and V for every full-attention layer.
- `Sliding`: local-attention layers keep a per-sequence ring of `2 * (W - 1)` slots. A rollback restores three integers instead of copying tensors.
- `Full`: a grow-on-demand contiguous cache.

Batch growth is transactional. A `PagedKvBatchReservation` reserves blocks for every lane before a forward, and dropping it without commit rolls all of them back. Blocks can be shared between tables by reference counting; appending to a shared partial tail block copies it first (copy-on-write).

<sub>Sources: [cache.rs:119-125](../../crates/macllm-kv/src/cache.rs#L119-L125) · [cache.rs:165-169](../../crates/macllm-kv/src/cache.rs#L165-L169) · [cache.rs:1432-1441](../../crates/macllm-kv/src/cache.rs#L1432-L1441) · [cache.rs:1481-1491](../../crates/macllm-kv/src/cache.rs#L1481-L1491) · [cache.rs:2423-2428](../../crates/macllm-kv/src/cache.rs#L2423-L2428) · +3</sub>

## Automatic prefix cache

`TokenPrefixCache` is a cost-bounded LRU of model-state snapshots keyed by exact token IDs. Hash blocks only narrow the candidates; every hit is confirmed by a token-by-token comparison from position zero. A snapshot is stored only at each inserted prompt endpoint, never at every block, so a set of progressively longer prompts does not retain a quadratic amount of state.

After prefill, `publish_prompt_prefix` stores the prompt. Attention-only models store it truncated to a block boundary as a forked block table, which shares pages by reference count. Before the next prompt, `reset_session_for_prompt` restores the longest match into the session and reports it as `cached_tokens`.

Recurrent layers (ShortConv, Gated DeltaNet, Mamba2) need exact-endpoint snapshots of their state, because recurrent state cannot be truncated. Sliding-window rings are mutable and cannot be forked, so `validate_prefix_cache_topology` rejects any sliding layer. The cache and sliding-window attention are therefore mutually exclusive, and a non-zero capacity on such a model is a load error, not a silent no-op. Benchmark waves can opt out of publishing through `EngineRequestPolicy`.

<sub>Sources: [prefix_cache.rs:1-7](../../crates/macllm-kv/src/prefix_cache.rs#L1-L7) · [prefix_cache.rs:116-122](../../crates/macllm-kv/src/prefix_cache.rs#L116-L122) · [decoder.rs:3473-3511](../../src/decoder.rs#L3473-L3511) · [decoder.rs:3513-3552](../../src/decoder.rs#L3513-L3552) · [decoder.rs:4320-4352](../../src/decoder.rs#L4320-L4352) · +2</sub>

## Custom Metal kernels are compiled at runtime

Beyond Candle's operations, `macllm-kernels` adds three Metal sources in `crates/macllm-kernels/src/metal/`: `paged_decode.metal` (paged attention that reads K/V through a block page table, including a batched GQA-4 variant), `gated_delta_decay.metal` (Gated DeltaNet) and `mamba2_scan.metal` (Mamba2 scan). Each is embedded with `include_str!` and compiled from source by `new_library_with_source` the first time a device needs it. The pipelines are cached per Metal device `registry_id`.

`build.rs` never touches the kernels; it only embeds build and Git provenance for benchmarks. A kernel compile error therefore appears on first use, and benchmarks use `--warmup-runs` to keep the compile cost out of measurements.

The paged kernel is used only when its preflight passes: batch 1, at most 128 query tokens per lane and 128 lanes, head dimension at most 256, F32 or F16, and a Metal device with Tier 2 argument buffers. Otherwise it returns `None` and the decoder falls back to materialized attention.

<sub>Sources: [paged_attention.rs:800](../../crates/macllm-kernels/src/paged_attention.rs#L800) · [paged_attention.rs:1205-1236](../../crates/macllm-kernels/src/paged_attention.rs#L1205-L1236) · [paged_attention.rs:5-55](../../crates/macllm-kernels/src/paged_attention.rs#L5-L55) · [paged_attention.rs:892-896](../../crates/macllm-kernels/src/paged_attention.rs#L892-L896) · [mamba2.rs:332](../../crates/macllm-kernels/src/mamba2.rs#L332) · +10</sub>
