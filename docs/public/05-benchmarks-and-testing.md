---
title: Benchmarks, Gates and Contributing
---

# Benchmarks, Gates and Contributing

<sub>**English** · [中文](zh/05-benchmarks-and-testing.md)</sub>

Claims in this project rest on two kinds of evidence. Checkpoint-pinned reference tests prove token-level correctness against `llama.cpp` or Candle. Manifest-pinned benchmarks under `macllm scoreboard`, `macllm demo` and `macllm bench` measure performance. Every measurement command refuses to run in an unoptimized build (`ensure_optimized_measurement_execution`), so use `--release`.

<sub>Sources: [mod.rs:1](../../src/benchmark/mod.rs#L1) · [gemma_gguf_reference.rs:1-8](../../tests/gemma_gguf_reference.rs#L1-L8) · [main.rs:763-808](../../src/main.rs#L763-L808)</sub>

## Pinned reference gates

Real-model tests are `#[ignore]`d and need their fixture files. The Gemma 1 gate is typical. It pins the public file by size and SHA-256 and checks the prompt token IDs. On CPU and on Metal, it asserts that cached decode matches a fresh full forward (same argmax, maximum-difference and relative-RMS logit bounds) and that greedy generation reproduces `llama.cpp`'s token IDs. Other gates compare against `llama.cpp` goldens, Candle's independent implementations, or both.

| Architecture | Pinned fixture |
| --- | --- |
| `gemma` | Gemma 1 2B-it Q2_K (`tensorblock/gemma-2b-it-GGUF`) |
| `gemma2` / `gemma3` | Gemma 2 2B-it Q4_K_M / Gemma 3 270M-it QAT Q4_0 |
| `smollm3` | SmolLM3-3B Q4_K_M (official) |
| `llama` | SmolLM2-135M, Llama 3.2 1B Q4_K_M, TinyLlama Q2_K and Q3_K_M, TinyMistral-6x248M MoE |
| `qwen2` / `qwen3` / `qwen35` | Qwen2.5-0.5B Q4_0 / Qwen3-0.6B Q8_0 (also as 12 shards) and Qwen3-4B Q4_K_M / Qwen3.5-0.8B Q4_0 |
| `qwen3moe` / `granitemoe` | loggenix-moe Q8_0 / Granite 3.1 1B-A400M Q4_K_M |
| `lfm2` / `falcon-h1` | LFM2-350M Q4_K_M / Falcon-H1-Tiny-90M Q8_0 |
| tasks | Qwen3 Embedding and Reranker checkpoints vs Candle and Qwen's examples |

Run one with its fixture variable, for example `MACLLM_GEMMA_GGUF_MODEL=models/gemma-2b-it-q2_k cargo test --release --test gemma_gguf_reference -- --ignored --nocapture`.

<sub>Sources: [gemma_gguf_reference.rs:1-8](../../tests/gemma_gguf_reference.rs#L1-L8) · [gemma_gguf_reference.rs:27-54](../../tests/gemma_gguf_reference.rs#L27-L54) · [gemma_gguf_reference.rs:148-201](../../tests/gemma_gguf_reference.rs#L148-L201) · [gemma_gguf_reference.rs:290-312](../../tests/gemma_gguf_reference.rs#L290-L312) · [smollm3_gguf_reference.rs:1-8](../../tests/smollm3_gguf_reference.rs#L1-L8) · +17</sub>

## Scoreboard manifests

A scoreboard manifest (`schema_version` 1, see `benchmarks/manifests/`) pins six things: `host` (machine model, chip, memory, CPU architecture); `model` (path relative to the manifest, Hugging Face repository, revision, filename, size, SHA-256, architecture, quantization); `llama_cpp` (binary directory, build number, commit, GPU layers, flash attention, K/V cache types, mmap); `workload` (prompt file with its SHA-256, token count and token-ID SHA-256, decode steps, greedy sampling, concurrency widths 1/4/8, warmups, samples, prefix-cache mode); and `runtime` (device, dtype, prefill quantum, K/V settings).

`scoreboard verify --manifest PATH` loads no weights. It checks the host, the model bytes, the detected architecture, a single GGUF file, the prompt token IDs, and the commit of the `llama-server`/`llama-bench` binaries, then prints a JSON summary. Any mismatch fails. The measuring subcommands run `verify` first: `macllm-kernel`, `llama-bench`, `llama-server` (`--port`, default 18081), `macllm-serving`, and `report`, which runs every stage and writes `--json` and `--markdown`.

<sub>Sources: [main.rs:348-394](../../src/main.rs#L348-L394) · [runner.rs:72-175](../../src/benchmark/runner.rs#L72-L175) · [smollm3-3b-q4_k_m-m4pro.json](../../benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json) · [main.rs:716-761](../../src/main.rs#L716-L761)</sub>

## Prefix-cache demos and local bench

`demo prefix-cache` and `demo fair-prefix` turn a repository's `Cargo.toml` and `src/**/*.rs` (`--repository`, default `.`) into an exact 16K-token shared prefix and send it as eight concurrent requests to both runtimes. They default to ports 18085 and 18086 and write `prefix-cache-demo.{json,md}` or `fair-prefix-benchmark.{json,md}`. Results are point estimates for that host and run.

`fair-prefix` also requires the same HTTP transport and F32 K/V precision on both sides. `main.rs` passes `MAC_PREFIX_DEMO_DEFAULT_MEASURED_WAVES` (20) as its measured waves per implementation. `validate_inputs` requires at least 20 waves, a count that splits evenly across 4 counterbalanced process lifetimes, and a manifest `fair_prefix` block that matches exactly. That block must declare 5 warm-up waves per lifetime (`FAIR_WARMUP_WAVES_PER_LIFETIME`), 32-token completions, a warm prefix cache, and `samples_per_concurrency` equal to waves times 8 (160). A 3-wave steady-state window (`FAIR_STEADY_STATE_WINDOW_WAVES`) checks the warm-ups.

`bench generate|embed|rerank` is a local diagnostic without a manifest: `--runs` (default 20) measured repetitions after `--warmup-runs` (default 1), greedy decoding, and rejection of any run whose output drifts. `scripts/bench-llama-cpp.sh MODEL.gguf` runs plain `llama-bench` and reads `PROMPT_TOKENS` (512), `OUTPUT_TOKENS` (128) and `RUNS` (3).

<sub>Sources: [main.rs:396-438](../../src/main.rs#L396-L438) · [main.rs:660-673](../../src/main.rs#L660-L673) · [prefix_demo_mac.rs:39-46](../../src/benchmark/prefix_demo_mac.rs#L39-L46) · [fair_prefix.rs:59-61](../../src/benchmark/fair_prefix.rs#L59-L61) · [fair_prefix.rs:912-956](../../src/benchmark/fair_prefix.rs#L912-L956) · +5</sub>

## Adding a model family

A new family touches only its adapter and a dispatch arm. The adapter boundary is explained in [How It Works](04-how-it-works.md), and format dispatch in [Overview and Support Matrix](01-overview.md).

1. Add `src/models/<family>.rs` implementing `ModelAdapter`: `family`, `runtime_spec`, `decoder_spec`, `build_model` (normally `DecoderModel::load`, or `load_with_weight_names` for non-standard tensor names) and `format_chat_messages`.
2. For GGUF, add a `<family>_spec` builder with an explicit list of accepted metadata keys, plus match arms in `GgufModel::open` and `GgufModel::architecture_adapter`. For Safetensors, add a `model_type` arm in `detect`.
3. Pin known chat templates by exact identity: the Qwen3 classifier in `crates/macllm-chat/src/chat_templates/` matches only byte length and SHA-256. An unpinned GGUF template falls back to the generic Jinja renderer ([How It Works](04-how-it-works.md)). Implement `agent_template_profile` only once a native tool template is verified.
4. Add an ignored gate under `tests/` that pins a public fixture by SHA-256 and compares greedy tokens with `llama.cpp`.

If a new layer type needs its own mixer, extend `SequenceMixerSpec` and the decoder, and say whether the prefix cache can snapshot it.

<sub>Sources: [gguf_model.rs:64-83](../../src/gguf_model.rs#L64-L83) · [gguf_model.rs:391-435](../../src/gguf_model.rs#L391-L435) · [qwen3.rs:42-67](../../crates/macllm-chat/src/chat_templates/qwen3.rs#L42-L67) · [mod.rs:76-80](../../src/models/mod.rs#L76-L80) · [qwen3.rs:223-228](../../src/models/qwen3.rs#L223-L228) · +2</sub>
