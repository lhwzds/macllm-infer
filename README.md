# MacLLM Infer

[English](README.md) | [Simplified Chinese](README_CN.md)

MacLLM Infer is an Apple Silicon-first, text-only LLM inference runtime written in
Rust. It runs generation, embedding, and reranking on one shared decoder engine.

The core is architecture-general: model-specific configuration, chat templates,
and weight mapping live in adapters, while loading, tokenization, batching, caching,
sampling, and serving are reused. `llama.cpp` is the reference baseline for GGUF
compatibility, output correctness, memory use, and performance—not an ABI target.

> MacLLM Infer is an early-stage runtime for research and local agent workloads. Its
> support claims are checkpoint-specific, not promises for an entire model family.

## Highlights

- Apple Metal and CPU execution, with quantized GGUF support.
- GGUF v3 single files and canonical shards, plus selected Safetensors profiles.
- Dispatch for 12 architectures: `llama`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35`,
  `gemma`, `gemma2`, `gemma3`, `granitemoe`, `lfm2`, `falcon-h1`, and `smollm3`.
- Full and sliding-window GQA, RoPE/NoPE, Sparse MoE, ShortConv, Gated DeltaNet,
  Mamba2, and paged KV caching.
- Continuous batching and an exact-token automatic prefix cache with bounded LRU,
  refcounting, and copy-on-write.
- Generation, embedding, and reranking through the `run`, `embed`, and `rerank`
  commands, backed by the same runtime.
- A text-only Chat Completions-compatible API subset with constrained JSON
  generation, stop sequences, native tool calls for supported Qwen3/Qwen3MoE
  profiles (forced, or `tool_choice=auto` with at most one grammar-constrained
  call per turn), and cached-token accounting.
- Reproducible MacLLM/llama.cpp benchmarks with pinned model and runtime provenance.

The current validated vertical slices include SmolLM3-3B Q4_K_M generation and
Qwen3 Embedding/Reranker 0.6B. Model weights are not included in this repository.

## Quick Start

```bash
cargo build --release

./target/release/macllm run \
  --model /path/to/model.gguf \
  --prompt 'The capital of France is' \
  --max-new-tokens 32

./target/release/macllm embed \
  --model /path/to/Qwen3-Embedding-0.6B \
  --input-type query \
  --input 'What is the capital of China?'

./target/release/macllm rerank \
  --model /path/to/Qwen3-Reranker-0.6B \
  --query 'What is the capital of China?' \
  --document 'The capital of China is Beijing.' \
  --document 'Gravity attracts bodies.'

./target/release/macllm serve \
  --model /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080
```

Metal is the default device; pass `--device cpu` to use the CPU. For an unknown
chat template, pass `--raw-prompt` explicitly. MacLLM does not execute arbitrary
Jinja templates or silently guess a template.

For agent clients, `serve --default-max-completion-tokens N` sets the budget of
requests that omit one (default 128), and `--disable-thinking` renders Qwen3
prompts with `enable_thinking=false`; a request may instead send
`chat_template_kwargs: {"enable_thinking": false}`.

## Benchmarking

The checked-in manifests pin the host, model bytes and SHA-256, prompt token IDs,
quantization, and llama.cpp build. The sample manifests target the recorded M4 Pro
environment; other hosts require a matching manifest.

```bash
./target/release/macllm scoreboard verify \
  --manifest benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json
```

The `fair-prefix` lane controls transport, K/V precision, prefix length, concurrency,
output length, and token parity. Its output is a point estimate for one exact run
and host—not a general claim that MacLLM is faster than llama.cpp. Run
`./target/release/macllm demo fair-prefix --help` for the complete invocation.

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- \
  -D warnings -A clippy::too_many_lines -A clippy::doc_markdown
cargo test --all-targets
```

Real-model integration tests are marked `#[ignore]` because they require pinned
model fixtures and, for Metal tests, an Apple GPU.

## Scope and Security

- The runtime is text-only; vision, audio, OCR, and multimodal models are out of
  scope today.
- Model directories should be treated as trusted and immutable while loaded.
- The server defaults to loopback. Public deployment requires an authenticated,
  TLS-enabled, rate-limited reverse proxy.
- Source code is MIT licensed. Model weights remain subject to their own licenses
  and terms of use.
