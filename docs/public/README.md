---
title: macllm-infer
---

# macllm-infer

<sub>**English** · [中文](zh/README.md)</sub>

These docs go one level below the README of MacLLM Infer, a Rust, Apple
Silicon-first, text-only LLM runtime. They state what the code at this commit
actually supports (which formats and architectures, which CLI flags and HTTP
fields, which defaults), how a request flows from model loading through the
serving scheduler, paged K/V cache and prefix cache to the Metal kernels, and
how the pinned reference tests and benchmark manifests gate correctness and
performance claims. They describe current capabilities only.

## Contents

1. [Overview and Support Matrix](01-overview.md) — Supported formats and architectures · Current limits at a glance
2. [Command-Line Usage](02-cli.md) — Model, device and K/V flags · Generating text with run · Embedding and reranking · Downloading and inspecting checkpoints
3. [HTTP Server and API Subset](03-http-api.md) — Endpoints · What /v1/chat/completions accepts · Structured output and tool calls · Startup checks and the prefix-cache flag
4. [How It Works](04-how-it-works.md) — Adapters and the shared decoder · Loading weights without copying them · The serving scheduler · K/V cache layout · Automatic prefix cache · Custom Metal kernels are compiled at runtime
5. [Benchmarks, Gates and Contributing](05-benchmarks-and-testing.md) — Pinned reference gates · Scoreboard manifests · Prefix-cache demos and local bench · Adding a model family

## Appendix

- [Code structure](structure.md) — generated from the code: areas, dependencies, pages

<sub>Generated from https://github.com/lhwzds/macllm-infer @ `15bfbd64e8`</sub>
