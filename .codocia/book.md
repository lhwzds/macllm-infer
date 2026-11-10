---
title: macllm-infer
repo: ..
out: docs/public
commit: 789a9829af81f30a826d470a725dfb25075103c6
remote: https://github.com/lhwzds/macllm-infer
lang: en
max_pages: 8
max_page_words: 900
max_note_words: 180
pages:
  - overview: Overview and Support Matrix
  - cli: Command-Line Usage
  - http-api: HTTP Server and API Subset
  - how-it-works: How It Works
  - benchmarks-and-testing: Benchmarks, Gates and Contributing
ignore:
  - tests/fixtures/**
  - Cargo.lock
  - LICENSE-MIT
  - README_CN.md
  - .gitignore
langs:
  - zh
translations:
  - README_CN.md: README.md #dbfb8fa5fde7
---

These docs go one level below the README of MacLLM Infer, a Rust, Apple
Silicon-first, text-only LLM runtime. They state what the code at this commit
actually supports (which formats and architectures, which CLI flags and HTTP
fields, which defaults), how a request flows from model loading through the
serving scheduler, paged K/V cache and prefix cache to the Metal kernels, and
how the pinned reference tests and benchmark manifests gate correctness and
performance claims. They describe current capabilities only.
