---
title: Scoreboard manifests
order: 20
evidence:
  - src/main.rs:348-394 #076441cfeed9
  - src/benchmark/runner.rs:72-175 #59a0b4bfc1bd
  - benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json #3bd03b67c52f
  - src/main.rs:716-761 #db0d30ebce46
---

A scoreboard manifest (`schema_version` 1, see `benchmarks/manifests/`) pins six things: `host` (machine model, chip, memory, CPU architecture); `model` (path relative to the manifest, Hugging Face repository, revision, filename, size, SHA-256, architecture, quantization); `llama_cpp` (binary directory, build number, commit, GPU layers, flash attention, K/V cache types, mmap); `workload` (prompt file with its SHA-256, token count and token-ID SHA-256, decode steps, greedy sampling, concurrency widths 1/4/8, warmups, samples, prefix-cache mode); and `runtime` (device, dtype, prefill quantum, K/V settings).

`scoreboard verify --manifest PATH` loads no weights. It checks the host, the model bytes, the detected architecture, a single GGUF file, the prompt token IDs, and the commit of the `llama-server`/`llama-bench` binaries, then prints a JSON summary. Any mismatch fails. The measuring subcommands run `verify` first: `macllm-kernel`, `llama-bench`, `llama-server` (`--port`, default 18081), `macllm-serving`, and `report`, which runs every stage and writes `--json` and `--markdown`.
