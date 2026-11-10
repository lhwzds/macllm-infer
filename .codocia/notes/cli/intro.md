---
title:
order: 0
evidence:
  - src/main.rs:62-96 #6989fce2c111
---

The `macllm` binary has nine subcommands: `download`, `inspect`, `run`, `embed`, `rerank`, `bench`, `scoreboard`, `demo` and `serve`. All flags below are copied from the clap definitions in `src/main.rs`. There are no runtime environment variables; the `MACLLM_*` variables in the code only point the ignored reference tests at fixture files (see [[benchmarks-and-testing]]). `serve` is covered in [[http-api]]; `bench`, `scoreboard` and `demo` in [[benchmarks-and-testing]].
