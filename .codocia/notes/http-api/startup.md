---
title: Startup checks and the prefix-cache flag
order: 40
evidence:
  - src/engine.rs:863-867 #7e222780d077
  - src/engine.rs:1224-1239 #6b46661f3ce1
  - src/engine.rs:1269-1360 #31fec79830d7
  - src/models/mod.rs:115-117 #c9d67efdd94d
  - src/server.rs:233-240 #250a50219dee
  - src/decoder.rs:3371-3381 #75231b283fba
  - src/decoder.rs:4331-4352 #b05eaac38643
  - src/gguf_model.rs:1401-1404 #05b21046d651
  - src/gguf_model.rs:1434-1441 #38d6c51ca06f
  - src/gguf_model.rs:1862-1871 #43f02d70a55b
  - src/gguf_model.rs:1491-1507 #dea327920bdd
---

`EngineHandle::start` returns only after the worker has loaded the model, so `serve` fails before binding if any check fails:

1. The model must support packed ragged batches.
2. The adapter must render a health-check chat prompt through its pinned or generic template, and that prompt must tokenize to at least one token. Checkpoints without a usable template can be used with `run --raw-prompt` but not served.
3. A server-wide `--disable-thinking` must be honoured by the template.
4. All `--max-active-sequences` sessions are created up front.

`serve` then checks that `--default-max-completion-tokens` fits the model context.

`serve` enables the automatic prefix cache by default (32768 tokens), and the cache cannot coexist with sliding-window attention ([[how-it-works]]). A model with any `SlidingCausal` layer therefore fails to load with "sliding-attention storage is mutable". Gemma 2 and Gemma 3 normally interleave such layers, and GGUF Llama gets them when `llama.attention.sliding_window` is non-zero. Serve those models with `--prefix-cache-capacity-tokens 0`.
