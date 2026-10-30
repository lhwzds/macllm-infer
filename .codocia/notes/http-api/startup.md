---
title: Startup checks and the prefix-cache flag
order: 40
evidence:
  - src/engine.rs:1141-1245 #b7c25b5aad06
  - src/runtime.rs:94-97 #b5a3fdb889a1
  - src/decoder.rs:3359-3369 #75231b283fba
  - src/decoder.rs:4319-4340 #b05eaac38643
  - src/gguf_model.rs:1322-1325 #05b21046d651
  - src/gguf_model.rs:1355-1362 #38d6c51ca06f
  - src/gguf_model.rs:1783-1792 #43f02d70a55b
  - src/gguf_model.rs:1412-1428 #dea327920bdd
---

`EngineHandle::start` returns only after the worker has loaded the model, so `serve` fails before binding if any check fails:

1. The model must support packed ragged batches.
2. The adapter must render a health-check chat prompt through its recognised template, and that prompt must tokenize to at least one token. Checkpoints without a supported template can be used with `run --raw-prompt` but not served.
3. All `--max-active-sequences` sessions are created up front.

`serve` enables the automatic prefix cache by default (32768 tokens), and the cache cannot coexist with sliding-window attention ([[how-it-works]]). A model with any `SlidingCausal` layer therefore fails to load with "sliding-attention storage is mutable". Gemma 2 and Gemma 3 normally interleave such layers, and GGUF Llama gets them when `llama.attention.sliding_window` is non-zero. Serve those models with `--prefix-cache-capacity-tokens 0`.
