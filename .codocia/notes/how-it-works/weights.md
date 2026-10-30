---
title: Loading weights without copying them
order: 20
evidence:
  - src/weights.rs:240-275 #edd28a180f78
  - src/file_backed_quant.rs:1-6 #e74119a66673
  - src/model_weights.rs:1-40 #c6c5317e0d35
  - src/model_weights.rs:305-306 #06a0a2c8df30
  - src/runtime.rs:113-128 #3f7ebb3c986e
---

`ModelFiles::weight_source` memory-maps Safetensors through Candle's `VarBuilder`. GGUF goes through the project's own parser and an mmap-backed `GgufWeightSource`, and block-quantized matrices stay compressed through model build and run as `QMatMul`. With the `metal` feature, `file_backed_quant` wraps each GGUF shard's read-only mapping in one shared Metal buffer and binds each tensor at its file offset. Candle's stock loader would instead copy every weight into an anonymous buffer, turning the whole model into dirty unified memory.

Because of this, model files must not be modified while a model is loaded; the loader is `unsafe` for that reason. After loading, macOS is asked to return the pages freed by tokenizer and metadata construction (`malloc_zone_pressure_relief`).
