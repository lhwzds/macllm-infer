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

`ModelFiles::weight_source` memory-maps Safetensors through Candle's `VarBuilder`. GGUF goes through the project's own parser and an mmap-backed `GgufWeightSource`, and block-quantized matrices stay compressed through model build and run as `QMatMul`. With the `metal` feature, `file_backed_quant` wraps each GGUF shard's read-only mapping in one shared Metal buffer and binds each tensor at its file offset. Candle's stock loader would copy every weight into an anonymous buffer, making the model dirty unified memory.

Model files must therefore stay unmodified and untruncated while a model built from them is alive (the loader is `unsafe`); GGUF keeps its opened handle, so replacing its path after parsing is safe. After loading, macOS is asked to return the pages freed by tokenizer and metadata construction (`malloc_zone_pressure_relief`).
