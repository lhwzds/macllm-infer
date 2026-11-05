---
title: Model, device and K/V flags
order: 10
evidence:
  - src/main.rs:80-106 #1d447c9eea9b
  - src/runtime.rs:34-50 #9a6860ed6964
  - src/runtime.rs:162-175 #2a6eaa18eca5
  - src/backend.rs:30-37 #02796efe9930
---

Every model-loading command shares `RuntimeArgs`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m, --model` | required | `.gguf` file or model directory |
| `--device` | `metal` | `metal` or `cpu` |
| `--dtype` | see below | `f32`, `f16`, `bf16` activations |
| `--kv-cache-block-tokens` | `16` | tokens per logical K/V block |
| `--kv-cache-slab-blocks` | `8` | blocks per physical slab allocation |
| `--kv-cache-capacity-tokens` | one model context | total K/V tokens shared by all sequences, rounded up to a block |

`resolve_dtype` picks `f32` on CPU or when the GGUF has quantized tensors, `f16` otherwise. The CPU/Accelerate backend rejects anything but `f32`.
