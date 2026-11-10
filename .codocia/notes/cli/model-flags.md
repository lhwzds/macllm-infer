---
title: Model, device and K/V flags
order: 10
evidence:
  - src/main.rs:36-48 #64f5ed49625e
  - src/main.rs:114-150 #040594124a2d
  - src/runtime.rs:37-40 #94c9fb5f1bba
  - src/runtime.rs:167-180 #2a6eaa18eca5
  - src/backend.rs:61-71 #304892e4bac1
---

Every model-loading command shares `RuntimeArgs`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m, --model` | required | `.gguf` file or model directory |
| `--device` | `metal` | `metal`, `cuda` or `cpu`; builds without Metal default to `cuda` if built with it, else `cpu` |
| `--dtype` | see below | `f32`, `f16`, `bf16` activations |
| `--kv-cache-block-tokens` | `16` | tokens per logical K/V block |
| `--kv-cache-slab-blocks` | `8` | blocks per physical slab allocation |
| `--kv-cache-capacity-tokens` | one model context | total K/V tokens shared by all sequences, rounded up to a block |
| `--prefill-attention` | `sdpa` | multi-row prefill chunks use Candle's fused SDPA kernel; `native` keeps the paged Metal kernel with its dense fallback |

`resolve_dtype` picks `f32` on CPU or when the GGUF has quantized tensors, `f16` otherwise. The CPU/Accelerate backend rejects anything but `f32`; Metal and CUDA accept all three.
