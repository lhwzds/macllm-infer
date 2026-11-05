---
title: Pinned reference gates
order: 10
evidence:
  - tests/gemma_gguf_reference.rs:1-8 #552695cb68a6
  - tests/gemma_gguf_reference.rs:27-54 #432d5e94f9b0
  - tests/gemma_gguf_reference.rs:148-201 #287f0023d83b
  - tests/gemma_gguf_reference.rs:290-312 #dad3cedaf8cb
  - tests/smollm3_gguf_reference.rs:1-8 #bbaffe11272a
  - tests/tinyllama_lower_bit_gguf_reference.rs:1-9 #d519d8e473e9
  - tests/gemma2_gguf_reference.rs:1-8 #314b966a3af7
  - tests/gemma3_gguf_reference.rs:1-7 #a3b8f12ea4ac
  - tests/qwen3_gguf_reference.rs:1-5 #f1e449bfd0c2
  - tests/qwen35_gguf_reference.rs:1-6 #6b327bdbf010
  - tests/falcon_h1_gguf_reference.rs:1-9 #9fbfc87f19b6
  - tests/lfm2_gguf_reference.rs:1-5 #010b85ba7b83
  - tests/granite_moe_gguf_reference.rs:1-10 #76c5ea56503f
  - tests/qwen3_moe_gguf_reference.rs:1-9 #118a19d7df59
  - tests/mixtral_gguf_reference.rs:1-8 #ee4369682084
  - tests/llama3_gguf_reference.rs:1-4 #c235f2f92ed8
  - tests/qwen3_tasks_reference.rs:1-7 #e036d10b55a0
  - tests/gguf_reference.rs:1-5 #a14855f0b937
  - tests/qwen_gguf_reference.rs:1-10 #5ed1fc560dc8
  - tests/sharded_gguf_reference.rs:1-5 #0b2a66d9f595
  - tests/gemma_gguf_reference.rs:129-146 #a773fc14c5db
---

Real-model tests are `#[ignore]`d and need their fixture files. The Gemma 1 gate is typical. It pins the public file by size and SHA-256 and checks the prompt token IDs. On CPU and on Metal, it asserts that cached decode matches a fresh full forward (same argmax, maximum-difference and relative-RMS logit bounds) and that greedy generation reproduces `llama.cpp`'s token IDs. Other gates compare against `llama.cpp` goldens, Candle's independent implementations, or both.

| Architecture | Pinned fixture |
| --- | --- |
| `gemma` | Gemma 1 2B-it Q2_K (`tensorblock/gemma-2b-it-GGUF`) |
| `gemma2` / `gemma3` | Gemma 2 2B-it Q4_K_M / Gemma 3 270M-it QAT Q4_0 |
| `smollm3` | SmolLM3-3B Q4_K_M (official) |
| `llama` | SmolLM2-135M, Llama 3.2 1B Q4_K_M, TinyLlama Q2_K and Q3_K_M, TinyMistral-6x248M MoE |
| `qwen2` / `qwen3` / `qwen35` | Qwen2.5-0.5B Q4_0 / Qwen3-0.6B Q8_0 (also as 12 shards) / Qwen3.5-0.8B Q4_0 |
| `qwen3moe` / `granitemoe` | loggenix-moe Q8_0 / Granite 3.1 1B-A400M Q4_K_M |
| `lfm2` / `falcon-h1` | LFM2-350M Q4_K_M / Falcon-H1-Tiny-90M Q8_0 |
| tasks | Qwen3 Embedding and Reranker checkpoints vs Candle and Qwen's examples |

Run one with its fixture variable, for example `MACLLM_GEMMA_GGUF_MODEL=models/gemma-2b-it-q2_k cargo test --release --test gemma_gguf_reference -- --ignored --nocapture`.
