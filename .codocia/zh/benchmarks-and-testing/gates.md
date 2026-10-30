---
title: 固定的参考门禁测试
source: #d4959fdc420b
---

真实模型测试标记为 `#[ignore]`，需要对应的 fixture 文件。Gemma 1 的门禁测试很有代表性：它按大小和 SHA-256 固定公开文件，并检查 prompt token ID。在 CPU 和 Metal 上，它都断言带缓存的 decode 与全新的完整前向计算一致（argmax 相同，logit 的最大差值和相对 RMS 在界限内），并且贪心生成能复现 `llama.cpp` 的 token ID。其他门禁测试则与 `llama.cpp` 的 golden 输出、Candle 的独立实现之一或两者进行比较。

| 架构 | 固定的 fixture |
| --- | --- |
| `gemma` | Gemma 1 2B-it Q2_K（`tensorblock/gemma-2b-it-GGUF`） |
| `gemma2` / `gemma3` | Gemma 2 2B-it Q4_K_M / Gemma 3 270M-it QAT Q4_0 |
| `smollm3` | SmolLM3-3B Q4_K_M（官方） |
| `llama` | SmolLM2-135M、Llama 3.2 1B Q4_K_M、TinyLlama Q2_K 和 Q3_K_M、TinyMistral-6x248M MoE |
| `qwen2` / `qwen3` / `qwen35` | Qwen2.5-0.5B Q4_0 / Qwen3-0.6B Q8_0（也有 12 分片版本）/ Qwen3.5-0.8B Q4_0 |
| `qwen3moe` / `granitemoe` | loggenix-moe Q8_0 / Granite 3.1 1B-A400M Q4_K_M |
| `lfm2` / `falcon-h1` | LFM2-350M Q4_K_M / Falcon-H1-Tiny-90M Q8_0 |
| 任务 | Qwen3 Embedding 和 Reranker checkpoint，对照 Candle 和 Qwen 的示例 |

运行某个门禁测试时需设置其 fixture 变量，例如 `MACLLM_GEMMA_GGUF_MODEL=models/gemma-2b-it-q2_k cargo test --release --test gemma_gguf_reference -- --ignored --nocapture`。
