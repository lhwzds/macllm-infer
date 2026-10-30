---
title: 向量嵌入和重排序
source: #c4ab5e9737c1
---

`embed` 和 `rerank` 只支持 Safetensors 格式的 Qwen3 任务 checkpoint。`Qwen3EmbeddingProfile` 和 `Qwen3RerankerProfile` 会检查 sentence-transformers 文件（`modules.json`、`config_sentence_transformers.json`，以及 pooling 或 logit-score 模块配置）、EOS token、last-token pooling 或互不相同的 yes/no 标签 ID；随后 `validate_task_checkpoint` 要求必须是 Safetensors 格式。这些任务只能通过 CLI 使用。

- `embed --input-type query|document --input TEXT [--input TEXT ...] [--instruction TEXT]`；`--instruction` 只对 query 有效。
- `rerank --query TEXT --document TEXT [--document TEXT ...] [--instruction TEXT]`。

超过 `min(max_position_embeddings, 8192)` 个 token 的输入会被拒绝，而不是截断。输出是 stdout 上格式化后的 JSON，按输入的 `index` 排序，不回显输入。向量嵌入的每一行带有单位长度的 `embedding`；重排序的每一行带有 `margin`（`yes_logit - no_logit`）和 `yes_probability`（其 sigmoid 值）。
