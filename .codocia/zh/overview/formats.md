---
title: 支持的格式和架构
source: #637d49d44d26
---

`ModelFiles::from_dir` 接受单个 `.gguf` 文件或一个目录。同时包含 GGUF 和 Safetensors 权重的目录会被拒绝；其他情况下发现文件时 GGUF 优先，规范的 GGUF 分片集合会作为一个逻辑文件打开。

两种格式覆盖的架构并不相同：

| 格式 | Dispatch 键 | 架构 |
| --- | --- | --- |
| GGUF v3 | `GgufModel::open` 中的 `general.architecture` | `falcon-h1`、`gemma`、`gemma2`、`gemma3`、`granitemoe`、`lfm2`、`llama`、`qwen2`、`qwen3`、`qwen3moe`、`qwen35`、`smollm3` |
| Safetensors | `config.json` 中的 `model_type`（`detect`） | `llama`、`lfm2`、`qwen2`、`qwen3` |

因此 12 种架构的列表只适用于 GGUF。Safetensors 目录还需要 `config.json` 和 `tokenizer.json`；GGUF 使用内嵌的 tokenizer。`GgufTensorType` 接受的 GGUF 张量类型为 F32、F16、BF16、Q4_0、Q5_0、Q5_1、Q8_0，以及 Q2_K 到 Q6_K。
