---
title: 模型、设备和 K/V 参数
source: #2464ca828c83
---

所有加载模型的命令共用 `RuntimeArgs`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `-m, --model` | 必填 | `.gguf` 文件或模型目录 |
| `--device` | `metal` | `metal` 或 `cpu` |
| `--dtype` | 见下文 | 激活值使用 `f32`、`f16`、`bf16` |
| `--kv-cache-block-tokens` | `16` | 每个逻辑 K/V block 的 token 数 |
| `--kv-cache-slab-blocks` | `8` | 每次物理 slab 分配的 block 数 |
| `--kv-cache-capacity-tokens` | 一个模型上下文 | 所有序列共享的 K/V token 总数，向上取整到一个 block |

`resolve_dtype` 在 CPU 上或 GGUF 含有量化张量时选择 `f32`，否则选择 `f16`。CPU/Accelerate 后端只接受 `f32`。
