---
title: 模型、设备和 K/V 参数
source: #83a814ef1580
---

所有加载模型的命令共用 `RuntimeArgs`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `-m, --model` | 必填 | `.gguf` 文件或模型目录 |
| `--device` | `metal` | `metal`、`cuda` 或 `cpu`；不带 Metal 的构建如果启用了 CUDA 则默认 `cuda`，否则默认 `cpu` |
| `--dtype` | 见下文 | 激活值使用 `f32`、`f16`、`bf16` |
| `--kv-cache-block-tokens` | `16` | 每个逻辑 K/V block 的 token 数 |
| `--kv-cache-slab-blocks` | `8` | 每次物理 slab 分配的 block 数 |
| `--kv-cache-capacity-tokens` | 一个模型上下文 | 所有序列共享的 K/V token 总数，向上取整到一个 block |
| `--prefill-attention` | `sdpa` | 多行 prefill 分块使用 Candle 的融合 SDPA kernel；`native` 保留分页 Metal kernel 及其稠密回退路径 |

`resolve_dtype` 在 CPU 上或 GGUF 含有量化张量时选择 `f32`，否则选择 `f16`。CPU/Accelerate 后端只接受 `f32`；Metal 和 CUDA 三种都接受。
