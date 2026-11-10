---
title: 概览与支持矩阵
---

# 概览与支持矩阵

<sub>[English](../01-overview.md) · **中文**</sub>

MacLLM Infer 是一个以 Apple Silicon 为先的 Rust workspace，通过 Candle 运行纯文本 decoder LLM。主 crate 是 `macllm-infer`（二进制为 `macllm`）；对话和工具调用处理、K/V cache 以及自定义 kernel 分别放在 `macllm-chat`、`macllm-kv` 和 `macllm-kernels` 这几个 crate 中。Metal 和 Accelerate 是默认 feature；可选的 `cuda` feature 通过 `--device cuda` 支持 NVIDIA GPU。所有支持的模型家族都由同一个共享 decoder 执行；家族相关代码仅限于很薄的 adapter，负责把 checkpoint 的配置、权重名称和 prompt 模板转换成与格式无关的 spec。同一个已加载的模型既用于阻塞式 CLI 生成、Qwen3 向量嵌入和重排序，也用于长期运行的 HTTP worker。`llama.cpp` 被用作 GGUF 解析、贪心解码 token 的正确性和性能的参考，但从不作为链接依赖。

本页准确说明代码接受什么。用法见 [命令行用法](02-cli.md)，设计见 [工作原理](04-how-it-works.md)，模块结构见 [代码结构](structure.md)。

<sub>来源: [Cargo.toml:1-22](../../../Cargo.toml#L1-L22) · [Cargo.toml:46-71](../../../Cargo.toml#L46-L71) · [lib.rs:1-7](../../../src/lib.rs#L1-L7) · [lib.rs:40-45](../../../src/lib.rs#L40-L45) · [backend.rs:34-58](../../../src/backend.rs#L34-L58) · +1</sub>

## 支持的格式和架构

`ModelFiles::from_dir` 接受单个 `.gguf` 文件或一个目录。同时包含 GGUF 和 Safetensors 权重的目录会被拒绝；`macllm download` 标记为未下载完成的目录也会被拒绝。其他情况下发现文件时 GGUF 优先，规范的 GGUF 分片集合会作为一个逻辑文件打开。

两种格式覆盖的架构并不相同：

| 格式 | Dispatch 键 | 架构 |
| --- | --- | --- |
| GGUF v3 | `GgufModel::open` 中的 `general.architecture` | `falcon-h1`、`gemma`、`gemma2`、`gemma3`、`granitemoe`、`lfm2`、`llama`、`qwen2`、`qwen3`、`qwen3moe`、`qwen35`、`smollm3` |
| Safetensors | `config.json` 中的 `model_type`（`detect`） | `llama`、`lfm2`、`qwen2`、`qwen3` |

因此 12 种架构的列表只适用于 GGUF。Safetensors 目录还需要 `config.json` 和 `tokenizer.json`；GGUF 使用内嵌的 tokenizer。`GgufTensorType` 接受的 GGUF 张量类型为 F32、F16、BF16、Q4_0、Q5_0、Q5_1、Q8_0，以及 Q2_K 到 Q6_K。

<sub>来源: [weights.rs:58-115](../../../src/weights.rs#L58-L115) · [snapshot.rs:83-96](../../../src/snapshot.rs#L83-L96) · [mod.rs:698-714](../../../src/models/mod.rs#L698-L714) · [gguf_model.rs:316-335](../../../src/gguf_model.rs#L316-L335) · [gguf.rs:283-297](../../../src/gguf.rs#L283-L297)</sub>

## 当前限制一览

支持范围以具体 checkpoint 为边界，不满足条件时直接失败（fail closed）。新用户最先遇到的边界：

- 向量嵌入和重排序只接受 Safetensors 格式的 Qwen3 任务 checkpoint（`validate_task_checkpoint`），并且只以 `embed`/`rerank` CLI 命令的形式提供；HTTP 路由中没有向量嵌入或重排序路由。见 [命令行用法](02-cli.md)。
- 服务暴露 `/healthz`、`/v1/models`、`/v1/chat/completions` 和一个用于 benchmark 的 `/completion` 路由。模型能做哪些工具调用取决于它的对话模板。见 [HTTP 服务与 API 子集](03-http-api.md)。
- 自动 prefix cache 与 sliding-window attention 层互斥：在含有 sliding 层的模型上启用该缓存会在加载时失败。见 [工作原理](04-how-it-works.md)。
- CPU 后端要求 `--dtype f32`。
- 正确性结论来自按 checkpoint 固定的参考测试，例如 Gemma 1 2B Q2_K 和 SmolLM3-3B Q4_K_M。见 [Benchmark、门禁测试与贡献](05-benchmarks-and-testing.md)。

<sub>来源: [main.rs:1010-1023](../../../src/main.rs#L1010-L1023) · [server.rs:224-228](../../../src/server.rs#L224-L228) · [decoder.rs:4331-4352](../../../src/decoder.rs#L4331-L4352) · [backend.rs:61-71](../../../src/backend.rs#L61-L71)</sub>
