---
title:
source: #7930839f3961
---

MacLLM Infer 是一个以 Apple Silicon 为先的 Rust workspace，通过 Candle 运行纯文本 decoder LLM。主 crate 是 `macllm-infer`（二进制为 `macllm`）；对话和工具调用处理、K/V cache 以及自定义 kernel 分别放在 `macllm-chat`、`macllm-kv` 和 `macllm-kernels` 这几个 crate 中。Metal 和 Accelerate 是默认 feature；可选的 `cuda` feature 通过 `--device cuda` 支持 NVIDIA GPU。所有支持的模型家族都由同一个共享 decoder 执行；家族相关代码仅限于很薄的 adapter，负责把 checkpoint 的配置、权重名称和 prompt 模板转换成与格式无关的 spec。同一个已加载的模型既用于阻塞式 CLI 生成、Qwen3 向量嵌入和重排序，也用于长期运行的 HTTP worker。`llama.cpp` 被用作 GGUF 解析、贪心解码 token 的正确性和性能的参考，但从不作为链接依赖。

本页准确说明代码接受什么。用法见 [[cli]]，设计见 [[how-it-works]]，模块结构见 [[structure]]。
