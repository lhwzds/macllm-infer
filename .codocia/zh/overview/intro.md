---
title: 
source: #ab930a7fddfa
---

MacLLM Infer 是一个单独的 Rust crate（`macllm-infer`，二进制为 `macllm`），通过 Candle 在 Apple Silicon 上运行纯文本 decoder LLM，默认 feature 为 Metal 和 Accelerate。所有支持的模型家族都由同一个共享 decoder 执行；家族相关代码仅限于很薄的 adapter，负责把 checkpoint 的配置、权重名称和 prompt 模板转换成与格式无关的 spec。同一个已加载的模型既用于阻塞式 CLI 生成、Qwen3 向量嵌入和重排序，也用于长期运行的 HTTP worker。`llama.cpp` 被用作 GGUF 解析、贪心解码 token 的正确性和性能的参考，但从不作为链接依赖。

本页准确说明代码接受什么。用法见 [[cli]]，设计见 [[how-it-works]]，模块结构见 [[structure]]。
