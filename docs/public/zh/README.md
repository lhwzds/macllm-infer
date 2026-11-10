---
title: macllm-infer
---

# macllm-infer

<sub>[English](../README.md) · **中文**</sub>

这套文档比 MacLLM Infer 的 README 更深入一层。MacLLM Infer 是一个用 Rust 编写、
Apple Silicon 优先的纯文本 LLM 运行时。文档说明当前提交的代码实际支持什么（哪些格式和架构、
哪些 CLI 参数和 HTTP 字段、哪些默认值），一个请求如何从模型加载开始，依次经过服务调度器、
paged K/V cache 和 prefix cache，最终到达 Metal kernel，以及固定的参考测试和 benchmark
manifest 如何为正确性和性能结论把关。文档只描述当前能力。

## 目录

1. [概览与支持矩阵](01-overview.md) — 支持的格式和架构 · 当前限制一览
2. [命令行用法](02-cli.md) — 模型、设备和 K/V 参数 · 用 run 生成文本 · 向量嵌入和重排序 · 下载和检查 checkpoint
3. [HTTP 服务与 API 子集](03-http-api.md) — 端点 · /v1/chat/completions 接受哪些字段 · 结构化输出和 tool call · 启动检查与 prefix cache 参数
4. [工作原理](04-how-it-works.md) — Adapter 与共享 decoder · 零拷贝加载权重 · 服务调度器 · K/V cache 布局 · 自动 prefix cache · 自定义 Metal kernel 在运行时编译
5. [Benchmark、门禁测试与贡献](05-benchmarks-and-testing.md) — 固定的参考门禁测试 · Scoreboard manifest · Prefix cache 演示与本地 bench · 添加模型家族

## 附录

- [代码结构](structure.md) — 从代码生成：功能区、依赖、讲解页

<sub>生成自 https://github.com/lhwzds/macllm-infer @ `789a9829af`</sub>
