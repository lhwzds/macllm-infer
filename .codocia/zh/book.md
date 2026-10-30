---
title: macllm-infer
source: #b57a88fb5d88
pages:
  - overview: 概览与支持矩阵
  - cli: 命令行用法
  - http-api: HTTP 服务与 API 子集
  - how-it-works: 工作原理
  - benchmarks-and-testing: Benchmark、门禁测试与贡献
---

这套文档比 MacLLM Infer 的 README 更深入一层。MacLLM Infer 是一个用 Rust 编写、
Apple Silicon 优先的纯文本 LLM 运行时。文档说明当前提交的代码实际支持什么（哪些格式和架构、
哪些 CLI 参数和 HTTP 字段、哪些默认值），一个请求如何从模型加载开始，依次经过服务调度器、
paged K/V cache 和 prefix cache，最终到达 Metal kernel，以及固定的参考测试和 benchmark
manifest 如何为正确性和性能结论把关。文档只描述当前能力。
