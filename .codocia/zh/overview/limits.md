---
title: 当前限制一览
source: #c031cc8e2408
---

支持范围以具体 checkpoint 为边界，不满足条件时直接失败（fail closed）。新用户最先遇到的边界：

- 向量嵌入和重排序只接受 Safetensors 格式的 Qwen3 任务 checkpoint（`validate_task_checkpoint`），并且只以 `embed`/`rerank` CLI 命令的形式提供；HTTP 路由中没有向量嵌入或重排序路由。见 [[cli]]。
- 服务暴露 `/healthz`、`/v1/models`、`/v1/chat/completions` 和一个用于 benchmark 的 `/completion` 路由。`tool_choice=auto` 会以未实现为由被拒绝；只支持强制调用单个工具。见 [[http-api]]。
- 自动 prefix cache 与 sliding-window attention 层互斥：在含有 sliding 层的模型上启用该缓存会在加载时失败。见 [[how-it-works]]。
- CPU 后端要求 `--dtype f32`。
- 正确性结论来自按 checkpoint 固定的参考测试，例如 Gemma 1 2B Q2_K 和 SmolLM3-3B Q4_K_M。见 [[benchmarks-and-testing]]。
