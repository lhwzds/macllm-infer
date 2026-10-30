---
title: /v1/chat/completions 接受哪些字段
source: #3ee458a97cf9
---

未知 JSON 字段返回 400；已知但不支持的字段返回 OpenAI 格式的错误，而不是被丢弃。

- `model` 必须等于服务的模型名；`n` 必须为 1。
- `max_completion_tokens`（或已弃用的 `max_tokens`，两者不能同时出现）默认为 128，且不能超过模型上下文。
- `temperature` 默认 1.0，范围 [0, 2]；`top_p` 默认 1.0，范围 (0, 1]；`seed` 必须非负。服务端把 `top_k` 固定为 0，重复惩罚固定为 1.0。
- `stop` 接受一个字符串或最多 4 个字符串。
- `stream` 支持 `stream_options.include_usage`；`include_obfuscation=true` 会被拒绝。
- 角色：`system`、`user`、`assistant`。`developer`、`name`、多模态内容、`logprobs`、`top_logprobs` 和非零的 `frequency_penalty`/`presence_penalty` 都会被拒绝。

`usage.prompt_tokens_details.cached_tokens` 报告有多少 prompt token 来自 prefix cache。
