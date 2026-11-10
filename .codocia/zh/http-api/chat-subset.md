---
title: /v1/chat/completions 接受哪些字段
source: #dd0475f698da
---

未知 JSON 字段返回 400；已知但不支持的字段返回 OpenAI 格式的错误，而不是被丢弃。

- `model` 必须等于服务的模型名；`n` 必须为 1。
- `max_completion_tokens`（或已弃用的 `max_tokens`，两者不能同时出现）默认取 `--default-max-completion-tokens`，且不能超过模型上下文。
- `temperature`（范围 [0, 2]）和 `top_p`（范围 (0, 1]）默认取 `--sampling-profile` 的值，该 profile 也决定 `top_k`；`presence_penalty` 和 `frequency_penalty` 接受 [-2, 2]；`seed` 必须非负。重复惩罚固定为 1.0。
- `stop` 接受一个字符串或最多 4 个字符串。
- `stream` 支持 `stream_options.include_usage`；`include_obfuscation=true` 会被拒绝。
- `chat_template_kwargs` 只接受布尔值的 `enable_thinking`。
- 角色：`system`、`user`、`assistant`，以及工具调用历史中的 `tool`（字符串内容加上一个 `tool_call_id`，回应前一个 assistant 轮次中的某个调用）。该轮次只能包含一个调用；在检测到工具调用格式的通用模板上可以包含多个。assistant 的 `reasoning_content` 只被 Qwen3、Qwen3MoE 和 Qwen3.5 的模板接受。`function`、`developer`、`name`、多模态内容、`logprobs` 和 `top_logprobs` 都会被拒绝。

`usage.prompt_tokens_details.cached_tokens` 报告有多少 prompt token 来自 prefix cache。
