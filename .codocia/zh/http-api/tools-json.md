---
title: 结构化输出和 tool call
source: #40d0d4eca7dc
---

这两项功能都通过 `src/grammar.rs` 中的受约束解码 grammar 实现，它会把每个候选 token 解码后的字节与一个封闭的 JSON Schema 子集进行比对。未知的 schema 关键字会被拒绝，object schema 必须设置 `additionalProperties: false`。

- `response_format` 只接受 `{"type":"json_schema","json_schema":{...,"strict":true}}`，不能与 `stop`、工具或 tool call 历史同时使用。
- 工具：`tool_choice=auto` 尚未实现，会返回错误。声明了工具的请求默认是 `auto`，因此客户端必须显式设置 `tool_choice`。可接受的值：指定函数、只声明一个工具时的 `"required"`，或 `"none"`。需设置 `parallel_tool_calls=false`；强制调用时不允许使用 `stop`。强制调用的参数在其 parameters schema 的约束下生成。
- 只有对话模板哈希与固定 profile 匹配的 Qwen3 和 Qwen3MoE adapter 才有 native tool prompt。其他 adapter 在 `format_agent_prompt` 中直接失败（fail closed）。
