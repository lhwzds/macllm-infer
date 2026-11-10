---
title: 结构化输出和 tool call
source: #459c522df175
---

`crates/macllm-chat/src/grammar.rs` 中的约束 grammar 会把每个候选 token 解码后的字节与一个封闭的 JSON Schema 子集进行比对。未知的 schema 关键字会被拒绝，object schema 必须设置 `additionalProperties: false`。

- `response_format` 只接受 `{"type":"json_schema","json_schema":{...,"strict":true}}`，不能与 `stop`、工具或 tool call 历史同时使用。
- 工具：最多 128 个名称互不相同的函数。声明了工具时，`tool_choice` 默认为 `auto`；也接受 `none`、指定函数，或只声明一个工具时的 `"required"`。启用工具时不允许使用 `stop`。
- 在固定的 Qwen3、Qwen3MoE 或 Qwen3.5 模板上使用 `auto`：答案末尾可以有一个调用，由 lazy grammar 约束。在通用 GGUF 模板上，调用格式从模板中检测（Hermes、Qwen XML、Llama 3 JSON、Mistral JSON、Granite JSON、Pythonic）；每个解析出的调用都会按其声明校验；除非 `parallel_tool_calls` 允许多个，否则只返回第一个。
- 强制调用需要 `parallel_tool_calls=false`，参数在 schema 约束下生成。它适用于固定的 Qwen3/Qwen3MoE profile 和通用 Hermes 模板；Qwen3.5 会拒绝强制调用，prompt 打开了推理块的通用模板也会拒绝。
- 其他 adapter 在 `format_agent_prompt` 中直接失败（fail closed）。
