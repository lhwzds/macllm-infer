---
title: Structured output and tool calls
order: 30
evidence:
  - crates/macllm-chat/src/grammar.rs:1-20 #20aa647ee6ad
  - crates/macllm-chat/src/openai.rs:232-247 #ab92ec209a39
  - crates/macllm-chat/src/openai.rs:632-650 #602e8b820b95
  - crates/macllm-chat/src/openai.rs:652-689 #0ad0b112e8f5
  - crates/macllm-chat/src/openai.rs:691-795 #7a4d96696099
  - crates/macllm-chat/src/tool_format.rs:14-21 #70e3fce3ff15
  - crates/macllm-chat/src/tool_format.rs:50-88 #a4eddc61384d
  - src/server.rs:866-879 #658592057edb
  - src/server.rs:955-973 #564ac9ed7f9d
  - src/server.rs:1052-1069 #1ee1566b3e40
  - src/server.rs:1107-1177 #ab00a7a08adc
  - src/server.rs:1178-1222 #897337783364
  - src/server.rs:1760-1788 #a237e25e41a7
  - src/models/mod.rs:83-89 #435936025d66
  - src/models/mod.rs:243-262 #bd65100dd8c6
  - src/models/qwen3.rs:284-307 #cca128be08db
  - src/models/qwen35.rs:153-175 #b937578df188
  - src/gguf_model.rs:378-386 #225852538526
---

The constraint grammar in `crates/macllm-chat/src/grammar.rs` checks each candidate token's decoded bytes against a closed JSON Schema subset. Unknown schema keywords are rejected, and object schemas need `additionalProperties: false`.

- `response_format` takes only `{"type":"json_schema","json_schema":{...,"strict":true}}`. It cannot be combined with `stop`, tools or tool-call history.
- Tools: up to 128 uniquely named functions. With tools, `tool_choice` defaults to `auto`; `none`, a named function, or `"required"` with exactly one tool are also accepted. `stop` is not allowed while tools are enabled.
- `auto` on a pinned Qwen3, Qwen3MoE or Qwen3.5 template: the answer may end in one call, constrained by a lazy grammar. On a generic GGUF template, the call format is detected from the template (Hermes, Qwen XML, Llama 3 JSON, Mistral JSON, Granite JSON, Pythonic); every parsed call is checked against its declaration; only the first is returned unless `parallel_tool_calls` allows more.
- Forced calls need `parallel_tool_calls=false` and generate the arguments under the schema. They work on pinned Qwen3/Qwen3MoE profiles and generic Hermes templates; Qwen3.5 refuses them, as does a generic template whose prompt opens a reasoning block.
- Other adapters fail closed in `format_agent_prompt`.
