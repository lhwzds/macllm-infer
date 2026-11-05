---
title: Structured output and tool calls
order: 30
evidence:
  - src/openai.rs:517-530 #bae2ca0df033
  - src/openai.rs:572-674 #8e349805bb66
  - src/server.rs:803-830 #2a7708656845
  - src/grammar.rs:1-15 #d57629491b3d
  - src/models/qwen3.rs:264-277 #3259fd4f5e83
  - src/models/qwen3_moe.rs:79-92 #4d5452523d91
  - src/models/mod.rs:61-71 #2035af77ceef
  - src/openai.rs:324-356 #29db9e647d2d
---

Both features run through the constrained-decoding grammar in `src/grammar.rs`, which checks each candidate token's decoded bytes against a closed JSON Schema subset. Unknown schema keywords are rejected, and object schemas need `additionalProperties: false`.

- `response_format` takes only `{"type":"json_schema","json_schema":{...,"strict":true}}`. It cannot be combined with `stop`, tools or tool-call history.
- Tools: `tool_choice=auto` is not implemented and returns an error. The default for a request that declares tools is `auto`, so clients must set `tool_choice` explicitly. Accepted: a named function, `"required"` with exactly one declared tool, or `"none"`. Set `parallel_tool_calls=false`; `stop` is not allowed with a forced call. The forced call's arguments are generated under its parameters schema.
- Native tool prompts exist only for Qwen3 and Qwen3MoE adapters whose chat template hash matches a pinned profile. Other adapters fail closed in `format_agent_prompt`.
