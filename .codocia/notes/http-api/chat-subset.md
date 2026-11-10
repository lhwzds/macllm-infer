---
title: What /v1/chat/completions accepts
order: 20
evidence:
  - crates/macllm-chat/src/openai.rs:54-108 #162f108129f8
  - crates/macllm-chat/src/openai.rs:136-230 #f7a44f069c00
  - crates/macllm-chat/src/openai.rs:600-630 #826b0439669c
  - crates/macllm-chat/src/output.rs:4 #6043f29711fb
  - crates/macllm-chat/src/openai.rs:937-1035 #f77fab3c9cc6
  - crates/macllm-chat/src/openai.rs:1120-1185 #8b3b1cbcb7b3
  - src/server.rs:866-910 #095d29d1c720
  - src/server.rs:1044-1050 #09a63f84a249
  - src/server.rs:1087-1094 #b787416f08ea
  - src/server.rs:1234-1245 #e1ec225d2d4d
  - crates/macllm-chat/src/openai.rs:1477-1487 #e610cb116f08
---

Unknown JSON fields are a 400, and known-but-unsupported fields return an OpenAI-shaped error instead of being dropped.

- `model` must equal the served model name; `n` must be 1.
- `max_completion_tokens` (or the deprecated `max_tokens`, never both) defaults to `--default-max-completion-tokens` and may not exceed the model context.
- `temperature` in [0, 2] and `top_p` in (0, 1] default to the `--sampling-profile` values, which also fix `top_k`; `presence_penalty` and `frequency_penalty` take [-2, 2]; `seed` must be non-negative. Repetition penalty is fixed to 1.0.
- `stop` takes a string or up to 4 strings.
- `stream` with `stream_options.include_usage`; `include_obfuscation=true` is rejected.
- `chat_template_kwargs` accepts only a boolean `enable_thinking`.
- Roles: `system`, `user`, `assistant`, and `tool` in tool-call histories (string content and a `tool_call_id` answering a call of the preceding assistant turn). That turn holds exactly one call, or several on generic templates with a detected tool-call format. Assistant `reasoning_content` is accepted only for the Qwen3, Qwen3MoE and Qwen3.5 model families, whichever template renders them. `function`, `developer`, `name`, multimodal content, `logprobs` and `top_logprobs` are rejected.

`usage.prompt_tokens_details.cached_tokens` reports how many prompt tokens came from the prefix cache.
