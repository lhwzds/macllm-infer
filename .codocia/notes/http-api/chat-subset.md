---
title: What /v1/chat/completions accepts
order: 20
evidence:
  - src/openai.rs:19-72 #5f874e5829db
  - src/openai.rs:90-170 #498dfcc5402d
  - src/openai.rs:489-515 #acf184826343
  - src/openai.rs:815-870 #a777a2838a38
  - src/server.rs:793-802 #ee96a4d33df1
  - src/output.rs:4 #6043f29711fb
  - src/openai.rs:1316-1326 #e610cb116f08
  - src/openai.rs:880-890 #acaca0935581
  - src/server.rs:737-753 #376f8700b532
  - src/openai.rs:1018-1057 #c2318dda9887
---

Unknown JSON fields are a 400, and known-but-unsupported fields return an OpenAI-shaped error instead of being dropped.

- `model` must equal the served model name; `n` must be 1.
- `max_completion_tokens` (or the deprecated `max_tokens`, never both) defaults to 128 and may not exceed the model context.
- `temperature` defaults to 1.0 within [0, 2]; `top_p` defaults to 1.0 within (0, 1]; `seed` must be non-negative. The server fixes `top_k` to 0 and repetition penalty to 1.0.
- `stop` takes a string or up to 4 strings.
- `stream` with `stream_options.include_usage`; `include_obfuscation=true` is rejected.
- Roles: `system`, `user`, `assistant`, and `tool` in tool-call histories (string content and a `tool_call_id` answering the preceding assistant turn's single tool call). `function`, `developer`, `name`, multimodal content, `logprobs`, `top_logprobs` and non-zero `frequency_penalty`/`presence_penalty` are rejected.

`usage.prompt_tokens_details.cached_tokens` reports how many prompt tokens came from the prefix cache.
