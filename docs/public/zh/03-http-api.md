---
title: HTTP 服务与 API 子集
---

# HTTP 服务与 API 子集

<sub>[English](../03-http-api.md) · **中文**</sub>

`macllm serve` 在一个专用 worker 线程上加载一个模型，并用 axum 通过 HTTP/1 提供服务，直到按下 Ctrl-C。它默认绑定 loopback，自身不带认证或 TLS；向量嵌入和重排序只通过 CLI 提供。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8080` | 绑定地址 |
| `--served-model-name` | 模型路径的文件名 | 客户端作为 `model` 发送的 ID |
| `--queue-capacity` | `8` | 等待活跃槽位的请求数 |
| `--max-active-sequences` | `2` | 在 worker 上轮转的序列数 |
| `--scheduler-token-budget` | `256` | 每轮调度的模型输入 token 数 |
| `--prefill-token-quantum` | `128` | 每个序列每轮的 prompt token 数 |
| `--event-capacity` | `16` | 每个请求的事件通道大小 |
| `--event-send-timeout-seconds` | `30` | 客户端停止读取时取消请求 |
| `--prefix-cache-capacity-tokens` | `32768` | prefix cache 预算；`0` 表示禁用 |
| `--default-max-completion-tokens` | `128` | 未设置输出长度的对话请求的预算；必须不超过模型上下文 |
| `--disable-thinking` | 关闭 | 请求未设置时使用 `enable_thinking=false`；模板必须使用该选项 |
| `--sampling-profile` | `openai` | 未指定的采样字段：`openai`（1.0 / 1.0，无 top-k），或按 thinking 模式取模型卡推荐值的 `qwen3` / `qwen35` |
| `--reasoning-format` | `deepseek` | 以 `reasoning_content` 返回 thinking；`none` 把它留在答案中 |

共用的模型参数（`--model`、`--device`、`--dtype`、`--kv-cache-*`）与 [命令行用法](02-cli.md) 中相同。

<sub>来源: [main.rs:440-491](../../../src/main.rs#L440-L491) · [main.rs:1357-1392](../../../src/main.rs#L1357-L1392) · [server.rs:142-170](../../../src/server.rs#L142-L170) · [server.rs:232-262](../../../src/server.rs#L232-L262)</sub>

## 端点

| 路由 | 用途 |
| --- | --- |
| `GET /healthz` | `status`、`worker`（`starting`、`ready`、`busy`、`poisoned`、`stopped`）、`model`、`device`（实际运行的后端：`metal`、`cuda` 或 `cpu`）、`dtype`；不健康时返回 503 |
| `GET /v1/models` | 唯一的服务模型 |
| `POST /v1/chat/completions` | 严格的 Chat Completions 子集，JSON 或 SSE |
| `POST /completion` | 不套模板、与 llama-server 形状一致的路由，供 benchmark 使用 |

`/completion` 要求提供 `RawCompletionRequest` 的每一个字段（`prompt`、`n_predict`、`temperature`、`top_k`、`top_p`、`seed`、`cache_prompt`、`stream`、`return_tokens`、`id_slot`、`ignore_eos`、`repeat_penalty`、`response_fields`），并拒绝未知字段，这样 benchmark 中的两个服务端遵循完全相同的约定。

队列已满时返回 429；worker 已停止或处于 poisoned 状态时返回 503。流式响应每 15 秒发送一条 keep-alive 注释，并以 `data: [DONE]` 结束。

<sub>来源: [server.rs:224-228](../../../src/server.rs#L224-L228) · [server.rs:264-299](../../../src/server.rs#L264-L299) · [engine.rs:1287-1292](../../../src/engine.rs#L1287-L1292) · [backend.rs:74-88](../../../src/backend.rs#L74-L88) · [server.rs:327-344](../../../src/server.rs#L327-L344) · +3</sub>

## /v1/chat/completions 接受哪些字段

未知 JSON 字段返回 400；已知但不支持的字段返回 OpenAI 格式的错误，而不是被丢弃。

- `model` 必须等于服务的模型名；`n` 必须为 1。
- `max_completion_tokens`（或已弃用的 `max_tokens`，两者不能同时出现）默认取 `--default-max-completion-tokens`，且不能超过模型上下文。
- `temperature`（范围 [0, 2]）和 `top_p`（范围 (0, 1]）默认取 `--sampling-profile` 的值，该 profile 也决定 `top_k`；`presence_penalty` 和 `frequency_penalty` 接受 [-2, 2]；`seed` 必须非负。重复惩罚固定为 1.0。
- `stop` 接受一个字符串或最多 4 个字符串。
- `stream` 支持 `stream_options.include_usage`；`include_obfuscation=true` 会被拒绝。
- `chat_template_kwargs` 只接受布尔值的 `enable_thinking`。
- 角色：`system`、`user`、`assistant`，以及工具调用历史中的 `tool`（字符串内容加上一个 `tool_call_id`，回应前一个 assistant 轮次中的某个调用）。该轮次只能包含一个调用；在检测到工具调用格式的通用模板上可以包含多个。assistant 的 `reasoning_content` 只被 Qwen3、Qwen3MoE 和 Qwen3.5 的模板接受。`function`、`developer`、`name`、多模态内容、`logprobs` 和 `top_logprobs` 都会被拒绝。

`usage.prompt_tokens_details.cached_tokens` 报告有多少 prompt token 来自 prefix cache。

<sub>来源: [openai.rs:54-108](../../../crates/macllm-chat/src/openai.rs#L54-L108) · [openai.rs:136-230](../../../crates/macllm-chat/src/openai.rs#L136-L230) · [openai.rs:600-630](../../../crates/macllm-chat/src/openai.rs#L600-L630) · [output.rs:4](../../../crates/macllm-chat/src/output.rs#L4) · [openai.rs:937-1035](../../../crates/macllm-chat/src/openai.rs#L937-L1035) · +6</sub>

## 结构化输出和 tool call

`crates/macllm-chat/src/grammar.rs` 中的约束 grammar 会把每个候选 token 解码后的字节与一个封闭的 JSON Schema 子集进行比对。未知的 schema 关键字会被拒绝，object schema 必须设置 `additionalProperties: false`。

- `response_format` 只接受 `{"type":"json_schema","json_schema":{...,"strict":true}}`，不能与 `stop`、工具或 tool call 历史同时使用。
- 工具：最多 128 个名称互不相同的函数。声明了工具时，`tool_choice` 默认为 `auto`；也接受 `none`、指定函数，或只声明一个工具时的 `"required"`。启用工具时不允许使用 `stop`。
- 在固定的 Qwen3、Qwen3MoE 或 Qwen3.5 模板上使用 `auto`：答案末尾可以有一个调用，由 lazy grammar 约束。在通用 GGUF 模板上，调用格式从模板中检测（Hermes、Qwen XML、Llama 3 JSON、Mistral JSON、Granite JSON、Pythonic）；每个解析出的调用都会按其声明校验；除非 `parallel_tool_calls` 允许多个，否则只返回第一个。
- 强制调用需要 `parallel_tool_calls=false`，参数在 schema 约束下生成。它适用于固定的 Qwen3/Qwen3MoE profile 和通用 Hermes 模板；Qwen3.5 会拒绝强制调用，prompt 打开了推理块的通用模板也会拒绝。
- 其他 adapter 在 `format_agent_prompt` 中直接失败（fail closed）。

<sub>来源: [grammar.rs:1-20](../../../crates/macllm-chat/src/grammar.rs#L1-L20) · [openai.rs:232-247](../../../crates/macllm-chat/src/openai.rs#L232-L247) · [openai.rs:632-650](../../../crates/macllm-chat/src/openai.rs#L632-L650) · [openai.rs:652-689](../../../crates/macllm-chat/src/openai.rs#L652-L689) · [openai.rs:691-795](../../../crates/macllm-chat/src/openai.rs#L691-L795) · +13</sub>

## 启动检查与 prefix cache 参数

`EngineHandle::start` 只有在 worker 加载完模型后才返回，因此任何一项检查失败，`serve` 都会在绑定端口之前退出：

1. 模型必须支持 packed ragged batch。
2. Adapter 必须能通过它固定的模板或通用模板渲染一个健康检查用的 chat prompt，并且该 prompt 至少要 tokenize 出一个 token。没有可用模板的 checkpoint 可以用 `run --raw-prompt` 运行，但不能提供服务。
3. 服务端级别的 `--disable-thinking` 必须被模板支持。
4. 所有 `--max-active-sequences` 个会话都会预先创建。

之后 `serve` 还会检查 `--default-max-completion-tokens` 不超过模型上下文。

`serve` 默认启用自动 prefix cache（32768 个 token），而该缓存不能与 sliding-window attention 共存（[工作原理](04-how-it-works.md)）。因此任何含有 `SlidingCausal` 层的模型都会加载失败，报错 "sliding-attention storage is mutable"。Gemma 2 和 Gemma 3 通常会交替使用这类层，GGUF Llama 在 `llama.attention.sliding_window` 非零时也会有这类层。为这些模型提供服务时请使用 `--prefix-cache-capacity-tokens 0`。

<sub>来源: [engine.rs:863-867](../../../src/engine.rs#L863-L867) · [engine.rs:1224-1239](../../../src/engine.rs#L1224-L1239) · [engine.rs:1269-1360](../../../src/engine.rs#L1269-L1360) · [mod.rs:115-117](../../../src/models/mod.rs#L115-L117) · [server.rs:233-240](../../../src/server.rs#L233-L240) · +6</sub>
