---
title: HTTP 服务与 API 子集
---

# HTTP 服务与 API 子集

<sub>[English](../03-http-api.md) · **中文**</sub>

`macllm serve` 在一个专用 worker 线程上加载一个模型，并用 axum 通过 HTTP/1 提供服务，直到按下 Ctrl-C。它默认绑定 loopback，自身不带认证或 TLS。向量嵌入和重排序不通过服务提供，请使用 CLI（[命令行用法](02-cli.md)）。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8080` | 绑定地址 |
| `--served-model-name` | 模型路径的文件名 | 客户端必须作为 `model` 发送的 ID |
| `--queue-capacity` | `8` | 等待活跃槽位的请求数 |
| `--max-active-sequences` | `2` | 在 worker 上轮转的序列数 |
| `--scheduler-token-budget` | `256` | 每轮调度的模型输入 token 数 |
| `--prefill-token-quantum` | `128` | 每个序列每轮的 prompt token 数 |
| `--event-capacity` | `16` | 每个请求的事件通道大小 |
| `--event-send-timeout-seconds` | `30` | 客户端停止读取时取消请求 |
| `--prefix-cache-capacity-tokens` | `32768` | prefix cache 预算；`0` 表示禁用 |

共用的模型参数（`--model`、`--device`、`--dtype`、`--kv-cache-*`）与 [命令行用法](02-cli.md) 中相同。

<sub>来源: [main.rs:396-429](../../../src/main.rs#L396-L429) · [main.rs:1257-1283](../../../src/main.rs#L1257-L1283) · [server.rs:123-150](../../../src/server.rs#L123-L150)</sub>

## 端点

| 路由 | 用途 |
| --- | --- |
| `GET /healthz` | `status`、`worker`（`starting`、`ready`、`busy`、`poisoned`、`stopped`）、`model`、`device`、`dtype`；不健康时返回 503 |
| `GET /v1/models` | 唯一的服务模型 |
| `POST /v1/chat/completions` | 严格的 Chat Completions 子集，JSON 或 SSE |
| `POST /completion` | 不套模板、与 llama-server 形状一致的路由，供 benchmark 使用 |

`/completion` 要求提供 `RawCompletionRequest` 的每一个字段（`prompt`、`n_predict`、`temperature`、`top_k`、`top_p`、`seed`、`cache_prompt`、`stream`、`return_tokens`、`id_slot`、`ignore_eos`、`repeat_penalty`、`response_fields`），并拒绝未知字段，这样 benchmark 中的两个服务端遵循完全相同的约定。

队列已满时返回 429；worker 已停止或处于 poisoned 状态时返回 503。流式响应每 15 秒发送一条 keep-alive 注释，并以 `data: [DONE]` 结束。

<sub>来源: [server.rs:123-131](../../../src/server.rs#L123-L131) · [server.rs:156-185](../../../src/server.rs#L156-L185) · [server.rs:219-236](../../../src/server.rs#L219-L236) · [server.rs:1265-1276](../../../src/server.rs#L1265-L1276) · [server.rs:1088-1096](../../../src/server.rs#L1088-L1096) · +1</sub>

## /v1/chat/completions 接受哪些字段

未知 JSON 字段返回 400；已知但不支持的字段返回 OpenAI 格式的错误，而不是被丢弃。

- `model` 必须等于服务的模型名；`n` 必须为 1。
- `max_completion_tokens`（或已弃用的 `max_tokens`，两者不能同时出现）默认为 128，且不能超过模型上下文。
- `temperature` 默认 1.0，范围 [0, 2]；`top_p` 默认 1.0，范围 (0, 1]；`seed` 必须非负。服务端把 `top_k` 固定为 0，重复惩罚固定为 1.0。
- `stop` 接受一个字符串或最多 4 个字符串。
- `stream` 支持 `stream_options.include_usage`；`include_obfuscation=true` 会被拒绝。
- 角色：`system`、`user`、`assistant`。`developer`、`name`、多模态内容、`logprobs`、`top_logprobs` 和非零的 `frequency_penalty`/`presence_penalty` 都会被拒绝。

`usage.prompt_tokens_details.cached_tokens` 报告有多少 prompt token 来自 prefix cache。

<sub>来源: [openai.rs:19-72](../../../src/openai.rs#L19-L72) · [openai.rs:90-170](../../../src/openai.rs#L90-L170) · [openai.rs:489-515](../../../src/openai.rs#L489-L515) · [openai.rs:815-870](../../../src/openai.rs#L815-L870) · [server.rs:793-802](../../../src/server.rs#L793-L802) · +4</sub>

## 结构化输出和 tool call

这两项功能都通过 `src/grammar.rs` 中的受约束解码 grammar 实现，它会把每个候选 token 解码后的字节与一个封闭的 JSON Schema 子集进行比对。未知的 schema 关键字会被拒绝，object schema 必须设置 `additionalProperties: false`。

- `response_format` 只接受 `{"type":"json_schema","json_schema":{...,"strict":true}}`，不能与 `stop`、工具或 tool call 历史同时使用。
- 工具：`tool_choice=auto` 尚未实现，会返回错误。声明了工具的请求默认是 `auto`，因此客户端必须显式设置 `tool_choice`。可接受的值：指定函数、只声明一个工具时的 `"required"`，或 `"none"`。需设置 `parallel_tool_calls=false`；强制调用时不允许使用 `stop`。强制调用的参数在其 parameters schema 的约束下生成。
- 只有对话模板哈希与固定 profile 匹配的 Qwen3 和 Qwen3MoE adapter 才有 native tool prompt。其他 adapter 在 `format_agent_prompt` 中直接失败（fail closed）。

<sub>来源: [openai.rs:517-530](../../../src/openai.rs#L517-L530) · [openai.rs:572-674](../../../src/openai.rs#L572-L674) · [server.rs:803-830](../../../src/server.rs#L803-L830) · [grammar.rs:1-15](../../../src/grammar.rs#L1-L15) · [qwen3.rs:264-277](../../../src/models/qwen3.rs#L264-L277) · +3</sub>

## 启动检查与 prefix cache 参数

`EngineHandle::start` 只有在 worker 加载完模型后才返回，因此任何一项检查失败，`serve` 都会在绑定端口之前退出：

1. 模型必须支持 packed ragged batch。
2. Adapter 必须能通过它识别的模板渲染一个健康检查用的 chat prompt，并且该 prompt 至少要 tokenize 出一个 token。没有受支持模板的 checkpoint 可以用 `run --raw-prompt` 运行，但不能提供服务。
3. 所有 `--max-active-sequences` 个会话都会预先创建。

`serve` 默认启用自动 prefix cache（32768 个 token），而该缓存不能与 sliding-window attention 共存（[工作原理](04-how-it-works.md)）。因此任何含有 `SlidingCausal` 层的模型都会加载失败，报错 "sliding-attention storage is mutable"。Gemma 2 和 Gemma 3 通常会交替使用这类层，GGUF Llama 在 `llama.attention.sliding_window` 非零时也会有这类层。为这些模型提供服务时请使用 `--prefix-cache-capacity-tokens 0`。

<sub>来源: [engine.rs:1141-1245](../../../src/engine.rs#L1141-L1245) · [runtime.rs:94-97](../../../src/runtime.rs#L94-L97) · [decoder.rs:3359-3369](../../../src/decoder.rs#L3359-L3369) · [decoder.rs:4319-4340](../../../src/decoder.rs#L4319-L4340) · [gguf_model.rs:1322-1325](../../../src/gguf_model.rs#L1322-L1325) · +3</sub>
