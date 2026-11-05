---
title: 端点
source: #150fd1150a99
---

| 路由 | 用途 |
| --- | --- |
| `GET /healthz` | `status`、`worker`（`starting`、`ready`、`busy`、`poisoned`、`stopped`）、`model`、`device`、`dtype`；不健康时返回 503 |
| `GET /v1/models` | 唯一的服务模型 |
| `POST /v1/chat/completions` | 严格的 Chat Completions 子集，JSON 或 SSE |
| `POST /completion` | 不套模板、与 llama-server 形状一致的路由，供 benchmark 使用 |

`/completion` 要求提供 `RawCompletionRequest` 的每一个字段（`prompt`、`n_predict`、`temperature`、`top_k`、`top_p`、`seed`、`cache_prompt`、`stream`、`return_tokens`、`id_slot`、`ignore_eos`、`repeat_penalty`、`response_fields`），并拒绝未知字段，这样 benchmark 中的两个服务端遵循完全相同的约定。

队列已满时返回 429；worker 已停止或处于 poisoned 状态时返回 503。流式响应每 15 秒发送一条 keep-alive 注释，并以 `data: [DONE]` 结束。
