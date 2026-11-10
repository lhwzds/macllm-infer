---
title:
source: #d381e7f8f194
---

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

共用的模型参数（`--model`、`--device`、`--dtype`、`--kv-cache-*`）与 [[cli]] 中相同。
