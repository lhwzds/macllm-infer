---
title: 服务调度器
source: #851c2641f1ed
---

运行时不要求模型实现 `Send`，因此 `EngineHandle::start` 在名为 `macllm-inference` 的线程中构建模型，并且从不移动它。HTTP handler 通过有界通道（`--queue-capacity`）发送 `InferenceJob`，并拿回一个 `EngineTicket`，即请求本地的事件流。客户端断开连接时，正是通过丢弃 ticket 来取消请求。

Worker 按轮执行 continuous batching。每轮最多访问 `--max-active-sequences` 个活跃序列。每个序列最多分到 `--prefill-token-quantum` 个输入 token，一轮在达到 `--scheduler-token-budget` 时结束。所有被访问的 lane 作为一次 packed ragged 前向计算执行。Decode lane 只占一个 token，因此长 prompt 会被分块，不会饿死正在进行的生成。客户端在 `--event-send-timeout-seconds` 内一直不消费事件的 lane 会被取消。

循环内的任何错误或 panic 都会让 worker 进入 poisoned 状态（`/healthz` 报告 `poisoned`），而不是带着可能不一致的模型状态继续运行。
