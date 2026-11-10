---
title: 服务调度器
source: #c053f36b065b
---

模型不必实现 `Send`：`EngineHandle::start` 在 `macllm-inference` 线程中构建模型，并且从不移动它。HTTP handler 通过有界通道（`--queue-capacity`）发送 `InferenceJob`，并拿回一个 `EngineTicket`，即请求本地的事件流。丢弃 ticket 就会取消该请求。

Worker 按轮执行 continuous batching。每轮最多访问 `--max-active-sequences` 个活跃序列。每个序列最多分到 `--prefill-token-quantum` 个输入 token，一轮在达到 `--scheduler-token-budget` 时结束。所有被访问的 lane 作为一次 packed ragged 前向计算执行。Decode lane 只占一个 token，因此长 prompt 不会饿死正在进行的生成。客户端在 `--event-send-timeout-seconds` 内一直不消费事件的 lane 会被取消。

panic，或者从准入、共享的 `drive_once` 步骤中逃逸出来的错误（例如模型前向计算失败），会让 worker 进入 poisoned 状态（`/healthz` 报告 `poisoned`），并使活跃和排队中的请求失败。请求本地的失败（非法采样参数、单个 lane 的 KV cache 耗尽、未完成的 JSON schema 输出）只会让该请求失败。
