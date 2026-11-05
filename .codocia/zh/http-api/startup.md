---
title: 启动检查与 prefix cache 参数
source: #8444e6f8cb7f
---

`EngineHandle::start` 只有在 worker 加载完模型后才返回，因此任何一项检查失败，`serve` 都会在绑定端口之前退出：

1. 模型必须支持 packed ragged batch。
2. Adapter 必须能通过它识别的模板渲染一个健康检查用的 chat prompt，并且该 prompt 至少要 tokenize 出一个 token。没有受支持模板的 checkpoint 可以用 `run --raw-prompt` 运行，但不能提供服务。
3. 所有 `--max-active-sequences` 个会话都会预先创建。

`serve` 默认启用自动 prefix cache（32768 个 token），而该缓存不能与 sliding-window attention 共存（[[how-it-works]]）。因此任何含有 `SlidingCausal` 层的模型都会加载失败，报错 "sliding-attention storage is mutable"。Gemma 2 和 Gemma 3 通常会交替使用这类层，GGUF Llama 在 `llama.attention.sliding_window` 非零时也会有这类层。为这些模型提供服务时请使用 `--prefix-cache-capacity-tokens 0`。
