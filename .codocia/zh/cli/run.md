---
title: 用 run 生成文本
source: #6b13edffa689
---

`run` 通过 adapter 的对话模板渲染 `--prompt`（以及可选的 `--system`），完成 tokenization 后把解码出的文本流式输出到 stdout。Adapter 逐字节渲染它固定（pin）的对话模板；GGUF 文件中内嵌的未固定模板交给通用 Jinja 渲染器；两者都没有的 checkpoint 会直接失败。`--raw-prompt` 会绕过模板，原样对 prompt 做 tokenization。`--add-special-tokens` 让 tokenizer 添加 BOS/EOS；对于本身已经包含 BOS 的模板，不要打开它。

采样默认值：`--max-new-tokens 128`、`--temperature 0.7`（0 表示贪心 argmax）、`--top-k 20`、`--top-p 0.8`、`--repetition-penalty 1.1`、`--seed 42`。

`--prefill-token-quantum` 限制每次 prefill 前向计算的 prompt token 数。不设置时，Metal 使用 128（`--dtype bf16` 除外）；CPU、CUDA 和 BF16 一次完成 prefill。生成结束后，stderr 上会输出一行指标，报告 prompt、cached 和 processed token 数、TTFT，以及 prefill 和 decode 的每秒 token 数。
