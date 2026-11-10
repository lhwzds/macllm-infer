---
title: Adapter 与共享 decoder
source: #9fc22bffe97f
---

`ModelAdapter` 是唯一存放模型家族代码的地方。Adapter 把配置或 GGUF 元数据转换成 `RuntimeSpec` 和 `DecoderSpec`，映射权重名称，并渲染 chat 或 agent prompt。未固定的 GGUF 模板交给通用 Jinja 渲染器（Qwen3、Qwen3MoE 和 Qwen3.5 在 `attach_prompt_tokens` 中处理，其余包括 Qwen2 由 `with_jinja_fallback` 处理）。每个 adapter 的 `build_model` 都返回 `DecoderModel`，因此 attention、缓存、批处理和采样只需实现一次。

`DecoderSpec` 是逐层 spec 的列表。每一层选择一种 `SequenceMixerSpec`：`Attention`（full 或 sliding causal，RoPE 或 NoPE）、`ShortConvolution`（LFM2）、`GatedDeltaNet`（Qwen3.5）或 `ParallelAttentionMamba2`（Falcon-H1），再加上一个 dense 或 sparse-MoE 前馈层。混合架构模型只是不同的层列表；decoder 实现了运行时其余部分所使用的 `CausalLm` trait。

`ModelAdapter::build` 检查构建出的模型的运行时元数据与 adapter 是否一致，`validate` 在接触任何权重之前检查 spec 的一致性。
