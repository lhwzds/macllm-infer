---
title: Adapter 与共享 decoder
source: #5a0e3e8e0936
---

`ModelAdapter` 是唯一存放模型家族代码的地方。Adapter 把配置或 GGUF 元数据转换成 `RuntimeSpec` 和 `DecoderSpec`，映射权重名称，并渲染 chat 或 agent prompt。每个 adapter 的 `build_model` 都返回 `DecoderModel`，因此 attention、缓存、批处理和采样只需实现一次。

`DecoderSpec` 是逐层 spec 的列表。每一层选择一种 `SequenceMixerSpec`：`Attention`（full 或 sliding causal，RoPE 或 NoPE）、`ShortConvolution`（LFM2）、`GatedDeltaNet`（Qwen3.5）或 `ParallelAttentionMamba2`（Falcon-H1），再加上一个 dense 或 sparse-MoE 前馈层。因此混合架构模型只是不同的层列表，decoder 实现了运行时其余部分所使用的 `CausalLm` trait。

`ModelAdapter::build` 检查构建出的模型的运行时元数据与 adapter 的一致，`validate` 在接触任何权重之前检查 spec 自身是否一致。
