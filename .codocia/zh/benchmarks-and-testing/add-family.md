---
title: 添加模型家族
source: #916d18233647
---

新增一个模型家族只需改动它的 adapter 和一个 dispatch 分支。Adapter 的边界见 [[how-it-works]]，格式 dispatch 见 [[overview]]。

1. 添加实现 `ModelAdapter` 的 `src/models/<family>.rs`：`family`、`runtime_spec`、`decoder_spec`、`build_model`（通常是 `DecoderModel::load`，张量名称不标准时用 `load_with_weight_names`）和 `format_chat_messages`。
2. 对于 GGUF，添加一个带有明确的可接受元数据键列表的 `<family>_spec` 构建器，并在 `GgufModel::open` 和 `GgufModel::architecture_adapter` 中添加 match 分支。对于 Safetensors，在 `detect` 中添加一个 `model_type` 分支。
3. 按精确身份固定（pin）已知的对话模板：`crates/macllm-chat/src/chat_templates/` 中的 Qwen3 分类器只按字节长度和 SHA-256 匹配。未固定的 GGUF 模板会回退到通用 Jinja 渲染器（[[how-it-works]]）。只有在验证过 native tool 模板之后才实现 `agent_template_profile`。
4. 在 `tests/` 下添加一个被忽略的门禁测试，按 SHA-256 固定一个公开 fixture，并与 `llama.cpp` 比较贪心 token。

如果新的层类型需要自己的 mixer，就扩展 `SequenceMixerSpec` 和 decoder，并说明 prefix cache 能否为它保存快照。
