---
title: Adding a model family
order: 40
evidence:
  - src/gguf_model.rs:64-83 #33d1824053cb
  - src/gguf_model.rs:391-435 #087af1368a92
  - crates/macllm-chat/src/chat_templates/qwen3.rs:42-67 #a9acf0396ebf
  - src/models/mod.rs:76-80 #0ad6fbe59a99
  - src/models/qwen3.rs:223-228 #35a71f9c827c
  - src/models/falcon_h1.rs:52-61 #dce8521ffe8e
  - crates/macllm-chat/src/agent.rs:1-5 #c04a6a330e2a
---

A new family touches only its adapter and a dispatch arm. The adapter boundary is explained in [[how-it-works]], and format dispatch in [[overview]].

1. Add `src/models/<family>.rs` implementing `ModelAdapter`: `family`, `runtime_spec`, `decoder_spec`, `build_model` (normally `DecoderModel::load`, or `load_with_weight_names` for non-standard tensor names) and `format_chat_messages`.
2. For GGUF, add a `<family>_spec` builder with an explicit list of accepted metadata keys, plus match arms in `GgufModel::open` and `GgufModel::architecture_adapter`. For Safetensors, add a `model_type` arm in `detect`.
3. Pin known chat templates by exact identity: the Qwen3 classifier in `crates/macllm-chat/src/chat_templates/` matches only byte length and SHA-256. An unpinned GGUF template falls back to the generic Jinja renderer ([[how-it-works]]). Implement `agent_template_profile` only once a native tool template is verified.
4. Add an ignored gate under `tests/` that pins a public fixture by SHA-256 and compares greedy tokens with `llama.cpp`.

If a new layer type needs its own mixer, extend `SequenceMixerSpec` and the decoder, and say whether the prefix cache can snapshot it.
