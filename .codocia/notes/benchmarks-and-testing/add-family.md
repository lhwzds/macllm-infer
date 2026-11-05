---
title: Adding a model family
order: 40
evidence:
  - src/models/mod.rs:57-100 #bd1106369692
  - src/gguf_model.rs:24-43 #33d1824053cb
  - src/gguf_model.rs:312-365 #08ac3bd3e409
  - src/models/qwen3.rs:140-168 #e5f5d819228c
  - src/models/qwen3.rs:246-251 #35a71f9c827c
  - src/models/falcon_h1.rs:61-70 #dce8521ffe8e
  - src/agent.rs:1-5 #c04a6a330e2a
---

A new family touches only its adapter and a dispatch arm. The adapter boundary is explained in [[how-it-works]], and format dispatch in [[overview]].

1. Add `src/models/<family>.rs` implementing `ModelAdapter`: `family`, `runtime_spec`, `decoder_spec`, `build_model` (normally `DecoderModel::load`, or `load_with_weight_names` for non-standard tensor names) and `format_chat_messages`.
2. For GGUF, add a `<family>_spec` builder with an explicit list of accepted metadata keys, plus match arms in `GgufModel::open` and `GgufModel::adapter`. For Safetensors, add a `model_type` arm in `detect`.
3. Recognise chat templates by exact identity rather than executing Jinja. The Qwen3 adapter, for example, accepts only templates whose byte length and SHA-256 match a pinned constant. Implement `agent_template_profile` only once a native tool template is verified.
4. Add an ignored gate under `tests/` that pins a public fixture by SHA-256 and compares greedy tokens with `llama.cpp`.

If a new layer type needs its own mixer, extend `SequenceMixerSpec` and the decoder, and say whether the prefix cache can snapshot it.
