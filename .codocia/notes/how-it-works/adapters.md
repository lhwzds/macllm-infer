---
title: Adapters and the shared decoder
order: 10
evidence:
  - src/models/mod.rs:28-56 #8d52821e4471
  - src/models/mod.rs:102-118 #d1fbf6cba622
  - src/architecture.rs:224-229 #f0853310677a
  - src/architecture.rs:1038-1064 #e5ac5ded0ec5
  - src/models/gemma.rs:71-76 #35a71f9c827c
  - src/models/lfm2.rs:309-315 #88f336c1ae73
  - src/gguf_model.rs:519-522 #6e214288f42d
  - src/gguf_model.rs:743-746 #075ea1f0fe10
  - src/gguf_model.rs:2571-2574 #8832fc29b798
---

`ModelAdapter` is the only place family code lives. An adapter turns a config or GGUF metadata into a `RuntimeSpec` and a `DecoderSpec`, maps weight names, and renders chat or agent prompts. Every adapter's `build_model` returns a `DecoderModel`, so attention, caching, batching and sampling are written once.

A `DecoderSpec` is a list of per-layer specs. Each layer picks a `SequenceMixerSpec`: `Attention` (full or sliding causal, RoPE or NoPE), `ShortConvolution` (LFM2), `GatedDeltaNet` (Qwen3.5) or `ParallelAttentionMamba2` (Falcon-H1), plus a dense or sparse-MoE feed-forward. Hybrid models are therefore just layer lists, and the decoder implements the `CausalLm` trait the rest of the runtime uses.

`ModelAdapter::build` checks that the built model's runtime metadata matches the adapter's, and `validate` checks that the spec is internally consistent before any weight is touched.
