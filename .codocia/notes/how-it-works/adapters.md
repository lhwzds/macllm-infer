---
title: Adapters and the shared decoder
order: 10
evidence:
  - src/models/mod.rs:35-66 #224584175aa8
  - src/models/mod.rs:128-155 #b6abb4b7252a
  - src/models/mod.rs:609-627 #94f27cefbc5b
  - src/gguf_model.rs:365-387 #539115b2ad64
  - src/models/mod.rs:698-714 #d1fbf6cba622
  - src/architecture.rs:224-229 #f0853310677a
  - src/architecture.rs:1038-1040 #62826089db0c
  - src/models/gemma.rs:75-80 #35a71f9c827c
  - src/models/lfm2.rs:313-319 #88f336c1ae73
  - src/gguf_model.rs:598-601 #6e214288f42d
  - src/gguf_model.rs:822-825 #075ea1f0fe10
  - src/gguf_model.rs:2650-2653 #8832fc29b798
---

`ModelAdapter` is the only place family code lives. An adapter turns a config or GGUF metadata into a `RuntimeSpec` and a `DecoderSpec`, maps weight names, and renders chat or agent prompts. Unpinned GGUF templates go to a generic Jinja renderer (`attach_prompt_tokens` in the Qwen adapters, `with_jinja_fallback` for the rest). Every adapter's `build_model` returns a `DecoderModel`, so attention, caching, batching and sampling are written once.

A `DecoderSpec` is a list of per-layer specs. Each layer picks a `SequenceMixerSpec`: `Attention` (full or sliding causal, RoPE or NoPE), `ShortConvolution` (LFM2), `GatedDeltaNet` (Qwen3.5) or `ParallelAttentionMamba2` (Falcon-H1), plus a dense or sparse-MoE feed-forward. Hybrid models are just layer lists; the decoder implements the `CausalLm` trait the rest of the runtime uses.

`ModelAdapter::build` checks the built model's runtime metadata against the adapter's, and `validate` checks the spec's consistency before any weight is touched.
