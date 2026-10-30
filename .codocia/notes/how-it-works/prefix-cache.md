---
title: Automatic prefix cache
order: 50
evidence:
  - src/prefix_cache.rs:1-7 #497fa2620036
  - src/prefix_cache.rs:116-122 #2342b90343e3
  - src/decoder.rs:3461-3499 #6f04069653c4
  - src/decoder.rs:3501-3540 #b846a89ce8ec
  - src/decoder.rs:4308-4340 #845d56ced21b
  - src/engine.rs:166-172 #ff0594e8d5c4
  - src/generation.rs:513-517 #42a0d50164d2
---

`TokenPrefixCache` is a cost-bounded LRU of model-state snapshots keyed by exact token IDs. Hash blocks only narrow the candidates; every hit is confirmed by a token-by-token comparison from position zero. A snapshot is stored only at each inserted prompt endpoint, never at every block, so a set of progressively longer prompts does not retain a quadratic amount of state.

After prefill, `publish_prompt_prefix` stores the prompt. Attention-only models store it truncated to a block boundary as a forked block table, which shares pages by reference count. Before the next prompt, `reset_session_for_prompt` restores the longest match into the session and reports it as `cached_tokens`.

Recurrent layers (ShortConv, Gated DeltaNet, Mamba2) need exact-endpoint snapshots of their state, because recurrent state cannot be truncated. Sliding-window rings are mutable and cannot be forked, so `validate_prefix_cache_topology` rejects any sliding layer. The cache and sliding-window attention are therefore mutually exclusive, and a non-zero capacity on such a model is a load error, not a silent no-op. Benchmark waves can opt out of publishing through `EngineRequestPolicy`.
