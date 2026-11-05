---
title: Downloading and inspecting checkpoints
order: 40
evidence:
  - src/main.rs:46-62 #fa53772a7e01
  - src/hub.rs:14-47 #74afba0a5d18
  - src/hub.rs:70-120 #32e060cf486b
  - src/main.rs:710-757 #0178275bc915
---

`download [REPO]` (default `Qwen/Qwen2.5-0.5B-Instruct`, output `-o models/qwen2.5-0.5b-instruct`, optional `--revision`) fetches a fixed Safetensors allow-list: config, generation and tokenizer JSON, `model*.safetensors` plus index, and the sentence-transformers sidecars. `--gguf FILENAME` instead fetches one root-level GGUF, or every shard when given the canonical first shard. Files land in a temporary staging directory, must pass `ModelFiles::from_dir`, and only then replace the destination. A `.macllm-revision` marker records the resolved commit.

`inspect -m PATH` parses metadata without loading weights and prints family, layer geometry, F16 K/V bytes per token, fixed recurrent state per sequence, maximum context, vocabulary, stop token IDs, format and GGUF tensor-type counts.
