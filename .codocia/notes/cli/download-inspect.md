---
title: Downloading and inspecting checkpoints
order: 40
evidence:
  - src/main.rs:64-76 #e238777a21de
  - src/hub.rs:14-47 #5efbe0477cdb
  - src/hub.rs:59-109 #32e060cf486b
  - src/snapshot.rs:60-96 #261a5af72882
  - src/main.rs:810-857 #0178275bc915
---

`download [REPO]` (default `Qwen/Qwen2.5-0.5B-Instruct`, output `-o models/qwen2.5-0.5b-instruct`, optional `--revision`) fetches a fixed Safetensors allow-list: config, generation and tokenizer JSON, `model*.safetensors` plus index, and the sentence-transformers sidecars. `--gguf FILENAME` instead fetches one root-level GGUF, or every shard when given the canonical first shard. Files land in a temporary staging directory, must pass `ModelFiles::from_dir`, and only then replace the destination. A `.macllm-revision` marker records the resolved commit and whether the download completed; a model directory whose marker says incomplete is refused at load.

`inspect -m PATH` parses metadata without loading weights and prints family, layer geometry, F16 K/V bytes per token, fixed recurrent state per sequence, maximum context, vocabulary, stop token IDs, format and GGUF tensor-type counts.
