---
title: Embedding and reranking
order: 30
evidence:
  - src/main.rs:216-276 #751645b4991d
  - src/main.rs:908-950 #63dcbb443b9a
  - src/main.rs:965-1023 #3ba659717363
  - src/main.rs:1025-1068 #5260b65e939d
  - src/models/qwen3_tasks.rs:1-40 #1504c73ec1fc
  - src/models/qwen3_tasks.rs:62-95 #c0d214e056de
  - src/models/qwen3_tasks.rs:158-188 #e4532b0306a5
  - src/models/qwen3_tasks.rs:309-316 #b75cb9e80774
  - src/rerank.rs:8-15 #61771c4ae19b
  - src/embedding.rs:8-15 #1ad701437f6b
---

`embed` and `rerank` work only with Safetensors Qwen3 task checkpoints. `Qwen3EmbeddingProfile` and `Qwen3RerankerProfile` check the sentence-transformers files (`modules.json`, `config_sentence_transformers.json`, and the pooling or logit-score module config), the EOS token, last-token pooling or distinct yes/no label IDs; `validate_task_checkpoint` then requires the Safetensors format. These tasks are CLI-only.

- `embed --input-type query|document --input TEXT [--input TEXT ...] [--instruction TEXT]`; `--instruction` is valid only for queries.
- `rerank --query TEXT --document TEXT [--document TEXT ...] [--instruction TEXT]`.

Inputs longer than `min(max_position_embeddings, 8192)` tokens are rejected, not truncated. Output is pretty-printed JSON on stdout, ordered by input `index` and without echoing inputs. Embedding rows carry a unit-length `embedding`; rerank rows carry `margin` (`yes_logit - no_logit`) and `yes_probability` (its sigmoid).
