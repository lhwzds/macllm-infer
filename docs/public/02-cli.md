---
title: Command-Line Usage
---

# Command-Line Usage

<sub>**English** · [中文](zh/02-cli.md)</sub>

The `macllm` binary has nine subcommands: `download`, `inspect`, `run`, `embed`, `rerank`, `bench`, `scoreboard`, `demo` and `serve`. All flags below are copied from the clap definitions in `src/main.rs`. There are no runtime environment variables; the `MACLLM_*` variables in the code only point the ignored reference tests at fixture files (see [Benchmarks, Gates and Contributing](05-benchmarks-and-testing.md)). `serve` is covered in [HTTP Server and API Subset](03-http-api.md); `bench`, `scoreboard` and `demo` in [Benchmarks, Gates and Contributing](05-benchmarks-and-testing.md).

<sub>Sources: [main.rs:62-96](../../src/main.rs#L62-L96)</sub>

## Model, device and K/V flags

Every model-loading command shares `RuntimeArgs`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m, --model` | required | `.gguf` file or model directory |
| `--device` | `metal` | `metal`, `cuda` or `cpu`; builds without Metal default to `cuda` if built with it, else `cpu` |
| `--dtype` | see below | `f32`, `f16`, `bf16` activations |
| `--kv-cache-block-tokens` | `16` | tokens per logical K/V block |
| `--kv-cache-slab-blocks` | `8` | blocks per physical slab allocation |
| `--kv-cache-capacity-tokens` | one model context | total K/V tokens shared by all sequences, rounded up to a block |
| `--prefill-attention` | `sdpa` | multi-row prefill chunks use Candle's fused SDPA kernel; `native` keeps the paged Metal kernel with its dense fallback |

`resolve_dtype` picks `f32` on CPU or when the GGUF has quantized tensors, `f16` otherwise. The CPU/Accelerate backend rejects anything but `f32`; Metal and CUDA accept all three.

<sub>Sources: [main.rs:36-48](../../src/main.rs#L36-L48) · [main.rs:114-150](../../src/main.rs#L114-L150) · [runtime.rs:37-40](../../src/runtime.rs#L37-L40) · [runtime.rs:167-180](../../src/runtime.rs#L167-L180) · [backend.rs:61-71](../../src/backend.rs#L61-L71)</sub>

## Generating text with run

`run` renders `--prompt` (plus optional `--system`) through the adapter's chat template, tokenizes it, and streams decoded text to stdout. Adapters render the templates they pin byte-for-byte; an unpinned template embedded in a GGUF file goes through the generic Jinja renderer, and a checkpoint with neither fails. `--raw-prompt` bypasses the template and tokenizes the prompt verbatim. `--add-special-tokens` asks the tokenizer to add BOS/EOS; leave it off for templates that already contain BOS.

Sampling defaults: `--max-new-tokens 128`, `--temperature 0.7` (0 means greedy argmax), `--top-k 20`, `--top-p 0.8`, `--repetition-penalty 1.1`, `--seed 42`.

`--prefill-token-quantum` caps prompt tokens per prefill forward. Unset, Metal uses 128 unless `--dtype bf16`; CPU, CUDA and BF16 prefill in one shot. After generation, a metrics line on stderr reports prompt, cached and processed tokens, TTFT, prefill and decode tokens per second.

<sub>Sources: [main.rs:152-190](../../src/main.rs#L152-L190) · [main.rs:1320-1334](../../src/main.rs#L1320-L1334) · [main.rs:1345-1356](../../src/main.rs#L1345-L1356) · [mod.rs:515-536](../../src/models/mod.rs#L515-L536) · [gguf_model.rs:365-387](../../src/gguf_model.rs#L365-L387) · +1</sub>

## Embedding and reranking

`embed` and `rerank` work only with Safetensors Qwen3 task checkpoints. `Qwen3EmbeddingProfile` and `Qwen3RerankerProfile` check the sentence-transformers files (`modules.json`, `config_sentence_transformers.json`, and the pooling or logit-score module config), the EOS token, last-token pooling or distinct yes/no label IDs; `validate_task_checkpoint` then requires the Safetensors format. These tasks are CLI-only.

- `embed --input-type query|document --input TEXT [--input TEXT ...] [--instruction TEXT]`; `--instruction` is valid only for queries.
- `rerank --query TEXT --document TEXT [--document TEXT ...] [--instruction TEXT]`.

Inputs longer than `min(max_position_embeddings, 8192)` tokens are rejected, not truncated. Output is pretty-printed JSON on stdout, ordered by input `index` and without echoing inputs. Embedding rows carry a unit-length `embedding`; rerank rows carry `margin` (`yes_logit - no_logit`) and `yes_probability` (its sigmoid).

<sub>Sources: [main.rs:216-276](../../src/main.rs#L216-L276) · [main.rs:908-950](../../src/main.rs#L908-L950) · [main.rs:965-1023](../../src/main.rs#L965-L1023) · [main.rs:1025-1068](../../src/main.rs#L1025-L1068) · [qwen3_tasks.rs:1-40](../../src/models/qwen3_tasks.rs#L1-L40) · +5</sub>

## Downloading and inspecting checkpoints

`download [REPO]` (default `Qwen/Qwen2.5-0.5B-Instruct`, output `-o models/qwen2.5-0.5b-instruct`, optional `--revision`) fetches a fixed Safetensors allow-list: config, generation and tokenizer JSON, `model*.safetensors` plus index, and the sentence-transformers sidecars. `--gguf FILENAME` instead fetches one root-level GGUF, or every shard when given the canonical first shard. Files land in a temporary staging directory, must pass `ModelFiles::from_dir`, and only then replace the destination. A `.macllm-revision` marker records the resolved commit and whether the download completed; a model directory whose marker says incomplete is refused at load.

`inspect -m PATH` parses metadata without loading weights and prints family, layer geometry, F16 K/V bytes per token, fixed recurrent state per sequence, maximum context, vocabulary, stop token IDs, format and GGUF tensor-type counts.

<sub>Sources: [main.rs:64-76](../../src/main.rs#L64-L76) · [hub.rs:14-47](../../src/hub.rs#L14-L47) · [hub.rs:59-109](../../src/hub.rs#L59-L109) · [snapshot.rs:60-96](../../src/snapshot.rs#L60-L96) · [main.rs:810-857](../../src/main.rs#L810-L857)</sub>
