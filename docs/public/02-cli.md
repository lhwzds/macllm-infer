---
title: Command-Line Usage
---

# Command-Line Usage

The `macllm` binary has nine subcommands: `download`, `inspect`, `run`, `embed`, `rerank`, `bench`, `scoreboard`, `demo` and `serve`. All flags below are copied from the clap definitions in `src/main.rs`. There are no runtime environment variables; the `MACLLM_*` variables in the code only point the ignored reference tests at fixture files (see [Benchmarks, Gates and Contributing](05-benchmarks-and-testing.md)). `serve` is covered in [HTTP Server and API Subset](03-http-api.md); `bench`, `scoreboard` and `demo` in [Benchmarks, Gates and Contributing](05-benchmarks-and-testing.md).

<sub>Sources: [main.rs:44-78](../../src/main.rs#L44-L78)</sub>

## Model, device and K/V flags

Every model-loading command shares `RuntimeArgs`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m, --model` | required | `.gguf` file or model directory |
| `--device` | `metal` | `metal` or `cpu` |
| `--dtype` | see below | `f32`, `f16`, `bf16` activations |
| `--kv-cache-block-tokens` | `16` | tokens per logical K/V block |
| `--kv-cache-slab-blocks` | `8` | blocks per physical slab allocation |
| `--kv-cache-capacity-tokens` | one model context | total K/V tokens shared by all sequences, rounded up to a block |

`resolve_dtype` picks `f32` on CPU or when the GGUF has quantized tensors, `f16` otherwise. The CPU/Accelerate backend rejects anything but `f32`.

<sub>Sources: [main.rs:80-106](../../src/main.rs#L80-L106) · [runtime.rs:34-50](../../src/runtime.rs#L34-L50) · [runtime.rs:162-175](../../src/runtime.rs#L162-L175) · [backend.rs:30-37](../../src/backend.rs#L30-L37)</sub>

## Generating text with run

`run` renders `--prompt` (plus optional `--system`) through the adapter's chat template, tokenizes it, and streams decoded text to stdout. Adapters render only templates they recognise and otherwise fail; `--raw-prompt` bypasses the template and tokenizes the prompt verbatim. `--add-special-tokens` asks the tokenizer to add BOS/EOS; leave it off for templates that already contain BOS.

Sampling defaults: `--max-new-tokens 128`, `--temperature 0.7` (0 means greedy argmax), `--top-k 20`, `--top-p 0.8`, `--repetition-penalty 1.1`, `--seed 42`.

`--prefill-token-quantum` caps prompt tokens per prefill forward. Unset, Metal uses 128 unless `--dtype bf16`; CPU and BF16 prefill in one shot. After generation, a metrics line on stderr reports prompt, cached and processed tokens, TTFT, prefill and decode tokens per second.

<sub>Sources: [main.rs:108-160](../../src/main.rs#L108-L160) · [main.rs:1220-1255](../../src/main.rs#L1220-L1255) · [main.rs:1285-1300](../../src/main.rs#L1285-L1300)</sub>

## Embedding and reranking

`embed` and `rerank` work only with Safetensors Qwen3 task checkpoints. `Qwen3EmbeddingProfile` and `Qwen3RerankerProfile` check the sentence-transformers files (`modules.json`, `config_sentence_transformers.json`, and the pooling or logit-score module config), the EOS token, last-token pooling or distinct yes/no label IDs; `validate_task_checkpoint` then requires the Safetensors format. These tasks are CLI-only.

- `embed --input-type query|document --input TEXT [--input TEXT ...] [--instruction TEXT]`; `--instruction` is valid only for queries.
- `rerank --query TEXT --document TEXT [--document TEXT ...] [--instruction TEXT]`.

Inputs longer than `min(max_position_embeddings, 8192)` tokens are rejected, not truncated. Output is pretty-printed JSON on stdout, ordered by input `index` and without echoing inputs. Embedding rows carry a unit-length `embedding`; rerank rows carry `margin` (`yes_logit - no_logit`) and `yes_probability` (its sigmoid).

<sub>Sources: [main.rs:172-232](../../src/main.rs#L172-L232) · [main.rs:808-850](../../src/main.rs#L808-L850) · [main.rs:865-923](../../src/main.rs#L865-L923) · [main.rs:925-968](../../src/main.rs#L925-L968) · [qwen3_tasks.rs:1-40](../../src/models/qwen3_tasks.rs#L1-L40) · +5</sub>

## Downloading and inspecting checkpoints

`download [REPO]` (default `Qwen/Qwen2.5-0.5B-Instruct`, output `-o models/qwen2.5-0.5b-instruct`, optional `--revision`) fetches a fixed Safetensors allow-list: config, generation and tokenizer JSON, `model*.safetensors` plus index, and the sentence-transformers sidecars. `--gguf FILENAME` instead fetches one root-level GGUF, or every shard when given the canonical first shard. Files land in a temporary staging directory, must pass `ModelFiles::from_dir`, and only then replace the destination. A `.macllm-revision` marker records the resolved commit.

`inspect -m PATH` parses metadata without loading weights and prints family, layer geometry, F16 K/V bytes per token, fixed recurrent state per sequence, maximum context, vocabulary, stop token IDs, format and GGUF tensor-type counts.

<sub>Sources: [main.rs:46-62](../../src/main.rs#L46-L62) · [hub.rs:14-47](../../src/hub.rs#L14-L47) · [hub.rs:70-120](../../src/hub.rs#L70-L120) · [main.rs:710-757](../../src/main.rs#L710-L757)</sub>
