# MacLLM Infer

[English](README.md) | [Simplified Chinese](README_CN.md)

MacLLM Infer is an Apple Silicon-first local runtime for text-only model inference,
written in Rust. Phase 1 provides three tasks—`generate`, `embed`, and `rerank`—on
the same decoder runtime. It uses `llama.cpp` as the baseline for GGUF support,
output behavior, memory, and performance, but does not aim for ABI compatibility
or complete feature parity.

The project's core goal is to be **architecture-general**. Adding a decoder-only
model should usually require only a new `ModelAdapter` and any genuinely missing
operator semantics. Loading, tokenization, the KV cache, batching, generation,
sampling, continuous batching, and serving continue to share the same runtime.
Embedding and reranking execution plumbing is reused only when the task semantics
match; prompts, pooling, and labels remain explicitly defined by each checkpoint
profile.

The product direction is not "llama.cpp rewritten in Rust." It is a **local
coding-agent runtime for Apple Silicon**: `llama.cpp` supplies the GGUF, token,
memory, and performance baselines, while MacLLM focuses on auditable Rust serving,
an Agent API, and an automatic prefix cache that works correctly for both dense
and hybrid models.

## Current Capabilities

- GGUF v3 single-file models and canonical shards; selected Llama, Qwen, and LFM2
  Safetensors profiles.
- Dispatch for 12 GGUF architectures: `llama`, `qwen2`, `qwen3`, `qwen3moe`,
  `qwen35`, `gemma`, `gemma2`, `gemma3`, `granitemoe`, `lfm2`, `falcon-h1`, and
  `smollm3`.
- Full and sliding-window GQA, explicit RoPE/NoPE, paged KV, Sparse MoE, ShortConv,
  Gated DeltaNet, Mamba2, and quantized execution on CPU and Metal.
- `generate`, `embed`, and `rerank` CLIs backed by a shared loader and decoder,
  together with corresponding local benchmarks.
- Qwen3 Embedding 0.6B: the last-token, final-normalized hidden state followed by
  L2 normalization; Qwen3 Reranker 0.6B: the margin between the last-token `yes/no`
  logits followed by a sigmoid.
- A strict, text-only, Chat Completions-compatible server that shares the generation
  loop.
- Automatic prefix caching with exact token-block identity, paged-KV refcounting
  and copy-on-write, bounded LRU eviction, and snapshot/restore of ShortConv,
  Gated DeltaNet, and Mamba2 recurrent state alongside the KV state.
- An Agent API with strict JSON Schema-constrained sampling, stop sequences,
  Qwen3/Qwen3MoE forced-single native tools and tool history, plus `cached_tokens`
  in OpenAI-compatible usage data.
- A manifest-pinned MacLLM/llama.cpp scoreboard, an eight-request physical page-reuse
  demo for an exact 16K repository/system prefix, and a `fair-prefix` serving
  comparison with the same transport, precision, and sample contract.
- Support claims published per checkpoint profile rather than extending one adapter's
  status to an entire model family.

The latest SmolLM3 vertical slice pins the revision, size, and SHA-256 of the
official `SmolLM3-Q4_K_M.gguf`. It covers the `smaug-bpe` tokenizer, explicit NoPE
every fourth layer, CPU and Metal cache/full execution, and raw and
deterministic-chat greedy oracles from llama.cpp build 10050. Every support claim
is bounded by the pinned checkpoint revision, file identity, and independent oracle
in the corresponding integration test. Other sizes, quantizations, or template
revisions of the same architecture do not automatically become officially supported.

## Quick Start

```bash
cargo build --release

./target/release/macllm inspect --model /path/to/model.gguf

./target/release/macllm run \
  --model /path/to/model.gguf \
  --prompt 'The capital of France is' \
  --max-new-tokens 32

./target/release/macllm embed \
  --model /path/to/Qwen3-Embedding-0.6B \
  --input-type query \
  --input 'What is the capital of China?'

./target/release/macllm rerank \
  --model /path/to/Qwen3-Reranker-0.6B \
  --query 'What is the capital of China?' \
  --document 'The capital of China is Beijing.' \
  --document 'Gravity attracts bodies.'

./target/release/macllm serve \
  --model /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080
```

For an unknown chat template, pass `--raw-prompt` explicitly. MacLLM neither
executes arbitrary Jinja nor silently guesses a template. Metal is the default
device; use `--device cpu` for CPU execution.

`run` and `bench generate` accept `--prefill-token-quantum TOKENS` to limit the
number of prompt tokens in one blocking prefill operation. When it is not set
explicitly, Metal uses 128 except when BF16 is selected explicitly; CPU and explicit
BF16 remain one-shot. An explicit positive integer, including a value greater than
128, is not capped. This default controls the Metal prefill working set; it does not
imply that 128 is faster for every model and prompt.

## Reproducible Comparisons

The pinned manifest verifies the Mac model, model byte count and SHA, prompt token
IDs, quantization, llama.cpp commit, and executable. The complete scoreboard reports
the MacLLM kernel, llama-bench throughput-only results, serving/TTFT/memory on both
sides, and greedy parity separately instead of combining different measurement
boundaries into one number:

```bash
./target/release/macllm scoreboard verify \
  --manifest benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json

./target/release/macllm scoreboard report \
  --manifest benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json \
  --json /tmp/macllm-scoreboard.json \
  --markdown /tmp/macllm-scoreboard.md
```

The two exact-16K commands answer different questions. The schema v4 `prefix-cache`
report demonstrates physical page/refcount reuse during MacLLM's first admission.
Its MacLLM side uses a direct `EngineHandle`, while its llama.cpp side uses loopback
HTTP/SSE. Because the transports differ, the report fails closed and does not publish
a speed ratio. The llama.cpp `cache_n` value also remains a logical counter and is not
converted into an estimate of MacLLM physical-page savings:

```bash
./target/release/macllm demo prefix-cache \
  --manifest benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json \
  --repository . \
  --json /tmp/macllm-prefix-demo.json \
  --markdown /tmp/macllm-prefix-demo.md
```

`fair-prefix` is the publication path for the controlled serving four-pair point
estimate. The dedicated manifest's `fair_prefix` section pins the real prompt source
to `prepared_repository_bundle`; the generic `workload.prompt` remains only as
scoreboard compatibility metadata. Both sides use the same raw `POST /completion`,
loopback HTTP/SSE, F32 K/V, eight-way concurrency, exact 16K prefix, 32 output tokens
per request, and 20 retained measured waves each. The shared request also fixes a
`response_fields` projection, preventing the llama.cpp terminal event from returning
the entire 16K prompt or unspecified metadata. Raw SSE keepalive is disabled on both
sides, and fragment and terminal fields, token accounting, generation settings, and
the cache-timings schema are validated event by event; unknown or additional fields
fail closed. Each measured stream also retains the 32 token IDs produced by the real
generation state machine. For the same task, they must match item by item across all
waves and between MacLLM and llama.cpp:

```bash
./target/release/macllm demo fair-prefix \
  --manifest benchmarks/manifests/smollm3-3b-q4_k_m-m4pro-fair-prefix.json \
  --repository . \
  --port 18086 \
  --json /tmp/macllm-fair-prefix.json \
  --markdown /tmp/macllm-fair-prefix.md
```

Each side's 20 waves are distributed across four fresh OS process lifetimes, with
five retained waves per lifetime. The four pairs run in `AB / BA / BA / AB` order
(A=`llama.cpp`, B=`MacLLM`). After each seed, the process stabilizes for a fixed
30 seconds and then runs five warm-up waves that are excluded from the result. A
process lifetime enters measurement only if the final three warm-ups have both an
aggregate-TPS coefficient of variation of `CV <= 10%` and a first-to-last relative
trend of `<= 10%`. Otherwise, the command fails immediately and produces no result.
Each side performs eight seeds per lifetime and eight excluded seed-state restores
after every warm-up and measured wave. This totals 32 seeds and
`4 × (5 warm-up + 5 measured) × 8 = 320` conditioning/restores per side, making the
conditioning request bodies and counts exactly symmetric between waves. Restore uses
the prompt for the other task and proves, at the token level and with fail-closed
validation, that it shares exactly 16K with the next measured task. Both sides
therefore restore with `cache_n=16,384` and perform the same suffix work.

The initial seed computation is not symmetric: all eight fixed llama.cpp slots
cold-prefill 16K, whereas MacLLM cold-prefills once and then reuses 16,368 tokens for
the remaining seven seeds, recomputing only the final 16-token block. All seeds are
untimed and occur before the 30-second stabilization period, five symmetric warm-up
waves, and steady-state gate. The report preserves this boundary explicitly instead
of describing the seed stage as equal computation. Measured and conditioning requests
on both sides send `cache_prompt=true`. MacLLM's physical KV pool holds 65,536 tokens,
but its prefix-snapshot budget is limited to exactly 16,384 tokens, so a measured task
cannot move the cache endpoint to a longer, task-specific prefix.

This lane's name and boundary are `controlled_exact_slot_cache_lane`; it is not a
product-default or best-tuned product comparison. The llama.cpp options
`--no-kv-unified --cache-ram 0` and `--batch-size 1024 --ubatch-size 128` fix the
exact-slot cache and batch geometry as control variables. They are not presented as
llama.cpp's optimal settings. A future best-tuned lane must tune each side on an
independent calibration set, freeze both configurations, and compare them only on
held-out measurement waves.

Every measured MacLLM process proves through `/healthz` that the model is loaded and
that it is actually using `device=metal` and `dtype=f32`. All measured llama.cpp
phases use `--log-disable` to prevent verbosity logging from contaminating timing.
Only after all measured phases finish does the runner start a separate, untimed
`-lv 4` attestation process. It proves `MTL0`, full-layer GPU offload, MTL0 F32 K/V,
resolved Flash Attention, and the actual slots/context/batch/ubatch values. Its binary,
model, and every non-logging argument must exactly match those of measured llama.cpp.
The model, precision, cache, batch, and transport arguments in the launch commands on
both sides must also match the report contract item by item. Every measured or control
child starts with inherited environment variables cleared, retaining only the
contract-pinned `LANG=C`, `LC_ALL=C`, and a minimal system `PATH`.

Throughout a long run, the llama-server executable and model file are guarded by open
file descriptors, device/inode/metadata checks, and two SHA-256 passes. Any attestation
or provenance drift fails closed. The current provenance boundary remains explicit:
the report pins the Homebrew llama-server executable but does not yet hash its
non-system dynamic-library closure, so this is not supply-chain-complete provenance.

The pooled p50 and p95 values in the report are descriptive only. The formal result
is a **four-pair point estimate**, calculated as the geometric mean of four paired
phase-throughput ratios. Each phase's throughput is the lifetime's total output tokens
divided by the sum of external wall time, not its phase p50. Four pairs do not provide
a reliable uncertainty interval, so even a point estimate greater than 1 cannot be
reported as an uncertainty-qualified conclusion that "MacLLM is faster." The result
describes this exact run on this one host; it is not a long-term SLA. The runner also
does not collect or gate the power source, Low Power Mode, thermal pressure or clock
state, or background load. Counterbalancing, cooldown, and the steady-state gate can
mitigate ordering effects and drift, but cannot prove that the host environments were
fully equivalent. The canonical scope and uncertainty warnings are validated schema
content; changing either causes report validation to fail. Both publication-grade
commands require the release executable and source to come from the same clean commit
and remain unchanged before and after the long run. On success, JSON and Markdown are
published as a pair in the same directory using temporary files, `fsync`, rename, and
rollback on failure. The Markdown header records the SHA-256 of its authoritative JSON
companion so artifacts from different runs cannot be mistaken for a pair.
The historical `9.712x` figure came from an unfair methodology and cannot replace the
new eligibility-gated `fair-prefix` artifact.

## Verification

```bash
cargo fmt --all -- --check
cargo check --all-targets
cargo check --all-targets --no-default-features
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

Real-model gates are marked `#[ignore]` and require the corresponding fixture and
device. `tests/qwen3_tasks_reference.rs` uses SHA-pinned Safetensors and sidecars to
compare embedding and reranking against an independent Candle Qwen3 graph and the
official examples, respectively. The source of every real-model gate states its
required environment variables, fixture identity, tolerance, and llama.cpp or Candle
oracle explicitly, preventing ordinary unit tests from being represented as real-model
evidence.

The code is licensed under the MIT License. Model weights remain subject to the license
or terms of use of their respective repositories.
