---
title: Custom Metal kernels are compiled at runtime
order: 60
evidence:
  - src/paged_attention.rs:790 #af34d7d9bca7
  - src/paged_attention.rs:1195-1226 #2f4b99a20678
  - src/paged_attention.rs:14-47 #6987f10e1677
  - src/mamba2.rs:331 #6608e8617a59
  - src/mamba2.rs:480-491 #3e29f260eef1
  - src/gated_delta.rs:470 #b81086646b29
  - src/gated_delta.rs:675-690 #e7d31928d260
  - build.rs:9-24 #6de3bf039d77
  - src/main.rs:262-270 #faaa28bd63e7
  - src/paged_attention.rs:116-122 #f801046b5887
  - src/metal/mamba2_scan.metal:18 #6fb326740006
  - src/metal/gated_delta_decay.metal:10 #8b6e65cd3fc8
  - src/metal/gated_delta_decay.metal:34 #3090c1a1f2f0
  - src/metal/paged_decode.metal:72 #60997dd9de48
---

Most tensor operations come from Candle. The project adds three Metal sources in `src/metal/`: `paged_decode.metal` (paged attention that reads K/V through a block page table, including a batched GQA-4 variant), `gated_delta_decay.metal` (Gated DeltaNet) and `mamba2_scan.metal` (Mamba2 scan). Each is embedded with `include_str!` and compiled from source by `new_library_with_source` the first time a device needs it. The pipelines are cached per Metal device `registry_id`.

`build.rs` does not touch these kernels. It only embeds `MACLLM_BUILD_PROFILE`, `MACLLM_BUILD_OPT_LEVEL`, `MACLLM_BUILD_GIT_COMMIT` and `MACLLM_BUILD_GIT_DIRTY` for benchmark provenance. A kernel compile error therefore appears at run time, on first use, and benchmarks use `--warmup-runs` to keep that compile cost out of measurements.

The paged kernel is used only when its preflight passes: batch 1, at most 128 query tokens per lane and 128 lanes, head dimension at most 256, and F32 or F16. Otherwise it returns `None` and the decoder falls back to materialized attention.
