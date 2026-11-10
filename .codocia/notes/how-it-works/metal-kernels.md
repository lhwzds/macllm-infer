---
title: Custom Metal kernels are compiled at runtime
order: 60
evidence:
  - crates/macllm-kernels/src/paged_attention.rs:800 #af34d7d9bca7
  - crates/macllm-kernels/src/paged_attention.rs:1205-1236 #2f4b99a20678
  - crates/macllm-kernels/src/paged_attention.rs:5-55 #acafb4d43da1
  - crates/macllm-kernels/src/paged_attention.rs:892-896 #ce0595898081
  - crates/macllm-kernels/src/mamba2.rs:332 #6608e8617a59
  - crates/macllm-kernels/src/mamba2.rs:481-492 #3e29f260eef1
  - crates/macllm-kernels/src/gated_delta.rs:473 #b81086646b29
  - crates/macllm-kernels/src/gated_delta.rs:678-693 #e7d31928d260
  - build.rs:9-24 #6de3bf039d77
  - src/main.rs:306-314 #faaa28bd63e7
  - crates/macllm-kernels/src/paged_attention.rs:125-131 #f801046b5887
  - crates/macllm-kernels/src/metal/mamba2_scan.metal:18 #6fb326740006
  - crates/macllm-kernels/src/metal/gated_delta_decay.metal:10 #8b6e65cd3fc8
  - crates/macllm-kernels/src/metal/gated_delta_decay.metal:34 #3090c1a1f2f0
  - crates/macllm-kernels/src/metal/paged_decode.metal:72 #60997dd9de48
---

Beyond Candle's operations, `macllm-kernels` adds three Metal sources in `crates/macllm-kernels/src/metal/`: `paged_decode.metal` (paged attention that reads K/V through a block page table, including a batched GQA-4 variant), `gated_delta_decay.metal` (Gated DeltaNet) and `mamba2_scan.metal` (Mamba2 scan). Each is embedded with `include_str!` and compiled from source by `new_library_with_source` the first time a device needs it. The pipelines are cached per Metal device `registry_id`.

`build.rs` never touches the kernels; it only embeds build and Git provenance for benchmarks. A kernel compile error therefore appears on first use, and benchmarks use `--warmup-runs` to keep the compile cost out of measurements.

The paged kernel is used only when its preflight passes: batch 1, at most 128 query tokens per lane and 128 lanes, head dimension at most 256, F32 or F16, and a Metal device with Tier 2 argument buffers. Otherwise it returns `None` and the decoder falls back to materialized attention.
