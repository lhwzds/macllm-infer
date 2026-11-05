---
title: Prefix-cache demos and local bench
order: 30
evidence:
  - src/main.rs:352-394 #182e259b0967
  - src/main.rs:560-573 #86482cfdda54
  - src/benchmark/prefix_demo_mac.rs:39-46 #70d0343130f2
  - src/benchmark/fair_prefix.rs:59-61 #24a47f0bcf07
  - src/benchmark/fair_prefix.rs:912-956 #5faa4affde26
  - src/benchmark/fair_prefix.rs:1010-1022 #01ca7c0de766
  - benchmarks/manifests/smollm3-3b-q4_k_m-m4pro-fair-prefix.json:61-81 #3bfdafb1badc
  - src/main.rs:247-302 #1f4fd4de7719
  - src/main.rs:1165-1208 #d7f2c0b47dfb
  - scripts/bench-llama-cpp.sh #aa2fd647f93d
---

`demo prefix-cache` and `demo fair-prefix` turn a repository's `Cargo.toml` and `src/**/*.rs` (`--repository`, default `.`) into an exact 16K-token shared prefix and send it as eight concurrent requests to both runtimes. They default to ports 18085 and 18086 and write `prefix-cache-demo.{json,md}` or `fair-prefix-benchmark.{json,md}`. Results are point estimates for that host and run.

`fair-prefix` also requires the same HTTP transport and F32 K/V precision on both sides. `main.rs` passes `MAC_PREFIX_DEMO_DEFAULT_MEASURED_WAVES` (20) as its measured waves per implementation. `validate_inputs` requires at least 20 waves, a count that splits evenly across 4 counterbalanced process lifetimes, and a manifest `fair_prefix` block that matches exactly. That block must declare 5 warm-up waves per lifetime (`FAIR_WARMUP_WAVES_PER_LIFETIME`), 32-token completions, a warm prefix cache, and `samples_per_concurrency` equal to waves times 8 (160). A 3-wave steady-state window (`FAIR_STEADY_STATE_WINDOW_WAVES`) checks the warm-ups.

`bench generate|embed|rerank` is a local diagnostic without a manifest: `--runs` (default 20) measured repetitions after `--warmup-runs` (default 1), greedy decoding, and rejection of any run whose output drifts. `scripts/bench-llama-cpp.sh MODEL.gguf` runs plain `llama-bench` and reads `PROMPT_TOKENS` (512), `OUTPUT_TOKENS` (128) and `RUNS` (3).
