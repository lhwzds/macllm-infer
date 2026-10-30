---
title: Prefix cache 演示与本地 bench
source: #1d2a76230b21
---

`demo prefix-cache` 和 `demo fair-prefix` 把一个仓库的 `Cargo.toml` 和 `src/**/*.rs`（`--repository`，默认 `.`）转换成精确的 16K token 共享前缀，并以八个并发请求发送给两个运行时。它们默认使用端口 18085 和 18086，分别写出 `prefix-cache-demo.{json,md}` 或 `fair-prefix-benchmark.{json,md}`。结果只是该主机上该次运行的 点估计（point estimate）。

`fair-prefix` 还要求两边使用相同的 HTTP transport 和 F32 K/V 精度。`main.rs` 把 `MAC_PREFIX_DEMO_DEFAULT_MEASURED_WAVES`（20）作为每个实现的测量波次数传入。`validate_inputs` 要求至少 20 个波次，且波次数能平均分配到 4 个相互平衡（counterbalanced）的进程生命周期中，同时 manifest 中的 `fair_prefix` 块必须完全匹配。该块必须声明每个生命周期 5 个预热波次（`FAIR_WARMUP_WAVES_PER_LIFETIME`）、32 token 的输出、已预热的 prefix cache，以及等于波次数乘以 8（160）的 `samples_per_concurrency`。一个 3 波次的稳态窗口（`FAIR_STEADY_STATE_WINDOW_WAVES`）用于检查预热。

`bench generate|embed|rerank` 是不需要 manifest 的本地诊断工具：在 `--warmup-runs`（默认 1）之后进行 `--runs`（默认 20）次测量，使用贪心解码，并拒绝任何输出发生漂移的运行。`scripts/bench-llama-cpp.sh MODEL.gguf` 运行原生的 `llama-bench`，并读取 `PROMPT_TOKENS`（512）、`OUTPUT_TOKENS`（128）和 `RUNS`（3）。
