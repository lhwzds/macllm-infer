---
title: 自定义 Metal kernel 在运行时编译
source: #e373b924a8c1
---

大多数张量运算来自 Candle。项目在 `src/metal/` 中另外添加了三个 Metal 源文件：`paged_decode.metal`（通过 block 页表读取 K/V 的 paged attention，包括一个批量 GQA-4 变体）、`gated_delta_decay.metal`（Gated DeltaNet）和 `mamba2_scan.metal`（Mamba2 scan）。每个文件都用 `include_str!` 嵌入，在设备第一次需要时由 `new_library_with_source` 从源码编译。Pipeline 按 Metal 设备的 `registry_id` 缓存。

`build.rs` 不处理这些 kernel。它只嵌入 `MACLLM_BUILD_PROFILE`、`MACLLM_BUILD_OPT_LEVEL`、`MACLLM_BUILD_GIT_COMMIT` 和 `MACLLM_BUILD_GIT_DIRTY`，用于 benchmark 溯源。因此 kernel 编译错误会在运行时首次使用时出现，benchmark 使用 `--warmup-runs` 把这部分编译开销排除在测量之外。

只有预检通过时才会使用 paged kernel：batch 为 1，每个 lane 最多 128 个 query token、最多 128 个 lane，head 维度不超过 256，且为 F32 或 F16。否则它返回 `None`，decoder 回退到先物化完整矩阵的 attention（materialized attention）。
