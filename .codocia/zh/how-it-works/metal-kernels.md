---
title: 自定义 Metal kernel 在运行时编译
source: #976ae29aa0b7
---

在 Candle 的运算之外，`macllm-kernels` 在 `crates/macllm-kernels/src/metal/` 中添加了三个 Metal 源文件：`paged_decode.metal`（通过 block 页表读取 K/V 的 paged attention，包括一个批量 GQA-4 变体）、`gated_delta_decay.metal`（Gated DeltaNet）和 `mamba2_scan.metal`（Mamba2 scan）。每个文件都用 `include_str!` 嵌入，在设备第一次需要时由 `new_library_with_source` 从源码编译。Pipeline 按 Metal 设备的 `registry_id` 缓存。

`build.rs` 从不处理这些 kernel；它只为 benchmark 嵌入构建信息和 Git 溯源信息。因此 kernel 编译错误会在首次使用时出现，benchmark 使用 `--warmup-runs` 把编译开销排除在测量之外。

只有预检通过时才会使用 paged kernel：batch 为 1，每个 lane 最多 128 个 query token、最多 128 个 lane，head 维度不超过 256，F32 或 F16，并且 Metal 设备支持 Tier 2 argument buffer。否则它返回 `None`，decoder 回退到先物化完整矩阵的 attention（materialized attention）。
