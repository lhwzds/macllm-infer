---
title: 自动 prefix cache
source: #c3deb960013b
---

`TokenPrefixCache` 是一个按开销限额的 LRU，缓存模型状态快照，以精确的 token ID 为键。哈希 block 只用于缩小候选范围；每次命中都要从位置 0 开始逐 token 比较确认。快照只在每个插入的 prompt 末尾保存，而不是在每个 block 上保存，因此一组逐渐变长的 prompt 不会占用平方级的状态。

Prefill 之后，`publish_prompt_prefix` 保存 prompt。纯 attention 模型会把它截断到 block 边界，存成一个 fork 出来的 block 表，通过引用计数共享页面。处理下一个 prompt 之前，`reset_session_for_prompt` 把最长匹配恢复到会话中，并报告为 `cached_tokens`。

循环层（ShortConv、Gated DeltaNet、Mamba2）需要在精确末尾位置保存状态快照，因为循环状态无法截断。Sliding-window 环形缓冲是可变的，无法 fork，所以 `validate_prefix_cache_topology` 会拒绝任何 sliding 层。因此该缓存与 sliding-window attention 互斥，在这类模型上设置非零容量会导致加载错误，而不是静默地不生效。Benchmark 的波次可以通过 `EngineRequestPolicy` 选择不发布前缀。
