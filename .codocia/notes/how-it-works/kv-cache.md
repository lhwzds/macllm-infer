---
title: K/V cache layout
order: 40
evidence:
  - src/cache.rs:116-122 #291fe6476793
  - src/cache.rs:162-166 #04e9dec5006c
  - src/cache.rs:1401-1410 #0ba84f39c357
  - src/cache.rs:1450-1460 #9934aa08ff6e
  - src/cache.rs:2392-2397 #01be815e7523
  - src/cache.rs:3133-3142 #0b83d691c9f5
  - src/cache.rs:3353-3360 #769d4a45a1d6
  - src/runtime.rs:133-158 #c702fcc25b47
---

Each decoder layer gets a `KvCachePolicy`:

- `Paged`: full-attention layers share one model-wide pool. Its blocks are `--kv-cache-block-tokens` long, allocated `--kv-cache-slab-blocks` at a time, up to `--kv-cache-capacity-tokens` in total (by default one model context for all sequences together, not per sequence). Each sequence holds a `PagedKvBlockTable` of block IDs, and one block stores K and V for every full-attention layer.
- `Sliding`: local-attention layers keep a per-sequence ring of `2 * (W - 1)` slots. A rollback restores three integers instead of copying tensors.
- `Full`: a grow-on-demand contiguous cache.

Batch growth is transactional. A `PagedKvBatchReservation` reserves blocks for every lane before a forward, and dropping it without commit rolls all of them back. Blocks can be shared between tables by reference counting; appending to a shared partial tail block copies it first (copy-on-write).
