---
title: 
source: #cb7b6a1d8cfa
---

本项目的结论建立在两类证据之上。按 checkpoint 固定的参考测试证明 token 级别的正确性，对照对象是 `llama.cpp` 或 Candle。`macllm scoreboard`、`macllm demo` 和 `macllm bench` 下按 manifest 固定的 benchmark 衡量性能。所有测量命令都拒绝在未优化的构建中运行（`ensure_optimized_measurement_execution`），所以请使用 `--release`。
