---
title: 
source: #2a47608e6edb
---

`macllm` 二进制有九个子命令：`download`、`inspect`、`run`、`embed`、`rerank`、`bench`、`scoreboard`、`demo` 和 `serve`。下文所有参数都照搬自 `src/main.rs` 中的 clap 定义。运行时不读取任何环境变量；代码中的 `MACLLM_*` 变量只用于让被忽略的参考测试找到 fixture 文件（见 [[benchmarks-and-testing]]）。`serve` 见 [[http-api]]；`bench`、`scoreboard` 和 `demo` 见 [[benchmarks-and-testing]]。
