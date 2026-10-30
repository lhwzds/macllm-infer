---
title: Scoreboard manifest
source: #910f4e0ac19e
---

Scoreboard manifest（`schema_version` 1，见 `benchmarks/manifests/`）固定六类信息：`host`（机器型号、芯片、内存、CPU 架构）；`model`（相对于 manifest 的路径、Hugging Face 仓库、revision、文件名、大小、SHA-256、架构、量化方式）；`llama_cpp`（二进制目录、build 号、commit、GPU 层数、flash attention、K/V cache 类型、mmap）；`workload`（prompt 文件及其 SHA-256、token 数和 token ID 的 SHA-256、decode 步数、贪心采样、并发宽度 1/4/8、预热次数、样本数、prefix cache 模式）；以及 `runtime`（设备、dtype、prefill quantum、K/V 设置）。

`scoreboard verify --manifest PATH` 不加载权重。它检查主机、模型字节、检测出的架构、是否为单个 GGUF 文件、prompt token ID，以及 `llama-server`/`llama-bench` 二进制的 commit，然后输出 JSON 摘要。任何不匹配都会导致失败。各个测量子命令都会先运行 `verify`：`macllm-kernel`、`llama-bench`、`llama-server`（`--port`，默认 18081）、`macllm-serving`，以及 `report`，后者运行所有阶段并写出 `--json` 和 `--markdown`。
