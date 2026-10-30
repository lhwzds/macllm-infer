---
title: Benchmark、门禁测试与贡献
---

# Benchmark、门禁测试与贡献

<sub>[English](../05-benchmarks-and-testing.md) · **中文**</sub>

本项目的结论建立在两类证据之上。按 checkpoint 固定的参考测试证明 token 级别的正确性，对照对象是 `llama.cpp` 或 Candle。`macllm scoreboard`、`macllm demo` 和 `macllm bench` 下按 manifest 固定的 benchmark 衡量性能。所有测量命令都拒绝在未优化的构建中运行（`ensure_optimized_measurement_execution`），所以请使用 `--release`。

<sub>来源: [mod.rs:1](../../../src/benchmark/mod.rs#L1) · [gemma_gguf_reference.rs:1-8](../../../tests/gemma_gguf_reference.rs#L1-L8) · [main.rs:663-708](../../../src/main.rs#L663-L708)</sub>

## 固定的参考门禁测试

真实模型测试标记为 `#[ignore]`，需要对应的 fixture 文件。Gemma 1 的门禁测试很有代表性：它按大小和 SHA-256 固定公开文件，并检查 prompt token ID。在 CPU 和 Metal 上，它都断言带缓存的 decode 与全新的完整前向计算一致（argmax 相同，logit 的最大差值和相对 RMS 在界限内），并且贪心生成能复现 `llama.cpp` 的 token ID。其他门禁测试则与 `llama.cpp` 的 golden 输出、Candle 的独立实现之一或两者进行比较。

| 架构 | 固定的 fixture |
| --- | --- |
| `gemma` | Gemma 1 2B-it Q2_K（`tensorblock/gemma-2b-it-GGUF`） |
| `gemma2` / `gemma3` | Gemma 2 2B-it Q4_K_M / Gemma 3 270M-it QAT Q4_0 |
| `smollm3` | SmolLM3-3B Q4_K_M（官方） |
| `llama` | SmolLM2-135M、Llama 3.2 1B Q4_K_M、TinyLlama Q2_K 和 Q3_K_M、TinyMistral-6x248M MoE |
| `qwen2` / `qwen3` / `qwen35` | Qwen2.5-0.5B Q4_0 / Qwen3-0.6B Q8_0（也有 12 分片版本）/ Qwen3.5-0.8B Q4_0 |
| `qwen3moe` / `granitemoe` | loggenix-moe Q8_0 / Granite 3.1 1B-A400M Q4_K_M |
| `lfm2` / `falcon-h1` | LFM2-350M Q4_K_M / Falcon-H1-Tiny-90M Q8_0 |
| 任务 | Qwen3 Embedding 和 Reranker checkpoint，对照 Candle 和 Qwen 的示例 |

运行某个门禁测试时需设置其 fixture 变量，例如 `MACLLM_GEMMA_GGUF_MODEL=models/gemma-2b-it-q2_k cargo test --release --test gemma_gguf_reference -- --ignored --nocapture`。

<sub>来源: [gemma_gguf_reference.rs:1-8](../../../tests/gemma_gguf_reference.rs#L1-L8) · [gemma_gguf_reference.rs:27-54](../../../tests/gemma_gguf_reference.rs#L27-L54) · [gemma_gguf_reference.rs:148-201](../../../tests/gemma_gguf_reference.rs#L148-L201) · [gemma_gguf_reference.rs:290-312](../../../tests/gemma_gguf_reference.rs#L290-L312) · [smollm3_gguf_reference.rs:1-8](../../../tests/smollm3_gguf_reference.rs#L1-L8) · +16</sub>

## Scoreboard manifest

Scoreboard manifest（`schema_version` 1，见 `benchmarks/manifests/`）固定六类信息：`host`（机器型号、芯片、内存、CPU 架构）；`model`（相对于 manifest 的路径、Hugging Face 仓库、revision、文件名、大小、SHA-256、架构、量化方式）；`llama_cpp`（二进制目录、build 号、commit、GPU 层数、flash attention、K/V cache 类型、mmap）；`workload`（prompt 文件及其 SHA-256、token 数和 token ID 的 SHA-256、decode 步数、贪心采样、并发宽度 1/4/8、预热次数、样本数、prefix cache 模式）；以及 `runtime`（设备、dtype、prefill quantum、K/V 设置）。

`scoreboard verify --manifest PATH` 不加载权重。它检查主机、模型字节、检测出的架构、是否为单个 GGUF 文件、prompt token ID，以及 `llama-server`/`llama-bench` 二进制的 commit，然后输出 JSON 摘要。任何不匹配都会导致失败。各个测量子命令都会先运行 `verify`：`macllm-kernel`、`llama-bench`、`llama-server`（`--port`，默认 18081）、`macllm-serving`，以及 `report`，后者运行所有阶段并写出 `--json` 和 `--markdown`。

<sub>来源: [main.rs:304-350](../../../src/main.rs#L304-L350) · [runner.rs:72-175](../../../src/benchmark/runner.rs#L72-L175) · [smollm3-3b-q4_k_m-m4pro.json](../../../benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json) · [main.rs:616-661](../../../src/main.rs#L616-L661)</sub>

## Prefix cache 演示与本地 bench

`demo prefix-cache` 和 `demo fair-prefix` 把一个仓库的 `Cargo.toml` 和 `src/**/*.rs`（`--repository`，默认 `.`）转换成精确的 16K token 共享前缀，并以八个并发请求发送给两个运行时。它们默认使用端口 18085 和 18086，分别写出 `prefix-cache-demo.{json,md}` 或 `fair-prefix-benchmark.{json,md}`。结果只是该主机上该次运行的 点估计（point estimate）。

`fair-prefix` 还要求两边使用相同的 HTTP transport 和 F32 K/V 精度。`main.rs` 把 `MAC_PREFIX_DEMO_DEFAULT_MEASURED_WAVES`（20）作为每个实现的测量波次数传入。`validate_inputs` 要求至少 20 个波次，且波次数能平均分配到 4 个相互平衡（counterbalanced）的进程生命周期中，同时 manifest 中的 `fair_prefix` 块必须完全匹配。该块必须声明每个生命周期 5 个预热波次（`FAIR_WARMUP_WAVES_PER_LIFETIME`）、32 token 的输出、已预热的 prefix cache，以及等于波次数乘以 8（160）的 `samples_per_concurrency`。一个 3 波次的稳态窗口（`FAIR_STEADY_STATE_WINDOW_WAVES`）用于检查预热。

`bench generate|embed|rerank` 是不需要 manifest 的本地诊断工具：在 `--warmup-runs`（默认 1）之后进行 `--runs`（默认 20）次测量，使用贪心解码，并拒绝任何输出发生漂移的运行。`scripts/bench-llama-cpp.sh MODEL.gguf` 运行原生的 `llama-bench`，并读取 `PROMPT_TOKENS`（512）、`OUTPUT_TOKENS`（128）和 `RUNS`（3）。

<sub>来源: [main.rs:352-394](../../../src/main.rs#L352-L394) · [main.rs:560-573](../../../src/main.rs#L560-L573) · [prefix_demo_mac.rs:39-46](../../../src/benchmark/prefix_demo_mac.rs#L39-L46) · [fair_prefix.rs:59-61](../../../src/benchmark/fair_prefix.rs#L59-L61) · [fair_prefix.rs:912-956](../../../src/benchmark/fair_prefix.rs#L912-L956) · +5</sub>

## 添加模型家族

新增一个模型家族只需改动它的 adapter 和一个 dispatch 分支。Adapter 的边界见 [工作原理](04-how-it-works.md)，格式 dispatch 见 [概览与支持矩阵](01-overview.md)。

1. 添加实现 `ModelAdapter` 的 `src/models/<family>.rs`：`family`、`runtime_spec`、`decoder_spec`、`build_model`（通常是 `DecoderModel::load`，张量名称不标准时用 `load_with_weight_names`）和 `format_chat_messages`。
2. 对于 GGUF，添加一个带有明确的可接受元数据键列表的 `<family>_spec` 构建器，并在 `GgufModel::open` 和 `GgufModel::adapter` 中添加 match 分支。对于 Safetensors，在 `detect` 中添加一个 `model_type` 分支。
3. 通过精确的身份识别对话模板，而不是执行 Jinja。例如，Qwen3 adapter 只接受字节长度和 SHA-256 与固定常量一致的模板。只有在验证过 native tool 模板之后才实现 `agent_template_profile`。
4. 在 `tests/` 下添加一个被忽略的门禁测试，按 SHA-256 固定一个公开 fixture，并与 `llama.cpp` 比较贪心 token。

如果新的层类型需要自己的 mixer，就扩展 `SequenceMixerSpec` 和 decoder，并说明 prefix cache 能否为它保存快照。

<sub>来源: [mod.rs:57-100](../../../src/models/mod.rs#L57-L100) · [gguf_model.rs:24-43](../../../src/gguf_model.rs#L24-L43) · [gguf_model.rs:312-365](../../../src/gguf_model.rs#L312-L365) · [qwen3.rs:140-168](../../../src/models/qwen3.rs#L140-L168) · [qwen3.rs:246-251](../../../src/models/qwen3.rs#L246-L251) · +2</sub>
