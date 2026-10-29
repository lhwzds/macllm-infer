# MacLLM Infer

[English](README.md) | [简体中文](README_CN.md)

MacLLM Infer 是一个用 Rust 编写、Apple Silicon 优先的本地纯文本 LLM 推理引擎。
它在同一套 decoder runtime 上运行生成、向量嵌入和重排序任务。

项目核心采用通用架构：模型配置、对话模板和权重映射放在 adapter 中；加载、
tokenization、批处理、缓存、采样和服务层由所有模型复用。项目将 `llama.cpp` 作为 GGUF
兼容性、输出正确性、内存和性能基线，但不追求 ABI 兼容。

> MacLLM Infer 目前是面向研究和本地 Agent 工作负载的早期项目。支持结论以具体
> checkpoint 为边界，不代表整个模型家族都已得到支持。

## 核心能力

- Apple Metal 和 CPU 执行，支持量化 GGUF 模型。
- GGUF v3 单文件、规范分片，以及部分 Safetensors profile。
- 12 种架构 dispatch：`llama`、`qwen2`、`qwen3`、`qwen3moe`、`qwen35`、
  `gemma`、`gemma2`、`gemma3`、`granitemoe`、`lfm2`、`falcon-h1` 和 `smollm3`。
- Full/sliding-window GQA、RoPE/NoPE、Sparse MoE、ShortConv、Gated DeltaNet、
  Mamba2 和 paged KV cache。
- Continuous batching，以及带有 bounded LRU、引用计数和 Copy-on-Write 的
  exact-token automatic prefix cache。
- 通过共用同一 runtime 的 `run`、`embed` 和 `rerank` 命令完成生成、向量嵌入和重排序。
- 纯文本 Chat Completions-compatible API 子集，支持受约束 JSON 生成、stop sequence、
  已适配 Qwen3/Qwen3MoE profile 的 native tool call（forced，或 `tool_choice=auto`，
  每轮最多一个受 grammar 约束的调用），以及 cached-token 统计。
- 固定模型和 runtime provenance 的 MacLLM/llama.cpp 可重复 benchmark。

当前已经完成真实验证的垂直切片包括 SmolLM3-3B Q4_K_M 生成，以及 Qwen3
Embedding/Reranker 0.6B。本仓库不包含模型权重。

## 快速开始

```bash
cargo build --release

./target/release/macllm run \
  --model /path/to/model.gguf \
  --prompt 'The capital of France is' \
  --max-new-tokens 32

./target/release/macllm embed \
  --model /path/to/Qwen3-Embedding-0.6B \
  --input-type query \
  --input 'What is the capital of China?'

./target/release/macllm rerank \
  --model /path/to/Qwen3-Reranker-0.6B \
  --query 'What is the capital of China?' \
  --document 'The capital of China is Beijing.' \
  --document 'Gravity attracts bodies.'

./target/release/macllm serve \
  --model /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080
```

默认设备是 Metal；使用 `--device cpu` 可以切换到 CPU。遇到未知 chat template 时，
请显式传入 `--raw-prompt`。MacLLM 不执行任意 Jinja，也不会静默猜测模板。

面向 agent 客户端时，`serve --default-max-completion-tokens N` 设置未指定输出长度的请求的
默认上限（默认 128），`--disable-thinking` 以 `enable_thinking=false` 渲染 Qwen3 prompt；
单个请求也可以发送 `chat_template_kwargs: {"enable_thinking": false}`。

## Benchmark

仓库中的 manifest 会固定主机、模型 bytes/SHA-256、prompt token IDs、量化方式和
llama.cpp build。示例 manifest 对应已经记录的 M4 Pro 环境；其他主机需要使用与本机
匹配的 manifest。

```bash
./target/release/macllm scoreboard verify \
  --manifest benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json
```

`fair-prefix` 会控制 transport、K/V precision、prefix 长度、并发数、输出长度和
token parity。结果只是指定主机上一次精确运行的 point estimate，不能泛化成
“MacLLM 比 llama.cpp 更快”的结论。完整用法请运行
`./target/release/macllm demo fair-prefix --help`。

## 验证

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- \
  -D warnings -A clippy::too_many_lines -A clippy::doc_markdown
cargo test --all-targets
```

真实模型 integration tests 标记为 `#[ignore]`，因为它们需要固定的模型 fixture；
Metal 测试还需要 Apple GPU。

## 范围与安全

- 当前 runtime 仅支持纯文本；视觉、音频、OCR 和多模态模型暂不在范围内。
- 模型加载期间应将模型目录视为可信且不可变。
- 服务默认绑定 loopback。部署到公网时必须使用带认证、TLS 和限流的反向代理。
- 源码采用 MIT License；模型权重遵循各自的许可证和使用条款。
