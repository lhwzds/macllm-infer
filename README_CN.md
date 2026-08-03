# MacLLM Infer

[English](README.md) | [简体中文](README_CN.md)

MacLLM Infer 是一个用 Rust 编写、Apple Silicon 优先的本地 text-only 模型
推理 runtime。第一阶段在同一个 decoder runtime 上提供 `generate`、`embed` 和
`rerank` 三种任务。它以 `llama.cpp` 作为 GGUF、输出行为、内存和性能基线，但不追求
ABI 兼容或完整复刻全部功能。

项目的核心目标是 **architecture-general**：新增 decoder-only 模型时，通常只需
增加 `ModelAdapter` 和真正缺失的算子语义，加载、tokenizer、KV cache、批处理、生成、采样、
continuous batching 与 serving 继续复用同一套 runtime。只有 task semantics 相同时才复用
embedding/reranking execution plumbing；prompt、pooling 与 labels 仍由 checkpoint profile 明确描述。

产品方向不是“Rust 重写一份 llama.cpp”，而是 **Apple Silicon 上的本地 coding-agent
runtime**：`llama.cpp` 提供 GGUF、token、内存与性能基线；MacLLM 重点实现可审计的
Rust serving、Agent API，以及 dense/hybrid 模型都能正确复用的 automatic prefix cache。

## 当前能力

- GGUF v3 单文件与 canonical shards；选定 Llama/Qwen/LFM2 Safetensors profile。
- 12 个 GGUF architecture dispatch：`llama`、`qwen2`、`qwen3`、`qwen3moe`、
  `qwen35`、`gemma`、`gemma2`、`gemma3`、`granitemoe`、`lfm2`、`falcon-h1`、
  `smollm3`。
- Full/sliding GQA、显式 RoPE/NoPE、paged KV、Sparse MoE、ShortConv、Gated DeltaNet、
  Mamba2，以及 CPU/Metal quantized execution。
- 共用 loader/decoder 的 `generate`、`embed`、`rerank` CLI 与对应本机 benchmark。
- Qwen3 Embedding 0.6B：last-token final-normalized hidden state 加 L2 normalization；
  Qwen3 Reranker 0.6B：last-token `yes/no` logits 的 margin 与 sigmoid。
- 共用 generation loop 的严格 text-only
  Chat Completions-compatible server。
- Automatic prefix cache：token-block exact identity、paged-KV refcount/COW、bounded LRU，
  并把 ShortConv、Gated DeltaNet、Mamba2 recurrent state 与 KV 一起快照和恢复。
- Agent API：严格 JSON Schema constrained sampling、stop sequences、Qwen3/Qwen3MoE
  forced-single native tools/tool history，以及 OpenAI usage 中的 `cached_tokens`。
- Manifest-pinned MacLLM/llama.cpp scoreboard、exact-16K repository/system prefix 的
  8-request physical page-reuse demo，以及同 transport/precision/sample contract 的
  `fair-prefix` serving 对照。
- 支持结论按 checkpoint profile 发布，不把一个 adapter 扩大成整个模型家族承诺。

最新的 SmolLM3 垂直切片固定官方 `SmolLM3-Q4_K_M.gguf` 的 revision、大小与
SHA-256，覆盖 `smaug-bpe` tokenizer、每四层一次的显式 NoPE、CPU/Metal cache/full，
以及 llama.cpp build 10050 的 raw 与 deterministic-chat greedy oracle。所有支持结论都以
对应 integration test 中固定的 checkpoint revision、文件 identity 和独立 oracle 为边界；
同架构的其他尺寸、量化或模板 revision 不会自动升级为正式支持。

## 快速开始

```bash
cargo build --release

./target/release/macllm inspect --model /path/to/model.gguf

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

对未知 chat template，可显式使用 `--raw-prompt`；MacLLM 不执行任意 Jinja，也不会
静默猜测模板。默认设备是 Metal，CPU 可用 `--device cpu`。

`run` 与 `bench generate` 可用 `--prefill-token-quantum TOKENS` 限制一次 blocking prefill 的
prompt token 数。未显式设置时，Metal（显式 BF16 除外）使用 128，CPU 与显式 BF16
保持 one-shot；显式正整数（包括大于 128 的值）不会被截断。这个默认值用于控制
Metal prefill 的工作集，不代表 128 在所有模型和 prompt 上都更快。

## 可重复对比

固定 manifest 会校验 Mac 型号、模型 bytes/SHA、prompt token IDs、量化、llama.cpp
commit 与二进制。完整 scoreboard 分开报告 MacLLM kernel、llama-bench throughput-only、
两侧 serving/TTFT/memory 与 greedy parity，不把不同 measurement boundary 混成一个数字：

```bash
./target/release/macllm scoreboard verify \
  --manifest benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json

./target/release/macllm scoreboard report \
  --manifest benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json \
  --json /tmp/macllm-scoreboard.json \
  --markdown /tmp/macllm-scoreboard.md
```

两个 exact-16K 命令回答不同问题。`prefix-cache` 的 schema v4 报告用于证明 MacLLM
first-admission 的物理 page/refcount 复用；它的 MacLLM 侧是 direct `EngineHandle`，
llama.cpp 侧是 loopback HTTP/SSE，因此 transport 不同，报告会 fail closed 且不发布速度
ratio。llama.cpp `cache_n` 也只作为逻辑 counter，不会被换算成 MacLLM 物理页节省：

```bash
./target/release/macllm demo prefix-cache \
  --manifest benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json \
  --repository . \
  --json /tmp/macllm-prefix-demo.json \
  --markdown /tmp/macllm-prefix-demo.md
```

`fair-prefix` 才是受控 serving four-pair point estimate 的发布入口。专用 manifest 的
`fair_prefix` 段把真实
prompt source 固定为 `prepared_repository_bundle`；通用 `workload.prompt` 只保留作 scoreboard
compatibility metadata。两侧使用相同 raw `POST /completion`、loopback HTTP/SSE、F32 K/V、
8 并发、exact 16K prefix、每请求 32 个输出 token，并各保留 20 个 measured waves。shared
request 还固定 `response_fields` projection，禁止 llama.cpp terminal 回传完整 16K prompt 或
未约定 metadata；两侧 raw SSE 都关闭 keepalive，fragment 与 terminal 的字段、token accounting、
generation settings 和 cache timings schema 会逐事件验证，未知/额外字段 fail closed。每个
measured stream 还保留来自真实 generation 状态机的 32 个 token IDs；同一 task 必须在所有
waves 以及 MacLLM/llama.cpp 之间逐项一致：

```bash
./target/release/macllm demo fair-prefix \
  --manifest benchmarks/manifests/smollm3-3b-q4_k_m-m4pro-fair-prefix.json \
  --repository . \
  --port 18086 \
  --json /tmp/macllm-fair-prefix.json \
  --markdown /tmp/macllm-fair-prefix.md
```

每侧的 20 个 waves 分布在 4 个全新 OS process lifetimes 中，每个 lifetime 保留 5 个 waves；
四组 pair 按 `AB / BA / BA / AB`（A=`llama.cpp`，B=`MacLLM`）执行。每次 seed 后固定稳定
30 秒，再运行 5 个不计入结果的 warm-up waves。只有最后 3 个 warm-up 的 aggregate TPS
同时满足变异系数 `CV <= 10%` 和首尾 relative trend `<= 10%`，该 process lifetime 才进入
measurement；否则命令直接失败、没有结果。两侧每个 lifetime 都做 8 次 seed，并在每个 warm-up
及每个 measured wave 后各做 8 次 excluded seed-state restore，因此每侧总计 32 次 seed 和
`4 × (5 warm-up + 5 measured) × 8 = 320` 次 conditioning/restore；这使波次间 conditioning
请求 body 与数量完全对称。restore 使用“另一个 task 的 prompt”，并在 token 层 fail closed
证明它与下一条 measured task 恰好只共享 16K；所以
两侧 restore 都是 `cache_n=16,384` 且执行相同 suffix 工作。

初始 seed 的计算量并不对称：llama.cpp 的 8 个 fixed slots 都 cold-prefill 16K；MacLLM
第一次 cold-prefill，后 7 次复用 16,368 tokens、只重算最后一个 16-token block。seed 全部
不计时，且位于 30 秒稳定期、5 个对称 warm-up waves 和稳态门之前；报告明确保留这条边界，
而不是把 seed 阶段说成等算量。双方 measured/conditioning 请求都发送 `cache_prompt=true`；
MacLLM 的
physical KV pool 是 65,536 tokens，但 prefix-snapshot budget 精确限制为 16,384 tokens，
因此 measured task 不能把 cache endpoint 推到更长的 task-specific prefix。

这个 lane 的名字与边界是 `controlled_exact_slot_cache_lane`，不是 product-default 或
best-tuned 产品对比。llama.cpp 的 `--no-kv-unified --cache-ram 0` 以及
`--batch-size 1024 --ubatch-size 128` 是为了固定 exact-slot cache 与 batch 几何的控制变量，
不是宣称它们是 llama.cpp 的最优参数。未来 best-tuned lane 必须先在独立 calibration set
上各自选参并冻结配置，再只在 held-out measurement waves 上比较。

每个 MacLLM measured process 都从 `/healthz` 证明已加载 model 且实际
`device=metal`、`dtype=f32`。所有 llama.cpp measured phases 则使用 `--log-disable`，避免
verbosity logging 污染计时；全部 measured phases 结束后，才单独启动一个不计时的 `-lv 4`
attestation process，证明 `MTL0`、全层 GPU offload、MTL0 F32 K/V、resolved Flash
Attention，以及实际 slots/context/batch/ubatch。它与 measured llama.cpp 的 binary、model
和所有非 logging argv 必须完全相同。两侧启动命令的 model、precision、cache、batch 与
transport argv 也必须与 report contract 逐项相等。所有 measured/control child 都清空 inherited
environment，只保留 contract 固定的 `LANG=C`、`LC_ALL=C` 与最小系统 `PATH`。

llama-server executable 与 model file 都在长跑期间由 open-fd、
device/inode/metadata 与双 SHA-256 守卫；任何 attestation 或 provenance 漂移都会 fail closed。
当前 provenance 仍有明确限制：报告 pin 了 Homebrew llama-server executable，但尚未 hash
其 non-system dynamic-library closure，不能把它当作 supply-chain-complete provenance。

报告中的 pooled p50/p95 只作描述。正式数值是四组 paired phase-throughput ratios 几何平均
得到的 **four-pair point estimate**；每个 phase 的 throughput 都是该 lifetime 的“输出 token
总数 / external wall time 总和”，不是 phase p50。四对样本没有给出可靠 uncertainty interval，
所以即使 point estimate 大于 1，也不能写成 uncertainty-qualified “MacLLM 更快”结论。它只
描述这一台 host 上这次 exact-run，不是长期 SLA。runner 也不采集或门控电源来源、Low Power
Mode、thermal pressure/clock state 与后台负载；counterbalance、cooldown 和稳态门只能缓解
顺序与漂移，不能证明主机环境完全等价。canonical scope/uncertainty warnings 也是
schema 的受验证内容，任意改写都会使 report validation 失败。两个发布级
命令都要求 release executable 与源码来自同一个 clean commit，并在长跑前后保持不变。
成功时 JSON/Markdown 在同一目录以临时文件、`fsync`、rename 和失败回滚成对发布；
Markdown 顶部写入 authoritative JSON companion 的 SHA-256，避免把不同一次运行的两个文件
误当成一对。
历史旧口径中的 `9.712x` 是不公平的历史结果，不能替代新的 eligibility-gated
`fair-prefix` artifact。

## 验证

```bash
cargo fmt --all -- --check
cargo check --all-targets
cargo check --all-targets --no-default-features
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

真实模型 gates 标为 `#[ignore]`，需要对应 fixture 与设备。其中
`tests/qwen3_tasks_reference.rs` 使用 SHA-pinned Safetensors/sidecars，将 embed/rerank
分别与独立 Candle Qwen3 图和官方示例比对。每个真实 gate 的源码会明确给出所需环境变量、
fixture identity、允许误差和 llama.cpp/Candle oracle，避免把普通单元测试误写成真实模型证明。

代码采用 MIT License；模型权重遵循各自仓库的许可证或使用条款。
