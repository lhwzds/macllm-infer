---
title: 命令行用法
---

# 命令行用法

<sub>[English](../02-cli.md) · **中文**</sub>

`macllm` 二进制有九个子命令：`download`、`inspect`、`run`、`embed`、`rerank`、`bench`、`scoreboard`、`demo` 和 `serve`。下文所有参数都照搬自 `src/main.rs` 中的 clap 定义。运行时不读取任何环境变量；代码中的 `MACLLM_*` 变量只用于让被忽略的参考测试找到 fixture 文件（见 [Benchmark、门禁测试与贡献](05-benchmarks-and-testing.md)）。`serve` 见 [HTTP 服务与 API 子集](03-http-api.md)；`bench`、`scoreboard` 和 `demo` 见 [Benchmark、门禁测试与贡献](05-benchmarks-and-testing.md)。

<sub>来源: [main.rs:44-78](../../../src/main.rs#L44-L78)</sub>

## 模型、设备和 K/V 参数

所有加载模型的命令共用 `RuntimeArgs`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `-m, --model` | 必填 | `.gguf` 文件或模型目录 |
| `--device` | `metal` | `metal` 或 `cpu` |
| `--dtype` | 见下文 | 激活值使用 `f32`、`f16`、`bf16` |
| `--kv-cache-block-tokens` | `16` | 每个逻辑 K/V block 的 token 数 |
| `--kv-cache-slab-blocks` | `8` | 每次物理 slab 分配的 block 数 |
| `--kv-cache-capacity-tokens` | 一个模型上下文 | 所有序列共享的 K/V token 总数，向上取整到一个 block |

`resolve_dtype` 在 CPU 上或 GGUF 含有量化张量时选择 `f32`，否则选择 `f16`。CPU/Accelerate 后端只接受 `f32`。

<sub>来源: [main.rs:80-106](../../../src/main.rs#L80-L106) · [runtime.rs:34-50](../../../src/runtime.rs#L34-L50) · [runtime.rs:162-175](../../../src/runtime.rs#L162-L175) · [backend.rs:30-37](../../../src/backend.rs#L30-L37)</sub>

## 用 run 生成文本

`run` 通过 adapter 的对话模板渲染 `--prompt`（以及可选的 `--system`），完成 tokenization 后把解码出的文本流式输出到 stdout。Adapter 只渲染它能识别的模板，否则直接失败；`--raw-prompt` 会绕过模板，原样对 prompt 做 tokenization。`--add-special-tokens` 让 tokenizer 添加 BOS/EOS；对于本身已经包含 BOS 的模板，不要打开它。

采样默认值：`--max-new-tokens 128`、`--temperature 0.7`（0 表示贪心 argmax）、`--top-k 20`、`--top-p 0.8`、`--repetition-penalty 1.1`、`--seed 42`。

`--prefill-token-quantum` 限制每次 prefill 前向计算的 prompt token 数。不设置时，Metal 使用 128（`--dtype bf16` 除外）；CPU 和 BF16 一次完成 prefill。生成结束后，stderr 上会输出一行指标，报告 prompt、cached 和 processed token 数、TTFT，以及 prefill 和 decode 的每秒 token 数。

<sub>来源: [main.rs:108-160](../../../src/main.rs#L108-L160) · [main.rs:1220-1255](../../../src/main.rs#L1220-L1255) · [main.rs:1285-1300](../../../src/main.rs#L1285-L1300)</sub>

## 向量嵌入和重排序

`embed` 和 `rerank` 只支持 Safetensors 格式的 Qwen3 任务 checkpoint。`Qwen3EmbeddingProfile` 和 `Qwen3RerankerProfile` 会检查 sentence-transformers 文件（`modules.json`、`config_sentence_transformers.json`，以及 pooling 或 logit-score 模块配置）、EOS token、last-token pooling 或互不相同的 yes/no 标签 ID；随后 `validate_task_checkpoint` 要求必须是 Safetensors 格式。这些任务只能通过 CLI 使用。

- `embed --input-type query|document --input TEXT [--input TEXT ...] [--instruction TEXT]`；`--instruction` 只对 query 有效。
- `rerank --query TEXT --document TEXT [--document TEXT ...] [--instruction TEXT]`。

超过 `min(max_position_embeddings, 8192)` 个 token 的输入会被拒绝，而不是截断。输出是 stdout 上格式化后的 JSON，按输入的 `index` 排序，不回显输入。向量嵌入的每一行带有单位长度的 `embedding`；重排序的每一行带有 `margin`（`yes_logit - no_logit`）和 `yes_probability`（其 sigmoid 值）。

<sub>来源: [main.rs:172-232](../../../src/main.rs#L172-L232) · [main.rs:808-850](../../../src/main.rs#L808-L850) · [main.rs:865-923](../../../src/main.rs#L865-L923) · [main.rs:925-968](../../../src/main.rs#L925-L968) · [qwen3_tasks.rs:1-40](../../../src/models/qwen3_tasks.rs#L1-L40) · +5</sub>

## 下载和检查 checkpoint

`download [REPO]`（默认 `Qwen/Qwen2.5-0.5B-Instruct`，输出 `-o models/qwen2.5-0.5b-instruct`，可选 `--revision`）按固定的 Safetensors 允许列表下载：config、generation 和 tokenizer JSON，`model*.safetensors` 及其 index，以及 sentence-transformers 的附属文件。`--gguf FILENAME` 则改为下载仓库根目录下的一个 GGUF 文件；如果给的是规范的第一个分片，就下载全部分片。文件先落到临时暂存目录，必须通过 `ModelFiles::from_dir` 校验，之后才会替换目标目录。`.macllm-revision` 标记文件记录解析出的 commit。

`inspect -m PATH` 只解析元数据而不加载权重，输出模型家族、层结构、每个 token 的 F16 K/V 字节数、每个序列固定的循环状态大小、最大上下文、词表、stop token ID、格式，以及各 GGUF 张量类型的数量。

<sub>来源: [main.rs:46-62](../../../src/main.rs#L46-L62) · [hub.rs:14-47](../../../src/hub.rs#L14-L47) · [hub.rs:70-120](../../../src/hub.rs#L70-L120) · [main.rs:710-757](../../../src/main.rs#L710-L757)</sub>
