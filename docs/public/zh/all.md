---
title: macllm-infer — 全书单页
---

# macllm-infer

<sub>[English](../all.md) · **中文**</sub>

这套文档比 MacLLM Infer 的 README 更深入一层。MacLLM Infer 是一个用 Rust 编写、
Apple Silicon 优先的纯文本 LLM 运行时。文档说明当前提交的代码实际支持什么（哪些格式和架构、
哪些 CLI 参数和 HTTP 字段、哪些默认值），一个请求如何从模型加载开始，依次经过服务调度器、
paged K/V cache 和 prefix cache，最终到达 Metal kernel，以及固定的参考测试和 benchmark
manifest 如何为正确性和性能结论把关。文档只描述当前能力。

<a id="overview"></a>

## 概览与支持矩阵

MacLLM Infer 是一个单独的 Rust crate（`macllm-infer`，二进制为 `macllm`），通过 Candle 在 Apple Silicon 上运行纯文本 decoder LLM，默认 feature 为 Metal 和 Accelerate。所有支持的模型家族都由同一个共享 decoder 执行；家族相关代码仅限于很薄的 adapter，负责把 checkpoint 的配置、权重名称和 prompt 模板转换成与格式无关的 spec。同一个已加载的模型既用于阻塞式 CLI 生成、Qwen3 向量嵌入和重排序，也用于长期运行的 HTTP worker。`llama.cpp` 被用作 GGUF 解析、贪心解码 token 的正确性和性能的参考，但从不作为链接依赖。

本页准确说明代码接受什么。用法见 [命令行用法](#cli)，设计见 [工作原理](#how-it-works)，模块结构见 [代码结构](#structure)。

<sub>来源: [Cargo.toml:1-22](../../../Cargo.toml#L1-L22) · [lib.rs:1-7](../../../src/lib.rs#L1-L7) · [mod.rs:1](../../../src/benchmark/mod.rs#L1)</sub>

### 支持的格式和架构

`ModelFiles::from_dir` 接受单个 `.gguf` 文件或一个目录。同时包含 GGUF 和 Safetensors 权重的目录会被拒绝；其他情况下发现文件时 GGUF 优先，规范的 GGUF 分片集合会作为一个逻辑文件打开。

两种格式覆盖的架构并不相同：

| 格式 | Dispatch 键 | 架构 |
| --- | --- | --- |
| GGUF v3 | `GgufModel::open` 中的 `general.architecture` | `falcon-h1`、`gemma`、`gemma2`、`gemma3`、`granitemoe`、`lfm2`、`llama`、`qwen2`、`qwen3`、`qwen3moe`、`qwen35`、`smollm3` |
| Safetensors | `config.json` 中的 `model_type`（`detect`） | `llama`、`lfm2`、`qwen2`、`qwen3` |

因此 12 种架构的列表只适用于 GGUF。Safetensors 目录还需要 `config.json` 和 `tokenizer.json`；GGUF 使用内嵌的 tokenizer。`GgufTensorType` 接受的 GGUF 张量类型为 F32、F16、BF16、Q4_0、Q5_0、Q5_1、Q8_0，以及 Q2_K 到 Q6_K。

<sub>来源: [weights.rs:58-115](../../../src/weights.rs#L58-L115) · [mod.rs:102-118](../../../src/models/mod.rs#L102-L118) · [gguf_model.rs:271-300](../../../src/gguf_model.rs#L271-L300) · [gguf.rs:283-296](../../../src/gguf.rs#L283-L296)</sub>

### 当前限制一览

支持范围以具体 checkpoint 为边界，不满足条件时直接失败（fail closed）。新用户最先遇到的边界：

- 向量嵌入和重排序只接受 Safetensors 格式的 Qwen3 任务 checkpoint（`validate_task_checkpoint`），并且只以 `embed`/`rerank` CLI 命令的形式提供；HTTP 路由中没有向量嵌入或重排序路由。见 [命令行用法](#cli)。
- 服务暴露 `/healthz`、`/v1/models`、`/v1/chat/completions` 和一个用于 benchmark 的 `/completion` 路由。`tool_choice=auto` 会以未实现为由被拒绝；只支持强制调用单个工具。见 [HTTP 服务与 API 子集](#http-api)。
- 自动 prefix cache 与 sliding-window attention 层互斥：在含有 sliding 层的模型上启用该缓存会在加载时失败。见 [工作原理](#how-it-works)。
- CPU 后端要求 `--dtype f32`。
- 正确性结论来自按 checkpoint 固定的参考测试，例如 Gemma 1 2B Q2_K 和 SmolLM3-3B Q4_K_M。见 [Benchmark、门禁测试与贡献](#benchmarks-and-testing)。

<sub>来源: [main.rs:910-923](../../../src/main.rs#L910-L923) · [server.rs:123-131](../../../src/server.rs#L123-L131) · [openai.rs:626-674](../../../src/openai.rs#L626-L674) · [decoder.rs:4319-4340](../../../src/decoder.rs#L4319-L4340) · [backend.rs:30-37](../../../src/backend.rs#L30-L37)</sub>

<a id="cli"></a>

## 命令行用法

`macllm` 二进制有九个子命令：`download`、`inspect`、`run`、`embed`、`rerank`、`bench`、`scoreboard`、`demo` 和 `serve`。下文所有参数都照搬自 `src/main.rs` 中的 clap 定义。运行时不读取任何环境变量；代码中的 `MACLLM_*` 变量只用于让被忽略的参考测试找到 fixture 文件（见 [Benchmark、门禁测试与贡献](#benchmarks-and-testing)）。`serve` 见 [HTTP 服务与 API 子集](#http-api)；`bench`、`scoreboard` 和 `demo` 见 [Benchmark、门禁测试与贡献](#benchmarks-and-testing)。

<sub>来源: [main.rs:44-78](../../../src/main.rs#L44-L78)</sub>

### 模型、设备和 K/V 参数

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

### 用 run 生成文本

`run` 通过 adapter 的对话模板渲染 `--prompt`（以及可选的 `--system`），完成 tokenization 后把解码出的文本流式输出到 stdout。Adapter 只渲染它能识别的模板，否则直接失败；`--raw-prompt` 会绕过模板，原样对 prompt 做 tokenization。`--add-special-tokens` 让 tokenizer 添加 BOS/EOS；对于本身已经包含 BOS 的模板，不要打开它。

采样默认值：`--max-new-tokens 128`、`--temperature 0.7`（0 表示贪心 argmax）、`--top-k 20`、`--top-p 0.8`、`--repetition-penalty 1.1`、`--seed 42`。

`--prefill-token-quantum` 限制每次 prefill 前向计算的 prompt token 数。不设置时，Metal 使用 128（`--dtype bf16` 除外）；CPU 和 BF16 一次完成 prefill。生成结束后，stderr 上会输出一行指标，报告 prompt、cached 和 processed token 数、TTFT，以及 prefill 和 decode 的每秒 token 数。

<sub>来源: [main.rs:108-160](../../../src/main.rs#L108-L160) · [main.rs:1220-1255](../../../src/main.rs#L1220-L1255) · [main.rs:1285-1300](../../../src/main.rs#L1285-L1300)</sub>

### 向量嵌入和重排序

`embed` 和 `rerank` 只支持 Safetensors 格式的 Qwen3 任务 checkpoint。`Qwen3EmbeddingProfile` 和 `Qwen3RerankerProfile` 会检查 sentence-transformers 文件（`modules.json`、`config_sentence_transformers.json`，以及 pooling 或 logit-score 模块配置）、EOS token、last-token pooling 或互不相同的 yes/no 标签 ID；随后 `validate_task_checkpoint` 要求必须是 Safetensors 格式。这些任务只能通过 CLI 使用。

- `embed --input-type query|document --input TEXT [--input TEXT ...] [--instruction TEXT]`；`--instruction` 只对 query 有效。
- `rerank --query TEXT --document TEXT [--document TEXT ...] [--instruction TEXT]`。

超过 `min(max_position_embeddings, 8192)` 个 token 的输入会被拒绝，而不是截断。输出是 stdout 上格式化后的 JSON，按输入的 `index` 排序，不回显输入。向量嵌入的每一行带有单位长度的 `embedding`；重排序的每一行带有 `margin`（`yes_logit - no_logit`）和 `yes_probability`（其 sigmoid 值）。

<sub>来源: [main.rs:172-232](../../../src/main.rs#L172-L232) · [main.rs:808-850](../../../src/main.rs#L808-L850) · [main.rs:865-923](../../../src/main.rs#L865-L923) · [main.rs:925-968](../../../src/main.rs#L925-L968) · [qwen3_tasks.rs:1-40](../../../src/models/qwen3_tasks.rs#L1-L40) · +5</sub>

### 下载和检查 checkpoint

`download [REPO]`（默认 `Qwen/Qwen2.5-0.5B-Instruct`，输出 `-o models/qwen2.5-0.5b-instruct`，可选 `--revision`）按固定的 Safetensors 允许列表下载：config、generation 和 tokenizer JSON，`model*.safetensors` 及其 index，以及 sentence-transformers 的附属文件。`--gguf FILENAME` 则改为下载仓库根目录下的一个 GGUF 文件；如果给的是规范的第一个分片，就下载全部分片。文件先落到临时暂存目录，必须通过 `ModelFiles::from_dir` 校验，之后才会替换目标目录。`.macllm-revision` 标记文件记录解析出的 commit。

`inspect -m PATH` 只解析元数据而不加载权重，输出模型家族、层结构、每个 token 的 F16 K/V 字节数、每个序列固定的循环状态大小、最大上下文、词表、stop token ID、格式，以及各 GGUF 张量类型的数量。

<sub>来源: [main.rs:46-62](../../../src/main.rs#L46-L62) · [hub.rs:14-47](../../../src/hub.rs#L14-L47) · [hub.rs:70-120](../../../src/hub.rs#L70-L120) · [main.rs:710-757](../../../src/main.rs#L710-L757)</sub>

<a id="http-api"></a>

## HTTP 服务与 API 子集

`macllm serve` 在一个专用 worker 线程上加载一个模型，并用 axum 通过 HTTP/1 提供服务，直到按下 Ctrl-C。它默认绑定 loopback，自身不带认证或 TLS。向量嵌入和重排序不通过服务提供，请使用 CLI（[命令行用法](#cli)）。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8080` | 绑定地址 |
| `--served-model-name` | 模型路径的文件名 | 客户端必须作为 `model` 发送的 ID |
| `--queue-capacity` | `8` | 等待活跃槽位的请求数 |
| `--max-active-sequences` | `2` | 在 worker 上轮转的序列数 |
| `--scheduler-token-budget` | `256` | 每轮调度的模型输入 token 数 |
| `--prefill-token-quantum` | `128` | 每个序列每轮的 prompt token 数 |
| `--event-capacity` | `16` | 每个请求的事件通道大小 |
| `--event-send-timeout-seconds` | `30` | 客户端停止读取时取消请求 |
| `--prefix-cache-capacity-tokens` | `32768` | prefix cache 预算；`0` 表示禁用 |

共用的模型参数（`--model`、`--device`、`--dtype`、`--kv-cache-*`）与 [命令行用法](#cli) 中相同。

<sub>来源: [main.rs:396-429](../../../src/main.rs#L396-L429) · [main.rs:1257-1283](../../../src/main.rs#L1257-L1283) · [server.rs:123-150](../../../src/server.rs#L123-L150)</sub>

### 端点

| 路由 | 用途 |
| --- | --- |
| `GET /healthz` | `status`、`worker`（`starting`、`ready`、`busy`、`poisoned`、`stopped`）、`model`、`device`、`dtype`；不健康时返回 503 |
| `GET /v1/models` | 唯一的服务模型 |
| `POST /v1/chat/completions` | 严格的 Chat Completions 子集，JSON 或 SSE |
| `POST /completion` | 不套模板、与 llama-server 形状一致的路由，供 benchmark 使用 |

`/completion` 要求提供 `RawCompletionRequest` 的每一个字段（`prompt`、`n_predict`、`temperature`、`top_k`、`top_p`、`seed`、`cache_prompt`、`stream`、`return_tokens`、`id_slot`、`ignore_eos`、`repeat_penalty`、`response_fields`），并拒绝未知字段，这样 benchmark 中的两个服务端遵循完全相同的约定。

队列已满时返回 429；worker 已停止或处于 poisoned 状态时返回 503。流式响应每 15 秒发送一条 keep-alive 注释，并以 `data: [DONE]` 结束。

<sub>来源: [server.rs:123-131](../../../src/server.rs#L123-L131) · [server.rs:156-185](../../../src/server.rs#L156-L185) · [server.rs:219-236](../../../src/server.rs#L219-L236) · [server.rs:1265-1276](../../../src/server.rs#L1265-L1276) · [server.rs:1088-1096](../../../src/server.rs#L1088-L1096) · +1</sub>

### /v1/chat/completions 接受哪些字段

未知 JSON 字段返回 400；已知但不支持的字段返回 OpenAI 格式的错误，而不是被丢弃。

- `model` 必须等于服务的模型名；`n` 必须为 1。
- `max_completion_tokens`（或已弃用的 `max_tokens`，两者不能同时出现）默认为 128，且不能超过模型上下文。
- `temperature` 默认 1.0，范围 [0, 2]；`top_p` 默认 1.0，范围 (0, 1]；`seed` 必须非负。服务端把 `top_k` 固定为 0，重复惩罚固定为 1.0。
- `stop` 接受一个字符串或最多 4 个字符串。
- `stream` 支持 `stream_options.include_usage`；`include_obfuscation=true` 会被拒绝。
- 角色：`system`、`user`、`assistant`。`developer`、`name`、多模态内容、`logprobs`、`top_logprobs` 和非零的 `frequency_penalty`/`presence_penalty` 都会被拒绝。

`usage.prompt_tokens_details.cached_tokens` 报告有多少 prompt token 来自 prefix cache。

<sub>来源: [openai.rs:19-72](../../../src/openai.rs#L19-L72) · [openai.rs:90-170](../../../src/openai.rs#L90-L170) · [openai.rs:489-515](../../../src/openai.rs#L489-L515) · [openai.rs:815-870](../../../src/openai.rs#L815-L870) · [server.rs:793-802](../../../src/server.rs#L793-L802) · +4</sub>

### 结构化输出和 tool call

这两项功能都通过 `src/grammar.rs` 中的受约束解码 grammar 实现，它会把每个候选 token 解码后的字节与一个封闭的 JSON Schema 子集进行比对。未知的 schema 关键字会被拒绝，object schema 必须设置 `additionalProperties: false`。

- `response_format` 只接受 `{"type":"json_schema","json_schema":{...,"strict":true}}`，不能与 `stop`、工具或 tool call 历史同时使用。
- 工具：`tool_choice=auto` 尚未实现，会返回错误。声明了工具的请求默认是 `auto`，因此客户端必须显式设置 `tool_choice`。可接受的值：指定函数、只声明一个工具时的 `"required"`，或 `"none"`。需设置 `parallel_tool_calls=false`；强制调用时不允许使用 `stop`。强制调用的参数在其 parameters schema 的约束下生成。
- 只有对话模板哈希与固定 profile 匹配的 Qwen3 和 Qwen3MoE adapter 才有 native tool prompt。其他 adapter 在 `format_agent_prompt` 中直接失败（fail closed）。

<sub>来源: [openai.rs:517-530](../../../src/openai.rs#L517-L530) · [openai.rs:572-674](../../../src/openai.rs#L572-L674) · [server.rs:803-830](../../../src/server.rs#L803-L830) · [grammar.rs:1-15](../../../src/grammar.rs#L1-L15) · [qwen3.rs:264-277](../../../src/models/qwen3.rs#L264-L277) · +3</sub>

### 启动检查与 prefix cache 参数

`EngineHandle::start` 只有在 worker 加载完模型后才返回，因此任何一项检查失败，`serve` 都会在绑定端口之前退出：

1. 模型必须支持 packed ragged batch。
2. Adapter 必须能通过它识别的模板渲染一个健康检查用的 chat prompt，并且该 prompt 至少要 tokenize 出一个 token。没有受支持模板的 checkpoint 可以用 `run --raw-prompt` 运行，但不能提供服务。
3. 所有 `--max-active-sequences` 个会话都会预先创建。

`serve` 默认启用自动 prefix cache（32768 个 token），而该缓存不能与 sliding-window attention 共存（[工作原理](#how-it-works)）。因此任何含有 `SlidingCausal` 层的模型都会加载失败，报错 "sliding-attention storage is mutable"。Gemma 2 和 Gemma 3 通常会交替使用这类层，GGUF Llama 在 `llama.attention.sliding_window` 非零时也会有这类层。为这些模型提供服务时请使用 `--prefix-cache-capacity-tokens 0`。

<sub>来源: [engine.rs:1141-1245](../../../src/engine.rs#L1141-L1245) · [runtime.rs:94-97](../../../src/runtime.rs#L94-L97) · [decoder.rs:3359-3369](../../../src/decoder.rs#L3359-L3369) · [decoder.rs:4319-4340](../../../src/decoder.rs#L4319-L4340) · [gguf_model.rs:1322-1325](../../../src/gguf_model.rs#L1322-L1325) · +3</sub>

<a id="how-it-works"></a>

## 工作原理

每个命令都经过 `LoadedRuntime::load`：打开文件，选择 adapter，从 `WeightSource` 构建共享 decoder，然后配置 paged K/V 池和 prefix cache。阻塞式命令直接调用模型。`serve` 在一个 worker 线程中构建同样的运行时，该线程在整个生命周期内独占模型。

```mermaid
flowchart LR
  F["ModelFiles<br/>GGUF or Safetensors"] --> A["ModelAdapter<br/>src/models/*"]
  A --> S["DecoderSpec"]
  S --> D["DecoderModel<br/>src/decoder.rs"]
  W["WeightSource<br/>mmap / quantized"] --> D
  D --> KV["Paged K/V pool<br/>+ prefix cache"]
  H["axum handlers<br/>src/server.rs"] -->|EngineRequest| E["Worker thread<br/>ServingScheduler"]
  E --> D
  C["CLI run/embed/rerank"] --> D
```

<sub>来源: [lib.rs:1-7](../../../src/lib.rs#L1-L7) · [runtime.rs:62-103](../../../src/runtime.rs#L62-L103) · [engine.rs:773-805](../../../src/engine.rs#L773-L805)</sub>

### Adapter 与共享 decoder

`ModelAdapter` 是唯一存放模型家族代码的地方。Adapter 把配置或 GGUF 元数据转换成 `RuntimeSpec` 和 `DecoderSpec`，映射权重名称，并渲染 chat 或 agent prompt。每个 adapter 的 `build_model` 都返回 `DecoderModel`，因此 attention、缓存、批处理和采样只需实现一次。

`DecoderSpec` 是逐层 spec 的列表。每一层选择一种 `SequenceMixerSpec`：`Attention`（full 或 sliding causal，RoPE 或 NoPE）、`ShortConvolution`（LFM2）、`GatedDeltaNet`（Qwen3.5）或 `ParallelAttentionMamba2`（Falcon-H1），再加上一个 dense 或 sparse-MoE 前馈层。因此混合架构模型只是不同的层列表，decoder 实现了运行时其余部分所使用的 `CausalLm` trait。

`ModelAdapter::build` 检查构建出的模型的运行时元数据与 adapter 的一致，`validate` 在接触任何权重之前检查 spec 自身是否一致。

<sub>来源: [mod.rs:28-56](../../../src/models/mod.rs#L28-L56) · [mod.rs:102-118](../../../src/models/mod.rs#L102-L118) · [architecture.rs:224-229](../../../src/architecture.rs#L224-L229) · [architecture.rs:1038-1064](../../../src/architecture.rs#L1038-L1064) · [gemma.rs:71-76](../../../src/models/gemma.rs#L71-L76) · +4</sub>

### 零拷贝加载权重

`ModelFiles::weight_source` 通过 Candle 的 `VarBuilder` 对 Safetensors 做内存映射。GGUF 走项目自己的解析器和基于 mmap 的 `GgufWeightSource`，块量化矩阵在模型构建期间始终保持压缩，并以 `QMatMul` 运行。启用 `metal` feature 时，`file_backed_quant` 把每个 GGUF 分片的只读映射包装成一个共享的 Metal buffer，并按文件偏移绑定每个张量。而 Candle 自带的加载器会把每个权重复制到匿名 buffer 中，使整个模型都变成 dirty 的统一内存。

正因如此，模型加载期间不能修改模型文件；加载器因此是 `unsafe` 的。加载完成后，会请求 macOS 归还构建 tokenizer 和元数据时释放的内存页（`malloc_zone_pressure_relief`）。

<sub>来源: [weights.rs:240-275](../../../src/weights.rs#L240-L275) · [file_backed_quant.rs:1-6](../../../src/file_backed_quant.rs#L1-L6) · [model_weights.rs:1-40](../../../src/model_weights.rs#L1-L40) · [model_weights.rs:305-306](../../../src/model_weights.rs#L305-L306) · [runtime.rs:113-128](../../../src/runtime.rs#L113-L128)</sub>

### 服务调度器

运行时不要求模型实现 `Send`，因此 `EngineHandle::start` 在名为 `macllm-inference` 的线程中构建模型，并且从不移动它。HTTP handler 通过有界通道（`--queue-capacity`）发送 `InferenceJob`，并拿回一个 `EngineTicket`，即请求本地的事件流。客户端断开连接时，正是通过丢弃 ticket 来取消请求。

Worker 按轮执行 continuous batching。每轮最多访问 `--max-active-sequences` 个活跃序列。每个序列最多分到 `--prefill-token-quantum` 个输入 token，一轮在达到 `--scheduler-token-budget` 时结束。所有被访问的 lane 作为一次 packed ragged 前向计算执行。Decode lane 只占一个 token，因此长 prompt 会被分块，不会饿死正在进行的生成。客户端在 `--event-send-timeout-seconds` 内一直不消费事件的 lane 会被取消。

循环内的任何错误或 panic 都会让 worker 进入 poisoned 状态（`/healthz` 报告 `poisoned`），而不是带着可能不一致的模型状态继续运行。

<sub>来源: [runtime.rs:52-55](../../../src/runtime.rs#L52-L55) · [engine.rs:773-822](../../../src/engine.rs#L773-L822) · [engine.rs:1452-1522](../../../src/engine.rs#L1452-L1522) · [engine.rs:1795-1822](../../../src/engine.rs#L1795-L1822) · [engine.rs:1860-1945](../../../src/engine.rs#L1860-L1945) · +2</sub>

### K/V cache 布局

每个 decoder 层都有一个 `KvCachePolicy`：

- `Paged`：full-attention 层共享一个全模型范围的池。池中的 block 长度为 `--kv-cache-block-tokens`，每次分配 `--kv-cache-slab-blocks` 个，总量最多 `--kv-cache-capacity-tokens`（默认是所有序列合计一个模型上下文，而不是每个序列一个）。每个序列持有一个由 block ID 组成的 `PagedKvBlockTable`，一个 block 存放所有 full-attention 层的 K 和 V。
- `Sliding`：局部 attention 层为每个序列维护一个 `2 * (W - 1)` 个槽位的环形缓冲。回滚时只需恢复三个整数，无需复制张量。
- `Full`：按需增长的连续缓存。

批次增长是事务性的。`PagedKvBatchReservation` 在前向计算之前为每个 lane 预留 block，未提交就丢弃时会全部回滚。Block 可以通过引用计数在多个表之间共享；向共享的、未填满的尾部 block 追加时会先复制它（copy-on-write）。

<sub>来源: [cache.rs:116-122](../../../src/cache.rs#L116-L122) · [cache.rs:162-166](../../../src/cache.rs#L162-L166) · [cache.rs:1401-1410](../../../src/cache.rs#L1401-L1410) · [cache.rs:1450-1460](../../../src/cache.rs#L1450-L1460) · [cache.rs:2392-2397](../../../src/cache.rs#L2392-L2397) · +3</sub>

### 自动 prefix cache

`TokenPrefixCache` 是一个按开销限额的 LRU，缓存模型状态快照，以精确的 token ID 为键。哈希 block 只用于缩小候选范围；每次命中都要从位置 0 开始逐 token 比较确认。快照只在每个插入的 prompt 末尾保存，而不是在每个 block 上保存，因此一组逐渐变长的 prompt 不会占用平方级的状态。

Prefill 之后，`publish_prompt_prefix` 保存 prompt。纯 attention 模型会把它截断到 block 边界，存成一个 fork 出来的 block 表，通过引用计数共享页面。处理下一个 prompt 之前，`reset_session_for_prompt` 把最长匹配恢复到会话中，并报告为 `cached_tokens`。

循环层（ShortConv、Gated DeltaNet、Mamba2）需要在精确末尾位置保存状态快照，因为循环状态无法截断。Sliding-window 环形缓冲是可变的，无法 fork，所以 `validate_prefix_cache_topology` 会拒绝任何 sliding 层。因此该缓存与 sliding-window attention 互斥，在这类模型上设置非零容量会导致加载错误，而不是静默地不生效。Benchmark 的波次可以通过 `EngineRequestPolicy` 选择不发布前缀。

<sub>来源: [prefix_cache.rs:1-7](../../../src/prefix_cache.rs#L1-L7) · [prefix_cache.rs:116-122](../../../src/prefix_cache.rs#L116-L122) · [decoder.rs:3461-3499](../../../src/decoder.rs#L3461-L3499) · [decoder.rs:3501-3540](../../../src/decoder.rs#L3501-L3540) · [decoder.rs:4308-4340](../../../src/decoder.rs#L4308-L4340) · +2</sub>

### 自定义 Metal kernel 在运行时编译

大多数张量运算来自 Candle。项目在 `src/metal/` 中另外添加了三个 Metal 源文件：`paged_decode.metal`（通过 block 页表读取 K/V 的 paged attention，包括一个批量 GQA-4 变体）、`gated_delta_decay.metal`（Gated DeltaNet）和 `mamba2_scan.metal`（Mamba2 scan）。每个文件都用 `include_str!` 嵌入，在设备第一次需要时由 `new_library_with_source` 从源码编译。Pipeline 按 Metal 设备的 `registry_id` 缓存。

`build.rs` 不处理这些 kernel。它只嵌入 `MACLLM_BUILD_PROFILE`、`MACLLM_BUILD_OPT_LEVEL`、`MACLLM_BUILD_GIT_COMMIT` 和 `MACLLM_BUILD_GIT_DIRTY`，用于 benchmark 溯源。因此 kernel 编译错误会在运行时首次使用时出现，benchmark 使用 `--warmup-runs` 把这部分编译开销排除在测量之外。

只有预检通过时才会使用 paged kernel：batch 为 1，每个 lane 最多 128 个 query token、最多 128 个 lane，head 维度不超过 256，且为 F32 或 F16。否则它返回 `None`，decoder 回退到先物化完整矩阵的 attention（materialized attention）。

<sub>来源: [paged_attention.rs:790](../../../src/paged_attention.rs#L790) · [paged_attention.rs:1195-1226](../../../src/paged_attention.rs#L1195-L1226) · [paged_attention.rs:14-47](../../../src/paged_attention.rs#L14-L47) · [mamba2.rs:331](../../../src/mamba2.rs#L331) · [mamba2.rs:480-491](../../../src/mamba2.rs#L480-L491) · +9</sub>

<a id="benchmarks-and-testing"></a>

## Benchmark、门禁测试与贡献

本项目的结论建立在两类证据之上。按 checkpoint 固定的参考测试证明 token 级别的正确性，对照对象是 `llama.cpp` 或 Candle。`macllm scoreboard`、`macllm demo` 和 `macllm bench` 下按 manifest 固定的 benchmark 衡量性能。所有测量命令都拒绝在未优化的构建中运行（`ensure_optimized_measurement_execution`），所以请使用 `--release`。

<sub>来源: [mod.rs:1](../../../src/benchmark/mod.rs#L1) · [gemma_gguf_reference.rs:1-8](../../../tests/gemma_gguf_reference.rs#L1-L8) · [main.rs:663-708](../../../src/main.rs#L663-L708)</sub>

### 固定的参考门禁测试

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

### Scoreboard manifest

Scoreboard manifest（`schema_version` 1，见 `benchmarks/manifests/`）固定六类信息：`host`（机器型号、芯片、内存、CPU 架构）；`model`（相对于 manifest 的路径、Hugging Face 仓库、revision、文件名、大小、SHA-256、架构、量化方式）；`llama_cpp`（二进制目录、build 号、commit、GPU 层数、flash attention、K/V cache 类型、mmap）；`workload`（prompt 文件及其 SHA-256、token 数和 token ID 的 SHA-256、decode 步数、贪心采样、并发宽度 1/4/8、预热次数、样本数、prefix cache 模式）；以及 `runtime`（设备、dtype、prefill quantum、K/V 设置）。

`scoreboard verify --manifest PATH` 不加载权重。它检查主机、模型字节、检测出的架构、是否为单个 GGUF 文件、prompt token ID，以及 `llama-server`/`llama-bench` 二进制的 commit，然后输出 JSON 摘要。任何不匹配都会导致失败。各个测量子命令都会先运行 `verify`：`macllm-kernel`、`llama-bench`、`llama-server`（`--port`，默认 18081）、`macllm-serving`，以及 `report`，后者运行所有阶段并写出 `--json` 和 `--markdown`。

<sub>来源: [main.rs:304-350](../../../src/main.rs#L304-L350) · [runner.rs:72-175](../../../src/benchmark/runner.rs#L72-L175) · [smollm3-3b-q4_k_m-m4pro.json](../../../benchmarks/manifests/smollm3-3b-q4_k_m-m4pro.json) · [main.rs:616-661](../../../src/main.rs#L616-L661)</sub>

### Prefix cache 演示与本地 bench

`demo prefix-cache` 和 `demo fair-prefix` 把一个仓库的 `Cargo.toml` 和 `src/**/*.rs`（`--repository`，默认 `.`）转换成精确的 16K token 共享前缀，并以八个并发请求发送给两个运行时。它们默认使用端口 18085 和 18086，分别写出 `prefix-cache-demo.{json,md}` 或 `fair-prefix-benchmark.{json,md}`。结果只是该主机上该次运行的 点估计（point estimate）。

`fair-prefix` 还要求两边使用相同的 HTTP transport 和 F32 K/V 精度。`main.rs` 把 `MAC_PREFIX_DEMO_DEFAULT_MEASURED_WAVES`（20）作为每个实现的测量波次数传入。`validate_inputs` 要求至少 20 个波次，且波次数能平均分配到 4 个相互平衡（counterbalanced）的进程生命周期中，同时 manifest 中的 `fair_prefix` 块必须完全匹配。该块必须声明每个生命周期 5 个预热波次（`FAIR_WARMUP_WAVES_PER_LIFETIME`）、32 token 的输出、已预热的 prefix cache，以及等于波次数乘以 8（160）的 `samples_per_concurrency`。一个 3 波次的稳态窗口（`FAIR_STEADY_STATE_WINDOW_WAVES`）用于检查预热。

`bench generate|embed|rerank` 是不需要 manifest 的本地诊断工具：在 `--warmup-runs`（默认 1）之后进行 `--runs`（默认 20）次测量，使用贪心解码，并拒绝任何输出发生漂移的运行。`scripts/bench-llama-cpp.sh MODEL.gguf` 运行原生的 `llama-bench`，并读取 `PROMPT_TOKENS`（512）、`OUTPUT_TOKENS`（128）和 `RUNS`（3）。

<sub>来源: [main.rs:352-394](../../../src/main.rs#L352-L394) · [main.rs:560-573](../../../src/main.rs#L560-L573) · [prefix_demo_mac.rs:39-46](../../../src/benchmark/prefix_demo_mac.rs#L39-L46) · [fair_prefix.rs:59-61](../../../src/benchmark/fair_prefix.rs#L59-L61) · [fair_prefix.rs:912-956](../../../src/benchmark/fair_prefix.rs#L912-L956) · +5</sub>

### 添加模型家族

新增一个模型家族只需改动它的 adapter 和一个 dispatch 分支。Adapter 的边界见 [工作原理](#how-it-works)，格式 dispatch 见 [概览与支持矩阵](#overview)。

1. 添加实现 `ModelAdapter` 的 `src/models/<family>.rs`：`family`、`runtime_spec`、`decoder_spec`、`build_model`（通常是 `DecoderModel::load`，张量名称不标准时用 `load_with_weight_names`）和 `format_chat_messages`。
2. 对于 GGUF，添加一个带有明确的可接受元数据键列表的 `<family>_spec` 构建器，并在 `GgufModel::open` 和 `GgufModel::adapter` 中添加 match 分支。对于 Safetensors，在 `detect` 中添加一个 `model_type` 分支。
3. 通过精确的身份识别对话模板，而不是执行 Jinja。例如，Qwen3 adapter 只接受字节长度和 SHA-256 与固定常量一致的模板。只有在验证过 native tool 模板之后才实现 `agent_template_profile`。
4. 在 `tests/` 下添加一个被忽略的门禁测试，按 SHA-256 固定一个公开 fixture，并与 `llama.cpp` 比较贪心 token。

如果新的层类型需要自己的 mixer，就扩展 `SequenceMixerSpec` 和 decoder，并说明 prefix cache 能否为它保存快照。

<sub>来源: [mod.rs:57-100](../../../src/models/mod.rs#L57-L100) · [gguf_model.rs:24-43](../../../src/gguf_model.rs#L24-L43) · [gguf_model.rs:312-365](../../../src/gguf_model.rs#L312-L365) · [qwen3.rs:140-168](../../../src/models/qwen3.rs#L140-L168) · [qwen3.rs:246-251](../../../src/models/qwen3.rs#L246-L251) · +2</sub>

<a id="structure"></a>

## 代码结构

本页由 Codocia 根据代码自动生成，每次 build 重新计算：功能区划分、各区自带的说明、区之间的 import 依赖，以及讲解各区的页面。

范围内共 96 个文件，92,879 行；已排除 `tests/fixtures/**`、`Cargo.lock`、`LICENSE-MIT`、`README_CN.md`、`.gitignore`

### 文件结构

**仓库根目录**

- （直接文件） — 304 行，3 个文件。An Apple Silicon-first Rust LLM inference runtime\
  核心文件：`build.rs`\
  讲解：[概览与支持矩阵](#overview) (1), [工作原理](#how-it-works) (1)
- `benchmarks/` — 144 行，3 个文件\
  讲解：[Benchmark、门禁测试与贡献](#benchmarks-and-testing) (2)
- `scripts/` — 25 行，1 个文件\
  核心文件：`bench-llama-cpp.sh`\
  讲解：[Benchmark、门禁测试与贡献](#benchmarks-and-testing) (1)
- `tests/` — 7,137 行，23 个文件\
  讲解：[Benchmark、门禁测试与贡献](#benchmarks-and-testing) (2)

**`src/`**

- （直接文件） — 56,303 行，30 个文件。`MacLLM Infer` is an Apple Silicon-first Rust LLM inference runtime.\
  核心文件：`lib.rs` `architecture.rs` `gguf.rs`\
  讲解：[概览与支持矩阵](#overview) (3), [命令行用法](#cli) (5), [HTTP 服务与 API 子集](#http-api) (5), [工作原理](#how-it-works) (7), [Benchmark、门禁测试与贡献](#benchmarks-and-testing) (4)
- `benchmark/` — 23,014 行，19 个文件。Reproducible, fail-closed `MacLLM` versus `llama.cpp` scoreboard primitives.\
  核心文件：`schema.rs` `mod.rs` `prefix_demo_mac.rs`\
  讲解：[概览与支持矩阵](#overview) (1), [Benchmark、门禁测试与贡献](#benchmarks-and-testing) (3)
- `metal/` — 652 行，3 个文件\
  讲解：[工作原理](#how-it-works) (1)
- `models/` — 5,300 行，14 个文件\
  核心文件：`qwen3.rs` `llama.rs` `qwen3_tasks.rs`\
  讲解：[概览与支持矩阵](#overview) (1), [命令行用法](#cli) (1), [HTTP 服务与 API 子集](#http-api) (1), [工作原理](#how-it-works) (1), [Benchmark、门禁测试与贡献](#benchmarks-and-testing) (1)


### 依赖结构

箭头 A → B 表示 A 中有文件 import 了 B，数字是这样的文件级 import 数。只画最强的 4 条（共 4 条）。

```mermaid
flowchart LR
  a3["src/*"]
  a4["src/benchmark/"]
  a6["src/models/"]
  a7["tests/"]
  a7 -->|22| a3
  a6 -->|8| a3
  a4 -->|2| a3
  a3 -->|1| a4
```
