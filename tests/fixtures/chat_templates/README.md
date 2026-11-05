# Chat-template reference renders

Byte-exact expected prompts for the native chat and tool templates that MacLLM recognises. They were rendered by `gen_fixtures.py` the way Hugging Face transformers' `apply_chat_template` does:
- environment: `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)` with the `loopcontrols` extension;
- `tojson`: `json.dumps(ensure_ascii=False, sort_keys=False)` with default separators;
- render arguments: `add_generation_prompt=True`, `bos_token=""`, `eos_token="<|im_end|>"`.

| Template file | Source | Bytes | SHA-256 (prefix) |
| --- | --- | --- | --- |
| `templates/qwen35-official.jinja` | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, `chat_template.jinja` | 7,756 | `a4aee8afcf2e0711` |
| `templates/qwen35-unsloth.jinja` | `tokenizer.chat_template` embedded in `unsloth/Qwen3.5-4B-GGUF` @ `e87f1764`, `Qwen3.5-4B-Q4_K_M.gguf` (identical in the 2B and 9B GGUFs) | 7,816 | `7f0e529032c25183` |
| `templates/qwen3-current.jinja` | `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`, `tokenizer_config.json` `chat_template` (MacLLM `Qwen3ChatTemplate::Current`) | 4,168 | `a55ee1b1660128b7` |
| `templates/qwen3-gguf.jinja` | `tokenizer.chat_template` in `Qwen/Qwen3-4B-GGUF` @ `bc640142`, `Qwen3-4B-Q4_K_M.gguf` (MacLLM `Qwen3ChatTemplate::Gguf`) | 4,100 | `57f1fd00f0013a2b` |

`rendered/<family>/<template>__<case>__thinking-<default|off|on>.txt` is one render per case, in three variants: no `enable_thinking`, `false`, `true`. The conversations are in `gen_fixtures.py` (`CASES`). For every template, each case also has a `__string_args` variant, where tool-call `arguments` arrive as a JSON string (the OpenAI wire format) instead of an object.

The generator also writes `cases.json` here with every input, thinking variant,
and the template/case membership of each family. The `macllm-chat` integration
tests read this manifest to compare all 144 reference renders with the generic
Jinja renderer and the pinned Rust chat/tool renderers. Wire-format argument
strings are parsed into ordered objects before both generic rendering and
reference rendering, matching the `ChatMessage` contract. Qwen3.5 in particular
expects argument mappings. Hugging Face's [tool-use documentation](https://huggingface.co/docs/transformers/main/en/chat_extras#tool-calling-example)
specifies dictionaries for tool-call arguments and warns that OpenAI's JSON
string form can produce errors.

The generic renderer enables Python-compatible string/list/dict methods and
loop controls, disables HTML autoescaping, and overrides `tojson` with Python
JSON spacing, Unicode handling, and insertion order. `messages`,
`add_generation_prompt`, `bos_token`, and `eos_token` are always available;
`tools` and `enable_thinking` are omitted unless explicitly supplied. It also
provides `raise_exception(message)` and `strftime_now(format)`, using local
wall-clock time with naive datetime zone fields, like Python's
`datetime.now().strftime(format)`.

Differences between the two Qwen3.5 templates:
- The official template thinks by default; unsloth's does not, unless `enable_thinking` is true.
- Unsloth's skips the parameters of a call whose `arguments` is not a mapping. The official template iterates `arguments|items`.

So a server must parse JSON-string arguments into an ordered object before it renders Qwen3.5 history.

The templates are Apache-2.0, from Qwen. Run from this directory: `uv run --no-project --with jinja2==3.1.6 python3 gen_fixtures.py templates rendered/qwen35 qwen35`, then the same with `rendered/qwen3 qwen3`.

## Pinned template sources

`templates/pinned/` holds the exact source of every pinned chat-template
variant, one file per variant. Each file is a byte-for-byte copy of the
template the family pins, and every family asserts that its file still
classifies to its own variant: by string equality for the string-pinned
variants, by length and SHA-256 for the digest-pinned ones.

| Template file | Source | Bytes | SHA-256 (prefix) |
| --- | --- | --- | --- |
| `templates/pinned/gemma-official.jinja` | the official Gemma 1 GGUF template revision; repository and revision not recorded | 591 | `ecd6ae513fe103f0` |
| `templates/pinned/gemma-legacy.jinja` | the older official Gemma 1 GGUF template revision; repository and revision not recorded | 576 | `3919b9309d8949ee` |
| `templates/pinned/gemma2-official.jinja` | the official Gemma 2 instruction tokenizer template; repository and revision not recorded | 591 | `ecd6ae513fe103f0` |
| `templates/pinned/gemma3-official.jinja` | origin not recorded | 1,532 | `7de1c58e208eda46` |
| `templates/pinned/granite_moe-legacy-granite3-1-instruct.jinja` | the confirmed `tokenizer.chat_template` variant of the Granite 3.1 1B-A400M Instruct GGUFs; repository and revision not recorded | 3,224 | `22da301945ac6b61` |
| `templates/pinned/granite_moe-current-granite3-1-instruct.jinja` | the confirmed `tokenizer.chat_template` variant of the Granite 3.1 1B-A400M Instruct GGUFs; repository and revision not recorded | 3,543 | `08962c2f15d56767` |
| `templates/pinned/lfm2-official-gguf.jinja` | `LiquidAI/LFM2-350M-GGUF` @ `8fdc9d526b7ed346b19257551b05816c7912ecc2` | 209 | `a805e50fed68938a` |
| `templates/pinned/llama-llama3-2-instruct.jinja` | `bartowski/Llama-3.2-1B-Instruct-GGUF` @ `067b946cf014` | 3,827 | `5816fce10444e03c` |
| `templates/pinned/llama-smollm2-chatml.jinja` | origin not recorded | 368 | `872be49dbb638044` |
| `templates/pinned/qwen2-official-qwen2-5.jinja` | origin not recorded | 2,509 | `d5495a1e5db06111` |
| `templates/pinned/qwen35-official-text-subset.jinja` | a local `Qwen3.5-0.8B-Q8_0.gguf` | 7,755 | `273d8e0e683b8850` |
| `templates/pinned/smollm3-official-system-override-text-subset.jinja` | `ggml-org/SmolLM3-3B-GGUF` @ `4965cb60b150` | 5,493 | `b9b66f04c64fbb86` |

The pinned sources live in a directory of their own so that `templates/` keeps
exactly the four reference templates the table above lists, which the
`macllm-chat` parity test counts. Their names follow the
`<family>-<variant>.jinja` convention of the reference templates.

`macllm-infer`'s `src/models/template_parity.rs` renders every pinned variant
twice: once through the pinned renderer of its family and once through the
generic Jinja engine that compiles the very same source. Both arms run the
server's own routing, so a refusal in the table is the refusal a client
would see. The measured outcomes live in `parity_outcomes.txt`, one line per
variant, case and thinking mode, and the test fails on any difference from
that file. For the cases in `parity_cases.json` and the routes they take, the
table compares prompt bytes for the BOS, EOS and token lists that
`parity_variants.json` lists; it does not cover constraints, sampling or
parsing. No case sets `response_format`, so the `constrained` route is not
measured.

## Forced tool-call suffixes

`forced_calls/` holds the exact bytes the pinned Qwen 3 agent renderers append
to the automatic turn when the prompt forces a single tool call, one file per
agent profile: `qwen3-current.txt` and `qwen3-gguf.txt`. The Qwen 3.5
profiles have no suffix: their routing refuses a forced tool choice, and the
measurement records that refusal.
