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
