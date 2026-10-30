"""Render Qwen3.5 chat-template fixtures the way transformers' apply_chat_template does."""
import json, pathlib, sys
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.exceptions import TemplateError

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

def raise_exception(message):
    raise TemplateError(message)

def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception

SHELL = {"type": "function", "function": {"name": "shell", "description": "Run a shell command and return its exit code and output.",
         "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
WRITE = {"type": "function", "function": {"name": "write_file", "description": "Write a file.",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string", "description": "Relative path."},
             "content": {"type": "string"},
             "mode": {"type": "integer"},
             "append": {"type": "boolean"},
             "tags": {"type": "array", "items": {"type": "string"}},
             "meta": {"type": "object", "properties": {"owner": {"type": "string"}}}},
             "required": ["path", "content"]}}}
SYS = {"role": "system", "content": "You are a careful coding agent."}
USER = {"role": "user", "content": "List the files in the current directory."}
CALL_LS = {"id": "call_1", "type": "function", "function": {"name": "shell", "arguments": {"command": "ls -la"}}}
CASES = {
    "plain": dict(messages=[SYS, {"role": "user", "content": "What is 2 + 2?"}], tools=None),
    "tools_first_turn": dict(messages=[SYS, USER], tools=[SHELL, WRITE]),
    "tools_no_system": dict(messages=[USER], tools=[SHELL]),
    "tool_loop": dict(messages=[SYS, USER,
        {"role": "assistant", "content": "", "tool_calls": [CALL_LS]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt"}], tools=[SHELL]),
    "reasoning_and_text_before_call": dict(messages=[SYS, USER,
        {"role": "assistant", "content": "<think>\nI should list the files.\n</think>\n\nLet me check.", "tool_calls": [CALL_LS]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt"}], tools=[SHELL]),
    "typed_arguments_then_new_user": dict(messages=[SYS, {"role": "user", "content": "Save a note."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "write_file",
            "arguments": {"path": "notes/todo.md", "content": "line one\nline two", "mode": 420, "append": False,
                          "tags": ["a", "b"], "meta": {"owner": "me"}}}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
        {"role": "assistant", "content": "Saved."},
        {"role": "user", "content": "Thanks. Now list files."}], tools=[SHELL, WRITE]),
}
THINKING = {"default": {}, "off": {"enable_thinking": False}, "on": {"enable_thinking": True}}

out.mkdir(parents=True, exist_ok=True)
FAMILIES = {"qwen35": ("qwen35-official", "qwen35-unsloth"), "qwen3": ("qwen3-current", "qwen3-gguf")}
family = sys.argv[3] if len(sys.argv) > 3 else "qwen35"
if family == "qwen3":
    import copy
    for name in list(CASES):
        case = copy.deepcopy(CASES[name])
        for m in case["messages"]:
            for c in m.get("tool_calls", []):
                c["function"]["arguments"] = json.dumps(c["function"]["arguments"], ensure_ascii=False)
        CASES[name + "__string_args"] = case
for tname in FAMILIES[family]:
    tmpl = env.from_string((src / f"{tname}.jinja").read_text())
    for cname, case in CASES.items():
        for th, kw in THINKING.items():
            kwargs = dict(messages=case["messages"], add_generation_prompt=True, bos_token="", eos_token="<|im_end|>", **kw)
            if case["tools"] is not None:
                kwargs["tools"] = case["tools"]
            (out / f"{tname}__{cname}__thinking-{th}.txt").write_text(tmpl.render(**kwargs))
(out / "cases.json").write_text(json.dumps({"cases": CASES, "thinking": THINKING,
    "render": "transformers semantics: ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks, loopcontrols); tojson(ensure_ascii=False, sort_keys=False); add_generation_prompt=True"},
    ensure_ascii=False, indent=2))
print(len(list(out.glob("*.txt"))), "renders")
