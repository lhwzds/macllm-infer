#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/model.gguf" >&2
  exit 2
fi

model_path="$1"
prompt_tokens="${PROMPT_TOKENS:-512}"
output_tokens="${OUTPUT_TOKENS:-128}"
runs="${RUNS:-3}"

if ! command -v llama-bench >/dev/null 2>&1; then
  echo "llama-bench is required; install or build llama.cpp first" >&2
  exit 1
fi

llama-bench \
  -m "$model_path" \
  -p "$prompt_tokens" \
  -n "$output_tokens" \
  -ngl 99 \
  -r "$runs" \
  -o csv
