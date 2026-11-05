---
title: Generating text with run
order: 20
evidence:
  - src/main.rs:108-160 #977278991218
  - src/main.rs:1220-1255 #96d57fbf0abd
  - src/main.rs:1285-1300 #7486b655a047
---

`run` renders `--prompt` (plus optional `--system`) through the adapter's chat template, tokenizes it, and streams decoded text to stdout. Adapters render only templates they recognise and otherwise fail; `--raw-prompt` bypasses the template and tokenizes the prompt verbatim. `--add-special-tokens` asks the tokenizer to add BOS/EOS; leave it off for templates that already contain BOS.

Sampling defaults: `--max-new-tokens 128`, `--temperature 0.7` (0 means greedy argmax), `--top-k 20`, `--top-p 0.8`, `--repetition-penalty 1.1`, `--seed 42`.

`--prefill-token-quantum` caps prompt tokens per prefill forward. Unset, Metal uses 128 unless `--dtype bf16`; CPU and BF16 prefill in one shot. After generation, a metrics line on stderr reports prompt, cached and processed tokens, TTFT, prefill and decode tokens per second.
