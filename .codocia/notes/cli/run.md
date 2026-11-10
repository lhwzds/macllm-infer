---
title: Generating text with run
order: 20
evidence:
  - src/main.rs:152-190 #0fc1d684b718
  - src/main.rs:1320-1334 #433719f34f7f
  - src/main.rs:1345-1356 #e075c18eb0ce
  - src/models/mod.rs:515-536 #928b3b5fc65c
  - src/gguf_model.rs:365-387 #539115b2ad64
  - src/main.rs:1394-1408 #f90154d8c457
---

`run` renders `--prompt` (plus optional `--system`) through the adapter's chat template, tokenizes it, and streams decoded text to stdout. Adapters render the templates they pin byte-for-byte; an unpinned template embedded in a GGUF file goes through the generic Jinja renderer, and a checkpoint with neither fails. `--raw-prompt` bypasses the template and tokenizes the prompt verbatim. `--add-special-tokens` asks the tokenizer to add BOS/EOS; leave it off for templates that already contain BOS.

Sampling defaults: `--max-new-tokens 128`, `--temperature 0.7` (0 means greedy argmax), `--top-k 20`, `--top-p 0.8`, `--repetition-penalty 1.1`, `--seed 42`.

`--prefill-token-quantum` caps prompt tokens per prefill forward. Unset, Metal uses 128 unless `--dtype bf16`; CPU, CUDA and BF16 prefill in one shot. After generation, a metrics line on stderr reports prompt, cached and processed tokens, TTFT, prefill and decode tokens per second.
