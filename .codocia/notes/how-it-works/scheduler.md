---
title: The serving scheduler
order: 30
evidence:
  - src/runtime.rs:52-55 #1b5ba85c18b8
  - src/engine.rs:773-822 #28a502ee8cb4
  - src/engine.rs:1452-1522 #9541b57917a3
  - src/engine.rs:1795-1822 #9ed9b9d2ab1c
  - src/engine.rs:1860-1945 #62b34566aef5
  - src/engine.rs:993-1000 #e17a27132dff
  - src/engine.rs:258-277 #95dcbcb1e558
  - src/engine.rs:1995-2012 #07bdbdfa1b85
  - src/engine.rs:4798-4830 #912bae62c3c2
  - src/engine.rs:4831-4905 #7cc7b65b4bcc
  - src/engine.rs:5103-5160 #fff68b7eb109
---

The runtime does not require models to be `Send`, so `EngineHandle::start` builds the model inside a thread named `macllm-inference` and never moves it. HTTP handlers send `InferenceJob`s over a bounded channel (`--queue-capacity`) and get back an `EngineTicket`, a request-local event stream. Dropping the ticket is how a client disconnect cancels a request.

The worker runs continuous batching in rounds. In each round it visits up to `--max-active-sequences` active sequences. Each sequence gets at most `--prefill-token-quantum` input tokens, and the round stops at `--scheduler-token-budget`. All visited lanes run as one packed ragged forward. Decode lanes cost one token, so long prompts are chunked and cannot starve running generations. A lane whose client stops draining events for `--event-send-timeout-seconds` is cancelled.

A panic, or an error escaping admission or the shared `drive_once` step (a failed forward), poisons the worker (`/healthz` reports `poisoned`) and fails active and queued requests. Request-local failures (invalid sampling, one lane's KV-cache exhaustion, incomplete JSON-schema output) fail only that request (`request_local_preflight_failure_does_not_poison_the_worker`).
