---
title: The serving scheduler
order: 30
evidence:
  - src/runtime.rs:56-59 #1b5ba85c18b8
  - src/engine.rs:829-869 #e2339798d62e
  - src/engine.rs:1578-1648 #9541b57917a3
  - src/engine.rs:1938-1965 #9ed9b9d2ab1c
  - src/engine.rs:2003-2088 #62b34566aef5
  - src/engine.rs:1056-1064 #ff4c25208f1c
  - src/engine.rs:308-327 #9d9ab97847a0
  - src/engine.rs:2138-2155 #07bdbdfa1b85
  - src/engine.rs:5347-5379 #912bae62c3c2
  - src/engine.rs:5380-5400 #fd53c1f32ab2
  - src/engine.rs:5799-5856 #fff68b7eb109
---

Models need not be `Send`: `EngineHandle::start` builds the model inside the `macllm-inference` thread and never moves it. HTTP handlers send `InferenceJob`s over a bounded channel (`--queue-capacity`) and get back an `EngineTicket`, a request-local event stream. Dropping the ticket cancels the request.

The worker runs continuous batching in rounds. In each round it visits up to `--max-active-sequences` active sequences. Each sequence gets at most `--prefill-token-quantum` input tokens, and the round stops at `--scheduler-token-budget`. All visited lanes run as one packed ragged forward. Decode lanes cost one token, so long prompts cannot starve running generations. A lane whose client stops draining events for `--event-send-timeout-seconds` is cancelled.

A panic, or an error escaping admission or the shared `drive_once` step (a failed forward), poisons the worker (`/healthz` reports `poisoned`) and fails active and queued requests. Request-local failures (invalid sampling, one lane's KV-cache exhaustion, incomplete JSON-schema output) fail only that request.
