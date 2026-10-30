---
title: 
order: 0
evidence:
  - src/lib.rs:1-7 #6b4901cd87d9
  - src/runtime.rs:62-103 #c7c5452e59f2
  - src/engine.rs:773-805 #49f2765b5ff1
---

Every command goes through `LoadedRuntime::load`: open the files, choose an adapter, build the shared decoder from a `WeightSource`, then configure the paged K/V pool and the prefix cache. Blocking commands call the model directly. `serve` builds the same runtime inside one worker thread that owns the model for its whole life.

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
