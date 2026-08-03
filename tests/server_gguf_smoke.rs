//! Ignored real-model gate for the text-only Chat Completions serving path.
//!
//! Run outside a restrictive macOS sandbox with:
//!
//! ```text
//! cargo test --test server_gguf_smoke -- --ignored --nocapture
//! ```
//!
//! Set `MACLLM_SMOKE_MODEL` to select another supported GGUF fixture.

use std::{path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Context, Result, ensure};
use axum::{
    body::Body,
    http::{Request, header},
};
use http_body_util::BodyExt;
use macllm_infer::{
    backend::DeviceKind,
    engine::{EngineConfig, EngineHandle},
    runtime::ModelLoadConfig,
    server::router,
};
use serde_json::{Value, json};
use tower::ServiceExt;

const SERVED_MODEL: &str = "smollm2-smoke";

#[tokio::test]
#[ignore = "loads a real GGUF model and requires unsandboxed Metal access"]
async fn real_metal_json_sse_disconnect_and_reuse() -> Result<()> {
    let model = std::env::var_os("MACLLM_SMOKE_MODEL").map_or_else(default_model, PathBuf::from);
    ensure!(
        model.is_file(),
        "real server fixture is missing: {}; set MACLLM_SMOKE_MODEL",
        model.display()
    );

    let mut engine_config = EngineConfig::new(
        ModelLoadConfig::new(model, DeviceKind::Metal, None),
        SERVED_MODEL,
    );
    // A one-event channel makes the undrained SSE request exercise slot-local
    // backpressure before its short peer completes.
    engine_config.event_capacity = 1;
    engine_config.event_send_timeout = Duration::from_mins(2);
    let engine = EngineHandle::start(engine_config)?;
    let app = router(Arc::new(engine));

    let first = json_completion(app.clone(), "Reply with exactly: OK").await?;
    let second = json_completion(app.clone(), "Reply with exactly: OK").await?;
    assert_eq!(
        first["choices"][0]["message"]["content"],
        second["choices"][0]["message"]["content"]
    );
    assert_eq!(first["usage"], second["usage"]);

    let (concurrent_first, concurrent_second) = tokio::join!(
        json_completion(app.clone(), "Reply with exactly: OK"),
        json_completion(app.clone(), "Reply with exactly: OK"),
    );
    assert_eq!(
        concurrent_first?["choices"][0]["message"]["content"],
        concurrent_second?["choices"][0]["message"]["content"]
    );

    let response = app
        .clone()
        .oneshot(chat_request(
            "Count from one to one hundred in words.",
            128,
            true,
            false,
        ))
        .await?;
    let mut body = response.into_body();
    let mut received = String::new();
    while !received.contains("\"content\"") {
        let frame = body
            .frame()
            .await
            .context("SSE ended before the first content chunk")??;
        if let Some(data) = frame.data_ref() {
            received.push_str(std::str::from_utf8(data)?);
        }
    }
    let peer = tokio::time::timeout(
        Duration::from_secs(5),
        json_completion(app.clone(), "Reply with exactly: OK"),
    )
    .await
    .context("a backpressured streaming peer blocked another active sequence")??;
    assert_eq!(
        first["choices"][0]["message"]["content"],
        peer["choices"][0]["message"]["content"]
    );
    let next_slow_frame = tokio::time::timeout(Duration::from_secs(5), body.frame())
        .await
        .context("backpressured sequence did not resume after its peer finished")?
        .context("backpressured SSE ended before another content chunk")??;
    let next_slow_data = next_slow_frame
        .data_ref()
        .context("backpressured SSE frame did not contain data")?;
    let next_slow_text = std::str::from_utf8(next_slow_data)?;
    ensure!(next_slow_text.contains("\"content\""));
    ensure!(!next_slow_text.contains("data: [DONE]"));
    drop(body);

    let after_disconnect = json_completion(app.clone(), "Reply with exactly: OK").await?;
    assert_eq!(
        first["choices"][0]["message"]["content"],
        after_disconnect["choices"][0]["message"]["content"]
    );

    let response = app
        .oneshot(chat_request("Say hello briefly.", 8, true, true))
        .await?;
    let sse = String::from_utf8(response.into_body().collect().await?.to_bytes().to_vec())?;
    ensure!(sse.contains("\"role\":\"assistant\""));
    ensure!(sse.contains("\"finish_reason\":"));
    ensure!(sse.contains("\"choices\":[],\"usage\":"));
    ensure!(sse.contains("data: [DONE]"));
    Ok(())
}

async fn json_completion(app: axum::Router, prompt: &str) -> Result<Value> {
    let response = app.oneshot(chat_request(prompt, 8, false, false)).await?;
    ensure!(response.status().is_success());
    Ok(serde_json::from_slice(
        &response.into_body().collect().await?.to_bytes(),
    )?)
}

fn chat_request(
    prompt: &str,
    max_completion_tokens: usize,
    stream: bool,
    include_usage: bool,
) -> Request<Body> {
    let mut body = json!({
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": max_completion_tokens,
        "stream": stream
    });
    if include_usage {
        body["stream_options"] = json!({"include_usage": true});
    }
    Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("static smoke-test request is valid")
}

fn default_model() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/smollm2-135m-instruct-q4_0/SmolLM2-135M-Instruct-q4_0.gguf")
}
