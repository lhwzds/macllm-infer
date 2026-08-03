use std::process::Command;

#[test]
fn help_uses_the_public_cli_name() {
    let output = Command::new(env!("CARGO_BIN_EXE_macllm"))
        .arg("--help")
        .output()
        .expect("the macllm binary should start");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("help output should be UTF-8");
    assert!(stdout.starts_with("An Apple Silicon-first Rust LLM inference runtime"));
    assert!(stdout.contains("Usage: macllm <COMMAND>"));
    assert!(!stdout.to_ascii_lowercase().contains("lucid"));
}

#[test]
fn download_exposes_revision_selection() {
    let output = Command::new(env!("CARGO_BIN_EXE_macllm"))
        .args(["download", "--help"])
        .output()
        .expect("the macllm binary should start");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("help output should be UTF-8");
    assert!(stdout.contains("--revision <REVISION>"));
    assert!(stdout.contains("branch, tag or commit hash"));
    assert!(stdout.contains("--gguf <FILENAME>"));
    assert!(stdout.contains("every shard when it is a canonical first shard"));
}

#[test]
fn serve_exposes_the_bounded_openai_surface() {
    let output = Command::new(env!("CARGO_BIN_EXE_macllm"))
        .args(["serve", "--help"])
        .output()
        .expect("the macllm binary should start");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("help output should be UTF-8");
    assert!(stdout.contains("text-only Chat Completions-compatible subset"));
    assert!(stdout.contains("--served-model-name <SERVED_MODEL_NAME>"));
    assert!(stdout.contains("--queue-capacity <QUEUE_CAPACITY>"));
    assert!(stdout.contains("--max-active-sequences <MAX_ACTIVE_SEQUENCES>"));
    assert!(stdout.contains("--scheduler-token-budget <SCHEDULER_TOKEN_BUDGET>"));
    assert!(stdout.contains("--prefill-token-quantum <PREFILL_TOKEN_QUANTUM>"));
    assert!(stdout.contains("--event-capacity <EVENT_CAPACITY>"));
    assert!(stdout.contains("--event-send-timeout-seconds <EVENT_SEND_TIMEOUT_SECONDS>"));
    assert!(stdout.contains("[default: 127.0.0.1]"));
}
