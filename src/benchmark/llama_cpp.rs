//! Parsers and identity checks for the pinned `llama.cpp` command-line tools.

use std::{path::Path, process::Command};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::schema::{LlamaCppRequirement, LlamaFlashAttention, is_lower_hex};

/// Build identity printed by `llama-server --version` and related tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaCppVersion {
    pub build: u32,
    pub commit_abbreviation: String,
    pub compiler_and_target: Option<String>,
}

impl LlamaCppVersion {
    /// Check the printed build and an independently attested full source hash.
    pub fn verify(
        &self,
        requirement: &LlamaCppRequirement,
        attested_full_commit: &str,
    ) -> Result<()> {
        ensure!(
            self.build == requirement.build,
            "llama.cpp build mismatch: expected {}, found {}",
            requirement.build,
            self.build
        );
        ensure!(
            is_lower_hex(attested_full_commit, 40),
            "attested llama.cpp commit must be a full lowercase hash"
        );
        ensure!(
            attested_full_commit == requirement.commit,
            "llama.cpp full commit does not match the manifest"
        );
        ensure!(
            requirement.commit.starts_with(&self.commit_abbreviation),
            "llama.cpp binary abbreviation does not match the attested commit"
        );
        Ok(())
    }
}

/// Normalized record from one `llama-bench -o json` result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlamaBenchRecord {
    pub build_commit: String,
    pub build_number: u32,
    pub model_filename: String,
    pub model_type: String,
    pub model_size_bytes: u64,
    pub model_parameters: u64,
    pub backend: String,
    pub n_batch: usize,
    pub n_ubatch: usize,
    pub n_threads: usize,
    pub n_gpu_layers: i32,
    pub flash_attention: LlamaFlashAttention,
    pub mmap: bool,
    pub prompt_tokens: usize,
    pub generation_tokens: usize,
    pub average_nanoseconds: u64,
    pub standard_deviation_nanoseconds: u64,
    pub average_tokens_per_second: f64,
    pub standard_deviation_tokens_per_second: f64,
}

impl LlamaBenchRecord {
    /// Classify the workload without inferring it from the human-readable label.
    #[must_use]
    pub const fn kind(&self) -> LlamaBenchKind {
        match (self.prompt_tokens, self.generation_tokens) {
            (0, 0) => LlamaBenchKind::Empty,
            (0, _) => LlamaBenchKind::Decode,
            (_, 0) => LlamaBenchKind::Prefill,
            (_, _) => LlamaBenchKind::Mixed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlamaBenchKind {
    Empty,
    Prefill,
    Decode,
    Mixed,
}

/// Parse version output even when backend discovery emitted preceding lines.
pub fn parse_llama_cpp_version(output: &str) -> Result<LlamaCppVersion> {
    let version_line = output
        .lines()
        .find_map(|line| line.trim().strip_prefix("version:"))
        .context("llama.cpp version output has no version line")?
        .trim();
    let (build, commit) = version_line
        .split_once('(')
        .context("llama.cpp version line has no commit abbreviation")?;
    let build = build
        .trim()
        .parse::<u32>()
        .context("llama.cpp build is not an integer")?;
    let commit = commit
        .trim()
        .strip_suffix(')')
        .context("llama.cpp version commit is missing its closing parenthesis")?;
    ensure!(build > 0, "llama.cpp build must be positive");
    ensure!(
        (7..=40).contains(&commit.len())
            && commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "llama.cpp version commit abbreviation is invalid"
    );
    let compiler_and_target = output
        .lines()
        .find_map(|line| line.trim().strip_prefix("built with "))
        .map(str::to_owned);
    Ok(LlamaCppVersion {
        build,
        commit_abbreviation: commit.to_owned(),
        compiler_and_target,
    })
}

/// Execute one version-capable `llama.cpp` binary and parse its identity.
pub fn collect_llama_cpp_version(binary: impl AsRef<Path>) -> Result<LlamaCppVersion> {
    let binary = binary.as_ref();
    let output = Command::new(binary)
        .arg("--version")
        .output()
        .with_context(|| format!("failed to execute {} --version", binary.display()))?;
    ensure!(
        output.status.success(),
        "llama.cpp version command failed: {}",
        binary.display()
    );
    let mut text =
        String::from_utf8(output.stdout).context("llama.cpp version stdout was not UTF-8")?;
    if !output.stderr.is_empty() {
        text.push('\n');
        text.push_str(
            std::str::from_utf8(&output.stderr)
                .context("llama.cpp version stderr was not UTF-8")?,
        );
    }
    parse_llama_cpp_version(&text)
}

/// Extract the full source hash recorded by a Homebrew installation receipt.
pub fn parse_homebrew_scm_revision(bytes: &[u8]) -> Result<String> {
    let value: Value = serde_json::from_slice(bytes).context("invalid Homebrew receipt JSON")?;
    let revision = value
        .get("source")
        .and_then(|source| source.get("scm_revision"))
        .and_then(Value::as_str)
        .context("Homebrew receipt has no source.scm_revision")?;
    ensure!(
        is_lower_hex(revision, 40),
        "Homebrew receipt source revision is not a full lowercase commit hash"
    );
    Ok(revision.to_owned())
}

/// Parse the JSON array emitted by `llama-bench -o json`.
pub fn parse_llama_bench_json(output: &str) -> Result<Vec<LlamaBenchRecord>> {
    let values: Vec<Value> =
        serde_json::from_str(output.trim()).context("llama-bench stdout is not a JSON array")?;
    ensure!(!values.is_empty(), "llama-bench returned no records");
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let object = value
                .as_object()
                .with_context(|| format!("llama-bench record {index} is not an object"))?;
            parse_bench_record(object)
                .with_context(|| format!("invalid llama-bench record {index}"))
        })
        .collect()
}

/// Parse the JSON Lines emitted by `llama-bench -o jsonl`.
pub fn parse_llama_bench_jsonl(output: &str) -> Result<Vec<LlamaBenchRecord>> {
    let mut records = Vec::new();
    for (line_number, line) in output.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("llama-bench JSONL line {} is invalid", line_number + 1))?;
        let object = value.as_object().with_context(|| {
            format!(
                "llama-bench JSONL line {} is not an object",
                line_number + 1
            )
        })?;
        records.push(
            parse_bench_record(object)
                .with_context(|| format!("invalid llama-bench JSONL line {}", line_number + 1))?,
        );
    }
    ensure!(!records.is_empty(), "llama-bench returned no JSONL records");
    Ok(records)
}

fn parse_bench_record(record: &Map<String, Value>) -> Result<LlamaBenchRecord> {
    let parsed = LlamaBenchRecord {
        build_commit: required_string(record, &["build_commit"])?,
        build_number: required_u32(record, &["build_number"])?,
        model_filename: required_string(record, &["model_filename"])?,
        model_type: required_string(record, &["model_type"])?,
        model_size_bytes: required_u64(record, &["model_size"])?,
        model_parameters: required_u64(record, &["model_n_params", "model_parameters"])?,
        backend: required_string(record, &["backend", "backends"])?,
        n_batch: required_usize(record, &["n_batch"])?,
        n_ubatch: required_usize(record, &["n_ubatch"])?,
        n_threads: required_usize(record, &["n_threads"])?,
        n_gpu_layers: required_i32(record, &["n_gpu_layers"])?,
        flash_attention: required_flash_attention(record, &["flash_attn", "flash_attention"])?,
        mmap: required_bool(record, &["use_mmap", "mmap"])?,
        prompt_tokens: required_usize(record, &["n_prompt", "prompt_tokens"])?,
        generation_tokens: required_usize(record, &["n_gen", "generation_tokens"])?,
        average_nanoseconds: required_u64(record, &["avg_ns", "average_nanoseconds"])?,
        standard_deviation_nanoseconds: required_u64(
            record,
            &["stddev_ns", "standard_deviation_nanoseconds"],
        )?,
        average_tokens_per_second: required_f64(record, &["avg_ts", "average_tokens_per_second"])?,
        standard_deviation_tokens_per_second: required_f64(
            record,
            &["stddev_ts", "standard_deviation_tokens_per_second"],
        )?,
    };
    ensure!(
        (7..=40).contains(&parsed.build_commit.len())
            && parsed
                .build_commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "llama-bench build_commit is invalid"
    );
    ensure!(parsed.build_number > 0, "llama-bench build number is zero");
    ensure!(
        !matches!(parsed.kind(), LlamaBenchKind::Empty),
        "llama-bench record has no prompt or generation tokens"
    );
    ensure!(
        parsed.average_nanoseconds > 0,
        "llama-bench average duration is zero"
    );
    ensure!(
        parsed.average_tokens_per_second.is_finite()
            && parsed.average_tokens_per_second > 0.0
            && parsed.standard_deviation_tokens_per_second.is_finite()
            && parsed.standard_deviation_tokens_per_second >= 0.0,
        "llama-bench token rates must be finite and non-negative"
    );
    Ok(parsed)
}

fn required_value<'a>(record: &'a Map<String, Value>, keys: &[&str]) -> Result<&'a Value> {
    keys.iter()
        .find_map(|key| record.get(*key))
        .with_context(|| format!("missing required field `{}`", keys.join("` or `")))
}

fn required_string(record: &Map<String, Value>, keys: &[&str]) -> Result<String> {
    let value = required_value(record, keys)?
        .as_str()
        .with_context(|| format!("field `{}` is not a string", keys[0]))?;
    ensure!(!value.trim().is_empty(), "field `{}` is empty", keys[0]);
    Ok(value.to_owned())
}

fn required_u64(record: &Map<String, Value>, keys: &[&str]) -> Result<u64> {
    required_value(record, keys)?
        .as_u64()
        .with_context(|| format!("field `{}` is not an unsigned integer", keys[0]))
}

fn required_u32(record: &Map<String, Value>, keys: &[&str]) -> Result<u32> {
    u32::try_from(required_u64(record, keys)?)
        .with_context(|| format!("field `{}` does not fit u32", keys[0]))
}

fn required_usize(record: &Map<String, Value>, keys: &[&str]) -> Result<usize> {
    usize::try_from(required_u64(record, keys)?)
        .with_context(|| format!("field `{}` does not fit usize", keys[0]))
}

fn required_i32(record: &Map<String, Value>, keys: &[&str]) -> Result<i32> {
    let value = required_value(record, keys)?
        .as_i64()
        .with_context(|| format!("field `{}` is not an integer", keys[0]))?;
    i32::try_from(value).with_context(|| format!("field `{}` does not fit i32", keys[0]))
}

fn required_f64(record: &Map<String, Value>, keys: &[&str]) -> Result<f64> {
    required_value(record, keys)?
        .as_f64()
        .with_context(|| format!("field `{}` is not numeric", keys[0]))
}

fn required_flash_attention(
    record: &Map<String, Value>,
    keys: &[&str],
) -> Result<LlamaFlashAttention> {
    let value = required_value(record, keys)?;
    if let Some(enabled) = value.as_bool() {
        return Ok(if enabled {
            LlamaFlashAttention::On
        } else {
            LlamaFlashAttention::Off
        });
    }
    match value.as_i64() {
        Some(-1) => Ok(LlamaFlashAttention::Auto),
        Some(0) => Ok(LlamaFlashAttention::Off),
        Some(1) => Ok(LlamaFlashAttention::On),
        _ => anyhow::bail!(
            "field `{}` is not boolean or the llama.cpp -1/0/1 flash-attention state",
            keys[0]
        ),
    }
}

fn required_bool(record: &Map<String, Value>, keys: &[&str]) -> Result<bool> {
    let value = required_value(record, keys)?;
    if let Some(value) = value.as_bool() {
        return Ok(value);
    }
    match value.as_i64() {
        Some(0) => Ok(false),
        Some(1) => Ok(true),
        _ => anyhow::bail!("field `{}` is not boolean or 0/1", keys[0]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VERSION: &str = "load_backend: loaded Metal backend\n\
        version: 10050 (b15ca938a)\n\
        built with AppleClang 21.0.0 for Darwin arm64\n";

    const PREFILL: &str = r#"{
      "build_commit":"b15ca938a",
      "build_number":10050,
      "model_filename":"/models/SmolLM3-Q4_K_M.gguf",
      "model_type":"smollm3 3B Q4_K - Medium",
      "model_size":1915305312,
      "model_n_params":3000000000,
      "backend":"Metal",
      "n_batch":2048,
      "n_ubatch":512,
      "n_threads":10,
      "n_gpu_layers":99,
      "flash_attn":1,
      "use_mmap":true,
      "n_prompt":512,
      "n_gen":0,
      "avg_ns":100000000,
      "stddev_ns":1000000,
      "avg_ts":5120.0,
      "stddev_ts":51.2,
      "new_upstream_field":"ignored by the explicit adapter"
    }"#;

    #[test]
    fn version_parser_handles_backend_noise_and_verifies_full_commit() -> Result<()> {
        let version = parse_llama_cpp_version(VERSION)?;
        assert_eq!(version.build, 10_050);
        assert_eq!(version.commit_abbreviation, "b15ca938a");
        assert_eq!(
            version.compiler_and_target.as_deref(),
            Some("AppleClang 21.0.0 for Darwin arm64")
        );
        let requirement = LlamaCppRequirement {
            binary_directory: "/opt/llama/bin".into(),
            source_directory: None,
            build: 10_050,
            commit: "b15ca938ad00aa6b3ee6c2edda7363fd02826b18".to_owned(),
            gpu_layers: 99,
            flash_attention: super::super::schema::LlamaFlashAttention::On,
            cache_type_k: "f16".to_owned(),
            cache_type_v: "f16".to_owned(),
            mmap: true,
        };
        version.verify(&requirement, "b15ca938ad00aa6b3ee6c2edda7363fd02826b18")?;
        assert!(
            version
                .verify(&requirement, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn homebrew_receipt_requires_a_full_source_revision() {
        let receipt = br#"{
          "source": {"scm_revision":"b15ca938ad00aa6b3ee6c2edda7363fd02826b18"}
        }"#;
        assert_eq!(
            parse_homebrew_scm_revision(receipt).unwrap(),
            "b15ca938ad00aa6b3ee6c2edda7363fd02826b18"
        );
        assert!(parse_homebrew_scm_revision(br#"{"source":{}}"#).is_err());
    }

    #[test]
    fn bench_json_and_jsonl_normalize_the_same_record() -> Result<()> {
        let json = format!("[{PREFILL}]");
        let array = parse_llama_bench_json(&json)?;
        let compact = serde_json::to_string(&serde_json::from_str::<Value>(PREFILL)?)?;
        let lines = parse_llama_bench_jsonl(&compact)?;
        assert_eq!(array, lines);
        assert_eq!(array[0].kind(), LlamaBenchKind::Prefill);
        assert_eq!(array[0].flash_attention, LlamaFlashAttention::On);
        assert!((array[0].average_tokens_per_second - 5120.0).abs() <= f64::EPSILON);
        Ok(())
    }

    #[test]
    fn bench_parser_preserves_llama_cpp_flash_attention_tristate() -> Result<()> {
        let auto = PREFILL.replace(r#""flash_attn":1"#, r#""flash_attn":-1"#);
        assert_eq!(
            parse_llama_bench_json(&format!("[{auto}]"))?[0].flash_attention,
            LlamaFlashAttention::Auto
        );
        let off = PREFILL.replace(r#""flash_attn":1"#, r#""flash_attn":false"#);
        assert_eq!(
            parse_llama_bench_json(&format!("[{off}]"))?[0].flash_attention,
            LlamaFlashAttention::Off
        );
        let invalid = PREFILL.replace(r#""flash_attn":1"#, r#""flash_attn":2"#);
        assert!(parse_llama_bench_json(&format!("[{invalid}]")).is_err());
        Ok(())
    }

    #[test]
    fn bench_parser_rejects_missing_metrics_and_empty_workloads() {
        let missing_rate = PREFILL.replace("\"avg_ts\":5120.0,", "");
        assert!(parse_llama_bench_json(&format!("[{missing_rate}]")).is_err());

        let empty = PREFILL.replace("\"n_prompt\":512", "\"n_prompt\":0");
        assert!(parse_llama_bench_json(&format!("[{empty}]")).is_err());
    }
}
