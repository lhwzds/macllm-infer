//! Privacy-preserving Apple host discovery for benchmark reports.

#[cfg(target_os = "macos")]
use std::process::Command;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

/// Explicit allowlist of publishable host attributes.
///
/// Serial numbers, hardware UUIDs, and provisioning identifiers have no fields
/// in this type and are discarded while parsing `system_profiler` output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SanitizedHost {
    pub machine_name: String,
    pub machine_model: String,
    pub chip: String,
    pub memory_bytes: u64,
    pub total_cores: Option<usize>,
    pub performance_cores: Option<usize>,
    pub efficiency_cores: Option<usize>,
    pub os_version: String,
    pub os_build: String,
    pub architecture: String,
}

impl SanitizedHost {
    /// Validate the complete allowlisted host record before publication.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.machine_name.trim().is_empty(),
            "hardware machine name cannot be empty"
        );
        ensure!(
            !self.machine_model.trim().is_empty(),
            "hardware machine model cannot be empty"
        );
        ensure!(
            !self.chip.trim().is_empty(),
            "hardware chip cannot be empty"
        );
        ensure!(self.memory_bytes > 0, "hardware memory must be positive");
        ensure!(
            !self.os_version.trim().is_empty() && !self.os_build.trim().is_empty(),
            "operating system identity cannot be empty"
        );
        ensure!(
            !self.architecture.trim().is_empty(),
            "hardware architecture cannot be empty"
        );
        ensure!(
            self.total_cores.is_none_or(|cores| cores > 0)
                && self.performance_cores.is_none_or(|cores| cores > 0),
            "total and performance core counts must be positive when present"
        );
        if let (Some(total), Some(performance), Some(efficiency)) = (
            self.total_cores,
            self.performance_cores,
            self.efficiency_cores,
        ) {
            ensure!(
                performance.checked_add(efficiency) == Some(total),
                "performance and efficiency core counts do not match the total"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct SystemProfilerRoot {
    #[serde(rename = "SPHardwareDataType")]
    hardware: Vec<HardwareOverview>,
}

/// Deliberately omits every unique hardware identifier present in the input.
#[derive(Debug, Deserialize)]
struct HardwareOverview {
    machine_name: String,
    machine_model: String,
    chip_type: String,
    physical_memory: String,
    #[serde(default)]
    number_processors: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CoreCounts {
    total: usize,
    performance: Option<usize>,
    efficiency: Option<usize>,
}

/// Parse only allowlisted fields from `system_profiler -json` bytes.
pub fn parse_system_profiler_hardware(
    bytes: &[u8],
    os_version: impl Into<String>,
    os_build: impl Into<String>,
    architecture: impl Into<String>,
) -> Result<SanitizedHost> {
    let root: SystemProfilerRoot =
        serde_json::from_slice(bytes).context("failed to parse sanitized hardware overview")?;
    ensure!(
        root.hardware.len() == 1,
        "system profiler returned an unexpected number of hardware overviews"
    );
    let hardware = root
        .hardware
        .into_iter()
        .next()
        .context("system profiler returned no hardware overview")?;
    let cores = hardware
        .number_processors
        .as_deref()
        .map(parse_core_counts)
        .transpose()?;
    let host = SanitizedHost {
        machine_name: hardware.machine_name,
        machine_model: hardware.machine_model,
        chip: hardware.chip_type,
        memory_bytes: parse_memory_bytes(&hardware.physical_memory)?,
        total_cores: cores.map(|counts| counts.total),
        performance_cores: cores.and_then(|counts| counts.performance),
        efficiency_cores: cores.and_then(|counts| counts.efficiency),
        os_version: os_version.into(),
        os_build: os_build.into(),
        architecture: architecture.into(),
    };
    host.validate()?;
    Ok(host)
}

/// Collect the local macOS profile without retaining raw identifying fields.
#[cfg(target_os = "macos")]
pub fn collect_macos_host() -> Result<SanitizedHost> {
    let hardware = Command::new("/usr/sbin/system_profiler")
        .args(["SPHardwareDataType", "-json"])
        .output()
        .context("failed to run system_profiler")?;
    ensure!(
        hardware.status.success(),
        "system_profiler failed while collecting benchmark host metadata"
    );
    let os_version = sw_vers("-productVersion")?;
    let os_build = sw_vers("-buildVersion")?;
    parse_system_profiler_hardware(
        &hardware.stdout,
        os_version,
        os_build,
        std::env::consts::ARCH,
    )
}

/// Apple Silicon scoreboards are intentionally unavailable on other hosts.
#[cfg(not(target_os = "macos"))]
pub fn collect_macos_host() -> Result<SanitizedHost> {
    anyhow::bail!("MacLLM benchmark host collection requires macOS")
}

#[cfg(target_os = "macos")]
fn sw_vers(flag: &str) -> Result<String> {
    let output = Command::new("/usr/bin/sw_vers")
        .arg(flag)
        .output()
        .with_context(|| format!("failed to run sw_vers {flag}"))?;
    ensure!(
        output.status.success(),
        "sw_vers failed while collecting benchmark host metadata"
    );
    let value = std::str::from_utf8(&output.stdout)
        .context("sw_vers returned non-UTF-8 output")?
        .trim();
    ensure!(!value.is_empty(), "sw_vers returned an empty value");
    Ok(value.to_owned())
}

fn parse_memory_bytes(value: &str) -> Result<u64> {
    let mut parts = value.split_ascii_whitespace();
    let amount = parts
        .next()
        .context("hardware memory has no numeric value")?
        .replace(',', "")
        .parse::<u64>()
        .context("hardware memory is not an integer")?;
    let unit = parts
        .next()
        .context("hardware memory has no unit")?
        .to_ascii_uppercase();
    ensure!(
        parts.next().is_none(),
        "hardware memory has unexpected trailing data"
    );
    let multiplier = match unit.as_str() {
        "B" => 1,
        "KB" => 1_u64 << 10,
        "MB" => 1_u64 << 20,
        "GB" => 1_u64 << 30,
        "TB" => 1_u64 << 40,
        _ => anyhow::bail!("hardware memory uses an unsupported unit"),
    };
    amount
        .checked_mul(multiplier)
        .context("hardware memory byte count overflow")
}

fn parse_core_counts(value: &str) -> Result<CoreCounts> {
    let encoded = value
        .split_ascii_whitespace()
        .last()
        .context("hardware core count is empty")?;
    let values = encoded
        .split(':')
        .map(|part| {
            part.parse::<usize>()
                .context("hardware core count is not an integer")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !values.is_empty() && values[0] > 0,
        "hardware core total must be positive"
    );
    let performance = values.get(1).copied();
    let efficiency = values.get(2).copied();
    if let (Some(performance), Some(efficiency)) = (performance, efficiency) {
        ensure!(
            performance.checked_add(efficiency) == Some(values[0]),
            "hardware core classes do not match the total"
        );
    }
    Ok(CoreCounts {
        total: values[0],
        performance,
        efficiency,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const HARDWARE_JSON_WITH_SECRETS: &str = r#"{
      "SPHardwareDataType": [{
        "_name": "hardware_overview",
        "machine_name": "MacBook Pro",
        "machine_model": "Mac16,7",
        "chip_type": "Apple M4 Pro",
        "physical_memory": "48 GB",
        "number_processors": "proc 14:10:4:0",
        "serial_number": "DO-NOT-PUBLISH",
        "platform_UUID": "SECRET-UUID",
        "provisioning_UDID": "SECRET-UDID"
      }]
    }"#;

    #[test]
    fn parser_drops_every_unique_hardware_identifier() -> Result<()> {
        let host = parse_system_profiler_hardware(
            HARDWARE_JSON_WITH_SECRETS.as_bytes(),
            "26.5.2",
            "25F84",
            "arm64",
        )?;
        assert_eq!(host.machine_model, "Mac16,7");
        assert_eq!(host.chip, "Apple M4 Pro");
        assert_eq!(host.memory_bytes, 48 * 1024 * 1024 * 1024);
        assert_eq!(host.total_cores, Some(14));
        assert_eq!(host.performance_cores, Some(10));
        assert_eq!(host.efficiency_cores, Some(4));

        let serialized = serde_json::to_string(&host)?;
        let lowercase = serialized.to_ascii_lowercase();
        for forbidden in ["serial", "uuid", "udid", "do-not-publish", "secret"] {
            assert!(!lowercase.contains(forbidden), "leaked {forbidden}");
        }
        Ok(())
    }

    #[test]
    fn invalid_memory_and_core_shapes_are_rejected_without_echoing_input() {
        assert!(parse_memory_bytes("48 parsecs").is_err());
        assert!(parse_core_counts("proc 14:8:4:0").is_err());
        assert!(parse_core_counts("").is_err());

        let mut host = parse_system_profiler_hardware(
            HARDWARE_JSON_WITH_SECRETS.as_bytes(),
            "26.5.2",
            "25F84",
            "arm64",
        )
        .unwrap();
        host.performance_cores = Some(0);
        host.efficiency_cores = Some(14);
        assert!(host.validate().is_err());
    }

    #[test]
    fn serialized_host_schema_rejects_future_or_sensitive_fields() -> Result<()> {
        let host = parse_system_profiler_hardware(
            HARDWARE_JSON_WITH_SECRETS.as_bytes(),
            "26.5.2",
            "25F84",
            "arm64",
        )?;
        let mut value = serde_json::to_value(host)?;
        value["serial_number"] = serde_json::json!("forbidden");
        assert!(serde_json::from_value::<SanitizedHost>(value).is_err());
        Ok(())
    }
}
