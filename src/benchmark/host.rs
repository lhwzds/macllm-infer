//! Privacy-preserving host discovery for benchmark reports.
//!
//! macOS keeps the Apple profile that `system_profiler` reports. Linux and
//! Windows report a sanitized profile of their own: the operating system
//! family, the hardware model, the processor, the installed memory, and the
//! core counts. No host name, serial number, or other unique identifier ever
//! reaches a report.

use std::{
    fs,
    io::Read,
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

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
///
/// This refusal is what keeps benchmark publication off Windows: some
/// guards run before it, but nothing is published. The `\` separators in
/// checkpoint paths are fixed now (`LocalFileDigest::capture` in `task.rs`),
/// and a task benchmark records the sanitized host of every operating system
/// (`collect_task_host`), but publication still needs the file identity
/// (including directories, see `file_stamp`), the no-follow bundle reads
/// (`prefix_demo.rs`), and the directory sync that `commit_artifact_pair`
/// skips.
#[cfg(not(target_os = "macos"))]
pub fn collect_macos_host() -> Result<SanitizedHost> {
    macos_host_refusal()
}

/// The refusal that every host that is not macOS returns.
///
/// The scoreboard verification calls `collect_macos_host`, and the prefix
/// demo report carries the host that the same verification produced, so this
/// refusal is what keeps every published artifact on macOS. Task benchmarks
/// use [`collect_task_host`] instead, and they publish nothing.
#[cfg(any(not(target_os = "macos"), test))]
fn macos_host_refusal() -> Result<SanitizedHost> {
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

/// The machine name that a Linux report carries.
///
/// It names the operating system family, never the host name of the machine
/// that ran the benchmark.
const LINUX_MACHINE_NAME: &str = "Linux host";

/// The machine name that a Windows report carries.
///
/// It names the operating system family, never the host name of the machine
/// that ran the benchmark.
const WINDOWS_MACHINE_NAME: &str = "Windows host";

/// The value that a host record carries for a field the host does not expose.
const UNKNOWN_HOST_FIELD: &str = "unknown";

/// The deadline that every child process that collects host metadata gets.
const HOST_COMMAND_DEADLINE: Duration = Duration::from_secs(10);

/// The most bytes that one stream of a host command may produce.
///
/// Every reader stops reading at this cap, so a child that prints without
/// end cannot grow the memory of the benchmark. The output past the cap is
/// not read at all: a child that keeps writing blocks while the pipe is
/// full, and it gets a write error once the reader stopped at the cap and
/// dropped the pipe. The deadline of the command ends it either way, and the
/// output that was read is parsed as it is, which is harmless for the `reg`,
/// `powershell` and `sysctl` output that the host commands collect (a few
/// hundred bytes).
const HOST_COMMAND_OUTPUT_CAP: u64 = 1024 * 1024;

/// The size of one read of a stream of a host command.
///
/// The reader reads in chunks, so a reader that outlives its call stops
/// appending at the next chunk instead of at the end of its buffer.
const READER_CHUNK_BYTES: usize = 8 * 1024;

/// The bound on the polls that reap a child after it was killed.
const KILL_REAP_WAIT: Duration = Duration::from_secs(2);

/// The bound on joining the reader threads of one host command.
const READER_JOIN_WAIT: Duration = Duration::from_secs(2);

/// The pause between two polls of one child process.
const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// The number of characters that an error message keeps from a child.
const STDERR_SUMMARY_CHARS: usize = 200;

/// The DMI file that names the hardware model of a Linux machine.
const LINUX_DMI_PRODUCT_NAME: &str = "/sys/devices/virtual/dmi/id/product_name";

/// The processor list of a Linux machine.
const LINUX_CPUINFO: &str = "/proc/cpuinfo";

/// The memory list of a Linux machine.
const LINUX_MEMINFO: &str = "/proc/meminfo";

/// The operating system release of a Linux machine.
const LINUX_OS_RELEASE: &str = "/etc/os-release";

/// The kernel release of a Linux machine.
const LINUX_KERNEL_RELEASE: &str = "/proc/sys/kernel/osrelease";

/// The registry key that names the processor of a Windows machine.
const WINDOWS_PROCESSOR_KEY: &str = r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0";

/// The registry key that names the hardware model of a Windows machine.
const WINDOWS_BIOS_KEY: &str = r"HKLM\HARDWARE\DESCRIPTION\System\BIOS";

/// The registry key that names the operating system of a Windows machine.
const WINDOWS_CURRENT_VERSION_KEY: &str = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion";

/// The system root that a Windows machine carries when it names none.
const WINDOWS_DEFAULT_SYSTEM_ROOT: &str = r"C:\Windows";

/// The `reg.exe` of a Windows machine, inside its system directory.
const WINDOWS_REG_PROGRAM: &str = r"System32\reg.exe";

/// The `powershell.exe` of a Windows machine, inside its system directory.
const WINDOWS_POWERSHELL_PROGRAM: &str = r"System32\WindowsPowerShell\v1.0\powershell.exe";

/// Collect the host record that a task benchmark reports.
///
/// macOS returns exactly what [`collect_macos_host`] returns. Linux and
/// Windows return a sanitized record of their own hardware and operating
/// system, and both name the operating system family rather than the host.
pub fn collect_task_host() -> Result<SanitizedHost> {
    #[cfg(target_os = "macos")]
    {
        collect_macos_host()
    }
    #[cfg(target_os = "linux")]
    {
        collect_linux_host()
    }
    #[cfg(target_os = "windows")]
    {
        collect_windows_host()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        anyhow::bail!("MacLLM task benchmarks require macOS, Linux, or Windows")
    }
}

/// Collect the sanitized host profile of a Linux machine.
///
/// The record names the hardware model, the first processor of the machine,
/// the installed memory, the operating system release, and the number of
/// logical cores the benchmark may use. Performance and efficiency core counts
/// stay empty, because a Linux machine does not report them.
///
/// The record reads the files that Linux exposes, so this only succeeds on a
/// Linux machine.
pub fn collect_linux_host() -> Result<SanitizedHost> {
    linux_host_from_paths(
        LINUX_DMI_PRODUCT_NAME,
        LINUX_CPUINFO,
        LINUX_MEMINFO,
        LINUX_OS_RELEASE,
        LINUX_KERNEL_RELEASE,
    )
}

/// Read the files that a Linux machine exposes and build the host record.
///
/// A file that the machine hides leaves its field empty, so the record names
/// the unknown marker rather than a value that the machine never reported.
fn linux_host_from_paths(
    product_name_path: &str,
    cpuinfo_path: &str,
    meminfo_path: &str,
    os_release_path: &str,
    kernel_release_path: &str,
) -> Result<SanitizedHost> {
    let product_name = read_optional_file(product_name_path);
    let cpuinfo = read_optional_file(cpuinfo_path).unwrap_or_default();
    let meminfo = fs::read_to_string(meminfo_path)
        .with_context(|| format!("failed to read {meminfo_path}"))?;
    let os_release = read_optional_file(os_release_path).unwrap_or_default();
    let kernel = read_optional_file(kernel_release_path).unwrap_or_default();
    linux_host_from_files(
        product_name.as_deref(),
        &cpuinfo,
        &meminfo,
        &os_release,
        &kernel,
    )
}

/// Build the Linux host record from the documents that the host exposes.
///
/// A document that the host hides leaves its field empty, so the record names
/// the unknown marker rather than a value that the machine never reported.
fn linux_host_from_files(
    product_name: Option<&str>,
    cpuinfo: &str,
    meminfo: &str,
    os_release: &str,
    kernel_release: &str,
) -> Result<SanitizedHost> {
    let machine_model = product_name
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(UNKNOWN_HOST_FIELD)
        .to_owned();
    let chip = parse_linux_cpuinfo(cpuinfo).unwrap_or_else(|_| UNKNOWN_HOST_FIELD.to_owned());
    let memory_bytes = parse_linux_meminfo(meminfo)?;
    let release = parse_linux_os_release(os_release);
    let kernel = kernel_release.trim();
    let kernel = if kernel.is_empty() {
        UNKNOWN_HOST_FIELD
    } else {
        kernel
    };
    let os_version = release.version_id.unwrap_or_else(|| kernel.to_owned());
    let os_build = release.build_id.unwrap_or_else(|| kernel.to_owned());
    let host = SanitizedHost {
        machine_name: LINUX_MACHINE_NAME.to_owned(),
        machine_model,
        chip,
        memory_bytes,
        total_cores: available_parallelism(),
        performance_cores: None,
        efficiency_cores: None,
        os_version,
        os_build,
        architecture: std::env::consts::ARCH.to_owned(),
    };
    host.validate()?;
    Ok(host)
}

/// Collect the sanitized host profile of a Windows machine.
///
/// The record names the hardware model, the processor, the installed memory,
/// the operating system version and build, and the number of logical cores
/// the benchmark may use. Performance and efficiency core counts stay empty,
/// because a Windows machine does not report them.
///
/// The record reads the registry and the memory of the host, so this only
/// succeeds on a Windows machine.
pub fn collect_windows_host() -> Result<SanitizedHost> {
    let processor = windows_registry_value(WINDOWS_PROCESSOR_KEY, "ProcessorNameString").ok();
    let product = windows_registry_value(WINDOWS_BIOS_KEY, "SystemProductName").ok();
    let os_version = windows_registry_value(WINDOWS_CURRENT_VERSION_KEY, "DisplayVersion").ok();
    let os_build = windows_registry_value(WINDOWS_CURRENT_VERSION_KEY, "CurrentBuildNumber").ok();
    let memory_bytes = windows_total_physical_memory()?;
    windows_host_from_values(
        processor.as_deref(),
        product.as_deref(),
        os_version.as_deref(),
        os_build.as_deref(),
        memory_bytes,
    )
}

/// Build the Windows host record from the values that the host exposes.
///
/// A value that the registry does not carry leaves its field empty, so the
/// record names the unknown marker rather than a value that the machine never
/// reported.
fn windows_host_from_values(
    processor: Option<&str>,
    product: Option<&str>,
    os_version: Option<&str>,
    os_build: Option<&str>,
    memory_bytes: u64,
) -> Result<SanitizedHost> {
    let host = SanitizedHost {
        machine_name: WINDOWS_MACHINE_NAME.to_owned(),
        machine_model: nonempty_or_unknown(product.unwrap_or_default()),
        chip: nonempty_or_unknown(processor.unwrap_or_default()),
        memory_bytes,
        total_cores: available_parallelism(),
        performance_cores: None,
        efficiency_cores: None,
        os_version: nonempty_or_unknown(os_version.unwrap_or_default()),
        os_build: nonempty_or_unknown(os_build.unwrap_or_default()),
        architecture: std::env::consts::ARCH.to_owned(),
    };
    host.validate()?;
    Ok(host)
}

/// Read one value of the Windows registry.
///
/// The registry is read with `reg query`, so no new dependency and no
/// privileged access is needed. The program runs by absolute path, like the
/// macOS collection does: Rust searches the directory of the running
/// executable before the system directory, so a `reg.exe` or a `powershell.exe`
/// dropped next to the benchmark must not win.
fn windows_registry_value(key: &str, value_name: &str) -> Result<String> {
    let reg = windows_system_program(
        &windows_system_root(std::env::var("SystemRoot").ok()),
        WINDOWS_REG_PROGRAM,
    );
    let output = run_host_command(
        reg.as_str(),
        &["query", key, "/v", value_name],
        HOST_COMMAND_DEADLINE,
    )
    .with_context(|| format!("failed to query the Windows registry value {value_name}"))?;
    parse_windows_reg_value(&output, value_name)
}

/// Read the installed physical memory of a Windows machine, in bytes.
///
/// The size comes from the Windows management instrumentation, which needs no
/// new dependency and runs without an elevated shell. The shell runs by
/// absolute path for the same reason as `reg.exe` above.
fn windows_total_physical_memory() -> Result<u64> {
    let powershell = windows_system_program(
        &windows_system_root(std::env::var("SystemRoot").ok()),
        WINDOWS_POWERSHELL_PROGRAM,
    );
    let output = run_host_command(
        powershell.as_str(),
        &[
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ],
        HOST_COMMAND_DEADLINE,
    )
    .context("failed to query the physical memory of the Windows host")?;
    parse_windows_memory_bytes(&output)
}

/// The system root of this Windows machine.
///
/// The environment names it, and a machine that hides the variable still
/// carries the directory that every Windows installation uses.
fn windows_system_root(environment_root: Option<String>) -> String {
    environment_root.unwrap_or_else(|| WINDOWS_DEFAULT_SYSTEM_ROOT.to_owned())
}

/// The absolute path of one program of the Windows system directory.
fn windows_system_program(system_root: &str, relative: &str) -> String {
    format!(r"{system_root}\{relative}")
}

/// Parse the byte count that the Windows memory query prints.
///
/// The query prints one number, so a shape that carries no number at all is a
/// failure rather than a guess. A host that adds a notice before the number
/// still reports the number on the last line that carries one.
pub fn parse_windows_memory_bytes(output: &str) -> Result<u64> {
    let value = output.trim();
    ensure!(
        !value.is_empty(),
        "the Windows memory query returned no value"
    );
    if let Ok(bytes) = value.parse::<u64>() {
        return Ok(bytes);
    }
    for line in value.lines().rev() {
        let line = line.trim();
        if let Ok(bytes) = line.parse::<u64>() {
            return Ok(bytes);
        }
    }
    anyhow::bail!("the Windows memory query did not return a byte count")
}

/// Parse one named value out of the output of `reg query`.
///
/// The output names the key on its own line, then one line per value that
/// holds the value name, its registry type, and its data.
pub fn parse_windows_reg_value(output: &str, value_name: &str) -> Result<String> {
    for line in output.lines() {
        let trimmed = line.trim();
        let Some(prefix) = trimmed.get(..value_name.len()) else {
            continue;
        };
        if !prefix.eq_ignore_ascii_case(value_name) {
            continue;
        }
        let rest = trimmed[value_name.len()..].trim_start();
        let Some(kind) = rest.split_whitespace().next() else {
            continue;
        };
        if !kind.to_ascii_uppercase().starts_with("REG_") {
            continue;
        }
        let value = rest[kind.len()..].trim();
        ensure!(
            !value.is_empty(),
            "the Windows registry value {value_name} is empty"
        );
        return Ok(value.to_owned());
    }
    anyhow::bail!("the Windows registry output has no {value_name} value")
}

/// Parse the first `model name` entry of a `/proc/cpuinfo` document.
pub fn parse_linux_cpuinfo(text: &str) -> Result<String> {
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.trim() != "model name" {
            continue;
        }
        let value = value.trim();
        ensure!(!value.is_empty(), "the Linux processor name is empty");
        return Ok(value.to_owned());
    }
    anyhow::bail!("the Linux processor list has no model name entry")
}

/// Parse the `MemTotal` entry of a `/proc/meminfo` document into bytes.
pub fn parse_linux_meminfo(text: &str) -> Result<u64> {
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.trim() != "MemTotal" {
            continue;
        }
        let mut fields = value.split_ascii_whitespace();
        let amount = fields
            .next()
            .context("the Linux memory total has no numeric value")?
            .parse::<u64>()
            .context("the Linux memory total is not an integer")?;
        let multiplier = match fields.next() {
            None | Some("kB" | "KB") => 1_u64 << 10,
            Some("B") => 1,
            Some("mB" | "MB") => 1_u64 << 20,
            Some("gB" | "GB") => 1_u64 << 30,
            Some(unit) => anyhow::bail!("the Linux memory total uses an unsupported unit: {unit}"),
        };
        return amount
            .checked_mul(multiplier)
            .context("the Linux memory byte count overflows");
    }
    anyhow::bail!("the Linux memory list has no MemTotal entry")
}

/// The operating system identity that an `/etc/os-release` document carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinuxOsRelease {
    /// The `VERSION_ID` field, when the document carries one.
    pub version_id: Option<String>,
    /// The `BUILD_ID` field, when the document carries one.
    pub build_id: Option<String>,
}

/// Parse the `VERSION_ID` and `BUILD_ID` fields of an `/etc/os-release`.
///
/// A field that the document does not carry stays empty, so the caller can
/// fall back to the kernel release.
#[must_use]
pub fn parse_linux_os_release(text: &str) -> LinuxOsRelease {
    let mut release = LinuxOsRelease {
        version_id: None,
        build_id: None,
    };
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = unquote_os_release_value(value);
        if value.is_empty() {
            continue;
        }
        match key.trim() {
            "VERSION_ID" => release.version_id = Some(value),
            "BUILD_ID" => release.build_id = Some(value),
            _ => {}
        }
    }
    release
}

/// Strip the optional quotes that an `/etc/os-release` value carries.
///
/// Both quote styles appear in the wild, and only a matching pair is stripped,
/// so a value that quotes itself on one side keeps that quote.
fn unquote_os_release_value(value: &str) -> String {
    let trimmed = value.trim();
    let unquoted = trimmed
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .or_else(|| {
            trimmed
                .strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        })
        .unwrap_or(trimmed);
    unquoted.to_owned()
}

/// Read one text file of the host, ignoring a file that the host hides.
fn read_optional_file(path: &str) -> Option<String> {
    fs::read_to_string(path).ok()
}

/// The number of logical cores that this process may run on.
fn available_parallelism() -> Option<usize> {
    std::thread::available_parallelism()
        .ok()
        .map(std::num::NonZeroUsize::get)
}

/// The value that a host record carries for a field that the host left empty.
fn nonempty_or_unknown(value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        UNKNOWN_HOST_FIELD.to_owned()
    } else {
        value.to_owned()
    }
}

/// Run one child process that collects host metadata under a deadline.
///
/// The child never reads the benchmark's standard input, and it is always
/// reaped: a process that outlives its deadline is killed and then polled for
/// a bounded time, so no child survives the benchmark and no wait blocks
/// without an end. Neither reader outlives this call either, except the
/// grandchild case that [`join_command_readers`] documents: both are joined
/// before it returns, and each stops at [`HOST_COMMAND_OUTPUT_CAP`] bytes. A
/// reader that outlives the call stops appending once the call returned. A
/// failing child is an error with context, never a panic.
fn run_host_command(program: &str, args: &[&str], deadline: Duration) -> Result<String> {
    let start = Instant::now();
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to run {program}"))?;
    // A pipe that the child does not carry leaves the child running, so
    // every failure of this function stops and reaps it before it returns.
    let Some(stdout) = child.stdout.take() else {
        let reap = kill_and_reap(&mut child, program);
        return Err(with_reap(
            anyhow::anyhow!("the host command has no standard output pipe"),
            reap,
        ));
    };
    let Some(stderr) = child.stderr.take() else {
        let reap = kill_and_reap(&mut child, program);
        return Err(with_reap(
            anyhow::anyhow!("the host command has no standard error pipe"),
            reap,
        ));
    };
    let (sender, receiver) = mpsc::channel();
    // The readers stop appending once this flag is set, so a reader that
    // outlives this call cannot fill the memory of the benchmark after the
    // call returned.
    let stop = Arc::new(AtomicBool::new(false));
    let stdout_sender = sender.clone();
    let stdout_stop = Arc::clone(&stop);
    let stdout_reader = thread::spawn(move || {
        read_command_stream(
            stdout,
            stdout_sender,
            HostCommandStream::Stdout,
            stdout_stop,
        );
    });
    let stderr_stop = Arc::clone(&stop);
    let stderr_reader = thread::spawn(move || {
        read_command_stream(stderr, sender, HostCommandStream::Stderr, stderr_stop);
    });
    let mut stderr_output = Vec::new();
    let outcome = collect_command_output(
        &mut child,
        &receiver,
        &mut stderr_output,
        program,
        deadline,
        start,
    );
    // Every return path joins both readers: the child has been killed and
    // reaped by the time this runs when its reap succeeded, so its pipes are
    // closed and the joins end at once. A reap that failed leaves the pipes
    // open, but the join is bounded either way and the error of the command
    // carries the failure of the reap. The one exception, a grandchild that
    // holds a pipe open, is documented on `join_command_readers`.
    let joined = join_command_readers(
        stdout_reader,
        stderr_reader,
        program,
        reader_join_bound(deadline, start),
    );
    // The readers stop appending once this call returns, so a reader that
    // outlived it (a grandchild that holds a pipe open) keeps no buffer
    // that grows after the call returned.
    stop.store(true, Ordering::SeqCst);
    // A reader that finished after the loop delivered its stream through the
    // channel, and both senders are gone once the readers joined, so what is
    // left there completes the standard error of the child.
    if let Some(buffer) = drain_command_stderr(&receiver) {
        stderr_output = buffer;
    }
    let (output, status) = match (outcome, joined) {
        (Ok(outcome), Ok(())) => outcome,
        (Err(error), Ok(())) => return Err(error),
        (Ok(_), Err(join_error)) => {
            return Err(join_error.context(format!("failed to collect the output of {program}")));
        }
        // A reader that outlived the command follows the error of the command
        // in the chain: anyhow prints the context first, so the failing
        // command stays the primary message. The whole chain of the command
        // error is the context, so a reap that failed is not lost. The join
        // error follows the same rule as a failed reap, so both chains are
        // built by `with_reap`.
        (Err(error), Err(join_error)) => {
            return Err(with_reap(error, Err(join_error)));
        }
    };
    ensure!(
        status.success(),
        "{program} failed with status {status}: {}",
        stderr_summary(&stderr_output)
    );
    Ok(String::from_utf8_lossy(&output).into_owned())
}

/// One stream that a host command produced.
enum HostCommandStream {
    Stdout(Vec<u8>),
    Stderr(Vec<u8>),
}

/// Read one stream of a host command to its end and report it.
///
/// The read runs through [`Read::take`], so it stops at
/// [`HOST_COMMAND_OUTPUT_CAP`] bytes even when the child never closes the
/// stream. The stream is read in chunks of [`READER_CHUNK_BYTES`], and the
/// reader stops appending once `stop` is set: a reader that outlives its
/// call must not fill the memory of the benchmark behind the back of the
/// caller.
fn read_command_stream<R: Read>(
    stream: R,
    sender: mpsc::Sender<HostCommandStream>,
    wrap: impl Fn(Vec<u8>) -> HostCommandStream,
    stop: Arc<AtomicBool>,
) {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; READER_CHUNK_BYTES];
    let mut reader = stream.take(HOST_COMMAND_OUTPUT_CAP);
    loop {
        if stop.load(Ordering::SeqCst) {
            // The call returned, so the buffer of this reader stops here.
            break;
        }
        let count = match reader.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        if stop.load(Ordering::SeqCst) {
            // The call returned while this chunk was read, so the chunk is
            // dropped instead of appended.
            break;
        }
        buffer.extend_from_slice(&chunk[..count]);
    }
    let _ = sender.send(wrap(buffer));
}

/// Wait for the standard output and the exit of one host command.
///
/// The standard error that arrives while the command runs is collected into
/// `stderr`; the rest of it is collected once both readers joined.
fn collect_command_output(
    child: &mut Child,
    receiver: &mpsc::Receiver<HostCommandStream>,
    stderr: &mut Vec<u8>,
    program: &str,
    deadline: Duration,
    start: Instant,
) -> Result<(Vec<u8>, ExitStatus)> {
    let output = loop {
        let remaining = deadline.saturating_sub(start.elapsed());
        if remaining.is_zero() {
            return Err(command_deadline_error(child, program, deadline));
        }
        match receiver.recv_timeout(remaining) {
            Ok(HostCommandStream::Stdout(buffer)) => break buffer,
            // The standard error is collected once both readers joined.
            Ok(HostCommandStream::Stderr(buffer)) => *stderr = buffer,
            // This arm runs once the deadline *has* passed: the child below
            // is the one that ran out of time.
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err(command_deadline_error(child, program, deadline));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let reap = kill_and_reap(child, program);
                let error = anyhow::anyhow!("{program} produced no output");
                return Err(with_reap(error, reap));
            }
        }
    };
    let status = wait_for_child(child, program, deadline, start)?;
    Ok((output, status))
}

/// Join both reader threads of one host command, bounded by `bound`.
///
/// The child has been killed and reaped by the time this runs when the reap
/// of the child succeeded, so its pipes are closed and the joins end at
/// once. A reap that failed leaves the pipes open, but the join is bounded
/// by `bound` either way, and the error of the command carries the failure
/// of the reap. The one case that can still hold a reader open is a
/// grandchild that inherited the pipe: that reader is left blocked on the
/// pipe (it still stops at [`HOST_COMMAND_OUTPUT_CAP`], and it stops
/// appending once the call returned) and the command fails, because waiting
/// for a process the benchmark never spawned has no end.
fn join_command_readers(
    stdout: thread::JoinHandle<()>,
    stderr: thread::JoinHandle<()>,
    program: &str,
    bound: Duration,
) -> Result<()> {
    let limit = Instant::now() + bound;
    while !stdout.is_finished() || !stderr.is_finished() {
        if Instant::now() >= limit {
            // Join every reader that did finish, so only the one that a
            // grandchild keeps open outlives this call.
            if stdout.is_finished() {
                let _ = stdout.join();
            }
            if stderr.is_finished() {
                let _ = stderr.join();
            }
            anyhow::bail!(
                "{program} left a reader running after {bound:?}: a grandchild holds the pipe open"
            );
        }
        thread::sleep(CHILD_POLL_INTERVAL);
    }
    stdout
        .join()
        .map_err(|_| anyhow::anyhow!("a reader of {program} panicked"))?;
    stderr
        .join()
        .map_err(|_| anyhow::anyhow!("a reader of {program} panicked"))?;
    Ok(())
}

/// The time that joining the readers of one host command may still take.
///
/// A command that has time left gives it to its readers, and a command that
/// already spent its deadline still gets the join bound, because the reaped
/// child has closed its pipes.
fn reader_join_bound(deadline: Duration, start: Instant) -> Duration {
    READER_JOIN_WAIT.max(deadline.saturating_sub(start.elapsed()))
}

/// Collect the standard error that the joined readers delivered.
///
/// Each stream is reported once, so this returns the standard error of the
/// child when a reader finished after the wait loop stopped reading.
fn drain_command_stderr(receiver: &mpsc::Receiver<HostCommandStream>) -> Option<Vec<u8>> {
    let mut stderr = None;
    while let Ok(stream) = receiver.try_recv() {
        if let HostCommandStream::Stderr(buffer) = stream {
            stderr = Some(buffer);
        }
    }
    stderr
}

/// The last line of a child's standard error, for an error message.
fn stderr_summary(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let line = text
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("no error output");
    line.chars().take(STDERR_SUMMARY_CHARS).collect()
}

/// Wait for one child process, killing it when it outlives the deadline.
fn wait_for_child(
    child: &mut Child,
    program: &str,
    deadline: Duration,
    start: Instant,
) -> Result<ExitStatus> {
    let remaining = deadline.saturating_sub(start.elapsed());
    match poll_child(child, remaining) {
        Ok(Some(status)) => Ok(status),
        Ok(None) => Err(command_deadline_error(child, program, deadline)),
        // A child whose status cannot be read is stopped and reaped, so no
        // child survives the failure.
        Err(error) => {
            let reap = kill_and_reap(child, program);
            let error = anyhow::Error::new(error).context(format!("failed to wait for {program}"));
            Err(with_reap(error, reap))
        }
    }
}

/// Poll one child until it exits, at the latest after `bound`.
///
/// The same bounded poll waits for a running child and reaps a killed one, so
/// no path of a host command blocks without an end. A child that is still
/// running at the bound returns [`None`] rather than a wait.
fn poll_child(child: &mut Child, bound: Duration) -> std::io::Result<Option<ExitStatus>> {
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(Some(status)),
            Ok(None) => {}
            Err(error) => return Err(error),
        }
        let elapsed = start.elapsed();
        if elapsed >= bound {
            return Ok(None);
        }
        thread::sleep(bound.saturating_sub(elapsed).min(CHILD_POLL_INTERVAL));
    }
}

/// The error of a host command that outlived its deadline.
///
/// The deadline message stays the primary one, and a reap that failed follows
/// it as context, because the kill and the reap are what the deadline caused.
fn command_deadline_error(child: &mut Child, program: &str, deadline: Duration) -> anyhow::Error {
    let reap = kill_and_reap(child, program);
    with_reap(
        anyhow::anyhow!("{program} did not finish within {deadline:?}"),
        reap,
    )
}

/// Attach the result of a reap to the error that caused the kill.
///
/// The message of the command stays the primary one and the failed reap
/// follows it in the chain: anyhow prints the context first, so the message of
/// the command is the context here and the failed reap is its cause. The
/// whole chain of the command error is the context, so a cause of that error
/// is not lost.
///
/// The join of the readers follows the same rule, so the error of a reader
/// that outlived the command keeps the whole chain of the command error as
/// well.
fn with_reap(error: anyhow::Error, reap: Result<()>) -> anyhow::Error {
    match reap {
        Ok(()) => error,
        Err(reap_error) => reap_error.context(format!("{error:#}")),
    }
}

/// Kill one child process and reap it, so that it leaves no zombie behind.
///
/// The kill is checked, and the reap is a bounded poll rather than a bare
/// wait: a child that the kill did not stop must fail the command within
/// [`KILL_REAP_WAIT`] instead of holding the benchmark open.
fn kill_and_reap(child: &mut Child, program: &str) -> Result<()> {
    if let Err(error) = child.kill() {
        // A child that already exited cannot be killed again, which is not a
        // failure of the command; anything else is.
        let already_exited = error.kind() == std::io::ErrorKind::InvalidInput
            || child.try_wait().is_ok_and(|status| status.is_some());
        if !already_exited {
            return Err(error).with_context(|| format!("failed to kill {program}"));
        }
    }
    match poll_child(child, KILL_REAP_WAIT) {
        Ok(Some(_)) => Ok(()),
        Ok(None) => {
            anyhow::bail!("{program} was killed but did not exit within {KILL_REAP_WAIT:?}")
        }
        Err(error) => Err(error).with_context(|| format!("failed to reap {program}")),
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::{
        ffi::OsStr,
        path::{Path, PathBuf},
    };

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

    const LINUX_CPUINFO_TEXT: &str = "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz\ncpu family\t: 6\n\nprocessor\t: 1\nmodel name\t: Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz\n";

    const LINUX_MEMINFO_TEXT: &str =
        "MemTotal:       65792436 kB\nMemFree:        1000 kB\nBuffers:          200 kB\n";

    const LINUX_OS_RELEASE_TEXT: &str = "NAME=\"Ubuntu\"\nVERSION=\"22.04.4 LTS (Jammy Jellyfish)\"\nID=ubuntu\nVERSION_ID=\"22.04\"\nBUILD_ID=\"20240315\"\n";

    const WINDOWS_REG_OUTPUT: &str = "\r\nHKEY_LOCAL_MACHINE\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0\r\n    ProcessorNameString    REG_SZ    Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz\r\n\r\n";

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

    #[test]
    fn linux_processor_parser_reads_the_first_model_name() -> Result<()> {
        let chip = parse_linux_cpuinfo(LINUX_CPUINFO_TEXT)?;
        assert_eq!(chip, "Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz");
        Ok(())
    }

    #[test]
    fn linux_processor_parser_rejects_a_list_without_a_model_name() {
        assert!(parse_linux_cpuinfo("processor\t: 0\nvendor_id\t: GenuineIntel\n").is_err());
        assert!(parse_linux_cpuinfo("model name\t: \n").is_err());
        assert!(parse_linux_cpuinfo("").is_err());
    }

    #[test]
    fn linux_memory_parser_converts_the_total_into_bytes() -> Result<()> {
        assert_eq!(parse_linux_meminfo(LINUX_MEMINFO_TEXT)?, 65_792_436 * 1024);
        assert_eq!(parse_linux_meminfo("MemTotal: 1024 kB\n")?, 1024 * 1024);
        assert_eq!(parse_linux_meminfo("MemTotal: 2048\n")?, 2048 * 1024);
        assert_eq!(parse_linux_meminfo("MemTotal: 16 B\n")?, 16);
        Ok(())
    }

    #[test]
    fn linux_memory_parser_rejects_a_list_without_a_total() {
        assert!(parse_linux_meminfo("MemFree: 1000 kB\n").is_err());
        assert!(parse_linux_meminfo("MemTotal: many kB\n").is_err());
        assert!(parse_linux_meminfo("MemTotal: 12 parsecs\n").is_err());
        assert!(parse_linux_meminfo("").is_err());
    }

    #[test]
    fn linux_release_parser_reads_the_version_and_the_build() {
        let release = parse_linux_os_release(LINUX_OS_RELEASE_TEXT);
        assert_eq!(release.version_id.as_deref(), Some("22.04"));
        assert_eq!(release.build_id.as_deref(), Some("20240315"));
    }

    #[test]
    fn linux_release_parser_strips_both_quote_styles() {
        let release = parse_linux_os_release(
            "NAME='Debian GNU/Linux'\nVERSION_ID='12'\nBUILD_ID=\"20240617\"\n",
        );
        assert_eq!(release.version_id.as_deref(), Some("12"));
        assert_eq!(release.build_id.as_deref(), Some("20240617"));

        // Only a matching pair is stripped.
        assert_eq!(unquote_os_release_value("'12"), "'12");
        assert_eq!(unquote_os_release_value("12'"), "12'");
        assert_eq!(unquote_os_release_value("\""), "\"");
    }

    #[test]
    fn linux_release_parser_leaves_missing_fields_empty() {
        let release = parse_linux_os_release("NAME=\"Debian GNU/Linux\"\nID=debian\n");
        assert_eq!(release.version_id, None);
        assert_eq!(release.build_id, None);

        let empty = parse_linux_os_release("VERSION_ID=\"\"\nBUILD_ID=\"\"\n");
        assert_eq!(empty.version_id, None);
        assert_eq!(empty.build_id, None);
    }

    #[test]
    fn windows_programs_run_by_absolute_path_from_the_system_directory() {
        assert_eq!(
            windows_system_program(r"C:\Windows", WINDOWS_REG_PROGRAM),
            r"C:\Windows\System32\reg.exe"
        );
        assert_eq!(
            windows_system_program(r"D:\WinDir", WINDOWS_POWERSHELL_PROGRAM),
            r"D:\WinDir\System32\WindowsPowerShell\v1.0\powershell.exe"
        );
        assert_eq!(
            windows_system_root(None),
            r"C:\Windows",
            "a machine that names no system root still runs the system programs"
        );
        assert_eq!(windows_system_root(Some(r"E:\Win".to_owned())), r"E:\Win");
    }

    #[test]
    fn windows_registry_parser_reads_the_named_value() -> Result<()> {
        let value = parse_windows_reg_value(WINDOWS_REG_OUTPUT, "ProcessorNameString")?;
        assert_eq!(value, "Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz");
        Ok(())
    }

    #[test]
    fn windows_registry_parser_rejects_a_missing_value() {
        assert!(parse_windows_reg_value(WINDOWS_REG_OUTPUT, "SystemProductName").is_err());
        assert!(parse_windows_reg_value("", "ProcessorNameString").is_err());
        assert!(
            parse_windows_reg_value(
                "    ProcessorNameString    REG_SZ    \r\n",
                "ProcessorNameString"
            )
            .is_err()
        );
    }

    #[test]
    fn windows_registry_parser_ignores_a_longer_value_name() {
        let output = "    DisplayVersionExtra    REG_SZ    23H2\r\n";
        assert!(parse_windows_reg_value(output, "DisplayVersion").is_err());
    }

    #[test]
    fn windows_memory_parser_reads_the_byte_count() -> Result<()> {
        assert_eq!(
            parse_windows_memory_bytes("137438953472\r\n")?,
            137_438_953_472
        );
        assert_eq!(
            parse_windows_memory_bytes("  137438953472  ")?,
            137_438_953_472
        );
        Ok(())
    }

    #[test]
    fn windows_memory_parser_reads_the_number_of_a_noisy_query() -> Result<()> {
        let output = "WARNING: the query took longer than usual\r\n137438953472\r\n";
        assert_eq!(parse_windows_memory_bytes(output)?, 137_438_953_472);
        Ok(())
    }

    #[test]
    fn windows_memory_parser_rejects_an_empty_or_invalid_value() {
        assert!(parse_windows_memory_bytes("").is_err());
        assert!(parse_windows_memory_bytes("   \r\n").is_err());
        assert!(parse_windows_memory_bytes("137438953472 kB").is_err());
        assert!(parse_windows_memory_bytes("many").is_err());
    }

    #[test]
    fn linux_host_record_names_the_family_and_the_hardware() -> Result<()> {
        let host = linux_host_from_files(
            Some("OptiPlex 7090\n"),
            LINUX_CPUINFO_TEXT,
            LINUX_MEMINFO_TEXT,
            LINUX_OS_RELEASE_TEXT,
            "5.15.0-105-generic\n",
        )?;
        assert_eq!(host.machine_name, "Linux host");
        assert_eq!(host.machine_model, "OptiPlex 7090");
        assert_eq!(host.chip, "Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz");
        assert_eq!(host.memory_bytes, 65_792_436 * 1024);
        assert_eq!(host.os_version, "22.04");
        assert_eq!(host.os_build, "20240315");
        assert_eq!(host.total_cores, available_parallelism());
        assert_eq!(host.performance_cores, None);
        assert_eq!(host.efficiency_cores, None);
        assert_eq!(host.architecture, std::env::consts::ARCH);
        host.validate()?;
        Ok(())
    }

    #[test]
    fn linux_host_record_falls_back_to_the_kernel_and_the_unknown_marker() -> Result<()> {
        let host = linux_host_from_files(
            None,
            "processor\t: 0\n",
            LINUX_MEMINFO_TEXT,
            "NAME=\"Debian GNU/Linux\"\n",
            "6.1.0-18-amd64\n",
        )?;
        assert_eq!(host.machine_model, "unknown");
        assert_eq!(host.chip, "unknown");
        assert_eq!(host.os_version, "6.1.0-18-amd64");
        assert_eq!(host.os_build, "6.1.0-18-amd64");
        host.validate()?;

        let bare = linux_host_from_files(Some("  "), "", LINUX_MEMINFO_TEXT, "", "")?;
        assert_eq!(bare.machine_model, "unknown");
        assert_eq!(bare.os_version, "unknown");
        assert_eq!(bare.os_build, "unknown");
        bare.validate()?;
        Ok(())
    }

    #[test]
    fn linux_host_collection_reads_the_files_of_the_host() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let product_name = directory.path().join("product_name");
        let cpuinfo = directory.path().join("cpuinfo");
        let meminfo = directory.path().join("meminfo");
        let os_release = directory.path().join("os-release");
        let kernel = directory.path().join("osrelease");
        fs::write(&product_name, "OptiPlex 7090\n")?;
        fs::write(&cpuinfo, LINUX_CPUINFO_TEXT)?;
        fs::write(&meminfo, LINUX_MEMINFO_TEXT)?;
        fs::write(&os_release, LINUX_OS_RELEASE_TEXT)?;
        fs::write(&kernel, "5.15.0-105-generic\n")?;
        let host = linux_host_from_paths(
            product_name
                .to_str()
                .context("the fixture path is not UTF-8")?,
            cpuinfo.to_str().context("the fixture path is not UTF-8")?,
            meminfo.to_str().context("the fixture path is not UTF-8")?,
            os_release
                .to_str()
                .context("the fixture path is not UTF-8")?,
            kernel.to_str().context("the fixture path is not UTF-8")?,
        )?;
        assert_eq!(host.machine_name, LINUX_MACHINE_NAME);
        assert_eq!(host.machine_model, "OptiPlex 7090");
        assert_eq!(host.chip, "Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz");
        assert_eq!(host.memory_bytes, 65_792_436 * 1024);
        assert_eq!(host.os_version, "22.04");
        assert_eq!(host.os_build, "20240315");
        host.validate()?;
        Ok(())
    }

    #[test]
    fn linux_host_collection_falls_back_when_the_host_hides_a_file() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let meminfo = directory.path().join("meminfo");
        fs::write(&meminfo, LINUX_MEMINFO_TEXT)?;
        let missing = directory.path().join("missing");
        let host = linux_host_from_paths(
            missing.to_str().context("the fixture path is not UTF-8")?,
            missing.to_str().context("the fixture path is not UTF-8")?,
            meminfo.to_str().context("the fixture path is not UTF-8")?,
            missing.to_str().context("the fixture path is not UTF-8")?,
            missing.to_str().context("the fixture path is not UTF-8")?,
        )?;
        assert_eq!(host.machine_model, UNKNOWN_HOST_FIELD);
        assert_eq!(host.chip, UNKNOWN_HOST_FIELD);
        assert_eq!(host.os_version, UNKNOWN_HOST_FIELD);
        assert_eq!(host.os_build, UNKNOWN_HOST_FIELD);
        host.validate()?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_host_collection_reads_the_host_of_this_machine() -> Result<()> {
        let host = collect_linux_host()?;
        assert_eq!(host.machine_name, LINUX_MACHINE_NAME);
        assert_eq!(host.performance_cores, None);
        assert_eq!(host.efficiency_cores, None);
        assert!(host.memory_bytes > 0);
        assert_eq!(host.architecture, std::env::consts::ARCH);
        host.validate()?;
        Ok(())
    }

    #[test]
    fn linux_host_record_rejects_a_machine_without_memory() {
        assert!(linux_host_from_files(None, "", "MemFree: 1 kB\n", "", "").is_err());
    }

    #[test]
    fn windows_host_record_names_the_family_and_the_hardware() -> Result<()> {
        let host = windows_host_from_values(
            Some("Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz"),
            Some("PowerEdge R750"),
            Some("23H2"),
            Some("22631"),
            137_438_953_472,
        )?;
        assert_eq!(host.machine_name, "Windows host");
        assert_eq!(host.machine_model, "PowerEdge R750");
        assert_eq!(host.chip, "Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz");
        assert_eq!(host.memory_bytes, 137_438_953_472);
        assert_eq!(host.os_version, "23H2");
        assert_eq!(host.os_build, "22631");
        assert_eq!(host.total_cores, available_parallelism());
        assert_eq!(host.performance_cores, None);
        assert_eq!(host.efficiency_cores, None);
        host.validate()?;
        Ok(())
    }

    #[test]
    fn windows_host_record_marks_a_missing_display_version_unknown() -> Result<()> {
        // No fallback to CurrentVersion ("6.3" on Windows 10 and 11),
        // ReleaseId (frozen at "2009") or ProductName ("Windows 10" on
        // Windows 11): all of them lie about the release, so the record
        // carries the unknown marker instead.
        let host = windows_host_from_values(
            Some("Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz"),
            Some("OptiPlex 7090"),
            None,
            Some("22631"),
            137_438_953_472,
        )?;
        assert_eq!(host.os_version, "unknown");
        assert_eq!(host.os_build, "22631");
        host.validate()?;
        Ok(())
    }

    #[test]
    fn windows_host_record_falls_back_to_the_unknown_marker() -> Result<()> {
        let host = windows_host_from_values(None, None, None, None, 1)?;
        assert_eq!(host.machine_name, "Windows host");
        assert_eq!(host.machine_model, "unknown");
        assert_eq!(host.chip, "unknown");
        assert_eq!(host.os_version, "unknown");
        assert_eq!(host.os_build, "unknown");
        host.validate()?;
        Ok(())
    }

    #[test]
    fn task_hosts_name_the_operating_system_family_only() {
        assert_eq!(LINUX_MACHINE_NAME, "Linux host");
        assert_eq!(WINDOWS_MACHINE_NAME, "Windows host");
        for name in [LINUX_MACHINE_NAME, WINDOWS_MACHINE_NAME] {
            assert!(!name.contains("host name"));
            assert!(name.ends_with("host"));
        }
    }

    #[test]
    fn scoreboard_publication_still_refuses_every_host_that_is_not_macos() {
        let error = macos_host_refusal().expect_err("the macOS-only refusal must fail");
        assert!(
            error.to_string().contains("requires macOS"),
            "unexpected refusal: {error}"
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn scoreboard_verification_refuses_every_host_that_is_not_macos() {
        let error = collect_macos_host().expect_err("the macOS-only refusal must fail");
        assert!(
            error.to_string().contains("requires macOS"),
            "unexpected refusal: {error}"
        );
    }

    #[test]
    fn host_commands_run_under_a_deadline_and_are_reaped() -> Result<()> {
        let (program, args): (&str, &[&str]) = if cfg!(windows) {
            ("cmd", &["/c", "echo", "bench-host"])
        } else {
            ("sh", &["-c", "echo bench-host"])
        };
        let output = run_host_command(program, args, Duration::from_secs(10))?;
        assert_eq!(output.trim(), "bench-host");
        Ok(())
    }

    #[test]
    fn host_command_failures_report_the_standard_error() {
        let (program, args): (&str, &[&str]) = if cfg!(windows) {
            ("cmd", &["/c", "echo bench-host-failure 1>&2 & exit /b 3"])
        } else {
            ("sh", &["-c", "echo bench-host-failure >&2; exit 3"])
        };
        let error = run_host_command(program, args, Duration::from_secs(10))
            .expect_err("a failing command must fail");
        let message = error.to_string();
        assert!(
            message.contains("bench-host-failure"),
            "unexpected failure: {message}"
        );
    }

    #[test]
    fn with_reap_keeps_the_message_of_the_command_and_the_failed_reap() {
        // This guards the chain of an error that carries a cause: the
        // message of the command, the cause of that message and the failed
        // reap all stay in the chain. A reap that is attached to the
        // message of the command alone loses the cause, because anyhow
        // keeps only the message of a context.
        let command = anyhow::anyhow!("io failure").context("failed to wait for sh");
        let reap = Err(anyhow::anyhow!("the child was killed but did not exit"));
        let error = with_reap(command, reap);
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("failed to wait for sh"),
            "unexpected error: {rendered}"
        );
        assert!(
            rendered.contains("io failure"),
            "unexpected error: {rendered}"
        );
        assert!(
            rendered.contains("did not exit"),
            "unexpected error: {rendered}"
        );
        // The join of the readers of a host command follows the same rule,
        // so the chain of the join error keeps the cause of the command
        // error as well.
    }

    #[test]
    fn poll_child_reports_a_running_child_at_its_bound() -> Result<()> {
        // This guards the bounded poll of `poll_child`: a child that is
        // still running at the bound is reported as running, and the poll
        // ends at its bound. A bare wait cannot pass this test, because it
        // blocks until the child ends.
        let (program, args): (String, Vec<String>) = if cfg!(windows) {
            let powershell = windows_system_program(
                &windows_system_root(std::env::var("SystemRoot").ok()),
                WINDOWS_POWERSHELL_PROGRAM,
            );
            (
                powershell,
                vec![
                    "-NoProfile".to_owned(),
                    "-Command".to_owned(),
                    "Start-Sleep -Seconds 30".to_owned(),
                ],
            )
        } else {
            ("sleep".to_owned(), vec!["30".to_owned()])
        };
        let mut child = Command::new(&program)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to run {program}"))?;
        let start = Instant::now();
        let outcome = poll_child(&mut child, Duration::from_millis(100));
        let elapsed = start.elapsed();
        match outcome {
            Ok(Some(status)) => {
                let _ = kill_and_reap(&mut child, &program);
                anyhow::bail!("the child ended at once with {status}, so the poll did not wait");
            }
            Ok(None) => {}
            Err(error) => {
                let _ = kill_and_reap(&mut child, &program);
                return Err(error).with_context(|| format!("failed to poll {program}"));
            }
        }
        assert!(
            elapsed < Duration::from_secs(1),
            "the poll of a running child took {elapsed:?}, so it waited for the child"
        );
        // The child is still running, so the kill and the reap end it.
        kill_and_reap(&mut child, &program)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn host_commands_read_no_standard_input() -> Result<()> {
        let output = run_host_command("sh", &["-c", "cat"], Duration::from_secs(10))?;
        assert!(
            output.is_empty(),
            "the child read the standard input of the benchmark"
        );
        Ok(())
    }

    #[test]
    fn host_commands_that_outlive_their_deadline_fail() {
        let error = if cfg!(windows) {
            let powershell = windows_system_program(
                &windows_system_root(std::env::var("SystemRoot").ok()),
                WINDOWS_POWERSHELL_PROGRAM,
            );
            run_host_command(
                powershell.as_str(),
                &["-NoProfile", "-Command", "Start-Sleep -Seconds 30"],
                Duration::from_millis(200),
            )
        } else {
            // `sleep` runs directly: killing a shell that only spawned it
            // would orphan the sleeper.
            run_host_command("sleep", &["30"], Duration::from_millis(200))
        }
        .expect_err("a command that outlives its deadline must fail");
        assert!(
            error.to_string().contains("did not finish within"),
            "unexpected failure: {error}"
        );
    }

    /// The bound that a host command may not exceed once its deadline has
    /// passed: the reap poll and the reader join, plus slack for a slow
    /// machine.
    #[cfg(unix)]
    fn command_return_bound(deadline: Duration) -> Duration {
        deadline + KILL_REAP_WAIT + READER_JOIN_WAIT + Duration::from_secs(5)
    }

    #[cfg(unix)]
    #[test]
    fn a_child_that_outlives_the_deadline_is_killed_and_reaped() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let pid_file = directory.path().join("pid");
        let pid_path = pid_file.to_str().context("the fixture path is not UTF-8")?;
        // The shell records its own pid and then execs `sleep`, so the pid
        // names the sleeper itself and killing it orphans nothing.
        let script = format!("echo $$ > '{pid_path}'; exec sleep 30");
        // The deadline is long enough that the shell records its pid even on
        // a loaded machine, where a short deadline expires before the shell
        // runs at all.
        let deadline = Duration::from_secs(2);
        let start = Instant::now();
        let error = run_host_command("sh", &["-c", &script], deadline)
            .expect_err("a command that outlives its deadline must fail");
        assert!(
            error.to_string().contains("did not finish within"),
            "unexpected failure: {error}"
        );
        assert!(
            start.elapsed() < command_return_bound(deadline),
            "the command took {:?}, well past its bounds",
            start.elapsed()
        );
        let pid: i32 = fs::read_to_string(&pid_file)?
            .trim()
            .parse()
            .context("the child recorded no pid")?;
        // A zombie still answers signal 0, so this fails unless the child was
        // reaped rather than merely killed.
        let signalled = unsafe { libc::kill(pid, 0) };
        let reason = std::io::Error::last_os_error();
        assert!(
            signalled == -1 && reason.raw_os_error() == Some(libc::ESRCH),
            "the child {pid} survived the deadline: {reason}"
        );
        // This is a regression test that the deadline path kills and reaps:
        // a bare wait after a successful kill reaps as well, so no test can
        // catch that difference, which is why `kill_and_reap` is a bounded
        // poll by construction. The poll itself is guarded by
        // `poll_child_reports_a_running_child_at_its_bound`, which a bare
        // wait cannot pass.
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_grandchild_that_holds_the_pipe_fails_the_command() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let pid_file = directory.path().join("pid");
        let pid_path = pid_file.to_str().context("the fixture path is not UTF-8")?;
        // The guard stops the background child of the fixture, also when an
        // assert below fails, so the test never leaves a child behind.
        let mut guard = ProcessGuard::from_pid_file(pid_file.clone(), "sleep");
        // The helper exits at once, and its background child keeps the
        // standard output pipe open, which is the one case a joined reader
        // cannot wait out. The background child writes its own pid before
        // anything else and then becomes the sleeper, so the pid file names
        // the sleeper itself. The sleeper ends on its own after five
        // seconds, so a cleanup that missed it (a pid file that was never
        // written) cannot keep the machine busy past the end of the test.
        let script = "sh -c 'echo $$ > \"$1\"; exec sleep 5' fixture \"$1\" & exit 0";
        let deadline = Duration::from_secs(2);
        let start = Instant::now();
        let error = run_host_command("sh", &["-c", script, "sh", pid_path], deadline)
            .expect_err("a command whose pipe a grandchild holds must fail");
        // The pid is read right after the call and before any assert, so the
        // guard knows which process to stop.
        let pid = guard.pid().map_err(|error| {
            anyhow::anyhow!(
                "{error:#}: the fixture recorded no background child, so there is nothing to clean up"
            )
        })?;
        assert!(
            start.elapsed() < command_return_bound(deadline),
            "the command took {:?}, well past its bounds",
            start.elapsed()
        );
        assert!(
            error.to_string().contains("did not finish within"),
            "unexpected failure: {error}"
        );
        assert!(
            format!("{error:#}").contains("a grandchild holds the pipe open"),
            "unexpected failure: {error:#}"
        );
        // The identity of the process is checked before the kill, so the
        // kill cannot touch a pid that the machine handed to another process
        // in the meantime. The sleeper may have ended on its own already,
        // which the guard below accepts as well.
        let executable = unix_process_executable(pid);
        let name = executable
            .as_deref()
            .map(Path::new)
            .and_then(Path::file_name);
        assert!(
            name.is_none() || fixture_executable_matches("sleep", name),
            "pid {pid} is {executable:?}, not the helper of the fixture"
        );
        // The guard kills the background child and waits until it is gone.
        guard.kill_and_confirm()?;
        // This guards that a command whose pipe a grandchild holds fails
        // with an error that names the grandchild. It fails on the d538cb6
        // code: the readers were detached there, so the call returned at
        // its deadline while a reader kept the pipe of a process the
        // benchmark never spawned, and the error said nothing about it.
        Ok(())
    }

    #[test]
    fn kill_and_reap_succeeds_for_an_exited_and_a_running_child() -> Result<()> {
        // This guards the kill of a child that already exited (the child is
        // reaped before the kill, so the kill itself fails) and the kill of
        // a child that is still running (the same call reaps it).
        let (program, args): (&str, &[&str]) = if cfg!(windows) {
            ("cmd", &["/c", "exit 0"])
        } else {
            ("sh", &["-c", "exit 0"])
        };
        let mut exited = Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to run {program}"))?;
        // The exit is collected by a bounded poll, so the child is reaped
        // before the kill: killing a child that already exited must still
        // succeed. The poll has a bound, and every path that fails kills and
        // reaps the child, so the test never leaves a child behind.
        match poll_child(&mut exited, Duration::from_secs(5)) {
            Ok(Some(_)) => {}
            Ok(None) => {
                let _ = kill_and_reap(&mut exited, program);
                anyhow::bail!("the child did not exit within 5s");
            }
            Err(error) => {
                let _ = kill_and_reap(&mut exited, program);
                return Err(error).with_context(|| format!("failed to wait for {program}"));
            }
        }
        kill_and_reap(&mut exited, program)?;

        // A child that is still running when the kill arrives is reaped by
        // the same call.
        if cfg!(windows) {
            let powershell = windows_system_program(
                &windows_system_root(std::env::var("SystemRoot").ok()),
                WINDOWS_POWERSHELL_PROGRAM,
            );
            kill_and_reap_running_child(
                &powershell,
                &["-NoProfile", "-Command", "Start-Sleep -Seconds 30"],
            )
        } else {
            kill_and_reap_running_child("sleep", &["30"])
        }
    }

    /// The executable of one running process, or nothing once it is gone.
    #[cfg(unix)]
    fn unix_process_executable(pid: i32) -> Option<String> {
        #[cfg(target_os = "macos")]
        {
            let mut buffer = vec![0_u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
            let length =
                unsafe { libc::proc_pidpath(pid, buffer.as_mut_ptr().cast(), buffer.len() as u32) };
            let length = usize::try_from(length).unwrap_or(0);
            (length > 0 && length <= buffer.len())
                .then(|| String::from_utf8_lossy(&buffer[..length]).into_owned())
        }
        #[cfg(not(target_os = "macos"))]
        {
            fs::read_link(format!("/proc/{pid}/exe"))
                .ok()
                .map(|path| path.display().to_string())
        }
    }

    /// Kill and reap one child that is still running.
    fn kill_and_reap_running_child(program: &str, args: &[&str]) -> Result<()> {
        let mut running = Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to run {program}"))?;
        let outcome = kill_and_reap(&mut running, program);
        if outcome.is_err() {
            // A reap that failed must not leave the child running, so the
            // kill and the reap are tried once more.
            let _ = kill_and_reap(&mut running, program);
        }
        outcome
    }

    /// A process that a test must not leave running.
    ///
    /// The guard stops the process when it goes out of scope, so a failing
    /// assert cannot leave a child behind. The identity of the process is
    /// checked before the kill, so a pid that the machine handed to another
    /// process in the meantime is never killed.
    #[cfg(unix)]
    struct ProcessGuard {
        pid: Option<i32>,
        start_time: Option<ProcessStartTime>,
        confirmed: bool,
        pid_file: PathBuf,
        executable: &'static str,
    }

    #[cfg(unix)]
    impl ProcessGuard {
        /// A guard for a process whose pid a fixture recorded.
        ///
        /// The pid is read from the file when it is asked for, so a test
        /// that failed before it read the pid still stops the process.
        fn from_pid_file(pid_file: PathBuf, executable: &'static str) -> Self {
            Self {
                pid: None,
                start_time: None,
                confirmed: false,
                pid_file,
                executable,
            }
        }

        /// The pid of the guarded process.
        ///
        /// The start time of the process is read with the pid, so that the
        /// check of the identity can tell this process from another one that
        /// the machine handed the same pid to.
        fn pid(&mut self) -> Result<i32> {
            if let Some(pid) = self.pid {
                return Ok(pid);
            }
            let path = &self.pid_file;
            let recorded = fs::read_to_string(path)
                .with_context(|| format!("the helper recorded no pid in {}", path.display()))?;
            let pid = recorded.trim().parse::<i32>().with_context(|| {
                format!("the helper recorded no usable pid in {}", path.display())
            })?;
            self.start_time = process_start_time(pid);
            self.pid = Some(pid);
            Ok(pid)
        }

        /// Kill the guarded process and wait until it is gone.
        ///
        /// The check of the identity comes first, and the wait for the end
        /// of the process is a bounded poll of `kill(pid, 0)`, so the guard
        /// neither blocks without an end nor waits for a process that it may
        /// not kill. The poll is free of `waitpid`, because the process is
        /// not a child of the test.
        ///
        /// The identity of a process is the name of its executable and the
        /// time it started: both must still match, so a pid that the machine
        /// handed to another process in the meantime is never killed. A
        /// successful call confirms the end of the process, so the drop of
        /// the guard does not kill it a second time.
        fn kill_and_confirm(&mut self) -> Result<()> {
            if self.pid.is_none() {
                // The test may have failed before it read the pid, so the
                // guard reads it now: the process is stopped either way.
                let _ = self.pid();
            }
            let Some(pid) = self.pid else {
                // Nothing was recorded, so there is nothing to clean up.
                return Ok(());
            };
            if process_is_gone(pid) {
                self.confirmed = true;
                return Ok(());
            }
            let executable = unix_process_executable(pid);
            let Some(name) = executable
                .as_deref()
                .map(Path::new)
                .and_then(Path::file_name)
            else {
                // The process ended between the check above and this one,
                // so there is nothing left to kill.
                self.confirmed = true;
                return Ok(());
            };
            if !fixture_executable_matches(self.executable, Some(name)) {
                anyhow::bail!(
                    "pid {pid} is {executable:?}, not the {} of the fixture",
                    self.executable
                );
            }
            if let Some(started) = self.start_time {
                // The start time of the process must still be the one that
                // was read with the pid: a pid that the machine handed to
                // another process names a process that started later. A
                // start time that the machine hid when the pid was read
                // cannot be compared, so the name of the executable is then
                // the only identity.
                let current = process_start_time(pid);
                ensure!(
                    current == Some(started),
                    "pid {pid} started at {current:?}, not at {started:?}"
                );
            }
            if unsafe { libc::kill(pid, libc::SIGKILL) } != 0 {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ESRCH) {
                    return Err(anyhow::Error::new(error).context(format!(
                        "failed to kill the {} of the fixture",
                        self.executable
                    )));
                }
            }
            let limit = Instant::now() + KILL_REAP_WAIT;
            while Instant::now() < limit && !process_is_gone(pid) {
                thread::sleep(CHILD_POLL_INTERVAL);
            }
            ensure!(
                process_is_gone(pid),
                "the {} {pid} survived the kill",
                self.executable
            );
            self.confirmed = true;
            Ok(())
        }
    }

    #[cfg(unix)]
    impl Drop for ProcessGuard {
        fn drop(&mut self) {
            // A confirmed kill needs no second one: the process is gone and
            // its pid names nothing.
            if self.confirmed {
                return;
            }
            // A cleanup that failed must not hide the failure of the test,
            // but the process must not survive the test either.
            if let Err(error) = self.kill_and_confirm() {
                eprintln!("the fixture left a process behind: {error:#}");
            }
        }
    }

    /// The start time of one process, so that a pid can be checked against
    /// it.
    ///
    /// The start time of a process never changes, so a pid that the machine
    /// handed to another process names a process with another start time.
    /// The coarse part is the number of seconds on macOS and the number of
    /// clock ticks on Linux; the fine part is the number of microseconds on
    /// macOS and always zero on Linux.
    #[cfg(unix)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ProcessStartTime {
        coarse: u64,
        fine: u64,
    }

    /// The start time of one process, or nothing when the machine hides it.
    #[cfg(target_os = "macos")]
    fn process_start_time(pid: i32) -> Option<ProcessStartTime> {
        let mut info = std::mem::MaybeUninit::<libc::proc_bsdinfo>::uninit();
        let size = libc::c_int::try_from(std::mem::size_of::<libc::proc_bsdinfo>())
            .unwrap_or(libc::c_int::MAX);
        let written = unsafe {
            libc::proc_pidinfo(
                pid,
                libc::PROC_PIDTBSDINFO,
                0,
                info.as_mut_ptr().cast(),
                size,
            )
        };
        if written <= 0 {
            // The machine refused the request, so the start time is unknown.
            return None;
        }
        let info = unsafe { info.assume_init() };
        Some(ProcessStartTime {
            coarse: info.pbi_start_tvsec,
            fine: info.pbi_start_tvusec,
        })
    }

    /// The start time of one process, or nothing when the machine hides it.
    #[cfg(target_os = "linux")]
    fn process_start_time(pid: i32) -> Option<ProcessStartTime> {
        // The start time is the field 22 of the stat file, which follows the
        // name of the process and its closing parenthesis.
        let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        let rest = stat.rsplit_once(") ")?.1;
        let start = rest.split_whitespace().nth(19)?;
        Some(ProcessStartTime {
            coarse: start.parse().ok()?,
            fine: 0,
        })
    }

    /// The start time of one process, or nothing when the machine hides it.
    #[cfg(all(unix, not(any(target_os = "macos", target_os = "linux"))))]
    fn process_start_time(_pid: i32) -> Option<ProcessStartTime> {
        None
    }

    /// Whether the name of an executable is the one that the fixture started.
    ///
    /// A `sleep` that is a symlink to a busybox resolves to the name of the
    /// busybox, so that name is accepted for a `sleep` as well.
    #[cfg(unix)]
    fn fixture_executable_matches(expected: &str, name: Option<&OsStr>) -> bool {
        let Some(name) = name else {
            return false;
        };
        if name == OsStr::new(expected) {
            return true;
        }
        expected == "sleep" && name == OsStr::new("busybox")
    }

    /// Whether one process is gone, so that its pid names nothing.
    ///
    /// A zombie counts as gone: the process was stopped already, and only
    /// the reap of the environment is missing, which this test cannot do
    /// because the process is not a child of it. A process whose state the
    /// machine hides is not gone either: only a signal of zero that the
    /// machine refuses with `ESRCH` proves that the pid names nothing.
    #[cfg(unix)]
    fn process_is_gone(pid: i32) -> bool {
        if unsafe { libc::kill(pid, 0) } == 0 {
            return process_is_zombie(pid);
        }
        std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
    }

    /// Whether one process is a zombie, which means that it was stopped but
    /// that nobody reaped it.
    ///
    /// A zombie is not running, so the guard of a process accepts it: the
    /// process is not a child of the test, and only the machine can reap it.
    #[cfg(unix)]
    fn process_is_zombie(pid: i32) -> bool {
        #[cfg(target_os = "macos")]
        {
            // The state of a process is the status of its BSD info: a
            // zombie waits for its reap, so its status is `SZOMB`.
            let mut info = std::mem::MaybeUninit::<libc::proc_bsdinfo>::uninit();
            let size = libc::c_int::try_from(std::mem::size_of::<libc::proc_bsdinfo>())
                .unwrap_or(libc::c_int::MAX);
            let written = unsafe {
                libc::proc_pidinfo(
                    pid,
                    libc::PROC_PIDTBSDINFO,
                    0,
                    info.as_mut_ptr().cast(),
                    size,
                )
            };
            if written <= 0 {
                // The machine refused the request, so the state of the
                // process is unknown: a process that is not known to be a
                // zombie is not gone.
                return false;
            }
            let info = unsafe { info.assume_init() };
            info.pbi_status == libc::SZOMB
        }
        #[cfg(not(target_os = "macos"))]
        {
            // The state of the process follows its name, which ends with a
            // closing parenthesis.
            let stat = fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
            stat.rsplit_once(") ")
                .and_then(|(_, rest)| rest.split_whitespace().next())
                == Some("Z")
        }
    }
}
