//! Process-scoped memory sampling for reproducible scoreboard intervals.
//!
//! macOS exposes both current resident/physical-footprint values and a
//! lifetime physical-footprint maximum. Serving widths use sampled current
//! values so an earlier width cannot silently contaminate a later width's
//! peak. The lifetime maximum is retained only as diagnostic evidence.

use std::{
    sync::mpsc::{self, RecvTimeoutError, Sender},
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use serde::Serialize;

/// Fixed sampling interval used by both implementations.
pub const MEMORY_SAMPLING_PERIOD_MS: u64 = 10;

/// One instantaneous `proc_pid_rusage` observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessMemoryObservation {
    pub resident_bytes: Option<u64>,
    pub physical_footprint_bytes: Option<u64>,
    pub lifetime_peak_physical_footprint_bytes: Option<u64>,
}

/// Maximum of current-state observations retained over one sampling interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessMemoryPeak {
    pub resident_bytes: Option<u64>,
    pub physical_footprint_bytes: Option<u64>,
    pub lifetime_peak_physical_footprint_bytes: Option<u64>,
    pub sample_count: usize,
}

impl ProcessMemoryPeak {
    fn new(observation: &ProcessMemoryObservation) -> Self {
        Self {
            resident_bytes: observation.resident_bytes,
            physical_footprint_bytes: observation.physical_footprint_bytes,
            lifetime_peak_physical_footprint_bytes: observation
                .lifetime_peak_physical_footprint_bytes,
            sample_count: 1,
        }
    }

    fn observe(&mut self, observation: &ProcessMemoryObservation) {
        retain_max(&mut self.resident_bytes, observation.resident_bytes);
        retain_max(
            &mut self.physical_footprint_bytes,
            observation.physical_footprint_bytes,
        );
        retain_max(
            &mut self.lifetime_peak_physical_footprint_bytes,
            observation.lifetime_peak_physical_footprint_bytes,
        );
        self.sample_count = self.sample_count.saturating_add(1);
    }
}

fn retain_max(current: &mut Option<u64>, observed: Option<u64>) {
    if let Some(observed) = observed {
        *current = Some(current.map_or(observed, |current| current.max(observed)));
    }
}

/// Background sampler with an explicit stop-and-join boundary.
pub struct ProcessMemorySampler {
    stop: Option<Sender<()>>,
    worker: Option<JoinHandle<Result<ProcessMemoryPeak>>>,
}

impl ProcessMemorySampler {
    /// Sample once synchronously, then continue at the fixed scoreboard period.
    pub fn start(pid: u32) -> Result<Self> {
        let initial = process_memory(pid)?;
        let (stop, receiver) = mpsc::channel();
        let period = Duration::from_millis(MEMORY_SAMPLING_PERIOD_MS);
        let worker = thread::spawn(move || {
            let mut peak = ProcessMemoryPeak::new(&initial);
            loop {
                match receiver.recv_timeout(period) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                        peak.observe(&process_memory(pid)?);
                        return Ok(peak);
                    }
                    Err(RecvTimeoutError::Timeout) => peak.observe(&process_memory(pid)?),
                }
            }
        });
        Ok(Self {
            stop: Some(stop),
            worker: Some(worker),
        })
    }

    /// Stop immediately, retain one final sample, and join the worker.
    pub fn finish(mut self) -> Result<ProcessMemoryPeak> {
        if let Some(stop) = self.stop.take() {
            let _ignored = stop.send(());
        }
        self.worker
            .take()
            .context("memory sampler worker is absent")?
            .join()
            .map_err(|_| anyhow::anyhow!("memory sampler worker panicked"))?
    }
}

impl Drop for ProcessMemorySampler {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ignored = stop.send(());
        }
        if let Some(worker) = self.worker.take() {
            let _ignored = worker.join();
        }
    }
}

/// Read current and lifetime process memory counters for an arbitrary PID.
#[cfg(target_os = "macos")]
pub fn process_memory(pid: u32) -> Result<ProcessMemoryObservation> {
    use std::mem::MaybeUninit;

    let pid = i32::try_from(pid).context("process ID does not fit macOS pid_t")?;
    let mut usage = MaybeUninit::<libc::rusage_info_v4>::uninit();
    // SAFETY: `usage` is writable storage for the exact RUSAGE_INFO_V4
    // structure and the PID conversion above preserves the caller's PID.
    let status =
        unsafe { libc::proc_pid_rusage(pid, libc::RUSAGE_INFO_V4, usage.as_mut_ptr().cast()) };
    ensure!(
        status == 0,
        "proc_pid_rusage(RUSAGE_INFO_V4) failed for PID {pid}"
    );
    // SAFETY: a zero return means the kernel initialized the requested buffer.
    let usage = unsafe { usage.assume_init() };
    Ok(ProcessMemoryObservation {
        resident_bytes: Some(usage.ri_resident_size),
        physical_footprint_bytes: Some(usage.ri_phys_footprint),
        lifetime_peak_physical_footprint_bytes: Some(usage.ri_lifetime_max_phys_footprint),
    })
}

/// Non-macOS builds retain schema compatibility but cannot publish memory.
#[cfg(not(target_os = "macos"))]
pub fn process_memory(_pid: u32) -> Result<ProcessMemoryObservation> {
    Ok(ProcessMemoryObservation {
        resident_bytes: None,
        physical_footprint_bytes: None,
        lifetime_peak_physical_footprint_bytes: None,
    })
}

/// Convenience wrapper for the current scoreboard process.
pub fn current_process_memory() -> Result<ProcessMemoryObservation> {
    process_memory(std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_uses_current_maxima_without_relabeling_lifetime_peak() {
        let mut peak = ProcessMemoryPeak::new(&ProcessMemoryObservation {
            resident_bytes: Some(10),
            physical_footprint_bytes: Some(20),
            lifetime_peak_physical_footprint_bytes: Some(30),
        });
        peak.observe(&ProcessMemoryObservation {
            resident_bytes: Some(15),
            physical_footprint_bytes: Some(18),
            lifetime_peak_physical_footprint_bytes: Some(40),
        });
        assert_eq!(peak.resident_bytes, Some(15));
        assert_eq!(peak.physical_footprint_bytes, Some(20));
        assert_eq!(peak.lifetime_peak_physical_footprint_bytes, Some(40));
        assert_eq!(peak.sample_count, 2);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn sampler_reads_and_joins_the_current_process() -> Result<()> {
        let observation = current_process_memory()?;
        assert!(observation.resident_bytes.is_some());
        assert!(observation.physical_footprint_bytes.is_some());
        let peak = ProcessMemorySampler::start(std::process::id())?.finish()?;
        assert!(peak.sample_count >= 2);
        assert!(peak.resident_bytes.is_some());
        assert!(peak.physical_footprint_bytes.is_some());
        Ok(())
    }
}
