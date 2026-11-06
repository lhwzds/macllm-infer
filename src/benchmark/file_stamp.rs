//! The identity of one file, as the benchmark artifacts record it.

use std::fs;

/// The identity of one file: the device and the inode that hold it, and the
/// times at which it was last modified and last changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FileStamp {
    /// The device that holds the file.
    pub(crate) device: u64,
    /// The inode of the file within its device.
    pub(crate) inode: u64,
    /// The whole seconds of the last modification time.
    pub(crate) modified_seconds: i64,
    /// The nanoseconds of the last modification time.
    pub(crate) modified_nanoseconds: i64,
    /// The whole seconds of the last status-change time.
    pub(crate) changed_seconds: i64,
    /// The nanoseconds of the last status-change time.
    pub(crate) changed_nanoseconds: i64,
}

/// The number of 100-nanosecond ticks in one second.
#[cfg(not(unix))]
const TICKS_PER_SECOND: u64 = 10_000_000;

/// The number of nanoseconds in one 100-nanosecond tick.
#[cfg(not(unix))]
const NANOSECONDS_PER_TICK: u64 = 100;

/// Reads the identity of one file on a Unix host.
///
/// The device and the inode name the file itself, and the two time pairs
/// record when it was last modified and when its status last changed.
#[cfg(unix)]
pub(crate) fn file_stamp(metadata: &fs::Metadata) -> FileStamp {
    use std::os::unix::fs::MetadataExt;

    FileStamp {
        device: metadata.dev(),
        inode: metadata.ino(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    }
}

/// Reads the identity of one file on a Windows host.
///
/// This stamp leaves the device and the inode zero because the standard
/// library's Windows accessors for them are unstable, so this identity is
/// weaker than the Unix one: the stamp carries the last write time and the
/// creation time of the file.
///
/// Benchmarks do not get past the host gate to publish on these hosts today,
/// because `collect_macos_host` refuses every host that is not macOS. `demo`
/// still computes this stamp before that refusal.
#[cfg(not(unix))]
pub(crate) fn file_stamp(metadata: &fs::Metadata) -> FileStamp {
    use std::os::windows::fs::MetadataExt;

    let (modified_seconds, modified_nanoseconds) = split_ticks(metadata.last_write_time());
    let (changed_seconds, changed_nanoseconds) = split_ticks(metadata.creation_time());
    FileStamp {
        device: 0,
        inode: 0,
        modified_seconds,
        modified_nanoseconds,
        changed_seconds,
        changed_nanoseconds,
    }
}

/// Splits one count of 100-nanosecond ticks into whole seconds and the
/// nanoseconds that remain.
#[cfg(not(unix))]
fn split_ticks(ticks: u64) -> (i64, i64) {
    let seconds = i64::try_from(ticks / TICKS_PER_SECOND).unwrap_or(i64::MAX);
    let nanoseconds = i64::try_from((ticks % TICKS_PER_SECOND) * NANOSECONDS_PER_TICK).unwrap_or(0);
    (seconds, nanoseconds)
}
