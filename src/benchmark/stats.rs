//! Deterministic statistics for benchmark samples.

use std::time::Duration;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// Summary calculated from the retained raw samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionSummary {
    pub sample_count: usize,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

impl DistributionSummary {
    /// Reject non-finite, negative, unordered, or empty summaries.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.sample_count > 0,
            "summary sample count must be positive"
        );
        ensure!(
            [self.mean, self.p50, self.p95, self.max]
                .into_iter()
                .all(|value| value.is_finite() && value >= 0.0),
            "summary values must be finite and non-negative"
        );
        ensure!(self.p50 <= self.p95, "summary p50 exceeds p95");
        ensure!(self.p95 <= self.max, "summary p95 exceeds maximum");
        ensure!(self.mean <= self.max, "summary mean exceeds maximum");
        Ok(())
    }

    /// Recalculate and compare a summary with its raw evidence.
    pub fn verify(&self, samples: &[f64]) -> Result<()> {
        self.validate()?;
        let recalculated = summarize(samples)?;
        ensure!(
            self.sample_count == recalculated.sample_count,
            "summary sample count does not match raw evidence"
        );
        for (label, recorded, actual) in [
            ("mean", self.mean, recalculated.mean),
            ("p50", self.p50, recalculated.p50),
            ("p95", self.p95, recalculated.p95),
            ("max", self.max, recalculated.max),
        ] {
            ensure!(
                approximately_equal(recorded, actual),
                "summary {label} does not match raw evidence"
            );
        }
        Ok(())
    }
}

/// Calculate mean, nearest-rank p50/p95, and maximum.
pub fn summarize(samples: &[f64]) -> Result<DistributionSummary> {
    ensure!(
        !samples.is_empty(),
        "statistics require at least one sample"
    );
    ensure!(
        samples
            .iter()
            .all(|sample| sample.is_finite() && *sample >= 0.0),
        "statistics samples must be finite and non-negative"
    );
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let sum = sorted.iter().sum::<f64>();
    let mean = sum / sorted.len() as f64;
    let max = sorted
        .last()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("statistics require at least one sample"))?;
    Ok(DistributionSummary {
        sample_count: sorted.len(),
        mean,
        p50: nearest_rank_sorted(&sorted, 50)?,
        p95: nearest_rank_sorted(&sorted, 95)?,
        max,
    })
}

/// Return a nearest-rank percentile where `percent` is in `1..=100`.
pub fn nearest_rank(samples: &[f64], percent: u8) -> Result<f64> {
    ensure!(
        !samples.is_empty(),
        "percentile requires at least one sample"
    );
    ensure!(
        samples
            .iter()
            .all(|sample| sample.is_finite() && *sample >= 0.0),
        "percentile samples must be finite and non-negative"
    );
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    nearest_rank_sorted(&sorted, percent)
}

fn nearest_rank_sorted(sorted: &[f64], percent: u8) -> Result<f64> {
    ensure!(
        !sorted.is_empty(),
        "percentile requires at least one sample"
    );
    ensure!(
        (1..=100).contains(&percent),
        "percentile must be in 1..=100"
    );
    let numerator = sorted
        .len()
        .checked_mul(usize::from(percent))
        .ok_or_else(|| anyhow::anyhow!("percentile rank overflow"))?;
    let rank = numerator.div_ceil(100);
    Ok(sorted[rank - 1])
}

/// Visible output tokens divided by a shared wall-clock interval.
pub fn aggregate_throughput(total_tokens: u64, wall_time: Duration) -> Result<f64> {
    ensure!(
        total_tokens > 0,
        "aggregate throughput requires output tokens"
    );
    ensure!(
        !wall_time.is_zero(),
        "aggregate throughput requires positive wall time"
    );
    Ok(total_tokens as f64 / wall_time.as_secs_f64())
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= f64::EPSILON * scale * 8.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_float_eq(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= f64::EPSILON);
    }

    #[test]
    fn nearest_rank_uses_the_auditable_integer_definition() {
        let samples = (1..=20).map(f64::from).collect::<Vec<_>>();
        assert_float_eq(nearest_rank(&samples, 50).unwrap(), 10.0);
        assert_float_eq(nearest_rank(&samples, 95).unwrap(), 19.0);
        assert_float_eq(nearest_rank(&samples, 100).unwrap(), 20.0);
    }

    #[test]
    fn summary_retains_sample_count_and_expected_percentiles() {
        let summary = summarize(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(summary.sample_count, 4);
        assert_float_eq(summary.mean, 2.5);
        assert_float_eq(summary.p50, 2.0);
        assert_float_eq(summary.p95, 4.0);
        assert_float_eq(summary.max, 4.0);
        summary.verify(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    }

    #[test]
    fn invalid_statistics_fail_closed() {
        assert!(summarize(&[]).is_err());
        assert!(summarize(&[f64::NAN]).is_err());
        assert!(summarize(&[-1.0]).is_err());
        assert!(nearest_rank(&[1.0], 0).is_err());
        assert!(nearest_rank(&[1.0], 101).is_err());

        let invalid = DistributionSummary {
            sample_count: 1,
            mean: 2.0,
            p50: 1.0,
            p95: 1.0,
            max: 1.0,
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn aggregate_throughput_uses_one_shared_interval() {
        let throughput = aggregate_throughput(256, Duration::from_secs(2)).unwrap();
        assert_float_eq(throughput, 128.0);
        assert!(aggregate_throughput(0, Duration::from_secs(1)).is_err());
        assert!(aggregate_throughput(1, Duration::ZERO).is_err());
    }
}
