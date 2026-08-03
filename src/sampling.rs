use std::cmp::Ordering;

use anyhow::{Result, ensure};
use candle_core::Tensor;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Request-local pre-sampling constraint.
///
/// Implementations mask disallowed vocabulary entries before temperature and
/// top-k/top-p are applied, then advance their state only for the selected
/// token. This keeps structured output enforcement inside sampling rather
/// than validating an already-emitted string.
pub trait LogitsConstraint {
    fn mask_logits(&mut self, logits: &mut [f32]) -> Result<()>;
    fn commit_token(&mut self, token: u32) -> Result<()>;

    /// Validate a non-cancelled generation boundary.
    ///
    /// Stateful constraints use this to reject a token-budget or caller stop
    /// that would otherwise expose an incomplete structured value. Plain
    /// token filters may keep the default no-op implementation.
    fn finish(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            repetition_penalty: 1.1,
            seed: 42,
        }
    }
}

pub struct Sampler {
    config: SamplingConfig,
    rng: ChaCha8Rng,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Result<Self> {
        ensure!(
            config.temperature.is_finite() && config.temperature >= 0.0,
            "temperature must be finite and non-negative"
        );
        ensure!(
            config.top_p.is_finite() && config.top_p > 0.0 && config.top_p <= 1.0,
            "top_p must be finite and in (0, 1]"
        );
        ensure!(
            config.repetition_penalty.is_finite() && config.repetition_penalty > 0.0,
            "repetition_penalty must be finite and positive"
        );
        Ok(Self {
            rng: ChaCha8Rng::seed_from_u64(config.seed),
            config,
        })
    }

    pub fn sample(&mut self, logits: &Tensor, history: &[u32]) -> Result<u32> {
        self.sample_with_constraint(logits, history, None)
    }

    pub fn sample_with_constraint(
        &mut self,
        logits: &Tensor,
        history: &[u32],
        mut constraint: Option<&mut (dyn LogitsConstraint + 'static)>,
    ) -> Result<u32> {
        let (batch, vocab) = logits.dims2()?;
        ensure!(batch == 1, "sampler currently supports batch size 1");
        let mut values = logits
            .to_dtype(candle_core::DType::F32)?
            .to_vec2::<f32>()?
            .remove(0);
        ensure!(
            values.len() == vocab,
            "logit shape changed while reading values"
        );
        apply_repetition_penalty(&mut values, history, self.config.repetition_penalty);
        if let Some(constraint) = constraint.as_deref_mut() {
            constraint.mask_logits(&mut values)?;
        }
        ensure!(
            values.iter().any(|value| value.is_finite()),
            "sampling constraint rejected every finite token candidate"
        );

        let token = if self.config.temperature <= 0.0 {
            argmax(&values) as u32
        } else {
            for value in &mut values {
                *value /= self.config.temperature as f32;
            }

            let mut ranked = (0..vocab).collect::<Vec<_>>();
            ranked.sort_unstable_by(|&left, &right| total_desc(values[left], values[right]));
            if self.config.top_k > 0 && self.config.top_k < ranked.len() {
                for &index in &ranked[self.config.top_k..] {
                    values[index] = f32::NEG_INFINITY;
                }
                ranked.truncate(self.config.top_k);
            }

            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probabilities = values
                .iter()
                .map(|value| (*value - maximum).exp())
                .collect::<Vec<_>>();
            let sum = probabilities.iter().sum::<f32>();
            ensure!(
                sum.is_finite() && sum > 0.0,
                "sampling distribution is empty"
            );
            for probability in &mut probabilities {
                *probability /= sum;
            }

            ranked.sort_unstable_by(|&left, &right| {
                total_desc(probabilities[left], probabilities[right])
            });
            if (0.0..1.0).contains(&self.config.top_p) {
                let mut cumulative = 0.0_f32;
                let mut keep = 0_usize;
                for &index in &ranked {
                    cumulative += probabilities[index];
                    keep += 1;
                    if cumulative >= self.config.top_p as f32 {
                        break;
                    }
                }
                for &index in &ranked[keep..] {
                    probabilities[index] = 0.0;
                }
            }

            let total = probabilities.iter().sum::<f32>();
            let target = self.rng.random::<f32>() * total;
            sample_from_probabilities(&probabilities, target) as u32
        };
        if let Some(constraint) = constraint {
            constraint.commit_token(token)?;
        }
        Ok(token)
    }
}

fn sample_from_probabilities(probabilities: &[f32], target: f32) -> usize {
    let mut cumulative = 0.0_f32;
    for (index, probability) in probabilities.iter().enumerate() {
        if *probability <= 0.0 {
            continue;
        }
        cumulative += probability;
        if target < cumulative {
            return index;
        }
    }
    argmax(probabilities)
}

fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if penalty <= 0.0 || (penalty - 1.0).abs() < f32::EPSILON {
        return;
    }
    let mut seen = history.to_vec();
    seen.sort_unstable();
    seen.dedup();
    for token in seen {
        if let Some(logit) = logits.get_mut(token as usize) {
            *logit = if *logit < 0.0 {
                *logit * penalty
            } else {
                *logit / penalty
            };
        }
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index].total_cmp(&values[best]) == Ordering::Greater {
            best = index;
        }
    }
    best
}

fn total_desc(left: f32, right: f32) -> Ordering {
    right.total_cmp(&left)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    struct OnlyToken {
        allowed: u32,
        committed: Vec<u32>,
    }

    impl LogitsConstraint for OnlyToken {
        fn mask_logits(&mut self, logits: &mut [f32]) -> Result<()> {
            for (index, logit) in logits.iter_mut().enumerate() {
                if index != self.allowed as usize {
                    *logit = f32::NEG_INFINITY;
                }
            }
            Ok(())
        }

        fn commit_token(&mut self, token: u32) -> Result<()> {
            self.committed.push(token);
            Ok(())
        }
    }

    #[test]
    fn greedy_chooses_largest_logit() -> Result<()> {
        let logits = Tensor::new(&[[0.0_f32, 3.0, 2.0]], &Device::Cpu)?;
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        })?;
        assert_eq!(sampler.sample(&logits, &[])?, 1);
        Ok(())
    }

    #[test]
    fn constraint_masks_before_greedy_selection_and_commits_the_winner() -> Result<()> {
        let logits = Tensor::new(&[[9.0_f32, 3.0, 2.0]], &Device::Cpu)?;
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        })?;
        let mut constraint = OnlyToken {
            allowed: 2,
            committed: Vec::new(),
        };
        assert_eq!(
            sampler.sample_with_constraint(&logits, &[], Some(&mut constraint))?,
            2
        );
        assert_eq!(constraint.committed, [2]);
        Ok(())
    }

    #[test]
    fn penalty_changes_seen_positive_and_negative_logits() {
        let mut logits = vec![2.0, -2.0, 1.0];
        apply_repetition_penalty(&mut logits, &[0, 1, 0], 2.0);
        assert_eq!(logits, vec![1.0, -4.0, 1.0]);
    }

    #[test]
    fn greedy_ties_choose_the_first_token() {
        assert_eq!(argmax(&[3.0, 3.0, 2.0]), 0);
    }

    #[test]
    fn zero_target_never_selects_a_masked_token() {
        assert_eq!(sample_from_probabilities(&[0.0, 1.0, 0.0], 0.0), 1);
    }

    #[test]
    fn rejects_invalid_sampling_parameters() {
        for invalid in [
            SamplingConfig {
                temperature: -1.0,
                ..SamplingConfig::default()
            },
            SamplingConfig {
                top_p: 0.0,
                ..SamplingConfig::default()
            },
            SamplingConfig {
                repetition_penalty: 0.0,
                ..SamplingConfig::default()
            },
        ] {
            assert!(Sampler::new(invalid).is_err());
        }
    }
}
