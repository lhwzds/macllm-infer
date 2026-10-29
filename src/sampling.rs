use std::cmp::Ordering;

use anyhow::{Result, bail, ensure};
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

    /// Whether [`Self::allows_token`] is implemented. The sampler then checks
    /// the few tokens it could select before paying for a full mask.
    fn checks_tokens(&self) -> bool {
        false
    }

    /// Whether [`Self::mask_logits`] would leave this token's finite logit
    /// selectable, without changing any state. Implementations must agree
    /// exactly with `mask_logits`, including failing when it would fail.
    fn allows_token(&self, _token: u32) -> Result<bool> {
        bail!("this sampling constraint only supports full logit masks")
    }
}

/// Candidates a constrained sampler checks, in descending logit order, before
/// it falls back to building the full vocabulary mask.
const CANDIDATE_CHECK_LIMIT: usize = 64;

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
            if constraint.checks_tokens()
                && let Some(token) = self.sample_checking_candidates(&values, constraint)?
            {
                constraint.commit_token(token)?;
                return Ok(token);
            }
            constraint.mask_logits(&mut values)?;
        }
        ensure!(
            values.iter().any(|value| value.is_finite()),
            "sampling constraint rejected every finite token candidate"
        );

        let token = if self.config.temperature <= 0.0 {
            argmax(&values) as u32
        } else {
            self.sample_stochastic(values)?
        };
        if let Some(constraint) = constraint {
            constraint.commit_token(token)?;
        }
        Ok(token)
    }

    /// Temperature, top-k and top-p sampling over a (possibly masked) row.
    fn sample_stochastic(&mut self, mut values: Vec<f32>) -> Result<u32> {
        let vocab = values.len();
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
        let (probabilities, _) = nucleus_probabilities(&values, &mut ranked, self.config.top_p)?;
        Ok(self.draw(&probabilities))
    }

    /// Draw one token from top-p-truncated probabilities. Every stochastic
    /// sample consumes exactly one random number here.
    fn draw(&mut self, probabilities: &[f32]) -> u32 {
        let total = probabilities.iter().sum::<f32>();
        let target = self.rng.random::<f32>() * total;
        sample_from_probabilities(probabilities, target) as u32
    }

    /// Sample under a constraint by checking only the tokens that could be
    /// selected, without building the full mask. `None` means the check was
    /// inconclusive and the caller must mask and sample as before; every
    /// inconclusive branch returns before drawing a random number, except
    /// accept-then-resample, whose second draw is part of its definition.
    ///
    /// Notation: `l` is the penalized logit row, `A` the allowed set, and
    /// the reference sampler masks `l` to `A` (disallowed logits become -inf)
    /// and then runs [`Self::sample_stochastic`] or `argmax`. Candidates are
    /// visited in the total order (value desc, index asc), so the first `n`
    /// candidates are exactly the `n` largest values with argmax's tie rule.
    ///
    /// * Greedy: the first allowed candidate has the largest allowed value
    ///   and, among equal values, the smallest index, which is exactly
    ///   `argmax` of the masked row. Bit-identical.
    /// * Top-k, all of `K_A` (the k largest allowed values) found: masking
    ///   then truncating leaves `K_A` finite and everything else -inf, so
    ///   rebuilding that row and running the shared tail
    ///   ([`nucleus_probabilities`] and [`Self::draw`]) performs the same
    ///   float operations in the same order and consumes the same single
    ///   random number. A tie at the k-th value, or a top-p cut between two
    ///   equally probable tokens, would make the reference depend on
    ///   `sort_unstable`'s tie order, so those cases fall back.
    ///   Bit-identical.
    /// * Top-p with one dominant allowed token: if even the largest possible
    ///   normalizer (known weights plus every unknown slot at the largest
    ///   unchecked weight) leaves that token above `max(top_p, 1/2)` with a
    ///   margin covering f32 rounding, the reference keeps only that token,
    ///   draws one random number and must return it. The fast path draws and
    ///   discards one number and returns it. Bit-identical.
    /// * No truncation (`top_k` off, `top_p = 1`): accept-then-resample.
    ///   Draw `t ~ p` from the unmasked row; accept if `t` is allowed, else
    ///   resample from the masked row. With `q = p|A / p(A)`,
    ///   `P(t) = p(t) + (1 - p(A)) q(t) = q(t)` for `t` in `A`: the masked
    ///   distribution exactly in real arithmetic (float softmax rounding is
    ///   ~1e-7 relative), but not token-identical per seed because the
    ///   random stream differs.
    /// * Top-p without top-k and no dominant token: no exact shortcut,
    ///   always falls back.
    fn sample_checking_candidates(
        &mut self,
        values: &[f32],
        constraint: &dyn LogitsConstraint,
    ) -> Result<Option<u32>> {
        // `mask_logits` leaves non-finite logits untouched; keep both paths
        // on the same candidate set and a total order.
        if values.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }
        let vocab = values.len();
        let token_id = |index: usize| u32::try_from(index).map_err(anyhow::Error::from);

        if self.config.temperature <= 0.0 {
            for index in descending_candidates(values, CANDIDATE_CHECK_LIMIT) {
                if constraint.allows_token(token_id(index)?)? {
                    return Ok(Some(token_id(index)?));
                }
            }
            return Ok(None);
        }

        let top_k =
            (self.config.top_k > 0 && self.config.top_k < vocab).then_some(self.config.top_k);
        let truncates_top_p = (0.0..1.0).contains(&self.config.top_p);
        if top_k.is_none() && !truncates_top_p {
            let token = self.sample_stochastic(values.to_vec())?;
            return Ok(constraint.allows_token(token)?.then_some(token));
        }

        let scaled = values
            .iter()
            .map(|value| *value / self.config.temperature as f32)
            .collect::<Vec<_>>();
        // One extra candidate bounds every unchecked value.
        let candidates = descending_candidates(&scaled, CANDIDATE_CHECK_LIMIT + 1);
        let checked = candidates.len().min(CANDIDATE_CHECK_LIMIT);
        let all_checked = checked == vocab;
        let mut allowed = Vec::new();
        for position in 0..checked {
            let index = candidates[position];
            if !constraint.allows_token(token_id(index)?)? {
                continue;
            }
            allowed.push(index);
            if top_k != Some(allowed.len()) {
                continue;
            }
            // `K_A` is complete unless an allowed token with the same value
            // could take the k-th place; an equal run that reaches the
            // unchecked candidates cannot be ruled out.
            let boundary = scaled[index];
            for (later_position, &later) in candidates.iter().enumerate().skip(position + 1) {
                if scaled[later].total_cmp(&boundary) != Ordering::Equal {
                    break;
                }
                if later_position >= checked || constraint.allows_token(token_id(later)?)? {
                    return Ok(None);
                }
            }
            return self.sample_known_top_set(&scaled, allowed);
        }
        if all_checked && !allowed.is_empty() {
            // Every allowed token is known: `K_A` is all of them.
            return self.sample_known_top_set(&scaled, allowed);
        }
        if !truncates_top_p || allowed.is_empty() {
            return Ok(None);
        }

        let top = allowed[0];
        let maximum = scaled[top];
        let weight = |index: usize| f64::from(scaled[index] - maximum).exp();
        let mut normalizer_bound = allowed.iter().map(|&index| weight(index)).sum::<f64>();
        let unchecked = vocab - checked;
        let unknown = top_k.map_or(unchecked, |k| (k - allowed.len()).min(unchecked));
        if unknown > 0 {
            normalizer_bound += unknown as f64 * weight(candidates[checked]);
        }
        // The reference sums at most `terms` f32 values; its result exceeds
        // the true normalizer by less than `terms` ulps, and exp/division
        // each add one more. Doubling that keeps the comparison conservative.
        let terms = top_k.unwrap_or(vocab) as f64;
        let margin = 2.0 * (terms + 2.0) * f64::from(f32::EPSILON);
        if 1.0 / normalizer_bound >= self.config.top_p.max(0.5) * (1.0 + margin) {
            let _ = self.rng.random::<f32>();
            return Ok(Some(token_id(top)?));
        }
        Ok(None)
    }

    /// Run the reference tail on the masked, top-k-truncated row whose finite
    /// entries are exactly `top_set`.
    fn sample_known_top_set(&mut self, scaled: &[f32], top_set: Vec<usize>) -> Result<Option<u32>> {
        let mut truncated = vec![f32::NEG_INFINITY; scaled.len()];
        for &index in &top_set {
            truncated[index] = scaled[index];
        }
        let mut ranked = top_set;
        let (probabilities, ambiguous_cut) =
            nucleus_probabilities(&truncated, &mut ranked, self.config.top_p)?;
        if ambiguous_cut {
            return Ok(None);
        }
        Ok(Some(self.draw(&probabilities)))
    }
}

/// The `limit` largest values in the total order (value desc, index asc).
fn descending_candidates(values: &[f32], limit: usize) -> Vec<usize> {
    let by_rank = |left: &usize, right: &usize| {
        values[*right]
            .total_cmp(&values[*left])
            .then(left.cmp(right))
    };
    let mut order = (0..values.len()).collect::<Vec<_>>();
    if limit < order.len() {
        order.select_nth_unstable_by(limit, by_rank);
        order.truncate(limit);
    }
    order.sort_unstable_by(by_rank);
    order
}

/// Softmax over top-k-truncated scaled logits, then top-p truncation along
/// `ranked`. Also reports whether the top-p cut separates two equally
/// probable tokens, where the survivor depends on the sort's tie order.
fn nucleus_probabilities(
    values: &[f32],
    ranked: &mut [usize],
    top_p: f64,
) -> Result<(Vec<f32>, bool)> {
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

    ranked.sort_unstable_by(|&left, &right| total_desc(probabilities[left], probabilities[right]));
    let mut ambiguous_cut = false;
    if (0.0..1.0).contains(&top_p) {
        let mut cumulative = 0.0_f32;
        let mut keep = 0_usize;
        for &index in ranked.iter() {
            cumulative += probabilities[index];
            keep += 1;
            if cumulative >= top_p as f32 {
                break;
            }
        }
        ambiguous_cut = keep < ranked.len()
            && probabilities[ranked[keep - 1]].total_cmp(&probabilities[ranked[keep]])
                == Ordering::Equal;
        for &index in &ranked[keep..] {
            probabilities[index] = 0.0;
        }
    }
    Ok((probabilities, ambiguous_cut))
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

    /// Allows a fixed token set. With `checks` false it supports only full
    /// masks, which is exactly the sampler before candidate checks existed.
    struct AllowedSet {
        allowed: Vec<bool>,
        checks: bool,
        masks: usize,
    }

    impl LogitsConstraint for AllowedSet {
        fn mask_logits(&mut self, logits: &mut [f32]) -> Result<()> {
            self.masks += 1;
            for (index, logit) in logits.iter_mut().enumerate() {
                if logit.is_finite() && !self.allowed[index] {
                    *logit = f32::NEG_INFINITY;
                }
            }
            Ok(())
        }

        fn commit_token(&mut self, token: u32) -> Result<()> {
            ensure!(self.allowed[token as usize], "committed a masked token");
            Ok(())
        }

        fn checks_tokens(&self) -> bool {
            self.checks
        }

        fn allows_token(&self, token: u32) -> Result<bool> {
            Ok(self.allowed[token as usize])
        }
    }

    const VOCAB: usize = 300;

    #[derive(Clone, Copy, Debug)]
    enum Row {
        Continuous,
        /// Logits on a coarse grid, so ties are frequent.
        Tied,
        /// One token far above the rest, like a structural grammar position.
        Peaked {
            top_allowed: bool,
        },
    }

    #[derive(Clone, Copy, Debug)]
    enum Allowed {
        Dense,
        Sparse,
    }

    fn step_input(rng: &mut ChaCha8Rng, row: Row, allowed: Allowed) -> (Vec<f32>, Vec<bool>) {
        let mut logits = (0..VOCAB)
            .map(|_| match row {
                Row::Tied => (rng.random::<f32>() * 16.0).floor() / 2.0 - 4.0,
                _ => rng.random::<f32>() * 8.0 - 4.0,
            })
            .collect::<Vec<_>>();
        let mut mask = match allowed {
            Allowed::Dense => (0..VOCAB).map(|_| rng.random::<f32>() < 0.9).collect(),
            Allowed::Sparse => {
                let mut mask = vec![false; VOCAB];
                for _ in 0..3 {
                    mask[rng.random_range(0..VOCAB)] = true;
                }
                mask
            }
        };
        if let Row::Peaked { top_allowed } = row {
            let top = rng.random_range(0..VOCAB);
            logits[top] = 14.0;
            mask[top] = top_allowed;
            if !mask.iter().any(|allowed| *allowed) {
                mask[(top + 1) % VOCAB] = true;
            }
        }
        (logits, mask)
    }

    /// Run the same constrained steps through the candidate-check path and
    /// the full-mask path; return both token sequences and mask counts.
    fn run_both(
        config: SamplingConfig,
        row: Row,
        allowed: Allowed,
        seed: u64,
    ) -> Result<[(Vec<u32>, usize); 2]> {
        let mut results = Vec::new();
        for checks in [true, false] {
            let mut inputs = ChaCha8Rng::seed_from_u64(seed ^ 0x5eed);
            let mut sampler = Sampler::new(SamplingConfig { seed, ..config })?;
            let mut tokens = Vec::new();
            let mut masks = 0;
            for _ in 0..6 {
                let (logits, mask) = step_input(&mut inputs, row, allowed);
                let mut constraint = AllowedSet {
                    allowed: mask,
                    checks,
                    masks: 0,
                };
                let logits = Tensor::from_vec(logits, (1, VOCAB), &Device::Cpu)?;
                tokens.push(sampler.sample_with_constraint(
                    &logits,
                    &tokens,
                    Some(&mut constraint),
                )?);
                masks += constraint.masks;
            }
            results.push((tokens, masks));
        }
        let fast = results.remove(0);
        Ok([fast, results.remove(0)])
    }

    #[test]
    fn candidate_checks_match_the_full_mask_token_for_token() -> Result<()> {
        let settings = [
            (0.0, 20, 0.8),
            (0.7, 20, 0.8),
            (0.6, 20, 0.95),
            (1.0, 1, 1.0),
            (1.0, 5, 1.0),
            (1.3, 100, 0.5),
            (0.7, 0, 0.8),
            (0.9, 0, 0.3),
        ];
        let rows = [
            Row::Continuous,
            Row::Tied,
            Row::Peaked { top_allowed: true },
            Row::Peaked { top_allowed: false },
        ];
        for (temperature, top_k, top_p) in settings {
            let config = SamplingConfig {
                temperature,
                top_k,
                top_p,
                repetition_penalty: 1.1,
                seed: 0,
            };
            let (mut fast_masks, mut full_masks) = (0, 0);
            for row in rows {
                for allowed in [Allowed::Dense, Allowed::Sparse] {
                    for seed in 0..40 {
                        let [fast, full] = run_both(config, row, allowed, seed)?;
                        assert_eq!(
                            fast.0, full.0,
                            "T={temperature} k={top_k} p={top_p} {row:?} {allowed:?} seed={seed}"
                        );
                        fast_masks += fast.1;
                        full_masks += full.1;
                    }
                }
            }
            // Every setting must actually take the fast path for part of the
            // matrix, or this test would compare the full mask with itself.
            // Tied rows and sparse allowed sets mostly fall back by design.
            assert!(
                fast_masks < full_masks,
                "T={temperature} k={top_k} p={top_p}: {fast_masks} of {full_masks} steps masked"
            );
        }
        Ok(())
    }

    #[test]
    fn candidate_checks_take_the_fast_path_where_it_is_exact() -> Result<()> {
        let qwen = SamplingConfig {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            repetition_penalty: 1.0,
            seed: 0,
        };
        let greedy = SamplingConfig {
            temperature: 0.0,
            ..qwen
        };
        // Greedy, and top-k with enough allowed candidates, never mask.
        for config in [greedy, qwen] {
            assert_eq!(
                run_both(config, Row::Continuous, Allowed::Dense, 3)?[0].1,
                0
            );
        }
        // A dominant allowed token decides top-p without knowing `K_A`.
        assert_eq!(
            run_both(qwen, Row::Peaked { top_allowed: true }, Allowed::Sparse, 3)?[0].1,
            0
        );
        // Few allowed tokens and no dominant one: `K_A` is unknowable cheaply.
        let no_top_p = SamplingConfig { top_p: 1.0, ..qwen };
        assert_eq!(
            run_both(no_top_p, Row::Continuous, Allowed::Sparse, 3)?[0].1,
            6
        );
        Ok(())
    }

    #[test]
    fn accept_then_resample_matches_the_masked_distribution() -> Result<()> {
        // Without truncation the fast path is exact in distribution but uses
        // the random stream differently, so compare frequencies over many
        // fixed draws against the exact masked softmax.
        let logits = [1.0_f32, 2.5, -1.0, 0.5, 2.0, 1.5, -0.5, 3.0];
        let allowed = [false, true, false, true, true, false, true, false];
        let temperature = 0.9;
        let weights = logits
            .iter()
            .zip(allowed)
            .map(|(logit, allowed)| {
                if allowed {
                    (f64::from(*logit) / temperature).exp()
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let normalizer = weights.iter().sum::<f64>();
        let draws = 40_000_u32;
        for checks in [true, false] {
            let mut sampler = Sampler::new(SamplingConfig {
                temperature,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
                seed: 11,
            })?;
            let mut counts = [0_usize; 8];
            let mut masks = 0;
            let row = Tensor::new(&[logits], &Device::Cpu)?;
            for _ in 0..draws {
                let mut constraint = AllowedSet {
                    allowed: allowed.to_vec(),
                    checks,
                    masks: 0,
                };
                counts
                    [sampler.sample_with_constraint(&row, &[], Some(&mut constraint))? as usize] +=
                    1;
                masks += constraint.masks;
            }
            for (token, count) in counts.iter().enumerate() {
                let expected = weights[token] / normalizer;
                let observed = *count as f64 / f64::from(draws);
                assert!(
                    (observed - expected).abs() < 0.01,
                    "checks={checks} token {token}: observed {observed}, expected {expected}"
                );
            }
            if checks {
                // Only a rejected first draw pays for a mask: 1 - p(A) of them.
                let unmasked = logits
                    .iter()
                    .map(|logit| (f64::from(*logit) / temperature).exp())
                    .sum::<f64>();
                let rejected = 1.0 - normalizer / unmasked;
                let observed = masks as f64 / f64::from(draws);
                assert!(
                    (observed - rejected).abs() < 0.01,
                    "{observed} vs {rejected}"
                );
            } else {
                assert_eq!(masks, usize::try_from(draws)?);
            }
        }
        Ok(())
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
