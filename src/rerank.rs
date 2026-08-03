//! Thin yes/no reranking task built on a decoder's last-token logits.

use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

use crate::architecture::{CausalLm, CausalLmBatchInput};

/// One ordered reranking result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RerankScore {
    /// Raw `yes_logit - no_logit` margin.
    pub margin: f32,
    /// Stable sigmoid of [`Self::margin`].
    pub yes_probability: f32,
}

/// Score one non-empty ragged token batch using two single-token labels.
///
/// Prompt construction and truncation are model-profile concerns; this
/// function executes exactly the token IDs supplied by the caller.
pub fn rerank_token_batch<M>(
    model: &M,
    token_batches: &[Vec<u32>],
    no_token_id: u32,
    yes_token_id: u32,
) -> Result<Vec<RerankScore>>
where
    M: CausalLm + ?Sized,
{
    ensure!(!token_batches.is_empty(), "rerank batch cannot be empty");
    ensure!(
        no_token_id != yes_token_id,
        "rerank no and yes token IDs must be distinct"
    );
    for (name, token) in [("no", no_token_id), ("yes", yes_token_id)] {
        ensure!(
            (token as usize) < model.runtime_spec().vocab_size,
            "rerank {name} token ID is outside the model vocabulary"
        );
    }
    for (lane, tokens) in token_batches.iter().enumerate() {
        ensure!(!tokens.is_empty(), "rerank lane {lane} cannot be empty");
        ensure!(
            tokens.len() <= model.runtime_spec().max_context,
            "rerank lane {lane} exceeds the model context limit"
        );
        ensure!(
            tokens
                .iter()
                .all(|&token| (token as usize) < model.runtime_spec().vocab_size),
            "rerank lane {lane} contains a token ID outside the model vocabulary"
        );
    }
    let label_ids = Tensor::new(&[no_token_id, yes_token_id], model.device())?;

    let mut sessions = (0..token_batches.len())
        .map(|_| model.new_session())
        .collect::<candle_core::Result<Vec<_>>>()?;
    let mut inputs = sessions
        .iter_mut()
        .zip(token_batches)
        .map(|(session, tokens)| CausalLmBatchInput::new(session.as_mut(), tokens, 0))
        .collect::<Vec<_>>();
    let logits = match model.forward_batch_tokens(&mut inputs) {
        Ok(logits) => logits,
        Err(error) => return failed_submission(model, error.into()),
    };
    drop(inputs);

    let (lanes, vocabulary) = match logits.dims2() {
        Ok(shape) => shape,
        Err(error) => return failed_submission(model, error.into()),
    };
    if lanes != token_batches.len() || vocabulary != model.runtime_spec().vocab_size {
        return failed_submission(
            model,
            anyhow::anyhow!(
                "rerank model returned logits [{lanes}, {vocabulary}], expected [{}, {}]",
                token_batches.len(),
                model.runtime_spec().vocab_size
            ),
        );
    }
    let selected = match select_label_logits(&logits, &label_ids) {
        Ok(selected) => selected,
        Err(error) => return failed_submission(model, error.into()),
    };
    let selected = match model.read_output_to_cpu(&selected) {
        Ok(selected) => selected,
        Err(error) => return failed_submission(model, error.into()),
    };
    let selected = selected.to_dtype(DType::F32)?;
    let (lanes, labels) = selected.dims2()?;
    ensure!(
        lanes == token_batches.len(),
        "rerank model returned {lanes} rows for {} inputs",
        token_batches.len()
    );
    ensure!(
        labels == 2,
        "rerank model returned {labels} selected label columns instead of 2"
    );

    selected
        .to_vec2::<f32>()?
        .into_iter()
        .enumerate()
        .map(|(row_index, row)| score_row(&row, row_index))
        .collect()
}

fn select_label_logits(logits: &Tensor, label_ids: &Tensor) -> candle_core::Result<Tensor> {
    logits.index_select(label_ids, 1)
}

fn failed_submission<M, T>(model: &M, error: anyhow::Error) -> Result<T>
where
    M: CausalLm + ?Sized,
{
    match model.synchronize() {
        Ok(()) => Err(error),
        Err(cleanup_error) => Err(anyhow::anyhow!(
            "{error}; rerank cleanup synchronization also failed: {cleanup_error}"
        )),
    }
}

fn score_row(row: &[f32], row_index: usize) -> Result<RerankScore> {
    let [no_logit, yes_logit] = row else {
        anyhow::bail!("rerank row {row_index} does not contain exactly two label logits");
    };
    ensure!(
        no_logit.is_finite() && yes_logit.is_finite(),
        "rerank row {row_index} contains a non-finite label logit"
    );
    let margin = yes_logit - no_logit;
    ensure!(
        margin.is_finite(),
        "rerank row {row_index} produced a non-finite margin"
    );
    Ok(RerankScore {
        margin,
        yes_probability: stable_sigmoid(margin),
    })
}

fn stable_sigmoid(value: f32) -> f32 {
    let value = f64::from(value);
    if value >= 0.0 {
        (1.0 / (1.0 + (-value).exp())) as f32
    } else {
        let exponent = value.exp();
        (exponent / (1.0 + exponent)) as f32
    }
}

#[cfg(test)]
mod tests {
    use std::{any::Any, cell::Cell};

    use anyhow::Result;
    use candle_core::{Device, Tensor};

    use crate::architecture::{CausalLmSession, RuntimeSpec};

    use super::*;

    #[derive(Debug, Default)]
    struct FakeSession {
        cache_len: usize,
    }

    impl CausalLmSession for FakeSession {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn clear_cache(&mut self) {
            self.cache_len = 0;
        }

        fn cache_len(&self) -> usize {
            self.cache_len
        }
    }

    struct FakeModel {
        spec: RuntimeSpec,
        output: Tensor,
        device: Device,
        sessions: Cell<usize>,
        forwards: Cell<usize>,
        readbacks: Cell<usize>,
        synchronizations: Cell<usize>,
        readback_columns: Cell<Option<usize>>,
        fail_forward: bool,
        fail_readback: bool,
        fail_synchronize: bool,
    }

    impl FakeModel {
        fn new(output: Tensor) -> Self {
            Self {
                spec: RuntimeSpec {
                    vocab_size: 5,
                    max_context: 4,
                    stop_token_ids: vec![],
                },
                output,
                device: Device::Cpu,
                sessions: Cell::new(0),
                forwards: Cell::new(0),
                readbacks: Cell::new(0),
                synchronizations: Cell::new(0),
                readback_columns: Cell::new(None),
                fail_forward: false,
                fail_readback: false,
                fail_synchronize: false,
            }
        }
    }

    impl CausalLm for FakeModel {
        fn runtime_spec(&self) -> &RuntimeSpec {
            &self.spec
        }

        fn new_session(&self) -> candle_core::Result<Box<dyn CausalLmSession>> {
            self.sessions.set(self.sessions.get() + 1);
            Ok(Box::new(FakeSession::default()))
        }

        fn validate_session(&self, session: &dyn CausalLmSession) -> candle_core::Result<()> {
            if session.as_any().is::<FakeSession>() {
                Ok(())
            } else {
                candle_core::bail!("unexpected fake session type")
            }
        }

        fn forward(
            &self,
            _session: &mut dyn CausalLmSession,
            _input_ids: &Tensor,
            _start_pos: usize,
        ) -> candle_core::Result<Tensor> {
            candle_core::bail!("fake reranker exposes only batched logits")
        }

        fn forward_batch_tokens(
            &self,
            inputs: &mut [CausalLmBatchInput<'_>],
        ) -> candle_core::Result<Tensor> {
            self.forwards.set(self.forwards.get() + 1);
            if self.fail_forward {
                candle_core::bail!("injected rerank forward failure");
            }
            for input in inputs {
                let token_count = input.input_ids().len();
                let session = input
                    .session_mut()
                    .as_any_mut()
                    .downcast_mut::<FakeSession>()
                    .ok_or_else(|| {
                        candle_core::Error::Msg("unexpected fake session type".into())
                    })?;
                session.cache_len += token_count;
            }
            Ok(self.output.clone())
        }

        fn read_output_to_cpu(&self, output: &Tensor) -> candle_core::Result<Tensor> {
            self.readbacks.set(self.readbacks.get() + 1);
            self.readback_columns
                .set(output.dims2().ok().map(|(_, columns)| columns));
            if self.fail_readback {
                candle_core::bail!("injected rerank readback failure");
            }
            output.to_device(&Device::Cpu)
        }

        fn synchronize(&self) -> candle_core::Result<()> {
            self.synchronizations.set(self.synchronizations.get() + 1);
            if self.fail_synchronize {
                candle_core::bail!("injected rerank synchronization failure");
            }
            Ok(())
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    fn output(values: &[f32], shape: impl Into<candle_core::Shape>) -> Result<Tensor> {
        Ok(Tensor::from_slice(values, shape, &Device::Cpu)?)
    }

    #[test]
    fn selects_two_label_columns_before_one_readback_and_preserves_order() -> Result<()> {
        let model = FakeModel::new(output(
            &[
                9.0, 1.0, 8.0, 3.0, 7.0, // margin 2
                6.0, 4.0, 5.0, -1.0, 2.0, // margin -5
            ],
            (2, 5),
        )?);
        let tokens = vec![vec![1_u32, 2, 3], vec![4_u32]];

        let scores = rerank_token_batch(&model, &tokens, 1, 3)?;

        assert_eq!(scores.len(), 2);
        assert!((scores[0].margin - 2.0).abs() < f32::EPSILON);
        assert!((scores[0].yes_probability - 0.880_797).abs() < 1e-6);
        assert!((scores[1].margin + 5.0).abs() < f32::EPSILON);
        assert!((scores[1].yes_probability - 0.006_692_851).abs() < 1e-7);
        assert_eq!(model.sessions.get(), 2);
        assert_eq!(model.forwards.get(), 1);
        assert_eq!(model.readbacks.get(), 1);
        assert_eq!(model.readback_columns.get(), Some(2));
        assert_eq!(model.synchronizations.get(), 0);
        Ok(())
    }

    #[test]
    fn stable_sigmoid_handles_zero_and_extreme_finite_margins() -> Result<()> {
        let model = FakeModel::new(output(
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, // margin 0
                0.0, 0.0, 0.0, 1_000.0, 0.0, // margin +1000
                0.0, 0.0, 0.0, -1_000.0, 0.0, // margin -1000
            ],
            (3, 5),
        )?);
        let scores = rerank_token_batch(&model, &[vec![1_u32], vec![2_u32], vec![3_u32]], 1, 3)?;

        assert!((scores[0].yes_probability - 0.5).abs() < f32::EPSILON);
        assert!(scores[1].yes_probability >= 1.0 - f32::EPSILON);
        assert!(scores[2].yes_probability <= f32::EPSILON);
        assert!(scores.iter().all(|score| score.yes_probability.is_finite()));
        Ok(())
    }

    #[test]
    fn validates_every_input_and_label_before_creating_sessions() -> Result<()> {
        let cases = [
            (vec![], 1, 3, "batch cannot be empty"),
            (vec![vec![]], 1, 3, "lane 0 cannot be empty"),
            (vec![vec![1_u32; 5]], 1, 3, "context limit"),
            (vec![vec![5_u32]], 1, 3, "outside the model vocabulary"),
            (vec![vec![1_u32]], 5, 3, "no token ID"),
            (vec![vec![1_u32]], 1, 5, "yes token ID"),
            (vec![vec![1_u32]], 3, 3, "must be distinct"),
        ];
        for (tokens, no_token, yes_token, expected) in cases {
            let model = FakeModel::new(output(&[1.0; 5], (1, 5))?);
            let error = rerank_token_batch(&model, &tokens, no_token, yes_token)
                .expect_err("invalid rerank input must fail");
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(model.sessions.get(), 0);
            assert_eq!(model.forwards.get(), 0);
            assert_eq!(model.readbacks.get(), 0);
        }
        Ok(())
    }

    #[test]
    fn converts_label_logits_to_f32_before_scoring() -> Result<()> {
        for dtype in [DType::F16, DType::F64] {
            let logits = output(&[0.0, 3.0, 0.0, 4.0, 0.0], (1, 5))?.to_dtype(dtype)?;
            let model = FakeModel::new(logits);
            let scores = rerank_token_batch(&model, &[vec![1_u32]], 1, 3)?;
            assert!((scores[0].margin - 1.0).abs() < f32::EPSILON);
            assert!((scores[0].yes_probability - 0.731_058_6).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_outputs_with_the_correct_completion_cleanup() -> Result<()> {
        for (logits, tokens) in [
            (output(&[1.0, 2.0], 2)?, vec![vec![1_u32]]),
            (output(&[1.0; 5], (1, 5))?, vec![vec![1_u32], vec![2_u32]]),
            (output(&[1.0; 4], (1, 4))?, vec![vec![1_u32]]),
        ] {
            let model = FakeModel::new(logits);
            let error = rerank_token_batch(&model, &tokens, 1, 3)
                .expect_err("invalid full-logits shape must fail before readback");
            assert!(!error.to_string().is_empty());
            assert_eq!(model.readbacks.get(), 0);
            assert_eq!(model.synchronizations.get(), 1);
        }

        for invalid in [f32::NAN, f32::INFINITY] {
            let model = FakeModel::new(output(&[0.0, 1.0, 0.0, invalid, 0.0], (1, 5))?);
            let error = rerank_token_batch(&model, &[vec![1_u32]], 1, 3)
                .expect_err("non-finite selected logit must fail");
            assert!(error.to_string().contains("row 0"), "{error}");
            assert_eq!(model.readbacks.get(), 1);
            assert_eq!(model.synchronizations.get(), 0);
        }

        let model = FakeModel::new(output(&[0.0, -f32::MAX, 0.0, f32::MAX, 0.0], (1, 5))?);
        let error = rerank_token_batch(&model, &[vec![1_u32]], 1, 3)
            .expect_err("overflowing margin must fail");
        assert!(error.to_string().contains("non-finite margin"), "{error}");
        assert_eq!(model.readbacks.get(), 1);
        assert_eq!(model.synchronizations.get(), 0);
        Ok(())
    }

    #[test]
    fn submission_failures_synchronize_and_preserve_cleanup_errors() -> Result<()> {
        for readback_failure in [false, true] {
            let mut model = FakeModel::new(output(&[1.0; 5], (1, 5))?);
            model.fail_forward = !readback_failure;
            model.fail_readback = readback_failure;
            let error = rerank_token_batch(&model, &[vec![1_u32]], 1, 3)
                .expect_err("injected rerank failure must surface");
            assert!(error.to_string().contains("rerank"), "{error}");
            assert_eq!(model.synchronizations.get(), 1);
        }

        let mut model = FakeModel::new(output(&[1.0; 5], (1, 5))?);
        model.fail_forward = true;
        model.fail_synchronize = true;
        let error = rerank_token_batch(&model, &[vec![1_u32]], 1, 3)
            .expect_err("cleanup failure must be retained");
        assert!(error.to_string().contains("forward failure"), "{error}");
        assert!(
            error.to_string().contains("cleanup synchronization"),
            "{error}"
        );
        assert!(
            error.to_string().contains("synchronization failure"),
            "{error}"
        );
        Ok(())
    }
}
