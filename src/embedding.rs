//! Thin embedding task built on a decoder's final-normalized last hidden state.

use anyhow::{Result, ensure};
use candle_core::DType;

use crate::architecture::{CausalLm, CausalLmBatchInput};

/// Embed one non-empty ragged token batch and return unit-length `F32` rows in
/// submission order.
///
/// Token formatting, EOS insertion and truncation are model-profile concerns;
/// this function executes exactly the token IDs supplied by the caller.
pub fn embed_token_batch<M>(model: &M, token_batches: &[Vec<u32>]) -> Result<Vec<Vec<f32>>>
where
    M: CausalLm + ?Sized,
{
    ensure!(!token_batches.is_empty(), "embedding batch cannot be empty");
    for (lane, tokens) in token_batches.iter().enumerate() {
        ensure!(!tokens.is_empty(), "embedding lane {lane} cannot be empty");
        ensure!(
            tokens.len() <= model.runtime_spec().max_context,
            "embedding lane {lane} exceeds the model context limit"
        );
        ensure!(
            tokens
                .iter()
                .all(|&token| (token as usize) < model.runtime_spec().vocab_size),
            "embedding lane {lane} contains a token ID outside the model vocabulary"
        );
    }

    let mut sessions = (0..token_batches.len())
        .map(|_| model.new_session())
        .collect::<candle_core::Result<Vec<_>>>()?;
    let mut inputs = sessions
        .iter_mut()
        .zip(token_batches)
        .map(|(session, tokens)| CausalLmBatchInput::new(session.as_mut(), tokens, 0))
        .collect::<Vec<_>>();
    let hidden = match model.forward_batch_last_hidden_tokens(&mut inputs) {
        Ok(hidden) => hidden,
        Err(error) => return failed_submission(model, error.into()),
    };
    drop(inputs);

    let hidden = match model.read_output_to_cpu(&hidden) {
        Ok(hidden) => hidden,
        Err(error) => return failed_submission(model, error.into()),
    };
    let hidden = hidden.to_dtype(DType::F32)?;
    let (lanes, width) = hidden.dims2()?;
    ensure!(
        lanes == token_batches.len(),
        "embedding model returned {lanes} rows for {} inputs",
        token_batches.len()
    );
    ensure!(width > 0, "embedding model returned zero-width rows");

    let mut rows = hidden.to_vec2::<f32>()?;
    for (row_index, row) in rows.iter_mut().enumerate() {
        normalize_l2(row, row_index)?;
    }
    Ok(rows)
}

fn failed_submission<M, T>(model: &M, error: anyhow::Error) -> Result<T>
where
    M: CausalLm + ?Sized,
{
    match model.synchronize() {
        Ok(()) => Err(error),
        Err(cleanup_error) => Err(anyhow::anyhow!(
            "{error}; embedding cleanup synchronization also failed: {cleanup_error}"
        )),
    }
}

fn normalize_l2(row: &mut [f32], row_index: usize) -> Result<()> {
    let squared_norm = row.iter().try_fold(0.0_f64, |sum, &value| {
        ensure!(
            value.is_finite(),
            "embedding row {row_index} contains a non-finite value"
        );
        Ok::<_, anyhow::Error>(sum + f64::from(value).powi(2))
    })?;
    ensure!(
        squared_norm.is_finite() && squared_norm > 0.0,
        "embedding row {row_index} has a zero or non-finite norm"
    );
    let norm = squared_norm.sqrt();
    for value in row {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{any::Any, cell::Cell, rc::Rc};

    use anyhow::Result;
    use candle_core::{Device, Tensor};

    use crate::architecture::{CausalLmSession, RuntimeSpec};

    use super::*;

    #[derive(Debug)]
    struct FakeSession {
        owner: Rc<()>,
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
        owner: Rc<()>,
        spec: RuntimeSpec,
        output: Tensor,
        device: Device,
        sessions: Cell<usize>,
        forwards: Cell<usize>,
        readbacks: Cell<usize>,
        synchronizations: Cell<usize>,
        fail_forward: bool,
        fail_readback: bool,
        fail_synchronize: bool,
    }

    impl FakeModel {
        fn new(output: Tensor) -> Self {
            Self {
                owner: Rc::new(()),
                spec: RuntimeSpec {
                    vocab_size: 16,
                    max_context: 4,
                    stop_token_ids: vec![],
                },
                output,
                device: Device::Cpu,
                sessions: Cell::new(0),
                forwards: Cell::new(0),
                readbacks: Cell::new(0),
                synchronizations: Cell::new(0),
                fail_forward: false,
                fail_readback: false,
                fail_synchronize: false,
            }
        }

        fn session_mut<'a>(
            &self,
            session: &'a mut dyn CausalLmSession,
        ) -> candle_core::Result<&'a mut FakeSession> {
            let session = session
                .as_any_mut()
                .downcast_mut::<FakeSession>()
                .ok_or_else(|| candle_core::Error::Msg("unexpected fake session type".into()))?;
            if !Rc::ptr_eq(&self.owner, &session.owner) {
                candle_core::bail!("fake session belongs to another model");
            }
            Ok(session)
        }
    }

    impl CausalLm for FakeModel {
        fn runtime_spec(&self) -> &RuntimeSpec {
            &self.spec
        }

        fn new_session(&self) -> candle_core::Result<Box<dyn CausalLmSession>> {
            self.sessions.set(self.sessions.get() + 1);
            Ok(Box::new(FakeSession {
                owner: Rc::clone(&self.owner),
                cache_len: 0,
            }))
        }

        fn validate_session(&self, session: &dyn CausalLmSession) -> candle_core::Result<()> {
            let session = session
                .as_any()
                .downcast_ref::<FakeSession>()
                .ok_or_else(|| candle_core::Error::Msg("unexpected fake session type".into()))?;
            if !Rc::ptr_eq(&self.owner, &session.owner) {
                candle_core::bail!("fake session belongs to another model");
            }
            Ok(())
        }

        fn forward(
            &self,
            _session: &mut dyn CausalLmSession,
            _input_ids: &Tensor,
            _start_pos: usize,
        ) -> candle_core::Result<Tensor> {
            candle_core::bail!("fake embedding model exposes only batched last hidden")
        }

        fn forward_batch_last_hidden_tokens(
            &self,
            inputs: &mut [CausalLmBatchInput<'_>],
        ) -> candle_core::Result<Tensor> {
            self.forwards.set(self.forwards.get() + 1);
            if self.fail_forward {
                candle_core::bail!("injected embedding forward failure");
            }
            for input in inputs {
                let token_count = input.input_ids().len();
                self.session_mut(input.session_mut())?.cache_len += token_count;
            }
            Ok(self.output.clone())
        }

        fn read_output_to_cpu(&self, output: &Tensor) -> candle_core::Result<Tensor> {
            self.readbacks.set(self.readbacks.get() + 1);
            if self.fail_readback {
                candle_core::bail!("injected embedding readback failure");
            }
            output.to_device(&Device::Cpu)
        }

        fn synchronize(&self) -> candle_core::Result<()> {
            self.synchronizations.set(self.synchronizations.get() + 1);
            if self.fail_synchronize {
                candle_core::bail!("injected embedding synchronization failure");
            }
            Ok(())
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    fn f32_output(values: &[f32], shape: impl Into<candle_core::Shape>) -> Result<Tensor> {
        Ok(Tensor::from_slice(values, shape, &Device::Cpu)?)
    }

    #[test]
    fn embeds_one_ragged_batch_with_ordered_unit_rows_and_one_readback() -> Result<()> {
        let model = FakeModel::new(f32_output(&[3.0, 4.0, 0.0, -2.0], (2, 2))?);
        let tokens = vec![vec![1_u32, 2, 3], vec![4_u32]];

        let rows = embed_token_batch(&model, &tokens)?;

        assert_eq!(rows, vec![vec![0.6, 0.8], vec![0.0, -1.0]]);
        assert_eq!(model.sessions.get(), 2);
        assert_eq!(model.forwards.get(), 1);
        assert_eq!(model.readbacks.get(), 1);
        assert_eq!(model.synchronizations.get(), 0);
        Ok(())
    }

    #[test]
    fn validates_every_lane_before_creating_sessions() -> Result<()> {
        for (tokens, expected) in [
            (vec![], "batch cannot be empty"),
            (vec![vec![]], "lane 0 cannot be empty"),
            (vec![vec![1_u32; 5]], "context limit"),
            (vec![vec![16_u32]], "outside the model vocabulary"),
        ] {
            let model = FakeModel::new(f32_output(&[1.0], (1, 1))?);
            let error = embed_token_batch(&model, &tokens).expect_err("invalid input must fail");
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(model.sessions.get(), 0);
            assert_eq!(model.forwards.get(), 0);
            assert_eq!(model.readbacks.get(), 0);
        }
        Ok(())
    }

    #[test]
    fn converts_supported_hidden_dtypes_to_f32() -> Result<()> {
        for dtype in [DType::F16, DType::F64] {
            let output = f32_output(&[3.0, 4.0], (1, 2))?.to_dtype(dtype)?;
            let model = FakeModel::new(output);
            let rows = embed_token_batch(&model, &[vec![1_u32]])?;
            assert!((rows[0][0] - 0.6).abs() < 1e-3);
            assert!((rows[0][1] - 0.8).abs() < 1e-3);
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_output_shapes_after_the_completion_barrier() -> Result<()> {
        let cases = [
            f32_output(&[1.0, 2.0], 2)?,
            f32_output(&[1.0, 2.0], (1, 2))?,
            Tensor::zeros((2, 0), DType::F32, &Device::Cpu)?,
        ];
        for output in cases {
            let model = FakeModel::new(output);
            let error = embed_token_batch(&model, &[vec![1_u32], vec![2_u32]])
                .expect_err("invalid output shape must fail");
            assert!(!error.to_string().is_empty());
            assert_eq!(model.readbacks.get(), 1);
            assert_eq!(model.synchronizations.get(), 0);
        }
        Ok(())
    }

    #[test]
    fn rejects_zero_and_non_finite_rows_with_the_lane_index() -> Result<()> {
        for invalid in [0.0_f32, f32::NAN, f32::INFINITY] {
            let model = FakeModel::new(f32_output(&[1.0, 0.0, invalid, 0.0], (2, 2))?);
            let error = embed_token_batch(&model, &[vec![1_u32], vec![2_u32]])
                .expect_err("invalid embedding row must fail");
            assert!(error.to_string().contains("row 1"), "{error}");
            assert_eq!(model.readbacks.get(), 1);
        }
        Ok(())
    }

    #[test]
    fn submission_failures_synchronize_and_preserve_cleanup_errors() -> Result<()> {
        for readback_failure in [false, true] {
            let mut model = FakeModel::new(f32_output(&[1.0], (1, 1))?);
            model.fail_forward = !readback_failure;
            model.fail_readback = readback_failure;
            let error = embed_token_batch(&model, &[vec![1_u32]])
                .expect_err("injected submission failure must surface");
            assert!(error.to_string().contains("embedding"), "{error}");
            assert_eq!(model.synchronizations.get(), 1);
        }

        let mut model = FakeModel::new(f32_output(&[1.0], (1, 1))?);
        model.fail_forward = true;
        model.fail_synchronize = true;
        let error = embed_token_batch(&model, &[vec![1_u32]])
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
