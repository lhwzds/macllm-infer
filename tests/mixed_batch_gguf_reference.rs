//! Real GGUF/Metal parity gate for native packed ragged mixed-sequence calls.
//!
//! ```text
//! MACLLM_MIXED_BATCH_MODEL=models/smollm2-135m-instruct-q4_0/SmolLM2-135M-Instruct-q4_0.gguf \
//! cargo test --release --test mixed_batch_gguf_reference -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use anyhow::{Result, ensure};
use candle_core::Tensor;
use macllm_infer::{
    architecture::{CausalLmBatchCapability, CausalLmBatchInput},
    backend::DeviceKind,
    runtime::{LoadedRuntime, ModelLoadConfig},
};

const MAX_ABSOLUTE_DIFFERENCE: f32 = 0.05;
const MAX_RELATIVE_RMS: f64 = 0.005;

#[test]
#[ignore = "loads a real quantized GGUF model and requires unsandboxed Metal access"]
fn packed_ragged_metal_matches_serial_model_calls() -> Result<()> {
    let model =
        std::env::var_os("MACLLM_MIXED_BATCH_MODEL").map_or_else(default_model, PathBuf::from);
    ensure!(
        model.is_file(),
        "mixed-batch fixture is missing: {}",
        model.display()
    );
    let runtime = LoadedRuntime::load(&ModelLoadConfig::new(model, DeviceKind::Metal, None))?;
    ensure!(
        runtime.model.mixed_batch_capability() == CausalLmBatchCapability::PackedRagged,
        "loaded decoder does not advertise packed ragged batching"
    );
    let mut mixed_first = runtime.model.new_session()?;
    let mut mixed_second = runtime.model.new_session()?;
    let mut serial_first = runtime.model.new_session()?;
    let mut serial_second = runtime.model.new_session()?;

    let first_prefix = [1_u32, 100, 200];
    let second_prefix = [1_u32, 300];
    let mixed_prefix = {
        let mut inputs = [
            CausalLmBatchInput::new(mixed_first.as_mut(), &first_prefix, 0),
            CausalLmBatchInput::new(mixed_second.as_mut(), &second_prefix, 0),
        ];
        runtime.model.forward_batch_tokens(&mut inputs)?
    };
    runtime.model.device().synchronize()?;
    let serial_first_prefix =
        runtime
            .model
            .forward_tokens(serial_first.as_mut(), &first_prefix, 0)?;
    runtime.model.device().synchronize()?;
    let serial_second_prefix =
        runtime
            .model
            .forward_tokens(serial_second.as_mut(), &second_prefix, 0)?;
    runtime.model.device().synchronize()?;
    assert_row_close(&mixed_prefix, 0, &serial_first_prefix)?;
    assert_row_close(&mixed_prefix, 1, &serial_second_prefix)?;

    let first_suffix = [400_u32];
    let second_suffix = [500_u32, 600];
    let mixed_suffix = {
        let mut inputs = [
            CausalLmBatchInput::new(mixed_first.as_mut(), &first_suffix, first_prefix.len()),
            CausalLmBatchInput::new(mixed_second.as_mut(), &second_suffix, second_prefix.len()),
        ];
        runtime.model.forward_batch_tokens(&mut inputs)?
    };
    runtime.model.device().synchronize()?;
    let serial_first_suffix =
        runtime
            .model
            .forward_tokens(serial_first.as_mut(), &first_suffix, first_prefix.len())?;
    runtime.model.device().synchronize()?;
    let serial_second_suffix = runtime.model.forward_tokens(
        serial_second.as_mut(),
        &second_suffix,
        second_prefix.len(),
    )?;
    runtime.model.device().synchronize()?;
    assert_row_close(&mixed_suffix, 0, &serial_first_suffix)?;
    assert_row_close(&mixed_suffix, 1, &serial_second_suffix)?;
    assert_eq!(mixed_first.cache_len(), serial_first.cache_len());
    assert_eq!(mixed_second.cache_len(), serial_second.cache_len());
    Ok(())
}

fn assert_row_close(mixed: &Tensor, row: usize, expected: &Tensor) -> Result<()> {
    let actual = mixed.narrow(0, row, 1)?;
    ensure!(actual.dims2()? == expected.dims2()?);
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    ensure!(actual.iter().all(|value| value.is_finite()));
    ensure!(expected.iter().all(|value| value.is_finite()));
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut maximum = 0.0_f32;
    for (&actual, &expected) in actual.iter().zip(&expected) {
        let difference = actual - expected;
        maximum = maximum.max(difference.abs());
        squared_error += f64::from(difference).powi(2);
        squared_reference += f64::from(expected).powi(2);
    }
    let relative_rms = (squared_error / squared_reference.max(f64::EPSILON)).sqrt();
    ensure!(
        maximum <= MAX_ABSOLUTE_DIFFERENCE,
        "mixed row {row} maximum difference {maximum} exceeded {MAX_ABSOLUTE_DIFFERENCE}"
    );
    ensure!(
        relative_rms <= MAX_RELATIVE_RMS,
        "mixed row {row} relative RMS {relative_rms} exceeded {MAX_RELATIVE_RMS}"
    );
    ensure!(argmax(&actual) == argmax(&expected));
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}

fn default_model() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/smollm2-135m-instruct-q4_0/SmolLM2-135M-Instruct-q4_0.gguf")
}
