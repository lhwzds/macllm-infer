use anyhow::{Result, bail, ensure};

use crate::{
    architecture::{DecoderSpec, RuntimeSpec},
    chat::ChatMessage,
    decoder::DecoderModel,
    model_weights::WeightSource,
    models::ModelAdapter,
};

// The shared decoder and the adapters read the same weight-name table, so it
// lives in the architecture module and is re-exported here.
pub(crate) use crate::architecture::FALCON_H1_WEIGHT_NAMES;

/// GGUF-backed Falcon-H1 adapter.
///
/// Falcon-H1 geometry and hybrid attention/Mamba2 scheduling are lowered into
/// a validated [`DecoderSpec`] by the GGUF loader. Chat formatting remains
/// deliberately disabled until an exact embedded template is supported; raw
/// prompts continue to use the same tokenizer and runtime path.
#[derive(Debug, Clone)]
pub struct FalconH1Adapter {
    spec: DecoderSpec,
}

impl FalconH1Adapter {
    /// Construct an adapter from metadata and tensors already validated by the
    /// GGUF loader.
    pub fn from_decoder_spec(spec: DecoderSpec, _chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "falcon-h1",
            "expected a falcon-h1 decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        Ok(Self { spec })
    }
}

impl ModelAdapter for FalconH1Adapter {
    fn family(&self) -> &'static str {
        "falcon-h1"
    }

    fn runtime_spec(&self) -> &RuntimeSpec {
        &self.spec.runtime
    }

    fn decoder_spec(&self) -> Option<&DecoderSpec> {
        Some(&self.spec)
    }

    fn build_model(
        &self,
        weights: &dyn WeightSource,
    ) -> candle_core::Result<Box<dyn crate::architecture::CausalLm>> {
        Ok(Box::new(DecoderModel::load_with_weight_names(
            self.spec.clone(),
            weights,
            &FALCON_H1_WEIGHT_NAMES,
        )?))
    }

    fn format_chat_messages(&self, _messages: &[ChatMessage]) -> Result<String> {
        bail!("Falcon-H1 chat templates are not supported yet; use `run --raw-prompt`")
    }
}
