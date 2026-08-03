use std::{fs, path::Path};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

use crate::{
    architecture::{
        DecoderLayerSpec, DecoderSpec, ProjectionBias, RopeLayout, RopeSpec, RuntimeSpec,
    },
    chat::ChatMessage,
    decoder::DecoderModel,
    model_weights::WeightSource,
    models::{
        ModelAdapter, format_chatml_messages, load_stop_token_ids, resolve_default_rope_theta,
    },
};

const DEFAULT_SYSTEM: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
pub(crate) const QWEN2_5_CHAT_TEMPLATE: &str = concat!(
    "{%- if tools %}\n",
    "    {{- '<|im_start|>system\\n' }}\n",
    "    {%- if messages[0]['role'] == 'system' %}\n",
    "        {{- messages[0]['content'] }}\n",
    "    {%- else %}\n",
    "        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n",
    "    {%- endif %}\n",
    "    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n",
    "    {%- for tool in tools %}\n",
    "        {{- \"\\n\" }}\n",
    "        {{- tool | tojson }}\n",
    "    {%- endfor %}\n",
    "    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}}\\n</tool_call><|im_end|>\\n\" }}\n",
    "{%- else %}\n",
    "    {%- if messages[0]['role'] == 'system' %}\n",
    "        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n",
    "    {%- else %}\n",
    "        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n",
    "    {%- endif %}\n",
    "{%- endif %}\n",
    "{%- for message in messages %}\n",
    "    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n",
    "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n",
    "    {%- elif message.role == \"assistant\" %}\n",
    "        {{- '<|im_start|>' + message.role }}\n",
    "        {%- if message.content %}\n",
    "            {{- '\\n' + message.content }}\n",
    "        {%- endif %}\n",
    "        {%- for tool_call in message.tool_calls %}\n",
    "            {%- if tool_call.function is defined %}\n",
    "                {%- set tool_call = tool_call.function %}\n",
    "            {%- endif %}\n",
    "            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n",
    "            {{- tool_call.name }}\n",
    "            {{- '\", \"arguments\": ' }}\n",
    "            {{- tool_call.arguments | tojson }}\n",
    "            {{- '}\\n</tool_call>' }}\n",
    "        {%- endfor %}\n",
    "        {{- '<|im_end|>\\n' }}\n",
    "    {%- elif message.role == \"tool\" %}\n",
    "        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n",
    "            {{- '<|im_start|>user' }}\n",
    "        {%- endif %}\n",
    "        {{- '\\n<tool_response>\\n' }}\n",
    "        {{- message.content }}\n",
    "        {{- '\\n</tool_response>' }}\n",
    "        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n",
    "            {{- '<|im_end|>\\n' }}\n",
    "        {%- endif %}\n",
    "    {%- endif %}\n",
    "{%- endfor %}\n",
    "{%- if add_generation_prompt %}\n",
    "    {{- '<|im_start|>assistant\\n' }}\n",
    "{%- endif %}\n",
);

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

impl Qwen2Config {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        serde_json::from_slice(
            &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("invalid Qwen config at {}", path.display()))
    }

    pub fn to_spec(&self) -> Result<DecoderSpec> {
        self.to_spec_with_stop_tokens(self.eos_token_id.into_iter().collect())
    }

    fn to_spec_with_stop_tokens(&self, stop_token_ids: Vec<u32>) -> Result<DecoderSpec> {
        ensure!(self.model_type == "qwen2", "expected model_type=qwen2");
        ensure!(self.hidden_act == "silu", "only Qwen SiLU is supported");
        ensure!(
            self.rope_scaling.is_none(),
            "Qwen RoPE scaling is not supported yet"
        );
        if self.use_sliding_window {
            bail!("sliding-window Qwen checkpoints are not supported yet");
        }
        ensure!(self.hidden_size > 0, "hidden_size must be positive");
        ensure!(
            self.num_attention_heads > 0,
            "num_attention_heads must be positive"
        );
        ensure!(
            self.hidden_size.is_multiple_of(self.num_attention_heads),
            "hidden_size must be divisible by attention heads"
        );
        let head_dim = self.hidden_size / self.num_attention_heads;
        let rope = RopeSpec {
            theta: resolve_default_rope_theta(
                "Qwen2",
                self.rope_theta,
                self.rope_parameters.as_ref(),
            )?,
            layout: RopeLayout::HalfSplit,
            rotary_dim: head_dim,
            frequency_factors: None,
        };
        let spec = DecoderSpec {
            architecture: "qwen2".into(),
            runtime: RuntimeSpec {
                vocab_size: self.vocab_size,
                max_context: self.max_position_embeddings,
                stop_token_ids,
            },
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_layers: self.num_hidden_layers,
            num_query_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim,
            norm_epsilon: self.rms_norm_eps,
            layer_specs: vec![DecoderLayerSpec::standard(rope, head_dim); self.num_hidden_layers],
            embedding_scale: 1.0,
            final_logit_softcap: None,
            final_logit_scale: 1.0,
            projection_bias: ProjectionBias {
                query: true,
                key: true,
                value: true,
                output: false,
            },
            mlp_bias: false,
            tied_embeddings: self.tie_word_embeddings,
        };
        spec.validate()?;
        Ok(spec)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2Adapter {
    spec: DecoderSpec,
    chat_template: Option<Qwen2ChatTemplate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen2ChatTemplate {
    OfficialQwen2_5,
}

impl Qwen2Adapter {
    pub fn from_config(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config = Qwen2Config::from_path(path)?;
        let stop_token_ids = load_stop_token_ids(path, config.eos_token_id.into_iter().collect())?;
        Ok(Self {
            spec: config.to_spec_with_stop_tokens(stop_token_ids)?,
            // Preserve the established Safetensors adapter contract. Its
            // family-specific config has always selected the official ChatML
            // rendering used by the verified Qwen2.5 checkpoint.
            chat_template: Some(Qwen2ChatTemplate::OfficialQwen2_5),
        })
    }

    /// Construct a Qwen2 adapter from GGUF decoder metadata. Unknown or absent
    /// templates remain loadable for raw prompts but cannot silently select a
    /// possibly incompatible chat rendering.
    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "qwen2",
            "expected a qwen2 decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        let chat_template = (chat_template == Some(QWEN2_5_CHAT_TEMPLATE))
            .then_some(Qwen2ChatTemplate::OfficialQwen2_5);
        Ok(Self {
            spec,
            chat_template,
        })
    }
}

impl ModelAdapter for Qwen2Adapter {
    fn family(&self) -> &'static str {
        "qwen2"
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
        Ok(Box::new(DecoderModel::load(self.spec.clone(), weights)?))
    }

    fn format_chat_messages(&self, messages: &[ChatMessage]) -> Result<String> {
        match self.chat_template {
            Some(Qwen2ChatTemplate::OfficialQwen2_5) => {
                format_chatml_messages(DEFAULT_SYSTEM, messages)
            }
            None => bail!(
                "unsupported or missing Qwen2 chat template; pass --raw-prompt or use the pinned official Qwen2.5 template"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_official_half_billion_geometry() -> Result<()> {
        let config: Qwen2Config = serde_json::from_str(
            r#"{
              "model_type":"qwen2","vocab_size":151936,"hidden_size":896,
              "intermediate_size":4864,"num_hidden_layers":24,
              "num_attention_heads":14,"num_key_value_heads":2,
              "max_position_embeddings":32768,"tie_word_embeddings":true,
              "rope_theta":1000000.0,"rms_norm_eps":0.000001,
              "hidden_act":"silu","use_sliding_window":false,
              "eos_token_id":151645
            }"#,
        )?;
        let spec = config.to_spec_with_stop_tokens(vec![151_645, 151_643])?;
        assert_eq!(spec.head_dim, 64);
        assert_eq!(spec.kv_groups(), 7);
        assert_eq!(spec.kv_bytes_per_token(2)?, 12 * 1024);
        assert_eq!(spec.runtime.stop_token_ids, vec![151_645, 151_643]);
        Ok(())
    }

    #[test]
    fn chat_template_does_not_add_bos() -> Result<()> {
        let adapter = Qwen2Adapter {
            spec: Qwen2Config {
                model_type: "qwen2".into(),
                vocab_size: 16,
                hidden_size: 8,
                intermediate_size: 16,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                max_position_embeddings: 16,
                tie_word_embeddings: true,
                rope_theta: Some(10_000.0),
                rms_norm_eps: 1e-6,
                hidden_act: "silu".into(),
                rope_scaling: None,
                rope_parameters: None,
                use_sliding_window: false,
                eos_token_id: None,
            }
            .to_spec()?,
            chat_template: Some(Qwen2ChatTemplate::OfficialQwen2_5),
        };
        let prompt = adapter.format_chat(None, "你好")?;
        assert_eq!(
            prompt,
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        );
        Ok(())
    }

    #[test]
    fn renders_exact_multi_turn_qwen_chatml() -> Result<()> {
        let adapter = Qwen2Adapter {
            spec: Qwen2Config {
                model_type: "qwen2".into(),
                vocab_size: 16,
                hidden_size: 8,
                intermediate_size: 16,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                max_position_embeddings: 16,
                tie_word_embeddings: true,
                rope_theta: Some(10_000.0),
                rms_norm_eps: 1e-6,
                hidden_act: "silu".into(),
                rope_scaling: None,
                rope_parameters: None,
                use_sliding_window: false,
                eos_token_id: None,
            }
            .to_spec()?,
            chat_template: Some(Qwen2ChatTemplate::OfficialQwen2_5),
        };
        let prompt = adapter.format_chat_messages(&[
            ChatMessage::system("Keep whitespace."),
            ChatMessage::user(" First? "),
            ChatMessage::assistant(" First. "),
            ChatMessage::user(" Second? "),
        ])?;
        assert_eq!(
            prompt,
            "<|im_start|>system\nKeep whitespace.<|im_end|>\n<|im_start|>user\n First? <|im_end|>\n<|im_start|>assistant\n First. <|im_end|>\n<|im_start|>user\n Second? <|im_end|>\n<|im_start|>assistant\n"
        );
        Ok(())
    }

    #[test]
    fn maps_untied_qwen_heads_and_rejects_unknown_rope_scaling() -> Result<()> {
        let mut config: Qwen2Config = serde_json::from_str(
            r#"{
              "model_type":"qwen2","vocab_size":64,"hidden_size":8,
              "intermediate_size":16,"num_hidden_layers":1,
              "num_attention_heads":2,"num_key_value_heads":1,
              "max_position_embeddings":16,"tie_word_embeddings":false,
              "rope_theta":10000.0,"rms_norm_eps":0.000001,
              "hidden_act":"silu","use_sliding_window":false,
              "eos_token_id":2
            }"#,
        )?;
        assert!(!config.to_spec()?.tied_embeddings);
        config.rope_scaling = Some(serde_json::json!({"type": "yarn", "factor": 4.0}));
        assert!(config.to_spec().is_err());
        Ok(())
    }

    #[test]
    fn gguf_adapter_only_formats_the_pinned_official_template() -> Result<()> {
        let config: Qwen2Config = serde_json::from_str(
            r#"{
              "model_type":"qwen2","vocab_size":64,"hidden_size":8,
              "intermediate_size":16,"num_hidden_layers":1,
              "num_attention_heads":2,"num_key_value_heads":1,
              "max_position_embeddings":16,"tie_word_embeddings":true,
              "rope_theta":10000.0,"rms_norm_eps":0.000001,
              "hidden_act":"silu","use_sliding_window":false,
              "eos_token_id":2
            }"#,
        )?;
        let spec = config.to_spec()?;
        let official = Qwen2Adapter::from_decoder_spec(spec.clone(), Some(QWEN2_5_CHAT_TEMPLATE))?;
        assert!(official.format_chat(None, "Hello").is_ok());

        for template in [None, Some("{{ unknown template }}")] {
            let raw_only = Qwen2Adapter::from_decoder_spec(spec.clone(), template)?;
            assert!(raw_only.format_chat(None, "Hello").is_err());
        }
        let mut wrong_family = spec;
        wrong_family.architecture = "llama".into();
        assert!(Qwen2Adapter::from_decoder_spec(wrong_family, None).is_err());
        Ok(())
    }

    #[test]
    fn rejects_zero_attention_heads_without_panicking() -> Result<()> {
        let config: Qwen2Config = serde_json::from_str(
            r#"{
              "model_type":"qwen2","vocab_size":64,"hidden_size":0,
              "intermediate_size":16,"num_hidden_layers":1,
              "num_attention_heads":0,"num_key_value_heads":1,
              "max_position_embeddings":16,"tie_word_embeddings":true,
              "rope_theta":10000.0,"rms_norm_eps":0.000001,
              "hidden_act":"silu"
            }"#,
        )?;
        assert!(config.to_spec().is_err());
        Ok(())
    }

    #[test]
    fn adapter_reads_stop_tokens_from_generation_metadata() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let config_path = temp.path().join("config.json");
        fs::write(
            &config_path,
            r#"{
              "model_type":"qwen2","vocab_size":64,"hidden_size":8,
              "intermediate_size":16,"num_hidden_layers":1,
              "num_attention_heads":2,"num_key_value_heads":1,
              "max_position_embeddings":16,"tie_word_embeddings":true,
              "rope_theta":10000.0,"rms_norm_eps":0.000001,
              "hidden_act":"silu","eos_token_id":2
            }"#,
        )?;
        fs::write(
            temp.path().join("generation_config.json"),
            r#"{"eos_token_id":[3,2,3]}"#,
        )?;
        let adapter = Qwen2Adapter::from_config(config_path)?;
        assert_eq!(adapter.runtime_spec().stop_token_ids, vec![3, 2]);
        Ok(())
    }
}
