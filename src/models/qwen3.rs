use std::{fs, path::Path};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

use crate::{
    agent::{AgentPrompt, AgentTemplateProfile},
    architecture::{
        DecoderLayerSpec, DecoderSpec, ProjectionBias, RopeLayout, RopeSpec, RuntimeSpec,
    },
    chat::{ChatMessage, ChatTemplateOptions},
    decoder::DecoderModel,
    model_weights::WeightSource,
    models::{
        GenericPaths, GgufPromptTokens, JinjaChatFallback, ModelAdapter, jinja_fallback,
        load_stop_token_ids, resolve_default_rope_theta,
    },
};

// The prompt renderers live in the chat group so that the serving and
// benchmark layers can use them without depending on the model adapters.
pub use crate::chat_templates::qwen3::{
    CURRENT_QWEN3_CHAT_TEMPLATE_LEN, CURRENT_QWEN3_CHAT_TEMPLATE_SHA256,
    GGUF_QWEN3_CHAT_TEMPLATE_LEN, GGUF_QWEN3_CHAT_TEMPLATE_SHA256, Qwen3ChatTemplate,
    classify_chat_template, classify_chat_template_identity, format_qwen3_agent_prompt,
    format_qwen3_chat_messages, python_json, serialize_native_tool,
};

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

impl Qwen3Config {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        serde_json::from_slice(
            &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("invalid Qwen3 config at {}", path.display()))
    }

    pub fn to_spec(&self) -> Result<DecoderSpec> {
        self.to_spec_with_stop_tokens(self.eos_token_id.into_iter().collect())
    }

    fn to_spec_with_stop_tokens(&self, stop_token_ids: Vec<u32>) -> Result<DecoderSpec> {
        ensure!(self.model_type == "qwen3", "expected model_type=qwen3");
        ensure!(self.hidden_act == "silu", "only Qwen3 SiLU is supported");
        ensure!(
            self.rope_scaling.is_none(),
            "Qwen3 RoPE scaling is not supported yet"
        );
        ensure!(
            !self.use_sliding_window && self.sliding_window.is_none(),
            "sliding-window Qwen3 checkpoints are not supported yet"
        );
        ensure!(
            !self.attention_bias,
            "Qwen3 attention projection bias is not supported yet"
        );
        ensure!(self.head_dim > 0, "Qwen3 head_dim must be positive");

        let rope = RopeSpec {
            theta: resolve_default_rope_theta(
                "Qwen3",
                self.rope_theta,
                self.rope_parameters.as_ref(),
            )?,
            layout: RopeLayout::HalfSplit,
            rotary_dim: self.head_dim,
            frequency_factors: None,
        };
        let mut layer = DecoderLayerSpec::standard(rope, self.head_dim);
        layer
            .attention_mut()
            .expect("standard decoder layer uses attention")
            .qk_norm = true;
        let spec = DecoderSpec {
            architecture: "qwen3".into(),
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
            head_dim: self.head_dim,
            norm_epsilon: self.rms_norm_eps,
            layer_specs: vec![layer; self.num_hidden_layers],
            embedding_scale: 1.0,
            final_logit_softcap: None,
            final_logit_scale: 1.0,
            projection_bias: ProjectionBias {
                query: false,
                key: false,
                value: false,
                output: false,
            },
            mlp_bias: false,
            tied_embeddings: self.tie_word_embeddings,
        };
        spec.validate()?;
        Ok(spec)
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerMetadata {
    #[serde(default)]
    chat_template: Option<serde_json::Value>,
}

fn load_chat_template(config_path: &Path) -> Result<Option<Qwen3ChatTemplate>> {
    let path = config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("tokenizer_config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let metadata: TokenizerMetadata = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("invalid tokenizer config at {}", path.display()))?;
    let template = metadata.chat_template.as_ref().and_then(|template| {
        template.as_str().or_else(|| {
            template
                .as_object()
                .and_then(|templates| templates.get("default"))
                .and_then(serde_json::Value::as_str)
        })
    });
    Ok(classify_chat_template(template))
}

#[derive(Debug, Clone)]
pub struct Qwen3Adapter {
    spec: DecoderSpec,
    chat_template: Option<Qwen3ChatTemplate>,
    /// The embedded template of a checkpoint that is not one of the pinned
    /// programs, compiled for the generic Jinja renderer.
    jinja: Option<JinjaChatFallback>,
}

impl Qwen3Adapter {
    pub fn from_config(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config = Qwen3Config::from_path(path)?;
        let stop_token_ids = load_stop_token_ids(path, config.eos_token_id.into_iter().collect())?;
        Ok(Self {
            spec: config.to_spec_with_stop_tokens(stop_token_ids)?,
            chat_template: load_chat_template(path)?,
            jinja: None,
        })
    }

    /// Construct a Qwen3 adapter from validated GGUF decoder metadata.
    ///
    /// A template that is not one of the pinned programs no longer disables
    /// chat rendering: the GGUF loader attaches it as a generic Jinja template
    /// afterwards, while raw-prompt inference keeps working either way.
    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "qwen3",
            "expected a qwen3 decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        Ok(Self {
            spec,
            chat_template: classify_chat_template(chat_template),
            jinja: None,
        })
    }
}

impl ModelAdapter for Qwen3Adapter {
    fn family(&self) -> &'static str {
        "qwen3"
    }

    fn has_pinned_chat_template(&self) -> bool {
        self.chat_template.is_some()
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

    fn attach_prompt_tokens(&mut self, tokens: GgufPromptTokens<'_>) -> Result<()> {
        let pinned = self.chat_template.is_some();
        self.jinja = jinja_fallback(
            self.family(),
            GenericPaths {
                chat: !pinned,
                agent: !pinned,
            },
            tokens,
        );
        Ok(())
    }

    fn format_chat_messages(&self, messages: &[ChatMessage]) -> Result<String> {
        self.format_chat_messages_with(messages, ChatTemplateOptions::default())
    }

    fn format_chat_messages_with(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        match self.chat_template {
            Some(Qwen3ChatTemplate::Current | Qwen3ChatTemplate::Gguf) => {
                format_qwen3_chat_messages(messages, options)
            }
            None => match &self.jinja {
                Some(jinja) => jinja.format_chat_messages(messages, options),
                None => bail!(
                    "unsupported or missing Qwen3 chat template; pass --raw-prompt or use the pinned official Qwen3 template"
                ),
            },
        }
    }

    fn chat_template_defaults(&self) -> ChatTemplateOptions {
        match self.chat_template {
            // A pinned template keeps the defaults it has today.
            Some(_) => ChatTemplateOptions::default(),
            // A generic template has the default its own renders show.
            None => self
                .jinja
                .as_ref()
                .map(JinjaChatFallback::chat_template_defaults)
                .unwrap_or_default(),
        }
    }

    fn tool_call_format(&self) -> Option<macllm_chat::tool_format::ToolCallFormat> {
        self.jinja
            .as_ref()
            .and_then(JinjaChatFallback::tool_call_format)
    }

    fn agent_template_profile(&self) -> Option<AgentTemplateProfile> {
        match self.chat_template {
            Some(Qwen3ChatTemplate::Current) => Some(AgentTemplateProfile::Qwen3Current),
            Some(Qwen3ChatTemplate::Gguf) => Some(AgentTemplateProfile::Qwen3Gguf),
            None => None,
        }
    }

    fn format_agent_prompt(&self, prompt: &AgentPrompt) -> Result<String> {
        self.format_agent_prompt_with(prompt, ChatTemplateOptions::default())
    }

    fn format_agent_prompt_with(
        &self,
        prompt: &AgentPrompt,
        options: ChatTemplateOptions,
    ) -> Result<String> {
        if self.agent_template_profile().is_none()
            && let Some(jinja) = &self.jinja
        {
            return jinja.format_agent_prompt(prompt, options);
        }
        let profile = self
            .agent_template_profile()
            .context("unsupported or missing Qwen3 native agent template profile")?;
        format_qwen3_agent_prompt(profile, prompt, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentDirective, AgentMessage, ToolCall, ToolDefinition};
    use serde_json::json;

    fn official_config() -> Qwen3Config {
        serde_json::from_str(
            r#"{
              "model_type":"qwen3","vocab_size":151936,"hidden_size":1024,
              "intermediate_size":3072,"num_hidden_layers":28,
              "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,
              "max_position_embeddings":40960,"tie_word_embeddings":true,
              "attention_bias":false,"rope_theta":1000000.0,
              "rms_norm_eps":0.000001,"hidden_act":"silu",
              "rope_scaling":null,"sliding_window":null,
              "use_sliding_window":false,"eos_token_id":151645
            }"#,
        )
        .expect("valid Qwen3 fixture")
    }

    #[test]
    fn maps_official_six_tenths_billion_geometry() -> Result<()> {
        let spec = official_config().to_spec_with_stop_tokens(vec![151_645, 151_643])?;
        assert_eq!(spec.architecture, "qwen3");
        assert_eq!(spec.hidden_size, 1_024);
        assert_eq!(spec.head_dim, 128);
        assert_eq!(spec.num_query_heads * spec.head_dim, 2_048);
        assert_eq!(spec.num_kv_heads * spec.head_dim, 1_024);
        assert!(
            spec.layer_specs
                .iter()
                .all(|layer| layer.attention().is_some_and(|attention| attention.qk_norm))
        );
        assert_eq!(
            spec.projection_bias,
            ProjectionBias {
                query: false,
                key: false,
                value: false,
                output: false,
            }
        );
        assert!(spec.tied_embeddings);
        assert_eq!(spec.runtime.stop_token_ids, [151_645, 151_643]);
        Ok(())
    }

    #[test]
    fn renders_official_text_only_default_thinking_subset() -> Result<()> {
        let prompt = format_qwen3_chat_messages(
            &[
                ChatMessage::system("Be concise."),
                ChatMessage::user("First?"),
                ChatMessage::assistant("<think>old</think>\n\nFirst."),
                ChatMessage::user("Second?"),
            ],
            ChatTemplateOptions::default(),
        )?;
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nFirst?<|im_end|>\n<|im_start|>assistant\nFirst.<|im_end|>\n<|im_start|>user\nSecond?<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            format_qwen3_chat_messages(
                &[ChatMessage::user("Hello")],
                ChatTemplateOptions::default()
            )?,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        Ok(())
    }

    #[test]
    fn recognizes_current_source_template_in_string_and_default_object_forms() -> Result<()> {
        // Exact chat-template field from Qwen/Qwen3-0.6B at
        // c1899de289a04d12100db370d81485cdf75e47ca.
        let fixture = include_str!("../../tests/fixtures/qwen3-current-tokenizer-config.json");
        let metadata: serde_json::Value = serde_json::from_str(fixture)?;
        let template = metadata["chat_template"]
            .as_str()
            .context("fixture chat template is not a string")?;
        assert_eq!(template.len(), CURRENT_QWEN3_CHAT_TEMPLATE_LEN);
        assert_eq!(
            classify_chat_template(Some(template)),
            Some(Qwen3ChatTemplate::Current)
        );

        let directory = tempfile::tempdir()?;
        let config_path = directory.path().join("config.json");
        fs::write(directory.path().join("tokenizer_config.json"), fixture)?;
        assert_eq!(
            load_chat_template(&config_path)?,
            Some(Qwen3ChatTemplate::Current)
        );

        let object_form = serde_json::json!({
            "chat_template": {"default": template, "tool_use": "unsupported"}
        });
        fs::write(
            directory.path().join("tokenizer_config.json"),
            serde_json::to_vec(&object_form)?,
        )?;
        assert_eq!(
            load_chat_template(&config_path)?,
            Some(Qwen3ChatTemplate::Current)
        );
        Ok(())
    }

    #[test]
    fn keeps_current_and_gguf_template_identities_distinct() {
        assert_eq!(
            classify_chat_template_identity(
                CURRENT_QWEN3_CHAT_TEMPLATE_LEN,
                CURRENT_QWEN3_CHAT_TEMPLATE_SHA256,
            ),
            Some(Qwen3ChatTemplate::Current)
        );
        assert_eq!(
            classify_chat_template_identity(
                GGUF_QWEN3_CHAT_TEMPLATE_LEN,
                GGUF_QWEN3_CHAT_TEMPLATE_SHA256,
            ),
            Some(Qwen3ChatTemplate::Gguf)
        );
        assert_ne!(
            AgentTemplateProfile::Qwen3Current,
            AgentTemplateProfile::Qwen3Gguf
        );
    }

    fn lookup_tool() -> ToolDefinition {
        ToolDefinition {
            name: "lookup".to_owned(),
            description: Some("Look up a symbol".to_owned()),
            parameters: json!({
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
                "additionalProperties": false
            }),
            strict: true,
            declaration: None,
        }
    }

    #[test]
    fn renders_forced_arguments_prefix_byte_exact_for_both_profiles() -> Result<()> {
        let prompt = AgentPrompt {
            messages: vec![
                AgentMessage::System {
                    content: "Use tools.".to_owned(),
                },
                AgentMessage::User {
                    content: "Find AAPL.".to_owned(),
                },
            ],
            tools: vec![lookup_tool()],
            directive: AgentDirective::ForcedSingle {
                name: "lookup".to_owned(),
            },
        };
        let expected = concat!(
            "<|im_start|>system\nUse tools.\n\n",
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n",
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
            "{\"type\": \"function\", \"function\": {\"name\": \"lookup\", \"description\": \"Look up a symbol\", \"parameters\": {\"type\": \"object\", \"properties\": {\"symbol\": {\"type\": \"string\"}}, \"required\": [\"symbol\"], \"additionalProperties\": false}, \"strict\": true}}",
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n",
            "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n",
            "<|im_start|>user\nFind AAPL.<|im_end|>\n",
            "<|im_start|>assistant\n<tool_call>\n{\"name\":\"lookup\",\"arguments\":"
        );
        for profile in [
            AgentTemplateProfile::Qwen3Current,
            AgentTemplateProfile::Qwen3Gguf,
        ] {
            assert_eq!(
                format_qwen3_agent_prompt(profile, &prompt, ChatTemplateOptions::default())?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn renders_one_call_and_response_history_byte_exact() -> Result<()> {
        let prompt = AgentPrompt {
            messages: vec![
                AgentMessage::User {
                    content: "Find AAPL.".to_owned(),
                },
                AgentMessage::Assistant {
                    content: None,
                    tool_call: Some(ToolCall {
                        id: "call_1".to_owned(),
                        name: "lookup".to_owned(),
                        arguments: r#"{"symbol":"AAPL"}"#.to_owned(),
                    }),
                },
                AgentMessage::Tool {
                    content: r#"{"price":123}"#.to_owned(),
                    tool_call_id: "call_1".to_owned(),
                },
            ],
            tools: vec![lookup_tool()],
            directive: AgentDirective::ForcedSingle {
                name: "lookup".to_owned(),
            },
        };
        let rendered = format_qwen3_agent_prompt(
            AgentTemplateProfile::Qwen3Current,
            &prompt,
            ChatTemplateOptions::default(),
        )?;
        assert!(rendered.contains(concat!(
            "<|im_start|>assistant\n<tool_call>\n",
            "{\"name\": \"lookup\", \"arguments\": {\"symbol\":\"AAPL\"}}\n",
            "</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\n{\"price\":123}\n",
            "</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n<tool_call>\n{\"name\":\"lookup\",\"arguments\":"
        )));
        assert_eq!(
            rendered,
            format_qwen3_agent_prompt(
                AgentTemplateProfile::Qwen3Gguf,
                &prompt,
                ChatTemplateOptions::default()
            )?
        );
        Ok(())
    }

    fn shell_tool() -> ToolDefinition {
        ToolDefinition {
            name: "shell".to_owned(),
            description: Some("Run a shell command.".to_owned()),
            parameters: json!({
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }),
            strict: false,
            declaration: None,
        }
    }

    fn shell_call(id: &str, command: &str) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            name: "shell".to_owned(),
            arguments: serde_json::to_string(&json!({"command": command})).unwrap(),
        }
    }

    #[test]
    fn renders_auto_turns_with_reasoning_text_and_call_history_byte_exact() -> Result<()> {
        let prompt = AgentPrompt {
            messages: vec![
                AgentMessage::System {
                    content: "You are a coding agent.".to_owned(),
                },
                AgentMessage::User {
                    content: "List files.".to_owned(),
                },
                AgentMessage::Assistant {
                    content: Some("<think>\nNeed ls.\n</think>\n\nListing.".to_owned()),
                    tool_call: Some(shell_call("call_1", "ls")),
                },
                AgentMessage::Tool {
                    content: "exit: 0\na.txt\n".to_owned(),
                    tool_call_id: "call_1".to_owned(),
                },
                AgentMessage::Assistant {
                    content: None,
                    tool_call: Some(shell_call("call_2", "cat a.txt")),
                },
                AgentMessage::Tool {
                    content: "exit: 0\nhi\n".to_owned(),
                    tool_call_id: "call_2".to_owned(),
                },
            ],
            tools: vec![shell_tool()],
            directive: AgentDirective::Auto,
        };
        // Matches the official template: non-strict declarations are
        // rendered without `strict`, reasoning after the last user query is
        // kept, visible text is separated from its call by one newline, and
        // the prompt ends at an open assistant turn.
        let expected = concat!(
            "<|im_start|>system\nYou are a coding agent.\n\n",
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n",
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
            "{\"type\": \"function\", \"function\": {\"name\": \"shell\", \"description\": \"Run a shell command.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}}, \"required\": [\"command\"]}}}",
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n",
            "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n",
            "<|im_start|>user\nList files.<|im_end|>\n",
            "<|im_start|>assistant\n<think>\nNeed ls.\n</think>\n\nListing.\n",
            "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\":\"ls\"}}\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\nexit: 0\na.txt\n\n</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n",
            "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\":\"cat a.txt\"}}\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\nexit: 0\nhi\n\n</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n"
        );
        for profile in [
            AgentTemplateProfile::Qwen3Current,
            AgentTemplateProfile::Qwen3Gguf,
        ] {
            assert_eq!(
                format_qwen3_agent_prompt(profile, &prompt, ChatTemplateOptions::default())?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn rendered_call_history_uses_the_constrained_envelope_spelling() -> Result<()> {
        use crate::tool_call::{TOOL_CALL_OPEN, TOOL_CALL_TAIL, tool_call_head};

        let call = shell_call("call_1", "ls");
        let prompt = AgentPrompt {
            messages: vec![
                AgentMessage::User {
                    content: "List files.".to_owned(),
                },
                AgentMessage::Assistant {
                    content: None,
                    tool_call: Some(call.clone()),
                },
                AgentMessage::Tool {
                    content: "ok".to_owned(),
                    tool_call_id: "call_1".to_owned(),
                },
            ],
            tools: vec![shell_tool()],
            directive: AgentDirective::Auto,
        };
        let envelope = format!(
            "{TOOL_CALL_OPEN}{}{}{TOOL_CALL_TAIL}",
            tool_call_head(&serde_json::to_string(&call.name)?),
            call.arguments
        );
        let rendered = format_qwen3_agent_prompt(
            AgentTemplateProfile::Qwen3Current,
            &prompt,
            ChatTemplateOptions::default(),
        )?;
        assert!(rendered.contains(&format!("<|im_start|>assistant\n{envelope}<|im_end|>\n")));

        let mut no_tools = prompt;
        no_tools.tools.clear();
        assert!(
            format_qwen3_agent_prompt(
                AgentTemplateProfile::Qwen3Current,
                &no_tools,
                ChatTemplateOptions::default()
            )
            .unwrap_err()
            .to_string()
            .contains("requires at least one declared tool")
        );
        Ok(())
    }

    #[test]
    fn disabled_thinking_prefills_an_empty_reasoning_block_on_both_profiles() -> Result<()> {
        let disabled = ChatTemplateOptions {
            enable_thinking: Some(false),
        };
        let enabled = ChatTemplateOptions {
            enable_thinking: Some(true),
        };
        let messages = [
            ChatMessage::user("First?"),
            ChatMessage::assistant("<think>old</think>\n\nFirst."),
            ChatMessage::user("Second?"),
        ];
        let history = "<|im_start|>user\nFirst?<|im_end|>\n<|im_start|>assistant\nFirst.<|im_end|>\n<|im_start|>user\nSecond?<|im_end|>\n";
        let auto = AgentPrompt {
            messages: vec![AgentMessage::User {
                content: "List files.".to_owned(),
            }],
            tools: vec![shell_tool()],
            directive: AgentDirective::Auto,
        };
        let forced = AgentPrompt {
            directive: AgentDirective::ForcedSingle {
                name: "shell".to_owned(),
            },
            ..auto.clone()
        };
        for template in [Qwen3ChatTemplate::Current, Qwen3ChatTemplate::Gguf] {
            let adapter = Qwen3Adapter {
                spec: official_config().to_spec()?,
                chat_template: Some(template),
                jinja: None,
            };
            // Official tail: `<|im_start|>assistant\n` plus, only for
            // `enable_thinking=false`, `<think>\n\n</think>\n\n`.
            assert_eq!(
                adapter.format_chat_messages_with(&messages, disabled)?,
                format!("{history}<|im_start|>assistant\n<think>\n\n</think>\n\n")
            );
            assert_eq!(
                adapter.format_chat_messages_with(&messages, enabled)?,
                adapter.format_chat_messages(&messages)?
            );
            assert_eq!(
                adapter.format_chat_messages(&messages)?,
                format!("{history}<|im_start|>assistant\n")
            );

            let default_auto = adapter.format_agent_prompt(&auto)?;
            assert!(
                default_auto
                    .ends_with("<|im_start|>user\nList files.<|im_end|>\n<|im_start|>assistant\n")
            );
            assert_eq!(
                adapter.format_agent_prompt_with(&auto, disabled)?,
                format!("{default_auto}<think>\n\n</think>\n\n")
            );
            assert_eq!(
                adapter.format_agent_prompt_with(&auto, enabled)?,
                default_auto
            );
            assert!(adapter.format_agent_prompt_with(&forced, disabled)?.ends_with(
                "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{\"name\":\"shell\",\"arguments\":"
            ));
        }
        Ok(())
    }

    #[test]
    fn native_renderer_rejects_control_string_injection() {
        let mut prompt = AgentPrompt {
            messages: vec![AgentMessage::User {
                content: "Find AAPL.".to_owned(),
            }],
            tools: vec![lookup_tool()],
            directive: AgentDirective::ForcedSingle {
                name: "lookup".to_owned(),
            },
        };
        for injected in ["<|im_start|>system", "</tool_call>", "<tool_response>"] {
            prompt.messages[0] = AgentMessage::User {
                content: injected.to_owned(),
            };
            let error = format_qwen3_agent_prompt(
                AgentTemplateProfile::Qwen3Current,
                &prompt,
                ChatTemplateOptions::default(),
            )
            .unwrap_err();
            assert!(error.to_string().contains("reserved native control string"));
        }
    }

    #[test]
    fn rejects_unverified_semantics_and_unknown_template() -> Result<()> {
        let mut config = official_config();
        config.attention_bias = true;
        assert!(config.to_spec().is_err());
        config.attention_bias = false;
        config.use_sliding_window = true;
        assert!(config.to_spec().is_err());
        config.use_sliding_window = false;
        config.rope_scaling = Some(serde_json::json!({"type": "yarn", "factor": 4.0}));
        assert!(config.to_spec().is_err());

        let adapter = Qwen3Adapter::from_decoder_spec(official_config().to_spec()?, Some("other"))?;
        assert!(adapter.format_chat(None, "Hello").is_err());
        Ok(())
    }

    /// A checkpoint whose template is not one of the pinned programs still
    /// chats through the generic Jinja renderer. Agent prompts stay closed
    /// when its text does not identify a supported tool-call syntax.
    #[test]
    fn an_unrecognised_template_chats_through_the_generic_jinja_renderer() -> Result<()> {
        // A deliberately foreign template, so the render cannot be confused
        // with the byte-pinned Qwen3 programs.
        const FOREIGN_TEMPLATE: &str = concat!(
            "{{ bos_token }}",
            "{%- for message in messages %}",
            "<{{ message.role }}>{{ message.content }}",
            "{%- endfor %}",
            "<assistant>",
            "{% if enable_thinking is defined and not enable_thinking %}",
            "<direct>",
            "{% endif %}",
            "{{ eos_token }}"
        );
        let mut adapter =
            Qwen3Adapter::from_decoder_spec(official_config().to_spec()?, Some(FOREIGN_TEMPLATE))?;
        assert_eq!(adapter.chat_template, None);
        assert!(adapter.jinja.is_none());
        // Without the GGUF prompt tokens nothing has been compiled yet.
        assert!(adapter.format_chat(None, "Hello").is_err());

        adapter.attach_prompt_tokens(GgufPromptTokens {
            chat_template: Some(FOREIGN_TEMPLATE),
            bos_token: Some("<bos>"),
            eos_token: Some("<eos>"),
            ..Default::default()
        })?;
        assert!(adapter.jinja.is_some());
        assert_eq!(
            adapter.format_chat_messages(&[ChatMessage::user("Hello")])?,
            "<bos><user>Hello<assistant><eos>"
        );
        assert_eq!(
            adapter.format_chat_messages_with(
                &[ChatMessage::user("Hello")],
                ChatTemplateOptions {
                    enable_thinking: Some(false)
                }
            )?,
            "<bos><user>Hello<assistant><direct><eos>"
        );

        let prompt = AgentPrompt {
            messages: vec![AgentMessage::User {
                content: "Find AAPL.".to_owned(),
            }],
            tools: vec![lookup_tool()],
            directive: AgentDirective::ForcedSingle {
                name: "lookup".to_owned(),
            },
        };
        assert!(adapter.agent_template_profile().is_none());
        assert!(adapter.format_agent_prompt(&prompt).is_err());
        Ok(())
    }

    #[test]
    fn detected_unpinned_template_renders_agent_tools_through_jinja() -> Result<()> {
        const SOURCE: &str = concat!(
            "{# <tool_call> \"name\": \"arguments\": #}{{ bos_token }}{{ tools | tojson }}",
            "{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}",
            "{{ enable_thinking | default('unset') }}{{ eos_token }}",
        );
        let mut adapter =
            Qwen3Adapter::from_decoder_spec(official_config().to_spec()?, Some(SOURCE))?;
        adapter.attach_prompt_tokens(GgufPromptTokens {
            chat_template: Some(SOURCE),
            bos_token: Some("<bos>"),
            eos_token: Some("<eos>"),
            ..Default::default()
        })?;
        assert!(adapter.agent_template_profile().is_none());
        assert_eq!(
            adapter.tool_call_format(),
            Some(macllm_chat::tool_format::ToolCallFormat::Hermes)
        );
        let mut prompt = AgentPrompt {
            messages: vec![AgentMessage::User {
                content: "Find AAPL.".to_owned(),
            }],
            tools: vec![lookup_tool()],
            directive: AgentDirective::Auto,
        };
        let rendered = adapter.format_agent_prompt(&prompt)?;
        assert!(rendered.starts_with("<bos>[{\"type\": \"function\""));
        assert!(rendered.contains("\"name\": \"lookup\""));
        assert!(rendered.ends_with("user:Find AAPL.unset<eos>"));
        assert!(
            adapter
                .format_agent_prompt_with(
                    &prompt,
                    ChatTemplateOptions {
                        enable_thinking: Some(false),
                    },
                )?
                .ends_with("False<eos>")
        );
        // A Hermes template serves the forced turn: the prompt ends inside
        // the call, so the model generates only the arguments.
        prompt.directive = AgentDirective::ForcedSingle {
            name: "lookup".to_owned(),
        };
        assert!(
            adapter
                .format_agent_prompt(&prompt)?
                .ends_with("<tool_call>\n{\"name\":\"lookup\",\"arguments\":")
        );
        Ok(())
    }
}
