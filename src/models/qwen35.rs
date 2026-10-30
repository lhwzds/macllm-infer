use std::fmt::Write;

use anyhow::{Result, bail, ensure};
use sha2::{Digest, Sha256};

use crate::{
    agent::{AgentMessage, AgentPrompt, AgentTemplateProfile},
    architecture::{DecoderSpec, RuntimeSpec},
    chat::{ChatMessage, ChatRole, ChatTemplateOptions, validate_chat_messages},
    decoder::{
        DecoderLayerWeightNames, DecoderModel, DecoderWeightNames, STANDARD_DECODER_WEIGHT_NAMES,
    },
    model_weights::WeightSource,
    models::{
        ModelAdapter,
        qwen3::{python_json, serialize_native_tool},
    },
};

pub(crate) const QWEN35_WEIGHT_NAMES: DecoderWeightNames = DecoderWeightNames {
    layer: DecoderLayerWeightNames {
        mixer_norm: "attn_norm",
        feed_forward_norm: "attn_post_norm",
        gated_delta: Some("linear_attn"),
        ..STANDARD_DECODER_WEIGHT_NAMES.layer
    },
    ..STANDARD_DECODER_WEIGHT_NAMES
};

// `tokenizer.chat_template` embedded in ggml-org/Qwen3.5-0.8B-GGUF at
// revision 8fea620810c4afa23dd6443f999a48574c1611a3.
const OFFICIAL_GGUF_CHAT_TEMPLATE_LEN: usize = 7_755;
const OFFICIAL_GGUF_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x27, 0x3d, 0x8e, 0x0e, 0x68, 0x3b, 0x88, 0x50, 0x71, 0xfb, 0x17, 0xe0, 0x8d, 0x71, 0xe5, 0xf2,
    0xa5, 0xdd, 0xfb, 0x53, 0x09, 0x75, 0x61, 0x81, 0x68, 0x1d, 0xe4, 0xf5, 0xa1, 0x82, 0x2d, 0x80,
];

// `chat_template.jinja` from Qwen/Qwen3.5-4B at revision
// 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
const OFFICIAL_CHAT_TEMPLATE_LEN: usize = 7_756;
const OFFICIAL_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0xa4, 0xae, 0xe8, 0xaf, 0xcf, 0x2e, 0x07, 0x11, 0x94, 0x2c, 0xf8, 0x48, 0x89, 0x9b, 0xe6, 0x60,
    0x16, 0xf8, 0xd1, 0x4a, 0x88, 0x9f, 0xf9, 0xed, 0xe0, 0x7b, 0xca, 0x09, 0x9c, 0x28, 0xf7, 0x15,
];

// Embedded in unsloth/Qwen3.5-4B-GGUF at revision e87f1764 (also the 2B
// and 9B GGUFs).
const UNSLOTH_GGUF_CHAT_TEMPLATE_LEN: usize = 7_816;
const UNSLOTH_GGUF_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x7f, 0x0e, 0x52, 0x90, 0x32, 0xc2, 0x51, 0x83, 0xbc, 0xd6, 0x6c, 0x7f, 0x23, 0x8d, 0xa2, 0xd3,
    0x77, 0xf4, 0x3b, 0xe7, 0x54, 0xa9, 0x4e, 0x27, 0x25, 0xa5, 0x8c, 0x4e, 0x16, 0xd2, 0xed, 0x67,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen35ChatTemplate {
    OfficialTextSubset,
    Official,
    Unsloth,
}

fn classify_chat_template(template: Option<&str>) -> Option<Qwen35ChatTemplate> {
    let template = template?;
    let digest: [u8; 32] = Sha256::digest(template.as_bytes()).into();
    match (template.len(), digest) {
        (OFFICIAL_GGUF_CHAT_TEMPLATE_LEN, OFFICIAL_GGUF_CHAT_TEMPLATE_SHA256) => {
            Some(Qwen35ChatTemplate::OfficialTextSubset)
        }
        (OFFICIAL_CHAT_TEMPLATE_LEN, OFFICIAL_CHAT_TEMPLATE_SHA256) => {
            Some(Qwen35ChatTemplate::Official)
        }
        (UNSLOTH_GGUF_CHAT_TEMPLATE_LEN, UNSLOTH_GGUF_CHAT_TEMPLATE_SHA256) => {
            Some(Qwen35ChatTemplate::Unsloth)
        }
        _ => None,
    }
}

/// Adapter for the dense Qwen3.5 text trunk stored in canonical llama.cpp
/// GGUF form.
///
/// Exact template identities are recognized by digest. The two newer
/// identities support text and native tool history; the older GGUF identity
/// remains limited to its verified plain-text subset. Multimodal input is not
/// supported.
#[derive(Debug, Clone)]
pub struct Qwen35Adapter {
    spec: DecoderSpec,
    chat_template: Option<Qwen35ChatTemplate>,
}

impl Qwen35Adapter {
    /// Construct an adapter from the metadata and tensors already validated by
    /// the GGUF loader.
    pub fn from_decoder_spec(spec: DecoderSpec, chat_template: Option<&str>) -> Result<Self> {
        ensure!(
            spec.architecture == "qwen35",
            "expected a qwen35 decoder spec, got {}",
            spec.architecture
        );
        spec.validate()?;
        Ok(Self {
            spec,
            chat_template: classify_chat_template(chat_template),
        })
    }
}

impl ModelAdapter for Qwen35Adapter {
    fn family(&self) -> &'static str {
        "qwen35"
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
            &QWEN35_WEIGHT_NAMES,
        )?))
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
            Some(Qwen35ChatTemplate::OfficialTextSubset) => {
                options.ensure_default(self.family())?;
                format_qwen35_chat_messages(messages)
            }
            Some(template @ (Qwen35ChatTemplate::Official | Qwen35ChatTemplate::Unsloth)) => {
                format_qwen35_plain(messages, template == Qwen35ChatTemplate::Official, options)
            }
            None => bail!(
                "unsupported or missing Qwen3.5 chat template; use `run --raw-prompt` or a pinned Qwen3.5 template"
            ),
        }
    }

    fn chat_template_defaults(&self) -> ChatTemplateOptions {
        ChatTemplateOptions {
            enable_thinking: match self.chat_template {
                Some(Qwen35ChatTemplate::Official) => Some(true),
                Some(Qwen35ChatTemplate::Unsloth) => Some(false),
                Some(Qwen35ChatTemplate::OfficialTextSubset) | None => None,
            },
        }
    }

    fn agent_template_profile(&self) -> Option<AgentTemplateProfile> {
        match self.chat_template {
            Some(Qwen35ChatTemplate::Official) => Some(AgentTemplateProfile::Qwen35Official),
            Some(Qwen35ChatTemplate::Unsloth) => Some(AgentTemplateProfile::Qwen35Unsloth),
            _ => None,
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
        let profile = self
            .agent_template_profile()
            .ok_or_else(|| anyhow::anyhow!("unsupported Qwen3.5 native agent template"))?;
        format_qwen35_agent_prompt(profile, prompt, options)
    }
}

fn format_qwen35_chat_messages(messages: &[ChatMessage]) -> Result<String> {
    validate_chat_messages(messages)?;
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(message.role.as_str());
        prompt.push('\n');
        let content = message.content.trim();
        if message.role == ChatRole::Assistant {
            // The official template removes historical reasoning before the
            // next user turn. The common contract always ends in user, so all
            // assistant messages reaching this branch are historical.
            prompt.push_str(
                content
                    .rsplit_once("</think>")
                    .map_or(content, |(_, answer)| answer.trim_start_matches('\n')),
            );
        } else {
            prompt.push_str(content);
        }
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    Ok(prompt)
}

fn generation(rendered: &mut String, thinking: bool) {
    rendered.push_str("<|im_start|>assistant\n<think>\n");
    if !thinking {
        rendered.push_str("\n</think>\n\n");
    }
}

pub(crate) fn format_qwen35_plain(
    messages: &[ChatMessage],
    default_thinking: bool,
    options: ChatTemplateOptions,
) -> Result<String> {
    validate_chat_messages(messages)?;
    let mut out = String::new();
    for message in messages {
        let content = message.content.trim();
        let content = if message.role == ChatRole::Assistant {
            content
                .rsplit_once("</think>")
                .map_or(content, |(_, s)| s.trim_start_matches('\n'))
        } else {
            content
        };
        write!(
            out,
            "<|im_start|>{}\n{}<|im_end|>\n",
            message.role.as_str(),
            content
        )?;
    }
    generation(
        &mut out,
        options.enable_thinking.unwrap_or(default_thinking),
    );
    Ok(out)
}

pub(crate) fn format_qwen35_agent_prompt(
    profile: AgentTemplateProfile,
    prompt: &AgentPrompt,
    options: ChatTemplateOptions,
) -> Result<String> {
    ensure!(
        matches!(
            profile,
            AgentTemplateProfile::Qwen35Official | AgentTemplateProfile::Qwen35Unsloth
        ),
        "invalid Qwen3.5 profile"
    );
    prompt.validate()?;
    let mut out = String::new();
    let system = match prompt.messages.first() {
        Some(AgentMessage::System { content }) => Some(content.trim()),
        _ => None,
    };
    if !prompt.tools.is_empty() {
        out.push_str(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>",
        );
        for tool in &prompt.tools {
            out.push('\n');
            out.push_str(&serialize_native_tool(tool)?);
        }
        out.push_str("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
        if let Some(system) = system.filter(|s| !s.is_empty()) {
            out.push_str("\n\n");
            out.push_str(system);
        }
        out.push_str("<|im_end|>\n");
    } else if let Some(system) = system {
        write!(out, "<|im_start|>system\n{system}<|im_end|>\n")?;
    }
    let last_user = prompt
        .messages
        .iter()
        .rposition(|m| matches!(m, AgentMessage::User { .. }))
        .ok_or_else(|| anyhow::anyhow!("no user query"))?;
    for (index, message) in prompt.messages.iter().enumerate() {
        match message {
            AgentMessage::System { .. } => {}
            AgentMessage::User { content } => {
                write!(out, "<|im_start|>user\n{}<|im_end|>\n", content.trim())?;
            }
            AgentMessage::Assistant { content, tool_call } => {
                let calls: &[crate::agent::ToolCall] = tool_call.as_slice();
                let content = content.as_deref().unwrap_or_default().trim();
                let (reasoning, visible) = if content.contains("</think>") {
                    let first = content
                        .split("</think>")
                        .next()
                        .unwrap_or_default()
                        .trim_end_matches('\n');
                    (
                        first
                            .rsplit("<think>")
                            .next()
                            .unwrap_or_default()
                            .trim_start_matches('\n')
                            .trim(),
                        content
                            .rsplit("</think>")
                            .next()
                            .unwrap_or_default()
                            .trim_start_matches('\n'),
                    )
                } else {
                    ("", content)
                };
                out.push_str("<|im_start|>assistant\n");
                if index > last_user {
                    write!(out, "<think>\n{reasoning}\n</think>\n\n")?;
                }
                out.push_str(visible);
                for (call_index, call) in calls.iter().enumerate() {
                    if call_index > 0 {
                        out.push('\n');
                    } else if !visible.trim().is_empty() {
                        out.push_str("\n\n");
                    }
                    write!(out, "<tool_call>\n<function={}>\n", call.name)?;
                    let args: serde_json::Value = serde_json::from_str(&call.arguments)?;
                    let args = args
                        .as_object()
                        .ok_or_else(|| anyhow::anyhow!("tool arguments must encode an object"))?;
                    for (key, value) in args {
                        let text = match value {
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Bool(b) => {
                                if *b { "True" } else { "False" }.to_owned()
                            }
                            serde_json::Value::Null => "None".to_owned(),
                            _ => python_json(value)?,
                        };
                        write!(out, "<parameter={key}>\n{text}\n</parameter>\n")?;
                    }
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            AgentMessage::Tool { content, .. } => {
                if index == 0 || !matches!(prompt.messages[index - 1], AgentMessage::Tool { .. }) {
                    out.push_str("<|im_start|>user");
                }
                write!(
                    out,
                    "\n<tool_response>\n{}\n</tool_response>",
                    content.trim()
                )?;
                if index + 1 == prompt.messages.len()
                    || !matches!(prompt.messages[index + 1], AgentMessage::Tool { .. })
                {
                    out.push_str("<|im_end|>\n");
                }
            }
        }
    }
    generation(
        &mut out,
        options
            .enable_thinking
            .unwrap_or(profile == AgentTemplateProfile::Qwen35Official),
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_official_plain_text_default_non_thinking_subset() -> Result<()> {
        let prompt = format_qwen35_chat_messages(&[
            ChatMessage::system("  Be concise.  "),
            ChatMessage::user(" First? "),
            ChatMessage::assistant("<think>old reasoning</think>\n\nFirst. "),
            ChatMessage::user("Second?"),
        ])?;
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nFirst?<|im_end|>\n<|im_start|>assistant\nFirst.<|im_end|>\n<|im_start|>user\nSecond?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            format_qwen35_chat_messages(&[ChatMessage::user("What is the capital of France?")])?,
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        Ok(())
    }

    #[test]
    fn recognizes_exact_new_identities() {
        for (name, expected) in [
            ("official", Qwen35ChatTemplate::Official),
            ("unsloth", Qwen35ChatTemplate::Unsloth),
        ] {
            let path = format!(
                "{}/tests/fixtures/chat_templates/templates/qwen35-{name}.jinja",
                env!("CARGO_MANIFEST_DIR")
            );
            let mut bytes = std::fs::read(path).unwrap();
            assert_eq!(
                classify_chat_template(Some(std::str::from_utf8(&bytes).unwrap())),
                Some(expected)
            );
            bytes[0] ^= 1;
            assert_eq!(
                classify_chat_template(Some(std::str::from_utf8(&bytes).unwrap())),
                None
            );
        }
    }

    #[test]
    fn unknown_template_stays_raw_only() {
        assert_eq!(classify_chat_template(None), None);
        assert_eq!(classify_chat_template(Some("unverified")), None);
    }
}
