//! Byte-exact Qwen3.5 chat and native-tool prompt renderers.
//!
//! The renderers reproduce the two pinned upstream Jinja programs (the
//! official `Qwen/Qwen3.5-4B` template and the `unsloth` GGUF variant)
//! without executing a checkpoint's template. The prompt bytes are part of
//! the served contract, so the adapters reuse these functions unchanged.

use std::fmt::Write;

use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};

use crate::{
    agent::{AgentMessage, AgentPrompt, AgentTemplateProfile},
    chat::{ChatMessage, ChatRole, ChatTemplateOptions, validate_chat_messages},
    chat_templates::qwen3::{python_json, serialize_native_tool},
};

// `tokenizer.chat_template` embedded in ggml-org/Qwen3.5-0.8B-GGUF at
// revision 8fea620810c4afa23dd6443f999a48574c1611a3.
pub const OFFICIAL_GGUF_CHAT_TEMPLATE_LEN: usize = 7_755;
pub const OFFICIAL_GGUF_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x27, 0x3d, 0x8e, 0x0e, 0x68, 0x3b, 0x88, 0x50, 0x71, 0xfb, 0x17, 0xe0, 0x8d, 0x71, 0xe5, 0xf2,
    0xa5, 0xdd, 0xfb, 0x53, 0x09, 0x75, 0x61, 0x81, 0x68, 0x1d, 0xe4, 0xf5, 0xa1, 0x82, 0x2d, 0x80,
];

// `chat_template.jinja` from Qwen/Qwen3.5-4B at revision
// 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
pub const OFFICIAL_CHAT_TEMPLATE_LEN: usize = 7_756;
pub const OFFICIAL_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0xa4, 0xae, 0xe8, 0xaf, 0xcf, 0x2e, 0x07, 0x11, 0x94, 0x2c, 0xf8, 0x48, 0x89, 0x9b, 0xe6, 0x60,
    0x16, 0xf8, 0xd1, 0x4a, 0x88, 0x9f, 0xf9, 0xed, 0xe0, 0x7b, 0xca, 0x09, 0x9c, 0x28, 0xf7, 0x15,
];

// Embedded in unsloth/Qwen3.5-4B-GGUF at revision e87f1764 (also the 2B
// and 9B GGUFs).
pub const UNSLOTH_GGUF_CHAT_TEMPLATE_LEN: usize = 7_816;
pub const UNSLOTH_GGUF_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x7f, 0x0e, 0x52, 0x90, 0x32, 0xc2, 0x51, 0x83, 0xbc, 0xd6, 0x6c, 0x7f, 0x23, 0x8d, 0xa2, 0xd3,
    0x77, 0xf4, 0x3b, 0xe7, 0x54, 0xa9, 0x4e, 0x27, 0x25, 0xa5, 0x8c, 0x4e, 0x16, 0xd2, 0xed, 0x67,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35ChatTemplate {
    OfficialTextSubset,
    Official,
    Unsloth,
}

#[must_use]
pub fn classify_chat_template(template: Option<&str>) -> Option<Qwen35ChatTemplate> {
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

pub fn format_qwen35_chat_messages(messages: &[ChatMessage]) -> Result<String> {
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

pub fn format_qwen35_plain(
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

pub fn format_qwen35_agent_prompt(
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
