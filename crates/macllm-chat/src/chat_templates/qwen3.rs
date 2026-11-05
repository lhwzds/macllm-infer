//! Byte-exact Qwen3 chat and native-tool prompt renderers.
//!
//! The renderers reproduce the two pinned upstream Jinja programs (the
//! official `Qwen/Qwen3-0.6B` template and the `GGUF` variant) without
//! executing a checkpoint's template. The prompt bytes are part of the
//! served contract, so the adapters reuse these functions unchanged.

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    agent::{AgentDirective, AgentMessage, AgentPrompt, AgentTemplateProfile, ToolDefinition},
    chat::{ChatMessage, ChatRole, ChatTemplateOptions, validate_chat_messages},
};

// Identity of the official Qwen3 tokenizer template published with
// Qwen/Qwen3-0.6B. Recognizing the exact Jinja program keeps raw-prompt
// inference available for other checkpoints without silently applying a
// merely similar prompt grammar.
pub const CURRENT_QWEN3_CHAT_TEMPLATE_LEN: usize = 4_168;
pub const CURRENT_QWEN3_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0xa5, 0x5e, 0xe1, 0xb1, 0x66, 0x01, 0x28, 0xb7, 0x09, 0x87, 0x23, 0xe0, 0xab, 0xcd, 0x92, 0xca,
    0xa0, 0x78, 0x80, 0x61, 0x05, 0x1c, 0x62, 0xd5, 0x1c, 0xbe, 0x87, 0xd9, 0xcf, 0x19, 0x74, 0xd8,
];
pub const GGUF_QWEN3_CHAT_TEMPLATE_LEN: usize = 4_100;
pub const GGUF_QWEN3_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x57, 0xf1, 0xfd, 0x00, 0xf0, 0x01, 0x3a, 0x2b, 0xe9, 0x6a, 0xa7, 0x9b, 0x85, 0x73, 0x91, 0xf2,
    0x7e, 0x23, 0xdf, 0x5b, 0x5f, 0x84, 0x70, 0x72, 0xb5, 0x24, 0xc8, 0x97, 0xe2, 0x4d, 0x03, 0x61,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3ChatTemplate {
    Current,
    Gguf,
}

#[must_use]
pub fn classify_chat_template(template: Option<&str>) -> Option<Qwen3ChatTemplate> {
    let template = template?;
    if !matches!(
        template.len(),
        CURRENT_QWEN3_CHAT_TEMPLATE_LEN | GGUF_QWEN3_CHAT_TEMPLATE_LEN
    ) {
        return None;
    }
    let digest: [u8; 32] = Sha256::digest(template.as_bytes()).into();
    classify_chat_template_identity(template.len(), digest)
}

#[must_use]
pub fn classify_chat_template_identity(
    byte_len: usize,
    digest: [u8; 32],
) -> Option<Qwen3ChatTemplate> {
    match (byte_len, digest) {
        (CURRENT_QWEN3_CHAT_TEMPLATE_LEN, CURRENT_QWEN3_CHAT_TEMPLATE_SHA256) => {
            Some(Qwen3ChatTemplate::Current)
        }
        (GGUF_QWEN3_CHAT_TEMPLATE_LEN, GGUF_QWEN3_CHAT_TEMPLATE_SHA256) => {
            Some(Qwen3ChatTemplate::Gguf)
        }
        _ => None,
    }
}

/// Append the official generation prompt. Both pinned templates end with
/// `<|im_start|>assistant\n` and, only when `enable_thinking` is false,
/// `<think>\n\n</think>\n\n`, which closes an empty reasoning block so the
/// model answers directly. Historical turns render identically either way.
fn push_qwen3_generation_prompt(rendered: &mut String, options: ChatTemplateOptions) {
    rendered.push_str("<|im_start|>assistant\n");
    if options.enable_thinking == Some(false) {
        rendered.push_str("<think>\n\n</think>\n\n");
    }
}

pub fn format_qwen3_chat_messages(
    messages: &[ChatMessage],
    options: ChatTemplateOptions,
) -> Result<String> {
    validate_chat_messages(messages)?;
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(message.role.as_str());
        prompt.push('\n');
        let content = if message.role == ChatRole::Assistant {
            // The official template removes historical reasoning before the
            // next user turn. All assistant messages accepted by the common
            // chat contract are historical because the history ends in user.
            message
                .content
                .rsplit_once("</think>")
                .map_or(message.content.as_str(), |(_, answer)| {
                    answer.trim_start_matches('\n')
                })
        } else {
            &message.content
        };
        prompt.push_str(content);
        prompt.push_str("<|im_end|>\n");
    }
    push_qwen3_generation_prompt(&mut prompt, options);
    Ok(prompt)
}

#[derive(Serialize)]
struct NativeTool<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: NativeFunction<'a>,
}

#[derive(Serialize)]
struct NativeFunction<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    parameters: &'a serde_json::Value,
    // The official template renders the client's declaration verbatim, and
    // clients normally omit `strict` unless they enable it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    strict: bool,
}

pub fn serialize_native_tool(tool: &ToolDefinition) -> Result<String> {
    // `preserve_order` retains client order throughout nested schemas. The
    // grammar explicitly sorts exact literals for its old duplicate and
    // matching behavior; property traversal may use declared order. Other
    // consumers access maps by key or treat their order as presentation only.
    let fallback = serde_json::to_value(NativeTool {
        kind: "function",
        function: NativeFunction {
            name: &tool.name,
            description: tool.description.as_deref(),
            parameters: &tool.parameters,
            strict: tool.strict,
        },
    })?;
    python_json(tool.declaration.as_ref().unwrap_or(&fallback))
}

pub fn python_json(value: &serde_json::Value) -> Result<String> {
    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, PythonJsonFormatter);
    value
        .serialize(&mut serializer)
        .context("failed to serialize Python-style JSON")?;
    Ok(String::from_utf8(bytes)?)
}

// Python json.dumps defaults use a space after both separators.
struct PythonJsonFormatter;
impl serde_json::ser::Formatter for PythonJsonFormatter {
    fn write_f64<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        value: f64,
    ) -> std::io::Result<()> {
        // Rust and Python use shortest-roundtrip float representations with
        // the same fixed/scientific thresholds. Python pads the exponent to
        // two digits and always includes its sign.
        let number = format!("{value:?}");
        if let Some((mantissa, exponent)) = number.split_once('e') {
            let exponent: i32 = exponent.parse().expect("formatted exponent");
            write!(writer, "{mantissa}e{exponent:+03}")
        } else {
            writer.write_all(number.as_bytes())
        }
    }
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

/// Render the audited Qwen3 native-tool subset without executing Jinja from a
/// checkpoint.  The Current and GGUF profiles intentionally converge on the
/// same bytes for this subset; they remain distinct identities because their
/// full upstream Jinja programs have different null-safety semantics.
#[allow(clippy::too_many_lines)]
pub fn format_qwen3_agent_prompt(
    profile: AgentTemplateProfile,
    prompt: &AgentPrompt,
    options: ChatTemplateOptions,
) -> Result<String> {
    ensure!(
        matches!(
            profile,
            AgentTemplateProfile::Qwen3Current | AgentTemplateProfile::Qwen3Gguf
        ),
        "Qwen3 renderer received a non-Qwen3 agent template profile"
    );
    prompt.validate()?;

    let mut rendered = String::new();
    let first_system = match prompt.messages.first() {
        Some(AgentMessage::System { content }) => Some(content.as_str()),
        _ => None,
    };
    if prompt.tools.is_empty() {
        if let Some(system) = first_system {
            rendered.push_str("<|im_start|>system\n");
            rendered.push_str(system);
            rendered.push_str("<|im_end|>\n");
        }
    } else {
        rendered.push_str("<|im_start|>system\n");
        if let Some(system) = first_system {
            rendered.push_str(system);
            rendered.push_str("\n\n");
        }
        rendered.push_str(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>",
        );
        for tool in &prompt.tools {
            rendered.push('\n');
            rendered.push_str(&serialize_native_tool(tool)?);
        }
        rendered.push_str(
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n",
        );
    }

    let last_query_index = prompt
        .messages
        .iter()
        .rposition(|message| matches!(message, AgentMessage::User { .. }))
        .context("Qwen3 agent history contains no user query")?;
    for (index, message) in prompt.messages.iter().enumerate() {
        match message {
            AgentMessage::System { content } if index != 0 => {
                rendered.push_str("<|im_start|>system\n");
                rendered.push_str(content);
                rendered.push_str("<|im_end|>\n");
            }
            AgentMessage::System { .. } => {}
            AgentMessage::User { content } => {
                rendered.push_str("<|im_start|>user\n");
                rendered.push_str(content);
                rendered.push_str("<|im_end|>\n");
            }
            AgentMessage::Assistant { content, tool_call } => {
                let content = content.as_deref().unwrap_or_default();
                let (reasoning, visible) = content.rsplit_once("</think>").map_or(
                    ("", content),
                    |(reasoning, visible)| {
                        (
                            reasoning
                                .trim_end_matches('\n')
                                .rsplit_once("<think>")
                                .map_or(reasoning, |(_, reasoning)| reasoning)
                                .trim_start_matches('\n'),
                            visible.trim_start_matches('\n'),
                        )
                    },
                );
                rendered.push_str("<|im_start|>assistant\n");
                if index > last_query_index
                    && (index + 1 == prompt.messages.len() || !reasoning.is_empty())
                {
                    rendered.push_str("<think>\n");
                    rendered.push_str(reasoning.trim_matches('\n'));
                    rendered.push_str("\n</think>\n\n");
                    rendered.push_str(visible.trim_start_matches('\n'));
                } else {
                    rendered.push_str(visible);
                }
                if let Some(call) = tool_call {
                    if !visible.is_empty() {
                        rendered.push('\n');
                    }
                    rendered.push_str("<tool_call>\n{\"name\": \"");
                    rendered.push_str(&call.name);
                    rendered.push_str("\", \"arguments\": ");
                    rendered.push_str(&call.arguments);
                    rendered.push_str("}\n</tool_call>");
                }
                rendered.push_str("<|im_end|>\n");
            }
            AgentMessage::AssistantBatch { .. } => {
                anyhow::bail!("multiple calls are unsupported by this renderer")
            }
            AgentMessage::Tool { content, .. } => {
                rendered.push_str("<|im_start|>user\n<tool_response>\n");
                rendered.push_str(content);
                rendered.push_str("\n</tool_response><|im_end|>\n");
            }
        }
    }
    // Text-only and automatic turns both end at an open assistant turn; for
    // `Auto` the serving layer constrains a call lazily once the model opens
    // one with the envelope rendered in the system block above. A forced
    // call's envelope follows the (possibly pre-closed) reasoning block.
    push_qwen3_generation_prompt(&mut rendered, options);
    if let AgentDirective::ForcedSingle { name } = &prompt.directive {
        rendered.push_str("<tool_call>\n{\"name\":\"");
        rendered.push_str(name);
        rendered.push_str("\",\"arguments\":");
    }
    Ok(rendered)
}
