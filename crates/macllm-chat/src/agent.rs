//! Protocol-independent contracts for native agent prompt rendering.
//!
//! HTTP wire types are deliberately kept out of this module.  A serving
//! protocol must validate and lower its own request into these types before a
//! model adapter may render a prompt.

use std::collections::HashSet;

use anyhow::{Result, bail, ensure};
use serde_json::Value;

const NATIVE_CONTROL_STRINGS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<tools>",
    "</tools>",
];

fn reject_native_control(label: &str, value: &str) -> Result<()> {
    if let Some(control) = NATIVE_CONTROL_STRINGS
        .iter()
        .find(|control| value.contains(**control))
    {
        bail!("{label} contains reserved native control string `{control}`");
    }
    Ok(())
}

/// Check every string of a JSON value against the reserved controls.
///
/// The label names the part of the prompt the value belongs to, so a
/// declaration error does not read like an argument error.
fn check_json_strings(
    value: &Value,
    label: &str,
    check: &impl Fn(&str, &str) -> Result<()>,
) -> Result<()> {
    match value {
        Value::String(text) => check(label, text)?,
        Value::Array(values) => {
            for value in values {
                check_json_strings(value, label, check)?;
            }
        }
        Value::Object(values) => {
            for (key, value) in values {
                check(label, key)?;
                check_json_strings(value, label, check)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// One prompt submitted through a model adapter's agent template renderer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentPrompt {
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<ToolDefinition>,
    pub directive: AgentDirective,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AgentHistoryState<'a> {
    NeedUser,
    NeedAssistant,
    NeedTool { id: &'a str },
    MayGenerate,
}

impl AgentPrompt {
    /// Validate a generic history while preserving batches for template rendering.
    pub fn validate_generic(&self) -> Result<()> {
        let mut messages = Vec::new();
        let mut iter = self.messages.iter();
        while let Some(message) = iter.next() {
            if let AgentMessage::AssistantBatch {
                content,
                tool_calls,
            } = message
            {
                ensure!(!tool_calls.is_empty(), "tool call batch must not be empty");
                let mut replies = Vec::new();
                for _ in tool_calls {
                    let Some(AgentMessage::Tool {
                        content,
                        tool_call_id,
                    }) = iter.next()
                    else {
                        bail!("tool call batch must be followed by its tool responses");
                    };
                    replies.push((tool_call_id, content));
                }
                let mut ids = HashSet::new();
                for (index, call) in tool_calls.iter().enumerate() {
                    ensure!(ids.insert(&call.id), "duplicate tool call id");
                    let (_, reply) =
                        replies
                            .iter()
                            .find(|(id, _)| *id == &call.id)
                            .ok_or_else(|| {
                                anyhow::anyhow!("tool response id does not match its call")
                            })?;
                    messages.push(AgentMessage::Assistant {
                        content: if index == 0 { content.clone() } else { None },
                        tool_call: Some(call.clone()),
                    });
                    messages.push(AgentMessage::Tool {
                        content: (*reply).clone(),
                        tool_call_id: call.id.clone(),
                    });
                }
            } else {
                messages.push(message.clone());
            }
        }
        Self {
            messages,
            tools: self.tools.clone(),
            directive: self.directive.clone(),
        }
        .validate()
    }

    /// Reject the reserved control strings in untrusted history.
    ///
    /// The reserved list holds the strings a client must not forge: the
    /// tokenizer's special tokens and the markers of the template.
    pub fn reject_generic_controls(&self, reserved: &[String]) -> Result<()> {
        let check = |label: &str, value: &str| -> Result<()> {
            for control in reserved.iter().map(String::as_str) {
                if !control.is_empty() && value.contains(control) {
                    bail!("{label} contains reserved native control string `{control}`");
                }
            }
            Ok(())
        };
        for tool in &self.tools {
            check("tool name", &tool.name)?;
            if let Some(description) = &tool.description {
                check("tool description", description)?;
            }
            check_json_strings(&tool.parameters, "tool declaration", &check)?;
            if let Some(declaration) = &tool.declaration {
                check_json_strings(declaration, "tool declaration", &check)?;
            }
        }
        for message in &self.messages {
            match message {
                AgentMessage::System { content } | AgentMessage::User { content } => {
                    check("content", content)?;
                }
                AgentMessage::Tool {
                    content,
                    tool_call_id,
                } => {
                    check("content", content)?;
                    check("tool call id", tool_call_id)?;
                }
                AgentMessage::Assistant { content, tool_call } => {
                    if let Some(content) = content {
                        check("assistant content", content)?;
                    }
                    if let Some(call) = tool_call {
                        check("tool call id", &call.id)?;
                        check("tool arguments", &call.arguments)?;
                        check_json_strings(
                            &serde_json::from_str(&call.arguments)?,
                            "tool arguments",
                            &check,
                        )?;
                    }
                }
                AgentMessage::AssistantBatch {
                    content,
                    tool_calls,
                } => {
                    if let Some(content) = content {
                        check("assistant content", content)?;
                    }
                    for call in tool_calls {
                        check("tool call id", &call.id)?;
                        check("tool arguments", &call.arguments)?;
                        check_json_strings(
                            &serde_json::from_str(&call.arguments)?,
                            "tool arguments",
                            &check,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Validate the common single-call history grammar before any
    /// family-specific renderer observes it.
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.messages.is_empty(),
            "agent messages must not be empty"
        );

        let mut tool_names = HashSet::with_capacity(self.tools.len());
        for tool in &self.tools {
            ensure!(!tool.name.is_empty(), "tool names must not be empty");
            reject_native_control("tool name", &tool.name)?;
            if let Some(description) = &tool.description {
                reject_native_control("tool description", description)?;
            }
            reject_native_control("tool parameters", &tool.parameters.to_string())?;
            if let Some(declaration) = &tool.declaration {
                reject_native_control("tool declaration", &declaration.to_string())?;
            }
            ensure!(
                tool_names.insert(tool.name.as_str()),
                "duplicate tool name `{}`",
                tool.name
            );
            ensure!(
                tool.parameters.is_object(),
                "tool parameters must be a JSON object"
            );
        }
        match &self.directive {
            AgentDirective::TextOnly => {}
            AgentDirective::ForcedSingle { name } => {
                reject_native_control("forced tool name", name)?;
                ensure!(
                    self.tools.iter().any(|tool| tool.name == *name),
                    "forced tool `{name}` is not declared"
                );
            }
            AgentDirective::Auto => ensure!(
                !self.tools.is_empty(),
                "automatic tool choice requires at least one declared tool"
            ),
        }

        let mut state = AgentHistoryState::NeedUser;
        for (index, message) in self.messages.iter().enumerate() {
            match message {
                AgentMessage::System { content } => {
                    reject_native_control("system content", content)?;
                    ensure!(index == 0, "system message is only valid at index zero");
                    ensure!(
                        state == AgentHistoryState::NeedUser,
                        "system message is out of order"
                    );
                }
                AgentMessage::User { content } => {
                    reject_native_control("user content", content)?;
                    ensure!(
                        state == AgentHistoryState::NeedUser,
                        "user message is out of order"
                    );
                    state = AgentHistoryState::NeedAssistant;
                }
                AgentMessage::Assistant { content, tool_call } => {
                    if let Some(content) = content {
                        reject_native_control("assistant content", content)?;
                    }
                    ensure!(
                        matches!(
                            state,
                            AgentHistoryState::NeedAssistant | AgentHistoryState::MayGenerate
                        ),
                        "assistant message is out of order"
                    );
                    if let Some(call) = tool_call {
                        ensure!(!call.id.is_empty(), "tool call id must not be empty");
                        ensure!(!call.name.is_empty(), "tool call name must not be empty");
                        reject_native_control("tool call id", &call.id)?;
                        reject_native_control("tool call name", &call.name)?;
                        reject_native_control("tool arguments", &call.arguments)?;
                        let arguments: Value = serde_json::from_str(&call.arguments)?;
                        ensure!(
                            arguments.is_object(),
                            "tool arguments must encode an object"
                        );
                        state = AgentHistoryState::NeedTool { id: &call.id };
                    } else {
                        ensure!(content.is_some(), "text assistant message needs content");
                        state = AgentHistoryState::NeedUser;
                    }
                }
                AgentMessage::AssistantBatch { .. } => {
                    bail!("multiple calls are unsupported by this renderer")
                }
                AgentMessage::Tool {
                    content,
                    tool_call_id,
                } => {
                    reject_native_control("tool response content", content)?;
                    reject_native_control("tool response id", tool_call_id)?;
                    let AgentHistoryState::NeedTool { id } = state else {
                        bail!("tool response is out of order");
                    };
                    ensure!(
                        tool_call_id == id,
                        "tool response id does not match its call"
                    );
                    state = AgentHistoryState::MayGenerate;
                }
            }
        }
        ensure!(
            matches!(
                state,
                AgentHistoryState::NeedAssistant | AgentHistoryState::MayGenerate
            ),
            "agent history must end at an assistant generation boundary"
        );
        Ok(())
    }
}

/// A model-native rendering profile whose identity has been byte-verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentTemplateProfile {
    Qwen3Current,
    Qwen3Gguf,
    Qwen35Official,
    Qwen35Unsloth,
}

/// Whether the next assistant turn is ordinary text, exactly one named
/// function call, or the model's choice between the two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentDirective {
    TextOnly,
    /// The prompt ends inside the function envelope so the model generates
    /// only its JSON arguments.
    ForcedSingle {
        name: String,
    },
    /// The prompt ends at an open assistant turn. The model may answer in
    /// text or emit calls to declared tools. Pinned renderers allow at most
    /// one call and use a lazy grammar once it opens; generic calls are
    /// validated after parsing.
    Auto,
}

/// The bytes a forced call opens with: the tool-call envelope of the Hermes
/// format and the opening quote of the forced tool's name.
pub const FORCED_TOOL_CALL_PREFIX: &str = "<tool_call>\n{\"name\":\"";
/// The bytes a forced call closes the name with: the closing quote and the
/// arguments the model continues with.
pub const FORCED_TOOL_CALL_ARGUMENTS: &str = "\",\"arguments\":";

/// A protocol-neutral function declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: bool,
    /// Original ordered wire declaration, when supplied by a client.
    pub declaration: Option<Value>,
}

/// One function call retained in conversation history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Validated JSON text for one arguments object.
    pub arguments: String,
}

/// A tool history turn. Pinned renderers permit one call per turn; generic
/// templates additionally accept complete batches of historical calls. A call
/// turn may carry visible text and reasoning before the calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        tool_call: Option<ToolCall>,
    },
    /// A batch of calls from a single assistant turn, for generic templates.
    AssistantBatch {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        content: String,
        tool_call_id: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_control_checks_every_message_and_decoded_arguments() {
        let marker = "<|eot_id|>";
        let call = ToolCall {
            id: "id".into(),
            name: "f".into(),
            arguments: r#"{"x":"\u003c|eot_id|>"}"#.into(),
        };
        for message in [
            AgentMessage::System {
                content: marker.into(),
            },
            AgentMessage::User {
                content: marker.into(),
            },
            AgentMessage::Tool {
                content: marker.into(),
                tool_call_id: "id".into(),
            },
            AgentMessage::Assistant {
                content: Some(marker.into()),
                tool_call: None,
            },
            AgentMessage::Assistant {
                content: None,
                tool_call: Some(call.clone()),
            },
            AgentMessage::AssistantBatch {
                content: None,
                tool_calls: vec![call.clone()],
            },
        ] {
            let prompt = AgentPrompt {
                messages: vec![message],
                tools: vec![],
                directive: AgentDirective::TextOnly,
            };
            assert!(
                prompt
                    .reject_generic_controls(&[marker.to_owned()])
                    .is_err()
            );
        }
    }

    #[test]
    fn generic_checks_all_declaration_strings_and_both_history_ids() {
        let marker = "[INST]";
        let check = |prompt: &AgentPrompt| {
            prompt
                .reject_generic_controls(&[marker.to_owned()])
                .is_err()
        };
        let base = AgentPrompt {
            messages: vec![AgentMessage::User {
                content: "ok".into(),
            }],
            tools: vec![ToolDefinition {
                name: "f".into(),
                description: None,
                parameters: serde_json::json!({"type":"object"}),
                strict: false,
                declaration: None,
            }],
            directive: AgentDirective::TextOnly,
        };
        let mut prompt = base.clone();
        prompt.tools[0].name = marker.into();
        assert!(check(&prompt));
        prompt = base.clone();
        prompt.tools[0].parameters = serde_json::json!({"nested":[{"key":marker}]});
        assert!(check(&prompt));
        prompt = base.clone();
        prompt.tools[0].declaration = Some(serde_json::json!({marker:"x"}));
        assert!(check(&prompt));
        for message in [
            AgentMessage::Assistant {
                content: None,
                tool_call: Some(ToolCall {
                    id: marker.into(),
                    name: "f".into(),
                    arguments: "{}".into(),
                }),
            },
            AgentMessage::AssistantBatch {
                content: None,
                tool_calls: vec![ToolCall {
                    id: marker.into(),
                    name: "f".into(),
                    arguments: "{}".into(),
                }],
            },
            AgentMessage::Tool {
                content: "ok".into(),
                tool_call_id: marker.into(),
            },
        ] {
            prompt = base.clone();
            prompt.messages = vec![message];
            assert!(check(&prompt));
        }
    }

    #[test]
    fn declaration_and_argument_errors_carry_their_own_label() {
        let marker = "[INST]";
        let base = AgentPrompt {
            messages: vec![AgentMessage::User {
                content: "ok".into(),
            }],
            tools: vec![ToolDefinition {
                name: "f".into(),
                description: None,
                parameters: serde_json::json!({"type":"object"}),
                strict: false,
                declaration: None,
            }],
            directive: AgentDirective::TextOnly,
        };
        // A declaration key or value belongs to the declaration, not to the
        // arguments of a call.
        let mut prompt = base.clone();
        prompt.tools[0].parameters = serde_json::json!({"nested":[{"key":marker}]});
        let error = prompt
            .reject_generic_controls(&[marker.to_owned()])
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("tool declaration"), "{message}");
        assert!(!message.contains("tool arguments"), "{message}");

        let mut prompt = base.clone();
        prompt.tools[0].declaration = Some(serde_json::json!({marker:"x"}));
        let error = prompt
            .reject_generic_controls(&[marker.to_owned()])
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("tool declaration"), "{message}");

        // The arguments of a call keep the argument label.
        let mut prompt = base.clone();
        prompt.messages = vec![AgentMessage::Assistant {
            content: None,
            tool_call: Some(ToolCall {
                id: "id".into(),
                name: "f".into(),
                arguments: serde_json::json!({"key":marker}).to_string(),
            }),
        }];
        let error = prompt
            .reject_generic_controls(&[marker.to_owned()])
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("tool arguments"), "{message}");
        assert!(!message.contains("tool declaration"), "{message}");
    }

    fn batch_prompt() -> AgentPrompt {
        AgentPrompt {
            messages: vec![
                AgentMessage::User {
                    content: "Weather in both cities?".into(),
                },
                AgentMessage::AssistantBatch {
                    content: Some("Checking both.".into()),
                    tool_calls: ["a", "b"]
                        .into_iter()
                        .map(|id| ToolCall {
                            id: id.into(),
                            name: "weather".into(),
                            arguments: r#"{"city":"東京"}"#.into(),
                        })
                        .collect(),
                },
                AgentMessage::Tool {
                    content: "rain".into(),
                    tool_call_id: "b".into(),
                },
                AgentMessage::Tool {
                    content: "sun".into(),
                    tool_call_id: "a".into(),
                },
            ],
            tools: Vec::new(),
            directive: AgentDirective::TextOnly,
        }
    }

    #[test]
    fn generic_history_accepts_complete_batches_without_reordering_them() {
        let prompt = batch_prompt();
        let original = prompt.clone();
        prompt.validate_generic().unwrap();
        assert_eq!(prompt, original);
        assert!(prompt.validate().is_err());
    }

    #[test]
    fn generic_history_requires_one_matching_response_per_call() {
        for reply_ids in [vec!["a"], vec!["a", "a"], vec!["a", "unknown"]] {
            let mut prompt = batch_prompt();
            prompt.messages.truncate(2);
            prompt
                .messages
                .extend(reply_ids.into_iter().map(|id| AgentMessage::Tool {
                    content: "result".into(),
                    tool_call_id: id.into(),
                }));
            assert!(prompt.validate_generic().is_err());
        }
    }

    #[test]
    fn generic_history_validates_every_call_in_a_batch() {
        for invalid in ["duplicate", "empty_id", "invalid_arguments", "empty_batch"] {
            let mut prompt = batch_prompt();
            let AgentMessage::AssistantBatch { tool_calls, .. } = &mut prompt.messages[1] else {
                unreachable!();
            };
            match invalid {
                "duplicate" => tool_calls[1].id = "a".into(),
                "empty_id" => tool_calls[1].id.clear(),
                "invalid_arguments" => tool_calls[1].arguments = "[]".into(),
                "empty_batch" => tool_calls.clear(),
                _ => unreachable!(),
            }
            assert!(prompt.validate_generic().is_err(), "accepted {invalid}");
        }
    }
}
