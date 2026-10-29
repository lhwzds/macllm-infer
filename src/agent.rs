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

/// One prompt submitted through a model adapter's pinned native agent
/// template profile.
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
    /// text, or emit text followed by at most one native call to a declared
    /// tool; a lazy grammar constrains the call once it opens.
    Auto,
}

/// A protocol-neutral function declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: bool,
}

/// One function call retained in conversation history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Canonical JSON text for one arguments object.
    pub arguments: String,
}

/// The deliberately small history subset supported by the first native tool
/// runtime.  Validation permits at most one call and exactly one matching
/// response per assistant tool turn.  A tool-call turn may also carry the
/// visible text (including reasoning) the model produced before its call.
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
    Tool {
        content: String,
        tool_call_id: String,
    },
}
