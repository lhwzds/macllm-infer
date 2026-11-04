use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize, de::Deserializer, ser::Serializer};

/// A role supported by the model chat templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    /// A tool response answering the assistant tool call before it. The
    /// text-chat contract rejects it; chat templates render it.
    Tool,
}

impl ChatRole {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// Tool-call arguments retained as JSON text, so a message stays cheap to
/// clone and comparable. Templates observe an ordered JSON object; see
/// [`ChatToolArguments::to_template_value`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChatToolArguments(String);

impl ChatToolArguments {
    /// Retain the JSON text of one call's arguments.
    #[must_use]
    pub fn from_json_text(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    /// The retained JSON text.
    #[must_use]
    pub fn as_json_text(&self) -> &str {
        &self.0
    }

    /// The value a chat template observes: the ordered JSON object encoded by
    /// the retained text, or the text itself when it is not an object.
    /// `serde_json` preserves insertion order, so argument order survives.
    #[must_use]
    pub fn to_template_value(&self) -> serde_json::Value {
        serde_json::from_str::<serde_json::Value>(&self.0)
            .ok()
            .filter(serde_json::Value::is_object)
            .unwrap_or_else(|| serde_json::Value::String(self.0.clone()))
    }
}

impl Serialize for ChatToolArguments {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ChatToolArguments {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // The OpenAI wire format sends a JSON string; Hugging Face
        // conversations keep the object itself.
        match serde_json::Value::deserialize(deserializer)? {
            serde_json::Value::String(text) => Ok(Self(text)),
            object @ serde_json::Value::Object(_) => Ok(Self(object.to_string())),
            other => Err(serde::de::Error::custom(format!(
                "tool call arguments must be a JSON string or object, got {other}"
            ))),
        }
    }
}

/// The `function` member of one retained tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatToolFunction {
    pub name: String,
    pub arguments: ChatToolArguments,
}

/// One function call retained in an assistant message, shaped the way a
/// Hugging Face chat template iterates it (`id`, `type`, `function`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type", default = "default_tool_call_type")]
    pub kind: String,
    pub function: ChatToolFunction,
}

fn default_tool_call_type() -> String {
    "function".to_owned()
}

impl ChatToolCall {
    /// Retain one call; every retained call has type `function`.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: ChatToolArguments,
    ) -> Self {
        Self {
            id: id.into(),
            kind: default_tool_call_type(),
            function: ChatToolFunction {
                name: name.into(),
                arguments,
            },
        }
    }
}

/// One message in a validated chat history. `role` and `content` cover every
/// text turn; the remaining members carry the tool and reasoning payload that
/// a generic chat template renders when it is present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// The participant name supplied by the caller, as in the wire format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Reasoning kept apart from the visible answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// The calls an assistant turn made.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
    /// The call a tool turn answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }

    /// A tool response answering `tool_call_id`.
    #[must_use]
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            tool_call_id: Some(tool_call_id.into()),
            ..Self::new(ChatRole::Tool, content)
        }
    }

    /// The calls this assistant turn made, consumed by chat templates.
    #[must_use]
    pub fn with_tool_calls(mut self, tool_calls: Vec<ChatToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }
}

/// Chat-template keyword options, mirroring the `chat_template_kwargs`
/// convention of vLLM, `SGLang` and llama.cpp. `None` keeps the template's own
/// default. Adapters reject any option they do not implement instead of
/// silently ignoring it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChatTemplateOptions {
    /// Qwen3's `enable_thinking`: `false` pre-fills an empty reasoning block
    /// in the generation prompt so the model answers directly.
    pub enable_thinking: Option<bool>,
}

impl ChatTemplateOptions {
    /// Fill every option the request left unset from server-wide defaults.
    #[must_use]
    pub const fn or(self, defaults: Self) -> Self {
        Self {
            enable_thinking: match self.enable_thinking {
                Some(enabled) => Some(enabled),
                None => defaults.enable_thinking,
            },
        }
    }

    /// Whether every option keeps the template default.
    #[must_use]
    pub const fn is_default(self) -> bool {
        self.enable_thinking.is_none()
    }

    /// Fail closed for an adapter that implements no template options.
    pub fn ensure_default(self, family: &str) -> Result<()> {
        ensure!(
            self.enable_thinking.is_none(),
            "chat template option `enable_thinking` is not supported by the {family} adapter"
        );
        Ok(())
    }
}

/// Validate the common text-chat subset supported by every model adapter.
///
/// A conversation may begin with one system message. The remaining messages
/// must alternate user/assistant, beginning and ending with a user message so
/// the rendered prompt always requests the next assistant turn.
pub fn validate_chat_messages(messages: &[ChatMessage]) -> Result<()> {
    ensure!(!messages.is_empty(), "chat messages must not be empty");

    let first_turn = usize::from(messages[0].role == ChatRole::System);
    ensure!(
        first_turn < messages.len(),
        "chat messages must include a user turn after the system message"
    );

    for (turn, message) in messages[first_turn..].iter().enumerate() {
        let expected = if turn.is_multiple_of(2) {
            ChatRole::User
        } else {
            ChatRole::Assistant
        };
        ensure!(
            message.role == expected,
            "chat message {} must have role `{}`, got `{}`",
            first_turn + turn,
            expected.as_str(),
            message.role.as_str()
        );
    }

    ensure!(
        messages
            .last()
            .is_some_and(|message| message.role == ChatRole::User),
        "chat messages must end with a user turn"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    fn accepts_one_or_many_complete_user_turns() -> Result<()> {
        validate_chat_messages(&[ChatMessage::user("hello")])?;
        validate_chat_messages(&[
            ChatMessage::system("be concise"),
            ChatMessage::user("first"),
            ChatMessage::assistant("answer"),
            ChatMessage::user("follow-up"),
        ])?;
        Ok(())
    }

    #[test]
    fn rejects_incomplete_or_misordered_histories() {
        for messages in [
            vec![],
            vec![ChatMessage::system("system only")],
            vec![ChatMessage::assistant("answer first")],
            vec![ChatMessage::user("question"), ChatMessage::user("again")],
            vec![
                ChatMessage::user("question"),
                ChatMessage::assistant("answer"),
            ],
            vec![
                ChatMessage::system("one"),
                ChatMessage::system("two"),
                ChatMessage::user("question"),
            ],
        ] {
            assert!(validate_chat_messages(&messages).is_err());
        }
    }

    #[test]
    fn serde_uses_openai_role_names_and_rejects_unknown_roles() -> Result<()> {
        let message: ChatMessage =
            serde_json::from_str(r#"{"role":"assistant","content":"hello"}"#)?;
        assert_eq!(message, ChatMessage::assistant("hello"));
        assert!(serde_json::from_str::<ChatMessage>(r#"{"role":"tool"}"#).is_err());
        assert!(
            serde_json::from_str::<ChatMessage>(r#"{"role":"user","content":"hi","extra":1}"#)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn retains_tool_turns_and_both_argument_spellings() -> Result<()> {
        let call: ChatMessage = serde_json::from_str(
            r#"{"role":"assistant","content":"","tool_calls":[{"id":"call_1",
               "type":"function","function":{"name":"shell",
               "arguments":"{\"command\": \"ls\"}"}}]}"#,
        )?;
        let first = call
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
            .context("one retained call")?;
        assert_eq!(
            (first.id.as_str(), first.kind.as_str()),
            ("call_1", "function")
        );
        assert_eq!(first.function.name, "shell");
        assert_eq!(
            first.function.arguments.as_json_text(),
            "{\"command\": \"ls\"}"
        );
        assert_eq!(
            first.function.arguments.to_template_value(),
            serde_json::json!({"command": "ls"})
        );

        let typed: ChatMessage = serde_json::from_str(
            r#"{"role":"assistant","content":"","reasoning_content":"think",
               "tool_calls":[{"id":"call_2","type":"function",
               "function":{"name":"shell","arguments":{"command": "ls"}}}]}"#,
        )?;
        assert_eq!(
            typed
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.first())
                .map(|call| call.function.arguments.to_template_value()),
            Some(serde_json::json!({"command": "ls"}))
        );
        assert_eq!(typed.reasoning_content.as_deref(), Some("think"));

        let response = ChatMessage::tool("ok", "call_1");
        assert_eq!(response.role.as_str(), "tool");
        assert_eq!(response.tool_call_id.as_deref(), Some("call_1"));
        assert!(
            serde_json::from_str::<ChatMessage>(
                r#"{"role":"assistant","content":"","tool_calls":[{"id":"c",
                   "type":"function","function":{"name":"n","arguments":7}}]}"#
            )
            .is_err()
        );
        Ok(())
    }
}
