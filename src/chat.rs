use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// A text-only role supported by the model chat templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// One text message in a validated chat history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
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
}

/// Chat-template keyword options, mirroring the `chat_template_kwargs`
/// convention of vLLM, SGLang and llama.cpp. `None` keeps the template's own
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
        assert!(
            serde_json::from_str::<ChatMessage>(r#"{"role":"tool","content":"hello"}"#).is_err()
        );
        Ok(())
    }
}
