//! Byte-exact parity between the generic Jinja renderer and the reference
//! renders in `tests/fixtures/chat_templates`.
//!
//! The fixtures were produced by `gen_fixtures.py` with Jinja2 and the
//! environment Hugging Face `apply_chat_template` documents; see the
//! `README.md` next to them. Every template, conversation and thinking
//! variant must render byte-for-byte through `JinjaChatTemplate`, and the
//! pinned Rust chat and agent renderers must agree with the generic one on
//! the same inputs, including tool declarations and histories.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use macllm_chat::agent::{
    AgentDirective, AgentMessage, AgentPrompt, AgentTemplateProfile, ToolCall, ToolDefinition,
};
use macllm_chat::chat::{ChatMessage, ChatRole, ChatTemplateOptions};
use macllm_chat::chat_templates::{qwen3, qwen35};
use macllm_chat::jinja::JinjaChatTemplate;
use serde::Deserialize;

mod common;
use common::assert_same_bytes;

const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/chat_templates"
);
const BOS_TOKEN: &str = "";
const EOS_TOKEN: &str = "<|im_end|>";

#[derive(Deserialize)]
struct Manifest {
    cases: BTreeMap<String, Case>,
    thinking: BTreeMap<String, Thinking>,
    families: BTreeMap<String, Family>,
}

#[derive(Deserialize)]
struct Case {
    messages: Vec<ChatMessage>,
    tools: Option<Vec<serde_json::Value>>,
}

#[derive(Deserialize)]
struct Thinking {
    #[serde(default)]
    enable_thinking: Option<bool>,
}

#[derive(Deserialize)]
struct Family {
    templates: Vec<String>,
}

fn manifest() -> Result<Manifest> {
    let path = Path::new(FIXTURES).join("cases.json");
    let source = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&source)
        .with_context(|| format!("invalid fixture manifest at {}", path.display()))
}

fn options(thinking: &Thinking) -> ChatTemplateOptions {
    ChatTemplateOptions {
        enable_thinking: thinking.enable_thinking,
    }
}

/// Lower the fixture into the existing pinned renderer's tool-aware contract.
fn agent_prompt(case: &Case) -> Result<AgentPrompt> {
    let messages = case
        .messages
        .iter()
        .map(|message| {
            Ok(match message.role {
                ChatRole::System => AgentMessage::System {
                    content: message.content.clone(),
                },
                ChatRole::User => AgentMessage::User {
                    content: message.content.clone(),
                },
                ChatRole::Assistant => {
                    let calls = message.tool_calls.as_deref().unwrap_or_default();
                    anyhow::ensure!(calls.len() <= 1, "fixture needs a single-call history");
                    let tool_call = calls
                        .first()
                        .map(|call| {
                            Ok::<_, anyhow::Error>(ToolCall {
                                id: call.id.clone(),
                                name: call.function.name.clone(),
                                arguments: qwen3::python_json(
                                    &call.function.arguments.to_template_value(),
                                )?,
                            })
                        })
                        .transpose()?;
                    AgentMessage::Assistant {
                        content: Some(message.content.clone()),
                        tool_call,
                    }
                }
                ChatRole::Tool => AgentMessage::Tool {
                    content: message.content.clone(),
                    tool_call_id: message
                        .tool_call_id
                        .clone()
                        .context("fixture tool call id")?,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let tools = case
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|declaration| {
            let function = &declaration["function"];
            Ok(ToolDefinition {
                name: function["name"]
                    .as_str()
                    .context("fixture tool name")?
                    .to_owned(),
                description: function["description"].as_str().map(str::to_owned),
                parameters: function["parameters"].clone(),
                strict: function["strict"].as_bool().unwrap_or(false),
                declaration: Some(declaration.clone()),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let directive = if tools.is_empty() {
        AgentDirective::TextOnly
    } else {
        AgentDirective::Auto
    };
    Ok(AgentPrompt {
        messages,
        tools,
        directive,
    })
}

#[test]
fn the_generic_renderer_reproduces_every_reference_render() -> Result<()> {
    let root = Path::new(FIXTURES);
    let manifest = manifest()?;
    let mut template_files = std::fs::read_dir(root.join("templates"))?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    template_files.retain(|path| path.extension().is_some_and(|ext| ext == "jinja"));
    let listed = manifest
        .families
        .values()
        .map(|family| family.templates.len())
        .sum::<usize>();
    assert_eq!(template_files.len(), listed);
    for path in template_files {
        let name = path.file_stem().context("template name")?.to_string_lossy();
        assert_eq!(
            manifest
                .families
                .values()
                .filter(|family| family.templates.iter().any(|template| template == &*name))
                .count(),
            1,
            "template {name} must appear exactly once"
        );
    }
    let mut compared = 0_usize;
    for (family, entry) in &manifest.families {
        for template_name in &entry.templates {
            let path = root
                .join("templates")
                .join(format!("{template_name}.jinja"));
            let source = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let template = JinjaChatTemplate::new(&source, BOS_TOKEN, EOS_TOKEN)?;
            for (case_name, case) in &manifest.cases {
                for (variant, thinking) in &manifest.thinking {
                    let rendered = template.render(
                        &case.messages,
                        case.tools.as_deref(),
                        &options(thinking),
                        true,
                    )?;
                    let path = root.join("rendered").join(family).join(format!(
                        "{template_name}__{case_name}__thinking-{variant}.txt"
                    ));
                    let expected = std::fs::read(&path)
                        .with_context(|| format!("failed to read {}", path.display()))?;
                    assert_same_bytes(
                        &format!("{template_name} / {case_name} / thinking-{variant}"),
                        rendered.as_bytes(),
                        &expected,
                    );
                    compared += 1;
                }
            }
        }
    }
    // Four templates, twelve cases and three thinking variants.
    assert_eq!(compared, 144);
    Ok(())
}

#[test]
fn the_pinned_renderers_and_the_generic_one_agree_on_every_case() -> Result<()> {
    let root = Path::new(FIXTURES);
    let manifest = manifest()?;
    let mut compared = 0_usize;
    for (family, entry) in &manifest.families {
        for template_name in &entry.templates {
            let path = root
                .join("templates")
                .join(format!("{template_name}.jinja"));
            let source = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let template = JinjaChatTemplate::new(&source, BOS_TOKEN, EOS_TOKEN)?;
            for (case_name, case) in &manifest.cases {
                let prompt = agent_prompt(case)?;
                for (variant, thinking) in &manifest.thinking {
                    let options = options(thinking);
                    let pinned = match template_name.as_str() {
                        "qwen3-current" => qwen3::format_qwen3_agent_prompt(
                            AgentTemplateProfile::Qwen3Current,
                            &prompt,
                            options,
                        )?,
                        "qwen3-gguf" => qwen3::format_qwen3_agent_prompt(
                            AgentTemplateProfile::Qwen3Gguf,
                            &prompt,
                            options,
                        )?,
                        "qwen35-official" => qwen35::format_qwen35_agent_prompt(
                            AgentTemplateProfile::Qwen35Official,
                            &prompt,
                            options,
                        )?,
                        "qwen35-unsloth" => qwen35::format_qwen35_agent_prompt(
                            AgentTemplateProfile::Qwen35Unsloth,
                            &prompt,
                            options,
                        )?,
                        other => bail!("no pinned renderer for the template `{other}`"),
                    };
                    let generic =
                        template.render(&case.messages, case.tools.as_deref(), &options, true)?;
                    if case.tools.is_none() {
                        let plain = match template_name.as_str() {
                            "qwen3-current" | "qwen3-gguf" => {
                                qwen3::format_qwen3_chat_messages(&case.messages, options)?
                            }
                            "qwen35-official" => {
                                qwen35::format_qwen35_plain(&case.messages, true, options)?
                            }
                            "qwen35-unsloth" => {
                                qwen35::format_qwen35_plain(&case.messages, false, options)?
                            }
                            other => bail!("no pinned text renderer for `{other}`"),
                        };
                        assert_same_bytes(
                            &format!("{template_name} / {case_name} / thinking-{variant}"),
                            plain.as_bytes(),
                            generic.as_bytes(),
                        );
                    }
                    assert_same_bytes(
                        &format!("{family} / {template_name} / {case_name} / thinking-{variant}"),
                        pinned.as_bytes(),
                        generic.as_bytes(),
                    );
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, 144);
    Ok(())
}
