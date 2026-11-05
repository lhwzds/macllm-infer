//! The pinned template sources of every family, and the parity measurement of
//! every pinned variant against the generic Jinja renderer. Every file under
//! `tests/fixtures/chat_templates/templates/pinned` holds the exact source of
//! one pinned template variant, and every family asserts that its own file
//! still classifies in its own test module.
//!
//! The `parity` module below measures the production path of every pinned
//! variant: both arms run the server's own routing, the pinned arm renders
//! through the family's own renderer and the generic arm through the shared
//! Jinja engine that compiles the very same source. The renders are compared
//! against the golden outcomes in `parity_outcomes.txt`.

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::chat_templates::{qwen3, qwen35};

/// The directory holding the pinned template sources.
const PINNED_TEMPLATES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/chat_templates/templates/pinned"
);

/// The directory holding the shared reference templates.
const REFERENCE_TEMPLATES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/chat_templates/templates"
);

/// The SHA-256 of one file's bytes.
pub(crate) fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Read one pinned template source as text.
pub(crate) fn pinned_text(fixture: &str) -> String {
    let path = Path::new(PINNED_TEMPLATES).join(fixture);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    String::from_utf8(bytes).expect("a template source is UTF-8")
}

/// The names of the `.jinja` files of one template directory.
fn template_names(directory: &str) -> Vec<String> {
    let mut names = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("failed to read {directory}: {error}"))
        .filter_map(|entry| {
            let entry = entry.unwrap_or_else(|error| panic!("failed to read a template: {error}"));
            let path = entry.path();
            if path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension == "jinja")
            {
                Some(entry.file_name().to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect::<Vec<String>>();
    names.sort();
    names
}

/// The measurement of every pinned variant against the generic renderer: both
/// arms are the production paths, the pinned arm is the family's own renderer
/// and the generic arm is the shared Jinja engine that compiles the very same
/// source. Both arms lower the request through the server's own routing, so a
/// refusal is the refusal a client would see.
#[cfg(test)]
mod parity {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fmt::Write as _;
    use std::path::Path;

    use anyhow::{Context, Result, ensure};
    use serde::Deserialize;
    use serde_json::Value;
    use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE};

    use super::{PINNED_TEMPLATES, REFERENCE_TEMPLATES, qwen3, qwen35};
    use crate::{
        agent::AgentPrompt,
        architecture::{
            DecoderLayerSpec, DecoderSpec, ProjectionBias, RopeLayout, RopeSpec, RuntimeSpec,
        },
        chat::{ChatMessage, ChatTemplateOptions},
        engine::EngineInput,
        gguf_model::prompt_control_tokens,
        jinja::JinjaChatTemplate,
        models::{
            GgufPromptTokens, ModelAdapter, gemma::GemmaAdapter, gemma2::Gemma2Adapter,
            gemma3::Gemma3Adapter, granite_moe::GraniteMoeAdapter, lfm2::Lfm2Adapter,
            llama::LlamaAdapter, qwen2::Qwen2Adapter, qwen3::Qwen3Adapter, qwen35::Qwen35Adapter,
            smollm3::SmolLm3Adapter, with_jinja_fallback,
        },
        openai::ChatCompletionRequest,
        server::{SamplingProfile, check_chat_completion_gates, to_engine_request_with_syntax},
    };

    /// The fixture directory of the chat-template references.
    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/chat_templates");
    /// The model name the measurement serves.
    const SERVED_MODEL: &str = "parity";
    /// The completion budget the measurement validates against.
    const MAX_COMPLETION_TOKENS: usize = 128;
    /// One arm of the measurement: the route it took and the render it made.
    type Arm = (&'static str, Option<String>);
    /// The arm refused the request.
    const REFUSED: Arm = ("refused", None);

    /// One pinned template variant of one family: its key, family, source, tokens and modes.
    #[derive(Debug, Deserialize)]
    struct Variant {
        key: String,
        family: String,
        source: String,
        bos: String,
        eos: String,
        special: String,
        user: String,
        thinking: Vec<String>,
        agent: bool,
    }

    /// The variant table of the measurement.
    #[derive(Debug, Deserialize)]
    struct Variants {
        tokens: BTreeMap<String, Vec<String>>,
        variants: Vec<Variant>,
    }

    /// The measured outcome of every variant, case and thinking mode.
    fn measure(variants: &Variants) -> Result<Vec<(String, String)>> {
        let cases = cases()?;
        // The mask holds the dates around today, so a run that crosses midnight masks the same text.
        let clock = clock_dates();
        let mut measured = Vec::new();
        for variant in &variants.variants {
            let (pinned, generic) = renderers(variants, variant)?;
            for (name, case) in &cases {
                for thinking in &variant.thinking {
                    let key = format!("{} {name} {thinking}", variant.key);
                    let body = wire_body(case, thinking);
                    let pinned = run(pinned.as_ref(), &body, &[]);
                    let generic = run(generic.as_ref(), &body, &clock);
                    measured.push((
                        key,
                        format!("{}/{} {}", pinned.0, generic.0, outcome(&pinned, &generic)),
                    ));
                }
            }
        }
        Ok(measured)
    }

    /// The outcome of the two renders of one case.
    fn outcome(pinned: &Arm, generic: &Arm) -> String {
        match (&pinned.1, &generic.1) {
            (None, None) => "x-both".to_owned(),
            (None, Some(render)) => format!("x-pinned {}", digest16(render.as_bytes())),
            (Some(render), None) => format!("x-generic {}", digest16(render.as_bytes())),
            (Some(pinned), Some(generic)) if pinned == generic => "=".to_owned(),
            (Some(pinned), Some(generic)) => {
                let offset = pinned
                    .bytes()
                    .zip(generic.bytes())
                    .position(|(left, right)| left != right)
                    .unwrap_or_else(|| pinned.len().min(generic.len()));
                format!(
                    "! {offset} {} {}",
                    digest16(pinned.as_bytes()),
                    digest16(generic.as_bytes())
                )
            }
        }
    }

    /// The route and the render of one arm: the wire request is parsed, the
    /// handler gates are applied, the request is validated and lowered, and
    /// the engine input is rendered. Every stage may refuse the request.
    fn run(adapter: &dyn ModelAdapter, body: &[u8], clock: &[String]) -> Arm {
        let format = adapter.tool_call_format();
        let defaults = adapter.chat_template_defaults();
        let Ok(mut request) = ChatCompletionRequest::from_json(body) else {
            return REFUSED;
        };
        if check_chat_completion_gates(&mut request, adapter.family(), format).is_err() {
            return REFUSED;
        }
        let Ok(request) =
            request.validate_with_generic(SERVED_MODEL, MAX_COMPLETION_TOKENS, format.is_some())
        else {
            return REFUSED;
        };
        let Ok((engine_request, _output)) = to_engine_request_with_syntax(
            request,
            0,
            SamplingProfile::OpenAi.defaults(true),
            adapter.family() == "qwen35",
            format,
        ) else {
            return REFUSED;
        };
        let options = engine_request.chat_template.or(defaults);
        let chat = |messages: &[ChatMessage]| adapter.format_chat_messages_with(messages, options);
        let agent = |prompt: &AgentPrompt| adapter.format_agent_prompt_with(prompt, options);
        let (route, rendered) = match &engine_request.input {
            EngineInput::Chat(messages) => ("chat", chat(messages)),
            EngineInput::ConstrainedChat { messages, .. } => ("constrained", chat(messages)),
            EngineInput::Agent { prompt, .. } => ("agent", agent(prompt)),
            EngineInput::RawPrompt(_) | EngineInput::TokenIds(_) => return REFUSED,
        };
        let Ok(rendered) = rendered else {
            return REFUSED;
        };
        // The wall clock reaches a render only through the Jinja engine's
        // `strftime_now`, so the mask holds the dates around today and the
        // generic arm masks them. The pinned family renderers print fixed
        // dates of their own, so the pinned arm masks nothing.
        let mut masked = rendered.clone();
        for date in clock {
            masked = masked.replace(date, "<DATE>");
        }
        (route, Some(masked))
    }

    /// The wire body of one case, as a client would send it: the case's own
    /// fields, the served model, and the thinking mode of the measurement. A
    /// mode of `on` or `off` replaces the `chat_template_kwargs` the case
    /// sets, and the mode `default` leaves the case's own value.
    fn wire_body(case: &Value, thinking: &str) -> Vec<u8> {
        let mut body = case.clone();
        let object = body.as_object_mut().expect("a case is a JSON object");
        object.insert("model".to_owned(), Value::String(SERVED_MODEL.to_owned()));
        if let Some(kwargs) = match thinking {
            "on" => Some(serde_json::json!({ "enable_thinking": true })),
            "off" => Some(serde_json::json!({ "enable_thinking": false })),
            _ => None,
        } {
            object.insert("chat_template_kwargs".to_owned(), kwargs);
        }
        serde_json::to_vec(&body).expect("a request body serializes")
    }

    /// The first eight bytes of a render's digest, in hex.
    fn digest16(bytes: &[u8]) -> String {
        let mut text = String::with_capacity(16);
        for byte in &super::digest(bytes)[..8] {
            write!(text, "{byte:02x}").expect("writing to a String cannot fail");
        }
        text
    }

    /// Read one JSON fixture of the measurement.
    fn read_json<T: serde::de::DeserializeOwned>(name: &str) -> Result<T> {
        let path = Path::new(FIXTURES).join(name);
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_str(&source)
            .with_context(|| format!("invalid fixture at {}", path.display()))
    }

    /// The cases every variant renders, read from the wire-shaped file.
    fn cases() -> Result<BTreeMap<String, Value>> {
        let document: Value = read_json("parity_cases.json")?;
        let cases = document["cases"]
            .as_object()
            .context("the case file holds no cases")?;
        Ok(cases
            .iter()
            .map(|(name, case)| (name.clone(), case.clone()))
            .collect())
    }

    /// The golden outcomes of every measured line: the key, the value, the reason.
    fn expected() -> Result<BTreeMap<String, (String, String)>> {
        let path = Path::new(FIXTURES).join("parity_outcomes.txt");
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let mut expected = BTreeMap::new();
        for line in source.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (head, reason) = line.split_once("  #").unwrap_or((line, ""));
            let fields: Vec<&str> = head.split_whitespace().collect();
            ensure!(fields.len() >= 5, "the outcome line `{line}` is too short");
            expected.insert(
                fields[..3].join(" "),
                (fields[3..].join(" "), reason.trim().to_owned()),
            );
        }
        Ok(expected)
    }

    /// The template source of one variant.
    fn template_source(variant: &Variant) -> Result<String> {
        let path = Path::new(FIXTURES).join(&variant.source);
        std::fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))
    }

    /// The two renderers one variant is measured through.
    fn renderers(
        variants: &Variants,
        variant: &Variant,
    ) -> Result<(Box<dyn ModelAdapter>, Box<dyn ModelAdapter>)> {
        let source = template_source(variant)?;
        let mut tokenizer = Tokenizer::new(BPE::default());
        for (name, special) in [(&variant.special, true), (&variant.user, false)] {
            let tokens = variants
                .tokens
                .get(name)
                .map(Vec::as_slice)
                .unwrap_or_default();
            let added = tokens.iter().map(|token| AddedToken::from(token, special));
            if special {
                tokenizer
                    .add_special_tokens(added)
                    .expect("a special token is added");
            } else {
                tokenizer
                    .add_tokens(added)
                    .expect("a user-defined token is added");
            }
        }
        let special = prompt_control_tokens(&tokenizer, Some(&source));
        let tokens = GgufPromptTokens {
            chat_template: Some(&source),
            bos_token: Some(&variant.bos),
            eos_token: Some(&variant.eos),
            special_tokens: &special,
        };
        let mut pinned = adapter(variant, Some(&source))?;
        // The GGUF loader attaches the prompt tokens, so the pinned arm sees it.
        pinned.attach_prompt_tokens(tokens)?;
        let generic = with_jinja_fallback(adapter(variant, None)?, tokens);
        Ok((pinned, generic))
    }

    /// The pinned adapter of one variant, built from its own source.
    fn adapter(variant: &Variant, source: Option<&str>) -> Result<Box<dyn ModelAdapter>> {
        let spec = tiny_spec(&variant.family);
        Ok(match variant.family.as_str() {
            "gemma" => Box::new(GemmaAdapter::from_decoder_spec(spec, source)?),
            "gemma2" => Box::new(Gemma2Adapter::from_decoder_spec(spec, source)?),
            "gemma3" => Box::new(Gemma3Adapter::from_decoder_spec(spec, source)?),
            "granitemoe" => Box::new(GraniteMoeAdapter::from_decoder_spec(spec, source)?),
            "lfm2" => Box::new(Lfm2Adapter::from_decoder_spec(spec, source)?),
            "llama" => Box::new(LlamaAdapter::from_decoder_spec(spec, source)?),
            "qwen2" => Box::new(Qwen2Adapter::from_decoder_spec(spec, source)?),
            "qwen3" => Box::new(Qwen3Adapter::from_decoder_spec(spec, source)?),
            "qwen35" => Box::new(Qwen35Adapter::from_decoder_spec(spec, source)?),
            "smollm3" => Box::new(SmolLm3Adapter::from_decoder_spec(spec, source)?),
            other => anyhow::bail!("the variant table names an unknown family `{other}`"),
        })
    }

    /// A minimal decoder geometry for one family.
    fn tiny_spec(architecture: &str) -> DecoderSpec {
        let rope = RopeSpec {
            theta: 10_000.0,
            layout: RopeLayout::HalfSplit,
            rotary_dim: 4,
            frequency_factors: None,
        };
        DecoderSpec {
            architecture: architecture.to_owned(),
            runtime: RuntimeSpec {
                vocab_size: 32,
                max_context: 16,
                stop_token_ids: vec![1],
            },
            hidden_size: 8,
            intermediate_size: 16,
            num_layers: 1,
            num_query_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            norm_epsilon: 1e-6,
            layer_specs: vec![DecoderLayerSpec::standard(rope, 4)],
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
            tied_embeddings: true,
        }
    }

    /// The dates the wall clock prints around today, in the three formats the
    /// clock-dependent templates print: a run that spans local midnight
    /// renders the new date, so the days around today are masked as well.
    fn clock_dates() -> Vec<String> {
        let now = unsafe { libc::time(std::ptr::null_mut()) };
        let mut dates = Vec::new();
        for format in ["%B %d, %Y", "%d %B %Y", "%d %b %Y"] {
            for offset in [-1, 0, 1] {
                let mut calendar: libc::tm = unsafe { std::mem::zeroed() };
                unsafe { libc::localtime_r(&raw const now, &raw mut calendar) };
                calendar.tm_mday += offset;
                calendar.tm_hour = 12;
                calendar.tm_isdst = -1;
                let epoch = unsafe { libc::mktime(&raw mut calendar) };
                let mut normalized: libc::tm = unsafe { std::mem::zeroed() };
                unsafe { libc::localtime_r(&raw const epoch, &raw mut normalized) };
                let format = std::ffi::CString::new(format).expect("a date format without a NUL");
                let mut buffer = [0_u8; 64];
                let written = unsafe {
                    libc::strftime(
                        buffer.as_mut_ptr().cast(),
                        buffer.len(),
                        format.as_ptr(),
                        &raw const normalized,
                    )
                };
                dates.push(String::from_utf8_lossy(&buffer[..written]).into_owned());
            }
        }
        dates
    }

    /// The measured outcomes match the golden file exactly: every difference
    /// from it fails the test.
    #[test]
    fn the_measured_outcomes_match_the_golden_file() -> Result<()> {
        let measured = measure(&read_json("parity_variants.json")?)?;
        let expected = expected()?;
        let mut failures = Vec::new();
        for (key, value) in &measured {
            match expected.get(key) {
                Some((expected, reason)) if expected == value => {
                    let outcome = value.split_once(' ').map_or(value.as_str(), |(_, o)| o);
                    let names_a_side = outcome.starts_with('!')
                        || outcome.starts_with("x-pinned")
                        || outcome.starts_with("x-generic");
                    if names_a_side && reason.is_empty() {
                        failures.push(format!("  {key}: the outcome `{value}` needs a reason"));
                    }
                }
                Some((expected, reason)) => failures.push(format!(
                    "  {key}: expected `{expected}`, measured `{value}` ({reason})"
                )),
                None => failures.push(format!("  {key}: missing from the outcomes file")),
            }
        }
        for (key, (value, _)) in &expected {
            if !measured.iter().any(|(measured, _)| measured == key) {
                failures.push(format!("  {key}: not measured any more (file: `{value}`)"));
            }
        }
        let failures = failures.join("\n");
        assert!(
            failures.is_empty(),
            "the measured outcomes differ from the golden file:\n{failures}"
        );
        Ok(())
    }

    /// The clock mask holds the days around the measurement.
    #[test]
    fn the_clock_mask_holds_the_days_around_today() -> Result<()> {
        let dates = clock_dates();
        ensure!(
            dates.len() == 9,
            "the mask holds three days in three formats"
        );
        ensure!(
            dates.iter().collect::<BTreeSet<_>>().len() == 9,
            "every masked date is a text of its own"
        );
        let options = ChatTemplateOptions::default();
        let clock = JinjaChatTemplate::new("{{ strftime_now('%B %d, %Y') }}", "", "")?;
        ensure!(
            dates.contains(&clock.render(&[], None, &options, true)?),
            "the mask holds today's date"
        );
        Ok(())
    }

    /// The SmolLM3 template differs from its pinned renderer only by its
    /// generation markers: the engine renders the template without the
    /// markers, which print nothing, and the two renders agree.
    #[test]
    fn the_smollm3_template_differs_from_its_pinned_renderer_only_by_its_generation_markers()
    -> Result<()> {
        let variants: Variants = read_json("parity_variants.json")?;
        let variant = variants.variants.iter().find(|v| v.family == "smollm3");
        let variant = variant.context("the variant table names no SmolLM3 variant")?;
        let source = template_source(variant)?;
        let compiled = JinjaChatTemplate::new(&source, &variant.bos, &variant.eos);
        assert!(compiled.is_err(), "the engine refuses the real source");
        let stripped = source
            .replace("{% generation %}", "")
            .replace("{% endgeneration %}", "");
        let template = JinjaChatTemplate::new(&stripped, &variant.bos, &variant.eos)?;
        let adapter = adapter(variant, Some(&source))?;
        let options = ChatTemplateOptions::default();
        let cases = cases()?;
        for name in ["smollm3_system_override", "smollm3_system_override_history"] {
            let case = cases.get(name).context("no such case")?;
            let request = ChatCompletionRequest::from_json(&wire_body(case, "default"))?;
            let request =
                request.validate_with_generic(SERVED_MODEL, MAX_COMPLETION_TOKENS, false)?;
            let messages = request.into_text_runtime_messages()?;
            let pinned = adapter.format_chat_messages_with(&messages, options)?;
            let rendered = template.render(&messages, None, &options, true)?;
            assert_eq!(pinned, rendered, "{name}");
        }
        Ok(())
    }

    /// The forced tool-call suffixes are pinned per agent profile: a forced
    /// call renders the automatic turn and then the bytes the pinned renderer
    /// appends to it. The Qwen 3 profiles are the ones whose routing serves a
    /// forced call.
    #[test]
    fn the_forced_tool_call_suffixes_are_pinned() -> Result<()> {
        let variants = read_json::<Variants>("parity_variants.json")?;
        let cases = cases()?;
        let forced = cases.get("forced_tool_call").context("no forced case")?;
        let automatic = cases
            .get("automatic_tool_call")
            .context("no automatic case")?;
        let forced_body = wire_body(forced, "default");
        let automatic_body = wire_body(automatic, "default");
        let profiles = variants
            .variants
            .iter()
            .filter(|variant| variant.agent && variant.family == "qwen3")
            .collect::<Vec<_>>();
        ensure!(
            !profiles.is_empty(),
            "the variant table names no Qwen3 agent profile"
        );
        for variant in profiles {
            let (adapter, _) = renderers(&variants, variant)?;
            let forced = run(adapter.as_ref(), &forced_body, &[]);
            let Some(forced) = forced.1 else {
                anyhow::bail!("{}: the forced turn must render", variant.key);
            };
            let automatic = run(adapter.as_ref(), &automatic_body, &[]);
            let Some(automatic) = automatic.1 else {
                anyhow::bail!("{}: the automatic turn must render", variant.key);
            };
            let suffix = forced
                .strip_prefix(&automatic)
                .context("the forced render must extend the automatic render")?;
            let name = Path::new(&variant.source)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .replace(".jinja", ".txt");
            let expected = std::fs::read(Path::new(FIXTURES).join("forced_calls").join(name))?;
            assert_eq!(
                suffix.as_bytes(),
                expected.as_slice(),
                "{}: the forced tool-call suffix changed",
                variant.key
            );
        }
        Ok(())
    }

    /// The Qwen 3 and Qwen 3.5 templates still classify to their variants.
    #[test]
    fn the_qwen_templates_still_classify_to_their_variants() -> Result<()> {
        for variant in read_json::<Variants>("parity_variants.json")?
            .variants
            .iter()
            .filter(|v| matches!(v.family.as_str(), "qwen3" | "qwen35"))
        {
            let text = template_source(variant)?;
            let classified = match variant.family.as_str() {
                "qwen3" => format!("{:?}", qwen3::classify_chat_template(Some(&text))),
                _ => format!("{:?}", qwen35::classify_chat_template(Some(&text))),
            };
            let name = variant.key.split('/').nth(1).unwrap_or_default();
            assert_eq!(classified, format!("Some({name})"), "{}", variant.key);
        }
        Ok(())
    }

    /// The template directories hold exactly the sources of the variants.
    #[test]
    fn the_template_directories_hold_exactly_the_sources_of_the_variants() -> Result<()> {
        let variants: Variants = read_json("parity_variants.json")?;
        for (directory, parent) in [
            (PINNED_TEMPLATES, "templates/pinned"),
            (REFERENCE_TEMPLATES, "templates"),
        ] {
            let mut expected: Vec<String> = variants
                .variants
                .iter()
                .filter(|variant| Path::new(&variant.source).parent() == Some(Path::new(parent)))
                .filter_map(|variant| {
                    Path::new(&variant.source)
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                })
                .collect();
            expected.sort();
            assert_eq!(
                super::template_names(directory),
                expected,
                "{directory} holds exactly the sources of the variants"
            );
        }
        Ok(())
    }
}
