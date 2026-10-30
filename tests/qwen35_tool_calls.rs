use std::sync::Arc;

use anyhow::Result;
use candle_core::{Device, Tensor};
use macllm_infer::{
    grammar::{
        CompiledJsonSchema, CompiledToolCalls, GrammarStatus, LazyToolCallConstraint,
        ToolCallGrammar,
    },
    sampling::{LogitsConstraint, Sampler, SamplingConfig},
    tool_call::parse_xml_tool_call,
};
use serde_json::{Value, json};
use tokenizers::{Tokenizer, decoders::byte_fallback::ByteFallback, models::bpe::BPE};

fn compile_calls(parameters: &Value) -> Result<CompiledToolCalls> {
    CompiledToolCalls::qwen35([(
        "run".to_owned(),
        CompiledJsonSchema::compile_closed(parameters)?,
    )])
}

fn envelope(parameters: &str) -> String {
    format!("\n<function=run>\n{parameters}</function>\n</tool_call>")
}

fn parameter(name: &str, text: &str) -> String {
    format!("<parameter={name}>\n{text}\n</parameter>\n")
}

#[test]
fn xml_value_grammar_and_conversion_table() -> Result<()> {
    for (schema, text, expected) in [
        (json!({"type":"string"}), "hello", json!("hello")),
        (json!({"type":"string"}), "", json!("")),
        (json!({"type":"string"}), "a\n\nb\n", json!("a\n\nb\n")),
        (
            json!({"type":"string"}),
            "  🚀 <tag>&amp;\"\\\n</parameteX>  ",
            json!("  🚀 <tag>&amp;\"\\\n</parameteX>  "),
        ),
        (
            json!({"type":"string", "enum":["red", "blue\ngreen"]}),
            "blue\ngreen",
            json!("blue\ngreen"),
        ),
        (
            json!({"type":"string", "const":"True"}),
            "True",
            json!("True"),
        ),
        (json!({"enum":["red", "blue"]}), "blue", json!("blue")),
        (json!({"const":true}), "True", json!(true)),
        (json!({"type":"integer"}), "-12", json!(-12)),
        (json!({"type":"number"}), "-1.25e2", json!(-125.0)),
        (json!({"type":"integer", "enum":[1,2]}), "2", json!(2)),
        (json!({"type":"boolean"}), "True", json!(true)),
        (json!({"type":"boolean"}), "true", json!(true)),
        (json!({"type":"boolean"}), "False", json!(false)),
        (
            json!({"type":"boolean", "const":false}),
            "false",
            json!(false),
        ),
        (json!({"type":"null"}), "null", Value::Null),
        (json!({"type":"null"}), "None", Value::Null),
        (
            json!({"type":"object", "properties":{"a":{"type":"array", "items":{"type":"integer"}}}, "required":["a"]}),
            " \n{\"a\": [1, 2]} \n",
            json!({"a":[1,2]}),
        ),
        (
            json!({"type":"array", "items":{"type":"object", "properties":{"ok":{"type":"boolean"}}, "required":["ok"]}}),
            "[{\"ok\":true}, {\"ok\":false}]",
            json!([{"ok":true},{"ok":false}]),
        ),
        // Non-strict annotations/bounds remain visible in the prompt, but are
        // deliberately not enforced, just as on the Qwen3 path.
        (json!({"type":"integer", "minimum":100}), "0", json!(0)),
        (
            json!({"type":"string", "minLength":100, "pattern":"^x$"}),
            "y",
            json!("y"),
        ),
    ] {
        let calls = compile_calls(
            &json!({"type":"object", "properties":{"value":schema}, "required":["value"]}),
        )?;
        let text = envelope(&parameter("value", text));
        let mut grammar = ToolCallGrammar::new(calls.clone());
        for c in text.chars() {
            grammar.commit_token_text(&c.to_string())?;
        }
        assert_eq!(grammar.status(), GrammarStatus::Complete, "{text:?}");
        grammar.finish()?;
        assert!(!grammar.allows_token_text("\n"));
        assert!(!grammar.allows_token_text("<tool_call>"));
        let parsed = calls.parse_call(&text)?;
        assert_eq!(parsed.name, "run");
        assert_eq!(parsed.arguments, json!({"value": expected}), "{text:?}");
        assert_eq!(
            parsed.arguments_text,
            serde_json::to_string(&parsed.arguments)?
        );
    }
    Ok(())
}

#[test]
fn xml_required_optional_and_emitted_key_order() -> Result<()> {
    let calls = compile_calls(&json!({"type":"object", "properties":{
        "a":{"type":"string"}, "b":{"type":"boolean"}, "c":{"type":"null"}
    }, "required":["a", "b"]}))?;
    let mut grammar = ToolCallGrammar::new(calls.clone());
    grammar.commit_token_text("\n<function=run>\n")?;
    assert!(!grammar.allows_token_text("</function>"));
    grammar.commit_token_text(&parameter("b", "True"))?;
    assert!(!grammar.allows_token_text("</function>"));
    assert!(!grammar.allows_token_text("<parameter=b>"));
    assert!(!grammar.allows_token_text("<parameter=unknown>"));
    grammar.commit_token_text(&parameter("a", "first\nsecond"))?;
    grammar.commit_token_text("</function>\n</tool_call>")?;
    assert_eq!(
        calls.parse_call(grammar.output_text())?.arguments_text,
        r#"{"b":true,"a":"first\nsecond"}"#
    );
    let no_parameters = compile_calls(&json!({"type":"object", "properties":{}}))?;
    let mut grammar = ToolCallGrammar::new(no_parameters.clone());
    grammar.commit_token_text(&envelope(""))?;
    grammar.finish()?;
    assert_eq!(
        no_parameters
            .parse_call(grammar.output_text())?
            .arguments_text,
        "{}"
    );
    Ok(())
}

#[test]
fn xml_rejects_invalid_names_duplicates_values_and_envelopes() -> Result<()> {
    let calls = compile_calls(&json!({"type":"object", "properties":{
        "s":{"type":"string", "enum":["yes", "no"]},
        "i":{"type":"integer"}, "n":{"type":"number"},
        "b":{"type":"boolean"}, "z":{"type":"null"},
        "o":{"type":"object", "properties":{"x":{"type":"integer"}}, "required":["x"]},
        "a":{"type":"array", "items":{"type":"integer"}}
    }, "required":["s"]}))?;
    for invalid in [
        "\n<function=unknown>\n".to_owned(),
        envelope(""),
        envelope(&parameter("unknown", "x")),
        envelope(&(parameter("s", "yes") + &parameter("s", "no"))),
        envelope(&parameter("s", "maybe")),
        envelope(&parameter("i", "1.2")),
        envelope(&parameter("i", "01")),
        envelope(&parameter("n", "NaN")),
        envelope(&parameter("b", "TRUE")),
        envelope(&parameter("z", "none")),
        envelope(&parameter("o", "{\"x\":1,\"extra\":2}")),
        envelope(&parameter("o", "{}")),
        envelope(&parameter("a", "[\"x\"]")),
        envelope(&parameter("s", "yes")) + "\n",
        envelope(&parameter("s", "yes")) + &envelope(""),
        envelope(&parameter("s", "yes")).replace("</function>\n", "</function>"),
    ] {
        assert!(
            !ToolCallGrammar::new(calls.clone()).allows_token_text(&invalid),
            "{invalid:?}"
        );
        assert!(calls.parse_call(&invalid).is_err(), "{invalid:?}");
    }
    for invalid in [
        "",
        "\n<function=run>\n",
        "\n<function=run>\n<parameter=s>\nyes",
    ] {
        assert!(parse_xml_tool_call(invalid).is_err());
        let mut grammar = ToolCallGrammar::new(calls.clone());
        if !invalid.is_empty() {
            grammar.commit_token_text(invalid)?;
        }
        assert!(grammar.finish().is_err());
    }
    Ok(())
}

/// The full-mask sampler uses exactly the same constraint, with only the
/// candidate-check optimization disabled.
struct FullMask(LazyToolCallConstraint);
impl LogitsConstraint for FullMask {
    fn mask_logits(&mut self, logits: &mut [f32]) -> Result<()> {
        self.0.mask_logits(logits)
    }
    fn commit_token(&mut self, token: u32) -> Result<()> {
        self.0.commit_token(token)
    }
    fn finish(&self) -> Result<()> {
        self.0.finish()
    }
}

#[test]
fn xml_lazy_constraint_candidate_checks_and_greedy_match_full_masks() -> Result<()> {
    let pieces = [
        "<think>",
        "Use <tool_call> inside reasoning",
        "</think>",
        "Looking.\n",
        "<tool_call>",
        "\n<function=run>\n",
        "<parameter=b>\n",
        "True",
        "\n</parameter>\n",
        "</function>\n</tool_call>",
        "<eos>",
        "<pad>",
        "<parameter=unknown>\n",
        "\n<function=unknown>\n",
        "\n<function=run>\n</function>",
        "True\n</parameter>\n</function>\n</tool_call>",
    ];
    let vocabulary: [(String, u32); 16] =
        std::array::from_fn(|id| (pieces[id].to_owned(), id as u32));
    let mut tokenizer = Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocabulary, Vec::new())
            .build()
            .unwrap(),
    );
    tokenizer.with_decoder(Some(ByteFallback::default()));
    let mut constraint = LazyToolCallConstraint::new(
        Arc::new(tokenizer),
        vec![10],
        compile_calls(&json!({
            "type":"object", "properties":{"b":{"type":"boolean"}}, "required":["b"]
        }))?,
    )?;
    assert!(constraint.checks_tokens());
    for token in [0, 1, 2, 3] {
        constraint.commit_token(token)?;
        assert!(!constraint.call_opened());
        let mut logits = vec![1.0; pieces.len()];
        constraint.mask_logits(&mut logits)?;
        assert!(logits.iter().all(|value| value.is_finite()));
    }
    constraint.commit_token(4)?;
    assert!(constraint.call_opened());
    for selected in [5, 6, 7, 8, 9, 10] {
        let mut mask = vec![1.0; pieces.len()];
        constraint.clone().mask_logits(&mut mask)?;
        for (id, logit) in mask.iter().enumerate() {
            assert_eq!(constraint.allows_token(id as u32)?, logit.is_finite());
        }
        assert_eq!(
            constraint.allows_token(10)?,
            selected == 10,
            "EOS only after the closing tag"
        );
        // Make every vocabulary item the top-ranked candidate in turn. This
        // exercises rejected maxima, ties and tokens spanning several states.
        for top in 0..pieces.len() {
            let scores = (0..pieces.len())
                .map(|id| if id == top { 100.0 } else { (id % 4) as f32 })
                .collect::<Vec<_>>();
            let logits = Tensor::from_vec(scores, (1, pieces.len()), &Device::Cpu)?;
            let config = SamplingConfig {
                temperature: 0.0,
                ..SamplingConfig::default()
            };
            let mut fast = Sampler::new(config)?;
            let mut full = Sampler::new(config)?;
            let mut fast_constraint = constraint.clone();
            let mut full_constraint = FullMask(constraint.clone());
            let actual = fast.sample_with_constraint(&logits, &[], Some(&mut fast_constraint))?;
            let expected = full.sample_with_constraint(&logits, &[], Some(&mut full_constraint))?;
            assert_eq!(actual, expected, "top={top}, next={selected}");
            assert_eq!(
                fast_constraint.grammar().output_text(),
                full_constraint.0.grammar().output_text()
            );
        }
        constraint.commit_token(selected)?;
    }
    constraint.finish()?;
    assert!(constraint.is_finished());
    assert!(constraint.allows_token(10).is_err());
    Ok(())
}

#[test]
fn xml_selects_each_declared_functions_own_schema() -> Result<()> {
    let calls = CompiledToolCalls::qwen35([
        (
            "run".to_owned(),
            CompiledJsonSchema::compile(&json!({"type":"object", "properties":{
            "value":{"type":"boolean"}
        }, "required":["value"], "additionalProperties":false}))?,
        ),
        (
            "run_text".to_owned(),
            CompiledJsonSchema::compile_closed(&json!({"type":"object", "properties":{
            "value":{"type":"string", "const":"True"}, "optional":{"type":"integer"}
        }, "required":["value"]}))?,
        ),
    ])?;
    for (name, expected) in [("run", json!(true)), ("run_text", json!("True"))] {
        let text = envelope(&parameter("value", "True"))
            .replace("function=run>", &format!("function={name}>"));
        let mut grammar = ToolCallGrammar::new(calls.clone());
        grammar.commit_token_text(&text)?;
        grammar.finish()?;
        assert_eq!(
            calls.parse_call(&text)?.arguments,
            json!({"value":expected})
        );
    }
    let mut grammar = ToolCallGrammar::new(calls.clone());
    grammar.commit_token_text("\n<function=run_text>\n<parameter=value>\n")?;
    assert!(!grammar.allows_token_text("False"));
    assert!(grammar.allows_token_text("True"));
    assert!(
        !ToolCallGrammar::new(calls).allows_token_text("\n<function=run>\n<parameter=optional>")
    );
    Ok(())
}

#[test]
fn xml_preserves_literal_parameter_names_and_rejects_ambiguous_tag_fields() -> Result<()> {
    let declarations = json!({"type":"object", "properties":{
        "":{"type":"integer"}, "x=y":{"type":"string"}
    }, "required":["", "x=y"]});
    let calls = compile_calls(&declarations)?;
    let text = envelope(&(parameter("x=y", "yes") + &parameter("", "1")));
    let mut grammar = ToolCallGrammar::new(calls.clone());
    grammar.commit_token_text(&text)?;
    assert_eq!(
        calls.parse_call(&text)?.arguments_text,
        r#"{"x=y":"yes","":1}"#
    );
    assert!(
        compile_calls(&json!({"type":"object", "properties":{"bad>\nname":{"type":"string"}}}))
            .is_err()
    );
    Ok(())
}
