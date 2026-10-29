use std::sync::Arc;

use anyhow::Result;
use macllm_infer::{
    grammar::{
        CompiledJsonSchema, CompiledToolCalls, GrammarStatus, JsonSchemaTokenConstraint,
        LazyToolCallConstraint,
    },
    sampling::LogitsConstraint,
};
use serde_json::json;
use tokenizers::{
    AddedToken, Tokenizer,
    decoders::byte_fallback::ByteFallback,
    models::{bpe::BPE, wordlevel::WordLevel},
};

fn contextual_wordlevel_tokenizer() -> Tokenizer {
    let vocabulary = [
        ("\"".to_owned(), 0),
        ("true".to_owned(), 1),
        ("<unk>".to_owned(), 2),
    ];
    let model = WordLevel::builder()
        .vocab(vocabulary.into())
        .unk_token("<unk>".to_owned())
        .build()
        .expect("test WordLevel vocabulary is valid");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer
        .add_special_tokens([
            AddedToken::from("<eos>", true),
            AddedToken::from("<pad>", true),
        ])
        .expect("test special tokens are valid");
    assert_eq!(tokenizer.token_to_id("<eos>"), Some(3));
    assert_eq!(tokenizer.token_to_id("<pad>"), Some(4));
    tokenizer
}

fn byte_fallback_tokenizer() -> Tokenizer {
    let vocabulary = [
        ("\"".to_owned(), 0),
        ("<0xC3>".to_owned(), 1),
        ("<0xA9>".to_owned(), 2),
        ("<eos>".to_owned(), 3),
    ];
    let model = BPE::builder()
        .vocab_and_merges(vocabulary, Vec::new())
        .byte_fallback(true)
        .build()
        .expect("test BPE vocabulary is valid");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteFallback::default()));
    tokenizer
}

#[test]
fn constraint_uses_contextual_decode_stream_deltas_not_raw_vocab_strings() -> Result<()> {
    let tokenizer = Arc::new(contextual_wordlevel_tokenizer());
    let mut constraint =
        JsonSchemaTokenConstraint::from_schema(tokenizer, vec![3], &json!({"type":"string"}))?;

    constraint.commit_token(0)?;
    constraint.commit_token(1)?;
    assert_eq!(constraint.grammar().output_text(), "\" true");
    constraint.commit_token(0)?;

    assert_eq!(constraint.grammar().status(), GrammarStatus::Complete);
    assert_eq!(constraint.grammar().output_text(), "\" true \"");
    constraint.commit_token(3)?;
    assert!(constraint.is_finished());
    assert_eq!(constraint.selected_token_ids(), [0, 1, 0, 3]);
    Ok(())
}

#[test]
fn eos_is_masked_until_the_json_value_is_complete() -> Result<()> {
    let tokenizer = Arc::new(contextual_wordlevel_tokenizer());
    let mut constraint =
        JsonSchemaTokenConstraint::from_schema(tokenizer, vec![3], &json!({"type":"string"}))?;

    assert!(constraint.finish().is_err());
    let mut logits = vec![1.0, 1.0, 1.0, 100.0, 200.0];
    constraint.mask_logits(&mut logits)?;
    assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
    assert!(
        logits[4].is_infinite() && logits[4].is_sign_negative(),
        "a skipped non-EOS special token must not become a zero-width transition"
    );

    constraint.commit_token(0)?;
    constraint.commit_token(0)?;
    let mut logits = vec![1.0, 1.0, 1.0, 100.0, 200.0];
    constraint.mask_logits(&mut logits)?;
    assert!(logits[..3].iter().all(|logit| logit.is_infinite()));
    assert_eq!(logits[3].to_bits(), 100.0_f32.to_bits());
    assert!(constraint.commit_token(0).is_err());
    constraint.commit_token(3)?;
    constraint.finish()?;
    assert!(constraint.mask_logits(&mut logits).is_err());
    assert!(constraint.commit_token(3).is_err());
    Ok(())
}

#[test]
fn pending_byte_fallback_candidates_are_conservatively_rejected() -> Result<()> {
    let tokenizer = Arc::new(byte_fallback_tokenizer());
    let mut constraint =
        JsonSchemaTokenConstraint::from_schema(tokenizer, vec![3], &json!({"type":"string"}))?;
    constraint.commit_token(0)?;

    let mut logits = vec![f32::NEG_INFINITY, 10.0, 9.0, f32::NEG_INFINITY];
    assert!(constraint.mask_logits(&mut logits).is_err());
    assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
    assert!(logits[2].is_infinite() && logits[2].is_sign_negative());
    assert!(constraint.commit_token(1).is_err());
    assert_eq!(constraint.selected_token_ids(), [0]);
    assert_eq!(constraint.grammar().output_text(), "\"");
    Ok(())
}

const EOS: u32 = 10;

/// Vocabulary pieces decode by plain concatenation, so each token's exact
/// text delta is its vocabulary string.
fn tool_call_tokenizer() -> Tokenizer {
    let pieces = [
        "Sure",
        "<think>",
        "\nwhy not write <tool_call> here?\n",
        "</think>",
        "\n\n",
        "<tool_call>",
        "\n{\"name\": \"shell\", \"arguments\": ",
        "{\"command\": \"ls\"}",
        "}\n",
        "</tool_call>",
        "<eos>",
        "\n{\"name\": \"rm\", \"arguments\": ",
        "<tool_call>{",
        "{\"cmd\": 1}",
    ];
    let vocabulary: [(String, u32); 14] =
        std::array::from_fn(|id| (pieces[id].to_owned(), u32::try_from(id).unwrap()));
    let model = BPE::builder()
        .vocab_and_merges(vocabulary, Vec::new())
        .build()
        .expect("test BPE vocabulary is valid");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteFallback::default()));
    tokenizer
}

fn shell_constraint() -> Result<LazyToolCallConstraint> {
    let schema = CompiledJsonSchema::compile_closed(&json!({
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"]
    }))?;
    LazyToolCallConstraint::new(
        Arc::new(tool_call_tokenizer()),
        vec![EOS],
        CompiledToolCalls::new([("shell".to_owned(), schema)])?,
    )
}

fn allowed_after_mask(constraint: &mut LazyToolCallConstraint) -> Result<Vec<u32>> {
    let mut logits = vec![1.0_f32; 14];
    constraint.mask_logits(&mut logits)?;
    Ok((0..14)
        .filter(|&id| logits[id as usize].is_finite())
        .collect())
}

#[test]
fn lazy_tool_call_constraint_leaves_text_answers_unmasked() -> Result<()> {
    let mut constraint = shell_constraint()?;
    assert_eq!(allowed_after_mask(&mut constraint)?.len(), 14);
    constraint.commit_token(0)?;
    assert_eq!(allowed_after_mask(&mut constraint)?.len(), 14);
    constraint.commit_token(EOS)?;
    assert!(constraint.is_finished() && !constraint.call_opened());
    constraint.finish()?;
    assert!(constraint.mask_logits(&mut [1.0; 14]).is_err());
    Ok(())
}

#[test]
fn lazy_tool_call_constraint_opens_after_reasoning_and_ends_at_the_first_call() -> Result<()> {
    let mut constraint = shell_constraint()?;
    // Reasoning may spell the opening tag without starting a call.
    for token in [1, 2, 3, 4] {
        constraint.commit_token(token)?;
        assert!(!constraint.call_opened());
    }
    constraint.commit_token(5)?;
    assert!(constraint.call_opened());
    assert!(constraint.finish().is_err(), "an open call cannot end");

    assert_eq!(allowed_after_mask(&mut constraint)?, [6]);
    constraint.commit_token(6)?;
    assert_eq!(allowed_after_mask(&mut constraint)?, [7]);
    assert!(constraint.commit_token(13).is_err());
    constraint.commit_token(7)?;
    assert_eq!(allowed_after_mask(&mut constraint)?, [8]);
    constraint.commit_token(8)?;
    assert_eq!(allowed_after_mask(&mut constraint)?, [9]);
    constraint.commit_token(9)?;
    assert_eq!(constraint.grammar().status(), GrammarStatus::Complete);
    // A second call cannot follow the first: only EOS remains.
    assert_eq!(allowed_after_mask(&mut constraint)?, [EOS]);
    assert!(constraint.commit_token(5).is_err());
    constraint.commit_token(EOS)?;
    constraint.finish()?;
    Ok(())
}

#[test]
fn lazy_tool_call_constraint_fails_closed_on_incomplete_or_malformed_calls() -> Result<()> {
    let mut eos_inside = shell_constraint()?;
    eos_inside.commit_token(5)?;
    assert!(eos_inside.commit_token(EOS).is_err());
    assert!(eos_inside.commit_token(11).is_err());
    assert!(eos_inside.finish().is_err());

    let mut fused = shell_constraint()?;
    assert!(
        fused.commit_token(12).is_err(),
        "bytes sampled with the opening tag must fit the envelope"
    );
    Ok(())
}
