use std::sync::Arc;

use anyhow::Result;
use macllm_infer::{
    grammar::{GrammarStatus, JsonSchemaTokenConstraint},
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
