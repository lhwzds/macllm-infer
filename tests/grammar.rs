use std::sync::Arc;

use anyhow::{Context, Result};
use macllm_infer::{
    grammar::{
        CompiledJsonSchema, CompiledToolCalls, GrammarStatus, JsonSchemaGrammar,
        JsonSchemaTokenConstraint, LazyToolCallConstraint,
    },
    sampling::LogitsConstraint,
    weights::ModelFiles,
};
use serde_json::{Value, json};
use tokenizers::{
    AddedToken, Tokenizer,
    decoders::DecoderWrapper,
    decoders::byte_fallback::ByteFallback,
    decoders::byte_level::ByteLevel,
    decoders::fuse::Fuse,
    decoders::sequence::Sequence as DecoderSequence,
    decoders::strip::Strip,
    models::{bpe::BPE, wordlevel::WordLevel},
    pre_tokenizers::metaspace::{Metaspace, PrependScheme},
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
fn byte_fallback_tokens_that_split_one_character_are_allowed_and_committed() -> Result<()> {
    let tokenizer = Arc::new(byte_fallback_tokenizer());
    let mut constraint =
        JsonSchemaTokenConstraint::from_schema(tokenizer, vec![3], &json!({"type":"string"}))?;
    constraint.commit_token(0)?;

    let mut logits = vec![1.0_f32; 4];
    constraint.mask_logits(&mut logits)?;
    assert!(
        logits[1].is_finite(),
        "the leading half of a character must stay selectable"
    );
    assert!(
        logits[2].is_infinite(),
        "a trailing half alone can never spell a character"
    );

    constraint.commit_token(1)?;
    assert_eq!(constraint.grammar().output_text(), "\"");
    let mut logits = vec![1.0_f32; 4];
    constraint.mask_logits(&mut logits)?;
    assert!(
        logits[2].is_finite(),
        "the trailing half of a character must stay selectable"
    );
    constraint.commit_token(2)?;
    assert_eq!(constraint.grammar().output_text(), "\"\u{e9}");
    Ok(())
}

#[test]
fn a_byte_fallback_token_that_is_not_a_byte_is_refused_while_a_byte_is_pending() -> Result<()> {
    let tokenizer = Arc::new(byte_fallback_tokenizer());
    let mut constraint =
        JsonSchemaTokenConstraint::from_schema(tokenizer, vec![3], &json!({"type":"string"}))?;
    constraint.commit_token(0)?;
    constraint.commit_token(1)?;
    // The quote is not a `<0xHH>` entry, so it cannot complete the pending
    // byte.
    assert!(!masked_tokens_of(&constraint, 4)[0]);
    assert!(constraint.commit_token(0).is_err());
    assert_eq!(constraint.grammar().output_text(), "\"");
    assert_eq!(constraint.selected_token_ids(), [0, 1]);
    Ok(())
}

/// The byte-level alphabet: bytes `!..=~`, `0xA1..=0xAC`, and `0xAE..=0xFF`
/// map to themselves, every other byte maps to the characters from U+0100 up.
fn byte_level_alphabet() -> Vec<char> {
    let mut alphabet = vec!['\u{0}'; 256];
    let mut next = 0x100_u32;
    for (index, slot) in alphabet.iter_mut().enumerate() {
        let byte = u8::try_from(index).expect("the alphabet holds 256 bytes");
        let direct = matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        *slot = if direct {
            char::from(byte)
        } else {
            let character = char::from_u32(next).expect("byte-level characters are valid");
            next += 1;
            character
        };
    }
    alphabet
}

/// A byte-level vocabulary: one token per byte, decoded by the byte-level
/// decoder, whose lossy spelling hides every incomplete prefix alike.
fn byte_level_tokenizer() -> Tokenizer {
    let alphabet = byte_level_alphabet();
    let vocabulary: [(String, u32); 256] = std::array::from_fn(|byte| {
        (
            alphabet[byte].to_string(),
            u32::try_from(byte).expect("a byte fits in a token ID"),
        )
    });
    let model = WordLevel::builder()
        .vocab(vocabulary.into())
        .unk_token("<unk>".to_owned())
        .build()
        .expect("test WordLevel vocabulary is valid");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    tokenizer
        .add_special_tokens([
            AddedToken::from("<eos>", true),
            AddedToken::from("<pad>", true),
        ])
        .expect("test special tokens are valid");
    assert_eq!(tokenizer.token_to_id("<eos>"), Some(BYTE_LEVEL_EOS));
    assert_eq!(tokenizer.token_to_id("<pad>"), Some(BYTE_LEVEL_PAD));
    tokenizer
}

const BYTE_LEVEL_EOS: u32 = 256;
const BYTE_LEVEL_PAD: u32 = 257;
const BYTE_LEVEL_VOCAB: usize = 258;

fn byte_token(byte: u8) -> u32 {
    u32::from(byte)
}

fn byte_tokens(text: &str) -> Vec<u32> {
    text.as_bytes().iter().copied().map(u32::from).collect()
}

fn byte_level_constraint(schema: &Value) -> Result<JsonSchemaTokenConstraint> {
    JsonSchemaTokenConstraint::from_schema(
        Arc::new(byte_level_tokenizer()),
        vec![BYTE_LEVEL_EOS],
        schema,
    )
}

/// The tokens the grammar mask leaves selectable.
fn masked_tokens(constraint: &JsonSchemaTokenConstraint) -> Vec<bool> {
    masked_tokens_of(constraint, BYTE_LEVEL_VOCAB)
}

/// The tokens the grammar mask leaves selectable in a given vocabulary.
fn masked_tokens_of(constraint: &JsonSchemaTokenConstraint, vocabulary_size: usize) -> Vec<bool> {
    let mut logits = vec![1.0_f32; vocabulary_size];
    let mut masked = constraint.clone();
    let _ = masked.mask_logits(&mut logits);
    logits.iter().map(|logit| logit.is_finite()).collect()
}

/// The mask and the commit must agree on every candidate.
fn assert_mask_and_commit_agree(constraint: &JsonSchemaTokenConstraint) {
    let allowed = masked_tokens(constraint);
    for (token_id, allowed) in allowed.iter().enumerate() {
        let mut probe = constraint.clone();
        let committed = probe
            .commit_token(u32::try_from(token_id).expect("a token ID fits in u32"))
            .is_ok();
        assert_eq!(*allowed, committed, "token {token_id}");
    }
}

#[test]
fn a_character_split_over_several_tokens_reaches_the_output_byte_for_byte() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
    let text = "\"\u{1f9d1}\u{200d}\u{1f680}\"";
    let tokens = byte_tokens(text);
    // The joiner spans three tokens, each emoji spans four.
    assert_eq!(tokens.len(), 13);

    for token in &tokens {
        let allowed = masked_tokens(&constraint);
        assert!(
            allowed[usize::try_from(*token).expect("a token ID fits in usize")],
            "token {token} must stay selectable"
        );
        constraint.commit_token(*token)?;
    }
    assert_eq!(constraint.grammar().output_bytes(), text.as_bytes());
    assert_eq!(constraint.grammar().output_text(), text);
    Ok(())
}

#[test]
fn a_zero_width_joiner_split_over_three_tokens_is_committed_as_one_character() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
    constraint.commit_token(byte_token(b'"'))?;
    for byte in [0xE2, 0x80, 0x8D] {
        let allowed = masked_tokens(&constraint);
        assert!(
            allowed[usize::try_from(byte_token(byte)).unwrap()],
            "byte {byte:#04x} must stay selectable"
        );
        constraint.commit_token(byte_token(byte))?;
        // The pending bytes stay out of the output until the character is
        // complete.
        assert_eq!(
            constraint.grammar().output_text(),
            if byte == 0x8D { "\"\u{200d}" } else { "\"" }
        );
    }
    assert_eq!(constraint.grammar().output_bytes(), "\"\u{200d}".as_bytes());
    Ok(())
}

#[test]
fn a_partial_character_is_refused_outside_a_string() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({
        "type":"object",
        "properties":{"a":{"type":"string"}},
        "required":["a"],
        "additionalProperties":false
    }))?;
    // The grammar expects `{`, so no character can start the object.
    let partial = byte_token(0xF0);
    assert!(!masked_tokens(&constraint)[usize::try_from(partial).unwrap()]);
    assert!(constraint.commit_token(partial).is_err());
    assert!(constraint.selected_token_ids().is_empty());
    assert_mask_and_commit_agree(&constraint);
    Ok(())
}

#[test]
fn a_partial_character_is_refused_when_its_bytes_do_not_fit_an_enum() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({"enum":["yes"]}))?;
    constraint.commit_token(byte_token(b'"'))?;
    let partial = byte_token(0xF0);
    assert!(!masked_tokens(&constraint)[usize::try_from(partial).unwrap()]);
    assert!(constraint.commit_token(partial).is_err());
    assert_eq!(constraint.selected_token_ids(), [byte_token(b'"')]);
    assert_mask_and_commit_agree(&constraint);
    Ok(())
}

#[test]
fn eos_and_special_tokens_are_refused_while_a_partial_is_pending() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
    constraint.commit_token(byte_token(b'"'))?;
    constraint.commit_token(byte_token(0xF0))?;
    let allowed = masked_tokens(&constraint);
    assert!(!allowed[usize::try_from(BYTE_LEVEL_EOS).unwrap()]);
    assert!(!allowed[usize::try_from(BYTE_LEVEL_PAD).unwrap()]);
    assert!(constraint.commit_token(BYTE_LEVEL_EOS).is_err());
    assert!(constraint.commit_token(BYTE_LEVEL_PAD).is_err());
    assert_eq!(constraint.grammar().output_text(), "\"");
    assert_mask_and_commit_agree(&constraint);
    Ok(())
}

#[test]
fn a_second_lead_byte_cannot_continue_a_pending_character() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
    constraint.commit_token(byte_token(b'"'))?;
    constraint.commit_token(byte_token(0xF0))?;
    // `F0` starts a four-byte character, and another `F0` is not one of its
    // continuation bytes.
    let second = byte_token(0xF0);
    assert!(!masked_tokens(&constraint)[usize::try_from(second).unwrap()]);
    assert!(constraint.commit_token(second).is_err());
    assert_eq!(constraint.grammar().output_text(), "\"");
    assert_eq!(
        constraint.selected_token_ids(),
        [byte_token(b'"'), byte_token(0xF0)]
    );
    assert_mask_and_commit_agree(&constraint);
    Ok(())
}

#[test]
fn bytes_that_never_start_a_character_are_refused_outright() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
    constraint.commit_token(byte_token(b'"'))?;
    for byte in [0x80, 0xC0, 0xC1, 0xF5] {
        let token = byte_token(byte);
        assert!(
            !masked_tokens(&constraint)[usize::try_from(token).unwrap()],
            "byte {byte:#04x} must not stay selectable"
        );
        assert!(
            constraint.commit_token(token).is_err(),
            "byte {byte:#04x} must not be committed"
        );
    }
    assert_eq!(constraint.selected_token_ids(), [byte_token(b'"')]);
    assert_mask_and_commit_agree(&constraint);
    Ok(())
}

#[test]
fn a_lone_high_surrogate_escape_admits_only_a_low_surrogate_escape() -> Result<()> {
    let mut grammar = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
    grammar.commit_token_text("\"\\ud800")?;
    assert_eq!(grammar.status(), GrammarStatus::Viable);

    // The escape may still continue with the low surrogate that completes
    // the pair, so the backslash stays viable.
    assert!(
        grammar.allows_token_text("\\"),
        "the low surrogate escape must stay viable"
    );

    // No other byte can follow a high surrogate: the value is already dead.
    assert!(
        !grammar.allows_token_text("a"),
        "a byte that cannot start the low surrogate escape must be refused"
    );
    assert!(
        grammar.commit_token_text("a").is_err(),
        "a byte that cannot start the low surrogate escape must not commit"
    );
    assert_eq!(grammar.status(), GrammarStatus::Viable);
    Ok(())
}

#[test]
fn a_lone_high_surrogate_escape_refuses_the_bytes_of_the_character() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
    for byte in b"\"\\ud800" {
        constraint.commit_token(byte_token(*byte))?;
    }
    assert_eq!(constraint.grammar().output_text(), "\"\\ud800");

    // The character U+10000 needs the low surrogate escape after the high
    // one: its raw bytes cannot follow a high surrogate escape.
    let lead = byte_token(0xF0);
    assert!(
        !masked_tokens(&constraint)[usize::try_from(lead).unwrap()],
        "the bytes of the character must not stay selectable"
    );
    assert!(
        constraint.commit_token(lead).is_err(),
        "the bytes of the character must not be committed"
    );
    assert_eq!(constraint.grammar().output_text(), "\"\\ud800");
    Ok(())
}

#[test]
fn a_surrogate_pair_escape_completes_the_string() -> Result<()> {
    let mut grammar = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
    grammar.commit_token_text("\"\\ud800\\udc00")?;
    assert_eq!(grammar.status(), GrammarStatus::Viable);
    grammar.commit_token_text("\"")?;
    assert_eq!(grammar.status(), GrammarStatus::Complete);
    assert!(grammar.can_end());
    grammar.finish()?;
    Ok(())
}

#[test]
fn a_partial_escape_that_can_reach_a_valid_code_unit_stays_viable() -> Result<()> {
    let grammar = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
    // Every one of these prefixes can still become a valid JSON string: the
    // missing hex digits can still spell a code unit the grammar accepts.
    for viable in [
        "\"\\ud800\\ud",
        "\"\\ud800\\udc",
        "\"\\ud",
        "\"\\ud8",
        "\"\\u0",
    ] {
        assert!(
            grammar.allows_token_text(viable),
            "{viable:?} must stay viable"
        );
        let mut committed = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
        committed.commit_token_text(viable)?;
        assert_eq!(committed.status(), GrammarStatus::Viable, "{viable:?}");
    }
    Ok(())
}

#[test]
fn a_partial_escape_that_cannot_reach_a_valid_code_unit_is_refused() -> Result<()> {
    let grammar = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
    assert_eq!(grammar.status(), GrammarStatus::Viable);
    // Every one of these prefixes is dead: no continuation can repair them.
    for refused in [
        "\"\\ud800\\u0",
        "\"\\ud800\\ud8",
        "\"\\ud800\\ue",
        "\"\\udc",
        "\"\\udfff",
    ] {
        assert!(
            !grammar.allows_token_text(refused),
            "{refused:?} must be refused"
        );
        let mut committed = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
        assert!(
            committed.commit_token_text(refused).is_err(),
            "{refused:?} must not commit"
        );
    }
    Ok(())
}

#[test]
fn a_partial_escape_stays_selectable_in_the_byte_level_mask() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
    // The escape of the low surrogate is still one hex digit short of a valid
    // code unit, so every byte of the prefix stays selectable.
    for byte in b"\"\\ud800\\ud" {
        let allowed = masked_tokens(&constraint);
        assert!(
            allowed[usize::try_from(byte_token(*byte)).expect("a token ID fits in usize")],
            "byte {byte:#04x} must stay selectable"
        );
        constraint.commit_token(byte_token(*byte))?;
    }
    // The low surrogate escape can still be completed.
    let digit = byte_token(b'c');
    assert!(
        masked_tokens(&constraint)[usize::try_from(digit).expect("a token ID fits in usize")],
        "the low surrogate escape must stay viable"
    );
    assert_mask_and_commit_agree(&constraint);
    Ok(())
}

#[test]
fn a_partial_escape_that_cannot_reach_a_valid_code_unit_is_not_selectable() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
    for byte in b"\"\\ud" {
        constraint.commit_token(byte_token(*byte))?;
    }
    // The escape can only spell a lone low surrogate, so the value is dead.
    let digit = byte_token(b'c');
    assert!(
        !masked_tokens(&constraint)[usize::try_from(digit).expect("a token ID fits in usize")],
        "a lone low surrogate must not stay selectable"
    );
    assert!(
        constraint.commit_token(digit).is_err(),
        "a lone low surrogate must not be committed"
    );
    assert_eq!(constraint.grammar().output_text(), "\"\\ud");
    Ok(())
}

#[test]
fn every_viable_escape_prefix_stays_selectable_in_the_byte_level_mask() -> Result<()> {
    // Every one of these prefixes can still become a valid JSON string: the
    // missing hex digits can still spell a code unit the grammar accepts.
    for viable in [
        "\"\\ud800\\ud",
        "\"\\ud800\\udc",
        "\"\\ud",
        "\"\\ud8",
        "\"\\u0",
    ] {
        let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
        for byte in viable.as_bytes() {
            let token = byte_token(*byte);
            let allowed = masked_tokens(&constraint);
            assert!(
                allowed[usize::try_from(token).expect("a token ID fits in usize")],
                "byte {byte:#04x} of {viable:?} must stay selectable"
            );
            constraint.commit_token(token)?;
        }
        assert_eq!(
            constraint.grammar().status(),
            GrammarStatus::Viable,
            "{viable:?} must stay viable"
        );
        assert_mask_and_commit_agree(&constraint);
    }
    Ok(())
}

#[test]
fn every_unreachable_escape_prefix_is_refused_in_the_byte_level_mask() -> Result<()> {
    // Each pair is a prefix that cannot reach a valid code unit and the
    // longest head of it that still can: the byte after the head is the one
    // that kills the value, and no continuation can repair it.
    for (refused, head) in [
        ("\"\\ud800\\u0", "\"\\ud800\\u"),
        ("\"\\ud800\\ud8", "\"\\ud800\\ud"),
        ("\"\\ud800\\ue", "\"\\ud800\\u"),
        ("\"\\udc", "\"\\ud"),
        ("\"\\udfff", "\"\\ud"),
    ] {
        let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
        for byte in head.as_bytes() {
            let token = byte_token(*byte);
            let allowed = masked_tokens(&constraint);
            assert!(
                allowed[usize::try_from(token).expect("a token ID fits in usize")],
                "byte {byte:#04x} of {head:?} must stay selectable"
            );
            constraint.commit_token(token)?;
        }
        // The byte that completes the dead escape must not stay selectable.
        let killer = refused.as_bytes()[head.len()];
        let token = byte_token(killer);
        assert!(
            !masked_tokens(&constraint)[usize::try_from(token).expect("a token ID fits in usize")],
            "byte {killer:#04x} of {refused:?} must not stay selectable"
        );
        assert!(
            constraint.commit_token(token).is_err(),
            "byte {killer:#04x} of {refused:?} must not be committed"
        );
        assert_mask_and_commit_agree(&constraint);
    }
    Ok(())
}

#[test]
fn the_complete_characters_are_committed_byte_by_byte() -> Result<()> {
    // The two characters the escapes stand for: one outside the basic plane
    // and one inside it.
    for text in ["\"\u{10000}\"", "\"é\""] {
        // The text API accepts the whole character as one token.
        let grammar = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
        assert!(
            grammar.allows_token_text(text),
            "{text:?} must be allowed as one token"
        );

        let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
        for byte in text.as_bytes() {
            let token = byte_token(*byte);
            let allowed = masked_tokens(&constraint);
            assert!(
                allowed[usize::try_from(token).expect("a token ID fits in usize")],
                "byte {byte:#04x} of {text:?} must stay selectable"
            );
            constraint.commit_token(token)?;
        }
        assert_eq!(
            constraint.grammar().status(),
            GrammarStatus::Complete,
            "{text:?} must complete the string"
        );
        assert_eq!(constraint.grammar().output_text(), text);
        constraint.finish()?;
    }
    Ok(())
}

#[test]
fn a_complete_escape_of_a_character_completes_the_string() -> Result<()> {
    // The text API accepts the two characters the escapes stand for: one
    // outside the basic plane and one inside it.
    let grammar = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
    assert!(
        grammar.allows_token_text("\"𐀀\""),
        "the character outside the basic plane must be allowed"
    );
    assert!(
        grammar.allows_token_text("\"é\""),
        "the character inside the basic plane must be allowed"
    );

    let mut grammar = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
    grammar.commit_token_text("\"\u{10000}\"")?;
    assert_eq!(grammar.status(), GrammarStatus::Complete);
    assert!(grammar.can_end());
    grammar.finish()?;

    let mut grammar = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
    grammar.commit_token_text("\"\u{e9}\"")?;
    assert_eq!(grammar.status(), GrammarStatus::Complete);
    assert!(grammar.can_end());
    grammar.finish()?;
    Ok(())
}

#[test]
fn the_character_outside_the_basic_plane_completes_the_string() -> Result<()> {
    let mut grammar = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
    grammar.commit_token_text("\"\u{10000}\"")?;
    assert_eq!(grammar.status(), GrammarStatus::Complete);
    assert!(grammar.can_end());
    grammar.finish()?;
    Ok(())
}

#[test]
fn a_continuation_that_leaves_the_character_invalid_is_refused() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
    constraint.commit_token(byte_token(b'"'))?;
    constraint.commit_token(byte_token(0xF0))?;
    // `A` cannot continue the pending four-byte character.
    let continuation = byte_token(b'A');
    assert!(!masked_tokens(&constraint)[usize::try_from(continuation).unwrap()]);
    assert!(constraint.commit_token(continuation).is_err());
    assert_eq!(constraint.grammar().output_text(), "\"");
    assert_mask_and_commit_agree(&constraint);
    Ok(())
}

/// A byte-level vocabulary whose entries spell the listed bytes, plus one
/// added EOS token.
fn byte_level_tokenizer_of<const N: usize>(entries: [&[u8]; N]) -> (Tokenizer, u32) {
    let alphabet = byte_level_alphabet();
    let vocabulary: [(String, u32); N] = std::array::from_fn(|index| {
        let token = entries[index]
            .iter()
            .map(|byte| alphabet[usize::from(*byte)])
            .collect::<String>();
        (token, u32::try_from(index).expect("a token ID fits in u32"))
    });
    let model = WordLevel::builder()
        .vocab(vocabulary.into())
        .unk_token("<unk>".to_owned())
        .build()
        .expect("test WordLevel vocabulary is valid");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    let eos = u32::try_from(N).expect("a token ID fits in u32");
    tokenizer
        .add_special_tokens([AddedToken::from("<eos>", true)])
        .expect("test special tokens are valid");
    (tokenizer, eos)
}

#[test]
fn a_token_that_spells_no_bytes_is_refused_while_a_character_is_pending() -> Result<()> {
    // The first token spells the quote, the second one the lead byte of a
    // four-byte character, the third one spells no bytes at all, and the
    // fourth one completes the character.
    let entries: [&[u8]; 4] = [b"\"", &[0xF0], b"", &[0x9F, 0x98, 0x80]];
    let (tokenizer, eos) = byte_level_tokenizer_of(entries);
    let mut constraint = JsonSchemaTokenConstraint::from_schema(
        Arc::new(tokenizer),
        vec![eos],
        &json!({"type":"string"}),
    )?;
    constraint.commit_token(0)?;
    constraint.commit_token(1)?;
    assert_eq!(constraint.grammar().output_text(), "\"");
    // The token that spells no bytes cannot complete the held-back character,
    // so it must not stay selectable: it would be accepted again and again
    // while the output stays where it is.
    assert!(
        !masked_tokens_of(&constraint, 5)[2],
        "a token that spells no bytes must not stay selectable"
    );
    assert!(
        constraint.commit_token(2).is_err(),
        "a token that spells no bytes must not be committed"
    );
    assert_eq!(constraint.grammar().output_text(), "\"");
    assert_eq!(constraint.selected_token_ids(), [0, 1]);
    // The held-back character is still pending, so the token that completes it
    // stays selectable.
    assert!(
        masked_tokens_of(&constraint, 5)[3],
        "the token that completes the character must stay selectable"
    );
    constraint.commit_token(3)?;
    assert_eq!(constraint.grammar().output_text(), "\"\u{1f600}");
    Ok(())
}

#[test]
fn a_literal_replacement_character_reaches_the_output_byte_for_byte() -> Result<()> {
    // One token spells the three bytes of a literal U+FFFD, another one the
    // quote.
    let (tokenizer, eos) = byte_level_tokenizer_of([&[0xEF, 0xBF, 0xBD], b"\""]);
    let mut constraint = JsonSchemaTokenConstraint::from_schema(
        Arc::new(tokenizer),
        vec![eos],
        &json!({"type":"string"}),
    )?;
    constraint.commit_token(1)?;
    assert_eq!(constraint.grammar().output_text(), "\"");

    // The bytes of a literal U+FFFD are valid UTF-8, but the decoder holds the
    // text back because it ends in U+FFFD.
    assert!(masked_tokens_of(&constraint, 3)[0]);
    constraint.commit_token(0)?;
    assert_eq!(constraint.grammar().output_text(), "\"");

    // The closing quote releases the held-back character.
    assert!(masked_tokens_of(&constraint, 3)[1]);
    constraint.commit_token(1)?;
    assert_eq!(
        constraint.grammar().output_bytes(),
        "\"\u{FFFD}\"".as_bytes()
    );
    assert_eq!(constraint.grammar().output_text(), "\"\u{FFFD}\"");
    assert!(constraint.grammar().can_end());
    Ok(())
}

#[test]
fn a_token_that_opens_a_string_and_starts_a_character_fits_only_where_it_fits() -> Result<()> {
    // The first token spells a quote and the lead byte of a four-byte
    // character, the second one the rest of that character.
    let (tokenizer, eos) = byte_level_tokenizer_of([&[0x22, 0xF0], &[0x9F, 0x98, 0x80]]);

    // An enum value is matched byte for byte, so the quote of the token is
    // the only byte that fits the allowed literal.
    let mut enumerated = JsonSchemaTokenConstraint::from_schema(
        Arc::new(tokenizer.clone()),
        vec![eos],
        &json!({"enum":[""]}),
    )?;
    assert!(
        !masked_tokens_of(&enumerated, 3)[0],
        "a token that opens a different literal must not stay selectable"
    );
    assert!(
        enumerated.commit_token(0).is_err(),
        "a token that opens a different literal must not be committed"
    );
    assert!(enumerated.selected_token_ids().is_empty());

    // A free string accepts the quote and the character it starts.
    let mut constraint = JsonSchemaTokenConstraint::from_schema(
        Arc::new(tokenizer),
        vec![eos],
        &json!({"type":"string"}),
    )?;
    // The quote opens the string and the lead byte starts a character inside
    // it, so the token fits here.
    assert!(masked_tokens_of(&constraint, 3)[0]);
    constraint.commit_token(0)?;
    assert_eq!(constraint.grammar().output_text(), "");
    assert!(masked_tokens_of(&constraint, 3)[1]);
    constraint.commit_token(1)?;
    assert_eq!(constraint.grammar().output_text(), "\"\u{1f600}");
    Ok(())
}

/// A byte-level vocabulary whose entries spell the listed bytes, with gaps
/// between the token IDs.
fn sparse_byte_level_tokenizer_of<const N: usize>(entries: [(&[u8], u32); N]) -> (Tokenizer, u32) {
    let alphabet = byte_level_alphabet();
    let vocabulary: [(String, u32); N] = std::array::from_fn(|index| {
        let (bytes, token_id) = entries[index];
        let token = bytes
            .iter()
            .map(|byte| alphabet[usize::from(*byte)])
            .collect::<String>();
        (token, token_id)
    });
    let model = WordLevel::builder()
        .vocab(vocabulary.into())
        .unk_token("<unk>".to_owned())
        .build()
        .expect("test WordLevel vocabulary is valid");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    // The added token takes the first ID the vocabulary leaves free.
    tokenizer
        .add_special_tokens([AddedToken::from("<eos>", true)])
        .expect("test special tokens are valid");
    let eos = tokenizer
        .token_to_id("<eos>")
        .expect("the added token has an ID");
    (tokenizer, eos)
}

#[test]
fn a_sparse_vocabulary_keeps_the_bytes_of_the_tokens_above_its_entry_count() -> Result<()> {
    // The vocabulary leaves gaps between its token IDs, so the number of its
    // entries says nothing about its highest ID: the token that completes a
    // character sits above the entry count and still has to spell its bytes.
    let (tokenizer, eos) = sparse_byte_level_tokenizer_of([
        (b"\"".as_slice(), 0),
        ([0xF0].as_slice(), 2),
        ([0x9F, 0x98, 0x80].as_slice(), 4),
    ]);
    // Three entries, but the highest token ID is four.
    assert_eq!(tokenizer.get_vocab_size(false), 3);
    assert_eq!(eos, 3);
    let mut constraint = JsonSchemaTokenConstraint::from_schema(
        Arc::new(tokenizer),
        vec![eos],
        &json!({"type":"string"}),
    )?;
    constraint.commit_token(0)?;
    // The lead byte of a four-byte character is held back.
    assert!(masked_tokens_of(&constraint, 5)[2]);
    constraint.commit_token(2)?;
    assert_eq!(constraint.grammar().output_text(), "\"");

    // The token that completes the character sits above the entry count of
    // the vocabulary, so it must still spell its bytes.
    assert!(
        masked_tokens_of(&constraint, 5)[4],
        "the token that completes the character must stay selectable"
    );
    constraint.commit_token(4)?;
    assert_eq!(constraint.grammar().output_text(), "\"\u{1f600}");
    // The string closes, so the value can end.
    constraint.commit_token(0)?;
    assert_eq!(constraint.grammar().output_text(), "\"\u{1f600}\"");
    assert!(constraint.grammar().can_end());
    Ok(())
}

/// A byte-level vocabulary whose entries spell the listed bytes, decoded by a
/// decoder sequence that wraps the byte-level decoder.
fn byte_level_sequence_tokenizer_of<const N: usize>(entries: [&[u8]; N]) -> (Tokenizer, u32) {
    let (mut tokenizer, eos) = byte_level_tokenizer_of(entries);
    tokenizer.with_decoder(Some(DecoderSequence::new(vec![
        DecoderWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        DecoderWrapper::Fuse(Fuse::new()),
    ])));
    (tokenizer, eos)
}

#[test]
fn a_byte_level_decoder_inside_a_sequence_decides_split_characters() -> Result<()> {
    // The Llama 3 style tokenizers wrap the byte-level decoder in a sequence,
    // and the exact bytes still decide what a token spells.
    let (tokenizer, eos) =
        byte_level_sequence_tokenizer_of([b"\"", &[0xF0], &[0x9F, 0x98, 0x80], &[0xC0]]);
    let mut constraint = JsonSchemaTokenConstraint::from_schema(
        Arc::new(tokenizer),
        vec![eos],
        &json!({"type":"string"}),
    )?;
    constraint.commit_token(0)?;
    // The lead byte is held back until the character completes.
    constraint.commit_token(1)?;
    assert_eq!(constraint.grammar().output_text(), "\"");
    constraint.commit_token(2)?;
    assert_eq!(constraint.grammar().output_text(), "\"\u{1f600}");
    // A byte that can never spell a character is refused outright.
    assert!(constraint.commit_token(3).is_err());
    assert_eq!(constraint.selected_token_ids(), [0, 1, 2]);
    Ok(())
}

/// A word-level tokenizer without a decoder: its tokens are joined with a
/// space, and a token that spells no text decodes to no text at all.
fn wordlevel_tokenizer_without_decoder() -> Tokenizer {
    let vocabulary = [
        ("\"".to_owned(), 0),
        ("true".to_owned(), 1),
        (String::new(), 2),
        ("<eos>".to_owned(), 3),
    ];
    let model = WordLevel::builder()
        .vocab(vocabulary.into())
        .unk_token("<unk>".to_owned())
        .build()
        .expect("test WordLevel vocabulary is valid");
    Tokenizer::new(model)
}

#[test]
fn a_tokenizer_without_a_decoder_refuses_a_skipped_special_token() -> Result<()> {
    let tokenizer = Arc::new(contextual_wordlevel_tokenizer());
    let mut constraint =
        JsonSchemaTokenConstraint::from_schema(tokenizer, vec![3], &json!({"type":"string"}))?;
    // The added `<pad>` token is dropped by the decoder, so it spells no text.
    let mut logits = vec![1.0_f32; 5];
    constraint.mask_logits(&mut logits)?;
    assert!(
        logits[4].is_infinite(),
        "a skipped special token must not become a zero-width transition"
    );
    assert!(constraint.commit_token(4).is_err());
    assert!(constraint.selected_token_ids().is_empty());
    Ok(())
}

#[test]
fn a_tokenizer_without_a_decoder_refuses_a_token_that_decodes_to_no_text() -> Result<()> {
    let tokenizer = Arc::new(wordlevel_tokenizer_without_decoder());
    let mut constraint =
        JsonSchemaTokenConstraint::from_schema(tokenizer, vec![3], &json!({"type":"string"}))?;
    // The token spells no text, and a tokenizer without a decoder offers no
    // bytes to judge it by, so it must not become a zero-width transition.
    let mut logits = vec![1.0_f32; 4];
    constraint.mask_logits(&mut logits)?;
    assert!(
        logits[2].is_infinite(),
        "a token that decodes to no text must not stay selectable"
    );
    assert!(
        logits[0].is_finite(),
        "a token that spells text must stay selectable"
    );
    assert!(
        constraint.commit_token(2).is_err(),
        "a token that decodes to no text must not be committed"
    );
    assert!(constraint.selected_token_ids().is_empty());
    Ok(())
}

#[test]
fn the_mask_and_the_commit_agree_on_every_candidate() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({"type":"string"}))?;
    assert_mask_and_commit_agree(&constraint);
    for token in byte_tokens("\"\u{1f9d1}\u{200d}\u{1f680}\"") {
        assert_mask_and_commit_agree(&constraint);
        constraint.commit_token(token)?;
    }
    assert_mask_and_commit_agree(&constraint);
    Ok(())
}

#[test]
fn an_enum_value_may_be_split_at_its_bytes() -> Result<()> {
    // The enum value is a character outside the basic multilingual plane, so
    // its four bytes have to reach the output one token at a time.
    let mut constraint = byte_level_constraint(&json!({"enum":["\u{10FFFD}"]}))?;
    constraint.commit_token(byte_token(b'"'))?;
    assert_eq!(constraint.grammar().output_text(), "\"");

    // A byte that no character of the value can spell is refused: the value
    // is matched byte for byte.
    let wrong = byte_token(0xC2);
    assert!(
        !masked_tokens(&constraint)[usize::try_from(wrong).unwrap()],
        "a byte outside the allowed literal must not stay selectable"
    );
    assert!(
        constraint.commit_token(wrong).is_err(),
        "a byte outside the allowed literal must not be committed"
    );
    assert_eq!(constraint.selected_token_ids(), [byte_token(b'"')]);

    // The four bytes of the value are held back until the character is
    // complete, and then reach the output.
    for byte in [0xF4, 0x8F, 0xBF, 0xBD] {
        let token = byte_token(byte);
        assert!(
            masked_tokens(&constraint)[usize::try_from(token).unwrap()],
            "byte {byte:#04x} must stay selectable"
        );
        constraint.commit_token(token)?;
        assert_mask_and_commit_agree(&constraint);
    }
    assert_eq!(constraint.grammar().output_text(), "\"\u{10FFFD}");
    constraint.commit_token(byte_token(b'"'))?;
    assert_eq!(constraint.grammar().output_text(), "\"\u{10FFFD}\"");
    assert!(constraint.grammar().can_end());
    Ok(())
}

#[test]
fn an_enum_value_split_at_its_bytes_refuses_another_character() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({"enum":["\u{e9}"]}))?;
    constraint.commit_token(byte_token(b'"'))?;

    // The lead byte of the value is held back until the character is
    // complete.
    let lead = byte_token(0xC3);
    assert!(masked_tokens(&constraint)[usize::try_from(lead).unwrap()]);
    constraint.commit_token(lead)?;
    assert_eq!(constraint.grammar().output_text(), "\"");

    // The trailing byte completes the character and the value.
    let trail = byte_token(0xA9);
    assert!(masked_tokens(&constraint)[usize::try_from(trail).unwrap()]);
    constraint.commit_token(trail)?;
    assert_eq!(constraint.grammar().output_text(), "\"\u{e9}");

    // A different character is refused: the value is matched byte for byte.
    let mut other = byte_level_constraint(&json!({"enum":["\u{e9}"]}))?;
    other.commit_token(byte_token(b'"'))?;
    other.commit_token(byte_token(0xC3))?;
    let wrong = byte_token(0xA8);
    assert!(
        !masked_tokens(&other)[usize::try_from(wrong).unwrap()],
        "another character must not stay selectable"
    );
    assert!(
        other.commit_token(wrong).is_err(),
        "another character must not be committed"
    );
    assert_eq!(other.grammar().output_text(), "\"");
    Ok(())
}

#[test]
fn a_property_key_may_be_split_at_its_bytes() -> Result<()> {
    let mut constraint = byte_level_constraint(&json!({
        "type":"object",
        "properties":{"\u{e9}":{"type":"string"}},
        "required":["\u{e9}"],
        "additionalProperties":false
    }))?;
    constraint.commit_token(byte_token(b'{'))?;
    constraint.commit_token(byte_token(b'"'))?;
    assert_eq!(constraint.grammar().output_text(), "{\"");

    // The key is matched byte for byte, so its lead byte is held back until
    // the character is complete.
    let lead = byte_token(0xC3);
    assert!(masked_tokens(&constraint)[usize::try_from(lead).unwrap()]);
    constraint.commit_token(lead)?;
    assert_eq!(constraint.grammar().output_text(), "{\"");
    let trail = byte_token(0xA9);
    assert!(masked_tokens(&constraint)[usize::try_from(trail).unwrap()]);
    constraint.commit_token(trail)?;
    assert_eq!(constraint.grammar().output_text(), "{\"\u{e9}");

    // The key closes and the value of the property still follows.
    constraint.commit_token(byte_token(b'"'))?;
    constraint.commit_token(byte_token(b':'))?;
    constraint.commit_token(byte_token(b'"'))?;
    constraint.commit_token(byte_token(b'a'))?;
    constraint.commit_token(byte_token(b'"'))?;
    constraint.commit_token(byte_token(b'}'))?;
    assert_eq!(constraint.grammar().output_text(), "{\"\u{e9}\":\"a\"}");
    assert!(constraint.grammar().can_end());
    Ok(())
}

/// A byte-fallback vocabulary decoded by the Gemma 3 decoder shape: a
/// byte-fallback decoder followed by a metaspace decoder.
fn byte_fallback_metaspace_tokenizer() -> Tokenizer {
    let vocabulary = [
        ("\"".to_owned(), 0),
        ("<0xC3>".to_owned(), 1),
        ("<0xA9>".to_owned(), 2),
        ("hello".to_owned(), 3),
        ("<eos>".to_owned(), 4),
    ];
    let model = BPE::builder()
        .vocab_and_merges(vocabulary, Vec::new())
        .byte_fallback(true)
        .build()
        .expect("test BPE vocabulary is valid");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(DecoderSequence::new(vec![
        DecoderWrapper::ByteFallback(ByteFallback::default()),
        DecoderWrapper::Metaspace(Metaspace::new('\u{2581}', PrependScheme::Always, true)),
    ])));
    tokenizer
}

#[test]
fn a_byte_fallback_and_metaspace_sequence_decides_split_characters() -> Result<()> {
    let tokenizer = Arc::new(byte_fallback_metaspace_tokenizer());
    let mut constraint =
        JsonSchemaTokenConstraint::from_schema(tokenizer, vec![4], &json!({"type":"string"}))?;
    constraint.commit_token(0)?;
    assert_eq!(constraint.grammar().output_text(), "\"");

    // The lead byte of a character is held back until the character is
    // complete.
    let mut logits = vec![1.0_f32; 5];
    constraint.mask_logits(&mut logits)?;
    assert!(
        logits[1].is_finite(),
        "the leading half of a character must stay selectable"
    );
    constraint.commit_token(1)?;
    assert_eq!(constraint.grammar().output_text(), "\"");

    // An ordinary token cannot continue a held-back byte: it spells no bytes
    // the grammar could judge.
    let mut logits = vec![1.0_f32; 5];
    constraint.mask_logits(&mut logits)?;
    assert!(
        logits[3].is_infinite(),
        "an ordinary token must not stay selectable while a byte is pending"
    );
    assert!(
        constraint.commit_token(3).is_err(),
        "an ordinary token must not be committed while a byte is pending"
    );

    // The trailing half completes the character.
    let mut logits = vec![1.0_f32; 5];
    constraint.mask_logits(&mut logits)?;
    assert!(
        logits[2].is_finite(),
        "the trailing half of a character must stay selectable"
    );
    constraint.commit_token(2)?;
    assert_eq!(constraint.grammar().output_text(), "\"\u{e9}");
    Ok(())
}

/// A byte-level vocabulary whose decoder drops the leading space of every
/// token.
fn byte_level_strip_tokenizer() -> Tokenizer {
    let alphabet = byte_level_alphabet();
    let vocabulary: [(String, u32); 256] = std::array::from_fn(|byte| {
        (
            alphabet[byte].to_string(),
            u32::try_from(byte).expect("a byte fits in a token ID"),
        )
    });
    let model = WordLevel::builder()
        .vocab(vocabulary.into())
        .unk_token("<unk>".to_owned())
        .build()
        .expect("test WordLevel vocabulary is valid");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(DecoderSequence::new(vec![
        DecoderWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        DecoderWrapper::Strip(Strip::new(' ', 1, 0)),
    ])));
    tokenizer
        .add_special_tokens([
            AddedToken::from("<eos>", true),
            AddedToken::from("<pad>", true),
        ])
        .expect("test special tokens are valid");
    tokenizer
}

#[test]
fn a_token_whose_complete_text_the_decoder_swallows_is_refused() -> Result<()> {
    let tokenizer = Arc::new(byte_level_strip_tokenizer());
    let mut constraint = JsonSchemaTokenConstraint::from_schema(
        tokenizer,
        vec![BYTE_LEVEL_EOS],
        &json!({"type":"string"}),
    )?;
    // The token spells a complete space, but the strip decoder drops it, so
    // the decoder releases no text at all. Holding the bytes back would
    // leave the output stuck, so the token is refused instead.
    let space = byte_token(0x20);
    assert!(
        !masked_tokens(&constraint)[usize::try_from(space).unwrap()],
        "a token the decoder swallows must not stay selectable"
    );
    assert!(
        constraint.commit_token(space).is_err(),
        "a token the decoder swallows must not be committed"
    );
    assert!(constraint.selected_token_ids().is_empty());
    assert_mask_and_commit_agree(&constraint);
    Ok(())
}

/// Load a tokenizer exactly as the server does: from a checkpoint directory,
/// from a GGUF file, or from a `tokenizer.json` file.
fn load_checked_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    if path.is_dir()
        || path
            .extension()
            .is_some_and(|extension| extension == "gguf")
    {
        // The server opens a checkpoint the same way, and a GGUF file carries
        // its own tokenizer.
        let tokenizer = {
            let files = ModelFiles::from_dir(path)?;
            files.tokenizer()?
        };
        // The checkpoint was the only other owner of the tokenizer, so the
        // sweep can take it out of the shared handle.
        return Arc::try_unwrap(tokenizer)
            .map_err(|_| anyhow::anyhow!("the checkpoint tokenizer has more than one owner"));
    }
    Tokenizer::from_file(path).map_err(|error| anyhow::anyhow!(error.to_string()))
}

/// The special token the sweep adds to a tokenizer that has none.
const SWEEP_EOS_TOKEN: &str = "<macllm-sweep-eos>";

/// What one tokenizer sweep found.
#[derive(Debug)]
struct TokenizerSweep {
    /// The end-of-sequence tokens the constraint used.
    eos_token_ids: Vec<u32>,
    /// The text the opening quote contributed to the JSON string.
    quote_output: String,
    /// The number of tokens the sweep committed into the JSON string.
    checked: usize,
    /// The output every checked token produced, with the token ID.
    committed: Vec<(u32, String)>,
    /// The first token that did not fit, if any.
    mismatch: Option<String>,
}

/// The special token IDs of a tokenizer.
fn special_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    let special = tokenizer.get_added_vocabulary();
    special
        .get_vocab()
        .iter()
        .filter(|(token, _)| special.is_special_token(token))
        .map(|(_, token_id)| *token_id)
        .collect()
}

/// Every vocabulary entry of a tokenizer, with the token ID the model uses.
///
/// The vocabulary entries carry the token IDs the model uses, which are not
/// necessarily the first ones. The added tokens keep their own IDs: the
/// merged vocabulary would give one ID to two tokens.
fn vocabulary_entries(tokenizer: &Tokenizer) -> Vec<(u32, String)> {
    let special = tokenizer.get_added_vocabulary();
    let mut entries = tokenizer
        .get_vocab(false)
        .into_iter()
        .map(|(token, token_id)| (token_id, token))
        .collect::<Vec<_>>();
    entries.extend(
        special
            .get_vocab()
            .iter()
            .map(|(token, token_id)| (*token_id, token.clone())),
    );
    entries.sort_by_key(|(token_id, _)| *token_id);
    entries
}

/// The token that opens the JSON string, if the tokenizer has one.
///
/// The token may spell whitespace around the quote: the JSON grammar skips
/// the whitespace that comes before a value.
fn quote_token_id(tokenizer: &Tokenizer, entries: &[(u32, String)]) -> Option<u32> {
    entries
        .iter()
        .find(|(token_id, token)| {
            !tokenizer.get_added_vocabulary().is_special_token(token)
                && tokenizer
                    .decode(&[*token_id], true)
                    .ok()
                    .is_some_and(|text| text.trim() == "\"")
        })
        .map(|(token_id, _)| *token_id)
}

/// Sweep every token of a tokenizer against a JSON string grammar.
///
/// The opening quote is the quote the tokenizer itself releases: the text
/// that quote contributes is the text the JSON string starts with, the
/// whitespace around it included. Every token is then judged by the text it
/// spells on its own: the sweep commits the token to a fresh constraint that
/// already holds the quote, so the decoder starts from an empty state and
/// releases the text of the token itself. A token whose own text is empty,
/// or that cannot appear inside a JSON string, is not tested: the JSON
/// grammar cannot hold it.
fn sweep_tokens_that_decode_to_plain_text(tokenizer: &mut Tokenizer) -> Result<TokenizerSweep> {
    // The constraint needs an end-of-sequence token. A tokenizer without
    // special tokens has none, and borrowing an ordinary token would give
    // that token two roles at once: the sweep would test it as the end of
    // the turn and as ordinary text, and the two roles disagree. The sweep
    // adds a dedicated special token instead.
    let mut eos_token_ids = special_token_ids(tokenizer);
    if eos_token_ids.is_empty() {
        tokenizer
            .add_special_tokens([AddedToken::from(SWEEP_EOS_TOKEN, true)])
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let eos_token_id = tokenizer
            .token_to_id(SWEEP_EOS_TOKEN)
            .context("the sweep end-of-sequence token is missing")?;
        eos_token_ids.push(eos_token_id);
    }
    let entries = vocabulary_entries(tokenizer);
    let quote_token_id = quote_token_id(tokenizer, &entries)
        .context("the tokenizer has no token that opens a JSON string")?;
    let tokenizer = Arc::new(tokenizer.clone());
    // The quote opens the JSON string. The quote token of the tokenizer
    // releases the text the string starts with.
    let mut base = JsonSchemaTokenConstraint::from_schema(
        Arc::clone(&tokenizer),
        eos_token_ids.clone(),
        &json!({"type":"string"}),
    )?;
    base.commit_token(quote_token_id)
        .map_err(|error| anyhow::anyhow!("the quote token cannot open a JSON string: {error:#}"))?;
    let quote_output = base.grammar().output_text().to_owned();
    // Every token is judged on its own: the sweep commits it to a fresh
    // constraint that holds the quote, so the decoder starts from an empty
    // state and releases the text of the token. The clone shares the table
    // of the exact bytes every token spells, so the sweep stays cheap:
    // building that table again for every token would cost a whole
    // vocabulary scan per token.
    let mut grammar = JsonSchemaGrammar::from_schema(&json!({"type":"string"}))?;
    grammar.commit_token_text(&quote_output)?;
    let fresh = JsonSchemaTokenConstraint::with_grammar(
        Arc::clone(&tokenizer),
        eos_token_ids.clone(),
        grammar,
    )?;
    let mut checked = 0_usize;
    let mut outputs: Vec<(u32, String)> = Vec::new();
    let mut mismatch: Option<String> = None;

    for (token_id, token) in entries {
        if tokenizer.get_added_vocabulary().is_special_token(&token) {
            continue;
        }
        // The text the token spells on its own. A token whose own text is
        // empty, or that cannot appear inside a JSON string, is not tested:
        // the JSON grammar cannot hold it.
        let decoded = tokenizer
            .decode(&[token_id], true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        if decoded.is_empty()
            || decoded.contains('"')
            || decoded.contains('\\')
            || decoded.contains('\u{FFFD}')
            || decoded.chars().any(char::is_control)
        {
            continue;
        }
        let mut constraint = fresh.clone();
        let committed = constraint.commit_token(token_id);
        let output = constraint.grammar().output_text().to_owned();
        let expected = format!("{quote_output}{decoded}");
        checked += 1;
        outputs.push((token_id, output.clone()));
        if committed.is_err() || output != expected {
            mismatch.get_or_insert_with(|| {
                format!(
                    "token {token_id} ({token:?}) decoded to {decoded:?}: commit {committed:?}, output {output:?}, expected {expected:?}"
                )
            });
        }
    }

    Ok(TokenizerSweep {
        eos_token_ids,
        quote_output,
        checked,
        committed: outputs,
        mismatch,
    })
}

/// Check every token of a real tokenizer against a JSON string grammar.
///
/// The test needs a checkpoint: `MACLLM_TOKENIZER_CHECK=/path/to/model.gguf
/// cargo test --test grammar -- --ignored --nocapture`.
#[test]
#[ignore = "reads a real tokenizer from MACLLM_TOKENIZER_CHECK"]
fn every_token_that_decodes_to_plain_text_fits_a_json_string() -> Result<()> {
    let Ok(path) = std::env::var("MACLLM_TOKENIZER_CHECK") else {
        eprintln!("MACLLM_TOKENIZER_CHECK is unset; skipping the tokenizer check");
        return Ok(());
    };
    let path = std::path::PathBuf::from(path);
    let mut tokenizer = load_checked_tokenizer(&path)?;
    let sweep = sweep_tokens_that_decode_to_plain_text(&mut tokenizer)?;
    println!(
        "checked {} tokens of {} (the opening quote contributed {:?})",
        sweep.checked,
        path.display(),
        sweep.quote_output
    );
    if let Some(mismatch) = sweep.mismatch {
        panic!("{mismatch}");
    }
    Ok(())
}

/// A vocabulary whose tokens spell a space before their text, so that the
/// tokenizer releases a space with every token it decodes.
fn space_prefixed_tokenizer() -> Tokenizer {
    let vocabulary = [
        ("\u{2581}\"".to_owned(), 0),
        ("\u{2581}hello".to_owned(), 1),
    ];
    let model = BPE::builder()
        .vocab_and_merges(vocabulary, Vec::new())
        .build()
        .expect("test BPE vocabulary is valid");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(DecoderWrapper::Metaspace(Metaspace::new(
        '\u{2581}',
        PrependScheme::Never,
        true,
    ))));
    tokenizer
}

#[test]
fn the_tokenizer_sweep_opens_the_string_with_the_tokenizer_own_quote() -> Result<()> {
    // The quote token of this tokenizer releases a space and a quote. The
    // sweep has to open the string with the text the tokenizer releases: a
    // sweep that committed a quote of its own to the grammar would judge
    // every token against text the server never produces.
    let mut tokenizer = space_prefixed_tokenizer();
    let sweep = sweep_tokens_that_decode_to_plain_text(&mut tokenizer)?;
    assert_eq!(sweep.quote_output, " \"");
    assert_eq!(sweep.mismatch, None, "the sweep found a mismatch");
    assert_eq!(sweep.checked, 1, "the sweep checked every plain token");
    Ok(())
}

/// A vocabulary without special tokens, whose first token spells plain text.
fn plain_wordlevel_tokenizer() -> Tokenizer {
    let vocabulary = [
        ("hello".to_owned(), 0),
        ("\"".to_owned(), 1),
        ("world".to_owned(), 2),
    ];
    let model = WordLevel::builder()
        .vocab(vocabulary.into())
        .unk_token("<unk>".to_owned())
        .build()
        .expect("test WordLevel vocabulary is valid");
    Tokenizer::new(model)
}

/// A vocabulary in the Gemma 3 shape: the decoder is a byte fallback
/// followed by a metaspace decoder, so a token that spells the word marker
/// releases a space in every decode but the first one.
fn gemma_shape_sweep_tokenizer() -> Tokenizer {
    let vocabulary = [
        ("\"".to_owned(), 0),
        ("\u{2581}hello".to_owned(), 1),
        ("\u{2581}world".to_owned(), 2),
        ("<0xC3>".to_owned(), 3),
        ("<0xA9>".to_owned(), 4),
    ];
    let model = BPE::builder()
        .vocab_and_merges(vocabulary, Vec::new())
        .byte_fallback(true)
        .build()
        .expect("test BPE vocabulary is valid");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(DecoderSequence::new(vec![
        DecoderWrapper::ByteFallback(ByteFallback::default()),
        DecoderWrapper::Metaspace(Metaspace::new('\u{2581}', PrependScheme::Always, true)),
    ])));
    tokenizer
}

#[test]
fn the_tokenizer_sweep_judges_every_token_by_its_own_decode() -> Result<()> {
    // The metaspace decoder drops the word marker of the first token of a
    // decode and turns it into a space in every later one, so the text the
    // tokenizer releases after the opening quote is not the text the token
    // spells on its own. The sweep judges every token by the text it spells
    // on its own: the committed output is the quote plus that text.
    let mut tokenizer = gemma_shape_sweep_tokenizer();
    let sweep = sweep_tokens_that_decode_to_plain_text(&mut tokenizer)?;
    assert_eq!(sweep.quote_output, "\"");
    assert_eq!(sweep.mismatch, None, "the sweep found a mismatch");
    // The tokens that spell half a character decode to a replacement
    // character, so they are not tested: the JSON string cannot hold them.
    assert_eq!(sweep.checked, 2, "the sweep checked every plain token");
    assert_eq!(
        sweep.committed,
        [(1, "\"hello".to_owned()), (2, "\"world".to_owned())],
        "the sweep commits the text the token spells on its own"
    );
    Ok(())
}

#[test]
fn the_tokenizer_sweep_gives_a_tokenizer_without_special_tokens_its_own_end_token() -> Result<()> {
    // The constraint needs an end-of-sequence token, and a tokenizer without
    // special tokens has none. The sweep adds a dedicated special token
    // instead of borrowing an ordinary one: an ordinary token would end the
    // turn in one role and spell text in another, and the sweep would
    // report a mismatch that the tokenizer does not have.
    let mut tokenizer = plain_wordlevel_tokenizer();
    let sweep = sweep_tokens_that_decode_to_plain_text(&mut tokenizer)?;
    assert_eq!(sweep.mismatch, None, "the sweep found a mismatch");
    assert_eq!(sweep.eos_token_ids.len(), 1, "the sweep uses one end token");
    assert!(
        tokenizer
            .get_added_vocabulary()
            .is_special_token(SWEEP_EOS_TOKEN),
        "the end token of the sweep is a special token"
    );
    // The plain tokens of the vocabulary are checked, and the dedicated end
    // token is not one of them.
    assert_eq!(sweep.checked, 2, "the sweep checked every plain token");
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
