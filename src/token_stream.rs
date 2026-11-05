use std::collections::HashSet;

use anyhow::Result;
use tokenizers::{
    DecodeStream, DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper,
    PreTokenizerWrapper, Tokenizer,
};

type InnerStream<'a> = DecodeStream<
    'a,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// Incremental tokenizer decoder that preserves incomplete UTF-8 byte state.
pub struct TokenStream<'a> {
    tokenizer: &'a Tokenizer,
    inner: InnerStream<'a>,
    token_ids: Vec<u32>,
    emitted: String,
    finished: bool,
    suppressed_special_ids: Option<HashSet<u32>>,
}

impl<'a> TokenStream<'a> {
    #[must_use]
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            inner: tokenizer.decode_stream(true),
            token_ids: Vec::new(),
            emitted: String::new(),
            finished: false,
            suppressed_special_ids: None,
        }
    }

    /// Preserve selected marker token IDs while continuing to suppress other special tokens.
    #[must_use]
    pub fn with_markers(mut self, markers: &[&str]) -> Self {
        // Feed retained markers through the same contextual decoder as text.
        // Resetting at a marker would strip SentencePiece's next leading space.
        self.inner = self.tokenizer.decode_stream(false);
        self.suppressed_special_ids = Some(
            self.tokenizer
                .get_added_tokens_decoder()
                .into_iter()
                .filter_map(|(id, token)| {
                    (token.special && !markers.contains(&token.content.as_str())).then_some(id)
                })
                .collect(),
        );
        self
    }

    pub fn push(&mut self, token: u32) -> Result<Option<String>> {
        anyhow::ensure!(
            !self.finished,
            "cannot push a token after the token stream has finished"
        );

        if self
            .suppressed_special_ids
            .as_ref()
            .is_some_and(|ids| ids.contains(&token))
        {
            return Ok(None);
        }

        let chunk = self
            .inner
            .step(token)
            .map_err(|error| anyhow::anyhow!("tokenizer failed while streaming output: {error}"))?;
        self.token_ids.push(token);
        if let Some(text) = &chunk {
            self.emitted.push_str(text);
        }
        Ok(chunk)
    }

    /// Completes a normally terminated stream and returns any text held back by
    /// the incremental decoder.
    ///
    /// `DecodeStream` deliberately returns `None` while byte-fallback tokens
    /// end in incomplete UTF-8. A full decode at the request boundary preserves
    /// that trailing output as the Unicode replacement character instead of
    /// silently dropping it. Calling `finish` again is harmless and returns
    /// `None`; pushing more tokens after finishing is an error. Cancelled
    /// requests should simply drop their request-local `TokenStream`.
    pub fn finish(&mut self) -> Result<Option<String>> {
        if self.finished {
            return Ok(None);
        }

        let decoded = self
            .tokenizer
            .decode(&self.token_ids, self.suppressed_special_ids.is_none())
            .map_err(|error| {
                anyhow::anyhow!("tokenizer failed while finalizing output: {error}")
            })?;
        anyhow::ensure!(
            decoded.starts_with(&self.emitted),
            "tokenizer produced inconsistent incremental and final output"
        );

        let suffix = decoded[self.emitted.len()..].to_owned();
        self.finished = true;
        if suffix.is_empty() {
            Ok(None)
        } else {
            self.emitted.push_str(&suffix);
            Ok(Some(suffix))
        }
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::{
        AddedToken, Tokenizer, decoders::byte_fallback::ByteFallback, models::bpe::BPE,
    };

    use super::TokenStream;

    #[test]
    fn preserving_markers_keeps_sentencepiece_decoder_context() -> anyhow::Result<()> {
        use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

        let model = BPE::builder()
            .vocab_and_merges(
                [("▁before".to_owned(), 0), ("▁after".to_owned(), 1)],
                Vec::new(),
            )
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(Some(Metaspace::new('▁', PrependScheme::Always, true)));
        tokenizer
            .add_special_tokens([AddedToken::from("<tool_call>", true)])
            .unwrap();
        let marker = tokenizer.token_to_id("<tool_call>").unwrap();
        let mut stream = TokenStream::new(&tokenizer).with_markers(&["<tool_call>"]);
        let mut text = String::new();
        for id in [0, marker, 1] {
            text.push_str(&stream.push(id)?.unwrap_or_default());
        }
        text.push_str(&stream.finish()?.unwrap_or_default());
        assert_eq!(text, "before<tool_call> after");
        Ok(())
    }

    #[test]
    fn preserves_only_selected_special_markers_for_parser() -> anyhow::Result<()> {
        use ToolCallFormat::{GraniteJson, Hermes, Llama3Json, MistralJson, Pythonic, QwenXml};
        use macllm_chat::tool_format::{ToolCallFormat, ToolCallParser};

        let formats = [
            Hermes,
            QwenXml,
            Llama3Json,
            MistralJson,
            GraniteJson,
            Pythonic,
        ];
        let vocab: tokenizers::models::bpe::Vocab = (0..=255)
            .map(|byte| (format!("<0x{byte:02X}>"), byte))
            .collect();
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .byte_fallback(true)
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(Some(ByteFallback::default()));
        for format in formats {
            tokenizer
                .add_special_tokens(format.markers().iter().map(|m| AddedToken::from(*m, true)))
                .unwrap();
        }
        tokenizer
            .add_special_tokens([AddedToken::from("<|unrelated|>", true)])
            .unwrap();

        for (format, fragments) in [
            (
                Hermes,
                vec![
                    "<tool_call>",
                    r#"{"name":"f","arguments":{"x":"é"}}"#,
                    "</tool_call>",
                ],
            ),
            (
                QwenXml,
                vec![
                    "<tool_call>",
                    "<function=",
                    "f>",
                    "<parameter=",
                    "x>é",
                    "</parameter>",
                    "</function>",
                    "</tool_call>",
                ],
            ),
            (
                Llama3Json,
                vec!["<|python_tag|>", r#"{"name":"f","parameters":{"x":"é"}}"#],
            ),
            (
                MistralJson,
                vec!["[TOOL_CALLS]", "f", "[ARGS]", r#"{"x":"é"}"#],
            ),
            (
                GraniteJson,
                vec!["<|tool_call|>", r#"[{"name":"f","arguments":{"x":"é"}}]"#],
            ),
            (
                Pythonic,
                vec!["<|tool_call_start|>", "[f(x='é')]", "<|tool_call_end|>"],
            ),
        ] {
            let mut ids = Vec::new();
            // Other formats' special markers must stay suppressed too.
            for other in formats {
                for marker in other.markers() {
                    if !format.markers().contains(marker) {
                        ids.push(tokenizer.token_to_id(marker).unwrap());
                    }
                }
            }
            ids.push(tokenizer.token_to_id("<|unrelated|>").unwrap());
            let mut ordinary = String::new();
            for fragment in fragments {
                if format.markers().contains(&fragment) {
                    ids.push(tokenizer.token_to_id(fragment).unwrap());
                } else {
                    ordinary.push_str(fragment);
                    ids.extend(fragment.bytes().map(u32::from));
                }
            }
            ids.push(tokenizer.token_to_id("<|unrelated|>").unwrap());
            assert_eq!(tokenizer.decode(&ids, true).unwrap(), ordinary);

            let mut stream = TokenStream::new(&tokenizer).with_markers(format.markers());
            let mut pinned_stream = TokenStream::new(&tokenizer);
            let mut parser = ToolCallParser::new(format);
            let mut calls = Vec::new();
            let mut content = String::new();
            let mut pinned_text = String::new();
            for id in ids {
                if let Some(chunk) = stream.push(id)? {
                    let parsed = parser.push(&chunk);
                    content.push_str(&parsed.content);
                    calls.extend(parsed.calls);
                }
                if let Some(chunk) = pinned_stream.push(id)? {
                    pinned_text.push_str(&chunk);
                }
            }
            if let Some(chunk) = stream.finish()? {
                let parsed = parser.push(&chunk);
                content.push_str(&parsed.content);
                calls.extend(parsed.calls);
            }
            let end = parser.finish()?;
            content.push_str(&end.content);
            calls.extend(end.calls);
            assert!(content.is_empty(), "{format:?}: {content}");
            assert_eq!(calls.len(), 1, "{format:?}");
            assert_eq!(calls[0].arguments["x"], "é", "{format:?}");
            pinned_text.push_str(&pinned_stream.finish()?.unwrap_or_default());
            assert_eq!(pinned_text, ordinary, "{format:?}");
        }
        Ok(())
    }

    fn byte_fallback_tokenizer() -> tokenizers::Tokenizer {
        let vocab = [
            ("<0x20>".to_owned(), 0),
            ("<0xC3>".to_owned(), 1),
            ("<0xA9>".to_owned(), 2),
            (" plain".to_owned(), 3),
        ];
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .byte_fallback(true)
            .build()
            .expect("test BPE should be valid");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(Some(ByteFallback::default()));
        tokenizer
    }

    #[test]
    fn finish_preserves_complete_unicode_without_duplication() -> anyhow::Result<()> {
        let tokenizer = byte_fallback_tokenizer();
        let mut stream = TokenStream::new(&tokenizer);

        assert_eq!(stream.push(1)?, None);
        assert_eq!(stream.push(2)?.as_deref(), Some("é"));
        assert_eq!(stream.push(3)?.as_deref(), Some(" plain"));
        assert_eq!(stream.finish()?, None);

        Ok(())
    }

    #[test]
    fn finish_flushes_incomplete_byte_fallback_and_is_idempotent() -> anyhow::Result<()> {
        let tokenizer = byte_fallback_tokenizer();
        let mut stream = TokenStream::new(&tokenizer);

        assert_eq!(stream.push(1)?, None);
        assert_eq!(stream.finish()?.as_deref(), Some("�"));
        assert_eq!(stream.finish()?, None);
        assert!(
            stream.push(2).is_err(),
            "a finalized request must not accept more tokens"
        );

        Ok(())
    }

    #[test]
    fn request_local_streams_do_not_share_incomplete_byte_state() -> anyhow::Result<()> {
        let tokenizer = byte_fallback_tokenizer();
        {
            let mut cancelled = TokenStream::new(&tokenizer);
            assert_eq!(cancelled.push(1)?, None);
        }

        let mut next_request = TokenStream::new(&tokenizer);
        assert_eq!(next_request.push(1)?, None);
        assert_eq!(next_request.push(2)?.as_deref(), Some("é"));
        assert_eq!(next_request.finish()?, None);

        Ok(())
    }
}
