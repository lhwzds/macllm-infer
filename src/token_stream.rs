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
        }
    }

    pub fn push(&mut self, token: u32) -> Result<Option<String>> {
        anyhow::ensure!(
            !self.finished,
            "cannot push a token after the token stream has finished"
        );

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
            .decode(&self.token_ids, true)
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
    use tokenizers::{Tokenizer, decoders::byte_fallback::ByteFallback, models::bpe::BPE};

    use super::TokenStream;

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
