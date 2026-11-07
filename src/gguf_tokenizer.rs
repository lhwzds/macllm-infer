use std::collections::HashSet;

use anyhow::{Context, Result, anyhow, ensure};
use tokenizers::{
    AddedToken, DecoderWrapper, SplitDelimiterBehavior, Tokenizer,
    decoders::{byte_fallback::ByteFallback, sequence::Sequence as DecoderSequence},
    models::bpe::{BPE, Vocab},
    normalizers::unicode::NFC,
    pre_tokenizers::{
        PreTokenizerWrapper,
        byte_level::ByteLevel,
        digits::Digits,
        metaspace::{Metaspace, PrependScheme},
        sequence::Sequence as PreTokenizerSequence,
        split::{Split, SplitPattern},
    },
    processors::{
        PostProcessorWrapper, sequence::Sequence as PostProcessorSequence,
        template::TemplateProcessing,
    },
};

const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_UNKNOWN: i32 = 2;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;
const TOKEN_TYPE_UNUSED: i32 = 5;
const TOKEN_TYPE_BYTE: i32 = 6;

// Exact Split regex serialized by the official Qwen2/Qwen2.5 tokenizer.json.
// Keep this profile-specific: tokenizer.ggml.pre is llama.cpp's compatibility
// identifier for a complete pre-tokenizer, not a generic family name.
const QWEN2_SPLIT_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
// Qwen3.5 extends Qwen2's letter runs with combining marks. Its tokenizer does
// not normalize input, so the mark-aware boundary must preserve the original
// code-point sequence while keeping each letter+mark run in one pre-token.
const QWEN35_SPLIT_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const LLAMA3_SPLIT_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BpeProfile {
    SmolLm,
    /// Llama 3-style byte BPE used by the current `SmolLM3` GGUF conversion.
    /// Unlike the Llama 3 profile, it does not add BOS by default.
    Smaug,
    /// REFACT-style GPT-2 BPE used by IBM Granite `MoE` GGUFs.
    Refact,
    Qwen2,
    Qwen35,
    Llama3,
    Lfm2,
}

impl BpeProfile {
    fn from_metadata(model: &str, pre_tokenizer: &str) -> Result<Self> {
        ensure!(
            model == "gpt2",
            "unsupported embedded tokenizer model `{model}`"
        );
        match pre_tokenizer {
            "smollm" => Ok(Self::SmolLm),
            "smaug-bpe" => Ok(Self::Smaug),
            "refact" => Ok(Self::Refact),
            // Official Qwen2.5 GGUFs use tokenizer.ggml.model=gpt2 and
            // tokenizer.ggml.pre=qwen2. The two fields describe different
            // parts of the tokenizer contract and must not be interchanged.
            "qwen2" => Ok(Self::Qwen2),
            "qwen35" => Ok(Self::Qwen35),
            // llama.cpp assigns Falcon-H1 the same regex, ignore-merges and
            // automatic-BOS behavior as its Llama 3 tokenizer profile.
            "llama-bpe" | "falcon-h1" => Ok(Self::Llama3),
            "lfm2" => Ok(Self::Lfm2),
            unsupported => Err(anyhow!(
                "unsupported embedded tokenizer pre-tokenizer `{unsupported}`"
            )),
        }
    }

    fn pre_tokenizer(self) -> Result<PreTokenizerSequence> {
        let stages = match self {
            Self::SmolLm | Self::Refact => vec![
                PreTokenizerWrapper::Digits(Digits::new(true)),
                PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, true)),
            ],
            Self::Qwen2 => vec![
                PreTokenizerWrapper::Split(
                    Split::new(
                        SplitPattern::Regex(QWEN2_SPLIT_REGEX.to_owned()),
                        SplitDelimiterBehavior::Isolated,
                        false,
                    )
                    .map_err(|error| anyhow!(error.to_string()))?,
                ),
                PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
            ],
            Self::Qwen35 => vec![
                PreTokenizerWrapper::Split(
                    Split::new(
                        SplitPattern::Regex(QWEN35_SPLIT_REGEX.to_owned()),
                        SplitDelimiterBehavior::Isolated,
                        false,
                    )
                    .map_err(|error| anyhow!(error.to_string()))?,
                ),
                PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
            ],
            Self::Smaug | Self::Llama3 | Self::Lfm2 => vec![
                PreTokenizerWrapper::Split(
                    Split::new(
                        SplitPattern::Regex(LLAMA3_SPLIT_REGEX.to_owned()),
                        SplitDelimiterBehavior::Isolated,
                        false,
                    )
                    .map_err(|error| anyhow!(error.to_string()))?,
                ),
                PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
            ],
        };
        Ok(PreTokenizerSequence::new(stages))
    }

    fn decoder(self) -> ByteLevel {
        match self {
            Self::SmolLm | Self::Smaug | Self::Refact | Self::Llama3 | Self::Lfm2 => {
                ByteLevel::new(true, true, true)
            }
            Self::Qwen2 | Self::Qwen35 => ByteLevel::new(false, false, false),
        }
    }

    fn ignore_merges(self) -> bool {
        matches!(self, Self::Smaug | Self::Llama3)
    }

    fn default_add_bos(self) -> bool {
        matches!(self, Self::Llama3 | Self::Lfm2)
    }
}

/// Format-neutral input extracted from the `tokenizer.*` GGUF metadata keys.
///
/// Each accepted `model`/`pre_tokenizer` pair names an exact reproducible
/// profile. Future tokenizer profiles can extend this enum-like boundary
/// without changing model loading or generation.
#[derive(Debug, Clone)]
pub struct EmbeddedBpeTokenizer {
    pub model: String,
    pub pre_tokenizer: String,
    pub tokens: Vec<String>,
    pub token_types: Vec<i32>,
    pub merges: Vec<String>,
    pub add_space_prefix: bool,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl EmbeddedBpeTokenizer {
    /// The effective policy of the tokenizer's BOS token.
    ///
    /// The metadata key is optional: a profile that adds its BOS token by
    /// default keeps that default when the key is missing. The policy is the
    /// one the tokenizer itself applies, so the prompt path and the tokenizer
    /// agree on every checkpoint.
    pub fn effective_add_bos_token(&self) -> Result<bool> {
        let profile = BpeProfile::from_metadata(&self.model, &self.pre_tokenizer)?;
        Ok(self
            .add_bos_token
            .unwrap_or_else(|| profile.default_add_bos()))
    }

    pub fn build(self) -> Result<Tokenizer> {
        let profile = BpeProfile::from_metadata(&self.model, &self.pre_tokenizer)?;
        ensure!(
            !self.add_space_prefix,
            "the {} tokenizer profile requires add_space_prefix=false",
            self.pre_tokenizer
        );
        let (bos_token_id, eos_token_id) = self.special_token_ids(profile)?;
        ensure!(
            !self.tokens.is_empty(),
            "embedded tokenizer vocabulary is empty"
        );
        ensure!(
            self.tokens.len() == self.token_types.len(),
            "tokenizer tokens and token types have different lengths"
        );

        let mut vocab = Vocab::with_capacity(self.tokens.len());
        let mut seen_tokens = HashSet::with_capacity(self.tokens.len());
        let mut added_tokens = Vec::new();
        let mut added_token_ids = Vec::new();
        for (index, (token, token_type)) in self.tokens.iter().zip(&self.token_types).enumerate() {
            ensure!(!token.is_empty(), "tokenizer token {index} is empty");
            let token_id = u32::try_from(index).map_err(|_| anyhow!("token ID exceeds u32"))?;
            ensure!(
                seen_tokens.insert(token.as_str()),
                "duplicate tokenizer token `{token}`"
            );
            match *token_type {
                TOKEN_TYPE_NORMAL => {
                    vocab.insert(token.clone(), token_id);
                }
                TOKEN_TYPE_UNUSED => {
                    // Converter-created padding entries reserve an output ID
                    // but are not real vocabulary tokens. Leaving the ID
                    // absent makes direct decode return no text, matching the
                    // official tokenizer and llama.cpp.
                }
                TOKEN_TYPE_CONTROL => {
                    vocab.insert(token.clone(), token_id);
                    added_tokens.push(AddedToken::from(token.clone(), true).normalized(false));
                    added_token_ids.push((token.clone(), token_id));
                }
                TOKEN_TYPE_USER_DEFINED => {
                    // llama.cpp's GGUF converter pre-normalizes user-defined
                    // tokens. Register them as literal, non-special AddedTokens:
                    // they remain atomic while decode(skip_special=true) keeps
                    // their text, matching Hugging Face's Qwen tokenizer.
                    vocab.insert(token.clone(), token_id);
                    added_tokens.push(AddedToken::from(token.clone(), false).normalized(false));
                    added_token_ids.push((token.clone(), token_id));
                }
                unsupported => {
                    return Err(anyhow!(
                        "unsupported GGUF token type {unsupported} for token {index}"
                    ));
                }
            }
        }

        let merges = self
            .merges
            .iter()
            .enumerate()
            .map(|(index, merge)| parse_merge(index, merge))
            .collect::<Result<Vec<_>>>()?;
        let model = BPE::builder()
            .vocab_and_merges(vocab, merges)
            // Qwen2 uses a byte-level alphabet in the vocabulary; that is
            // distinct from BPE's `<0xNN>` byte-fallback mode. The official
            // Qwen2/Qwen2.5 tokenizer explicitly sets byte_fallback=false.
            .byte_fallback(false)
            .ignore_merges(profile.ignore_merges())
            .build()
            .map_err(|error| anyhow!(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(model);
        match profile {
            BpeProfile::SmolLm | BpeProfile::Smaug | BpeProfile::Refact => {}
            BpeProfile::Qwen2 => {
                tokenizer
                    .with_normalizer(Some(NFC))
                    .map_err(|error| anyhow!(error.to_string()))?;
                tokenizer.with_post_processor(Some(ByteLevel::new(false, false, false)));
            }
            BpeProfile::Qwen35 => {
                tokenizer.with_post_processor(Some(ByteLevel::new(false, false, false)));
            }
            BpeProfile::Llama3 | BpeProfile::Lfm2 => {
                let template = special_token_template(&self.tokens, bos_token_id, eos_token_id)?;
                tokenizer.with_post_processor(Some(PostProcessorSequence::new(vec![
                    PostProcessorWrapper::ByteLevel(ByteLevel::new(true, false, true)),
                    PostProcessorWrapper::Template(template),
                ])));
            }
        }
        tokenizer.with_pre_tokenizer(Some(profile.pre_tokenizer()?));
        tokenizer.with_decoder(Some(profile.decoder()));
        tokenizer
            .add_tokens(added_tokens)
            .map_err(|error| anyhow!(error.to_string()))?;
        for (token, expected_id) in added_token_ids {
            ensure!(
                tokenizer.token_to_id(&token) == Some(expected_id),
                "embedded tokenizer changed the GGUF ID of token `{token}` from {expected_id}"
            );
        }
        Ok(tokenizer)
    }

    fn special_token_ids(&self, profile: BpeProfile) -> Result<(Option<u32>, Option<u32>)> {
        let prepend_beginning = self.effective_add_bos_token()?;
        let append_ending = self.add_eos_token.unwrap_or(false);
        ensure!(
            matches!(profile, BpeProfile::Llama3 | BpeProfile::Lfm2)
                || (!prepend_beginning && !append_ending),
            "automatic BOS/EOS insertion is only implemented for Llama 3-compatible and lfm2 profiles"
        );
        let beginning = prepend_beginning
            .then(|| {
                self.bos_token_id
                    .context("BPE add_bos_token=true requires a BOS token ID")
            })
            .transpose()?;
        let ending = append_ending
            .then(|| {
                self.eos_token_id
                    .context("BPE add_eos_token=true requires an EOS token ID")
            })
            .transpose()?;
        Ok((beginning, ending))
    }
}

/// SentencePiece-style tokenizer embedded in a GGUF file.
///
/// Gemma 3 uses the GGUF compatibility pair `model=llama`, `pre=default`:
/// token scores prioritize `SentencePiece` BPE merges, Metaspace represents
/// whitespace with `▁`, and type-6 `<0xNN>` pieces provide lossless byte
/// fallback. This is deliberately not a Unigram model: llama.cpp maps
/// `tokenizer.ggml.model=llama` to its score-driven SPM BPE implementation.
/// Token IDs remain the indices of the three parallel GGUF arrays.
#[derive(Debug, Clone)]
pub struct EmbeddedSentencePieceTokenizer {
    pub model: String,
    pub pre_tokenizer: String,
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub token_types: Vec<i32>,
    pub add_space_prefix: bool,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unknown_token_id: Option<u32>,
}

impl EmbeddedSentencePieceTokenizer {
    /// The effective policy of the tokenizer's BOS token.
    ///
    /// The metadata key is optional: a `SentencePiece` tokenizer adds its BOS
    /// token to every prompt, and keeps that default when the key is
    /// missing.
    #[must_use]
    pub fn effective_add_bos_token(&self) -> bool {
        self.add_bos_token.unwrap_or(true)
    }

    pub fn build(self) -> Result<Tokenizer> {
        let unknown_index = self.validate_profile()?;

        let mut seen_tokens = HashSet::with_capacity(self.tokens.len());
        let mut vocab = Vocab::with_capacity(self.tokens.len());
        let mut added_tokens = Vec::new();
        let mut added_token_ids = Vec::new();
        for (index, ((token, score), token_type)) in self
            .tokens
            .iter()
            .zip(&self.scores)
            .zip(&self.token_types)
            .enumerate()
        {
            ensure!(!token.is_empty(), "tokenizer token {index} is empty");
            ensure!(score.is_finite(), "tokenizer score {index} is not finite");
            ensure!(
                seen_tokens.insert(token.as_str()),
                "duplicate tokenizer token `{token}`"
            );
            let token_id = u32::try_from(index).map_err(|_| anyhow!("token ID exceeds u32"))?;
            match *token_type {
                TOKEN_TYPE_NORMAL | TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_UNUSED => {}
                TOKEN_TYPE_BYTE => ensure!(
                    parse_byte_token(token).is_some(),
                    "GGUF byte token {index} must use the `<0xNN>` form"
                ),
                TOKEN_TYPE_CONTROL => {
                    added_tokens.push(AddedToken::from(token.clone(), true).normalized(false));
                    added_token_ids.push((token.clone(), token_id));
                }
                TOKEN_TYPE_USER_DEFINED => {
                    added_tokens.push(AddedToken::from(token.clone(), false).normalized(false));
                    added_token_ids.push((token.clone(), token_id));
                }
                unsupported => {
                    return Err(anyhow!(
                        "unsupported GGUF token type {unsupported} for token {index}"
                    ));
                }
            }
            vocab.insert(token.clone(), token_id);
        }

        let (bos_token_id, eos_token_id) = self.special_token_ids()?;
        let merges =
            sentencepiece_bpe_merges(&self.tokens, &self.scores, &self.token_types, &vocab)?;
        let model = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token(self.tokens[unknown_index].clone())
            .byte_fallback(true)
            .build()
            .map_err(|error| anyhow!(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(model);
        let prepend_scheme = if self.add_space_prefix {
            PrependScheme::Always
        } else {
            PrependScheme::Never
        };
        let metaspace = Metaspace::new('▁', prepend_scheme, true);
        tokenizer.with_pre_tokenizer(Some(metaspace.clone()));
        tokenizer.with_decoder(Some(DecoderSequence::new(vec![
            DecoderWrapper::ByteFallback(ByteFallback::new()),
            DecoderWrapper::Metaspace(metaspace),
        ])));
        let template = special_token_template(&self.tokens, bos_token_id, eos_token_id)?;
        tokenizer.with_post_processor(Some(template));
        tokenizer
            .add_tokens(added_tokens)
            .map_err(|error| anyhow!(error.to_string()))?;
        for (token, expected_id) in added_token_ids {
            ensure!(
                tokenizer.token_to_id(&token) == Some(expected_id),
                "embedded tokenizer changed the GGUF ID of token `{token}` from {expected_id}"
            );
        }
        Ok(tokenizer)
    }

    fn validate_profile(&self) -> Result<usize> {
        ensure!(
            self.model == "llama" && self.pre_tokenizer == "default",
            "unsupported embedded SentencePiece tokenizer profile `{}/{}`",
            self.model,
            self.pre_tokenizer
        );
        ensure!(
            !self.tokens.is_empty(),
            "embedded tokenizer vocabulary is empty"
        );
        ensure!(
            self.tokens.len() == self.scores.len(),
            "tokenizer tokens and scores have different lengths"
        );
        ensure!(
            self.tokens.len() == self.token_types.len(),
            "tokenizer tokens and token types have different lengths"
        );
        let unknown_token_id = self
            .unknown_token_id
            .context("the llama/default tokenizer profile requires an unknown token ID")?;
        let unknown_index = usize::try_from(unknown_token_id)
            .ok()
            .filter(|index| *index < self.tokens.len())
            .with_context(|| {
                format!("GGUF unknown token ID {unknown_token_id} is outside the vocabulary")
            })?;
        ensure!(
            matches!(
                self.token_types[unknown_index],
                TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_CONTROL
            ),
            "GGUF unknown token ID {unknown_token_id} has token type {}, expected unknown or control",
            self.token_types[unknown_index]
        );
        Ok(unknown_index)
    }

    fn special_token_ids(&self) -> Result<(Option<u32>, Option<u32>)> {
        let beginning = self
            .effective_add_bos_token()
            .then(|| {
                self.bos_token_id
                    .context("llama/default add_bos_token=true requires a BOS token ID")
            })
            .transpose()?;
        let ending = self
            .add_eos_token
            .unwrap_or(false)
            .then(|| {
                self.eos_token_id
                    .context("llama/default add_eos_token=true requires an EOS token ID")
            })
            .transpose()?;
        Ok((beginning, ending))
    }
}

/// Reconstruct the implicit `SentencePiece` BPE merge table carried by GGUF
/// token scores. llama.cpp considers every adjacent pair whose concatenation
/// is a vocabulary token and merges the pair with the highest target-token
/// score first. `tokenizers::BPE` represents the same ordering as ascending
/// integer ranks, so all valid splits are emitted in descending score order.
fn sentencepiece_bpe_merges(
    tokens: &[String],
    scores: &[f32],
    token_types: &[i32],
    vocab: &Vocab,
) -> Result<Vec<(String, String)>> {
    let mut targets = token_types
        .iter()
        .enumerate()
        .filter_map(|(index, token_type)| (*token_type == TOKEN_TYPE_NORMAL).then_some(index))
        .collect::<Vec<_>>();
    targets.sort_unstable_by(|left, right| {
        scores[*right]
            .total_cmp(&scores[*left])
            .then_with(|| left.cmp(right))
    });

    let mut merges = Vec::new();
    for target in targets {
        let token = &tokens[target];
        for (split, _) in token.char_indices().skip(1) {
            let (left, right) = token.split_at(split);
            if vocab.contains_key(left) && vocab.contains_key(right) {
                merges.push((left.to_owned(), right.to_owned()));
            }
        }
    }
    ensure!(
        u32::try_from(merges.len()).is_ok(),
        "SentencePiece BPE merge count exceeds u32"
    );
    Ok(merges)
}

fn parse_byte_token(token: &str) -> Option<u8> {
    (token.len() == 6 && token.starts_with("<0x") && token.ends_with('>'))
        .then(|| u8::from_str_radix(&token[3..5], 16).ok())
        .flatten()
}

fn special_token_template(
    tokens: &[String],
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
) -> Result<TemplateProcessing> {
    let special_token = |kind: &str, token_id: u32| -> Result<(String, u32)> {
        let token = usize::try_from(token_id)
            .ok()
            .and_then(|index| tokens.get(index))
            .with_context(|| {
                format!("GGUF {kind} token ID {token_id} is outside the vocabulary")
            })?;
        Ok((token.clone(), token_id))
    };
    let bos = bos_token_id
        .map(|token_id| special_token("BOS", token_id))
        .transpose()?;
    let eos = eos_token_id
        .map(|token_id| special_token("EOS", token_id))
        .transpose()?;

    let mut single = Vec::new();
    let mut pair = Vec::new();
    let mut special_tokens = Vec::new();
    if let Some((token, token_id)) = &bos {
        single.push(token.as_str());
        pair.push(token.as_str());
        special_tokens.push((token.as_str(), *token_id));
    }
    single.push("$A");
    pair.extend(["$A", "$B:1"]);
    if let Some((token, token_id)) = &eos {
        single.push(token.as_str());
        pair.push(token.as_str());
        special_tokens.push((token.as_str(), *token_id));
    }
    TemplateProcessing::builder()
        .try_single(single)
        .map_err(anyhow::Error::msg)?
        .try_pair(pair)
        .map_err(anyhow::Error::msg)?
        .special_tokens(special_tokens)
        .build()
        .map_err(|error| anyhow!(error.to_string()))
}

fn parse_merge(index: usize, merge: &str) -> Result<(String, String)> {
    let mut pieces = merge.split(' ');
    let left = pieces.next().unwrap_or_default();
    let right = pieces.next().unwrap_or_default();
    ensure!(
        !left.is_empty() && !right.is_empty() && pieces.next().is_none(),
        "invalid tokenizer merge {index}: expected exactly two non-empty tokens"
    );
    Ok((left.to_owned(), right.to_owned()))
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::*;

    fn tiny_smollm_tokenizer() -> EmbeddedBpeTokenizer {
        EmbeddedBpeTokenizer {
            model: "gpt2".into(),
            pre_tokenizer: "smollm".into(),
            tokens: vec!["<|im_start|>".into(), "a".into(), "1".into(), "2".into()],
            token_types: vec![3, 1, 1, 1],
            merges: Vec::new(),
            add_space_prefix: false,
            add_bos_token: Some(false),
            add_eos_token: Some(false),
            bos_token_id: None,
            eos_token_id: None,
        }
    }

    fn tiny_qwen2_tokenizer() -> (EmbeddedBpeTokenizer, u32, u32, u32, u32, u32, u32) {
        let mut tokens = ByteLevel::alphabet()
            .into_iter()
            .map(|character| character.to_string())
            .collect::<Vec<_>>();
        tokens.sort_unstable();
        let one_id = u32::try_from(
            tokens
                .iter()
                .position(|token| token == "1")
                .expect("ByteLevel alphabet contains ASCII digits"),
        )
        .unwrap();
        let two_id = u32::try_from(
            tokens
                .iter()
                .position(|token| token == "2")
                .expect("ByteLevel alphabet contains ASCII digits"),
        )
        .unwrap();
        let mut token_types = vec![TOKEN_TYPE_NORMAL; tokens.len()];

        let mut push = |token: &str, token_type: i32| {
            let id = u32::try_from(tokens.len()).unwrap();
            tokens.push(token.to_owned());
            token_types.push(token_type);
            id
        };
        let merged_digit_id = push("12", TOKEN_TYPE_NORMAL);
        let control_id = push("<|im_end|>", TOKEN_TYPE_CONTROL);
        let user_defined_id = push("<tool_call>", TOKEN_TYPE_USER_DEFINED);
        let unused_id = push("[PAD-hole]", TOKEN_TYPE_UNUSED);

        (
            EmbeddedBpeTokenizer {
                model: "gpt2".into(),
                pre_tokenizer: "qwen2".into(),
                tokens,
                token_types,
                merges: vec!["1 2".into()],
                add_space_prefix: false,
                add_bos_token: Some(false),
                add_eos_token: Some(false),
                bos_token_id: None,
                eos_token_id: None,
            },
            one_id,
            two_id,
            merged_digit_id,
            control_id,
            user_defined_id,
            unused_id,
        )
    }

    fn tiny_llama3_tokenizer() -> (EmbeddedBpeTokenizer, u32, u32, u32) {
        let mut tokens = ByteLevel::alphabet()
            .into_iter()
            .map(|character| character.to_string())
            .collect::<Vec<_>>();
        tokens.sort_unstable();
        let four_id = u32::try_from(
            tokens
                .iter()
                .position(|token| token == "4")
                .expect("ByteLevel alphabet contains ASCII digits"),
        )
        .unwrap();
        let mut token_types = vec![TOKEN_TYPE_NORMAL; tokens.len()];
        let three_digits_id = u32::try_from(tokens.len()).unwrap();
        tokens.push("123".into());
        token_types.push(TOKEN_TYPE_NORMAL);
        let bos_id = u32::try_from(tokens.len()).unwrap();
        tokens.push("<|begin_of_text|>".into());
        token_types.push(TOKEN_TYPE_CONTROL);

        (
            EmbeddedBpeTokenizer {
                model: "gpt2".into(),
                pre_tokenizer: "llama-bpe".into(),
                tokens,
                token_types,
                merges: Vec::new(),
                add_space_prefix: false,
                add_bos_token: None,
                add_eos_token: None,
                bos_token_id: Some(bos_id),
                eos_token_id: None,
            },
            three_digits_id,
            four_id,
            bos_id,
        )
    }

    fn tiny_gemma_sentencepiece_tokenizer() -> EmbeddedSentencePieceTokenizer {
        EmbeddedSentencePieceTokenizer {
            model: "llama".into(),
            pre_tokenizer: "default".into(),
            tokens: vec![
                "<unk>".into(),
                "<bos>".into(),
                "<eos>".into(),
                "▁".into(),
                "h".into(),
                "e".into(),
                "l".into(),
                "o".into(),
                "he".into(),
                "hel".into(),
                "hell".into(),
                "hello".into(),
                "▁hello".into(),
                "w".into(),
                "r".into(),
                "d".into(),
                "wo".into(),
                "wor".into(),
                "worl".into(),
                "world".into(),
                "▁world".into(),
                "<tool>".into(),
                "<0xE2>".into(),
                "<0x98>".into(),
                "<0x83>".into(),
            ],
            scores: vec![
                -1000.0, -1000.0, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 90.0, 80.0, 70.0, 60.0,
                0.0, 0.0, 0.0, 100.0, 90.0, 80.0, 70.0, 60.0, -1000.0, 0.0, 0.0, 0.0,
            ],
            token_types: vec![
                // Current Gemma 3 GGUFs classify `<unk>` as control while
                // older SentencePiece converters use the dedicated unknown
                // type. Both carry the same `unknown_token_id` semantics.
                TOKEN_TYPE_CONTROL,
                TOKEN_TYPE_CONTROL,
                TOKEN_TYPE_CONTROL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_USER_DEFINED,
                TOKEN_TYPE_BYTE,
                TOKEN_TYPE_BYTE,
                TOKEN_TYPE_BYTE,
            ],
            add_space_prefix: true,
            add_bos_token: None,
            add_eos_token: None,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            unknown_token_id: Some(0),
        }
    }

    fn tiny_score_ordered_sentencepiece_tokenizer() -> EmbeddedSentencePieceTokenizer {
        EmbeddedSentencePieceTokenizer {
            model: "llama".into(),
            pre_tokenizer: "default".into(),
            tokens: vec![
                "<unk>".into(),
                "<bos>".into(),
                "<eos>".into(),
                "a".into(),
                "b".into(),
                "c".into(),
                "ab".into(),
                "bc".into(),
            ],
            scores: vec![-1000.0, -1000.0, -1000.0, 0.0, 0.0, 0.0, 10.0, 20.0],
            token_types: vec![
                TOKEN_TYPE_CONTROL,
                TOKEN_TYPE_CONTROL,
                TOKEN_TYPE_CONTROL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
                TOKEN_TYPE_NORMAL,
            ],
            add_space_prefix: false,
            add_bos_token: Some(false),
            add_eos_token: Some(false),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            unknown_token_id: Some(0),
        }
    }

    #[test]
    fn builds_smollm_byte_bpe_and_preserves_control_token_ids() -> Result<()> {
        let tokenizer = tiny_smollm_tokenizer().build()?;
        assert!(tokenizer.get_normalizer().is_none());
        let encoding = tokenizer
            .encode("<|im_start|>a12", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(encoding.get_ids(), &[0, 1, 2, 3]);
        assert_eq!(
            tokenizer
                .decode(&[1, 2, 3], false)
                .map_err(|error| anyhow!(error.to_string()))?,
            "a12"
        );
        Ok(())
    }

    #[test]
    fn builds_granite_refact_profile_with_digit_isolation() -> Result<()> {
        let mut metadata = tiny_smollm_tokenizer();
        metadata.pre_tokenizer = "refact".into();
        let tokenizer = metadata.build()?;
        assert!(tokenizer.get_normalizer().is_none());
        let encoding = tokenizer
            .encode("<|im_start|>a12", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(encoding.get_ids(), &[0, 1, 2, 3]);
        assert_eq!(
            tokenizer
                .decode(&[1, 2, 3], false)
                .map_err(|error| anyhow!(error.to_string()))?,
            "a12"
        );
        Ok(())
    }

    #[test]
    fn builds_exact_qwen2_profile_and_preserves_gguf_token_ids() -> Result<()> {
        let (metadata, one_id, two_id, merged_digit_id, control_id, user_defined_id, unused_id) =
            tiny_qwen2_tokenizer();
        let tokenizer = metadata.build()?;
        assert!(tokenizer.get_normalizer().is_some());

        // Qwen's `\p{N}` alternative isolates one digit at a time, so a BPE
        // merge that would otherwise form `12` must not cross that boundary.
        let digits = tokenizer
            .encode("12", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(digits.get_ids(), &[one_id, two_id]);
        assert!(!digits.get_ids().contains(&merged_digit_id));

        let added = tokenizer
            .encode("<|im_end|><tool_call>", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(added.get_ids(), &[control_id, user_defined_id]);
        assert_eq!(tokenizer.token_to_id("<|im_end|>"), Some(control_id));
        assert_eq!(tokenizer.token_to_id("<tool_call>"), Some(user_defined_id));
        assert_eq!(tokenizer.token_to_id("[PAD-hole]"), None);
        for skip_special in [false, true] {
            assert_eq!(
                tokenizer
                    .decode(&[unused_id], skip_special)
                    .map_err(|error| anyhow!(error.to_string()))?,
                ""
            );
        }
        assert_eq!(
            tokenizer
                .decode(added.get_ids(), true)
                .map_err(|error| anyhow!(error.to_string()))?,
            "<tool_call>"
        );

        let unicode = " café 🦀\n";
        let encoding = tokenizer
            .encode(unicode, false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(
            tokenizer
                .decode(encoding.get_ids(), false)
                .map_err(|error| anyhow!(error.to_string()))?,
            unicode
        );
        let precomposed = tokenizer
            .encode("café", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        let decomposed = tokenizer
            .encode("cafe\u{301}", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(decomposed.get_ids(), precomposed.get_ids());
        Ok(())
    }

    #[test]
    fn qwen35_keeps_combining_marks_in_the_letter_pretoken() -> Result<()> {
        // Arabic kasra remains a separate combining mark after NFC. Build a
        // merge chain that can collapse the entire letter+mark only when the
        // split regex keeps both code points in one pre-token.
        let text = "ب\u{650}";
        let (mut metadata, ..) = tiny_qwen2_tokenizer();
        metadata.merges.clear();
        metadata.pre_tokenizer = "qwen35".into();
        let probe = metadata.clone().build()?;
        let byte_tokens = probe
            .encode(text, false)
            .map_err(|error| anyhow!(error.to_string()))?
            .get_tokens()
            .to_vec();
        assert!(byte_tokens.len() > 2);

        let mut merged = byte_tokens[0].clone();
        let mut final_id = 0;
        for token in &byte_tokens[1..] {
            metadata.merges.push(format!("{merged} {token}"));
            merged.push_str(token);
            final_id = u32::try_from(metadata.tokens.len())?;
            metadata.tokens.push(merged.clone());
            metadata.token_types.push(TOKEN_TYPE_NORMAL);
        }

        let qwen35 = metadata.clone().build()?;
        assert!(qwen35.get_normalizer().is_none());
        let qwen35_ids = qwen35
            .encode(text, false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(qwen35_ids.get_ids(), &[final_id]);

        metadata.pre_tokenizer = "qwen2".into();
        let qwen2 = metadata.build()?;
        let qwen2_ids = qwen2
            .encode(text, false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert!(!qwen2_ids.get_ids().contains(&final_id));
        Ok(())
    }

    #[test]
    fn builds_exact_llama3_profile_with_bos_and_ignore_merges() -> Result<()> {
        let (metadata, three_digits_id, four_id, bos_id) = tiny_llama3_tokenizer();
        let tokenizer = metadata.clone().build()?;
        assert!(tokenizer.get_normalizer().is_none());

        let without_special = tokenizer
            .encode("1234", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(without_special.get_ids(), &[three_digits_id, four_id]);
        let with_special = tokenizer
            .encode("1234", true)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(with_special.get_ids(), &[bos_id, three_digits_id, four_id]);
        let explicit_bos = tokenizer
            .encode("<|begin_of_text|>123", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(explicit_bos.get_ids(), &[bos_id, three_digits_id]);
        let doubled_bos = tokenizer
            .encode("<|begin_of_text|>123", true)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(doubled_bos.get_ids(), &[bos_id, bos_id, three_digits_id]);

        let mut no_automatic_bos = metadata;
        no_automatic_bos.add_bos_token = Some(false);
        let no_automatic_bos = no_automatic_bos.build()?;
        assert_eq!(
            no_automatic_bos
                .encode("123", true)
                .map_err(|error| anyhow!(error.to_string()))?
                .get_ids(),
            &[three_digits_id]
        );

        let precomposed = tokenizer
            .encode("café", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        let decomposed = tokenizer
            .encode("cafe\u{301}", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_ne!(decomposed.get_ids(), precomposed.get_ids());
        for text in ["I've", "1234567", "café 👋\n"] {
            let encoding = tokenizer
                .encode(text, false)
                .map_err(|error| anyhow!(error.to_string()))?;
            assert_eq!(
                tokenizer
                    .decode(encoding.get_ids(), false)
                    .map_err(|error| anyhow!(error.to_string()))?,
                text
            );
        }
        Ok(())
    }

    #[test]
    fn builds_exact_smaug_profile_without_automatic_bos() -> Result<()> {
        let (mut metadata, three_digits_id, four_id, bos_id) = tiny_llama3_tokenizer();
        metadata.pre_tokenizer = "smaug-bpe".into();
        metadata.add_bos_token = None;
        let tokenizer = metadata.build()?;
        assert!(tokenizer.get_normalizer().is_none());

        // Smaug uses the same Llama 3 regex and ignore-merges behavior, but
        // the SmolLM3 tokenizer's TemplateProcessing adds no automatic token.
        for add_special_tokens in [false, true] {
            let encoding = tokenizer
                .encode("1234", add_special_tokens)
                .map_err(|error| anyhow!(error.to_string()))?;
            assert_eq!(encoding.get_ids(), &[three_digits_id, four_id]);
        }
        let explicit_bos = tokenizer
            .encode("<|begin_of_text|>123", true)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(explicit_bos.get_ids(), &[bos_id, three_digits_id]);
        for text in ["I've", "1234567", "café 👋\n"] {
            let encoding = tokenizer
                .encode(text, false)
                .map_err(|error| anyhow!(error.to_string()))?;
            assert_eq!(
                tokenizer
                    .decode(encoding.get_ids(), false)
                    .map_err(|error| anyhow!(error.to_string()))?,
                text
            );
        }
        Ok(())
    }

    #[test]
    fn falcon_h1_uses_the_exact_llama3_bpe_profile() -> Result<()> {
        let (llama3_metadata, three_digits_id, four_id, bos_id) = tiny_llama3_tokenizer();
        let llama3 = llama3_metadata.clone().build()?;

        let mut falcon_metadata = llama3_metadata;
        falcon_metadata.pre_tokenizer = "falcon-h1".into();
        let falcon_h1 = falcon_metadata.build()?;

        for (text, add_special_tokens) in [("1234", false), ("1234", true), ("I've", true)] {
            let llama3_ids = llama3
                .encode(text, add_special_tokens)
                .map_err(|error| anyhow!(error.to_string()))?;
            let falcon_ids = falcon_h1
                .encode(text, add_special_tokens)
                .map_err(|error| anyhow!(error.to_string()))?;
            assert_eq!(falcon_ids.get_ids(), llama3_ids.get_ids());
        }
        assert_eq!(
            falcon_h1
                .encode("1234", true)
                .map_err(|error| anyhow!(error.to_string()))?
                .get_ids(),
            &[bos_id, three_digits_id, four_id]
        );
        Ok(())
    }

    #[test]
    fn builds_gemma_sentencepiece_profile_with_scores_and_automatic_bos() -> Result<()> {
        let metadata = tiny_gemma_sentencepiece_tokenizer();
        let tokenizer = metadata.clone().build()?;
        assert!(tokenizer.get_normalizer().is_none());

        let without_special = tokenizer
            .encode("hello world", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(without_special.get_ids(), &[12, 20]);
        let with_special = tokenizer
            .encode("hello world", true)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(with_special.get_ids(), &[1, 12, 20]);
        assert_eq!(
            tokenizer
                .decode(with_special.get_ids(), true)
                .map_err(|error| anyhow!(error.to_string()))?,
            "hello world"
        );

        let mut no_space_prefix = metadata.clone();
        no_space_prefix.add_space_prefix = false;
        let no_space_prefix = no_space_prefix.build()?;
        assert_eq!(
            no_space_prefix
                .encode("world", false)
                .map_err(|error| anyhow!(error.to_string()))?
                .get_ids(),
            &[19]
        );

        let mut no_automatic_bos = metadata;
        no_automatic_bos.add_bos_token = Some(false);
        let no_automatic_bos = no_automatic_bos.build()?;
        assert_eq!(
            no_automatic_bos
                .encode("hello", true)
                .map_err(|error| anyhow!(error.to_string()))?
                .get_ids(),
            &[12]
        );
        Ok(())
    }

    #[test]
    fn sentencepiece_bpe_uses_target_scores_as_merge_priority() -> Result<()> {
        let metadata = tiny_score_ordered_sentencepiece_tokenizer();
        let higher_bc = metadata.clone().build()?;
        assert_eq!(
            higher_bc
                .encode("abc", false)
                .map_err(|error| anyhow!(error.to_string()))?
                .get_ids(),
            &[3, 7]
        );

        let mut higher_ab = metadata;
        higher_ab.scores[6] = 30.0;
        let higher_ab = higher_ab.build()?;
        assert_eq!(
            higher_ab
                .encode("abc", false)
                .map_err(|error| anyhow!(error.to_string()))?
                .get_ids(),
            &[6, 5]
        );
        Ok(())
    }

    #[test]
    fn gemma_sentencepiece_preserves_added_tokens_and_byte_fallback() -> Result<()> {
        let tokenizer = tiny_gemma_sentencepiece_tokenizer().build()?;

        let added = tokenizer
            .encode("<bos><tool>", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(added.get_ids(), &[1, 21]);
        assert_eq!(
            tokenizer
                .decode(added.get_ids(), true)
                .map_err(|error| anyhow!(error.to_string()))?,
            "<tool>"
        );

        let fallback = tokenizer
            .encode("hello ☃", false)
            .map_err(|error| anyhow!(error.to_string()))?;
        assert_eq!(fallback.get_ids(), &[12, 3, 22, 23, 24]);
        assert_eq!(
            tokenizer
                .decode(fallback.get_ids(), false)
                .map_err(|error| anyhow!(error.to_string()))?,
            "hello ☃"
        );
        assert_eq!(
            tokenizer
                .decode(&[22, 23, 24], false)
                .map_err(|error| anyhow!(error.to_string()))?,
            "☃"
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_gemma_sentencepiece_metadata() {
        let mut metadata = tiny_gemma_sentencepiece_tokenizer();
        metadata.scores.pop();
        assert!(metadata.build().is_err());

        let mut metadata = tiny_gemma_sentencepiece_tokenizer();
        metadata.tokens[22] = "not-a-byte".into();
        assert!(metadata.build().is_err());

        let mut metadata = tiny_gemma_sentencepiece_tokenizer();
        metadata.unknown_token_id = None;
        assert!(metadata.build().is_err());

        let mut metadata = tiny_gemma_sentencepiece_tokenizer();
        metadata.pre_tokenizer = "gemma".into();
        assert!(metadata.build().is_err());

        let mut metadata = tiny_gemma_sentencepiece_tokenizer();
        metadata.token_types[0] = TOKEN_TYPE_NORMAL;
        assert!(metadata.build().is_err());

        let mut legacy_unknown_type = tiny_gemma_sentencepiece_tokenizer();
        legacy_unknown_type.token_types[0] = TOKEN_TYPE_UNKNOWN;
        assert!(legacy_unknown_type.build().is_ok());
    }

    #[test]
    fn rejects_profiles_that_would_be_loaded_incorrectly() {
        let mut metadata = tiny_smollm_tokenizer();
        metadata.pre_tokenizer = "default".into();
        assert!(metadata.build().is_err());

        let mut metadata = tiny_smollm_tokenizer();
        metadata.token_types[1] = 6;
        assert!(metadata.build().is_err());

        let mut metadata = tiny_smollm_tokenizer();
        metadata.merges.push("three invalid parts".into());
        assert!(metadata.build().is_err());

        let (mut metadata, ..) = tiny_qwen2_tokenizer();
        metadata.model = "qwen2".into();
        assert!(metadata.build().is_err());

        let (mut metadata, ..) = tiny_qwen2_tokenizer();
        metadata.pre_tokenizer = "gpt2".into();
        assert!(metadata.build().is_err());

        let (mut metadata, ..) = tiny_llama3_tokenizer();
        metadata.bos_token_id = None;
        assert!(metadata.build().is_err());

        let (mut metadata, ..) = tiny_llama3_tokenizer();
        metadata.bos_token_id = Some(u32::MAX);
        assert!(metadata.build().is_err());
    }

    fn embedded_qwen2_from_hf_json(json: &serde_json::Value) -> Result<Tokenizer> {
        let vocab = json["model"]["vocab"]
            .as_object()
            .ok_or_else(|| anyhow!("official tokenizer model.vocab is not an object"))?;
        let added = json["added_tokens"]
            .as_array()
            .ok_or_else(|| anyhow!("official tokenizer added_tokens is not an array"))?;
        let vocab_len = vocab
            .values()
            .chain(added.iter().map(|token| &token["id"]))
            .map(|id| {
                id.as_u64()
                    .ok_or_else(|| anyhow!("official tokenizer ID is not u64"))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .ok_or_else(|| anyhow!("official tokenizer vocabulary is empty"))?
            .checked_add(1)
            .ok_or_else(|| anyhow!("official tokenizer vocabulary length overflow"))?;
        let mut tokens = vec![None; usize::try_from(vocab_len)?];
        let mut token_types = vec![TOKEN_TYPE_NORMAL; tokens.len()];
        for (token, id) in vocab {
            let id = usize::try_from(
                id.as_u64()
                    .ok_or_else(|| anyhow!("official tokenizer ID is not u64"))?,
            )?;
            tokens[id] = Some(token.clone());
        }
        for token in added {
            let id = usize::try_from(
                token["id"]
                    .as_u64()
                    .ok_or_else(|| anyhow!("official added-token ID is not u64"))?,
            )?;
            let content = token["content"]
                .as_str()
                .ok_or_else(|| anyhow!("official added-token content is not a string"))?;
            tokens[id] = Some(content.to_owned());
            token_types[id] = if token["special"].as_bool() == Some(true) {
                TOKEN_TYPE_CONTROL
            } else {
                TOKEN_TYPE_USER_DEFINED
            };
        }
        let tokens = tokens
            .into_iter()
            .enumerate()
            .map(|(id, token)| token.ok_or_else(|| anyhow!("official tokenizer ID {id} is absent")))
            .collect::<Result<Vec<_>>>()?;
        let merges = json["model"]["merges"]
            .as_array()
            .ok_or_else(|| anyhow!("official tokenizer model.merges is not an array"))?
            .iter()
            .map(|merge| {
                merge
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| anyhow!("official tokenizer merge is not a string"))
            })
            .collect::<Result<Vec<_>>>()?;
        EmbeddedBpeTokenizer {
            model: "gpt2".into(),
            pre_tokenizer: "qwen2".into(),
            tokens,
            token_types,
            merges,
            add_space_prefix: false,
            add_bos_token: Some(false),
            add_eos_token: Some(false),
            bos_token_id: None,
            eos_token_id: None,
        }
        .build()
    }

    fn embedded_sentencepiece_from_gguf(path: &std::path::Path) -> Result<Tokenizer> {
        use crate::gguf::{GgufFile, GgufValue};

        let gguf = GgufFile::open(path)?;
        let metadata = gguf.metadata();
        let array = |key: &str| {
            metadata
                .array(key)
                .with_context(|| format!("GGUF metadata `{key}` is not an array"))
        };
        let tokens = array("tokenizer.ggml.tokens")?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .with_context(|| format!("GGUF token {index} is not a string"))
            })
            .collect::<Result<Vec<_>>>()?;
        let scores = array("tokenizer.ggml.scores")?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_f32()
                    .with_context(|| format!("GGUF token score {index} is not f32"))
            })
            .collect::<Result<Vec<_>>>()?;
        let token_types = array("tokenizer.ggml.token_type")?
            .iter()
            .enumerate()
            .map(|(index, value)| match value {
                GgufValue::I32(value) => Ok(*value),
                _ => Err(anyhow!("GGUF token type {index} is not i32")),
            })
            .collect::<Result<Vec<_>>>()?;

        EmbeddedSentencePieceTokenizer {
            model: metadata
                .string("tokenizer.ggml.model")
                .context("GGUF tokenizer model is missing")?
                .to_owned(),
            pre_tokenizer: metadata
                .string("tokenizer.ggml.pre")
                .context("GGUF tokenizer pre profile is missing")?
                .to_owned(),
            tokens,
            scores,
            token_types,
            add_space_prefix: metadata
                .bool("tokenizer.ggml.add_space_prefix")
                .unwrap_or(true),
            add_bos_token: metadata.bool("tokenizer.ggml.add_bos_token"),
            add_eos_token: metadata.bool("tokenizer.ggml.add_eos_token"),
            bos_token_id: metadata.u32("tokenizer.ggml.bos_token_id"),
            eos_token_id: metadata.u32("tokenizer.ggml.eos_token_id"),
            unknown_token_id: metadata.u32("tokenizer.ggml.unknown_token_id"),
        }
        .build()
    }

    #[test]
    #[ignore = "requires the pinned Gemma 3 270M Q4_0 GGUF fixture"]
    fn gemma3_270m_gguf_tokenizer_matches_llama_cpp_goldens() -> Result<()> {
        let path = std::env::var_os("MACLLM_GEMMA3_GGUF").map_or_else(
            || {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("models/gemma-3-270m-it-qat-q4_0/gemma-3-270m-it-qat-Q4_0.gguf")
            },
            PathBuf::from,
        );
        let tokenizer = embedded_sentencepiece_from_gguf(&path)?;
        for (text, expected) in [
            ("hello world", vec![2_u32, 23_391, 1_902]),
            ("café ☃", vec![2_u32, 123_125, 236_859, 236_743, 246_690]),
            (
                "<start_of_turn>user\nHello<end_of_turn>\n",
                vec![2_u32, 105, 2_364, 107, 9_259, 106, 107],
            ),
            ("🜏", vec![2_u32, 478, 397, 394, 381]),
        ] {
            let actual = tokenizer
                .encode(text, true)
                .map_err(|error| anyhow!(error.to_string()))?;
            assert_eq!(actual.get_ids(), expected, "token IDs for {text:?}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires the official Qwen2.5 tokenizer.json fixture"]
    fn qwen2_gguf_profile_matches_official_tokenizer_json() -> Result<()> {
        let path = std::env::var_os("MACLLM_QWEN_TOKENIZER").map_or_else(
            || {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("models/qwen2.5-0.5b-instruct/tokenizer.json")
            },
            PathBuf::from,
        );
        let reference = Tokenizer::from_file(&path).map_err(|error| anyhow!(error.to_string()))?;
        let json: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        let embedded = embedded_qwen2_from_hf_json(&json)?;

        for text in [
            "Hello, world!",
            "I've been 'TOLD he's there; 12345",
            "café 🦀 — Rust\n第二行",
            "cafe\u{301} 🦀 — Rust\n第二行",
            "<|im_start|>assistant\nParis<|im_end|>",
            "<tool_call>{\"name\":\"weather\"}</tool_call>",
        ] {
            let expected = reference
                .encode(text, false)
                .map_err(|error| anyhow!(error.to_string()))?;
            let actual = embedded
                .encode(text, false)
                .map_err(|error| anyhow!(error.to_string()))?;
            assert_eq!(
                actual.get_ids(),
                expected.get_ids(),
                "token IDs for {text:?}"
            );
            assert_eq!(
                actual.get_offsets(),
                expected.get_offsets(),
                "token offsets for {text:?}"
            );
            for skip_special in [false, true] {
                assert_eq!(
                    embedded
                        .decode(actual.get_ids(), skip_special)
                        .map_err(|error| anyhow!(error.to_string()))?,
                    reference
                        .decode(expected.get_ids(), skip_special)
                        .map_err(|error| anyhow!(error.to_string()))?,
                    "decoded text for {text:?}, skip_special={skip_special}"
                );
            }
        }
        for (tokenizer, name) in [(&reference, "official"), (&embedded, "embedded GGUF")] {
            let precomposed = tokenizer
                .encode("café", false)
                .map_err(|error| anyhow!(error.to_string()))?;
            let decomposed = tokenizer
                .encode("cafe\u{301}", false)
                .map_err(|error| anyhow!(error.to_string()))?;
            assert_eq!(
                decomposed.get_ids(),
                precomposed.get_ids(),
                "{name} tokenizer must apply NFC before BPE"
            );
        }
        Ok(())
    }
}
