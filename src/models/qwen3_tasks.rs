use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokenizers::Tokenizer;

use super::qwen3::Qwen3Config;

pub const QWEN3_TASK_MAX_TOKENS: usize = 8_192;
pub const DEFAULT_RETRIEVAL_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

const QWEN3_END_OF_TEXT_TOKEN_ID: u32 = 151_643;
const QWEN3_IM_START_TOKEN_ID: u32 = 151_644;
const QWEN3_IM_END_TOKEN_ID: u32 = 151_645;
const QWEN3_THINK_START_TOKEN_ID: u32 = 151_667;
const QWEN3_THINK_END_TOKEN_ID: u32 = 151_668;
const EMBEDDING_QUERY_PROMPT: &str =
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";
const RERANK_PREFIX: &str = concat!(
    "<|im_start|>system\n",
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. ",
    "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n",
    "<|im_start|>user\n"
);
const RERANK_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

const EMBEDDING_MODULES: [(usize, &str, &str, &str); 3] = [
    (0, "0", "", "sentence_transformers.models.Transformer"),
    (1, "1", "1_Pooling", "sentence_transformers.models.Pooling"),
    (
        2,
        "2",
        "2_Normalize",
        "sentence_transformers.models.Normalize",
    ),
];
const RERANK_MODULES: [(usize, &str, &str, &str); 2] = [
    (
        0,
        "0",
        "",
        "sentence_transformers.base.modules.transformer.Transformer",
    ),
    (
        1,
        "1",
        "1_LogitScore",
        "sentence_transformers.cross_encoder.modules.logit_score.LogitScore",
    ),
];

/// Exact sentence-transformers semantics required by Qwen3 embedding models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3EmbeddingProfile {
    dimension: usize,
    vocab_size: usize,
    max_tokens: usize,
}

impl Qwen3EmbeddingProfile {
    pub fn from_model_dir(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let config = load_qwen3_config(root)?;
        ensure!(
            config.eos_token_id == Some(QWEN3_END_OF_TEXT_TOKEN_ID),
            "Qwen3 embedding config must use <|endoftext|> as EOS"
        );
        validate_modules(root, &EMBEDDING_MODULES)?;

        let pooling: PoolingConfig = load_json(&root.join("1_Pooling/config.json"))?;
        ensure!(
            pooling.word_embedding_dimension == config.hidden_size,
            "embedding pooling dimension {} disagrees with Qwen3 hidden size {}",
            pooling.word_embedding_dimension,
            config.hidden_size
        );
        ensure!(
            pooling == PoolingConfig::last_token(config.hidden_size),
            "Qwen3 embedding requires last-token pooling with include_prompt=true"
        );

        let metadata: EmbeddingMetadata =
            load_json(&root.join("config_sentence_transformers.json"))?;
        ensure!(
            metadata == EmbeddingMetadata::official(),
            "unsupported Qwen3 embedding sentence-transformers profile"
        );

        Ok(Self {
            dimension: config.hidden_size,
            vocab_size: config.vocab_size,
            max_tokens: config.max_position_embeddings.min(QWEN3_TASK_MAX_TOKENS),
        })
    }

    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    #[must_use]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn validate_tokenizer(&self, tokenizer: &Tokenizer) -> Result<()> {
        ensure!(
            tokenizer.get_vocab_size(true) == self.vocab_size,
            "Qwen3 embedding tokenizer vocabulary size disagrees with the model"
        );
        ensure!(
            tokenizer.token_to_id("<|endoftext|>") == Some(QWEN3_END_OF_TEXT_TOKEN_ID),
            "Qwen3 embedding tokenizer has an unexpected <|endoftext|> token ID"
        );
        let without_special = encode(tokenizer, "macllm tokenizer probe", false)?;
        let with_special = encode(tokenizer, "macllm tokenizer probe", true)?;
        ensure!(
            with_special.len() == without_special.len() + 1
                && with_special[..without_special.len()] == without_special
                && with_special.last() == Some(&QWEN3_END_OF_TEXT_TOKEN_ID),
            "Qwen3 embedding tokenizer must append exactly one <|endoftext|> token"
        );
        Ok(())
    }

    pub fn tokenize_query(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        instruction: Option<&str>,
    ) -> Result<Vec<u32>> {
        self.validate_tokenizer(tokenizer)?;
        let tokens = encode(tokenizer, &embedding_query(query, instruction)?, true)?;
        validate_token_count("embedding query", tokens.len(), self.max_tokens)?;
        Ok(tokens)
    }

    pub fn tokenize_document(&self, tokenizer: &Tokenizer, document: &str) -> Result<Vec<u32>> {
        self.validate_tokenizer(tokenizer)?;
        validate_text("embedding document", document)?;
        let tokens = encode(tokenizer, document, true)?;
        validate_token_count("embedding document", tokens.len(), self.max_tokens)?;
        Ok(tokens)
    }
}

/// Exact sentence-transformers semantics required by Qwen3 reranker models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3RerankerProfile {
    false_token_id: u32,
    true_token_id: u32,
    vocab_size: usize,
    max_tokens: usize,
}

impl Qwen3RerankerProfile {
    pub fn from_model_dir(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let config = load_qwen3_config(root)?;
        ensure!(
            config.eos_token_id == Some(QWEN3_IM_END_TOKEN_ID),
            "Qwen3 reranker config must use <|im_end|> as EOS"
        );
        validate_modules(root, &RERANK_MODULES)?;

        let labels: LogitScoreConfig = load_json(&root.join("1_LogitScore/config.json"))?;
        ensure!(
            labels.false_token_id != labels.true_token_id,
            "Qwen3 reranker true and false token IDs must differ"
        );
        ensure!(
            (labels.false_token_id as usize) < config.vocab_size
                && (labels.true_token_id as usize) < config.vocab_size,
            "Qwen3 reranker label token ID is outside the model vocabulary"
        );

        let metadata: RerankerMetadata =
            load_json(&root.join("config_sentence_transformers.json"))?;
        metadata.validate()?;

        Ok(Self {
            false_token_id: labels.false_token_id,
            true_token_id: labels.true_token_id,
            vocab_size: config.vocab_size,
            max_tokens: config.max_position_embeddings.min(QWEN3_TASK_MAX_TOKENS),
        })
    }

    #[must_use]
    pub fn no_token_id(&self) -> u32 {
        self.false_token_id
    }

    #[must_use]
    pub fn yes_token_id(&self) -> u32 {
        self.true_token_id
    }

    #[must_use]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn validate_tokenizer(&self, tokenizer: &Tokenizer) -> Result<()> {
        ensure!(
            tokenizer.get_vocab_size(true) == self.vocab_size,
            "Qwen3 reranker tokenizer vocabulary size disagrees with the model"
        );
        for (label, expected) in [("no", self.false_token_id), ("yes", self.true_token_id)] {
            ensure!(
                tokenizer.token_to_id(label) == Some(expected),
                "Qwen3 reranker tokenizer has an unexpected `{label}` token ID"
            );
            ensure!(
                encode(tokenizer, label, false)? == [expected],
                "Qwen3 reranker label `{label}` must encode to exactly one token"
            );
        }
        for (control, expected) in [
            ("<|im_start|>", QWEN3_IM_START_TOKEN_ID),
            ("<|im_end|>", QWEN3_IM_END_TOKEN_ID),
            ("<think>", QWEN3_THINK_START_TOKEN_ID),
            ("</think>", QWEN3_THINK_END_TOKEN_ID),
        ] {
            ensure!(
                tokenizer.token_to_id(control) == Some(expected)
                    && encode(tokenizer, control, false)? == [expected],
                "Qwen3 reranker tokenizer has an unexpected `{control}` control token"
            );
        }
        let prefix_controls = task_control_tokens(&encode(tokenizer, RERANK_PREFIX, false)?);
        let suffix_controls = task_control_tokens(&encode(tokenizer, RERANK_SUFFIX, false)?);
        ensure!(
            prefix_controls
                == [
                    QWEN3_IM_START_TOKEN_ID,
                    QWEN3_IM_END_TOKEN_ID,
                    QWEN3_IM_START_TOKEN_ID,
                ]
                && suffix_controls
                    == [
                        QWEN3_IM_END_TOKEN_ID,
                        QWEN3_IM_START_TOKEN_ID,
                        QWEN3_THINK_START_TOKEN_ID,
                        QWEN3_THINK_END_TOKEN_ID,
                    ],
            "Qwen3 reranker tokenizer changed the prefix or suffix control-token order"
        );
        ensure!(
            encode(tokenizer, "macllm tokenizer probe", true)?
                == encode(tokenizer, "macllm tokenizer probe", false)?,
            "Qwen3 reranker tokenizer must not append automatic special tokens"
        );
        Ok(())
    }

    pub fn tokenize_pair(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        document: &str,
        instruction: Option<&str>,
    ) -> Result<Vec<u32>> {
        self.validate_tokenizer(tokenizer)?;
        let body = rerank_body(query, document, instruction)?;
        let prefix_tokens = encode(tokenizer, RERANK_PREFIX, false)?;
        let body_tokens = encode(tokenizer, &body, false)?;
        let suffix_tokens = encode(tokenizer, RERANK_SUFFIX, false)?;
        let token_count = prefix_tokens
            .len()
            .checked_add(body_tokens.len())
            .and_then(|count| count.checked_add(suffix_tokens.len()))
            .context("Qwen3 reranker token count overflow")?;
        validate_token_count("reranker pair", token_count, self.max_tokens)?;
        let mut tokens = Vec::with_capacity(token_count);
        tokens.extend(prefix_tokens);
        tokens.extend(body_tokens);
        tokens.extend(suffix_tokens);
        Ok(tokens)
    }
}

fn embedding_query(query: &str, instruction: Option<&str>) -> Result<String> {
    validate_text("embedding query", query)?;
    if let Some(instruction) = instruction {
        validate_text("embedding instruction", instruction)?;
        Ok(format!("Instruct: {instruction}\nQuery:{query}"))
    } else {
        Ok(format!("{EMBEDDING_QUERY_PROMPT}{query}"))
    }
}

fn rerank_body(query: &str, document: &str, instruction: Option<&str>) -> Result<String> {
    validate_text("reranker query", query)?;
    validate_text("reranker document", document)?;
    let instruction = instruction.unwrap_or(DEFAULT_RETRIEVAL_INSTRUCTION);
    validate_text("reranker instruction", instruction)?;
    Ok(format!(
        "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    ))
}

fn validate_text(label: &str, text: &str) -> Result<()> {
    ensure!(!text.trim().is_empty(), "{label} must not be empty");
    Ok(())
}

fn validate_token_count(label: &str, actual: usize, maximum: usize) -> Result<()> {
    ensure!(
        actual <= maximum,
        "{label} has {actual} tokens, exceeding the phase-1 limit of {maximum}; automatic truncation is disabled"
    );
    ensure!(actual > 0, "{label} tokenized to an empty sequence");
    Ok(())
}

fn task_control_tokens(tokens: &[u32]) -> Vec<u32> {
    tokens
        .iter()
        .copied()
        .filter(|token| {
            matches!(
                *token,
                QWEN3_IM_START_TOKEN_ID
                    | QWEN3_IM_END_TOKEN_ID
                    | QWEN3_THINK_START_TOKEN_ID
                    | QWEN3_THINK_END_TOKEN_ID
            )
        })
        .collect()
}

fn load_qwen3_config(root: &Path) -> Result<Qwen3Config> {
    ensure!(
        root.is_dir(),
        "model path is not a directory: {}",
        root.display()
    );
    let config = Qwen3Config::from_path(root.join("config.json"))?;
    config.to_spec()?;
    Ok(config)
}

fn load_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("invalid task profile JSON at {}", path.display()))
}

fn encode(tokenizer: &Tokenizer, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
    tokenizer
        .encode(text, add_special_tokens)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SentenceTransformerModule {
    idx: usize,
    name: String,
    path: String,
    #[serde(rename = "type")]
    kind: String,
}

fn validate_modules(root: &Path, expected: &[(usize, &str, &str, &str)]) -> Result<()> {
    let modules: Vec<SentenceTransformerModule> = load_json(&root.join("modules.json"))?;
    ensure!(
        modules.len() == expected.len(),
        "sentence-transformers module count changed: expected {}, found {}",
        expected.len(),
        modules.len()
    );
    for (position, (module, expected)) in modules.iter().zip(expected).enumerate() {
        ensure!(
            module.idx == expected.0
                && module.name == expected.1
                && module.path == expected.2
                && module.kind == expected.3,
            "unsupported sentence-transformers module at position {position}"
        );
    }
    Ok(())
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PoolingConfig {
    word_embedding_dimension: usize,
    pooling_mode_cls_token: bool,
    pooling_mode_mean_tokens: bool,
    pooling_mode_max_tokens: bool,
    pooling_mode_mean_sqrt_len_tokens: bool,
    pooling_mode_weightedmean_tokens: bool,
    pooling_mode_lasttoken: bool,
    include_prompt: bool,
}

impl PoolingConfig {
    fn last_token(dimension: usize) -> Self {
        Self {
            word_embedding_dimension: dimension,
            pooling_mode_cls_token: false,
            pooling_mode_mean_tokens: false,
            pooling_mode_max_tokens: false,
            pooling_mode_mean_sqrt_len_tokens: false,
            pooling_mode_weightedmean_tokens: false,
            pooling_mode_lasttoken: true,
            include_prompt: true,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct EmbeddingMetadata {
    prompts: BTreeMap<String, String>,
    default_prompt_name: Option<String>,
    similarity_fn_name: String,
}

impl EmbeddingMetadata {
    fn official() -> Self {
        Self {
            prompts: [
                ("document".to_owned(), String::new()),
                ("query".to_owned(), EMBEDDING_QUERY_PROMPT.to_owned()),
            ]
            .into_iter()
            .collect(),
            default_prompt_name: None,
            similarity_fn_name: "cosine".to_owned(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LogitScoreConfig {
    true_token_id: u32,
    false_token_id: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RerankerMetadata {
    #[serde(rename = "__version__")]
    versions: BTreeMap<String, String>,
    activation_fn: String,
    default_prompt_name: String,
    model_type: String,
    prompts: BTreeMap<String, String>,
}

impl RerankerMetadata {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.versions.contains_key("sentence_transformers"),
            "Qwen3 reranker metadata has no sentence-transformers version"
        );
        ensure!(
            self.activation_fn == "torch.nn.modules.linear.Identity",
            "Qwen3 reranker requires identity score activation"
        );
        ensure!(
            self.default_prompt_name == "query" && self.model_type == "CrossEncoder",
            "unsupported Qwen3 reranker sentence-transformers profile"
        );
        ensure!(
            self.prompts
                == [("query".to_owned(), DEFAULT_RETRIEVAL_INSTRUCTION.to_owned())]
                    .into_iter()
                    .collect(),
            "unsupported Qwen3 reranker prompt profile"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use super::*;
    use tokenizers::{
        AddedToken, Tokenizer, models::wordlevel::WordLevel,
        pre_tokenizers::whitespace::Whitespace, processors::template::TemplateProcessing,
    };

    fn write_base_config(
        root: &Path,
        hidden_size: usize,
        max_position_embeddings: usize,
        eos_token_id: u32,
    ) -> Result<()> {
        let config = serde_json::json!({
            "model_type": "qwen3",
            "vocab_size": 151_669,
            "hidden_size": hidden_size,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": max_position_embeddings,
            "tie_word_embeddings": true,
            "attention_bias": false,
            "rope_theta": 1_000_000.0,
            "rms_norm_eps": 0.000_001,
            "hidden_act": "silu",
            "rope_scaling": null,
            "sliding_window": null,
            "use_sliding_window": false,
            "eos_token_id": eos_token_id
        });
        fs::write(root.join("config.json"), serde_json::to_vec(&config)?)?;
        Ok(())
    }

    fn write_modules(root: &Path, modules: &[(usize, &str, &str, &str)]) -> Result<()> {
        let modules = modules
            .iter()
            .map(|(idx, name, path, kind)| {
                serde_json::json!({"idx": idx, "name": name, "path": path, "type": kind})
            })
            .collect::<Vec<_>>();
        fs::write(root.join("modules.json"), serde_json::to_vec(&modules)?)?;
        Ok(())
    }

    fn embedding_fixture() -> Result<tempfile::TempDir> {
        let root = tempfile::tempdir()?;
        write_base_config(root.path(), 1_024, 32_768, QWEN3_END_OF_TEXT_TOKEN_ID)?;
        write_modules(root.path(), &EMBEDDING_MODULES)?;
        fs::create_dir(root.path().join("1_Pooling"))?;
        fs::write(
            root.path().join("1_Pooling/config.json"),
            serde_json::to_vec(&PoolingConfig::last_token(1_024))?,
        )?;
        fs::write(
            root.path().join("config_sentence_transformers.json"),
            serde_json::to_vec(&EmbeddingMetadata::official())?,
        )?;
        Ok(root)
    }

    fn reranker_fixture() -> Result<tempfile::TempDir> {
        let root = tempfile::tempdir()?;
        write_base_config(root.path(), 1_024, 40_960, QWEN3_IM_END_TOKEN_ID)?;
        write_modules(root.path(), &RERANK_MODULES)?;
        fs::create_dir(root.path().join("1_LogitScore"))?;
        fs::write(
            root.path().join("1_LogitScore/config.json"),
            r#"{"true_token_id":9693,"false_token_id":2152}"#,
        )?;
        fs::write(
            root.path().join("config_sentence_transformers.json"),
            serde_json::to_vec(&serde_json::json!({
                "__version__": {"pytorch": "2.10.0", "sentence_transformers": "5.4.0"},
                "activation_fn": "torch.nn.modules.linear.Identity",
                "default_prompt_name": "query",
                "model_type": "CrossEncoder",
                "prompts": {"query": DEFAULT_RETRIEVAL_INSTRUCTION}
            }))?,
        )?;
        Ok(root)
    }

    fn build_task_tokenizer(append_eos: bool) -> Tokenizer {
        let mut entries = (0..151_669_u32)
            .map(|token| (format!("[unused-{token}]"), token))
            .collect::<Vec<_>>();
        for (token, id) in [
            ("[UNK]", 0),
            ("macllm", 1),
            ("tokenizer", 2),
            ("probe", 3),
            ("no", 2_152),
            ("yes", 9_693),
            ("<|endoftext|>", QWEN3_END_OF_TEXT_TOKEN_ID),
            ("<|im_start|>", QWEN3_IM_START_TOKEN_ID),
            ("<|im_end|>", QWEN3_IM_END_TOKEN_ID),
            ("<think>", QWEN3_THINK_START_TOKEN_ID),
            ("</think>", QWEN3_THINK_END_TOKEN_ID),
        ] {
            entries[id as usize].0 = token.to_owned();
        }
        let model = WordLevel::builder()
            .vocab(entries.into_iter().collect())
            .unk_token("[UNK]".to_owned())
            .build()
            .expect("full task tokenizer vocabulary is valid");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
            .add_special_tokens([
                AddedToken::from("<|endoftext|>", true),
                AddedToken::from("<|im_start|>", true),
                AddedToken::from("<|im_end|>", true),
            ])
            .expect("task tokenizer special tokens are valid");
        tokenizer
            .add_tokens([
                AddedToken::from("<think>", false),
                AddedToken::from("</think>", false),
            ])
            .expect("task tokenizer added tokens are valid");
        if append_eos {
            let template = TemplateProcessing::builder()
                .try_single(vec!["$A", "<|endoftext|>"])
                .expect("embedding single template is valid")
                .try_pair(vec!["$A", "$B:1", "<|endoftext|>"])
                .expect("embedding pair template is valid")
                .special_tokens(vec![("<|endoftext|>", QWEN3_END_OF_TEXT_TOKEN_ID)])
                .build()
                .expect("embedding tokenizer post-processor is valid");
            tokenizer.with_post_processor(Some(template));
        }
        tokenizer
    }

    fn task_tokenizer(append_eos: bool) -> &'static Tokenizer {
        static WITH_EOS: LazyLock<Tokenizer> = LazyLock::new(|| build_task_tokenizer(true));
        static WITHOUT_EOS: LazyLock<Tokenizer> = LazyLock::new(|| build_task_tokenizer(false));
        if append_eos { &WITH_EOS } else { &WITHOUT_EOS }
    }

    #[test]
    fn embedding_profile_validates_pipeline_and_formats_inputs_exactly() -> Result<()> {
        let root = embedding_fixture()?;
        let profile = Qwen3EmbeddingProfile::from_model_dir(root.path())?;

        assert_eq!(profile.dimension(), 1_024);
        assert_eq!(profile.max_tokens(), QWEN3_TASK_MAX_TOKENS);
        assert_eq!(
            embedding_query("Where?", None)?,
            concat!(
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\n",
                "Query:Where?"
            )
        );
        assert_eq!(
            embedding_query("Where?", Some("Find a place"))?,
            "Instruct: Find a place\nQuery:Where?"
        );
        let tokenizer = task_tokenizer(true);
        assert_eq!(tokenizer.get_vocab_size(true), 151_669);
        profile.validate_tokenizer(tokenizer)?;
        assert_eq!(
            profile.tokenize_document(tokenizer, "macllm")?,
            [1, QWEN3_END_OF_TEXT_TOKEN_ID]
        );
        Ok(())
    }

    #[test]
    fn embedding_profile_rejects_semantic_drift_and_missing_eos_processing() -> Result<()> {
        let root = embedding_fixture()?;
        let mut pooling = PoolingConfig::last_token(1_024);
        pooling.pooling_mode_lasttoken = false;
        pooling.pooling_mode_mean_tokens = true;
        fs::write(
            root.path().join("1_Pooling/config.json"),
            serde_json::to_vec(&pooling)?,
        )?;
        assert!(Qwen3EmbeddingProfile::from_model_dir(root.path()).is_err());

        let valid = embedding_fixture()?;
        let profile = Qwen3EmbeddingProfile::from_model_dir(valid.path())?;
        assert!(
            profile
                .tokenize_document(task_tokenizer(false), "document")
                .is_err()
        );
        assert!(embedding_query(" ", None).is_err());
        assert!(validate_text("embedding document", "").is_err());

        let two_token_profile = Qwen3EmbeddingProfile {
            dimension: 1_024,
            vocab_size: 151_669,
            max_tokens: 2,
        };
        let tokenizer = task_tokenizer(true);
        assert!(
            two_token_profile
                .tokenize_document(tokenizer, "macllm")
                .is_ok()
        );
        let error = two_token_profile
            .tokenize_document(tokenizer, "macllm tokenizer")
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("automatic truncation is disabled")
        );

        let wrong_eos = embedding_fixture()?;
        write_base_config(wrong_eos.path(), 1_024, 32_768, QWEN3_IM_END_TOKEN_ID)?;
        assert!(Qwen3EmbeddingProfile::from_model_dir(wrong_eos.path()).is_err());
        Ok(())
    }

    #[test]
    fn profile_context_cap_never_exceeds_the_model_limit() -> Result<()> {
        let root = embedding_fixture()?;
        write_base_config(root.path(), 1_024, 4_096, QWEN3_END_OF_TEXT_TOKEN_ID)?;
        let profile = Qwen3EmbeddingProfile::from_model_dir(root.path())?;
        assert_eq!(profile.max_tokens(), 4_096);
        Ok(())
    }

    #[test]
    fn reranker_profile_validates_labels_and_formats_the_official_prompt() -> Result<()> {
        let root = reranker_fixture()?;
        let profile = Qwen3RerankerProfile::from_model_dir(root.path())?;

        assert_eq!(profile.no_token_id(), 2_152);
        assert_eq!(profile.yes_token_id(), 9_693);
        assert_eq!(profile.max_tokens(), QWEN3_TASK_MAX_TOKENS);
        let tokenizer = task_tokenizer(false);
        profile.validate_tokenizer(tokenizer)?;
        assert!(
            profile
                .tokenize_pair(task_tokenizer(true), "query", "document", None)
                .is_err()
        );
        assert_eq!(
            format!(
                "{RERANK_PREFIX}{}{RERANK_SUFFIX}",
                rerank_body("Capital?", "Beijing.", None)?
            ),
            concat!(
                "<|im_start|>system\n",
                "Judge whether the Document meets the requirements based on the Query and the Instruct provided. ",
                "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n",
                "<|im_start|>user\n",
                "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n",
                "<Query>: Capital?\n<Document>: Beijing.",
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        );
        let expected = [
            encode(tokenizer, RERANK_PREFIX, false)?,
            encode(
                tokenizer,
                &rerank_body("Capital?", "Beijing.", None)?,
                false,
            )?,
            encode(tokenizer, RERANK_SUFFIX, false)?,
        ]
        .concat();
        assert_eq!(
            profile.tokenize_pair(tokenizer, "Capital?", "Beijing.", None)?,
            expected
        );
        let exact = Qwen3RerankerProfile {
            false_token_id: 2_152,
            true_token_id: 9_693,
            vocab_size: 151_669,
            max_tokens: expected.len(),
        };
        assert!(
            exact
                .tokenize_pair(tokenizer, "Capital?", "Beijing.", None)
                .is_ok()
        );
        let too_small = Qwen3RerankerProfile {
            max_tokens: expected.len() - 1,
            ..exact
        };
        let error = too_small
            .tokenize_pair(tokenizer, "Capital?", "Beijing.", None)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("automatic truncation is disabled")
        );
        Ok(())
    }

    #[test]
    fn task_profiles_reject_swapped_labels_wrong_modules_and_cross_task_use() -> Result<()> {
        let root = reranker_fixture()?;
        fs::write(
            root.path().join("1_LogitScore/config.json"),
            r#"{"true_token_id":2152,"false_token_id":9693}"#,
        )?;
        let swapped = Qwen3RerankerProfile::from_model_dir(root.path())?;
        assert!(
            swapped
                .tokenize_pair(task_tokenizer(false), "query", "document", None)
                .is_err()
        );
        assert!(rerank_body("", "document", None).is_err());
        assert!(rerank_body("query", " ", None).is_err());
        assert!(rerank_body("query", "document", Some("")).is_err());

        let valid = reranker_fixture()?;
        fs::write(
            valid.path().join("modules.json"),
            serde_json::to_vec(&serde_json::json!([]))?,
        )?;
        assert!(Qwen3RerankerProfile::from_model_dir(valid.path()).is_err());

        let embedding = embedding_fixture()?;
        let reranker = reranker_fixture()?;
        assert!(Qwen3RerankerProfile::from_model_dir(embedding.path()).is_err());
        assert!(Qwen3EmbeddingProfile::from_model_dir(reranker.path()).is_err());
        Ok(())
    }
}
