//! Deterministic prompt preparation for the shared-prefix coding-agent demo.
//!
//! This module deliberately stops before model execution. It opens checkpoint
//! metadata only far enough to obtain the verified adapter and tokenizer,
//! builds a reproducible repository bundle, and finds a UTF-8 boundary whose
//! rendered prompts all resolve to the requested physical cache-page boundary.

use std::{
    collections::{BTreeSet, HashMap, hash_map::Entry},
    fs::{self, OpenOptions},
    io::Read,
    path::{Component, Path, PathBuf},
};

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::{
    chat::{ChatMessage, ChatRole, validate_chat_messages},
    models::ModelAdapter,
    weights::ModelFiles,
};

use super::{
    FileProvenance,
    provenance::{file_provenance, sha256_bytes, token_ids_sha256, verify_file},
};

/// On-disk JSON schema version for prepared prefix-demo evidence.
pub const PREFIX_DEMO_SCHEMA_VERSION: u32 = 1;
/// Default exact shared prefix used by the public coding-agent demo.
pub const DEFAULT_SHARED_PREFIX_TOKENS: usize = 16_384;
/// Default prefill/cache publication quantum used by the public demo.
pub const DEFAULT_PREFILL_QUANTUM: usize = 128;
/// KV page alignment used by the public demo and default serving runtime.
pub const DEFAULT_KV_CACHE_BLOCK_TOKENS: usize =
    crate::runtime::ModelLoadConfig::DEFAULT_KV_CACHE_BLOCK_TOKENS;
/// The public demo always submits eight coding-agent requests together.
pub const PREFIX_DEMO_TASK_COUNT: usize = 8;
/// Auditable transform applied before repository bytes enter a native template.
pub const REPOSITORY_PAYLOAD_ENCODING: &str = "template_control_escape_v1";

const SYSTEM_INSTRUCTION: &str = concat!(
    "/system_override/no_think\n",
    "You are MacLLM's local coding agent. Use only the repository bundle prefix below. ",
    "Cite concrete file paths, do not invent missing code, and keep the answer concise. ",
    "The deterministic bundle may end at a UTF-8 boundary inside the final file.\n\n",
    "REPOSITORY BUNDLE PREFIX\n",
);

const SEED_TASK: &str =
    "0: From Cargo.toml, state the package purpose, default features, and release tuning.";
const DEMO_TASKS: [&str; PREFIX_DEMO_TASK_COUNT] = [
    "A: In src/agent.rs, list the fail-closed request rules enforced by validate_agent_request.",
    "B: In src/agent.rs, trace how assistant tool-call history is rendered and validated.",
    "C: In src/architecture.rs, explain DecoderSpec::validate invariants for heads and layers.",
    "D: In src/architecture.rs, compare the represented full, sliding, and recurrent state semantics.",
    "E: In src/architecture.rs, explain PrefixCacheDiagnostics counters and physical owner meaning.",
    "F: In src/backend.rs, explain how device and dtype selection fail closed on unsupported builds.",
    "G: In src/benchmark/assemble.rs, explain how MacllmProvenanceGuard pins executable and source identity.",
    "H: In src/benchmark/assemble.rs, audit one capture-time check that rejects mixed provenance.",
];

const DEMO_REQUIRED_BUNDLE_FILES: [&str; 5] = [
    "Cargo.toml",
    "src/agent.rs",
    "src/architecture.rs",
    "src/backend.rs",
    "src/benchmark/assemble.rs",
];

/// Stable identity of the complete repository bundle before prefix selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryBundleEvidence {
    pub file_count: usize,
    pub byte_count: u64,
    pub sha256: String,
    pub files: Vec<String>,
}

/// Complete deterministic repository bundle plus its serializable identity.
///
/// The source text stays separate from [`PreparedPrefixDemo`] so persisted
/// evidence does not duplicate the repository in addition to its messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepositoryBundle {
    evidence: RepositoryBundleEvidence,
    text: String,
}

impl RepositoryBundle {
    #[must_use]
    pub fn evidence(&self) -> &RepositoryBundleEvidence {
        &self.evidence
    }

    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// One fully rendered and tokenized seed or measured task prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedPrefixPrompt {
    pub name: String,
    /// `None` identifies the cache-warming seed; measured tasks are one-based.
    pub task_index: Option<usize>,
    pub messages: Vec<ChatMessage>,
    pub rendered_prompt_byte_count: usize,
    pub rendered_prompt_sha256: String,
    pub prompt_token_count: usize,
    pub prompt_token_ids_sha256: String,
    /// Exact token LCP against the seed; absent only on the seed itself.
    pub lcp_with_seed_tokens: Option<usize>,
    pub cache_endpoint_tokens: usize,
    pub common_prefix_token_ids_sha256: String,
}

/// Serializable, fail-closed input and evidence for the execution-layer demo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedPrefixDemo {
    pub schema_version: u32,
    pub model_family: String,
    pub model_file: FileProvenance,
    pub repository_bundle: RepositoryBundleEvidence,
    pub repository_payload_encoding: String,
    pub encoded_repository_bundle_byte_count: u64,
    pub encoded_repository_bundle_sha256: String,
    /// Byte count inside the template-safe encoded repository payload.
    pub selected_bundle_byte_count: u64,
    /// Character count inside the template-safe encoded repository payload.
    pub selected_bundle_char_count: usize,
    /// SHA-256 of the selected template-safe encoded repository payload.
    pub selected_bundle_sha256: String,
    pub requested_shared_tokens: usize,
    pub prefill_quantum: usize,
    pub kv_cache_block_tokens: usize,
    pub common_prefix_token_ids_sha256: String,
    pub seed: PreparedPrefixPrompt,
    pub tasks: Vec<PreparedPrefixPrompt>,
}

impl PreparedPrefixDemo {
    /// Recheck every structural invariant available in persisted evidence.
    pub fn validate(&self) -> Result<()> {
        validate_prepared_header(self)?;
        validate_prepared_prompt(&self.seed, None, self)?;
        ensure!(
            self.tasks.len() == PREFIX_DEMO_TASK_COUNT,
            "prefix demo must contain exactly {PREFIX_DEMO_TASK_COUNT} measured tasks"
        );

        let seed_system = system_message(&self.seed.messages)?;
        let selected_bundle = seed_system
            .strip_prefix(SYSTEM_INSTRUCTION)
            .context("prefix-demo system message lost its repository bundle marker")?;
        validate_demo_task_grounding(selected_bundle)?;
        ensure!(
            self.seed.messages[1].content == SEED_TASK,
            "prefix-demo seed task text changed"
        );
        let mut first_characters = BTreeSet::new();
        first_characters.insert(user_first_character(&self.seed.messages)?);
        for (offset, task) in self.tasks.iter().enumerate() {
            let index = offset + 1;
            validate_prepared_prompt(task, Some(index), self)?;
            ensure!(
                system_message(&task.messages)? == seed_system,
                "task {index} system message differs from the seed"
            );
            ensure!(
                task.messages[1].content == DEMO_TASKS[offset],
                "task {index} text changed"
            );
            ensure!(
                first_characters.insert(user_first_character(&task.messages)?),
                "seed and task user messages must differ at their first character"
            );
        }
        Ok(())
    }
}

/// Search output retaining exact token IDs for immediate evidence assembly.
///
/// Token IDs are intentionally not embedded in [`PreparedPrefixDemo`]; their
/// canonical digests are persisted instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedPrefixSelection {
    pub bundle_byte_count: usize,
    pub prompt_token_ids: Vec<Vec<u32>>,
    pub lcp_with_seed_tokens: Vec<usize>,
    pub cache_endpoint_tokens: usize,
    pub cache_block_tokens: usize,
}

/// Reconstructed prompt bytes and IDs, verified against prepared evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedPrefixPrompt {
    pub name: String,
    pub rendered_prompt: String,
    pub token_ids: Vec<u32>,
}

/// The long reference prompt, an exact raw cache seed, and eight task prompts
/// reconstructed from live inputs.
///
/// `seed` remains the task-bearing prompt used to prove the selected LCP.
/// `cache_seed` is separately derived from the raw common rendered prefix and
/// retokenized to exactly the requested cache endpoint.  A server benchmark
/// must warm slots with the latter rather than with the longer reference task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedPrefixDemo {
    pub seed: MaterializedPrefixPrompt,
    pub cache_seed: MaterializedPrefixPrompt,
    pub tasks: Vec<MaterializedPrefixPrompt>,
}

#[derive(Debug)]
struct BundleFile {
    relative: String,
    absolute: PathBuf,
}

#[derive(Debug, Clone)]
struct PrefixMeasurement {
    prompt_token_ids: Vec<Vec<u32>>,
    lcp_with_seed_tokens: Vec<usize>,
    minimum_lcp: usize,
}

impl PrefixMeasurement {
    fn from_prompt_tokens(prompt_token_ids: Vec<Vec<u32>>) -> Result<Self> {
        ensure!(
            prompt_token_ids.len() >= 2,
            "prefix search requires one seed and at least one task prompt"
        );
        let seed = &prompt_token_ids[0];
        let lcp_with_seed_tokens = prompt_token_ids[1..]
            .iter()
            .map(|task| longest_common_prefix(seed, task))
            .collect::<Vec<_>>();
        let minimum_lcp = lcp_with_seed_tokens
            .iter()
            .copied()
            .min()
            .context("prefix search produced no task LCP measurements")?;
        Ok(Self {
            prompt_token_ids,
            lcp_with_seed_tokens,
            minimum_lcp,
        })
    }

    fn matches_exact_prefix(&self, requested_shared_tokens: usize) -> bool {
        self.lcp_with_seed_tokens
            .iter()
            .all(|&lcp| lcp == requested_shared_tokens)
    }
}

#[derive(Debug)]
struct RenderedPrompt {
    name: String,
    task_index: Option<usize>,
    messages: Vec<ChatMessage>,
    rendered: String,
    token_ids: Vec<u32>,
}

/// Bundle `Cargo.toml` and every `src/**/*.rs` file in raw-byte path order.
///
/// Symlinks, non-file/non-directory entries, non-UTF-8 paths, non-UTF-8 source
/// contents, and files that change while being read are rejected.
pub fn build_repository_bundle(repository_root: impl AsRef<Path>) -> Result<RepositoryBundle> {
    let root = repository_root.as_ref();
    validate_repository_root(root)?;

    let mut files = vec![regular_bundle_file(root, Path::new("Cargo.toml"))?];
    collect_rust_sources(root, Path::new("src"), &mut files)?;
    files.sort_unstable_by(|left, right| left.relative.as_bytes().cmp(right.relative.as_bytes()));
    ensure!(
        files.len() > 1,
        "repository bundle requires Cargo.toml and at least one src/**/*.rs file"
    );

    let mut text = String::new();
    let mut paths = Vec::with_capacity(files.len());
    for file in files {
        let contents = read_stable_utf8_file(&file)?;
        text.push_str("--- FILE: ");
        text.push_str(&file.relative);
        text.push_str(" ---\n");
        text.push_str(&contents);
        if !contents.ends_with('\n') {
            text.push('\n');
        }
        paths.push(file.relative);
    }

    let byte_count = u64::try_from(text.len()).context("repository bundle size exceeds u64")?;
    let evidence = RepositoryBundleEvidence {
        file_count: paths.len(),
        byte_count,
        sha256: sha256_bytes(text.as_bytes()),
        files: paths,
    };
    Ok(RepositoryBundle { evidence, text })
}

/// Escape native-template controls while leaving ordinary source readable.
///
/// The raw bundle and this transformed payload have separate hashes in the
/// prepared evidence. Escapes are literal ASCII `\\u....` markers; the model
/// sees them as repository data, never as chat-template delimiters or dynamic
/// `SmolLM3` mode directives.
#[must_use]
pub fn encode_repository_payload(bundle: &str) -> String {
    bundle
        .replace("<|", "\\u003C|")
        .replace("/no_think", "\\u002Fno_think")
        .replace("/think", "\\u002Fthink")
        .replace("/system_override", "\\u002Fsystem_override")
}

/// Find a UTF-8 bundle boundary whose seed/task LCPs land in one exact cache block.
///
/// The evaluator is deliberately protocol-free: synthetic renderers and
/// tokenizers can return any seed-first token-vector set. The primary search is
/// logarithmic. If tokenizer behavior is non-monotonic at that crossing, a
/// deterministic outward search eventually examines every UTF-8 boundary.
pub fn search_quantized_prefix_boundary<F>(
    bundle: &str,
    requested_shared_tokens: usize,
    cache_block_tokens: usize,
    mut evaluate: F,
) -> Result<QuantizedPrefixSelection>
where
    F: FnMut(&str) -> Result<Vec<Vec<u32>>>,
{
    ensure!(
        cache_block_tokens > 0,
        "prefix-search cache block size must be positive"
    );
    ensure!(
        requested_shared_tokens > 0,
        "requested shared prefix must be positive"
    );
    ensure!(
        requested_shared_tokens.is_multiple_of(cache_block_tokens),
        "requested shared prefix must be an exact multiple of the cache block size"
    );

    let boundaries = char_boundaries(bundle);
    let mut measurements = HashMap::new();
    let last_index = boundaries.len() - 1;
    let full = evaluate_boundary(
        bundle,
        &boundaries,
        last_index,
        &mut measurements,
        &mut evaluate,
    )?;
    if full.matches_exact_prefix(requested_shared_tokens) {
        return take_selection(
            &boundaries,
            last_index,
            requested_shared_tokens,
            cache_block_tokens,
            measurements,
        );
    }

    let crossing = if full.minimum_lcp >= requested_shared_tokens {
        let mut lower = 0;
        let mut upper = last_index;
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            let measurement = evaluate_boundary(
                bundle,
                &boundaries,
                middle,
                &mut measurements,
                &mut evaluate,
            )?;
            if measurement.matches_exact_prefix(requested_shared_tokens) {
                return take_selection(
                    &boundaries,
                    middle,
                    requested_shared_tokens,
                    cache_block_tokens,
                    measurements,
                );
            }
            if measurement.minimum_lcp < requested_shared_tokens {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        lower
    } else {
        // A lower full-bundle LCP does not prove insufficiency when a synthetic
        // or future tokenizer is non-monotonic. Start at the final boundary and
        // let the exhaustive fallback establish the truthful result.
        last_index
    };

    if let Some(index) = find_matching_progressively(
        bundle,
        &boundaries,
        crossing,
        requested_shared_tokens,
        &mut measurements,
        &mut evaluate,
    )? {
        return take_selection(
            &boundaries,
            index,
            requested_shared_tokens,
            cache_block_tokens,
            measurements,
        );
    }

    let maximum_lcp = measurements
        .values()
        .map(|measurement| measurement.minimum_lcp)
        .max()
        .unwrap_or(0);
    if maximum_lcp < requested_shared_tokens {
        bail!(
            "repository bundle is insufficient for a {requested_shared_tokens}-token shared prefix: exhaustive boundary search found a maximum minimum seed/task LCP of {maximum_lcp} tokens"
        );
    }
    bail!(
        "no UTF-8 bundle boundary produced exact cache endpoint {requested_shared_tokens} at block size {cache_block_tokens} after exhaustive search"
    )
}

/// Prepare the default 16K coding-agent prefix demo with 128-token prefill
/// publication quanta and 16-token KV pages.
pub fn prepare_default_prefix_demo(
    model_path: impl AsRef<Path>,
    repository_root: impl AsRef<Path>,
) -> Result<PreparedPrefixDemo> {
    prepare_prefix_demo(
        model_path,
        repository_root,
        DEFAULT_SHARED_PREFIX_TOKENS,
        DEFAULT_PREFILL_QUANTUM,
    )
}

/// Prepare exact chat messages and cryptographic evidence without loading model
/// weights or constructing a model runtime.
pub fn prepare_prefix_demo(
    model_path: impl AsRef<Path>,
    repository_root: impl AsRef<Path>,
    requested_shared_tokens: usize,
    prefill_quantum: usize,
) -> Result<PreparedPrefixDemo> {
    validate_task_first_characters()?;
    let bundle = build_repository_bundle(repository_root)?;
    let encoded_bundle = encode_repository_payload(bundle.text());

    // `ModelFiles` validates the checkpoint and exposes metadata-backed adapter
    // and tokenizer objects. This preparation layer never asks it for a weight
    // source and therefore never constructs or loads the decoder tensors.
    let files = ModelFiles::from_dir(model_path).context("failed to open prefix-demo model")?;
    ensure!(
        files.weight_file_count() == 1,
        "prefix demo requires one exact model weight file"
    );
    let model_file = file_provenance(files.weight_paths()[0])
        .context("failed to fingerprint prefix-demo model")?;
    let adapter = files
        .adapter()
        .context("failed to load prefix-demo model adapter")?;
    adapter.validate()?;
    ensure!(
        adapter.family() == "smollm3",
        "the deterministic coding-agent prefix demo currently requires SmolLM3, found {}",
        adapter.family()
    );
    let tokenizer = files
        .tokenizer()
        .context("failed to load prefix-demo tokenizer")?;

    let selection = search_quantized_prefix_boundary(
        &encoded_bundle,
        requested_shared_tokens,
        DEFAULT_KV_CACHE_BLOCK_TOKENS,
        |prefix| tokenize_demo_prompts(adapter.as_ref(), &tokenizer, prefix),
    )?;
    ensure!(
        selection.prompt_token_ids.len() == PREFIX_DEMO_TASK_COUNT + 1,
        "prefix preparation must produce one seed and {PREFIX_DEMO_TASK_COUNT} tasks"
    );

    let selected_bundle = &encoded_bundle[..selection.bundle_byte_count];
    validate_demo_task_grounding(selected_bundle)?;
    let rendered_prompts = render_demo_prompts(adapter.as_ref(), &tokenizer, selected_bundle)?;
    ensure!(
        rendered_prompts
            .iter()
            .map(|prompt| prompt.token_ids.as_slice())
            .eq(selection.prompt_token_ids.iter().map(Vec::as_slice)),
        "adapter or tokenizer produced non-deterministic prefix-demo prompts"
    );
    ensure!(
        rendered_prompts
            .iter()
            .all(|prompt| prompt.token_ids.len() < adapter.runtime_spec().max_context),
        "prepared prefix-demo prompt leaves no model context for generation"
    );

    assemble_prepared_demo(
        adapter.family(),
        model_file,
        bundle.evidence().clone(),
        &encoded_bundle,
        selected_bundle,
        requested_shared_tokens,
        prefill_quantum,
        DEFAULT_KV_CACHE_BLOCK_TOKENS,
        selection,
        rendered_prompts,
    )
}

/// Rebuild repository bytes, model identity, template rendering, and token IDs
/// from live inputs before execution consumes persisted preparation evidence.
pub fn materialize_prefix_demo(
    model_path: impl AsRef<Path>,
    repository_root: impl AsRef<Path>,
    prepared: &PreparedPrefixDemo,
) -> Result<MaterializedPrefixDemo> {
    prepared.validate()?;

    let bundle = build_repository_bundle(repository_root)?;
    ensure!(
        bundle.evidence() == &prepared.repository_bundle,
        "live repository bundle differs from prepared prefix-demo evidence"
    );
    let encoded_bundle = encode_repository_payload(bundle.text());
    ensure!(
        u64::try_from(encoded_bundle.len()).ok()
            == Some(prepared.encoded_repository_bundle_byte_count)
            && sha256_bytes(encoded_bundle.as_bytes()) == prepared.encoded_repository_bundle_sha256,
        "live encoded repository payload differs from prepared evidence"
    );
    let selected_bytes = usize::try_from(prepared.selected_bundle_byte_count)
        .context("selected bundle byte count exceeds usize")?;
    ensure!(
        selected_bytes <= encoded_bundle.len() && encoded_bundle.is_char_boundary(selected_bytes),
        "prepared selected payload boundary is not valid in the live encoded repository bundle"
    );
    let selected_bundle = &encoded_bundle[..selected_bytes];
    validate_demo_task_grounding(selected_bundle)?;
    ensure!(
        selected_bundle.chars().count() == prepared.selected_bundle_char_count
            && sha256_bytes(selected_bundle.as_bytes()) == prepared.selected_bundle_sha256,
        "live selected repository prefix differs from prepared evidence"
    );
    let expected_system = format!("{SYSTEM_INSTRUCTION}{selected_bundle}");
    ensure!(
        system_message(&prepared.seed.messages)? == expected_system,
        "prepared system message does not contain the selected live repository prefix"
    );

    let files = ModelFiles::from_dir(model_path).context("failed to reopen prefix-demo model")?;
    ensure!(
        files.weight_file_count() == 1,
        "prefix demo requires one exact model weight file"
    );
    let current_model = verify_file(
        files.weight_paths()[0],
        prepared.model_file.size_bytes,
        &prepared.model_file.sha256,
    )
    .context("live prefix-demo model differs from prepared evidence")?;
    ensure!(
        current_model == prepared.model_file,
        "live prefix-demo model path or filename differs from prepared evidence"
    );
    let adapter = files.adapter()?;
    adapter.validate()?;
    ensure!(
        adapter.family() == prepared.model_family,
        "live model adapter family differs from prepared evidence"
    );
    let tokenizer = files.tokenizer()?;

    let seed = materialize_prompt(adapter.as_ref(), &tokenizer, &prepared.seed)?;
    let tasks = prepared
        .tasks
        .iter()
        .map(|task| materialize_prompt(adapter.as_ref(), &tokenizer, task))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        tasks.len() == PREFIX_DEMO_TASK_COUNT,
        "materialized prefix demo lost one or more tasks"
    );
    let shared_tokens = seed
        .token_ids
        .get(..prepared.requested_shared_tokens)
        .context("materialized seed is shorter than its shared prefix")?;
    ensure!(
        token_ids_sha256(shared_tokens) == prepared.common_prefix_token_ids_sha256,
        "materialized seed common-prefix digest differs from prepared evidence"
    );
    for (task, evidence) in tasks.iter().zip(&prepared.tasks) {
        ensure!(
            longest_common_prefix(&seed.token_ids, &task.token_ids)
                == prepared.requested_shared_tokens,
            "materialized prompt {} does not preserve the exact shared token LCP",
            task.name
        );
        ensure!(
            task.token_ids.get(..prepared.requested_shared_tokens) == Some(shared_tokens),
            "materialized prompt {} changed inside the common prefix",
            task.name
        );
        ensure!(
            evidence.lcp_with_seed_tokens == Some(prepared.requested_shared_tokens),
            "prepared prompt {} has inconsistent LCP evidence",
            task.name
        );
    }
    let cache_seed = materialize_exact_cache_seed(
        &tokenizer,
        &seed,
        &tasks,
        prepared.requested_shared_tokens,
        &prepared.common_prefix_token_ids_sha256,
    )?;
    Ok(MaterializedPrefixDemo {
        seed,
        cache_seed,
        tasks,
    })
}

fn materialize_prompt(
    adapter: &dyn ModelAdapter,
    tokenizer: &Tokenizer,
    evidence: &PreparedPrefixPrompt,
) -> Result<MaterializedPrefixPrompt> {
    let rendered_prompt = adapter
        .format_chat_messages(&evidence.messages)
        .with_context(|| format!("failed to rerender prepared prompt {}", evidence.name))?;
    ensure!(
        rendered_prompt.len() == evidence.rendered_prompt_byte_count
            && sha256_bytes(rendered_prompt.as_bytes()) == evidence.rendered_prompt_sha256,
        "rerendered prompt {} differs from prepared byte evidence",
        evidence.name
    );
    let token_ids = tokenizer
        .encode(rendered_prompt.as_str(), false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| format!("failed to retokenize prepared prompt {}", evidence.name))?
        .get_ids()
        .to_vec();
    ensure!(
        token_ids.len() == evidence.prompt_token_count
            && token_ids_sha256(&token_ids) == evidence.prompt_token_ids_sha256,
        "retokenized prompt {} differs from prepared token evidence",
        evidence.name
    );
    Ok(MaterializedPrefixPrompt {
        name: evidence.name.clone(),
        rendered_prompt,
        token_ids,
    })
}

/// Derive the raw prompt accepted by completion endpoints for precisely the
/// shared token range.  A token-vector slice alone is insufficient because the
/// HTTP APIs receive text and may tokenize a truncated byte prefix differently.
fn materialize_exact_cache_seed(
    tokenizer: &Tokenizer,
    seed: &MaterializedPrefixPrompt,
    tasks: &[MaterializedPrefixPrompt],
    endpoint_tokens: usize,
    expected_token_ids_sha256: &str,
) -> Result<MaterializedPrefixPrompt> {
    ensure!(endpoint_tokens > 0, "cache endpoint must be positive");
    ensure!(
        !tasks.is_empty(),
        "cache-seed derivation requires at least one measured task"
    );
    let common_token_ids = seed
        .token_ids
        .get(..endpoint_tokens)
        .context("reference seed is shorter than the requested cache endpoint")?;
    ensure!(
        token_ids_sha256(common_token_ids) == expected_token_ids_sha256,
        "reference seed common-prefix digest differs from prepared evidence"
    );
    for task in tasks {
        ensure!(
            task.token_ids.get(..endpoint_tokens) == Some(common_token_ids),
            "task {} differs inside the requested cache endpoint",
            task.name
        );
    }

    let encoding = tokenizer
        .encode(seed.rendered_prompt.as_str(), false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to retokenize the reference prompt for cache-seed extraction")?;
    ensure!(
        encoding.get_ids() == seed.token_ids.as_slice(),
        "reference prompt token IDs changed before cache-seed extraction"
    );
    let shared_rendered_byte_count = common_rendered_prefix_byte_count(seed, tasks);
    let mut candidates = Vec::with_capacity(2);
    if let Some((_, token_end)) = encoding.get_offsets().get(endpoint_tokens - 1)
        && *token_end <= shared_rendered_byte_count
        && seed.rendered_prompt.is_char_boundary(*token_end)
    {
        candidates.push(seed.rendered_prompt[..*token_end].to_owned());
    }
    if let Ok(decoded) = tokenizer.decode(common_token_ids, false) {
        candidates.push(decoded);
    }

    for rendered_prompt in candidates {
        if rendered_prompt.is_empty()
            || !seed.rendered_prompt.starts_with(&rendered_prompt)
            || tasks
                .iter()
                .any(|task| !task.rendered_prompt.starts_with(&rendered_prompt))
        {
            continue;
        }
        let token_ids = tokenizer
            .encode(rendered_prompt.as_str(), false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to retokenize a candidate raw cache seed")?
            .get_ids()
            .to_vec();
        if token_ids == common_token_ids {
            return Ok(MaterializedPrefixPrompt {
                name: "cache_seed".to_owned(),
                rendered_prompt,
                token_ids,
            });
        }
    }
    bail!(
        "cannot derive a raw cache seed that retokenizes to the exact {endpoint_tokens}-token common prefix"
    )
}

fn common_rendered_prefix_byte_count(
    seed: &MaterializedPrefixPrompt,
    tasks: &[MaterializedPrefixPrompt],
) -> usize {
    let mut shared = seed.rendered_prompt.len();
    for task in tasks {
        shared = seed
            .rendered_prompt
            .as_bytes()
            .iter()
            .take(shared)
            .zip(task.rendered_prompt.as_bytes())
            .take_while(|(left, right)| left == right)
            .count();
    }
    while shared > 0 && !seed.rendered_prompt.is_char_boundary(shared) {
        shared -= 1;
    }
    shared
}

fn validate_prepared_header(prepared: &PreparedPrefixDemo) -> Result<()> {
    ensure!(
        prepared.schema_version == PREFIX_DEMO_SCHEMA_VERSION,
        "unsupported prefix-demo schema version {}",
        prepared.schema_version
    );
    ensure!(
        prepared.model_family == "smollm3",
        "prefix-demo model family must be smollm3"
    );
    ensure!(
        prepared.model_file.canonical_path.is_absolute()
            && !prepared.model_file.filename.is_empty()
            && prepared.model_file.size_bytes > 0
            && is_sha256(&prepared.model_file.sha256),
        "prefix-demo model provenance is invalid"
    );
    ensure!(
        prepared.prefill_quantum > 0,
        "prefix-demo prefill quantum must be positive"
    );
    ensure!(
        prepared.kv_cache_block_tokens > 0,
        "prefix-demo KV cache block size must be positive"
    );
    ensure!(
        prepared.requested_shared_tokens > 0
            && prepared
                .requested_shared_tokens
                .is_multiple_of(prepared.prefill_quantum),
        "prefix-demo shared tokens must be a positive multiple of its prefill quantum"
    );
    ensure!(
        prepared
            .requested_shared_tokens
            .is_multiple_of(prepared.kv_cache_block_tokens),
        "prefix-demo shared tokens must be a multiple of its KV cache block size"
    );
    ensure!(
        prepared.repository_payload_encoding == REPOSITORY_PAYLOAD_ENCODING,
        "unsupported prefix-demo repository payload encoding"
    );
    ensure!(
        prepared.encoded_repository_bundle_byte_count > 0
            && is_sha256(&prepared.encoded_repository_bundle_sha256),
        "encoded repository bundle evidence is invalid"
    );
    ensure!(
        prepared.selected_bundle_byte_count <= prepared.encoded_repository_bundle_byte_count,
        "selected payload prefix exceeds the encoded repository bundle"
    );
    ensure!(
        prepared.repository_bundle.file_count == prepared.repository_bundle.files.len(),
        "repository bundle file count disagrees with its file list"
    );
    ensure!(
        !prepared.repository_bundle.files.is_empty(),
        "repository bundle file list cannot be empty"
    );
    ensure!(
        prepared
            .repository_bundle
            .files
            .windows(2)
            .all(|paths| { paths[0].as_bytes() < paths[1].as_bytes() }),
        "repository bundle paths are not in strict raw-byte order"
    );
    for (name, digest) in [
        ("repository bundle", &prepared.repository_bundle.sha256),
        (
            "encoded repository bundle",
            &prepared.encoded_repository_bundle_sha256,
        ),
        ("selected bundle", &prepared.selected_bundle_sha256),
        (
            "common prefix token IDs",
            &prepared.common_prefix_token_ids_sha256,
        ),
    ] {
        ensure!(is_sha256(digest), "{name} SHA-256 is invalid");
    }
    Ok(())
}

fn validate_prepared_prompt(
    prompt: &PreparedPrefixPrompt,
    expected_task_index: Option<usize>,
    prepared: &PreparedPrefixDemo,
) -> Result<()> {
    ensure!(
        prompt.task_index == expected_task_index,
        "prepared prompt {} has the wrong task index",
        prompt.name
    );
    let expected_name =
        expected_task_index.map_or_else(|| "seed".to_owned(), |index| format!("task_{index}"));
    ensure!(
        prompt.name == expected_name,
        "prepared prompt name must be {expected_name}, found {}",
        prompt.name
    );
    validate_chat_messages(&prompt.messages)?;
    ensure!(
        prompt.messages.len() == 2
            && prompt.messages[0].role == ChatRole::System
            && prompt.messages[1].role == ChatRole::User,
        "prepared prompt {} must contain exactly system and user messages",
        prompt.name
    );
    let system = &prompt.messages[0].content;
    ensure!(
        system.contains("/system_override") && system.contains("/no_think"),
        "prepared prompt {} lacks deterministic SmolLM3 system controls",
        prompt.name
    );
    ensure!(
        prompt.rendered_prompt_byte_count > 0
            && prompt.prompt_token_count > prepared.requested_shared_tokens,
        "prepared prompt {} is too short for its declared shared prefix",
        prompt.name
    );
    ensure!(
        prompt.cache_endpoint_tokens == prepared.requested_shared_tokens,
        "prepared prompt {} has the wrong cache endpoint",
        prompt.name
    );
    ensure!(
        prompt.common_prefix_token_ids_sha256 == prepared.common_prefix_token_ids_sha256,
        "prepared prompt {} has a different common-prefix digest",
        prompt.name
    );
    ensure!(
        is_sha256(&prompt.rendered_prompt_sha256)
            && is_sha256(&prompt.prompt_token_ids_sha256)
            && is_sha256(&prompt.common_prefix_token_ids_sha256),
        "prepared prompt {} contains an invalid SHA-256 digest",
        prompt.name
    );
    match (expected_task_index, prompt.lcp_with_seed_tokens) {
        (None, None) => {}
        (Some(_), Some(lcp)) => {
            ensure!(
                lcp == prepared.requested_shared_tokens,
                "prepared prompt {} does not have the exact requested token LCP",
                prompt.name
            );
            ensure!(
                lcp < prompt.prompt_token_count && lcp < prepared.seed.prompt_token_count,
                "prepared prompt {} does not diverge from the seed",
                prompt.name
            );
        }
        _ => bail!(
            "prepared prompt {} has inconsistent seed/task LCP evidence",
            prompt.name
        ),
    }
    Ok(())
}

fn validate_repository_root(root: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(root)
        .with_context(|| format!("failed to inspect repository root {}", root.display()))?;
    ensure!(
        !metadata.file_type().is_symlink() && metadata.file_type().is_dir(),
        "repository root must be a real directory, not a symlink: {}",
        root.display()
    );
    Ok(())
}

fn collect_rust_sources(
    root: &Path,
    relative_directory: &Path,
    files: &mut Vec<BundleFile>,
) -> Result<()> {
    let directory = root.join(relative_directory);
    let metadata = fs::symlink_metadata(&directory)
        .with_context(|| format!("failed to inspect source directory {}", directory.display()))?;
    ensure!(
        !metadata.file_type().is_symlink() && metadata.file_type().is_dir(),
        "repository source path must be a real directory: {}",
        directory.display()
    );

    for entry in fs::read_dir(&directory)
        .with_context(|| format!("failed to read source directory {}", directory.display()))?
    {
        let entry = entry.with_context(|| {
            format!(
                "failed to enumerate source directory {}",
                directory.display()
            )
        })?;
        let name = entry.file_name();
        let name = name.to_str().with_context(|| {
            format!(
                "source path under {} is not valid UTF-8",
                directory.display()
            )
        })?;
        ensure!(
            !name.contains(['\n', '\r']),
            "source path contains a line break under {}",
            directory.display()
        );
        let relative = relative_directory.join(name);
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
        ensure!(
            !file_type.is_symlink(),
            "repository bundle rejects symlink {}",
            relative.display()
        );
        if file_type.is_dir() {
            collect_rust_sources(root, &relative, files)?;
        } else if file_type.is_file() {
            if relative
                .extension()
                .is_some_and(|extension| extension == "rs")
            {
                files.push(regular_bundle_file(root, &relative)?);
            }
        } else {
            bail!(
                "repository bundle rejects non-regular source entry {}",
                relative.display()
            );
        }
    }
    Ok(())
}

fn regular_bundle_file(root: &Path, relative: &Path) -> Result<BundleFile> {
    let relative_text = relative_path_text(relative)?;
    ensure!(
        !relative_text.contains(['\n', '\r']),
        "repository bundle path contains a line break: {relative_text:?}"
    );
    let absolute = root.join(relative);
    let metadata = fs::symlink_metadata(&absolute)
        .with_context(|| format!("failed to inspect repository file {}", absolute.display()))?;
    ensure!(
        !metadata.file_type().is_symlink() && metadata.file_type().is_file(),
        "repository bundle path is not a regular file: {}",
        relative.display()
    );
    Ok(BundleFile {
        relative: relative_text,
        absolute,
    })
}

/// Render one bundle-relative path with `/` separators on every host.
///
/// The recorded bundle bytes and their digest must not depend on the host,
/// so the native separator is never part of a bundle path.
fn relative_path_text(relative: &Path) -> Result<String> {
    let mut components = Vec::new();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            bail!(
                "repository bundle path is not a normalized relative path: {}",
                relative.display()
            );
        };
        components.push(
            component
                .to_str()
                .context("repository bundle path is not valid UTF-8")?,
        );
    }
    Ok(components.join("/"))
}

fn read_stable_utf8_file(file: &BundleFile) -> Result<String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW);
    let mut opened = options
        .open(&file.absolute)
        .with_context(|| format!("failed to open repository file {}", file.absolute.display()))?;
    let before = opened
        .metadata()
        .with_context(|| format!("failed to inspect open file {}", file.absolute.display()))?;
    ensure!(
        before.file_type().is_file(),
        "repository bundle path stopped being a regular file: {}",
        file.relative
    );
    let mut bytes = Vec::new();
    opened
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read repository file {}", file.absolute.display()))?;
    let after = opened
        .metadata()
        .with_context(|| format!("failed to recheck open file {}", file.absolute.display()))?;
    validate_stable_fd_metadata(&before, &after, bytes.len(), &file.relative)?;
    String::from_utf8(bytes)
        .with_context(|| format!("repository file is not valid UTF-8: {}", file.relative))
}

fn validate_stable_fd_metadata(
    before: &fs::Metadata,
    after: &fs::Metadata,
    bytes_read: usize,
    relative: &str,
) -> Result<()> {
    let before_modified = before
        .modified()
        .with_context(|| format!("failed to read initial modification time for {relative}"))?;
    let after_modified = after
        .modified()
        .with_context(|| format!("failed to read final modification time for {relative}"))?;
    ensure!(
        before.file_type().is_file()
            && after.file_type().is_file()
            && before.len() == after.len()
            && u64::try_from(bytes_read).ok() == Some(after.len())
            && before_modified == after_modified,
        "repository file changed while being bundled: {relative}"
    );
    #[cfg(unix)]
    ensure!(
        before.dev() == after.dev()
            && before.ino() == after.ino()
            && before.mtime() == after.mtime()
            && before.mtime_nsec() == after.mtime_nsec()
            && before.ctime() == after.ctime()
            && before.ctime_nsec() == after.ctime_nsec(),
        "repository file identity or timestamps changed while being bundled: {relative}"
    );
    Ok(())
}

fn char_boundaries(text: &str) -> Vec<usize> {
    let mut boundaries = text
        .char_indices()
        .map(|(offset, _)| offset)
        .collect::<Vec<_>>();
    if boundaries.last().copied() != Some(text.len()) {
        boundaries.push(text.len());
    }
    boundaries
}

fn evaluate_boundary<'a, F>(
    bundle: &str,
    boundaries: &[usize],
    index: usize,
    measurements: &'a mut HashMap<usize, PrefixMeasurement>,
    evaluate: &mut F,
) -> Result<&'a PrefixMeasurement>
where
    F: FnMut(&str) -> Result<Vec<Vec<u32>>>,
{
    match measurements.entry(index) {
        Entry::Occupied(entry) => Ok(entry.into_mut()),
        Entry::Vacant(entry) => {
            let boundary = boundaries[index];
            let prompt_tokens = evaluate(&bundle[..boundary])
                .with_context(|| format!("failed to evaluate bundle boundary {boundary}"))?;
            let measurement = PrefixMeasurement::from_prompt_tokens(prompt_tokens)?;
            Ok(entry.insert(measurement))
        }
    }
}

fn find_matching_progressively<F>(
    bundle: &str,
    boundaries: &[usize],
    center: usize,
    requested_shared_tokens: usize,
    measurements: &mut HashMap<usize, PrefixMeasurement>,
    evaluate: &mut F,
) -> Result<Option<usize>>
where
    F: FnMut(&str) -> Result<Vec<Vec<u32>>>,
{
    let furthest = center.max(boundaries.len() - 1 - center);
    for distance in 0..=furthest {
        if let Some(left) = center.checked_sub(distance)
            && evaluate_boundary(bundle, boundaries, left, measurements, evaluate)?
                .matches_exact_prefix(requested_shared_tokens)
        {
            return Ok(Some(left));
        }
        if distance > 0
            && let Some(right) = center.checked_add(distance)
            && right < boundaries.len()
            && evaluate_boundary(bundle, boundaries, right, measurements, evaluate)?
                .matches_exact_prefix(requested_shared_tokens)
        {
            return Ok(Some(right));
        }
    }
    Ok(None)
}

fn take_selection(
    boundaries: &[usize],
    index: usize,
    requested_shared_tokens: usize,
    cache_block_tokens: usize,
    mut measurements: HashMap<usize, PrefixMeasurement>,
) -> Result<QuantizedPrefixSelection> {
    let measurement = measurements
        .remove(&index)
        .context("selected prefix boundary has no token measurement")?;
    Ok(QuantizedPrefixSelection {
        bundle_byte_count: boundaries[index],
        prompt_token_ids: measurement.prompt_token_ids,
        lcp_with_seed_tokens: measurement.lcp_with_seed_tokens,
        cache_endpoint_tokens: requested_shared_tokens,
        cache_block_tokens,
    })
}

fn tokenize_demo_prompts(
    adapter: &dyn ModelAdapter,
    tokenizer: &Tokenizer,
    bundle_prefix: &str,
) -> Result<Vec<Vec<u32>>> {
    render_demo_prompts(adapter, tokenizer, bundle_prefix)
        .map(|prompts| prompts.into_iter().map(|prompt| prompt.token_ids).collect())
}

fn render_demo_prompts(
    adapter: &dyn ModelAdapter,
    tokenizer: &Tokenizer,
    bundle_prefix: &str,
) -> Result<Vec<RenderedPrompt>> {
    let system = format!("{SYSTEM_INSTRUCTION}{bundle_prefix}");
    let mut prompts = Vec::with_capacity(PREFIX_DEMO_TASK_COUNT + 1);
    prompts.push(render_demo_prompt(
        adapter,
        tokenizer,
        "seed".to_owned(),
        None,
        &system,
        SEED_TASK,
    )?);
    for (offset, task) in DEMO_TASKS.iter().enumerate() {
        let index = offset + 1;
        prompts.push(render_demo_prompt(
            adapter,
            tokenizer,
            format!("task_{index}"),
            Some(index),
            &system,
            task,
        )?);
    }
    Ok(prompts)
}

fn render_demo_prompt(
    adapter: &dyn ModelAdapter,
    tokenizer: &Tokenizer,
    name: String,
    task_index: Option<usize>,
    system: &str,
    task: &str,
) -> Result<RenderedPrompt> {
    let messages = vec![ChatMessage::system(system), ChatMessage::user(task)];
    let rendered = adapter
        .format_chat_messages(&messages)
        .with_context(|| format!("failed to render prefix-demo prompt {name}"))?;
    let encoding = tokenizer
        .encode(rendered.as_str(), false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| format!("failed to tokenize prefix-demo prompt {name}"))?;
    Ok(RenderedPrompt {
        name,
        task_index,
        messages,
        rendered,
        token_ids: encoding.get_ids().to_vec(),
    })
}

#[allow(clippy::too_many_arguments)]
fn assemble_prepared_demo(
    model_family: &str,
    model_file: FileProvenance,
    bundle: RepositoryBundleEvidence,
    encoded_bundle: &str,
    selected_bundle: &str,
    requested_shared_tokens: usize,
    prefill_quantum: usize,
    kv_cache_block_tokens: usize,
    selection: QuantizedPrefixSelection,
    rendered_prompts: Vec<RenderedPrompt>,
) -> Result<PreparedPrefixDemo> {
    ensure!(
        selection.cache_block_tokens == kv_cache_block_tokens,
        "prefix selection cache block size changed during evidence assembly"
    );
    let shared_tokens = &selection.prompt_token_ids[0][..requested_shared_tokens];
    let common_prefix_token_ids_sha256 = token_ids_sha256(shared_tokens);
    for prompt_tokens in &selection.prompt_token_ids {
        ensure!(
            prompt_tokens.get(..requested_shared_tokens) == Some(shared_tokens),
            "prepared prompt changed within the requested shared token prefix"
        );
        ensure!(
            token_ids_sha256(&prompt_tokens[..requested_shared_tokens])
                == common_prefix_token_ids_sha256,
            "prepared prompt has a different common-prefix token digest"
        );
    }

    let mut prompts = rendered_prompts.into_iter();
    let seed_rendered = prompts
        .next()
        .context("prefix-demo rendered prompt set has no seed")?;
    let seed = prepared_prompt(
        seed_rendered,
        None,
        requested_shared_tokens,
        &common_prefix_token_ids_sha256,
    );
    let tasks = prompts
        .zip(selection.lcp_with_seed_tokens.iter().copied())
        .map(|(prompt, lcp)| {
            prepared_prompt(
                prompt,
                Some(lcp),
                requested_shared_tokens,
                &common_prefix_token_ids_sha256,
            )
        })
        .collect::<Vec<_>>();
    ensure!(
        tasks.len() == PREFIX_DEMO_TASK_COUNT,
        "prefix-demo evidence assembly lost one or more tasks"
    );

    let prepared = PreparedPrefixDemo {
        schema_version: PREFIX_DEMO_SCHEMA_VERSION,
        model_family: model_family.to_owned(),
        model_file,
        repository_bundle: bundle,
        repository_payload_encoding: REPOSITORY_PAYLOAD_ENCODING.to_owned(),
        encoded_repository_bundle_byte_count: u64::try_from(encoded_bundle.len())
            .context("encoded repository bundle size exceeds u64")?,
        encoded_repository_bundle_sha256: sha256_bytes(encoded_bundle.as_bytes()),
        selected_bundle_byte_count: u64::try_from(selected_bundle.len())
            .context("selected bundle size exceeds u64")?,
        selected_bundle_char_count: selected_bundle.chars().count(),
        selected_bundle_sha256: sha256_bytes(selected_bundle.as_bytes()),
        requested_shared_tokens,
        prefill_quantum,
        kv_cache_block_tokens,
        common_prefix_token_ids_sha256,
        seed,
        tasks,
    };
    prepared.validate()?;
    Ok(prepared)
}

fn prepared_prompt(
    prompt: RenderedPrompt,
    lcp_with_seed_tokens: Option<usize>,
    cache_endpoint_tokens: usize,
    common_prefix_token_ids_sha256: &str,
) -> PreparedPrefixPrompt {
    PreparedPrefixPrompt {
        name: prompt.name,
        task_index: prompt.task_index,
        messages: prompt.messages,
        rendered_prompt_byte_count: prompt.rendered.len(),
        rendered_prompt_sha256: sha256_bytes(prompt.rendered.as_bytes()),
        prompt_token_count: prompt.token_ids.len(),
        prompt_token_ids_sha256: token_ids_sha256(&prompt.token_ids),
        lcp_with_seed_tokens,
        cache_endpoint_tokens,
        common_prefix_token_ids_sha256: common_prefix_token_ids_sha256.to_owned(),
    }
}

fn longest_common_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn validate_task_first_characters() -> Result<()> {
    let mut characters = BTreeSet::new();
    for task in std::iter::once(SEED_TASK).chain(DEMO_TASKS) {
        let first = task
            .chars()
            .next()
            .context("prefix-demo task cannot be empty")?;
        ensure!(
            characters.insert(first),
            "seed and all prefix-demo tasks must differ at their first character"
        );
    }
    Ok(())
}

fn validate_demo_task_grounding(selected_bundle: &str) -> Result<()> {
    for path in DEMO_REQUIRED_BUNDLE_FILES {
        let header = format!("--- FILE: {path} ---\n");
        ensure!(
            selected_bundle.contains(&header),
            "exact-16K repository prefix does not contain required task evidence file {path}"
        );
    }
    ensure!(
        selected_bundle.contains("pub struct MacllmProvenanceGuard"),
        "exact-16K repository prefix truncates before the provenance guard used by demo tasks"
    );
    Ok(())
}

fn system_message(messages: &[ChatMessage]) -> Result<&str> {
    messages
        .first()
        .filter(|message| message.role == ChatRole::System)
        .map(|message| message.content.as_str())
        .context("prepared prefix prompt has no system message")
}

fn user_first_character(messages: &[ChatMessage]) -> Result<char> {
    messages
        .last()
        .filter(|message| message.role == ChatRole::User)
        .context("prepared prefix prompt has no user message")?
        .content
        .chars()
        .next()
        .context("prepared prefix prompt user message is empty")
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use tokenizers::{
        Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };

    use super::*;

    fn write_minimal_repository(root: &Path) -> Result<()> {
        fs::create_dir(root.join("src"))?;
        fs::write(root.join("Cargo.toml"), "[package]\nname = \"demo\"\n")?;
        fs::write(root.join("src/z.rs"), "pub fn z() {}\n")?;
        fs::write(root.join("src/a.rs"), "pub fn a() {}")?;
        fs::create_dir(root.join("src/nested"))?;
        fs::write(root.join("src/nested/b.rs"), "pub fn b() {}\n")?;
        fs::write(root.join("src/ignored.md"), "ignored")?;
        Ok(())
    }

    fn synthetic_prompt_tokens(lcp: usize, task_count: usize) -> Vec<Vec<u32>> {
        let common = (0..lcp)
            .map(|token| u32::try_from(token).expect("synthetic token fits u32"))
            .collect::<Vec<_>>();
        let mut prompts = Vec::with_capacity(task_count + 1);
        let mut seed = common.clone();
        seed.push(1_000_000);
        prompts.push(seed);
        for task in 0..task_count {
            let mut tokens = common.clone();
            tokens.push(2_000_000 + u32::try_from(task).expect("synthetic task fits u32"));
            prompts.push(tokens);
        }
        prompts
    }

    #[test]
    fn repository_bundle_is_byte_sorted_and_exactly_formatted() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_repository(temp.path())?;
        let bundle = build_repository_bundle(temp.path())?;
        assert_eq!(
            bundle.evidence().files,
            ["Cargo.toml", "src/a.rs", "src/nested/b.rs", "src/z.rs"]
        );
        let expected = concat!(
            "--- FILE: Cargo.toml ---\n",
            "[package]\nname = \"demo\"\n",
            "--- FILE: src/a.rs ---\n",
            "pub fn a() {}\n",
            "--- FILE: src/nested/b.rs ---\n",
            "pub fn b() {}\n",
            "--- FILE: src/z.rs ---\n",
            "pub fn z() {}\n",
        );
        assert_eq!(bundle.text(), expected);
        assert_eq!(bundle.evidence().file_count, 4);
        assert_eq!(bundle.evidence().byte_count, expected.len() as u64);
        assert_eq!(bundle.evidence().sha256, sha256_bytes(expected.as_bytes()));
        Ok(())
    }

    #[test]
    fn repository_payload_escapes_native_template_controls() {
        let raw = "<|im_start|> /no_think /think /system_override ordinary/path";
        let encoded = encode_repository_payload(raw);
        for reserved in ["<|", "/no_think", "/think", "/system_override"] {
            assert!(!encoded.contains(reserved));
        }
        assert!(encoded.contains("\\u003C|im_start|>"));
        assert!(encoded.contains("ordinary/path"));
    }

    #[test]
    fn quantized_search_rejects_an_insufficient_bundle() {
        let error = search_quantized_prefix_boundary("abcdef", 128, 128, |_| {
            Ok(synthetic_prompt_tokens(127, 8))
        })
        .expect_err("127 shared tokens cannot satisfy a 128-token endpoint");
        assert!(error.to_string().contains("insufficient"));
    }

    #[test]
    fn quantized_search_progresses_beyond_the_old_fixed_neighborhood() -> Result<()> {
        let bundle = "x".repeat(2_048);
        let selection = search_quantized_prefix_boundary(&bundle, 128, 128, |prefix| {
            let lcp = match prefix.len() {
                0..=511 => 64,
                900 => 128,
                _ => 300,
            };
            Ok(synthetic_prompt_tokens(lcp, 8))
        })?;
        assert_eq!(selection.bundle_byte_count, 900);
        assert_eq!(selection.cache_endpoint_tokens, 128);
        assert_eq!(selection.lcp_with_seed_tokens, vec![128; 8]);
        Ok(())
    }

    #[test]
    fn quantized_search_finds_an_earlier_match_when_full_bundle_lcp_is_lower() -> Result<()> {
        let bundle = "x".repeat(400);
        let selection = search_quantized_prefix_boundary(&bundle, 128, 128, |prefix| {
            let lcp = if prefix.len() == 100 { 128 } else { 64 };
            Ok(synthetic_prompt_tokens(lcp, 8))
        })?;
        assert_eq!(selection.bundle_byte_count, 100);
        assert_eq!(selection.lcp_with_seed_tokens, vec![128; 8]);
        Ok(())
    }

    #[test]
    fn quantized_search_rejects_the_next_window_boundary() {
        let error = search_quantized_prefix_boundary("x", 128, 128, |_| {
            Ok(synthetic_prompt_tokens(256, 8))
        })
        .expect_err("256 tokens belong to the next cache quantum");
        assert!(error.to_string().contains("no UTF-8 bundle boundary"));
    }

    #[test]
    fn cache_search_requires_exact_lcp_and_physical_block_alignment() -> Result<()> {
        let selected = search_quantized_prefix_boundary("x", 128, 16, |_| {
            Ok(synthetic_prompt_tokens(128, 8))
        })?;
        assert_eq!(selected.cache_endpoint_tokens, 128);
        assert_eq!(selected.cache_block_tokens, 16);

        let error =
            search_quantized_prefix_boundary("x", 128, 16, |_| Ok(synthetic_prompt_tokens(129, 8)))
                .expect_err("the public demo requires an exact 128-token textual LCP");
        assert!(error.to_string().contains("no UTF-8 bundle boundary"));

        assert!(
            search_quantized_prefix_boundary("x", 130, 16, |_| {
                Ok(synthetic_prompt_tokens(130, 8))
            })
            .is_err()
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn repository_bundle_rejects_source_symlinks() -> Result<()> {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir()?;
        write_minimal_repository(temp.path())?;
        symlink("a.rs", temp.path().join("src/link.rs"))?;
        let error = build_repository_bundle(temp.path())
            .expect_err("source symlinks must be rejected")
            .to_string();
        assert!(error.contains("rejects symlink"));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn stable_reader_refuses_a_symlink_at_no_follow_open() -> Result<()> {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir()?;
        let target = temp.path().join("target.rs");
        let link = temp.path().join("link.rs");
        fs::write(&target, "fn target() {}\n")?;
        symlink(&target, &link)?;
        let file = BundleFile {
            relative: "src/link.rs".to_owned(),
            absolute: link,
        };
        let error = read_stable_utf8_file(&file)
            .expect_err("O_NOFOLLOW must reject the swapped-in symlink")
            .to_string();
        assert!(error.contains("failed to open repository file"));
        Ok(())
    }

    #[test]
    fn stable_fd_metadata_rejects_a_changed_open_file() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("changing.rs");
        fs::write(&path, "one")?;
        let opened = OpenOptions::new().read(true).open(&path)?;
        let before = opened.metadata()?;
        fs::write(&path, "a different length")?;
        let after = opened.metadata()?;
        let error = validate_stable_fd_metadata(&before, &after, 3, "src/changing.rs")
            .expect_err("metadata changes must fail closed")
            .to_string();
        assert!(error.contains("changed while being bundled"));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn repository_bundle_rejects_non_utf8_paths() {
        use std::{ffi::OsString, os::unix::ffi::OsStringExt};

        let invalid = OsString::from_vec(vec![b'x', 0xff, b'.', b'r', b's']);
        let error = relative_path_text(Path::new(&invalid))
            .expect_err("non-UTF-8 paths must be rejected")
            .to_string();
        assert!(error.contains("not valid UTF-8"));
    }

    #[test]
    fn task_user_messages_diverge_on_the_first_character() -> Result<()> {
        validate_task_first_characters()?;
        let first = std::iter::once(SEED_TASK)
            .chain(DEMO_TASKS)
            .map(|task| task.chars().next().expect("task is non-empty"))
            .collect::<BTreeSet<_>>();
        assert_eq!(first.len(), PREFIX_DEMO_TASK_COUNT + 1);
        Ok(())
    }

    #[test]
    fn coding_tasks_require_their_evidence_files_inside_the_selected_prefix() {
        use std::fmt::Write as _;

        let mut grounded = String::new();
        for path in DEMO_REQUIRED_BUNDLE_FILES {
            writeln!(grounded, "--- FILE: {path} ---").expect("writing to String cannot fail");
        }
        grounded.push_str("pub struct MacllmProvenanceGuard;\n");
        assert!(validate_demo_task_grounding(&grounded).is_ok());
        assert!(validate_demo_task_grounding("--- FILE: Cargo.toml ---\n").is_err());
    }

    #[test]
    fn cache_seed_is_a_raw_prefix_with_exact_common_token_ids() -> Result<()> {
        let vocab = [
            ("[UNK]".to_owned(), 0),
            ("shared".to_owned(), 1),
            ("context".to_owned(), 2),
            ("reference".to_owned(), 3),
            ("first".to_owned(), 4),
            ("second".to_owned(), 5),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_owned())
            .build()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));

        let prompt = |name: &str, rendered_prompt: &str| -> Result<MaterializedPrefixPrompt> {
            let token_ids = tokenizer
                .encode(rendered_prompt, false)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?
                .get_ids()
                .to_vec();
            Ok(MaterializedPrefixPrompt {
                name: name.to_owned(),
                rendered_prompt: rendered_prompt.to_owned(),
                token_ids,
            })
        };
        let seed = prompt("seed", "shared context reference")?;
        let tasks = vec![
            prompt("task_1", "shared context first")?,
            prompt("task_2", "shared context second")?,
        ];

        let cache_seed =
            materialize_exact_cache_seed(&tokenizer, &seed, &tasks, 2, &token_ids_sha256(&[1, 2]))?;
        assert_eq!(cache_seed.name, "cache_seed");
        assert_eq!(cache_seed.rendered_prompt, "shared context");
        assert_eq!(cache_seed.token_ids, vec![1, 2]);
        assert!(
            seed.rendered_prompt
                .starts_with(&cache_seed.rendered_prompt)
        );
        assert!(tasks.iter().all(|task| {
            task.rendered_prompt
                .starts_with(&cache_seed.rendered_prompt)
        }));
        Ok(())
    }

    #[test]
    fn provenance_guard_anchor_precedes_the_generic_file_guard_in_the_live_bundle() -> Result<()> {
        let bundle = build_repository_bundle(env!("CARGO_MANIFEST_DIR"))?;
        let assemble = bundle
            .text()
            .split_once("--- FILE: src/benchmark/assemble.rs ---\n")
            .context("live repository bundle omitted src/benchmark/assemble.rs")?
            .1;
        let provenance_guard = assemble
            .find("pub struct MacllmProvenanceGuard")
            .context("live repository bundle omitted MacllmProvenanceGuard")?;
        let generic_file_guard = assemble
            .find("pub struct FileProvenanceGuard")
            .context("live repository bundle omitted FileProvenanceGuard")?;
        assert!(
            provenance_guard < generic_file_guard,
            "the coding-agent prefix must reach MacllmProvenanceGuard before the large generic guard"
        );
        Ok(())
    }
}
