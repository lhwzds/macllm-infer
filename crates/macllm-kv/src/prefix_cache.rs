//! Request-prefix index for reusable model-state snapshots.
//!
//! This first version stores one snapshot at each inserted entry endpoint. It
//! deliberately does not materialize a snapshot after every hash block: doing
//! so would make a set of progressively longer prompts retain a quadratic
//! amount of snapshot state. Hash blocks are still useful as a deterministic
//! lookup accelerator, while the stored token IDs remain the authority.

use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    error::Error,
    fmt,
};

/// Version mixed into every prefix-cache hash block.
pub const PREFIX_CACHE_HASH_DOMAIN_VERSION: u32 = 1;

const HASH_DOMAIN: &[u8] = b"macllm-infer/token-prefix\0";
const EMPTY_CHAIN: [u8; 32] = [0; 32];

type PrefixDigest = [u8; 32];
type EntryId = u64;

/// Prefix-cache construction or insertion failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefixCacheError {
    /// A hash block must contain at least one token.
    ZeroBlockTokens,
    /// Hash metadata is encoded as little-endian `u32` values.
    BlockTokensTooLarge { block_tokens: usize },
    /// Empty snapshots cannot skip any model work and are not indexed.
    EmptyPrefix,
    /// The absolute fixed-block index cannot be represented by the hash format.
    PrefixTooLong { token_count: usize },
    /// A single snapshot cannot fit in the configured budget.
    CostExceedsBudget { cost: usize, budget: usize },
    /// The process-local entry identifier space was exhausted.
    EntryIdExhausted,
}

impl fmt::Display for PrefixCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBlockTokens => write!(formatter, "prefix-cache block size must be positive"),
            Self::BlockTokensTooLarge { block_tokens } => write!(
                formatter,
                "prefix-cache block size {block_tokens} exceeds the u32 hash format"
            ),
            Self::EmptyPrefix => write!(formatter, "prefix-cache entries must contain tokens"),
            Self::PrefixTooLong { token_count } => write!(
                formatter,
                "prefix-cache entry with {token_count} tokens exceeds the u32 block-index format"
            ),
            Self::CostExceedsBudget { cost, budget } => write!(
                formatter,
                "prefix-cache entry cost {cost} exceeds the configured budget {budget}"
            ),
            Self::EntryIdExhausted => write!(formatter, "prefix-cache entry identifiers exhausted"),
        }
    }
}

impl Error for PrefixCacheError {}

/// Cumulative activity plus the index's current resident footprint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefixCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub matched_tokens: u64,
    pub entries: usize,
    pub cost: usize,
    pub evictions: u64,
}

/// A longest-prefix match and its reusable snapshot.
#[derive(Debug)]
pub struct PrefixCacheHit<'a, T> {
    /// Number of leading prompt tokens that the caller may reuse.
    pub matched_tokens: usize,
    /// Exact token IDs that were verified after the hash lookup.
    pub tokens: &'a [u32],
    /// State captured at `snapshot_tokens`; it may enclose a shorter match and
    /// must be forked/truncated to `matched_tokens` before use in that case.
    pub snapshot: &'a T,
    /// Full token endpoint at which `snapshot` was captured.
    pub snapshot_tokens: usize,
    /// Budget cost charged by this resident snapshot.
    pub cost: usize,
}

/// A snapshot removed by [`TokenPrefixCache::evict_lru`].
#[derive(Debug)]
pub struct EvictedPrefix<T> {
    pub tokens: Vec<u32>,
    pub snapshot: T,
    pub cost: usize,
}

#[derive(Debug)]
struct Entry<T> {
    tokens: Vec<u32>,
    snapshot: T,
    cost: usize,
    digest: PrefixDigest,
    full_block_digests: Vec<PrefixDigest>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BlockEndpoint {
    entry_id: EntryId,
    token_count: usize,
}

/// Cost-bounded LRU index of exact token prefixes.
///
/// Hashes only select a small candidate bucket. Every hit is confirmed by an
/// exact token comparison beginning at absolute prompt position zero, so a
/// digest collision or a matching substring can never restore the wrong
/// snapshot.
#[derive(Debug)]
pub struct TokenPrefixCache<T> {
    block_tokens: usize,
    max_cost: usize,
    resident_cost: usize,
    next_id: EntryId,
    entries: HashMap<EntryId, Entry<T>>,
    by_digest: HashMap<PrefixDigest, Vec<EntryId>>,
    by_block_digest: HashMap<PrefixDigest, Vec<BlockEndpoint>>,
    lengths: BTreeMap<usize, usize>,
    block_lengths: BTreeMap<usize, usize>,
    lru: VecDeque<EntryId>,
    hits: u64,
    misses: u64,
    matched_tokens: u64,
    evictions: u64,
    #[cfg(test)]
    forced_digest: Option<PrefixDigest>,
}

impl<T> TokenPrefixCache<T> {
    /// Create an empty index.
    ///
    /// `block_tokens` is part of the hash domain and must fit in `u32`.
    /// `max_cost` is an application-defined unit, such as retained KV blocks
    /// or bytes. Zero-cost entries are allowed even when the budget is zero.
    pub fn new(block_tokens: usize, max_cost: usize) -> Result<Self, PrefixCacheError> {
        if block_tokens == 0 {
            return Err(PrefixCacheError::ZeroBlockTokens);
        }
        if u32::try_from(block_tokens).is_err() {
            return Err(PrefixCacheError::BlockTokensTooLarge { block_tokens });
        }

        Ok(Self {
            block_tokens,
            max_cost,
            resident_cost: 0,
            next_id: 0,
            entries: HashMap::new(),
            by_digest: HashMap::new(),
            by_block_digest: HashMap::new(),
            lengths: BTreeMap::new(),
            block_lengths: BTreeMap::new(),
            lru: VecDeque::new(),
            hits: 0,
            misses: 0,
            matched_tokens: 0,
            evictions: 0,
            #[cfg(test)]
            forced_digest: None,
        })
    }

    /// Insert or replace one endpoint snapshot.
    ///
    /// Inserting the same exact token prefix replaces and drops its old value;
    /// replacement is not counted as an eviction. A successful insertion is
    /// the most-recently-used entry. Budget eviction happens before insertion.
    pub fn insert(
        &mut self,
        tokens: Vec<u32>,
        snapshot: T,
        cost: usize,
    ) -> Result<(), PrefixCacheError> {
        if tokens.is_empty() {
            return Err(PrefixCacheError::EmptyPrefix);
        }
        if cost > self.max_cost {
            return Err(PrefixCacheError::CostExceedsBudget {
                cost,
                budget: self.max_cost,
            });
        }
        validate_token_span(tokens.len(), self.block_tokens)?;
        let digest = self.index_digest(hash_endpoint(&tokens, self.block_tokens));
        let full_block_digests: Vec<PrefixDigest> =
            hash_full_block_endpoints(&tokens, self.block_tokens)
                .into_iter()
                .map(|block_digest| self.index_digest(block_digest))
                .collect();
        let next_id = self
            .next_id
            .checked_add(1)
            .ok_or(PrefixCacheError::EntryIdExhausted)?;

        if let Some(existing_id) = self.exact_entry_id(digest, &tokens) {
            drop(self.remove_entry(existing_id, false));
        }

        let target_old_cost = self.max_cost - cost;
        while self.resident_cost > target_old_cost {
            let evicted = self.evict_lru();
            debug_assert!(evicted.is_some(), "resident cost must belong to an entry");
            if evicted.is_none() {
                break;
            }
        }

        let id = self.next_id;
        self.next_id = next_id;
        self.resident_cost += cost;
        self.by_digest.entry(digest).or_default().push(id);
        for (block_index, &block_digest) in full_block_digests.iter().enumerate() {
            let token_count = (block_index + 1) * self.block_tokens;
            self.by_block_digest
                .entry(block_digest)
                .or_default()
                .push(BlockEndpoint {
                    entry_id: id,
                    token_count,
                });
            *self.block_lengths.entry(token_count).or_default() += 1;
        }
        *self.lengths.entry(tokens.len()).or_default() += 1;
        self.lru.push_back(id);
        self.entries.insert(
            id,
            Entry {
                tokens,
                snapshot,
                cost,
                digest,
                full_block_digests,
            },
        );
        Ok(())
    }

    /// Find the longest exact leading prefix below both exclusive bounds.
    ///
    /// A candidate must be strictly shorter than `prompt.len()` and strictly
    /// shorter than `exclusive_max_tokens`. This intentionally leaves at least
    /// one prompt token for a forward pass that produces first-token logits.
    /// A hit becomes most recently used; every call records exactly one hit or
    /// miss.
    pub fn lookup(
        &mut self,
        prompt: &[u32],
        exclusive_max_tokens: usize,
    ) -> Option<PrefixCacheHit<'_, T>> {
        let exclusive_limit = prompt.len().min(exclusive_max_tokens);
        let candidate_lengths: Vec<usize> = self
            .lengths
            .range(..exclusive_limit)
            .map(|(&length, _)| length)
            .rev()
            .collect();

        let Some(&maximum_candidate) = candidate_lengths.first() else {
            self.record_miss();
            return None;
        };
        let hashes = PromptHashPlan::new(prompt, maximum_candidate, self.block_tokens);

        let mut matched = None;
        for length in candidate_lengths {
            let digest = self.index_digest(hashes.endpoint(length));
            let Some(ids) = self.by_digest.get(&digest) else {
                continue;
            };
            for &id in ids {
                let Some(entry) = self.entries.get(&id) else {
                    continue;
                };
                if entry.tokens.as_slice() == &prompt[..length] {
                    matched = Some(id);
                    break;
                }
            }
            if matched.is_some() {
                break;
            }
        }

        let Some(id) = matched else {
            self.record_miss();
            return None;
        };

        let matched_tokens = self.entries.get(&id)?.tokens.len();
        Some(self.finish_hit(id, matched_tokens))
    }

    /// Find the longest block-aligned common prefix inside a stored snapshot.
    ///
    /// Unlike [`Self::lookup`], this may return an entry whose snapshot endpoint
    /// is longer than the actual match. The caller must fork or truncate that
    /// snapshot to `hit.matched_tokens` before restoring it. Only complete hash
    /// blocks are eligible, and the match is still strictly below both the
    /// prompt length and `exclusive_max_tokens`.
    ///
    /// This keeps one snapshot per entry while allowing a long shared system
    /// prompt to be reused when later user-suffix tokens diverge.
    pub fn lookup_aligned_common_prefix(
        &mut self,
        prompt: &[u32],
        exclusive_max_tokens: usize,
    ) -> Option<PrefixCacheHit<'_, T>> {
        let exclusive_limit = prompt.len().min(exclusive_max_tokens);
        let candidate_lengths: Vec<usize> = self
            .block_lengths
            .range(..exclusive_limit)
            .map(|(&length, _)| length)
            .rev()
            .collect();

        let Some(&maximum_candidate) = candidate_lengths.first() else {
            self.record_miss();
            return None;
        };
        let hashes = PromptHashPlan::new(prompt, maximum_candidate, self.block_tokens);

        for matched_tokens in candidate_lengths {
            let digest = self.index_digest(hashes.endpoint(matched_tokens));
            let Some(endpoints) = self.by_block_digest.get(&digest) else {
                continue;
            };

            // Multiple enclosing snapshots may share this exact block prefix.
            // Prefer the most recently used one so selection is deterministic
            // and agrees with the cache's eviction policy.
            let matched_id = self.lru.iter().rev().copied().find(|id| {
                endpoints.iter().any(|endpoint| {
                    endpoint.entry_id == *id && endpoint.token_count == matched_tokens
                }) && self.entries.get(id).is_some_and(|entry| {
                    entry.tokens.len() >= matched_tokens
                        && entry.tokens[..matched_tokens] == prompt[..matched_tokens]
                })
            });
            if let Some(id) = matched_id {
                return Some(self.finish_hit(id, matched_tokens));
            }
        }

        self.record_miss();
        None
    }

    fn finish_hit(&mut self, id: EntryId, matched_tokens: usize) -> PrefixCacheHit<'_, T> {
        self.hits = self.hits.saturating_add(1);
        self.matched_tokens = self
            .matched_tokens
            .saturating_add(u64::try_from(matched_tokens).unwrap_or(u64::MAX));
        self.touch(id);

        let entry = self
            .entries
            .get(&id)
            .expect("touched prefix entry must remain resident");
        PrefixCacheHit {
            matched_tokens,
            tokens: &entry.tokens[..matched_tokens],
            snapshot: &entry.snapshot,
            snapshot_tokens: entry.tokens.len(),
            cost: entry.cost,
        }
    }

    /// Remove and return the least-recently-used resident snapshot.
    ///
    /// This is available to a KV allocator that needs to relieve pressure even
    /// when the configured accounting budget has not been reached.
    pub fn evict_lru(&mut self) -> Option<EvictedPrefix<T>> {
        let id = self.lru.front().copied()?;
        let entry = self.remove_entry(id, true)?;
        Some(EvictedPrefix {
            tokens: entry.tokens,
            snapshot: entry.snapshot,
            cost: entry.cost,
        })
    }

    /// Change the budget, evicting LRU entries until the cache fits.
    ///
    /// Returns the number of snapshots evicted by this operation.
    pub fn set_max_cost(&mut self, max_cost: usize) -> usize {
        self.max_cost = max_cost;
        let mut evicted_count = 0usize;
        while self.resident_cost > self.max_cost {
            let evicted = self.evict_lru();
            debug_assert!(evicted.is_some(), "resident cost must belong to an entry");
            match evicted {
                Some(_) => evicted_count = evicted_count.saturating_add(1),
                None => break,
            }
        }
        evicted_count
    }

    #[must_use]
    pub const fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    #[must_use]
    pub const fn max_cost(&self) -> usize {
        self.max_cost
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            hits: self.hits,
            misses: self.misses,
            matched_tokens: self.matched_tokens,
            entries: self.entries.len(),
            cost: self.resident_cost,
            evictions: self.evictions,
        }
    }

    fn exact_entry_id(&self, digest: PrefixDigest, tokens: &[u32]) -> Option<EntryId> {
        self.by_digest.get(&digest)?.iter().copied().find(|id| {
            self.entries
                .get(id)
                .is_some_and(|entry| entry.tokens == tokens)
        })
    }

    fn touch(&mut self, id: EntryId) {
        if let Some(position) = self.lru.iter().position(|candidate| *candidate == id) {
            self.lru.remove(position);
        }
        self.lru.push_back(id);
    }

    fn record_miss(&mut self) {
        self.misses = self.misses.saturating_add(1);
    }

    fn remove_entry(&mut self, id: EntryId, count_eviction: bool) -> Option<Entry<T>> {
        let entry = self.entries.remove(&id)?;
        self.resident_cost = self
            .resident_cost
            .checked_sub(entry.cost)
            .expect("resident prefix-cache cost must not underflow");

        let digest_bucket_empty = if let Some(ids) = self.by_digest.get_mut(&entry.digest) {
            ids.retain(|candidate| *candidate != id);
            ids.is_empty()
        } else {
            false
        };
        if digest_bucket_empty {
            self.by_digest.remove(&entry.digest);
        }

        for (block_index, &block_digest) in entry.full_block_digests.iter().enumerate() {
            let token_count = (block_index + 1) * self.block_tokens;
            let block_bucket_empty =
                if let Some(endpoints) = self.by_block_digest.get_mut(&block_digest) {
                    endpoints.retain(|endpoint| {
                        endpoint.entry_id != id || endpoint.token_count != token_count
                    });
                    endpoints.is_empty()
                } else {
                    false
                };
            if block_bucket_empty {
                self.by_block_digest.remove(&block_digest);
            }

            let block_length_empty = if let Some(count) = self.block_lengths.get_mut(&token_count) {
                *count = count
                    .checked_sub(1)
                    .expect("prefix-cache block-length count must not underflow");
                *count == 0
            } else {
                false
            };
            if block_length_empty {
                self.block_lengths.remove(&token_count);
            }
        }

        let length_bucket_empty = if let Some(count) = self.lengths.get_mut(&entry.tokens.len()) {
            *count = count
                .checked_sub(1)
                .expect("prefix-cache length count must not underflow");
            *count == 0
        } else {
            false
        };
        if length_bucket_empty {
            self.lengths.remove(&entry.tokens.len());
        }

        if let Some(position) = self.lru.iter().position(|candidate| *candidate == id) {
            self.lru.remove(position);
        }
        if count_eviction {
            self.evictions = self.evictions.saturating_add(1);
        }
        Some(entry)
    }

    // `self` is used by the collision-injection hook in test builds only.
    #[allow(clippy::unused_self)]
    fn index_digest(&self, digest: PrefixDigest) -> PrefixDigest {
        #[cfg(test)]
        if let Some(forced) = self.forced_digest {
            return forced;
        }
        digest
    }

    #[cfg(test)]
    fn force_collisions_for_tests(&mut self) {
        assert!(
            self.entries.is_empty(),
            "collision mode requires an empty cache"
        );
        self.forced_digest = Some([0xa5; 32]);
    }
}

fn validate_token_span(token_count: usize, block_tokens: usize) -> Result<(), PrefixCacheError> {
    let last_block = (token_count - 1) / block_tokens;
    u32::try_from(last_block)
        .map(|_| ())
        .map_err(|_| PrefixCacheError::PrefixTooLong { token_count })
}

/// Hash format, where every integer is little-endian `u32`:
///
/// `SHA256(domain || version || block_width || absolute_block_index ||
/// valid_length || previous_digest || token_ids...)`.
///
/// The first block's previous digest is 32 zero bytes. A partial endpoint is a
/// real final block with its own valid length; it is not padded.
fn hash_block(
    previous: PrefixDigest,
    block_tokens: usize,
    absolute_block_index: usize,
    tokens: &[u32],
) -> PrefixDigest {
    let block_tokens = u32::try_from(block_tokens).expect("validated prefix-cache block width");
    let absolute_block_index =
        u32::try_from(absolute_block_index).expect("validated absolute prefix-cache block index");
    let valid_length =
        u32::try_from(tokens.len()).expect("valid block length cannot exceed block width");

    let mut hasher = Sha256::new();
    hasher.update(HASH_DOMAIN);
    hasher.update(PREFIX_CACHE_HASH_DOMAIN_VERSION.to_le_bytes());
    hasher.update(block_tokens.to_le_bytes());
    hasher.update(absolute_block_index.to_le_bytes());
    hasher.update(valid_length.to_le_bytes());
    hasher.update(previous);
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hasher.finalize().into()
}

fn hash_endpoint(tokens: &[u32], block_tokens: usize) -> PrefixDigest {
    let mut chain = EMPTY_CHAIN;
    for (block_index, block) in tokens.chunks(block_tokens).enumerate() {
        chain = hash_block(chain, block_tokens, block_index, block);
    }
    chain
}

fn hash_full_block_endpoints(tokens: &[u32], block_tokens: usize) -> Vec<PrefixDigest> {
    let full_blocks = tokens.len() / block_tokens;
    let mut endpoints = Vec::with_capacity(full_blocks);
    let mut chain = EMPTY_CHAIN;
    for block_index in 0..full_blocks {
        let start = block_index * block_tokens;
        let end = start + block_tokens;
        chain = hash_block(chain, block_tokens, block_index, &tokens[start..end]);
        endpoints.push(chain);
    }
    endpoints
}

struct PromptHashPlan<'a> {
    prompt: &'a [u32],
    block_tokens: usize,
    full_chains: Vec<PrefixDigest>,
}

impl<'a> PromptHashPlan<'a> {
    fn new(prompt: &'a [u32], maximum_endpoint: usize, block_tokens: usize) -> Self {
        let full_blocks = maximum_endpoint / block_tokens;
        let mut full_chains = Vec::with_capacity(full_blocks + 1);
        full_chains.push(EMPTY_CHAIN);
        for block_index in 0..full_blocks {
            let start = block_index * block_tokens;
            let end = start + block_tokens;
            let next = hash_block(
                *full_chains.last().expect("genesis chain is present"),
                block_tokens,
                block_index,
                &prompt[start..end],
            );
            full_chains.push(next);
        }
        Self {
            prompt,
            block_tokens,
            full_chains,
        }
    }

    fn endpoint(&self, token_count: usize) -> PrefixDigest {
        let full_blocks = token_count / self.block_tokens;
        let remainder = token_count % self.block_tokens;
        let chain = self.full_chains[full_blocks];
        if remainder == 0 {
            return chain;
        }

        let start = full_blocks * self.block_tokens;
        hash_block(
            chain,
            self.block_tokens,
            full_blocks,
            &self.prompt[start..start + remainder],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    fn cache(budget: usize) -> TokenPrefixCache<&'static str> {
        TokenPrefixCache::new(4, budget).expect("valid test cache")
    }

    fn lookup_value<'a>(
        cache: &'a mut TokenPrefixCache<&'static str>,
        prompt: &[u32],
        limit: usize,
    ) -> Option<(usize, &'a str)> {
        cache
            .lookup(prompt, limit)
            .map(|hit| (hit.matched_tokens, *hit.snapshot))
    }

    fn lookup_aligned_value<'a>(
        cache: &'a mut TokenPrefixCache<&'static str>,
        prompt: &[u32],
        limit: usize,
    ) -> Option<(usize, usize, &'a str)> {
        cache
            .lookup_aligned_common_prefix(prompt, limit)
            .map(|hit| (hit.matched_tokens, hit.snapshot_tokens, *hit.snapshot))
    }

    #[test]
    fn validates_configuration_and_insertions() {
        assert_eq!(
            TokenPrefixCache::<()>::new(0, 10).unwrap_err(),
            PrefixCacheError::ZeroBlockTokens
        );

        let mut cache = TokenPrefixCache::new(4, 3).unwrap();
        assert_eq!(
            cache.insert(Vec::new(), (), 0),
            Err(PrefixCacheError::EmptyPrefix)
        );
        assert_eq!(
            cache.insert(vec![1], (), 4),
            Err(PrefixCacheError::CostExceedsBudget { cost: 4, budget: 3 })
        );
        assert!(cache.is_empty());
    }

    #[test]
    fn validates_u32_hash_metadata_bounds_without_allocating_large_prefixes() {
        let oversized_block = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
        assert_eq!(
            TokenPrefixCache::<()>::new(oversized_block, 0).unwrap_err(),
            PrefixCacheError::BlockTokensTooLarge {
                block_tokens: oversized_block,
            }
        );

        let oversized_span = usize::try_from(u64::from(u32::MAX) + 2).unwrap();
        assert_eq!(
            validate_token_span(oversized_span, 1),
            Err(PrefixCacheError::PrefixTooLong {
                token_count: oversized_span,
            })
        );
    }

    #[test]
    fn returns_longest_exact_prefix_from_absolute_zero() {
        let mut cache = cache(10);
        cache.insert(vec![1, 2], "two", 1).unwrap();
        cache.insert(vec![1, 2, 3], "three", 1).unwrap();
        cache.insert(vec![1, 2, 3, 4], "four", 1).unwrap();

        assert_eq!(
            lookup_value(&mut cache, &[1, 2, 3, 4, 9], 5),
            Some((4, "four"))
        );
        assert_eq!(
            lookup_value(&mut cache, &[0, 1, 2, 3, 4, 9], 6),
            None,
            "a matching substring must not be treated as a prefix"
        );
    }

    #[test]
    fn excludes_the_prompt_and_maximum_limit_endpoints() {
        let mut cache = cache(10);
        cache.insert(vec![1, 2, 3], "three", 1).unwrap();
        cache.insert(vec![1, 2, 3, 4], "four", 1).unwrap();

        assert_eq!(
            lookup_value(&mut cache, &[1, 2, 3, 4], usize::MAX),
            Some((3, "three")),
            "the full prompt endpoint must remain uncached"
        );
        assert_eq!(
            lookup_value(&mut cache, &[1, 2, 3, 4, 5], 4),
            Some((3, "three")),
            "the supplied maximum is also exclusive"
        );
        assert_eq!(lookup_value(&mut cache, &[1, 2, 3], 3), None);
    }

    #[test]
    fn backs_off_from_a_hash_candidate_that_fails_exact_verification() {
        let mut cache = cache(10);
        cache.insert(vec![7, 8], "short", 1).unwrap();
        cache.insert(vec![7, 8, 9, 10], "long", 1).unwrap();

        assert_eq!(
            lookup_value(&mut cache, &[7, 8, 99, 10, 11], 5),
            Some((2, "short"))
        );
    }

    #[test]
    fn aligned_lookup_reuses_a_long_snapshot_before_a_divergent_suffix() {
        let mut cache = cache(20);
        cache
            .insert(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "enclosing", 7)
            .unwrap();

        let hit = cache
            .lookup_aligned_common_prefix(&[1, 2, 3, 4, 5, 6, 99, 100], 8)
            .unwrap();
        assert_eq!(hit.matched_tokens, 4);
        assert_eq!(hit.tokens, &[1, 2, 3, 4]);
        assert_eq!(hit.snapshot_tokens, 12);
        assert_eq!(*hit.snapshot, "enclosing");
        assert_eq!(hit.cost, 7);

        assert_eq!(
            lookup_aligned_value(&mut cache, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 77, 78], 11,),
            Some((8, 12, "enclosing"))
        );
    }

    #[test]
    fn aligned_lookup_handles_shorter_prompts_and_strict_bounds() {
        let mut cache = cache(20);
        cache
            .insert(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "long", 1)
            .unwrap();

        assert_eq!(
            lookup_aligned_value(&mut cache, &[1, 2, 3, 4, 5, 6], 6),
            Some((4, 12, "long")),
            "a shorter prompt may fork an enclosing snapshot"
        );
        assert_eq!(
            lookup_aligned_value(&mut cache, &[1, 2, 3, 4, 5, 6], 4),
            None,
            "the maximum-token bound is exclusive"
        );
        assert_eq!(
            lookup_aligned_value(&mut cache, &[1, 2, 3, 4], usize::MAX),
            None,
            "an aligned endpoint equal to the prompt is not reusable"
        );
        assert_eq!(
            lookup_aligned_value(&mut cache, &[0, 1, 2, 3, 4, 5, 6], 7),
            None,
            "a cached sequence appearing only as a substring is not reusable"
        );
    }

    #[test]
    fn aligned_lookup_exactly_verifies_injected_block_hash_collisions() {
        let mut cache = cache(20);
        cache.force_collisions_for_tests();
        cache
            .insert(vec![1, 2, 3, 4, 5, 6, 7, 8, 10], "first", 1)
            .unwrap();
        cache
            .insert(vec![9, 8, 7, 6, 5, 4, 3, 2, 10], "second", 1)
            .unwrap();

        assert_eq!(
            lookup_aligned_value(&mut cache, &[1, 2, 3, 4, 5, 6, 7, 8, 99], 9),
            Some((8, 9, "first"))
        );
        assert_eq!(
            lookup_aligned_value(&mut cache, &[9, 8, 7, 6, 5, 4, 3, 2, 99], 9),
            Some((8, 9, "second"))
        );
        assert_eq!(
            lookup_aligned_value(&mut cache, &[4, 4, 4, 4, 4, 4, 4, 4, 0], 9),
            None
        );
    }

    #[test]
    fn exact_tokens_defeat_injected_digest_collisions() {
        let mut cache = cache(10);
        cache.force_collisions_for_tests();
        cache.insert(vec![1, 2, 3], "first", 1).unwrap();
        cache.insert(vec![9, 8, 7], "second", 1).unwrap();

        assert_eq!(
            lookup_value(&mut cache, &[1, 2, 3, 4], 4),
            Some((3, "first"))
        );
        assert_eq!(
            lookup_value(&mut cache, &[9, 8, 7, 6], 4),
            Some((3, "second"))
        );
        assert_eq!(lookup_value(&mut cache, &[5, 5, 5, 0], 4), None);
    }

    #[test]
    fn replacement_is_exact_even_inside_a_collision_bucket() {
        #[derive(Debug)]
        struct DropProbe(Rc<Cell<usize>>, &'static str);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let drops = Rc::new(Cell::new(0));

        let mut cache = TokenPrefixCache::new(4, 10).unwrap();
        cache.force_collisions_for_tests();
        cache
            .insert(vec![1, 2], DropProbe(Rc::clone(&drops), "old"), 2)
            .unwrap();
        cache
            .insert(vec![8, 9], DropProbe(Rc::clone(&drops), "other"), 2)
            .unwrap();
        cache
            .insert(vec![1, 2], DropProbe(Rc::clone(&drops), "new"), 3)
            .unwrap();

        assert_eq!(drops.get(), 1, "the replaced value must be dropped");
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.stats().cost, 5);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.lookup(&[1, 2, 3], 3).unwrap().snapshot.1, "new");
        assert_eq!(cache.lookup(&[8, 9, 0], 3).unwrap().snapshot.1, "other");
    }

    #[test]
    fn hit_refreshes_lru_and_explicit_eviction_returns_the_oldest() {
        let mut cache = cache(10);
        cache.insert(vec![1], "one", 1).unwrap();
        cache.insert(vec![2], "two", 2).unwrap();
        cache.insert(vec![3], "three", 3).unwrap();

        assert_eq!(lookup_value(&mut cache, &[1, 9], 2), Some((1, "one")));
        let evicted = cache.evict_lru().unwrap();
        assert_eq!(evicted.tokens, vec![2]);
        assert_eq!(evicted.snapshot, "two");
        assert_eq!(evicted.cost, 2);
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.stats().cost, 4);
    }

    #[test]
    fn insertions_evict_lru_until_the_budget_fits() {
        let mut cache = cache(5);
        cache.insert(vec![1], "one", 2).unwrap();
        cache.insert(vec![2], "two", 2).unwrap();
        assert_eq!(lookup_value(&mut cache, &[1, 0], 2), Some((1, "one")));

        cache.insert(vec![3], "three", 3).unwrap();
        assert_eq!(lookup_value(&mut cache, &[2, 0], 2), None);
        assert_eq!(lookup_value(&mut cache, &[1, 0], 2), Some((1, "one")));
        assert_eq!(lookup_value(&mut cache, &[3, 0], 2), Some((1, "three")));
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.stats().cost, 5);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn aligned_block_indexes_follow_replacement_and_lru_eviction() {
        let mut cache = cache(2);
        cache.insert(vec![1, 1, 1, 1, 1], "first-old", 1).unwrap();
        cache.insert(vec![2, 2, 2, 2, 2], "second", 1).unwrap();
        assert_eq!(
            lookup_aligned_value(&mut cache, &[1, 1, 1, 1, 9], 5),
            Some((4, 5, "first-old"))
        );

        cache.insert(vec![3, 3, 3, 3, 3], "third", 1).unwrap();
        assert_eq!(
            lookup_aligned_value(&mut cache, &[2, 2, 2, 2, 9], 5),
            None,
            "budget eviction must remove the old block index"
        );

        cache.insert(vec![1, 1, 1, 1, 1], "first-new", 1).unwrap();
        assert_eq!(
            lookup_aligned_value(&mut cache, &[1, 1, 1, 1, 9], 5),
            Some((4, 5, "first-new")),
            "replacement must point block matches at the new snapshot"
        );

        assert_eq!(cache.evict_lru().unwrap().snapshot, "third");
        assert_eq!(lookup_aligned_value(&mut cache, &[3, 3, 3, 3, 9], 5), None);
    }

    #[test]
    fn shrinking_budget_and_zero_cost_entries_are_supported() {
        let mut cache = cache(10);
        cache.insert(vec![1], "one", 4).unwrap();
        cache.insert(vec![2], "two", 3).unwrap();
        cache.insert(vec![3], "free", 0).unwrap();

        assert_eq!(cache.set_max_cost(3), 1);
        assert_eq!(cache.stats().cost, 3);
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.set_max_cost(0), 1);
        assert_eq!(cache.stats().cost, 0);
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(lookup_value(&mut cache, &[3, 0], 2), Some((1, "free")));
    }

    #[test]
    fn stats_count_each_lookup_once_and_accumulate_matched_tokens() {
        let mut cache = cache(10);
        cache.insert(vec![1, 2], "two", 1).unwrap();
        cache.insert(vec![1, 2, 3], "three", 1).unwrap();

        assert_eq!(
            lookup_value(&mut cache, &[1, 2, 3, 4], 4),
            Some((3, "three"))
        );
        assert_eq!(lookup_value(&mut cache, &[1, 2, 9], 3), Some((2, "two")));
        assert_eq!(lookup_value(&mut cache, &[8, 8, 8], 3), None);
        assert_eq!(
            cache.stats(),
            PrefixCacheStats {
                hits: 2,
                misses: 1,
                matched_tokens: 5,
                entries: 2,
                cost: 2,
                evictions: 0,
            }
        );
    }

    #[test]
    fn endpoint_hash_is_independent_of_lookup_fragmentation() {
        let tokens = [10, 20, 30, 40, 50, 60, 70];
        let plan = PromptHashPlan::new(&tokens, tokens.len(), 4);
        for length in 1..=tokens.len() {
            assert_eq!(plan.endpoint(length), hash_endpoint(&tokens[..length], 4));
        }
    }

    #[test]
    fn hash_has_a_stable_domain_separated_golden_value() {
        assert_eq!(
            hash_endpoint(&[0x0102_0304, 5, 6, 7, 8], 4),
            [
                234, 223, 154, 217, 54, 25, 155, 93, 168, 35, 250, 114, 218, 228, 3, 17, 159, 115,
                165, 90, 62, 133, 54, 226, 28, 250, 17, 165, 115, 7, 250, 146,
            ]
        );
    }

    #[test]
    fn empty_and_tiny_bounds_are_misses_without_panics() {
        let mut cache = cache(10);
        cache.insert(vec![1], "one", 1).unwrap();

        assert_eq!(lookup_value(&mut cache, &[], 100), None);
        assert_eq!(lookup_value(&mut cache, &[1, 2], 0), None);
        assert_eq!(lookup_value(&mut cache, &[1, 2], 1), None);
        assert_eq!(cache.stats().misses, 3);
    }
}
