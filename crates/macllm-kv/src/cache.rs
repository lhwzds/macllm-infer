use std::{
    cell::{Cell, RefCell},
    fmt,
    num::NonZeroUsize,
    rc::Rc,
};

use candle_core::{DType, Device, Result, Tensor};

const MIN_ALLOCATION_TOKENS: usize = 16;

#[cfg(any(test, feature = "test-exports"))]
thread_local! {
    static PAGED_BATCH_MATERIALIZATIONS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(any(test, feature = "test-exports"))]
#[doc(hidden)]
pub fn reset_paged_batch_materialization_count() {
    PAGED_BATCH_MATERIALIZATIONS.set(0);
}

#[cfg(any(test, feature = "test-exports"))]
#[must_use]
#[doc(hidden)]
pub fn paged_batch_materialization_count() -> usize {
    PAGED_BATCH_MATERIALIZATIONS.get()
}

/// Result type used by the model-shared paged K/V allocator.
pub type PagedKvResult<T> = std::result::Result<T, PagedKvPoolError>;

/// Structured failures raised before or while operating the paged K/V pool.
#[derive(Debug)]
pub enum PagedKvPoolError {
    InvalidConfig(String),
    InvalidGeometry(String),
    /// The configured logical block budget cannot satisfy a reservation.
    Exhausted {
        requested_blocks: usize,
        free_blocks: usize,
        unallocated_blocks: usize,
    },
    ContextLimit {
        limit: usize,
        requested: usize,
    },
    PoolBusy,
    InvalidEpoch {
        submitted: u64,
        completed: u64,
        requested: u64,
    },
    InvalidTransaction(String),
    StaleBlockId {
        slot: u32,
        expected_generation: u64,
        actual_generation: u64,
    },
    HostAllocation(&'static str),
    Tensor(candle_core::Error),
}

impl fmt::Display for PagedKvPoolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid paged KV config: {message}"),
            Self::InvalidGeometry(message) => {
                write!(formatter, "invalid paged KV geometry: {message}")
            }
            Self::Exhausted {
                requested_blocks,
                free_blocks,
                unallocated_blocks,
            } => write!(
                formatter,
                "paged KV pool exhausted: requested {requested_blocks} blocks, {free_blocks} free and {unallocated_blocks} unallocated"
            ),
            Self::ContextLimit { limit, requested } => write!(
                formatter,
                "paged KV table limit {limit} exceeded by requested length {requested}"
            ),
            Self::PoolBusy => write!(formatter, "paged KV pool is already mutably borrowed"),
            Self::InvalidEpoch {
                submitted,
                completed,
                requested,
            } => write!(
                formatter,
                "invalid paged KV completion epoch {requested}; submitted={submitted}, completed={completed}"
            ),
            Self::InvalidTransaction(message) => {
                write!(formatter, "invalid paged KV transaction: {message}")
            }
            Self::StaleBlockId {
                slot,
                expected_generation,
                actual_generation,
            } => write!(
                formatter,
                "stale paged KV block {slot}: expected generation {expected_generation}, actual generation {actual_generation}"
            ),
            Self::HostAllocation(context) => {
                write!(formatter, "host allocation failed while {context}")
            }
            Self::Tensor(error) => write!(formatter, "paged KV tensor operation failed: {error}"),
        }
    }
}

impl std::error::Error for PagedKvPoolError {}

impl From<candle_core::Error> for PagedKvPoolError {
    fn from(error: candle_core::Error) -> Self {
        Self::Tensor(error)
    }
}

/// Fixed storage and capacity policy for one model-level paged K/V pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvPoolConfig {
    block_tokens: NonZeroUsize,
    slab_blocks: NonZeroUsize,
    max_blocks: NonZeroUsize,
}

impl PagedKvPoolConfig {
    pub fn new(block_tokens: usize, slab_blocks: usize, max_blocks: usize) -> PagedKvResult<Self> {
        let block_tokens = NonZeroUsize::new(block_tokens).ok_or_else(|| {
            PagedKvPoolError::InvalidConfig("block_tokens must be positive".into())
        })?;
        let slab_blocks = NonZeroUsize::new(slab_blocks).ok_or_else(|| {
            PagedKvPoolError::InvalidConfig("slab_blocks must be positive".into())
        })?;
        let max_blocks = NonZeroUsize::new(max_blocks)
            .ok_or_else(|| PagedKvPoolError::InvalidConfig("max_blocks must be positive".into()))?;
        if max_blocks.get() > u32::MAX as usize {
            return Err(PagedKvPoolError::InvalidConfig(
                "max_blocks exceeds the generational block-ID range".into(),
            ));
        }
        Ok(Self {
            block_tokens,
            slab_blocks,
            max_blocks,
        })
    }

    #[must_use]
    pub fn block_tokens(self) -> usize {
        self.block_tokens.get()
    }

    #[must_use]
    pub fn slab_blocks(self) -> usize {
        self.slab_blocks.get()
    }

    #[must_use]
    pub fn max_blocks(self) -> usize {
        self.max_blocks.get()
    }
}

/// Uniform full-attention geometry stored in one physical block bundle.
///
/// One block owns K and V for every full-attention layer. Architectures whose
/// full layers use different K/V shapes can create one pool per geometry group
/// while keeping the sequence-level block-table API unchanged.
#[derive(Debug, Clone)]
pub struct PagedKvGeometry {
    full_layers: NonZeroUsize,
    kv_heads: NonZeroUsize,
    head_dim: NonZeroUsize,
    dtype: DType,
    device: Device,
}

impl PagedKvGeometry {
    pub fn new(
        full_layers: usize,
        kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> PagedKvResult<Self> {
        let full_layers = NonZeroUsize::new(full_layers).ok_or_else(|| {
            PagedKvPoolError::InvalidGeometry("full_layers must be positive".into())
        })?;
        let kv_heads = NonZeroUsize::new(kv_heads)
            .ok_or_else(|| PagedKvPoolError::InvalidGeometry("kv_heads must be positive".into()))?;
        let head_dim = NonZeroUsize::new(head_dim)
            .ok_or_else(|| PagedKvPoolError::InvalidGeometry("head_dim must be positive".into()))?;
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(PagedKvPoolError::InvalidGeometry(format!(
                "dtype {dtype:?} is not a supported K/V element type"
            )));
        }
        Ok(Self {
            full_layers,
            kv_heads,
            head_dim,
            dtype,
            device: device.clone(),
        })
    }

    #[must_use]
    pub fn full_layers(&self) -> usize {
        self.full_layers.get()
    }

    #[must_use]
    pub fn kv_heads(&self) -> usize {
        self.kv_heads.get()
    }

    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim.get()
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Monotonic model-submission marker used to quarantine blocks until device
/// work that may still reference them has completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PagedKvEpoch(u64);

impl PagedKvEpoch {
    const ZERO: Self = Self(0);

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Stable logical reference to a reusable physical block slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvBlockId {
    slot: u32,
    generation: u64,
}

impl PagedKvBlockId {
    #[must_use]
    pub const fn slot(self) -> u32 {
        self.slot
    }

    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

/// Model-level physical pool diagnostics. `allocated_bytes` counts each slab
/// exactly once and is therefore safe to aggregate across sequence tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvPoolStats {
    pub block_tokens: usize,
    /// Physical K/V bytes owned by one complete block across every
    /// full-attention layer. Prefix snapshots use this as the accounting unit
    /// when recurrent state must share the same bounded LRU budget.
    pub block_bytes: usize,
    pub slab_blocks: usize,
    pub max_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub reserved_blocks: usize,
    pub live_blocks: usize,
    /// Total logical owners across every live physical block. This exceeds
    /// `live_blocks` when sequence tables or prefix snapshots share pages.
    pub live_references: usize,
    /// Number of live physical blocks referenced by more than one owner.
    pub shared_blocks: usize,
    pub retired_blocks: usize,
    pub allocated_bytes: usize,
    pub submitted_epoch: PagedKvEpoch,
    pub completed_epoch: PagedKvEpoch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PagedKvBlockState {
    Free,
    Reserved { transaction: u64 },
    Live { references: NonZeroUsize },
    Retired { until: PagedKvEpoch },
}

#[derive(Debug)]
struct PagedKvBlockSlot {
    generation: u64,
    state: PagedKvBlockState,
    last_use: PagedKvEpoch,
    slab: usize,
    slab_offset: usize,
}

/// Constant-time aggregate of the mutable slot-state contributions.
///
/// Allocated and free blocks are deliberately derived from `slots.len()` and
/// these mutually exclusive state counts. Keeping only one representation of
/// those values makes a counter drift observable as an invariant failure
/// instead of allowing two independently updated totals to disagree.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PagedKvPoolOccupancy {
    reserved_blocks: usize,
    live_blocks: usize,
    live_references: usize,
    shared_blocks: usize,
    retired_blocks: usize,
}

impl PagedKvPoolOccupancy {
    fn occupied_blocks(self) -> PagedKvResult<usize> {
        self.reserved_blocks
            .checked_add(self.live_blocks)
            .and_then(|blocks| blocks.checked_add(self.retired_blocks))
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV occupancy counters overflow usize".into(),
                )
            })
    }
}

#[derive(Debug)]
struct PagedKvSlab {
    key: Tensor,
    value: Tensor,
    blocks: usize,
}

/// Stable tensor handles for one physical slab used by indirect attention.
///
/// The pool remains the authoritative owner. Cloning the handles here keeps
/// the Metal resources alive while a custom operation is being encoded,
/// without exposing allocator mutation through the attention layer.
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct PagedKvSlabSnapshot {
    key: Tensor,
    value: Tensor,
}

impl PagedKvSlabSnapshot {
    #[must_use]
    #[doc(hidden)]
    pub fn key(&self) -> &Tensor {
        &self.key
    }

    #[must_use]
    #[doc(hidden)]
    pub fn value(&self) -> &Tensor {
        &self.value
    }
}

/// One device page table and slab snapshot shared by every full-attention
/// layer in a reserved packed batch.
#[derive(Debug)]
struct PagedKvAttentionBatch {
    slabs: Vec<PagedKvSlabSnapshot>,
    page_table: Tensor,
    lane_page_offsets: Vec<usize>,
    lane_page_counts: Vec<usize>,
    lane_logical_lens: Vec<usize>,
    block_tokens: usize,
    full_layers: usize,
    kv_heads: usize,
    head_dim: usize,
    dtype: DType,
}

/// Immutable layer/lane view consumed by page-table-aware attention.
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
#[doc(hidden)]
pub struct PagedKvLayerRead<'a> {
    batch: &'a PagedKvAttentionBatch,
    lane: usize,
    layer: usize,
}

/// Immutable whole-batch view for one written full-attention layer.
///
/// All lane metadata indexes the single compact page table owned by the
/// reservation. Keeping the batch borrowed ensures its slab handles and page
/// table remain alive for the duration of one packed attention operation.
#[derive(Debug)]
#[doc(hidden)]
pub struct PagedKvLayerBatchRead<'a> {
    batch: &'a PagedKvAttentionBatch,
    layer: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
impl PagedKvLayerRead<'_> {
    #[must_use]
    #[doc(hidden)]
    pub fn slabs(&self) -> &[PagedKvSlabSnapshot] {
        &self.batch.slabs
    }

    #[must_use]
    #[doc(hidden)]
    pub fn page_table(&self) -> &Tensor {
        &self.batch.page_table
    }

    #[must_use]
    #[doc(hidden)]
    pub fn page_offset(&self) -> usize {
        self.batch.lane_page_offsets[self.lane]
    }

    #[must_use]
    #[doc(hidden)]
    pub fn page_count(&self) -> usize {
        self.batch.lane_page_counts[self.lane]
    }

    #[must_use]
    #[doc(hidden)]
    pub fn logical_len(&self) -> usize {
        self.batch.lane_logical_lens[self.lane]
    }

    #[must_use]
    #[doc(hidden)]
    pub fn block_tokens(&self) -> usize {
        self.batch.block_tokens
    }

    #[must_use]
    #[doc(hidden)]
    pub fn full_layers(&self) -> usize {
        self.batch.full_layers
    }

    #[must_use]
    #[doc(hidden)]
    pub fn kv_heads(&self) -> usize {
        self.batch.kv_heads
    }

    #[must_use]
    #[doc(hidden)]
    pub fn head_dim(&self) -> usize {
        self.batch.head_dim
    }

    #[must_use]
    #[doc(hidden)]
    pub fn dtype(&self) -> DType {
        self.batch.dtype
    }

    #[must_use]
    #[doc(hidden)]
    pub fn layer(&self) -> usize {
        self.layer
    }
}

impl PagedKvLayerBatchRead<'_> {
    #[must_use]
    #[doc(hidden)]
    pub fn slabs(&self) -> &[PagedKvSlabSnapshot] {
        &self.batch.slabs
    }

    #[must_use]
    #[doc(hidden)]
    pub fn page_table(&self) -> &Tensor {
        &self.batch.page_table
    }

    #[must_use]
    #[doc(hidden)]
    pub fn lane_page_offsets(&self) -> &[usize] {
        &self.batch.lane_page_offsets
    }

    #[must_use]
    #[doc(hidden)]
    pub fn lane_page_counts(&self) -> &[usize] {
        &self.batch.lane_page_counts
    }

    #[must_use]
    #[doc(hidden)]
    pub fn lane_logical_lens(&self) -> &[usize] {
        &self.batch.lane_logical_lens
    }

    #[must_use]
    #[doc(hidden)]
    pub fn block_tokens(&self) -> usize {
        self.batch.block_tokens
    }

    #[must_use]
    #[doc(hidden)]
    pub fn full_layers(&self) -> usize {
        self.batch.full_layers
    }

    #[must_use]
    #[doc(hidden)]
    pub fn layer(&self) -> usize {
        self.layer
    }

    #[must_use]
    #[doc(hidden)]
    pub fn kv_heads(&self) -> usize {
        self.batch.kv_heads
    }

    #[must_use]
    #[doc(hidden)]
    pub fn head_dim(&self) -> usize {
        self.batch.head_dim
    }

    #[must_use]
    #[doc(hidden)]
    pub fn dtype(&self) -> DType {
        self.batch.dtype
    }

    #[must_use]
    #[doc(hidden)]
    pub fn lane_count(&self) -> usize {
        self.batch.lane_page_offsets.len()
    }
}

impl PagedKvSlab {
    fn new(
        blocks: usize,
        config: PagedKvPoolConfig,
        geometry: &PagedKvGeometry,
    ) -> PagedKvResult<Self> {
        let shape = (
            blocks,
            geometry.full_layers(),
            geometry.kv_heads(),
            config.block_tokens(),
            geometry.head_dim(),
        );
        let key = Tensor::zeros(shape, geometry.dtype(), geometry.device())?;
        let value = Tensor::zeros(shape, geometry.dtype(), geometry.device())?;
        Ok(Self { key, value, blocks })
    }

    fn layer_views(&self, block: usize, layer: usize) -> PagedKvResult<(Tensor, Tensor)> {
        if block >= self.blocks {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "slab block {block} is outside {} physical blocks",
                self.blocks
            )));
        }
        let layer_view = |tensor: &Tensor| -> PagedKvResult<Tensor> {
            Ok(tensor
                .narrow(0, block, 1)?
                .narrow(1, layer, 1)?
                .squeeze(1)?)
        };
        Ok((layer_view(&self.key)?, layer_view(&self.value)?))
    }
}

#[derive(Debug)]
struct PagedKvPool {
    config: PagedKvPoolConfig,
    geometry: PagedKvGeometry,
    block_bytes: usize,
    slabs: Vec<PagedKvSlab>,
    slots: Vec<PagedKvBlockSlot>,
    occupancy: PagedKvPoolOccupancy,
    next_transaction: u64,
    active_transaction: Option<u64>,
    submitted_epoch: PagedKvEpoch,
    completed_epoch: PagedKvEpoch,
}

impl PagedKvPool {
    fn new(config: PagedKvPoolConfig, geometry: PagedKvGeometry) -> PagedKvResult<Self> {
        let max_slabs = config
            .max_blocks()
            .checked_add(config.slab_blocks() - 1)
            .and_then(|value| value.checked_div(config.slab_blocks()))
            .ok_or_else(|| {
                PagedKvPoolError::InvalidConfig("maximum slab count overflows usize".into())
            })?;
        let block_bytes = bytes_per_block(config, &geometry)?;
        let _ = block_bytes
            .checked_mul(config.max_blocks())
            .ok_or_else(|| {
                PagedKvPoolError::InvalidConfig("maximum pool byte size overflows usize".into())
            })?;
        let mut slabs = Vec::new();
        slabs
            .try_reserve_exact(max_slabs)
            .map_err(|_| PagedKvPoolError::HostAllocation("reserving slab metadata"))?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(config.max_blocks())
            .map_err(|_| PagedKvPoolError::HostAllocation("reserving block metadata"))?;
        Ok(Self {
            config,
            geometry,
            block_bytes,
            slabs,
            slots,
            occupancy: PagedKvPoolOccupancy::default(),
            next_transaction: 0,
            active_transaction: None,
            submitted_epoch: PagedKvEpoch::ZERO,
            completed_epoch: PagedKvEpoch::ZERO,
        })
    }

    fn free_blocks(&self) -> PagedKvResult<usize> {
        self.slots
            .len()
            .checked_sub(self.occupancy.occupied_blocks()?)
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV occupancy exceeds allocated block count".into(),
                )
            })
    }

    fn ensure_free_blocks(&mut self, requested: usize) -> PagedKvResult<()> {
        let free_blocks = self.free_blocks()?;
        let unallocated_blocks = self.config.max_blocks() - self.slots.len();
        if requested > free_blocks.saturating_add(unallocated_blocks) {
            return Err(PagedKvPoolError::Exhausted {
                requested_blocks: requested,
                free_blocks,
                unallocated_blocks,
            });
        }
        let missing = requested.saturating_sub(free_blocks);
        if missing == 0 {
            return Ok(());
        }

        let mut planned_blocks = 0_usize;
        let mut new_slabs = Vec::new();
        let slab_count = missing
            .checked_add(self.config.slab_blocks() - 1)
            .and_then(|value| value.checked_div(self.config.slab_blocks()))
            .ok_or_else(|| {
                PagedKvPoolError::InvalidConfig("slab reservation count overflows usize".into())
            })?;
        new_slabs
            .try_reserve_exact(slab_count)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging new slabs"))?;
        while planned_blocks < missing {
            let remaining_capacity = self
                .config
                .max_blocks()
                .checked_sub(self.slots.len())
                .and_then(|value| value.checked_sub(planned_blocks))
                .ok_or_else(|| {
                    PagedKvPoolError::InvalidConfig("slab capacity accounting underflow".into())
                })?;
            let blocks = self.config.slab_blocks().min(remaining_capacity);
            if blocks == 0 {
                return Err(PagedKvPoolError::Exhausted {
                    requested_blocks: requested,
                    free_blocks,
                    unallocated_blocks,
                });
            }
            new_slabs.push(PagedKvSlab::new(blocks, self.config, &self.geometry)?);
            planned_blocks = planned_blocks.checked_add(blocks).ok_or_else(|| {
                PagedKvPoolError::InvalidConfig("planned block count overflows usize".into())
            })?;
        }

        // Publish slab tensors and their free slots only after every K/V
        // allocation above succeeded.
        for slab in new_slabs {
            let slab_index = self.slabs.len();
            let blocks = slab.blocks;
            self.slabs.push(slab);
            for slab_offset in 0..blocks {
                self.slots.push(PagedKvBlockSlot {
                    generation: 0,
                    state: PagedKvBlockState::Free,
                    last_use: self.completed_epoch,
                    slab: slab_index,
                    slab_offset,
                });
            }
        }
        Ok(())
    }

    fn next_transaction_and_epoch(&mut self) -> PagedKvResult<(u64, PagedKvEpoch)> {
        if let Some(active) = self.active_transaction {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "paged KV transaction {active} is still active"
            )));
        }
        let transaction = self.next_transaction.checked_add(1).ok_or_else(|| {
            PagedKvPoolError::InvalidTransaction("transaction counter overflow".into())
        })?;
        let epoch = self.submitted_epoch.0.checked_add(1).ok_or_else(|| {
            PagedKvPoolError::InvalidTransaction("submission epoch overflow".into())
        })?;
        self.next_transaction = transaction;
        self.active_transaction = Some(transaction);
        self.submitted_epoch = PagedKvEpoch(epoch);
        Ok((transaction, self.submitted_epoch))
    }

    fn reserve_blocks(
        &mut self,
        count: usize,
        transaction: u64,
    ) -> PagedKvResult<Vec<PagedKvBlockId>> {
        if self.active_transaction != Some(transaction) {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "transaction {transaction} is not the active paged KV transaction"
            )));
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        self.ensure_free_blocks(count)?;
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(count)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging block IDs"))?;
        for (index, slot) in self.slots.iter().enumerate() {
            if slot.state == PagedKvBlockState::Free {
                if slot.generation == u64::MAX {
                    return Err(PagedKvPoolError::InvalidTransaction(format!(
                        "block slot {index} generation counter overflow"
                    )));
                }
                indices.push(index);
                if indices.len() == count {
                    break;
                }
            }
        }
        if indices.len() != count {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "reserved {} free blocks after preflighting {count}",
                indices.len()
            )));
        }

        let mut ids = Vec::new();
        ids.try_reserve_exact(count)
            .map_err(|_| PagedKvPoolError::HostAllocation("building block IDs"))?;
        for &index in &indices {
            let slot = &self.slots[index];
            ids.push(PagedKvBlockId {
                slot: u32::try_from(index).map_err(|_| {
                    PagedKvPoolError::InvalidConfig("block slot exceeds u32".into())
                })?,
                generation: slot.generation.checked_add(1).ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction(format!(
                        "block slot {index} generation counter overflow"
                    ))
                })?,
            });
        }
        let next_reserved = self
            .occupancy
            .reserved_blocks
            .checked_add(count)
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV reserved-block counter overflows usize".into(),
                )
            })?;

        // Every fallible operation and aggregate-counter update was
        // preflighted above. Publish slots and their occupancy as one mutation.
        for (&index, &id) in indices.iter().zip(&ids) {
            let slot = &mut self.slots[index];
            slot.generation = id.generation;
            slot.state = PagedKvBlockState::Reserved { transaction };
            slot.last_use = self.completed_epoch;
        }
        self.occupancy.reserved_blocks = next_reserved;
        Ok(ids)
    }

    fn slot(&self, id: PagedKvBlockId) -> PagedKvResult<&PagedKvBlockSlot> {
        let slot = self
            .slots
            .get(id.slot as usize)
            .ok_or(PagedKvPoolError::StaleBlockId {
                slot: id.slot,
                expected_generation: id.generation,
                actual_generation: 0,
            })?;
        if slot.generation != id.generation {
            return Err(PagedKvPoolError::StaleBlockId {
                slot: id.slot,
                expected_generation: id.generation,
                actual_generation: slot.generation,
            });
        }
        Ok(slot)
    }

    fn slot_mut(&mut self, id: PagedKvBlockId) -> PagedKvResult<&mut PagedKvBlockSlot> {
        let actual_generation = self
            .slots
            .get(id.slot as usize)
            .map_or(0, |slot| slot.generation);
        if actual_generation != id.generation {
            return Err(PagedKvPoolError::StaleBlockId {
                slot: id.slot,
                expected_generation: id.generation,
                actual_generation,
            });
        }
        self.slots
            .get_mut(id.slot as usize)
            .ok_or(PagedKvPoolError::StaleBlockId {
                slot: id.slot,
                expected_generation: id.generation,
                actual_generation,
            })
    }

    fn validate_owned_block(
        &self,
        id: PagedKvBlockId,
        transaction: Option<u64>,
    ) -> PagedKvResult<()> {
        let slot = self.slot(id)?;
        let valid = match (slot.state, transaction) {
            (PagedKvBlockState::Live { .. }, _) => true,
            (PagedKvBlockState::Reserved { transaction: owner }, Some(expected)) => {
                owner == expected
            }
            _ => false,
        };
        if !valid {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "block {} is not owned by the append transaction",
                id.slot
            )));
        }
        Ok(())
    }

    fn live_reference_plan(
        &self,
        ids: &[PagedKvBlockId],
        context: &'static str,
    ) -> PagedKvResult<Vec<(usize, usize)>> {
        let mut counts = Vec::<(usize, usize)>::new();
        counts
            .try_reserve_exact(ids.len())
            .map_err(|_| PagedKvPoolError::HostAllocation(context))?;
        for &id in ids {
            let slot = self.slot(id)?;
            if !matches!(slot.state, PagedKvBlockState::Live { .. }) {
                return Err(PagedKvPoolError::InvalidTransaction(format!(
                    "block {} is not live",
                    id.slot
                )));
            }
            let index = id.slot as usize;
            counts.push((index, 1));
        }
        counts.sort_unstable_by_key(|(index, _)| *index);
        let mut compact_len = 0_usize;
        for read in 0..counts.len() {
            let (index, count) = counts[read];
            if compact_len > 0 && counts[compact_len - 1].0 == index {
                counts[compact_len - 1].1 = counts[compact_len - 1]
                    .1
                    .checked_add(count)
                    .ok_or_else(|| {
                        PagedKvPoolError::InvalidTransaction(
                            "paged KV reference multiplicity overflows usize".into(),
                        )
                    })?;
            } else {
                counts[compact_len] = (index, count);
                compact_len += 1;
            }
        }
        counts.truncate(compact_len);
        Ok(counts)
    }

    fn live_references(&self, id: PagedKvBlockId) -> PagedKvResult<usize> {
        let slot = self.slot(id)?;
        match slot.state {
            PagedKvBlockState::Live { references } => Ok(references.get()),
            _ => Err(PagedKvPoolError::InvalidTransaction(format!(
                "block {} is not live",
                id.slot
            ))),
        }
    }

    /// Retain every requested live block or leave all reference counts
    /// unchanged. Duplicate IDs represent multiple logical owners and are
    /// counted explicitly after their common generation has been validated.
    fn retain_live(&mut self, ids: &[PagedKvBlockId]) -> PagedKvResult<()> {
        let plan = self.live_reference_plan(ids, "staging paged KV retains")?;
        let mut newly_shared = 0_usize;
        for &(index, count) in &plan {
            let PagedKvBlockState::Live { references } = self.slots[index].state else {
                unreachable!("live retain plan was validated")
            };
            let next = references.get().checked_add(count).ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(format!(
                    "block {index} reference count overflows usize"
                ))
            })?;
            newly_shared = newly_shared
                .checked_add(usize::from(references.get() == 1 && next > 1))
                .ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction(
                        "paged KV shared-block counter overflows usize".into(),
                    )
                })?;
        }
        let next_live_references = self
            .occupancy
            .live_references
            .checked_add(ids.len())
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV live-reference counter overflows usize".into(),
                )
            })?;
        let next_shared_blocks = self
            .occupancy
            .shared_blocks
            .checked_add(newly_shared)
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV shared-block counter overflows usize".into(),
                )
            })?;

        for (index, count) in plan {
            let PagedKvBlockState::Live { references } = self.slots[index].state else {
                unreachable!("live retain plan was validated")
            };
            let next = references
                .get()
                .checked_add(count)
                .expect("live retain overflow was preflighted");
            self.slots[index].state = PagedKvBlockState::Live {
                references: NonZeroUsize::new(next).expect("retained references stay positive"),
            };
        }
        self.occupancy.live_references = next_live_references;
        self.occupancy.shared_blocks = next_shared_blocks;
        Ok(())
    }

    /// Release every requested live reference or leave all blocks unchanged.
    /// The last owner preserves the existing asynchronous epoch quarantine.
    fn release_live(&mut self, ids: &[PagedKvBlockId]) -> PagedKvResult<()> {
        let plan = self.live_reference_plan(ids, "staging paged KV releases")?;
        let next_occupancy = self.occupancy_after_live_releases(&plan, self.occupancy)?;
        self.apply_live_release_states(&plan);
        self.occupancy = next_occupancy;
        Ok(())
    }

    /// Project a validated release plan against an arbitrary aggregate base.
    ///
    /// Commit uses a base that already includes its Reserved -> Live(1)
    /// transitions, allowing the new owners and superseded owners to publish as
    /// one all-or-nothing counter update.
    fn occupancy_after_live_releases(
        &self,
        plan: &[(usize, usize)],
        mut next: PagedKvPoolOccupancy,
    ) -> PagedKvResult<PagedKvPoolOccupancy> {
        let mut released_references = 0_usize;
        let mut released_blocks = 0_usize;
        let mut unshared_blocks = 0_usize;
        let mut newly_retired = 0_usize;
        for &(index, count) in plan {
            let PagedKvBlockState::Live { references } = self.slots[index].state else {
                unreachable!("live release plan was validated")
            };
            if references.get() < count {
                return Err(PagedKvPoolError::InvalidTransaction(format!(
                    "block {index} has {} references but release requires {count}",
                    references.get()
                )));
            }
            let remaining = references.get() - count;
            released_references = released_references.checked_add(count).ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV released-reference count overflows usize".into(),
                )
            })?;
            if remaining == 0 {
                released_blocks = released_blocks.checked_add(1).ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction(
                        "paged KV released-block count overflows usize".into(),
                    )
                })?;
                if self.slots[index].last_use > self.completed_epoch {
                    newly_retired = newly_retired.checked_add(1).ok_or_else(|| {
                        PagedKvPoolError::InvalidTransaction(
                            "paged KV retired-block counter overflows usize".into(),
                        )
                    })?;
                }
            }
            if references.get() > 1 && remaining <= 1 {
                unshared_blocks = unshared_blocks.checked_add(1).ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction(
                        "paged KV unshared-block count overflows usize".into(),
                    )
                })?;
            }
        }
        next.live_references = next
            .live_references
            .checked_sub(released_references)
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV live-reference counter underflows".into(),
                )
            })?;
        next.live_blocks = next
            .live_blocks
            .checked_sub(released_blocks)
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV live-block counter underflows".into(),
                )
            })?;
        next.shared_blocks = next
            .shared_blocks
            .checked_sub(unshared_blocks)
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV shared-block counter underflows".into(),
                )
            })?;
        next.retired_blocks = next
            .retired_blocks
            .checked_add(newly_retired)
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV retired-block counter overflows usize".into(),
                )
            })?;
        Ok(next)
    }

    fn apply_live_release_states(&mut self, plan: &[(usize, usize)]) {
        let completed = self.completed_epoch;
        for &(index, count) in plan {
            let slot = &mut self.slots[index];
            let PagedKvBlockState::Live { references } = slot.state else {
                unreachable!("live release plan was validated")
            };
            let remaining = references.get() - count;
            slot.state = if let Some(references) = NonZeroUsize::new(remaining) {
                PagedKvBlockState::Live { references }
            } else if slot.last_use <= completed {
                PagedKvBlockState::Free
            } else {
                PagedKvBlockState::Retired {
                    until: slot.last_use,
                }
            };
        }
    }

    fn layer_views(
        &self,
        id: PagedKvBlockId,
        layer: usize,
        transaction: Option<u64>,
    ) -> PagedKvResult<(Tensor, Tensor)> {
        self.validate_owned_block(id, transaction)?;
        if layer >= self.geometry.full_layers() {
            return Err(PagedKvPoolError::InvalidGeometry(format!(
                "full layer {layer} is outside {} layers",
                self.geometry.full_layers()
            )));
        }
        let slot = self.slot(id)?;
        self.slabs
            .get(slot.slab)
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(format!(
                    "block {} references missing slab {}",
                    id.slot, slot.slab
                ))
            })?
            .layer_views(slot.slab_offset, layer)
    }

    fn mark_use_attempt(
        &mut self,
        ids: &[PagedKvBlockId],
        transaction: u64,
        epoch: PagedKvEpoch,
    ) -> PagedKvResult<()> {
        if self.active_transaction != Some(transaction) {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "transaction {transaction} is not the active paged KV transaction"
            )));
        }
        for &id in ids {
            self.validate_owned_block(id, Some(transaction))?;
        }
        for &id in ids {
            let slot = self.slot_mut(id)?;
            slot.last_use = slot.last_use.max(epoch);
        }
        Ok(())
    }

    fn commit_reserved(&mut self, ids: &[PagedKvBlockId], transaction: u64) -> PagedKvResult<()> {
        self.commit_reserved_replacing(ids, &[], transaction)
    }

    fn commit_reserved_replacing(
        &mut self,
        ids: &[PagedKvBlockId],
        superseded: &[PagedKvBlockId],
        transaction: u64,
    ) -> PagedKvResult<()> {
        if self.active_transaction != Some(transaction) {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "transaction {transaction} is not the active paged KV transaction"
            )));
        }
        for &id in ids {
            let slot = self.slot(id)?;
            if slot.state != (PagedKvBlockState::Reserved { transaction }) {
                return Err(PagedKvPoolError::InvalidTransaction(format!(
                    "block {} is not reserved by transaction {transaction}",
                    id.slot
                )));
            }
        }
        let release_plan =
            self.live_reference_plan(superseded, "staging superseded paged KV references")?;
        let mut next_occupancy = self.occupancy;
        next_occupancy.reserved_blocks = next_occupancy
            .reserved_blocks
            .checked_sub(ids.len())
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV reserved-block counter underflows during commit".into(),
                )
            })?;
        next_occupancy.live_blocks = next_occupancy
            .live_blocks
            .checked_add(ids.len())
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV live-block counter overflows during commit".into(),
                )
            })?;
        next_occupancy.live_references = next_occupancy
            .live_references
            .checked_add(ids.len())
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV live-reference counter overflows during commit".into(),
                )
            })?;
        next_occupancy = self.occupancy_after_live_releases(&release_plan, next_occupancy)?;

        // IDs and generations were validated above; no fallible work remains
        // between publishing slot states and the corresponding aggregate.
        for &id in ids {
            self.slots[id.slot as usize].state = PagedKvBlockState::Live {
                references: NonZeroUsize::MIN,
            };
        }
        self.apply_live_release_states(&release_plan);
        self.occupancy = next_occupancy;
        self.active_transaction = None;
        Ok(())
    }

    fn rollback_reserved(&mut self, ids: &[PagedKvBlockId], transaction: u64) {
        if self.active_transaction != Some(transaction) {
            return;
        }
        let completed = self.completed_epoch;
        let mut released_reserved = 0_usize;
        let mut newly_retired = 0_usize;
        for (position, &id) in ids.iter().enumerate() {
            if ids[..position].contains(&id) {
                continue;
            }
            let Ok(slot) = self.slot(id) else {
                continue;
            };
            if slot.state != (PagedKvBlockState::Reserved { transaction }) {
                continue;
            }
            let Some(next_released) = released_reserved.checked_add(1) else {
                // Rollback is fail-closed and cannot surface a new error.
                return;
            };
            released_reserved = next_released;
            if slot.last_use > completed {
                let Some(next_retired) = newly_retired.checked_add(1) else {
                    return;
                };
                newly_retired = next_retired;
            }
        }
        let Some(next_reserved_blocks) = self
            .occupancy
            .reserved_blocks
            .checked_sub(released_reserved)
        else {
            return;
        };
        let Some(next_retired_blocks) = self.occupancy.retired_blocks.checked_add(newly_retired)
        else {
            return;
        };

        for (position, &id) in ids.iter().enumerate() {
            if ids[..position].contains(&id) {
                continue;
            }
            let Ok(slot) = self.slot_mut(id) else {
                continue;
            };
            if slot.state != (PagedKvBlockState::Reserved { transaction }) {
                continue;
            }
            slot.state = if slot.last_use <= completed {
                PagedKvBlockState::Free
            } else {
                PagedKvBlockState::Retired {
                    until: slot.last_use,
                }
            };
        }
        self.occupancy.reserved_blocks = next_reserved_blocks;
        self.occupancy.retired_blocks = next_retired_blocks;
        self.active_transaction = None;
    }

    fn mark_completed_through(&mut self, epoch: PagedKvEpoch) -> PagedKvResult<()> {
        if let Some(active) = self.active_transaction {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "cannot publish device completion while paged KV transaction {active} is active"
            )));
        }
        if epoch < self.completed_epoch || epoch > self.submitted_epoch {
            return Err(PagedKvPoolError::InvalidEpoch {
                submitted: self.submitted_epoch.0,
                completed: self.completed_epoch.0,
                requested: epoch.0,
            });
        }
        let reclaimed = self
            .slots
            .iter()
            .filter(
                |slot| matches!(slot.state, PagedKvBlockState::Retired { until } if until <= epoch),
            )
            .count();
        let next_retired_blocks = self
            .occupancy
            .retired_blocks
            .checked_sub(reclaimed)
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV retired-block counter underflows on completion".into(),
                )
            })?;

        self.completed_epoch = epoch;
        for slot in &mut self.slots {
            if matches!(
                slot.state,
                PagedKvBlockState::Retired { until } if until <= epoch
            ) {
                slot.state = PagedKvBlockState::Free;
            }
        }
        self.occupancy.retired_blocks = next_retired_blocks;
        Ok(())
    }

    fn stats(&self) -> PagedKvResult<PagedKvPoolStats> {
        let allocated_bytes = self
            .slots
            .len()
            .checked_mul(self.block_bytes)
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV allocated-byte counter overflows usize".into(),
                )
            })?;
        Ok(PagedKvPoolStats {
            block_tokens: self.config.block_tokens(),
            block_bytes: self.block_bytes,
            slab_blocks: self.config.slab_blocks(),
            max_blocks: self.config.max_blocks(),
            allocated_blocks: self.slots.len(),
            free_blocks: self.free_blocks()?,
            reserved_blocks: self.occupancy.reserved_blocks,
            live_blocks: self.occupancy.live_blocks,
            live_references: self.occupancy.live_references,
            shared_blocks: self.occupancy.shared_blocks,
            retired_blocks: self.occupancy.retired_blocks,
            allocated_bytes,
            submitted_epoch: self.submitted_epoch,
            completed_epoch: self.completed_epoch,
        })
    }

    #[cfg(test)]
    fn scan_occupancy(&self) -> PagedKvPoolOccupancy {
        let mut occupancy = PagedKvPoolOccupancy::default();
        for slot in &self.slots {
            match slot.state {
                PagedKvBlockState::Free => {}
                PagedKvBlockState::Reserved { .. } => {
                    occupancy.reserved_blocks += 1;
                }
                PagedKvBlockState::Live { references } => {
                    occupancy.live_blocks += 1;
                    occupancy.live_references += references.get();
                    occupancy.shared_blocks += usize::from(references.get() > 1);
                }
                PagedKvBlockState::Retired { .. } => {
                    occupancy.retired_blocks += 1;
                }
            }
        }
        occupancy
    }
}

fn bytes_per_block(config: PagedKvPoolConfig, geometry: &PagedKvGeometry) -> PagedKvResult<usize> {
    2_usize
        .checked_mul(geometry.full_layers())
        .and_then(|value| value.checked_mul(geometry.kv_heads()))
        .and_then(|value| value.checked_mul(config.block_tokens()))
        .and_then(|value| value.checked_mul(geometry.head_dim()))
        .and_then(|value| value.checked_mul(geometry.dtype().size_in_bytes()))
        .ok_or_else(|| PagedKvPoolError::InvalidGeometry("bytes per block overflow usize".into()))
}

/// Cloneable, model-owned handle to the shared physical paged K/V pool.
#[derive(Clone)]
pub struct SharedPagedKvPool {
    inner: Rc<RefCell<PagedKvPool>>,
}

impl fmt::Debug for SharedPagedKvPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SharedPagedKvPool")
            .field("strong_count", &Rc::strong_count(&self.inner))
            .finish_non_exhaustive()
    }
}

impl SharedPagedKvPool {
    pub fn new(config: PagedKvPoolConfig, geometry: PagedKvGeometry) -> PagedKvResult<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(PagedKvPool::new(config, geometry)?)),
        })
    }

    pub fn ensure_free_blocks(&self, requested: usize) -> PagedKvResult<()> {
        let mut pool = self
            .inner
            .try_borrow_mut()
            .map_err(|_| PagedKvPoolError::PoolBusy)?;
        if let Some(active) = pool.active_transaction {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "cannot grow paged KV storage while transaction {active} is active"
            )));
        }
        pool.ensure_free_blocks(requested)
    }

    pub fn mark_completed_through(&self, epoch: PagedKvEpoch) -> PagedKvResult<()> {
        self.inner
            .try_borrow_mut()
            .map_err(|_| PagedKvPoolError::PoolBusy)?
            .mark_completed_through(epoch)
    }

    pub fn stats(&self) -> PagedKvResult<PagedKvPoolStats> {
        self.inner
            .try_borrow()
            .map_err(|_| PagedKvPoolError::PoolBusy)?
            .stats()
    }
}

/// Per-sequence logical mapping over a model-shared physical block pool.
#[derive(Debug)]
pub struct PagedKvBlockTable {
    pool: SharedPagedKvPool,
    identity: Rc<PagedKvTableIdentity>,
    blocks: Vec<PagedKvBlockId>,
    logical_len: usize,
    limit: usize,
}

#[derive(Debug)]
struct PagedKvTableIdentity {
    in_flight: Cell<Option<u64>>,
    alive: Cell<bool>,
}

impl PagedKvBlockTable {
    pub fn new(pool: SharedPagedKvPool, limit: usize) -> PagedKvResult<Self> {
        if limit == 0 {
            return Err(PagedKvPoolError::InvalidConfig(
                "sequence table limit must be positive".into(),
            ));
        }
        Ok(Self {
            pool,
            identity: Rc::new(PagedKvTableIdentity {
                in_flight: Cell::new(None),
                alive: Cell::new(true),
            }),
            blocks: Vec::new(),
            logical_len: 0,
            limit,
        })
    }

    #[must_use]
    pub fn logical_len(&self) -> usize {
        self.logical_len
    }

    #[must_use]
    pub fn owned_blocks(&self) -> usize {
        self.blocks.len()
    }

    #[must_use]
    pub fn block_ids(&self) -> &[PagedKvBlockId] {
        &self.blocks
    }

    /// Attach an immutable prefix from another table in the same physical
    /// pool. The selected pages are retained atomically and remain shared
    /// until either owner clears, drops, or replaces a partial tail through
    /// copy-on-write.
    pub fn attach_prefix(
        &mut self,
        source: &PagedKvBlockTable,
        logical_len: usize,
    ) -> PagedKvResult<()> {
        if Rc::ptr_eq(&self.identity, &source.identity) {
            return Err(PagedKvPoolError::InvalidTransaction(
                "a paged KV table cannot attach itself as a prefix".into(),
            ));
        }
        if !Rc::ptr_eq(&self.pool.inner, &source.pool.inner) {
            return Err(PagedKvPoolError::InvalidTransaction(
                "prefix source belongs to another paged KV pool".into(),
            ));
        }
        if !self.identity.alive.get()
            || self.identity.in_flight.get().is_some()
            || !self.blocks.is_empty()
            || self.logical_len != 0
        {
            return Err(PagedKvPoolError::InvalidTransaction(
                "prefix destination must be a live, empty, unlocked table".into(),
            ));
        }
        if !source.identity.alive.get() || source.identity.in_flight.get().is_some() {
            return Err(PagedKvPoolError::InvalidTransaction(
                "prefix source must be live and unlocked".into(),
            ));
        }
        if logical_len > source.logical_len {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "prefix length {logical_len} exceeds source length {}",
                source.logical_len
            )));
        }
        if logical_len > self.limit {
            return Err(PagedKvPoolError::ContextLimit {
                limit: self.limit,
                requested: logical_len,
            });
        }

        let (source_blocks, prefix_blocks) = {
            let pool = self
                .pool
                .inner
                .try_borrow()
                .map_err(|_| PagedKvPoolError::PoolBusy)?;
            let block_tokens = pool.config.block_tokens();
            let source_blocks = div_ceil_checked(source.logical_len, block_tokens)?;
            if source.blocks.len() != source_blocks {
                return Err(PagedKvPoolError::InvalidTransaction(
                    "prefix source block table disagrees with its logical length".into(),
                ));
            }
            for &id in &source.blocks {
                pool.validate_owned_block(id, None)?;
            }
            (source_blocks, div_ceil_checked(logical_len, block_tokens)?)
        };
        debug_assert_eq!(source_blocks, source.blocks.len());
        self.blocks
            .try_reserve_exact(prefix_blocks)
            .map_err(|_| PagedKvPoolError::HostAllocation("reserving shared prefix table"))?;
        let prefix = source.blocks.get(..prefix_blocks).ok_or_else(|| {
            PagedKvPoolError::InvalidTransaction(
                "prefix source does not contain every requested page".into(),
            )
        })?;
        self.pool
            .inner
            .try_borrow_mut()
            .map_err(|_| PagedKvPoolError::PoolBusy)?
            .retain_live(prefix)?;
        self.blocks.extend_from_slice(prefix);
        self.logical_len = logical_len;
        Ok(())
    }

    /// Create a new table that shares this table's prefix and inherits its
    /// logical limit.
    pub fn fork_prefix(&self, logical_len: usize) -> PagedKvResult<Self> {
        let mut fork = Self::new(self.pool.clone(), self.limit)?;
        fork.attach_prefix(self, logical_len)?;
        Ok(fork)
    }

    /// Per-layer, per-sequence diagnostics for the paged representation.
    ///
    /// Shared free slabs are reported by [`SharedPagedKvPool::stats`]. Here,
    /// allocated capacity and bytes describe this sequence's logical page
    /// ownership. Shared prefixes therefore appear in each owning sequence;
    /// use pool stats for de-duplicated physical memory accounting.
    pub fn stats(&self) -> PagedKvResult<KvCacheStats> {
        let pool = self
            .pool
            .inner
            .try_borrow()
            .map_err(|_| PagedKvPoolError::PoolBusy)?;
        let block_tokens = pool.config.block_tokens();
        let allocated_capacity = self.blocks.len().checked_mul(block_tokens).ok_or_else(|| {
            PagedKvPoolError::InvalidConfig("sequence-owned paged capacity overflows usize".into())
        })?;
        let bytes_per_layer_block = 2_usize
            .checked_mul(pool.geometry.kv_heads())
            .and_then(|value| value.checked_mul(block_tokens))
            .and_then(|value| value.checked_mul(pool.geometry.head_dim()))
            .and_then(|value| value.checked_mul(pool.geometry.dtype().size_in_bytes()))
            .ok_or_else(|| {
                PagedKvPoolError::InvalidGeometry(
                    "per-layer paged block bytes overflow usize".into(),
                )
            })?;
        let allocated_bytes = self
            .blocks
            .len()
            .checked_mul(bytes_per_layer_block)
            .ok_or_else(|| {
                PagedKvPoolError::InvalidGeometry(
                    "sequence-owned paged bytes overflow usize".into(),
                )
            })?;
        let pool_physical_capacity = pool
            .config
            .max_blocks()
            .checked_mul(block_tokens)
            .ok_or_else(|| {
                PagedKvPoolError::InvalidConfig("paged pool token capacity overflows usize".into())
            })?;
        Ok(KvCacheStats {
            policy: KvCachePolicy::Paged,
            logical_len: self.logical_len,
            resident_start: 0,
            resident_len: self.logical_len,
            allocated_capacity,
            allocated_bytes,
            physical_capacity: self.limit.min(pool_physical_capacity),
            logical_limit: self.limit,
        })
    }

    pub fn additional_blocks_for(&self, new_tokens: usize) -> PagedKvResult<usize> {
        let pool = self
            .pool
            .inner
            .try_borrow()
            .map_err(|_| PagedKvPoolError::PoolBusy)?;
        let block_tokens = pool.config.block_tokens();
        let requested = self.logical_len.checked_add(new_tokens).ok_or_else(|| {
            PagedKvPoolError::ContextLimit {
                limit: self.limit,
                requested: usize::MAX,
            }
        })?;
        if requested > self.limit {
            return Err(PagedKvPoolError::ContextLimit {
                limit: self.limit,
                requested,
            });
        }
        let current_blocks = div_ceil_checked(self.logical_len, block_tokens)?;
        if self.blocks.len() != current_blocks {
            return Err(PagedKvPoolError::InvalidTransaction(
                "sequence block table disagrees with its logical length".into(),
            ));
        }
        let required_blocks = div_ceil_checked(requested, block_tokens)?;
        let growth_blocks = required_blocks.saturating_sub(self.blocks.len());
        let needs_tail_copy = if new_tokens > 0 && !self.logical_len.is_multiple_of(block_tokens) {
            let source = self.blocks.last().copied().ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "partial paged KV table has no tail block".into(),
                )
            })?;
            pool.live_references(source)? > 1
        } else {
            false
        };
        growth_blocks
            .checked_add(usize::from(needs_tail_copy))
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV reservation count overflows usize".into(),
                )
            })
    }

    /// Begin a fail-closed append. Capacity and host table storage are reserved
    /// before the transaction can enqueue a K/V write.
    #[allow(clippy::too_many_lines)]
    pub fn begin_append(&mut self, new_tokens: usize) -> PagedKvResult<PagedKvAppend<'_>> {
        if self.identity.in_flight.get().is_some() {
            return Err(PagedKvPoolError::InvalidTransaction(
                "sequence table already has an owned batch reservation".into(),
            ));
        }
        if new_tokens == 0 {
            return Err(PagedKvPoolError::InvalidTransaction(
                "append token count must be positive".into(),
            ));
        }
        let old_len = self.logical_len;
        let new_len =
            old_len
                .checked_add(new_tokens)
                .ok_or_else(|| PagedKvPoolError::ContextLimit {
                    limit: self.limit,
                    requested: usize::MAX,
                })?;
        if new_len > self.limit {
            return Err(PagedKvPoolError::ContextLimit {
                limit: self.limit,
                requested: new_len,
            });
        }

        let (block_tokens, full_layers, cow_source) = {
            let pool = self
                .pool
                .inner
                .try_borrow()
                .map_err(|_| PagedKvPoolError::PoolBusy)?;
            if let Some(active) = pool.active_transaction {
                return Err(PagedKvPoolError::InvalidTransaction(format!(
                    "paged KV transaction {active} is still active"
                )));
            }
            let block_tokens = pool.config.block_tokens();
            let expected_blocks = div_ceil_checked(old_len, block_tokens)?;
            if self.blocks.len() != expected_blocks {
                return Err(PagedKvPoolError::InvalidTransaction(
                    "sequence block table disagrees with its logical length".into(),
                ));
            }
            for &id in &self.blocks {
                pool.validate_owned_block(id, None)?;
            }
            let cow_source = if old_len.is_multiple_of(block_tokens) {
                None
            } else {
                let source = self.blocks.last().copied().ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction(
                        "partial paged KV table has no tail block".into(),
                    )
                })?;
                (pool.live_references(source)? > 1).then_some((
                    self.blocks.len() - 1,
                    source,
                    old_len % block_tokens,
                ))
            };
            (block_tokens, pool.geometry.full_layers(), cow_source)
        };
        let required_blocks = div_ceil_checked(new_len, block_tokens)?;
        let growth_blocks = required_blocks.saturating_sub(self.blocks.len());
        let reserved_blocks = growth_blocks
            .checked_add(usize::from(cow_source.is_some()))
            .ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "paged KV reservation count overflows usize".into(),
                )
            })?;
        let mut provisional_blocks = if cow_source.is_some() {
            let mut blocks = Vec::new();
            blocks.try_reserve_exact(required_blocks).map_err(|_| {
                PagedKvPoolError::HostAllocation("staging copy-on-write sequence table")
            })?;
            blocks.extend_from_slice(&self.blocks);
            Some(blocks)
        } else {
            self.blocks
                .try_reserve_exact(growth_blocks)
                .map_err(|_| PagedKvPoolError::HostAllocation("reserving sequence block table"))?;
            None
        };
        let mut written_layers = Vec::new();
        written_layers.try_reserve_exact(full_layers).map_err(|_| {
            PagedKvPoolError::HostAllocation("reserving per-layer transaction state")
        })?;
        written_layers.resize(full_layers, false);

        // A configured-capacity failure happens before epoch publication or any
        // live cache write. New slabs, if needed, are fully staged by the pool.
        self.pool.ensure_free_blocks(reserved_blocks)?;
        let (transaction, epoch, reserved) = {
            let mut pool = self
                .pool
                .inner
                .try_borrow_mut()
                .map_err(|_| PagedKvPoolError::PoolBusy)?;
            let (transaction, epoch) = pool.next_transaction_and_epoch()?;
            let reserved = match pool.reserve_blocks(reserved_blocks, transaction) {
                Ok(reserved) => reserved,
                Err(error) => {
                    pool.rollback_reserved(&[], transaction);
                    return Err(error);
                }
            };
            (transaction, epoch, reserved)
        };
        let cow = cow_source.map(|(block_index, source, valid_tokens)| PagedKvCowPlan {
            block_index,
            source,
            valid_tokens,
        });
        let growth_offset = usize::from(cow.is_some());
        let staging_result = (|| -> PagedKvResult<()> {
            if let Some(blocks) = &mut provisional_blocks {
                let cow = cow.ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction(
                        "provisional table is missing its copy-on-write plan".into(),
                    )
                })?;
                let replacement = reserved.first().copied().ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction(
                        "copy-on-write append has no replacement page".into(),
                    )
                })?;
                *blocks.get_mut(cow.block_index).ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction(
                        "copy-on-write tail index is outside its provisional table".into(),
                    )
                })? = replacement;
                blocks.extend_from_slice(&reserved[growth_offset..]);
            }
            Ok(())
        })();
        if let Err(error) = staging_result {
            if let Ok(mut pool) = self.pool.inner.try_borrow_mut() {
                pool.rollback_reserved(&reserved, transaction);
            }
            return Err(error);
        }
        Ok(PagedKvAppend {
            table: self,
            transaction,
            epoch,
            old_len,
            new_len,
            new_tokens,
            reserved,
            provisional_blocks,
            cow,
            written_layers,
            finalized: false,
        })
    }

    /// Reset logical state and return completed blocks to the shared pool.
    /// Blocks with pending device writes enter the pool's retired quarantine.
    ///
    /// The normal path is allocation-free. A re-entrant pool borrow or active
    /// batch reservation fails closed without changing logical state or private
    /// block ownership.
    pub fn clear(&mut self) {
        if self.identity.in_flight.get().is_some() {
            return;
        }
        let Ok(mut pool) = self.pool.inner.try_borrow_mut() else {
            return;
        };
        if pool.release_live(&self.blocks).is_err() {
            return;
        }
        self.blocks.clear();
        self.logical_len = 0;
    }
}

impl Drop for PagedKvBlockTable {
    fn drop(&mut self) {
        self.identity.alive.set(false);
        let Ok(mut pool) = self.pool.inner.try_borrow_mut() else {
            // Fail closed. The pool slot remains Live and therefore cannot be
            // handed to another table even though this owner is disappearing.
            return;
        };
        let _ = pool.release_live(&self.blocks);
    }
}

#[derive(Debug, Clone, Copy)]
struct PagedKvCowPlan {
    block_index: usize,
    source: PagedKvBlockId,
    valid_tokens: usize,
}

/// Provisional append over one sequence and every full-attention layer.
/// Dropping without `commit` rolls new block reservations back fail closed.
pub struct PagedKvAppend<'a> {
    table: &'a mut PagedKvBlockTable,
    transaction: u64,
    epoch: PagedKvEpoch,
    old_len: usize,
    new_len: usize,
    new_tokens: usize,
    reserved: Vec<PagedKvBlockId>,
    provisional_blocks: Option<Vec<PagedKvBlockId>>,
    cow: Option<PagedKvCowPlan>,
    written_layers: Vec<bool>,
    finalized: bool,
}

impl fmt::Debug for PagedKvAppend<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PagedKvAppend")
            .field("transaction", &self.transaction)
            .field("epoch", &self.epoch)
            .field("old_len", &self.old_len)
            .field("new_len", &self.new_len)
            .field("reserved", &self.reserved)
            .field("cow", &self.cow)
            .field("written_layers", &self.written_layers)
            .finish_non_exhaustive()
    }
}

impl PagedKvAppend<'_> {
    #[must_use]
    pub fn epoch(&self) -> PagedKvEpoch {
        self.epoch
    }

    #[must_use]
    pub fn old_len(&self) -> usize {
        self.old_len
    }

    #[must_use]
    pub fn new_len(&self) -> usize {
        self.new_len
    }

    fn block_id(&self, index: usize) -> PagedKvResult<PagedKvBlockId> {
        if let Some(blocks) = &self.provisional_blocks {
            return blocks.get(index).copied().ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(format!(
                    "provisional block index {index} is unavailable"
                ))
            });
        }
        if let Some(&id) = self.table.blocks.get(index) {
            return Ok(id);
        }
        let reserved_index = index.checked_sub(self.table.blocks.len()).ok_or_else(|| {
            PagedKvPoolError::InvalidTransaction("block-table index underflow".into())
        })?;
        self.reserved.get(reserved_index).copied().ok_or_else(|| {
            PagedKvPoolError::InvalidTransaction(format!(
                "provisional block index {index} is unavailable"
            ))
        })
    }

    /// Write one full layer's K/V for this append. Inputs must be shaped
    /// `[1, kv_heads, new_tokens, head_dim]` on the pool device and dtype.
    pub fn write_layer(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> PagedKvResult<()> {
        let geometry = {
            self.table
                .pool
                .inner
                .try_borrow()
                .map_err(|_| PagedKvPoolError::PoolBusy)?
                .geometry
                .clone()
        };
        if layer >= geometry.full_layers() {
            return Err(PagedKvPoolError::InvalidGeometry(format!(
                "full layer {layer} is outside {} layers",
                geometry.full_layers()
            )));
        }
        if self.written_layers[layer] {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "full layer {layer} was already written"
            )));
        }
        validate_paged_input(key, value, self.new_tokens, &geometry)?;

        if let Some(cow) = self.cow {
            let replacement = self.block_id(cow.block_index)?;
            copy_paged_prefix(
                &self.table.pool,
                self.transaction,
                self.epoch,
                layer,
                cow,
                replacement,
            )?;
        }

        let block_tokens = {
            self.table
                .pool
                .inner
                .try_borrow()
                .map_err(|_| PagedKvPoolError::PoolBusy)?
                .config
                .block_tokens()
        };
        let mut writes = Vec::new();
        let write_count = div_ceil_checked(
            self.new_tokens
                .checked_add(self.old_len % block_tokens)
                .ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction("write span overflows usize".into())
                })?,
            block_tokens,
        )?;
        writes
            .try_reserve_exact(write_count)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging paged K/V writes"))?;
        let mut absolute = self.old_len;
        let mut source_offset = 0_usize;
        while source_offset < self.new_tokens {
            let block_index = absolute / block_tokens;
            let target_offset = absolute % block_tokens;
            let len = (block_tokens - target_offset).min(self.new_tokens - source_offset);
            let id = self.block_id(block_index)?;
            let (target_key, target_value) = self
                .table
                .pool
                .inner
                .try_borrow()
                .map_err(|_| PagedKvPoolError::PoolBusy)?
                .layer_views(id, layer, Some(self.transaction))?;
            writes.push(PagedKvWrite {
                id,
                source_offset,
                target_offset,
                len,
                target_key,
                target_value,
            });
            source_offset += len;
            absolute = absolute.checked_add(len).ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction("write position overflows usize".into())
            })?;
        }

        apply_paged_writes(
            &self.table.pool,
            self.transaction,
            self.epoch,
            key,
            value,
            writes,
        )?;
        self.written_layers[layer] = true;
        Ok(())
    }

    /// Materialize one layer's provisional history in block-table order.
    ///
    /// A history that fits in one block is handed over as the narrowed views
    /// of that block, without a copy; those views are views into the pool's
    /// slabs. A longer history is a new tensor joined by `Tensor::cat`, and
    /// that join may be strided: a partly filled last block in a pool with
    /// more than one layer has no outer stride for the token axis, so candle
    /// concatenates it through its transpose path. Callers that need a
    /// contiguous K/V history make it contiguous themselves.
    ///
    /// A one-block view aliases the slab, so a later write to the same block
    /// changes what the view reads; callers use it within the step that
    /// materialized it.
    pub fn materialize_layer(&self, layer: usize) -> PagedKvResult<(Tensor, Tensor)> {
        if layer >= self.written_layers.len() || !self.written_layers[layer] {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "full layer {layer} has not been written"
            )));
        }
        let block_tokens = self
            .table
            .pool
            .inner
            .try_borrow()
            .map_err(|_| PagedKvPoolError::PoolBusy)?
            .config
            .block_tokens();
        let block_count = div_ceil_checked(self.new_len, block_tokens)?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        keys.try_reserve_exact(block_count)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging materialized K pages"))?;
        values
            .try_reserve_exact(block_count)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging materialized V pages"))?;
        let mut ids = Vec::new();
        ids.try_reserve_exact(block_count)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging materialized block IDs"))?;
        for index in 0..block_count {
            ids.push(self.block_id(index)?);
        }
        // Materialization enqueues reads from every historical page, not just
        // writes to the current suffix. Quarantine all of them through this
        // transaction's completion epoch before Tensor::cat can submit work.
        self.table
            .pool
            .inner
            .try_borrow_mut()
            .map_err(|_| PagedKvPoolError::PoolBusy)?
            .mark_use_attempt(&ids, self.transaction, self.epoch)?;
        let mut remaining = self.new_len;
        for id in ids {
            let (key, value) = self
                .table
                .pool
                .inner
                .try_borrow()
                .map_err(|_| PagedKvPoolError::PoolBusy)?
                .layer_views(id, layer, Some(self.transaction))?;
            let len = remaining.min(block_tokens);
            keys.push(key.narrow(2, 0, len)?);
            values.push(value.narrow(2, 0, len)?);
            remaining -= len;
        }
        if keys.len() == 1 {
            // The whole history fits in one block, so the narrowed block
            // views already are the history: they are handed over as they
            // are, without a copy.
            return match (keys.pop(), values.pop()) {
                (Some(key), Some(value)) => Ok((key, value)),
                _ => Err(PagedKvPoolError::InvalidTransaction(
                    "the one-block history is missing its K/V views".into(),
                )),
            };
        }
        let key_refs = keys.iter().collect::<Vec<_>>();
        let value_refs = values.iter().collect::<Vec<_>>();
        // `Tensor::cat` joins the blocks. The joined history is strided for
        // that pool with candle 0.11 whenever a block view has no outer
        // stride for the token axis, so callers that need a contiguous K/V
        // history copy it themselves.
        Ok((Tensor::cat(&key_refs, 2)?, Tensor::cat(&value_refs, 2)?))
    }

    /// Atomically publish the new logical length and every newly reserved block.
    pub fn commit(mut self) -> PagedKvResult<PagedKvEpoch> {
        if self.written_layers.iter().any(|written| !written) {
            return Err(PagedKvPoolError::InvalidTransaction(
                "cannot commit before every full layer was written".into(),
            ));
        }
        {
            let mut pool = self
                .table
                .pool
                .inner
                .try_borrow_mut()
                .map_err(|_| PagedKvPoolError::PoolBusy)?;
            if let Some(cow) = self.cow {
                pool.commit_reserved_replacing(
                    &self.reserved,
                    std::slice::from_ref(&cow.source),
                    self.transaction,
                )?;
            } else {
                pool.commit_reserved(&self.reserved, self.transaction)?;
            }
        }
        if let Some(blocks) = self.provisional_blocks.take() {
            self.table.blocks = blocks;
        } else {
            self.table.blocks.extend_from_slice(&self.reserved);
        }
        self.table.logical_len = self.new_len;
        self.finalized = true;
        Ok(self.epoch)
    }

    /// Explicitly abandon the append. Device-touched new blocks are quarantined.
    pub fn rollback(mut self) {
        self.rollback_inner();
    }

    fn rollback_inner(&mut self) {
        if self.finalized {
            return;
        }
        if let Ok(mut pool) = self.table.pool.inner.try_borrow_mut() {
            pool.rollback_reserved(&self.reserved, self.transaction);
        }
        self.finalized = true;
    }
}

impl Drop for PagedKvAppend<'_> {
    fn drop(&mut self) {
        self.rollback_inner();
    }
}

/// Immutable host-only description of one lane's next K/V growth.
#[derive(Debug, Clone, Copy)]
pub struct PagedKvGrowthRequest<'a> {
    table: &'a PagedKvBlockTable,
    new_tokens: usize,
}

impl<'a> PagedKvGrowthRequest<'a> {
    #[must_use]
    pub const fn new(table: &'a PagedKvBlockTable, new_tokens: usize) -> Self {
        Self { table, new_tokens }
    }

    #[must_use]
    pub const fn new_tokens(self) -> usize {
        self.new_tokens
    }
}

#[derive(Debug)]
struct PagedKvOwnedLane {
    identity: Rc<PagedKvTableIdentity>,
    old_len: usize,
    new_len: usize,
    new_tokens: usize,
    old_block_count: usize,
    growth_blocks: usize,
    reserved_blocks: usize,
    provisional_blocks: Vec<PagedKvBlockId>,
    cow: Option<PagedKvCowPlan>,
    written_layers: Vec<bool>,
}

impl SharedPagedKvPool {
    #[allow(clippy::too_many_lines)]
    fn prepare_batch_lanes(
        &self,
        requests: &[PagedKvGrowthRequest<'_>],
        block_tokens: usize,
        full_layers: usize,
    ) -> PagedKvResult<(Vec<PagedKvOwnedLane>, usize)> {
        let mut lanes = Vec::new();
        lanes
            .try_reserve_exact(requests.len())
            .map_err(|_| PagedKvPoolError::HostAllocation("staging batch lane reservations"))?;
        let mut total_reserved = 0_usize;
        for (lane_index, request) in requests.iter().enumerate() {
            let table = request.table;
            if !Rc::ptr_eq(&self.inner, &table.pool.inner) {
                return Err(PagedKvPoolError::InvalidTransaction(format!(
                    "lane {lane_index} table belongs to another paged KV pool"
                )));
            }
            if !table.identity.alive.get() || table.identity.in_flight.get().is_some() {
                return Err(PagedKvPoolError::InvalidTransaction(format!(
                    "lane {lane_index} table is unavailable or already reserved"
                )));
            }
            if requests[..lane_index]
                .iter()
                .any(|peer| Rc::ptr_eq(&peer.table.identity, &table.identity))
            {
                return Err(PagedKvPoolError::InvalidTransaction(format!(
                    "lane {lane_index} duplicates an earlier table"
                )));
            }
            if request.new_tokens == 0 {
                return Err(PagedKvPoolError::InvalidTransaction(format!(
                    "lane {lane_index} append token count must be positive"
                )));
            }
            let cow = {
                let pool = self
                    .inner
                    .try_borrow()
                    .map_err(|_| PagedKvPoolError::PoolBusy)?;
                for &id in &table.blocks {
                    pool.validate_owned_block(id, None)?;
                }
                let expected_blocks = div_ceil_checked(table.logical_len, block_tokens)?;
                if table.blocks.len() != expected_blocks {
                    return Err(PagedKvPoolError::InvalidTransaction(format!(
                        "lane {lane_index} block table disagrees with its logical length"
                    )));
                }
                if table.logical_len % block_tokens != 0 {
                    let source = table.blocks.last().copied().ok_or_else(|| {
                        PagedKvPoolError::InvalidTransaction(format!(
                            "lane {lane_index} partial table has no tail block"
                        ))
                    })?;
                    (pool.live_references(source)? > 1).then_some(PagedKvCowPlan {
                        block_index: table.blocks.len() - 1,
                        source,
                        valid_tokens: table.logical_len % block_tokens,
                    })
                } else {
                    None
                }
            };
            let new_len =
                checked_paged_append_len(table.logical_len, request.new_tokens, table.limit)?;
            let required_blocks = div_ceil_checked(new_len, block_tokens)?;
            let growth_blocks = required_blocks.saturating_sub(table.blocks.len());
            let reserved_blocks = growth_blocks
                .checked_add(usize::from(cow.is_some()))
                .ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction(
                        "lane block reservation overflows usize".into(),
                    )
                })?;
            total_reserved = total_reserved.checked_add(reserved_blocks).ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "batch block reservation overflows usize".into(),
                )
            })?;
            let mut provisional_blocks = Vec::new();
            provisional_blocks
                .try_reserve_exact(required_blocks.max(table.blocks.len()))
                .map_err(|_| {
                    PagedKvPoolError::HostAllocation("staging provisional sequence table")
                })?;
            provisional_blocks.extend_from_slice(&table.blocks);
            let mut written_layers = Vec::new();
            written_layers.try_reserve_exact(full_layers).map_err(|_| {
                PagedKvPoolError::HostAllocation("reserving per-layer transaction state")
            })?;
            written_layers.resize(full_layers, false);
            lanes.push(PagedKvOwnedLane {
                identity: Rc::clone(&table.identity),
                old_len: table.logical_len,
                new_len,
                new_tokens: request.new_tokens,
                old_block_count: table.blocks.len(),
                growth_blocks,
                reserved_blocks,
                provisional_blocks,
                cow,
                written_layers,
            });
        }
        Ok((lanes, total_reserved))
    }

    /// Reserve one epoch and every new block for an ordered lane batch.
    ///
    /// All table snapshots and host allocations are validated first. In
    /// particular, configured exhaustion returns before allocating a slab,
    /// publishing an epoch, locking a table, or writing device storage.
    #[allow(clippy::too_many_lines)]
    pub fn reserve_batch(
        &self,
        requests: &[PagedKvGrowthRequest<'_>],
    ) -> PagedKvResult<PagedKvBatchReservation> {
        if requests.is_empty() {
            return Err(PagedKvPoolError::InvalidTransaction(
                "paged KV reservation batch cannot be empty".into(),
            ));
        }
        let (block_tokens, full_layers) = {
            let pool = self
                .inner
                .try_borrow()
                .map_err(|_| PagedKvPoolError::PoolBusy)?;
            if let Some(active) = pool.active_transaction {
                return Err(PagedKvPoolError::InvalidTransaction(format!(
                    "paged KV transaction {active} is still active"
                )));
            }
            (pool.config.block_tokens(), pool.geometry.full_layers())
        };
        let (mut lanes, total_reserved) =
            self.prepare_batch_lanes(requests, block_tokens, full_layers)?;
        let cow_count = lanes.iter().filter(|lane| lane.cow.is_some()).count();
        let mut superseded = Vec::new();
        superseded
            .try_reserve_exact(cow_count)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging superseded tail pages"))?;

        self.ensure_free_blocks(total_reserved)?;
        let (transaction, epoch, reserved) = {
            let mut pool = self
                .inner
                .try_borrow_mut()
                .map_err(|_| PagedKvPoolError::PoolBusy)?;
            let (transaction, epoch) = pool.next_transaction_and_epoch()?;
            let reserved = match pool.reserve_blocks(total_reserved, transaction) {
                Ok(reserved) => reserved,
                Err(error) => {
                    pool.rollback_reserved(&[], transaction);
                    return Err(error);
                }
            };
            (transaction, epoch, reserved)
        };
        let partition_result = (|| -> PagedKvResult<()> {
            let mut reserved_offset = 0_usize;
            for lane in &mut lanes {
                let end = reserved_offset
                    .checked_add(lane.reserved_blocks)
                    .ok_or_else(|| {
                        PagedKvPoolError::InvalidTransaction(
                            "lane reserved-block range overflows usize".into(),
                        )
                    })?;
                let blocks = reserved.get(reserved_offset..end).ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction(
                        "batch reserved-block partition is inconsistent".into(),
                    )
                })?;
                let growth_offset = if let Some(cow) = lane.cow {
                    let replacement = blocks.first().copied().ok_or_else(|| {
                        PagedKvPoolError::InvalidTransaction(
                            "copy-on-write lane has no replacement page".into(),
                        )
                    })?;
                    *lane
                        .provisional_blocks
                        .get_mut(cow.block_index)
                        .ok_or_else(|| {
                            PagedKvPoolError::InvalidTransaction(
                                "copy-on-write tail index is outside its provisional table".into(),
                            )
                        })? = replacement;
                    superseded.push(cow.source);
                    1
                } else {
                    0
                };
                let growth = &blocks[growth_offset..];
                if growth.len() != lane.growth_blocks {
                    return Err(PagedKvPoolError::InvalidTransaction(
                        "lane growth-page partition is inconsistent".into(),
                    ));
                }
                lane.provisional_blocks.extend_from_slice(growth);
                reserved_offset = end;
            }
            if reserved_offset != reserved.len() {
                return Err(PagedKvPoolError::InvalidTransaction(
                    "batch left unassigned reserved blocks".into(),
                ));
            }
            Ok(())
        })();
        if let Err(error) = partition_result {
            if let Ok(mut pool) = self.inner.try_borrow_mut() {
                pool.rollback_reserved(&reserved, transaction);
            }
            return Err(error);
        }
        for lane in &lanes {
            lane.identity.in_flight.set(Some(transaction));
        }
        Ok(PagedKvBatchReservation {
            pool: self.clone(),
            transaction,
            epoch,
            lanes,
            reserved,
            superseded,
            attention_batch: None,
            finalized: false,
        })
    }
}

/// Owned reservation for all lanes of one future model submission.
///
/// This token has no borrow lifetime, so scheduler and decoder code may carry
/// it independently of mutable session borrows. Drop rolls back every new
/// block and unlocks every table unless one ordered batch commit succeeds.
pub struct PagedKvBatchReservation {
    pool: SharedPagedKvPool,
    transaction: u64,
    epoch: PagedKvEpoch,
    lanes: Vec<PagedKvOwnedLane>,
    reserved: Vec<PagedKvBlockId>,
    superseded: Vec<PagedKvBlockId>,
    attention_batch: Option<PagedKvAttentionBatch>,
    finalized: bool,
}

impl fmt::Debug for PagedKvBatchReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PagedKvBatchReservation")
            .field("transaction", &self.transaction)
            .field("epoch", &self.epoch)
            .field("lanes", &self.lanes)
            .field("reserved", &self.reserved)
            .field("superseded", &self.superseded)
            .field("has_attention_batch", &self.attention_batch.is_some())
            .finish_non_exhaustive()
    }
}

impl PagedKvBatchReservation {
    #[must_use]
    pub fn epoch(&self) -> PagedKvEpoch {
        self.epoch
    }

    #[must_use]
    pub fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    pub fn lane_old_len(&self, lane: usize) -> PagedKvResult<usize> {
        self.lanes
            .get(lane)
            .map(|state| state.old_len)
            .ok_or_else(|| invalid_paged_lane(lane, self.lanes.len()))
    }

    pub fn lane_new_len(&self, lane: usize) -> PagedKvResult<usize> {
        self.lanes
            .get(lane)
            .map(|state| state.new_len)
            .ok_or_else(|| invalid_paged_lane(lane, self.lanes.len()))
    }

    fn block_id(&self, lane: usize, index: usize) -> PagedKvResult<PagedKvBlockId> {
        let state = self
            .lanes
            .get(lane)
            .ok_or_else(|| invalid_paged_lane(lane, self.lanes.len()))?;
        state.provisional_blocks.get(index).copied().ok_or_else(|| {
            PagedKvPoolError::InvalidTransaction(format!(
                "lane {lane} provisional block index {index} is unavailable"
            ))
        })
    }

    fn validate_table_snapshot(&self, lane: usize, table: &PagedKvBlockTable) -> PagedKvResult<()> {
        let state = self
            .lanes
            .get(lane)
            .ok_or_else(|| invalid_paged_lane(lane, self.lanes.len()))?;
        let snapshot_matches = table.blocks.iter().enumerate().all(|(index, &id)| {
            state.cow.filter(|cow| cow.block_index == index).map_or(
                state.provisional_blocks.get(index).copied() == Some(id),
                |cow| cow.source == id,
            )
        });
        if !Rc::ptr_eq(&self.pool.inner, &table.pool.inner)
            || !Rc::ptr_eq(&state.identity, &table.identity)
            || !table.identity.alive.get()
            || table.identity.in_flight.get() != Some(self.transaction)
            || table.logical_len != state.old_len
            || table.blocks.len() != state.old_block_count
            || state.provisional_blocks.len() < state.old_block_count
            || !snapshot_matches
        {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "lane {lane} table identity or snapshot changed after reservation"
            )));
        }
        Ok(())
    }

    /// Write one full layer for one lane without borrowing its block table.
    pub fn write_layer(
        &mut self,
        lane: usize,
        layer: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> PagedKvResult<()> {
        let geometry = self
            .pool
            .inner
            .try_borrow()
            .map_err(|_| PagedKvPoolError::PoolBusy)?
            .geometry
            .clone();
        let lane_count = self.lanes.len();
        let state = self
            .lanes
            .get(lane)
            .ok_or_else(|| invalid_paged_lane(lane, lane_count))?;
        if !state.identity.alive.get() || state.identity.in_flight.get() != Some(self.transaction) {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "lane {lane} table is no longer live or reserved"
            )));
        }
        if layer >= geometry.full_layers() {
            return Err(PagedKvPoolError::InvalidGeometry(format!(
                "full layer {layer} is outside {} layers",
                geometry.full_layers()
            )));
        }
        if state.written_layers[layer] {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "lane {lane} full layer {layer} was already written"
            )));
        }
        validate_paged_input(key, value, state.new_tokens, &geometry)?;

        if let Some(cow) = state.cow {
            let replacement = self.block_id(lane, cow.block_index)?;
            copy_paged_prefix(
                &self.pool,
                self.transaction,
                self.epoch,
                layer,
                cow,
                replacement,
            )?;
        }

        let block_tokens = self
            .pool
            .inner
            .try_borrow()
            .map_err(|_| PagedKvPoolError::PoolBusy)?
            .config
            .block_tokens();
        let old_len = state.old_len;
        let new_tokens = state.new_tokens;
        let write_count = div_ceil_checked(
            new_tokens
                .checked_add(old_len % block_tokens)
                .ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction("write span overflows usize".into())
                })?,
            block_tokens,
        )?;
        let mut writes = Vec::new();
        writes
            .try_reserve_exact(write_count)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging paged K/V writes"))?;
        let mut absolute = old_len;
        let mut source_offset = 0_usize;
        while source_offset < new_tokens {
            let block_index = absolute / block_tokens;
            let target_offset = absolute % block_tokens;
            let len = (block_tokens - target_offset).min(new_tokens - source_offset);
            let id = self.block_id(lane, block_index)?;
            let (target_key, target_value) = self
                .pool
                .inner
                .try_borrow()
                .map_err(|_| PagedKvPoolError::PoolBusy)?
                .layer_views(id, layer, Some(self.transaction))?;
            writes.push(PagedKvWrite {
                id,
                source_offset,
                target_offset,
                len,
                target_key,
                target_value,
            });
            source_offset += len;
            absolute = absolute.checked_add(len).ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction("write position overflows usize".into())
            })?;
        }

        apply_paged_writes(&self.pool, self.transaction, self.epoch, key, value, writes)?;
        self.lanes[lane].written_layers[layer] = true;
        Ok(())
    }

    /// Return an immutable page-table view for one written full-attention
    /// layer. The first call snapshots all lanes and uploads one packed page
    /// table; subsequent layers reuse it without repeating host mapping work.
    #[cfg_attr(not(test), allow(dead_code))]
    #[doc(hidden)]
    pub fn layer_read(&mut self, lane: usize, layer: usize) -> PagedKvResult<PagedKvLayerRead<'_>> {
        let state = self
            .lanes
            .get(lane)
            .ok_or_else(|| invalid_paged_lane(lane, self.lanes.len()))?;
        if !state.identity.alive.get()
            || state.identity.in_flight.get() != Some(self.transaction)
            || layer >= state.written_layers.len()
            || !state.written_layers[layer]
        {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "lane {lane} full layer {layer} is unavailable for indirect attention"
            )));
        }
        if self.attention_batch.is_none() {
            self.attention_batch = Some(self.build_attention_batch()?);
        }
        Ok(PagedKvLayerRead {
            batch: self
                .attention_batch
                .as_ref()
                .expect("attention batch was initialized"),
            lane,
            layer,
        })
    }

    /// Return one immutable packed page-table view after every lane has
    /// written the requested full-attention layer.
    ///
    /// The first successful lane or batch read builds the compact snapshot;
    /// every later layer view borrows that same page table and slab set.
    #[doc(hidden)]
    pub fn layer_batch_read(&mut self, layer: usize) -> PagedKvResult<PagedKvLayerBatchRead<'_>> {
        for (lane, state) in self.lanes.iter().enumerate() {
            if !state.identity.alive.get()
                || state.identity.in_flight.get() != Some(self.transaction)
                || layer >= state.written_layers.len()
                || !state.written_layers[layer]
            {
                return Err(PagedKvPoolError::InvalidTransaction(format!(
                    "lane {lane} full layer {layer} is unavailable for packed indirect attention"
                )));
            }
        }
        if self.attention_batch.is_none() {
            self.attention_batch = Some(self.build_attention_batch()?);
        }
        Ok(PagedKvLayerBatchRead {
            batch: self
                .attention_batch
                .as_ref()
                .expect("attention batch was initialized"),
            layer,
        })
    }

    fn build_attention_batch(&self) -> PagedKvResult<PagedKvAttentionBatch> {
        let (block_tokens, geometry, allocated_slabs) = {
            let pool = self
                .pool
                .inner
                .try_borrow()
                .map_err(|_| PagedKvPoolError::PoolBusy)?;
            (
                pool.config.block_tokens(),
                pool.geometry.clone(),
                pool.slabs.len(),
            )
        };
        let total_pages = self.lanes.iter().try_fold(0_usize, |total, lane| {
            let page_count = div_ceil_checked(lane.new_len, block_tokens)?;
            if page_count != lane.provisional_blocks.len() {
                return Err(PagedKvPoolError::InvalidTransaction(
                    "provisional block table length disagrees with logical length".into(),
                ));
            }
            total.checked_add(page_count).ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction(
                    "packed page-table length overflows usize".into(),
                )
            })
        })?;

        let mut ids = Vec::new();
        ids.try_reserve_exact(total_pages)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging indirect block IDs"))?;
        let mut lane_page_offsets = Vec::new();
        let mut lane_page_counts = Vec::new();
        let mut lane_logical_lens = Vec::new();
        for (target, context) in [
            (&mut lane_page_offsets, "staging lane page offsets"),
            (&mut lane_page_counts, "staging lane page counts"),
            (&mut lane_logical_lens, "staging lane logical lengths"),
        ] {
            target
                .try_reserve_exact(self.lanes.len())
                .map_err(|_| PagedKvPoolError::HostAllocation(context))?;
        }
        for lane in &self.lanes {
            if !lane.identity.alive.get() || lane.identity.in_flight.get() != Some(self.transaction)
            {
                return Err(PagedKvPoolError::InvalidTransaction(
                    "cannot snapshot a lane that is no longer live and reserved".into(),
                ));
            }
            lane_page_offsets.push(ids.len());
            lane_page_counts.push(lane.provisional_blocks.len());
            lane_logical_lens.push(lane.new_len);
            ids.extend_from_slice(&lane.provisional_blocks);
        }

        let (slabs, flat_page_table) = self.compact_attention_pages(&ids, allocated_slabs)?;
        let page_table = Tensor::from_vec(flat_page_table, (total_pages, 2), geometry.device())?;

        // Indirect argument-buffer reads are invisible to Candle's tensor
        // graph. Publish every referenced page use before the custom kernel can
        // be enqueued, so rollback/clear still quarantines them through epoch.
        self.pool
            .inner
            .try_borrow_mut()
            .map_err(|_| PagedKvPoolError::PoolBusy)?
            .mark_use_attempt(&ids, self.transaction, self.epoch)?;

        Ok(PagedKvAttentionBatch {
            slabs,
            page_table,
            lane_page_offsets,
            lane_page_counts,
            lane_logical_lens,
            block_tokens,
            full_layers: geometry.full_layers(),
            kv_heads: geometry.kv_heads(),
            head_dim: geometry.head_dim(),
            dtype: geometry.dtype(),
        })
    }

    fn compact_attention_pages(
        &self,
        ids: &[PagedKvBlockId],
        allocated_slabs: usize,
    ) -> PagedKvResult<(Vec<PagedKvSlabSnapshot>, Vec<u32>)> {
        let mut slab_remap = Vec::<Option<u32>>::new();
        let mut slabs = Vec::<PagedKvSlabSnapshot>::new();
        let mut flat_page_table = Vec::<u32>::new();
        slab_remap
            .try_reserve_exact(allocated_slabs)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging physical slab remap"))?;
        slab_remap.resize(allocated_slabs, None);
        slabs
            .try_reserve_exact(ids.len().min(allocated_slabs))
            .map_err(|_| PagedKvPoolError::HostAllocation("staging slab tensor handles"))?;
        flat_page_table
            .try_reserve_exact(ids.len().checked_mul(2).ok_or_else(|| {
                PagedKvPoolError::InvalidTransaction("page-table width overflows usize".into())
            })?)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging packed page table"))?;

        let pool = self
            .pool
            .inner
            .try_borrow()
            .map_err(|_| PagedKvPoolError::PoolBusy)?;
        for &id in ids {
            pool.validate_owned_block(id, Some(self.transaction))?;
            let slot = pool.slot(id)?;
            let compact_slab = if let Some(index) = slab_remap.get(slot.slab).copied().flatten() {
                index
            } else {
                let slab = pool.slabs.get(slot.slab).ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction(format!(
                        "block {} references missing slab {}",
                        id.slot, slot.slab
                    ))
                })?;
                let index = u32::try_from(slabs.len()).map_err(|_| {
                    PagedKvPoolError::InvalidConfig(
                        "compact slab index exceeds the device page-table range".into(),
                    )
                })?;
                *slab_remap.get_mut(slot.slab).ok_or_else(|| {
                    PagedKvPoolError::InvalidTransaction(format!(
                        "block {} references unmapped slab {}",
                        id.slot, slot.slab
                    ))
                })? = Some(index);
                slabs.push(PagedKvSlabSnapshot {
                    key: slab.key.clone(),
                    value: slab.value.clone(),
                });
                index
            };
            flat_page_table.push(compact_slab);
            flat_page_table.push(u32::try_from(slot.slab_offset).map_err(|_| {
                PagedKvPoolError::InvalidConfig(
                    "slab block offset exceeds the device page-table range".into(),
                )
            })?);
        }
        Ok((slabs, flat_page_table))
    }

    /// Materialize one lane/layer's provisional history in page order.
    ///
    /// A history that fits in one block is handed over as the narrowed views
    /// of that block, without a copy; those views are views into the pool's
    /// slabs. A longer history is a new tensor joined by `Tensor::cat`, and
    /// that join may be strided: a partly filled last block in a pool with
    /// more than one layer has no outer stride for the token axis, so candle
    /// concatenates it through its transpose path. Callers that need a
    /// contiguous K/V history make it contiguous themselves.
    ///
    /// A one-block view aliases the slab, so a later write to the same block
    /// changes what the view reads; callers use it within the step that
    /// materialized it.
    pub fn materialize_layer(&self, lane: usize, layer: usize) -> PagedKvResult<(Tensor, Tensor)> {
        #[cfg(any(test, feature = "test-exports"))]
        PAGED_BATCH_MATERIALIZATIONS.set(PAGED_BATCH_MATERIALIZATIONS.get().saturating_add(1));
        let state = self
            .lanes
            .get(lane)
            .ok_or_else(|| invalid_paged_lane(lane, self.lanes.len()))?;
        if !state.identity.alive.get()
            || state.identity.in_flight.get() != Some(self.transaction)
            || layer >= state.written_layers.len()
            || !state.written_layers[layer]
        {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "lane {lane} full layer {layer} is unavailable for materialization"
            )));
        }
        let block_tokens = self
            .pool
            .inner
            .try_borrow()
            .map_err(|_| PagedKvPoolError::PoolBusy)?
            .config
            .block_tokens();
        let block_count = div_ceil_checked(state.new_len, block_tokens)?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        keys.try_reserve_exact(block_count)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging materialized K pages"))?;
        values
            .try_reserve_exact(block_count)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging materialized V pages"))?;
        let mut ids = Vec::new();
        ids.try_reserve_exact(block_count)
            .map_err(|_| PagedKvPoolError::HostAllocation("staging materialized block IDs"))?;
        for index in 0..block_count {
            ids.push(self.block_id(lane, index)?);
        }
        // Tensor::cat reads all historical pages. Mark their use before any
        // asynchronous materialization work can be enqueued so a failed
        // forward cannot clear and immediately recycle a still-read page.
        self.pool
            .inner
            .try_borrow_mut()
            .map_err(|_| PagedKvPoolError::PoolBusy)?
            .mark_use_attempt(&ids, self.transaction, self.epoch)?;
        let mut remaining = state.new_len;
        for id in ids {
            let (key, value) = self
                .pool
                .inner
                .try_borrow()
                .map_err(|_| PagedKvPoolError::PoolBusy)?
                .layer_views(id, layer, Some(self.transaction))?;
            let len = remaining.min(block_tokens);
            keys.push(key.narrow(2, 0, len)?);
            values.push(value.narrow(2, 0, len)?);
            remaining -= len;
        }
        if keys.len() == 1 {
            // The whole history fits in one block, so the narrowed block
            // views already are the history: they are handed over as they
            // are, without a copy.
            return match (keys.pop(), values.pop()) {
                (Some(key), Some(value)) => Ok((key, value)),
                _ => Err(PagedKvPoolError::InvalidTransaction(
                    "the one-block history is missing its K/V views".into(),
                )),
            };
        }
        let key_refs = keys.iter().collect::<Vec<_>>();
        let value_refs = values.iter().collect::<Vec<_>>();
        // `Tensor::cat` joins the blocks. The joined history is strided for
        // that pool with candle 0.11 whenever a block view has no outer
        // stride for the token axis, so callers that need a contiguous K/V
        // history copy it themselves.
        Ok((Tensor::cat(&key_refs, 2)?, Tensor::cat(&value_refs, 2)?))
    }

    /// Atomically publish all lanes after exact identity, snapshot, and order
    /// validation. All table Vec storage was allocated during reservation.
    pub fn commit(mut self, tables: &mut [&mut PagedKvBlockTable]) -> PagedKvResult<PagedKvEpoch> {
        if tables.len() != self.lanes.len() {
            return Err(PagedKvPoolError::InvalidTransaction(format!(
                "commit received {} tables for {} reserved lanes",
                tables.len(),
                self.lanes.len()
            )));
        }
        for (lane, table) in tables.iter().enumerate() {
            self.validate_table_snapshot(lane, table)?;
            if self.lanes[lane]
                .written_layers
                .iter()
                .any(|written| !written)
            {
                return Err(PagedKvPoolError::InvalidTransaction(format!(
                    "lane {lane} cannot commit before every full layer was written"
                )));
            }
        }
        self.pool
            .inner
            .try_borrow_mut()
            .map_err(|_| PagedKvPoolError::PoolBusy)?
            .commit_reserved_replacing(&self.reserved, &self.superseded, self.transaction)?;
        for (table, lane) in tables.iter_mut().zip(&mut self.lanes) {
            table.blocks = std::mem::take(&mut lane.provisional_blocks);
            table.logical_len = lane.new_len;
            table.identity.in_flight.set(None);
        }
        self.finalized = true;
        Ok(self.epoch)
    }

    pub fn rollback(mut self) {
        self.rollback_inner();
    }

    fn rollback_inner(&mut self) {
        if self.finalized {
            return;
        }
        if let Ok(mut pool) = self.pool.inner.try_borrow_mut() {
            pool.rollback_reserved(&self.reserved, self.transaction);
        }
        for lane in &self.lanes {
            if lane.identity.in_flight.get() == Some(self.transaction) {
                lane.identity.in_flight.set(None);
            }
        }
        self.finalized = true;
    }
}

impl Drop for PagedKvBatchReservation {
    fn drop(&mut self) {
        self.rollback_inner();
    }
}

fn checked_paged_append_len(
    old_len: usize,
    new_tokens: usize,
    limit: usize,
) -> PagedKvResult<usize> {
    let requested = old_len
        .checked_add(new_tokens)
        .ok_or(PagedKvPoolError::ContextLimit {
            limit,
            requested: usize::MAX,
        })?;
    if requested > limit {
        return Err(PagedKvPoolError::ContextLimit { limit, requested });
    }
    Ok(requested)
}

fn invalid_paged_lane(lane: usize, lanes: usize) -> PagedKvPoolError {
    PagedKvPoolError::InvalidTransaction(format!(
        "paged KV lane {lane} is outside {lanes} reserved lanes"
    ))
}

struct PagedKvWrite {
    id: PagedKvBlockId,
    source_offset: usize,
    target_offset: usize,
    len: usize,
    target_key: Tensor,
    target_value: Tensor,
}

/// Copy the valid prefix of a shared partial tail into its transaction-owned
/// replacement before the append writes the suffix. Both the source read and
/// replacement write are published to epoch quarantine first.
fn copy_paged_prefix(
    pool: &SharedPagedKvPool,
    transaction: u64,
    epoch: PagedKvEpoch,
    layer: usize,
    cow: PagedKvCowPlan,
    replacement: PagedKvBlockId,
) -> PagedKvResult<()> {
    if replacement == cow.source {
        return Err(PagedKvPoolError::InvalidTransaction(
            "copy-on-write replacement aliases its shared source".into(),
        ));
    }
    let (source_key, source_value, target_key, target_value) = {
        let pool = pool
            .inner
            .try_borrow()
            .map_err(|_| PagedKvPoolError::PoolBusy)?;
        if cow.valid_tokens == 0 || cow.valid_tokens >= pool.config.block_tokens() {
            return Err(PagedKvPoolError::InvalidTransaction(
                "copy-on-write tail must contain a strict partial page".into(),
            ));
        }
        let (source_key, source_value) = pool.layer_views(cow.source, layer, None)?;
        let (target_key, target_value) = pool.layer_views(replacement, layer, Some(transaction))?;
        (source_key, source_value, target_key, target_value)
    };
    pool.inner
        .try_borrow_mut()
        .map_err(|_| PagedKvPoolError::PoolBusy)?
        .mark_use_attempt(&[cow.source, replacement], transaction, epoch)?;

    let detach_prefix = |source: &Tensor| -> PagedKvResult<Tensor> {
        let source = source.narrow(2, 0, cow.valid_tokens)?.contiguous()?;
        // Physical pages are views into a common slab. Candle rejects an
        // in-place copy whenever source and destination share that backing
        // storage, even when their page ranges do not overlap. Stage through
        // an independently allocated tensor so this is safe on CPU and Metal.
        let detached = Tensor::zeros(source.shape(), source.dtype(), source.device())?;
        detached.slice_set(&source, 2, 0)?;
        Ok(detached)
    };
    let source_key = detach_prefix(&source_key)?;
    let source_value = detach_prefix(&source_value)?;
    target_key.slice_set(&source_key, 2, 0)?;
    target_value.slice_set(&source_value, 2, 0)?;
    Ok(())
}

fn apply_paged_writes(
    pool: &SharedPagedKvPool,
    transaction: u64,
    epoch: PagedKvEpoch,
    key: &Tensor,
    value: &Tensor,
    writes: Vec<PagedKvWrite>,
) -> PagedKvResult<()> {
    let mut touched = Vec::new();
    touched
        .try_reserve_exact(writes.len())
        .map_err(|_| PagedKvPoolError::HostAllocation("staging touched block IDs"))?;
    for write in &writes {
        if !touched.contains(&write.id) {
            touched.push(write.id);
        }
    }
    pool.inner
        .try_borrow_mut()
        .map_err(|_| PagedKvPoolError::PoolBusy)?
        .mark_use_attempt(&touched, transaction, epoch)?;
    for write in writes {
        let source_key = key
            .narrow(2, write.source_offset, write.len)?
            .contiguous()?;
        let source_value = value
            .narrow(2, write.source_offset, write.len)?
            .contiguous()?;
        write
            .target_key
            .slice_set(&source_key, 2, write.target_offset)?;
        write
            .target_value
            .slice_set(&source_value, 2, write.target_offset)?;
    }
    Ok(())
}

fn validate_paged_input(
    key: &Tensor,
    value: &Tensor,
    new_tokens: usize,
    geometry: &PagedKvGeometry,
) -> PagedKvResult<()> {
    if key.dims() != value.dims() {
        return Err(PagedKvPoolError::InvalidGeometry(format!(
            "K/V shape mismatch: {:?} vs {:?}",
            key.shape(),
            value.shape()
        )));
    }
    let expected = (1, geometry.kv_heads(), new_tokens, geometry.head_dim());
    if key.dims4()? != expected {
        return Err(PagedKvPoolError::InvalidGeometry(format!(
            "K/V append shape {:?} does not match expected {expected:?}",
            key.shape()
        )));
    }
    if key.dtype() != geometry.dtype() || value.dtype() != geometry.dtype() {
        return Err(PagedKvPoolError::InvalidGeometry(format!(
            "K/V dtype {:?}/{:?} does not match pool {:?}",
            key.dtype(),
            value.dtype(),
            geometry.dtype()
        )));
    }
    if key.device().location() != geometry.device().location()
        || value.device().location() != geometry.device().location()
    {
        return Err(PagedKvPoolError::InvalidGeometry(
            "K/V device does not match the pool device".into(),
        ));
    }
    Ok(())
}

fn div_ceil_checked(value: usize, divisor: usize) -> PagedKvResult<usize> {
    value
        .checked_add(divisor - 1)
        .and_then(|sum| sum.checked_div(divisor))
        .ok_or_else(|| PagedKvPoolError::InvalidConfig("paged block count overflows usize".into()))
}

/// Grow-on-demand per-layer cache. Keys are stored after `RoPE` and are never
/// expanded from KV heads to query heads in the cached representation.
#[derive(Debug)]
pub struct KvCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
    len: usize,
    allocated_capacity: usize,
    limit: usize,
}

/// Immutable K/V view over one contiguous run of absolute token positions.
///
/// A wrapped sliding cache exposes two segments in oldest-to-newest order.
/// Keeping the absolute start beside each view prevents physical ring offsets
/// from leaking into attention or `RoPE` semantics.
#[derive(Debug, Clone)]
pub struct KvSegment {
    pub key: Tensor,
    pub value: Tensor,
    pub absolute_start: usize,
}

impl KvSegment {
    #[must_use]
    pub fn len(&self) -> usize {
        self.key.dims().get(2).copied().unwrap_or(0)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Physical retention policy selected for a decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCachePolicy {
    /// Retain every absolute position in a grow-on-demand contiguous tensor.
    Full,
    /// Retain full-attention history through a model-shared physical page pool.
    Paged,
    /// Retain only the bounded local-attention history in a transactional ring.
    Sliding,
}

/// Observable cache geometry used by tests and runtime diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheStats {
    /// Physical retention policy used by this layer.
    pub policy: KvCachePolicy,
    /// Number of tokens processed since the last clear. This remains absolute
    /// even when old sliding-window entries have been evicted physically.
    pub logical_len: usize,
    /// Absolute position of the oldest physically resident token.
    pub resident_start: usize,
    /// Number of physically resident tokens.
    pub resident_len: usize,
    /// Number of token slots currently allocated for each of K and V.
    pub allocated_capacity: usize,
    /// Bytes currently allocated by the persistent K and V tensors together.
    pub allocated_bytes: usize,
    /// Maximum number of token slots the physical representation may allocate
    /// for each of K and V.
    pub physical_capacity: usize,
    /// Maximum absolute sequence length accepted by the model.
    pub logical_limit: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct SlidingCacheCheckpoint {
    absolute_len: usize,
    resident_len: usize,
    write_cursor: usize,
}

/// Per-layer cache policy selected by the format-neutral decoder spec.
#[derive(Debug)]
pub enum LayerKvCache {
    Full(KvCache),
    Sliding(SlidingKvCache),
}

impl LayerKvCache {
    #[must_use]
    pub fn logical_len(&self) -> usize {
        match self {
            Self::Full(cache) => cache.len(),
            Self::Sliding(cache) => cache.logical_len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.logical_len() == 0
    }

    pub fn clear(&mut self) {
        match self {
            Self::Full(cache) => cache.clear(),
            Self::Sliding(cache) => cache.clear(),
        }
    }

    #[must_use]
    pub fn stats(&self) -> KvCacheStats {
        match self {
            Self::Full(cache) => cache.stats(),
            Self::Sliding(cache) => cache.stats(),
        }
    }
}

impl KvCache {
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            key: None,
            value: None,
            len: 0,
            allocated_capacity: 0,
            limit,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.limit
    }

    #[must_use]
    pub fn allocated_capacity(&self) -> usize {
        self.allocated_capacity
    }

    #[must_use]
    pub fn stats(&self) -> KvCacheStats {
        KvCacheStats {
            policy: KvCachePolicy::Full,
            logical_len: self.len,
            resident_start: 0,
            resident_len: self.len,
            allocated_capacity: self.allocated_capacity,
            allocated_bytes: tensor_bytes(self.key.as_ref()) + tensor_bytes(self.value.as_ref()),
            physical_capacity: self.limit,
            logical_limit: self.limit,
        }
    }

    /// Reset logical length while retaining allocated device buffers for reuse.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Roll a partially completed model step back to its previous logical length.
    pub fn truncate(&mut self, len: usize) {
        debug_assert!(len <= self.len);
        self.len = self.len.min(len);
    }

    /// Append `[batch, kv_heads, sequence, head_dim]` tensors along sequence.
    pub fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        if key.dims() != value.dims() {
            candle_core::bail!(
                "K/V cache shape mismatch: {:?} vs {:?}",
                key.shape(),
                value.shape()
            );
        }
        let new_tokens = key.dim(2)?;
        let required = self
            .len
            .checked_add(new_tokens)
            .ok_or_else(|| candle_core::Error::Msg("KV cache length overflow".into()))?;
        if required > self.limit {
            candle_core::bail!(
                "KV cache capacity {} exceeded by requested length {}",
                self.limit,
                required
            );
        }
        self.ensure_capacity(key, value, required)?;

        // slice_set writes only the new range; old K/V are not copied each step.
        self.key
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("K cache was not allocated".into()))?
            .slice_set(&key.contiguous()?, 2, self.len)?;
        self.value
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("V cache was not allocated".into()))?
            .slice_set(&value.contiguous()?, 2, self.len)?;
        self.len += new_tokens;

        let key = self
            .key
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("K cache was not allocated".into()))?
            .narrow(2, 0, self.len)?;
        let value = self
            .value
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("V cache was not allocated".into()))?
            .narrow(2, 0, self.len)?;
        Ok((key, value))
    }

    fn ensure_capacity(&mut self, key: &Tensor, value: &Tensor, required: usize) -> Result<()> {
        if required <= self.allocated_capacity {
            return Ok(());
        }

        let new_capacity = required
            .max(MIN_ALLOCATION_TOKENS)
            .next_power_of_two()
            .min(self.limit);
        let mut shape = key.dims().to_vec();
        shape[2] = new_capacity;
        let new_key = Tensor::zeros(shape.clone(), key.dtype(), key.device())?;
        let new_value = Tensor::zeros(shape, value.dtype(), value.device())?;

        if self.len > 0 {
            let old_key = self
                .key
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("K cache storage is missing".into()))?
                .narrow(2, 0, self.len)?
                .contiguous()?;
            let old_value = self
                .value
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("V cache storage is missing".into()))?
                .narrow(2, 0, self.len)?
                .contiguous()?;
            new_key.slice_set(&old_key, 2, 0)?;
            new_value.slice_set(&old_value, 2, 0)?;
        }

        self.key = Some(new_key);
        self.value = Some(new_value);
        self.allocated_capacity = new_capacity;
        Ok(())
    }
}

/// Bounded single-sequence cache for local/sliding attention.
///
/// A window of `W` needs at most `W - 1` persistent tokens because the current
/// forward's K/V tensors remain available until attention completes. The ring
/// owns `2 * (W - 1)` slots: the live history occupies at most one half, so a
/// complete replacement can be written into the other half without touching a
/// checkpoint's live data. Rollback therefore restores three integers instead
/// of copying K/V tensors.
#[derive(Debug)]
pub struct SlidingKvCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
    absolute_len: usize,
    resident_len: usize,
    write_cursor: usize,
    history_limit: usize,
    storage_capacity: usize,
    logical_limit: usize,
}

impl SlidingKvCache {
    pub fn new(window: usize, logical_limit: usize) -> Result<Self> {
        if window == 0 {
            candle_core::bail!("sliding KV cache window must be positive");
        }
        // Positions beyond the model's logical limit can never become live.
        let history_limit = (window - 1).min(logical_limit.saturating_sub(1));
        let storage_capacity = history_limit
            .checked_mul(2)
            .ok_or_else(|| candle_core::Error::Msg("sliding KV capacity overflow".into()))?;
        Ok(Self {
            key: None,
            value: None,
            absolute_len: 0,
            resident_len: 0,
            write_cursor: 0,
            history_limit,
            storage_capacity,
            logical_limit,
        })
    }

    #[must_use]
    pub fn logical_len(&self) -> usize {
        self.absolute_len
    }

    #[must_use]
    pub fn history_limit(&self) -> usize {
        self.history_limit
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.absolute_len == 0
    }

    #[must_use]
    pub fn stats(&self) -> KvCacheStats {
        KvCacheStats {
            policy: KvCachePolicy::Sliding,
            logical_len: self.absolute_len,
            resident_start: self.absolute_len - self.resident_len,
            resident_len: self.resident_len,
            allocated_capacity: self.key.as_ref().map_or(0, |_| self.storage_capacity),
            allocated_bytes: tensor_bytes(self.key.as_ref()) + tensor_bytes(self.value.as_ref()),
            physical_capacity: self.storage_capacity,
            logical_limit: self.logical_limit,
        }
    }

    /// Reset logical and ring metadata while retaining device allocation.
    pub fn clear(&mut self) {
        self.absolute_len = 0;
        self.resident_len = 0;
        self.write_cursor = 0;
    }

    #[must_use]
    #[doc(hidden)]
    pub fn checkpoint(&self) -> SlidingCacheCheckpoint {
        SlidingCacheCheckpoint {
            absolute_len: self.absolute_len,
            resident_len: self.resident_len,
            write_cursor: self.write_cursor,
        }
    }

    #[doc(hidden)]
    pub fn restore(&mut self, checkpoint: SlidingCacheCheckpoint) {
        let SlidingCacheCheckpoint {
            absolute_len,
            resident_len,
            write_cursor,
        } = checkpoint;
        debug_assert!(absolute_len <= self.absolute_len);
        debug_assert!(resident_len <= self.history_limit);
        self.absolute_len = absolute_len;
        self.resident_len = resident_len;
        self.write_cursor = write_cursor;
    }

    /// Return the live history in oldest-to-newest absolute order.
    pub fn history_segments(&self) -> Result<Vec<KvSegment>> {
        if self.resident_len == 0 {
            return Ok(Vec::new());
        }
        let key = self
            .key
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("sliding K cache storage is missing".into()))?;
        let value = self
            .value
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("sliding V cache storage is missing".into()))?;
        let physical_start =
            (self.write_cursor + self.storage_capacity - self.resident_len) % self.storage_capacity;
        let absolute_start = self.absolute_len - self.resident_len;
        let first_len = self
            .resident_len
            .min(self.storage_capacity - physical_start);
        let mut segments = vec![KvSegment {
            key: key.narrow(2, physical_start, first_len)?,
            value: value.narrow(2, physical_start, first_len)?,
            absolute_start,
        }];
        let second_len = self.resident_len - first_len;
        if second_len > 0 {
            segments.push(KvSegment {
                key: key.narrow(2, 0, second_len)?,
                value: value.narrow(2, 0, second_len)?,
                absolute_start: absolute_start + first_len,
            });
        }
        Ok(segments)
    }

    /// Advance the absolute sequence and retain only its newest history suffix.
    pub fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<()> {
        if key.dims() != value.dims() {
            candle_core::bail!(
                "K/V cache shape mismatch: {:?} vs {:?}",
                key.shape(),
                value.shape()
            );
        }
        let _ = key.dims4()?;
        let new_tokens = key.dim(2)?;
        let required = self
            .absolute_len
            .checked_add(new_tokens)
            .ok_or_else(|| candle_core::Error::Msg("KV cache length overflow".into()))?;
        if required > self.logical_limit {
            candle_core::bail!(
                "KV cache capacity {} exceeded by requested length {}",
                self.logical_limit,
                required
            );
        }
        if new_tokens == 0 {
            return Ok(());
        }
        if self.history_limit == 0 {
            self.absolute_len = required;
            return Ok(());
        }

        self.ensure_storage(key, value)?;
        let suffix_len = new_tokens.min(self.history_limit);
        let source_start = new_tokens - suffix_len;
        let first_len = suffix_len.min(self.storage_capacity - self.write_cursor);
        self.write_part(key, value, source_start, self.write_cursor, first_len)?;
        let second_len = suffix_len - first_len;
        if second_len > 0 {
            self.write_part(key, value, source_start + first_len, 0, second_len)?;
        }

        // Publish metadata only after every K/V write succeeds. The written
        // range is disjoint from the checkpoint's live history by construction.
        self.absolute_len = required;
        self.resident_len = self
            .history_limit
            .min(self.resident_len.saturating_add(new_tokens));
        self.write_cursor = (self.write_cursor + suffix_len) % self.storage_capacity;
        Ok(())
    }

    fn ensure_storage(&mut self, key: &Tensor, value: &Tensor) -> Result<()> {
        if self.key.is_some() {
            return Ok(());
        }
        let mut shape = key.dims().to_vec();
        shape[2] = self.storage_capacity;
        let key = Tensor::zeros(shape.clone(), key.dtype(), key.device())?;
        let value = Tensor::zeros(shape, value.dtype(), value.device())?;
        self.key = Some(key);
        self.value = Some(value);
        Ok(())
    }

    fn write_part(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        source_start: usize,
        target_start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let key = key.narrow(2, source_start, len)?.contiguous()?;
        let value = value.narrow(2, source_start, len)?.contiguous()?;
        self.key
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("sliding K cache was not allocated".into()))?
            .slice_set(&key, 2, target_start)?;
        self.value
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("sliding V cache was not allocated".into()))?
            .slice_set(&value, 2, target_start)?;
        Ok(())
    }
}

fn tensor_bytes(tensor: Option<&Tensor>) -> usize {
    tensor.map_or(0, |tensor| {
        tensor
            .elem_count()
            .saturating_mul(tensor.dtype().size_in_bytes())
    })
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Storage, Tensor};

    use super::*;

    fn token_values(start: usize, len: usize, device: &Device) -> Result<Tensor> {
        Tensor::from_vec(
            (start..start + len).map(|value| value as f32).collect(),
            (1, 1, len, 1),
            device,
        )
    }

    /// Flattens the key history of a sliding cache and reports its shape.
    ///
    /// The history is the concatenation of the cache's segments along the
    /// token axis, so its shape is the shape of one segment with the token
    /// count of every segment.
    fn sliding_key_history(cache: &SlidingKvCache) -> Result<(Vec<f32>, Vec<usize>)> {
        let segments = cache.history_segments()?;
        let mut values = Vec::new();
        let mut tokens = 0_usize;
        for segment in &segments {
            values.extend(segment.key.flatten_all()?.to_vec1::<f32>()?);
            tokens += segment.key.dim(2)?;
        }
        let mut dims = segments
            .first()
            .map_or_else(Vec::new, |segment| segment.key.dims().to_vec());
        if let Some(token_axis) = dims.get_mut(2) {
            *token_axis = tokens;
        }
        Ok((values, dims))
    }

    /// Compares two flat buffers read from tensors without printing them.
    ///
    /// The buffers are flat slices, and the shapes of the read and of the
    /// expected data are compared first, so a read of another shape with the
    /// same values fails. The report of a failure names the buffer, both
    /// shapes, both lengths, the number of differing elements and the first
    /// differing index with its two values. A failing assert never dumps a
    /// whole buffer.
    fn assert_values<T>(
        name: &str,
        actual_shape: &[usize],
        expected_shape: &[usize],
        actual: &[T],
        expected: &[T],
    ) where
        T: PartialEq + std::fmt::Display + Copy,
    {
        assert!(
            actual_shape == expected_shape,
            "{name}: shape mismatch (actual shape {actual_shape:?}, expected shape {expected_shape:?})"
        );
        let mut differing = 0_usize;
        let mut first = None;
        for index in 0..actual.len().max(expected.len()) {
            let actual_value = actual.get(index).copied();
            let expected_value = expected.get(index).copied();
            if actual_value != expected_value {
                differing += 1;
                if first.is_none() {
                    first = Some((index, actual_value, expected_value));
                }
            }
        }
        let Some((index, actual_value, expected_value)) = first else {
            return;
        };
        panic!(
            "{name} (actual shape {actual_shape:?}, expected shape {expected_shape:?}): \
             {differing} of {} elements differ (actual length {}, expected length {}); \
             first difference at index {index}: actual {}, expected {}",
            actual.len().max(expected.len()),
            actual.len(),
            expected.len(),
            actual_value.map_or_else(|| "missing".to_string(), |value| value.to_string()),
            expected_value.map_or_else(|| "missing".to_string(), |value| value.to_string()),
        );
    }

    /// Flattens a nested page table in row-major order.
    fn flatten_page_table(rows: &[Vec<u32>]) -> Vec<u32> {
        rows.iter().flat_map(|row| row.iter().copied()).collect()
    }

    #[test]
    fn assert_values_accepts_equal_buffers() {
        assert_values(
            "the helper buffer",
            &[2],
            &[2],
            &[1.0_f32, 2.0],
            &[1.0, 2.0],
        );
    }

    #[test]
    #[should_panic(
        expected = "the helper buffer (actual shape [2], expected shape [2]): 1 of 2 elements differ"
    )]
    fn assert_values_reports_the_first_difference_without_the_buffers() {
        assert_values(
            "the helper buffer",
            &[2],
            &[2],
            &[1.0_f32, 2.0],
            &[1.0, 3.0],
        );
    }

    #[test]
    #[should_panic(expected = "the helper buffer: shape mismatch")]
    fn assert_values_rejects_equal_values_with_another_shape() {
        assert_values(
            "the helper buffer",
            &[1, 2],
            &[2, 1],
            &[1.0_f32, 2.0],
            &[1.0, 2.0],
        );
    }

    #[test]
    #[should_panic(expected = "actual length 2, expected length 1")]
    fn assert_values_reports_different_lengths() {
        assert_values("the helper buffer", &[2], &[2], &[1.0_f32, 2.0], &[1.0]);
    }

    fn paged_pool(
        full_layers: usize,
        kv_heads: usize,
        head_dim: usize,
        block_tokens: usize,
        slab_blocks: usize,
        max_blocks: usize,
    ) -> PagedKvResult<SharedPagedKvPool> {
        SharedPagedKvPool::new(
            PagedKvPoolConfig::new(block_tokens, slab_blocks, max_blocks)?,
            PagedKvGeometry::new(full_layers, kv_heads, head_dim, DType::F32, &Device::Cpu)?,
        )
    }

    /// Test-only independent oracle for the maintained O(1) occupancy.
    fn assert_paged_occupancy_matches_scan(pool: &SharedPagedKvPool) -> PagedKvResult<()> {
        let pool = pool
            .inner
            .try_borrow()
            .map_err(|_| PagedKvPoolError::PoolBusy)?;
        let scanned = pool.scan_occupancy();
        assert_eq!(pool.occupancy, scanned);

        let scanned_free = pool
            .slots
            .iter()
            .filter(|slot| slot.state == PagedKvBlockState::Free)
            .count();
        let scanned_bytes = pool.slabs.iter().try_fold(0_usize, |total, slab| {
            total
                .checked_add(tensor_bytes(Some(&slab.key)))?
                .checked_add(tensor_bytes(Some(&slab.value)))
        });
        let stats = pool.stats()?;
        assert_eq!(stats.free_blocks, scanned_free);
        assert_eq!(Some(stats.allocated_bytes), scanned_bytes);
        assert_eq!(
            stats.allocated_blocks,
            stats
                .free_blocks
                .checked_add(stats.reserved_blocks)
                .and_then(|blocks| blocks.checked_add(stats.live_blocks))
                .and_then(|blocks| blocks.checked_add(stats.retired_blocks))
                .expect("test pool block counts fit usize")
        );
        Ok(())
    }

    fn paged_values(
        base: usize,
        start: usize,
        len: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> PagedKvResult<Tensor> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(kv_heads * len * head_dim)
            .map_err(|_| PagedKvPoolError::HostAllocation("building paged test values"))?;
        for head in 0..kv_heads {
            for position in start..start + len {
                for dimension in 0..head_dim {
                    values.push((base + head * 1_000 + position * 10 + dimension) as f32);
                }
            }
        }
        Ok(Tensor::from_vec(
            values,
            (1, kv_heads, len, head_dim),
            &Device::Cpu,
        )?)
    }

    /// The address of a CPU tensor's storage, used to prove that a
    /// materialized view shares the slab allocation instead of copying it.
    fn cpu_storage_address(tensor: &Tensor) -> Option<usize> {
        let (storage, _) = tensor.storage_and_layout();
        match &*storage {
            Storage::Cpu(storage) => storage
                .as_slice::<f32>()
                .ok()
                .map(|slice| slice.as_ptr() as usize),
            _ => None,
        }
    }

    /// The largest absolute difference between two F32 tensors of one shape.
    fn paged_max_difference(left: &Tensor, right: &Tensor) -> Result<f32> {
        left.sub(right)?.abs()?.max_all()?.to_scalar::<f32>()
    }

    /// The paged test values of one base over a range of positions.
    fn paged_range(
        base: usize,
        positions: std::ops::Range<usize>,
        kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut values = Vec::new();
        for head in 0..kv_heads {
            for position in positions.clone() {
                for dimension in 0..head_dim {
                    values.push((base + head * 1_000 + position * 10 + dimension) as f32);
                }
            }
        }
        values
    }

    /// The paged test values of a six-token history that spans two blocks:
    /// the first three tokens carry `first_base`, the last three `second_base`.
    fn paged_two_block_expected(
        first_base: usize,
        second_base: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut values = Vec::new();
        for head in 0..kv_heads {
            for position in 0..6 {
                let base = if position < 3 {
                    first_base
                } else {
                    second_base
                };
                for dimension in 0..head_dim {
                    values.push((base + head * 1_000 + position * 10 + dimension) as f32);
                }
            }
        }
        values
    }

    fn paged_expected(base: usize, len: usize, kv_heads: usize, head_dim: usize) -> Vec<f32> {
        let mut values = Vec::new();
        for head in 0..kv_heads {
            for position in 0..len {
                for dimension in 0..head_dim {
                    values.push((base + head * 1_000 + position * 10 + dimension) as f32);
                }
            }
        }
        values
    }

    #[test]
    fn paged_pool_rejects_invalid_and_overflowing_geometry() {
        assert!(PagedKvPoolConfig::new(0, 1, 1).is_err());
        assert!(PagedKvPoolConfig::new(1, 0, 1).is_err());
        assert!(PagedKvPoolConfig::new(1, 1, 0).is_err());
        assert!(
            PagedKvPoolConfig::new(1, 1, u32::MAX as usize + 1)
                .unwrap_err()
                .to_string()
                .contains("block-ID range")
        );
        assert!(PagedKvGeometry::new(0, 1, 1, DType::F32, &Device::Cpu).is_err());
        assert!(PagedKvGeometry::new(1, 0, 1, DType::F32, &Device::Cpu).is_err());
        assert!(PagedKvGeometry::new(1, 1, 0, DType::F32, &Device::Cpu).is_err());
        assert!(PagedKvGeometry::new(1, 1, 1, DType::U32, &Device::Cpu).is_err());

        let config = PagedKvPoolConfig::new(2, 1, 2).unwrap();
        let geometry = PagedKvGeometry::new(usize::MAX, 2, 2, DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            SharedPagedKvPool::new(config, geometry),
            Err(PagedKvPoolError::InvalidGeometry(_))
        ));
    }

    #[test]
    fn paged_maintained_occupancy_matches_scan_across_refcount_cow_and_rollback()
    -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 2, 2, 4)?;
        let mut source = PagedKvBlockTable::new(pool.clone(), 8)?;
        assert_paged_occupancy_matches_scan(&pool)?;

        let prefix_epoch = {
            let mut append = source.begin_append(3)?;
            let stats = pool.stats()?;
            assert_eq!(stats.reserved_blocks, 2);
            assert_eq!(stats.live_blocks, 0);
            assert_paged_occupancy_matches_scan(&pool)?;
            let prefix = paged_values(100, 0, 3, 1, 1)?;
            append.write_layer(0, &prefix, &prefix)?;
            append.commit()?
        };
        assert_eq!(
            pool.stats()?.live_references,
            2,
            "the source owns two freshly committed pages"
        );
        assert_paged_occupancy_matches_scan(&pool)?;
        pool.mark_completed_through(prefix_epoch)?;
        assert_paged_occupancy_matches_scan(&pool)?;

        let mut first_fork = source.fork_prefix(3)?;
        let mut second_fork = source.fork_prefix(3)?;
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 2);
        assert_eq!(stats.live_references, 6);
        assert_eq!(stats.shared_blocks, 2);
        assert_paged_occupancy_matches_scan(&pool)?;

        first_fork.clear();
        let stats = pool.stats()?;
        assert_eq!(stats.live_references, 4);
        assert_eq!(stats.shared_blocks, 2);
        assert_paged_occupancy_matches_scan(&pool)?;

        let cow_epoch = {
            let mut append = second_fork.begin_append(1)?;
            let stats = pool.stats()?;
            assert_eq!(stats.reserved_blocks, 1);
            assert_eq!(stats.live_references, 4);
            assert_paged_occupancy_matches_scan(&pool)?;
            let suffix = paged_values(900, 3, 1, 1, 1)?;
            append.write_layer(0, &suffix, &suffix)?;
            append.commit()?
        };
        let stats = pool.stats()?;
        assert_eq!(stats.reserved_blocks, 0);
        assert_eq!(stats.live_blocks, 3);
        assert_eq!(stats.live_references, 4);
        assert_eq!(stats.shared_blocks, 1);
        assert_paged_occupancy_matches_scan(&pool)?;

        let mut rolled_back_fork = source.fork_prefix(3)?;
        let failed_epoch = {
            let mut append = rolled_back_fork.begin_append(1)?;
            let suffix = paged_values(1_900, 3, 1, 1, 1)?;
            append.write_layer(0, &suffix, &suffix)?;
            let epoch = append.epoch();
            append.rollback();
            epoch
        };
        let stats = pool.stats()?;
        assert_eq!(stats.reserved_blocks, 0);
        assert_eq!(stats.live_blocks, 3);
        assert_eq!(stats.live_references, 6);
        assert_eq!(stats.shared_blocks, 2);
        assert_eq!(stats.retired_blocks, 1);
        assert_paged_occupancy_matches_scan(&pool)?;

        pool.mark_completed_through(failed_epoch)?;
        assert_eq!(pool.stats()?.retired_blocks, 0);
        assert_paged_occupancy_matches_scan(&pool)?;

        source.clear();
        second_fork.clear();
        rolled_back_fork.clear();
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 0);
        assert_eq!(stats.live_references, 0);
        assert_eq!(stats.shared_blocks, 0);
        assert_eq!(stats.free_blocks, stats.allocated_blocks);
        assert!(stats.completed_epoch >= cow_epoch);
        assert_paged_occupancy_matches_scan(&pool)?;
        Ok(())
    }

    #[test]
    fn paged_append_materializes_cross_layer_history_across_slabs() -> PagedKvResult<()> {
        const LAYERS: usize = 2;
        const HEADS: usize = 2;
        const DIM: usize = 2;
        let pool = paged_pool(LAYERS, HEADS, DIM, 2, 2, 4)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 8)?;

        let first_epoch = {
            let mut append = table.begin_append(3)?;
            for layer in 0..LAYERS {
                let key = paged_values(layer * 10_000, 0, 3, HEADS, DIM)?;
                let value = paged_values(50_000 + layer * 10_000, 0, 3, HEADS, DIM)?;
                append.write_layer(layer, &key, &value)?;
                let (actual_key, actual_value) = append.materialize_layer(layer)?;
                assert_values(
                    &format!("the materialized key history of layer {layer}"),
                    actual_key.dims(),
                    &[1, HEADS, 3, DIM],
                    &actual_key.flatten_all()?.to_vec1::<f32>()?,
                    &paged_expected(layer * 10_000, 3, HEADS, DIM),
                );
                assert_values(
                    &format!("the materialized value history of layer {layer}"),
                    actual_value.dims(),
                    &[1, HEADS, 3, DIM],
                    &actual_value.flatten_all()?.to_vec1::<f32>()?,
                    &paged_expected(50_000 + layer * 10_000, 3, HEADS, DIM),
                );
            }
            append.commit()?
        };
        assert_eq!(first_epoch.get(), 1);
        assert_eq!(table.logical_len(), 3);
        assert_eq!(table.owned_blocks(), 2);

        let second_epoch = {
            let mut append = table.begin_append(2)?;
            for layer in 0..LAYERS {
                let key = paged_values(layer * 10_000, 3, 2, HEADS, DIM)?;
                let value = paged_values(50_000 + layer * 10_000, 3, 2, HEADS, DIM)?;
                append.write_layer(layer, &key, &value)?;
                let (actual_key, actual_value) = append.materialize_layer(layer)?;
                assert_values(
                    &format!("the materialized key history of layer {layer}"),
                    actual_key.dims(),
                    &[1, HEADS, 5, DIM],
                    &actual_key.flatten_all()?.to_vec1::<f32>()?,
                    &paged_expected(layer * 10_000, 5, HEADS, DIM),
                );
                assert_values(
                    &format!("the materialized value history of layer {layer}"),
                    actual_value.dims(),
                    &[1, HEADS, 5, DIM],
                    &actual_value.flatten_all()?.to_vec1::<f32>()?,
                    &paged_expected(50_000 + layer * 10_000, 5, HEADS, DIM),
                );
            }
            append.commit()?
        };
        assert_eq!(second_epoch.get(), 2);
        assert_eq!(table.logical_len(), 5);
        assert_eq!(table.owned_blocks(), 3);
        let stats = pool.stats()?;
        assert_eq!(stats.allocated_blocks, 4);
        assert_eq!(stats.live_blocks, 3);
        assert_eq!(stats.free_blocks, 1);
        assert_eq!(stats.allocated_bytes, 512);

        pool.mark_completed_through(second_epoch)?;
        table.clear();
        let stats = pool.stats()?;
        assert_eq!(stats.free_blocks, 4);
        assert_eq!(stats.live_blocks, 0);
        assert_eq!(stats.retired_blocks, 0);
        Ok(())
    }

    /// A history that fits in one block is handed over as the narrowed views
    /// of that block, so the materialization copies nothing; the two-block
    /// half of the same table keeps the values it had before.
    #[test]
    #[allow(clippy::too_many_lines)] // The one-block and two-block halves of one table stay in one test.
    fn paged_one_block_history_materializes_as_a_slab_view() -> PagedKvResult<()> {
        const LAYERS: usize = 2;
        const HEADS: usize = 2;
        const DIM: usize = 2;
        const BLOCK_TOKENS: usize = 4;
        let pool = paged_pool(LAYERS, HEADS, DIM, BLOCK_TOKENS, 2, 4)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 8)?;

        let mut append = table.begin_append(3)?;
        for layer in 0..LAYERS {
            let key = paged_values(layer * 10_000, 0, 3, HEADS, DIM)?;
            let value = paged_values(50_000 + layer * 10_000, 0, 3, HEADS, DIM)?;
            append.write_layer(layer, &key, &value)?;
        }
        let (key, value) = append.materialize_layer(1)?;
        assert_eq!(key.dim(2)?, 3);
        assert_eq!(value.dim(2)?, 3);
        // The history is a window into the slab, not a fresh copy.
        assert!(!key.is_contiguous());
        assert!(!value.is_contiguous());
        // The layer-1 window starts after the layer-0 window of the block.
        let layer_offset = HEADS * BLOCK_TOKENS * DIM;
        assert_eq!(key.layout().start_offset(), layer_offset);
        assert_eq!(value.layout().start_offset(), layer_offset);
        // The token axis keeps the per-head stride of the slab.
        assert_eq!(key.stride()[2], DIM);
        assert_eq!(key.stride()[1], BLOCK_TOKENS * DIM);
        // The view shares the slab allocation. Both addresses must be
        // known first: a pool whose storage is not CPU F32 has no
        // address, and `None == None` would let the comparison pass
        // without proving that the view aliases the slab.
        let (slab_key, slab_value) = {
            let pool = pool
                .inner
                .try_borrow()
                .map_err(|_| PagedKvPoolError::PoolBusy)?;
            (pool.slabs[0].key.clone(), pool.slabs[0].value.clone())
        };
        let key_address = cpu_storage_address(&key);
        let slab_key_address = cpu_storage_address(&slab_key);
        assert!(
            key_address.is_some(),
            "the one-block key view has no CPU storage address"
        );
        assert!(
            slab_key_address.is_some(),
            "the slab key has no CPU storage address"
        );
        assert_eq!(
            key_address, slab_key_address,
            "the one-block key history does not share the slab storage"
        );
        let value_address = cpu_storage_address(&value);
        let slab_value_address = cpu_storage_address(&slab_value);
        assert!(
            value_address.is_some(),
            "the one-block value view has no CPU storage address"
        );
        assert!(
            slab_value_address.is_some(),
            "the slab value has no CPU storage address"
        );
        assert_eq!(
            value_address, slab_value_address,
            "the one-block value history does not share the slab storage"
        );
        // The values are the ones that were written.
        let expected_key = Tensor::from_vec(
            paged_range(10_000, 0..3, HEADS, DIM),
            (1, HEADS, 3, DIM),
            &Device::Cpu,
        )?;
        let expected_value = Tensor::from_vec(
            paged_range(60_000, 0..3, HEADS, DIM),
            (1, HEADS, 3, DIM),
            &Device::Cpu,
        )?;
        let difference = paged_max_difference(&key, &expected_key)?;
        assert!(
            difference == 0.0,
            "the one-block key history differs by {difference} (shape {:?})",
            key.dims()
        );
        let difference = paged_max_difference(&value, &expected_value)?;
        assert!(
            difference == 0.0,
            "the one-block value history differs by {difference} (shape {:?})",
            value.dims()
        );
        append.commit()?;

        // A history that spans two blocks keeps the values it had before.
        let mut append = table.begin_append(3)?;
        for layer in 0..LAYERS {
            let key = paged_values(70_000 + layer * 10_000, 3, 3, HEADS, DIM)?;
            let value = paged_values(90_000 + layer * 10_000, 3, 3, HEADS, DIM)?;
            append.write_layer(layer, &key, &value)?;
        }
        let (key, value) = append.materialize_layer(0)?;
        assert_eq!(key.dim(2)?, 6);
        assert_eq!(value.dim(2)?, 6);
        let expected_key = Tensor::from_vec(
            paged_two_block_expected(0, 70_000, HEADS, DIM),
            (1, HEADS, 6, DIM),
            &Device::Cpu,
        )?;
        let difference = paged_max_difference(&key, &expected_key)?;
        assert!(
            difference == 0.0,
            "the two-block key history differs by {difference} (shape {:?})",
            key.dims()
        );
        let expected_value = Tensor::from_vec(
            paged_two_block_expected(50_000, 90_000, HEADS, DIM),
            (1, HEADS, 6, DIM),
            &Device::Cpu,
        )?;
        let difference = paged_max_difference(&value, &expected_value)?;
        assert!(
            difference == 0.0,
            "the two-block value history differs by {difference} (shape {:?})",
            value.dims()
        );
        // These asserts pin that the joined history is strided for this
        // two-layer pool with candle 0.11: a layer view of a multi-layer
        // pool has no outer stride for the token axis, so candle concatenates
        // the blocks through its transpose path. A candle release that
        // returns a contiguous join fails only these asserts, and the doc
        // comments of `materialize_layer` then need updating.
        assert!(
            !key.is_contiguous(),
            "the joined key history is contiguous (shape {:?})",
            key.dims()
        );
        assert!(
            !value.is_contiguous(),
            "the joined value history is contiguous (shape {:?})",
            value.dims()
        );
        append.commit()?;
        Ok(())
    }

    /// A history that spans several blocks is joined by `Tensor::cat`, and
    /// the joined history is strided for this pool with candle 0.11: the
    /// partly filled last block of a pool with more than one layer has no
    /// outer stride for the token axis, so candle concatenates the blocks
    /// through its transpose path. The contiguity asserts pin candle's
    /// current behavior: a candle release that returns a contiguous join
    /// fails only these asserts, and the doc comments of `materialize_layer`
    /// then need updating.
    #[test]
    fn paged_multi_block_history_materializes_as_a_strided_join() -> PagedKvResult<()> {
        const LAYERS: usize = 2;
        const HEADS: usize = 2;
        const DIM: usize = 2;
        const BLOCK_TOKENS: usize = 4;
        let pool = paged_pool(LAYERS, HEADS, DIM, BLOCK_TOKENS, 2, 4)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 8)?;

        // Six tokens fill one block and leave the second one partly filled.
        let mut append = table.begin_append(6)?;
        for layer in 0..LAYERS {
            let key = paged_values(layer * 10_000, 0, 6, HEADS, DIM)?;
            let value = paged_values(50_000 + layer * 10_000, 0, 6, HEADS, DIM)?;
            append.write_layer(layer, &key, &value)?;
        }
        let (key, value) = append.materialize_layer(0)?;
        assert_eq!(key.dim(2)?, 6);
        assert_eq!(value.dim(2)?, 6);
        // The joined history holds the values that were written. The values
        // are checked before the layout, so a wrong history is reported as
        // a wrong value and not as a wrong layout.
        let expected_key = Tensor::from_vec(
            paged_expected(0, 6, HEADS, DIM),
            (1, HEADS, 6, DIM),
            &Device::Cpu,
        )?;
        let difference = paged_max_difference(&key, &expected_key)?;
        assert!(
            difference == 0.0,
            "the joined key history differs by {difference} (shape {:?})",
            key.dims()
        );
        let expected_value = Tensor::from_vec(
            paged_expected(50_000, 6, HEADS, DIM),
            (1, HEADS, 6, DIM),
            &Device::Cpu,
        )?;
        let difference = paged_max_difference(&value, &expected_value)?;
        assert!(
            difference == 0.0,
            "the joined value history differs by {difference} (shape {:?})",
            value.dims()
        );
        // The join is strided for this pool with candle 0.11, exactly as the
        // doc comment of `materialize_layer` says it may be. These asserts
        // pin candle's current behavior: a candle release that returns a
        // contiguous join fails only these asserts, and the doc comments of
        // `materialize_layer` then need updating.
        assert!(
            !key.is_contiguous(),
            "the joined key history is contiguous (shape {:?})",
            key.dims()
        );
        assert!(
            !value.is_contiguous(),
            "the joined value history is contiguous (shape {:?})",
            value.dims()
        );
        append.commit()?;
        Ok(())
    }

    /// The batch materialization hands over the same one-block views.
    #[test]
    fn paged_batch_one_block_history_materializes_as_a_slab_view() -> PagedKvResult<()> {
        const HEADS: usize = 2;
        const DIM: usize = 2;
        const BLOCK_TOKENS: usize = 4;
        let pool = paged_pool(1, HEADS, DIM, BLOCK_TOKENS, 2, 4)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 8)?;

        let mut reservation = pool.reserve_batch(&[PagedKvGrowthRequest::new(&table, 3)])?;
        let key = paged_values(10_000, 0, 3, HEADS, DIM)?;
        let value = paged_values(20_000, 0, 3, HEADS, DIM)?;
        reservation.write_layer(0, 0, &key, &value)?;
        let (materialized_key, materialized_value) = reservation.materialize_layer(0, 0)?;
        assert_eq!(materialized_key.dim(2)?, 3);
        assert_eq!(materialized_value.dim(2)?, 3);
        // The history is a window into the slab, not a fresh copy.
        assert!(!materialized_key.is_contiguous());
        assert!(!materialized_value.is_contiguous());
        // The token axis keeps the per-head stride of the slab.
        assert_eq!(materialized_key.stride()[2], DIM);
        assert_eq!(materialized_key.stride()[1], BLOCK_TOKENS * DIM);
        assert_eq!(materialized_value.stride()[2], DIM);
        assert_eq!(materialized_value.stride()[1], BLOCK_TOKENS * DIM);
        // The views share the slab allocation. Both addresses must be
        // known first: a pool whose storage is not CPU F32 has no
        // address, and `None == None` would let the comparison pass
        // without proving that the view aliases the slab.
        let (slab_key, slab_value) = {
            let pool = pool
                .inner
                .try_borrow()
                .map_err(|_| PagedKvPoolError::PoolBusy)?;
            (pool.slabs[0].key.clone(), pool.slabs[0].value.clone())
        };
        let key_address = cpu_storage_address(&materialized_key);
        let slab_key_address = cpu_storage_address(&slab_key);
        assert!(
            key_address.is_some(),
            "the one-block key view has no CPU storage address"
        );
        assert!(
            slab_key_address.is_some(),
            "the slab key has no CPU storage address"
        );
        assert_eq!(
            key_address, slab_key_address,
            "the one-block key history does not share the slab storage"
        );
        let value_address = cpu_storage_address(&materialized_value);
        let slab_value_address = cpu_storage_address(&slab_value);
        assert!(
            value_address.is_some(),
            "the one-block value view has no CPU storage address"
        );
        assert!(
            slab_value_address.is_some(),
            "the slab value has no CPU storage address"
        );
        assert_eq!(
            value_address, slab_value_address,
            "the one-block value history does not share the slab storage"
        );
        // The values are the ones that were written.
        let expected_key = Tensor::from_vec(
            paged_range(10_000, 0..3, HEADS, DIM),
            (1, HEADS, 3, DIM),
            &Device::Cpu,
        )?;
        let difference = paged_max_difference(&materialized_key, &expected_key)?;
        assert!(
            difference == 0.0,
            "the one-block key history differs by {difference} (shape {:?})",
            materialized_key.dims()
        );
        let expected_value = Tensor::from_vec(
            paged_range(20_000, 0..3, HEADS, DIM),
            (1, HEADS, 3, DIM),
            &Device::Cpu,
        )?;
        let difference = paged_max_difference(&materialized_value, &expected_value)?;
        assert!(
            difference == 0.0,
            "the one-block value history differs by {difference} (shape {:?})",
            materialized_value.dims()
        );
        reservation.commit(&mut [&mut table])?;
        Ok(())
    }

    #[test]
    fn paged_prefix_fork_retains_pages_until_the_last_owner_releases() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 4, 1, 1)?;
        let mut source = PagedKvBlockTable::new(pool.clone(), 8)?;
        let values = paged_values(100, 0, 4, 1, 1)?;
        let epoch = {
            let mut append = source.begin_append(4)?;
            append.write_layer(0, &values, &values)?;
            append.commit()?
        };
        pool.mark_completed_through(epoch)?;

        let mut fork = source.fork_prefix(4)?;
        assert_eq!(fork.block_ids(), source.block_ids());
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 1);
        assert_eq!(stats.live_references, 2);
        assert_eq!(stats.shared_blocks, 1);

        source.clear();
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 1);
        assert_eq!(stats.live_references, 1);
        assert_eq!(stats.shared_blocks, 0);

        fork.clear();
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 0);
        assert_eq!(stats.live_references, 0);
        assert_eq!(stats.free_blocks, 1);
        Ok(())
    }

    #[test]
    fn paged_attach_prefix_failures_leave_ownership_unchanged() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 2, 2, 2)?;
        let mut source = PagedKvBlockTable::new(pool.clone(), 8)?;
        let values = paged_values(100, 0, 3, 1, 1)?;
        let epoch = {
            let mut append = source.begin_append(3)?;
            append.write_layer(0, &values, &values)?;
            append.commit()?
        };
        pool.mark_completed_through(epoch)?;
        let before = pool.stats()?;

        let foreign_pool = paged_pool(1, 1, 1, 2, 1, 1)?;
        let mut foreign = PagedKvBlockTable::new(foreign_pool, 8)?;
        assert!(matches!(
            foreign.attach_prefix(&source, 2),
            Err(PagedKvPoolError::InvalidTransaction(_))
        ));
        assert_eq!(foreign.logical_len(), 0);
        assert_eq!(foreign.owned_blocks(), 0);
        assert_eq!(pool.stats()?, before);

        let mut too_small = PagedKvBlockTable::new(pool.clone(), 2)?;
        assert!(matches!(
            too_small.attach_prefix(&source, 3),
            Err(PagedKvPoolError::ContextLimit {
                limit: 2,
                requested: 3,
            })
        ));
        assert_eq!(too_small.logical_len(), 0);
        assert_eq!(pool.stats()?, before);

        let mut too_long = PagedKvBlockTable::new(pool.clone(), 8)?;
        assert!(matches!(
            too_long.attach_prefix(&source, 4),
            Err(PagedKvPoolError::InvalidTransaction(_))
        ));
        assert_eq!(too_long.logical_len(), 0);
        assert_eq!(pool.stats()?, before);

        let mut occupied = source.fork_prefix(1)?;
        let shared = pool.stats()?;
        assert!(matches!(
            occupied.attach_prefix(&source, 1),
            Err(PagedKvPoolError::InvalidTransaction(_))
        ));
        assert_eq!(occupied.logical_len(), 1);
        assert_eq!(pool.stats()?, shared);
        Ok(())
    }

    #[test]
    fn paged_live_reference_updates_are_atomic_for_stale_and_duplicate_ids() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 4, 1, 1)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 8)?;
        let first = paged_values(100, 0, 1, 1, 1)?;
        let first_epoch = {
            let mut append = table.begin_append(1)?;
            append.write_layer(0, &first, &first)?;
            append.commit()?
        };
        let stale = table.block_ids()[0];
        pool.mark_completed_through(first_epoch)?;
        table.clear();

        let second = paged_values(200, 0, 1, 1, 1)?;
        let mut append = table.begin_append(1)?;
        append.write_layer(0, &second, &second)?;
        append.commit()?;
        let live = table.block_ids()[0];
        assert_eq!(live.slot(), stale.slot());
        assert!(live.generation() > stale.generation());
        let before = pool.stats()?;
        assert_paged_occupancy_matches_scan(&pool)?;

        assert!(matches!(
            pool.inner.borrow_mut().retain_live(&[live, stale]),
            Err(PagedKvPoolError::StaleBlockId { .. })
        ));
        assert_eq!(pool.stats()?, before);
        assert_paged_occupancy_matches_scan(&pool)?;
        assert!(matches!(
            pool.inner.borrow_mut().release_live(&[live, stale]),
            Err(PagedKvPoolError::StaleBlockId { .. })
        ));
        assert_eq!(pool.stats()?, before);
        assert_paged_occupancy_matches_scan(&pool)?;

        pool.inner.borrow_mut().retain_live(&[live, live])?;
        let shared = pool.stats()?;
        assert_eq!(shared.live_blocks, 1);
        assert_eq!(shared.live_references, 3);
        assert_eq!(shared.shared_blocks, 1);
        assert_paged_occupancy_matches_scan(&pool)?;
        pool.inner.borrow_mut().release_live(&[live, live])?;
        assert_eq!(pool.stats()?, before);
        assert_paged_occupancy_matches_scan(&pool)?;

        assert!(matches!(
            pool.inner.borrow_mut().release_live(&[live, live]),
            Err(PagedKvPoolError::InvalidTransaction(_))
        ));
        assert_eq!(pool.stats()?, before);
        assert_paged_occupancy_matches_scan(&pool)?;
        Ok(())
    }

    #[test]
    fn paged_shared_full_prefix_appends_on_a_fresh_page() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 4, 2, 2)?;
        let mut source = PagedKvBlockTable::new(pool.clone(), 8)?;
        let prefix = paged_values(100, 0, 4, 1, 1)?;
        let prefix_epoch = {
            let mut append = source.begin_append(4)?;
            append.write_layer(0, &prefix, &prefix)?;
            append.commit()?
        };
        pool.mark_completed_through(prefix_epoch)?;
        let source_id = source.block_ids()[0];
        let mut fork = source.fork_prefix(4)?;
        assert_eq!(fork.additional_blocks_for(1)?, 1);

        let suffix = paged_values(100, 4, 1, 1, 1)?;
        let epoch = {
            let mut append = fork.begin_append(1)?;
            append.write_layer(0, &suffix, &suffix)?;
            let (key, value) = append.materialize_layer(0)?;
            assert_values(
                "the forked key history",
                key.dims(),
                &[1, 1, 5, 1],
                &key.flatten_all()?.to_vec1::<f32>()?,
                &paged_expected(100, 5, 1, 1),
            );
            assert_values(
                "the forked value history",
                value.dims(),
                &[1, 1, 5, 1],
                &value.flatten_all()?.to_vec1::<f32>()?,
                &paged_expected(100, 5, 1, 1),
            );
            append.commit()?
        };
        assert_eq!(source.block_ids(), &[source_id]);
        assert_eq!(fork.block_ids()[0], source_id);
        assert_ne!(fork.block_ids()[1], source_id);
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 2);
        assert_eq!(stats.live_references, 3);
        assert_eq!(stats.shared_blocks, 1);
        pool.mark_completed_through(epoch)?;
        Ok(())
    }

    #[test]
    fn paged_shared_partial_prefix_cows_without_mutating_the_source() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 4, 2, 2)?;
        let mut source = PagedKvBlockTable::new(pool.clone(), 8)?;
        let prefix = paged_values(100, 0, 4, 1, 1)?;
        let prefix_epoch = {
            let mut append = source.begin_append(4)?;
            append.write_layer(0, &prefix, &prefix)?;
            append.commit()?
        };
        pool.mark_completed_through(prefix_epoch)?;
        let source_id = source.block_ids()[0];
        let mut fork = source.fork_prefix(2)?;
        assert_eq!(fork.additional_blocks_for(1)?, 1);

        let suffix = paged_values(900, 2, 1, 1, 1)?;
        let epoch = {
            let mut append = fork.begin_append(1)?;
            assert_ne!(append.block_id(0)?, source_id);
            append.write_layer(0, &suffix, &suffix)?;
            let (key, value) = append.materialize_layer(0)?;
            assert_values(
                "the copied key history",
                key.dims(),
                &[1, 1, 3, 1],
                &key.flatten_all()?.to_vec1::<f32>()?,
                &[100.0, 110.0, 920.0],
            );
            assert_values(
                "the copied value history",
                value.dims(),
                &[1, 1, 3, 1],
                &value.flatten_all()?.to_vec1::<f32>()?,
                &[100.0, 110.0, 920.0],
            );
            append.commit()?
        };

        assert_eq!(source.block_ids(), &[source_id]);
        assert_ne!(fork.block_ids()[0], source_id);
        let (source_key, source_value) = pool.inner.borrow().layer_views(source_id, 0, None)?;
        assert_values(
            "the source key history",
            source_key.dims(),
            &[1, 1, 4, 1],
            &source_key.flatten_all()?.to_vec1::<f32>()?,
            &paged_expected(100, 4, 1, 1),
        );
        assert_values(
            "the source value history",
            source_value.dims(),
            &[1, 1, 4, 1],
            &source_value.flatten_all()?.to_vec1::<f32>()?,
            &paged_expected(100, 4, 1, 1),
        );
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 2);
        assert_eq!(stats.live_references, 2);
        assert_eq!(stats.shared_blocks, 0);
        pool.mark_completed_through(epoch)?;
        Ok(())
    }

    #[test]
    fn paged_batch_cows_two_lanes_that_share_one_partial_tail() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 4, 3, 3)?;
        let mut source = PagedKvBlockTable::new(pool.clone(), 8)?;
        let prefix = paged_values(100, 0, 2, 1, 1)?;
        let prefix_epoch = {
            let mut append = source.begin_append(2)?;
            append.write_layer(0, &prefix, &prefix)?;
            append.commit()?
        };
        pool.mark_completed_through(prefix_epoch)?;
        let source_id = source.block_ids()[0];
        let mut first = source.fork_prefix(2)?;
        let mut second = source.fork_prefix(2)?;
        assert_eq!(pool.stats()?.live_references, 3);

        let mut batch = pool.reserve_batch(&[
            PagedKvGrowthRequest::new(&first, 1),
            PagedKvGrowthRequest::new(&second, 1),
        ])?;
        let first_suffix = paged_values(100, 2, 1, 1, 1)?;
        let second_suffix = paged_values(900, 2, 1, 1, 1)?;
        batch.write_layer(0, 0, &first_suffix, &first_suffix)?;
        batch.write_layer(1, 0, &second_suffix, &second_suffix)?;
        let (first_key, _) = batch.materialize_layer(0, 0)?;
        let (second_key, _) = batch.materialize_layer(1, 0)?;
        assert_values(
            "the first lane key history",
            first_key.dims(),
            &[1, 1, 3, 1],
            &first_key.flatten_all()?.to_vec1::<f32>()?,
            &[100.0, 110.0, 120.0],
        );
        assert_values(
            "the second lane key history",
            second_key.dims(),
            &[1, 1, 3, 1],
            &second_key.flatten_all()?.to_vec1::<f32>()?,
            &[100.0, 110.0, 920.0],
        );
        let epoch = batch.commit(&mut [&mut first, &mut second])?;

        assert_eq!(source.block_ids(), &[source_id]);
        assert_ne!(first.block_ids()[0], source_id);
        assert_ne!(second.block_ids()[0], source_id);
        assert_ne!(first.block_ids()[0], second.block_ids()[0]);
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 3);
        assert_eq!(stats.live_references, 3);
        assert_eq!(stats.shared_blocks, 0);
        assert_paged_occupancy_matches_scan(&pool)?;
        pool.mark_completed_through(epoch)?;
        Ok(())
    }

    #[test]
    fn paged_batch_partial_tail_cow_rollback_quarantines_only_the_replacement() -> PagedKvResult<()>
    {
        let pool = paged_pool(1, 1, 1, 4, 2, 2)?;
        let mut source = PagedKvBlockTable::new(pool.clone(), 8)?;
        let prefix = paged_values(100, 0, 4, 1, 1)?;
        let prefix_epoch = {
            let mut append = source.begin_append(4)?;
            append.write_layer(0, &prefix, &prefix)?;
            append.commit()?
        };
        pool.mark_completed_through(prefix_epoch)?;
        let source_id = source.block_ids()[0];
        let fork = source.fork_prefix(2)?;

        let mut batch = pool.reserve_batch(&[PagedKvGrowthRequest::new(&fork, 1)])?;
        let replacement = batch.block_id(0, 0)?;
        assert_ne!(replacement, source_id);
        let suffix = paged_values(900, 2, 1, 1, 1)?;
        batch.write_layer(0, 0, &suffix, &suffix)?;
        let failed_epoch = batch.epoch();
        batch.rollback();

        assert_eq!(source.block_ids(), &[source_id]);
        assert_eq!(fork.block_ids(), &[source_id]);
        assert_eq!(source.logical_len(), 4);
        assert_eq!(fork.logical_len(), 2);
        let (source_key, _) = pool.inner.borrow().layer_views(source_id, 0, None)?;
        assert_values(
            "the source key history",
            source_key.dims(),
            &[1, 1, 4, 1],
            &source_key.flatten_all()?.to_vec1::<f32>()?,
            &paged_expected(100, 4, 1, 1),
        );
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 1);
        assert_eq!(stats.live_references, 2);
        assert_eq!(stats.shared_blocks, 1);
        assert_eq!(stats.retired_blocks, 1);
        assert_eq!(stats.free_blocks, 0);
        assert_paged_occupancy_matches_scan(&pool)?;

        pool.mark_completed_through(failed_epoch)?;
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 1);
        assert_eq!(stats.free_blocks, 1);
        assert_eq!(stats.retired_blocks, 0);
        assert_paged_occupancy_matches_scan(&pool)?;
        Ok(())
    }

    #[test]
    fn paged_batch_capacity_preflight_fails_before_allocating_a_slab() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 2, 2, 3)?;
        let first = PagedKvBlockTable::new(pool.clone(), 8)?;
        let second = PagedKvBlockTable::new(pool.clone(), 8)?;
        let requested = first
            .additional_blocks_for(5)?
            .checked_add(second.additional_blocks_for(1)?)
            .unwrap();
        assert_eq!(requested, 4);
        assert!(matches!(
            pool.ensure_free_blocks(requested),
            Err(PagedKvPoolError::Exhausted {
                requested_blocks: 4,
                free_blocks: 0,
                unallocated_blocks: 3,
            })
        ));
        let stats = pool.stats()?;
        assert_eq!(stats.allocated_blocks, 0);
        assert_eq!(stats.submitted_epoch, PagedKvEpoch::ZERO);

        pool.ensure_free_blocks(3)?;
        let stats = pool.stats()?;
        assert_eq!(stats.allocated_blocks, 3);
        assert_eq!(stats.free_blocks, 3);
        assert_eq!(stats.submitted_epoch, PagedKvEpoch::ZERO);
        Ok(())
    }

    #[test]
    fn paged_owned_batch_commits_two_lanes_in_one_epoch() -> PagedKvResult<()> {
        const LAYERS: usize = 2;
        let pool = paged_pool(LAYERS, 1, 1, 2, 2, 4)?;
        let mut first = PagedKvBlockTable::new(pool.clone(), 8)?;
        let mut second = PagedKvBlockTable::new(pool.clone(), 8)?;
        let mut batch = pool.reserve_batch(&[
            PagedKvGrowthRequest::new(&first, 3),
            PagedKvGrowthRequest::new(&second, 1),
        ])?;

        assert_eq!(batch.epoch().get(), 1);
        assert_eq!(batch.lane_count(), 2);
        assert_eq!(batch.lane_old_len(0)?, 0);
        assert_eq!(batch.lane_new_len(0)?, 3);
        assert_eq!(batch.lane_new_len(1)?, 1);
        for (lane, len) in [(0, 3), (1, 1)] {
            for layer in 0..LAYERS {
                let base = lane * 10_000 + layer * 1_000;
                let key = paged_values(base, 0, len, 1, 1)?;
                let value = paged_values(50_000 + base, 0, len, 1, 1)?;
                batch.write_layer(lane, layer, &key, &value)?;
                let (actual_key, actual_value) = batch.materialize_layer(lane, layer)?;
                assert_values(
                    &format!("the key history of lane {lane}"),
                    actual_key.dims(),
                    &[1, 1, len, 1],
                    &actual_key.flatten_all()?.to_vec1::<f32>()?,
                    &paged_expected(base, len, 1, 1),
                );
                assert_values(
                    &format!("the value history of lane {lane}"),
                    actual_value.dims(),
                    &[1, 1, len, 1],
                    &actual_value.flatten_all()?.to_vec1::<f32>()?,
                    &paged_expected(50_000 + base, len, 1, 1),
                );
            }
        }

        let epoch = batch.commit(&mut [&mut first, &mut second])?;
        assert_eq!(epoch.get(), 1);
        assert_eq!(first.logical_len(), 3);
        assert_eq!(first.owned_blocks(), 2);
        assert_eq!(second.logical_len(), 1);
        assert_eq!(second.owned_blocks(), 1);
        assert_eq!(
            first.stats()?,
            KvCacheStats {
                policy: KvCachePolicy::Paged,
                logical_len: 3,
                resident_start: 0,
                resident_len: 3,
                allocated_capacity: 4,
                allocated_bytes: 32,
                physical_capacity: 8,
                logical_limit: 8,
            }
        );
        let stats = pool.stats()?;
        assert_eq!(stats.submitted_epoch, epoch);
        assert_eq!(stats.live_blocks, 3);
        assert_eq!(stats.reserved_blocks, 0);
        Ok(())
    }

    #[test]
    fn paged_attention_batch_reuses_one_compact_page_table_across_layers() -> PagedKvResult<()> {
        const LAYERS: usize = 2;
        let pool = paged_pool(LAYERS, 1, 2, 2, 2, 4)?;
        let mut first = PagedKvBlockTable::new(pool.clone(), 8)?;
        let mut second = PagedKvBlockTable::new(pool.clone(), 8)?;
        let mut batch = pool.reserve_batch(&[
            PagedKvGrowthRequest::new(&first, 3),
            PagedKvGrowthRequest::new(&second, 1),
        ])?;
        let mut compact_page_table_id = None;

        for layer in 0..LAYERS {
            for (lane, len) in [(0, 3), (1, 1)] {
                let values = paged_values(layer * 1_000 + lane * 100, 0, len, 1, 2)?;
                batch.write_layer(lane, layer, &values, &values)?;
                let read = batch.layer_read(lane, layer)?;
                assert_eq!(read.page_offset(), if lane == 0 { 0 } else { 2 });
                assert_eq!(read.page_count(), if lane == 0 { 2 } else { 1 });
                assert_eq!(read.logical_len(), len);
                assert_eq!(read.block_tokens(), 2);
                assert_eq!(read.full_layers(), LAYERS);
                assert_eq!(read.kv_heads(), 1);
                assert_eq!(read.head_dim(), 2);
                assert_eq!(read.dtype(), DType::F32);
                assert_eq!(read.layer(), layer);
                assert_eq!(read.slabs().len(), 2);
                let expected_page_table = vec![vec![0_u32, 0], vec![0, 1], vec![1, 0]];
                let page_table = read.page_table().to_vec2::<u32>()?;
                assert_eq!(
                    page_table.len(),
                    expected_page_table.len(),
                    "the page table has one row per page"
                );
                assert_values(
                    "the page table",
                    read.page_table().dims(),
                    &[3, 2],
                    &flatten_page_table(&page_table),
                    &flatten_page_table(&expected_page_table),
                );
            }

            let read = batch.layer_batch_read(layer)?;
            assert_eq!(read.lane_count(), 2);
            assert_eq!(read.lane_page_offsets(), &[0, 2]);
            assert_eq!(read.lane_page_counts(), &[2, 1]);
            assert_eq!(read.lane_logical_lens(), &[3, 1]);
            assert_eq!(read.block_tokens(), 2);
            assert_eq!(read.full_layers(), LAYERS);
            assert_eq!(read.layer(), layer);
            assert_eq!(read.kv_heads(), 1);
            assert_eq!(read.head_dim(), 2);
            assert_eq!(read.dtype(), DType::F32);
            assert_eq!(read.slabs().len(), 2);
            let expected_page_table = vec![vec![0_u32, 0], vec![0, 1], vec![1, 0]];
            let page_table = read.page_table().to_vec2::<u32>()?;
            assert_eq!(
                page_table.len(),
                expected_page_table.len(),
                "the page table has one row per page"
            );
            assert_values(
                "the page table",
                read.page_table().dims(),
                &[3, 2],
                &flatten_page_table(&page_table),
                &flatten_page_table(&expected_page_table),
            );
            let current_page_table_id = read.page_table().id();
            if let Some(first_page_table_id) = compact_page_table_id {
                assert_eq!(current_page_table_id, first_page_table_id);
            } else {
                compact_page_table_id = Some(current_page_table_id);
            }
        }

        let epoch = batch.commit(&mut [&mut first, &mut second])?;
        pool.mark_completed_through(epoch)?;
        Ok(())
    }

    #[test]
    fn paged_attention_batch_rejects_a_partially_written_layer() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 2, 2, 4)?;
        let mut first = PagedKvBlockTable::new(pool.clone(), 8)?;
        let mut second = PagedKvBlockTable::new(pool.clone(), 8)?;
        let mut batch = pool.reserve_batch(&[
            PagedKvGrowthRequest::new(&first, 1),
            PagedKvGrowthRequest::new(&second, 1),
        ])?;
        let values = paged_values(100, 0, 1, 1, 1)?;
        batch.write_layer(0, 0, &values, &values)?;

        assert!(matches!(
            batch.layer_batch_read(0),
            Err(PagedKvPoolError::InvalidTransaction(message))
                if message.contains("lane 1 full layer 0")
        ));
        assert!(batch.attention_batch.is_none());

        batch.write_layer(1, 0, &values, &values)?;
        let read = batch.layer_batch_read(0)?;
        assert_eq!(read.lane_page_offsets(), &[0, 1]);
        assert_eq!(read.lane_page_counts(), &[1, 1]);
        assert_eq!(read.lane_logical_lens(), &[1, 1]);

        let epoch = batch.commit(&mut [&mut first, &mut second])?;
        pool.mark_completed_through(epoch)?;
        Ok(())
    }

    #[test]
    fn paged_zero_growth_batch_reuses_a_partial_page_with_free_peer_slots() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 4, 2, 2)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 8)?;

        let mut first = pool.reserve_batch(&[PagedKvGrowthRequest::new(&table, 1)])?;
        let value = paged_values(10, 0, 1, 1, 1)?;
        first.write_layer(0, 0, &value, &value)?;
        let first_epoch = first.commit(&mut [&mut table])?;
        pool.mark_completed_through(first_epoch)?;
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 1);
        assert_eq!(stats.free_blocks, 1);

        // The append fits in the existing partial page, so the batch owns an
        // epoch but reserves no new block. Unrelated free slots must not turn
        // that zero-growth reservation into a transaction failure.
        let mut second = pool.reserve_batch(&[PagedKvGrowthRequest::new(&table, 1)])?;
        let value = paged_values(10, 1, 1, 1, 1)?;
        second.write_layer(0, 0, &value, &value)?;
        let (key, _) = second.materialize_layer(0, 0)?;
        assert_values(
            "the key history of the reused page",
            key.dims(),
            &[1, 1, 2, 1],
            &key.flatten_all()?.to_vec1::<f32>()?,
            &paged_expected(10, 2, 1, 1),
        );
        let second_epoch = second.commit(&mut [&mut table])?;
        pool.mark_completed_through(second_epoch)?;
        assert_eq!(table.logical_len(), 2);
        assert_eq!(table.owned_blocks(), 1);
        Ok(())
    }

    #[test]
    fn paged_materialization_quarantines_every_historical_page_on_rollback() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 2, 2, 2)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 8)?;

        let mut prefix = pool.reserve_batch(&[PagedKvGrowthRequest::new(&table, 3)])?;
        let values = paged_values(0, 0, 3, 1, 1)?;
        prefix.write_layer(0, 0, &values, &values)?;
        let prefix_epoch = prefix.commit(&mut [&mut table])?;
        pool.mark_completed_through(prefix_epoch)?;

        let mut failed = pool.reserve_batch(&[PagedKvGrowthRequest::new(&table, 1)])?;
        let values = paged_values(0, 3, 1, 1, 1)?;
        failed.write_layer(0, 0, &values, &values)?;
        let _ = failed.materialize_layer(0, 0)?;
        let failed_epoch = failed.epoch();
        failed.rollback();
        table.clear();

        let stats = pool.stats()?;
        assert_eq!(stats.free_blocks, 0);
        assert_eq!(stats.live_blocks, 0);
        assert_eq!(stats.retired_blocks, 2);
        pool.mark_completed_through(failed_epoch)?;
        assert_eq!(pool.stats()?.free_blocks, 2);
        Ok(())
    }

    #[test]
    fn paged_indirect_read_quarantines_every_historical_page_on_rollback() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 2, 2, 2)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 8)?;

        let mut prefix = pool.reserve_batch(&[PagedKvGrowthRequest::new(&table, 3)])?;
        let values = paged_values(0, 0, 3, 1, 1)?;
        prefix.write_layer(0, 0, &values, &values)?;
        let prefix_epoch = prefix.commit(&mut [&mut table])?;
        pool.mark_completed_through(prefix_epoch)?;

        let mut failed = pool.reserve_batch(&[PagedKvGrowthRequest::new(&table, 1)])?;
        let values = paged_values(0, 3, 1, 1, 1)?;
        failed.write_layer(0, 0, &values, &values)?;
        let failed_epoch = failed.epoch();
        {
            let read = failed.layer_read(0, 0)?;
            assert_eq!(read.page_count(), 2);
        }
        failed.rollback();
        table.clear();

        let stats = pool.stats()?;
        assert_eq!(stats.free_blocks, 0);
        assert_eq!(stats.live_blocks, 0);
        assert_eq!(stats.retired_blocks, 2);
        pool.mark_completed_through(failed_epoch)?;
        assert_eq!(pool.stats()?.free_blocks, 2);
        Ok(())
    }

    #[test]
    fn paged_pool_serializes_reservations_to_preserve_epoch_submission_order() -> PagedKvResult<()>
    {
        let pool = paged_pool(1, 1, 1, 2, 2, 2)?;
        let first_table = PagedKvBlockTable::new(pool.clone(), 4)?;
        let second_table = PagedKvBlockTable::new(pool.clone(), 4)?;
        let first = pool.reserve_batch(&[PagedKvGrowthRequest::new(&first_table, 1)])?;
        let first_epoch = first.epoch();

        assert!(matches!(
            pool.reserve_batch(&[PagedKvGrowthRequest::new(&second_table, 1)]),
            Err(PagedKvPoolError::InvalidTransaction(_))
        ));
        assert!(matches!(
            pool.mark_completed_through(first_epoch),
            Err(PagedKvPoolError::InvalidTransaction(_))
        ));
        assert_eq!(pool.stats()?.submitted_epoch, first_epoch);
        drop(first);

        let second = pool.reserve_batch(&[PagedKvGrowthRequest::new(&second_table, 1)])?;
        assert_eq!(second.epoch().get(), first_epoch.get() + 1);
        drop(second);
        Ok(())
    }

    #[test]
    fn paged_owned_batch_second_lane_exhaustion_is_atomic() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 2, 1, 2)?;
        let first = PagedKvBlockTable::new(pool.clone(), 8)?;
        let second = PagedKvBlockTable::new(pool.clone(), 8)?;
        let before = pool.stats()?;

        assert!(matches!(
            pool.reserve_batch(&[
                PagedKvGrowthRequest::new(&first, 2),
                PagedKvGrowthRequest::new(&second, 3),
            ]),
            Err(PagedKvPoolError::Exhausted {
                requested_blocks: 3,
                free_blocks: 0,
                unallocated_blocks: 2,
            })
        ));
        assert_eq!(pool.stats()?, before);
        assert_eq!(first.logical_len(), 0);
        assert_eq!(first.owned_blocks(), 0);
        assert_eq!(second.logical_len(), 0);
        assert_eq!(second.owned_blocks(), 0);
        assert_eq!(first.identity.in_flight.get(), None);
        assert_eq!(second.identity.in_flight.get(), None);
        Ok(())
    }

    #[test]
    fn paged_owned_batch_rejects_wrong_commit_order_and_unlocks() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 2, 2, 2)?;
        let mut first = PagedKvBlockTable::new(pool.clone(), 4)?;
        let mut second = PagedKvBlockTable::new(pool.clone(), 4)?;
        let batch = pool.reserve_batch(&[
            PagedKvGrowthRequest::new(&first, 1),
            PagedKvGrowthRequest::new(&second, 1),
        ])?;

        assert!(matches!(
            batch.commit(&mut [&mut second, &mut first]),
            Err(PagedKvPoolError::InvalidTransaction(_))
        ));
        assert_eq!(first.identity.in_flight.get(), None);
        assert_eq!(second.identity.in_flight.get(), None);
        assert_eq!(first.logical_len(), 0);
        assert_eq!(second.logical_len(), 0);
        let stats = pool.stats()?;
        assert_eq!(stats.reserved_blocks, 0);
        assert_eq!(stats.free_blocks, 2);

        // The legacy one-lane path also observes the owned-reservation lock.
        let append = first.begin_append(1)?;
        append.rollback();
        Ok(())
    }

    #[test]
    fn paged_owned_batch_rejects_duplicate_table_without_pool_mutation() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 2, 1, 2)?;
        let table = PagedKvBlockTable::new(pool.clone(), 4)?;
        let before = pool.stats()?;
        assert!(matches!(
            pool.reserve_batch(&[
                PagedKvGrowthRequest::new(&table, 1),
                PagedKvGrowthRequest::new(&table, 1),
            ]),
            Err(PagedKvPoolError::InvalidTransaction(_))
        ));
        assert_eq!(pool.stats()?, before);
        assert_eq!(table.identity.in_flight.get(), None);
        Ok(())
    }

    #[test]
    fn paged_owned_batch_blocks_clear_until_drop_rollback() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 2, 1, 1)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 4)?;
        let batch = pool.reserve_batch(&[PagedKvGrowthRequest::new(&table, 1)])?;
        assert!(table.identity.in_flight.get().is_some());

        table.clear();
        assert_eq!(table.logical_len(), 0);
        assert_eq!(table.owned_blocks(), 0);
        assert!(matches!(
            table.begin_append(1),
            Err(PagedKvPoolError::InvalidTransaction(_))
        ));

        drop(batch);
        assert_eq!(table.identity.in_flight.get(), None);
        let stats = pool.stats()?;
        assert_eq!(stats.reserved_blocks, 0);
        assert_eq!(stats.free_blocks, 1);
        Ok(())
    }

    #[test]
    fn paged_partial_tail_rollback_stays_private_and_retry_overwrites_it() -> PagedKvResult<()> {
        let pool = paged_pool(2, 1, 1, 4, 1, 1)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 8)?;
        let prefix_epoch = {
            let mut append = table.begin_append(2)?;
            for layer in 0..2 {
                let values = paged_values(layer * 100, 0, 2, 1, 1)?;
                append.write_layer(layer, &values, &values)?;
            }
            append.commit()?
        };
        pool.mark_completed_through(prefix_epoch)?;

        let failed_epoch = {
            let mut append = table.begin_append(1)?;
            let dirty = paged_values(9_000, 2, 1, 1, 1)?;
            append.write_layer(0, &dirty, &dirty)?;
            let epoch = append.epoch();
            append.rollback();
            epoch
        };
        assert_eq!(table.logical_len(), 2);
        assert_eq!(table.owned_blocks(), 1);
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 1);
        assert_eq!(stats.retired_blocks, 0);
        assert_eq!(stats.submitted_epoch, failed_epoch);

        let retry_epoch = {
            let mut append = table.begin_append(1)?;
            for layer in 0..2 {
                let values = paged_values(layer * 100, 2, 1, 1, 1)?;
                append.write_layer(layer, &values, &values)?;
                let (key, value) = append.materialize_layer(layer)?;
                assert_values(
                    &format!("the retried key history of layer {layer}"),
                    key.dims(),
                    &[1, 1, 3, 1],
                    &key.flatten_all()?.to_vec1::<f32>()?,
                    &paged_expected(layer * 100, 3, 1, 1),
                );
                assert_values(
                    &format!("the retried value history of layer {layer}"),
                    value.dims(),
                    &[1, 1, 3, 1],
                    &value.flatten_all()?.to_vec1::<f32>()?,
                    &paged_expected(layer * 100, 3, 1, 1),
                );
            }
            append.commit()?
        };
        pool.mark_completed_through(retry_epoch)?;
        assert_eq!(table.logical_len(), 3);
        assert_eq!(table.owned_blocks(), 1);
        Ok(())
    }

    #[test]
    fn paged_transaction_rolls_back_all_layers_after_a_physical_write() -> PagedKvResult<()> {
        let pool = paged_pool(2, 1, 1, 2, 1, 1)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 4)?;
        let (old_id, epoch) = {
            let mut append = table.begin_append(1)?;
            let id = append.block_id(0)?;
            let first = paged_values(0, 0, 1, 1, 1)?;
            append.write_layer(0, &first, &first)?;
            let (_, _) = append.materialize_layer(0)?;
            let epoch = append.epoch();
            append.rollback();
            (id, epoch)
        };
        assert_eq!(table.logical_len(), 0);
        assert_eq!(table.owned_blocks(), 0);
        let stats = pool.stats()?;
        assert_eq!(stats.retired_blocks, 1);
        assert_eq!(stats.free_blocks, 0);

        let mut peer = PagedKvBlockTable::new(pool.clone(), 4)?;
        assert!(matches!(
            peer.begin_append(1).unwrap_err(),
            PagedKvPoolError::Exhausted {
                requested_blocks: 1,
                free_blocks: 0,
                unallocated_blocks: 0,
            }
        ));
        pool.mark_completed_through(epoch)?;
        let new_id = {
            let mut append = peer.begin_append(1)?;
            let id = append.block_id(0)?;
            for layer in 0..2 {
                let values = paged_values(100 + layer, 0, 1, 1, 1)?;
                append.write_layer(layer, &values, &values)?;
            }
            let completion = append.commit()?;
            pool.mark_completed_through(completion)?;
            id
        };
        assert_eq!(new_id.slot(), old_id.slot());
        assert!(new_id.generation() > old_id.generation());
        assert!(matches!(
            pool.inner.borrow().slot(old_id),
            Err(PagedKvPoolError::StaleBlockId { .. })
        ));
        Ok(())
    }

    #[test]
    fn paged_exhaustion_is_typed_and_does_not_publish_an_epoch() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 2, 1, 1)?;
        let mut owner = PagedKvBlockTable::new(pool.clone(), 4)?;
        let values = paged_values(0, 0, 1, 1, 1)?;
        let epoch = {
            let mut append = owner.begin_append(1)?;
            append.write_layer(0, &values, &values)?;
            append.commit()?
        };
        let before = pool.stats()?;
        assert_eq!(before.submitted_epoch, epoch);

        let mut waiter = PagedKvBlockTable::new(pool.clone(), 4)?;
        assert!(matches!(
            waiter.begin_append(1).unwrap_err(),
            PagedKvPoolError::Exhausted { .. }
        ));
        assert_eq!(pool.stats()?, before);
        assert_eq!(waiter.logical_len(), 0);
        assert_eq!(waiter.owned_blocks(), 0);
        Ok(())
    }

    #[test]
    fn paged_commit_requires_every_full_layer_and_quarantines_writes() -> PagedKvResult<()> {
        let pool = paged_pool(2, 1, 1, 2, 1, 1)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 4)?;
        let values = paged_values(0, 0, 1, 1, 1)?;
        let epoch = {
            let mut append = table.begin_append(1)?;
            let epoch = append.epoch();
            append.write_layer(0, &values, &values)?;
            assert!(
                append
                    .commit()
                    .unwrap_err()
                    .to_string()
                    .contains("every full layer")
            );
            epoch
        };
        assert_eq!(table.logical_len(), 0);
        let stats = pool.stats()?;
        assert_eq!(stats.reserved_blocks, 0);
        assert_eq!(stats.retired_blocks, 1);
        pool.mark_completed_through(epoch)?;
        assert_eq!(pool.stats()?.free_blocks, 1);
        Ok(())
    }

    #[test]
    fn paged_geometry_error_before_write_returns_reserved_block_directly() -> PagedKvResult<()> {
        let pool = paged_pool(1, 2, 2, 2, 1, 1)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 4)?;
        {
            let mut append = table.begin_append(1)?;
            let wrong = Tensor::zeros((1, 1, 1, 2), DType::F32, &Device::Cpu)?;
            assert!(matches!(
                append.write_layer(0, &wrong, &wrong),
                Err(PagedKvPoolError::InvalidGeometry(_))
            ));
        }
        let stats = pool.stats()?;
        assert_eq!(stats.free_blocks, 1);
        assert_eq!(stats.retired_blocks, 0);
        assert_eq!(table.logical_len(), 0);
        Ok(())
    }

    #[test]
    fn paged_clear_retires_until_completion_then_reuses_with_new_generation() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 4, 1, 1)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 8)?;
        let old_id;
        let epoch;
        {
            let values = paged_values(0, 0, 1, 1, 1)?;
            let mut append = table.begin_append(1)?;
            append.write_layer(0, &values, &values)?;
            epoch = append.commit()?;
            old_id = table.block_ids()[0];
        }
        table.clear();
        assert_eq!(pool.stats()?.retired_blocks, 1);

        let mut next = PagedKvBlockTable::new(pool.clone(), 8)?;
        assert!(matches!(
            next.begin_append(1).unwrap_err(),
            PagedKvPoolError::Exhausted { .. }
        ));
        pool.mark_completed_through(epoch)?;
        let new_id;
        let next_epoch;
        {
            let values = paged_values(900, 0, 1, 1, 1)?;
            let mut append = next.begin_append(1)?;
            new_id = append.block_id(0)?;
            append.write_layer(0, &values, &values)?;
            let (key, _) = append.materialize_layer(0)?;
            assert_values(
                "the key history of the reused block",
                key.dims(),
                &[1, 1, 1, 1],
                &key.flatten_all()?.to_vec1::<f32>()?,
                &[900.0],
            );
            next_epoch = append.commit()?;
        }
        assert_eq!(new_id.slot(), old_id.slot());
        assert_eq!(new_id.generation(), old_id.generation() + 1);
        pool.mark_completed_through(next_epoch)?;
        next.clear();
        assert_eq!(pool.stats()?.free_blocks, 1);
        Ok(())
    }

    #[test]
    fn paged_partial_reuse_never_materializes_stale_tail() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 4, 1, 1)?;
        let mut first = PagedKvBlockTable::new(pool.clone(), 4)?;
        let completion = {
            let old = paged_values(100, 0, 4, 1, 1)?;
            let mut append = first.begin_append(4)?;
            append.write_layer(0, &old, &old)?;
            append.commit()?
        };
        pool.mark_completed_through(completion)?;
        first.clear();

        let mut second = PagedKvBlockTable::new(pool.clone(), 4)?;
        let completion = {
            let fresh = paged_values(7, 0, 1, 1, 1)?;
            let mut append = second.begin_append(1)?;
            append.write_layer(0, &fresh, &fresh)?;
            let (key, value) = append.materialize_layer(0)?;
            assert_eq!(key.dims4()?, (1, 1, 1, 1));
            assert_values(
                "the key history of the reused block",
                key.dims(),
                &[1, 1, 1, 1],
                &key.flatten_all()?.to_vec1::<f32>()?,
                &[7.0],
            );
            assert_values(
                "the value history of the reused block",
                value.dims(),
                &[1, 1, 1, 1],
                &value.flatten_all()?.to_vec1::<f32>()?,
                &[7.0],
            );
            append.commit()?
        };
        pool.mark_completed_through(completion)?;
        Ok(())
    }

    #[test]
    fn paged_clear_and_drop_fail_closed_during_reentrant_pool_borrow() -> PagedKvResult<()> {
        let pool = paged_pool(1, 1, 1, 2, 1, 2)?;
        let mut table = PagedKvBlockTable::new(pool.clone(), 4)?;
        let values = paged_values(0, 0, 1, 1, 1)?;
        let epoch = {
            let mut append = table.begin_append(1)?;
            append.write_layer(0, &values, &values)?;
            append.commit()?
        };
        {
            let _borrow = pool.inner.borrow_mut();
            table.clear();
            assert_eq!(table.logical_len(), 1);
            assert_eq!(table.owned_blocks(), 1);
        }
        table.clear();
        assert_eq!(pool.stats()?.retired_blocks, 1);
        pool.mark_completed_through(epoch)?;

        let mut dropped = PagedKvBlockTable::new(pool.clone(), 4)?;
        let epoch = {
            let mut append = dropped.begin_append(1)?;
            append.write_layer(0, &values, &values)?;
            append.commit()?
        };
        {
            let _borrow = pool.inner.borrow_mut();
            drop(dropped);
        }
        // Drop cannot risk reusing the block while it cannot inspect the pool;
        // the slot deliberately remains live instead of panicking.
        let stats = pool.stats()?;
        assert_eq!(stats.live_blocks, 1);
        assert_eq!(stats.submitted_epoch, epoch);
        Ok(())
    }

    #[test]
    fn appends_only_on_sequence_axis_and_reuses_storage() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new(32);
        let first = Tensor::zeros((1, 2, 3, 4), candle_core::DType::F32, &device)?;
        let next = Tensor::ones((1, 2, 1, 4), candle_core::DType::F32, &device)?;
        let _ = cache.append(&first, &first)?;
        assert_eq!(cache.allocated_capacity(), 16);
        let (key, value) = cache.append(&next, &next)?;
        assert_eq!(key.dims4()?, (1, 2, 4, 4));
        assert_eq!(value.dims4()?, (1, 2, 4, 4));
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.capacity(), 32);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.allocated_capacity(), 16);
        let (key, _) = cache.append(&next, &next)?;
        assert_eq!(key.dims4()?, (1, 2, 1, 4));
        assert_values(
            "the key history of the dense cache",
            key.dims(),
            &[1, 2, 1, 4],
            &key.flatten_all()?.to_vec1::<f32>()?,
            &[1.0; 8],
        );
        Ok(())
    }

    #[test]
    fn grows_geometrically_and_preserves_existing_tokens() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new(33);
        let first = Tensor::zeros((1, 1, 1, 2), candle_core::DType::F32, &device)?;
        let more = Tensor::ones((1, 1, 16, 2), candle_core::DType::F32, &device)?;
        let _ = cache.append(&first, &first)?;
        assert_eq!(cache.allocated_capacity(), 16);
        let (key, value) = cache.append(&more, &more)?;
        assert_eq!(cache.allocated_capacity(), 32);
        assert_eq!(key.dims4()?, (1, 1, 17, 2));
        assert_eq!(value.dims4()?, (1, 1, 17, 2));
        let expected = std::iter::repeat_n(0.0, 2)
            .chain(std::iter::repeat_n(1.0, 32))
            .collect::<Vec<_>>();
        assert_values(
            "the key history of the grown cache",
            key.dims(),
            &[1, 1, 17, 2],
            &key.flatten_all()?.to_vec1::<f32>()?,
            &expected,
        );
        assert_values(
            "the value history of the grown cache",
            value.dims(),
            &[1, 1, 17, 2],
            &value.flatten_all()?.to_vec1::<f32>()?,
            &expected,
        );
        Ok(())
    }

    #[test]
    fn refuses_capacity_overflow_and_can_rollback() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new(2);
        let token = Tensor::zeros((1, 1, 1, 2), candle_core::DType::F32, &device)?;
        let _ = cache.append(&token, &token)?;
        let _ = cache.append(&token, &token)?;
        assert!(cache.append(&token, &token).is_err());
        cache.truncate(1);
        assert_eq!(cache.len(), 1);
        Ok(())
    }

    #[test]
    fn sliding_cache_keeps_absolute_progress_and_a_bounded_ordered_tail() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = SlidingKvCache::new(4, 32)?;
        let first = token_values(0, 3, &device)?;
        cache.append(&first, &first)?;
        let (values, dims) = sliding_key_history(&cache)?;
        assert_values(
            "the sliding key history",
            &dims,
            &[1, 1, 3, 1],
            &values,
            &[0.0, 1.0, 2.0],
        );
        assert_eq!(
            cache.stats(),
            KvCacheStats {
                policy: KvCachePolicy::Sliding,
                logical_len: 3,
                resident_start: 0,
                resident_len: 3,
                allocated_capacity: 6,
                allocated_bytes: 48,
                physical_capacity: 6,
                logical_limit: 32,
            }
        );

        for position in 3..=6 {
            let token = token_values(position, 1, &device)?;
            cache.append(&token, &token)?;
        }
        assert_eq!(cache.logical_len(), 7);
        assert_eq!(cache.stats().resident_start, 4);
        assert_eq!(cache.stats().resident_len, 3);
        assert_eq!(cache.stats().allocated_capacity, 6);
        let (values, dims) = sliding_key_history(&cache)?;
        assert_values(
            "the sliding key history",
            &dims,
            &[1, 1, 3, 1],
            &values,
            &[4.0, 5.0, 6.0],
        );
        let segments = cache.history_segments()?;
        assert_eq!(
            segments.len(),
            2,
            "the live tail should cross the ring wrap"
        );
        assert_eq!(segments[0].absolute_start, 4);
        assert_eq!(segments[1].absolute_start, 6);
        Ok(())
    }

    #[test]
    fn sliding_cache_one_shot_and_chunked_appends_keep_the_same_tail() -> Result<()> {
        let device = Device::Cpu;
        let all = token_values(0, 11, &device)?;
        let mut one_shot = SlidingKvCache::new(4, 32)?;
        one_shot.append(&all, &all)?;

        let mut chunked = SlidingKvCache::new(4, 32)?;
        for (start, len) in [(0, 3), (3, 2), (5, 6)] {
            let chunk = token_values(start, len, &device)?;
            chunked.append(&chunk, &chunk)?;
        }
        assert_eq!(one_shot.logical_len(), 11);
        assert_eq!(chunked.logical_len(), 11);
        let (one_shot_values, one_shot_dims) = sliding_key_history(&one_shot)?;
        let (chunked_values, chunked_dims) = sliding_key_history(&chunked)?;
        assert_values(
            "the one-shot sliding key history",
            &one_shot_dims,
            &[1, 1, 3, 1],
            &one_shot_values,
            &[8.0, 9.0, 10.0],
        );
        assert_values(
            "the chunked sliding key history",
            &chunked_dims,
            &one_shot_dims,
            &chunked_values,
            &one_shot_values,
        );
        assert_eq!(chunked.stats().allocated_capacity, 6);
        Ok(())
    }

    #[test]
    fn sliding_cache_metadata_rollback_restores_wrapped_contents() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = LayerKvCache::Sliding(SlidingKvCache::new(4, 32)?);
        let LayerKvCache::Sliding(sliding) = &mut cache else {
            unreachable!()
        };
        let first = token_values(0, 3, &device)?;
        sliding.append(&first, &first)?;
        let second = token_values(3, 2, &device)?;
        sliding.append(&second, &second)?;
        let (values, dims) = sliding_key_history(sliding)?;
        assert_values(
            "the sliding key history",
            &dims,
            &[1, 1, 3, 1],
            &values,
            &[2.0, 3.0, 4.0],
        );

        let checkpoint = sliding.checkpoint();
        // Cursor is at physical slot 5. Q=H writes slots 5,0,1 and therefore
        // crosses the ring tail without touching the checkpoint's slots 2,3,4.
        let failed_step = token_values(5, 3, &device)?;
        let LayerKvCache::Sliding(sliding) = &mut cache else {
            unreachable!()
        };
        sliding.append(&failed_step, &failed_step)?;
        let (values, dims) = sliding_key_history(sliding)?;
        assert_values(
            "the sliding key history",
            &dims,
            &[1, 1, 3, 1],
            &values,
            &[5.0, 6.0, 7.0],
        );
        let LayerKvCache::Sliding(sliding) = &mut cache else {
            unreachable!()
        };
        sliding.restore(checkpoint);
        assert_eq!(sliding.logical_len(), 5);
        let (values, dims) = sliding_key_history(sliding)?;
        assert_values(
            "the restored sliding key history",
            &dims,
            &[1, 1, 3, 1],
            &values,
            &[2.0, 3.0, 4.0],
        );

        let LayerKvCache::Sliding(sliding) = &mut cache else {
            unreachable!()
        };
        sliding.append(&failed_step, &failed_step)?;
        assert_eq!(sliding.logical_len(), 8);
        let (values, dims) = sliding_key_history(sliding)?;
        assert_values(
            "the sliding key history",
            &dims,
            &[1, 1, 3, 1],
            &values,
            &[5.0, 6.0, 7.0],
        );
        Ok(())
    }

    #[test]
    fn sliding_cache_clear_after_wrap_matches_a_fresh_cache() -> Result<()> {
        let device = Device::Cpu;
        let mut reused = SlidingKvCache::new(4, 32)?;
        for (start, len) in [(0, 3), (3, 2), (5, 1), (6, 1)] {
            let old = token_values(start, len, &device)?;
            reused.append(&old, &old)?;
        }
        let (values, dims) = sliding_key_history(&reused)?;
        assert_values(
            "the reused sliding key history",
            &dims,
            &[1, 1, 3, 1],
            &values,
            &[4.0, 5.0, 6.0],
        );
        assert_eq!(reused.history_segments()?.len(), 2);
        reused.clear();
        assert!(reused.is_empty());
        assert_eq!(reused.stats().allocated_capacity, 6);

        let next = token_values(20, 4, &device)?;
        reused.append(&next, &next)?;
        let mut fresh = SlidingKvCache::new(4, 32)?;
        fresh.append(&next, &next)?;
        let (reused_values, reused_dims) = sliding_key_history(&reused)?;
        let (fresh_values, fresh_dims) = sliding_key_history(&fresh)?;
        assert_values(
            "the reused sliding key history",
            &reused_dims,
            &fresh_dims,
            &reused_values,
            &fresh_values,
        );
        assert_eq!(reused.logical_len(), fresh.logical_len());
        Ok(())
    }

    #[test]
    fn window_one_tracks_absolute_length_without_allocating_history() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = SlidingKvCache::new(1, 8)?;
        let tokens = token_values(0, 5, &device)?;
        cache.append(&tokens, &tokens)?;
        assert_eq!(cache.logical_len(), 5);
        assert!(cache.history_segments()?.is_empty());
        assert_eq!(cache.stats().resident_len, 0);
        assert_eq!(cache.stats().allocated_capacity, 0);
        assert_eq!(cache.stats().physical_capacity, 0);
        let overflow = token_values(5, 4, &device)?;
        assert!(cache.append(&overflow, &overflow).is_err());
        assert_eq!(cache.logical_len(), 5);
        Ok(())
    }

    #[test]
    fn window_two_uses_a_two_slot_ring_and_one_resident_token() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = SlidingKvCache::new(2, 16)?;
        for position in 0..7 {
            let token = token_values(position, 1, &device)?;
            cache.append(&token, &token)?;
            assert_eq!(cache.logical_len(), position + 1);
            assert_eq!(cache.stats().resident_start, position);
            assert_eq!(cache.stats().resident_len, 1);
            assert_eq!(cache.stats().allocated_capacity, 2);
            let (values, dims) = sliding_key_history(&cache)?;
            assert_values(
                "the sliding key history",
                &dims,
                &[1, 1, 1, 1],
                &values,
                &[position as f32],
            );
        }
        Ok(())
    }
}
