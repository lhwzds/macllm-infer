#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

#define MAX_SLABS 1024
#define THREADS 256
#define SIMD_WIDTH 32
#define KEY_TILE (THREADS / SIMD_WIDTH)
#define MAX_HEAD_DIM 256
#define VALUES_PER_LANE (MAX_HEAD_DIM / SIMD_WIDTH)
#define GQA_TILE 4
#define GQA_THREADS 128
#define GQA_KEY_GROUPS (GQA_THREADS / SIMD_WIDTH)
#define GQA_MAX_HEAD_DIM 128
#define GQA_VALUES_PER_LANE (GQA_MAX_HEAD_DIM / SIMD_WIDTH)

struct PagedAttentionParams {
    uint logical_len;
    uint query_start;
    uint query_len;
    uint page_offset;
    uint page_count;
    uint block_tokens;
    uint full_layers;
    uint layer;
    uint kv_heads;
    uint query_heads;
    uint head_dim;
    uint groups;
    uint slab_count;
    float scale;
    float softcap;
};

struct BatchedPagedAttentionParams {
    uint query_len;
    uint block_tokens;
    uint full_layers;
    uint layer;
    uint kv_heads;
    uint query_heads;
    uint head_dim;
    uint groups;
    uint slab_count;
    uint lane_count;
    uint max_query_len;
    float scale;
    float softcap;
};

struct PagedAttentionLane {
    uint logical_len;
    uint query_start;
    uint query_offset;
    uint query_len;
    uint page_offset;
    uint page_count;
};

struct KvSlabsF32 {
    array<device const float*, MAX_SLABS> keys [[id(0)]];
    array<device const float*, MAX_SLABS> values [[id(MAX_SLABS)]];
};

struct KvSlabsF16 {
    array<device const half*, MAX_SLABS> keys [[id(0)]];
    array<device const half*, MAX_SLABS> values [[id(MAX_SLABS)]];
};

#define PAGED_ATTENTION_KERNEL(NAME, SCALAR, SLABS)                                   \
kernel void NAME(                                                                     \
    device const SCALAR* query [[buffer(0)]],                                          \
    device const uint2* page_table [[buffer(1)]],                                      \
    constant SLABS& slabs [[buffer(2)]],                                               \
    device SCALAR* output [[buffer(3)]],                                               \
    constant PagedAttentionParams& params [[buffer(4)]],                               \
    uint3 group_position [[threadgroup_position_in_grid]],                             \
    uint thread_index [[thread_index_in_threadgroup]],                                 \
    uint simd_group [[simdgroup_index_in_threadgroup]],                                \
    uint simd_lane [[thread_index_in_simdgroup]]) {                                    \
    threadgroup float scratch[KEY_TILE * MAX_HEAD_DIM];                                \
    const uint query_row = group_position.x;                                           \
    const uint query_head = group_position.y;                                          \
    const uint kv_head = query_head / params.groups;                                   \
    const uint visible_len = min(                                                      \
        params.logical_len, params.query_start + query_row + 1);                       \
    const ulong query_base =                                                          \
        (ulong(query_head) * params.query_len + query_row) * params.head_dim;          \
    float accumulated[VALUES_PER_LANE];                                                \
    for (uint slot = 0; slot < VALUES_PER_LANE; ++slot) {                              \
        accumulated[slot] = 0.0f;                                                     \
    }                                                                                 \
    float local_maximum = -INFINITY;                                                   \
    float local_normalizer = 0.0f;                                                    \
    for (uint position = simd_group; position < visible_len; position += KEY_TILE) {   \
        const uint page = position / params.block_tokens;                              \
        const uint in_page = position % params.block_tokens;                           \
        uint2 page_ref = uint2(0);                                                     \
        bool page_valid = false;                                                       \
        if (page < params.page_count) {                                                \
            page_ref = page_table[params.page_offset + page];                          \
            page_valid = page_ref.x < params.slab_count;                               \
        }                                                                             \
        float partial = 0.0f;                                                         \
        ulong value_base = 0;                                                          \
        if (page_valid) {                                                              \
            const device SCALAR* key = slabs.keys[page_ref.x];                         \
            value_base =                                                              \
                (((ulong(page_ref.y) * params.full_layers + params.layer)              \
                   * params.kv_heads + kv_head)                                        \
                  * params.block_tokens + in_page)                                     \
                * params.head_dim;                                                     \
            for (uint dimension = simd_lane; dimension < params.head_dim;              \
                 dimension += SIMD_WIDTH) {                                            \
                partial += float(query[query_base + dimension])                        \
                    * float(key[value_base + dimension]);                              \
            }                                                                         \
        }                                                                             \
        partial = simd_sum(partial);                                                     \
        float score = page_valid ? partial * params.scale : -INFINITY;                 \
        if (page_valid && params.softcap > 0.0f) {                                     \
            score = tanh(score / params.softcap) * params.softcap;                     \
        }                                                                             \
        const float next_maximum = max(local_maximum, score);                          \
        const float old_weight = isfinite(local_maximum)                               \
            ? exp(local_maximum - next_maximum)                                        \
            : 0.0f;                                                                    \
        const float new_weight = page_valid                                            \
            ? exp(score - next_maximum)                                                \
            : 0.0f;                                                                    \
        for (uint slot = 0; slot < VALUES_PER_LANE; ++slot) {                          \
            const uint dimension = simd_lane + slot * SIMD_WIDTH;                      \
            if (dimension < params.head_dim) {                                         \
                const float value = page_valid                                         \
                    ? float(slabs.values[page_ref.x][value_base + dimension])           \
                    : 0.0f;                                                            \
                accumulated[slot] = accumulated[slot] * old_weight                     \
                    + value * new_weight;                                              \
            }                                                                         \
        }                                                                             \
        local_normalizer = local_normalizer * old_weight + new_weight;                 \
        local_maximum = next_maximum;                                                  \
    }                                                                                   \
    if (simd_lane == 0) {                                                              \
        scratch[simd_group] = local_maximum;                                           \
        scratch[KEY_TILE + simd_group] = local_normalizer;                             \
    }                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    float global_maximum = -INFINITY;                                                  \
    for (uint group = 0; group < KEY_TILE; ++group) {                                  \
        global_maximum = max(global_maximum, scratch[group]);                          \
    }                                                                                 \
    float global_normalizer = 0.0f;                                                   \
    for (uint group = 0; group < KEY_TILE; ++group) {                                  \
        const float group_maximum = scratch[group];                                    \
        const float group_weight = isfinite(group_maximum)                             \
            ? exp(group_maximum - global_maximum)                                      \
            : 0.0f;                                                                    \
        global_normalizer += scratch[KEY_TILE + group] * group_weight;                 \
    }                                                                                 \
    const float group_weight = isfinite(local_maximum)                                 \
        ? exp(local_maximum - global_maximum)                                          \
        : 0.0f;                                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    for (uint slot = 0; slot < VALUES_PER_LANE; ++slot) {                              \
        const uint dimension = simd_lane + slot * SIMD_WIDTH;                          \
        if (dimension < params.head_dim) {                                             \
            scratch[simd_group * MAX_HEAD_DIM + dimension] =                           \
                accumulated[slot] * group_weight;                                      \
        }                                                                             \
    }                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    if (thread_index < params.head_dim) {                                               \
        float combined = 0.0f;                                                        \
        for (uint group = 0; group < KEY_TILE; ++group) {                              \
            combined += scratch[group * MAX_HEAD_DIM + thread_index];                  \
        }                                                                             \
        const ulong output_index =                                                     \
            (ulong(query_head) * params.query_len + query_row) * params.head_dim        \
            + thread_index;                                                            \
        output[output_index] = SCALAR(combined / global_normalizer);                   \
    }                                                                                   \
}

PAGED_ATTENTION_KERNEL(paged_attention_f32, float, KvSlabsF32)
PAGED_ATTENTION_KERNEL(paged_attention_f16, half, KvSlabsF16)

#define BATCHED_PAGED_ATTENTION_KERNEL(NAME, SCALAR, SLABS)                           \
kernel void NAME(                                                                     \
    device const SCALAR* query [[buffer(0)]],                                          \
    device const uint2* page_table [[buffer(1)]],                                      \
    constant SLABS& slabs [[buffer(2)]],                                               \
    device SCALAR* output [[buffer(3)]],                                               \
    constant BatchedPagedAttentionParams& params [[buffer(4)]],                       \
    constant PagedAttentionLane* lanes [[buffer(5)]],                                 \
    uint3 group_position [[threadgroup_position_in_grid]],                             \
    uint thread_index [[thread_index_in_threadgroup]],                                 \
    uint simd_group [[simdgroup_index_in_threadgroup]],                                \
    uint simd_lane [[thread_index_in_simdgroup]]) {                                    \
    const uint lane_index = group_position.z;                                          \
    const uint lane_query_row = group_position.x;                                      \
    if (lane_index >= params.lane_count) {                                             \
        return;                                                                        \
    }                                                                                 \
    const PagedAttentionLane lane = lanes[lane_index];                                 \
    if (lane_query_row >= lane.query_len) {                                            \
        return;                                                                        \
    }                                                                                 \
    threadgroup float scratch[KEY_TILE * MAX_HEAD_DIM];                                \
    const uint query_head = group_position.y;                                          \
    const uint kv_head = query_head / params.groups;                                   \
    const uint visible_len = min(                                                      \
        lane.logical_len, lane.query_start + lane_query_row + 1);                      \
    const uint query_row = lane.query_offset + lane_query_row;                         \
    const ulong query_base =                                                          \
        (ulong(query_head) * params.query_len + query_row) * params.head_dim;           \
    float accumulated[VALUES_PER_LANE];                                                \
    for (uint slot = 0; slot < VALUES_PER_LANE; ++slot) {                              \
        accumulated[slot] = 0.0f;                                                     \
    }                                                                                 \
    float local_maximum = -INFINITY;                                                   \
    float local_normalizer = 0.0f;                                                    \
    for (uint position = simd_group; position < visible_len; position += KEY_TILE) {   \
        const uint page = position / params.block_tokens;                              \
        const uint in_page = position % params.block_tokens;                           \
        uint2 page_ref = uint2(0);                                                     \
        bool page_valid = false;                                                       \
        if (page < lane.page_count) {                                                  \
            page_ref = page_table[lane.page_offset + page];                            \
            page_valid = page_ref.x < params.slab_count;                               \
        }                                                                             \
        float partial = 0.0f;                                                         \
        ulong value_base = 0;                                                          \
        if (page_valid) {                                                              \
            const device SCALAR* key = slabs.keys[page_ref.x];                         \
            value_base =                                                              \
                (((ulong(page_ref.y) * params.full_layers + params.layer)              \
                   * params.kv_heads + kv_head)                                        \
                  * params.block_tokens + in_page)                                     \
                * params.head_dim;                                                     \
            for (uint dimension = simd_lane; dimension < params.head_dim;              \
                 dimension += SIMD_WIDTH) {                                            \
                partial += float(query[query_base + dimension])                        \
                    * float(key[value_base + dimension]);                              \
            }                                                                         \
        }                                                                             \
        partial = simd_sum(partial);                                                   \
        float score = page_valid ? partial * params.scale : -INFINITY;                 \
        if (page_valid && params.softcap > 0.0f) {                                     \
            score = tanh(score / params.softcap) * params.softcap;                     \
        }                                                                             \
        const float next_maximum = max(local_maximum, score);                          \
        const float old_weight = isfinite(local_maximum)                               \
            ? exp(local_maximum - next_maximum)                                        \
            : 0.0f;                                                                    \
        const float new_weight = page_valid                                            \
            ? exp(score - next_maximum)                                                \
            : 0.0f;                                                                    \
        for (uint slot = 0; slot < VALUES_PER_LANE; ++slot) {                          \
            const uint dimension = simd_lane + slot * SIMD_WIDTH;                      \
            if (dimension < params.head_dim) {                                         \
                const float value = page_valid                                         \
                    ? float(slabs.values[page_ref.x][value_base + dimension])           \
                    : 0.0f;                                                            \
                accumulated[slot] = accumulated[slot] * old_weight                     \
                    + value * new_weight;                                              \
            }                                                                         \
        }                                                                             \
        local_normalizer = local_normalizer * old_weight + new_weight;                 \
        local_maximum = next_maximum;                                                  \
    }                                                                                 \
    if (simd_lane == 0) {                                                              \
        scratch[simd_group] = local_maximum;                                           \
        scratch[KEY_TILE + simd_group] = local_normalizer;                             \
    }                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    float global_maximum = -INFINITY;                                                  \
    for (uint group = 0; group < KEY_TILE; ++group) {                                  \
        global_maximum = max(global_maximum, scratch[group]);                          \
    }                                                                                 \
    float global_normalizer = 0.0f;                                                   \
    for (uint group = 0; group < KEY_TILE; ++group) {                                  \
        const float group_maximum = scratch[group];                                    \
        const float group_weight = isfinite(group_maximum)                             \
            ? exp(group_maximum - global_maximum)                                      \
            : 0.0f;                                                                    \
        global_normalizer += scratch[KEY_TILE + group] * group_weight;                 \
    }                                                                                 \
    const float group_weight = isfinite(local_maximum)                                 \
        ? exp(local_maximum - global_maximum)                                          \
        : 0.0f;                                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    for (uint slot = 0; slot < VALUES_PER_LANE; ++slot) {                              \
        const uint dimension = simd_lane + slot * SIMD_WIDTH;                          \
        if (dimension < params.head_dim) {                                             \
            scratch[simd_group * MAX_HEAD_DIM + dimension] =                           \
                accumulated[slot] * group_weight;                                      \
        }                                                                             \
    }                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    if (thread_index < params.head_dim) {                                              \
        float combined = 0.0f;                                                        \
        for (uint group = 0; group < KEY_TILE; ++group) {                              \
            combined += scratch[group * MAX_HEAD_DIM + thread_index];                  \
        }                                                                             \
        const ulong output_index =                                                     \
            (ulong(query_head) * params.query_len + query_row) * params.head_dim        \
            + thread_index;                                                            \
        output[output_index] = SCALAR(combined / global_normalizer);                   \
    }                                                                                 \
}

BATCHED_PAGED_ATTENTION_KERNEL(batched_paged_attention_f32, float, KvSlabsF32)
BATCHED_PAGED_ATTENTION_KERNEL(batched_paged_attention_f16, half, KvSlabsF16)

// SmolLM3 maps four contiguous query heads to every K/V head. The generic
// packed kernel above assigns one threadgroup to each query head, so it reads
// the same 16K K/V stream four times. This specialization assigns one
// threadgroup to a K/V head and keeps four independent online-softmax states
// while sharing every K/V load. Query length one intentionally remains on the
// generic path; host-side routing selects this kernel only for chunks >= 4.
//
// Scratch is dynamically sized by the host. The first
// `GQA_KEY_GROUPS * GQA_TILE * 2` floats hold (maximum, normalizer) pairs. The
// remainder holds one normalized output partial per key-partition/query-head.
#define BATCHED_PAGED_ATTENTION_GQA4_KERNEL(NAME, SCALAR, SLABS)                      \
kernel void NAME(                                                                     \
    device const SCALAR* query [[buffer(0)]],                                          \
    device const uint2* page_table [[buffer(1)]],                                      \
    constant SLABS& slabs [[buffer(2)]],                                               \
    device SCALAR* output [[buffer(3)]],                                               \
    constant BatchedPagedAttentionParams& params [[buffer(4)]],                       \
    constant PagedAttentionLane* lanes [[buffer(5)]],                                 \
    threadgroup float* scratch [[threadgroup(0)]],                                    \
    uint3 group_position [[threadgroup_position_in_grid]],                             \
    uint thread_index [[thread_index_in_threadgroup]],                                 \
    uint simd_group [[simdgroup_index_in_threadgroup]],                                \
    uint simd_lane [[thread_index_in_simdgroup]]) {                                    \
    const uint lane_index = group_position.z;                                          \
    const uint lane_query_row = group_position.x;                                      \
    if (lane_index >= params.lane_count) {                                             \
        return;                                                                        \
    }                                                                                 \
    const PagedAttentionLane lane = lanes[lane_index];                                 \
    if (lane_query_row >= lane.query_len) {                                            \
        return;                                                                        \
    }                                                                                 \
    const uint kv_head = group_position.y;                                             \
    const uint query_row = lane.query_offset + lane_query_row;                         \
    const uint visible_len = min(                                                      \
        lane.logical_len, lane.query_start + lane_query_row + 1);                      \
    float query_values[GQA_TILE][GQA_VALUES_PER_LANE];                                \
    float accumulated[GQA_TILE][GQA_VALUES_PER_LANE];                                 \
    float local_maximum[GQA_TILE];                                                     \
    float local_normalizer[GQA_TILE];                                                  \
    for (uint query_group = 0; query_group < GQA_TILE; ++query_group) {                \
        const uint query_head = kv_head * GQA_TILE + query_group;                      \
        const ulong query_base =                                                      \
            (ulong(query_head) * params.query_len + query_row) * params.head_dim;      \
        local_maximum[query_group] = -INFINITY;                                        \
        local_normalizer[query_group] = 0.0f;                                         \
        for (uint slot = 0; slot < GQA_VALUES_PER_LANE; ++slot) {                      \
            const uint dimension = simd_lane + slot * SIMD_WIDTH;                      \
            query_values[query_group][slot] = dimension < params.head_dim             \
                ? float(query[query_base + dimension])                                 \
                : 0.0f;                                                               \
            accumulated[query_group][slot] = 0.0f;                                    \
        }                                                                             \
    }                                                                                 \
    const uint visible_pages =                                                        \
        (visible_len + params.block_tokens - 1) / params.block_tokens;                 \
    for (uint page = 0; page < visible_pages; ++page) {                                \
        if (page >= lane.page_count) {                                                 \
            continue;                                                                 \
        }                                                                             \
        const uint2 page_ref = page_table[lane.page_offset + page];                    \
        if (page_ref.x >= params.slab_count) {                                         \
            continue;                                                                 \
        }                                                                             \
        const uint page_start = page * params.block_tokens;                            \
        const uint visible_in_page = min(params.block_tokens, visible_len - page_start); \
        const device SCALAR* key = slabs.keys[page_ref.x];                             \
        const device SCALAR* value = slabs.values[page_ref.x];                         \
        for (uint in_page = simd_group; in_page < visible_in_page;                     \
             in_page += GQA_KEY_GROUPS) {                                              \
            const ulong value_base =                                                  \
                (((ulong(page_ref.y) * params.full_layers + params.layer)              \
                   * params.kv_heads + kv_head)                                        \
                  * params.block_tokens + in_page)                                     \
                * params.head_dim;                                                     \
            float key_values[GQA_VALUES_PER_LANE];                                    \
            float value_values[GQA_VALUES_PER_LANE];                                  \
            for (uint slot = 0; slot < GQA_VALUES_PER_LANE; ++slot) {                  \
                const uint dimension = simd_lane + slot * SIMD_WIDTH;                  \
                key_values[slot] = dimension < params.head_dim                        \
                    ? float(key[value_base + dimension])                               \
                    : 0.0f;                                                           \
                value_values[slot] = dimension < params.head_dim                      \
                    ? float(value[value_base + dimension])                             \
                    : 0.0f;                                                           \
            }                                                                         \
            for (uint query_group = 0; query_group < GQA_TILE; ++query_group) {        \
                float partial = 0.0f;                                                  \
                for (uint slot = 0; slot < GQA_VALUES_PER_LANE; ++slot) {              \
                    partial += query_values[query_group][slot] * key_values[slot];      \
                }                                                                     \
                partial = simd_sum(partial);                                           \
                float score = partial * params.scale;                                 \
                if (params.softcap > 0.0f) {                                           \
                    score = tanh(score / params.softcap) * params.softcap;             \
                }                                                                     \
                const float next_maximum = max(local_maximum[query_group], score);     \
                const float old_weight = isfinite(local_maximum[query_group])          \
                    ? exp(local_maximum[query_group] - next_maximum)                    \
                    : 0.0f;                                                           \
                const float new_weight = exp(score - next_maximum);                    \
                for (uint slot = 0; slot < GQA_VALUES_PER_LANE; ++slot) {              \
                    accumulated[query_group][slot] =                                   \
                        accumulated[query_group][slot] * old_weight                    \
                        + value_values[slot] * new_weight;                             \
                }                                                                     \
                local_normalizer[query_group] =                                       \
                    local_normalizer[query_group] * old_weight + new_weight;           \
                local_maximum[query_group] = next_maximum;                             \
            }                                                                         \
        }                                                                             \
    }                                                                                 \
    constexpr uint stats_count = GQA_KEY_GROUPS * GQA_TILE * 2;                       \
    if (simd_lane == 0) {                                                              \
        for (uint query_group = 0; query_group < GQA_TILE; ++query_group) {            \
            const uint stats = (simd_group * GQA_TILE + query_group) * 2;              \
            scratch[stats] = local_maximum[query_group];                              \
            scratch[stats + 1] = local_normalizer[query_group];                       \
        }                                                                             \
    }                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    for (uint query_group = 0; query_group < GQA_TILE; ++query_group) {                \
        float global_maximum = -INFINITY;                                              \
        for (uint group = 0; group < GQA_KEY_GROUPS; ++group) {                       \
            const uint stats = (group * GQA_TILE + query_group) * 2;                   \
            global_maximum = max(global_maximum, scratch[stats]);                     \
        }                                                                             \
        float global_normalizer = 0.0f;                                                \
        for (uint group = 0; group < GQA_KEY_GROUPS; ++group) {                       \
            const uint stats = (group * GQA_TILE + query_group) * 2;                   \
            const float group_maximum = scratch[stats];                               \
            const float group_weight = isfinite(group_maximum)                        \
                ? exp(group_maximum - global_maximum)                                  \
                : 0.0f;                                                               \
            global_normalizer += scratch[stats + 1] * group_weight;                   \
        }                                                                             \
        const float group_weight = isfinite(local_maximum[query_group])                \
            ? exp(local_maximum[query_group] - global_maximum)                         \
                / global_normalizer                                                    \
            : 0.0f;                                                                   \
        for (uint slot = 0; slot < GQA_VALUES_PER_LANE; ++slot) {                      \
            const uint dimension = simd_lane + slot * SIMD_WIDTH;                      \
            if (dimension < params.head_dim) {                                         \
                const ulong partial_index = stats_count                               \
                    + (ulong(simd_group * GQA_TILE + query_group) * params.head_dim)   \
                    + dimension;                                                       \
                scratch[partial_index] =                                               \
                    accumulated[query_group][slot] * group_weight;                    \
            }                                                                         \
        }                                                                             \
    }                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    const uint output_elements = GQA_TILE * params.head_dim;                           \
    for (uint flat = thread_index; flat < output_elements; flat += GQA_THREADS) {      \
        const uint query_group = flat / params.head_dim;                              \
        const uint dimension = flat % params.head_dim;                                \
        float combined = 0.0f;                                                        \
        for (uint group = 0; group < GQA_KEY_GROUPS; ++group) {                       \
            const ulong partial_index = stats_count                                   \
                + (ulong(group * GQA_TILE + query_group) * params.head_dim)            \
                + dimension;                                                          \
            combined += scratch[partial_index];                                       \
        }                                                                             \
        const uint query_head = kv_head * GQA_TILE + query_group;                      \
        const ulong output_index =                                                    \
            (ulong(query_head) * params.query_len + query_row) * params.head_dim       \
            + dimension;                                                              \
        output[output_index] = SCALAR(combined);                                       \
    }                                                                                 \
}

BATCHED_PAGED_ATTENTION_GQA4_KERNEL(batched_paged_attention_gqa4_f32, float, KvSlabsF32)
BATCHED_PAGED_ATTENTION_GQA4_KERNEL(batched_paged_attention_gqa4_f16, half, KvSlabsF16)
