#include <metal_stdlib>

using namespace metal;

struct GatedDeltaDecayParams {
    uint count;
    uint heads;
};

kernel void gated_delta_decay_f32(
    device const float* alpha [[buffer(0)]],
    device const float* time_bias [[buffer(1)]],
    device const float* state_decay [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant GatedDeltaDecayParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= params.count) {
        return;
    }
    const uint head = index % params.heads;
    const float shifted = alpha[index] + time_bias[head];
    const float softplus = max(shifted, 0.0f) + log(1.0f + exp(-abs(shifted)));
    output[index] = exp(softplus * state_decay[head]);
}

struct GatedDeltaRecurrenceParams {
    uint tokens;
    uint heads;
    uint key_dim;
    uint value_dim;
    uint state_elements;
};

kernel void gated_delta_decode_recurrence_f32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* recurrent [[buffer(2)]],
    device const float* value [[buffer(3)]],
    device const float* decay [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant GatedDeltaRecurrenceParams& params [[buffer(7)]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint head = group_position.x;
    if (head >= params.heads) {
        return;
    }
    const ulong state_base = ulong(head) * params.key_dim * params.value_dim;
    for (uint value_index = thread_index; value_index < params.value_dim;
         value_index += 256) {
        for (uint token = 0; token < params.tokens; ++token) {
            const ulong token_head = ulong(token) * params.heads + head;
            const ulong key_base = token_head * params.key_dim;
            const ulong value_base = token_head * params.value_dim;
            const float head_decay = decay[token_head];
            const float head_beta = beta[token_head];
            float memory = 0.0f;
            for (uint key_index = 0; key_index < params.key_dim; ++key_index) {
                const ulong state_index =
                    state_base + ulong(key_index) * params.value_dim + value_index;
                const float previous = token == 0
                    ? recurrent[state_index]
                    : output[state_index];
                memory += previous * head_decay * key[key_base + key_index];
            }
            const float delta =
                (value[value_base + value_index] - memory) * head_beta;
            float projected = 0.0f;
            for (uint key_index = 0; key_index < params.key_dim; ++key_index) {
                const ulong state_index =
                    state_base + ulong(key_index) * params.value_dim + value_index;
                const float previous = token == 0
                    ? recurrent[state_index]
                    : output[state_index];
                const float next = previous * head_decay
                    + key[key_base + key_index] * delta;
                output[state_index] = next;
                projected += next * query[key_base + key_index];
            }
            output[params.state_elements + value_base + value_index] = projected;
        }
    }
}
