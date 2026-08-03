#include <metal_stdlib>

using namespace metal;

struct Mamba2ScanParams {
    uint tokens;
    uint heads;
    uint head_dim;
    uint state_size;
    uint heads_per_group;
    uint inner_size;
    uint group_state_width;
    uint xbc_width;
    uint state_elements;
    uint threadgroup_width;
};

kernel void mamba2_scan_f32(
    device const float* xbc [[buffer(0)]],
    device const float* raw_dt [[buffer(1)]],
    device const float* initial_state [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device const float* a [[buffer(4)]],
    device const float* d [[buffer(5)]],
    device float* output_and_state [[buffer(6)]],
    constant Mamba2ScanParams& params [[buffer(7)]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint lane = group_position.x;
    if (lane >= params.inner_size || thread_index >= params.threadgroup_width) {
        return;
    }

    const uint head = lane / params.head_dim;
    const uint group = head / params.heads_per_group;
    const ulong state_base = ulong(lane) * params.state_size;
    threadgroup float partials[256];

    for (uint state_index = thread_index; state_index < params.state_size;
         state_index += params.threadgroup_width) {
        output_and_state[state_base + state_index] =
            initial_state[state_base + state_index];
    }

    for (uint token = 0; token < params.tokens; ++token) {
        const ulong projection_base = ulong(token) * params.xbc_width;
        const ulong b_base = projection_base + params.inner_size
            + ulong(group) * params.state_size;
        const ulong c_base = projection_base + params.inner_size
            + params.group_state_width + ulong(group) * params.state_size;
        const ulong token_head = ulong(token) * params.heads + head;
        const float shifted = raw_dt[token_head] + dt_bias[head];
        const float delta = max(shifted, 0.0f)
            + log(1.0f + exp(-abs(shifted)));
        const float decay = exp(delta * a[head]);
        const float value = xbc[projection_base + lane];
        float partial = 0.0f;
        for (uint state_index = thread_index; state_index < params.state_size;
             state_index += params.threadgroup_width) {
            const ulong recurrent_index = state_base + state_index;
            const float next = decay * output_and_state[recurrent_index]
                + delta * value * xbc[b_base + state_index];
            output_and_state[recurrent_index] = next;
            partial += next * xbc[c_base + state_index];
        }
        partials[thread_index] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = params.threadgroup_width / 2; stride > 0; stride /= 2) {
            if (thread_index < stride) {
                partials[thread_index] += partials[thread_index + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_index == 0) {
            output_and_state[params.state_elements + ulong(token) * params.inner_size + lane] =
                partials[0] + d[head] * value;
        }
    }
}
