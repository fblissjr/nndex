// High-dimension dot product shader (dims > 1024).
// One workgroup (256 threads) per matrix row; cooperative reduction via shared memory.
// Supports batched queries via workgroup_id.y.
// 8x loop unrolling for improved instruction-level parallelism.

struct Params {
    rows: u32,
    dims_vec4: u32,
    query_count: u32,
    _pad: u32,
}

@group(0) @binding(0)
var<storage, read> matrix: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> queries: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> output_scores: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let row = wid.x;
    let query_idx = wid.y;

    if row >= params.rows || query_idx >= params.query_count {
        partial_sums[lid] = 0.0;
        workgroupBarrier();
        return;
    }

    let dv4 = params.dims_vec4;
    let matrix_base = row * dv4;
    let query_base = query_idx * dv4;

    let stride = 256u;
    let stride8 = stride * 8u;

    // 8x unrolled strided accumulation with independent dependency chains.
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0);
    var a5 = vec4<f32>(0.0);
    var a6 = vec4<f32>(0.0);
    var a7 = vec4<f32>(0.0);

    var col = lid;
    loop {
        if col + 7u * stride >= dv4 {
            break;
        }
        let c1 = col + stride;
        let c2 = col + 2u * stride;
        let c3 = col + 3u * stride;
        let c4 = col + 4u * stride;
        let c5 = col + 5u * stride;
        let c6 = col + 6u * stride;
        let c7 = col + 7u * stride;
        a0 = fma(matrix[matrix_base + col], queries[query_base + col], a0);
        a1 = fma(matrix[matrix_base + c1], queries[query_base + c1], a1);
        a2 = fma(matrix[matrix_base + c2], queries[query_base + c2], a2);
        a3 = fma(matrix[matrix_base + c3], queries[query_base + c3], a3);
        a4 = fma(matrix[matrix_base + c4], queries[query_base + c4], a4);
        a5 = fma(matrix[matrix_base + c5], queries[query_base + c5], a5);
        a6 = fma(matrix[matrix_base + c6], queries[query_base + c6], a6);
        a7 = fma(matrix[matrix_base + c7], queries[query_base + c7], a7);
        col += stride8;
    }

    // Remainder: process remaining strided elements.
    loop {
        if col >= dv4 {
            break;
        }
        a0 = fma(matrix[matrix_base + col], queries[query_base + col], a0);
        col += stride;
    }

    let combined = (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);
    partial_sums[lid] = dot(combined, vec4<f32>(1.0));
    workgroupBarrier();

    // Tree reduction across 256 threads (8 steps: 256 -> 128 -> ... -> 1).
    if lid < 128u { partial_sums[lid] += partial_sums[lid + 128u]; }
    workgroupBarrier();
    if lid < 64u { partial_sums[lid] += partial_sums[lid + 64u]; }
    workgroupBarrier();
    if lid < 32u { partial_sums[lid] += partial_sums[lid + 32u]; }
    workgroupBarrier();
    if lid < 16u { partial_sums[lid] += partial_sums[lid + 16u]; }
    workgroupBarrier();
    if lid < 8u { partial_sums[lid] += partial_sums[lid + 8u]; }
    workgroupBarrier();
    if lid < 4u { partial_sums[lid] += partial_sums[lid + 4u]; }
    workgroupBarrier();
    if lid < 2u { partial_sums[lid] += partial_sums[lid + 2u]; }
    workgroupBarrier();

    if lid == 0u {
        output_scores[query_idx * params.rows + row] = partial_sums[0] + partial_sums[1];
    }
}
