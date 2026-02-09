// High-dimension dot product shader (dims > 1024).
// One workgroup (256 threads) per matrix row; cooperative reduction via shared memory.
// Supports batched queries via workgroup_id.y.

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

    // Each thread accumulates a strided slice of the dot product.
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var col = lid;
    loop {
        if col >= dv4 {
            break;
        }
        acc = fma(matrix[matrix_base + col], queries[query_base + col], acc);
        col += 256u;
    }

    partial_sums[lid] = acc.x + acc.y + acc.z + acc.w;
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
