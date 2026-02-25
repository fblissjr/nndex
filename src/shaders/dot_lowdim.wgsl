// Low-dimension dot product shader (dims <= 1024).
// One thread per matrix row; uses vec4 loads and shared-memory query caching.
// Supports batched queries via gid.y.
// 4x loop unrolling for improved instruction-level parallelism.

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

var<workgroup> shared_query: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let row = gid.x;
    let query_idx = gid.y;

    if query_idx >= params.query_count {
        return;
    }

    let dv4 = params.dims_vec4;
    let query_base = query_idx * dv4;

    // Cooperatively load the query vector into workgroup shared memory.
    var i = lid;
    loop {
        if i >= dv4 {
            break;
        }
        shared_query[i] = queries[query_base + i];
        i += 256u;
    }
    workgroupBarrier();

    if row >= params.rows {
        return;
    }

    let matrix_base = row * dv4;

    // 4x unrolled accumulation with independent dependency chains.
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let dv4_aligned = dv4 & ~3u;
    for (var col = 0u; col < dv4_aligned; col += 4u) {
        let mb = matrix_base + col;
        acc0 = fma(matrix[mb], shared_query[col], acc0);
        acc1 = fma(matrix[mb + 1u], shared_query[col + 1u], acc1);
        acc2 = fma(matrix[mb + 2u], shared_query[col + 2u], acc2);
        acc3 = fma(matrix[mb + 3u], shared_query[col + 3u], acc3);
    }

    // Remainder iterations (0-3 vec4s).
    for (var col = dv4_aligned; col < dv4; col += 1u) {
        acc0 = fma(matrix[matrix_base + col], shared_query[col], acc0);
    }

    let combined = (acc0 + acc1) + (acc2 + acc3);
    output_scores[query_idx * params.rows + row] = dot(combined, vec4<f32>(1.0));
}
