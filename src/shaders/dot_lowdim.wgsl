// Low-dimension dot product shader (dims <= 1024).
// One thread per matrix row; uses vec4 loads and shared-memory query caching.
// Supports batched queries via gid.y.

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

    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var col = 0u; col < dv4; col += 1u) {
        acc = fma(matrix[matrix_base + col], shared_query[col], acc);
    }

    let score = acc.x + acc.y + acc.z + acc.w;
    output_scores[query_idx * params.rows + row] = score;
}
