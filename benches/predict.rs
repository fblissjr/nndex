mod common;

use std::sync::atomic::{AtomicUsize, Ordering};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nndex::{BackendPreference, IndexOptions, NNdex};

use common::{
    PREDICT_BENCH_SIZES, criterion_with_runtime_defaults, generate_matrix, generate_queries,
    should_run_case,
};

const QUERY_BANK_ROWS: usize = 64;
const TOP_K_VALUES: &[usize] = &[1, 10];

fn bench_predict_backend(
    criterion: &mut criterion::Criterion,
    group_name: &str,
    backend: BackendPreference,
    approx: bool,
    matrix_seed_salt: u64,
    query_seed_salt: u64,
    expect_label: &str,
) {
    let mut group = criterion.benchmark_group(group_name);
    for &(rows, dims) in PREDICT_BENCH_SIZES {
        if !should_run_case(rows, dims) {
            continue;
        }

        let matrix = generate_matrix(rows, dims, matrix_seed_salt + rows as u64 + dims as u64);
        let query_bank = generate_queries(
            QUERY_BANK_ROWS,
            dims,
            query_seed_salt + rows as u64 + dims as u64,
        );
        let next_query = AtomicUsize::new(0);
        let index = NNdex::new(
            &matrix,
            rows,
            dims,
            IndexOptions {
                normalized: false,
                approx,
                backend,
                enable_cache: false,
                gpu_device_index: None,
            },
        )
        .expect(expect_label);

        for &top_k in TOP_K_VALUES {
            group.bench_with_input(
                BenchmarkId::new("rows_dims_k", format!("{rows}x{dims}/k={top_k}")),
                &top_k,
                |bench, top_k| {
                    bench.iter(|| {
                        let query_index =
                            next_query.fetch_add(1, Ordering::Relaxed) % QUERY_BANK_ROWS;
                        let start = query_index * dims;
                        let end = start + dims;
                        let result = index.search(
                            std::hint::black_box(&query_bank[start..end]),
                            std::hint::black_box(*top_k),
                        );
                        std::hint::black_box(result.expect(expect_label));
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_predict_cpu(criterion: &mut criterion::Criterion) {
    bench_predict_backend(
        criterion,
        "predict/cpu",
        BackendPreference::Cpu,
        false,
        0xA11CE,
        0xC0FFEE,
        "CPU search should succeed",
    );
}

fn bench_predict_cpu_ann(criterion: &mut criterion::Criterion) {
    bench_predict_backend(
        criterion,
        "predict_ann/cpu",
        BackendPreference::Cpu,
        true,
        0xA11CE,
        0xC0FFEE,
        "CPU ANN search should succeed",
    );
}

#[cfg(feature = "gpu")]
fn bench_predict_gpu(criterion: &mut criterion::Criterion) {
    bench_predict_backend(
        criterion,
        "predict/gpu",
        BackendPreference::Gpu,
        false,
        0xBADC0DE,
        0xDEADBEEF,
        "GPU search should succeed",
    );
}

#[cfg(feature = "gpu")]
fn bench_predict_gpu_ann(criterion: &mut criterion::Criterion) {
    bench_predict_backend(
        criterion,
        "predict_ann/gpu",
        BackendPreference::Gpu,
        true,
        0xBADC0DE,
        0xDEADBEEF,
        "GPU ANN search should succeed",
    );
}

fn criterion_benches(criterion: &mut Criterion) {
    let mut configured = criterion_with_runtime_defaults().configure_from_args();
    let _ = criterion;
    bench_predict_cpu(&mut configured);
    bench_predict_cpu_ann(&mut configured);
    #[cfg(feature = "gpu")]
    {
        bench_predict_gpu(&mut configured);
        bench_predict_gpu_ann(&mut configured);
    }
}

criterion_group!(benches, criterion_benches);
criterion_main!(benches);
