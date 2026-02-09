mod common;

use std::sync::atomic::{AtomicUsize, Ordering};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nndex::{BackendPreference, IndexOptions, NNdex};

use common::{
    bench_cache_enabled, criterion_with_runtime_defaults, generate_matrix, generate_queries,
};

const MATRIX_ROWS: usize = 10_000_000;
const MATRIX_DIMS: usize = 24;
const QUERY_BATCHES: &[usize] = &[1, 4, 128, 2_048, 16_384];

fn matrix_rows() -> usize {
    benchmark_env("NNDEX_BENCH_BATCH_ROWS", "FASTDOT_BENCH_BATCH_ROWS")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(MATRIX_ROWS)
}

fn query_batches() -> Vec<usize> {
    benchmark_env("NNDEX_BENCH_QUERY_BATCHES", "FASTDOT_BENCH_QUERY_BATCHES")
        .map(|value| {
            value
                .split(',')
                .filter_map(|batch| batch.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|batches| !batches.is_empty())
        .unwrap_or_else(|| QUERY_BATCHES.to_vec())
}

fn benchmark_env(primary: &str, legacy: &str) -> Option<String> {
    std::env::var(primary)
        .ok()
        .or_else(|| std::env::var(legacy).ok())
}

fn bench_batch_cpu(criterion: &mut criterion::Criterion) {
    let rows = matrix_rows();
    let matrix = generate_matrix(rows, MATRIX_DIMS, 0xFACEFEED);
    let index = NNdex::new(
        &matrix,
        rows,
        MATRIX_DIMS,
        IndexOptions {
            normalized: false,
            approx: false,
            backend: BackendPreference::Cpu,
            enable_cache: bench_cache_enabled(),
        },
    )
    .expect("CPU index construction should succeed");

    let mut group = criterion.benchmark_group("batch/cpu");
    for query_rows in query_batches() {
        let queries = generate_queries(query_rows, MATRIX_DIMS, 0xABCD_0000 + query_rows as u64);
        group.bench_with_input(
            BenchmarkId::new("query_rows", query_rows),
            &query_rows,
            |bench, _| {
                bench.iter(|| {
                    let result = index.search_batch(
                        std::hint::black_box(&queries),
                        std::hint::black_box(query_rows),
                        std::hint::black_box(10),
                    );
                    std::hint::black_box(result.expect("CPU batch search should succeed"));
                });
            },
        );
    }
    group.finish();
}

fn bench_batch_cpu_ann(criterion: &mut criterion::Criterion) {
    let rows = matrix_rows();
    let matrix = generate_matrix(rows, MATRIX_DIMS, 0xFA11BA11);
    let index = NNdex::new(
        &matrix,
        rows,
        MATRIX_DIMS,
        IndexOptions {
            normalized: false,
            approx: true,
            backend: BackendPreference::Cpu,
            enable_cache: bench_cache_enabled(),
        },
    )
    .expect("CPU ANN index construction should succeed");

    let mut group = criterion.benchmark_group("batch_ann/cpu");
    for query_rows in query_batches() {
        let query_bank_rows = query_rows * 8;
        let query_bank = generate_queries(
            query_bank_rows,
            MATRIX_DIMS,
            0xAA00_0000 + query_rows as u64,
        );
        let next_batch = AtomicUsize::new(0);
        group.bench_with_input(
            BenchmarkId::new("query_rows", query_rows),
            &query_rows,
            |bench, _| {
                bench.iter(|| {
                    let batch_id = next_batch.fetch_add(1, Ordering::Relaxed) % 8;
                    let start = batch_id * query_rows * MATRIX_DIMS;
                    let end = start + query_rows * MATRIX_DIMS;
                    let result = index.search_batch(
                        std::hint::black_box(&query_bank[start..end]),
                        std::hint::black_box(query_rows),
                        std::hint::black_box(10),
                    );
                    std::hint::black_box(result.expect("CPU ANN batch search should succeed"));
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_batch_gpu(criterion: &mut criterion::Criterion) {
    let rows = matrix_rows();
    let matrix = generate_matrix(rows, MATRIX_DIMS, 0xF00DFACE);
    let index = NNdex::new(
        &matrix,
        rows,
        MATRIX_DIMS,
        IndexOptions {
            normalized: false,
            approx: false,
            backend: BackendPreference::Gpu,
            enable_cache: bench_cache_enabled(),
        },
    )
    .expect("GPU index construction should succeed");

    let mut group = criterion.benchmark_group("batch/gpu");
    for query_rows in query_batches() {
        let queries = generate_queries(query_rows, MATRIX_DIMS, 0x1234_0000 + query_rows as u64);
        group.bench_with_input(
            BenchmarkId::new("query_rows", query_rows),
            &query_rows,
            |bench, _| {
                bench.iter(|| {
                    let result = index.search_batch(
                        std::hint::black_box(&queries),
                        std::hint::black_box(query_rows),
                        std::hint::black_box(10),
                    );
                    std::hint::black_box(result.expect("GPU batch search should succeed"));
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_batch_gpu_ann(criterion: &mut criterion::Criterion) {
    let rows = matrix_rows();
    let matrix = generate_matrix(rows, MATRIX_DIMS, 0xCA11AB1E);
    let index = NNdex::new(
        &matrix,
        rows,
        MATRIX_DIMS,
        IndexOptions {
            normalized: false,
            approx: true,
            backend: BackendPreference::Gpu,
            enable_cache: bench_cache_enabled(),
        },
    )
    .expect("GPU ANN index construction should succeed");

    let mut group = criterion.benchmark_group("batch_ann/gpu");
    for query_rows in query_batches() {
        let query_bank_rows = query_rows * 8;
        let query_bank = generate_queries(
            query_bank_rows,
            MATRIX_DIMS,
            0xBB00_0000 + query_rows as u64,
        );
        let next_batch = AtomicUsize::new(0);
        group.bench_with_input(
            BenchmarkId::new("query_rows", query_rows),
            &query_rows,
            |bench, _| {
                bench.iter(|| {
                    let batch_id = next_batch.fetch_add(1, Ordering::Relaxed) % 8;
                    let start = batch_id * query_rows * MATRIX_DIMS;
                    let end = start + query_rows * MATRIX_DIMS;
                    let result = index.search_batch(
                        std::hint::black_box(&query_bank[start..end]),
                        std::hint::black_box(query_rows),
                        std::hint::black_box(10),
                    );
                    std::hint::black_box(result.expect("GPU ANN batch search should succeed"));
                });
            },
        );
    }
    group.finish();
}

fn criterion_benches(criterion: &mut Criterion) {
    let mut configured = criterion_with_runtime_defaults().configure_from_args();
    let _ = criterion;
    bench_batch_cpu(&mut configured);
    bench_batch_cpu_ann(&mut configured);
    #[cfg(feature = "gpu")]
    {
        bench_batch_gpu(&mut configured);
        bench_batch_gpu_ann(&mut configured);
    }
}

criterion_group!(benches, criterion_benches);
criterion_main!(benches);
