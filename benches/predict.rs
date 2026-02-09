mod common;

use std::sync::atomic::{AtomicUsize, Ordering};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nndex::{BackendPreference, IndexOptions, NNdex};

use common::{
    PREDICT_BENCH_SIZES, bench_cache_enabled, criterion_with_runtime_defaults, generate_matrix,
    generate_queries, should_run_case,
};

fn bench_predict_cpu(criterion: &mut criterion::Criterion) {
    let mut group = criterion.benchmark_group("predict/cpu");
    for &(rows, dims) in PREDICT_BENCH_SIZES {
        if !should_run_case(rows, dims) {
            continue;
        }

        let matrix = generate_matrix(rows, dims, 0xA11CE + rows as u64 + dims as u64);
        let query = generate_queries(1, dims, 0xC0FFEE + rows as u64);
        let index = NNdex::new(
            &matrix,
            rows,
            dims,
            IndexOptions {
                normalized: false,
                approx: false,
                backend: BackendPreference::Cpu,
                enable_cache: bench_cache_enabled(),
            },
        )
        .expect("CPU index construction should succeed");

        group.bench_with_input(
            BenchmarkId::new("rows_dims", format!("{rows}x{dims}")),
            &dims,
            |bench, _| {
                bench.iter(|| {
                    let result =
                        index.search(std::hint::black_box(&query), std::hint::black_box(10));
                    std::hint::black_box(result.expect("CPU search should succeed"));
                });
            },
        );
    }
    group.finish();
}

fn bench_predict_cpu_ann(criterion: &mut criterion::Criterion) {
    let mut group = criterion.benchmark_group("predict_ann/cpu");
    for &(rows, dims) in PREDICT_BENCH_SIZES {
        if !should_run_case(rows, dims) {
            continue;
        }

        let matrix = generate_matrix(rows, dims, 0xBEEF + rows as u64 + dims as u64);
        let query_bank = generate_queries(64, dims, 0xD00D + rows as u64 + dims as u64);
        let next_query = AtomicUsize::new(0);
        let index = NNdex::new(
            &matrix,
            rows,
            dims,
            IndexOptions {
                normalized: false,
                approx: true,
                backend: BackendPreference::Cpu,
                enable_cache: bench_cache_enabled(),
            },
        )
        .expect("CPU ANN index construction should succeed");

        group.bench_with_input(
            BenchmarkId::new("rows_dims", format!("{rows}x{dims}")),
            &dims,
            |bench, _| {
                bench.iter(|| {
                    let query_index = next_query.fetch_add(1, Ordering::Relaxed) % 64;
                    let start = query_index * dims;
                    let end = start + dims;
                    let result = index.search(
                        std::hint::black_box(&query_bank[start..end]),
                        std::hint::black_box(10),
                    );
                    std::hint::black_box(result.expect("CPU ANN search should succeed"));
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_predict_gpu(criterion: &mut criterion::Criterion) {
    let mut group = criterion.benchmark_group("predict/gpu");
    for &(rows, dims) in PREDICT_BENCH_SIZES {
        if !should_run_case(rows, dims) {
            continue;
        }

        let matrix = generate_matrix(rows, dims, 0xBADC0DE + rows as u64 + dims as u64);
        let query = generate_queries(1, dims, 0xDEADBEEF + rows as u64);
        let index = NNdex::new(
            &matrix,
            rows,
            dims,
            IndexOptions {
                normalized: false,
                approx: false,
                backend: BackendPreference::Gpu,
                enable_cache: bench_cache_enabled(),
            },
        )
        .expect("GPU index construction should succeed");

        group.bench_with_input(
            BenchmarkId::new("rows_dims", format!("{rows}x{dims}")),
            &dims,
            |bench, _| {
                bench.iter(|| {
                    let result =
                        index.search(std::hint::black_box(&query), std::hint::black_box(10));
                    std::hint::black_box(result.expect("GPU search should succeed"));
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_predict_gpu_ann(criterion: &mut criterion::Criterion) {
    let mut group = criterion.benchmark_group("predict_ann/gpu");
    for &(rows, dims) in PREDICT_BENCH_SIZES {
        if !should_run_case(rows, dims) {
            continue;
        }

        let matrix = generate_matrix(rows, dims, 0x1CEB00DA + rows as u64 + dims as u64);
        let query_bank = generate_queries(64, dims, 0x1CEDFACE + rows as u64 + dims as u64);
        let next_query = AtomicUsize::new(0);
        let index = NNdex::new(
            &matrix,
            rows,
            dims,
            IndexOptions {
                normalized: false,
                approx: true,
                backend: BackendPreference::Gpu,
                enable_cache: bench_cache_enabled(),
            },
        )
        .expect("GPU ANN index construction should succeed");

        group.bench_with_input(
            BenchmarkId::new("rows_dims", format!("{rows}x{dims}")),
            &dims,
            |bench, _| {
                bench.iter(|| {
                    let query_index = next_query.fetch_add(1, Ordering::Relaxed) % 64;
                    let start = query_index * dims;
                    let end = start + dims;
                    let result = index.search(
                        std::hint::black_box(&query_bank[start..end]),
                        std::hint::black_box(10),
                    );
                    std::hint::black_box(result.expect("GPU ANN search should succeed"));
                });
            },
        );
    }
    group.finish();
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
