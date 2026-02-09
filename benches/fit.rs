mod common;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nndex::{BackendPreference, IndexOptions, NNdex};

use common::{
    PREDICT_BENCH_SIZES, criterion_with_runtime_defaults, generate_matrix, should_run_case,
};

fn bench_fit_cpu(criterion: &mut criterion::Criterion) {
    let mut group = criterion.benchmark_group("fit/cpu");
    for &(rows, dims) in PREDICT_BENCH_SIZES {
        if !should_run_case(rows, dims) {
            continue;
        }

        let matrix = generate_matrix(rows, dims, rows as u64 ^ ((dims as u64) << 16));
        group.bench_with_input(
            BenchmarkId::new("rows_dims", format!("{rows}x{dims}")),
            &(rows, dims),
            |bench, &(rows, dims)| {
                bench.iter(|| {
                    let index = NNdex::new(
                        &matrix,
                        rows,
                        dims,
                        IndexOptions {
                            normalized: false,
                            approx: false,
                            backend: BackendPreference::Cpu,
                            // Fit benchmarks should measure true construction cost.
                            enable_cache: false,
                        },
                    )
                    .expect("CPU fit should succeed");
                    std::hint::black_box(index);
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_fit_gpu(criterion: &mut criterion::Criterion) {
    let mut group = criterion.benchmark_group("fit/gpu");
    for &(rows, dims) in PREDICT_BENCH_SIZES {
        if !should_run_case(rows, dims) {
            continue;
        }

        let matrix = generate_matrix(rows, dims, (rows as u64).wrapping_add((dims as u64) << 24));
        group.bench_with_input(
            BenchmarkId::new("rows_dims", format!("{rows}x{dims}")),
            &(rows, dims),
            |bench, &(rows, dims)| {
                bench.iter(|| {
                    let index = NNdex::new(
                        &matrix,
                        rows,
                        dims,
                        IndexOptions {
                            normalized: false,
                            approx: false,
                            backend: BackendPreference::Gpu,
                            // Fit benchmarks should measure true construction cost.
                            enable_cache: false,
                        },
                    )
                    .expect("GPU fit should succeed");
                    std::hint::black_box(index);
                });
            },
        );
    }
    group.finish();
}

fn criterion_benches(criterion: &mut Criterion) {
    let mut configured = criterion_with_runtime_defaults().configure_from_args();
    let _ = criterion;
    bench_fit_cpu(&mut configured);
    #[cfg(feature = "gpu")]
    bench_fit_gpu(&mut configured);
}

criterion_group!(benches, criterion_benches);
criterion_main!(benches);
