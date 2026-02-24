mod common;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nndex::{BackendPreference, IndexOptions, NNdex};

use common::{
    PREDICT_BENCH_SIZES, criterion_with_runtime_defaults, generate_matrix, maybe_extreme_enabled,
    should_run_case,
};

/// Scale GPU fit rows to reduce fixed dispatch overhead influence in benchmark timings.
const GPU_FIT_ROW_MULTIPLIER: usize = 100;
/// Default safety cap for total matrix elements in scaled GPU fit benchmarks.
const GPU_FIT_MAX_ELEMENTS: usize = 100_000_000;

fn scaled_gpu_fit_rows(rows: usize, dims: usize) -> Option<usize> {
    let scaled_rows = rows.checked_mul(GPU_FIT_ROW_MULTIPLIER)?;
    if maybe_extreme_enabled() {
        return Some(scaled_rows);
    }

    let total_elements = scaled_rows.checked_mul(dims)?;
    (total_elements <= GPU_FIT_MAX_ELEMENTS).then_some(scaled_rows)
}

fn sanitize_zero_norm_rows(matrix: &mut [f32], dims: usize) {
    for row in matrix.chunks_exact_mut(dims) {
        let norm_sq: f32 = row.iter().map(|value| value * value).sum();
        if norm_sq <= f32::EPSILON {
            // Keep benchmark construction stable by avoiding invalid all-zero input rows.
            row[0] = 1.0;
        }
    }
}

fn bench_fit_cpu(criterion: &mut criterion::Criterion) {
    let mut group = criterion.benchmark_group("fit/cpu");
    for &(rows, dims) in PREDICT_BENCH_SIZES {
        if !should_run_case(rows, dims) {
            continue;
        }

        let mut matrix = generate_matrix(rows, dims, rows as u64 ^ ((dims as u64) << 16));
        sanitize_zero_norm_rows(&mut matrix, dims);
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
                            gpu_device_index: None,
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
    for &(base_rows, dims) in PREDICT_BENCH_SIZES {
        let Some(rows) = scaled_gpu_fit_rows(base_rows, dims) else {
            continue;
        };

        let mut matrix =
            generate_matrix(rows, dims, (rows as u64).wrapping_add((dims as u64) << 24));
        sanitize_zero_norm_rows(&mut matrix, dims);
        group.bench_with_input(
            BenchmarkId::new("rows_dims", format!("{rows}x{dims}/base_rows={base_rows}")),
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
                            gpu_device_index: None,
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
