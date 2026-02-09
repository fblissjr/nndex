#![allow(dead_code)]

use std::time::Duration;

use criterion::Criterion;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub const PREDICT_BENCH_SIZES: &[(usize, usize)] = &[
    (1_000, 48),
    (1_000, 768),
    (10_000, 48),
    (10_000, 768),
    (50_000, 128),
    (100_000, 2),
    (1_000, 196_608),
    (10_000_000, 24),
];

pub fn criterion_with_runtime_defaults() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4))
        .sample_size(10)
}

pub fn generate_matrix(rows: usize, dims: usize, seed: u64) -> Vec<f32> {
    let len = rows.saturating_mul(dims);
    let bar = ProgressBar::new(len as u64);
    bar.set_style(
        ProgressStyle::with_template("{msg} {wide_bar} {pos}/{len}")
            .expect("progress style template should be valid"),
    );
    bar.set_message(format!("Generating matrix {rows}x{dims}"));

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(len);
    const CHUNK: usize = 16_384;

    while out.len() < len {
        let remaining = len - out.len();
        let batch = remaining.min(CHUNK);
        out.extend((0..batch).map(|_| rng.random_range(-1.0_f32..1.0_f32)));
        bar.inc(batch as u64);
    }

    bar.finish_and_clear();
    out
}

pub fn generate_queries(rows: usize, dims: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..rows.saturating_mul(dims))
        .map(|_| rng.random_range(-1.0_f32..1.0_f32))
        .collect()
}

pub fn maybe_extreme_enabled() -> bool {
    benchmark_flag("NNDEX_BENCH_EXTREME", "FASTDOT_BENCH_EXTREME")
}

pub fn should_run_case(rows: usize, dims: usize) -> bool {
    if maybe_extreme_enabled() {
        return true;
    }

    rows.saturating_mul(dims) <= 50_000_000
}

/// Returns `false` when `NNDEX_BENCH_NOCACHE=1` is set, signalling that
/// internal query-result caches should be disabled for accurate benchmarking.
pub fn bench_cache_enabled() -> bool {
    !benchmark_flag("NNDEX_BENCH_NOCACHE", "FASTDOT_BENCH_NOCACHE")
}

fn benchmark_flag(primary: &str, legacy: &str) -> bool {
    [primary, legacy]
        .into_iter()
        .find_map(|name| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse::<u8>().ok())
        })
        .unwrap_or(0)
        == 1
}
