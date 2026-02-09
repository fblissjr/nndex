/// Controls whether index-level query-result and constructor caches are active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CachePolicy {
    /// Cache hits return cloned results, avoiding recomputation.
    Enabled,
    /// Every search performs a fresh computation (useful for benchmarking).
    Disabled,
}

impl CachePolicy {
    /// Convert a boolean flag into the corresponding policy variant.
    #[inline]
    pub(crate) fn from_enabled(enabled: bool) -> Self {
        if enabled {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }

    #[cfg(feature = "gpu")]
    #[inline]
    pub(crate) fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// Strategy used for exact single-query CPU search, selected by [`TuningProfile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CpuSearchStrategy {
    /// Single-threaded linear scan.
    Serial,
    /// Parallel dot products into a shared scores buffer, then sequential top-k selection.
    ParallelScores,
    /// Matrix split into fixed-size chunks with per-chunk top-k merge.
    ParallelChunked,
    /// Rayon fold/reduce over individual rows.
    ParallelFold,
}

/// Strategy used for exact CPU batch search, selected by matrix and query shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CpuBatchStrategy {
    /// Chunk the matrix for L2-cache locality; all queries scan each chunk together.
    MatrixChunked,
    /// Parallelize across queries, each performing a serial matrix scan.
    QueryParallel,
    /// Parallelize across queries, each using its own tuning-profiled search strategy.
    ProfiledPerQuery,
}

/// Runtime tuning profile selected from matrix/query shape.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TuningProfile {
    pub(crate) cpu_parallel_cutoff_rows: usize,
    pub(crate) cache_policy: CachePolicy,
}

impl TuningProfile {
    /// Select parallelism thresholds based on total workload (rows * dims * queries).
    pub(crate) fn for_shape(
        rows: usize,
        dims: usize,
        query_rows: usize,
        cache_policy: CachePolicy,
    ) -> Self {
        let workload = rows.saturating_mul(dims).saturating_mul(query_rows.max(1));
        if workload >= 200_000_000 {
            return Self {
                cpu_parallel_cutoff_rows: 1_024,
                cache_policy,
            };
        }
        if workload >= 20_000_000 {
            return Self {
                cpu_parallel_cutoff_rows: 2_048,
                cache_policy,
            };
        }

        Self {
            cpu_parallel_cutoff_rows: 8_192,
            cache_policy,
        }
    }

    #[cfg(feature = "gpu")]
    #[inline]
    pub(crate) fn cache_enabled(self) -> bool {
        self.cache_policy.is_enabled()
    }

    /// Choose a single-query CPU search strategy from row count and parallelism thresholds.
    #[inline]
    pub(crate) fn cpu_search_strategy(
        self,
        rows: usize,
        parallel_cutoff_rows: usize,
        chunk_rows: usize,
        scores_vec_max_rows: usize,
    ) -> CpuSearchStrategy {
        if rows <= self.cpu_parallel_cutoff_rows.min(parallel_cutoff_rows) {
            return CpuSearchStrategy::Serial;
        }
        if rows <= scores_vec_max_rows {
            return CpuSearchStrategy::ParallelScores;
        }
        if chunk_rows >= 128 {
            return CpuSearchStrategy::ParallelChunked;
        }
        CpuSearchStrategy::ParallelFold
    }

    /// Choose a batch CPU search strategy from matrix size and query count.
    #[inline]
    pub(crate) fn cpu_batch_strategy(
        rows: usize,
        dims: usize,
        query_rows: usize,
        prefer_query_parallel: bool,
    ) -> CpuBatchStrategy {
        let matrix_bytes = rows
            .saturating_mul(dims)
            .saturating_mul(std::mem::size_of::<f32>());
        if query_rows >= 8 && matrix_bytes > 64 * 1024 * 1024 {
            return CpuBatchStrategy::MatrixChunked;
        }
        if prefer_query_parallel {
            return CpuBatchStrategy::QueryParallel;
        }
        CpuBatchStrategy::ProfiledPerQuery
    }
}

/// Returns `true` when the matrix is large enough for ANN prefiltering to outperform
/// full GPU exact search.
#[cfg(feature = "gpu")]
#[inline]
pub(crate) fn should_use_gpu_ann_prefilter(rows: usize, dims: usize) -> bool {
    rows >= 4_096 && dims >= 8
}

/// Returns `true` when GPU exact dispatch is expected to be faster than CPU-side ANN
/// for a batch, based on the number of required sub-batch dispatches.
#[cfg(feature = "gpu")]
#[inline]
pub(crate) fn prefer_gpu_exact_for_batch(
    query_count: usize,
    max_queries_per_sub_batch: usize,
    matrix_chunks: usize,
) -> bool {
    let sub_batches = query_count.div_ceil(max_queries_per_sub_batch.max(1));
    sub_batches.saturating_mul(matrix_chunks) <= 128
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- TuningProfile::for_shape ----

    #[test]
    fn for_shape_aggressive_on_large_workload() {
        // 200M+ workload -> cutoff 1024
        let profile = TuningProfile::for_shape(10_000, 100, 200, CachePolicy::Enabled);
        assert_eq!(profile.cpu_parallel_cutoff_rows, 1_024);
    }

    #[test]
    fn for_shape_moderate_on_medium_workload() {
        // 20M-200M workload -> cutoff 2048
        let profile = TuningProfile::for_shape(1_000, 100, 200, CachePolicy::Enabled);
        assert_eq!(profile.cpu_parallel_cutoff_rows, 2_048);
    }

    #[test]
    fn for_shape_conservative_on_small_workload() {
        // <20M workload -> cutoff 8192
        let profile = TuningProfile::for_shape(100, 32, 1, CachePolicy::Enabled);
        assert_eq!(profile.cpu_parallel_cutoff_rows, 8_192);
    }

    // ---- cpu_search_strategy ----

    #[test]
    fn search_strategy_serial_for_small_row_count() {
        let profile = TuningProfile::for_shape(100, 32, 1, CachePolicy::Enabled);
        let strategy = profile.cpu_search_strategy(100, 500, 128, 4_096);
        assert_eq!(strategy, CpuSearchStrategy::Serial);
    }

    #[test]
    fn search_strategy_parallel_scores_for_medium_row_count() {
        let profile = TuningProfile::for_shape(100_000, 768, 1, CachePolicy::Enabled);
        let strategy = profile.cpu_search_strategy(3_000, 500, 128, 4_096);
        assert_eq!(strategy, CpuSearchStrategy::ParallelScores);
    }

    #[test]
    fn search_strategy_parallel_chunked_when_large_chunks() {
        let profile = TuningProfile::for_shape(100_000, 768, 1, CachePolicy::Enabled);
        let strategy = profile.cpu_search_strategy(10_000, 500, 256, 4_096);
        assert_eq!(strategy, CpuSearchStrategy::ParallelChunked);
    }

    #[test]
    fn search_strategy_parallel_fold_when_small_chunks() {
        let profile = TuningProfile::for_shape(100_000, 768, 1, CachePolicy::Enabled);
        let strategy = profile.cpu_search_strategy(10_000, 500, 64, 4_096);
        assert_eq!(strategy, CpuSearchStrategy::ParallelFold);
    }

    // ---- cpu_batch_strategy ----

    #[test]
    fn batch_strategy_matrix_chunked_for_large_matrix_many_queries() {
        // Matrix > 64MB with >= 8 queries
        let strategy = TuningProfile::cpu_batch_strategy(100_000, 768, 10, true);
        assert_eq!(strategy, CpuBatchStrategy::MatrixChunked);
    }

    #[test]
    fn batch_strategy_query_parallel_when_preferred() {
        let strategy = TuningProfile::cpu_batch_strategy(1_000, 32, 10, true);
        assert_eq!(strategy, CpuBatchStrategy::QueryParallel);
    }

    #[test]
    fn batch_strategy_profiled_per_query_as_fallback() {
        let strategy = TuningProfile::cpu_batch_strategy(1_000, 32, 2, false);
        assert_eq!(strategy, CpuBatchStrategy::ProfiledPerQuery);
    }

    // ---- CachePolicy ----

    #[test]
    fn cache_policy_from_bool() {
        assert_eq!(CachePolicy::from_enabled(true), CachePolicy::Enabled);
        assert_eq!(CachePolicy::from_enabled(false), CachePolicy::Disabled);
    }
}
