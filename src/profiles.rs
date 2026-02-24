/// Controls whether index-level query-result and constructor caches are active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CachePolicy {
    /// Cache hits return cloned results, avoiding recomputation.
    Enabled,
    /// Every search performs a fresh computation.
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
    /// BLAS `sgemv` (matrix-vector multiply) to compute all dot products in one call,
    /// then O(n) top-k extraction. Leverages BLAS cache-optimal tiling for matrices
    /// where the workload justifies the call overhead.
    Gemv,
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
    /// Single GEMM via ndarray: scores = queries * matrix^T, then parallel top-k extraction.
    /// Leverages cache-optimal tiling in the GEMM kernel (BLAS or matrixmultiply fallback).
    Gemm,
    /// GEMM applied in chunks when the full score matrix exceeds the memory budget but
    /// GEMM is still profitable. Splits the stored matrix into ~64 MB chunks, runs GEMM
    /// per chunk, and merges per-query top-k results.
    GemmChunked,
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
    ///
    /// When BLAS is available, uses `sgemv` for matrices that would otherwise use
    /// the ParallelScores or Serial strategies, provided dimensionality is high
    /// enough for BLAS cache-optimal tiling to outperform row-by-row SIMD.
    /// For larger matrices (beyond the scores-vec threshold), rayon parallelism
    /// with chunking outperforms single-threaded BLAS.
    #[inline]
    pub(crate) fn cpu_search_strategy(
        self,
        rows: usize,
        dims: usize,
        parallel_cutoff_rows: usize,
        chunk_rows: usize,
        scores_vec_max_rows: usize,
        has_blas: bool,
    ) -> CpuSearchStrategy {
        let effective_cutoff = self.cpu_parallel_cutoff_rows.min(parallel_cutoff_rows);
        if rows <= effective_cutoff {
            // BLAS sgemv outperforms serial row-by-row SIMD for high-dimensional
            // vectors because it uses cache-optimal tiling.
            if has_blas && dims >= 64 && rows >= 128 {
                return CpuSearchStrategy::Gemv;
            }
            return CpuSearchStrategy::Serial;
        }
        // For moderate row counts where a single scores vector fits in memory,
        // BLAS sgemv outperforms parallel SIMD when dims are high enough that
        // tiling dominates over parallelism overhead.
        if rows <= scores_vec_max_rows {
            if has_blas && dims >= 64 {
                return CpuSearchStrategy::Gemv;
            }
            return CpuSearchStrategy::ParallelScores;
        }
        if chunk_rows >= 128 {
            return CpuSearchStrategy::ParallelChunked;
        }
        CpuSearchStrategy::ParallelFold
    }

    /// Choose a batch CPU search strategy from matrix size and query count.
    ///
    /// Prefers GEMM (matrix multiplication) for batches where the GEMM kernel's
    /// cache-optimal tiling outperforms row-by-row dot products. Uses graduated
    /// thresholds: higher dims need fewer queries to justify the GEMM overhead.
    #[inline]
    pub(crate) fn cpu_batch_strategy(
        rows: usize,
        dims: usize,
        query_rows: usize,
        prefer_query_parallel: bool,
    ) -> CpuBatchStrategy {
        // Graduated GEMM eligibility: higher dims need fewer queries because
        // GEMM's cache-tiled inner loop amortizes better over longer vectors.
        let gemm_eligible = (dims >= 64 && query_rows >= 4)
            || (dims >= 32 && query_rows >= 8)
            || (dims >= 16 && query_rows >= 16);

        if gemm_eligible {
            let score_matrix_bytes = query_rows
                .saturating_mul(rows)
                .saturating_mul(std::mem::size_of::<f32>());
            // Full GEMM when score matrix fits in 128 MB.
            if score_matrix_bytes <= 128 * 1024 * 1024 {
                return CpuBatchStrategy::Gemm;
            }
            // Chunked GEMM when score matrix is too large but GEMM is still profitable.
            return CpuBatchStrategy::GemmChunked;
        }

        let matrix_bytes = rows
            .saturating_mul(dims)
            .saturating_mul(std::mem::size_of::<f32>());
        if query_rows >= 4 && matrix_bytes > 64 * 1024 * 1024 {
            return CpuBatchStrategy::MatrixChunked;
        }
        if prefer_query_parallel {
            return CpuBatchStrategy::QueryParallel;
        }
        CpuBatchStrategy::ProfiledPerQuery
    }
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
        // rows=100 < 128 minimum for Gemv, stays Serial even with BLAS
        let strategy = profile.cpu_search_strategy(100, 128, 500, 128, 4_096, true);
        assert_eq!(strategy, CpuSearchStrategy::Serial);
    }

    #[test]
    fn search_strategy_gemv_in_serial_regime_high_dims() {
        let profile = TuningProfile::for_shape(1_000, 768, 1, CachePolicy::Enabled);
        // rows=1000 <= cutoff, dims=768 >= 64, rows >= 128 => Gemv
        let strategy = profile.cpu_search_strategy(1_000, 768, 10_000, 128, 4_096, true);
        assert_eq!(strategy, CpuSearchStrategy::Gemv);
    }

    #[test]
    fn search_strategy_gemv_replaces_parallel_scores_for_high_dims() {
        let profile = TuningProfile::for_shape(100_000, 768, 1, CachePolicy::Enabled);
        // rows=3000 > cutoff=1024 but <= scores_vec_max_rows=4096, dims >= 64 => Gemv
        let strategy = profile.cpu_search_strategy(3_000, 768, 500, 128, 4_096, true);
        assert_eq!(strategy, CpuSearchStrategy::Gemv);
    }

    #[test]
    fn search_strategy_parallel_scores_for_low_dims() {
        let profile = TuningProfile::for_shape(100_000, 48, 1, CachePolicy::Enabled);
        // rows > cutoff, dims=48 < 64 => ParallelScores (no Gemv)
        let strategy = profile.cpu_search_strategy(3_000, 48, 500, 128, 4_096, true);
        assert_eq!(strategy, CpuSearchStrategy::ParallelScores);
    }

    #[test]
    fn search_strategy_gemv_skipped_without_blas() {
        let profile = TuningProfile::for_shape(1_000, 768, 1, CachePolicy::Enabled);
        // has_blas=false => Serial
        let strategy = profile.cpu_search_strategy(1_000, 768, 10_000, 128, 4_096, false);
        assert_eq!(strategy, CpuSearchStrategy::Serial);
    }

    #[test]
    fn search_strategy_gemv_skipped_for_low_dims_serial() {
        let profile = TuningProfile::for_shape(1_000, 32, 1, CachePolicy::Enabled);
        // dims=32 < 64 => Serial
        let strategy = profile.cpu_search_strategy(1_000, 32, 10_000, 128, 4_096, true);
        assert_eq!(strategy, CpuSearchStrategy::Serial);
    }

    #[test]
    fn search_strategy_parallel_chunked_when_large_rows() {
        let profile = TuningProfile::for_shape(100_000, 768, 1, CachePolicy::Enabled);
        // rows=10000 > scores_vec_max=4096 => bypasses Gemv, uses ParallelChunked
        let strategy = profile.cpu_search_strategy(10_000, 768, 500, 256, 4_096, true);
        assert_eq!(strategy, CpuSearchStrategy::ParallelChunked);
    }

    #[test]
    fn search_strategy_parallel_fold_when_small_chunks() {
        let profile = TuningProfile::for_shape(100_000, 768, 1, CachePolicy::Enabled);
        let strategy = profile.cpu_search_strategy(10_000, 768, 500, 64, 4_096, true);
        assert_eq!(strategy, CpuSearchStrategy::ParallelFold);
    }

    // ---- cpu_batch_strategy ----

    #[test]
    fn batch_strategy_gemm_for_high_dim_batch() {
        // dims >= 64, query_rows >= 4, score matrix within memory limit
        let strategy = TuningProfile::cpu_batch_strategy(18_181, 640, 96, true);
        assert_eq!(strategy, CpuBatchStrategy::Gemm);
    }

    #[test]
    fn batch_strategy_gemm_for_medium_dims_many_queries() {
        // dims >= 32, query_rows >= 8: graduated threshold triggers GEMM
        let strategy = TuningProfile::cpu_batch_strategy(10_000, 32, 128, true);
        assert_eq!(strategy, CpuBatchStrategy::Gemm);
    }

    #[test]
    fn batch_strategy_gemm_for_low_dims_high_queries() {
        // dims >= 16, query_rows >= 16: graduated threshold triggers GEMM
        let strategy = TuningProfile::cpu_batch_strategy(5_000, 24, 20, true);
        assert_eq!(strategy, CpuBatchStrategy::Gemm);
    }

    #[test]
    fn batch_strategy_gemm_skipped_for_very_low_dims() {
        // dims < 16: GEMM threshold not met regardless of query count
        let strategy = TuningProfile::cpu_batch_strategy(1_000, 8, 128, true);
        assert_eq!(strategy, CpuBatchStrategy::QueryParallel);
    }

    #[test]
    fn batch_strategy_gemm_chunked_for_huge_score_matrix() {
        // Score matrix would exceed 128 MB (1M * 100 * 4 = 400 MB) but GEMM eligible
        let strategy = TuningProfile::cpu_batch_strategy(1_000_000, 128, 100, true);
        assert_eq!(strategy, CpuBatchStrategy::GemmChunked);
    }

    #[test]
    fn batch_strategy_matrix_chunked_for_large_matrix_low_dims() {
        // Matrix > 64MB with >= 4 queries, dims < 16 so GEMM is skipped
        let strategy = TuningProfile::cpu_batch_strategy(10_000_000, 8, 5, true);
        assert_eq!(strategy, CpuBatchStrategy::MatrixChunked);
    }

    #[test]
    fn batch_strategy_query_parallel_when_preferred() {
        // Low dims, few queries, prefer_query_parallel = true
        let strategy = TuningProfile::cpu_batch_strategy(1_000, 8, 10, true);
        assert_eq!(strategy, CpuBatchStrategy::QueryParallel);
    }

    #[test]
    fn batch_strategy_profiled_per_query_as_fallback() {
        let strategy = TuningProfile::cpu_batch_strategy(1_000, 8, 2, false);
        assert_eq!(strategy, CpuBatchStrategy::ProfiledPerQuery);
    }

    // ---- CachePolicy ----

    #[test]
    fn cache_policy_from_bool() {
        assert_eq!(CachePolicy::from_enabled(true), CachePolicy::Enabled);
        assert_eq!(CachePolicy::from_enabled(false), CachePolicy::Disabled);
    }
}
