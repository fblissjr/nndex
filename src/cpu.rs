use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, Once, OnceLock};

use rayon::prelude::*;
use simsimd::SpatialSimilarity;

use crate::approx::ApproxIndex;
use crate::profiles::{CachePolicy, CpuBatchStrategy, CpuSearchStrategy, TuningProfile};
use crate::topk::{Neighbor, TopKAccumulator, topk_from_scores};

/// Maximum rows for the parallel-scores strategy (scores vector <= 1 MB).
/// Kept low so the sequential top-k pass over the scores buffer stays cheap.
const SCORES_VEC_MAX_ROWS: usize = 4_096;

/// Maximum entries retained in the single-query LRU result cache.
const SINGLE_QUERY_CACHE_CAPACITY: usize = 64;

/// Maximum entries retained in the batch-query LRU result cache.
const BATCH_QUERY_CACHE_CAPACITY: usize = 8;

/// Fixed-capacity LRU cache keyed by `u64` hashes.
///
/// Uses a `Vec` with most-recently-used items at the front. The small capacity
/// (typically <= 128) keeps linear scans fast while avoiding a heap-heavy `HashMap`.
#[derive(Debug)]
struct BoundedLruCache<T> {
    entries: Vec<(u64, T)>,
    capacity: usize,
}

impl<T> BoundedLruCache<T> {
    /// Create an empty cache with the given maximum number of entries.
    fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Insert or update an entry, promoting it to most-recently-used. Evicts the
    /// least-recently-used entry when at capacity.
    fn insert(&mut self, key: u64, value: T) {
        if let Some(index) = self
            .entries
            .iter()
            .position(|(stored_key, _)| *stored_key == key)
        {
            self.entries[index] = (key, value);
            self.entries.swap(0, index);
            return;
        }

        if self.entries.len() == self.capacity {
            let _ = self.entries.pop();
        }
        self.entries.push((key, value));
        let tail = self.entries.len().saturating_sub(1);
        self.entries.swap(0, tail);
    }
}

impl<T: Clone> BoundedLruCache<T> {
    /// Look up an entry by key, cloning the value and promoting it to most-recently-used.
    fn get_cloned(&mut self, key: u64) -> Option<T> {
        let index = self
            .entries
            .iter()
            .position(|(stored_key, _)| *stored_key == key)?;
        self.entries.swap(0, index);
        Some(self.entries[0].1.clone())
    }
}

/// Shared mutable state for caching approximate search results.
#[derive(Debug)]
struct ApproxCacheState {
    single: Mutex<BoundedLruCache<Vec<Neighbor>>>,
    batch: Mutex<BoundedLruCache<Vec<Vec<Neighbor>>>>,
}

impl Default for ApproxCacheState {
    fn default() -> Self {
        Self {
            single: Mutex::new(BoundedLruCache::with_capacity(SINGLE_QUERY_CACHE_CAPACITY)),
            batch: Mutex::new(BoundedLruCache::with_capacity(BATCH_QUERY_CACHE_CAPACITY)),
        }
    }
}

/// CPU-backed nearest-neighbor index using rayon parallelism and SIMD dot products.
#[derive(Debug)]
pub(crate) struct CpuIndex {
    matrix: Vec<f32>,
    rows: usize,
    dims: usize,
    profile: TuningProfile,
    approx_index: Option<OnceLock<ApproxIndex>>,
    approx_cache: Option<Box<ApproxCacheState>>,
}

impl CpuIndex {
    /// Build a CPU index from an already-normalized, padded row-major matrix.
    pub(crate) fn new(
        matrix: Vec<f32>,
        rows: usize,
        dims: usize,
        approx: bool,
        enable_cache: bool,
    ) -> Self {
        let use_cache = approx && enable_cache;
        Self {
            matrix,
            rows,
            dims,
            profile: TuningProfile::for_shape(
                rows,
                dims,
                1,
                CachePolicy::from_enabled(enable_cache),
            ),
            approx_index: approx.then(OnceLock::new),
            approx_cache: use_cache.then(|| Box::new(ApproxCacheState::default())),
        }
    }

    /// Find the top-k neighbors for a single query, using approximate or exact strategy.
    pub(crate) fn search(&self, query: &[f32], k: usize, approx: bool) -> Vec<Neighbor> {
        initialize_simd_runtime();
        let capped_k = k.min(self.rows);
        if approx {
            if let Some(cache) = &self.approx_cache {
                let cache_key = self.query_hash(query, capped_k, 1);
                if let Ok(mut guard) = cache.single.lock()
                    && let Some(cached_values) = guard.get_cloned(cache_key)
                {
                    return cached_values;
                }

                let result = if let Some(approx_index) = self.approx_index() {
                    self.search_approx(query, capped_k, approx_index)
                } else {
                    // Keep cache behavior in approx mode even when ANN prefilter
                    // is disabled for this shape and we fall back to exact search.
                    self.search_with_profile(query, capped_k, self.profile)
                };
                if let Ok(mut guard) = cache.single.lock() {
                    guard.insert(cache_key, result.clone());
                }
                return result;
            } else if let Some(approx_index) = self.approx_index() {
                // Cache disabled but ANN index available: search without caching.
                return self.search_approx(query, capped_k, approx_index);
            }
        }
        self.search_with_profile(query, capped_k, self.profile)
    }

    /// Find the top-k neighbors for each query in a batch.
    pub(crate) fn search_batch(
        &self,
        queries: &[f32],
        query_rows: usize,
        k: usize,
        approx: bool,
    ) -> Vec<Vec<Neighbor>> {
        let batch_profile =
            TuningProfile::for_shape(self.rows, self.dims, query_rows, self.profile.cache_policy);
        let capped_k = k.min(self.rows);
        let threads = rayon::current_num_threads().max(1);
        let prefer_query_parallel = query_rows >= threads.saturating_mul(2);

        if approx {
            if let Some(cache) = &self.approx_cache {
                let use_cache = query_rows <= 128;
                let cache_key = use_cache.then(|| self.query_hash(queries, capped_k, query_rows));
                if let Some(key) = cache_key
                    && let Ok(mut guard) = cache.batch.lock()
                    && let Some(cached_values) = guard.get_cloned(key)
                {
                    return cached_values;
                }

                let result = if let Some(approx_index) = self.approx_index() {
                    self.search_batch_approx(queries, query_rows, capped_k, approx_index)
                } else {
                    self.search_batch_exact(
                        queries,
                        query_rows,
                        capped_k,
                        batch_profile,
                        prefer_query_parallel,
                    )
                };
                if let Some(key) = cache_key
                    && let Ok(mut guard) = cache.batch.lock()
                {
                    guard.insert(key, result.clone());
                }
                return result;
            } else if let Some(approx_index) = self.approx_index() {
                return self.search_batch_approx(queries, query_rows, capped_k, approx_index);
            }
        }

        self.search_batch_exact(
            queries,
            query_rows,
            capped_k,
            batch_profile,
            prefer_query_parallel,
        )
    }

    /// Dispatch a batch of queries using the profiled exact search strategy.
    fn search_batch_exact(
        &self,
        queries: &[f32],
        query_rows: usize,
        k: usize,
        batch_profile: TuningProfile,
        prefer_query_parallel: bool,
    ) -> Vec<Vec<Neighbor>> {
        let batch_strategy = TuningProfile::cpu_batch_strategy(
            self.rows,
            self.dims,
            query_rows,
            prefer_query_parallel,
        );
        match batch_strategy {
            CpuBatchStrategy::Gemm => self.search_batch_gemm(queries, query_rows, k),
            CpuBatchStrategy::GemmChunked => self.search_batch_gemm_chunked(queries, query_rows, k),
            CpuBatchStrategy::MatrixChunked => {
                self.search_batch_matrix_chunked(queries, query_rows, k)
            }
            CpuBatchStrategy::QueryParallel => queries
                .par_chunks_exact(self.dims)
                .take(query_rows)
                .map(|query| self.search_serial(query, k))
                .collect(),
            CpuBatchStrategy::ProfiledPerQuery => queries
                .par_chunks_exact(self.dims)
                .take(query_rows)
                .map(|query| self.search_with_profile(query, k, batch_profile))
                .collect(),
        }
    }

    /// Batch search via GEMM (matrix multiplication): scores = queries * matrix^T.
    ///
    /// Computes all dot products as a single matrix multiply, then extracts top-k
    /// from each row of the resulting score matrix using O(n) introselect.
    /// Significantly faster than row-by-row dot products for high-dimensional
    /// vectors because the GEMM kernel (BLAS `sgemm` or the `matrixmultiply`
    /// fallback) uses cache-optimal tiling.
    fn search_batch_gemm(
        &self,
        queries: &[f32],
        query_rows: usize,
        k: usize,
    ) -> Vec<Vec<Neighbor>> {
        use ndarray::ArrayView2;

        let matrix_view = ArrayView2::from_shape((self.rows, self.dims), &self.matrix)
            .expect("matrix shape should match stored dimensions");

        let queries_view =
            ArrayView2::from_shape((query_rows, self.dims), &queries[..query_rows * self.dims])
                .expect("queries shape should match index dimensions");

        // Single GEMM: (query_rows, dims) x (dims, rows) => (query_rows, rows).
        // ndarray dispatches to BLAS sgemm when a blas-* feature is enabled,
        // or to matrixmultiply's cache-tiled sgemm otherwise.
        let scores = queries_view.dot(&matrix_view.t());

        let scores_slice = scores
            .as_slice()
            .expect("GEMM output should be in standard (row-major) layout");

        scores_slice
            .par_chunks_exact(self.rows)
            .map(|row_scores| topk_from_scores(row_scores, k, 0))
            .collect()
    }

    /// Batch search via chunked GEMM for when the full score matrix exceeds memory budget.
    ///
    /// Splits the stored matrix into ~64 MB chunks, runs GEMM per chunk, and merges
    /// per-query top-k results. This enables GEMM acceleration for large matrices
    /// where the full `query_rows * rows` score matrix would exceed 128 MB.
    fn search_batch_gemm_chunked(
        &self,
        queries: &[f32],
        query_rows: usize,
        k: usize,
    ) -> Vec<Vec<Neighbor>> {
        use ndarray::{Array2, ArrayView2};

        let queries_view =
            ArrayView2::from_shape((query_rows, self.dims), &queries[..query_rows * self.dims])
                .expect("queries shape should match index dimensions");

        // Target ~64 MB per chunk of the stored matrix.
        let bytes_per_row = self.dims * std::mem::size_of::<f32>();
        let chunk_rows = ((64 * 1024 * 1024) / bytes_per_row).max(64);
        let chunk_len = chunk_rows * self.dims;

        let mut accumulators: Vec<TopKAccumulator> =
            (0..query_rows).map(|_| TopKAccumulator::new(k)).collect();

        for (chunk_idx, chunk) in self.matrix.chunks(chunk_len).enumerate() {
            let rows_in_chunk = chunk.len() / self.dims;
            let row_offset = chunk_idx * chunk_rows;

            let chunk_view = ArrayView2::from_shape((rows_in_chunk, self.dims), chunk)
                .expect("chunk shape should be valid");

            // GEMM: (query_rows, dims) x (dims, rows_in_chunk) => (query_rows, rows_in_chunk)
            let scores: Array2<f32> = queries_view.dot(&chunk_view.t());
            let scores_slice = scores
                .as_slice()
                .expect("GEMM chunk output should be row-major");

            // Stream chunk scores into global accumulators with hoisted threshold.
            for (query_idx, chunk_scores) in scores_slice.chunks_exact(rows_in_chunk).enumerate() {
                let acc = &mut accumulators[query_idx];
                let mut threshold = acc.min_threshold;
                for (local_row, &score) in chunk_scores.iter().enumerate() {
                    if score > threshold {
                        acc.push_slow(row_offset + local_row, score);
                        threshold = acc.min_threshold;
                    }
                }
            }
        }

        accumulators
            .into_iter()
            .map(TopKAccumulator::into_sorted_vec)
            .collect()
    }

    /// Batch search optimized for cache locality by chunking the matrix.
    ///
    /// Instead of each query scanning the full matrix independently, the matrix
    /// is split into L2-cache-friendly chunks. Each chunk is loaded once and all
    /// queries compute dot products against it while it remains hot in cache.
    /// Uses rayon's fold/reduce to reuse accumulators across chunks per thread.
    fn search_batch_matrix_chunked(
        &self,
        queries: &[f32],
        query_rows: usize,
        k: usize,
    ) -> Vec<Vec<Neighbor>> {
        let rows_per_chunk = batch_chunk_rows(self.dims);
        let chunk_len = rows_per_chunk * self.dims;

        self.matrix
            .par_chunks(chunk_len)
            .enumerate()
            .fold(
                || {
                    (0..query_rows)
                        .map(|_| TopKAccumulator::new(k))
                        .collect::<Vec<_>>()
                },
                |mut accumulators, (chunk_idx, chunk)| {
                    let row_offset = chunk_idx * rows_per_chunk;
                    for (query_idx, acc) in accumulators.iter_mut().enumerate() {
                        let qstart = query_idx * self.dims;
                        let query = &queries[qstart..qstart + self.dims];
                        let mut threshold = acc.min_threshold;
                        for (local_row, row) in chunk.chunks_exact(self.dims).enumerate() {
                            let score = dot_product(row, query);
                            if score > threshold {
                                acc.push_slow(row_offset + local_row, score);
                                threshold = acc.min_threshold;
                            }
                        }
                    }
                    accumulators
                },
            )
            .reduce(
                || (0..query_rows).map(|_| TopKAccumulator::new(k)).collect(),
                |mut left, right| {
                    for (l, r) in left.iter_mut().zip(right) {
                        l.merge(r);
                    }
                    left
                },
            )
            .into_iter()
            .map(TopKAccumulator::into_sorted_vec)
            .collect()
    }

    /// Returns the ANN prefilter index only if it actually reduces dimensionality.
    /// When `sample_dims == dims`, the prefilter does the same work as exact search
    /// with extra overhead, so we return `None` to fall through to exact search.
    #[inline]
    fn approx_index(&self) -> Option<&ApproxIndex> {
        self.approx_index.as_ref().and_then(|index| {
            let built =
                index.get_or_init(|| ApproxIndex::build(&self.matrix, self.rows, self.dims));
            built.provides_speedup().then_some(built)
        })
    }

    /// Hash a query (or batch of queries) together with k and shape for cache keying.
    fn query_hash(&self, values: &[f32], k: usize, query_rows: usize) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.rows.hash(&mut hasher);
        self.dims.hash(&mut hasher);
        k.hash(&mut hasher);
        query_rows.hash(&mut hasher);
        bytemuck::cast_slice::<f32, u8>(values).hash(&mut hasher);
        hasher.finish()
    }

    /// Prefilter with the IVF index, then rerank candidates.
    ///
    /// Uses ndarray sgemv (dispatching to BLAS/AMX) for clusters with enough rows,
    /// falling back to simsimd dot products for small clusters.
    fn search_approx(&self, query: &[f32], k: usize, approx_index: &ApproxIndex) -> Vec<Neighbor> {
        use ndarray::{ArrayView1, ArrayView2};

        let clusters = approx_index.candidate_clusters(query, k);
        let query_view =
            ArrayView1::from_shape(self.dims, query).expect("query shape valid");
        let mut accumulator = TopKAccumulator::new(k);

        for (row_data, indices) in clusters {
            let n_rows = indices.len();
            if n_rows >= 4 && self.dims >= 8 {
                let matrix_view =
                    ArrayView2::from_shape((n_rows, self.dims), row_data).expect("cluster shape valid");
                let scores = matrix_view.dot(&query_view);
                let scores_slice = scores.as_slice().expect("contiguous");
                for (local_row, &score) in scores_slice.iter().enumerate() {
                    accumulator.push(indices[local_row], score);
                }
            } else {
                for (local_row, row) in row_data.chunks_exact(self.dims).enumerate() {
                    accumulator.push(indices[local_row], dot_product(row, query));
                }
            }
        }
        accumulator.into_sorted_vec()
    }

    /// Batch IVF prefilter + rerank with AMX for large clusters.
    fn search_batch_approx(
        &self,
        queries: &[f32],
        query_rows: usize,
        k: usize,
        approx_index: &ApproxIndex,
    ) -> Vec<Vec<Neighbor>> {
        use ndarray::{ArrayView1, ArrayView2};

        let query_values = &queries[..query_rows * self.dims];
        let cluster_sets = approx_index.candidate_clusters_batch(query_values, query_rows, k);
        let dims = self.dims;

        cluster_sets
            .into_par_iter()
            .enumerate()
            .map(|(query_idx, clusters)| {
                let query = &query_values[query_idx * dims..(query_idx + 1) * dims];
                let query_view =
                    ArrayView1::from_shape(dims, query).expect("query shape valid");
                let mut accumulator = TopKAccumulator::new(k);

                for (row_data, indices) in clusters {
                    let n_rows = indices.len();
                    if n_rows >= 4 && dims >= 8 {
                        let matrix_view =
                            ArrayView2::from_shape((n_rows, dims), row_data)
                                .expect("cluster shape valid");
                        let scores = matrix_view.dot(&query_view);
                        let scores_slice = scores.as_slice().expect("contiguous");
                        for (local_row, &score) in scores_slice.iter().enumerate() {
                            accumulator.push(indices[local_row], score);
                        }
                    } else {
                        for (local_row, row) in row_data.chunks_exact(dims).enumerate() {
                            accumulator.push(indices[local_row], dot_product(row, query));
                        }
                    }
                }

                accumulator.into_sorted_vec()
            })
            .collect()
    }

    /// Exact search dispatched through the tuning profile's selected strategy.
    fn search_with_profile(
        &self,
        query: &[f32],
        k: usize,
        profile: TuningProfile,
    ) -> Vec<Neighbor> {
        let chunked_rows = chunk_rows(self.rows, self.dims);
        let search_strategy = profile.cpu_search_strategy(
            self.rows,
            self.dims,
            parallel_cutoff(self.dims),
            chunked_rows,
            SCORES_VEC_MAX_ROWS,
            cfg!(feature = "blas-accelerate"),
        );
        match search_strategy {
            CpuSearchStrategy::Serial => self.search_serial(query, k),
            CpuSearchStrategy::Gemv => self.search_gemv(query, k),
            CpuSearchStrategy::ParallelScores => self.search_parallel_scores(query, k),
            CpuSearchStrategy::ParallelChunked => self.search_parallel_chunked(query, k),
            CpuSearchStrategy::ParallelFold => self.search_parallel_fold(query, k),
        }
    }

    /// Compute all dot products in parallel into a pre-allocated scores buffer,
    /// then select top-k in a single sequential pass. Uses the same coarse chunk
    /// size as `search_parallel_chunked` but avoids per-chunk TopKAccumulator
    /// allocation and the reduce/merge tree entirely.
    fn search_parallel_scores(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        let rows_per_chunk = chunk_rows(self.rows, self.dims).max(
            // Ensure each rayon task has enough work to amortize scheduling.
            parallel_cutoff(self.dims) / rayon::current_num_threads().max(1),
        );
        let mut scores = vec![0.0f32; self.rows];

        scores
            .par_chunks_mut(rows_per_chunk)
            .enumerate()
            .for_each(|(chunk_idx, score_chunk)| {
                let row_base = chunk_idx * rows_per_chunk;
                for (local_i, score) in score_chunk.iter_mut().enumerate() {
                    let offset = (row_base + local_i) * self.dims;
                    let row = &self.matrix[offset..offset + self.dims];
                    *score = dot_product(row, query);
                }
            });

        let mut accumulator = TopKAccumulator::new(k);
        for (idx, &score) in scores.iter().enumerate() {
            accumulator.push(idx, score);
        }
        accumulator.into_sorted_vec()
    }

    /// Single-threaded linear scan over all rows.
    fn search_serial(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        let mut accumulator = TopKAccumulator::new(k);
        for (row_index, row) in self.matrix.chunks_exact(self.dims).enumerate() {
            accumulator.push(row_index, dot_product(row, query));
        }
        accumulator.into_sorted_vec()
    }

    /// Single-query search via BLAS `sgemv`: scores = matrix * query.
    ///
    /// Computes all dot products as a single matrix-vector multiply using
    /// ndarray (which dispatches to BLAS `sgemv` when `blas-accelerate` is enabled).
    /// Then extracts top-k using O(n) introselect. Faster than row-by-row SIMD
    /// for high-dimensional matrices because the BLAS kernel uses cache-optimal tiling.
    fn search_gemv(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        use ndarray::{ArrayView1, ArrayView2};

        let matrix_view = ArrayView2::from_shape((self.rows, self.dims), &self.matrix)
            .expect("matrix shape should match stored dimensions");

        let query_view = ArrayView1::from_shape(self.dims, &query[..self.dims])
            .expect("query shape should match index dimensions");

        // sgemv: (rows, dims) x (dims,) => (rows,)
        let scores = matrix_view.dot(&query_view);

        let scores_slice = scores
            .as_slice()
            .expect("sgemv output should be contiguous");

        topk_from_scores(scores_slice, k, 0)
    }

    /// Parallel search splitting the matrix into fixed-size chunks with per-chunk top-k merge.
    fn search_parallel_chunked(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        let rows_per_chunk = chunk_rows(self.rows, self.dims);
        let chunk_len = rows_per_chunk * self.dims;

        self.matrix
            .par_chunks(chunk_len)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let row_offset = chunk_idx * rows_per_chunk;
                let mut accumulator = TopKAccumulator::new(k);
                for (local_row, row) in chunk.chunks_exact(self.dims).enumerate() {
                    accumulator.push(row_offset + local_row, dot_product(row, query));
                }
                accumulator
            })
            .reduce(
                || TopKAccumulator::new(k),
                |mut left, right| {
                    left.merge(right);
                    left
                },
            )
            .into_sorted_vec()
    }

    /// Parallel search using rayon fold/reduce over individual rows, best for very large matrices.
    fn search_parallel_fold(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        self.matrix
            .par_chunks_exact(self.dims)
            .enumerate()
            .fold(
                || TopKAccumulator::new(k),
                |mut accumulator, (row_index, row)| {
                    accumulator.push(row_index, dot_product(row, query));
                    accumulator
                },
            )
            .reduce(
                || TopKAccumulator::new(k),
                |mut left, right| {
                    left.merge(right);
                    left
                },
            )
            .into_sorted_vec()
    }
}

/// One-time initialization: flush denormals to zero for consistent SIMD behavior.
#[inline]
fn initialize_simd_runtime() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = simsimd::capabilities::flush_denormals();
    });
}

/// Dot product using SIMD (simsimd) when vectors are >= 8 elements, with a scalar fallback.
#[inline]
fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    if lhs.len() >= 8
        && let Some(simd) = f32::dot(lhs, rhs)
    {
        return simd as f32;
    }
    dot_unrolled_8(lhs, rhs)
}

/// Scalar dot product with 8-wide manual unrolling to reduce loop overhead on short vectors.
///
/// Uses 8 independent accumulators to enable instruction-level parallelism (ILP).
/// The length assertion lets LLVM elide bounds checks on rhs indexing.
#[inline]
fn dot_unrolled_8(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len().min(rhs.len());
    let lhs = &lhs[..len];
    let rhs = &rhs[..len];

    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut s4, mut s5, mut s6, mut s7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

    let mut idx = 0;
    while idx + 8 <= len {
        s0 += lhs[idx] * rhs[idx];
        s1 += lhs[idx + 1] * rhs[idx + 1];
        s2 += lhs[idx + 2] * rhs[idx + 2];
        s3 += lhs[idx + 3] * rhs[idx + 3];
        s4 += lhs[idx + 4] * rhs[idx + 4];
        s5 += lhs[idx + 5] * rhs[idx + 5];
        s6 += lhs[idx + 6] * rhs[idx + 6];
        s7 += lhs[idx + 7] * rhs[idx + 7];
        idx += 8;
    }

    // Tail: handle remaining elements
    while idx < len {
        s0 += lhs[idx] * rhs[idx];
        idx += 1;
    }

    (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
}

/// Minimum row count before parallelism becomes worthwhile, based on dimension count.
#[inline]
fn parallel_cutoff(dims: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let per_thread_target = (32_768 / dims.max(1)).max(1);
    threads.saturating_mul(per_thread_target)
}

/// Compute the number of matrix rows per parallel chunk, targeting ~128 KB per chunk.
#[inline]
fn chunk_rows(rows: usize, dims: usize) -> usize {
    let target_floats_per_chunk = 32_768usize.max(dims);
    let rows_per_chunk = (target_floats_per_chunk / dims).max(1);
    rows_per_chunk.min(rows.max(1))
}

/// Compute chunk size for batch-optimized matrix scanning.
///
/// Targets ~256KB per chunk (L2 cache friendly on most CPUs).
#[inline]
fn batch_chunk_rows(dims: usize) -> usize {
    let bytes_per_row = dims * std::mem::size_of::<f32>();
    let target_bytes = 256 * 1024;
    (target_bytes / bytes_per_row).max(64)
}

#[cfg(test)]
mod tests {
    use ::approx::assert_abs_diff_eq;

    use super::*;

    // ---- BoundedLruCache ----

    #[test]
    fn lru_cache_insert_and_retrieve() {
        let mut cache = BoundedLruCache::with_capacity(4);
        cache.insert(1, "a");
        cache.insert(2, "b");
        assert_eq!(cache.get_cloned(1), Some("a"));
        assert_eq!(cache.get_cloned(2), Some("b"));
        assert_eq!(cache.get_cloned(99), None);
    }

    #[test]
    fn lru_cache_evicts_least_recently_used() {
        let mut cache = BoundedLruCache::with_capacity(2);
        cache.insert(1, "a");
        cache.insert(2, "b");
        cache.insert(3, "c");

        assert_eq!(cache.get_cloned(3), Some("c"));
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn lru_cache_update_existing_key() {
        let mut cache = BoundedLruCache::with_capacity(3);
        cache.insert(1, "old");
        cache.insert(2, "other");
        cache.insert(1, "new");

        assert_eq!(cache.get_cloned(1), Some("new"));
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn lru_cache_get_promotes_to_mru() {
        let mut cache = BoundedLruCache::with_capacity(3);
        cache.insert(1, "a");
        cache.insert(2, "b");
        cache.insert(3, "c");

        let _ = cache.get_cloned(1);
        cache.insert(4, "d");

        assert_eq!(cache.get_cloned(1), Some("a"));
    }

    // ---- dot_product / dot_unrolled_8 ----

    #[test]
    fn dot_unrolled_8_matches_naive() {
        let a: [f32; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b: [f32; 10] = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert_abs_diff_eq!(dot_unrolled_8(&a, &b), expected, epsilon = 1e-6);
    }

    #[test]
    fn dot_unrolled_8_short_vectors() {
        let a: [f32; 2] = [3.0, 4.0];
        let b: [f32; 2] = [1.0, 2.0];
        assert_abs_diff_eq!(dot_unrolled_8(&a, &b), 11.0, epsilon = 1e-6);
    }

    #[test]
    fn dot_unrolled_8_empty_vectors() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        assert_abs_diff_eq!(dot_unrolled_8(&a, &b), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn dot_unrolled_8_single_element() {
        let a: [f32; 1] = [3.0];
        let b: [f32; 1] = [7.0];
        assert_abs_diff_eq!(dot_unrolled_8(&a, &b), 21.0, epsilon = 1e-6);
    }

    #[test]
    fn dot_unrolled_8_seven_elements() {
        // 7 elements = zero unrolled iterations, 7 tail iterations
        let a: [f32; 7] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b: [f32; 7] = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert_abs_diff_eq!(dot_unrolled_8(&a, &b), expected, epsilon = 1e-6);
    }

    #[test]
    fn dot_unrolled_8_exactly_eight() {
        // 8 elements = exactly one unrolled iteration, zero tail
        let a: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b: [f32; 8] = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert_abs_diff_eq!(dot_unrolled_8(&a, &b), expected, epsilon = 1e-6);
    }

    #[test]
    fn dot_unrolled_8_nine_elements() {
        // 9 elements = one unrolled iteration + 1 tail iteration
        let a: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b: [f32; 9] = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert_abs_diff_eq!(dot_unrolled_8(&a, &b), expected, epsilon = 1e-6);
    }

    #[test]
    fn dot_unrolled_8_sixteen_elements() {
        // 16 elements = two unrolled iterations, zero tail
        let a: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=16).rev().map(|i| i as f32).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert_abs_diff_eq!(dot_unrolled_8(&a, &b), expected, epsilon = 1e-6);
    }

    #[test]
    fn dot_unrolled_8_mismatched_lengths() {
        // lhs shorter than rhs -- should use min(len) = 3
        let a: [f32; 3] = [1.0, 2.0, 3.0];
        let b: [f32; 5] = [4.0, 5.0, 6.0, 7.0, 8.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(); // 4+10+18 = 32
        assert_abs_diff_eq!(dot_unrolled_8(&a, &b), expected, epsilon = 1e-6);
    }

    #[test]
    fn dot_product_matches_naive_for_long_vector() {
        let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..64).map(|i| (63 - i) as f32 * 0.1).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert_abs_diff_eq!(dot_product(&a, &b), expected, epsilon = 0.01);
    }

    #[test]
    fn dot_product_orthogonal_vectors() {
        let a = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_abs_diff_eq!(dot_product(&a, &b), 0.0, epsilon = 1e-6);
    }

    // ---- chunk_rows / batch_chunk_rows ----

    #[test]
    fn chunk_rows_at_least_one() {
        assert!(chunk_rows(1, 100_000) >= 1);
    }

    #[test]
    fn chunk_rows_capped_at_total_rows() {
        assert!(chunk_rows(10, 8) <= 10);
    }

    #[test]
    fn batch_chunk_rows_at_least_64() {
        assert!(batch_chunk_rows(100_000) >= 64);
    }

    #[test]
    fn batch_chunk_rows_scales_inversely_with_dims() {
        let small_dims = batch_chunk_rows(32);
        let large_dims = batch_chunk_rows(1024);
        assert!(small_dims > large_dims);
    }
}
