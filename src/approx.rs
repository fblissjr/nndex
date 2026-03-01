#[cfg(feature = "cpu")]
use rayon::prelude::*;

/// IVF (Inverted File Index) for approximate nearest-neighbor prefiltering.
///
/// Partitions the row space into `nlist` clusters via spherical k-means.
/// At query time, only rows belonging to the `nprobe` nearest clusters are
/// scanned. Rows are stored in cluster-contiguous order so that probing a
/// cluster reads sequential memory, dramatically improving cache locality.
#[derive(Debug, Clone)]
pub(crate) struct ApproxIndex {
    rows: usize,
    dims: usize,
    nlist: usize,
    /// Centroid matrix, shape `(nlist, dims)`, row-major.
    centroids: Vec<f32>,
    /// Row data reordered so all rows in cluster 0 come first, then cluster 1, etc.
    /// Shape: `(rows, dims)`, row-major.
    reordered_matrix: Vec<f32>,
    /// Maps position in `reordered_matrix` back to the original row index.
    original_indices: Vec<usize>,
    /// `cluster_offsets[c]..cluster_offsets[c+1]` gives the range of rows in
    /// `reordered_matrix` belonging to cluster `c`. Length `nlist + 1`.
    cluster_offsets: Vec<usize>,
}

/// Maximum number of spherical k-means iterations.
const KMEANS_MAX_ITERS: usize = 12;

/// Minimum row count for IVF to outperform exact search.
const MIN_ROWS_FOR_IVF: usize = 256;

impl ApproxIndex {
    /// Returns `true` when the IVF prefilter is expected to outperform exact search.
    pub(crate) fn provides_speedup(&self) -> bool {
        self.rows >= MIN_ROWS_FOR_IVF && self.nlist >= 4
    }

    /// Build an IVF index with cluster-contiguous row storage.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Row-major `(rows, dims)` matrix (assumed unit-normalized rows).
    /// * `rows` - Number of rows.
    /// * `dims` - Number of columns.
    pub(crate) fn build(matrix: &[f32], rows: usize, dims: usize) -> Self {
        let nlist = choose_nlist(rows);

        if rows < MIN_ROWS_FOR_IVF || nlist < 2 {
            return Self {
                rows,
                dims,
                nlist: 1,
                centroids: Vec::new(),
                reordered_matrix: matrix.to_vec(),
                original_indices: (0..rows).collect(),
                cluster_offsets: vec![0, rows],
            };
        }

        let centroids = kmeans_plusplus_init(matrix, rows, dims, nlist);
        let (centroids, assignments) = spherical_kmeans(matrix, rows, dims, nlist, centroids);

        // Build cluster-contiguous storage.
        let mut cluster_counts = vec![0_usize; nlist];
        for &c in &assignments {
            cluster_counts[c] += 1;
        }

        let mut cluster_offsets = Vec::with_capacity(nlist + 1);
        cluster_offsets.push(0);
        for &count in &cluster_counts {
            cluster_offsets.push(cluster_offsets.last().copied().unwrap_or(0) + count);
        }

        // Fill reordered matrix: write each row at its cluster's next slot.
        let mut write_pos = cluster_offsets[..nlist].to_vec();
        let mut reordered_matrix = vec![0.0_f32; rows * dims];
        let mut original_indices = vec![0_usize; rows];

        for (row_idx, &cluster) in assignments.iter().enumerate() {
            let dest = write_pos[cluster];
            write_pos[cluster] += 1;
            original_indices[dest] = row_idx;
            let src_start = row_idx * dims;
            let dst_start = dest * dims;
            reordered_matrix[dst_start..dst_start + dims]
                .copy_from_slice(&matrix[src_start..src_start + dims]);
        }

        Self {
            rows,
            dims,
            nlist,
            centroids,
            reordered_matrix,
            original_indices,
            cluster_offsets,
        }
    }

    /// Return candidate row indices for a single query by probing the nearest clusters.
    ///
    /// Reads contiguous memory for each probed cluster, then returns original indices.
    /// This returns a flat row-index list for GPU reranking and tests.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector with length `dims`.
    /// * `k` - Number of final neighbors requested (used to tune nprobe).
    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn candidate_indices(&self, query: &[f32], k: usize) -> Vec<usize> {
        debug_assert_eq!(query.len(), self.dims);

        if self.nlist <= 1 {
            return self.original_indices.clone();
        }

        let nprobe = choose_nprobe(self.nlist, k, self.rows);

        // Score query against all centroids.
        let mut centroid_scores: Vec<(usize, f32)> = self
            .centroids
            .chunks_exact(self.dims)
            .enumerate()
            .map(|(idx, centroid)| (idx, dot_small(centroid, query)))
            .collect();

        let nprobe = nprobe.min(centroid_scores.len());
        centroid_scores.select_nth_unstable_by(nprobe - 1, |a, b| b.1.total_cmp(&a.1));

        let mut candidates = Vec::new();
        for &(cluster_id, _) in &centroid_scores[..nprobe] {
            let start = self.cluster_offsets[cluster_id];
            let end = self.cluster_offsets[cluster_id + 1];
            candidates.extend_from_slice(&self.original_indices[start..end]);
        }
        candidates
    }

    /// Return candidate data for a single query: contiguous row data + original indices.
    ///
    /// Instead of returning scattered indices, returns contiguous slices from the
    /// reordered matrix that can be scored with sequential memory access.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector with length `dims`.
    /// * `k` - Number of final neighbors requested.
    ///
    /// # Returns
    ///
    /// Vector of `(cluster_row_data_slice, original_indices_slice)` per probed cluster.
    ///
    /// UPDATE: By stripping out the ArrayView2 matrix-vector product, we avoid the dynamic allocation overhead of creating an output vector for the centroid scores
    /// instead, the code will natively use our zero-allocation dot_small inline loop (which already correctly handles SIMD acceleration when the cpu feature is active).
    pub(crate) fn candidate_clusters(&self, query: &[f32], k: usize) -> Vec<(&[f32], &[usize])> {
        debug_assert_eq!(query.len(), self.dims);

        if self.nlist <= 1 {
            return vec![(&self.reordered_matrix, &self.original_indices)];
        }

        let nprobe = choose_nprobe(self.nlist, k, self.rows);

        // Stream centroid scores directly using the zero-allocation inner loop.
        // This completely removes the ndarray dynamic allocation and branching.
        let mut centroid_scores: Vec<(usize, f32)> = self
            .centroids
            .chunks_exact(self.dims)
            .enumerate()
            .map(|(idx, centroid)| (idx, dot_small(centroid, query)))
            .collect();

        let nprobe = nprobe.min(centroid_scores.len());
        centroid_scores.select_nth_unstable_by(nprobe - 1, |a, b| b.1.total_cmp(&a.1));

        centroid_scores[..nprobe]
            .iter()
            .map(|&(cluster_id, _)| {
                let start = self.cluster_offsets[cluster_id];
                let end = self.cluster_offsets[cluster_id + 1];
                let row_data = &self.reordered_matrix[start * self.dims..end * self.dims];
                let indices = &self.original_indices[start..end];
                (row_data, indices)
            })
            .collect()
    }

    /// Return candidate cluster data for a batch of queries.
    ///
    /// Uses BLAS GEMM to score all queries against centroids, then per-query
    /// returns contiguous cluster slices for cache-friendly reranking.
    ///
    /// # Arguments
    ///
    /// * `queries` - Row-major `(query_rows, dims)` matrix.
    /// * `query_rows` - Number of query rows.
    /// * `k` - Number of final neighbors requested per query.
    #[cfg(feature = "cpu")]
    pub(crate) fn candidate_clusters_batch(
        &self,
        queries: &[f32],
        query_rows: usize,
        k: usize,
    ) -> Vec<Vec<(&[f32], &[usize])>> {
        debug_assert_eq!(queries.len(), query_rows.saturating_mul(self.dims));

        if query_rows == 0 {
            return Vec::new();
        }

        if self.nlist <= 1 {
            let all = vec![(&self.reordered_matrix[..], &self.original_indices[..])];
            return vec![all; query_rows];
        }

        let nprobe = choose_nprobe(self.nlist, k, self.rows);

        // For tiny batches, fall back to per-query scoring.
        if query_rows < 4 || self.nlist < 8 {
            return queries
                .chunks_exact(self.dims)
                .take(query_rows)
                .map(|query| self.candidate_clusters(query, k))
                .collect();
        }

        // 1. Keep the AMX-accelerated GEMM for bulk centroid scoring
        use ndarray::ArrayView2;
        let queries_view =
            ArrayView2::from_shape((query_rows, self.dims), &queries[..query_rows * self.dims])
                .expect("query batch shape should be valid");
        let centroids_view = ArrayView2::from_shape((self.nlist, self.dims), &self.centroids)
            .expect("centroid matrix shape should be valid");

        let scores = queries_view
            .dot(&centroids_view.t())
            .as_standard_layout()
            .to_owned();
        let scores_slice = scores
            .as_slice()
            .expect("centroid GEMM output should be contiguous");

        // 2. Allocate ONE reusable buffer for sorting, eliminating per-query allocations
        let mut out = Vec::with_capacity(query_rows);
        let mut indexed_buffer = Vec::with_capacity(self.nlist);

        for query_scores in scores_slice.chunks_exact(self.nlist) {
            let nprobe = nprobe.min(query_scores.len());

            // Reuse capacity
            indexed_buffer.clear();
            indexed_buffer.extend(query_scores.iter().enumerate().map(|(i, &s)| (i, s)));
            indexed_buffer.select_nth_unstable_by(nprobe - 1, |a, b| b.1.total_cmp(&a.1));

            let clusters = indexed_buffer[..nprobe]
                .iter()
                .map(|&(cluster_id, _)| {
                    let start = self.cluster_offsets[cluster_id];
                    let end = self.cluster_offsets[cluster_id + 1];
                    let row_data = &self.reordered_matrix[start * self.dims..end * self.dims];
                    let indices = &self.original_indices[start..end];
                    (row_data, indices)
                })
                .collect();

            out.push(clusters);
        }

        out
    }
}

/// Choose the number of IVF clusters, adaptive to row count.
///
/// Uses `isqrt(rows)` clamped to `[16, 512]`.
fn choose_nlist(rows: usize) -> usize {
    if rows < MIN_ROWS_FOR_IVF {
        return 1;
    }
    let sqrt = isqrt(rows);
    sqrt.clamp(16, 512)
}

/// Choose the number of clusters to probe at query time.
///
/// Targets a scan fraction that achieves >= 5x speedup over exact search.
/// With cluster-contiguous storage, sequential memory access allows scanning
/// ~5-8% of rows while still achieving high recall.
fn choose_nprobe(nlist: usize, k: usize, rows: usize) -> usize {
    let scan_pct = if k >= 50 {
        8
    } else if k >= 20 {
        6
    } else {
        4
    };

    let nprobe = (nlist * scan_pct).div_ceil(100).max(1);

    // Ensure enough candidates for quality: target scanning at least
    // k*40 rows so the top-k has reasonable overlap with exact.
    let avg_cluster_size = (rows / nlist.max(1)).max(1);
    let min_candidates = k.saturating_mul(40);
    let min_nprobe = min_candidates.div_ceil(avg_cluster_size).max(1);

    nprobe.max(min_nprobe).clamp(1, nlist)
}

/// Integer square root via Newton's method.
fn isqrt(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// K-means++ initialization: select `nlist` initial centroids with probability
/// proportional to squared distance from the nearest existing centroid.
///
/// For unit-normalized vectors, distance^2 = 2 - 2*dot, so we use `1 - dot`
/// as the distance proxy.
fn kmeans_plusplus_init(matrix: &[f32], rows: usize, dims: usize, nlist: usize) -> Vec<f32> {
    // Deterministic seed based on matrix shape for reproducibility.
    let mut rng_state: u64 = (rows as u64)
        .wrapping_mul(2654435761)
        .wrapping_add(dims as u64);

    let mut centroids = Vec::with_capacity(nlist * dims);

    let first_idx = pick_medoid_row(matrix, rows, dims);
    centroids.extend_from_slice(&matrix[first_idx * dims..(first_idx + 1) * dims]);

    let mut min_dists = vec![f32::MAX; rows];

    for c in 1..nlist {
        let last_centroid = &centroids[(c - 1) * dims..c * dims];

        let mut total_dist = 0.0_f64;
        for (row_idx, min_dist) in min_dists.iter_mut().enumerate() {
            let row = &matrix[row_idx * dims..(row_idx + 1) * dims];
            let dist = (1.0 - dot_small(row, last_centroid)).max(0.0);
            if dist < *min_dist {
                *min_dist = dist;
            }
            total_dist += *min_dist as f64;
        }

        if total_dist <= 0.0 {
            let dup: Vec<f32> = last_centroid.to_vec();
            centroids.extend_from_slice(&dup);
            continue;
        }

        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let threshold = (rng_state as f64 / u64::MAX as f64) * total_dist;

        let mut cumulative = 0.0_f64;
        let mut chosen = rows - 1;
        for (row_idx, &dist) in min_dists.iter().enumerate() {
            cumulative += dist as f64;
            if cumulative >= threshold {
                chosen = row_idx;
                break;
            }
        }

        centroids.extend_from_slice(&matrix[chosen * dims..(chosen + 1) * dims]);
    }

    centroids
}

/// Find the row closest to the global mean (medoid).
fn pick_medoid_row(matrix: &[f32], rows: usize, dims: usize) -> usize {
    let mut mean = vec![0.0_f64; dims];
    for row in matrix.chunks_exact(dims).take(rows) {
        for (m, &v) in mean.iter_mut().zip(row) {
            *m += v as f64;
        }
    }
    let inv_rows = 1.0 / rows as f64;
    let mean_f32: Vec<f32> = mean.iter().map(|&m| (m * inv_rows) as f32).collect();

    let mut best_idx = 0;
    let mut best_dot = f32::NEG_INFINITY;
    for (idx, row) in matrix.chunks_exact(dims).take(rows).enumerate() {
        let d = dot_small(row, &mean_f32);
        if d > best_dot {
            best_dot = d;
            best_idx = idx;
        }
    }
    best_idx
}

/// Run spherical k-means for a fixed number of iterations.
///
/// Returns the final centroids and per-row cluster assignments.
fn spherical_kmeans(
    matrix: &[f32],
    rows: usize,
    dims: usize,
    nlist: usize,
    mut centroids: Vec<f32>,
) -> (Vec<f32>, Vec<usize>) {
    let mut assignments = vec![0_usize; rows];

    for _iter in 0..KMEANS_MAX_ITERS {
        #[cfg(feature = "cpu")]
        {
            let centroid_ref = &centroids;
            assignments
                .par_chunks_mut(1024)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let row_offset = chunk_idx * 1024;
                    for (local_i, assignment) in chunk.iter_mut().enumerate() {
                        let row_idx = row_offset + local_i;
                        if row_idx >= rows {
                            break;
                        }
                        let row = &matrix[row_idx * dims..(row_idx + 1) * dims];
                        let mut best_cluster = 0;
                        let mut best_score = f32::NEG_INFINITY;
                        for (c_idx, centroid) in centroid_ref.chunks_exact(dims).enumerate() {
                            let score = dot_small(row, centroid);
                            if score > best_score {
                                best_score = score;
                                best_cluster = c_idx;
                            }
                        }
                        *assignment = best_cluster;
                    }
                });
        }

        #[cfg(not(feature = "cpu"))]
        {
            for row_idx in 0..rows {
                let row = &matrix[row_idx * dims..(row_idx + 1) * dims];
                let mut best_cluster = 0;
                let mut best_score = f32::NEG_INFINITY;
                for (c_idx, centroid) in centroids.chunks_exact(dims).enumerate() {
                    let score = dot_small(row, centroid);
                    if score > best_score {
                        best_score = score;
                        best_cluster = c_idx;
                    }
                }
                assignments[row_idx] = best_cluster;
            }
        }

        // Recompute centroids as mean of assigned rows, then re-normalize.
        let mut new_centroids = vec![0.0_f64; nlist * dims];
        let mut cluster_sizes = vec![0_usize; nlist];

        for (row_idx, &cluster) in assignments.iter().enumerate() {
            cluster_sizes[cluster] += 1;
            let row = &matrix[row_idx * dims..(row_idx + 1) * dims];
            let centroid_start = cluster * dims;
            for (d, &val) in row.iter().enumerate() {
                new_centroids[centroid_start + d] += val as f64;
            }
        }

        centroids.clear();
        centroids.reserve(nlist * dims);
        for (c, &count) in cluster_sizes.iter().enumerate().take(nlist) {
            let start = c * dims;
            let end = start + dims;

            if count == 0 {
                centroids.extend(std::iter::repeat_n(0.0_f32, dims));
                continue;
            }

            let inv_count = 1.0 / count as f64;
            let slice = &new_centroids[start..end];

            let mut norm_sq = 0.0_f64;
            for &val in slice {
                let scaled = val * inv_count;
                norm_sq += scaled * scaled;
            }

            let inv_norm = if norm_sq > 1e-30 {
                1.0 / norm_sq.sqrt()
            } else {
                1.0
            };

            for &val in slice {
                centroids.push((val * inv_count * inv_norm) as f32);
            }
        }
    }

    (centroids, assignments)
}

/// Dot product for vectors, using SIMD when available and beneficial.
#[inline]
fn dot_small(lhs: &[f32], rhs: &[f32]) -> f32 {
    #[cfg(feature = "cpu")]
    if lhs.len() >= 8
        && let Some(simd) = simsimd::SpatialSimilarity::dot(lhs, rhs)
    {
        return simd as f32;
    }

    let mut sum = 0.0f32;
    for idx in 0..lhs.len() {
        sum += lhs[idx] * rhs[idx];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isqrt_known_values() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(10_000), 100);
        assert_eq!(isqrt(65_536), 256);
    }

    #[test]
    fn choose_nlist_small_dataset() {
        assert_eq!(choose_nlist(100), 1);
        assert_eq!(choose_nlist(200), 1);
    }

    #[test]
    fn choose_nlist_medium_dataset() {
        assert_eq!(choose_nlist(10_000), 100);
    }

    #[test]
    fn choose_nlist_large_dataset() {
        assert_eq!(choose_nlist(1_000_000), 512);
    }

    #[test]
    fn choose_nlist_clamped_minimum() {
        assert!(choose_nlist(300) >= 16);
    }

    #[test]
    fn choose_nprobe_returns_valid_range() {
        for nlist in [16, 64, 128, 256, 512] {
            for k in [1, 5, 10, 20, 50, 100] {
                for rows in [1_000, 10_000, 100_000] {
                    let nprobe = choose_nprobe(nlist, k, rows);
                    assert!(nprobe >= 1, "nprobe must be >= 1");
                    assert!(nprobe <= nlist, "nprobe must be <= nlist");
                }
            }
        }
    }

    #[test]
    fn provides_speedup_requires_enough_rows() {
        let small = ApproxIndex::build(&vec![1.0; 100 * 32], 100, 32);
        assert!(!small.provides_speedup());
    }

    #[test]
    fn provides_speedup_true_for_large_dataset() {
        let rows = 4_096;
        let dims = 64;
        let matrix: Vec<f32> = (0..rows * dims)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let index = ApproxIndex::build(&matrix, rows, dims);
        assert!(index.provides_speedup());
    }

    #[test]
    fn build_creates_valid_cluster_contiguous_storage() {
        let rows = 1_000;
        let dims = 32;
        let matrix: Vec<f32> = (0..rows * dims)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect();
        let index = ApproxIndex::build(&matrix, rows, dims);

        // Every original row should appear exactly once in original_indices.
        let mut seen = vec![false; rows];
        for &orig_idx in &index.original_indices {
            assert!(orig_idx < rows);
            assert!(!seen[orig_idx], "row {orig_idx} appears multiple times");
            seen[orig_idx] = true;
        }
        for (row_idx, &was_seen) in seen.iter().enumerate() {
            assert!(was_seen, "row {row_idx} missing from reordered storage");
        }

        // Cluster offsets should be monotonically increasing and sum to rows.
        assert_eq!(index.cluster_offsets.first(), Some(&0));
        assert_eq!(index.cluster_offsets.last(), Some(&rows));
        for w in index.cluster_offsets.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn candidate_indices_returns_valid_row_indices() {
        let rows = 2_048;
        let dims = 64;
        let matrix: Vec<f32> = (0..rows * dims)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let query: Vec<f32> = (0..dims).map(|i| ((i as f32) * 0.019).cos()).collect();
        let index = ApproxIndex::build(&matrix, rows, dims);
        let candidates = index.candidate_indices(&query, 10);

        assert!(!candidates.is_empty());
        for &idx in &candidates {
            assert!(idx < rows);
        }
    }

    #[test]
    fn candidate_clusters_returns_contiguous_data() {
        let rows = 2_048;
        let dims = 64;
        let matrix: Vec<f32> = (0..rows * dims)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let query: Vec<f32> = (0..dims).map(|i| ((i as f32) * 0.019).cos()).collect();
        let index = ApproxIndex::build(&matrix, rows, dims);
        let clusters = index.candidate_clusters(&query, 10);

        assert!(!clusters.is_empty());
        for (row_data, indices) in &clusters {
            assert_eq!(row_data.len(), indices.len() * dims);
            for &idx in *indices {
                assert!(idx < rows);
            }
        }
    }

    #[test]
    fn candidate_indices_k_one_returns_candidates() {
        let rows = 512;
        let dims = 128;
        let matrix: Vec<f32> = (0..rows * dims).map(|i| i as f32 * 0.001).collect();
        let query = vec![0.5_f32; dims];
        let index = ApproxIndex::build(&matrix, rows, dims);
        let candidates = index.candidate_indices(&query, 1);
        assert!(!candidates.is_empty());
        for &idx in &candidates {
            assert!(idx < rows);
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn batch_clusters_returns_correct_count() {
        let rows = 2_048;
        let dims = 64;
        let query_rows = 16;
        let k = 10;

        let matrix: Vec<f32> = (0..rows * dims)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let queries: Vec<f32> = (0..query_rows * dims)
            .map(|i| ((i as f32) * 0.017).cos())
            .collect();

        let index = ApproxIndex::build(&matrix, rows, dims);
        let batch = index.candidate_clusters_batch(&queries, query_rows, k);

        assert_eq!(batch.len(), query_rows);
        for query_clusters in &batch {
            assert!(!query_clusters.is_empty());
            for (row_data, indices) in query_clusters {
                assert_eq!(row_data.len(), indices.len() * dims);
            }
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn batch_clusters_empty_input() {
        let index = ApproxIndex::build(&vec![1.0; 1000 * 32], 1000, 32);
        let result = index.candidate_clusters_batch(&[], 0, 10);
        assert!(result.is_empty());
    }
}
