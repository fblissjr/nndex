use crate::topk::TopKAccumulator;

#[cfg(feature = "cpu")]
use rayon::prelude::*;
#[cfg(feature = "cpu")]
use simsimd::SpatialSimilarity;

/// Approximate prefilter index used to build a small candidate set before exact reranking.
#[derive(Debug, Clone)]
pub(crate) struct ApproxIndex {
    rows: usize,
    dims: usize,
    sample_dims: usize,
    sample_indices: Vec<usize>,
    sampled_matrix: Vec<f32>,
}

impl ApproxIndex {
    /// Returns `true` when the prefilter reduces dimensionality enough to
    /// outperform exact search. A 3x reduction ratio is required because the
    /// two-phase approach (prefilter scan + scattered-access reranking) has
    /// higher constant overhead than a single sequential exact scan.
    pub(crate) fn provides_speedup(&self) -> bool {
        self.sample_dims.saturating_mul(3) <= self.dims
    }

    /// Construct the prefilter by selecting evenly-spaced dimension samples and building
    /// a reduced-dimension copy of the matrix.
    pub(crate) fn build(matrix: &[f32], rows: usize, dims: usize) -> Self {
        let sample_dims = choose_sample_dims(dims);
        let sample_indices = evenly_spaced_indices(dims, sample_dims);
        let sampled_matrix = build_sampled_matrix(matrix, rows, dims, &sample_indices);

        Self {
            rows,
            dims,
            sample_dims,
            sample_indices,
            sampled_matrix,
        }
    }

    /// Score every row using only the sampled dimensions and return the indices of the
    /// top candidates for exact reranking.
    pub(crate) fn candidate_indices(&self, query: &[f32], k: usize) -> Vec<usize> {
        debug_assert_eq!(query.len(), self.dims);
        // Stack buffer avoids per-query heap allocation (sample_dims <= 64).
        let mut sampled_query_buffer = [0.0_f32; 64];
        for (&sample_index, sampled_value) in self
            .sample_indices
            .iter()
            .zip(sampled_query_buffer.iter_mut())
        {
            *sampled_value = query[sample_index];
        }
        let sampled_query = &sampled_query_buffer[..self.sample_dims];
        let candidate_count = self.candidate_count(k);

        #[cfg(feature = "cpu")]
        {
            if self.rows >= parallel_rows_threshold(self.sample_dims) {
                return self
                    .sampled_matrix
                    .par_chunks_exact(self.sample_dims)
                    .enumerate()
                    .fold(
                        || TopKAccumulator::new(candidate_count),
                        |mut accumulator, (row_index, sampled_row)| {
                            accumulator.push(row_index, dot_small(sampled_row, sampled_query));
                            accumulator
                        },
                    )
                    .reduce(
                        || TopKAccumulator::new(candidate_count),
                        |mut left, right| {
                            left.merge(right);
                            left
                        },
                    )
                    .into_sorted_vec()
                    .into_iter()
                    .map(|neighbor| neighbor.index)
                    .collect();
            }
        }

        let mut accumulator = TopKAccumulator::new(candidate_count);
        for (row_index, sampled_row) in self
            .sampled_matrix
            .chunks_exact(self.sample_dims)
            .enumerate()
        {
            accumulator.push(row_index, dot_small(sampled_row, sampled_query));
        }
        accumulator
            .into_sorted_vec()
            .into_iter()
            .map(|neighbor| neighbor.index)
            .collect()
    }

    /// Determine how many candidates to retrieve, scaling with k and dimension count.
    fn candidate_count(&self, k: usize) -> usize {
        let minimum = (k.saturating_mul(4)).max(48).min(self.rows);
        let maximum = (self.rows / 16).max(minimum).min(4_096);
        let multiplier = match self.dims {
            0..=255 => 10,
            256..=1023 => 7,
            _ => 5,
        };
        k.saturating_mul(multiplier)
            .clamp(minimum, maximum)
            .min(self.rows)
    }
}

/// Choose the number of sampled dimensions (capped at 64, rounded down to a multiple of 8).
fn choose_sample_dims(dims: usize) -> usize {
    let capped = dims.min(64);
    if capped <= 8 {
        return capped.max(1);
    }
    capped - (capped % 8)
}

/// Generate `sample_dims` evenly-spaced column indices from `[0, dims)`.
fn evenly_spaced_indices(dims: usize, sample_dims: usize) -> Vec<usize> {
    if sample_dims == dims {
        return (0..dims).collect();
    }

    (0..sample_dims)
        .map(|idx| ((idx * dims) / sample_dims).min(dims.saturating_sub(1)))
        .collect()
}

/// Extract only the sampled columns from each row into a compact reduced-dimension matrix.
fn build_sampled_matrix(
    matrix: &[f32],
    rows: usize,
    dims: usize,
    sample_indices: &[usize],
) -> Vec<f32> {
    let mut sampled = Vec::with_capacity(rows.saturating_mul(sample_indices.len()));
    for row in matrix.chunks_exact(dims).take(rows) {
        for &sample_index in sample_indices {
            sampled.push(row[sample_index]);
        }
    }
    sampled
}

/// Dot product for short sampled vectors, using SIMD when >= 16 elements.
#[inline]
fn dot_small(lhs: &[f32], rhs: &[f32]) -> f32 {
    #[cfg(feature = "cpu")]
    if lhs.len() >= 16
        && let Some(simd) = f32::dot(lhs, rhs)
    {
        return simd as f32;
    }

    let mut sum = 0.0f32;
    for idx in 0..lhs.len() {
        sum += lhs[idx] * rhs[idx];
    }
    sum
}

/// Minimum row count before candidate search switches from serial to parallel.
#[cfg(feature = "cpu")]
#[inline]
fn parallel_rows_threshold(sample_dims: usize) -> usize {
    if sample_dims >= 64 { 4_096 } else { 8_192 }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- choose_sample_dims ----

    #[test]
    fn choose_sample_dims_caps_at_64() {
        assert_eq!(choose_sample_dims(256), 64);
        assert_eq!(choose_sample_dims(1024), 64);
    }

    #[test]
    fn choose_sample_dims_rounds_to_multiple_of_8() {
        assert_eq!(choose_sample_dims(30), 24);
        assert_eq!(choose_sample_dims(17), 16);
        assert_eq!(choose_sample_dims(64), 64);
    }

    #[test]
    fn choose_sample_dims_small_values_pass_through() {
        assert_eq!(choose_sample_dims(1), 1);
        assert_eq!(choose_sample_dims(4), 4);
        assert_eq!(choose_sample_dims(8), 8);
    }

    // ---- evenly_spaced_indices ----

    #[test]
    fn evenly_spaced_identity_when_sample_equals_dims() {
        let indices = evenly_spaced_indices(8, 8);
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn evenly_spaced_indices_spread_across_range() {
        let indices = evenly_spaced_indices(100, 4);
        assert_eq!(indices.len(), 4);
        // Should produce evenly-spaced values spanning [0, 99].
        assert_eq!(indices[0], 0);
        assert!(indices[3] <= 99);
        // Verify monotonically increasing.
        for window in indices.windows(2) {
            assert!(window[0] < window[1]);
        }
    }

    #[test]
    fn evenly_spaced_indices_never_exceed_dims() {
        let indices = evenly_spaced_indices(10, 8);
        for &idx in &indices {
            assert!(idx < 10);
        }
    }

    // ---- provides_speedup ----

    #[test]
    fn provides_speedup_requires_3x_reduction() {
        let index = ApproxIndex::build(&vec![1.0; 1000 * 192], 1000, 192);
        // sample_dims for 192 = min(192, 64) = 64; 64 * 3 = 192 <= 192 => true
        assert!(index.provides_speedup());
    }

    #[test]
    fn provides_speedup_false_for_low_dims() {
        let index = ApproxIndex::build(&vec![1.0; 100 * 16], 100, 16);
        // sample_dims for 16 = 16; 16 * 3 = 48 > 16 => false
        assert!(!index.provides_speedup());
    }

    // ---- candidate_count ----

    #[test]
    fn candidate_count_never_exceeds_rows() {
        let index = ApproxIndex::build(&vec![1.0; 50 * 256], 50, 256);
        assert!(index.candidate_count(100) <= 50);
    }

    #[test]
    fn candidate_count_scales_with_k() {
        let index = ApproxIndex::build(&vec![1.0; 4096 * 256], 4096, 256);
        let small_k = index.candidate_count(5);
        let large_k = index.candidate_count(50);
        assert!(large_k >= small_k);
    }

    // ---- build_sampled_matrix ----

    #[test]
    fn build_sampled_matrix_extracts_correct_columns() {
        // 2 rows x 4 dims
        let matrix = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let indices = vec![0, 2];
        let sampled = build_sampled_matrix(&matrix, 2, 4, &indices);
        assert_eq!(sampled, vec![10.0, 30.0, 50.0, 70.0]);
    }
}
