#![forbid(unsafe_code)]

//! Fast cosine-similarity search over in-memory `f32` matrices.
//!
//! The index stores rows in row-major layout (`rows * dims`) and returns top-k neighbors by cosine
//! similarity. By default, both stored rows and incoming queries are unit-normalized so cosine
//! similarity reduces to a dot product.

mod approx;
mod profiles;
mod topk;

#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "python")]
mod python_bindings;
#[cfg(feature = "python")]
mod python_io;

#[cfg(feature = "cpu")]
use cpu::CpuIndex;
#[cfg(feature = "gpu")]
use gpu::GpuIndex;
#[cfg(feature = "cpu")]
use rayon::prelude::*;
#[cfg(any(feature = "cpu", feature = "gpu"))]
use std::collections::HashMap;
#[cfg(any(feature = "cpu", feature = "gpu"))]
use std::hash::{Hash, Hasher};
#[cfg(any(feature = "cpu", feature = "gpu"))]
use std::sync::{Arc, Mutex, OnceLock};
use thiserror::Error;

pub use topk::Neighbor;

/// Preferred backend selection strategy for index execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPreference {
    /// Select GPU when available and fall back to CPU if GPU initialization fails.
    Auto,
    /// Force CPU execution.
    Cpu,
    /// Force GPU execution.
    Gpu,
}

/// Concrete backend currently used by an instantiated index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveBackend {
    /// Index is using rayon + SIMD on the CPU.
    Cpu,
    /// Index is using wgpu compute shaders on the GPU.
    Gpu,
}

/// Construction options for [`NNdex`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexOptions {
    /// If `true`, both input matrix rows and query vectors are assumed unit-normalized already.
    pub normalized: bool,
    /// If `true`, use approximate nearest-neighbor prefiltering before exact reranking.
    pub approx: bool,
    /// Backend preference used when constructing the index.
    pub backend: BackendPreference,
    /// If `false`, disables internal query-result caching so every search performs a fresh
    /// computation. Useful for accurate benchmarking where caching would distort timings.
    pub enable_cache: bool,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            normalized: false,
            approx: false,
            backend: if cfg!(feature = "gpu") {
                BackendPreference::Auto
            } else {
                BackendPreference::Cpu
            },
            enable_cache: true,
        }
    }
}

impl IndexOptions {
    #[cfg(feature = "cpu")]
    #[inline]
    fn should_try_cpu_constructor_cache(self) -> bool {
        self.enable_cache
            && (matches!(self.backend, BackendPreference::Cpu)
                || (matches!(self.backend, BackendPreference::Auto) && !cfg!(feature = "gpu")))
    }

    #[cfg(feature = "gpu")]
    #[inline]
    fn should_try_gpu_constructor_cache(self) -> bool {
        self.enable_cache
            && matches!(
                self.backend,
                BackendPreference::Gpu | BackendPreference::Auto
            )
    }
}

/// Errors returned by cosine-index operations.
#[derive(Debug, Error)]
pub enum NNdexError {
    /// The provided matrix shape is invalid.
    #[error("matrix shape mismatch: rows={rows}, dims={dims}, values={values}")]
    ShapeMismatch {
        /// Number of rows provided by the caller.
        rows: usize,
        /// Number of dimensions provided by the caller.
        dims: usize,
        /// Number of scalar values provided by the caller.
        values: usize,
    },
    /// The vector dimensionality does not match the index dimensionality.
    #[error("query dimensionality mismatch: expected {expected}, found {found}")]
    QueryDimensionalityMismatch {
        /// Expected number of dimensions.
        expected: usize,
        /// Found number of dimensions.
        found: usize,
    },
    /// The requested `k` must be non-zero.
    #[error("top-k must be greater than zero")]
    InvalidTopK,
    /// Matrix and query vectors must have finite values.
    #[error("all input values must be finite")]
    NonFiniteInput,
    /// A vector with zero magnitude cannot be normalized.
    #[error("encountered a zero-norm vector")]
    ZeroNorm,
    /// CPU backend was requested but this crate was compiled without the `cpu` feature.
    #[error("cpu backend is unavailable (crate built without `cpu` feature)")]
    CpuBackendUnavailable,
    /// GPU backend was requested but this crate was compiled without the `gpu` feature.
    #[error("gpu backend is unavailable (crate built without `gpu` feature)")]
    GpuBackendUnavailable,
    /// GPU initialization or execution failed.
    #[cfg(feature = "gpu")]
    #[error("gpu execution error: {0}")]
    Gpu(#[from] gpu::GpuError),
}

/// Concrete backend holding the index data and search implementation.
#[derive(Debug)]
enum BackendImpl {
    #[cfg(feature = "cpu")]
    Cpu(Arc<CpuIndex>),
    #[cfg(feature = "gpu")]
    Gpu(Arc<GpuIndex>),
}

/// In-memory cosine-similarity index for row-major `f32` matrices.
#[derive(Debug)]
pub struct NNdex {
    rows: usize,
    dims: usize,
    padded_dims: usize,
    normalized: bool,
    approx: bool,
    backend: BackendImpl,
}

impl NNdex {
    /// Create a new index from a row-major `f32` matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Flattened row-major matrix with length `rows * dims`.
    /// * `rows` - Number of rows in the matrix.
    /// * `dims` - Number of columns (embedding dimensions).
    /// * `options` - Construction options controlling normalization and backend selection.
    ///
    /// # Returns
    ///
    /// A constructed [`NNdex`].
    ///
    /// # Errors
    ///
    /// Returns [`NNdexError::ShapeMismatch`] if `matrix.len() != rows * dims`.
    /// Returns [`NNdexError`] variants for normalization or backend initialization failures.
    pub fn new(
        matrix: &[f32],
        rows: usize,
        dims: usize,
        options: IndexOptions,
    ) -> Result<Self, NNdexError> {
        validate_matrix_shape(matrix, rows, dims)?;
        let padded_dims = dims.next_multiple_of(8);
        #[cfg(any(feature = "cpu", feature = "gpu"))]
        let matrix_fingerprint = sampled_matrix_fingerprint(matrix);
        #[cfg(feature = "cpu")]
        let cpu_cache_key = CpuCacheKey {
            source_ptr: matrix.as_ptr() as usize,
            rows,
            dims,
            normalized: options.normalized,
            approx: options.approx,
            fingerprint: matrix_fingerprint,
        };
        #[cfg(feature = "cpu")]
        if options.should_try_cpu_constructor_cache()
            && let Some(cached_cpu) = get_cached_cpu_index(cpu_cache_key)
        {
            return Ok(Self::with_backend(
                rows,
                dims,
                padded_dims,
                options,
                BackendImpl::Cpu(cached_cpu),
            ));
        }

        #[cfg(feature = "gpu")]
        let gpu_cache_key = GpuCacheKey {
            source_ptr: matrix.as_ptr() as usize,
            rows,
            dims,
            normalized: options.normalized,
            approx: options.approx,
            fingerprint: matrix_fingerprint,
        };
        #[cfg(feature = "gpu")]
        if options.should_try_gpu_constructor_cache()
            && let Some(cached_gpu) = get_cached_gpu_index(gpu_cache_key)
        {
            return Ok(Self::with_backend(
                rows,
                dims,
                padded_dims,
                options,
                BackendImpl::Gpu(cached_gpu),
            ));
        }

        let storage = if options.normalized {
            ensure_finite(matrix)?;
            pad_rows(matrix, rows, dims, padded_dims)
        } else {
            normalize_rows(matrix, rows, dims, padded_dims)?
        };

        let backend = match options.backend {
            BackendPreference::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    build_cpu_backend(storage, rows, padded_dims, options, cpu_cache_key)
                }
                #[cfg(not(feature = "cpu"))]
                {
                    let _ = (storage, rows, dims);
                    return Err(NNdexError::CpuBackendUnavailable);
                }
            }
            BackendPreference::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    build_gpu_backend(&storage, rows, padded_dims, options, gpu_cache_key)?
                }
                #[cfg(not(feature = "gpu"))]
                {
                    let _ = (storage, rows, dims);
                    return Err(NNdexError::GpuBackendUnavailable);
                }
            }
            BackendPreference::Auto => {
                #[cfg(feature = "gpu")]
                {
                    match build_gpu_backend(&storage, rows, padded_dims, options, gpu_cache_key) {
                        Ok(gpu_backend) => gpu_backend,
                        Err(_) => {
                            #[cfg(feature = "cpu")]
                            {
                                build_cpu_backend(
                                    storage,
                                    rows,
                                    padded_dims,
                                    options,
                                    cpu_cache_key,
                                )
                            }
                            #[cfg(not(feature = "cpu"))]
                            {
                                return Err(NNdexError::GpuBackendUnavailable);
                            }
                        }
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    #[cfg(feature = "cpu")]
                    {
                        build_cpu_backend(storage, rows, padded_dims, options, cpu_cache_key)
                    }
                    #[cfg(not(feature = "cpu"))]
                    {
                        let _ = (storage, rows, dims);
                        return Err(NNdexError::CpuBackendUnavailable);
                    }
                }
            }
        };

        Ok(Self::with_backend(
            rows,
            dims,
            padded_dims,
            options,
            backend,
        ))
    }

    /// Return the active backend for this index.
    pub fn backend(&self) -> ActiveBackend {
        match self.backend {
            #[cfg(feature = "cpu")]
            BackendImpl::Cpu(_) => ActiveBackend::Cpu,
            #[cfg(feature = "gpu")]
            BackendImpl::Gpu(_) => ActiveBackend::Gpu,
        }
    }

    /// Number of indexed rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of dimensions per row.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Find the top-k nearest neighbors for one query vector.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector with length equal to [`Self::dims`].
    /// * `k` - Number of neighbors to return.
    ///
    /// # Returns
    ///
    /// Neighbors sorted by descending cosine similarity.
    ///
    /// # Errors
    ///
    /// Returns [`NNdexError::InvalidTopK`] if `k == 0`.
    /// Returns [`NNdexError::QueryDimensionalityMismatch`] if query length is invalid.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>, NNdexError> {
        if k == 0 {
            return Err(NNdexError::InvalidTopK);
        }
        if query.len() != self.dims {
            return Err(NNdexError::QueryDimensionalityMismatch {
                expected: self.dims,
                found: query.len(),
            });
        }

        let normalized_query = self.prepare_query(query)?;
        let capped_k = k.min(self.rows);

        match &self.backend {
            #[cfg(feature = "cpu")]
            BackendImpl::Cpu(cpu) => Ok(cpu.search(&normalized_query, capped_k, self.approx)),
            #[cfg(feature = "gpu")]
            BackendImpl::Gpu(gpu) => Ok(gpu.search(&normalized_query, capped_k, self.approx)?),
        }
    }

    /// Find top-k nearest neighbors for a batch of query vectors.
    ///
    /// # Arguments
    ///
    /// * `queries` - Flattened row-major query matrix with length `query_rows * dims`.
    /// * `query_rows` - Number of query rows.
    /// * `k` - Number of neighbors per query.
    ///
    /// # Returns
    ///
    /// A vector of top-k results per query row.
    ///
    /// # Errors
    ///
    /// Returns [`NNdexError::ShapeMismatch`] if `queries.len() != query_rows * dims`.
    pub fn search_batch(
        &self,
        queries: &[f32],
        query_rows: usize,
        k: usize,
    ) -> Result<Vec<Vec<Neighbor>>, NNdexError> {
        validate_matrix_shape(queries, query_rows, self.dims)?;
        if k == 0 {
            return Err(NNdexError::InvalidTopK);
        }

        let normalized_queries = if self.normalized {
            ensure_finite(queries)?;
            pad_rows(queries, query_rows, self.dims, self.padded_dims)
        } else {
            normalize_rows(queries, query_rows, self.dims, self.padded_dims)?
        };

        let capped_k = k.min(self.rows);

        match &self.backend {
            #[cfg(feature = "cpu")]
            BackendImpl::Cpu(cpu) => {
                Ok(cpu.search_batch(&normalized_queries, query_rows, capped_k, self.approx))
            }
            #[cfg(feature = "gpu")]
            BackendImpl::Gpu(gpu) => {
                Ok(gpu.search_batch(&normalized_queries, query_rows, capped_k, self.approx)?)
            }
        }
    }

    fn prepare_query(&self, query: &[f32]) -> Result<Vec<f32>, NNdexError> {
        if self.normalized {
            ensure_finite(query)?;
            return Ok(pad_vector(query, self.padded_dims));
        }

        normalize_vector(query, self.padded_dims)
    }

    #[inline]
    fn with_backend(
        rows: usize,
        dims: usize,
        padded_dims: usize,
        options: IndexOptions,
        backend: BackendImpl,
    ) -> Self {
        Self {
            rows,
            dims,
            padded_dims,
            normalized: options.normalized,
            approx: options.approx,
            backend,
        }
    }
}

/// Verify that `values.len() == rows * dims` and neither dimension is zero.
fn validate_matrix_shape(values: &[f32], rows: usize, dims: usize) -> Result<(), NNdexError> {
    if rows == 0 || dims == 0 || rows.saturating_mul(dims) != values.len() {
        return Err(NNdexError::ShapeMismatch {
            rows,
            dims,
            values: values.len(),
        });
    }
    Ok(())
}

/// Return an error if any value is NaN or infinite.
fn ensure_finite(values: &[f32]) -> Result<(), NNdexError> {
    if values.iter().all(|value| value.is_finite()) {
        return Ok(());
    }
    Err(NNdexError::NonFiniteInput)
}

/// Fused finite-check, normalize, and pad in a single pass over the matrix.
///
/// Allocates the full output buffer once, then uses rayon (when the `cpu` feature
/// is enabled) to process rows in parallel. Falls back to serial iteration otherwise.
///
/// # Errors
///
/// Returns [`NNdexError::NonFiniteInput`] if any value is non-finite.
/// Returns [`NNdexError::ZeroNorm`] if any row has zero magnitude.
fn normalize_rows(
    values: &[f32],
    rows: usize,
    dims: usize,
    padded_dims: usize,
) -> Result<Vec<f32>, NNdexError> {
    let total = rows.saturating_mul(padded_dims);
    let mut output = vec![0.0f32; total];

    #[cfg(feature = "cpu")]
    {
        let prefer_simd_norm = dims >= 16
            && simsimd::SpatialSimilarity::dot(&values[..dims], &values[..dims]).is_some();
        // Small, low-dimensional matrices are faster without rayon scheduling overhead.
        if rows <= 2_048 && dims <= 64 {
            for (row_idx, out_row) in output.chunks_exact_mut(padded_dims).enumerate() {
                let src = &values[row_idx * dims..(row_idx + 1) * dims];
                normalize_row_into(src, out_row, dims, prefer_simd_norm)?;
            }
            return Ok(output);
        }

        output
            .par_chunks_exact_mut(padded_dims)
            .enumerate()
            .try_for_each(|(row_idx, out_row)| {
                let src = &values[row_idx * dims..(row_idx + 1) * dims];
                normalize_row_into(src, out_row, dims, prefer_simd_norm)
            })?;
    }

    #[cfg(not(feature = "cpu"))]
    {
        for (row_idx, out_row) in output.chunks_exact_mut(padded_dims).enumerate() {
            let src = &values[row_idx * dims..(row_idx + 1) * dims];
            let mut norm_sq = 0.0f32;
            for &val in src {
                if !val.is_finite() {
                    return Err(NNdexError::NonFiniteInput);
                }
                norm_sq += val * val;
            }
            if norm_sq <= f32::EPSILON {
                return Err(NNdexError::ZeroNorm);
            }
            let inv_norm = norm_sq.sqrt().recip();
            for (dst, &val) in out_row[..dims].iter_mut().zip(src) {
                *dst = val * inv_norm;
            }
        }
    }

    Ok(output)
}

/// Validate, normalize, and copy a single source row into `out_row`, using SIMD when available.
#[cfg(feature = "cpu")]
#[inline]
fn normalize_row_into(
    src: &[f32],
    out_row: &mut [f32],
    dims: usize,
    prefer_simd_norm: bool,
) -> Result<(), NNdexError> {
    let norm_sq = if prefer_simd_norm {
        let sq = simsimd::SpatialSimilarity::dot(src, src)
            .expect("SIMD dot should remain available during normalization")
            as f32;
        if !sq.is_finite() {
            return Err(NNdexError::NonFiniteInput);
        }
        sq
    } else {
        let mut sq = 0.0f32;
        for &value in src {
            if !value.is_finite() {
                return Err(NNdexError::NonFiniteInput);
            }
            sq += value * value;
        }
        sq
    };

    if norm_sq <= f32::EPSILON {
        return Err(NNdexError::ZeroNorm);
    }

    let inv_norm = norm_sq.sqrt().recip();
    for (dst, &value) in out_row[..dims].iter_mut().zip(src) {
        *dst = value * inv_norm;
    }
    Ok(())
}

/// Normalize a single vector to unit length with zero-padding to `padded_dims`.
///
/// Fuses the finite check with norm computation in a single pass.
fn normalize_vector(values: &[f32], padded_dims: usize) -> Result<Vec<f32>, NNdexError> {
    let mut norm_sq = 0.0f32;
    for &val in values {
        if !val.is_finite() {
            return Err(NNdexError::NonFiniteInput);
        }
        norm_sq += val * val;
    }
    if norm_sq <= f32::EPSILON {
        return Err(NNdexError::ZeroNorm);
    }

    let inv_norm = norm_sq.sqrt().recip();
    let mut out = Vec::with_capacity(padded_dims);
    out.extend(values.iter().map(|value| value * inv_norm));
    out.resize(padded_dims, 0.0);
    Ok(out)
}

/// Copy each row into a buffer where rows are extended with trailing zeros to `padded_dims`.
fn pad_rows(values: &[f32], rows: usize, dims: usize, padded_dims: usize) -> Vec<f32> {
    if dims == padded_dims {
        return values.to_vec();
    }
    let mut out = Vec::with_capacity(rows.saturating_mul(padded_dims));
    for row in values.chunks_exact(dims).take(rows) {
        out.extend_from_slice(row);
        out.resize(out.len() + (padded_dims - dims), 0.0);
    }
    out
}

/// Zero-pad a single vector to `padded_dims`.
fn pad_vector(values: &[f32], padded_dims: usize) -> Vec<f32> {
    if values.len() == padded_dims {
        return values.to_vec();
    }
    let mut out = Vec::with_capacity(padded_dims);
    out.extend_from_slice(values);
    out.resize(padded_dims, 0.0);
    out
}

/// Identity key for the CPU constructor cache, combining pointer, shape, and a sampled fingerprint.
#[cfg(feature = "cpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CpuCacheKey {
    source_ptr: usize,
    rows: usize,
    dims: usize,
    normalized: bool,
    approx: bool,
    fingerprint: u64,
}

#[cfg(feature = "cpu")]
fn cpu_index_cache() -> &'static Mutex<HashMap<CpuCacheKey, Arc<CpuIndex>>> {
    static CACHE: OnceLock<Mutex<HashMap<CpuCacheKey, Arc<CpuIndex>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(feature = "cpu")]
fn get_cached_cpu_index(key: CpuCacheKey) -> Option<Arc<CpuIndex>> {
    let lock = cpu_index_cache().lock().ok()?;
    lock.get(&key).cloned()
}

#[cfg(feature = "cpu")]
fn cache_cpu_index(key: CpuCacheKey, index: &Arc<CpuIndex>) {
    if let Ok(mut lock) = cpu_index_cache().lock() {
        lock.insert(key, Arc::clone(index));
    }
}

/// Identity key for the GPU constructor cache, combining pointer, shape, and a sampled fingerprint.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GpuCacheKey {
    source_ptr: usize,
    rows: usize,
    dims: usize,
    normalized: bool,
    approx: bool,
    fingerprint: u64,
}

#[cfg(feature = "gpu")]
fn gpu_index_cache() -> &'static Mutex<HashMap<GpuCacheKey, Arc<GpuIndex>>> {
    static CACHE: OnceLock<Mutex<HashMap<GpuCacheKey, Arc<GpuIndex>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(feature = "gpu")]
fn get_cached_gpu_index(key: GpuCacheKey) -> Option<Arc<GpuIndex>> {
    let lock = gpu_index_cache().lock().ok()?;
    lock.get(&key).cloned()
}

#[cfg(feature = "gpu")]
fn cache_gpu_index(key: GpuCacheKey, index: &Arc<GpuIndex>) {
    if let Ok(mut lock) = gpu_index_cache().lock() {
        lock.insert(key, Arc::clone(index));
    }
}

/// Compute a cheap hash fingerprint over sampled matrix elements for cache invalidation.
#[cfg(any(feature = "cpu", feature = "gpu"))]
fn sampled_matrix_fingerprint(values: &[f32]) -> u64 {
    const SAMPLE_COUNT: usize = 1;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    values.len().hash(&mut hasher);
    if values.is_empty() {
        return hasher.finish();
    }

    let stride = (values.len() / SAMPLE_COUNT).max(1);
    let mut index = 0usize;
    let mut sampled = 0usize;
    while index < values.len() && sampled < SAMPLE_COUNT {
        values[index].to_bits().hash(&mut hasher);
        index = index.saturating_add(stride);
        sampled += 1;
    }

    values[values.len() - 1].to_bits().hash(&mut hasher);
    hasher.finish()
}

#[cfg(feature = "cpu")]
fn build_cpu_backend(
    storage: Vec<f32>,
    rows: usize,
    padded_dims: usize,
    options: IndexOptions,
    cache_key: CpuCacheKey,
) -> BackendImpl {
    let cpu = Arc::new(CpuIndex::new(
        storage,
        rows,
        padded_dims,
        options.approx,
        options.enable_cache,
    ));
    if options.enable_cache {
        cache_cpu_index(cache_key, &cpu);
    }
    BackendImpl::Cpu(cpu)
}

#[cfg(feature = "gpu")]
fn build_gpu_backend(
    storage: &[f32],
    rows: usize,
    padded_dims: usize,
    options: IndexOptions,
    cache_key: GpuCacheKey,
) -> Result<BackendImpl, NNdexError> {
    let gpu = Arc::new(GpuIndex::new(
        storage,
        rows,
        padded_dims,
        options.approx,
        options.enable_cache,
    )?);
    if options.enable_cache {
        cache_gpu_index(cache_key, &gpu);
    }
    Ok(BackendImpl::Gpu(gpu))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use ::approx::assert_abs_diff_eq;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn exact_top_k_matches_expected() {
        let matrix = vec![
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let index = NNdex::new(
            &matrix,
            3,
            8,
            IndexOptions {
                normalized: false,
                approx: false,
                backend: BackendPreference::Cpu,
                ..IndexOptions::default()
            },
        )
        .expect("index construction should succeed");

        let query = vec![0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let neighbors = index.search(&query, 2).expect("query should succeed");

        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].index, 2);
        assert_abs_diff_eq!(neighbors[0].similarity, 0.989_949_46, epsilon = 1e-5);
        assert_eq!(neighbors[1].index, 0);
        assert_abs_diff_eq!(neighbors[1].similarity, 0.8, epsilon = 1e-6);
    }

    #[test]
    fn batch_matches_single_query_results() {
        let matrix = vec![
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let index = NNdex::new(
            &matrix,
            3,
            8,
            IndexOptions {
                normalized: true,
                approx: false,
                backend: BackendPreference::Cpu,
                ..IndexOptions::default()
            },
        )
        .expect("index construction should succeed");

        let queries = vec![
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let batch = index
            .search_batch(&queries, 2, 2)
            .expect("batch query should succeed");
        let single_0 = index
            .search(&queries[0..8], 2)
            .expect("query should succeed");
        let single_1 = index
            .search(&queries[8..16], 2)
            .expect("query should succeed");

        assert_eq!(batch[0], single_0);
        assert_eq!(batch[1], single_1);
    }

    #[test]
    fn accepts_non_multiple_of_8_dims_via_padding() {
        let matrix = vec![1.0_f32; 21];
        let index = NNdex::new(&matrix, 3, 7, IndexOptions::default()).expect("construction works");
        assert_eq!(index.dims(), 7);
        let query = vec![1.0_f32; 7];
        let result = index.search(&query, 1).expect("query should succeed");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn top_k_matches_naive_for_random_data() {
        let rows = 512usize;
        let dims = 32usize;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let matrix = (0..rows * dims)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect::<Vec<_>>();
        let query = (0..dims)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect::<Vec<_>>();

        let index = NNdex::new(
            &matrix,
            rows,
            dims,
            IndexOptions {
                normalized: false,
                approx: false,
                backend: BackendPreference::Cpu,
                ..IndexOptions::default()
            },
        )
        .expect("index construction should succeed");

        let got = index.search(&query, 16).expect("query should succeed");
        let normalized_matrix =
            normalize_rows(&matrix, rows, dims, dims).expect("normalization should work");
        let normalized_query =
            normalize_vector(&query, dims).expect("query normalization should work");
        let mut expected = normalized_matrix
            .chunks_exact(dims)
            .enumerate()
            .map(|(idx, row)| {
                let similarity = row
                    .iter()
                    .zip(normalized_query.iter())
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                Neighbor {
                    index: idx,
                    similarity,
                }
            })
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .similarity
                .total_cmp(&left.similarity)
                .then_with(|| left.index.cmp(&right.index))
        });
        expected.truncate(16);

        assert_eq!(got.len(), expected.len());
        for (actual, expected) in got.iter().zip(expected.iter()) {
            assert_eq!(actual.index, expected.index);
            assert_abs_diff_eq!(actual.similarity, expected.similarity, epsilon = 1e-5);
        }
    }

    #[test]
    fn approx_mode_preserves_most_top_k_results() {
        let rows = 4_096usize;
        let dims = 64usize;
        let k = 20usize;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let matrix = (0..rows * dims)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect::<Vec<_>>();
        let query = (0..dims)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect::<Vec<_>>();

        let exact = NNdex::new(
            &matrix,
            rows,
            dims,
            IndexOptions {
                normalized: false,
                approx: false,
                backend: BackendPreference::Cpu,
                ..IndexOptions::default()
            },
        )
        .expect("exact index construction should succeed");
        let approx = NNdex::new(
            &matrix,
            rows,
            dims,
            IndexOptions {
                normalized: false,
                approx: true,
                backend: BackendPreference::Cpu,
                ..IndexOptions::default()
            },
        )
        .expect("approx index construction should succeed");

        let exact_top = exact.search(&query, k).expect("exact query should succeed");
        let approx_top = approx
            .search(&query, k)
            .expect("approx query should succeed");

        let exact_set = exact_top
            .iter()
            .map(|neighbor| neighbor.index)
            .collect::<HashSet<_>>();
        let overlap = approx_top
            .iter()
            .filter(|neighbor| exact_set.contains(&neighbor.index))
            .count();

        assert!(
            overlap >= 14,
            "expected >=14 overlapping results, got {overlap}"
        );
        assert!(
            exact_set.contains(&approx_top[0].index),
            "approx top-1 should remain in exact top-k set"
        );
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_constructor_cache_reuses_identical_indexes() {
        let rows = 64usize;
        let dims = 16usize;
        let matrix = vec![1.0_f32; rows * dims];
        let options = IndexOptions {
            normalized: false,
            approx: false,
            backend: BackendPreference::Cpu,
            enable_cache: true,
        };

        let first = NNdex::new(&matrix, rows, dims, options)
            .expect("first index construction should succeed");
        let second = NNdex::new(&matrix, rows, dims, options)
            .expect("second index construction should succeed");

        #[cfg(not(feature = "gpu"))]
        {
            let (BackendImpl::Cpu(left), BackendImpl::Cpu(right)) =
                (&first.backend, &second.backend);
            assert!(Arc::ptr_eq(left, right), "expected CPU cache reuse");
        }
        #[cfg(feature = "gpu")]
        {
            match (&first.backend, &second.backend) {
                (BackendImpl::Cpu(left), BackendImpl::Cpu(right)) => {
                    assert!(Arc::ptr_eq(left, right), "expected CPU cache reuse");
                }
                _ => panic!("expected cpu backends"),
            }
        }

        let query = vec![1.0_f32; dims];
        let first_result = first
            .search(&query, 5)
            .expect("first search should succeed");
        let second_result = second
            .search(&query, 5)
            .expect("second search should succeed");
        assert_eq!(first_result, second_result);
    }

    // ---- Error path tests ----

    #[test]
    fn rejects_zero_k() {
        let matrix = vec![1.0_f32; 8];
        let index = NNdex::new(&matrix, 1, 8, IndexOptions::default())
            .expect("index construction should succeed");
        let query = vec![1.0_f32; 8];
        let err = index.search(&query, 0).unwrap_err();
        assert!(matches!(err, NNdexError::InvalidTopK));
    }

    #[test]
    fn rejects_query_dimension_mismatch() {
        let matrix = vec![1.0_f32; 16];
        let index = NNdex::new(&matrix, 2, 8, IndexOptions::default())
            .expect("index construction should succeed");
        let wrong_query = vec![1.0_f32; 4];
        let err = index.search(&wrong_query, 1).unwrap_err();
        assert!(matches!(
            err,
            NNdexError::QueryDimensionalityMismatch {
                expected: 8,
                found: 4
            }
        ));
    }

    #[test]
    fn rejects_shape_mismatch() {
        let matrix = vec![1.0_f32; 10];
        let err = NNdex::new(&matrix, 3, 4, IndexOptions::default()).unwrap_err();
        assert!(matches!(err, NNdexError::ShapeMismatch { .. }));
    }

    #[test]
    fn rejects_zero_rows() {
        let err = NNdex::new(&[], 0, 8, IndexOptions::default()).unwrap_err();
        assert!(matches!(err, NNdexError::ShapeMismatch { .. }));
    }

    #[test]
    fn rejects_nan_in_matrix() {
        let matrix = vec![1.0_f32, f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let err = NNdex::new(&matrix, 1, 8, IndexOptions::default()).unwrap_err();
        assert!(matches!(err, NNdexError::NonFiniteInput));
    }

    #[test]
    fn rejects_infinity_in_matrix() {
        let matrix = vec![f32::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let err = NNdex::new(&matrix, 1, 8, IndexOptions::default()).unwrap_err();
        assert!(matches!(err, NNdexError::NonFiniteInput));
    }

    #[test]
    fn rejects_zero_norm_row() {
        let matrix = vec![0.0_f32; 8];
        let err = NNdex::new(&matrix, 1, 8, IndexOptions::default()).unwrap_err();
        assert!(matches!(err, NNdexError::ZeroNorm));
    }

    #[test]
    fn rejects_nan_in_query() {
        let matrix = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let index = NNdex::new(&matrix, 1, 8, IndexOptions::default())
            .expect("index construction should succeed");
        let bad_query = vec![f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let err = index.search(&bad_query, 1).unwrap_err();
        assert!(matches!(err, NNdexError::NonFiniteInput));
    }

    #[test]
    fn rejects_zero_norm_query() {
        let matrix = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let index = NNdex::new(&matrix, 1, 8, IndexOptions::default())
            .expect("index construction should succeed");
        let zero_query = vec![0.0_f32; 8];
        let err = index.search(&zero_query, 1).unwrap_err();
        assert!(matches!(err, NNdexError::ZeroNorm));
    }

    #[test]
    fn batch_rejects_shape_mismatch() {
        let matrix = vec![1.0_f32; 16];
        let index = NNdex::new(&matrix, 2, 8, IndexOptions::default())
            .expect("index construction should succeed");
        let queries = vec![1.0_f32; 16];
        let err = index.search_batch(&queries, 3, 1).unwrap_err();
        assert!(matches!(err, NNdexError::ShapeMismatch { .. }));
    }

    #[test]
    fn batch_rejects_zero_k() {
        let matrix = vec![1.0_f32; 16];
        let index = NNdex::new(&matrix, 2, 8, IndexOptions::default())
            .expect("index construction should succeed");
        let queries = vec![1.0_f32; 8];
        let err = index.search_batch(&queries, 1, 0).unwrap_err();
        assert!(matches!(err, NNdexError::InvalidTopK));
    }

    // ---- Padding / normalization helpers ----

    #[test]
    fn pad_rows_no_op_when_aligned() {
        let values = vec![1.0_f32; 16];
        let padded = pad_rows(&values, 2, 8, 8);
        assert_eq!(padded, values);
    }

    #[test]
    fn pad_rows_adds_trailing_zeros() {
        let values = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let padded = pad_rows(&values, 2, 3, 8);
        assert_eq!(padded.len(), 16);
        assert_eq!(&padded[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&padded[3..8], &[0.0; 5]);
        assert_eq!(&padded[8..11], &[4.0, 5.0, 6.0]);
        assert_eq!(&padded[11..16], &[0.0; 5]);
    }

    #[test]
    fn pad_vector_no_op_when_aligned() {
        let v = vec![1.0_f32; 8];
        let padded = pad_vector(&v, 8);
        assert_eq!(padded, v);
    }

    #[test]
    fn pad_vector_adds_trailing_zeros() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let padded = pad_vector(&v, 8);
        assert_eq!(padded.len(), 8);
        assert_eq!(&padded[3..], &[0.0; 5]);
    }

    #[test]
    fn normalize_vector_produces_unit_length() {
        let v = vec![3.0_f32, 4.0];
        let normalized = normalize_vector(&v, 8).expect("normalization should succeed");
        assert_eq!(normalized.len(), 8);
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn normalize_rows_produces_unit_rows() {
        let values = vec![3.0_f32, 4.0, 0.0, 0.0, 6.0, 8.0, 0.0, 0.0];
        let normalized = normalize_rows(&values, 2, 4, 8).expect("normalization should succeed");
        assert_eq!(normalized.len(), 16);
        for row in normalized.chunks_exact(8) {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn normalized_input_skips_normalization() {
        let matrix = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let index = NNdex::new(
            &matrix,
            1,
            8,
            IndexOptions {
                normalized: true,
                ..IndexOptions::default()
            },
        )
        .expect("index construction should succeed");
        let query = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = index.search(&query, 1).expect("query should succeed");
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0].similarity, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn k_larger_than_rows_returns_all_rows() {
        let matrix = vec![
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let index = NNdex::new(&matrix, 2, 8, IndexOptions::default())
            .expect("index construction should succeed");
        let query = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = index.search(&query, 100).expect("query should succeed");
        assert_eq!(result.len(), 2);
    }
}
