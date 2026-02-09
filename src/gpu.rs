use std::borrow::Cow;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::approx::ApproxIndex;
use crate::profiles::{
    CachePolicy, TuningProfile, prefer_gpu_exact_for_batch as should_prefer_gpu_exact_for_batch,
    should_use_gpu_ann_prefilter,
};
use crate::topk::{Neighbor, TopKAccumulator};

const WORKGROUP_SIZE: u32 = 256;

/// Dimension threshold: dims <= this value use the low-dim (1-thread-per-row) shader;
/// larger dims use the high-dim (1-workgroup-per-row cooperative reduction) shader.
const HIGH_DIM_THRESHOLD: usize = 1024;

/// Maximum entries in the single-query result cache.
const SINGLE_CACHE_CAPACITY: usize = 128;

/// Maximum entries in the batch-query result cache.
const BATCH_CACHE_CAPACITY: usize = 16;

/// Enable reusable single-query GPU buffers for matrices up to this row count.
///
/// This avoids per-search buffer/bind-group allocation overhead in common online-query paths while
/// capping extra memory usage for very large matrices.
const SINGLE_QUERY_REUSE_MAX_ROWS: usize = 1_000_000;

/// Errors specific to GPU index initialization and dispatch.
#[derive(Debug, Error)]
pub enum GpuError {
    /// No wgpu adapter matching the requested power preference was found.
    #[error("no compatible GPU adapter was found")]
    AdapterNotFound,
    /// The adapter was found but device creation failed (e.g. unsupported limits).
    #[error("failed to request GPU device: {0}")]
    RequestDevice(String),
    /// The output buffer could not be mapped back to the CPU for reading.
    #[error("failed to map GPU buffer for reading")]
    MapReadFailed,
}

/// Shader uniform parameters matching the WGSL `Params` struct layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ShaderParams {
    rows: u32,
    dims_vec4: u32,
    query_count: u32,
    _pad: u32,
}

/// A contiguous slice of the matrix uploaded to a single GPU storage buffer.
#[derive(Debug)]
struct MatrixChunk {
    row_start: usize,
    row_count: usize,
    matrix_buffer: wgpu::Buffer,
}

/// Pre-allocated GPU buffers and bind group for a single matrix chunk in the reusable
/// single-query path, avoiding per-search allocation overhead.
#[derive(Debug)]
struct SingleQueryChunkDispatch {
    row_start: usize,
    row_count: usize,
    output_size: u64,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Reusable dispatch state for single-query searches: one shared query buffer and
/// pre-built chunk dispatches.
#[derive(Debug)]
struct SingleQueryDispatch {
    query_buffer: wgpu::Buffer,
    chunks: Vec<SingleQueryChunkDispatch>,
}

/// GPU-backed nearest-neighbor index using wgpu compute shaders for dot-product scoring.
#[derive(Debug)]
pub(crate) struct GpuIndex {
    device: wgpu::Device,
    queue: wgpu::Queue,
    rows: usize,
    dims: usize,
    /// dims rounded up to multiple of 4 for vec4 shader access.
    dims_vec4: usize,
    chunks: Vec<MatrixChunk>,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_lowdim: wgpu::ComputePipeline,
    pipeline_highdim: wgpu::ComputePipeline,
    profile: TuningProfile,
    host_matrix: Vec<f32>,
    approx_index: Option<OnceLock<ApproxIndex>>,
    /// Bounded cache for single-query results keyed by query hash.
    cached_single: Mutex<HashMap<u64, Vec<Neighbor>>>,
    /// Bounded cache for batch-query results keyed by query hash.
    cached_batch: Mutex<HashMap<u64, Vec<Vec<Neighbor>>>>,
    /// Max queries that fit in one sub-batch before output buffer exceeds adapter limits.
    max_queries_per_sub_batch: usize,
    /// Reusable buffers for single-query exact GPU dispatch.
    single_query_dispatch: Option<Mutex<SingleQueryDispatch>>,
}

impl GpuIndex {
    /// Create a new GPU index from a row-major matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Flattened row-major matrix (already padded to `dims`).
    /// * `rows` - Number of rows.
    /// * `dims` - Number of dimensions per row (already padded to multiple of 8).
    /// * `approx` - Whether to build the ANN prefilter index.
    /// * `enable_cache` - Whether to enable query-result caching.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::AdapterNotFound`] if no GPU is available.
    /// Returns [`GpuError::RequestDevice`] on device creation failure.
    pub(crate) fn new(
        matrix: &[f32],
        rows: usize,
        dims: usize,
        approx: bool,
        enable_cache: bool,
    ) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|_| GpuError::AdapterNotFound)?;

        let requested_limits = adapter.limits();
        let max_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size as usize;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("nndex-device"),
            required_features: wgpu::Features::empty(),
            required_limits: requested_limits,
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::default(),
        }))
        .map_err(|error| GpuError::RequestDevice(error.to_string()))?;

        let dims_vec4 = dims.next_multiple_of(4);

        let shader_lowdim = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nndex-shader-lowdim"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/dot_lowdim.wgsl"
            ))),
        });
        let shader_highdim = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nndex-shader-highdim"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/dot_highdim.wgsl"
            ))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nndex-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nndex-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline_lowdim = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nndex-pipeline-lowdim"),
            layout: Some(&pipeline_layout),
            module: &shader_lowdim,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let pipeline_highdim = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nndex-pipeline-highdim"),
            layout: Some(&pipeline_layout),
            module: &shader_highdim,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let padded_matrix = pad_to_vec4(matrix, rows, dims, dims_vec4);

        let max_binding_bytes =
            max_buffer_binding_size.max(dims_vec4.saturating_mul(std::mem::size_of::<f32>()));
        let bytes_per_row = dims_vec4.saturating_mul(std::mem::size_of::<f32>());
        let rows_per_chunk = (max_binding_bytes / bytes_per_row).max(1);

        let mut chunks = Vec::new();
        let mut row_start = 0usize;
        while row_start < rows {
            let row_count = (rows - row_start).min(rows_per_chunk);
            let start = row_start * dims_vec4;
            let end = start + row_count * dims_vec4;
            let matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nndex-matrix-chunk"),
                contents: bytemuck::cast_slice(&padded_matrix[start..end]),
                usage: wgpu::BufferUsages::STORAGE,
            });

            chunks.push(MatrixChunk {
                row_start,
                row_count,
                matrix_buffer,
            });
            row_start += row_count;
        }

        let max_chunk_rows = chunks.iter().map(|c| c.row_count).max().unwrap_or(1);
        let bytes_per_query = max_chunk_rows * std::mem::size_of::<f32>();
        let max_queries_per_sub_batch = (max_buffer_binding_size / bytes_per_query).max(1);
        let single_query_dispatch = (rows <= SINGLE_QUERY_REUSE_MAX_ROWS).then(|| {
            Mutex::new(build_single_query_dispatch(
                &device,
                &bind_group_layout,
                &chunks,
                dims_vec4,
            ))
        });

        Ok(Self {
            device,
            queue,
            rows,
            dims,
            dims_vec4,
            chunks,
            bind_group_layout,
            pipeline_lowdim,
            pipeline_highdim,
            profile: TuningProfile::for_shape(
                rows,
                dims,
                1,
                CachePolicy::from_enabled(enable_cache),
            ),
            host_matrix: if approx { matrix.to_vec() } else { Vec::new() },
            approx_index: approx.then(OnceLock::new),
            cached_single: Mutex::new(HashMap::with_capacity(SINGLE_CACHE_CAPACITY)),
            cached_batch: Mutex::new(HashMap::with_capacity(BATCH_CACHE_CAPACITY)),
            max_queries_per_sub_batch,
            single_query_dispatch,
        })
    }

    /// Search for the top-k nearest neighbors of a single query.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::MapReadFailed`] on GPU readback failure.
    pub(crate) fn search(
        &self,
        query: &[f32],
        k: usize,
        approx: bool,
    ) -> Result<Vec<Neighbor>, GpuError> {
        if self.profile.cache_enabled() {
            let cache_key = self.query_hash(query, k, 1, approx);
            if let Ok(guard) = self.cached_single.lock()
                && let Some(cached) = guard.get(&cache_key)
            {
                return Ok(cached.clone());
            }

            let capped_k = k.min(self.rows);
            let result = self.dispatch_single_or_approx(query, capped_k, approx)?;

            if let Ok(mut guard) = self.cached_single.lock() {
                if guard.len() >= SINGLE_CACHE_CAPACITY {
                    guard.clear();
                }
                guard.insert(cache_key, result.clone());
            }
            return Ok(result);
        }

        let capped_k = k.min(self.rows);
        self.dispatch_single_or_approx(query, capped_k, approx)
    }

    /// Route a single query through CPU-side ANN prefilter or GPU exact dispatch.
    fn dispatch_single_or_approx(
        &self,
        query: &[f32],
        k: usize,
        approx: bool,
    ) -> Result<Vec<Neighbor>, GpuError> {
        if approx
            && should_use_gpu_ann_prefilter(self.rows, self.dims)
            && let Some(approx_index) = self.approx_index()
        {
            return Ok(self.search_approx_cpu(query, k, approx_index));
        }
        self.search_exact_single(query, k)
    }

    /// Search for the top-k nearest neighbors of a batch of queries.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::MapReadFailed`] on GPU readback failure.
    pub(crate) fn search_batch(
        &self,
        queries: &[f32],
        query_rows: usize,
        k: usize,
        approx: bool,
    ) -> Result<Vec<Vec<Neighbor>>, GpuError> {
        if self.profile.cache_enabled() {
            let cache_key = self.query_hash(queries, k, query_rows, approx);
            if let Ok(guard) = self.cached_batch.lock()
                && let Some(cached) = guard.get(&cache_key)
            {
                return Ok(cached.clone());
            }

            let out = self.dispatch_batch_or_approx(queries, query_rows, k, approx)?;

            if let Ok(mut guard) = self.cached_batch.lock() {
                if guard.len() >= BATCH_CACHE_CAPACITY {
                    guard.clear();
                }
                guard.insert(cache_key, out.clone());
            }
            return Ok(out);
        }

        self.dispatch_batch_or_approx(queries, query_rows, k, approx)
    }

    /// Dispatch batch query via GPU exact or CPU-side ANN, depending on heuristics.
    fn dispatch_batch_or_approx(
        &self,
        queries: &[f32],
        query_rows: usize,
        k: usize,
        approx: bool,
    ) -> Result<Vec<Vec<Neighbor>>, GpuError> {
        let capped_k = k.min(self.rows);
        if approx
            && should_use_gpu_ann_prefilter(self.rows, self.dims)
            && !self.prefer_gpu_exact_for_batch(query_rows)
            && let Some(approx_index) = self.approx_index()
        {
            #[cfg(feature = "cpu")]
            {
                use rayon::prelude::*;
                return Ok(queries
                    .par_chunks_exact(self.dims)
                    .take(query_rows)
                    .map(|query| self.search_approx_cpu(query, capped_k, approx_index))
                    .collect());
            }
            #[cfg(not(feature = "cpu"))]
            {
                return Ok(queries
                    .chunks_exact(self.dims)
                    .take(query_rows)
                    .map(|query| self.search_approx_cpu(query, capped_k, approx_index))
                    .collect());
            }
        }
        self.search_exact_batch(queries, query_rows, capped_k)
    }

    /// Returns the ANN prefilter index only if it actually provides a speedup.
    /// Falls through to GPU exact search when prefiltering would not reduce work.
    #[inline]
    fn approx_index(&self) -> Option<&ApproxIndex> {
        self.approx_index.as_ref().and_then(|index| {
            let built =
                index.get_or_init(|| ApproxIndex::build(&self.host_matrix, self.rows, self.dims));
            built.provides_speedup().then_some(built)
        })
    }

    /// Rerank ANN candidates using CPU dot products.
    ///
    /// For small candidate sets (typically <=4096 rows), CPU SIMD dot products are
    /// faster than GPU dispatch overhead.
    fn search_approx_cpu(
        &self,
        query: &[f32],
        k: usize,
        approx_index: &ApproxIndex,
    ) -> Vec<Neighbor> {
        let candidates = approx_index.candidate_indices(query, k.min(self.rows));
        let mut accumulator = TopKAccumulator::new(k.min(self.rows));
        for &candidate in &candidates {
            let start = candidate * self.dims;
            let end = start + self.dims;
            let row = &self.host_matrix[start..end];
            accumulator.push(candidate, cpu_dot(row, query));
        }
        accumulator.into_sorted_vec()
    }

    /// Check whether GPU exact dispatch would be faster than CPU-side ANN for
    /// a batch of `query_count` queries.
    ///
    /// GPU exact is preferred when the total number of sub-batch dispatches is
    /// small enough that dispatch + readback overhead stays below the cost of
    /// CPU-side ANN candidate search.
    #[inline]
    fn prefer_gpu_exact_for_batch(&self, query_count: usize) -> bool {
        should_prefer_gpu_exact_for_batch(
            query_count,
            self.max_queries_per_sub_batch,
            self.chunks.len(),
        )
    }

    /// Select the appropriate pipeline based on dimensionality.
    #[inline]
    fn select_pipeline(&self) -> &wgpu::ComputePipeline {
        if self.dims <= HIGH_DIM_THRESHOLD {
            &self.pipeline_lowdim
        } else {
            &self.pipeline_highdim
        }
    }

    /// Compute workgroup dispatch dimensions for a given row count and query count.
    #[inline]
    fn dispatch_dims(&self, chunk_rows: u32, query_count: u32) -> (u32, u32) {
        if self.dims <= HIGH_DIM_THRESHOLD {
            let wg_x = chunk_rows.div_ceil(WORKGROUP_SIZE);
            (wg_x, query_count)
        } else {
            (chunk_rows, query_count)
        }
    }

    /// Execute an exact GPU search for a single query, using reusable buffers when available.
    fn search_exact_single(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>, GpuError> {
        let padded_query = if self.dims == self.dims_vec4 {
            Cow::Borrowed(query)
        } else {
            Cow::Owned(pad_query_vec4(query, self.dims, self.dims_vec4))
        };
        if let Some(reused_dispatch) = &self.single_query_dispatch {
            return self.dispatch_single_reused(padded_query.as_ref(), k, reused_dispatch);
        }
        let results = self.dispatch_batched(padded_query.as_ref(), 1, k)?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    /// Dispatch a single query through the pre-allocated reusable buffer path.
    fn dispatch_single_reused(
        &self,
        padded_query: &[f32],
        k: usize,
        reused_dispatch: &Mutex<SingleQueryDispatch>,
    ) -> Result<Vec<Neighbor>, GpuError> {
        let dispatch = match reused_dispatch.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        self.queue.write_buffer(
            &dispatch.query_buffer,
            0,
            bytemuck::cast_slice(padded_query),
        );
        let pipeline = self.select_pipeline();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nndex-single-encoder"),
            });

        for chunk in &dispatch.chunks {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("nndex-single-compute-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &chunk.bind_group, &[]);
                let (wg_x, wg_y) = self.dispatch_dims(chunk.row_count as u32, 1);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            encoder.copy_buffer_to_buffer(
                &chunk.output_buffer,
                0,
                &chunk.readback_buffer,
                0,
                chunk.output_size,
            );
        }

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let mut accumulator = TopKAccumulator::new(k);
        for chunk in &dispatch.chunks {
            let slice = chunk.readback_buffer.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

            let mapped = pollster::block_on(receiver.receive());
            if !matches!(mapped, Some(Ok(()))) {
                return Err(GpuError::MapReadFailed);
            }

            let data = slice.get_mapped_range();
            let scores = bytemuck::cast_slice::<u8, f32>(&data);
            for (local_row, &score) in scores.iter().enumerate().take(chunk.row_count) {
                accumulator.push(chunk.row_start + local_row, score);
            }
            drop(data);
            chunk.readback_buffer.unmap();
        }

        Ok(accumulator.into_sorted_vec())
    }

    /// Execute an exact GPU search for a batch of queries.
    fn search_exact_batch(
        &self,
        queries: &[f32],
        query_rows: usize,
        k: usize,
    ) -> Result<Vec<Vec<Neighbor>>, GpuError> {
        let padded_queries = if self.dims == self.dims_vec4 {
            Cow::Borrowed(queries)
        } else {
            Cow::Owned(pad_to_vec4(queries, query_rows, self.dims, self.dims_vec4))
        };
        self.dispatch_batched(padded_queries.as_ref(), query_rows, k)
    }

    /// Batched GPU dispatch with automatic query sub-batching for buffer size limits.
    ///
    /// Splits queries into sub-batches when the output buffer would exceed
    /// the adapter's `max_storage_buffer_binding_size`.
    fn dispatch_batched(
        &self,
        padded_queries: &[f32],
        query_count: usize,
        k: usize,
    ) -> Result<Vec<Vec<Neighbor>>, GpuError> {
        if query_count <= self.max_queries_per_sub_batch {
            return self.dispatch_sub_batch(padded_queries, query_count, k);
        }

        let mut all_results: Vec<Vec<Neighbor>> = Vec::with_capacity(query_count);
        let mut offset = 0;
        while offset < query_count {
            let batch_size = (query_count - offset).min(self.max_queries_per_sub_batch);
            let start = offset * self.dims_vec4;
            let end = start + batch_size * self.dims_vec4;
            let sub_results =
                self.dispatch_sub_batch(&padded_queries[start..end], batch_size, k)?;
            all_results.extend(sub_results);
            offset += batch_size;
        }
        Ok(all_results)
    }

    /// Dispatch a single sub-batch of queries across all matrix chunks.
    fn dispatch_sub_batch(
        &self,
        padded_queries: &[f32],
        query_count: usize,
        k: usize,
    ) -> Result<Vec<Vec<Neighbor>>, GpuError> {
        let query_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nndex-queries"),
                contents: bytemuck::cast_slice(padded_queries),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let pipeline = self.select_pipeline();
        let dims_v4 = (self.dims_vec4 / 4) as u32;

        let chunk_buffers: Vec<(wgpu::Buffer, wgpu::Buffer, u64)> = self
            .chunks
            .iter()
            .map(|chunk| {
                let output_size =
                    (chunk.row_count * query_count * std::mem::size_of::<f32>()) as u64;
                let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("nndex-batch-output"),
                    size: output_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("nndex-batch-readback"),
                    size: output_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                (output_buffer, readback_buffer, output_size)
            })
            .collect();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nndex-batch-encoder"),
            });

        for (chunk_idx, chunk) in self.chunks.iter().enumerate() {
            let (output_buffer, _, output_size) = &chunk_buffers[chunk_idx];

            let params = ShaderParams {
                rows: chunk.row_count as u32,
                dims_vec4: dims_v4,
                query_count: query_count as u32,
                _pad: 0,
            };

            let params_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("nndex-params-chunk"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nndex-bind-group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: chunk.matrix_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: query_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("nndex-compute-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let (wg_x, wg_y) = self.dispatch_dims(chunk.row_count as u32, query_count as u32);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            let (_, readback_buffer, _) = &chunk_buffers[chunk_idx];
            encoder.copy_buffer_to_buffer(output_buffer, 0, readback_buffer, 0, *output_size);
        }

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let mut accumulators: Vec<TopKAccumulator> =
            (0..query_count).map(|_| TopKAccumulator::new(k)).collect();

        for (chunk_idx, chunk) in self.chunks.iter().enumerate() {
            let (_, readback_buffer, _) = &chunk_buffers[chunk_idx];
            let slice = readback_buffer.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

            let mapped = pollster::block_on(receiver.receive());
            if !matches!(mapped, Some(Ok(()))) {
                return Err(GpuError::MapReadFailed);
            }

            let data = slice.get_mapped_range();
            let scores = bytemuck::cast_slice::<u8, f32>(&data);

            for (query_idx, acc) in accumulators.iter_mut().enumerate() {
                let base = query_idx * chunk.row_count;
                for row in 0..chunk.row_count {
                    acc.push(chunk.row_start + row, scores[base + row]);
                }
            }
            drop(data);
            readback_buffer.unmap();
        }

        Ok(accumulators
            .into_iter()
            .map(TopKAccumulator::into_sorted_vec)
            .collect())
    }

    /// Hash query data, k, and search mode for cache keying.
    fn query_hash(&self, values: &[f32], k: usize, query_rows: usize, approx: bool) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.rows.hash(&mut hasher);
        self.dims.hash(&mut hasher);
        k.hash(&mut hasher);
        query_rows.hash(&mut hasher);
        approx.hash(&mut hasher);
        bytemuck::cast_slice::<f32, u8>(values).hash(&mut hasher);
        hasher.finish()
    }
}

/// Pre-allocate all GPU buffers and bind groups needed for reusable single-query dispatch.
fn build_single_query_dispatch(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    chunks: &[MatrixChunk],
    dims_vec4: usize,
) -> SingleQueryDispatch {
    let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nndex-single-query"),
        size: (dims_vec4 * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dims_v4 = (dims_vec4 / 4) as u32;

    let chunk_dispatches = chunks
        .iter()
        .map(|chunk| {
            let output_size = (chunk.row_count * std::mem::size_of::<f32>()) as u64;
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nndex-single-output"),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nndex-single-readback"),
                size: output_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let params = ShaderParams {
                rows: chunk.row_count as u32,
                dims_vec4: dims_v4,
                query_count: 1,
                _pad: 0,
            };
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nndex-single-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nndex-single-bind-group"),
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: chunk.matrix_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: query_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            SingleQueryChunkDispatch {
                row_start: chunk.row_start,
                row_count: chunk.row_count,
                output_size,
                output_buffer,
                readback_buffer,
                bind_group,
            }
        })
        .collect();

    SingleQueryDispatch {
        query_buffer,
        chunks: chunk_dispatches,
    }
}

/// CPU dot product with optional SIMD acceleration via simsimd.
///
/// For small candidate sets, CPU dot products avoid GPU dispatch overhead entirely.
/// When the `cpu` feature is enabled, leverages simsimd for hardware-accelerated SIMD.
#[inline]
fn cpu_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "cpu")]
    {
        use simsimd::SpatialSimilarity;
        if a.len() >= 8
            && let Some(result) = f32::dot(a, b)
        {
            return result as f32;
        }
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Pad a row-major matrix so each row has `padded_dims` columns (for vec4 alignment).
fn pad_to_vec4(matrix: &[f32], rows: usize, dims: usize, padded_dims: usize) -> Vec<f32> {
    if dims == padded_dims {
        return matrix.to_vec();
    }
    let mut out = vec![0.0f32; rows * padded_dims];
    for (row_idx, src_row) in matrix.chunks_exact(dims).enumerate().take(rows) {
        let dst_start = row_idx * padded_dims;
        out[dst_start..dst_start + dims].copy_from_slice(src_row);
    }
    out
}

/// Pad a single query vector to `padded_dims` for vec4 alignment.
fn pad_query_vec4(query: &[f32], dims: usize, padded_dims: usize) -> Vec<f32> {
    if dims == padded_dims {
        return query.to_vec();
    }
    let mut out = vec![0.0f32; padded_dims];
    out[..dims].copy_from_slice(&query[..dims]);
    out
}
