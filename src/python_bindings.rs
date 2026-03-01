use numpy::{
    PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::types::{PyAny, PyDict, PyList};

#[cfg(feature = "gpu")]
use crate::gpu;
use crate::python_io::load_matrix_from_disk;
use crate::{ActiveBackend, BackendPreference, IndexOptions, NNdex, Neighbor};

/// Discriminant for the two supported DataFrame libraries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DataFrameKind {
    Pandas,
    Polars,
}

/// Intermediate representation of parsed Python input data as a flat f32 matrix.
#[derive(Debug)]
struct ParsedInput {
    values: Vec<f32>,
    rows: usize,
    dims: usize,
}

/// Python-facing wrapper around [`NNdex`], exposed as `nndex.NNdex`.
#[pyclass(name = "NNdex", unsendable)]
pub struct PyNNdex {
    index: NNdex,
    rows: usize,
    dims: usize,
}

#[pymethods]
impl PyNNdex {
    #[new]
    #[pyo3(signature = (data, normalized=false, approx=false, backend="cpu", enable_cache=true, gpu_device_index=None))]
    fn new(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        normalized: bool,
        approx: bool,
        backend: &str,
        enable_cache: bool,
        gpu_device_index: Option<usize>,
    ) -> PyResult<Self> {
        let index_options =
            parse_index_options(normalized, approx, backend, enable_cache, gpu_device_index)?;

        if let Ok(array) = data.extract::<PyReadonlyArray2<'_, f32>>() {
            let shape = array.shape();
            let (rows, dims) = (shape[0], shape[1]);
            if let Ok(slice) = array.as_slice() {
                return build_index(slice, rows, dims, index_options);
            }
            let values: Vec<f32> = array.as_array().iter().copied().collect();
            return build_index(&values, rows, dims, index_options);
        }

        let parsed = parse_input(py, data)?;
        build_index(&parsed.values, parsed.rows, parsed.dims, index_options)
    }

    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, *, key=None, normalized=false, approx=false, backend="cpu", enable_cache=true, gpu_device_index=None))]
    fn from_file(
        _cls: &Bound<'_, PyType>,
        path: &str,
        key: Option<&str>,
        normalized: bool,
        approx: bool,
        backend: &str,
        enable_cache: bool,
        gpu_device_index: Option<usize>,
    ) -> PyResult<Self> {
        let index_options =
            parse_index_options(normalized, approx, backend, enable_cache, gpu_device_index)?;
        let matrix = load_matrix_from_disk(path, key)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        build_index(&matrix.values, matrix.rows, matrix.dims, index_options)
    }

    #[getter]
    fn backend(&self) -> String {
        match self.index.backend() {
            ActiveBackend::Cpu => "cpu".to_string(),
            ActiveBackend::Gpu => "gpu".to_string(),
        }
    }

    #[getter]
    fn rows(&self) -> usize {
        self.rows
    }

    #[getter]
    fn dims(&self) -> usize {
        self.dims
    }

    /// Returns a dict with GPU adapter info, or `None` when the CPU backend is active.
    #[getter]
    fn gpu_info(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        #[cfg(feature = "gpu")]
        {
            if let Some(info) = self.index.gpu_info() {
                let dict = PyDict::new(py);
                dict.set_item("index", info.index)?;
                dict.set_item("name", info.name)?;
                dict.set_item("driver", info.driver)?;
                dict.set_item("driver_info", info.driver_info)?;
                dict.set_item("backend", info.backend)?;
                dict.set_item("device_type", info.device_type)?;
                return Ok(Some(dict.unbind()));
            }
        }
        Ok(None)
    }

    #[pyo3(signature = (query, k=10, dataframe=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
        dataframe: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let output_dataframe = resolve_output_dataframe(py, dataframe, self.rows)?;
        let query_is_dataframe = detect_dataframe_kind(query).is_some();

        if !query_is_dataframe && let Ok(array) = query.extract::<PyReadonlyArray1<'_, f32>>() {
            let dims = array.shape()[0];
            if dims != self.dims {
                return Err(PyValueError::new_err(format!(
                    "query dims mismatch: expected {}, got {dims}",
                    self.dims
                )));
            }
            if let Ok(slice) = array.as_slice() {
                let neighbors = self
                    .index
                    .search(slice, k)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                if let Some((kind, source_dataframe)) = &output_dataframe {
                    return neighbors_to_source_dataframe(
                        py,
                        &[neighbors],
                        *kind,
                        source_dataframe.bind(py),
                    );
                }
                return neighbors_to_numpy(py, &neighbors);
            }
        }

        if !query_is_dataframe && let Ok(array) = query.extract::<PyReadonlyArray2<'_, f32>>() {
            let shape = array.shape();
            let (rows, dims) = (shape[0], shape[1]);
            if dims != self.dims {
                return Err(PyValueError::new_err(format!(
                    "query dims mismatch: expected {}, got {dims}",
                    self.dims
                )));
            }
            if rows == 1
                && let Ok(slice) = array.as_slice()
            {
                let neighbors = self
                    .index
                    .search(slice, k)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                if let Some((kind, source_dataframe)) = &output_dataframe {
                    return neighbors_to_source_dataframe(
                        py,
                        &[neighbors],
                        *kind,
                        source_dataframe.bind(py),
                    );
                }
                return neighbors_to_numpy(py, &neighbors);
            }
            let values: Vec<f32> = if let Ok(slice) = array.as_slice() {
                slice.to_vec()
            } else {
                array.as_array().iter().copied().collect()
            };
            let batch = self
                .index
                .search_batch(&values, rows, k)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            if let Some((kind, source_dataframe)) = &output_dataframe {
                return neighbors_to_source_dataframe(py, &batch, *kind, source_dataframe.bind(py));
            }
            return batch_neighbors_to_numpy(py, &batch);
        }

        let parsed = parse_input(py, query)?;
        if parsed.dims != self.dims {
            return Err(PyValueError::new_err(format!(
                "query dims mismatch: expected {}, got {}",
                self.dims, parsed.dims
            )));
        }

        if parsed.rows == 1 {
            let neighbors = self
                .index
                .search(&parsed.values, k)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            if let Some((kind, source_dataframe)) = &output_dataframe {
                return neighbors_to_source_dataframe(
                    py,
                    &[neighbors],
                    *kind,
                    source_dataframe.bind(py),
                );
            }
            return neighbors_to_numpy(py, &neighbors);
        }

        let batch = self
            .index
            .search_batch(&parsed.values, parsed.rows, k)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

        if let Some((kind, source_dataframe)) = &output_dataframe {
            return neighbors_to_source_dataframe(py, &batch, *kind, source_dataframe.bind(py));
        }

        batch_neighbors_to_numpy(py, &batch)
    }
}

/// Construct a [`PyNNdex`] from a flat f32 slice, mapping Rust errors to Python exceptions.
fn build_index(
    values: &[f32],
    rows: usize,
    dims: usize,
    index_options: IndexOptions,
) -> PyResult<PyNNdex> {
    let index = NNdex::new(values, rows, dims, index_options)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyNNdex { index, rows, dims })
}

/// Convert Python keyword arguments into [`IndexOptions`].
fn parse_index_options(
    normalized: bool,
    approx: bool,
    backend: &str,
    enable_cache: bool,
    gpu_device_index: Option<usize>,
) -> PyResult<IndexOptions> {
    let preference = parse_backend(backend)?;
    Ok(IndexOptions {
        normalized,
        approx,
        backend: preference,
        enable_cache,
        gpu_device_index,
    })
}

/// List all available GPU adapters as a list of dicts.
///
/// Each dict contains: `index`, `name`, `driver`, `driver_info`, `backend`, `device_type`.
/// Returns an empty list when no GPU is available.
#[cfg(feature = "gpu")]
#[pyfunction]
fn list_gpu_devices(py: Python<'_>) -> PyResult<Py<PyList>> {
    let devices = gpu::list_gpu_devices();
    let list = PyList::empty(py);
    for device in devices {
        let dict = PyDict::new(py);
        dict.set_item("index", device.index)?;
        dict.set_item("name", device.name)?;
        dict.set_item("driver", device.driver)?;
        dict.set_item("driver_info", device.driver_info)?;
        dict.set_item("backend", device.backend)?;
        dict.set_item("device_type", device.device_type)?;
        list.append(dict)?;
    }
    Ok(list.unbind())
}

#[pymodule]
pub fn _nndex(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNNdex>()?;
    #[cfg(feature = "gpu")]
    module.add_function(wrap_pyfunction!(list_gpu_devices, module)?)?;
    Ok(())
}

/// Map a user-supplied backend string (`"cpu"`, `"gpu"`) to the enum.
fn parse_backend(raw: &str) -> PyResult<BackendPreference> {
    match raw.to_ascii_lowercase().as_str() {
        "cpu" => Ok(BackendPreference::Cpu),
        "gpu" => Ok(BackendPreference::Gpu),
        other => Err(PyValueError::new_err(format!(
            "unsupported backend '{other}', expected one of: cpu, gpu"
        ))),
    }
}

/// Convert an arbitrary Python object (numpy array, list, DataFrame, etc.) into a flat f32 matrix.
fn parse_input(_py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<ParsedInput> {
    if value.hasattr("to_numpy")? {
        let numpy_like = value.call_method0("to_numpy")?;
        if !numpy_like.is(value) {
            return parse_input(_py, &numpy_like);
        }
    }

    if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, f32>>() {
        let view = array.as_array();
        let shape = view.shape().to_vec();
        let values = view.iter().copied().collect::<Vec<_>>();
        return match shape.as_slice() {
            [dims] => Ok(ParsedInput {
                values,
                rows: 1,
                dims: *dims,
            }),
            [rows, dims] => Ok(ParsedInput {
                values,
                rows: *rows,
                dims: *dims,
            }),
            _ => Err(PyValueError::new_err(
                "only 1D and 2D arrays are supported for inputs",
            )),
        };
    }

    if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, f64>>() {
        let view = array.as_array();
        let shape = view.shape().to_vec();
        let values = view.iter().copied().map(|v| v as f32).collect::<Vec<_>>();
        return match shape.as_slice() {
            [dims] => Ok(ParsedInput {
                values,
                rows: 1,
                dims: *dims,
            }),
            [rows, dims] => Ok(ParsedInput {
                values,
                rows: *rows,
                dims: *dims,
            }),
            _ => Err(PyValueError::new_err(
                "only 1D and 2D arrays are supported for inputs",
            )),
        };
    }

    if let Ok(rows) = value.extract::<Vec<Vec<f32>>>() {
        if rows.is_empty() || rows[0].is_empty() {
            return Err(PyValueError::new_err("input matrix must be non-empty"));
        }
        let dims = rows[0].len();
        if rows.iter().any(|row| row.len() != dims) {
            return Err(PyValueError::new_err(
                "2D list input must have consistent row lengths",
            ));
        }

        let mut flat = Vec::with_capacity(rows.len() * dims);
        for row in rows {
            flat.extend(row);
        }
        let row_count = flat.len() / dims;
        return Ok(ParsedInput {
            values: flat,
            rows: row_count,
            dims,
        });
    }

    if let Ok(vector) = value.extract::<Vec<f32>>() {
        if vector.is_empty() {
            return Err(PyValueError::new_err("input vector must be non-empty"));
        }
        let dims = vector.len();
        return Ok(ParsedInput {
            values: vector,
            rows: 1,
            dims,
        });
    }

    Err(PyValueError::new_err(
        "unsupported input type: expected list, numpy array, \
         pandas/polars Series, or DataFrame",
    ))
}

/// Inspect the Python type's module to determine if the object is a pandas or polars DataFrame.
fn detect_dataframe_kind(value: &Bound<'_, PyAny>) -> Option<DataFrameKind> {
    let type_obj = value.get_type();
    let module = type_obj.module().ok()?.to_string();
    let name = type_obj.name().ok()?.to_string();

    if name == "DataFrame" && module.contains("pandas") {
        return Some(DataFrameKind::Pandas);
    }
    if name == "DataFrame" && module.contains("polars") {
        return Some(DataFrameKind::Polars);
    }
    None
}

/// Validate and extract the optional output DataFrame, checking row count matches the index.
fn resolve_output_dataframe(
    _py: Python<'_>,
    dataframe: Option<&Bound<'_, PyAny>>,
    expected_rows: usize,
) -> PyResult<Option<(DataFrameKind, Py<PyAny>)>> {
    let Some(dataframe) = dataframe else {
        return Ok(None);
    };
    let kind = detect_dataframe_kind(dataframe).ok_or_else(|| {
        PyValueError::new_err("`dataframe` must be a pandas.DataFrame or polars.DataFrame")
    })?;
    let row_count = dataframe_row_count(dataframe, kind)?;
    if row_count != expected_rows {
        return Err(PyValueError::new_err(format!(
            "provided dataframe row count must match index rows (index rows={expected_rows}, \
             dataframe rows={row_count})"
        )));
    }
    Ok(Some((kind, dataframe.clone().unbind())))
}

/// Extract the row count from a pandas (`.shape[0]`) or polars (`.height`) DataFrame.
fn dataframe_row_count(value: &Bound<'_, PyAny>, kind: DataFrameKind) -> PyResult<usize> {
    match kind {
        DataFrameKind::Pandas => value.getattr("shape")?.get_item(0)?.extract::<usize>(),
        DataFrameKind::Polars => value.getattr("height")?.extract::<usize>(),
    }
}

/// Build a `(indices, similarities)` tuple of 1D numpy arrays directly from Rust.
fn neighbors_to_numpy(py: Python<'_>, neighbors: &[Neighbor]) -> PyResult<Py<PyAny>> {
    let indices: Vec<u64> = neighbors.iter().map(|n| n.index as u64).collect();
    let similarities: Vec<f32> = neighbors.iter().map(|n| n.similarity).collect();

    let index_array = PyArray1::from_vec(py, indices);
    let similarity_array = PyArray1::from_vec(py, similarities);
    Ok((index_array, similarity_array)
        .into_pyobject(py)?
        .unbind()
        .into_any())
}

/// Build a `(indices, similarities)` tuple of 2D numpy arrays directly from Rust.
///
/// UPDATE: batch_neighbors_to_numpy maps &[Vec<Neighbor>] into Vec<Vec<u64>> and Vec<Vec<f32>>.
/// For a batch of 16,384 queries, this allocates 32,768 separate heap vectors to satisfy the bridge interface.
/// To improve, we allocate a single flat vector and instantly reshape it into a 2D matrix directly via NumPy.
fn batch_neighbors_to_numpy(py: Python<'_>, batch: &[Vec<Neighbor>]) -> PyResult<Py<PyAny>> {
    let rows = batch.len();
    let cols = batch.first().map_or(0, |row| row.len());

    let mut flat_indices = Vec::with_capacity(rows * cols);
    let mut flat_similarities = Vec::with_capacity(rows * cols);

    for row in batch {
        for n in row.iter().take(cols) {
            flat_indices.push(n.index as u64);
            flat_similarities.push(n.similarity);
        }
        for _ in row.len()..cols {
            // Safety padding
            flat_indices.push(0);
            flat_similarities.push(0.0);
        }
    }

    let index_array = PyArray1::from_vec(py, flat_indices)
        .reshape((rows, cols))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let similarity_array = PyArray1::from_vec(py, flat_similarities)
        .reshape((rows, cols))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((index_array, similarity_array)
        .into_pyobject(py)?
        .unbind()
        .into_any())
}

/// Slice the source DataFrame by neighbor indices and attach a `similarity` column.
/// Returns a single DataFrame for one query, or a list of DataFrames for a batch.
fn neighbors_to_source_dataframe(
    py: Python<'_>,
    batch: &[Vec<Neighbor>],
    query_kind: DataFrameKind,
    source_dataframe: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if batch.len() == 1 {
        return neighbors_to_single_dataframe(py, &batch[0], query_kind, source_dataframe);
    }

    let frames = batch
        .iter()
        .map(|neighbors| neighbors_to_single_dataframe(py, neighbors, query_kind, source_dataframe))
        .collect::<PyResult<Vec<_>>>()?;
    let list = PyList::empty(py);
    for frame in frames {
        list.append(frame)?;
    }
    Ok(list.unbind().into_any())
}

/// Build a single output DataFrame from one set of neighbors and the source DataFrame.
fn neighbors_to_single_dataframe(
    py: Python<'_>,
    neighbors: &[Neighbor],
    kind: DataFrameKind,
    source_dataframe: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let row_indices: Vec<usize> = neighbors.iter().map(|neighbor| neighbor.index).collect();
    let similarities: Vec<f32> = neighbors
        .iter()
        .map(|neighbor| neighbor.similarity)
        .collect();

    let dataframe = match kind {
        DataFrameKind::Pandas => {
            let matched = source_dataframe.call_method1("take", (row_indices,))?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("similarity", similarities)?;
            matched.call_method("assign", (), Some(&kwargs))?
        }
        DataFrameKind::Polars => {
            let matched = source_dataframe.get_item(row_indices)?;
            let series = py
                .import("polars")?
                .getattr("Series")?
                .call1(("similarity", similarities))?;
            matched.call_method1("with_columns", (series,))?
        }
    };

    Ok(dataframe.unbind().into_any())
}
