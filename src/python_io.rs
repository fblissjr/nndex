use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use arrow_array::{Array, FixedSizeListArray, Float32Array, ListArray};
use ndarray::Array2;
use ndarray_npy::{NpzReader, ReadNpyError, ReadNpzError};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::ParquetError;
use thiserror::Error;

/// A row-major f32 matrix loaded from disk, ready to be passed to [`NNdex::new`].
#[derive(Debug)]
pub(crate) struct DiskMatrix {
    pub(crate) values: Vec<f32>,
    pub(crate) rows: usize,
    pub(crate) dims: usize,
}

/// Errors that can occur when loading a matrix from a file on disk.
#[derive(Debug, Error)]
pub(crate) enum DiskLoadError {
    #[error("unsupported file extension for '{path}', expected .npy, .npz, or .parquet")]
    UnsupportedExtension { path: String },
    #[error("missing required key for {format} file '{path}'")]
    MissingKey { path: String, format: &'static str },
    #[error("could not open '{path}': {source}")]
    OpenFile {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read .npy matrix from '{path}': {source}")]
    ReadNpy {
        path: String,
        #[source]
        source: ReadNpyError,
    },
    #[error("failed to read .npz archive from '{path}': {source}")]
    OpenNpz {
        path: String,
        #[source]
        source: ReadNpzError,
    },
    #[error("failed reading key '{key}' from .npz '{path}': {source}")]
    ReadNpzKey {
        path: String,
        key: String,
        #[source]
        source: ReadNpzError,
    },
    #[error("key '{key}' not found in .npz '{path}'. Available keys: {available_keys}")]
    NpzKeyNotFound {
        path: String,
        key: String,
        available_keys: String,
    },
    #[error("failed to read parquet file '{path}': {source}")]
    ReadParquet {
        path: String,
        #[source]
        source: ParquetError,
    },
    #[error("column '{key}' was not found in parquet file '{path}'")]
    ParquetColumnNotFound { path: String, key: String },
    #[error("parquet key '{key}' in '{path}' is not an array/list<f32> column")]
    UnsupportedParquetColumnType { path: String, key: String },
    #[error("parquet key '{key}' in '{path}' contains null values, which are unsupported")]
    NullValuesInParquetColumn { path: String, key: String },
    #[error("parquet key '{key}' in '{path}' has inconsistent row dimensions")]
    InconsistentParquetRowDims { path: String, key: String },
    #[error("matrix loaded from '{path}' is empty or has zero dimensions")]
    EmptyMatrix { path: String },
}

/// Load a 2D f32 matrix from a `.npy`, `.npz`, or `.parquet` file.
///
/// For `.npz` and `.parquet` formats, a `key` must be provided to select the
/// array or column within the file.
pub(crate) fn load_matrix_from_disk(
    path: &str,
    key: Option<&str>,
) -> Result<DiskMatrix, DiskLoadError> {
    let source_path = Path::new(path);
    match source_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("npy") => load_npy(source_path),
        Some("npz") => {
            let matrix_key = key.ok_or_else(|| DiskLoadError::MissingKey {
                path: path.to_string(),
                format: "npz",
            })?;
            load_npz(source_path, matrix_key)
        }
        Some("parquet") => {
            let matrix_key = key.ok_or_else(|| DiskLoadError::MissingKey {
                path: path.to_string(),
                format: "parquet",
            })?;
            load_parquet(source_path, matrix_key)
        }
        _ => Err(DiskLoadError::UnsupportedExtension {
            path: path.to_string(),
        }),
    }
}

/// Read a single 2D f32 array from a `.npy` file.
fn load_npy(path: &Path) -> Result<DiskMatrix, DiskLoadError> {
    let array =
        ndarray_npy::read_npy::<_, Array2<f32>>(path).map_err(|source| DiskLoadError::ReadNpy {
            path: path.display().to_string(),
            source,
        })?;
    matrix_from_ndarray(path, array)
}

/// Read a named 2D f32 array from a `.npz` archive.
fn load_npz(path: &Path, key: &str) -> Result<DiskMatrix, DiskLoadError> {
    let file = File::open(path).map_err(|source| DiskLoadError::OpenFile {
        path: path.display().to_string(),
        source,
    })?;
    let mut reader =
        NpzReader::new(BufReader::new(file)).map_err(|source| DiskLoadError::OpenNpz {
            path: path.display().to_string(),
            source,
        })?;

    let available = reader.names().map_err(|source| DiskLoadError::OpenNpz {
        path: path.display().to_string(),
        source,
    })?;
    let candidate_with_suffix = if key.ends_with(".npy") {
        key.to_string()
    } else {
        format!("{key}.npy")
    };
    let selected_key = if available.iter().any(|name| name == &candidate_with_suffix) {
        candidate_with_suffix
    } else if available.iter().any(|name| name == key) {
        key.to_string()
    } else {
        return Err(DiskLoadError::NpzKeyNotFound {
            path: path.display().to_string(),
            key: key.to_string(),
            available_keys: available.join(", "),
        });
    };

    let array = reader
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>(&selected_key)
        .map_err(|source| DiskLoadError::ReadNpzKey {
            path: path.display().to_string(),
            key: key.to_string(),
            source,
        })?;

    matrix_from_ndarray(path, array)
}

/// Read a list/fixed-size-list column of f32 values from a Parquet file as a 2D matrix.
fn load_parquet(path: &Path, key: &str) -> Result<DiskMatrix, DiskLoadError> {
    let file = File::open(path).map_err(|source| DiskLoadError::OpenFile {
        path: path.display().to_string(),
        source,
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|source| {
        DiskLoadError::ReadParquet {
            path: path.display().to_string(),
            source,
        }
    })?;
    let column_index =
        builder
            .schema()
            .index_of(key)
            .map_err(|_| DiskLoadError::ParquetColumnNotFound {
                path: path.display().to_string(),
                key: key.to_string(),
            })?;
    let reader = builder
        .build()
        .map_err(|source| DiskLoadError::ReadParquet {
            path: path.display().to_string(),
            source,
        })?;

    let mut values = Vec::<f32>::new();
    let mut rows = 0usize;
    let mut dims = None::<usize>;
    for batch_result in reader {
        let batch = batch_result.map_err(|source| DiskLoadError::ReadParquet {
            path: path.display().to_string(),
            source: source.into(),
        })?;
        let column = batch.column(column_index);

        if let Some(fixed_size_list) = column.as_any().downcast_ref::<FixedSizeListArray>() {
            append_fixed_size_parquet_column(
                fixed_size_list,
                &mut values,
                &mut rows,
                &mut dims,
                path,
                key,
            )?;
            continue;
        }
        if let Some(list_array) = column.as_any().downcast_ref::<ListArray>() {
            append_list_parquet_column(list_array, &mut values, &mut rows, &mut dims, path, key)?;
            continue;
        }
        return Err(DiskLoadError::UnsupportedParquetColumnType {
            path: path.display().to_string(),
            key: key.to_string(),
        });
    }

    let final_dims = dims.unwrap_or(0);
    if rows == 0 || final_dims == 0 {
        return Err(DiskLoadError::EmptyMatrix {
            path: path.display().to_string(),
        });
    }
    Ok(DiskMatrix {
        values,
        rows,
        dims: final_dims,
    })
}

/// Convert an ndarray `Array2<f32>` into a [`DiskMatrix`], ensuring standard row-major layout.
fn matrix_from_ndarray(path: &Path, array: Array2<f32>) -> Result<DiskMatrix, DiskLoadError> {
    let (rows, dims) = array.dim();
    if rows == 0 || dims == 0 {
        return Err(DiskLoadError::EmptyMatrix {
            path: path.display().to_string(),
        });
    }

    let standard = array.as_standard_layout().to_owned();
    let (raw_values, offset) = standard.into_raw_vec_and_offset();
    let value_count = rows.saturating_mul(dims);
    let start = offset.unwrap_or(0);
    let end = start.saturating_add(value_count);
    let values = if start == 0 && raw_values.len() == value_count {
        raw_values
    } else {
        raw_values
            .get(start..end)
            .ok_or_else(|| DiskLoadError::EmptyMatrix {
                path: path.display().to_string(),
            })?
            .to_vec()
    };

    Ok(DiskMatrix { values, rows, dims })
}

/// Append rows from a `FixedSizeListArray` column into the accumulating flat matrix.
fn append_fixed_size_parquet_column(
    column: &FixedSizeListArray,
    values: &mut Vec<f32>,
    rows: &mut usize,
    dims: &mut Option<usize>,
    path: &Path,
    key: &str,
) -> Result<(), DiskLoadError> {
    if column.null_count() > 0 {
        return Err(DiskLoadError::NullValuesInParquetColumn {
            path: path.display().to_string(),
            key: key.to_string(),
        });
    }
    let row_dims = usize::try_from(column.value_length()).map_err(|_| {
        DiskLoadError::UnsupportedParquetColumnType {
            path: path.display().to_string(),
            key: key.to_string(),
        }
    })?;
    if row_dims == 0 {
        return Err(DiskLoadError::EmptyMatrix {
            path: path.display().to_string(),
        });
    }

    let element_values = column
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| DiskLoadError::UnsupportedParquetColumnType {
            path: path.display().to_string(),
            key: key.to_string(),
        })?;
    if element_values.null_count() > 0 {
        return Err(DiskLoadError::NullValuesInParquetColumn {
            path: path.display().to_string(),
            key: key.to_string(),
        });
    }

    if let Some(existing_dims) = dims {
        if *existing_dims != row_dims {
            return Err(DiskLoadError::InconsistentParquetRowDims {
                path: path.display().to_string(),
                key: key.to_string(),
            });
        }
    } else {
        *dims = Some(row_dims);
    }

    values.extend_from_slice(element_values.values());
    *rows += column.len();
    Ok(())
}

/// Append rows from a variable-length `ListArray` column into the accumulating flat matrix.
fn append_list_parquet_column(
    column: &ListArray,
    values: &mut Vec<f32>,
    rows: &mut usize,
    dims: &mut Option<usize>,
    path: &Path,
    key: &str,
) -> Result<(), DiskLoadError> {
    if column.null_count() > 0 {
        return Err(DiskLoadError::NullValuesInParquetColumn {
            path: path.display().to_string(),
            key: key.to_string(),
        });
    }

    for row_idx in 0..column.len() {
        let row_values = column.value(row_idx);
        let row_values = row_values
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| DiskLoadError::UnsupportedParquetColumnType {
                path: path.display().to_string(),
                key: key.to_string(),
            })?;
        if row_values.null_count() > 0 {
            return Err(DiskLoadError::NullValuesInParquetColumn {
                path: path.display().to_string(),
                key: key.to_string(),
            });
        }
        let row_dims = row_values.len();
        if row_dims == 0 {
            return Err(DiskLoadError::EmptyMatrix {
                path: path.display().to_string(),
            });
        }

        if let Some(existing_dims) = dims {
            if *existing_dims != row_dims {
                return Err(DiskLoadError::InconsistentParquetRowDims {
                    path: path.display().to_string(),
                    key: key.to_string(),
                });
            }
        } else {
            *dims = Some(row_dims);
        }

        values.extend_from_slice(row_values.values());
        *rows += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unsupported_extension() {
        let error = load_matrix_from_disk("matrix.csv", None).expect_err("csv should be rejected");
        assert!(matches!(error, DiskLoadError::UnsupportedExtension { .. }));
    }

    #[test]
    fn requires_key_for_npz() {
        let error = load_matrix_from_disk("matrix.npz", None).expect_err("npz should require key");
        assert!(matches!(
            error,
            DiskLoadError::MissingKey { format: "npz", .. }
        ));
    }

    #[test]
    fn requires_key_for_parquet() {
        let error =
            load_matrix_from_disk("matrix.parquet", None).expect_err("parquet should require key");
        assert!(matches!(
            error,
            DiskLoadError::MissingKey {
                format: "parquet",
                ..
            }
        ));
    }
}
