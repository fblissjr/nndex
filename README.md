# nndex

[PyPI](https://pypi.org/project/nndex/) | [Crates.io](https://crates.io/crates/nndex)

A high-performance Rust library with Python bindings for nearest-neighbor vector search with zero configuration necessary that's so fast, it's faster than [numpy](https://numpy.org)! This crate leverages the computational trick where if the source vectors and query vectors are all unit-normalized, performing a dot-product — an operation faster than vector distance calculations — returns the cosine similarity.

Features:

- CPU backend using [rayon](https://crates.io/crates/rayon) parallelism + SIMD (via [simsimd](https://crates.io/crates/simsimd)), along with highly bespoke compute profiles for maximum CPU performance
- GPU backend using [wgpu](https://crates.io/crates/wgpu) compute shaders, supporting Vulkan, Metal, D3D12, and OpenGL graphics APIs
- Approximate nearest-neighbor (ANN) mode with exact reranking by building an IVF index for even faster lookups
- Batch search for multiple queries at once
- Python bindings via PyO3 with [numpy](https://numpy.org), [pandas](https://pandas.pydata.org), and [polars](https://pola.rs) support
- Load embeddings directly from `.npy`, `.npz`, and `.parquet` files
- Internal query-result caching for repeated searches

_**Disclosure:** This library was mostly coded with the assistance of Claude Opus 4.6 and GPT-5.3-Codex as research into the discovery that those models can now successfuly hyperoptimize Rust code. However, I personally have reviewed all code to ensure it is accurate, have added numerous tests and benchmarks to ensure it works as both intended and advertised, and have edited documentation and comments to provide greater signal as to how the package operates. I have given this project the same care and attention as I would give a project I have written from scratch._

## Installation

### Python

```bash
pip install nndex
```

```bash
uv pip install nndex
```

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
nndex = "0.2.1"
```

## Python Usage

See also the [demo notebooks](https://github.com/minimaxir/nndex/tree/main/notebooks) for more interactive examples.

### Building an Index

```python
import numpy as np
from nndex import NNdex

rng = np.random.default_rng(42)
matrix = rng.normal(size=(50_000, 128)).astype(np.float32)

# Build an index (auto-selects GPU if available, falls back to CPU)
index = NNdex(matrix)
print(index.backend)  # "gpu" or "cpu"
print(index.rows, index.dims)  # 50000 128
```

### Single Query

`search()` returns a tuple of `(indices, similarities)` as numpy arrays, ordered from most similar to least similar.

```python
query = rng.normal(size=(dims,)).astype(np.float32)

indices, scores = index.search(query, k=5)
print(indices)  # [43576 14100 15993 35409 38916]
print(scores)   # [0.360 0.353 0.335 0.332 0.323]
```

### Batch Query

Pass a 2D array to search multiple queries at once more efficienctly than querying one at a time. Returns 2D numpy arrays.

```python
queries = rng.normal(size=(4, dims)).astype(np.float32)

batch_indices, batch_scores = index.search(queries, k=5)
print(batch_indices.shape)  # (4, 5)
print(batch_scores.shape)   # (4, 5)
```

### Approximate Nearest Neighbors

Enable `approx=True` for faster queries (3x minimum, greater benefit on larger matrices): uses a quick IVF index followed by exact reranking.

```python
index_ann = NNdex(matrix, approx=True)

indices, scores = index_ann.search(query, k=5)
```

### Backend Selection

```python
# CPU (default)
cpu_index = NNdex(matrix, backend="cpu")

# Force GPU
gpu_index = NNdex(matrix, backend="gpu")
```

### Pre-Normalized Data

If your embeddings are already unit-normalized, set `normalized=True` to skip the internal normalization step for more computational speed:

```python
normalized_matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
index = NNdex(normalized_matrix, normalized=True)
```

### DataFrame Output

A conveience function which allows you to pass a pandas or polars DataFrame to `dataframe=` to get results as DataFrames with a `similarity` column appended. The DataFrame must have the same number of rows as the index.

```python
import pandas as pd

df = pd.DataFrame(matrix, columns=[f"d{i}" for i in range(dims)])
index = NNdex(matrix, backend="cpu")

# Single query: returns a DataFrame
result = index.search(query, k=3, dataframe=df)

# Batch query: returns a list of DataFrames
results = index.search(queries, k=3, dataframe=df)
```

This also works with polars DataFrames:

```python
import polars as pl

pldf = pl.DataFrame({f"d{i}": matrix[:, i] for i in range(dims)})
result = index.search(query, k=3, dataframe=pldf)
```

### Loading from Disk

`NNdex.from_file()` loads embeddings directly from `.npy`, `.npz`, or `.parquet` files in Rust, avoiding Python-side deserialization overhead. `.npz` and `.parquet` require a `key` argument.

```python
# .npy (single 2D array)
index = NNdex.from_file("embeddings.npy")

# .npz (keyed archive)
index = NNdex.from_file("embeddings.npz", key="matrix_store")

# .parquet (list/fixed-size-list column of f32)
index = NNdex.from_file("embeddings.parquet", key="embedding")
```

## Rust Usage

```rust
use nndex::{NNdex, IndexOptions, BackendPreference, Neighbor};

fn main() -> Result<(), nndex::NNdexError> {
    let matrix = vec![
        1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    let index = NNdex::new(&matrix, 3, 8, IndexOptions {
        normalized: false,
        approx: false,
        backend: BackendPreference::Cpu,
        ..IndexOptions::default()
    })?;

    let query = vec![0.8_f32, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let neighbors = index.search(&query, 2)?;

    for neighbor in &neighbors {
        println!("index: {}, similarity: {:.4}", neighbor.index, neighbor.similarity);
    }

    Ok(())
}
```

## Benchmarks

[BENCHMARK IMAGES TO BE ADDED]

Rough Python benchmarks are available in [ANN vs. Exact](notebooks/benchmark_cpu_ann_vs_exact.ipynb) and [vs. numpy](notebooks/benchmark_cpu_vs_numpy.ipynb) Jupyter Notebooks.

## Notes

- nndex is **NOT** a vector store/database which implies that the vectors can be created/updated/deleted from the matrix, and it is not intending to be. It's intended to be used with a fixed matrix of data, although this crate is so fast that you could reinitialize the `NNdex` without much overhead if needed.
- For Apple Silicon in particular, the use of the GPU backend (Metal) is not recommended below 100k rows due to the dispatch overhead of `wgpu` being greater than the inference speed. This is not the case with discrete GPUs.
- BLAS is only supported for macOS because the underlying BLAS library ([accelerate](https://developer.apple.com/accelerate/)) is included by default. There are tradeoffs for Linux/Windows and I am still determining what to do there.

## API Reference

### Python

#### `NNdex(data, ...)`

| Parameter      | Type       | Default  | Description                                              |
| -------------- | ---------- | -------- | -------------------------------------------------------- |
| `data`         | array-like | required | 2D numpy array, list, or pandas/polars DataFrame         |
| `normalized`   | bool       | `False`  | Skip internal normalization if data is already unit-norm |
| `approx`       | bool       | `False`  | Enable ANN prefiltering with exact reranking             |
| `backend`      | str        | `"cpu"`  | `"cpu"` or `"gpu"`                                       |
| `enable_cache` | bool       | `True`   | Cache repeated query results                             |

#### `NNdex.from_file(path, ...)`

| Parameter      | Type     | Default  | Description                                              |
| -------------- | -------- | -------- | -------------------------------------------------------- |
| `path`         | str      | required | Path to `.npy`, `.npz`, or `.parquet` file               |
| `key`          | str/None | `None`   | Array key (`.npz`) or column name (`.parquet`)           |
| `normalized`   | bool     | `False`  | Skip internal normalization if data is already unit-norm |
| `approx`       | bool     | `False`  | Enable ANN prefiltering with exact reranking             |
| `backend`      | str      | `"cpu"`  | `"cpu"` or `"gpu"`                                       |
| `enable_cache` | bool     | `True`   | Cache repeated query results                             |

#### `index.search(query, k=10, dataframe=None)`

| Parameter   | Type           | Default  | Description                                             |
| ----------- | -------------- | -------- | ------------------------------------------------------- |
| `query`     | array-like     | required | 1D vector or 2D matrix of queries                       |
| `k`         | int            | `10`     | Number of neighbors to return per query                 |
| `dataframe` | DataFrame/None | `None`   | Source DataFrame for output; must match index row count |

**Returns:** Without `dataframe`: tuple of `(indices, similarities)` as numpy arrays (1D for single query, 2D for batch). With `dataframe`: a DataFrame (single query) or list of DataFrames (batch) with a `similarity` column.

#### Properties

| Property  | Type | Description                      |
| --------- | ---- | -------------------------------- |
| `backend` | str  | Active backend (`"cpu"`/`"gpu"`) |
| `rows`    | int  | Number of indexed rows           |
| `dims`    | int  | Number of dimensions per row     |

### Rust

#### `NNdex::new(matrix, rows, dims, options)`

Constructs an index from a flattened row-major `&[f32]` matrix.

#### `index.search(query, k) -> Result<Vec<Neighbor>>`

Returns top-k neighbors sorted by descending cosine similarity.

#### `index.search_batch(queries, query_rows, k) -> Result<Vec<Vec<Neighbor>>>`

Batch search over multiple query vectors.

#### `IndexOptions`

| Field          | Type                | Default | Description                                |
| -------------- | ------------------- | ------- | ------------------------------------------ |
| `normalized`   | `bool`              | `false` | Skip normalization for pre-normalized data |
| `approx`       | `bool`              | `false` | Enable ANN prefiltering                    |
| `backend`      | `BackendPreference` | `Cpu`   | `Cpu`, or `Gpu`                            |
| `enable_cache` | `bool`              | `true`  | Cache constructor and query results        |

## Maintainer/Creator

Max Woolf ([@minimaxir](https://minimaxir.com))

_Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir) and [GitHub Sponsors](https://github.com/sponsors/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use._

## License

MIT
