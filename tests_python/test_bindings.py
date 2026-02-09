from pathlib import Path

import numpy as np
import pytest

from nndex import NNdex


def test_list_input_and_search_output_shapes() -> None:
    matrix = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    model = NNdex(matrix, normalized=False, backend="cpu")
    indices, scores = model.search([0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], k=2)

    assert indices.shape == (2,)
    assert scores.shape == (2,)
    assert int(indices[0]) == 2
    assert float(scores[0]) > float(scores[1])


def test_numpy_batch_query() -> None:
    matrix = np.eye(8, dtype=np.float32)
    model = NNdex(matrix, normalized=True, backend="cpu")

    queries = np.vstack([np.eye(8, dtype=np.float32)[0], np.eye(8, dtype=np.float32)[1]])
    indices, scores = model.search(queries, k=2)

    assert indices.shape == (2, 2)
    assert scores.shape == (2, 2)
    assert int(indices[0, 0]) == 0
    assert int(indices[1, 0]) == 1


def test_backend_property_available() -> None:
    matrix = np.eye(8, dtype=np.float32)
    model = NNdex(matrix, normalized=True, backend="auto")
    assert model.backend in {"cpu", "gpu"}


def test_approx_constructor_flag_is_supported() -> None:
    matrix = np.eye(8, dtype=np.float32)
    model = NNdex(matrix, normalized=True, approx=True, backend="cpu")
    indices, _scores = model.search(matrix[0], k=1)
    assert int(indices[0]) == 0


def test_pandas_dataframe_returns_dataframe() -> None:
    pd = pytest.importorskip("pandas")

    matrix = pd.DataFrame(
        {
            "a": [1.0, 0.0, 1.0],
            "b": [0.0, 1.0, 1.0],
            "c": [0.0, 0.0, 0.0],
            "d": [0.0, 0.0, 0.0],
            "e": [0.0, 0.0, 0.0],
            "f": [0.0, 0.0, 0.0],
            "g": [0.0, 0.0, 0.0],
            "h": [0.0, 0.0, 0.0],
        }
    )
    model = NNdex(matrix, normalized=False, backend="cpu")
    result = model.search(matrix.iloc[[0]], k=2, dataframe=matrix)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [*matrix.columns, "similarity"]
    assert len(result) == 2
    assert "similarity" in result.columns
    assert float(result.iloc[0]["similarity"]) >= float(result.iloc[1]["similarity"])


def test_pandas_dataframe_multi_query_returns_list_of_dataframes() -> None:
    pd = pytest.importorskip("pandas")

    matrix = pd.DataFrame(np.eye(8, dtype=np.float32))
    model = NNdex(matrix, normalized=True, backend="cpu")
    result = model.search(matrix.iloc[[0, 1]], k=2, dataframe=matrix)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(frame, pd.DataFrame) for frame in result)
    assert all("similarity" in frame.columns for frame in result)


def test_polars_dataframe_returns_dataframe() -> None:
    pl = pytest.importorskip("polars")

    matrix = pl.DataFrame(
        {
            "a": [1.0, 0.0, 1.0],
            "b": [0.0, 1.0, 1.0],
            "c": [0.0, 0.0, 0.0],
            "d": [0.0, 0.0, 0.0],
            "e": [0.0, 0.0, 0.0],
            "f": [0.0, 0.0, 0.0],
            "g": [0.0, 0.0, 0.0],
            "h": [0.0, 0.0, 0.0],
        }
    )
    model = NNdex(matrix, normalized=False, backend="cpu")
    result = model.search(matrix[:1], k=2, dataframe=matrix)

    assert isinstance(result, pl.DataFrame)
    assert result.columns == [*matrix.columns, "similarity"]
    assert result.height == 2
    assert "similarity" in result.columns


def test_polars_dataframe_multi_query_returns_list_of_dataframes() -> None:
    pl = pytest.importorskip("polars")

    matrix = pl.DataFrame(np.eye(8, dtype=np.float32))
    model = NNdex(matrix, normalized=True, backend="cpu")
    result = model.search(matrix[:2], k=2, dataframe=matrix)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(frame, pl.DataFrame) for frame in result)
    assert all("similarity" in frame.columns for frame in result)


def test_pandas_dataframe_query_works_when_index_built_from_numpy() -> None:
    pd = pytest.importorskip("pandas")

    matrix = np.eye(8, dtype=np.float32)
    dataframe = pd.DataFrame(matrix)
    model = NNdex(matrix, normalized=True, backend="cpu")
    result = model.search(dataframe, k=1, dataframe=dataframe)

    assert isinstance(result, list)
    assert len(result) == matrix.shape[0]
    assert all(isinstance(frame, pd.DataFrame) for frame in result)
    assert all("similarity" in frame.columns for frame in result)
    assert all(len(frame) == 1 for frame in result)


def test_pandas_dataframe_query_rows_mismatch_raises_when_index_built_from_numpy() -> None:
    pd = pytest.importorskip("pandas")

    matrix = np.eye(8, dtype=np.float32)
    dataframe = pd.DataFrame(matrix)
    model = NNdex(matrix, normalized=True, backend="cpu")

    with pytest.raises(ValueError, match="row count must match index rows"):
        model.search(matrix[0], k=1, dataframe=dataframe.iloc[:2])


def test_polars_dataframe_query_works_when_index_built_from_numpy() -> None:
    pl = pytest.importorskip("polars")

    matrix = np.eye(8, dtype=np.float32)
    dataframe = pl.DataFrame(matrix)
    model = NNdex(matrix, normalized=True, backend="cpu")
    result = model.search(dataframe, k=1, dataframe=dataframe)

    assert isinstance(result, list)
    assert len(result) == matrix.shape[0]
    assert all(isinstance(frame, pl.DataFrame) for frame in result)
    assert all("similarity" in frame.columns for frame in result)
    assert all(frame.height == 1 for frame in result)


def test_polars_dataframe_query_rows_mismatch_raises_when_index_built_from_numpy() -> None:
    pl = pytest.importorskip("polars")

    matrix = np.eye(8, dtype=np.float32)
    dataframe = pl.DataFrame(matrix)
    model = NNdex(matrix, normalized=True, backend="cpu")

    with pytest.raises(ValueError, match="row count must match index rows"):
        model.search(matrix[0], k=1, dataframe=dataframe[:2])


def test_dataframe_query_without_optional_dataframe_uses_default_output() -> None:
    pd = pytest.importorskip("pandas")

    matrix = np.eye(8, dtype=np.float32)
    query_df = pd.DataFrame(matrix[:2])
    model = NNdex(matrix, normalized=True, backend="cpu")
    indices, scores = model.search(query_df, k=1)

    assert indices.shape == (2, 1)
    assert scores.shape == (2, 1)


def test_from_file_npy_roundtrip(tmp_path: Path) -> None:
    matrix = np.eye(8, dtype=np.float32)
    file_path = tmp_path / "matrix.npy"
    np.save(file_path, matrix)

    model = NNdex.from_file(str(file_path), normalized=True, backend="cpu")
    indices, scores = model.search(matrix[0], k=1)

    assert int(indices[0]) == 0
    assert float(scores[0]) == pytest.approx(1.0, rel=1e-6)


def test_from_file_npz_with_key(tmp_path: Path) -> None:
    matrix = np.eye(8, dtype=np.float32)
    file_path = tmp_path / "matrix.npz"
    np.savez(file_path, embeddings=matrix)

    model = NNdex.from_file(str(file_path), key="embeddings", normalized=True, backend="cpu")
    indices, _scores = model.search(matrix[3], k=1)

    assert int(indices[0]) == 3


def test_from_file_parquet_with_fixed_size_list_column(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    matrix = np.eye(8, dtype=np.float32)
    embedding_values = matrix.tolist()
    embedding_array = pa.array(embedding_values, type=pa.list_(pa.float32(), list_size=8))
    table = pa.table({"embedding": embedding_array})

    file_path = tmp_path / "matrix.parquet"
    pq.write_table(table, file_path)

    model = NNdex.from_file(str(file_path), key="embedding", normalized=True, backend="cpu")
    indices, _scores = model.search(matrix[5], k=1)

    assert int(indices[0]) == 5
