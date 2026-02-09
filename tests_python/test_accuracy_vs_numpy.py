import numpy as np

from nndex import NNdex


def numpy_topk(matrix: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    matrix_norm = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    query_norm = query / max(float(np.linalg.norm(query)), 1e-12)
    scores = matrix_norm @ query_norm
    top_idx = np.argpartition(scores, -k)[-k:]
    ordered = top_idx[np.argsort(scores[top_idx])[::-1]]
    return ordered, scores[ordered]


def test_python_binding_matches_numpy_topk() -> None:
    rng = np.random.default_rng(123)
    rows, dims, k = 4096, 128, 25
    matrix = rng.normal(size=(rows, dims)).astype(np.float32)
    query = rng.normal(size=(dims,)).astype(np.float32)

    model = NNdex(matrix, normalized=False, backend="cpu")
    indices, scores = model.search(query, k=k)
    np_indices, np_scores = numpy_topk(matrix, query, k)

    assert np.array_equal(indices, np_indices)
    assert np.allclose(scores, np_scores, atol=1e-5)


def test_approx_mode_remains_roughly_accurate() -> None:
    rng = np.random.default_rng(456)
    rows, dims, k = 4096, 128, 25
    matrix = rng.normal(size=(rows, dims)).astype(np.float32)
    query = rng.normal(size=(dims,)).astype(np.float32)

    exact_model = NNdex(matrix, normalized=False, approx=False, backend="cpu")
    approx_model = NNdex(matrix, normalized=False, approx=True, backend="cpu")
    exact_indices, _ = exact_model.search(query, k=k)
    approx_indices, _ = approx_model.search(query, k=k)

    overlap = len(set(exact_indices.tolist()) & set(approx_indices.tolist()))
    assert overlap >= 15
