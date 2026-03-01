from __future__ import annotations

import numpy as np

try:
    import simsimd
except ImportError:  # pragma: no cover
    simsimd = None

from nndex import NNdex

from python_benchmarks.common import (
    PREDICT_BENCH_SIZES,
    benchmark_ns,
    fmt_row,
    generate_matrix,
    repeats_for_case,
    should_run_case,
    topk_from_scores,
)

# Standard batch size to test the PyO3 flattening and GemmChunked paths
QUERY_ROWS = 128


def numpy_prepare(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe_norms


def numpy_predict_batch(
    matrix_norm: np.ndarray, queries: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    queries_norm = numpy_prepare(queries)
    scores = queries_norm @ matrix_norm.T

    # Argpartition along the rows for top-k
    candidate_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    row_ids = np.arange(scores.shape[0])[:, None]
    candidate_scores = scores[row_ids, candidate_idx]
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]

    topk_idx = candidate_idx[row_ids, order]
    topk_scores = candidate_scores[row_ids, order]
    return topk_idx, topk_scores


def simsimd_predict_batch(
    matrix_norm: np.ndarray,
    queries: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if simsimd is None:
        raise RuntimeError("simsimd is not installed")
    queries_norm = numpy_prepare(queries)
    # simsimd cdist returns distances, convert to cosine similarity
    distances = np.asarray(
        simsimd.cdist(matrix_norm, queries_norm, "cosine"),
        dtype=np.float32,
    )
    # scores = 1 - distance. Note cdist output shape is (matrix_rows, query_rows), so we transpose it
    scores = (1.0 - distances).T

    candidate_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    row_ids = np.arange(scores.shape[0])[:, None]
    candidate_scores = scores[row_ids, candidate_idx]
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]

    return candidate_idx[row_ids, order], candidate_scores[row_ids, order]


def main() -> None:
    print(f"benchmark (batch={QUERY_ROWS})   rows     dims         median time")
    for rows, dims in PREDICT_BENCH_SIZES:
        if not should_run_case(rows, dims):
            continue

        matrix = generate_matrix(rows, dims, seed=rows ^ dims)
        # Generate a 2D matrix of queries instead of a 1D vector
        queries = generate_matrix(QUERY_ROWS, dims, seed=rows + dims * 7)
        # Reduce repeats slightly because batch queries take longer
        repeats = max(3, repeats_for_case(rows, dims) // 3)

        cpu_model = NNdex(matrix, normalized=False, backend="cpu", enable_cache=False)
        cpu_approx_model = NNdex(
            matrix,
            normalized=False,
            approx=True,
            backend="cpu",
            enable_cache=False,
        )
        matrix_norm = numpy_prepare(matrix)

        # Warmup
        cpu_model.search(queries, k=10)
        cpu_approx_model.search(queries, k=10)

        cpu_ns = benchmark_ns(lambda: cpu_model.search(queries, k=10), repeats=repeats)
        cpu_approx_ns = benchmark_ns(
            lambda: cpu_approx_model.search(queries, k=10), repeats=repeats
        )
        numpy_ns = benchmark_ns(
            lambda: numpy_predict_batch(matrix_norm, queries, k=10), repeats=repeats
        )

        simsimd_ns = None
        if simsimd is not None:
            try:
                simsimd_ns = benchmark_ns(
                    lambda: simsimd_predict_batch(matrix_norm, queries, k=10),
                    repeats=repeats,
                )
            except Exception:
                pass  # Skip if simsimd memory allocation fails on massive matrices

        print(fmt_row("binding_batch_cpu", rows, dims, cpu_ns))
        print(fmt_row("binding_batch_cpu_apx", rows, dims, cpu_approx_ns))
        print(fmt_row("numpy_batch_cpu", rows, dims, numpy_ns))
        if simsimd_ns is not None:
            print(fmt_row("simsimd_batch_cpu", rows, dims, simsimd_ns))
        print()


if __name__ == "__main__":
    main()
