from __future__ import annotations

import numpy as np

from nndex import NNdex

from python_benchmarks.common import (
    PREDICT_BENCH_SIZES,
    benchmark_ns,
    fmt_row,
    generate_matrix,
    generate_query,
    repeats_for_case,
    should_run_case,
)


def numpy_prepare(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe_norms


def numpy_predict(
    matrix_norm: np.ndarray, query: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    query_norm = query / max(float(np.linalg.norm(query)), 1e-12)
    scores = matrix_norm @ query_norm
    top_idx = np.argpartition(scores, -k)[-k:]
    ordered = top_idx[np.argsort(scores[top_idx])[::-1]]
    return ordered, scores[ordered]


def main() -> None:
    print("benchmark            rows     dims         median time")
    for rows, dims in PREDICT_BENCH_SIZES:
        if not should_run_case(rows, dims):
            continue

        matrix = generate_matrix(rows, dims, seed=rows ^ dims)
        query = generate_query(dims, seed=rows + dims * 7)
        repeats = repeats_for_case(rows, dims)

        cpu_model = NNdex(matrix, normalized=False, backend="cpu")
        cpu_approx_model = NNdex(
            matrix, normalized=False, approx=True, backend="cpu"
        )
        gpu_model = NNdex(matrix, normalized=False, backend="gpu")
        gpu_approx_model = NNdex(
            matrix, normalized=False, approx=True, backend="gpu"
        )
        matrix_norm = numpy_prepare(matrix)

        # Warmup: trigger lazy ANN index build and GPU pipeline init
        # so first-call overhead doesn't pollute timing.
        cpu_model.search(query, k=10)
        cpu_approx_model.search(query, k=10)
        gpu_model.search(query, k=10)
        gpu_approx_model.search(query, k=10)

        cpu_ns = benchmark_ns(
            lambda: cpu_model.search(query, k=10), repeats=repeats
        )
        cpu_approx_ns = benchmark_ns(
            lambda: cpu_approx_model.search(query, k=10), repeats=repeats
        )
        gpu_ns = benchmark_ns(
            lambda: gpu_model.search(query, k=10), repeats=repeats
        )
        gpu_approx_ns = benchmark_ns(
            lambda: gpu_approx_model.search(query, k=10), repeats=repeats
        )
        numpy_ns = benchmark_ns(
            lambda: numpy_predict(matrix_norm, query, k=10), repeats=repeats
        )

        print(fmt_row("binding_predict_cpu", rows, dims, cpu_ns))
        print(fmt_row("binding_predict_cpu_apx", rows, dims, cpu_approx_ns))
        print(fmt_row("binding_predict_gpu", rows, dims, gpu_ns))
        print(fmt_row("binding_predict_gpu_apx", rows, dims, gpu_approx_ns))
        print(fmt_row("numpy_predict_cpu", rows, dims, numpy_ns))
        print()


if __name__ == "__main__":
    main()
