from __future__ import annotations

import numpy as np
try:
    import simsimd
except ImportError:  # pragma: no cover - optional benchmarking dependency
    simsimd = None

from nndex import NNdex

from python_benchmarks.common import (
    PREDICT_BENCH_SIZES,
    benchmark_ns,
    fmt_row,
    generate_matrix,
    generate_query,
    repeats_for_case,
    should_run_case,
    topk_from_scores,
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
    return topk_from_scores(scores, k)


def simsimd_predict(
    matrix_norm: np.ndarray,
    query: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if simsimd is None:
        raise RuntimeError("simsimd is not installed")
    query_norm = query / max(float(np.linalg.norm(query)), 1e-12)
    distances = np.asarray(
        simsimd.cdist(matrix_norm, query_norm.reshape(1, -1), "cosine"),
        dtype=np.float32,
    ).reshape(-1)
    scores = 1.0 - distances
    return topk_from_scores(scores, k)


def main() -> None:
    print("benchmark            rows     dims         median time")
    for rows, dims in PREDICT_BENCH_SIZES:
        if not should_run_case(rows, dims):
            continue

        matrix = generate_matrix(rows, dims, seed=rows ^ dims)
        query = generate_query(dims, seed=rows + dims * 7)
        repeats = repeats_for_case(rows, dims)

        cpu_model = NNdex(
            matrix, normalized=False, backend="cpu", enable_cache=False
        )
        cpu_approx_model = NNdex(
            matrix,
            normalized=False,
            approx=True,
            backend="cpu",
            enable_cache=False,
        )
        gpu_model = NNdex(
            matrix, normalized=False, backend="gpu", enable_cache=False
        )
        gpu_approx_model = NNdex(
            matrix,
            normalized=False,
            approx=True,
            backend="gpu",
            enable_cache=False,
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
        simsimd_ns = (
            benchmark_ns(
                lambda: simsimd_predict(matrix_norm, query, k=10), repeats=repeats
            )
            if simsimd is not None
            else None
        )

        print(fmt_row("binding_predict_cpu", rows, dims, cpu_ns))
        print(fmt_row("binding_predict_cpu_apx", rows, dims, cpu_approx_ns))
        print(fmt_row("binding_predict_gpu", rows, dims, gpu_ns))
        print(fmt_row("binding_predict_gpu_apx", rows, dims, gpu_approx_ns))
        print(fmt_row("numpy_predict_cpu", rows, dims, numpy_ns))
        if simsimd_ns is not None:
            print(fmt_row("simsimd_predict_cpu", rows, dims, simsimd_ns))
        print()


if __name__ == "__main__":
    main()
