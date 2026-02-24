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
    repeats_for_case,
    should_run_case,
)


def numpy_fit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe_norms


def simsimd_fit(matrix: np.ndarray, zero_vector: np.ndarray) -> np.ndarray:
    if simsimd is None:
        raise RuntimeError("simsimd is not installed")
    norms_sq = np.asarray(
        simsimd.cdist(matrix, zero_vector, "sqeuclidean"),
        dtype=np.float32,
    )
    safe_norms = np.maximum(np.sqrt(norms_sq), 1e-12)
    return matrix / safe_norms


def main() -> None:
    print("benchmark            rows     dims         median time")
    for rows, dims in PREDICT_BENCH_SIZES:
        if not should_run_case(rows, dims):
            continue

        matrix = generate_matrix(rows, dims, seed=rows + dims)
        zero_vector = np.zeros((1, dims), dtype=np.float32)
        repeats = repeats_for_case(rows, dims)

        cpu_ns = benchmark_ns(
            lambda: NNdex(
                matrix, normalized=False, backend="cpu", enable_cache=False
            ),
            repeats=repeats,
        )
        cpu_approx_ns = benchmark_ns(
            lambda: NNdex(
                matrix,
                normalized=False,
                approx=True,
                backend="cpu",
                enable_cache=False,
            ),
            repeats=repeats,
        )
        gpu_ns = benchmark_ns(
            lambda: NNdex(
                matrix, normalized=False, backend="gpu", enable_cache=False
            ),
            repeats=repeats,
        )
        gpu_approx_ns = benchmark_ns(
            lambda: NNdex(
                matrix,
                normalized=False,
                approx=True,
                backend="gpu",
                enable_cache=False,
            ),
            repeats=repeats,
        )
        numpy_ns = benchmark_ns(lambda: numpy_fit(matrix), repeats=repeats)
        simsimd_ns = (
            benchmark_ns(
                lambda: simsimd_fit(matrix, zero_vector),
                repeats=repeats,
            )
            if simsimd is not None
            else None
        )

        print(fmt_row("binding_fit_cpu", rows, dims, cpu_ns))
        print(fmt_row("binding_fit_cpu_apx", rows, dims, cpu_approx_ns))
        print(fmt_row("binding_fit_gpu", rows, dims, gpu_ns))
        print(fmt_row("binding_fit_gpu_apx", rows, dims, gpu_approx_ns))
        print(fmt_row("numpy_fit_cpu", rows, dims, numpy_ns))
        if simsimd_ns is not None:
            print(fmt_row("simsimd_fit_cpu", rows, dims, simsimd_ns))
        print()


if __name__ == "__main__":
    main()
