from __future__ import annotations

import numpy as np

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


def main() -> None:
    print("benchmark            rows     dims         median time")
    for rows, dims in PREDICT_BENCH_SIZES:
        if not should_run_case(rows, dims):
            continue

        matrix = generate_matrix(rows, dims, seed=rows + dims)
        repeats = repeats_for_case(rows, dims)

        cpu_ns = benchmark_ns(
            lambda: NNdex(matrix, normalized=False, backend="cpu"),
            repeats=repeats,
        )
        cpu_approx_ns = benchmark_ns(
            lambda: NNdex(matrix, normalized=False, approx=True, backend="cpu"),
            repeats=repeats,
        )
        gpu_ns = benchmark_ns(
            lambda: NNdex(matrix, normalized=False, backend="gpu"),
            repeats=repeats,
        )
        gpu_approx_ns = benchmark_ns(
            lambda: NNdex(matrix, normalized=False, approx=True, backend="gpu"),
            repeats=repeats,
        )
        numpy_ns = benchmark_ns(lambda: numpy_fit(matrix), repeats=repeats)

        print(fmt_row("binding_fit_cpu", rows, dims, cpu_ns))
        print(fmt_row("binding_fit_cpu_apx", rows, dims, cpu_approx_ns))
        print(fmt_row("binding_fit_gpu", rows, dims, gpu_ns))
        print(fmt_row("binding_fit_gpu_apx", rows, dims, gpu_approx_ns))
        print(fmt_row("numpy_fit_cpu", rows, dims, numpy_ns))
        print()


if __name__ == "__main__":
    main()
