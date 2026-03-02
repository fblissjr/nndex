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
)

QUERY_BATCH_SIZES = [4, 128, 2048]


def numpy_prepare(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe_norms


def numpy_predict_batch(
    matrix_norm: np.ndarray, queries_norm: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    scores = queries_norm @ matrix_norm.T

    candidate_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    row_ids = np.arange(scores.shape[0])[:, None]
    candidate_scores = scores[row_ids, candidate_idx]
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]

    topk_idx = candidate_idx[row_ids, order]
    topk_scores = candidate_scores[row_ids, order]
    return topk_idx, topk_scores


def simsimd_predict_batch(
    matrix_norm: np.ndarray,
    queries_norm: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if simsimd is None:
        raise RuntimeError("simsimd is not installed")
    # queries first, matrix second => shape (query_rows, matrix_rows)
    distances = np.asarray(
        simsimd.cdist(queries_norm, matrix_norm, "cosine"),
        dtype=np.float32,
    )
    scores = 1.0 - distances

    candidate_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    row_ids = np.arange(scores.shape[0])[:, None]
    candidate_scores = scores[row_ids, candidate_idx]
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]

    return candidate_idx[row_ids, order], candidate_scores[row_ids, order]


def main() -> None:
    for query_rows in QUERY_BATCH_SIZES:
        print(f"\n--- batch={query_rows} ---")
        print(f"{'benchmark':24s} {'rows':>10s} {'dims':>8s} {'median time':>15s}")
        for rows, dims in PREDICT_BENCH_SIZES:
            if not should_run_case(rows, dims):
                continue

            print(f"\n  [{rows:,} x {dims}] generating data...", end="", flush=True)
            matrix = generate_matrix(rows, dims, seed=rows ^ dims)
            queries = generate_matrix(query_rows, dims, seed=rows + dims * 7)
            # Pre-normalize queries so all implementations start from the same place
            queries_norm = numpy_prepare(queries)
            repeats = max(3, repeats_for_case(rows, dims) // 3)

            print(" building indexes...", end="", flush=True)
            cpu_model = NNdex(matrix, normalized=False, backend="cpu", enable_cache=False)
            cpu_approx_model = NNdex(
                matrix,
                normalized=False,
                approx=True,
                backend="cpu",
                enable_cache=False,
            )
            matrix_norm = numpy_prepare(matrix)

            # Warmup: trigger lazy ANN index build so first-call overhead
            # doesn't pollute timing.
            print(" warming up...", end="", flush=True)
            cpu_model.search(queries_norm, k=10)
            cpu_approx_model.search(queries_norm, k=10)

            print(f" benchmarking ({repeats} repeats)...")

            cpu_ns = benchmark_ns(lambda: cpu_model.search(queries_norm, k=10), repeats=repeats)
            cpu_approx_ns = benchmark_ns(
                lambda: cpu_approx_model.search(queries_norm, k=10), repeats=repeats
            )
            numpy_ns = benchmark_ns(
                lambda: numpy_predict_batch(matrix_norm, queries_norm, k=10), repeats=repeats
            )

            simsimd_ns = None
            if simsimd is not None:
                try:
                    simsimd_ns = benchmark_ns(
                        lambda: simsimd_predict_batch(matrix_norm, queries_norm, k=10),
                        repeats=repeats,
                    )
                except Exception as exc:
                    print(f"  simsimd skipped: {type(exc).__name__}: {exc}")

            print(fmt_row("binding_batch_cpu", rows, dims, cpu_ns))
            print(fmt_row("binding_batch_cpu_apx", rows, dims, cpu_approx_ns))
            print(fmt_row("numpy_batch_cpu", rows, dims, numpy_ns))
            if simsimd_ns is not None:
                print(fmt_row("simsimd_batch_cpu", rows, dims, simsimd_ns))


if __name__ == "__main__":
    main()
