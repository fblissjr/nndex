from __future__ import annotations

import os
import statistics
import time
from typing import Callable

import numpy as np

PREDICT_BENCH_SIZES: list[tuple[int, int]] = [
    (1_000, 48),
    (1_000, 768),
    (10_000, 48),
    (10_000, 768),
    (50_000, 128),
    (100_000, 2),
    (1_000, 196_608),
    (10_000_000, 24),
]


def should_run_case(rows: int, dims: int) -> bool:
    extreme = bench_flag("NNDEX_BENCH_EXTREME", "FASTDOT_BENCH_EXTREME")
    if extreme:
        return True
    return rows * dims <= 50_000_000


def generate_matrix(rows: int, dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, dims)).astype(np.float32)


def generate_query(dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(dims,)).astype(np.float32)


def benchmark_ns(fn: Callable[[], object], repeats: int) -> int:
    samples: list[int] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        _ = fn()
        end = time.perf_counter_ns()
        samples.append(end - start)
    return int(statistics.median(samples))


def repeats_for_case(rows: int, dims: int) -> int:
    work = rows * dims
    if work >= 100_000_000:
        return 5
    if work >= 10_000_000:
        return 10
    return 20


def fmt_row(name: str, rows: int, dims: int, ns: int) -> str:
    ms = ns / 1_000_000
    return f"{name:20s} {rows:>10d} {dims:>8d} {ns:>15d} ns {ms:>12.6f} ms"


def bench_flag(primary: str, legacy: str) -> bool:
    value = os.environ.get(primary, os.environ.get(legacy, "0"))
    return int(value) == 1
