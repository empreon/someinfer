from __future__ import annotations

from typing import Callable, Dict

import pycuda.driver as cuda


def gpu_benchmark(
    fn: Callable[[], None], warmup: int = 3, repeat: int = 20
) -> Dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    cuda.Context.synchronize()

    samples = []
    for _ in range(max(1, repeat)):
        start = cuda.Event()
        stop = cuda.Event()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.time_till(stop))

    mean_ms = float(sum(samples) / len(samples))
    min_ms = float(min(samples))
    max_ms = float(max(samples))
    return {
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "samples": samples,
    }


def gemm_gflops(m: int, k: int, n: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0.0:
        return 0.0
    return (2.0 * m * k * n) / (elapsed_ms * 1e6)
