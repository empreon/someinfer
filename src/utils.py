from __future__ import annotations

from typing import Callable, Dict, Tuple

import pycuda.driver as cuda


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def build_matmul_launch_config(
    m: int, n: int, block: Tuple[int, int, int] = (16, 16, 1)
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    grid = (ceil_div(n, block[0]), ceil_div(m, block[1]), 1)
    return grid, block


def build_matmul_tiled_launch_config(
    m: int, n: int, tile_size: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    block = (tile_size, tile_size, 1)
    grid = (ceil_div(n, tile_size), ceil_div(m, tile_size), 1)
    return grid, block


def build_matmul_vectorized_launch_config(
    m: int,
    n: int,
    tile_size: int = 32,
    vec_width: int = 4,
    vblock_rows: int = 2,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    block = (tile_size // vec_width, tile_size // vblock_rows, 1)
    grid = (ceil_div(n, tile_size), ceil_div(m, tile_size), 1)
    return grid, block


def build_elementwise_launch_config(
    numel: int, block: Tuple[int, int, int] = (256, 1, 1)
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    grid = (ceil_div(numel, block[0]), 1, 1)
    return grid, block


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
