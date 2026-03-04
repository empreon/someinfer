from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import CudaEngine
from src.utils import build_matmul_vectorized_launch_config, gemm_gflops, gpu_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark custom FP32 CUDA matmul kernels against each other."
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--kernels", nargs="+", choices=["naive", "tiled", "vectorized"])
    parser.add_argument("--block-x", type=int, default=16)
    parser.add_argument("--block-y", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def _resolve_block(args: argparse.Namespace, m: int, n: int, kernel: str) -> tuple[int, int, int]:
    if kernel == "vectorized":
        _, block = build_matmul_vectorized_launch_config(m, n)
        return block
    return (args.block_x, args.block_y, 1)


def main() -> None:
    args = parse_args()
    kernels = args.kernels or ["naive", "tiled", "vectorized"]

    np.random.seed(args.seed)
    m, k, n = args.m, args.k, args.n

    a_host = np.random.randn(m, k).astype(np.float32)
    b_host = np.random.randn(k, n).astype(np.float32)
    c_ref = a_host @ b_host

    engine = CudaEngine()
    a_gpu = engine.upload(a_host, name="A")
    b_gpu = engine.upload(b_host, name="B")

    results: list[dict[str, float | str | bool]] = []
    c_nbytes = m * n * np.dtype(np.float32).itemsize
    for kernel in kernels:
        c_gpu = engine.alloc(f"C_{kernel}", c_nbytes, reuse=False)
        block = _resolve_block(args, m, n, kernel)
        stats = gpu_benchmark(
            lambda: engine.run_matmul(
                a_gpu, b_gpu, c_gpu, m=m, k=k, n=n, kernel=kernel, block=block
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        c_host = engine.download(c_gpu, shape=(m, n), dtype=np.float32)
        max_abs_diff = float(np.max(np.abs(c_host - c_ref)))
        results.append(
            {
                "kernel": kernel,
                "block": str(block),
                "mean_ms": float(stats["mean_ms"]),
                "gflops": gemm_gflops(m, k, n, float(stats["mean_ms"])),
                "allclose": bool(np.allclose(c_host, c_ref, atol=args.atol)),
                "max_abs_diff": max_abs_diff,
            }
        )

    print("=== Custom Kernel Benchmark ===")
    print(f"Shapes: A=({m}, {k}), B=({k}, {n}), C=({m}, {n})")
    print(f"{'kernel':<11} {'block':<12} {'mean_ms':>10} {'gflops':>10} {'allclose':>10} {'max_abs_diff':>14}")
    for result in results:
        print(
            f"{result['kernel']:<11} {result['block']:<12} "
            f"{result['mean_ms']:>10.4f} {result['gflops']:>10.2f} "
            f"{str(result['allclose']):>10} {result['max_abs_diff']:>14.6e}"
        )


if __name__ == "__main__":
    main()
