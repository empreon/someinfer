from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pycuda.driver as cuda

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import CudaEngine
from src.utils import build_matmul_vectorized_launch_config, gemm_gflops, gpu_benchmark


def _run_cublas_row_major(
    handle: int,
    a_gpu: cuda.DeviceAllocation,
    b_gpu: cuda.DeviceAllocation,
    c_gpu: cuda.DeviceAllocation,
    m: int,
    k: int,
    n: int,
) -> None:
    import skcuda.cublas as cublas

    alpha = np.float32(1.0)
    beta = np.float32(0.0)

    # cublas assumes column-major matrices.
    # To emulate C_row_major = A_row_major @ B_row_major without extra transpose kernels:
    # C_col_major(NxM) = B_col_major(NxK) @ A_col_major(KxM)
    cublas.cublasSgemm(
        handle,
        "n",
        "n",
        n,
        m,
        k,
        alpha,
        int(b_gpu),
        n,
        int(a_gpu),
        k,
        beta,
        int(c_gpu),
        n,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark custom PyCUDA matmul kernels against cuBLAS SGEMM."
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--kernel", choices=["naive", "tiled", "vectorized"], default="naive")
    parser.add_argument("--block-x", type=int, default=16)
    parser.add_argument("--block-y", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import skcuda.cublas as cublas
    except ImportError as exc:
        raise RuntimeError(
            "scikit-cuda is required for cuBLAS comparison. Install with `pip install scikit-cuda`."
        ) from exc

    np.random.seed(args.seed)
    m, k, n = args.m, args.k, args.n

    a_host = np.random.randn(m, k).astype(np.float32)
    b_host = np.random.randn(k, n).astype(np.float32)

    engine = CudaEngine()

    a_gpu = engine.upload(a_host, name="A")
    b_gpu = engine.upload(b_host, name="B")
    c_custom_gpu = engine.alloc("C_custom", m * n * np.dtype(np.float32).itemsize, reuse=False)
    c_cublas_gpu = engine.alloc("C_cublas", m * n * np.dtype(np.float32).itemsize, reuse=False)

    if args.kernel == "vectorized":
        _, block = build_matmul_vectorized_launch_config(m, n)
    else:
        block = (args.block_x, args.block_y, 1)

    handle = cublas.cublasCreate()
    try:
        custom_stats = gpu_benchmark(
            lambda: engine.run_matmul(
                a_gpu, b_gpu, c_custom_gpu, m=m, k=k, n=n, kernel=args.kernel, block=block
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        cublas_stats = gpu_benchmark(
            lambda: _run_cublas_row_major(handle, a_gpu, b_gpu, c_cublas_gpu, m, k, n),
            warmup=args.warmup,
            repeat=args.repeat,
        )
    finally:
        cublas.cublasDestroy(handle)

    c_custom = engine.download(c_custom_gpu, shape=(m, n), dtype=np.float32)
    c_cublas = engine.download(c_cublas_gpu, shape=(m, n), dtype=np.float32)

    is_close = np.allclose(c_custom, c_cublas, atol=args.atol)
    max_abs_diff = float(np.max(np.abs(c_custom - c_cublas)))

    custom_mean = custom_stats["mean_ms"]
    cublas_mean = cublas_stats["mean_ms"]
    speed_ratio = custom_mean / cublas_mean if cublas_mean > 0 else float("inf")

    print("=== Matrix Multiply Benchmark ===")
    print(f"Shapes: A=({m}, {k}), B=({k}, {n}), C=({m}, {n})")
    print(f"Custom kernel: {args.kernel}, block={block}")
    print(f"Custom mean: {custom_mean:.4f} ms")
    print(f"cuBLAS mean: {cublas_mean:.4f} ms")
    print(
        f"Custom GFLOPS: {gemm_gflops(m, k, n, custom_mean):.2f}, "
        f"cuBLAS GFLOPS: {gemm_gflops(m, k, n, cublas_mean):.2f}"
    )
    print(f"Latency ratio (custom/cuBLAS): {speed_ratio:.3f}x")
    print(f"np.allclose(atol={args.atol}): {is_close}")
    print(f"max abs diff: {max_abs_diff:.6e}")


if __name__ == "__main__":
    main()
