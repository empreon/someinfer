from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.layers import LinearLayer
from src.engine import CudaEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end scaffold benchmark for custom engine vs TensorRT."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _benchmark_custom_layer(args: argparse.Namespace) -> dict[str, float]:
    np.random.seed(args.seed)

    engine = CudaEngine()
    layer = LinearLayer(
        engine=engine,
        in_features=args.in_features,
        out_features=args.out_features,
        matmul_kernel="tiled",
    )
    x = np.random.randn(args.batch_size, args.in_features).astype(np.float32)

    timings_ms: list[float] = []
    for _ in range(max(1, args.repeat)):
        start = perf_counter()
        layer.forward(x)
        elapsed_ms = (perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)

    return {
        "mean_ms": float(sum(timings_ms) / len(timings_ms)),
        "min_ms": float(min(timings_ms)),
        "max_ms": float(max(timings_ms)),
    }


def main() -> None:
    args = parse_args()

    custom_stats = _benchmark_custom_layer(args)

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("TensorRT is not installed in this environment.")
        print("Custom engine baseline was measured successfully.")
        print(
            f"custom_mean_ms={custom_stats['mean_ms']:.4f}, "
            f"custom_min_ms={custom_stats['min_ms']:.4f}, "
            f"custom_max_ms={custom_stats['max_ms']:.4f}"
        )
        print("Install TensorRT and extend benchmarks/bench_tensorrt.py for model-specific E2E tests.")
        return

    print("TensorRT module is available.")
    print(
        "This scaffold currently benchmarks the custom engine path only. "
        "Add model build/inference steps to compare TensorRT end-to-end."
    )
    print(
        f"custom_mean_ms={custom_stats['mean_ms']:.4f}, "
        f"custom_min_ms={custom_stats['min_ms']:.4f}, "
        f"custom_max_ms={custom_stats['max_ms']:.4f}"
    )


if __name__ == "__main__":
    main()
