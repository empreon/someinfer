from __future__ import annotations

from typing import Optional

import numpy as np
import pycuda.driver as cuda

from src.engine import Engine


class LinearLayer:
    def __init__(
        self,
        engine: Engine,
        in_features: int,
        out_features: int,
        weights: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        matmul_kernel: str = "tiled",
    ) -> None:
        self.engine = engine
        self.in_features = in_features
        self.out_features = out_features
        self.matmul_kernel = matmul_kernel

        if weights is None:
            scale = np.sqrt(2.0 / max(1, in_features))
            weights = (np.random.randn(in_features, out_features) * scale).astype(np.float32)
        else:
            weights = np.ascontiguousarray(weights, dtype=np.float32)
        if weights.shape != (in_features, out_features):
            raise ValueError(
                f"Weights must have shape {(in_features, out_features)}, got {weights.shape}"
            )

        if bias is not None:
            bias = np.ascontiguousarray(bias, dtype=np.float32)
            if bias.shape != (out_features,):
                raise ValueError(f"Bias must have shape {(out_features,)}, got {bias.shape}")

        self.weights_host = weights
        self.bias_host = bias

        prefix = f"linear_{id(self)}"
        self.weights_gpu = self.engine.upload(self.weights_host, name=f"{prefix}_weights")
        self.bias_gpu = (
            self.engine.upload(self.bias_host, name=f"{prefix}_bias")
            if self.bias_host is not None
            else None
        )

    def forward_device(
        self,
        input_gpu: cuda.DeviceAllocation,
        batch_size: int,
        output_gpu: Optional[cuda.DeviceAllocation] = None,
        block: tuple[int, int, int] = (16, 16, 1),
        stream: Optional[cuda.Stream] = None,
    ) -> cuda.DeviceAllocation:
        output_nbytes = batch_size * self.out_features * np.dtype(np.float32).itemsize
        if output_gpu is None:
            output_gpu = cuda.mem_alloc(output_nbytes)

        self.engine.run_matmul(
            input_gpu,
            self.weights_gpu,
            output_gpu,
            m=batch_size,
            k=self.in_features,
            n=self.out_features,
            kernel=self.matmul_kernel,
            block=block,
            stream=stream,
        )
        return output_gpu

    def forward(
        self,
        input_host: np.ndarray,
        block: tuple[int, int, int] = (16, 16, 1),
        stream: Optional[cuda.Stream] = None,
    ) -> np.ndarray:
        input_host = np.ascontiguousarray(input_host, dtype=np.float32)
        if input_host.ndim != 2 or input_host.shape[1] != self.in_features:
            raise ValueError(
                f"Input must have shape (batch_size, {self.in_features}), got {input_host.shape}"
            )

        input_gpu = self.engine.upload(input_host)
        output_gpu = self.forward_device(
            input_gpu, batch_size=input_host.shape[0], block=block, stream=stream
        )

        output_host = self.engine.download(
            output_gpu, shape=(input_host.shape[0], self.out_features), dtype=np.float32
        )
        if self.bias_host is not None:
            output_host += self.bias_host

        input_gpu.free()
        output_gpu.free()
        return output_host
