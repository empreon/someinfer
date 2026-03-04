from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class Engine:
    TILED_TILE = 32
    VEC_TILE = 32
    VEC_WIDTH = 4
    VBLOCK_ROWS = 2

    def __init__(
        self, kernel_dir: Optional[Path | str] = None, nvcc_options: Optional[list[str]] = None
    ) -> None:
        self.kernel_dir = (
            Path(kernel_dir).resolve()
            if kernel_dir is not None
            else Path(__file__).resolve().parents[1] / "kernels"
        )
        if not self.kernel_dir.exists():
            raise FileNotFoundError(f"Kernel directory not found: {self.kernel_dir}")

        self.nvcc_options = nvcc_options or ["-std=c++11"]
        self._modules: Dict[str, SourceModule] = {}
        self._kernel_cache: Dict[Tuple[str, str], cuda.Function] = {}
        self._memory_pool: Dict[str, Tuple[cuda.DeviceAllocation, int]] = {}

    def _normalize_module_name(self, file_name: str | Path) -> str:
        return Path(file_name).as_posix()

    def _build_compile_options(self, module_name: str) -> list[str]:
        options = [*self.nvcc_options, "-I", str(self.kernel_dir)]
        if module_name == "fp32/matmul.cu":
            options.extend(
                [
                    f"-DMM_TILED_TILE={self.TILED_TILE}",
                    f"-DMM_VEC_TILE={self.VEC_TILE}",
                    f"-DMM_VEC_WIDTH={self.VEC_WIDTH}",
                    f"-DMM_VBLOCK_ROWS={self.VBLOCK_ROWS}",
                ]
            )
        return options

    def compile_kernel_file(self, file_name: str | Path) -> SourceModule:
        module_name = self._normalize_module_name(file_name)
        source_path = self.kernel_dir / module_name
        if not source_path.exists():
            raise FileNotFoundError(f"Kernel source not found: {source_path}")

        source = source_path.read_text(encoding="utf-8")
        options = self._build_compile_options(module_name)
        module = SourceModule(source, options=options, no_extern_c=True)
        self._modules[module_name] = module

        stale_keys = [key for key in self._kernel_cache if key[0] == module_name]
        for key in stale_keys:
            del self._kernel_cache[key]
        return module

    def get_kernel(
        self, kernel_name: str, module_name: Optional[str | Path] = None
    ) -> cuda.Function:
        if module_name is not None:
            module_name = self._normalize_module_name(module_name)
            cache_key = (module_name, kernel_name)
            if cache_key in self._kernel_cache:
                return self._kernel_cache[cache_key]

            if module_name not in self._modules:
                self.compile_kernel_file(module_name)
            module = self._modules[module_name]
            function = module.get_function(kernel_name)
            self._kernel_cache[cache_key] = function
            return function

        for current_module_name, module in self._modules.items():
            cache_key = (current_module_name, kernel_name)
            if cache_key in self._kernel_cache:
                return self._kernel_cache[cache_key]
            try:
                function = module.get_function(kernel_name)
            except cuda.LogicError:
                continue
            self._kernel_cache[cache_key] = function
            return function

        raise KeyError(f"Kernel function '{kernel_name}' was not found in compiled modules.")

    def alloc(self, name: str, nbytes: int, reuse: bool = True) -> cuda.DeviceAllocation:
        if name in self._memory_pool:
            allocation, current_nbytes = self._memory_pool[name]
            if reuse and current_nbytes >= nbytes:
                return allocation
            allocation.free()
            del self._memory_pool[name]

        allocation = cuda.mem_alloc(nbytes)
        self._memory_pool[name] = (allocation, nbytes)
        return allocation

    def free(self, name: str) -> None:
        if name in self._memory_pool:
            allocation, _ = self._memory_pool.pop(name)
            allocation.free()

    def free_all(self) -> None:
        for allocation, _ in self._memory_pool.values():
            allocation.free()
        self._memory_pool.clear()

    def upload(self, host_array: np.ndarray, name: Optional[str] = None) -> cuda.DeviceAllocation:
        contiguous = np.ascontiguousarray(host_array)
        if name is None:
            device_allocation = cuda.mem_alloc(contiguous.nbytes)
        else:
            device_allocation = self.alloc(name, contiguous.nbytes)
        cuda.memcpy_htod(device_allocation, contiguous)
        return device_allocation

    def download(
        self, device_allocation: cuda.DeviceAllocation, shape: tuple[int, ...], dtype: np.dtype
    ) -> np.ndarray:
        host_array = np.empty(shape, dtype=dtype)
        cuda.memcpy_dtoh(host_array, device_allocation)
        return host_array

    def run_matmul(
        self,
        a_gpu: cuda.DeviceAllocation,
        b_gpu: cuda.DeviceAllocation,
        c_gpu: cuda.DeviceAllocation,
        m: int,
        k: int,
        n: int,
        kernel: str = "naive",
        block: tuple[int, int, int] = (16, 16, 1),
        stream: Optional[cuda.Stream] = None,
    ) -> None:
        kernel_map = {
            "naive": "matmul_naive",
            "tiled": "matmul_tiled",
            "vectorized": "matmul_vectorized",
        }
        if kernel not in kernel_map:
            raise ValueError(f"Unsupported matmul kernel: {kernel}")

        function = self.get_kernel(kernel_map[kernel], module_name="fp32/matmul.cu")
        if kernel == "naive":
            launch_block = block
            grid = (_ceil_div(n, launch_block[0]), _ceil_div(m, launch_block[1]), 1)
        elif kernel == "tiled":
            launch_block = (self.TILED_TILE, self.TILED_TILE, 1)
            grid = (_ceil_div(n, self.TILED_TILE), _ceil_div(m, self.TILED_TILE), 1)
        else:
            launch_block = (self.VEC_TILE // self.VEC_WIDTH, self.VEC_TILE // self.VBLOCK_ROWS, 1)
            expected_block = (
                self.VEC_TILE // self.VEC_WIDTH,
                self.VEC_TILE // self.VBLOCK_ROWS,
                1,
            )
            if block not in ((16, 16, 1), expected_block):
                raise ValueError(
                    f"Vectorized matmul uses fixed block={expected_block}. "
                    "Use default block or explicitly pass the fixed value."
                )
            grid = (_ceil_div(n, self.VEC_TILE), _ceil_div(m, self.VEC_TILE), 1)

        function(
            a_gpu,
            b_gpu,
            c_gpu,
            np.int32(m),
            np.int32(k),
            np.int32(n),
            block=launch_block,
            grid=grid,
            stream=stream,
        )

    def run_activation(
        self,
        activation: str,
        x_gpu: cuda.DeviceAllocation,
        y_gpu: cuda.DeviceAllocation,
        numel: int,
        block: tuple[int, int, int] = (256, 1, 1),
        stream: Optional[cuda.Stream] = None,
    ) -> None:
        activation_map = {
            "relu": "relu_forward",
            "sigmoid": "sigmoid_forward",
            "tanh": "tanh_forward",
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")

        function = self.get_kernel(activation_map[activation], module_name="fp32/activations.cu")
        grid = (_ceil_div(numel, block[0]), 1, 1)
        function(
            x_gpu,
            y_gpu,
            np.int32(numel),
            block=block,
            grid=grid,
            stream=stream,
        )
