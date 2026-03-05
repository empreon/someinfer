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
    TILED_TILE = 64
    VEC_TILE = 64
    VEC_WIDTH = 4
    VBLOCK_ROWS = 8

    def __init__(self, kernel_dir: Optional[Path | str] = None, nvcc_options: Optional[list[str]] = None) -> None:
        self.kernel_dir = Path(kernel_dir).resolve() if kernel_dir is not None else Path(__file__).resolve().parents[1] / "kernels"
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
            options.extend([f"-DMM_TILED_TILE={self.TILED_TILE}", f"-DMM_VEC_TILE={self.VEC_TILE}", f"-DMM_VEC_WIDTH={self.VEC_WIDTH}", f"-DMM_VBLOCK_ROWS={self.VBLOCK_ROWS}"])
        return options

    def compile_kernel_file(self, file_name: str | Path) -> SourceModule:
        module_name = self._normalize_module_name(file_name)
        source_path = self.kernel_dir / module_name
        if not source_path.exists():
            raise FileNotFoundError(f"Kernel source not found: {source_path}")
        source = source_path.read_text(encoding="utf-8")
        module = SourceModule(source, options=self._build_compile_options(module_name), no_extern_c=True)
        self._modules[module_name] = module
        for key in [key for key in self._kernel_cache if key[0] == module_name]:
            del self._kernel_cache[key]
        return module

    def get_kernel(self, kernel_name: str, module_name: Optional[str | Path] = None) -> cuda.Function:
        if module_name is not None:
            module_name = self._normalize_module_name(module_name)
            cache_key = (module_name, kernel_name)
            if cache_key in self._kernel_cache:
                return self._kernel_cache[cache_key]
            if module_name not in self._modules:
                self.compile_kernel_file(module_name)
            function = self._modules[module_name].get_function(kernel_name)
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
            self._memory_pool.pop(name)[0].free()

    def free_all(self) -> None:
        for allocation, _ in self._memory_pool.values():
            allocation.free()
        self._memory_pool.clear()

    def upload(self, host_array: np.ndarray, name: Optional[str] = None) -> cuda.DeviceAllocation:
        contiguous = np.ascontiguousarray(host_array)
        device_allocation = cuda.mem_alloc(contiguous.nbytes) if name is None else self.alloc(name, contiguous.nbytes)
        cuda.memcpy_htod(device_allocation, contiguous)
        return device_allocation

    def download(self, device_allocation: cuda.DeviceAllocation, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        host_array = np.empty(shape, dtype=dtype)
        cuda.memcpy_dtoh(host_array, device_allocation)
        return host_array

    def _validate_matmul_config(self, a_gpu: cuda.DeviceAllocation, b_gpu: cuda.DeviceAllocation, k: int, n: int, block: tuple[int, int, int]) -> tuple[int, int, int]:
        if self.VEC_TILE <= 0 or self.VEC_WIDTH <= 0 or self.VBLOCK_ROWS <= 0:
            raise ValueError("Invalid matmul config: VEC_TILE, VEC_WIDTH and VBLOCK_ROWS must be > 0.")
        if self.VEC_TILE % self.VEC_WIDTH != 0:
            raise ValueError(f"Invalid matmul config: VEC_TILE ({self.VEC_TILE}) must be divisible by VEC_WIDTH ({self.VEC_WIDTH}).")
        if self.VEC_TILE % self.VBLOCK_ROWS != 0:
            raise ValueError(f"Invalid matmul config: VEC_TILE ({self.VEC_TILE}) must be divisible by VBLOCK_ROWS ({self.VBLOCK_ROWS}).")
        if k % self.VEC_TILE != 0:
            raise ValueError(f"matmul currently expects K to be divisible by {self.VEC_TILE}, got K={k}.")
        if n % self.VEC_TILE != 0:
            raise ValueError(f"matmul currently expects N to be divisible by {self.VEC_TILE}, got N={n}.")
        if int(a_gpu) % 16 != 0 or int(b_gpu) % 16 != 0:
            raise ValueError("matmul currently expects A and B device pointers to be 16-byte aligned.")
        expected_block = (self.VEC_TILE // self.VEC_WIDTH, self.VEC_TILE // self.VBLOCK_ROWS, 1)
        if block != expected_block:
            raise ValueError(f"matmul currently expects block={expected_block}. Update kernels/fp32/matmul.cu::matmul and this launch config together.")
        return expected_block

    def matmul(self, a_gpu: cuda.DeviceAllocation, b_gpu: cuda.DeviceAllocation, c_gpu: cuda.DeviceAllocation, m: int, k: int, n: int, block: Optional[tuple[int, int, int]] = None, stream: Optional[cuda.Stream] = None) -> None:
        default_block = (self.VEC_TILE // self.VEC_WIDTH, self.VEC_TILE // self.VBLOCK_ROWS, 1)
        launch_block = default_block if block is None else block
        self._validate_matmul_config(a_gpu, b_gpu, k, n, launch_block)
        function = self.get_kernel("matmul", module_name="fp32/matmul.cu")
        function.set_attribute(cuda.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, 66560)
        function(a_gpu, b_gpu, c_gpu, np.int32(m), np.int32(k), np.int32(n), block=launch_block, grid=(_ceil_div(n, self.VEC_TILE), _ceil_div(m, self.VEC_TILE), 1), stream=stream, shared=66560)

    def __matmul__(self, operands: tuple[object, ...]) -> cuda.DeviceAllocation:
        if not isinstance(operands, tuple):
            raise TypeError("Engine @ expects a tuple: (a_gpu, b_gpu, c_gpu, m, k, n[, block[, stream]])")
        if len(operands) < 6 or len(operands) > 8:
            raise ValueError("Engine @ expects 6-8 tuple entries: (a_gpu, b_gpu, c_gpu, m, k, n[, block[, stream]])")
        a_gpu, b_gpu, c_gpu, m, k, n, *optional = operands
        self.matmul(a_gpu, b_gpu, c_gpu, m=int(m), k=int(k), n=int(n), block=optional[0] if len(optional) >= 1 else None, stream=optional[1] if len(optional) >= 2 else None)
        return c_gpu
