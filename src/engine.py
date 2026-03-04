from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from src.utils import (
    build_elementwise_launch_config,
    build_matmul_launch_config,
    build_matmul_tiled_launch_config,
    build_matmul_vectorized_launch_config,
)


class CudaEngine:
    def __init__(
        self,
        kernel_dir: Optional[Path | str] = None,
        nvcc_options: Optional[list[str]] = None,
        compile_on_init: bool = True,
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
        self._matmul_tuning = self._build_matmul_tuning()

        if compile_on_init:
            self.compile_all()

    def _normalize_module_name(self, file_name: str | Path) -> str:
        return Path(file_name).as_posix()

    def _build_matmul_tuning(self) -> Dict[str, int]:
        defaults = {
            "tiled_tile": 32,
            "vec_tile": 32,
            "vec_width": 4,
            "vblock_rows": 2,
        }
        try:
            attributes = cuda.Context.get_device().get_attributes()
        except cuda.LogicError:
            return defaults

        max_threads = int(attributes.get(cuda.device_attribute.MAX_THREADS_PER_BLOCK, 1024))
        max_shared = int(
            attributes.get(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK, 48 * 1024)
        )
        max_registers = int(attributes.get(cuda.device_attribute.MAX_REGISTERS_PER_BLOCK, 65536))

        # RTX 4050 Laptop class profile
        # 32x32 vectorized tile with vblock_rows=2 (block=(8,16,1)).
        if (
            max_threads >= 1024
            and max_shared >= 48 * 1024
            and max_registers >= 65536
        ):
            return defaults

        tiled_tile = defaults["tiled_tile"]
        for candidate_tile in (32, 16, 8):
            threads = candidate_tile * candidate_tile
            shared_bytes = 2 * candidate_tile * candidate_tile * np.dtype(np.float32).itemsize
            reg_budget_per_thread = max_registers // max(1, threads)
            if (
                threads <= max_threads
                and shared_bytes <= max_shared
                and reg_budget_per_thread >= 32
            ):
                tiled_tile = candidate_tile
                break

        vec_width = defaults["vec_width"]
        best_score: Optional[Tuple[int, int, int]] = None
        vec_tile = 16
        vblock_rows = 1
        for candidate_tile in (32, 16):
            if candidate_tile % vec_width != 0:
                continue
            shared_bytes = 2 * candidate_tile * candidate_tile * np.dtype(np.float32).itemsize
            if shared_bytes > max_shared:
                continue
            for candidate_vblock_rows in (2, 1, 4):
                if candidate_tile % candidate_vblock_rows != 0:
                    continue
                threads = (candidate_tile // vec_width) * (
                    candidate_tile // candidate_vblock_rows
                )
                reg_budget_per_thread = max_registers // max(1, threads)
                if threads > max_threads or threads < 64 or reg_budget_per_thread < 32:
                    continue
                score = (
                    candidate_tile,
                    -abs(threads - 128),
                    int(candidate_vblock_rows == 2),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    vec_tile = candidate_tile
                    vblock_rows = candidate_vblock_rows

        return {
            "tiled_tile": tiled_tile,
            "vec_tile": vec_tile,
            "vec_width": vec_width,
            "vblock_rows": vblock_rows,
        }

    def _build_compile_options(self, module_name: str) -> list[str]:
        options = [*self.nvcc_options, "-I", str(self.kernel_dir)]
        if module_name == "fp32/matmul.cu":
            options.extend(
                [
                    f"-DMM_TILED_TILE={self._matmul_tuning['tiled_tile']}",
                    f"-DMM_VEC_TILE={self._matmul_tuning['vec_tile']}",
                    f"-DMM_VEC_WIDTH={self._matmul_tuning['vec_width']}",
                    f"-DMM_VBLOCK_ROWS={self._matmul_tuning['vblock_rows']}",
                ]
            )
        return options

    def compile_all(self) -> None:
        cu_sources = sorted(self.kernel_dir.rglob("*.cu"))
        for cu_file in cu_sources:
            module_name = cu_file.relative_to(self.kernel_dir).as_posix()
            self.compile_kernel_file(module_name)

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
            grid, launch_block = build_matmul_launch_config(m, n, block=block)
        elif kernel == "tiled":
            grid, launch_block = build_matmul_tiled_launch_config(
                m, n, tile_size=self._matmul_tuning["tiled_tile"]
            )
        else:
            expected_block = (
                self._matmul_tuning["vec_tile"] // self._matmul_tuning["vec_width"],
                self._matmul_tuning["vec_tile"] // self._matmul_tuning["vblock_rows"],
                1,
            )
            if block not in ((16, 16, 1), (4, 16, 1), expected_block):
                raise ValueError(
                    f"Vectorized matmul uses auto-tuned block={expected_block}. "
                    "Use default block or explicitly pass the tuned value."
                )
            grid, launch_block = build_matmul_vectorized_launch_config(
                m,
                n,
                tile_size=self._matmul_tuning["vec_tile"],
                vec_width=self._matmul_tuning["vec_width"],
                vblock_rows=self._matmul_tuning["vblock_rows"],
            )

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
        grid, block = build_elementwise_launch_config(numel, block=block)
        function(
            x_gpu,
            y_gpu,
            np.int32(numel),
            block=block,
            grid=grid,
            stream=stream,
        )
