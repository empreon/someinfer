import pycuda.driver as cuda
import pycuda.autoinit

def recommend_matmul_tuning(attributes):
    max_threads_per_block = attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
    max_registers_per_block = attributes[cuda.device_attribute.MAX_REGISTERS_PER_BLOCK]
    max_shared_memory_per_block = attributes[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]

    # RTX 4050 Laptop profile shortcut.
    if (
        max_threads_per_block >= 1024
        and max_registers_per_block >= 65536
        and max_shared_memory_per_block >= 48 * 1024
    ):
        return {
            "tiled_tile": 32,
            "vec_tile": 32,
            "vec_width": 4,
            "vblock_rows": 2,
            "vec_block_x": 8,
            "vec_block_y": 16,
        }

    tiled_tile = 16
    for candidate_tile in (32, 16, 8):
        threads = candidate_tile * candidate_tile
        shared_bytes = 2 * candidate_tile * candidate_tile * 4
        reg_budget_per_thread = max_registers_per_block // max(1, threads)
        if (
            threads <= max_threads_per_block
            and shared_bytes <= max_shared_memory_per_block
            and reg_budget_per_thread >= 32
        ):
            tiled_tile = candidate_tile
            break

    vec_width = 4
    vec_tile = 16
    vblock_rows = 1
    best_score = None
    for candidate_tile in (32, 16):
        shared_bytes = 2 * candidate_tile * candidate_tile * 4
        if shared_bytes > max_shared_memory_per_block or candidate_tile % vec_width != 0:
            continue
        for candidate_vblock_rows in (2, 1, 4):
            if candidate_tile % candidate_vblock_rows != 0:
                continue
            threads = (candidate_tile // vec_width) * (candidate_tile // candidate_vblock_rows)
            reg_budget_per_thread = max_registers_per_block // max(1, threads)
            if threads > max_threads_per_block or threads < 64 or reg_budget_per_thread < 32:
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
        "vec_block_x": vec_tile // vec_width,
        "vec_block_y": vec_tile // vblock_rows,
    }

def print_gpu_limits():
    device = cuda.Device(0)
    attributes = device.get_attributes()
    
    print(f"--- {device.name()} Hardware Limits ---")
    
    # SM (Streaming Multiprocessor) count
    sm_count = attributes[cuda.device_attribute.MULTIPROCESSOR_COUNT]
    print(f"SM Count (Multiprocessor Count): {sm_count}")
    
    # Block and thread limits
    max_threads_per_block = attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
    print(f"Maximum Threads per Block: {max_threads_per_block}")
    
    # Register limits (most critical)
    max_registers_per_block = attributes[cuda.device_attribute.MAX_REGISTERS_PER_BLOCK]
    print(f"Maximum Registers per Block: {max_registers_per_block}")
    
    # Shared memory limits
    max_shared_memory_per_block = attributes[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
    print(f"Maximum Shared Memory per Block: {max_shared_memory_per_block} bytes ({max_shared_memory_per_block/1024:.2f} KB)")

    tuning = recommend_matmul_tuning(attributes)
    print("--- Recommended FP32 Matmul Tuning ---")
    print(f"Tiled Tile: {tuning['tiled_tile']}")
    print(f"Vectorized Tile: {tuning['vec_tile']}")
    print(f"Vector Width: {tuning['vec_width']}")
    print(f"VBlock Rows: {tuning['vblock_rows']}")
    print(f"Vectorized Block: ({tuning['vec_block_x']}, {tuning['vec_block_y']}, 1)")

if __name__ == "__main__":
    print_gpu_limits()
