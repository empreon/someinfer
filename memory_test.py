import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

print("PyCUDA initialized successfully!")
print("Active Device:", cuda.Device(0).name())

# 1. Create a matrix of random data on CPU (Host)
print("\n--- Data Transfer Test ---")
host_data = np.random.randn(1024, 1024).astype(np.float32)

# 2. Allocate GPU (Device) memory matching the data size
device_data = cuda.mem_alloc(host_data.nbytes)

# 3. Copy CPU data to GPU (Host to Device)
cuda.memcpy_htod(device_data, host_data)
print("1. Data copied successfully from RAM to VRAM (GPU).")

# 4. Create an empty CPU buffer to retrieve data from GPU
host_data_returned = np.empty_like(host_data)

# 5. Copy GPU data back to CPU (Device to Host)
cuda.memcpy_dtoh(host_data_returned, device_data)
print("2. Data copied successfully from VRAM (GPU) back to RAM.")

# 6. Compare the sent and received data
if np.allclose(host_data, host_data_returned):
    print("\nRESULT: SUCCESS! Data matches bit-for-bit.")
else:
    print("\nRESULT: ERROR! Data loss or corruption detected during transfer.")
