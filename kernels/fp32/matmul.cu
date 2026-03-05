#include "common.cuh"

__device__ __forceinline__ void matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float value = 0.0f;
    for (int idx = 0; idx < K; ++idx) {
      value += A[ROW_MAJOR_INDEX(row, idx, K)] *
               B[ROW_MAJOR_INDEX(idx, col, N)];
    }
    C[ROW_MAJOR_INDEX(row, col, N)] = value;
  }
}

__device__ __forceinline__ void matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N) {
  constexpr int kTileSize = MM_TILED_TILE;
  if (blockDim.x != kTileSize || blockDim.y != kTileSize) {
    return;
  }

  __shared__ float tile_a[kTileSize][kTileSize];
  __shared__ float tile_b[kTileSize][kTileSize];

  const int local_row = threadIdx.y;
  const int local_col = threadIdx.x;
  const int row = blockIdx.y * kTileSize + local_row;
  const int col = blockIdx.x * kTileSize + local_col;

  float value = 0.0f;
  const int tile_count = CEIL_DIV(K, kTileSize);

  for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const int k_a = tile_idx * kTileSize + local_col;
    const int k_b = tile_idx * kTileSize + local_row;

    tile_a[local_row][local_col] =
        (row < M && k_a < K) ? A[ROW_MAJOR_INDEX(row, k_a, K)] : 0.0f;
    tile_b[local_row][local_col] =
        (k_b < K && col < N) ? B[ROW_MAJOR_INDEX(k_b, col, N)] : 0.0f;

    __syncthreads();

    for (int k_local = 0; k_local < kTileSize; ++k_local) {
      value += tile_a[local_row][k_local] * tile_b[k_local][local_col];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[ROW_MAJOR_INDEX(row, col, N)] = value;
  }
}

__device__ __forceinline__ void matmul_vectorized(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
  constexpr int kVecTileSize = MM_VEC_TILE;
  constexpr int kVecWidth = MM_VEC_WIDTH;
  constexpr int kVBlockRows = MM_VBLOCK_ROWS;
  constexpr int kExpectedBlockX = kVecTileSize / kVecWidth;
  constexpr int kExpectedBlockY = kVecTileSize / kVBlockRows;

  if (blockDim.x != kExpectedBlockX || blockDim.y != kExpectedBlockY) {
    return;
  }

  __shared__ float tile_a[kVecTileSize][kVecTileSize];
  __shared__ float tile_b[kVecTileSize][kVecTileSize];

  const int local_row_block = threadIdx.y;
  const int local_col_vec = threadIdx.x;
  const int row_base = blockIdx.y * kVecTileSize + local_row_block * kVBlockRows;
  const int col_base = blockIdx.x * kVecTileSize + local_col_vec * kVecWidth;

  float4 acc[kVBlockRows];
  #pragma unroll
  for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
    acc[row_offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  const int tile_count = CEIL_DIV(K, kVecTileSize);

  for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const int k_col_start = tile_idx * kVecTileSize + local_col_vec * kVecWidth;

    #pragma unroll
    for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
      const int tile_row = local_row_block * kVBlockRows + row_offset;
      const int row = row_base + row_offset;

      float4 a_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (row < M) {
        const float* a_row = A + ROW_MAJOR_INDEX(row, 0, K);
        a_vec = *reinterpret_cast<const float4*>(a_row + k_col_start);
      }
      tile_a[tile_row][local_col_vec * kVecWidth + 0] = a_vec.x;
      tile_a[tile_row][local_col_vec * kVecWidth + 1] = a_vec.y;
      tile_a[tile_row][local_col_vec * kVecWidth + 2] = a_vec.z;
      tile_a[tile_row][local_col_vec * kVecWidth + 3] = a_vec.w;

      const int k_b = tile_idx * kVecTileSize + tile_row;
      const float* b_row = B + ROW_MAJOR_INDEX(k_b, 0, N);
      const float4 b_vec = *reinterpret_cast<const float4*>(b_row + col_base);
      tile_b[tile_row][local_col_vec * kVecWidth + 0] = b_vec.x;
      tile_b[tile_row][local_col_vec * kVecWidth + 1] = b_vec.y;
      tile_b[tile_row][local_col_vec * kVecWidth + 2] = b_vec.z;
      tile_b[tile_row][local_col_vec * kVecWidth + 3] = b_vec.w;
    }

    __syncthreads();

    #pragma unroll
    for (int k_local = 0; k_local < kVecTileSize; ++k_local) {
      const int b_col_start = local_col_vec * kVecWidth;
      const float4 b_values = make_float4(
          tile_b[k_local][b_col_start + 0],
          tile_b[k_local][b_col_start + 1],
          tile_b[k_local][b_col_start + 2],
          tile_b[k_local][b_col_start + 3]);

      #pragma unroll
      for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
        const int row = row_base + row_offset;
        if (row >= M) {
          continue;
        }
        const int tile_row = local_row_block * kVBlockRows + row_offset;
        const float a_scalar = tile_a[tile_row][k_local];

        acc[row_offset].x += a_scalar * b_values.x;
        acc[row_offset].y += a_scalar * b_values.y;
        acc[row_offset].z += a_scalar * b_values.z;
        acc[row_offset].w += a_scalar * b_values.w;
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
    const int row = row_base + row_offset;
    if (row >= M) {
      continue;
    }
    if (col_base + 0 < N) C[ROW_MAJOR_INDEX(row, col_base + 0, N)] = acc[row_offset].x;
    if (col_base + 1 < N) C[ROW_MAJOR_INDEX(row, col_base + 1, N)] = acc[row_offset].y;
    if (col_base + 2 < N) C[ROW_MAJOR_INDEX(row, col_base + 2, N)] = acc[row_offset].z;
    if (col_base + 3 < N) C[ROW_MAJOR_INDEX(row, col_base + 3, N)] = acc[row_offset].w;
  }
}

__device__ __forceinline__ void matmul_vectorized_2d_tiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
  constexpr int kVecTileSize = MM_VEC_TILE;
  constexpr int kVecWidth = MM_VEC_WIDTH;
  constexpr int kVBlockRows = MM_VBLOCK_ROWS;

  __shared__ float tile_a[kVecTileSize][kVecTileSize + 1];
  __shared__ float tile_b[kVecTileSize][kVecTileSize + 1];

  const int local_row_block = threadIdx.y;
  const int local_col_vec = threadIdx.x;
  const int row_base = blockIdx.y * kVecTileSize + local_row_block * kVBlockRows;
  const int col_base = blockIdx.x * kVecTileSize + local_col_vec * kVecWidth;

  const float* a_ptr = A + (row_base * K) + (local_col_vec * kVecWidth);
  const float* b_ptr = B + (local_row_block * kVBlockRows * N) + col_base;

  float4 acc[kVBlockRows];
  #pragma unroll
  for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
    acc[row_offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  const int tile_count = CEIL_DIV(K, kVecTileSize);

  for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const float* current_a = a_ptr;
    const float* current_b = b_ptr;

    #pragma unroll
    for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
      const int tile_row = local_row_block * kVBlockRows + row_offset;

      float4 a_vec = *reinterpret_cast<const float4*>(current_a);
      float4 b_vec = *reinterpret_cast<const float4*>(current_b);

      tile_a[tile_row][local_col_vec * kVecWidth + 0] = a_vec.x;
      tile_a[tile_row][local_col_vec * kVecWidth + 1] = a_vec.y;
      tile_a[tile_row][local_col_vec * kVecWidth + 2] = a_vec.z;
      tile_a[tile_row][local_col_vec * kVecWidth + 3] = a_vec.w;

      tile_b[tile_row][local_col_vec * kVecWidth + 0] = b_vec.x;
      tile_b[tile_row][local_col_vec * kVecWidth + 1] = b_vec.y;
      tile_b[tile_row][local_col_vec * kVecWidth + 2] = b_vec.z;
      tile_b[tile_row][local_col_vec * kVecWidth + 3] = b_vec.w;

      current_a += K;
      current_b += N;
    }

    __syncthreads();

    #pragma unroll
    for (int k_local = 0; k_local < kVecTileSize; ++k_local) {
      const int b_col_start = local_col_vec * kVecWidth;
      const float4 b_values = make_float4(
          tile_b[k_local][b_col_start + 0],
          tile_b[k_local][b_col_start + 1],
          tile_b[k_local][b_col_start + 2],
          tile_b[k_local][b_col_start + 3]);

      #pragma unroll
      for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
        const int tile_row = local_row_block * kVBlockRows + row_offset;
        const float a_scalar = tile_a[tile_row][k_local];
        acc[row_offset].x += a_scalar * b_values.x;
        acc[row_offset].y += a_scalar * b_values.y;
        acc[row_offset].z += a_scalar * b_values.z;
        acc[row_offset].w += a_scalar * b_values.w;
      }
    }
    __syncthreads();

    a_ptr += kVecTileSize;
    b_ptr += kVecTileSize * N;
  }

  const int c_col_start = col_base;
  #pragma unroll
  for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
    const int row = row_base + row_offset;
    if (c_col_start + 0 < N) C[ROW_MAJOR_INDEX(row, c_col_start + 0, N)] = acc[row_offset].x;
    if (c_col_start + 1 < N) C[ROW_MAJOR_INDEX(row, c_col_start + 1, N)] = acc[row_offset].y;
    if (c_col_start + 2 < N) C[ROW_MAJOR_INDEX(row, c_col_start + 2, N)] = acc[row_offset].z;
    if (c_col_start + 3 < N) C[ROW_MAJOR_INDEX(row, c_col_start + 3, N)] = acc[row_offset].w;
  }
}

__device__ __forceinline__ void matmul_pipelined_2d_tiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
  constexpr int kVecTileSize = MM_VEC_TILE;
  constexpr int kVecWidth = MM_VEC_WIDTH;
  constexpr int kVBlockRows = MM_VBLOCK_ROWS;

  // __shared__ float tile_a[2][kVecTileSize][kVecTileSize + 1];
  // __shared__ float tile_b[2][kVecTileSize][kVecTileSize + 1];

  extern __shared__ float shared_mem[]; // Bypass 65KB shared memory limit for larger tile sizes
  float (*tile_a)[kVecTileSize][kVecTileSize + 1] = reinterpret_cast<float (*)[kVecTileSize][kVecTileSize + 1]>(shared_mem);
  constexpr int a_size = 2 * kVecTileSize * (kVecTileSize + 1); // 8320 float
  float (*tile_b)[kVecTileSize][kVecTileSize + 1] = reinterpret_cast<float (*)[kVecTileSize][kVecTileSize + 1]>(shared_mem + a_size);

  const int local_row_block = threadIdx.y;
  const int local_col_vec = threadIdx.x;
  const int row_base = blockIdx.y * kVecTileSize + local_row_block * kVBlockRows;
  const int col_base = blockIdx.x * kVecTileSize + local_col_vec * kVecWidth;

  const float* a_ptr = A + (row_base * K) + (local_col_vec * kVecWidth);
  const float* b_ptr = B + (local_row_block * kVBlockRows * N) + col_base;

  float4 acc[kVBlockRows];
  #pragma unroll
  for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
    acc[row_offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  const int tile_count = CEIL_DIV(K, kVecTileSize);

  int tile_idx = 0; // Preload the first tile
  const float* current_a = a_ptr;
  const float* current_b = b_ptr;

  #pragma unroll
  for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
    const int tile_row = local_row_block * kVBlockRows + row_offset;

    float4 a_vec = *reinterpret_cast<const float4*>(current_a);
    float4 b_vec = *reinterpret_cast<const float4*>(current_b);

    tile_a[0][tile_row][local_col_vec * kVecWidth + 0] = a_vec.x;
    tile_a[0][tile_row][local_col_vec * kVecWidth + 1] = a_vec.y;
    tile_a[0][tile_row][local_col_vec * kVecWidth + 2] = a_vec.z;
    tile_a[0][tile_row][local_col_vec * kVecWidth + 3] = a_vec.w;

    tile_b[0][tile_row][local_col_vec * kVecWidth + 0] = b_vec.x;
    tile_b[0][tile_row][local_col_vec * kVecWidth + 1] = b_vec.y;
    tile_b[0][tile_row][local_col_vec * kVecWidth + 2] = b_vec.z;
    tile_b[0][tile_row][local_col_vec * kVecWidth + 3] = b_vec.w;

    current_a += K;
    current_b += N;
  }

  __syncthreads();

  a_ptr += kVecTileSize;
  b_ptr += kVecTileSize * N;

  #pragma unroll
  for (tile_idx = 1; tile_idx < tile_count; ++tile_idx) {
    const int current_buffer = tile_idx % 2;
    const int next_buffer = (tile_idx + 1) % 2;

    float4 a_vec_next = *reinterpret_cast<const float4*>(a_ptr);
    float4 b_vec_next = *reinterpret_cast<const float4*>(b_ptr);

    #pragma unroll
    for (int k_local = 0; k_local < kVecTileSize; ++k_local) {
      const int b_col_start = local_col_vec * kVecWidth;
      const float4 b_values = make_float4(
          tile_b[current_buffer][k_local][b_col_start + 0],
          tile_b[current_buffer][k_local][b_col_start + 1],
          tile_b[current_buffer][k_local][b_col_start + 2],
          tile_b[current_buffer][k_local][b_col_start + 3]);

      #pragma unroll
      for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
        const int tile_row = local_row_block * kVBlockRows + row_offset;
        const float a_scalar = tile_a[current_buffer][tile_row][k_local];
        acc[row_offset].x += a_scalar * b_values.x;
        acc[row_offset].y += a_scalar * b_values.y;
        acc[row_offset].z += a_scalar * b_values.z;
        acc[row_offset].w += a_scalar * b_values.w;
      }
    }

    #pragma unroll
    for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
      const int tile_row = local_row_block * kVBlockRows + row_offset;

      tile_a[next_buffer][tile_row][local_col_vec * kVecWidth + 0] = a_vec_next.x;
      tile_a[next_buffer][tile_row][local_col_vec * kVecWidth + 1] = a_vec_next.y;
      tile_a[next_buffer][tile_row][local_col_vec * kVecWidth + 2] = a_vec_next.z;
      tile_a[next_buffer][tile_row][local_col_vec * kVecWidth + 3] = a_vec_next.w;

      tile_b[next_buffer][tile_row][local_col_vec * kVecWidth + 0] = b_vec_next.x;
      tile_b[next_buffer][tile_row][local_col_vec * kVecWidth + 1] = b_vec_next.y;
      tile_b[next_buffer][tile_row][local_col_vec * kVecWidth + 2] = b_vec_next.z;
      tile_b[next_buffer][tile_row][local_col_vec * kVecWidth + 3] = b_vec_next.w;
    }
    __syncthreads();

    a_ptr += kVecTileSize;
    b_ptr += kVecTileSize * N;
  }

  #pragma unroll
  for (int k_local = 0; k_local < kVecTileSize; ++k_local) {
    const int b_col_start = local_col_vec * kVecWidth;
    const float4 b_values = make_float4(
        tile_b[tile_idx % 2][k_local][b_col_start + 0],
        tile_b[tile_idx % 2][k_local][b_col_start + 1],
        tile_b[tile_idx % 2][k_local][b_col_start + 2],
        tile_b[tile_idx % 2][k_local][b_col_start + 3]);

    #pragma unroll
    for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
      const int tile_row = local_row_block * kVBlockRows + row_offset;
      const float a_scalar = tile_a[tile_idx % 2][tile_row][k_local];
      acc[row_offset].x += a_scalar * b_values.x;
      acc[row_offset].y += a_scalar * b_values.y;
      acc[row_offset].z += a_scalar * b_values.z;
      acc[row_offset].w += a_scalar * b_values.w;
    }
  }

  const int c_col_start = col_base;
  #pragma unroll
  for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
    const int row = row_base + row_offset;
    if (c_col_start + 0 < N) C[ROW_MAJOR_INDEX(row, c_col_start + 0, N)] = acc[row_offset].x;
    if (c_col_start + 1 < N) C[ROW_MAJOR_INDEX(row, c_col_start + 1, N)] = acc[row_offset].y;
    if (c_col_start + 2 < N) C[ROW_MAJOR_INDEX(row, c_col_start + 2, N)] = acc[row_offset].z;
    if (c_col_start + 3 < N) C[ROW_MAJOR_INDEX(row, c_col_start + 3, N)] = acc[row_offset].w;
  }
}


extern "C" __global__ void matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
  // matmul_naive(A, B, C, M, K, N);
  // matmul_tiled(A, B, C, M, K, N);
  //matmul_vectorized(A, B, C, M, K, N);
  //matmul_vectorized_2d_tiling(A, B, C, M, K, N);
  matmul_pipelined_2d_tiling(A, B, C, M, K, N);
}
