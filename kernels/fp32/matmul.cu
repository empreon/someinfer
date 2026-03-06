#include "common.cuh"

__device__ __forceinline__ void matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float value = 0.0f;
    for (int idx = 0; idx < K; ++idx) value += A[ROW_MAJOR_INDEX(row, idx, K)] * B[ROW_MAJOR_INDEX(idx, col, N)];
    C[ROW_MAJOR_INDEX(row, col, N)] = value;
  }
}

__device__ __forceinline__ void matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N) {
  constexpr int kTileSize = MM_VEC_TILE;
  if (blockDim.x != kTileSize || blockDim.y != kTileSize) {
    return;
  }

  extern __shared__ float shared_mem[];
  float (*tile_a)[kTileSize] = reinterpret_cast<float (*)[kTileSize]>(shared_mem);
  float (*tile_b)[kTileSize] = reinterpret_cast<float (*)[kTileSize]>(shared_mem + kTileSize * kTileSize);

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

    for (int k_local = 0; k_local < kTileSize; ++k_local) value += tile_a[local_row][k_local] * tile_b[k_local][local_col];

    __syncthreads();
  }

  if (row < M && col < N) C[ROW_MAJOR_INDEX(row, col, N)] = value;
}

__device__ __forceinline__ void matmul_vectorized_2d(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
  constexpr int kVecTileSize = MM_VEC_TILE;
  constexpr int kVecWidth = MM_VEC_WIDTH;
  constexpr int kVBlockRows = MM_VBLOCK_ROWS;

  extern __shared__ float shared_mem[];
  float (*tile_a)[kVecTileSize + 1] = reinterpret_cast<float (*)[kVecTileSize + 1]>(shared_mem);
  float (*tile_b)[kVecTileSize + 1] = reinterpret_cast<float (*)[kVecTileSize + 1]>(shared_mem + kVecTileSize * (kVecTileSize + 1));

  const int local_row_block = threadIdx.y;
  const int local_col_vec = threadIdx.x;
  const int row_base = blockIdx.y * kVecTileSize + local_row_block * kVBlockRows;
  const int col_base = blockIdx.x * kVecTileSize + local_col_vec * kVecWidth;

  const float* a_ptr = A + (row_base * K) + (local_col_vec * kVecWidth);
  const float* b_ptr = B + (local_row_block * kVBlockRows * N) + col_base;

  float4 acc[kVBlockRows];
  #pragma unroll
  for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) acc[row_offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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

__device__ __forceinline__ void matmul_pipelined_2d(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
  constexpr int kVecTileSize = MM_VEC_TILE;
  constexpr int kVecWidth = MM_VEC_WIDTH;
  constexpr int kVBlockRows = MM_VBLOCK_ROWS;

  extern __shared__ float shared_mem[]; // Bypass 65KB shared memory limit for larger tile sizes
  float (*tile_a)[kVecTileSize][kVecTileSize + 1] = reinterpret_cast<float (*)[kVecTileSize][kVecTileSize + 1]>(shared_mem);
  float (*tile_b)[kVecTileSize][kVecTileSize + 1] = reinterpret_cast<float (*)[kVecTileSize][kVecTileSize + 1]>(shared_mem + 2 * kVecTileSize * (kVecTileSize + 1));

  const int local_row_block = threadIdx.y;
  const int local_col_vec = threadIdx.x;
  const int row_base = blockIdx.y * kVecTileSize + local_row_block * kVBlockRows;
  const int col_base = blockIdx.x * kVecTileSize + local_col_vec * kVecWidth;

  const float* a_ptr = A + (row_base * K) + (local_col_vec * kVecWidth);
  const float* b_ptr = B + (local_row_block * kVBlockRows * N) + col_base;

  float4 acc[kVBlockRows];
  #pragma unroll
  for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) acc[row_offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int tile_count = CEIL_DIV(K, kVecTileSize);

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
  for (int tile_idx = 1; tile_idx < tile_count; ++tile_idx) {
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

  const int last_buffer = (tile_count - 1) % 2;
  #pragma unroll
  for (int k_local = 0; k_local < kVecTileSize; ++k_local) {
    const int b_col_start = local_col_vec * kVecWidth;
    const float4 b_values = make_float4(
        tile_b[last_buffer][k_local][b_col_start + 0],
        tile_b[last_buffer][k_local][b_col_start + 1],
        tile_b[last_buffer][k_local][b_col_start + 2],
        tile_b[last_buffer][k_local][b_col_start + 3]);

    #pragma unroll
    for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
      const int tile_row = local_row_block * kVBlockRows + row_offset;
      const float a_scalar = tile_a[last_buffer][tile_row][k_local];
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

__device__ __forceinline__ void matmul_asymetric_pipelined_2d(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
  constexpr int kVecTileSizeM = MM_VEC_TILE_M;
  constexpr int kVecTileSizeN = MM_VEC_TILE_N;
  constexpr int kVecTileSizeK = MM_VEC_TILE_K;
  constexpr int kVecWidth = MM_VEC_WIDTH; 
  constexpr int kVBlockRows = MM_VBLOCK_ROWS;

  constexpr int kThreadsX = kVecTileSizeN / kVecWidth;
  constexpr int kThreadsY = kVecTileSizeM / kVBlockRows;
  constexpr int kTotalThreads = kThreadsX * kThreadsY;

  constexpr int kVecsPerRowA = kVecTileSizeK / kVecWidth;
  constexpr int kStepsA = (kVecTileSizeM * kVecsPerRowA) / kTotalThreads;
  constexpr int kStrideA = kTotalThreads / kVecsPerRowA;

  constexpr int kVecsPerRowB = kVecTileSizeN / kVecWidth;
  constexpr int kStepsB = (kVecTileSizeK * kVecsPerRowB) / kTotalThreads;
  constexpr int kStrideB = kTotalThreads / kVecsPerRowB;

  extern __shared__ float shared_mem[];
  float (*tile_a)[kVecTileSizeM][kVecTileSizeK + 1] = reinterpret_cast<float (*)[kVecTileSizeM][kVecTileSizeK + 1]>(shared_mem);
  float (*tile_b)[kVecTileSizeK][kVecTileSizeN + 1] = reinterpret_cast<float (*)[kVecTileSizeK][kVecTileSizeN + 1]>(shared_mem + 2 * kVecTileSizeM * (kVecTileSizeK + 1));

  int local_row_block = threadIdx.y;
  int local_col_vec = threadIdx.x;
  int row_base = blockIdx.y * kVecTileSizeM + local_row_block * kVBlockRows;
  int col_base = blockIdx.x * kVecTileSizeN + local_col_vec * kVecWidth;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  int load_a_row = tid / (kVecTileSizeK / kVecWidth);
  int load_a_col = (tid % (kVecTileSizeK / kVecWidth)) * kVecWidth;
  const float* a_load_ptr = A + (blockIdx.y * kVecTileSizeM + load_a_row) * K + load_a_col;

  int load_b_row = tid / (kVecTileSizeN / kVecWidth);
  int load_b_col = (tid % (kVecTileSizeN / kVecWidth)) * kVecWidth;
  const float* b_load_ptr = B + (load_b_row * N) + (blockIdx.x * kVecTileSizeN + load_b_col);

  float4 acc[kVBlockRows];
  #pragma unroll
  for (int i = 0; i < kVBlockRows; ++i) acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int tile_count = CEIL_DIV(K, kVecTileSizeK);

  #pragma unroll
  for (int step = 0; step < kStepsA; ++step) {
    float4 a_vec = *reinterpret_cast<const float4*>(a_load_ptr + (step * kStrideA) * K);
    tile_a[0][load_a_row + (step * kStrideA)][load_a_col + 0] = a_vec.x;
    tile_a[0][load_a_row + (step * kStrideA)][load_a_col + 1] = a_vec.y;
    tile_a[0][load_a_row + (step * kStrideA)][load_a_col + 2] = a_vec.z;
    tile_a[0][load_a_row + (step * kStrideA)][load_a_col + 3] = a_vec.w;
  }

  #pragma unroll
  for (int step = 0; step < kStepsB; ++step) {
    float4 b_vec = *reinterpret_cast<const float4*>(b_load_ptr + (step * kStrideB) * N);
    tile_b[0][load_b_row + (step * kStrideB)][load_b_col + 0] = b_vec.x;
    tile_b[0][load_b_row + (step * kStrideB)][load_b_col + 1] = b_vec.y;
    tile_b[0][load_b_row + (step * kStrideB)][load_b_col + 2] = b_vec.z;
    tile_b[0][load_b_row + (step * kStrideB)][load_b_col + 3] = b_vec.w;
  }
  __syncthreads();

  a_load_ptr += kVecTileSizeK;
  b_load_ptr += kVecTileSizeK * N;

  #pragma unroll
  for (int tile_idx = 1; tile_idx < tile_count; ++tile_idx) {
    const int current_buffer = tile_idx % 2;
    const int next_buffer = (tile_idx + 1) % 2;

    float4 a_vec_next[kStepsA];
    float4 b_vec_next[kStepsB];
    #pragma unroll
    for (int step = 0; step < kStepsA; ++step) {
      a_vec_next[step] = *reinterpret_cast<const float4*>(a_load_ptr + (step * kStrideA) * K);
    }

    #pragma unroll
    for (int step = 0; step < kStepsB; ++step) {
      b_vec_next[step] = *reinterpret_cast<const float4*>(b_load_ptr + (step * kStrideB) * N);
    }

    #pragma unroll
    for (int k_local = 0; k_local < kVecTileSizeK; ++k_local) {
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
    for (int step = 0; step < kStepsA; ++step) {
      tile_a[next_buffer][load_a_row + (step * kStrideA)][load_a_col + 0] = a_vec_next[step].x;
      tile_a[next_buffer][load_a_row + (step * kStrideA)][load_a_col + 1] = a_vec_next[step].y;
      tile_a[next_buffer][load_a_row + (step * kStrideA)][load_a_col + 2] = a_vec_next[step].z;
      tile_a[next_buffer][load_a_row + (step * kStrideA)][load_a_col + 3] = a_vec_next[step].w;
    }

    #pragma unroll
    for (int step = 0; step < kStepsB; ++step) {
      tile_b[next_buffer][load_b_row + (step * kStrideB)][load_b_col + 0] = b_vec_next[step].x;
      tile_b[next_buffer][load_b_row + (step * kStrideB)][load_b_col + 1] = b_vec_next[step].y;
      tile_b[next_buffer][load_b_row + (step * kStrideB)][load_b_col + 2] = b_vec_next[step].z;
      tile_b[next_buffer][load_b_row + (step * kStrideB)][load_b_col + 3] = b_vec_next[step].w;
    }
    __syncthreads();

    a_load_ptr += kVecTileSizeK;
    b_load_ptr += kVecTileSizeK * N;
  }

  const int last_buffer = (tile_count - 1) % 2;
  #pragma unroll
  for (int k_local = 0; k_local < kVecTileSizeK; ++k_local) {
    const int b_col_start = local_col_vec * kVecWidth;
    const float4 b_values = make_float4(
        tile_b[last_buffer][k_local][b_col_start + 0],
        tile_b[last_buffer][k_local][b_col_start + 1],
        tile_b[last_buffer][k_local][b_col_start + 2],
        tile_b[last_buffer][k_local][b_col_start + 3]);

    #pragma unroll
    for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
      const int tile_row = local_row_block * kVBlockRows + row_offset;
      const float a_scalar = tile_a[last_buffer][tile_row][k_local];
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

__device__ __forceinline__ void matmul_register_pipelined_2d(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
  constexpr int kVecTileSizeM = MM_VEC_TILE_M;
  constexpr int kVecTileSizeN = MM_VEC_TILE_N;
  constexpr int kVecTileSizeK = MM_VEC_TILE_K;
  constexpr int kVecWidth = MM_VEC_WIDTH; 
  constexpr int kVBlockRows = MM_VBLOCK_ROWS;

  constexpr int kThreadsX = kVecTileSizeN / (kVecWidth * 2); // Each thread loads 2 vectors from B
  constexpr int kThreadsY = kVecTileSizeM / kVBlockRows;
  constexpr int kTotalThreads = kThreadsX * kThreadsY;

  constexpr int kVecsPerRowA = kVecTileSizeK / kVecWidth;
  constexpr int kStepsA = (kVecTileSizeM * kVecsPerRowA) / kTotalThreads;
  constexpr int kStrideA = kTotalThreads / kVecsPerRowA;

  constexpr int kVecsPerRowB = kVecTileSizeN / kVecWidth;
  constexpr int kStepsB = (kVecTileSizeK * kVecsPerRowB) / kTotalThreads;
  constexpr int kStrideB = kTotalThreads / kVecsPerRowB;

  extern __shared__ float shared_mem[];
  float (*tile_a)[kVecTileSizeM][kVecTileSizeK + 1] = reinterpret_cast<float (*)[kVecTileSizeM][kVecTileSizeK + 1]>(shared_mem);
  float (*tile_b)[kVecTileSizeK][kVecTileSizeN + 1] = reinterpret_cast<float (*)[kVecTileSizeK][kVecTileSizeN + 1]>(shared_mem + 2 * kVecTileSizeM * (kVecTileSizeK + 1));

  int local_row_block = threadIdx.y;
  int local_col_vec = threadIdx.x;
  int row_base = blockIdx.y * kVecTileSizeM + local_row_block * kVBlockRows;
  int col_base = blockIdx.x * kVecTileSizeN + local_col_vec * (kVecWidth * 2);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  int load_a_row = tid / (kVecTileSizeK / kVecWidth);
  int load_a_col = (tid % (kVecTileSizeK / kVecWidth)) * kVecWidth;
  const float* a_load_ptr = A + (blockIdx.y * kVecTileSizeM + load_a_row) * K + load_a_col;

  int load_b_row = tid / (kVecTileSizeN / kVecWidth);
  int load_b_col = (tid % (kVecTileSizeN / kVecWidth)) * kVecWidth;
  const float* b_load_ptr = B + (load_b_row * N) + (blockIdx.x * kVecTileSizeN + load_b_col);

  float4 acc[kVBlockRows][2];
  #pragma unroll
  for (int i = 0; i < kVBlockRows; ++i) {
    acc[i][0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    acc[i][1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  const int tile_count = CEIL_DIV(K, kVecTileSizeK);

  #pragma unroll
  for (int step = 0; step < kStepsA; ++step) {
    float4 a_vec = *reinterpret_cast<const float4*>(a_load_ptr + (step * kStrideA) * K);
    tile_a[0][load_a_row + (step * kStrideA)][load_a_col + 0] = a_vec.x;
    tile_a[0][load_a_row + (step * kStrideA)][load_a_col + 1] = a_vec.y;
    tile_a[0][load_a_row + (step * kStrideA)][load_a_col + 2] = a_vec.z;
    tile_a[0][load_a_row + (step * kStrideA)][load_a_col + 3] = a_vec.w;
  }

  #pragma unroll
  for (int step = 0; step < kStepsB; ++step) {
    float4 b_vec = *reinterpret_cast<const float4*>(b_load_ptr + (step * kStrideB) * N);
    tile_b[0][load_b_row + (step * kStrideB)][load_b_col + 0] = b_vec.x;
    tile_b[0][load_b_row + (step * kStrideB)][load_b_col + 1] = b_vec.y;
    tile_b[0][load_b_row + (step * kStrideB)][load_b_col + 2] = b_vec.z;
    tile_b[0][load_b_row + (step * kStrideB)][load_b_col + 3] = b_vec.w;
  }
  __syncthreads();

  a_load_ptr += kVecTileSizeK;
  b_load_ptr += kVecTileSizeK * N;

  #pragma unroll
  for (int tile_idx = 1; tile_idx < tile_count; ++tile_idx) {
    const int current_buffer = tile_idx % 2;
    const int next_buffer = (tile_idx + 1) % 2;

    float4 a_vec_next[kStepsA];
    #pragma unroll
    for (int step = 0; step < kStepsA; ++step) {
      a_vec_next[step] = *reinterpret_cast<const float4*>(a_load_ptr + (step * kStrideA) * K);
    }

    float4 b_vec_next[kStepsB];
    #pragma unroll
    for (int step = 0; step < kStepsB; ++step) {
      b_vec_next[step] = *reinterpret_cast<const float4*>(b_load_ptr + (step * kStrideB) * N);
    }

    #pragma unroll
    for (int k_local = 0; k_local < kVecTileSizeK; ++k_local) {
      const int b_col_start = local_col_vec * (kVecWidth * 2);
      const float4 b_values1 = make_float4(
          tile_b[current_buffer][k_local][b_col_start + 0],
          tile_b[current_buffer][k_local][b_col_start + 1],
          tile_b[current_buffer][k_local][b_col_start + 2],
          tile_b[current_buffer][k_local][b_col_start + 3]);
      const float4 b_values2 = make_float4(
          tile_b[current_buffer][k_local][b_col_start + 4],
          tile_b[current_buffer][k_local][b_col_start + 5],
          tile_b[current_buffer][k_local][b_col_start + 6],
          tile_b[current_buffer][k_local][b_col_start + 7]);
      
      float a_flags[kVBlockRows];
      #pragma unroll
      for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
        a_flags[row_offset] = tile_a[current_buffer][local_row_block * kVBlockRows + row_offset][k_local];
      }

      #pragma unroll
      for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
        acc[row_offset][0].x += a_flags[row_offset] * b_values1.x;
        acc[row_offset][0].y += a_flags[row_offset] * b_values1.y;
        acc[row_offset][0].z += a_flags[row_offset] * b_values1.z;
        acc[row_offset][0].w += a_flags[row_offset] * b_values1.w;
        acc[row_offset][1].x += a_flags[row_offset] * b_values2.x;
        acc[row_offset][1].y += a_flags[row_offset] * b_values2.y;
        acc[row_offset][1].z += a_flags[row_offset] * b_values2.z;
        acc[row_offset][1].w += a_flags[row_offset] * b_values2.w;
      }
    }

    #pragma unroll
    for (int step = 0; step < kStepsA; ++step) {
      tile_a[next_buffer][load_a_row + (step * kStrideA)][load_a_col + 0] = a_vec_next[step].x;
      tile_a[next_buffer][load_a_row + (step * kStrideA)][load_a_col + 1] = a_vec_next[step].y;
      tile_a[next_buffer][load_a_row + (step * kStrideA)][load_a_col + 2] = a_vec_next[step].z;
      tile_a[next_buffer][load_a_row + (step * kStrideA)][load_a_col + 3] = a_vec_next[step].w;
    }

    #pragma unroll
    for (int step = 0; step < kStepsB; ++step) {
      tile_b[next_buffer][load_b_row + (step * kStrideB)][load_b_col + 0] = b_vec_next[step].x;
      tile_b[next_buffer][load_b_row + (step * kStrideB)][load_b_col + 1] = b_vec_next[step].y;
      tile_b[next_buffer][load_b_row + (step * kStrideB)][load_b_col + 2] = b_vec_next[step].z;
      tile_b[next_buffer][load_b_row + (step * kStrideB)][load_b_col + 3] = b_vec_next[step].w;
    }
    __syncthreads();

    a_load_ptr += kVecTileSizeK;
    b_load_ptr += kVecTileSizeK * N;
  }

  const int last_buffer = (tile_count - 1) % 2;
  #pragma unroll
  for (int k_local = 0; k_local < kVecTileSizeK; ++k_local) {
    const int b_col_start = local_col_vec * (kVecWidth * 2);
    const float4 b_values1 = make_float4(
        tile_b[last_buffer][k_local][b_col_start + 0],
        tile_b[last_buffer][k_local][b_col_start + 1],
        tile_b[last_buffer][k_local][b_col_start + 2],
        tile_b[last_buffer][k_local][b_col_start + 3]);
    const float4 b_values2 = make_float4(
        tile_b[last_buffer][k_local][b_col_start + 4],
        tile_b[last_buffer][k_local][b_col_start + 5],
        tile_b[last_buffer][k_local][b_col_start + 6],
        tile_b[last_buffer][k_local][b_col_start + 7]);

    float a_flags[kVBlockRows];
    #pragma unroll
    for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
      a_flags[row_offset] = tile_a[last_buffer][local_row_block * kVBlockRows + row_offset][k_local];
    }

    #pragma unroll
    for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
      acc[row_offset][0].x += a_flags[row_offset] * b_values1.x;
      acc[row_offset][0].y += a_flags[row_offset] * b_values1.y;
      acc[row_offset][0].z += a_flags[row_offset] * b_values1.z;
      acc[row_offset][0].w += a_flags[row_offset] * b_values1.w;
      acc[row_offset][1].x += a_flags[row_offset] * b_values2.x;
      acc[row_offset][1].y += a_flags[row_offset] * b_values2.y;
      acc[row_offset][1].z += a_flags[row_offset] * b_values2.z;
      acc[row_offset][1].w += a_flags[row_offset] * b_values2.w;
    }
  }

  const int c_col_start = col_base;
  float *c_ptr = &C[ROW_MAJOR_INDEX(row_base, c_col_start, N)];
  #pragma unroll
  for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
    *reinterpret_cast<float4*>(c_ptr + row_offset * N) = acc[row_offset][0];
    *reinterpret_cast<float4*>(c_ptr + row_offset * N + 4) = acc[row_offset][1];
  }
}

extern "C" __global__ void matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
}

extern "C" {
  __global__ void run_matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    matmul_naive(A, B, C, M, K, N);
  }
  __global__ void run_matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N) {
    matmul_tiled(A, B, C, M, K, N);
  }
  __global__ void run_matmul_vectorized_2d(const float* A, const float* B, float* C, int M, int K, int N) {
    matmul_vectorized_2d(A, B, C, M, K, N);
  }
  __global__ void run_matmul_pipelined_2d(const float* A, const float* B, float* C, int M, int K, int N) {
    matmul_pipelined_2d(A, B, C, M, K, N);
  }
  __global__ void run_matmul_asymetric_pipelined_2d(const float* A, const float* B, float* C, int M, int K, int N) {
    matmul_asymetric_pipelined_2d(A, B, C, M, K, N);
  }
  __global__ void run_matmul_register_pipelined_2d(const float* A, const float* B, float* C, int M, int K, int N) {
    matmul_register_pipelined_2d(A, B, C, M, K, N);
  }
}
