#include <cstdint>

#include "utils.cuh"

constexpr int N_ARITHMETIC_INSTRUCTIONS = 900;
constexpr int N_MEMORY_INSTRUCTIONS = 60;

template <typename T>
__device__ T arithmetic_throughput_bound(T value, const T zero) {
#pragma unroll
  for (int i{0}; i < (N_ARITHMETIC_INSTRUCTIONS / 2); ++i) {
    // Each of these instructions sets value to zero
    value *= zero;
    value &= zero;
  }

  return value;
}

template <typename T>
__device__ T cache_throughput_bound(T value, const T zero,
                                    const T *ptr_to_zero) {
  value *= zero; // Ensure value is set to zero at runtime

#pragma unroll
  for (int i{0}; i < N_MEMORY_INSTRUCTIONS; ++i) {
    // Effectively repeatedly adds zero to zero
    value += ptr_to_zero[value];
  }

  return value;
}

template <bool USE_ARITHMETIC_THROUGHPUT, typename T = int>
__global__ void single_phase_kernel(const T *ptr_to_zero, T *out) {
  const T zero = *ptr_to_zero;
  T value = 1;

  // Single phase
  if (USE_ARITHMETIC_THROUGHPUT) {
    value = arithmetic_throughput_bound<T>(value, zero);
  } else {
    value = cache_throughput_bound<T>(value, zero, ptr_to_zero);
  }

  if (value == 1) {
    *out = 1;
  }
}

template <bool FIRST_USE_ARITHMETIC_THROUGHPUT,
          bool SECOND_USE_ARITHMETIC_THROUGHPUT>
__global__ void two_phase_kernel(const int *ptr_to_zero, int *out) {
  const int zero = *ptr_to_zero;
  int value = 1;

  // First phase
  if (FIRST_USE_ARITHMETIC_THROUGHPUT) {
    value = arithmetic_throughput_bound<int>(value, zero);
  } else {
    value = cache_throughput_bound<int>(value, zero, ptr_to_zero);
  }

  // Second phase
  if (SECOND_USE_ARITHMETIC_THROUGHPUT) {
    value = arithmetic_throughput_bound<int>(value, zero);
  } else {
    value = cache_throughput_bound<int>(value, zero, ptr_to_zero);
  }

  if (value == 1) {
    *out = 1;
  }
}

int32_t main(const int32_t argc, const char **args) {
  constexpr int32_t INPUT_SIZE = 100 * 1024 * 1024;
  constexpr int32_t WARP_SIZE = 32;
  constexpr int32_t BLOCK_SIZE = 2 * WARP_SIZE;
  constexpr int32_t GRID_SIZE = INPUT_SIZE / BLOCK_SIZE;

  const int32_t zero_32 = 0;
  const int64_t zero_64 = 0;

  int32_t *d_zero_32 = allocate_array<int32_t>(1, &zero_32);
  int64_t *d_zero_64 = allocate_array<int64_t>(1, &zero_64);
  int32_t *d_out_32 = allocate_array<int32_t>(1);
  int64_t *d_out_64 = allocate_array<int64_t>(1);

  single_phase_kernel<true><<<GRID_SIZE, BLOCK_SIZE>>>(d_zero_32, d_out_32);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  single_phase_kernel<true, int64_t>
      <<<GRID_SIZE, BLOCK_SIZE>>>(d_zero_64, d_out_64);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  single_phase_kernel<false><<<GRID_SIZE, BLOCK_SIZE>>>(d_zero_32, d_out_32);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  two_phase_kernel<true, true><<<GRID_SIZE, BLOCK_SIZE>>>(d_zero_32, d_out_32);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  two_phase_kernel<true, false><<<GRID_SIZE, BLOCK_SIZE>>>(d_zero_32, d_out_32);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  two_phase_kernel<false, true><<<GRID_SIZE, BLOCK_SIZE>>>(d_zero_32, d_out_32);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  two_phase_kernel<false, false>
      <<<GRID_SIZE, BLOCK_SIZE>>>(d_zero_32, d_out_32);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  free_device_pointer(d_out_32);
  free_device_pointer(d_out_64);
  free_device_pointer(d_zero_32);
  free_device_pointer(d_zero_64);

  return 0;
}
