#include <algorithm>
#include <cstdint>
#include <random>

#include "utils.cuh"

constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_SHARED_MEM_PER_BLOCK = 48 * 1024;

template <unsigned SHARED_BYTES_SIZE>
__device__ int use_shared_memory(const int thread_id, const int zero) {
  using SHARED_MEMORY_T = int;
  constexpr int SHARED_ARRAY_SIZE = SHARED_BYTES_SIZE / sizeof(SHARED_MEMORY_T);

  __shared__ SHARED_MEMORY_T memory[SHARED_ARRAY_SIZE];

  memory[zero] = zero;
  return memory[0];
}

template <unsigned CHASE_N_PTRS, unsigned ILP_FACTOR>
__device__ int chase_ptr(const int thread_id, const int *ptr_to_arr,
                         const int arr_size) {
  const int GRID_DIM_X = gridDim.x;

  int chasers[ILP_FACTOR];
#pragma unroll
  for (int v{0}; v < ILP_FACTOR; ++v) {
    // The grid dim x is divided by the ILP factor for a fair comparison,
    // so we initialize each chaser with the index of a warp that is outside
    // of the higher ILP grid, to ensure that all pointers are followed
    const int start_index = (thread_id + (v * GRID_DIM_X)) % arr_size;
    chasers[v] = start_index;
  }

  // Do the pointer chase for each value
#pragma unroll
  for (int i{0}; i < CHASE_N_PTRS; ++i) {
#pragma unroll
    for (int v{0}; v < ILP_FACTOR; ++v) {
      chasers[v] = ptr_to_arr[chasers[v]];
    }
  }

  // Read each element in array in non-dynamic fashion
  // Prevents compiler from optimizing out non used values
  int result = thread_id;
#pragma unroll
  for (int v{0}; v < ILP_FACTOR; ++v) {
    result &= chasers[v];
  }

  return result;
}

template <unsigned SHARED_BYTES_SIZE, unsigned CHASE_N_PTRS,
          unsigned ILP_FACTOR>
__global__ void ptr_chasing_kernel(const int *ptr_to_arr, const int arr_size,
                                   int *out) {
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  // Returns 0 if thread_id does not overflow
  const int zero = thread_id == -1;

  const bool is_first_thread_in_warp = thread_id % WARP_SIZE == 0;
  if (!is_first_thread_in_warp)
    return;

  thread_id = use_shared_memory<SHARED_BYTES_SIZE>(thread_id, zero);

  int value =
      chase_ptr<CHASE_N_PTRS, ILP_FACTOR>(thread_id, ptr_to_arr, arr_size);

  if (value == -1) {
    *out = 1;
  }
}

template <typename T>
T *fill_array_with_sequence(T *array, const size_t n_values) {
  for (size_t i{0}; i < n_values; ++i) {
    array[i] = i;
  }

  return array;
}

int32_t main(const int32_t argc, const char **args) {
  constexpr int32_t TOTAL_THREADS = 10000 * 1024;
  constexpr int32_t GLOBAL_ARRAY_SIZE = 10 * 1024 * 1024;

  // Create random permutation of indices
  std::random_device random_device;
  std::default_random_engine random_engine(random_device());
  int32_t *array = fill_array_with_sequence(new int32_t[GLOBAL_ARRAY_SIZE],
                                            GLOBAL_ARRAY_SIZE);
  std::shuffle(array, array + GLOBAL_ARRAY_SIZE, random_engine);

  constexpr int32_t CHASE_N_PTRS = 100;
  int32_t *out = 0;

  int32_t *d_array = allocate_array<int32_t>(GLOBAL_ARRAY_SIZE, array);
  int32_t *d_out = allocate_array<int32_t>(1);

  for (size_t block_size{WARP_SIZE}; block_size <= 1024;
       block_size += WARP_SIZE) {
    const int32_t grid_size = TOTAL_THREADS / block_size;
    ptr_chasing_kernel<MAX_SHARED_MEM_PER_BLOCK, CHASE_N_PTRS, 1>
        <<<grid_size / 1, block_size>>>(d_array, GLOBAL_ARRAY_SIZE, d_out);
    ptr_chasing_kernel<MAX_SHARED_MEM_PER_BLOCK, CHASE_N_PTRS, 2>
        <<<grid_size / 2, block_size>>>(d_array, GLOBAL_ARRAY_SIZE, d_out);
    ptr_chasing_kernel<MAX_SHARED_MEM_PER_BLOCK, CHASE_N_PTRS, 4>
        <<<grid_size / 4, block_size>>>(d_array, GLOBAL_ARRAY_SIZE, d_out);
    ptr_chasing_kernel<MAX_SHARED_MEM_PER_BLOCK, CHASE_N_PTRS, 8>
        <<<grid_size / 8, block_size>>>(d_array, GLOBAL_ARRAY_SIZE, d_out);
  }

  free_device_pointer(d_out);
  free_device_pointer(d_array);

  return *out == 10 ? 1 : 0;
}
