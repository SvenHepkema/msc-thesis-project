#include "../flsgpu/flsgpu-api.cuh"

#ifndef GPU_DEVICE_UTILS_CUH
#define GPU_DEVICE_UTILS_CUH

template <typename T, unsigned UNPACK_N_VECTORS> struct VectorToThreadMapping {
  static constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  static constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  __device__ __forceinline__ lane_t get_lane() const {
    return threadIdx.x % N_LANES;
  }

  __device__ __forceinline__ vi_t get_vector_index() const {
    // Concurrent vectors per block: how many vectors can be processed
    // by the block simultaneously, assuming that each thread is 1 lane

    const int32_t concurrent_vectors_per_block = blockDim.x / N_LANES;
    const int32_t vectors_per_block =
        concurrent_vectors_per_block * UNPACK_N_VECTORS;

    const int32_t concurrent_vector_index = threadIdx.x / N_LANES;
    const int32_t block_index = blockIdx.x;

    return vectors_per_block * block_index +
           concurrent_vector_index * UNPACK_N_VECTORS;
  }
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          unsigned N_LANES>
__device__ __forceinline__ void
write_registers_to_global(const lane_t lane, const si_t index_offset,
                          const T *__restrict registers, T *__restrict out) {
  for (int v{0}; v < UNPACK_N_VECTORS; ++v) {
    for (int w{0}; w < UNPACK_N_VALUES; ++w) {
      out[lane + (index_offset + w) * N_LANES + v * consts::VALUES_PER_VECTOR] =
          registers[w + v * UNPACK_N_VALUES];
    }
  }
}

template <typename T, unsigned N_VALUES> struct MagicChecker {
  const T magic_value;
  bool no_magic_found = true;

  __device__ __forceinline__ MagicChecker(const T magic_value)
      : magic_value(magic_value) {}

  __device__ __forceinline__ void check(const T *__restrict registers) {
#pragma unroll
    for (int i = 0; i < N_VALUES; ++i) {
      no_magic_found &= registers[i] != magic_value;
    }
  }

  __device__ __forceinline__ void write_result(bool *__restrict out) {
    // This is a branch, as we do not want to write 0s, only emit a write
    // if we found a magic value
    if (!no_magic_found) {
      *out = true;
    }
  }
};

#endif // GPU_DEVICE_UTILS_CUH
