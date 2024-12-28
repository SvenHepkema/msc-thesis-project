#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include "gpu-device-types.cuh"

#ifndef GPU_DEVICE_UTILS_CUH
#define GPU_DEVICE_UTILS_CUH

template <typename T, unsigned UNPACK_N_VECTORS>
struct VectorToThreadMapping {
  static constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  static constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  __device__ __forceinline__ lane_t get_lane() const {
    return threadIdx.x % N_LANES;
  }

  __device__ __forceinline__ vi_t get_vector_index() const {
		// Concurrent vectors per block: how many vectors can be processed
		// by the block simultaneously, assuming that each thread is 1 lane

    const int32_t concurrent_vectors_per_block = blockDim.x / N_LANES;
    const int32_t vectors_per_block = concurrent_vectors_per_block * UNPACK_N_VECTORS;

    const int32_t concurrent_vector_index = threadIdx.x / N_LANES;
    const int32_t block_index = blockIdx.x;

    return vectors_per_block * block_index + concurrent_vector_index * UNPACK_N_VECTORS;
  }
};

#endif // GPU_DEVICE_UTILS_CUH
