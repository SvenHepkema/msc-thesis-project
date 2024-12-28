#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include <__clang_cuda_runtime_wrapper.h>

#ifndef GPU_DEVICE_UTILS_CUH
#define GPU_DEVICE_UTILS_CUH

template <typename T, unsigned UNPACK_N_VECTORS> struct VectorToWarpOrientation {
  static constexpr uint32_t N_LANES = utils::get_values_per_lane<T>();
  static constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  __device__ __forceinline__ int16_t get_lane() const {
    return threadIdx.x % N_LANES;
  }


  __device__ __forceinline__ int32_t get_vector_index() const {
		const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
		const int32_t vectors_per_block =  warps_per_block * UNPACK_N_VECTORS;

		const int32_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
		const int32_t block_index = blockIdx.x;

		return vectors_per_block * block_index + warp_index * UNPACK_N_VECTORS;
	}
};

#endif // GPU_DEVICE_UTILS_CUH
