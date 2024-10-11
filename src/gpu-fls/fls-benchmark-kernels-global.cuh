#include "fls.cuh"

#include "../common/consts.hpp"
#include "../common/utils.hpp"

#ifndef FLS_BENCHMARK_KERNELS_GLOBAL_H
#define FLS_BENCHMARK_KERNELS_GLOBAL_H

namespace kernels {
namespace fls {
namespace global {
namespace bench {

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void
query_bp_contains_zero(const T *__restrict in, T *__restrict out,
                       int32_t value_bit_width) {
  const uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
  const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t vectors_per_warp =  1 * UNPACK_N_VECTORS;
  const int32_t vectors_per_block =  warps_per_block * vectors_per_warp;
  const int32_t vector_index = vectors_per_block * block_index + warp_index * vectors_per_warp;

  in += vector_index * utils::get_compressed_vector_size<T>(value_bit_width);


  T registers[N_VALUES];
  T none_zero = 1;

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    bitunpack_vector<T, UnpackingType::LaneArray, UNPACK_N_VECTORS,
                     UNPACK_N_VALUES>(in, registers, lane, value_bit_width, i);

#pragma unroll
    for (int j = 0; j < N_VALUES; ++j) {
      none_zero *= registers[j] != consts::as<T>::MAGIC_NUMBER;
    }
  }

	if (!none_zero) {
		*out = 1;
	}
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void
query_ffor_contains_zero(const T *__restrict in, T *__restrict out,
                       int32_t value_bit_width, const T *__restrict base_p) {
  const uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
  const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t vectors_per_warp =  1 * UNPACK_N_VECTORS;
  const int32_t vectors_per_block =  warps_per_block * vectors_per_warp;
  const int32_t vector_index = vectors_per_block * block_index + warp_index * vectors_per_warp;

  in += vector_index * utils::get_compressed_vector_size<T>(value_bit_width);

  T registers[N_VALUES];
  T none_zero = 1;

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unffor_vector<T, UnpackingType::VectorArray, UNPACK_N_VECTORS,
                     UNPACK_N_VALUES>(in, registers, lane, value_bit_width, i, base_p);

#pragma unroll
    for (int j = 0; j < N_VALUES; ++j) {
      none_zero *= registers[j] != 0;
    }
  }

	if (!none_zero) {
		*out = 1;
	}
}

} // namespace bench
} // namespace global
} // namespace fls
} // namespace kernels

#endif // FLS_BENCHMARK_KERNELS_GLOBAL_H
