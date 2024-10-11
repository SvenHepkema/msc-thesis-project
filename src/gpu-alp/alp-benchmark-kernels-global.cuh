#include "alp.cuh"

#include "../common/utils.hpp"

#ifndef ALP_BENCHMARK_GLOBAL_CUH
#define ALP_BENCHMARK_GLOBAL_CUH

namespace alp {
namespace kernels {
namespace global {
namespace bench {

template <typename T, typename UINT_T, int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES>
__global__ void decode_baseline(T *out, const T *in) {
  const uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
  const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t vectors_per_warp = 1 * UNPACK_N_VECTORS;
  const int32_t vectors_per_block = warps_per_block * vectors_per_warp;
  const int32_t vector_index =
      vectors_per_block * block_index + warp_index * vectors_per_warp;

  in += vector_index * consts::VALUES_PER_VECTOR;

  T registers[N_VALUES];
  bool none_magic = true;

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
#pragma unroll
    for (int j = 0; j < UNPACK_N_VALUES; ++j) {
#pragma unroll
      for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
        uint32_t index = (v * consts::VALUES_PER_VECTOR + i +
                         j * N_LANES) * N_LANES + lane;
        registers[v * UNPACK_N_VALUES + j] = in[index];
      }
    }

#pragma unroll
    for (int j = 0; j < N_VALUES; ++j) {
      none_magic &= registers[j] != consts::as<T>::MAGIC_NUMBER;
    }
  }

  if (!none_magic) {
    *out = 1.0;
  }
}

template <typename T, typename UINT_T, int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES>
__global__ void decode_complete_alp_vector(T *out, AlpColumn<T> data) {
  const uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
  const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t vectors_per_warp = 1 * UNPACK_N_VECTORS;
  const int32_t vectors_per_block = warps_per_block * vectors_per_warp;
  const int32_t vector_index =
      vectors_per_block * block_index + warp_index * vectors_per_warp;

  T registers[N_VALUES];
  bool none_magic = true;

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unalp<UINT_T, T, UnpackingType::LaneArray, UNPACK_N_VECTORS,
          UNPACK_N_VALUES>(registers, data, vector_index, lane, i);
#pragma unroll
    for (int j = 0; j < N_VALUES; ++j) {
      none_magic &= registers[j] != consts::as<T>::MAGIC_NUMBER;
    }
  }

  if (!none_magic) {
    *out = 1.0;
  }
}

template <typename T, typename UINT_T, int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES>
__global__ void decode_complete_alprd_vector(T *out, AlpRdColumn<T> data) {
  const uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
  const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t vectors_per_warp = 1 * UNPACK_N_VECTORS;
  const int32_t vectors_per_block = warps_per_block * vectors_per_warp;
  const int32_t vector_index =
      vectors_per_block * block_index + warp_index * vectors_per_warp;

  T registers[N_VALUES];
  bool none_magic = true;

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unalprd<UINT_T, T, UnpackingType::LaneArray, UNPACK_N_VECTORS,
          UNPACK_N_VALUES>(registers, data, vector_index, lane, i);
#pragma unroll
    for (int j = 0; j < N_VALUES; ++j) {
      none_magic &= registers[j] != consts::as<T>::MAGIC_NUMBER;
    }
  }

  if (!none_magic) {
    *out = 1.0;
  }
}

} // namespace bench
} // namespace global
} // namespace kernels
} // namespace alp

#endif // ALP_BENCHMARK_GLOBAL_CUH
