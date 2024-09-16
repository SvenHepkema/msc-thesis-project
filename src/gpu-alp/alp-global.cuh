#include "alp.cuh"
#include "gpu-bindings-alp.hpp"

#include "../common/utils.hpp"

#ifndef ALP_GLOBAL_CUH
#define ALP_GLOBAL_CUH

template <typename T, typename UINT_T, int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES>
__global__ void alp_global(T *out, AlpColumn<T> data) {
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int16_t vector_index = threadIdx.x / N_LANES;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t n_vectors_per_block = UNPACK_N_VECTORS;

  out += (block_index * n_vectors_per_block + vector_index) *
         consts::VALUES_PER_VECTOR;

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    alp_vector<UINT_T, T, UnpackingType::VectorArray, UNPACK_N_VECTORS,
                     UNPACK_N_VALUES>(out, data, block_index, lane, i);
    out += UNPACK_N_VALUES * N_LANES;
  }
}

#endif // ALP_GLOBAL_CUH
