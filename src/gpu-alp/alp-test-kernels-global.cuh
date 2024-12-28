#include <cstdint>

#include "../common/utils.hpp"
#include "../gpu-common/gpu-device-utils.cuh"
#include "alp.cuh"

#ifndef ALP_TEST_KERNELS_GLOBAL_CUH
#define ALP_TEST_KERNELS_GLOBAL_CUH

namespace alp {
namespace kernels {
namespace global {
namespace test {

template <typename T, typename UINT_T, int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES>
__global__ void decode_alp_vector_stateless(T *out, AlpColumn<T> data) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const int16_t lane = mapping.get_lane();
  const int32_t vector_index = mapping.get_vector_index();

  T registers[N_VALUES];
  out += vector_index * consts::VALUES_PER_VECTOR;

  for (int i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unalp<UINT_T, T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(registers, data,
                                                        vector_index, lane, i);

    for (int i = 0; i < UNPACK_N_VALUES; i++) {
      out[lane + i * mapping.N_LANES] = registers[i];
    }

    out += UNPACK_N_VALUES * mapping.N_LANES;
  }
}

template <typename T, typename UINT_T, int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES>
__global__ void decode_alp_vector_stateful(T *out, AlpColumn<T> data) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const int16_t lane = mapping.get_lane();
  const int32_t vector_index = mapping.get_vector_index();

  T registers[N_VALUES];
  out += vector_index * consts::VALUES_PER_VECTOR;

  auto iterator = Unpacker<UINT_T, T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
      vector_index, lane, data);

  for (int i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    iterator.unpack_next_into(registers);

    for (int i = 0; i < UNPACK_N_VALUES; i++) {
      out[lane + i * mapping.N_LANES] = registers[i];
    }

    out += UNPACK_N_VALUES * mapping.N_LANES;
  }
}

template <typename T, typename UINT_T, int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES>
__global__ void decode_alp_vector_stateful_extended(T *out,
                                                    AlpExtendedColumn<T> data) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const int16_t lane = mapping.get_lane();
  const int32_t vector_index = mapping.get_vector_index();

  out += vector_index * consts::VALUES_PER_VECTOR;

  T registers[N_VALUES];

  auto iterator =
      ExtendedUnpacker<UINT_T, T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
          vector_index, lane, data);

  for (int i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    iterator.unpack_next_into(registers);

    for (int v{0}; v < UNPACK_N_VECTORS; ++v) {
      for (int w{0}; w < UNPACK_N_VALUES; ++w) {
        out[lane + (i + w) * mapping.N_LANES + v * consts::VALUES_PER_VECTOR] =
            registers[w + v * UNPACK_N_VALUES];
      }
    }
  }
}

} // namespace test
} // namespace global
} // namespace kernels
} // namespace alp

#endif // ALP_TEST_KERNELS_GLOBAL_CUH
