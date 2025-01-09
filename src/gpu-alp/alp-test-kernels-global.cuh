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

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES,
          typename UnpackerT, typename ColumnT>
__device__ __forceinline__ void decompress_into_out(T *out, ColumnT data) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const lane_t lane = mapping.get_lane();
  const int32_t vector_index = mapping.get_vector_index();

  T registers[N_VALUES];
  out += vector_index * consts::VALUES_PER_VECTOR;

  auto iterator = UnpackerT(data, vector_index, lane);

  for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    iterator.unpack_next_into(registers);

    write_registers_to_global<T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
                              mapping.N_LANES>(lane, i, registers, out);
  }
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void decode_alp_vector_stateless(T *out, AlpColumn<T> data) {
  decompress_into_out<
      T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
      AlpStatelessUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>>(out, data);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void decode_alp_vector_stateful(T *out, AlpColumn<T> data) {
  decompress_into_out<
      T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
      AlpStatefulUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>>(out, data);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void decode_alp_vector_stateful_extended(T *out,
                                                    AlpExtendedColumn<T> data) {
  decompress_into_out<
      T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
      AlpStatefulExtendedUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>>(out,
                                                                         data);
}

} // namespace test
} // namespace global
} // namespace kernels
} // namespace alp

#endif // ALP_TEST_KERNELS_GLOBAL_CUH
