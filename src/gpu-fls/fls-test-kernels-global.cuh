#include "fls.cuh"

#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include "../gpu-common/gpu-device-utils.cuh"

#ifndef FLS_GLOBAL_CUH
#define FLS_GLOBAL_CUH

namespace kernels {
namespace fls {
namespace global {
namespace test {

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES,
          typename UnpackerT>
__global__ void bitunpack(T *__restrict out, const T *__restrict in,
                          const vbw_t value_bit_width) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const lane_t lane = mapping.get_lane();
  const vi_t vector_index = mapping.get_vector_index();

  in += vector_index * utils::get_compressed_vector_size<T>(value_bit_width);
  out += vector_index * consts::VALUES_PER_VECTOR;

  T registers[N_VALUES];

  UnpackerT unpacker(in, lane, value_bit_width, BPFunctor<T>());

  for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unpacker.unpack_next_into(registers);

    write_registers_to_global<T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
                              mapping.N_LANES>(lane, i, registers, out);
  }
}

} // namespace test
} // namespace global
} // namespace fls
} // namespace kernels

#endif // FLS_GLOBAL_CUH
