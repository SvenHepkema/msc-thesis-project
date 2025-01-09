#include <cstdint>

#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include "../gpu-common/gpu-device-utils.cuh"
#include "fls.cuh"
#include "old-fls.cuh"

#ifndef FLS_BENCHMARK_KERNELS_GLOBAL_H
#define FLS_BENCHMARK_KERNELS_GLOBAL_H

namespace kernels {
namespace fls {
namespace global {
namespace bench {

template <typename T, int UNPACK_N_VALUES>
__global__ void query_baseline_contains_zero(const T *__restrict in,
                                             T *__restrict out) {
  const auto mapping = VectorToThreadMapping<T, 1>();

  in += mapping.get_vector_index() * consts::VALUES_PER_VECTOR +
        mapping.get_lane();

  T registers[UNPACK_N_VALUES];
  auto checker = MagicChecker<T, UNPACK_N_VALUES>(consts::as<T>::MAGIC_NUMBER);

  for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
#pragma unroll
    for (int j = 0; j < UNPACK_N_VALUES; ++j) {
      registers[j] = in[(j + i) * mapping.N_LANES];
    }

    checker.check(registers);
  }

  checker.write_result(out);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void query_old_fls_contains_zero(const T *__restrict in,
                                            T *__restrict out,
                                            vbw_t value_bit_width) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();

  in += mapping.get_vector_index() *
        utils::get_compressed_vector_size<T>(value_bit_width);

  T registers[N_VALUES];
  auto checker = MagicChecker<T, UNPACK_N_VALUES>(consts::as<T>::MAGIC_NUMBER);

  // const lane_t lane = mapping.get_lane();
  for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    oldfls::original::unpack(in, registers, value_bit_width);
    // oldfls::adjusted::unpack(in + lane, registers, value_bit_width);

    checker.check(registers);
  }

  checker.write_result(out);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES,
          typename UnpackerT, typename ProcessorT>
__device__ __forceinline__ void
check_for_magic(T *__restrict out, const T *__restrict in,
                const vbw_t value_bit_width, ProcessorT processor,
                const T magic_value) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const lane_t lane = mapping.get_lane();
  const vi_t vector_index = mapping.get_vector_index();

  in += vector_index * utils::get_compressed_vector_size<T>(value_bit_width);

  T registers[N_VALUES];
  auto checker = MagicChecker<T, UNPACK_N_VALUES>(magic_value);
  UnpackerT unpacker(in, lane, value_bit_width, processor);

  for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unpacker.unpack_next_into(registers);
    checker.check(registers);
  }

  checker.write_result(out);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void query_bp_contains_zero(const T *__restrict in,
                                       T *__restrict out,
                                       vbw_t value_bit_width) {
  check_for_magic<
      T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
      BitUnpackerStateless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, BPFunctor<T>>>(
      out, in, value_bit_width, BPFunctor<T>(), consts::as<T>::MAGIC_NUMBER);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void query_bp_stateful_contains_zero(const T *__restrict in,
                                                T *__restrict out,
                                                vbw_t value_bit_width) {
  check_for_magic<
      T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
      BitUnpackerStateful<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, BPFunctor<T>>>(
      out, in, value_bit_width, BPFunctor<T>(), consts::as<T>::MAGIC_NUMBER);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void
query_ffor_contains_zero(const T *__restrict in, T *__restrict out,
                         vbw_t value_bit_width, const T *__restrict base_p) {
  check_for_magic<T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
                  BitUnpackerStateless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
                                       FFORFunctor<T>>>(
      out, in, value_bit_width, FFORFunctor<T>(*base_p),
      consts::as<T>::MAGIC_NUMBER);
}

} // namespace bench
} // namespace global
} // namespace fls
} // namespace kernels

#endif // FLS_BENCHMARK_KERNELS_GLOBAL_H
