#include "fls.cuh"

#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include "../gpu-common/gpu-device-utils.cuh"
#include "old-fls.cuh"
#include <cstdint>

#ifndef FLS_BENCHMARK_KERNELS_GLOBAL_H
#define FLS_BENCHMARK_KERNELS_GLOBAL_H

namespace kernels {
namespace fls {
namespace global {
namespace bench {

template <typename T, int UNPACK_N_VALUES>
__global__ void query_baseline_contains_zero(const T *__restrict in,
                                             T *__restrict out) {
  const auto orientation = VectorToWarpOrientation<T, 1>();

  in += orientation.get_vector_index() * consts::VALUES_PER_VECTOR +
        orientation.get_lane();

  T registers[UNPACK_N_VALUES];
  T none_zero = 1;

  for (int i = 0; i < orientation.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
#pragma unroll
    for (int j = 0; j < UNPACK_N_VALUES; ++j) {
      registers[j] = in[(j + i) * orientation.N_LANES];
    }

#pragma unroll
    for (int j = 0; j < UNPACK_N_VALUES; ++j) {
      none_zero *= registers[j] != consts::as<T>::MAGIC_NUMBER;
    }
  }

  if (!none_zero) {
    *out = 1;
  }
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void query_old_fls_contains_zero(const T *__restrict in,
                                            T *__restrict out,
                                            int32_t value_bit_width) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto orientation = VectorToWarpOrientation<T, 1>();

  in += orientation.get_vector_index() *
        utils::get_compressed_vector_size<T>(value_bit_width);

  T registers[N_VALUES];
  T none_zero = 1;

  // const int16_t lane = threadIdx.x % N_LANES;
  for (int i = 0; i < orientation.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    oldfls::original::unpack(in, registers, value_bit_width);
    // oldfls::adjusted::unpack(in + lane, registers, value_bit_width);

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
__global__ void query_bp_contains_zero(const T *__restrict in,
                                       T *__restrict out,
                                       int32_t value_bit_width) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto orientation = VectorToWarpOrientation<T, 1>();

  in += orientation.get_vector_index() *
        utils::get_compressed_vector_size<T>(value_bit_width);

  T registers[N_VALUES];
  T none_zero = 1;
  int16_t lane = orientation.get_lane();

  for (int i = 0; i < orientation.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
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
__global__ void query_bp_stateful_contains_zero(const T *__restrict in,
                                                T *__restrict out,
                                                int32_t value_bit_width) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto orientation = VectorToWarpOrientation<T, 1>();

  in += orientation.get_vector_index() * utils::get_compressed_vector_size<T>(value_bit_width);

  T registers[N_VALUES];
  T none_zero = 1;
  int16_t lane = orientation.get_lane();

  using UINT_T = typename utils::same_width_uint<T>::type;
  BitUnpacker<UINT_T, T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
              BPFunctor<UINT_T, T>>
      iterator(in, lane, value_bit_width, BPFunctor<UINT_T, T>());
  for (int i = 0; i < orientation.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    iterator.unpack_into(registers);

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
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto orientation = VectorToWarpOrientation<T, 1>();
  int16_t lane = orientation.get_lane();

  in += orientation.get_vector_index() * utils::get_compressed_vector_size<T>(value_bit_width);

  T registers[N_VALUES];
  T none_zero = 1;

  for (int i = 0; i < orientation.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unffor_vector<T, UnpackingType::VectorArray, UNPACK_N_VECTORS,
                  UNPACK_N_VALUES>(in, registers, lane, value_bit_width, i,
                                   base_p);

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
