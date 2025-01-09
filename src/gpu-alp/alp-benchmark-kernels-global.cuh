#include "../common/utils.hpp"
#include "../gpu-common/gpu-device-utils.cuh"
#include "alp.cuh"

#ifndef ALP_BENCHMARK_GLOBAL_CUH
#define ALP_BENCHMARK_GLOBAL_CUH

namespace alp {
namespace kernels {
namespace global {
namespace bench {

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void decode_baseline(T *out, const T *in,
                                const int n_values_in_lane) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const lane_t lane = mapping.get_lane();
  const int32_t vector_index = mapping.get_vector_index();

  in += vector_index * consts::VALUES_PER_VECTOR;

  T registers[N_VALUES];
  auto checker = MagicChecker<T, N_VALUES>(consts::as<T>::MAGIC_NUMBER);

  for (si_t i = 0; i < n_values_in_lane; i += UNPACK_N_VALUES) {
#pragma unroll
    for (int j = 0; j < UNPACK_N_VALUES; ++j) {
#pragma unroll
      for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
        uint32_t index =
            (v * consts::VALUES_PER_VECTOR + i + j * mapping.N_LANES) *
                mapping.N_LANES +
            lane;
        registers[v * UNPACK_N_VALUES + j] = in[index];
      }
    }

    checker.check(registers);
  }

  checker.write_result(out);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES,
          typename UnpackerT, typename ColumnT>
__device__ __forceinline__ void check_for_magic(T *out, ColumnT column,
                                                const T magic_value) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const lane_t lane = mapping.get_lane();
  const int32_t vector_index = mapping.get_vector_index();

  T registers[N_VALUES];
  auto checker = MagicChecker<T, N_VALUES>(magic_value);

  UnpackerT unpacker = UnpackerT(column, vector_index, lane);

  for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unpacker.unpack_next_into(registers);
    checker.check(registers);
  }

  checker.write_result(out);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void contains_magic_stateless(T *out, AlpColumn<T> data,
                                         const T magic_value) {
  check_for_magic<T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
                  AlpStatelessUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>>(
      out, data, magic_value);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void contains_magic_stateful(T *out, AlpColumn<T> column,
                                        const T magic_value) {
  check_for_magic<T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
                  AlpStatefulUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>>(
      out, column, magic_value);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void contains_magic_stateful_extended(T *out,
                                                 AlpExtendedColumn<T> column,
                                                 const T magic_value) {
  check_for_magic<
      T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
      AlpStatefulExtendedUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>>(
      out, column, magic_value);
}

template <typename T, typename UINT_T, int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES>
__global__ void decode_multiple_alp_vectors(T *out, AlpColumn<T> column_0,
                                            AlpColumn<T> column_1) {
  /*
const uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

const lane_t lane = threadIdx.x % N_LANES;
const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
const int32_t block_index = blockIdx.x;

constexpr int32_t vectors_per_warp = 1 * UNPACK_N_VECTORS;
const int32_t vectors_per_block = warps_per_block * vectors_per_warp;
const int32_t vector_index =
vectors_per_block * block_index + warp_index * vectors_per_warp;

T registers_column_0[N_VALUES];
T registers_column_1[N_VALUES];
bool none_equal = true;

for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
unalp<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(registers_column_0, column_0,
                                          vector_index, lane, i);
unalp<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(registers_column_1, column_1,
                                          vector_index, lane, i);
#pragma unroll
for (int j = 0; j < N_VALUES; ++j) {
none_equal &= registers_column_0[j] != registers_column_1[j];
}
}

if (!none_equal) {
*out = 1.0;
}
  */
}
template <typename T, typename UINT_T, int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES>
__global__ void decode_multiple_alp_vectors(T *out, AlpColumn<T> column_0,
                                            AlpColumn<T> column_1,
                                            AlpColumn<T> column_2) {
  /*
const uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

const lane_t lane = threadIdx.x % N_LANES;
const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
const int32_t block_index = blockIdx.x;

constexpr int32_t vectors_per_warp = 1 * UNPACK_N_VECTORS;
const int32_t vectors_per_block = warps_per_block * vectors_per_warp;
const int32_t vector_index =
vectors_per_block * block_index + warp_index * vectors_per_warp;

T registers_column_0[N_VALUES];
T registers_column_1[N_VALUES];
T registers_column_2[N_VALUES];
bool none_equal = true;

for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
unalp<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(registers_column_0, column_0,
                                          vector_index, lane, i);
unalp<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(registers_column_1, column_1,
                                          vector_index, lane, i);
unalp<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(registers_column_2, column_2,
                                          vector_index, lane, i);
#pragma unroll
for (int j = 0; j < N_VALUES; ++j) {
none_equal &= registers_column_0[j] != registers_column_1[j] &&
              registers_column_1[j] != registers_column_2[j];
}
}

if (!none_equal) {
*out = 1.0;
}
  */
}
template <typename T, typename UINT_T, int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES>
__global__ void decode_multiple_alp_vectors(T *out, AlpColumn<T> column_0,
                                            AlpColumn<T> column_1,
                                            AlpColumn<T> column_2,
                                            AlpColumn<T> column_3) {
  /*
const uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

const lane_t lane = threadIdx.x % N_LANES;
const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
const int32_t block_index = blockIdx.x;

constexpr int32_t vectors_per_warp = 1 * UNPACK_N_VECTORS;
const int32_t vectors_per_block = warps_per_block * vectors_per_warp;
const int32_t vector_index =
vectors_per_block * block_index + warp_index * vectors_per_warp;

T registers_column_0[N_VALUES];
T registers_column_1[N_VALUES];
T registers_column_2[N_VALUES];
T registers_column_3[N_VALUES];
bool none_equal = true;

for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
unalp<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(registers_column_0, column_0,
                                          vector_index, lane, i);
unalp<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(registers_column_1, column_1,
                                          vector_index, lane, i);
unalp<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(registers_column_2, column_2,
                                          vector_index, lane, i);
unalp<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(registers_column_3, column_3,
                                          vector_index, lane, i);

#pragma unroll
for (int j = 0; j < N_VALUES; ++j) {
none_equal &= registers_column_0[j] != registers_column_1[j] &&
              registers_column_1[j] != registers_column_2[j] &&
              registers_column_2[j] != registers_column_3[j];
}
}

if (!none_equal) {
*out = 1.0;
}
  */
}

} // namespace bench
} // namespace global
} // namespace kernels
} // namespace alp

#endif // ALP_BENCHMARK_GLOBAL_CUH
