#include "fls.cuh"

#include "../common/consts.hpp"
#include "../common/utils.hpp"

#ifndef FLS_GLOBAL_CUH
#define FLS_GLOBAL_CUH

namespace kernels {
namespace fls {
namespace global {
namespace test {

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void bitunpack(const T *__restrict in, T *__restrict out,
                          int32_t value_bit_width) {
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int16_t vector_index = threadIdx.x / N_LANES;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t n_vectors_per_block = UNPACK_N_VECTORS;

  in += (n_vectors_per_block * block_index + vector_index) *
        utils::get_compressed_vector_size<T>(value_bit_width);
  out += (block_index * n_vectors_per_block + vector_index) *
         consts::VALUES_PER_VECTOR;

  constexpr auto N_VALUES = UNPACK_N_VECTORS * UNPACK_N_VALUES;
  T registers[N_VALUES];

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    bitunpack_vector<T, UnpackingType::LaneArray, UNPACK_N_VECTORS,
                     UNPACK_N_VALUES>(in, registers, lane, value_bit_width, i);

    for (int v{0}; v < UNPACK_N_VECTORS; ++v) {
      for (int w{0}; w < UNPACK_N_VALUES; ++w) {
        out[lane + (i + w) * N_LANES + v * consts::VALUES_PER_VECTOR] =
            registers[w + v * UNPACK_N_VALUES];
      }
    }
  }
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void bitunpack_with_state(T *__restrict in, T *__restrict out,
                                     int32_t value_bit_width) {
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int32_t block_index = blockIdx.x;
  constexpr int32_t n_vectors_per_block = UNPACK_N_VECTORS;
  const int16_t vector_index =
      block_index * n_vectors_per_block + (threadIdx.x / N_LANES);

  T registers[N_VALUES];
  out += vector_index * consts::VALUES_PER_VECTOR;

  using UINT_T = typename utils::same_width_uint<T>::type;
  BitUnpacker<UINT_T, T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
              BPFunctor<UINT_T, T>>
      iterator(in + vector_index *
                        utils::get_compressed_vector_size<T>(value_bit_width),
               lane, value_bit_width, BPFunctor<UINT_T, T>());

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    iterator.unpack_into(registers);

    for (int i = 0; i < UNPACK_N_VALUES; i++) {
      out[lane + i * N_LANES] = registers[i];
    }

    out += UNPACK_N_VALUES * N_LANES;
  }
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void unffor(const T *__restrict in, T *__restrict out,
                       int32_t value_bit_width, const T *__restrict base_p) {
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int16_t vector_index = threadIdx.x / N_LANES;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t n_vectors_per_block = UNPACK_N_VECTORS;

  in += (n_vectors_per_block * block_index + vector_index) *
        utils::get_compressed_vector_size<T>(value_bit_width);
  out += (block_index * n_vectors_per_block + vector_index) *
         consts::VALUES_PER_VECTOR;

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unffor_vector<T, UnpackingType::VectorArray, UNPACK_N_VECTORS,
                  UNPACK_N_VALUES>(in, out, lane, value_bit_width, i, base_p);
    out += UNPACK_N_VALUES * N_LANES;
  }
}

} // namespace test
} // namespace global
} // namespace fls
} // namespace kernels

#endif // FLS_GLOBAL_CUH
