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

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    bitunpack_vector_new<T, UnpackingType::VectorArray, UNPACK_N_VECTORS,
                         UNPACK_N_VALUES>(in, out, lane, value_bit_width, i);
    out += UNPACK_N_VALUES * N_LANES;
  }
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void bitunpack_with_state(const T *__restrict in, T *__restrict out,
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

  out += lane;
  auto iterator = BPUnpacker<T, T, UnpackingType::VectorArray, UNPACK_N_VALUES>(
      in, lane, value_bit_width);
  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    iterator.unpack_next_into(out);

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
