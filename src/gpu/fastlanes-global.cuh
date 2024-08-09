#include "fastlanes-global.h"
#include "fastlanes.cuh"

#include "../consts.h"
#include "../utils.h"

template <typename T_in, typename T_out, int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES>
__global__ void bitunpack_with_function_global(const T_in *__restrict in,
                                               T_out *__restrict out,
                                               int32_t value_bit_width) {
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T_in>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T_in>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T_in>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int16_t vector_index = threadIdx.x / N_LANES;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t n_vectors_per_block = UNPACK_N_VECTORS;
  uint32_t vector_offset = (n_vectors_per_block * block_index + vector_index) *
                           utils::get_compressed_vector_size<T_in>(value_bit_width);

  T_out registers[UNPACK_N_VALUES * UNPACK_N_VECTORS];

  in += vector_offset;
  out += (block_index * n_vectors_per_block + vector_index) *
             consts::VALUES_PER_VECTOR +
         lane;

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unpack_vector<T_in, T_out, UnpackingType::LaneArray, UNPACK_N_VECTORS,
                 UNPACK_N_VALUES>(in, registers, lane, value_bit_width, i);

#pragma unroll
    for (int va = 0; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
      for (int ve = 0; ve < UNPACK_N_VECTORS; ++ve) {
        *(out + ve * consts::VALUES_PER_VECTOR) =
            registers[ve * UNPACK_N_VALUES + va];
      }
      out += N_LANES;
    }
  }
}

template <typename T_in, typename T_out, int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES>
__global__ void bitunpack_with_reader_global(const T_in *__restrict in,
                                      T_out *__restrict out,
                                      int32_t value_bit_width) {
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T_in>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T_in>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T_in>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int16_t vector_index = threadIdx.x / N_LANES;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t n_vectors_per_block = UNPACK_N_VECTORS;
  uint32_t vector_offset = (n_vectors_per_block * block_index + vector_index) *
                           utils::get_compressed_vector_size<T_in>(value_bit_width);

  T_out registers[UNPACK_N_VALUES * UNPACK_N_VECTORS];

  in += vector_offset;
  out += (block_index * n_vectors_per_block + vector_index) *
             consts::VALUES_PER_VECTOR +
         lane;

  auto scanner =
      MultiVecScanner<T_in, UnpackingType::LaneArray, UNPACK_N_VECTORS,
                      UNPACK_N_VALUES>(in, value_bit_width, lane);


  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    scanner.unpack_next(registers);

#pragma unroll
    for (int va = 0; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
      for (int ve = 0; ve < UNPACK_N_VECTORS; ++ve) {
        *(out + ve * consts::VALUES_PER_VECTOR) =
            registers[ve * UNPACK_N_VALUES + va];
      }
      out += N_LANES;
    }
  }
}

template <typename T_in, typename T_out>
void bitunpack_with_function(T_in *__restrict in, T_out *__restrict out,
                             int32_t value_bit_width) {
  bitunpack_with_function_global<T_in, T_out, 1,
                                 utils::get_values_per_lane<T_in>()>(
      in, out, value_bit_width);
}

template <typename T_in, typename T_out>
void bitunpack_with_reader(T_in *__restrict in, T_out *__restrict out,
                             int32_t value_bit_width) {
  bitunpack_with_reader_global<T_in, T_out, 1,
                                 utils::get_values_per_lane<T_in>()>(
      in, out, value_bit_width);
}

