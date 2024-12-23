#include "fls.cuh"

#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include "old-fls.cuh"
#include <__clang_cuda_runtime_wrapper.h>

#ifndef FLS_BENCHMARK_KERNELS_GLOBAL_H
#define FLS_BENCHMARK_KERNELS_GLOBAL_H

namespace kernels {
namespace fls {
namespace global {
namespace bench {

template <typename T, int UNPACK_N_VALUES>
__global__ void
query_baseline_contains_zero(const T *__restrict in, T *__restrict out) {
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
  const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t vectors_per_warp =  1;
  const int32_t vectors_per_block =  warps_per_block * vectors_per_warp;
  const int32_t vector_index = vectors_per_block * block_index + warp_index * vectors_per_warp;

  in += vector_index * consts::VALUES_PER_VECTOR + lane;


  T registers[UNPACK_N_VALUES];
  T none_zero = 1;

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
#pragma unroll
    for (int j = 0; j < UNPACK_N_VALUES; ++j) {
      registers[j] = in[(j + i) * N_LANES];
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
__global__ void
query_old_fls_contains_zero(const T *__restrict in, T *__restrict out,
                       int32_t value_bit_width) {
  const uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
  const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t vectors_per_warp =  1 * UNPACK_N_VECTORS;
  const int32_t vectors_per_block =  warps_per_block * vectors_per_warp;
  const int32_t vector_index = vectors_per_block * block_index + warp_index * vectors_per_warp;

  in += vector_index * utils::get_compressed_vector_size<T>(value_bit_width);


  T registers[N_VALUES];
  T none_zero = 1;

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
		oldfls::original::unpack(in, registers, value_bit_width);
		//oldfls::adjusted::unpack(in + lane, registers, value_bit_width);

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
query_bp_contains_zero(const T *__restrict in, T *__restrict out,
                       int32_t value_bit_width) {
  const uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
  const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t vectors_per_warp =  1 * UNPACK_N_VECTORS;
  const int32_t vectors_per_block =  warps_per_block * vectors_per_warp;
  const int32_t vector_index = vectors_per_block * block_index + warp_index * vectors_per_warp;

  in += vector_index * utils::get_compressed_vector_size<T>(value_bit_width);


  T registers[N_VALUES];
  T none_zero = 1;

  //out += lane;
  //auto iterator = BPUnpacker<T, T, UnpackingType::LaneArray, UNPACK_N_VALUES>(
      //in, lane, value_bit_width);
  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    //iterator.unpack_next_into(registers);
		/*
    bitunpack_vector<T, UnpackingType::LaneArray, UNPACK_N_VECTORS,
                     UNPACK_N_VALUES>(in, registers, lane, value_bit_width, i);
		*/
    bitunpack_vector_new<T, UnpackingType::LaneArray, UNPACK_N_VECTORS,
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
__global__ void
query_bp_stateful_contains_zero(const T *__restrict in, T *__restrict out,
                       int32_t value_bit_width) {
  const uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
  const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t vectors_per_warp =  1 * UNPACK_N_VECTORS;
  const int32_t vectors_per_block =  warps_per_block * vectors_per_warp;
  const int32_t vector_index = vectors_per_block * block_index + warp_index * vectors_per_warp;

  in += vector_index * utils::get_compressed_vector_size<T>(value_bit_width);


  T registers[N_VALUES];
  T none_zero = 1;

  using UINT_T = typename utils::same_width_uint<T>::type;
  BitUnpacker<UINT_T, T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
              BPFunctor<UINT_T, T>>
      iterator(in , lane, value_bit_width, BPFunctor<UINT_T, T>());
  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
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
  const uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  constexpr uint32_t N_VALUES_IN_LANE = utils::get_values_per_lane<T>();

  const int16_t lane = threadIdx.x % N_LANES;
  const int32_t warps_per_block = blockDim.x / consts::THREADS_PER_WARP;
  const int16_t warp_index = threadIdx.x / consts::THREADS_PER_WARP;
  const int32_t block_index = blockIdx.x;

  constexpr int32_t vectors_per_warp =  1 * UNPACK_N_VECTORS;
  const int32_t vectors_per_block =  warps_per_block * vectors_per_warp;
  const int32_t vector_index = vectors_per_block * block_index + warp_index * vectors_per_warp;

  in += vector_index * utils::get_compressed_vector_size<T>(value_bit_width);

  T registers[N_VALUES];
  T none_zero = 1;

  for (int i = 0; i < N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unffor_vector<T, UnpackingType::VectorArray, UNPACK_N_VECTORS,
                     UNPACK_N_VALUES>(in, registers, lane, value_bit_width, i, base_p);

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
