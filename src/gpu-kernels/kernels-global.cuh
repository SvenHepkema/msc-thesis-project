#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include "old-fls.cuh"

#include "device-utils.cuh"

#include "alp.cuh"
#include "fls.cuh"

#ifndef FLS_GLOBAL_CUH
#define FLS_GLOBAL_CUH

namespace kernels {
namespace device {
namespace fls {

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename OutputProcessor>
struct Baseline : BitUnpackerBase<T> {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const UINT_T *in;

  __device__ __forceinline__
  Baseline(const UINT_T *__restrict a_in, const lane_t lane,
           [[maybe_unused]] const vbw_t value_bit_width,
           [[maybe_unused]] OutputProcessor processor)
      : in(a_in + lane){};

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) override {
    constexpr int32_t N_LANES = utils::get_n_lanes<UINT_T>();

#pragma unroll
    for (int j = 0; j < UNPACK_N_VALUES; ++j) {
      out[j] = in[j * N_LANES];
    }

    in += UNPACK_N_VALUES * N_LANES;
  }
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename OutputProcessor>
struct OldFLSAdjusted : BitUnpackerBase<T> {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const UINT_T *in;
  const vbw_t value_bit_width;

  __device__ __forceinline__
  OldFLSAdjusted(const UINT_T *__restrict a_in, const lane_t lane,
                 [[maybe_unused]] const vbw_t a_value_bit_width,
                 [[maybe_unused]] OutputProcessor processor)
      : in(a_in + lane), value_bit_width(a_value_bit_width) {
    static_assert(UNPACK_N_VECTORS == 1, "Old FLS can only unpack 1 at a time");
    static_assert(UNPACK_N_VALUES == utils::get_values_per_lane<T>(),
                  "Old FLS can only unpack entire lanes");
  };

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) override {
    oldfls::adjusted::unpack(in, out, value_bit_width);
  }
};

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES,
          typename UnpackerT>
__global__ void decompress_column(T *__restrict out, const T *__restrict in,
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

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES,
          typename UnpackerT>
__global__ void query_column_contains_zero(T *__restrict out,
                                           const T *__restrict in,
                                           const vbw_t value_bit_width) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const lane_t lane = mapping.get_lane();
  const vi_t vector_index = mapping.get_vector_index();

  in += vector_index * utils::get_compressed_vector_size<T>(value_bit_width);

  T registers[N_VALUES];
  auto checker = MagicChecker<T, UNPACK_N_VALUES>(consts::as<T>::MAGIC_NUMBER);
  UnpackerT unpacker(in, lane, value_bit_width, BPFunctor<T>());

  for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unpacker.unpack_next_into(registers);
    checker.check(registers);
  }

  checker.write_result(out);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES,
          typename UnpackerT, int N_REPETITIONS>
__global__ void compute_column(T *__restrict out, const T *__restrict in,
                               const vbw_t value_bit_width,
                               const T runtime_zero) {
  constexpr T RANDOM_VALUE = 3;
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const lane_t lane = mapping.get_lane();
  const vi_t vector_index = mapping.get_vector_index();

  in += vector_index * utils::get_compressed_vector_size<T>(value_bit_width);

  T registers[N_VALUES];
  auto checker = MagicChecker<T, UNPACK_N_VALUES>(1);
  UnpackerT unpacker(in, lane, value_bit_width, BPFunctor<T>());

  for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unpacker.unpack_next_into(registers);

#pragma unroll
    for (int32_t j{0}; j < N_VALUES; ++j) {
#pragma unroll
      for (int32_t k{0}; k < N_REPETITIONS; ++k) {
        registers[j] *= RANDOM_VALUE;
        registers[j] <<= RANDOM_VALUE;
        registers[j] += RANDOM_VALUE;
        registers[j] &= runtime_zero;
      }
    }

    checker.check(registers);
  }

  checker.write_result(out);
}

} // namespace fls

namespace alp {

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES,
          typename UnpackerT, typename ColumnT>
__global__ void decompress_column(T *out, ColumnT data) {
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

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct DummyALPExceptionPatcher : ALPExceptionPatcherBase<T> {

public:
  void __device__ __forceinline__ patch_if_needed(T *out) override {}

  __device__ __forceinline__ DummyALPExceptionPatcher(const AlpColumn<T> column,
                                                      const vi_t vector_index,
                                                      const lane_t lane) {}
};

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES,
          typename UnpackerT, typename ColumnT>
__global__ void query_column_contains_magic(T *out, ColumnT column,
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

} // namespace alp

} // namespace device
} // namespace kernels

#endif // FLS_GLOBAL_CUH
