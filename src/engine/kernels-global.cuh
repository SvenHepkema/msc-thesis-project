#include "device-utils.cuh"
#include "old-fls.cuh"
#include "../flsgpu/flsgpu-api.cuh"

#ifndef FLS_GLOBAL_CUH
#define FLS_GLOBAL_CUH

namespace kernels {

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename OutputProcessor>
struct Baseline : flsgpu::device::BitUnpackerBase<T> {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const UINT_T *in;

  __device__ __forceinline__ Baseline(const UINT_T *__restrict a_in,
                                   const lane_t lane,
                                   [[maybe_unused]] const vbw_t value_bit_width,
                                   [[maybe_unused]] OutputProcessor processor)
      : in(a_in + lane){};

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) override {
    constexpr int32_t N_LANES = utils::get_n_lanes<UINT_T>();

#pragma unroll
    for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
#pragma unroll
      for (int j = 0; j < UNPACK_N_VALUES; ++j) {
        out[v * UNPACK_N_VALUES + j] =
            in[v * consts::VALUES_PER_VECTOR + j * N_LANES];
      }
    }

    in += UNPACK_N_VALUES * N_LANES;
  }
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct DummyALPExceptionPatcher : flsgpu::device::ALPExceptionPatcherBase<T> {

public:
  void __device__ __forceinline__ patch_if_needed(T *out) override {}

  __device__ __forceinline__ DummyALPExceptionPatcher(const flsgpu::device::ALPColumn<T> column,
                                                      const vi_t vector_index,
                                                      const lane_t lane) {}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename OutputProcessor>
struct OldFLSAdjusted : flsgpu::device::BitUnpackerBase<T> {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const UINT_T *in;
  const vbw_t value_bit_width;

  __device__ __forceinline__ OldFLSAdjusted(
      const UINT_T *__restrict a_in, const lane_t lane,
      const vbw_t a_value_bit_width, [[maybe_unused]] OutputProcessor processor)
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
          typename DecompressorT, typename ColumnT>
__global__ void decompress_column(const ColumnT column, T *out) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const lane_t lane = mapping.get_lane();
  const int32_t vector_index = mapping.get_vector_index();

  T registers[N_VALUES];
  out += vector_index * consts::VALUES_PER_VECTOR;

  auto iterator = DecompressorT(column, vector_index, lane);

  for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    iterator.unpack_next_into(registers);

    write_registers_to_global<T, UNPACK_N_VECTORS, UNPACK_N_VALUES,
                              mapping.N_LANES>(lane, i, registers, out);
  }
}


template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES,
          typename DecompressorT, typename ColumnT>
__global__ void query_column(const ColumnT column, T *out, const T magic_value) {
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const lane_t lane = mapping.get_lane();
  const int32_t vector_index = mapping.get_vector_index();

  T registers[N_VALUES];
  auto checker = MagicChecker<T, N_VALUES>(magic_value);

  DecompressorT unpacker = DecompressorT(column, vector_index, lane);

  for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    unpacker.unpack_next_into(registers);
    checker.check(registers);
  }

  checker.write_result(out);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES,
          typename DecompressorT, typename ColumnT, int N_REPETITIONS = 10>
__global__ void compute_column(const ColumnT column, T *__restrict out, 
                               const T runtime_zero) {
  constexpr T RANDOM_VALUE = 3;
  constexpr uint32_t N_VALUES = UNPACK_N_VALUES * UNPACK_N_VECTORS;
  const auto mapping = VectorToThreadMapping<T, UNPACK_N_VECTORS>();
  const lane_t lane = mapping.get_lane();
  const vi_t vector_index = mapping.get_vector_index();

  T registers[N_VALUES];
  auto checker = MagicChecker<T, N_VALUES>(1);
  DecompressorT decompressor = DecompressorT(column, vector_index, lane);

  for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
    decompressor.unpack_next_into(registers);

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

} // namespace kernels

#endif // FLS_GLOBAL_CUH
