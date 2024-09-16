#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "../alp/constants.hpp"
#include "../common/utils.hpp"
#include "../gpu-fls/fls.cuh"

#ifndef ALP_CUH
#define ALP_CUH

template <typename T> struct AlpColumn {
  using UINT_T = typename utils::same_width_uint<T>::type;

  UINT_T *ffor_array;
  UINT_T *ffor_bases;
  uint8_t *bit_widths;
  uint8_t *exponents;
  uint8_t *factors;

  T *exceptions;
  uint16_t *positions;
  uint16_t *counts;
};

template <typename T> struct AlpRdColumn {
  using UINT_T = typename utils::same_width_uint<T>::type;

  uint16_t *left_ffor_array;
  uint16_t *left_ffor_bases;
  uint8_t *left_bit_widths;

  UINT_T *right_ffor_array;
  UINT_T *right_ffor_bases;
  uint8_t *right_bit_widths;

  uint16_t *left_parts_dicts;

  uint16_t *exceptions;
  uint16_t *positions;
  uint16_t *counts;
};

namespace constant_memory {
constexpr int32_t F_FACT_ARR_COUNT = 10;
constexpr int32_t F_FRAC_ARR_COUNT = 11;
__constant__ int32_t F_FACT_ARRAY[F_FACT_ARR_COUNT];
__constant__ float F_FRAC_ARRAY[F_FRAC_ARR_COUNT];

constexpr int32_t D_FACT_ARR_COUNT = 19;
constexpr int32_t D_FRAC_ARR_COUNT = 21;
__constant__ int64_t D_FACT_ARRAY[D_FACT_ARR_COUNT];
__constant__ double D_FRAC_ARRAY[D_FRAC_ARR_COUNT];

__host__ void load_alp_constants() {
  cudaMemcpyToSymbol(constant_memory::F_FACT_ARRAY,
                     alp::Constants<float>::FACT_ARR,
                     F_FACT_ARR_COUNT * sizeof(int32_t));
  cudaMemcpyToSymbol(constant_memory::F_FRAC_ARRAY,
                     alp::Constants<float>::FRAC_ARR,
                     F_FACT_ARR_COUNT * sizeof(float));

  cudaMemcpyToSymbol(constant_memory::D_FACT_ARRAY,
                     alp::Constants<double>::FACT_ARR,
                     D_FACT_ARR_COUNT * sizeof(int64_t));
  cudaMemcpyToSymbol(constant_memory::D_FRAC_ARRAY,
                     alp::Constants<double>::FRAC_ARR,
                     D_FRAC_ARR_COUNT * sizeof(double));
}

template <typename T> __device__ __forceinline__ T *get_frac_arr();
template <> __device__ __forceinline__ float *get_frac_arr() {
  return F_FRAC_ARRAY;
}
template <> __device__ __forceinline__ double *get_frac_arr() {
  return D_FRAC_ARRAY;
}

template <typename T> __device__ __forceinline__ T *get_fact_arr();
template <> __device__ __forceinline__ int32_t *get_fact_arr() {
  return F_FACT_ARRAY;
}
template <> __device__ __forceinline__ int64_t *get_fact_arr() {
  return D_FACT_ARRAY;
}
} // namespace constant_memory

// WARNING
// WARNING
// TODO WARNING IS IT NOT FASTER TO PASS THESE ARGUMENTS IN FULL WIDTH?

// SO uint8_T -> uint32_t (if it gets multiplied with 32) This saves a cast
// in each kernel, and we do not care how big parameters are, as they are
// passed via const
// INFO Hypothesis: not stalling on arithmetic, so it does not matter in
// execution time. Check # executed instructions tho.
// WARNING
// WARNING
template <typename T_in, typename T_out, UnpackingType unpacking_type,
          unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
__device__ void alp_vector(T_out *__restrict out, const AlpColumn<T_out> column,
                           const uint16_t vector_index, const uint16_t lane,
                           const uint16_t start_index) {
  static_assert((std::is_same<T_in, uint32_t>::value &&
                 std::is_same<T_out, float>::value) ||
                    (std::is_same<T_in, uint64_t>::value &&
                     std::is_same<T_out, double>::value),
                "Wrong type arguments");
  using INT_T = typename utils::same_width_int<T_out>::type;
  using UINT_T = typename utils::same_width_int<T_out>::type;

  T_in *in = column.ffor_array + consts::VALUES_PER_VECTOR * vector_index;
  uint16_t value_bit_width = column.bit_widths[vector_index];
  UINT_T base = column.ffor_bases[vector_index];
  INT_T factor =
      constant_memory::get_fact_arr<INT_T>()[column.factors[vector_index]];
  T_out frac10 = constant_memory::get_frac_arr<
      T_out>()[column.exponents[vector_index]]; // WARNING TODO implement a
                                                // compile time switch to grab
                                                // float array
  auto lambda = [base, factor, frac10](const T_in value) -> T_out {
    return T_out{static_cast<INT_T>((value + base) *
                                    static_cast<UINT_T>(factor))} *
           frac10;
  };

  unpack_vector<T_in, T_out, unpacking_type, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
      in, out, lane, value_bit_width, start_index, lambda);

  // Patch exceptions
  auto n_lanes = utils::get_n_lanes<INT_T>();
  auto exceptions_count = column.counts[vector_index];

  auto vec_exceptions =
      column.exceptions + consts::VALUES_PER_VECTOR * vector_index;
  auto vec_exceptions_positions =
      column.positions + consts::VALUES_PER_VECTOR * vector_index;
  for (int i{lane}; i < exceptions_count; i += n_lanes) {
    // WARNING Currently assumes that you are decoding an entire vector
    // TODO Implement an if (position > startindex && position < (start_index +
    // UNPACK_N_VALUES * n_lanes) {...}
    auto position = vec_exceptions_positions[i];
    out[position] = vec_exceptions[i];
  }
}

template <typename T_in, typename T_out, UnpackingType unpacking_type,
          unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
__device__ void alprd_vector(T_out *__restrict out, const AlpRdColumn<T_out> column,
                           const uint16_t vector_index, const uint16_t lane,
                           const uint16_t start_index) {
  static_assert((std::is_same<T_in, uint32_t>::value &&
                 std::is_same<T_out, float>::value) ||
                    (std::is_same<T_in, uint64_t>::value &&
                     std::is_same<T_out, double>::value),
                "Wrong type arguments");
  using INT_T = typename utils::same_width_int<T_out>::type;
  using UINT_T = typename utils::same_width_int<T_out>::type;

}

#endif // ALP_CUH
