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

namespace constant_memory {
constexpr int32_t FACT_ARR_COUNT = 19;
constexpr int32_t FRAC_ARR_D_COUNT = 21;

__constant__ int64_t FACT_ARR[FACT_ARR_COUNT];
__constant__ double FRAC_ARR_D[FRAC_ARR_D_COUNT];

__host__ void load_alp_constants() {
  cudaMemcpyToSymbol(constant_memory::FACT_ARR, alp::FACT_ARR,
                     FACT_ARR_COUNT * sizeof(int64_t));
  cudaMemcpyToSymbol(constant_memory::FRAC_ARR_D,
                     alp::Constants<double>::FRAC_ARR,
                     FRAC_ARR_D_COUNT * sizeof(double));
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
  INT_T factor = constant_memory::FACT_ARR[column.factors[vector_index]];
  T_out frac10 = constant_memory::FRAC_ARR_D
      [column.exponents[vector_index]]; // WARNING TODO implement a
                                        // compile time switch to grab float
                                        // array
  auto lambda = [base, factor, frac10](const T_in value) -> T_out {
    return T_out{static_cast<INT_T>((value + base) * static_cast<UINT_T>(factor))} * frac10;
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

#endif // ALP_CUH
