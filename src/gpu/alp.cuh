#include <cstdint>
#include <type_traits>

#include "../alp/constants.hpp"
#include "fls.cuh"

#ifndef ALP_CUH
#define ALP_CUH

template <typename T> struct AlpData {
  using UINT_T =
      typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;
  UINT_T *ffor_array;
  UINT_T *ffor_bases;
  uint8_t *bit_widths;
  uint8_t *exponents;
  uint8_t *factors;
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
__device__ void alp_vector(T_out *__restrict out, const AlpData<T_out> data,
                           const uint16_t vector_index, const uint16_t lane,
                           const uint16_t start_index) {
  static_assert((std::is_same<T_in, uint32_t>::value &&
                 std::is_same<T_out, float>::value) ||
                    (std::is_same<T_in, uint64_t>::value &&
                     std::is_same<T_out, double>::value),
                "Wrong type arguments");
  using INT_T =
      typename std::conditional<sizeof(T_out) == 4, int32_t, int64_t>::type;

	T_in* in = data.ffor_array + consts::VALUES_PER_VECTOR * vector_index;
	uint16_t value_bit_width = data.bit_widths[vector_index];
  T_in base = data.ffor_bases[vector_index];
  INT_T factor = constant_memory::FACT_ARR[data.factors[vector_index]];
  T_out frac10 = constant_memory::FRAC_ARR_D[data.exponents[vector_index]]; // WARNING TODO implement a
                                                   // switch to grab float array
  auto lambda = [base, factor, frac10](const T_in value) -> T_out {
    return T_out{(value + base) * factor} * frac10;
  };

  unpack_vector<T_in, T_out, unpacking_type, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
      in, out, lane, value_bit_width, start_index, lambda);
}

#endif // ALP_CUH
