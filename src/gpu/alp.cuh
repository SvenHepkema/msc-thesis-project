#include <cstdint>
#include <type_traits>

#include "fls.cuh"

#ifndef ALP_CUH
#define ALP_CUH

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
__device__ void falp_vector(const T_in *__restrict in, T_out *__restrict out,
                            const uint16_t lane, const uint16_t value_bit_width,
                            const uint16_t start_index,
                            const T_in *__restrict a_base_p, const uint8_t fac,
                            const uint8_t exp) {
  static_assert((std::is_same<T_in, uint32_t>::value &&
                 std::is_same<T_out, float>::value) ||
                    (std::is_same<T_in, uint64_t>::value &&
                     std::is_same<T_out, double>::value),
                "Wrong type arguments");
  using INT_T =
      typename std::conditional<sizeof(T_out) == 4, int32_t, int64_t>::type;
  using FP_T =
      typename std::conditional<sizeof(T_out) == 4, float, double>::type;

  T_out base = *a_base_p;
  INT_T factor = fac; // WARNING TODO Do array indirection
  FP_T frac10 = exp;   // WARNING TODO Do array indirection

  auto lambda = [base, factor, frac10](const T_in value) -> T_out {
    return FP_T{(value + base) * factor} * frac10;
  };
  unpack_vector<T_in, T_out, unpacking_type, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
      in, out, lane, value_bit_width, start_index, lambda);
}

#endif // ALP_CUH
