#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "../alp/constants.hpp"
#include "../common/utils.hpp"
#include "../gpu-fls/fls.cuh"
#include "alp-exception-patchers.cuh"
#include "src/alp/config.hpp"

#ifndef ALP_CUH
#define ALP_CUH

template <typename T> struct AlpColumn {
  using UINT_T = typename utils::same_width_uint<T>::type;

  UINT_T *ffor_array;
  UINT_T *ffor_bases;
  uint8_t *bit_widths; // INFO Would it be faster to bitpack theses three
                       // uint8_t into a 16 or 32?
  uint8_t *exponents;  // INFO For float this can be bitpacked with exponent and
                       // factor
  uint8_t *factors;

  T *exceptions;
  uint16_t *positions;
  uint16_t *counts;
};

template <typename T> struct AlpExtendedColumn {
  using UINT_T = typename utils::same_width_uint<T>::type;

  UINT_T *ffor_array;
  UINT_T *ffor_bases;
  uint8_t *bit_widths;
  uint8_t *exponents;
  uint8_t *factors;

  T *exceptions;
  uint16_t *positions;

  // Length: vecs * n_lanes
  // Bitpacked count (5 bits) + offset (10 bits)
  // offset = offsets_counts & 0x3FF
  // count = offsets_counts >> 10
  uint16_t *offsets_counts;
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

template <typename T> __host__ void load_alp_constants() {
  cudaMemcpyToSymbol(F_FACT_ARRAY, alp::Constants<float>::FACT_ARR,
                     F_FACT_ARR_COUNT * sizeof(int32_t));
  cudaMemcpyToSymbol(F_FRAC_ARRAY, alp::Constants<float>::FRAC_ARR,
                     F_FRAC_ARR_COUNT * sizeof(float));

  cudaMemcpyToSymbol(D_FACT_ARRAY, alp::Constants<double>::FACT_ARR,
                     D_FACT_ARR_COUNT * sizeof(int64_t));
  cudaMemcpyToSymbol(D_FRAC_ARRAY, alp::Constants<double>::FRAC_ARR,
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

template <typename T> struct ALPFunctor {
private:
  using INT_T = typename utils::same_width_int<T>::type;
  using UINT_T = typename utils::same_width_uint<T>::type;

  UINT_T base;
  INT_T factor;
  T frac10;

public:
  __device__ __forceinline__ ALPFunctor(const UINT_T base,
                                        const uint8_t factor_index,
                                        const uint8_t frac10_index)
      : base(base),
        factor(constant_memory::get_fact_arr<INT_T>()[factor_index]),
        frac10(constant_memory::get_frac_arr<T>()[frac10_index]) {}

  T __device__ __forceinline__ operator()(UINT_T value) const {
    return static_cast<T>(static_cast<INT_T>((value + base) * factor)) * frac10;
  }
};

// WARNING Never uses this baseclass directly, virtual function calls are
// horribly slow. The reason this class is here, is to ensure that all of the
// unpackers conform to the same API
template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct AlpUnpackerBase {
  virtual __device__ __forceinline__ void unpack_next_into(T *__restrict out);
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct AlpStatelessUnpacker
    : AlpUnpackerBase<T, UNPACK_N_VECTORS, UNPACK_N_VALUES> {

private:
  const AlpColumn<T> column;
  const vi_t vector_index;
  const lane_t lane;
  si_t start_index = 0;

public:
  __device__ void unpack_next_into(T *__restrict out) override {
    using INT_T = typename utils::same_width_int<T>::type;
    using UINT_T = typename utils::same_width_uint<T>::type;

    UINT_T *in = column.ffor_array + consts::VALUES_PER_VECTOR * vector_index;
    vbw_t value_bit_width = column.bit_widths[vector_index];
    UINT_T base = column.ffor_bases[vector_index];
    INT_T factor =
        constant_memory::get_fact_arr<INT_T>()[column.factors[vector_index]];
    T frac10 = constant_memory::get_frac_arr<
        T>()[column.exponents[vector_index]]; // WARNING TODO implement a
                                              // compile time switch to grab
                                              // float array
    auto lambda = [base, factor, frac10](const UINT_T value) -> T {
      return static_cast<T>(static_cast<INT_T>((value + base) *
                                               static_cast<UINT_T>(factor))) *
             frac10;
    };

    unpack_vector<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
        in, out, lane, value_bit_width, start_index, lambda);

    // Patch exceptions
    constexpr auto N_LANES = utils::get_n_lanes<INT_T>();
    auto exceptions_count = column.counts[vector_index];

    auto vec_exceptions =
        column.exceptions + consts::VALUES_PER_VECTOR * vector_index;
    auto vec_exceptions_positions =
        column.positions + consts::VALUES_PER_VECTOR * vector_index;

    const int first_pos = start_index * N_LANES + lane;
    const int last_pos = first_pos + N_LANES * (UNPACK_N_VALUES - 1);
    start_index += UNPACK_N_VALUES;
    for (int i{0}; i < exceptions_count; i++) {
      auto position = vec_exceptions_positions[i];
      auto exception = vec_exceptions[i];
      if (position >= first_pos) {
        if (position <= last_pos && position % N_LANES == lane) {
          out[(position - first_pos) / N_LANES] = exception;
        }
        if (position + 1 > last_pos) {
          return;
        }
      }
    }
  }

  __device__ __forceinline__ AlpStatelessUnpacker(const AlpColumn<T> column,
                                                  const vi_t vector_index,
                                                  const lane_t lane)
      : column(column), vector_index(vector_index), lane(lane){};
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct AlpStatelessWithScannerUnpacker
    : AlpUnpackerBase<T, UNPACK_N_VECTORS, UNPACK_N_VALUES> {

private:
  const AlpColumn<T> column;
  const vi_t vector_index;
  const lane_t lane;
  si_t start_index = 0;

public:
  __device__ void unpack_next_into(T *__restrict out) override {
    using INT_T = typename utils::same_width_int<T>::type;
    using UINT_T = typename utils::same_width_uint<T>::type;

    UINT_T *in = column.ffor_array + consts::VALUES_PER_VECTOR * vector_index;
    vbw_t value_bit_width = column.bit_widths[vector_index];
    UINT_T base = column.ffor_bases[vector_index];
    INT_T factor =
        constant_memory::get_fact_arr<INT_T>()[column.factors[vector_index]];
    T frac10 = constant_memory::get_frac_arr<
        T>()[column.exponents[vector_index]]; // WARNING TODO implement a
                                              // compile time switch to grab
                                              // float array
    auto lambda = [base, factor, frac10](const UINT_T value) -> T {
      return static_cast<T>(static_cast<INT_T>((value + base) *
                                               static_cast<UINT_T>(factor))) *
             frac10;
    };

    unpack_vector<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
        in, out, lane, value_bit_width, start_index, lambda);

    // Patch exceptions
    constexpr auto N_LANES = utils::get_n_lanes<INT_T>();
    auto exceptions_count = column.counts[vector_index];

    auto vec_exceptions =
        column.exceptions + consts::VALUES_PER_VECTOR * vector_index;
    auto vec_exceptions_positions =
        column.positions + consts::VALUES_PER_VECTOR * vector_index;

    const int first_pos = start_index * N_LANES + lane;
    const int last_pos = first_pos + N_LANES * (UNPACK_N_VALUES - 1);
    constexpr int32_t SCANNER_SIZE = 1;
    uint16_t scanner[SCANNER_SIZE];
    start_index += UNPACK_N_VALUES;

    for (int i{0}; i < exceptions_count; i += SCANNER_SIZE) {
      for (int j{0}; j < SCANNER_SIZE && j + i < exceptions_count; ++j) {
        scanner[j] = vec_exceptions_positions[j + i];
      }

      for (int j{0}; j < SCANNER_SIZE && j + i < exceptions_count; ++j) {
        auto position = scanner[j];
        if (position >= first_pos) {
          if (position <= last_pos && position % N_LANES == lane) {
            out[(position - first_pos) / N_LANES] = vec_exceptions[j + i];
          }
          if (position + 1 > last_pos) {
            return;
          }
        }
      }
    }
  }

  __device__ __forceinline__ AlpStatelessWithScannerUnpacker(
      const AlpColumn<T> column, const vi_t vector_index, const lane_t lane)
      : column(column), vector_index(vector_index), lane(lane){};
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct AlpStatefulUnpacker
    : AlpUnpackerBase<T, UNPACK_N_VECTORS, UNPACK_N_VALUES> {
  using INT_T = typename utils::same_width_int<T>::type;
  using UINT_T = typename utils::same_width_uint<T>::type;

  const lane_t lane;
  si_t start_index = 0;
  int32_t exception_index = 0;

  UINT_T *in;
  UINT_T base;
  vbw_t value_bit_width;
  INT_T factor;
  T frac10;

  uint16_t vec_exceptions_count;
  uint16_t *vec_exceptions_positions;
  T *vec_exceptions;

  __device__ __forceinline__ AlpStatefulUnpacker(const AlpColumn<T> column,
                                                 const vi_t vector_index,
                                                 const lane_t lane)
      : lane(lane) {
    in = column.ffor_array + consts::VALUES_PER_VECTOR * vector_index;
    value_bit_width = column.bit_widths[vector_index];
    base = column.ffor_bases[vector_index];
    factor =
        constant_memory::get_fact_arr<INT_T>()[column.factors[vector_index]];
    frac10 = constant_memory::get_frac_arr<T>()[column.exponents[vector_index]];
    vec_exceptions_count = column.counts[vector_index];

    vec_exceptions =
        column.exceptions + consts::VALUES_PER_VECTOR * vector_index;
    vec_exceptions_positions =
        column.positions + consts::VALUES_PER_VECTOR * vector_index;
  }

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) override {
    static_assert((std::is_same<UINT_T, uint32_t>::value &&
                   std::is_same<T, float>::value) ||
                      (std::is_same<UINT_T, uint64_t>::value &&
                       std::is_same<T, double>::value),
                  "Wrong type arguments");
    auto lambda = [base = this->base, factor = this->factor,
                   frac10 = this->frac10](const UINT_T value) -> T {
      return static_cast<T>(static_cast<INT_T>((value + base) * factor)) *
             frac10;
    };

    unpack_vector<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
        in, out, lane, value_bit_width, start_index, lambda);

    // Patch exceptions
    constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

    const int first_pos = start_index * N_LANES + lane;
    const int last_pos = first_pos + N_LANES * (UNPACK_N_VALUES - 1);

    start_index += UNPACK_N_VALUES;
    for (; exception_index < vec_exceptions_count; exception_index++) {
      auto position = vec_exceptions_positions[exception_index];
      auto exception = vec_exceptions[exception_index];
      if (position >= first_pos) {
        if (position <= last_pos && position % N_LANES == lane) {
          out[(position - first_pos) / N_LANES] = exception;
        }
        if (position + 1 > last_pos) {
          return;
        }
      }
    }
  }
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct AlpStatefulExtendedUnpacker
    : AlpUnpackerBase<T, UNPACK_N_VECTORS, UNPACK_N_VALUES> {
  using UINT_T = typename utils::same_width_uint<T>::type;
  UINT_T *in;
  vbw_t value_bit_width;

  BitUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ALPFunctor<T>> bit_unpacker;
  SimpleALPExceptionPatcher<T, UNPACK_N_VECTORS> patcher;

  __device__ __forceinline__
  AlpStatefulExtendedUnpacker(const AlpExtendedColumn<T> column,
                              const vi_t vector_index, const lane_t lane)
      : value_bit_width(column.bit_widths[vector_index]),
        in(column.ffor_array + consts::VALUES_PER_VECTOR * vector_index),
        patcher(column.offsets_counts +
                    (vector_index * utils::get_n_lanes<T>() + lane),
                column.positions + consts::VALUES_PER_VECTOR * vector_index,
                column.exceptions + consts::VALUES_PER_VECTOR * vector_index,
                lane),
        bit_unpacker(in, lane, value_bit_width,
                     ALPFunctor<T>(column.ffor_bases[vector_index],
                                   column.factors[vector_index],
                                   column.exponents[vector_index])) {}

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) override {
    bit_unpacker.unpack_into(out);
    patcher.patch_if_needed(out);
  }
};

#endif // ALP_CUH
