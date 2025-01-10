#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "../alp/constants.hpp"
#include "../common/utils.hpp"
#include "../gpu-fls/fls.cuh"
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

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename UnpackerT, typename PatcherT, typename ColumnT>
struct AlpUnpacker {
  UnpackerT unpacker;
  PatcherT patcher;

  __device__ __forceinline__ AlpUnpacker(const ColumnT column,
                                         const vi_t vector_index,
                                         const lane_t lane)
      : unpacker(UnpackerT(column.ffor_array +
                               consts::VALUES_PER_VECTOR * vector_index,
                           lane, column.bit_widths[vector_index],
                           ALPFunctor<T>(column.ffor_bases[vector_index],
                                         column.factors[vector_index],
                                         column.exponents[vector_index]))),
        patcher(PatcherT(column, vector_index, lane)) {}

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) {
    unpacker.unpack_next_into(out);
    patcher.patch_if_needed(out);
  }
};

template <typename T, unsigned UNPACK_N_VECTORS>
struct ALPExceptionPatcherBase {
public:
  virtual void __device__ __forceinline__ patch_if_needed(T *out);
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatelessALPExceptionPatcher
    : ALPExceptionPatcherBase<T, UNPACK_N_VECTORS> {
  using INT_T = typename utils::same_width_int<T>::type;

  si_t start_index = 0;
  const uint16_t exceptions_count;
  const uint16_t *vec_exceptions_positions;
  const T *vec_exceptions;
  const lane_t lane;

public:
  void __device__ __forceinline__ patch_if_needed(T *out) override {
    constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

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

  __device__ __forceinline__ StatelessALPExceptionPatcher(
      const AlpColumn<T> column, const vi_t vector_index, const lane_t lane)
      : exceptions_count(column.counts[vector_index]),
        vec_exceptions_positions(column.positions +
                                 consts::VALUES_PER_VECTOR * vector_index),
        vec_exceptions(column.exceptions +
                       consts::VALUES_PER_VECTOR * vector_index),
        lane(lane) {}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatelessWithScannerALPExceptionPatcher
    : ALPExceptionPatcherBase<T, UNPACK_N_VECTORS> {
  using INT_T = typename utils::same_width_int<T>::type;

  si_t start_index = 0;
  const uint16_t exceptions_count;
  const uint16_t *vec_exceptions_positions;
  const T *vec_exceptions;
  const lane_t lane;

public:
  void __device__ __forceinline__ patch_if_needed(T *out) override {
    constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

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

  __device__ __forceinline__ StatelessWithScannerALPExceptionPatcher(
      const AlpColumn<T> column, const vi_t vector_index, const lane_t lane)
      : exceptions_count(column.counts[vector_index]),
        vec_exceptions_positions(column.positions +
                                 consts::VALUES_PER_VECTOR * vector_index),
        vec_exceptions(column.exceptions +
                       consts::VALUES_PER_VECTOR * vector_index),
        lane(lane) {}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatefulALPExceptionPatcher
    : ALPExceptionPatcherBase<T, UNPACK_N_VECTORS> {
  using INT_T = typename utils::same_width_int<T>::type;

  si_t start_index = 0;
  const uint16_t exceptions_count;
  const uint16_t *vec_exceptions_positions;
  const T *vec_exceptions;
  const lane_t lane;
  int32_t exception_index = 0;

public:
  void __device__ __forceinline__ patch_if_needed(T *out) override {
    constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

    const int first_pos = start_index * N_LANES + lane;
    const int last_pos = first_pos + N_LANES * (UNPACK_N_VALUES - 1);
    start_index += UNPACK_N_VALUES;

    for (; exception_index < exceptions_count; exception_index++) {
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

  __device__ __forceinline__ StatefulALPExceptionPatcher(
      const AlpColumn<T> column, const vi_t vector_index, const lane_t lane)
      : exceptions_count(column.counts[vector_index]),
        vec_exceptions_positions(column.positions +
                                 consts::VALUES_PER_VECTOR * vector_index),
        vec_exceptions(column.exceptions +
                       consts::VALUES_PER_VECTOR * vector_index),
        lane(lane) {}
};

template <typename T, unsigned UNPACK_N_VECTORS>
struct SimpleALPExceptionPatcher
    : ALPExceptionPatcherBase<T, UNPACK_N_VECTORS> {
private:
  uint16_t count[UNPACK_N_VECTORS];
  uint16_t *positions[UNPACK_N_VECTORS];
  T *exceptions[UNPACK_N_VECTORS];
  uint16_t position;

public:
  __device__ __forceinline__
  SimpleALPExceptionPatcher(const AlpExtendedColumn<T> column,
                            const vi_t vector_index, const lane_t lane)
      : position(lane) {
#pragma unroll
    for (int v{0}; v < UNPACK_N_VECTORS; ++v) {
      const vi_t current_vector_index = vector_index + v;

      const auto offset_count =
          column.offsets_counts[current_vector_index * utils::get_n_lanes<T>() +
                                lane];
      count[v] = offset_count >> 10;

      const auto offset = (offset_count & 0x3FF);
      // These consts should be N_LANES, but the arrays are not packed yet
      positions[v] = column.positions +
                     consts::VALUES_PER_VECTOR * current_vector_index + offset;
      exceptions[v] = column.exceptions +
                      consts::VALUES_PER_VECTOR * current_vector_index + offset;
    }
  }

  void __device__ __forceinline__ patch_if_needed(T *out) override {
#pragma unroll
    for (int v{0}; v < UNPACK_N_VECTORS; ++v) {
      if (count[v] > 0) {
        if (position == *(positions[v])) {
          out[v] = *(exceptions[v]);
          ++(positions[v]);
          ++(exceptions[v]);
          --(count[v]);
        }
        position += utils::get_n_lanes<T>();
      }
    }
  }
};

template <typename T, unsigned UNPACK_N_VECTORS>
struct PrefetchPositionALPExceptionPatcher
    : ALPExceptionPatcherBase<T, UNPACK_N_VECTORS> {
private:
  uint16_t count;
  uint16_t *positions;
  T *exceptions;
  uint16_t next_position;

public:
  __device__ __forceinline__ PrefetchPositionALPExceptionPatcher(
      const AlpExtendedColumn<T> column, const vi_t vector_index,
      const lane_t lane) {
    auto offset_count = column.offsets_counts[vector_index];
    count = offset_count >> 10;
    positions = column.positions + vector_index * utils::get_n_lanes<T>() +
                (offset_count & 0x3FF);
    exceptions = column.exceptions + vector_index * utils::get_n_lanes<T>() +
                 (offset_count & 0x3FF);
    next_position = *positions;
  }

  void __device__ __forceinline__
  patch_if_needed(T *out, const int32_t position) override {
    if (count > 0 && position == next_position) {
      *out = *exceptions;
      ++positions;
      ++exceptions;
      --count;
      next_position = *positions;
    }
  }
};

template <typename T, unsigned UNPACK_N_VECTORS>
struct PrefetchAllALPExceptionPatcher
    : ALPExceptionPatcherBase<T, UNPACK_N_VECTORS> {
private:
  uint16_t count;
  uint16_t *positions;
  T *exceptions;

  uint16_t index = 0;
  int16_t next_position;
  T next_exception;

public:
  void __device__ __forceinline__ read_next_exception() {
    if (index < count) {
      next_position = *positions;
      next_exception = *exceptions;
      ++positions;
      ++exceptions;
      ++index;
    } else {
      next_position = -1;
    }
  }

  __device__ __forceinline__ PrefetchAllALPExceptionPatcher(
      const AlpColumn<T> column, const vi_t vector_index, const lane_t lane) {
    auto offset_count = column.offsets_counts[vector_index];
    count = offset_count >> 10;

    positions = column.positions + vector_index * utils::get_n_lanes<T>() +
                (offset_count & 0x3FF);
    exceptions = column.exceptions + vector_index * utils::get_n_lanes<T>() +
                 (offset_count & 0x3FF);

    next_position = *positions;
    next_exception = *exceptions;

    read_next_exception();
  }

  void __device__ __forceinline__
  patch_if_needed(T *out, const int32_t position) override {
    if (position == next_position) {
      *out = next_exception;
      read_next_exception();
    }
  }
};

#endif // ALP_CUH
