#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "../alp/constants.hpp"
#include "../common/utils.hpp"
#include "../gpu-fls/fls.cuh"

#ifndef ALP_EXCEPTION_PATCHERS_CUH
#define ALP_EXCEPTION_PATCHERS_CUH

template <typename T, unsigned UNPACK_N_VECTORS>
struct ALPExceptionPatcherBase {
public:
  virtual void __device__ __forceinline__ patch_if_needed(T *out);
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatelessALPExceptionPatcher
    : ALPExceptionPatcherBase<T, UNPACK_N_VECTORS> {
  using INT_T = typename utils::same_width_int<T>::type;

  const si_t start_index;
  const uint16_t exceptions_count;
  const uint16_t *vec_exceptions_positions;
  const T *vec_exceptions;
  const lane_t lane;

public:
  void __device__ __forceinline__ patch_if_needed(T *out) override {
    constexpr auto N_LANES = utils::get_n_lanes<INT_T>();

    const int first_pos = start_index * N_LANES + lane;
    const int last_pos = first_pos + N_LANES * (UNPACK_N_VALUES - 1);
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
      const vi_t vector_index, const si_t start_index,
      const uint16_t *offset_counts, const uint16_t *vec_exceptions_positions,
      const T *vec_exceptions, const lane_t lane)
      : start_index(start_index), exceptions_count(offset_counts[vector_index]),
        vec_exceptions_positions(vec_exceptions_positions +
                                 consts::VALUES_PER_VECTOR * vector_index),
        vec_exceptions(vec_exceptions +
                       consts::VALUES_PER_VECTOR * vector_index),
        lane(lane) {}
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct StatelessWithScannerALPExceptionPatcher
    : ALPExceptionPatcherBase<T, UNPACK_N_VECTORS> {
  using INT_T = typename utils::same_width_int<T>::type;

  const si_t start_index;
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
      const vi_t vector_index, const si_t start_index,
      const uint16_t *offset_counts, const uint16_t *vec_exceptions_positions,
      const T *vec_exceptions, const lane_t lane)
      : start_index(start_index), exceptions_count(offset_counts[vector_index]),
        vec_exceptions_positions(vec_exceptions_positions +
                                 consts::VALUES_PER_VECTOR * vector_index),
        vec_exceptions(vec_exceptions +
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
  __device__ __forceinline__ SimpleALPExceptionPatcher(
      const uint16_t *offset_counts, uint16_t *vec_exceptions_positions,
      T *vec_exceptions, const lane_t lane)
      : position(lane) {
#pragma unroll
    for (int v{0}; v < UNPACK_N_VECTORS; ++v) {
      auto offset_count = offset_counts[v * utils::get_n_lanes<T>()];
      count[v] = offset_count >> 10;

      auto v_offset = v * consts::VALUES_PER_VECTOR + (offset_count & 0x3FF);
      positions[v] = vec_exceptions_positions + v_offset;
      exceptions[v] = vec_exceptions + v_offset;
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
      const uint16_t offset_count, uint16_t *vec_exceptions_positions,
      T *vec_exceptions)
      : count(offset_count >> 10),
        positions(vec_exceptions_positions + (offset_count & 0x3FF)),
        exceptions(vec_exceptions + (offset_count & 0x3FF))

  {
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
      const uint16_t offset_count, uint16_t *vec_exceptions_positions,
      T *vec_exceptions)
      : count(offset_count >> 10),
        positions(vec_exceptions_positions + (offset_count & 0x3FF)),
        exceptions(vec_exceptions + (offset_count & 0x3FF))

  {
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

#endif // ALP_EXCEPTION_PATCHERS_CUH
