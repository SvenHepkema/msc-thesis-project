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
