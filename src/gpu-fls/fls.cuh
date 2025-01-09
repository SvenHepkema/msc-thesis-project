#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "../common/utils.hpp"
#include "../gpu-common/gpu-device-types.cuh"

#ifndef FLS_CUH
#define FLS_CUH

template <typename T>
__device__ __forceinline__ constexpr T c_set_first_n_bits(const T count) {
  static_assert(std::is_unsigned<T>::value, "Should be unsigned");
  return (1U << count) - 1;
}

template <typename T> struct BPFunctor {
  using UINT_T = typename utils::same_width_uint<T>::type;
  __device__ __forceinline__ BPFunctor(){};
  __device__ __forceinline__ T operator()(const UINT_T value) const {
    return value;
  }
};

template <typename T> struct FFORFunctor {
  using UINT_T = typename utils::same_width_uint<T>::type;
  const T base;
  __device__ __forceinline__ FFORFunctor(const T base) : base(base){};
  __device__ __forceinline__ T operator()(const UINT_T value) const {
    return value + base;
  }
};

template <typename T> struct BitUnpackerBase {
  virtual __device__ __forceinline__ void unpack_next_into(T *__restrict out);
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename lambda_T>
__device__ void unpack_vector_stateless_old(
    const typename utils::same_width_uint<T>::type *__restrict in,
    T *__restrict out, const lane_t lane, const vbw_t value_bit_width,
    const si_t start_index, lambda_T lambda) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  static_assert(std::is_unsigned<UINT_T>::value,
                "Packing function only supports unsigned types. Cast signed "
                "arrays to unsigned equivalent.");
  constexpr uint8_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<UINT_T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<UINT_T>();
  uint16_t preceding_bits = (start_index * value_bit_width);
  uint16_t buffer_offset = preceding_bits % LANE_BIT_WIDTH;
  uint16_t n_input_line = preceding_bits / LANE_BIT_WIDTH;
  UINT_T value_mask = utils::set_first_n_bits<UINT_T>(value_bit_width);

  UINT_T line_buffer[UNPACK_N_VECTORS];
  UINT_T buffer_offset_mask;

  // WARNING This causes quite some latency, test replacing it with a
  // constant memory table lookup
  // INFO Constant memory table lookup might be applicable in more places
  // in this function.
  int32_t encoded_vector_offset =
      utils::get_compressed_vector_size<UINT_T>(value_bit_width);

  in += lane;

#pragma unroll
  for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
    line_buffer[v] = *(in + n_input_line * N_LANES + v * encoded_vector_offset);
  }
  n_input_line++;

  UINT_T value[UNPACK_N_VECTORS];

#pragma unroll
  for (int i = 0; i < UNPACK_N_VALUES; ++i) {
    bool line_buffer_is_empty = buffer_offset == LANE_BIT_WIDTH;
    if (line_buffer_is_empty) {
#pragma unroll
      for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
        line_buffer[v] =
            *(in + n_input_line * N_LANES + v * encoded_vector_offset);
      }
      ++n_input_line;
      buffer_offset -= LANE_BIT_WIDTH;
    }

#pragma unroll
    for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
      value[v] =
          (line_buffer[v] & (value_mask << buffer_offset)) >> buffer_offset;
    }
    buffer_offset += value_bit_width;

    bool value_continues_on_next_line = buffer_offset > LANE_BIT_WIDTH;
    if (value_continues_on_next_line) {
#pragma unroll
      for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
        line_buffer[v] =
            *(in + n_input_line * N_LANES + v * encoded_vector_offset);
      }
      ++n_input_line;
      buffer_offset -= LANE_BIT_WIDTH;

      buffer_offset_mask =
          (UINT_T{1} << static_cast<UINT_T>(buffer_offset)) - UINT_T{1};
#pragma unroll
      for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
        value[v] |= (line_buffer[v] & buffer_offset_mask)
                    << (value_bit_width - buffer_offset);
      }
    }

#pragma unroll
    for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
      *(out + v * UNPACK_N_VALUES) = lambda(value[v]);
    }
    ++out;
  }
}

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename OutputProcessor>
struct BitUnpackerStatelessOld : BitUnpackerBase<T> {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const UINT_T *__restrict in;
  const lane_t lane;
  const vbw_t value_bit_width;
  OutputProcessor lambda;

  si_t start_index = 0;

  __device__ __forceinline__
  BitUnpackerStatelessOld(const UINT_T *__restrict in, const lane_t lane,
                          const vbw_t value_bit_width, OutputProcessor lambda)
      : in(in), lane(lane), value_bit_width(value_bit_width), lambda(lambda) {}

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) override {
    unpack_vector_stateless_old<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
        in, out, lane, value_bit_width, start_index, lambda);
    start_index += UNPACK_N_VALUES;
  }
};

template <typename T, unsigned UNPACK_N_VECTORS> struct SimpleLoader {
  T buffers[UNPACK_N_VECTORS];
  const T *in;
  int32_t vector_offset;

  __device__ __forceinline__ SimpleLoader(const T *in,
                                          const int32_t vector_offset)
      : in(in), vector_offset(vector_offset) {
    next_line();
  };

  __device__ __forceinline__ const T *get() const { return buffers; }

  __device__ __forceinline__ void next_line() {

#pragma unroll
    for (int v{0}; v < UNPACK_N_VECTORS; ++v) {
      buffers[v] = *(in + v * vector_offset);
    }
    in += utils::get_n_lanes<T>();
  }
};

template <typename T, unsigned UNPACK_N_VECTORS> struct Masker {
  const vbw_t value_bit_width;
  const T value_mask;
  uint16_t buffer_offset = 0;

  __device__ __forceinline__ Masker(const vbw_t value_bit_width)
      : value_bit_width(value_bit_width),
        value_mask(utils::set_first_n_bits<T>(value_bit_width)){};

  __device__ __forceinline__ Masker(const uint16_t buffer_offset,
                                    const vbw_t value_bit_width)
      : buffer_offset(buffer_offset), value_bit_width(value_bit_width),
        value_mask(utils::set_first_n_bits<T>(value_bit_width)){};

  __device__ __forceinline__ void mask_and_increment(T *values,
                                                     const T *buffers) {
#pragma unroll
    for (int v{0}; v < UNPACK_N_VECTORS; ++v) {
      values[v] = (buffers[v] & (value_mask << buffer_offset)) >> buffer_offset;
    }
    buffer_offset += value_bit_width;
  }

  __device__ __forceinline__ void next_line() {
    buffer_offset -= utils::get_lane_bitwidth<T>();
  }
  __device__ __forceinline__ bool is_buffer_empty() const {
    return buffer_offset == utils::get_lane_bitwidth<T>();
  }

  __device__ __forceinline__ bool continues_on_next_line() const {
    return buffer_offset > utils::get_lane_bitwidth<T>();
  }

  __device__ __forceinline__ void
  mask_and_insert_remaining_value(T *values, const T *buffers) const {
    T buffer_offset_mask = (T{1} << static_cast<T>(buffer_offset)) - T{1};

#pragma unroll
    for (int v{0}; v < UNPACK_N_VECTORS; ++v) {
      values[v] |= (buffers[v] & buffer_offset_mask)
                   << (value_bit_width - buffer_offset);
    }
  }
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename lambda_T>
__device__ void unpack_vector_stateless(
    const typename utils::same_width_uint<T>::type *__restrict in,
    T *__restrict out, const lane_t lane, const vbw_t value_bit_width,
    const si_t start_index, lambda_T lambda) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  constexpr uint8_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<UINT_T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<UINT_T>();
  uint16_t preceding_bits = (start_index * value_bit_width);
  uint16_t buffer_offset = preceding_bits % LANE_BIT_WIDTH;
  uint16_t n_input_line = preceding_bits / LANE_BIT_WIDTH;

  SimpleLoader<UINT_T, UNPACK_N_VECTORS> loader(
      in + n_input_line * N_LANES + lane,
      utils::get_compressed_vector_size<UINT_T>(value_bit_width));
  Masker<UINT_T, UNPACK_N_VECTORS> masker(buffer_offset, value_bit_width);

  UINT_T values[UNPACK_N_VECTORS];

#pragma unroll
  for (int i = 0; i < UNPACK_N_VALUES; ++i) {
    if (masker.is_buffer_empty()) {
      loader.next_line();
      masker.next_line();
    }

    masker.mask_and_increment(values, loader.get());

    if (masker.continues_on_next_line()) {
      loader.next_line();
      masker.next_line();
      masker.mask_and_insert_remaining_value(values, loader.get());
    }

#pragma unroll
    for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
      *(out + i + v * UNPACK_N_VALUES) = lambda(values[v]);
    }
  }
}

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename OutputProcessor>
struct BitUnpackerStateless : BitUnpackerBase<T> {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const UINT_T *__restrict in;
  const lane_t lane;
  const vbw_t value_bit_width;
  OutputProcessor lambda;

  si_t start_index = 0;

  __device__ __forceinline__ BitUnpackerStateless(const UINT_T *__restrict in,
                                                  const lane_t lane,
                                                  const vbw_t value_bit_width,
                                                  OutputProcessor lambda)
      : in(in), lane(lane), value_bit_width(value_bit_width), lambda(lambda) {}

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) override {
    unpack_vector_stateless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
        in, out, lane, value_bit_width, start_index, lambda);
    start_index += UNPACK_N_VALUES;
  }
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename lambda_T>
__device__ void unpack_vector_stateless_branchless(
    const typename utils::same_width_uint<T>::type *__restrict in,
    T *__restrict out, const lane_t lane, const vbw_t value_bit_width,
    const si_t start_index, lambda_T lambda) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  constexpr int32_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<UINT_T>();
  constexpr int32_t N_LANES = utils::get_n_lanes<UINT_T>();
  constexpr int32_t BIT_COUNT = utils::sizeof_in_bits<T>();

  int32_t preceding_bits_first = (start_index * value_bit_width);
  int32_t n_input_line = preceding_bits_first / LANE_BIT_WIDTH;
  int32_t offset_first = preceding_bits_first % LANE_BIT_WIDTH;
  int32_t offset_second = BIT_COUNT - offset_first;
  UINT_T value_mask = c_set_first_n_bits(value_bit_width);

  UINT_T value = 0;

  in += n_input_line * N_LANES + lane;
  value |= (in[0] & (value_mask << offset_first)) >> offset_first;
  value |= (in[N_LANES] & (value_mask >> offset_second)) << offset_second;

  *(out) = lambda(value);
}

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename OutputProcessor>
struct BitUnpackerStatelessBranchless : BitUnpackerBase<T> {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const UINT_T *__restrict in;
  const lane_t lane;
  const vbw_t value_bit_width;
  OutputProcessor lambda;

  si_t start_index = 0;

  __device__ __forceinline__ BitUnpackerStatelessBranchless(
      const UINT_T *__restrict in, const lane_t lane,
      const vbw_t value_bit_width, OutputProcessor lambda)
      : in(in), lane(lane), value_bit_width(value_bit_width), lambda(lambda) {}

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) override {
    unpack_vector_stateless_branchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
        in, out, lane, value_bit_width, start_index, lambda);
    start_index += UNPACK_N_VALUES;
  }
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename OutputProcessor>
struct BitUnpackerStateful : BitUnpackerBase<T> {
  using UINT_T = typename utils::same_width_uint<T>::type;
  SimpleLoader<UINT_T, UNPACK_N_VECTORS> loader;
  Masker<UINT_T, UNPACK_N_VECTORS> masker;
  const OutputProcessor processor;

  __device__ __forceinline__ BitUnpackerStateful(const UINT_T *__restrict in,
                                                 const lane_t lane,
                                                 const vbw_t value_bit_width,
                                                 OutputProcessor processor)
      : loader(in + lane,
               utils::get_compressed_vector_size<UINT_T>(value_bit_width)),
        masker(value_bit_width), processor(processor) {}

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) override {
    UINT_T values[UNPACK_N_VECTORS];

#pragma unroll
    for (int i = 0; i < UNPACK_N_VALUES; ++i) {
      if (masker.is_buffer_empty()) {
        loader.next_line();
        masker.next_line();
      }

      masker.mask_and_increment(values, loader.get());

      if (masker.continues_on_next_line()) {
        loader.next_line();
        masker.next_line();
        masker.mask_and_insert_remaining_value(values, loader.get());
      }

#pragma unroll
      for (int v{0}; v < UNPACK_N_VECTORS; ++v) {
        *(out + i + v * UNPACK_N_VALUES) = processor(values[v]);
      }
    }
  }
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename OutputProcessor>
struct BitUnpackerStatefulBranchless : BitUnpackerBase<T> {
  using UINT_T = typename utils::same_width_uint<T>::type;
  const OutputProcessor processor;

  __device__ __forceinline__ BitUnpackerStatefulBranchless(
      const UINT_T *__restrict in, const lane_t lane,
      const vbw_t value_bit_width, OutputProcessor processor)
      : processor(processor) {}

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) override {
    // TODO Implement branchless, but keep certain variables in between calls
  }
};

#endif // FLS_CUH
