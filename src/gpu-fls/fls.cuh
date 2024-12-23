#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "../common/utils.hpp"

#ifndef FLS_CUH
#define FLS_CUH

enum UnpackingType { LaneArray, VectorArray };

template <typename T, uint32_t OFFSET = 1, uint8_t BUFFER_SIZE = 4>
struct FlexibleBufferReader {
  T buffer[BUFFER_SIZE];
  const T *ptr;
  uint8_t index = BUFFER_SIZE;

  __device__ __forceinline__ T get() { return buffer[index]; }

  __device__ __forceinline__ void next() {
    if (index >= BUFFER_SIZE) {
      index = 0;

#pragma unroll
      for (unsigned i{0}; i < BUFFER_SIZE; ++i) {
        buffer[i] = *(ptr + i * OFFSET);
      }

      ptr += BUFFER_SIZE * OFFSET;
    } else {
      ++index;
    }
  }

  __device__ __forceinline__ FlexibleBufferReader(const T *ptr) : ptr(ptr) {
    next();
  };
};

template <typename T, uint32_t OFFSET = 1, uint8_t BUFFER_SIZE = 4>
struct FlexibleBufferReaderWithCurrent {
  T buffer[BUFFER_SIZE];
  T current;
  const T *ptr;
  uint8_t index = BUFFER_SIZE;

  __device__ __forceinline__ T get() { return current; }

  __device__ __forceinline__ void next() {
    if (index >= BUFFER_SIZE) {
      index = 0;

#pragma unroll
      for (unsigned i{0}; i < BUFFER_SIZE; ++i) {
        buffer[i] = *(ptr + i * OFFSET);
      }

      ptr += BUFFER_SIZE * OFFSET;
    }

    current = buffer[index];
    ++index;
  }

  __device__ __forceinline__ FlexibleBufferReaderWithCurrent(const T *ptr)
      : ptr(ptr) {
    next();
  };
};

template <typename T, uint32_t OFFSET = 1> struct FixedBufferReader {
  T buffer[4];
  const T *ptr;
  uint8_t index = 4;

  __device__ __forceinline__ T get() {
    switch (index) {
    case 0:
      return buffer[0];
    case 1:
      return buffer[1];
    case 2:
      return buffer[2];
    case 3:
      return buffer[3];
    }
  }

  __device__ __forceinline__ void next() {
    if (index >= 4) {
      index = 0;

#pragma unroll
      for (unsigned i{0}; i < 4; ++i) {
        buffer[i] = *(ptr + i * OFFSET);
      }

      ptr += 4 * OFFSET;
    } else {
      index++;
    }
  }

  __device__ __forceinline__ FixedBufferReader(const T *ptr) : ptr(ptr) {
    next();
  };
};

template <typename T, uint32_t OFFSET = 1, uint8_t BUFFER_SIZE = 4>
struct BufferReader {
  T buffer[BUFFER_SIZE];
  T current;
  const T *ptr;
  uint8_t index = BUFFER_SIZE;

  __device__ __forceinline__ T get() { return current; }

  __device__ __forceinline__ void next() {
    switch (index) {
    case 0:
      current = buffer[0];
      index = 1;
      break;
    case 1:
      current = buffer[1];
      index = 2;
      break;
    case 2:
      current = buffer[2];
      index = 3;
      break;
    case 3:
      current = buffer[3];
      index = 4;
      break;
    default:
#pragma unroll
      for (unsigned i{0}; i < BUFFER_SIZE; ++i) {
        buffer[i] = *(ptr + i * OFFSET);
        if (i == 0) {
          current = buffer[i];
        }
      }

      ptr += BUFFER_SIZE * OFFSET;
      index = 0;
      break;
    }
  }

  __device__ __forceinline__ BufferReader(const T *ptr) : ptr(ptr) {
    static_assert(BUFFER_SIZE == 4, "Buffer[4] reader has wrong size");
    next();
  };
};

template <typename T, uint32_t OFFSET = 1, uint8_t BUFFER_SIZE = 4>
struct NoRefreshBufferReader {
  T buffer[BUFFER_SIZE];
  T current;
  uint8_t index = 0;

  __device__ __forceinline__ T get() { return current; }

  __device__ __forceinline__ void next() {
    switch (index) {
    case 0:
      current = buffer[0];
      index = 1;
      break;
    case 1:
      current = buffer[1];
      index = 2;
      break;
    case 2:
      current = buffer[2];
      index = 3;
      break;
    case 3:
      current = buffer[3];
      index = 4;
      break;
    }
  }

  __device__ __forceinline__ NoRefreshBufferReader(const T *ptr) {
    static_assert(BUFFER_SIZE == 4, "Buffer[4] reader has wrong size");
#pragma unroll
      for (unsigned i{0}; i < BUFFER_SIZE; ++i) {
        buffer[i] = *(ptr + i * OFFSET);
        if (i == 0) {
          current = buffer[i];
        }
      }
  };
};

template <typename T_in, typename T_out, UnpackingType unpacking_type,
          unsigned UNPACK_N_VALUES>
struct BPUnpacker {
  const T_in *__restrict in;
  const uint16_t value_bit_width;
  BufferReader<T_in, utils::get_n_lanes<T_in>()> line_buffer;
  uint16_t buffer_offset = 0;

  __device__ __forceinline__ BPUnpacker(const T_in *__restrict in,
                                        const uint16_t lane,
                                        const uint16_t value_bit_width)
      : line_buffer(in + lane), value_bit_width(value_bit_width){};

  __device__ __forceinline__ void unpack_next_into(T_out *__restrict out) {
    // out += unpacking_type == UnpackingType::VectorArray ? lane : 0;
    static_assert(std::is_unsigned<T_in>::value,
                  "Packing function only supports unsigned types. Cast signed "
                  "arrays to unsigned equivalent.");
    constexpr uint8_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<T_in>();
    constexpr uint32_t N_LANES = utils::get_n_lanes<T_in>();
    const T_in value_mask = utils::set_first_n_bits<T_in>(value_bit_width);

    T_in buffer_offset_mask;

    T_in value;

#pragma unroll
    for (int i = 0; i < UNPACK_N_VALUES; ++i) {
      bool line_buffer_is_empty = buffer_offset == LANE_BIT_WIDTH;
      if (line_buffer_is_empty) {
        line_buffer.next();
        buffer_offset -= LANE_BIT_WIDTH;
      }

      value =
          (line_buffer.get() & (value_mask << buffer_offset)) >> buffer_offset;
      buffer_offset += value_bit_width;

      bool value_continues_on_next_line = buffer_offset > LANE_BIT_WIDTH;
      if (value_continues_on_next_line) {
        line_buffer.next();
        buffer_offset -= LANE_BIT_WIDTH;

        buffer_offset_mask =
            (T_in{1} << static_cast<T_in>(buffer_offset)) - T_in{1};
        value |= (line_buffer.get() & buffer_offset_mask)
                 << (value_bit_width - buffer_offset);
      }

      *(out) = value;
      out += unpacking_type == UnpackingType::VectorArray ? N_LANES : 1;
    }
  }
};

template <typename T_in, typename T_out, UnpackingType unpacking_type,
          unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename lambda_T>
__device__ void unpack_vector_new(const T_in *__restrict in,
                                  T_out *__restrict out, const uint16_t lane,
                                  const uint16_t value_bit_width,
                                  const uint16_t start_index, lambda_T lambda) {
  static_assert(std::is_unsigned<T_in>::value,
                "Packing function only supports unsigned types. Cast signed "
                "arrays to unsigned equivalent.");
  constexpr uint8_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<T_in>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T_in>();
  uint16_t preceding_bits = (start_index * value_bit_width);
  uint16_t buffer_offset = preceding_bits % LANE_BIT_WIDTH;
  uint16_t n_input_line = preceding_bits / LANE_BIT_WIDTH;
  T_in value_mask = utils::set_first_n_bits<T_in>(value_bit_width);

  T_in buffer_offset_mask;

  int32_t encoded_vector_offset =
      utils::get_compressed_vector_size<T_in>(value_bit_width);

  in += lane + n_input_line * N_LANES;

  // auto line_buffer = BufferReader<T_in, N_LANES, UNPACK_N_VALUES>(in);
  // auto line_buffer = BufferReaderWithCurrent<T_in, N_LANES,
  // UNPACK_N_VALUES>(in); auto line_buffer = FixedBufferReader<T_in,
  // N_LANES>(in);
  //auto line_buffer = BufferReader<T_in, N_LANES>(in);
  auto line_buffer = BufferReader<T_in, N_LANES>(in);

  out += unpacking_type == UnpackingType::VectorArray ? lane : 0;

  T_in value[UNPACK_N_VECTORS];

#pragma unroll
  for (int i = 0; i < UNPACK_N_VALUES; ++i) {
    bool line_buffer_is_empty = buffer_offset == LANE_BIT_WIDTH;
    if (line_buffer_is_empty) {
      line_buffer.next();
      buffer_offset -= LANE_BIT_WIDTH;
    }

#pragma unroll
    for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
      value[v] =
          (line_buffer.get() & (value_mask << buffer_offset)) >> buffer_offset;
    }
    buffer_offset += value_bit_width;

    bool value_continues_on_next_line = buffer_offset > LANE_BIT_WIDTH;
    if (value_continues_on_next_line) {
      line_buffer.next();
      buffer_offset -= LANE_BIT_WIDTH;

      buffer_offset_mask =
          (T_in{1} << static_cast<T_in>(buffer_offset)) - T_in{1};
#pragma unroll
      for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
        value[v] |= (line_buffer.get() & buffer_offset_mask)
                    << (value_bit_width - buffer_offset);
      }
    }

#pragma unroll
    for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
      *(out + v * UNPACK_N_VALUES) = lambda(value[v]);
    }
    out += unpacking_type == UnpackingType::VectorArray ? N_LANES : 1;
  }
}

template <typename T_in, typename T_out, UnpackingType unpacking_type,
          unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename lambda_T>
__device__ void unpack_vector(const T_in *__restrict in, T_out *__restrict out,
                              const uint16_t lane,
                              const uint16_t value_bit_width,
                              const uint16_t start_index, lambda_T lambda) {
  static_assert(std::is_unsigned<T_in>::value,
                "Packing function only supports unsigned types. Cast signed "
                "arrays to unsigned equivalent.");
  constexpr uint8_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<T_in>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T_in>();
  uint16_t preceding_bits = (start_index * value_bit_width);
  uint16_t buffer_offset = preceding_bits % LANE_BIT_WIDTH;
  uint16_t n_input_line = preceding_bits / LANE_BIT_WIDTH;
  T_in value_mask = utils::set_first_n_bits<T_in>(value_bit_width);

  T_in line_buffer[UNPACK_N_VECTORS];
  T_in buffer_offset_mask;

	// WARNING This causes quite some latency, test replacing it with a
	// constant memory table lookup
	// INFO Constant memory table lookup might be applicable in more places 
	// in this function.
  int32_t encoded_vector_offset = 
		utils::get_compressed_vector_size<T_in>(value_bit_width);

  in += lane;

#pragma unroll
  for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
    line_buffer[v] = *(in + n_input_line * N_LANES + v * encoded_vector_offset);
  }
  out += unpacking_type == UnpackingType::VectorArray ? lane : 0;
  n_input_line++;

  T_in value[UNPACK_N_VECTORS];

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
          (T_in{1} << static_cast<T_in>(buffer_offset)) - T_in{1};
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
    out += unpacking_type == UnpackingType::VectorArray ? N_LANES : 1;
  }
}

template <typename T_in, typename T_out, UnpackingType unpacking_type,
          unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename lambda_T>
__device__ void unpack_vector_alp(const T_in *__restrict in, T_out *__restrict out,
                              const uint16_t lane,
                              const uint16_t value_bit_width,
                              const uint16_t start_index, lambda_T lambda) {
  static_assert(std::is_unsigned<T_in>::value,
                "Packing function only supports unsigned types. Cast signed "
                "arrays to unsigned equivalent.");
  constexpr uint8_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<T_in>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T_in>();
  uint16_t preceding_bits = (start_index * value_bit_width);
  uint16_t buffer_offset = preceding_bits % LANE_BIT_WIDTH;
  uint16_t n_input_line = preceding_bits / LANE_BIT_WIDTH;
  T_in value_mask = utils::set_first_n_bits<T_in>(value_bit_width);

  T_in line_buffer[UNPACK_N_VECTORS];
  T_in buffer_offset_mask;

	// WARNING TODO Fix this with proper indexing into an offset array,
	// and then copying the approach from normal unpack with 
	// utils::get_compressed_vector_offset(vbw of UNPACK_N_VECTORS)
  int32_t encoded_vector_offset = consts::VALUES_PER_VECTOR;

  in += lane;

#pragma unroll
  for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
    line_buffer[v] = *(in + n_input_line * N_LANES + v * encoded_vector_offset);
  }
  out += unpacking_type == UnpackingType::VectorArray ? lane : 0;
  n_input_line++;

  T_in value[UNPACK_N_VECTORS];

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
          (T_in{1} << static_cast<T_in>(buffer_offset)) - T_in{1};
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
    out += unpacking_type == UnpackingType::VectorArray ? N_LANES : 1;
  }
}

template <typename T, UnpackingType unpacking_type, unsigned UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES>
__device__ void
bitunpack_vector(const T *__restrict in, T *__restrict out, const uint16_t lane,
                 const uint16_t value_bit_width, const uint16_t start_index) {
  auto lambda = [=](const T value) -> T { return value; };
  unpack_vector<T, T, unpacking_type, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
      in, out, lane, value_bit_width, start_index, lambda);
}

template <typename T, UnpackingType unpacking_type, unsigned UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES>
__device__ void bitunpack_vector_new(const T *__restrict in, T *__restrict out,
                                     const uint16_t lane,
                                     const uint16_t value_bit_width,
                                     const uint16_t start_index) {
  auto lambda = [=](const T value) -> T { return value; };
  unpack_vector_new<T, T, unpacking_type, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
      in, out, lane, value_bit_width, start_index, lambda);
}

template <typename T, UnpackingType unpacking_type, unsigned UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES>
__device__ void
unffor_vector(const T *__restrict in, T *__restrict out, const uint16_t lane,
              const uint16_t value_bit_width, const uint16_t start_index,
              const T *__restrict a_base_p) {
  T base = *a_base_p;
  auto lambda = [base](const T value) -> T { return value + base; };
  unpack_vector<T, T, unpacking_type, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
      in, out, lane, value_bit_width, start_index, lambda);
}

template <typename T, typename T_dict, UnpackingType unpacking_type,
          unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
__device__ void
undict_vector(const T *__restrict in, T *__restrict out, const uint16_t lane,
              const uint16_t value_bit_width, const uint16_t start_index,
              const T *__restrict a_base_p, const T_dict *__restrict dict) {
  T base = *a_base_p;
  auto lambda = [base, dict](const T value) -> T { return dict[value + base]; };
  unpack_vector<T, T_dict, unpacking_type, UNPACK_N_VECTORS, UNPACK_N_VALUES>(
      in, out, lane, value_bit_width, start_index, lambda);
}

template <typename T> struct SimpleLoader {
  T line_buffer;
  const T *in;

  __device__ __forceinline__ SimpleLoader(const T *in) : in(in){
		next_line();
	};

  __device__ __forceinline__ T get() { return line_buffer; }

  __device__ __forceinline__ void next_line() {
    line_buffer = *in;
    in += utils::get_n_lanes<T>();
  }
};

template <typename T> struct PrefetchLoader {
  T line_buffer;
  T prefetch_buffer;
  const T *in;

  __device__ __forceinline__ PrefetchLoader(const T *a_in) : in(a_in){
    line_buffer = *in;
    in += utils::get_n_lanes<T>();
    prefetch_buffer = *in;
    in += utils::get_n_lanes<T>();
	};

  __device__ __forceinline__ T get() { return line_buffer; }

  __device__ __forceinline__ void next_line() {
    line_buffer = prefetch_buffer;
    prefetch_buffer = *in;
    in += utils::get_n_lanes<T>();
  }
};


template <typename T> struct Masker {
  const uint16_t value_bit_width;
  const T value_mask;
  uint16_t buffer_offset = 0;

  __device__ __forceinline__ Masker(const uint16_t value_bit_width)
      : value_bit_width(value_bit_width),
        value_mask(utils::set_first_n_bits<T>(value_bit_width)){};

  __device__ __forceinline__ T mask_and_increment(T value) {
    value = (value & (value_mask << buffer_offset)) >> buffer_offset;
    buffer_offset += value_bit_width;
		return value;
  };

  __device__ __forceinline__ void next_line() {
    buffer_offset -= utils::get_lane_bitwidth<T>();
  }
  __device__ __forceinline__ bool is_buffer_empty() {
    return buffer_offset == utils::get_lane_bitwidth<T>();
  };

  __device__ __forceinline__ bool does_value_continue_on_next_line() {
    return buffer_offset > utils::get_lane_bitwidth<T>();
  }

  __device__ __forceinline__ T mask_remaining_value(T value) {
    T buffer_offset_mask = (T{1} << static_cast<T>(buffer_offset)) - T{1};
    return (value & buffer_offset_mask) << (value_bit_width - buffer_offset);
  }
};

template<typename T_in, typename T_out>
struct BPFunctor {
	__device__ __forceinline__ BPFunctor() {};
	__device__ __forceinline__ T_out operator()(const T_in value) const { return value; }
};

template <typename T_in, typename T_out, unsigned UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES, typename OutputProcessor>
struct BitUnpacker {
  SimpleLoader<T_in> loader;
  Masker<T_in> masker;
  const OutputProcessor processor;

  __device__ __forceinline__ BitUnpacker(const T_in *__restrict in,
                                         const uint16_t lane,
                                         const uint16_t value_bit_width,
                                         OutputProcessor processor)
      : loader(in + lane), masker(value_bit_width), processor(processor) { }

  __device__ __forceinline__ void unpack_into(T_out *__restrict out) {
    T_in value;

#pragma unroll
    for (int i = 0; i < UNPACK_N_VALUES; ++i) {
      if (masker.is_buffer_empty()) {
        loader.next_line();
        masker.next_line();
      }

      value = masker.mask_and_increment(loader.get());

      if (masker.does_value_continue_on_next_line()) {
        loader.next_line();
        masker.next_line();
        value |= masker.mask_remaining_value(loader.get());
      }

      *(out + i) = processor(value);
    }
  }
};

#endif // FLS_CUH
