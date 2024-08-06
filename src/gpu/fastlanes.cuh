#include <assert.h>
#include <cstdint>
#include <type_traits>

#include "../utils.h"

#ifndef FASTLANES_CUH
#define FASTLANES_CUH

enum UnpackingType { LaneArray, VectorArray };

template <typename T_in, typename T_out, UnpackingType unpacking_type,
          unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
__device__ void unpack_vector(const T_in *__restrict a_in,
                             T_out *__restrict a_out, const uint16_t lane,
                             const uint16_t value_bit_width,
                             const uint16_t start_index) {

  using unsigned_T_in = typename std::make_unsigned<T_in>::type;
  const unsigned_T_in *in = reinterpret_cast<const unsigned_T_in *>(a_in);
  using unsigned_T_out = typename std::make_unsigned<T_out>::type;
  unsigned_T_out *out = reinterpret_cast<unsigned_T_out *>(a_out);

  constexpr uint8_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<T_in>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T_in>();
  uint16_t preceding_bits = (start_index * value_bit_width);
  uint16_t buffer_offset = preceding_bits % LANE_BIT_WIDTH;
  uint16_t n_input_line = preceding_bits / LANE_BIT_WIDTH;
  unsigned_T_in value_mask = utils::set_first_n_bits<unsigned_T_in>(value_bit_width);

  unsigned_T_in line_buffer[UNPACK_N_VECTORS];
  unsigned_T_in buffer_offset_mask;

  int32_t encoded_vector_offset =
      utils::get_compressed_vector_size<T_in>(value_bit_width);

  in += lane;

#pragma unroll
  for (int i = 0; i < UNPACK_N_VECTORS; ++i) {
    line_buffer[i] = *(in + n_input_line * N_LANES + i * encoded_vector_offset);
  }
  out += unpacking_type == UnpackingType::VectorArray ? lane : 0;
  n_input_line++;

  T_out value[UNPACK_N_VECTORS];

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

      buffer_offset_mask = ((1 << buffer_offset) - 1);
#pragma unroll
      for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
        value[v] |= (line_buffer[v] & buffer_offset_mask)
                    << (value_bit_width - buffer_offset);
      }
    }

#pragma unroll
    for (int i = 0; i < UNPACK_N_VECTORS; ++i) {
      *(out + i * UNPACK_N_VALUES) = value[i];
    }
    out += unpacking_type == UnpackingType::VectorArray ? N_LANES : 1;
  }
}

template <typename T, UnpackingType unpacking_type, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct MultiVecScanner {
	const typename std::make_unsigned<T>::type* in;
	typename std::make_unsigned<T>::type        line_buffers[UNPACK_N_VECTORS];
	uint8_t                                     counter = 0;
	uint8_t                                     value_bit_width;

	__device__ MultiVecScanner(const T* a_in, const uint8_t a_value_bit_width, const int lane_index) {
		using unsigned_T = typename std::make_unsigned<T>::type;
		value_bit_width  = a_value_bit_width;
		in               = reinterpret_cast<const typename std::make_unsigned<T>::type*>(a_in) + lane_index;
	}

__device__ __forceinline__ void _unpack_value(
                             T *__restrict a_out, 
                             const uint16_t start_index) {

  using unsigned_T = typename std::make_unsigned<T>::type;
  unsigned_T *out = reinterpret_cast<unsigned_T *>(a_out);

  constexpr uint8_t LANE_BIT_WIDTH = utils::sizeof_in_bits<T>();
  constexpr uint32_t N_LANES = utils::get_n_lanes<T>();
  uint16_t preceding_bits = (start_index * value_bit_width);
  uint16_t buffer_offset = preceding_bits % LANE_BIT_WIDTH; // Make this a class var
  unsigned_T value_mask = utils::set_first_n_bits<T>(value_bit_width);

  unsigned_T buffer_offset_mask;

  int32_t encoded_vector_offset =
      utils::get_compressed_vector_size<T>(value_bit_width); // can be a class var

	bool line_buffer_is_empty = buffer_offset == 0;
	if (line_buffer_is_empty) {
#pragma unroll
		for (int i = 0; i < UNPACK_N_VECTORS; ++i) {
			line_buffers[i] =
					*(in + i * encoded_vector_offset);
		}
		in += N_LANES;
	}

  //out += unpacking_type == UnpackingType::VectorArray ? lane : 0;

  T value[UNPACK_N_VECTORS];

#pragma unroll
  for (int i = 0; i < UNPACK_N_VALUES; ++i) {
    bool line_buffer_is_empty = buffer_offset == LANE_BIT_WIDTH;
    if (line_buffer_is_empty) {
#pragma unroll
      for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
        line_buffers[v] =
            *(in + v * encoded_vector_offset);
      }
      in += N_LANES;
      buffer_offset -= LANE_BIT_WIDTH;
    }

#pragma unroll
    for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
      value[v] =
          (line_buffers[v] & (value_mask << buffer_offset)) >> buffer_offset;
    }
    buffer_offset += value_bit_width;

    bool value_continues_on_next_line = buffer_offset > LANE_BIT_WIDTH;
    if (value_continues_on_next_line) {
#pragma unroll
      for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
        line_buffers[v] =
            *(in + v * encoded_vector_offset);
      }
      in += N_LANES;
      buffer_offset -= LANE_BIT_WIDTH;

      buffer_offset_mask = ((1 << buffer_offset) - 1);
#pragma unroll
      for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
        value[v] |= (line_buffers[v] & buffer_offset_mask)
                    << (value_bit_width - buffer_offset);
      }
    }

#pragma unroll // TODO: remove value buffer, write directly to out, assume lane buffer
    for (int i = 0; i < UNPACK_N_VECTORS; ++i) {
      *(out + i * UNPACK_N_VALUES) = value[i];
    }
    out += unpacking_type == UnpackingType::VectorArray ? N_LANES : 1;
  }
}

	__device__ void unpack_next(T *__restrict a_out) {
		_unpack_value(a_out, counter);
		counter += UNPACK_N_VALUES;
	}
};



#endif // FASTLANES_CUH
