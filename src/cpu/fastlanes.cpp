#include <cstdint>
#include <cstdio>
#include <functional>
#include <type_traits>

#include "../utils.h"
#include "fastlanes.h"

namespace compression {

template <typename T, int32_t VALUE_BIT_WIDTH, typename lambda_T>
void pack(const T *__restrict in, T *__restrict out, lambda_T lambda) {
  static_assert(std::is_unsigned<T>::value,
                "Packing function only supports unsigned types. Cast signed "
                "arrays to unsigned equivalent.");
  constexpr int32_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<T>();
  constexpr int32_t N_LANES = utils::get_n_lanes<T>();
  constexpr int32_t VALUES_PER_LANE = utils::get_values_per_lane<T>();
  constexpr T VALUE_MASK = utils::set_first_n_bits<T>(VALUE_BIT_WIDTH);

  T buffer = 0;
  T value;

  uint32_t buffer_offset;
  for (int32_t lane{0}; lane < N_LANES; lane++) {
    buffer_offset = 0;

#pragma clang loop unroll(full)
    for (int32_t n_value{0}; n_value < VALUES_PER_LANE; n_value++) {
      value =
          static_cast<T>(lambda(*(in + N_LANES * n_value + lane))) & VALUE_MASK;

      buffer |= value << buffer_offset;
      buffer_offset += VALUE_BIT_WIDTH;
      bool next_value_should_be_on_next_line = buffer_offset >= LANE_BIT_WIDTH;
      if (next_value_should_be_on_next_line) {
        *(out + lane) = static_cast<T>(buffer);
        out += N_LANES;
        buffer = 0;

        buffer_offset %= LANE_BIT_WIDTH;
        bool value_did_not_fit_in_buffer = buffer_offset > 0;
        if (value_did_not_fit_in_buffer) {
          buffer |= value >> (VALUE_BIT_WIDTH - buffer_offset);
        }
      }
    }
    out -= N_LANES * VALUE_BIT_WIDTH;
  }
}

template <typename T, int32_t VALUE_BIT_WIDTH, int32_t UNPACK_N_VALUES,
          int32_t START_INDEX, typename lambda_T>
void unpack(const T *__restrict in, T *__restrict out, lambda_T lambda) {
  static_assert(std::is_unsigned<T>::value,
                "Packing function only supports unsigned types. Cast signed "
                "arrays to unsigned equivalent.");
  constexpr int32_t LANE_BIT_WIDTH = utils::get_lane_bitwidth<T>();
  constexpr int32_t N_LANES = utils::get_n_lanes<T>();
  constexpr T VALUE_MASK = utils::set_first_n_bits<T>(VALUE_BIT_WIDTH);

  constexpr int32_t PRECEDING_BITS =
      ((START_INDEX / N_LANES) * VALUE_BIT_WIDTH);
  constexpr int32_t INITIAL_BUFFER_OFFSET = PRECEDING_BITS % LANE_BIT_WIDTH;
  constexpr int32_t INITIAL_N_INPUT_LINE = PRECEDING_BITS / LANE_BIT_WIDTH;

  for (int32_t lane{0}; lane < N_LANES; lane++) {
    T line_buffer = 0U;
    uint32_t buffer_offset = INITIAL_BUFFER_OFFSET;
    uint32_t n_input_line = INITIAL_N_INPUT_LINE;
    T buffer_offset_mask;

    line_buffer = static_cast<T>(*(in + n_input_line * N_LANES + lane));
    n_input_line++;

#pragma clang loop unroll(full)
    for (int32_t i{0}; i < UNPACK_N_VALUES; ++i) {
      T value;

      bool line_buffer_is_empty = buffer_offset == LANE_BIT_WIDTH;
      if (line_buffer_is_empty) {
        line_buffer = static_cast<T>(*(in + n_input_line * N_LANES + lane));
        ++n_input_line;
        buffer_offset -= LANE_BIT_WIDTH;
      }

      value = (line_buffer & (VALUE_MASK << buffer_offset)) >> buffer_offset;
      buffer_offset += VALUE_BIT_WIDTH;

      bool value_continues_on_next_line = buffer_offset > LANE_BIT_WIDTH;
      if (value_continues_on_next_line) {
        line_buffer = static_cast<T>(*(in + n_input_line * N_LANES + lane));

        ++n_input_line;
        buffer_offset -= LANE_BIT_WIDTH;

        buffer_offset_mask = static_cast<T>((1 << buffer_offset) - 1);
        value |= (line_buffer & buffer_offset_mask)
                 << (VALUE_BIT_WIDTH - buffer_offset);
      }

      *(out + lane) = lambda(static_cast<T>(value));

      out += N_LANES;
    }
    out -= N_LANES * UNPACK_N_VALUES;
  }
}

template <typename T, unsigned VALUE_BIT_WIDTH>
void bitpack(const T *__restrict in, T *__restrict out) {
  auto lambda = [](const T value) -> T { return value; };
  compression::pack<T, VALUE_BIT_WIDTH>(in, out, lambda);
}

template <typename T, unsigned VALUE_BIT_WIDTH>
void bitunpack(const T *__restrict in, T *__restrict out) {
  auto lambda = [](const T value) -> T { return value; };
  compression::unpack<T, VALUE_BIT_WIDTH, utils::get_values_per_lane<T>(), 0>(
      in, out, lambda);
}

template <typename T, unsigned VALUE_BIT_WIDTH>
void ffor(const T *__restrict in, T *__restrict out,
          const T *__restrict base_p) {
  auto lambda = [base_p](const T value) -> T { return value - *(base_p); };
  compression::pack<T, VALUE_BIT_WIDTH>(in, out, lambda);
}

template <typename T, unsigned VALUE_BIT_WIDTH>
void unffor(const T *__restrict in, T *__restrict out,
            const T *__restrict base_p) {
  auto lambda = [base_p](const T value) -> T { return value + *(base_p); };
  compression::unpack<T, VALUE_BIT_WIDTH, utils::get_values_per_lane<T>(), 0>(
      in, out, lambda);
}

} // namespace compression

namespace cpu {

template <typename T>
void bitpack(const T *__restrict in, T *__restrict out,
             [[maybe_unused]] const int32_t value_bit_width) {
#ifdef VBW
  compression::bitpack<T, VBW>(in, out);
#else
  switch (value_bit_width) {
  case 1:
    compression::bitpack<T, 1>(in, out);
    break;
  case 2:
    compression::bitpack<T, 2>(in, out);
    break;
  case 3:
    compression::bitpack<T, 3>(in, out);
    break;
  case 4:
    compression::bitpack<T, 4>(in, out);
    break;
  case 5:
    compression::bitpack<T, 5>(in, out);
    break;
  case 6:
    compression::bitpack<T, 6>(in, out);
    break;
  case 7:
    compression::bitpack<T, 7>(in, out);
    break;
  case 8:
    compression::bitpack<T, 8>(in, out);
    break;
  case 9:
    compression::bitpack<T, 9>(in, out);
    break;
  case 10:
    compression::bitpack<T, 10>(in, out);
    break;
  case 11:
    compression::bitpack<T, 11>(in, out);
    break;
  case 12:
    compression::bitpack<T, 12>(in, out);
    break;
  case 13:
    compression::bitpack<T, 13>(in, out);
    break;
  case 14:
    compression::bitpack<T, 14>(in, out);
    break;
  case 15:
    compression::bitpack<T, 15>(in, out);
    break;
  case 16:
    compression::bitpack<T, 16>(in, out);
    break;
  case 17:
    compression::bitpack<T, 17>(in, out);
    break;
  case 18:
    compression::bitpack<T, 18>(in, out);
    break;
  case 19:
    compression::bitpack<T, 19>(in, out);
    break;
  case 20:
    compression::bitpack<T, 20>(in, out);
    break;
  case 21:
    compression::bitpack<T, 21>(in, out);
    break;
  case 22:
    compression::bitpack<T, 22>(in, out);
    break;
  case 23:
    compression::bitpack<T, 23>(in, out);
    break;
  case 24:
    compression::bitpack<T, 24>(in, out);
    break;
  case 25:
    compression::bitpack<T, 25>(in, out);
    break;
  case 26:
    compression::bitpack<T, 26>(in, out);
    break;
  case 27:
    compression::bitpack<T, 27>(in, out);
    break;
  case 28:
    compression::bitpack<T, 28>(in, out);
    break;
  case 29:
    compression::bitpack<T, 29>(in, out);
    break;
  case 30:
    compression::bitpack<T, 30>(in, out);
    break;
  case 31:
    compression::bitpack<T, 31>(in, out);
    break;
  case 32:
    compression::bitpack<T, 32>(in, out);
    break;
  case 33:
    compression::bitpack<T, 33>(in, out);
    break;
  case 34:
    compression::bitpack<T, 34>(in, out);
    break;
  case 35:
    compression::bitpack<T, 35>(in, out);
    break;
  case 36:
    compression::bitpack<T, 36>(in, out);
    break;
  case 37:
    compression::bitpack<T, 37>(in, out);
    break;
  case 38:
    compression::bitpack<T, 38>(in, out);
    break;
  case 39:
    compression::bitpack<T, 39>(in, out);
    break;
  case 40:
    compression::bitpack<T, 40>(in, out);
    break;
  case 41:
    compression::bitpack<T, 41>(in, out);
    break;
  case 42:
    compression::bitpack<T, 42>(in, out);
    break;
  case 43:
    compression::bitpack<T, 43>(in, out);
    break;
  case 44:
    compression::bitpack<T, 44>(in, out);
    break;
  case 45:
    compression::bitpack<T, 45>(in, out);
    break;
  case 46:
    compression::bitpack<T, 46>(in, out);
    break;
  case 47:
    compression::bitpack<T, 47>(in, out);
    break;
  case 48:
    compression::bitpack<T, 48>(in, out);
    break;
  case 49:
    compression::bitpack<T, 49>(in, out);
    break;
  case 50:
    compression::bitpack<T, 50>(in, out);
    break;
  case 51:
    compression::bitpack<T, 51>(in, out);
    break;
  case 52:
    compression::bitpack<T, 52>(in, out);
    break;
  case 53:
    compression::bitpack<T, 53>(in, out);
    break;
  case 54:
    compression::bitpack<T, 54>(in, out);
    break;
  case 55:
    compression::bitpack<T, 55>(in, out);
    break;
  case 56:
    compression::bitpack<T, 56>(in, out);
    break;
  case 57:
    compression::bitpack<T, 57>(in, out);
    break;
  case 58:
    compression::bitpack<T, 58>(in, out);
    break;
  case 59:
    compression::bitpack<T, 59>(in, out);
    break;
  case 60:
    compression::bitpack<T, 60>(in, out);
    break;
  case 61:
    compression::bitpack<T, 61>(in, out);
    break;
  case 62:
    compression::bitpack<T, 62>(in, out);
    break;
  case 63:
    compression::bitpack<T, 63>(in, out);
    break;
  case 64:
    compression::bitpack<T, 64>(in, out);
    break;
  }
#endif
}
template <typename T>
void bitunpack(const T *__restrict in, T *__restrict out,
               [[maybe_unused]] const int32_t value_bit_width) {
#ifdef VBW
  compression::bitunpack<T, VBW>(in, out);
#else
  switch (value_bit_width) {
  case 1:
    compression::bitunpack<T, 1>(in, out);
    break;
  case 2:
    compression::bitunpack<T, 2>(in, out);
    break;
  case 3:
    compression::bitunpack<T, 3>(in, out);
    break;
  case 4:
    compression::bitunpack<T, 4>(in, out);
    break;
  case 5:
    compression::bitunpack<T, 5>(in, out);
    break;
  case 6:
    compression::bitunpack<T, 6>(in, out);
    break;
  case 7:
    compression::bitunpack<T, 7>(in, out);
    break;
  case 8:
    compression::bitunpack<T, 8>(in, out);
    break;
  case 9:
    compression::bitunpack<T, 9>(in, out);
    break;
  case 10:
    compression::bitunpack<T, 10>(in, out);
    break;
  case 11:
    compression::bitunpack<T, 11>(in, out);
    break;
  case 12:
    compression::bitunpack<T, 12>(in, out);
    break;
  case 13:
    compression::bitunpack<T, 13>(in, out);
    break;
  case 14:
    compression::bitunpack<T, 14>(in, out);
    break;
  case 15:
    compression::bitunpack<T, 15>(in, out);
    break;
  case 16:
    compression::bitunpack<T, 16>(in, out);
    break;
  case 17:
    compression::bitunpack<T, 17>(in, out);
    break;
  case 18:
    compression::bitunpack<T, 18>(in, out);
    break;
  case 19:
    compression::bitunpack<T, 19>(in, out);
    break;
  case 20:
    compression::bitunpack<T, 20>(in, out);
    break;
  case 21:
    compression::bitunpack<T, 21>(in, out);
    break;
  case 22:
    compression::bitunpack<T, 22>(in, out);
    break;
  case 23:
    compression::bitunpack<T, 23>(in, out);
    break;
  case 24:
    compression::bitunpack<T, 24>(in, out);
    break;
  case 25:
    compression::bitunpack<T, 25>(in, out);
    break;
  case 26:
    compression::bitunpack<T, 26>(in, out);
    break;
  case 27:
    compression::bitunpack<T, 27>(in, out);
    break;
  case 28:
    compression::bitunpack<T, 28>(in, out);
    break;
  case 29:
    compression::bitunpack<T, 29>(in, out);
    break;
  case 30:
    compression::bitunpack<T, 30>(in, out);
    break;
  case 31:
    compression::bitunpack<T, 31>(in, out);
    break;
  case 32:
    compression::bitunpack<T, 32>(in, out);
    break;
  case 33:
    compression::bitunpack<T, 33>(in, out);
    break;
  case 34:
    compression::bitunpack<T, 34>(in, out);
    break;
  case 35:
    compression::bitunpack<T, 35>(in, out);
    break;
  case 36:
    compression::bitunpack<T, 36>(in, out);
    break;
  case 37:
    compression::bitunpack<T, 37>(in, out);
    break;
  case 38:
    compression::bitunpack<T, 38>(in, out);
    break;
  case 39:
    compression::bitunpack<T, 39>(in, out);
    break;
  case 40:
    compression::bitunpack<T, 40>(in, out);
    break;
  case 41:
    compression::bitunpack<T, 41>(in, out);
    break;
  case 42:
    compression::bitunpack<T, 42>(in, out);
    break;
  case 43:
    compression::bitunpack<T, 43>(in, out);
    break;
  case 44:
    compression::bitunpack<T, 44>(in, out);
    break;
  case 45:
    compression::bitunpack<T, 45>(in, out);
    break;
  case 46:
    compression::bitunpack<T, 46>(in, out);
    break;
  case 47:
    compression::bitunpack<T, 47>(in, out);
    break;
  case 48:
    compression::bitunpack<T, 48>(in, out);
    break;
  case 49:
    compression::bitunpack<T, 49>(in, out);
    break;
  case 50:
    compression::bitunpack<T, 50>(in, out);
    break;
  case 51:
    compression::bitunpack<T, 51>(in, out);
    break;
  case 52:
    compression::bitunpack<T, 52>(in, out);
    break;
  case 53:
    compression::bitunpack<T, 53>(in, out);
    break;
  case 54:
    compression::bitunpack<T, 54>(in, out);
    break;
  case 55:
    compression::bitunpack<T, 55>(in, out);
    break;
  case 56:
    compression::bitunpack<T, 56>(in, out);
    break;
  case 57:
    compression::bitunpack<T, 57>(in, out);
    break;
  case 58:
    compression::bitunpack<T, 58>(in, out);
    break;
  case 59:
    compression::bitunpack<T, 59>(in, out);
    break;
  case 60:
    compression::bitunpack<T, 60>(in, out);
    break;
  case 61:
    compression::bitunpack<T, 61>(in, out);
    break;
  case 62:
    compression::bitunpack<T, 62>(in, out);
    break;
  case 63:
    compression::bitunpack<T, 63>(in, out);
    break;
  case 64:
    compression::bitunpack<T, 64>(in, out);
    break;
  }
#endif
}

template <typename T>
void ffor(const T *__restrict in, T *__restrict out, const T *__restrict base_p,
          [[maybe_unused]] const int32_t value_bit_width) {
#ifdef VBW
  compression::ffor<T, VBW>(in, out, base_p);
#else
  switch (value_bit_width) {
  case 1:
    compression::ffor<T, 1>(in, out, base_p);
    break;
  case 2:
    compression::ffor<T, 2>(in, out, base_p);
    break;
  case 3:
    compression::ffor<T, 3>(in, out, base_p);
    break;
  case 4:
    compression::ffor<T, 4>(in, out, base_p);
    break;
  case 5:
    compression::ffor<T, 5>(in, out, base_p);
    break;
  case 6:
    compression::ffor<T, 6>(in, out, base_p);
    break;
  case 7:
    compression::ffor<T, 7>(in, out, base_p);
    break;
  case 8:
    compression::ffor<T, 8>(in, out, base_p);
    break;
  case 9:
    compression::ffor<T, 9>(in, out, base_p);
    break;
  case 10:
    compression::ffor<T, 10>(in, out, base_p);
    break;
  case 11:
    compression::ffor<T, 11>(in, out, base_p);
    break;
  case 12:
    compression::ffor<T, 12>(in, out, base_p);
    break;
  case 13:
    compression::ffor<T, 13>(in, out, base_p);
    break;
  case 14:
    compression::ffor<T, 14>(in, out, base_p);
    break;
  case 15:
    compression::ffor<T, 15>(in, out, base_p);
    break;
  case 16:
    compression::ffor<T, 16>(in, out, base_p);
    break;
  case 17:
    compression::ffor<T, 17>(in, out, base_p);
    break;
  case 18:
    compression::ffor<T, 18>(in, out, base_p);
    break;
  case 19:
    compression::ffor<T, 19>(in, out, base_p);
    break;
  case 20:
    compression::ffor<T, 20>(in, out, base_p);
    break;
  case 21:
    compression::ffor<T, 21>(in, out, base_p);
    break;
  case 22:
    compression::ffor<T, 22>(in, out, base_p);
    break;
  case 23:
    compression::ffor<T, 23>(in, out, base_p);
    break;
  case 24:
    compression::ffor<T, 24>(in, out, base_p);
    break;
  case 25:
    compression::ffor<T, 25>(in, out, base_p);
    break;
  case 26:
    compression::ffor<T, 26>(in, out, base_p);
    break;
  case 27:
    compression::ffor<T, 27>(in, out, base_p);
    break;
  case 28:
    compression::ffor<T, 28>(in, out, base_p);
    break;
  case 29:
    compression::ffor<T, 29>(in, out, base_p);
    break;
  case 30:
    compression::ffor<T, 30>(in, out, base_p);
    break;
  case 31:
    compression::ffor<T, 31>(in, out, base_p);
    break;
  case 32:
    compression::ffor<T, 32>(in, out, base_p);
    break;
  case 33:
    compression::ffor<T, 33>(in, out, base_p);
    break;
  case 34:
    compression::ffor<T, 34>(in, out, base_p);
    break;
  case 35:
    compression::ffor<T, 35>(in, out, base_p);
    break;
  case 36:
    compression::ffor<T, 36>(in, out, base_p);
    break;
  case 37:
    compression::ffor<T, 37>(in, out, base_p);
    break;
  case 38:
    compression::ffor<T, 38>(in, out, base_p);
    break;
  case 39:
    compression::ffor<T, 39>(in, out, base_p);
    break;
  case 40:
    compression::ffor<T, 40>(in, out, base_p);
    break;
  case 41:
    compression::ffor<T, 41>(in, out, base_p);
    break;
  case 42:
    compression::ffor<T, 42>(in, out, base_p);
    break;
  case 43:
    compression::ffor<T, 43>(in, out, base_p);
    break;
  case 44:
    compression::ffor<T, 44>(in, out, base_p);
    break;
  case 45:
    compression::ffor<T, 45>(in, out, base_p);
    break;
  case 46:
    compression::ffor<T, 46>(in, out, base_p);
    break;
  case 47:
    compression::ffor<T, 47>(in, out, base_p);
    break;
  case 48:
    compression::ffor<T, 48>(in, out, base_p);
    break;
  case 49:
    compression::ffor<T, 49>(in, out, base_p);
    break;
  case 50:
    compression::ffor<T, 50>(in, out, base_p);
    break;
  case 51:
    compression::ffor<T, 51>(in, out, base_p);
    break;
  case 52:
    compression::ffor<T, 52>(in, out, base_p);
    break;
  case 53:
    compression::ffor<T, 53>(in, out, base_p);
    break;
  case 54:
    compression::ffor<T, 54>(in, out, base_p);
    break;
  case 55:
    compression::ffor<T, 55>(in, out, base_p);
    break;
  case 56:
    compression::ffor<T, 56>(in, out, base_p);
    break;
  case 57:
    compression::ffor<T, 57>(in, out, base_p);
    break;
  case 58:
    compression::ffor<T, 58>(in, out, base_p);
    break;
  case 59:
    compression::ffor<T, 59>(in, out, base_p);
    break;
  case 60:
    compression::ffor<T, 60>(in, out, base_p);
    break;
  case 61:
    compression::ffor<T, 61>(in, out, base_p);
    break;
  case 62:
    compression::ffor<T, 62>(in, out, base_p);
    break;
  case 63:
    compression::ffor<T, 63>(in, out, base_p);
    break;
  case 64:
    compression::ffor<T, 64>(in, out, base_p);
    break;
  }
#endif
}
template <typename T>
void unffor(const T *__restrict in, T *__restrict out,
            const T *__restrict base_p,
            [[maybe_unused]] const int32_t value_bit_width) {
#ifdef VBW
  compression::unffor<T, VBW>(in, out, base_p);
#else
  switch (value_bit_width) {
  case 1:
    compression::unffor<T, 1>(in, out, base_p);
    break;
  case 2:
    compression::unffor<T, 2>(in, out, base_p);
    break;
  case 3:
    compression::unffor<T, 3>(in, out, base_p);
    break;
  case 4:
    compression::unffor<T, 4>(in, out, base_p);
    break;
  case 5:
    compression::unffor<T, 5>(in, out, base_p);
    break;
  case 6:
    compression::unffor<T, 6>(in, out, base_p);
    break;
  case 7:
    compression::unffor<T, 7>(in, out, base_p);
    break;
  case 8:
    compression::unffor<T, 8>(in, out, base_p);
    break;
  case 9:
    compression::unffor<T, 9>(in, out, base_p);
    break;
  case 10:
    compression::unffor<T, 10>(in, out, base_p);
    break;
  case 11:
    compression::unffor<T, 11>(in, out, base_p);
    break;
  case 12:
    compression::unffor<T, 12>(in, out, base_p);
    break;
  case 13:
    compression::unffor<T, 13>(in, out, base_p);
    break;
  case 14:
    compression::unffor<T, 14>(in, out, base_p);
    break;
  case 15:
    compression::unffor<T, 15>(in, out, base_p);
    break;
  case 16:
    compression::unffor<T, 16>(in, out, base_p);
    break;
  case 17:
    compression::unffor<T, 17>(in, out, base_p);
    break;
  case 18:
    compression::unffor<T, 18>(in, out, base_p);
    break;
  case 19:
    compression::unffor<T, 19>(in, out, base_p);
    break;
  case 20:
    compression::unffor<T, 20>(in, out, base_p);
    break;
  case 21:
    compression::unffor<T, 21>(in, out, base_p);
    break;
  case 22:
    compression::unffor<T, 22>(in, out, base_p);
    break;
  case 23:
    compression::unffor<T, 23>(in, out, base_p);
    break;
  case 24:
    compression::unffor<T, 24>(in, out, base_p);
    break;
  case 25:
    compression::unffor<T, 25>(in, out, base_p);
    break;
  case 26:
    compression::unffor<T, 26>(in, out, base_p);
    break;
  case 27:
    compression::unffor<T, 27>(in, out, base_p);
    break;
  case 28:
    compression::unffor<T, 28>(in, out, base_p);
    break;
  case 29:
    compression::unffor<T, 29>(in, out, base_p);
    break;
  case 30:
    compression::unffor<T, 30>(in, out, base_p);
    break;
  case 31:
    compression::unffor<T, 31>(in, out, base_p);
    break;
  case 32:
    compression::unffor<T, 32>(in, out, base_p);
    break;
  case 33:
    compression::unffor<T, 33>(in, out, base_p);
    break;
  case 34:
    compression::unffor<T, 34>(in, out, base_p);
    break;
  case 35:
    compression::unffor<T, 35>(in, out, base_p);
    break;
  case 36:
    compression::unffor<T, 36>(in, out, base_p);
    break;
  case 37:
    compression::unffor<T, 37>(in, out, base_p);
    break;
  case 38:
    compression::unffor<T, 38>(in, out, base_p);
    break;
  case 39:
    compression::unffor<T, 39>(in, out, base_p);
    break;
  case 40:
    compression::unffor<T, 40>(in, out, base_p);
    break;
  case 41:
    compression::unffor<T, 41>(in, out, base_p);
    break;
  case 42:
    compression::unffor<T, 42>(in, out, base_p);
    break;
  case 43:
    compression::unffor<T, 43>(in, out, base_p);
    break;
  case 44:
    compression::unffor<T, 44>(in, out, base_p);
    break;
  case 45:
    compression::unffor<T, 45>(in, out, base_p);
    break;
  case 46:
    compression::unffor<T, 46>(in, out, base_p);
    break;
  case 47:
    compression::unffor<T, 47>(in, out, base_p);
    break;
  case 48:
    compression::unffor<T, 48>(in, out, base_p);
    break;
  case 49:
    compression::unffor<T, 49>(in, out, base_p);
    break;
  case 50:
    compression::unffor<T, 50>(in, out, base_p);
    break;
  case 51:
    compression::unffor<T, 51>(in, out, base_p);
    break;
  case 52:
    compression::unffor<T, 52>(in, out, base_p);
    break;
  case 53:
    compression::unffor<T, 53>(in, out, base_p);
    break;
  case 54:
    compression::unffor<T, 54>(in, out, base_p);
    break;
  case 55:
    compression::unffor<T, 55>(in, out, base_p);
    break;
  case 56:
    compression::unffor<T, 56>(in, out, base_p);
    break;
  case 57:
    compression::unffor<T, 57>(in, out, base_p);
    break;
  case 58:
    compression::unffor<T, 58>(in, out, base_p);
    break;
  case 59:
    compression::unffor<T, 59>(in, out, base_p);
    break;
  case 60:
    compression::unffor<T, 60>(in, out, base_p);
    break;
  case 61:
    compression::unffor<T, 61>(in, out, base_p);
    break;
  case 62:
    compression::unffor<T, 62>(in, out, base_p);
    break;
  case 63:
    compression::unffor<T, 63>(in, out, base_p);
    break;
  case 64:
    compression::unffor<T, 64>(in, out, base_p);
    break;
  }
#endif
}


} // namespace cpu

template 
void cpu::bitpack(const uint8_t *__restrict in, uint8_t *__restrict out,
             [[maybe_unused]] const int32_t value_bit_width);
template
void cpu::bitunpack(const uint8_t *__restrict in, uint8_t *__restrict out,
               [[maybe_unused]] const int32_t value_bit_width);
template 
void cpu::ffor(const uint8_t *__restrict in, uint8_t *__restrict out, const uint8_t *__restrict base_p,
          [[maybe_unused]] const int32_t value_bit_width);
template
void cpu::unffor(const uint8_t *__restrict in, uint8_t *__restrict out,
            const uint8_t *__restrict base_p,
            [[maybe_unused]] const int32_t value_bit_width);

template 
void cpu::bitpack(const uint16_t *__restrict in, uint16_t *__restrict out,
             [[maybe_unused]] const int32_t value_bit_width);
template
void cpu::bitunpack(const uint16_t *__restrict in, uint16_t *__restrict out,
               [[maybe_unused]] const int32_t value_bit_width);
template 
void cpu::ffor(const uint16_t *__restrict in, uint16_t *__restrict out, const uint16_t *__restrict base_p,
          [[maybe_unused]] const int32_t value_bit_width);
template
void cpu::unffor(const uint16_t *__restrict in, uint16_t *__restrict out,
            const uint16_t *__restrict base_p,
            [[maybe_unused]] const int32_t value_bit_width);

template 
void cpu::bitpack(const uint32_t *__restrict in, uint32_t *__restrict out,
             [[maybe_unused]] const int32_t value_bit_width);
template
void cpu::bitunpack(const uint32_t *__restrict in, uint32_t *__restrict out,
               [[maybe_unused]] const int32_t value_bit_width);
template 
void cpu::ffor(const uint32_t *__restrict in, uint32_t *__restrict out, const uint32_t *__restrict base_p,
          [[maybe_unused]] const int32_t value_bit_width);
template
void cpu::unffor(const uint32_t *__restrict in, uint32_t *__restrict out,
            const uint32_t *__restrict base_p,
            [[maybe_unused]] const int32_t value_bit_width);

template 
void cpu::bitpack(const uint64_t *__restrict in, uint64_t *__restrict out,
             [[maybe_unused]] const int32_t value_bit_width);
template
void cpu::bitunpack(const uint64_t *__restrict in, uint64_t *__restrict out,
               [[maybe_unused]] const int32_t value_bit_width);
template 
void cpu::ffor(const uint64_t *__restrict in, uint64_t *__restrict out, const uint64_t *__restrict base_p,
          [[maybe_unused]] const int32_t value_bit_width);
template
void cpu::unffor(const uint64_t *__restrict in, uint64_t *__restrict out,
            const uint64_t *__restrict base_p,
            [[maybe_unused]] const int32_t value_bit_width);
