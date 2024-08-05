#include <cstdint>
#include <functional>

#ifndef FASTLANES_H
#define FASTLANES_H

namespace consts {

constexpr uint32_t REGISTER_WIDTH = 1024;
constexpr uint32_t VALUES_PER_VECTOR = 1024;

} // namespace consts

namespace utils { // internal functions

template <typename T> constexpr T set_first_n_bits(const int32_t count) {
  return count < 64 ? ((1ULL << count) - 1) : ~0;
}

template <typename T, unsigned LANE_BIT_WIDTH, unsigned VALUE_BIT_WIDTH,
          typename lambda_T>
void pack(const T *__restrict in, T *__restrict out, lambda_T lambda) {
  constexpr uint64_t N_LANES = consts::REGISTER_WIDTH / LANE_BIT_WIDTH;
  constexpr uint64_t VALUES_PER_LANE = (consts::VALUES_PER_VECTOR / N_LANES);
  constexpr uint64_t LINES_PER_ENCODED_VECTOR =
      ((consts::VALUES_PER_VECTOR * VALUE_BIT_WIDTH) / consts::REGISTER_WIDTH);
  constexpr T VALUE_MASK = utils::set_first_n_bits<T>(VALUE_BIT_WIDTH);

  T buffer = 0;
  T value;
  int buffer_offset;
  for (int lane = 0; lane < N_LANES; lane++) {
    buffer_offset = 0;

#pragma clang loop unroll(full)
    for (int n_value = 0; n_value < VALUES_PER_LANE; n_value++) {
      value = lambda(*(in + N_LANES * n_value + lane)) & VALUE_MASK;

      buffer |= value << buffer_offset;
      buffer_offset += VALUE_BIT_WIDTH;
      bool next_value_should_be_on_next_line = buffer_offset >= LANE_BIT_WIDTH;
      if (next_value_should_be_on_next_line) {
        *(out + lane) = buffer;
        out += N_LANES;
        buffer = 0;

        buffer_offset %= LANE_BIT_WIDTH;
        bool value_did_not_fit_in_buffer = buffer_offset > 0;
        if (value_did_not_fit_in_buffer) {
          buffer |= value >> (VALUE_BIT_WIDTH - buffer_offset);
        }
      }
    }
    out -= N_LANES * LINES_PER_ENCODED_VECTOR;
  }
}

template <typename T, unsigned LANE_BIT_WIDTH, unsigned VALUE_BIT_WIDTH,
          unsigned UNPACK_N_VALUES, unsigned START_INDEX, typename lambda_T>
void unpack(const T *__restrict in, T *__restrict out, lambda_T lambda) {
  constexpr uint64_t N_LANES = consts::REGISTER_WIDTH / LANE_BIT_WIDTH;

  constexpr uint64_t VALUES_PER_LANE = (consts::VALUES_PER_VECTOR / N_LANES);
  constexpr uint64_t LINES_PER_LANE =
      (VALUES_PER_LANE * VALUE_BIT_WIDTH) / LANE_BIT_WIDTH;
  constexpr T VALUE_MASK = utils::set_first_n_bits<T>(VALUE_BIT_WIDTH);

  constexpr uint64_t PRECEDING_BITS =
      ((START_INDEX / N_LANES) * VALUE_BIT_WIDTH);
  constexpr uint64_t INITIAL_BUFFER_OFFSET = PRECEDING_BITS % LANE_BIT_WIDTH;
  constexpr uint64_t INITIAL_N_INPUT_LINE = PRECEDING_BITS / LANE_BIT_WIDTH;
  constexpr uint64_t END_INDEX = START_INDEX + (UNPACK_N_VALUES * N_LANES);

  for (int lane = 0; lane < N_LANES; lane++) {
    T line_buffer = 0U;
    uint64_t buffer_offset = INITIAL_BUFFER_OFFSET;
    uint64_t n_input_line = INITIAL_N_INPUT_LINE;
    T buffer_offset_mask;

    line_buffer = *(in + n_input_line * N_LANES + lane);
    n_input_line++;

#pragma clang loop unroll(full)
    for (int i = 0; i < UNPACK_N_VALUES; ++i) {
      T value;

      bool line_buffer_is_empty = buffer_offset == LANE_BIT_WIDTH;
      if (line_buffer_is_empty) {
        line_buffer = *(in + n_input_line * N_LANES + lane);
        ++n_input_line;
        buffer_offset -= LANE_BIT_WIDTH;
      }

      value = (line_buffer & (VALUE_MASK << buffer_offset)) >> buffer_offset;
      buffer_offset += VALUE_BIT_WIDTH;

      bool value_continues_on_next_line = buffer_offset > LANE_BIT_WIDTH;
      if (value_continues_on_next_line) {
        line_buffer = *(in + n_input_line * N_LANES + lane);

        ++n_input_line;
        buffer_offset -= LANE_BIT_WIDTH;

        buffer_offset_mask = ((1 << buffer_offset) - 1);
        value |= (line_buffer & buffer_offset_mask)
                 << (VALUE_BIT_WIDTH - buffer_offset);
      }

      *(out + lane) = lambda(value);

      out += N_LANES;
    }
    out -= N_LANES * UNPACK_N_VALUES;
  }
}

template <typename T, unsigned VALUE_BIT_WIDTH>
void bitpack(const T *__restrict in, T *__restrict out) {
  auto lambda = [](const T value) -> T { return value; };
  constexpr unsigned LANE_BIT_WIDTH = sizeof(T) * 8;
  utils::pack<T, LANE_BIT_WIDTH, VALUE_BIT_WIDTH>(in, out, lambda);
}

template <typename T, unsigned VALUE_BIT_WIDTH>
void bitunpack(const T *__restrict in, T *__restrict out) {
  auto lambda = [](const T value) -> T { return value; };
  constexpr unsigned LANE_BIT_WIDTH = sizeof(T) * 8;
  utils::unpack<T, LANE_BIT_WIDTH, VALUE_BIT_WIDTH, 32, 0>(in, out, lambda);
}

template <typename T, unsigned VALUE_BIT_WIDTH>
void ffor(const T *__restrict in, T *__restrict out,
          const T *__restrict base_p) {
  auto lambda = [base_p](const T value) -> T { return value - *(base_p); };
  constexpr unsigned LANE_BIT_WIDTH = sizeof(T) * 8;
  utils::pack<T, LANE_BIT_WIDTH, VALUE_BIT_WIDTH>(in, out, lambda);
}

template <typename T, unsigned VALUE_BIT_WIDTH>
void unffor(const T *__restrict in, T *__restrict out,
            const T *__restrict base_p) {
  auto lambda = [base_p](const T value) -> T { return value + *(base_p); };
  constexpr unsigned LANE_BIT_WIDTH = sizeof(T) * 8;
  utils::unpack<T, LANE_BIT_WIDTH, VALUE_BIT_WIDTH, 32, 0>(in, out, lambda);
}

} // namespace utils

namespace scalar {

template <typename T>
void bitpack(const T *__restrict in, T *__restrict out,
             const int32_t value_bit_width) {
  switch (value_bit_width) {
  case 1:
    utils::bitpack<T, 1>(in, out);
    break;
  case 2:
    utils::bitpack<T, 2>(in, out);
    break;
  case 3:
    utils::bitpack<T, 3>(in, out);
    break;
  case 4:
    utils::bitpack<T, 4>(in, out);
    break;
  case 5:
    utils::bitpack<T, 5>(in, out);
    break;
  case 6:
    utils::bitpack<T, 6>(in, out);
    break;
  case 7:
    utils::bitpack<T, 7>(in, out);
    break;
  case 8:
    utils::bitpack<T, 8>(in, out);
    break;
  case 9:
    utils::bitpack<T, 9>(in, out);
    break;
  case 10:
    utils::bitpack<T, 10>(in, out);
    break;
  case 11:
    utils::bitpack<T, 11>(in, out);
    break;
  case 12:
    utils::bitpack<T, 12>(in, out);
    break;
  case 13:
    utils::bitpack<T, 13>(in, out);
    break;
  case 14:
    utils::bitpack<T, 14>(in, out);
    break;
  case 15:
    utils::bitpack<T, 15>(in, out);
    break;
  case 16:
    utils::bitpack<T, 16>(in, out);
    break;
  case 17:
    utils::bitpack<T, 17>(in, out);
    break;
  case 18:
    utils::bitpack<T, 18>(in, out);
    break;
  case 19:
    utils::bitpack<T, 19>(in, out);
    break;
  case 20:
    utils::bitpack<T, 20>(in, out);
    break;
  case 21:
    utils::bitpack<T, 21>(in, out);
    break;
  case 22:
    utils::bitpack<T, 22>(in, out);
    break;
  case 23:
    utils::bitpack<T, 23>(in, out);
    break;
  case 24:
    utils::bitpack<T, 24>(in, out);
    break;
  case 25:
    utils::bitpack<T, 25>(in, out);
    break;
  case 26:
    utils::bitpack<T, 26>(in, out);
    break;
  case 27:
    utils::bitpack<T, 27>(in, out);
    break;
  case 28:
    utils::bitpack<T, 28>(in, out);
    break;
  case 29:
    utils::bitpack<T, 29>(in, out);
    break;
  case 30:
    utils::bitpack<T, 30>(in, out);
    break;
  case 31:
    utils::bitpack<T, 31>(in, out);
    break;
  case 32:
    utils::bitpack<T, 32>(in, out);
    break;
  case 33:
    utils::bitpack<T, 33>(in, out);
    break;
  case 34:
    utils::bitpack<T, 34>(in, out);
    break;
  case 35:
    utils::bitpack<T, 35>(in, out);
    break;
  case 36:
    utils::bitpack<T, 36>(in, out);
    break;
  case 37:
    utils::bitpack<T, 37>(in, out);
    break;
  case 38:
    utils::bitpack<T, 38>(in, out);
    break;
  case 39:
    utils::bitpack<T, 39>(in, out);
    break;
  case 40:
    utils::bitpack<T, 40>(in, out);
    break;
  case 41:
    utils::bitpack<T, 41>(in, out);
    break;
  case 42:
    utils::bitpack<T, 42>(in, out);
    break;
  case 43:
    utils::bitpack<T, 43>(in, out);
    break;
  case 44:
    utils::bitpack<T, 44>(in, out);
    break;
  case 45:
    utils::bitpack<T, 45>(in, out);
    break;
  case 46:
    utils::bitpack<T, 46>(in, out);
    break;
  case 47:
    utils::bitpack<T, 47>(in, out);
    break;
  case 48:
    utils::bitpack<T, 48>(in, out);
    break;
  case 49:
    utils::bitpack<T, 49>(in, out);
    break;
  case 50:
    utils::bitpack<T, 50>(in, out);
    break;
  case 51:
    utils::bitpack<T, 51>(in, out);
    break;
  case 52:
    utils::bitpack<T, 52>(in, out);
    break;
  case 53:
    utils::bitpack<T, 53>(in, out);
    break;
  case 54:
    utils::bitpack<T, 54>(in, out);
    break;
  case 55:
    utils::bitpack<T, 55>(in, out);
    break;
  case 56:
    utils::bitpack<T, 56>(in, out);
    break;
  case 57:
    utils::bitpack<T, 57>(in, out);
    break;
  case 58:
    utils::bitpack<T, 58>(in, out);
    break;
  case 59:
    utils::bitpack<T, 59>(in, out);
    break;
  case 60:
    utils::bitpack<T, 60>(in, out);
    break;
  case 61:
    utils::bitpack<T, 61>(in, out);
    break;
  case 62:
    utils::bitpack<T, 62>(in, out);
    break;
  case 63:
    utils::bitpack<T, 63>(in, out);
    break;
  case 64:
    utils::bitpack<T, 64>(in, out);
    break;
  }
}
template <typename T>
void bitunpack(const T *__restrict in, T *__restrict out,
               const int32_t value_bit_width) {
  switch (value_bit_width) {
  case 1:
    utils::bitunpack<T, 1>(in, out);
    break;
  case 2:
    utils::bitunpack<T, 2>(in, out);
    break;
  case 3:
    utils::bitunpack<T, 3>(in, out);
    break;
  case 4:
    utils::bitunpack<T, 4>(in, out);
    break;
  case 5:
    utils::bitunpack<T, 5>(in, out);
    break;
  case 6:
    utils::bitunpack<T, 6>(in, out);
    break;
  case 7:
    utils::bitunpack<T, 7>(in, out);
    break;
  case 8:
    utils::bitunpack<T, 8>(in, out);
    break;
  case 9:
    utils::bitunpack<T, 9>(in, out);
    break;
  case 10:
    utils::bitunpack<T, 10>(in, out);
    break;
  case 11:
    utils::bitunpack<T, 11>(in, out);
    break;
  case 12:
    utils::bitunpack<T, 12>(in, out);
    break;
  case 13:
    utils::bitunpack<T, 13>(in, out);
    break;
  case 14:
    utils::bitunpack<T, 14>(in, out);
    break;
  case 15:
    utils::bitunpack<T, 15>(in, out);
    break;
  case 16:
    utils::bitunpack<T, 16>(in, out);
    break;
  case 17:
    utils::bitunpack<T, 17>(in, out);
    break;
  case 18:
    utils::bitunpack<T, 18>(in, out);
    break;
  case 19:
    utils::bitunpack<T, 19>(in, out);
    break;
  case 20:
    utils::bitunpack<T, 20>(in, out);
    break;
  case 21:
    utils::bitunpack<T, 21>(in, out);
    break;
  case 22:
    utils::bitunpack<T, 22>(in, out);
    break;
  case 23:
    utils::bitunpack<T, 23>(in, out);
    break;
  case 24:
    utils::bitunpack<T, 24>(in, out);
    break;
  case 25:
    utils::bitunpack<T, 25>(in, out);
    break;
  case 26:
    utils::bitunpack<T, 26>(in, out);
    break;
  case 27:
    utils::bitunpack<T, 27>(in, out);
    break;
  case 28:
    utils::bitunpack<T, 28>(in, out);
    break;
  case 29:
    utils::bitunpack<T, 29>(in, out);
    break;
  case 30:
    utils::bitunpack<T, 30>(in, out);
    break;
  case 31:
    utils::bitunpack<T, 31>(in, out);
    break;
  case 32:
    utils::bitunpack<T, 32>(in, out);
    break;
  case 33:
    utils::bitunpack<T, 33>(in, out);
    break;
  case 34:
    utils::bitunpack<T, 34>(in, out);
    break;
  case 35:
    utils::bitunpack<T, 35>(in, out);
    break;
  case 36:
    utils::bitunpack<T, 36>(in, out);
    break;
  case 37:
    utils::bitunpack<T, 37>(in, out);
    break;
  case 38:
    utils::bitunpack<T, 38>(in, out);
    break;
  case 39:
    utils::bitunpack<T, 39>(in, out);
    break;
  case 40:
    utils::bitunpack<T, 40>(in, out);
    break;
  case 41:
    utils::bitunpack<T, 41>(in, out);
    break;
  case 42:
    utils::bitunpack<T, 42>(in, out);
    break;
  case 43:
    utils::bitunpack<T, 43>(in, out);
    break;
  case 44:
    utils::bitunpack<T, 44>(in, out);
    break;
  case 45:
    utils::bitunpack<T, 45>(in, out);
    break;
  case 46:
    utils::bitunpack<T, 46>(in, out);
    break;
  case 47:
    utils::bitunpack<T, 47>(in, out);
    break;
  case 48:
    utils::bitunpack<T, 48>(in, out);
    break;
  case 49:
    utils::bitunpack<T, 49>(in, out);
    break;
  case 50:
    utils::bitunpack<T, 50>(in, out);
    break;
  case 51:
    utils::bitunpack<T, 51>(in, out);
    break;
  case 52:
    utils::bitunpack<T, 52>(in, out);
    break;
  case 53:
    utils::bitunpack<T, 53>(in, out);
    break;
  case 54:
    utils::bitunpack<T, 54>(in, out);
    break;
  case 55:
    utils::bitunpack<T, 55>(in, out);
    break;
  case 56:
    utils::bitunpack<T, 56>(in, out);
    break;
  case 57:
    utils::bitunpack<T, 57>(in, out);
    break;
  case 58:
    utils::bitunpack<T, 58>(in, out);
    break;
  case 59:
    utils::bitunpack<T, 59>(in, out);
    break;
  case 60:
    utils::bitunpack<T, 60>(in, out);
    break;
  case 61:
    utils::bitunpack<T, 61>(in, out);
    break;
  case 62:
    utils::bitunpack<T, 62>(in, out);
    break;
  case 63:
    utils::bitunpack<T, 63>(in, out);
    break;
  case 64:
    utils::bitunpack<T, 64>(in, out);
    break;
  }
}

template <typename T>
void ffor(const T *__restrict in, T *__restrict out, const T *__restrict base_p,
          const int32_t value_bit_width) {
  switch (value_bit_width) {
  case 1:
    utils::ffor<T, 1>(in, out, base_p);
    break;
  case 2:
    utils::ffor<T, 2>(in, out, base_p);
    break;
  case 3:
    utils::ffor<T, 3>(in, out, base_p);
    break;
  case 4:
    utils::ffor<T, 4>(in, out, base_p);
    break;
  case 5:
    utils::ffor<T, 5>(in, out, base_p);
    break;
  case 6:
    utils::ffor<T, 6>(in, out, base_p);
    break;
  case 7:
    utils::ffor<T, 7>(in, out, base_p);
    break;
  case 8:
    utils::ffor<T, 8>(in, out, base_p);
    break;
  case 9:
    utils::ffor<T, 9>(in, out, base_p);
    break;
  case 10:
    utils::ffor<T, 10>(in, out, base_p);
    break;
  case 11:
    utils::ffor<T, 11>(in, out, base_p);
    break;
  case 12:
    utils::ffor<T, 12>(in, out, base_p);
    break;
  case 13:
    utils::ffor<T, 13>(in, out, base_p);
    break;
  case 14:
    utils::ffor<T, 14>(in, out, base_p);
    break;
  case 15:
    utils::ffor<T, 15>(in, out, base_p);
    break;
  case 16:
    utils::ffor<T, 16>(in, out, base_p);
    break;
  case 17:
    utils::ffor<T, 17>(in, out, base_p);
    break;
  case 18:
    utils::ffor<T, 18>(in, out, base_p);
    break;
  case 19:
    utils::ffor<T, 19>(in, out, base_p);
    break;
  case 20:
    utils::ffor<T, 20>(in, out, base_p);
    break;
  case 21:
    utils::ffor<T, 21>(in, out, base_p);
    break;
  case 22:
    utils::ffor<T, 22>(in, out, base_p);
    break;
  case 23:
    utils::ffor<T, 23>(in, out, base_p);
    break;
  case 24:
    utils::ffor<T, 24>(in, out, base_p);
    break;
  case 25:
    utils::ffor<T, 25>(in, out, base_p);
    break;
  case 26:
    utils::ffor<T, 26>(in, out, base_p);
    break;
  case 27:
    utils::ffor<T, 27>(in, out, base_p);
    break;
  case 28:
    utils::ffor<T, 28>(in, out, base_p);
    break;
  case 29:
    utils::ffor<T, 29>(in, out, base_p);
    break;
  case 30:
    utils::ffor<T, 30>(in, out, base_p);
    break;
  case 31:
    utils::ffor<T, 31>(in, out, base_p);
    break;
  case 32:
    utils::ffor<T, 32>(in, out, base_p);
    break;
  case 33:
    utils::ffor<T, 33>(in, out, base_p);
    break;
  case 34:
    utils::ffor<T, 34>(in, out, base_p);
    break;
  case 35:
    utils::ffor<T, 35>(in, out, base_p);
    break;
  case 36:
    utils::ffor<T, 36>(in, out, base_p);
    break;
  case 37:
    utils::ffor<T, 37>(in, out, base_p);
    break;
  case 38:
    utils::ffor<T, 38>(in, out, base_p);
    break;
  case 39:
    utils::ffor<T, 39>(in, out, base_p);
    break;
  case 40:
    utils::ffor<T, 40>(in, out, base_p);
    break;
  case 41:
    utils::ffor<T, 41>(in, out, base_p);
    break;
  case 42:
    utils::ffor<T, 42>(in, out, base_p);
    break;
  case 43:
    utils::ffor<T, 43>(in, out, base_p);
    break;
  case 44:
    utils::ffor<T, 44>(in, out, base_p);
    break;
  case 45:
    utils::ffor<T, 45>(in, out, base_p);
    break;
  case 46:
    utils::ffor<T, 46>(in, out, base_p);
    break;
  case 47:
    utils::ffor<T, 47>(in, out, base_p);
    break;
  case 48:
    utils::ffor<T, 48>(in, out, base_p);
    break;
  case 49:
    utils::ffor<T, 49>(in, out, base_p);
    break;
  case 50:
    utils::ffor<T, 50>(in, out, base_p);
    break;
  case 51:
    utils::ffor<T, 51>(in, out, base_p);
    break;
  case 52:
    utils::ffor<T, 52>(in, out, base_p);
    break;
  case 53:
    utils::ffor<T, 53>(in, out, base_p);
    break;
  case 54:
    utils::ffor<T, 54>(in, out, base_p);
    break;
  case 55:
    utils::ffor<T, 55>(in, out, base_p);
    break;
  case 56:
    utils::ffor<T, 56>(in, out, base_p);
    break;
  case 57:
    utils::ffor<T, 57>(in, out, base_p);
    break;
  case 58:
    utils::ffor<T, 58>(in, out, base_p);
    break;
  case 59:
    utils::ffor<T, 59>(in, out, base_p);
    break;
  case 60:
    utils::ffor<T, 60>(in, out, base_p);
    break;
  case 61:
    utils::ffor<T, 61>(in, out, base_p);
    break;
  case 62:
    utils::ffor<T, 62>(in, out, base_p);
    break;
  case 63:
    utils::ffor<T, 63>(in, out, base_p);
    break;
  case 64:
    utils::ffor<T, 64>(in, out, base_p);
    break;
  }
}
template <typename T>
void unffor(const T *__restrict in, T *__restrict out,
            const T *__restrict base_p, const int32_t value_bit_width) {
  switch (value_bit_width) {
  case 1:
    utils::unffor<T, 1>(in, out, base_p);
    break;
  case 2:
    utils::unffor<T, 2>(in, out, base_p);
    break;
  case 3:
    utils::unffor<T, 3>(in, out, base_p);
    break;
  case 4:
    utils::unffor<T, 4>(in, out, base_p);
    break;
  case 5:
    utils::unffor<T, 5>(in, out, base_p);
    break;
  case 6:
    utils::unffor<T, 6>(in, out, base_p);
    break;
  case 7:
    utils::unffor<T, 7>(in, out, base_p);
    break;
  case 8:
    utils::unffor<T, 8>(in, out, base_p);
    break;
  case 9:
    utils::unffor<T, 9>(in, out, base_p);
    break;
  case 10:
    utils::unffor<T, 10>(in, out, base_p);
    break;
  case 11:
    utils::unffor<T, 11>(in, out, base_p);
    break;
  case 12:
    utils::unffor<T, 12>(in, out, base_p);
    break;
  case 13:
    utils::unffor<T, 13>(in, out, base_p);
    break;
  case 14:
    utils::unffor<T, 14>(in, out, base_p);
    break;
  case 15:
    utils::unffor<T, 15>(in, out, base_p);
    break;
  case 16:
    utils::unffor<T, 16>(in, out, base_p);
    break;
  case 17:
    utils::unffor<T, 17>(in, out, base_p);
    break;
  case 18:
    utils::unffor<T, 18>(in, out, base_p);
    break;
  case 19:
    utils::unffor<T, 19>(in, out, base_p);
    break;
  case 20:
    utils::unffor<T, 20>(in, out, base_p);
    break;
  case 21:
    utils::unffor<T, 21>(in, out, base_p);
    break;
  case 22:
    utils::unffor<T, 22>(in, out, base_p);
    break;
  case 23:
    utils::unffor<T, 23>(in, out, base_p);
    break;
  case 24:
    utils::unffor<T, 24>(in, out, base_p);
    break;
  case 25:
    utils::unffor<T, 25>(in, out, base_p);
    break;
  case 26:
    utils::unffor<T, 26>(in, out, base_p);
    break;
  case 27:
    utils::unffor<T, 27>(in, out, base_p);
    break;
  case 28:
    utils::unffor<T, 28>(in, out, base_p);
    break;
  case 29:
    utils::unffor<T, 29>(in, out, base_p);
    break;
  case 30:
    utils::unffor<T, 30>(in, out, base_p);
    break;
  case 31:
    utils::unffor<T, 31>(in, out, base_p);
    break;
  case 32:
    utils::unffor<T, 32>(in, out, base_p);
    break;
  case 33:
    utils::unffor<T, 33>(in, out, base_p);
    break;
  case 34:
    utils::unffor<T, 34>(in, out, base_p);
    break;
  case 35:
    utils::unffor<T, 35>(in, out, base_p);
    break;
  case 36:
    utils::unffor<T, 36>(in, out, base_p);
    break;
  case 37:
    utils::unffor<T, 37>(in, out, base_p);
    break;
  case 38:
    utils::unffor<T, 38>(in, out, base_p);
    break;
  case 39:
    utils::unffor<T, 39>(in, out, base_p);
    break;
  case 40:
    utils::unffor<T, 40>(in, out, base_p);
    break;
  case 41:
    utils::unffor<T, 41>(in, out, base_p);
    break;
  case 42:
    utils::unffor<T, 42>(in, out, base_p);
    break;
  case 43:
    utils::unffor<T, 43>(in, out, base_p);
    break;
  case 44:
    utils::unffor<T, 44>(in, out, base_p);
    break;
  case 45:
    utils::unffor<T, 45>(in, out, base_p);
    break;
  case 46:
    utils::unffor<T, 46>(in, out, base_p);
    break;
  case 47:
    utils::unffor<T, 47>(in, out, base_p);
    break;
  case 48:
    utils::unffor<T, 48>(in, out, base_p);
    break;
  case 49:
    utils::unffor<T, 49>(in, out, base_p);
    break;
  case 50:
    utils::unffor<T, 50>(in, out, base_p);
    break;
  case 51:
    utils::unffor<T, 51>(in, out, base_p);
    break;
  case 52:
    utils::unffor<T, 52>(in, out, base_p);
    break;
  case 53:
    utils::unffor<T, 53>(in, out, base_p);
    break;
  case 54:
    utils::unffor<T, 54>(in, out, base_p);
    break;
  case 55:
    utils::unffor<T, 55>(in, out, base_p);
    break;
  case 56:
    utils::unffor<T, 56>(in, out, base_p);
    break;
  case 57:
    utils::unffor<T, 57>(in, out, base_p);
    break;
  case 58:
    utils::unffor<T, 58>(in, out, base_p);
    break;
  case 59:
    utils::unffor<T, 59>(in, out, base_p);
    break;
  case 60:
    utils::unffor<T, 60>(in, out, base_p);
    break;
  case 61:
    utils::unffor<T, 61>(in, out, base_p);
    break;
  case 62:
    utils::unffor<T, 62>(in, out, base_p);
    break;
  case 63:
    utils::unffor<T, 63>(in, out, base_p);
    break;
  case 64:
    utils::unffor<T, 64>(in, out, base_p);
    break;
  }
}
} // namespace scalar

#endif // FASTLANES_H
