#include <cstdint>
#include "consts.h"

#ifndef FASTLANES_UTILS_H
#define FASTLANES_UTILS_H

namespace utils { // internal functions

template <typename T> constexpr int32_t sizeof_in_bits() {
  return sizeof(T) * 8;
}

template <typename T> constexpr T set_first_n_bits(const int32_t count) {
  return (count < sizeof_in_bits<T>() ? static_cast<T>((int64_t{1} << int64_t{count}) - int64_t{1})
                                      : static_cast<T>(~T{0}));
}

template <typename T> constexpr int32_t get_lane_bitwidth() {
  return sizeof_in_bits<T>();
}

template <typename T> constexpr int32_t get_n_lanes() {
  return consts::REGISTER_WIDTH / get_lane_bitwidth<T>();
}

template <typename T> constexpr int32_t get_values_per_lane() {
  return consts::VALUES_PER_VECTOR / get_n_lanes<T>();
}

template <typename T>
constexpr int32_t get_compressed_vector_size(int32_t value_bit_width) {
  return (consts::VALUES_PER_VECTOR * value_bit_width) / sizeof_in_bits<T>();
}

} // namespace utils

#endif // FASTLANES_UTILS_H
