#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <time.h>
#include <type_traits>
#include <vector>

#include "cpu/fls.hpp"
#include "fls/compression.hpp"
#include "gpu/gpu-bindings-fls.hpp"

#ifndef DATAGENERATION_H
#define DATAGENERATION_H

namespace datageneration {

template <typename T> std::unique_ptr<T> allocate_column(const size_t count) {
  return std::unique_ptr<T>(new T[count]);
}

template <typename T>
std::unique_ptr<T> allocate_packed_column(const size_t count,
                                          const int32_t value_bit_width) {
  const size_t packed_count =
      count * static_cast<size_t>(value_bit_width) / (sizeof(T) * 8);
  return allocate_column<T>(packed_count);
}

template <typename T>
std::unique_ptr<T> generate_index_column(const size_t count, const T max,
                                         const T offset = 0) {
  auto column = allocate_column<T>(count);
  T *column_p = column.get();

  for (size_t i = 0; i < count; ++i) {
    column_p[i] = static_cast<T>(i % (size_t{max} - size_t{offset})) + offset;
  }

  return column;
}

template <typename T>
std::unique_ptr<T> generate_random_column(const size_t count, const T min,
                                          const T max) {
  auto column = allocate_column<T>(count);
  T *column_p = column.get();

  std::random_device random_device;
  std::default_random_engine random_engine(random_device());
  std::uniform_int_distribution<T> uniform_dist(min, max);
  auto generate_random_number = std::bind(uniform_dist, random_engine);

  for (size_t i = 0; i < count; ++i) {
    column_p[i] = generate_random_number();
  }

  return column;
}

template <typename T, typename U>
std::unique_ptr<U> cast_column(const std::unique_ptr<T> column,
                               const size_t count) {
  auto casted_column = allocate_column<U>(count);
  T *column_p = column.get();
  U *casted_column_p = casted_column.get();

  for (size_t i = 0; i < count; ++i) {
    casted_column_p[i] = static_cast<U>(column_p[i]);
  }

  return casted_column;
}

template <typename T>
using DataGenerationLambda =
    std::function<std::unique_ptr<T>(const size_t, const int32_t)>;

template <typename T>
DataGenerationLambda<T> generate_bp_data(const bool use_random_data) {
  if (use_random_data) {
    return [](size_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
      return generate_random_column<T>(
          count, T{0}, utils::set_first_n_bits<T>(value_bit_width));
    };
  } else {
    return [](size_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
      return generate_index_column<T>(
          count, utils::set_first_n_bits<T>(value_bit_width));
    };
  }
}

template <typename T>
DataGenerationLambda<T> generate_ffor_data(const bool use_random_data, T base) {
  auto get_max_value = [](const int32_t value_bit_width, T l_base) -> T {
    return utils::set_first_n_bits<T>(value_bit_width) +
           (value_bit_width == sizeof(T) * 8 ? T{0} : l_base);
  };

  if (use_random_data) {
    return [base, get_max_value](
               const size_t count,
               const int32_t value_bit_width) -> std::unique_ptr<T> {
      return generate_random_column<T>(
          count, base, get_max_value(value_bit_width, base));
    };
  } else {
    return [base, get_max_value](
               const size_t count,
               const int32_t value_bit_width) -> std::unique_ptr<T> {
      return generate_index_column<T>(
          count, get_max_value(value_bit_width, base), base);
    };
  }
}

template <typename T>
DataGenerationLambda<T>
generate_falp_no_exceptions_data([[maybe_unused]] const bool use_random_data) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "T should be float or double");
  using INT_T =
      typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;

  return [](size_t count,
            [[maybe_unused]] int32_t value_bit_width) -> std::unique_ptr<T> {
    return cast_column<INT_T, T>(
        generate_random_column<INT_T>(count, INT_T{0}, INT_T{1}), count);
  };
}

} // namespace datageneration

#endif // DATAGENERATION_H
