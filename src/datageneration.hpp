#include <algorithm>
#include <cmath>
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

namespace data {

namespace generation {
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
std::function<T()> get_random_number_generator(const T min, const T max) {
  std::random_device random_device;
  std::default_random_engine random_engine(random_device());
  std::uniform_int_distribution<T> uniform_dist(min, max);

  return std::bind(uniform_dist, random_engine);
}

template <typename T>
void generate_random_data(T *data, const size_t count, const T min,
                          const T max) {
  auto generate_random_number = get_random_number_generator<T>(min, max);

  for (size_t i = 0; i < count; ++i) {
    data[i] = generate_random_number();
  }
}

template <typename T>
std::unique_ptr<T> generate_random_column(const size_t count, const T min,
                                          const T max) {
  auto column = allocate_column<T>(count);

  generate_random_data(column.get(), count, min, max);

  return column;
}

template <typename T>
std::unique_ptr<T> generate_ffor_column_with_different_bases_per_vector(
    const size_t count, const int32_t value_bit_width) {
  T max_value = utils::set_first_n_bits<T>(value_bit_width);
  auto column = generate_random_column<T>(count, 0, max_value);

  //int32_t left_over_bits = int32_t{sizeof(T)} * 8 - value_bit_width;
	//T max_base = utils::set_first_n_bits<T>(left_over_bits);
  T max_base = max_value * 100;
  auto base_generator = get_random_number_generator<T>(0, max_base);

  auto column_p = column.get();

  T base = base_generator();
  for (size_t i{0}; i < count; ++i) {
    if (i % consts::VALUES_PER_VECTOR == 0) {
      base = base_generator();
    }

    column_p[i] += base;
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

} // namespace generation

namespace lambda {
template <typename T>
using DataGenerationLambda =
    std::function<std::unique_ptr<T>(const size_t, const int32_t)>;

template <typename T>
DataGenerationLambda<T> get_bp_data(const bool use_random_data) {
  if (use_random_data) {
    return [](size_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
      return generation::generate_random_column<T>(
          count, T{0}, utils::set_first_n_bits<T>(value_bit_width));
    };
  } else {
    return [](size_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
      return generation::generate_index_column<T>(
          count, utils::set_first_n_bits<T>(value_bit_width));
    };
  }
}

template <typename T>
DataGenerationLambda<T> get_ffor_data(const bool use_random_data, T base) {
  auto get_max_value = [](const int32_t value_bit_width, T l_base) -> T {
    return utils::set_first_n_bits<T>(value_bit_width) +
           (value_bit_width == sizeof(T) * 8 ? T{0} : l_base);
  };

  if (use_random_data) {
    return [base, get_max_value](
               const size_t count,
               const int32_t value_bit_width) -> std::unique_ptr<T> {
      return generation::generate_random_column<T>(
          count, base, get_max_value(value_bit_width, base));
    };
  } else {
    return [base, get_max_value](
               const size_t count,
               const int32_t value_bit_width) -> std::unique_ptr<T> {
      return generation::generate_index_column<T>(
          count, get_max_value(value_bit_width, base), base);
    };
  }
}

template <typename T>
DataGenerationLambda<T>
get_alp_data([[maybe_unused]] const bool use_random_data) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "T should be float or double");
  using INT_T =
      typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;

  return [](size_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
    return generation::cast_column<INT_T, T>(
        generation::generate_ffor_column_with_different_bases_per_vector<INT_T>(
            count, value_bit_width),
        count);
  };
}

} // namespace lambda
} // namespace data

#endif // DATAGENERATION_H
