#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <time.h>
#include <type_traits>
#include <vector>

#include "../common/consts.hpp"
#include "../common/utils.hpp"

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
std::function<T()> get_random_floating_point_generator(const T min,
                                                       const T max) {
  std::random_device random_device;
  std::default_random_engine random_engine(random_device());
  std::uniform_real_distribution<T> uniform_dist(min, max);

  return std::bind(uniform_dist, random_engine);
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
void generate_random_data(T *data, const size_t count, const T min,
                          const T max) {
  auto generate_random_number = get_random_number_generator<T>(min, max);

  for (size_t i = 0; i < count; ++i) {
    data[i] = generate_random_number();
  }
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void generate_random_data(T *data, const size_t count, const T min,
                          const T max) {
  auto generate_random_number =
      get_random_floating_point_generator<T>(min, max);

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
std::unique_ptr<T> generate_ffor_column_with_fixed_decimals(
    const size_t count, const int32_t value_bit_width,
    const int32_t exception_percentage, const int32_t decimals) {
  static_assert(std::is_floating_point<T>::value,
                "T should be a floating point type.");
  using INT_T = typename utils::same_width_int<T>::type;

  INT_T max_value = utils::set_first_n_bits<INT_T>(value_bit_width);
  auto int_column = generate_random_column<INT_T>(count, 0, max_value);
  auto column = cast_column<INT_T, T>(std::move(int_column), count);

  INT_T max_base = max_value * 100;
  auto base_generator = get_random_number_generator<INT_T>(-max_base, max_base);

  auto exception_picker = get_random_number_generator<int32_t>(0, 100);
  auto exception_generator = get_random_floating_point_generator<T>(
      std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

  auto column_p = column.get();

  INT_T base = base_generator();

  T decimal_offset = static_cast<T>(std::pow(10.0, -static_cast<T>(decimals)));

  for (size_t i{0}; i < count; ++i) {
    if (i % consts::VALUES_PER_VECTOR == 0) {
      base = base_generator();
    }

    column_p[i] *= decimal_offset;

    if (exception_picker() < exception_percentage) {
      column_p[i] = exception_generator();
    } else {
      column_p[i] += static_cast<T>(base);
    }
  }

  return column;
}

template <typename T>
std::unique_ptr<T> generate_ffor_column_with_real_doubles(const size_t count) {
  auto column = generate_random_column<T>(count, 0, 100000.0);

  auto base_generator = get_random_floating_point_generator<T>(0.0, 1000.0);

  const auto exceptions_per_vector = 30;
  auto exception_picker =
      get_random_number_generator<int32_t>(0, consts::VALUES_PER_VECTOR);
  auto exception_generator = generation::get_random_floating_point_generator(
      std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

  auto column_p = column.get();

  T base = static_cast<T>(base_generator());
  for (size_t i{0}; i < count; ++i) {
    if (i % consts::VALUES_PER_VECTOR == 0) {
      base = static_cast<T>(base_generator());
    }

    if (exception_picker() < exceptions_per_vector) {
      column_p[i] = exception_generator();
    } else {
      column_p[i] += static_cast<T>(base);
    }
  }

  return column;
}

} // namespace generation

namespace lambda {
template <typename T>
using DataGenerationLambda =
    std::function<std::unique_ptr<T>(const size_t, const int32_t)>;

template <typename T>
DataGenerationLambda<T> get_bp_data(const std::string dataset_name) {
  if (dataset_name == "index") {
    return [](size_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
      return generation::generate_index_column<T>(
          count, utils::set_first_n_bits<T>(value_bit_width));
    };
  } else if (dataset_name == "random") {
    return [](size_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
      return generation::generate_random_column<T>(
          count, T{0}, utils::set_first_n_bits<T>(value_bit_width));
    };
  } else {
    throw std::invalid_argument(
        "This data generator only accepts 'index' & 'random'");
  }
}

template <typename T>
DataGenerationLambda<T> get_ffor_data(const std::string dataset_name, T base) {
  auto get_max_value = [](const int32_t value_bit_width, T l_base) -> T {
    return utils::set_first_n_bits<T>(value_bit_width) +
           (value_bit_width == sizeof(T) * 8 ? T{0} : l_base);
  };

  if (dataset_name == "index") {
    return [base, get_max_value](
               const size_t count,
               const int32_t value_bit_width) -> std::unique_ptr<T> {
      return generation::generate_index_column<T>(
          count, get_max_value(value_bit_width, base), base);
    };
  } else if (dataset_name == "random") {
    return [base, get_max_value](
               const size_t count,
               const int32_t value_bit_width) -> std::unique_ptr<T> {
      return generation::generate_random_column<T>(
          count, base, get_max_value(value_bit_width, base));
    };
  } else {
    throw std::invalid_argument(
        "This data generator only accepts 'index' & 'random'");
  }
}

template <typename T>
DataGenerationLambda<T> get_alp_data(const std::string dataset_name) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "T should be float or double");

  if (dataset_name == "random") {
    return [](size_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
      auto decimals = value_bit_width % 3;
      return generation::generate_ffor_column_with_fixed_decimals<T>(
          count, value_bit_width, 3, decimals);
    };
  } else {
    throw std::invalid_argument(
        "This data generator only accepts 'index' & 'random'");
  }
}

template <typename T>
DataGenerationLambda<T>
get_alprd_data([[maybe_unused]] const std::string dataset_name) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "T should be float or double");

  if (dataset_name == "random") {
    return [](size_t count,
              [[maybe_unused]] int32_t value_bit_width) -> std::unique_ptr<T> {
      return generation::generate_ffor_column_with_real_doubles<T>(count);
    };
  } else {
    throw std::invalid_argument(
        "This data generator only accepts 'index' & 'random'");
  }
}

} // namespace lambda
} // namespace data

#endif // DATAGENERATION_H
