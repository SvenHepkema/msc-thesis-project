#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "fastlanes.h"

#ifndef VERIFICATION_H
#define VERIFICATION_H

namespace verification {
constexpr int32_t LOG_N_MISTAKES = 10;
namespace data {

template <typename T> std::unique_ptr<T> allocate_column(const size_t count) {
  return std::unique_ptr<T>(new T[count]);
}

template <typename T>
std::unique_ptr<T> allocate_packed_column(const size_t count,
                                          const int32_t value_bit_width) {
  size_t packed_count = count * value_bit_width / (sizeof(T) * 8);
  return std::unique_ptr<T>(new T[packed_count]);
}

template <typename T>
std::unique_ptr<T>
generate_column_with(const size_t count,
                     const std::function<T(const T)> lambda) {
  auto column = allocate_column<T>(count);

  for (size_t i = 0; i < count; ++i) {
    column[i] = lambda((T)i);
  }

  return column;
}

template <typename T>
std::unique_ptr<T> generate_index_column(const size_t count, const T max) {
  auto column = allocate_column<T>(count);
  T *column_p = column.get();

  for (size_t i = 0; i < count; ++i) {
    column_p[i] = i % max;
  }

  return column;
  /*
return generate_column_with(
count, [max](const T index) -> T { return index % max; });
                  */
}

template <typename T>
std::unique_ptr<T> generate_random_column(const size_t count, const T min,
                                          const T max) {
  auto column = allocate_column<T>(count);
  T *column_p = column.get();

  for (size_t i = 0; i < count; ++i) {
    column_p[i] = rand() % max;
  }

  return column;
  /*
std::random_device random_device;
std::default_random_engine random_engine(random_device());
std::uniform_int_distribution<T> uniform_dist(min, max);
auto generate_random_number = std::bind(uniform_dist, random_engine);

return generate_column_with(count,
                        [generate_random_number](const T index) -> T {
                          return generate_random_number();
                        });
                                                                                                                  */
}

} // namespace data

template <typename T> struct Difference {
  size_t index;
  T original;
  T other;

  void log() {
    fprintf(stderr, "[%ld] %ld != %ld\n", (int64_t)index, (int64_t)original,
            (int64_t)other);
  }
};

template <typename T> using Differences = std::vector<Difference<T>>;
template <typename T>
using VerificationResult = std::vector<std::pair<int32_t, Differences<T>>>;

template <typename T>
Differences<T> count_differences(const T *__restrict original,
                                 const T *__restrict other, const size_t length,
                                 const size_t max_differences) {
  Differences<T> differences;

  for (size_t i = 0; i < length; ++i) {
    if (original[i] != other[i]) {
      differences.push_back(Difference<T>{i, original[i], other[i]});
      if (differences.size() >= max_differences) {
        return differences;
      }
    }
  }

  return differences;
}

template <typename T>
Differences<T> verify_conversion(
    const size_t count, const int32_t value_bit_width,
    const std::function<std::unique_ptr<T>(int32_t, int32_t)> generate_data,
    const std::function<void(T *, T *, int32_t, int32_t)> compress,
    const std::function<void(T *, T *, int32_t, int32_t)> decompress) {
  int32_t max_bit_width = sizeof(T) * 8;
  auto compressed_column =
      data::allocate_packed_column<T>(count, value_bit_width);
  auto decompressed_column = data::allocate_column<T>(count);
  auto original = generate_data(count, value_bit_width);

  constexpr int LANE_BIT_WIDTH = sizeof(T) * 8;
  size_t compressed_vector_size =
      value_bit_width * consts::VALUES_PER_VECTOR / LANE_BIT_WIDTH;
  T *original_p = original.get(), *compressed_p = compressed_column.get();
  for (size_t i = 0; i < (count / consts::VALUES_PER_VECTOR); ++i) {
    compress(original_p, compressed_p, count, value_bit_width);
    original_p += consts::VALUES_PER_VECTOR;
    compressed_p += compressed_vector_size;
  }

  compressed_p = compressed_column.get();
  T *decompressed_p = decompressed_column.get();
  for (size_t i = 0; i < (count / consts::VALUES_PER_VECTOR); ++i) {
    decompress(compressed_p, decompressed_p, count, value_bit_width);
    compressed_p += compressed_vector_size;
    decompressed_p += consts::VALUES_PER_VECTOR;
  }

  return count_differences(original.get(), decompressed_column.get(), count,
                           LOG_N_MISTAKES);
}

template <typename T>
VerificationResult<T> verify_all_value_bit_widths(
    const size_t count,
    const std::function<std::unique_ptr<T>(int32_t, int32_t)> generate_data,
    const std::function<void(T *, T *, int32_t, int32_t)> compress,
    const std::function<void(T *, T *, int32_t, int32_t)> decompress) {
  auto value_bit_width_differences =
      std::vector<std::pair<int32_t, Differences<T>>>();
  Differences<T> result;
  int32_t max_bit_width = sizeof(T) * 8;

  for (int32_t value_bit_width = 1; value_bit_width <= max_bit_width;
       ++value_bit_width) {
    result = verify_conversion(count, value_bit_width, generate_data, compress,
                               decompress);

    if (result.size() != 0) {
      value_bit_width_differences.push_back(std::pair<int32_t, Differences<T>>(
          value_bit_width, std::move(result)));
    }
  }

  return value_bit_width_differences;
}

template <typename T>
VerificationResult<T> verify_bitpacking(const size_t count,
                                        bool use_random_data) {
  auto generate_data = use_random_data
      ? [](int32_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
    return data::generate_random_column(
        count, (T) 0, utils::set_first_n_bits<T>(value_bit_width));
  }
  : [](int32_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
      return data::generate_index_column(
          count, utils::set_first_n_bits<T>(value_bit_width));
    };

  auto compress = [](T *in, T *out, int32_t count,
                     int32_t value_bit_width) -> void {
    scalar::bitpack(in, out, value_bit_width);
  };
  auto decompress = [](T *in, T *out, int32_t count,
                       int32_t value_bit_width) -> void {
    scalar::bitunpack(in, out, value_bit_width);
  };

  return verify_all_value_bit_widths<T>(count, generate_data, compress,
                                        decompress);
}

template <typename T>
VerificationResult<T> verify_ffor(const size_t count, bool use_random_data) {
  T temp_base = 0;
  T *temp_base_p = &temp_base;

  auto generate_index_data = [](int32_t count,
                                int32_t value_bit_width) -> std::unique_ptr<T> {
    return data::generate_index_column(
        count, utils::set_first_n_bits<T>(value_bit_width));
  };
  auto generate_random_data =[](int32_t count,
                                int32_t value_bit_width) -> std::unique_ptr<T> {
    return data::generate_random_column(
        count, (T) 0, utils::set_first_n_bits<T>(value_bit_width));
  };

  auto compress = [temp_base_p](T *in, T *out, int32_t count,
                                int32_t value_bit_width) -> void {
    scalar::ffor(in, out, temp_base_p, value_bit_width);
  };
  auto decompress = [temp_base_p](T *in, T *out, int32_t count,
                                  int32_t value_bit_width) -> void {
    scalar::unffor(in, out, temp_base_p, value_bit_width);
  };

  return verify_all_value_bit_widths<T>(
      count, use_random_data ? generate_random_data : generate_index_data,
      compress, decompress);
}

} // namespace verification

#endif // VERIFICATION_H
