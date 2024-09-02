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

#ifndef VERIFICATION_H
#define VERIFICATION_H

namespace verification {
constexpr int32_t LOG_N_MISTAKES = 5;
namespace data {

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
      return data::generate_random_column<T>(
          count, T{0}, utils::set_first_n_bits<T>(value_bit_width));
    };
  } else {
    return [](size_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
      return data::generate_index_column<T>(
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
      return data::generate_random_column<T>(
          count, base, get_max_value(value_bit_width, base));
    };
  } else {
    return [base, get_max_value](
               const size_t count,
               const int32_t value_bit_width) -> std::unique_ptr<T> {
      return data::generate_index_column<T>(
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

  return [](size_t count, [[maybe_unused]] int32_t value_bit_width)
             -> std::unique_ptr<double> {
    return data::cast_column<INT_T, T>(
        data::generate_random_column<INT_T>(count, INT_T{0}, INT_T{1}), count);
  };
}

} // namespace data

template <typename T>
using CompressAllVectorsLambda =
    std::function<void(const T *, T *, const size_t, const int32_t)>;
template <typename T>
using CompressVectorLambda = std::function<void(const T *, T *, const int32_t)>;

template <typename T>
void compress_all_vectors(const T *in, T *out, const size_t count,
                          const int32_t value_bit_width,
                          const CompressVectorLambda<T> compress) {
  size_t n_vecs = (count / consts::VALUES_PER_VECTOR);
  int32_t compressed_vector_size =
      utils::get_compressed_vector_size<T>(value_bit_width);

  for (size_t i = 0; i < n_vecs; ++i) {
    compress(in, out, value_bit_width);
    in += consts::VALUES_PER_VECTOR;
    out += compressed_vector_size;
  }
}

template <typename T>
void decompress_all_vectors(const T *in, T *out, const size_t count,
                            const int32_t value_bit_width,
                            const CompressVectorLambda<T> decompress) {
  size_t n_vecs = (count / consts::VALUES_PER_VECTOR);
  int32_t compressed_vector_size =
      utils::get_compressed_vector_size<T>(value_bit_width);

  for (size_t i = 0; i < n_vecs; ++i) {
    decompress(in, out, value_bit_width);
    in += compressed_vector_size;
    out += consts::VALUES_PER_VECTOR;
  }
}

template <typename T>
CompressAllVectorsLambda<T>
apply_compression_to_all(CompressVectorLambda<T> lambda) {
  return [lambda](const T *in, T *out, const size_t count,
                  const int32_t value_bit_width) -> void {
    compress_all_vectors<T>(in, out, count, value_bit_width, lambda);
  };
}

template <typename T>
CompressAllVectorsLambda<T>
apply_decompression_to_all(CompressVectorLambda<T> lambda) {
  return [lambda](const T *in, T *out, const size_t count,
                  const int32_t value_bit_width) -> void {
    decompress_all_vectors<T>(in, out, count, value_bit_width, lambda);
  };
}

template <typename T> struct Difference {
  size_t index;
  T original;
  T other;

  void log() {
    fprintf(stderr, "[%lu] correct: %lu, unpacked: %lu\n",
            static_cast<uint64_t>(index), static_cast<uint64_t>(original),
            static_cast<uint64_t>(other));
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
Differences<T>
verify_conversion(const size_t count, const int32_t value_bit_width,
                  const data::DataGenerationLambda<T> generate_data,
                  const CompressAllVectorsLambda<T> compress,
                  const CompressAllVectorsLambda<T> decompress) {
  auto compressed_column =
      data::allocate_packed_column<T>(count, value_bit_width);
  auto decompressed_column = data::allocate_column<T>(count);
  auto original = generate_data(count, value_bit_width);

  compress(original.get(), compressed_column.get(), count, value_bit_width);
  decompress(compressed_column.get(), decompressed_column.get(), count,
             value_bit_width);

  return count_differences(original.get(), decompressed_column.get(), count,
                           LOG_N_MISTAKES);
}

template <typename T>
VerificationResult<T>
verify_all_value_bit_widths(const size_t count,
                            const data::DataGenerationLambda<T> generate_data,
                            const CompressAllVectorsLambda<T> compress,
                            const CompressAllVectorsLambda<T> decompress) {
  auto value_bit_width_differences =
      std::vector<std::pair<int32_t, Differences<T>>>();
  Differences<T> result;
  [[maybe_unused]] int32_t max_bit_width = sizeof(T) * 8;

#ifdef VBW
  {
    int32_t value_bit_width = VBW;
#else
  for (int32_t value_bit_width = 1; value_bit_width <= max_bit_width;
       ++value_bit_width) {
#endif
    result = verify_conversion(count, value_bit_width, generate_data, compress,
                               decompress);

    if (result.size() != 0) {
      value_bit_width_differences.push_back(std::pair<int32_t, Differences<T>>(
          value_bit_width, std::move(result)));
    }
  }

  return value_bit_width_differences;
}

} // namespace verification

#endif // VERIFICATION_H
