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
#include <vector>

#include "azim/compression.hpp"
#include "cpu/fastlanes.h"
#include "gpu/fastlanes-global.h"

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
  size_t packed_count =
      count * static_cast<size_t>(value_bit_width) / (sizeof(T) * 8);
  return std::unique_ptr<T>(new T[packed_count]);
}

template <typename T>
std::unique_ptr<T> generate_index_column(const size_t count, const T max,
                                         const T offset = 0) {
  auto column = allocate_column<T>(count);
  T *column_p = column.get();

  for (size_t i = 0; i < count; ++i) {
    column_p[i] = offset + (static_cast<T>(i) % max);
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

template <typename T>
std::function<std::unique_ptr<T>(size_t, int32_t)>
generate_bp_data(bool use_random_data) {
  return use_random_data
      ? [](size_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
    return data::generate_random_column<T>(
        count, T{0}, utils::set_first_n_bits<T>(value_bit_width));
  }
  : [](size_t count, int32_t value_bit_width) -> std::unique_ptr<T> {
      return data::generate_index_column<T>(
          count, utils::set_first_n_bits<T>(value_bit_width));
    };
}

} // namespace data

template <typename T>
using CompressAllVectorsLambda =
    std::function<void(const T *, T *, const size_t, const int32_t)>;
template <typename T>
using CompressVectorLambda = std::function<void(const T *, T *, const int32_t)>;

namespace verifyutils {

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
std::function<void(const T *, T *, const size_t, const int32_t)>
apply_compression_to_all(CompressVectorLambda<T> lambda) {
  return [lambda](const T *in, T *out, const size_t count,
                  const int32_t value_bit_width) -> void {
    compress_all_vectors<T>(in, out, count, value_bit_width, lambda);
  };
}

template <typename T>
std::function<void(const T *, T *, const size_t, const int32_t)>
apply_decompression_to_all(CompressVectorLambda<T> lambda) {
  return [lambda](const T *in, T *out, const size_t count,
                  const int32_t value_bit_width) -> void {
    decompress_all_vectors<T>(in, out, count, value_bit_width, lambda);
  };
}

} // namespace verifyutils

template <typename T> struct Difference {
  size_t index;
  T original;
  T other;

  void log() {
    fprintf(stderr, "[%ld] %ld != %ld\n", static_cast<int64_t>(index),
            static_cast<int64_t>(original), static_cast<int64_t>(other));
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
    const std::function<std::unique_ptr<T>(size_t, int32_t)> generate_data,
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
VerificationResult<T> verify_all_value_bit_widths(
    const size_t count,
    const std::function<std::unique_ptr<T>(size_t, int32_t)> generate_data,
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

template <typename T>
VerificationResult<T> verify_bitpacking(const size_t a_count,
                                        bool use_random_data) {
  auto compress = [](const T *in, T *out,
                     const int32_t value_bit_width) -> void {
    cpu::bitpack<T>(in, out, value_bit_width);
  };

  auto decompress = [](const T *in, T *out,
                       const int32_t value_bit_width) -> void {
    cpu::bitunpack<T>(in, out, value_bit_width);
  };

  return verify_all_value_bit_widths<T>(
      a_count, data::generate_bp_data<T>(use_random_data),
      verifyutils::apply_compression_to_all<T>(compress),
      verifyutils::apply_decompression_to_all<T>(decompress));
}

template <typename T>
VerificationResult<T> verify_azim_bitpacking(const size_t a_count,
                                             bool use_random_data) {

  auto compress = [](const T *in, T *out,
                     const int32_t value_bit_width) -> void {
    using unsigned_T = typename std::make_unsigned<T>::type;
    azim::pack(reinterpret_cast<const unsigned_T *>(in),
               reinterpret_cast<unsigned_T *>(out),
               static_cast<uint8_t>(value_bit_width));
  };

  auto decompress = [](const T *in, T *out,
                       const int32_t value_bit_width) -> void {
    using unsigned_T = typename std::make_unsigned<T>::type;
    azim::unpack(reinterpret_cast<const unsigned_T *>(in),
                 reinterpret_cast<unsigned_T *>(out),
                 static_cast<uint8_t>(value_bit_width));
  };

  return verify_all_value_bit_widths<T>(
      a_count, data::generate_bp_data<T>(use_random_data),
      verifyutils::apply_compression_to_all<T>(compress),
      verifyutils::apply_decompression_to_all<T>(decompress));
}

template <typename T>
VerificationResult<T> verify_gpu_bitpacking(const size_t a_count,
                                            bool use_random_data) {
  auto generate_data = use_random_data
      ? [](const size_t count,
           const int32_t value_bit_width) -> std::unique_ptr<T> {
    return data::generate_random_column(
        count, T{0}, utils::set_first_n_bits<T>(value_bit_width));
  }
  : [](const size_t count,
       const int32_t value_bit_width) -> std::unique_ptr<T> {
      return data::generate_index_column(
          count, utils::set_first_n_bits<T>(value_bit_width));
    };

  auto compress = [](const T *in, T *out,
                     const int32_t value_bit_width) -> void {
    cpu::bitpack<T>(in, out, value_bit_width);
  };

  auto decompress_all = [](const T *in, T *out, const size_t count,
                           const int32_t value_bit_width) -> void {
    gpu::bitunpack_with_function<T>(in, out, count, value_bit_width);
  };

  return verify_all_value_bit_widths<T>(
      a_count, generate_data, verifyutils::apply_compression_to_all<T>(compress),
      decompress_all);
}

template <typename T>
VerificationResult<T> verify_ffor(const size_t a_count, bool use_random_data) {
  T temp_base = 125;
  T *temp_base_p = &temp_base;

  auto compress = [temp_base_p](const T *in, T *out,
                                const int32_t value_bit_width) -> void {
    cpu::ffor(in, out, temp_base_p, value_bit_width);
  };
  auto decompress = [temp_base_p](const T *in, T *out,
                                  const int32_t value_bit_width) -> void {
    cpu::unffor(in, out, temp_base_p, value_bit_width);
  };

  if (use_random_data) {
    auto generate_data =
        [temp_base](const size_t count,
                    const int32_t value_bit_width) -> std::unique_ptr<T> {
      return data::generate_random_column<T>(
          count, temp_base,
          utils::set_first_n_bits<T>(value_bit_width) +
              (value_bit_width == sizeof(T) * 8 ? T{0} : temp_base));
    };
    return verify_all_value_bit_widths<T>(
        a_count, generate_data, verifyutils::apply_compression_to_all<T>(compress),
        verifyutils::apply_decompression_to_all<T>(decompress));
  } else {
    auto generate_data =
        [temp_base](const size_t count,
                    const int32_t value_bit_width) -> std::unique_ptr<T> {
      return data::generate_index_column<T>(
          count,
          utils::set_first_n_bits<T>(value_bit_width) +
              (value_bit_width == sizeof(T) * 8 ? T{0} : temp_base),
          temp_base);
    };
    return verify_all_value_bit_widths<T>(
        a_count, generate_data, verifyutils::apply_compression_to_all<T>(compress),
        verifyutils::apply_decompression_to_all<T>(decompress));
  }
}

} // namespace verification

#endif // VERIFICATION_H
