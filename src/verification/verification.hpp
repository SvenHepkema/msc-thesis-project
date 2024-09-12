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

#include "../cpu-fls/fls.hpp"
#include "../fls/compression.hpp"
#include "../gpu-fls/gpu-bindings-fls.hpp"
#include "datageneration.hpp"

#ifndef VERIFICATION_H
#define VERIFICATION_H

namespace verification {
constexpr int32_t LOG_N_MISTAKES = 5;

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

  template <typename U, std::enable_if_t<std::is_integral<U>::value, bool> = true>
  void log() {
    fprintf(stderr, "[%lu] correct: %lu, unpacked: %lu\n",
            static_cast<uint64_t>(index), static_cast<uint64_t>(original),
            static_cast<uint64_t>(other));
  }

  template <typename U, std::enable_if_t<std::is_floating_point<U>::value, bool> = true>
  void log() {
    fprintf(stderr, "[%lu] correct: %f, unpacked: %f\n", static_cast<uint64_t>(index),
            static_cast<U>(original), static_cast<U>(other));
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
                  const data::lambda::DataGenerationLambda<T> generate_data,
                  const CompressAllVectorsLambda<T> compress,
                  const CompressAllVectorsLambda<T> decompress) {
  auto compressed_column =
      data::generation::allocate_packed_column<T>(count, value_bit_width);
  auto decompressed_column = data::generation::allocate_column<T>(count);
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
    const data::lambda::DataGenerationLambda<T> generate_data,
    const CompressAllVectorsLambda<T> compress,
    const CompressAllVectorsLambda<T> decompress,
    const int32_t max_bit_width = 0) {
  auto value_bit_width_differences =
      std::vector<std::pair<int32_t, Differences<T>>>();
  Differences<T> result;
  int32_t max_value_bit_width =
      max_bit_width == 0 ? sizeof(T) * 8 : max_bit_width;

#ifdef VBW
  {
    int32_t value_bit_width = VBW;
#else
  for (int32_t value_bit_width = 1; value_bit_width <= max_value_bit_width;
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
