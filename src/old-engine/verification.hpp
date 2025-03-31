#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <sys/cdefs.h>
#include <time.h>
#include <type_traits>
#include <vector>

#include "../alp/alp-bindings.hpp"
#include "../fls/compression.hpp"

#ifndef VERIFICATION_H
#define VERIFICATION_H

namespace verification {

constexpr size_t LOG_N_MISTAKES = 5;

template <typename T> struct Difference {
  size_t index;
  T original;
  T other;

  template <typename U,
            std::enable_if_t<std::is_integral<U>::value, bool> = true>
  void log() {
    fprintf(stderr, "[%lu] correct: %lu, found: %lu\n",
            static_cast<uint64_t>(index), static_cast<uint64_t>(original),
            static_cast<uint64_t>(other));
  }

  template <typename U,
            std::enable_if_t<std::is_floating_point<U>::value, bool> = true>
  void log() {
    using UINT_T = typename utils::same_width_uint<T>::type;

    UINT_T *original_c = reinterpret_cast<UINT_T *>(&original);
    UINT_T *other_c = reinterpret_cast<UINT_T *>(&other);

    if (sizeof(U) == 8) {
      fprintf(stderr, "[%lu] correct: %f (%016lX), found: %f (%016lX)\n", index,
              static_cast<double>(original), static_cast<uint64_t>(*original_c),
              static_cast<double>(other), static_cast<uint64_t>(*other_c));
    } else {
      fprintf(stderr, "[%lu] correct: %f (%08X), found: %f (%08X)\n", index,
              static_cast<double>(original), static_cast<uint32_t>(*original_c),
              static_cast<double>(other), static_cast<uint32_t>(*other_c));
    }
  }
};

template <typename T> struct ExecutionResult {
  bool success;
  std::vector<Difference<T>> differences;
};

template <typename T>
using VerificationResult = std::vector<ExecutionResult<T>>;

template <typename T, typename DataParamsType>
using DataGenerator = std::function<T *(DataParamsType, size_t)>;

template <typename T, typename CompressedT, typename CompressionParamsType>
using CompressVectorFunction =
    std::function<void(const T *, CompressedT *, CompressionParamsType)>;

template <typename T, typename CompressedT, typename CompressionParamsType>
using CompressColumnFunction = std::function<void(
    const T *, CompressedT *&, CompressionParamsType, size_t count)>;

template <typename CompressedT, typename T, typename CompressionParamsType>
using DecompressVectorFunction =
    std::function<void(const CompressedT *, T *, CompressionParamsType)>;

template <typename CompressedT, typename T, typename CompressionParamsType>
using DecompressColumnFunction = std::function<void(
    const CompressedT *, T *, CompressionParamsType, size_t count)>;

template <typename T, typename CompressionParamsType, typename DataParamsType>
using VerificationFunction = std::function<ExecutionResult<T>(
    CompressionParamsType, DataParamsType, size_t, size_t)>;

template <typename T>
CompressColumnFunction<T, T, int32_t>
apply_fls_compression_to_column(CompressVectorFunction<T, T, int32_t> lambda) {
  return [lambda](const T *in, T *&out, const int32_t value_bit_width,
                  const size_t count) -> void {
    size_t n_vecs = (count / consts::VALUES_PER_VECTOR);
    size_t compressed_vector_size = static_cast<size_t>(
        utils::get_compressed_vector_size<T>(value_bit_width));
    T *compressed = new T[compressed_vector_size * n_vecs];
    out = compressed;
    for (size_t i = 0; i < n_vecs; ++i) {
      lambda(in, compressed, value_bit_width);
      in += consts::VALUES_PER_VECTOR;
      compressed += compressed_vector_size;
    }
  };
}

template <typename T>
DecompressColumnFunction<T, T, int32_t> apply_fls_decompression_to_column(
    DecompressVectorFunction<T, T, int32_t> lambda) {
  return [lambda](const T *in, T *out, const int32_t value_bit_width,
                  const size_t count) -> void {
    size_t n_vecs = (count / consts::VALUES_PER_VECTOR);
    int32_t compressed_vector_size =
        utils::get_compressed_vector_size<T>(value_bit_width);

    for (size_t i{0}; i < n_vecs; ++i) {
      lambda(in, out, value_bit_width);
      in += compressed_vector_size;
      out += consts::VALUES_PER_VECTOR;
    }
  };
}

template <typename T = int32_t>
std::vector<T> generate_integer_range(const T start, const T end = -1) {
  std::vector<T> integers = {start};

  for (T i{start + 1}; i <= end; i++) {
    integers.push_back(i);
  }

  return integers;
}

template <typename T = int32_t>
std::vector<T> generate_doubling_integer_range(const T start, const T end) {
  std::vector<T> integers = {start};

  T number = start * 2;
  while (number <= end) {
    integers.push_back(number);
    number *= 2;
  }

  return integers;
}

template <typename T> bool byte_compare(const T a, const T b) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  const UINT_T *a_c = reinterpret_cast<const UINT_T *>(&a);
  const UINT_T *b_c = reinterpret_cast<const UINT_T *>(&b);

  return (*a_c) == (*b_c);
}

template <typename T>
ExecutionResult<T> compare_data(const T *a, const T *b, const size_t size) {
  auto differences = std::vector<Difference<T>>();

  for (size_t i{0}; i < size; ++i) {
    if (!byte_compare(a[i], b[i])) {
      differences.push_back(Difference<T>{i, a[i], b[i]});

      if (differences.size() > LOG_N_MISTAKES) {
        break;
      }
    }
  }

  return ExecutionResult<T>{differences.size() == 0, differences};
}

template <typename T, typename CompressedDataType,
          typename CompressionParamsType, typename DataParamsType>
VerificationFunction<T, CompressionParamsType, DataParamsType>
get_equal_decompression_verifier(
    DataGenerator<CompressedDataType, DataParamsType> datagenerator,
    const DecompressColumnFunction<CompressedDataType, T, CompressionParamsType>
        column_decompressor_a,
    const DecompressColumnFunction<CompressedDataType, T, CompressionParamsType>
        column_decompressor_b,
    const bool delete_compressed_data = true) {
  return [datagenerator, column_decompressor_a, column_decompressor_b,
          delete_compressed_data](CompressionParamsType compression_parameters,
                                  DataParamsType data_parameters,
                                  size_t input_size,
                                  size_t output_size) -> ExecutionResult<T> {
    const CompressedDataType *compressed_data =
        datagenerator(data_parameters, input_size);

    T *result_a = new T[output_size];
    T *result_b = new T[output_size];

    column_decompressor_a(compressed_data, result_a, compression_parameters,
                          input_size);
    column_decompressor_b(compressed_data, result_b, compression_parameters,
                          input_size);

    auto result = compare_data<T>(result_a, result_b, output_size);

    if (delete_compressed_data) {
      delete compressed_data;
    }
    delete[] result_a;
    delete[] result_b;

    return result;
  };
}

template <typename T, typename CompressedDataType,
          typename CompressionParamsType, typename DataParamsType>
VerificationFunction<T, CompressionParamsType, DataParamsType>
get_compression_and_decompression_verifier(
    DataGenerator<T, DataParamsType> datagenerator,
    const CompressColumnFunction<T, CompressedDataType, CompressionParamsType>
        compress_column,
    const DecompressColumnFunction<CompressedDataType, T, CompressionParamsType>
        decompress_column) {
  return [datagenerator, compress_column,
          decompress_column](CompressionParamsType compression_parameters,
                             DataParamsType data_parameters, size_t input_size,
                             size_t output_size) -> ExecutionResult<T> {
    const T *original_data = datagenerator(data_parameters, input_size);
    CompressedDataType *compressed_data = nullptr;
    T *decompressed_data = new T[output_size];

    compress_column(original_data, compressed_data, compression_parameters,
                    input_size);
    decompress_column(compressed_data, decompressed_data,
                      compression_parameters, input_size);

    auto result =
        compare_data<T>(original_data, decompressed_data, output_size);

    delete original_data;
    delete compressed_data;
    delete[] decompressed_data;
    return result;
  };
}

template <typename T, typename CompressedT, typename CompressionParamsType,
          typename DataParamsType>
VerificationResult<T> run_verifier_on_parameters(
    const std::vector<CompressionParamsType> compression_parameters_set,
    const DataParamsType data_parameters, const size_t size,
    const VerificationFunction<T, CompressionParamsType, DataParamsType>
        verifier) {
  auto results = std::vector<ExecutionResult<T>>();

  for (auto parameters : compression_parameters_set) {
    results.push_back(verifier(parameters, data_parameters, size));
  }

  return results;
}

template <typename T, typename CompressedT, typename CompressionParamsType,
          typename DataParamsType>
VerificationResult<T> run_verifier_on_parameters(
    const std::vector<CompressionParamsType> compression_parameters_set,
    const std::vector<DataParamsType> data_parameters, const size_t input_size,
    const size_t output_size,
    const VerificationFunction<T, CompressionParamsType, DataParamsType>
        verifier) {
  auto results = std::vector<ExecutionResult<T>>();

  for (size_t i{0}; i < compression_parameters_set.size(); ++i) {
    results.push_back(verifier(compression_parameters_set[i],
                               data_parameters[i], input_size, output_size));
  }

  return results;
}

} // namespace verification

#endif // VERIFICATION_H
