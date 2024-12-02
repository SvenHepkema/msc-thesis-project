#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../verification/datageneration.hpp"
#include "../verification/verification.hpp"

#include "../alp/alp-bindings.hpp"
#include "../fls/compression.hpp"
#include "../gpu-alp/alp-benchmark-kernels-bindings.hpp"
#include "../gpu-alp/alp-test-kernels-bindings.hpp"
#include "../gpu-fls/fls-benchmark-kernels-bindings.hpp"

namespace benchmarkers {

template <typename T>
verification::VerificationResult<T>
bench_int_baseline(const size_t a_count,
                   [[maybe_unused]] const std::string dataset_name) {
  auto decompress_column_a = [](const T *in, T *out,
                                const int32_t value_bit_width,
                                const size_t count) -> void {
    bool none_magic = 1;
    T *temp = new T[consts::VALUES_PER_VECTOR];
    auto n_vecs = utils::get_n_vecs_from_size(count);

    for (size_t i{0}; i < n_vecs; ++i) {
      for (size_t j{0}; j < consts::VALUES_PER_VECTOR; ++j) {
        temp[j] = in[i * consts::VALUES_PER_VECTOR + j];
      }

      for (size_t j{0}; j < consts::VALUES_PER_VECTOR; ++j) {
        none_magic &= temp[j] != consts::as<T>::MAGIC_NUMBER;
      }
    }

    *out = !none_magic;
    delete[] temp;
  };
  auto decompress_column_b = [](const T *in, T *out,
                                const int32_t value_bit_width,
                                const size_t count) -> void {
    fls::gpu::bench::query_baseline_contains_zero<T>(in, out, count);
  };

  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(1, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(), decompress_column_a,
          decompress_column_b));
}

template <typename T>
verification::VerificationResult<T> bench_old_fls_contains_zero_value_bitwidths(
    const size_t a_count, [[maybe_unused]] const std::string dataset_name) {
  auto decompress_column_a = [](const T *in, T *out,
                                const int32_t value_bit_width,
                                const size_t count) -> void {
    bool none_magic = 1;
    T *temp = new T[1024];
    auto n_vecs = utils::get_n_vecs_from_size(count);
    size_t compressed_vector_size = static_cast<size_t>(
        utils::get_compressed_vector_size<T>(value_bit_width));

    for (size_t i{0}; i < n_vecs; ++i) {
      fls::unpack(in + i * compressed_vector_size, temp,
                  static_cast<uint8_t>(value_bit_width));

      for (size_t j{0}; j < 1024; ++j) {
        none_magic &= temp[j] != consts::as<T>::MAGIC_NUMBER;
      }
    }

    *out = !none_magic;
    delete[] temp;
  };
  auto decompress_column_b = [](const T *in, T *out,
                                const int32_t value_bit_width,
                                const size_t count) -> void {
    if (std::is_same<T, uint32_t>::value) {
      fls::gpu::bench::query_old_fls_contains_zero<uint32_t>(
          reinterpret_cast<const uint32_t *>(in),
          reinterpret_cast<uint32_t *>(out), count, value_bit_width);
    } else {
      throw std::invalid_argument("Invalid type.");
    }
  };

  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(1, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(), decompress_column_a,
          decompress_column_b));
}

template <typename T>
verification::VerificationResult<T> bench_bp_contains_zero_value_bitwidths(
    const size_t a_count, [[maybe_unused]] const std::string dataset_name) {
  auto decompress_column_a = [](const T *in, T *out,
                                const int32_t value_bit_width,
                                const size_t count) -> void {
    bool none_magic = 1;
    T *temp = new T[1024];
    auto n_vecs = utils::get_n_vecs_from_size(count);
    size_t compressed_vector_size = static_cast<size_t>(
        utils::get_compressed_vector_size<T>(value_bit_width));

    for (size_t i{0}; i < n_vecs; ++i) {
      fls::unpack(in + i * compressed_vector_size, temp,
                  static_cast<uint8_t>(value_bit_width));

      for (size_t j{0}; j < 1024; ++j) {
        none_magic &= temp[j] != consts::as<T>::MAGIC_NUMBER;
      }
    }

    *out = !none_magic;
    delete[] temp;
  };
  auto decompress_column_b = [](const T *in, T *out,
                                const int32_t value_bit_width,
                                const size_t count) -> void {
    fls::gpu::bench::query_bp_contains_zero<T>(in, out, count, value_bit_width,
                                               1);
  };

  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(1, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(), decompress_column_a,
          decompress_column_b));
}

template <typename T>
verification::VerificationResult<T>
bench_float_baseline(const size_t a_count,
                     [[maybe_unused]] const std::string dataset_name) {
  auto decompress_column_a = [](const T *in, T *out,
                                [[maybe_unused]] const int32_t value_bit_width,
                                const size_t count) -> void {
    bool none_magic = true;
    for (size_t i{0}; i < count; ++i) {
      none_magic &= in[i] != consts::as<T>::MAGIC_NUMBER;
    }
    *out = static_cast<T>(!none_magic);
  };
  auto decompress_column_b = [](const T *in, T *out,
                                [[maybe_unused]] const int32_t value_bit_width,
                                const size_t count) -> void {
    alp::gpu::bench::decode_baseline<T>(out, in, count);
  };

  auto value_bit_widths = verification::generate_integer_range<int32_t>(-1);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(), decompress_column_a,
          decompress_column_b));
}

template <typename T>
verification::VerificationResult<T> bench_alp_varying_exception_count(
    const size_t a_count, [[maybe_unused]] const std::string dataset_name) {
  auto decompress_column_a = [](const alp::AlpCompressionData<T> *in, T *out,
                                [[maybe_unused]] const int32_t value_bit_width,
                                [[maybe_unused]] const size_t count) -> void {
    T *temp = new T[count];
    alp::int_decode<T>(temp, in);

    bool none_magic = true;
    for (size_t i{0}; i < count; ++i) {
      none_magic &= temp[i] != consts::as<T>::MAGIC_NUMBER;
    }
    *out = static_cast<T>(!none_magic);

    delete[] temp;
  };
  auto decompress_column_b = [](const alp::AlpCompressionData<T> *in, T *out,
                                [[maybe_unused]] const int32_t value_bit_width,
                                [[maybe_unused]] const size_t count) -> void {
    // alp::gpu::bench::decode_complete_alp_vector<T>(out, in);
    alp::gpu::bench::decode_alp_vector_with_state<T>(out, in);
  };

  auto exception_percentages =
      verification::generate_integer_range<int32_t>(0, 70);

  auto [data, generator] = data::lambda::get_alp_reusable_datastructure<T>(
      "exceptions_per_vec", a_count);

  auto result =
      verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                               int32_t, int32_t>(
          exception_percentages, exception_percentages, a_count, 1,
          verification::get_equal_decompression_verifier<
              T, alp::AlpCompressionData<T>, int32_t, int32_t>(
              generator, decompress_column_a, decompress_column_b, false));

  delete data;

  return result;
}

template <typename T>
verification::VerificationResult<T>
bench_alp_multiple_columns(const size_t a_count,
                           [[maybe_unused]] const std::string dataset_name) {
  verification::DecompressColumnFunction<
      std::vector<alp::AlpCompressionData<T> *>, T, int32_t>
      decompress_column_a =
          [](const std::vector<alp::AlpCompressionData<T> *> *in, T *out,
             [[maybe_unused]] const int32_t column_count,
             [[maybe_unused]] const size_t count) -> void {
    std::vector<alp::AlpCompressionData<T> *> input_columns = *in;
    std::vector<T *> temps;

    for (auto column : input_columns) {
      T *temp = new T[count];
      alp::int_decode<T>(temp, column);
      temps.push_back(temp);
    }

    bool none_equal = true;
    for (size_t i{0}; i < count; ++i) {
      for (size_t j{1}; j < temps.size(); j++) {
        none_equal &= temps[j][i] != temps[0][i];
      }
    }
    *out = static_cast<T>(!none_equal);

    for (auto temp : temps) {
      delete[] temp;
    }
  };
  verification::DecompressColumnFunction<
      std::vector<alp::AlpCompressionData<T> *>, T, int32_t>
      decompress_column_b =
          [](const std::vector<alp::AlpCompressionData<T> *> *in, T *out,
             [[maybe_unused]] const int32_t column_count,
             [[maybe_unused]] const size_t count) -> void {
    alp::gpu::bench::decode_multiple_alp_vectors<T>(out, *in);
  };

  std::vector<alp::AlpCompressionData<T> *> columns{
      data::generation::generate_alp_datastructure<T>(a_count),
      data::generation::generate_alp_datastructure<T>(a_count),
      data::generation::generate_alp_datastructure<T>(a_count),
      data::generation::generate_alp_datastructure<T>(a_count),
  };
  auto column_counts =
      verification::generate_integer_range<int32_t>(2, columns.size());

  verification::DataGenerator<std::vector<alp::AlpCompressionData<T> *>,
                              int32_t>
      data_generator =
          [columns](int32_t column_count, [[maybe_unused]] size_t count) {
            return new std::vector<alp::AlpCompressionData<T> *>(
                columns.begin(), columns.begin() + column_count);
          };

  return verification::run_verifier_on_parameters<
      T, std::vector<alp::AlpCompressionData<T> *>, int32_t, int32_t>(
      column_counts, column_counts, a_count, 1,
      verification::get_equal_decompression_verifier<
          T, std::vector<alp::AlpCompressionData<T> *>, int32_t, int32_t>(
          data_generator, decompress_column_a, decompress_column_b));
}

template <typename T>
verification::VerificationResult<T> bench_alp_varying_value_bit_width(
    const size_t a_count, [[maybe_unused]] const std::string dataset_name) {
  auto decompress_column_a = [](const alp::AlpCompressionData<T> *in, T *out,
                                [[maybe_unused]] const int32_t value_bit_width,
                                [[maybe_unused]] const size_t count) -> void {
    T *temp = new T[count];
    alp::int_decode<T>(temp, in);

    bool none_magic = true;
    for (size_t i{0}; i < count; ++i) {
      none_magic &= temp[i] != consts::as<T>::MAGIC_NUMBER;
    }
    *out = static_cast<T>(!none_magic);

    delete[] temp;
  };
  auto decompress_column_b = [](const alp::AlpCompressionData<T> *in, T *out,
                                [[maybe_unused]] const int32_t value_bit_width,
                                [[maybe_unused]] const size_t count) -> void {
    //alp::gpu::bench::decode_complete_alp_vector<T>(out, in);
    alp::gpu::bench::decode_alp_vector_with_state<T>(out, in);
  };

  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(0, sizeof(T) * 4);

  auto [data, generator] = data::lambda::get_alp_reusable_datastructure<T>(
      "value_bit_width", a_count);

  auto result = verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          generator,
          decompress_column_a, decompress_column_b, false));

	delete data;
	return result;
}

} // namespace benchmarkers
