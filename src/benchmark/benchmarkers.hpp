#include <cstddef>
#include <cstdint>
#include <cstring>
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
      verification::generate_integer_range<int32_t>(1); //, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_columm<T>(), decompress_column_a,
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
          data::lambda::get_binary_columm<T>(), decompress_column_a,
          decompress_column_b));
}

template <typename T>
verification::VerificationResult<T>
bench_alp_baseline(const size_t a_count,
                   [[maybe_unused]] const std::string dataset_name) {
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
    alp::gpu::bench::decode_complete_alp_vector<T>(out, in);
  };

  auto exception_percentage = verification::generate_integer_range<int32_t>(10);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      exception_percentage, exception_percentage, a_count, 1,
      verification::get_equal_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_binary_alp_datastructure<T>(),
          decompress_column_a, decompress_column_b));
}

} // namespace benchmarkers
