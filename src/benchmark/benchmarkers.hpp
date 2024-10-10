#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>

#include "../verification/datageneration.hpp"
#include "../verification/verification.hpp"

#include "../alp/alp-bindings.hpp"
#include "../fls/compression.hpp"
#include "../gpu-alp/alp-test-kernels-bindings.hpp"
//#include "../gpu-fls/fls-benchmark-kernels-bindings.hpp"

namespace benchmarkers {

	/*
template <typename T>
verification::VerificationResult<T> bench_bp_contains_zero_value_bitwidths(
    const size_t a_count, [[maybe_unused]] const std::string dataset_name) {
  auto decompress_column_a = []([[maybe_unused]] const T *in, T *out,
                                [[maybe_unused]] const int32_t value_bit_width,
                                [[maybe_unused]] const size_t count) -> void {
    out = 1;
  };
  auto decompress_column_b = [](const T *in, T *out,
                                const int32_t value_bit_width,
                                const size_t count) -> void {
    fls::gpu::bench::query_bp_contains_zero(out, in, count, value_bit_width,
                                            utils::get_values_per_lane<T>());
  };

  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(1, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_bp_zero_column<T>(), decompress_column_a,
          decompress_column_b));
}
*/

template <typename T>
verification::VerificationResult<T>
bench_baseline(const size_t a_count, const std::string dataset_name) {
  auto compress_column = [](const T *in, T *&out,
                            [[maybe_unused]] const int32_t value_bit_width,
                            const size_t count) -> void {
    out = new T[count];
    std::memcpy(out, in, sizeof(T) * count);
  };

  auto decompress_column = [](const T *in, T *out,
                              [[maybe_unused]] const int32_t value_bit_width,
                              const size_t count) -> void {
    std::memcpy(out, in, sizeof(T) * count);
    // gpu::bench_baseline<T>(out, in, count);
  };

  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(sizeof(T) * 8 / 2);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_alp_data<T>(dataset_name), compress_column,
          decompress_column));
}

template <typename T>
verification::VerificationResult<T>
bench_gpu_alp(const size_t a_count, const std::string dataset_name) {
  auto decompress_column_a =
      [](const alp::AlpCompressionData<T> *in, T *out,
         [[maybe_unused]] const int32_t exception_percentage,
         [[maybe_unused]] const size_t count) -> void {
    /*
    T* data = new T[in->size];
alp::int_decode<T>(data, in);

    size_t counter{0};
    for (int32_t i{0}; i < in->size; ++i) {
            counter += value == data[i];
    }

    delete[] data;
    *out = counter;
    */
    alp::int_decode<T>(out, in);
  };

  auto decompress_column_b =
      [](const alp::AlpCompressionData<T> *in, T *out,
         [[maybe_unused]] const int32_t exception_percentage,
         [[maybe_unused]] const size_t count) -> void {
    // gpu::bench_alp_equal_to<T>(out, in, value);
    alp::gpu::test::decode_complete_alp_vector(out, in);
  };

  auto exception_percentages =
      verification::generate_integer_range<int32_t>(0, 70);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      exception_percentages, exception_percentages, a_count, a_count,
      verification::get_equal_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_datastructure<T>(dataset_name),
          decompress_column_a, decompress_column_b));
}

} // namespace benchmarkers
