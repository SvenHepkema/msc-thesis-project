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

#include "../verification/queries.h"

namespace verify_benchmarkers {

template <typename T>
verification::VerificationResult<T>
verify_bench_float_baseline(const size_t a_count,
                            [[maybe_unused]] const std::string dataset_name) {
  auto value_bit_widths = verification::generate_integer_range<int32_t>(-1);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, 1,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_binary_column<T>(),
          FloatBaseline_AnyValueIsMagicQueryFn<T>(),
          FloatBaseline_GPUAnyValueIsMagicQueryFn<T>()));
}

template <typename T>
verification::VerificationResult<T>
verify_bench_alp_stateless(const size_t a_count,
                           [[maybe_unused]] const std::string dataset_name) {
  auto exception_percentages =
      verification::generate_integer_range<int32_t>(0, 10);

  auto result =
      verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                               int32_t, int32_t>(
          exception_percentages, exception_percentages, a_count, 1,
          verification::get_equal_decompression_verifier<
              T, alp::AlpCompressionData<T>, int32_t, int32_t>(
              data::lambda::get_compressed_binary_column<T>(),
              ALP_CPUAnyValueIsMagicQueryFn<T>(),
              ALP_GPUStatelessAnyValueIsMagicQueryFn<T>(), false));

  return result;
}

template <typename T>
verification::VerificationResult<T>
verify_bench_alp_stateful(const size_t a_count,
                          [[maybe_unused]] const std::string dataset_name) {
  auto exception_percentages =
      verification::generate_integer_range<int32_t>(0, 10);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      exception_percentages, exception_percentages, a_count, 1,
      verification::get_equal_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_compressed_binary_column<T>(),
          ALP_CPUAnyValueIsMagicQueryFn<T>(),
          ALP_GPUStatefulAnyValueIsMagicQueryFn<T>(), false));
}

template <typename T>
verification::VerificationResult<T> verify_bench_alp_stateful_extended(
    const size_t a_count, [[maybe_unused]] const std::string dataset_name) {
  auto exception_percentages =
      verification::generate_integer_range<int32_t>(0, 10);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      exception_percentages, exception_percentages, a_count, 1,
      verification::get_equal_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_compressed_binary_column<T>(),
          ALP_CPUAnyValueIsMagicQueryFn<T>(),
          ALP_GPUStatefulExtendedAnyValueIsMagicQueryFn<T>(), false));
}

} // namespace verify_benchmarkers
