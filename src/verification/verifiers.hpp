#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "datageneration.hpp"
#include "verification.hpp"

#include "../alp/alp-bindings.hpp"
#include "../fls/compression.hpp"
#include "../gpu-alp/alp-test-kernels-bindings.hpp"
#include "../gpu-fls/fls-test-kernels-bindings.hpp"

#include "./compressors.h"
#include "./decompressors.h"

namespace verifiers {

template <typename T>
verification::VerificationResult<T>
verify_bitpacking(const size_t a_count, const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(0, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_bp_data<T>(dataset_name), BP_FLSCompressorFn<T>(),
          BP_FLSDecompressorFn<T>()));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_bp_stateless(const size_t a_count, const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(0, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_bp_data<T>(dataset_name), BP_FLSCompressorFn<T>(),
          BP_GPUStatelessDecompressorFn<T, 1>()));
}

template <typename T, unsigned N_VECTORS_AT_A_TIME>
verification::VerificationResult<T>
verify_gpu_bp_stateful_multivec(const size_t a_count,
                                const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(0, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_bp_data<T>(dataset_name), BP_FLSDecompressorFn<T>(),
          BP_GPUStatelessDecompressorFn<T, N_VECTORS_AT_A_TIME>()));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_bp_stateful(const size_t a_count, const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(0, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_bp_data<T>(dataset_name), BP_FLSCompressorFn<T>(),
          BP_GPUStatefulDecompressorFn<T>()));
}

template <typename T>
verification::VerificationResult<T>
verify_ffor(const size_t a_count, const std::string dataset_name) {
  T base = 125;
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(0, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_ffor_data<T>(dataset_name, base),
          FFOR_FLSCompressorFn<T>(base), FFOR_FLSDecompressorFn<T>(base)));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_unffor(const size_t a_count, const std::string dataset_name) {
  T base = 125;
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(0, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_ffor_data<T>(dataset_name, base),
          FFOR_FLSCompressorFn<T>(base),
          FFOR_GPUStatelessDecompressorFn<T>(base)));
}

template <typename T>
verification::VerificationResult<T> verify_alp(const size_t a_count,
                                               const std::string dataset_name) {
  // Higher value bit width causes more data variance.
  // ALP does not always choose int32 then as an encoding method
  // So we take reduced max value bit width to ensure it always chooses
  // alp_int
  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  auto value_bit_widths = verification::generate_integer_range<int32_t>(
      0, max_value_bitwidth_to_test);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_data<T>(dataset_name), ALP_FLSCompressorFn<T>(),
          ALP_FLSDecompressorFn<T>()));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alp_stateless(const size_t a_count, const std::string dataset_name) {
  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  auto value_bit_widths = verification::generate_integer_range<int32_t>(
      0, max_value_bitwidth_to_test);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_data<T>(dataset_name), ALP_FLSCompressorFn<T>(),
          ALP_GPUStatelessDecompressorFn<T>()));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alp_stateful(const size_t a_count, const std::string dataset_name) {
  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  auto value_bit_widths = verification::generate_integer_range<int32_t>(
      0, max_value_bitwidth_to_test);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_data<T>(dataset_name), ALP_FLSCompressorFn<T>(),
          ALP_GPUStatefulDecompressorFn<T>()));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alp_stateful_extended(const size_t a_count,
                                 const std::string dataset_name) {
  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  auto value_bit_widths = verification::generate_integer_range<int32_t>(
      0, max_value_bitwidth_to_test);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_data<T>(dataset_name), ALP_FLSCompressorFn<T>(),
          ALP_GPUStatefulExtendedDecompressorFn<T, 1>()));
}

template <typename T, unsigned UNPACK_N_VECTORS_AT_A_TIME>
verification::VerificationResult<T> verify_gpu_alp_stateful_extended_multivec(
    const size_t a_count, [[maybe_unused]] const std::string dataset_name) {
  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  auto value_bit_widths = verification::generate_integer_range<int32_t>(
      0, max_value_bitwidth_to_test);

  auto [data, generator] = data::lambda::get_alp_reusable_datastructure<T>(
      "value_bit_width", a_count);

  auto result = verification::run_verifier_on_parameters<
      T, alp::AlpCompressionData<T>, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_equal_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          generator, ALP_FLSDecompressorFn<T>(),
          ALP_GPUStatefulExtendedDecompressorFn<T,
                                                UNPACK_N_VECTORS_AT_A_TIME>(),
          false));

  delete data;
  return result;
}

} // namespace verifiers
