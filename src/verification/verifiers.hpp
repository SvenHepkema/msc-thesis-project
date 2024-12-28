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
verify_gpu_bitpacking(const size_t a_count, const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(0, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_bp_data<T>(dataset_name), BP_FLSCompressorFn<T>(),
          BP_GPUStatelessDecompressorFn<T, 1>()));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_bitpacking_multivec(const size_t a_count,
                               const std::string dataset_name) {
  auto value_bit_widths =
      verification::generate_integer_range<int32_t>(0, sizeof(T) * 8);

  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_equal_decompression_verifier<T, T, int32_t, int32_t>(
          data::lambda::get_bp_data<T>(dataset_name),
          BP_FLSDecompressorFn<T>(), BP_GPUStatelessDecompressorFn<T, 4>()));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_bitpacking_with_state(const size_t a_count,
                                 const std::string dataset_name) {
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
          FFOR_FLSCompressorFn<T>(base), FFOR_GPUStatelessDecompressorFn<T>(base)));
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
verify_gpu_alp_into_vec(const size_t a_count, const std::string dataset_name) {
  auto compress_column = [](const T *in, alp::AlpCompressionData<T> *&out,
                            [[maybe_unused]] const int32_t value_bit_width,
                            const size_t count) -> void {
    out = new alp::AlpCompressionData<T>(count);
    alp::int_encode<T>(in, count, out);
  };

  auto decompress_column = [](const alp::AlpCompressionData<T> *in, T *out,
                              [[maybe_unused]] const int32_t value_bit_width,
                              [[maybe_unused]] const size_t count) -> void {
    alp::gpu::test::decode_complete_alp_vector<T>(out, in);
  };

  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  auto value_bit_widths = verification::generate_integer_range<int32_t>(
      0, max_value_bitwidth_to_test);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_data<T>(dataset_name), compress_column,
          decompress_column));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alp_into_lane(const size_t a_count, const std::string dataset_name) {
  auto compress_column = [](const T *in, alp::AlpCompressionData<T> *&out,
                            [[maybe_unused]] const int32_t value_bit_width,
                            const size_t count) -> void {
    out = new alp::AlpCompressionData<T>(count);
    alp::int_encode<T>(in, count, out);
  };

  auto decompress_column = [](const alp::AlpCompressionData<T> *in, T *out,
                              [[maybe_unused]] const int32_t value_bit_width,
                              [[maybe_unused]] const size_t count) -> void {
    alp::gpu::test::decode_alp_vector_into_lane<T>(out, in);
  };

  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  auto value_bit_widths = verification::generate_integer_range<int32_t>(
      0, max_value_bitwidth_to_test);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_data<T>(dataset_name), compress_column,
          decompress_column));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alp_with_state(const size_t a_count,
                          const std::string dataset_name) {
  auto compress_column = [](const T *in, alp::AlpCompressionData<T> *&out,
                            [[maybe_unused]] const int32_t value_bit_width,
                            const size_t count) -> void {
    out = new alp::AlpCompressionData<T>(count);
    alp::int_encode<T>(in, count, out);
  };

  auto decompress_column = [](const alp::AlpCompressionData<T> *in, T *out,
                              [[maybe_unused]] const int32_t value_bit_width,
                              [[maybe_unused]] const size_t count) -> void {
    alp::gpu::test::decode_alp_vector_with_state<T>(out, in);
  };

  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  auto value_bit_widths = verification::generate_integer_range<int32_t>(
      0, max_value_bitwidth_to_test);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_data<T>(dataset_name), compress_column,
          decompress_column));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alp_with_extended_state(const size_t a_count,
                                   const std::string dataset_name) {
  auto compress_column = [](const T *in, alp::AlpCompressionData<T> *&out,
                            [[maybe_unused]] const int32_t value_bit_width,
                            const size_t count) -> void {
    out = new alp::AlpCompressionData<T>(count);
    alp::int_encode<T>(in, count, out);
  };

  auto decompress_column = [](const alp::AlpCompressionData<T> *in, T *out,
                              [[maybe_unused]] const int32_t value_bit_width,
                              [[maybe_unused]] const size_t count) -> void {
    alp::gpu::test::decode_alp_vector_with_extended_state<T>(out, in);
  };

  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  auto value_bit_widths = verification::generate_integer_range<int32_t>(
      0, max_value_bitwidth_to_test);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_data<T>(dataset_name), compress_column,
          decompress_column));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alp_with_extended_state_multivec(const size_t a_count,
                                            const std::string dataset_name) {
  auto decompress_column_cpu =
      [](const alp::AlpCompressionData<T> *in, T *out,
         [[maybe_unused]] const int32_t value_bit_width,
         [[maybe_unused]] const size_t count) -> void {
    alp::int_decode<T>(out, in);
  };

  auto decompress_column_gpu =
      [](const alp::AlpCompressionData<T> *in, T *out,
         [[maybe_unused]] const int32_t value_bit_width,
         [[maybe_unused]] const size_t count) -> void {
    alp::gpu::test::decode_alp_vector_with_extended_state<T, 4>(out, in);
  };

  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  auto value_bit_widths = verification::generate_integer_range<int32_t>(
      0, max_value_bitwidth_to_test);

  auto [data, generator] = data::lambda::get_alp_reusable_datastructure<T>(
      "value_bit_width", a_count);

  auto result =
      verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                               int32_t, int32_t>(
          value_bit_widths, value_bit_widths, a_count, a_count,
          verification::get_equal_decompression_verifier<
              T, alp::AlpCompressionData<T>, int32_t, int32_t>(
              generator, decompress_column_cpu, decompress_column_gpu, false));

  delete data;
  return result;
}

template <typename T>
verification::VerificationResult<T>
verify_alprd(const size_t a_count, const std::string dataset_name) {
  auto compress_column = [](const T *in, alp::AlpRdCompressionData<T> *&out,
                            [[maybe_unused]] const int32_t value_bit_width,
                            const size_t count) -> void {
    out = new alp::AlpRdCompressionData<T>(count);
    alp::rd_encode<T>(in, count, out);
  };

  auto decompress_column = [](const alp::AlpRdCompressionData<T> *in, T *out,
                              [[maybe_unused]] const int32_t value_bit_width,
                              [[maybe_unused]] const size_t count) -> void {
    alp::rd_decode<T>(out, in);
  };

  std::vector<int32_t> parameters({0});

  return verification::run_verifier_on_parameters<
      T, alp::AlpRdCompressionData<T>, int32_t, int32_t>(
      parameters, parameters, a_count, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpRdCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alprd_data<T>(dataset_name), compress_column,
          decompress_column));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alprd(const size_t a_count, const std::string dataset_name) {
  auto compress_column = [](const T *in, alp::AlpRdCompressionData<T> *&out,
                            [[maybe_unused]] const int32_t value_bit_width,
                            const size_t count) -> void {
    out = new alp::AlpRdCompressionData<T>(count);
    alp::rd_encode<T>(in, count, out);
  };

  auto decompress_column = [](const alp::AlpRdCompressionData<T> *in, T *out,
                              [[maybe_unused]] const int32_t value_bit_width,
                              [[maybe_unused]] const size_t count) -> void {
    alp::gpu::test::decode_complete_alprd_vector<T>(out, in);
  };

  std::vector<int32_t> parameters({0});

  return verification::run_verifier_on_parameters<
      T, alp::AlpRdCompressionData<T>, int32_t, int32_t>(
      parameters, parameters, a_count, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpRdCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alprd_data<T>(dataset_name), compress_column,
          decompress_column));
}

} // namespace verifiers
