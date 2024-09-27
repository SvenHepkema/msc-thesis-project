#include <cstdint>
#include <string>
#include <unordered_map>

#include "verification.hpp"
#include "datageneration.hpp"

#include "../alp/alp-bindings.hpp"
#include "../fls/compression.hpp"
#include "../gpu-alp/alp-test-kernels-bindings.hpp"
#include "../gpu-fls/gpu-bindings-fls.hpp"

namespace verifiers {

template <typename T>
verification::VerificationResult<T>
verify_bitpacking(const size_t a_count, const std::string dataset_name) {
  auto compress_column = verification::apply_fls_compression_to_column<T>(
      [](const T *in, T *out, const int32_t value_bit_width) -> void {
        fls::pack(in, out, static_cast<uint8_t>(value_bit_width));
      });

  auto decompress_column = verification::apply_fls_decompression_to_column<T>(
      [](const T *in, T *out, const int32_t value_bit_width) -> void {
        fls::unpack(in, out, static_cast<uint8_t>(value_bit_width));
      });

  auto value_bit_widths =
      verification::generate_value_bitwidth_parameterset<int32_t>(0, sizeof(T) *
                                                                         8);
  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_bp_data<T>(dataset_name), compress_column,
          decompress_column));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_bitpacking(const size_t a_count, const std::string dataset_name) {
  auto compress_column = verification::apply_fls_compression_to_column<T>(
      [](const T *in, T *out, const int32_t value_bit_width) -> void {
        fls::pack(in, out, static_cast<uint8_t>(value_bit_width));
      });

  auto decompress_column = [](const T *in, T *out,
                              const int32_t value_bit_width,
                              const size_t count) -> void {
    gpu::bitunpack<T>(in, out, count, value_bit_width);
  };

  auto value_bit_widths =
      verification::generate_value_bitwidth_parameterset<int32_t>(0, sizeof(T) *
                                                                         8);
  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_bp_data<T>(dataset_name), compress_column,
          decompress_column));
}

template <typename T>
verification::VerificationResult<T>
verify_ffor(const size_t a_count, const std::string dataset_name) {
  T temp_base = 125;
  T *temp_base_p = &temp_base;

  auto compress_column = verification::apply_fls_compression_to_column<T>(
      [temp_base_p](const T *in, T *out,
                    const int32_t value_bit_width) -> void {
        fls::ffor(in, out, static_cast<uint8_t>(value_bit_width), temp_base_p);
      });

  auto decompress_column = verification::apply_fls_decompression_to_column<T>(
      [temp_base_p](const T *in, T *out,
                    const int32_t value_bit_width) -> void {
        fls::unffor(in, out, static_cast<uint8_t>(value_bit_width),
                    temp_base_p);
      });

  auto value_bit_widths =
      verification::generate_value_bitwidth_parameterset<int32_t>(0, sizeof(T) *
                                                                         8);
  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_ffor_data<T>(dataset_name, temp_base),
          compress_column, decompress_column));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_unffor(const size_t a_count, const std::string dataset_name) {
  T temp_base = 125;
  T *temp_base_p = &temp_base;

  auto compress_column = verification::apply_fls_compression_to_column<T>(
      [temp_base_p](const T *in, T *out,
                    const int32_t value_bit_width) -> void {
        fls::ffor(in, out, static_cast<uint8_t>(value_bit_width), temp_base_p);
      });

  auto decompress_column = [temp_base_p](const T *in, T *out,
                                         const int32_t value_bit_width,
                                         const size_t count) -> void {
    gpu::unffor<T>(in, out, count, value_bit_width, temp_base_p);
  };

  auto value_bit_widths =
      verification::generate_value_bitwidth_parameterset<int32_t>(0, sizeof(T) *
                                                                         8);
  return verification::run_verifier_on_parameters<T, T, int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count,
      verification::get_compression_and_decompression_verifier<T, T, int32_t,
                                                               int32_t>(
          data::lambda::get_ffor_data<T>(dataset_name, temp_base),
          compress_column, decompress_column));
}

template <typename T>
verification::VerificationResult<T> verify_alp(const size_t a_count,
                                               const std::string dataset_name) {
  auto compress_column = [](const T *in, alp::AlpCompressionData<T> *out,
                            [[maybe_unused]] const int32_t value_bit_width,
                            const size_t count) -> void {
    out = new alp::AlpCompressionData<T>(count);
    alp::int_encode<T>(in, count, out);
  };

  auto decompress_column = [](const alp::AlpCompressionData<T> *in, T *out,
                              [[maybe_unused]] const int32_t value_bit_width,
                              [[maybe_unused]] const size_t count) -> void {
    alp::int_decode<T>(out, in);
    delete in;
  };

  // Higher value bit width causes more data variance.
  // ALP does not always choose int32 then as an encoding method
  // So we take reduced max value bit width to ensure it always chooses
  // alp_int
  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  auto value_bit_widths =
      verification::generate_value_bitwidth_parameterset<int32_t>(
          0, max_value_bitwidth_to_test);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_data<T>(dataset_name), compress_column,
          decompress_column));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alp(const size_t a_count, const std::string dataset_name) {
  auto compress_column = [](const T *in, alp::AlpCompressionData<T> *out,
                            [[maybe_unused]] const int32_t value_bit_width,
                            const size_t count) -> void {
    out = new alp::AlpCompressionData<T>(count);
    alp::int_encode<T>(in, count, out);
  };

  auto decompress_column = [](const alp::AlpCompressionData<T> *in, T *out,
                              [[maybe_unused]] const int32_t value_bit_width,
                              [[maybe_unused]] const size_t count) -> void {
    gpu::test_alp_complete_vector_decoding<T>(out, in);
    delete in;
  };

  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  auto value_bit_widths =
      verification::generate_value_bitwidth_parameterset<int32_t>(
          0, max_value_bitwidth_to_test);

  return verification::run_verifier_on_parameters<T, alp::AlpCompressionData<T>,
                                                  int32_t, int32_t>(
      value_bit_widths, value_bit_widths, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alp_data<T>(dataset_name), compress_column,
          decompress_column));
}

template <typename T>
verification::VerificationResult<T>
verify_alprd(const size_t a_count, const std::string dataset_name) {
  auto compress_column = [](const T *in, alp::AlpRdCompressionData<T> *out,
                            [[maybe_unused]] const int32_t value_bit_width,
                            const size_t count) -> void {
    out = new alp::AlpRdCompressionData<T>(count);
    alp::rd_encode<T>(in, count, out);
  };

  auto decompress_column = [](const alp::AlpRdCompressionData<T> *in, T *out,
                              [[maybe_unused]] const int32_t value_bit_width,
                              [[maybe_unused]] const size_t count) -> void {
    alp::rd_decode<T>(out, in);
    delete in;
  };

  std::vector<int32_t> parameters({0});

  return verification::run_verifier_on_parameters<
      T, alp::AlpRdCompressionData<T>, int32_t, int32_t>(
      parameters, parameters, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpRdCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alprd_data<T>(dataset_name), compress_column,
          decompress_column));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alprd(const size_t a_count, const std::string dataset_name) {
  auto compress_column = [](const T *in, alp::AlpRdCompressionData<T> *out,
                            [[maybe_unused]] const int32_t value_bit_width,
                            const size_t count) -> void {
    out = new alp::AlpRdCompressionData<T>(count);
    alp::rd_encode<T>(in, count, out);
  };

  auto decompress_column = [](const alp::AlpRdCompressionData<T> *in, T *out,
                              [[maybe_unused]] const int32_t value_bit_width,
                              [[maybe_unused]] const size_t count) -> void {
    gpu::test_alprd_complete_vector_decoding<T>(out, in);
    delete in;
  };

  std::vector<int32_t> parameters({0});

  return verification::run_verifier_on_parameters<
      T, alp::AlpRdCompressionData<T>, int32_t, int32_t>(
      parameters, parameters, a_count,
      verification::get_compression_and_decompression_verifier<
          T, alp::AlpRdCompressionData<T>, int32_t, int32_t>(
          data::lambda::get_alprd_data<T>(dataset_name), compress_column,
          decompress_column));
}

template <typename T>
using Verifier = std::function<verification::VerificationResult<T>(
    const size_t, const std::string)>;

template <class T> struct Fastlanes {
  static inline const std::unordered_map<std::string, Verifier<T>> verifiers = {
      {"bp",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verify_bitpacking<T>(count, dataset_name);
       }},
      {"gpu_bp",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verify_gpu_bitpacking<T>(count, dataset_name);
       }},
      {"flsffor",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verify_ffor<T>(count, dataset_name);
       }},
      {"gpu_unffor",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verify_gpu_unffor<T>(count, dataset_name);
       }},
  };
};

template <class T> struct Alp {
  static inline std::unordered_map<std::string, Verifier<T>> verifiers = {
      {"alp",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verify_alp<T>(count, dataset_name);
       }},
      {"gpu_alp",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verify_gpu_alp<T>(count, dataset_name);
       }},
      {"alprd",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verify_alprd<T>(count, dataset_name);
       }},
      {"gpu_alprd",
       [](const size_t count, const std::string dataset_name)
           -> verification::VerificationResult<T> {
         return verify_gpu_alprd<T>(count, dataset_name);
       }},
  };
};
} // namespace verifiers
