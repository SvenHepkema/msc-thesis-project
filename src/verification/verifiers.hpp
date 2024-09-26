#include <cstdint>
#include <string>
#include <unordered_map>

#include "verification.hpp"

#include "../alp/alp-bindings.hpp"
#include "../fls/compression.hpp"
#include "../gpu-alp/alp-test-kernels-bindings.hpp"
#include "../gpu-fls/gpu-bindings-fls.hpp"

namespace verifiers {

template <typename T>
verification::VerificationResult<T>
verify_bitpacking(const size_t a_count, const std::string dataset_name) {
  auto compress = [](const T *in, T *out,
                     const int32_t value_bit_width) -> void {
    fls::pack(in, out, static_cast<uint8_t>(value_bit_width));
  };

  auto decompress = [](const T *in, T *out,
                       const int32_t value_bit_width) -> void {
    fls::unpack(in, out, static_cast<uint8_t>(value_bit_width));
  };

  return verification::verify_all_value_bit_widths<T>(
      a_count, data::lambda::get_bp_data<T>(dataset_name),
      verification::apply_compression_to_all<T>(compress),
      verification::apply_decompression_to_all<T>(decompress));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_bitpacking(const size_t a_count, const std::string dataset_name) {
  auto compress = [](const T *in, T *out,
                     const int32_t value_bit_width) -> void {
    fls::pack(in, out, static_cast<uint8_t>(value_bit_width));
  };

  auto decompress_all = [](const T *in, T *out, const size_t count,
                           const int32_t value_bit_width) -> void {
    gpu::bitunpack<T>(in, out, count, value_bit_width);
  };

  return verification::verify_all_value_bit_widths<T>(
      a_count, data::lambda::get_bp_data<T>(dataset_name),
      verification::apply_compression_to_all<T>(compress), decompress_all);
}

template <typename T>
verification::VerificationResult<T>
verify_ffor(const size_t a_count, const std::string dataset_name) {
  T temp_base = 125;
  T *temp_base_p = &temp_base;

  auto compress = [temp_base_p](const T *in, T *out,
                                const int32_t value_bit_width) -> void {
    fls::ffor(in, out, static_cast<uint8_t>(value_bit_width), temp_base_p);
  };
  auto decompress = [temp_base_p](const T *in, T *out,
                                  const int32_t value_bit_width) -> void {
    fls::unffor(in, out, static_cast<uint8_t>(value_bit_width), temp_base_p);
  };

  return verification::verify_all_value_bit_widths<T>(
      a_count, data::lambda::get_ffor_data<T>(dataset_name, temp_base),
      verification::apply_compression_to_all<T>(compress),
      verification::apply_decompression_to_all<T>(decompress));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_unffor(const size_t a_count, const std::string dataset_name) {
  T temp_base = 125;
  T *temp_base_p = &temp_base;

  auto compress = [temp_base_p](const T *in, T *out,
                                const int32_t value_bit_width) -> void {
    fls::ffor(in, out, static_cast<uint8_t>(value_bit_width), temp_base_p);
  };

  auto decompress_all = [temp_base_p](const T *in, T *out, const size_t count,
                                      const int32_t value_bit_width) -> void {
    gpu::unffor<T>(in, out, count, value_bit_width, temp_base_p);
  };

  return verification::verify_all_value_bit_widths<T>(
      a_count, data::lambda::get_ffor_data<T>(dataset_name, temp_base),
      verification::apply_compression_to_all<T>(compress), decompress_all);
}

template <typename T>
verification::VerificationResult<T> verify_alp(const size_t a_count,
                                               const std::string dataset_name) {
  alp::AlpCompressionData<T> *alp_data_p = nullptr;

  auto compress_all =
      [&alp_data_p](const T *in, [[maybe_unused]] T *out, const size_t count,
                    [[maybe_unused]] const int32_t value_bit_width) -> void {
    alp_data_p = new alp::AlpCompressionData<T>(count);
    alp::int_encode<T>(in, count, alp_data_p);
  };

  auto decompress_all =
      [&alp_data_p]([[maybe_unused]] const T *in, T *out,
                    [[maybe_unused]] const size_t count,
                    [[maybe_unused]] const int32_t value_bit_width) -> void {
    alp::int_decode<T>(out, alp_data_p);
    delete alp_data_p;
  };

  // Higher value bit width causes more data variance.
  // ALP does not always choose int32 then as an encoding method
  // So we take reduced max value bit width to ensure it always chooses
  // alp_int
  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  return verification::verify_all_value_bit_widths<T>(
      a_count, data::lambda::get_alp_data<T>(dataset_name), compress_all,
      decompress_all, max_value_bitwidth_to_test);
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alp(const size_t a_count, const std::string dataset_name) {
  alp::AlpCompressionData<T> *alp_data_p = nullptr;

  auto compress_all =
      [&alp_data_p](const T *in, [[maybe_unused]] T *out, const size_t count,
                    [[maybe_unused]] const int32_t value_bit_width) -> void {
    alp_data_p = new alp::AlpCompressionData<T>(count);
    alp::int_encode<T>(in, count, alp_data_p);
  };

  auto decompress_all =
      [&alp_data_p]([[maybe_unused]] const T *in, T *out,
                    [[maybe_unused]] const size_t count,
                    [[maybe_unused]] const int32_t value_bit_width) -> void {
    gpu::test_alp_complete_vector_decoding<T>(out, alp_data_p);
    delete alp_data_p;
  };

  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  return verification::verify_all_value_bit_widths<T>(
      a_count, data::lambda::get_alp_data<T>(dataset_name), compress_all,
      decompress_all, max_value_bitwidth_to_test);
}

template <typename T>
verification::VerificationResult<T>
verify_alprd(const size_t a_count, const std::string dataset_name) {
  alp::AlpRdCompressionData<T> *alp_data_p = nullptr;

  auto compress_all =
      [&alp_data_p](const T *in, [[maybe_unused]] T *out, const size_t count,
                    [[maybe_unused]] const int32_t value_bit_width) -> void {
    alp_data_p = new alp::AlpRdCompressionData<T>(count);
    alp::rd_encode<T>(in, count, alp_data_p);
  };

  auto decompress_all =
      [&alp_data_p]([[maybe_unused]] const T *in, T *out,
                    [[maybe_unused]] const size_t count,
                    [[maybe_unused]] const int32_t value_bit_width) -> void {
    alp::rd_decode<T>(out, alp_data_p);
    delete alp_data_p;
  };

  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  return verification::verify_all_value_bit_widths<T>(
      a_count, data::lambda::get_alprd_data<T>(dataset_name), compress_all,
      decompress_all, max_value_bitwidth_to_test);
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_alprd(const size_t a_count, const std::string dataset_name) {
  alp::AlpRdCompressionData<T> *alp_data_p = nullptr;

  auto compress_all =
      [&alp_data_p](const T *in, [[maybe_unused]] T *out, const size_t count,
                    [[maybe_unused]] const int32_t value_bit_width) -> void {
    alp_data_p = new alp::AlpRdCompressionData<T>(count);
    alp::rd_encode<T>(in, count, alp_data_p);
  };

  auto decompress_all =
      [&alp_data_p]([[maybe_unused]] const T *in, T *out,
                    [[maybe_unused]] const size_t count,
                    [[maybe_unused]] const int32_t value_bit_width) -> void {
    gpu::test_alprd_complete_vector_decoding<T>(out, alp_data_p);
    delete alp_data_p;
  };

  int32_t max_value_bitwidth_to_test = sizeof(T) == 8 ? 32 : 16;
  return verification::verify_all_value_bit_widths<T>(
      a_count, data::lambda::get_alprd_data<T>(dataset_name), compress_all,
      decompress_all, max_value_bitwidth_to_test);
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
