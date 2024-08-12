#include "verification.h"

namespace verifiers {

template <typename T>
verification::VerificationResult<T> verify_bitpacking(const size_t a_count,
                                                      bool use_random_data) {
  auto compress = [](const T *in, T *out,
                     const int32_t value_bit_width) -> void {
    cpu::bitpack<T>(in, out, value_bit_width);
  };

  auto decompress = [](const T *in, T *out,
                       const int32_t value_bit_width) -> void {
    cpu::bitunpack<T>(in, out, value_bit_width);
  };

  return verification::verify_all_value_bit_widths<T>(
      a_count, verification::data::generate_bp_data<T>(use_random_data),
      verification::apply_compression_to_all<T>(compress),
      verification::apply_decompression_to_all<T>(decompress));
}

template <typename T>
verification::VerificationResult<T>
verify_azim_bitpacking(const size_t a_count, bool use_random_data) {
  auto compress = [](const T *in, T *out,
                     const int32_t value_bit_width) -> void {
    azim::pack(in, out, static_cast<uint8_t>(value_bit_width));
  };

  auto decompress = [](const T *in, T *out,
                       const int32_t value_bit_width) -> void {
    azim::unpack(in, out, static_cast<uint8_t>(value_bit_width));
  };

  return verification::verify_all_value_bit_widths<T>(
      a_count, verification::data::generate_bp_data<T>(use_random_data),
      verification::apply_compression_to_all<T>(compress),
      verification::apply_decompression_to_all<T>(decompress));
}

template <typename T>
verification::VerificationResult<T>
verify_gpu_bitpacking(const size_t a_count, bool use_random_data) {
  auto compress = [](const T *in, T *out,
                     const int32_t value_bit_width) -> void {
    cpu::bitpack<T>(in, out, value_bit_width);
  };

  auto decompress_all = [](const T *in, T *out, const size_t count,
                           const int32_t value_bit_width) -> void {
    gpu::bitunpack_with_function<T>(in, out, count, value_bit_width);
  };

  return verification::verify_all_value_bit_widths<T>(
      a_count, verification::data::generate_bp_data<T>(use_random_data),
      verification::apply_compression_to_all<T>(compress), decompress_all);
}

template <typename T>
verification::VerificationResult<T> verify_ffor(const size_t a_count,
                                                bool use_random_data) {
  T temp_base = 125;
  T *temp_base_p = &temp_base;

  auto compress = [temp_base_p](const T *in, T *out,
                                const int32_t value_bit_width) -> void {
    cpu::ffor(in, out, temp_base_p, value_bit_width);
  };
  auto decompress = [temp_base_p](const T *in, T *out,
                                  const int32_t value_bit_width) -> void {
    cpu::unffor(in, out, temp_base_p, value_bit_width);
  };

  return verification::verify_all_value_bit_widths<T>(
      a_count,
      verification::data::generate_ffor_data<T>(use_random_data, temp_base),
      verification::apply_compression_to_all<T>(compress),
      verification::apply_decompression_to_all<T>(decompress));
}

template <typename T>
std::function<verification::VerificationResult<T>(const size_t, const bool)>
get_verifier(const std::string name) {
  if (name == "bp") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verification::verifiers::verify_bitpacking<T>(count,
                                                               use_random_data);
        };
  } else if (name == "azim_bp") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verification::verifiers::verify_azim_bitpacking<T>(
              count, use_random_data);
        };
  } else if (name == "gpu_bp") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verification::verifiers::verify_gpu_bitpacking<T>(
              count, use_random_data);
        };
  } else if (name == "ffor") {
    return [](const size_t count, const bool use_random_data)
               -> verification::VerificationResult<T> {
      return verification::verifiers::verify_ffor<T>(count, use_random_data);
    };
  } else {
    throw std::invalid_argument("This compression type is not supported");
  }
}
} // namespace verifiers
