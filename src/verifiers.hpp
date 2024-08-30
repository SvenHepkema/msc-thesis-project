#include "verification.hpp"
#include "alp/alp-bindings.hpp"
#include <cstdint>

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
verify_fastlanes_bitpacking(const size_t a_count, bool use_random_data) {
  auto compress = [](const T *in, T *out,
                     const int32_t value_bit_width) -> void {
    fls::pack(in, out, static_cast<uint8_t>(value_bit_width));
  };

  auto decompress = [](const T *in, T *out,
                       const int32_t value_bit_width) -> void {
    fls::unpack(in, out, static_cast<uint8_t>(value_bit_width));
  };

  return verification::verify_all_value_bit_widths<T>(
      a_count, verification::data::generate_bp_data<T>(use_random_data),
      verification::apply_compression_to_all<T>(compress),
      verification::apply_decompression_to_all<T>(decompress));
}

template <typename T>
verification::VerificationResult<T>
verify_bitpacking_against_fastlanes(const size_t a_count, bool use_random_data) {
  auto compress = [](const T *in, T *out,
                     const int32_t value_bit_width) -> void {
    cpu::bitpack<T>(in, out, static_cast<uint8_t>(value_bit_width));
  };

  auto decompress = [](const T *in, T *out,
                       const int32_t value_bit_width) -> void {
    fls::unpack(in, out, static_cast<uint8_t>(value_bit_width));
  };

  return verification::verify_all_value_bit_widths<T>(
      a_count, verification::data::generate_bp_data<T>(use_random_data),
      verification::apply_compression_to_all<T>(compress),
      verification::apply_decompression_to_all<T>(decompress));
}

template <typename T>
verification::VerificationResult<T>
verify_bitunpacking_against_fastlanes(const size_t a_count, bool use_random_data) {
  auto compress = [](const T *in, T *out,
                     const int32_t value_bit_width) -> void {
    fls::pack(in, out, static_cast<uint8_t>(value_bit_width));
  };

  auto decompress = [](const T *in, T *out,
                       const int32_t value_bit_width) -> void {
    cpu::bitunpack<T>(in, out, static_cast<uint8_t>(value_bit_width));
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
    gpu::bitunpack<T>(in, out, count, value_bit_width);
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
verification::VerificationResult<T> verify_fastlanes_ffor(const size_t a_count,
                                                     bool use_random_data) {
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
      a_count,
      verification::data::generate_ffor_data<T>(use_random_data, temp_base),
      verification::apply_compression_to_all<T>(compress),
      verification::apply_decompression_to_all<T>(decompress));
}

template <typename T>
verification::VerificationResult<T>
verify_ffor_against_fastlanes(const size_t a_count, bool use_random_data) {
  T temp_base = 125;
  T *temp_base_p = &temp_base;

  auto compress = [temp_base_p](const T *in, T *out,
                                const int32_t value_bit_width) -> void {
    cpu::ffor(in, out, temp_base_p, value_bit_width);
  };
  auto decompress = [temp_base_p](const T *in, T *out,
                                  const int32_t value_bit_width) -> void {
    fls::unffor(in, out, static_cast<uint8_t>(value_bit_width), temp_base_p);
  };

  return verification::verify_all_value_bit_widths<T>(
      a_count,
      verification::data::generate_ffor_data<T>(use_random_data, temp_base),
      verification::apply_compression_to_all<T>(compress),
      verification::apply_decompression_to_all<T>(decompress));
}

template <typename T>
verification::VerificationResult<T>
verify_unffor_against_fastlanes(const size_t a_count, bool use_random_data) {
  T temp_base = 125;
  T *temp_base_p = &temp_base;

  auto compress = [temp_base_p](const T *in, T *out,
                                const int32_t value_bit_width) -> void {
    fls::ffor(in, out, static_cast<uint8_t>(value_bit_width), temp_base_p);
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
verification::VerificationResult<T>
verify_gpu_unffor(const size_t a_count, bool use_random_data) {
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
      a_count, verification::data::generate_ffor_data<T>(use_random_data, temp_base),
      verification::apply_compression_to_all<T>(compress), decompress_all);
}

template <typename T>
verification::VerificationResult<T>
verify_falp(const size_t a_count, bool use_random_data) {
	alp::AlpCompressionData* alp_data;
	// TODO: Allocate all the necessary arrays in the alp_data structure

  auto compress_all = [alp_data](const T *in, T *out, const size_t count,
                           const int32_t value_bit_width) -> void {
		alp::int_encode(in, count, alp_data);
  };

  auto decompress_all = [alp_data](const T *in, T *out, const size_t count,
                           const int32_t value_bit_width) -> void {
		alp::int_decode(out, count, alp_data);
  };

	// TODO: Use other kind of verification function
  return verification::verify_all_value_bit_widths<T>(
      a_count, 
			verification::data::generate_falp_no_exceptions_data<T>(use_random_data),
      compress_all, decompress_all);
}

/*
template <typename T>
verification::VerificationResult<T>
verify_falp(const size_t a_count, bool use_random_data) {
	void* alp_data;

  auto compress = [alp_data](const T *in, T *out,
                                const int32_t value_bit_width) -> void {
		alp::int_encode(alp_data);
  };

	// By generating data that has 0 exceptions, the patching step should be 
	// omissable. This is important to be able to test the GPU falp implementation
	// in isolation.
  auto decompress_all = [alp_data](const T *in, T *out, const size_t count,
                           const int32_t value_bit_width) -> void {
		alp::falp();
  };

  return verification::verify_all_value_bit_widths<T>(
      a_count, 
			verification::data::generate_falp_no_exceptions_data<T>(use_random_data),
      verification::apply_compression_to_all<T>(compress), decompress_all);
}
*/

template <typename T>
std::function<verification::VerificationResult<T>(const size_t, const bool)>
get_verifier(const std::string name) {
  if (name == "bp") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verify_bitpacking<T>(count, use_random_data);
        };
  } else if (name == "flsbp") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verify_fastlanes_bitpacking<T>(count, use_random_data);
        };
  } else if (name == "bp_fastlanesubp") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verify_bitpacking_against_fastlanes<T>(count, use_random_data);
        };
  } else if (name == "flsbp_ubp") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verify_bitunpacking_against_fastlanes<T>(count, use_random_data);
        };
  } else if (name == "gpu_bp") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verify_gpu_bitpacking<T>(count, use_random_data);
        };
  } else if (name == "ffor") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verify_ffor<T>(count, use_random_data);
        };
  } else if (name == "flsffor") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verify_fastlanes_ffor<T>(count, use_random_data);
        };
  } else if (name == "ffor_fastlanesunffor") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verify_ffor_against_fastlanes<T>(count, use_random_data);
        };
  } else if (name == "flsffor_unffor") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verify_unffor_against_fastlanes<T>(count, use_random_data);
        };
  } else if (name == "gpu_unffor") {
    return
        [](const size_t count,
           const bool use_random_data) -> verification::VerificationResult<T> {
          return verify_gpu_unffor<T>(count, use_random_data);
        };
  } else {
    throw std::invalid_argument("This compression type is not supported");
  }
}
} // namespace verifiers
