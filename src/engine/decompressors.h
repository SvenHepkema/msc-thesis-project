#include <cstdint>

#include "../alp/alp-bindings.hpp"
#include "../fls/compression.hpp"
#include "../gpu-kernels/kernels-bindings.hpp"
#include "verification.hpp"

#ifndef DECOMPRESSOR_H
#define DECOMPRESSOR_H

template <typename CompressedT, typename T, typename CompressionParamsType>
using DecompressVectorFunction =
    std::function<void(const CompressedT *, T *, CompressionParamsType)>;

template <typename T>
void apply_fls_decompression_to_column(
    const T *in, T *out, const int32_t value_bit_width, const size_t count,
    DecompressVectorFunction<T, T, int32_t> lambda) {
  size_t n_vecs = (count / consts::VALUES_PER_VECTOR);
  int32_t compressed_vector_size =
      utils::get_compressed_vector_size<T>(value_bit_width);

  for (size_t i{0}; i < n_vecs; ++i) {
    lambda(in, out, value_bit_width);
    in += compressed_vector_size;
    out += consts::VALUES_PER_VECTOR;
  }
}

template <typename T> struct BP_FLSDecompressorFn {
  void operator()(const T *a_in, T *a_out, const int32_t a_value_bit_width,
                  const size_t a_count) {
    apply_fls_decompression_to_column<T>(
        a_in, a_out, a_value_bit_width, a_count,
        [](const T *in, T *out, const int32_t value_bit_width) -> void {
          fls::unpack(in, out, static_cast<uint8_t>(value_bit_width));
        });
  }
};

template <typename T> struct BP_GPUDecompressorFn {
  const runspec::KernelSpecification spec;

  BP_GPUDecompressorFn(const runspec::KernelSpecification a_spec)
      : spec(a_spec) {}

  void operator()(const T *a_in, T *a_out, const int32_t a_value_bit_width,
                  const size_t a_count) {
    kernels::fls::verify_bitunpack(spec, a_in, a_out, a_count,
                                   a_value_bit_width);
  }
};

template <typename T> struct FFOR_FLSDecompressorFn {
  T base;

  FFOR_FLSDecompressorFn(T a_base) : base(a_base) {}

  void operator()(const T *a_in, T *a_out, const int32_t a_value_bit_width,
                  const size_t a_count) {
    T *base_p = &base;
    apply_fls_decompression_to_column<T>(
        a_in, a_out, a_value_bit_width, a_count,
        [base_p](const T *in, T *out, const int32_t value_bit_width) -> void {
          fls::unffor(in, out, static_cast<uint8_t>(value_bit_width), base_p);
        });
  }
};

template <typename T> struct ALP_FLSDecompressorFn {
  void operator()(const alp::AlpCompressionData<T> *in, T *out,
                  [[maybe_unused]] const int32_t value_bit_width,
                  [[maybe_unused]] const size_t count) {
    alp::int_decode<T>(out, in);
  }
};

template <typename T> struct ALP_GPUDecompressorFn {
  const runspec::KernelSpecification spec;

  ALP_GPUDecompressorFn(const runspec::KernelSpecification a_spec)
      : spec(a_spec) {}

  void operator()(const alp::AlpCompressionData<T> *in, T *out,
                  [[maybe_unused]] const int32_t value_bit_width,
                  [[maybe_unused]] const size_t count) {
    kernels::gpualp::verify_decompress_column(spec, out, in);
  }
};

#endif // DECOMPRESSOR_H
