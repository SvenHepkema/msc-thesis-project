#include <cstdint>

#include "verification.hpp"

#include "../alp/alp-bindings.hpp"
#include "../fls/compression.hpp"
#include "../gpu-alp/alp-test-kernels-bindings.hpp"
#include "../gpu-fls/fls-test-kernels-bindings.hpp"

#ifndef COMPRESSORS_H
#define COMPRESSORS_H

template <typename T, typename CompressedT, typename CompressionParamsType>
using CompressVectorFunction =
    std::function<void(const T *, CompressedT *, CompressionParamsType)>;

template <typename T>
void apply_fls_compression_to_column(
    const T *in, T *&out, const int32_t value_bit_width, const size_t count,
    CompressVectorFunction<T, T, int32_t> lambda) {

  size_t n_vecs = (count / consts::VALUES_PER_VECTOR);
  size_t compressed_vector_size = static_cast<size_t>(
      utils::get_compressed_vector_size<T>(value_bit_width));
  T *compressed = new T[compressed_vector_size * n_vecs];
  out = compressed;
  for (size_t i = 0; i < n_vecs; ++i) {
    lambda(in, compressed, value_bit_width);
    in += consts::VALUES_PER_VECTOR;
    compressed += compressed_vector_size;
  }
}

template <typename T> struct BP_FLSCompressorFn {
  void operator()(const T *a_in, T *&a_out, const int32_t a_value_bit_width,
                  const size_t a_count) {
    apply_fls_compression_to_column<T>(
        a_in, a_out, a_value_bit_width, a_count,
        [](const T *in, T *out, const int32_t value_bit_width) -> void {
          fls::pack(in, out, static_cast<uint8_t>(value_bit_width));
        });
  }
};

template <typename T> struct FFOR_FLSCompressorFn {
  T base;

  FFOR_FLSCompressorFn(T a_base) : base(a_base) {}

  void operator()(const T *a_in, T *&a_out, const int32_t a_value_bit_width,
                  const size_t a_count) {
    T *base_p = &base;
    apply_fls_compression_to_column<T>(
        a_in, a_out, a_value_bit_width, a_count,
        [base_p](const T *in, T *out, const int32_t value_bit_width) -> void {
          fls::ffor(in, out, static_cast<uint8_t>(value_bit_width), base_p);
        });
  }
};

template <typename T> struct ALP_FLSCompressorFn {
  void operator()(const T *in, alp::AlpCompressionData<T> *&out,
                  [[maybe_unused]] const int32_t value_bit_width,
                  const size_t count) {
    out = new alp::AlpCompressionData<T>(count);
    alp::int_encode<T>(in, count, out);
  }
};


#endif // COMPRESSORS_H
