#include <cstdint>

#include "verification.hpp"
#include "../alp/alp-bindings.hpp"
#include "../fls/compression.hpp"
#include "../gpu-alp/alp-test-kernels-bindings.hpp"
#include "../gpu-fls/fls-test-kernels-bindings.hpp"

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

template <typename T> struct BP_FLS_DecompressorFn {
  void operator()(const T *a_in, T *a_out, const int32_t a_value_bit_width,
                  const size_t a_count) {
    apply_fls_decompression_to_column<T>(
        a_in, a_out, a_value_bit_width, a_count,
        [](const T *in, T *out, const int32_t value_bit_width) -> void {
          fls::unpack(in, out, static_cast<uint8_t>(value_bit_width));
        });
  }
};

template <typename T, unsigned N_VECTORS_AT_A_TIME> struct BP_GPU_DecompressorFn {
  void operator()(const T *a_in, T *a_out, const int32_t a_value_bit_width,
                  const size_t a_count) {
    fls::gpu::test::bitunpack<T, N_VECTORS_AT_A_TIME>(a_in, a_out, a_count, a_value_bit_width);
  }
};

#endif // DECOMPRESSOR_H
